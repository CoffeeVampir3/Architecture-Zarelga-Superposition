import torch
import torch._inductor.config
import torch.nn as nn
import torch.nn.functional as F

torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
import math
import time
import json
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
from utils.trainutils import AimLogger, count_parameters_layerwise, save_checkpoint
from utils.tokenizer import load_tokenizer
from modeling.model import MoEModel
from modeling.model_config import ModelConfig
from cut_cross_entropy import linear_cross_entropy

from modeling.zRMSNorm import ZeroCenteredRMSNorm

#torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision('high')

class TokenizedJsonlDataset(Dataset):
    def __init__(self, records, max_length, pad_token_id, eos_token_id):
        self.records = records
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, offset = self.records[idx]

        with path.open("rb") as f:
            f.seek(offset)
            record = json.loads(f.readline().decode("utf-8"))

        token_ids = record["token_ids"]
        if token_ids and token_ids[-1] == self.eos_token_id:
            token_ids = token_ids[:self.max_length]
        else:
            # Reserve one slot for EOS so the model learns a stop signal at chunk boundaries.
            token_ids = token_ids[:self.max_length - 1] + [self.eos_token_id]
        return {"token_ids": token_ids}

class PackedTokenizedCollator:
    """Pack records into rows while preserving per-record causal boundaries."""

    def __init__(self, max_length, pad_token_id):
        self.max_length = max_length
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        rows = []
        row_token_ids = []
        row_segment_lengths = []

        def flush_row():
            nonlocal row_token_ids, row_segment_lengths
            if not row_token_ids:
                return
            rows.append((row_token_ids, row_segment_lengths))
            row_token_ids = []
            row_segment_lengths = []

        for example in examples:
            token_ids = example["token_ids"]
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            if not token_ids:
                continue

            if row_token_ids and len(row_token_ids) + len(token_ids) > self.max_length:
                flush_row()

            row_token_ids.extend(token_ids)
            row_segment_lengths.append(len(token_ids))

        flush_row()

        batch_size = len(rows)
        input_ids = torch.full((batch_size, self.max_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, self.max_length), dtype=torch.bool)
        loss_mask = torch.zeros((batch_size, self.max_length), dtype=torch.bool)
        position_ids = torch.zeros((batch_size, self.max_length), dtype=torch.long)

        cu_seqlens = [0]
        unpad_indices = []
        max_seqlen = 0

        for row_idx, (token_ids, segment_lengths) in enumerate(rows):
            row_len = len(token_ids)
            input_ids[row_idx, :row_len] = torch.tensor(token_ids, dtype=torch.long)
            attention_mask[row_idx, :row_len] = True
            loss_mask[row_idx, :row_len] = True

            cursor = 0
            for segment_len in segment_lengths:
                segment_end = cursor + segment_len
                loss_mask[row_idx, cursor] = False
                position_ids[row_idx, cursor:segment_end] = torch.arange(segment_len, dtype=torch.long)
                unpad_indices.extend(range(row_idx * self.max_length + cursor, row_idx * self.max_length + segment_end))
                cu_seqlens.append(cu_seqlens[-1] + segment_len)
                if segment_len > max_seqlen:
                    max_seqlen = segment_len
                cursor = segment_end

        bucketed_max_seqlen = min(((max_seqlen + 31) // 32) * 32, self.max_length)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
            "unpad_indices": torch.tensor(unpad_indices, dtype=torch.long),
            "max_seqlen": bucketed_max_seqlen,
        }

def load_and_preprocess_data(data_dir="outputs", max_length=ModelConfig.sequence_length):
    tokenizer = load_tokenizer()
    data_path = Path(data_dir)

    jsonl_files = sorted(data_path.glob("**/*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(
            f"No tokenized .jsonl files found under {data_path}. "
            "Expected pre-tokenized JSONL records with a `token_ids` field."
        )

    records = []
    for path in tqdm(jsonl_files, desc="Indexing tokenized data"):
        with path.open("rb") as f:
            offset = f.tell()
            line = f.readline()
            while line:
                records.append((path, offset))
                offset = f.tell()
                line = f.readline()

    if not records:
        raise ValueError(f"No tokenized records found under {data_path}.")

    return TokenizedJsonlDataset(records, max_length, tokenizer.pad_token_id, tokenizer.eos_token_id), tokenizer, max_length

# Auxillary loss free routing: https://arxiv.org/abs/2408.15664
# TLDR is that this does a direct update (not via backward) to the expert biases which will tip towards underfilled experts.
def auxillary_loss_free_update(model, all_topk_indices, update_rate, attention_mask=None):
    valid_tokens = attention_mask.bool() if attention_mask is not None else None

    with torch.no_grad():
        for layer_idx, topk_idx in enumerate(all_topk_indices):
            if valid_tokens is not None:
                topk_idx = topk_idx[valid_tokens]

            if topk_idx.numel() == 0:
                continue

            gate = model.layers[layer_idx].mlp.gate
            expert_counts = torch.bincount(topk_idx.flatten(), minlength=gate.n_routed_experts).float()
            errors = expert_counts.mean() - expert_counts
            gate.expert_biases.add_(update_rate * torch.sign(errors))

def build_next_token_loss_inputs(embeddings, input_ids, attention_mask, loss_mask=None, ignore_index=-100):
    source_mask = attention_mask[:, :-1].bool()
    target_mask = attention_mask[:, 1:].bool()
    if loss_mask is not None:
        target_mask = target_mask & loss_mask[:, 1:].bool()

    prediction_mask = source_mask & target_mask
    labels = input_ids[:, 1:].masked_fill(~prediction_mask, ignore_index)
    return embeddings[:, :-1, :].contiguous(), labels.contiguous()

def build_weight_decay_optm(model, learning_rate, weight_decay=0.01, betas=(0.9, 0.95)):
    zero_centered_rmsnorm_params = []
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        module = model.get_submodule('.'.join(name.split('.')[:-1]))

        if isinstance(module, ZeroCenteredRMSNorm):
            zero_centered_rmsnorm_params.append(param)
        elif any(exclude in name for exclude in [
            'bias', 'embedding', 'output_layer', 'norm.weight'
        ]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return torch.optim.AdamW([
        {'params': zero_centered_rmsnorm_params, 'weight_decay': 1e-4},
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=learning_rate, betas=betas, eps=1e-16)

class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, peak_lr, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.warmup_steps = min(max(1, warmup_steps), self.total_steps)
        self.peak_lr = peak_lr
        self.min_lr = peak_lr * min_lr_ratio
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.peak_lr * (self.step_count / self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

def _checkpoint_trainer_state_path(checkpoint_path):
    return checkpoint_path.with_suffix(".trainer.pt")

def _checkpoint_step(checkpoint_path):
    try:
        return int(checkpoint_path.stem.rsplit("_", 1)[1])
    except ValueError:
        return -1

def prune_rolling_checkpoints(checkpoint_dir, max_checkpoints):
    if max_checkpoints is None or max_checkpoints <= 0:
        return

    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_step_*.safetensors"),
        key=_checkpoint_step,
    )
    for checkpoint_path in checkpoints[:-max_checkpoints]:
        checkpoint_path.unlink(missing_ok=True)
        _checkpoint_trainer_state_path(checkpoint_path).unlink(missing_ok=True)

def train(
    model,
    train_dataset,
    tokenizer,
    sequence_length,
    num_epochs=1,
    batch_size=32,
    learning_rate=1e-4,
    update_rate=1e-5,
    checkpoint_interval_steps=10_000,
    max_rolling_checkpoints=5,
):
    device = torch.device("cuda")
    model.to(device)

    weight_decay = 0.01
    optimizer_betas = (0.9, 0.95)
    optimizer = build_weight_decay_optm(model, learning_rate, weight_decay=weight_decay, betas=optimizer_betas)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=PackedTokenizedCollator(sequence_length, tokenizer.pad_token_id),
    )

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    logger = AimLogger(repo='logs/aim', experiment='moe_training', detailed_frequency=20)
    logger.register_moe_layers(model)
    global_step = 0

    total_steps = len(train_loader) * num_epochs
    warmup_steps = min(500, max(1, total_steps // 20))
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        peak_lr=learning_rate,
        min_lr_ratio=0.1,
    )
    logger.log_params({
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'sequence_length': sequence_length,
        'update_rate': update_rate,
        'optimizer': 'AdamW',
        'optimizer_betas': optimizer_betas,
        'optimizer_eps': 1e-16,
        'weight_decay': weight_decay,
        'warmup_steps': warmup_steps,
        'min_lr': learning_rate * 0.1,
        'train_examples': len(train_dataset),
        'checkpoint_interval_steps': checkpoint_interval_steps,
        'max_rolling_checkpoints': max_rolling_checkpoints,
    })

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            position_ids = batch["position_ids"].to(device)
            cu_seqlens = batch["cu_seqlens"].to(device)
            unpad_indices = batch["unpad_indices"].to(device)
            max_seqlen = batch["max_seqlen"]
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                embeddings, all_topk_indices = model.headless_forward(
                    input_ids,
                    position_ids=position_ids,
                    cu_seqlens=cu_seqlens,
                    unpad_indices=unpad_indices,
                    max_seqlen=max_seqlen,
                )
                classifier = model.get_classifier_weights()
                loss_embeddings, labels = build_next_token_loss_inputs(
                    embeddings,
                    input_ids,
                    attention_mask,
                    loss_mask=loss_mask,
                )
                loss = linear_cross_entropy(loss_embeddings, classifier, labels, ignore_index=-100)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            auxillary_loss_free_update(model, all_topk_indices, update_rate, attention_mask)

            metrics = logger.log_training_metrics(loss, optimizer, update_rate, global_step, epoch, batch_idx)
            metrics.update(logger.log_moe_metrics(all_topk_indices, global_step, attention_mask))

            detailed_logging = (global_step % logger.detailed_frequency == 0)
            logger.log(metrics, step=global_step, model=model, detailed_logging=detailed_logging)

            global_step += 1

            if checkpoint_interval_steps > 0 and global_step % checkpoint_interval_steps == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_step_{global_step}.safetensors"
                save_checkpoint(
                    model,
                    optimizer,
                    str(checkpoint_path),
                    scheduler=scheduler,
                    global_step=global_step,
                    epoch=epoch + 1,
                )
                prune_rolling_checkpoints(checkpoint_dir, max_rolling_checkpoints)
                print(f"Checkpoint saved: {checkpoint_path}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        epoch_time = time.time() - epoch_start_time
        epoch_metrics = {
            'loss/epoch_loss': avg_loss,
            'training/epoch_time': epoch_time,
            'training/batches_per_epoch': len(train_loader),
        }
        logger.log(epoch_metrics, step=global_step, detailed_logging=True)

    final_checkpoint_path = checkpoint_dir / "checkpoint_final.safetensors"
    save_checkpoint(
        model,
        optimizer,
        str(final_checkpoint_path),
        scheduler=scheduler,
        global_step=global_step,
        epoch=num_epochs,
    )
    print(f"Final checkpoint saved: {final_checkpoint_path}")

    logger.close()

def main():
    train_dataset, tokenizer, sequence_length = load_and_preprocess_data()
    config = ModelConfig(vocab_size=len(tokenizer))

    model = MoEModel(config)

    count_parameters_layerwise(model)
    model.headless_forward = torch.compile(model.headless_forward, mode="reduce-overhead", dynamic=True)
    train(model, train_dataset, tokenizer, sequence_length)

if __name__ == "__main__":
    main()
