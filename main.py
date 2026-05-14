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
from modeling.mcce_fast_v2 import mcce_raw_token_mean_v2

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

def bucket_max_seqlen(max_seqlen, max_length):
    if max_seqlen <= 0:
        return 0
    return min(((max_seqlen + 31) // 32) * 32, max_length)

def build_token_superposition_batch(batch, superposition_bag_size, latent_sequence_length, pad_token_id):
    """Build a TST batch from the packed raw-token batch.

    Each packed document segment is folded independently so neither source bags
    nor next-bag labels cross causal reset boundaries.
    """
    input_ids = batch["input_ids"]
    cu_seqlens = batch["cu_seqlens"].tolist()
    unpad_indices = batch["unpad_indices"]

    if superposition_bag_size <= 1:
        raise ValueError("superposition_bag_size must be > 1 for TST batches.")

    flat_input_ids = input_ids.flatten()
    segments = []

    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        flat_positions = unpad_indices[start:end]
        segment_len = int(flat_positions.numel())
        full_bags = segment_len // superposition_bag_size
        if full_bags < 2:
            continue

        token_ids = flat_input_ids[flat_positions[: full_bags * superposition_bag_size]]
        source_bags = token_ids.view(full_bags, superposition_bag_size).contiguous()
        if source_bags.size(0) > latent_sequence_length:
            source_bags = source_bags[:latent_sequence_length]

        # Source bag j predicts the raw tokens in source bag j + 1.
        label_count = max(0, source_bags.size(0) - 1)
        if label_count == 0:
            continue

        segments.append((source_bags, source_bags[1:].contiguous()))

    if not segments:
        return None

    # First-Fit Decreasing: sort segments by descending bag count, then place
    # each into the first open row with room. Opens a new row only when no
    # existing row fits.
    segments.sort(key=lambda sl: sl[0].size(0), reverse=True)

    rows = []  # each entry: [list_of_(source_bags, labels), current_len]
    for source_bags, labels in segments:
        seg_len = source_bags.size(0)
        placed = False
        for row in rows:
            if row[1] + seg_len <= latent_sequence_length:
                row[0].append((source_bags, labels))
                row[1] += seg_len
                placed = True
                break
        if not placed:
            rows.append([[(source_bags, labels)], seg_len])

    rows = [row[0] for row in rows]

    batch_size = len(rows)
    input_bags = torch.full(
        (batch_size, latent_sequence_length, superposition_bag_size),
        pad_token_id,
        dtype=input_ids.dtype,
    )
    attention_mask = torch.zeros((batch_size, latent_sequence_length), dtype=torch.bool)
    position_ids = torch.zeros((batch_size, latent_sequence_length), dtype=torch.long)

    packed_labels = []
    loss_indices = []
    cu_bag_seqlens = [0]
    unpad_bag_indices = []
    max_seqlen = 0

    for row_idx, row_segments in enumerate(rows):
        cursor = 0
        for source_bags, labels in row_segments:
            seg_len = source_bags.size(0)
            segment_end = cursor + seg_len

            input_bags[row_idx, cursor:segment_end] = source_bags
            attention_mask[row_idx, cursor:segment_end] = True
            position_ids[row_idx, cursor:segment_end] = torch.arange(seg_len, dtype=torch.long)
            unpad_bag_indices.extend(
                range(row_idx * latent_sequence_length + cursor, row_idx * latent_sequence_length + segment_end)
            )
            cu_bag_seqlens.append(cu_bag_seqlens[-1] + seg_len)
            max_seqlen = max(max_seqlen, seg_len)

            label_count = labels.size(0)
            packed_labels.append(labels)
            loss_indices.extend(
                range(row_idx * latent_sequence_length + cursor, row_idx * latent_sequence_length + cursor + label_count)
            )

            cursor = segment_end

    labels = torch.cat(packed_labels, dim=0).contiguous()
    s = torch.full((labels.size(0),), superposition_bag_size, dtype=torch.int32)

    return {
        "input_ids": input_bags,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "cu_seqlens": torch.tensor(cu_bag_seqlens, dtype=torch.int32),
        "unpad_indices": torch.tensor(unpad_bag_indices, dtype=torch.long),
        "max_seqlen": bucket_max_seqlen(max_seqlen, latent_sequence_length),
        "loss_indices": torch.tensor(loss_indices, dtype=torch.long),
        "labels": labels,
        "s": s,
    }

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
    superposition_bag_size=1,
    superposition_ratio=0.0,
):
    device = torch.device("cuda")
    model.to(device)

    weight_decay = 0.01
    optimizer_betas = (0.9, 0.95)
    optimizer = build_weight_decay_optm(model, learning_rate, weight_decay=weight_decay, betas=optimizer_betas)

    use_superposition = superposition_bag_size > 1 and superposition_ratio > 0.0
    superposition_ratio = min(max(float(superposition_ratio), 0.0), 1.0)
    superposition_raw_length = sequence_length * superposition_bag_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=PackedTokenizedCollator(sequence_length, tokenizer.pad_token_id),
    )
    superposition_loader = None
    if use_superposition:
        superposition_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=PackedTokenizedCollator(superposition_raw_length, tokenizer.pad_token_id),
        )

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    logger = AimLogger(repo='logs/aim', experiment='moe_training', detailed_frequency=20)
    logger.register_moe_layers(model)
    global_step = 0

    total_steps = len(train_loader) * num_epochs
    superposition_steps = int(total_steps * superposition_ratio) if use_superposition else 0
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
        'token_superposition/enabled': use_superposition,
        'token_superposition/bag_size': superposition_bag_size,
        'token_superposition/ratio': superposition_ratio,
        'token_superposition/steps': superposition_steps,
        'token_superposition/raw_sequence_length': superposition_raw_length if use_superposition else None,
    })

    def next_from_loader(loader, iterator):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    def next_superposition_batch(loader, iterator):
        for _ in range(max(1, len(loader))):
            raw_batch, iterator = next_from_loader(loader, iterator)
            tst_batch = build_token_superposition_batch(
                raw_batch,
                superposition_bag_size,
                sequence_length,
                tokenizer.pad_token_id,
            )
            if tst_batch is not None:
                return tst_batch, iterator
        raise RuntimeError(
            "Could not build a TST batch with at least two full token bags per packed segment. "
            "Use a smaller token_superposition_bag_size or longer tokenized chunks."
        )

    model.train()
    total_loss = 0.0
    phase_loss = 0.0
    phase_steps = 0
    train_start_time = time.time()
    standard_iter = iter(train_loader)
    superposition_iter = iter(superposition_loader) if superposition_loader is not None else None
    current_phase = "tst" if superposition_steps > 0 else "recovery"

    for step_idx in tqdm(range(total_steps)):
        in_superposition_phase = step_idx < superposition_steps
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if in_superposition_phase:
            batch, superposition_iter = next_superposition_batch(superposition_loader, superposition_iter)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            position_ids = batch["position_ids"].to(device)
            cu_seqlens = batch["cu_seqlens"].to(device)
            unpad_indices = batch["unpad_indices"].to(device)
            max_seqlen = batch["max_seqlen"]
            loss_indices = batch["loss_indices"].to(device)
            labels = batch["labels"].to(device)
            s = batch["s"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                embeddings, all_topk_indices = model.headless_forward(
                    input_ids,
                    position_ids=position_ids,
                    cu_seqlens=cu_seqlens,
                    unpad_indices=unpad_indices,
                    max_seqlen=max_seqlen,
                )
                classifier = model.get_classifier_weights()
                loss_embeddings = embeddings.reshape(-1, embeddings.size(-1)).index_select(0, loss_indices)
                loss = mcce_raw_token_mean_v2(
                    loss_embeddings,
                    classifier,
                    labels,
                    s,
                    check_label_values=False,
                )
            phase = "tst"
            batch_idx = step_idx
        else:
            batch, standard_iter = next_from_loader(train_loader, standard_iter)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            position_ids = batch["position_ids"].to(device)
            cu_seqlens = batch["cu_seqlens"].to(device)
            unpad_indices = batch["unpad_indices"].to(device)
            max_seqlen = batch["max_seqlen"]

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
            phase = "recovery"
            batch_idx = step_idx - superposition_steps

        if phase != current_phase and phase_steps > 0:
            avg_phase_loss = phase_loss / phase_steps
            print(f"Phase {current_phase}: {phase_steps} steps, Loss: {avg_phase_loss:.4f}")
            logger.log(
                {
                    f"loss/{current_phase}_phase_loss": avg_phase_loss,
                    f"training/{current_phase}_phase_steps": phase_steps,
                },
                step=global_step,
                detailed_logging=True,
            )
            current_phase = phase
            phase_loss = 0.0
            phase_steps = 0

        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        total_loss += loss_value
        phase_loss += loss_value
        phase_steps += 1
        auxillary_loss_free_update(model, all_topk_indices, update_rate, attention_mask)

        epoch = step_idx // max(1, len(train_loader))
        metrics = logger.log_training_metrics(loss, optimizer, update_rate, global_step, epoch, batch_idx)
        metrics.update(logger.log_moe_metrics(all_topk_indices, global_step, attention_mask))
        metrics["training/phase"] = 0 if phase == "tst" else 1
        if phase == "tst":
            metrics["token_superposition/active_labels"] = labels.numel()

        detailed_logging = (global_step % logger.detailed_frequency == 0)
        logger.log(metrics, step=global_step, model=model, detailed_logging=detailed_logging)

        global_step += 1

        if global_step == total_steps and phase_steps > 0:
            avg_phase_loss = phase_loss / phase_steps
            print(f"Phase {current_phase}: {phase_steps} steps, Loss: {avg_phase_loss:.4f}")
            logger.log(
                {
                    f"loss/{current_phase}_phase_loss": avg_phase_loss,
                    f"training/{current_phase}_phase_steps": phase_steps,
                },
                step=global_step,
                detailed_logging=True,
            )
            phase_loss = 0.0
            phase_steps = 0

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

    avg_loss = total_loss / max(1, total_steps)
    train_time = time.time() - train_start_time
    logger.log(
        {
            'loss/train_loss': avg_loss,
            'training/train_time': train_time,
            'training/total_steps': total_steps,
        },
        step=global_step,
        detailed_logging=True,
    )

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
    config = ModelConfig()
    data_max_length = config.sequence_length
    if config.token_superposition_bag_size > 1 and config.token_superposition_ratio > 0.0:
        data_max_length = config.sequence_length * config.token_superposition_bag_size

    train_dataset, tokenizer, _ = load_and_preprocess_data(max_length=data_max_length)
    config = ModelConfig(vocab_size=len(tokenizer))

    model = MoEModel(config)

    count_parameters_layerwise(model)
    model.headless_forward = torch.compile(model.headless_forward, dynamic=True)
    train(
        model,
        train_dataset,
        tokenizer,
        config.sequence_length,
        superposition_bag_size=config.token_superposition_bag_size,
        superposition_ratio=config.token_superposition_ratio,
    )

if __name__ == "__main__":
    main()
