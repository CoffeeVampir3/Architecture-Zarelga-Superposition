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
from optimizer.muon import SingleDeviceMuonMDWithAuxAdam
from optimizer.hybrid import HybridGraphOptimizer
from engram.GramReaperSparse import GramReaperSparse
from modeling.model_config import ModelConfig
from current_configuration import build_config, TrainingConfig, DEFAULT_TRAINING
from cut_cross_entropy import linear_cross_entropy

from modeling.zRMSNorm import ZeroCenteredRMSNorm

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
            token_ids = token_ids[:self.max_length - 1] + [self.eos_token_id]
        return {"token_ids": token_ids}

class PackedTokenizedCollator:
    """Concatenate records into one flat unpadded token stream.

    Returns input_ids/position_ids as [1, total_tokens], labels as [total_tokens],
    and cu_seqlens as [num_segments + 1].
    """

    def __init__(self, max_length, pad_token_id, ignore_index=-100):
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, examples):
        flat_token_ids = []
        flat_position_ids = []
        flat_labels = []
        cu_seqlens = [0]
        max_seqlen = 0

        for example in examples:
            token_ids = example["token_ids"]
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            if not token_ids:
                continue

            segment_len = len(token_ids)
            flat_token_ids.extend(token_ids)
            flat_position_ids.extend(range(segment_len))
            flat_labels.extend(token_ids[1:])
            flat_labels.append(self.ignore_index)
            cu_seqlens.append(cu_seqlens[-1] + segment_len)
            if segment_len > max_seqlen:
                max_seqlen = segment_len

        total = len(flat_token_ids)
        bucketed_max_seqlen = min(((max_seqlen + 31) // 32) * 32, self.max_length) if total else 0

        input_ids = torch.tensor(flat_token_ids, dtype=torch.long).view(1, total)
        position_ids = torch.tensor(flat_position_ids, dtype=torch.long).view(1, total)
        labels = torch.tensor(flat_labels, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "labels": labels,
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
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

def auxillary_loss_free_update(model, all_topk_indices, update_rate, attention_mask=None):
    valid_tokens = attention_mask.bool() if attention_mask is not None else None

    with torch.no_grad():
        for layer_idx, topk_idx in enumerate(all_topk_indices):
            if topk_idx is None:
                continue
            if valid_tokens is not None:
                topk_idx = topk_idx[valid_tokens]

            if topk_idx.numel() == 0:
                continue

            gate = model.layers[layer_idx].mlp.gate
            expert_counts = torch.bincount(topk_idx.flatten(), minlength=gate.n_routed_experts).float()
            errors = expert_counts.mean() - expert_counts
            gate.expert_biases.add_(update_rate * torch.sign(errors))

def build_muonmd_optimizer(model, device, muon_lr=0.02, adam_lr=3e-4,
                           momentum=0.95, gain_lr=1e-3,
                           adam_betas=(0.9, 0.95), adam_eps=1e-16,
                           head_weight_decay=0.1,
                           embedding_lr=3e-3 * 2**0.5, embedding_beta=0.9,
                           capture_warmup_steps=5):
    """Build dense and sparse optimizer groups for the model."""
    embedding_param = model.embedding.weight

    engram_sparse_params = []
    engrams = getattr(model, "engrams", None)
    if engrams is not None:
        for engram in engrams.values():
            engram_sparse_params.append(engram.embedding.weight)
            if engram.importance_weighting:
                engram_sparse_params.append(engram.imp_table.weight)
    sparse_param_ids = {id(embedding_param)} | {id(p) for p in engram_sparse_params}

    muon_params, router_params, embed_params, scalar_params = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in sparse_param_ids:
            continue
        is_head = ('output_layer' in name)
        is_router = ('mlp.gate.weight' in name)
        if is_head:
            embed_params.append(param)
        elif param.ndim >= 2:
            (router_params if is_router else muon_params).append(param)
        else:
            scalar_params.append(param)

    muon_lr_tensor = torch.tensor(float(muon_lr), device=device)
    router_lr_tensor = torch.tensor(float(muon_lr), device=device)
    embed_lr_tensor = torch.tensor(float(adam_lr), device=device)
    scalar_lr_tensor = torch.tensor(float(adam_lr), device=device)

    param_groups = [
        dict(params=muon_params, use_muon=True, lr=muon_lr_tensor,
             momentum=momentum, gain_lr=gain_lr, capturable=True),
    ]
    if router_params:
        param_groups.append(
            dict(params=router_params, use_muon=True, lr=router_lr_tensor,
                 momentum=momentum, gain_lr=gain_lr, norm_axis="row",
                 gain_mode="col", rescale_mode="muon", capturable=True),
        )
    if embed_params:
        param_groups.append(
            dict(params=embed_params, use_muon=False, lr=embed_lr_tensor,
                 betas=adam_betas, eps=adam_eps,
                 weight_decay=head_weight_decay, capturable=True),
        )
    param_groups.append(
        dict(params=scalar_params, use_muon=False, lr=scalar_lr_tensor,
             betas=adam_betas, eps=adam_eps, weight_decay=0.0, capturable=True),
    )
    dense_optimizer = SingleDeviceMuonMDWithAuxAdam(param_groups)

    sparse_param_groups = [dict(params=[embedding_param], unit_norm=True)]
    if engram_sparse_params:
        sparse_param_groups.append(dict(params=engram_sparse_params, unit_norm=False))
    sparse_optimizer = GramReaperSparse(
        sparse_param_groups, lr=float(embedding_lr), beta=embedding_beta, unit_norm=True,
    )

    return HybridGraphOptimizer(
        dense_optimizer, sparse_optimizer, capture_warmup_steps=capture_warmup_steps,
    )

def _lr_scalar(value):
    return value.item() if torch.is_tensor(value) else value


class CosineWarmupScheduler:
    """Cosine decay with linear warmup."""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.warmup_steps = min(max(1, warmup_steps), self.total_steps)
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [_lr_scalar(pg['lr']) for pg in optimizer.param_groups]
        self.step_count = 0
        self.peak_lr = self.base_lrs[0]
        self.min_lr = self.peak_lr * min_lr_ratio

    def _fraction(self):
        if self.step_count <= self.warmup_steps:
            return self.step_count / self.warmup_steps
        progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)
        return self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))

    def step(self, increment=1):
        self.step_count += increment
        fraction = self._fraction()
        for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
            lr = base_lr * fraction
            lr_ref = param_group['lr']
            if torch.is_tensor(lr_ref):
                lr_ref.fill_(lr)
            else:
                param_group['lr'] = lr
        return self.base_lrs[0] * fraction

class LinearDecayScheduler:
    """Linear LR decay with optional warmup."""
    def __init__(self, optimizer, total_steps, warmup_steps=0, min_lr_ratio=0.0):
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.warmup_steps = min(max(0, warmup_steps), self.total_steps)
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [_lr_scalar(pg['lr']) for pg in optimizer.param_groups]
        self.step_count = 0
        self.peak_lr = self.base_lrs[0]
        self.min_lr = self.peak_lr * min_lr_ratio

    def _fraction(self):
        if self.warmup_steps > 0 and self.step_count < self.warmup_steps:
            return self.step_count / self.warmup_steps
        progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 1.0 - (1.0 - self.min_lr_ratio) * progress

    def step(self, increment=1):
        self.step_count += increment
        fraction = self._fraction()
        for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
            lr = base_lr * fraction
            lr_ref = param_group['lr']
            if torch.is_tensor(lr_ref):
                lr_ref.fill_(lr)
            else:
                param_group['lr'] = lr
        return self.base_lrs[0] * fraction


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

def format_token_count(n):
    """Human-readable token count, e.g. 4B, 19M, 1.25M, 870K, 512."""
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= threshold:
            s = f"{n / threshold:.2f}".rstrip("0").rstrip(".")
            return f"{s}{suffix}"
    return str(int(n))


def train(
    model,
    train_dataset,
    tokenizer,
    sequence_length,
    training: TrainingConfig = None,
):
    if training is None:
        training = DEFAULT_TRAINING

    # Pull the run knobs out of the config into locals the loop below reads.
    num_epochs = training.num_epochs
    batch_size = training.batch_size
    muon_lr = training.muon_lr
    adam_lr = training.adam_lr
    update_rate = training.update_rate
    head_weight_decay = training.head_weight_decay
    checkpoint_interval_steps = training.checkpoint_interval_steps
    max_rolling_checkpoints = training.max_rolling_checkpoints
    log_interval = training.log_interval

    device = torch.device("cuda")
    model.to(device)

    muon_momentum = training.muon_momentum
    gain_lr = training.gain_lr
    adam_betas = training.adam_betas
    adam_eps = training.adam_eps
    optimizer = build_muonmd_optimizer(
        model, device, muon_lr=muon_lr, adam_lr=adam_lr,
        momentum=muon_momentum, gain_lr=gain_lr,
        adam_betas=adam_betas, adam_eps=adam_eps,
        head_weight_decay=head_weight_decay,
        embedding_lr=training.embedding_lr, embedding_beta=training.embedding_beta,
        capture_warmup_steps=training.capture_warmup_steps,
    )

    collator = PackedTokenizedCollator(sequence_length, tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collator,
        num_workers=training.num_workers,
        persistent_workers=True,
        prefetch_factor=training.prefetch_factor,
        pin_memory=True,
    )

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    logger = AimLogger(repo='logs/aim', experiment='moe_training', detailed_frequency=20)
    logger.register_moe_layers(model)
    logger.register_engrams(model)
    global_step = 0

    total_steps = len(train_loader) * num_epochs
    warmup_steps = training.warmup_steps
    scheduler = LinearDecayScheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=training.min_lr_ratio,
    )
    logger.log_params({
        'batch_size': batch_size,
        'muon_lr': muon_lr,
        'adam_lr': adam_lr,
        'num_epochs': num_epochs,
        'sequence_length': sequence_length,
        'update_rate': update_rate,
        'optimizer': 'SingleDeviceMuonMDWithAuxAdam (capturable / CUDA-graph)',
        'optimizer_momentum': muon_momentum,
        'optimizer_gain_lr': gain_lr,
        'adam_betas': adam_betas,
        'adam_eps': adam_eps,
        'weight_decay': 0.0,
        'head_weight_decay': head_weight_decay,
        'lr_schedule': 'linear_decay_warmup_free',
        'warmup_steps': warmup_steps,
        'min_lr': muon_lr * scheduler.min_lr_ratio,
        'train_examples': len(train_dataset),
        'checkpoint_interval_steps': checkpoint_interval_steps,
        'max_rolling_checkpoints': max_rolling_checkpoints,
        'attention/do_rope': model.config.do_rope,
        'attention/pos_rope_dims': model.config.pos_rope_dims,
        'engram/enabled': model.config.engram.enabled,
        'engram/layers': list(model.config.engram.layers),
        'engram/orders': list(model.config.engram.orders),
        'engram/n_heads': model.config.engram.n_heads,
        'engram/rows_per_head': model.config.engram.rows_per_head,
        'engram/dim_per_head': model.config.engram.dim_per_head,
        'engram/alpha_init': model.config.engram.alpha_init,
        'engram/importance_weighting': model.config.engram.importance_weighting,
        'engram/head_norm': model.config.engram.head_norm,
        'engram/learned_gate': model.config.engram.learned_gate,
        'loss/backend': 'linear_cross_entropy',
    })

    def next_batch(loader, iterator):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    model.train()
    total_loss = 0.0
    total_tokens_processed = 0
    loss_accum = torch.zeros((), device=device)
    steps_since_log = 0
    train_start_time = time.time()
    train_iter = iter(train_loader)

    step_idx = 0
    optimizer_step_count = 0
    last_checkpoint_step = 0
    pbar = tqdm(total=total_steps)
    while step_idx < total_steps:
        optimizer.zero_grad(set_to_none=False)

        t = step_idx / max(1, total_steps - 1)
        scheduler.step(increment=1)

        batch, train_iter = next_batch(train_loader, train_iter)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        position_ids = batch["position_ids"].to(device, non_blocking=True)
        cu_seqlens = batch["cu_seqlens"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        max_seqlen = batch["max_seqlen"]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            embeddings, all_topk_indices = model.headless_forward(
                input_ids,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            classifier = model.get_classifier_weights()
            loss = linear_cross_entropy(
                embeddings.reshape(-1, embeddings.size(-1)),
                classifier,
                labels,
                ignore_index=-100,
            )

        loss.backward()

        optimizer.step()

        loss_accum += loss.detach()
        auxillary_loss_free_update(model, all_topk_indices, update_rate, attention_mask=None)

        step_idx += 1
        optimizer_step_count += 1
        global_step = step_idx
        steps_since_log += 1
        total_tokens_processed += int(input_ids.shape[1])
        epoch = step_idx // max(1, len(train_loader))
        pbar.update(1)

        if optimizer_step_count % log_interval == 0:
            chunk_loss = loss_accum.item()
            loss_accum.zero_()
            total_loss += chunk_loss
            mean_loss = chunk_loss / steps_since_log
            steps_since_log = 0

            num_tokens = int(input_ids.shape[1])
            num_segments = int(cu_seqlens.numel() - 1)
            mem_alloc_gb = torch.cuda.memory_allocated() / 1e9
            mem_reserved_gb = torch.cuda.memory_reserved() / 1e9
            mem_max_alloc_gb = torch.cuda.max_memory_allocated() / 1e9
            frag_gb = mem_reserved_gb - mem_alloc_gb
            total_tokens_human = format_token_count(total_tokens_processed)
            pbar.set_postfix(
                loss=f"{mean_loss:.3f}",
                tok=num_tokens,
                total=total_tokens_human,
                seg=num_segments,
                alloc=f"{mem_alloc_gb:.2f}",
                resv=f"{mem_reserved_gb:.2f}",
                frag=f"{frag_gb:.2f}",
            )
            tqdm.write(
                f"[mem] step={step_idx} tok={num_tokens} total={total_tokens_human} seg={num_segments} "
                f"alloc={mem_alloc_gb:.2f}GB resv={mem_reserved_gb:.2f}GB "
                f"frag={frag_gb:.2f}GB max_alloc={mem_max_alloc_gb:.2f}GB"
            )

            detailed_logging = (optimizer_step_count % logger.detailed_frequency == 0)
            metrics = logger.log_training_metrics(mean_loss, optimizer, update_rate)
            metrics.update(logger.log_moe_metrics(all_topk_indices, global_step, attention_mask=None))
            metrics["mem/num_tokens"] = num_tokens
            metrics["mem/total_tokens_processed"] = total_tokens_processed
            metrics["mem/num_segments"] = num_segments
            metrics["mem/allocated_gb"] = mem_alloc_gb
            metrics["mem/reserved_gb"] = mem_reserved_gb
            metrics["mem/max_allocated_gb"] = mem_max_alloc_gb
            metrics["mem/fragmentation_gb"] = frag_gb
            metrics.update(logger.log_engram_metrics(global_step, detailed=detailed_logging))
            logger.log(metrics, step=global_step, model=model, detailed_logging=detailed_logging)

        if checkpoint_interval_steps > 0 and (
            step_idx // checkpoint_interval_steps > last_checkpoint_step // checkpoint_interval_steps
        ):
            last_checkpoint_step = step_idx
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{step_idx}.safetensors"
            save_checkpoint(
                model,
                optimizer,
                str(checkpoint_path),
                scheduler=scheduler,
                global_step=step_idx,
                epoch=epoch + 1,
            )
            prune_rolling_checkpoints(checkpoint_dir, max_rolling_checkpoints)
            print(f"Checkpoint saved: {checkpoint_path}")

    if steps_since_log:
        total_loss += loss_accum.item()
    avg_loss = total_loss / max(1, optimizer_step_count)
    train_time = time.time() - train_start_time
    logger.log(
        {
            'loss/train_loss': avg_loss,
            'training/train_time': train_time,
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
    sizing = build_config(vocab_size=1, pad_token_id=0)
    data_max_length = sizing.sequence_length

    train_dataset, tokenizer, _ = load_and_preprocess_data(max_length=data_max_length)
    config = build_config(len(tokenizer), tokenizer.pad_token_id)

    model = MoEModel(config)

    count_parameters_layerwise(model)
    model.headless_forward = torch.compile(model.headless_forward, dynamic=True)
    train(
        model,
        train_dataset,
        tokenizer,
        config.sequence_length,
        training=TrainingConfig(),
    )

if __name__ == "__main__":
    main()
