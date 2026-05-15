import torch
import torch._inductor.config
import torch.nn as nn
import torch.nn.functional as F

torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
import math
import random
import time
import json
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
from utils.trainutils import AimLogger, count_parameters_layerwise, save_checkpoint
from utils.tokenizer import load_tokenizer
from modeling.model import MoEModel
from modeling.model_config import (
    ModelConfig,
    SUPERPOSITION_REFERENCE_SIZE,
    SUPERPOSITION_SCHEDULE_BETA,
    set_superposition,
)
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

def bag_label_entropy_floor(labels, s):
    """Return the irreducible per-raw-token entropy of each target bag."""
    if labels.numel() == 0:
        return torch.zeros((), device=labels.device, dtype=torch.float32)

    labels = labels.long()
    s_f = s.float()
    cols = torch.arange(labels.size(1), device=labels.device)[None, :]
    active = cols < s[:, None]

    inactive_label = labels.new_full((), -1)
    sorted_labels = torch.where(active, labels, inactive_label).sort(dim=1).values
    sorted_active = sorted_labels >= 0
    first_col = torch.ones((labels.size(0), 1), device=labels.device, dtype=torch.bool)
    run_start = sorted_active & torch.cat([first_col, sorted_labels[:, 1:] != sorted_labels[:, :-1]], dim=1)
    run_ids = (run_start.cumsum(dim=1) - 1).clamp_min(0)

    counts = torch.zeros_like(s_f[:, None].expand_as(labels), dtype=torch.float32)
    counts.scatter_add_(1, run_ids, sorted_active.float())
    probs = counts / s_f[:, None].clamp_min(1.0)
    entropy_terms = torch.where(
        counts > 0,
        -probs * torch.log(probs.clamp_min(torch.finfo(torch.float32).tiny)),
        torch.zeros_like(probs),
    )
    row_entropy = entropy_terms.sum(dim=-1)
    total_raw = s_f.sum().clamp_min(1.0)
    return (row_entropy * s_f).sum() / total_raw

def ce_equivalent_reducible_loss(raw_loss, vocab_size, entropy_floor):
    """Map bag CE onto the same scale as ordinary single-token CE."""
    entropy_floor = entropy_floor.to(device=raw_loss.device, dtype=raw_loss.dtype).detach()
    random_loss = raw_loss.new_tensor(math.log(vocab_size))
    reducible_gap = (random_loss - entropy_floor).clamp_min(1e-6)
    return (raw_loss - entropy_floor) * (random_loss / reducible_gap), reducible_gap

def bucket_max_seqlen(max_seqlen, max_length):
    if max_seqlen <= 0:
        return 0
    return min(((max_seqlen + 31) // 32) * 32, max_length)

def sample_superposition_size(t: float, max_size: int, beta: float) -> int:
    """Sample s ∈ {1, 2, 4, ..., effective_max} from a schedule-dependent categorical.

    Two schedules compose:
      1. Bucket ejection: every 20% of training, drop the currently-largest
         bucket from the support. So at t ∈ [0, 0.2) the support is the full
         {1, 2, …, max_size}; at t ∈ [0.2, 0.4) the top is removed; etc. Once
         the support collapses to {1}, s=1 is forced.
      2. Within the remaining support, sample with logits β · (1 − 2t) · log2(s).
         At t=0 large s is favored, at t=0.5 the distribution over remaining
         buckets is uniform, at t=1 small s is favored.
    """
    if max_size <= 1:
        return 1
    log2_max = int(math.log2(max_size))
    drops = min(int(t / 0.2), log2_max)
    effective_log2_max = log2_max - drops
    if effective_log2_max <= 0:
        return 1
    sizes = [1 << k for k in range(effective_log2_max + 1)]
    coef = beta * (1.0 - 2.0 * t)
    logits = [coef * k for k in range(effective_log2_max + 1)]
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    z = sum(exps)
    weights = [e / z for e in exps]
    return random.choices(sizes, weights=weights, k=1)[0]


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
    superposition_max_size=SUPERPOSITION_REFERENCE_SIZE,
    superposition_schedule_beta=SUPERPOSITION_SCHEDULE_BETA,
):
    device = torch.device("cuda")
    model.to(device)

    weight_decay = 0.01
    optimizer_betas = (0.9, 0.95)
    optimizer = build_weight_decay_optm(model, learning_rate, weight_decay=weight_decay, betas=optimizer_betas)

    superposition_enabled = superposition_max_size > 1
    superposition_raw_length = sequence_length * max(superposition_max_size, 1)

    # Single dataloader; collator.max_length is mutated per-step to the
    # sampled s. Safe because num_workers defaults to 0 (synchronous collate).
    collator = PackedTokenizedCollator(sequence_length, tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collator,
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
        'attention/do_rope': model.config.do_rope,
        'token_superposition/enabled': superposition_enabled,
        'token_superposition/max_size': superposition_max_size,
        'token_superposition/schedule_beta': superposition_schedule_beta,
        'token_superposition/max_raw_sequence_length': superposition_raw_length,
        'loss/normalization': 'ce_equivalent_reducible_loss',
    })

    def next_from_loader(loader, iterator):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    def next_tst_batch(s_value, loader, iterator):
        # Try up to len(loader) raw batches before giving up; some raw batches
        # may have no segment long enough to produce a TST batch at this s.
        for _ in range(max(1, len(loader))):
            raw_batch, iterator = next_from_loader(loader, iterator)
            tst_batch = build_token_superposition_batch(
                raw_batch,
                s_value,
                sequence_length,
                tokenizer.pad_token_id,
            )
            if tst_batch is not None:
                return tst_batch, iterator
        return None, iterator

    model.train()
    total_loss = 0.0
    train_start_time = time.time()
    train_iter = iter(train_loader)

    for step_idx in tqdm(range(total_steps)):
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        t = step_idx / max(1, total_steps - 1)
        s = sample_superposition_size(t, superposition_max_size, superposition_schedule_beta)
        collator.max_length = sequence_length * s

        if s == 1:
            batch, train_iter = next_from_loader(train_loader, train_iter)
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
                raw_loss = linear_cross_entropy(loss_embeddings, classifier, labels, ignore_index=-100)
            entropy_floor = raw_loss.new_zeros(())
            loss, loss_normalizer = ce_equivalent_reducible_loss(raw_loss, classifier.size(0), entropy_floor)
            active_labels = labels.numel()
        else:
            tst_batch, train_iter = next_tst_batch(s, train_loader, train_iter)
            if tst_batch is None:
                # No TST batch available at this s. Skip step; do not advance the optimizer.
                continue
            input_ids = tst_batch["input_ids"].to(device)
            attention_mask = tst_batch["attention_mask"].to(device)
            position_ids = tst_batch["position_ids"].to(device)
            cu_seqlens = tst_batch["cu_seqlens"].to(device)
            unpad_indices = tst_batch["unpad_indices"].to(device)
            max_seqlen = tst_batch["max_seqlen"]
            loss_indices = tst_batch["loss_indices"].to(device)
            labels = tst_batch["labels"].to(device)
            s_tensor = tst_batch["s"].to(device)

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
                raw_loss = mcce_raw_token_mean_v2(
                    loss_embeddings,
                    classifier,
                    labels,
                    s_tensor,
                    check_label_values=False,
                )
            entropy_floor = bag_label_entropy_floor(labels, s_tensor)
            loss, loss_normalizer = ce_equivalent_reducible_loss(raw_loss, classifier.size(0), entropy_floor)
            active_labels = labels.numel()

        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        raw_loss_value = raw_loss.item()
        entropy_floor_value = entropy_floor.item()
        loss_normalizer_value = loss_normalizer.item()
        total_loss += loss_value
        auxillary_loss_free_update(model, all_topk_indices, update_rate, attention_mask)

        epoch = step_idx // max(1, len(train_loader))
        metrics = logger.log_training_metrics(loss, optimizer, update_rate, global_step, epoch, step_idx)
        metrics.update(logger.log_moe_metrics(all_topk_indices, global_step, attention_mask))
        metrics["loss/raw_batch_loss"] = raw_loss_value
        metrics["loss/entropy_floor"] = entropy_floor_value
        metrics["loss/reducible_gap"] = loss_normalizer_value
        metrics["loss/reducible_raw_loss"] = raw_loss_value - entropy_floor_value
        metrics["token_superposition/s"] = s
        metrics["token_superposition/schedule_t"] = t
        metrics["token_superposition/active_labels"] = active_labels
        metrics[f"loss/s_{s}"] = loss_value
        metrics[f"loss/raw_s_{s}"] = raw_loss_value

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
    config = set_superposition(ModelConfig(), enabled=False)
    data_max_length = config.sequence_length * max(config.superposition_max_size, 1)

    train_dataset, tokenizer, _ = load_and_preprocess_data(max_length=data_max_length)
    config = set_superposition(ModelConfig(vocab_size=len(tokenizer)), enabled=False)

    model = MoEModel(config)

    count_parameters_layerwise(model)
    model.headless_forward = torch.compile(model.headless_forward, dynamic=True)
    train(
        model,
        train_dataset,
        tokenizer,
        config.sequence_length,
        superposition_max_size=config.superposition_max_size,
        superposition_schedule_beta=config.superposition_schedule_beta,
    )

if __name__ == "__main__":
    main()
