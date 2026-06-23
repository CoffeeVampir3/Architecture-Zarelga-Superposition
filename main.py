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
from modeling.model_config import (
    ModelConfig,
    SUPERPOSITION_REFERENCE_SIZE,
)
from model_variants import build_config
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
# `update_rate` is the bias-update speed (DeepSeek-V3's gamma). It is the *only* load-balancing
# force here, so it must be strong enough to overcome routing drift: V3 uses 1e-3 (see train()).
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

def sample_superposition_size(t: float, max_size: int) -> int:
    """Pick a packing group size s from a simple two-phase schedule.

      - t ∈ [0.0, 0.1): s = max_size (token grouping in groups of max_size)
      - t ∈ [0.1, 1.0]: s = 1 (normal raw-token training)

    when max_size <= 1 (TST disabled) s is always 1.
    """
    if max_size <= 1:
        return 1
    if t < 0.1:
        return max_size
    return 1


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

def build_muonmd_optimizer(model, device, muon_lr=0.02, adam_lr=3e-4,
                           momentum=0.95, gain_lr=1e-3,
                           adam_betas=(0.9, 0.95), adam_eps=1e-16,
                           embedding_weight_decay=0.1,
                           embedding_lr=3e-3, embedding_beta=0.9,
                           capture_warmup_steps=5):
    """Build the HybridGraphOptimizer: a CUDA-graph-captured dense MuonMD + aux-Adam
    optimizer, plus an eager GramReaperSparse for the sparse input embedding.

    Muon group (Frobenius sphere): attention q/k/v/o and head gate, the shared-expert
    MLPs, and the 3-D routed-expert stacks (E, out, in) (handled as E independent
    per-expert matrices). No weight decay (directions live on a fixed-norm sphere).
    Router group (row sphere): the MoE router gate, normalized along the expert axis
    (each expert's gating row on its own sphere), with the standard Muon shape factor
    max(1, sqrt(dout/din)) so the wide router matrix isn't scaled away from `lr` --
    per the paper's MoE recipe. gain_mode="col" (not the "both" default): a per-row
    gain is a per-expert magnitude that would undo the row-sphere normalization and
    let popular experts inflate their gating scores (rich-get-richer collapse), so the
    router keeps only the per-input-feature (column) gain, which is shared across
    experts. Load balancing is left entirely to the aux-loss-free bias.
    Output-head Adam group: the (dense) output_layer, with weight decay. Its gradient
    is dense (every vocab row participates in the softmax each step), so it stays in
    the captured dense optimizer rather than the sparse path.
    Scalar Adam group: all 1-D params (ZeroCenteredRMSNorm gains, the final norm, any
    biases) with NO weight decay -- decaying gains would fight the normalization layers.

    Sparse embedding (GramReaperSparse, eager): the input embedding table trains on its
    own per-row RMSProp with a unit-sphere projection (unit_norm=True), consuming the
    COO gradient nn.Embedding(sparse=True) produces. Its step has data-dependent shapes
    and a host sync, so it cannot be captured and runs outside the graph every step.

    Dense-group learning rates are 0-dim CUDA tensors so a CUDA-graph-captured step()
    reads the scheduler's in-place updates on every replay. capturable=True keeps the
    Adam step counters on-device so bias correction stays correct across graph replays.
    The sparse group's LR is a plain float (eager step), updated in place by the scheduler.
    """
    embedding_param = model.embedding.weight

    # Engram n-gram tables are sparse-gradient too, but zero-init and NOT unit-norm
    # constrained, so they ride their own GramReaperSparse group (unit_norm=False)
    # rather than the input embedding's sphere group. Their dense side-modules
    # (value_proj matrix -> Muon, value_proj bias + alpha -> scalar Adam) fall through
    # the normal name/ndim split below.
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
            continue  # routed to a sparse optimizer group below
        # The dense output head (untied LM head) -- dense grad, stays in aux-Adam.
        is_head = ('output_layer' in name)
        # The MoE router gate lives at `layers.*.mlp.gate.weight`; normalize it along
        # the expert axis (rows). The attention head gate (`...head_gate_proj.weight`)
        # and the shared-expert gate (`...shared_experts.gate_proj.weight`) do NOT match.
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
                 weight_decay=embedding_weight_decay, capturable=True),
        )
    param_groups.append(
        dict(params=scalar_params, use_muon=False, lr=scalar_lr_tensor,
             betas=adam_betas, eps=adam_eps, weight_decay=0.0, capturable=True),
    )
    dense_optimizer = SingleDeviceMuonMDWithAuxAdam(param_groups)

    # Sphere-constrained per-row RMSProp on the sparse embedding table (paper: keep
    # embedding rows at unit L2 norm throughout, no weight decay). Engram tables, if
    # present, get a second group with unit_norm=False: they are zero-init and must
    # keep their untouched zero rows at zero (a sphere projection would push them off).
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
    """Cosine decay with linear warmup. Each param group's LR is written into its
    0-dim LR tensor IN PLACE (.fill_), so a CUDA-graph-captured optimizer step reads
    the new value on every replay. (Re-binding the tensor would leave the captured
    graph pointing at a stale buffer -- the schedule would freeze at the captured
    value.)

    Each group keeps its own peak LR (read from its LR tensor at construction) and is
    scaled by the same warmup/cosine fraction, so the Muon and Adam groups can run at
    very different magnitudes while sharing one schedule shape.
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.warmup_steps = min(max(1, warmup_steps), self.total_steps)
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [_lr_scalar(pg['lr']) for pg in optimizer.param_groups]
        self.step_count = 0
        # Kept for checkpoint serialization (save_checkpoint reads these).
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
    """Warmup-free linear LR decay -- the paper's dense recipe (Hägele et al. 2026):
    'dense models use a linear LR decay to 1e-8 for all groups', and MD decoupling
    removes the need for warmup (the large early updates it exists to tame never
    appear on the sphere, and dropping warmup even improves the loss).

    Shares the in-place 0-dim LR tensor update of CosineWarmupScheduler so a
    CUDA-graph-captured optimizer step reads the new LR on every replay. Each param
    group keeps its own peak LR and is scaled by the same fraction; the gains' LR is
    a separate group key and is intentionally left unscaled (held at the Adam value).

    warmup_steps defaults to 0 (warmup-free); a small linear warmup can be restored by
    passing warmup_steps > 0. min_lr_ratio is the floor as a fraction of each group's
    peak (0.0 == decay to ~0, matching the paper's 1e-8 endpoint).
    """
    def __init__(self, optimizer, total_steps, warmup_steps=0, min_lr_ratio=0.0):
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.warmup_steps = min(max(0, warmup_steps), self.total_steps)
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [_lr_scalar(pg['lr']) for pg in optimizer.param_groups]
        self.step_count = 0
        # Kept for checkpoint serialization (save_checkpoint reads these).
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

def train(
    model,
    train_dataset,
    tokenizer,
    sequence_length,
    num_epochs=1,
    batch_size=16,
    muon_lr=0.02,
    adam_lr=3e-4,
    update_rate=1e-3,
    embedding_weight_decay=0.1,
    checkpoint_interval_steps=10_000,
    max_rolling_checkpoints=5,
    superposition_max_size=SUPERPOSITION_REFERENCE_SIZE,
):
    device = torch.device("cuda")
    model.to(device)

    muon_momentum = 0.95
    gain_lr = 1e-3
    adam_betas = (0.9, 0.95)
    adam_eps = 1e-16
    optimizer = build_muonmd_optimizer(
        model, device, muon_lr=muon_lr, adam_lr=adam_lr,
        momentum=muon_momentum, gain_lr=gain_lr,
        adam_betas=adam_betas, adam_eps=adam_eps,
        embedding_weight_decay=embedding_weight_decay,
    )

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
        collate_fn=lambda xs: xs,
    )

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    logger = AimLogger(repo='logs/aim', experiment='moe_training', detailed_frequency=20)
    logger.register_moe_layers(model)
    logger.register_engrams(model)
    global_step = 0

    total_steps = len(train_loader) * num_epochs
    # Paper recipe: warmup-free linear decay to ~0 (their 1e-8 endpoint) for all
    # groups. MD decoupling makes warmup unnecessary on the sphere.
    warmup_steps = 0
    scheduler = LinearDecayScheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=0.0,
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
        'weight_decay': 0.0,  # sphere (Muon/MD) groups -- no decay
        'embedding_weight_decay': embedding_weight_decay,
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
        'token_superposition/enabled': superposition_enabled,
        'token_superposition/max_size': superposition_max_size,
        'token_superposition/max_raw_sequence_length': superposition_raw_length,
        'loss/mcce_enabled': superposition_enabled,
        'loss/backend': 'mcce_raw_token_mean_v2' if superposition_enabled else 'linear_cross_entropy',
        'loss/normalization': 'ce_equivalent_reducible_loss',
    })

    def next_from_loader(loader, iterator):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    def fetch_records(s_value, loader, iterator):
        records = []
        for _ in range(s_value):
            batch_records, iterator = next_from_loader(loader, iterator)
            records.extend(batch_records)
        return records, iterator

    def next_tst_batch(s_value, loader, iterator):
        # Each retry consumes s_value loader batches; cap retries so a failure
        # to build a TST batch can't burn more than ~one dataset pass.
        max_retries = max(1, len(loader) // max(1, s_value))
        for _ in range(max_retries):
            records, iterator = fetch_records(s_value, loader, iterator)
            raw_batch = collator(records)
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

    step_idx = 0
    optimizer_step_count = 0
    last_checkpoint_step = 0
    # The optimizer (HybridGraphOptimizer) owns the CUDA-graph capture-once/replay-many
    # state machine for its dense MuonMD + aux-Adam step, and steps the eager sparse
    # embedding optimizer outside the graph. See optimizer/hybrid.py.
    pbar = tqdm(total=total_steps)
    while step_idx < total_steps:
        # set_to_none=False keeps the dense grad buffers at stable addresses across
        # iterations, which the captured optimizer-step graph reads on every replay.
        # The coordinator overrides this to set_to_none=True for the sparse embedding
        # grad (a fresh COO tensor each backward that cannot be zeroed in place).
        optimizer.zero_grad(set_to_none=False)

        t = step_idx / max(1, total_steps - 1)
        s = sample_superposition_size(t, superposition_max_size)
        collator.max_length = sequence_length * s
        scheduler.step(increment=s)

        if s == 1:
            records, train_iter = fetch_records(1, train_loader, train_iter)
            batch = collator(records)
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
                # next_tst_batch already consumed loader batches in its retries;
                # advance step_idx so the budget loop can't get stuck.
                step_idx += s
                pbar.update(s)
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

        # Dense step is captured-once/replayed and the sparse embedding step runs
        # eagerly; the coordinator handles both. The TST skip path above `continue`s
        # before reaching here, so a skipped batch never triggers a step.
        optimizer.step()

        loss_value = loss.item()
        entropy_floor_value = entropy_floor.item()
        loss_normalizer_value = loss_normalizer.item()
        total_loss += loss_value
        auxillary_loss_free_update(model, all_topk_indices, update_rate, attention_mask)

        step_idx += s
        optimizer_step_count += 1
        global_step = step_idx
        pbar.update(s)

        # --- memory / shape instrumentation (OOM debugging) ---
        # num_rows is the (variable) packed batch dimension; if it swings step to
        # step and reserved_gb climbs while allocated_gb stays flat, the OOM is
        # allocator fragmentation from non-stationary shapes, not a true leak.
        num_rows = int(input_ids.shape[0])
        num_valid_tokens = int(attention_mask.sum().item())
        mem_alloc_gb = torch.cuda.memory_allocated() / 1e9
        mem_reserved_gb = torch.cuda.memory_reserved() / 1e9
        mem_max_alloc_gb = torch.cuda.max_memory_allocated() / 1e9
        frag_gb = mem_reserved_gb - mem_alloc_gb
        pbar.set_postfix(
            rows=num_rows,
            tok=num_valid_tokens,
            alloc=f"{mem_alloc_gb:.2f}",
            resv=f"{mem_reserved_gb:.2f}",
            frag=f"{frag_gb:.2f}",
        )
        # Persisted console line (tqdm.write prints above the bar without clobbering
        # it) so the rows/tokens/memory trend up to an OOM survives in scrollback.
        tqdm.write(
            f"[mem] step={step_idx} s={s} rows={num_rows} tok={num_valid_tokens} "
            f"alloc={mem_alloc_gb:.2f}GB resv={mem_reserved_gb:.2f}GB "
            f"frag={frag_gb:.2f}GB max_alloc={mem_max_alloc_gb:.2f}GB"
        )

        epoch = step_idx // max(1, len(train_loader))
        metrics = logger.log_training_metrics(loss, optimizer, update_rate)
        metrics.update(logger.log_moe_metrics(all_topk_indices, global_step, attention_mask))
        metrics["loss/entropy_floor"] = entropy_floor_value
        metrics["loss/reducible_gap"] = loss_normalizer_value
        metrics["token_superposition/s"] = s
        metrics["token_superposition/schedule_t"] = t
        metrics["token_superposition/active_labels"] = active_labels
        metrics[f"loss/s_{s}"] = loss_value
        metrics["mem/num_rows"] = num_rows
        metrics["mem/num_valid_tokens"] = num_valid_tokens
        metrics["mem/allocated_gb"] = mem_alloc_gb
        metrics["mem/reserved_gb"] = mem_reserved_gb
        metrics["mem/max_allocated_gb"] = mem_max_alloc_gb
        metrics["mem/fragmentation_gb"] = frag_gb

        detailed_logging = (optimizer_step_count % logger.detailed_frequency == 0)
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
    # The model is declared once as a named variant in model_variants.py -- the single
    # source of truth shared with inference.py. Only tokenizer-derived values
    # (vocab_size, pad_token_id) are filled in per run; everything architectural,
    # including Engram placement and whether TST is enabled, lives in the variant.
    #
    # Data sizing needs only the tokenizer-independent architecture constants
    # (sequence length x superposition group size), so read them from the variant built
    # with a placeholder vocab before the tokenizer is available.
    sizing = build_config(vocab_size=1, pad_token_id=0)
    data_max_length = sizing.sequence_length * sizing.superposition_max_size

    train_dataset, tokenizer, _ = load_and_preprocess_data(max_length=data_max_length)
    config = build_config(len(tokenizer), tokenizer.pad_token_id)

    model = MoEModel(config)

    count_parameters_layerwise(model)
    # The full headless_forward is compiled, including the nn.Embedding(sparse=True)
    # lookup. Measured (measure_sparse_compile.py): inductor keeps the embedding's COO
    # gradient sparse (nnz == touched rows) through a compiled fwd+bwd with zero graph
    # breaks (fullgraph=True passes), so GramReaperSparse still receives a sparse grad.
    model.headless_forward = torch.compile(model.headless_forward, dynamic=True)
    train(
        model,
        train_dataset,
        tokenizer,
        config.sequence_length,
        superposition_max_size=config.superposition_max_size,
    )

if __name__ == "__main__":
    main()
