import torch
import torch.nn as nn
from aim import Distribution, Run
from safetensors.torch import save_file, load_file
from pathlib import Path
import time
from collections import defaultdict
import re

def count_parameters_layerwise(model):
    total_params = 0
    layer_params = {}

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        param_count = parameter.numel()
        layer_params[name] = param_count
        total_params += param_count

    print(f"\nModel Parameter Summary:")
    print("-" * 60)
    for name, count in layer_params.items():
        print(f"{name}: {count:,} parameters")
    print("-" * 60)
    print(f"Total Trainable Parameters: {total_params:,}\n")

    return total_params

def _trainer_state_path(filename):
    return Path(filename).with_suffix(".trainer.pt")

def save_checkpoint(model, optimizer, filename="checkpoint.safetensors", scheduler=None, global_step=None, epoch=None):
    base = model._orig_mod if hasattr(model, '_orig_mod') else model
    model_state = base.state_dict()

    # When the classifier is tied to the embedding, the two state-dict entries
    # share storage; safetensors refuses shared tensors. Drop the duplicate and
    # re-tie on load.
    if getattr(base, 'tie_word_embeddings', False):
        model_state.pop('output_layer.weight', None)

    save_file(model_state, filename)

    trainer_state = {"optimizer": optimizer.state_dict()}
    if scheduler is not None:
        trainer_state["scheduler"] = {
            "step_count": scheduler.step_count,
            "total_steps": scheduler.total_steps,
            "warmup_steps": scheduler.warmup_steps,
            "peak_lr": scheduler.peak_lr,
            "min_lr": scheduler.min_lr,
        }
    if global_step is not None:
        trainer_state["global_step"] = global_step
    if epoch is not None:
        trainer_state["epoch"] = epoch

    torch.save(trainer_state, _trainer_state_path(filename))

def load_checkpoint(model, optimizer, filename="checkpoint.safetensors", scheduler=None):
    model_state = load_file(filename)

    base = model._orig_mod if hasattr(model, '_orig_mod') else model
    tied = getattr(base, 'tie_word_embeddings', False)
    base.load_state_dict(model_state, strict=not tied)
    if tied:
        base.tie_weights()

    trainer_path = _trainer_state_path(filename)
    if not trainer_path.exists():
        return {}

    trainer_state = torch.load(trainer_path, map_location="cpu", weights_only=False)
    if "optimizer" in trainer_state:
        optimizer.load_state_dict(trainer_state["optimizer"])
    if scheduler is not None and "scheduler" in trainer_state:
        sched_state = trainer_state["scheduler"]
        scheduler.step_count = sched_state["step_count"]
        scheduler.total_steps = sched_state["total_steps"]
        scheduler.warmup_steps = sched_state["warmup_steps"]
        scheduler.peak_lr = sched_state["peak_lr"]
        scheduler.min_lr = sched_state["min_lr"]
    return {
        "global_step": trainer_state.get("global_step"),
        "epoch": trainer_state.get("epoch"),
    }

class AimLogger:
    def __init__(self, repo='logs/aim', experiment='moe_training', enable_detailed_logging=True, detailed_frequency=10):
        self.repo_path = Path(repo)
        self.repo_path.mkdir(parents=True, exist_ok=True)
        self.run = Run(repo=str(self.repo_path), experiment=experiment)
        self.enable_detailed_logging = enable_detailed_logging
        self.detailed_frequency = detailed_frequency
        self.start_time = time.time()
        self.moe_layers = []
        self.engrams = []

    def log_params(self, params):
        self.run["hparams"] = params

    def register_moe_layers(self, model):
        self.moe_layers = []
        for layer_idx, layer in enumerate(model.layers):
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
                self.moe_layers.append((layer_idx, layer.mlp.gate))

    def log_moe_metrics(self, all_topk_indices, step, attention_mask=None):
        if not self.moe_layers or not all_topk_indices:
            return {}

        metrics = {}
        valid_tokens = attention_mask.bool() if attention_mask is not None else None

        for (layer_idx, gate), topk_idx in zip(self.moe_layers, all_topk_indices):
            if valid_tokens is not None:
                topk_idx = topk_idx[valid_tokens]

            if topk_idx.numel() == 0:
                continue

            expert_counts = torch.bincount(topk_idx.flatten(), minlength=gate.n_routed_experts)
            total_tokens = topk_idx.numel()
            n_experts = gate.n_routed_experts

            experts_used = (expert_counts > 0).sum().item()
            pct_experts_used = 100.0 * experts_used / n_experts

            expected_load = total_tokens / n_experts
            load_balance = ((expert_counts.float() / expected_load - 1.0).abs().mean()).item()

            max_expert_pct = (expert_counts.max().float() * 100.0 / total_tokens).item()
            min_expert_pct = (expert_counts.min().float() * 100.0 / total_tokens).item()

            metrics[f'experts/layer_{layer_idx}_pct_experts_used'] = pct_experts_used
            metrics[f'experts/layer_{layer_idx}_load_imbalance'] = load_balance
            metrics[f'experts/layer_{layer_idx}_max_expert_pct'] = max_expert_pct
            metrics[f'experts/layer_{layer_idx}_min_expert_pct'] = min_expert_pct

        return metrics

    def register_engrams(self, model):
        """Snapshot (layer_idx, module) for each Engram branch so metrics can be
        attributed per layer. nn.ModuleDict is keyed by the string layer index."""
        self.engrams = []
        engrams = getattr(model, 'engrams', None)
        if engrams is None:
            return
        for key, engram in engrams.items():
            self.engrams.append((int(key), engram))

    def log_engram_metrics(self, step, detailed=False):
        """Per-Engram memory diagnostics.

        Cheap, every-step signals:
          * rows_activated      -- unique table rows touched this step (nnz of the COO
            grad). The table is addressed by hashed n-grams, so this is how much of the
            memory each step exercises. Read off engram.embedding.weight.grad, which
            still holds this backward's sparse grad until the next zero_grad.
          * rows_activated_frac -- the same as a fraction of total rows.
          * alpha_abs_mean      -- mean |LayerScale|, i.e. how hard the model gates the
            memory residual in (alpha->0 means the branch is being ignored).

        Detailed-cadence signals (full-table reductions / distributions):
          * grad_row_norm_mean  -- mean per-row grad norm of touched rows (the learning
            signal flowing into memory; the generic stats skip sparse grads).
          * row_norm / alpha    -- distributions (heavy tail => a few hot n-grams).

        Note: a cumulative-occupancy metric was removed -- the table is tiny relative to
        the n-gram traffic (one step's lookups oversubscribe each head's rows), so it
        saturates to ~1.0 within a few steps and can't discriminate table sizes.
        """
        if not self.engrams:
            return {}

        metrics = {}
        for layer_idx, engram in self.engrams:
            prefix = f'engram/layer_{layer_idx}'
            weight = engram.embedding.weight
            total_rows = weight.shape[0]

            grad = weight.grad
            if grad is not None and grad.is_sparse:
                coalesced = grad.coalesce()
                idx = coalesced.indices()[0]
                activated = idx.numel()
                metrics[f'{prefix}/rows_activated'] = activated
                metrics[f'{prefix}/rows_activated_frac'] = activated / total_rows
                if detailed and activated > 0:
                    g = coalesced.values()
                    metrics[f'{prefix}/grad_row_norm_mean'] = g.norm(dim=1).mean().item()

            alpha = engram.alpha.detach()
            metrics[f'{prefix}/alpha_abs_mean'] = alpha.abs().mean().item()

            # Importance weighting (a 2nd hash gating each head): the table is ones-init
            # (identity), so the signal is how far the touched weights have drifted from
            # 1.0 -- i.e. how much the model is up/down-weighting heads per n-gram.
            if detailed and getattr(engram, 'importance_weighting', False):
                imp_weight = engram.imp_table.weight
                imp_grad = imp_weight.grad
                if imp_grad is not None and imp_grad.is_sparse:
                    imp_idx = imp_grad.coalesce().indices()[0]
                    if imp_idx.numel() > 0:
                        vals = imp_weight.detach()[imp_idx].squeeze(-1)
                        metrics[f'{prefix}/imp_mean'] = vals.mean().item()
                        metrics[f'{prefix}/imp_abs_dev'] = (vals - 1.0).abs().mean().item()
                        metrics[f'{prefix}/imp'] = vals

            if detailed:
                row_norms = weight.detach().norm(dim=1)
                written = row_norms[row_norms > 0]
                if written.numel() > 0:
                    metrics[f'{prefix}/row_norm'] = written
                metrics[f'{prefix}/alpha'] = alpha

        return metrics

    def log_training_metrics(self, loss, optimizer, update_rate):
        metrics = {
            'loss/batch_loss': loss.item() if torch.is_tensor(loss) else loss,
            'training/learning_rate': optimizer.param_groups[0]['lr'],
            'training/update_rate': update_rate,
        }

        return metrics

    def _extract_layer_info(self, name):
        layer_match = re.search(r'layers\.(\d+)', name)
        layer_num = int(layer_match.group(1)) if layer_match else -1

        # Engram memory tables are named `engrams.<L>.embedding.weight`, so they'd
        # otherwise fall into the 'embed' bucket below and pollute the token
        # embedding's norm stats. They're intentionally not unit-norm constrained
        # (associative-memory slots, not a normalized vocab matrix), so their
        # unbounded growth is expected -- keep them on their own line.
        if 'engram' in name:
            layer_type = 'engram'
        elif 'mlp' in name or 'expert' in name:
            layer_type = 'mlp'
        elif 'attn' in name or 'self_attn' in name:
            layer_type = 'attention'
        elif 'embed' in name:
            layer_type = 'embedding'
        elif 'norm' in name:
            layer_type = 'norm'
        elif 'gate' in name:
            layer_type = 'gate'
        else:
            layer_type = 'other'

        return layer_type, layer_num

    def _track_scalar(self, name, value, step):
        self.run.track(value, name=name, step=step)

    def _track_distribution(self, name, samples, step):
        self.run.track(Distribution(samples, bin_count=64), name=name, step=step)

    def _track_value(self, name, value, step):
        if isinstance(value, (int, float)):
            self._track_scalar(name, value, step)
        elif isinstance(value, torch.Tensor):
            if value.numel() == 1:
                self._track_scalar(name, value.item(), step)
            elif value.numel() > 0:
                samples = value.detach().float().cpu().flatten().numpy()
                self._track_distribution(name, samples, step)
        elif isinstance(value, (list, tuple)) and all(isinstance(x, (int, float)) for x in value):
            self._track_distribution(name, value, step)

    def log(self, metrics, step=None, model=None, detailed_logging=False):
        for name, value in metrics.items():
            self._track_value(name, value, step)

        if model is not None and self.enable_detailed_logging:
            should_log_detailed = detailed_logging or (step is not None and step % self.detailed_frequency == 0)
            if should_log_detailed:
                self._log_model_stats(model, step)

    def _log_model_stats(self, model, step):
        grad_norm_sum = 0.0
        grad_mean_sum = 0.0
        grad_std_sum = 0.0
        param_norm_sum = 0.0
        param_mean_sum = 0.0
        param_std_sum = 0.0

        grad_count = 0
        param_count = 0
        max_grad_norm = float('-inf')
        min_grad_norm = float('inf')
        max_param_norm = float('-inf')
        min_param_norm = float('inf')

        grad_stats_by_type = defaultdict(lambda: {'sum': 0.0, 'count': 0})
        param_stats_by_type = defaultdict(lambda: {'sum': 0.0, 'count': 0})
        grad_stats_by_layer = defaultdict(lambda: {'norm_sum': 0.0, 'std_sum': 0.0, 'count': 0})
        param_stats_by_layer = defaultdict(lambda: {'norm_sum': 0.0, 'std_sum': 0.0, 'count': 0})

        problematic_grads = 0
        total_params = 0

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            layer_type, layer_num = self._extract_layer_info(name)
            total_params += 1

            param_norm = param.detach().norm().item()
            param_mean = param.detach().mean().item()
            param_std = param.detach().std().item()

            param_norm_sum += param_norm
            param_mean_sum += param_mean
            param_std_sum += param_std
            param_count += 1

            max_param_norm = max(max_param_norm, param_norm)
            min_param_norm = min(min_param_norm, param_norm)

            param_stats_by_type[layer_type]['sum'] += param_norm
            param_stats_by_type[layer_type]['count'] += 1

            if layer_num >= 0:
                param_stats_by_layer[layer_num]['norm_sum'] += param_norm
                param_stats_by_layer[layer_num]['std_sum'] += param_std
                param_stats_by_layer[layer_num]['count'] += 1

            # Skip sparse-gradient params (the nn.Embedding(sparse=True) table): the
            # dense reductions below aren't registered for the Sparse backend, and
            # coalescing + reducing would add host syncs to a diagnostic. The embedding
            # weight-norm stats above still log; only its grad stats are dropped.
            if param.grad is not None and not param.grad.is_sparse:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    problematic_grads += 1
                    continue

                grad_norm = param.grad.detach().norm().item()
                grad_mean = param.grad.detach().mean().item()
                grad_std = param.grad.detach().std().item()

                grad_norm_sum += grad_norm
                grad_mean_sum += grad_mean
                grad_std_sum += grad_std
                grad_count += 1

                max_grad_norm = max(max_grad_norm, grad_norm)
                min_grad_norm = min(min_grad_norm, grad_norm)

                grad_stats_by_type[layer_type]['sum'] += grad_norm
                grad_stats_by_type[layer_type]['count'] += 1

                if layer_num >= 0:
                    grad_stats_by_layer[layer_num]['norm_sum'] += grad_norm
                    grad_stats_by_layer[layer_num]['std_sum'] += grad_std
                    grad_stats_by_layer[layer_num]['count'] += 1

        if grad_count > 0:
            self._track_scalar("gradients/avg_norm", grad_norm_sum / grad_count, step)
            self._track_scalar("gradients/max_norm", max_grad_norm, step)
            self._track_scalar("gradients/min_norm", min_grad_norm, step)
            self._track_scalar("gradients/avg_mean", grad_mean_sum / grad_count, step)
            self._track_scalar("gradients/avg_std", grad_std_sum / grad_count, step)

        if param_count > 0:
            self._track_scalar("parameters/avg_norm", param_norm_sum / param_count, step)
            self._track_scalar("parameters/max_norm", max_param_norm, step)
            self._track_scalar("parameters/min_norm", min_param_norm, step)
            self._track_scalar("parameters/avg_mean", param_mean_sum / param_count, step)
            self._track_scalar("parameters/avg_std", param_std_sum / param_count, step)

        for layer_type, stats in grad_stats_by_type.items():
            if stats['count'] > 0:
                self._track_scalar(f"gradients/by_type/{layer_type}_avg_norm", stats['sum'] / stats['count'], step)

        for layer_type, stats in param_stats_by_type.items():
            if stats['count'] > 0:
                self._track_scalar(f"parameters/by_type/{layer_type}_avg_norm", stats['sum'] / stats['count'], step)

        for layer_num, stats in grad_stats_by_layer.items():
            if stats['count'] > 0:
                self._track_scalar(f"layer_stats/layer_{layer_num}/avg_grad_norm", stats['norm_sum'] / stats['count'], step)
                self._track_scalar(f"layer_stats/layer_{layer_num}/avg_grad_std", stats['std_sum'] / stats['count'], step)

        for layer_num, stats in param_stats_by_layer.items():
            if stats['count'] > 0:
                self._track_scalar(f"layer_stats/layer_{layer_num}/avg_param_norm", stats['norm_sum'] / stats['count'], step)
                self._track_scalar(f"layer_stats/layer_{layer_num}/avg_param_std", stats['std_sum'] / stats['count'], step)

        if total_params > 0:
            self._track_scalar("gradients/problematic_pct", 100.0 * problematic_grads / total_params, step)

    def close(self):
        self.run.close()
