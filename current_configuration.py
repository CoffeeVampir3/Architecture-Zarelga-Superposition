"""Singular place to configure a run.

Two halves live here:

* the **model** side — named ``ModelConfig`` variants describing the architecture
  (``build_config`` / ``VARIANTS``), and
* the **training** side — a ``TrainingConfig`` dataclass holding the run knobs
  (batch size, learning rates, schedule, checkpointing) that the trainer reads.

Keeping both here means there's one file to open to reconfigure a run instead of
hunting through the trainer for hardcoded defaults.
"""

from dataclasses import dataclass
from typing import Tuple

from modeling.model_config import ModelConfig, EngramSettings


# ---------------------------------------------------------------------------
# Model architecture variants
# ---------------------------------------------------------------------------

def base_moe(vocab_size: int, pad_token_id: int) -> ModelConfig:
    """Current model: explicit 8-layer fine-grained MoE architecture."""
    return ModelConfig(
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,

        embed_size=512,
        hidden_size=512,
        transformer_depth=8,

        first_k_dense_replace=3,
        dense_intermediate_size=1280,
        intermediate_size=160,
        n_experts=34,
        n_shared_experts=2,
        n_experts_per_token=2,

        n_attention_heads=8,
        n_key_value_heads=4,
        attention_window_pattern=(256, 256, 256, None, 512, 512, 512, None),
        use_qk_norm=True,

        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        sequence_length=4096,
        rope_theta=10000,
        do_rope=True,
        tie_word_embeddings=False,
        pos_rope_dims=16,

        engram=EngramSettings(
            layers=(2, 4),
            orders=(2, 3),
            n_heads=4,
            rows_per_head=65536,
            dim_per_head=64,
            alpha_init=0.1,
            importance_weighting=True,
            head_norm=True,
            learned_gate=False,
        ),
    )


VARIANTS = {
    "base_moe": base_moe,
}

DEFAULT_VARIANT = "base_moe"


def build_config(vocab_size: int, pad_token_id: int,
                 variant: str = DEFAULT_VARIANT) -> ModelConfig:
    """Instantiate a named variant's ModelConfig. Shared by the trainer and inference."""
    try:
        factory = VARIANTS[variant]
    except KeyError:
        raise KeyError(
            f"unknown model variant {variant!r}; known variants: {sorted(VARIANTS)}"
        ) from None
    return factory(vocab_size, pad_token_id)


# ---------------------------------------------------------------------------
# Training run knobs
# ---------------------------------------------------------------------------

SQRT2 = 2 ** 0.5


@dataclass
class TrainingConfig:
    """Tuning knobs for a training run.

    These are the "training facts" — everything that affects *how* the model is
    trained rather than *what* the model is. Edit here to reconfigure a run.
    """

    # --- schedule / data ---
    num_epochs: int = 1
    batch_size: int = 32
    num_workers: int = 4
    prefetch_factor: int = 2

    # --- learning rates (Muon / Adam paths) ---
    muon_lr: float = 0.02 * SQRT2
    adam_lr: float = 3e-4 * SQRT2
    gain_lr: float = 1e-3 * SQRT2
    embedding_lr: float = 3e-3 * SQRT2

    # --- optimizer hyperparameters ---
    muon_momentum: float = 0.95
    adam_betas: Tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-16
    embedding_beta: float = 0.9
    head_weight_decay: float = 0.1  # decoupled WD on the (untied) LM head; input embedding is unit-norm instead
    capture_warmup_steps: int = 5

    # --- LR schedule ---
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0

    # --- aux-loss-free expert balancing ---
    update_rate: float = 1e-3

    # --- checkpointing / logging ---
    checkpoint_interval_steps: int = 10_000
    max_rolling_checkpoints: int = 5
    log_interval: int = 10


DEFAULT_TRAINING = TrainingConfig()


def build_training_config(**overrides) -> TrainingConfig:
    """Instantiate the training knobs, optionally overriding individual fields."""
    return TrainingConfig(**overrides)
