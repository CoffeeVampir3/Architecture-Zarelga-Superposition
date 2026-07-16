from dataclasses import dataclass
from typing import Optional, Tuple

from modeling.model_config import ModelConfig, EngramSettings


# ---------------------------------------------------------------------------
# Model architecture variants
# ---------------------------------------------------------------------------

# Set to an int to pin the vocab size; leave None to use the tokenizer's.
VOCAB_SIZE: Optional[int] = None


def base_moe(vocab_size: int, pad_token_id: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,

        embed_size=512,
        hidden_size=512,
        transformer_depth=8,

        first_k_dense_replace=8,
        dense_intermediate_size=1792,
        intermediate_size=160,
        n_experts=34,
        n_shared_experts=2,
        n_experts_per_token=2,

        n_attention_heads=8,
        n_key_value_heads=4,
        attention_window_pattern=(256, 256, 256, None, 512, 512, 512, None),
        use_qk_norm=True,

        rms_norm_eps=1e-6,
        max_position_embeddings=8192,
        sequence_length=8192,
        rope_theta=100000,
        do_rope=True,
        tie_word_embeddings=False,
        pos_rope_dims=16,

        # Ablation-accepted configuration (runs/engram_ablating/EXPERIMENT_REPORT.md):
        # context gate + shrink-only row-norm cap, no forward-noise curriculum.
        # One weight-tied engram shared by the first sliding-window layers 0-2:
        # the token-addressed readout is computed once; each layer applies its
        # own context gate.
        engram=EngramSettings(
            layers={0: (0, 1, 2)},
            orders=(2, 3),
            n_heads=4,
            rows_per_head=65000,
            dim_per_head=64,
            importance_weighting=True,
            head_norm=True,
            gate_mode="context_gate",
            row_norm_cap=1.0,
            tokenizer_compress=True,
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
    if VOCAB_SIZE is not None:
        vocab_size = VOCAB_SIZE
    return factory(vocab_size, pad_token_id)


# ---------------------------------------------------------------------------
# Training run knobs
# ---------------------------------------------------------------------------

SQRT2 = 2 ** 0.5


@dataclass
class TrainingConfig:
    # --- schedule / data ---
    num_epochs: int = 1
    batch_size: int = 26
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
    norm_weight_decay: float = 0.1
    capture_warmup_steps: int = 5

    # --- LR schedule ---
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0

    # --- aux-loss-free expert balancing ---
    update_rate: float = 1e-3

    # --- z-loss (log-Z regularization on the LM head logits) ---
    # Penalizes mean(lse^2) over *valid* (non-ignored) tokens. Set to 0 to disable.
    z_loss_coef: float = 1e-4

    # --- checkpointing / logging ---
    checkpoint_interval_steps: int = 10_000
    max_rolling_checkpoints: int = 5
    log_interval: int = 10


DEFAULT_TRAINING = TrainingConfig()


def build_training_config(**overrides) -> TrainingConfig:
    """Instantiate the training knobs, optionally overriding individual fields."""
    return TrainingConfig(**overrides)
