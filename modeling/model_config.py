import math
from dataclasses import dataclass, field

SUPERPOSITION_REFERENCE_SIZE = 1
SUPERPOSITION_TRAINING_MAX_SIZE = 16
SUPERPOSITION_SCHEDULE_BETA = 2.0

# S-RoPE base frequencies (one per 2D pair on the S-band). Chosen so that
# log2(s) * omega for s in {1, 2, 4, 8, 16} gives well-separated angles on the
# torus with no aliasing -- (pi/3, pi/4) covers {0..4} cleanly on both pairs.
S_ROPE_DEFAULT_FREQS = (math.pi / 3, math.pi / 4)


def set_superposition(params, enabled=False):
    """Configure Token Superposition Training or the s=1 reference path."""
    params.superposition_max_size = (
        SUPERPOSITION_TRAINING_MAX_SIZE if enabled else SUPERPOSITION_REFERENCE_SIZE
    )
    params.superposition_schedule_beta = SUPERPOSITION_SCHEDULE_BETA
    return params


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    embed_size: int = 512
    hidden_size: int = 512

    transformer_depth: int = 5

    # MoE
    intermediate_size: int = 640
    n_experts: int = 8
    n_shared_experts: int = 2
    n_experts_per_token: int = 3
    n_routed_experts: int = field(init=False)

    # Traditional Attention
    n_attention_heads: int = 16
    n_key_value_heads: int = 4

    # HCA (Heavily Compressed Attention, DeepSeek-V4 §2.3.2)
    hca_block_size: int = 128       # m' -- non-overlapping compression stride
    hca_window_size: int = 128      # n_win -- uncompressed sliding-window length

    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 1024
    sequence_length: int = 256
    rope_theta: int = 1000
    do_rope: bool = True
    initializer_range: float = 0.02

    # Three-band rotation partition over head_dim. Sum must not exceed head_dim;
    # remainder is NoPE. pos_rope_dims is symmetric RoPE on Q+K (relative position).
    # s_rope_dims is asymmetric on K only (absolute log2(s) regime tag).
    pos_rope_dims: int = 12
    s_rope_dims: int = 4
    s_rope_freqs: tuple = S_ROPE_DEFAULT_FREQS

    # Token Superposition Training (TST). Per-step s is sampled from a
    # categorical over {1, 2, 4, ..., superposition_max_size} (powers of 2).
    # Logits are beta * (1 - 2t) * log2(s); t = step / total_steps. At t=0 the
    # distribution favors max_size, at t=1 it favors 1. Set max_size=1 to
    # disable superposition entirely.
    superposition_max_size: int = SUPERPOSITION_TRAINING_MAX_SIZE
    superposition_schedule_beta: float = SUPERPOSITION_SCHEDULE_BETA

    def __post_init__(self):
        self.n_routed_experts = self.n_experts - self.n_shared_experts

        if self.embed_size != self.hidden_size:
            raise ValueError("embed_size and hidden_size must match unless an input/output projection is added.")
        if self.n_routed_experts <= 0:
            raise ValueError("n_experts must be greater than n_shared_experts.")
        if self.n_experts_per_token <= 0:
            raise ValueError("n_experts_per_token must be positive.")
        if self.n_experts_per_token > self.n_routed_experts:
            raise ValueError("n_experts_per_token cannot exceed n_routed_experts.")
        if self.hidden_size % self.n_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by n_attention_heads.")
        if self.n_attention_heads % self.n_key_value_heads != 0:
            raise ValueError("n_attention_heads must be divisible by n_key_value_heads.")
        if self.sequence_length > self.max_position_embeddings:
            raise ValueError("sequence_length cannot exceed max_position_embeddings.")
        if self.hca_block_size <= 0:
            raise ValueError("hca_block_size must be positive.")
        if self.hca_window_size <= 0:
            raise ValueError("hca_window_size must be positive.")
        if self.superposition_max_size <= 0 or (self.superposition_max_size & (self.superposition_max_size - 1)) != 0:
            raise ValueError("superposition_max_size must be a positive power of 2.")
        if self.superposition_schedule_beta < 0.0:
            raise ValueError("superposition_schedule_beta must be non-negative.")

        head_dim = self.hidden_size // self.n_attention_heads
        if self.pos_rope_dims < 0 or self.s_rope_dims < 0:
            raise ValueError("pos_rope_dims and s_rope_dims must be non-negative.")
        if self.pos_rope_dims % 2 != 0 or self.s_rope_dims % 2 != 0:
            raise ValueError("pos_rope_dims and s_rope_dims must be even (2D rotation pairs).")
        if self.pos_rope_dims + self.s_rope_dims > head_dim:
            raise ValueError(
                f"pos_rope_dims + s_rope_dims ({self.pos_rope_dims + self.s_rope_dims}) "
                f"cannot exceed head_dim ({head_dim})."
            )
        if len(self.s_rope_freqs) * 2 != self.s_rope_dims:
            raise ValueError(
                f"s_rope_freqs must have s_rope_dims/2 = {self.s_rope_dims // 2} entries, "
                f"got {len(self.s_rope_freqs)}."
            )
