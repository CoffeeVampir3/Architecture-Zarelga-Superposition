import math
from dataclasses import dataclass, field


@dataclass
class EngramSettings:
    """Configuration for Engram conditional-memory branches."""
    layers: tuple = ()
    orders: tuple = (2, 3)
    n_heads: int = 4
    rows_per_head: int = 16384
    dim_per_head: int = 64
    alpha_init: float = 0.1
    importance_weighting: bool = False
    head_norm: bool = False
    learned_gate: bool = True

    @property
    def enabled(self) -> bool:
        return len(self.layers) > 0

    def __post_init__(self):
        if not self.enabled:
            return
        if self.n_heads <= 0:
            raise ValueError("engram n_heads must be positive.")
        if self.rows_per_head <= 0:
            raise ValueError("engram rows_per_head must be positive.")
        if self.dim_per_head <= 0:
            raise ValueError("engram dim_per_head must be positive.")
        if not self.orders or any(o < 2 for o in self.orders):
            raise ValueError("engram orders must be non-empty with each order >= 2.")


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    embed_size: int = 512
    hidden_size: int = 512

    transformer_depth: int = 8

    first_k_dense_replace: int = 0
    dense_intermediate_size: int = None

    intermediate_size: int = 160
    n_experts: int = 34
    n_shared_experts: int = 2
    n_experts_per_token: int = 2
    n_routed_experts: int = field(init=False)

    n_attention_heads: int = 16
    n_key_value_heads: int = 4

    # None means full causal attention for that layer.
    attention_window_pattern: tuple = (256, 256, 256, None, 512, 512, 512, None)

    use_qk_norm: bool = True

    hca_block_size: int = 128
    hca_window_size: int = 128

    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 4096
    sequence_length: int = 4096
    rope_theta: int = 10000
    do_rope: bool = True
    initializer_range: float = field(init=False)

    tie_word_embeddings: bool = False

    # Number of head_dim channels assigned to position RoPE.
    pos_rope_dims: int = 16

    engram: EngramSettings = field(default_factory=EngramSettings)

    pad_token_id: int = 0

    def __post_init__(self):
        self.n_routed_experts = self.n_experts - self.n_shared_experts

        if self.dense_intermediate_size is None:
            self.dense_intermediate_size = (
                (self.n_experts_per_token + self.n_shared_experts) * self.intermediate_size
            )

        self.initializer_range = 1.0 / math.sqrt(self.hidden_size)

        if self.embed_size != self.hidden_size:
            raise ValueError("embed_size and hidden_size must match unless an input/output projection is added.")
        if self.n_routed_experts <= 0:
            raise ValueError("n_experts must be greater than n_shared_experts.")
        if not (0 <= self.first_k_dense_replace <= self.transformer_depth):
            raise ValueError(
                f"first_k_dense_replace must be in [0, transformer_depth = "
                f"{self.transformer_depth}], got {self.first_k_dense_replace}."
            )
        if self.dense_intermediate_size <= 0:
            raise ValueError("dense_intermediate_size must be positive.")
        if self.n_experts_per_token <= 0:
            raise ValueError("n_experts_per_token must be positive.")
        if self.n_experts_per_token > self.n_routed_experts:
            raise ValueError("n_experts_per_token cannot exceed n_routed_experts.")
        if self.hidden_size % self.n_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by n_attention_heads.")
        if self.n_attention_heads % self.n_key_value_heads != 0:
            raise ValueError("n_attention_heads must be divisible by n_key_value_heads.")
        if len(self.attention_window_pattern) != self.transformer_depth:
            raise ValueError(
                f"attention_window_pattern must have transformer_depth = {self.transformer_depth} "
                f"entries, got {len(self.attention_window_pattern)}."
            )
        for w in self.attention_window_pattern:
            if w is not None and (not isinstance(w, int) or isinstance(w, bool) or w <= 0):
                raise ValueError(
                    f"attention_window_pattern entries must be None or a positive int, got {w!r}."
                )
        if self.sequence_length > self.max_position_embeddings:
            raise ValueError("sequence_length cannot exceed max_position_embeddings.")
        if self.hca_block_size <= 0:
            raise ValueError("hca_block_size must be positive.")
        if self.hca_window_size <= 0:
            raise ValueError("hca_window_size must be positive.")

        head_dim = self.hidden_size // self.n_attention_heads
        if self.pos_rope_dims < 0:
            raise ValueError("pos_rope_dims must be non-negative.")
        if self.pos_rope_dims % 2 != 0:
            raise ValueError("pos_rope_dims must be even (2D rotation pairs).")
        if self.pos_rope_dims > head_dim:
            raise ValueError(
                f"pos_rope_dims ({self.pos_rope_dims}) cannot exceed head_dim ({head_dim})."
            )

        for idx in self.engram.layers:
            if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0 or idx >= self.transformer_depth:
                raise ValueError(
                    f"engram.layers entries must be ints in [0, {self.transformer_depth}); got {idx!r}."
                )
