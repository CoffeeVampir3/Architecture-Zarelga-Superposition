import math
from dataclasses import dataclass, field


@dataclass
class EngramSettings:
    """Configuration for Engram conditional-memory branches.

    Gate and normalization defaults follow the ablation-accepted configuration
    (runs/engram_ablating/EXPERIMENT_REPORT.md): context gate + shrink-only row-norm
    cap; the annealed forward-noise curriculum was tested and discarded.
    """
    # Placement map: engram id -> transformer layers it's injected at (before the
    # block). One module (memory table + projections) per id; an id with several
    # layers is weight-tied across them — the token-addressed readout is shared and
    # only the gate is per-layer. Examples: {0: (0, 1, 2)} is one engram shared by
    # layers 0-2; {0: (2,), 1: (5,)} is two independent engrams at layers 2 and 5.
    # Ids double as hash seeds, so distinct ids hash token n-grams differently.
    layers: dict = field(default_factory=dict)
    orders: tuple = (2, 3)
    n_heads: int = 4
    rows_per_head: int = 16384
    dim_per_head: int = 64
    alpha_init: float = 0.1                 # used by the alpha gate modes only
    importance_weighting: bool = False
    head_norm: bool = False
    gate_mode: str = "context_gate"         # fixed_alpha | learned_per_channel_alpha | context_gate
    # Canonical-ID hashing: textually-equivalent tokens (case/accent/space-run
    # variants) share n-gram rows. The map is filled at model-build time from the
    # tokenizer (engram.apply_token_canon) and persists in checkpoints.
    tokenizer_compress: bool = True
    # Shrink-only L2 cap applied by the sparse optimizer to memory-table rows:
    # rows grow freely from zero-init (magnitude encodes confidence) but never
    # exceed the cap (bounded like the old unit_norm). 0 disables (unbounded).
    row_norm_cap: float = 1.0

    @property
    def enabled(self) -> bool:
        return len(self.layers) > 0

    def __post_init__(self):
        if not self.enabled:
            return
        if not isinstance(self.layers, dict):
            raise ValueError(
                "engram layers must be a dict mapping engram id -> layer indices, "
                f"e.g. {{0: (0, 1, 2)}}; got {self.layers!r}."
            )
        normalized = {}
        for eid, group in self.layers.items():
            if not isinstance(eid, int) or isinstance(eid, bool) or eid < 0:
                raise ValueError(f"engram ids must be non-negative ints; got {eid!r}.")
            group = tuple(group)
            if not group:
                raise ValueError(f"engram {eid} has an empty layer group.")
            if len(set(group)) != len(group):
                raise ValueError(f"engram {eid} lists a layer more than once: {group}.")
            normalized[eid] = group
        self.layers = normalized
        if self.n_heads <= 0:
            raise ValueError("engram n_heads must be positive.")
        if self.rows_per_head <= 0:
            raise ValueError("engram rows_per_head must be positive.")
        if self.dim_per_head <= 0:
            raise ValueError("engram dim_per_head must be positive.")
        if not self.orders or any(o < 2 for o in self.orders):
            raise ValueError("engram orders must be non-empty with each order >= 2.")
        if self.gate_mode not in ("fixed_alpha", "learned_per_channel_alpha", "context_gate"):
            raise ValueError(
                "engram gate_mode must be 'fixed_alpha', 'learned_per_channel_alpha', "
                f"or 'context_gate'; got {self.gate_mode!r}."
            )
        if self.row_norm_cap < 0.0:
            raise ValueError("engram row_norm_cap must be >= 0 (0 disables).")


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

    # Learnable per-head attention-sink logit (a virtual zero-value key that
    # only enlarges the softmax denominator). Gives heads a no-op escape when
    # nothing in context is relevant — matters most for SWA layers, where the
    # window evicts the early tokens a sink would otherwise form on naturally.
    use_attention_sink: bool = False

    hca_block_size: int = 128
    hca_window_size: int = 128

    # Attention Residuals (arXiv:2603.15031): replace the additive residual
    # stream with per-site softmax attention over depth-wise sources
    # [embedding, engram delta(s), block deltas]. The engram readout becomes a
    # depth-attention value source instead of a gated stream injection (its
    # context gate is not built), and site 0 stays embedding-only.
    attn_res: bool = False

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

        for eid, group in self.engram.layers.items():
            for idx in group:
                if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0 or idx >= self.transformer_depth:
                    raise ValueError(
                        f"engram {eid} layer entries must be ints in "
                        f"[0, {self.transformer_depth}); got {idx!r}."
                    )
