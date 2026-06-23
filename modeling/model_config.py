import math
from dataclasses import dataclass, field

SUPERPOSITION_REFERENCE_SIZE = 1
SUPERPOSITION_TRAINING_MAX_SIZE = 32


@dataclass
class EngramSettings:
    """Architectural knobs for the Engram conditional-memory branches (DeepSeek-style
    hashed n-gram residual memory; see engram/EngramEmbeddingSparse.py).

    `layers` is the tuple of transformer layer indices that get a PRE-attention Engram
    branch (h <- h + alpha * V(E[ngram hash])); empty () => disabled (baseline). The
    remaining fields size the hashed n-gram tables. Model-derived values (vocab_size,
    d_model, per-layer seed, pad id) are filled in by the model, not here.
    """
    layers: tuple = ()
    orders: tuple = (2, 3)             # suffix n-gram orders
    n_heads: int = 4                   # hash heads per order (heads_total = orders*heads)
    rows_per_head: int = 16384         # rows/head target; a distinct prime is found >= this
    dim_per_head: int = 64             # per-head embedding width (heads_total*dim => engram_dim)
    alpha_init: float = 0.1            # LayerScale init on the memory residual
    importance_weighting: bool = False
    head_norm: bool = False            # shrink-only per-head unit-ball cap on the table lookup (norm<=1; zeros stay zero)
    learned_gate: bool = True          # train the LayerScale gate; False freezes alpha at alpha_init (no learned gating)

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

    # MoE -- fine-grained: 32 routed + 2 shared experts, top-2 routed.
    intermediate_size: int = 160
    n_experts: int = 34
    n_shared_experts: int = 2
    n_experts_per_token: int = 2
    n_routed_experts: int = field(init=False)

    # Traditional Attention
    n_attention_heads: int = 16
    n_key_value_heads: int = 4

    # Per-layer attention span (Gemma-style local/global interleave).
    #   None => full causal (global) attention for that layer.
    #   int  => causal sliding window of that many tokens (local attention).
    # Length must equal transformer_depth. The default is two local groups
    # (windows 256 then 512), each capped by a global mixing layer:
    #   [256, 256, 256, GLOBAL, 512, 512, 512, GLOBAL]
    attention_window_pattern: tuple = (256, 256, 256, None, 512, 512, 512, None)

    # QK-norm: RMS-normalize per-head query and key vectors (over head_dim) before
    # RoPE (paper reference arch). Init gain == 1, so identity at start of training.
    use_qk_norm: bool = True

    # HCA (Heavily Compressed Attention, DeepSeek-V4 §2.3.2)
    hca_block_size: int = 128       # m' -- non-overlapping compression stride
    hca_window_size: int = 128      # n_win -- uncompressed sliding-window length

    rms_norm_eps: float = 1e-6
    # 4k context. max_position_embeddings sizes the RoPE cos/sin cache, so it must be
    # >= sequence_length. rope_theta = 10000 keeps the slowest position-RoPE wavelength
    # (~2*pi*theta**(14/16) ~ 20k positions over the 16-dim position band) well above
    # 4096, so positions don't alias within context.
    max_position_embeddings: int = 4096
    sequence_length: int = 4096
    rope_theta: int = 10000
    do_rope: bool = True
    # Derived in __post_init__ to the paper's 1/sqrt(d) (Hägele et al. 2026): matrix
    # params init at std 1/sqrt(d) and embeddings are upscaled by sqrt(d) at lookup
    # (see MoEModel._embed_tokens) to give an RMS of 1 going into the model. Under
    # MuonMD this sets each matrix's frozen sphere radius to the paper's value.
    initializer_range: float = field(init=False)

    # Weight tying: share the input embedding matrix with the output classifier.
    # Both are [vocab_size, hidden_size], so the alias is exact. Saves ~vocab*
    # hidden params (a large fraction of the non-expert weights at this scale).
    # Untied here: a bog-standard separate embedding and unembedding. The input
    # embedding gets the paper's sqrt(d) upscale (safe only when untied) and the
    # output head is its own [vocab, hidden] matrix (see MoEModel).
    tie_word_embeddings: bool = False

    # Two-band rotation partition over head_dim: the first pos_rope_dims dims are
    # symmetric position-RoPE on Q+K (relative position); the remainder is NoPE.
    # pos_rope_dims must be even and <= head_dim. (The old asymmetric S-RoPE band
    # was removed; its dims were folded into pos_rope_dims so the total rotated
    # width is unchanged -- it is all partial position-RoPE now.)
    pos_rope_dims: int = 16

    # Token Superposition Training (TST). superposition_enabled is the single
    # switch: when True, s follows a two-phase schedule (s=32 for the first 10%
    # of training, then s=1 for the remaining 90%); when False, s=1 every step.
    # s drives the data packing and the MCCE loss only -- it is no longer encoded
    # into the activations (the S-RoPE band that did so has been removed).
    superposition_enabled: bool = False

    # Derived from superposition_enabled in __post_init__.
    superposition_max_size: int = field(init=False)

    # Engram conditional-memory branches (pre-attention, at selected layers). The
    # default is disabled (layers=()); the model is identical to baseline at init
    # because the n-gram table is sparse-gradient + zero-init.
    engram: EngramSettings = field(default_factory=EngramSettings)

    # Padding token ID -- used as the Engram suffix-n-gram left-pad. Set from the
    # tokenizer in main(); harmless at the default when Engram is disabled.
    pad_token_id: int = 0

    def __post_init__(self):
        if self.superposition_enabled:
            self.superposition_max_size = SUPERPOSITION_TRAINING_MAX_SIZE
        else:
            self.superposition_max_size = SUPERPOSITION_REFERENCE_SIZE

        self.n_routed_experts = self.n_experts - self.n_shared_experts

        # Paper init: std 1/sqrt(d) for every matrix parameter (d = model width).
        self.initializer_range = 1.0 / math.sqrt(self.hidden_size)

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
        if self.superposition_max_size <= 0 or (self.superposition_max_size & (self.superposition_max_size - 1)) != 0:
            raise ValueError("superposition_max_size must be a positive power of 2.")

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
