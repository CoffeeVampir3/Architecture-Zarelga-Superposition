import math
from dataclasses import dataclass, field

SUPERPOSITION_REFERENCE_SIZE = 1
SUPERPOSITION_TRAINING_MAX_SIZE = 8

# S-RoPE base frequencies used only when Token Superposition Training is
# explicitly enabled. The default run keeps the S band disabled.
S_ROPE_TRAINING_FREQS = (math.pi / 3, math.pi / 4)
S_ROPE_DEFAULT_FREQS = ()


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    embed_size: int = 512
    hidden_size: int = 512

    transformer_depth: int = 5

    # MoE -- fine-grained: 32 routed + 2 shared experts, top-4 routed.
    # ~17.6% per-token activation; expert width chosen to hold total FFN
    # parameter count roughly constant (multiple of 32 for kernel-friendly
    # grouped GEMMs).
    intermediate_size: int = 160
    n_experts: int = 34
    n_shared_experts: int = 2
    n_experts_per_token: int = 4
    n_routed_experts: int = field(init=False)

    # Traditional Attention
    n_attention_heads: int = 16
    n_key_value_heads: int = 4

    # QK-norm: RMS-normalize per-head query and key vectors (over head_dim) before
    # RoPE (paper reference arch). Init gain == 1, so identity at start of training.
    use_qk_norm: bool = True

    # HCA (Heavily Compressed Attention, DeepSeek-V4 §2.3.2)
    hca_block_size: int = 128       # m' -- non-overlapping compression stride
    hca_window_size: int = 128      # n_win -- uncompressed sliding-window length

    rms_norm_eps: float = 1e-6
    # 4k context. max_position_embeddings sizes the RoPE cos/sin cache, so it must be
    # >= sequence_length. rope_theta = 10000 keeps the slowest position-RoPE wavelength
    # (~2*pi*theta**(10/12) ~ 13.5k positions over the 12-dim position band) well above
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
    tie_word_embeddings: bool = True

    # Three-band rotation partition over head_dim. Sum must not exceed head_dim;
    # remainder is NoPE. pos_rope_dims is symmetric RoPE on Q+K (relative position).
    # s_rope_dims is asymmetric on K only (absolute log2(s) regime tag); it is
    # derived from superposition_enabled below.
    pos_rope_dims: int = 12

    # Token Superposition Training (TST). superposition_enabled is the single
    # switch: when True, s follows a two-phase schedule (s=8 for the first 40%
    # of training, then s=1 for the remaining 60%), and an S-RoPE band is added
    # on K. When False, s=1 every step and S-RoPE is off.
    superposition_enabled: bool = False

    # Derived from superposition_enabled in __post_init__.
    superposition_max_size: int = field(init=False)
    s_rope_dims: int = field(init=False)
    s_rope_freqs: tuple = field(init=False)

    def __post_init__(self):
        if self.superposition_enabled:
            self.superposition_max_size = SUPERPOSITION_TRAINING_MAX_SIZE
            self.s_rope_dims = len(S_ROPE_TRAINING_FREQS) * 2
            self.s_rope_freqs = S_ROPE_TRAINING_FREQS
        else:
            self.superposition_max_size = SUPERPOSITION_REFERENCE_SIZE
            self.s_rope_dims = 0
            self.s_rope_freqs = ()

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
        if self.sequence_length > self.max_position_embeddings:
            raise ValueError("sequence_length cannot exceed max_position_embeddings.")
        if self.hca_block_size <= 0:
            raise ValueError("hca_block_size must be positive.")
        if self.hca_window_size <= 0:
            raise ValueError("hca_window_size must be positive.")
        if self.superposition_max_size <= 0 or (self.superposition_max_size & (self.superposition_max_size - 1)) != 0:
            raise ValueError("superposition_max_size must be a positive power of 2.")

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
