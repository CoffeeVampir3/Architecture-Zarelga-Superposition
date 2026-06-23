from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input
from typing import Optional
import torch
import torch.nn as nn
from einops import rearrange, repeat

from .liger_rope import LigerRopeFunction
from .model_config import ModelConfig
from .zRMSNorm import ZeroCenteredRMSNorm


# https://arxiv.org/abs/2505.06708
class GatedAttention(nn.Module):
    def __init__(self, config: ModelConfig, window_size: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.n_attention_heads
        self.num_key_value_heads = config.n_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.hidden_size // config.n_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.do_rope = config.do_rope
        self.initializer_range = config.initializer_range

        # Sliding-window local attention. None => full causal attention.
        # flash_attn window is (left, right); a causal local span of W tokens
        # (current token + W-1 previous) maps to (W-1, 0). (-1, -1) disables it.
        self.window_size = window_size
        self.window = (window_size - 1, 0) if window_size is not None else (-1, -1)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_attention_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.head_gate_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        # QK-norm: per-head RMSNorm over head_dim, applied to Q and K before RoPE.
        # Gain init == 1 (ZeroCenteredRMSNorm stores 1 + w with w init 0), so it is
        # identity at init; the gains are 1-D and train under Adam.
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = ZeroCenteredRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = ZeroCenteredRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.reset_parameters()

        # Two-band rotation partition over head_dim:
        #   pairs [0 .. pos_pairs)          : position-RoPE (symmetric Q+K)
        #   pairs [pos_pairs .. head_dim/2) : NoPE (identity on both)
        self.pos_rope_dims = config.pos_rope_dims
        self.pos_rope_pairs = config.pos_rope_dims // 2

        if self.do_rope:
            pos_cos, pos_sin = self._build_pos_rope_tables(
                self.max_position_embeddings,
                self.head_dim,
                self.rope_theta,
                dtype=torch.float32,
                device=self.q_proj.weight.device,
            )
        else:
            pos_cos = pos_sin = None
        self.register_buffer("pos_cos_cache", pos_cos, persistent=False)
        self.register_buffer("pos_sin_cache", pos_sin, persistent=False)

    def reset_parameters(self):
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.head_gate_proj.weight, mean=0.0, std=self.initializer_range)

    def _build_pos_rope_tables(self, max_position_embeddings, head_dim, base, dtype, device):
        """Position-RoPE cos/sin on the first `pos_rope_pairs` 2D pairs.

        Tables are shaped [max_pos, head_dim/2]; identity (cos=1, sin=0) outside
        the position band so the kernel's rotation on those pairs is a no-op.
        """
        half = head_dim // 2
        cos_table = torch.ones(max_position_embeddings, half, dtype=dtype, device=device)
        sin_table = torch.zeros(max_position_embeddings, half, dtype=dtype, device=device)
        if self.pos_rope_pairs == 0:
            return cos_table, sin_table

        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.pos_rope_dims, 2, device=device, dtype=torch.float32) / self.pos_rope_dims)
        )
        t = torch.arange(max_position_embeddings, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [max_pos, pos_pairs]
        cos_table[:, : self.pos_rope_pairs] = freqs.cos().to(dtype)
        sin_table[:, : self.pos_rope_pairs] = freqs.sin().to(dtype)
        return cos_table, sin_table

    def _flash_attention_varlen(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        unpad_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len = query_states.shape[:2]

        if cu_seqlens is None:
            # No mask: every token in the (bsz, seq_len) grid is valid. Pack
            # each batch row as its own sequence.
            query_states = rearrange(query_states, "b s h d -> (b s) h d")
            key_states = rearrange(key_states, "b s h d -> (b s) h d")
            value_states = rearrange(value_states, "b s h d -> (b s) h d")
            cu_seqlens = torch.arange(
                0,
                (bsz + 1) * seq_len,
                step=seq_len,
                dtype=torch.int32,
                device=query_states.device,
            )

            attn_output = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=seq_len,
                max_seqlen_k=seq_len,
                dropout_p=0.0,
                causal=True,
                window_size=self.window,
            )
            return rearrange(attn_output, "(b s) h d -> b s h d", b=bsz)

        query_states = index_first_axis(
            rearrange(query_states, "b s ... -> (b s) ..."),
            unpad_indices,
        )
        key_states = index_first_axis(
            rearrange(key_states, "b s ... -> (b s) ..."),
            unpad_indices,
        )
        value_states = index_first_axis(
            rearrange(value_states, "b s ... -> (b s) ..."),
            unpad_indices,
        )

        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            causal=True,
            window_size=self.window,
        )
        return pad_input(attn_output, unpad_indices, bsz, seq_len)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        unpad_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # In B S (H D)
        bsz, seq_len, _ = hidden_states.size()

        if self.do_rope and position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device)
            position_ids = repeat(position_ids, 'l -> b l', b=bsz)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = rearrange(query_states, "b s (h d) -> b s h d", h=self.num_heads, d=self.head_dim)
        key_states = rearrange(key_states, "b s (h d) -> b s h d", h=self.num_key_value_heads, d=self.head_dim)
        value_states = rearrange(value_states, "b s (h d) -> b s h d", h=self.num_key_value_heads, d=self.head_dim)

        # QK-norm over head_dim (per head), before RoPE.
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        if self.do_rope:
            # Symmetric position-RoPE: Q and K share the same cos/sin on the
            # position band; the rest of head_dim is NoPE (identity in the table).
            cos = self.pos_cos_cache[position_ids]   # [bsz, seq_len, head_dim/2]
            sin = self.pos_sin_cache[position_ids]
            query_states, key_states = LigerRopeFunction.apply(
                query_states,
                key_states,
                cos,
                sin,
                cos,
                sin,
            )

        attn_output = self._flash_attention_varlen(
            query_states,
            key_states,
            value_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            unpad_indices=unpad_indices,
        )

        gate_scores = torch.sigmoid(self.head_gate_proj(hidden_states))
        attn_output = attn_output * gate_scores.unsqueeze(-1)

        attn_output = rearrange(attn_output, "b s h d -> b s (h d)")
        return self.o_proj(attn_output)
