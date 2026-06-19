import math
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
    def __init__(self, config: ModelConfig):
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

        # Three-band rotation partition over head_dim:
        #   pairs [0 .. pos_pairs)                  : position-RoPE (symmetric Q+K)
        #   pairs [pos_pairs .. pos_pairs+s_pairs)  : S-RoPE (asymmetric, K only)
        #   pairs [pos_pairs+s_pairs .. head_dim/2) : NoPE (identity on both)
        self.pos_rope_dims = config.pos_rope_dims
        self.s_rope_dims = config.s_rope_dims
        self.pos_rope_pairs = config.pos_rope_dims // 2
        self.s_rope_pairs = config.s_rope_dims // 2
        self.s_rope_freqs = tuple(config.s_rope_freqs)
        # S-table indexed by log2(s); s_max determines table size.
        s_max = max(1, int(config.superposition_max_size))
        self.s_log2_max = max(0, int(round(math.log2(s_max))))
        self.s_table_size = self.s_log2_max + 1

        if self.do_rope:
            pos_cos, pos_sin = self._build_pos_rope_tables(
                self.max_position_embeddings,
                self.head_dim,
                self.rope_theta,
                dtype=torch.float32,
                device=self.q_proj.weight.device,
            )
            s_cos, s_sin = self._build_s_rope_tables(
                self.head_dim,
                dtype=torch.float32,
                device=self.q_proj.weight.device,
            )
        else:
            pos_cos = pos_sin = s_cos = s_sin = None
        self.register_buffer("pos_cos_cache", pos_cos, persistent=False)
        self.register_buffer("pos_sin_cache", pos_sin, persistent=False)
        self.register_buffer("s_cos_cache", s_cos, persistent=False)
        self.register_buffer("s_sin_cache", s_sin, persistent=False)

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

    def _build_s_rope_tables(self, head_dim, dtype, device):
        """S-RoPE cos/sin on the `s_rope_pairs` pairs after the position band.

        Indexed by log2(s) (rows: 0, 1, ..., s_log2_max). Identity outside the
        s band. These tables drive the K-side rotation only -- Q's S-band stays
        at identity (provided via the pos table, which is identity there).
        """
        half = head_dim // 2
        cos_table = torch.ones(self.s_table_size, half, dtype=dtype, device=device)
        sin_table = torch.zeros(self.s_table_size, half, dtype=dtype, device=device)
        if self.s_rope_pairs == 0:
            return cos_table, sin_table

        log_s = torch.arange(self.s_table_size, device=device, dtype=torch.float32)  # [s_table]
        omegas = torch.tensor(self.s_rope_freqs, device=device, dtype=torch.float32)  # [s_pairs]
        angles = log_s.unsqueeze(1) * omegas.unsqueeze(0)  # [s_table, s_pairs]
        start = self.pos_rope_pairs
        end = self.pos_rope_pairs + self.s_rope_pairs
        cos_table[:, start:end] = angles.cos().to(dtype)
        sin_table[:, start:end] = angles.sin().to(dtype)
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
        )
        return pad_input(attn_output, unpad_indices, bsz, seq_len)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        s_value: int = 1,
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
            # Pos band: same cos/sin for Q and K (symmetric rotation).
            # S band: identity on Q (already baked into pos_*_cache outside pos band),
            #         log2(s)-driven rotation on K (assembled from s_*_cache).
            q_cos = self.pos_cos_cache[position_ids]   # [bsz, seq_len, head_dim/2]
            q_sin = self.pos_sin_cache[position_ids]
            k_cos = q_cos.clone()
            k_sin = q_sin.clone()
            if self.s_rope_pairs > 0:
                log_s = int(round(math.log2(max(1, int(s_value)))))
                log_s = max(0, min(log_s, self.s_log2_max))
                start = self.pos_rope_pairs
                end = start + self.s_rope_pairs
                k_cos[..., start:end] = self.s_cos_cache[log_s, start:end]
                k_sin[..., start:end] = self.s_sin_cache[log_s, start:end]

            query_states, key_states = LigerRopeFunction.apply(
                query_states,
                key_states,
                q_cos,
                q_sin,
                k_cos,
                k_sin,
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
