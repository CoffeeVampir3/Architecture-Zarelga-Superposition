import flash_attn
from flash_attn import flash_attn_varlen_func
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_backward,
    _flash_attn_varlen_forward,
)
from typing import Optional
import torch
import torch.nn as nn
from einops import rearrange, repeat

from .liger_rope import LigerRopeFunction
from .model_config import ModelConfig
from .zRMSNorm import ZeroCenteredRMSNorm

# The sink path calls flash-attn's underscore-prefixed varlen kernels directly
# (the public wrapper drops the lse cotangent in backward, which would bias
# dq/dk). Their signatures are internal and have changed across releases; the
# call pattern below is validated against 2.7-2.8.
_FA_VERSION = tuple(int(p) for p in flash_attn.__version__.split(".")[:2])
_SINK_FA_COMPATIBLE = (2, 7) <= _FA_VERSION < (3, 0)


class _SinkFlashAttnVarlen(torch.autograd.Function):
    """Varlen flash attention with a learnable per-head attention-sink logit.

    The sink acts as one virtual key per query with logit `sink[h]` and a zero
    value vector: it takes softmax mass without contributing to the output,
    giving heads a no-op escape when nothing in the (windowed) context is
    relevant. Folding it in only requires the log-sum-exp the kernel already
    computes:

      forward:  lse_m = logaddexp(lse, sink);  out_m = out * exp(lse - lse_m)
      backward: the stock kernel rebuilds p_ij = exp(s_ij - lse) from whatever
                lse it is given, so passing (out_m, lse_m) makes it reconstruct
                exactly the sinked distribution. The sink's zero value keeps
                the softmax-Jacobian rowsum dout.out_m complete, so dq/dk/dv
                are exact, and dsink reduces to a closed form.

    Cost over the vanilla path: elementwise ops only — no extra attention.
    """

    @staticmethod
    def forward(ctx, q, k, v, sink, cu_seqlens, max_seqlen, window_left, softmax_scale):
        out, lse, _, _ = _flash_attn_varlen_forward(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
            0.0, softmax_scale, True, window_left, 0,
        )
        lse_m = torch.logaddexp(lse, sink.float()[:, None])  # [heads, total_q]
        out = out * rearrange(torch.exp(lse - lse_m), "h t -> t h 1").to(out.dtype)
        ctx.save_for_backward(q, k, v, sink, out, lse_m, cu_seqlens)
        ctx.max_seqlen = max_seqlen
        ctx.window_left = window_left
        ctx.softmax_scale = softmax_scale
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, sink, out, lse_m, cu_seqlens = ctx.saved_tensors
        dout = dout.contiguous()
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_varlen_backward(
            dout, q, k, v, out, lse_m, dq, dk, dv, cu_seqlens, cu_seqlens,
            ctx.max_seqlen, ctx.max_seqlen, 0.0, ctx.softmax_scale, True,
            ctx.window_left, 0, 0.0, None, False, None,
        )
        rowsum = (dout.float() * out.float()).sum(-1)  # [total_q, heads]
        dsink = -(torch.exp(sink.float()[:, None] - lse_m) * rowsum.T).sum(-1)
        return dq, dk, dv, dsink.to(sink.dtype), None, None, None, None


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

        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = ZeroCenteredRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = ZeroCenteredRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.use_attention_sink = config.use_attention_sink
        if self.use_attention_sink:
            if not _SINK_FA_COMPATIBLE:
                raise RuntimeError(
                    "use_attention_sink drives flash-attn internal varlen kernels whose "
                    f"signatures are only validated on 2.7-2.8; found {flash_attn.__version__}. "
                    "Re-validate _SinkFlashAttnVarlen against the new signatures before lifting "
                    "this check."
                )
            # Raw logit of the virtual zero-value sink key, one per query head.
            # Zero-init: the sink starts with the weight of one average key.
            self.attn_sink = nn.Parameter(torch.zeros(self.num_heads))
        else:
            self.attn_sink = None

        self.reset_parameters()

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
        if self.attn_sink is not None:
            nn.init.zeros_(self.attn_sink)

    def _build_pos_rope_tables(self, max_position_embeddings, head_dim, base, dtype, device):
        """Build RoPE tables shaped [max_pos, head_dim / 2]."""
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
    ) -> torch.Tensor:
        bsz, seq_len = query_states.shape[:2]

        if cu_seqlens is None:
            cu_seqlens = torch.arange(
                0,
                (bsz + 1) * seq_len,
                step=seq_len,
                dtype=torch.int32,
                device=query_states.device,
            )
            max_seqlen = seq_len

        q = rearrange(query_states, "b s h d -> (b s) h d")
        k = rearrange(key_states, "b s h d -> (b s) h d")
        v = rearrange(value_states, "b s h d -> (b s) h d")

        if self.attn_sink is not None:
            attn_output = _SinkFlashAttnVarlen.apply(
                q,
                k,
                v,
                self.attn_sink,
                cu_seqlens,
                max_seqlen,
                self.window[0],
                self.head_dim ** -0.5,
            )
        else:
            attn_output = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=0.0,
                causal=True,
                window_size=self.window,
            )
        return rearrange(attn_output, "(b s) h d -> b s h d", b=bsz)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
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

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        if self.do_rope and self.pos_rope_pairs > 0:
            # Pass only the position band (width pos_rope_pairs) to the kernel; NoPE dims untouched.
            cos = self.pos_cos_cache[:, : self.pos_rope_pairs][position_ids]  # [bsz, seq, pos_rope_pairs]
            sin = self.pos_sin_cache[:, : self.pos_rope_pairs][position_ids]
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
        )

        gate_scores = torch.sigmoid(self.head_gate_proj(hidden_states))
        attn_output = attn_output * gate_scores.unsqueeze(-1)

        attn_output = rearrange(attn_output, "b s h d -> b s (h d)")
        return self.o_proj(attn_output)
