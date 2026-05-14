"""HCA - Heavily Compressed Attention from the DeepSeek-V4 technical report.

Implements section 2.3.2 (equations 20-27) and section 2.3.3 of the V4 paper.
Each HCA layer attends from full-resolution queries to a single KV stream that
concatenates two sources:

  1. Heavily compressed KV entries -- one entry per non-overlapping block of
     m' raw tokens, produced by a learned softmax-weighted sum (eq 20-23).
  2. Sliding-window KV entries -- the last n_win raw tokens, uncompressed.

These are fed into ONE causal softmax (single denominator), not merged via
LSE or gated sum. An attention sink (eq 27) is implemented as a learned per
head K_sink with V=0: this adds exp(q . K_sink) to the softmax denominator
and contributes nothing to the output, matching the eq 27 z'_h logit (in a
slightly more expressive, data-dependent form).

Deviations from V4 worth knowing:
  * RoPE is applied over the FULL head_dim, matching this repo's existing
    GatedAttention. V4 uses partial-dim RoPE on the last 64 dims.
  * The attention-sink logit is data-dependent (q . K_sink) rather than the
    pure learned scalar z'_h. Strictly more expressive; reduces to V4 when
    K_sink is set to a fixed direction with a learnable scale.
  * K and V are tied (eq 26 CoreAttn(q, C^Comp, C^Comp)) -- a single shared
    KV head, MQA. The sliding-window entries reuse the same content
    projection W^KV (uncompressed), so all KV positions live in one space.

The varlen path runs a per-sequence loop in Python: each segment carved out
by cu_seqlens is forwarded through the dense path independently, then the
flat outputs are re-padded into the [B, S, D] grid. Correct but not fused.
A vectorized/Triton path is possible but not implemented here.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flash_attn.bert_padding import index_first_axis, pad_input

from .model_config import ModelConfig


class HCACompressor(nn.Module):
    """Eq 20-23: learned softmax-weighted sum over non-overlapping blocks of m' tokens.

    Returns both the uncompressed per-token content C (for use as sliding-window
    KV entries) and the compressed per-block content C_comp.
    """

    def __init__(
        self,
        hidden_size: int,
        compressed_dim: int,
        block_size: int,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.compressed_dim = compressed_dim
        self.block_size = block_size

        self.W_kv = nn.Linear(hidden_size, compressed_dim, bias=False)
        self.W_z = nn.Linear(hidden_size, compressed_dim, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(block_size, compressed_dim))

        nn.init.normal_(self.W_kv.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(self.W_z.weight, mean=0.0, std=initializer_range)
        # position_bias stays zero at init -> softmax is uniform -> initial compressor
        # behaves like (per-channel-weighted) mean pooling.

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        hidden_states: [B, N, D]
        returns:
          C        : [B, N, c]         uncompressed projected content
          C_comp   : [B, n_blocks, c]  one compressed entry per fully-formed block
          n_blocks = N // m' (trailing partial block is left to the sliding window)
        """
        B, N, _ = hidden_states.shape
        m = self.block_size

        C = self.W_kv(hidden_states)
        n_blocks = N // m
        if n_blocks == 0:
            C_comp = C.new_zeros(B, 0, self.compressed_dim)
            return C, C_comp

        Z = self.W_z(hidden_states[:, : n_blocks * m])
        C_blocks = C[:, : n_blocks * m].view(B, n_blocks, m, self.compressed_dim)
        Z_blocks = Z.view(B, n_blocks, m, self.compressed_dim)

        # eq 22: per-channel softmax over the m positions within each block
        weights = F.softmax(Z_blocks + self.position_bias, dim=2)
        # eq 23: weighted sum
        C_comp = (weights * C_blocks).sum(dim=2)
        return C, C_comp


class HeavilyCompressedAttention(nn.Module):
    """V4 HCA layer. Drop-in replacement for GatedAttention's forward signature."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.n_attention_heads
        self.head_dim = config.hidden_size // config.n_attention_heads
        # Per V4, the compressed KV serves as a single MQA head, so c = head_dim.
        self.compressed_dim = self.head_dim
        self.block_size = config.hca_block_size
        self.window_size = config.hca_window_size
        self.rope_theta = config.rope_theta
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by n_attention_heads")
        if self.block_size <= 0 or self.window_size <= 0:
            raise ValueError("hca_block_size and hca_window_size must be positive")

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self.compressor = HCACompressor(
            hidden_size=self.hidden_size,
            compressed_dim=self.compressed_dim,
            block_size=self.block_size,
            initializer_range=self.initializer_range,
        )
        # eq 27 sink: learnable per-head K with V=0. exp(q . K_sink) lands in the
        # softmax denominator; V=0 means it contributes nothing to the output.
        self.k_sink = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

        nn.init.normal_(self.q_proj.weight, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.k_sink, mean=0.0, std=self.initializer_range)

        cos_cached, sin_cached = self._compute_rope_embeddings(
            self.max_position_embeddings,
            self.head_dim,
            self.rope_theta,
            dtype=torch.float32,
            device=self.q_proj.weight.device,
        )
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    @staticmethod
    def _compute_rope_embeddings(
        max_position_embeddings, head_dim, base, dtype, device
    ):
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
                / head_dim
            )
        )
        t = torch.arange(max_position_embeddings, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def _apply_rope(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        """Apply HuggingFace-style RoPE. x: [B, N, H, D], position_ids: [B, N]."""
        cos = self.cos_cached[position_ids]  # [B, N, D]
        sin = self.sin_cached[position_ids]
        cos = cos.unsqueeze(2).to(x.dtype)  # [B, N, 1, D]
        sin = sin.unsqueeze(2).to(x.dtype)
        d = x.shape[-1]
        x1 = x[..., : d // 2]
        x2 = x[..., d // 2 :]
        cos1 = cos[..., : d // 2]
        sin1 = sin[..., : d // 2]
        rotated_1 = x1 * cos1 - x2 * sin1
        rotated_2 = x2 * cos1 + x1 * sin1
        return torch.cat([rotated_1, rotated_2], dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        unpad_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cu_seqlens is None:
            return self._forward_dense(hidden_states, position_ids)
        return self._forward_varlen(
            hidden_states, position_ids, cu_seqlens, unpad_indices
        )

    # ------------------------------------------------------------------
    # Dense path -- each batch row is one sequence; one attention call.
    # ------------------------------------------------------------------
    def _forward_dense(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim
        m = self.block_size
        n_win = self.window_size
        device = hidden_states.device

        if position_ids is None:
            position_ids = (
                torch.arange(N, device=device).unsqueeze(0).expand(B, N)
            )

        # ---- queries ----
        q = self.q_proj(hidden_states)
        q = rearrange(q, "b n (h d) -> b n h d", h=H, d=Dh)
        q = self._apply_rope(q, position_ids)

        # ---- compressed + uncompressed KV content ----
        C, C_comp = self.compressor(hidden_states)
        n_blocks = C_comp.shape[1]

        # Each compressed entry's representative position = last token of its block.
        # Causality: query at position t can see compressed entry i iff t > rep_pos[i].
        comp_positions = (
            torch.arange(n_blocks, device=device) * m + (m - 1)
        )  # [n_blocks]

        K_comp = C_comp.unsqueeze(2)  # MQA single KV head
        V_comp = C_comp.unsqueeze(2)
        K_win = C.unsqueeze(2)
        V_win = C.unsqueeze(2)

        if n_blocks > 0:
            comp_positions_b = comp_positions.unsqueeze(0).expand(B, -1)
            K_comp = self._apply_rope(K_comp, comp_positions_b)
        K_win = self._apply_rope(K_win, position_ids)

        K = torch.cat([K_comp, K_win], dim=1)
        V = torch.cat([V_comp, V_win], dim=1)
        K = K.expand(-1, -1, H, -1)
        V = V.expand(-1, -1, H, -1)

        k_sink = self.k_sink.view(1, 1, H, Dh).expand(B, 1, H, Dh)
        v_sink = torch.zeros_like(k_sink)
        K = torch.cat([K, k_sink], dim=1)
        V = torch.cat([V, v_sink], dim=1)

        # ---- mask: [B, 1, N, KV_len] (broadcast over heads) ----
        if n_blocks > 0:
            comp_vis = position_ids.unsqueeze(-1) > comp_positions.view(1, 1, -1)
        else:
            comp_vis = torch.zeros(B, N, 0, dtype=torch.bool, device=device)
        win_pos = position_ids.unsqueeze(1)
        q_pos = position_ids.unsqueeze(-1)
        win_vis = (win_pos <= q_pos) & (win_pos > q_pos - n_win)
        sink_vis = torch.ones(B, N, 1, dtype=torch.bool, device=device)
        mask = torch.cat([comp_vis, win_vis, sink_vis], dim=-1).unsqueeze(1)

        q_sdpa = rearrange(q, "b n h d -> b h n d")
        k_sdpa = rearrange(K, "b n h d -> b h n d")
        v_sdpa = rearrange(V, "b n h d -> b h n d")
        out = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
        )
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.o_proj(out)

    # ------------------------------------------------------------------
    # Varlen path -- packed sequences with cu_seqlens / unpad_indices.
    # Each segment is forwarded through the dense path independently.
    # ------------------------------------------------------------------
    def _forward_varlen(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        cu_seqlens: torch.Tensor,
        unpad_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape
        device = hidden_states.device

        if unpad_indices is None:
            raise ValueError(
                "unpad_indices must be provided alongside cu_seqlens in the varlen path."
            )

        # Unpad to flat [total_valid, ...]
        flat_hidden = index_first_axis(
            rearrange(hidden_states, "b s d -> (b s) d"), unpad_indices
        )

        if position_ids is None:
            # Synthesize per-sequence positions from cu_seqlens.
            seglens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            flat_pos = torch.cat(
                [torch.arange(L, device=device, dtype=torch.long) for L in seglens],
                dim=0,
            )
        else:
            flat_pos = index_first_axis(
                rearrange(position_ids.to(torch.long).unsqueeze(-1), "b s d -> (b s) d"),
                unpad_indices,
            ).squeeze(-1)

        outputs = []
        cu = cu_seqlens.tolist()
        for s_idx in range(len(cu) - 1):
            start, end = cu[s_idx], cu[s_idx + 1]
            if end == start:
                continue
            seq_h = flat_hidden[start:end].unsqueeze(0)        # [1, L, D]
            seq_pos = flat_pos[start:end].unsqueeze(0)         # [1, L]
            seq_out = self._forward_dense(seq_h, seq_pos)
            outputs.append(seq_out.squeeze(0))

        flat_out = torch.cat(outputs, dim=0)                   # [total_valid, hidden]
        return pad_input(flat_out, unpad_indices, B, S)
