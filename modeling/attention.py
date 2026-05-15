from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input
from typing import Optional
import torch
import torch.nn as nn
from einops import rearrange, repeat

from .liger_rope import LigerRopeFunction
from .model_config import ModelConfig


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
        self.reset_parameters()

        if self.do_rope:
            cos_cached, sin_cached = self._compute_rope_embeddings(
                self.max_position_embeddings,
                self.head_dim,
                self.rope_theta,
                dtype=torch.float32,
                device=self.q_proj.weight.device,
            )
        else:
            cos_cached, sin_cached = None, None
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def reset_parameters(self):
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.head_gate_proj.weight, mean=0.0, std=self.initializer_range)

    def _compute_rope_embeddings(self, max_position_embeddings, head_dim, base=10000, dtype=None, device=None):
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
        t = torch.arange(max_position_embeddings, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        return cos.unsqueeze(0), sin.unsqueeze(0)

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

        if self.do_rope:
            # Slice off position specific rope freqs from the cached freqs.
            cos = self.cos_cached[:, position_ids]  # [1, bsz, seq_len, dim]
            sin = self.sin_cached[:, position_ids]  # [1, bsz, seq_len, dim]

            query_states, key_states = LigerRopeFunction.apply(
                query_states,
                key_states,
                cos.squeeze(0),
                sin.squeeze(0),
                position_ids
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
