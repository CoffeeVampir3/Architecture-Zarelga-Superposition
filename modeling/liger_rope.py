"""Vendored Liger RoPE kernel, modified to accept separate cos/sin for Q and K.

The original Liger kernel applies one (cos, sin) to both Q and K. This fork
takes (q_cos, q_sin, k_cos, k_sin) so Q and K can in principle be rotated by
different angles within the same head_dim. The model currently uses it for the
symmetric two-band scheme (the same cos/sin is passed for Q and K):

  pairs [0 .. pos_pairs)    : symmetric position-RoPE
  pairs [pos_pairs .. hd/2) : NoPE (identity for both)

The separate-Q/K signature is retained for generality (e.g. a future asymmetric
band), but no asymmetric rotation is applied today.

All shape/stride conventions are unchanged from the upstream Liger kernel.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _triton_rope(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    q_cos,
    q_cos_row_stride,
    q_sin,
    q_sin_row_stride,
    k_cos,
    k_cos_row_stride,
    k_sin,
    k_sin_row_stride,
    sl,
    bs: tl.constexpr,
    cos_bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rot_pairs: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_rot: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BACKWARD_PASS: tl.constexpr = False,
):
    # q size: (bsz, seq_len, num_q_heads, head_dim)
    # k size: (bsz, seq_len, num_kv_heads, head_dim)
    # q_cos/q_sin/k_cos/k_sin size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
    pid = tl.program_id(0).to(tl.int64)

    q_ptr = q_ptr + pid * q_row_stride
    k_ptr = k_ptr + pid * k_row_stride

    batch_idx = pid // sl
    cos_row_idx = pid % sl

    q_cos = q_cos + tl.where(
        cos_bs == 1,
        cos_row_idx * q_cos_row_stride,
        batch_idx * (sl * q_cos_row_stride) + cos_row_idx * q_cos_row_stride,
    )
    q_sin = q_sin + tl.where(
        cos_bs == 1,
        cos_row_idx * q_sin_row_stride,
        batch_idx * (sl * q_sin_row_stride) + cos_row_idx * q_sin_row_stride,
    )
    k_cos = k_cos + tl.where(
        cos_bs == 1,
        cos_row_idx * k_cos_row_stride,
        batch_idx * (sl * k_cos_row_stride) + cos_row_idx * k_cos_row_stride,
    )
    k_sin = k_sin + tl.where(
        cos_bs == 1,
        cos_row_idx * k_sin_row_stride,
        batch_idx * (sl * k_sin_row_stride) + cos_row_idx * k_sin_row_stride,
    )

    cos_offsets = tl.arange(0, pad_rot)
    cos_mask = cos_offsets < rot_pairs
    q_cos_row = tl.load(q_cos + cos_offsets, mask=cos_mask, other=0)
    q_sin_row = tl.load(q_sin + cos_offsets, mask=cos_mask, other=0)
    k_cos_row = tl.load(k_cos + cos_offsets, mask=cos_mask, other=0)
    k_sin_row = tl.load(k_sin + cos_offsets, mask=cos_mask, other=0)

    # Only the rotated band: pairs [0, rot_pairs) with [hd/2, hd/2 + rot_pairs); NoPE dims untouched.
    first_half_q_offsets = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_rot)[None, :]
    first_half_k_offsets = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_rot)[None, :]
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (tl.arange(0, pad_rot)[None, :] < rot_pairs)
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (tl.arange(0, pad_rot)[None, :] < rot_pairs)
    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(q_sin_row.dtype)
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(k_sin_row.dtype)

    second_half_q_offsets = first_half_q_offsets + (hd // 2)
    second_half_k_offsets = first_half_k_offsets + (hd // 2)
    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=first_q_mask, other=0).to(q_sin_row.dtype)
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=first_k_mask, other=0).to(k_sin_row.dtype)

    if not BACKWARD_PASS:
        # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
        new_q_tile_1 = q_tile_1 * q_cos_row - q_tile_2 * q_sin_row
        tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
        new_q_tile_2 = q_tile_2 * q_cos_row + q_tile_1 * q_sin_row
        tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=first_q_mask)

        new_k_tile_1 = k_tile_1 * k_cos_row - k_tile_2 * k_sin_row
        tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
        new_k_tile_2 = k_tile_2 * k_cos_row + k_tile_1 * k_sin_row
        tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=first_k_mask)
    else:
        # dy = [dx1, dx2] * [cos, cos] + [-dx2, dx1] * [-sin, -sin]
        new_q_tile_1 = q_tile_1 * q_cos_row + q_tile_2 * q_sin_row
        tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
        new_q_tile_2 = q_tile_2 * q_cos_row - q_tile_1 * q_sin_row
        tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=first_q_mask)

        new_k_tile_1 = k_tile_1 * k_cos_row + k_tile_2 * k_sin_row
        tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
        new_k_tile_2 = k_tile_2 * k_cos_row - k_tile_1 * k_sin_row
        tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=first_k_mask)


def _launch(q, k, q_cos, q_sin, k_cos, k_sin, backward):
    batch_size, seq_len, n_q_head, head_dim = q.shape
    n_kv_head = k.shape[2]
    # Pairs to rotate == cos/sin band width; the rest of the head is NoPE (untouched).
    rot_pairs = q_cos.shape[-1]
    pad_rot = triton.next_power_of_2(rot_pairs)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)
    BLOCK_SIZE = max(pad_n_q_head, pad_n_kv_head)

    q = q.contiguous()
    k = k.contiguous()
    q_cos = q_cos.contiguous()
    q_sin = q_sin.contiguous()
    k_cos = k_cos.contiguous()
    k_sin = k_sin.contiguous()

    # q_cos and k_cos must share the same batch-dim shape for the cos_bs constexpr;
    # in practice both come from the same builder and are either [1, sl, hd] or [bs, sl, hd].
    cos_batch_size = q_cos.shape[0]
    assert k_cos.shape[0] == cos_batch_size, (
        f"q_cos and k_cos must share batch dim, got {q_cos.shape} vs {k_cos.shape}"
    )

    n_row = batch_size * seq_len
    _triton_rope[(n_row,)](
        q,
        q.stride(1),
        k,
        k.stride(1),
        q_cos,
        q_cos.stride(-2),
        q_sin,
        q_sin.stride(-2),
        k_cos,
        k_cos.stride(-2),
        k_sin,
        k_sin.stride(-2),
        seq_len,
        batch_size,
        cos_batch_size,
        n_q_head,
        n_kv_head,
        head_dim,
        rot_pairs,
        pad_n_q_head,
        pad_n_kv_head,
        pad_rot,
        BLOCK_SIZE=BLOCK_SIZE,
        BACKWARD_PASS=backward,
    )
    return q, k


def rope_forward(q, k, q_cos, q_sin, k_cos, k_sin):
    q, k = _launch(q, k, q_cos, q_sin, k_cos, k_sin, backward=False)
    return q, k, q_cos, q_sin, k_cos, k_sin


def rope_backward(dq, dk, q_cos, q_sin, k_cos, k_sin):
    dq, dk = _launch(dq, dk, q_cos, q_sin, k_cos, k_sin, backward=True)
    return dq, dk


class LigerRopeFunction(torch.autograd.Function):
    """Triton RoPE op with separate cos/sin for Q and K.

    For the symmetric case (standard RoPE), pass the same tensors for both Q
    and K. The signature also allows asymmetric rotations (different cos/sin
    per side), though the model does not use that path currently.
    """

    @staticmethod
    def forward(ctx, q, k, q_cos, q_sin, k_cos, k_sin):
        """
        q size: (bsz, seq_len, n_q_head, head_dim)
        k size: (bsz, seq_len, n_kv_head, head_dim)
        q_cos/q_sin/k_cos/k_sin: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        """
        q, k, q_cos, q_sin, k_cos, k_sin = rope_forward(q, k, q_cos, q_sin, k_cos, k_sin)
        ctx.save_for_backward(q_cos, q_sin, k_cos, k_sin)
        return q, k

    @staticmethod
    def backward(ctx, dq, dk):
        q_cos, q_sin, k_cos, k_sin = ctx.saved_tensors
        dq, dk = rope_backward(dq, dk, q_cos, q_sin, k_cos, k_sin)
        return dq, dk, None, None, None, None
