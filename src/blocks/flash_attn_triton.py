import triton
import triton.language as tl
import torch


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Qi = tl.load(Q_block_ptr)

    offs_m = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    Oi_new = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE, 1), dtype=tl.float32)
    mi = tl.full((Q_TILE_SIZE, 1), float("-inf"), dtype=tl.float32)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):

        Kj = tl.load(
            K_block_ptr, boundary_check=(0, 1), padding_option="zero"
        )  # K_tile_size x D
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        Sij = tl.dot(Qi, tl.trans(Kj)) * scale  # Q_tile_size x K_tile_size

        if is_causal:
            offs_n = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = offs_m[:, None] >= offs_n[None, :]
            Sij = Sij + tl.where(mask, 0.0, -1.0e6)

        m_ij = tl.max(Sij, 1)
        mi_new = tl.maximum(mi, m_ij[:, None])  #

        mi_scale = tl.exp(mi - mi_new)
        Pij = tl.exp(Sij - mi_new)  # Q_tile_size x K_tile_size

        li_new = mi_scale * li + tl.sum(Pij, 1)[:, None]
        Oi_new = mi_scale * Oi_new + tl.dot(Pij.to(dtype=Vj.dtype), Vj)

        mi = mi_new
        li = li_new

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    Li = mi + tl.log(li)
    Oi_final = Oi_new / li

    # tl.device_print(li.type.element_ty)
    # tl.device_print(mi.type.element_ty)
    # tl.device_print(Oi_new.type.element_ty)

    tl.store(O_block_ptr, Oi_final)
    tl.store(L_block_ptr, tl.reshape(Li, (Q_TILE_SIZE,)))


class FlashAttnTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):

        bq = 16
        bk = 16

        batch_size = Q.size(0)
        Nq = Q.size(1)
        Nk = K.size(1)
        d = Q.size(2)

        Tq = Nq // bq
        # Tk = Nk // bk

        O = torch.empty_like(Q)
        L = torch.empty((batch_size, Nq), device=Q.device, dtype=torch.float32)

        # program_id(0) ranges: 0 … Tq-1
        # program_id(1) ranges: 0 … batch_size-1
        flash_fwd_kernel[(Tq, batch_size)](
            Q,
            K,
            V,
            O,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            Nq,
            Nk,
            1 / (d**0.5),
            D=d,
            Q_TILE_SIZE=bq,
            K_TILE_SIZE=bk,
            is_causal=is_causal,
        )

        ctx.save_for_backward(Q, K, V, O, L.squeeze(-1))
        return O, L
