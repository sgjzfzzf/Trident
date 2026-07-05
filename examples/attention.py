# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""
Fused Attention (forward-only example).

This implementation is adapted from the Triton flash-attention v2 tutorial and
the triton-tvm-ffi attention example, but wired to Trident's usage style.
"""

import math
import os

import torch
import triton
import triton.language as tl

import trident

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    desc_k,
    desc_v,
    offset_y,
    dtype: tl.constexpr,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr,
):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX

    offsetk_y = offset_y + lo
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo

    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]

        if dtype == tl.float8e5:
            v = desc_v.load([0, offsetv_y]).T
        else:
            v = desc_v.load([offsetv_y, 0])
        p = p.to(dtype)
        acc = tl.dot(p, v, acc)

        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N

    return acc, l_i, m_i


NUM_STAGES_OPTIONS = [2, 3, 4]
configs = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn},
        num_stages=s,
        num_warps=w,
    )
    for bm in [64, 128]
    for bn in [32, 64, 128]
    for s in NUM_STAGES_OPTIONS
    for w in [4, 8]
]

if "PYTEST_VERSION" in os.environ:
    configs = [
        triton.Config(
            dict(BLOCK_M=128, BLOCK_N=64),
            num_stages=2,
            num_warps=4,
        ),
    ]


def keep(conf):
    block_m = conf.kwargs["BLOCK_M"]
    block_n = conf.kwargs["BLOCK_N"]
    return not (
        torch.cuda.get_device_capability()[0] == 9
        and block_m * block_n < 128 * 128
        and conf.num_warps == 8
    )


def prune_invalid_configs(configs, named_args, **kwargs):
    del named_args
    n_ctx = kwargs["N_CTX"]
    return [conf for conf in configs if conf.kwargs.get("BLOCK_M", 0) <= n_ctx]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.autotune(
    configs=list(filter(keep, configs)),
    key=["N_CTX", "HEAD_DIM"],
    prune_configs_by={"early_config_prune": prune_invalid_configs},
)
@triton.jit
def _attn_fwd(
    sm_scale,
    M,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    dtype = tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(
        desc_q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    desc_v = _maybe_make_tensor_desc(
        desc_v,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_k = _maybe_make_tensor_desc(
        desc_k,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = _maybe_make_tensor_desc(
        desc_o,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale
    qk_scale *= 1.44269504
    q = desc_q.load([qo_offset_y, 0])

    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            desc_k,
            desc_v,
            offset_y,
            dtype,
            start_m,
            qk_scale,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,
        )
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            desc_k,
            desc_v,
            offset_y,
            dtype,
            start_m,
            qk_scale,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            2,
            offs_m,
            offs_n,
            N_CTX,
        )

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


def attention_forward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
) -> torch.Tensor:
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4, "expected [B, H, N, D] tensors"
    assert q.shape == k.shape == v.shape, "q, k, v must share shape"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()

    head_dim = q.shape[-1]
    assert head_dim in {16, 32, 64, 128, 256}

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    o = torch.empty_like(q)
    m = torch.empty(
        (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
    )
    stage = 3 if causal else 1

    def grid(meta):
        return (triton.cdiv(q.shape[2], meta["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

    _attn_fwd[grid](
        sm_scale,
        m,
        q.shape[0],
        q.shape[1],
        q,
        k,
        v,
        o,
        N_CTX=q.shape[2],
        HEAD_DIM=head_dim,
        STAGE=stage,
    )
    return o


def attn_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
) -> torch.Tensor:
    n_ctx = q.shape[2]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    mask = torch.tril(torch.ones((n_ctx, n_ctx), device=q.device, dtype=torch.bool))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p = torch.where(mask, p, torch.full_like(p, float("-inf")))
    p = torch.softmax(p.float(), dim=-1).to(q.dtype)
    return torch.matmul(p, v)


def attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
) -> torch.Tensor:
    return attention_forward_triton(q, k, v, causal=causal, sm_scale=sm_scale)


@trident.jit
def attention_jit(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
) -> torch.Tensor:
    # Torch-only fallback keeps trident.jit path compatible with torch._dynamo.
    return attn_torch(q, k, v, causal=causal, sm_scale=sm_scale)


if __name__ == "__main__":
    torch.manual_seed(20)

    b, h, n_ctx, head_dim = 1, 2, 128, 64
    dtype = torch.float16
    sm_scale = 0.5

    q = torch.empty((b, h, n_ctx, head_dim), dtype=dtype, device=DEVICE).normal_(
        mean=0.0, std=0.5
    )
    k = torch.empty((b, h, n_ctx, head_dim), dtype=dtype, device=DEVICE).normal_(
        mean=0.0, std=0.5
    )
    v = torch.empty((b, h, n_ctx, head_dim), dtype=dtype, device=DEVICE).normal_(
        mean=0.0, std=0.5
    )

    for causal in [False, True]:
        ref_out = attn_torch(q, k, v, causal=causal, sm_scale=sm_scale)
        tri_out = attention_triton(q, k, v, causal=causal, sm_scale=sm_scale)
        jit_out = attention_jit(
            q=q,
            k=k,
            v=v,
            causal=causal,
            sm_scale=sm_scale,
        )

        torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)
        torch.testing.assert_close(jit_out, ref_out, atol=1e-2, rtol=0)
