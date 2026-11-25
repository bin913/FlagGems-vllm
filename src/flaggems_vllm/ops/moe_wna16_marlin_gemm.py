# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Weight-only (WNA16) Marlin MoE GEMM -- generic Triton implementation.

Aligns with vLLM 0.32.0::

    vllm._custom_ops.moe_wna16_marlin_gemm
    torch.ops._moe_C.moe_wna16_marlin_gemm

``fused_marlin_moe`` calls this operator twice per MoE layer (gate/up and down
projection).  It is a single grouped GEMM over the padded, expert-sorted token
list; routing, activation and the final reduction stay with the caller.

Expected weight layout -- vLLM's Marlin repack (``gptq_marlin_repack`` /
``awq_marlin_repack``), *not* the plain GPTQ/AWQ layout used by
``flaggems_vllm.fused_marlin_moe``::

    b_qweight : (E, K // 16, N * 16 // pack) int32, pack = 32 // bits
    b_scales  : (E, num_groups, N)
    b_zeros   : (E, num_groups, N // pack) int32   (AWQ only)
    b_bias    : (E, N)                             (``marlin_permute_bias``)

``sorted_token_ids`` / ``expert_ids`` come from ``moe_align_block_size``:
entries are ``token * top_k + slot`` (values >= ``size_m * top_k`` mark
padding), one expert per ``moe_block_size`` block, and ``-1`` marks a block
owned by another expert-parallel rank.  The output row of a token slot is its
flattened ``(size_m, top_k)`` index.

Supported scope (all other combinations raise ``NotImplementedError``):
  * 16-bit activations (float16 / bfloat16) without ``a_scales``;
  * 4/8-bit integer weights, GPTQ (``uint4b8`` / ``uint8b128``, bias encoded in
    the type) and AWQ (``uint4`` / ``uint8`` with zero points);
  * any group size dividing ``K`` (vLLM uses -1/32/64/128);
  * optional bias, ``mul_topk_weights`` and ``use_atomic_add``;
  * no FP8/FP4 weights, no ``global_scale``, no float zero points.
"""

from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

_MARLIN_TILE = 16
_SUPPORTED_BITS = (4, 8)
_SUPPORTED_MOE_BLOCK_SIZES = (8, 16, 32, 48, 64)
_MIN_BLOCK_N = 64

# Tuned on H100; kept as a fallback so the op also runs on backends without a
# ``moe_wna16_marlin_gemm`` entry in their tune_configs.yaml.
_TUNE_CONFIGS = runtime.get_tuned_config("moe_wna16_marlin_gemm") or [
    triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 256}, num_warps=8, num_stages=3),
]


class _MoeWna16Params(NamedTuple):
    a: torch.Tensor
    c: torch.Tensor
    b_qweight: torch.Tensor
    b_bias: Optional[torch.Tensor]
    b_scales: torch.Tensor
    b_q_zeros: Optional[torch.Tensor]
    topk_weights: torch.Tensor
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_past_padded: torch.Tensor
    size_m: int
    size_n: int
    size_k: int
    top_k: int
    moe_block_size: int
    bits: int
    group_size: int
    grouped: bool
    has_zp: bool
    has_bias: bool
    mul_topk_weights: bool
    zp_offset: int
    block_m: int


def _scalar_attr(b_q_type, name: str) -> Optional[int]:
    """Read ``size_bits`` / ``bias`` from a vLLM ``ScalarType`` (or a mock)."""
    value = getattr(b_q_type, name, None)
    if callable(value):
        value = value()
    if value is None and name == "size_bits":
        raise NotImplementedError(
            "moe_wna16_marlin_gemm: b_q_type must expose size_bits()"
        )
    return value


def _prepare_params(
    input: torch.Tensor,
    output: Optional[torch.Tensor],
    b_qweight: torch.Tensor,
    b_bias: Optional[torch.Tensor],
    b_scales: torch.Tensor,
    a_scales: Optional[torch.Tensor],
    global_scale: Optional[torch.Tensor],
    b_q_zeros: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_past_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    moe_block_size: int,
    top_k: int,
    mul_topk_weights: bool,
    b_q_type,
    size_m: int,
    size_n: int,
    size_k: int,
    use_atomic_add: bool,
    is_zp_float: bool,
) -> _MoeWna16Params:
    if input.dim() != 2:
        raise ValueError("moe_wna16_marlin_gemm: input must be 2D")
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError(
            "moe_wna16_marlin_gemm: only 16-bit activations are supported, "
            f"got {input.dtype}"
        )
    if not input.is_contiguous():
        raise ValueError("moe_wna16_marlin_gemm: input must be contiguous")
    if a_scales is not None and a_scales.numel() > 0:
        raise NotImplementedError(
            "moe_wna16_marlin_gemm: 8-bit activations (a_scales) are not supported"
        )
    if global_scale is not None and global_scale.numel() > 0:
        raise NotImplementedError(
            "moe_wna16_marlin_gemm: NVFP4 global_scale is not supported"
        )
    if is_zp_float:
        raise NotImplementedError(
            "moe_wna16_marlin_gemm: float zero points are not supported"
        )
    if use_atomic_add:
        raise NotImplementedError(
            "moe_wna16_marlin_gemm: use_atomic_add=True is not supported; vLLM's "
            "Marlin MoE schedule uses use_atomic_add=False with a full reduction"
        )

    bits = int(_scalar_attr(b_q_type, "size_bits"))
    if bits not in _SUPPORTED_BITS:
        raise NotImplementedError(
            "moe_wna16_marlin_gemm: only 4/8-bit integer weights are supported, "
            f"got {bits} bits"
        )
    zp_offset = int(_scalar_attr(b_q_type, "bias") or 0)

    has_zp = b_q_zeros is not None and b_q_zeros.numel() > 0
    if has_zp and zp_offset != 0:
        raise NotImplementedError(
            "moe_wna16_marlin_gemm: biased weight types cannot carry zero points"
        )
    if not has_zp and zp_offset == 0 and bits in _SUPPORTED_BITS:
        # uint4 / uint8 without zero points is not a valid vLLM combination.
        raise ValueError(
            "moe_wna16_marlin_gemm: missing zero points for an unbiased "
            "quantization type"
        )

    if size_k % _MARLIN_TILE != 0:
        raise ValueError(
            f"moe_wna16_marlin_gemm: size_k ({size_k}) must be a multiple of "
            f"{_MARLIN_TILE}"
        )
    if size_n % _MIN_BLOCK_N != 0:
        raise ValueError(
            f"moe_wna16_marlin_gemm: size_n ({size_n}) must be a multiple of "
            f"{_MIN_BLOCK_N}"
        )
    if moe_block_size not in _SUPPORTED_MOE_BLOCK_SIZES:
        raise ValueError(
            "moe_wna16_marlin_gemm: moe_block_size must be one of "
            f"{_SUPPORTED_MOE_BLOCK_SIZES}, got {moe_block_size}"
        )

    if b_scales.dim() != 3 or b_scales.shape[2] != size_n:
        raise ValueError(
            "moe_wna16_marlin_gemm: b_scales must be (E, num_groups, size_n)"
        )
    num_groups = b_scales.shape[1]
    if num_groups > 1:
        if size_k % num_groups != 0:
            raise ValueError(
                f"moe_wna16_marlin_gemm: size_k ({size_k}) is not divisible by "
                f"num_groups ({num_groups})"
            )
        group_size = size_k // num_groups
        grouped = True
    else:
        group_size = -1
        grouped = False

    pack = 32 // bits
    if has_zp:
        if b_q_zeros.shape[-1] != size_n // pack:
            raise ValueError(
                "moe_wna16_marlin_gemm: b_q_zeros must be "
                "(E, num_groups, size_n // pack)"
            )
        b_q_zeros = b_q_zeros.contiguous()

    has_bias = b_bias is not None and b_bias.numel() > 0
    if has_bias:
        if b_bias.shape[-1] != size_n:
            raise ValueError("moe_wna16_marlin_gemm: b_bias must be (E, size_n)")
        b_bias = b_bias.contiguous()

    c_dtype = input.dtype
    if output is None:
        c = torch.empty((size_m * top_k, size_n), dtype=c_dtype, device=input.device)
    else:
        if output.shape != (size_m * top_k, size_n):
            raise ValueError(
                "moe_wna16_marlin_gemm: output must be (size_m * top_k, size_n)"
            )
        if output.dtype != c_dtype:
            raise ValueError("moe_wna16_marlin_gemm: output dtype must match input")
        if not output.is_contiguous():
            raise ValueError("moe_wna16_marlin_gemm: output must be contiguous")
        c = output

    block_m = max(16, triton.next_power_of_2(moe_block_size))

    return _MoeWna16Params(
        a=input,
        c=c,
        b_qweight=b_qweight,
        b_bias=b_bias if has_bias else None,
        b_scales=b_scales,
        b_q_zeros=b_q_zeros if has_zp else None,
        topk_weights=topk_weights,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_past_padded=num_tokens_past_padded,
        size_m=size_m,
        size_n=size_n,
        size_k=size_k,
        top_k=top_k,
        moe_block_size=moe_block_size,
        bits=bits,
        group_size=group_size,
        grouped=grouped,
        has_zp=has_zp,
        has_bias=has_bias,
        mul_topk_weights=bool(mul_topk_weights),
        zp_offset=zp_offset,
        block_m=block_m,
    )


@triton.jit
def _scale_index_single(offs_n):
    """Stored column of a channel-wise scale (``scale_perm_single``)."""
    n_in = offs_n % 32
    return (offs_n // 32) * 32 + 8 * ((n_in % 8) // 2) + 2 * (n_in // 8) + (n_in % 2)


@triton.jit
def _zp_index(offs_n, BITS: tl.constexpr):
    """Stored (word, shift, mask) of an AWQ zero point per logical column."""
    n_in = offs_n % 64
    p1 = (offs_n // 64) * 64 + 8 * (n_in % 8) + n_in // 8
    if BITS == 4:
        p2 = (p1 // 8) * 8 + (p1 % 8) // 2 + 4 * ((p1 % 8) % 2)
        return p2 // 8, (p2 % 8) * 4, 0xF
    else:
        p2 = (p1 // 4) * 4 + 2 * ((p1 % 4) % 2) + (p1 % 4) // 2
        return p2 // 4, (p2 % 4) * 8, 0xFF


@libentry()
@libtuner(
    configs=_TUNE_CONFIGS,
    key=["N", "K"],
)
@triton.jit
def moe_wna16_marlin_gemm_kernel(
    a_ptr,
    c_ptr,
    b_ptr,
    bias_ptr,
    scale_ptr,
    zp_ptr,
    topk_ptr,
    sorted_ptr,
    expert_ptr,
    num_post_ptr,
    num_valid_tokens,
    N,
    K,
    stride_am,
    stride_ak,
    stride_cm,
    stride_cn,
    stride_be,
    stride_bk,
    stride_bn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bze,
    stride_bzk,
    stride_bzn,
    GROUP_SIZE: tl.constexpr,
    GROUPED: tl.constexpr,
    BITS: tl.constexpr,
    HAS_ZP: tl.constexpr,
    ZP_OFFSET: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BIAS_STRIDE: tl.constexpr,
    MUL_TOPK: tl.constexpr,
    TOP_K: tl.constexpr,
    MOE_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One grouped WNA16 GEMM over the Marlin-repacked expert weights."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    num_post = tl.load(num_post_ptr)
    row0 = pid_m * MOE_BLOCK
    if row0 >= num_post:
        return
    expert = tl.load(expert_ptr + pid_m)
    if expert < 0:
        # Block owned by another expert-parallel rank.  vLLM's Marlin MoE
        # kernel skips these blocks (the rows are never produced on this rank),
        # so keep the caller's buffer untouched for parity.
        return

    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < MOE_BLOCK
    pos = row0 + offs_m
    tids = tl.load(
        sorted_ptr + pos,
        mask=m_mask & (pos < num_post),
        other=num_valid_tokens,
    )
    valid = (tids < num_valid_tokens) & m_mask
    a_row = tids // TOP_K

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    # Weight/scale/zero-point columns are independent of K and hoisted here.
    n_in = offs_n % 64
    chunk = offs_n // 64
    j_n = n_in // 16
    col = (n_in % 16) % 8
    bb = (n_in % 16) // 8
    if BITS == 4:
        word_base = chunk * 128 + 16 * col + j_n
    else:
        word_base = chunk * 256 + 32 * col + 2 * j_n + bb

    if GROUPED:
        scale_col = chunk * 64 + 8 * (n_in % 8) + n_in // 8
    else:
        scale_col = _scale_index_single(offs_n)

    if HAS_ZP:
        zp_word, zp_shift, zp_mask = _zp_index(offs_n, BITS)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_K)
    b_ptr += expert * stride_be
    scale_ptr += expert * stride_bse
    if HAS_ZP:
        zp_ptr += expert * stride_bze

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < K

        a = tl.load(
            a_ptr + a_row[:, None] * stride_am + k[None, :] * stride_ak,
            mask=valid[:, None] & k_mask[None, :],
            other=0.0,
        )

        k_in = k % 16
        row = k // 16
        g = (k_in // 2) % 4
        hi = k_in // 8
        parity = k_in % 2

        if BITS == 4:
            word_col = word_base[None, :] + 4 * g[:, None]
            shift = (4 * parity[:, None] + 2 * bb[None, :] + hi[:, None]) * 4
        else:
            word_col = word_base[None, :] + 8 * g[:, None]
            shift = (2 * parity[:, None] + hi[:, None]) * 8

        word = tl.load(
            b_ptr + row[:, None] * stride_bk + word_col * stride_bn,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0,
        )
        if BITS == 4:
            code = (word >> shift) & 0xF
        else:
            code = (word >> shift) & 0xFF

        if GROUPED:
            group = k // GROUP_SIZE
            scale = tl.load(
                scale_ptr
                + group[:, None] * stride_bsk
                + scale_col[None, :] * stride_bsn,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
        else:
            group = k * 0
            scale = tl.load(
                scale_ptr + scale_col * stride_bsn,
                mask=n_mask,
                other=0.0,
            )[None, :]
        scale = scale.to(tl.float32)

        if HAS_ZP:
            z = tl.load(
                zp_ptr + group[:, None] * stride_bzk + zp_word[None, :] * stride_bzn,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0,
            )
            z = ((z >> zp_shift[None, :]) & zp_mask).to(tl.float32)
            b = (code.to(tl.float32) - z) * scale
        else:
            b = (code.to(tl.float32) - ZP_OFFSET) * scale

        acc = tl.dot(a, b.to(a.dtype), acc=acc)

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + expert * BIAS_STRIDE + _scale_index_single(offs_n),
            mask=n_mask,
            other=0.0,
        )
        acc += bias.to(tl.float32)[None, :]
    if MUL_TOPK:
        weight = tl.load(topk_ptr + tids, mask=valid, other=0.0)
        acc *= weight[:, None]

    out = acc.to(c_ptr.dtype.element_ty)
    c_off = tids[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = valid[:, None] & n_mask[None, :]
    tl.store(c_ptr + c_off, out, mask=c_mask)


def _launch_generic(p: _MoeWna16Params) -> torch.Tensor:
    num_blocks = p.sorted_token_ids.numel() // p.moe_block_size
    if num_blocks == 0:
        return p.c

    # Placeholders for optional tensors; never dereferenced when the
    # corresponding HAS_* flag is false.
    bias = p.b_bias if p.b_bias is not None else p.b_scales
    zeros = p.b_q_zeros if p.b_q_zeros is not None else p.b_qweight

    grid = lambda meta: (num_blocks, triton.cdiv(p.size_n, meta["BLOCK_N"]))
    args = (
        p.a,
        p.c,
        p.b_qweight,
        bias,
        p.b_scales,
        zeros,
        p.topk_weights,
        p.sorted_token_ids,
        p.expert_ids,
        p.num_tokens_past_padded,
        p.size_m * p.top_k,
        p.size_n,
        p.size_k,
        p.a.stride(0),
        p.a.stride(1),
        p.c.stride(0),
        p.c.stride(1),
        p.b_qweight.stride(0),
        p.b_qweight.stride(1),
        p.b_qweight.stride(2),
        p.b_scales.stride(0),
        p.b_scales.stride(1),
        p.b_scales.stride(2),
        p.b_q_zeros.stride(0) if p.has_zp else 0,
        p.b_q_zeros.stride(1) if p.has_zp else 0,
        p.b_q_zeros.stride(2) if p.has_zp else 0,
    )
    consts = {
        "GROUP_SIZE": p.group_size,
        "GROUPED": p.grouped,
        "BITS": p.bits,
        "HAS_ZP": p.has_zp,
        "ZP_OFFSET": p.zp_offset,
        "HAS_BIAS": p.has_bias,
        "BIAS_STRIDE": p.b_bias.stride(0) if p.has_bias else 0,
        "MUL_TOPK": p.mul_topk_weights,
        "TOP_K": p.top_k,
        "MOE_BLOCK": p.moe_block_size,
        "BLOCK_M": p.block_m,
        "BLOCK_K": 16,
    }
    moe_wna16_marlin_gemm_kernel[grid](*args, **consts)
    return p.c


def moe_wna16_marlin_gemm(
    input: torch.Tensor,
    output: Optional[torch.Tensor],
    b_qweight: torch.Tensor,
    b_bias: Optional[torch.Tensor],
    b_scales: torch.Tensor,
    a_scales: Optional[torch.Tensor],
    global_scale: Optional[torch.Tensor],
    b_q_zeros: Optional[torch.Tensor],
    workspace: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_past_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    moe_block_size: int,
    top_k: int,
    mul_topk_weights: bool,
    b_q_type,
    size_m: int,
    size_n: int,
    size_k: int,
    use_atomic_add: bool,
    use_fp32_reduce: bool,
    is_zp_float: bool,
    thread_k: int = -1,
    thread_n: int = -1,
    blocks_per_sm: int = -1,
) -> torch.Tensor:
    """Grouped WNA16 Marlin MoE GEMM (vLLM 0.32.0 interface).

    ``workspace`` / ``thread_k`` / ``thread_n`` / ``blocks_per_sm`` /
    ``use_fp32_reduce`` describe vLLM's CUDA kernel's split-K schedule.  This
    implementation always traverses the full reduction dimension inside one
    program, so they are accepted for API compatibility and ignored.
    """
    del workspace, use_fp32_reduce, thread_k, thread_n, blocks_per_sm
    params = _prepare_params(
        input,
        output,
        b_qweight,
        b_bias,
        b_scales,
        a_scales,
        global_scale,
        b_q_zeros,
        sorted_token_ids,
        expert_ids,
        num_tokens_past_padded,
        topk_weights,
        moe_block_size,
        top_k,
        mul_topk_weights,
        b_q_type,
        size_m,
        size_n,
        size_k,
        use_atomic_add,
        is_zp_float,
    )
    if params.size_m == 0 or params.size_n == 0:
        return params.c
    return _launch_generic(params)


__all__ = ["moe_wna16_marlin_gemm"]
