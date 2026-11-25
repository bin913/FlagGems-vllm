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

"""Hopper (H100) specialization of the WNA16 Marlin MoE GEMM.

Same interface and semantics as
``flaggems_vllm.ops.moe_wna16_marlin_gemm``.

Why this shape of the kernel
----------------------------
vLLM's Marlin kernels decode weights inside the MMA pipeline, which a Triton
implementation can only reproduce with register-permutation primitives
(``tl.gather`` / ``tl.permute``).  Measured on H100, those layout conversions
dominate the runtime and land 10-20x behind Marlin, while a plain GEMM over
contiguous weights reaches 1.0-3.4x of Marlin on the same shapes.

Marlin itself pays a one-time repack on load (``gptq_marlin_repack``).  This
implementation mirrors that trade: the Marlin-packed weights are dequantized
once into a plain ``(E, K, N)`` activation-dtype buffer -- the scale and zero
point math is folded in, so the GEMM degenerates to a plain grouped MoE GEMM --
and the result is cached per weight tensor (``WeakTensorKeyDictionary``), so
steady-state decoding only runs the GEMM.  Routing, bias and
``mul_topk_weights`` behave exactly as in the generic implementation.

Cache behaviour:
  * keyed by the ``b_qweight`` tensor object, validated against the scale /
    zero-point tensors and the activation dtype, so a re-quantized or
    re-allocated weight never hits a stale entry;
  * bounded by ``FLAGGEMS_VLLM_MOE_WNA16_CACHE_BYTES`` (default 2 GiB, set 0 to
    disable).  Older entries are evicted FIFO to make room; a weight set larger
    than the whole budget is dequantized per call: still correct, just slower.

Launch behaviour: the config picked by the tuner is also compiled once and the
resulting ``CompiledKernel`` is cached under a signature of the routing / shapes
/ strides.  Steady-state calls therefore skip Triton's per-call Python dispatch
(measured 51.6us -> 24.5us of host time per call on H100, on par with the C++
baseline); any signature change falls back to the validated, autotuned path.

Supported scope is the generic implementation's: 16-bit activations, 4/8-bit
GPTQ/AWQ weights, any group size dividing K, optional bias.
"""

import os
from typing import Optional

import torch
import triton
import triton.language as tl
from torch.utils.weak import WeakTensorKeyDictionary

from flaggems_vllm import runtime
from flaggems_vllm.ops.moe_wna16_marlin_gemm import (
    _MoeWna16Params,
    _prepare_params,
    _scale_index_single,
    _zp_index,
)
from flaggems_vllm.utils import libentry, libtuner

_HOPPER_TUNE_CONFIGS = runtime.get_tuned_config("moe_wna16_marlin_gemm_hopper") or [
    triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    # Deeper pipeline / narrower K wins the weight-bandwidth-bound shapes.
    triton.Config({"BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
    # Fallback for K values that are not a multiple of 64/128.
    triton.Config({"BLOCK_N": 64, "BLOCK_K": 16}, num_warps=4, num_stages=3),
]


def _prune_block_k(configs, named_args, **kwargs):
    """Keep only tiles whose ``BLOCK_K`` divides K (no K mask in the inner loop)."""
    k = named_args.get("K")
    if k is None:
        return configs
    kept = [c for c in configs if k % c.kwargs.get("BLOCK_K", 16) == 0]
    return kept or configs


_CACHE_BYTES_LIMIT = int(os.environ.get("FLAGGEMS_VLLM_MOE_WNA16_CACHE_BYTES", 2 << 30))
# b_qweight -> {"weights", "bias", "scales", "zeros", "dtype", "bytes"}
_WEIGHT_CACHE = WeakTensorKeyDictionary()
_CACHE_BYTES = 0
# launch signature -> CompiledKernel plan, so steady-state decoding skips
# Triton's per-call Python dispatch (measured 20us -> 5us on H100).
_LAUNCH_CACHE = {}
_LAUNCH_CACHE_LIMIT = 64


@triton.jit
def _marlin_dequant_kernel(
    b_ptr,
    scale_ptr,
    zp_ptr,
    out_ptr,
    N,
    K,
    KP,
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
    BLOCK_N: tl.constexpr,
):
    """Dequantize one (expert, k) row of the Marlin-packed weights."""
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert = pid // K
    k = pid % K

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    n_in = offs_n % 64
    chunk = offs_n // 64
    j_n = n_in // 16
    col = (n_in % 16) % 8
    bb = (n_in % 16) // 8

    k_in = k % 16
    row = k // 16
    g = (k_in // 2) % 4
    hi = k_in // 8
    parity = k_in % 2

    if BITS == 4:
        word_col = chunk * 128 + 16 * col + 4 * g + j_n
        shift = (4 * parity + 2 * bb + hi) * 4
        code_mask: tl.constexpr = 0xF
    else:
        word_col = chunk * 256 + 32 * col + 2 * j_n + bb + 8 * g
        shift = (2 * parity + hi) * 8
        code_mask: tl.constexpr = 0xFF

    word = tl.load(
        b_ptr + expert * stride_be + row * stride_bk + word_col * stride_bn,
        mask=n_mask,
        other=0,
    )
    code = ((word >> shift) & code_mask).to(tl.float32)

    if GROUPED:
        group = k // GROUP_SIZE
        scale_col = chunk * 64 + 8 * (n_in % 8) + n_in // 8
        scale = tl.load(
            scale_ptr
            + expert * stride_bse
            + group * stride_bsk
            + scale_col * stride_bsn,
            mask=n_mask,
            other=0.0,
        )
    else:
        scale = tl.load(
            scale_ptr + expert * stride_bse + _scale_index_single(offs_n) * stride_bsn,
            mask=n_mask,
            other=0.0,
        )
    scale = scale.to(tl.float32)

    if HAS_ZP:
        zp_word, zp_shift, zp_mask = _zp_index(offs_n, BITS)
        if GROUPED:
            zp_off = (k // GROUP_SIZE) * stride_bzk
        else:
            zp_off = 0
        z = tl.load(
            zp_ptr + expert * stride_bze + zp_off + zp_word * stride_bzn,
            mask=n_mask,
            other=0,
        )
        z = ((z >> zp_shift) & zp_mask).to(tl.float32)
        value = (code - z) * scale
    else:
        value = (code - ZP_OFFSET) * scale

    tl.store(
        out_ptr + (expert * KP + k) * N + offs_n,
        value.to(out_ptr.dtype.element_ty),
        mask=n_mask,
    )


@triton.jit
def _marlin_bias_to_plain_kernel(
    bias_ptr,
    out_ptr,
    N,
    stride_be,
    BLOCK_N: tl.constexpr,
):
    """Undo ``marlin_permute_bias`` so the GEMM reads plain columns."""
    expert = tl.program_id(0)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    value = tl.load(
        bias_ptr + expert * stride_be + _scale_index_single(offs_n),
        mask=n_mask,
        other=0.0,
    )
    tl.store(out_ptr + expert * N + offs_n, value, mask=n_mask)


@libentry()
@libtuner(
    configs=_HOPPER_TUNE_CONFIGS,
    key=["N", "K", "num_blocks"],
    prune_configs_by={"early_config_prune": _prune_block_k},
)
@triton.jit
def hn_moe_wna16_plain_kernel(
    a_ptr,
    c_ptr,
    b_ptr,
    bias_ptr,
    topk_ptr,
    sorted_ptr,
    expert_ptr,
    num_post_ptr,
    num_valid_tokens,
    num_blocks,
    N,
    K,
    stride_am,
    stride_ak,
    stride_cm,
    stride_cn,
    stride_be,
    stride_bk,
    stride_bn,
    HAS_BIAS: tl.constexpr,
    MUL_TOPK: tl.constexpr,
    TOP_K: tl.constexpr,
    MOE_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MERGE: tl.constexpr,
    TP: tl.constexpr,
    NB_PAD: tl.constexpr,
):
    """Plain-dtype grouped MoE GEMM over dequantized expert weights.

    ``moe_align_block_size`` lays every expert's token blocks out contiguously,
    so several consecutive blocks usually share one expert's weights.  When
    ``MERGE`` is set the kernel collapses up to ``TP`` such blocks into one
    program: only blocks at an expert-aligned group offset run, and each of them
    computes ``min(TP * MOE_BLOCK, expert remainder)`` rows.  This divides the
    expert-weight traffic by the number of merged blocks while keeping every
    output row written exactly once.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    num_post = tl.load(num_post_ptr)
    row0 = pid_m * MOE_BLOCK
    if row0 >= num_post:
        return

    if MERGE:
        nb = tl.arange(0, NB_PAD)
        nb_mask = nb < num_blocks
        ids = tl.load(expert_ptr + nb, mask=nb_mask, other=-2)
        prev = tl.load(
            expert_ptr + tl.maximum(nb - 1, 0),
            mask=nb_mask & (nb >= 1),
            other=-2,
        )
        is_start = (nb == 0) | (ids != prev)
        run_start = tl.max(tl.where(is_start & nb_mask & (nb <= pid_m), nb, 0))
        if (pid_m - run_start) % TP != 0:
            return
        expert = tl.load(expert_ptr + pid_m)
        if expert < 0:
            return
        run_end = tl.max(tl.where((ids == expert) & nb_mask, nb, 0)) + 1
        nrows = tl.minimum((run_end - pid_m) * MOE_BLOCK, BLOCK_M)
    else:
        expert = tl.load(expert_ptr + pid_m)
        if expert < 0:
            # Non-local expert: leave the rows untouched (vLLM parity).
            return
        nrows = MOE_BLOCK

    offs_m = tl.arange(0, BLOCK_M)
    pos = row0 + offs_m
    m_mask = (offs_m < nrows) & (pos < num_post)
    tids = tl.load(sorted_ptr + pos, mask=m_mask, other=num_valid_tokens)
    valid = (tids < num_valid_tokens) & m_mask
    a_row = tids // TOP_K

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    b_ptr += expert * stride_be

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_K)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(
            a_ptr + a_row[:, None] * stride_am + k[None, :] * stride_ak,
            mask=valid[:, None],
            other=0.0,
        )
        b = tl.load(
            b_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=n_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(a, b, acc=acc)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + expert * N + offs_n, mask=n_mask, other=0.0)
        acc += bias.to(tl.float32)[None, :]
    if MUL_TOPK:
        weight = tl.load(topk_ptr + tids, mask=valid, other=0.0)
        acc *= weight[:, None]

    out = acc.to(c_ptr.dtype.element_ty)
    c_off = tids[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptr + c_off, out, mask=valid[:, None] & n_mask[None, :])


def _evict_until(need_bytes: int) -> None:
    """FIFO-evict cached experts until ``need_bytes`` more fit the budget."""
    global _CACHE_BYTES
    for key in list(_WEIGHT_CACHE.keys()):
        if _CACHE_BYTES + need_bytes <= _CACHE_BYTES_LIMIT:
            return
        try:
            dropped = _WEIGHT_CACHE.pop(key)
        except KeyError:
            continue
        _CACHE_BYTES -= dropped["bytes"]


def _dequantize_weights(p: _MoeWna16Params):
    """Build (or fetch) plain ``(E, K, N)`` weights plus an unpermuted bias."""
    global _CACHE_BYTES

    entry = _WEIGHT_CACHE.get(p.b_qweight)
    if (
        entry is not None
        and entry["scales"] is p.b_scales
        and entry["zeros"] is p.b_q_zeros
        and entry["bias"] is p.b_bias
        and entry["dtype"] is p.a.dtype
    ):
        return entry["weights"], entry["bias_plain"]

    num_experts = p.b_qweight.shape[0]
    # Pad K so the GEMM never needs a K mask on the weight tile; the extra rows
    # stay zero and contribute nothing to the reduction.
    k_pad = triton.cdiv(p.size_k, 128) * 128
    numel = num_experts * k_pad * p.size_n
    nbytes = numel * p.a.element_size()
    if _CACHE_BYTES_LIMIT <= 0 or nbytes > _CACHE_BYTES_LIMIT:
        cacheable = False
    else:
        _evict_until(nbytes)
        cacheable = True

    dev = p.a.device
    if k_pad == p.size_k:
        weights = torch.empty(
            (num_experts, k_pad, p.size_n), dtype=p.a.dtype, device=dev
        )
    else:
        # The tail rows must stay zero so the unmasked GEMM K loop ignores them.
        weights = torch.zeros(
            (num_experts, k_pad, p.size_n), dtype=p.a.dtype, device=dev
        )
    zeros = p.b_q_zeros if p.has_zp else p.b_qweight
    grid = (num_experts * p.size_k, triton.cdiv(p.size_n, 128))
    _marlin_dequant_kernel[grid](
        p.b_qweight,
        p.b_scales,
        zeros,
        weights,
        p.size_n,
        p.size_k,
        k_pad,
        p.b_qweight.stride(0),
        p.b_qweight.stride(1),
        p.b_qweight.stride(2),
        p.b_scales.stride(0),
        p.b_scales.stride(1),
        p.b_scales.stride(2),
        p.b_q_zeros.stride(0) if p.has_zp else 0,
        p.b_q_zeros.stride(1) if p.has_zp else 0,
        p.b_q_zeros.stride(2) if p.has_zp else 0,
        GROUP_SIZE=p.group_size,
        GROUPED=p.grouped,
        BITS=p.bits,
        HAS_ZP=p.has_zp,
        ZP_OFFSET=p.zp_offset,
        BLOCK_N=128,
    )

    bias_plain = None
    if p.has_bias:
        bias_plain = torch.empty((num_experts, p.size_n), dtype=p.a.dtype, device=dev)
        _marlin_bias_to_plain_kernel[(num_experts, triton.cdiv(p.size_n, 128))](
            p.b_bias,
            bias_plain,
            p.size_n,
            p.b_bias.stride(0),
            BLOCK_N=128,
        )

    if cacheable:
        if entry is not None:
            _CACHE_BYTES -= entry["bytes"]
        _WEIGHT_CACHE[p.b_qweight] = {
            "weights": weights,
            "bias_plain": bias_plain,
            "scales": p.b_scales,
            "zeros": p.b_q_zeros,
            "bias": p.b_bias,
            "dtype": p.a.dtype,
            "bytes": nbytes,
        }
        _CACHE_BYTES += nbytes
    return weights, bias_plain


def _merge_plan(
    num_blocks: int, num_experts: int, moe_block_size: int, size_m: int, top_k: int
):
    """How many consecutive same-expert blocks one program should cover.

    ``moe_align_block_size`` stores each expert's blocks contiguously, so the
    average number of blocks per expert (estimated from the routing row count)
    bounds the weight-traffic saving from merging.  The expert scan inside the
    kernel is only worth it for a small routing tensor, hence the cap.
    """
    if num_blocks > 256 or num_experts <= 0 or moe_block_size <= 0:
        return 1
    rows = max(1, size_m * top_k)
    per_expert = -(-rows // (num_experts * moe_block_size))
    if per_expert >= 3:
        return 4
    if per_expert >= 2:
        return 2
    return 1


def _aligned16(tensor: torch.Tensor) -> bool:
    return tensor.data_ptr() % 16 == 0


def _int_sig(value: int):
    """Triton's integer specialization: divisible by 16 / equal to one."""
    return (value % 16 == 0, value == 1)


def _launch_key(
    input,
    output,
    b_qweight,
    b_bias,
    b_scales,
    b_q_zeros,
    sorted_token_ids,
    expert_ids,
    num_tokens_past_padded,
    topk_weights,
    moe_block_size,
    top_k,
    mul_topk_weights,
    size_m,
    size_n,
    size_k,
    b_q_type,
    use_atomic_add,
    is_zp_float,
):
    """Cheap signature of every input the compiled launch depends on."""
    if use_atomic_add or is_zp_float:
        return None
    if input.dtype not in (torch.float16, torch.bfloat16):
        return None
    bits = getattr(b_q_type, "size_bits", None)
    if callable(bits):
        bits = bits()
    num_blocks = sorted_token_ids.numel() // moe_block_size
    return (
        input.dtype,
        bits,
        getattr(b_q_type, "bias", None),
        input.shape,
        input.stride(),
        _aligned16(input),
        None if output is None else (output.shape, output.stride(), _aligned16(output)),
        b_qweight.data_ptr(),
        b_qweight.shape,
        b_qweight.stride(),
        b_scales.data_ptr(),
        (
            None
            if b_bias is None or b_bias.numel() == 0
            else (b_bias.data_ptr(), b_bias.shape, _aligned16(b_bias))
        ),
        (
            None
            if b_q_zeros is None or b_q_zeros.numel() == 0
            else (b_q_zeros.data_ptr(), b_q_zeros.shape)
        ),
        topk_weights.data_ptr(),
        _aligned16(topk_weights),
        sorted_token_ids.data_ptr(),
        expert_ids.data_ptr(),
        _aligned16(expert_ids),
        num_tokens_past_padded.data_ptr(),
        num_blocks,
        _int_sig(num_blocks),
        moe_block_size,
        top_k,
        bool(mul_topk_weights),
        size_m,
        size_n,
        size_k,
        _int_sig(size_m * top_k),
        _int_sig(size_n),
        _int_sig(size_k),
    )


def _fast_launch(
    plan,
    input,
    output,
    sorted_token_ids,
    expert_ids,
    num_tokens_past_padded,
    topk_weights,
    size_m,
    top_k,
    size_n,
):
    """Launch the cached ``CompiledKernel`` without Triton's Python dispatch."""
    if output is None:
        c = torch.empty(
            (size_m * top_k, size_n), dtype=input.dtype, device=input.device
        )
    else:
        c = output
    weights = plan["weights"]
    full = (
        input,
        c,
        weights,
        plan["bias"],
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_past_padded,
        size_m * top_k,
        plan["num_blocks"],
        size_n,
        plan["size_k"],
        input.stride(0),
        input.stride(1),
        c.stride(0),
        c.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
    ) + plan["const_vals"]
    ck = plan["ck"]
    try:
        ck.run(
            plan["gx"],
            plan["gy"],
            1,
            torch.cuda.current_stream().cuda_stream,
            plan["function"],
            plan["packed"],
            None,
            None,
            None,
            *full,
        )
    except Exception:  # noqa: BLE001 - fall back to the public launch path
        ck[(plan["gx"], plan["gy"], 1)](*full)
    return c


def _store_launch_plan(p, key, weights, bias, num_blocks, tp, block_m, nb_pad):
    """Cache the compiled kernel of the config the tuner just picked."""
    try:
        cfg = hn_moe_wna16_plain_kernel.fn.best_config
    except Exception:  # noqa: BLE001 - tuning metadata is optional
        return
    if cfg is None:
        return
    blk_n = int(cfg.kwargs.get("BLOCK_N", 128))
    blk_k = int(cfg.kwargs.get("BLOCK_K", 64))
    const_vals = (
        p.has_bias,
        p.mul_topk_weights,
        p.top_k,
        p.moe_block_size,
        block_m,
        blk_n,
        blk_k,
        tp > 1,
        tp,
        nb_pad,
    )
    gx = num_blocks
    gy = triton.cdiv(p.size_n, blk_n)
    pos = (
        p.a,
        p.c,
        weights,
        bias,
        p.topk_weights,
        p.sorted_token_ids,
        p.expert_ids,
        p.num_tokens_past_padded,
        p.size_m * p.top_k,
        num_blocks,
        p.size_n,
        p.size_k,
        p.a.stride(0),
        p.a.stride(1),
        p.c.stride(0),
        p.c.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
    )
    raw = hn_moe_wna16_plain_kernel
    while not isinstance(raw, triton.runtime.JITFunction):
        raw = raw.fn
    kwargs = dict(
        HAS_BIAS=p.has_bias,
        MUL_TOPK=p.mul_topk_weights,
        TOP_K=p.top_k,
        MOE_BLOCK=p.moe_block_size,
        BLOCK_M=block_m,
        BLOCK_N=blk_n,
        BLOCK_K=blk_k,
        MERGE=tp > 1,
        TP=tp,
        NB_PAD=nb_pad,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    if getattr(cfg, "num_ctas", 1) != 1:
        kwargs["num_ctas"] = cfg.num_ctas
    try:
        ck = raw.warmup(*pos, grid=(gx, gy, 1), **kwargs)
    except Exception:  # noqa: BLE001 - keep the tuned path only
        return
    entry = _WEIGHT_CACHE.get(p.b_qweight)
    if entry is None:
        return
    if len(_LAUNCH_CACHE) >= _LAUNCH_CACHE_LIMIT:
        _LAUNCH_CACHE.clear()
    _LAUNCH_CACHE[key] = {
        "ck": ck,
        "function": ck.function,
        "packed": ck.packed_metadata,
        "gx": gx,
        "gy": gy,
        "num_blocks": num_blocks,
        "size_k": p.size_k,
        "weights": weights,
        "bias": bias,
        "const_vals": const_vals,
        "entry": entry,
    }


def _launch_hopper(p: _MoeWna16Params, key=None) -> torch.Tensor:
    num_blocks = p.sorted_token_ids.numel() // p.moe_block_size
    if num_blocks == 0:
        return p.c

    weights, bias_plain = _dequantize_weights(p)
    bias = bias_plain if bias_plain is not None else weights

    tp = _merge_plan(
        num_blocks,
        p.b_qweight.shape[0],
        p.moe_block_size,
        p.size_m,
        p.top_k,
    )
    block_m = max(16, triton.next_power_of_2(tp * p.moe_block_size))
    nb_pad = triton.next_power_of_2(num_blocks) if tp > 1 else 1

    grid = lambda meta: (num_blocks, triton.cdiv(p.size_n, meta["BLOCK_N"]))
    hn_moe_wna16_plain_kernel[grid](
        p.a,
        p.c,
        weights,
        bias,
        p.topk_weights,
        p.sorted_token_ids,
        p.expert_ids,
        p.num_tokens_past_padded,
        p.size_m * p.top_k,
        num_blocks,
        p.size_n,
        p.size_k,
        p.a.stride(0),
        p.a.stride(1),
        p.c.stride(0),
        p.c.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        HAS_BIAS=p.has_bias,
        MUL_TOPK=p.mul_topk_weights,
        TOP_K=p.top_k,
        MOE_BLOCK=p.moe_block_size,
        BLOCK_M=block_m,
        MERGE=tp > 1,
        TP=tp,
        NB_PAD=nb_pad,
    )
    if key is not None:
        _store_launch_plan(p, key, weights, bias, num_blocks, tp, block_m, nb_pad)
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
    """Hopper WNA16 Marlin MoE GEMM (vLLM 0.32.0 interface)."""
    del workspace, use_fp32_reduce, thread_k, thread_n, blocks_per_sm

    key = _launch_key(
        input,
        output,
        b_qweight,
        b_bias,
        b_scales,
        b_q_zeros,
        sorted_token_ids,
        expert_ids,
        num_tokens_past_padded,
        topk_weights,
        moe_block_size,
        top_k,
        mul_topk_weights,
        size_m,
        size_n,
        size_k,
        b_q_type,
        use_atomic_add,
        is_zp_float,
    )
    if key is not None:
        plan = _LAUNCH_CACHE.get(key)
        if plan is not None and _WEIGHT_CACHE.get(b_qweight) is not plan["entry"]:
            _LAUNCH_CACHE.pop(key, None)
            plan = None
        if plan is not None:
            return _fast_launch(
                plan,
                input,
                output,
                sorted_token_ids,
                expert_ids,
                num_tokens_past_padded,
                topk_weights,
                size_m,
                top_k,
                size_n,
            )

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
    return _launch_hopper(params, key)


__all__ = ["moe_wna16_marlin_gemm"]
