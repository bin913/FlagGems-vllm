# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Hopper (SM90) `fused_norm_rope` for DeepSeek-V3.2 (DSA) layers.

Same vLLM 0.32 interface and side effects as
``flaggems_vllm.ops.fused_norm_rope``, but tuned for H100:

* 2D grid ``(num_tokens, 3)`` instead of ``(num_tokens, 4)``: the top-k buffer
  clear is folded into the indexer program, and the two-program form is used
  when there is no indexer, so no launch block is idle.
* The interleaved ``k_pe`` RoPE is loaded, rotated and stored as one contiguous
  vector (reshape -> ``tl.split`` -> ``tl.join``) instead of two stride-2
  gathers, which keeps 128-bit vector memory instructions.
* The indexer-K rotation partner is gathered from the register-resident
  normalized row instead of re-reading the raw row, weight and bias from global
  memory three more times per token.
* Streaming loads use ``evict_first``.

Launch path: at 1-512 tokens the op is host-bound, and the baseline (vLLM's
plain Triton launcher) takes ~7.8 us. The generic ``@libentry``/``@libtuner``
dispatch adds ~27 us per call, so the Hopper path resolves its launch config
once from ``tune_configs.yaml`` and caches the compiled runner per kernel
specialization (the same pattern as ``dsv41_o_proj_single_launch.py``).
Repeat calls with the same buffers pay only the launch itself.

The ``fp8_ds_mla`` byte layout is rare and bandwidth-bound by the same stores,
so it delegates to the generic implementation instead of duplicating it.
"""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.fused_norm_rope import (
    _FP8_MAX,
    _INDEXER_CACHE_BLOCK_TILE,
    _fp8_quant_and_cache_write,
    _prepare_fused_norm_rope,
    _rms_norm,
)
from flaggems_vllm.ops.fused_norm_rope import (
    fused_norm_rope as _generic_fused_norm_rope,
)

# Compiled runners keyed by every input identity, eps values and switches that
# select or feed the compiled kernel. Entries keep references to the keyed
# tensors so their ids cannot be recycled while an entry is cached; the small
# cap bounds that retained memory (serving reuses one set of KV-cache buffers).
_LAUNCH_CACHE = {}
_MAX_CACHE_ENTRIES = 8


@triton.jit
def _fused_norm_rope_hopper_kernel(
    pos_ptr,
    # Q RMS norm
    q_c_ptr,
    q_c_stride,
    q_rms_norm_w_ptr,
    q_rms_eps,
    q_c_out_ptr,
    q_c_out_stride,
    Q_DIM: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr,
    # KV RMS norm
    kv_ptr,
    kv_stride,
    kv_rms_norm_w_ptr,
    kv_rms_eps,
    kv_out_ptr,
    kv_out_stride,
    KV_DIM: tl.constexpr,
    # KV RoPE
    kpe_ptr,
    kpe_stride,
    kpe_rope_cos_sin_cache_ptr,
    kpe_rope_cos_sin_cache_stride,
    kpe_out_ptr,
    kpe_out_stride,
    KPE_ROT_DIM: tl.constexpr,
    # Index K layer norm
    index_k_ptr,
    index_k_stride,
    index_k_layer_norm_w_ptr,
    index_k_layer_norm_bias_ptr,
    index_k_layer_norm_eps,
    INDEX_K_DIM: tl.constexpr,
    INDEX_K_BLOCK_SIZE: tl.constexpr,
    # Index K RoPE
    index_k_rope_cos_sin_cache_ptr,
    index_k_rope_cos_sin_cache_stride,
    index_k_out_ptr,
    index_k_out_stride,
    INDEX_K_HALF_ROT_DIM: tl.constexpr,
    # Cache params
    slot_mapping_ptr,
    indexer_slot_mapping_ptr,
    # Index K FP8 cache
    indexer_cache_ptr,
    indexer_cache_scale_ptr,
    indexer_cache_block_size,
    indexer_cache_stride,
    INDEXER_CACHE_SHUFFLE: tl.constexpr,
    INDEXER_CACHE_BLOCK_TILE: tl.constexpr,
    INDEXER_CACHE_HEAD_TILE: tl.constexpr,
    # MLA KV cache
    mla_cache_ptr,
    mla_cache_block_stride,
    mla_cache_entry_stride,
    MLA_CACHE_BLOCK_SIZE: tl.constexpr,
    MLA_CACHE_FP8: tl.constexpr,
    mla_cache_scale_ptr,
    # Top k indices
    topk_indices_ptr,
    topk_indices_stride,
    TOPK: tl.constexpr,
    TOPK_BLOCK_SIZE: tl.constexpr,
    HAS_INDEXER: tl.constexpr,
    INDEX_ROPE_INTERLEAVE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    num_tokens,
):
    tok_idx = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1)

    if part == 0:
        # Q RMSNorm runs for every row (padding / DCP non-owner rows included).
        q_block = tl.arange(0, Q_BLOCK_SIZE)
        q_mask = q_block < Q_DIM
        q_c = tl.load(
            q_c_ptr + tok_idx * q_c_stride + q_block,
            mask=q_mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        q_c_rms_w = tl.load(q_rms_norm_w_ptr + q_block, mask=q_mask, other=0.0)
        q_c = _rms_norm(q_c, q_c_rms_w, q_rms_eps, Q_DIM)
        tl.store(q_c_out_ptr + tok_idx * q_c_out_stride + q_block, q_c, mask=q_mask)
        return

    if part == 2 and HAS_INDEXER:
        # Clear the top-k buffer before the slot gate: DCP non-owner rows keep
        # their -1 buffer even though their cache slots are skipped.
        for i in range(0, TOPK, TOPK_BLOCK_SIZE):
            offset = i + tl.arange(0, TOPK_BLOCK_SIZE)
            tl.store(
                topk_indices_ptr + tok_idx * topk_indices_stride + offset,
                -1,
                mask=offset < TOPK,
            )

    if slot_mapping_ptr is None:
        if kv_out_ptr is None and kpe_out_ptr is None and index_k_out_ptr is None:
            return
    elif tl.load(slot_mapping_ptr + tok_idx) < 0 and (
        part != 1 or (kv_out_ptr is None and kpe_out_ptr is None)
    ):
        # Padding, or a token whose KV slot another rank owns.
        return

    if part == 1:
        # KV RMSNorm + interleaved RoPE + MLA concat_and_cache.
        kv_block = tl.arange(0, KV_DIM)
        kv_c = tl.load(
            kv_ptr + tok_idx * kv_stride + kv_block, eviction_policy="evict_first"
        )
        kv_c_rms_w = tl.load(kv_rms_norm_w_ptr + kv_block)
        kv_c = _rms_norm(kv_c, kv_c_rms_w, kv_rms_eps, KV_DIM)

        pos = tl.load(pos_ptr + tok_idx)
        half_rot: tl.constexpr = KPE_ROT_DIM // 2
        rot_off = tl.arange(0, half_rot)
        cos = tl.load(
            kpe_rope_cos_sin_cache_ptr + pos * kpe_rope_cos_sin_cache_stride + rot_off
        ).to(tl.float32)
        sin = tl.load(
            kpe_rope_cos_sin_cache_ptr
            + pos * kpe_rope_cos_sin_cache_stride
            + half_rot
            + rot_off
        ).to(tl.float32)
        # One contiguous load, rotate adjacent pairs in registers, one store.
        kpe_off = tl.arange(0, KPE_ROT_DIM)
        x = tl.load(
            kpe_ptr + tok_idx * kpe_stride + kpe_off, eviction_policy="evict_first"
        ).to(tl.float32)
        x1, x2 = tl.split(tl.reshape(x, (half_rot, 2)))
        r1 = x1 * cos - x2 * sin
        r2 = x2 * cos + x1 * sin
        roped = tl.reshape(tl.join(r1, r2), (KPE_ROT_DIM,))

        if kv_out_ptr is not None:
            tl.store(kv_out_ptr + tok_idx * kv_out_stride + kv_block, kv_c)
        if kpe_out_ptr is not None:
            tl.store(
                kpe_out_ptr + tok_idx * kpe_out_stride + kpe_off,
                roped.to(kpe_out_ptr.dtype.element_ty),
            )

        if slot_mapping_ptr is not None and mla_cache_entry_stride != 0:
            slot_idx = tl.load(slot_mapping_ptr + tok_idx)
            if slot_idx >= 0:
                dst = (
                    mla_cache_ptr
                    + (slot_idx // MLA_CACHE_BLOCK_SIZE) * mla_cache_block_stride
                    + (slot_idx % MLA_CACHE_BLOCK_SIZE) * mla_cache_entry_stride
                )
                if MLA_CACHE_FP8:
                    scale = tl.load(mla_cache_scale_ptr)
                    tl.store(
                        dst + kv_block, (kv_c.to(tl.float32) / scale).to(tl.float8e4nv)
                    )
                    tl.store(
                        dst + KV_DIM + kpe_off,
                        (roped / scale).to(tl.float8e4nv),
                    )
                else:
                    tl.store(dst + kv_block, kv_c)
                    tl.store(dst + KV_DIM + kpe_off, roped)

    elif part == 2:
        if not HAS_INDEXER:
            return

        index_k_block = tl.arange(0, INDEX_K_BLOCK_SIZE)
        index_k_mask = index_k_block < INDEX_K_DIM
        index_k = tl.load(
            index_k_ptr + tok_idx * index_k_stride + index_k_block,
            mask=index_k_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        index_k_w = tl.load(
            index_k_layer_norm_w_ptr + index_k_block, mask=index_k_mask, other=0.0
        ).to(tl.float32)
        index_k_b = tl.load(
            index_k_layer_norm_bias_ptr + index_k_block, mask=index_k_mask, other=0.0
        ).to(tl.float32)

        mean = tl.sum(index_k, axis=0) / INDEX_K_DIM
        diff = tl.where(index_k_mask, index_k - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) / INDEX_K_DIM
        rstd = tl.rsqrt(var + index_k_layer_norm_eps)
        normed = (index_k - mean) * rstd * index_k_w + index_k_b

        pos = tl.load(pos_ptr + tok_idx)
        in_rope = index_k_block < 2 * INDEX_K_HALF_ROT_DIM
        if INDEX_ROPE_INTERLEAVE:
            cos_idx = index_k_block // 2
            partner_offs = tl.where(in_rope, index_k_block ^ 1, index_k_block)
            sign = tl.where(index_k_block % 2 == 0, -1.0, 1.0)
        else:
            cos_idx = index_k_block % INDEX_K_HALF_ROT_DIM
            partner_offs = tl.where(
                in_rope, index_k_block ^ INDEX_K_HALF_ROT_DIM, index_k_block
            )
            sign = tl.where(index_k_block < INDEX_K_HALF_ROT_DIM, -1.0, 1.0)
        cos_full = tl.load(
            index_k_rope_cos_sin_cache_ptr
            + pos * index_k_rope_cos_sin_cache_stride
            + cos_idx,
            mask=in_rope,
            other=1.0,
        ).to(tl.float32)
        sin_full = tl.load(
            index_k_rope_cos_sin_cache_ptr
            + pos * index_k_rope_cos_sin_cache_stride
            + INDEX_K_HALF_ROT_DIM
            + cos_idx,
            mask=in_rope,
            other=0.0,
        ).to(tl.float32)
        # The rotation partner is already normalized in registers.
        normed_partner = tl.gather(normed, partner_offs, 0)
        roped = normed * cos_full + sign * normed_partner * sin_full
        result = tl.where(in_rope, roped, normed)

        if index_k_out_ptr is not None:
            tl.store(
                index_k_out_ptr + tok_idx * index_k_out_stride + index_k_block,
                result.to(index_k_out_ptr.dtype.element_ty),
                mask=index_k_mask,
            )

        if indexer_cache_ptr is not None and indexer_slot_mapping_ptr is not None:
            slot_idx = tl.load(indexer_slot_mapping_ptr + tok_idx)
            _fp8_quant_and_cache_write(
                result,
                index_k_mask,
                slot_idx,
                indexer_cache_ptr,
                indexer_cache_scale_ptr,
                indexer_cache_block_size,
                indexer_cache_stride,
                index_k_block,
                INDEX_K_DIM,
                INDEXER_CACHE_SHUFFLE,
                INDEXER_CACHE_BLOCK_TILE,
                INDEXER_CACHE_HEAD_TILE,
                FP8_MAX,
            )


def _launch_config():
    """Resolve the launch resources once from the Hopper tune config.

    The one-warp candidate wins the H100 sweep at every token count (see
    ``benchmark/test_fused_norm_rope.py``): each program owns a single token row
    and occupancy comes from the number of programs, not from the CTA width.
    """
    configs = runtime.get_tuned_config("fused_norm_rope_hopper")
    if not configs:
        return 1, 1
    best = min(configs, key=lambda config: config.num_warps)
    return best.num_warps, best.num_stages


def _build_kernel_args(args, positions, q_out):
    """Order the prepared values exactly like the kernel signature."""
    return [
        positions,
        args["q_c"],
        args["q_c_stride"],
        args["q_rms_norm_w"],
        args["q_rms_eps"],
        q_out,
        args["q_c_out_stride"],
        args["q_dim"],
        args["q_block"],
        args["kv_c"],
        args["kv_stride"],
        args["kv_rms_norm_w"],
        args["kv_rms_eps"],
        args["kv_c_out"],
        args["kv_out_stride"],
        args["kv_dim"],
        args["k_pe"],
        args["kpe_stride"],
        args["kpe_cos_sin"],
        args["kpe_cos_sin_stride"],
        args["k_pe_out"],
        args["kpe_out_stride"],
        args["kpe_half_rot"] * 2,
        args["index_k"],
        args["index_k_stride"],
        args["index_k_w"],
        args["index_k_b"],
        args["index_k_eps"],
        args["index_k_dim"],
        args["index_k_block"],
        args["index_k_cos_sin"],
        args["index_k_cos_sin_stride"],
        args["index_k_out"],
        args["index_k_out_stride"],
        args["index_k_half_rot"],
        args["slot_mapping"],
        args["indexer_slot_mapping"],
        args["indexer_k_cache"],
        args["idx_cache_scale_view"],
        args["idx_cache_block_size"],
        args["idx_cache_stride"],
        args["idx_cache_shuffle"],
        _INDEXER_CACHE_BLOCK_TILE,
        args["idx_cache_head_tile"],
        args["mla_cache"],
        args["mla_block_stride"],
        args["mla_entry_stride"],
        args["mla_block_size"],
        args["mla_fp8"],
        args["mla_scale"],
        args["topk"],
        args["topk_stride"],
        args["topk_len"],
        1024,
        args["has_indexer"],
        args["index_interleave"],
        _FP8_MAX,
        args["num_tokens"],
    ]


def fused_norm_rope(
    positions: torch.Tensor,
    q_c: torch.Tensor,
    q_rms_norm_w: torch.Tensor,
    q_rms_eps: float,
    kv_c: torch.Tensor,
    kv_rms_norm_w: torch.Tensor,
    kv_rms_eps: float,
    k_pe: torch.Tensor,
    k_rope_cos_sin_cache: torch.Tensor,
    index_k: torch.Tensor | None,
    index_k_layer_norm_w: torch.Tensor | None,
    index_k_layer_norm_bias: torch.Tensor | None,
    index_k_layer_norm_eps: float,
    index_k_rope_cos_sin_cache: torch.Tensor | None,
    topk_indices_buffer: torch.Tensor,
    slot_mapping: torch.Tensor | None = None,
    indexer_slot_mapping: torch.Tensor | None = None,
    indexer_k_cache: torch.Tensor | None = None,
    indexer_cache_shuffled: bool = False,
    mla_kv_cache: torch.Tensor | None = None,
    mla_kv_cache_dtype: str = "auto",
    mla_k_scale: torch.Tensor | None = None,
    has_indexer: bool = True,
    index_rope_interleave: bool = False,
    q_c_out: torch.Tensor | None = None,
    kv_c_out: torch.Tensor | None = None,
    k_pe_out: torch.Tensor | None = None,
    index_k_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hopper entry point; same contract as the generic implementation."""
    if mla_kv_cache_dtype == "fp8_ds_mla":
        # Rare byte-addressed layout; reuse the generic kernel unchanged.
        return _generic_fused_norm_rope(
            positions,
            q_c,
            q_rms_norm_w,
            q_rms_eps,
            kv_c,
            kv_rms_norm_w,
            kv_rms_eps,
            k_pe,
            k_rope_cos_sin_cache,
            index_k,
            index_k_layer_norm_w,
            index_k_layer_norm_bias,
            index_k_layer_norm_eps,
            index_k_rope_cos_sin_cache,
            topk_indices_buffer,
            slot_mapping,
            indexer_slot_mapping,
            indexer_k_cache,
            indexer_cache_shuffled,
            mla_kv_cache,
            mla_kv_cache_dtype,
            mla_k_scale,
            has_indexer,
            index_rope_interleave,
            q_c_out,
            kv_c_out,
            k_pe_out,
            index_k_out,
        )

    num_tokens = positions.shape[0]
    if num_tokens == 0:
        return q_c_out if q_c_out is not None else torch.empty_like(q_c)

    # Everything that selects the compiled kernel or is baked into its argument
    # list. q_c_out is patched per call, so only its alignment matters.
    key = (
        id(positions),
        id(q_c),
        id(q_rms_norm_w),
        id(kv_c),
        id(kv_rms_norm_w),
        id(k_pe),
        id(k_rope_cos_sin_cache),
        id(index_k),
        id(index_k_layer_norm_w),
        id(index_k_layer_norm_bias),
        id(index_k_rope_cos_sin_cache),
        id(topk_indices_buffer),
        id(slot_mapping),
        id(indexer_slot_mapping),
        id(indexer_k_cache),
        id(mla_kv_cache),
        id(mla_k_scale),
        id(kv_c_out),
        id(k_pe_out),
        id(index_k_out),
        q_rms_eps,
        kv_rms_eps,
        index_k_layer_norm_eps,
        has_indexer,
        index_rope_interleave,
        indexer_cache_shuffled,
        mla_kv_cache_dtype,
        num_tokens,
        q_c_out is None or q_c_out.data_ptr() % 16 == 0,
    )
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        args = _prepare_fused_norm_rope(
            positions,
            q_c,
            q_rms_norm_w,
            q_rms_eps,
            kv_c,
            kv_rms_norm_w,
            kv_rms_eps,
            k_pe,
            k_rope_cos_sin_cache,
            index_k,
            index_k_layer_norm_w,
            index_k_layer_norm_bias,
            index_k_layer_norm_eps,
            index_k_rope_cos_sin_cache,
            topk_indices_buffer,
            slot_mapping,
            indexer_slot_mapping,
            indexer_k_cache,
            indexer_cache_shuffled,
            mla_kv_cache,
            mla_kv_cache_dtype,
            mla_k_scale,
            has_indexer,
            index_rope_interleave,
            q_c_out,
            kv_c_out,
            k_pe_out,
            index_k_out,
        )
        if args["num_tokens"] == 0:
            return args["q_c_out"]
        q_out = args["q_c_out"]
        kernel_args = _build_kernel_args(args, positions, q_out)
        grid = (num_tokens, 3 if has_indexer else 2, 1)
        num_warps, num_stages = _launch_config()
        compiled = _fused_norm_rope_hopper_kernel[grid](
            *kernel_args, num_warps=num_warps, num_stages=num_stages
        )
        if len(_LAUNCH_CACHE) >= _MAX_CACHE_ENTRIES:
            _LAUNCH_CACHE.clear()
        # Keep the keyed tensors alive so their ids stay unique, and reuse the
        # argument list (only q_c_out is refreshed on later calls).
        entry = (
            compiled[grid],
            kernel_args,
            (
                positions,
                q_c,
                q_rms_norm_w,
                kv_c,
                kv_rms_norm_w,
                k_pe,
                k_rope_cos_sin_cache,
                index_k,
                index_k_layer_norm_w,
                index_k_layer_norm_bias,
                index_k_rope_cos_sin_cache,
                topk_indices_buffer,
                slot_mapping,
                indexer_slot_mapping,
                indexer_k_cache,
                mla_kv_cache,
                mla_k_scale,
                kv_c_out,
                k_pe_out,
                index_k_out,
            ),
        )
        _LAUNCH_CACHE[key] = entry
        return q_out

    run, kernel_args, _ = entry
    q_out = q_c_out if q_c_out is not None else torch.empty_like(q_c)
    kernel_args[5] = q_out
    run(*kernel_args)
    return q_out


__all__ = ["fused_norm_rope"]
