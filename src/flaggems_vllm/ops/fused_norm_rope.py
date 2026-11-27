# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Fused DeepSeek-V3.2 (DSA) preprocessing: norm + RoPE + cache insert.

Generic backend implementation of vLLM 0.32's
``vllm.models.deepseek_v32.common.kernels.fused_norm_rope``. One launch does:

  * ``q_c``      : RMSNorm (no RoPE)
  * ``kv_c``     : RMSNorm, then ``k_pe`` interleaved RoPE, then a fused
                   ``concat_and_cache_mla`` insert (bf16, per-tensor FP8, or the
                   byte-addressed ``fp8_ds_mla`` layout)
  * ``index_k``  : LayerNorm + RoPE (interleaved or NeoX) + UE8M0 FP8 quant +
                   packed indexer-cache insert
  * top-k buffer : filled with ``-1`` on indexer layers

The per-token work is split over a 2D grid ``(num_tokens, 4)``, exactly like the
vLLM reference kernel. ``pid == 3`` clears the top-k buffer, ``pid == 2`` does
the Q RMSNorm, ``pid == 1`` the KV path and ``pid == 0`` the indexer path, so a
single launch covers every fused stage. This is the portable path; the Hopper
backend replaces it with a narrower grid and vectorized interleaved-RoPE loads.
"""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner

# NVIDIA FP8 (OCP e4m3). The FP8 cache paths assume the OCP range; FNUZ (AMD)
# needs 224 and is not enabled by this generic path.
_FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
_FP8_MAX = 448.0

# Tile shape of the indexer-K cache's shuffled layout. The block tile counts
# tokens, the head tile counts bytes; the kernel converts it to elements.
_INDEXER_CACHE_BLOCK_TILE = 16
_INDEXER_CACHE_HEAD_TILE_BYTES = 16

# Tiny reused dummies for the has_indexer=False path so the indexer arguments
# never allocate per call. The kernel never dereferences them there.
_DUMMY_CACHE: dict = {}


def _dummy(shape, dtype, device):
    key = (shape, dtype, device)
    tensor = _DUMMY_CACHE.get(key)
    if tensor is None:
        tensor = torch.empty(shape, dtype=dtype, device=device)
        _DUMMY_CACHE[key] = tensor
    return tensor


def _is_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    """Mirror ``vllm.utils.torch_utils.is_quantized_kv_cache`` without vLLM."""
    return (
        kv_cache_dtype.startswith("fp8")
        or kv_cache_dtype.endswith("per_token_head")
        or kv_cache_dtype.startswith("nvfp4")
    )


@triton.jit
def _rms_norm(x, w, eps, HIDDEN: tl.constexpr):
    x = x.to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / HIDDEN
    rrms = tl.rsqrt(mean_sq + eps)
    return (x * rrms) * w.to(tl.float32)


@triton.jit
def _get_cos_sin(cos_sin_cache_ptr, cos_sin_cache_stride, pos, HALF_ROT: tl.constexpr):
    block = tl.arange(0, HALF_ROT)
    cos = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block)
    sin = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block + HALF_ROT)
    return cos.to(tl.float32), sin.to(tl.float32)


@triton.jit
def _fp8_ue8m0_quantize(vals, FP8_MAX: tl.constexpr):
    """Per-row FP8 E4M3 with a power-of-two (UE8M0) scale."""
    vals = vals.to(tl.float32)
    amax = tl.max(tl.abs(vals))
    scale = tl.div_rn(tl.maximum(amax, 1e-4), FP8_MAX)
    scale = tl.math.exp2(tl.math.ceil(tl.math.log2(scale)))
    return tl.div_rn(vals, scale).to(tl.float8e4nv), scale


@triton.jit
def _fp8_quant_and_cache_write(
    vals,
    mask,
    slot_idx,
    cache_ptr,
    cache_scale_ptr,
    cache_block_size,
    cache_block_stride,
    offsets,
    HEAD_DIM: tl.constexpr,
    SHUFFLE: tl.constexpr,
    BLOCK_TILE: tl.constexpr,
    HEAD_TILE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """Quantize one indexer-K row and store [values | per-token scale]."""
    k_fp8, scale = _fp8_ue8m0_quantize(vals, FP8_MAX)

    block_idx = slot_idx // cache_block_size
    block_offset = slot_idx % cache_block_size
    block_start = block_idx * cache_block_stride

    # Shuffled layout: [blk/BLOCK_TILE, HEAD_DIM/HEAD_TILE, BLOCK_TILE, HEAD_TILE]
    if SHUFFLE:
        value_off = (
            block_offset // BLOCK_TILE * BLOCK_TILE * HEAD_DIM
            + block_offset % BLOCK_TILE * HEAD_TILE
            + offsets // HEAD_TILE * BLOCK_TILE * HEAD_TILE
            + offsets % HEAD_TILE
        )
    else:
        value_off = block_offset * HEAD_DIM + offsets
    tl.store(cache_ptr + block_start + value_off, k_fp8, mask=mask)

    scale_byte_off = block_start + cache_block_size * HEAD_DIM + block_offset * 4
    tl.store(cache_scale_ptr + scale_byte_off // 4, scale)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("fused_norm_rope"),
    key=["num_tokens"],
    strategy=["log"],
)
@triton.jit
def _fused_norm_rope_kernel(
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
    KPE_HALF_ROT_DIM: tl.constexpr,
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
    # MLA KV cache (concat kv_c_normed + k_pe_roped, uses slot_mapping_ptr)
    mla_cache_ptr,
    mla_cache_block_stride,
    mla_cache_entry_stride,
    MLA_CACHE_BLOCK_SIZE: tl.constexpr,
    MLA_CACHE_FP8: tl.constexpr,
    mla_cache_scale_ptr,
    # fp8_ds_mla: block-scaled FP8 NoPE + unquantized bf16 RoPE views
    mla_cache_ds_scale_ptr,
    mla_cache_ds_rope_ptr,
    MLA_CACHE_DS_MLA: tl.constexpr,
    MLA_NUM_TILES: tl.constexpr,
    MLA_TILE_DIM: tl.constexpr,
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
    pid = tl.program_id(1)

    if pid == 3:
        # Shared (no-indexer) layers reuse the previous indexer layer's top-k.
        if not HAS_INDEXER:
            return
        for i in range(0, TOPK, TOPK_BLOCK_SIZE):
            offset = i + tl.arange(0, TOPK_BLOCK_SIZE)
            tl.store(
                topk_indices_ptr + tok_idx * topk_indices_stride + offset,
                -1,
                mask=offset < TOPK,
            )
        return

    if pid == 2:
        # Q RMSNorm. Runs for every row, including padding / DCP non-owner rows.
        q_block = tl.arange(0, Q_BLOCK_SIZE)
        q_mask = q_block < Q_DIM
        q_c = tl.load(q_c_ptr + tok_idx * q_c_stride + q_block, mask=q_mask, other=0.0)
        q_c_rms_w = tl.load(q_rms_norm_w_ptr + q_block, mask=q_mask, other=0.0)
        q_c = _rms_norm(q_c, q_c_rms_w, q_rms_eps, Q_DIM)
        tl.store(q_c_out_ptr + tok_idx * q_c_out_stride + q_block, q_c, mask=q_mask)
        return

    if slot_mapping_ptr is None:
        if kv_out_ptr is None and kpe_out_ptr is None and index_k_out_ptr is None:
            return
    elif tl.load(slot_mapping_ptr + tok_idx) < 0 and (
        pid != 1 or (kv_out_ptr is None and kpe_out_ptr is None)
    ):
        # Padding, or a token whose KV slot another rank owns.
        return

    if pid == 1:
        # KV RMSNorm + KV RoPE + MLA concat_and_cache, all in registers.
        kv_block = tl.arange(0, KV_DIM)
        kv_c = tl.load(kv_ptr + tok_idx * kv_stride + kv_block)
        kv_c_rms_w = tl.load(kv_rms_norm_w_ptr + kv_block)
        kv_c = _rms_norm(kv_c, kv_c_rms_w, kv_rms_eps, KV_DIM)

        pos = tl.load(pos_ptr + tok_idx)
        cos, sin = _get_cos_sin(
            kpe_rope_cos_sin_cache_ptr,
            kpe_rope_cos_sin_cache_stride,
            pos,
            KPE_HALF_ROT_DIM,
        )
        dim_off = tl.arange(0, KPE_HALF_ROT_DIM)
        kpe_base = kpe_ptr + tok_idx * kpe_stride
        x1 = tl.load(kpe_base + dim_off * 2).to(tl.float32)
        x2 = tl.load(kpe_base + dim_off * 2 + 1).to(tl.float32)
        r1 = x1 * cos - x2 * sin
        r2 = x2 * cos + x1 * sin

        if kv_out_ptr is not None:
            tl.store(kv_out_ptr + tok_idx * kv_out_stride + kv_block, kv_c)
        if kpe_out_ptr is not None:
            tl.store(
                kpe_out_ptr + tok_idx * kpe_out_stride + dim_off * 2,
                r1.to(kpe_out_ptr.dtype.element_ty),
            )
            tl.store(
                kpe_out_ptr + tok_idx * kpe_out_stride + dim_off * 2 + 1,
                r2.to(kpe_out_ptr.dtype.element_ty),
            )

        if slot_mapping_ptr is not None:
            if mla_cache_entry_stride == 0:
                return
            slot_idx = tl.load(slot_mapping_ptr + tok_idx)
            if slot_idx < 0:
                return
            mla_block_idx = slot_idx // MLA_CACHE_BLOCK_SIZE
            mla_block_off = slot_idx % MLA_CACHE_BLOCK_SIZE
            byte_base = (
                mla_block_idx * mla_cache_block_stride
                + mla_block_off * mla_cache_entry_stride
            )

            if MLA_CACHE_DS_MLA:
                # fp8_ds_mla layout: per-128-tile FP8 NoPE with a power-of-two
                # float32 scale per tile, then bf16 (unquantized) RoPE tail.
                kv_2d = tl.reshape(kv_c, (MLA_NUM_TILES, MLA_TILE_DIM))
                tile_amax = tl.max(tl.abs(kv_2d), axis=1, keep_dims=True)
                tile_scale = tl.maximum(tile_amax * (1.0 / FP8_MAX), 1e-4)
                tile_scale = tl.math.exp2(tl.math.ceil(tl.math.log2(tile_scale)))
                kv_c_fp8 = tl.reshape((kv_2d / tile_scale).to(tl.float8e4nv), (KV_DIM,))
                tl.store(mla_cache_ptr + byte_base + kv_block, kv_c_fp8)
                tile_off = tl.arange(0, MLA_NUM_TILES)
                tl.store(
                    mla_cache_ds_scale_ptr + byte_base // 4 + KV_DIM // 4 + tile_off,
                    tl.reshape(tile_scale, (MLA_NUM_TILES,)),
                )
                rope_dst = mla_cache_ds_rope_ptr + byte_base // 2 + (KV_DIM // 2 + 8)
                tl.store(rope_dst + dim_off * 2, r1.to(tl.bfloat16))
                tl.store(rope_dst + dim_off * 2 + 1, r2.to(tl.bfloat16))
                return

            dst = mla_cache_ptr + byte_base
            if MLA_CACHE_FP8:
                scale = tl.load(mla_cache_scale_ptr)
                tl.store(
                    dst + kv_block, (kv_c.to(tl.float32) / scale).to(tl.float8e4nv)
                )
                tl.store(dst + KV_DIM + dim_off * 2, (r1 / scale).to(tl.float8e4nv))
                tl.store(dst + KV_DIM + dim_off * 2 + 1, (r2 / scale).to(tl.float8e4nv))
            else:
                tl.store(dst + kv_block, kv_c)
                tl.store(dst + KV_DIM + dim_off * 2, r1)
                tl.store(dst + KV_DIM + dim_off * 2 + 1, r2)

    elif pid == 0:
        if not HAS_INDEXER:
            return

        index_k_block = tl.arange(0, INDEX_K_BLOCK_SIZE)
        index_k_mask = index_k_block < INDEX_K_DIM
        index_k = tl.load(
            index_k_ptr + tok_idx * index_k_stride + index_k_block,
            mask=index_k_mask,
            other=0.0,
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
        # Re-normalize the rotation partner from the raw row and its affine.
        raw_partner = tl.load(
            index_k_ptr + tok_idx * index_k_stride + partner_offs,
            mask=index_k_mask,
            other=0.0,
        ).to(tl.float32)
        w_partner = tl.load(
            index_k_layer_norm_w_ptr + partner_offs, mask=index_k_mask, other=0.0
        ).to(tl.float32)
        b_partner = tl.load(
            index_k_layer_norm_bias_ptr + partner_offs, mask=index_k_mask, other=0.0
        ).to(tl.float32)
        normed_partner = (raw_partner - mean) * rstd * w_partner + b_partner
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


def _prepare_fused_norm_rope(
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
) -> dict:
    """Validate the vLLM 0.32 arguments and normalize them for the kernels.

    Both the generic and the Hopper launcher consume this dict, so the argument
    normalization (cache views, strides, dummy tensors) is shared verbatim.
    """
    assert positions.ndim == 1
    assert q_c.ndim == 2
    assert kv_c.ndim == 2
    assert k_pe.ndim == 2
    assert topk_indices_buffer.ndim == 2

    num_tokens = positions.shape[0]
    q_dim = q_c.shape[-1]
    kv_dim = kv_c.shape[-1]
    device = positions.device

    if mla_kv_cache_dtype == "nvfp4_ds_mla":
        raise NotImplementedError(
            "fused_norm_rope: nvfp4_ds_mla caches require SM100 FP4 packing"
        )

    if not has_indexer:
        # Shared layers: dummies are never dereferenced (HAS_INDEXER skips them).
        indexer_k_cache = None
        index_k = _dummy((1, 1), q_c.dtype, device)
        index_k_layer_norm_w = _dummy((1,), torch.float32, device)
        index_k_layer_norm_bias = _dummy((1,), torch.float32, device)
        index_k_rope_cos_sin_cache = k_rope_cos_sin_cache
    assert index_k is not None
    assert index_k_rope_cos_sin_cache is not None
    index_k_dim = index_k.shape[-1]
    topk = topk_indices_buffer.shape[-1]

    if indexer_k_cache is not None or mla_kv_cache is not None:
        assert slot_mapping is not None, "slot_mapping is required with a cache"
    else:
        slot_mapping = None

    # --- Indexer K cache setup ---
    if indexer_k_cache is not None:
        assert slot_mapping is not None
        if indexer_slot_mapping is None:
            indexer_slot_mapping = slot_mapping
        idx_cache_scale_view = indexer_k_cache.view(torch.uint8).view(torch.float32)
        idx_cache_block_size = indexer_k_cache.shape[1]
        idx_cache_shuffle = bool(indexer_cache_shuffled)
        if idx_cache_shuffle:
            assert idx_cache_block_size % _INDEXER_CACHE_BLOCK_TILE == 0, (
                f"indexer K cache block size {idx_cache_block_size} must be a "
                f"multiple of {_INDEXER_CACHE_BLOCK_TILE} for the shuffled layout"
            )
        idx_cache_stride = indexer_k_cache.stride(0)
        if indexer_k_cache.dtype == torch.uint8:
            indexer_k_cache = indexer_k_cache.view(_FP8_DTYPE)
        idx_cache_head_tile = (
            _INDEXER_CACHE_HEAD_TILE_BYTES // indexer_k_cache.element_size()
        )
    else:
        idx_cache_scale_view = None
        idx_cache_block_size = 1
        idx_cache_stride = 0
        idx_cache_shuffle = False
        idx_cache_head_tile = _INDEXER_CACHE_HEAD_TILE_BYTES

    # --- MLA KV cache setup ---
    mla_cache_ds_mla = mla_kv_cache_dtype == "fp8_ds_mla"
    mla_cache_fp8 = _is_quantized_kv_cache(mla_kv_cache_dtype) and not mla_cache_ds_mla
    mla_num_tiles = 1
    mla_ds_scale_view = torch.empty(0, dtype=torch.float32, device=device)
    mla_ds_rope_view = torch.empty(0, dtype=torch.bfloat16, device=device)
    if mla_kv_cache is not None:
        mla_block_size = mla_kv_cache.shape[1]
        if mla_cache_ds_mla:
            # 656 B/token byte-addressed layout (512 FP8 NoPE + 4 f32 scales +
            # 64 bf16 RoPE). The fp8 view makes the strides byte offsets.
            assert kv_dim == 512, "fp8_ds_mla requires kv_lora_rank == 512"
            mla_num_tiles = kv_dim // 128
            u8_cache = mla_kv_cache.view(torch.uint8)
            mla_block_stride = u8_cache.stride(0)
            mla_entry_stride = u8_cache.stride(1)
            mla_ds_scale_view = u8_cache.view(torch.float32)
            mla_ds_rope_view = u8_cache.view(torch.bfloat16)
            mla_kv_cache = u8_cache.view(_FP8_DTYPE)
        else:
            mla_block_stride = mla_kv_cache.stride(0)
            mla_entry_stride = mla_kv_cache.stride(1)
            if mla_cache_fp8 and mla_kv_cache.dtype == torch.uint8:
                mla_kv_cache = mla_kv_cache.view(_FP8_DTYPE)
        if mla_k_scale is None:
            mla_k_scale = torch.ones(1, dtype=torch.float32, device=device)
    else:
        # Zero entry stride disables the cache write in the kernel.
        mla_kv_cache = torch.empty(0, dtype=torch.bfloat16, device=device)
        mla_block_stride = 0
        mla_entry_stride = 0
        mla_block_size = 1
        mla_k_scale = _dummy((1,), torch.float32, device)

    if q_c_out is None:
        q_c_out = torch.empty_like(q_c)
    kv_out_stride = 0
    kpe_out_stride = 0
    index_k_out_stride = 0
    if kv_c_out is not None:
        assert kv_c_out.shape == kv_c.shape
        kv_out_stride = kv_c_out.stride(0)
    if k_pe_out is not None:
        assert k_pe_out.shape == k_pe.shape
        kpe_out_stride = k_pe_out.stride(0)
    if index_k_out is not None:
        assert index_k_out.shape == index_k.shape
        index_k_out_stride = index_k_out.stride(0)

    return dict(
        num_tokens=num_tokens,
        device=device,
        q_c_out=q_c_out,
        q_c=q_c,
        q_c_stride=q_c.stride(0),
        q_rms_norm_w=q_rms_norm_w,
        q_rms_eps=q_rms_eps,
        q_c_out_stride=q_c_out.stride(0),
        q_dim=q_dim,
        q_block=triton.next_power_of_2(q_dim),
        kv_c=kv_c,
        kv_stride=kv_c.stride(0),
        kv_rms_norm_w=kv_rms_norm_w,
        kv_rms_eps=kv_rms_eps,
        kv_c_out=kv_c_out,
        kv_out_stride=kv_out_stride,
        kv_dim=kv_dim,
        k_pe=k_pe,
        kpe_stride=k_pe.stride(0),
        kpe_cos_sin=k_rope_cos_sin_cache,
        kpe_cos_sin_stride=k_rope_cos_sin_cache.stride(0),
        k_pe_out=k_pe_out,
        kpe_out_stride=kpe_out_stride,
        kpe_half_rot=k_rope_cos_sin_cache.shape[-1] // 2,
        index_k=index_k,
        index_k_stride=index_k.stride(0),
        index_k_w=index_k_layer_norm_w,
        index_k_b=index_k_layer_norm_bias,
        index_k_eps=index_k_layer_norm_eps,
        index_k_dim=index_k_dim,
        index_k_block=triton.next_power_of_2(index_k_dim),
        index_k_cos_sin=index_k_rope_cos_sin_cache,
        index_k_cos_sin_stride=index_k_rope_cos_sin_cache.stride(0),
        index_k_out=index_k_out,
        index_k_out_stride=index_k_out_stride,
        index_k_half_rot=index_k_rope_cos_sin_cache.shape[-1] // 2,
        slot_mapping=slot_mapping,
        indexer_slot_mapping=indexer_slot_mapping,
        indexer_k_cache=indexer_k_cache,
        idx_cache_scale_view=idx_cache_scale_view,
        idx_cache_block_size=idx_cache_block_size,
        idx_cache_stride=idx_cache_stride,
        idx_cache_shuffle=idx_cache_shuffle,
        idx_cache_head_tile=idx_cache_head_tile,
        mla_cache=mla_kv_cache,
        mla_block_stride=mla_block_stride,
        mla_entry_stride=mla_entry_stride,
        mla_block_size=mla_block_size,
        mla_fp8=mla_cache_fp8,
        mla_scale=mla_k_scale,
        mla_ds_scale=mla_ds_scale_view,
        mla_ds_rope=mla_ds_rope_view,
        mla_ds=mla_cache_ds_mla,
        mla_tiles=mla_num_tiles,
        mla_tile_dim=kv_dim // mla_num_tiles if mla_cache_ds_mla else 1,
        topk=topk_indices_buffer,
        topk_stride=topk_indices_buffer.stride(0),
        topk_len=topk,
        has_indexer=has_indexer,
        index_interleave=index_rope_interleave,
    )


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
    """Fused Q/KV/indexer norm + RoPE + paged cache insert.

    Mirrors vLLM 0.32's ``fused_norm_rope`` signature and side effects: the MLA
    KV cache, the indexer K cache and the top-k buffer are written in place and
    the normalized query is returned (``q_c_out`` when supplied).

    Unsupported: ``mla_kv_cache_dtype="nvfp4_ds_mla"`` needs SM100 / Blackwell
    FP4 packing, so it raises ``NotImplementedError`` here.
    """
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

    with torch_device_fn.device(args["device"]):
        _fused_norm_rope_kernel[(args["num_tokens"], 4)](
            positions,
            # Q RMS norm
            args["q_c"],
            args["q_c_stride"],
            args["q_rms_norm_w"],
            args["q_rms_eps"],
            args["q_c_out"],
            args["q_c_out_stride"],
            args["q_dim"],
            args["q_block"],
            # KV RMS norm
            args["kv_c"],
            args["kv_stride"],
            args["kv_rms_norm_w"],
            args["kv_rms_eps"],
            args["kv_c_out"],
            args["kv_out_stride"],
            args["kv_dim"],
            # KV RoPE
            args["k_pe"],
            args["kpe_stride"],
            args["kpe_cos_sin"],
            args["kpe_cos_sin_stride"],
            args["k_pe_out"],
            args["kpe_out_stride"],
            args["kpe_half_rot"],
            # Index K layer norm
            args["index_k"],
            args["index_k_stride"],
            args["index_k_w"],
            args["index_k_b"],
            args["index_k_eps"],
            args["index_k_dim"],
            args["index_k_block"],
            # Index K RoPE
            args["index_k_cos_sin"],
            args["index_k_cos_sin_stride"],
            args["index_k_out"],
            args["index_k_out_stride"],
            args["index_k_half_rot"],
            # Cache params
            args["slot_mapping"],
            args["indexer_slot_mapping"],
            # Index K FP8 cache
            args["indexer_k_cache"],
            args["idx_cache_scale_view"],
            args["idx_cache_block_size"],
            args["idx_cache_stride"],
            args["idx_cache_shuffle"],
            _INDEXER_CACHE_BLOCK_TILE,
            args["idx_cache_head_tile"],
            # MLA KV cache
            args["mla_cache"],
            args["mla_block_stride"],
            args["mla_entry_stride"],
            args["mla_block_size"],
            args["mla_fp8"],
            args["mla_scale"],
            args["mla_ds_scale"],
            args["mla_ds_rope"],
            args["mla_ds"],
            args["mla_tiles"],
            args["mla_tile_dim"],
            # Top k indices
            args["topk"],
            args["topk_stride"],
            args["topk_len"],
            1024,
            args["has_indexer"],
            args["index_interleave"],
            _FP8_MAX,
            args["num_tokens"],
        )
    return args["q_c_out"]


__all__ = ["fused_norm_rope"]
