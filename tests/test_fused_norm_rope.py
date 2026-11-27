# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Accuracy tests for ``flaggems_vllm.fused_norm_rope``.

The reference is vLLM 0.32's DeepSeek-V3.2 fused kernel
(``vllm.models.deepseek_v32.common.kernels.fused_norm_rope``): its torch
reference math and its test shapes are reused here, and the last test compares
the FlagGems kernels against the vLLM kernel itself when vLLM is importable.
"""

from typing import NamedTuple

import pytest
import torch

import flaggems_vllm

from .conftest import QUICK_MODE

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
EPS = 1e-6

pytestmark = [
    pytest.mark.fused_norm_rope,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

try:
    from vllm.models.deepseek_v32.common.kernels import (
        fused_norm_rope as vllm_fused_norm_rope,
    )
except Exception:  # pragma: no cover - vLLM without DeepSeek-V3.2 kernels
    vllm_fused_norm_rope = None


class ModelConfig(NamedTuple):
    """Per-model attention dims (num_heads is the TP8 local count)."""

    q_lora: int
    kv_lora: int
    rope_dim: int
    index_head_dim: int


# Same model configs as vLLM's tests/kernels/test_fused_deepseek_v32_norm_rope.py
MODEL_CONFIGS = [
    ModelConfig(2048, 512, 64, 128),  # GLM-5.2
    ModelConfig(1536, 512, 64, 128),  # DeepSeek-V3.2
]
MODEL_IDS = ["glm5.2", "dsv3.2"]
NUM_TOKENS = [1, 17] if QUICK_MODE else [1, 4, 17, 512, 4096]
MAX_POS = 8192


# ── reference helpers (same math as the vLLM kernel and its tests) ────────────


def make_cos_sin(max_pos: int, rot_dim: int, device) -> torch.Tensor:
    """cos||sin cache: row[pos] = [cos(theta)(rot/2), sin(theta)(rot/2)]."""
    half = rot_dim // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, half, dtype=torch.float32, device=device) / half)
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


def rms_norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    ms = xf.pow(2).mean(dim=-1, keepdim=True)
    return xf * torch.rsqrt(ms + EPS) * w.float()


def layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = (xf - mean).pow(2).mean(dim=-1, keepdim=True)
    return (xf - mean) * torch.rsqrt(var + EPS) * w.float() + b.float()


def rope(
    x: torch.Tensor, pos: torch.Tensor, cos_sin: torch.Tensor, interleave: bool
) -> torch.Tensor:
    """Apply RoPE to the first ``rot_dim`` elements of x's last dim."""
    rot = cos_sin.shape[-1]
    half = rot // 2
    cs = cos_sin[pos.long()]
    cos, sin = cs[..., :half], cs[..., half:]
    out = x.float().clone()
    r = out[..., :rot]
    if interleave:
        x1, x2 = r[..., 0::2].clone(), r[..., 1::2].clone()
        r[..., 0::2] = x1 * cos - x2 * sin
        r[..., 1::2] = x2 * cos + x1 * sin
    else:
        x1, x2 = r[..., :half].clone(), r[..., half:].clone()
        r[..., :half] = x1 * cos - x2 * sin
        r[..., half:] = x2 * cos + x1 * sin
    return out


def ue8m0_quant(vals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row UE8M0 FP8 quant matching the kernel's fp8 quantization."""
    amax = vals.float().abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(amax, min=1e-4) / FP8_MAX
    scale = torch.exp2(torch.ceil(torch.log2(scale)))
    q = (vals.float() / scale).to(FP8)
    return q, scale.squeeze(-1)


def _bf16_ulp(a: torch.Tensor, b: torch.Tensor) -> int:
    def key(t):
        u = t.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
        return torch.where(u >= 0x8000, 0xFFFF - u, u + 0x8000)

    return int((key(a) - key(b)).abs().max().item())


def _fp8_ulp(a: torch.Tensor, b: torch.Tensor) -> int:
    def key(t):
        u = t.contiguous().view(torch.uint8).to(torch.int64)
        return torch.where(u >= 0x80, 0xFF - u, u + 0x80)

    return int((key(a) - key(b)).abs().max().item())


def assert_bf16(got: torch.Tensor, ref_fp32: torch.Tensor, msg: str):
    # The kernels keep norm/RoPE in fp32 and round to bf16 once; torch's
    # reduction/FMA order can flip a round-to-nearest tie. Same tolerance as the
    # vLLM DeepSeek-V3.2 fused-kernel tests.
    torch.testing.assert_close(
        got.float(), ref_fp32.float(), rtol=1e-2, atol=1e-2, msg=lambda m: f"{msg}: {m}"
    )


def assert_fp8(got: torch.Tensor, ref: torch.Tensor, msg: str):
    assert _fp8_ulp(got, ref) <= 1, f"{msg}: >1 fp8 ULP"


def _indexer_cache_views(cache: torch.Tensor, block_size: int, head_dim: int):
    """Split the packed [values | 4-byte scales] indexer cache of one block."""
    flat = cache.reshape(-1)
    vals = flat[: block_size * head_dim].view(FP8).view(block_size, head_dim)
    scales = flat[block_size * head_dim :].view(torch.float32)
    return vals, scales


def _build_case(
    cfg: ModelConfig,
    num_tokens: int,
    *,
    mla_dtype: str = "bfloat16",
    has_indexer: bool = True,
    index_interleave: bool = True,
    slot_is_negative: bool = False,
    with_outputs: bool = False,
    seed: int = 0,
):
    """Allocate the vLLM-shaped inputs and cache buffers for one case."""
    torch.manual_seed(seed)
    dev = "cuda"
    pos = torch.arange(num_tokens, device=dev, dtype=torch.int64) % MAX_POS

    q_c = torch.randn(num_tokens, cfg.q_lora, device=dev, dtype=torch.bfloat16)
    kv_c = torch.randn(num_tokens, cfg.kv_lora, device=dev, dtype=torch.bfloat16)
    k_pe = torch.randn(num_tokens, cfg.rope_dim, device=dev, dtype=torch.bfloat16)
    qw = torch.randn(cfg.q_lora, device=dev, dtype=torch.bfloat16)
    kvw = torch.randn(cfg.kv_lora, device=dev, dtype=torch.bfloat16)
    if has_indexer:
        ik = torch.randn(
            num_tokens, cfg.index_head_dim, device=dev, dtype=torch.bfloat16
        )
        ikw = torch.randn(cfg.index_head_dim, device=dev, dtype=torch.float32)
        ikb = torch.randn(cfg.index_head_dim, device=dev, dtype=torch.float32)
    else:
        ik = ikw = ikb = None

    mla_cos_sin = make_cos_sin(MAX_POS, cfg.rope_dim, dev)
    idx_cos_sin = make_cos_sin(MAX_POS, cfg.rope_dim, dev)

    block_size = MAX_POS
    mla_dim = cfg.kv_lora + cfg.rope_dim
    fp8_mla = mla_dtype == "fp8"
    if fp8_mla:
        mla_cache = torch.zeros(1, block_size, mla_dim, device=dev, dtype=torch.uint8)
        mla_k_scale = torch.tensor([0.3], device=dev, dtype=torch.float32)
    else:
        mla_cache = torch.zeros(
            1, block_size, mla_dim, device=dev, dtype=torch.bfloat16
        )
        mla_k_scale = None
    idx_row = cfg.index_head_dim + cfg.index_head_dim // 128 * 4
    if has_indexer:
        idx_cache = torch.zeros(1, block_size, idx_row, device=dev, dtype=torch.uint8)
    else:
        idx_cache = None
    slot = torch.arange(num_tokens, device=dev, dtype=torch.int64)
    if slot_is_negative:
        slot = torch.full_like(slot, -1)
    topk = torch.full((num_tokens, 2048), 7, device=dev, dtype=torch.int32)

    return {
        "pos": pos,
        "q_c": q_c,
        "kv_c": kv_c,
        "k_pe": k_pe,
        "qw": qw,
        "kvw": kvw,
        "ik": ik,
        "ikw": ikw,
        "ikb": ikb,
        "mla_cos_sin": mla_cos_sin,
        "idx_cos_sin": idx_cos_sin,
        "mla_cache": mla_cache,
        "mla_k_scale": mla_k_scale,
        "idx_cache": idx_cache,
        "slot": slot,
        "topk": topk,
        "idx_row": idx_row,
        "block_size": block_size,
        "mla_dtype": mla_dtype,
        "has_indexer": has_indexer,
        "index_interleave": index_interleave,
        "outputs": (
            {
                "q_c_out": torch.empty_like(q_c),
                "kv_c_out": torch.empty_like(kv_c),
                "k_pe_out": torch.empty_like(k_pe),
                "index_k_out": torch.empty_like(ik) if ik is not None else None,
            }
            if with_outputs
            else {
                "q_c_out": None,
                "kv_c_out": None,
                "k_pe_out": None,
                "index_k_out": None,
            }
        ),
    }


def _call_case(case, op=flaggems_vllm.fused_norm_rope):
    out = case["outputs"]
    return op(
        case["pos"],
        case["q_c"],
        case["qw"],
        EPS,
        case["kv_c"],
        case["kvw"],
        EPS,
        case["k_pe"],
        case["mla_cos_sin"],
        case["ik"],
        case["ikw"],
        case["ikb"],
        EPS,
        case["idx_cos_sin"],
        case["topk"],
        slot_mapping=case["slot"],
        indexer_k_cache=case["idx_cache"],
        mla_kv_cache=case["mla_cache"],
        mla_kv_cache_dtype=case["mla_dtype"],
        mla_k_scale=case["mla_k_scale"],
        has_indexer=case["has_indexer"],
        index_rope_interleave=case["index_interleave"],
        q_c_out=out["q_c_out"],
        kv_c_out=out["kv_c_out"],
        k_pe_out=out["k_pe_out"],
        index_k_out=out["index_k_out"],
    )


# ── accuracy ──────────────────────────────────────────────────────────────────


def _verify_main_case(case, cfg, num_tokens, mla_dtype, index_interleave, op):
    q_out = _call_case(case, op=op)

    assert_bf16(q_out, rms_norm(case["q_c"], case["qw"]), "q_c rmsnorm")

    # MLA cache: [kv_c_normed | k_pe_roped(interleaved)]
    kv_ref = rms_norm(case["kv_c"], case["kvw"])
    kpe_ref = rope(case["k_pe"].float(), case["pos"], case["mla_cos_sin"], True)
    if mla_dtype == "fp8":
        cache = case["mla_cache"].view(FP8)[0, :num_tokens]
        scale = case["mla_k_scale"].item()
        assert_fp8(cache[:, : cfg.kv_lora], (kv_ref / scale).to(FP8), "MLA kv fp8")
        assert_fp8(cache[:, cfg.kv_lora :], (kpe_ref / scale).to(FP8), "MLA k_pe fp8")
    else:
        cache = case["mla_cache"][0, :num_tokens]
        assert_bf16(cache[:, : cfg.kv_lora], kv_ref, "MLA kv bf16")
        assert_bf16(cache[:, cfg.kv_lora :], kpe_ref, "MLA k_pe bf16")

    # Indexer-K cache (packed [bs * head_dim fp8 | bs * 4 fp32 scale]).
    ik_ref = layer_norm(case["ik"], case["ikw"], case["ikb"])
    ik_ref = rope(ik_ref, case["pos"], case["idx_cos_sin"], index_interleave)
    q_ref, s_ref = ue8m0_quant(ik_ref)
    vals, scales = _indexer_cache_views(
        case["idx_cache"][0], case["block_size"], cfg.index_head_dim
    )
    assert_fp8(vals[:num_tokens], q_ref, "indexer-K fp8")
    torch.testing.assert_close(scales[:num_tokens], s_ref, rtol=0, atol=0)

    assert (case["topk"] == -1).all(), "topk buffer not cleared on indexer layer"


@pytest.mark.fused_norm_rope
@pytest.mark.parametrize("cfg", MODEL_CONFIGS, ids=MODEL_IDS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("index_interleave", [True, False])
@pytest.mark.parametrize("mla_dtype", ["bfloat16", "fp8"])
def test_fused_norm_rope(num_tokens, index_interleave, mla_dtype, cfg):
    case = _build_case(
        cfg, num_tokens, mla_dtype=mla_dtype, index_interleave=index_interleave
    )
    _verify_main_case(
        case,
        cfg,
        num_tokens,
        mla_dtype,
        index_interleave,
        flaggems_vllm.fused_norm_rope,
    )


@pytest.mark.fused_norm_rope
@pytest.mark.parametrize("cfg", MODEL_CONFIGS, ids=MODEL_IDS)
@pytest.mark.parametrize("num_tokens", [1, 17] if QUICK_MODE else [1, 17, 512])
@pytest.mark.parametrize("mla_dtype", ["bfloat16", "fp8"])
def test_fused_norm_rope_generic_impl(num_tokens, mla_dtype, cfg):
    """The portable (non-Hopper) implementation must match the same reference."""
    from flaggems_vllm.ops.fused_norm_rope import fused_norm_rope as generic_op

    case = _build_case(cfg, num_tokens, mla_dtype=mla_dtype, index_interleave=False)
    _verify_main_case(case, cfg, num_tokens, mla_dtype, False, generic_op)


@pytest.mark.fused_norm_rope
@pytest.mark.parametrize("cfg", MODEL_CONFIGS, ids=MODEL_IDS)
@pytest.mark.parametrize("num_tokens", [1, 17] if QUICK_MODE else [1, 17, 512])
def test_fused_norm_rope_no_indexer(num_tokens, cfg):
    """Shared (no-indexer) layer: q + kv/MLA only; top-k buffer untouched."""
    case = _build_case(cfg, num_tokens, has_indexer=False)
    q_out = _call_case(case)

    assert_bf16(q_out, rms_norm(case["q_c"], case["qw"]), "q_c rmsnorm (no-indexer)")
    cache = case["mla_cache"][0, :num_tokens]
    assert_bf16(cache[:, : cfg.kv_lora], rms_norm(case["kv_c"], case["kvw"]), "MLA kv")
    assert_bf16(
        cache[:, cfg.kv_lora :],
        rope(case["k_pe"].float(), case["pos"], case["mla_cos_sin"], True),
        "MLA k_pe",
    )
    assert (case["topk"] == 7).all(), "topk buffer should be untouched"


@pytest.mark.fused_norm_rope
@pytest.mark.parametrize("cfg", MODEL_CONFIGS, ids=MODEL_IDS)
@pytest.mark.parametrize("has_indexer", [False, True])
def test_fused_norm_rope_materializes_outputs(has_indexer, cfg):
    """PCP: local normalized/rotated rows without local cache slots."""
    num_tokens = 17
    case = _build_case(
        cfg,
        num_tokens,
        has_indexer=has_indexer,
        slot_is_negative=True,
        with_outputs=True,
        seed=6,
    )
    # vLLM skips the indexer program entirely when the MLA slot is negative, so
    # only Q/KV/K_PE are materialized on a non-owner rank.
    case["outputs"]["index_k_out"] = None
    q_out = _call_case(case)
    out = case["outputs"]

    assert q_out.data_ptr() == out["q_c_out"].data_ptr()
    assert_bf16(q_out, rms_norm(case["q_c"], case["qw"]), "PCP q norm")
    assert_bf16(out["kv_c_out"], rms_norm(case["kv_c"], case["kvw"]), "PCP kv norm")
    assert_bf16(
        out["k_pe_out"],
        rope(case["k_pe"].float(), case["pos"], case["mla_cos_sin"], True),
        "PCP k_pe RoPE",
    )
    assert not case["mla_cache"].any(), "non-owner rank wrote the MLA KV cache"
    if case["idx_cache"] is not None:
        assert not case["idx_cache"].any(), "non-owner rank wrote the indexer K cache"
    if has_indexer:
        assert (case["topk"] == -1).all(), "topk buffer not cleared on non-owner rank"
    else:
        assert (case["topk"] == 7).all(), "topk buffer should be untouched"


@pytest.mark.fused_norm_rope
@pytest.mark.parametrize("cfg", MODEL_CONFIGS, ids=MODEL_IDS)
@pytest.mark.parametrize("has_indexer", [False, True])
def test_fused_norm_rope_materializes_without_caches(has_indexer, cfg):
    """PCP without cache writes: every normalized/rotated row is materialized."""
    num_tokens = 17
    case = _build_case(cfg, num_tokens, has_indexer=has_indexer, with_outputs=True)
    out = case["outputs"]
    q_out = flaggems_vllm.fused_norm_rope(
        case["pos"],
        case["q_c"],
        case["qw"],
        EPS,
        case["kv_c"],
        case["kvw"],
        EPS,
        case["k_pe"],
        case["mla_cos_sin"],
        case["ik"],
        case["ikw"],
        case["ikb"],
        EPS,
        case["idx_cos_sin"],
        case["topk"],
        has_indexer=has_indexer,
        index_rope_interleave=True,
        q_c_out=out["q_c_out"],
        kv_c_out=out["kv_c_out"],
        k_pe_out=out["k_pe_out"],
        index_k_out=out["index_k_out"],
    )
    assert q_out.data_ptr() == out["q_c_out"].data_ptr()
    assert_bf16(q_out, rms_norm(case["q_c"], case["qw"]), "PCP q norm")
    assert_bf16(out["kv_c_out"], rms_norm(case["kv_c"], case["kvw"]), "PCP kv norm")
    assert_bf16(
        out["k_pe_out"],
        rope(case["k_pe"].float(), case["pos"], case["mla_cos_sin"], True),
        "PCP k_pe RoPE",
    )
    if has_indexer:
        ref = rope(
            layer_norm(case["ik"], case["ikw"], case["ikb"]),
            case["pos"],
            case["idx_cos_sin"],
            True,
        )
        assert_bf16(out["index_k_out"], ref, "PCP indexer-K")


@pytest.mark.fused_norm_rope
@pytest.mark.parametrize("cfg", MODEL_CONFIGS, ids=MODEL_IDS)
def test_fused_norm_rope_packed_indexer_block_stride(cfg):
    """Indexer writes use their HMA block stride and independent slot map."""
    dev = "cuda"
    num_tokens = block_size = 2
    torch.manual_seed(7)
    pos = torch.arange(num_tokens, device=dev, dtype=torch.int64)
    q_c = torch.randn(num_tokens, cfg.q_lora, device=dev, dtype=torch.bfloat16)
    kv_c = torch.randn(num_tokens, cfg.kv_lora, device=dev, dtype=torch.bfloat16)
    k_pe = torch.randn(num_tokens, cfg.rope_dim, device=dev, dtype=torch.bfloat16)
    qw = torch.randn(cfg.q_lora, device=dev, dtype=torch.bfloat16)
    kvw = torch.randn(cfg.kv_lora, device=dev, dtype=torch.bfloat16)
    ik = torch.randn(num_tokens, cfg.index_head_dim, device=dev, dtype=torch.bfloat16)
    ikw = torch.randn(cfg.index_head_dim, device=dev, dtype=torch.float32)
    ikb = torch.randn(cfg.index_head_dim, device=dev, dtype=torch.float32)
    cos_sin = make_cos_sin(32, cfg.rope_dim, dev)

    idx_row = cfg.index_head_dim + cfg.index_head_dim // 128 * 4
    packed_block_stride = block_size * idx_row + 64
    backing = torch.zeros(2 * packed_block_stride, device=dev, dtype=torch.uint8)
    idx_cache = torch.as_strided(
        backing, (2, block_size, idx_row), (packed_block_stride, idx_row, 1)
    )
    mla_cache = torch.zeros(
        1, block_size, cfg.kv_lora + cfg.rope_dim, device=dev, dtype=torch.bfloat16
    )
    topk = torch.zeros(num_tokens, 8, device=dev, dtype=torch.int32)

    flaggems_vllm.fused_norm_rope(
        pos,
        q_c,
        qw,
        EPS,
        kv_c,
        kvw,
        EPS,
        k_pe,
        cos_sin,
        ik,
        ikw,
        ikb,
        EPS,
        cos_sin,
        topk,
        slot_mapping=torch.arange(num_tokens, device=dev, dtype=torch.int64),
        indexer_slot_mapping=torch.arange(
            block_size, block_size + num_tokens, device=dev, dtype=torch.int64
        ),
        indexer_k_cache=idx_cache,
        mla_kv_cache=mla_cache,
        has_indexer=True,
    )

    ik_ref = rope(layer_norm(ik, ikw, ikb), pos, cos_sin, interleave=False)
    q_ref, s_ref = ue8m0_quant(ik_ref)
    packed = idx_cache[1].reshape(-1)
    values = packed[: block_size * cfg.index_head_dim].view(FP8)
    values = values.view(block_size, cfg.index_head_dim)
    scales = packed[block_size * cfg.index_head_dim :].view(torch.float32)
    assert_fp8(values, q_ref, "packed indexer-K fp8")
    torch.testing.assert_close(scales, s_ref, rtol=0, atol=0)
    assert (backing[block_size * idx_row : packed_block_stride] == 0).all()


@pytest.mark.fused_norm_rope
@pytest.mark.parametrize("cfg", MODEL_CONFIGS, ids=MODEL_IDS)
@pytest.mark.parametrize("num_tokens", [1, 17] if QUICK_MODE else [1, 4, 17, 512])
def test_fused_norm_rope_ds_mla(num_tokens, cfg):
    """fp8_ds_mla MLA cache layout: 512 fp8 NoPE | 4 f32 scales | 64 bf16 RoPE."""
    case = _build_case(cfg, num_tokens, has_indexer=False, seed=5)
    mla_cache = torch.zeros(1, MAX_POS, 656, device="cuda", dtype=torch.uint8)

    flaggems_vllm.fused_norm_rope(
        case["pos"],
        case["q_c"],
        case["qw"],
        EPS,
        case["kv_c"],
        case["kvw"],
        EPS,
        case["k_pe"],
        case["mla_cos_sin"],
        None,
        None,
        None,
        EPS,
        None,
        case["topk"],
        slot_mapping=case["slot"],
        indexer_k_cache=None,
        mla_kv_cache=mla_cache,
        mla_kv_cache_dtype="fp8_ds_mla",
        mla_k_scale=None,
        has_indexer=False,
    )

    kv_ref = rms_norm(case["kv_c"], case["kvw"])
    kpe_ref = rope(case["k_pe"].float(), case["pos"], case["mla_cos_sin"], True)
    tiles = kv_ref.view(num_tokens, 4, 128)
    ref_scale = torch.clamp(tiles.abs().amax(dim=-1) / FP8_MAX, min=1e-4)
    ref_scale = torch.exp2(torch.ceil(torch.log2(ref_scale)))
    ref_nope = (tiles / ref_scale[..., None]).reshape(num_tokens, cfg.kv_lora)
    ref_nope = ref_nope.to(FP8)

    cache = mla_cache[0, :num_tokens]
    nope = cache[:, : cfg.kv_lora].view(FP8)
    scales = cache.view(torch.float32)[:, cfg.kv_lora // 4 : cfg.kv_lora // 4 + 4]
    rope_off = cfg.kv_lora // 2 + 8
    rope_vals = cache.view(torch.bfloat16)[:, rope_off : rope_off + cfg.rope_dim]

    torch.testing.assert_close(scales, ref_scale, rtol=0, atol=0)
    assert_fp8(nope, ref_nope, "ds_mla NoPE fp8")
    assert_bf16(rope_vals, kpe_ref, "ds_mla RoPE bf16")
    assert (case["topk"] == 7).all(), "topk buffer should be untouched"


@pytest.mark.fused_norm_rope
def test_fused_norm_rope_shuffled_indexer_cache():
    """The shuffled indexer layout maps each row to its permuted byte offsets."""
    dev = "cuda"
    cfg = MODEL_CONFIGS[1]
    num_tokens = block_size = 16
    torch.manual_seed(11)
    pos = torch.arange(num_tokens, device=dev, dtype=torch.int64)
    q_c = torch.randn(num_tokens, cfg.q_lora, device=dev, dtype=torch.bfloat16)
    kv_c = torch.randn(num_tokens, cfg.kv_lora, device=dev, dtype=torch.bfloat16)
    k_pe = torch.randn(num_tokens, cfg.rope_dim, device=dev, dtype=torch.bfloat16)
    qw = torch.randn(cfg.q_lora, device=dev, dtype=torch.bfloat16)
    kvw = torch.randn(cfg.kv_lora, device=dev, dtype=torch.bfloat16)
    ik = torch.randn(num_tokens, cfg.index_head_dim, device=dev, dtype=torch.bfloat16)
    ikw = torch.randn(cfg.index_head_dim, device=dev, dtype=torch.float32)
    ikb = torch.randn(cfg.index_head_dim, device=dev, dtype=torch.float32)
    cos_sin = make_cos_sin(32, cfg.rope_dim, dev)
    idx_row = cfg.index_head_dim + 4
    idx_cache = torch.zeros(1, block_size, idx_row, device=dev, dtype=torch.uint8)
    mla_cache = torch.zeros(
        1, block_size, cfg.kv_lora + cfg.rope_dim, device=dev, dtype=torch.bfloat16
    )
    topk = torch.zeros(num_tokens, 8, device=dev, dtype=torch.int32)

    flaggems_vllm.fused_norm_rope(
        pos,
        q_c,
        qw,
        EPS,
        kv_c,
        kvw,
        EPS,
        k_pe,
        cos_sin,
        ik,
        ikw,
        ikb,
        EPS,
        cos_sin,
        topk,
        slot_mapping=torch.arange(num_tokens, device=dev, dtype=torch.int64),
        indexer_k_cache=idx_cache,
        mla_kv_cache=mla_cache,
        indexer_cache_shuffled=True,
        has_indexer=True,
    )

    ik_ref = rope(layer_norm(ik, ikw, ikb), pos, cos_sin, interleave=False)
    q_ref, s_ref = ue8m0_quant(ik_ref)
    flat = idx_cache[0].reshape(-1)
    head_dim = cfg.index_head_dim
    block_tile, head_tile = 16, 16
    offs = torch.arange(head_dim, device=dev)
    row = torch.arange(block_size, device=dev)[:, None]
    value_off = (
        (row // block_tile) * block_tile * head_dim
        + (row % block_tile) * head_tile
        + (offs[None, :] // head_tile) * block_tile * head_tile
        + (offs[None, :] % head_tile)
    )
    got = flat[value_off.reshape(-1)].view(FP8).view(block_size, head_dim)
    assert_fp8(got, q_ref, "shuffled indexer-K fp8")
    scales = flat[block_size * head_dim :].view(torch.float32)
    torch.testing.assert_close(scales, s_ref, rtol=0, atol=0)


@pytest.mark.fused_norm_rope
def test_fused_norm_rope_repeated_calls():
    """The Hopper launch cache must read live buffers, not a snapshot."""
    cfg = MODEL_CONFIGS[1]
    num_tokens = 17
    case = _build_case(cfg, num_tokens, seed=13)
    _call_case(case)

    # Same tensors, mutated contents: the second call must see the new values.
    new_q = torch.randn_like(case["q_c"])
    new_kv = torch.randn_like(case["kv_c"])
    case["q_c"].copy_(new_q)
    case["kv_c"].copy_(new_kv)
    q_out = _call_case(case)
    assert_bf16(q_out, rms_norm(new_q, case["qw"]), "cached q norm")
    cache = case["mla_cache"][0, :num_tokens]
    assert_bf16(
        cache[:, : cfg.kv_lora],
        rms_norm(new_kv, case["kvw"]),
        "cached MLA kv",
    )

    # A caller-provided output buffer is patched into the cached launch.
    out_buf = torch.empty_like(case["q_c"])
    case["outputs"]["q_c_out"] = out_buf
    returned = _call_case(case)
    assert returned.data_ptr() == out_buf.data_ptr()
    assert_bf16(out_buf, rms_norm(new_q, case["qw"]), "patched q_c_out")


@pytest.mark.fused_norm_rope
def test_fused_norm_rope_empty_and_unsupported():
    cfg = MODEL_CONFIGS[1]
    case = _build_case(cfg, 0)
    q_out = _call_case(case)
    assert q_out.shape == (0, cfg.q_lora)

    case = _build_case(cfg, 4)
    with pytest.raises(NotImplementedError):
        flaggems_vllm.fused_norm_rope(
            case["pos"],
            case["q_c"],
            case["qw"],
            EPS,
            case["kv_c"],
            case["kvw"],
            EPS,
            case["k_pe"],
            case["mla_cos_sin"],
            case["ik"],
            case["ikw"],
            case["ikb"],
            EPS,
            case["idx_cos_sin"],
            case["topk"],
            slot_mapping=case["slot"],
            mla_kv_cache=case["mla_cache"],
            mla_kv_cache_dtype="nvfp4_ds_mla",
        )


# ── parity with vLLM 0.32 ─────────────────────────────────────────────────────


@pytest.mark.fused_norm_rope
@pytest.mark.skipif(
    vllm_fused_norm_rope is None,
    reason="vLLM 0.32 fused_norm_rope unavailable",
)
@pytest.mark.parametrize("cfg", MODEL_CONFIGS, ids=MODEL_IDS)
@pytest.mark.parametrize("num_tokens", [1, 17] if QUICK_MODE else [1, 17, 512])
@pytest.mark.parametrize("mla_dtype", ["bfloat16", "fp8"])
@pytest.mark.parametrize("index_interleave", [True, False])
def test_fused_norm_rope_matches_vllm(num_tokens, mla_dtype, index_interleave, cfg):
    gems_case = _build_case(
        cfg, num_tokens, mla_dtype=mla_dtype, index_interleave=index_interleave
    )
    ref_case = _build_case(
        cfg, num_tokens, mla_dtype=mla_dtype, index_interleave=index_interleave
    )

    gems_q = _call_case(gems_case)
    vllm_q = _call_case(ref_case, op=vllm_fused_norm_rope)

    assert _bf16_ulp(gems_q, vllm_q) <= 1, "q_c rmsnorm differs from vLLM"
    if mla_dtype == "fp8":
        gems = gems_case["mla_cache"].view(FP8)[0, :num_tokens]
        ref = ref_case["mla_cache"].view(FP8)[0, :num_tokens]
        assert_fp8(gems[:, : cfg.kv_lora], ref[:, : cfg.kv_lora], "MLA kv vs vLLM")
        assert_fp8(gems[:, cfg.kv_lora :], ref[:, cfg.kv_lora :], "MLA k_pe vs vLLM")
    else:
        gems = gems_case["mla_cache"][0, :num_tokens]
        ref = ref_case["mla_cache"][0, :num_tokens]
        assert _bf16_ulp(gems, ref) <= 1, "MLA cache differs from vLLM"

    head_dim = cfg.index_head_dim
    gems_vals, gems_scales = _indexer_cache_views(
        gems_case["idx_cache"][0], gems_case["block_size"], head_dim
    )
    ref_vals, ref_scales = _indexer_cache_views(
        ref_case["idx_cache"][0], ref_case["block_size"], head_dim
    )
    assert_fp8(gems_vals[:num_tokens], ref_vals[:num_tokens], "indexer-K vs vLLM")
    torch.testing.assert_close(
        gems_scales[:num_tokens], ref_scales[:num_tokens], rtol=0, atol=0
    )
    assert torch.equal(gems_case["topk"], ref_case["topk"])
