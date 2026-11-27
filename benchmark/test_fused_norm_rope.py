# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Performance tests for ``fused_norm_rope`` against vLLM 0.32.

Baseline: ``vllm.models.deepseek_v32.common.kernels.fused_norm_rope`` (the vLLM
0.32 Triton kernel). Candidate: ``flaggems_vllm.fused_norm_rope`` (Hopper
implementation on SM90). Shapes follow the vLLM DeepSeek-V3.2 kernel tests:
GLM-5.2 / DeepSeek-V3.2 dims over the token counts of its ``num_tokens``
parametrization.
"""

import pytest
import torch

import flaggems_vllm

from . import base, consts

try:
    from vllm.models.deepseek_v32.common.kernels import (
        fused_norm_rope as vllm_fused_norm_rope,
    )
except Exception:  # pragma: no cover - vLLM without DeepSeek-V3.2 kernels
    vllm_fused_norm_rope = None

EPS = 1e-6
MAX_POS = 8192

pytestmark = pytest.mark.skipif(
    vllm_fused_norm_rope is None, reason="vLLM 0.32 fused_norm_rope unavailable"
)


def make_cos_sin(max_pos, rot_dim, device):
    half = rot_dim // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, half, dtype=torch.float32, device=device) / half)
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


def make_inputs(shape, device, mla_dtype="bfloat16"):
    """vLLM-shaped inputs; the trailing dict becomes the op kwargs.

    The operator keeps one interface for both MLA-cache write paths. The
    benchmark times each path separately under the same op name, selecting the
    baseline row with ``Benchmark.base_variant`` ("bf16" / "fp8").
    """
    num_tokens, q_lora, kv_lora, rope_dim, index_dim = shape
    pos = torch.arange(num_tokens, device=device, dtype=torch.int64) % MAX_POS
    q_c = torch.randn(num_tokens, q_lora, device=device, dtype=torch.bfloat16)
    kv_c = torch.randn(num_tokens, kv_lora, device=device, dtype=torch.bfloat16)
    k_pe = torch.randn(num_tokens, rope_dim, device=device, dtype=torch.bfloat16)
    qw = torch.randn(q_lora, device=device, dtype=torch.bfloat16)
    kvw = torch.randn(kv_lora, device=device, dtype=torch.bfloat16)
    ik = torch.randn(num_tokens, index_dim, device=device, dtype=torch.bfloat16)
    ikw = torch.randn(index_dim, device=device, dtype=torch.float32)
    ikb = torch.randn(index_dim, device=device, dtype=torch.float32)
    cos_sin = make_cos_sin(MAX_POS, rope_dim, device)
    idx_cos_sin = make_cos_sin(MAX_POS, rope_dim, device)

    # Single decode/prefill block covering every token, like the vLLM tests.
    block_size = MAX_POS
    mla_dim = kv_lora + rope_dim
    if mla_dtype == "fp8":
        mla_cache = torch.zeros(
            1, block_size, mla_dim, device=device, dtype=torch.uint8
        )
        mla_k_scale = torch.tensor([0.3], device=device, dtype=torch.float32)
    else:
        mla_cache = torch.zeros(
            1, block_size, mla_dim, device=device, dtype=torch.bfloat16
        )
        mla_k_scale = None
    idx_row = index_dim + index_dim // 128 * 4
    idx_cache = torch.zeros(1, block_size, idx_row, device=device, dtype=torch.uint8)
    slot = torch.arange(num_tokens, device=device, dtype=torch.int64)
    topk = torch.empty(num_tokens, 2048, device=device, dtype=torch.int32)

    args = (
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
        idx_cos_sin,
        topk,
    )
    kwargs = dict(
        slot_mapping=slot,
        indexer_k_cache=idx_cache,
        mla_kv_cache=mla_cache,
        mla_kv_cache_dtype=mla_dtype,
        mla_k_scale=mla_k_scale,
        has_indexer=True,
        index_rope_interleave=False,
    )
    return (*args, kwargs)


class FusedNormRopeBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:]
    mla_dtype = "bfloat16"

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            yield make_inputs(shape, self.device, self.mla_dtype)


@pytest.mark.fused_norm_rope
def test_perf_fused_norm_rope():
    """Default BF16 MLA-cache write path."""
    bench = FusedNormRopeBenchmark(
        op_name="fused_norm_rope",
        torch_op=vllm_fused_norm_rope,
        gems_op=flaggems_vllm.fused_norm_rope,
        dtypes=[torch.bfloat16],
    )
    bench.base_variant = "bf16"
    bench.run()


@pytest.mark.fused_norm_rope
def test_perf_fused_norm_rope_fp8_cache():
    """Per-tensor FP8 MLA-cache write path of the same operator interface."""
    bench = FusedNormRopeBenchmark(
        op_name="fused_norm_rope",
        torch_op=vllm_fused_norm_rope,
        gems_op=flaggems_vllm.fused_norm_rope,
        dtypes=[torch.bfloat16],
        mla_dtype="fp8",
    )
    bench.base_variant = "fp8"
    bench.run()
