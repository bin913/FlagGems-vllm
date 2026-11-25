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

"""Functional tests for ``moe_wna16_marlin_gemm``.

Weights come from vLLM's own Marlin quantizers, the token list from vLLM's
``moe_align_block_size``, and each case is checked twice:

  * against a PyTorch reference built from the dequantized (``w_ref``) experts;
  * against vLLM 0.32.0's ``moe_wna16_marlin_gemm`` when that kernel accepts
    the configuration (its CUDA thread-config heuristic rejects some small
    8-bit shapes).

Shapes follow vLLM's ``MARLIN_MOE_SCENARIOS`` and ``FUSED_MOE_WN16_MNK_FACTORS``.
"""

import importlib

import pytest
import torch

import flaggems_vllm
from flaggems_vllm import moe_wna16_marlin_gemm

from . import conftest as cfg
from .accuracy_utils import gems_assert_close

CUDA_AVAILABLE = flaggems_vllm.device == "cuda" and torch.cuda.is_available()
pytestmark = [pytest.mark.moe_wna16_marlin_gemm]

try:
    from vllm import _custom_ops as _vllm_ops
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_bias,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        awq_marlin_quantize,
        marlin_quantize,
    )
    from vllm.scalar_type import scalar_types

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

# (m, n, k, e, topk): n/k are the GEMM's N and K, not the model's intermediate
# size, so these cover the gate/up and down projections of vLLM's scenarios.
SCENARIOS = [
    (1, 128, 256, 5, 2),
    (1, 1024, 2048, 5, 2),
    (133, 256, 256, 5, 2),
    (133, 1024, 2048, 12, 3),
    (128, 256, 256, 5, 2),
    (128, 1024, 2048, 12, 3),
]
QUICK_SCENARIOS = SCENARIOS[:2]

QUANT_CONFIGS = [
    ("uint4b8", 128),
    ("uint8b128", 64),
    ("uint4", 128),
]
QUICK_QUANT_CONFIGS = QUANT_CONFIGS[:1]

DTYPES = [torch.bfloat16, torch.float16]
QUICK_DTYPES = [torch.bfloat16]

ATOL = 4e-2


def _skip_if_unavailable():
    if not CUDA_AVAILABLE:
        pytest.skip("requires a CUDA device")
    if not HAS_VLLM:
        pytest.skip("requires vLLM")


def _scalar_type(name):
    return getattr(scalar_types, name)


def _make_weights(W, quant_type, group_size):
    """Quantize (E, N, K) expert weights with vLLM's Marlin repack."""
    has_zp = quant_type in (scalar_types.uint4, scalar_types.uint8)
    qweights, scales, zeros, refs = [], [], [], []
    for expert in range(W.shape[0]):
        # vLLM quantizes (size_k, size_n) and groups along size_k.
        if has_zp:
            ref, qweight, scale, zp = awq_marlin_quantize(
                W[expert].t().contiguous(), quant_type, group_size
            )
            zeros.append(zp)
        else:
            ref, qweight, scale = marlin_quantize(
                W[expert].t().contiguous(), quant_type, group_size
            )
        qweights.append(qweight)
        scales.append(scale)
        refs.append(ref.t())
    return (
        torch.stack(qweights).contiguous(),
        torch.stack(scales).contiguous(),
        torch.stack(zeros).contiguous() if zeros else None,
        torch.stack(refs),
    )


def _routing(m, num_experts, topk, block, expert_map=None):
    score = torch.randn(m, num_experts, device="cuda")
    topk_weights, topk_ids = torch.topk(
        torch.softmax(score.float(), dim=-1), topk, dim=-1
    )
    sorted_ids, expert_ids, num_post = moe_align_block_size(
        topk_ids.to(torch.int32),
        block,
        num_experts,
        expert_map,
        ignore_invalid_experts=True,
    )
    return topk_weights.float(), sorted_ids, expert_ids, num_post


def _torch_reference(
    a,
    refs,
    topk_weights,
    sorted_ids,
    expert_ids,
    num_post,
    block,
    topk,
    mul_topk,
    bias_ref=None,
):
    m = a.shape[0]
    n = refs.shape[1]
    out = torch.zeros(m * topk, n, device="cuda", dtype=torch.float32)
    for blk in range(int(num_post.item()) // block):
        expert = int(expert_ids[blk].item())
        if expert < 0:
            continue
        tids = sorted_ids[blk * block : (blk + 1) * block].long()
        rows = tids[tids < m * topk]
        if rows.numel() == 0:
            continue
        cur = a[rows // topk].float() @ refs[expert].t().float()
        if bias_ref is not None:
            cur = cur + bias_ref[expert].float()[None, :]
        if mul_topk:
            cur = cur * topk_weights.reshape(-1)[rows][:, None]
        out[rows] = cur
    return out


def _call(
    a,
    b_qweight,
    b_bias,
    b_scales,
    b_q_zeros,
    sorted_ids,
    expert_ids,
    num_post,
    topk_weights,
    block,
    topk,
    mul_topk,
    quant_type,
    m,
    n,
    k,
    op=None,
):
    workspace = torch.zeros(8192, dtype=torch.int32, device="cuda")
    return (op or moe_wna16_marlin_gemm)(
        a,
        None,
        b_qweight,
        b_bias,
        b_scales,
        None,
        None,
        b_q_zeros,
        workspace,
        sorted_ids,
        expert_ids,
        num_post,
        topk_weights,
        moe_block_size=block,
        top_k=topk,
        mul_topk_weights=mul_topk,
        b_q_type=quant_type,
        size_m=m,
        size_n=n,
        size_k=k,
        use_atomic_add=False,
        use_fp32_reduce=True,
        is_zp_float=False,
    )


@pytest.mark.parametrize(
    "scenario", SCENARIOS if not cfg.QUICK_MODE else QUICK_SCENARIOS
)
@pytest.mark.parametrize(
    "quant,group_size",
    QUANT_CONFIGS if not cfg.QUICK_MODE else QUICK_QUANT_CONFIGS,
)
@pytest.mark.parametrize("dtype", DTYPES if not cfg.QUICK_MODE else QUICK_DTYPES)
def test_moe_wna16_marlin_gemm(scenario, quant, group_size, dtype):
    _skip_if_unavailable()
    torch.manual_seed(0)
    m, n, k, num_experts, topk = scenario
    block = 8 if m == 1 else 16

    W = torch.randn(num_experts, n, k, device="cuda", dtype=dtype) / 10
    b_qweight, b_scales, b_q_zeros, refs = _make_weights(
        W, _scalar_type(quant), group_size
    )
    a = torch.randn(m, k, device="cuda", dtype=dtype) / 10
    topk_weights, sorted_ids, expert_ids, num_post = _routing(
        m, num_experts, topk, block
    )

    out = _call(
        a,
        b_qweight,
        None,
        b_scales,
        b_q_zeros,
        sorted_ids,
        expert_ids,
        num_post,
        topk_weights,
        block,
        topk,
        False,
        _scalar_type(quant),
        m,
        n,
        k,
    )
    ref = _torch_reference(
        a,
        refs,
        topk_weights,
        sorted_ids,
        expert_ids,
        num_post,
        block,
        topk,
        False,
    )
    gems_assert_close(out, ref, dtype, atol=ATOL)

    try:
        baseline = _vllm_ops.moe_wna16_marlin_gemm(
            a,
            None,
            b_qweight,
            None,
            b_scales,
            None,
            None,
            b_q_zeros,
            torch.zeros(8192, dtype=torch.int32, device="cuda"),
            sorted_ids,
            expert_ids,
            num_post,
            topk_weights,
            moe_block_size=block,
            top_k=topk,
            mul_topk_weights=False,
            b_q_type=_scalar_type(quant),
            size_m=m,
            size_n=n,
            size_k=k,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        )
    except RuntimeError:
        # vLLM's CUDA thread-config heuristic rejects some small 8-bit shapes.
        baseline = None
    if baseline is not None:
        gems_assert_close(out, baseline, dtype, atol=ATOL)


@pytest.mark.parametrize("mul_topk", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_moe_wna16_marlin_gemm_epilogue(mul_topk, with_bias):
    _skip_if_unavailable()
    torch.manual_seed(1)
    m, n, k, num_experts, topk, block = 128, 512, 256, 5, 2, 16
    dtype = torch.bfloat16
    quant = scalar_types.uint4b8

    W = torch.randn(num_experts, n, k, device="cuda", dtype=dtype) / 10
    b_qweight, b_scales, _, refs = _make_weights(W, quant, 128)
    bias = torch.randn(num_experts, n, device="cuda", dtype=dtype) / 10
    marlin_bias = torch.stack(
        [marlin_permute_bias(bias[i]) for i in range(num_experts)]
    )
    a = torch.randn(m, k, device="cuda", dtype=dtype) / 10
    topk_weights, sorted_ids, expert_ids, num_post = _routing(
        m, num_experts, topk, block
    )

    out = _call(
        a,
        b_qweight,
        marlin_bias if with_bias else None,
        b_scales,
        None,
        sorted_ids,
        expert_ids,
        num_post,
        topk_weights,
        block,
        topk,
        mul_topk,
        quant,
        m,
        n,
        k,
    )
    ref = _torch_reference(
        a,
        refs,
        topk_weights,
        sorted_ids,
        expert_ids,
        num_post,
        block,
        topk,
        mul_topk,
        bias if with_bias else None,
    )
    gems_assert_close(out, ref, dtype, atol=ATOL)


@pytest.mark.parametrize("block", [8, 32, 64])
def test_moe_wna16_marlin_gemm_block_sizes(block):
    _skip_if_unavailable()
    torch.manual_seed(2)
    m, n, k, num_experts, topk = 64, 256, 256, 4, 2
    dtype = torch.bfloat16
    quant = scalar_types.uint4

    W = torch.randn(num_experts, n, k, device="cuda", dtype=dtype) / 10
    b_qweight, b_scales, b_q_zeros, refs = _make_weights(W, quant, 64)
    a = torch.randn(m, k, device="cuda", dtype=dtype) / 10
    topk_weights, sorted_ids, expert_ids, num_post = _routing(
        m, num_experts, topk, block
    )
    out = _call(
        a,
        b_qweight,
        None,
        b_scales,
        b_q_zeros,
        sorted_ids,
        expert_ids,
        num_post,
        topk_weights,
        block,
        topk,
        False,
        quant,
        m,
        n,
        k,
    )
    ref = _torch_reference(
        a,
        refs,
        topk_weights,
        sorted_ids,
        expert_ids,
        num_post,
        block,
        topk,
        False,
    )
    gems_assert_close(out, ref, dtype, atol=ATOL)


def test_moe_wna16_marlin_gemm_expert_parallel():
    """Blocks owned by another expert-parallel rank carry expert id -1."""
    _skip_if_unavailable()
    torch.manual_seed(3)
    m, n, k, num_experts, topk, block = 64, 256, 256, 8, 2, 16
    dtype = torch.bfloat16
    quant = scalar_types.uint4b8
    ep_size = 4
    local_experts = num_experts // ep_size

    ids = torch.randperm(num_experts, device="cuda")[:local_experts].to(torch.int32)
    expert_map = torch.full((num_experts,), -1, device="cuda", dtype=torch.int32)
    expert_map[ids] = torch.arange(local_experts, device="cuda", dtype=torch.int32)

    W = torch.randn(local_experts, n, k, device="cuda", dtype=dtype) / 10
    b_qweight, b_scales, _, refs = _make_weights(W, quant, 128)
    a = torch.randn(m, k, device="cuda", dtype=dtype) / 10
    topk_weights, sorted_ids, expert_ids, num_post = _routing(
        m, num_experts, topk, block, expert_map
    )
    out = _call(
        a,
        b_qweight,
        None,
        b_scales,
        None,
        sorted_ids,
        expert_ids,
        num_post,
        topk_weights,
        block,
        topk,
        False,
        quant,
        m,
        n,
        k,
    )
    ref = _torch_reference(
        a,
        refs,
        topk_weights,
        sorted_ids,
        expert_ids,
        num_post,
        block,
        topk,
        False,
    )
    # Tokens routed to non-local experts are dropped from the routing, so only
    # the covered rows are defined by the operator.
    covered = torch.unique(sorted_ids[: int(num_post.item())].long())
    covered = covered[covered < m * topk]
    gems_assert_close(out[covered], ref[covered], dtype, atol=ATOL)

    try:
        baseline = _vllm_ops.moe_wna16_marlin_gemm(
            a,
            None,
            b_qweight,
            None,
            b_scales,
            None,
            None,
            None,
            torch.zeros(8192, dtype=torch.int32, device="cuda"),
            sorted_ids,
            expert_ids,
            num_post,
            topk_weights,
            moe_block_size=block,
            top_k=topk,
            mul_topk_weights=False,
            b_q_type=quant,
            size_m=m,
            size_n=n,
            size_k=k,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        )
    except RuntimeError:
        baseline = None
    if baseline is not None:
        gems_assert_close(out[covered], baseline[covered], dtype, atol=ATOL)


def test_moe_wna16_marlin_gemm_cached_weights():
    """The Hopper weight cache must not leak stale results across calls."""
    _skip_if_unavailable()
    torch.manual_seed(4)
    m, n, k, num_experts, topk, block = 32, 256, 256, 4, 2, 8
    dtype = torch.bfloat16
    quant = scalar_types.uint4b8

    W = torch.randn(num_experts, n, k, device="cuda", dtype=dtype) / 10
    b_qweight, b_scales, _, refs = _make_weights(W, quant, 128)
    a = torch.randn(m, k, device="cuda", dtype=dtype) / 10
    topk_weights, sorted_ids, expert_ids, num_post = _routing(
        m, num_experts, topk, block
    )

    def run():
        return _call(
            a,
            b_qweight,
            None,
            b_scales,
            None,
            sorted_ids,
            expert_ids,
            num_post,
            topk_weights,
            block,
            topk,
            False,
            quant,
            m,
            n,
            k,
        )

    first = run()
    second = run()
    assert torch.equal(first, second)
    ref = _torch_reference(
        a,
        refs,
        topk_weights,
        sorted_ids,
        expert_ids,
        num_post,
        block,
        topk,
        False,
    )
    gems_assert_close(second, ref, dtype, atol=ATOL)


def test_moe_wna16_marlin_gemm_empty():
    """A zero-token call returns the empty output without touching weights."""
    _skip_if_unavailable()
    n, k = 256, 256
    quant = scalar_types.uint4b8
    W = torch.randn(2, n, k, device="cuda", dtype=torch.bfloat16) / 10
    b_qweight, b_scales, _, _ = _make_weights(W, quant, 128)
    a = torch.empty(0, k, device="cuda", dtype=torch.bfloat16)
    empty_ids = torch.empty(0, dtype=torch.int32, device="cuda")
    num_post = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = _call(
        a,
        b_qweight,
        None,
        b_scales,
        None,
        empty_ids,
        empty_ids,
        num_post,
        torch.empty(0, device="cuda"),
        16,
        2,
        False,
        quant,
        0,
        n,
        k,
    )
    assert out.shape == (0, n)


def test_moe_wna16_marlin_gemm_unsupported_activation():
    _skip_if_unavailable()
    n, k = 128, 256
    quant = scalar_types.uint4b8
    W = torch.randn(2, n, k, device="cuda", dtype=torch.bfloat16) / 10
    b_qweight, b_scales, _, _ = _make_weights(W, quant, 128)
    a = torch.randn(4, k, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    empty_ids = torch.zeros(64, dtype=torch.int32, device="cuda")
    num_post = torch.zeros(1, dtype=torch.int32, device="cuda")
    with pytest.raises(NotImplementedError):
        _call(
            a,
            b_qweight,
            None,
            b_scales,
            None,
            empty_ids,
            empty_ids,
            num_post,
            torch.zeros(8, device="cuda"),
            8,
            2,
            False,
            quant,
            4,
            n,
            k,
        )


@pytest.mark.parametrize("quant,group_size", [("uint4b8", 128), ("uint4", 64)])
def test_moe_wna16_marlin_gemm_generic_impl(quant, group_size):
    """The arch-independent fused implementation (used off Hopper) must agree."""
    _skip_if_unavailable()
    generic = importlib.import_module("flaggems_vllm.ops.moe_wna16_marlin_gemm")
    torch.manual_seed(7)
    m, n, k, num_experts, topk, block = 32, 256, 256, 4, 2, 16
    dtype = torch.bfloat16
    quant_type = _scalar_type(quant)

    W = torch.randn(num_experts, n, k, device="cuda", dtype=dtype) / 10
    b_qweight, b_scales, b_q_zeros, refs = _make_weights(W, quant_type, group_size)
    a = torch.randn(m, k, device="cuda", dtype=dtype) / 10
    topk_weights, sorted_ids, expert_ids, num_post = _routing(
        m, num_experts, topk, block
    )
    out = _call(
        a,
        b_qweight,
        None,
        b_scales,
        b_q_zeros,
        sorted_ids,
        expert_ids,
        num_post,
        topk_weights,
        block,
        topk,
        False,
        quant_type,
        m,
        n,
        k,
        op=generic.moe_wna16_marlin_gemm,
    )
    ref = _torch_reference(
        a,
        refs,
        topk_weights,
        sorted_ids,
        expert_ids,
        num_post,
        block,
        topk,
        False,
    )
    gems_assert_close(out, ref, dtype, atol=ATOL)
