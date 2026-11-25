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

"""Benchmark ``moe_wna16_marlin_gemm`` against vLLM 0.32.0.

Both sides receive identical Marlin-repacked experts and the same
``moe_align_block_size`` routing, so this is kernel-for-kernel: vLLM's Marlin
CUDA kernel versus FlagGems-vllm (the H100 specialization dequantizes the
experts once into a bounded cache, then runs a plain grouped GEMM).

Shapes are taken from vLLM's ``MARLIN_MOE_SCENARIOS`` /
``FUSED_MOE_WN16_MNK_FACTORS`` plus one dense MoE layer from a DeepSeek-sized
model ``(hidden 4096, 14336 intermediate)``.
"""

import pytest
import torch

import flaggems_vllm

from . import base

try:
    from vllm import _custom_ops as _vllm_ops
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        awq_marlin_quantize,
        marlin_quantize,
    )
    from vllm.scalar_type import scalar_types

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

CUDA_AVAILABLE = flaggems_vllm.device == "cuda" and torch.cuda.is_available()
DTYPES = [torch.bfloat16, torch.float16]

# (m, n, k, e, topk, block, quant_type, group_size)
CASES = [
    (1, 128, 128, 8, 2, 8, "uint4b8", 128),
    (1, 1024, 1024, 8, 2, 16, "uint4b8", 128),
    (32, 2048, 128, 8, 2, 16, "uint4b8", 128),
    (222, 2048, 1024, 8, 2, 32, "uint4b8", 128),
    (128, 512, 256, 5, 2, 16, "uint4b8", 128),
    (133, 2048, 1024, 12, 3, 16, "uint4b8", 128),
    (128, 2048, 1024, 12, 3, 32, "uint4", 128),
    (222, 2048, 1024, 8, 2, 32, "uint4", 128),
    (512, 14336, 4096, 8, 2, 32, "uint4", 128),
]
CORE_CASES = CASES[:5]


def _make_inputs(case, dtype):
    m, n, k, num_experts, topk, block, quant, group_size = case
    quant_type = getattr(scalar_types, quant)
    has_zp = quant_type in (scalar_types.uint4, scalar_types.uint8)

    torch.manual_seed(0)
    W = torch.randn(num_experts, n, k, device="cuda", dtype=dtype) / 10
    qweights, scales, zeros = [], [], []
    for expert in range(num_experts):
        if has_zp:
            _ref, qweight, scale, zp = awq_marlin_quantize(
                W[expert].t().contiguous(), quant_type, group_size
            )
            zeros.append(zp)
        else:
            _ref, qweight, scale = marlin_quantize(
                W[expert].t().contiguous(), quant_type, group_size
            )
        qweights.append(qweight)
        scales.append(scale)
    b_qweight = torch.stack(qweights).contiguous()
    b_scales = torch.stack(scales).contiguous()
    b_q_zeros = torch.stack(zeros).contiguous() if zeros else None

    a = torch.randn(m, k, device="cuda", dtype=dtype) / 10
    score = torch.randn(m, num_experts, device="cuda")
    topk_weights, topk_ids = torch.topk(
        torch.softmax(score.float(), dim=-1), topk, dim=-1
    )
    sorted_ids, expert_ids, num_post = moe_align_block_size(
        topk_ids.to(torch.int32), block, num_experts, None, ignore_invalid_experts=True
    )
    workspace = torch.zeros(8192, dtype=torch.int32, device="cuda")
    kwargs = dict(
        moe_block_size=block,
        top_k=topk,
        mul_topk_weights=False,
        b_q_type=quant_type,
        size_m=m,
        size_n=n,
        size_k=k,
        use_atomic_add=False,
        use_fp32_reduce=True,
        is_zp_float=False,
    )
    return (
        a,
        b_qweight,
        None,
        b_scales,
        b_q_zeros,
        workspace,
        sorted_ids,
        expert_ids,
        num_post,
        topk_weights.float(),
        kwargs,
    )


def _vllm_moe(
    a,
    b_qweight,
    b_bias,
    b_scales,
    b_q_zeros,
    workspace,
    sorted_ids,
    expert_ids,
    num_post,
    topk_weights,
    **kwargs,
):
    return _vllm_ops.moe_wna16_marlin_gemm(
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
        **kwargs,
    )


def _gems_moe(
    a,
    b_qweight,
    b_bias,
    b_scales,
    b_q_zeros,
    workspace,
    sorted_ids,
    expert_ids,
    num_post,
    topk_weights,
    **kwargs,
):
    return flaggems_vllm.moe_wna16_marlin_gemm(
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
        **kwargs,
    )


class MoeWna16MarlinGemmBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            op_name="moe_wna16_marlin_gemm",
            torch_op=_vllm_moe,
            gems_op=_gems_moe,
            dtypes=DTYPES,
        )

    def set_shapes(self, shape_file_path=None):
        core = getattr(base.Config, "bench_level", None) == base.consts.BenchLevel.CORE
        self.shapes = CORE_CASES if core else CASES
        self.shape_desc = "M, N, K, E, topK, block, quant, group_size"

    def get_input_iter(self, dtype):
        for case in self.shapes:
            yield _make_inputs(case, dtype)


@pytest.mark.moe_wna16_marlin_gemm
@pytest.mark.skipif(
    not (CUDA_AVAILABLE and HAS_VLLM), reason="requires CUDA and vLLM 0.32"
)
def test_moe_wna16_marlin_gemm_benchmark():
    MoeWna16MarlinGemmBenchmark().run()
