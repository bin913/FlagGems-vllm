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

"""fused_norm_rope baseline, aligned with the FlagGems-vllm benchmark.

native: ``vllm.models.deepseek_v32.common.kernels.fused_norm_rope`` (vLLM
        0.32.0), the fused Q/KV/indexer norm + RoPE + cache-insert kernel.

Inputs are built exactly like
``FlagGems-vllm/benchmark/test_fused_norm_rope.py::make_inputs`` (same tensors,
dtypes and cache layouts). One operator interface serves both MLA-cache write
paths, selected by ``mla_kv_cache_dtype``; this module collects both as
benchmark variants and tags them with ``mla_dtype``:

  * ``bfloat16`` -- default BF16 MLA cache, shape key ``[tokens, q_lora, ...]``
  * ``fp8``      -- per-tensor FP8 MLA cache (uint8 cache + FP32 scale tensor),
                    shape key ``fp8_<tokens>x<q_lora>x<kv>x<rope>x<idx>``

The ``fp8_`` prefix keeps both variants in the collector's flat ``shapes`` dict;
the benchmark's baseline file nests them under
``fused_norm_rope.variants.{bf16,fp8}``, and ``Benchmark.base_variant`` selects
the row (same convention as ``fused_marlin_moe_w4a16``).

Shapes mirror ``benchmark/core_shapes.yaml`` for ``fused_norm_rope``:
``(num_tokens, q_lora)`` over q_lora 1536 (DeepSeek-V3.2) and 2048 (GLM-5.2).

``key_shape`` is the composite ``[tokens, q_lora, kv_lora, rope_dim,
index_head_dim]``: the benchmark's first tensor is ``positions`` whose
``[tokens]`` alone cannot identify the q_lora variant, so the benchmark matches
this record by the full tensor-shape signature built from ``config()``.
"""

import importlib

import torch

OP_NAME = "fused_norm_rope"
DTYPES = [torch.bfloat16]
# The kernel writes only its caches and the top-k buffer, and every write is a
# pure function of the (fixed) inputs, so repeating the call does not drift
# state. Keeping this False makes the collector time the call with
# ``triton.testing.do_bench(warmup=25ms, rep=100ms, median)`` -- the protocol
# ``benchmark/base.py`` uses for the FlagGems-vllm side.
IS_INPLACE = False

MAX_POS = 8192
KV_LORA = 512
ROPE_DIM = 64
INDEX_HEAD_DIM = 128
INDEX_ROW = INDEX_HEAD_DIM + INDEX_HEAD_DIM // 128 * 4  # 132
EPS = 1e-6
TOPK = 2048

# FlagGems-vllm benchmark shapes (benchmark/core_shapes.yaml).
_TOKENS = {
    1536: [1, 4, 17, 512, 4096],
    2048: [1, 4, 17, 512, 4096],
}
_MLA_DTYPES = ("bfloat16", "fp8")
_CANDIDATES = [
    ("vllm.models.deepseek_v32.common.kernels", "fused_norm_rope"),
]


def native():
    for module, symbol in _CANDIDATES:
        try:
            mod = importlib.import_module(module)
        except ImportError:
            continue
        op = getattr(mod, symbol, None)
        if callable(op):
            return op
    return None


def grid():
    return [
        {"tokens": tokens, "q_lora": q_lora, "mla_dtype": mla_dtype}
        for mla_dtype in _MLA_DTYPES
        for q_lora, token_counts in _TOKENS.items()
        for tokens in token_counts
    ]


def _make_cos_sin(max_pos, rot_dim, device):
    """``[max_pos, rot_dim]`` fp32, laid out as ``cos || sin``."""
    half = rot_dim // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, half, dtype=torch.float32, device=device) / half)
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


def build_inputs(binding, dtype, device):
    """Same tensors as the FlagGems-vllm benchmark's ``make_inputs``."""
    num_tokens = binding["tokens"]
    q_lora = binding["q_lora"]
    mla_dtype = binding["mla_dtype"]

    pos = torch.arange(num_tokens, device=device, dtype=torch.int64) % MAX_POS
    q_c = torch.randn(num_tokens, q_lora, device=device, dtype=dtype)
    kv_c = torch.randn(num_tokens, KV_LORA, device=device, dtype=dtype)
    k_pe = torch.randn(num_tokens, ROPE_DIM, device=device, dtype=dtype)
    qw = torch.randn(q_lora, device=device, dtype=dtype)
    kvw = torch.randn(KV_LORA, device=device, dtype=dtype)
    ik = torch.randn(num_tokens, INDEX_HEAD_DIM, device=device, dtype=dtype)
    ikw = torch.randn(INDEX_HEAD_DIM, device=device, dtype=torch.float32)
    ikb = torch.randn(INDEX_HEAD_DIM, device=device, dtype=torch.float32)
    cos_sin = _make_cos_sin(MAX_POS, ROPE_DIM, device)
    idx_cos_sin = _make_cos_sin(MAX_POS, ROPE_DIM, device)

    # Single block covering every token, like the vLLM kernel tests.
    block_size = MAX_POS
    mla_dim = KV_LORA + ROPE_DIM
    if mla_dtype == "fp8":
        mla_cache = torch.zeros(
            1, block_size, mla_dim, device=device, dtype=torch.uint8
        )
        mla_k_scale = torch.tensor([0.3], device=device, dtype=torch.float32)
    else:
        mla_cache = torch.zeros(1, block_size, mla_dim, device=device, dtype=dtype)
        mla_k_scale = None
    idx_cache = torch.zeros(1, block_size, INDEX_ROW, device=device, dtype=torch.uint8)
    slot = torch.arange(num_tokens, device=device, dtype=torch.int64)
    topk = torch.empty(num_tokens, TOPK, device=device, dtype=torch.int32)

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
    return args, kwargs


def key_shape(binding):
    shape = [
        binding["tokens"],
        binding["q_lora"],
        KV_LORA,
        ROPE_DIM,
        INDEX_HEAD_DIM,
    ]
    if binding["mla_dtype"] == "fp8":
        # Keep both variants in the collector's flat shapes dict; the benchmark
        # nests them under fused_norm_rope.variants.{bf16,fp8}.
        return "fp8_" + "x".join(str(v) for v in shape)
    return shape


def config(binding, dtype):
    num_tokens = binding["tokens"]
    q_lora = binding["q_lora"]
    mla_dtype = binding["mla_dtype"]
    dt = str(dtype)
    inputs = {
        "positions": {"shape": [num_tokens], "dtype": "torch.int64"},
        "q_c": {"shape": [num_tokens, q_lora], "dtype": dt, "fill": "randn"},
        "q_rms_norm_w": {"shape": [q_lora], "dtype": dt, "fill": "randn"},
        "kv_c": {"shape": [num_tokens, KV_LORA], "dtype": dt, "fill": "randn"},
        "kv_rms_norm_w": {"shape": [KV_LORA], "dtype": dt, "fill": "randn"},
        "k_pe": {"shape": [num_tokens, ROPE_DIM], "dtype": dt, "fill": "randn"},
        "k_rope_cos_sin_cache": {
            "shape": [MAX_POS, ROPE_DIM],
            "dtype": "torch.float32",
        },
        "index_k": {
            "shape": [num_tokens, INDEX_HEAD_DIM],
            "dtype": dt,
            "fill": "randn",
        },
        "index_k_layer_norm_w": {
            "shape": [INDEX_HEAD_DIM],
            "dtype": "torch.float32",
        },
        "index_k_layer_norm_bias": {
            "shape": [INDEX_HEAD_DIM],
            "dtype": "torch.float32",
        },
        "index_k_rope_cos_sin_cache": {
            "shape": [MAX_POS, ROPE_DIM],
            "dtype": "torch.float32",
        },
        "topk_indices_buffer": {
            "shape": [num_tokens, TOPK],
            "dtype": "torch.int32",
        },
        "slot_mapping": {"shape": [num_tokens], "dtype": "torch.int64"},
        "indexer_k_cache": {
            "shape": [1, MAX_POS, INDEX_ROW],
            "dtype": "torch.uint8",
        },
        "mla_kv_cache": {
            "shape": [1, MAX_POS, KV_LORA + ROPE_DIM],
            "dtype": "torch.uint8" if mla_dtype == "fp8" else dt,
        },
    }
    if mla_dtype == "fp8":
        inputs["mla_k_scale"] = {"shape": [1], "dtype": "torch.float32"}
    return {
        "inputs": inputs,
        "outputs": {
            "q_c_out": {"shape": [num_tokens, q_lora], "dtype": dt},
            "mla_kv_cache": {
                "shape": [1, MAX_POS, KV_LORA + ROPE_DIM],
                "dtype": "torch.uint8" if mla_dtype == "fp8" else dt,
                "note": "in-place concat(kv_c_normed, k_pe_roped) insert",
            },
            "indexer_k_cache": {
                "shape": [1, MAX_POS, INDEX_ROW],
                "dtype": "torch.uint8",
                "note": "in-place fp8 values + per-token fp32 scale",
            },
            "topk_indices_buffer": {
                "shape": [num_tokens, TOPK],
                "dtype": "torch.int32",
                "note": "cleared to -1",
            },
        },
        "dims": {
            "num_tokens": num_tokens,
            "q_lora": q_lora,
            "kv_lora": KV_LORA,
            "rope_dim": ROPE_DIM,
            "index_head_dim": INDEX_HEAD_DIM,
            "mla_kv_cache_dtype": mla_dtype,
        },
    }
