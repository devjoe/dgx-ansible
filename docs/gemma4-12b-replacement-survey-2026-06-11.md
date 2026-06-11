# Gemma 4 12B Replacement Survey (2026-06-11)

Goal: evaluate whether the new Gemma 4 12B family can replace the current
DGX Spark Tier B backend:

- current production: `Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound`
  + `z-lab/Qwen3.6-35B-A3B-DFlash`
- served name: `qwen3.6-35b`
- current baseline: 2K input / 256 output / greedy / concurrency 1 at
  `79.79 tok/s`; 8K input / 256 output / concurrency 1 at `76.22 tok/s`

## Upstream Findings

Official Gemma 4 12B exists as `google/gemma-4-12B-it`.

Relevant properties from the model card:

- architecture tag: `gemma4_unified`
- total parameters: `11.95B`
- dense model, not MoE
- context length: `256K`
- supported modalities: text, image, and audio
- native system prompt support
- model card reports Gemma 4 12B at MMLU Pro `77.2`, AIME 2026 `77.5`,
  LiveCodeBench v6 `72.0`, and MMMLU `83.4`

Candidate search through the Hugging Face API found these practical variants:

| Candidate | Reason to test | Result |
| --- | --- | --- |
| `coolthor/gemma-4-12B-it-NVFP4A16` | Has a DGX Spark / GB10 vLLM benchmark claim and keeps multimodal output intact. | Runs, but slower than Qwen production. |
| `google/gemma-4-12B-it-qat-w4a16-ct` | Official QAT compressed-tensors format advertised for native vLLM inference. | Does not currently load in the tested vLLM image. |
| `google/gemma-4-12B-it` | Official BF16 baseline. | Not tested after NVFP4A16 was already far below Qwen. |
| GGUF variants | Useful for Ollama / llama.cpp, not a drop-in vLLM Tier B replacement. | Deferred. |

Runtime image selected:

```text
vllm/vllm-openai:gemma4-unified-arm64-cu130
```

The DGX Spark is `linux/arm64`, so the ARM64 image is required. This image
contains:

```text
vLLM 0.1.dev17235+gf52870f26.d20260603
Transformers 5.10.1
vllm.model_executor.models.gemma4_unified: present
```

The existing native DGX venv is not enough for this test:

```text
vLLM 0.20.2
Transformers 5.8.0
vllm.model_executor.models.gemma4_unified: absent
```

## Runner

Reusable runner:

```bash
scripts/run_gemma4_12b_smoke.sh <coolthor-nvfp4a16|google-qat-w4a16|google-bf16> [smoke|bench]
```

The runner:

- stops `vllm` and `vllm-pna-proxy`
- starts the candidate in an isolated Docker container on `127.0.0.1:8001`
- runs `/v1/models`
- runs a short Traditional Chinese chat smoke
- optionally runs `vllm bench serve`
- restores production `vllm` and `vllm-pna-proxy` in `trap` cleanup

Remote artifacts:

```text
/home/devjoe/Projects/Ollama/benchmarks/gemma4-12b-replace-20260611/
```

## Results

### `coolthor/gemma-4-12B-it-NVFP4A16`

Smoke:

- `/v1/models` returned `gemma4-12b`
- root: `coolthor/gemma-4-12B-it-NVFP4A16`
- Traditional Chinese chat smoke succeeded

Short speed bench:

```text
shape: 2K input / 256 output / greedy / concurrency 1
successful requests: 4
failed requests: 0
output throughput: 22.11 tok/s
peak output throughput: 24.00 tok/s
median TTFT: 1005.52 ms
median TPOT: 42.34 ms
```

This is close to the external model-card claim of about `24.9 tok/s`, so the
measurement looks plausible. It is still much slower than current production
Qwen3.6 + DFlash at `79.79 tok/s` for the same 2K/256/c1 shape.

The model card also reports a Traditional Chinese quality cost for this
NVFP4A16 build: TMMLU+ falls from `47.21%` BF16 to `41.24%`, while FP8 dynamic
is nearly lossless at `46.97%`. That matters for `fb-reader`, whose initial
users are Taiwan readers.

### `google/gemma-4-12B-it-qat-w4a16-ct`

Direct multimodal launch failed before readiness:

```text
AttributeError: 'Gemma4UnifiedVisionConfig' object has no attribute 'num_soft_tokens'
```

Text-only fallback with `--language-model-only` also failed during weight load:

```text
ValueError: There is no module or parameter named
'vision_embedder.patch_dense.weight' in Gemma4UnifiedForConditionalGeneration.
The available parameters belonging to vision_embedder.patch_dense are:
{'vision_embedder.patch_dense.weight_scale',
 'vision_embedder.patch_dense.weight_shape',
 'vision_embedder.patch_dense.weight_packed',
 'vision_embedder.patch_dense.bias'}
```

vLLM did recognize this model as:

```text
quantization=compressed-tensors
CompressedTensorsWNA16 -> MarlinLinearKernel
AttentionBackendEnum.TRITON_ATTN
```

but the current image cannot load the official compressed-tensors weights.

## Disk And Production Restore

After image and model downloads:

```text
/dev/nvme0n1p2  916G  582G  288G  67% /
```

Production restore was confirmed after every candidate run:

```text
served id: qwen3.6-35b
root: Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound
```

## Decision

Do not replace Qwen3.6 with Gemma 4 12B yet.

Reasons:

1. The fastest tested Gemma 4 12B variant is about `22.11 tok/s`, far below the
   current Qwen3.6 + DFlash `79.79 tok/s` baseline.
2. The fastest variant has documented Traditional Chinese degradation, which is
   product-relevant for `fb-reader`.
3. The official QAT compressed-tensors variant currently fails to load in the
   available Gemma4 unified vLLM image.
4. This test only covered a short speed smoke. It did not yet run the full
   stance / Taiwan / CIB risk suite.

The reasonable next test, if Gemma 4 12B remains strategically interesting, is
not a production cutover. It is a small quality/stance A/B using
`coolthor/gemma-4-12B-it-NVFP4A16` or a TC-safer FP8 build, with the same 21-item
stance set and DS4 contested/settled slices. For speed, the current Qwen3.6
production backend remains clearly better.

