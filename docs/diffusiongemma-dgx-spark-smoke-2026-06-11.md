# DiffusionGemma DGX Spark Smoke (2026-06-11)

Goal: check whether DiffusionGemma is worth pursuing as a DGX Spark Tier B
candidate for `fb-reader`, especially as a possible way around autoregressive
decode bottlenecks.

## Candidate

Official model family:

- `google/diffusiongemma-26B-A4B-it`
- architecture tag: `diffusion_gemma`
- Gemma 4 26B A4B MoE derivative
- block / discrete diffusion style generation rather than normal token-by-token
  autoregressive decoding

Practical quantized candidates found on Hugging Face:

- `RedHatAI/diffusiongemma-26B-A4B-it-NVFP4`
- `RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic`
- `nvidia/diffusiongemma-26B-A4B-it-NVFP4`
- `unsloth/diffusiongemma-26B-A4B-it-GGUF`

This DGX smoke used:

```text
RedHatAI/diffusiongemma-26B-A4B-it-NVFP4
RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic
```

## Runtime Support

Native DGX vLLM does not support this architecture yet:

```text
native vllm: 0.20.2
native transformers: 5.8.0
vllm.model_executor.models.diffusion_gemma: absent
transformers.models.diffusion_gemma: absent
```

The useful Docker image is:

```text
vllm/vllm-openai:gemma-aarch64-cu130
```

It contains:

```text
vllm: 0.22.1rc1.dev357+g74b5964f0
transformers: 5.10.2
vllm.model_executor.models.diffusion_gemma: present
transformers.models.diffusion_gemma: absent
```

The earlier `vllm/vllm-openai:gemma4-unified-arm64-cu130` image does not include
the vLLM DiffusionGemma model executor.

## Runner

Reusable runner:

```bash
scripts/run_diffusiongemma_smoke.sh <google-bf16|nvidia-nvfp4|redhat-nvfp4|redhat-fp8> [smoke|bench]
```

The runner:

- stops production `vllm` and `vllm-pna-proxy`
- starts an isolated Docker container on `127.0.0.1:8001`
- runs `/v1/models`
- runs a short Traditional Chinese chat smoke
- optionally runs `vllm bench serve`
- restores production Qwen `vllm` and `vllm-pna-proxy` in `trap` cleanup

Remote artifacts:

```text
/home/devjoe/Projects/Ollama/benchmarks/diffusiongemma-20260611/
```

## Launch Notes

The first `redhat-nvfp4` launch failed during warmup:

```text
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
```

The stack was in DiffusionGemma / Gemma4 forward during KV cache update through
Triton attention, reached via `flashinfer_autotune`.

The fix was to disable FlashInfer autotune:

```bash
DIFFUSIONGEMMA_EXTRA_ARGS="--no-enable-flashinfer-autotune" \
  scripts/run_diffusiongemma_smoke.sh redhat-nvfp4 bench
```

With that flag, the model became ready and the Traditional Chinese smoke
completed.

## Bench Results

Current production reference:

| Backend | Shape | Output tok/s | Median TTFT |
| --- | --- | ---: | ---: |
| Qwen3.6 AutoRound + DFlash k=8 | 2K input / 256 output / c1 | 79.79 | 376 ms |

DiffusionGemma candidates:

| Backend | Shape | Output tok/s | Median TTFT | Notes |
| --- | --- | ---: | ---: | --- |
| DiffusionGemma RedHat NVFP4 | 2K input / 256 output / c1 | 46.70 | 5.49 s | TPOT effectively 0 because output is blocky/bursty. |
| DiffusionGemma RedHat NVFP4 | 2K input / 1024 output / c1 | 53.30 | 5.27 s | Longer output helps only slightly. |
| DiffusionGemma RedHat FP8 dynamic | 2K input / 256 output / c1 | 39.18 | 6.50 s | Slower than NVFP4 on the short-output shape. |
| DiffusionGemma RedHat FP8 dynamic | 2K input / 1024 output / c1 | 44.78 | 6.58 s | Also slower than NVFP4 on long output. |

Important interpretation: vLLM's ordinary serving benchmark is not a perfect fit
for block diffusion output. The `TPOT` and `ITL` metrics behave differently from
autoregressive models. Still, end-to-end output throughput and TTFT are directly
relevant for `fb-reader`.

## Disk And Restore

After pulling the new image plus RedHat NVFP4 and FP8 weights:

```text
/dev/nvme0n1p2  916G  654G  216G  76% /
```

Production restore was confirmed after each run:

```text
served id: qwen3.6-35b
root: Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound
```

## Decision

DiffusionGemma is worth tracking, but it is not a production replacement yet.

Reasons:

1. The tested RedHat NVFP4 path works only after disabling FlashInfer autotune.
2. Short and long output shapes remain below the current Qwen3.6 + DFlash
   throughput baseline.
3. FP8 dynamic does not fix the gap; it is slower than NVFP4 in both tested
   shapes.
4. Median TTFT around 5.3-6.6 seconds is high for `fb-reader` interactive use.
5. Pulling both RedHat quantized candidates raised root disk use to 76%, so more
   variants should be tested selectively.
6. We have not run stance / Taiwan / CIB / structured-output quality tests
   because the speed and latency gate did not pass.
7. The serving path is new and sensitive to vLLM image version and kernel flags.

The next useful experiment is not a production cutover. Only revisit
DiffusionGemma if vLLM's Gemma/DiffusionGemma backend changes, a DGX-specific
MoE FP8/NVFP4 config lands, or a smaller/faster DiffusionGemma checkpoint becomes
available.
