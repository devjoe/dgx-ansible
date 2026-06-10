# Qwen3.6 post-update benchmark and candidate survey

Date: 2026-06-11

This note records the post-DGX-update Qwen3.6 speed check and the follow-up
survey of realistic replacement or tuning candidates for the fb-reader Tier B
backend.

## Current Baseline

The Ansible-managed vLLM service is still running:

- Target model: `Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound`
- Served name: `qwen3.6-35b`
- Drafter: `z-lab/Qwen3.6-35B-A3B-DFlash`
- Speculative config in service: `method=dflash`, `num_speculative_tokens=8`
- Max model length: `262144`
- GPU memory utilization: `0.85`

Benchmark artifacts were saved on the DGX under:

```text
/home/devjoe/Projects/Ollama/benchmarks/qwen36-speed-20260611/
```

| Run | Context | Output | Concurrency | Output tok/s | Median TTFT | DFlash acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| greedy | 2K | 256 | 1 | 79.79 | 376 ms | 26.99% |
| greedy | 8K | 256 | 1 | 76.22 | 1.34 s | 44.60% |
| greedy | 2K | 256 | 4 | 187.35 aggregate | 500 ms | 41.52% |
| sampling default | 2K | 256 | 1 | 52.35 | 380 ms | 12.13% |

The greedy 2K result remains in the established Qwen DFlash range rather than
showing a large post-update jump. The lower sampling-default result is not a
like-for-like comparison with earlier greedy runs because speculative acceptance
dropped sharply.

## Candidate Survey Result

The practical candidate set is narrower than the public open-model landscape.
DGX Spark has a 128 GB unified-memory envelope, Blackwell-specific kernel paths,
and a production need for stable OpenAI-compatible serving.

| Priority | Candidate | Why it is worth testing | Main risk |
| ---: | --- | --- | --- |
| 1 | Qwen3.6 PrismaQuant / DFlash recipe revival | Same model family as production; possible route toward the old 95+ tok/s target. | Earlier attempt hit vLLM V1 and `compressed-tensors` incompatibility on Blackwell. |
| 2 | Qwen3.6 DFlash parameter sweep | Low-risk change to current serving path; DFlash docs use larger draft lengths than the current service. | Longer drafts can lower acceptance or worsen latency on fb-reader-style prompts. |
| 3 | Nemotron 3 Super 120B-A12B NVFP4 | Officially targets 1x DGX Spark and has 120B total / 12B active NVFP4 + MTP shape. | Known Spark decode numbers are closer to 23 tok/s, so this is a quality/long-context candidate, not a speed candidate. |
| 4 | DeepSeek V4 Flash local smoke | 284B total / 13B active FP4+FP8 MoE; vLLM lists DeepSeek V4 support. | Large download, uncertain Spark recipe, and prior cloud endpoint instability around sensitive prompts. |
| 5 | Gemma 4 latest vLLM/MTP path | Gemma remains strategically useful for settled-control and Taiwan/CIB calibration. | Prior Gemma runs did not beat Qwen as the practical default unless backend support materially changed. |

MiniMax M2.7, GLM-5.1, and Kimi K2.6 remain useful as cloud or API comparison
models, but they are not first-line DGX Spark local candidates without a proven
small-active-parameter quantized serving recipe. Their public model sizes and
active-parameter profiles are too large for a quick local replacement attempt.

## Next Experiment

Revive the Qwen3.6 PrismaQuant / DFlash branch first:

1. Confirm the local PrismaQuant and DFlash assets still exist on the DGX.
2. Re-check whether the current post-update vLLM can load the PrismaQuant model,
   especially with a V0-compatible path for `compressed-tensors`.
3. If it loads, run a short DFlash sweep:
   - `num_speculative_tokens=6`
   - `num_speculative_tokens=8`
   - `num_speculative_tokens=12`
   - optionally `num_speculative_tokens=15` if acceptance remains healthy
4. Gate any candidate with the same small speed check used for the current
   Qwen service plus a minimal fb-reader text/image smoke.

Promotion should require a clear win over the current greedy baseline around
80 tok/s single-request decode without schema, stance, or image-regression
breakage.

## PrismaQuant Revival Result

The revival smoke and short speed checks completed after the DGX software
update. The reusable runner is:

```bash
scripts/run_prismaquant_smoke.sh <base|dflash> <spec_tokens> [smoke|bench]
```

It stops the production `vllm` and `vllm-pna-proxy` services, launches an
isolated PrismaQuant server on `127.0.0.1:8011`, runs a `/v1/models` and
completion smoke, optionally runs `vllm bench serve`, and restores production
services in `trap` cleanup.

Remote artifacts:

```text
/home/devjoe/Projects/Ollama/benchmarks/prismaquant-revival-20260611/
```

Load findings:

- `vllm 0.20.2` can load
  `/home/devjoe/Projects/Ollama/models/qwen3.6-35b-prismaquant`.
- The model config is now recognized as `quantization=compressed-tensors`.
- vLLM uses NVFP4 / MXFP8 kernels instead of failing at startup.
- `VLLM_USE_V1=0` is no longer a valid override in this vLLM build; the log says
  `Unknown vLLM environment variable detected: VLLM_USE_V1`, and the engine
  initializes as V1.
- Initial profiling is expensive: base load had about 250 s warmup/profiling on
  the first successful run. Subsequent runs still spent roughly 97-118 s in
  engine init for these short 4K profiles.

Speed result, same 2K input / 256 output / greedy / concurrency 1 shape as the
post-update Qwen baseline:

| Candidate | Output tok/s | Median TTFT | Acceptance | Notes |
| --- | ---: | ---: | ---: | --- |
| Current AutoRound + DFlash k=8 | 79.79 | 376 ms | 26.99% | Production baseline |
| PrismaQuant base | 44.92 | 362 ms | n/a | Loads, but too slow without DFlash |
| PrismaQuant + DFlash k=6 | 59.69 | 370 ms | 26.19% | DFlash helps, but still below production |
| PrismaQuant + DFlash k=8 | 74.05 | 368 ms | 30.61% | Best measured PrismaQuant profile, still below production |

Decision: PrismaQuant / DFlash is revived as a runnable experiment path, but it
is not a promotion candidate yet. It closes most of the gap at k=8, but still
does not beat the current AutoRound + DFlash production baseline and carries a
large initialization/profiling cost. Do not spend another long sweep on k=12
until there is a new serving-side reason to expect better acceptance or faster
compressed-tensors kernels.
