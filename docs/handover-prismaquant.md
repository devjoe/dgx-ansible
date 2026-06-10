# Handover: PrismaQuant & DFlash Investigation on DGX Spark

This document captures the status of the investigation into running PrismaQuant models on the NVIDIA DGX Spark (GB10/Blackwell).

## Current Status
- **Investigation revived and rechecked**: 2026-06-11.
- **Stable State**: System is running `Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound` (+ DFlash) via vLLM (systemd mode).
- **Network**: Wi-Fi `10Design2` was enabled on DGX to restore internet access. Keep IPv6 disabled on this outward-facing Wi-Fi profile; the upstream router has been unstable on IPv6 and previously caused Docker Hub / Hugging Face download stalls. The validated profile state is IPv4 `192.168.1.102/24`, no global IPv6 address on `wlP9s9`, and DNS `8.8.8.8` / `8.8.4.4`.

## PrismaQuant Findings
- **Post-update result**: vLLM `0.20.2` can now load the local PrismaQuant model
  through the V1 engine. The old immediate `compressed-tensors` startup failure
  is no longer the active blocker.
- **V1 Engine Behavior**: `VLLM_USE_V1=0` is no longer accepted in this vLLM
  build, so the current path is V1. Logs show `quantization=compressed-tensors`
  and NVFP4 / MXFP8 kernels.
- **Local Assets**: PrismaQuant model and DFlash speculator files were manually pushed to DGX Spark at `~/Projects/Ollama/models/`.

## Historical Target Config (95+ tok/s)
The community configuration that originally motivated this branch used:
- **Model**: `rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm`
- **Speculator**: `z-lab/Qwen3.6-35B-A3B-DFlash` (`num_speculative_tokens: 6`)
- **Old hurdle**: Finding a clean engine path for `compressed-tensors` on
  Blackwell.
- **Current hurdle**: The model now loads, but the measured throughput still
  trails the current AutoRound + DFlash production baseline.

## 2026-06-11 Smoke And Speed Result

Reusable runner:

```bash
scripts/run_prismaquant_smoke.sh <base|dflash> <spec_tokens> [smoke|bench]
```

Remote artifacts:

```text
/home/devjoe/Projects/Ollama/benchmarks/prismaquant-revival-20260611/
```

| Candidate | 2K/256 greedy output tok/s | Median TTFT | Acceptance |
| --- | ---: | ---: | ---: |
| Current AutoRound + DFlash k=8 | 79.79 | 376 ms | 26.99% |
| PrismaQuant base | 44.92 | 362 ms | n/a |
| PrismaQuant + DFlash k=6 | 59.69 | 370 ms | 26.19% |
| PrismaQuant + DFlash k=8 | 74.05 | 368 ms | 30.61% |

Decision: the PrismaQuant path is runnable again, including local DFlash, but it
does not beat production AutoRound + DFlash. Keep the production backend
unchanged. Revisit only if vLLM improves compressed-tensors / NVFP4 kernels or
if a new recipe reports a clear win on this exact DGX Spark generation path.
