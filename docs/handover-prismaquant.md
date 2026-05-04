# Handover: PrismaQuant & DFlash Investigation on DGX Spark

This document captures the status of the investigation into running PrismaQuant models on the NVIDIA DGX Spark (GB10/Blackwell).

## Current Status
- **Investigation Paused**: 2026-05-04.
- **Stable State**: System is running `Qwen/Qwen3.6-35B-A3B-AWQ` via vLLM (systemd mode).
- **Network**: Wi-Fi `10Design2` was enabled on DGX to restore internet access. IPv6 was temporarily disabled to fix download stalls.

## PrismaQuant Findings
- **V1 Engine Conflict**: vLLM nightly/latest attempts to use the V1 engine on Blackwell by default. This engine is currently incompatible with `compressed-tensors` (PrismaQuant), causing `NotImplementedError`.
- **Local Assets**: PrismaQuant model and DFlash speculator files were manually pushed to DGX Spark at `~/Projects/Ollama/models/`.

## Target Config (95+ tok/s)
The community configuration to reach 95 tok/s uses:
- **Model**: `rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm`
- **Speculator**: `z-lab/Qwen3.6-35B-A3B-DFlash` (`num_speculative_tokens: 6`)
- **Key hurdle**: Identifying the exact flags/env vars to force a clean **V0 engine** load for `compressed-tensors` on Blackwell in newer vLLM versions.
