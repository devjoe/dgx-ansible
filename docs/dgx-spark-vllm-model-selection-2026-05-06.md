# DGX Spark vLLM Model Selection (fb-reader + OpenCode)

Last updated: 2026-05-06

Goal: pick **one** vLLM-served model on DGX Spark (GB10, 128 GB UMA) that:

1. Is strong enough for **OpenCode** (code + repo reasoning, long context).
2. Is fast/stable enough for **fb-reader Tier B** (zh-TW JSON, occasional vision).
3. Has a practical, repeatable deployment recipe (no one-off hand edits).

This doc is **local-evidence driven**: all key decisions below are based on
captured Tier B traffic + replayed benchmarks, not synthetic tok/s only.

## Benchmark Method (Realistic Tier B Replay)

1. Capture 50 real FB posts via `fb-reader` remote debug:
   - Output: `~/Projects/fb-reader/tmp/tier-b-corpus-2026-05-06T07-53-23-804Z/tier-b-cases.json`
   - 40/50 cases contain images (data-URI JPEGs).
2. Replay the corpus against DGX vLLM:
   - `make replay-tier-b-corpus` (curl transport)
   - Output: `~/Projects/fb-reader/tmp/tier-b-replay/*.json`

Metrics used:
- JSON parse + schema checks (must be stable)
- latency p50/p90 (all / image-only / text-only)

## Results (2026-05-06 Corpus)

All runs: 50/50 HTTP success, no timeouts.

### A0. Intel/Qwen3.6 AutoRound INT4 + DFlash (k=8), max_model_len=131072

Result file:
- `fb-reader/tmp/tier-b-replay/expA0-intel-dflash-1778056034.json`

Latency:
- all: p50 4.71s, p90 9.78s
- image: p50 4.94s, p90 10.54s
- text: p50 3.09s, p90 3.51s

Stability:
- parse_ok 49/50, schema_ok 49/50
- 1 case hit `max_tokens=600` and got truncated.

### A1. Same model, Speculative OFF (no DFlash), max_model_len=131072

Result file:
- `fb-reader/tmp/tier-b-replay/expA1-intel-no-spec-1778057621.json`

Latency:
- all: p50 7.96s, p90 14.50s
- image: p50 8.06s, p90 15.49s
- text: p50 5.96s, p90 6.99s

Takeaway:
- DFlash improves p50 by ~1.7x on this workload (including image cases).

### B0. cklaus/gemma-4-26B-A4B-it-NVFP4 (modelopt), attention_backend=auto

Operational note:
- Gemma4 NVFP4 needs `ninja` for torch.compile on this stack.
  We installed `ninja-build` in the vLLM role.

Result file:
- `fb-reader/tmp/tier-b-replay/expB0-gemma4-nvfp4-1778062122.json`

Latency:
- all: p50 12.49s, p90 17.27s
- image: p50 12.92s, p90 17.62s
- text: p50 9.52s, p90 10.19s

Takeaway:
- Much slower than Qwen+DFlash for Tier B classification.

### A2. Intel/Qwen3.6 AutoRound INT4 + DFlash, max_model_len=262144 (OpenCode-capable)

Result file:
- `fb-reader/tmp/tier-b-replay/expA2-intel-dflash-262k-1778063245.json`

Latency:
- all: p50 4.97s, p90 11.61s
- image: p50 5.22s, p90 12.02s
- text: p50 3.10s, p90 3.98s

Takeaway:
- 262K ceiling works and does not break Tier B, but p90 moved a bit (restart
  warmup / cache effects likely; rerun for a tighter confidence interval).

## Recommendation (Single Shared vLLM Default)

Use **Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound + DFlash (k=8)** with:

- `max_model_len: 262144` (OpenCode long-context headroom)
- `max_num_batched_tokens: 32768` (chunked prefill)
- `attention_backend: flash_attn` (FlashInfer breaks on non-causal/multimodal paths for this stack)
- `speculative_config: dflash k=8` (big latency win on this workload)

Gemma4 NVFP4 is not a good Tier B default today on this workload (latency).

## How to Apply (in This Repo)

`group_vars/dgx.yml` is the single source of truth:

```bash
make deploy
make status-vllm
```

For Tier B replay:

```bash
cd ~/Projects/fb-reader
CORPUS=tmp/tier-b-corpus-2026-05-06T07-53-23-804Z/tier-b-cases.json make replay-tier-b-corpus
```
