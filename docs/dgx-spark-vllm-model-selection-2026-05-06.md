# DGX Spark vLLM Model Selection (fb-reader + OpenCode)

Last updated: 2026-05-06

Goal: pick **one** vLLM-served model on DGX Spark (GB10, 128 GB UMA) that:

1. Is strong enough for **OpenCode** (code + repo reasoning, long context).
2. Is fast/stable enough for **fb-reader Tier B** (zh-TW JSON, occasional vision).
3. Has a practical, repeatable deployment recipe (no one-off hand edits).

This doc is evidence-driven. It points to public GB10 measurements and captures
the exact flags needed to reproduce them in our `dgx-ansible` systemd service.

## Candidates (2026-05)

### A. Qwen3.6 35B-A3B FP8 (recommended default)

Why it fits the combined goal:

- Strong “agentic coding” profile (OpenCode target).
- Native 262K context target.
- Proven working tool-call / reasoning parsers in vLLM.
- Community GB10 numbers show **~70–78 tok/s** decode at large contexts.

Evidence (GB10):
- NVIDIA forum thread with `llama-benchy` numbers and a working `vllm serve`
  command for `Qwen/Qwen3.6-35B-A3B-FP8` (includes context up to 131072 with
  tg128 ~64 tok/s). citeturn3view1

Suggested vLLM serve shape (single Spark):

- `vllm_model: Qwen/Qwen3.6-35B-A3B-FP8`
- `--kv-cache-dtype fp8`
- `--attention-backend flashinfer` (Blackwell prefers FlashInfer)
- `--max-model-len 262144`
- `--enable-prefix-caching --enable-chunked-prefill`

Notes:
- You can add speculative decoding (MTP or DFlash) later, but FP8 baseline is
  already fast and operationally simpler than pre-merge PR stacks.

### B. Gemma 4 26B-A4B NVFP4 + assistant MTP (high-upside, higher moving parts)

Why it’s interesting:
- Same 256K context class.
- Very strong long-context + multimodal (helpful for fb-reader’s image-heavy posts).
- New assistant (“-assistant”) models can provide large decoding speedups.

Evidence (GB10):
- NVIDIA forum post reporting **mean acceptance length 3.68/4**, **67–69%**
  acceptance, and speedups like **2.34x** wall (sequential ×3) and **~175 tok/s
  peak aggregate throughput** for NVFP4 target + BF16 assistant. citeturn4view3

Operational caveat:
- As of 2026-05-06, this path may require vLLM PR head + Transformers main to
  recognize `gemma4_assistant` (until upstream releases land). The forum thread
  includes the “works today” recipe. citeturn0search1turn1search3

### C. Intel AutoRound INT4 Qwen3.6 + DFlash (current deployed)

Why it exists:
- Easy to fit + fast enough for Tier B.
- DFlash can yield substantial latency improvements, but the win is workload-
  dependent and may require very specific vLLM builds in some periods.

Evidence / reference:
- DFlash model card includes a vLLM launch example and reports up to ~2.9x
  speedup on some tasks (different hardware; still a useful reference). citeturn11view0

## Recommendation (single-model default)

Start with **Qwen/Qwen3.6-35B-A3B-FP8** as the shared DGX model for both
fb-reader Tier B and OpenCode.

Rationale:
- It directly optimizes for OpenCode’s “coding first” workload.
- It retains long-context headroom (262K).
- The GB10 decode numbers are already strong without needing a pre-merge PR
  stack. citeturn3view1

Keep Gemma4+assistant as an experimental branch: it may become the best
“fast multimodal” choice once vLLM + Transformers land clean releases.

## How to apply (in this repo)

`group_vars/dgx.yml` is the single source of truth.

For Qwen3.6 FP8, set:

- `vllm_model: Qwen/Qwen3.6-35B-A3B-FP8`
- `vllm_quantization: ""` (or `fp8` if your vLLM build expects it)
- `vllm_attention_backend: "flashinfer"`
- `vllm_kv_cache_dtype: "fp8"`
- `vllm_max_model_len: 262144`
- `vllm_max_num_batched_tokens: 16384` (then tune upward if safe)
- `vllm_reasoning_parser: "qwen3"`
- `vllm_tool_call_parser: "qwen3_coder"`
- `vllm_speculative_config: ""` (baseline first; add later if desired)

Then:

```bash
make deploy
make status-vllm
python3 scripts/run_vllm_classification.py --connect-mode auto
```

## What we still need to measure locally

Public GB10 numbers are a good filter, but we still need “our workload” A/Bs:

1. OpenCode: long prompt + tool-use-ish code edits (TTFT + tok/s).
2. fb-reader Tier B: 10–20 real fixtures (zh-TW + images) for correctness +
   latency under realistic `max_tokens=600`.
3. Concurrency: ensure we don’t regress interactive p50 while improving batch.

