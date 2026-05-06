# DGX Spark vLLM Model Selection (fb-reader + OpenCode)

Last updated: 2026-05-07

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

### B1. RedHatAI Gemma4 26B-A4B-it FP8 + Google MTP, max_model_len=262144

Result file:
- `fb-reader/tmp/tier-b-replay/expB1-redhat-gemma4-fp8-it-mtp-262k-1778091460.json`

Runtime:
- Docker image: `vllm/vllm-openai:gemma4-0505-cu130`
- Patched MTP implementation bind-mounted from vLLM PR #41745 head SHA
  `d8b3826648da6b407f8c55457a2103be9aeb5d83`.
- target: `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`
- assistant: `google/gemma-4-26B-A4B-it-assistant`
- `speculative_config`: `{"method":"mtp","model":"google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4}`
- `kv_cache_dtype=fp8`, `gpu_memory_utilization=0.55`,
  `max_num_batched_tokens=16384`, CUDA graphs enabled.

Latency:
- all: p50 7.00s, p90 9.97s
- image: p50 7.54s, p90 10.73s
- text: p50 5.25s, p90 7.19s

Stability:
- parse_ok 50/50, schema_ok 50/50
- 50/50 HTTP success, no timeouts.

Operational observations:
- 262K + CUDA graphs reached `/v1/models` successfully.
- Checkpoint size: 26.67 GiB.
- Model loading memory: 26.58 GiB.
- Available KV cache: 33.22 GiB.
- GPU KV cache size: 2,092,288 tokens.
- vLLM-estimated max concurrency for 262,144-token requests: 7.98x.
- Cold start was expensive:
  - target download: 890s on first run
  - target weight load: 234s
  - engine profile / compile / warmup: 141s
  - multimodal warmup: 38s
- vLLM warned: draft model does not support multimodal inputs, so it falls
  back to text-only mode. Practically, the Gemma target handles image prefill;
  MTP helps mostly during text decoding.
- vLLM also warned that the GB10 FP8 MoE tuning config was missing:
  `E=128,N=704,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json`. Performance may
  improve when upstream ships or we generate that config.
- During replay, MTP acceptance was healthy but variable:
  - mean acceptance length roughly 3.1-4.4 tokens
  - average draft acceptance rate roughly 53-86%, commonly around 60-74%

Takeaway:
- This is a large improvement over Gemma4 NVFP4 no-MTP:
  - p50 12.37s -> 7.00s
  - p90 17.24s -> 9.97s
  - image p90 17.60s -> 10.73s
- It is still slower than Qwen DFlash on median latency:
  - Qwen all p50 4.96s vs Gemma FP8-it MTP 7.00s
  - Qwen image p50 5.22s vs Gemma FP8-it MTP 7.54s
- It is competitive on tail latency in this one run:
  - Qwen all p90 11.57s vs Gemma FP8-it MTP 9.97s
  - Qwen image p90 11.99s vs Gemma FP8-it MTP 10.73s
- Because fb-reader requests are short-output and often image-heavy, the
  public pure-text MTP tok/s headline does not translate directly into a Tier B
  win. Gemma FP8-it MTP is now a serious candidate, but not yet the default.

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
Gemma4 FP8-it + MTP is much better and should stay on the candidate list, but
does not yet replace Qwen as the single shared default because median latency
is still worse, the serving path requires a preview Docker image plus a patched
MTP file, and the assistant is text-only for multimodal requests.

## Gemma4 MTP Follow-up (2026-05-07)

New public information changed the Gemma4 picture: Google released Gemma4
Multi-Token Prediction (MTP) assistant checkpoints, and DGX Spark users have
reported successful GB10 runs with vLLM.

Important references:
- Google assistant checkpoint:
  `google/gemma-4-26B-A4B-it-assistant`
  - https://huggingface.co/google/gemma-4-26B-A4B-it-assistant
- vLLM speculative decoding docs:
  - https://docs.vllm.ai/en/latest/features/speculative_decoding/
- DGX Spark / GB10 working recipe:
  - https://forums.developer.nvidia.com/t/gemma-4-mtp/369123

Key facts from the 2026-05-07 check:
- Current native DGX venv has `vllm 0.20.1` and `transformers 5.8.0`, but does
  **not** contain `vllm/model_executor/models/gemma4_mtp.py`.
- Docker is available on DGX, and `vllm/vllm-openai:gemma4-0505-cu130` has an
  `linux/arm64` manifest, so the Docker path is viable on GB10.
- The community recipe uses:
  - target: `nvidia/Gemma-4-26B-A4B-NVFP4`
  - assistant: `google/gemma-4-26B-A4B-it-assistant`
  - `--speculative-config '{"method":"mtp","model":"google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4}'`
  - `--gpu-memory-utilization 0.55`
  - `--kv-cache-dtype fp8`
  - `--max-model-len 262144`
  - `--max-num-batched-tokens 16384`
  - `--enforce-eager`
  - `--no-enable-flashinfer-autotune`
- A later community report showed a faster instruction-tuned FP8 path:
  - target: `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`
  - assistant: `google/gemma-4-26B-A4B-it-assistant`
  - patched `gemma4_mtp.py` from PR #41745 is required for quantized targets
    because the preview image was built before the `intermediate_size` and
    draft `quant_config` fixes.
  - reference: https://ai-muninn.com/en/blog/dgx-spark-gemma4-mtp-108-toks

Experiment protocol:
1. Stop the systemd `vllm` service temporarily.
2. Launch the Gemma4 MTP Docker server on `gx10.local:8000`.
3. Verify `/v1/models`.
4. Replay the same Tier B corpus and write a new result file.
5. Stop the Docker container and restore the Ansible-managed Qwen DFlash
   systemd service.

Acceptance bar:
- Gemma4 MTP must beat the previous Gemma4 NVFP4 p50/p90 by a large margin.
- It must also come close enough to Qwen DFlash Tier B latency to justify its
  likely quality / vision benefits.
- `schema_ok` must remain 50/50 or the output/parser path needs adjustment.

### Local Attempt: 2026-05-07

Docker image:
- `vllm/vllm-openai:gemma4-0505-cu130`
- image vLLM: `0.20.2rc1.dev49+g9b4e83934`
- image Transformers: `5.8.0`
- contains `vllm/model_executor/models/gemma4_mtp.py`

Attempt 1: community target + assistant

```bash
docker run --rm --gpus all --ipc=host --network host \
  -v /home/devjoe/.cache/huggingface:/root/.cache/huggingface \
  -v /home/devjoe/.cache/vllm:/root/.cache/vllm \
  vllm/vllm-openai:gemma4-0505-cu130 \
  nvidia/Gemma-4-26B-A4B-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3.6-35b \
  --gpu-memory-utilization 0.55 \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 16384 \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --trust-remote-code \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice \
  --speculative-config '{"method":"mtp","model":"google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4}'
```

Observed:
- The image correctly resolves target `Gemma4ForConditionalGeneration`.
- It also resolves assistant `Gemma4MTPModel`, so the MTP code path is present.
- Download of `nvidia/Gemma-4-26B-A4B-NVFP4` stalled before the server became
  ready:
  - first run reached about `11G` in HF cache, then stopped growing
  - retry reset/settled around `5.0G`
  - `/v1/models` stayed `000`
- `google/gemma-4-26B-A4B-it-assistant` downloaded to about `801M`.

Attempt 2: cached `cklaus/gemma-4-26B-A4B-it-NVFP4` target + Google assistant

Observed:
- Target started loading from existing cache.
- Assistant load failed with a weight shape assertion inside
  `vllm/model_executor/models/gemma4_mtp.py`.
- Root cause: the assistant must match the exact target family/layout. The
  Google `-it-assistant` checkpoint should be paired with the expected Google /
  NVIDIA target, not the independently packaged cklaus target.

Outcome:
- No Tier B replay result was produced for Gemma4 MTP yet because no MTP server
  reached `/v1/models`.
- Stable service was restored to
  `Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound + DFlash`, `max_model_len=262144`.

Attempt 3: `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic` target + Google assistant

Observed:
- Patched `gemma4_mtp.py` from PR head SHA
  `d8b3826648da6b407f8c55457a2103be9aeb5d83` fixed the quantized-target
  shape mismatch seen in the earlier `cklaus` attempt.
- Server successfully reached `/v1/models` with 262K context and CUDA graphs.
- Stable Qwen service was restored after the replay.

Result:
- See B1 above. Gemma FP8-it MTP is a real candidate now, especially for
  image-heavy tail latency, but Qwen DFlash remains the stable default.

Next viable MTP steps:
- Rerun B1 once warm, now that the model and torch compile cache are present,
  to remove cold JIT artifacts from the first replay.
- Test an OpenCode-style corpus separately; the current evidence is Tier B
  fb-reader replay, not repo-editing quality/latency.
- Investigate GB10 FP8 MoE tuning config generation for Gemma4:
  `E=128,N=704,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json`.

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
