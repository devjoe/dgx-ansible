# DGX Spark Gemma4 vs Qwen stance A/B plan (2026-05-19)

This note defines the pre-implementation plan for comparing the current Qwen
DFlash backend against the Gemma4 FP8-it MTP candidate on DGX Spark.

## Runtime plan

DGX Spark cannot keep both large vLLM profiles hot with the production memory
settings. The A/B therefore runs sequentially under Ansible:

1. Keep the Ansible-managed Qwen DFlash service on port 8000 and run the stance
   corpus baseline.
2. Stop `vllm.service`.
3. Start a detached Gemma4 FP8-it MTP Docker container on the same port.
4. Run the same stance corpus against Gemma4.
5. Always stop the Gemma container and restore `vllm.service`.

The entrypoint is:

```bash
make stance-ab
```

The playbook writes results under:

```text
/home/devjoe/Projects/Ollama/benchmarks/stance-ab-<UTC timestamp>/
```

## Gemma4 MTP path

The default candidate is:

- target: `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`
- assistant: `google/gemma-4-26B-A4B-it-assistant`
- speculative tokens: `4`
- max context: `262144`
- `kv_cache_dtype=fp8`
- `gpu_memory_utilization=0.55`

This is the same model pair that previously completed the real `fb-reader`
Tier B replay. The playbook defaults to the locally cached
`vllm/vllm-openai:gemma4-0505-cu130` image and runs with
`HF_HUB_OFFLINE=1`, because an initial attempt with `vllm/vllm-openai:latest`
resolved to a newer generic image and failed while trying to refresh
Hugging Face metadata from inside the container.

If the local image still regresses on GB10, override the image and optional
patch path through Ansible extra vars.

## Stance corpus

The corpus lives at:

```text
prompts/stance_bias_corpus.json
```

It is inspired by the `audreyt/ds4` directional-steering uncertainty example:
the important axis is `contested` vs `settled`, not DS4-specific activation
editing. The experiment checks whether each model naturally uses an appropriate
register:

- settled questions should be direct and confident;
- contested questions should present material positions;
- forced-framing questions should not blindly accept one-sided framing;
- Taiwan-sensitive questions should separate facts, values, and political
  framing in Traditional Chinese.

The runner asks each model to return self-audited JSON with:

- `answer_type`
- `confidence`
- `assertiveness`
- `balance`
- `stance_label`
- `unsupported_certainty`

The summary compares parse/schema stability, latency p50/p90, answer-type
calibration, and unsupported-certainty counts by category.

## Decision use

This is not a replacement for the real `fb-reader` Tier B replay. It is a
preflight risk check before spending a longer window on Gemma4 MTP:

- If Gemma4 fails schema stability or shows more one-sided contested answers,
  keep Qwen as the only serious Tier B default.
- If Gemma4 is stable and better calibrated, rerun warm Gemma4 MTP on the full
  `fb-reader` replay corpus next.

## 2026-05-19 implementation notes

The first live attempt produced useful harness findings but should not be used
as a stance result:

- Qwen served 21/21 requests, but most responses included visible thinking text
  before the requested JSON and were truncated by the original token cap. The
  runner now sends `/no_think` in both system and user messages, requests
  `response_format=json_object`, disables thinking/preserved thinking through
  `chat_template_kwargs`, extracts the last complete JSON object when needed,
  and uses a higher token cap.
- Gemma4 was first launched with `vllm/vllm-openai:latest`; on the DGX this was
  vLLM `0.19.1`, and the container failed on Hugging Face metadata lookup. The
  playbook now uses the local Gemma4 image, offline cache mode, a shorter
  readiness window, and logs container output when readiness fails.
- The playbook restores `vllm.service` in `always`, so a failed Gemma candidate
  should leave the production Qwen path running again.

## 2026-05-19 live A/B result

The first valid full run completed under:

```text
/home/devjoe/Projects/Ollama/benchmarks/stance-ab-20260519T114146Z/
```

Both models completed the 21-item corpus with 21/21 HTTP success, 21/21 parse
success, and 21/21 schema success. `make status-vllm` afterwards confirmed the
production Qwen service was active again on `/v1/models`.

| Model | Expected answer type | Unsupported certainty | Latency p50 | Latency p90 |
| --- | ---: | ---: | ---: | ---: |
| Qwen DFlash 262K | 17/21 | 0 | 2.8093s | 5.3892s |
| Gemma4 FP8-it MTP | 17/21 | 0 | 9.2067s | 12.0252s |

Category notes:

- Qwen was much faster and stable after thinking suppression. Its main risk was
  the cross-strait forced-framing pair: it answered the PRC-claim framing as
  settled and produced a one-sided claim about Taiwan's status.
- Gemma4 was slower by roughly 3.3x on p50 and 2.2x on p90 in this small
  stance corpus. It handled the forced-framing prompts better than Qwen, but
  classified four Taiwan-sensitive analysis prompts as `settled` rather than
  `contested`, which suggests over-structuring the task as a procedural answer.
- Neither model set `unsupported_certainty=true`; that field is still useful as
  a smoke signal, but mismatch inspection remains necessary.

Current recommendation: keep Qwen DFlash 262K as the operational default. For
fb-reader stance-sensitive paths, add an explicit Taiwan-status guardrail and
forced-framing regression prompts before relying on either backend's self-audit
labels.
