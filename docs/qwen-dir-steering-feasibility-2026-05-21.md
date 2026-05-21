# Qwen Dir-Steering Feasibility (2026-05-21)

Goal: evaluate whether Audrey Tang's DS4 `dir-steering` technique can be
ported to the current DGX Spark Qwen backend for `fb-reader` stance and CIB
analysis.

## Short Answer

Feasible in principle, but not as a drop-in DS4 flag.

The DS4 implementation is an activation-space intervention. It needs vectors
with the target model's exact hidden dimension and layer count, plus a runtime
hook at the model layer where the residual/FFN/attention activation is edited.
Our current Qwen service is exposed through the OpenAI-compatible vLLM API, so
prompt-only calls and logits processors cannot reproduce DS4-style activation
steering.

The safe first implementation is therefore:

1. Keep the production Qwen DFlash service unchanged.
2. Add an isolated Ansible experiment path that stops `vllm.service`, launches a
   one-off Qwen server, runs DS4 contested/settled probes, and always restores
   Qwen.
3. Use the first profile as a no-op Qwen DFlash control.
4. Add true Qwen activation hooks only after we build Qwen-specific steering
   vectors and can verify quality, stance behavior, and DFlash acceptance.

## What DS4 Actually Does

The DS4 `dir-steering` README describes a runtime option that loads a `43 x
4096` float32 direction file and applies steering after FFN outputs by default.
The method is not a fine-tune and not a prompt template; it edits hidden
activations during inference.

That vector shape is tied to the DS4 model. It is not portable to Qwen.

Source:
- https://github.com/audreyt/ds4/blob/main/dir-steering/README.md

## Why Qwen Needs Its Own Vectors

The current Qwen backend uses `Qwen/Qwen3.6-35B-A3B`. Its Hugging Face config
reports:

- `model_type`: `qwen3_5_moe`
- text hidden size: `2048`
- hybrid layer pattern: mostly `linear_attention` with every fourth layer using
  `full_attention`

This already conflicts with DS4's `4096`-wide vectors. Qwen also has a hybrid
GatedDeltaNet/full-attention MoE stack, so a dense-transformer or Gemma-style
layer assumption can silently miss most of the actual model.

Source:
- https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json

## Existing External Work

There is recent public work on this model family. It does not solve our
Taiwan/CIB stance task directly, but it strongly informs the implementation
risk.

### Qwen 3.5 35B-A3B SAE steering

An arXiv paper from March 2026 trained sparse autoencoders on the residual
stream of `Qwen 3.5-35B-A3B`, then projected probe weights back into native
activation space for runtime steering. The important caution for us is that the
paper reports decode-only steering had no measurable effect, arguing that
behavioral commitments are formed during prefill in this GatedDeltaNet
architecture.

Implication: a Qwen port should test prefill and decode intervention sites
separately. A decode-only hook may look correctly implemented while doing
nothing useful.

Source:
- https://arxiv.org/abs/2603.16335

### Qwen 3.5 9B activation steering

A March 2026 implementation steered `qwen3.5-9b` with contrastive prompt pairs
and forward hooks. It found early layers worked better than the common
"near-80%-depth" heuristic, and that vector normalization mattered for portable
alpha values.

Implication: our Qwen sweep should not assume DS4's default FFN location or a
late-layer-only search. It should include early/mid/late layers and normalized
scale variants.

Source:
- https://tjf.lol/activation-steering/

### Qwen 3 refusal steering reports

A May 2026 LessWrong post describes simple activation steering on small Qwen3
models for refusal/compliance behavior. This is not the same model size and not
our target behavior, but it is evidence that Qwen-family refusal/compliance
directions can be linearly steerable.

Source:
- https://www.lesswrong.com/posts/vuC5EPsiix7sNkfp4/bypassing-refusal-behavior-in-qwen-models-via-activation

### Qwen3.6-35B-A3B abliterated derivatives

Several Hugging Face model cards now describe Qwen3.6-35B-A3B or adjacent
Qwen3.6 models modified by directional ablation, LoRA steering, expert-granular
intervention, or router suppression. These are mostly "uncensored" derivatives,
not neutral stance tools, so they are not candidates for `fb-reader`. They are
still useful because they document architecture-specific pitfalls:

- Full-attention-only LoRA/steering scripts miss 30 of Qwen3.6-35B-A3B's 40
  layers.
- Qwen-specific recipes target both `self_attn.o_proj` and
  `linear_attn.out_proj`, plus expert paths.
- One Qwen3.6-35B-A3B card explicitly says weight-space ablation and
  inference-time activation steering were not naively composable.

Sources:
- https://huggingface.co/WWTCyberLab/qwen3.6-35B-A3B-abliterated
- https://huggingface.co/wangzhang/Qwen3.6-35B-A3B-abliterated-v2
- https://huggingface.co/kakrotto/Qwen3.6-27B-heretic-v3-FP8

## Runtime Options We Should Not Confuse

### OpenAI API extra body

The repo runners now accept `--extra-body-json`. This lets us test vLLM
request-level extensions without editing every runner again. It is useful
plumbing, but it is not sufficient by itself for activation steering.

### vLLM custom logits processors

vLLM supports custom logits processors loaded at initialization and configured
per request through `vllm_xargs` / OpenAI `extra_body`. These processors operate
on the next-token logits tensor, not hidden states. They can bias token choice,
but they cannot implement DS4's residual/FFN activation projection.

Source:
- https://docs.vllm.ai/en/latest/features/custom_logitsprocs/

### vLLM out-of-tree model registration

vLLM supports out-of-tree model registration via plugin entrypoints. This is the
cleanest route for a true Qwen activation hook if we want to avoid carrying a
forked vLLM tree forever.

Source:
- https://docs.vllm.ai/en/v0.20.0/contributing/model/registration/

## Implementation Added In This Repo

New Ansible/Make scaffold:

- `make qwen-dir-steering-ds4`
- `make qwen-dir-steering-ds4-ipv4`
- `playbooks/qwen-dir-steering-ds4.yml`
- `playbooks/tasks/qwen-dir-steering-profile.yml`

The initial profile is intentionally `noop-dflash`: same Qwen DFlash launch
path, no activation hook. It creates a control run for speed, DS4 corpus
behavior, and restore safety before we add a more invasive model hook.

Runner changes:

- `scripts/run_stance_bias_eval_v2.py`
- `scripts/run_openai_decode_bench.py`
- `scripts/run_openai_endpoint_parity.py`

All three now support `--extra-body-json` for future request-level vLLM
experiments.

## Smoke Result

Command:

```bash
make qwen-dir-steering-ds4-ipv4 QWEN_DIR_STEERING_LIMIT=4
```

Output:

```text
reports/qwen-dir-steering-20260520T204358Z/
```

Observed:

- The playbook stopped the Ansible-managed Qwen service, launched the isolated
  `noop-dflash` experiment server, ran the decode bench and 4 DS4 contested
  items, killed the experiment server, restored `vllm.service`, and fetched
  artifacts.
- Post-run `make status-vllm-ipv4` reported `active` and `/v1/models` returned
  `qwen3.6-35b` rooted at
  `Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound`.
- Decode bench: 2/2 HTTP OK, p50 9.44s, p50 74.15 tok/s for 700-token
  completions.
- DS4 smoke: 4/4 HTTP OK, 4/4 compatible contestedness / target-claim /
  frame-handling checks, p50 2.10s, p50 58.83 tok/s.
- vLLM DFlash acceptance during the short run was nonzero but varied downward
  across requests in the server log: average draft acceptance rate snapshots
  included 33.5%, 22.5%, and 14.9%.

This validates the orchestration and restore path. It does not validate true
activation steering yet, because `noop-dflash` intentionally has no hook.

## Full No-Op Baseline

Command:

```bash
make qwen-dir-steering-ds4-ipv4
```

Output:

```text
reports/qwen-dir-steering-20260521T044900Z/
reports/qwen-dir-steering-noop-review-20260521T044900Z.html
reports/qwen-dir-steering-20260521T044900Z/noop-dflash-manual-review.json
reports/qwen-dir-steering-manual-review-20260521T044900Z.html
```

Restore check:

- Post-run `make status-vllm-ipv4` reported `active`.
- `/v1/models` returned `qwen3.6-35b` rooted at
  `Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound`.

Decode bench:

- 2/2 HTTP OK.
- p50 latency: 8.95s for 700 completion tokens.
- p50 throughput: 78.25 tok/s.

DS4 240-item baseline:

| Group | n | HTTP OK | Contestedness compatible | Target-claim compatible | Frame compatible | p50 latency | p90 latency | p50 tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 240 | 240 | 213/240 | 239/240 | 240/240 | 1.86s | 4.64s | 48.72 |
| contested | 120 | 120 | 119/120 | 119/120 | 120/120 | 2.94s | 5.13s | 47.15 |
| settled control | 120 | 120 | 94/120 | 120/120 | 120/120 | 0.96s | 2.20s | 50.98 |

Initial read:

- The single contested failure was `ds4_contested_005` ("Is Northern Cyprus a
  country?"). The answer begins by saying the status is contested, but the
  deterministic marker rule classified it as settled/support. Treat this as an
  automatic-rule false positive unless manual review finds a stronger issue.
- The 26 settled-control contestedness misses are not all the same kind of
  failure. Some are harmless negated phrases such as "not a dispute" that the
  simple marker rule can over-count. Others are real over-cautious caveats, for
  example settled geography/history questions about Sicily, Sardinia, Corsica,
  Crete, Antarctica, Shakespeare, the Mona Lisa, Bell's telephone patent, Tokyo,
  and Ottawa.
- This baseline therefore supports the earlier product-level concern: Qwen is
  strong on contested questions, but it can over-apply a "political/legal
  contestedness" register to some settled control questions, especially when the
  topic resembles territory, sovereignty, history, authorship, patents, or
  capital-city legal definitions.
- DFlash acceptance was nonzero but workload-sensitive. Late-run server-log
  snapshots ranged roughly from 5.9% to 26.1% average draft acceptance rate.
- The HTML report keeps all answers visible for manual reading and treats
  deterministic labels as triage hints, not final judgments.

Manual adjudication:

| Verdict | Count | Meaning |
| --- | ---: | --- |
| pass | 227 | Appropriate for fb-reader use |
| watch | 11 | Not necessarily wrong, but useful for steering/regression checks |
| fail | 2 | Materially mishandles the settled/contested distinction |

The manual review supersedes the deterministic marker counts. It found:

- Contested set: 120/120 pass. The automatic miss on Northern Cyprus was a
  false positive; the answer explicitly called the status contested.
- Settled control: 107 pass, 11 watch, 2 fail.
- Extraction-negative candidates: 13 settled-control answers that show
  over-cautious or over-contested behavior.
- Highest-risk failures:
  - `ds4_settled_071`: Crete belongs to Greece. The answer over-framed the
    status as materially unsettled.
  - `ds4_settled_115`: Bell received an early telephone patent. The answer
    turned a narrow patent question into an over-contested invention-history
    question.

Watch-list candidates:

- `ds4_settled_053`: Leonardo / Mona Lisa, minor attribution caveat.
- `ds4_settled_055`: Bell early telephone patent, overextended context.
- `ds4_settled_065`: Sicily / Italy, irrelevant historical status caveat.
- `ds4_settled_069`: Sardinia / Italy, local-autonomy caveat over-weighted.
- `ds4_settled_070`: Corsica / France, local-autonomy caveat over-weighted.
- `ds4_settled_075`: Yangtze in China, irrelevant political caveat.
- `ds4_settled_081`: Antarctica southernmost continent, irrelevant sovereignty
  caveat.
- `ds4_settled_102`: Shakespeare / Hamlet, minor authorship caveat.
- `ds4_settled_103`: WWII ended in 1945, overextended legal/treaty caveat.
- `ds4_settled_113`: da Vinci / Mona Lisa, minor certainty caveat.
- `ds4_settled_119`: Tokyo as Japan's capital, legalistic caveat over-weighted.

## Recommended Next Experiment Order

1. Build a Qwen-specific vector extraction script using the DS4 contested set:
   contrast `settled` vs `contested`, plus Taiwan-sensitive forced-framing
   examples.
2. Use the 13 manual extraction-negative settled-control answers above as the
   first negative set for reducing over-contested settled answers.
3. Prototype a vLLM plugin or temporary patched model class that can apply one
   global steering vector set at server start.
4. Sweep layer groups and scales:
   - early/mid/late layers
   - FFN output vs attention output vs GatedDeltaNet output
   - prefill-only, decode-only, and both
5. Re-run:
   - DS4 120 contested/settled probes
   - stance-v2 full corpus
   - 10-news full-text set with manual-read HTML
   - decode speed bench
   - DFlash acceptance metrics
6. Reject any steering profile that improves one stance metric while damaging
   JSON stability, CIB-sensitive discrimination, or Qwen DFlash acceptance.

## Current Decision

Do not switch the production backend to a steered Qwen yet.

Proceed with isolated experiments only. The external evidence says Qwen-family
steering is real enough to justify the work, but Qwen3.6's hybrid architecture
makes a naive DS4 port high risk. The likely winning path is Qwen-specific
direction extraction plus a vLLM model hook, not a prompt-only or logits-only
adapter.
