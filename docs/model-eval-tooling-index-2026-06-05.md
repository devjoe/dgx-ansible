# Model Evaluation Tooling Index

Date: 2026-06-05

This note maps the current evaluation scripts, prompt corpora, and durable
reports so future runs do not require rediscovering the workflow from shell
history.

## Prompt Corpora

- `prompts/ds4/contested.txt`: upstream-style contested examples used for
  dir-steering and settled/contested checks.
- `prompts/ds4/settled.txt`: upstream-style settled control examples.
- `prompts/stance_bias_corpus.json`: stance-v2 corpus for Taiwan status,
  framing, risk slices, and general controls.
- `prompts/news_context_stance_corpus.json`: short news-context stance tests.
- `prompts/news_fulltext_stance_sources.json`: fulltext news stance sources.
- `prompts/news_fulltext10_stance_sources.json`: expanded fulltext news source
  set.
- `prompts/qwen_settled_watch_regression.json`: Qwen settled-control watch list.
- `prompts/tiananmen_1989_stance_corpus.json`: Tiananmen stance corpus with
  no-system/system/fb-reader prompt variants.
- `prompts/tiananmen_1989_stance_corpus.html`: human-readable Tiananmen corpus
  review page.

## Qwen Steering Tools

- `scripts/capture_qwen_hidden_directions.py`: extracts hidden-state directions
  for Qwen steering experiments.
- `scripts/launch_vllm_with_qwen_steering.py`: launches vLLM with Qwen steering
  hooks.
- `scripts/qwen_dir_steering_vllm_plugin.py`: vLLM plugin for direction
  steering.
- `scripts/build_qwen_dir_steering_extraction_corpus.py`: builds extraction
  corpus for steering direction capture.
- `scripts/build_qwen_ds4_manual_review.py`: builds manual-review artifacts for
  Qwen DS4-style settled/contested output.

Durable docs:

- `docs/qwen-dir-steering-feasibility-2026-05-21.md`
- `docs/qwen-dir-steering-manual-review-handoff-2026-06-01.md`
- `docs/qwen-gemma-ds4-stance-experiment-summary-2026-06-03.md`

## Stance Evaluation Tools

- `scripts/run_stance_bias_eval.py`: original stance runner.
- `scripts/run_stance_bias_eval_v2.py`: stance-v2 runner, including prompt
  variants and current backend settings.
- `scripts/rescore_stance_bias_eval_v2.py`: deterministic rescoring helper.
- `scripts/render_stance_manual_html.py`: manual-review HTML renderer.
- `scripts/render_news_context_stance_html.py`: news-context stance HTML
  renderer.
- `scripts/build_news_fulltext_stance_corpus.py`: builds fulltext news-context
  test corpora.

Durable docs:

- `docs/dgx-spark-gemma4-qwen-stance-ab-2026-05-19.md`
- `docs/facebook-qwen-steering-reflection-2026-05-22.md`

## Ollama Cloud And NVIDIA API Tools

- `scripts/run_ollama_ds4_contestedness_eval.py`: Ollama Cloud DS4
  settled/contested runner.
- `scripts/run_ollama_tiananmen_stance_eval.py`: Ollama Cloud Tiananmen stance
  runner.
- `scripts/render_ollama_tiananmen_manual_review.py`: Tiananmen manual-review
  HTML renderer.
- `scripts/render_ollama_cloud_eval_report.py`: combined Ollama Cloud report
  renderer for public/internal model comparisons.
- `scripts/run_ollama_cloud_routing_audit.py`: routing, latency, header, and
  error-ref audit runner for local-daemon vs direct Ollama Cloud API paths.
- `scripts/run_nvidia_ds4_contestedness_eval.py`: NVIDIA API DS4
  settled/contested runner.

Durable docs:

- `docs/ollama-cloud-routing-audit-2026-06-04.md`
- `docs/ollama-cloud-support-inquiry-deepseek-v4-flash-2026-06-04.md`

## Artifact Policy

Raw outputs belong under `reports/` and are ignored by default. If a report is
intended to be shared or preserved in git, either summarize it in `docs/` or
force-add only the curated report file.
