# dgx-ansible docs

This directory holds durable notes and decision records for the DGX Spark and
model-evaluation work. Raw benchmark and evaluation outputs are generated under
`reports/` and are treated as local artifacts by default.

## Operational Notes

- `dgx-spark-software-update-guidance-2026-05-11.md`: DGX Spark software update
  guidance and operational cautions.
- `qwen36-benchmark-and-candidate-survey-2026-06-11.md`: post-update Qwen3.6
  speed baseline, alternative model survey, and next PrismaQuant/DFlash plan.
- `gemma4-12b-replacement-survey-2026-06-11.md`: Gemma 4 12B replacement survey,
  vLLM unified image check, and DGX Spark smoke/bench result.
- `dgx-spark-vllm-model-selection-2026-05-06.md`: vLLM model-selection notes for
  DGX Spark.
- `acceleration-research-2026-05-05.md`: acceleration and serving research notes.
- `handover-prismaquant.md`: PrismaQuant handoff notes.

## Stance And Steering Experiments

- `dgx-spark-gemma4-qwen-stance-ab-2026-05-19.md`: early Gemma/Qwen stance A/B
  planning and results.
- `qwen-dir-steering-feasibility-2026-05-21.md`: feasibility review for moving
  dir-steering ideas to the Qwen backend.
- `facebook-qwen-steering-reflection-2026-05-22.md`: reader-facing reflection on
  the Qwen steering work.
- `qwen-dir-steering-manual-review-handoff-2026-06-01.md`: manual-review handoff
  for Qwen dir-steering results.
- `qwen-gemma-ds4-stance-experiment-summary-2026-06-03.md`: readable summary of
  the Qwen/Gemma/DS4 stance experiments and backend state.

## Ollama Cloud And DS4 Follow-Up

- `ollama-cloud-routing-audit-2026-06-04.md`: routing, header, latency, and
  endpoint-stability audit for Ollama Cloud DeepSeek V4 Flash.
- `ollama-cloud-support-inquiry-deepseek-v4-flash-2026-06-04.md`: support email
  draft for asking Ollama about DeepSeek V4 Flash data residency and routing.

## Artifact Policy

`reports/` can contain large, repeated, and sometimes sensitive raw outputs. The
repo now ignores newly generated reports by default. Keep durable conclusions in
`docs/`; only force-add report files when they are intentionally curated and
safe to publish.
