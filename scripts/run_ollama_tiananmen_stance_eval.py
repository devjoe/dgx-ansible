#!/usr/bin/env python3
"""Run the Tiananmen stance corpus against Ollama Cloud models."""

from __future__ import annotations

import argparse
import html
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_CORPUS = Path("prompts/tiananmen_1989_stance_corpus.json")
DEFAULT_ENDPOINT = "http://127.0.0.1:11434"
DEFAULT_MODELS = [
    "minimax-m3:cloud",
    "gemma4:31b-cloud",
    "glm-5.1:cloud",
    "deepseek-v4-flash:cloud",
    "deepseek-v4-pro:cloud",
    "kimi-k2.6:cloud",
]


def slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (rank - lower)


def load_cases(corpus_path: Path, variants: set[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    for item in corpus.get("items", []):
        prompt_variants = item.get("prompt_variants") or [
            {
                "id": item.get("prompt_variant_default") or "default",
                "label": "Default",
                "purpose": "Default prompt",
                "prompt": item["prompt"],
            }
        ]
        for variant in prompt_variants:
            variant_id = str(variant["id"])
            if variants and variant_id not in variants:
                continue
            cases.append(
                {
                    "case_id": f"{item['id']}::{variant_id}",
                    "item_id": item["id"],
                    "variant_id": variant_id,
                    "variant_label": variant.get("label"),
                    "variant_purpose": variant.get("purpose"),
                    "category": item.get("category"),
                    "topic": item.get("topic"),
                    "expected_answer_type": item.get("expected_answer_type"),
                    "expected_topic_contestedness": item.get("expected_topic_contestedness"),
                    "compatible_topic_contestedness": item.get("compatible_topic_contestedness"),
                    "target_claim": item.get("target_claim"),
                    "expected_stance_to_target_claim": item.get("expected_stance_to_target_claim"),
                    "compatible_stance_to_target_claim": item.get("compatible_stance_to_target_claim"),
                    "expected_prompt_frame_handling": item.get("expected_prompt_frame_handling"),
                    "compatible_prompt_frame_handling": item.get("compatible_prompt_frame_handling"),
                    "review_focus": item.get("review_focus"),
                    "review_rubric": item.get("review_rubric") or [],
                    "prompt": variant["prompt"],
                }
            )
    return corpus, cases


def post_chat(
    endpoint: str,
    model: str,
    prompt: str,
    timeout: float,
    max_tokens: int,
) -> tuple[int | None, dict[str, Any] | None, str | None, float]:
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return a reader-facing answer only. Use the language requested "
                    "by the user. Do not include self-audit or hidden reasoning."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_predict": max_tokens,
        },
    }
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/api/chat",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            elapsed = time.perf_counter() - started
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
            return response.status, payload, None, elapsed
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - started
        raw = exc.read().decode("utf-8", errors="replace")
        return exc.code, None, raw[:1200], elapsed
    except Exception as exc:  # noqa: BLE001 - preserve provider/runtime failures.
        elapsed = time.perf_counter() - started
        return None, None, repr(exc), elapsed


def message_text(payload: dict[str, Any] | None) -> str:
    message = (payload or {}).get("message") or {}
    content = message.get("content")
    return content.strip() if isinstance(content, str) else ""


def reasoning_text(payload: dict[str, Any] | None) -> str:
    message = (payload or {}).get("message") or {}
    content = message.get("thinking") or message.get("reasoning")
    return content.strip() if isinstance(content, str) else ""


def completion_tokens(payload: dict[str, Any] | None) -> int | None:
    value = (payload or {}).get("eval_count")
    return value if isinstance(value, int) else None


def run_one(
    endpoint: str,
    model: str,
    case: dict[str, Any],
    timeout: float,
    max_tokens: int,
    retry_max_tokens: int,
) -> dict[str, Any]:
    attempts = []
    for attempt, token_budget in enumerate([max_tokens, retry_max_tokens], start=1):
        status, payload, error, elapsed = post_chat(endpoint, model, case["prompt"], timeout, token_budget)
        answer = message_text(payload)
        reasoning = reasoning_text(payload)
        attempts.append(
            {
                "attempt": attempt,
                "status": status,
                "http_ok": status == 200,
                "latency_s": round(elapsed, 4),
                "completion_tokens": completion_tokens(payload),
                "answer": answer,
                "answer_chars": len(answer),
                "reasoning": reasoning,
                "reasoning_chars": len(reasoning),
                "error": error,
                "max_tokens": token_budget,
            }
        )
        if status == 200 and answer:
            break
        if status not in {500, 502, 503, 504, 200, None}:
            break
    selected = attempts[-1]
    return {
        **case,
        "model": model,
        "status": selected["status"],
        "http_ok": selected["http_ok"],
        "latency_s": selected["latency_s"],
        "completion_tokens": selected["completion_tokens"],
        "answer": selected["answer"],
        "answer_chars": selected["answer_chars"],
        "reasoning": selected["reasoning"],
        "reasoning_chars": selected["reasoning_chars"],
        "error": selected["error"],
        "attempts": attempts,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"all": summarize_scope(rows)}
    for model in sorted({row["model"] for row in rows}):
        scoped = [row for row in rows if row["model"] == model]
        summary[f"model:{model}"] = summarize_scope(scoped)
    for variant in sorted({row["variant_id"] for row in rows}):
        scoped = [row for row in rows if row["variant_id"] == variant]
        summary[f"variant:{variant}"] = summarize_scope(scoped)
    return summary


def summarize_scope(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [row["latency_s"] for row in rows if row.get("http_ok")]
    tokens = [row["completion_tokens"] for row in rows if isinstance(row.get("completion_tokens"), int)]
    return {
        "n": len(rows),
        "http_ok": sum(1 for row in rows if row.get("http_ok")),
        "empty_answer": sum(1 for row in rows if row.get("http_ok") and not row.get("answer")),
        "endpoint_failure": sum(1 for row in rows if not row.get("http_ok")),
        "latency_p50_s": round(percentile(latencies, 0.5), 4) if latencies else None,
        "latency_p90_s": round(percentile(latencies, 0.9), 4) if latencies else None,
        "completion_tokens_mean": round(statistics.fmean(tokens), 2) if tokens else None,
    }


def render_html(report: dict[str, Any]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in report["results"]:
        grouped.setdefault(str(row["case_id"]), []).append(row)
    cards = []
    for case_id, rows in grouped.items():
        first = rows[0]
        answer_blocks = []
        for row in sorted(rows, key=lambda value: value["model"]):
            body = row.get("answer") or row.get("reasoning") or row.get("error") or ""
            status = "ok" if row.get("http_ok") and row.get("answer") else "fail"
            answer_blocks.append(
                f"""
                <section class="answer {status}">
                  <div class="answer-head">
                    <strong>{html.escape(str(row['model']))}</strong>
                    <span>status={html.escape(str(row.get('status')))}</span>
                    <span>{html.escape(str(row.get('latency_s')))}s</span>
                    <span>{html.escape(str(row.get('completion_tokens')))} tok</span>
                  </div>
                  <pre>{html.escape(str(body))}</pre>
                </section>
                """
            )
        rubric = "".join(f"<li>{html.escape(str(point))}</li>" for point in first.get("review_rubric", []))
        cards.append(
            f"""
            <article class="case" id="{html.escape(case_id)}">
              <div class="meta">
                <strong>{html.escape(str(case_id))}</strong>
                <span>{html.escape(str(first.get('category')))}</span>
                <span>{html.escape(str(first.get('topic')))}</span>
                <span>{html.escape(str(first.get('variant_id')))}</span>
              </div>
              <h2>{html.escape(str(first.get('variant_label') or first.get('variant_id')))}</h2>
              <div class="prompt">{html.escape(str(first.get('prompt') or ''))}</div>
              <div class="rubric"><strong>Manual rubric</strong><ol>{rubric}</ol></div>
              <div class="answers">{''.join(answer_blocks)}</div>
            </article>
            """
        )
    summary = report["summary"]["all"]
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ollama Cloud Tiananmen Stance Eval</title>
  <style>
    :root {{ color-scheme: light; --bg:#f6f7f9; --panel:#fff; --text:#17202b; --muted:#667085; --line:#d8dee8; --soft:#f9fafb; --ok:#176548; --bad:#b42318; --blue:#1f4e8c; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ max-width:1280px; margin:0 auto; padding:28px 20px 64px; }}
    h1 {{ margin:0 0 6px; font-size:32px; line-height:1.15; letter-spacing:0; }}
    h2 {{ margin:10px 0; font-size:18px; letter-spacing:0; }}
    .sub {{ color:var(--muted); margin-bottom:20px; }}
    .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; margin:18px 0 24px; }}
    .stat,.case {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; }}
    .stat {{ padding:12px; }}
    .stat span {{ display:block; color:var(--muted); font-size:12px; }}
    .stat strong {{ display:block; font-size:22px; }}
    .case {{ padding:16px; margin:14px 0; }}
    .meta,.answer-head {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; color:var(--muted); font-size:12px; }}
    .meta span,.answer-head span {{ border:1px solid var(--line); border-radius:999px; padding:2px 8px; }}
    .prompt,.rubric {{ background:var(--soft); border:1px solid var(--line); border-radius:8px; padding:12px; margin:10px 0; }}
    .rubric ol {{ margin:6px 0 0; padding-left:22px; }}
    .answers {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }}
    .answer {{ border:1px solid var(--line); border-radius:8px; padding:12px; min-width:0; }}
    .answer.ok {{ border-color:#b7ddc9; }}
    .answer.fail {{ border-color:#f1b6b0; }}
    .answer-head strong {{ color:var(--blue); }}
    pre {{ white-space:pre-wrap; overflow-wrap:anywhere; margin:10px 0 0; font:14px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    @media (max-width:900px) {{ .answers {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>Ollama Cloud Tiananmen Stance Eval</h1>
  <div class="sub">Generated {html.escape(report['generated_at'])} · corpus={html.escape(report['corpus'])}</div>
  <section class="stats">
    <div class="stat"><span>Total</span><strong>{summary['n']}</strong></div>
    <div class="stat"><span>HTTP OK</span><strong>{summary['http_ok']}/{summary['n']}</strong></div>
    <div class="stat"><span>Failures</span><strong>{summary['endpoint_failure']}</strong></div>
    <div class="stat"><span>p50 Latency</span><strong>{summary['latency_p50_s'] or ''}s</strong></div>
  </section>
  {''.join(cards)}
</main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--variants", default="", help="Comma-separated variant ids. Empty runs all variants.")
    parser.add_argument("--case-ids", default="", help="Comma-separated expanded case ids such as item::variant.")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/ollama-cloud-tiananmen"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--retry-max-tokens", type=int, default=1200)
    args = parser.parse_args()

    models = [value.strip() for value in args.models.split(",") if value.strip()]
    variants = {value.strip() for value in args.variants.split(",") if value.strip()}
    case_ids = {value.strip() for value in args.case_ids.split(",") if value.strip()}
    corpus, cases = load_cases(args.corpus, variants)
    if case_ids:
        cases = [case for case in cases if case["case_id"] in case_ids]
    if args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No cases selected")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    total = len(models) * len(cases)
    print(f"Running {total} calls: {len(models)} models x {len(cases)} cases", flush=True)
    index = 0
    for case in cases:
        for model in models:
            index += 1
            row = run_one(args.endpoint, model, case, args.timeout, args.max_tokens, args.retry_max_tokens)
            rows.append(row)
            status = row.get("status")
            chars = row.get("answer_chars")
            print(
                f"[{index}/{total}] {model} {case['case_id']} status={status} chars={chars}",
                flush=True,
            )

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    model_slug = "matrix" if len(models) > 1 else slug(models[0])
    base = f"{stamp}-{model_slug}"
    report = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoint": args.endpoint,
        "models": models,
        "corpus": str(args.corpus),
        "corpus_schema_version": corpus.get("schema_version"),
        "prompt_variant_policy": corpus.get("prompt_variant_policy"),
        "summary": summarize(rows),
        "results": rows,
    }
    json_path = args.out_dir / f"{base}.json"
    jsonl_path = args.out_dir / f"{base}.jsonl"
    html_path = args.out_dir / f"{base}.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    html_path.write_text(render_html(report), encoding="utf-8")
    print(json.dumps(report["summary"]["all"], ensure_ascii=False, indent=2), flush=True)
    print(json_path, flush=True)
    print(jsonl_path, flush=True)
    print(html_path, flush=True)
    return 0 if all(row.get("http_ok") for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
