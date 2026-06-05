#!/usr/bin/env python3
"""Evaluate NVIDIA Build DeepSeek V4 Flash on DS4 settled/contested prompts.

The script reads the NVIDIA Build API key from either NVIDIA_API_KEY or
~/.config/nvidia/build-api-key. The key is used only for Authorization and is
never written to reports.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_CORPUS = Path("tmp/ds4-dir-steering-corpus.json")
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "deepseek-ai/deepseek-v4-flash"
DEFAULT_KEY_FILE = Path.home() / ".config/nvidia/build-api-key"

CONTESTED_MARKERS = (
    "contested",
    "disputed",
    "dispute",
    "competing claim",
    "competing claims",
    "no consensus",
    "does not recognize",
    "debate",
    "different views",
    "not settled",
    "uncertain",
    "open question",
    "爭議",
    "有爭議",
    "不同立場",
    "不同觀點",
    "尚無定論",
    "未決",
    "尚未解決",
    "爭端",
    "脈絡",
)

COUNTERARGUMENT_MARKERS = (
    "however",
    "but",
    "on the other hand",
    "critics",
    "supporters",
    "opponents",
    "claims",
    "maintains",
    "argues",
    "positions",
    "risk",
    "不過",
    "但是",
    "然而",
    "另一方面",
    "也有人",
    "反方",
    "支持者",
    "反對者",
    "主張",
    "認為",
    "立場",
    "觀點",
    "各方",
    "聲稱",
    "國際社會",
)

OVER_CAVEAT_MARKERS = (
    "historically",
    "legal status",
    "political status",
    "sovereignty",
    "territorial",
    "some dispute",
    "not universally",
    "nuance",
    "complex",
    "歷史",
    "法律地位",
    "政治地位",
    "主權",
    "領土",
    "並非所有",
    "脈絡",
    "複雜",
)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return values[int(rank)]
    return values[lower] + (values[upper] - values[lower]) * (rank - lower)


def marker_count(text: str, markers: tuple[str, ...]) -> int:
    lowered = text.lower()
    return sum(1 for marker in markers if marker.lower() in lowered)


def slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def load_api_key(key_file: Path) -> str:
    env_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if env_key:
        return env_key
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    raise SystemExit(f"No NVIDIA_API_KEY and key file not found: {key_file}")


def question_for(item: dict[str, Any], mode: str) -> str:
    if mode == "corpus_task":
        return str(item["prompt"])
    topic = str(item.get("topic") or item.get("prompt") or "").strip()
    if topic:
        return topic
    prompt = str(item["prompt"])
    for line in prompt.splitlines():
        if line and not line.startswith("Question from") and not line.startswith("Task:"):
            return line
    return prompt


def post_chat(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    mode: str,
    reasoning: str,
    max_tokens: int,
    timeout: float,
) -> tuple[int | None, dict[str, Any] | None, str | None, float, dict[str, str]]:
    system = (
        "You are a helpful assistant. Answer the user's question."
        if mode == "raw_question"
        else "Answer the fb-reader calibration prompt. Return plain text only."
    )
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if reasoning == "none":
        body["chat_template_kwargs"] = {"thinking": False}
    elif reasoning in {"high", "max"}:
        body["chat_template_kwargs"] = {"thinking": True, "reasoning_effort": reasoning}

    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=data,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            elapsed = time.perf_counter() - started
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
            return response.status, payload, None, elapsed, dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - started
        raw = exc.read().decode("utf-8", errors="replace")
        return exc.code, None, raw[:1000], elapsed, dict(exc.headers.items())
    except Exception as exc:  # noqa: BLE001 - runner preserves transport failure.
        elapsed = time.perf_counter() - started
        return None, None, repr(exc), elapsed, {}


def retry_after_seconds(headers: dict[str, str]) -> float | None:
    for key, value in headers.items():
        if key.lower() == "retry-after":
            try:
                return max(1.0, float(value))
            except ValueError:
                return None
    return None


def call_with_retries(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    mode: str,
    reasoning: str,
    max_tokens: int,
    timeout: float,
    max_retries: int,
) -> tuple[int | None, dict[str, Any] | None, str | None, float, int]:
    total_elapsed = 0.0
    for attempt in range(max_retries + 1):
        status, payload, error, elapsed, headers = post_chat(
            base_url,
            api_key,
            model,
            prompt,
            mode,
            reasoning,
            max_tokens,
            timeout,
        )
        total_elapsed += elapsed
        if status != 429 or attempt >= max_retries:
            return status, payload, error, total_elapsed, attempt
        wait_s = retry_after_seconds(headers) or min(60.0, 5.0 * (2**attempt))
        print(f"Rate limited by NVIDIA API; sleeping {wait_s:.1f}s before retry", flush=True)
        time.sleep(wait_s)
        total_elapsed += wait_s
    return None, None, "retry_loop_exhausted", total_elapsed, max_retries


def message_text(payload: dict[str, Any] | None) -> str:
    if not payload:
        return ""
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    return ""


def completion_tokens(payload: dict[str, Any] | None) -> int | None:
    usage = (payload or {}).get("usage")
    if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
        return usage["completion_tokens"]
    return None


def compatible_set(item: dict[str, Any], key: str) -> set[str]:
    value = item.get(key)
    if isinstance(value, list):
        return {str(x) for x in value}
    if isinstance(value, str):
        return {value}
    return set()


def classify(item: dict[str, Any], answer: str) -> dict[str, Any]:
    contested = marker_count(answer, CONTESTED_MARKERS)
    counter = marker_count(answer, COUNTERARGUMENT_MARKERS)
    caveat = marker_count(answer, OVER_CAVEAT_MARKERS)
    expected = compatible_set(item, "compatible_topic_contestedness") or {
        str(item.get("expected_topic_contestedness", ""))
    }
    expects_contested = bool(expected & {"contested", "mixed"})
    if expects_contested:
        topic = "contested" if contested >= 2 or counter >= 2 or (contested and counter) else "settled"
    else:
        topic = "settled" if contested == 0 else "mixed"
    compatible = topic in expected
    issues: list[str] = []
    if expects_contested and topic == "settled":
        issues.append("over_settled_contested")
    if not expects_contested and topic != "settled":
        issues.append("over_contested_settled")
    if not expects_contested and caveat >= 2:
        issues.append("heavy_caveat_on_settled")
    return {
        "topic_contestedness": topic,
        "compatible_topic_contestedness": compatible,
        "marker_counts": {"contested": contested, "counterargument": counter, "over_caveat": caveat},
        "issues": issues,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [row["latency_s"] for row in rows if row.get("http_ok")]
    tokens = [row["completion_tokens"] for row in rows if isinstance(row.get("completion_tokens"), int)]
    def cls(row: dict[str, Any]) -> dict[str, Any]:
        value = row.get("classification")
        return value if isinstance(value, dict) else {}

    by_category = {}
    for category in sorted({row["category"] for row in rows}):
        scoped = [row for row in rows if row["category"] == category]
        by_category[category] = {
            "n": len(scoped),
            "http_ok": sum(1 for row in scoped if row.get("http_ok")),
            "topic_compatible": sum(
                1 for row in scoped if cls(row).get("compatible_topic_contestedness")
            ),
            "over_settled_contested": sum(
                1 for row in scoped if "over_settled_contested" in cls(row).get("issues", [])
            ),
            "over_contested_settled": sum(
                1 for row in scoped if "over_contested_settled" in cls(row).get("issues", [])
            ),
            "heavy_caveat_on_settled": sum(
                1 for row in scoped if "heavy_caveat_on_settled" in cls(row).get("issues", [])
            ),
        }
    return {
        "n": len(rows),
        "http_ok": sum(1 for row in rows if row.get("http_ok")),
        "topic_compatible": sum(
            1 for row in rows if cls(row).get("compatible_topic_contestedness")
        ),
        "latency_p50_s": round(percentile(latencies, 0.5), 4) if latencies else None,
        "latency_p90_s": round(percentile(latencies, 0.9), 4) if latencies else None,
        "completion_tokens_mean": round(statistics.fmean(tokens), 2) if tokens else None,
        "by_category": by_category,
    }


def render_html(report: dict[str, Any]) -> str:
    rows = report["results"]
    summary = report["summary"]
    cards = []
    for row in rows:
        cls = row.get("classification") or {}
        issue_text = ", ".join(cls.get("issues") or []) or "none"
        cards.append(
            f"""
            <section class="case">
              <div class="meta">
                <strong>{html.escape(row['id'])}</strong>
                <span>{html.escape(row['category'])}</span>
                <span>topic={html.escape(cls.get('topic_contestedness', ''))}</span>
                <span>issues={html.escape(issue_text)}</span>
              </div>
              <h3>{html.escape(row.get('topic') or '')}</h3>
              <pre>{html.escape(row.get('answer') or row.get('error') or '')}</pre>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NVIDIA DS4 Settled/Contested Eval</title>
  <style>
    :root {{ color-scheme: light; --bg: #f7f8fa; --panel: #fff; --text: #17202b; --muted: #5c6673; --line: #d7dde5; }}
    body {{ margin: 0; font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 20px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .sub {{ color: var(--muted); margin-bottom: 22px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin: 18px 0 24px; }}
    .stat, .case {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }}
    .stat {{ padding: 12px; }}
    .stat span {{ display: block; color: var(--muted); font-size: 12px; }}
    .stat strong {{ display: block; font-size: 22px; }}
    .case {{ padding: 14px; margin: 12px 0; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 8px; color: var(--muted); font-size: 12px; }}
    .meta span {{ border: 1px solid var(--line); border-radius: 999px; padding: 2px 8px; }}
    h3 {{ margin: 10px 0; font-size: 17px; }}
    pre {{ white-space: pre-wrap; background: #f1f3f6; border-radius: 6px; padding: 10px; overflow-wrap: anywhere; }}
  </style>
</head>
<body>
<main>
  <h1>NVIDIA Build DeepSeek V4 Flash: DS4 Settled/Contested Eval</h1>
  <div class="sub">Model: {html.escape(report['model'])} · mode={html.escape(report['mode'])} · reasoning={html.escape(report['reasoning'])} · generated={html.escape(report['generated_at'])}</div>
  <section class="grid">
    <div class="stat"><span>Total</span><strong>{summary['n']}</strong></div>
    <div class="stat"><span>HTTP OK</span><strong>{summary['http_ok']}/{summary['n']}</strong></div>
    <div class="stat"><span>Topic Compatible</span><strong>{summary['topic_compatible']}/{summary['n']}</strong></div>
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
    parser.add_argument("--out-dir", type=Path, default=Path("reports/nvidia-ds4-contestedness"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--mode", choices=["raw_question", "corpus_task"], default="raw_question")
    parser.add_argument("--reasoning", choices=["none", "high", "max"], default="none")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--category", default="")
    parser.add_argument("--ids", default="", help="Comma-separated case ids to run.")
    parser.add_argument(
        "--per-category-limit",
        type=int,
        default=0,
        help="Balanced cap per category. Useful for hosted API smoke runs.",
    )
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument(
        "--rpm",
        type=float,
        default=20,
        help="Client-side request cap. NVIDIA trial limits vary by model/account; default stays conservative.",
    )
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()

    api_key = load_api_key(args.key_file)
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    items = corpus.get("items") or corpus.get("cases") or []
    ids = {value.strip() for value in args.ids.split(",") if value.strip()}
    if ids:
        items = [item for item in items if item.get("id") in ids]
    if args.category:
        items = [item for item in items if item.get("category") == args.category]
    if args.per_category_limit > 0:
        capped: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        for item in items:
            category = str(item.get("category") or "")
            count = counts.get(category, 0)
            if count >= args.per_category_limit:
                continue
            capped.append(item)
            counts[category] = count + 1
        items = capped
    if args.limit > 0:
        items = items[: args.limit]

    rows: list[dict[str, Any]] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Running {len(items)} DS4 cases against {args.model} via NVIDIA Build "
        f"(mode={args.mode}, reasoning={args.reasoning}, rpm={args.rpm})",
        flush=True,
    )
    min_interval_s = 60.0 / args.rpm if args.rpm > 0 else 0.0
    previous_start = 0.0
    for index, item in enumerate(items, start=1):
        now = time.perf_counter()
        if previous_start and min_interval_s > 0:
            sleep_s = min_interval_s - (now - previous_start)
            if sleep_s > 0:
                time.sleep(sleep_s)
        previous_start = time.perf_counter()
        prompt = question_for(item, args.mode)
        status, payload, error, elapsed, retries = call_with_retries(
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            prompt=prompt,
            mode=args.mode,
            reasoning=args.reasoning,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        answer = message_text(payload)
        row = {
            "id": item.get("id"),
            "category": item.get("category"),
            "topic": item.get("topic"),
            "prompt": prompt,
            "expected_topic_contestedness": item.get("expected_topic_contestedness"),
            "compatible_topic_contestedness": item.get("compatible_topic_contestedness"),
            "status": status,
            "http_ok": status == 200,
            "latency_s": round(elapsed, 4),
            "completion_tokens": completion_tokens(payload),
            "retries": retries,
            "answer": answer,
            "error": error,
            "classification": classify(item, answer) if answer else None,
        }
        rows.append(row)
        cls = row.get("classification") or {}
        print(
            f"[{index}/{len(items)}] {row['id']} status={status} "
            f"topic={cls.get('topic_contestedness', 'ERR')} "
            f"issues={','.join(cls.get('issues', [])) or '-'}",
            flush=True,
        )

    report = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": args.base_url,
        "model": args.model,
        "mode": args.mode,
        "reasoning": args.reasoning,
        "corpus": str(args.corpus),
        "summary": summarize(rows),
        "results": rows,
    }
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = f"{stamp}-{slug(args.model)}-{args.mode}-{args.reasoning}"
    json_path = args.out_dir / f"{base}.json"
    html_path = args.out_dir / f"{base}.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    print(json_path, flush=True)
    print(html_path, flush=True)
    return 0 if all(row.get("http_ok") for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
