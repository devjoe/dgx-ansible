#!/usr/bin/env python3
"""Run a small OpenAI-compatible decode speed benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = (
    "Write a neutral, structured Traditional Chinese analysis of how a reader "
    "should evaluate a contentious Taiwan social-media post. Cover factual "
    "claims, value judgments, framing, missing context, and uncertainty. "
    "Use clear section labels and enough detail to produce a long answer."
)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (rank - lower)


def extract_message_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    return ""


def completion_tokens(payload: dict[str, Any]) -> int | None:
    usage = payload.get("usage")
    if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
        return usage["completion_tokens"]
    return None


def post_chat_completion(
    base_url: str,
    model: str,
    prompt: str,
    timeout: float,
    max_tokens: int,
    extra_body: dict[str, Any] | None = None,
) -> tuple[int | None, dict[str, Any] | None, str | None, float]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if extra_body:
        body.update(extra_body)
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return response.status, payload, None, time.perf_counter() - start
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        return exc.code, None, text, time.perf_counter() - start
    except Exception as exc:  # noqa: BLE001 - benchmark artifact should record failures.
        return None, None, repr(exc), time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument(
        "--extra-body-json",
        default="",
        help="JSON object merged into each OpenAI-compatible request body.",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    extra_body: dict[str, Any] | None = None
    if args.extra_body_json:
        parsed_extra = json.loads(args.extra_body_json)
        if not isinstance(parsed_extra, dict):
            raise SystemExit("--extra-body-json must decode to a JSON object")
        extra_body = parsed_extra

    results: list[dict[str, Any]] = []
    for idx in range(args.repeats):
        status, payload, error, latency = post_chat_completion(
            args.base_url,
            args.model,
            args.prompt,
            args.timeout,
            args.max_tokens,
            extra_body,
        )
        text = extract_message_text(payload or {})
        tokens = completion_tokens(payload or {})
        results.append(
            {
                "index": idx,
                "status": status,
                "http_ok": status == 200,
                "latency_s": round(latency, 4),
                "completion_tokens": tokens,
                "tokens_per_s": round(tokens / latency, 4) if tokens and latency > 0 else None,
                "output_chars": len(text),
                "output_preview": text[:500],
                "error": error,
            }
        )

    ok_rows = [row for row in results if row["http_ok"]]
    latencies = [row["latency_s"] for row in ok_rows]
    token_rates = [row["tokens_per_s"] for row in ok_rows if row["tokens_per_s"] is not None]
    token_counts = [row["completion_tokens"] for row in ok_rows if row["completion_tokens"] is not None]
    output = {
        "label": args.label,
        "model": args.model,
        "created_at_unix": int(time.time()),
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "extra_body": extra_body,
        "summary": {
            "n": len(results),
            "http_ok": len(ok_rows),
            "latency_p50_s": round(percentile(latencies, 0.50), 4) if latencies else None,
            "latency_p90_s": round(percentile(latencies, 0.90), 4) if latencies else None,
            "tokens_per_s_p50": round(percentile(token_rates, 0.50), 4) if token_rates else None,
            "tokens_per_s_p90": round(percentile(token_rates, 0.90), 4) if token_rates else None,
            "completion_tokens_mean": round(statistics.fmean(token_counts), 2)
            if token_counts
            else None,
        },
        "results": results,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False))
    return 0 if len(ok_rows) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
