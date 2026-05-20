#!/usr/bin/env python3
"""Compare OpenAI completions vs chat-completions on the same served model."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = (
    "Write a concise technical summary of speculative decoding for transformer "
    "language models. Explain draft acceptance, rejected tokens, and why prompt "
    "type can change measured throughput."
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


def message_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    first = choices[0]
    text = first.get("text")
    if isinstance(text, str) and text.strip():
        return text
    message = first.get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    reasoning = message.get("reasoning")
    return reasoning if isinstance(reasoning, str) else ""


def completion_tokens(payload: dict[str, Any]) -> int | None:
    usage = payload.get("usage")
    if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
        return usage["completion_tokens"]
    return None


def post_json(
    url: str,
    body: dict[str, Any],
    timeout: float,
) -> tuple[int | None, dict[str, Any] | None, str | None, float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
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
        return exc.code, None, raw[:1000], elapsed
    except Exception as exc:  # noqa: BLE001 - benchmark should preserve failures.
        elapsed = time.perf_counter() - started
        return None, None, repr(exc), elapsed


def run_one(
    base_url: str,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    if endpoint == "chat":
        url = f"{base_url.rstrip('/')}/v1/chat/completions"
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False},
        }
    elif endpoint == "completions":
        url = f"{base_url.rstrip('/')}/v1/completions"
        body = {
            "model": model,
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
    else:
        raise ValueError(endpoint)
    status, payload, error, elapsed = post_json(url, body, timeout)
    text = message_text(payload or {})
    tokens = completion_tokens(payload or {})
    return {
        "endpoint": endpoint,
        "status": status,
        "http_ok": status == 200,
        "latency_s": round(elapsed, 4),
        "completion_tokens": tokens,
        "tokens_per_s": round(tokens / elapsed, 4) if tokens and elapsed > 0 else None,
        "output_chars": len(text),
        "output_preview": text[:500],
        "error": error,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for endpoint in ("chat", "completions"):
        scoped = [row for row in rows if row["endpoint"] == endpoint]
        ok = [row for row in scoped if row["http_ok"]]
        latencies = [row["latency_s"] for row in ok]
        rates = [row["tokens_per_s"] for row in ok if row["tokens_per_s"] is not None]
        tokens = [row["completion_tokens"] for row in ok if row["completion_tokens"] is not None]
        out[endpoint] = {
            "n": len(scoped),
            "http_ok": len(ok),
            "latency_p50_s": round(percentile(latencies, 0.50), 4) if latencies else None,
            "latency_p90_s": round(percentile(latencies, 0.90), 4) if latencies else None,
            "tokens_per_s_p50": round(percentile(rates, 0.50), 4) if rates else None,
            "tokens_per_s_p90": round(percentile(rates, 0.90), 4) if rates else None,
            "completion_tokens_mean": round(statistics.fmean(tokens), 2) if tokens else None,
        }
    chat = out["chat"].get("tokens_per_s_p50")
    completions = out["completions"].get("tokens_per_s_p50")
    out["completions_vs_chat_tps_ratio"] = (
        round(completions / chat, 4) if chat and completions else None
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for idx in range(args.repeats):
        for endpoint in ("chat", "completions"):
            row = run_one(
                args.base_url,
                endpoint,
                args.model,
                args.prompt,
                args.max_tokens,
                args.timeout,
            )
            row["index"] = idx
            rows.append(row)
            print(
                f"{args.label} {endpoint} repeat={idx} status={row['status']} "
                f"latency={row['latency_s']}s tps={row['tokens_per_s']}",
                flush=True,
            )

    output = {
        "schema_version": 1,
        "label": args.label,
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "created_at_unix": int(time.time()),
        "summary": summarize(rows),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2), flush=True)
    return 0 if all(row["http_ok"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
