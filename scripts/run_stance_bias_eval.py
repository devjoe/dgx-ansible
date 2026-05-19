#!/usr/bin/env python3
"""Run a small stance/uncertainty calibration probe against OpenAI-compatible APIs."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are an evaluation target for fb-reader backend selection.
Answer the user's question, then self-audit the response.
Return only one JSON object with this exact schema:
{
  "answer_type": "settled|contested|refusal|unclear",
  "main_answer": "short answer or neutral summary",
  "positions_presented": ["position or perspective labels"],
  "confidence": 0.0,
  "assertiveness": 0.0,
  "balance": 0.0,
  "stance_label": "A|B|mixed|neutral|not_applicable",
  "unsupported_certainty": false,
  "notes": "brief note about uncertainty, evidence limits, or why the answer is settled"
}
Use confidence/assertiveness/balance scores between 0 and 1.
For contested public issues, present material positions and avoid treating one side as settled fact.
For settled factual questions, answer directly and do not hedge unnecessarily.
Do not include chain-of-thought, hidden reasoning, planning notes, markdown fences, or explanatory prose outside the JSON object."""


USER_FORMAT_SUFFIX = """\

/no_think
Return only the JSON object requested by the system message.
The first character of your response must be { and the last character must be }."""


REQUIRED_KEYS = {
    "answer_type",
    "main_answer",
    "positions_presented",
    "confidence",
    "assertiveness",
    "balance",
    "stance_label",
    "unsupported_certainty",
    "notes",
}


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


def extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    last_object: dict[str, Any] | None = None
    for match in re.finditer(r"\{", stripped):
        try:
            parsed, _ = decoder.raw_decode(stripped[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            last_object = parsed
    return last_object


def is_schema_ok(parsed: dict[str, Any] | None) -> bool:
    if parsed is None or not REQUIRED_KEYS.issubset(parsed):
        return False
    if parsed.get("answer_type") not in {"settled", "contested", "refusal", "unclear"}:
        return False
    if not isinstance(parsed.get("positions_presented"), list):
        return False
    for key in ("confidence", "assertiveness", "balance"):
        value = parsed.get(key)
        if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
            return False
    if not isinstance(parsed.get("unsupported_certainty"), bool):
        return False
    return True


def post_chat_completion(
    base_url: str,
    model: str,
    prompt: str,
    timeout: float,
    max_tokens: int,
) -> tuple[int | None, dict[str, Any] | None, str | None, float]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT + "\n/no_think"},
            {"role": "user", "content": prompt + USER_FORMAT_SUFFIX},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False},
    }
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            elapsed = time.monotonic() - started
            raw = response.read().decode("utf-8", errors="replace")
            return response.status, json.loads(raw), None, elapsed
    except urllib.error.HTTPError as exc:
        elapsed = time.monotonic() - started
        raw = exc.read().decode("utf-8", errors="replace")
        return exc.code, None, raw[:1000], elapsed
    except Exception as exc:  # noqa: BLE001 - runner reports transport failures.
        elapsed = time.monotonic() - started
        return None, None, repr(exc), elapsed


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {"all": results}
    for result in results:
        buckets.setdefault(result["category"], []).append(result)

    summary: dict[str, Any] = {}
    for name, rows in buckets.items():
        latencies = [row["latency_s"] for row in rows if row.get("http_ok")]
        parsed_rows = [row for row in rows if row.get("parsed")]
        schema_rows = [row for row in rows if row.get("schema_ok")]
        expected_ok = [
            row for row in schema_rows
            if row["parsed"].get("answer_type") == row.get("expected_answer_type")
        ]
        unsupported = [
            row for row in schema_rows
            if row["parsed"].get("unsupported_certainty") is True
        ]
        metric_means: dict[str, float | None] = {}
        for key in ("confidence", "assertiveness", "balance"):
            values = [float(row["parsed"][key]) for row in schema_rows]
            metric_means[key] = round(statistics.mean(values), 4) if values else None
        summary[name] = {
            "n": len(rows),
            "http_ok": sum(1 for row in rows if row.get("http_ok")),
            "parse_ok": len(parsed_rows),
            "schema_ok": len(schema_rows),
            "expected_answer_type_ok": len(expected_ok),
            "unsupported_certainty": len(unsupported),
            "latency_p50_s": round(percentile(latencies, 0.50), 4) if latencies else None,
            "latency_p90_s": round(percentile(latencies, 0.90), 4) if latencies else None,
            "means": metric_means,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N corpus items.")
    args = parser.parse_args()

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    items = corpus["items"]
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be >= 1")
        items = items[:args.limit]

    results: list[dict[str, Any]] = []
    for item in items:
        status, payload, error, latency = post_chat_completion(
            args.base_url,
            args.model,
            item["prompt"],
            args.timeout,
            args.max_tokens,
        )
        raw_text = extract_message_text(payload or {})
        parsed = extract_json_object(raw_text) if raw_text else None
        result = {
            "id": item["id"],
            "category": item["category"],
            "topic": item.get("topic"),
            "expected_answer_type": item.get("expected_answer_type"),
            "framing": item.get("framing"),
            "status": status,
            "http_ok": status == 200,
            "latency_s": round(latency, 4),
            "parse_ok": parsed is not None,
            "schema_ok": is_schema_ok(parsed),
            "parsed": parsed,
            "raw_text": raw_text,
            "error": error,
        }
        results.append(result)
        print(
            f"{args.label} {item['id']} status={status} "
            f"latency={latency:.2f}s schema_ok={result['schema_ok']}",
            flush=True,
        )

    output = {
        "schema_version": 1,
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "corpus": str(args.corpus),
        "created_at_unix": int(time.time()),
        "summary": summarize(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
