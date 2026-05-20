#!/usr/bin/env python3
"""Summarize fb-reader Tier B replay and stance-v2 A/B artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


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


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_replay(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    rows = payload.get("results") or []
    ok_rows = [row for row in rows if row.get("latency_ms") is not None]
    image_rows = [row for row in ok_rows if (row.get("image_count") or 0) > 0]
    text_rows = [row for row in ok_rows if (row.get("image_count") or 0) == 0]

    def latency_summary(scoped: list[dict[str, Any]]) -> dict[str, float | None]:
        latencies = [float(row["latency_ms"]) / 1000 for row in scoped]
        return {
            "p50_s": round(percentile(latencies, 0.50), 4) if latencies else None,
            "p90_s": round(percentile(latencies, 0.90), 4) if latencies else None,
        }

    token_rates = [
        float(row["completion_tokens"]) / (float(row["latency_ms"]) / 1000)
        for row in ok_rows
        if row.get("completion_tokens") and row.get("latency_ms")
    ]
    return {
        "count": len(rows),
        "http_ok": len(ok_rows),
        "parse_json_ok": sum(1 for row in rows if row.get("parse_json_ok")),
        "schema_ok": sum(1 for row in rows if row.get("schema_ok")),
        "timeouts": sum(1 for row in rows if row.get("timeout")),
        "errors": sum(1 for row in rows if row.get("error")),
        "all_latency": latency_summary(ok_rows),
        "image_latency": latency_summary(image_rows),
        "text_latency": latency_summary(text_rows),
        "tokens_per_s_p50": round(percentile(token_rates, 0.50), 4) if token_rates else None,
    }


def summarize_stance(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    summary = payload.get("summary") or {}
    return summary.get("all") or summary


def model_summary(replay: Path | None, stance: Path | None) -> dict[str, Any]:
    return {
        "replay": summarize_replay(load_json(replay)),
        "stance_v2": summarize_stance(load_json(stance)),
        "artifacts": {
            "replay": str(replay) if replay else None,
            "stance_v2": str(stance) if stance else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-replay", type=Path)
    parser.add_argument("--gemma-replay", type=Path)
    parser.add_argument("--qwen-stance", type=Path)
    parser.add_argument("--gemma-stance", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output = {
        "schema_version": 1,
        "qwen_dflash": model_summary(args.qwen_replay, args.qwen_stance),
        "gemma4_fp8_mtp": model_summary(args.gemma_replay, args.gemma_stance),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
