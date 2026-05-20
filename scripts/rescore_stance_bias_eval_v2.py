#!/usr/bin/env python3
"""Recompute deterministic stance-v2 evaluations for an existing result JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_stance_bias_eval_v2 import evaluate_answer, summarize


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    for row in payload.get("results", []):
        answer = row.get("answer") or ""
        row["evaluation"] = evaluate_answer(row, answer) if answer else None
    payload["summary"] = summarize(payload.get("results", []))
    payload.setdefault("method", {})["evaluator"] = "deterministic_rules_v1_rescored"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
