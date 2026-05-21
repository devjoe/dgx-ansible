#!/usr/bin/env python3
"""Build the first Qwen dir-steering extraction corpus from manual DS4 review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TARGET_SYSTEM_PROMPT = """You are a backend candidate for fb-reader.
Answer the user's question for a careful social-media reader.
Use neutral wording, present material uncertainty, and do not overstate disputed
public issues as settled facts. Do not self-audit. Return plain text only.
/no_think"""

USER_SUFFIX = """\

/no_think
Return only the reader-facing answer. Do not return JSON."""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def row_to_item(row: dict[str, Any], group: str) -> dict[str, Any]:
    review = row.get("manual_review") or {}
    prompt = row.get("topic") or row.get("ds4_question") or ""
    return {
        "id": row["id"],
        "group": group,
        "topic": prompt,
        "messages": [
            {"role": "system", "content": TARGET_SYSTEM_PROMPT},
            {"role": "user", "content": prompt + USER_SUFFIX},
        ],
        "expected_topic_contestedness": row.get("expected_topic_contestedness"),
        "manual_verdict": review.get("verdict"),
        "manual_topic_handling": review.get("topic_handling"),
        "manual_product_risk": review.get("product_risk"),
        "manual_rationale": review.get("rationale"),
        "manual_rationale_zh": review.get("rationale_zh"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--positive-limit",
        type=int,
        default=0,
        help="Limit contested-positive items. 0 keeps all available positives.",
    )
    parser.add_argument(
        "--negative-limit",
        type=int,
        default=0,
        help="Limit extraction-negative items. 0 keeps all available negatives.",
    )
    args = parser.parse_args()

    payload = load_json(args.manual_review)
    rows = payload.get("results", [])
    negatives = [
        row
        for row in rows
        if row.get("manual_review", {}).get("extraction_negative") is True
    ]
    positives = [
        row
        for row in rows
        if row.get("category") == "ds4_contested"
        and row.get("manual_review", {}).get("verdict") == "pass"
    ]
    if args.negative_limit > 0:
        negatives = negatives[: args.negative_limit]
    if args.positive_limit > 0:
        positives = positives[: args.positive_limit]

    items = [
        *(row_to_item(row, "contested_positive") for row in positives),
        *(row_to_item(row, "over_contested_settled_negative") for row in negatives),
    ]
    output = {
        "schema_version": 1,
        "source": str(args.manual_review),
        "direction_intent": "Reduce over-contested framing on settled questions while preserving contested-question caution.",
        "positive_group": "contested_positive",
        "negative_group": "over_contested_settled_negative",
        "method_note": (
            "First-pass mean-difference vectors should be interpreted as diagnostics. "
            "The sign is negative_mean - positive_mean; steering direction may need sign/scale sweeps."
        ),
        "counts": {
            "contested_positive": len(positives),
            "over_contested_settled_negative": len(negatives),
            "items": len(items),
        },
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
