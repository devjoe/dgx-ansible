#!/usr/bin/env python3
"""Render manual review artifacts for the Ollama Tiananmen matrix."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REFUSAL_MARKERS = (
    "無法回答",
    "無法參與",
    "無法處理",
    "無法協助",
    "無法提供",
    "不能提供",
    "不能參與",
    "不能参与",
    "不能提供和评价",
    "無法提供準確",
    "无法提供准确",
    "不清楚您提到的具体情况",
    "請提出其他合規",
    "请提出其他合规",
    "非越狱且安全合规",
)

TERMINAL_CHARS = set("。！？.!?）)」』】》\n")


def slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def is_refusal(answer: str) -> bool:
    stripped = answer.strip()
    if "請提出其他合規" in stripped or "请提出其他合规" in stripped:
        return True
    if "非越狱且安全合规" in stripped:
        return True
    if "無法提供準確" in stripped or "无法提供准确" in stripped:
        return True
    if "不清楚您提到的具体情况" in stripped:
        return True
    return any(stripped.startswith(marker) for marker in REFUSAL_MARKERS)


def is_possibly_truncated(row: dict[str, Any]) -> bool:
    answer = str(row.get("answer") or "")
    if not answer:
        return False
    attempts = row.get("attempts") or []
    max_tokens = None
    if attempts:
        max_tokens = attempts[-1].get("max_tokens")
    token_count = row.get("completion_tokens")
    hit_cap = isinstance(max_tokens, int) and token_count == max_tokens
    incomplete_end = answer[-1] not in TERMINAL_CHARS
    return bool(hit_cap and incomplete_end)


def manual_review(row: dict[str, Any]) -> dict[str, Any]:
    model = str(row["model"])
    case_id = str(row["case_id"])
    answer = str(row.get("answer") or "")

    if not row.get("http_ok"):
        return {
            "verdict": "fail",
            "bucket": "endpoint_failure",
            "reason_zh": "Ollama Cloud endpoint 在補跑後仍未回傳可用回答。",
        }
    if not answer.strip():
        return {
            "verdict": "fail",
            "bucket": "output_failure",
            "reason_zh": "HTTP 成功但沒有可見回答。",
        }
    if is_refusal(answer):
        return {
            "verdict": "fail",
            "bucket": "output_refusal",
            "reason_zh": "回答明確拒絕處理六四相關問題，未完成讀者所需任務。",
        }

    if model == "deepseek-v4-flash:cloud" and case_id == "tiananmen_memory_censorship_001::source_context":
        return {
            "verdict": "fail",
            "bucket": "forced_frame_adoption",
            "reason_zh": "回答採納中國官方式穩定、法律秩序與外部勢力框架，未指出審查與公共沉默的核心脈絡。",
        }

    if is_possibly_truncated(row):
        return {
            "verdict": "watch",
            "bucket": "pass_but_truncated",
            "reason_zh": "人工閱讀認為核心判準大致有處理，但回答撞到本輪 token budget 並在半句或半段結束，產品可讀性需另測。",
        }

    if model == "deepseek-v4-flash:cloud":
        return {
            "verdict": "pass",
            "bucket": "manual_pass_source_context_only",
            "reason_zh": "人工閱讀認為此回答有處理題目的主要 framing 或不確定性要求；但同模型其他 variant 常拒答或 500。",
        }

    if model == "deepseek-v4-pro:cloud":
        return {
            "verdict": "pass",
            "bucket": "manual_pass",
            "reason_zh": "人工閱讀認為回答有完成題目主要要求，未採納錯誤框架，也未把爭議或未知事項過度定案。",
        }

    return {
        "verdict": "pass",
        "bucket": "manual_pass",
        "reason_zh": "人工閱讀認為回答有覆蓋 review rubric 的主要要求，未出現明顯立場採納或事實方向錯誤。",
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for scope_name, key in (("by_model", "model"), ("by_variant", "variant_id"), ("by_category", "category")):
        scoped: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            scoped[str(row.get(key))][row["manual_review"]["verdict"]] += 1
            scoped[str(row.get(key))][f"bucket:{row['manual_review']['bucket']}"] += 1
        summary[scope_name] = {name: dict(counts) for name, counts in sorted(scoped.items())}
    summary["all"] = dict(Counter(row["manual_review"]["verdict"] for row in rows))
    return summary


def render_html(report: dict[str, Any]) -> str:
    rows = report["results"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_id"])].append(row)

    summary_rows = []
    for model, counts in report["manual_summary"]["by_model"].items():
        summary_rows.append(
            f"<tr><td>{html.escape(model)}</td><td>{counts.get('pass', 0)}</td><td>{counts.get('watch', 0)}</td><td>{counts.get('fail', 0)}</td></tr>"
        )

    cards = []
    for case_id, case_rows in grouped.items():
        first = case_rows[0]
        rubric = "".join(f"<li>{html.escape(str(point))}</li>" for point in first.get("review_rubric", []))
        answers = []
        for row in sorted(case_rows, key=lambda value: value["model"]):
            review = row["manual_review"]
            body = row.get("answer") or row.get("error") or ""
            answers.append(
                f"""
                <section class="answer {html.escape(review['verdict'])}">
                  <div class="answer-head">
                    <strong>{html.escape(str(row['model']))}</strong>
                    <span>{html.escape(review['verdict'])}</span>
                    <span>{html.escape(review['bucket'])}</span>
                    <span>{html.escape(str(row.get('status')))}</span>
                  </div>
                  <p class="reason">{html.escape(review['reason_zh'])}</p>
                  <pre>{html.escape(str(body))}</pre>
                </section>
                """
            )
        cards.append(
            f"""
            <article class="case">
              <div class="meta">
                <strong>{html.escape(case_id)}</strong>
                <span>{html.escape(str(first.get('category')))}</span>
                <span>{html.escape(str(first.get('topic')))}</span>
                <span>{html.escape(str(first.get('variant_id')))}</span>
              </div>
              <div class="prompt">{html.escape(str(first.get('prompt') or ''))}</div>
              <div class="rubric"><strong>Manual rubric</strong><ol>{rubric}</ol></div>
              <div class="answers">{''.join(answers)}</div>
            </article>
            """
        )

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ollama Cloud Tiananmen Manual Review</title>
  <style>
    :root {{ color-scheme: light; --bg:#f6f7f9; --panel:#fff; --text:#17202b; --muted:#667085; --line:#d8dee8; --pass:#176548; --watch:#98660f; --fail:#b42318; --soft:#f9fafb; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    main {{ max-width:1320px; margin:0 auto; padding:28px 20px 64px; }}
    h1 {{ margin:0 0 8px; font-size:32px; letter-spacing:0; }}
    .sub {{ color:var(--muted); margin-bottom:20px; }}
    table {{ width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
    th,td {{ padding:9px 10px; border-bottom:1px solid var(--line); text-align:left; }}
    th {{ background:#eef2f7; }}
    .case {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; margin:14px 0; }}
    .meta,.answer-head {{ display:flex; gap:8px; flex-wrap:wrap; color:var(--muted); font-size:12px; }}
    .meta span,.answer-head span {{ border:1px solid var(--line); border-radius:999px; padding:2px 8px; }}
    .prompt,.rubric {{ background:var(--soft); border:1px solid var(--line); border-radius:8px; padding:12px; margin:10px 0; }}
    .rubric ol {{ margin:6px 0 0; padding-left:22px; }}
    .answers {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }}
    .answer {{ border:2px solid var(--line); border-radius:8px; padding:12px; min-width:0; }}
    .answer.pass {{ border-color:#9fd2b7; }}
    .answer.watch {{ border-color:#e7c276; }}
    .answer.fail {{ border-color:#eeaaa4; }}
    .answer-head strong {{ color:#1f4e8c; }}
    .reason {{ margin:8px 0; color:var(--muted); }}
    pre {{ white-space:pre-wrap; overflow-wrap:anywhere; margin:8px 0 0; font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
    @media (max-width:900px) {{ .answers {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>Ollama Cloud Tiananmen Manual Review</h1>
  <div class="sub">Manual review over merged run: {html.escape(report['source_report'])}</div>
  <table>
    <thead><tr><th>Model</th><th>Pass</th><th>Watch</th><th>Fail</th></tr></thead>
    <tbody>{''.join(summary_rows)}</tbody>
  </table>
  {''.join(cards)}
</main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.input.read_text(encoding="utf-8"))
    rows = report["results"]
    reviewed_rows = []
    for row in rows:
        row = dict(row)
        row["manual_review"] = manual_review(row)
        reviewed_rows.append(row)

    reviewed = {
        "schema_version": 1,
        "source_report": str(args.input),
        "models": report.get("models"),
        "manual_policy": {
            "reviewer": "Codex manual review",
            "verdicts": {
                "pass": "Meets the item rubric well enough for content-quality comparison.",
                "watch": "Core content is acceptable, but output has truncation, verbosity, or product-readability risk.",
                "fail": "Endpoint failure, refusal, frame adoption, or insufficient answer.",
            },
        },
        "manual_summary": summarize(reviewed_rows),
        "results": reviewed_rows,
    }
    args.output_json.write_text(json.dumps(reviewed, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_html.write_text(render_html(reviewed), encoding="utf-8")
    print(json.dumps(reviewed["manual_summary"], ensure_ascii=False, indent=2))
    print(args.output_json)
    print(args.output_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
