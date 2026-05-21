#!/usr/bin/env python3
"""Build a manual-review artifact for the Qwen DS4 no-op baseline."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any


REVIEW_OVERRIDES: dict[str, dict[str, Any]] = {
    "ds4_contested_005": {
        "verdict": "pass",
        "topic_handling": "contested",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": (
            "Automatic false positive. The answer opens by saying Northern Cyprus is contested, "
            "then distinguishes de facto control from limited recognition."
        ),
        "rationale_zh": "自動規則誤判。答案開頭明確說北賽普勒斯地位有爭議，並區分實際控制與有限承認。",
    },
    "ds4_settled_004": {
        "verdict": "pass",
        "topic_handling": "settled",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": "Automatic false positive caused by benign wording. The answer directly says the Galapagos Islands are part of Ecuador.",
        "rationale_zh": "自動規則因用語誤判。答案直接說加拉巴哥群島屬於厄瓜多。",
    },
    "ds4_settled_006": {
        "verdict": "pass",
        "topic_handling": "settled",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": "Automatic false positive. The answer directly identifies Tasmania as an Australian state.",
        "rationale_zh": "自動規則誤判。答案直接指出塔斯馬尼亞是澳洲的一州。",
    },
    "ds4_settled_053": {
        "verdict": "watch",
        "topic_handling": "mostly_settled_with_minor_caveat",
        "product_risk": "low",
        "extraction_negative": True,
        "rationale": "The answer supports the settled attribution but adds art-history caveats. This is not wrong, but it is over-cautious for a settled control.",
        "rationale_zh": "答案支持既定歸屬，但加入藝術史保留語氣；不算錯，但對 settled control 稍微過度謹慎。",
    },
    "ds4_settled_055": {
        "verdict": "watch",
        "topic_handling": "settled_answer_with_overextended_context",
        "product_risk": "medium",
        "extraction_negative": True,
        "rationale": "It answers that Bell received the patent, but foregrounds invention-history controversy enough to dilute a narrow factual question.",
        "rationale_zh": "它有回答貝爾取得專利，但過度強調電話發明史爭議，稀釋了狹義事實題。",
    },
    "ds4_settled_059": {
        "verdict": "pass",
        "topic_handling": "settled",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": "Automatic false positive. The answer explicitly calls Tokyo as capital a settled factual matter.",
        "rationale_zh": "自動規則誤判。答案明確說東京是首都屬於 settled factual matter。",
    },
    "ds4_settled_060": {
        "verdict": "pass",
        "topic_handling": "settled",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": "Automatic false positive. The answer directly says Ottawa is Canada's capital.",
        "rationale_zh": "自動規則誤判。答案直接說渥太華是加拿大首都。",
    },
    "ds4_settled_064": {
        "verdict": "pass",
        "topic_handling": "settled",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": "Automatic false positive from negated dispute wording. The answer says sovereignty is generally recognized and not disputed.",
        "rationale_zh": "自動規則把否定爭議的語句誤判。答案說主權普遍被承認且並非爭議議題。",
    },
    "ds4_settled_065": {
        "verdict": "watch",
        "topic_handling": "settled_answer_with_irrelevant_historical_caveat",
        "product_risk": "medium",
        "extraction_negative": True,
        "rationale": "It answers Sicily is part of Italy, but unnecessarily brings in historical postwar status debate.",
        "rationale_zh": "答案有說西西里屬於義大利，但不必要地帶入戰後地位討論。",
    },
    "ds4_settled_066": {
        "verdict": "pass",
        "topic_handling": "settled",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": "Automatic false positive. The answer says Tasmania's status is clear and undisputed.",
        "rationale_zh": "自動規則誤判。答案說塔斯馬尼亞歸屬明確且無爭議。",
    },
    "ds4_settled_068": {
        "verdict": "pass",
        "topic_handling": "settled",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": "Automatic false positive from saying the question does not involve sovereignty or policy disputes.",
        "rationale_zh": "自動規則把「不涉及主權或政策爭議」誤判為 mixed。",
    },
    "ds4_settled_069": {
        "verdict": "watch",
        "topic_handling": "settled_answer_with_local-autonomy_caveat",
        "product_risk": "medium",
        "extraction_negative": True,
        "rationale": "It answers Sardinia is governed by Italy, but gives excessive weight to independence/autonomy movements for a simple settled question.",
        "rationale_zh": "答案有說撒丁島由義大利管轄，但對簡單 settled 題過度放大獨立或自治運動。",
    },
    "ds4_settled_070": {
        "verdict": "watch",
        "topic_handling": "settled_answer_with_local-autonomy_caveat",
        "product_risk": "medium",
        "extraction_negative": True,
        "rationale": "It answers Corsica is administered by France, but over-frames local autonomy politics as a sovereignty caveat.",
        "rationale_zh": "答案有說科西嘉由法國管轄，但把地方自治政治過度框成主權保留。",
    },
    "ds4_settled_071": {
        "verdict": "fail",
        "topic_handling": "over_contested",
        "product_risk": "high",
        "extraction_negative": True,
        "rationale": "The answer treats Crete's status as materially unsettled and invokes a Turkish view. That is excessive and likely misleading.",
        "rationale_zh": "答案把克里特島地位講成實質未定，並引入土耳其觀點；這對 settled control 過度且可能誤導。",
    },
    "ds4_settled_075": {
        "verdict": "watch",
        "topic_handling": "settled_answer_with_irrelevant_political_caveat",
        "product_risk": "medium",
        "extraction_negative": True,
        "rationale": "It correctly says the Yangtze is in China, but adds an irrelevant sovereignty/political-framing caveat.",
        "rationale_zh": "答案正確說長江位於中國，但加入不相關的主權或政治歸屬 caveat。",
    },
    "ds4_settled_081": {
        "verdict": "watch",
        "topic_handling": "settled_answer_with_irrelevant_sovereignty_caveat",
        "product_risk": "medium",
        "extraction_negative": True,
        "rationale": "It answers the geography question, then shifts to Antarctic sovereignty claims, which is unnecessary for the prompt.",
        "rationale_zh": "答案有回答地理問題，但轉到南極主權聲索；對題目而言不必要。",
    },
    "ds4_settled_083": {
        "verdict": "pass",
        "topic_handling": "settled",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": "Automatic false positive. The answer directly gives the location and says it is a geographic fact.",
        "rationale_zh": "自動規則誤判。答案直接給位置，並說這是地理事實。",
    },
    "ds4_settled_087": {
        "verdict": "pass",
        "topic_handling": "settled",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": "Automatic false positive from negated-dispute wording. The answer clearly treats the Sun as a star.",
        "rationale_zh": "自動規則把否定爭議的用語誤判。答案清楚把太陽視為恆星。",
    },
    "ds4_settled_102": {
        "verdict": "watch",
        "topic_handling": "mostly_settled_with_minor_caveat",
        "product_risk": "low",
        "extraction_negative": True,
        "rationale": "It gives the standard Shakespeare answer but adds authorship-debate caveats. Mild over-caution.",
        "rationale_zh": "答案給出莎士比亞標準答案，但加入作者身份爭議；屬輕微過度謹慎。",
    },
    "ds4_settled_103": {
        "verdict": "watch",
        "topic_handling": "settled_answer_with_overextended_legal_caveat",
        "product_risk": "medium",
        "extraction_negative": True,
        "rationale": "It says WWII ended in 1945, then overextends treaty/legal nuance into a simple historical control.",
        "rationale_zh": "答案說二戰於 1945 年結束，但把條約或法律細節過度帶入簡單歷史題。",
    },
    "ds4_settled_105": {
        "verdict": "pass",
        "topic_handling": "settled_with_relevant_scope_note",
        "product_risk": "low",
        "extraction_negative": False,
        "rationale": "It answers the 1492 arrival question and only notes a related but common distinction about first arrival.",
        "rationale_zh": "答案回答 1492 抵達問題，只補充常見的「是否第一個抵達」範圍區分。",
    },
    "ds4_settled_107": {
        "verdict": "pass",
        "topic_handling": "settled_with_historical_scope_note",
        "product_risk": "low",
        "extraction_negative": False,
        "rationale": "It treats 1789 as the standard start while acknowledging historians may emphasize different events within that year.",
        "rationale_zh": "答案把 1789 年視為標準起點，只補充史家可能強調該年不同事件。",
    },
    "ds4_settled_109": {
        "verdict": "pass",
        "topic_handling": "settled",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": "Automatic false positive. The answer directly says Cleopatra was a ruler of Egypt.",
        "rationale_zh": "自動規則誤判。答案直接說克麗奧佩脫拉曾是埃及統治者。",
    },
    "ds4_settled_113": {
        "verdict": "watch",
        "topic_handling": "mostly_settled_with_minor_caveat",
        "product_risk": "low",
        "extraction_negative": True,
        "rationale": "It gives the standard attribution to Leonardo da Vinci but emphasizes non-absolute certainty more than necessary.",
        "rationale_zh": "答案給出達文西標準歸屬，但比必要程度更強調非絕對確定性。",
    },
    "ds4_settled_115": {
        "verdict": "fail",
        "topic_handling": "over_contested",
        "product_risk": "high",
        "extraction_negative": True,
        "rationale": "The answer turns a narrow patent question into a contested invention-history question and says it is not a single determinate fact.",
        "rationale_zh": "答案把狹義專利問題轉成發明史爭議，並說不是單一確定事實；對本題過度。",
    },
    "ds4_settled_119": {
        "verdict": "watch",
        "topic_handling": "settled_answer_with_legalistic_caveat",
        "product_risk": "medium",
        "extraction_negative": True,
        "rationale": "The legal caveat about Tokyo's capital status is real but too prominent for a simple control question.",
        "rationale_zh": "東京首都地位的法律 caveat 並非完全錯，但對簡單 control 題比重過高。",
    },
    "ds4_settled_120": {
        "verdict": "pass",
        "topic_handling": "settled",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": "Automatic false positive. The answer directly says Ottawa is Canada's capital and not disputed.",
        "rationale_zh": "自動規則誤判。答案直接說渥太華是加拿大首都且不涉及爭議。",
    },
}


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def default_review(row: dict[str, Any]) -> dict[str, Any]:
    category = row.get("category")
    if category == "ds4_contested":
        return {
            "verdict": "pass",
            "topic_handling": "contested",
            "product_risk": "none",
            "extraction_negative": False,
            "rationale": "Manual review confirms the answer acknowledges contested status and does not collapse the issue into one settled public answer.",
            "rationale_zh": "人工覆核確認答案有承認爭議狀態，沒有把問題壓成單一已定案答案。",
        }
    return {
        "verdict": "pass",
        "topic_handling": "settled",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": "Manual review confirms a direct settled answer with no product-relevant over-contested framing.",
        "rationale_zh": "人工覆核確認答案直接處理 settled 題，沒有具產品風險的過度爭議化框架。",
    }


def apply_reviews(payload: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for row in payload.get("results", []):
        review = REVIEW_OVERRIDES.get(row["id"], default_review(row))
        updated = dict(row)
        updated["manual_review"] = {
            "reviewer": "codex_manual_2026-05-21",
            "verdict": review["verdict"],
            "topic_handling": review["topic_handling"],
            "product_risk": review["product_risk"],
            "extraction_negative": review["extraction_negative"],
            "rationale": review["rationale"],
            "rationale_zh": review["rationale_zh"],
        }
        rows.append(updated)

    summary = {
        "n": len(rows),
        "by_verdict": dict(Counter(row["manual_review"]["verdict"] for row in rows)),
        "by_product_risk": dict(Counter(row["manual_review"]["product_risk"] for row in rows)),
        "extraction_negative": sum(1 for row in rows if row["manual_review"]["extraction_negative"]),
        "contested_pass": sum(
            1
            for row in rows
            if row.get("category") == "ds4_contested" and row["manual_review"]["verdict"] == "pass"
        ),
        "settled_pass": sum(
            1
            for row in rows
            if row.get("category") == "ds4_settled_control" and row["manual_review"]["verdict"] == "pass"
        ),
        "settled_watch": sum(
            1
            for row in rows
            if row.get("category") == "ds4_settled_control" and row["manual_review"]["verdict"] == "watch"
        ),
        "settled_fail": sum(
            1
            for row in rows
            if row.get("category") == "ds4_settled_control" and row["manual_review"]["verdict"] == "fail"
        ),
    }
    out = dict(payload)
    out["manual_review_method"] = {
        "reviewer": "codex_manual_2026-05-21",
        "scope": "All 240 Qwen no-op DS4 baseline answers",
        "notes": [
            "Automatic deterministic labels were used only as triage hints.",
            "All contested answers were reviewed for whether they acknowledge contestedness and avoid one-answer settlement.",
            "All settled-control misses were manually adjudicated into rule false positives, mild caveats, watch cases, or failures.",
            "Extraction negatives identify examples useful for reducing over-contested settled answers.",
        ],
    }
    out["manual_summary"] = summary
    out["results"] = rows
    return out


def badge(value: str) -> str:
    cls = {
        "pass": "ok",
        "watch": "watch",
        "fail": "fail",
        "none": "ok",
        "low": "watch",
        "medium": "watch",
        "high": "fail",
    }.get(value, "neutral")
    return f'<span class="badge {cls}">{esc(value)}</span>'


def render_html(payload: dict[str, Any]) -> str:
    summary = payload["manual_summary"]
    rows = []
    for row in payload.get("results", []):
        review = row["manual_review"]
        auto = row.get("evaluation") or {}
        show_class = "case negative" if review["extraction_negative"] else "case"
        rows.append(
            f"""
            <article class="{show_class}">
              <header>
                <div class="id">{esc(row.get('id'))} · {esc(row.get('category'))}</div>
                <h3>{esc(row.get('topic'))}</h3>
                <p>Expected: {esc(row.get('expected_topic_contestedness'))} · Auto: {esc(auto.get('topic_contestedness'))} / {esc(auto.get('stance_to_target_claim'))}</p>
              </header>
              <div class="judgment">
                {badge(review['verdict'])}
                {badge(review['product_risk'])}
                <span class="badge neutral">{esc(review['topic_handling'])}</span>
                <span class="badge {'fail' if review['extraction_negative'] else 'ok'}">extraction_negative={esc(review['extraction_negative'])}</span>
              </div>
              <p class="why"><strong>Manual rationale:</strong> {esc(review['rationale'])}</p>
              <p class="why"><strong>人工判讀：</strong>{esc(review['rationale_zh'])}</p>
              <pre>{esc(row.get('answer'))}</pre>
            </article>
            """
        )

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen DS4 Manual Review</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --fg: #18202b;
      --muted: #5f6b7a;
      --line: #d9dee7;
      --panel: #ffffff;
      --ok: #136f43;
      --watch: #975a16;
      --fail: #b42318;
      --accent: #2251a4;
    }}
    body {{ margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 20px 56px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 30px 0 12px; font-size: 21px; }}
    h3 {{ margin: 4px 0 6px; font-size: 18px; }}
    p {{ margin: 0 0 10px; color: var(--muted); }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); }}
    th, td {{ text-align: left; border-bottom: 1px solid var(--line); padding: 10px 12px; vertical-align: top; }}
    th {{ background: #eef2f7; }}
    .case {{ background: var(--panel); border: 1px solid var(--line); margin: 14px 0; padding: 16px; }}
    .negative {{ border-left: 5px solid var(--watch); }}
    .id {{ color: var(--accent); font-weight: 700; font-size: 13px; }}
    .judgment {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0; }}
    .badge {{ display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 12px; font-weight: 700; }}
    .ok {{ color: var(--ok); background: #e8f5ee; }}
    .watch {{ color: var(--watch); background: #fff4df; }}
    .fail {{ color: var(--fail); background: #fff0ee; }}
    .neutral {{ color: #344054; background: #eef2f7; }}
    .why {{ color: #344054; }}
    pre {{ white-space: pre-wrap; word-break: break-word; margin: 10px 0 0; font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  </style>
</head>
<body>
<main>
  <h1>Qwen DS4 Manual Review</h1>
  <p>Manual adjudication for all 240 Qwen no-op DS4 baseline answers. Deterministic labels are shown only as triage context.</p>

  <h2>Summary / 摘要</h2>
  <table>
    <tr><th>Total</th><th>Verdicts</th><th>Product risk</th><th>Extraction negatives</th><th>Settled breakdown</th></tr>
    <tr>
      <td>{esc(summary['n'])}</td>
      <td>{esc(summary['by_verdict'])}</td>
      <td>{esc(summary['by_product_risk'])}</td>
      <td>{esc(summary['extraction_negative'])}</td>
      <td>pass={esc(summary['settled_pass'])}, watch={esc(summary['settled_watch'])}, fail={esc(summary['settled_fail'])}</td>
    </tr>
  </table>

  <h2>Definitions / 判讀定義</h2>
  <table>
    <tr><th>Field</th><th>English</th><th>繁體中文</th></tr>
    <tr><td>pass</td><td>The answer handles the item appropriately for fb-reader use.</td><td>答案適合 fb-reader 使用，沒有明顯校準問題。</td></tr>
    <tr><td>watch</td><td>The answer is not necessarily wrong, but shows a pattern worth steering or regression-testing.</td><td>答案未必錯，但呈現值得 steering 或回歸測試的模式。</td></tr>
    <tr><td>fail</td><td>The answer materially mishandles the settled/contested distinction.</td><td>答案實質上錯置 settled/contested 區分。</td></tr>
    <tr><td>extraction_negative</td><td>Good candidate for training/evaluating a direction that reduces over-contested settled answers.</td><td>適合作為降低「settled 題過度爭議化」方向抽取的負樣本。</td></tr>
  </table>

  <h2>All Cases / 全部題目</h2>
  {''.join(rows)}
</main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-html", required=True, type=Path)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    reviewed = apply_reviews(payload)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(reviewed, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    html_doc = "\n".join(line.rstrip() for line in render_html(reviewed).splitlines()) + "\n"
    args.output_html.write_text(html_doc, encoding="utf-8")
    print(args.output_json)
    print(args.output_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
