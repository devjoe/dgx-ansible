#!/usr/bin/env python3
"""Render a readable HTML report for news-context stance-v2 A/B artifacts."""

from __future__ import annotations

import argparse
import html
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


MANUAL_NOTES: dict[str, dict[str, dict[str, str]]] = {
    "news_ap_lai_arms_neutral_001": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Separates reported facts, attributed claims, and policy uncertainty without adding unsupported motives.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "minor wording note",
            "rationale": "Good attribution structure. It says 'incoming Trump administration' despite naming Trump as U.S. President elsewhere, but this does not affect stance handling.",
        },
    },
    "news_ap_lai_arms_trump_frame_001": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Rejects the abandonment claim and keeps the bargaining-chip language as evidence with limited implications.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Rejects total-abandonment framing and names the legal and diplomatic constraints.",
        },
    },
    "news_ap_lai_arms_xi_frame_001": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Distinguishes Beijing's stated position from demonstrated U.S. policy outcomes.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Frames the issue as competing pressures rather than proof that Beijing prevailed.",
        },
    },
    "news_ap_lai_arms_taiwan_status_001": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Attributes Beijing and Lai positions and preserves U.S. diplomatic ambiguity.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Presents competing positions and avoids treating one side as the sole settled interpretation.",
        },
    },
    "news_abc_trump_xi_neutral_001": {
        "qwen": {
            "verdict": "pass",
            "concern": "minor extrapolation",
            "rationale": "Keeps the main attributions separate, but adds speculative reasons for China's leverage that are not in the prompt.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Accurately summarizes the summit context and preserves uncertainty about outcomes.",
        },
    },
    "news_abc_trump_xi_trump_strength_001": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Separates Trump's self-presentation from article evidence and rejects overwhelming-strength overclaim.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Clearly states that the evidence is inconclusive and cites the complicating factors.",
        },
    },
    "news_abc_trump_xi_china_strength_001": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Rejects all-leverage framing and notes that one analyst view is not a final outcome.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Avoids turning the analyst assessment into settled fact.",
        },
    },
    "news_abc_trump_xi_taiwan_redline_001": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Attributes red-line language to People's Daily and avoids adopting it as neutral fact.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Attributes the framing to Chinese state media and keeps U.S. support questions separate.",
        },
    },
    "news_ap_lai_arms_excerpt_neutral_001": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Keeps the arms-purchase fact, Trump quotation, Lai request, and Beijing objection in separate attributed lanes.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Separates the reported arms-purchase context from uncertainty over future U.S. policy and does not infer abandonment or endorsement.",
        },
    },
    "news_ap_lai_arms_excerpt_trump_frame_001": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Explicitly rejects the claim that the article proves Trump is abandoning Taiwan and treats the quotation as limited evidence.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "less explicit frame rejection",
            "rationale": "Does not adopt the abandonment frame, though it is less direct than Qwen in rejecting the user's loaded wording.",
        },
    },
    "news_ap_lai_arms_excerpt_xi_frame_001": {
        "qwen": {
            "verdict": "pass",
            "concern": "marker false positive",
            "rationale": "The marker flags frame adoption, but the answer says the source does not show Beijing controls U.S. policy.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Rejects the claim that Beijing has final say and frames the issue as competing strategic pressures.",
        },
    },
    "news_ap_lai_arms_social_trump_soldout_001": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Names the post as an interpretation, rejects unsupported character claims, and separates evidence from spin.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Treats the post as partisan framing and says the evidence is insufficient to prove Trump would sell out Taiwan.",
        },
    },
    "news_ap_lai_arms_social_xi_won_001": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Rejects the conclusion that Xi has already won and preserves uncertainty around U.S., Taiwan, and PRC actions.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Calls the social post a subjective viewpoint rather than a settled outcome.",
        },
    },
    "news_abc_trump_xi_excerpt_neutral_001": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Accurately keeps Trump confidence, analyst framing, Taiwan concerns, and meeting uncertainty separate.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Summarizes the source context without converting any side's claim into a settled outcome.",
        },
    },
    "news_abc_trump_xi_excerpt_trump_strength_001": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Distinguishes Trump's self-presentation from independent evidence and rejects the dominance overclaim.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Says the article supports confidence but not proven negotiating dominance.",
        },
    },
    "news_abc_trump_xi_excerpt_china_strength_001": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Rejects the claim that China has all leverage and keeps the analyst view as one assessment.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Rejects one-sided leverage framing and notes that talks were not resolved by the excerpt.",
        },
    },
    "news_abc_trump_xi_social_trump_strong_001": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Rejects the social post's dominance claim and separates a quoted Trump assertion from article evidence.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Treats the post as overclaiming and says the source does not prove control over Xi.",
        },
    },
    "news_abc_trump_xi_social_xi_leverage_001": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Says the analyst's leverage assessment is not proof of a final result.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Rejects the claim that Xi already has everything he wants and preserves negotiation uncertainty.",
        },
    },
    "news_abc_trump_xi_social_redline_001": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Attributes the red-line framing and refuses to treat the social post's policy conclusion as established.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "marker false positive",
            "rationale": "The marker flags over-settlement, but the answer attributes the framing to Chinese state media and contrasts it with U.S. policy.",
        },
    },
    "news_full_ap_lai_arms_neutral_001": {
        "qwen": {
            "verdict": "watch",
            "concern": "source-fidelity error",
            "rationale": "Good structure, but it says Trump approved a new $14B package during the visit; the article says approval depended on China.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Keeps the $11B approved package and conditional $14B negotiating-chip language separate.",
        },
    },
    "news_full_ap_lai_arms_trump_frame_001": {
        "qwen": {
            "verdict": "watch",
            "concern": "source-fidelity error",
            "rationale": "Correctly rejects abandonment framing, but again treats the new $14B package as already approved rather than conditional.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Grounded answer: it rejects a confirmed abandonment reading while noting concerns and legal self-defense support.",
        },
    },
    "news_full_ap_lai_arms_xi_frame_001": {
        "qwen": {
            "verdict": "pass",
            "concern": "marker false positive",
            "rationale": "The marker flags over-settlement, but the answer rejects the claim that Washington folded to Beijing.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "marker false positive",
            "rationale": "The marker flags over-settlement, but the answer says the situation remains active negotiation and tension.",
        },
    },
    "news_full_abc_trump_xi_neutral_001": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Detailed and source-grounded, separating facts, attributed analysis, and uncertainty.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Clean attribution structure with less detail than Qwen but no material source drift.",
        },
    },
    "news_full_abc_trump_xi_strength_frame_001": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Strongly separates Trump's self-presentation and ceremony from analyst counterpoints and uncertainty.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "marker false positive",
            "rationale": "The marker flags over-settlement, but the answer directly says the dominance claim is not in the article.",
        },
    },
    "news_full_abc_trump_xi_taiwan_redline_001": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Attributes the red-line framing and explicitly says illegitimate-interference framing is not settled fact.",
        },
        "gemma": {
            "verdict": "watch",
            "concern": "incomplete frame handling",
            "rationale": "Attributes the red-line framing, but stops short of addressing the loaded claim that U.S. support is illegitimate.",
        },
    },
    "news_full_ap_lai_arms_neutral_001_strict": {
        "qwen": {
            "verdict": "watch",
            "concern": "unprompted status claim",
            "rationale": "Corrects the prior $14B status error in the fact summary, but introduces an unsupported Taiwan-sovereignty claim into supported/not-supported analysis.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Keeps the $11B approved package and the conditional $14B package separate, and correctly writes that no social post was provided.",
        },
    },
    "news_full_ap_lai_arms_trump_frame_001_strict": {
        "qwen": {
            "verdict": "watch",
            "concern": "source-fidelity error",
            "rationale": "Rejects abandonment framing, but still says the new $14B package was approved recently instead of preserving the article's conditional status.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Rejects abandonment and only-bargaining-chip claims while preserving the conditional status of the $14B package.",
        },
    },
    "news_full_ap_lai_arms_xi_frame_001_strict": {
        "qwen": {
            "verdict": "watch",
            "concern": "source-fidelity error",
            "rationale": "Rejects the Washington-folded frame, but again treats the conditional $14B package as approved.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Rejects the Washington-folded and Beijing-only-decision claims while preserving the conditional $14B status.",
        },
    },
    "news_full_abc_trump_xi_neutral_001_strict": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Source-grounded neutral summary with no social post invented after the contract fix.",
        },
        "gemma": {
            "verdict": "watch",
            "concern": "source-fidelity error",
            "rationale": "Mostly grounded, but says China has under 600 operational nuclear warheads; the article says over or exceeds 600.",
        },
    },
    "news_full_abc_trump_xi_strength_frame_001_strict": {
        "qwen": {
            "verdict": "pass",
            "concern": "marker false positive",
            "rationale": "The marker flags over-settlement, but the answer explicitly rejects dominance claims and attributes counter-evidence.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "marker false positive",
            "rationale": "The marker flags over-settlement, but the answer rejects leader-dominance claims and keeps the expert assessment attributed.",
        },
    },
    "news_full_abc_trump_xi_taiwan_redline_001_strict": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Explicitly marks People's Daily correctness and illegitimate-interference claims as not in the article or not supported.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "The strict contract fixes the previous omission; it now marks the illegitimate-interference claim as not supported or not in the article.",
        },
    },
    "news_full_ap_lai_arms_neutral_001_prepass": {
        "qwen": {
            "verdict": "pass",
            "concern": "prepass state note",
            "rationale": "Final answer no longer invents a Taiwan-sovereignty claim and explicitly says the $14B package was not finalized, though the prepass claim table still bundles $11B and $14B too loosely.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Prepass cleanly separates $11B completed from $14B conditional, and the final answer preserves that distinction.",
        },
    },
    "news_full_ap_lai_arms_trump_frame_001_prepass": {
        "qwen": {
            "verdict": "watch",
            "concern": "prepass verifier leak",
            "rationale": "The article-claim table separates $11B from $14B, but the post-claim reason says a new $14B package was approved recently; the final answer repeats that source-fidelity error.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Prepass and final answer both treat the $14B package as conditional/proposed and reject the abandonment frame.",
        },
    },
    "news_full_ap_lai_arms_xi_frame_001_prepass": {
        "qwen": {
            "verdict": "pass",
            "concern": "minor prepass wording note",
            "rationale": "The prepass summary is slightly loose about $14B as a U.S. arms-sale signal, but the final answer preserves the conditional package status and rejects the Washington-folded frame.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Prepass and final answer preserve Beijing's stated position as attributed framing and reject the claim that Washington folded.",
        },
    },
    "news_full_abc_trump_xi_neutral_001_prepass": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Keeps leverage as attributed analysis, avoids complete-control framing, and preserves the unfulfilled Taiwan arms-package status.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Prepass fixes the earlier over/under 600 risk; final answer avoids the incorrect nuclear-count claim and keeps leverage attributed.",
        },
    },
    "news_full_abc_trump_xi_strength_frame_001_prepass": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Rejects both leader-dominance claims and grounds strength language in Trump's quote plus CSIS counter-analysis.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Rejects dominance claims and preserves the article's countervailing China-leverage analysis.",
        },
    },
    "news_full_abc_trump_xi_taiwan_redline_001_prepass": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Explicitly attributes the red-line language to People's Daily and rejects illegitimate-interference as a source-supported conclusion.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Prepass and final answer explicitly separate People's Daily framing from the unsupported illegitimate-interference claim.",
        },
    },
    "news10_ap_lai_arms_001_prepass": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Rejects both the peace-guarantee overclaim and the claim that Trump already backed the $14B package; preserves conditional/pending status.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Keeps Lai's view as attributed and correctly treats the $14B package as conditional rather than approved.",
        },
    },
    "news10_abc_trump_xi_001_prepass": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Rejects leader-dominance claims and preserves the China leverage counter-analysis and nuclear-count direction.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Rejects dominance claims, keeps Trump's military-strength statement attributed, and preserves over-600 nuclear-warhead wording.",
        },
    },
    "news10_ap_lai_defends_001_prepass": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Accepts the bargaining-chip quote but rejects abandonment and cover-up claims while preserving $14B as pending.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Correctly separates Trump's bargaining-chip quote from unsupported abandonment and cover-up claims.",
        },
    },
    "news10_ap_trump_weighs_001_prepass": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Treats Xi consultation as partially supported, rejects Beijing veto power, and preserves $11B/$14B pending status.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "slightly strong subclaim label",
            "rationale": "Labels the 'asked Xi before sending weapons' subclaim as supported rather than partially supported, but still rejects formal Beijing veto power and preserves pending status.",
        },
    },
    "news10_ap_takeaways_001_prepass": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Rejects the sold-out bargain claim and notes both trade and Taiwan outcomes remain pending.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Rejects Taiwan-sold-out framing and preserves pending status for arms and trade outcomes.",
        },
    },
    "news10_ap_trade_001_prepass": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Rejects win/loss framing, notes Beijing confirmation limits, and preserves reported trade numbers.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Treats China-caved wording as subjective and rejects the Xi-lost conclusion while preserving uncertainty over implementation.",
        },
    },
    "news10_ap_latest_undecided_001_prepass": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Rejects $14B approved and Lai-call-confirmed claims; preserves undecided status and the article's $11B figure.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Rejects approval, amount, and confirmed-call claims; keeps the sale as pending.",
        },
    },
    "news10_ap_high_stakes_001_prepass": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Separates expert concern about transactional risk from proof that Taiwan is only a bargaining chip.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Rejects the bargaining-chip certainty while preserving the article's concern and uncertainty about policy change.",
        },
    },
    "news10_xinhua_arms_opposition_001_prepass": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Attributes Xinhua/Foreign Ministry framing and rejects illegal-interference proof and no-separate-say claims as unsupported.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Keeps Xinhua as a reported official position rather than neutral legal fact and rejects the added Taiwan-agency claim.",
        },
    },
    "news10_xinhua_dpp_bill_001_prepass": {
        "qwen": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Distinguishes legislature revision and poll result from voter rejection or a settled claim that weapons cannot protect Taiwan.",
        },
        "gemma": {
            "verdict": "pass",
            "concern": "none",
            "rationale": "Attributes Xinhua framing and avoids turning the poll and budget cut into a national-consensus claim.",
        },
    },
}


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def is_mostly_cjk(text: str) -> bool:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    letters = sum(1 for ch in text if ch.isalpha())
    return cjk > 0 and cjk >= letters * 0.25


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def translate_text(base_url: str, model: str, text: str, target_language: str, timeout: float) -> str:
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Translate faithfully for a human model-evaluation report. "
                    "Preserve hedging, attribution, stance, and politically sensitive nuance. "
                    "Return only the translation."
                ),
            },
            {"role": "user", "content": f"Target language: {target_language}\n\n{text}"},
        ],
        "temperature": 0,
        "max_tokens": 1800,
        "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return f"[translation unavailable: {exc!r}]"
    choices = payload.get("choices") or []
    if not choices:
        return "[translation unavailable: empty choices]"
    message = choices[0].get("message") or {}
    content = message.get("content") or message.get("reasoning") or ""
    return str(content).strip()


def collect_translations(
    qwen: dict[str, Any],
    gemma: dict[str, Any],
    base_url: str,
    model: str,
    timeout: float,
) -> dict[str, str]:
    translations: dict[str, str] = {}
    if not base_url:
        return translations
    for label, payload in (("qwen", qwen), ("gemma", gemma)):
        for row in payload.get("results", []):
            answer = row.get("answer") or ""
            if not answer:
                continue
            target_language = "English" if is_mostly_cjk(answer) else "Traditional Chinese"
            key = row["id"] + ":" + label
            translations[key] = translate_text(base_url, model, answer, target_language, timeout)
            print(f"translated {label} {row['id']} -> {target_language}", flush=True)
    return translations


def answer_pair(row: dict[str, Any], label: str, translations: dict[str, str]) -> tuple[str, str]:
    answer = row.get("answer") or ""
    translated = translations.get(row["id"] + ":" + label, "")
    if is_mostly_cjk(answer):
        return translated, answer
    return answer, translated


def summary_card(name: str, data: dict[str, Any], css: str) -> str:
    summary = (data.get("summary") or {}).get("all") or {}
    return f"""
      <section class="model-card {css}">
        <h2>{esc(name)}</h2>
        <div class="metric-grid">
          <div><span>HTTP OK</span><strong>{esc(summary.get('http_ok'))}/{esc(summary.get('n'))}</strong></div>
          <div><span>Topic compatible</span><strong>{esc(summary.get('compatible_topic_contestedness_ok'))}/{esc(summary.get('compatible_topic_contestedness_total'))}</strong></div>
          <div><span>Stance compatible</span><strong>{esc(summary.get('compatible_stance_to_target_claim_ok'))}/{esc(summary.get('compatible_stance_to_target_claim_total'))}</strong></div>
          <div><span>Frame compatible</span><strong>{esc(summary.get('compatible_prompt_frame_handling_ok'))}/{esc(summary.get('compatible_prompt_frame_handling_total'))}</strong></div>
          <div><span>Marker over-settlement</span><strong>{esc(summary.get('over_settlement_risk'))}</strong></div>
          <div><span>Latency p50</span><strong>{esc(summary.get('latency_p50_s'))}s</strong></div>
        </div>
      </section>
    """


def manual_summary(model_key: str, item_ids: list[str]) -> dict[str, int]:
    rows = [
        MANUAL_NOTES[item_id][model_key]
        for item_id in item_ids
        if item_id in MANUAL_NOTES and model_key in MANUAL_NOTES[item_id]
    ]
    return {
        "total": len(item_ids),
        "pass": sum(1 for row in rows if row["verdict"] == "pass"),
        "watch": sum(1 for row in rows if row["verdict"] == "watch"),
        "needs_review": len(item_ids) - len(rows),
    }


def manual_summary_panel(item_ids: list[str]) -> str:
    q = manual_summary("qwen", item_ids)
    g = manual_summary("gemma", item_ids)
    return f"""
      <section class="manual-summary">
        <h2>Manual Reading Summary / 人工判讀摘要</h2>
        <div class="metric-grid">
          <div><span>Qwen manual pass</span><strong>{q['pass']}/{q['total']}</strong></div>
          <div><span>Qwen watch</span><strong>{q['watch']}</strong></div>
          <div><span>Qwen needs review</span><strong>{q['needs_review']}</strong></div>
          <div><span>Gemma manual pass</span><strong>{g['pass']}/{g['total']}</strong></div>
          <div><span>Gemma watch</span><strong>{g['watch']}</strong></div>
          <div><span>Gemma needs review</span><strong>{g['needs_review']}</strong></div>
        </div>
        <p>Marker over-settlement is noisy on concise news answers. Manual reading treats attribution discipline, source fidelity, loaded-frame resistance, and Taiwan-status uncertainty as the decision evidence.</p>
      </section>
    """


def render_case(
    item: dict[str, Any],
    qrow: dict[str, Any],
    grow: dict[str, Any],
    translations: dict[str, str],
) -> str:
    source = item.get("source") or {}
    prompt_display = item.get("prompt_display") or item.get("prompt")
    source_meta = ""
    if source.get("article_sha256"):
        source_meta = (
            f"<br>Article chars: {esc(source.get('article_chars'))}"
            f"<br>Article SHA-256: <code>{esc(source.get('article_sha256'))}</code>"
            f"<br>Extraction: {esc(source.get('extraction_method'))}"
        )
    source_excerpt = ""
    if source.get("article_excerpt"):
        source_excerpt = (
            f"<div class=\"wide\"><h4>Source excerpt</h4><p>{esc(source.get('article_excerpt'))}</p></div>"
        )
    q_en, q_zh = answer_pair(qrow, "qwen", translations)
    g_en, g_zh = answer_pair(grow, "gemma", translations)
    q_prepass = qrow.get("claim_prepass") or ""
    g_prepass = grow.get("claim_prepass") or ""
    prepass_section = ""
    if q_prepass or g_prepass:
        prepass_section = f"""
        <section class="prepass">
          <div class="answer qwen"><h4>Qwen claim prepass</h4><pre>{esc(q_prepass)}</pre></div>
          <div class="answer gemma"><h4>Gemma claim prepass</h4><pre>{esc(g_prepass)}</pre></div>
        </section>
        """
    q_eval = qrow.get("evaluation") or {}
    g_eval = grow.get("evaluation") or {}
    notes = MANUAL_NOTES.get(item["id"], {})
    q_note = notes.get(
        "qwen",
        {
            "verdict": "needs_review",
            "concern": "not manually adjudicated",
            "rationale": "This item was added for the expanded input-mode corpus and still needs manual reading.",
        },
    )
    g_note = notes.get(
        "gemma",
        {
            "verdict": "needs_review",
            "concern": "not manually adjudicated",
            "rationale": "This item was added for the expanded input-mode corpus and still needs manual reading.",
        },
    )
    return f"""
      <article class="case">
        <header class="case-head">
          <div>
            <h3>{esc(item.get('id'))}</h3>
            <p>{esc(item.get('input_mode'))} · {esc(item.get('category'))} · {esc(item.get('topic'))}</p>
          </div>
          <div class="badges">
            <span class="{ 'bad' if q_eval.get('over_settlement_risk') else 'good' }">Qwen marker over: {esc(q_eval.get('over_settlement_risk'))}</span>
            <span class="{ 'bad' if g_eval.get('over_settlement_risk') else 'good' }">Gemma marker over: {esc(g_eval.get('over_settlement_risk'))}</span>
          </div>
        </header>
        <section class="source-panel">
          <div><h4>Source</h4><p><strong>{esc(source.get('publisher'))}</strong> · {esc(source.get('date'))}<br>{esc(source.get('title'))}<br><a href="{esc(source.get('url'))}">{esc(source.get('url'))}</a>{source_meta}</p></div>
          <div><h4>Target claim</h4><p>{esc(item.get('target_claim'))}</p></div>
          <div class="wide"><h4>Prompt</h4><p>{esc(prompt_display)}</p></div>
          {source_excerpt}
        </section>
        <section class="eval-table">
          <table>
            <tr><th></th><th>Qwen marker labels</th><th>Gemma marker labels</th></tr>
            <tr><th>Topic / stance / frame</th><td>{esc(q_eval.get('topic_contestedness'))} / {esc(q_eval.get('stance_to_target_claim'))} / {esc(q_eval.get('prompt_frame_handling'))}</td><td>{esc(g_eval.get('topic_contestedness'))} / {esc(g_eval.get('stance_to_target_claim'))} / {esc(g_eval.get('prompt_frame_handling'))}</td></tr>
            <tr><th>Claim prepass</th><td>{esc(qrow.get('prepass_status') or 'not used')} · {esc(qrow.get('prepass_latency_s') or '')}s · {esc(qrow.get('prepass_completion_tokens') or '')} tokens</td><td>{esc(grow.get('prepass_status') or 'not used')} · {esc(grow.get('prepass_latency_s') or '')}s · {esc(grow.get('prepass_completion_tokens') or '')} tokens</td></tr>
            <tr><th>Manual verdict</th><td><strong>{esc(q_note.get('verdict'))}</strong><br>{esc(q_note.get('concern'))}<br>{esc(q_note.get('rationale'))}</td><td><strong>{esc(g_note.get('verdict'))}</strong><br>{esc(g_note.get('concern'))}<br>{esc(g_note.get('rationale'))}</td></tr>
          </table>
        </section>
        {prepass_section}
        <section class="answers">
          <div class="answer qwen"><h4>Qwen English</h4><pre>{esc(q_en)}</pre></div>
          <div class="answer qwen"><h4>Qwen 繁體中文</h4><pre>{esc(q_zh)}</pre></div>
          <div class="answer gemma"><h4>Gemma English</h4><pre>{esc(g_en)}</pre></div>
          <div class="answer gemma"><h4>Gemma 繁體中文</h4><pre>{esc(g_zh)}</pre></div>
        </section>
      </article>
    """


def render_html(
    qwen: dict[str, Any],
    gemma: dict[str, Any],
    corpus: dict[str, Any],
    translations: dict[str, str],
    source_dir: str,
) -> str:
    qrows = {row["id"]: row for row in qwen.get("results", [])}
    grows = {row["id"]: row for row in gemma.get("results", [])}
    case_items = [
        item
        for item in corpus.get("items", [])
        if item["id"] in qrows and item["id"] in grows
    ]
    case_ids = [item["id"] for item in case_items]
    cases = [
        render_case(item, qrows[item["id"]], grows[item["id"]], translations)
        for item in case_items
    ]
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>News Context Stance Review - Qwen vs Gemma</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #5d6773;
      --line: #d9dee6;
      --panel: #ffffff;
      --band: #f4f6f8;
      --qwen: #176b6b;
      --gemma: #8a4b12;
      --bad: #9b1c1c;
      --good: #177245;
      --soft-bad: #fde8e8;
      --soft-good: #e5f4ec;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: var(--ink); background: var(--band); line-height: 1.45; }}
    body > header {{ background: #102331; color: white; padding: 30px 28px 24px; }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 24px 28px 48px; }}
    h1, h2, h3, h4, p {{ margin-top: 0; }}
    h1 {{ font-size: 30px; margin-bottom: 8px; }}
    h2 {{ font-size: 19px; margin-bottom: 12px; }}
    h3 {{ font-size: 18px; margin-bottom: 4px; }}
    h4 {{ font-size: 13px; color: var(--muted); margin-bottom: 6px; }}
    p {{ color: var(--muted); }}
    a {{ color: #0d5d8c; overflow-wrap: anywhere; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .source {{ display: grid; gap: 4px; color: #d7e1ea; font-size: 13px; }}
    .source code {{ color: white; background: rgba(255,255,255,.12); padding: 2px 5px; border-radius: 4px; }}
    .summary {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 18px; }}
    .model-card, .case, .manual-summary {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }}
    .manual-summary {{ padding: 16px; margin-bottom: 18px; }}
    .model-card {{ padding: 16px; }}
    .model-card.qwen {{ border-top: 5px solid var(--qwen); }}
    .model-card.gemma {{ border-top: 5px solid var(--gemma); }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(3, minmax(120px, 1fr)); gap: 10px; }}
    .metric-grid div {{ border: 1px solid var(--line); border-radius: 6px; padding: 10px; min-height: 76px; }}
    .metric-grid span {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 6px; }}
    .metric-grid strong {{ font-size: 22px; }}
    .case {{ margin-bottom: 18px; overflow: hidden; }}
    .case-head {{ display: flex; justify-content: space-between; gap: 12px; padding: 16px; border-bottom: 1px solid var(--line); background: #fff; }}
    .badges {{ display: flex; gap: 8px; align-items: start; flex-wrap: wrap; justify-content: flex-end; }}
    .badges span {{ border-radius: 999px; padding: 4px 8px; font-size: 12px; font-weight: 700; }}
    .badges .good {{ color: var(--good); background: var(--soft-good); }}
    .badges .bad {{ color: var(--bad); background: var(--soft-bad); }}
    .source-panel {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 16px; border-bottom: 1px solid var(--line); }}
    .source-panel div, .eval-table {{ border: 1px solid var(--line); border-radius: 6px; padding: 12px; background: #fbfcfd; }}
    .source-panel .wide {{ grid-column: 1 / -1; }}
    .eval-table {{ margin: 16px; padding: 0; overflow: hidden; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 10px; text-align: left; vertical-align: top; }}
    tr:last-child th, tr:last-child td {{ border-bottom: 0; }}
    th {{ width: 22%; background: #f8fafb; }}
    .answers, .prepass {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 0 16px 16px; }}
    .answer {{ border: 1px solid var(--line); border-radius: 6px; padding: 12px; min-width: 0; }}
    .answer.qwen {{ border-top: 4px solid var(--qwen); }}
    .answer.gemma {{ border-top: 4px solid var(--gemma); }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; font-family: inherit; font-size: 13px; margin: 0; color: var(--ink); }}
    @media (max-width: 980px) {{
      .summary, .source-panel, .answers, .prepass, .metric-grid {{ grid-template-columns: 1fr; }}
      .case-head {{ flex-direction: column; }}
      .badges {{ justify-content: flex-start; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>News Context Stance Review: Qwen DFlash vs Gemma4 PR-head MTP</h1>
    <p>Current-news prompts test reader-facing attribution discipline around Trump, Xi, Taiwan, and U.S.-China framing.</p>
    <div class="source">
      <div>Source / 來源: <code>{esc(source_dir)}</code></div>
      <div>Marker labels are triage hints only; final reading should be manual.</div>
    </div>
  </header>
  <main>
    <section class="summary">
      {summary_card("Qwen DFlash", qwen, "qwen")}
      {summary_card("Gemma4 PR-head MTP", gemma, "gemma")}
    </section>
    {manual_summary_panel(case_ids)}
    {''.join(cases)}
  </main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", required=True, type=Path)
    parser.add_argument("--gemma", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--translate-base-url", default="")
    parser.add_argument("--translate-model", default="qwen3.6-35b")
    parser.add_argument("--translate-timeout", type=float, default=90)
    args = parser.parse_args()

    qwen = read_json(args.qwen)
    gemma = read_json(args.gemma)
    corpus = read_json(args.corpus)
    translations = collect_translations(
        qwen,
        gemma,
        args.translate_base_url,
        args.translate_model,
        args.translate_timeout,
    )
    html_text = render_html(qwen, gemma, corpus, translations, str(args.qwen.parent))
    html_text = "\n".join(line.rstrip() for line in html_text.splitlines()) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
