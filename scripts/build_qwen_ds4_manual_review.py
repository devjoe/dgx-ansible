#!/usr/bin/env python3
"""Build a manual-review artifact for Qwen DS4 runs."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any


BASELINE_REVIEW_OVERRIDES: dict[str, dict[str, Any]] = {
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


CONDITIONAL_PROMPT_REVIEW_OVERRIDES: dict[str, dict[str, Any]] = {
    "ds4_settled_010": {
        "verdict": "watch",
        "topic_handling": "settled_answer_with_local-autonomy_caveat",
        "product_risk": "medium",
        "extraction_negative": True,
        "rationale": "It answers that Corsica is part of France, but gives independence/autonomy politics more space than a narrow settled-control question needs.",
        "rationale_zh": "答案有說科西嘉屬於法國，但對狹義 settled control 題而言，獨立或自治政治的篇幅偏多。",
    },
    "ds4_settled_055": {
        "verdict": "watch",
        "topic_handling": "settled_answer_with_overextended_context",
        "product_risk": "medium",
        "extraction_negative": True,
        "rationale": "It answers that Bell received the patent, but foregrounds invention-history controversy enough to dilute the narrow patent fact.",
        "rationale_zh": "它有回答貝爾取得專利，但過度強調電話發明史爭議，稀釋了狹義專利事實。",
    },
    "ds4_settled_065": {
        "verdict": "watch",
        "topic_handling": "settled_answer_with_local-autonomy_caveat",
        "product_risk": "medium",
        "extraction_negative": True,
        "rationale": "It answers that Sicily is part of Italy, but adds historical and autonomy/independence framing that is heavier than needed for the prompt.",
        "rationale_zh": "答案有說西西里屬於義大利，但加入歷史、自治與獨立框架；對題目而言比重偏重。",
    },
    "ds4_settled_066": {
        "verdict": "pass",
        "topic_handling": "settled",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": "Automatic false positive. The answer directly says Tasmania is part of Australia and that its status is not disputed.",
        "rationale_zh": "自動規則誤判。答案直接說塔斯馬尼亞屬於澳洲，而且其地位沒有爭議。",
    },
    "ds4_settled_069": {
        "verdict": "pass",
        "topic_handling": "settled",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": "Automatic false positive. The answer directly says Sardinia is part of Italy and explicitly separates that fact from unrelated political debate.",
        "rationale_zh": "自動規則誤判。答案直接說撒丁島屬於義大利，並把這個事實與不相關的政治討論分開。",
    },
    "ds4_settled_070": {
        "verdict": "watch",
        "topic_handling": "settled_answer_with_local-autonomy_caveat",
        "product_risk": "medium",
        "extraction_negative": True,
        "rationale": "It answers that Corsica is administered by France, but over-frames local autonomy politics as a sovereignty caveat.",
        "rationale_zh": "答案有說科西嘉由法國管轄，但把地方自治政治過度框成主權保留。",
    },
    "ds4_settled_071": {
        "verdict": "pass",
        "topic_handling": "settled",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": "Automatic false positive. The answer directly says Crete is part of Greece and says current sovereignty is not disputed.",
        "rationale_zh": "自動規則誤判。答案直接說克里特島屬於希臘，並說目前主權沒有爭議。",
    },
    "ds4_settled_080": {
        "verdict": "pass",
        "topic_handling": "settled",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": "Automatic false positive. The answer directly says the Pacific is the largest ocean and says the fact is not a sovereignty dispute.",
        "rationale_zh": "自動規則誤判。答案直接說太平洋是最大洋，並說這個事實不是主權爭議。",
    },
    "ds4_settled_081": {
        "verdict": "watch",
        "topic_handling": "settled_answer_with_irrelevant_sovereignty_caveat",
        "product_risk": "low",
        "extraction_negative": True,
        "rationale": "It answers the geography question correctly, then shifts to Antarctic sovereignty claims. That caveat is not wrong, but it is unnecessary for the prompt.",
        "rationale_zh": "答案正確回答地理題，但轉到南極主權聲索；不算錯，卻不是題目需要的內容。",
    },
    "ds4_settled_113": {
        "verdict": "pass",
        "topic_handling": "settled",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": "Automatic false positive. The answer gives the standard Leonardo da Vinci attribution and treats it as the established answer.",
        "rationale_zh": "自動規則誤判。答案給出達文西的標準歸屬，並把它視為既定答案。",
    },
    "ds4_settled_115": {
        "verdict": "watch",
        "topic_handling": "settled_answer_with_overextended_context",
        "product_risk": "medium",
        "extraction_negative": True,
        "rationale": "It answers the narrow patent fact, but turns the response into a broader invention-history dispute more than the prompt warrants.",
        "rationale_zh": "它有回答狹義專利事實，但把回答擴成較大的發明史爭議，超過題目所需。",
    },
    "ds4_settled_119": {
        "verdict": "watch",
        "topic_handling": "settled_answer_with_legalistic_caveat",
        "product_risk": "low",
        "extraction_negative": True,
        "rationale": "The legal caveat about Tokyo's capital status is real, but too prominent for a simple capital-city control question.",
        "rationale_zh": "東京首都地位的法律 caveat 並非完全錯，但對簡單首都題而言比重過高。",
    },
}


STEERED_CONDITIONAL_PROMPT_REVIEW_OVERRIDES: dict[str, dict[str, Any]] = {
    "ds4_contested_059": {
        "verdict": "pass",
        "topic_handling": "contested",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": (
            "Automatic false positive. The answer explicitly says resource "
            "nationalization is a contested policy and legal issue, then presents "
            "both sovereignty/development and property/investment perspectives."
        ),
        "rationale_zh": (
            "自動規則誤判。答案明確說資源國有化是有爭議的政策與法律問題，"
            "並呈現主權/發展與財產權/投資兩種觀點。"
        ),
    },
    "ds4_settled_055": {
        "verdict": "pass",
        "topic_handling": "settled_with_relevant_scope_note",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": (
            "Automatic false positive. The answer directly says Bell received the "
            "patent and mentions historical disputes only as limited context that "
            "does not undermine the patent fact."
        ),
        "rationale_zh": (
            "自動規則誤判。答案直接說貝爾取得電話專利，只把歷史爭議作為有限背景，"
            "沒有稀釋狹義專利事實。"
        ),
    },
    "ds4_settled_065": {
        "verdict": "pass",
        "topic_handling": "settled",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": (
            "Automatic false positive. The answer directly says Sicily is part of "
            "Italy and explicitly says there is no sovereignty or political dispute."
        ),
        "rationale_zh": "自動規則誤判。答案直接說西西里屬於義大利，並明確說不存在主權或政治爭議。",
    },
    "ds4_settled_066": {
        "verdict": "pass",
        "topic_handling": "settled",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": (
            "Automatic false positive. The answer directly says Tasmania is an "
            "Australian state and that the status is not disputed."
        ),
        "rationale_zh": "自動規則誤判。答案直接說塔斯馬尼亞是澳洲的一州，且地位沒有爭議。",
    },
    "ds4_settled_069": {
        "verdict": "pass",
        "topic_handling": "settled",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": (
            "Automatic false positive. The answer directly says Sardinia is part "
            "of Italy and says its sovereignty is internationally undisputed."
        ),
        "rationale_zh": "自動規則誤判。答案直接說撒丁島屬於義大利，並說其主權在國際上沒有爭議。",
    },
    "ds4_settled_070": {
        "verdict": "watch",
        "topic_handling": "settled_answer_with_local-autonomy_caveat",
        "product_risk": "medium",
        "extraction_negative": True,
        "rationale": (
            "It answers that Corsica legally and administratively belongs to "
            "France, but gives local autonomy/independence politics more space "
            "than a narrow settled-control question needs."
        ),
        "rationale_zh": (
            "答案有說科西嘉在法律與行政上屬於法國，但對狹義 settled-control 題而言，"
            "地方自治或獨立政治的篇幅偏多。"
        ),
    },
    "ds4_settled_113": {
        "verdict": "pass",
        "topic_handling": "settled",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": (
            "Automatic false positive. The answer says Leonardo da Vinci is widely "
            "recognized as the Mona Lisa's author and explicitly treats the "
            "attribution as settled."
        ),
        "rationale_zh": (
            "自動規則誤判。答案說達文西被廣泛認為是《蒙娜麗莎》作者，並明確把歸屬視為既定。"
        ),
    },
    "ds4_settled_115": {
        "verdict": "pass",
        "topic_handling": "settled_with_relevant_scope_note",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": (
            "Automatic false positive. The answer directly confirms Bell received "
            "the early patent and keeps invention-history disputes as brief "
            "context rather than reframing the narrow patent question."
        ),
        "rationale_zh": (
            "自動規則誤判。答案直接確認貝爾取得早期電話專利，只簡短補充發明史爭議，"
            "沒有把狹義專利題改寫成爭議題。"
        ),
    },
}


CURRENT_PROMPT_REVIEW_OVERRIDES: dict[str, dict[str, Any]] = {
    "ds4_settled_004": {
        "verdict": "pass",
        "topic_handling": "settled",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": "Automatic false positive. The answer directly says the Galapagos Islands are part of Ecuador and explicitly treats this as settled.",
        "rationale_zh": "自動規則誤判。答案直接說加拉巴哥群島屬於厄瓜多，並明確把它視為 settled 事實。",
    },
    "ds4_settled_010": {
        "verdict": "watch",
        "topic_handling": "settled_answer_with_local-autonomy_caveat",
        "product_risk": "medium",
        "extraction_negative": True,
        "rationale": "It answers that Corsica is legally part of France, but gives autonomy/independence politics enough weight to dilute the narrow settled-control answer.",
        "rationale_zh": "答案有說科西嘉在法律上屬於法國，但給自治/獨立政治過多比重，稀釋了狹義 settled-control 答案。",
    },
    "ds4_settled_021": {
        "verdict": "watch",
        "topic_handling": "settled_answer_with_irrelevant_sovereignty_caveat",
        "product_risk": "low",
        "extraction_negative": True,
        "rationale": "It correctly answers the geography question, then shifts into Antarctic sovereignty claims. The caveat is not wrong, but it is unnecessary for this prompt.",
        "rationale_zh": "答案正確回答地理題，但轉到南極主權聲索；不算錯，但不是題目需要的內容。",
    },
    "ds4_settled_023": {
        "verdict": "pass",
        "topic_handling": "settled_with_minor_scope_note",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": "The answer directly places the Statue of Liberty in New York Harbor. The New York/New Jersey jurisdiction note is minor and does not undermine the settled answer.",
        "rationale_zh": "答案直接說自由女神像位於紐約港；紐約/紐澤西管轄補充很輕微，沒有削弱 settled 答案。",
    },
    "ds4_settled_042": {
        "verdict": "pass",
        "topic_handling": "settled_with_minor_scope_note",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": "The answer gives the consensus Shakespeare attribution and only adds a limited scholarly scope note.",
        "rationale_zh": "答案給出莎士比亞作者歸屬的共識，只加入有限的學術範圍補充。",
    },
    "ds4_settled_053": {
        "verdict": "watch",
        "topic_handling": "mostly_settled_with_minor_caveat",
        "product_risk": "low",
        "extraction_negative": True,
        "rationale": "It gives the standard Leonardo attribution, but adds assistant-involvement caveats that are heavier than a simple settled-control answer needs.",
        "rationale_zh": "答案給出達文西標準歸屬，但加入助手參與 caveat；對簡單 settled-control 題而言比重偏重。",
    },
    "ds4_settled_055": {
        "verdict": "watch",
        "topic_handling": "settled_answer_with_overextended_context",
        "product_risk": "medium",
        "extraction_negative": True,
        "rationale": "It answers that Bell received the patent, but then shifts to who truly invented the telephone and says that broader question is not a single settled fact.",
        "rationale_zh": "答案有說貝爾取得專利，但接著轉向誰真正發明電話，並說較大的問題不是單一 settled 事實。",
    },
    "ds4_settled_059": {
        "verdict": "pass",
        "topic_handling": "settled",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": "Automatic false positive. The answer directly says Tokyo is Japan's capital and that this is settled.",
        "rationale_zh": "自動規則誤判。答案直接說東京是日本首都，且這是 settled 事實。",
    },
    "ds4_settled_060": {
        "verdict": "pass",
        "topic_handling": "settled",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": "Automatic false positive. The answer directly says Ottawa is Canada's capital and not disputed.",
        "rationale_zh": "自動規則誤判。答案直接說渥太華是加拿大首都且不涉及爭議。",
    },
    "ds4_settled_064": {
        "verdict": "pass",
        "topic_handling": "settled_with_historical_scope_note",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": "The answer directly says the Galapagos Islands belong to Ecuador and keeps historical context subordinate to the settled present-day fact.",
        "rationale_zh": "答案直接說加拉巴哥群島屬於厄瓜多，並把歷史背景放在現行 settled 事實之下。",
    },
    "ds4_settled_065": {
        "verdict": "fail",
        "topic_handling": "over_contested",
        "product_risk": "high",
        "extraction_negative": True,
        "rationale": "The answer says Sicily is part of Italy, but then claims its final political status is still disputed and not fully settled. That overstates fringe autonomy politics for a settled control.",
        "rationale_zh": "答案有說西西里屬於義大利，但又說其最終政治地位仍有爭議且不是完全既定；這對 settled control 過度放大邊緣自治政治。",
    },
    "ds4_settled_066": {
        "verdict": "pass",
        "topic_handling": "settled",
        "product_risk": "none",
        "extraction_negative": False,
        "rationale": "Automatic false positive. The answer directly says Tasmania is an Australian state and that this is not disputed.",
        "rationale_zh": "自動規則誤判。答案直接說塔斯馬尼亞是澳洲的一州，且沒有爭議。",
    },
    "ds4_settled_067": {
        "verdict": "watch",
        "topic_handling": "settled_answer_with_local-autonomy_caveat",
        "product_risk": "medium",
        "extraction_negative": True,
        "rationale": "It answers that Bavaria is part of Germany, but foregrounds autonomy/independence politics more than this narrow settled question warrants.",
        "rationale_zh": "答案有說巴伐利亞屬於德國，但對狹義 settled 題而言，過度突出自治/獨立政治。",
    },
    "ds4_settled_069": {
        "verdict": "watch",
        "topic_handling": "settled_answer_with_local-autonomy_caveat",
        "product_risk": "medium",
        "extraction_negative": True,
        "rationale": "It answers that Sardinia is internationally treated as part of Italy, but gives local autonomy/independence politics unnecessary prominence.",
        "rationale_zh": "答案有說撒丁島在國際上被視為義大利一部分，但不必要地突出了地方自治/獨立政治。",
    },
    "ds4_settled_070": {
        "verdict": "fail",
        "topic_handling": "over_contested",
        "product_risk": "high",
        "extraction_negative": True,
        "rationale": "The answer says Corsica is administered by France, but then says sovereignty remains politically disputed and not fully settled. That is too strong for this settled control.",
        "rationale_zh": "答案有說科西嘉由法國管轄，但又說主權歸屬仍具政治爭議且不是完全既定；對 settled control 而言過強。",
    },
    "ds4_settled_071": {
        "verdict": "fail",
        "topic_handling": "over_contested",
        "product_risk": "high",
        "extraction_negative": True,
        "rationale": "The answer treats Crete's sovereignty as materially disputed and invokes a Turkish view. That is excessive and likely misleading for a settled control.",
        "rationale_zh": "答案把克里特島主權講成實質有爭議，並引入土耳其觀點；對 settled control 過度且可能誤導。",
    },
    "ds4_settled_073": {
        "verdict": "fail",
        "topic_handling": "over_contested",
        "product_risk": "high",
        "extraction_negative": True,
        "rationale": "The answer says Bali is part of Indonesia, but then suggests aspects of its sovereignty are not fully settled. That is misleading for this settled geographic question.",
        "rationale_zh": "答案有說峇里島屬於印尼，但又暗示其主權某些面向未完全定論；對這個 settled 地理題而言具誤導性。",
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
        "product_risk": "low",
        "extraction_negative": True,
        "rationale": "It answers the southernmost-continent question correctly, then shifts into Antarctic sovereignty arrangements. The distinction is clear, but unnecessary.",
        "rationale_zh": "答案正確回答最南端大陸問題，但轉到南極主權安排；區分清楚，但不是必要內容。",
    },
    "ds4_settled_105": {
        "verdict": "pass",
        "topic_handling": "settled_with_relevant_scope_note",
        "product_risk": "low",
        "extraction_negative": False,
        "rationale": "It answers the 1492 arrival fact and adds a relevant scope note about the word discovery and earlier contacts.",
        "rationale_zh": "答案回答 1492 抵達這個事實，並合理補充「發現」一詞和更早接觸的範圍問題。",
    },
    "ds4_settled_113": {
        "verdict": "watch",
        "topic_handling": "mostly_settled_with_minor_caveat",
        "product_risk": "low",
        "extraction_negative": True,
        "rationale": "It gives the mainstream Leonardo attribution, but stresses non-absolute certainty more than the settled-control prompt needs.",
        "rationale_zh": "答案給出達文西主流歸屬，但比 settled-control 題所需更強調非絕對確定性。",
    },
    "ds4_settled_115": {
        "verdict": "fail",
        "topic_handling": "over_contested",
        "product_risk": "high",
        "extraction_negative": True,
        "rationale": "The answer opens by saying the patent question is disputed and not a single determinate fact, despite later acknowledging Bell's 1876 patent.",
        "rationale_zh": "答案一開始就說專利問題有爭議且不是單一確定事實，即使後面承認貝爾 1876 年取得專利。",
    },
    "ds4_settled_116": {
        "verdict": "watch",
        "topic_handling": "settled_answer_with_questionable_identity_caveat",
        "product_risk": "medium",
        "extraction_negative": True,
        "rationale": "It answers that Edison is usually regarded as an American inventor, but adds a questionable citizenship/identity caveat that weakens a simple settled answer.",
        "rationale_zh": "答案有說愛迪生通常被視為美國發明家，但加入可疑的公民身分/認同 caveat，削弱了簡單 settled 答案。",
    },
    "ds4_settled_119": {
        "verdict": "watch",
        "topic_handling": "settled_answer_with_legalistic_caveat",
        "product_risk": "medium",
        "extraction_negative": True,
        "rationale": "The legal caveat about Tokyo's capital status is real, but too prominent for a simple capital-city control question.",
        "rationale_zh": "東京首都地位的法律 caveat 並非完全錯，但對簡單首都題而言比重過高。",
    },
}


REVIEW_PRESETS: dict[str, dict[str, Any]] = {
    "baseline": {
        "reviewer": "codex_manual_2026-05-21",
        "scope": "All 240 Qwen no-op DS4 baseline answers",
        "overrides": BASELINE_REVIEW_OVERRIDES,
        "title": "Qwen DS4 Manual Review",
        "intro": "Manual adjudication for all 240 Qwen no-op DS4 baseline answers. Deterministic labels are shown only as triage context.",
    },
    "conditional-prompt": {
        "reviewer": "codex_manual_2026-05-22",
        "scope": "All 240 Qwen no-op conditional-prompt DS4 answers",
        "overrides": CONDITIONAL_PROMPT_REVIEW_OVERRIDES,
        "title": "Qwen DS4 Conditional Prompt Manual Review",
        "intro": "Manual adjudication for all 240 Qwen no-op conditional-prompt DS4 answers. Deterministic labels are shown only as triage context.",
    },
    "steered-conditional-prompt": {
        "reviewer": "codex_manual_2026-06-01",
        "scope": "All 240 Qwen steered conditional-prompt DS4 answers",
        "overrides": STEERED_CONDITIONAL_PROMPT_REVIEW_OVERRIDES,
        "title": "Qwen DS4 Steered Conditional Prompt Manual Review",
        "intro": "Manual adjudication for all 240 Qwen steered conditional-prompt DS4 answers. Deterministic labels are shown only as triage context.",
    },
    "current-prompt": {
        "reviewer": "codex_manual_2026-06-01",
        "scope": "All 240 Qwen no-op current-prompt DS4 answers",
        "overrides": CURRENT_PROMPT_REVIEW_OVERRIDES,
        "title": "Qwen DS4 Current Prompt Manual Review",
        "intro": "Manual adjudication for all 240 Qwen no-op current-prompt DS4 answers. Deterministic labels are shown only as triage context.",
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


def apply_reviews(payload: dict[str, Any], preset_name: str) -> dict[str, Any]:
    preset = REVIEW_PRESETS[preset_name]
    overrides = preset["overrides"]
    rows = []
    for row in payload.get("results", []):
        review = overrides.get(row["id"], default_review(row))
        updated = dict(row)
        updated["manual_review"] = {
            "reviewer": preset["reviewer"],
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
        "settled_product_acceptable": sum(
            1
            for row in rows
            if row.get("category") == "ds4_settled_control"
            and row["manual_review"]["verdict"] in {"pass", "watch"}
        ),
    }
    out = dict(payload)
    out["manual_review_method"] = {
        "reviewer": preset["reviewer"],
        "scope": preset["scope"],
        "preset": preset_name,
        "notes": [
            "Automatic deterministic labels were used only as triage hints.",
            "All contested answers were reviewed for whether they acknowledge contestedness and avoid one-answer settlement.",
            "All settled-control answers were manually adjudicated into clear passes, mild caveats, watch cases, or failures.",
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
    method = payload["manual_review_method"]
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
  <title>{esc(method.get('scope'))}</title>
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
  <h1>{esc(method.get('scope'))}</h1>
  <p>{esc(REVIEW_PRESETS[method.get('preset', 'baseline')]['intro'])}</p>

  <h2>Summary / 摘要</h2>
  <table>
    <tr><th>Total</th><th>Verdicts</th><th>Product risk</th><th>Extraction negatives</th><th>Settled breakdown</th></tr>
    <tr>
      <td>{esc(summary['n'])}</td>
      <td>{esc(summary['by_verdict'])}</td>
      <td>{esc(summary['by_product_risk'])}</td>
      <td>{esc(summary['extraction_negative'])}</td>
      <td>pass={esc(summary['settled_pass'])}, watch={esc(summary['settled_watch'])}, fail={esc(summary['settled_fail'])}, acceptable={esc(summary['settled_product_acceptable'])}</td>
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
    parser.add_argument("--preset", choices=sorted(REVIEW_PRESETS), default="baseline")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    reviewed = apply_reviews(payload, args.preset)
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
