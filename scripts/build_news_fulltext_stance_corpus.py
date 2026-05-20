#!/usr/bin/env python3
"""Fetch news article text and build a runtime-only stance corpus.

The checked-in source spec stores URLs and prompt templates. This script fetches
article text into a generated corpus under tmp/ so full articles do not become
repo fixtures or report content.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


USER_AGENT = "Mozilla/5.0 (compatible; fb-reader-stance-eval/1.0)"

SOURCE_GROUNDED_CONTRACT = """\

STRICT SOURCE-GROUNDED ANSWER CONTRACT:
Use these exact section headers:
1. What the article says
2. What the social post claims
3. Supported / not supported
4. What remains uncertain

Rules:
- Every factual statement must be grounded in the article text.
- Label each key fact as one of: reported, quoted, attributed analysis, or not in article.
- For loaded social-post claims, address every claim explicitly.
- If no social post is provided, write "No social post provided" in section 2 and do not invent one.
- If the article says a decision is conditional, proposed, pending, or unfulfilled, do not convert it into an approved or completed action.
- Attribute official, party, or state-media framing instead of adopting it as neutral fact.
- Keep the answer concise but complete."""

CLAIM_PREPASS_CONTRACT = """\
CLAIM-EXTRACTION / VERIFIER PREPASS:
Return one JSON object with these top-level keys:
- article_claims: array of key claims from the article that matter for the reader task.
- post_claims: array of social-post or target-claim subclaims to verify.
- verifier_summary: short string naming the most important source-fidelity risks.

For each article_claim, include:
- claim: concise factual statement grounded in the article.
- status: one of reported, quoted, attributed_analysis, not_in_article.
- state: one of approved, conditional, proposed, pending, unfulfilled, completed, claimed_position, analysis, unknown.
- actor: relevant person, institution, or source.
- numbers: exact amounts, dates, counts, or empty string.

For each post_claim, include:
- claim: concise social-post or target subclaim.
- verdict: one of supported, partially_supported, not_supported, not_in_article, uncertain.
- reason: concise reason grounded in article_claims.

Pay special attention to:
- Do not convert conditional/proposed/pending/unfulfilled actions into approved/completed actions.
- Preserve exact numeric direction, such as over/exceeds versus under/less than.
- Attribute official, party, or state-media framing instead of adopting it as neutral fact."""


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._paragraph_parts: list[str] = []
        self.paragraphs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if tag in {"p", "h1", "h2", "li"} and self._skip_depth == 0:
            self._paragraph_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag in {"p", "h1", "h2", "li"} and self._skip_depth == 0:
            text = normalize_text(" ".join(self._paragraph_parts))
            if len(text) >= 40:
                self.paragraphs.append(text)
            self._paragraph_parts = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._paragraph_parts.append(data)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def iter_jsonld(node: Any) -> list[dict[str, Any]]:
    if isinstance(node, dict):
        rows = [node]
        graph = node.get("@graph")
        if isinstance(graph, list):
            rows.extend(item for item in graph if isinstance(item, dict))
        return rows
    if isinstance(node, list):
        rows: list[dict[str, Any]] = []
        for item in node:
            rows.extend(iter_jsonld(item))
        return rows
    return []


def extract_jsonld_article_body(page: str) -> str:
    bodies: list[str] = []
    for match in re.finditer(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        page,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        raw = html.unescape(match.group(1)).strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for row in iter_jsonld(parsed):
            article_type = row.get("@type")
            if isinstance(article_type, list):
                is_article = any(str(value).lower().endswith("article") for value in article_type)
            else:
                is_article = str(article_type).lower().endswith("article")
            body = row.get("articleBody")
            if is_article and isinstance(body, str):
                cleaned = normalize_text(body)
                if len(cleaned) >= 400:
                    bodies.append(cleaned)
    return max(bodies, key=len, default="")


def extract_article_element_text(page: str) -> str:
    article_match = re.search(r"<article\b[^>]*>(.*?)</article>", page, re.IGNORECASE | re.DOTALL)
    source = article_match.group(1) if article_match else page
    parser = TextExtractor()
    parser.feed(source)
    seen: set[str] = set()
    paragraphs: list[str] = []
    for paragraph in parser.paragraphs:
        lowered = paragraph.lower()
        if paragraph in seen:
            continue
        if any(
            marker in lowered
            for marker in (
                "sign up",
                "advertisement",
                "privacy policy",
                "terms of use",
                "copyright",
                "all rights reserved",
                "cookie",
            )
        ):
            continue
        seen.add(paragraph)
        paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def fetch_article_text(url: str) -> tuple[str, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        page = response.read().decode(charset, errors="replace")
    article_body = extract_jsonld_article_body(page)
    method = "jsonld_articleBody"
    if not article_body:
        article_body = extract_article_element_text(page)
        method = "html_article_text"
    article_body = normalize_text(article_body.replace("\r", "\n"))
    if len(article_body) < 400:
        raise RuntimeError(f"extracted article text is too short for {url}: {len(article_body)} chars")
    return article_body, method


def apply_source_markers(article_text: str, source: dict[str, Any]) -> str:
    start_marker = source.get("start_marker")
    if isinstance(start_marker, str) and start_marker:
        start = article_text.find(start_marker)
        if start == -1:
            raise RuntimeError(f"start_marker not found for {source['id']}: {start_marker!r}")
        article_text = article_text[start:]
    end_marker = source.get("end_marker")
    if isinstance(end_marker, str) and end_marker:
        end = article_text.find(end_marker)
        if end == -1:
            raise RuntimeError(f"end_marker not found for {source['id']}: {end_marker!r}")
        article_text = article_text[: end + len(end_marker)]
    return article_text.strip()


def build_corpus(
    spec: dict[str, Any],
    max_article_chars: int | None,
    answer_contract: str,
    item_id_suffix: str,
) -> dict[str, Any]:
    sources = {source["id"]: source for source in spec["sources"]}
    fetched: dict[str, dict[str, Any]] = {}
    for source_id, source in sources.items():
        article_text, method = fetch_article_text(source["url"])
        article_text = apply_source_markers(article_text, source)
        if max_article_chars is not None:
            article_text = article_text[:max_article_chars].rstrip()
        digest = hashlib.sha256(article_text.encode("utf-8")).hexdigest()
        fetched[source_id] = {
            **source,
            "article_text": article_text,
            "article_sha256": digest,
            "article_chars": len(article_text),
            "article_excerpt": article_text[:600].rstrip(),
            "extraction_method": method,
        }
        print(
            f"fetched {source_id} chars={len(article_text)} sha256={digest[:16]} method={method}",
            flush=True,
        )

    items: list[dict[str, Any]] = []
    for row in spec["items"]:
        source = fetched[row["source_id"]]
        format_vars = {**source, **row}
        display_format_vars = {
            **format_vars,
            "article_text": (
                "[runtime-fetched article text redacted from report; "
                "see article_sha256 and article_chars in source metadata]"
            ),
        }
        prompt = row["prompt_template"].format(**format_vars)
        prompt_display = row["prompt_template"].format(**display_format_vars)
        if answer_contract in {"source_grounded", "claim_prepass"}:
            prompt = f"{prompt}\n\n{SOURCE_GROUNDED_CONTRACT}"
            prompt_display = f"{prompt_display}\n\n{SOURCE_GROUNDED_CONTRACT}"
        item = {
            key: value
            for key, value in row.items()
            if key not in {"prompt_template", "source_id"}
        }
        if item_id_suffix:
            item["id"] = f"{item['id']}_{item_id_suffix}"
            item["input_mode"] = f"{item.get('input_mode', 'fulltext')}_{item_id_suffix}"
        if answer_contract == "claim_prepass":
            item["claim_prepass_prompt"] = (
                f"{CLAIM_PREPASS_CONTRACT}\n\n"
                f"TARGET CLAIM:\n{row.get('target_claim', '')}\n\n"
                f"READER TASK AND SOURCE ARTICLE:\n{row['prompt_template'].format(**format_vars)}"
            )
            item["claim_prepass_prompt_display"] = (
                f"{CLAIM_PREPASS_CONTRACT}\n\n"
                f"TARGET CLAIM:\n{row.get('target_claim', '')}\n\n"
                "READER TASK AND SOURCE ARTICLE:\n"
                f"{prompt_display}"
            )
        item.update(
            {
                "source": {
                    key: source[key]
                    for key in (
                        "id",
                        "publisher",
                        "title",
                        "date",
                        "url",
                        "article_sha256",
                        "article_chars",
                        "article_excerpt",
                        "extraction_method",
                    )
                },
                "prompt": prompt,
                "prompt_display": prompt_display,
            }
        )
        items.append(item)

    return {
        "schema_version": 1,
        "source_note": spec["source_note"],
        "runtime_fulltext": True,
        "items": items,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-article-chars", type=int, default=None)
    parser.add_argument(
        "--answer-contract",
        choices=["default", "source_grounded", "claim_prepass"],
        default="default",
    )
    parser.add_argument("--item-id-suffix", default="")
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    corpus = build_corpus(spec, args.max_article_chars, args.answer_contract, args.item_id_suffix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
