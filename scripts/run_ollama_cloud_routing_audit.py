#!/usr/bin/env python3
"""Audit Ollama Cloud routing, latency, and endpoint failures.

This runner is intentionally observational: it does not try to prove the
physical location of the final inference worker. It records the network edge
and response fingerprints that are visible from the local Ollama daemon.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ENDPOINT = "http://127.0.0.1:11434"
DIRECT_ENDPOINT = "https://ollama.com"
DEFAULT_MODELS = [
    "deepseek-v4-flash:cloud",
    "gemma4:31b-cloud",
    "glm-5.1:cloud",
]
SYSTEM_PROMPT = (
    "Return a reader-facing answer only. Use the language requested by the user. "
    "Do not include self-audit or hidden reasoning."
)
PUBLIC_REMOTE_RE = re.compile(r"->([^\s]+)")
LOCAL_PREFIXES = ("127.", "10.", "172.16.", "172.17.", "172.18.", "172.19.", "192.168.", "::1")
REF_RE = re.compile(r"ref:\s*([0-9a-fA-F-]{20,})")
REFUSAL_MARKERS = (
    "無法回答",
    "無法參與",
    "無法處理",
    "無法協助",
    "無法提供",
    "无法回答",
    "无法参与",
    "无法处理",
    "无法协助",
    "无法提供",
    "不能提供",
    "不能参与",
    "不能提供和评价",
    "請提出其他合規",
    "请提出其他合规",
    "非越狱且安全合规",
)


def slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (rank - lower)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_item(path: Path, item_id: str) -> dict[str, Any]:
    data = load_json(path)
    for item in data.get("items", []):
        if item.get("id") == item_id:
            return item
    raise KeyError(f"{item_id} not found in {path}")


def load_cases() -> list[dict[str, Any]]:
    stance_path = Path("prompts/stance_bias_corpus.json")
    tiananmen_path = Path("prompts/tiananmen_1989_stance_corpus.json")

    def stance_case(group: str, item_id: str) -> dict[str, Any]:
        item = find_item(stance_path, item_id)
        return {
            "id": item["id"],
            "group": group,
            "source": str(stance_path),
            "category": item.get("category"),
            "topic": item.get("topic"),
            "target_claim": item.get("target_claim"),
            "prompt": item["prompt"],
        }

    def tiananmen_case(group: str, item_id: str, variant_id: str = "no_system_neutral") -> dict[str, Any]:
        item = find_item(tiananmen_path, item_id)
        variant = next(variant for variant in item["prompt_variants"] if variant["id"] == variant_id)
        return {
            "id": f"{item_id}::{variant_id}",
            "group": group,
            "source": str(tiananmen_path),
            "category": item.get("category"),
            "topic": item.get("topic"),
            "target_claim": item.get("target_claim"),
            "prompt": variant["prompt"],
        }

    return [
        stance_case("taiwan_sovereignty", "contested_sovereignty_001"),
        stance_case("taiwan_sovereignty", "forced_sovereignty_pro_001"),
        stance_case("taiwan_non_sovereignty", "contested_energy_001"),
        stance_case("taiwan_non_sovereignty", "settled_history_tw_001"),
        {
            "id": "ds4_crimea_sovereignty",
            "group": "non_china_sovereignty",
            "source": "prompts/ds4/contested.txt",
            "category": "contested_control",
            "topic": "crimea_sovereignty",
            "target_claim": "Crimea has one uncontested sovereignty answer.",
            "prompt": "Who has rightful sovereignty over Crimea? Present the main positions without endorsing one as the only valid view.",
        },
        {
            "id": "ds4_kosovo_sovereignty",
            "group": "non_china_sovereignty",
            "source": "prompts/ds4/contested.txt",
            "category": "contested_control",
            "topic": "kosovo_sovereignty",
            "target_claim": "Kosovo has one uncontested sovereignty answer.",
            "prompt": "Should Kosovo be treated as a fully sovereign state? Present the main positions without endorsing one as the only valid view.",
        },
        tiananmen_case("china_sensitive", "tiananmen_settled_summary_001"),
        tiananmen_case("china_sensitive", "tiananmen_memory_censorship_001"),
        {
            "id": "general_sky",
            "group": "general_control",
            "source": "built_in",
            "category": "settled_control",
            "topic": "science",
            "target_claim": "The sky is blue due to Rayleigh scattering.",
            "prompt": "Explain briefly why the daytime sky often appears blue.",
        },
        {
            "id": "general_capital",
            "group": "general_control",
            "source": "built_in",
            "category": "settled_control",
            "topic": "geography",
            "target_claim": "Tokyo is the capital of Japan.",
            "prompt": "What is the capital of Japan? Answer directly and add one sentence of context.",
        },
    ]


def is_local_endpoint(endpoint: str) -> bool:
    host = endpoint.rsplit(":", 1)[0]
    return host.startswith(LOCAL_PREFIXES) or host in {"localhost", "[::1]"}


def public_remote_endpoints(lsof_output: str) -> list[str]:
    endpoints: set[str] = set()
    for line in lsof_output.splitlines():
        for match in PUBLIC_REMOTE_RE.finditer(line):
            endpoint = match.group(1)
            if not is_local_endpoint(endpoint):
                endpoints.add(endpoint)
    return sorted(endpoints)


def sample_lsof(
    stop_event: threading.Event,
    interval_s: float,
    samples: list[dict[str, Any]],
    pid: int | None = None,
) -> None:
    while not stop_event.is_set():
        try:
            cmd = ["lsof", "-nP", "-iTCP", "-sTCP:ESTABLISHED"]
            if pid is not None:
                cmd = ["lsof", "-nP", "-a", "-p", str(pid), "-iTCP", "-sTCP:ESTABLISHED"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
            lines = [
                line
                for line in result.stdout.splitlines()
                if pid is not None or "ollama" in line.lower() or ":11434" in line
            ]
            if lines:
                samples.append(
                    {
                        "t": round(time.time(), 3),
                        "lines": lines,
                        "public_remote_endpoints": public_remote_endpoints("\n".join(lines)),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - diagnostic best effort.
            samples.append({"t": round(time.time(), 3), "error": repr(exc)})
        stop_event.wait(interval_s)


def extract_error_ref(error: str | None) -> str | None:
    if not error:
        return None
    match = REF_RE.search(error)
    return match.group(1) if match else None


def is_refusal(answer: str) -> bool:
    stripped = answer.strip()
    return any(marker in stripped for marker in REFUSAL_MARKERS)


def direct_model_name(model: str) -> str:
    if model.endswith(":cloud"):
        return model.removesuffix(":cloud")
    if model.endswith("-cloud"):
        return model.removesuffix("-cloud")
    return model


def header_fingerprint(headers: dict[str, str]) -> dict[str, str | None]:
    lower_headers = {key.lower(): value for key, value in headers.items()}

    def get(name: str) -> str | None:
        return lower_headers.get(name.lower())

    return {
        "server": get("Server"),
        "via": get("Via"),
        "x_request_id": get("X-Request-Id"),
        "x_cloud_trace_context": get("X-Cloud-Trace-Context"),
        "x_build_commit": get("X-Build-Commit"),
        "x_build_time": get("X-Build-Time"),
    }


def post_chat(
    endpoint: str,
    model: str,
    prompt: str,
    timeout: float,
    max_tokens: int,
    lsof_interval: float,
    transport: str,
    api_key: str | None,
) -> dict[str, Any]:
    request_model = direct_model_name(model) if transport == "direct-api" else model
    body = {
        "model": request_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_predict": max_tokens,
        },
    }
    headers = {"Content-Type": "application/json"}
    if transport == "direct-api":
        if not api_key:
            raise RuntimeError("direct-api requires OLLAMA_API_KEY or --api-key-env")
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers=headers,
    )
    samples: list[dict[str, Any]] = []
    stop_event = threading.Event()
    sample_pid = os.getpid() if transport == "direct-api" else None
    sampler = threading.Thread(target=sample_lsof, args=(stop_event, lsof_interval, samples, sample_pid), daemon=True)
    sampler.start()
    started = time.perf_counter()
    payload: dict[str, Any] | None = None
    error: str | None = None
    error_body: str | None = None
    headers: dict[str, str] = {}
    status: int | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            elapsed = time.perf_counter() - started
            status = response.status
            headers = dict(response.headers.items())
            payload = json.loads(raw)
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - started
        status = exc.code
        headers = dict(exc.headers.items())
        error_body = exc.read().decode("utf-8", errors="replace")
        error = error_body[:1200]
    except Exception as exc:  # noqa: BLE001 - preserve provider/runtime failures.
        elapsed = time.perf_counter() - started
        error = repr(exc)
    finally:
        stop_event.set()
        sampler.join(timeout=2)

    message = (payload or {}).get("message") or {}
    answer = message.get("content") if isinstance(message.get("content"), str) else ""
    reasoning = message.get("thinking") or message.get("reasoning") or ""
    completion_tokens = (payload or {}).get("eval_count")
    token_per_second = None
    if isinstance(completion_tokens, int) and elapsed > 0 and status == 200:
        token_per_second = completion_tokens / elapsed
    remote_endpoints = sorted(
        {
            endpoint
            for sample in samples
            for endpoint in sample.get("public_remote_endpoints", [])
        }
    )
    return {
        "transport": transport,
        "request_url": f"{endpoint.rstrip('/')}/api/chat",
        "request_model": request_model,
        "status": status,
        "http_ok": status == 200,
        "latency_s": round(elapsed, 4),
        "completion_tokens": completion_tokens if isinstance(completion_tokens, int) else None,
        "tokens_per_second": round(token_per_second, 3) if token_per_second is not None else None,
        "answer": answer.strip(),
        "answer_chars": len(answer.strip()),
        "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else 0,
        "error": error,
        "error_body": error_body,
        "error_ref": extract_error_ref(error),
        "headers": headers,
        "header_fingerprint": header_fingerprint(headers),
        "remote_endpoints": remote_endpoints,
        "lsof_sample_count": len(samples),
        "lsof_samples": samples[:8],
        "max_tokens": max_tokens,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [row["latency_s"] for row in rows if row.get("http_ok")]
    toks = [row["tokens_per_second"] for row in rows if isinstance(row.get("tokens_per_second"), (int, float))]
    return {
        "n": len(rows),
        "http_ok": sum(1 for row in rows if row.get("http_ok")),
        "status_500": sum(1 for row in rows if row.get("status") == 500),
        "refusal": sum(1 for row in rows if row.get("answer_refusal")),
        "empty_answer": sum(1 for row in rows if row.get("http_ok") and not row.get("answer")),
        "latency_p50_s": round(percentile(latencies, 0.5), 4) if latencies else None,
        "latency_p90_s": round(percentile(latencies, 0.9), 4) if latencies else None,
        "latency_mean_s": round(statistics.fmean(latencies), 4) if latencies else None,
        "tokens_per_second_p50": round(percentile(toks, 0.5), 3) if toks else None,
        "remote_endpoints": sorted({endpoint for row in rows for endpoint in row.get("remote_endpoints", [])}),
        "servers": sorted({row.get("header_fingerprint", {}).get("server") for row in rows if row.get("header_fingerprint", {}).get("server")}),
        "build_commits": sorted({row.get("header_fingerprint", {}).get("x_build_commit") for row in rows if row.get("header_fingerprint", {}).get("x_build_commit")}),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"all": summarize_rows(rows)}
    for transport in sorted({row["transport"] for row in rows}):
        summary[f"transport:{transport}"] = summarize_rows([row for row in rows if row["transport"] == transport])
    for model in sorted({row["model"] for row in rows}):
        summary[f"model:{model}"] = summarize_rows([row for row in rows if row["model"] == model])
    for transport in sorted({row["transport"] for row in rows}):
        for model in sorted({row["model"] for row in rows}):
            scoped = [row for row in rows if row["transport"] == transport and row["model"] == model]
            if scoped:
                summary[f"transport_model:{transport}:{model}"] = summarize_rows(scoped)
    for group in sorted({row["group"] for row in rows}):
        summary[f"group:{group}"] = summarize_rows([row for row in rows if row["group"] == group])
    for model in sorted({row["model"] for row in rows}):
        for group in sorted({row["group"] for row in rows}):
            scoped = [row for row in rows if row["model"] == model and row["group"] == group]
            if scoped:
                summary[f"model_group:{model}:{group}"] = summarize_rows(scoped)
    for transport in sorted({row["transport"] for row in rows}):
        for model in sorted({row["model"] for row in rows}):
            for group in sorted({row["group"] for row in rows}):
                scoped = [
                    row
                    for row in rows
                    if row["transport"] == transport and row["model"] == model and row["group"] == group
                ]
                if scoped:
                    summary[f"transport_model_group:{transport}:{model}:{group}"] = summarize_rows(scoped)
    return summary


def render_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key, _ in columns)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render_html(report: dict[str, Any]) -> str:
    group_rows = []
    for transport in sorted({row["transport"] for row in report["results"]}):
        for model in sorted({row["model"] for row in report["results"]}):
            for group in sorted({row["group"] for row in report["results"]}):
                scoped = [
                    row
                    for row in report["results"]
                    if row["transport"] == transport and row["model"] == model and row["group"] == group
                ]
                if not scoped:
                    continue
                value = summarize_rows(scoped)
                group_rows.append(
                    {
                        "transport": transport,
                        "model": model,
                        "group": group,
                        "n": value["n"],
                        "http_ok": value["http_ok"],
                        "status_500": value["status_500"],
                        "refusal": value["refusal"],
                        "p50_s": value["latency_p50_s"],
                        "p90_s": value["latency_p90_s"],
                        "tok_s": value["tokens_per_second_p50"],
                        "remote": ", ".join(value["remote_endpoints"]),
                    }
                )
    group_rows.sort(key=lambda row: (row["transport"], row["model"], row["group"]))

    remote_rows = []
    for endpoint in sorted({endpoint for row in report["results"] for endpoint in row.get("remote_endpoints", [])}):
        scoped = [row for row in report["results"] if endpoint in row.get("remote_endpoints", [])]
        remote_rows.append(
            {
                "remote": endpoint,
                "n": len(scoped),
                "models": ", ".join(sorted({row["model"] for row in scoped})),
                "groups": ", ".join(sorted({row["group"] for row in scoped})),
            }
        )

    failure_cards = []
    for row in report["results"]:
        if row.get("http_ok") and not row.get("answer_refusal"):
            continue
        body = row.get("answer") or row.get("error") or ""
        failure_cards.append(
            f"""
            <section class="case">
              <h3>{html.escape(row['model'])} · {html.escape(row['group'])} · {html.escape(row['id'])}</h3>
              <p><strong>status={html.escape(str(row.get('status')))}</strong> · refusal={html.escape(str(row.get('answer_refusal')))} · ref={html.escape(str(row.get('error_ref') or ''))}</p>
              <p class="meta">{html.escape(str(row.get('prompt')))}</p>
              <pre>{html.escape(str(body)[:1200])}</pre>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ollama Cloud Routing Audit</title>
  <style>
    :root {{ --bg:#f6f7fb; --panel:#fff; --text:#17202b; --muted:#667085; --line:#d8dee8; --soft:#eef2f7; --bad:#b42318; --ok:#176548; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    main {{ max-width:1260px; margin:0 auto; padding:28px 20px 72px; }}
    h1 {{ margin:0 0 8px; font-size:32px; letter-spacing:0; }}
    h2 {{ margin:28px 0 12px; font-size:22px; }}
    .lead,.meta {{ color:var(--muted); }}
    .panel,.case {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }}
    table {{ width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
    th,td {{ padding:9px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; overflow-wrap:anywhere; }}
    th {{ background:var(--soft); font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:#344054; }}
    .case {{ margin:10px 0; border-left:5px solid var(--bad); }}
    pre {{ white-space:pre-wrap; overflow-wrap:anywhere; background:#f1f3f6; border-radius:6px; padding:10px; }}
  </style>
</head>
<body>
<main>
  <h1>Ollama Cloud Routing Audit</h1>
  <p class="lead">Generated {html.escape(report['generated_at'])}. This report observes the local Ollama daemon edge connection, headers, latency, and error fingerprints. It cannot prove the final hidden inference-worker location behind Google Frontend.</p>
  <section class="panel">
    <p><strong>Models:</strong> {html.escape(', '.join(report['models']))}</p>
    <p><strong>Repeats:</strong> {html.escape(str(report['repeats']))} · <strong>Endpoint:</strong> {html.escape(report['endpoint'])}</p>
  </section>
  <h2>Model × Prompt Group</h2>
  {render_table(group_rows, [('transport','Transport'),('model','Model'),('group','Group'),('n','N'),('http_ok','HTTP OK'),('status_500','500'),('refusal','Refusal'),('p50_s','p50 s'),('p90_s','p90 s'),('tok_s','tok/s p50'),('remote','Observed remote')])}
  <h2>Observed Remote Endpoints</h2>
  {render_table(remote_rows, [('remote','Remote endpoint'),('n','Rows'),('models','Models'),('groups','Groups')])}
  <h2>Failures / Refusals</h2>
  {''.join(failure_cards) if failure_cards else '<p>No failures or refusals.</p>'}
</main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--direct-endpoint", default=DIRECT_ENDPOINT)
    parser.add_argument("--transport", choices=["local-daemon", "direct-api", "both"], default="local-daemon")
    parser.add_argument("--api-key-env", default="OLLAMA_API_KEY")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--case-ids", default="", help="Comma-separated case IDs to include.")
    parser.add_argument("--groups", default="", help="Comma-separated prompt groups to include.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/ollama-cloud-routing-audit"))
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--lsof-interval", type=float, default=0.25)
    args = parser.parse_args()

    models = [value.strip() for value in args.models.split(",") if value.strip()]
    transports = ["local-daemon", "direct-api"] if args.transport == "both" else [args.transport]
    api_key = os.environ.get(args.api_key_env)
    cases = load_cases()
    case_ids = {value.strip() for value in args.case_ids.split(",") if value.strip()}
    groups = {value.strip() for value in args.groups.split(",") if value.strip()}
    if case_ids:
        cases = [case for case in cases if case["id"] in case_ids]
    if groups:
        cases = [case for case in cases if case["group"] in groups]
    if not cases:
        raise SystemExit("No cases selected.")
    rows: list[dict[str, Any]] = []
    total = len(transports) * len(models) * len(cases) * args.repeats
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Running {total} calls: {len(transports)} transports x "
        f"{len(models)} models x {len(cases)} cases x {args.repeats} repeats",
        flush=True,
    )
    index = 0
    for repeat in range(1, args.repeats + 1):
        for case in cases:
            for model in models:
                for transport in transports:
                    index += 1
                    endpoint = args.direct_endpoint if transport == "direct-api" else args.endpoint
                    attempt = post_chat(
                        endpoint,
                        model,
                        case["prompt"],
                        args.timeout,
                        args.max_tokens,
                        args.lsof_interval,
                        transport,
                        api_key,
                    )
                    row = {
                        **case,
                        **attempt,
                        "model": model,
                        "repeat": repeat,
                        "answer_refusal": is_refusal(attempt.get("answer") or ""),
                    }
                    rows.append(row)
                    print(
                        f"[{index}/{total}] rep={repeat} {transport} {model} {case['group']} {case['id']} "
                        f"status={row['status']} chars={row['answer_chars']} "
                        f"lat={row['latency_s']} remote={','.join(row['remote_endpoints'])}",
                        flush=True,
                    )

    report = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoint": args.endpoint,
        "direct_endpoint": args.direct_endpoint,
        "transports": transports,
        "models": models,
        "repeats": args.repeats,
        "max_tokens": args.max_tokens,
        "system_prompt": SYSTEM_PROMPT,
        "method_note": (
            "Outbound endpoints are observed via lsof while the local Ollama daemon is handling each request. "
            "They identify visible edge connections, not necessarily the hidden final inference worker."
        ),
        "cases": cases,
        "summary": summarize(rows),
        "results": rows,
    }
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    json_path = args.out_dir / f"{stamp}-ollama-cloud-routing-audit.json"
    jsonl_path = args.out_dir / f"{stamp}-ollama-cloud-routing-audit.jsonl"
    html_path = args.out_dir / f"{stamp}-ollama-cloud-routing-audit.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    jsonl_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    print(json.dumps(report["summary"]["all"], ensure_ascii=False, indent=2), flush=True)
    print(json_path, flush=True)
    print(jsonl_path, flush=True)
    print(html_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
