#!/usr/bin/env python3
"""
Minimal vLLM latency benchmark for the DGX Spark Tier B endpoint.

Why this exists:
- On some macOS setups, Python's default networking stack may prefer IPv6 for
  *.local hostnames. If your LAN advertises an IPv6 address that is unroutable,
  each request can incur a long connect delay or fail, making benchmarks look
  ~5s slower than they really are.

Use --connect-mode auto to pick a routable address (IPv4 or IPv6) once and send
all requests to that address.
"""

import argparse
import concurrent.futures
import json
import socket
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request


PROMPT = (
    "分類這篇 Facebook 貼文：今天去爬山，風景很好，拍了很多照片，"
    "但下山時膝蓋有點痛。只回傳 JSON："
    '{"commercial":0.0,"political":0.0,"emotional":0.0,"personal":0.0}'
)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def resolve_ipv4(host: str, port: int) -> str:
    infos = socket.getaddrinfo(host, port, family=socket.AF_INET, type=socket.SOCK_STREAM)
    if not infos:
        raise RuntimeError(f"No IPv4 address found for host={host!r}")
    return infos[0][4][0]

def resolve_connectable(host: str, port: int, timeout_s: float = 0.25) -> tuple[str, int]:
    """Return (address, family) for the first connectable getaddrinfo result."""
    infos = socket.getaddrinfo(host, port, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM)
    if not infos:
        raise RuntimeError(f"No addresses found for host={host!r}")
    last_err: Exception | None = None
    for family, socktype, proto, _canon, sockaddr in infos:
        addr = sockaddr[0]
        try:
            sock = socket.socket(family, socktype, proto)
            sock.settimeout(timeout_s)
            sock.connect(sockaddr)
            sock.close()
            return addr, family
        except Exception as exc:  # pragma: no cover
            last_err = exc
            try:
                sock.close()
            except Exception:
                pass
    raise RuntimeError(f"No routable address found for host={host!r}: {last_err!r}")


def maybe_rewrite_base_url(api_base: str, connect_mode: str) -> str:
    if connect_mode == "system":
        return api_base

    parsed = urllib.parse.urlparse(api_base)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported api-base scheme: {parsed.scheme}")
    if not parsed.hostname:
        raise ValueError("api-base missing hostname")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if connect_mode == "ipv4":
        addr = resolve_ipv4(parsed.hostname, port)
        family = socket.AF_INET
    elif connect_mode == "auto":
        addr, family = resolve_connectable(parsed.hostname, port)
    else:
        raise ValueError(f"Unknown connect-mode: {connect_mode!r}")

    # Replace hostname with the selected address.
    host_for_url = addr
    if family == socket.AF_INET6:
        host_for_url = f"[{addr}]"

    netloc = host_for_url
    if parsed.port:
        netloc = f"{host_for_url}:{parsed.port}"
    rewritten = parsed._replace(netloc=netloc)
    return urllib.parse.urlunparse(rewritten)


def one_request(api_base: str, model: str | None, max_tokens: int, timeout_s: int) -> dict:
    payload: dict = {
        "messages": [
            {"role": "system", "content": "你是 Facebook 貼文分類專家。只輸出 JSON。"},
            {"role": "user", "content": PROMPT},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if model:
        payload["model"] = model

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{api_base.rstrip('/')}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as response:
        body = json.loads(response.read().decode("utf-8"))
    elapsed = time.perf_counter() - start
    usage = body.get("usage") or {}
    output_tokens = usage.get("completion_tokens") or 0
    content = (((body.get("choices") or [{}])[0]).get("message") or {}).get("content") or ""
    return {
        "latency_s": elapsed,
        "output_tokens": output_tokens,
        "content": content,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://gx10.local:8000/v1")
    parser.add_argument("--model", default="", help="Optional. Leave empty to let vLLM use its default model.")
    parser.add_argument(
        "--connect-mode",
        choices=("system", "ipv4", "auto"),
        default="system",
        help="How to pick the target IP for api-base (fixes some macOS/Python routing quirks).",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()

    api_base = maybe_rewrite_base_url(args.api_base, args.connect_mode)
    model = args.model.strip() or None

    start = time.perf_counter()
    results: list[dict] = []
    errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(one_request, api_base, model, args.max_tokens, args.timeout)
            for _ in range(args.requests)
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                errors.append(repr(exc))

    wall_s = time.perf_counter() - start
    latencies = [item["latency_s"] for item in results]
    output_tokens = sum(item["output_tokens"] for item in results)
    summary = {
        "api_base": args.api_base,
        "api_base_effective": api_base,
        "model": args.model,
        "connect_mode": args.connect_mode,
        "concurrency": args.concurrency,
        "requested": args.requests,
        "completed": len(results),
        "errors": len(errors),
        "wall_s": round(wall_s, 3),
        "output_tokens": output_tokens,
        "output_tok_per_s": round(output_tokens / wall_s, 3) if wall_s else 0,
        "latency_s": {
            "mean": round(statistics.mean(latencies), 3) if latencies else 0,
            "p50": round(percentile(latencies, 0.50), 3),
            "p90": round(percentile(latencies, 0.90), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3) if latencies else 0,
        },
        "sample": (results[0]["content"][:200] if results else ""),
        "first_error": (errors[0] if errors else ""),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
