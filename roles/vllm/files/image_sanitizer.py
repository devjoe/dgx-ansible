"""Image-sanitizing reverse proxy in front of vLLM.

Why this exists
---------------
vLLM's multimodal pipeline calls Pillow under the hood. When Pillow can't
decode an image (animated WebP / multi-frame GIF, FB CDN URL whose signed
token mismatched and returned a 403 HTML body, SVG data URI, etc.) the
HuggingFace `image_processor` raises

    ValueError: not enough values to unpack (expected 2, got 1)

while trying to read `image.size`, which kills the WHOLE request — text
prompt and all other images included. The fb-reader client retries
text-only on this signature, but it costs a wasted round trip and the
post then has zero visual analysis.

This proxy sits in front of vLLM on port 8000 and:

1. Accepts the same `/v1/chat/completions` payload.
2. Walks every `messages[*].content[*]` item; for `type: "image_url"` it
   fetches the URL (with a short timeout), decodes via Pillow in a way
   that tolerates animated frames + EXIF + alpha + truncated bytes,
   re-encodes as RGB JPEG, and replaces the URL with a base64 data URI
   that Pillow on the vLLM side will trivially decode.
3. Any image that fails to fetch OR fails to decode is SILENTLY DROPPED
   from the message — vLLM never sees the bad URL, so it never crashes.
   The response headers gain `X-Sanitized-Dropped: <n>` so the client
   can tell how many images survived.
4. All other endpoints (`/v1/models`, etc.) pass through unchanged.

Listens on the public port (configurable via env vars) and proxies to
vLLM on localhost. vLLM itself is firewalled to 127.0.0.1 once this
proxy is in place.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, ImageFile, UnidentifiedImageError

# Tolerate truncated JPEGs / partial reads instead of raising.
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Higher pixel ceiling — FB CDN cover photos can exceed Pillow's default.
Image.MAX_IMAGE_PIXELS = 200_000_000

VLLM_UPSTREAM = os.environ.get("VLLM_UPSTREAM", "http://127.0.0.1:8001")
LISTEN_HOST = os.environ.get("SANITIZER_HOST", "0.0.0.0")
LISTEN_PORT = int(os.environ.get("SANITIZER_PORT", "8000"))
FETCH_TIMEOUT = float(os.environ.get("SANITIZER_FETCH_TIMEOUT", "8.0"))
JPEG_QUALITY = int(os.environ.get("SANITIZER_JPEG_QUALITY", "85"))
MAX_EDGE = int(os.environ.get("SANITIZER_MAX_EDGE", "1280"))

log = logging.getLogger("vllm-sanitizer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI()
http_client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def _startup() -> None:
    global http_client
    # Two clients aren't worth the bookkeeping; one is plenty.
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(FETCH_TIMEOUT, connect=4.0),
        follow_redirects=True,
        headers={"User-Agent": "fb-reader-vllm-sanitizer/1.0"},
    )


@app.on_event("shutdown")
async def _shutdown() -> None:
    if http_client is not None:
        await http_client.aclose()


async def _fetch_and_reencode(url: str) -> str | None:
    """Return a `data:image/jpeg;base64,...` URI, or None if undecodable.

    `data:` URIs themselves are treated specially: we still re-encode them
    so SVG/GIF data URIs that vLLM can't handle get filtered out the same
    way URL-fetched bad bytes do.
    """
    try:
        if url.startswith("data:"):
            header, _, b64 = url.partition(",")
            if not b64:
                return None
            raw = base64.b64decode(b64, validate=False)
        else:
            assert http_client is not None
            resp = await http_client.get(url)
            if resp.status_code != 200:
                log.info("drop url status=%s url=%s", resp.status_code, url[:120])
                return None
            ct = resp.headers.get("content-type", "")
            if "html" in ct.lower() or "xml" in ct.lower():
                # Almost certainly a 403 error page or SVG — Pillow will choke.
                log.info("drop url ct=%s url=%s", ct, url[:120])
                return None
            raw = resp.content

        try:
            img = Image.open(io.BytesIO(raw))
            # Force a real decode (Image.open is lazy).
            img.load()
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
            log.info("drop decode-fail url=%s err=%s", url[:120], exc)
            return None

        # Animated / multi-frame: take frame 0 only. vLLM is not video-aware
        # and the multi-frame path is exactly where the original crash
        # manifested.
        if getattr(img, "is_animated", False):
            try:
                img.seek(0)
            except EOFError:
                pass

        # Drop alpha + EXIF orientation quirks by going through RGB.
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Bound the longest edge so we don't blow vLLM's image-token budget
        # on 4k FB cover photos. Maintains aspect ratio.
        w, h = img.size
        long_edge = max(w, h)
        if long_edge > MAX_EDGE:
            scale = MAX_EDGE / long_edge
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

        out = io.BytesIO()
        img.save(out, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        b64 = base64.b64encode(out.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    except Exception as exc:  # noqa: BLE001 — last-line defense, must not crash the proxy
        log.warning("drop unexpected url=%s err=%s", url[:120], exc)
        return None


async def _sanitize_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    """Walk the chat-completions payload, replace image URLs with re-encoded
    data URIs, drop ones that fail. Returns (payload, kept, dropped)."""
    kept = 0
    dropped = 0
    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        return payload, 0, 0

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        # Collect (index, url) pairs first, sanitize concurrently, then
        # rewrite. Concurrent fetches matter on bandwidth-bound GB10.
        targets: list[tuple[int, str]] = []
        for i, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") == "image_url":
                iu = part.get("image_url")
                if isinstance(iu, dict):
                    url = iu.get("url")
                elif isinstance(iu, str):
                    url = iu
                else:
                    url = None
                if isinstance(url, str):
                    targets.append((i, url))

        if not targets:
            continue

        results = await asyncio.gather(*(_fetch_and_reencode(u) for _, u in targets))

        new_content: list[Any] = []
        replacements = dict(zip([i for i, _ in targets], results))
        for i, part in enumerate(content):
            if i in replacements:
                rep = replacements[i]
                if rep is None:
                    dropped += 1
                    continue  # drop the part entirely
                kept += 1
                new_content.append({"type": "image_url", "image_url": {"url": rep}})
            else:
                new_content.append(part)
        msg["content"] = new_content

    return payload, kept, dropped


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    try:
        payload = await request.json()
    except Exception:
        body = await request.body()
        # Pass through unparseable bodies — let vLLM produce its own error.
        return await _proxy_raw(request, body)

    payload, kept, dropped = await _sanitize_payload(payload)

    assert http_client is not None
    upstream_resp = await http_client.post(
        f"{VLLM_UPSTREAM}/v1/chat/completions",
        json=payload,
        timeout=httpx.Timeout(120.0, connect=4.0),
    )
    headers = {
        "X-Sanitized-Kept": str(kept),
        "X-Sanitized-Dropped": str(dropped),
    }
    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        headers={**headers, "Content-Type": upstream_resp.headers.get("content-type", "application/json")},
    )


async def _proxy_raw(request: Request, body: bytes) -> Response:
    """Generic pass-through for non-chat endpoints (e.g. /v1/models)."""
    assert http_client is not None
    url = f"{VLLM_UPSTREAM}{request.url.path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    fwd_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }
    upstream_resp = await http_client.request(
        request.method, url, headers=fwd_headers, content=body,
    )
    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        headers={
            k: v for k, v in upstream_resp.headers.items()
            if k.lower() not in ("content-encoding", "transfer-encoding", "connection")
        },
    )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def passthrough(path: str, request: Request) -> Response:
    body = await request.body()
    return await _proxy_raw(request, body)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=LISTEN_HOST, port=LISTEN_PORT, log_level="info")
