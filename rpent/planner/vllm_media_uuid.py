# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Send each chat image to a vLLM server once, then refer to it by UUID.

vLLM keeps processed images in its multimodal processor cache, keyed by the
``uuid`` field of an ``image_url`` content part. A later part that carries the
same ``uuid`` with ``image_url: null`` reuses the cached item, so the history
images of a long agent conversation cross the network once instead of on
every turn. The planner's message history is not changed; only the request
body sent on the wire is rewritten.

A server restart or a cache eviction makes vLLM reject a reference with
``Cache miss ... data is not provided``; the transport then forgets every
sent image and repeats the request once with full image data.

This uses a vLLM-specific request field, so it is enabled only for the
``openai`` provider with a custom base URL and ``RPENT_VLLM_MEDIA_UUIDS=1``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

import httpx

from rpent.utils.logging import get_logger

logger = get_logger("api")

_CACHE_MISS_MARKER = b"Cache miss"


class VllmMediaUuidTransport(httpx.AsyncBaseTransport):
    """Rewrite chat-completion image parts into vLLM UUID references.

    Args:
        inner: Transport that performs the HTTP exchange.
    """

    def __init__(
        self, inner: httpx.AsyncBaseTransport, log_path: str | None = None
    ) -> None:
        self._inner = inner
        self._sent: set[str] = set()
        self._log_path = log_path

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method != "POST" or not request.url.path.endswith(
            "/chat/completions"
        ):
            return await self._inner.handle_async_request(request)
        started = time.monotonic()
        body = json.loads(await request.aread())
        uuids = _tag_images(body)
        record = {"images": len(uuids), "referenced": 0, "cache_miss_retry": False}
        if not uuids:
            record["request_bytes"] = len(request.content)
            response = await self._inner.handle_async_request(request)
        else:
            response = await self._exchange(request, body, uuids, record)
        if self._log_path is None:
            return response
        return await self._log(response, request, record, started)

    async def _exchange(
        self,
        request: httpx.Request,
        body: dict[str, Any],
        uuids: list[str],
        record: dict[str, Any],
    ) -> httpx.Response:
        full = json.dumps(body).encode()
        referenced = json.dumps(_reference_sent(body, self._sent)).encode()
        record["referenced"] = sum(uuid in self._sent for uuid in uuids)
        record["request_bytes"] = len(referenced)
        response = await self._inner.handle_async_request(
            _with_body(request, referenced)
        )
        if response.status_code == 400:
            content = await response.aread()
            await response.aclose()
            if _CACHE_MISS_MARKER not in content:
                return _rebuild(response, content, request)
            logger.warning(
                "vLLM media cache miss; resending %d image(s) in full", len(uuids)
            )
            self._sent.clear()
            record.update(cache_miss_retry=True, request_bytes=len(full))
            response = await self._inner.handle_async_request(_with_body(request, full))
        if response.is_success:
            self._sent.update(uuids)
        return response

    async def _log(
        self,
        response: httpx.Response,
        request: httpx.Request,
        record: dict[str, Any],
        started: float,
    ) -> httpx.Response:
        """Append one JSONL telemetry record once the response body ends.

        Token usage comes from a JSON body, or from the final ``usage`` chunk
        of a server-sent event stream.
        """
        record["status"] = response.status_code

        def write(usage: dict[str, Any] | None) -> None:
            details = (usage or {}).get("prompt_tokens_details") or {}
            record.update(
                prompt_tokens=(usage or {}).get("prompt_tokens"),
                cached_tokens=details.get("cached_tokens"),
                completion_tokens=(usage or {}).get("completion_tokens"),
                ts=time.time(),
                elapsed_s=round(time.monotonic() - started, 3),
            )
            with open(self._log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

        if "text/event-stream" in response.headers.get("content-type", ""):
            return httpx.Response(
                response.status_code,
                headers=response.headers,
                stream=_UsageStream(response.stream, write),
                request=request,
                extensions=response.extensions,
            )
        content = await response.aread()
        await response.aclose()
        try:
            usage = json.loads(content).get("usage")
        except (ValueError, AttributeError):
            usage = None
        write(usage)
        return _rebuild(response, content, request)

    async def aclose(self) -> None:
        await self._inner.aclose()


class _UsageStream(httpx.AsyncByteStream):
    """Pass a server-sent event stream through and report its last ``usage``."""

    def __init__(self, inner: Any, on_done: Any) -> None:
        self._inner = inner
        self._on_done = on_done
        self._usage: dict[str, Any] | None = None
        self._pending = b""
        self._done = False

    async def __aiter__(self):
        async for chunk in self._inner:
            self._scan(chunk)
            yield chunk
        self._finish()

    async def aclose(self) -> None:
        await self._inner.aclose()
        self._finish()

    def _scan(self, chunk: bytes) -> None:
        *lines, self._pending = (self._pending + chunk).split(b"\n")
        for line in lines:
            if line.startswith(b"data: {") and b'"usage"' in line:
                usage = json.loads(line[len(b"data: ") :]).get("usage")
                if usage:
                    self._usage = usage

    def _finish(self) -> None:
        if not self._done:
            self._done = True
            self._on_done(self._usage)


def _image_parts(body: dict[str, Any]):
    """Yield every ``image_url`` content part that carries inline data."""
    for message in body.get("messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "image_url"
                and isinstance(part.get("image_url"), dict)
                and str(part["image_url"].get("url", "")).startswith("data:")
            ):
                yield part


def _tag_images(body: dict[str, Any]) -> list[str]:
    """Add a content-hash ``uuid`` to each inline image and return the UUIDs."""
    uuids = []
    for part in _image_parts(body):
        digest = hashlib.sha256(part["image_url"]["url"].encode()).hexdigest()
        part["uuid"] = f"sha256:{digest}"
        uuids.append(part["uuid"])
    return uuids


def _reference_sent(body: dict[str, Any], sent: set[str]) -> dict[str, Any]:
    """Return a copy of ``body`` whose already-sent images carry no data."""
    copy = json.loads(json.dumps(body))
    for part in _image_parts(copy):
        if part["uuid"] in sent:
            part["image_url"] = None
    return copy


def _with_body(request: httpx.Request, content: bytes) -> httpx.Request:
    headers = httpx.Headers(request.headers)
    headers["content-length"] = str(len(content))
    return httpx.Request(
        request.method,
        request.url,
        headers=headers,
        content=content,
        extensions=request.extensions,
    )


def _rebuild(
    response: httpx.Response, content: bytes, request: httpx.Request
) -> httpx.Response:
    """Return an already-read response with its decoded body."""
    headers = httpx.Headers(response.headers)
    headers.pop("content-encoding", None)
    headers["content-length"] = str(len(content))
    return httpx.Response(
        response.status_code,
        headers=headers,
        content=content,
        request=request,
        extensions=response.extensions,
    )


def build_http_client() -> httpx.AsyncClient:
    """Build an OpenAI HTTP client that sends each image once.

    ``RPENT_VLLM_MEDIA_LOG`` names an optional JSONL file that receives one
    record per chat request: image counts, request bytes, token usage, latency.
    """
    return httpx.AsyncClient(
        transport=VllmMediaUuidTransport(
            httpx.AsyncHTTPTransport(), os.environ.get("RPENT_VLLM_MEDIA_LOG")
        ),
        timeout=httpx.Timeout(600, connect=5),
    )
