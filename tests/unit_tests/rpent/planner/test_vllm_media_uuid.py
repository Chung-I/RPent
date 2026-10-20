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

import asyncio
import json

import httpx

from rpent.planner.vllm_media_uuid import VllmMediaUuidTransport

URL = "http://vllm.test/v1/chat/completions"
IMAGE_A = "data:image/png;base64,QUFB"
IMAGE_B = "data:image/png;base64,QkJC"


class FakeVllm:
    """Record request bodies and emulate vLLM's media cache."""

    def __init__(self) -> None:
        self.bodies: list[dict] = []
        self.cache: set[str] = set()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        for message in body["messages"]:
            for part in message["content"]:
                if part.get("type") != "image_url":
                    continue
                if part["image_url"] is None and part["uuid"] not in self.cache:
                    return httpx.Response(
                        400,
                        json={
                            "error": {
                                "message": "Cache miss for image at index 0 but data is not provided."
                            }
                        },
                    )
                self.cache.add(part["uuid"])
        return httpx.Response(200, json={"ok": True})


def _chat(*images: str) -> dict:
    content = [{"type": "image_url", "image_url": {"url": url}} for url in images]
    return {
        "messages": [
            {"role": "user", "content": [*content, {"type": "text", "text": "hi"}]}
        ]
    }


def _image_urls(body: dict) -> list:
    return [
        part["image_url"]
        for message in body["messages"]
        for part in message["content"]
        if part.get("type") == "image_url"
    ]


async def _post_all(server: FakeVllm, *bodies: dict) -> list[httpx.Response]:
    transport = VllmMediaUuidTransport(httpx.MockTransport(server))
    async with httpx.AsyncClient(transport=transport) as client:
        return [await client.post(URL, json=body) for body in bodies]


def test_history_images_are_sent_once_then_referenced():
    server = FakeVllm()
    responses = asyncio.run(_post_all(server, _chat(IMAGE_A), _chat(IMAGE_A, IMAGE_B)))

    assert [r.status_code for r in responses] == [200, 200]
    assert _image_urls(server.bodies[0]) == [{"url": IMAGE_A}]
    assert _image_urls(server.bodies[1]) == [None, {"url": IMAGE_B}]
    uuids = [part["uuid"] for part in server.bodies[1]["messages"][0]["content"][:2]]
    assert uuids[0] == server.bodies[0]["messages"][0]["content"][0]["uuid"]
    assert uuids[0] != uuids[1]


def test_cache_miss_resends_full_images_once():
    server = FakeVllm()

    async def run() -> list[httpx.Response]:
        transport = VllmMediaUuidTransport(httpx.MockTransport(server))
        async with httpx.AsyncClient(transport=transport) as client:
            first = await client.post(URL, json=_chat(IMAGE_A))
            server.cache.clear()  # server restart or LRU eviction
            second = await client.post(URL, json=_chat(IMAGE_A, IMAGE_B))
            third = await client.post(URL, json=_chat(IMAGE_A, IMAGE_B))
            return [first, second, third]

    responses = asyncio.run(run())

    assert [r.status_code for r in responses] == [200, 200, 200]
    assert _image_urls(server.bodies[1]) == [None, {"url": IMAGE_B}]
    assert _image_urls(server.bodies[2]) == [{"url": IMAGE_A}, {"url": IMAGE_B}]
    assert _image_urls(server.bodies[3]) == [None, None]


def test_other_errors_are_returned_unchanged():
    def reject(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad tool schema"}})

    async def run() -> httpx.Response:
        transport = VllmMediaUuidTransport(httpx.MockTransport(reject))
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post(URL, json=_chat(IMAGE_A))
            return await client.post(URL, json=_chat(IMAGE_A))

    response = asyncio.run(run())

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "bad tool schema"


def test_log_records_image_counts_and_token_usage(tmp_path):
    log_path = tmp_path / "media.jsonl"

    def reply(request: httpx.Request) -> httpx.Response:
        usage = {
            "prompt_tokens": 2100,
            "completion_tokens": 7,
            "prompt_tokens_details": {"cached_tokens": 1664},
        }
        return httpx.Response(200, json={"usage": usage})

    async def run() -> list[httpx.Response]:
        transport = VllmMediaUuidTransport(httpx.MockTransport(reply), str(log_path))
        async with httpx.AsyncClient(transport=transport) as client:
            return [
                await client.post(URL, json=body)
                for body in (_chat(IMAGE_A), _chat(IMAGE_A, IMAGE_B))
            ]

    responses = asyncio.run(run())
    records = [json.loads(line) for line in log_path.read_text().splitlines()]

    assert responses[1].json()["usage"]["prompt_tokens"] == 2100
    assert [(r["images"], r["referenced"]) for r in records] == [(1, 0), (2, 1)]
    assert records[1]["cached_tokens"] == 1664 and records[1]["status"] == 200


def test_openai_chat_model_uses_the_transport_when_enabled(monkeypatch):
    from rpent.planner.base import build_api_model

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("RPENT_VLLM_MEDIA_UUIDS", "1")
    model = build_api_model("openai-chat:Qwen/Qwen3.8-27B-FP8", "http://vllm.test/v1")

    assert isinstance(model.client._client._transport, VllmMediaUuidTransport)


def test_log_reads_usage_from_the_final_stream_chunk(tmp_path):
    log_path = tmp_path / "media.jsonl"
    usage = {
        "prompt_tokens": 900,
        "completion_tokens": 3,
        "prompt_tokens_details": {"cached_tokens": 832},
    }
    events = [
        b'data: {"choices": [{"delta": {"content": "ok"}}], "usage": null}\n\n',
        b"data: " + json.dumps({"choices": [], "usage": usage}).encode() + b"\n\n",
        b"data: [DONE]\n\n",
    ]

    def reply(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=b"".join(events)
        )

    async def run() -> bytes:
        transport = VllmMediaUuidTransport(httpx.MockTransport(reply), str(log_path))
        async with httpx.AsyncClient(transport=transport) as client:
            async with client.stream("POST", URL, json=_chat(IMAGE_A)) as response:
                return b"".join([chunk async for chunk in response.aiter_bytes()])

    body = asyncio.run(run())
    (record,) = [json.loads(line) for line in log_path.read_text().splitlines()]

    assert body == b"".join(events)
    assert (record["prompt_tokens"], record["cached_tokens"]) == (900, 832)
