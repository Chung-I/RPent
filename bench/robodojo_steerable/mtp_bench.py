"""Replay planner requests against one vLLM server and record decode speed.

Each request runs twice: once with the server's default sampling (as the planner
sends it) and once greedy, for an output comparison across server configs.
Speculative-decoding counters come from the server's /metrics before and after.

Usage: python mtp_bench.py URL REQUESTS.json OUT.json --label NAME [--max-tokens 4096]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request


def post_stream(url: str, body: dict, key: str) -> dict:
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    start = time.monotonic()
    first = None
    text, reasoning, calls, usage = [], [], [], None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                piece = (
                    (delta.get("content") or "")
                    + (delta.get("reasoning") or delta.get("reasoning_content") or "")
                    + json.dumps(delta.get("tool_calls") or "")
                )
                if piece.strip('"') and first is None:
                    first = time.monotonic()
                text.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning") or delta.get("reasoning_content") or "")
                if delta.get("tool_calls"):
                    calls.append(delta["tool_calls"])
    end = time.monotonic()
    tokens = usage["completion_tokens"]
    return {
        "ttft_s": round(first - start, 3),
        "total_s": round(end - start, 3),
        "completion_tokens": tokens,
        "prompt_tokens": usage["prompt_tokens"],
        "decode_tok_s": round((tokens - 1) / (end - first), 2) if tokens > 1 else None,
        "output": "".join(reasoning) + "\n<content>\n" + "".join(text) + json.dumps(calls),
    }


def spec_counters(url: str) -> dict:
    body = urllib.request.urlopen(url + "/metrics", timeout=30).read().decode()
    out = {}
    for name in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
        m = re.findall(rf"^vllm:spec_decode_{name}_total{{[^}}]*}} ([0-9.e+]+)$", body, re.M)
        out[name] = sum(float(v) for v in m)
    out["raw"] = [l for l in body.splitlines() if "spec_decode" in l and not l.startswith("#")]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("requests")
    p.add_argument("out")
    p.add_argument("--label", required=True)
    p.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    p.add_argument("--max-tokens", type=int, default=4096)
    args = p.parse_args()
    key = os.environ["VLLM_API_KEY"]
    data = json.load(open(args.requests))

    def body(req: dict, greedy: bool) -> dict:
        b = {
            "model": args.model,
            "messages": req["messages"],
            "tools": data["tools"],
            "max_tokens": args.max_tokens,
            "reasoning_effort": "medium",
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if greedy:
            b["temperature"] = 0.0
        return b

    post_stream(args.url, body(data["requests"][-1], True) | {"max_tokens": 16}, key)  # warm-up
    before = spec_counters(args.url)
    rows = []
    for req in data["requests"]:
        for mode in ("default", "greedy"):
            r = post_stream(args.url, body(req, mode == "greedy"), key)
            r.update(turn=req["turn"], mode=mode)
            rows.append(r)
            print(args.label, mode, req["turn"], {k: r[k] for k in ("ttft_s", "total_s", "completion_tokens", "decode_tok_s")}, flush=True)
    after = spec_counters(args.url)
    spec = {k: after[k] - before[k] for k in after if k != "raw"}
    spec["raw_after"] = after["raw"]
    if spec["num_drafts"]:
        spec["mean_acceptance_length"] = round(1 + spec["num_accepted_tokens"] / spec["num_drafts"], 3)
        spec["acceptance_rate"] = round(spec["num_accepted_tokens"] / spec["num_draft_tokens"], 3)
    summary = {}
    for mode in ("default", "greedy"):
        sel = [r for r in rows if r["mode"] == mode]
        toks = sum(r["completion_tokens"] for r in sel)
        decode = sum(r["total_s"] - r["ttft_s"] for r in sel)
        summary[mode] = {
            "completion_tokens": toks,
            "total_s": round(sum(r["total_s"] for r in sel), 2),
            "decode_tok_s": round(toks / decode, 2),
        }
    print(args.label, "summary", summary, "spec", {k: v for k, v in spec.items() if k != "raw_after"}, flush=True)
    json.dump({"label": args.label, "summary": summary, "spec": spec, "rows": rows}, open(args.out, "w"))


if __name__ == "__main__":
    main()
