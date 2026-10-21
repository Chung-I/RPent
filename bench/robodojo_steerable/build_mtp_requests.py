"""Build replay chat requests for mtp_bench.py from a steerable RoboDojo transcript.

Each request is a prefix of the recorded conversation that ends just before one
planner turn, with the steerable system prompt, the planner tool schemas and the
run's three camera frames, so the benchmark decodes the same kind of reasoning
and tool calls as a live episode.

Usage (local RPent venv, RPENT_ROBODOJO_STEERABLE=1):
    python bench/robodojo_steerable/build_mtp_requests.py RUN_DIR OUT.json [--count 8]
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from robots.robodojo import steerable
from robots.robodojo import tools as robodojo_tools
from robots.robodojo.prompt_bundle import system_prompt
from rpent.prompt.utils import format_prompt
from rpent.tools import iter_tools
from rpent.tools.common import CommonTools


def openai_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in iter_tools(CommonTools, robodojo_tools, steerable)
    ]


def convert(messages: list[dict], images: list[str]) -> list[dict]:
    """Transcript messages -> OpenAI chat messages; images follow the first tool result."""
    out, placed = [], False
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            text = "".join(b["text"] for b in m["content"] if b["type"] == "text")
            calls = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {"name": b["name"], "arguments": json.dumps(b["input"])},
                }
                for b in m["content"]
                if b["type"] == "tool_use"
            ]
            msg = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:
            out.append(
                {"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]}
            )
            if not placed:
                parts = [{"type": "text", "text": "Camera images (head, left wrist, right wrist):"}]
                parts += [{"type": "image_url", "image_url": {"url": u}} for u in images]
                out.append({"role": "user", "content": parts})
                placed = True
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--count", type=int, default=8)
    args = p.parse_args()
    transcript = json.loads(next(args.run_dir.glob("transcript_*.json")).read_text())
    messages = transcript["messages"]
    images = [
        "data:image/png;base64,"
        + base64.b64encode((args.run_dir / f"cam_{c}.png" / "00.png").read_bytes()).decode()
        for c in ("head", "left_wrist", "right_wrist")
    ]
    system = format_prompt(
        system_prompt({}), variables={"output_dir": "/tmp/rpent_out"}
    )
    turns = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    step = max(1, len(turns) // args.count)
    cuts = turns[::step][: args.count]
    requests = [
        {
            "turn": turns.index(cut),
            "messages": [{"role": "system", "content": system}]
            + convert(messages[:cut], images),
        }
        for cut in cuts
    ]
    args.out.write_text(json.dumps({"tools": openai_tools(), "requests": requests}))
    print(f"{len(requests)} requests at turns {[r['turn'] for r in requests]} -> {args.out}")


if __name__ == "__main__":
    main()
