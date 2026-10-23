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

"""Finish an exploration corpus with RPent's own merge, after a full backup.

0. Rewrite list-valued ``evidence.attempts`` (attempt numbers such as
   ``[1, 2]``) in published notes and pending inbox drafts to their count,
   because merging into such a note fails with a ``TypeError``. The original
   list is kept under ``evidence.source_attempts``.
1. Publish cells whose environment reported success but whose planner timed
   out before the automatic merge (``rpent-memory merge --solved``, the path
   the memory guide documents for verified success).
2. Re-merge global drafts rejected only because ``evidence`` was not a mapping:
   rewrite that one field to ``{cells: [<cell>], attempts: <n>}`` and keep the
   original value under ``source_evidence``. No other field or text changes.
Every action is written to ``<memory-dir>/_internal/bench_memory_fix.json``.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import yaml

EVIDENCE_ERROR = "evidence must be a mapping"


def merge(
    rpent_memory: str, memory: Path, cell: str, run_dir: Path, solved: bool
) -> dict:
    cmd = [
        rpent_memory,
        "--memory-dir",
        str(memory),
        "merge",
        "--cell",
        cell,
        "--output-dir",
        str(run_dir),
    ]
    out = subprocess.run(
        cmd + (["--solved"] if solved else []),
        capture_output=True,
        text=True,
    )
    if out.returncode:
        raise SystemExit(f"merge {cell} failed:\n{out.stderr}")
    return json.loads(out.stdout)


def automatic_merges(runs: Path) -> dict[str, tuple[Path, dict]]:
    """Return {cell: (run_dir, merge result)} parsed from each run's log."""
    found = {}
    for run_dir in sorted(p for p in runs.iterdir() if (p / "stdout.log").exists()):
        match = re.search(
            r"memory merged: (\{.*\})", (run_dir / "stdout.log").read_text()
        )
        if match:
            result = ast.literal_eval(match.group(1))
            found[result["cell"]] = (run_dir, result)
    return found


def repair_evidence(path: Path, cell: str) -> str:
    text = path.read_text()
    _, front, body = text.split("---\n", 2)
    meta = yaml.safe_load(front)
    original = meta["evidence"]
    attempts = len(re.findall(r"\bA\d+\b", str(original))) or 1
    meta["evidence"] = {
        "cells": [cell],
        "attempts": attempts,
        "source_evidence": original if isinstance(original, list) else [original],
    }
    return (
        "---\n"
        + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)
        + "---\n"
        + body
    )


def repair_attempts(memory: Path) -> list[dict]:
    """Count list-valued ``evidence.attempts`` in notes a merge reads."""
    paths = [
        *(
            p
            for tier in ("global", "task-family", "task-specific")
            for p in sorted((memory / tier).glob("*.md"))
        ),
        *sorted((memory / "_internal" / "inbox").glob("*/*.md")),
    ]
    repairs = []
    for path in paths:
        text = path.read_text()
        if not text.startswith("---\n"):
            continue
        _, front, body = text.split("---\n", 2)
        try:
            meta = yaml.safe_load(front)
        except yaml.YAMLError:
            continue  # rpent-memory reports these itself
        evidence = meta.get("evidence") if isinstance(meta, dict) else None
        if not isinstance(evidence, dict) or not isinstance(
            evidence.get("attempts"), list
        ):
            continue
        original = evidence["attempts"]
        evidence["attempts"] = len(original) or 1
        evidence["source_attempts"] = original
        path.write_text(
            "---\n"
            + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)
            + "---\n"
            + body
        )
        repairs.append(
            {
                "path": str(path.relative_to(memory)),
                "from": original,
                "to": evidence["attempts"],
            }
        )
    return repairs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--memory-dir", type=Path, required=True)
    p.add_argument("--runs-dir", type=Path, required=True)
    p.add_argument("--rpent-memory", required=True)
    args = p.parse_args()
    memory, runs = args.memory_dir, args.runs_dir
    log: dict = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "solved_merges": [],
        "evidence_repairs": [],
    }

    backup = memory.with_name(memory.name + ".before_bench_fix")
    if not backup.exists():
        shutil.copytree(memory, backup)
    log["backup"] = str(backup)
    log["attempts_repairs"] = repair_attempts(memory)

    results = automatic_merges(runs)
    for run_dir in sorted(runs.iterdir()):
        result_file = run_dir / "result.json"
        if (
            not result_file.exists()
            or json.loads(result_file.read_text())["outcome"] != "success"
        ):
            continue
        cell = run_dir.name.removeprefix("libero_")
        if (memory / "task-specific" / f"{cell}_recipe.jsonl").exists():
            continue
        result = merge(args.rpent_memory, memory, cell, run_dir, solved=True)
        results[cell] = (run_dir, result)
        log["solved_merges"].append(result)

    for cell, (run_dir, result) in sorted(results.items()):
        names = [
            s.split(": ", 1)[0]
            for s in result.get("skipped", [])
            if s.endswith(EVIDENCE_ERROR)
        ]
        if not names:
            continue
        archive = memory / "_internal" / "merged" / cell
        keep = memory / "_internal" / "merged" / f"{cell}__first_merge"
        if not keep.exists():
            shutil.copytree(archive, keep)
        inbox = memory / "_internal" / "inbox" / cell
        inbox.mkdir(parents=True, exist_ok=False)
        for name in names:
            (inbox / name).write_text(repair_evidence(keep / name, cell))
        log["evidence_repairs"].append(
            {
                "cell": cell,
                "drafts": names,
                "merge": merge(args.rpent_memory, memory, cell, run_dir, solved=False),
            }
        )

    for command in ("validate", "build-index"):
        out = subprocess.run(
            [args.rpent_memory, "--memory-dir", str(memory), command],
            capture_output=True,
            text=True,
        )
        log[command] = out.stdout.strip().splitlines()[:20]
    (memory / "_internal" / "bench_memory_fix.json").write_text(
        json.dumps(log, indent=1, default=str)
    )
    counts = {
        d: len(list((memory / d).glob("*")))
        for d in ("task-specific", "task-family", "global")
    }
    repaired = sum(len(r["drafts"]) for r in log["evidence_repairs"])
    still = sum(len(r["merge"].get("skipped", [])) for r in log["evidence_repairs"])
    print(
        json.dumps(
            {
                "attempts_repaired": len(log["attempts_repairs"]),
                "solved_merges": len(log["solved_merges"]),
                "drafts_repaired": repaired,
                "still_skipped_after_repair": still,
                "counts": counts,
                "validate": log["validate"][:3],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
