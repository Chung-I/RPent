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

"""Run LIBERO-Pro cells from a shared on-disk queue and log results to W&B.

Each slot claims one ``suite:task:seed`` cell with an atomic ``mkdir`` under
``<out-root>/claims``, runs ``rpent`` for it, classifies the outcome, and
writes ``<out-root>/<cell>/result.json``. Cells with a result are skipped, so
restarting a job resumes the run. Infrastructure failures are retried once;
task failures (planner finish, turn budget, timeout, truncation) are final.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

SUITES = [
    f"libero_{family}_{kind}"
    for family in ("spatial", "object", "goal", "10")
    for kind in ("task", "swap")
]
MAX_ATTEMPTS = 2


class ConfigError(RuntimeError):
    """The server rejected a request as invalid; every cell would fail."""


def all_cells() -> list[str]:
    # Seed-major order spreads early results over every suite and task.
    return [
        f"{suite}:{task}:{seed}"
        for seed in range(10)
        for suite in SUITES
        for task in range(10)
    ]


def cell_dir(root: Path, cell: str) -> Path:
    suite, task, seed = cell.split(":")
    return root / f"{suite}_t{task}_s{seed}"


def running_job_ids() -> set[str] | None:
    """Return the caller's running Slurm job ids, or None without Slurm."""
    try:
        out = subprocess.run(
            ["squeue", "--me", "-h", "-o", "%A"],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        return None
    return set(out.stdout.split())


def reclaim_dead_claims(root: Path) -> list[str]:
    """Release claims held by jobs that no longer run and left no result."""
    alive = running_job_ids()
    if alive is None:
        return []
    released = []
    for claim in (root / "claims").glob("*"):
        owner = (
            (claim / "owner").read_text().strip() if (claim / "owner").exists() else ""
        )
        cell = claim.name.replace("__", ":")
        if owner not in alive and not (cell_dir(root, cell) / "result.json").exists():
            shutil.rmtree(claim)
            released.append(cell)
    return released


def classify(out: Path) -> dict:
    """Return the outcome of one finished rpent process from its artifacts."""
    transcripts = list(out.glob("transcript_*.json"))
    states = out / "states.json"
    if not transcripts or not states.exists():
        return {
            "outcome": "infra_failure",
            "error": "missing transcript or states.json",
        }
    transcript = json.loads(transcripts[0].read_text())
    steps = json.loads(states.read_text()).get("steps") or []
    env_terminated = bool(steps and steps[-1].get("terminated"))
    error = transcript.get("error")
    record = {
        "error": error,
        "finish_status": (transcript.get("finish") or {}).get("status"),
        "elapsed_s": transcript.get("elapsed_s"),
        "env_steps_recorded": len(steps),
    }
    if transcript.get("environment_success") or env_terminated:
        record["outcome"] = "success"
    elif error is None or "timed out" in error or "response truncated" in error:
        record["outcome"] = "task_failure"
    elif "status_code: 400" in error:
        record["outcome"] = "config_error"
    else:
        record["outcome"] = "infra_failure"
    return record


def prune_world_maps(out: Path) -> None:
    """Delete the 1024x1024 pixel-to-world maps once the cell is scored.

    They are about 70% of an episode's disk use and are not needed for
    scoring; the low-resolution maps, depth, and images stay.
    """
    for path in out.glob("*_world_high.npz"):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def request_stats(out: Path) -> dict:
    path = out / "llm_requests.jsonl"
    rows = (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )
    usage = [r for r in rows if r.get("prompt_tokens")]
    prompt = sum(r["prompt_tokens"] for r in usage)
    return {
        "llm_requests": len(rows),
        "max_images": max((r["images"] for r in rows), default=0),
        "request_mb": round(sum(r["request_bytes"] for r in rows) / 1e6, 2),
        "prompt_tokens": prompt,
        "cached_tokens": sum(r["cached_tokens"] or 0 for r in usage),
        "completion_tokens": sum(r["completion_tokens"] or 0 for r in usage),
        "max_prompt_tokens": max((r["prompt_tokens"] for r in usage), default=0),
        "llm_seconds": round(sum(r["elapsed_s"] for r in usage), 1),
        "cache_miss_retries": sum(r["cache_miss_retry"] for r in rows),
    }


class Worker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = Path(args.out_root)
        (self.root / "claims").mkdir(parents=True, exist_ok=True)
        (self.root / "_infra").mkdir(exist_ok=True)
        self.job_id = os.environ.get("SLURM_JOB_ID", "local")
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.run = None
        if args.wandb_project:
            import wandb

            self.run = wandb.init(
                project=args.wandb_project,
                name=f"{args.tag}-job{self.job_id}",
                group=args.tag,
                job_type="eval-worker",
                config={k: v for k, v in vars(args).items() if k != "api_key_file"},
            )

    def cells(self) -> list[str]:
        cells = [
            c
            for c in all_cells()
            if not self.args.suites or c.split(":")[0] in self.args.suites
        ]
        return cells[: self.args.limit] if self.args.limit else cells

    def claim(self) -> str | None:
        for cell in self.cells():
            if (cell_dir(self.root, cell) / "result.json").exists():
                continue
            claim = self.root / "claims" / cell.replace(":", "__")
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            (claim / "owner").write_text(self.job_id)
            return cell
        return None

    def run_cell(self, cell: str, slot: int) -> dict:
        """Run a cell, retrying one infrastructure failure, and store its result."""
        out = cell_dir(self.root, cell)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            record = self.attempt_cell(cell, slot, attempt)
            if record["outcome"] != "infra_failure" or attempt == MAX_ATTEMPTS:
                break
            failed = self.root / "_infra" / f"{out.name}_attempt{attempt}"
            shutil.move(str(out), str(failed))
            (failed / "result.json").write_text(json.dumps(record, indent=1))
            prune_world_maps(failed)
        (out / "result.json").write_text(json.dumps(record, indent=1))
        prune_world_maps(out)
        return record

    def attempt_cell(self, cell: str, slot: int, attempt: int) -> dict:
        suite, task, seed = cell.split(":")
        out = cell_dir(self.root, cell)
        out.mkdir(parents=True, exist_ok=True)
        env = dict(
            os.environ,
            OPENAI_API_KEY=Path(self.args.api_key_file).read_text().strip(),
            RPENT_VLLM_MEDIA_UUIDS="1",
            RPENT_VLLM_MEDIA_LOG=str(out / "llm_requests.jsonl"),
            LIBERO_TYPE="pro",
            MUJOCO_GL="egl",
            HF_HUB_OFFLINE="1",
            PYDANTIC_AI_NO_BANNER="1",
        )
        cmd = [
            self.args.rpent,
            "--robot",
            "libero",
            "--suite",
            suite,
            "--task",
            task,
            "--seed",
            seed,
            "--libero-type",
            "pro",
            "--planner",
            "api",
            "--model",
            self.args.model,
            "--base-url",
            self.args.llm_url,
            "--reasoning-effort",
            self.args.reasoning_effort,
            "--max-turns",
            "100",
            "--planner-timeout-s",
            "5000",
            "--max-episode-steps",
            "10000",
            "--memory-version",
            self.args.memory_version,
            "--vla-endpoint",
            self.args.vla_url,
            "--sam3-endpoint",
            self.args.sam3_url,
            "--cuda-device",
            str(self.args.env_cuda_device),
            "--output-dir",
            str(out),
        ]
        started = time.time()
        with open(out / "stdout.log", "w") as log:
            proc = subprocess.run(
                cmd, cwd=self.args.repo, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        record = {
            "cell": cell,
            "suite": suite,
            "task": int(task),
            "seed": int(seed),
            "attempt": attempt,
            "job_id": self.job_id,
            "slot": slot,
            "exit_code": proc.returncode,
            "wall_s": round(time.time() - started, 1),
            "commit": subprocess.run(
                ["git", "-C", self.args.repo, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
            ).stdout.strip(),
            **classify(out),
            **request_stats(out),
        }
        return record

    def slot_loop(self, slot: int) -> None:
        while not self.stop.is_set():
            cell = self.claim()
            if cell is None:
                return
            record = self.run_cell(cell, slot)
            print(
                json.dumps(
                    {
                        k: record[k]
                        for k in (
                            "cell",
                            "outcome",
                            "attempt",
                            "wall_s",
                            "llm_requests",
                        )
                    }
                ),
                flush=True,
            )
            if self.run is not None:
                with self.lock:
                    self.run.log(
                        {
                            k: v
                            for k, v in record.items()
                            if isinstance(v, (int, float, bool))
                        }
                        | {"success": float(record["outcome"] == "success")}
                    )
            if record["outcome"] == "config_error":
                print(
                    f"CONFIG ERROR in {cell}: {record['error']}; stopping this worker",
                    flush=True,
                )
                self.stop.set()

    def main(self) -> None:
        released = reclaim_dead_claims(self.root)
        if released:
            print(f"released {len(released)} stale claims: {released}", flush=True)
        threads = []
        for slot in range(self.args.slots):
            thread = threading.Thread(target=self.slot_loop, args=(slot,))
            thread.start()
            threads.append(thread)
            time.sleep(self.args.stagger_s)
        for thread in threads:
            thread.join()
        if self.run is not None:
            self.run.finish()
        if self.stop.is_set():
            raise ConfigError("a cell hit a server configuration error")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", required=True)
    p.add_argument("--slots", type=int, default=12)
    p.add_argument("--stagger-s", type=float, default=20.0)
    p.add_argument("--repo", required=True)
    p.add_argument("--rpent", required=True)
    p.add_argument("--model", default="openai-chat:Qwen/Qwen3.8-27B-FP8")
    p.add_argument("--reasoning-effort", default="xhigh")
    p.add_argument("--memory-version", default="GPT_5.5_xhigh")
    p.add_argument("--llm-url", required=True)
    p.add_argument("--vla-url", required=True)
    p.add_argument("--sam3-url", required=True)
    p.add_argument("--env-cuda-device", type=int, default=0)
    p.add_argument("--api-key-file", required=True)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--tag", default="qwen38-xhigh")
    p.add_argument("--suites", nargs="*", default=None, help="only these suites")
    p.add_argument("--limit", type=int, default=None, help="only the first N cells")
    return p.parse_args()


if __name__ == "__main__":
    Worker(parse_args()).main()
