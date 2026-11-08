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

"""The GraspGen menu pilot's request step: each case's cloud (curve_graspgen clouds) to the GraspGenX server, once.

It runs in GraspGenX's own venv (its zmq client; numpy only from this repo's side) and imports nothing else of RPent:
python robots/mujoco_lab/graspgen_request.py --clouds DIR --out DIR. One request per case at the franka_panda gripper,
num_grasps 200, grasp_threshold 0.0 and topk_num_grasps -1 (every generated grasp with a score at or above 0). The
server does not seed its sampler, so the grasp files are the record: the run step reads them and never asks again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np

GRIPPER, NUM_GRASPS, THRESHOLD, TOPK = "franka_panda", 200, 0.0, -1


def sha_arrays(grasps: Any, conf: Any) -> str:
    return hashlib.sha256(np.asarray(grasps, dtype=np.float32).tobytes() + np.asarray(conf, dtype=np.float32).tobytes()).hexdigest()


def request_all(clouds: str, out: str, infer: Callable[[np.ndarray], tuple[Any, Any, float]], meta: dict[str, Any]) -> dict[str, Any]:
    """One request per line of ``clouds``/index.jsonl, in order: the cloud file's hash must be its line's (else the line's
    error), then ``infer(cloud)`` -> (grasps (K, 4, 4), confidences (K,), the server's milliseconds); an exception is the
    line's error. Writes ``out``/<object>__<idx>.npz, ``out``/index.jsonl, run_config.json (``meta``) and run_metrics.json;
    refuses an ``out`` that has an index. Returns the metrics."""
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "index.jsonl")):
        raise SystemExit(f"{out}: an index exists: pick a new --out")
    lines = [json.loads(x) for x in open(os.path.join(clouds, "index.jsonl"))]
    with open(os.path.join(out, "run_config.json"), "w") as f:
        json.dump(dict(meta, clouds=clouds, cases=len(lines), gripper=GRIPPER, num_grasps=NUM_GRASPS, grasp_threshold=THRESHOLD,
                       topk_num_grasps=TOPK), f, indent=1)
    t0, errors = time.time(), 0
    with open(os.path.join(out, "index.jsonl"), "w") as idx:
        for line in lines:
            rec = {k: line.get(k) for k in ("object", "idx", "side", "fold", "cloud_sha256")}
            rec.update(file=None, grasps_sha256=None, k=None, infer_ms=None, error=None)
            t1 = time.time()
            try:
                cloud = np.load(os.path.join(clouds, line["file"]))["cloud"]
                if hashlib.sha256(cloud.tobytes()).hexdigest() != line["cloud_sha256"]:
                    raise ValueError("the cloud file's hash differs from its index line")
                grasps, conf, ms = infer(cloud)
                grasps, conf = np.asarray(grasps, dtype=np.float32), np.asarray(conf, dtype=np.float32)
                name = f"{line['object']}__{int(line['idx'])}.npz"
                np.savez(os.path.join(out, name), grasps=grasps, conf=conf)
                rec.update(file=name, grasps_sha256=sha_arrays(grasps, conf), k=int(len(conf)), infer_ms=float(ms))
            except Exception as exc:  # noqa: BLE001  a failed request is this case's error
                rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                errors += 1
            rec["seconds"] = round(time.time() - t1, 3)
            idx.write(json.dumps(rec) + "\n")
            idx.flush()
            print(f"{rec['object']} {rec['idx']}: {rec['k']} grasps" + (f", error {rec['error']}" if rec["error"] else ""),
                  file=sys.stderr, flush=True)
    metrics = {"cases": len(lines), "errors": errors, "wall_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(out, "run_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    return metrics


def main(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--clouds", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--graspgenx", default=os.path.expanduser("~/Codes/GraspGenX"), help="the GraspGenX checkout (for its commit)")
    a = p.parse_args(argv)
    from graspgenx.serving.zmq_client import GraspGenXClient

    client = GraspGenXClient(host=a.host, port=a.port)
    rev = subprocess.run(["git", "-C", a.graspgenx, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", a.graspgenx, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.count("\n")

    def infer(cloud: np.ndarray) -> tuple[Any, Any, float]:
        t0 = time.time()
        g, s = client.infer(cloud, gripper_name=GRIPPER, num_grasps=NUM_GRASPS, grasp_threshold=THRESHOLD, topk_num_grasps=TOPK)
        return g, s, (time.time() - t0) * 1000.0

    m = request_all(a.clouds, a.out, infer, {"graspgenx_commit": rev, "graspgenx_dirty_files": dirty, "host": a.host, "port": a.port})
    print(f"done: {json.dumps(m)}", flush=True)
    return m


if __name__ == "__main__":
    main(sys.argv[1:])
