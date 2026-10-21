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


"""Minimal GraspGen HTTP server for the steerable RoboDojo tools.

Run in GraspGen's own environment:
    ~/Codes/GraspGen/.venv-native/bin/python -u bench/robodojo_steerable/graspgen_server.py \
        --config ~/Codes/GraspGen/models/checkpoints/graspgen_franka_panda.yml --port 8796

POST /grasps  {"points": b64(np.save(Nx3 float32))}
           -> {"poses": b64(np.save(Kx4x4)), "scores": b64(np.save(K)), "elapsed_s": float}
GET  /health  -> {"status": "ok"}
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np


def _encode(a: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, np.asarray(a), allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8796)
    parser.add_argument("--num-grasps", type=int, default=200)
    parser.add_argument("--topk", type=int, default=100)
    args = parser.parse_args()

    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

    sampler = GraspGenSampler(load_grasp_cfg(args.config))
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            self._send(200 if self.path == "/health" else 404, {"status": "ok"})

        def do_POST(self):  # noqa: N802
            if self.path != "/grasps":
                self._send(404, {"error": "unknown path"})
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                points = np.load(
                    io.BytesIO(base64.b64decode(req["points"])), allow_pickle=False
                )
                if points.ndim != 2 or points.shape[1] != 3 or len(points) < 32:
                    raise ValueError(
                        f"points must be Nx3 with N>=32, got {points.shape}"
                    )
                t0 = time.time()
                with lock:
                    grasps, scores = GraspGenSampler.run_inference(
                        points.astype(np.float32),
                        sampler,
                        grasp_threshold=-1.0,
                        num_grasps=args.num_grasps,
                        topk_num_grasps=args.topk,
                    )
                poses = (
                    grasps.detach().cpu().numpy()
                    if len(grasps)
                    else np.zeros((0, 4, 4))
                )
                conf = scores.detach().cpu().numpy() if len(scores) else np.zeros((0,))
                self._send(
                    200,
                    {
                        "poses": _encode(poses),
                        "scores": _encode(conf),
                        "elapsed_s": round(time.time() - t0, 3),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                self._send(400, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, fmt, *a):
            print(f"[graspgen] {self.address_string()} {fmt % a}", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"graspgen listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
