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


"""Probe corpus: the push and weigh probes under drawn hidden properties.

    python -P -m robots.mujoco_lab.probe_corpus --out PATH --draws 24 --workers 16 --seed 7 [--objects a,b] [--resume]

Per object and draw (theta from ``case_rng(seed, object, idx)``, as in the corpus): scene A pushes,
then weighs from the state the push left; scene B, a fresh reset under the same theta, weighs alone.
One row per probe run, key ``<object>__s<seed>__d<idx>__<scene><probe>`` (Apush, Aweigh, Bweigh),
with the pose before the probe and the base features (``extract`` at the s 0.5 proposal with the
weigh squeeze, v_lift 0.03, no transport, table centre). A push that throws the object (off the
table top or more than 10 cm away) ends scene A: no Aweigh row.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from typing import Any

import numpy as np

from robots.mujoco_lab.corpus import case_rng, draw_theta
from rpent.skills.pick_place.probes import weigh_force

PROBE_S = 0.5
PROBE_V_LIFT = 0.03
PROBE_DESTINATION = "table_centre"


def _state(env, force: float) -> tuple[Any, dict[str, Any], dict[str, float]]:
    """(the s 0.5 proposal, its pose, the base features) of the current state."""
    from rpent.skills.pick_place.features import extract

    prop = next(p for p in env.proposals() if abs(p.s - PROBE_S) < 1e-9)
    truth = env.truth()
    base = extract(env.geometry(), np.asarray(prop.point), prop.yaw, truth["mass"], truth["mu"], force, PROBE_V_LIFT, 0.0,
                   PROBE_DESTINATION)
    return prop, env.proposal_pose(prop), base


def probe_rows(env, object_id: str, idx: int, seed: int) -> list[dict[str, Any]]:
    """Scene A (push, then weigh unless the push threw the object) and scene B (weigh) of one draw."""
    from robots.mujoco_lab.adapter import object_props

    rng = case_rng(seed, object_id, idx)
    props = object_props(object_id)
    theta, com_end = draw_theta(rng, props["mass"])
    f_std = float(env.s["t3_make"].std_force(props["mass"]))
    force = weigh_force(f_std)
    reset = {"seed": seed * 1000 + idx, "com_end": com_end, "fixture": None}
    env.reset(object_id, theta, **reset)
    truth = env.truth()
    common = {"object": object_id, "draw": idx, "seed": seed, "theta": theta, "com_end": com_end, "force": force, "f_std": f_std,
              "com_obj_true": truth["com_obj"], "truth": {k: truth[k] for k in ("mass", "mu", "com_s")}}

    def row(tag: str, order: int, pose: dict[str, Any], base: dict[str, float], res) -> dict[str, Any]:
        return {"key": f"{object_id}__s{seed}__d{idx}__{tag}", **common, "scene": tag[0], "order": order, "pose": pose,
                "base": base, **res.to_data()}

    _, pose, base = _state(env, force)
    push = env.probe_push()
    rows = [row("Apush", 0, pose, base, push)]
    if not push.thrown:
        prop, pose, base = _state(env, force)
        rows.append(row("Aweigh", 1, pose, base, env.probe_weigh(prop, force)))
    env.reset(object_id, theta, **reset)
    prop, pose, base = _state(env, force)
    rows.append(row("Bweigh", 0, pose, base, env.probe_weigh(prop, force)))
    return rows


def _worker(args: tuple[str, int, int, list[int]]) -> list[dict[str, Any]]:
    object_id, draws, seed, skip = args
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    env = MuJoCoLabEnv()
    rows: list[dict[str, Any]] = []
    for idx in range(draws):
        if idx in skip:
            continue
        try:
            rows.extend(probe_rows(env, object_id, idx, seed))
        except Exception as exc:  # noqa: BLE001  one bad draw must not kill the object's batch
            rows.append({"key": f"{object_id}__s{seed}__d{idx}", "object": object_id, "draw": idx, "error": str(exc)[:200]})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=24)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--objects", default=None, help="comma-separated object ids (default: all)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resume", action="store_true", help="skip the draws that already have rows in --out")
    a = ap.parse_args(argv)
    from robots.mujoco_lab.adapter import object_ids

    objects = a.objects.split(",") if a.objects else object_ids()
    done: dict[str, set[int]] = {}
    if a.resume and os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" not in r:
                done.setdefault(r["object"], set()).add(int(r["draw"]))
    jobs = [(oid, a.draws, a.seed, sorted(done.get(oid, set()))) for oid in objects]
    written = 0
    mode = "a" if a.resume and os.path.exists(a.out) else "w"
    with open(a.out, mode) as f:
        if a.workers <= 1:
            results = map(_worker, jobs)
        else:
            pool = mp.get_context("spawn").Pool(a.workers)
            results = pool.imap_unordered(_worker, jobs)
        for rows in results:
            for r in rows:
                f.write(json.dumps(r) + "\n")
                written += "error" not in r
            f.flush()
            print(f"{rows[0]['object'] if rows else '-'}: {len(rows)} rows", file=sys.stderr, flush=True)
        if a.workers > 1:
            pool.close()
            pool.join()
    return written


if __name__ == "__main__":
    sys.exit(0 if main() >= 0 else 1)
