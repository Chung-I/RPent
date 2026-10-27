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

"""Corpus generator: pick, carry and set down every object under drawn hidden properties.

    python -m robots.mujoco_lab.corpus --out PATH --draws 48 --workers 8 [--objects a,b] [--resume]

One row per rollout (JSON lines): the hidden draw, the action, the measured
geometry, the features (true-θ, see features.extract), the planned trajectory
and the outcome label with the loss record. One worker process per object.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from typing import Any

import numpy as np

from rpent.skills.pick_place.destinations import DESTINATIONS, MESH_IDS, PRIMITIVE_IDS, release_height

MASS_SCALE_RANGE = (0.5, 3.0)
MU_RANGE = (0.1, 0.9)
COM_S_RANGE = (0.3, 0.8)
V_LIFT_CHOICES = (0.05, 0.15, 0.4, 0.8)
V_TRANSPORT_CHOICES = (0.2, 0.5, 1.0)
S_CHOICES = (0.3, 0.5, 0.7)
P_COM_SHIFT = 0.5
P_DROP_RELEASE = 0.3
RELEASE_KEYS = ("tilt_deg", "displacement_xy_m", "drop_height_m", "final_zone", "final_speed")


def draw_theta(rng: np.random.Generator, nominal_mass: float) -> tuple[dict[str, Any], int]:
    """Hidden properties for one rollout and the end the CoM mass sits at (±1)."""
    lo, hi = MASS_SCALE_RANGE
    theta = {
        "mass_scale": float(np.exp(rng.uniform(np.log(lo), np.log(hi)))),
        "mu": float(rng.uniform(*MU_RANGE)),
        "com_s": None if rng.random() < P_COM_SHIFT else float(rng.uniform(*COM_S_RANGE)),
        "v_lift": float(rng.choice(V_LIFT_CHOICES)),
    }
    com_end = 1 if rng.random() < 0.5 else -1
    return theta, com_end


def draw_action(rng: np.random.Generator, f_std: float, use_fixtures: bool = False) -> dict[str, Any]:
    if use_fixtures and MESH_IDS and rng.random() < 0.5:
        dest_id = str(rng.choice(sorted(MESH_IDS)))
    else:
        dest_id = str(rng.choice(sorted(PRIMITIVE_IDS)))
    dest = DESTINATIONS[dest_id]
    rel = None
    if dest["kind"] in ("rim", "cavity", "open_bin") and rng.random() < P_DROP_RELEASE:
        rel = 0.05 + float(dest["rim_height"])            # the careless "drop it in" release
    return {
        "destination": dest_id,
        "s": float(rng.choice(S_CHOICES)),
        "force": float(f_std if rng.random() < 0.5 else 2 * f_std),
        "v_transport": float(rng.choice(V_TRANSPORT_CHOICES)),
        "release_height": rel,
    }


def _release_summary(rel: dict[str, Any] | None) -> dict[str, Any] | None:
    if rel is None:
        return None
    out = {k: rel.get(k) for k in RELEASE_KEYS}
    fc = rel.get("first_contact")
    out["impact_speed"] = None if fc is None else fc.get("impact_speed")
    out["first_zone"] = None if fc is None else fc.get("zone")
    return out


def rollout_row(env, object_id: str, idx: int, seed: int, use_fixtures: bool = False) -> dict[str, Any]:
    """One rollout of `object_id` under a fresh hidden draw; returns the corpus row."""
    from robots.mujoco_lab.adapter import object_props
    from rpent.skills.pick_place.features import extract

    rng = np.random.default_rng([seed, idx])
    props = object_props(object_id)
    theta, com_end = draw_theta(rng, props["mass"])
    f_std = env.s["t3_make"].std_force(props["mass"])
    act = draw_action(rng, f_std, use_fixtures)
    fixture = act["destination"] if act["destination"] in MESH_IDS else None
    env.reset(object_id, theta, seed=seed * 1000 + idx, com_end=com_end, fixture=fixture)
    props_list = env.proposals()
    prop = next(p for p in props_list if abs(p.s - act["s"]) < 1e-9)
    truth = env.truth()
    loc = env.locate("object")
    g = env.geometry()
    feats = extract(g, np.asarray(prop.point), prop.yaw, truth["mass"], truth["mu"], act["force"], theta["v_lift"],
                    act["v_transport"], act["destination"], act["release_height"])
    r = env.execute(prop, act["force"], theta["v_lift"], act["v_transport"], act["destination"], act["release_height"])
    dest = DESTINATIONS[act["destination"]]
    return {
        "key": f"{object_id}__d{idx}",
        "object": object_id,
        "theta": theta,
        "com_end": com_end,
        "truth": {k: truth[k] for k in ("mass", "mu", "com_s")},
        "seed": seed,
        "destination": act["destination"],
        "fixture": fixture,
        "s": act["s"],
        "force": act["force"],
        "v_lift": theta["v_lift"],
        "v_transport": act["v_transport"],
        "release_height": act["release_height"] if act["release_height"] is not None else release_height(dest),
        "width": prop.width,
        "label": r.label,
        "max_z": r.max_z,
        "drop_z": r.drop_z,
        "loss_phase": r.loss_phase,
        "loss_step": r.loss_step,
        "loss_t": r.loss_t,
        "loss_speed": r.loss_speed,
        "zone": r.zone,
        "release": _release_summary(r.release),
        "planned": [s.to_dict() for s in r.planned],
        "geometry": {"base_extent": g.base_extent, "com_height": g.com_height, "footprint_half_width": g.footprint_half_width,
                     "length": g.length, "volume": loc["volume"], "material": loc["material"], "hollow": loc["hollow"]},
        "features": feats,
    }


def _worker(args: tuple[str, int, int, list[int], bool]) -> list[dict[str, Any]]:
    object_id, draws, seed, skip, use_fixtures = args
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    env = MuJoCoLabEnv()
    rows = []
    for idx in range(draws):
        if idx in skip:
            continue
        try:
            rows.append(rollout_row(env, object_id, idx, seed, use_fixtures))
        except Exception as exc:  # noqa: BLE001  one bad rollout must not kill the object's batch
            rows.append({"key": f"{object_id}__d{idx}", "object": object_id, "error": str(exc)[:200]})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=48)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--objects", default=None, help="comma-separated object ids (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fixtures", action="store_true", help="draw half the destinations from the inspected mesh fixtures")
    a = ap.parse_args(argv)
    from robots.mujoco_lab.adapter import object_ids

    objects = a.objects.split(",") if a.objects else object_ids()
    done: dict[str, set[int]] = {}
    if a.resume and os.path.exists(a.out):
        for line in open(a.out):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in row:
                continue
            done.setdefault(row["object"], set()).add(int(row["key"].rsplit("d", 1)[1]))
    jobs = [(oid, a.draws, a.seed, sorted(done.get(oid, set())), a.fixtures) for oid in objects]
    written = 0
    mode = "a" if a.resume and os.path.exists(a.out) else "w"
    with open(a.out, mode) as f:
        if a.workers <= 1:
            results = map(_worker, jobs)
        else:
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(a.workers)
            results = pool.imap_unordered(_worker, jobs)
        for rows in results:
            for row in rows:
                f.write(json.dumps(row) + "\n")
                written += "error" not in row
            f.flush()
            print(f"{rows[0]['object'] if rows else '-'}: {len(rows)} rows", file=sys.stderr, flush=True)
        if a.workers > 1:
            pool.close()
            pool.join()
    return written


if __name__ == "__main__":
    sys.exit(0 if main() >= 0 else 1)
