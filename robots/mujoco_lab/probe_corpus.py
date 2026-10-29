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


"""Probe corpus: the push, weigh and squeeze probes under drawn hidden properties.

    python -P -m robots.mujoco_lab.probe_corpus --out PATH --draws 24 --workers 16 --seed 7 [--objects a,b] [--resume]
        [--classes PATH] [--thresholds PATH] [--mode {v8,m2}]

Per object and draw (theta from ``case_rng(seed, object, idx)``, as in the corpus): scene A pushes,
then weighs from the state the push left; scene B, a fresh reset under the same theta, weighs alone.
One row per probe run, key ``<object>__s<seed>__d<idx>__<scene><probe>`` (Apush, Aweigh, Bweigh),
with the pose before the probe and the base features (``extract`` with the weigh squeeze, v_lift 0.03,
no transport, table centre) at the proposal the menu would probe at: the usable one nearest s 0.5
(``menu.mid_proposal``), whose s the row carries. A push that throws the object (off the table top or
more than 10 cm away) ends scene A: no Aweigh row. When no proposal fits the gripper the skill declines
before any probe: the draw gets one row ``<object>__s<seed>__d<idx>`` that says why ("declined"), and an
Aweigh after a push that leaves no fitting proposal gets a declined row in place of its run.

Probe corpus v8 (milestone 2b): each draw also takes the corpus's damage draw (corpus.draw_damage: the fill and the
hidden fragility, from the damage stream), every row carries ``fill``, ``theta_frag``, ``spill_angle_deg`` and
``entry``, and each run its ``meter`` and ``damage``. A push that damaged the object ends scene A as a throw does. A
crushable or bruisable object gets scene C, a fresh reset that squeezes (``Csqueeze``) at the cap of
``probes.squeeze_cap`` on the m2b prior (2000 particles from the stream [seed, crc32(object), idx, 10]); its row's
``force`` and base features are the cap's. ``--mode m2`` is probe corpus v7's world: v8's draws, no liquid, no
fragility, no scene C.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from typing import Any

import numpy as np

import zlib

from robots.mujoco_lab.corpus import apply_damage, case_rng, damage_rng, draw_damage, draw_theta, world
from rpent.skills.pick_place.probes import squeeze_cap, weigh_force

PROBE_V_LIFT = 0.03
PROBE_DESTINATION = "table_centre"
SQUEEZE_CLASSES = ("crushable", "bruisable")
SQUEEZE_PRIOR_N = 2000
SQUEEZE_STREAM = 10
MODES = ("v8", "m2")


def _state(env, force: float) -> tuple[Any, dict[str, Any], dict[str, float]] | str:
    """(the proposal the menu would probe at, its pose, the base features) of the current state, or the
    menu's decline reason when no proposal fits the gripper."""
    from rpent.skills.pick_place.features import extract
    from rpent.skills.pick_place.menu import decline_reason, mid_proposal, pinch_limit, usable_proposals

    props, lim = env.proposals(), pinch_limit(env)
    usable = usable_proposals(props, lim)
    if not usable:
        return decline_reason(props, lim)
    prop = mid_proposal(usable)
    truth = env.truth()
    base = extract(env.geometry(), np.asarray(prop.point), prop.yaw, truth["mass"], truth["mu"], force, PROBE_V_LIFT, 0.0,
                   PROBE_DESTINATION)
    return prop, env.proposal_pose(prop), base


def probe_rows(env, object_id: str, idx: int, seed: int, classes: str | dict[str, Any] | None = None,
               thresholds: Any = None, mode: str = "v8") -> list[dict[str, Any]]:
    """Scene A (push, then weigh unless the push threw or damaged the object), scene B (weigh) and, for a crushable
    or bruisable object, scene C (squeeze at the prior's cap) of one draw (``mode`` m2: v7's world, no scene C)."""
    from robots.mujoco_lab.adapter import object_props
    from robots.mujoco_lab.fragility import FLAGS, entry_of
    from robots.mujoco_lab.prior import make_prior

    rng = case_rng(seed, object_id, idx)
    props = object_props(object_id)
    theta, com_end = draw_theta(rng, props["mass"])
    f_std = float(env.s["t3_make"].std_force(props["mass"]))
    C, th = world(classes, thresholds)
    entry, _ = entry_of(object_id, C)
    if mode not in MODES:
        raise ValueError(f"unknown probe corpus mode {mode!r}; known: {MODES}")
    d = draw_damage(damage_rng(seed, object_id, idx), entry, th, f_std)
    if mode == "m2":                          # v7's world: nothing hidden beyond theta
        d = {**d, "fill": 0.0, "theta_frag": None, "spill_angle_deg": None}
    theta, _ = apply_damage(theta, {}, d)      # a probe takes no squeeze or carry from the draw
    force = weigh_force(f_std)
    env.slosh_zeta = th.slosh_zeta
    reset = {"seed": seed * 1000 + idx, "com_end": com_end, "fixture": None, "fill": d["fill"], "frag": d["theta_frag"]}
    env.reset(object_id, theta, **reset)
    start = dict(env.reset_info)
    truth = env.truth()
    common = {"object": object_id, "draw": idx, "seed": seed, "theta": theta, "com_end": com_end, "force": force, "f_std": f_std,
              "com_obj_true": truth["com_obj"], "truth": {k: truth[k] for k in ("mass", "mu", "com_s")},
              "entry": {k: entry.get(k) for k in FLAGS}, "theta_frag": d["theta_frag"], "fill": d["fill"],
              "spill_angle_deg": d["spill_angle_deg"]}

    def row(tag: str, order: int, st: tuple[Any, dict[str, Any], dict[str, float]], res) -> dict[str, Any]:
        prop, pose, base = st
        return {"key": f"{object_id}__s{seed}__d{idx}__{tag}", **common, "scene": tag[0], "order": order, "s": prop.s,
                "pose": pose, "base": base, **res.to_data(), "reset": start}

    def declined(tag: str, order: int, why: str) -> dict[str, Any]:
        return {"key": f"{object_id}__s{seed}__d{idx}__{tag}", **common, "scene": tag[0], "order": order, "declined": why,
                "reset": start}

    st = _state(env, force)
    if isinstance(st, str):                     # no proposal fits: the skill declines before any probe
        return [{"key": f"{object_id}__s{seed}__d{idx}", **common, "declined": st, "reset": start}]
    push = env.probe_push()
    rows = [row("Apush", 0, st, push)]
    if not push.thrown and push.damage is None:          # a throw or damage ends scene A
        st = _state(env, force)
        rows.append(declined("Aweigh", 1, st) if isinstance(st, str) else row("Aweigh", 1, st, env.probe_weigh(st[0], force)))
    env.reset(object_id, theta, **reset)
    start = dict(env.reset_info)
    st = _state(env, force)
    rows.append(declined("Bweigh", 0, st) if isinstance(st, str) else row("Bweigh", 0, st, env.probe_weigh(st[0], force)))
    if mode != "m2" and entry.get("class") in SQUEEZE_CLASSES:
        env.reset(object_id, theta, **reset)
        start = dict(env.reset_info)
        belief = make_prior(C, th)(env.pose_info(), SQUEEZE_PRIOR_N,
                                   np.random.default_rng([int(seed), zlib.crc32(object_id.encode()), int(idx), SQUEEZE_STREAM]))
        cap = float(squeeze_cap(belief, f_std))
        st = _state(env, cap)
        if isinstance(st, str):
            rows.append({**declined("Csqueeze", 0, st), "force": cap})
        else:
            rows.append({**row("Csqueeze", 0, st, env.probe_squeeze(st[0], cap)), "force": cap})
    return rows


def _worker(args: tuple) -> list[dict[str, Any]]:
    object_id, draws, seed, skip, classes, thresholds, mode = args
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    env = MuJoCoLabEnv()
    rows: list[dict[str, Any]] = []
    for idx in range(draws):
        if idx in skip:
            continue
        try:
            rows.extend(probe_rows(env, object_id, idx, seed, classes, thresholds, mode))
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
    ap.add_argument("--classes", default=None, help="the fragility class file (default: data/fragility_classes.json)")
    ap.add_argument("--thresholds", default=None, help="the thresholds file (default: data/thresholds_v0.json)")
    ap.add_argument("--mode", default="v8", choices=MODES, help="v8: the damage draws and scene C; m2: v7's world")
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
    jobs = [(oid, a.draws, a.seed, sorted(done.get(oid, set())), a.classes, a.thresholds, a.mode) for oid in objects]
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
