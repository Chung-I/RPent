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
"""The level-A rerun driver (design 2026-11-08-sixdof-rerun-design.md, unit (f)): the pilot's cases with the 30–45°
rows added. The funnel of those rows (funnel6): the width rules on the pinch's own closing axis, the kinematic
clearance (ruling 1), the configuration and path check (choose) with the hand boxes in the planner's set, then the
cached cuRobo free move and execute6. The ≤ 30° rows (funnel30) keep the pilot's funnel and executor without its cap,
with the published planner set (ruling A) and the clearance as a flag (amendment 6)."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from robots.mujoco_lab import sixdof

VERTICAL_BIN = sixdof.BINS[0]         # "≤ 30°": the pilot's vertical rows, its mapping and executor
LEVEL_A_BIN = sixdof.BINS[1]          # "30–45°": the rows level A adds; the pilot's vertical rows stay its own
FUNNEL6 = ("generated", "level_a", "chosen", "disagree")                 # funnel6's counts
FUNNEL30 = ("generated", "vertical", "mapped", "width", "centred", "reach", "distinct", "kept", "width_b", "pads_free", "reach_b",
            "distinct_b", "kept_b", "clearance_fail")                     # funnel30's: the pilot's (FUNNEL, FUNNEL_B), clearance_fail
SAME6 = ("label", "artifact", "error", "damage", "drop_z", "loss_phase", "max_z", "qpos_sha", "trace_sha", "free", "unreachable")
#                                                                          G3's replay: curve_poseset.SAME, the free move, G2
CHECK_SPEEDS = {"v_lift": 0.1, "v_transport": 0.5, "release_height": None}   # the speeds the pilot's reach() checks at
LEVEL_A_SIM = "ee882f9b5574a4adc30559fd92c9df3bf69afdf3"    # daily-logs: the contacts sim (amendment 8), the level-A record's
LEVEL_A_CODE = "f63ff35ce11cca646650c99c14e5eb9c95a1fa77"   # RPent sixdof: the level-A record's code (unit f, step 6)


@contextmanager
def planner_set(env: Any, without_boxes: bool) -> Iterator[Any]:
    """env.plan_without_hand_boxes at ``without_boxes`` inside the block (on: ruling A's published set, the arm links
    and the pads), the caller's value after it."""
    old = env.plan_without_hand_boxes
    env.plan_without_hand_boxes = without_boxes
    try:
        yield env
    finally:
        env.plan_without_hand_boxes = old


def _path_reason(f: dict[str, Any]) -> str:
    hit = f.get("hit")
    return f"path: {f.get('reason')} ({f.get('phase')}{', ' + '/'.join(map(str, hit[:2])) if hit else ''})"


def hand_boxes(sim: Any) -> dict[str, str]:
    """The hand's boxes (sixdof.hand_geoms) by the name reach6 gives a hit's robot geom, its MJCF name: {name: palm,
    wrist or finger}. The finger boxes have no name ("")."""
    return {sim.m.geom(g).name: body for body, gs in sixdof.hand_geoms(sim).items() for g in gs}


def box_hit(failed: list[dict[str, Any]], boxes: dict[str, str]) -> dict[str, Any] | None:
    """The first of choose()'s failed candidates whose reach6 collision has a hand box as its robot geom: {index, box,
    phase, other}, or None."""
    for f in failed:
        if f["reason"] == "collision" and f["hit"][0] in boxes:
            return {"index": f["index"], "box": boxes[f["hit"][0]], "phase": f["phase"], "other": f["hit"][1]}
    return None


def disagree(clear: dict[str, Any] | None, candidate: dict[str, Any] | None, hit: dict[str, Any] | None) -> str | None:
    """The disagreement of the clearance (its failure, or None) and reach6 (choose()'s candidate or None, and its
    box_hit) that the funnel table counts: "clearance pass, reach6 box hit" or "clearance fail, reach6 pass", else
    None."""
    if clear is None and candidate is None and hit is not None:
        return "clearance pass, reach6 box hit"
    if clear is not None and candidate is not None:
        return "clearance fail, reach6 pass"
    return None


def assess(c: dict[str, Any] | None, ch: dict[str, Any], boxes: dict[str, str]) -> tuple[dict[str, Any], str | None]:
    """A width-passing row's fields from its clearance (``c``: the failure or None) and choose() (``ch``) results:
    clearance, reach, disagree and candidate (only when both pass), and the step the row stops at (the clearance
    first, then the path check; None: the row goes to the planner)."""
    cand, hit = ch["candidate"], box_hit(ch["failed"], boxes)
    out: dict[str, Any] = {"clearance": None if c is None else {"first": c["first"], "pairs": sorted(c["pairs"])},
                           "reach": {"goal": bool(ch["goal"]), "candidate": None if cand is None else int(cand["index"]),
                                     "box_hit": hit, "failed": ch["failed"]},
                           "disagree": disagree(c, cand, hit), "candidate": None}
    if c is not None:
        return out, f"clearance: {c['first']['pair']}"
    if cand is None:
        return out, _path_reason(ch["failed"][0]) if ch["failed"] else "path: no candidate"
    out["candidate"] = {"index": int(cand["index"]), "twin": bool(cand["twin"]), "seed": cand["seed"], "q": [float(x) for x in cand["q"]]}
    return out, None


def funnel6(env: Any, grasps: Any, conf: Any, destination: str, case: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """The level-A funnel over one case's grasps, by confidence (a stable sort): the rows in the 30–45° bin, each with
    both width rules (width_rule6: ``why_width``, ``why_width_b``), and, where one passes, the clearance (its first pair
    and all pairs, or None) and choose(), also after a clearance failure (``reach``: goal, the chosen candidate's index
    or None, box_hit, the failed candidates), and ``disagree`` of the two. ``candidate`` (index, twin, seed, q): only
    for a row that passes both. ``why`` / ``why_b``: the first step each rule's row stops at (None: the row goes to the
    planner). The path check sees the hand boxes whatever the caller's planner set (planner_set off). Counts: generated,
    level_a, chosen, disagree."""
    G, S = np.asarray(grasps, dtype=float), np.asarray(conf, dtype=float)
    boxes = hand_boxes(env.sim)
    rows: list[dict[str, Any]] = []
    with planner_set(env, False):
        for rank, k in enumerate(np.argsort(-S, kind="stable")):
            p = sixdof.from_graspgen(G[k])
            deg = float(sixdof.approach_deg(p))
            if sixdof.angle_bin(deg) != LEVEL_A_BIN:
                continue
            rid = f"g{int(k)}"
            wr = sixdof.width_rule6(env, p)
            why, why_b = wr["why"], wr["why_b"]
            r: dict[str, Any] = {"id": rid, "rank": rank, "conf": float(S[k]), "approach_deg": deg, "bin": LEVEL_A_BIN,
                                 "width": float(wr["width"]), "why_width": why, "why_width_b": why_b, "clearance": None,
                                 "reach": None, "disagree": None, "candidate": None}
            if why is None or why_b is None:
                p = sixdof.from_graspgen(G[k], width=float(wr["width"]))
                c = sixdof.clearance(env, p, float(wr["width"]), destination)
                out, stop = assess(c, sixdof.choose(env, p, sixdof.row_key(case, rid), destination), boxes)
                r.update(out)
                why, why_b = why or stop, why_b or stop
            r.update(why=why, why_b=why_b)
            rows.append(r)
    counts = {"generated": int(len(S)), "level_a": len(rows), "chosen": sum(r["candidate"] is not None for r in rows),
              "disagree": sum(r["disagree"] is not None for r in rows)}
    return rows, counts


def vertical_clearance(env: Any, r: dict[str, Any], destination: str) -> dict[str, Any] | None:
    """Ruling 1's clearance of a vertical row (amendment 6): along today's planned path of the row's Proposal
    (palm_audit.today_path at CHECK_SPEEDS), the closed fingers on the exact slab width (as width_rule6 measures a
    tilted row). sixdof.clearance_on_path's result, or None when clear."""
    from robots.mujoco_lab import curve_graspgen as gg
    from robots.mujoco_lab.palm_audit import today_path

    path = today_path(env, gg._proposal(r), dict(CHECK_SPEEDS, destination=destination))
    return sixdof.clearance_on_path(env, path, float(r["slab"][1] - r["slab"][0]), destination)


def funnel30(env: Any, grasps: Any, conf: Any, destination: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """The ≤ 30° rows: the pilot's funnel (curve_graspgen.funnel: the vertical mapping, both width rules, the centring,
    reach()) with ruling A's planner set (planner_set on, as Stage 2c: the published planner, and the physics decides),
    and ruling 1's clearance on every row that reaches reach() (vertical_clearance; ``clearance``: its first pair and
    all pairs, or None). A row that fails the clearance stays kept and runs, flagged (amendment 6). ``kept`` /
    ``kept_b``: the pilot's flags under its cap (KEEP_ROWS); ``kept_all`` / ``kept_all_b``: without the cap, feasible
    and not a duplicate (the pilot's ``distinct`` counts). Counts: the pilot's and clearance_fail."""
    from robots.mujoco_lab import curve_graspgen as gg

    with planner_set(env, True):
        rows, counts = gg.funnel(env, grasps, conf, destination)
    for r in rows:
        c = vertical_clearance(env, r, destination) if r["reach"] is not None else None
        r.update(bin=VERTICAL_BIN, clearance=None if c is None else {"first": c["first"], "pairs": sorted(c["pairs"])},
                 kept_all=bool(r["feasible"]) and r["dup_of"] is None, kept_all_b=bool(r["feasible_b"]) and r["dup_of_b"] is None)
    counts["clearance_fail"] = sum(r["clearance"] is not None for r in rows)
    return rows, counts


# ------------------------------------------------------------------ the prepass: the funnels at the case's reset, the free-move plans

def goals6(rows6: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The free-move goals of the tilted rows that pass the funnel (a candidate), in the rows' order: [{"row", "q"}]."""
    return [{"row": r["id"], "q": r["candidate"]["q"]} for r in rows6 if r["candidate"] is not None]


def case_funnel(env: Any, case: dict[str, Any], grasps: Any, conf: Any, destination: str) -> dict[str, Any]:
    """One case's funnels at its reset (the object at rest, the arm at its start): funnel30's and funnel6's rows and
    counts, and the free-move request of the tilted rows that pass (sixdof_plan.request with world_export's world and
    the arm's desired joints as the start, both read before the funnels, and goals6), None when no tilted row passes.
    The execute pass reads the same world and start after its own reset of the case."""
    from robots.mujoco_lab import sixdof_plan as sp

    world, start = sp.world_export(env.sim), [float(x) for x in env.sim.arm.desired()]
    rows30, counts30 = funnel30(env, grasps, conf, destination)
    rows6, counts6 = funnel6(env, grasps, conf, destination, case)
    goals = goals6(rows6)
    return {"rows30": rows30, "counts30": counts30, "rows6": rows6, "counts6": counts6,
            "request": sp.request((case["object"], int(case["idx"])), world, start, goals) if goals else None}


def prepass_case(prims: Any, rec1: dict[str, Any], seed: int, world: Any, entry: dict[str, Any] | None) -> dict[str, Any]:
    """One case's prepass on the lab side, with the pilot's case checks (curve_graspgen.graspgen_case): Stage 2's skip
    rules; the case's draw and its reset at its own seed (curve_graspgen._setup; Stage 1's fingerprint); the cloud of
    the hulls there must be the one the generator saw; commit_std's base (base_gate); then case_funnel at that reset. A
    failed check is the case's stop, before the funnels. The record holds what the execute pass needs (the
    destination, the base, the hulls' hash, both funnels' rows, the request) and the pilot's case fields. An exception
    is the case's error (the case drops)."""
    from robots.mujoco_lab import com_gap
    from robots.mujoco_lab import curve_graspgen as gg
    from robots.mujoco_lab.curve import stage2_skip
    from robots.mujoco_lab.curve_poseset import KEEP, KIND_OF, UPRIGHT_DEG, _err, balance_fraction, base_gate, commit_std_base, tilt_deg

    o, idx = rec1["object"], int(rec1["idx"])
    rec: dict[str, Any] = dict({k: rec1.get(k) for k in KEEP}, kind=KIND_OF.get(o, "other"), hammer="hammer" in o, rows30=[],
                               counts30=None, rows6=[], counts6=None, request=None, n_grasps=None, base=None, destination=None,
                               hull_sha=None, skipped=stage2_skip(rec1), dropped=False, stop=None, error=None)
    if rec["skipped"]:
        return dict(rec, dropped=True)
    if entry is None:
        return dict(rec, stop="no grasps for this case")
    if entry.get("error"):
        return dict(rec, stop=f"the generator failed on this case: {entry['error']}")
    env = prims.env
    st = gg._setup(prims, rec1, seed, world)
    if st["stop"]:
        return dict(rec, stop=st["stop"])
    try:
        st["reset_at"](0)()
        if com_gap.fingerprint(env) != rec1["fingerprint"]:
            return dict(rec, stop="the reset's fingerprint differs from Stage 1's")
        sha = gg._sha_bytes(gg.surface_cloud(env._hull_vertex_list(), seed=gg.cloud_seed(o, idx)))
        if sha != entry["cloud_sha256"]:
            return dict(rec, stop="the cloud at this reset is not the cloud the generator saw")
        t = env.truth()
        tilt = tilt_deg(env.sim.obj_mat(), env.props["axis"])
        rec.update(tilt_deg=tilt, upright=tilt > UPRIGHT_DEG, length=float(env.props["length"]), cloud_sha256=sha,
                   com_f=balance_fraction(env.props, t["com_obj"]), sim_truth={k: t.get(k) for k in ("mass", "com_s", "com_obj", "fill")})
        base = commit_std_base(env, o, st["destination"])
        why = base_gate(rec1, base)
        if why:
            return dict(rec, stop=why)
        rec.update(base=base, destination=st["destination"], hull_sha=gg.hulls_sha(env), n_grasps=int(len(entry["conf"])))
        rec.update(case_funnel(env, {"object": o, "idx": idx}, entry["grasps"], entry["conf"], st["destination"]))
    except Exception as exc:  # noqa: BLE001  a case that raises outside the planner is dropped, as in the pilot
        return dict(rec, error=_err(exc), dropped=True)
    return rec


def plan_prepass(rec: dict[str, Any], cache: str, plan: Any = None) -> dict[str, Any]:
    """The free-move plans of one prepass record: its request, when the case neither stopped nor dropped, to one cuRobo
    process (``plan``, default sixdof_plan.plan_case: the uncached goals planned and cached, failures included).
    ``plan``: the process's summary {"goals", "planned", "cached"}, else None; ``plan_error``: its failure (the case's
    goals stay unplanned, so the execute pass counts them as cache misses, G3)."""
    from robots.mujoco_lab import sixdof_plan as sp
    from robots.mujoco_lab.curve_poseset import _err

    out = dict(rec, plan=None, plan_error=None)
    if rec.get("stop") or rec.get("dropped") or rec.get("request") is None:
        return out
    try:
        out["plan"] = (plan or sp.plan_case)(rec["request"], cache)
    except Exception as exc:  # noqa: BLE001  a planner process that fails is this case's record, not the run's end
        out["plan_error"] = _err(exc)
    return out


# ------------------------------------------------------------------ the execute pass: lab only, cached plans only (G3)

def straight_path(start: Any, goal: Any, hz: float, v: float) -> list[np.ndarray]:
    """The straight joint line from ``start`` to ``goal``, past the start, in steps of at most ``v``/``hz`` rad on every
    joint; it ends at the goal."""
    a, b = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    n = max(1, int(math.ceil(float(np.max(np.abs(b - a))) / (v / hz))))
    return [a + (b - a) * (k / n) for k in range(1, n + 1)]


def plan_failure(sim: Any, start: Any, goal: Any, hz: float) -> dict[str, Any]:
    """Why cuRobo's cached free move to ``goal`` failed, read with the lab's exact geometry (sixdof_plan.free_hits) on
    the straight joint line from the start to the goal (straight_path at the arm's FREE_V, the goal included): "planner
    spheres" when nothing touches the scene, a lower-bound test of the failures that cuRobo's sphere cover (its
    protrusion) causes, since a blocked straight line does not prove that no path exists; else "path blocked".
    {"reason", "hits", "first"}."""
    from robots.mujoco_lab import sixdof_plan as sp

    hits = sp.free_hits(sim, straight_path(start, goal, hz, float(sixdof._armlib(sim.arm).FREE_V)))
    return {"reason": "path blocked" if hits else "planner spheres", "hits": len(hits), "first": hits[0] if hits else None}


def hits_summary(hits: list[dict[str, Any]]) -> dict[str, Any]:
    """sixdof_plan.free_hits in short: the number of hits, the first, and each robot/other pair once (sorted)."""
    return {"n": len(hits), "first": hits[0] if hits else None, "pairs": sorted({f"{h['robot']}/{h['other']}" for h in hits})}


def run_row30(env: Any, reset: Any, r: dict[str, Any], base: dict[str, Any], fingerprint: Any, want: str,
              hull_sha: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """A ≤ 30° row through today's executor after its own reset (curve_sixdof.run_row at the row's Proposal), with
    ruling A's planner set and the box contacts recorded (BoxContacts: ``box_contacts``). run_row's return."""
    from robots.mujoco_lab import curve_graspgen as gg
    from robots.mujoco_lab.curve_sixdof import BoxContacts, run_row

    with planner_set(env, True), BoxContacts(env) as bc:
        new, stop = run_row(env, reset, lambda e: (gg._proposal(r), None), base, fingerprint, want, hull_sha)
    if new is not None:
        new["box_contacts"] = bc.summary()
    return new, stop


def run_row6(env: Any, reset: Any, r: dict[str, Any], p: sixdof.Grasp6, cache: str, base: dict[str, Any], fingerprint: Any,
             want: str, hull_sha: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """A 30–45° row after its own reset (the reset checks of curve_sixdof.run_row), with the hand boxes in the planner's
    set: the world and the arm's start there, and the free move to the row's candidate from the plan cache, never
    planned here (G3). ``free``: "miss" (PlanCacheMiss: the row stops, counted) or "failed" (PlanFailed: the row stops,
    ``free_failure`` = plan_failure with cuRobo's status) before any motion, else "ok": ``free_hits`` (the cached
    waypoints against the exact scene, a record) and execute6 at ``base`` on ``p`` with the box contacts recorded; a
    G2 failure (Unreachable6) is ``unreachable`` (its check), else the outcome fields and ``sixdof`` (tilt, turn, G2).
    Returns (the fields with the error and the seconds, None) or (None, the reason the case stops)."""
    from robots.mujoco_lab import sixdof_plan as sp
    from robots.mujoco_lab.curve_poseset import _err
    from robots.mujoco_lab.curve_sixdof import BoxContacts, _checked_reset, outcome

    with planner_set(env, False):
        at_rest, stop = _checked_reset(env, reset, fingerprint, want, hull_sha)
        if stop:
            return None, stop
        t0 = time.time()
        new: dict[str, Any] = {"label": None, "artifact": False, "error": None, "free": None, "free_failure": None,
                               "free_hits": None, "unreachable": None}
        try:
            hz = float(env.s["scene"].CTRL_HZ)
            start, goal = [float(x) for x in env.sim.arm.desired()], r["candidate"]["q"]
            try:
                free = sp.free_from_cache(cache, sp.world_export(env.sim), start, goal, hz)
            except sp.PlanCacheMiss:
                new["free"] = "miss"
            except sp.PlanFailed as exc:
                new.update(free="failed", free_failure=dict(plan_failure(env.sim, start, goal, hz), status=str(exc)))
            else:
                new.update(free="ok", free_hits=hits_summary(sp.free_hits(env.sim, free)))
                with BoxContacts(env) as bc:
                    try:
                        ro = sixdof.execute6(env, p, free, base["force"], base["v_lift"], base["v_transport"], base["destination"],
                                             base["release_height"], a_transport=base["a_transport"])
                    except sixdof.Unreachable6 as exc:
                        new["unreachable"] = exc.record
                    else:
                        new.update(outcome(env, ro, at_rest), sixdof=ro.sixdof)
                new["box_contacts"] = bc.summary()
        except Exception as exc:  # noqa: BLE001  a simulator fault is this row's error
            new["error"] = _err(exc)
        new["seconds"] = round(time.time() - t0, 2)
    return new, None


@contextmanager
def replan_on(env: Any) -> Iterator[Any]:
    """The executor's re-plan at the hold on inside the block (design amendment 8, rule (f): the rerun driver sets it
    on), the caller's value after it."""
    old = getattr(env, "replan_hold", False)
    env.replan_hold = True
    try:
        yield env
    finally:
        env.replan_hold = old


def _pinch(entry: dict[str, Any], r: dict[str, Any]) -> sixdof.Grasp6:
    """A 30–45° row's pinch: its grasp at its width, or the twin when its candidate is the twin's."""
    p = sixdof.from_graspgen(entry["grasps"][int(r["id"][1:])], width=float(r["width"]))
    return sixdof.twin(p) if r["candidate"]["twin"] else p


def replay_rows6(env: Any, reset: Any, rows30: list[dict[str, Any]], rows6: list[dict[str, Any]], entry: dict[str, Any], cache: str,
                 base: dict[str, Any], fingerprint: Any, want: str, hull_sha: str | None) -> dict[str, Any]:
    """G3's 2-seed replay (curve_graspgen.replay_rows): every row that ran, once more in the reverse order of the first
    pass, each after its own reset at the second seed (``reset``), with cached plans only (run_row6 never plans), each
    against its first run on SAME6. {"rows", "mismatches": [{"id", "first", "replay"}], "stop": "gN: why" or None}."""
    done = [r for r in reversed(rows6) if r.get("run") is not None] + [r for r in reversed(rows30) if r.get("run") is not None]
    bad: list[dict[str, Any]] = []
    for n, r in enumerate(done):
        if r.get("candidate") is not None:
            new, stop = run_row6(env, reset, r, _pinch(entry, r), cache, base, fingerprint, want, hull_sha)
        else:
            new, stop = run_row30(env, reset, r, base, fingerprint, want, hull_sha)
        if stop:
            return {"rows": n, "mismatches": bad, "stop": f"{r['id']}: {stop}"}
        first, again = [r["run"].get(k) for k in SAME6], [new.get(k) for k in SAME6]
        if first != again:
            bad.append({"id": r["id"], "first": first, "replay": again})
    return {"rows": len(done), "mismatches": bad, "stop": None}


def execute_case(prims: Any, rec1: dict[str, Any], seed: int, world: Any, prep: dict[str, Any], entry: dict[str, Any],
                 cache: str, replay: bool = False) -> dict[str, Any]:
    """The execute pass of one prepass record (lab only): nothing for a case that stopped or dropped in the prepass;
    else the case's setup (curve_graspgen._setup), a reset with the prepass's checks (Stage 1's fingerprint, the
    record's hulls, its base), then every row the record keeps, each after its own reset, with the executor's re-plan
    at the hold on (replan_on): the ≤ 30° rows that either width rule keeps without the cap (kept_all, kept_all_b;
    run_row30), then the 30–45° rows with a candidate (run_row6 on the row's pinch, or on its twin when the candidate
    is the twin's). Each row's fields go into the row as ``run``. A failed check stops the case (``exec_stop``); an
    exception outside a row's rollout is the case's ``exec_error``. With ``replay``, G3's 2-seed replay (replay_rows6
    at the case's second reset seed; ``replay``, ``replay_seconds``): a row that differs or stops stops the case."""
    from robots.mujoco_lab import com_gap
    from robots.mujoco_lab import curve_graspgen as gg
    from robots.mujoco_lab.curve_poseset import _err, commit_std_base
    from robots.mujoco_lab.curve_sixdof import _checked_reset, _js

    out = dict(prep, exec_stop=None, exec_error=None, replay=None)
    if prep.get("stop") or prep.get("dropped"):
        return out
    env = prims.env
    st = gg._setup(prims, rec1, seed, world)
    if st["stop"]:
        return dict(out, exec_stop=st["stop"])
    reset, want, hull = st["reset_at"](0), rec1["fingerprint"], prep["hull_sha"]
    try:
        with replan_on(env):
            _, stop = _checked_reset(env, reset, com_gap.fingerprint, want, hull)
            if stop:
                return dict(out, exec_stop=stop)
            base = commit_std_base(env, prep["object"], st["destination"])
            if _js(base) != _js(prep["base"]):
                return dict(out, exec_stop=f"the base {base} differs from the prepass's")
            rows30 = [dict(r) for r in prep["rows30"]]
            rows6 = [dict(r) for r in prep["rows6"]]
            out.update(rows30=rows30, rows6=rows6)
            for r in rows30:
                if r["kept_all"] or r["kept_all_b"]:
                    new, stop = run_row30(env, reset, r, base, com_gap.fingerprint, want, hull)
                    if stop:
                        return dict(out, exec_stop=f"{r['id']}: {stop}")
                    r["run"] = new
            for r in rows6:
                if r["candidate"] is None:
                    continue
                new, stop = run_row6(env, reset, r, _pinch(entry, r), cache, base, com_gap.fingerprint, want, hull)
                if stop:
                    return dict(out, exec_stop=f"{r['id']}: {stop}")
                r["run"] = new
            if replay:
                t1 = time.time()
                rep = replay_rows6(env, st["reset_at"](1), rows30, rows6, entry, cache, base, com_gap.fingerprint, want, hull)
                out.update(replay=rep, replay_seconds=round(time.time() - t1, 2))
                if rep["stop"]:
                    return dict(out, exec_stop=f"replay: {rep['stop']}")
                if rep["mismatches"]:
                    return dict(out, exec_stop=f"replay: {len(rep['mismatches'])} rows differ at the second seed (G3)")
    except Exception as exc:  # noqa: BLE001  a case that raises outside a row's rollout is this case's error
        out["exec_error"] = _err(exc)
    return out


# ------------------------------------------------------------------ the unit's counts, the stop rule, the smoke report (no label)

def _total(counts: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, int]:
    return {k: sum(int(c.get(k, 0)) for c in counts) for k in keys}


def exec_metrics(done: list[dict[str, Any]]) -> dict[str, Any]:
    """Executed records' counts and times, no label: cases, skips, drops, stops and errors (prepass or execute), both
    funnels' totals, the ≤ 30° rows run (and those that run with a clearance flag, amendment 6) and their errors; the
    30–45° rows that passed the lab's path check (a candidate), their free moves by status ("ok", "miss": G3,
    "failed": cuRobo, by reason), G2 failures by check, row errors; G3's replay rows and mismatches; the case seconds
    (prepass and execute, the replay included)."""
    from collections import Counter

    rows30 = [r for rec in done for r in rec.get("rows30") or []]
    run30 = [r["run"] for r in rows30 if r.get("run") is not None]
    passed = [r for rec in done for r in rec.get("rows6") or [] if r.get("candidate") is not None]
    run6 = [r["run"] for r in passed if r.get("run") is not None]
    g2 = [x["unreachable"] for x in run6 if x.get("unreachable") is not None]
    rep = [rec["replay"] for rec in done if rec.get("replay")]
    secs = [float(rec.get("seconds") or 0.0) + float(rec.get("exec_seconds") or 0.0) for rec in done]
    return {"cases": len(done), "skipped": sum(bool(r.get("skipped")) for r in done), "dropped": sum(bool(r.get("dropped")) for r in done),
            "stops": sum(bool(r.get("stop") or r.get("exec_stop")) for r in done),
            "errors": sum(bool(r.get("error") or r.get("exec_error")) for r in done),
            "funnel30": _total([r["counts30"] for r in done if r.get("counts30")], FUNNEL30),
            "funnel6": _total([r["counts6"] for r in done if r.get("counts6")], FUNNEL6),
            "rows30_run": len(run30), "rows30_flagged_run": sum(r.get("run") is not None and r.get("clearance") is not None for r in rows30),
            "rows30_errors": sum(bool(x.get("error")) for x in run30),
            "path_passed": len(passed), "free": {k: sum(x.get("free") == k for x in run6) for k in ("ok", "miss", "failed")},
            "failed_by_reason": dict(sorted(Counter(x["free_failure"]["reason"] for x in run6
                                                    if x.get("free") == "failed" and x.get("free_failure")).items())),
            "g2_fail": len(g2), "g2_by_check": dict(sorted(Counter(u["check"] for u in g2).items())),
            "rows6_errors": sum(bool(x.get("error")) for x in run6),
            "replay_rows": sum(int(x["rows"]) for x in rep), "replay_mismatches": sum(len(x["mismatches"]) for x in rep),
            "case_seconds": round(sum(secs), 1), "case_seconds_max": max(secs, default=0.0)}


def stop_rule(m: dict[str, Any]) -> list[str]:
    """Design section 6's stop rule on exec_metrics' counts, with G3's two zeros (section 4, reading 7) and the cases
    that stopped or raised: the reasons the builder stops before the full run (empty: go on). Tracking (G2) fails on
    more than 10 % of the planned rows (free move "ok"); cuRobo fails on more than 10 % of the rows that passed the
    lab's path check; any plan-cache miss; any row that differs at the second seed; any case stop or error."""
    out = []
    ok, passed, failed = int(m["free"]["ok"]), int(m["path_passed"]), int(m["free"]["failed"])
    if ok and 10 * int(m["g2_fail"]) > ok:
        out.append(f"tracking (G2) fails on {m['g2_fail']} of {ok} planned rows (more than 10 %)")
    if passed and 10 * failed > passed:
        out.append(f"cuRobo fails on {failed} of {passed} rows that passed the lab's path check (more than 10 %)")
    if m["free"]["miss"]:
        out.append(f"G3: {m['free']['miss']} plan-cache misses (0 required)")
    if m["replay_mismatches"]:
        out.append(f"G3: {m['replay_mismatches']} rows differ at the second seed (0 required)")
    if m["stops"] or m["errors"]:
        out.append(f"{m['stops']} case stops and {m['errors']} case errors")
    return out


def plan_counts(plans: list[dict[str, Any]]) -> dict[str, int]:
    """The plan step's counts over its lines: cases, cases planned (a summary), its goals (planned, cached), and the
    planner processes that failed."""
    with_plan = [p["plan"] for p in plans if p.get("plan") is not None]
    return {"cases": len(plans), "with_plan": len(with_plan), **_total(with_plan, ("goals", "planned", "cached")),
            "errors": sum(bool(p.get("plan_error")) for p in plans)}


def read_provenance(d: str) -> dict[str, Any]:
    """The code and sim lines of a run dir's provenance.txt, as the launcher (run_curve.sh) writes them: the code
    commit and its dirty files, the sim folder, its last commit and its dirty files. A missing file or line gives None."""
    import re

    out: dict[str, Any] = {"code": None, "code_dirty": None, "sim_dir": None, "sim": None, "sim_dirty": None}
    path = os.path.join(d, "provenance.txt")
    if not os.path.exists(path):
        return out
    for line in open(path):
        c = re.match(r"code ([0-9a-f]{40}) .*dirty files (\d+)\s*$", line)
        s = re.match(r"sim (.+), last commit ([0-9a-f]{40}), dirty files (\d+)\s*$", line)
        if c:
            out.update(code=c[1], code_dirty=int(c[2]))
        if s:
            out.update(sim_dir=s[1], sim=s[2], sim_dirty=int(s[3]))
    return out


def provenance_gate(execute_dirs: list[str], plan_dir: str, sim: str = LEVEL_A_SIM, code: str = LEVEL_A_CODE) -> dict[str, Any]:
    """The level-A record's provenance gate (the coordinator's ruling of 9 Nov, after a resume script ran the execute
    pass on the launcher's default sim with EXIT 0 everywhere). Every execute unit, and the prepass unit it read (its
    run_config's ``prepass``), must name the code ``code`` and the sim ``sim`` in its provenance.txt, each with no dirty
    file. The plan step must name ``code`` with no dirty file. Its sim line is recorded, not checked: the plan reads no
    sim (cuRobo takes the request's world and the committed robot files). Anything else raises SystemExit with every
    failing dir. Returns what the report prints. The level-A summary calls it too, before it opens a record."""
    bad: list[str] = []

    def check(d: str, with_sim: bool) -> dict[str, Any]:
        p = read_provenance(d)
        why = [] if p["code"] == code and p["code_dirty"] == 0 else [f"code {p['code']}, dirty files {p['code_dirty']}"]
        if with_sim and not (p["sim"] == sim and p["sim_dirty"] == 0):
            why.append(f"sim {p['sim']} ({p['sim_dir']}), dirty files {p['sim_dirty']}")
        if why:
            bad.append(f"{d}: " + "; ".join(why))
        return p

    prepass: list[str] = []
    for d in execute_dirs:
        check(d, True)
        pre = json.load(open(os.path.join(d, "run_config.json"))).get("prepass")
        if not pre:
            bad.append(f"{d}: its run_config names no prepass run")
        elif pre not in prepass:
            prepass.append(pre)
            check(pre, True)
    p = check(plan_dir, False)
    if bad:
        raise SystemExit(f"provenance gate: not every unit ran on code {code} and sim {sim} with no dirty file:\n" + "\n".join(bad))
    return {"code": code, "sim": sim, "execute_units": len(execute_dirs), "prepass_units": len(prepass),
            "plan": {"code": p["code"], "sim": p["sim"], "sim_dir": p["sim_dir"]}}


def smoke_report(done: list[dict[str, Any]], plans: list[dict[str, Any]], full: dict[str, int]) -> dict[str, Any]:
    """The smoke's numbers, no label: exec_metrics and the stop rule; both funnels by object and in all; where the
    30–45° rows leave (``why``) and the clearance/reach6 disagreements, by object; the plan step's counts; and the full
    run's projection: each object's mean case time (the prepass and the first execute pass, not the replay) and mean
    plan time times its ``full`` cases (an object without a smoke case takes the largest mean)."""
    from collections import Counter, defaultdict

    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in done:
        by[r["object"]].append(r)
    funnels: dict[str, dict[str, dict[str, int]]] = {}
    for name, key, keys in (("funnel30", "counts30", FUNNEL30), ("funnel6", "counts6", FUNNEL6)):
        f = {o: _total([r[key] for r in rs if r.get(key)], keys) for o, rs in sorted(by.items())}
        funnels[name] = dict(f, all=_total(list(f.values()), keys))
    why = {o: dict(sorted(Counter(str(x.get("why")) for r in rs for x in r.get("rows6") or []).items())) for o, rs in sorted(by.items())}
    disagree = {o: sum(x.get("disagree") is not None for r in rs for x in r.get("rows6") or []) for o, rs in sorted(by.items())}
    first = {o: float(np.mean([float(r.get("seconds") or 0.0) + float(r.get("exec_seconds") or 0.0) - float(r.get("replay_seconds") or 0.0)
                               for r in rs])) for o, rs in by.items()}
    pby: dict[str, list[float]] = defaultdict(list)
    for p in plans:
        pby[p["object"]].append(float(p["seconds"]))
    gmean = {o: float(np.mean(v)) for o, v in pby.items()}
    m = exec_metrics(done)
    return {"counts": m, "stop": stop_rule(m), **funnels, "why6": why, "disagree": disagree, "plans": plan_counts(plans),
            "stops": [f"{r['object']} {r['idx']}: {r.get('stop') or r.get('exec_stop')}" for r in done if r.get("stop") or r.get("exec_stop")],
            "errors": [f"{r['object']} {r['idx']}: {r.get('error') or r.get('exec_error')}" for r in done if r.get("error") or r.get("exec_error")],
            "case_seconds": {o: round(v, 1) for o, v in sorted(first.items())}, "plan_seconds": {o: round(v, 1) for o, v in sorted(gmean.items())},
            "full_cases": dict(full),
            "projection_cpu_h": sum(first.get(o, max(first.values(), default=0.0)) * n for o, n in full.items()) / 3600,
            "projection_gpu_h": sum(gmean.get(o, max(gmean.values(), default=0.0)) * n for o, n in full.items()) / 3600}


def smoke_markdown(sm: dict[str, Any]) -> str:
    """The smoke report as markdown (counts only, no label)."""
    m = sm["counts"]
    L = ["# Sixdof rerun, level A: the smoke (counts only, no label)", ""]
    if sm.get("provenance"):
        g = sm["provenance"]
        L += [f"Provenance gate passed: code {g['code']} and sim {g['sim']}, no dirty file, on all {g['execute_units']} execute "
              f"units and their {g['prepass_units']} prepass units. The plan step: code {g['plan']['code']}; its sim line "
              f"{g['plan']['sim']} ({g['plan']['sim_dir']}) is not checked, because the plan reads no sim.", ""]
    L += [f"{m['cases']} cases ({m['dropped']} dropped, {m['stops']} stops, {m['errors']} case errors).", "",
         "## The stop rule (design section 6, with G3's zeros)", ""]
    L += [f"- STOP: {x}" for x in sm["stop"]] or ["- No reason to stop: the full run can follow after the clip review."]
    L += ["", "## The ≤ 30° rows (the pilot's funnel without the cap; ruling A's planner set)", "",
          "| object | " + " | ".join(FUNNEL30) + " |", "|---|" + "---|" * len(FUNNEL30)]
    L += [f"| {o} | " + " | ".join(str(v[k]) for k in FUNNEL30) + " |" for o, v in sm["funnel30"].items()]
    L += ["", f"Rows run {m['rows30_run']} ({m['rows30_flagged_run']} with a clearance flag, amendment 6), row errors {m['rows30_errors']}.",
          "", "## The 30–45° rows (level A: width rules, clearance, choose)", "",
          "| object | " + " | ".join(FUNNEL6) + " |", "|---|" + "---|" * len(FUNNEL6)]
    L += [f"| {o} | " + " | ".join(str(v[k]) for k in FUNNEL6) + " |" for o, v in sm["funnel6"].items()]
    L += ["", "Where the rows leave (the first failing step; None: to the planner):", ""]
    L += [f"- {o}: {w}; clearance/reach6 disagreements {sm['disagree'][o]}" for o, w in sm["why6"].items()]
    p = sm["plans"]
    L += ["", "## The free moves, G2 and G3", "",
          f"- Rows that passed the lab's path check: {m['path_passed']}. Free moves: {m['free']['ok']} planned, "
          f"{m['free']['failed']} cuRobo failures {m['failed_by_reason']}, {m['free']['miss']} cache misses (G3).",
          f"- G2 (tracking) failures: {m['g2_fail']} {m['g2_by_check']}. Row errors: {m['rows6_errors']}.",
          f"- G3 replay: {m['replay_rows']} rows at the second seed, {m['replay_mismatches']} differ.",
          f"- Plan step: {p['cases']} cases, {p['with_plan']} planned ({p['goals']} goals: {p['planned']} planned, {p['cached']} "
          f"cached), {p['errors']} planner process errors.", "",
          "## Projection", "",
          f"{sm['projection_cpu_h']:.2f} CPU-h and {sm['projection_gpu_h']:.2f} GPU-h for {sum(sm['full_cases'].values())} cases "
          f"(mean case seconds by object: {sm['case_seconds']}; mean plan seconds: {sm['plan_seconds']})."]
    if sm["stops"] or sm["errors"]:
        L += ["", "Stops and errors: " + "; ".join(sm["stops"] + sm["errors"])]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ the commands

def _json_default(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"{type(x).__name__} is not JSON serializable")


def _fresh(path: str) -> None:
    if os.path.exists(path):
        raise SystemExit(f"{path} exists: pick a new --out")


def _finished(d: str) -> None:
    if not os.path.exists(os.path.join(d, "run_metrics.json")):
        raise SystemExit(f"{d}: the run has not finished (no run_metrics.json)")


def _object_prims(bundle: str) -> tuple[Any, Any]:
    """One object's fold bundle as the pilot sets it up (curve_graspgen.graspgen_object): the absorbing attempt only, no
    probe models, the job's force noise. Returns (prims, the probe models to restore)."""
    from robots.mujoco_lab import calibrate as cal

    prims = cal._primitives(bundle, None, None)
    if getattr(prims, "reset_s", None) is not None:
        raise RuntimeError("the rerun scores the absorbing attempt only (this bundle restores the object after damage)")
    probes, prims.probe_models = getattr(prims, "probe_models", None), {}
    prims.force_noise = cal.job_force_noise(None, prims.manifest)
    prims.env.force_noise = prims.force_noise
    return prims, probes


def prepass_line(rec: dict[str, Any]) -> str:
    """A prepass record's log line: counts and times, no label."""
    c30, c6 = rec.get("counts30") or {}, rec.get("counts6") or {}
    return (f"{rec['object']} {rec['idx']}: <= 30 deg {c30.get('kept', 0)} kept by the pilot's cap, "
            f"{sum(bool(r.get('kept_all') or r.get('kept_all_b')) for r in rec.get('rows30') or [])} without it; 30-45 deg "
            f"{c6.get('level_a', 0)} rows, {c6.get('chosen', 0)} to the planner; {float(rec.get('seconds') or 0.0):.1f} s"
            + (f", dropped ({rec['skipped']})" if rec.get("skipped") else "") + (f", error {rec['error']}" if rec.get("error") else "")
            + (f", STOP {rec['stop']}" if rec.get("stop") else ""))


def exec_line(rec: dict[str, Any]) -> str:
    """An executed record's log line: rows, free moves, G2, G3, times; no label."""
    m = exec_metrics([rec])
    return (f"{rec['object']} {rec['idx']}: <= 30 deg rows run {m['rows30_run']}; 30-45 deg free moves {m['free']}, G2 failures "
            f"{m['g2_fail']}; replay {m['replay_rows']} rows, {m['replay_mismatches']} differ; {float(rec.get('exec_seconds') or 0.0):.1f} s"
            + (f", error {rec['exec_error']}" if rec.get("exec_error") else "") + (f", STOP {rec['exec_stop']}" if rec.get("exec_stop") else ""))


def prepass_object(job: tuple[str, list[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    """One object's prepass: its cases on its fold's bundle with the pilot's grasps (prepass_case each, timed); a stop
    ends the object (``stop``)."""
    from robots.mujoco_lab import curve_graspgen as gg
    from robots.mujoco_lab.prior import fragility_world

    o, recs1, s = job
    entries = gg.load_grasps(s["grasps"])
    prims, probes = _object_prims(s["bundle"])
    world = fragility_world(prims.manifest)
    out: dict[str, Any] = {"records": [], "stop": None}
    try:
        for rec1 in recs1:
            t0 = time.time()
            rec = prepass_case(prims, rec1, int(s["seed"]), world, entries.get((o, int(rec1["idx"]))))
            rec["seconds"] = round(time.time() - t0, 2)
            out["records"].append(rec)
            print(prepass_line(rec), file=sys.stderr, flush=True)
            if rec["stop"]:
                out["stop"] = f"{o} {rec1['idx']}: {rec['stop']}"
                break
    finally:
        prims.probe_models = probes
    return out


def execute_object(job: tuple[str, list[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]]) -> dict[str, Any]:
    """One object's execute pass: each (Stage 1 case, prepass record) on its fold's bundle with the pilot's grasps and
    the plan cache (execute_case, timed as ``exec_seconds``; G3's replay when the job says so); an execute-pass stop
    ends the object (``stop``)."""
    from robots.mujoco_lab import curve_graspgen as gg
    from robots.mujoco_lab.prior import fragility_world

    o, items, s = job
    entries = gg.load_grasps(s["grasps"])
    prims, probes = _object_prims(s["bundle"])
    world = fragility_world(prims.manifest)
    out: dict[str, Any] = {"records": [], "stop": None}
    try:
        for rec1, prep in items:
            t0 = time.time()
            rec = execute_case(prims, rec1, int(s["seed"]), world, prep, entries.get((o, int(rec1["idx"]))), s["cache"],
                               replay=bool(s["replay"]))
            rec["exec_seconds"] = round(time.time() - t0, 2)
            out["records"].append(rec)
            print(exec_line(rec), file=sys.stderr, flush=True)
            if rec["exec_stop"]:
                out["stop"] = f"{o} {rec1['idx']}: {rec['exec_stop']}"
                break
    finally:
        prims.probe_models = probes
    return out


def _run_jobs(fn: Any, jobs: list[Any], workers: int, rec_path: str) -> str | None:
    """The object jobs in order (``workers`` 1) or in a spawn pool, each job's records appended to ``rec_path`` as it
    ends; the first job that stops ends the run. Returns that stop or None."""
    import multiprocessing as mp

    stop = None
    with open(rec_path, "w") as f:
        pool = None if workers == 1 else mp.get_context("spawn").Pool(workers)
        try:
            for res in (map(fn, jobs) if pool is None else pool.imap_unordered(fn, jobs)):
                for r in res["records"]:
                    f.write(json.dumps(r, default=_json_default) + "\n")
                f.flush()
                if res["stop"]:
                    stop = res["stop"]
                    break
        finally:
            if pool is not None:
                (pool.terminate if stop else pool.close)()
                pool.join()
    return stop


def _write_metrics(out: str, name: str, config: dict[str, Any], metrics: dict[str, Any]) -> None:
    from robots.mujoco_lab.tracking import log_run

    metrics["wandb"] = log_run(name, config, metrics)
    with open(os.path.join(out, "run_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print(f"done: {json.dumps(metrics)}", flush=True)


def prepass_cmd(a: Any) -> str:
    """The prepass (CPU): the asked cases of Stage 1 (curve_graspgen.select_cases) on the pilot's grasps, by object
    (prepass_object). Refuses an existing --out and a pilot run (--pilot) with other grasps, Stage 1, bundle or seed
    (ruling 4). prepass_records.jsonl, run_config.json and run_metrics.json (counts and times only; wandb). Exit 3 at a
    stop. Returns --out."""
    from collections import defaultdict

    from robots.mujoco_lab import curve_graspgen as gg
    from robots.mujoco_lab.curve import _peak_rss_mb, _sha
    from robots.mujoco_lab.curve_sixdof import _js
    from robots.mujoco_lab.estimation_check import _revision
    from robots.mujoco_lab.identify_fit import require_bundle_generator

    os.makedirs(a.out, exist_ok=True)
    rec_path = os.path.join(a.out, "prepass_records.jsonl")
    _fresh(rec_path)
    recs1, cfg1, dirs = gg._stage1(a)
    generator = require_bundle_generator(cfg1["bundle"])
    mine = {"grasps_index_sha256": _sha(os.path.join(a.grasps, "index.jsonl")), "bundle": cfg1["bundle"], "seed": cfg1["seed"],
            "stage1_records_sha256": {d: _sha(os.path.join(d, "records.jsonl")) for d in dirs}}
    pilot = json.load(open(os.path.join(a.pilot, "run_config.json")))
    for k, v in mine.items():
        if _js(pilot.get(k)) != _js(v):
            raise SystemExit(f"the pilot's {k} is not this run's (ruling 4: the pilot's grasps, Stage 1, bundle and seed)")
    recs = gg.select_cases(recs1, a.cases, a.objects, a.sides, a.per_object)
    config = {**vars(a), **_revision(), **mine, "generator": generator, "pilot_code_revision": pilot.get("code_revision"),
              "cases": [[r["object"], int(r["idx"])] for r in recs], "check_speeds": CHECK_SPEEDS, "bins": [VERTICAL_BIN, LEVEL_A_BIN]}
    with open(os.path.join(a.out, "run_config.json"), "w") as f:
        json.dump(config, f, indent=1)
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        by[r["object"]].append(r)
    jobs = [(o, items, {"bundle": os.path.join(cfg1["bundle"], "folds", f"fold_{items[0]['fold']}"), "seed": cfg1["seed"], "grasps": a.grasps})
            for o, items in by.items()]
    t0 = time.time()
    stop = _run_jobs(prepass_object, jobs, int(a.workers), rec_path)
    done = [json.loads(line) for line in open(rec_path)]
    metrics = dict(exec_metrics(done), requests=sum(r.get("request") is not None for r in done),
                   goals=sum(len(r["request"]["goals"]) for r in done if r.get("request")), of_cases=len(recs), stop=stop,
                   wall_seconds=round(time.time() - t0, 1), peak_rss_mb=_peak_rss_mb())
    _write_metrics(a.out, "sixdof-rerun-prepass", config, metrics)
    if stop:
        raise SystemExit(3)
    return a.out


def plan_cmd(a: Any) -> str:
    """The plan step (GPU): every prepass record of the finished --prepass runs, in turn, through plan_prepass (one
    cuRobo process per case with a request, one at a time) into the --cache. plan_records.jsonl (one line per case:
    the process's summary or failure, seconds), run_config.json, run_metrics.json (plan_counts; wandb). Exit 3 when a
    planner process failed. Returns --out."""
    from robots.mujoco_lab.curve import _sha
    from robots.mujoco_lab.estimation_check import _revision

    os.makedirs(a.out, exist_ok=True)
    rec_path = os.path.join(a.out, "plan_records.jsonl")
    _fresh(rec_path)
    dirs = a.prepass.split(",")
    for d in dirs:
        _finished(d)
    paths = [os.path.join(d, "prepass_records.jsonl") for d in dirs]
    recs = [json.loads(line) for p in paths for line in open(p)]
    config = {**vars(a), **_revision(), "prepass_records_sha256": {p: _sha(p) for p in paths},
              "cases": [[r["object"], int(r["idx"])] for r in recs]}
    with open(os.path.join(a.out, "run_config.json"), "w") as f:
        json.dump(config, f, indent=1)
    t0, lines = time.time(), []
    with open(rec_path, "w") as f:
        for rec in recs:
            t1 = time.time()
            got = plan_prepass(rec, a.cache)
            line = {"object": rec["object"], "idx": int(rec["idx"]), "plan": got["plan"], "plan_error": got["plan_error"],
                    "seconds": round(time.time() - t1, 2)}
            f.write(json.dumps(line) + "\n")
            f.flush()
            lines.append(line)
            print(f"{rec['object']} {rec['idx']}: plan {line['plan']}, error {line['plan_error']}, {line['seconds']:.1f} s",
                  file=sys.stderr, flush=True)
    metrics = dict(plan_counts(lines), wall_seconds=round(time.time() - t0, 1))
    _write_metrics(a.out, "sixdof-rerun-plan", config, metrics)
    if metrics["errors"]:
        raise SystemExit(3)
    return a.out


def execute_cmd(a: Any) -> str:
    """The execute pass (CPU, lab only): the records of one finished --prepass run with their Stage 1 cases, by object
    (execute_object), on the run's bundle, seed and grasps, with the --cache's plans only (G3), and G3's replay with
    --replay. Refuses an unfinished prepass or plan step and a case with a request but no plan record.
    exec_records.jsonl, run_config.json, run_metrics.json (exec_metrics; wandb). Exit 3 at a stop. Returns --out."""
    from collections import defaultdict

    from robots.mujoco_lab.curve import _peak_rss_mb, _sha, load_stage1
    from robots.mujoco_lab.estimation_check import _revision

    os.makedirs(a.out, exist_ok=True)
    rec_path = os.path.join(a.out, "exec_records.jsonl")
    _fresh(rec_path)
    _finished(a.prepass)
    _finished(a.plan)
    pcfg = json.load(open(os.path.join(a.prepass, "run_config.json")))
    ppath, lpath = os.path.join(a.prepass, "prepass_records.jsonl"), os.path.join(a.plan, "plan_records.jsonl")
    preps = [json.loads(line) for line in open(ppath)]
    planned = {(x["object"], int(x["idx"])) for x in (json.loads(line) for line in open(lpath))}
    lost = [f"{p['object']}:{p['idx']}" for p in preps if p.get("request") is not None and (p["object"], int(p["idx"])) not in planned]
    if lost:
        raise SystemExit(f"no plan record for the cases {lost}: run the plan step on this prepass first")
    recs1, _ = load_stage1(list(pcfg["stage1_records_sha256"]))
    by1 = {(r["object"], int(r["idx"])): r for r in recs1}
    config = {**vars(a), **_revision(), "bundle": pcfg["bundle"], "seed": pcfg["seed"], "grasps": pcfg["grasps"],
              "prepass_code_revision": pcfg.get("code_revision"), "prepass_records_sha256": _sha(ppath), "plan_records_sha256": _sha(lpath),
              "cases": [[p["object"], int(p["idx"])] for p in preps]}
    with open(os.path.join(a.out, "run_config.json"), "w") as f:
        json.dump(config, f, indent=1)
    by: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for p in preps:
        by[p["object"]].append((by1[(p["object"], int(p["idx"]))], p))
    jobs = [(o, items, {"bundle": os.path.join(pcfg["bundle"], "folds", f"fold_{items[0][0]['fold']}"), "seed": pcfg["seed"],
                        "grasps": pcfg["grasps"], "cache": a.cache, "replay": bool(a.replay)}) for o, items in by.items()]
    t0 = time.time()
    stop = _run_jobs(execute_object, jobs, int(a.workers), rec_path)
    done = [json.loads(line) for line in open(rec_path)]
    metrics = dict(exec_metrics(done), of_cases=len(preps), stop=stop, wall_seconds=round(time.time() - t0, 1), peak_rss_mb=_peak_rss_mb())
    _write_metrics(a.out, "sixdof-rerun-execute", config, metrics)
    if stop:
        raise SystemExit(3)
    return a.out


def smoke_cmd(a: Any) -> dict[str, Any]:
    """The smoke report on finished execute runs (--execute) and their plan step (--plan): smoke_report with the full
    run's cases per object from --full-cases (a comma-separated object:idx file, the pilot's cases.txt). The provenance
    gate runs first: a unit off the level-A code or sim stops the report before it writes or logs anything. The report
    prints what the gate checked. smoke_summary.{json,md} in --out (refuses an existing one), wandb. Exit 3 when the stop
    rule fires."""
    from collections import Counter

    from robots.mujoco_lab.tracking import log_run

    os.makedirs(a.out, exist_ok=True)
    jpath = os.path.join(a.out, "smoke_summary.json")
    _fresh(jpath)
    dirs = a.execute.split(",")
    for d in dirs + [a.plan]:
        _finished(d)
    gate = provenance_gate(dirs, a.plan)
    done = [json.loads(line) for d in dirs for line in open(os.path.join(d, "exec_records.jsonl"))]
    plans = [json.loads(line) for line in open(os.path.join(a.plan, "plan_records.jsonl"))]
    full = Counter(c.strip().rsplit(":", 1)[0] for c in open(a.full_cases).read().split(",") if c.strip())
    sm = smoke_report(done, plans, dict(sorted(full.items())))
    sm["provenance"] = gate
    sm["wandb"] = log_run("sixdof-rerun-smoke", {"execute": a.execute, "plan": a.plan, "full_cases": a.full_cases},
                          {"counts": sm["counts"], "plans": sm["plans"], "stop": sm["stop"], "projection_cpu_h": sm["projection_cpu_h"],
                           "projection_gpu_h": sm["projection_gpu_h"]})
    with open(jpath, "w") as f:
        json.dump(sm, f, indent=1)
    md = smoke_markdown(sm)
    with open(os.path.join(a.out, "smoke_summary.md"), "w") as f:
        f.write(md)
    print(md, flush=True)
    if sm["stop"]:
        raise SystemExit(3)
    return sm


def main(argv: list[str] | None = None) -> Any:
    import argparse

    ap = argparse.ArgumentParser(prog="curve_sixdof_rerun", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepass", help="the funnels and the free-move requests of the asked cases (CPU)")
    p.add_argument("--stage1", required=True, help="comma-separated Stage 1 run dirs")
    p.add_argument("--grasps", required=True, help="the pilot's request step's output dir")
    p.add_argument("--pilot", required=True, help="a pilot run dir: its grasps, Stage 1, bundle and seed must be this run's")
    p.add_argument("--cases", default=None, help="comma-separated object:idx cases")
    p.add_argument("--objects", default=None, help="comma-separated objects (with --sides)")
    p.add_argument("--sides", default="test")
    p.add_argument("--per-object", type=int, default=None, help="the first N cases of each object by idx")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", required=True)
    q = sub.add_parser("plan", help="the free moves of finished prepass runs, one cuRobo process per case, in turn (GPU)")
    q.add_argument("--prepass", required=True, help="comma-separated finished prepass run dirs")
    q.add_argument("--cache", required=True, help="the plan cache dir")
    q.add_argument("--out", required=True)
    e = sub.add_parser("execute", help="the rows of one finished prepass run in the lab, cached plans only (CPU)")
    e.add_argument("--prepass", required=True, help="a finished prepass run dir")
    e.add_argument("--plan", required=True, help="the finished plan step that planned its requests")
    e.add_argument("--cache", required=True, help="the plan cache dir")
    e.add_argument("--replay", action="store_true", help="G3's 2-seed replay: every row again at the second seed")
    e.add_argument("--workers", type=int, default=1)
    e.add_argument("--out", required=True)
    m = sub.add_parser("smoke-report", help="the smoke report (counts only, no label) and the stop rule")
    m.add_argument("--execute", required=True, help="comma-separated finished execute run dirs")
    m.add_argument("--plan", required=True, help="their finished plan step")
    m.add_argument("--full-cases", required=True, help="the full run's cases: a comma-separated object:idx file")
    m.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"prepass": prepass_cmd, "plan": plan_cmd, "execute": execute_cmd, "smoke-report": smoke_cmd}[a.cmd](a)


if __name__ == "__main__":
    main()
