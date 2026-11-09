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

import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from robots.mujoco_lab import sixdof

VERTICAL_BIN = sixdof.BINS[0]         # "≤ 30°": the pilot's vertical rows, its mapping and executor
LEVEL_A_BIN = sixdof.BINS[1]          # "30–45°": the rows level A adds; the pilot's vertical rows stay its own
CHECK_SPEEDS = {"v_lift": 0.1, "v_transport": 0.5, "release_height": None}   # the speeds the pilot's reach() checks at


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


def execute_case(prims: Any, rec1: dict[str, Any], seed: int, world: Any, prep: dict[str, Any], entry: dict[str, Any],
                 cache: str) -> dict[str, Any]:
    """The execute pass of one prepass record (lab only): nothing for a case that stopped or dropped in the prepass;
    else the case's setup (curve_graspgen._setup), a reset with the prepass's checks (Stage 1's fingerprint, the
    record's hulls, its base), then every row the record keeps, each after its own reset, with the executor's re-plan
    at the hold on (replan_on): the ≤ 30° rows that either width rule keeps without the cap (kept_all, kept_all_b;
    run_row30), then the 30–45° rows with a candidate (run_row6 on the row's pinch, or on its twin when the candidate
    is the twin's). Each row's fields go into the row as ``run``. A failed check stops the case (``exec_stop``); an
    exception outside a row's rollout is the case's ``exec_error``."""
    from robots.mujoco_lab import com_gap
    from robots.mujoco_lab import curve_graspgen as gg
    from robots.mujoco_lab.curve_poseset import _err, commit_std_base
    from robots.mujoco_lab.curve_sixdof import _checked_reset, _js

    out = dict(prep, exec_stop=None, exec_error=None)
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
                p = sixdof.from_graspgen(entry["grasps"][int(r["id"][1:])], width=float(r["width"]))
                new, stop = run_row6(env, reset, r, sixdof.twin(p) if r["candidate"]["twin"] else p, cache, base,
                                     com_gap.fingerprint, want, hull)
                if stop:
                    return dict(out, exec_stop=f"{r['id']}: {stop}")
                r["run"] = new
    except Exception as exc:  # noqa: BLE001  a case that raises outside a row's rollout is this case's error
        out["exec_error"] = _err(exc)
    return out
