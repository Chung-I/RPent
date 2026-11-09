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
