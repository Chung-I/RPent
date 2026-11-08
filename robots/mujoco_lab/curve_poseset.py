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

"""The pose-set ablation, stage A (daily-logs design 2026-11-08-pose-set-ablation-design.md, amended in its section 3.1).

Ten pose rows at commit_std's force and speeds: pinch fractions 0.1 to 0.9 of the long axis, the pads closing across
the axis (as Stage 2) or along it (the hand turned by -pi/2; on an upright object the other horizontal side). The three
current rows (across at 0.3, 0.5 and 0.7) keep the menu's rule (the width rule and reach()); the seven new rows also
need both ends of the pinch between the open pads, in the pad slab and in the column above it (the centring rule).
Each feasible row runs once at the case's own reset seed (the attempt is deterministic per case: the reset seed sets
only the sensor-noise streams). A case is achievable when any row places it: the capability ceiling, an upper bound
with a perfect selector, over the ten rows and over the three current rows (the matched baseline).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from rpent.skills.pick_place.protocol import IRREVERSIBLE

FRACS = (0.1, 0.3, 0.5, 0.7, 0.9)
CURRENT = (0.3, 0.5, 0.7)                  # the adapter's default proposals: the Stage 2 menu's pinches
TURN = {"across": 0.0, "along": -math.pi / 2}
# the selector's order: across first, then nearest the middle, the lower fraction first (rounded: abs(0.7 - 0.5) is
# 0.19999999999999996 in floats, so the menu's mid_proposal takes 0.7 before 0.3 when 0.5 is unusable)
ROW_ORDER = tuple((d, f) for d in TURN for f in sorted(FRACS, key=lambda f: (round(abs(f - 0.5), 9), f)))
UPRIGHT_DEG = 45.0                         # an axis tilted more than this from the table is upright
OFF_CENTRE = "an end of the pinch lies past an open pad"
OFF_CENTRE_COLUMN = "the object reaches past an open pad above the pinch"
LIFT_FAILS = ("not_lifted", "missed")      # the labels of a pinch that never raised the object
OUTCOME = ("label", "artifact", "error", "damage", "drop_z", "loss_phase", "max_z", "qpos_sha", "trace_sha", "seconds")
SAME = ("label", "artifact", "error", "damage", "drop_z", "loss_phase", "max_z", "qpos_sha", "trace_sha")   # the replay gate
POSE_CHANGES = ("the oracle gives up", "no pose succeeds", "the oracle's pose", "fraction change only", "direction change only",
                "fraction and direction change")                   # design reading 6, the smallest change first
HAMMER_CLASSES = ("placed by a pose", "lifted, never placed", "not lifted by any pose", "no clean simulation",
                  "no feasible pose")                               # design reading 4's hammer line
KINDS = ("hammers and tools", "boxes", "cans and jars", "bottles", "fruit", "bowls and containers", "other")
_KINDS = {
    "hammers and tools": "handal__hammer handal__hammer_1 handal__hammer_2 handal__hammer_3 handal__hammer_4 handal__hammer_5 "
                         "handal__hammer_6 handal__hammer_7 handal__hammer_8 handal__ladle handal__measuring_spoon "
                         "handal__pink_spaghetti_spoon handal__red_serving_spoon handal__green_serving_spoon "
                         "handal__measuring_cups handal__measuring_cups_1 hot3d__spatula hot3d__wooden_spoons "
                         "hot3d__potato_masher ycb__clamp ycb__spring_clamp ycb__dry_erase_marker ycb__scissors "
                         "ycb__cordless_drill",
    "boxes": "hope__butter hope__cream_cheese hope__macaroni_and_cheese hope__milk_carton hope__oatmeal_raisin_cookies "
             "hope__orange_juice_carton hope__popcorn_box hope__raisin_box hot3d__frozen_waffles ycb__jello ycb__sugar_box "
             "hope__chocolate_pudding_mix hope__granola_bars hope__spaghetti hot3d__frozen_vegetable_block ycb__cheez_it "
             "ycb__chocolate_pudding hot3d__storage_box",
    "cans and jars": "hope__canned_mushrooms hope__canned_peaches hope__parmesan_cheese_canister hope__pineapple_slices_can "
                     "hope__tomato_sauce_can ycb__spam_can ycb__tuna_can hope__canned_tuna hope__corn_can "
                     "hope__green_beans_can hope__peas_and_carrots hope__pitted_cherries hot3d__soup_can ycb__coffee_can "
                     "ycb__tomato_soup_can",
    "bottles": "hope__mayonnaise_bottle ycb__soft_scrub hope__bbq_sauce_bottle hope__ketchup_bottle hope__mustard_bottle "
               "hope__ranch_dressing hot3d__bbq_sauce_bottle ycb__mustard",
    "fruit": "fruits_veggies__avocado fruits_veggies__orange1 fruits_veggies__lemon1 fruits_veggies__lime "
             "fruits_veggies__red_onion ycb__banana",
    "bowls and containers": "hot3d__ceramic_mug hot3d__coffee_pot hot3d__pitcher ycb__bowl ycb__bowl2 ycb__pitcher hot3d__mug "
                            "hot3d__wooden_bowl ycb__mug hope__yogurt_cup hot3d__clay_plates",
    "other": "hot3d__birdhouse hot3d__foam_roller hot3d__glasses hot3d__lizard_figurine hot3d__rubiks_cube hot3d__smartphone "
             "hot3d__computer_mouse hot3d__dumbbell hot3d__keyboard hot3d__megaphone hot3d__remote_control ycb__brick "
             "ycb__wood_block",
}
KIND_OF = {o: k for k, names in _KINDS.items() for o in names.split()}   # design reading 4's kinds, fixed before any record


def row_id(direction: str, f: float) -> str:
    return f"{direction}_{f:g}"


def centred(extent: Any, limit: float) -> bool:
    """Both ends of a signed extent (lo, hi) within +-limit of the pinch point (nan: no)."""
    lo, hi = float(extent[0]), float(extent[1])
    return bool(np.isfinite(lo) and np.isfinite(hi) and lo >= -limit and hi <= limit)


def pose_rows(env: Any, destination: str) -> list[dict[str, Any]]:
    """The ten pose rows in ROW_ORDER on the env's current state: the proposal, its width, its signed extent in the pad
    slab and in the column above it, and whether it is feasible. A current row (across at 0.3, 0.5, 0.7) needs the
    width rule and reach(), as the menu; a new row also needs the centring rule (both extents within half the widest
    pinch: the open pads' inner faces less the width rule's margin). reach() runs only on a row that passes the rest."""
    from rpent.skills.pick_place.menu import unusable_reason

    wmax = float(env.max_pinch_width)
    limit = wmax / 2
    by = {d: {p.s: p for p in env.proposals(FRACS, turn=t)} for d, t in TURN.items()}
    out = []
    for d, f in ROW_ORDER:
        p = by[d][f]
        current = d == "across" and f in CURRENT
        slab, column = env.pinch_extent(p.point, p.yaw), env.column_extent(p.point, p.yaw)
        cen_slab, cen_col = centred(slab, limit), centred(column, limit)
        why = unusable_reason(p.width, wmax)
        if why is None and not current and not (cen_slab and cen_col):
            why = OFF_CENTRE if not cen_slab else OFF_CENTRE_COLUMN
        reach = None
        if why is None:
            r = env.reach(p, destination)
            reach = "ok" if r is None else f"{r.get('reason')} ({r.get('phase')})"
            if r is not None:
                why = f"unreachable: {reach}"
        out.append({"id": row_id(d, f), "dir": d, "f": float(f), "current": current, "point": [float(x) for x in p.point],
                    "yaw": float(p.yaw), "width": float(p.width), "slab": [float(x) for x in slab],
                    "column": [float(x) for x in column], "centred": cen_slab and cen_col, "reach": reach,
                    "feasible": why is None, "why": why})
    return out


def _sha_of(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _round9(x: Any) -> Any:
    if isinstance(x, float):
        return round(x, 9)
    if isinstance(x, dict):
        return {k: _round9(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round9(v) for v in x]
    return x


SENSOR_FIELDS = ("fz_joint",)              # trace readings with sensor noise (the torque stream, set by the reset seed)


def trace_sha(trace: Any) -> str:
    """A hash of every physical field of every trace step (SENSOR_FIELDS left out), floats to 9 digits (the final
    state's hash rounds qpos the same way)."""
    return _sha_of(_round9([{k: v for k, v in t.items() if k not in SENSOR_FIELDS} for t in trace]))


def simulate_rows(env: Any, reset: Any, rows: list[dict[str, Any]], base: dict[str, Any], fingerprint: Any,
                  want: str) -> str | None:
    """Run each feasible row once, in place: reset() (the case's own seed), the reset's fingerprint must be ``want``
    (else the case stops: the reason is returned), the proposal fetched again after the reset, then env.execute at
    ``base`` (commit_std's force, speeds, destination and release height). Each simulated row gets its label, the
    artifact flag (also when ``reset`` returns False: the object was not at rest), the damage and loss fields, the
    highest rise, hashes of the final state and of the trace, and its seconds; an exception is the row's error."""
    for r in rows:
        if not r["feasible"]:
            continue
        at_rest = reset()
        if fingerprint(env) != want:
            return f"{r['id']}: the reset's fingerprint differs from Stage 1's"
        t0 = time.time()
        r.update(label=None, artifact=False, error=None)
        try:
            p = {q.s: q for q in env.proposals(FRACS, turn=TURN[r["dir"]])}[r["f"]]
            if [float(x) for x in p.point] != r["point"] or float(p.yaw) != r["yaw"]:
                return f"{r['id']}: the pinch after the reset differs from the first reset's"
            ro = env.execute(p, base["force"], base["v_lift"], base["v_transport"], base["destination"], base["release_height"],
                             a_transport=base["a_transport"])
            d = getattr(getattr(env, "sim", None), "d", None)
            r.update(label=ro.label, artifact=bool(getattr(ro, "artifact", False)) or at_rest is False,
                     damage=getattr(ro, "damage", None), drop_z=getattr(ro, "drop_z", None), loss_phase=getattr(ro, "loss_phase", None),
                     max_z=getattr(ro, "max_z", None),
                     qpos_sha=None if d is None else hashlib.sha256(np.round(np.asarray(d.qpos, dtype=float), 9).tobytes()).hexdigest()[:16],
                     trace_sha=trace_sha(getattr(ro, "trace", [])))
        except Exception as exc:  # noqa: BLE001  a simulator fault is this row's error
            r["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        r["seconds"] = round(time.time() - t0, 2)
    return None


def ceiling(rows: list[dict[str, Any]], current_only: bool = False) -> dict[str, Any]:
    """The capability ceiling over the feasible rows (the current ones only with ``current_only``), in episode form
    (curve.episode's keys): success when a clean row (no artifact, no error) placed the object; the selected row is the
    first clean success in ROW_ORDER, else the first clean row without an irreversible label, else the first clean row;
    no feasible row is a give-up. ``determined``: a clean row succeeds, or every row is clean."""
    cand = [r for r in rows if r["feasible"] and (r["current"] or not current_only)]
    out: dict[str, Any] = {"success": False, "irr": False, "give_up": not cand, "attempts": 0, "labels": [], "rows": [],
                           "n_rows": len(cand), "n_success": 0, "determined": True}
    if not cand:
        return out
    clean = [r for r in cand if r.get("label") is not None and not r.get("artifact") and not r.get("error")]
    succ = [r for r in clean if r["label"] == "placed_stable"]
    sel = succ[0] if succ else next((r for r in clean if r["label"] not in IRREVERSIBLE), clean[0] if clean else None)
    out.update(success=bool(succ), n_success=len(succ), determined=bool(succ) or len(clean) == len(cand))
    if sel is not None:
        out.update(attempts=1, labels=[sel["label"]], rows=[sel["id"]], irr=sel["label"] in IRREVERSIBLE)
    return out


def replay_rows(env: Any, reset: Any, rows: list[dict[str, Any]], base: dict[str, Any], fingerprint: Any,
                want: str) -> dict[str, Any]:
    """The 2-seed determinism gate (design 3.1): the feasible rows once more, in reverse ROW_ORDER, on copies without
    their outcomes (``reset`` at another reset seed), each against its first run on SAME. Returns the rows replayed, the
    mismatches ({"id", "first", "replay"}: SAME's values) and simulate_rows's stop."""
    twin = [{k: v for k, v in r.items() if k not in OUTCOME} for r in reversed(rows) if r["feasible"]]
    stop = simulate_rows(env, reset, twin, base, fingerprint, want)
    first = {r["id"]: r for r in rows}
    done = [t for t in twin if "seconds" in t]
    bad = [{"id": t["id"], "first": [first[t["id"]].get(k) for k in SAME], "replay": [t.get(k) for k in SAME]}
           for t in done if [first[t["id"]].get(k) for k in SAME] != [t.get(k) for k in SAME]]
    return {"rows": len(done), "mismatches": bad, "stop": stop}


# ---------------------------------------------------------------- the case's fields and gates

def stage1_action(rec1: dict[str, Any], rid: str) -> dict[str, Any] | None:
    """The action of a Stage 1 row (its key's JSON, with s), or None when the menu has no such row."""
    for r in rec1.get("rows") or []:
        if r["id"] == rid:
            return json.loads(r["action"].split("|", 1)[1])
    return None


def commit_std_base(env: Any, o: str, destination: str) -> dict[str, Any]:
    """commit_std's action without the pinch: the standard squeeze of the nominal mass and menu.STD's speeds (menu._rows)."""
    from rpent.skills.pick_place.menu import STD
    from rpent.skills.pick_place.tools import standard_force

    return {"force": standard_force(env.nominal_mass(o)), "v_lift": STD["v_lift"], "v_transport": STD["v_transport"],
            "a_transport": STD["a_transport"], "destination": destination, "release_height": None}


def base_gate(rec1: dict[str, Any], base: dict[str, Any]) -> str | None:
    """Why the pose set's base is not Stage 1's commit_std action at one of the current pinches (None: it is, or the
    menu has no commit_std row)."""
    a = stage1_action(rec1, "commit_std")
    if a is None:
        return None
    s = a.pop("s", None)
    if a != base:
        return f"commit_std's action {a} is not the pose set's base {base}"
    if s not in CURRENT:
        return f"commit_std's pinch at s {s} is not a current pinch"
    return None


def balance_fraction(props: dict[str, Any], com_obj: Any) -> float:
    """The CoM's place along the grasp axis as a pinch fraction (grasp.axis_points' frame: 0 at the axis_lo end)."""
    ax, c = np.asarray(props["axis"], dtype=float), np.asarray(props["centre"], dtype=float)
    return float(((np.asarray(com_obj, dtype=float) - c) @ ax - float(props["axis_lo"])) / float(props["length"]))


def tilt_deg(R: Any, axis: Any) -> float:
    """The long axis's angle to the table (0: lying, 90: upright)."""
    axw = np.asarray(R, dtype=float) @ np.asarray(axis, dtype=float)
    return float(np.degrees(np.arcsin(min(1.0, abs(float(axw[2]))))))


def _clean(r: dict[str, Any]) -> bool:
    return bool(r["feasible"]) and r.get("label") is not None and not r.get("artifact") and not r.get("error")


def pose_change(rows: list[dict[str, Any]], oracle_s: float | None) -> str:
    """Design reading 6 on the set of poses that place the object (clean rows), against the oracle arm's first pinch
    (across at ``oracle_s``; None: the oracle gave up): the smallest change that succeeds, in POSE_CHANGES' order."""
    if oracle_s is None:
        return POSE_CHANGES[0]
    succ = {(r["dir"], r["f"]) for r in rows if _clean(r) and r["label"] == "placed_stable"}
    if not succ:
        return POSE_CHANGES[1]
    if ("across", oracle_s) in succ:
        return POSE_CHANGES[2]
    if any(d == "across" for d, _ in succ):
        return POSE_CHANGES[3]
    return POSE_CHANGES[4] if ("along", oracle_s) in succ else POSE_CHANGES[5]


def nearest_row(rows: list[dict[str, Any]], com_f: float) -> dict[str, Any] | None:
    """The clean row whose fraction is nearest the balance point (to 9 digits; a tie goes to the first in ROW_ORDER)."""
    clean = [r for r in rows if _clean(r)]
    return min(clean, key=lambda r: round(abs(r["f"] - com_f), 9)) if clean else None


def hammer_class(rows: list[dict[str, Any]]) -> str:
    """Design reading 4's hammer line for one case: placed by a pose, lifted (a label other than LIFT_FAILS) but never
    placed, not lifted by any pose, no clean simulation, or no feasible pose."""
    if not any(r["feasible"] for r in rows):
        return HAMMER_CLASSES[4]
    clean = [r for r in rows if _clean(r)]
    if not clean:
        return HAMMER_CLASSES[3]
    if any(r["label"] == "placed_stable" for r in clean):
        return HAMMER_CLASSES[0]
    return HAMMER_CLASSES[1] if any(r["label"] not in LIFT_FAILS for r in clean) else HAMMER_CLASSES[2]


def stage2_gate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Design 3.1's gate of the direct execute against the pipeline: over joined cases ({"pose", "stage1", "stage2"}),
    every Stage 2 simulation at the main seed whose first row has the pose set's base action at an across pinch s must
    have the label and artifact flag of the pose row across at s. A matching Stage 2 row whose pose row was not
    simulated is counted as missing."""
    compared, missing, bad = 0, 0, []
    for c in cases:
        p, s1, s2 = c["pose"], c["stage1"], c["stage2"]
        if not p.get("base"):
            continue
        across = {r["f"]: r for r in p["rows"] if r["dir"] == "across"}
        for sim in s2.get("sims") or []:
            if sim["seed_k"] != 0:
                continue
            a = stage1_action(s1, sim["first"])
            if a is None:
                continue
            s = a.pop("s", None)
            if a != p["base"] or s not in across:
                continue
            r = across[s]
            if "label" not in r:
                missing += 1
                continue
            compared += 1
            x, y = [r["label"], bool(r["artifact"])], [sim["label"], bool(sim["artifact"])]
            if x != y:
                bad.append({"object": p["object"], "idx": int(p["idx"]), "row": r["id"], "stage2_row": sim["first"], "pose": x,
                            "stage2": y})
    return {"compared": compared, "missing": missing, "mismatches": bad}


def off_centre_current(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """The disclosure of design 3.1: the feasible current rows that the centring rule would refuse, by object, and the
    cases whose every feasible current row it would refuse."""
    n, objs, every = 0, Counter(), 0
    for rec in recs:
        cur = [r for r in rec.get("rows") or [] if r["current"] and r["feasible"]]
        flagged = [r for r in cur if not r["centred"]]
        n += len(flagged)
        if flagged:
            objs[rec["object"]] += len(flagged)
        every += bool(cur) and len(flagged) == len(cur)
    return {"rows": n, "objects": dict(objs), "cases_all_flagged": every, "cases": len(recs)}


# ---------------------------------------------------------------- one case, the run

KEEP = ("object", "idx", "fold", "side", "class", "law", "fillable", "thin", "destination", "status", "fingerprint")


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:300]}"


def poseset_case(prims: Any, rec1: dict[str, Any], seed: int, world: Any, replay: bool = False) -> dict[str, Any]:
    """One case: Stage 2's skip rules on the Stage 1 record; curve.stage1_case's reset at the case's own seed (its
    fingerprint must be Stage 1's, else the case stops the run); the tilt, the length and the balance point (env.truth);
    commit_std's base (base_gate against Stage 1's row, else a stop); the ten pose rows; one simulation per feasible row,
    each after its own reset; both ceilings (an undetermined one drops the case); with ``replay`` the 2-seed gate at
    evpi_sim.seeds_of's second seed (replay_rows: a row that differs stops the case)."""
    from robots.mujoco_lab import calibrate as cal
    from robots.mujoco_lab import com_gap, evpi_sim
    from robots.mujoco_lab import adapter
    from robots.mujoco_lab.curve import stage2_skip

    o, idx = rec1["object"], int(rec1["idx"])
    rec: dict[str, Any] = dict({k: rec1.get(k) for k in KEEP}, kind=KIND_OF.get(o, "other"), hammer="hammer" in o, rows=[],
                               base=None, ceiling10=None, ceiling3=None, skipped=stage2_skip(rec1), dropped=False, stop=None,
                               error=None, replay=None)
    if rec["skipped"]:
        return dict(rec, dropped=True)
    env = prims.env
    theta, com_end, destination, hidden = cal.case_draw(seed, o, idx, adapter.object_props(o)["mass"], False, fragility=world)
    if destination != rec1["destination"]:
        return dict(rec, stop=f"the case's draw gives the destination {destination}, Stage 1's is {rec1['destination']}")
    if world is not None:
        env.slosh_zeta, env.slosh_scale = world[1].slosh_zeta, world[1].slosh_scale
    env.classes = None if world is None else world[0]
    seeds = evpi_sim.seeds_of(seed, o, idx)
    fixture = destination if destination in cal.MESH_IDS else None

    def reset_at(k: int) -> Any:
        def reset() -> bool:
            env.reset(o, theta, seed=int(seeds[k]), com_end=com_end, fixture=fixture, fill=hidden["fill"], frag=hidden["frag"])
            return bool(dict(getattr(env, "reset_info", None) or {}).get("at_rest", True))
        return reset

    want = rec1["fingerprint"]
    try:
        reset_at(0)()
        if com_gap.fingerprint(env) != want:
            return dict(rec, stop="the reset's fingerprint differs from Stage 1's")
        t = env.truth()
        tilt = tilt_deg(env.sim.obj_mat(), env.props["axis"])
        rec.update(tilt_deg=tilt, upright=tilt > UPRIGHT_DEG, length=float(env.props["length"]),
                   com_f=balance_fraction(env.props, t["com_obj"]), sim_truth={k: t.get(k) for k in ("mass", "com_s", "com_obj", "fill")})
        base = commit_std_base(env, o, destination)
        why = base_gate(rec1, base)
        if why:
            return dict(rec, stop=why)
        rows = pose_rows(env, destination)
        rec.update(base=base, rows=rows)
        stop = simulate_rows(env, reset_at(0), rows, base, com_gap.fingerprint, want)
        if stop:
            return dict(rec, stop=stop)
        if replay:
            t1 = time.time()
            rec["replay"] = replay_rows(env, reset_at(1), rows, base, com_gap.fingerprint, want)
            rec["replay_seconds"] = round(time.time() - t1, 2)
            if rec["replay"]["stop"]:
                return dict(rec, stop=f"replay: {rec['replay']['stop']}")
            if rec["replay"]["mismatches"]:
                return dict(rec, stop=f"replay: {len(rec['replay']['mismatches'])} rows differ at the second seed (the 2-seed gate)")
    except Exception as exc:  # noqa: BLE001  a case that raises outside a row's simulation is dropped
        return dict(rec, error=_err(exc), dropped=True)
    c10, c3 = ceiling(rows), ceiling(rows, current_only=True)
    return dict(rec, ceiling10=c10, ceiling3=c3, dropped=not (c10["determined"] and c3["determined"]))


def poseset_object(job: tuple[str, list[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    """One object's cases on its fold's bundle (curve_oracle.oracle_object's sensor and world, no probe models). The log
    line per case holds counts and errors, no label (no early look). A gate that fails ends the object with ``stop``."""
    from robots.mujoco_lab import calibrate as cal
    from robots.mujoco_lab.prior import fragility_world

    o, recs1, s = job
    prims = cal._primitives(s["bundle"], None, None)
    if getattr(prims, "reset_s", None) is not None:
        raise RuntimeError("the pose set scores the absorbing attempt only (this bundle restores the object after damage)")
    probes, prims.probe_models = getattr(prims, "probe_models", None), {}
    prims.force_noise = cal.job_force_noise(None, prims.manifest)
    prims.env.force_noise = prims.force_noise
    world = fragility_world(prims.manifest)
    out: dict[str, Any] = {"records": [], "stop": None}
    try:
        for rec1 in recs1:
            t0 = time.time()
            rec = poseset_case(prims, rec1, int(s["seed"]), world, replay=bool(s.get("replay")))
            rec["seconds"] = round(time.time() - t0, 2)
            out["records"].append(rec)
            n = sum("seconds" in r for r in rec["rows"])
            bad = sum(bool(r.get("error")) for r in rec["rows"])
            print(f"{o} {rec1['idx']}: {n} rows simulated, {rec['seconds']:.1f} s" + (f", {bad} row errors" if bad else "")
                  + (f", skipped ({rec['skipped']})" if rec["skipped"] else "") + (f", error {rec['error']}" if rec["error"] else "")
                  + (f", STOP {rec['stop']}" if rec["stop"] else ""), file=sys.stderr, flush=True)
            if rec["stop"]:
                out["stop"] = f"{o} {rec1['idx']}: {rec['stop']}"
                break
    finally:
        prims.probe_models = probes
    return out


def _metrics(done: list[dict[str, Any]]) -> dict[str, Any]:
    """A run's counts and times, no label: cases, skips, drops, errors, rows simulated, row errors, the replay gate's
    rows and mismatches."""
    rows = [r for rec in done for r in rec["rows"] if "seconds" in r]
    rep = [rec["replay"] for rec in done if rec.get("replay")]
    return {"cases": len(done), "skipped": sum(bool(r["skipped"]) for r in done), "dropped": sum(bool(r["dropped"]) for r in done),
            "case_errors": sum(bool(r["error"]) for r in done), "rows_simulated": len(rows),
            "row_errors": sum(bool(r.get("error")) for r in rows), "row_seconds": round(sum(float(r["seconds"]) for r in rows), 1),
            "replay_rows": sum(x["rows"] for x in rep), "replay_mismatches": sum(len(x["mismatches"]) for x in rep),
            "case_seconds": round(sum(float(r["seconds"]) for r in done), 1),
            "case_seconds_max": max((float(r["seconds"]) for r in done), default=0.0)}


def run(a: argparse.Namespace) -> str:
    """The pose-set run: records_validation.jsonl and records_test.jsonl (one poseset_case record per case, by object as
    each ends), run_config.json and run_metrics.json (times, counts and errors only; wandb when on). A gate that fails
    stops the run (exit 3). Returns --out."""
    import multiprocessing as mp

    from robots.mujoco_lab.curve import SIDES, _peak_rss_mb, _sha, _stage2_cases, load_stage1, records_path
    from robots.mujoco_lab.estimation_check import _revision
    from robots.mujoco_lab.identify_fit import require_bundle_generator
    from robots.mujoco_lab.tracking import log_run

    os.makedirs(a.out, exist_ok=True)
    if any(os.path.exists(records_path(a.out, side)) for side in SIDES):
        raise SystemExit(f"{a.out}: a records file exists: pick a new --out")
    dirs = a.stage1.split(",")
    recs1, cfg1 = load_stage1(dirs)
    generator = require_bundle_generator(cfg1["bundle"])
    recs = _stage2_cases(a, recs1)
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        by[r["object"]].append(r)
    config = {**vars(a), **_revision(), "generator": generator, "bundle": cfg1["bundle"], "seed": cfg1["seed"],
              "stage1_code_revision": cfg1["code_revision"],
              "stage1_records_sha256": {d: _sha(os.path.join(d, "records.jsonl")) for d in dirs},
              "split_sha256": cfg1["split_sha256"], "cases": [(r["object"], int(r["idx"])) for r in recs], "fracs": FRACS,
              "current": CURRENT, "turn": TURN, "row_order": ROW_ORDER, "upright_deg": UPRIGHT_DEG}
    with open(os.path.join(a.out, "run_config.json"), "w") as f:
        json.dump(config, f, indent=1)
    jobs = [(o, items, {"bundle": os.path.join(cfg1["bundle"], "folds", f"fold_{items[0]['fold']}"), "seed": cfg1["seed"],
                        "replay": bool(a.replay)}) for o, items in by.items()]
    t0, stop = time.time(), None
    outs = {side: open(records_path(a.out, side), "w") for side in SIDES}
    try:
        pool = None if a.workers == 1 else mp.get_context("spawn").Pool(a.workers)
        results = map(poseset_object, jobs) if pool is None else pool.imap_unordered(poseset_object, jobs)
        try:
            for res in results:
                for r in res["records"]:
                    outs[r["side"]].write(json.dumps(r) + "\n")
                for f in outs.values():
                    f.flush()
                if res["stop"]:
                    stop = res["stop"]
                    break
        finally:
            if pool is not None:
                (pool.terminate if stop else pool.close)()
                pool.join()
    finally:
        for f in outs.values():
            f.close()
    done = [json.loads(line) for side in SIDES for line in open(records_path(a.out, side))]
    metrics = dict(_metrics(done), of_cases=len(recs), stop=stop, wall_seconds=round(time.time() - t0, 1), peak_rss_mb=_peak_rss_mb())
    metrics["wandb"] = log_run("curve-poseset", config, metrics)
    with open(os.path.join(a.out, "run_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print(f"done: {json.dumps(metrics)}", flush=True)
    if stop:
        raise SystemExit(3)
    return a.out


# ---------------------------------------------------------------- the smoke

def _key(r: dict[str, Any]) -> tuple[str, int]:
    return r["object"], int(r["idx"])


def smoke_report(recs: list[dict[str, Any]], s2: list[dict[str, Any]], recs1: list[dict[str, Any]]) -> dict[str, Any]:
    """The smoke report on validation records: _metrics's counts, the stops, a table per case (the feasible rows and
    their labels, both ceilings), the 2-seed gate (replay_rows) and the Stage 2 gate (stage2_gate on the cases Stage 2
    holds), and the full run's projection: the mean first-pass seconds per case (the replay's seconds left out) times
    the Stage 1 test cases that are not skipped."""
    from robots.mujoco_lab.curve import stage2_skip

    if any(r["side"] != "validation" for r in recs):
        raise SystemExit("the smoke reads validation records only")
    by1, by2 = {_key(r): r for r in recs1}, {_key(r): r for r in s2}
    joined = [{"pose": r, "stage1": by1[_key(r)], "stage2": by2[_key(r)]} for r in recs if not r["skipped"] and _key(r) in by2]
    table = [{"object": r["object"], "idx": int(r["idx"]), "upright": r.get("upright"), "skipped": r["skipped"],
              "dropped": r["dropped"], "stop": r["stop"], "feasible": [x["id"] for x in r["rows"] if x["feasible"]],
              "labels": {x["id"]: x.get("label") for x in r["rows"] if "seconds" in x},
              "artifacts": [x["id"] for x in r["rows"] if x.get("artifact")], "errors": [x["id"] for x in r["rows"] if x.get("error")],
              "ceiling10": r["ceiling10"], "ceiling3": r["ceiling3"], "seconds": r["seconds"],
              "replay_seconds": r.get("replay_seconds")} for r in recs]
    rep = [r["replay"] for r in recs if r.get("replay")]
    secs = [float(r["seconds"]) - float(r.get("replay_seconds") or 0.0) for r in recs if not r["skipped"]]
    run_cases = sum(r["side"] == "test" and stage2_skip(r) is None for r in recs1)
    sm = dict(_metrics(recs), stops=[r["stop"] for r in recs if r["stop"]], table=table,
              replay={"cases": len(rep), "rows": sum(x["rows"] for x in rep), "mismatches": sum(len(x["mismatches"]) for x in rep)},
              replay_mismatches_by_case=[{"object": r["object"], "idx": int(r["idx"]), "mismatches": r["replay"]["mismatches"]}
                                         for r in recs if r.get("replay") and r["replay"]["mismatches"]],
              stage2_gate=stage2_gate(joined), run_cases=run_cases,
              seconds_per_case=float(np.mean(secs)) if secs else None, seconds_per_case_max=max(secs) if secs else None)
    sm["projection_cpu_h"] = sm["seconds_per_case"] * run_cases / 3600.0 if secs else None
    sm["projection_cpu_h_at_max"] = sm["seconds_per_case_max"] * run_cases / 3600.0 if secs else None
    return sm


# ---------------------------------------------------------------- the readings (design 3 and 3.1)

POSE_LIMIT, POSE_NOT = "the pose set is a limit", "the pose set is not a limit"
EXPECTED_COVERAGE = {"across_0.1": 297, "across_0.3": 384, "across_0.5": 398, "across_0.7": 384, "across_0.9": 364, "along_0.1": 67,
                     "along_0.3": 103, "along_0.5": 150, "along_0.7": 99, "along_0.9": 98}   # design 3.1, the probe's 500 cases
ARMS = ("c10", "c3", "oracle", "r4")


def ceiling_reading(b: dict[str, Any], eps: float | None = None) -> str:
    """Design reading 1's rule, as written, on a paired success difference g with its interval [lo, hi]: g > eps and
    lo > 0, the pose set is a limit; hi < eps, it is not; else undecided at the view's case count. An interval below 0 is
    _block's flag beside the rule."""
    from robots.mujoco_lab.curve import EPS

    eps = EPS if eps is None else eps
    x = b["success"]
    if x["diff"] > eps and x["lo"] > 0.0:
        return POSE_LIMIT
    if x["hi"] < eps:
        return POSE_NOT
    return f"undecided at n = {b['cases']}"


def paired_set3(joined: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """The joined cases that no arm drops (Stage 2, the oracle arm, the pose set), and the numbers dropped (by each arm,
    and by the pose set alone)."""
    d = [(bool(j["stage2"]["dropped"]), bool(j["oracle"]["dropped"]), bool(j["pose"]["dropped"])) for j in joined]
    return ([j for j, x in zip(joined, d) if not any(x)],
            {"dropped": sum(any(x) for x in d), "stage2": sum(x[0] for x in d), "oracle": sum(x[1] for x in d),
             "pose": sum(x[2] for x in d), "pose_only": sum(x[2] and not x[0] and not x[1] for x in d)})


def views(joined: list[dict[str, Any]], k0: str, k4: str) -> list[dict[str, Any]]:
    """Per joined case: its object, strata, rows, Stage 2 group (curve_oracle.menu_group), the episodes (c10 and c3: the
    ceilings; oracle; R0 and R4 from Stage 2) and the oracle's first pinch (None: it gave up)."""
    from robots.mujoco_lab.curve import GIVE_UP_ROW, episode
    from robots.mujoco_lab.curve_oracle import _action, _akey, menu_group

    out = []
    for j in joined:
        p, s2, o = j["pose"], j["stage2"], j["oracle"]
        fo = o["decisions"][k0]
        out.append({"object": p["object"], "idx": int(p["idx"]), "thin": bool(p["thin"]), "kind": p["kind"], "hammer": bool(p["hammer"]),
                    "upright": bool(p["upright"]), "com_f": p["com_f"], "rows": p["rows"], "group": menu_group(s2["decisions"], k0, k4),
                    "c10": p["ceiling10"], "c3": p["ceiling3"], "oracle": episode(o["decisions"], o["sims"], k0),
                    "r0": episode(s2["decisions"], s2["sims"], k0), "r4": episode(s2["decisions"], s2["sims"], k4),
                    "oracle_s": None if fo == GIVE_UP_ROW else _action(_akey(o["rows"], fo)).get("s")})
    return out


def label_mix(eps: list[dict[str, Any]]) -> dict[str, int]:
    """Design reading 3: the label that ends each episode (its last attempt's), or "give-up" when it made no attempt."""
    return dict(Counter(e["labels"][-1] if e["labels"] else "give-up" for e in eps))


def _rates(eps: list[dict[str, Any]]) -> dict[str, Any]:
    from robots.mujoco_lab.curve_oracle import _rates as rates

    return rates(eps)


def kind_table(vs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Design reading 4: per kind (KINDS), the cases, the objects and each arm's rates (the two ceilings, the oracle, R4)."""
    out = {}
    for k in KINDS:
        x = [v for v in vs if v["kind"] == k]
        out[k] = dict({"n": len(x), "objects": len({v["object"] for v in x})}, **{a: _rates([v[a] for v in x]) for a in ARMS})
    return out


def hammer_line(vs: list[dict[str, Any]]) -> dict[str, Any]:
    """Design reading 4's hammer line: the hammer cases by hammer_class, the distances from the balance point (pinch
    fractions) of the clean rows that placed the hammer and of those that did not, and how often the clean row nearest the
    balance point placed it; per case its class and the rows that placed or lifted it."""
    hv = [v for v in vs if v["hammer"]]
    classes = {c: 0 for c in HAMMER_CLASSES}
    succ, fail, near, per = [], [], {"cases": 0, "succeeds": 0}, []
    for v in hv:
        c = hammer_class(v["rows"])
        classes[c] += 1
        clean = [r for r in v["rows"] if _clean(r)]
        for r in clean:
            (succ if r["label"] == "placed_stable" else fail).append(abs(r["f"] - v["com_f"]))
        nr = nearest_row(v["rows"], v["com_f"])
        if nr is not None:
            near["cases"] += 1
            near["succeeds"] += nr["label"] == "placed_stable"
        per.append({"object": v["object"], "idx": v["idx"], "com_f": v["com_f"], "class": c,
                    "placed": [r["id"] for r in clean if r["label"] == "placed_stable"],
                    "lifted": [r["id"] for r in clean if r["label"] not in LIFT_FAILS]})
    return {"cases": len(hv), "objects": sorted({v["object"] for v in hv}), "classes": classes, "success_distance": succ,
            "failure_distance": fail, "nearest_succeeds": near, "per_case": per}


def no_row_line(vs: list[dict[str, Any]]) -> dict[str, Any]:
    """Design reading 5: the group (a) cases (R4 gives up: no menu row), how many gain a feasible pose row, by object, and
    the 10-row ceiling over all of them and over those that gain a row."""
    a = [v for v in vs if v["group"] == "a"]
    gained = [v for v in a if v["c10"]["n_rows"] > 0]
    by: dict[str, dict[str, int]] = {}
    for v in a:
        x = by.setdefault(v["object"], {"cases": 0, "gain_a_row": 0})
        x["cases"] += 1
        x["gain_a_row"] += v["c10"]["n_rows"] > 0
    return {"cases": len(a), "gain_a_row": len(gained), "by_object": by, "c10": _rates([v["c10"] for v in a]),
            "c10_gained": _rates([v["c10"] for v in gained])}


def pose_change_table(vs: list[dict[str, Any]]) -> dict[str, Any]:
    """Design reading 6: pose_change over every case and over the gain cases (the 10-row ceiling places the object, the
    oracle arm does not), with each class's 10-row ceiling and oracle rates."""
    cls = [pose_change(v["rows"], v["oracle_s"]) for v in vs]
    return {"all": dict(Counter(cls)),
            "gain": dict(Counter(c for c, v in zip(cls, vs) if v["c10"]["success"] and not v["oracle"]["success"])),
            "outcomes": {c: {"c10": _rates([v["c10"] for x, v in zip(cls, vs) if x == c]),
                             "oracle": _rates([v["oracle"] for x, v in zip(cls, vs) if x == c])} for c in sorted(set(cls))}}


def balance_line(vs: list[dict[str, Any]]) -> dict[str, Any]:
    """Design reading 6's balance point: over the cases whose clean rows both place and fail, how often the row nearest
    the balance point places the object, against chance (the mean share of placing rows in those cases)."""
    mixed, near, chance = 0, 0, []
    for v in vs:
        clean = [r for r in v["rows"] if _clean(r)]
        ns = sum(r["label"] == "placed_stable" for r in clean)
        if not clean or ns in (0, len(clean)):
            continue
        mixed += 1
        near += nearest_row(v["rows"], v["com_f"])["label"] == "placed_stable"
        chance.append(ns / len(clean))
    return {"mixed": mixed, "nearest_succeeds": near, "chance": float(np.mean(chance)) if chance else None}


def coverage(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """The feasible rows of the cases not skipped, by pose, against the geometry probe's (design 3.1, 500 cases)."""
    kept = [r for r in recs if not r["skipped"]]
    by = {row_id(d, f): sum(any(x["id"] == row_id(d, f) and x["feasible"] for x in r["rows"]) for r in kept) for d, f in ROW_ORDER}
    return {"cases": len(kept), "by_row": by, "rows": sum(bool(x["feasible"]) for r in kept for x in r["rows"]),
            "new_rows": sum(bool(x["feasible"]) and not x["current"] for r in kept for x in r["rows"]),
            "any_row": sum(any(x["feasible"] for x in r["rows"]) for r in kept), "expected": EXPECTED_COVERAGE,
            "expected_rows": 2344, "expected_new_rows": 1178}


def _ceil_cell(c: dict[str, Any] | None) -> str:
    if c is None:
        return "–"
    if c["give_up"]:
        return "no row"
    out = ("placed" if c["success"] else "irreversible" if c["irr"] else "not placed") + f" ({', '.join(c['rows']) or '–'})"
    return out + ("" if c["determined"] else ", undetermined")


def smoke_markdown(sm: dict[str, Any]) -> str:
    g, r = sm["stage2_gate"], sm["replay"]
    L = ["# Pose set: the smoke test (validation cases only)", "",
         f"- Cases {sm['cases']}: skipped {sm['skipped']}, dropped {sm['dropped']}, case errors {sm['case_errors']}, rows simulated "
         f"{sm['rows_simulated']}, row errors {sm['row_errors']}, stops {len(sm['stops'])}.",
         f"- 2-seed gate: {r['cases']} cases, {r['rows']} rows replayed, {r['mismatches']} mismatches.",
         f"- Stage 2 gate: {g['compared']} rows compared, {len(g['mismatches'])} mismatches, {g['missing']} matching Stage 2 rows "
         "without a pose-set simulation.",
         f"- First-pass seconds per case: mean {sm['seconds_per_case']:.1f}, max {sm['seconds_per_case_max']:.1f}." if sm["seconds_per_case"]
         is not None else "- First-pass seconds per case: –.",
         f"- Projection for {sm['run_cases']} test cases: {sm['projection_cpu_h']:.2f} CPU-h (at the slowest smoke case: "
         f"{sm['projection_cpu_h_at_max']:.2f} CPU-h)." if sm["projection_cpu_h"] is not None else "- Projection: –.", "",
         "| case | pose | feasible rows | labels | 10-row ceiling | 3-row ceiling | seconds (replay) |", "|---|---|---|---|---|---|---|"]
    for t in sm["table"]:
        pose = "–" if t["upright"] is None else "upright" if t["upright"] else "lying"
        labels = ", ".join(f"{k} {v}" for k, v in t["labels"].items()) or "–"
        if t["artifacts"] or t["errors"]:
            labels += f" (artifacts: {', '.join(t['artifacts']) or '–'}; errors: {', '.join(t['errors']) or '–'})"
        L.append(f"| {t['object']} {t['idx']} | {pose} | {len(t['feasible'])} | {labels} | {_ceil_cell(t['ceiling10'])} | "
                 f"{_ceil_cell(t['ceiling3'])} | {t['seconds']} ({t['replay_seconds']}) |" + (f" STOP {t['stop']}" if t["stop"] else "")
                 + (f" skipped ({t['skipped']})" if t["skipped"] else ""))
    return "\n".join(L) + "\n"


def smoke(a: argparse.Namespace) -> dict[str, Any]:
    """The smoke report (smoke_report) on validation runs only (refuses a run that holds test records), with Stage 2's
    validation records for the Stage 2 gate; smoke_summary.{json,md} in --out (refuses an existing one), wandb."""
    from robots.mujoco_lab.curve import load_stage1, load_stage2, records_path
    from robots.mujoco_lab.tracking import log_run

    os.makedirs(a.out, exist_ok=True)
    jpath = os.path.join(a.out, "smoke_summary.json")
    if os.path.exists(jpath):
        raise SystemExit(f"{jpath} exists: pick a new --out")
    dirs = a.runs.split(",")
    for d in dirs:
        t = records_path(d, "test")
        if os.path.exists(t) and os.path.getsize(t) > 0:
            raise SystemExit(f"{d} holds test records: the smoke reads validation cases only")
    recs = [json.loads(line) for d in dirs for line in open(records_path(d, "validation"))]
    cfg = json.load(open(os.path.join(dirs[0], "run_config.json")))
    s2, _ = load_stage2(a.stage2.split(","), ("validation",))
    recs1, _ = load_stage1(list(cfg["stage1_records_sha256"]))
    sm = smoke_report(recs, s2, recs1)
    sm["wandb"] = log_run("curve-poseset-smoke", {"runs": a.runs, "stage2": a.stage2},
                          {k: v for k, v in sm.items() if k not in ("table", "replay_mismatches_by_case")})
    with open(jpath, "w") as f:
        json.dump(sm, f, indent=1)
    with open(os.path.join(a.out, "smoke_summary.md"), "w") as f:
        f.write(smoke_markdown(sm))
    print(json.dumps({k: v for k, v in sm.items() if k != "table"}, indent=1), flush=True)
    return sm


# ---------------------------------------------------------------- the summary

def _block(vs: list[dict[str, Any]]) -> dict[str, Any]:
    """Readings 1', 1, 2 and the selector term on one view, paired by case (curve_oracle.paired_outcomes: the object
    bootstrap, BOOT_DRAWS draws, BOOT_SEED), with design reading 1's rule on 1' (design 3.1) and 1 only, and beside each
    rule the flag that the success interval lies below 0 (the ceiling under the comparison arm)."""
    from robots.mujoco_lab.curve_oracle import paired_outcomes

    def b(x: str, y: str) -> dict[str, Any]:
        return paired_outcomes([(v["object"], v[x], v[y]) for v in vs])

    out = {"1p_pose_term": b("c10", "c3"), "1_vs_oracle": b("c10", "oracle"), "2_vs_r4": b("c10", "r4"), "2_vs_r0": b("c10", "r0"),
           "selector_term": b("c3", "oracle")}
    for k, name in (("1p", "1p_pose_term"), ("1", "1_vs_oracle")):
        out[f"{k}_reading"] = ceiling_reading(out[name]) if vs else None
        out[f"{k}_below"] = bool(vs) and out[name]["success"]["hi"] < 0.0
    return out


def _four_to_six(vs: list[dict[str, Any]]) -> dict[str, Any]:
    return {"4_kinds": kind_table(vs), "4_hammer": hammer_line(vs), "5_no_row": no_row_line(vs), "6_pose_change": pose_change_table(vs),
            "6_balance": balance_line(vs)}


def _damage(vs: list[dict[str, Any]]) -> dict[str, Any]:
    return {arm: {"rates": _rates([v[arm] for v in vs]), "labels": label_mix([v[arm] for v in vs])} for arm in ARMS + ("r0",)}


def _load_lines(paths: list[str]) -> list[dict[str, Any]]:
    return [json.loads(line) for path in paths for line in open(path) if line.strip()]


def _read_once(paths: list[str]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """The records in ``paths`` and each file's sha256, from one open per file."""
    import hashlib

    recs, shas = [], {}
    for path in paths:
        with open(path, "rb") as f:
            raw = f.read()
        shas[path] = hashlib.sha256(raw).hexdigest()
        recs += [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    return recs, shas


def summary(a: argparse.Namespace) -> dict[str, Any]:
    """The readings (design 3 and 3.1) from finished pose-set test runs, Stage 2's test records and the oracle arm's:
    poseset_summary.{json,md} and poseset_points.png in --out (refuses an existing summary), wandb. Refuses a run without
    EXIT 0, Stage 2 or oracle records other than the ones their summaries read, oracle runs on another bundle, seed, Stage 1
    or Stage 2, Stage 1 records that changed, and records that do not cover the same cases. The pose set's test records are opened once. A Stage 2 gate mismatch (design 3.1)
    writes poseset_summary_stop.json and exits 3 before any reading."""
    from robots.mujoco_lab.curve import BOOT_DRAWS, BOOT_SEED, EPS, _curves, _sha, load_stage1, load_stage2, records_path
    from robots.mujoco_lab.curve_oracle import _finished
    from robots.mujoco_lab.estimation_check import _revision
    from robots.mujoco_lab.tracking import log_run

    os.makedirs(a.out, exist_ok=True)
    jpath, spath = os.path.join(a.out, "poseset_summary.json"), os.path.join(a.out, "poseset_summary_stop.json")
    if os.path.exists(jpath) or os.path.exists(spath):
        raise SystemExit(f"{a.out}: a summary exists: pick a new --out")
    lk = json.load(open(a.locked))["lock"]
    k0, k4 = lk["R0"]["key"], lk["R4"]["key"]
    dirs = a.runs.split(",")
    cfgs = [_finished(d) for d in dirs]
    for k in ("bundle", "seed", "stage1_records_sha256", "code_revision", "fracs", "row_order", "replay"):
        if len({json.dumps(c.get(k), sort_keys=True) for c in cfgs}) > 1:
            raise SystemExit(f"the pose-set runs differ in {k}")
    cfg = cfgs[0]
    s2dirs, odirs = a.stage2.split(","), a.oracle.split(",")
    prov2 = json.load(open(a.stage2_summary))["provenance"]["records_sha256"]
    for d in s2dirs:
        if _sha(records_path(d, "test")) != prov2.get(f"{d} test"):
            raise SystemExit(f"{d}: the Stage 2 test records are not the ones the Stage 2 summary read")
    oprov = json.load(open(a.oracle_summary))["provenance"]
    if list(oprov.get("keys") or []) != [k0, k4]:
        raise SystemExit(f"the oracle summary's keys {oprov.get('keys')} are not the lock's {[k0, k4]}")
    for d in odirs:
        if _sha(records_path(d, "test")) != oprov["oracle_records_sha256"].get(d):
            raise SystemExit(f"{d}: the oracle test records are not the ones the oracle summary read")
        ocfg = _finished(d)
        for k in ("bundle", "seed", "stage1_records_sha256"):
            if json.dumps(ocfg.get(k), sort_keys=True) != json.dumps(cfg.get(k), sort_keys=True):
                raise SystemExit(f"{d}: the oracle run's {k} is not the pose-set runs'")
    if set(str(oprov.get("stage2") or "").split(",")) != set(s2dirs) or sorted(oprov.get("stage1") or []) != sorted(cfg["stage1_records_sha256"]):
        raise SystemExit("the oracle summary read another stage2 or Stage 1 than the pose-set summary")
    s2, s2cfg = load_stage2(s2dirs, ("test",))
    if s2cfg["stage1_records_sha256"] != cfg["stage1_records_sha256"]:
        raise SystemExit("the pose-set runs and Stage 2 read different Stage 1 records")
    for d, h in cfg["stage1_records_sha256"].items():
        if _sha(os.path.join(d, "records.jsonl")) != h:
            raise SystemExit(f"{d}: the Stage 1 records changed since the pose-set runs")
    recs1, _ = load_stage1(list(cfg["stage1_records_sha256"]))
    orecs = _load_lines([records_path(d, "test") for d in odirs])
    precs, psha = _read_once([records_path(d, "test") for d in dirs])            # the pose set's test records, once
    if any(r["side"] != "test" or r.get("stop") for r in precs):
        raise SystemExit("a pose-set test record holds a stop or another side")
    byp, byo, by1 = {_key(r): r for r in precs}, {_key(r): r for r in orecs}, {_key(r): r for r in recs1}
    if len(byp) != len(precs) or len(byo) != len(orecs):
        raise SystemExit("a case appears twice in the pose-set or oracle records")
    have2 = {_key(r) for r in s2}
    if set(byp) != have2 or set(byo) != have2:
        raise SystemExit(f"the cases differ: Stage 2 {len(have2)}, the pose set {len(byp)}, the oracle {len(byo)}")
    joined = [{"pose": byp[_key(r)], "stage2": r, "stage1": by1[_key(r)], "oracle": byo[_key(r)]} for r in s2]
    gate = stage2_gate([j for j in joined if not j["pose"]["skipped"]])
    me = _revision()
    prov = {"runs": a.runs, "records_sha256": {d: psha[records_path(d, "test")] for d in dirs}, "code_revision": cfg["code_revision"],
            "summary_code_revision": me["code_revision"], "summary_dirty": me["dirty"], "stage2": a.stage2,
            "stage2_summary": f"{a.stage2_summary} (sha256 {_sha(a.stage2_summary)})", "oracle": a.oracle,
            "oracle_summary": f"{a.oracle_summary} (sha256 {_sha(a.oracle_summary)})", "locked": f"{a.locked} (sha256 {_sha(a.locked)})",
            "stage1": sorted(cfg["stage1_records_sha256"]), "bootstrap": f"{BOOT_DRAWS} object draws, seed {BOOT_SEED}, 95 % percentile interval",
            "eps": EPS, "keys": [k0, k4]}
    if gate["mismatches"]:
        with open(spath, "w") as f:
            json.dump({"stage2_gate": gate, "provenance": prov}, f, indent=1)
        print(f"STOP: {len(gate['mismatches'])} Stage 2 gate mismatches; wrote {spath}", flush=True)
        raise SystemExit(3)
    paired, n_dropped = paired_set3(joined)
    vs = views(paired, k0, k4)
    nt = [v for v in vs if not v["thin"]]
    kept = [r for r in precs if not r["skipped"]]
    sim_rows = [x for r in kept for x in r["rows"] if "seconds" in x]
    counts = {"cases": len(precs), "skipped": len(precs) - len(kept), "pose_dropped": sum(bool(r["dropped"]) for r in precs),
              "pose_undetermined": sum(bool(r["dropped"]) and not r["skipped"] and not r["error"] for r in precs),
              "case_errors": sum(bool(r["error"]) for r in precs), "rows_simulated": len(sim_rows),
              "row_errors": sum(bool(x.get("error")) for x in sim_rows), "row_artifacts": sum(bool(x.get("artifact")) for x in sim_rows),
              "paired": len(vs), "paired_dropped": n_dropped, "objects": len({v["object"] for v in vs}), "thin_cases": len(vs) - len(nt),
              "upright_cases": sum(v["upright"] for v in vs), "groups": dict(Counter(v["group"] for v in vs))}
    readings = {"all": _block(vs), "no_thin": _block(nt), "upright": _block([v for v in vs if v["upright"]]),
                "lying": _block([v for v in vs if not v["upright"]]),
                "3_damage": {"all": _damage(vs), "no_thin": _damage(nt)}, **_four_to_six(vs), "4_to_6_no_thin": _four_to_six(nt)}
    disclosures = {"stage2_gate": gate, "off_centre_current": off_centre_current(kept), "coverage": coverage(precs)}
    rules = tuple(s2cfg["rules"])
    figure = {"curves": _curves([j["stage2"] for j in paired], rules), "locked": {r: lk[r]["knob"] for r in rules},
              "oracle": _rates([v["oracle"] for v in vs]), "c10": _rates([v["c10"] for v in vs]), "c3": _rates([v["c3"] for v in vs])}
    rep = {"counts": counts, "readings": readings, "disclosures": disclosures, "figure": figure, "provenance": prov}
    r1 = readings["all"]["1p_pose_term"]
    rep["wandb"] = log_run("curve-poseset-summary", {"runs": a.runs, "stage2": a.stage2, "oracle": a.oracle},
                           {"counts": counts, "reading": readings["all"]["1p_reading"], "reading_no_thin": readings["no_thin"]["1p_reading"],
                            "pose_term": {k: r1[k] for k in ("success", "irr", "give_up")}, "stage2_gate": gate})
    with open(jpath, "w") as f:
        json.dump(rep, f, indent=1)
    with open(os.path.join(a.out, "poseset_summary.md"), "w") as f:
        f.write(summary_markdown(rep))
    poseset_figure(rep, os.path.join(a.out, "poseset_points.png"))
    print(f"wrote {a.out}", flush=True)
    return rep


def _ci(b: dict[str, Any], name: str) -> str:
    x = b.get(name)
    return "–" if x is None else f"{100 * x['diff']:+.2f} [{100 * x['lo']:+.2f}, {100 * x['hi']:+.2f}]"


def _pc(x: float | None, nd: int = 1) -> str:
    return "–" if x is None else f"{100 * x:.{nd}f}"


def _cells(r: dict[str, Any]) -> str:
    return f"{_pc(r['success'])} | {_pc(r['irr'], 2)} | {_pc(r['give_up'])}"


ARM_NAMES = {"c10": "10-row ceiling", "c3": "3-row ceiling", "oracle": "oracle arm", "r0": "R0 at r 2", "r4": "R4"}
COMPARISONS = (("1p_pose_term", "1′ (primary): 10-row − 3-row ceiling (the pose term)"),
               ("1_vs_oracle", "1: 10-row ceiling − oracle arm"), ("2_vs_r4", "2: 10-row ceiling − R4"),
               ("2_vs_r0", "2: 10-row ceiling − R0 at r 2"), ("selector_term", "selector term: 3-row ceiling − oracle arm"))


def _rule(x: dict[str, Any], k: str) -> str:
    """A reading rule's label, with the flag that its success interval lies below 0."""
    out = x[k] or "–"
    return out + (" (interval below 0: the ceiling is under the comparison arm)" if x.get(k.replace("reading", "below")) else "")


def _md4(X: dict[str, Any]) -> list[str]:
    L = ["| kind | cases (objects) | 10-row ceiling | 3-row ceiling | oracle arm | R4 |", "|---|---|---|---|---|---|"]
    for k, x in X["4_kinds"].items():
        L.append(f"| {k} | {x['n']} ({x['objects']}) | " + " | ".join(_cells(x[a]).replace(" | ", " / ") for a in ARMS) + " |")
    h = X["4_hammer"]
    med = lambda v: "–" if not v else f"{float(np.median(v)):.2f}"  # noqa: E731
    return L + ["", f"Hammers ({h['cases']} cases, {', '.join(h['objects']) or '–'}): {h['classes']}. Distance from the balance point "
                f"(pinch fractions), median: {med(h['success_distance'])} for the clean rows that placed a hammer "
                f"({len(h['success_distance'])} rows), {med(h['failure_distance'])} for the others ({len(h['failure_distance'])} rows). "
                f"The clean row nearest the balance point placed the hammer in {h['nearest_succeeds']['succeeds']} of "
                f"{h['nearest_succeeds']['cases']} cases.", ""]


def _md5(X: dict[str, Any]) -> list[str]:
    n5 = X["5_no_row"]
    return [f"{n5['gain_a_row']} of {n5['cases']} cases gain a feasible pose row. By object: {n5['by_object']}. The 10-row ceiling "
            f"over all {n5['cases']}: {_cells(n5['c10']).replace(' | ', ' / ')}; over the {n5['gain_a_row']} with a row: "
            f"{_cells(n5['c10_gained']).replace(' | ', ' / ')} (success % / irreversible % / give-up %).", ""]


def _md6(X: dict[str, Any]) -> list[str]:
    t, bl = X["6_pose_change"], X["6_balance"]
    L = ["| class | cases | of them gain cases | 10-row ceiling success % | oracle arm success % |", "|---|---|---|---|---|"]
    for cl in POSE_CHANGES:
        if cl in t["all"]:
            L.append(f"| {cl} | {t['all'][cl]} | {t['gain'].get(cl, 0)} | {_pc(t['outcomes'][cl]['c10']['success'])} | "
                     f"{_pc(t['outcomes'][cl]['oracle']['success'])} |")
    return L + ["", f"Balance point: in {bl['mixed']} cases some clean rows place the object and some do not; the row nearest the "
                f"balance point places it in {bl['nearest_succeeds']} ({_pc(bl['nearest_succeeds'] / bl['mixed']) if bl['mixed'] else '–'} %), "
                f"chance {_pc(bl['chance'])} %.", ""]


def summary_markdown(rep: dict[str, Any]) -> str:
    c, R, D = rep["counts"], rep["readings"], rep["disclosures"]
    nd, g, cov, oc = c["paired_dropped"], D["stage2_gate"], D["coverage"], D["off_centre_current"]
    L = ["# Pose set: the readings (design 2026-11-08, amended in its section 3.1)", ""] + [f"- {k}: {v}" for k, v in rep["provenance"].items()]
    L += ["", "## Counts and checks", "",
          f"- Pose-set test cases {c['cases']}: skipped {c['skipped']} (Stage 2's skip rules), dropped {c['pose_dropped']} "
          f"({c['pose_undetermined']} with an undetermined ceiling, {c['case_errors']} case errors); rows simulated {c['rows_simulated']}, "
          f"row errors {c['row_errors']}, row artifacts {c['row_artifacts']}.",
          f"- Paired cases {c['paired']} on {c['objects']} objects ({c['thin_cases']} on thin plastic, {c['upright_cases']} upright); "
          f"Stage 2 groups {c['groups']}.",
          f"- Dropped from the paired set (a case any arm drops): {nd['dropped']} (Stage 2 {nd['stage2']}, the oracle arm {nd['oracle']}, "
          f"the pose set {nd['pose']}, the pose set alone {nd['pose_only']}).",
          f"- Stage 2 gate on the test cases: {g['compared']} rows compared, 0 mismatches, {g['missing']} matching Stage 2 rows without a "
          "pose-set simulation.",
          f"- Disclosure (design 3.1): the centring rule would refuse {oc['rows']} feasible current rows on {len(oc['objects'])} objects "
          f"{oc['objects']}; in {oc['cases_all_flagged']} of {oc['cases']} cases it would refuse every feasible current row. Nothing in "
          "Stage 2 or the oracle arm is changed.", "",
          "| pose row | " + " | ".join(cov["by_row"]) + " |", "|---|" + "---|" * len(cov["by_row"]),
          "| feasible (cases) | " + " | ".join(str(v) for v in cov["by_row"].values()) + " |",
          "| geometry probe (500 cases) | " + " | ".join(str(cov["expected"][k]) for k in cov["by_row"]) + " |", "",
          f"Feasible rows {cov['rows']} ({cov['new_rows']} new) in {cov['cases']} cases, {cov['any_row']} cases with a row; the probe: "
          f"{cov['expected_rows']} ({cov['expected_new_rows']} new).", "",
          "## Readings 1′, 1 and 2, and the selector term (points, paired by case, 95 % object bootstrap)", "",
          "| comparison | view | cases (objects) | success | irreversible | give-up | rule |", "|---|---|---|---|---|---|---|"]
    rule_of = {"1p_pose_term": "1p_reading", "1_vs_oracle": "1_reading"}
    for k, name in COMPARISONS:
        for view, label in (("all", "all"), ("no_thin", "without thin plastic")):
            b = R[view][k]
            L.append(f"| {name} | {label} | {b['cases']} ({b['objects']}) | {_ci(b, 'success')} | {_ci(b, 'irr')} | {_ci(b, 'give_up')} | "
                     f"{_rule(R[view], rule_of[k]) if k in rule_of else '–'} |")
    L += ["", "The selector term also carries the oracle arm's force and speed rows and its second attempt: it is not pure.", "",
          "| arm (paired cases) | success % | irreversible % | give-up % |", "|---|---|---|---|"]
    b = R["all"]
    for arm, src in (("c10", b["1p_pose_term"]["rate_a"]), ("c3", b["1p_pose_term"]["rate_b"]), ("oracle", b["1_vs_oracle"]["rate_b"]),
                     ("r0", b["2_vs_r0"]["rate_b"]), ("r4", b["2_vs_r4"]["rate_b"])):
        L.append(f"| {ARM_NAMES[arm]} | {_cells(src)} |")
    L += ["", "## Upright and lying objects apart", "", "| view | cases (objects) | 1′ success | 1′ rule | 1 success | 1 rule |", "|---|---|---|---|---|---|"]
    for view in ("upright", "lying"):
        x = R[view]
        L.append(f"| {view} | {x['1p_pose_term']['cases']} ({x['1p_pose_term']['objects']}) | {_ci(x['1p_pose_term'], 'success')} | "
                 f"{_rule(x, '1p_reading')} | {_ci(x['1_vs_oracle'], 'success')} | {_rule(x, '1_reading')} |")
    L += ["", "## 3. Irreversible damage and the label that ends each episode", "",
          "| arm | view | irreversible % | labels |", "|---|---|---|---|"]
    for arm in ARMS + ("r0",):
        for view, label in (("all", "all"), ("no_thin", "without thin plastic")):
            x = R["3_damage"][view][arm]
            L.append(f"| {ARM_NAMES[arm]} | {label} | {_pc(x['rates']['irr'], 2)} | {dict(sorted(x['labels'].items()))} |")
    nt = R["4_to_6_no_thin"]
    L += ["", "## 4. By kind (success % | irreversible % | give-up %)", ""] + _md4(R) + ["Without thin plastic:", ""] + _md4(nt)
    L += ["## 5. The no-row cases (Stage 2 group a)", ""] + _md5(R) + ["Without thin plastic: " + _md5(nt)[0], ""]
    L += ["## 6. The smallest pose change that places the object, against the oracle arm's first pinch", ""] + _md6(R)
    L += ["Without thin plastic:", ""] + _md6(nt)
    return "\n".join(L) + "\n"


def poseset_figure(rep: dict[str, Any], path: str) -> None:
    """Simulated success against irreversible damage on the paired test cases: each Stage 2 rule's curve (its locked point
    ringed), R4 (a star), the oracle arm (a diamond) and the two capability ceilings (a square: 10 rows, a triangle: the 3
    current rows), on one pair of axes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from robots.mujoco_lab.curve import RULE_COLOURS, RULE_NAMES, _end_labels

    f = rep["figure"]
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ends = []
    for rule, pts in f["curves"].items():
        if rule == "R4":
            continue
        x, y = [100 * p["irr"] for p in pts], [100 * p["success"] for p in pts]
        ax.plot(x, y, color=RULE_COLOURS[rule], lw=1.6, marker="o", ms=3.5, mec="white", mew=0.8, zorder=3, label=RULE_NAMES[rule])
        at = next(p for p in pts if p["knob"] == f["locked"][rule])
        ax.scatter([100 * at["irr"]], [100 * at["success"]], s=80, facecolors="none", edgecolors=RULE_COLOURS[rule], linewidths=1.4,
                   zorder=4)
        ends.append((x[-1], y[-1], rule))
    r4 = f["curves"]["R4"][0]
    ax.scatter([100 * r4["irr"]], [100 * r4["success"]], marker="*", s=110, color="#52514e", edgecolors="white", linewidths=1,
               zorder=5, label=RULE_NAMES["R4"])
    ax.annotate("R4", (100 * r4["irr"], 100 * r4["success"]), xytext=(5, -9), textcoords="offset points", fontsize=8, color="#52514e")
    for k, marker, label, dx, ha in (("oracle", "D", "oracle arm: R0 at r 2, belief at the true setting", -7, "right"),
                                     ("c3", "^", "3-row ceiling: any current pinch places it", -7, "right"),
                                     ("c10", "s", "10-row ceiling: any of the ten pinches places it", 7, "left")):
        o = f[k]
        ax.scatter([100 * o["irr"]], [100 * o["success"]], marker=marker, s=60, color="#1f1e1c", edgecolors="white", linewidths=1,
                   zorder=6, label=label)
        ax.annotate(ARM_NAMES[k], (100 * o["irr"], 100 * o["success"]), xytext=(dx, 3), textcoords="offset points", ha=ha, fontsize=8,
                    color="#1f1e1c")
    ax.set_xlim(left=0.0)
    _end_labels(ax, ends)
    ax.set_xlabel("irreversible damage (% of episodes)")
    ax.set_ylabel("task success (% of episodes)")
    ax.set_title(f"Test objects, {f['c10']['n']} paired cases; rings: Stage 2's locked points", fontsize=10)
    ax.grid(True, color="#e6e5e0", linewidth=0.8, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def figure(a: argparse.Namespace) -> str:
    """The figure again from a finished summary's poseset_summary.json (its figure data only; no records file is opened)."""
    from robots.mujoco_lab.curve import _sha

    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    poseset_figure(json.load(open(a.summary)), a.out)
    print(f"wrote {a.out} from {a.summary} (sha256 {_sha(a.summary)})", flush=True)
    return a.out


def main(argv: list[str] | None = None) -> Any:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="the pose rows' simulations")
    r.add_argument("--stage1", required=True, help="comma-separated Stage 1 run dirs (Stage 2's --stage1)")
    r.add_argument("--out", required=True)
    r.add_argument("--cases", default=None, help="comma-separated object:idx cases in place of the shard")
    r.add_argument("--sides", default="test")
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--shards", type=int, default=1)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--replay", action="store_true", help="the 2-seed gate: every simulated row again at the second reset seed")
    s = sub.add_parser("smoke", help="the smoke report (validation runs only), both gates and the full run's projection")
    s.add_argument("--runs", required=True, help="comma-separated smoke run dirs")
    s.add_argument("--stage2", required=True, help="comma-separated Stage 2 run dirs")
    s.add_argument("--out", required=True)
    m = sub.add_parser("summary", help="the readings from finished test runs (opens the pose set's test records once)")
    m.add_argument("--runs", required=True, help="comma-separated pose-set test run dirs")
    m.add_argument("--stage2", required=True, help="comma-separated Stage 2 run dirs")
    m.add_argument("--stage2-summary", required=True, help="Stage 2's stage2_summary.json")
    m.add_argument("--oracle", required=True, help="comma-separated oracle-arm test run dirs")
    m.add_argument("--oracle-summary", required=True, help="the oracle arm's oracle_summary.json")
    m.add_argument("--locked", required=True, help="Stage 2's locked.json")
    m.add_argument("--out", required=True)
    g = sub.add_parser("figure", help="the figure again from a finished summary's json (reads no records)")
    g.add_argument("--summary", required=True, help="poseset_summary.json")
    g.add_argument("--out", required=True, help="the png to write")
    a = p.parse_args(argv)
    return {"run": run, "smoke": smoke, "summary": summary, "figure": figure}[a.cmd](a)


if __name__ == "__main__":
    main(sys.argv[1:])
