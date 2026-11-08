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


def trace_sha(trace: Any) -> str:
    """A hash of every field of every trace step, floats to 9 digits (the final state's hash rounds qpos the same way)."""
    return _sha_of(_round9(list(trace)))


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
    evpi_sim.seeds_of's second seed (replay_rows)."""
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
    a = p.parse_args(argv)
    return {"run": run, "smoke": smoke}[a.cmd](a)


if __name__ == "__main__":
    main(sys.argv[1:])
