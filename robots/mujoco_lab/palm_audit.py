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

"""The palm-contact audit (daily-logs design 2026-11-09-palm-contact-audit-design.md): sixdof's kinematic clearance
(clearance_on_path) of the hand bodies the sim never collides (the palm, the wrist and the finger boxes) along today's
planned hand path (MuJoCoLabEnv.hand_path) of every row that Stage 2, the oracle arm, the pose set and the GraspGen pilot
simulated, with the row's own action, at the object's pose where the row started. Geometry only: a second attempt
starts where its first attempt left the object, so that first attempt is replayed through curve.LabCase's own fresh and
attempt, under the gate that the replayed label, artifact and error equal the stored ones."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from typing import Any

import numpy as np

from robots.mujoco_lab import sixdof
from robots.mujoco_lab.adapter import A_TRANSPORT

SOURCES = ("stage2", "oracle", "poseset", "pilot")
FLAG_MM = {"any": 0.0, "gt1": 1.0, "gt2": 2.0, "gt5": 5.0}    # a row is flagged when its deepest overlap is deeper (mm)
PRIMARY = "gt1"                                              # the design's primary threshold


def parse_action(a: str) -> tuple[str, dict[str, Any]]:
    """A menu row's action key (evpi_props.action_key: "<row id>|<the action's JSON>") as (row id, action)."""
    rid, js = a.split("|", 1)
    return rid, json.loads(js)


def action_of(rows: list[dict[str, Any]], rid: str) -> dict[str, Any]:
    """The action of the one row ``rid`` of a stored menu; its action key must name the same row."""
    hit = [r for r in rows if r["id"] == rid]
    if len(hit) != 1:
        raise ValueError(f"{len(hit)} rows {rid} in the menu")
    got, act = parse_action(hit[0]["action"])
    if got != rid:
        raise ValueError(f"row {rid}'s action key names {got}")
    return act


def flags_of(c: dict[str, Any] | None) -> dict[str, Any]:
    """A clearance result (clearance_on_path's; None when clear) as the audit's fields: the deepest overlap (mm), the flag at
    each FLAG_MM threshold, and per failing pair its body, obstacle kind, first phase, steps, deepest overlap (mm), geoms and
    per phase its steps and deepest overlap (mm)."""
    by = (c or {}).get("by_pair") or {}
    pairs = {pr: {"body": pr.split("/")[0], "kind": pr.split("/")[1], "first_phase": b["phase"], "steps": int(b["steps"]),
                  "depth_mm": 1000.0 * float(b["depth"]), "geoms": list(b["geoms"]),
                  "phases": {ph: {"steps": int(q["steps"]), "depth_mm": 1000.0 * float(q["depth"])} for ph, q in b["phases"].items()}}
             for pr, b in sorted(by.items())}
    depth = max((p["depth_mm"] for p in pairs.values()), default=0.0)
    return {"depth_mm": depth, "flags": {k: depth > v for k, v in FLAG_MM.items()}, "pairs": pairs}


def replay_differs(sim: dict[str, Any], d1: dict[str, Any], at_rest: bool) -> dict[str, list[Any]]:
    """The replay gate (curve.simulate_case's own test for a replayed first attempt): the fields of (label, artifact, error)
    where the replay ``d1`` (LabCase.attempt's payload; ``at_rest`` from its fresh) differs from the stored simulation,
    each as [stored, replayed]. Empty: the replay passes."""
    again = {"label": d1.get("outcome"), "artifact": bool(d1.get("artifact")) or not at_rest, "error": d1.get("error")}
    return {k: [sim.get(k), v] for k, v in again.items() if sim.get(k) != v}


def episode_rows(rec: dict[str, Any], key: str) -> list[tuple[int, str, str, str]]:
    """The rows one rule's episode executed on a Stage 2 or oracle record (curve.episode), as (attempt, its first row, the
    row, its label)."""
    from robots.mujoco_lab.curve import episode

    e = episode(rec["decisions"], rec["sims"], key)
    return [(k + 1, e["rows"][0], row, lab) for k, (row, lab) in enumerate(zip(e["rows"], e["labels"]))]


def ceiling_without(rows: list[dict[str, Any]], drop: set[str], current_only: bool) -> dict[str, Any]:
    """The pose set's ceiling (curve_poseset.ceiling) over its rows less the rows ``drop`` names."""
    from robots.mujoco_lab.curve_poseset import ceiling

    return ceiling([r for r in rows if r["id"] not in drop], current_only)


def proposal_at(env: Any, s: float) -> Any:
    """The pinch tools._attempt executes for an action at fraction ``s``: the one default proposal at s (to 1e-9)."""
    hit = [p for p in env.proposals() if abs(float(p.s) - float(s)) < 1e-9]
    if len(hit) != 1:
        raise ValueError(f"{len(hit)} default proposals at s {s}")
    return hit[0]


def today_path(env: Any, prop: Any, act: dict[str, Any]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Today's planned hand path of ``prop`` under the action's speeds, destination and release height
    (MuJoCoLabEnv.hand_path), each yaw as a rotation about the vertical (clearance_on_path's form)."""
    path = env.hand_path(prop, act["v_lift"], act["v_transport"], act["destination"], act.get("release_height"),
                         a_transport=act.get("a_transport", A_TRANSPORT))
    return [(ph, pos, sixdof._rz(float(yaw))) for ph, pos, yaw in path]


def pads_on_extent(env: Any, prop: Any) -> tuple[float, float]:
    """The left and right finger joints that put each pad's inner face on its side of the hulls inside the pads' slab
    (MuJoCoLabEnv.pinch_extent's lo and hi along the closing axis): the two finger motors close independently, so each pad
    stops on the object's face on its own side (the stored width is a lower bound on the slab's, and the slab can be off
    centre). Within the joints' range [0, MAX_OPEN - 0.002]."""
    sc = env.s["scene"]
    lo, hi = env.pinch_extent(prop.point, prop.yaw)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"no hull inside the pads' slab at s {prop.s}")
    top, t = sc.MAX_OPEN - 0.002, 2.0 * sc.PAD_HALF[0]
    return float(np.clip(lo + sc.MAX_OPEN - t, 0.0, top)), float(np.clip(sc.MAX_OPEN - t - hi, 0.0, top))


def audit_path(env: Any, prop: Any, act: dict[str, Any]) -> dict[str, Any]:
    """flags_of the clearance of ``prop``'s planned path under ``act``: the palm, wrist and finger boxes against the object
    from the pregrasp to the close (the design's; sixdof's own check leaves the finger boxes to the width rule) and against
    the table, the fixtures and the destination at every step, the closed fingers each on its side of the slab
    (pads_on_extent)."""
    return flags_of(sixdof.clearance_on_path(env, today_path(env, prop, act), float(prop.width), act["destination"],
                                             fingers_on_object=True, closed_q=pads_on_extent(env, prop)))


def state_sha(env: Any) -> str:
    """A hash of the sim's state (time, qpos, qvel, act; to 9 digits): an audit must leave it as it found it."""
    d = env.sim.d
    x = np.concatenate([[float(d.time)], np.asarray(d.qpos, dtype=float), np.asarray(d.qvel, dtype=float), np.asarray(d.act, dtype=float)])
    return hashlib.sha256(np.round(x, 9).tobytes()).hexdigest()[:16]


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:300]}"


def audit_row(env: Any, act: dict[str, Any], prop: Any = None, **key: Any) -> dict[str, Any]:
    """One row's audit: its ``key`` fields, its pinch (``prop``; None: tools._attempt's, proposal_at the action's s), the
    action's destination and speeds, and audit_path's fields; an exception is the row's error. Raises when the sim's state
    changed (a geometry check moves nothing)."""
    before = state_sha(env)
    row: dict[str, Any] = dict(key, destination=act["destination"], v_lift=float(act["v_lift"]), v_transport=float(act["v_transport"]),
                               a_transport=float(act.get("a_transport", A_TRANSPORT)), release_height=act.get("release_height"), error=None)
    try:
        p = proposal_at(env, act["s"]) if prop is None else prop
        row.update(s=float(p.s), point=[float(x) for x in p.point], yaw=float(p.yaw), width=float(p.width))
        row.update(audit_path(env, p, act))
    except Exception as exc:  # noqa: BLE001  a row the audit cannot place is that row's error
        row["error"] = _err(exc)
    if state_sha(env) != before:
        raise RuntimeError("the audit changed the sim's state")
    return row


# ---------------------------------------------------------------- the cases

def _prims(bundle: str, fold: int) -> tuple[Any, Any]:
    """curve.stage2_object's primitives on one fold's bundle (no probe models, the job's force noise) and its world."""
    from robots.mujoco_lab import calibrate as cal
    from robots.mujoco_lab.prior import fragility_world

    prims = cal._primitives(os.path.join(bundle, "folds", f"fold_{int(fold)}"), None, None)
    if getattr(prims, "reset_s", None) is not None:
        raise RuntimeError("the curve scores the absorbing episode only (this bundle restores the object after damage)")
    prims.probe_models = {}
    prims.force_noise = cal.job_force_noise(None, prims.manifest)
    prims.env.force_noise = prims.force_noise
    return prims, fragility_world(prims.manifest)


def reset_lab(lab: Any, reset_seed: int) -> bool:
    """LabCase.fresh's reset (curve_poseset.poseset_case and curve_graspgen._setup reset the same way); at_rest."""
    env = lab.prims.env
    env.reset(lab.o, lab.theta, seed=int(reset_seed), com_end=lab.com_end, fixture=lab.fixture, fill=lab.hidden["fill"],
              frag=lab.hidden["frag"])
    return bool(dict(getattr(env, "reset_info", None) or {}).get("at_rest", True))


def _case_out(rec: dict[str, Any]) -> dict[str, Any]:
    return {"object": rec["object"], "idx": int(rec["idx"]), "side": rec["side"], "rows": [], "replays": [], "stop": None, "error": None}


def lab_case(prims: Any, world: Any, seed: int, rec1: dict[str, Any], rec: dict[str, Any], oracle: bool = False) -> dict[str, Any]:
    """A Stage 2 or oracle-arm case: each simulation's first row from the case's reset (the fingerprint Stage 1's), with
    the first menu's action (Stage 1's rows; the oracle's own); then per simulation with second rows its first attempt
    replayed through LabCase (OracleLabCase) fresh and attempt, under replay_differs's gate, and each second row (the
    stored second menu's action) from where the replay left the object. A second row of a replay that differs, stops or
    raises is not audited."""
    from robots.mujoco_lab.com_gap import fingerprint
    from robots.mujoco_lab.curve import LabCase
    from robots.mujoco_lab.curve_oracle import OracleLabCase

    out = _case_out(rec)
    env = prims.env
    lab = (OracleLabCase if oracle else LabCase)(prims, rec["object"], int(rec["idx"]), seed, world)
    if lab.destination != rec1["destination"]:
        return dict(out, stop=f"the case's draw gives the destination {lab.destination}, Stage 1's is {rec1['destination']}")
    menu1, ref = (rec["rows"], dict(rec, fingerprint=rec1["fingerprint"])) if oracle else (rec1["rows"], rec1)
    if any(s["seed_k"] != 0 for s in rec["sims"]):
        return dict(out, stop="a simulation at another reset seed")
    reset_state: dict[int, str] = {}
    for s in rec["sims"]:
        reset_lab(lab, s["reset_seed"])
        if fingerprint(env) != rec1["fingerprint"]:
            return dict(out, stop=f"{s['first']}: the reset's fingerprint differs from Stage 1's")
        reset_state[int(s["reset_seed"])] = state_sha(env)
        out["rows"].append(audit_row(env, action_of(menu1, s["first"]), seed_k=0, first=s["first"], attempt=1, row=s["first"]))
    for s in rec["sims"]:
        if not s["second"]:
            continue
        t0 = time.time()
        rp: dict[str, Any] = {"first": s["first"], "differs": None, "stop": None, "error": None, "fresh_state_equal": None}
        try:
            store, stop, at_rest = lab.fresh(int(s["reset_seed"]), ref)
            if stop:
                rp["stop"] = stop
            else:
                rp["fresh_state_equal"] = state_sha(env) == reset_state[int(s["reset_seed"])]
                rp["differs"] = replay_differs(s, lab.attempt(store, s["first"]), at_rest)
        except Exception as exc:  # noqa: BLE001  a replay that raises leaves its second rows not audited
            rp["error"] = _err(exc)
        rp["audited"] = rp["differs"] == {} and rp["stop"] is None and rp["error"] is None
        for row2 in s["second"]:
            key = {"seed_k": 0, "first": s["first"], "attempt": 2, "row": row2}
            out["rows"].append(audit_row(env, action_of(s["menu2"]["rows"], row2), **key) if rp["audited"]
                               else dict(key, error=None, not_audited="the replayed first attempt " + ("differs" if rp["differs"] else "stopped or raised")))
        rp["seconds"] = round(time.time() - t0, 2)
        out["replays"].append(rp)
    return out


def poseset_case(prims: Any, world: Any, seed: int, rec1: dict[str, Any], rec: dict[str, Any]) -> dict[str, Any]:
    """A pose-set case: every simulated row from the case's reset (the fingerprint Stage 1's), its pinch fetched again
    (curve_poseset.simulate_rows's: it must be the stored point and yaw) and the stored base action."""
    from robots.mujoco_lab.com_gap import fingerprint
    from robots.mujoco_lab.curve import LabCase
    from robots.mujoco_lab.curve_poseset import FRACS, TURN, commit_std_base
    from robots.mujoco_lab.evpi_sim import seeds_of

    out = _case_out(rec)
    env = prims.env
    lab = LabCase(prims, rec["object"], int(rec["idx"]), seed, world)
    if lab.destination != rec1["destination"]:
        return dict(out, stop=f"the case's draw gives the destination {lab.destination}, Stage 1's is {rec1['destination']}")
    reset_lab(lab, seeds_of(seed, rec["object"], int(rec["idx"]))[0])
    if fingerprint(env) != rec1["fingerprint"]:
        return dict(out, stop="the reset's fingerprint differs from Stage 1's")
    base = rec["base"]
    if base != commit_std_base(env, rec["object"], lab.destination):
        return dict(out, stop="the stored base is not commit_std's")
    by = {d: {p.s: p for p in env.proposals(FRACS, turn=t)} for d, t in TURN.items()}
    for r in rec["rows"]:
        if "seconds" not in r:
            continue
        p = by[r["dir"]][r["f"]]
        if [float(x) for x in p.point] != r["point"] or float(p.yaw) != r["yaw"]:
            return dict(out, stop=f"{r['id']}: the pinch after the reset differs from the stored one")
        out["rows"].append(audit_row(env, base, prop=p, id=r["id"], current=bool(r["current"])))
    return out


def pilot_case(prims: Any, world: Any, seed: int, rec1: dict[str, Any], rec: dict[str, Any]) -> dict[str, Any]:
    """A GraspGen pilot case: every simulated row (its kept rows and its gate row) from the case's reset (the fingerprint
    Stage 1's, the hulls the stored ones), at the stored pinch (curve_graspgen._proposal) with the stored base action."""
    from robots.mujoco_lab import curve_graspgen as gg
    from robots.mujoco_lab.com_gap import fingerprint
    from robots.mujoco_lab.curve import LabCase
    from robots.mujoco_lab.evpi_sim import seeds_of

    out = _case_out(rec)
    env = prims.env
    lab = LabCase(prims, rec["object"], int(rec["idx"]), seed, world)
    if lab.destination != rec1["destination"]:
        return dict(out, stop=f"the case's draw gives the destination {lab.destination}, Stage 1's is {rec1['destination']}")
    reset_lab(lab, seeds_of(seed, rec["object"], int(rec["idx"]))[0])
    if fingerprint(env) != rec1["fingerprint"]:
        return dict(out, stop="the reset's fingerprint differs from Stage 1's")
    if gg.hulls_sha(env) != rec["hull_sha"]:
        return dict(out, stop="the hulls after the reset differ from the stored ones")
    for r in rec["rows"]:
        if "seconds" in r:
            out["rows"].append(audit_row(env, rec["base"], prop=gg._proposal(r), id=r["id"], gate=bool(r.get("gate"))))
    return out


def audited_cases(source: str, recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The records a source's audit covers: Stage 2's and the oracle arm's cases that are not dropped (the readings' cases),
    and the pose set's and the pilot's cases with a simulated row."""
    if source in ("stage2", "oracle"):
        return [r for r in recs if not r["dropped"] and not r.get("skipped")]
    return [r for r in recs if any("seconds" in x for x in r.get("rows") or [])]


def _shard(objects: list[str], k: int, n: int) -> list[str]:
    from robots.mujoco_lab.curve import shard_objects

    return shard_objects(objects, k, n)


def run(a: argparse.Namespace) -> str:
    """One audit unit: the source's records (records_<side>.jsonl of --runs, which must agree on Stage 1, the bundle and
    the seed) on the shard's objects, one line per case in --out/audit_records.jsonl; run_config.json and run_metrics.json
    (counts and times; wandb when on). Refuses an existing --out/audit_records.jsonl."""
    from robots.mujoco_lab.curve import _sha, load_stage1, records_path
    from robots.mujoco_lab.estimation_check import _revision
    from robots.mujoco_lab.identify_fit import require_bundle_generator
    from robots.mujoco_lab.tracking import log_run

    path = os.path.join(a.out, "audit_records.jsonl")
    if os.path.exists(path):
        raise SystemExit(f"{path} exists: pick a new --out")
    os.makedirs(a.out, exist_ok=True)
    dirs = a.runs.split(",")
    cfgs = [json.load(open(os.path.join(d, "run_config.json"))) for d in dirs]
    for k in ("stage1", "bundle", "seed"):
        if len({json.dumps(c.get(k)) for c in cfgs}) > 1:
            raise SystemExit(f"the runs differ in {k}")
    cfg = cfgs[0]
    generator = require_bundle_generator(cfg["bundle"])
    recs1, _ = load_stage1(cfg["stage1"].split(","))
    by1 = {(r["object"], int(r["idx"])): r for r in recs1}
    files = [records_path(d, a.side) for d in dirs]
    recs = audited_cases(a.source, [json.loads(line) for f in files for line in open(f) if line.strip()])
    if a.cases:
        want = {(c.rsplit(":", 1)[0], int(c.rsplit(":", 1)[1])) for c in a.cases.split(",")}
        recs = [r for r in recs if (r["object"], int(r["idx"])) in want]
        if len(recs) != len(want):
            raise SystemExit(f"{len(want) - len(recs)} of the --cases are not audited cases of the runs")
    objs = set(_shard([r["object"] for r in recs], a.shard, a.shards))
    mine = sorted((r for r in recs if r["object"] in objs), key=lambda r: (r["object"], int(r["idx"])))
    config = {**vars(a), **_revision(), "generator": generator, "bundle": cfg["bundle"], "seed": cfg["seed"], "stage1": cfg["stage1"],
              "records_sha256": {f: _sha(f) for f in files}, "source_code_revisions": sorted({c.get("code_revision") for c in cfgs}),
              "flag_mm": FLAG_MM, "cases": [(r["object"], int(r["idx"])) for r in mine]}
    with open(os.path.join(a.out, "run_config.json"), "w") as f:
        json.dump(config, f, indent=1)
    t_all, stop, done = time.time(), None, []
    with open(path, "w") as f:
        for o in sorted(objs):
            items = [r for r in mine if r["object"] == o]
            folds = {by1[(o, int(r["idx"]))]["fold"] for r in items}
            if len(folds) != 1:
                stop = f"{o}: cases on {len(folds)} folds"
                break
            prims, world = _prims(cfg["bundle"], folds.pop())
            for r in items:
                t0, rec1 = time.time(), by1[(o, int(r["idx"]))]
                try:
                    if a.source == "poseset":
                        res = poseset_case(prims, world, int(cfg["seed"]), rec1, r)
                    elif a.source == "pilot":
                        res = pilot_case(prims, world, int(cfg["seed"]), rec1, r)
                    else:
                        res = lab_case(prims, world, int(cfg["seed"]), rec1, r, oracle=a.source == "oracle")
                except Exception as exc:  # noqa: BLE001  a case that raises is reported, and the unit stops
                    res = dict(_case_out(r), error=_err(exc))
                res.update(source=a.source, seconds=round(time.time() - t0, 2))
                f.write(json.dumps(res) + "\n")
                f.flush()
                done.append(res)
                print(f"{o} {r['idx']}: {len(res['rows'])} rows, {sum(bool(x.get('flags', {}).get(PRIMARY)) for x in res['rows'])} flagged > 1 mm, "
                      f"{len(res['replays'])} replays ({sum(not x['audited'] for x in res['replays'])} not passed), {res['seconds']:.1f} s"
                      + (f", STOP {res['stop']}" if res["stop"] else "") + (f", ERROR {res['error']}" if res["error"] else ""), flush=True)
                if res["stop"] or res["error"]:
                    stop = f"{o} {r['idx']}: {res['stop'] or res['error']}"
                    break
            if stop:
                break
    rows = [x for r in done for x in r["rows"]]
    reps = [x for r in done for x in r["replays"]]
    metrics = {"cases": len(done), "of_cases": len(mine), "stop": stop, "rows": len(rows),
               "rows_audited": sum("flags" in x for x in rows), "row_errors": sum(bool(x.get("error")) for x in rows),
               "not_audited": sum("not_audited" in x for x in rows),
               "flagged": {k: sum(bool(x.get("flags", {}).get(k)) for x in rows) for k in FLAG_MM},
               "replays": len(reps), "replays_differ": sum(bool(x["differs"]) for x in reps),
               "replays_stopped": sum(bool(x["stop"] or x["error"]) for x in reps),
               "fresh_state_differs": sum(x["fresh_state_equal"] is False for x in reps),
               "replay_seconds": round(sum(x["seconds"] for x in reps), 1), "wall_seconds": round(time.time() - t_all, 1)}
    metrics["wandb"] = log_run("palm-audit", config, {k: v for k, v in metrics.items() if not isinstance(v, (list, dict, str)) or k == "flagged"})
    with open(os.path.join(a.out, "run_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print(f"done: {json.dumps(metrics)}", flush=True)
    if stop:
        raise SystemExit(3)
    return a.out


# ---------------------------------------------------------------- the summary

SIDES_OF = (("stage2", "test"), ("oracle", "test"), ("poseset", "test"), ("poseset", "validation"), ("pilot", "test"), ("pilot", "validation"))


def row_key(source: str, side: str, obj: str, idx: int, row: dict[str, Any]) -> tuple[Any, ...]:
    """An audited row's key: (source, side, object, idx, its first row, attempt, row) for Stage 2 and the oracle arm, and
    (source, side, object, idx, row id) for the pose set and the pilot."""
    if source in ("stage2", "oracle"):
        return (source, side, obj, int(idx), row["first"], int(row["attempt"]), row["row"])
    return (source, side, obj, int(idx), row["id"])


def index_rows(audit: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Every audited row by row_key; a row audited twice raises."""
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    for r in audit:
        for x in r["rows"]:
            k = row_key(r["source"], r["side"], r["object"], r["idx"], x)
            if k in out:
                raise ValueError(f"{k} is audited twice")
            out[k] = x
    return out


def status(x: dict[str, Any] | None, level: str = PRIMARY) -> str:
    """A row's audit at one FLAG_MM threshold: "flagged", "clear", or "not audited" (no audit line, an error, or a second
    row whose replay did not pass)."""
    if x is None or x.get("error") or "flags" not in x:
        return "not audited"
    return "flagged" if x["flags"][level] else "clear"


def worst(states: list[str]) -> str:
    """flagged if any row is, else not audited if any row is, else clear (also for no row)."""
    return "flagged" if "flagged" in states else ("not audited" if "not audited" in states else "clear")


def episode_status(idx: dict[tuple[Any, ...], dict[str, Any]], source: str, side: str, rec: dict[str, Any], key: str,
                   level: str = PRIMARY) -> str:
    """worst over the rows one rule's episode executed (episode_rows); a give-up executes none and is clear."""
    return worst([status(idx.get((source, side, rec["object"], int(rec["idx"]), first, att, row)), level)
                  for att, first, row, _ in episode_rows(rec, key)])


def label_shares(items: list[tuple[str | None, str]]) -> dict[str, dict[str, Any]]:
    """Reading 2 over (label, status) pairs: per group (placed, irreversible, all) its rows, the rows audited, the flagged
    rows and their share of the audited rows (None when none is audited)."""
    from rpent.skills.pick_place.protocol import IRREVERSIBLE

    groups = (("placed", lambda y: y == "placed_stable"), ("irreversible", lambda y: y in IRREVERSIBLE), ("all", lambda y: True))
    out = {}
    for name, test in groups:
        st = [s for y, s in items if test(y)]
        aud, f = sum(s != "not audited" for s in st), sum(s == "flagged" for s in st)
        out[name] = {"rows": len(st), "audited": aud, "flagged": f, "share": f / aud if aud else None}
    return out


def reading1(rows: list[tuple[str, dict[str, Any]]], level: str = PRIMARY) -> dict[str, Any]:
    """Reading 1 over (object, audit row) pairs: the rows, the rows audited, the flagged rows at each FLAG_MM threshold; at
    ``level`` the flagged rows by body, by obstacle and by phase (a row counts under each with a pair, or a pair's phase,
    deeper than the threshold), and per object its audited rows, its flagged rows and the median and the maximum depth
    over them (mm; None without one)."""
    thr = FLAG_MM[level]
    aud = [(o, x) for o, x in rows if status(x, level) != "not audited"]
    body: Counter = Counter()
    kind: Counter = Counter()
    phase: Counter = Counter()
    per: dict[str, list[float]] = {}
    n: Counter = Counter()
    for o, x in aud:
        n[o] += 1
        per.setdefault(o, [])
        deep = [v for v in x["pairs"].values() if v["depth_mm"] > thr]
        body.update({v["body"] for v in deep})
        kind.update({v["kind"] for v in deep})
        phase.update({ph for v in deep for ph, q in v["phases"].items() if q["depth_mm"] > thr})
        if x["flags"][level]:
            per[o].append(float(x["depth_mm"]))
    return {"rows": len(rows), "audited": len(aud), "flagged": {k: sum(bool(x["flags"][k]) for _, x in aud) for k in FLAG_MM},
            "by_body": dict(body), "by_obstacle": dict(kind), "by_phase": dict(phase),
            "by_object": {o: {"rows": n[o], "flagged": len(d), "depth_median_mm": float(np.median(d)) if d else None,
                              "depth_max_mm": max(d) if d else None} for o, d in sorted(per.items())}}


def differs(got: dict[str, Any], want: dict[str, Any], names: tuple[str, ...] = ("success", "irr")) -> list[str]:
    """The gate against a published paired difference: each field (the cases; diff, lo and hi per name) that is not equal."""
    bad = [] if got.get("cases") == want.get("cases") else [f"cases {got.get('cases')} vs {want.get('cases')}"]
    return bad + [f"{n}.{k} {got[n][k]!r} vs {want[n][k]!r}" for n in names for k in ("diff", "lo", "hi") if got[n][k] != want[n][k]]


def poseset_without(vs: list[dict[str, Any]], idx: dict[tuple[Any, ...], dict[str, Any]], side: str = "test",
                    level: str = PRIMARY) -> list[dict[str, Any]]:
    """Reading 3's pose-set views: each view with both ceilings over its rows less its simulated rows flagged at ``level``
    (ceiling_without); a simulated row that is not audited stays and is listed (kept_unaudited)."""
    out = []
    for v in vs:
        sim = [r for r in v["rows"] if "seconds" in r]
        st = {r["id"]: status(idx.get(("poseset", side, v["object"], int(v["idx"]), r["id"])), level) for r in sim}
        drop = [i for i, x in st.items() if x == "flagged"]
        out.append(dict(v, c10=ceiling_without(v["rows"], set(drop), False), c3=ceiling_without(v["rows"], set(drop), True), removed=drop,
                        kept_unaudited=[i for i, x in st.items() if x == "not audited"]))
    return out


def stored_labels(source: str, side: str, recs: list[dict[str, Any]]) -> dict[tuple[Any, ...], Any]:
    """Each simulated row's stored label by row_key (the reading-4 list carries it for G1')."""
    out: dict[tuple[Any, ...], Any] = {}
    for r in recs:
        o, i = r["object"], int(r["idx"])
        if source in ("stage2", "oracle"):
            for s in r.get("sims") or []:
                out[(source, side, o, i, s["first"], 1, s["first"])] = s["label"]
                for row2, y in (s.get("second") or {}).items():
                    out[(source, side, o, i, s["first"], 2, row2)] = y["label"]
        else:
            for x in r.get("rows") or []:
                if "seconds" in x:
                    out[(source, side, o, i, x["id"])] = x.get("label")
    return out


def _units(q: str) -> list[str]:
    import glob

    return sorted(glob.glob(os.path.join(q, "unit_*")), key=lambda d: int(d.rsplit("_", 1)[1]))


def load_audit(q: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The audit units in queue dir ``q``: each ended with EXIT 0 and no stop, on one clean code revision with FLAG_MM, and
    the source records each read are unchanged; no case twice in a (source, side); no case line holds a stop or an error."""
    from robots.mujoco_lab.curve import _sha
    from robots.mujoco_lab.curve_oracle import _finished

    recs, cfgs = [], []
    for d in _units(q):
        c = _finished(d)
        for f, h in c["records_sha256"].items():
            if _sha(f) != h:
                raise SystemExit(f"{f}: the records changed since the audit read them")
        cfgs.append(c)
        recs += [json.loads(line) for line in open(os.path.join(d, "audit_records.jsonl"))]
    if not cfgs:
        raise SystemExit(f"{q}: no audit unit")
    if len({c["code_revision"] for c in cfgs}) != 1 or any(c["dirty"] for c in cfgs):
        raise SystemExit("the audit units ran on more than one code revision, or on a dirty tree")
    if any(c["flag_mm"] != FLAG_MM for c in cfgs):
        raise SystemExit("an audit unit used other thresholds")
    twice = [k for k, n in Counter((r["source"], r["side"], r["object"], int(r["idx"])) for r in recs).items() if n > 1]
    if twice:
        raise SystemExit(f"{len(twice)} cases audited twice, the first {twice[0]}")
    bad = [r for r in recs if r.get("stop") or r.get("error")]
    if bad:
        raise SystemExit(f"{len(bad)} audited cases hold a stop or an error, the first {bad[0]['object']} {bad[0]['idx']}")
    return recs, cfgs


def _pairs_cell(b: dict[str, Any] | None, name: str) -> str:
    if not b or b.get(name) is None:
        return "–"
    x = b[name]
    return f"{100 * x['diff']:+.2f} [{100 * x['lo']:+.2f}, {100 * x['hi']:+.2f}]"


def summary_markdown(rep: dict[str, Any]) -> str:
    """audit_summary.md: the counts, the gate, and readings 1 to 4 (numbers only; the README states them)."""
    L = ["# Palm-contact audit: summary", "", f"Primary threshold: deeper than {FLAG_MM[PRIMARY]:g} mm. Depth: MuJoCo's penetration "
         "distance (the shortest move that separates the two geoms).", "", "## Counts", ""]
    c = rep["counts"]
    L += [f"- {k}: {json.dumps(v)}" for k, v in c.items()]
    L += ["", "## Gate (the published numbers recomputed from the records)", "", "- " + ("all equal" if not any(rep["gate"].values())
                                                                                      else json.dumps(rep["gate"])), "",
          "## Reading 1: flagged rows per experiment", "",
          "| experiment | rows | audited | any | > 1 mm | > 2 mm | > 5 mm | by body (> 1 mm) | by obstacle (> 1 mm) | by phase (> 1 mm) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in rep["readings"]["1"].items():
        f = r["flagged"]
        L.append(f"| {name} | {r['rows']} | {r['audited']} | " + " | ".join(f"{f[k]} ({100 * f[k] / r['audited']:.1f} %)" if r["audited"] else "–"
                                                                       for k in FLAG_MM)
                 + f" | {json.dumps(r['by_body'])} | {json.dumps(r['by_obstacle'])} | {json.dumps(r['by_phase'])} |")
    L += ["", "Objects with a row flagged at > 1 mm (rows audited, flagged, depth median and maximum in mm):", ""]
    for name, r in rep["readings"]["1"].items():
        objs = [(o, x) for o, x in r["by_object"].items() if x["flagged"]]
        if objs:
            L.append(f"- {name}: " + "; ".join(f"{o} {x['flagged']}/{x['rows']} ({x['depth_median_mm']:.1f}, {x['depth_max_mm']:.1f})"
                                               for o, x in sorted(objs, key=lambda t: -t[1]["flagged"])))
    L += ["", "## Reading 2: flagged share (> 1 mm) among the executed rows, per arm", "",
          "| arm | placed rows | flagged | share | irreversible rows | flagged | share | all rows | flagged | share |", "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, r in rep["readings"]["2"][PRIMARY].items():
        L.append(f"| {arm} | " + " | ".join(f"{r[g]['audited']} | {r[g]['flagged']} | " + ("–" if r[g]["share"] is None else f"{100 * r[g]['share']:.1f} %")
                                           for g in ("placed", "irreversible", "all")) + " |")
    L += ["", "## Reading 3: headline sensitivity (points, 95 % object bootstrap), beside the published numbers", "",
          "| reading | published | " + " | ".join(f"unflagged at {k}" for k in FLAG_MM) + " |", "|---|---|" + "---|" * len(FLAG_MM)]
    pub, r3 = rep["published"], rep["readings"]["3"]
    three = ("success", "irr", "give_up")
    for key, label, names in (("stage2_r0_r4", "Stage 2 R0 − R4", ("success", "irr")), ("oracle_r0", "oracle − R0", three),
                              ("poseset_1p", "pose set 10-row − 3-row", three), ("poseset_1p_upright", "pose set 10-row − 3-row, upright", three)):
        for n in names:
            L.append(f"| {label}, {n} | {_pairs_cell(pub[key], n)} (n {pub[key]['cases']}) | "
                     + " | ".join(f"{_pairs_cell(r3[k][key], n)} (n {r3[k][key]['cases'] if r3[k][key] else 0})" for k in FLAG_MM) + " |")
    L += ["", "Cases left out of each sensitivity line (flagged, not audited) and pose-set rows removed:", ""]
    for k in FLAG_MM:
        L.append(f"- {k}: " + json.dumps({key: {x: r3[k][key].get(x) for x in ("excluded", "removed_rows", "changed", "undetermined", "kept_unaudited")
                                               if r3[k][key] and x in r3[k][key]} for key in r3[k]}))
    L += ["", "## Reading 4: the flagged list", "", f"- {rep['counts']['flagged_list_lines']} lines in flagged_rows.jsonl: every audited row "
          "with any penetration, and every row not audited, keyed by (source, side, object, idx, id) or (…, first, attempt, row)."]
    return "\n".join(L) + "\n"


def summary(a: argparse.Namespace) -> dict[str, Any]:
    """The audit's readings (design section 3) from the finished units in --queue: audit_summary.{json,md} and
    flagged_rows.jsonl in --out (refuses an existing summary), wandb. The source records are the published summaries'
    (Stage 2's, the oracle arm's, the pose set's; their record hashes must hold) and the audit units' for the pose set's
    validation side and the pilot; the units must cover exactly the cases audited_cases names. Labels are read from the
    records. First the gate: the published headline numbers recomputed from the records must be equal (else
    audit_summary_stop.json and exit 3)."""
    from robots.mujoco_lab import curve_oracle as co
    from robots.mujoco_lab import curve_poseset as cp
    from robots.mujoco_lab.curve import BOOT_DRAWS, BOOT_SEED, _sha, load_stage1, load_stage2, paired_bootstrap, records_path
    from robots.mujoco_lab.estimation_check import _revision
    from robots.mujoco_lab.tracking import log_run

    os.makedirs(a.out, exist_ok=True)
    jpath, spath = os.path.join(a.out, "audit_summary.json"), os.path.join(a.out, "audit_summary_stop.json")
    if os.path.exists(jpath) or os.path.exists(spath):
        raise SystemExit(f"{a.out}: a summary exists: pick a new --out")
    audit, cfgs = load_audit(a.queue)
    lk = json.load(open(a.locked))["lock"]
    k0, k4 = lk["R0"]["key"], lk["R4"]["key"]
    s2sum, osum, psum = (json.load(open(x)) for x in (a.stage2_summary, a.oracle_summary, a.poseset_summary))
    if [s2sum["q1"]["boot"]["a"], s2sum["q1"]["boot"]["b"]] != [k0, k4]:
        raise SystemExit("the Stage 2 summary compared other keys than the lock's R0 and R4")
    for name, sm in (("Stage 2", s2sum), ("oracle", osum), ("pose-set", psum)):
        if sm["provenance"]["locked"].split(" (sha256 ")[1].rstrip(")") != _sha(a.locked):
            raise SystemExit(f"the {name} summary read another lock")
    for name, sm in (("oracle", osum), ("pose-set", psum)):
        if list(sm["provenance"]["keys"]) != [k0, k4]:
            raise SystemExit(f"the {name} summary read other keys")
    s2dirs, odirs, pdirs = (sm["provenance"]["runs"].split(",") for sm in (s2sum, osum, psum))
    for d in s2dirs:
        if _sha(records_path(d, "test")) != s2sum["provenance"]["records_sha256"].get(f"{d} test"):
            raise SystemExit(f"{d}: the Stage 2 test records are not the ones its summary read")
    for d in odirs:
        if _sha(records_path(d, "test")) != osum["provenance"]["oracle_records_sha256"].get(d):
            raise SystemExit(f"{d}: the oracle test records are not the ones its summary read")
    for d in pdirs:
        if _sha(records_path(d, "test")) != psum["provenance"]["records_sha256"].get(d):
            raise SystemExit(f"{d}: the pose-set test records are not the ones its summary read")
    runs_of: dict[tuple[str, str], set[str]] = {}
    for c in cfgs:
        k = (c["source"], c["side"])
        if runs_of.setdefault(k, set(c["runs"].split(","))) != set(c["runs"].split(",")):
            raise SystemExit(f"the {k} units read different runs")
    if set(runs_of) != set(SIDES_OF):
        raise SystemExit(f"the units cover {sorted(runs_of)}, not {list(SIDES_OF)}")
    for k, dirs in ((("stage2", "test"), s2dirs), (("oracle", "test"), odirs), (("poseset", "test"), pdirs)):
        if runs_of[k] != set(dirs):
            raise SystemExit(f"the {k} units read other runs than the published summary")
    if len({c["stage1"] for c in cfgs}) != 1 or any(sorted(cfgs[0]["stage1"].split(",")) != sorted(sm["provenance"]["stage1"]) for sm in (osum, psum)):
        raise SystemExit("the units read another Stage 1 than the oracle and pose-set summaries")
    s2, _ = load_stage2(s2dirs, ("test",))
    recs1, _ = load_stage1(cfgs[0]["stage1"].split(","))
    src = {("stage2", "test"): s2}
    for k in SIDES_OF[1:]:
        src[k] = [json.loads(line) for d in sorted(runs_of[k]) for line in open(records_path(d, k[1])) if line.strip()]
    for k, recs in src.items():
        want = {(r["object"], int(r["idx"])) for r in audited_cases(k[0], recs)}
        got = {(r["object"], int(r["idx"])) for r in audit if (r["source"], r["side"]) == k}
        if want != got:
            raise SystemExit(f"{k}: the audit covers {len(got)} cases, the records name {len(want)} ({len(want - got)} missing)")
    idx = index_rows(audit)
    labels = {}
    for k, recs in src.items():
        labels.update(stored_labels(k[0], k[1], audited_cases(k[0], recs)))
    key_of = lambda r: (r["object"], int(r["idx"]))  # noqa: E731
    by1, byo, byp = ({key_of(r): r for r in x} for x in (recs1, src[("oracle", "test")], src[("poseset", "test")]))

    # the gate: the published numbers from the same records
    b_s2 = paired_bootstrap(s2, k0, k4)
    paired_o, _ = co.paired_set([{"oracle": byo[key_of(r)], "stage2": r, "stage1": by1[key_of(r)]} for r in s2])
    views_o = co._views(paired_o, k0, k4)
    b_o = co.paired_outcomes([(v["object"], v["oracle"], v["r0"]) for v in views_o])
    paired_p, _ = cp.paired_set3([{"pose": byp[key_of(r)], "stage2": r, "stage1": by1[key_of(r)], "oracle": byo[key_of(r)]} for r in s2])
    vs = cp.views(paired_p, k0, k4)
    b_p = co.paired_outcomes([(v["object"], v["c10"], v["c3"]) for v in vs])
    b_pu = co.paired_outcomes([(v["object"], v["c10"], v["c3"]) for v in vs if v["upright"]])
    three = ("success", "irr", "give_up")
    gate = {"stage2_r0_r4": differs(b_s2, s2sum["q1"]["boot"]),
            "oracle_r0": differs(b_o, osum["readings"]["1_oracle_vs_r0"]["all"], three),
            "poseset_1p": differs(b_p, psum["readings"]["all"]["1p_pose_term"], three),
            "poseset_1p_upright": differs(b_pu, psum["readings"]["upright"]["1p_pose_term"], three),
            "poseset_ceilings": [f"{v['object']} {v['idx']}" for v in vs
                                 if ceiling_without(v["rows"], set(), False) != v["c10"] or ceiling_without(v["rows"], set(), True) != v["c3"]]}
    me = _revision()
    prov = {"queue": a.queue, "units": len(cfgs), "audit_code_revision": cfgs[0]["code_revision"], "summary_code_revision": me["code_revision"],
            "summary_dirty": me["dirty"], "runs": {f"{k[0]} {k[1]}": sorted(v) for k, v in runs_of.items()},
            "stage2_summary": f"{a.stage2_summary} (sha256 {_sha(a.stage2_summary)})", "oracle_summary": f"{a.oracle_summary} (sha256 {_sha(a.oracle_summary)})",
            "poseset_summary": f"{a.poseset_summary} (sha256 {_sha(a.poseset_summary)})", "locked": f"{a.locked} (sha256 {_sha(a.locked)})",
            "keys": [k0, k4], "bootstrap": f"{BOOT_DRAWS} object draws, seed {BOOT_SEED}, 95 % percentile interval", "flag_mm": FLAG_MM, "primary": PRIMARY}
    if any(gate.values()):
        with open(spath, "w") as f:
            json.dump({"gate": gate, "provenance": prov}, f, indent=1)
        print(f"STOP: the published numbers are not reproduced; wrote {spath}", flush=True)
        raise SystemExit(3)

    # reading 1
    r1 = {f"{k[0]} {k[1]}": reading1([(r["object"], x) for r in audit if (r["source"], r["side"]) == k for x in r["rows"]]) for k in SIDES_OF}
    # reading 2
    kept2 = [r for r in s2 if not r["dropped"]]

    def lab_items(source: str, recs: list[dict[str, Any]], key: str, level: str) -> list[tuple[str | None, str]]:
        return [(y, status(idx.get((source, "test", r["object"], int(r["idx"]), first, att, row)), level))
                for r in recs for att, first, row, y in episode_rows(r, key)]

    def pose_items(current_only: bool, level: str) -> list[tuple[str | None, str]]:
        return [(x.get("label"), status(idx.get(("poseset", "test", v["object"], int(v["idx"]), x["id"])), level))
                for v in vs for x in v["rows"] if "seconds" in x and (x["current"] or not current_only)]

    r2 = {lv: {"R0": label_shares(lab_items("stage2", kept2, k0, lv)), "R4": label_shares(lab_items("stage2", kept2, k4, lv)),
               "oracle": label_shares(lab_items("oracle", [j["oracle"] for j in paired_o], k0, lv)),
               "c10": label_shares(pose_items(False, lv)), "c3": label_shares(pose_items(True, lv))} for lv in FLAG_MM}
    # reading 3
    r3: dict[str, dict[str, Any]] = {}
    for lv in FLAG_MM:
        st2 = [worst([episode_status(idx, "stage2", "test", r, k0, lv), episode_status(idx, "stage2", "test", r, k4, lv)]) for r in kept2]
        sub2 = [r for r, x in zip(kept2, st2) if x == "clear"]
        sto = [worst([episode_status(idx, "oracle", "test", j["oracle"], k0, lv), episode_status(idx, "stage2", "test", j["stage2"], k0, lv)])
               for j in paired_o]
        subo = [v for v, x in zip(views_o, sto) if x == "clear"]
        vw = poseset_without(vs, idx, "test", lv)

        def pose_line(sel: list[dict[str, Any]], orig: list[dict[str, Any]]) -> dict[str, Any]:
            b = co.paired_outcomes([(v["object"], v["c10"], v["c3"]) for v in sel])
            ch = sum(any(w[c][f] != v[c][f] for c in ("c10", "c3") for f in ("success", "irr", "give_up")) for w, v in zip(sel, orig))
            return dict(b, removed_rows=sum(len(w["removed"]) for w in sel), changed=ch,
                        undetermined=sum(not (w["c10"]["determined"] and w["c3"]["determined"]) for w in sel),
                        kept_unaudited=sum(len(w["kept_unaudited"]) for w in sel))

        up = [i for i, v in enumerate(vs) if v["upright"]]
        r3[lv] = {"stage2_r0_r4": dict(paired_bootstrap(sub2, k0, k4), excluded=dict(Counter(x for x in st2 if x != "clear"))) if sub2 else None,
                  "oracle_r0": dict(co.paired_outcomes([(v["object"], v["oracle"], v["r0"]) for v in subo]),
                                    excluded=dict(Counter(x for x in sto if x != "clear"))),
                  "poseset_1p": pose_line(vw, vs), "poseset_1p_upright": pose_line([vw[i] for i in up], [vs[i] for i in up])}
    # reading 4
    flagged = []
    for r in audit:
        for x in r["rows"]:
            k = row_key(r["source"], r["side"], r["object"], r["idx"], x)
            st = status(x, "any")
            if st == "clear":
                continue
            line = dict({f: x[f] for f in ("id", "first", "attempt", "row") if f in x}, source=r["source"], side=r["side"], object=r["object"],
                        idx=int(r["idx"]), label=labels.get(k))
            if st == "not audited":
                line["not_audited"] = x.get("not_audited") or x.get("error")
            else:
                line.update(depth_mm=x["depth_mm"], flags=x["flags"],
                            pairs={p: {"depth_mm": v["depth_mm"], "phases": {ph: q["depth_mm"] for ph, q in v["phases"].items()}} for p, v in x["pairs"].items()})
            flagged.append(line)
    reps = [x for r in audit for x in r["replays"]]
    counts = {"cases": {f"{k[0]} {k[1]}": sum((r["source"], r["side"]) == k for r in audit) for k in SIDES_OF},
              "rows": len(idx), "replays": len(reps), "replays_passed": sum(bool(x["audited"]) for x in reps),
              "replays_differ": sum(bool(x["differs"]) for x in reps), "replays_fresh_state_differs": sum(x["fresh_state_equal"] is False for x in reps),
              "rows_not_audited": sum(status(x) == "not audited" for x in idx.values()), "flagged_list_lines": len(flagged),
              "stage2_kept": len(kept2), "oracle_paired": len(views_o), "poseset_paired": len(vs), "poseset_upright": len(up)}
    published = {"stage2_r0_r4": b_s2, "oracle_r0": b_o, "poseset_1p": b_p, "poseset_1p_upright": b_pu}
    rep = {"counts": counts, "gate": gate, "published": published, "readings": {"1": r1, "2": r2, "3": r3}, "provenance": prov}
    with open(os.path.join(a.out, "flagged_rows.jsonl"), "w") as f:
        for line in flagged:
            f.write(json.dumps(line) + "\n")
    rep["wandb"] = log_run("palm-audit-summary", {"queue": a.queue, "out": a.out},
                           {"counts": counts, "flagged": {k: v["flagged"] for k, v in r1.items()},
                            "sensitivity_success": {k: (r3[PRIMARY][k] or {}).get("success") for k in r3[PRIMARY]}})
    with open(jpath, "w") as f:
        json.dump(rep, f, indent=1)
    with open(os.path.join(a.out, "audit_summary.md"), "w") as f:
        f.write(summary_markdown(rep))
    print(f"wrote {a.out}", flush=True)
    return rep


def main(argv: list[str] | None = None) -> Any:
    ap = argparse.ArgumentParser(prog="palm_audit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--source", required=True, choices=SOURCES)
    r.add_argument("--runs", required=True, help="the source's run dirs, comma-separated")
    r.add_argument("--side", default="test", choices=("test", "validation"))
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--shards", type=int, default=1)
    r.add_argument("--cases", default=None, help="object:idx,... (a smoke): only these cases")
    r.add_argument("--out", required=True)
    m = sub.add_parser("summary")
    m.add_argument("--queue", required=True, help="the queue dir holding the audit units (unit_k)")
    m.add_argument("--stage2-summary", required=True)
    m.add_argument("--oracle-summary", required=True)
    m.add_argument("--poseset-summary", required=True)
    m.add_argument("--locked", required=True)
    m.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return run(a) if a.cmd == "run" else summary(a)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
