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
    a = ap.parse_args(argv)
    return run(a)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
