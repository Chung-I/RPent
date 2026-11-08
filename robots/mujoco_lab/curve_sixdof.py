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

"""The sixdof rerun's driver (daily-logs design 2026-11-08-sixdof-rerun-design.md).

Gate G1 (design section 2 and 2.2): every simulated row of the GraspGen pilot (1,756) and of the pose-set ablation (its
test and validation sides) runs again through the new executor path (sixdof.execute), each after its own reset at the
case's own seed, and must match its record on every replay-gate field (curve_poseset.SAME: the label, the artifact flag,
the error, the damage and loss fields, the highest rise, the final state's and the trace's hashes). A pilot GraspGen
row runs as the Grasp6 of its grasp (the request step's file, sha-checked): its vertical mapping must give its record's
point and yaw bit for bit, and its hand path today's. The pilot's gate row and a pose-set row run as their Proposal (a
pose-set row's fetched after the reset, as the pose set did). One mismatch fails G1; nothing is tuned to pass it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from robots.mujoco_lab import curve_graspgen as gg
from robots.mujoco_lab import sixdof
from robots.mujoco_lab.curve_poseset import SAME, trace_sha

SOURCES = ("pilot", "poseset")


def _js(x: Any) -> str:
    return json.dumps(x, sort_keys=True)


def outcome(env: Any, ro: Any, at_rest: Any) -> dict[str, Any]:
    """A rollout's outcome fields as the pilot and the pose set write them (simulate_rows): the label, the artifact flag
    (also when the reset was not at rest), the damage and loss fields, the highest rise, the hashes of the final state
    (qpos to 9 digits) and of the trace."""
    d = getattr(getattr(env, "sim", None), "d", None)
    return {"label": ro.label, "artifact": bool(getattr(ro, "artifact", False)) or at_rest is False,
            "damage": getattr(ro, "damage", None), "drop_z": getattr(ro, "drop_z", None), "loss_phase": getattr(ro, "loss_phase", None),
            "max_z": getattr(ro, "max_z", None),
            "qpos_sha": None if d is None else hashlib.sha256(np.round(np.asarray(d.qpos, dtype=float), 9).tobytes()).hexdigest()[:16],
            "trace_sha": trace_sha(getattr(ro, "trace", []))}


def _checked_reset(env: Any, reset: Any, fingerprint: Any, want: str, hull_sha: str | None) -> tuple[Any, str | None]:
    at_rest = reset()
    if fingerprint(env) != want:
        return at_rest, "the reset's fingerprint differs from Stage 1's"
    if hull_sha is not None and gg.hulls_sha(env) != hull_sha:
        return at_rest, "the hulls after the reset differ from the record's"
    return at_rest, None


def run_row(env: Any, reset: Any, make: Any, base: dict[str, Any], fingerprint: Any, want: str,
            hull_sha: str | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """One row through the new path after its own reset: reset(), the fingerprint must be ``want`` and the hulls
    ``hull_sha`` (when given), make(env) gives the pinch (a Proposal or a Grasp6, or a reason it cannot), then
    sixdof.execute at ``base`` (commit_std's force, speeds, destination and release height). Returns (the outcome
    fields with the error and the seconds, None) or (None, the reason the case stops); an exception is the row's error."""
    at_rest, stop = _checked_reset(env, reset, fingerprint, want, hull_sha)
    if stop:
        return None, stop
    t0 = time.time()
    new: dict[str, Any] = {"label": None, "artifact": False, "error": None}
    try:
        p, why = make(env)
        if why:
            return None, why
        ro = sixdof.execute(env, p, base["force"], base["v_lift"], base["v_transport"], base["destination"], base["release_height"],
                            a_transport=base["a_transport"])
        new.update(outcome(env, ro, at_rest))
    except Exception as exc:  # noqa: BLE001  a simulator fault is this row's error
        new["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    new["seconds"] = round(time.time() - t0, 2)
    return new, None


def same_fields(stored: dict[str, Any], new: dict[str, Any]) -> dict[str, list[Any]]:
    """The replay-gate fields (SAME) whose JSON differs between a record's row and its replay: {field: [stored, new]}."""
    return {k: [stored.get(k), new.get(k)] for k in SAME if _js(stored.get(k)) != _js(new.get(k))}


def _paths_equal(a: list[tuple[str, Any, float]], b: list[tuple[str, Any, float]]) -> bool:
    return len(a) == len(b) and all(x[0] == y[0] and np.array_equal(x[1], y[1]) and float(x[2]) == float(y[2]) for x, y in zip(a, b))


def _g1_row(env: Any, reset: Any, r: dict[str, Any], source: str, entry: dict[str, Any] | None, base: dict[str, Any],
            fingerprint: Any, want: str, hull: str | None) -> tuple[dict[str, Any] | None, str | None]:
    from robots.mujoco_lab.curve_poseset import FRACS, TURN

    res: dict[str, Any] = {"id": r["id"], "mapping": None, "hand_path": None}
    if source == "poseset":
        def make(e: Any) -> tuple[Any, str | None]:
            p = {q.s: q for q in e.proposals(FRACS, turn=TURN[r["dir"]])}[r["f"]]
            if [float(x) for x in p.point] != r["point"] or float(p.yaw) != r["yaw"]:
                return None, "the pinch after the reset differs from the record's"
            return p, None
    elif r.get("gate"):
        q0 = gg._proposal(r)

        def make(e: Any) -> tuple[Any, str | None]:
            return q0, None
    else:
        p6 = sixdof.from_graspgen(entry["grasps"][int(r["id"][1:])], s=float(r.get("f", math.nan)), width=float(r["width"]))
        try:
            q = sixdof.as_vertical(p6)
            res["mapping"] = q.point == r["point"] and q.yaw == r["yaw"] and sixdof.approach_deg(p6) == r["approach_deg"]
        except ValueError:
            res["mapping"] = False
        _, stop = _checked_reset(env, reset, fingerprint, want, hull)
        if stop:
            return None, stop
        args = (base["v_lift"], base["v_transport"], base["destination"], base["release_height"])
        res["hand_path"] = res["mapping"] and _paths_equal(sixdof.hand_path(env, p6, *args, a_transport=base["a_transport"]),
                                                           env.hand_path(gg._proposal(r), *args, a_transport=base["a_transport"]))

        def make(e: Any) -> tuple[Any, str | None]:
            return p6, None
    new, stop = run_row(env, reset, make, base, fingerprint, want, hull)
    if stop:
        return None, stop
    diffs = same_fields(r, new)
    res.update(diffs=diffs, seconds=new["seconds"], same=not diffs and res["mapping"] is not False and res["hand_path"] is not False)
    return res, None


def g1_case(prims: Any, rec1: dict[str, Any], seed: int, world: Any, stored: dict[str, Any], source: str,
            entry: dict[str, Any] | None = None) -> dict[str, Any]:
    """G1 for one stored case of ``source`` (pilot or poseset): nothing when it has no simulated row; else the case's
    setup (curve_graspgen._setup: the draw's destination), its reset (Stage 1's fingerprint, the record's hulls for a
    pilot case), the record's base (commit_std_base, base_gate), then each simulated row in the record's order (_g1_row:
    a pilot GraspGen row's hand path after its own reset, then the row after its own reset). A check that fails stops the
    case; an exception outside a row's rollout is the case's error."""
    from robots.mujoco_lab import com_gap
    from robots.mujoco_lab.curve_poseset import _err, base_gate, commit_std_base

    o, idx = stored["object"], int(stored["idx"])
    out: dict[str, Any] = {"object": o, "idx": idx, "side": stored.get("side"), "source": source, "rows": [], "stop": None, "error": None}
    rows = [r for r in stored["rows"] if "seconds" in r]
    if not rows:
        return out
    if (rec1["object"], int(rec1["idx"])) != (o, idx):
        return dict(out, stop="the Stage 1 record is another case's")
    if rec1["fingerprint"] != stored["fingerprint"]:
        return dict(out, stop="Stage 1's fingerprint is not the record's")
    if source == "pilot" and (entry is None or entry.get("error") or entry.get("grasps") is None):
        return dict(out, stop="no grasps for this case")
    env = prims.env
    st = gg._setup(prims, rec1, seed, world)
    if st["stop"]:
        return dict(out, stop=st["stop"])
    reset, want = st["reset_at"](0), rec1["fingerprint"]
    hull = stored.get("hull_sha") if source == "pilot" else None
    try:
        _, stop = _checked_reset(env, reset, com_gap.fingerprint, want, hull)
        if stop:
            return dict(out, stop=stop)
        base = commit_std_base(env, o, st["destination"])
        if _js(base) != _js(stored.get("base")):
            return dict(out, stop=f"the base {base} differs from the record's")
        why = base_gate(rec1, base)
        if why:
            return dict(out, stop=why)
        for r in rows:
            res, stop = _g1_row(env, reset, r, source, entry, base, com_gap.fingerprint, want, hull)
            if stop:
                out["stop"] = f"{r['id']}: {stop}"
                break
            out["rows"].append(res)
    except Exception as exc:  # noqa: BLE001  a case that raises outside a row's rollout is this case's error
        out["error"] = _err(exc)
    return out


def g1_line(c: dict[str, Any]) -> str:
    """A case's log line: counts only, no label."""
    bad = sum(not r["same"] for r in c["rows"])
    return (f"{c['source']} {c['object']} {c['idx']}: {len(c['rows'])} rows replayed, {bad} differ, {c.get('seconds', 0.0):.1f} s"
            + (f", STOP {c['stop']}" if c["stop"] else "") + (f", error {c['error']}" if c["error"] else ""))


def g1_object(job: tuple[str, list[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]]) -> list[dict[str, Any]]:
    """One object's stored cases, in the record's order, on its fold's bundle with the run's sensor and world
    (curve_graspgen.graspgen_object's set-up); the pilot's grasps from the request step's files."""
    from robots.mujoco_lab import calibrate as cal
    from robots.mujoco_lab.prior import fragility_world

    o, items, s = job
    entries = gg.load_grasps(s["grasps"]) if s["source"] == "pilot" else {}
    prims = cal._primitives(s["bundle"], None, None)
    if getattr(prims, "reset_s", None) is not None:
        raise RuntimeError("G1 replays the absorbing attempt only (this bundle restores the object after damage)")
    probes, prims.probe_models = getattr(prims, "probe_models", None), {}
    prims.force_noise = cal.job_force_noise(None, prims.manifest)
    prims.env.force_noise = prims.force_noise
    world = fragility_world(prims.manifest)
    out = []
    try:
        for rec1, stored in items:
            t0 = time.time()
            c = g1_case(prims, rec1, int(s["seed"]), world, stored, s["source"], entries.get((o, int(stored["idx"]))))
            c["seconds"] = round(time.time() - t0, 2)
            out.append(c)
            print(g1_line(c), file=sys.stderr, flush=True)
    finally:
        prims.probe_models = probes
    return out


def g1_metrics(done: list[dict[str, Any]]) -> dict[str, Any]:
    """G1's counts, no label: cases, rows replayed, rows that differ (by field), mapping and hand-path failures, stops,
    case errors, times, the counts by source and side, and whether G1 passed (rows replayed, none differs, no stop, no
    error)."""
    rows = [r for c in done for r in c["rows"]]
    by: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"cases": 0, "rows": 0, "mismatches": 0}))
    for c in done:
        b = by[c["source"]][c["side"]]
        b["cases"] += 1
        b["rows"] += len(c["rows"])
        b["mismatches"] += sum(not r["same"] for r in c["rows"])
    m = {"cases": len(done), "rows": len(rows), "mismatches": sum(not r["same"] for r in rows),
         "by_field": dict(Counter(k for r in rows for k in r.get("diffs") or {})),
         "mapping_failures": sum(r["mapping"] is False for r in rows), "hand_path_failures": sum(r["hand_path"] is False for r in rows),
         "stops": sum(bool(c["stop"]) for c in done), "case_errors": sum(bool(c["error"]) for c in done),
         "row_seconds": round(sum(float(r.get("seconds") or 0.0) for r in rows), 1),
         "case_seconds": round(sum(float(c.get("seconds") or 0.0) for c in done), 1),
         "by_source": {s: dict(v) for s, v in by.items()}}
    m["passed"] = m["rows"] > 0 and m["mismatches"] == 0 and m["stops"] == 0 and m["case_errors"] == 0
    return m


def g1(a: argparse.Namespace) -> str:
    """The G1 command: the stored records of --runs (one source's finished runs; the same Stage 1, seed, bundle and, for
    the pilot, grasps), their cases of --objects and --sides by object in the records' order, each replayed (g1_object);
    g1_records.jsonl (one line per case), run_config.json and g1_metrics.json (counts only; wandb). Exit 3 when G1 fails.
    Returns --out."""
    from robots.mujoco_lab.curve import SIDES, _peak_rss_mb, _sha, load_stage1, records_path
    from robots.mujoco_lab.estimation_check import _revision
    from robots.mujoco_lab.tracking import log_run

    os.makedirs(a.out, exist_ok=True)
    rec_path = os.path.join(a.out, "g1_records.jsonl")
    if os.path.exists(rec_path):
        raise SystemExit(f"{rec_path} exists: pick a new --out")
    runs = a.runs.split(",")
    for d in runs:
        if not os.path.exists(os.path.join(d, "run_metrics.json")):
            raise SystemExit(f"{d}: the run has not finished (no run_metrics.json)")
    cfgs = [json.load(open(os.path.join(d, "run_config.json"))) for d in runs]
    keys = ("stage1", "seed", "bundle") + (("grasps",) if a.source == "pilot" else ())
    for k in keys:
        if len({_js(c.get(k)) for c in cfgs}) > 1:
            raise SystemExit(f"the runs differ in {k}")
    cfg = cfgs[0]
    sides = set(a.sides.split(","))
    objects = set(a.objects.split(",")) if a.objects else None
    paths = [records_path(d, side) for d in runs for side in SIDES if side in sides and os.path.exists(records_path(d, side))]
    stored = [json.loads(line) for p in paths for line in open(p)]
    stored = [r for r in stored if objects is None or r["object"] in objects]
    recs1, _ = load_stage1(cfg["stage1"].split(","))
    by1 = {(r["object"], int(r["idx"])): r for r in recs1}
    by: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for r in stored:
        by[r["object"]].append((by1[(r["object"], int(r["idx"]))], r))
    config = {**vars(a), **_revision(), **{k: cfg.get(k) for k in keys}, "records_sha256": {p: _sha(p) for p in paths},
              "cases": [(r["object"], int(r["idx"])) for r in stored], "rows": sum("seconds" in x for r in stored for x in r["rows"])}
    with open(os.path.join(a.out, "run_config.json"), "w") as f:
        json.dump(config, f, indent=1)
    jobs = [(o, items, {"bundle": os.path.join(cfg["bundle"], "folds", f"fold_{items[0][0]['fold']}"), "seed": cfg["seed"],
                        "source": a.source, "grasps": cfg.get("grasps")}) for o, items in by.items()]
    t0 = time.time()
    with open(rec_path, "w") as f:
        for job in jobs:
            for c in g1_object(job):
                f.write(json.dumps(c) + "\n")
                f.flush()
    done = [json.loads(line) for line in open(rec_path)]
    metrics = dict(g1_metrics(done), of_rows=config["rows"], wall_seconds=round(time.time() - t0, 1), peak_rss_mb=_peak_rss_mb())
    metrics["wandb"] = log_run("sixdof-g1", config, metrics)
    with open(os.path.join(a.out, "g1_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print(f"done: {json.dumps(metrics)}", flush=True)
    if not metrics["passed"]:
        raise SystemExit(3)
    return a.out


def main(argv: list[str] | None = None) -> Any:
    ap = argparse.ArgumentParser(prog="curve_sixdof", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("g1", help="gate G1: the stored rows of one source's runs again through the new executor path")
    g.add_argument("--source", required=True, choices=SOURCES)
    g.add_argument("--runs", required=True, help="comma-separated finished run dirs of that source")
    g.add_argument("--objects", default=None, help="comma-separated objects (default: all)")
    g.add_argument("--sides", default="validation,test")
    g.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"g1": g1}[a.cmd](a)


if __name__ == "__main__":
    main()
