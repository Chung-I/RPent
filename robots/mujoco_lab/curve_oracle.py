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


"""The oracle arm (daily-logs researches/property-belief-manipulation/designs/2026-11-08-oracle-arm-design.md, approved by
the user on 2026-11-08; the experiment's README is experiments/2026-11-08-oracle-arm): rule R0 at Stage 2's locked knob
(r 2, A_MAIN, up to 2 attempts) on the Stage 2 cases, with the belief replaced by one particle at the case's true setting
(evpi_props.truth_particle of the case's own draw, weight 1, as evpi_props.reveal "all" builds it).

The first menu is curve.stage1_case's with that belief (the prior's draw still runs, so every stream is as in Stage 1).
The episodes are curve.simulate_case's through OracleLabCase: ``fresh`` prices the same point mass, and the second menu
sets the belief back to it before the reassessment, so the attempt's readout is not applied (its update would refresh a
one-particle belief from the prior). One reset seed per case (the case's own) and Stage 2's skip rules.

    python -P -m robots.mujoco_lab.curve_oracle run --stage1 DIRS --locked locked.json --out DIR [--sides test]
        [--cases o:i,...] [--shard k --shards n] [--workers w]
    python -P -m robots.mujoco_lab.curve_oracle smoke --runs DIRS --stage2 DIRS --out DIR
    python -P -m robots.mujoco_lab.curve_oracle summary --runs DIRS --stage2 DIRS --stage2-summary stage2_summary.json
        --locked locked.json --out DIR

A run's log and metrics hold times, counts and errors, no labels. The smoke reads validation records only. The summary
opens the oracle's test records once, after every run ended EXIT 0; a determinism mismatch (design reading 6) stops it
before any reading (exit 3).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from robots.mujoco_lab import evpi_props as ev
from robots.mujoco_lab.curve import (A_MAIN, BOOT_DRAWS, BOOT_SEED, EPS, GIVE_UP_ROW, R_GRID, RULE_COLOURS, RULE_NAMES, SIDES,
                                     STAGE1_CARE, LabCase, _curves, _end_labels, _err, _kept, _pc, _peak_rss_mb, _row_of, _sha,
                                     _stage2_cases, counted, decide, dkey, episode, load_stage1, load_stage2, no_continuation,
                                     predicted_point, records_path, simulate_case, smoke_summary, stage1_case, stage2_record,
                                     stage2_skip)
from robots.mujoco_lab.evpi_props import SharedMenus
from robots.mujoco_lab.evpi_sim import seeds_of
from robots.mujoco_lab.identify_fit import require_bundle_generator
from rpent.skills.pick_place.protocol import IRREVERSIBLE

DESIGN_KNOB = 2.0                 # design 2: R0's knob is Stage 2's locked value
RULES = ("R0",)
NO_CLASS = {"class": "none", "law": None, "fillable": False}       # stage1_case's entry without a fragility world
FORCE_SPEED = ("force", "v_lift", "v_transport", "a_transport")     # the action fields of a force or speed change
GROUPS = ("R0 gave up, the oracle grasped", "both grasped, different rows", "both grasped, the same row",
          "the oracle gave up, R0 grasped", "both gave up")
MENU_GROUPS = {"a": "no rows (every rule gives up)", "b": "R0 gives up with rows", "c": "R0 grasps"}


# ---------------------------------------------------------------- the belief at the truth

def oracle_belief(b0: Any, truth: dict[str, Any]) -> Any:
    """b0 as one particle at ``truth`` (a copy) with weight 1: evpi_props.reveal "all" (no history, not degenerate; b0's
    flags and visibility kept)."""
    return ev.reveal(b0, "all", copy.deepcopy(truth), None, None, None, None, 0.0)[0]


def truth_belief(b0: Any, pose: dict[str, Any], theta: dict[str, Any], com_end: int, hidden: dict[str, Any],
                 entry: dict[str, Any]) -> Any:
    """The oracle belief of a case's draw: its generator truth (evpi_props.truth_particle, the prior's particle format)."""
    return oracle_belief(b0, ev.truth_particle(pose, theta, com_end, hidden, entry))


def _plain(x: Any) -> Any:
    """x as JSON values (numpy scalars and arrays as numbers and lists)."""
    return json.loads(json.dumps(x, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))


class OracleLabCase(LabCase):
    """curve.LabCase with the belief at the case's truth: the first menu prices it (``first_belief``), and the second menu
    sets the store's belief back to it and reassesses (the attempt's readout and its update are not applied)."""

    def __init__(self, prims: Any, o: str, idx: int, seed: int, world: Any):
        from robots.mujoco_lab.fragility import entry_of

        super().__init__(prims, o, idx, seed, world)
        self.entry = entry_of(o, world[0])[0] if world is not None else dict(NO_CLASS)
        self.point: Any = None
        self.truth: dict[str, Any] | None = None

    def first_belief(self, b0: Any, pose: dict[str, Any]) -> Any:
        self.truth = ev.truth_particle(pose, self.theta, self.com_end, self.hidden, self.entry)
        self.point = oracle_belief(b0, self.truth)
        return self.point

    def second_menu(self, store: Any, d: dict[str, Any]) -> dict[str, Any]:
        from rpent.skills.pick_place import tools

        oid = self.prims.env.locate("object")["object_id"]
        tools._beliefs(store)[oid]["belief"] = oracle_belief(self.point, self.truth)
        a2 = tools._reassess("object", self.destination, STAGE1_CARE, self.prims, store)
        return super().second_menu(store, dict(d, assessment=a2))


# ---------------------------------------------------------------- the decisions and the case

def curve_decisions(rows: list[dict[str, Any]]) -> dict[str, str]:
    """R0's first decision at every knob of R_GRID (A_MAIN), keyed by dkey: the model-only curve (reading 7)."""
    return {dkey("R0", A_MAIN, r): decide("R0", rows, r, A_MAIN) for r in R_GRID}


def oracle_second(knob: float) -> Any:
    """simulate_case's second decisions: R0 at ``knob`` only, as curve.second_decisions decides it."""
    def second(rows: list[dict[str, Any]], rules: tuple[str, ...]) -> dict[str, str]:
        return {dkey("R0", A_MAIN, knob): decide("R0", rows, knob, A_MAIN, no_continuation)}

    return second


def locked_knob(path: str) -> float:
    """R0's knob in Stage 2's lock file; it must be the design's (r 2)."""
    k = json.load(open(path))["lock"]["R0"]["knob"]
    if k != DESIGN_KNOB:
        raise SystemExit(f"{path}: R0's locked knob is {k}, the design's is {DESIGN_KNOB:g}")
    return float(k)


def oracle_case(prims: Any, rec1: dict[str, Any], s: dict[str, Any], world: Any, knob: float) -> dict[str, Any]:
    """One case: Stage 2's skip rules on the Stage 1 record, the first menu at the truth (stage1_case with truth_belief;
    its reset must have Stage 1's fingerprint, else the case stops the run), R0's decision at ``knob`` and, if it grasps,
    the episode at the case's own reset seed (simulate_case through OracleLabCase, R0's second decision only). The record
    is curve.stage2_record's on the oracle's first menu, plus the oracle's rows, R0's decision at every knob, the truth
    particle and the simulator's truth at the reset (env.truth)."""
    o, idx, key = rec1["object"], int(rec1["idx"]), dkey("R0", A_MAIN, knob)
    extra: dict[str, Any] = {"oracle": True, "knob": knob, "curve_decisions": {}, "rows": [], "truth": None, "sim_truth": None}
    if stage2_skip(rec1):
        return dict(stage2_record(rec1, {}, RULES, None, 1), row_stats={}, **extra)
    got: dict[str, Any] = {}

    def belief_fn(b0: Any, pose: dict[str, Any], theta: dict[str, Any], com_end: int, hidden: dict[str, Any],
                  entry: dict[str, Any]) -> Any:
        got["truth"] = ev.truth_particle(pose, theta, com_end, hidden, entry)
        got["sim_truth"] = _plain(prims.env.truth())
        return oracle_belief(b0, got["truth"])

    with SharedMenus(prims) as shared:
        o1 = stage1_case(prims, o, idx, {"seed": int(s["seed"]), "fold": rec1["fold"], "side": rec1["side"]}, world, shared,
                         belief_fn=belief_fn)
    extra.update(truth=got.get("truth"), sim_truth=got.get("sim_truth"))
    if o1.get("fingerprint") != rec1.get("fingerprint"):
        res = {"sims": [], "stop": "the oracle's reset fingerprint differs from Stage 1's"}
        return dict(stage2_record(o1, {}, RULES, res, 1), **extra)
    if o1.get("status") == "error" or "error" in o1 or o1.get("artifact"):
        why = "oracle reset artifact" if o1.get("artifact") and "error" not in o1 else "oracle first menu error"
        return dict(stage2_record(o1, {}, RULES, None, 1), skipped=why, dropped=True, error=o1.get("error"), **extra)
    rows = o1["rows"]
    dec = {key: decide("R0", rows, knob, A_MAIN)}
    res: dict[str, Any] = {"sims": [], "stop": None}
    if dec[key] != GIVE_UP_ROW:
        try:
            lab = OracleLabCase(prims, o, idx, int(s["seed"]), world)
            res = simulate_case(lab, o1, dec, RULES, seeds_of(int(s["seed"]), o, idx)[:1], second=oracle_second(knob))
        except Exception as exc:  # noqa: BLE001  a case that raises is dropped (curve design 5.5)
            res = {"sims": [], "stop": None, "error": _err(exc)}
    return dict(stage2_record(o1, dec, RULES, res, 1), curve_decisions=curve_decisions(rows), rows=rows,
                **{k: v for k, v in extra.items() if k not in ("curve_decisions", "rows")})


def oracle_object(job: tuple[str, list[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    """One object's oracle cases on its fold's bundle (curve.stage2_object's sensor and world, no probe models). The log
    line per case holds no label (no early look). A gate that fails ends the object with ``stop``."""
    from robots.mujoco_lab import calibrate as cal
    from robots.mujoco_lab.prior import fragility_world

    o, recs1, s = job
    prims = cal._primitives(s["bundle"], None, None)
    if getattr(prims, "reset_s", None) is not None:
        raise RuntimeError("the curve scores the absorbing episode only (this bundle restores the object after damage)")
    probes, prims.probe_models = getattr(prims, "probe_models", None), {}
    prims.force_noise = cal.job_force_noise(None, prims.manifest)
    prims.env.force_noise = prims.force_noise
    world = fragility_world(prims.manifest)
    out: dict[str, Any] = {"records": [], "stop": None}
    try:
        for rec1 in recs1:
            t0 = time.time()
            rec = oracle_case(prims, rec1, s, world, float(s["knob"]))
            rec["seconds"] = round(time.time() - t0, 2)
            out["records"].append(rec)
            bad = sum(bool(x.get("error")) for x in rec["sims"]) + sum(bool(y.get("error")) for x in rec["sims"] for y in x["second"].values())
            print(f"{o} {rec1['idx']}: done" + (f", {bad} simulation errors" if bad else "")
                  + (f", skipped ({rec['skipped']})" if rec["skipped"] else "") + (f", error {rec['error']}" if rec.get("error") else "")
                  + (f", STOP {rec['stop']}" if rec["stop"] else ""), file=sys.stderr, flush=True)
            if rec["stop"]:
                out["stop"] = f"{o} {rec1['idx']}: {rec['stop']}"
                break
    finally:
        prims.probe_models = probes
    return out


def run(a: argparse.Namespace) -> str:
    """The oracle run: records_validation.jsonl and records_test.jsonl (one oracle_case record per case, by object as each
    ends), run_config.json and run_metrics.json (times, counts and errors only; wandb when on). A gate that fails stops
    the run (exit 3). Returns --out."""
    import multiprocessing as mp

    from robots.mujoco_lab.estimation_check import _revision
    from robots.mujoco_lab.tracking import log_run

    os.makedirs(a.out, exist_ok=True)
    if any(os.path.exists(records_path(a.out, side)) for side in SIDES):
        raise SystemExit(f"{a.out}: a records file exists: pick a new --out")
    knob = locked_knob(a.locked)
    dirs = a.stage1.split(",")
    recs1, cfg1 = load_stage1(dirs)
    generator = require_bundle_generator(cfg1["bundle"])
    if generator != "v11":
        raise SystemExit(f"the bundle's generator is {generator}: evpi_props.truth_particle is built for v11")
    recs = _stage2_cases(a, recs1)
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        by[r["object"]].append(r)
    config = {**vars(a), **_revision(), "generator": generator, "bundle": cfg1["bundle"], "seed": cfg1["seed"],
              "stage1_code_revision": cfg1["code_revision"],
              "stage1_records_sha256": {d: _sha(os.path.join(d, "records.jsonl")) for d in dirs},
              "split_sha256": cfg1["split_sha256"], "locked_sha256": _sha(a.locked), "knob": knob, "key": dkey("R0", A_MAIN, knob),
              "rules": RULES, "n_seeds": 1, "cases": [(r["object"], int(r["idx"])) for r in recs], "a_main": A_MAIN,
              "stage1_care": STAGE1_CARE, "r_grid": R_GRID}
    with open(os.path.join(a.out, "run_config.json"), "w") as f:
        json.dump(config, f, indent=1)
    jobs = [(o, items, {"bundle": os.path.join(cfg1["bundle"], "folds", f"fold_{items[0]['fold']}"), "seed": cfg1["seed"],
                        "knob": knob}) for o, items in by.items()]
    t0, stop = time.time(), None
    outs = {side: open(records_path(a.out, side), "w") for side in SIDES}
    try:
        pool = None if a.workers == 1 else mp.get_context("spawn").Pool(a.workers)
        results = map(oracle_object, jobs) if pool is None else pool.imap_unordered(oracle_object, jobs)
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
    metrics = {"cases": len(done), "of_cases": len(recs), "stop": stop, "skipped": sum(bool(r["skipped"]) for r in done),
               "dropped": sum(bool(r["dropped"]) for r in done), "case_errors": sum(bool(r["error"]) for r in done),
               "simulation_errors": sum(bool(x.get("error")) for r in done for x in r["sims"])
               + sum(bool(y.get("error")) for r in done for x in r["sims"] for y in x["second"].values()),
               "wall_seconds": round(time.time() - t0, 1), "case_seconds": round(sum(r["seconds"] for r in done), 1),
               "case_seconds_max": max((r["seconds"] for r in done), default=0.0), "peak_rss_mb": _peak_rss_mb()}
    metrics["wandb"] = log_run("curve-oracle", config, metrics)
    with open(os.path.join(a.out, "run_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print(f"done: {json.dumps(metrics)}", flush=True)
    if stop:
        raise SystemExit(3)
    return a.out


# ---------------------------------------------------------------- the readings (design 3)

def _rates(eps: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(eps)
    out: dict[str, Any] = {"n": n}
    for k in ("success", "irr", "give_up"):
        out[f"n_{k}"] = sum(bool(e[k]) for e in eps)
        out[k] = out[f"n_{k}"] / n if n else None
    out["n_second"] = sum(e["attempts"] == 2 for e in eps)
    return out


def paired_outcomes(pairs: list[tuple[str, dict[str, Any], dict[str, Any]]], draws: int = BOOT_DRAWS,
                    seed: int = BOOT_SEED) -> dict[str, Any]:
    """a - b in success, irreversible damage and give-up over paired episodes [(object, a, b)] with curve.paired_bootstrap's
    95 % object bootstrap (the same object order and draws), and each arm's rates (rate_a, rate_b)."""
    names = ("success", "irr", "give_up")
    objs = sorted({o for o, _, _ in pairs})
    out: dict[str, Any] = {"objects": len(objs), "cases": len(pairs), "draws": int(draws), "seed": int(seed),
                           "rate_a": _rates([x for _, x, _ in pairs]), "rate_b": _rates([y for _, _, y in pairs])}
    if not pairs:
        return dict(out, **{k: None for k in names})
    at = {o: i for i, o in enumerate(objs)}
    d, cnt = {k: np.zeros(len(objs)) for k in names}, np.zeros(len(objs))
    for o, x, y in pairs:
        for k in names:
            d[k][at[o]] += float(x[k]) - float(y[k])
        cnt[at[o]] += 1.0
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, len(objs), size=(int(draws), len(objs)))
    M = np.zeros((int(draws), len(objs)))
    np.add.at(M, (np.arange(int(draws))[:, None], idx), 1.0)
    n = M @ cnt
    for k in names:
        lo, hi = np.percentile((M @ d[k]) / n, [2.5, 97.5])
        out[k] = {"diff": float(d[k].sum() / cnt.sum()), "lo": float(lo), "hi": float(hi)}
    return out


def oracle_reading(b: dict[str, Any], eps: float = EPS) -> str:
    """Design reading 3 on the success difference oracle - R0. Within eps (the interval includes 0 or sits below +eps):
    the belief is not what limits success. Above eps with the interval above 0: the belief is a limit. Else (the interval
    above 0, its upper end at eps or more, the difference at eps or less): no reading by the stated rule."""
    s = b["success"]
    if s["lo"] <= 0.0 <= s["hi"] or s["hi"] < eps:
        return "the belief is not what limits success"
    if s["diff"] > eps and s["lo"] > 0.0:
        return "the belief is a limit"
    return "no reading by the stated rule"


def first_decision_groups(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Design reading 4: the cases ({"oracle", "r0": episode, "same": the same first action}) by their first decisions,
    with each arm's rates and the oracle's first labels."""
    by: dict[str, list[dict[str, Any]]] = {g: [] for g in GROUPS}
    for c in cases:
        go, g0 = c["oracle"]["attempts"] > 0, c["r0"]["attempts"] > 0
        g = (GROUPS[2] if c["same"] else GROUPS[1]) if go and g0 else (GROUPS[0] if go else GROUPS[3] if g0 else GROUPS[4])
        by[g].append(c)
    return {g: {"n": len(v), "oracle": _rates([c["oracle"] for c in v]), "r0": _rates([c["r0"] for c in v]),
                "oracle_first_labels": dict(Counter(c["oracle"]["labels"][0] for c in v if c["oracle"]["labels"]))}
            for g, v in by.items()}


def _action(key: str) -> dict[str, Any]:
    return json.loads(key.split("|", 1)[1]) if "|" in key else {}


def row_change(ko: str | None, kr: str) -> str:
    """Design reading 8 (c): the oracle's first action key against R0's. Same row (the same action), pose change (another
    pinch s), force or speed change (FORCE_SPEED), other change (the destination or the release height), or the oracle
    gives up (ko None)."""
    if ko is None:
        return "the oracle gives up"
    a, b = _action(ko), _action(kr)
    if a == b:
        return "same row"
    if a.get("s") != b.get("s"):
        return "pose change"
    if any(a.get(k) != b.get(k) for k in FORCE_SPEED):
        return "force or speed change"
    return "other change"


def menu_group(dec: dict[str, str], k0: str, k4: str) -> str:
    """Design reading 8's group of a Stage 2 case: a (R4 gives up: no rows), b (R0 gives up with rows), c (R0 grasps)."""
    if dec[k4] == GIVE_UP_ROW:
        if dec[k0] != GIVE_UP_ROW:
            raise ValueError(f"R0 grasps {dec[k0]} where R4 gives up")
        return "a"
    return "b" if dec[k0] == GIVE_UP_ROW else "c"


def reliability(p: list[float], y: list[Any]) -> dict[str, Any]:
    """Predicted probabilities against 0/1 labels: ten bins of near-equal size by sorted prediction (fewer with fewer than
    ten), each with its mean prediction and observed rate, the Brier score and the sums."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    if len(p) == 0:
        return {"n": 0, "bins": [], "brier": None, "predicted_sum": 0.0, "observed_sum": 0}
    order = np.argsort(p, kind="stable")
    bins = [{"n": len(i), "predicted": float(p[i].mean()), "observed": float(y[i].mean()), "lo": float(p[i].min()),
             "hi": float(p[i].max())} for i in np.array_split(order, min(10, len(p)))]
    return {"n": len(p), "bins": bins, "brier": float(np.mean((p - y) ** 2)), "predicted_sum": float(p.sum()),
            "observed_sum": int(y.sum())}


def attempts_at_truth(recs: list[dict[str, Any]], key: str) -> list[tuple[float, float, str, int]]:
    """(P(placed), P(irr), label, attempt) of every attempt of the episodes at ``key`` over the records that are not
    dropped: the first attempt's row from the first menu (row_stats), the second's from the second menu."""
    out = []
    for r in _kept(recs):
        e = episode(r["decisions"], r["sims"], key)
        for j, (row, label) in enumerate(zip(e["rows"], e["labels"])):
            if j == 0:
                st = r["row_stats"][row]
            else:
                s = next(x for x in r["sims"] if x["seed_k"] == 0 and x["first"] == e["rows"][0])
                st = _row_of(s["menu2"]["rows"], row)
            out.append((float(st["p_placed"]), float(st["p_irr"]), label, j + 1))
    return out


def _akey(rows: list[dict[str, Any]], rid: str) -> str:
    """A row's action key on a menu (its id when the menu lacks it)."""
    return next((x["action"] for x in rows if x["id"] == rid), rid)


def _first_sim(r: dict[str, Any], first: str) -> dict[str, Any] | None:
    return next((x for x in r.get("sims") or [] if x["seed_k"] == 0 and x["first"] == first), None)


def determinism(cases: list[dict[str, Any]], k0: str, k4: str) -> dict[str, Any]:
    """Design reading 6 over joined cases ({"oracle", "stage2", "stage1"} records): where the oracle's first row has the
    action key of R0's or R4's Stage 2 first row, its first label and artifact flag must be Stage 2's (same reset seed)."""
    compared, missing, bad = 0, 0, []
    for c in cases:
        o, s2, s1 = c["oracle"], c["stage2"], c["stage1"]
        first = (o.get("decisions") or {}).get(k0)
        if o.get("skipped") or first in (None, GIVE_UP_ROW):
            continue
        ka = _akey(o["rows"], first)
        match = next((d for d in (s2["decisions"].get(k0), s2["decisions"].get(k4))
                      if d not in (None, GIVE_UP_ROW) and _akey(s1["rows"], d) == ka), None)
        if match is None:
            continue
        so, ss = _first_sim(o, first), _first_sim(s2, match)
        if so is None or ss is None:
            missing += 1
            continue
        compared += 1
        x, y = [so["label"], bool(so["artifact"])], [ss["label"], bool(ss["artifact"])]
        if x != y:
            bad.append({"object": o["object"], "idx": int(o["idx"]), "row": first, "oracle": x, "stage2": y})
    return {"compared": compared, "missing": missing, "mismatches": bad}


def model_curves(orecs: list[dict[str, Any]], recs1: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Design reading 7 (model only): curve.predicted_point of R0 at every knob of R_GRID, on the oracle's first menus
    (records not skipped) and on Stage 1's prior menus of the same cases."""
    keep = [r for r in orecs if not r.get("skipped")]
    have = {(r["object"], int(r["idx"])) for r in keep}
    mo = [{"rows": r["rows"], "decisions": r["curve_decisions"]} for r in keep]
    mp = [{"rows": r["rows"], "decisions": curve_decisions(r["rows"])} for r in counted(recs1, None)
          if (r["object"], int(r["idx"])) in have]
    return {name: [dict(knob=k, **predicted_point(m, dkey("R0", A_MAIN, k))) for k in R_GRID]
            for name, m in (("oracle", mo), ("r0_stage1", mp))}


def truth_gap(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """The simulator's truth at the reset (env.truth) against the oracle's particle: the relative mass gap and the CoM gap
    in mm (median, 95th percentile, max) over the records that hold both."""
    m, c = [], []
    for r in recs:
        t, s = r.get("truth"), r.get("sim_truth")
        if t and s:
            m.append(abs(float(s["mass"]) - float(t["mass"])) / float(t["mass"]))
            c.append(1000.0 * float(np.linalg.norm(np.asarray(s["com_obj"], dtype=float) - np.asarray(t["com_obj"], dtype=float))))

    def q(x: list[float]) -> dict[str, float] | None:
        return {"median": float(np.median(x)), "p95": float(np.percentile(x, 95)), "max": float(np.max(x))} if x else None

    return {"records": len(m), "mass_rel": q(m), "com_mm": q(c)}


def _views(joined: list[dict[str, Any]], k0: str, k4: str) -> list[dict[str, Any]]:
    """Per joined case: its object, thin flag, reading-8 group, the three episodes (oracle, R0, R4) and the row change."""
    out = []
    for j in joined:
        o, s2, s1 = j["oracle"], j["stage2"], j["stage1"]
        fo, f0 = o["decisions"][k0], s2["decisions"][k0]
        ko = None if fo == GIVE_UP_ROW else _akey(o["rows"], fo)
        change = None if f0 == GIVE_UP_ROW else row_change(ko, _akey(s1["rows"], f0))
        out.append({"object": o["object"], "thin": bool(s2["thin"]), "group": menu_group(s2["decisions"], k0, k4),
                    "oracle": episode(o["decisions"], o["sims"], k0), "r0": episode(s2["decisions"], s2["sims"], k0),
                    "r4": episode(s2["decisions"], s2["sims"], k4), "change": change, "same": change == "same row", "rec": o})
    return out


def _calibration(recs: list[dict[str, Any]], key: str) -> dict[str, Any]:
    att = attempts_at_truth(recs, key)
    return {"attempts": len(att), "first_attempts": sum(x[3] == 1 for x in att),
            "placed": reliability([x[0] for x in att], [x[2] == "placed_stable" for x in att]),
            "irr": reliability([x[1] for x in att], [x[2] in IRREVERSIBLE for x in att])}


def _block(views: list[dict[str, Any]], k0: str) -> dict[str, Any]:
    b0 = paired_outcomes([(v["object"], v["oracle"], v["r0"]) for v in views])
    b4 = paired_outcomes([(v["object"], v["oracle"], v["r4"]) for v in views])
    return {"oracle_vs_r0": b0, "oracle_vs_r4": b4, "reading": oracle_reading(b0) if views else None,
            "groups": first_decision_groups(views), "calibration": _calibration([v["rec"] for v in views], k0)}


def _menu_block(views: list[dict[str, Any]], k0: str) -> dict[str, Any]:
    ch = Counter(v["change"] for v in views if v["change"] is not None)
    return {"n": len(views), "n_no_thin": sum(not v["thin"] for v in views),
            "oracle_rows": sum(v["oracle"]["attempts"] > 0 for v in views),
            "all": _block(views, k0), "no_thin": _block([v for v in views if not v["thin"]], k0), "row_change": dict(ch),
            "row_change_outcomes": {c: {"oracle": _rates([v["oracle"] for v in views if v["change"] == c]),
                                        "r0": _rates([v["r0"] for v in views if v["change"] == c])} for c in ch}}


# ---------------------------------------------------------------- the commands

def _by_case(recs: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(r["object"], int(r["idx"])): r for r in recs}


def _first_label(r: dict[str, Any], key: str) -> list[Any]:
    first = (r.get("decisions") or {}).get(key)
    s = None if first in (None, GIVE_UP_ROW) else _first_sim(r, first)
    return [first, None if s is None else s["label"]]


def smoke(a: argparse.Namespace) -> dict[str, Any]:
    """The smoke report on validation runs only (refuses a run that holds test records): curve.smoke_summary's counts,
    a table per case (the oracle's first row and labels, Stage 2's R0 and R4 first rows and labels), the determinism
    check against Stage 2's validation records, and the full run's projection: the mean seconds of the smoke cases that
    simulated times the Stage 1 test cases that are not skipped (every one counted as simulating)."""
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
    key, key4 = cfg["key"], dkey("R4", A_MAIN, None)
    s2, _ = load_stage2(a.stage2.split(","), ("validation",))
    recs1, _ = load_stage1(list(cfg["stage1_records_sha256"]))
    by1, by2 = _by_case(recs1), _by_case(s2)
    joined = [{"oracle": r, "stage2": by2[k], "stage1": by1[k]} for r in recs for k in [(r["object"], int(r["idx"]))]]
    table = []
    for j in joined:
        r = j["oracle"]
        s = None if r["skipped"] else _first_sim(r, r["decisions"].get(key))
        labels = [] if s is None else [s["label"]] + [x["label"] for x in s["second"].values()]
        table.append({"object": r["object"], "idx": int(r["idx"]), "status": r["status"], "ess": r["ess"],
                      "first": None if r["skipped"] else r["decisions"].get(key), "labels": labels, "seconds": r["seconds"],
                      "skipped": r["skipped"], "dropped": r["dropped"], "stage2_r0": _first_label(j["stage2"], key),
                      "stage2_r4": _first_label(j["stage2"], key4), "truth_gap": truth_gap([r])})
    secs = [float(r["seconds"]) for r in recs if r["sims"]] or [float(r["seconds"]) for r in recs if not r["skipped"]]
    run_cases = sum(r["side"] == "test" and stage2_skip(r) is None for r in recs1)
    sm = dict(smoke_summary(recs), cases=len(recs), dropped=sum(bool(r["dropped"]) for r in recs),
              case_errors=sum(bool(r["error"]) for r in recs), stops=[r["stop"] for r in recs if r["stop"]], table=table,
              determinism=determinism(joined, key, key4), run_cases=run_cases,
              seconds_per_case=float(np.mean(secs)) if secs else None, seconds_per_case_max=max(secs) if secs else None)
    sm["projection_cpu_h"] = sm["seconds_per_case"] * run_cases / 3600.0 if secs else None
    sm["projection_cpu_h_at_max"] = sm["seconds_per_case_max"] * run_cases / 3600.0 if secs else None
    sm["wandb"] = log_run("curve-oracle-smoke", {"runs": a.runs, "stage2": a.stage2},
                          {k: v for k, v in sm.items() if k not in ("table",)})
    with open(jpath, "w") as f:
        json.dump(sm, f, indent=1)
    with open(os.path.join(a.out, "smoke_summary.md"), "w") as f:
        f.write(smoke_markdown(sm))
    print(json.dumps({k: v for k, v in sm.items() if k != "table"}, indent=1), flush=True)
    return sm


def smoke_markdown(sm: dict[str, Any]) -> str:
    d = sm["determinism"]
    L = ["# Oracle arm: the smoke test (validation cases only)", "",
         f"- Cases {sm['cases']}: dropped {sm['dropped']}, case errors {sm['case_errors']}, simulation errors {sm['errors']}, "
         f"artifacts {sm['artifacts']}, stops {len(sm['stops'])}.",
         f"- Determinism (reading 6): {d['compared']} cases compared, {len(d['mismatches'])} mismatches, {d['missing']} without a simulation.",
         f"- Seconds per case (cases that simulated): mean {_num(sm['seconds_per_case'], 1)}, max {_num(sm['seconds_per_case_max'], 1)}.",
         f"- Projection for {sm['run_cases']} test cases (every one counted as simulating): {_num(sm['projection_cpu_h'], 2)} CPU-h "
         f"(at the slowest smoke case: {_num(sm['projection_cpu_h_at_max'], 2)} CPU-h).", "",
         "| case | status | ESS | oracle first row | oracle labels | seconds | Stage 2 R0 first row, label | Stage 2 R4 first row, label |",
         "|---|---|---|---|---|---|---|---|"]
    for t in sm["table"]:
        L.append(f"| {t['object']} {t['idx']} | {t['status']} | {t['ess']} | {t['first']} | {', '.join(map(str, t['labels'])) or '–'} | "
                 f"{t['seconds']} | {t['stage2_r0'][0]}, {t['stage2_r0'][1]} | {t['stage2_r4'][0]}, {t['stage2_r4'][1]} |")
    return "\n".join(L) + "\n"


def _num(x: float | None, nd: int) -> str:
    return "–" if x is None else f"{x:.{nd}f}"


def _gap(q: dict[str, float] | None, scale: float, nd: int) -> str:
    return "–" if q is None else ", ".join(f"{k} {scale * q[k]:.{nd}f}" for k in ("median", "p95", "max"))


def _finished(d: str) -> dict[str, Any]:
    ex = os.path.join(d, "run.exit")
    if not os.path.exists(ex) or open(ex).read().split() != ["EXIT", "0"]:
        raise SystemExit(f"{d}: the run did not end with EXIT 0 (run.exit)")
    if json.load(open(os.path.join(d, "run_metrics.json"))).get("stop"):
        raise SystemExit(f"{d}: the run stopped on a gate")
    return json.load(open(os.path.join(d, "run_config.json")))


def summary(a: argparse.Namespace) -> dict[str, Any]:
    """The readings (design 3) from finished oracle runs and Stage 2's test records: oracle_summary.{json,md} and
    oracle_points.png in --out (refuses an existing summary), wandb. Refuses a run without EXIT 0, inputs that changed
    since the runs, and oracle records that do not cover Stage 2's test cases. A determinism mismatch writes
    oracle_summary_stop.json and exits 3 before any reading."""
    from robots.mujoco_lab.tracking import log_run

    os.makedirs(a.out, exist_ok=True)
    jpath, spath = os.path.join(a.out, "oracle_summary.json"), os.path.join(a.out, "oracle_summary_stop.json")
    if os.path.exists(jpath) or os.path.exists(spath):
        raise SystemExit(f"{a.out}: a summary exists: pick a new --out")
    knob = locked_knob(a.locked)
    lk = json.load(open(a.locked))["lock"]
    k0, k4 = lk["R0"]["key"], lk["R4"]["key"]
    dirs = a.runs.split(",")
    cfgs = [_finished(d) for d in dirs]
    for k in ("knob", "key", "bundle", "seed", "locked_sha256", "stage1_records_sha256", "code_revision"):
        if len({json.dumps(c.get(k), sort_keys=True) for c in cfgs}) > 1:
            raise SystemExit(f"the oracle runs differ in {k}")
    cfg = cfgs[0]
    if cfg["knob"] != knob or cfg["key"] != k0:
        raise SystemExit(f"the runs' key {cfg['key']} is not the lock's {k0}")
    if cfg["locked_sha256"] != _sha(a.locked):
        raise SystemExit("the lock file changed since the oracle runs")
    s2dirs = a.stage2.split(",")
    prov2 = json.load(open(a.stage2_summary))["provenance"]["records_sha256"]
    for d in s2dirs:
        if _sha(records_path(d, "test")) != prov2.get(f"{d} test"):
            raise SystemExit(f"{d}: the Stage 2 test records are not the ones the Stage 2 summary read")
    s2, s2cfg = load_stage2(s2dirs, ("test",))
    if s2cfg["stage1_records_sha256"] != cfg["stage1_records_sha256"]:
        raise SystemExit("the oracle runs and Stage 2 read different Stage 1 records")
    for d, h in cfg["stage1_records_sha256"].items():
        if _sha(os.path.join(d, "records.jsonl")) != h:
            raise SystemExit(f"{d}: the Stage 1 records changed since the oracle runs")
    recs1, _ = load_stage1(list(cfg["stage1_records_sha256"]))
    orecs = [json.loads(line) for d in dirs for line in open(records_path(d, "test"))]      # the oracle's test records, once
    if any(r["side"] != "test" or r.get("stop") for r in orecs):
        raise SystemExit("an oracle test record holds a stop or another side")
    if len(_by_case(orecs)) != len(orecs):
        raise SystemExit("an oracle case appears twice")
    byo, by1 = _by_case(orecs), _by_case(recs1)
    have2 = set(_by_case(s2))
    missing, extra = have2 - set(byo), set(byo) - have2
    if missing or extra:
        raise SystemExit(f"{len(missing)} Stage 2 test cases are missing from the oracle runs, {len(extra)} are not Stage 2's")
    joined = [{"oracle": byo[k], "stage2": r, "stage1": by1[k]} for r in s2 for k in [(r["object"], int(r["idx"]))]]
    det = determinism(joined, k0, k4)
    prov = {"runs": a.runs, "oracle_records_sha256": {d: _sha(records_path(d, "test")) for d in dirs},
            "code_revision": cfg["code_revision"], "stage2": a.stage2, "stage2_summary": f"{a.stage2_summary} (sha256 {_sha(a.stage2_summary)})",
            "locked": f"{a.locked} (sha256 {_sha(a.locked)})", "stage1": sorted(cfg["stage1_records_sha256"]),
            "bootstrap": f"{BOOT_DRAWS} object draws, seed {BOOT_SEED}, 95 % percentile interval", "eps": EPS, "keys": [k0, k4]}
    if det["mismatches"]:
        with open(spath, "w") as f:
            json.dump({"determinism": det, "provenance": prov}, f, indent=1)
        print(f"STOP: {len(det['mismatches'])} determinism mismatches (reading 6); wrote {spath}", flush=True)
        raise SystemExit(3)
    paired = [j for j in joined if not j["stage2"]["dropped"] and not j["oracle"]["dropped"]]
    views = _views(paired, k0, k4)
    nt = [v for v in views if not v["thin"]]
    ba, bn = _block(views, k0), _block(nt, k0)
    counts = {"stage2_test_cases": len(s2), "stage2_dropped": sum(bool(r["dropped"]) for r in s2),
              "oracle_dropped": sum(bool(j["oracle"]["dropped"]) for j in joined),
              "oracle_only_dropped": sum(bool(j["oracle"]["dropped"]) and not j["stage2"]["dropped"] for j in joined),
              "oracle_skips": dict(Counter(j["oracle"]["skipped"] for j in joined if j["oracle"]["skipped"])),
              "paired": len(views), "objects": len({v["object"] for v in views}), "thin_cases": len(views) - len(nt),
              "groups": dict(Counter(v["group"] for v in views))}
    readings = {"1_oracle_vs_r0": {"all": ba["oracle_vs_r0"], "no_thin": bn["oracle_vs_r0"]},
                "2_oracle_vs_r4": {"all": ba["oracle_vs_r4"], "no_thin": bn["oracle_vs_r4"]},
                "3_reading": ba["reading"], "3_reading_no_thin": bn["reading"],
                "4_groups": {"all": ba["groups"], "no_thin": bn["groups"]},
                "5_calibration": {"all": ba["calibration"], "no_thin": bn["calibration"]},
                "6_determinism": det,
                "7_model_curves": model_curves(orecs, [r for r in recs1 if r["side"] == "test"]),
                "8_menu_limit": {g: _menu_block([v for v in views if v["group"] == g], k0) for g in MENU_GROUPS}}
    rules = tuple(s2cfg["rules"])
    figure = {"curves": _curves([j["stage2"] for j in paired], rules), "locked": {r: lk[r]["knob"] for r in rules},
              "oracle": _rates([v["oracle"] for v in views])}
    rep = {"knob": knob, "counts": counts, "readings": readings, "truth_gap": truth_gap([v["rec"] for v in views]),
           "figure": figure, "provenance": prov}
    r1 = readings["1_oracle_vs_r0"]["all"]
    rep["wandb"] = log_run("curve-oracle-summary", {"runs": a.runs, "stage2": a.stage2, "knob": knob},
                           {"counts": counts, "reading": readings["3_reading"], "reading_no_thin": readings["3_reading_no_thin"],
                            "oracle_vs_r0": {k: r1[k] for k in ("success", "irr", "give_up")}, "determinism": det})
    with open(jpath, "w") as f:
        json.dump(rep, f, indent=1)
    with open(os.path.join(a.out, "oracle_summary.md"), "w") as f:
        f.write(summary_markdown(rep))
    oracle_figure(rep, os.path.join(a.out, "oracle_points.png"))
    print(f"wrote {a.out}", flush=True)
    return rep


def _ci(b: dict[str, Any], name: str) -> str:
    x = b.get(name)
    return "–" if x is None else f"{100 * x['diff']:+.2f} [{100 * x['lo']:+.2f}, {100 * x['hi']:+.2f}]"


def _rate_cells(r: dict[str, Any]) -> str:
    return f"{_pc(r['success'])} | {_pc(r['irr'], 2)} | {_pc(r['give_up'])}"


def summary_markdown(rep: dict[str, Any]) -> str:
    c, R, tg = rep["counts"], rep["readings"], rep["truth_gap"]
    L = ["# Oracle arm: the readings (design 2026-11-08)", ""] + [f"- {k}: {v}" for k, v in rep["provenance"].items()] + [""]
    L += ["## Counts and checks", "",
          f"- Stage 2 test cases {c['stage2_test_cases']}: Stage 2 dropped {c['stage2_dropped']}, the oracle dropped "
          f"{c['oracle_dropped']} ({c['oracle_only_dropped']} of them kept by Stage 2; skips {c['oracle_skips']}).",
          f"- Paired cases {c['paired']} on {c['objects']} objects ({c['thin_cases']} on thin plastic); groups {c['groups']}.",
          f"- Determinism (reading 6): {R['6_determinism']['compared']} cases compared, 0 mismatches, "
          f"{R['6_determinism']['missing']} without a simulation.",
          f"- The simulator's truth against the particle ({tg['records']} cases): mass gap {_gap(tg['mass_rel'], 100.0, 4)} % of the "
          f"mass, CoM gap {_gap(tg['com_mm'], 1.0, 4)} mm.", "",
          "## 1 and 2. Differences, paired by case (points, 95 % object bootstrap)", "",
          "| comparison | cases (objects) | success | irreversible | give-up |", "|---|---|---|---|---|"]
    for name, k in (("oracle − R0", "1_oracle_vs_r0"), ("oracle − R4", "2_oracle_vs_r4")):
        for part, label in (("all", "all"), ("no_thin", "without thin plastic")):
            b = R[k][part]
            L.append(f"| {name}, {label} | {b['cases']} ({b['objects']}) | {_ci(b, 'success')} | {_ci(b, 'irr')} | {_ci(b, 'give_up')} |")
    b0, b4 = R["1_oracle_vs_r0"]["all"], R["2_oracle_vs_r4"]["all"]
    L += ["", "| arm (paired cases) | success % | irreversible % | give-up % |", "|---|---|---|---|",
          f"| oracle | {_rate_cells(b0['rate_a'])} |", f"| R0 at r 2 | {_rate_cells(b0['rate_b'])} |", f"| R4 | {_rate_cells(b4['rate_b'])} |",
          "", f"## 3. The reading rule: {R['3_reading']} (without thin plastic: {R['3_reading_no_thin']})", "",
          "## 4. The first decisions, oracle against R0 (paired cases)", "",
          "| group | cases | oracle success % | oracle irreversible % | R0 success % | R0 irreversible % | the oracle's first labels |",
          "|---|---|---|---|---|---|---|"]
    for g, v in R["4_groups"]["all"].items():
        L.append(f"| {g} | {v['n']} | {_pc(v['oracle']['success'])} | {_pc(v['oracle']['irr'], 2)} | {_pc(v['r0']['success'])} | "
                 f"{_pc(v['r0']['irr'], 2)} | {v['oracle_first_labels']} |")
    L += ["", "## 5. Calibration at the truth (the oracle's attempts)", ""]
    for part, label in (("all", "all"), ("no_thin", "without thin plastic")):
        cal = R["5_calibration"][part]
        for stat in ("placed", "irr"):
            x = cal[stat]
            L.append(f"- {label}, P({stat}): {x['n']} attempts, predicted sum {x['predicted_sum']:.2f}, observed {x['observed_sum']}, "
                     f"Brier {x['brier'] if x['brier'] is None else round(x['brier'], 4)}.")
    L += ["", "| bin | attempts | P(placed) predicted | observed | P(irr) predicted | observed |", "|---|---|---|---|---|---|"]
    cal = R["5_calibration"]["all"]
    for i, (bp, bi) in enumerate(zip(cal["placed"]["bins"], cal["irr"]["bins"])):
        L.append(f"| {i + 1} | {bp['n']} | {bp['predicted']:.3f} | {bp['observed']:.3f} | {bi['predicted']:.3f} | {bi['observed']:.3f} |")
    L += ["", "(The P(placed) and P(irr) bins sort the attempts separately: row i of the two halves holds different attempts.)", "",
          "## 7. Model-only curves of R0 (predicted, test cases; for the figure only)", "",
          "| r | oracle success | oracle P(irr) | oracle give-up | Stage 1 prior success | Stage 1 P(irr) | Stage 1 give-up |",
          "|---|---|---|---|---|---|---|"]
    for po, pp in zip(R["7_model_curves"]["oracle"], R["7_model_curves"]["r0_stage1"]):
        L.append(f"| {po['knob']:g} | {_pc(po['success'])} | {_pc(po['irr'], 2)} | {_pc(po['give_up'])} | {_pc(pp['success'])} | "
                 f"{_pc(pp['irr'], 2)} | {_pc(pp['give_up'])} |")
    L += ["", "## 8. The menu-limit groups (from Stage 2's R0 and R4 decisions)", "",
          "| group | cases | oracle rows | oracle − R0 success | irreversible | give-up | oracle − R4 success | irreversible | give-up | reading rule |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for g, name in MENU_GROUPS.items():
        v = R["8_menu_limit"][g]
        b0, b4 = v["all"]["oracle_vs_r0"], v["all"]["oracle_vs_r4"]
        L.append(f"| ({g}) {name} | {v['n']} | {v['oracle_rows']} | {_ci(b0, 'success')} | {_ci(b0, 'irr')} | {_ci(b0, 'give_up')} | "
                 f"{_ci(b4, 'success')} | {_ci(b4, 'irr')} | {_ci(b4, 'give_up')} | {v['all']['reading']} |")
    L += ["", "| group, without thin plastic | cases | oracle − R0 irreversible | oracle − R4 irreversible |", "|---|---|---|---|"]
    for g, name in MENU_GROUPS.items():
        v = R["8_menu_limit"][g]["no_thin"]
        L.append(f"| ({g}) {name} | {v['oracle_vs_r0']['cases']} | {_ci(v['oracle_vs_r0'], 'irr')} | {_ci(v['oracle_vs_r4'], 'irr')} |")
    c8 = R["8_menu_limit"]["c"]
    L += ["", "| (c): the oracle's first row against R0's | cases | oracle success % | oracle irreversible % | R0 success % | R0 irreversible % |",
          "|---|---|---|---|---|---|"]
    for ch, x in c8["row_change_outcomes"].items():
        L.append(f"| {ch} | {c8['row_change'][ch]} | {_pc(x['oracle']['success'])} | {_pc(x['oracle']['irr'], 2)} | "
                 f"{_pc(x['r0']['success'])} | {_pc(x['r0']['irr'], 2)} |")
    return "\n".join(L) + "\n"


def oracle_figure(rep: dict[str, Any], path: str) -> None:
    """Simulated success against irreversible damage on the paired test cases: each Stage 2 rule's curve (its locked point
    ringed), R4 (a star) and the oracle (a diamond), on one pair of axes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    r4, o = f["curves"]["R4"][0], f["oracle"]
    ax.scatter([100 * r4["irr"]], [100 * r4["success"]], marker="*", s=110, color="#52514e", edgecolors="white", linewidths=1,
               zorder=5, label=RULE_NAMES["R4"])
    ax.annotate("R4", (100 * r4["irr"], 100 * r4["success"]), xytext=(5, -9), textcoords="offset points", fontsize=8, color="#52514e")
    ax.scatter([100 * o["irr"]], [100 * o["success"]], marker="D", s=60, color="#1f1e1c", edgecolors="white", linewidths=1, zorder=6,
               label="oracle: R0 at r 2, belief at the true setting")
    ax.annotate("oracle", (100 * o["irr"], 100 * o["success"]), xytext=(6, 4), textcoords="offset points", fontsize=8, color="#1f1e1c")
    ax.set_xlim(left=0.0)
    _end_labels(ax, ends)
    ax.set_xlabel("irreversible damage (% of episodes)")
    ax.set_ylabel("task success (% of episodes)")
    ax.set_title(f"Test objects, {o['n']} paired cases; rings: Stage 2's locked points", fontsize=10)
    ax.grid(True, color="#e6e5e0", linewidth=0.8, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> Any:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="the oracle's episodes in the simulator")
    r.add_argument("--stage1", required=True, help="comma-separated Stage 1 run dirs (Stage 2's --stage1)")
    r.add_argument("--locked", required=True, help="Stage 2's locked.json")
    r.add_argument("--out", required=True)
    r.add_argument("--cases", default=None, help="comma-separated object:idx cases in place of the shard")
    r.add_argument("--sides", default="test")
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--shards", type=int, default=1)
    r.add_argument("--workers", type=int, default=1)
    s = sub.add_parser("smoke", help="the smoke report (validation runs only) and the full run's projection")
    s.add_argument("--runs", required=True, help="comma-separated smoke run dirs")
    s.add_argument("--stage2", required=True, help="comma-separated Stage 2 run dirs")
    s.add_argument("--out", required=True)
    m = sub.add_parser("summary", help="the readings from finished runs")
    m.add_argument("--runs", required=True, help="comma-separated oracle run dirs")
    m.add_argument("--stage2", required=True, help="comma-separated Stage 2 run dirs")
    m.add_argument("--stage2-summary", required=True, help="Stage 2's stage2_summary.json")
    m.add_argument("--locked", required=True, help="Stage 2's locked.json")
    m.add_argument("--out", required=True)
    a = p.parse_args(argv)
    return {"run": run, "smoke": smoke, "summary": summary}[a.cmd](a)


if __name__ == "__main__":
    main(sys.argv[1:])
