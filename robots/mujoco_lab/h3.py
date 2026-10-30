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


"""Plan H3, recoverability × uncertainty (the quasi-option value): the output checks, case lists, rebuilt policy runs,
common cohort, contrasts, interactions, power, gates and summary over calibrate outputs (plan
``plans/2026-10-31-grasp-place-h3-recoverability-uncertainty-plan.md`` in daily-logs).

A condition is a recovery level (absorbing, rec20, rec0: calibrate ``--reset-s`` none, 20, 0) and a spread (wide f = 1,
narrow f = 0.25), all at the one cost cell κ = 10, A = 10, s = 1. Each condition and seed has four arms: never
(policy_no_probe), probe_once, sham_once and the policy. At the verdict seeds the policy runs on the never arm's probe
cases only (``calibrate --cases``), and full_policy rebuilds the full run: every other case takes its never row (gate G1).
Every contrast is a mean over one common cohort, the cases (object, idx, seed) counted (no error, no artifact) in every
arm and every condition of their seed, so D = I + M holds case by case and the cells of an interaction share their
cases. Intervals are 95 % percentile intervals of one joint object bootstrap (sweep.bootstrap_weights): one multiplicity
per object for every seed, condition and arm, every contrast and interaction recomputed inside each draw.

    python -P -m robots.mujoco_lab.h3 valid FILE --revision R --protocol P --reset-s X --spread F --seed S (--n N | --cases F)
    python -P -m robots.mujoco_lab.h3 cases --never FILE --out F
    python -P -m robots.mujoco_lab.h3 gates --dir D --m4-dir M --out F
    python -P -m robots.mujoco_lab.h3 power --dir D --out F
    python -P -m robots.mujoco_lab.h3 summary --dir D --out F --markdown M [--power F]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from numbers import Real
from typing import Any, Callable

import numpy as np

from robots.mujoco_lab import tracking
from robots.mujoco_lab.calibrate import FORCED, FORCED_RULE, cases_sha256
from robots.mujoco_lab.corpus import COM_S_RANGE, FILLED_MASS_SCALE_RANGE, MASS_SCALE_RANGE, MU_RANGE, shrunk_range
from robots.mujoco_lab.sweep import (_ci, _dump, _event, _key, _num, _ok, _strip, arm_metrics, bootstrap_weights, cell,
                                     check_identical, load_dir, load_run, probe_cases, variance_split, verdict)
from robots.mujoco_lab.sweep import valid as sweep_valid
from rpent.skills.pick_place.probes import DAMAGED
from rpent.skills.pick_place.protocol import IRREVERSIBLE

CARE, ABSTAIN, SCALE = 10.0, 10.0, 1.0                          # decision 2: the one cost cell (5κ + A = 60)
RECOVERY = {"absorbing": None, "rec20": 20.0, "rec0": 0.0}      # decision 3: calibrate --reset-s (None: absorbing)
SPREAD = {"wide": 1.0, "narrow": 0.25}                          # decision 4: calibrate --spread
CONDITIONS = tuple((r, u) for r in RECOVERY for u in SPREAD)
ARMS = {"policy_no_probe": "never", "probe_once": "probe_once", "sham_once": "sham_once", "policy": "policy"}   # protocol: arm
QUANTITIES = {"D": ("never", "probe_once"), "I": ("sham_once", "probe_once"), "M": ("never", "sham_once"), "P": ("never", "policy")}
X_WEIGHTS = {("absorbing", "wide"): 1.0, ("rec20", "wide"): -1.0, ("absorbing", "narrow"): -1.0, ("rec20", "narrow"): 1.0}
PILOT_SEED = 11               # stage 0: gates and power, never a verdict
K_CHOICES = (2, 4, 6)         # decision 9: the verdict seeds 12 to 11 + k
POWER_SHARE = 0.7             # decision 9: 1.96 SE(k) ≤ 0.7 X̂_D (80 % power)
N_BOOT = 2000
GR = {"pos_err_mm": 1.0, "rot_err_deg": 0.1, "qpos_err": 1e-9, "qvel_err": 1e-9, "mocap_err": 1e-9}   # gate Gr's limits (decision 8)
TRACE = ("probe", "kind", "reading", "thrown", "displacement_m", "tilt_deg", "cost")    # gate Gp: the probe trace
CHANGES = ("same", "other_row", "decline", "terminal")          # decision 6: the decision change where the probe ran
M4_POLICY, M4_NEVER = "policy_k10_A10_s1_seed11.json", "policy_no_probe_k10_A10_s1_seed11_cases.json"   # gate Ga
WALL = ("seconds",)            # a row's wall time, which no rerun reproduces: gate Ga skips it


def label(cond: tuple[str, str]) -> str:
    return f"{cond[0]}/{cond[1]}"


def condition(run: dict[str, Any]) -> tuple[str, str]:
    """(recovery, spread) of a run; a ValueError for a run outside the design (another cost cell, reset time or spread)."""
    care, A, s = cell(run)
    if not (_num(care, CARE) and _num(A, ABSTAIN) and _num(s, SCALE)):
        raise ValueError(f"{run.get('protocol')} at care {care}, A {A}, s {s} is not H3's cell ({CARE:g}, {ABSTAIN:g}, {SCALE:g})")
    spread = 1.0 if run.get("spread") is None else run["spread"]
    rec = next((r for r, x in RECOVERY.items() if _num(run.get("reset_s"), x)), None)
    u = next((k for k, f in SPREAD.items() if _num(spread, f)), None)
    if rec is None or u is None:
        raise ValueError(f"reset_s {run.get('reset_s')} and spread {run.get('spread')} are not an H3 condition")
    return rec, u


def _rows(run: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {_key(r): r for r in run["rows"]}


# ---------------------------------------------------------------- the runs

def full_policy(never: dict[str, Any], policy_subset: dict[str, Any]) -> dict[str, Any]:
    """The full policy run of one condition and seed (decision 5) from the policy's run on the probe cases: each case of
    ``never`` whose first recommendation is a probe takes its subset row, every other case its never row (the policy
    runs the same actions there: gate G1). A ValueError when a probe case has no row in the subset, or when the subset
    holds a case that is not a probe case of ``never`` (so a filled-in case is never a probe case)."""
    if int(never["seed"]) != int(policy_subset["seed"]) or condition(never) != condition(policy_subset):
        raise ValueError("the never run and the policy subset differ in seed or condition")
    sub = _rows(policy_subset)
    probes = {_key(r) for r in never["rows"] if r.get("recommendation") == "probe"}
    extra = sorted(set(sub) - probes)
    if extra:
        raise ValueError(f"the policy subset holds {len(extra)} cases that are not a probe case of the never run, e.g. {extra[:3]}")
    missing = sorted(probes - set(sub))
    if missing:
        raise ValueError(f"{len(missing)} probe cases have no row in the policy subset, e.g. {missing[:3]}")
    rows = sorted((sub.get(_key(r), r) for r in never["rows"]), key=_key)
    return dict(policy_subset, rows=rows, n=len(rows), rebuilt={"subset": len(sub), "filled": len(rows) - len(sub)})


def index(runs: list[dict[str, Any]]) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    """{(recovery, spread, arm, seed): run} over H3's protocols (other runs are skipped); a policy run on a cases file is
    rebuilt by full_policy from its never run. At a verdict seed whose never run has no probe case and no policy run, the
    policy run is the never run (decision 5 at an empty subset: the runner skips that job). A ValueError for two runs of
    one key or a subset without its never run."""
    keyed: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for run in runs:
        arm = ARMS.get(run.get("protocol"))
        if arm is None:
            continue
        key = (*condition(run), arm, int(run["seed"]))
        if key in keyed:
            raise ValueError(f"two runs of {key}")
        keyed[key] = run
    out = {}
    for key, run in keyed.items():
        if key[2] == "policy" and run.get("cases"):
            never = keyed.get((key[0], key[1], "never", key[3]))
            if never is None:
                raise ValueError(f"the policy subset {key} has no never run")
            run = full_policy(never, run)
        out[key] = run
    for (rec, u, arm, seed), never in keyed.items():
        if (arm == "never" and seed != PILOT_SEED and (rec, u, "policy", seed) not in keyed
                and not any(r.get("recommendation") == "probe" for r in never["rows"])):
            empty = dict(never, protocol="policy", rows=[], cases={"path": None, "n": 0, "sha256": None})
            out[(rec, u, "policy", seed)] = full_policy(never, empty)
    return out


def _cohort(runs: dict[tuple[str, str, str, int], dict[str, Any]], ok: Callable[[dict[str, Any]], bool] = _ok,
            pairs: tuple[str, ...] | None = None, conditions: tuple[tuple[str, str], ...] = CONDITIONS) -> set[tuple[str, int, int]]:
    """The cases (object, idx, seed) whose rows pass ``ok`` in every arm (or the arms ``pairs``) of every condition (or
    ``conditions``) of their seed."""
    out = set()
    for seed in sorted({k[3] for k in runs}):
        keep = None
        for cond in conditions:
            for arm in pairs or tuple(ARMS.values()):
                run = runs.get((*cond, arm, seed))
                if run is None:
                    raise ValueError(f"seed {seed} has no {arm} run in {label(cond)}")
                good = {_key(r) for r in run["rows"] if ok(r)}
                keep = good if keep is None else keep & good
        out |= {(o, i, seed) for o, i in keep or ()}
    return out


def cohort(runs: dict[tuple[str, str, str, int], dict[str, Any]]) -> set[tuple[str, int, int]]:
    """Decision 6's common cohort over index()'s runs: the cases (object, idx, seed) counted (no error, no artifact) in
    every arm and every condition of their seed. A ValueError when a seed lacks an arm in a condition."""
    return _cohort(runs)


# ---------------------------------------------------------------- contrasts and interactions

def _costs(runs: dict[tuple[str, str, str, int], dict[str, Any]], keys: list[tuple[str, int, int]], cond: tuple[str, str],
           arm: str) -> np.ndarray:
    rows = {s: _rows(runs[(*cond, arm, s)]) for s in sorted({k[2] for k in keys})}
    return np.array([float(rows[s][(o, i)]["cost"]) for o, i, s in keys])


def case_values(runs: dict[tuple[str, str, str, int], dict[str, Any]], cases: set[tuple[str, int, int]],
                conditions: tuple[tuple[str, str], ...] = CONDITIONS,
                arms: tuple[str, ...] = tuple(ARMS.values())) -> tuple[list[tuple[str, int, int]], dict[tuple[str, str], dict[str, np.ndarray]]]:
    """The cases (sorted) and, per condition, each arm's cost (of ``arms``) and each quantity of QUANTITIES whose two arms
    are read (cost(a) − cost(b)) per case."""
    keys = sorted(cases)
    vals: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for cond in conditions:
        cost = {arm: _costs(runs, keys, cond, arm) for arm in arms}
        vals[cond] = {**{f"cost_{arm}": c for arm, c in cost.items()},
                      **{q: cost[a] - cost[b] for q, (a, b) in QUANTITIES.items() if a in cost and b in cost}}
    return keys, vals


def mean_draws(u: np.ndarray, case_objects: list[str], objects: list[str], W: np.ndarray,
               weights: np.ndarray | None = None) -> dict[str, Any]:
    """The mean of the per-case values ``u`` (weighted by ``weights`` when given), its joint-bootstrap draws (W: rows of
    object multiplicities over ``objects``) and 95 % interval; n is the number of cases."""
    pos = {o: j for j, o in enumerate(objects)}
    at = [pos[o] for o in case_objects]
    w = np.ones(len(at)) if weights is None else np.asarray(weights, dtype=float)
    S, n = np.zeros(len(objects)), np.zeros(len(objects))
    np.add.at(S, at, w * np.asarray(u, dtype=float))
    np.add.at(n, at, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (W @ S) / (W @ n)
    N = float(n.sum())
    return {"value": float(S.sum() / N) if N else math.nan, "ci": _ci(draws), "draws": draws, "n": len(at)}


def contrasts(runs: dict[tuple[str, str, str, int], dict[str, Any]], W: np.ndarray, objects: list[str],
              cohort: set[tuple[str, int, int]]) -> dict[str, Any]:
    """Decision 6 per condition: D = cost(never) − cost(probe_once), I = cost(sham_once) − cost(probe_once),
    M = cost(never) − cost(sham_once) and P = cost(never) − cost(policy), each the mean over ``cohort`` with its
    joint-bootstrap draws (W, shared by every condition) and interval; each arm's mean cost; the number of cases."""
    keys, vals = case_values(runs, cohort)
    objs = [k[0] for k in keys]
    out = {}
    for cond in CONDITIONS:
        out[cond] = {q: mean_draws(vals[cond][q], objs, objects, W) for q in QUANTITIES}
        out[cond]["cost"] = {arm: mean_draws(vals[cond][f"cost_{arm}"], objs, objects, W) for arm in ARMS.values()}
        out[cond]["n"] = len(keys)
    return {"conditions": out, "n": len(keys)}


def combine(c: dict[str, Any], quantity: str, weights: dict[tuple[str, str], float]) -> dict[str, Any]:
    """Σ weight × quantity over the conditions of contrasts' result ``c``: the value, the draws (combined inside each
    joint draw) and the interval."""
    value = sum(w * c["conditions"][k][quantity]["value"] for k, w in weights.items())
    draws = sum(w * c["conditions"][k][quantity]["draws"] for k, w in weights.items())
    return {"value": float(value), "ci": _ci(draws), "draws": draws}


def interaction(c: dict[str, Any], quantity: str) -> dict[str, Any]:
    """Decision 6: X_q = [q(absorbing, wide) − q(rec20, wide)] − [q(absorbing, narrow) − q(rec20, narrow)] from contrasts'
    result ``c``: the value, its interval (X recomputed in every joint draw) and verdict (pass when the interval lies above
    0, fail below, otherwise undecided)."""
    out = combine(c, quantity, X_WEIGHTS)
    out["verdict"] = verdict(out["ci"])
    return out


# ---------------------------------------------------------------- power (decision 9)

def power(pilot: dict[tuple[str, str, str, int], dict[str, Any]]) -> dict[str, Any]:
    """Decision 9 on the pilot (index()'s runs of the pilot seed, the policy in full): X̂_D on the common cohort, the
    variance split (σ_b², σ_w²) of the per-case contributions to X_D (each case's D in the four cells, combined as X),
    SE(k)² = (σ_b² + σ_w² / (c k)) / n for k verdict seeds (c: the cohort's cases per object, n: its objects) and k, the
    smallest of K_CHOICES with 1.96 SE(k) ≤ 0.7 X̂_D; k = 6 and ``target_met`` False when none qualifies or X̂_D ≤ 0.
    Also the never arm's first-recommendation probe cases per condition (the size of a verdict seed's policy subset)
    and the cohort's share of the cases."""
    keys, vals = case_values(pilot, cohort(pilot), conditions=tuple(X_WEIGHTS))
    if not keys:
        raise ValueError("the pilot has no case in its common cohort")
    u = sum(w * vals[cond]["D"] for cond, w in X_WEIGHTS.items())
    objs = np.array([k[0] for k in keys])
    sb, sw = variance_split(u, objs)
    n_obj = len(set(objs.tolist()))
    c = len(keys) / n_obj
    x = float(np.mean(u))
    se = {str(k): math.sqrt((sb + sw / (c * k)) / n_obj) for k in K_CHOICES}
    k = next((k for k in K_CHOICES if x > 0 and 1.96 * se[str(k)] <= POWER_SHARE * x), None)
    never = {cond: [run for key, run in pilot.items() if key[:3] == (*cond, "never")] for cond in CONDITIONS}
    probes = {label(cond): sum(r.get("recommendation") == "probe" for run in rs for r in run["rows"]) for cond, rs in never.items()}
    n_all = sum(len(run["rows"]) for run in never[("absorbing", "wide")])
    return {"x_d": x, "sigma_b2": sb, "sigma_w2": sw, "n_objects": n_obj, "cases_per_object": c, "n_cases": len(keys), "se": se,
            "k": K_CHOICES[-1] if k is None else k, "target_met": k is not None, "note": None if k is not None else "power target not met",
            "probe_cases": probes, "cohort_share": len(keys) / n_all if n_all else math.nan}


# ---------------------------------------------------------------- gates (decision 8)

def _gate(failures: list[Any], **extra: Any) -> dict[str, Any]:
    return {"pass": not failures, "n_failures": len(failures), "failures": failures[:20], **extra}


def restore_ok(check: dict[str, Any], start_at_rest: bool = True) -> bool:
    """Gate Gr on one restore's lab check: the object's pose within 1 mm and 0.1°, the joint positions, velocities and
    the hand as at the start, the same case (θ, fill, fragility, fixture), the same episode, and the scene at rest exactly
    when the case's start was (a start that never settled, an artifact case, is rebuilt as it was)."""
    return (all(isinstance(check.get(k), Real) and not isinstance(check.get(k), bool) and check[k] <= lim for k, lim in GR.items())
            and all(check.get(k) is True for k in ("same_case", "episode")) and check.get("at_rest") is bool(start_at_rest))


def _start_at_rest(row: dict[str, Any]) -> bool:
    return bool((row.get("reset") or {}).get("at_rest", True))


def _below_root(path: Any) -> Any:
    """A bundle path from robots/mujoco_lab/ on, so that runs made in two checkouts compare."""
    if not isinstance(path, str):
        return path
    i = path.find("robots/mujoco_lab/")
    return path if i < 0 else path[i:]


def row_diff(h3: dict[str, Any], m4: dict[str, Any]) -> list[str]:
    """The keys of a milestone 4 row whose value differs in the H3 row (the bundle compared below its checkout root; the
    wall time, WALL, skipped)."""
    out = []
    for k, v in m4.items():
        if k in WALL:
            continue
        if k not in h3:
            out.append(k)
        elif (_below_root(h3[k]) != _below_root(v)) if k == "bundle" else h3[k] != v:
            out.append(k)
    return out


def _sham_ok(p: dict[str, Any]) -> bool:
    h = p.get("belief_hash_before")
    ended = p.get("kind") == DAMAGED or bool(p.get("thrown"))
    return (p.get("sham") is True and h is not None and p.get("belief_hash_after") == h
            and (p.get("belief_hash_next") == h or (p.get("belief_hash_next") is None and ended)))


def _checks(runs: dict[tuple[str, str, str, int], dict[str, Any]], seed: int,
            keep: set[tuple[str, int]] | None = None) -> tuple[dict[str, list[Any]], Counter]:
    """Decision 8's case checks at one seed, per condition, over every row (or the cases ``keep``): first_menus, the
    policy and the never arm record the same first menu and recommendation; Gr, every restore's lab check passes
    restore_ok against its case's start; Gs, every sham probe's belief hashes are equal before and after, and at the next
    menu, which is missing only when the probe ended the case, and no real probe carries a sham record; Gp, probe_once and
    sham_once force the same probe with the same trace (TRACE). A run missing at the seed is listed under ``missing``."""
    f: dict[str, list[Any]] = {"first_menus": [], "Gr": [], "Gs": [], "Gp": [], "missing": []}
    n: Counter = Counter()
    for cond in CONDITIONS:
        got = {arm: runs.get((*cond, arm, seed)) for arm in ARMS.values()}
        lost = [arm for arm, r in got.items() if r is None]
        if lost:
            f["missing"] += [(label(cond), arm, seed) for arm in lost]
            continue
        rows = {arm: [r for r in run["rows"] if keep is None or _key(r) in keep] for arm, run in got.items()}
        nv = _rows(got["never"])
        for r in rows["policy"]:
            m = nv.get(_key(r))
            if m is None or m.get("recommendation") != r.get("recommendation") or m.get("first_menu") != r.get("first_menu"):
                f["first_menus"].append((label(cond), seed, *_key(r)))
        for arm, rs in rows.items():
            for r in rs:
                for rst in r.get("restores") or []:
                    n["restores"] += 1
                    if not restore_ok(rst.get("check") or {}, _start_at_rest(r)):
                        f["Gr"].append((label(cond), arm, seed, *_key(r), rst.get("check")))
        sv = _rows(got["sham_once"])
        for r in rows["probe_once"]:
            s = sv.get(_key(r))
            n["pairs"] += 1
            rp, sp = r.get("probes") or [], (s or {}).get("probes") or []
            if any("sham" in p for p in rp):
                f["Gs"].append((label(cond), "probe_once", seed, *_key(r), "a real probe carries a sham record"))
            if (s is None or s.get("forced_probe") != r.get("forced_probe") or len(sp) != len(rp)
                    or any({k: p.get(k) for k in TRACE} != {k: q.get(k) for k in TRACE} for p, q in zip(rp, sp))):
                f["Gp"].append((label(cond), seed, *_key(r)))
        for s in rows["sham_once"]:
            for p in s.get("probes") or []:
                n["shams"] += 1
                if not _sham_ok(p):
                    f["Gs"].append((label(cond), "sham_once", seed, *_key(s)))
    return f, n


def gates(runs: dict[tuple[str, str, str, int], dict[str, Any]], m4_dir: str, seed: int = PILOT_SEED) -> dict[str, Any]:
    """Decision 8 on the pilot seed's runs (index(); the policy in full), each gate with its failures (the first 20):
    G1, per condition, the cases whose first recommendation is not a probe run the same in the policy and the never arm
    (sweep.check_identical); first_menus, Gr, Gs and Gp, _checks over every row; Ga, the absorbing and wide condition
    reproduces milestone 4's seed-11 outputs in ``m4_dir``: the policy run row for row and the never arm on the cases of
    milestone 4's never subset run, on milestone 4's row keys (row_diff). ``pass`` when all pass; a missing run fails its
    gates."""
    f, n = _checks(runs, seed)
    g1, ga, missing = [], [], f["missing"]
    for cond in CONDITIONS:
        policy, never = runs.get((*cond, "policy", seed)), runs.get((*cond, "never", seed))
        if policy is None or never is None:
            continue
        g1 += [(label(cond), *k) for k in check_identical(policy, never)]
        n["g1_checked"] += sum(r.get("recommendation") != "probe" for r in policy["rows"])
    for name, arm in ((M4_POLICY, "policy"), (M4_NEVER, "never")):
        h3 = runs.get(("absorbing", "wide", arm, seed))
        try:
            m4 = load_run(os.path.join(m4_dir, name))
        except (OSError, ValueError) as exc:
            ga.append((name, f"unreadable: {exc}"[:200]))
            continue
        if h3 is None:
            ga.append((name, "no H3 run"))
            continue
        hv = _rows(h3)
        if arm == "policy" and set(hv) != {_key(r) for r in m4["rows"]}:
            ga.append((name, "the case sets differ"))
        for r in m4["rows"]:
            diff = ["missing"] if _key(r) not in hv else row_diff(hv[_key(r)], r)
            n["ga_rows"] += 1
            if diff:
                ga.append((name, *_key(r), diff[:5]))
    out = {"G1": _gate(g1 + missing, checked=n["g1_checked"]), "first_menus": _gate(f["first_menus"] + missing),
           "Gr": _gate(f["Gr"] + missing, restores=n["restores"]), "Gs": _gate(f["Gs"] + missing, shams=n["shams"]),
           "Gp": _gate(f["Gp"] + missing, pairs=n["pairs"]), "Ga": _gate(ga, rows=n["ga_rows"]), "seed": seed}
    out["pass"] = all(out[k]["pass"] for k in ("G1", "first_menus", "Gr", "Gs", "Gp", "Ga"))
    return out


# ---------------------------------------------------------------- the summary

def in_narrow(row: dict[str, Any], f: float = SPREAD["narrow"]) -> bool:
    """Whether a case's truth lies in the narrow class's ranges (decision 4's matched-truth check): the mass scale (on a
    filled container's range when its fill is above 0), mu, and a non-None com_s inside shrunk_range at ``f``."""
    th = row["theta"]
    lo, hi = shrunk_range(FILLED_MASS_SCALE_RANGE if float(row.get("fill") or 0.0) > 0.0 else MASS_SCALE_RANGE, f, log=True)
    ok = lo <= float(th["mass_scale"]) <= hi
    lo, hi = shrunk_range(MU_RANGE, f)
    ok = ok and lo <= float(th["mu"]) <= hi
    if th.get("com_s") is not None:
        lo, hi = shrunk_range(COM_S_RANGE, f)
        ok = ok and lo <= float(th["com_s"]) <= hi
    return bool(ok)


def _range_mass(bounds: tuple[float, float], f: float, log: bool = False) -> float:
    """The share of a uniform law on ``bounds`` (log-uniform when ``log``) inside shrunk_range(bounds, f)."""
    lo, hi = shrunk_range(bounds, f, log=log)
    g = math.log if log else float
    return (g(hi) - g(lo)) / (g(bounds[1]) - g(bounds[0]))


def selection_p(row: dict[str, Any], f: float = SPREAD["narrow"]) -> float:
    """P(in_narrow | the case's branch) under the truth generator (corpus.draw_theta and draw_damage: the mass scale
    log-uniform on its range, a filled container's on its own; mu uniform; com_s uniform when the CoM shifts): the
    product of the constrained ranges' masses, f² without a shift or for a filled container and f³ with one."""
    filled = float(row.get("fill") or 0.0) > 0.0
    p = _range_mass(FILLED_MASS_SCALE_RANGE if filled else MASS_SCALE_RANGE, f, log=True) * _range_mass(MU_RANGE, f)
    return p * _range_mass(COM_S_RANGE, f) if row["theta"].get("com_s") is not None else p


def matched_truth(runs: dict[tuple[str, str, str, int], dict[str, Any]], W: np.ndarray, objects: list[str],
                  cases: set[tuple[str, int, int]]) -> dict[str, Any]:
    """Decision 4's matched-truth check (no verdict): on the cohort's cases whose wide truth lies in the narrow ranges
    (in_narrow, read from the absorbing and wide never row), D and I in the wide conditions and the recovery difference
    Q = D(absorbing, wide) − D(rec20, wide). Each case enters with weight 1 / selection_p: the selection keeps every case
    without a CoM shift that a shift case needs a third range for, so the weights restore the narrow class's mixture of
    branches and fills. n counts the cases; n_eff is Kish's (Σw)² / Σw²."""
    keys, vals = case_values(runs, cases, conditions=(("absorbing", "wide"), ("rec20", "wide"), ("rec0", "wide")))
    rows = {s: _rows(runs[("absorbing", "wide", "never", s)]) for s in sorted({k[2] for k in keys})}
    mask = np.array([in_narrow(rows[s][(o, i)]) for o, i, s in keys], dtype=bool)
    objs = [k[0] for k, m in zip(keys, mask) if m]
    w = np.array([1.0 / selection_p(rows[s][(o, i)]) for (o, i, s), m in zip(keys, mask) if m])
    out: dict[str, Any] = {"n": int(mask.sum()), "share": float(mask.mean()) if len(mask) else math.nan,
                           "n_eff": float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else 0.0}
    for cond in (("absorbing", "wide"), ("rec20", "wide"), ("rec0", "wide")):
        out[label(cond)] = {q: mean_draws(vals[cond][q][mask], objs, objects, W, weights=w) for q in ("D", "I")}
    out["Q"] = mean_draws((vals[("absorbing", "wide")]["D"] - vals[("rec20", "wide")]["D"])[mask], objs, objects, W, weights=w)
    return out


def decision_change(runs: dict[tuple[str, str, str, int], dict[str, Any]], cases: set[tuple[str, int, int]]) -> dict[str, Any]:
    """Decision 6 per condition, over the cohort's probe_once cases: the shares of CHANGES among the cases where the probe
    ran, the cases without a probe (a guard flag) and those without a category (an error or an unreachable object)."""
    out = {}
    for cond in CONDITIONS:
        rows = {s: _rows(runs[(*cond, "probe_once", s)]) for s in sorted({k[2] for k in cases})}
        cnt = Counter(rows[s][(o, i)].get("changed") for o, i, s in cases)
        probed = sum(cnt[k] for k in CHANGES)
        out[label(cond)] = {"n_probed": probed, "share": {k: cnt[k] / probed if probed else None for k in CHANGES},
                            "counts": {k: cnt[k] for k in CHANGES}, "no_probe": cnt["no_probe"],
                            "other": sum(v for k, v in cnt.items() if k not in CHANGES and k != "no_probe")}
    return out


def irreversible_events(row: dict[str, Any]) -> int:
    """A case's irreversible events: its attempts with an irreversible outcome and its probes that threw or damaged the
    object (sweep's irreversible cases count a case once; under recovery a case can hold several)."""
    return sum(t.get("outcome") in IRREVERSIBLE for t in row.get("attempts") or []) + sum(_event(p) for p in row.get("probes") or [])


def _arm_tables(runs: dict[tuple[str, str, str, int], dict[str, Any]], W: np.ndarray, objects: list[str],
                cases: set[tuple[str, int, int]]) -> dict[str, Any]:
    """sweep.arm_metrics per condition and arm over the cohort's rows of every seed (the never arm's row is the
    condition's difficulty: decision 4), with the irreversible events per 1,000 cases (decision 6) next to its
    irreversible cases per 1,000."""
    out = {}
    for cond in CONDITIONS:
        for arm in ARMS.values():
            rows = {s: _rows(runs[(*cond, arm, s)]) for s in sorted({k[2] for k in cases})}
            sel = [rows[s][(o, i)] for o, i, s in sorted(cases)]
            m = arm_metrics({"rows": sel}, W, objects)
            e = mean_draws(np.array([irreversible_events(r) for r in sel], dtype=float), [r["object"] for r in sel], objects, W)
            m["irreversible_events_per_1000"] = 1000.0 * e["value"]
            m["irreversible_events_per_1000_ci"] = [1000.0 * x for x in e["ci"]]
            out[f"{label(cond)}/{arm}"] = m
    return out


def verdict_checks(runs: dict[tuple[str, str, str, int], dict[str, Any]], cases: set[tuple[str, int, int]]) -> dict[str, Any]:
    """Decision 8 at the verdict seeds: _checks on each seed's cohort cases (the cases that enter a verdict), so Gs holds
    in every sham case, not only the pilot's; ``pass`` when every check passes."""
    agg: dict[str, list[Any]] = {"first_menus": [], "Gr": [], "Gs": [], "Gp": []}
    n: Counter = Counter()
    for seed in sorted({k[2] for k in cases}):
        f, c = _checks(runs, seed, keep={(o, i) for o, i, s in cases if s == seed})
        for k in agg:
            agg[k] += f[k] + f["missing"]
        n.update(c)
    out: dict[str, Any] = {"first_menus": _gate(agg["first_menus"]), "Gr": _gate(agg["Gr"], restores=n["restores"]),
                           "Gs": _gate(agg["Gs"], shams=n["shams"]), "Gp": _gate(agg["Gp"], pairs=n["pairs"])}
    out["pass"] = all(g["pass"] for g in out.values())
    return out


def _sensitivity(runs: dict[tuple[str, str, str, int], dict[str, Any]], W: np.ndarray, objects: list[str]) -> dict[str, Any]:
    """Decision 6's sensitivity rows: X_D and X_I with pairwise cohorts (each condition's quantity on the cases counted in
    its two arms) and with artifact cases counted (the common cohort of the cases without an error)."""
    out: dict[str, Any] = {}
    pair = {}
    for cond in X_WEIGHTS:
        pair[cond] = {}
        for q in ("D", "I"):
            cases = _cohort(runs, pairs=QUANTITIES[q], conditions=(cond,))
            keys, vals = case_values(runs, cases, conditions=(cond,), arms=QUANTITIES[q])
            pair[cond][q] = mean_draws(vals[cond][q], [k[0] for k in keys], objects, W)
    out["pairwise"] = {q: combine({"conditions": pair}, q, X_WEIGHTS) for q in ("D", "I")}
    loose = _cohort(runs, ok=lambda r: "error" not in r)
    c = contrasts(runs, W, objects, loose)
    out["artifacts_counted"] = {"n": c["n"], **{q: combine(c, q, X_WEIGHTS) for q in ("D", "I")}}
    return out


def summary(runs: list[dict[str, Any]], n_boot: int = N_BOOT, rng_seed: int = 0, power_result: dict[str, Any] | None = None) -> dict[str, Any]:
    """Decision 6's verdicts and tables over the verdict seeds' runs (the pilot seed is left out): H3 (X_D) and H3-info
    (X_I) with verdicts; secondary, without a verdict: X_M, X_P, the quasi-option value per spread
    Q_u = D(absorbing, u) − D(rec20, u), rec0 against the ordered prediction D(absorbing) ≥ D(rec20) ≥ D(rec0), and the
    matched-truth check; the contrasts per condition, the decision change, the arm and difficulty tables, the
    sensitivity rows and decision 8's checks at every verdict seed (verdict_checks); with ``markdown``. JSON-ready: the
    per-draw arrays are dropped."""
    idx = index([r for r in runs if int(r["seed"]) != PILOT_SEED])
    if not idx:
        raise ValueError("no H3 run outside the pilot seed")
    seeds = sorted({k[3] for k in idx})
    objects = sorted({r["object"] for run in idx.values() for r in run["rows"]})
    W = bootstrap_weights(objects, n_boot, np.random.default_rng(rng_seed))
    co = cohort(idx)
    c = contrasts(idx, W, objects, co)
    out: dict[str, Any] = {"seeds": seeds, "n_objects": len(objects), "n_cohort": len(co), "n_boot": n_boot,
                           "h3": interaction(c, "D"), "h3_info": interaction(c, "I")}
    out["secondary"] = {
        "x_m": combine(c, "M", X_WEIGHTS), "x_p": combine(c, "P", X_WEIGHTS),
        "q": {u: combine(c, "D", {("absorbing", u): 1.0, ("rec20", u): -1.0}) for u in SPREAD},
        "rec0_order": {u: {"absorbing_minus_rec20": combine(c, "D", {("absorbing", u): 1.0, ("rec20", u): -1.0}),
                           "rec20_minus_rec0": combine(c, "D", {("rec20", u): 1.0, ("rec0", u): -1.0})} for u in SPREAD},
        "matched_truth": matched_truth(idx, W, objects, co)}
    out["conditions"] = {label(k): v for k, v in c["conditions"].items()}
    out["decision_change"] = decision_change(idx, co)
    out["arms"] = _arm_tables(idx, W, objects, co)
    out["excluded"] = {label(k[:2]) + f"/{k[2]}/seed{k[3]}": sum(not _ok(r) for r in run["rows"]) for k, run in sorted(idx.items())}
    out["sensitivity"] = _sensitivity(idx, W, objects)
    out["checks"] = verdict_checks(idx, co)
    out["power"] = power_result
    out["markdown"] = markdown(out)
    return _strip(out)


def _f(x: Any, nd: int = 3) -> str:
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def _ci_s(d: dict[str, Any], nd: int = 3) -> str:
    return f"{_f(d['value'], nd)} [{_f(d['ci'][0], nd)}, {_f(d['ci'][1], nd)}]"


def markdown(s: dict[str, Any]) -> str:
    """The summary's tables."""
    L = [f"## H3: recoverability × uncertainty (κ 10, A 10, s 1; seeds {s['seeds']}; {s['n_objects']} objects; "
         f"cohort {s['n_cohort']} cases; {s['n_boot']} joint draws)", "",
         "| quantity | value [95 % interval] | verdict |", "|---|---|---|",
         f"| H3: X_D | {_ci_s(s['h3'])} | {s['h3']['verdict']} |", f"| H3-info: X_I | {_ci_s(s['h3_info'])} | {s['h3_info']['verdict']} |",
         f"| X_M (reported) | {_ci_s(s['secondary']['x_m'])} | — |", f"| X_P (reported) | {_ci_s(s['secondary']['x_p'])} | — |", "",
         "| condition | n | D | I | M | P | never cost |", "|---|---|---|---|---|---|---|"]
    for k, v in s["conditions"].items():
        L.append(f"| {k} | {v['n']} | " + " | ".join(_ci_s(v[q]) for q in QUANTITIES) + f" | {_ci_s(v['cost']['never'])} |")
    L += ["", "| spread | Q = D(absorbing) − D(rec20) | D(rec20) − D(rec0) |", "|---|---|---|"]
    for u in SPREAD:
        r = s["secondary"]["rec0_order"][u]
        L.append(f"| {u} | {_ci_s(s['secondary']['q'][u])} | {_ci_s(r['rec20_minus_rec0'])} |")
    m = s["secondary"]["matched_truth"]
    L += ["", f"Matched truth (wide cases inside the narrow ranges: {m['n']}, share {_f(m['share'])}, effective n "
          f"{_f(m['n_eff'], 1)}, weighted by 1 / P(selected | branch)): "
          f"D(absorbing/wide) {_ci_s(m['absorbing/wide']['D'])}, I(absorbing/wide) {_ci_s(m['absorbing/wide']['I'])}, "
          f"Q {_ci_s(m['Q'])}; narrow Q {_ci_s(s['secondary']['q']['narrow'])}.", "",
          "| condition | probed | same | other row | decline | terminal | no probe |", "|---|---|---|---|---|---|---|"]
    for k, v in s["decision_change"].items():
        L.append(f"| {k} | {v['n_probed']} | " + " | ".join(_f(v["share"][c]) for c in CHANGES) + f" | {v['no_probe']} |")
    L += ["", "| condition / arm | placed per 1,000 | abstain rate | irreversible cases per 1,000 | irreversible events per 1,000 | cost |",
          "|---|---|---|---|---|---|"]
    for k, v in s["arms"].items():
        L.append(f"| {k} | {_f(v['placed_per_1000'], 1)} | {_f(v['abstain_rate'])} | {_f(v['irreversible_per_1000'], 1)} | "
                 f"{_f(v['irreversible_events_per_1000'], 1)} | {_f(v['cost'])} |")
    sens = s["sensitivity"]
    L += ["", f"Sensitivity: pairwise cohorts X_D {_ci_s(sens['pairwise']['D'])}, X_I {_ci_s(sens['pairwise']['I'])}; "
          f"artifact cases counted (n {sens['artifacts_counted']['n']}) X_D {_ci_s(sens['artifacts_counted']['D'])}, "
          f"X_I {_ci_s(sens['artifacts_counted']['I'])}."]
    ck = s["checks"]
    L += ["", "Checks at the verdict seeds (decision 8): " + ", ".join(
        f"{k} {'ok' if ck[k]['pass'] else 'FAILED (' + str(ck[k]['n_failures']) + ')'}" for k in ("first_menus", "Gr", "Gs", "Gp"))
          + f" ({ck['Gr']['restores']} restores, {ck['Gs']['shams']} shams, {ck['Gp']['pairs']} pairs)."]
    if s.get("power"):
        p = s["power"]
        L += ["", f"Power (pilot): X̂_D {_f(p['x_d'])}, σ_b² {_f(p['sigma_b2'], 4)}, σ_w² {_f(p['sigma_w2'], 4)}, k {p['k']}"
              + ("" if p.get("target_met") else " (power target not met)") + "."]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- output checks and the command line

def valid(path: str, expect: dict[str, Any]) -> list[str]:
    """Why the output at ``path`` cannot be reused (empty when it can), decision 11: sweep.valid at H3's cell (the code
    revision on a clean tree, the protocol, κ = 10, A = 10, s = 1, the seed and the case count), the reset time, the
    spread, the forced-probe rule (calibrate.FORCED_RULE in the forced arms, else None) and, with ``expect["cases"]`` (a
    cases file), the file's sha256 and the case membership; without one, a full run (no cases record). ``expect``:
    revision, protocol, reset_s, spread, seed, and n or cases."""
    listed = None
    if expect.get("cases"):
        try:
            spec = json.load(open(expect["cases"]))
            listed = {(o, int(i)) for o, idx in spec["cases"].items() for i in idx}
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            return [f"cases file {expect['cases']}: {exc}"[:200]]
    why = sweep_valid(path, {"revision": expect["revision"], "protocol": expect["protocol"], "care": CARE, "abstain": ABSTAIN,
                             "scale": SCALE, "seed": expect["seed"], "n": len(listed) if listed is not None else expect["n"]})
    try:
        run = load_run(path)
        rows = run["rows"]
    except (OSError, ValueError, EOFError, KeyError, TypeError):
        return why
    if not _num(run.get("reset_s"), expect.get("reset_s")):
        why.append(f"reset_s {run.get('reset_s')} is not {expect.get('reset_s')}")
    if not _num(1.0 if run.get("spread") is None else run["spread"], expect["spread"]):
        why.append(f"spread {run.get('spread')} is not {expect['spread']}")
    rule = FORCED_RULE if expect["protocol"] in FORCED else None
    if run.get("forced_rule") != rule:
        why.append(f"forced-probe rule {run.get('forced_rule')!r} is not {rule!r}")
    rec = run.get("cases")
    if listed is None:
        if rec:
            why.append(f"a run on the cases file {rec.get('path')} where a full run is due")
    elif not rec:
        why.append(f"a full run where a run on the cases file {expect['cases']} is due")
    else:
        if rec.get("sha256") != cases_sha256(expect["cases"]):
            why.append(f"cases sha256 {rec.get('sha256')} is not that of {expect['cases']}")
        if {_key(r) for r in rows} != listed:
            why.append(f"the rows are not the cases of {expect['cases']}")
    return why


def _reset_arg(text: str) -> float | None:
    return None if text == "none" else float(text)


def _pilot_index(path: str) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    return index([r for r in load_dir(path) if int(r["seed"]) == PILOT_SEED])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("valid", help="exit 1 when an output cannot be reused")
    v.add_argument("file")
    v.add_argument("--revision", required=True)
    v.add_argument("--protocol", required=True)
    v.add_argument("--reset-s", type=_reset_arg, required=True, help="none (absorbing) or the reset time in seconds")
    v.add_argument("--spread", type=float, required=True)
    v.add_argument("--seed", type=int, required=True)
    v.add_argument("--n", type=int, default=None)
    v.add_argument("--cases", default=None, help="the cases file of a subset run")
    c = sub.add_parser("cases", help="the never arm's probe cases, the policy's subset at a verdict seed")
    c.add_argument("--never", required=True)
    c.add_argument("--out", required=True)
    for name in ("gates", "power", "summary"):
        sp = sub.add_parser(name)
        sp.add_argument("--dir", required=True)
        sp.add_argument("--out", required=True)
        if name == "gates":
            sp.add_argument("--m4-dir", required=True, help="milestone 4's outputs (its seed-11 policy and never runs)")
        if name == "summary":
            sp.add_argument("--markdown", required=True)
            sp.add_argument("--power", default=None, help="the stage-0 power output")
            sp.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)

    if a.cmd == "valid":
        if (a.n is None) == (a.cases is None):
            ap.error("give --n or --cases")
        why = valid(a.file, {"revision": a.revision, "protocol": a.protocol, "reset_s": a.reset_s, "spread": a.spread, "seed": a.seed,
                             "n": a.n, "cases": a.cases})
        print("valid" if not why else "not valid: " + "; ".join(why))
        return 0 if not why else 1
    if a.cmd == "cases":
        never = load_run(a.never)
        if never.get("protocol") != "policy_no_probe" or never.get("cases"):
            print(f"cases: {a.never} is not a full never run")
            return 1
        spec = probe_cases([never])
        _dump(spec, a.out)
        print(f"cases: {sum(len(x) for x in spec['cases'].values())} probe cases over {len(spec['cases'])} objects -> {a.out}")
        return 0
    if a.cmd == "gates":
        out = gates(_pilot_index(a.dir), a.m4_dir)
        _dump(out, a.out)
        tracking.log_run("h3_gates", {"dir": a.dir, "m4_dir": a.m4_dir, "seed": out["seed"]},
                         {"pass": out["pass"], **{f"{k}_pass": g["pass"] for k, g in out.items() if isinstance(g, dict)},
                          **{f"{k}_failures": g["n_failures"] for k, g in out.items() if isinstance(g, dict)}})
        print(json.dumps({k: (g if not isinstance(g, dict) else {x: g[x] for x in g if x != "failures"}) for k, g in _strip(out).items()}, indent=1))
        return 0 if out["pass"] else 1
    if a.cmd == "power":
        out = power(_pilot_index(a.dir))
        _dump(out, a.out)
        tracking.log_run("h3_power", {"dir": a.dir, "seed": PILOT_SEED},
                         {k: out[k] for k in ("x_d", "sigma_b2", "sigma_w2", "n_objects", "n_cases", "cohort_share", "k", "target_met")})
        print(json.dumps(_strip(out), indent=1))
        return 0
    runs = load_dir(a.dir)
    pw = json.load(open(a.power)) if a.power else None
    out = summary(runs, n_boot=a.n_boot, power_result=pw)
    with open(a.markdown, "w") as f:
        f.write(out["markdown"])
    md = out.pop("markdown")
    _dump(out, a.out)
    metrics = {"x_d": out["h3"]["value"], "h3_verdict": out["h3"]["verdict"], "x_i": out["h3_info"]["value"],
               "h3_info_verdict": out["h3_info"]["verdict"], "n_cohort": out["n_cohort"], "checks_pass": out["checks"]["pass"]}
    out["wandb"] = tracking.log_run("h3", {"dir": a.dir, "seeds": out["seeds"]}, metrics)
    print(md)
    if not out["checks"]["pass"]:
        print("h3 summary: a decision 8 check failed at a verdict seed (see checks); the verdicts do not stand", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
