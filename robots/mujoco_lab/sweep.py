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


"""Milestone 4, when does probing pay: the sweep's output checks, case lists, gates, power analysis, paired savings,
verdicts and tables over calibrate outputs (plan ``plans/2026-10-30-grasp-place-milestone4-plan.md`` in daily-logs).

An arm is a calibrate protocol run at one cell (care κ, abstain cost A, probe time scale s) and seed. A saving pairs two
arms case by case: d = cost(b) − cost(a) over the cases counted in both (no error, no artifact), where a is the policy (or
the forced probe) and b the policy that never probes. A case whose first recommendation is not a probe runs the same
actions in the policy and in the never-probe arm (gate G1), so a subset never-probe run (``calibrate --cases``) covers
the probe cases only, and every other case pairs with itself. The never-probe arm runs no probe, so its run at s = 1
pairs with the policy at every s (gate G1b checks this). Intervals are 95 % percentile intervals of one joint object
bootstrap: each draw gives every case of an object, in every seed and cell, the object's multiplicity, and every ratio
is recomputed inside the draw.

    python -P -m robots.mujoco_lab.sweep valid FILE --revision R --sim-revision SIM --protocol P --care K --abstain A --scale S --seed X (--n N | --cases F)
    python -P -m robots.mujoco_lab.sweep cases --dir D --seed S --care K --abstain A --out F
    python -P -m robots.mujoco_lab.sweep check --dir D --seed S
    python -P -m robots.mujoco_lab.sweep risk --dir D --seed S --out F
    python -P -m robots.mujoco_lab.sweep power --dir D --seed S --out F
    python -P -m robots.mujoco_lab.sweep summary --dir D --out F --markdown M [--power F]
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import poisson

from robots.mujoco_lab import tracking
from robots.mujoco_lab.calibrate import _counted
from robots.mujoco_lab.corpus import FILLED_MASS_SCALE_RANGE, MASS_SCALE_RANGE
from rpent.skills.pick_place.evaluate import object_weights
from rpent.skills.pick_place.protocol import IRREVERSIBLE

KAPPAS = (0.5, 1.0, 3.0, 10.0, 30.0)
ABSTAIN = (1.0, 3.0, 10.0)
SCALES = (1.0, 0.25, 0.0)
TEST_CELLS = tuple((k, A, s) for k in (10.0, 30.0) for A in (3.0, 10.0) for s in SCALES)    # 5κ + A from 53 to 160
CHEAP = ((0.5, 1.0, 1.0), (1.0, 1.0, 1.0), (0.5, 3.0, 1.0))                                 # 5κ + A at most 6
GATE_RUNS = (tuple(("policy", (k, A, s)) for k in KAPPAS for A in ABSTAIN for s in SCALES)         # G1 and G1b at a seed (decision 4)
             + tuple(("policy_no_probe", (k, A, 1.0)) for k in KAPPAS for A in ABSTAIN)
             + tuple(("policy_no_probe", c) for c in TEST_CELLS if c[2] != 1.0))
DIAGNOSTIC = ((1.0, 1.0, 1.0), (10.0, 1.0, 1.0))
PILOT_SEED = 11               # stage 0: power and smoke only, never a verdict
BASE_SEED = 12                # stage 1: every arm, the common sample of the 45 cells
MARGIN = 0.02                 # H1-only: a saving below 2 % of J is practically zero
PROBE_RATE_MAX = 0.02         # H1-only (a)
CROSS_RATE = 0.05             # the probe crossover: a probe rate of 5 %
CROSS_REL = 0.05              # the saving crossover: Δ/J of 5 % (A1's rule)
N_BOOT = 2000
SEED_CHOICES = (4, 6, 8)      # decision 7: the number of stage-2 seeds
P_GATE = 0.01                 # G2: the one-sided Poisson p-value


# ---------------------------------------------------------------- runs

def load_run(path: str) -> dict[str, Any]:
    """A calibrate output (``.json`` or ``.json.gz``)."""
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        return json.load(f)


def load_dir(path: str) -> list[dict[str, Any]]:
    """Every calibrate output directly in ``path``; other JSON files (cases files, summaries) are skipped."""
    runs = []
    for p in sorted(glob.glob(os.path.join(path, "*.json")) + glob.glob(os.path.join(path, "*.json.gz"))):
        run = load_run(p)
        if isinstance(run, dict) and "rows" in run and "protocol" in run:
            runs.append(run)
    return runs


def cell(run: dict[str, Any]) -> tuple[float, float | None, float]:
    """(care scale, abstain cost, probe time scale) of a run."""
    A = run.get("abstain_cost")
    return float(run["care_scale"]), None if A is None else float(A), float(run.get("probe_time_scale", 1.0))


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return row["object"], int(row["idx"])


def _num(x: Any, y: Any) -> bool:
    """Two run settings agree: numbers within 1e-9, or both None."""
    if x is None or y is None:
        return x is None and y is None
    return abs(float(x) - float(y)) <= 1e-9


def _subset(never: dict[str, Any]) -> bool:
    """A never-probe run that covers only the probe cases: every run of a seed other than BASE_SEED, and any run made
    with a cases file. Subset pairing on a full run changes nothing, since every case then has its row."""
    return int(never["seed"]) != BASE_SEED or bool(never.get("cases"))


def _ok(row: dict[str, Any]) -> bool:
    return "error" not in row and not row.get("artifact")


def _label(c: tuple[float, float | None, float]) -> str:
    return f"k{c[0]:g}_A{c[1]:g}_s{c[2]:g}"


def valid(path: str, expect: dict[str, Any]) -> list[str]:
    """Why the output at ``path`` cannot be reused (empty when it can): it loads, it comes from ``expect["revision"]`` on
    a tree without uncommitted changes and from the simulator at ``expect["sim_revision"]`` without uncommitted changes,
    neither changed during the run, and its protocol, care, abstain cost, probe time scale, seed and number of cases match
    ``expect`` (keys revision, sim_revision, protocol, care, abstain, scale, seed, n); for a subset run, its cases are the
    (object, idx) set ``expect["cases"]``."""
    try:
        run = load_run(path)
    except FileNotFoundError:
        return ["missing"]
    except (OSError, ValueError, EOFError) as exc:
        return [f"unreadable: {exc}"[:200]]
    if not isinstance(run, dict) or "rows" not in run:
        return ["not a calibrate output"]
    why = []
    code = run.get("code") or {}
    if code.get("revision") != expect["revision"]:
        why.append(f"revision {code.get('revision')} is not {expect['revision']}")
    if code.get("dirty") is not False:
        why.append(f"made from a tree with uncommitted changes (dirty: {code.get('dirty')})")
    if code.get("sim_revision") != expect["sim_revision"]:
        why.append(f"simulator revision {code.get('sim_revision')} is not {expect['sim_revision']}")
    if code.get("sim_dirty") is not False:
        why.append(f"made with uncommitted simulator changes (sim_dirty: {code.get('sim_dirty')})")
    if code.get("changed_during_run") is not False:
        why.append(f"the code or the simulator changed during the run (changed_during_run: {code.get('changed_during_run')})")
    if run.get("protocol") != expect["protocol"]:
        why.append(f"protocol {run.get('protocol')} is not {expect['protocol']}")
    try:
        care, A, s = cell(run)
    except (KeyError, TypeError, ValueError) as exc:
        return why + [f"no cell: {exc}"]
    for name, got, want in (("care", care, expect["care"]), ("abstain cost", A, expect["abstain"]), ("probe time scale", s, expect["scale"])):
        if not _num(got, want):
            why.append(f"{name} {got} is not {want}")
    if int(run.get("seed", -1)) != int(expect["seed"]):
        why.append(f"seed {run.get('seed')} is not {expect['seed']}")
    n = len(run["rows"])
    if n != int(expect["n"]) or int(run.get("n", -1)) != n:
        why.append(f"{n} cases (n {run.get('n')}), not {expect['n']}")
    if expect.get("cases") is not None:
        got = {_key(r) for r in run["rows"]}
        if got != expect["cases"]:
            why.append(f"its cases are not the cases file's ({len(got - expect['cases'])} others, {len(expect['cases'] - got)} missing)")
    return why


# ---------------------------------------------------------------- case lists and gates

def probe_cases(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """The cases file of a subset never-probe run: the union over ``runs`` (one seed) of the cases whose first
    recommendation is probe, artifact cases included (the sensitivity row counts them)."""
    seeds = {int(r["seed"]) for r in runs}
    if len(seeds) != 1:
        raise ValueError(f"the runs span the seeds {sorted(seeds)}; a cases file is for one seed")
    cases: dict[str, set[int]] = defaultdict(set)
    for run in runs:
        for row in run["rows"]:
            if row.get("recommendation") == "probe":
                cases[row["object"]].add(int(row["idx"]))
    return {"seed": seeds.pop(), "cases": {o: sorted(v) for o, v in sorted(cases.items())}}


def _signature(row: dict[str, Any]) -> tuple:
    return (row.get("ended"), "error" in row, bool(row.get("artifact")),
            [(t.get("option"), t.get("outcome")) for t in row.get("attempts") or []],
            [(p.get("probe"), p.get("kind"), bool(p.get("thrown"))) for p in row.get("probes") or []],
            None if row.get("cost") is None else round(float(row["cost"]), 9))


def check_identical(policy: dict[str, Any], never: dict[str, Any]) -> list[tuple[str, int]]:
    """G1: the cases of ``policy`` whose first recommendation is not a probe and whose never-probe row differs (how it
    ended, each attempt's option and outcome, the cost) or is missing."""
    nv = {_key(r): r for r in never["rows"]}
    return sorted(_key(r) for r in policy["rows"]
                  if r.get("recommendation") != "probe" and (_key(r) not in nv or _signature(r) != _signature(nv[_key(r)])))


def check_same(a: dict[str, Any], b: dict[str, Any]) -> list[tuple[str, int]]:
    """G1b: the cases whose rows differ between two runs, or that one run lacks."""
    ra, rb = {_key(r): r for r in a["rows"]}, {_key(r): r for r in b["rows"]}
    return sorted(k for k in set(ra) | set(rb) if k not in ra or k not in rb or _signature(ra[k]) != _signature(rb[k]))


def _declared_risk(p: dict[str, Any]) -> float:
    d = p.get("declared") or {}
    thrown, damaged = float(d.get("p_thrown") or 0.0), float((d.get("p_kind") or {}).get("damaged", 0.0))
    return thrown + (1.0 - thrown) * damaged if p.get("kinds_not_thrown", True) else thrown + damaged


def _event(p: dict[str, Any]) -> bool:
    return bool(p.get("thrown")) or p.get("kind") == "damaged" or bool(p.get("damage"))


def probe_risk(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """G2: per probe position ("first", "second", "after-attempt") and type, the distinct probes (one per seed,
    object, case, position and probe: the same probe of a case is the same simulation in every cell) without an
    artifact flag: their number, the declared sum of P(thrown or damaged), the realized events, and the one-sided
    Poisson p-value of at least that many events (pass at P_GATE or more)."""
    seen: dict[tuple, dict[str, Any]] = {}
    for run in runs:
        for row in run["rows"]:
            for p in row.get("probes") or []:
                if not p.get("artifact"):
                    seen.setdefault((int(run["seed"]), row["object"], int(row["idx"]), int(p.get("position", 0)), p["probe"]), p)
    out: dict[str, dict[str, Any]] = {}
    for (_, _, _, pos, probe), p in sorted(seen.items()):
        where = "after-attempt" if int(p.get("after_attempts", 0)) > 0 else ("first" if pos == 0 else "second")
        g = out.setdefault(f"{where} {probe}", {"n": 0, "declared": 0.0, "events": 0})
        g["n"] += 1
        g["declared"] += _declared_risk(p)
        g["events"] += int(_event(p))
    for g in out.values():
        if g["events"] == 0:
            g["p_value"] = 1.0
        else:
            g["p_value"] = float(poisson.sf(g["events"] - 1, g["declared"])) if g["declared"] > 0 else 0.0
        g["verdict"] = "pass" if g["p_value"] >= P_GATE else "fail"
    return out


# ---------------------------------------------------------------- paired savings and the joint bootstrap

def declared_saving(first_menu: dict[str, Any] | None, A: float | None) -> float:
    """min(c, A) − the recommended probe row's expected cost at the first menu (c: the best row's); 0 without both."""
    if not first_menu or first_menu.get("best_cost") is None or first_menu.get("recommended_cost") is None:
        return 0.0
    c = float(first_menu["best_cost"])
    return (c if A is None else min(c, float(A))) - float(first_menu["recommended_cost"])


def pairs(a: dict[str, Any], b: dict[str, Any], subset: bool = False) -> dict[str, Any]:
    """The cases of run ``a`` paired with run ``b`` (same seed, care, abstain cost and force noise): arrays ``object``,
    ``idx``, ``probe`` (a's first recommendation is a probe), ``cost_a``, ``cost_b``, ``d`` = cost_b − cost_a,
    ``declared`` (a's declared saving on probe cases, else 0), ``counted`` (no error, no artifact in either row) and
    ``no_error``; ``n`` counted pairs, ``dropped`` the others. ``b`` at another probe time scale must have run no probe.
    With ``subset``, a case of ``a`` without a probe recommendation and without a row in ``b`` pairs with itself; a
    probe case without a row is an error, never a saving of 0."""
    ca, cb = cell(a), cell(b)
    if not (_num(ca[0], cb[0]) and _num(ca[1], cb[1]) and _num(a.get("force_noise"), b.get("force_noise"))):
        raise ValueError(f"different costs: {a['protocol']} at care {ca[0]}, A {ca[1]}, force noise {a.get('force_noise')} against "
                         f"{b['protocol']} at care {cb[0]}, A {cb[1]}, force noise {b.get('force_noise')}")
    if int(a["seed"]) != int(b["seed"]):
        raise ValueError(f"different seeds: {a['seed']} and {b['seed']}")
    if not _num(ca[2], cb[2]) and any(r.get("probes") for r in b["rows"]):
        raise ValueError(f"{b['protocol']} at s = {cb[2]} ran a probe, so it cannot pair with {a['protocol']} at s = {ca[2]}")
    rb = {_key(r): r for r in b["rows"]}
    out: dict[str, list] = {k: [] for k in ("object", "idx", "probe", "cost_a", "cost_b", "declared", "counted", "no_error")}
    for r in a["rows"]:
        k, probe = _key(r), r.get("recommendation") == "probe"
        if k in rb:
            s = rb[k]
        elif subset and not probe:
            s = r
        else:
            raise ValueError(f"case {k} of {a['protocol']} has no row in {b['protocol']}" + (" (a probe case)" if probe else ""))
        out["object"].append(k[0])
        out["idx"].append(k[1])
        out["probe"].append(probe)
        out["cost_a"].append(float(r.get("cost", math.nan)))
        out["cost_b"].append(float(s.get("cost", math.nan)))
        out["declared"].append(declared_saving(r.get("first_menu"), ca[1]) if probe else 0.0)
        out["counted"].append(_ok(r) and _ok(s))
        out["no_error"].append("error" not in r and "error" not in s)
    p: dict[str, Any] = {k: np.asarray(v, dtype=object if k == "object" else None) for k, v in out.items()}
    p["object"] = np.asarray(out["object"], dtype=str)
    p["counted"], p["no_error"], p["probe"] = (np.asarray(out[k], dtype=bool) for k in ("counted", "no_error", "probe"))
    p["d"] = p["cost_b"] - p["cost_a"]
    p["seed"], p["n"] = int(a["seed"]), int(p["counted"].sum())
    p["dropped"] = len(out["object"]) - p["n"]
    return p


def object_sums(p: dict[str, Any], objects: list[str], mask: np.ndarray | None = None) -> np.ndarray:
    """Per object (rows in the order of ``objects``): the sums of d, of the declared saving and of cost(b), and the
    number of pairs, over the counted pairs (or ``mask``)."""
    keep = p["counted"] if mask is None else np.asarray(mask, dtype=bool)
    index = {o: i for i, o in enumerate(objects)}
    S = np.zeros((len(objects), 4))
    for o, d, dh, cb, k in zip(p["object"], p["d"], p["declared"], p["cost_b"], keep):
        if k:
            S[index[str(o)]] += (d, dh, cb, 1.0)
    return S


def bootstrap_weights(objects: list[str], n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """The joint object bootstrap: n_boot rows of each object's multiplicity (one draw for every seed and cell)."""
    return object_weights(len(objects), n_boot, rng)


def _ci(x: np.ndarray) -> list[float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [math.nan, math.nan]


def cell_draws(S: np.ndarray, W: np.ndarray) -> dict[str, Any]:
    """Δ (the mean saving), Δ̂ (the mean declared saving), J (the mean cost of b) and Δ/J from per-object sums: point
    values, intervals, and the per-draw arrays (``draws``) for pooled statistics."""
    n = S[:, 3]
    N = float(n.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        pt = S.sum(axis=0) / N if N else np.full(4, math.nan)
        WN = W @ n
        draws = {"delta": (W @ S[:, 0]) / WN, "declared": (W @ S[:, 1]) / WN, "J": (W @ S[:, 2]) / WN}
        draws["rel"] = draws["delta"] / draws["J"]
    delta, declared, J = float(pt[0]), float(pt[1]), float(pt[2])
    return {"n": int(N), "delta": delta, "declared": declared, "J": J, "rel": delta / J if J else math.nan,
            "delta_ci": _ci(draws["delta"]), "declared_ci": _ci(draws["declared"]), "J_ci": _ci(draws["J"]), "rel_ci": _ci(draws["rel"]),
            "draws": draws}


def verdict(ci: list[float], margin: float = 0.0) -> str:
    """pass when the interval lies above ``margin``, fail when below, otherwise undecided."""
    if ci[0] > margin:
        return "pass"
    if ci[1] < margin:
        return "fail"
    return "undecided"


def h1_pays(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """H1-pays: R = the mean over the test cells of Δ/J, its interval (R recomputed in every joint draw) and verdict."""
    R = float(np.mean([c["rel"] for c in cells]))
    draws = np.mean([c["draws"]["rel"] for c in cells], axis=0)
    ci = _ci(draws)
    return {"R": R, "ci": ci, "verdict": verdict(ci), "n_cells": len(cells), "draws": draws}


def h1_only(cells: dict[tuple[float, float, float], dict[str, Any]], m: float = MARGIN) -> dict[str, Any]:
    """H1-only over the CHEAP cells: (a) a probe rate of at most PROBE_RATE_MAX, (b) the policy's Δ/J and (c) the forced
    probe's D_f/J with upper bounds below ``m``. Pass when all hold in every cheap cell; fail when a lower bound of (b)
    or (c) lies above ``m`` in any; otherwise (a cell without its runs too) undecided."""
    rows, ok, bad = {}, True, False
    for key in CHEAP:
        c = cells.get(key)
        if c is None:
            rows[_label(key)] = {"missing": True}
            ok = False
            continue
        a, b, f = c["probe_rate"] <= PROBE_RATE_MAX, c["policy"]["rel_ci"][1] < m, c["forced"]["rel_ci"][1] < m
        bad = bad or c["policy"]["rel_ci"][0] > m or c["forced"]["rel_ci"][0] > m
        ok = ok and a and b and f
        rows[_label(key)] = {"probe_rate": c["probe_rate"], "a": a, "b": b, "c": f, "policy_rel": c["policy"]["rel"],
                             "policy_rel_ci": c["policy"]["rel_ci"], "forced_rel": c["forced"]["rel"], "forced_rel_ci": c["forced"]["rel_ci"]}
    return {"verdict": "fail" if bad else ("pass" if ok else "undecided"), "margin": m, "cells": rows}


# ---------------------------------------------------------------- power (decision 7)

def variance_split(u: np.ndarray, objects: np.ndarray) -> tuple[float, float]:
    """(σ_b², σ_w²): a one-way analysis of variance of the per-case values ``u`` by object."""
    u, objects = np.asarray(u, dtype=float), np.asarray(objects)
    groups = [u[objects == o] for o in np.unique(objects)]
    N, K = len(u), len(groups)
    grand = float(u.mean())
    ssw = sum(float(((g - g.mean()) ** 2).sum()) for g in groups)
    ssb = sum(len(g) * (float(g.mean()) - grand) ** 2 for g in groups)
    sw = ssw / (N - K) if N > K else 0.0
    msb = ssb / (K - 1) if K > 1 else 0.0
    return float(max(0.0, (msb - sw) / (N / K))), float(sw)


def seeds_needed(effect: float, sigma_b2: float, sigma_w2: float, n_objects: int = 95, per_seed: float = 11,
                 choices: tuple[int, ...] = SEED_CHOICES) -> int:
    """Decision 7: the smallest k with 1.96 SE(k) ≤ 0.7 effect (80 % power), SE(k)² = (σ_b² + σ_w² / (per_seed (1 + k))) /
    n_objects; the largest choice when none qualifies or the effect is not above 0."""
    if not effect > 0:
        return choices[-1]
    for k in choices:
        if 1.96 * math.sqrt((sigma_b2 + sigma_w2 / (per_seed * (1 + k))) / n_objects) <= 0.7 * effect:
            return k
    return choices[-1]


def power(cell_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Decision 7 on the pilot seed: R̂ (realized) and R̂_d (declared) over the test cells, the variance split of the
    per-case contributions u = the mean over cells of d/J, the projected SE(k) and the number k of stage-2 seeds."""
    rel, rel_d, per_case = [], [], defaultdict(list)
    for p in cell_pairs:
        m = p["counted"]
        J = float(p["cost_b"][m].mean())
        rel.append(float(p["d"][m].mean()) / J)
        rel_d.append(float(p["declared"][m].mean()) / J)
        for o, i, d in zip(p["object"][m], p["idx"][m], p["d"][m]):
            per_case[(str(o), int(i))].append(float(d) / J)
    keys = sorted(per_case)
    u, objs = np.array([np.mean(per_case[k]) for k in keys]), np.array([k[0] for k in keys])
    sb, sw = variance_split(u, objs)
    n_obj = len(set(objs.tolist()))
    per_seed = len(keys) / n_obj
    R, Rd = float(np.mean(rel)), float(np.mean(rel_d))
    effect = min(R, Rd)
    se = {str(k): math.sqrt((sb + sw / (per_seed * (1 + k))) / n_obj) for k in SEED_CHOICES}
    return {"R": R, "R_declared": Rd, "effect": effect, "sigma_b2": sb, "sigma_w2": sw, "n_objects": n_obj, "cases_per_object": per_seed,
            "n_cells": len(cell_pairs), "se": se, "k": seeds_needed(effect, sb, sw, n_obj, per_seed)}


# ---------------------------------------------------------------- descriptive tables

def _irreversible(row: dict[str, Any]) -> bool:
    return any(t.get("outcome") in IRREVERSIBLE for t in row.get("attempts") or []) or any(_event(p) for p in row.get("probes") or [])


def arm_metrics(run: dict[str, Any], W: np.ndarray, objects: list[str]) -> dict[str, Any]:
    """One run's counted cases: the rates of a first probe recommendation, of a probe run, of abstaining, the placed and
    irreversible cases per 1,000 (an attempt with an irreversible outcome, or a probe that threw or damaged the object),
    the realized cost, all with joint-bootstrap intervals, the median declared EVOI of the first probes, and the
    excluded cases."""
    rows = _counted(run["rows"])
    index = {o: i for i, o in enumerate(objects)}
    cols = ("probe", "probed", "abstained", "placed", "irreversible", "cost")
    S = np.zeros((len(objects), len(cols) + 1))
    for r in rows:
        S[index[r["object"]]] += (r.get("recommendation") == "probe", bool(r.get("probes")), r.get("ended") == "abstained",
                                  r.get("ended") == "placed", _irreversible(r), float(r["cost"]), 1.0)
    n = len(rows)
    with np.errstate(invalid="ignore", divide="ignore"):
        WN = W @ S[:, -1]
        draws = {c: (W @ S[:, j]) / WN for j, c in enumerate(cols)}
    pt = S.sum(axis=0) / max(n, 1)
    evoi = [float(p["declared"]["evoi"]) for r in rows for p in r.get("probes") or []
            if int(p.get("position", 0)) == 0 and (p.get("declared") or {}).get("evoi") is not None]
    out = {"n": n, "errors": sum("error" in r for r in run["rows"]),
           "artifacts": sum("error" not in r and bool(r.get("artifact")) for r in run["rows"])}
    for j, c in enumerate(cols):
        scale = 1000.0 if c in ("placed", "irreversible") else 1.0
        name = {"probe": "probe_rate", "probed": "probed_rate", "abstained": "abstain_rate", "placed": "placed_per_1000",
                "irreversible": "irreversible_per_1000", "cost": "cost"}[c]
        out[name] = float(pt[j]) * scale
        out[name + "_ci"] = [scale * x for x in _ci(draws[c])]
    out["median_evoi"] = float(np.median(evoi)) if evoi else None
    return out


def subgroup(case: dict[str, Any], classes: dict[str, dict[str, Any]]) -> tuple[str, str]:
    """(public class, prior-draw tercile): container (fillable), crushable, bruisable, breakable or rigid (class none);
    the draw's quantile within its own generator range (log-uniform on MASS_SCALE_RANGE, or FILLED_MASS_SCALE_RANGE for
    a filled container) in thirds: low, mid, high."""
    from robots.mujoco_lab.fragility import entry_of

    entry = entry_of(case["object"], classes)[0]
    cls = "container" if entry.get("fillable") else (entry.get("class") if entry.get("class") in ("crushable", "bruisable", "breakable")
                                                    else "rigid")
    lo, hi = FILLED_MASS_SCALE_RANGE if float(case.get("fill") or 0.0) > 0.0 else MASS_SCALE_RANGE
    q = math.log(float(case["theta"]["mass_scale"]) / lo) / math.log(hi / lo)
    return cls, "low" if q < 1 / 3 else ("mid" if q < 2 / 3 else "high")


def _neg(c: dict[str, Any]) -> dict[str, Any]:
    """A cell_draws result for the reversed difference (cost(a) − cost(b))."""
    return {"n": c["n"], "diff": -c["delta"], "diff_ci": [-c["delta_ci"][1], -c["delta_ci"][0]]}


def _strip(x: Any) -> Any:
    """JSON-ready: drops the per-draw arrays, turns numpy values and tuple keys into plain ones."""
    if isinstance(x, dict):
        return {(_label(k) if isinstance(k, tuple) else str(k)): _strip(v) for k, v in x.items() if k != "draws"}
    if isinstance(x, (list, tuple)):
        return [_strip(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_strip(v) for v in x.tolist()]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _index(runs: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    out: dict[tuple, dict[str, Any]] = {}
    for r in runs:
        k = (r["protocol"], *cell(r), int(r["seed"]))
        if k in out:
            raise ValueError(f"two runs of {k}")
        out[k] = r
    return out


def summary(runs: list[dict[str, Any]], classes: dict[str, dict[str, Any]], n_boot: int = N_BOOT, rng_seed: int = 0,
            power_result: dict[str, Any] | None = None) -> dict[str, Any]:
    """The milestone 4 tables and verdicts (plan decision 6) from every run of stages 1 and 2 (the pilot seed is left
    out), with ``markdown``."""
    runs = [r for r in runs if int(r["seed"]) != PILOT_SEED]
    idx = _index(runs)
    objects = sorted({r["object"] for run in runs for r in run["rows"]})
    W = bootstrap_weights(objects, n_boot, np.random.default_rng(rng_seed))
    seeds = sorted({k[4] for k in idx})

    def get(protocol, c, seed=BASE_SEED):
        return idx.get((protocol, c[0], c[1], c[2], seed))

    def never(c, seed=BASE_SEED):
        return idx.get(("policy_no_probe", c[0], c[1], 1.0, seed))

    cells, raw = {}, {}
    for c in ((k, A, s) for k in KAPPAS for A in ABSTAIN for s in SCALES):
        P, N = get("policy", c), never(c)
        if P is None:
            continue
        row: dict[str, Any] = {"policy": arm_metrics(P, W, objects)}
        if N is not None:
            raw[c] = cell_draws(object_sums(pairs(P, N, subset=_subset(N)), objects), W)
            row["saving"] = raw[c]
        C = get("commit", (c[0], c[1], 1.0))
        if C is not None:
            row["policy_minus_commit"] = _neg(cell_draws(object_sums(pairs(P, C), objects), W))
        F = get("policy_probe_one", c) if c[2] == 1.0 else None
        if F is not None and N is not None:
            row["forced"] = cell_draws(object_sums(pairs(F, N), objects), W)
            row["choose"] = cell_draws(object_sums(pairs(P, F), objects), W)     # cost(probe_one) − cost(policy)
        cells[c] = row

    want = set(seeds)                     # every seed of a run, and every seed the power result asked for
    if power_result and power_result.get("k"):
        want |= set(range(BASE_SEED, BASE_SEED + 1 + int(power_result["k"])))
    test, test_art, test_pairs, missing = [], [], [], set()
    for c in TEST_CELLS:
        S, S_art, used = np.zeros((len(objects), 4)), np.zeros((len(objects), 4)), []
        for seed in sorted(want):
            P, N = get("policy", c, seed), never(c, seed)
            if P is None or N is None:
                missing.update(f"{p} {_label(x)} seed {seed}" for p, x, r in (("policy", c, P), ("policy_no_probe", (c[0], c[1], 1.0), N))
                               if r is None)
                continue
            p = pairs(P, N, subset=_subset(N))
            S += object_sums(p, objects)
            S_art += object_sums(p, objects, p["no_error"])
            test_pairs.append((c, P, p))
            used.append(seed)
        if used:
            test.append(dict(cell_draws(S, W), cell=_label(c), seeds=used))
            test_art.append(cell_draws(S_art, W))
    pays = h1_pays(test) if test else {"R": None, "ci": None, "verdict": "missing"}
    if test and missing:
        pays["verdict"] = "incomplete"
    pays["missing"] = sorted(missing)
    pays["cells"] = test
    pays["artifacts_counted"] = h1_pays(test_art) if test_art else None

    cheap = {}
    for c in CHEAP:
        row = cells.get(c) or {}
        if "saving" in row and "forced" in row:
            cheap[c] = {"probe_rate": row["policy"]["probe_rate"], "policy": row["saving"], "forced": row["forced"]}
    only = h1_only(cheap)

    if test:
        sd = np.sum([t["draws"]["delta"] for t in test], axis=0)
        sdh = np.sum([t["draws"]["declared"] for t in test], axis=0)
        value = {"sum_delta": float(sum(t["delta"] for t in test)), "sum_delta_ci": _ci(sd),
                 "sum_declared": float(sum(t["declared"] for t in test)), "sum_declared_ci": _ci(sdh),
                 "sum_delta_minus_declared": float(sum(t["delta"] - t["declared"] for t in test)), "sum_delta_minus_declared_ci": _ci(sd - sdh)}
    else:
        value = None

    contrasts = {}
    for A in ABSTAIN:
        for s in SCALES:
            hi, lo = raw.get((30.0, A, s)), raw.get((0.5, A, s))
            if hi is not None and lo is not None:
                contrasts[f"A{A:g}_s{s:g}"] = {"rel_k30_minus_k0.5": hi["rel"] - lo["rel"], "ci": _ci(hi["draws"]["rel"] - lo["draws"]["rel"])}

    crossovers = {}
    for A in ABSTAIN:
        for s in SCALES:
            got = [(k, cells.get((k, A, s))) for k in KAPPAS]
            got = [(k, r) for k, r in got if r is not None and "saving" in r]
            rate = next((k for k, r in got if r["policy"]["probe_rate"] >= CROSS_RATE), None)
            sav = next(((k, r["saving"]["rel_ci"]) for k, r in got if r["saving"]["rel"] >= CROSS_REL), None)
            sig = next((k for k, r in got if r["saving"]["delta_ci"][0] > 0), None)
            crossovers[f"A{A:g}_s{s:g}"] = {"probe_rate": rate if rate is not None else "none",
                                            "saving": sav[0] if sav else "none", "saving_rel_ci": sav[1] if sav else None,
                                            "significance": sig if sig is not None else "none"}

    groups: dict[str, dict[str, np.ndarray]] = {"class": {}, "tercile": {}}
    for c, P, p in test_pairs:
        by_key = {_key(r): r for r in P["rows"]}
        rows = [by_key[(str(o), int(i))] for o, i in zip(p["object"], p["idx"])]
        labels = [subgroup(r, classes) if "theta" in r else None for r in rows]      # an error row has no draw, and never counts
        for j, kind in enumerate(("class", "tercile")):
            for g in sorted({lab[j] for lab in labels if lab is not None}):
                mask = p["counted"] & np.array([lab is not None and lab[j] == g for lab in labels], dtype=bool)
                groups[kind][g] = groups[kind].get(g, np.zeros((len(objects), 4))) + object_sums(p, objects, mask)
    subgroups = {kind: {g: cell_draws(S, W) for g, S in sorted(d.items())} for kind, d in groups.items()}

    exclusions = {}
    for k, run in sorted(idx.items(), key=lambda kv: str(kv[0])):
        tally: dict[str, list[int]] = {"probe": [0, 0], "other": [0, 0], "unknown": [0, 0]}
        for r in run["rows"]:
            rec = r.get("recommendation")       # None: the case raised before its first menu
            t = tally["unknown" if rec is None else "probe" if rec == "probe" else "other"]
            t[0] += "error" in r
            t[1] += "error" not in r and bool(r.get("artifact"))
        exclusions[f"{k[0]}_{_label(k[1:4])}_seed{k[4]}"] = {g: {"errors": e, "artifacts": a} for g, (e, a) in tally.items()}

    diagnostic = {}
    for c in DIAGNOSTIC:
        diagnostic[_label(c)] = {p: arm_metrics(get(p, c), W, objects) for p in ("policy", "policy_no_probe", "commit", "repair",
                                                                                  "probe_all", "random_row") if get(p, c) is not None}

    out = {"seeds": seeds, "n_objects": len(objects), "n_boot": n_boot, "h1_pays": pays, "h1_only": only, "probe_value": value,
           "cells": cells, "kappa_contrasts": contrasts, "crossovers": crossovers, "subgroups": subgroups, "exclusions": exclusions,
           "diagnostic": diagnostic, "power": power_result}
    out = _strip(out)
    out["markdown"] = markdown(out)
    return out


def _f(x: Any, nd: int = 3) -> str:
    return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _ci_s(ci: Any, nd: int = 3) -> str:
    return "–" if not ci or ci[0] is None else f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def markdown(s: dict[str, Any]) -> str:
    """The summary as Markdown tables."""
    L = [f"# Milestone 4 sweep (seeds {s['seeds']}, {s['n_objects']} objects, {s['n_boot']} joint bootstrap draws)", ""]
    p = s["h1_pays"]
    L += ["## Verdicts", "", f"- H1-pays: R = {_f(p.get('R'))} {_ci_s(p.get('ci'))} → **{p['verdict']}**"]
    if p.get("missing"):
        L.append(f"  - missing runs ({len(p['missing'])}): {', '.join(p['missing'])}")
    if p.get("artifacts_counted"):
        L.append(f"  - artifacts counted: R = {_f(p['artifacts_counted']['R'])} {_ci_s(p['artifacts_counted']['ci'])}")
    L.append(f"- H1-only (margin {s['h1_only']['margin']}): **{s['h1_only']['verdict']}**")
    if s["probe_value"]:
        v = s["probe_value"]
        L.append(f"- Probe value check (one step): ΣΔ = {_f(v['sum_delta'])} {_ci_s(v['sum_delta_ci'])}, ΣΔ̂ = {_f(v['sum_declared'])}, "
                 f"Σ(Δ − Δ̂) = {_f(v['sum_delta_minus_declared'])} {_ci_s(v['sum_delta_minus_declared_ci'])}")
    L += ["", "## H1-pays cells", "", "| cell | seeds | n | Δ | Δ CI | J | Δ/J | Δ/J CI | Δ̂ |", "|---|---|---|---|---|---|---|---|---|"]
    for t in p["cells"]:
        L.append(f"| {t['cell']} | {t['seeds']} | {t['n']} | {_f(t['delta'])} | {_ci_s(t['delta_ci'])} | {_f(t['J'])} | {_f(t['rel'])} | "
                 f"{_ci_s(t['rel_ci'])} | {_f(t['declared'])} |")
    L += ["", "## H1-only cells", "", "| cell | probe rate | (a) | policy Δ/J | CI | (b) | forced D_f/J | CI | (c) |", "|---|---|---|---|---|---|---|---|---|"]
    for k, c in s["h1_only"]["cells"].items():
        if c.get("missing"):
            L.append(f"| {k} | missing | | | | | | | |")
        else:
            L.append(f"| {k} | {_f(c['probe_rate'])} | {c['a']} | {_f(c['policy_rel'])} | {_ci_s(c['policy_rel_ci'])} | {c['b']} | "
                     f"{_f(c['forced_rel'])} | {_ci_s(c['forced_rel_ci'])} | {c['c']} |")
    L += ["", f"## All cells (seed {BASE_SEED})", "",
          "| cell | n | probe | abstain | placed/1000 | irrev/1000 | cost | Δ | Δ CI | Δ/J | policy − commit | D_f/J | choose |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, c in s["cells"].items():
        m, sv = c["policy"], c.get("saving") or {}
        pc, fo, ch = c.get("policy_minus_commit") or {}, c.get("forced") or {}, c.get("choose") or {}
        L.append(f"| {k} | {m['n']} | {_f(m['probe_rate'])} | {_f(m['abstain_rate'])} | {_f(m['placed_per_1000'], 0)} | "
                 f"{_f(m['irreversible_per_1000'], 1)} | {_f(m['cost'])} | {_f(sv.get('delta'))} | {_ci_s(sv.get('delta_ci'))} | "
                 f"{_f(sv.get('rel'))} | {_f(pc.get('diff'))} | {_f(fo.get('rel'))} | {_f(ch.get('delta'))} |")
    L += ["", "## Crossovers (seed 12)", "", "| A, s | probe rate ≥ 5 % | Δ/J ≥ 5 % | its CI | Δ CI above 0 |", "|---|---|---|---|---|"]
    for k, c in s["crossovers"].items():
        L.append(f"| {k} | {c['probe_rate']} | {c['saving']} | {_ci_s(c['saving_rel_ci'])} | {c['significance']} |")
    L += ["", "## κ contrasts (Δ/J at κ 30 minus κ 0.5, seed 12)", "", "| A, s | contrast | CI |", "|---|---|---|"]
    for k, c in s["kappa_contrasts"].items():
        L.append(f"| {k} | {_f(c['rel_k30_minus_k0.5'])} | {_ci_s(c['ci'])} |")
    L += ["", "## Subgroups (test cells, all seeds)", "", "| group | n | Δ | Δ CI | J | Δ/J |", "|---|---|---|---|---|---|"]
    for kind, d in s["subgroups"].items():
        for g, c in d.items():
            L.append(f"| {kind}: {g} | {c['n']} | {_f(c['delta'])} | {_ci_s(c['delta_ci'])} | {_f(c['J'])} | {_f(c['rel'])} |")
    L += ["", "## Diagnostic protocols", "", "| cell | protocol | n | probe | abstain | placed/1000 | irrev/1000 | cost | cost CI |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, d in s["diagnostic"].items():
        for proto, m in d.items():
            L.append(f"| {k} | {proto} | {m['n']} | {_f(m['probe_rate'])} | {_f(m['abstain_rate'])} | {_f(m['placed_per_1000'], 0)} | "
                     f"{_f(m['irreversible_per_1000'], 1)} | {_f(m['cost'])} | {_ci_s(m['cost_ci'])} |")
    if s.get("power"):
        pw = s["power"]
        L += ["", "## Power (stage 0, seed 11)", "", f"R̂ = {_f(pw['R'])}, R̂_d = {_f(pw['R_declared'])}, σ_b² = {_f(pw['sigma_b2'], 5)}, "
              f"σ_w² = {_f(pw['sigma_w2'], 5)}, SE(k) = {pw['se']}, k = {pw['k']}"]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- CLI

def _runs_of(path: str, seed: int) -> list[dict[str, Any]]:
    return [r for r in load_dir(path) if int(r["seed"]) == seed]


def _dump(obj: Any, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_strip(obj), f, indent=1)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("valid", help="exit 1 when an output cannot be reused")
    v.add_argument("file")
    v.add_argument("--revision", required=True)
    v.add_argument("--sim-revision", required=True, help="the simulator folder's last commit (an output's code.sim_revision)")
    v.add_argument("--protocol", required=True)
    for name in ("--care", "--abstain", "--scale"):
        v.add_argument(name, type=float, required=True)
    v.add_argument("--seed", type=int, required=True)
    v.add_argument("--n", type=int, default=None)
    v.add_argument("--cases", default=None, help="the cases file of a subset run (its case count is the expected n)")
    for name in ("cases", "check", "risk", "power", "summary"):
        sp = sub.add_parser(name)
        sp.add_argument("--dir", required=True)
        if name != "summary":
            sp.add_argument("--seed", type=int, required=True)
        if name == "cases":
            sp.add_argument("--care", type=float, required=True)
            sp.add_argument("--abstain", type=float, required=True)
        if name != "check":
            sp.add_argument("--out", required=True)
        if name == "summary":
            sp.add_argument("--markdown", required=True)
            sp.add_argument("--power", default=None, help="the stage-0 power output")
            sp.add_argument("--manifest", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "m2b2", "manifest.json"))
            sp.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)

    if a.cmd == "valid":
        if (a.n is None) == (a.cases is None):
            ap.error("give --n or --cases")
        listed = json.load(open(a.cases))["cases"] if a.cases else None
        n = a.n if a.n is not None else sum(len(v) for v in listed.values())
        why = valid(a.file, {"revision": a.revision, "sim_revision": a.sim_revision, "protocol": a.protocol, "care": a.care,
                             "abstain": a.abstain, "scale": a.scale, "seed": a.seed, "n": n,
                             "cases": None if listed is None else {(o, int(i)) for o, v in listed.items() for i in v}})
        print("valid" if not why else "not valid: " + "; ".join(why))
        return 0 if not why else 1
    if a.cmd == "cases":
        runs = [r for r in _runs_of(a.dir, a.seed) if r["protocol"] == "policy" and _num(cell(r)[0], a.care) and _num(cell(r)[1], a.abstain)]
        scales = sorted(cell(r)[2] for r in runs)
        if scales != sorted(SCALES):
            print(f"cases: the policy runs at care {a.care}, A {a.abstain}, seed {a.seed} have the scales {scales}, not {sorted(SCALES)}")
            return 1
        spec = probe_cases(runs)
        _dump(spec, a.out)
        print(f"cases: {sum(len(v) for v in spec['cases'].values())} probe cases over {len(spec['cases'])} objects -> {a.out}")
        return 0
    if a.cmd == "check":
        runs = _index(_runs_of(a.dir, a.seed))
        out: dict[str, Any] = {"G1": {}, "G1b": {}, "missing": [f"{p} {_label(c)}" for p, c in GATE_RUNS if (p, *c, a.seed) not in runs]}
        for k, run in sorted(runs.items(), key=lambda kv: str(kv[0])):
            if k[0] == "policy":
                nv = runs.get(("policy_no_probe", k[1], k[2], 1.0, k[4]))
                if nv is None:
                    out["missing"].append(f"policy_no_probe for {_label(k[1:4])}")
                    continue
                bad = check_identical(run, nv)
                out["G1"][_label(k[1:4])] = {"checked": sum(r.get("recommendation") != "probe" for r in run["rows"]), "mismatches": bad}
            elif k[0] == "policy_no_probe" and k[3] != 1.0:
                ref = runs.get(("policy_no_probe", k[1], k[2], 1.0, k[4]))
                if ref is None:
                    out["missing"].append(f"policy_no_probe at s 1 for {_label(k[1:4])}")
                    continue
                out["G1b"][_label(k[1:4])] = {"checked": len(run["rows"]), "mismatches": check_same(ref, run)}
        n_bad = sum(len(g["mismatches"]) for part in ("G1", "G1b") for g in out[part].values()) + len(out["missing"])
        out["verdict"] = "pass" if n_bad == 0 else "stop"
        print(json.dumps(_strip(out), indent=1))
        return 0 if n_bad == 0 else 1
    if a.cmd == "risk":
        out = probe_risk(_runs_of(a.dir, a.seed))
        _dump(out, a.out)
        print(json.dumps(_strip(out), indent=1))
        return 0
    if a.cmd == "power":
        runs = _index(_runs_of(a.dir, a.seed))
        cps = []
        for c in TEST_CELLS:
            P, N = runs.get(("policy", *c, a.seed)), runs.get(("policy_no_probe", c[0], c[1], 1.0, a.seed))
            if P is None or N is None:
                print(f"power: no policy or never-probe run for {_label(c)}")
                return 1
            cps.append(pairs(P, N, subset=_subset(N)))
        out = power(cps)
        _dump(out, a.out)
        print(json.dumps(_strip(out), indent=1))
        return 0
    from robots.mujoco_lab.prior import fragility_world

    world = fragility_world(json.load(open(a.manifest)))
    classes = world[0] if world is not None else {}
    runs = load_dir(a.dir)
    pw = json.load(open(a.power)) if a.power else None
    out = summary(runs, classes, n_boot=a.n_boot, power_result=pw)
    with open(a.markdown, "w") as f:
        f.write(out["markdown"])
    md = out.pop("markdown")
    _dump(out, a.out)
    metrics = {"h1_pays_R": out["h1_pays"].get("R"), "h1_pays_verdict": out["h1_pays"]["verdict"], "h1_only_verdict": out["h1_only"]["verdict"]}
    if out.get("probe_value"):
        metrics.update(sum_delta=out["probe_value"]["sum_delta"], sum_declared=out["probe_value"]["sum_declared"])
    out["wandb"] = tracking.log_run("sweep", {"dir": a.dir, "n_runs": len(runs), "seeds": out["seeds"]}, metrics)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
