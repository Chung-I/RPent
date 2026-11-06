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


"""The success-damage curve (daily-logs researches/property-belief-manipulation/designs/2026-11-06-success-damage-curve-design.md,
approved by the user on 2026-11-07; the experiment's README is experiments/2026-11-07-success-damage-curve).

Stage 0 (``stage0_points``): a first look at the m4 sweep (bundle m2b2, seed 12, 1,045 cases per cell, s = 1), descriptive
only. Per (care kappa, abstain cost A) cell, the policy (it may probe, up to POLICY_ATTEMPTS attempts) and commit-only (one
attempt at the best row by J; R4 picks by P(placed) instead) as points: task success against irreversible damage, with
r = A / (5 kappa). Rates are over the counted cases (calibrate._counted: no error, no artifact), give-ups included:
  success       the case ended placed;
  irreversible  an attempt ended in an irreversible label, or a probe damaged the object;
  probe_damage  a probe damaged the object (part of irreversible);
  give_up       the case ended abstained, no_row or unreachable.

    python -P -m robots.mujoco_lab.curve stage0 --m4 DIR --out DIR
    python -P -m robots.mujoco_lab.curve split --bundle B --scenes S --out FILE [--seed 20261107]
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from rpent.skills.pick_place.costs import COST, TIME_COST_PER_S, expected_cost
from rpent.skills.pick_place.protocol import IRREVERSIBLE, OUTCOMES, RECOVERABLE, Prediction

M4_PROTOCOLS = ("commit", "policy")
GIVE_UP = ("abstained", "no_row", "unreachable")
SPLIT_SEED = 20261107
RULES = ("R0", "R1", "R2", "R3", "R4", "R5", "R6")
R_GRID = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)        # r = A / (5 kappa), at A = A_MAIN (design 4)
DELTA_GRID = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)          # the damage caps of R2 and R5
A_MAIN = 10.0
R_CHECK = ((3.0, 0.2), (3.0, 2.0))                               # the only-r-matters check: (A, r), decisions only
QUANTILE = 0.9                                                   # R3 and R5: the tail of the per-particle P(irr)
CVAR_LEVEL = 0.1                                                 # R6: the lowest share of the particle weight
ROW_ORDER = ("commit_std", "slow_lift", "firm_grip", "light_grip", "gentle_carry", "regrasp_com", "inboard_place",
             "drop_release")                                     # menu._rows' order: a tie goes to the earlier row
GIVE_UP_ROW = "give_up"                                          # a rule's decision not to grasp (not an m4 ended value)
PRIOR_KEY = "prior"                                              # evpi_props.PRIOR: the prior belief's key in its records
KNOBS = {"R0": R_GRID, "R1": R_GRID, "R2": DELTA_GRID, "R3": R_GRID, "R4": (None,), "R5": DELTA_GRID, "R6": R_GRID}
CHECK_RULES = ("R0", "R1", "R3", "R6")                           # the rules whose knob is r
CVAR_KEYS = tuple((A_MAIN, r) for r in R_GRID) + R_CHECK
STAGE1_CARE = 100.0               # the first menu's own (care, A) = experiment 1's k100_A10 (r 0.02): the replay check
NO_GRASP = ("decline", "unreachable", "no_row", "error")         # evpi_props.choice_of values that grasp nothing
ESS_APART = 100.0                 # records whose belief has a lower ESS are reported apart (design 5.3)
LOW_SUCCESS = 0.2                 # design 5.4: the share of selected rows with P(placed) below this
QS = (0.0, 0.25, 0.5, 0.75, 1.0)  # the selected rows' distribution: min, quartiles, max
M4_NAME = re.compile(r"^(?P<protocol>[a-z_]+)_k(?P<care>[0-9.]+)_A(?P<A>[0-9.]+)_s(?P<s>[0-9.]+)_seed(?P<seed>\d+)\.json\.gz$")


# ---------------------------------------------------------------- the object split (design 5.1)

def stratum(o: str, entry: dict[str, Any]) -> tuple[str, str]:
    """(source, damageable or rigid): the source is the id's prefix (ycb, hope, hot3d, handal, fruits_veggies); an object
    can be damaged when its fragility class is not none or it is fillable (it can spill)."""
    dmg = entry.get("class", "none") != "none" or bool(entry.get("fillable"))
    return o.split("__", 1)[0], "damageable" if dmg else "rigid"


def make_split(objects: list[str], entries: dict[str, dict[str, Any]], seed: int = SPLIT_SEED) -> dict[str, Any]:
    """Half of the objects are validation, half test, per stratum (``stratum``). Strata in sorted order k = 0, 1, ...: the
    stratum's objects sorted by id, permuted by default_rng([seed, k]); the first half is validation; an odd stratum's
    extra object goes to validation, then test, alternately over the odd strata in that order."""
    import numpy as np

    by: dict[tuple[str, str], list[str]] = {}
    for o in sorted(set(objects)):
        by.setdefault(stratum(o, entries[o]), []).append(o)
    out: dict[str, Any] = {"seed": int(seed), "validation": [], "test": [], "strata": {}}
    n_odd = 0
    for k, key in enumerate(sorted(by)):
        objs = by[key]
        perm = [objs[int(i)] for i in np.random.default_rng([int(seed), k]).permutation(len(objs))]
        n_val = len(objs) // 2
        if len(objs) % 2:
            n_val += 1 if n_odd % 2 == 0 else 0
            n_odd += 1
        val, test = sorted(perm[:n_val]), sorted(perm[n_val:])
        out["strata"]["|".join(key)] = {"validation": val, "test": test}
        out["validation"] += val
        out["test"] += test
    out["validation"], out["test"] = sorted(out["validation"]), sorted(out["test"])
    return out


# ---------------------------------------------------------------- Stage 0: the m4 sweep's points

def m4_point(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Success, irreversible damage, probe damage, give-up and second-attempt rates over the counted cases (see the
    module docstring)."""
    rs = [r for r in rows if "error" not in r and not r.get("artifact")]
    n = len(rs)

    def probe_dmg(r: dict[str, Any]) -> bool:
        return any(p.get("kind") == "damaged" or p.get("damage") for p in r.get("probes") or [])

    def irr(r: dict[str, Any]) -> bool:
        return any(a.get("label") in IRREVERSIBLE for a in r.get("attempts") or []) or probe_dmg(r)

    rate = lambda k: (k / n) if n else None  # noqa: E731
    return {"n": n, "success": rate(sum(r["ended"] == "placed" for r in rs)), "irreversible": rate(sum(irr(r) for r in rs)),
            "probe_damage": rate(sum(probe_dmg(r) for r in rs)), "give_up": rate(sum(r["ended"] in GIVE_UP for r in rs)),
            "second_attempt": rate(sum(len(r.get("attempts") or []) > 1 for r in rs))}


def stage0_points(m4_dir: str, s: float = 1.0, seed: int = 12) -> list[dict[str, Any]]:
    """One point per M4_PROTOCOLS file at time scale ``s`` and ``seed``, sorted by (protocol, kappa, A)."""
    pts = []
    for name in sorted(os.listdir(m4_dir)):
        m = M4_NAME.match(name)
        if not m or m["protocol"] not in M4_PROTOCOLS or float(m["s"]) != float(s) or int(m["seed"]) != int(seed):
            continue
        with gzip.open(os.path.join(m4_dir, name), "rt") as f:
            rows = json.load(f)["rows"]
        care, A = float(m["care"]), float(m["A"])
        pts.append(dict(m4_point(rows), protocol=m["protocol"], care=care, A=A, r=A / (5.0 * care), file=name))
    return sorted(pts, key=lambda p: (p["protocol"], p["care"], p["A"]))


def stage0_markdown(pts: list[dict[str, Any]]) -> str:
    pct = lambda x: "-" if x is None else f"{100.0 * x:.1f}"  # noqa: E731
    by = {(p["protocol"], p["care"], p["A"]): p for p in pts}
    cells = sorted({(p["care"], p["A"]) for p in pts}, key=lambda c: (c[1] / (5.0 * c[0]), c))
    out = ["| kappa | A | r | policy success % | policy irreversible % (probe) | policy give-up % | commit success % | "
           "commit irreversible % | commit give-up % | n (policy, commit) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for care, A in cells:
        p, c = by.get(("policy", care, A)), by.get(("commit", care, A))
        f = lambda x, k: "-" if x is None else pct(x[k])  # noqa: E731
        out.append(f"| {care:g} | {A:g} | {A / (5.0 * care):.3g} | {f(p, 'success')} | {f(p, 'irreversible')} "
                   f"({f(p, 'probe_damage')}) | {f(p, 'give_up')} | {f(c, 'success')} | {f(c, 'irreversible')} | {f(c, 'give_up')} | "
                   f"{p['n'] if p else '-'}, {c['n'] if c else '-'} |")
    return "\n".join(out) + "\n"


def _label_groups(pts: list[tuple[float, float, float]], dx: float = 0.15, dy: float = 0.6) -> list[tuple[tuple[float, float], list[float]]]:
    """Points (x, y, r) within (dx, dy) of each other (single linkage) share one label at the group's highest point, its
    r values in increasing order: overlapping labels are unreadable."""
    groups: list[list[tuple[float, float, float]]] = []
    for p in sorted(pts):
        near = [g for g in groups if any(abs(p[0] - q[0]) <= dx and abs(p[1] - q[1]) <= dy for q in g)]
        merged = [p] + [q for g in near for q in g]
        groups = [g for g in groups if g not in near] + [merged]
    out = []
    for g in groups:
        top = max(g, key=lambda q: (q[1], q[0]))
        out.append(((top[0], top[1]), sorted(q[2] for q in g)))
    return sorted(out)


def stage0_figure(pts: list[dict[str, Any]], path: str) -> None:
    """Success against irreversible damage: the policy (blue, labelled with r) and commit-only (orange)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"policy": "#2a78d6", "commit": "#eb6834"}
    names = {"policy": "policy (may probe, up to 3 attempts)", "commit": "commit-only (best row by J, 1 attempt)"}
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for proto in ("commit", "policy"):
        ps = [p for p in pts if p["protocol"] == proto]
        ax.scatter([100 * p["irreversible"] for p in ps], [100 * p["success"] for p in ps], s=36, color=colours[proto],
                   edgecolors="white", linewidths=1.5, zorder=3, label=names[proto])
        if proto == "policy":
            groups = sorted(_label_groups([(100 * p["irreversible"], 100 * p["success"], p["r"]) for p in ps]), key=lambda g: -g[0][1])
            y_lab = 1e9
            for xy, rs in groups:              # one label column in the empty middle, top down, 1.8 points apart, leader lines
                y_lab = min(xy[1], y_lab - 1.8)
                ax.annotate("r " + ", ".join(f"{r:.3g}" for r in rs), xy, xytext=(3.4, y_lab), textcoords="data", fontsize=7,
                            color="#52514e", va="center", arrowprops={"arrowstyle": "-", "color": "#c3c2b7", "lw": 0.6})
    ax.set_xlabel("irreversible damage (% of cases)")
    ax.set_ylabel("task success (% of cases)")
    ax.set_title("m4 sweep (m2b2, s = 1): where the tested settings sit", fontsize=10)
    ax.grid(True, color="#e6e5e0", linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def stage0(a: argparse.Namespace) -> list[dict[str, Any]]:
    os.makedirs(a.out, exist_ok=True)
    pts = stage0_points(a.m4)
    with open(os.path.join(a.out, "stage0_points.json"), "w") as f:
        json.dump(pts, f, indent=1)
    with open(os.path.join(a.out, "stage0_points.md"), "w") as f:
        f.write(stage0_markdown(pts))
    stage0_figure(pts, os.path.join(a.out, "stage0_points.png"))
    print(stage0_markdown(pts))
    return pts


def split(a: argparse.Namespace) -> dict[str, Any]:
    """Write the object split of the scenes' objects (``make_split``) with the bundle's fragility entries; refuses an
    existing file."""
    import hashlib

    from robots.mujoco_lab.fragility import entry_of
    from robots.mujoco_lab.prior import fragility_world

    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists: the split is fixed once")
    manifest = json.load(open(os.path.join(a.bundle, "manifest.json")))
    classes = fragility_world(manifest)[0]
    objects = sorted(json.load(open(a.scenes)))
    entries = {o: entry_of(o, classes)[0] for o in objects}
    sp = make_split(objects, entries, a.seed)
    sha = lambda path: hashlib.sha256(open(path, "rb").read()).hexdigest()  # noqa: E731
    sp["provenance"] = {"bundle": os.path.realpath(a.bundle), "manifest_sha256": sha(os.path.join(a.bundle, "manifest.json")),
                        "scenes": os.path.realpath(a.scenes), "scenes_sha256": sha(a.scenes), "n_objects": len(objects),
                        "laws": {o: entries[o].get("law") for o in objects}}
    with open(a.out, "w") as f:
        json.dump(sp, f, indent=1)
    for key, st in sp["strata"].items():
        print(f"{key}: validation {len(st['validation'])}, test {len(st['test'])}")
    print(f"validation {len(sp['validation'])}, test {len(sp['test'])}")
    return sp


# ---------------------------------------------------------------- Stage 1: the per-particle statistics (design 4, 5.4)

def weighted_quantile(x: Any, w: Any, q: float) -> float:
    """The lower inverse CDF of the weighted sample: the smallest x whose weight at or below it reaches q of the total
    weight (a relative 1e-12 slack absorbs the rounding of the cumulative sum)."""
    x, w = np.asarray(x, dtype=float), np.asarray(w, dtype=float)
    order = np.argsort(x, kind="stable")
    cw = np.cumsum(w[order])
    i = int(np.searchsorted(cw, q * cw[-1] * (1.0 - 1e-12), side="left"))
    return float(x[order][min(i, len(x) - 1)])


def cvar_lower(x: Any, w: Any, level: float) -> float:
    """The weighted mean of x over the lowest ``level`` share of the weight; the particle on the boundary counts with the
    part of its weight that falls inside."""
    x, w = np.asarray(x, dtype=float), np.asarray(w, dtype=float)
    order = np.argsort(x, kind="stable")
    xs, ws = x[order], w[order] / w.sum()
    take = np.clip(level - (np.cumsum(ws) - ws), 0.0, ws)
    return float(take @ xs / take.sum())


def stats_from_matrix(P: Any, w: Any, cost_s: float, keys: tuple[tuple[float, float], ...] = ()) -> dict[str, Any]:
    """One row's statistics from its per-particle outcome probabilities P (n x 11, OUTCOMES order) and the particle weights
    w: "p" the weighted mean by outcome, p_placed, p_irr, rec (the recoverable labels' sum of p x COST), cost_s, q_irr
    (the QUANTILE weighted quantile of the per-particle P(irr)) and "cvar": per (A, r) in ``keys``, keyed f"{A:g}|{r:g}",
    the CVAR_LEVEL lower CVaR of the per-particle score P_i(placed) - P_i(irr) / r - (rec_i + TIME_COST_PER_S cost_s) / A
    (the score form of J: J / A = 1 - score)."""
    P, w = np.asarray(P, dtype=float), np.asarray(w, dtype=float)
    w = w / w.sum()
    placed = P[:, OUTCOMES.index("placed_stable")]
    irr = P[:, [OUTCOMES.index(k) for k in IRREVERSIBLE]].sum(axis=1)
    rec = P[:, [OUTCOMES.index(k) for k in RECOVERABLE]] @ np.array([COST[k] for k in RECOVERABLE])
    c = rec + TIME_COST_PER_S * cost_s
    return {"p": {k: float(v) for k, v in zip(OUTCOMES, w @ P)}, "p_placed": float(w @ placed), "p_irr": float(w @ irr),
            "rec": float(w @ rec), "cost_s": float(cost_s), "q_irr": weighted_quantile(irr, w, QUANTILE),
            "cvar": {f"{A:g}|{r:g}": cvar_lower(placed - irr / r - c / A, w, CVAR_LEVEL) for A, r in keys}}


# ---------------------------------------------------------------- Stage 1: the rules (design 4)

def care_of(A: float, r: float) -> float:
    """kappa from r = A / (5 kappa)."""
    return float(A) / (5.0 * float(r))


def J(row: dict[str, Any], care: float, A: float) -> float:
    """The menu's expected cost of a row (costs.expected_cost with the abstain cost A): R0's price."""
    return expected_cost(Prediction(p=row["p"]), row["cost_s"], care, A)


def J1(row: dict[str, Any], care: float, A: float, V: dict[str, float]) -> float:
    """R1's price: a recoverable label y costs COST[y] + min(A, V[y]) in place of COST[y] + A (V[y]: the expected cost of
    going on after y)."""
    return J(row, care, A) - sum(row["p"][k] * (A - min(A, V[k])) for k in RECOVERABLE)


def J3(row: dict[str, Any], care: float, A: float) -> float:
    """R3's price: J with P(irr) in the irreversible term replaced by its QUANTILE tail q_irr (every irreversible label
    costs 5 kappa, costs.COST)."""
    return J(row, care, A) + 5.0 * care * (row["q_irr"] - row["p_irr"])


def _first(rows: list[dict[str, Any]], key: Any) -> dict[str, Any]:
    """The row with the smallest key; a tie goes to the earlier row in ROW_ORDER."""
    return min(rows, key=lambda row: (key(row), ROW_ORDER.index(row["id"]) if row["id"] in ROW_ORDER else len(ROW_ORDER),
                                      row["id"]))


def decide(rule: str, rows: list[dict[str, Any]], knob: float | None, A: float = A_MAIN, V: Any = None) -> str:
    """The row id a rule grasps with, or GIVE_UP_ROW. ``rows``: the same-destination commit and lower-risk rows' statistics
    (stats_from_matrix plus "id"); ``knob``: r for R0, R1, R3 and R6 (kappa = care_of(A, r)), the cap delta for R2 and R5,
    None for R4; ``V`` (R1): (care, A) -> {recoverable label: V}.
      R0  the cheapest row by J; grasp if J <= A.
      R1  the cheapest row by J1; grasp if J1 <= A.
      R2  among the rows with p_irr <= delta, the highest p_placed.
      R3  the cheapest row by J3; grasp if J3 <= A.
      R4  the highest p_placed; gives up only without rows.
      R5  among the rows with q_irr <= delta, the highest p_placed.
      R6  the highest CVaR of the score; grasp if it is above 0."""
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; known: {', '.join(RULES)}")
    if not rows:
        return GIVE_UP_ROW
    if rule in ("R0", "R1", "R3"):
        care = care_of(A, knob)
        if rule == "R1":
            v = V(care, A)
            price = lambda row: J1(row, care, A, v)  # noqa: E731
        else:
            price = (lambda row: J(row, care, A)) if rule == "R0" else (lambda row: J3(row, care, A))  # noqa: E731
        best = _first(rows, price)
        return best["id"] if price(best) <= A else GIVE_UP_ROW
    if rule in ("R2", "R5"):
        tail = "p_irr" if rule == "R2" else "q_irr"
        ok = [row for row in rows if row[tail] <= knob]
        return _first(ok, lambda row: -row["p_placed"])["id"] if ok else GIVE_UP_ROW
    if rule == "R4":
        return _first(rows, lambda row: -row["p_placed"])["id"]
    key = f"{A:g}|{knob:g}"
    best = _first(rows, lambda row: -row["cvar"][key])
    return best["id"] if best["cvar"][key] > 0 else GIVE_UP_ROW


def retry_continuation(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """R1's inputs from a retry run's rows (rows with an error or an artifact are skipped): per recoverable label y of the
    first attempt, n cases, q the share whose later attempts placed the object, d the share whose later attempts include
    an irreversible label, t TIME_COST_PER_S x the mean over the n cases of the later attempts' summed cost_s."""
    later: dict[str, list[list[dict[str, Any]]]] = {}
    for row in rows:
        if "error" in row or row.get("artifact") or not row.get("attempts"):
            continue
        if row["attempts"][0]["label"] in RECOVERABLE:
            later.setdefault(row["attempts"][0]["label"], []).append(row["attempts"][1:])
    return {y: {"n": len(ls),
                "q": sum(any(a["label"] == "placed_stable" for a in l) for l in ls) / len(ls),
                "d": sum(any(a["label"] in IRREVERSIBLE for a in l) for l in ls) / len(ls),
                "t": TIME_COST_PER_S * sum(sum(a["cost_s"] for a in l) for l in ls) / len(ls)}
            for y, ls in later.items()}


def continuation_values(c: dict[str, dict[str, Any]], care: float, A: float) -> dict[str, float]:
    """V[y] = (1 - q) A + d 5 kappa + t for every recoverable label y (retry_continuation's c); inf for a label without
    data, so that R1 prices it as R0 does."""
    return {y: (1.0 - c[y]["q"]) * A + c[y]["d"] * 5.0 * care + c[y]["t"] if y in c else float("inf") for y in RECOVERABLE}


# ---------------------------------------------------------------- Stage 1: the run (design 5.4)

def row_record(o: Any, per: list[Prediction], w: Any) -> dict[str, Any]:
    """A menu row's Stage 1 record: stats_from_matrix of its per-particle predictions (CVAR_KEYS), its id, kind and action
    key (evpi_props.action_key), and mean_gap, the largest gap between the statistics' mean and the menu's own prediction."""
    from robots.mujoco_lab.evpi_props import action_key
    from rpent.skills.pick_place.costs import prediction_matrix

    st = stats_from_matrix(prediction_matrix(per), w, o.cost_s, CVAR_KEYS)
    gap = max(abs(st["p"][k] - o.pred.p.get(k, 0.0)) for k in OUTCOMES)
    return dict(st, id=o.id, kind=o.kind, action=action_key(o.id, {o.id: o}), mean_gap=gap)


def case_status(a: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """unreachable (no pinch the arm reaches), no_rows (no row to the requested destination) or ok."""
    if not a.get("assessed", True):
        return "unreachable"
    return "ok" if rows else "no_rows"


def shard_objects(objects: list[str], k: int, n: int) -> list[str]:
    """Shard k of n: every n-th object of the sorted list, from the k-th."""
    if not 0 <= int(k) < int(n):
        raise ValueError(f"shard {k} of {n}")
    return sorted(set(objects))[int(k)::int(n)]


def dkey(rule: str, A: float, knob: float | None) -> str:
    return f"{rule}|{A:g}|{'-' if knob is None else format(knob, 'g')}"


def decision_keys() -> list[tuple[str, float, float | None]]:
    """(rule, A, knob): every rule's knobs at A_MAIN, then the only-r-matters check (CHECK_RULES at R_CHECK)."""
    return ([(rule, A_MAIN, k) for rule in RULES for k in KNOBS[rule]]
            + [(rule, A, r) for rule in CHECK_RULES for A, r in R_CHECK])


def decisions(rec: dict[str, Any], V: Any) -> dict[str, str]:
    """Every rule's first decision on a Stage 1 record, keyed by dkey."""
    rows = rec.get("rows") or []
    return {dkey(rule, A, k): decide(rule, rows, k, A, V) for rule, A, k in decision_keys()}


def stage1_case(prims: Any, o: str, idx: int, s: dict[str, Any], world: Any, shared: Any) -> dict[str, Any]:
    """One case's first menu at the class prior, as evpi_props.run_case builds it (the case's reset, the prior's draw on
    calibrate._stream k 1, the menu stream after it) with no probe rows, under (STAGE1_CARE, A_MAIN); the record holds
    every same-destination commit and lower-risk row (row_record) and the menu's own decision."""
    from robots.mujoco_lab import calibrate as cal
    from robots.mujoco_lab import evpi_props as ev
    from robots.mujoco_lab.adapter import object_props
    from robots.mujoco_lab.com_gap import fingerprint
    from robots.mujoco_lab.fragility import entry_of
    from rpent.skills.pick_place import menu as menu_mod
    from rpent.skills.pick_place.tools import _particles, _spread_kw

    t0 = time.time()
    env, seed = prims.env, int(s["seed"])
    theta, com_end, destination, hidden = cal.case_draw(seed, o, idx, object_props(o)["mass"], False, fragility=world)
    if world is not None:
        env.slosh_zeta, env.slosh_scale = world[1].slosh_zeta, world[1].slosh_scale
    env.classes = None if world is None else world[0]
    env.reset(o, theta, seed=seed * 1000 + idx, com_end=com_end, fixture=destination if destination in cal.MESH_IDS else None,
              fill=hidden["fill"], frag=hidden["frag"])
    shared.clear_case()
    reset_info = dict(getattr(env, "reset_info", None) or {})
    entry = entry_of(o, world[0])[0] if world is not None else {"class": "none", "law": None, "fillable": False}
    rec: dict[str, Any] = {"object": o, "idx": idx, "fold": s["fold"], "side": s["side"], "destination": destination,
                           "fingerprint": fingerprint(env), "at_rest": reset_info.get("at_rest", True),
                           "class": entry.get("class", "none"), "law": entry.get("law"), "fillable": bool(entry.get("fillable")),
                           "thin": entry.get("law") == "thin_plastic"}
    try:
        pose, n = env.pose_info(), _particles(prims)
        g = cal._stream(seed, o, idx, 1)
        b0 = prims.prior(pose, n, g, **_spread_kw(prims))
        state = copy.deepcopy(g.bit_generator.state)
        m = ev.first_menu(prims, b0, destination, STAGE1_CARE, A_MAIN, state)
        a = m["payload"]
        rows = []
        for x in m["menu"].values():
            if x.kind in ("commit", "lower_risk") and x.action.get("destination") == destination:
                _, per = menu_mod.estimate(prims.model, b0, x._base, x._pose, x.action["force"], x._planned)   # the menu's memo
                rows.append(row_record(x, per, b0.weights))
        guard = a.get("guard") or {}
        rec.update(n_particles=n, ess=float(b0.effective_size()), status=case_status(a, rows), choice=ev.choice_of(a),
                   recommendation=a.get("recommendation"), best_option=a.get("best_option"), without_probe=a.get("without_probe"),
                   guard_flag=bool(guard.get("flag")), first_menu=cal._first_menu(a), rows=rows)
        shared.clear()
    except Exception as exc:  # noqa: BLE001  one bad case must not stop the run
        rec.update(status="error", error=f"{type(exc).__name__}: {str(exc)[:300]}")
    rec["artifact"] = cal.case_artifact(rec, reset_info)
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def stage1_object(job: tuple[str, list[int], dict[str, Any]]) -> list[dict[str, Any]]:
    """One object's Stage 1 cases on its fold's bundle (evpi_props.run_object's sensor and world), no probe models."""
    from robots.mujoco_lab import calibrate as cal
    from robots.mujoco_lab.evpi_props import SharedMenus
    from robots.mujoco_lab.prior import fragility_world

    o, idxs, s = job
    prims = cal._primitives(s["bundle"], None, None)
    probes, prims.probe_models = getattr(prims, "probe_models", None), {}
    prims.force_noise = cal.job_force_noise(None, prims.manifest)
    prims.env.force_noise = prims.force_noise
    world = fragility_world(prims.manifest)
    out = []
    try:
        with SharedMenus(prims) as shared:
            for idx in idxs:
                rec = stage1_case(prims, o, idx, s, world, shared)
                out.append(rec)
                print(f"{o} {idx}: {rec['seconds']:.1f} s, {len(rec.get('rows') or [])} rows, {rec['status']}"
                      + (f", error {rec['error']}" if "error" in rec else ""), file=sys.stderr, flush=True)
                shared.hits.clear()
    finally:
        prims.probe_models = probes
    return out


def _sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _peak_rss_mb() -> dict[str, float]:
    import resource

    return {"self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "children": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0}


def stage1(a: argparse.Namespace) -> str:
    """The Stage 1 run: records.jsonl (one record per case), run_config.json and run_metrics.json (times, counts and
    errors only; wandb when on)."""
    import multiprocessing as mp

    from robots.mujoco_lab.estimation_check import _revision
    from robots.mujoco_lab.identify_fit import require_bundle_generator
    from robots.mujoco_lab.tracking import log_run

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "records.jsonl")
    if os.path.exists(path):
        raise SystemExit(f"{path} exists: pick a new --out")
    generator = require_bundle_generator(a.bundle)
    folds = json.load(open(os.path.join(a.bundle, "folds", "folds.json")))
    if sorted(json.load(open(a.scenes))) != sorted(folds):
        raise SystemExit("the scenes' objects differ from the bundle's folds")
    sp = json.load(open(a.split))
    side = {**{o: "validation" for o in sp["validation"]}, **{o: "test" for o in sp["test"]}}
    if sorted(side) != sorted(folds):
        raise SystemExit("the split's objects differ from the bundle's folds")
    if a.cases:
        cases = [(c.rsplit(":", 1)[0], int(c.rsplit(":", 1)[1])) for c in a.cases.split(",")]
    else:
        objs = shard_objects(a.objects.split(",") if a.objects else list(folds), a.shard, a.shards)
        cases = [(o, i) for o in objs for i in range(a.repeats)]
    by: dict[str, list[int]] = defaultdict(list)
    for o, i in cases:
        by[o].append(i)
    config = {**vars(a), **_revision(), "generator": generator, "split_sha256": _sha(a.split), "scenes_sha256": _sha(a.scenes),
              "manifest_sha256": _sha(os.path.join(a.bundle, "manifest.json")), "cases": cases, "rules": RULES, "r_grid": R_GRID,
              "delta_grid": DELTA_GRID, "a_main": A_MAIN, "r_check": R_CHECK, "quantile": QUANTILE, "cvar_level": CVAR_LEVEL,
              "stage1_care": STAGE1_CARE}
    with open(os.path.join(a.out, "run_config.json"), "w") as f:
        json.dump(config, f, indent=1)
    jobs = [(o, idxs, {"bundle": os.path.join(a.bundle, "folds", f"fold_{folds[o]}"), "fold": folds[o], "seed": a.seed,
                       "side": side[o]}) for o, idxs in by.items()]
    t0 = time.time()
    with open(path, "w") as out:
        if a.workers == 1:
            results = map(stage1_object, jobs)
            pool = None
        else:
            pool = mp.get_context("spawn").Pool(a.workers)
            results = pool.imap_unordered(stage1_object, jobs)
        try:
            for rows in results:
                for r in rows:
                    out.write(json.dumps(r) + "\n")
                out.flush()
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    recs = [json.loads(line) for line in open(path)]
    metrics = {"cases": len(recs), "errors": sum("error" in r for r in recs), "artifacts": sum(bool(r.get("artifact")) for r in recs),
               "status": dict(Counter(r["status"] for r in recs)), "wall_seconds": round(time.time() - t0, 1),
               "case_seconds": round(sum(r["seconds"] for r in recs), 1),
               "case_seconds_max": max((r["seconds"] for r in recs), default=0.0), "peak_rss_mb": _peak_rss_mb()}
    metrics["wandb"] = log_run("curve-stage1", config, metrics)
    with open(os.path.join(a.out, "run_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print(f"done: {json.dumps(metrics)}", flush=True)
    return path


# ---------------------------------------------------------------- Stage 1: the model-only report (design 5.4)

def counted(recs: list[dict[str, Any]], side: str | None) -> list[dict[str, Any]]:
    """The records a Stage 1 reading counts: no error, no artifact, on ``side`` (None: both sides)."""
    return [r for r in recs if "error" not in r and not r.get("artifact") and (side is None or r.get("side") == side)]


def predicted_point(recs: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """One rule and knob (``key`` in each record's "decisions"): the mean over the records of the chosen first row's
    P(placed) and P(irr) (a give-up counts 0 and 0), the give-up rate, the share of the chosen rows with P(placed) below
    LOW_SUCCESS and the chosen rows' P(placed) and P(irr) at the quantiles QS."""
    chosen = []
    for r in recs:
        d = r["decisions"][key]
        if d != GIVE_UP_ROW:
            chosen.append(next(row for row in r["rows"] if row["id"] == d))
    n = len(recs)
    pl, ir = [c["p_placed"] for c in chosen], [c["p_irr"] for c in chosen]
    return {"n": n, "success": sum(pl) / n if n else None, "irr": sum(ir) / n if n else None,
            "give_up": (n - len(chosen)) / n if n else None,
            "low_success": sum(p < LOW_SUCCESS for p in pl) / len(pl) if pl else None,
            "placed_q": [float(x) for x in np.quantile(pl, QS)] if pl else None,
            "irr_q": [float(x) for x in np.quantile(ir, QS)] if ir else None}


def curve_of(recs: list[dict[str, Any]], rule: str, A: float = A_MAIN) -> list[dict[str, Any]]:
    return [dict(knob=k, **predicted_point(recs, dkey(rule, A, k))) for k in KNOBS[rule]]


def at_equal_success(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per point of curve a, curve b's P(irr) at the same predicted success (linear between b's points; None outside
    b's success range)."""
    pts = sorted((p["success"], p["irr"]) for p in b if p["success"] is not None)
    xs, ys = [x for x, _ in pts], [y for _, y in pts]
    out = []
    for p in a:
        inside = bool(xs) and p["success"] is not None and xs[0] <= p["success"] <= xs[-1]
        out.append({"knob": p["knob"], "success": p["success"], "irr": p["irr"],
                    "irr_other": float(np.interp(p["success"], xs, ys)) if inside else None})
    return out


def only_r(recs: list[dict[str, Any]], rule: str, r: float, A: float = 3.0) -> dict[str, Any]:
    """The share of records whose first decision at (A, r) differs from the one at (A_MAIN, r)."""
    n = len(recs)
    differ = sum(x["decisions"][dkey(rule, A, r)] != x["decisions"][dkey(rule, A_MAIN, r)] for x in recs)
    return {"n": n, "differ": differ, "share": differ / n if n else None}


def e1_check(mine: list[dict[str, Any]], theirs: list[dict[str, Any]], setting: str = "k100_A10", prior: str = "class") -> dict[str, int]:
    """Stage 1 against experiment 1's records at kappa 100, A 10 (r 0.02), class prior; counts only. Experiment 1's
    decision without its probe rows: the best row when its without_probe is commit, else give up. Counted: the cases
    (no error), the ones experiment 1 lacks, the fingerprint and the guard flag that differ, the menus whose decisions
    differ; on the cases with no guard flag in either (the rules ignore the guard), R0 at r 0.02 against this run's menu
    and against experiment 1."""
    e1 = {(t["object"], int(t["idx"])): t for t in theirs if t.get("prior") == prior and t.get("setting") == setting and "error" not in t}
    keys = ("cases", "missing", "fingerprint_differs", "guard_differs", "menu_differs", "no_guard", "r0_vs_menu_differs",
            "r0_vs_e1_differs")
    out = dict.fromkeys(keys, 0)
    for r in mine:
        if "error" in r:
            continue
        out["cases"] += 1
        t = e1.get((r["object"], int(r["idx"])))
        if t is None:
            out["missing"] += 1
            continue
        m = t["menu"][PRIOR_KEY]
        theirs_d = m["best_option"] if m.get("without_probe") == "commit" else GIVE_UP_ROW
        mine_d = GIVE_UP_ROW if r["choice"] in NO_GRASP else r["choice"]
        r0 = r["decisions"][dkey("R0", A_MAIN, 0.02)]
        out["fingerprint_differs"] += r["fingerprint"] != t.get("fingerprint")
        out["guard_differs"] += bool(r["guard_flag"]) != bool(m.get("guard_flag"))
        out["menu_differs"] += mine_d != theirs_d
        if not r["guard_flag"] and not m.get("guard_flag"):
            out["no_guard"] += 1
            out["r0_vs_menu_differs"] += r0 != mine_d
            out["r0_vs_e1_differs"] += r0 != theirs_d
    return out


def main(argv: list[str] | None = None) -> Any:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s0 = sub.add_parser("stage0", help="the m4 sweep's points (descriptive)")
    s0.add_argument("--m4", required=True)
    s0.add_argument("--out", required=True)
    sp = sub.add_parser("split", help="the object split, fixed before any simulation")
    sp.add_argument("--bundle", required=True)
    sp.add_argument("--scenes", required=True)
    sp.add_argument("--seed", type=int, default=SPLIT_SEED)
    sp.add_argument("--out", required=True)
    s1 = sub.add_parser("stage1", help="Stage 1: the first menus at the class prior, model only")
    s1.add_argument("--bundle", required=True)
    s1.add_argument("--scenes", required=True)
    s1.add_argument("--split", required=True)
    s1.add_argument("--out", required=True)
    s1.add_argument("--seed", type=int, default=11, help="the run seed (experiment 1: 11)")
    s1.add_argument("--repeats", type=int, default=11)
    s1.add_argument("--objects", default=None, help="comma-separated objects (default: the bundle's 95)")
    s1.add_argument("--cases", default=None, help="comma-separated object:idx cases in place of the objects")
    s1.add_argument("--shard", type=int, default=0)
    s1.add_argument("--shards", type=int, default=1)
    s1.add_argument("--workers", type=int, default=1)
    a = p.parse_args(argv)
    return {"stage0": stage0, "split": split, "stage1": stage1}[a.cmd](a)


if __name__ == "__main__":
    main(sys.argv[1:])
