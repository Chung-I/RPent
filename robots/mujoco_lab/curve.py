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
EPS = 0.02                        # design 6: the success margin of the test reading (2 points)
BOOT_DRAWS = 2000                 # design 6: the object bootstrap's draws
BOOT_SEED = 20261108              # the bootstrap's stream (fixed before any Stage 2 record)
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


STAGE1_CONFIG_KEYS = ("bundle", "seed", "split_sha256", "scenes_sha256", "manifest_sha256", "code_revision", "generator", "r_grid",
                      "delta_grid", "a_main", "r_check", "quantile", "cvar_level", "stage1_care")


def load_stage1(dirs: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The records of finished Stage 1 runs (each dir: run_config.json, records.jsonl and run_metrics.json, which the run
    writes at its end); the runs must agree on STAGE1_CONFIG_KEYS and hold each case once."""
    recs: list[dict[str, Any]] = []
    cfgs = []
    for d in dirs:
        if not os.path.exists(os.path.join(d, "run_metrics.json")):
            raise SystemExit(f"{d}: the run has not finished (no run_metrics.json)")
        cfgs.append(json.load(open(os.path.join(d, "run_config.json"))))
        recs += [json.loads(line) for line in open(os.path.join(d, "records.jsonl"))]
    for k in STAGE1_CONFIG_KEYS:
        if len({json.dumps(c.get(k)) for c in cfgs}) > 1:
            raise SystemExit(f"the runs differ in {k}")
    twice = [k for k, n in Counter((r["object"], int(r["idx"])) for r in recs).items() if n > 1]
    if twice:
        raise SystemExit(f"{len(twice)} cases appear twice, the first {twice[0]}")
    return recs, cfgs[0]


def stage1_report(recs: list[dict[str, Any]], V: Any, stage0_pts: list[dict[str, Any]], e1_recs: list[dict[str, Any]]) -> dict[str, Any]:
    """The model-only check (design 5.4) on the validation objects: each rule's predicted curve (also without the
    thin-plastic objects and without the records whose ESS is below ESS_APART), each rule against R0 at equal predicted
    success, the only-r-matters shares with the Stage 0 points at the same r, and the counts: the records, the experiment 1
    check (every case without an error; counts only) and the largest mean_gap."""
    for r in recs:
        if "error" not in r:
            r["decisions"] = decisions(r, V)
    val = counted(recs, "validation")
    no_thin = [r for r in val if not r.get("thin")]
    ess_ok = [r for r in val if r.get("ess", float("inf")) >= ESS_APART]
    curves = {rule: curve_of(val, rule) for rule in RULES}
    ok = [r for r in recs if "error" not in r]
    counts = {"records": len(recs), "errors": len(recs) - len(ok), "artifacts": sum(bool(r.get("artifact")) for r in recs),
              "validation": len(val), "test_counted": len(counted(recs, "test")), "thin_validation": len(val) - len(no_thin),
              "ess_low_validation": len(val) - len(ess_ok), "status_validation": dict(Counter(r["status"] for r in val)),
              "guard_validation": sum(bool(r.get("guard_flag")) for r in val),
              "n_particles": sorted({int(r["n_particles"]) for r in ok if r.get("n_particles") is not None}),
              "rows_per_case_validation": {str(k): v for k, v in sorted(Counter(len(r.get("rows") or []) for r in val).items())}}
    return {"counts": counts, "curves": curves,
            "curves_no_thin": {rule: curve_of(no_thin, rule) for rule in RULES},
            "curves_ess_ok": {rule: curve_of(ess_ok, rule) for rule in RULES},
            "vs_r0": {rule: at_equal_success(curves[rule], curves["R0"]) for rule in RULES if rule != "R0"},
            "only_r": {dkey(rule, A, r): only_r(val, rule, r, A) for rule in CHECK_RULES for A, r in R_CHECK},
            "stage0_at_check": [p for p in stage0_pts if any(abs(float(p["r"]) - r) < 1e-9 for _, r in R_CHECK)],
            "e1": e1_check(ok, e1_recs),
            "mean_gap_max": max((row["mean_gap"] for r in ok for row in r.get("rows") or []), default=None)}


def _pc(x: float | None, nd: int = 1) -> str:
    return "–" if x is None else f"{100 * x:.{nd}f}"


def _knob(k: float | None) -> str:
    return "–" if k is None else f"{k:g}"


def _q(qs: list[float] | None, nd: int) -> str:
    return "–" if qs is None else " / ".join(f"{100 * x:.{nd}f}" for x in qs)


def stage1_markdown(rep: dict[str, Any]) -> str:
    c, e = rep["counts"], rep["e1"]
    pv = rep.get("provenance") or {}
    L = ["# Success-damage curve, Stage 1: the model-only check (validation objects only)", ""]
    if pv:
        L += [f"- {k}: {v}" for k, v in pv.items()] + [""]
    L += ["## Counts and checks", "",
          f"- Records {c['records']}: errors {c['errors']}, artifacts {c['artifacts']}. Validation cases counted {c['validation']} "
          f"(status {c['status_validation']}); test cases counted {c['test_counted']}, not read here.",
          f"- Validation: thin plastic {c['thin_validation']} cases, ESS below {ESS_APART:g} {c['ess_low_validation']}, guard flags "
          f"{c['guard_validation']} (reported, not applied). Rows per case {c['rows_per_case_validation']}. Particles {c['n_particles']}.",
          f"- Largest gap between a row's statistics and the menu's own mean prediction: {rep['mean_gap_max']}.",
          f"- Experiment 1 check (k100_A10, class prior; every case without an error, both sides; counts only): cases {e['cases']}, "
          f"missing {e['missing']}, fingerprint differs {e['fingerprint_differs']}, guard flag differs {e['guard_differs']}, "
          f"menu decision without probe rows differs {e['menu_differs']}; without a guard flag ({e['no_guard']}): R0 at r 0.02 "
          f"differs from this menu {e['r0_vs_menu_differs']}, from experiment 1 {e['r0_vs_e1_differs']}.", "",
          "## 1. Predicted curves (design 5.4)", "",
          "Per rule and knob, over the validation cases: the mean P(placed) and P(irr) of the first action (a give-up counts 0 "
          "and 0), in %. Chosen rows: their P(placed) and P(irr) as min / q1 / median / q3 / max (%), and the share with "
          f"P(placed) < {LOW_SUCCESS:g}.", "",
          "| rule | knob | success | P(irr) | give-up | chosen P(placed) < 0.2 | chosen P(placed) | chosen P(irr) |",
          "|---|---|---|---|---|---|---|---|"]
    for rule in RULES:
        for p in rep["curves"][rule]:
            L.append(f"| {rule} | {_knob(p['knob'])} | {_pc(p['success'])} | {_pc(p['irr'], 2)} | {_pc(p['give_up'])} | "
                     f"{_pc(p['low_success'])} | {_q(p['placed_q'], 1)} | {_q(p['irr_q'], 2)} |")
    n_nt = rep["curves_no_thin"]["R4"][0]["n"]
    L += ["", f"## 2. The same without the 9 thin-plastic objects ({n_nt} validation cases)", "",
          "| rule | knob | success | P(irr) | give-up | P(irr), all objects |", "|---|---|---|---|---|---|"]
    for rule in RULES:
        for p, q in zip(rep["curves_no_thin"][rule], rep["curves"][rule]):
            L.append(f"| {rule} | {_knob(p['knob'])} | {_pc(p['success'])} | {_pc(p['irr'], 2)} | {_pc(p['give_up'])} | {_pc(q['irr'], 2)} |")
    L += ["", "## 3. At equal predicted success: each rule's P(irr) and R0's", "",
          "R0's P(irr) is linear between its two points that bracket the success; – when the success is outside R0's range.", "",
          "| rule | knob | success | P(irr) | R0's P(irr) at that success |", "|---|---|---|---|---|"]
    for rule, pts in rep["vs_r0"].items():
        for p in pts:
            L.append(f"| {rule} | {_knob(p['knob'])} | {_pc(p['success'])} | {_pc(p['irr'], 2)} | {_pc(p['irr_other'], 2)} |")
    L += ["", "## 4. The give-up rate near the model's damage floor (R2 and R5, δ 0.005 to 0.02)", "",
          "| rule | δ | give-up | success | P(irr) |", "|---|---|---|---|---|"]
    for rule in ("R2", "R5"):
        for p in rep["curves"][rule]:
            if 0.005 - 1e-12 <= p["knob"] <= 0.02 + 1e-12:
                L.append(f"| {rule} | {p['knob']:g} | {_pc(p['give_up'])} | {_pc(p['success'])} | {_pc(p['irr'], 2)} |")
    L += ["", "## 5. Does only r matter? First decisions at A 3 against A 10, same r (validation cases)", "",
          "| rule | r | cases | decisions differ | share |", "|---|---|---|---|---|"]
    for k, v in rep["only_r"].items():
        rule, _, r = k.split("|")
        L.append(f"| {rule} | {r} | {v['n']} | {v['differ']} | {_pc(v['share'])} % |")
    L += ["", "Stage 0 beside it: the m4 sweep's simulated points at the same r (bundle m2b2, seed 12, descriptive):", "",
          "| protocol | κ | A | r | success | irreversible | give-up |", "|---|---|---|---|---|---|---|"]
    for p in sorted(rep["stage0_at_check"], key=lambda p: (p["r"], p["protocol"], p["A"])):
        L.append(f"| {p['protocol']} | {p['care']:g} | {p['A']:g} | {p['r']:g} | {_pc(p['success'])} | {_pc(p['irreversible'])} | "
                 f"{_pc(p.get('give_up'))} |")
    n_lo = c["ess_low_validation"]
    L += ["", f"## 6. Records with ESS below {ESS_APART:g}: {n_lo} validation cases", ""]
    if n_lo:
        L += ["The curves without them:", "", "| rule | knob | success | P(irr) | give-up |", "|---|---|---|---|---|"]
        for rule in RULES:
            for p in rep["curves_ess_ok"][rule]:
                L.append(f"| {rule} | {_knob(p['knob'])} | {_pc(p['success'])} | {_pc(p['irr'], 2)} | {_pc(p['give_up'])} |")
    return "\n".join(L) + "\n"


RULE_COLOURS = {"R0": "#2a78d6", "R1": "#eb6834", "R3": "#1baf7a", "R6": "#eda100", "R2": "#e87ba4", "R5": "#008300"}
RULE_NAMES = {"R0": "R0 J", "R1": "R1 J + retry value", "R2": "R2 cap on P(irr)", "R3": "R3 J with the P(irr) tail",
              "R4": "R4 always grasp", "R5": "R5 cap on the P(irr) tail", "R6": "R6 CVaR score"}


def _end_labels(ax: Any, ends: list[tuple[float, float, str]]) -> None:
    """Line-end labels in one column right of the data, top down, at least 4.5 % of the y range apart, with leader lines."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    gap, xl = 0.045 * (y1 - y0), x1 + 0.03 * (x1 - x0)
    ax.set_xlim(x0, x1 + 0.16 * (x1 - x0))
    last = None
    for x, y, text in sorted(ends, key=lambda e: -e[1]):
        last = y if last is None else min(y, last - gap)
        ax.annotate(text, (x, y), xytext=(xl, last), textcoords="data", fontsize=8, color="#52514e", va="center",
                    arrowprops={"arrowstyle": "-", "color": "#c3c2b7", "lw": 0.6})


def stage1_figure(rep: dict[str, Any], path: str) -> None:
    """Predicted success against predicted damage on the validation cases: the rules whose knob is r (left) and the caps
    with R0 for reference (right); R4 is one point in both. Each line runs through its knob values, labelled at its end."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), sharey=True)
    r4 = rep["curves"]["R4"][0]
    for ax, rules, title in ((axes[0], ("R0", "R1", "R3", "R6"), "rules traced by r"),
                             (axes[1], ("R0", "R2", "R5"), "caps traced by δ (R0 for reference)")):
        ends = []
        for rule in rules:
            pts = rep["curves"][rule]
            x, y = [100 * p["irr"] for p in pts], [100 * p["success"] for p in pts]
            ax.plot(x, y, color=RULE_COLOURS[rule], lw=2, marker="o", ms=4, mec="white", mew=1, zorder=3, label=RULE_NAMES[rule])
            ends.append((x[-1], y[-1], rule))
        ax.scatter([100 * r4["irr"]], [100 * r4["success"]], marker="*", s=110, color="#52514e", edgecolors="white", linewidths=1,
                   zorder=4, label=RULE_NAMES["R4"])
        ax.annotate("R4", (100 * r4["irr"], 100 * r4["success"]), xytext=(5, -9), textcoords="offset points", fontsize=8,
                    color="#52514e")
        _end_labels(ax, ends)
        ax.set_xlabel("predicted irreversible damage (% of cases)")
        ax.set_title(title, fontsize=10)
        ax.grid(True, color="#e6e5e0", linewidth=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    axes[0].set_ylabel("predicted task success (% of cases)")
    fig.suptitle(f"Stage 1, model only: first actions on {rep['counts']['validation']} validation cases (m2b2_v12b, class prior)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def stage1_summary(a: argparse.Namespace) -> dict[str, Any]:
    """The Stage 1 report from finished runs: stage1_summary.{json,md,png} in --out (refuses an existing json), wandb."""
    import glob

    from robots.mujoco_lab.tracking import log_run

    os.makedirs(a.out, exist_ok=True)
    jpath = os.path.join(a.out, "stage1_summary.json")
    if os.path.exists(jpath):
        raise SystemExit(f"{jpath} exists: pick a new --out")
    recs, cfg = load_stage1(a.runs.split(","))
    c = retry_continuation(json.load(gzip.open(a.retry))["rows"])
    V = lambda care, A: continuation_values(c, care, A)  # noqa: E731
    stage0_pts = json.load(open(a.stage0))
    e1 = []
    for f in sorted(glob.glob(os.path.join(a.e1, "shard_*", "records.jsonl.gz"))):
        for line in gzip.open(f, "rt"):
            t = json.loads(line)
            if t.get("prior") == "class" and t.get("setting") == "k100_A10":
                e1.append(t)
    rep = stage1_report(recs, V, stage0_pts, e1)
    rep["provenance"] = {"runs": a.runs, "code_revision": cfg["code_revision"], "bundle": cfg["bundle"], "seed": cfg["seed"],
                         "split": f"{a.split_commit} (sha256 {cfg['split_sha256']})", "retry": f"{a.retry} (sha256 {_sha(a.retry)})",
                         "retry_continuation": c, "stage0": a.stage0, "e1": f"{a.e1} ({len(e1)} class k100_A10 records)"}
    rep["wandb"] = log_run("curve-stage1-summary", {"runs": a.runs, **{k: cfg.get(k) for k in STAGE1_CONFIG_KEYS}},
                           {"counts": rep["counts"], "e1": rep["e1"],
                            "curves": {f"{rule}/{_knob(p['knob'])}": {"success": p["success"], "irr": p["irr"], "give_up": p["give_up"]}
                                       for rule in RULES for p in rep["curves"][rule]}})
    with open(jpath, "w") as f:
        json.dump(rep, f, indent=1)
    with open(os.path.join(a.out, "stage1_summary.md"), "w") as f:
        f.write(stage1_markdown(rep))
    stage1_figure(rep, os.path.join(a.out, "stage1_curves.png"))
    print(f"wrote {a.out}", flush=True)
    return rep


# ---------------------------------------------------------------- Stage 2: the episodes (design 5.2, 5.5)

SIM_RULES = ("R0", "R1", "R2", "R3", "R4", "R5", "R6")   # R6: admitted by the Stage 1 check (its curve differs from R0's, 7 Nov)


def no_continuation(care: float, A: float) -> dict[str, float]:
    """R1's continuation values at the second decision: no third attempt, so a recoverable label costs C(y) + A, as in R0."""
    return {k: float("inf") for k in RECOVERABLE}


def first_rows(dec: dict[str, str], rules: tuple[str, ...]) -> list[str]:
    """The distinct rows the rules grasp with at A_MAIN (the r-check's A 3 decisions are not simulated), in ROW_ORDER."""
    ids = {d for k, d in dec.items() if d != GIVE_UP_ROW and k.split("|")[0] in rules and float(k.split("|")[1]) == A_MAIN}
    return sorted(ids, key=lambda i: (ROW_ORDER.index(i) if i in ROW_ORDER else len(ROW_ORDER), i))


def second_decisions(rows: list[dict[str, Any]], rules: tuple[str, ...]) -> dict[str, str]:
    """Every rule's decision at the second (last) attempt from the second menu's rows, keyed by dkey at A_MAIN."""
    return {dkey(rule, A_MAIN, k): decide(rule, rows, k, A_MAIN, no_continuation) for rule in rules for k in KNOBS[rule]}


def menu_matches(now: list[dict[str, Any]], then: list[dict[str, Any]], tol: float = 1e-12) -> tuple[bool, str]:
    """(match, reason): the same row ids, the same action keys, and mean predictions within tol."""
    a, b = {r["id"]: r for r in now}, {r["id"]: r for r in then}
    if sorted(a) != sorted(b):
        return False, f"rows {sorted(a)} against {sorted(b)}"
    for i in sorted(a):
        if a[i]["action"] != b[i]["action"]:
            return False, f"{i}: action {a[i]['action']} against {b[i]['action']}"
        gap = max(abs(a[i]["p"].get(k, 0.0) - b[i]["p"].get(k, 0.0)) for k in set(a[i]["p"]) | set(b[i]["p"]))
        if gap > tol:
            return False, f"{i}: the mean prediction differs by {gap:.3g}"
    return True, ""


def episode(dec: dict[str, str], sims: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """One rule and knob's episode on a case (the main seed's simulations): the first decision, its attempt, and after a
    recoverable label the second decision (stored with the second menu) and its attempt. A give-up at either decision
    ends the episode unplaced; an irreversible label ends it."""
    out: dict[str, Any] = {"attempts": 0, "success": False, "irr": False, "give_up": False, "labels": [], "rows": []}
    first = dec[key]
    if first == GIVE_UP_ROW:
        return dict(out, give_up=True)
    s = next(x for x in sims if x["seed_k"] == 0 and x["first"] == first)
    out.update(attempts=1, labels=[s["label"]], rows=[first])
    if s["label"] == "placed_stable":
        return dict(out, success=True)
    if s["label"] in IRREVERSIBLE:
        return dict(out, irr=True)
    second = s["menu2"]["decisions"][key]
    if second == GIVE_UP_ROW:
        return dict(out, give_up=True)
    y2 = s["second"][second]["label"]
    return dict(out, attempts=2, labels=[s["label"], y2], rows=[first, second], success=y2 == "placed_stable", irr=y2 in IRREVERSIBLE)


def case_dropped(sims: list[dict[str, Any]]) -> bool:
    """Design 5.5: an artifact or an error on any simulation of the case at the main seed drops it from every rule."""
    for s in sims:
        if s["seed_k"] != 0:
            continue
        if s.get("error") or s.get("artifact") or (s.get("menu2") or {}).get("status") == "error":
            return True
        if any(x.get("error") or x.get("artifact") for x in (s.get("second") or {}).values()):
            return True
    return False



def simulate_case(lab: Any, rec1: dict[str, Any], dec: dict[str, str], rules: tuple[str, ...], seeds: list[int]) -> dict[str, Any]:
    """One case's simulations through ``lab`` (LabCase: fresh, attempt, second_menu). Per reset seed and per distinct first
    row of the rules (first_rows): a fresh episode (the reset and the first menu, gated against Stage 1's rows), the first
    attempt, and after a recoverable label without an artifact the second menu, every rule's second decision
    (second_decisions) and each distinct second row. The first second row runs in the same episode; each later one runs in
    a fresh episode whose replayed first attempt must give the same label and the same second menu. An exception is the
    error of the simulation it occurs in (a second menu that raises has the error status). Returns {"sims", "stop"}: a
    gate that fails stops the case (and the run) with its reason."""
    sims: list[dict[str, Any]] = []
    for k, reset_seed in enumerate(seeds):
        for first in first_rows(dec, rules):
            t0 = time.time()
            sim: dict[str, Any] = {"seed_k": k, "reset_seed": int(reset_seed), "first": first, "label": None, "artifact": False,
                                   "error": None, "menu2": None, "second": {}}
            try:
                store, stop, at_rest = lab.fresh(reset_seed, rec1)
                if stop:
                    return {"sims": sims, "stop": f"seed {k}, first row {first}: {stop}"}
                d1 = lab.attempt(store, first)
            except Exception as exc:  # noqa: BLE001  this simulation's error (design 5.5: at seed 0 it drops the case)
                d1, at_rest = {"error": _err(exc)}, True
            sim.update(label=d1.get("outcome"), artifact=bool(d1.get("artifact")) or not at_rest, error=d1.get("error"))
            if sim["error"] is None and sim["label"] in RECOVERABLE and not sim["artifact"]:
                try:
                    m2 = lab.second_menu(store, d1)
                except Exception as exc:  # noqa: BLE001
                    m2 = {"status": "error", "error": _err(exc), "rows": []}
                m2["decisions"] = second_decisions(m2["rows"], rules)
                sim["menu2"] = m2
                for j, row2 in enumerate(first_rows(m2["decisions"], rules) if m2["status"] != "error" else []):
                    try:
                        if j:
                            store, stop, at_rest = lab.fresh(reset_seed, rec1)
                            if stop:
                                return {"sims": sims + [sim], "stop": f"seed {k}, first row {first}, second row {row2}: {stop}"}
                            d1b = lab.attempt(store, first)
                            again = (d1b.get("outcome"), bool(d1b.get("artifact")) or not at_rest, d1b.get("error"))
                            if again != (sim["label"], sim["artifact"], sim["error"]):
                                return {"sims": sims + [sim], "stop": f"seed {k}: the replayed first attempt of {first} gave {again}"}
                            m2b = lab.second_menu(store, d1b)
                            ok, why = menu_matches(m2b["rows"], m2["rows"])
                            if not ok or m2b["status"] != m2["status"]:
                                return {"sims": sims + [sim], "stop": f"seed {k}: the replayed second menu after {first} differs: {why}"}
                        d2 = lab.attempt(store, row2)
                    except Exception as exc:  # noqa: BLE001
                        d2 = {"error": _err(exc)}
                    sim["second"][row2] = {"label": d2.get("outcome"), "artifact": bool(d2.get("artifact")), "error": d2.get("error")}
            sim["seconds"] = round(time.time() - t0, 2)
            sims.append(sim)
    return {"sims": sims, "stop": None}


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:300]}"


def stage2_skip(rec1: dict[str, Any]) -> str | None:
    """Why a Stage 1 case is not simulated: a Stage 1 error or a reset artifact drops it from every rule (paired)."""
    if rec1.get("status") == "error" or "error" in rec1:
        return "stage1 error"
    if rec1.get("artifact"):
        return "stage1 artifact"
    return None


def stage2_record(rec1: dict[str, Any], dec: dict[str, str], rules: tuple[str, ...], res: dict[str, Any] | None,
                  n_seeds: int) -> dict[str, Any]:
    """A case's Stage 2 record: its Stage 1 identity and strata, every first decision (dkey), the simulated first rows,
    the first menu's rows' P(placed), P(irr) and QUANTILE tail (for the calibration), the simulations (simulate_case),
    and whether the case is skipped (stage2_skip) or dropped (case_dropped, or an exception in its simulations)."""
    skip = stage2_skip(rec1)
    sims = [] if skip or res is None else res["sims"]
    keep = ("object", "idx", "fold", "side", "class", "law", "fillable", "thin", "destination", "ess", "guard_flag", "status")
    return dict({k: rec1.get(k) for k in keep}, decisions=dec, first_rows=[] if skip else first_rows(dec, rules),
                row_stats={r["id"]: {"p_placed": r["p_placed"], "p_irr": r["p_irr"], "q_irr": r["q_irr"]} for r in rec1.get("rows") or []},
                sims=sims, skipped=skip, error=None if res is None else res.get("error"),
                dropped=bool(skip) or bool((res or {}).get("error")) or case_dropped(sims), stop=None if res is None else res.get("stop"),
                n_seeds=int(n_seeds))


def smoke_summary(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """The smoke test's counts and times (no rates): the (case, first row) pairs and (case, first row, second row) triples
    whose label differs across seeds, the triples that ran at some but not all of their first row's seeds, the summed
    simulation seconds per seed, and the seconds per first row at seed 0 (the projection's unit)."""
    first: dict[tuple, dict[int, Any]] = defaultdict(dict)
    second: dict[tuple, dict[int, Any]] = defaultdict(dict)
    secs: dict[str, float] = defaultdict(float)
    errors = artifacts = 0
    for rec in recs:
        for s in rec["sims"]:
            key = (rec["object"], rec["idx"], s["first"])
            first[key][s["seed_k"]] = s["label"]
            secs[str(s["seed_k"])] += float(s.get("seconds") or 0.0)
            errors += bool(s.get("error"))
            artifacts += bool(s.get("artifact"))
            for row2, x in (s.get("second") or {}).items():
                second[key + (row2,)][s["seed_k"]] = x["label"]
                errors += bool(x.get("error"))
                artifacts += bool(x.get("artifact"))
    n0 = sum(0 in v for v in first.values())
    return {"first_pairs": len(first), "first_changed": sum(len(set(v.values())) > 1 for v in first.values()),
            "second_triples": len(second), "second_changed": sum(len(set(v.values())) > 1 for v in second.values()),
            "second_presence_changed": sum(set(v) != set(first[k[:3]]) for k, v in second.items()),
            "seconds_per_seed": dict(sorted(secs.items())), "first_rows_seed0": n0,
            "seconds_per_first_row_seed0": secs.get("0", 0.0) / n0 if n0 else None, "errors": errors, "artifacts": artifacts}


def projection(sm: dict[str, Any], n_first_rows: int) -> float:
    """CPU-h of the full run (seed 0 only): the smoke test's seconds per first row at seed 0 (second rows included) times
    the run's number of distinct (case, first row) pairs."""
    return float(sm["seconds_per_first_row_seed0"]) * int(n_first_rows) / 3600.0



# ---------------------------------------------------------------- Stage 2: the lock and the test reading (design 6)

def _kept(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in recs if not r["dropped"]]


def rule_point(recs: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """One rule and knob's episodes (``episode``) over the cases that are not dropped: counts and rates of success,
    irreversible damage, give-ups (at either decision) and second attempts; every episode counts, give-ups included."""
    eps = [episode(r["decisions"], r["sims"], key) for r in _kept(recs)]
    n = len(eps)
    out: dict[str, Any] = {"key": key, "n": n, "n_success": sum(e["success"] for e in eps), "n_irr": sum(e["irr"] for e in eps),
                           "n_give_up": sum(e["give_up"] for e in eps), "n_second": sum(e["attempts"] == 2 for e in eps)}
    for k in ("success", "irr", "give_up", "second"):
        out[k] = out[f"n_{k}"] / n if n else None
    return out


def knob_of(key: str) -> float | None:
    k = key.split("|")[2]
    return None if k == "-" else float(k)


def lock(val: list[dict[str, Any]], rules: tuple[str, ...]) -> dict[str, Any]:
    """Each rule's operating point on validation records only (design 6): among its knob values whose success count is at
    least R4's, the lowest irreversible count (then the higher success, then the grid order); if none qualifies, the
    highest success (then the lower damage, then the grid order). R4 has one point."""
    if any(r["side"] != "validation" for r in val):
        raise ValueError("lock reads validation records only")
    r4 = rule_point(val, dkey("R4", A_MAIN, None))
    out: dict[str, Any] = {"r4_success": r4["success"], "r4_n_success": r4["n_success"], "n": r4["n"],
                           "R4": {"knob": None, "key": r4["key"], "qualifies": True, "point": r4}}
    for rule in rules:
        if rule == "R4":
            continue
        pts = [(i, rule_point(val, dkey(rule, A_MAIN, k))) for i, k in enumerate(KNOBS[rule])]
        ok = [(i, p) for i, p in pts if p["n_success"] >= r4["n_success"]]
        if ok:
            i, p = min(ok, key=lambda t: (t[1]["n_irr"], -t[1]["n_success"], t[0]))
        else:
            i, p = min(pts, key=lambda t: (-t[1]["n_success"], t[1]["n_irr"], t[0]))
        out[rule] = {"knob": KNOBS[rule][i], "key": p["key"], "qualifies": bool(ok), "point": p}
    return out


def paired_bootstrap(recs: list[dict[str, Any]], key_a: str, key_b: str, draws: int = BOOT_DRAWS,
                     seed: int = BOOT_SEED) -> dict[str, Any]:
    """a - b in success and in irreversible damage, paired by case over the cases that are not dropped, with a 95 % object
    bootstrap (objects drawn with replacement, every case of a drawn object kept; percentile interval)."""
    kept = _kept(recs)
    objs = sorted({r["object"] for r in kept})
    at = {o: i for i, o in enumerate(objs)}
    ds, di, cnt = np.zeros(len(objs)), np.zeros(len(objs)), np.zeros(len(objs))
    for r in kept:
        ea, eb = episode(r["decisions"], r["sims"], key_a), episode(r["decisions"], r["sims"], key_b)
        j = at[r["object"]]
        ds[j] += float(ea["success"]) - float(eb["success"])
        di[j] += float(ea["irr"]) - float(eb["irr"])
        cnt[j] += 1.0
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, len(objs), size=(int(draws), len(objs)))
    M = np.zeros((int(draws), len(objs)))
    np.add.at(M, (np.arange(int(draws))[:, None], idx), 1.0)
    n = M @ cnt
    out: dict[str, Any] = {"a": key_a, "b": key_b, "objects": len(objs), "cases": len(kept), "draws": int(draws), "seed": int(seed)}
    for name, d in (("success", ds), ("irr", di)):
        lo, hi = np.percentile((M @ d) / n, [2.5, 97.5])
        out[name] = {"diff": float(d.sum() / cnt.sum()), "lo": float(lo), "hi": float(hi)}
    return out


def q1_reading(b: dict[str, Any], eps: float = EPS) -> str:
    """Design 6, Q1 (a = R0, b = R4): (a) the success difference's lower bound above -eps, (b) the damage difference's
    upper bound below 0."""
    kept, cut = b["success"]["lo"] > -eps, b["irr"]["hi"] < 0
    if kept and cut:
        return "SAGE keeps success and cuts damage"
    return "cuts damage, success not shown to be kept" if cut else "undecided"


def vs_r0_reading(b: dict[str, Any], eps: float = EPS) -> str:
    """Design 6, Q2-Q4 (a = the rule, b = R0): "differs" only if the damage interval excludes 0 and the rule with the lower
    damage is non-inferior in success (its success lower bound against the other above -eps)."""
    if b["irr"]["hi"] < 0:
        return "differs (lower damage than R0)" if b["success"]["lo"] > -eps else "no detectable difference"
    if b["irr"]["lo"] > 0:
        return "differs (higher damage than R0)" if b["success"]["hi"] < eps else "no detectable difference"
    return "no detectable difference"


def _row_of(rows: list[dict[str, Any]], rid: str) -> dict[str, Any]:
    return next(x for x in rows if x["id"] == rid)


def calibration(recs: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Predicted P(irr) of the rows the rule grasped with (first attempts: the first menu's row_stats; second attempts: the
    second menu's rows) against the irreversible labels, over the cases that are not dropped. For R2 and R5 also on the
    attempts whose capped statistic (p_irr, q_irr) is within a factor of 2 of the cap delta."""
    rule = key.split("|")[0]
    cap = {"R2": "p_irr", "R5": "q_irr"}.get(rule)
    delta = knob_of(key) if cap else None
    out: dict[str, Any] = {"all": {"attempts": 0, "predicted": 0.0, "observed": 0}}
    if cap:
        out["near_cap"] = {"attempts": 0, "predicted": 0.0, "observed": 0, "window": [delta / 2.0, 2.0 * delta], "stat": cap}
    for r in _kept(recs):
        e = episode(r["decisions"], r["sims"], key)
        for j, (row, label) in enumerate(zip(e["rows"], e["labels"])):
            if j == 0:
                st = r["row_stats"][row]
            else:
                s = next(x for x in r["sims"] if x["seed_k"] == 0 and x["first"] == e["rows"][0])
                st = _row_of(s["menu2"]["rows"], row)
            bins = ["all"] + (["near_cap"] if cap and delta / 2.0 <= st[cap] <= 2.0 * delta else [])
            for b in bins:
                out[b]["attempts"] += 1
                out[b]["predicted"] += float(st["p_irr"])
                out[b]["observed"] += int(label in IRREVERSIBLE)
    return out


def second_decision_mismatches(recs: list[dict[str, Any]], rules: tuple[str, ...]) -> int:
    """The stored second decisions that differ from second_decisions on the stored second menu (every record and seed)."""
    bad = 0
    for r in recs:
        for s in r["sims"]:
            m = s.get("menu2")
            if not m or "decisions" not in m:
                continue
            again = second_decisions(m["rows"], rules)
            bad += sum(m["decisions"].get(k) != v for k, v in again.items())
    return bad



STAGE2_CONFIG_KEYS = ("bundle", "seed", "rules", "n_seeds", "stage1_records_sha256", "retry_sha256", "split_sha256")


SIDES = ("validation", "test")


def records_path(d: str, side: str) -> str:
    """A Stage 2 run's records of one side: the lock opens the validation file only."""
    return os.path.join(d, f"records_{side}.jsonl")


def load_stage2(dirs: list[str], sides: tuple[str, ...] = SIDES) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The records of finished Stage 2 runs that agree on STAGE2_CONFIG_KEYS, each case once, from the records files of
    ``sides`` only (stage2-lock never opens the test file). A run that stopped is refused."""
    recs: list[dict[str, Any]] = []
    cfgs = []
    for d in dirs:
        if not os.path.exists(os.path.join(d, "run_metrics.json")):
            raise SystemExit(f"{d}: the run has not finished (no run_metrics.json)")
        if json.load(open(os.path.join(d, "run_metrics.json"))).get("stop"):
            raise SystemExit(f"{d}: the run stopped on a gate")
        cfgs.append(json.load(open(os.path.join(d, "run_config.json"))))
        for side in sides:
            for line in open(records_path(d, side)):
                r = json.loads(line)
                if r.get("stop"):
                    raise SystemExit(f"{d}: {r['object']} {r['idx']} holds a stop: the run did not finish")
                if r["side"] != side:
                    raise SystemExit(f"{d}: {r['object']} {r['idx']} has side {r['side']} in the {side} file")
                recs.append(r)
    for k in STAGE2_CONFIG_KEYS:
        if len({json.dumps(c.get(k)) for c in cfgs}) > 1:
            raise SystemExit(f"the runs differ in {k}")
    twice = [k for k, n in Counter((r["object"], int(r["idx"])) for r in recs).items() if n > 1]
    if twice:
        raise SystemExit(f"{len(twice)} cases appear twice, the first {twice[0]}")
    return recs, cfgs[0]


def stage2_checks(recs: list[dict[str, Any]], cfg: dict[str, Any], sides: tuple[str, ...]) -> dict[str, Any]:
    """Before any reading: the Stage 1 records and the retry run are the ones the run hashed, every Stage 1 case of the
    sides is here once, and every stored first and second decision equals its recomputation. Refuses a failed check."""
    rules = tuple(cfg["rules"])
    dirs1 = cfg["stage1"].split(",")
    for d in dirs1:
        if _sha(os.path.join(d, "records.jsonl")) != cfg["stage1_records_sha256"][d]:
            raise SystemExit(f"{d}: the Stage 1 records changed since the Stage 2 run")
    if _sha(cfg["retry"]) != cfg["retry_sha256"]:
        raise SystemExit("the retry run changed since the Stage 2 run")
    recs1, _ = load_stage1(dirs1)
    by1 = {(r["object"], int(r["idx"])): r for r in recs1 if r["side"] in sides}
    have = {(r["object"], int(r["idx"])) for r in recs}
    c = retry_continuation(json.load(gzip.open(cfg["retry"]))["rows"])
    V = lambda care, A: continuation_values(c, care, A)  # noqa: E731
    keys = [dkey(rule, A_MAIN, k) for rule in rules for k in KNOBS[rule]]
    first_bad = 0
    for r in recs:
        r1 = by1.get((r["object"], int(r["idx"])))
        if r1 is not None:
            again = decisions(r1, V)
            first_bad += sum(r["decisions"].get(k) != again[k] for k in keys)
    out = {"cases": len(recs), "missing_cases": len(set(by1) - have), "extra_cases": len(have - set(by1)),
           "first_decision_mismatches": first_bad, "second_decision_mismatches": second_decision_mismatches(recs, rules),
           "skipped": sum(bool(r["skipped"]) for r in recs), "case_errors": sum(bool(r.get("error")) for r in recs),
           "dropped": sum(bool(r["dropped"]) for r in recs)}
    bad = {k: out[k] for k in ("missing_cases", "extra_cases", "first_decision_mismatches", "second_decision_mismatches") if out[k]}
    if bad:
        raise SystemExit(f"the Stage 2 records fail the checks: {bad}")
    return out


def _curves(recs: list[dict[str, Any]], rules: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    return {rule: [dict(rule_point(recs, dkey(rule, A_MAIN, k)), knob=k) for k in KNOBS[rule]] for rule in rules}


def stage2_lock(a: argparse.Namespace) -> dict[str, Any]:
    """The operating points from the validation records only (``lock``): locked.json and locked.md in --out (refuses an
    existing locked.json), wandb. Commit locked.json before stage2-summary reads a test record."""
    from robots.mujoco_lab.tracking import log_run

    os.makedirs(a.out, exist_ok=True)
    jpath = os.path.join(a.out, "locked.json")
    if os.path.exists(jpath):
        raise SystemExit(f"{jpath} exists: pick a new --out")
    dirs = a.runs.split(",")
    val, cfg = load_stage2(dirs, sides=("validation",))
    rules = tuple(cfg["rules"])
    checks = stage2_checks(val, cfg, ("validation",))
    lk = lock(val, rules)
    out = {"lock": lk, "curves_validation": _curves(val, rules), "checks": checks,
           "provenance": {"runs": a.runs, "records_sha256": {d: _sha(records_path(d, "validation")) for d in dirs},
                          "code_revision": cfg["code_revision"], "stage1": cfg["stage1"], "rules": rules}}
    out["wandb"] = log_run("curve-stage2-lock", {"runs": a.runs, "rules": rules},
                           {"checks": checks, "lock": {r: {"knob": v["knob"], "qualifies": v["qualifies"]} for r, v in lk.items()
                                                       if isinstance(v, dict)}})
    with open(jpath, "w") as f:
        json.dump(out, f, indent=1)
    L = ["# Success-damage curve, Stage 2: the locked operating points (validation objects only)", "",
         f"- Runs: {a.runs}", f"- Validation cases {checks['cases']}: dropped {checks['dropped']} (skipped {checks['skipped']}, "
         f"case errors {checks['case_errors']}); decision mismatches 0.",
         f"- R4's validation success: {_pc(lk['r4_success'])} % ({lk['r4_n_success']} of {lk['n']}).", "",
         "| rule | knob | qualifies | success | irreversible | give-up | second attempts |", "|---|---|---|---|---|---|---|"]
    for rule in rules:
        v = lk[rule]
        p = v["point"]
        L.append(f"| {rule} | {_knob(v['knob'])} | {'yes' if v['qualifies'] else 'no'} | {_pc(p['success'])} | {_pc(p['irr'], 2)} | "
                 f"{_pc(p['give_up'])} | {_pc(p['second'])} |")
    with open(os.path.join(a.out, "locked.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {jpath}", flush=True)
    return out


def _ci(b: dict[str, Any], name: str) -> str:
    x = b[name]
    return f"{100 * x['diff']:+.2f} [{100 * x['lo']:+.2f}, {100 * x['hi']:+.2f}]"


def stage2_markdown(rep: dict[str, Any]) -> str:
    pv, ck, lk = rep["provenance"], rep["checks"], rep["lock"]
    L = ["# Success-damage curve, Stage 2: the reading (design 6)", ""] + [f"- {k}: {v}" for k, v in pv.items()] + [""]
    L += ["## Counts and checks", "",
          f"- Cases {ck['cases']}: dropped {ck['dropped']} (skipped {ck['skipped']}, case errors {ck['case_errors']}). Missing 0, "
          "first and second decision mismatches 0.", "",
          "## 1. Locked operating points (validation objects, locked before the test reading)", "",
          f"R4's validation success {_pc(lk['r4_success'])} %.", "",
          "| rule | knob | qualifies | validation success | validation irreversible |", "|---|---|---|---|---|"]
    for rule in rep["rules"]:
        v = lk[rule]
        L.append(f"| {rule} | {_knob(v['knob'])} | {'yes' if v['qualifies'] else 'no'} | {_pc(v['point']['success'])} | "
                 f"{_pc(v['point']['irr'], 2)} |")
    q, qn = rep["q1"], rep["q1_no_thin"]
    L += ["", "## 2. Q1 on test objects: R0 against R4 (differences in points, 95 % object bootstrap)", "",
          "| cases | success R0 − R4 | irreversible R0 − R4 | reading |", "|---|---|---|---|",
          f"| all test ({q['boot']['cases']}, {q['boot']['objects']} objects) | {_ci(q['boot'], 'success')} | {_ci(q['boot'], 'irr')} | "
          f"{q['reading']} |",
          f"| without thin plastic ({qn['boot']['cases']}, {qn['boot']['objects']} objects; beside, not the reading) | "
          f"{_ci(qn['boot'], 'success')} | {_ci(qn['boot'], 'irr')} | {qn['reading']} |", "",
          "## 3. Q2–Q4 on test objects: each rule against R0 at the locked points", "",
          "| rule | success rule − R0 | irreversible rule − R0 | reading | irreversible without thin plastic | reading without thin plastic |",
          "|---|---|---|---|---|---|"]
    for rule, v in rep["vs_r0"].items():
        L.append(f"| {rule} | {_ci(v['boot'], 'success')} | {_ci(v['boot'], 'irr')} | {v['reading']} | {_ci(v['no_thin']['boot'], 'irr')} | "
                 f"{v['no_thin']['reading']} |")
    L += ["", "## 4. Curves (rates over all episodes, give-ups included, %)", "",
          "| rule | knob | val success | val irr | val give-up | val second | test success | test irr | test give-up | test second | test irr without thin |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for rule in rep["rules"]:
        for pv_, pt, pn in zip(rep["curves"]["validation"][rule], rep["curves"]["test"][rule], rep["curves"]["test_no_thin"][rule]):
            L.append(f"| {rule} | {_knob(pv_['knob'])} | {_pc(pv_['success'])} | {_pc(pv_['irr'], 2)} | {_pc(pv_['give_up'])} | "
                     f"{_pc(pv_['second'])} | {_pc(pt['success'])} | {_pc(pt['irr'], 2)} | {_pc(pt['give_up'])} | {_pc(pt['second'])} | "
                     f"{_pc(pn['irr'], 2)} |")
    L += ["", "## 5. Calibration of P(irr) on the attempts at the locked points (test objects)", "",
          "| rule | attempts | predicted irreversible (sum) | observed | near the cap: attempts | predicted | observed |",
          "|---|---|---|---|---|---|---|"]
    for rule, c in rep["calibration"].items():
        nc = c.get("near_cap")
        L.append(f"| {rule} | {c['all']['attempts']} | {c['all']['predicted']:.2f} | {c['all']['observed']} | "
                 + (f"{nc['attempts']} | {nc['predicted']:.2f} | {nc['observed']} |" if nc else "– | – | – |"))
    L += ["", "## 6. Damage by object group at the locked points (test objects, irreversible %, cases)", "",
          "| group | " + " | ".join(rep["rules"]) + " |", "|---|" + "---|" * len(rep["rules"])]
    for g in sorted(next(iter(rep["groups"].values()))):
        L.append(f"| {g} | " + " | ".join(f"{_pc(rep['groups'][rule][g]['irr'], 2)} ({rep['groups'][rule][g]['n']})"
                                           for rule in rep["rules"]) + " |")
    return "\n".join(L) + "\n"


def stage2_figure(rep: dict[str, Any], path: str) -> None:
    """Simulated success against irreversible damage per rule (validation left, test right); locked points ringed, R4 a star."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), sharey=True)
    for ax, side in ((axes[0], "validation"), (axes[1], "test")):
        ends = []
        for rule in (r for r in rep["rules"] if r != "R4"):
            pts = rep["curves"][side][rule]
            x, y = [100 * p["irr"] for p in pts], [100 * p["success"] for p in pts]
            ax.plot(x, y, color=RULE_COLOURS[rule], lw=2, marker="o", ms=4, mec="white", mew=1, zorder=3, label=RULE_NAMES[rule])
            at = next(p for p in pts if p["knob"] == rep["lock"][rule]["knob"])
            ax.scatter([100 * at["irr"]], [100 * at["success"]], s=90, facecolors="none", edgecolors=RULE_COLOURS[rule], linewidths=1.5,
                       zorder=4)
            ends.append((x[-1], y[-1], rule))
        r4 = rep["curves"][side]["R4"][0]
        ax.scatter([100 * r4["irr"]], [100 * r4["success"]], marker="*", s=110, color="#52514e", edgecolors="white", linewidths=1,
                   zorder=5, label=RULE_NAMES["R4"])
        ax.annotate("R4", (100 * r4["irr"], 100 * r4["success"]), xytext=(5, -9), textcoords="offset points", fontsize=8, color="#52514e")
        ax.set_xlim(left=0.0)
        _end_labels(ax, ends)
        ax.set_xlabel("irreversible damage (% of episodes)")
        ax.set_title(f"{side} objects ({r4['n']} cases); rings: locked points", fontsize=10)
        ax.grid(True, color="#e6e5e0", linewidth=0.8, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    axes[0].set_ylabel("task success (% of episodes)")
    fig.suptitle("Stage 2, simulated episodes of up to 2 attempts (m2b2_v12b, class prior)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def stage2_summary(a: argparse.Namespace) -> dict[str, Any]:
    """The test reading (design 6) from finished Stage 2 runs and the committed lock: stage2_summary.{json,md} and
    stage2_curves.png in --out (refuses an existing json), wandb. Refuses a lock file that differs from the lock of the
    validation records."""
    from robots.mujoco_lab.tracking import log_run

    os.makedirs(a.out, exist_ok=True)
    jpath = os.path.join(a.out, "stage2_summary.json")
    if os.path.exists(jpath):
        raise SystemExit(f"{jpath} exists: pick a new --out")
    dirs = a.runs.split(",")
    val, cfg = load_stage2(dirs, ("validation",))
    rules = tuple(cfg["rules"])
    checks_val = stage2_checks(val, cfg, ("validation",))
    locked = json.load(open(a.locked))
    lk = lock(val, rules)
    if any(locked["lock"][rule]["knob"] != lk[rule]["knob"] for rule in rules):
        raise SystemExit("the lock file differs from the lock of the validation records")
    test, _ = load_stage2(dirs, ("test",))          # opened only after the lock file is checked
    checks_test = stage2_checks(test, cfg, ("test",))
    checks = {k: checks_val[k] + checks_test[k] for k in checks_val}
    key = {rule: lk[rule]["key"] for rule in rules}
    no_thin = [r for r in test if not r["thin"]]
    rep: dict[str, Any] = {"rules": rules, "lock": lk, "checks": checks}
    b, bn = paired_bootstrap(test, key["R0"], key["R4"]), paired_bootstrap(no_thin, key["R0"], key["R4"])
    rep["q1"], rep["q1_no_thin"] = {"boot": b, "reading": q1_reading(b)}, {"boot": bn, "reading": q1_reading(bn)}
    rep["vs_r0"] = {}
    for rule in (r for r in ("R1", "R2", "R3", "R5", "R6") if r in rules):     # R5 and R6 beside Q2-Q4, never a main reading
        b, bn = paired_bootstrap(test, key[rule], key["R0"]), paired_bootstrap(no_thin, key[rule], key["R0"])
        rep["vs_r0"][rule] = {"boot": b, "reading": vs_r0_reading(b), "no_thin": {"boot": bn, "reading": vs_r0_reading(bn)}}
    rep["curves"] = {"validation": _curves(val, rules), "test": _curves(test, rules), "test_no_thin": _curves(no_thin, rules)}
    rep["calibration"] = {rule: calibration(test, key[rule]) for rule in rules}
    groups = sorted({"|".join(stratum(r["object"], r)) for r in test})
    rep["groups"] = {rule: {g: rule_point([r for r in test if "|".join(stratum(r["object"], r)) == g], key[rule]) for g in groups}
                     for rule in rules}
    rep["provenance"] = {"runs": a.runs, "records_sha256": {f"{d} {side}": _sha(records_path(d, side)) for d in dirs for side in SIDES},
                         "code_revision": cfg["code_revision"], "stage1": cfg["stage1"],
                         "locked": f"{a.locked} (sha256 {_sha(a.locked)}, commit {a.lock_commit})",
                         "bootstrap": f"{BOOT_DRAWS} object draws, seed {BOOT_SEED}, 95 % percentile interval", "eps": EPS}
    rep["wandb"] = log_run("curve-stage2-summary", {"runs": a.runs, "rules": rules, "lock_commit": a.lock_commit},
                           {"checks": checks, "q1": {"reading": rep["q1"]["reading"], **{k: rep["q1"]["boot"][k] for k in ("success", "irr")}},
                            "vs_r0": {r: v["reading"] for r, v in rep["vs_r0"].items()}})
    with open(jpath, "w") as f:
        json.dump(rep, f, indent=1)
    with open(os.path.join(a.out, "stage2_summary.md"), "w") as f:
        f.write(stage2_markdown(rep))
    stage2_figure(rep, os.path.join(a.out, "stage2_curves.png"))
    print(f"wrote {a.out}", flush=True)
    return rep


def mean_row(x: Any) -> dict[str, Any]:
    """A menu row as the replay gate compares it: its id, action key and mean prediction."""
    from robots.mujoco_lab.evpi_props import action_key

    return {"id": x.id, "action": action_key(x.id, {x.id: x}), "p": dict(x.pred.p)}


def _same_destination(x: Any, destination: str) -> bool:
    return x.kind in ("commit", "lower_risk") and x.action.get("destination") == destination


class LabCase:
    """One case on the lab for simulate_case (evpi_props.run_case's draw and world): ``fresh`` resets at a seed, checks the
    fingerprint and rebuilds the first menu at the class prior on a fresh store under (STAGE1_CARE, A_MAIN), gated against
    Stage 1's rows (menu_matches); ``attempt`` runs one row; ``second_menu`` gives the reassessment's rows under the
    posterior (row_record), its status, ESS and guard flag."""

    def __init__(self, prims: Any, o: str, idx: int, seed: int, world: Any):
        from robots.mujoco_lab import calibrate as cal
        from robots.mujoco_lab.adapter import object_props

        self.prims, self.o, self.idx, self.seed = prims, o, int(idx), int(seed)
        env = prims.env
        self.theta, self.com_end, self.destination, self.hidden = cal.case_draw(self.seed, o, self.idx, object_props(o)["mass"], False,
                                                                                fragility=world)
        if world is not None:
            env.slosh_zeta, env.slosh_scale = world[1].slosh_zeta, world[1].slosh_scale
        env.classes = None if world is None else world[0]
        self.fixture = self.destination if self.destination in cal.MESH_IDS else None

    def fresh(self, reset_seed: int, rec1: dict[str, Any]) -> tuple[Any, str, bool]:
        from robots.mujoco_lab import calibrate as cal
        from robots.mujoco_lab import evpi_props as ev
        from robots.mujoco_lab.com_gap import fingerprint
        from robots.mujoco_lab.evpi_sim import _store
        from rpent.skills.pick_place.tools import _particles, _spread_kw, assess_pick

        prims, env = self.prims, self.prims.env
        env.reset(self.o, self.theta, seed=int(reset_seed), com_end=self.com_end, fixture=self.fixture, fill=self.hidden["fill"],
                  frag=self.hidden["frag"])
        at_rest = bool(dict(getattr(env, "reset_info", None) or {}).get("at_rest", True))
        if fingerprint(env) != rec1["fingerprint"]:
            return None, "the reset's fingerprint differs from Stage 1's", at_rest
        pose, n = env.pose_info(), _particles(prims)
        g = cal._stream(self.seed, self.o, self.idx, 1)
        b0 = prims.prior(pose, n, g, **_spread_kw(prims))
        state = copy.deepcopy(g.bit_generator.state)
        store = _store(env, b0)
        prims.abstain_cost, prims.rng = A_MAIN, ev._rng_at(state)
        assess_pick(object="object", destination=self.destination, care=STAGE1_CARE, primitives=prims, store=store)
        cache = list(store.get("pick_place", {}).values())
        if len(cache) > 1:
            return None, f"{len(cache)} cached assessments for one menu", at_rest
        menu = cache[0]["menu"] if cache else {}
        ok, why = menu_matches([mean_row(x) for x in menu.values() if _same_destination(x, self.destination)], rec1["rows"])
        return (store, "", at_rest) if ok else (None, f"replay gate: {why}", at_rest)

    def attempt(self, store: Any, row: str) -> dict[str, Any]:
        from rpent.skills.pick_place.tools import pick_and_place

        try:
            d = pick_and_place(object="object", destination=self.destination, care=STAGE1_CARE, primitives=self.prims, store=store,
                               option_id=row).data
        except Exception as exc:  # noqa: BLE001  a simulator fault ends this simulation as an error
            return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        if "outcome" not in d:
            return {"error": str(d.get("error") or d.get("reason"))}
        if d.get("restored") is not None:
            return {"error": "the lab restored the object (a recovery mode is on)"}
        return d

    def second_menu(self, store: Any, d: dict[str, Any]) -> dict[str, Any]:
        from rpent.skills.pick_place import menu as menu_mod

        a2 = d["assessment"]
        if "error" in a2:
            return {"status": "error", "error": str(a2["error"])[:300], "rows": []}
        if not a2.get("assessed", False):
            return {"status": "unreachable", "rows": []}
        cache = list(store.get("pick_place", {}).values())
        if len(cache) != 1:
            return {"status": "error", "error": f"{len(cache)} cached assessments (one expected)", "rows": []}
        b, rows = cache[0]["belief"], []
        failed = [str(f) for f in (b.flags or []) if str(f).startswith("belief update failed")]
        if failed:   # tools._update keeps the old belief when the update raises: the second menu would price the prior
            return {"status": "error", "error": failed[-1][:300], "rows": []}
        for x in cache[0]["menu"].values():
            if _same_destination(x, self.destination):
                _, per = menu_mod.estimate(self.prims.model, b, x._base, x._pose, x.action["force"], x._planned)
                rows.append(row_record(x, per, b.weights))
        return {"status": case_status(a2, rows), "rows": rows, "ess": float(b.effective_size()),
                "guard_flag": bool((a2.get("guard") or {}).get("flag"))}


def stage2_object(job: tuple[str, list[tuple[dict[str, Any], dict[str, str]]], dict[str, Any]]) -> dict[str, Any]:
    """One object's Stage 2 cases on its fold's bundle (stage1_object's sensor and world, no probe models): per case its
    record (stage2_record). A gate that fails ends the object with ``stop`` (the run stops)."""
    from robots.mujoco_lab import calibrate as cal
    from robots.mujoco_lab.evpi_sim import seeds_of
    from robots.mujoco_lab.prior import fragility_world

    o, items, s = job
    prims = cal._primitives(s["bundle"], None, None)
    if getattr(prims, "reset_s", None) is not None:
        raise RuntimeError("the curve scores the absorbing episode only (this bundle restores the object after damage)")
    probes, prims.probe_models = getattr(prims, "probe_models", None), {}
    prims.force_noise = cal.job_force_noise(None, prims.manifest)
    prims.env.force_noise = prims.force_noise
    world = fragility_world(prims.manifest)
    rules, n_seeds = tuple(s["rules"]), int(s["n_seeds"])
    out: dict[str, Any] = {"records": [], "stop": None}
    try:
        for rec1, dec in items:
            t0, idx, res = time.time(), int(rec1["idx"]), None
            if stage2_skip(rec1) is None:
                res = {"sims": [], "stop": None}
                if first_rows(dec, rules):
                    try:
                        lab = LabCase(prims, o, idx, int(s["seed"]), world)
                        res = simulate_case(lab, rec1, dec, rules, seeds_of(int(s["seed"]), o, idx)[:n_seeds])
                    except Exception as exc:  # noqa: BLE001  a case that raises is dropped from every rule (design 5.5)
                        res = {"sims": [], "stop": None, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            rec = dict(stage2_record(rec1, dec, rules, res, n_seeds), seconds=round(time.time() - t0, 2))
            out["records"].append(rec)
            n2 = sum(len(x["second"]) for x in rec["sims"])
            bad = sum(bool(x.get("error")) for x in rec["sims"]) + sum(bool(y.get("error")) for x in rec["sims"] for y in x["second"].values())
            print(f"{o} {idx}: {rec['seconds']:.1f} s, {len(rec['sims']) + n2} attempts, {bad} attempt errors"
                  + (f", skipped ({rec['skipped']})" if rec["skipped"] else "") + (f", error {rec['error']}" if rec["error"] else "")
                  + (f", STOP {rec['stop']}" if rec["stop"] else ""), file=sys.stderr, flush=True)
            if rec["stop"]:
                out["stop"] = f"{o} {idx}: {rec['stop']}"
                break
    finally:
        prims.probe_models = probes
    return out


def _stage2_cases(a: argparse.Namespace, recs1: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The run's Stage 1 records: --cases, or the shard's objects on the --sides, in (object, idx) order."""
    if a.cases:
        want = {(c.rsplit(":", 1)[0], int(c.rsplit(":", 1)[1])) for c in a.cases.split(",")}
        recs = [r for r in recs1 if (r["object"], int(r["idx"])) in want]
        if len(recs) != len(want):
            raise SystemExit(f"{len(want) - len(recs)} of the --cases are not in the Stage 1 records")
    else:
        sides = set(a.sides.split(","))
        objs = set(shard_objects([r["object"] for r in recs1 if r["side"] in sides], a.shard, a.shards))
        recs = [r for r in recs1 if r["object"] in objs and r["side"] in sides]
    return sorted(recs, key=lambda r: (r["object"], int(r["idx"])))


def stage2(a: argparse.Namespace) -> str:
    """The Stage 2 run (design 5.5): records_validation.jsonl and records_test.jsonl (one stage2_record per case, by
    object as each ends), run_config.json and run_metrics.json (times, counts and errors only; wandb when on). A gate that
    fails stops the run (exit 3). Returns --out."""
    import multiprocessing as mp

    from robots.mujoco_lab.estimation_check import _revision
    from robots.mujoco_lab.identify_fit import require_bundle_generator
    from robots.mujoco_lab.tracking import log_run

    os.makedirs(a.out, exist_ok=True)
    if any(os.path.exists(records_path(a.out, side)) for side in SIDES):
        raise SystemExit(f"{a.out}: a records file exists: pick a new --out")
    rules = tuple(a.rules.split(","))
    if any(r not in RULES for r in rules):
        raise SystemExit(f"unknown rules in {a.rules}")
    dirs = a.stage1.split(",")
    recs1, cfg1 = load_stage1(dirs)
    generator = require_bundle_generator(cfg1["bundle"])
    c = retry_continuation(json.load(gzip.open(a.retry))["rows"])
    V = lambda care, A: continuation_values(c, care, A)  # noqa: E731
    recs = _stage2_cases(a, recs1)
    by: dict[str, list[tuple[dict[str, Any], dict[str, str]]]] = defaultdict(list)
    for r in recs:
        by[r["object"]].append((r, decisions(r, V)))
    config = {**vars(a), **_revision(), "generator": generator, "bundle": cfg1["bundle"], "seed": cfg1["seed"],
              "stage1_code_revision": cfg1["code_revision"],
              "stage1_records_sha256": {d: _sha(os.path.join(d, "records.jsonl")) for d in dirs},
              "split_sha256": cfg1["split_sha256"], "retry_sha256": _sha(a.retry), "retry_continuation": c, "rules": rules,
              "n_seeds": a.seeds, "cases": [(r["object"], int(r["idx"])) for r in recs], "a_main": A_MAIN, "stage1_care": STAGE1_CARE,
              "r_grid": R_GRID, "delta_grid": DELTA_GRID, "quantile": QUANTILE}
    with open(os.path.join(a.out, "run_config.json"), "w") as f:
        json.dump(config, f, indent=1)
    jobs = [(o, items, {"bundle": os.path.join(cfg1["bundle"], "folds", f"fold_{items[0][0]['fold']}"), "seed": cfg1["seed"],
                        "rules": rules, "n_seeds": a.seeds}) for o, items in by.items()]
    t0, stop = time.time(), None
    outs = {side: open(records_path(a.out, side), "w") for side in SIDES}
    try:
        pool = None if a.workers == 1 else mp.get_context("spawn").Pool(a.workers)
        results = map(stage2_object, jobs) if pool is None else pool.imap_unordered(stage2_object, jobs)
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
    attempts = sum(len(r["sims"]) + sum(len(x["second"]) for x in r["sims"]) for r in done)
    metrics = {"cases": len(done), "of_cases": len(recs), "stop": stop, "skipped": sum(bool(r["skipped"]) for r in done),
               "dropped": sum(bool(r["dropped"]) for r in done), "case_errors": sum(bool(r["error"]) for r in done),
               "attempts": attempts, "wall_seconds": round(time.time() - t0, 1),
               "case_seconds": round(sum(r["seconds"] for r in done), 1),
               "case_seconds_max": max((r["seconds"] for r in done), default=0.0), "peak_rss_mb": _peak_rss_mb()}
    metrics["wandb"] = log_run("curve-stage2", config, metrics)
    with open(os.path.join(a.out, "run_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print(f"done: {json.dumps(metrics)}", flush=True)
    if stop:
        raise SystemExit(3)
    return a.out


def stage2_smoke(a: argparse.Namespace) -> dict[str, Any]:
    """The smoke test's report (times and counts, no rates): smoke_summary of the smoke runs and the full run's projection
    from the Stage 1 records (every non-skipped case's distinct first rows at seed 0)."""
    from robots.mujoco_lab.tracking import log_run

    os.makedirs(a.out, exist_ok=True)
    jpath = os.path.join(a.out, "smoke_summary.json")
    if os.path.exists(jpath):
        raise SystemExit(f"{jpath} exists: pick a new --out")
    recs = [json.loads(line) for d in a.runs.split(",") for side in SIDES for line in open(records_path(d, side))]
    cfg = json.load(open(os.path.join(a.runs.split(",")[0], "run_config.json")))
    rules = tuple(cfg["rules"])
    recs1, _ = load_stage1(cfg["stage1"].split(","))
    c = retry_continuation(json.load(gzip.open(cfg["retry"]))["rows"])
    V = lambda care, A: continuation_values(c, care, A)  # noqa: E731
    n_first = sum(len(first_rows(decisions(r, V), rules)) for r in recs1 if stage2_skip(r) is None)
    sm = dict(smoke_summary(recs), cases=len(recs), dropped=sum(bool(r["dropped"]) for r in recs),
              case_errors=sum(bool(r["error"]) for r in recs), stops=[r["stop"] for r in recs if r["stop"]],
              run_first_rows=n_first, run_cases=sum(stage2_skip(r) is None for r in recs1))
    sm["projection_cpu_h"] = projection(sm, n_first) if sm["seconds_per_first_row_seed0"] else None
    sm["wandb"] = log_run("curve-stage2-smoke", {"runs": a.runs, "rules": rules}, sm)
    with open(jpath, "w") as f:
        json.dump(sm, f, indent=1)
    print(json.dumps(sm, indent=1), flush=True)
    return sm


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
    ss = sub.add_parser("stage1-summary", help="the Stage 1 report from finished runs")
    ss.add_argument("--runs", required=True, help="comma-separated Stage 1 run dirs")
    ss.add_argument("--retry", required=True, help="the retry run (R1's continuation values)")
    ss.add_argument("--stage0", required=True, help="stage0_points.json")
    ss.add_argument("--e1", required=True, help="experiment 1's class records dir (shard_*/records.jsonl.gz)")
    ss.add_argument("--split-commit", required=True, help="the commit that holds split.json")
    ss.add_argument("--out", required=True)
    s2 = sub.add_parser("stage2", help="Stage 2: the episodes in the simulator (design 5.5)")
    s2.add_argument("--stage1", required=True, help="comma-separated Stage 1 run dirs")
    s2.add_argument("--retry", required=True, help="the retry run (R1's continuation values)")
    s2.add_argument("--out", required=True)
    s2.add_argument("--rules", default=",".join(SIM_RULES))
    s2.add_argument("--seeds", type=int, default=1, help="reset seeds per case (evpi_sim.seeds_of; 1 = the case's own)")
    s2.add_argument("--cases", default=None, help="comma-separated object:idx cases in place of the shard")
    s2.add_argument("--sides", default="validation,test")
    s2.add_argument("--shard", type=int, default=0)
    s2.add_argument("--shards", type=int, default=1)
    s2.add_argument("--workers", type=int, default=1)
    sk = sub.add_parser("stage2-smoke", help="the smoke test's times, counts and the full run's projection")
    sk.add_argument("--runs", required=True, help="comma-separated smoke run dirs")
    sk.add_argument("--out", required=True)
    sl = sub.add_parser("stage2-lock", help="the operating points from the validation records only")
    sl.add_argument("--runs", required=True, help="comma-separated Stage 2 run dirs")
    sl.add_argument("--out", required=True)
    sm = sub.add_parser("stage2-summary", help="the test reading (design 6) from finished runs and the committed lock")
    sm.add_argument("--runs", required=True, help="comma-separated Stage 2 run dirs")
    sm.add_argument("--locked", required=True, help="locked.json from stage2-lock, committed")
    sm.add_argument("--lock-commit", required=True, help="the commit that holds locked.json")
    sm.add_argument("--out", required=True)
    a = p.parse_args(argv)
    return {"stage0": stage0, "split": split, "stage1": stage1, "stage1-summary": stage1_summary, "stage2": stage2,
            "stage2-smoke": stage2_smoke, "stage2-lock": stage2_lock, "stage2-summary": stage2_summary}[a.cmd](a)


if __name__ == "__main__":
    main(sys.argv[1:])
