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


"""The pre-stated test of the v12 P(success) spread check (daily-logs experiments/2026-11-05-v12-masscom-spread,
"Pre-stated test", written and committed before any run). An arm is one stage and prior of psuccess_spread's records.
Per arm, on P(placed_stable) of commit_std (a probe stage) or of the attempted row (the attempt stage):

- (a) R_com, the CoM share of the spread removed: [sum D_com(prior) - sum D_com(after)] / [sum SD(prior) - sum SD(after)],
  D_com = SD(belief) - SD(CoM at truth); beside it the driver split's CoM row after, 1 - sum SD(CoM at truth) / sum SD;
- (b) the Spearman over objects between the prior CoM share (1 - sum SD(CoM at truth, prior) / sum SD(prior)) and the
  share removed (1 - sum SD(after) / sum SD(prior)), each a ratio of the object's sums;
- (c) F = mean |dP| of the far cases - mean |dP| of the near cases, dP = mean P after - mean P before, far: the pinch
  at least FAR of the pivot from the true CoM; beside it the far cases' signed dP and the change in |P - P(every input
  at truth)|.

A contrast between two arms uses the objects both have, resampled jointly (the same draws for both arms; 95 %
percentile interval, N_BOOT draws). Above 0: supported; below 0: contradicted; holding 0: undecided.

    python -P -m robots.mujoco_lab.psuccess_prestated --arm NAME=DIR:STAGE:PRIOR[:measured] ... \\
        --contrast A,B ... [--scenes SCENES] --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Callable

import numpy as np

FAR = 0.15                 # far: |s - c_true| >= FAR of the pivot (set from v11c distances alone, before any run)
N_BOOT = 2000
STATS = ("r_com", "rho", "F")


def case_values(r: dict[str, Any]) -> dict[str, Any]:
    """One record's numbers: the SDs and means of P(placed_stable) at the prior and after its stage (the belief's and
    the CoM-at-truth variant's), P at every input's truth, and d, the pinch's distance from the true CoM."""
    st = r["stage"]
    c = (r["row"] if st == "attempt" else r["commit_std"])["placed"]
    s = r["s"] if st == "attempt" else r["s_std"]
    p0, p1 = float(c["belief"]["prior"]["mean"]), float(c["belief"][st]["mean"])
    return {"object": r["object"], "key": r["key"], "kind": r.get("kind"),
            "sd_prior": float(c["belief"]["prior"]["sd"]), "sd_after": float(c["belief"][st]["sd"]),
            "sdc_prior": float(c["com_true"]["prior"]["sd"]), "sdc_after": float(c["com_true"][st]["sd"]),
            "p_prior": p0, "p_after": p1, "dp": p1 - p0, "p_all_true": float(c["all_true"][st]["mean"]),
            "d": abs(float(s) - float(r["truth"]["com"]))}


def _col(V: list[dict[str, Any]], k: str) -> np.ndarray:
    return np.array([v[k] for v in V], dtype=float)


def _cw(V: list[dict[str, Any]], w: dict[str, float] | None) -> np.ndarray:
    """Each case's weight: its object's bootstrap multiplicity (1 without a draw)."""
    return np.ones(len(V)) if w is None else np.array([w.get(v["object"], 0.0) for v in V])


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den != 0 else float("nan")


def r_com(V: list[dict[str, Any]], w: dict[str, float] | None = None) -> float:
    cw = _cw(V, w)
    sp, sa, cp, ca = (_col(V, k) for k in ("sd_prior", "sd_after", "sdc_prior", "sdc_after"))
    return _ratio(cw @ (sp - cp) - cw @ (sa - ca), cw @ sp - cw @ sa)


def split_com_after(V: list[dict[str, Any]], w: dict[str, float] | None = None) -> float:
    cw = _cw(V, w)
    return 1.0 - _ratio(cw @ _col(V, "sdc_after"), cw @ _col(V, "sd_after"))


def object_shares(V: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """Per object (its sums over its cases): the prior CoM share and the share of the spread removed."""
    out: dict[str, list[float]] = {}
    for v in V:
        t = out.setdefault(v["object"], [0.0, 0.0, 0.0])
        t[0] += v["sd_prior"]
        t[1] += v["sdc_prior"]
        t[2] += v["sd_after"]
    return {o: (1.0 - c / p, 1.0 - a / p) for o, (p, c, a) in sorted(out.items()) if p > 0}


def spearman(V: list[dict[str, Any]], w: dict[str, float] | None = None) -> float:
    """Spearman's rho of the object shares, each object counted by its multiplicity."""
    from scipy.stats import spearmanr

    sh = object_shares(V)
    reps = [int(round(1 if w is None else w.get(o, 0.0))) for o in sh]
    x = np.repeat([v[0] for v in sh.values()], reps)
    y = np.repeat([v[1] for v in sh.values()], reps)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def far_near(V: list[dict[str, Any]], w: dict[str, float] | None = None) -> dict[str, Any]:
    cw, dp, d = _cw(V, w), _col(V, "dp"), _col(V, "d")
    far = d >= FAR
    gap = np.abs(_col(V, "p_after") - _col(V, "p_all_true")) - np.abs(_col(V, "p_prior") - _col(V, "p_all_true"))
    mean = lambda x, m: _ratio(cw[m] @ x[m], cw[m].sum())
    return {"n_far": int(far.sum()), "n_near": int((~far).sum()), "F": mean(np.abs(dp), far) - mean(np.abs(dp), ~far),
            "far_abs_dp": mean(np.abs(dp), far), "near_abs_dp": mean(np.abs(dp), ~far), "far_signed_dp": mean(dp, far),
            "truth_gap_change": mean(gap, np.ones(len(V), dtype=bool))}


STAT_FN: dict[str, Callable[..., float]] = {"r_com": r_com, "rho": spearman, "F": lambda V, w=None: far_near(V, w)["F"]}


def decision(ci: list[float]) -> str:
    lo, hi = ci
    if np.isnan(lo) or np.isnan(hi):
        return "undecided"
    return "supported" if lo > 0 else ("contradicted" if hi < 0 else "undecided")


def _interval(draws: np.ndarray) -> tuple[list[float], int]:
    ok = draws[np.isfinite(draws)]
    if not len(ok):
        return [float("nan"), float("nan")], int(len(draws))
    return [float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))], int(len(draws) - len(ok))


def _draws(objects: list[str], n_boot: int, seed: int) -> list[dict[str, float]]:
    from rpent.skills.pick_place.evaluate import object_weights

    W = object_weights(len(objects), n_boot, np.random.default_rng(seed))
    return [dict(zip(objects, row)) for row in W]


def arm(records: list[dict[str, Any]], n_boot: int = N_BOOT, seed: int = 0) -> dict[str, Any]:
    """One arm's statistics with 95 % object-bootstrap intervals over its own objects."""
    V = [case_values(r) for r in records]
    objects = sorted({v["object"] for v in V})
    W = _draws(objects, n_boot, seed)
    out: dict[str, Any] = {"n_cases": len(V), "n_objects": len(objects), "split_com_after": split_com_after(V),
                           "far_near": far_near(V)}
    for k in STATS:
        ci, bad = _interval(np.array([STAT_FN[k](V, w) for w in W]))
        out[k] = {"value": STAT_FN[k](V), "ci": ci, "nan_draws": bad, "decision": decision(ci)}
    return out


def contrast(a: list[dict[str, Any]], b: list[dict[str, Any]], n_boot: int = N_BOOT, seed: int = 0) -> dict[str, Any]:
    """Arm a - arm b on the objects both arms have, the same bootstrap draws for both."""
    Va, Vb = [case_values(r) for r in a], [case_values(r) for r in b]
    objects = sorted({v["object"] for v in Va} & {v["object"] for v in Vb})
    keep = set(objects)
    Va, Vb = [v for v in Va if v["object"] in keep], [v for v in Vb if v["object"] in keep]
    W = _draws(objects, n_boot, seed)
    out: dict[str, Any] = {"n_objects": len(objects), "n_cases": [len(Va), len(Vb)]}
    for k in STATS:
        fa, fb = STAT_FN[k](Va), STAT_FN[k](Vb)
        ci, bad = _interval(np.array([STAT_FN[k](Va, w) - STAT_FN[k](Vb, w) for w in W]))
        out[k] = {"a": fa, "b": fb, "diff": fa - fb, "ci": ci, "nan_draws": bad, "decision": decision(ci)}
    return out


def arm_records(spec: str) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """NAME=DIR:STAGE:PRIOR[:measured]: the records of one stage and prior (measured: the identify cases with a reading)."""
    from robots.mujoco_lab.psuccess_spread import load_records

    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    d, stage, prior = parts[:3]
    measured = len(parts) > 3 and parts[3] == "measured"
    recs = [r for r in load_records(d) if r["stage"] == stage and r["prior"] == prior]
    if measured:
        recs = [r for r in recs if r.get("kind") == "measured"]
    if not recs:
        raise SystemExit(f"arm {name}: no {stage} records with prior {prior} in {d}")
    return name, recs, {"dir": d, "stage": stage, "prior": prior, "measured_only": measured}


def orientation(scenes: str | None) -> dict[str, str]:
    """object -> flat or upright, from the renders' scenes.json (as estimation_check reads it)."""
    if not scenes:
        return {}
    sc = json.load(open(scenes))
    items = sc.items() if isinstance(sc, dict) else enumerate(sc)
    return {(v.get("object") or v.get("name") or k): ("flat" if v.get("flat") else "upright") for k, v in items if isinstance(v, dict)}


def _f(x: float) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x:+.3f}"


def markdown(s: dict[str, Any]) -> str:
    L = ["# Pre-stated test (v12 P(success) spread check)", "", f"Label: {s['label']}" if s.get("label") else "", "",
         f"Far: d >= {FAR}. Intervals: 95 % object bootstrap, {s['n_boot']} draws; a contrast uses the objects both arms "
         "have, resampled jointly.", "", "## Contrasts", "",
         "| contrast | objects | statistic | a | b | a - b | 95 % interval | decision |", "|---|---|---|---|---|---|---|---|"]
    for name, c in s["contrasts"].items():
        for k in STATS:
            v = c[k]
            L.append(f"| {name} | {c['n_objects']} | {k} | {_f(v['a'])} | {_f(v['b'])} | {_f(v['diff'])} | "
                     f"[{_f(v['ci'][0])}, {_f(v['ci'][1])}] | {v['decision']} |")
    L += ["", "## Arms", "", "| arm | cases | objects | R_com | rho | F | far / near | far signed dP | |P - P(truth)| change | "
          "CoM row after |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, a in s["arms"].items():
        fn = a["far_near"]
        iv = lambda k: f"{_f(a[k]['value'])} [{_f(a[k]['ci'][0])}, {_f(a[k]['ci'][1])}]"
        L.append(f"| {name} | {a['n_cases']} | {a['n_objects']} | {iv('r_com')} | {iv('rho')} | {iv('F')} | {fn['n_far']} / "
                 f"{fn['n_near']} | {_f(fn['far_signed_dp'])} | {_f(fn['truth_gap_change'])} | {_f(a['split_com_after'])} |")
    if s.get("orientation"):
        L += ["", "## Flat and upright objects apart (reported, not tests)", "", "| arm | objects | cases | R_com | rho | F |",
              "|---|---|---|---|---|---|"]
        for name, by in s["orientation"].items():
            for o, v in by.items():
                L.append(f"| {name} | {o} | {v['n_cases']} | {_f(v['r_com'])} | {_f(v['rho'])} | {_f(v['F'])} |")
    return "\n".join(L) + "\n"


def run(a: argparse.Namespace) -> dict[str, Any]:
    from robots.mujoco_lab import tracking

    os.makedirs(a.out, exist_ok=True)
    arms, cfg_arms = {}, {}
    for spec in a.arm:
        name, recs, info = arm_records(spec)
        arms[name], cfg_arms[name] = recs, {**info, "n_records": len(recs)}
    t0 = time.time()
    s: dict[str, Any] = {"label": a.label, "n_boot": a.n_boot, "seed": a.seed, "far": FAR,
                         "arms": {n: arm(r, a.n_boot, a.seed) for n, r in arms.items()}, "contrasts": {}}
    for c in a.contrast:
        x, y = c.split(",")
        s["contrasts"][f"{x} - {y}"] = contrast(arms[x], arms[y], a.n_boot, a.seed)
    orient = orientation(a.scenes)
    if orient:
        s["orientation"] = {}
        for n, recs in arms.items():
            V = [case_values(r) for r in recs]
            s["orientation"][n] = {o: {"n_cases": len(Vo), "r_com": r_com(Vo), "rho": spearman(Vo), "F": far_near(Vo)["F"]}
                                   for o in ("flat", "upright") for Vo in [[v for v in V if orient.get(v["object"]) == o]] if Vo}
    s["wall_s"] = time.time() - t0
    cfg = {"arms": cfg_arms, "contrasts": a.contrast, "scenes": a.scenes, "n_boot": a.n_boot, "seed": a.seed, "far": FAR,
           "label": a.label, "out": a.out}
    s["wandb"] = tracking.log_run("psuccess_prestated", cfg, {k: v for k, v in s.items() if k not in ("wandb", "orientation")})
    with open(os.path.join(a.out, "prestated.json"), "w") as f:
        json.dump({"config": cfg, **s}, f, indent=1)
    with open(os.path.join(a.out, "prestated.md"), "w") as f:
        f.write(markdown(s))
    print(markdown(s), flush=True)
    return s


def main(argv: list[str] | None = None) -> Any:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", action="append", required=True, help="NAME=DIR:STAGE:PRIOR[:measured] (psuccess_spread records)")
    p.add_argument("--contrast", action="append", default=[], help="A,B: arm A - arm B on their shared objects")
    p.add_argument("--scenes", default=None, help="scenes.json: flat and upright objects apart (reported, not tests)")
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default=None)
    p.add_argument("--out", required=True)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    main()
