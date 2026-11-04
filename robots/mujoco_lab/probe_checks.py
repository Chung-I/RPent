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


"""Two checks on the milestone 4 sweep (``data/m4``), analysis only: no policy is rerun.

Check 1, the success contrast. The policy (probes allowed) against the never-probe arm on the 12 test cells (care κ 10
and 30 × abstain cost A 3 and 10 × probe time scale s 1, 0.25 and 0) at seeds 12 to 20, paired as ``sweep.pairs``
pairs them: the never-probe run at s = 1 pairs with every s (gate G1b), and a subset never-probe run (seeds 13 to 20)
covers the probe cases, so every other case pairs with itself (gate G1, rechecked here at seed 12, with a count of the
cases that probe later without a first probe recommendation). A pair counts when neither row has an error or an
artifact. Per pair: success (the last attempt placed_stable, which is ``ended == "placed"``; checked), irreversible
damage (``sweep._irreversible``: an irreversible attempt outcome, or a probe that threw or damaged the object), giving
up (``ended == "abstained"``) and a probe run. Paired differences policy − never-probe per cell and pooled over the
cells, on all counted pairs and on the pairs where the policy ran a probe.

Check 2, accuracy before and after a probe. policy_probe_one (one forced probe, then the policy) against
policy_no_probe at seed 12 and s = 1 for the 15 settings κ 0.5, 1, 3, 10, 30 × A 1, 3, 10, on the counted pairs where
probe_one ran a probe and both arms attempted. Each arm's first attempt: its declared outcome chances (rounded to 0.001
in the rows) against its own outcome, as the Brier score over the outcomes and the ECE of P(placed_stable) and of
P(irreversible) (10 equal bins). Primary: the pairs whose first attempts use the same option. Secondary: all probed
pairs, with the switchers' outcomes in each arm. The deployed calibration files cannot answer this: their probe_first
rows hold 0 (m2b), 0 (m2b2) and 1 (m2b2_v11c) probes.

Intervals are 95 % percentile intervals of one joint object bootstrap (2,000 draws): each draw gives every case of an
object, in every seed, cell and setting, the object's multiplicity, and every ratio is recomputed inside the draw.

    python -P -m robots.mujoco_lab.probe_checks success --dir D --out F --markdown M [--wandb]
    python -P -m robots.mujoco_lab.probe_checks accuracy --dir D --out F --markdown M [--wandb]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter
from typing import Any

import numpy as np

from robots.mujoco_lab import sweep, tracking
from rpent.skills.pick_place.evaluate import ece_from_sums
from rpent.skills.pick_place.evaluate import object_sums as ece_sums
from rpent.skills.pick_place.protocol import IRREVERSIBLE, OUTCOMES

CELLS = sweep.TEST_CELLS
SEEDS = tuple(range(sweep.BASE_SEED, sweep.BASE_SEED + 9))       # 12 to 20, the stage 1 and 2 seeds
ACC_SETTINGS = tuple((k, A, 1.0) for k in sweep.KAPPAS for A in sweep.ABSTAIN)
METRICS = ("success", "irreversible", "abstained", "probed")
MIN_SAME = 100                                                  # a setting is reported with this many same-option pairs
N_BOOT = sweep.N_BOOT


# ---------------------------------------------------------------- check 1

def success(row: dict[str, Any]) -> bool:
    """The task succeeded: the last attempt placed the object stably."""
    t = row.get("attempts") or []
    return bool(t) and t[-1].get("outcome") == "placed_stable"


def row_metrics(row: dict[str, Any]) -> np.ndarray:
    """(success, irreversible damage, gave up, ran a probe) of one row, as floats."""
    return np.array([success(row), sweep._irreversible(row), row.get("ended") == "abstained", bool(row.get("probes"))], dtype=float)


def paired_rows(a: dict[str, Any], b: dict[str, Any], subset: bool = False) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """The rows of run ``a`` with their rows in ``b``, as ``sweep.pairs`` pairs them: with ``subset``, a case of ``a``
    without a probe recommendation and without a row in ``b`` pairs with itself; any other missing row is an error."""
    rb = {sweep._key(r): r for r in b["rows"]}
    out = []
    for r in a["rows"]:
        k, probe = sweep._key(r), r.get("recommendation") == "probe"
        if k in rb:
            out.append((r, rb[k]))
        elif subset and not probe:
            out.append((r, r))
        else:
            raise ValueError(f"case {k} has no row in the other run" + (" (a probe case)" if probe else ""))
    return out


def object_sums(pairs: list[tuple[dict[str, Any], dict[str, Any]]], objects: list[str], probed_only: bool = False) -> np.ndarray:
    """Per object: the sums of each metric in arm a, then in arm b, then the pair count, over the counted pairs (and
    with ``probed_only``, the pairs where a ran a probe)."""
    index = {o: i for i, o in enumerate(objects)}
    S = np.zeros((len(objects), 2 * len(METRICS) + 1))
    for ra, rb in pairs:
        if not (sweep._ok(ra) and sweep._ok(rb)) or (probed_only and not ra.get("probes")):
            continue
        S[index[ra["object"]]] += np.concatenate([row_metrics(ra), row_metrics(rb), [1.0]])
    return S


def contrast(S: np.ndarray, W: np.ndarray) -> dict[str, Any]:
    """Per metric the rate in a, in b and the paired difference a − b, with joint-bootstrap intervals."""
    m, n = len(METRICS), S[:, -1]
    N = float(n.sum())
    out: dict[str, Any] = {"n": int(N), "metrics": {}}
    with np.errstate(invalid="ignore", divide="ignore"):
        WN = W @ n
        for j, q in enumerate(METRICS):
            da, db = (W @ S[:, j]) / WN, (W @ S[:, m + j]) / WN
            pa, pb = (float(S[:, j].sum() / N), float(S[:, m + j].sum() / N)) if N else (math.nan, math.nan)
            out["metrics"][q] = {"a": pa, "b": pb, "diff": pa - pb, "a_ci": sweep._ci(da), "b_ci": sweep._ci(db),
                                 "diff_ci": sweep._ci(da - db)}
    return out


def _objects(runs: list[dict[str, Any]]) -> list[str]:
    return sorted({r["object"] for run in runs for r in run["rows"]})


def success_check(runs: list[dict[str, Any]], n_boot: int = N_BOOT, seed: int = 0) -> dict[str, Any]:
    """Check 1 over the runs of the sweep directory (see the module)."""
    idx = sweep._index(runs)
    used_runs, plan = [], []
    for c in CELLS:
        for s in SEEDS:
            P, N = idx.get(("policy", *c, s)), idx.get(("policy_no_probe", c[0], c[1], 1.0, s))
            plan.append((c, s, P, N))
            used_runs += [x for x in (P, N) if x is not None]
    objects = _objects(used_runs)
    W = sweep.bootstrap_weights(objects, n_boot, np.random.default_rng(seed))
    checks: dict[str, Any] = {"missing": [], "g1_mismatches": {}, "later_probes": 0, "success_disagreements": 0,
                              "pairing_disagreements": 0}
    cells, width = {}, 2 * len(METRICS) + 1
    tot_all, tot_probed = np.zeros((len(objects), width)), np.zeros((len(objects), width))
    for c in CELLS:
        S_all, S_probed, used, dropped = np.zeros((len(objects), width)), np.zeros((len(objects), width)), [], 0
        for c2, s, P, N in plan:
            if c2 != c:
                continue
            if P is None or N is None:
                checks["missing"].append(f"{sweep._label(c)} seed {s}")
                continue
            sub = sweep._subset(N)
            pr, sp = paired_rows(P, N, subset=sub), sweep.pairs(P, N, subset=sub)
            counted = np.array([sweep._ok(x) and sweep._ok(y) for x, y in pr], dtype=bool)
            checks["pairing_disagreements"] += int((counted != sp["counted"]).sum()) + sum(
                x["object"] != o or int(x["idx"]) != int(i) for (x, _), o, i in zip(pr, sp["object"], sp["idx"]))
            if s == sweep.BASE_SEED:
                checks["g1_mismatches"][sweep._label(c)] = len(sweep.check_identical(P, N))
            checks["later_probes"] += sum(r.get("recommendation") != "probe" and bool(r.get("probes")) for r in P["rows"])
            checks["success_disagreements"] += sum((r.get("ended") == "placed") != success(r)
                                                   for r in P["rows"] + N["rows"] if sweep._ok(r))
            S_all += object_sums(pr, objects)
            S_probed += object_sums(pr, objects, probed_only=True)
            dropped += len(pr) - int(counted.sum())
            used.append(s)
        cells[sweep._label(c)] = {"seeds": used, "dropped": dropped, "all": contrast(S_all, W), "probed": contrast(S_probed, W)}
        tot_all += S_all
        tot_probed += S_probed
    return {"check": "success", "n_boot": n_boot, "seed": seed, "n_objects": len(objects), "checks": checks,
            "cells": cells, "pooled": {"all": contrast(tot_all, W), "probed": contrast(tot_probed, W)}}


# ---------------------------------------------------------------- check 2

def attempt_scores(t: dict[str, Any]) -> tuple[float, float, float, float, float]:
    """(Brier over the outcomes, declared P(placed_stable), placed_stable, declared P(irreversible), irreversible) of
    one attempt."""
    p, y = t["declared"]["p"], t["outcome"]
    brier = sum((float(p.get(k, 0.0)) - float(k == y)) ** 2 for k in OUTCOMES)
    return (brier, float(p.get("placed_stable", 0.0)), float(y == "placed_stable"),
            sum(float(p.get(k, 0.0)) for k in IRREVERSIBLE), float(y in IRREVERSIBLE))


def accuracy_pairs(F: dict[str, Any], N: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """The counted pairs where probe_one (F) ran a probe and both arms attempted: each arm's first attempt and whether
    they use the same option; with the counts of the probed pairs left out (no attempt in an arm)."""
    rows, left = [], Counter()
    for f, n in paired_rows(F, N):
        if not (sweep._ok(f) and sweep._ok(n)) or not f.get("probes"):
            continue
        if not f.get("attempts") or not n.get("attempts"):
            left["neither_attempted" if not (f.get("attempts") or n.get("attempts")) else
                 "probe_one_did_not_attempt" if not f.get("attempts") else "never_did_not_attempt"] += 1
            continue
        a, b = f["attempts"][0], n["attempts"][0]
        rows.append({"object": f["object"], "idx": int(f["idx"]), "probe_one": a, "never": b, "same": a.get("option") == b.get("option")})
    return rows, dict(left)


def accuracy_sums(rows: list[dict[str, Any]], objects: list[str]) -> dict[str, Any]:
    """Per object: the Brier sums of each arm, the ECE bin sums of P(placed_stable) and P(irreversible) per arm, and
    the pair counts."""
    index = {o: i for i, o in enumerate(objects)}
    inv = np.array([index[r["object"]] for r in rows], dtype=int)
    out: dict[str, Any] = {"n": np.bincount(inv, minlength=len(objects)).astype(float)}
    for arm, tag in (("probe_one", "a"), ("never", "b")):
        sc = np.array([attempt_scores(r[arm]) for r in rows], dtype=float).reshape(-1, 5)
        out[f"brier_{tag}"] = np.bincount(inv, weights=sc[:, 0], minlength=len(objects))
        out[f"placed_{tag}"] = ece_sums(sc[:, 1], sc[:, 2], inv, len(objects))[:2]
        out[f"irreversible_{tag}"] = ece_sums(sc[:, 3], sc[:, 4], inv, len(objects))[:2]
    return out


def accuracy_contrast(sums: dict[str, Any], W: np.ndarray) -> dict[str, Any]:
    """The Brier score and both ECEs in probe_one (a) and never-probe (b) and the paired difference a − b, with
    joint-bootstrap intervals."""
    n = sums["n"]
    N, one = float(n.sum()), np.ones((1, len(n)))
    out: dict[str, Any] = {"n": int(N)}
    with np.errstate(invalid="ignore", divide="ignore"):
        WN = W @ n
        da, db = (W @ sums["brier_a"]) / WN, (W @ sums["brier_b"]) / WN
        pa, pb = (float(sums["brier_a"].sum() / N), float(sums["brier_b"].sum() / N)) if N else (math.nan, math.nan)
        out["brier"] = {"a": pa, "b": pb, "diff": pa - pb, "a_ci": sweep._ci(da), "b_ci": sweep._ci(db), "diff_ci": sweep._ci(da - db)}
        for q in ("placed", "irreversible"):
            (SPa, SYa), (SPb, SYb) = sums[f"{q}_a"], sums[f"{q}_b"]
            ea, eb = ece_from_sums(W, SPa, SYa, n), ece_from_sums(W, SPb, SYb, n)
            pa, pb = (float(ece_from_sums(one, SPa, SYa, n)[0]), float(ece_from_sums(one, SPb, SYb, n)[0])) if N else (math.nan, math.nan)
            out[f"ece_{q}"] = {"a": pa, "b": pb, "diff": pa - pb, "a_ci": sweep._ci(ea), "b_ci": sweep._ci(eb), "diff_ci": sweep._ci(ea - eb)}
    return out


def _switchers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    for arm in ("probe_one", "never"):
        out[arm] = {"outcomes": dict(Counter(r[arm]["outcome"] for r in rows).most_common()),
                    "placed": float(np.mean([r[arm]["outcome"] == "placed_stable" for r in rows])) if rows else math.nan,
                    "irreversible": float(np.mean([r[arm]["outcome"] in IRREVERSIBLE for r in rows])) if rows else math.nan}
    return out


def accuracy_check(runs: list[dict[str, Any]], n_boot: int = N_BOOT, seed: int = 0) -> dict[str, Any]:
    """Check 2 over the runs of the sweep directory (see the module)."""
    idx = sweep._index(runs)
    plan = [(c, idx.get(("policy_probe_one", *c, sweep.BASE_SEED)), idx.get(("policy_no_probe", *c, sweep.BASE_SEED))) for c in ACC_SETTINGS]
    objects = _objects([x for _, F, N in plan for x in (F, N) if x is not None])
    W = sweep.bootstrap_weights(objects, n_boot, np.random.default_rng(seed))
    settings, missing, all_rows = {}, [], []
    for c, F, N in plan:
        if F is None or N is None:
            missing.append(sweep._label(c))
            continue
        rows, left = accuracy_pairs(F, N)
        same = [r for r in rows if r["same"]]
        settings[sweep._label(c)] = {"probed_pairs": len(rows), "left_out": left, "same": len(same), "switched": len(rows) - len(same),
                                     "reported": len(same) >= MIN_SAME,
                                     "primary": accuracy_contrast(accuracy_sums(same, objects), W),
                                     "secondary": accuracy_contrast(accuracy_sums(rows, objects), W),
                                     "switchers": _switchers([r for r in rows if not r["same"]])}
        all_rows += rows
    same = [r for r in all_rows if r["same"]]
    return {"check": "accuracy", "n_boot": n_boot, "seed": seed, "n_objects": len(objects), "min_same": MIN_SAME,
            "missing": missing, "settings": settings,
            "pooled": {"same": len(same), "switched": len(all_rows) - len(same),
                       "primary": accuracy_contrast(accuracy_sums(same, objects), W),
                       "secondary": accuracy_contrast(accuracy_sums(all_rows, objects), W),
                       "switchers": _switchers([r for r in all_rows if not r["same"]])}}


# ---------------------------------------------------------------- output

def _iv(x: dict[str, Any], key: str = "diff", d: int = 3) -> str:
    ci = x[f"{key}_ci"]
    return f"{x[key]:+.{d}f} [{ci[0]:+.{d}f}, {ci[1]:+.{d}f}]"


def markdown(s: dict[str, Any]) -> str:
    if s["check"] == "success":
        L = ["# Check 1: success with probes allowed against never probe (m4)", "",
             f"Paired policy − never-probe over counted pairs, 95 % joint object-bootstrap intervals ({s['n_boot']} draws, "
             f"{s['n_objects']} objects). Checks: {s['checks']}.", ""]
        for part, title in (("all", "all counted pairs"), ("probed", "pairs where the policy ran a probe")):
            L += [f"## {title}", "", "| cell | seeds | n | success policy | never | diff | irreversible diff | give-up diff | probe rate |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for name, C in list(s["cells"].items()) + [("pooled", {"seeds": [], **s["pooled"]})]:
                M = C[part]["metrics"]
                L.append(f"| {name} | {len(C['seeds']) or '-'} | {C[part]['n']} | {M['success']['a']:.3f} | {M['success']['b']:.3f} | "
                         f"{_iv(M['success'])} | {_iv(M['irreversible'])} | {_iv(M['abstained'])} | {M['probed']['a']:.3f} |")
            L.append("")
        return "\n".join(L)
    L = ["# Check 2: accuracy before and after a probe (m4 probe_one against never-probe, seed 12, s = 1)", "",
         f"Paired probe_one − never-probe on each arm's first attempt, 95 % joint object-bootstrap intervals ({s['n_boot']} draws, "
         f"{s['n_objects']} objects). Settings with at least {s['min_same']} same-option pairs, and the pool of all 15.", ""]
    for part, title in (("primary", "same-option pairs (primary)"), ("secondary", "all probed pairs (secondary)")):
        L += [f"## {title}", "", "| setting | n | Brier probe_one | never | diff | ECE P(placed) diff | ECE P(irreversible) diff |",
              "|---|---|---|---|---|---|---|"]
        for name, C in list(s["settings"].items()) + [("pooled", s["pooled"])]:
            if name != "pooled" and not C["reported"]:
                continue
            A = C[part]
            L.append(f"| {name} | {A['n']} | {A['brier']['a']:.4f} | {A['brier']['b']:.4f} | {_iv(A['brier'], d=4)} | "
                     f"{_iv(A['ece_placed'])} | {_iv(A['ece_irreversible'])} |")
        L.append("")
    L += ["## switchers (first attempts on different options)", "", "| setting | same | switched | placed probe_one | never | irreversible probe_one | never |",
          "|---|---|---|---|---|---|---|"]
    for name, C in list(s["settings"].items()) + [("pooled", s["pooled"])]:
        w = C["switchers"]
        L.append(f"| {name} | {C['same']} | {C['switched']} | {w['probe_one']['placed']:.3f} | {w['never']['placed']:.3f} | "
                 f"{w['probe_one']['irreversible']:.3f} | {w['never']['irreversible']:.3f} |")
    return "\n".join(L) + "\n"


def _flat(s: dict[str, Any]) -> dict[str, float]:
    out = {}
    if s["check"] == "success":
        for name, C in list(s["cells"].items()) + [("pooled", s["pooled"])]:
            for part in ("all", "probed"):
                for q, M in C[part]["metrics"].items():
                    out[f"{name}/{part}/{q}/diff"] = M["diff"]
                    out[f"{name}/{part}/{q}/a"] = M["a"]
    else:
        for name, C in list(s["settings"].items()) + [("pooled", s["pooled"])]:
            for part in ("primary", "secondary"):
                for q in ("brier", "ece_placed", "ece_irreversible"):
                    out[f"{name}/{part}/{q}/diff"] = C[part][q]["diff"]
    return out


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=("success", "accuracy"))
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--markdown", required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args(argv)
    runs = sweep.load_dir(a.dir)
    s = (success_check if a.check == "success" else accuracy_check)(runs, a.n_boot, a.seed)
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    s["code_revision"], s["dir"] = rev, os.path.realpath(a.dir)
    if a.wandb:
        s["wandb"] = tracking.log_run(f"probe_checks_{a.check}", {"dir": s["dir"], "n_boot": a.n_boot, "seed": a.seed,
                                                                   "code_revision": rev}, _flat(s))
    with open(a.out, "w") as f:
        json.dump(s, f, indent=1)
    with open(a.markdown, "w") as f:
        f.write(markdown(s))
    return s


if __name__ == "__main__":
    main(sys.argv[1:])
