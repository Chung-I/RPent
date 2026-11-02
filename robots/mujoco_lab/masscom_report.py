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


"""Part B of the mass/CoM estimation check: the priors side by side on held-out objects (each object through the m2b2 fold
bundle that never saw it).

* Accuracy: Part A's table (estimation_check summary) per prior, at the prior and after the first attempt.
* Downstream: the prior predictive of each attempt row's action against its label (Brier over the outcomes, the ECE of
  P(irreversible)), from the same summaries.
* First menus (first_menus records): the change rate against the class prior, and each prior's first choice priced on the
  class prior's menu of the same case and setting (its regret: the choice's expected cost minus that menu's cheapest row).
  The class prior is privileged (it knows each object's nominal mass and CoM), so this regret is the decision cost of
  replacing it, as the class model prices it; it is not a regret against the truth.
* Paired against the generic prior: each other prior's Brier minus the generic prior's on the same attempt rows, and its
  regret minus the generic prior's on the same first-menu cases: what the image adds over no image.

    python -P -m robots.mujoco_lab.masscom_report --part-a class=DIR --part-a generic=DIR [--part-a NAME=DIR ...]
        [--menus DIR] --out DIR [--wandb]
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any

import numpy as np

from robots.mujoco_lab import tracking
from robots.mujoco_lab.first_menus import change_rates
from rpent.skills.pick_place.evaluate import object_weights
from rpent.skills.pick_place.protocol import OUTCOMES

N_BOOT = 2000
STAGES = ("prior", "attempt", "weigh", "push")
HIDDEN_ROWS = ("inboard_place",)   # shown but never chosen: it sets the object down elsewhere (menu.py; e1_expected)


def choice_of(m: dict[str, Any]) -> str | None:
    """The row a menu's rule runs: the recommended row, or decline; None for a menu with an error."""
    if "error" in m:
        return None
    return "decline" if m.get("recommendation") == "decline" else m.get("recommended_option")


def _mean_ci(values: list[float], objects: list[str], rng: np.random.Generator, n_boot: int) -> dict[str, Any]:
    v = np.asarray(values, dtype=float)
    objs, inv = np.unique(np.asarray(objects), return_inverse=True)
    W = object_weights(len(objs), n_boot, rng)
    boot = (W @ np.bincount(inv, weights=v, minlength=len(objs))) / (W @ np.bincount(inv, minlength=len(objs)))
    return {"mean": float(v.mean()), "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def regret_vs_class(records: list[dict[str, Any]], n_boot: int = N_BOOT, seed: int = 0) -> dict[str, dict[str, Any]]:
    """Per setting and prior other than class: each case's first choice priced on the class prior's menu (every row
    with a cost but HIDDEN_ROWS; decline at A), minus that menu's cheapest row: the mean with a 95 % object-bootstrap
    interval, the class rule's own regret on the same cases (its probe margin and guard), and ``diff``, the two paired
    by case (the swap's cost). A case with an error under either prior is left out; a choice that is not a row of the
    class menu is counted as unmatched. ``vs_generic`` (each prior but class and generic): its regret minus the generic
    prior's on the cases both keep, from a second random stream (seed + 1), so the other intervals do not move."""
    rng = np.random.default_rng(seed)
    acc: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: {"v": [], "own": [], "obj": [], "case": [], "left_out": 0, "unmatched": 0})
    for r in records:
        mc = r["menus"]["class"]
        costs = {k: float(v) for k, v in (mc.get("costs") or {}).items() if v is not None and k not in HIDDEN_ROWS}
        own = choice_of(mc)
        for p, m in r["menus"].items():
            if p == "class":
                continue
            a = acc[(r["setting"], p)]
            c = choice_of(m)
            if own is None or c is None or own not in costs:
                a["left_out"] += 1
            elif c not in costs:
                a["unmatched"] += 1
            else:
                best = min(costs.values())
                a["v"].append(costs[c] - best)
                a["own"].append(costs[own] - best)
                a["obj"].append(r["object"])
                a["case"].append((r["object"], r["idx"]))
    out: dict[str, dict[str, Any]] = defaultdict(dict)
    for (setting, p), a in sorted(acc.items()):
        block = {"n": len(a["v"]), "n_left_out": a["left_out"], "n_unmatched": a["unmatched"]}
        if a["v"]:
            block.update(_mean_ci(a["v"], a["obj"], rng, n_boot))
            block["class_own"] = float(np.mean(a["own"]))
            block["diff"] = _mean_ci(list(np.subtract(a["v"], a["own"])), a["obj"], rng, n_boot)   # the swap's cost, case by case
        out[setting][p] = block
    rng_g = np.random.default_rng(seed + 1)
    for (setting, p), a in sorted(acc.items()):
        g = acc.get((setting, "generic"))
        if p == "generic" or g is None:
            continue
        gv = dict(zip(g["case"], g["v"]))
        pairs = [(v - gv[c], c[0]) for v, c in zip(a["v"], a["case"]) if c in gv]
        if pairs:
            out[setting][p]["vs_generic"] = {"n": len(pairs), **_mean_ci([x for x, _ in pairs], [o for _, o in pairs], rng_g, n_boot)}
    return dict(out)


def _attempt_brier(records: list[dict[str, Any]]) -> dict[tuple[str, Any, int], tuple[float, str, float]]:
    """(object, key, occurrence) -> (the Brier over OUTCOMES, the label, the true mass) of each attempt row with a prior
    predictive. Keys repeat across the corpus files; an object's rows all come from one fold's job, in its order."""
    out, seen = {}, defaultdict(int)
    for r in records:
        if r.get("stage") != "attempt" or not r.get("p_prior"):
            continue
        ok = (r["object"], r.get("key"))
        k = (*ok, seen[ok])
        seen[ok] += 1
        pp = np.array([float(r["p_prior"].get(o, 0.0)) for o in OUTCOMES])
        y = np.array([1.0 if r["label"] == o else 0.0 for o in OUTCOMES])
        out[k] = (float(np.sum((pp - y) ** 2)), r["label"], float(r["prior"]["mass"]["truth"]))
    return out


def paired_brier(a: list[dict[str, Any]], b: list[dict[str, Any]], n_boot: int = N_BOOT, seed: int = 0) -> dict[str, Any]:
    """Run a's Brier minus run b's on the same attempt rows (paired by object, key and occurrence), with a 95 %
    object-bootstrap interval. Raises ValueError when a pair's label or true mass differs (the rows do not pair)."""
    A, B = _attempt_brier(a), _attempt_brier(b)
    common = sorted(set(A) & set(B), key=str)
    for k in common:
        if A[k][1] != B[k][1] or not np.isclose(A[k][2], B[k][2]):
            raise ValueError(f"rows do not pair at {k}: {A[k][1:]} against {B[k][1:]}")
    out: dict[str, Any] = {"n": len(common), "n_only_a": len(A) - len(common), "n_only_b": len(B) - len(common)}
    out.update(_mean_ci([A[k][0] - B[k][0] for k in common], [k[0] for k in common], np.random.default_rng(seed), n_boot))
    return out


def _cell(c: dict[str, Any], d: int = 3) -> str:
    return f"{c['median']:.{d}f} [{c['ci'][0]:.{d}f}, {c['ci'][1]:.{d}f}]"


def _cov(c: dict[str, Any]) -> str:
    return f"{c['coverage']:.2f} [{c['ci'][0]:.2f}, {c['ci'][1]:.2f}]"


def _iv(x: dict[str, Any], k: str = "value", d: int = 4) -> str:
    return f"{x[k]:.{d}f} [{x['ci'][0]:.{d}f}, {x['ci'][1]:.{d}f}]"


def compare(part_a: dict[str, str], menus: str | None, n_boot: int = N_BOOT, seed: int = 0) -> tuple[dict[str, Any], str]:
    """The priors side by side: Part A's cells and the downstream scores from each prior's summary, and, with ``menus``,
    the first-menu change rates and the regret on the class menu."""
    priors = {}
    for name, d in part_a.items():
        s = json.load(open(os.path.join(d, "summary.json")))
        priors[name] = {"stages": s["stages"], "downstream": s["downstream"], "dir": d}
    out: dict[str, Any] = {"priors": priors, "menus": None}
    lines = ["# Mass and CoM priors on held-out objects", "",
             "Accuracy (Part A's cells): posterior-mean error, median [95 % object-bootstrap CI]; coverage of the 80 % interval. "
             "Mass: relative error; μ: absolute; CoM: fraction of the pivot length (and cm).", "",
             "| prior | stage | mass error | mass coverage | μ error | μ coverage | CoM error | CoM error (cm) | CoM coverage |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, p in priors.items():
        for st in STAGES:
            if st not in p["stages"]:
                continue
            b = p["stages"][st]
            cm = _cell(b["com"]["err_cm"], 2) if "err_cm" in b["com"] else "−"
            lines.append(f"| {name} | {st} | {_cell(b['mass']['err'])} | {_cov(b['mass']['coverage'])} | {_cell(b['mu']['err'])} | "
                         f"{_cov(b['mu']['coverage'])} | {_cell(b['com']['err'])} | {cm} | {_cov(b['com']['coverage'])} |")
    lines += ["", "Downstream: each attempt row's action predicted from the prior's particles, against its label.", "",
              "| prior | rows (objects) | Brier | ECE irreversible | ECE placed_stable | mean P(irreversible) | irreversible rate |",
              "|---|---|---|---|---|---|---|"]
    for name, p in priors.items():
        d = p["downstream"]
        lines.append(f"| {name} | {d['n']} ({d['n_objects']}) | {_iv(d['brier'])} | {_iv(d['ece_irreversible'])} | "
                     f"{_iv(d['ece_placed'])} | {d['mean_p_irreversible']:.4f} | {d['rate_irreversible']:.4f} |")
    if menus is not None:
        recs = [json.loads(line) for line in open(os.path.join(menus, "records.jsonl"))]
        rates, regret = change_rates(recs, n_boot, seed), regret_vs_class(recs, n_boot, seed)
        out["menus"] = {"dir": menus, "n_records": len(recs), "rates": rates, "regret": regret}
        lines += ["", "First menus against the class prior's (same case, setting and menu stream). Regret: the prior's first "
                  "choice priced on the class prior's menu, minus that menu's cheapest row (class own: the class rule's own).", "",
                  "| setting | prior | n | recommendation changed | recommended row changed | regret on the class menu | class own "
                  "| prior − class own | unmatched |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for setting, block in rates.items():
            for p, v in block.items():
                g = regret.get(setting, {}).get(p, {})
                reg = f"{g['mean']:+.3f} [{g['ci'][0]:+.3f}, {g['ci'][1]:+.3f}]" if g.get("n") else "−"
                own = f"{g['class_own']:+.3f}" if g.get("n") else "−"
                dif = f"{g['diff']['mean']:+.3f} [{g['diff']['ci'][0]:+.3f}, {g['diff']['ci'][1]:+.3f}]" if g.get("n") else "−"
                lines.append(f"| {setting} | {p} | {v['n']} | {_iv(v['recommendation'], 'rate', 3)} | {_iv(v['option'], 'rate', 3)} | "
                             f"{reg} | {own} | {dif} | {g.get('n_unmatched', 0)} |")
    out["vs_generic"] = {}
    gfile = os.path.join(part_a["generic"], "records.jsonl") if "generic" in part_a else None
    if gfile and os.path.exists(gfile):
        g = [json.loads(line) for line in open(gfile)]
        for name, d in part_a.items():
            f = os.path.join(d, "records.jsonl")
            if name not in ("class", "generic") and os.path.exists(f):
                out["vs_generic"][name] = {"brier": paired_brier([json.loads(line) for line in open(f)], g, n_boot, seed)}
    reg_g = [(st, p, v["vs_generic"]) for st, block in ((out["menus"] or {}).get("regret") or {}).items()
             for p, v in block.items() if "vs_generic" in v]
    if out["vs_generic"] or reg_g:
        lines += ["", "Paired against the generic prior: what the image adds over no image (below 0: better than generic). "
                  "Brier on the same attempt rows; regret on the same first-menu cases, both priced on the class menu.", ""]
        if out["vs_generic"]:
            lines += ["| prior | rows | Brier − generic |", "|---|---|---|"]
            for name, v in out["vs_generic"].items():
                b = v["brier"]
                lines.append(f"| {name} | {b['n']} | {b['mean']:+.4f} [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}] |")
        if reg_g:
            lines += ["", "| setting | prior | cases | regret − generic |", "|---|---|---|---|"]
            lines += [f"| {st} | {p} | {v['n']} | {v['mean']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] |" for st, p, v in reg_g]
    return out, "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--part-a", action="append", required=True, help="NAME=DIR: an estimation_check run with its summary; repeatable")
    ap.add_argument("--menus", default=None, help="a first_menus run directory")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args(argv)
    out, md = compare(dict(v.split("=", 1) for v in a.part_a), a.menus, a.n_boot, a.seed)
    os.makedirs(a.out, exist_ok=True)
    if a.wandb:
        flat = {f"{n}/downstream/{k}": p["downstream"][k]["value"] for n, p in out["priors"].items()
                for k in ("brier", "ece_irreversible")}
        for setting, block in ((out["menus"] or {}).get("regret") or {}).items():
            flat.update({f"{setting}/{p}/regret": g["mean"] for p, g in block.items() if g.get("n")})
        out["wandb"] = tracking.log_run("masscom_report", {"part_a": a.part_a, "menus": a.menus, "n_boot": a.n_boot}, flat)
    with open(os.path.join(a.out, "compare.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(a.out, "compare.md"), "w") as f:
        f.write(md)
    print(md, flush=True)
    return out


if __name__ == "__main__":
    main()
