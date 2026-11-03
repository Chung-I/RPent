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


"""W0 (2026-11-04): the decision value of an exact CoM at the first menu, from the records of ``first_menus run --exact``.

Priors: 1 generic (the generic mass and CoM), 2 exact_com (the generic mass, the exact CoM), 3 exact_mass (the exact mass,
the generic CoM), 4 class (the bundle's), 5 exact_both (the exact mass and CoM; a yardstick only). Exact: the case's
generator truth, the mass with log-sd 0.01 and the CoM fraction with sd 0.005 (vision_prior.EXACT_SD).

Per setting and prior, against prior 1:

* the share of cases whose recommendation (commit, probe, decline) and recommended row change (first_menus.change_rates);
* the expected-cost difference per case: the prior's first choice minus prior 1's, both priced on one menu, (ii) prior 5's
  (the answer) and (i) the class prior's; every row with a cost but HIDDEN_ROWS, decline at A (masscom_report.choice_of);

with 95 % object-bootstrap intervals (2,000 draws), and each split into objects laid flat and upright containers
(scenes.json). Then the objects whose cost falls most with the exact CoM (prior 2 against prior 1 on the answer's menu),
and the free proxy for the decision after the first attempt: prior 3 against prior 5 at the first menu (with the mass
known, what the exact CoM still changes).

    python -P -m robots.mujoco_lab.w0_report --dir DIR --scenes S [--wandb]
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from robots.mujoco_lab import tracking
from robots.mujoco_lab.first_menus import change_rates
from robots.mujoco_lab.masscom_report import HIDDEN_ROWS, _mean_ci, choice_of

N_BOOT = 2000
REF = "generic"
PRIORS = ("exact_com", "exact_mass", "class", "exact_both")
MENUS = (("ii", "exact_both"), ("i", "class"))
PROXY = ("exact_mass", "exact_both")
GROUPS = ("flat", "upright")


def _priced(r: dict[str, Any], menu: str) -> dict[str, float] | None:
    """The menu's row costs (HIDDEN_ROWS left out); None for a missing menu or one with an error."""
    m = r["menus"].get(menu)
    if m is None or "error" in m:
        return None
    return {k: float(v) for k, v in (m.get("costs") or {}).items() if v is not None and k not in HIDDEN_ROWS}


def paired_cost(records: list[dict[str, Any]], menu: str, ref: str = REF, n_boot: int = N_BOOT, seed: int = 0,
                group: dict[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """Per setting and prior other than ``ref``: each case's first choice minus ``ref``'s, both priced on ``menu``'s rows,
    the mean with a 95 % object-bootstrap interval, and with ``group`` (object -> label) the same per label. A case whose
    menu or either choice has an error is left out; a choice the menu has no row for is unmatched."""
    rng = np.random.default_rng(seed)
    acc: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"v": [], "obj": [], "left_out": 0, "unmatched": 0})
    for r in records:
        costs = _priced(r, menu)
        cr = choice_of(r["menus"][ref]) if ref in r["menus"] else None
        for p, m in r["menus"].items():
            if p == ref:
                continue
            a, c = acc[(r["setting"], p)], choice_of(m)
            if costs is None or cr is None or c is None:
                a["left_out"] += 1
            elif c not in costs or cr not in costs:
                a["unmatched"] += 1
            else:
                a["v"].append(costs[c] - costs[cr])
                a["obj"].append(r["object"])
    out: dict[str, dict[str, Any]] = defaultdict(dict)
    for (setting, p), a in sorted(acc.items()):
        block: dict[str, Any] = {"n": len(a["v"]), "n_left_out": a["left_out"], "n_unmatched": a["unmatched"]}
        if a["v"]:
            block.update(_mean_ci(a["v"], a["obj"], rng, n_boot))
            if group is not None:
                block["groups"] = {}
                for g in sorted({group[o] for o in a["obj"]}):
                    ix = [i for i, o in enumerate(a["obj"]) if group[o] == g]
                    block["groups"][g] = {"n": len(ix), **_mean_ci([a["v"][i] for i in ix], [a["obj"][i] for i in ix], rng, n_boot)}
        out[setting][p] = block
    return dict(out)


def object_gains(records: list[dict[str, Any]], menu: str = "exact_both", prior: str = "exact_com",
                 ref: str = REF) -> list[tuple[str, dict[str, Any]]]:
    """Per object, most gain first: per setting the mean gain (``ref``'s choice's cost minus ``prior``'s, on ``menu``), its
    cases and its changed choices; the mean gain over its settings; its cases' mean true CoM fraction and mean distance
    from the middle (the records' truth). Cases left out or unmatched in paired_cost are skipped."""
    per: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(lambda: {"g": [], "changes": Counter()}))
    truths: dict[tuple[str, int], dict[str, Any]] = {}
    for r in records:
        if r.get("truth"):
            truths[(r["object"], r["idx"])] = r["truth"]
        costs = _priced(r, menu)
        cr, cp = (choice_of(r["menus"][k]) if k in r["menus"] else None for k in (ref, prior))
        if costs is None or cr is None or cp is None or cr not in costs or cp not in costs:
            continue
        s = per[r["object"]][r["setting"]]
        s["g"].append(costs[cr] - costs[cp])
        if cr != cp:
            s["changes"][f"{cr}->{cp}"] += 1
    out = []
    for o, by in per.items():
        settings = {st: {"n": len(v["g"]), "gain": float(np.mean(v["g"])), "changes": dict(v["changes"])} for st, v in sorted(by.items())}
        fr = np.asarray([t["com_frac"] for (oo, _), t in truths.items() if oo == o], dtype=float)
        out.append((o, {"mean_gain": float(np.mean([v["gain"] for v in settings.values()])), "settings": settings,
                        "com_frac_mean": float(fr.mean()) if fr.size else None,
                        "com_off_middle": float(np.abs(fr - 0.5).mean()) if fr.size else None}))
    return sorted(out, key=lambda x: -x[1]["mean_gain"])


def truth_check(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The exact centre (the generator truth) against the sim's, over the cases: the largest relative mass gap and the
    largest CoM-fraction gap; and the cases whose reset did not reach rest."""
    cases = {(r["object"], r["idx"]): r for r in records}
    t = [r["truth"] for r in cases.values() if r.get("truth")]
    return {"cases": len(cases), "with_truth": len(t), "not_at_rest": sum(1 for r in cases.values() if not r.get("at_rest", True)),
            "max_mass_rel_gap": max((abs(x["sim_mass"] / x["mass"] - 1.0) for x in t), default=None),
            "max_com_frac_gap": max((abs(x["sim_com_frac"] - x["com_frac"]) for x in t), default=None)}


def _ci(x: dict[str, Any], key: str = "mean", d: int = 3) -> str:
    if not x or key not in x:
        return "–"
    return f"{x[key]:+.{d}f} [{x['ci'][0]:+.{d}f}, {x['ci'][1]:+.{d}f}]"


def _rate(x: dict[str, Any]) -> str:
    return f"{x['rate']:.3f} [{x['ci'][0]:.3f}, {x['ci'][1]:.3f}]" if x else "–"


def report(d: str, scenes: str, n_boot: int = N_BOOT, seed: int = 0) -> tuple[dict[str, Any], str]:
    from robots.mujoco_lab.names import object_name

    recs = [json.loads(line) for line in open(os.path.join(d, "records.jsonl"))]
    group = {o: ("flat" if v["flat"] else "upright") for o, v in json.load(open(scenes)).items()}
    out: dict[str, Any] = {"dir": d, "n_records": len(recs), "truth_check": truth_check(recs)}
    out["rates"] = change_rates(recs, n_boot, seed, ref=REF)
    out["rates_by_group"] = {g: change_rates([r for r in recs if group[r["object"]] == g], n_boot, seed, ref=REF) for g in GROUPS}
    out["cost"] = {tag: paired_cost(recs, menu, REF, n_boot, seed + 1 + i, group=group) for i, (tag, menu) in enumerate(MENUS)}
    out["proxy_rates"] = change_rates(recs, n_boot, seed + 3, ref=PROXY[1])
    out["proxy_cost"] = {tag: paired_cost(recs, menu, PROXY[1], n_boot, seed + 4 + i, group=group) for i, (tag, menu) in enumerate(MENUS)}
    gains = object_gains(recs, "exact_both", "exact_com", REF)
    out["com_gain_objects"] = [{"object": o, "name": object_name(o, "v11"), "group": group[o], **v} for o, v in gains]
    settings = list(out["rates"])
    tc = out["truth_check"]
    lines = ["# W0: the decision value of an exact CoM at the first menu", "",
             f"Records: `{d}` ({len(recs)} rows; {tc['cases']} cases, {tc['not_at_rest']} whose reset did not reach rest, kept as in "
             "the earlier first-menu runs). Exact centre = the generator truth; against the sim's: the largest relative mass gap "
             f"{tc['max_mass_rel_gap']:.2e}, the largest CoM-fraction gap {tc['max_com_frac_gap']:.2e}.", "",
             "## Recommendations changed against prior 1 (generic)", "",
             "| setting | prior | n | recommendation changed | recommended row changed | flat: recommendation changed | upright: recommendation changed |",
             "|---|---|---|---|---|---|---|"]
    for st in settings:
        for p in PRIORS:
            v = out["rates"][st].get(p)
            if not v:
                continue
            fl, up = (out["rates_by_group"][g].get(st, {}).get(p, {}).get("recommendation") for g in GROUPS)
            lines.append(f"| {st} | {p} | {v['n']} | {_rate(v['recommendation'])} | {_rate(v['option'])} | {_rate(fl)} | {_rate(up)} |")
    for tag, menu in MENUS:
        lines += ["", f"## Expected-cost difference per case against prior 1, priced ({tag}) on the {menu} menu", "",
                  "Below 0: the prior's first choice costs less than the generic prior's on that menu.", "",
                  "| setting | prior | n (unmatched, left out) | all | flat | upright |", "|---|---|---|---|---|---|"]
        for st in settings:
            for p in PRIORS:
                v = out["cost"][tag].get(st, {}).get(p)
                if not v:
                    continue
                gs = v.get("groups", {})
                lines.append(f"| {st} | {p} | {v['n']} ({v['n_unmatched']}, {v['n_left_out']}) | {_ci(v)} | "
                             f"{_ci(gs.get('flat', {}))} ({gs.get('flat', {}).get('n', 0)}) | {_ci(gs.get('upright', {}))} ({gs.get('upright', {}).get('n', 0)}) |")
    lines += ["", "## Free proxy for the decision after the first attempt: prior 3 (exact mass) against prior 5 (exact mass and CoM)", "",
              "| setting | n | recommendation changed | recommended row changed | cost (ii) on the exact_both menu | cost (i) on the class menu |",
              "|---|---|---|---|---|---|"]
    for st in settings:
        v = out["proxy_rates"].get(st, {}).get(PROXY[0])
        if not v:
            continue
        c2, c1 = (out["proxy_cost"][tag].get(st, {}).get(PROXY[0], {}) for tag, _ in MENUS)
        lines.append(f"| {st} | {v['n']} | {_rate(v['recommendation'])} | {_rate(v['option'])} | {_ci(c2)} | {_ci(c1)} |")
    lines += ["", "## The objects whose cost falls most with the exact CoM (prior 2 against prior 1, on the exact_both menu)", "",
              "Gain: the generic choice's cost minus the exact-CoM choice's (above 0: the exact CoM saves cost).", "",
              "| object | name | group | mean gain | per setting: gain (cases) | changed choices | mean true CoM fraction | mean distance from the middle |",
              "|---|---|---|---|---|---|---|---|"]
    for g in out["com_gain_objects"][:10]:
        per = "; ".join(f"{st} {v['gain']:+.3f} ({v['n']})" for st, v in g["settings"].items())
        ch = "; ".join(f"{st}: {json.dumps(v['changes'])}" for st, v in g["settings"].items() if v["changes"])
        cf = "–" if g["com_frac_mean"] is None else f"{g['com_frac_mean']:.3f}"
        co = "–" if g["com_off_middle"] is None else f"{g['com_off_middle']:.3f}"
        lines.append(f"| {g['object']} | {g['name']} | {g['group']} | {g['mean_gain']:+.3f} | {per} | {ch or '–'} | {cf} | {co} |")
    return out, "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", required=True, help="a first_menus run --exact output folder")
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args(argv)
    out, md = report(a.dir, a.scenes, a.n_boot, a.seed)
    with open(os.path.join(a.dir, "w0.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(a.dir, "w0.md"), "w") as f:
        f.write(md)
    print(md, flush=True)
    if a.wandb:
        flat: dict[str, Any] = {}
        for st, block in out["rates"].items():
            for p, v in block.items():
                flat[f"{st}/{p}/recommendation_changed"] = v["recommendation"]["rate"]
        for tag, by in out["cost"].items():
            for st, block in by.items():
                flat.update({f"{st}/{p}/cost_{tag}": v["mean"] for p, v in block.items() if v.get("n")})
        for tag, by in out["proxy_cost"].items():
            for st, block in by.items():
                if block.get(PROXY[0], {}).get("n"):
                    flat[f"{st}/proxy/cost_{tag}"] = block[PROXY[0]]["mean"]
        cfg = {k: v for k, v in json.load(open(os.path.join(a.dir, "run_config.json"))).items() if k != "tables"}
        out["wandb"] = tracking.log_run("w0_report", {**cfg, "w0_dir": a.dir}, flat)
        with open(os.path.join(a.dir, "w0.json"), "w") as f:
            json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    main()
