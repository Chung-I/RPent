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


"""The ESS of the success-damage curve's second menus (Stage 2; daily-logs experiments/2026-11-07-success-damage-curve).
Added after the v12b item result, before any curve reading, by the coordinator's call of 7 Nov; not pre-registered.

The second menu prices its rows under the posterior after the first attempt (the v12b attempt update with the joint term).
On the v12b items that update often left an ESS below 100, so this module counts, per side and per rule at its locked knob,
the episodes that reach a second menu and how many of those menus have an ESS below ESS_APART. Counts only: it reads no
outcome after the second menu, splits no outcome by ESS, and changes no lock, margin or reading. It runs after
stage2-summary, on the records that the summary read.

    python -P -m robots.mujoco_lab.curve_ess --runs <the Stage 2 run dirs> --summary <stage2-summary's --out> --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from robots.mujoco_lab.curve import (ESS_APART, GIVE_UP_ROW, IRREVERSIBLE, SIDES, _kept, _knob, _sha, load_stage2,
                                     records_path)

TITLE = "Added after the v12b item result, before any curve reading (not pre-registered)"


def second_menu(dec: dict[str, str], sims: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    """The second menu that the rule and knob's episode (curve.episode) reaches on a case: None after a give-up at the first
    decision, or when the first attempt at the main seed placed the object or damaged it."""
    first = dec[key]
    if first == GIVE_UP_ROW:
        return None
    s = next(x for x in sims if x["seed_k"] == 0 and x["first"] == first)
    if s["label"] == "placed_stable" or s["label"] in IRREVERSIBLE:
        return None
    if s["menu2"] is None:
        raise ValueError(f"the first attempt of {first} gave {s['label']} without a second menu")
    return s["menu2"]


def _count(menus: list[dict[str, Any]], threshold: float) -> dict[str, int]:
    return {"second_menus": len(menus), "ess_low": sum(m.get("ess") is not None and m["ess"] < threshold for m in menus),
            "no_ess": sum(m.get("ess") is None for m in menus)}


def ess_counts(recs: list[dict[str, Any]], lk: dict[str, Any], rules: tuple[str, ...],
               threshold: float = ESS_APART) -> dict[str, Any]:
    """Per side, over the cases that are not dropped: per rule at its locked knob (lk[rule]["key"]), the episodes that reach a
    second menu, how many of those menus have an ESS below ``threshold``, and how many have none (an unreachable menu);
    and the side's distinct second menus (each simulated first row at the main seed)."""
    out: dict[str, Any] = {}
    for side in SIDES:
        rs = [r for r in _kept(recs) if r["side"] == side]
        per = {}
        for rule in rules:
            menus = [m for m in (second_menu(r["decisions"], r["sims"], lk[rule]["key"]) for r in rs) if m is not None]
            per[rule] = {"knob": lk[rule]["knob"], **_count(menus, threshold)}
        distinct = [s["menu2"] for r in rs for s in r["sims"] if s["seed_k"] == 0 and s["menu2"] is not None]
        out[side] = {"cases": len(rs), "rules": per, "distinct": _count(distinct, threshold)}
    return out


def markdown(rep: dict[str, Any]) -> str:
    th = f"{rep['threshold']:g}"
    L = [f"## {TITLE}", "",
         f"The second menu prices its rows under the posterior after the first attempt. This section counts the second menus "
         f"whose belief has an ESS below {th}. It changes no lock, no margin and no reading. It reports no outcome split by ESS.",
         "", f"- Runs: {rep['provenance']['runs']}", f"- Summary: {rep['provenance']['summary']}",
         f"- Lock: {rep['provenance']['locked']}",
         "- Counts over the cases that are not dropped, per rule at its locked knob: the episodes that reach a second menu, the "
         f"menus among them with an ESS below {th}, and the menus without an ESS (unreachable).", "",
         f"| side | rule | locked knob | second menus | ESS below {th} | no ESS |", "|---|---|---|---|---|---|"]
    for side in SIDES:
        c = rep["counts"][side]
        for rule, v in c["rules"].items():
            L.append(f"| {side} | {rule} | {_knob(v['knob'])} | {v['second_menus']} | {v['ess_low']} | {v['no_ess']} |")
        d = c["distinct"]
        L.append(f"| {side} | distinct second menus (any rule) | | {d['second_menus']} | {d['ess_low']} | {d['no_ess']} |")
    return "\n".join(L) + "\n"


def run(a: argparse.Namespace) -> dict[str, Any]:
    """ess.{json,md} in --out (refuses an existing json), wandb. Refuses to run before the summary, or on records other than
    the ones the summary read."""
    from robots.mujoco_lab.tracking import log_run

    spath = os.path.join(a.summary, "stage2_summary.json")
    if not os.path.exists(spath):
        raise SystemExit(f"{spath} is missing: the pre-registered summary runs first")
    jpath = os.path.join(a.out, "ess.json")
    if os.path.exists(jpath):
        raise SystemExit(f"{jpath} exists: pick a new --out")
    rep = json.load(open(spath))
    dirs = a.runs.split(",")
    want = rep["provenance"]["records_sha256"]
    have = {f"{d} {side}": _sha(records_path(d, side)) for d in dirs for side in SIDES}
    if have != want:
        raise SystemExit("the records differ from the ones the summary read")
    recs, cfg = load_stage2(dirs)
    rules = tuple(cfg["rules"])
    counts = ess_counts(recs, rep["lock"], rules)
    out = {"counts": counts, "threshold": ESS_APART, "title": TITLE,
           "provenance": {"runs": a.runs, "summary": f"{spath} (sha256 {_sha(spath)})", "locked": rep["provenance"]["locked"],
                          "code_revision": cfg.get("code_revision")}}
    out["wandb"] = log_run("curve-stage2-ess", {"runs": a.runs, "summary": spath, "threshold": ESS_APART},
                           {side: {rule: {k: v[k] for k in ("second_menus", "ess_low", "no_ess")} for rule, v in c["rules"].items()}
                            for side, c in counts.items()})
    os.makedirs(a.out, exist_ok=True)
    with open(jpath, "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(a.out, "ess.md"), "w") as f:
        f.write(markdown(out))
    print(f"wrote {a.out}", flush=True)
    return out


def main(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description="the ESS of the curve's second menus (counts only, not pre-registered)")
    p.add_argument("--runs", required=True, help="comma-separated Stage 2 run dirs, as stage2-summary read them")
    p.add_argument("--summary", required=True, help="stage2-summary's --out")
    p.add_argument("--out", required=True)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    main()
