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

"""The help ladder's inputs (2026-11-04; throwaway branch ladder-pilot, off e1-v11; the registered ladder follows the
design document). Each rung adds one part to the rung below:

* A, the raw history: the actions with their grip, motion and time; after a probe its raw reading, after an attempt its
  outcome and the pad readouts (e1.render_result as the no-menu planner reads it);
* B, + the posterior: the skill's belief about the hidden properties: mass, friction, the centre of mass along the long
  axis and P(shifted toward one end), and, where the object has them, its damage thresholds, stiffness and fill;
* C, + the outcome predictions: each attempt row's probabilities of the eleven outcomes, each probe row's result kinds
  and P(thrown);
* D, + the action values: each row's expected cost at the true setting (the menu at the setting's kappa, A and recovery,
  as the oracle rule sees it), decline at A, and each probe row's value of information.

No rung shows the skill's recommendation, its decline reason or its guard; inboard_place is hidden, as in E1. The values
are the one-step rule's own, so the policy that always picks the lowest D value makes the oracle rule's first choice,
except where the guard declines a cheaper row ("guard"), where a probe row is the cheapest but beats the best row by
less than EVOI_Z Monte Carlo errors ("probe_margin"), or where two rows show the same (rounded) value ("tie"). Any other
mismatch ("other") is a ladder bug. No language model runs in ``inputs``, ``check`` or ``render``.

    python -P -m robots.mujoco_lab.ladder inputs --cases CASES.json --settings k100_A3_abs,k30_A3_abs,k100_A10_abs \\
        --crossfit robots/mujoco_lab/data/m2b2_v11c/folds --workers 8 --out DIR
    python -P -m robots.mujoco_lab.ladder check --dir DIR [--expected GAP_EXPECTED_DIR] [--wandb]
    python -P -m robots.mujoco_lab.ladder render --dir DIR --object OBJ --idx N --setting S
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from typing import Any

import numpy as np

from robots.mujoco_lab import e1, e1_expected, e1_gap
from rpent.skills.pick_place.features import DAMAGE_PARTICLE_KEYS
from rpent.skills.pick_place.protocol import IRREVERSIBLE, RECOVERABLE

RUNGS = ("A", "B", "C", "D")
SETTINGS = ("k100_A3_abs", "k30_A3_abs", "k100_A10_abs")       # the user-approved settings (2026-11-04)
POSTERIOR_KEYS = ("mass", "mu", "com_frac") + DAMAGE_PARTICLE_KEYS
TOL = 1e-9


# ---------------------------------------------------------------- settings and cases

def parse_settings(names: str) -> dict[str, dict[str, Any]]:
    """The named settings of the gap grid (e1_gap.grid: k{kappa}_A{A}_{abs|rec20}), in the order given."""
    grid = e1_gap.grid()
    out = {}
    for name in (x.strip() for x in names.split(",")):
        if not name:
            continue
        if name not in grid:
            raise ValueError(f"unknown setting {name!r}: the gap grid has {', '.join(grid)}")
        out[name] = dict(grid[name])
    return out


def load_cases(path: str) -> list[dict[str, Any]]:
    """The cases of a gap cases file ({"seed", "cases": [[object, idx], ...]}) or of a JSON list of {"object", "idx",
    "seed"} records (a "setting" kept when present), in file order."""
    data = json.load(open(path))
    if isinstance(data, dict):
        seed = int(data["seed"])
        return [{"object": str(o), "idx": int(i), "seed": seed} for o, i in data["cases"]]
    return [dict(r, idx=int(r["idx"]), seed=int(r["seed"])) for r in data]


def pick_smoke(never: dict[str, Any], n: int, plan_seed: int, exclude: set, settings) -> list[dict[str, Any]]:
    """The pilot's smoke cases: n cases of an H3 never run's cohort spread over the objects (e1_gap.pick_cases), none of
    ``exclude`` ((seed, object, idx)), each under one setting in turn (``settings`` cycled in the pick's order)."""
    spec = e1_gap.pick_cases(never, n, plan_seed, exclude)
    names = list(settings)
    return [{"object": o, "idx": int(i), "seed": int(spec["seed"]), "setting": names[k % len(names)]}
            for k, (o, i) in enumerate(spec["cases"])]


# ---------------------------------------------------------------- B, C and D

def _wquantile(v: np.ndarray, w: np.ndarray, q: float) -> float:
    """ParticleBelief.quantile's rule on (v, w)."""
    order = np.argsort(v, kind="stable")
    cw = np.cumsum(w[order])
    i = int(np.searchsorted(cw, q - 1e-12, side="left"))
    return float(v[order][min(i, len(v) - 1)])


def posterior(belief: Any) -> dict[str, Any]:
    """B: the weighted median and 80 % interval (p10, p90) of each hidden property the particles carry (POSTERIOR_KEYS;
    a key with no finite value is left out; "share" is the weight of the finite values when it is below 1),
    P(CoM shifted), the effective particles, the observations and the belief's flags."""
    arr, w = belief.arrays(), np.asarray(belief.weights, dtype=float)
    out: dict[str, Any] = {}
    for key in POSTERIOR_KEYS:
        if key not in arr:
            continue
        v = np.asarray(arr[key], dtype=float)
        ok = np.isfinite(v)
        share = float(w[ok].sum())
        if not ok.any() or share <= 0:
            continue
        if ok.all():
            out[key] = {"median": belief.quantile(key, 0.5), "p10": belief.quantile(key, 0.1), "p90": belief.quantile(key, 0.9)}
            continue
        ww = w[ok] / share
        out[key] = {"median": _wquantile(v[ok], ww, 0.5), "p10": _wquantile(v[ok], ww, 0.1), "p90": _wquantile(v[ok], ww, 0.9),
                    "share": share}
    out["p_com_shifted"] = belief.p_com_shifted()
    out["ess"] = belief.effective_size()
    out["n_observations"] = len(belief.history)
    out["flags"] = list(belief.flags)
    return out


def shown(a: dict[str, Any]) -> list[dict[str, Any]]:
    """The rows an agent sees, decline left out: every row but e1.HIDDEN_ROWS, in the menu's order."""
    return [o for o in a.get("options") or [] if o["kind"] != "decline" and o["id"] not in e1.HIDDEN_ROWS]


def predictions(a: dict[str, Any]) -> list[dict[str, Any]]:
    """C: per shown row its kind, action and time; an attempt row's outcome probabilities and its text ("what"), a probe
    row's result kinds and P(thrown). No cost and no value of information."""
    out = []
    for o in shown(a):
        r: dict[str, Any] = {"id": o["id"], "kind": o["kind"], "action": dict(o["action"]), "time_s": float(o["cost_s"])}
        if o["kind"] == "probe":
            r.update(p_kind=dict(o["probe"]["p_kind"]), p_thrown=o["probe"]["p_thrown"])
        else:
            r.update(p=dict(o["p"]), what=o.get("reason"))
        out.append(r)
    return out


def values(a: dict[str, Any]) -> dict[str, Any]:
    """D: each shown row's expected cost and decline's (A), as e1_expected.choosable gives them ("rows"), and each probe
    row's value of information, its Monte Carlo error and the expected cost of the best row after it ("probes")."""
    probes = {o["id"]: {k: o["probe"][k] for k in ("evoi", "evoi_se", "expected_cost_after")} for o in shown(a) if o["kind"] == "probe"}
    return {"rows": e1_expected.choosable(a), "probes": probes}


def rule_choice(a: dict[str, Any]) -> str:
    """The one-step rule's choice on menu ``a``: decline, else its recommended row (the best row or a probe row)."""
    return "decline" if a.get("recommendation") == "decline" else a.get("recommended_option")


def lowest_value_check(a: dict[str, Any]) -> dict[str, Any]:
    """The policy that picks the lowest D value (the first such row in the menu's order) against the rule's choice; a
    mismatch's reason: "tie" (the rule's row shows the lowest value too), "guard" (the guard declined a cheaper row),
    "probe_margin" (a probe row is the lowest, but the rule probes only when it beats the cheaper of the best row and A by
    EVOI_Z errors) or "other" (a ladder bug)."""
    rows = values(a)["rows"]
    lo = min(rows.values())
    ties = [k for k, v in rows.items() if v - lo <= TOL]
    lowest, rule = ties[0], rule_choice(a)
    guard = bool((a.get("guard") or {}).get("flag"))
    reason = None
    if lowest != rule:
        kind = {o["id"]: o["kind"] for o in a.get("options") or []}
        if rule in ties:
            reason = "tie"
        elif rule == "decline" and guard:
            reason = "guard"
        elif kind.get(lowest) == "probe" and kind.get(rule) != "probe":
            reason = "probe_margin"
        else:
            reason = "other"
    return {"lowest": lowest, "rule": rule, "ties": ties if len(ties) > 1 else [], "match": lowest == rule, "reason": reason,
            "lowest_value": lo, "rule_value": rows.get(rule), "guard_flag": guard}


# ---------------------------------------------------------------- the rung views

def _interval(e: dict[str, Any], nd: int, unit: str = "") -> str:
    u = f" {unit}" if unit else ""
    return f"{e['median']:.{nd}f}{u} (80 % interval {e['p10']:.{nd}f} to {e['p90']:.{nd}f}{u})"


def belief_text(B: dict[str, Any]) -> str:
    """B in words: the hidden properties with their medians and 80 % intervals (a property the object does not have is
    left out: no fill on an empty or unfillable object)."""
    parts = []
    if "mass" in B:
        parts.append("mass " + _interval(B["mass"], 3, "kg"))
    if "mu" in B:
        parts.append("friction " + _interval(B["mu"], 2))
    if "com_frac" in B:
        parts.append("centre of mass at " + _interval(B["com_frac"], 2) + " of the long axis (0 and 1 are its ends)")
    if B.get("p_com_shifted") is not None:
        parts.append(f"P(centre of mass shifted toward one end) {B['p_com_shifted']:.2f}")
    if "f_damage_n" in B:
        parts.append("damage force (a pad force above it crushes or bruises the object) " + _interval(B["f_damage_n"], 0, "N"))
    if "k_n_per_mm" in B:
        parts.append("stiffness " + _interval(B["k_n_per_mm"], 2, "N/mm"))
    if "v_bruise_ms" in B:
        parts.append("bruise speed (an impact faster than it bruises the object) " + _interval(B["v_bruise_ms"], 2, "m/s"))
    if "v_break_ms" in B:
        parts.append("break speed (an impact faster than it breaks the object) " + _interval(B["v_break_ms"], 2, "m/s"))
    if "fill" in B and B["fill"]["p90"] > 0:
        parts.append("fill (the share of the container filled with liquid) " + _interval(B["fill"], 2))
    if "spill_angle_deg" in B:
        parts.append("spill angle (a tilt above it spills the liquid) " + _interval(B["spill_angle_deg"], 0, "deg"))
    obs = int(B.get("n_observations") or 0)
    parts.append(f"{obs} observation{'s' if obs != 1 else ''} so far")
    return "; ".join(parts)


def prediction_text(r: dict[str, Any]) -> str:
    if r["kind"] == "probe":
        kinds = ", ".join(f"{k} {float(x):.2f}" for k, x in sorted(r["p_kind"].items(), key=lambda kv: -float(kv[1] or 0)))
        return f"result kinds {kinds}; P(thrown) {e1._f(r['p_thrown'])}"
    p = r["p"]
    irr, rec = sum(float(p[k]) for k in IRREVERSIBLE), sum(float(p[k]) for k in RECOVERABLE)
    return (f"P(placed_stable) {e1._f(p['placed_stable'])}; irreversible {irr:.2f} ({e1._top(p, IRREVERSIBLE)}); recoverable "
            f"{rec:.2f} ({e1._top(p, RECOVERABLE)})")


def value_text(r: dict[str, Any], D: dict[str, Any]) -> str:
    cost = D["rows"][r["id"]]
    if r["kind"] != "probe":
        return f"expected cost {e1._f(cost)}"
    v = D["probes"][r["id"]]
    return (f"value of information {e1._f(v['evoi'])} (Monte Carlo error {e1._f(v['evoi_se'])}); expected cost {e1._f(cost)} "
            f"(its time and risk plus the best row's expected cost after it, {e1._f(v['expected_cost_after'])})")


def render_state(rung: str, inputs: dict[str, Any], destination: str, rng: np.random.Generator) -> tuple[str, list[str]]:
    """(text, option ids) of one decision point at ``rung``: from B the belief, then the shown rows (C's) in a random
    order (``rng``) with their action and time, from C their predictions, from D their values; decline last."""
    if rung not in RUNGS:
        raise ValueError(f"unknown rung {rung!r}: one of {RUNGS}")
    k = RUNGS.index(rung)
    rows = list(inputs["C"])
    rows = [rows[i] for i in rng.permutation(len(rows))]
    ids = [r["id"] for r in rows] + ["decline"]
    lines = []
    if k >= 1:
        lines.append("The skill's belief about the object's hidden properties (median and 80 % interval): "
                     + belief_text(inputs["B"]) + ".")
    lines.append({0: "Actions (in random order):", 1: "Actions (in random order):",
                  2: "Actions (in random order) with the skill's outcome predictions:",
                  3: "Actions (in random order) with the skill's outcome predictions and expected costs:"}[k])
    for r in rows:
        what = e1._row_action({"kind": r["kind"], "action": r["action"], "reason": r.get("what")}, destination)
        line = f"- {r['id']} ({r['kind']}): {what}; about {r['time_s']:.0f} s."
        if k >= 2:
            line += "\n  " + prediction_text(r) + ("; " + value_text(r, inputs["D"]) if k >= 3 else "") + "."
        lines.append(line)
    tail = f"; cost {e1._f(inputs['D']['rows']['decline'])}" if k >= 3 else ""
    lines.append(f"- decline: abstain; the task ends without the object placed{tail}.")
    return "\n".join(lines), ids


# ---------------------------------------------------------------- the first decisions on the lab

def belief_of(store: dict[str, Any]) -> Any:
    """The case's belief in the skill tools' store (one object per case)."""
    beliefs = store.get("beliefs") or {}
    if len(beliefs) != 1:
        raise ValueError(f"the store holds {len(beliefs)} beliefs, not one")
    return next(iter(beliefs.values()))["belief"]


def first_record(a: dict[str, Any], store: dict[str, Any]) -> dict[str, Any]:
    """One first decision's record: B, C, D, the rule's fields and the lowest-value check; or why there is no menu."""
    if "error" in a or not a.get("assessed", False):
        return {"assessed": False, "reason": a.get("error") or a.get("reason")}
    return {"assessed": True, "B": posterior(belief_of(store)), "C": predictions(a), "D": values(a),
            "rule": {"recommendation": a.get("recommendation"), "recommended_option": a.get("recommended_option"),
                     "best_option": a.get("best_option"), "without_probe": a.get("without_probe"),
                     "guard_flag": bool((a.get("guard") or {}).get("flag")), "choice": rule_choice(a)},
            "check": lowest_value_check(a)}


def first_job(item: tuple[dict[str, Any], dict[str, Any]]) -> list[dict[str, Any]]:
    """One case's first decisions: per setting (the case's own "setting", else every setting of the run) a reset to the
    case and the true first menu (e1_gap.rule_config("oracle", ...), as the oracle rule's first menu)."""
    from robots.mujoco_lab import calibrate

    case, cfg = item
    prims = e1_expected._prims(cfg, case["object"])
    names = [case["setting"]] if case.get("setting") else list(cfg["settings"])
    out = []
    for name in names:
        t0 = time.time()
        conf = e1_gap.rule_config("oracle", cfg["settings"][name])
        prims.abstain_cost, prims.menu_recovery = conf["abstain_cost"], conf["menu_recovery"]
        prims.reset_s, prims.reset_price = conf["lab_recovery"]
        destination, reset_info, truth = e1.reset_case(prims, case["object"], int(case["idx"]), int(case["seed"]))
        tools = e1.LabTools(prims, destination, conf["care"])
        try:
            rec = first_record(tools.assess(), tools.store)
        except Exception as exc:  # noqa: BLE001  one bad case must not stop the run
            rec = {"assessed": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
        artifact = calibrate.case_artifact({"probes": [], "attempts": []}, reset_info)
        out.append({"object": case["object"], "idx": int(case["idx"]), "seed": int(case["seed"]), "setting": name,
                    "destination": destination, "truth": truth, "reset": reset_info, "reset_artifact": bool(artifact),
                    **rec, "seconds": round(time.time() - t0, 2)})
    prims.menu_recovery = None
    print(f"{case['object']} {case['idx']}: {len(out)} first decisions, {sum(r['seconds'] for r in out):.0f}s", file=sys.stderr,
          flush=True)
    return out


def _key(r: dict[str, Any]) -> tuple[str, int, int, str]:
    return r["object"], int(r["idx"]), int(r["seed"]), r["setting"]


def run_inputs(cases: list[dict[str, Any]], settings: dict[str, dict[str, Any]], crossfit: str, workers: int, out_dir: str,
               cases_path: str) -> str:
    """Every case's first decisions into ``out_dir``/inputs.jsonl (resuming: a (case, setting) with a record is skipped;
    a directory of another bundle, generator or setting list is an error); returns the path."""
    from robots.mujoco_lab.calibrate import code_revision
    from robots.mujoco_lab.corpus import lab_generator

    generator = e1.run_generator(out_dir)
    e1.check_bundle(out_dir, crossfit, generator)
    for held in sorted(p for p in os.listdir(out_dir) if p.startswith("run_config_")) if os.path.isdir(out_dir) else []:
        c = json.load(open(os.path.join(out_dir, held)))
        if c.get("settings") != settings:
            raise ValueError(f"{out_dir} holds a run of other settings ({held}); give a new directory")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"run_config_{time.strftime('%Y%m%dT%H%M%S')}.json"), "w") as f:
        json.dump({"cases": cases_path, "n_cases": len(cases), "settings": settings, "crossfit": crossfit,
                   "generator": lab_generator(), "workers": workers, "code": code_revision()}, f, indent=1)
    path = os.path.join(out_dir, "inputs.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {_key(json.loads(line)) for line in f if line.strip()}
    cfg = {"crossfit": crossfit, "folds": json.load(open(os.path.join(crossfit, "folds.json"))), "settings": settings}
    todo = []
    for c in cases:
        names = [c["setting"]] if c.get("setting") else list(settings)
        if any((c["object"], c["idx"], c["seed"], n) not in done for n in names):
            todo.append((c, cfg))
    print(f"ladder inputs: {len(todo)} of {len(cases)} cases to run", file=sys.stderr, flush=True)
    with open(path, "a") as f:
        def write(parts) -> None:
            for part in parts:
                for r in part:
                    if _key(r) not in done:
                        f.write(json.dumps(r, default=str) + "\n")
                f.flush()

        if workers <= 1:
            write(map(first_job, todo))
        else:
            with mp.get_context("spawn").Pool(workers) as pool:
                write(pool.imap_unordered(first_job, todo))
    return path


# ---------------------------------------------------------------- the check

def load_inputs(d: str) -> list[dict[str, Any]]:
    with open(os.path.join(d, "inputs.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def expected_oracle(d: str) -> dict[tuple, dict[str, Any]]:
    """The gap rescore's oracle records (e1_expected gap: records.jsonl[.gz]) by (object, idx, seed, setting)."""
    p = os.path.join(d, "records.jsonl")
    opener = (lambda: open(p)) if os.path.exists(p) else (lambda: gzip.open(p + ".gz", "rt"))
    with opener() as f:
        recs = [json.loads(line) for line in f if line.strip()]
    return {_key(r): r for r in recs if r["arm"] == "oracle"}


def check(recs: list[dict[str, Any]], expected: dict[tuple, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Per setting: the first decisions, those with no menu, the lowest-value policy's matches and its mismatches by
    reason (with the cases), B and C the same under every setting of a case (they do not depend on kappa or A), and,
    with the gap rescore's oracle records, the cases whose true first menu (rows, costs, recommendation) or recorded
    first choice differs from this run's."""
    by: dict[str, Any] = {}
    for r in recs:
        s = by.setdefault(r["setting"], {"n": 0, "no_menu": 0, "match": 0, "mismatch": Counter(), "cases": [], "reset_artifact": 0,
                                         "expected": {"compared": 0, "menu_differs": [], "choice_differs": []}})
        s["n"] += 1
        s["reset_artifact"] += bool(r.get("reset_artifact"))
        if not r.get("assessed"):
            s["no_menu"] += 1
            continue
        c = r["check"]
        if c["match"]:
            s["match"] += 1
        else:
            s["mismatch"][c["reason"]] += 1
            s["cases"].append({"case": f"{r['object']}/{r['idx']}", "reason": c["reason"], "lowest": c["lowest"],
                               "rule": c["rule"], "lowest_value": c["lowest_value"], "rule_value": c["rule_value"]})
        if expected is not None and _key(r) in expected:
            e = expected[_key(r)]
            s["expected"]["compared"] += 1
            tf = e.get("true_first") or {}
            if tf.get("rows") != r["D"]["rows"] or tf.get("recommendation") != r["rule"]["recommendation"]:
                s["expected"]["menu_differs"].append(f"{r['object']}/{r['idx']}")
            if (e.get("choices") or [None])[0] != r["rule"]["choice"]:
                s["expected"]["choice_differs"].append(f"{r['object']}/{r['idx']}")
    for s in by.values():
        s["mismatch"] = dict(s["mismatch"])
    cases: dict[tuple, list[dict[str, Any]]] = {}
    for r in recs:
        if r.get("assessed"):
            cases.setdefault((r["object"], int(r["idx"]), int(r["seed"])), []).append(r)
    same = [k for k, rs in cases.items() if all(json.dumps([x["B"], x["C"]], sort_keys=True) == json.dumps([rs[0]["B"], rs[0]["C"]], sort_keys=True) for x in rs)]
    return {"settings": by, "cases": len(cases), "bc_same_across_settings": len(same),
            "bc_differ": [f"{o}/{i}" for (o, i, _s), _rs in cases.items() if (o, i, _s) not in set(same)]}


def check_markdown(c: dict[str, Any]) -> str:
    lines = ["# Ladder inputs: the lowest-value check", "",
             "The policy that always picks the lowest D value (expected cost at the true setting, decline at A) against the "
             "one-step rule at the true setting, on every first decision. A mismatch's reason: tie (the rule's row shows "
             "the same value), guard (the novelty guard declined a cheaper row), probe_margin (a probe row is the lowest, but "
             "it beats the best row by less than EVOI_Z Monte Carlo errors), other (a ladder bug).", "",
             "| setting | first decisions | no menu | match | mismatch by reason | reset artifacts | gap-check oracle: compared, menu differs, first choice differs |",
             "|---|---|---|---|---|---|---|"]
    for name, s in c["settings"].items():
        e = s["expected"]
        lines.append(f"| {name} | {s['n']} | {s['no_menu']} | {s['match']} | {json.dumps(s['mismatch']) if s['mismatch'] else '0'} | "
                     f"{s['reset_artifact']} | {e['compared']}, {len(e['menu_differs'])}, {len(e['choice_differs'])} |")
    lines += ["", f"B and C are the same under every setting for {c['bc_same_across_settings']} of {c['cases']} cases"
              + (f" (they differ for {', '.join(c['bc_differ'])})" if c["bc_differ"] else "") + ".", ""]
    for name, s in c["settings"].items():
        for x in s["cases"]:
            lines.append(f"- {name} {x['case']}: {x['reason']}: lowest {x['lowest']} ({x['lowest_value']}), rule {x['rule']} "
                         f"({x['rule_value']})")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- the CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inputs", help="B, C and D of every case's first decision (no language model)")
    p.add_argument("--cases", required=True)
    p.add_argument("--settings", default=",".join(SETTINGS))
    p.add_argument("--crossfit", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", required=True)
    p = sub.add_parser("smoke-cases", help="the pilot's smoke cases, written before any model call")
    p.add_argument("--never", required=True, help="an H3 never run (its cohort)")
    p.add_argument("--exclude-cases", action="append", default=[], help="a gap cases file whose cases are left out")
    p.add_argument("--exclude-plan", default=None, help="an e1 plan whose jobs (every split) are left out")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--plan-seed", type=int, default=4)
    p.add_argument("--settings", default=",".join(SETTINGS))
    p.add_argument("--out", required=True)
    p = sub.add_parser("check", help="the lowest-value policy against the one-step rule at the true setting")
    p.add_argument("--dir", required=True)
    p.add_argument("--expected", default=None, help="an e1_expected gap directory: its oracle records")
    p.add_argument("--wandb", action="store_true")
    p = sub.add_parser("render", help="one case's B, C and D blocks at its first decision")
    p.add_argument("--dir", required=True)
    p.add_argument("--object", required=True)
    p.add_argument("--idx", type=int, required=True)
    p.add_argument("--setting", required=True)
    p.add_argument("--seed", type=int, default=0, help="the row order's rng seed")
    a = ap.parse_args(argv)
    if a.cmd == "inputs":
        path = run_inputs(load_cases(a.cases), parse_settings(a.settings), a.crossfit, a.workers, a.out, a.cases)
        print(path)
        return 0
    if a.cmd == "smoke-cases":
        if os.path.exists(a.out):
            print(f"refusing: {a.out} exists", file=sys.stderr)
            return 1
        with (gzip.open if a.never.endswith(".gz") else open)(a.never, "rt") as f:
            never = json.load(f)
        exclude = {(c["seed"], c["object"], c["idx"]) for p in a.exclude_cases for c in load_cases(p)}
        if a.exclude_plan:
            exclude |= {(int(j["seed"]), j["object"], int(j["idx"])) for j in json.load(open(a.exclude_plan))["jobs"]}
        cases = pick_smoke(never, a.n, a.plan_seed, exclude, parse_settings(a.settings))
        with open(a.out, "w") as f:
            json.dump(cases, f, indent=1)
        print(f"{len(cases)} smoke cases over {len({c['object'] for c in cases})} objects at seed {cases[0]['seed']} "
              f"({len(exclude)} cases excluded) -> {a.out}")
        return 0
    if a.cmd == "check":
        c = check(load_inputs(a.dir), expected_oracle(a.expected) if a.expected else None)
        with open(os.path.join(a.dir, "check.json"), "w") as f:
            json.dump(c, f, indent=1)
        md = check_markdown(c)
        with open(os.path.join(a.dir, "check.md"), "w") as f:
            f.write(md)
        print(md)
        if a.wandb:
            import wandb

            run = wandb.init(project="rpent-pick-place", job_type="ladder_inputs_check", config={"dir": a.dir, "expected": a.expected})
            for name, s in c["settings"].items():
                run.summary[f"{name}/n"] = s["n"]
                run.summary[f"{name}/match"] = s["match"]
                for reason, n in s["mismatch"].items():
                    run.summary[f"{name}/mismatch_{reason}"] = n
            run.finish()
        bad = sum(s["mismatch"].get("other", 0) for s in c["settings"].values())
        return 1 if bad else 0
    recs = [r for r in load_inputs(a.dir) if (r["object"], r["idx"], r["setting"]) == (a.object, a.idx, a.setting)]
    if len(recs) != 1 or not recs[0].get("assessed"):
        print(f"no single assessed record for {a.object}/{a.idx} {a.setting}", file=sys.stderr)
        return 1
    r = recs[0]
    for rung in RUNGS:
        text, _ = render_state(rung, r, r["destination"], np.random.default_rng(a.seed))
        print(f"===== rung {rung} =====\n{text}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
