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
* C, + the outcome predictions: each attempt row's probabilities of the eleven outcomes, each probe row's possible
  results and P(thrown);
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

The pilot (smoke test, not in the paper): ``run`` plays rungs A to D on each smoke case with a served model (its own
setting's true menu at every decision point; at most e1.MAX_CALLS calls and e1.MAX_ATTEMPTS attempts), and ``ops``
reports only the operational numbers (format failures, refusals, tokens, minutes), never a choice, an outcome or a cost.

    python -P -m robots.mujoco_lab.ladder run --cases SMOKE.json --templates T.json --crossfit C --base-url URL --model NAME --out DIR
    python -P -m robots.mujoco_lab.ladder ops --dir DIR
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import hashlib
import os
import re
import sys
import time
import zlib
from collections import Counter
from typing import Any

import numpy as np

from robots.mujoco_lab import e1, e1_expected, e1_gap
from rpent.skills.pick_place.belief import N_PARTICLES
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


THROWN_FIRST = "P(thrown: the probe knocks the object off its support)"   # the definition, on its first use in a view
SAME_PINCH = "the same pinch and motion as commit_std; it is charged {t:.0f} s (commit_std {t_std:.0f} s)"


def prediction_text(r: dict[str, Any], define_thrown: bool = False) -> str:
    """C's line of one row; ``define_thrown`` spells out P(thrown) (its first use in a view)."""
    if r["kind"] == "probe":
        kinds = ", ".join(f"{k} {float(x):.2f}" for k, x in sorted(r["p_kind"].items(), key=lambda kv: -float(kv[1] or 0)))
        return f"possible results: {kinds}; {THROWN_FIRST if define_thrown else 'P(thrown)'} {e1._f(r['p_thrown'])}"
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
            f"(its time and any push-tip charge plus {e1._f(v['expected_cost_after'])}, the expected cost after it, which counts "
            "the irreversible cost when the probe throws or damages the object)")


def row_text(r: dict[str, Any], rows: list[dict[str, Any]]) -> str | None:
    """The text of row ``r``'s action: the skill's, but for regrasp_com with commit_std's action (the same pinch and
    motion), which says so and names both rows' times (the skill's reason string stays in menu.py)."""
    std = next((x for x in rows if x["id"] == "commit_std"), None)
    if r["id"] == "regrasp_com" and std is not None and r["action"] == std["action"]:
        return SAME_PINCH.format(t=r["time_s"], t_std=std["time_s"])
    return r.get("what")


def render_state(rung: str, inputs: dict[str, Any], destination: str, rng: np.random.Generator) -> tuple[str, list[str]]:
    """(text, option ids) of one decision point at ``rung``: from B the belief, then the shown rows (C's) in a random
    order (``rng``) with their action and time, from C their predictions (P(thrown) defined on its first use), from D
    their values after a line on the cost unit; decline last."""
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
    if k >= 3:
        lines.append(f"Expected costs are in the task's cost units (decline costs {e1._f(inputs['D']['rows']['decline'])} here).")
    lines.append({0: "Actions (in random order):", 1: "Actions (in random order):",
                  2: "Actions (in random order) with the skill's outcome predictions:",
                  3: "Actions (in random order) with the skill's outcome predictions and expected costs:"}[k])
    thrown_defined = False
    for r in rows:
        what = e1._row_action({"kind": r["kind"], "action": r["action"], "reason": row_text(r, rows)}, destination)
        line = f"- {r['id']} ({r['kind']}): {what}; about {r['time_s']:.0f} s."
        if k >= 2:
            define = r["kind"] == "probe" and not thrown_defined
            thrown_defined = thrown_defined or define
            line += "\n  " + prediction_text(r, define) + ("; " + value_text(r, inputs["D"]) if k >= 3 else "") + "."
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
                    **rec, "particles": int(prims.n_particles), "seconds": round(time.time() - t0, 2)})
    prims.menu_recovery = None
    print(f"{case['object']} {case['idx']}: {len(out)} first decisions, {sum(r['seconds'] for r in out):.0f}s", file=sys.stderr,
          flush=True)
    return out


def _key(r: dict[str, Any]) -> tuple[str, int, int, str]:
    return r["object"], int(r["idx"]), int(r["seed"]), r["setting"]


def check_particles(out_dir: str, n_particles: int) -> None:
    """A ValueError when ``out_dir`` holds a run at another particle count per belief (a run config from before the
    option counts as belief.N_PARTICLES), so a resumed run never mixes counts."""
    for held in sorted(p for p in os.listdir(out_dir) if p.startswith("run_config_")) if os.path.isdir(out_dir) else []:
        n = int(json.load(open(os.path.join(out_dir, held))).get("particles", N_PARTICLES))
        if n != n_particles:
            raise ValueError(f"{out_dir} holds a run at {n} particles per belief ({held}), not {n_particles}; give a new "
                             "directory")


def run_inputs(cases: list[dict[str, Any]], settings: dict[str, dict[str, Any]], crossfit: str, workers: int, out_dir: str,
               cases_path: str, particles: int | None = None) -> str:
    """Every case's first decisions into ``out_dir``/inputs.jsonl (resuming: a (case, setting) with a record is skipped;
    a directory of another bundle, generator, setting list or particle count is an error); returns the path.
    ``particles``: the count per belief (e1_expected.resolve_particles: this override, else the bundle's, else
    belief.N_PARTICLES), the same rule as ladder_e.run, so B to D and E read one count."""
    from robots.mujoco_lab.calibrate import code_revision
    from robots.mujoco_lab.corpus import lab_generator

    generator = e1.run_generator(out_dir)
    e1.check_bundle(out_dir, crossfit, generator)
    n_particles, source = e1_expected.resolve_particles(particles, crossfit)
    for held in sorted(p for p in os.listdir(out_dir) if p.startswith("run_config_")) if os.path.isdir(out_dir) else []:
        c = json.load(open(os.path.join(out_dir, held)))
        if c.get("settings") != settings:
            raise ValueError(f"{out_dir} holds a run of other settings ({held}); give a new directory")
    check_particles(out_dir, n_particles)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"run_config_{time.strftime('%Y%m%dT%H%M%S')}.json"), "w") as f:
        json.dump({"cases": cases_path, "n_cases": len(cases), "settings": settings, "crossfit": crossfit,
                   "generator": lab_generator(), "workers": workers, "particles": n_particles, "particles_source": source,
                   "code": code_revision()}, f, indent=1)
    path = os.path.join(out_dir, "inputs.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {_key(json.loads(line)) for line in f if line.strip()}
    cfg = {"crossfit": crossfit, "folds": json.load(open(os.path.join(crossfit, "folds.json"))), "settings": settings,
           "particles": n_particles}
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
    reason (with the cases), B and C (by row id) the same under every setting of a case (they do not depend on kappa
    or A), and,
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
    def bc(x: dict[str, Any]) -> str:   # C by row id: the menu's row order follows the costs, so it changes with kappa and A
        return json.dumps([x["B"], {row["id"]: row for row in x["C"]}], sort_keys=True)

    same = [k for k, rs in cases.items() if all(bc(x) == bc(rs[0]) for x in rs)]
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


# ---------------------------------------------------------------- the pilot loop

REFUSAL = re.compile(r"\b(I can(?:no|')t|I cannot|I won't|I will not|unable to|refuse)\b", re.I)
FULL_RUNS = 4 * (96 + 145 + 185)       # rungs A-D at the n50 sizes of the three settings (e1_gap_v11_v11c.md)


def load_templates(path: str) -> dict[str, Any]:
    """A templates file: system (with {max_calls} and {max_attempts}), rungs (A to D), task and one instruction per
    setting (with {object} and {destination})."""
    t = json.load(open(path))
    missing = [k for k in ("system", "rungs", "task", "settings") if k not in t] + [r for r in RUNGS if r not in t.get("rungs", {})]
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    return t


def system_text(tpl: dict[str, Any], rung: str, max_calls: int, max_attempts: int) -> str:
    return tpl["system"].format(max_calls=max_calls, max_attempts=max_attempts) + "\n\n" + tpl["rungs"][rung]


def user_head(tpl: dict[str, Any], setting: str, object_name: str, destination: str) -> str:
    return (tpl["settings"][setting].format(object=object_name, destination=e1.display_destination(destination)) + "\n\n"
            + tpl["task"].format(object=object_name, destination=destination))


def view_inputs(a: dict[str, Any], store: dict[str, Any]) -> dict[str, Any]:
    return {"B": posterior(belief_of(store)), "C": predictions(a), "D": values(a)}


def ladder_loop(tools: Any, model: Any, rung: str, tpl: dict[str, Any], setting: str, object_name: str, destination: str,
                rng: np.random.Generator, max_calls: int = e1.MAX_CALLS, max_attempts: int = e1.MAX_ATTEMPTS) -> dict[str, Any]:
    """One pilot case at one rung: ``tools.assess()`` gives the true menu and ``tools.act(option_id)`` runs a row (as
    e1.planner_loop); each decision point shows render_state at ``rung``, each action's result its raw reading
    (e1.render_result as the no-menu planner reads it). The record: e1's probe and attempt fields, calls, decisions,
    malformed (no valid option id), no_tool_call, refusals (a reply with no tool call whose text reads as a refusal,
    REFUSAL), failed_calls and the transcript."""
    case: dict[str, Any] = {"rung": rung, "probes": [], "attempts": [], "restores": [], "ended": None, "calls": [], "decisions": [],
                            "malformed": 0, "no_tool_call": 0, "refusals": 0, "failed_calls": 0, "transcript": []}

    def end(reason: str, error: str | None = None) -> dict[str, Any]:
        case["ended"] = reason
        if error is not None:
            case["error"] = error
        return case

    a = tools.assess()
    if "error" in a:
        return end("error", a["error"])
    if not a.get("assessed", True):
        return end("unreachable")
    text, ids = render_state(rung, view_inputs(a, tools.store), destination, rng)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_text(tpl, rung, max_calls, max_attempts)},
                                      {"role": "user", "content": user_head(tpl, setting, object_name, destination) + "\n\n" + text}]
    case["transcript"] = list(messages)
    rows = {o["id"]: o for o in a["options"]}
    while True:
        if len(case["calls"]) >= max_calls:
            return end("call_cap")
        r = model(messages, e1.tool_schema(ids))
        case["calls"].append({k: r.get(k) for k in ("tool", "args", "content", "reasoning", "seconds", "usage", "error", "finish_reason")})
        if r.get("error"):
            case["failed_calls"] += 1
        choice = e1.parse_choice(r, ids)
        call_id = f"call_{len(case['calls'])}"
        if choice is None:
            case["malformed"] += 1
            if r.get("tool") == "choose":
                fix = [{"role": "assistant", "content": r.get("content") or "", "tool_calls": [
                           {"id": call_id, "type": "function", "function": {"name": "choose", "arguments": json.dumps(r.get("args") or {})}}]},
                       {"role": "tool", "tool_call_id": call_id,
                        "content": f"Error: option_id must be one of {ids}. Nothing ran. Call `choose` again."}]
            else:
                case["no_tool_call"] += 1
                case["refusals"] += bool(REFUSAL.search(r.get("content") or ""))
                fix = [{"role": "assistant", "content": r.get("content") or "(no tool call)"},
                       {"role": "user", "content": f"Nothing ran: call the tool `choose` once, with option_id one of {ids}."}]
            messages += fix
            case["transcript"] += fix
            continue
        reason = (r.get("args") or {}).get("reason") if isinstance(r.get("args"), dict) else None
        case["decisions"].append({"step": len(case["decisions"]), "choice": choice, "reason": reason})
        call = {"role": "assistant", "content": r.get("content") or "", "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "choose", "arguments": json.dumps({"option_id": choice, "reason": reason})}}]}
        if choice == "decline":
            case["transcript"].append(call)
            return end("abstained")
        row = rows[choice]
        d = tools.act(choice)
        if "error" in d and "outcome" not in d and "probed" not in d:
            case["transcript"].append(call)
            return end("error", str(d["error"]))
        ended = None
        if row["kind"] == "probe":
            case["probes"].append(e1._probe_record(d, row, len(case["probes"]), len(case["attempts"])))
            if d.get("restored") is not None:
                case["restores"].append(dict(d["restored"], probe=d["probed"]))
            elif d.get("kind") == e1.DAMAGED or d.get("damage"):
                ended = "damaged"
            elif d.get("thrown"):
                ended = "thrown"
        else:
            case["attempts"].append(e1._attempt_record(d, row, a))
            restored = d.get("restored") is not None
            if restored:
                case["restores"].append(dict(d["restored"], option=row["id"]))
            if d["outcome"] == "placed_stable":
                ended = "placed"
            elif (d["outcome"] not in RECOVERABLE and not restored) or len(case["attempts"]) >= max_attempts:
                ended = "failed"
        result = e1.render_result(d, row, "nomenu_planner")
        if ended is not None:
            case["transcript"] += [call, {"role": "tool", "tool_call_id": call_id, "content": result + " The task is over."}]
            return end(ended)
        a = d.get("assessment") or {}
        if "error" in a:
            return end("error", a["error"])
        if not a.get("assessed", False):
            case["transcript"] += [call, {"role": "tool", "tool_call_id": call_id, "content": result + f" {a.get('reason')}."}]
            return end("unreachable")
        text, ids = render_state(rung, view_inputs(a, tools.store), destination, rng)
        rows = {o["id"]: o for o in a["options"]}
        step = [call, {"role": "tool", "tool_call_id": call_id, "content": result + "\n\n" + text}]
        messages += step
        case["transcript"] += step


_WORKER: dict[str, Any] = {}


def _pilot_model(cfg: dict[str, Any]) -> Any:
    if "model" not in _WORKER:
        _WORKER["model"] = (e1.StubPlanner(os.getpid()) if cfg.get("stub")
                            else e1.GemmaPlanner(cfg["base_url"], cfg["model"], max_tokens=cfg["max_tokens"],
                                                 temperature=cfg.get("temperature", 0.0), thinking=True, top_p=cfg.get("top_p"),
                                                 top_k=cfg.get("top_k")))
    return _WORKER["model"]


def pilot_job(item: tuple[dict[str, Any], dict[str, Any]]) -> list[dict[str, Any]]:
    """One smoke case at every rung of the run, in turn on one lab: a reset to the case, the lab and menu at the case's own
    setting (e1_gap.rule_config("oracle", ...)), the ladder loop; the row order's rng is the same at every rung."""
    from robots.mujoco_lab import calibrate

    case, cfg = item
    prims = e1_expected._prims(cfg, case["object"])
    conf = e1_gap.rule_config("oracle", cfg["settings"][case["setting"]])
    out = []
    for rung in cfg["rungs"]:
        t0 = time.time()
        prims.abstain_cost, prims.menu_recovery = conf["abstain_cost"], conf["menu_recovery"]
        prims.reset_s, prims.reset_price = conf["lab_recovery"]
        destination, reset_info, truth = e1.reset_case(prims, case["object"], int(case["idx"]), int(case["seed"]))
        rng = np.random.default_rng([int(case["seed"]), zlib.crc32(case["object"].encode()), int(case["idx"]), 11])
        try:
            rec = ladder_loop(e1.LabTools(prims, destination, conf["care"]), _pilot_model(cfg), rung, cfg["templates"],
                              case["setting"], e1.display_object(case["object"]), destination, rng, max_calls=cfg["max_calls"])
        except Exception as exc:  # noqa: BLE001  one bad case must not stop the run
            rec = {"rung": rung, "ended": "error", "error": f"{type(exc).__name__}: {exc}"[:300], "probes": [], "attempts": [],
                   "calls": [], "decisions": [], "malformed": 0, "no_tool_call": 0, "refusals": 0, "failed_calls": 0}
        rec["artifact"] = calibrate.case_artifact(rec, reset_info)
        out.append(dict(case, destination=destination, truth=truth, reset=reset_info, **rec, particles=int(prims.n_particles),
                        seconds=round(time.time() - t0, 2)))
        print(f"{case['object']} {case['idx']} {case['setting']} rung {rung}: {len(rec.get('calls') or [])} calls, "
              f"{out[-1]['seconds']:.0f} s", file=sys.stderr, flush=True)
    prims.menu_recovery = None
    return out


def run_pilot(cases_path: str, rungs: list[str], templates_path: str, crossfit: str, base_url: str | None, model: str,
              stub: bool, workers: int, out_dir: str, max_calls: int = e1.MAX_CALLS, max_tokens: int = 6144,
              sampling: dict[str, Any] | None = None, particles: int | None = None) -> str:
    """Every smoke case at every rung (resuming: a (case, rung) in rows.jsonl is skipped; a directory of another
    bundle, generator, templates file, model, sampling or particle count is an error); each run segment's start and end
    in run_times.jsonl. ``sampling``: temperature (default 0, as e1), top_p and top_k (sent only when given).
    ``particles``: as run_inputs."""
    sampling = {"temperature": 0.0, "top_p": None, "top_k": None, **(sampling or {})}
    from robots.mujoco_lab.calibrate import code_revision
    from robots.mujoco_lab.corpus import lab_generator

    cases = load_cases(cases_path)
    if any(not c.get("setting") for c in cases):
        raise ValueError(f"{cases_path}: every smoke case needs its setting")
    tpl = load_templates(templates_path)
    sha = hashlib.sha256(open(templates_path, "rb").read()).hexdigest()
    generator = e1.run_generator(out_dir)
    e1.check_bundle(out_dir, crossfit, generator)
    n_particles, source = e1_expected.resolve_particles(particles, crossfit)
    if os.path.isdir(out_dir):
        for held in sorted(p for p in os.listdir(out_dir) if p.startswith("run_config_")):
            c = json.load(open(os.path.join(out_dir, held)))
            held_sampling = {k: (c.get("planner") or {}).get(k) for k in sampling}
            if (c.get("templates_sha256"), c.get("model"), c.get("stub"), held_sampling) != (sha, model, stub, sampling):
                raise ValueError(f"{out_dir} holds a run of other templates, another model, stub mode or sampling ({held})")
    check_particles(out_dir, n_particles)
    os.makedirs(out_dir, exist_ok=True)
    settings = {c["setting"]: e1_gap.grid()[c["setting"]] for c in cases}
    with open(os.path.join(out_dir, f"run_config_{time.strftime('%Y%m%dT%H%M%S')}.json"), "w") as f:
        json.dump({"cases": os.path.abspath(cases_path), "rungs": rungs, "templates": os.path.abspath(templates_path),
                   "templates_sha256": sha, "templates_version": tpl.get("version"), "crossfit": crossfit, "generator": lab_generator(),
                   "base_url": base_url, "model": model, "stub": stub, "workers": workers, "max_calls": max_calls,
                   "max_attempts": e1.MAX_ATTEMPTS, "planner": {**sampling, "max_tokens": max_tokens, "thinking": True},
                   "settings": settings, "particles": n_particles, "particles_source": source, "code": code_revision()}, f,
                  indent=1)
    path = os.path.join(out_dir, "rows.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {(r["object"], r["idx"], r["seed"], r["setting"], r["rung"]) for r in map(json.loads, filter(str.strip, f))}
    cfg = {"crossfit": crossfit, "folds": json.load(open(os.path.join(crossfit, "folds.json"))), "settings": settings,
           "templates": tpl, "base_url": base_url, "model": model, "stub": stub, "max_calls": max_calls, "max_tokens": max_tokens,
           "particles": n_particles, **sampling}
    todo = [(c, dict(cfg, rungs=[r])) for c in cases for r in rungs      # one job per (case, rung): an even load
            if (c["object"], c["idx"], c["seed"], c["setting"], r) not in done]
    print(f"ladder pilot: {len(todo)} of {len(cases) * len(rungs)} rung runs to do, rungs {rungs}", file=sys.stderr, flush=True)
    t0 = time.time()
    with open(path, "a") as f:
        def write(parts) -> None:
            for part in parts:
                for r in part:
                    f.write(json.dumps(r, default=str) + "\n")
                f.flush()

        if workers <= 1:
            write(map(pilot_job, todo))
        else:
            with mp.get_context("spawn").Pool(workers) as pool:
                write(pool.imap_unordered(pilot_job, todo))
    with open(os.path.join(out_dir, "run_times.jsonl"), "a") as f:
        f.write(json.dumps({"started": t0, "ended": time.time(), "jobs": len(todo), "runs": sum(len(c["rungs"]) for _, c in todo),
                            "workers": workers}) + "\n")
    return path


def _stats(v: list[float]) -> dict[str, Any]:
    if not v:
        return {"n": 0, "sum": 0, "mean": None, "p90": None, "max": None}
    a = np.asarray(v, dtype=float)
    return {"n": len(v), "sum": float(a.sum()), "mean": float(a.mean()), "p90": float(np.percentile(a, 90)), "max": float(a.max())}


def ops(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The pilot's operational numbers per rung and pooled ("all"): runs, errors, call-cap endings, malformed replies,
    replies with no tool call, refusals, failed calls, length finishes, calls per run, tokens per call and per run (in and
    out), model seconds per call, minutes per run and the resets that did not reach rest (the lab's, not the model's). It
    reads no choice, outcome or cost."""
    out: dict[str, Any] = {"rungs": {}}
    for rung in RUNGS + ("all",):
        rs = [r for r in rows if rung == "all" or r.get("rung") == rung]
        if not rs:
            continue
        calls = [c for r in rs for c in r.get("calls") or []]
        use = lambda c, k: int((c.get("usage") or {}).get(k) or 0)  # noqa: E731
        out["rungs"][rung] = {
            "runs": len(rs), "errors": sum(r.get("ended") == "error" for r in rs), "call_cap": sum(r.get("ended") == "call_cap" for r in rs),
            "malformed": sum(int(r.get("malformed") or 0) for r in rs), "no_tool_call": sum(int(r.get("no_tool_call") or 0) for r in rs),
            "refusals": sum(int(r.get("refusals") or 0) for r in rs), "failed_calls": sum(int(r.get("failed_calls") or 0) for r in rs),
            "length_finishes": sum(c.get("finish_reason") == "length" for c in calls),
            "reset_not_at_rest": sum(not (r.get("reset") or {}).get("at_rest", True) for r in rs),
            "calls": _stats([len(r.get("calls") or []) for r in rs]),
            "tokens_in": _stats([use(c, "prompt") for c in calls]), "tokens_out": _stats([use(c, "completion") for c in calls]),
            "tokens_in_per_run": _stats([sum(use(c, "prompt") for c in r.get("calls") or []) for r in rs]),
            "tokens_out_per_run": _stats([sum(use(c, "completion") for c in r.get("calls") or []) for r in rs]),
            "model_seconds_per_call": _stats([float(c.get("seconds") or 0.0) for c in calls]),
            "minutes_per_run": _stats([float(r.get("seconds") or 0.0) / 60.0 for r in rs])}
    return out


def ops_markdown(o: dict[str, Any], times: list[dict[str, Any]], server_start_min: float | None, title: str) -> str:
    f = lambda x, nd=1: "-" if x is None else f"{x:.{nd}f}"  # noqa: E731
    lines = [f"# {title}: operational numbers only", "",
             "No choice, outcome or cost is read here. Malformed: a reply with no valid option id (no tool call, or an id "
             "not on the list); refusal: a reply with no tool call whose text reads as a refusal (a keyword match).", "",
             "| rung | runs | errors | call cap | malformed | no tool call | refusals | failed calls | length finishes | resets not at rest | "
             "calls per run (mean, max) | tokens in per call (mean, max) | tokens out per call (mean, max) | tokens in per run "
             "(mean) | tokens out per run (mean) | model s per call (mean, p90) | minutes per run (mean, p90) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for rung, x in o["rungs"].items():
        lines.append(f"| {rung} | {x['runs']} | {x['errors']} | {x['call_cap']} | {x['malformed']} | {x['no_tool_call']} | "
                     f"{x['refusals']} | {x['failed_calls']} | {x['length_finishes']} | {x['reset_not_at_rest']} | {f(x['calls']['mean'], 2)}, {f(x['calls']['max'], 0)} | "
                     f"{f(x['tokens_in']['mean'], 0)}, {f(x['tokens_in']['max'], 0)} | {f(x['tokens_out']['mean'], 0)}, "
                     f"{f(x['tokens_out']['max'], 0)} | {f(x['tokens_in_per_run']['mean'], 0)} | {f(x['tokens_out_per_run']['mean'], 0)} | "
                     f"{f(x['model_seconds_per_call']['mean'])}, {f(x['model_seconds_per_call']['p90'])} | "
                     f"{f(x['minutes_per_run']['mean'], 2)}, {f(x['minutes_per_run']['p90'], 2)} |")
    if times:
        wall = sum(t["ended"] - t["started"] for t in times) / 60.0
        runs = sum(t["runs"] for t in times)
        per = wall / runs if runs else None
        workers = sorted({t["workers"] for t in times})
        lines += ["", f"Wall time: {wall:.1f} min for {runs} rung runs at {workers} workers on one GPU: "
                  f"{f(per, 3)} min per rung run." + ("" if server_start_min is None else f" Server start: {server_start_min:.1f} min.")]
        if per is not None:
            start = (server_start_min or 0.0) / 60.0
            est = FULL_RUNS * per / 60.0 + start
            lines.append(f"Full ladder at the n50 sizes (rungs A-D x 426 cases = {FULL_RUNS} rung runs, the same workers): "
                         f"about {est:.1f} GPU-hours for this model (from the wall time, with the pilot's tail).")
            mean = (o["rungs"].get("all") or {}).get("minutes_per_run", {}).get("mean")
            if mean is not None and len(workers) == 1:
                est2 = FULL_RUNS * mean / workers[0] / 60.0 + start
                lines.append(f"From the mean minutes per rung run ({mean:.2f}) at {workers[0]} workers: about {est2:.1f} "
                             "GPU-hours (no tail).")
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
    p.add_argument("--particles", type=int, default=None, help="particles per belief (default: the bundle's, else 2000)")
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
    p = sub.add_parser("run", help="the pilot: rungs A to D on the smoke cases with a served model")
    p.add_argument("--cases", required=True)
    p.add_argument("--rungs", default=",".join(RUNGS))
    p.add_argument("--templates", required=True)
    p.add_argument("--crossfit", required=True)
    p.add_argument("--base-url", default=None)
    p.add_argument("--model", default=e1.MODEL, help="the served model name")
    p.add_argument("--stub", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-calls", type=int, default=e1.MAX_CALLS)
    p.add_argument("--max-tokens", type=int, default=6144)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--particles", type=int, default=None, help="particles per belief (default: the bundle's, else 2000)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("ops", help="the pilot's operational numbers (no choice, outcome or cost)")
    p.add_argument("--dir", required=True)
    p.add_argument("--server-start-min", type=float, default=None)
    p.add_argument("--title", default="Ladder pilot")
    p.add_argument("--wandb", action="store_true")
    p = sub.add_parser("render", help="one case's B, C and D blocks at its first decision")
    p.add_argument("--dir", required=True)
    p.add_argument("--object", required=True)
    p.add_argument("--idx", type=int, required=True)
    p.add_argument("--setting", required=True)
    p.add_argument("--seed", type=int, default=0, help="the row order's rng seed")
    a = ap.parse_args(argv)
    if a.cmd == "inputs":
        path = run_inputs(load_cases(a.cases), parse_settings(a.settings), a.crossfit, a.workers, a.out, a.cases, a.particles)
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
    if a.cmd == "run":
        if not a.stub and not a.base_url:
            ap.error("give --base-url or --stub")
        rungs = [x for x in a.rungs.split(",") if x]
        if any(r not in RUNGS for r in rungs):
            ap.error(f"rungs must be among {RUNGS}")
        print(run_pilot(a.cases, rungs, a.templates, a.crossfit, a.base_url, a.model, a.stub, a.workers, a.out, a.max_calls,
                        a.max_tokens, {"temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k}, particles=a.particles))
        return 0
    if a.cmd == "ops":
        with open(os.path.join(a.dir, "rows.jsonl")) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        o = ops(rows)
        tp = os.path.join(a.dir, "run_times.jsonl")
        times = [json.loads(line) for line in open(tp) if line.strip()] if os.path.exists(tp) else []
        md = ops_markdown(o, times, a.server_start_min, a.title)
        with open(os.path.join(a.dir, "ops.json"), "w") as f:
            json.dump({"ops": o, "times": times, "server_start_min": a.server_start_min}, f, indent=1)
        with open(os.path.join(a.dir, "ops.md"), "w") as f:
            f.write(md)
        print(md)
        if a.wandb:
            import wandb

            run = wandb.init(project="rpent-pick-place", job_type="ladder_pilot_ops", config={"dir": a.dir, "title": a.title})
            for rung, x in o["rungs"].items():
                for k in ("runs", "errors", "call_cap", "malformed", "no_tool_call", "refusals", "failed_calls", "length_finishes",
                          "reset_not_at_rest"):
                    run.summary[f"{rung}/{k}"] = x[k]
                run.summary[f"{rung}/tokens_in_mean"] = x["tokens_in"]["mean"]
                run.summary[f"{rung}/tokens_out_mean"] = x["tokens_out"]["mean"]
                run.summary[f"{rung}/minutes_per_run_mean"] = x["minutes_per_run"]["mean"]
            print(f"wandb: {run.url}")
            run.finish()
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
