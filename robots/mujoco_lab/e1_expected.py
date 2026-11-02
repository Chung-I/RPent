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

"""The E1 expected-cost rescoring (2026-11-02; the user's choice of score, relayed by the paper session). Analysis only: no
new decision and no language-model call. A choice's score at a decision point is its expected cost under the true setting,
from the menu's own probabilities and probe pricing; its regret is that cost minus the cheapest row of the same true menu
(decline costs A; inboard_place is left out, as every arm leaves it out).

* First decision (the score): every arm sees the same first menu in a case. That menu is rebuilt at the true setting (the
  setting's kappa, A and recovery: a fresh reset and assessment, as the oracle rule's own first menu), and each arm's first
  recorded choice is scored on it.
* Every decision point (the second measure): the same regret, summed over every point where the run saw a menu. A later true
  menu comes from the run's own belief at that point. To reach that belief, the run's recorded choices are replayed on the
  same lab (deterministic: the gap check repeated every run identically); each replayed menu and outcome is checked against
  the record, and a run whose replay diverges gets no later score. The true menu is built on copies (the run's store without
  its menu cache, and prims.rng), so the replay goes on unchanged. The oracle's own later menus are its true menus.

Two sources: the gap check (data/e1/gap: default - oracle regret per setting, ranked by the lower bound of the first-decision
interval, split by the default menu's first recommendation, with e1_gap's power line) and the pilot (data/e1/final and
data/e1/neutral/final: each arm - the default rule, per setting and pooled).

    python -P -m robots.mujoco_lab.e1_expected gap --src DIR --workers 8 --out DIR
    python -P -m robots.mujoco_lab.e1_expected pilot --final DIR --neutral DIR --plan PLAN --workers 8 --out DIR
    python -P -m robots.mujoco_lab.e1_expected summary --dir DIR [--wandb]
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from typing import Any, Callable

import numpy as np

from robots.mujoco_lab import e1, e1_gap

TOL = 1e-9
RULE_ARMS = ("default", "oracle", "default_rule", "oracle_rule")
OWN_TRUE = ("oracle", "oracle_rule")      # their own menus are the true menus
PILOT_SETTINGS = ("hi_abs", "hi_rec", "lo_abs", "lo_rec")
PILOT_ARMS = ("oracle_rule", "menu_planner", "rec_planner", "nomenu_planner", "menu_planner_neutral", "rec_planner_neutral")


# ---------------------------------------------------------------- a choice on a menu

def choosable(a: dict[str, Any]) -> dict[str, float]:
    """The rows an arm can choose at menu ``a`` and their expected costs: every row but e1.HIDDEN_ROWS, decline at A."""
    return {o["id"]: float(o["expected_cost"]) for o in a.get("options") or []
            if o["id"] not in e1.HIDDEN_ROWS and o.get("expected_cost") is not None}


def score(a: dict[str, Any], choice: str) -> dict[str, Any] | None:
    """``choice`` on menu ``a``: its expected cost, the cheapest row and the regret (the cost minus the cheapest row's); None
    when the choice is not a row of the menu."""
    rows = choosable(a)
    if choice not in rows:
        return None
    best = min(rows, key=rows.get)
    return {"choice": choice, "cost": rows[choice], "min": rows[best], "min_row": best, "regret": rows[choice] - rows[best]}


def miss_reason(a: dict[str, Any], choice: str) -> str | None:
    """Why a rule's choice on its own menu is not the cheapest row: None (it is, within TOL); "guard" (a guard flag
    declined); "probe_margin" (a probe row is the cheapest, but the cost rule probes only when the probe beats the cheaper of
    the best row and A by EVOI_Z standard errors); "not_a_row"; "other"."""
    s = score(a, choice)
    if s is None:
        return "not_a_row"
    if s["regret"] <= TOL:
        return None
    if choice == "decline" and (a.get("guard") or {}).get("flag"):
        return "guard"
    kind = {o["id"]: o.get("kind") for o in a.get("options") or []}
    if kind.get(s["min_row"]) == "probe" and kind.get(choice) != "probe":
        return "probe_margin"
    return "other"


# ---------------------------------------------------------------- the recorded choices

def is_rule(arm: str) -> bool:
    return arm in RULE_ARMS


def rule_actions(case: dict[str, Any]) -> list[str]:
    """A rule run's choices in order: its probes (their rows), its attempts (their rows), and decline when it abstained (the
    policy probes only before its first attempt)."""
    if any(int(p.get("after_attempts") or 0) for p in case.get("probes") or []):
        raise ValueError("a rule probe after an attempt: the order of the choices is not recorded")
    out = [f"probe_{p['probe']}" for p in case.get("probes") or []] + [t["option"] for t in case.get("attempts") or []]
    return out + (["decline"] if case.get("ended") == "abstained" else [])


def planner_actions(case: dict[str, Any]) -> list[str]:
    """A planner run's valid choices in order (its decisions). At the call cap it saw a menu and chose nothing, which ends
    the case unplaced, as a decline does."""
    out = [d["choice"] for d in case.get("decisions") or []]
    return out + (["decline"] if case.get("ended") == "call_cap" else [])


def actions(case: dict[str, Any], arm: str) -> list[str]:
    return rule_actions(case) if is_rule(arm) else planner_actions(case)


def menu_fields(a: dict[str, Any]) -> dict[str, Any]:
    """The menu fields a record keeps: the recommendation, its row, the best row, the guard flag, the recommendation
    without probes, the best and recommended rows' costs, and each row's cost, action and probabilities (cost:<id>,
    action:<id>, p:<id>)."""
    cost = {o["id"]: o.get("expected_cost") for o in a.get("options") or []}
    out = {"recommendation": a.get("recommendation"), "recommended_option": a.get("recommended_option"),
           "best_option": a.get("best_option"), "guard_flag": bool((a.get("guard") or {}).get("flag")),
           "without_probe": a.get("without_probe"), "best_cost": cost.get(a.get("best_option")),
           "recommended_cost": cost.get(a.get("recommended_option"))}
    for o in a.get("options") or []:
        out.update({f"cost:{o['id']}": o.get("expected_cost"), f"action:{o['id']}": o.get("action"), f"p:{o['id']}": o.get("p")})
    return out


def outcome_fields(d: dict[str, Any]) -> dict[str, Any]:
    """The result fields a record keeps: a probe's name, kind, throw, reading and declared numbers, or an attempt's
    outcome; the artifact flag of either."""
    if "probed" in d:
        return {"probed": d["probed"], "kind": d.get("kind"), "thrown": bool(d.get("thrown")), "reading": d.get("reading"),
                "declared": d.get("declared"), "artifact": bool(d.get("artifact", False))}
    return {"outcome": d.get("outcome"), "artifact": bool(d.get("artifact", False))}


def expectations(case: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    """Per decision point of a recorded run: the choice, the menu fields the record kept for that point ("menu") and the
    action's result ("outcome"; None for decline). A planner decision whose action ended the case in an error has no
    recorded result, and the points end before it."""
    probes, attempts = list(case.get("probes") or []), list(case.get("attempts") or [])
    decisions = case.get("decisions") or []
    pts = []
    for k, c in enumerate(actions(case, arm)):
        menu: dict[str, Any] = {}
        if not is_rule(arm) and k < len(decisions):
            menu = {f: decisions[k][f] for f in ("recommendation", "recommended_option", "best_option", "guard_flag")}
        outcome = None
        if c != "decline" and not (probes if c.startswith("probe_") else attempts):
            break
        if c.startswith("probe_"):
            p = probes.pop(0)
            if p.get("option", c) != c:
                raise ValueError(f"point {k}: the choice {c} ran the probe row {p.get('option')}")
            outcome = {"probed": p["probe"], "kind": p["kind"], "thrown": bool(p["thrown"]), "reading": p.get("reading")}
            outcome.update({k: p[k] for k in ("artifact", "declared") if k in p})
        elif c != "decline":
            t = attempts.pop(0)
            if t["option"] != c:
                raise ValueError(f"point {k}: the choice {c} ran the row {t['option']}")
            if is_rule(arm):
                menu["recommendation"] = t["recommendation"]
            menu[f"cost:{c}"] = t["declared"]["expected_cost"]
            menu.update({f"{k}:{c}": v for k, v in (("action", t.get("action")), ("p", t["declared"].get("p"))) if v is not None})
            outcome = {"outcome": t["outcome"]}
            if "artifact" in t:
                outcome["artifact"] = bool(t["artifact"])
        pts.append({"choice": c, "menu": menu, "outcome": outcome})
    if pts and is_rule(arm):
        fm = case.get("first_menu") or {}
        pts[0]["menu"] = dict({"recommendation": case.get("recommendation"), "best_option": case.get("best_option"),
                               "guard_flag": bool((case.get("guard") or {}).get("flag")), "best_cost": fm.get("best_cost"),
                               "recommended_cost": fm.get("recommended_cost")}, **pts[0]["menu"])
    elif pts:
        pts[0]["menu"]["without_probe"] = (case.get("first_menu") or {}).get("without_probe")
    return pts


# ---------------------------------------------------------------- the replay and the true menus

def _json(x: Any) -> str:
    return json.dumps(x, sort_keys=True, default=str)


def _diff(expected: dict[str, Any], got: dict[str, Any]) -> list[str]:
    return [f"{k}: recorded {_json(v)}, replayed {_json(got.get(k))}" for k, v in expected.items() if _json(v) != _json(got.get(k))]


def replay(tools: Any, points: list[dict[str, Any]], build: Callable[[Any], dict[str, Any]] | None = None) -> dict[str, Any]:
    """Run a recorded run's choices again through ``tools`` (assess once, then act per choice; decline does not act), checking
    each point's menu fields and each action's result against ``points`` (expectations); it stops at the first divergence.
    ``build(tools)`` gives the true menus ({name: menu}) at each later point, before its action. Returns "first" (the first
    assessment), "menus" (the run's own menu at each point that checked), "true" (None at the first point, then build's
    menus) and "mismatch"."""
    out: dict[str, Any] = {"first": None, "menus": [], "true": [], "mismatch": []}
    a = out["first"] = tools.assess()
    for k, pt in enumerate(points):
        if not a.get("assessed", False) or "error" in a:
            out["mismatch"].append(f"point {k}: no menu")
            break
        bad = _diff(pt["menu"], menu_fields(a))
        if bad:
            out["mismatch"] += [f"point {k} menu {x}" for x in bad]
            break
        out["menus"].append(a)
        out["true"].append(build(tools) if build is not None and k > 0 else None)
        if pt["choice"] == "decline":
            break
        d = tools.act(pt["choice"])
        bad = _diff(pt["outcome"] or {}, outcome_fields(d))
        if bad:
            out["mismatch"] += [f"point {k} outcome {x}" for x in bad]
            break
        a = d.get("assessment") or {}
    return out


def true_menu(prims: Any, store: dict[str, Any], destination: str, conf: dict[str, Any],
              assess: Callable[..., Any] | None = None) -> dict[str, Any]:
    """The menu at ``conf`` (care, abstain_cost, menu_recovery) from the belief in a run's ``store``, on copies: the store
    without its menu cache (whose key has no A and no recovery) and prims.rng. The run's store, prims.rng and the prims menu
    settings stay as they were."""
    if assess is None:
        from rpent.skills.pick_place.tools import assess_pick as assess
    rng, A, recovery = prims.rng, prims.abstain_cost, prims.menu_recovery
    try:
        prims.rng = copy.deepcopy(rng)
        prims.abstain_cost, prims.menu_recovery = conf["abstain_cost"], conf["menu_recovery"]
        work = {k: copy.deepcopy(v) for k, v in store.items() if k != "pick_place"}
        return assess(object="object", destination=destination, care=conf["care"], primitives=prims, store=work).data
    finally:
        prims.rng, prims.abstain_cost, prims.menu_recovery = rng, A, recovery


# ---------------------------------------------------------------- a run's regret

def arm_score(choices: list[str], first_true: dict[str, Any], later_true: list[dict[str, Any]] | None) -> dict[str, Any]:
    """A run's regret: its first choice on the fresh true first menu, each later choice on the true menu from its own belief
    (``later_true``, one per later point; None when the replay diverged). A case without a first menu (out of reach) scores 0
    for every arm (no_menu); a first choice that is not a row of the true menu scores None."""
    if not choices and not first_true.get("assessed", False):
        return {"no_menu": True, "points": 0, "first": None, "later": [], "regret_first": 0.0, "regret_later": 0.0,
                "regret_total": 0.0}
    first = score(first_true, choices[0]) if choices else None
    later = None if later_true is None else [score(m, c) for m, c in zip(later_true, choices[1:])]
    r1 = None if first is None else first["regret"]
    rl = (None if later is None or len(later) != len(choices) - 1 or any(x is None for x in later)
          else float(sum(x["regret"] for x in later)))
    return {"no_menu": False, "points": len(choices), "first": first, "later": later, "regret_first": r1, "regret_later": rl,
            "regret_total": None if r1 is None or rl is None else r1 + rl}


def score_run(row: dict[str, Any], arm: str, setting: str, first_true: dict[str, Any], rep: dict[str, Any],
              key: str | None = None, start: list[str] = (), m0_ok: bool = True) -> dict[str, Any]:
    """The output record of one replayed run: its regret (``key``: its true menus' name in rep["true"]; None: its own menus
    are the true menus, as for the oracle), its checks (``start``: the reset against the record; any mismatch voids the
    later score) and, for the oracle, why a choice is not the cheapest row of its menu. ``m0_ok`` False (the true first
    menu did not match the oracle's record) voids the first-decision score; the oracle's own is its first_ok."""
    choices = actions(row, arm)
    mismatch = list(start) + rep["mismatch"]
    if not choices and row.get("ended") == "unreachable" and (rep["first"] or {}).get("assessed", False):
        mismatch.append("first menu: recorded out of reach, replayed a menu")
    later = None
    if not mismatch and len(rep["menus"]) == len(choices):
        later = rep["menus"][1:] if key is None else [t[key] for t in rep["true"][1:]]
    first_ok = not start and not any(m.startswith(("point 0 menu", "point 0:", "first menu")) for m in mismatch)
    m0_ok = first_ok if key is None else bool(m0_ok)
    s = arm_score(choices, first_true, later)
    if not m0_ok:
        s.update(regret_first=None, regret_total=None)
    out = {"object": row["object"], "idx": int(row["idx"]), "seed": int(row["seed"]), "arm": arm, "setting": setting,
           "rec": row.get("recommendation"), "ended": row.get("ended"), "artifact": bool(row.get("artifact")),
           "choices": choices, **s, "mismatch": mismatch, "first_ok": first_ok, "m0_ok": m0_ok}
    if arm in OWN_TRUE:
        out["miss"] = miss_reason(first_true, choices[0]) if choices and first_true.get("assessed", False) else None
        out["misses_later"] = [miss_reason(m, c) for m, c in zip(rep["menus"][1:], choices[1:])]
        out["true_first"] = {"rows": choosable(first_true), "recommendation": first_true.get("recommendation"),
                             "best_option": first_true.get("best_option"),
                             "guard_flag": bool((first_true.get("guard") or {}).get("flag"))}
    return out


# ---------------------------------------------------------------- the jobs on the lab

def _prims(cfg: dict[str, Any], obj: str) -> Any:
    from robots.mujoco_lab import calibrate

    prims = calibrate._primitives(os.path.join(cfg["crossfit"], f"fold_{cfg['folds'][obj]}"), None, None)
    prims.force_noise = calibrate.job_force_noise(None, prims.manifest)
    prims.probe_time_scale = 1.0
    prims.truth_spread = prims.prior_spread = 1.0
    return prims


def _start(prims: Any, conf: dict[str, Any], row: dict[str, Any]) -> tuple[Any, list[str]]:
    """Reset the lab to a recorded run's case under ``conf`` (as e1_gap.run_case and e1.run_job do); the tools and the
    reset's differences from the record."""
    prims.abstain_cost = conf["abstain_cost"]
    prims.menu_recovery = conf["menu_recovery"]
    prims.reset_s, prims.reset_price = conf["lab_recovery"]
    destination, reset_info, truth = e1.reset_case(prims, row["object"], int(row["idx"]), int(row["seed"]))
    bad = _diff({"destination": row.get("destination"), "truth": row.get("truth"), "reset": row.get("reset")},
                {"destination": destination, "truth": truth, "reset": reset_info})
    return e1.LabTools(prims, destination, conf["care"]), [f"reset {x}" for x in bad]


def _builder(prims: Any, confs: dict[str, dict[str, Any]]) -> Callable[[Any], dict[str, Any]]:
    return lambda tools: {name: true_menu(prims, tools.store, tools.destination, c) for name, c in confs.items()}


def gap_job(item: tuple[dict[str, Any], dict[str, Any]]) -> list[dict[str, Any]]:
    """One gap-check case: the oracle run of every setting (its first menu is the setting's true first menu), then the
    default run of each recovery with the true menus of that recovery's settings at its later points."""
    case, cfg = item
    t0 = time.time()
    prims, settings = _prims(cfg, case["object"]), cfg["settings"]
    out, first, m0_ok = [], {}, {}
    for name, s in settings.items():
        row = case["oracle"][name]
        tools, start = _start(prims, e1_gap.rule_config("oracle", s), row)
        rep = replay(tools, expectations(row, "oracle"))
        first[name] = rep["first"]
        out.append(score_run(row, "oracle", name, rep["first"], rep, None, start))
        m0_ok[name] = out[-1]["m0_ok"]
    for recovery, row in case["default"].items():
        names = [n for n, s in settings.items() if s["recovery"] == recovery]
        tools, start = _start(prims, e1_gap.rule_config("default", settings[names[0]]), row)
        build = _builder(prims, {n: e1_gap.rule_config("oracle", settings[n]) for n in names})
        rep = replay(tools, expectations(row, "default"), build)
        out += [score_run(row, "default", n, first[n], rep, n, start, m0_ok[n]) for n in names]
    prims.menu_recovery = None
    print(f"{case['object']} {case['idx']}: {len(out)} records, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    return out


def pilot_job(item: tuple[dict[str, Any], dict[str, Any]]) -> list[dict[str, Any]]:
    """One pilot (case, setting) job: the oracle rule's run (its first menu is the true first menu), then every other arm's
    run with the true menu at its later points."""
    job, cfg = item
    t0 = time.time()
    s = cfg["settings"][job["setting"]]
    prims = _prims(cfg, job["object"])
    rows = job["rows"]
    tools, start = _start(prims, e1_gap.rule_config("oracle", s), rows["oracle_rule"])
    rep = replay(tools, expectations(rows["oracle_rule"], "oracle_rule"))
    first = rep["first"]
    out = [score_run(rows["oracle_rule"], "oracle_rule", job["setting"], first, rep, None, start)]
    build = _builder(prims, {"true": e1_gap.rule_config("oracle", s)})
    for arm, row in rows.items():
        if arm == "oracle_rule":
            continue
        tools, start = _start(prims, e1_gap.rule_config("default", s), row)
        rep = replay(tools, expectations(row, arm), build)
        out.append(score_run(row, arm, job["setting"], first, rep, "true", start, out[0]["m0_ok"]))
    prims.menu_recovery = None
    print(f"{job['object']} {job['idx']} {job['setting']}: {len(out)} records, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    return out


def gap_items(src: str) -> list[dict[str, Any]]:
    """The gap check's cases with their runs: the oracle run of each setting and the default run of each recovery."""
    rows = e1.load_rows(_rows_path(src))
    cases: dict[tuple, dict[str, Any]] = {}
    for r in rows:
        c = cases.setdefault((r["object"], int(r["idx"]), int(r["seed"])),
                             {"object": r["object"], "idx": int(r["idx"]), "seed": int(r["seed"]), "oracle": {}, "default": {}})
        if r["role"] == "oracle":
            c["oracle"][r["setting"]] = r
        elif r["role"] == "default":
            c["default"][r["recovery"]] = r
    return [cases[k] for k in sorted(cases)]


def pilot_items(final: str, neutral: str | None) -> list[dict[str, Any]]:
    """The pilot's (case, setting) jobs with every arm's run; the neutral-text planners as <arm>_neutral."""
    rows = [dict(r, arm=r["arm"]) for r in e1.load_rows(_rows_path(final))]
    if neutral:
        rows += [dict(r, arm=f"{r['arm']}_neutral") for r in e1.load_rows(_rows_path(neutral))]
    jobs: dict[tuple, dict[str, Any]] = {}
    for r in rows:
        k = (r["object"], int(r["idx"]), int(r["seed"]), r["setting"])
        j = jobs.setdefault(k, {"object": r["object"], "idx": int(r["idx"]), "seed": int(r["seed"]), "setting": r["setting"], "rows": {}})
        if r["arm"] in j["rows"]:
            raise ValueError(f"two {r['arm']} runs of {k}")
        j["rows"][r["arm"]] = r
    return [jobs[k] for k in sorted(jobs)]


def _rows_path(d: str) -> str:
    p = os.path.join(d, "rows.jsonl")
    return p if os.path.exists(p) else p + ".gz"


def _job_key(x: dict[str, Any]) -> tuple:
    return x["object"], int(x["idx"]), int(x["seed"]), x.get("setting")


def run(kind: str, items: list[dict[str, Any]], cfg: dict[str, Any], source: dict[str, Any], workers: int, out_dir: str,
        limit: int | None = None) -> str:
    """Score every job (resuming: a job with records is skipped; a directory that holds another source is an error); returns
    the records path."""
    from robots.mujoco_lab.calibrate import code_revision

    held = os.path.join(out_dir, "source.json")
    if os.path.exists(held) and json.load(open(held)) != source:
        raise ValueError(f"{out_dir} holds another source ({held}); give a new directory")
    os.makedirs(out_dir, exist_ok=True)
    with open(held, "w") as f:
        json.dump(source, f, indent=1)
    with open(os.path.join(out_dir, f"run_config_{time.strftime('%Y%m%dT%H%M%S')}.json"), "w") as f:
        json.dump({"kind": kind, "source": source, "workers": workers, "limit": limit, "jobs": len(items),
                   "crossfit": cfg["crossfit"], "settings": cfg["settings"], "code": code_revision()}, f, indent=1)
    path = os.path.join(out_dir, "records.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {(r["object"], r["idx"], r["seed"], r["setting"] if kind == "pilot" else None)
                    for r in map(json.loads, filter(str.strip, f))}
    todo = [(x, cfg) for x in items[: limit or None] if _job_key(x) not in done]
    print(f"e1_expected {kind}: {len(todo)} of {len(items[: limit or None])} jobs to score", file=sys.stderr, flush=True)
    fn = gap_job if kind == "gap" else pilot_job
    with open(path, "a") as f:
        def write(parts) -> None:
            for part in parts:
                for r in part:
                    f.write(json.dumps(r, default=str) + "\n")
                f.flush()

        if workers <= 1:
            write(map(fn, todo))
        else:
            with mp.get_context("spawn").Pool(workers) as pool:
                write(pool.imap_unordered(fn, todo))
    return path


# ---------------------------------------------------------------- the summaries

def _key(r: dict[str, Any]) -> tuple[str, int, int]:
    return r["object"], int(r["idx"]), int(r["seed"])


def _drop(a: dict[str, Any], b: dict[str, Any]) -> str | None:
    return "error" if "error" in (a.get("ended"), b.get("ended")) else "artifact" if a.get("artifact") or b.get("artifact") else None


def _minus(a: dict[str, Any], b: dict[str, Any], k: str) -> float | None:
    return None if a.get(k) is None or b.get(k) is None else float(a[k]) - float(b[k])


def _mean(v: list[Any]) -> float | None:
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def _iv(prs: list[dict[str, Any]], field: str, name: str, n_boot: int) -> dict[str, Any]:
    sel = [p for p in prs if p[field] is not None]
    return e1_gap._interval([p[field] for p in sel], [p["object"] for p in sel], name, n_boot)


def gap_pairs(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One pair per (case, setting): default - oracle regret at the first decision ("first"), over every point ("total") and
    over the later points ("later"); dropped as in the realized-cost check (an error or an artifact in either run)."""
    runs = {(_key(r), r["arm"], r["setting"]): r for r in recs}
    out = []
    for c in sorted({k[0] for k in runs}):
        for s in dict.fromkeys(r["setting"] for r in recs):
            d, o = runs.get((c, "default", s)), runs.get((c, "oracle", s))
            if d is None or o is None:
                continue
            drop = _drop(d, o)
            out.append({"object": c[0], "idx": c[1], "seed": c[2], "setting": s, "rec": d.get("rec"), "kept": drop is None,
                        "drop": drop, "first": _minus(d, o, "regret_first"), "total": _minus(d, o, "regret_total"),
                        "later": _minus(d, o, "regret_later"), "default": d, "oracle": o})
    return out


def summarize_gap(prs: list[dict[str, Any]], n_boot: int = 2000) -> dict[str, Any]:
    """Per setting, over its kept pairs: default - oracle regret (first decision, every point, later points; every case and
    per first recommendation), each rule's mean regret, the power line on the first-decision gap (and n50 on the every-point
    gap), the pairs whose first choice differs, the oracle's choices off the cheapest row and the unscored pairs; the
    settings ranked by the lower bound of the first-decision interval."""
    out: dict[str, Any] = {"n_boot": n_boot, "settings": {}}
    for s in dict.fromkeys(p["setting"] for p in prs):
        ps = [p for p in prs if p["setting"] == s]
        kept = [p for p in ps if p["kept"]]
        x: dict[str, Any] = {"n": len(kept), "objects": len({p["object"] for p in kept}),
                             "dropped": {k: sum(p["drop"] == k for p in ps) for k in ("artifact", "error")}}
        for field in ("first", "total", "later"):
            x[field] = {"all": _iv(kept, field, f"{field}/{s}/all", n_boot)}
            for rec in e1_gap.RECS:
                x[field][rec] = _iv([p for p in kept if p["rec"] == rec], field, f"{field}/{s}/{rec}", n_boot)
        f1 = [p for p in kept if p["first"] is not None]
        ft = [p for p in kept if p["total"] is not None]
        x["power"] = {"n_half": e1_gap.power_n([p["first"] for p in f1], [p["object"] for p in f1], 0.5),
                      "n_full": e1_gap.power_n([p["first"] for p in f1], [p["object"] for p in f1], 1.0),
                      "n_half_total": e1_gap.power_n([p["total"] for p in ft], [p["object"] for p in ft], 0.5)}
        x["arms"] = {arm: {k: _mean([p[arm].get(k) for p in kept]) for k in ("regret_first", "regret_later", "regret_total", "points")}
                     for arm in ("default", "oracle")}
        x["differ"] = {"oracle_better": sum(p["first"] is not None and p["first"] > TOL for p in kept),
                       "oracle_worse": sum(p["first"] is not None and p["first"] < -TOL for p in kept)}
        x["oracle_miss"] = dict(Counter(p["oracle"]["miss"] for p in kept if p["oracle"].get("miss")))
        x["oracle_miss_later"] = dict(Counter(m for p in kept for m in p["oracle"].get("misses_later") or [] if m))
        x["oracle_points_later"] = sum(len(p["oracle"].get("misses_later") or []) for p in kept)
        x["unscored"] = {f: sum(p[f] is None for p in kept) for f in ("first", "total")}
        x["no_menu"] = sum(bool(p["default"].get("no_menu")) for p in kept)
        out["settings"][s] = x
    lower = {s: x["first"]["all"]["ci"][0] for s, x in out["settings"].items()}
    out["ranking"] = sorted(lower, key=lambda s: (lower[s] is None, -(lower[s] or 0.0)))
    return out


def pilot_pairs(recs: list[dict[str, Any]], arm: str, base: str = "default_rule") -> list[dict[str, Any]]:
    """One pair per (case, setting): ``arm`` - ``base`` regret (first decision, every point); dropped as in the pilot (an
    error or an artifact in either run)."""
    runs = {(_key(r), r["arm"], r["setting"]): r for r in recs}
    out = []
    for (c, a, s), r in sorted(runs.items()):
        b = runs.get((c, base, s))
        if a != arm or b is None:
            continue
        drop = _drop(r, b)
        out.append({"object": c[0], "idx": c[1], "seed": c[2], "setting": s, "kept": drop is None, "drop": drop,
                    "first": _minus(r, b, "regret_first"), "total": _minus(r, b, "regret_total")})
    return out


def summarize_pilot(recs: list[dict[str, Any]], n_boot: int = 2000) -> dict[str, Any]:
    """Each arm - the default rule (first decision and every point; per setting and pooled, kept pairs only), the arms' mean
    regret (all rows), the first choices that are not a row of the true menu, and the oracle's choices off the cheapest
    row."""
    settings = [s for s in PILOT_SETTINGS if any(r["setting"] == s for r in recs)]
    arms = [a for a in PILOT_ARMS if any(r["arm"] == a for r in recs)]
    out: dict[str, Any] = {"n_boot": n_boot, "settings": settings, "contrasts": {}, "arms": {}, "unmatched": {}}
    for arm in arms:
        prs = pilot_pairs(recs, arm)
        kept = [p for p in prs if p["kept"]]
        c: dict[str, Any] = {"dropped": len(prs) - len(kept), "unscored": {f: sum(p[f] is None for p in kept) for f in ("first", "total")}}
        for field in ("first", "total"):
            c[field] = {s: _iv([p for p in kept if s == "all" or p["setting"] == s], field, f"{arm}/{field}/{s}", n_boot)
                        for s in settings + ["all"]}
        out["contrasts"][arm] = c
    for arm in ["default_rule"] + arms:
        rows = [r for r in recs if r["arm"] == arm]
        out["arms"][arm] = {s: {"n": sum(s == "all" or r["setting"] == s for r in rows),
                                "first": _mean([r["regret_first"] for r in rows if s == "all" or r["setting"] == s]),
                                "total": _mean([r["regret_total"] for r in rows if s == "all" or r["setting"] == s])}
                            for s in settings + ["all"]}
        out["unmatched"][arm] = sum(r["regret_first"] is None and not r.get("no_menu") for r in rows)
    orc = [r for r in recs if r["arm"] == "oracle_rule"]
    out["oracle_miss"] = dict(Counter(r["miss"] for r in orc if r.get("miss")))
    out["oracle_miss_later"] = dict(Counter(m for r in orc for m in r.get("misses_later") or [] if m))
    return out


def checks(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """The replay checks per arm: the records, the first menus that did not match the record (for the oracle: the true first
    menu against the oracle's own record), the replays that diverged later, and the cases without a first menu."""
    out = {}
    for arm in dict.fromkeys(r["arm"] for r in recs):
        rs = [r for r in recs if r["arm"] == arm]
        bad = [r for r in rs if r["mismatch"]]
        out[arm] = {"records": len(rs), "first_mismatch": sum(not r["first_ok"] for r in rs),
                    "later_mismatch": sum(r["first_ok"] and bool(r["mismatch"]) for r in rs),
                    "no_menu": sum(bool(r.get("no_menu")) for r in rs), "m0_bad": sum(not r.get("m0_ok", True) for r in rs),
                    "examples": [f"{r['object']}/{r['idx']}/{r['setting']}: {r['mismatch'][0]}" for r in bad[:3]]}
    return out


def _ci(x: dict[str, Any], nd: int = 2) -> str:
    return "-" if x.get("mean") is None else f"{x['mean']:+.{nd}f} [{x['ci'][0]:+.{nd}f}, {x['ci'][1]:+.{nd}f}]"


def _f(v: float | None, nd: int = 2) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def _n(v: int | None) -> str:
    return "-" if v is None else str(v)


def markdown_gap(summary: dict[str, Any]) -> str:
    S = summary["settings"]
    lines = ["# E1 oracle-gap check: expected cost under the true setting", "",
             "Regret: a choice's expected cost under the true setting (the menu's own probabilities and probe pricing) minus "
             "the cheapest row of the same true menu; decline costs A. default - oracle: the default rule's regret minus the "
             "oracle rule's (above 0: the oracle is better); 95 % object-bootstrap intervals "
             f"({summary['n_boot']} draws). First decision: the first menu rebuilt at the true setting. Ranked by the lower "
             "bound of the first-decision interval. n50 (n100): the cases needed to show, at 80 % power, a planner that gets "
             "50 % (100 %) of the first-decision gap.", "",
             "| rank | setting | n (dropped) | default | oracle | default - oracle | probe cases | commit cases | decline cases "
             "| first choices differ (oracle better / worse) | n50 (n100) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, s in enumerate(summary["ranking"], 1):
        x = S[s]
        cells = " | ".join(f"{_ci(x['first'][r])} ({x['first'][r]['n']})" for r in e1_gap.RECS)
        lines.append(f"| {k} | {s} | {x['n']} ({sum(x['dropped'].values())}) | {_f(x['arms']['default']['regret_first'])} | "
                     f"{_f(x['arms']['oracle']['regret_first'])} | {_ci(x['first']['all'])} | {cells} | "
                     f"{x['differ']['oracle_better'] + x['differ']['oracle_worse']} ({x['differ']['oracle_better']} / "
                     f"{x['differ']['oracle_worse']}) | {_n(x['power']['n_half'])} ({_n(x['power']['n_full'])}) |")
    lines += ["", "Every decision point (the regret summed over every point where the run saw a menu; later true menus from "
              "each run's own belief):", "",
              "| setting | default | oracle | default - oracle | later points only | n50 | points per case (default / oracle) "
              "| unscored |", "|---|---|---|---|---|---|---|---|"]
    for s in summary["ranking"]:
        x = S[s]
        lines.append(f"| {s} | {_f(x['arms']['default']['regret_total'])} | {_f(x['arms']['oracle']['regret_total'])} | "
                     f"{_ci(x['total']['all'])} | {_ci(x['later']['all'])} | {_n(x['power']['n_half_total'])} | "
                     f"{_f(x['arms']['default']['points'])} / {_f(x['arms']['oracle']['points'])} | {x['unscored']['total']} |")
    lines += ["", "The oracle's choice off the cheapest row of its own (true) menu, by reason (kept pairs):", "",
              "| setting | first decision | later points (of all later points) |", "|---|---|---|"]
    for s in summary["ranking"]:
        x = S[s]
        lines.append(f"| {s} | {json.dumps(x['oracle_miss']) if x['oracle_miss'] else 0} | "
                     f"{json.dumps(x['oracle_miss_later']) if x['oracle_miss_later'] else 0} ({x['oracle_points_later']}) |")
    if summary.get("checks"):
        lines += ["", "Checks:", ""] + [f"- {k}: {json.dumps(v)}" for k, v in summary["checks"].items()]
    return "\n".join(lines) + "\n"


def markdown_pilot(summary: dict[str, Any]) -> str:
    S, cols = summary["settings"], summary["settings"] + ["all"]
    head = "| " + " | ".join(["contrast"] + cols) + " |"
    rule = "|" + "---|" * (len(cols) + 1)
    lines = ["# E1 pilot: expected cost under the true setting", "",
             "Regret as in the gap check (a choice's expected cost under the true setting minus the cheapest row of the same "
             "true menu). Each arm - default_rule, per setting and pooled (below 0: the arm's choices are cheaper under the "
             f"true setting); kept pairs (no error and no artifact in either run); 95 % object-bootstrap intervals "
             f"({summary['n_boot']} draws).", "", "First decision:", "", head, rule]
    for field, title in (("first", None), ("total", "Every decision point:")):
        if title:
            lines += ["", title, "", head, rule]
        for arm, c in summary["contrasts"].items():
            lines.append("| " + " | ".join([f"{arm} - default_rule"] + [f"{_ci(c[field][s])} ({c[field][s]['n']})" for s in cols]) + " |")
    lines += ["", "Mean regret per arm, all rows (first decision / every point):", "",
              "| " + " | ".join(["arm"] + cols) + " |", rule]
    for arm, a in summary["arms"].items():
        lines.append("| " + " | ".join([arm] + [f"{_f(a[s]['first'])} / {_f(a[s]['total'])}" for s in cols]) + " |")
    lines += ["", f"First choices that are not a row of the true menu: {json.dumps(summary['unmatched'])}.",
              f"The oracle off the cheapest row of its own menu: first decision {json.dumps(summary['oracle_miss'])}, later "
              f"points {json.dumps(summary['oracle_miss_later'])}."]
    if summary.get("checks"):
        lines += ["", "Checks:", ""] + [f"- {k}: {json.dumps(v)}" for k, v in summary["checks"].items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    from robots.mujoco_lab.train_models import repo_path

    ap = argparse.ArgumentParser(prog="e1_expected")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gap")
    g.add_argument("--src", required=True, help="the gap check's output directory (rows and cases)")
    p = sub.add_parser("pilot")
    p.add_argument("--final", required=True)
    p.add_argument("--neutral", default=None)
    p.add_argument("--plan", required=True, help="the pilot plan (its settings)")
    for q in (g, p):
        q.add_argument("--workers", type=int, default=8)
        q.add_argument("--limit", type=int, default=None)
        q.add_argument("--out", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--dir", required=True)
    s.add_argument("--n-boot", type=int, default=2000)
    s.add_argument("--wandb", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd in ("gap", "pilot"):
        crossfit = repo_path("robots/mujoco_lab/data/m2b2/folds")
        cfg = {"crossfit": crossfit, "folds": json.load(open(os.path.join(crossfit, "folds.json")))}
        if a.cmd == "gap":
            items, cfg["settings"] = gap_items(a.src), e1_gap.grid()
            source = {"kind": "gap", "src": os.path.abspath(a.src)}
        else:
            items = pilot_items(a.final, a.neutral)
            cfg["settings"] = {k: {f: v[f] for f in ("care_scale", "abstain_cost", "recovery")}
                               for k, v in json.load(open(a.plan))["settings"].items()}
            source = {"kind": "pilot", "final": os.path.abspath(a.final), "neutral": a.neutral and os.path.abspath(a.neutral),
                      "plan": os.path.abspath(a.plan)}
        print(run(a.cmd, items, cfg, source, a.workers, a.out, a.limit))
        return 0
    src = json.load(open(os.path.join(a.dir, "source.json")))
    path = os.path.join(a.dir, "records.jsonl")
    recs = e1.load_rows(path if os.path.exists(path) else path + ".gz")
    if src["kind"] == "gap":
        summ = summarize_gap(gap_pairs(recs), a.n_boot)
        metrics = {"settings": {s: {"first": x["first"]["all"]["mean"], "first_lo": x["first"]["all"]["ci"][0],
                                    "first_hi": x["first"]["all"]["ci"][1], "total": x["total"]["all"]["mean"],
                                    "n": x["n"], "n_half": x["power"]["n_half"], "rank": summ["ranking"].index(s) + 1}
                                for s, x in summ["settings"].items()}}
        md = markdown_gap
    else:
        summ = summarize_pilot(recs, a.n_boot)
        metrics = {"contrasts": {arm: {f: {s: c[f][s]["mean"] for s in summ["settings"] + ["all"]} for f in ("first", "total")}
                                 for arm, c in summ["contrasts"].items()}}
        md = markdown_pilot
    summ["checks"] = checks(recs)
    summ["source"] = src
    if a.wandb:
        from robots.mujoco_lab import tracking

        metrics["checks"] = {arm: c["first_mismatch"] + c["later_mismatch"] for arm, c in summ["checks"].items()}
        summ["wandb"] = tracking.log_run("e1_expected", {"dir": a.dir, "n_boot": a.n_boot, **src}, metrics)
        print(f"wandb: {summ['wandb']}")
    with open(os.path.join(a.dir, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    with open(os.path.join(a.dir, "summary.md"), "w") as f:
        f.write(md(summ))
    print(md(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
