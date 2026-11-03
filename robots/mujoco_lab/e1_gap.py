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

"""The E1 oracle-gap check (2026-11-02; rules only, no language model): in which hidden settings does the oracle rule
clearly beat the default rule? If none does, "beat the rule" may be close to impossible in this lab, and that is an answer
too.

The grid (the paper session's brief): care kappa in {1, 3, 30, 100} x task value A in {3, 10, 30} x recovery absorbing or
rec20, at the pilot's time price (s 1) and cost rule (e1.realized_cost); the default point (kappa 10, A 10, absorbing) is
not in it. The cases come from an H3 never run's cohort (every first recommendation, not only the probe cases), spread over
the objects, and every setting runs the same cases. The rules, as in e1.run_job:

* ``default``: the H3 policy (calibrate.run_protocol "policy") on the menu at the default (kappa 10, A 10, absorbing). Its
  actions do not depend on kappa or A (unlike e1.run_job, it keeps A 10 when the setting's A differs); the lab restores
  after an irreversible event under rec20, so it runs once per (case, recovery) and each setting rescores the run of its
  recovery;
* ``oracle``: the same policy on the menu at the setting's kappa, A and recovery, once per (case, setting).

Each case also runs two checks on the same lab: the oracle at the default point (``oracle_default``: it must repeat the
absorbing default run) and, after every other run, the default runs again (``default_repeat``: the determinism that the
rescoring relies on). The summary, per setting: default - oracle (95 % object-bootstrap intervals) over every case and per
first recommendation of the default menu, the cost parts, the decision rates, the pairs dropped (an error or an artifact in
either run) and the power line; the settings ranked by the lower bound of the interval.

    python -P -m robots.mujoco_lab.e1_gap cases --never NEVER.json.gz --exclude-plan plan.json --n 80 --out cases.json
    python -P -m robots.mujoco_lab.e1_gap run --cases cases.json --workers 8 --out DIR
    python -P -m robots.mujoco_lab.e1_gap summary --dir DIR [--wandb]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
import zlib
from collections import Counter
from typing import Any

import numpy as np

from robots.mujoco_lab import e1

KAPPAS = (1, 3, 30, 100)
TASK_VALUES = (3, 10, 30)
RECOVERIES = ("absorbing", "rec20")
SHORT = {"absorbing": "abs", "rec20": "rec20"}
RECS = ("probe", "commit", "decline")     # the default menu's first recommendation
Z_POWER = 1.959964 + 0.841621             # a 95 % interval above 0 at 80 % power
RECORD_KEYS = ("destination", "truth", "reset", "recommendation", "best_option", "guard", "first_menu", "probes", "attempts",
               "restores", "ended", "error", "artifact")


# ---------------------------------------------------------------- the grid and the rules

def grid(kappas=KAPPAS, task_values=TASK_VALUES, recoveries=RECOVERIES) -> dict[str, dict[str, Any]]:
    """The hidden settings, named k{kappa}_A{A}_{abs|rec20}."""
    return {f"k{k:g}_A{A:g}_{SHORT[r]}": {"care_scale": float(k), "abstain_cost": float(A), "recovery": r}
            for r in recoveries for k in kappas for A in task_values}


def rule_config(role: str, setting: dict[str, Any]) -> dict[str, Any]:
    """What a rule runs with under ``setting``: the menu's care, A and recovery (the default's are fixed) and the lab's
    restore (the setting's recovery for both rules)."""
    lab = e1.setting_recovery(setting)
    if role == "default":
        return {"care": float(e1.DEFAULT["care_scale"]), "abstain_cost": float(e1.DEFAULT["abstain_cost"]),
                "menu_recovery": e1.RECOVERY[e1.DEFAULT["recovery"]], "lab_recovery": lab}
    if role == "oracle":
        return {"care": float(setting["care_scale"]), "abstain_cost": float(setting["abstain_cost"]), "menu_recovery": lab,
                "lab_recovery": lab}
    raise ValueError(f"role must be default or oracle, not {role!r}")


# ---------------------------------------------------------------- scoring

def cost_parts(case: dict[str, Any], setting: dict[str, Any]) -> dict[str, float]:
    """e1.realized_cost in its three parts: the probes, the attempts and A (when the case ends without the object placed)."""
    from robots.mujoco_lab.calibrate import attempt_cost, probe_cost

    kappa, A = float(setting["care_scale"]), float(setting["abstain_cost"])
    reset_s, price = e1.setting_recovery(setting)
    p = float(sum(probe_cost(x["probe"], x["kind"], bool(x["thrown"]), kappa, 1.0, reset_s, price) for x in case.get("probes") or []))
    t = float(sum(attempt_cost(x["outcome"], float(x["cost_s"]), kappa, reset_s, price) for x in case.get("attempts") or []))
    a = 0.0 if case.get("ended") == "placed" else A
    return {"probes": p, "attempts": t, "A": a, "total": p + t + a}


def arm_record(case: dict[str, Any], setting: dict[str, Any]) -> dict[str, Any]:
    """A run scored under ``setting``: its cost parts and what the rule did."""
    return dict(cost_parts(case, setting), ended=case.get("ended"), probed=bool(case.get("probes")),
                abstained=case.get("ended") == "abstained", events=e1.irreversible_events(case))


def _case(r: dict[str, Any]) -> tuple[str, int, int]:
    return r["object"], int(r["idx"]), int(r["seed"])


def pairs(rows: list[dict[str, Any]], settings: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """One pair per (case, setting): the default run of the setting's recovery and the oracle run of the setting, both
    scored under the setting. ``kept`` is False when either run ended in an error (``drop`` "error") or has an artifact
    ("artifact"); ``rec`` is the default menu's first recommendation."""
    runs = {(_case(r), r["role"], r["recovery"] if r["role"] == "default" else r["setting"]): r
            for r in rows if r["role"] in ("default", "oracle")}
    out = []
    for c in sorted({k[0] for k in runs}):
        for name, s in settings.items():
            d, o = runs.get((c, "default", s["recovery"])), runs.get((c, "oracle", name))
            if d is None or o is None:
                continue
            drop = ("error" if "error" in (d.get("ended"), o.get("ended"))
                    else "artifact" if d.get("artifact") or o.get("artifact") else None)
            out.append({"object": c[0], "idx": c[1], "seed": c[2], "setting": name, "rec": d.get("recommendation"),
                        "kept": drop is None, "drop": drop, "default": arm_record(d, s), "oracle": arm_record(o, s)})
    return out


def power_n(d, objects, share: float = 0.5, z: float = Z_POWER) -> int | None:
    """The cases needed to show (a 95 % interval above 0 at 80 % power) a planner that gets ``share`` of the oracle gap: on
    a random ``share`` of the cases it pays the oracle's cost, on the rest the default's. From the gaps d (default - oracle)
    at N cases: the planner's mean gain is share x mean(d), and the object-clustered variance of that mean is
    (share (1 - share) sum d^2 + share^2 sum_o (S_o - mean(d) n_o)^2) / N^2 (S_o, n_o: an object's gap sum and its cases),
    which falls as N / N' at N' cases of the same mix; at least 1. None when the mean gap is not above 0."""
    v = np.asarray(d, dtype=float)
    if not len(v) or float(v.mean()) <= 0:
        return None
    N, mu = len(v), float(v.mean())
    index = {o: k for k, o in enumerate(sorted(set(objects)))}
    inv = np.array([index[o] for o in objects])
    S = np.bincount(inv, weights=v, minlength=len(index))
    n = np.bincount(inv, minlength=len(index)).astype(float)
    var = (share * (1.0 - share) * float((v ** 2).sum()) + share ** 2 * float(((S - mu * n) ** 2).sum())) / N ** 2
    return max(1, int(math.ceil(N * z ** 2 * var / (share * mu) ** 2)))


def _interval(values: list[float], objects: list[str], name: str, n_boot: int) -> dict[str, Any]:
    if not values:
        return {"n": 0, "objects": 0, "mean": None, "ci": [None, None]}
    mean, lo, hi = e1.object_bootstrap(values, objects, n_boot=n_boot, seed=zlib.crc32(name.encode()))
    return {"n": len(values), "objects": len(set(objects)), "mean": mean, "ci": [lo, hi]}


def _arm(recs: list[dict[str, Any]]) -> dict[str, Any]:
    if not recs:
        return {}
    out: dict[str, Any] = {k: float(np.mean([r[k] for r in recs])) for k in ("probes", "attempts", "A", "total")}
    out.update(probe=float(np.mean([r["probed"] for r in recs])), abstain=float(np.mean([r["abstained"] for r in recs])),
               irreversible=float(np.mean([r["events"] > 0 for r in recs])), events=int(sum(r["events"] for r in recs)))
    return out


def summarize(prs: list[dict[str, Any]], n_boot: int = 2000) -> dict[str, Any]:
    """Per setting, over its kept pairs: default - oracle (every case and per first recommendation), each rule's cost parts
    and decision rates, the pairs dropped and the power line (power_n at half and at all of the oracle gap); the settings
    ranked by the lower bound of the every-case interval."""
    out: dict[str, Any] = {"n_boot": n_boot, "settings": {}}
    for s in dict.fromkeys(p["setting"] for p in prs):
        ps = [p for p in prs if p["setting"] == s]
        kept = [p for p in ps if p["kept"]]
        d = [p["default"]["total"] - p["oracle"]["total"] for p in kept]
        objs = [p["object"] for p in kept]
        gap = {"all": _interval(d, objs, f"default - oracle/{s}/all", n_boot)}
        for rec in RECS:
            sel = [k for k, p in enumerate(kept) if p["rec"] == rec]
            gap[rec] = _interval([d[k] for k in sel], [objs[k] for k in sel], f"default - oracle/{s}/{rec}", n_boot)
        out["settings"][s] = {"n": len(kept), "objects": len(set(objs)),
                              "dropped": {"artifact": sum(p["drop"] == "artifact" for p in ps),
                                          "error": sum(p["drop"] == "error" for p in ps)},
                              "gap": gap, "arms": {arm: _arm([p[arm] for p in kept]) for arm in ("default", "oracle")},
                              "power": {"n_half": power_n(d, objs, 0.5), "n_full": power_n(d, objs, 1.0)}}
    lower = {s: x["gap"]["all"]["ci"][0] for s, x in out["settings"].items()}
    out["ranking"] = sorted(lower, key=lambda s: (lower[s] is None, -(lower[s] or 0.0)))
    return out


def _ci(x: dict[str, Any]) -> str:
    return "-" if x.get("mean") is None else f"{x['mean']:+.2f} [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}]"


def _n(v: int | None) -> str:
    return "-" if v is None else str(v)


def markdown(summary: dict[str, Any], title: str = "E1 oracle-gap check") -> str:
    S = summary["settings"]
    lines = [f"# {title}", "",
             "default - oracle: the default rule's realized cost minus the oracle rule's, per case (above 0: the oracle is "
             f"better); 95 % object-bootstrap intervals ({summary['n_boot']} draws). Ranked by the interval's lower bound. The "
             "last column: the cases needed to show a planner that gets 50 % of the oracle gap (in brackets: all of it), "
             "at 80 % power.", "",
             "| rank | setting | n (dropped) | default | oracle | default - oracle | probe cases | commit (grasp) cases "
             "| decline cases | n for 50 % (100 %) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for k, s in enumerate(summary["ranking"], 1):
        x = S[s]
        cells = " | ".join(f"{_ci(x['gap'][r])} ({x['gap'][r]['n']})" for r in RECS)
        lines.append(f"| {k} | {s} | {x['n']} ({x['dropped']['artifact'] + x['dropped']['error']}) | "
                     f"{x['arms']['default'].get('total', float('nan')):.2f} | {x['arms']['oracle'].get('total', float('nan')):.2f} | "
                     f"{_ci(x['gap']['all'])} | {cells} | {_n(x['power']['n_half'])} ({_n(x['power']['n_full'])}) |")

    def parts(r: dict[str, Any]) -> str:
        return f"{r['probes']:.2f} + {r['attempts']:.2f} + {r['A']:.2f}" if r else "-"

    def rates(r: dict[str, Any]) -> str:
        return f"{r['probe']:.2f} / {r['abstain']:.2f} / {r['irreversible']:.2f} ({r['events']})" if r else "-"

    lines += ["", "Cost parts (the mean per case: probes + attempts + A) and decision rates (the share of cases with a probe, "
              "an abstention, an irreversible event; in brackets the events):", "",
              "| setting | default parts | oracle parts | default probe / abstain / irreversible | oracle probe / abstain / irreversible |",
              "|---|---|---|---|---|"]
    for s in summary["ranking"]:
        a = S[s]["arms"]
        lines.append(f"| {s} | {parts(a['default'])} | {parts(a['oracle'])} | {rates(a['default'])} | {rates(a['oracle'])} |")
    if summary.get("checks"):
        lines += ["", "Checks:", ""] + [f"- {k}: {json.dumps(v)}" for k, v in summary["checks"].items()]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- the checks and the cases

def same_record(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Two runs of a case did the same: every RECORD_KEYS field is equal (the role, the setting and the timing aside)."""
    return all(json.dumps(a.get(k), sort_keys=True, default=str) == json.dumps(b.get(k), sort_keys=True, default=str)
               for k in RECORD_KEYS)


def _compare(twins: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    bad = [f"{a['object']}/{a['idx']}" for a, b in twins if not same_record(a, b)]
    return {"n": len(twins), "mismatch": len(bad), "examples": bad[:5]}


def checks(rows: list[dict[str, Any]], settings: dict[str, dict[str, Any]] | None = None,
           h3_recommendation: dict[str, Any] | None = None) -> dict[str, Any]:
    """The run's checks: ``repeat``, each default run against its repeat at the end of the job; ``oracle_at_default``, the
    oracle at the default point against the absorbing default run; ``recovery_without_event``, the absorbing default run
    against the rec20 one where the absorbing run had no irreversible event and no restore (the lab's recovery acts only
    after one); with ``settings``, ``oracle_protocol_cost``, the oracle's own cost against its realized cost; with
    ``h3_recommendation``, the default menu's first recommendation against H3's never run."""
    runs = {(_case(r), r["role"], r["setting"] if r["role"] == "oracle" else r["recovery"]): r for r in rows}
    defaults = [r for r in rows if r["role"] == "default"]
    out: dict[str, Any] = {
        "rows": dict(Counter(r["role"] for r in rows)), "errors": sum(r.get("ended") == "error" for r in rows),
        "artifacts": sum(bool(r.get("artifact")) for r in rows),
        "repeat": _compare([(r, t) for r in defaults if (t := runs.get((_case(r), "default_repeat", r["recovery"])))]),
        "oracle_at_default": _compare([(t, r) for r in rows if r["role"] == "oracle_default"
                                       and (t := runs.get((_case(r), "default", "absorbing")))]),
        "recovery_without_event": _compare([(r, t) for r in defaults if r["recovery"] == "absorbing" and not r.get("restores")
                                            and not e1.irreversible_events(r)
                                            and (t := runs.get((_case(r), "default", "rec20")))])}
    if settings is not None:
        oc = [r for r in rows if r["role"] == "oracle" and r.get("ended") != "error" and r.get("protocol_cost") is not None]
        out["oracle_protocol_cost"] = {"n": len(oc), "mismatch": sum(
            abs(float(r["protocol_cost"]) - e1.realized_cost(r, settings[r["setting"]])) > 1e-9 for r in oc)}
    if h3_recommendation is not None:
        twins = [(r, h3_recommendation.get(f"{r['object']}/{r['idx']}")) for r in defaults if r["recovery"] == "absorbing"]
        bad = [f"{r['object']}/{r['idx']}" for r, h in twins if r.get("recommendation") != h]
        out["h3_recommendation"] = {"n": len(twins), "mismatch": len(bad), "examples": bad[:5]}
    return out


def pick_cases(never: dict[str, Any], n: int, plan_seed: int, exclude=frozenset()) -> dict[str, Any]:
    """n cases of an H3 never run's cohort (every first recommendation), spread over the objects (e1.spread_cases), none
    of ``exclude`` ((seed, object, idx)); with H3's first recommendation of each."""
    seed = int(never["seed"])
    pool: dict[str, list[int]] = {}
    recs: dict[str, Any] = {}
    for r in never["rows"]:
        o, i = r["object"], int(r["idx"])
        if (seed, o, i) not in exclude:
            pool.setdefault(o, []).append(i)
            recs[f"{o}/{i}"] = r.get("recommendation")
    picked = e1.spread_cases(pool, n, plan_seed)
    return {"seed": seed, "plan_seed": int(plan_seed), "n": len(picked), "cases": [[o, int(i)] for o, i in picked],
            "h3_recommendation": {f"{o}/{i}": recs[f"{o}/{i}"] for o, i in picked}}


# ---------------------------------------------------------------- the runs

def _plan(settings: dict[str, dict[str, Any]]) -> list[tuple[str, str | None, dict[str, Any]]]:
    """A case's runs in order: (role, setting name, setting)."""
    point = dict(e1.DEFAULT)
    defaults = [dict(point, recovery=r) for r in RECOVERIES]
    return ([("default", None, s) for s in defaults] + [("oracle", name, s) for name, s in settings.items()]
            + [("oracle_default", None, point)] + [("default_repeat", None, s) for s in defaults])


def run_case(item: tuple[dict[str, Any], dict[str, Any]]) -> list[dict[str, Any]]:
    """Every rule run of one case, in turn on one lab (in _plan's order); a row per run with the case record."""
    from robots.mujoco_lab import calibrate

    case, cfg = item
    obj, idx, seed = case["object"], int(case["idx"]), int(case["seed"])
    prims = calibrate._primitives(os.path.join(cfg["crossfit"], f"fold_{cfg['folds'][obj]}"), None, None)
    prims.force_noise = calibrate.job_force_noise(None, prims.manifest)
    prims.probe_time_scale = 1.0
    prims.truth_spread = prims.prior_spread = 1.0
    rows = []
    for role, name, setting in _plan(cfg["settings"]):
        conf = rule_config("default" if role.startswith("default") else "oracle", setting)
        t0 = time.time()
        prims.abstain_cost = conf["abstain_cost"]
        prims.menu_recovery = conf["menu_recovery"]
        prims.reset_s, prims.reset_price = conf["lab_recovery"]
        destination, reset_info, truth = e1.reset_case(prims, obj, idx, seed)
        try:
            rec = calibrate.run_protocol(prims, "policy", destination, np.random.default_rng(0), care=conf["care"])
        except Exception as exc:  # noqa: BLE001  one bad run must not stop the job
            rec = {"ended": "error", "error": f"{type(exc).__name__}: {exc}"[:300], "probes": [], "attempts": []}
        rec["artifact"] = calibrate.case_artifact(rec, reset_info)
        rec["protocol_cost"] = rec.pop("cost", None)      # the run's own cost (its menu's care and A)
        rows.append(dict(rec, object=obj, idx=idx, seed=seed, role=role, setting=name, recovery=setting["recovery"],
                         menu_care=conf["care"], menu_A=conf["abstain_cost"], destination=destination, truth=truth,
                         reset=reset_info, seconds=round(time.time() - t0, 2)))
    prims.menu_recovery = None
    print(f"{obj} {idx}: {len(rows)} runs, {sum(r['seconds'] for r in rows):.0f}s", file=sys.stderr, flush=True)
    return rows


def _counts(path: str) -> Counter:
    if not os.path.exists(path):
        return Counter()
    with open(path) as f:
        return Counter(_case(json.loads(line)) for line in f if line.strip())


def run(cases_path: str, workers: int, out_dir: str, crossfit: str | None = None, limit: int | None = None) -> str:
    """Run every case's rule runs (resuming: a case with all its rows is skipped; a directory that holds another cases
    file is an error); returns the rows path."""
    from robots.mujoco_lab.calibrate import code_revision
    from robots.mujoco_lab.train_models import repo_path

    spec = json.load(open(cases_path))
    held = os.path.join(out_dir, "cases.json")
    if os.path.exists(held) and json.load(open(held)) != spec:
        raise ValueError(f"{out_dir} holds another cases file ({held}); give a new directory")
    generator = e1.run_generator(out_dir)
    crossfit = crossfit or repo_path("robots/mujoco_lab/data/m2b2/folds")
    settings = grid()
    per = len(_plan(settings))
    cfg = {"settings": settings, "crossfit": crossfit, "folds": json.load(open(os.path.join(crossfit, "folds.json")))}
    os.makedirs(out_dir, exist_ok=True)
    if os.path.abspath(cases_path) != os.path.abspath(os.path.join(out_dir, "cases.json")):
        shutil.copyfile(cases_path, os.path.join(out_dir, "cases.json"))
    with open(os.path.join(out_dir, f"run_config_{time.strftime('%Y%m%dT%H%M%S')}.json"), "w") as f:
        json.dump({"cases": os.path.abspath(cases_path), "n_cases": spec["n"], "workers": workers, "crossfit": crossfit,
                   "generator": generator, "limit": limit, "default": e1.DEFAULT, "settings": settings, "runs_per_case": per,
                   "code": code_revision()}, f, indent=1)
    path = os.path.join(out_dir, "rows.jsonl")
    have = _counts(path)
    partial = [c for c, k in have.items() if k != per]
    if partial:
        raise ValueError(f"{len(partial)} cases in {path} do not have {per} rows, e.g. {partial[:3]}")
    cases = [{"object": o, "idx": int(i), "seed": int(spec["seed"])} for o, i in spec["cases"]][: limit or None]
    todo = [(c, cfg) for c in cases if _case(c) not in have]
    print(f"e1_gap run: {len(todo)} of {len(cases)} cases to run, {per} runs each", file=sys.stderr, flush=True)
    with open(path, "a") as f:
        def write(parts) -> None:
            for part in parts:
                for r in part:
                    f.write(json.dumps(r, default=str) + "\n")
                f.flush()

        if workers <= 1:
            write(map(run_case, todo))
        else:
            with mp.get_context("spawn").Pool(workers) as pool:
                write(pool.imap_unordered(run_case, todo))
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="e1_gap")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cases")
    c.add_argument("--never", required=True, help="an H3 never run (policy_no_probe, absorbing, wide) of one seed")
    c.add_argument("--exclude-plan", default=None, help="an e1 plan whose final cases are left out")
    c.add_argument("--n", type=int, default=80)
    c.add_argument("--plan-seed", type=int, default=1)
    c.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cases", required=True)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--out", required=True)
    r.add_argument("--crossfit", default=None, help="the bundle's crossfit folds (default: data/m2b2/folds)")
    s = sub.add_parser("summary")
    s.add_argument("--dir", required=True)
    s.add_argument("--n-boot", type=int, default=2000)
    s.add_argument("--wandb", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "cases":
        with (gzip.open if a.never.endswith(".gz") else open)(a.never, "rt") as f:
            never = json.load(f)
        exclude = set()
        if a.exclude_plan:
            exclude = {(int(j["seed"]), j["object"], int(j["idx"])) for j in json.load(open(a.exclude_plan))["jobs"]
                       if j["split"] == "final"}
        spec = dict(pick_cases(never, a.n, a.plan_seed, exclude), source=os.path.abspath(a.never),
                    exclude_plan=a.exclude_plan and os.path.abspath(a.exclude_plan), excluded=len(exclude))
        with open(a.out, "w") as f:
            json.dump(spec, f, indent=1)
        print(f"cases: {spec['n']} over {len({o for o, _ in spec['cases']})} objects at seed {spec['seed']}, H3 first "
              f"recommendations {dict(Counter(spec['h3_recommendation'].values()))} -> {a.out}")
        return 0
    if a.cmd == "run":
        print(run(a.cases, a.workers, a.out, crossfit=a.crossfit, limit=a.limit))
        return 0
    path = os.path.join(a.dir, "rows.jsonl")
    rows = e1.load_rows(path if os.path.exists(path) else path + ".gz")
    spec = json.load(open(os.path.join(a.dir, "cases.json")))
    settings = grid()
    summ = summarize(pairs(rows, settings), a.n_boot)
    summ["checks"] = checks(rows, settings, spec.get("h3_recommendation"))
    summ["grid"] = settings
    if a.wandb:
        from robots.mujoco_lab import tracking

        metrics = {"settings": {s: {"gap": x["gap"]["all"]["mean"], "gap_lo": x["gap"]["all"]["ci"][0],
                                    "gap_hi": x["gap"]["all"]["ci"][1], "n": x["n"], "default": x["arms"]["default"].get("total"),
                                    "oracle": x["arms"]["oracle"].get("total"), "n_half": x["power"]["n_half"],
                                    "rank": summ["ranking"].index(s) + 1} for s, x in summ["settings"].items()},
                   "checks": {k: v["mismatch"] for k, v in summ["checks"].items() if isinstance(v, dict) and "mismatch" in v}}
        summ["wandb"] = tracking.log_run("e1_gap", {"dir": a.dir, "cases": spec.get("source"), "n_cases": spec["n"],
                                                    "seed": spec["seed"], "plan_seed": spec["plan_seed"], "grid": settings,
                                                    "n_boot": a.n_boot}, metrics)
        print(f"wandb: {summ['wandb']}")
    with open(os.path.join(a.dir, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    with open(os.path.join(a.dir, "summary.md"), "w") as f:
        f.write(markdown(summ))
    print(markdown(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
