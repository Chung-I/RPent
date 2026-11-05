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


"""Experiment 1, step 7, exploratory, added after the results (the coordinator's request of 2026-11-06): why the
realized gain is above the predicted one. On the main reading's records (S = all differs from a_prior, no probe on
either side, a kept seed), per cell and side (the rows a_prior chose, the rows a_S chose):
  the model's outcome probabilities at the truth particle (the prediction evpi_props.row_true_cost prices) next to the
  simulator's outcome, one draw per record (a row's label is the same at every seed; ``tables`` refuses otherwise);
  per irreversible outcome, the expected count under the model and the Poisson probability of the observed count or
  fewer;
  J split into its parts (``j_parts``: irreversible at 5 x the care factor, recoverable, A x P(not placed), time) next
  to the same parts of the realized cost.
Also per cell: the changed records by transition (decline -> row, row -> decline, row -> row) with their mean realized
gain, and the S = all mean and object-bootstrap interval without the object of the largest |gain| (a sensitivity; the
pre-stated readings do not change). ``levels`` rebuilds each record's beliefs and first menus (evpi_sim.replay_case,
which must pass its gate) and scores the chosen rows at the truth: the model only, no simulation. ``tables`` refuses a
J that differs from the record's.

    python -P -m robots.mujoco_lab.evpi_sim_levels levels --sample FILE --out DIR [--objects O,...]
    python -P -m robots.mujoco_lab.evpi_sim_levels summary --sample FILE --levels D1,... --runs D1,... --out DIR [--wandb]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from robots.mujoco_lab import calibrate as cal
from robots.mujoco_lab import evpi_props as ev
from robots.mujoco_lab import evpi_sim as es
from robots.mujoco_lab import tracking
from robots.mujoco_lab.com_gap import _mean_ci
from rpent.skills.pick_place.costs import COST, TIME_COST_PER_S, care_scale
from rpent.skills.pick_place.protocol import IRREVERSIBLE, OUTCOMES

LABEL = "exploratory, added after the results"
PARTS = ("irreversible", "recoverable", "unfinished", "time", "total")
SIDES = (("a_prior", ev.PRIOR), ("a_S", es.MAIN))
TRANSITIONS = ("decline → row", "row → decline", "row → row")
J_TOL = 1e-6


def j_parts(p: dict[str, float], cost_s: float, care: float, A: float) -> dict[str, float]:
    """J of a row at one particle (costs.particle_costs, absorbing) split into its parts: the irreversible outcomes at
    5 x the care factor, the recoverable ones, A x P(not placed) ("unfinished") and the row's time."""
    scale = care_scale(care)
    irr = sum(float(p.get(k, 0.0)) * COST[k] * scale for k in IRREVERSIBLE)
    rec = sum(float(p.get(k, 0.0)) * COST[k] for k in OUTCOMES if k not in IRREVERSIBLE)
    unf = float(A) * (1.0 - float(p.get("placed_stable", 0.0)))
    t = TIME_COST_PER_S * float(cost_s)
    return {"irreversible": irr, "recoverable": rec, "unfinished": unf, "time": t, "total": irr + rec + unf + t}


def realized_parts(label: str, cost_s: float, care: float, A: float) -> dict[str, float]:
    """The same parts of an executed row's realized cost (they sum to evpi_sim.row_cost)."""
    return j_parts({label: 1.0}, cost_s, care, A)


def poisson_le(k: int, lam: float) -> float:
    """P(X <= k) for X ~ Poisson(lam)."""
    if lam <= 0:
        return 1.0
    return float(sum(math.exp(-lam + i * math.log(lam) - math.lgamma(i + 1)) for i in range(int(k) + 1)))


def top_object(objs: list[str], d: list[float]) -> str:
    """The object of the record with the largest |gain| (a tie: the first object in sorted order)."""
    m = max(abs(x) for x in d)
    return min(o for o, x in zip(objs, d) if abs(x) == m)


def _transition(k0: str, k1: str) -> str:
    if k0 == "abstain":
        return "decline → decline" if k1 == "abstain" else "decline → row"
    return "row → decline" if k1 == "abstain" else "row → row"


def _wanted(r: dict[str, Any]) -> bool:
    """A record of the main reading's kind: a_S = all differs from a_prior, and neither is a probe."""
    return (r["action"][es.MAIN] != r["action"][ev.PRIOR]
            and not any(str(r["choice"][k]).startswith("probe_") for k in (ev.PRIOR, es.MAIN)))


# ---------------------------------------------------------------- the tables

def _targets(recs: list[dict[str, Any]], plans: list[dict[str, Any]], runs: list[dict[str, Any]]) -> dict[tuple, list[dict[str, Any]]]:
    """Per (setting, prior), the main reading's records: each with its plan, its S = all pair (evpi_sim.record_pairs)
    and, per side, its simulations at the kept seeds (both actions ran without an error or an artifact)."""
    pl = {(p["object"], p["idx"], p["prior"], p["setting"]): p for p in plans}
    by_case: dict[tuple, dict[tuple, dict[str, Any]]] = defaultdict(dict)
    for x in runs:
        by_case[(x["object"], x["idx"])][(tuple(x["job"]), int(x["seed_k"]))] = x
    out: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        if r["action"][es.MAIN] == r["action"][ev.PRIOR]:
            continue
        p = pl.get((r["object"], r["idx"], r["prior"], r["setting"]))
        if p is None or p["mismatch"]:
            raise SystemExit(f"{r['object']} {r['idx']} {r['prior']} {r['setting']}: not run, or its replay differed")
        cs = by_case[(r["object"], r["idx"])]
        q = es.record_pairs(r, p, cs).get(es.MAIN)
        if q is None or q["probe_pair"] or q["kept"] == 0:
            continue
        bs = p["beliefs"]

        def sim(key: str, k: int) -> dict[str, Any] | None:
            jk = es.job_key(p["prior"], p["setting"], key, bs[key])
            return None if jk is None else cs.get((jk, k))

        kept = [k for k in range(es.N_SEEDS) if es._ok(bs[ev.PRIOR], sim(ev.PRIOR, k)) and es._ok(bs[es.MAIN], sim(es.MAIN, k))]
        out[(r["setting"], r["prior"])].append({"r": r, "plan": p, "q": q,
                                                "runs": {key: [sim(key, k) for k in kept] for _, key in SIDES}})
    return out


def _mean(v: list[float]) -> float | None:
    return float(np.mean(v)) if v else None


def tables(recs: list[dict[str, Any]], plans: list[dict[str, Any]], runs: list[dict[str, Any]], levels: list[dict[str, Any]],
           n_boot: int = es.N_BOOT, seed: int = 0) -> dict[str, Any]:
    """The per-side model-next-to-simulator tables, the transitions and the sensitivity (module docstring)."""
    rng = np.random.default_rng(seed)
    lev: dict[tuple, dict[str, Any]] = {}
    for x in levels:
        k = (x["object"], x["idx"], x["prior"], x["setting"], x["key"])
        if k in lev:
            raise SystemExit(f"two levels for {k}")
        lev[k] = x
    sides: dict[str, dict[str, Any]] = defaultdict(dict)
    transitions: dict[str, dict[str, Any]] = defaultdict(dict)
    sensitivity: dict[str, dict[str, Any]] = defaultdict(dict)
    diffs: list[float] = []
    for (setting, prior), items in sorted(_targets(recs, plans, runs).items()):
        cell: dict[str, Any] = {}
        for side, key in SIDES:
            rows = []
            for it in items:
                r, b = it["r"], it["plan"]["beliefs"][key]
                if b["kind"] == "abstain":
                    continue
                L = lev.get((r["object"], r["idx"], r["prior"], r["setting"], key))
                if L is None:
                    raise SystemExit(f"no level for {r['object']} {r['idx']} {r['prior']} {r['setting']} {key}")
                if L["choice"] != r["choice"][key]:
                    raise SystemExit(f"{r['object']} {r['idx']} {key}: the level's choice {L['choice']} is not the record's")
                diffs.append(abs(float(L["J"]) - float(r["J"][key])))
                labels = {x["label"] for x in it["runs"][key]}
                if len(labels) != 1:
                    raise SystemExit(f"{r['object']} {r['idx']} {key}: a row's label changes across seeds ({sorted(labels)})")
                rows.append((L["p"], labels.pop(), float(b["cost_s"]), float(r["care"]), float(r["abstain_cost"])))
            observed = Counter(label for _, label, *_ in rows)
            expected = {k: float(sum(float(p.get(k, 0.0)) for p, *_ in rows)) for k in OUTCOMES}
            pois = {k: poisson_le(observed.get(k, 0), expected[k]) for k in IRREVERSIBLE if expected[k] > 0 or observed.get(k)}
            pois["irreversible_any"] = poisson_le(sum(observed.get(k, 0) for k in IRREVERSIBLE), sum(expected[k] for k in IRREVERSIBLE))
            cell[side] = {
                "rows": len(rows), "observed": dict(observed), "expected": expected, "poisson_le": pois,
                "p_mean": {k: expected[k] / len(rows) for k in OUTCOMES if rows and (expected[k] > 0 or observed.get(k))},
                "model": {part: _mean([j_parts(p, s, c, A)[part] for p, _, s, c, A in rows]) for part in PARTS},
                "realized": {part: _mean([realized_parts(lab, s, c, A)[part] for _, lab, s, c, A in rows]) for part in PARTS},
                "p_placed_mean": _mean([float(p.get("placed_stable", 0.0)) for p, *_ in rows]),
                "placed_observed": observed.get("placed_stable", 0)}
        sides[setting][prior] = cell
        groups: dict[str, list[float]] = defaultdict(list)
        for it in items:
            bs = it["plan"]["beliefs"]
            groups[_transition(bs[ev.PRIOR]["kind"], bs[es.MAIN]["kind"])].append(it["q"]["d"])
        transitions[setting][prior] = {g: {"n": len(v), "mean": _mean(v)}
                                       for g, v in sorted(groups.items(), key=lambda t: (TRANSITIONS + (t[0],)).index(t[0]))}
        objs, d = [it["r"]["object"] for it in items], [it["q"]["d"] for it in items]
        top = top_object(objs, d)
        keep = [(o, x) for o, x in zip(objs, d) if o != top]
        c = _mean_ci(np.array([x for _, x in keep], dtype=float), [o for o, _ in keep], rng, n_boot)
        sensitivity[setting][prior] = {"dropped_object": top, "dropped_records": len(items) - len(keep), "n": len(keep),
                                       "n_objects": len({o for o, _ in keep}), "mean": c["mean"], "ci": c["ci"],
                                       "all_records": {"n": len(items), "mean": _mean(d)}}
    check = {"rows": len(diffs), "max_abs_diff": max(diffs) if diffs else 0.0}
    if check["max_abs_diff"] > J_TOL:
        bad = sum(1 for x in diffs if x > J_TOL)
        raise SystemExit(f"J differs from the record's for {bad} of {len(diffs)} rows (max {check['max_abs_diff']:.3g})")
    return {"label": LABEL, "sides": dict(sides), "transitions": dict(transitions), "sensitivity": dict(sensitivity),
            "j_check": check, "n_boot": n_boot, "seed": seed}


def _n(x: float | None, nd: int = 3) -> str:
    return "" if x is None else f"{x:+.{nd}f}"


def markdown(t: dict[str, Any]) -> str:
    cells = [(s, p) for s, p in es.CELLS if s in t["sides"] and p in t["sides"][s]]
    out = [f"# Experiment 1, step 7: why the realized gain is above the predicted one ({t['label']})", "",
           "The main reading's records (S = all, non-probe pairs). A row's outcome is one draw per record: its label is "
           "the same at every seed. The model's P is its prediction at the truth particle, which J prices. "
           f"J check: {t['j_check']['rows']} rows, max |J - the record's J| {t['j_check']['max_abs_diff']:.2g}.", "",
           "## Outcomes: the model's mean P at the truth and the expected count, next to the observed count", "",
           "| cell | prior | side | rows | outcome | mean P | expected count | observed | Poisson P(observed or fewer) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for s, p in cells:
        for side, _ in SIDES:
            c = t["sides"][s][p][side]
            for k, mp in c["p_mean"].items():
                po = c["poisson_le"].get(k)
                out.append(f"| {s} | {p} | {side} | {c['rows']} | {k} | {mp:.4f} | {c['expected'][k]:.2f} | "
                           f"{c['observed'].get(k, 0)} | {'' if po is None else f'{po:.3f}'} |")
            irr_e = sum(c["expected"][k] for k in IRREVERSIBLE)
            irr_o = sum(c["observed"].get(k, 0) for k in IRREVERSIBLE)
            out.append(f"| {s} | {p} | {side} | {c['rows']} | irreversible (any) | {irr_e / c['rows'] if c['rows'] else 0:.4f} | "
                       f"{irr_e:.2f} | {irr_o} | {c['poisson_le']['irreversible_any']:.3f} |")
    out += ["", "## J split into its parts: the mean over the rows, the model at the truth / the simulator", "",
            "| cell | prior | side | rows | irreversible | recoverable | A × P(not placed) | time | total |",
            "|---|---|---|---|---|---|---|---|---|"]
    for s, p in cells:
        for side, _ in SIDES:
            c = t["sides"][s][p][side]
            out.append(f"| {s} | {p} | {side} | {c['rows']} | "
                       + " | ".join(f"{_n(c['model'][k])} / {_n(c['realized'][k])}" for k in PARTS) + " |")
    out += ["", "## The changed records by transition: count and mean realized gain", "",
            "| cell | prior | " + " | ".join(TRANSITIONS) + " |", "|---|---|" + "---|" * len(TRANSITIONS)]
    for s, p in cells:
        tr = t["transitions"][s][p]
        out.append(f"| {s} | {p} | " + " | ".join(f"{tr[g]['n']}, {_n(tr[g]['mean'])}" if g in tr else "0" for g in TRANSITIONS)
                   + " |")
    extra = sorted({g for s, p in cells for g in t["transitions"][s][p]} - set(TRANSITIONS))
    if extra:
        out.append(f"Other transitions: {extra}.")
    out += ["", "## Sensitivity: S = all without the object of the largest |gain| (report only; the readings do not change)",
            "", "| cell | prior | object left out (its records) | records (objects) | mean realized gain [95 %] | all records |",
            "|---|---|---|---|---|---|"]
    for s, p in cells:
        v = t["sensitivity"][s][p]
        ci = "" if v["ci"] is None else f" [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]"
        out.append(f"| {s} | {p} | {v['dropped_object']} ({v['dropped_records']}) | {v['n']} ({v['n_objects']}) | "
                   f"{_n(v['mean'])}{ci} | {v['all_records']['n']}, {_n(v['all_records']['mean'])} |")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- the model at the truth (no simulation)

def levels_object(job: tuple[str, list[dict[str, Any]], dict[str, Any]]) -> list[dict[str, Any]]:
    """One object's records (``_wanted``): the replay (its gate must pass), then for a_prior's and a_S's row the first
    menu again (as the replay builds it, under the record's prior), the row's prediction at the truth particle (the S = all
    belief) and its J as evpi_props.row_true_cost prices it."""
    from robots.mujoco_lab.prior import fragility_world
    from robots.mujoco_lab.vision_prior import shape_prior
    from rpent.skills.pick_place import menu as menu_mod
    from rpent.skills.pick_place.costs import particle_costs, prediction_matrix

    o, recs, s = job
    prims = cal._primitives(s["bundle"], None, None)
    base = prims.prior
    priors = {"class": base, "generic": shape_prior(base, {o: s["generic"][o]})}
    prims.force_noise = cal.job_force_noise(None, prims.manifest)
    prims.env.force_noise = prims.force_noise
    world = fragility_world(prims.manifest)
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        by[int(r["idx"])].append(r)
    out: list[dict[str, Any]] = []
    try:
        for idx in sorted(by):
            t0 = time.time()
            rc = es.replay_case(prims, o, idx, by[idx], s, priors, world)
            bad = [(p["prior"], p["setting"], p["mismatch"]) for p in rc["plans"] if p["mismatch"]]
            if bad:
                raise SystemExit(f"replay gate: {o} {idx} {bad}")
            for r in by[idx]:
                tb = rc["ctx"][(r["prior"], r["setting"], es.MAIN)][0]
                for _, key in SIDES:
                    if r["choice"][key] in ev.ABSTAINS:
                        continue
                    belief, care, A, state = rc["ctx"][(r["prior"], r["setting"], key)]
                    prims.prior = priors[r["prior"]]
                    m = ev.first_menu(prims, belief, rc["destination"], care, A, state)
                    c = ev.choice_of(m["payload"])
                    if c != r["choice"][key] or ev.action_key(c, m["menu"]) != r["action"][key]:
                        raise SystemExit(f"{o} {idx} {r['prior']} {r['setting']} {key}: the menu again gives {c}")
                    opt = m["menu"][c]
                    per = menu_mod.estimate(prims.model, tb, opt._base, opt._pose, opt.action["force"], opt._planned)[1]
                    P = prediction_matrix(per)[0]
                    out.append({"object": o, "idx": idx, "prior": r["prior"], "setting": r["setting"], "key": key, "choice": c,
                                "action": r["action"][key], "cost_s": float(opt.cost_s), "p": dict(zip(OUTCOMES, map(float, P))),
                                "J": float(particle_costs(per, opt.cost_s, care, A, None, "time")[0]),
                                "J_record": float(r["J"][key])})
            print(f"{o} {idx}: {time.time() - t0:.1f} s, {len(by[idx])} records", file=sys.stderr, flush=True)
    finally:
        prims.prior = base
    return out


def levels(a: argparse.Namespace) -> str:
    from robots.mujoco_lab.estimation_check import _revision

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "levels.jsonl")
    if os.path.exists(path):
        raise SystemExit(f"{a.out} holds a run: pick a new --out")
    smp = json.load(open(a.sample))
    cfg = smp["config"]
    recs = [r for r in smp["records"] if _wanted(r)]
    if a.objects:
        keep = set(a.objects.split(","))
        recs = [r for r in recs if r["object"] in keep]
    folds = json.load(open(os.path.join(cfg["bundle"], "folds", "folds.json")))
    with open(os.path.join(a.out, "run_config.json"), "w") as f:
        json.dump({**vars(a), **_revision(), "label": LABEL, "sample_sha256": es._sha(a.sample),
                   "records": [[r["object"], r["idx"], r["prior"], r["setting"]] for r in recs]}, f, indent=1)
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        by[r["object"]].append(r)
    t0 = time.time()
    with open(path, "w") as f:
        for o, rs in sorted(by.items()):
            s = {"bundle": os.path.join(cfg["bundle"], "folds", f"fold_{folds[o]}"), "seed": cfg["seed"],
                 "generic": {o: cfg["generic"][o]}}
            for row in levels_object((o, rs, s)):
                f.write(json.dumps(row) + "\n")
            f.flush()
    print(f"done in {time.time() - t0:.0f} s: {a.out}", flush=True)
    return a.out


def summary(a: argparse.Namespace) -> dict[str, Any]:
    smp = json.load(open(a.sample))
    sha = es._sha(a.sample)
    plans, runs, cfg = es.load_runs(a.runs.split(","))
    if cfg.get("sample_sha256") != sha:
        raise SystemExit("the runs used another sample file")
    levs, lcfg = [], None
    for d in a.levels.split(","):
        c = json.load(open(os.path.join(d, "run_config.json")))
        if c.get("dirty") or c.get("sample_sha256") != sha:
            raise SystemExit(f"{d}: dirty code or another sample file")
        if lcfg is not None and c.get("code_revision") != lcfg.get("code_revision"):
            raise SystemExit(f"{d}: another code revision")
        lcfg = lcfg or c
        levs += [json.loads(line) for line in open(os.path.join(d, "levels.jsonl")) if line.strip()]
    t = tables(smp["records"], plans, runs, levs, a.n_boot, a.seed)
    os.makedirs(a.out, exist_ok=True)
    md = markdown(t)
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write(md)
    print(md, flush=True)
    if a.wandb:
        flat: dict[str, Any] = {"j_check/max_abs_diff": t["j_check"]["max_abs_diff"], "j_check/rows": t["j_check"]["rows"]}
        for s, block in t["sides"].items():
            for p, cell in block.items():
                for side, c in cell.items():
                    pre = f"{s}/{p}/{side}"
                    flat[f"{pre}/rows"] = c["rows"]
                    flat[f"{pre}/p_placed_mean"] = c["p_placed_mean"]
                    flat[f"{pre}/placed_observed"] = c["placed_observed"]
                    flat[f"{pre}/irreversible_expected"] = sum(c["expected"][k] for k in IRREVERSIBLE)
                    flat[f"{pre}/irreversible_observed"] = sum(c["observed"].get(k, 0) for k in IRREVERSIBLE)
                    flat[f"{pre}/irreversible_poisson_le"] = c["poisson_le"]["irreversible_any"]
                    for part in PARTS:
                        flat[f"{pre}/model/{part}"] = c["model"][part]
                        flat[f"{pre}/realized/{part}"] = c["realized"][part]
                v = t["sensitivity"][s][p]
                flat[f"{s}/{p}/sensitivity/mean"] = v["mean"]
                flat[f"{s}/{p}/sensitivity/lo"] = None if v["ci"] is None else v["ci"][0]
                flat[f"{s}/{p}/sensitivity/hi"] = None if v["ci"] is None else v["ci"][1]
        conf = {"label": LABEL, "sample": a.sample, "sample_sha256": sha, "runs_code": cfg.get("code_revision"),
                "levels_code": (lcfg or {}).get("code_revision"), "n_boot": a.n_boot, "seed": a.seed}
        t["wandb"] = tracking.log_run("evpi_sim_levels", conf, flat)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(t, f, indent=1)
    return t


def main(argv: list[str] | None = None) -> Any:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("levels")
    v.add_argument("--sample", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--objects", default=None)
    s = sub.add_parser("summary")
    s.add_argument("--sample", required=True)
    s.add_argument("--levels", required=True, help="the levels run's shard dirs, comma-separated")
    s.add_argument("--runs", required=True, help="the step 7 run's shard dirs, comma-separated")
    s.add_argument("--out", required=True)
    s.add_argument("--n-boot", type=int, default=es.N_BOOT)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--wandb", action="store_true")
    a = ap.parse_args(argv)
    return {"levels": levels, "summary": summary}[a.cmd](a)


if __name__ == "__main__":
    main()
