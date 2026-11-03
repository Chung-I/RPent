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


"""First-menu decisions under each prior (the mass and CoM check, Part B): how often the generic or a VLM prior changes
the first menu's recommendation against the class prior.

The cases are calibrate's cross-fit cases (every object of the m2b2 folds times ``--repeats`` draws, ``--seed``): each
object's cases use its fold's bundle, and the generic and VLM priors fit their spreads on the other folds
(``estimation_check.job_prior``). Per case the lab is reset once to the case's draw (calibrate.run_case's reset); then per
prior and setting (care κ, abstain cost A) one assess_pick with a fresh store and the case's own menu stream, so every
prior sees the same scene, the same proposals and the same random numbers.

    python -P -m robots.mujoco_lab.first_menus run --bundle robots/mujoco_lab/data/m2b2 --scenes S --vlm qwen=Q --vlm gemma=G --out DIR
    python -P -m robots.mujoco_lab.first_menus summary --dir DIR
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from robots.mujoco_lab import tracking
from rpent.skills.pick_place.evaluate import object_weights

SETTINGS = (("k10_A10", 10.0, 10.0), ("k100_A3", 100.0, 3.0), ("k100_A10", 100.0, 10.0), ("k30_A3", 30.0, 3.0))
N_BOOT = 2000


def menu_fields(a: dict[str, Any]) -> dict[str, Any]:
    """The first menu's decision: the recommendation and its row, the best row, the guard flag and the rows' costs."""
    if "error" in a:
        return {"error": a["error"]}
    if not a.get("assessed", True):
        return {"error": f"not assessed: {a.get('reason')}"}
    g = a.get("guard") or {}
    return {"recommendation": a.get("recommendation"), "recommended_option": a.get("recommended_option"),
            "best_option": a.get("best_option"), "without_probe": a.get("without_probe"), "guard_flag": bool(g.get("flag")),
            "costs": {o["id"]: o.get("expected_cost") for o in a.get("options") or []}}


def _rate(flags: list[bool], objects: list[str], rng: np.random.Generator, n_boot: int) -> dict[str, Any]:
    f = np.asarray(flags, dtype=float)
    objs, inv = np.unique(np.asarray(objects), return_inverse=True)
    W = object_weights(len(objs), n_boot, rng)
    boot = (W @ np.bincount(inv, weights=f, minlength=len(objs))) / (W @ np.bincount(inv, minlength=len(objs)))
    return {"rate": float(f.mean()), "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def change_rates(records: list[dict[str, Any]], n_boot: int = N_BOOT, seed: int = 0) -> dict[str, dict[str, Any]]:
    """Per setting and prior other than class: the share of cases whose recommendation (commit, probe, decline) and whose
    recommended row differ from the class prior's, with 95 % object-bootstrap intervals, and the transitions. A case with
    an error under either prior is left out."""
    rng = np.random.default_rng(seed)
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by[r["setting"]].append(r)
    out: dict[str, dict[str, Any]] = {}
    for setting, recs in by.items():
        priors = sorted({p for r in recs for p in r["menus"]} - {"class"})
        block = {}
        for p in priors:
            ok = [r for r in recs if "error" not in r["menus"]["class"] and p in r["menus"] and "error" not in r["menus"][p]]
            objs = [r["object"] for r in ok]
            rec = [r["menus"][p]["recommendation"] != r["menus"]["class"]["recommendation"] for r in ok]
            opt = [r["menus"][p]["recommended_option"] != r["menus"]["class"]["recommended_option"] for r in ok]
            trans = Counter(f"{r['menus']['class']['recommendation']}->{r['menus'][p]['recommendation']}" for r, c in zip(ok, rec) if c)
            block[p] = {"n": len(ok), "n_left_out": len(recs) - len(ok), "recommendation": _rate(rec, objs, rng, n_boot),
                        "option": _rate(opt, objs, rng, n_boot), "transitions": dict(trans),
                        "class_recommendations": dict(Counter(r["menus"]["class"]["recommendation"] for r in ok)),
                        "prior_recommendations": dict(Counter(r["menus"][p]["recommendation"] for r in ok))}
        out[setting] = block
    return out


def prior_tables(source: str, scenes: str, vlm: dict[str, str]) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    """Per prior (generic, and one per VLM records file) the table of every object, each fold's objects with spreads fitted on
    the other folds (vision_prior.prior_table), and the spreads per prior and fold."""
    from robots.mujoco_lab.estimation_check import attempt_rows, case_truths, probe_rows, tuning_objects
    from robots.mujoco_lab.robot_spec import load_bundle
    from robots.mujoco_lab.vision_prior import prior_table

    folds = json.load(open(os.path.join(source, "folds", "folds.json")))
    classes = load_bundle(source).get("classes")
    rows = attempt_rows(source) + [r for r in probe_rows(source) if r["key"].rsplit("__", 1)[1] in ("Apush", "Bweigh")]
    truths = case_truths(rows, classes)
    flat = {o: bool(v["flat"]) for o, v in json.load(open(scenes)).items()}
    specs = {"generic": ("generic", None), **{n: ("vlm", [json.loads(line) for line in open(p)]) for n, p in vlm.items()}}
    tables: dict[str, dict[str, dict[str, Any]]] = {}
    spreads: dict[str, dict[int, Any]] = {}
    for name, (kind, records) in specs.items():
        tables[name], spreads[name] = {}, {}
        for k in sorted(set(folds.values())):
            tune = tuning_objects(folds, k)
            held = sorted(o for o, f in folds.items() if f == k)
            t, s = prior_table(kind, held, tune, {o: truths[o] for o in tune if o in truths}, flat, records)
            tables[name].update(t)
            spreads[name][k] = s
    return tables, spreads


def run_object(job: tuple[str, list[int], dict[str, Any]]) -> list[dict[str, Any]]:
    """One object's cases: one reset per case (calibrate.run_case's), then one first menu per setting and prior, each with a
    fresh store and the case's menu stream (calibrate._stream k 1)."""
    from robots.mujoco_lab import calibrate as cal
    from robots.mujoco_lab.adapter import object_props
    from robots.mujoco_lab.prior import fragility_world
    from robots.mujoco_lab.vision_prior import shape_prior
    from rpent.skills.pick_place.tools import assess_pick

    o, idxs, s = job
    prims = cal._primitives(s["bundle"], None, None)
    base = prims.prior
    priors = {"class": base, **{n: shape_prior(base, {o: t[o]}) for n, t in s["tables"].items()}}
    prims.force_noise = cal.job_force_noise(None, prims.manifest)      # calibrate._run_object's sensor
    env = prims.env
    env.force_noise = prims.force_noise
    world = fragility_world(prims.manifest)
    rows = []
    try:
        for idx in idxs:
            t0 = time.time()
            theta, com_end, destination, hidden = cal.case_draw(s["seed"], o, idx, object_props(o)["mass"], False, fragility=world)
            if world is not None:
                env.slosh_zeta, env.slosh_scale = world[1].slosh_zeta, world[1].slosh_scale
            env.classes = None if world is None else world[0]        # the reset's flags and damage law: the bundle's classes
            env.reset(o, theta, seed=s["seed"] * 1000 + idx, com_end=com_end,
                      fixture=destination if destination in cal.MESH_IDS else None, fill=hidden["fill"], frag=hidden["frag"])
            reset = dict(getattr(env, "reset_info", None) or {})
            for name, care, A in s["settings"]:
                menus = {}
                for pname, fn in priors.items():
                    prims.prior, prims.abstain_cost, prims.rng = fn, A, cal._stream(s["seed"], o, idx, 1)
                    try:
                        menus[pname] = menu_fields(assess_pick(object="object", destination=destination, care=care, primitives=prims,
                                                               store={}).data)
                    except Exception as exc:  # noqa: BLE001  one bad case must not stop the run
                        menus[pname] = {"error": str(exc)[:200]}
                rows.append({"object": o, "idx": idx, "fold": s["fold"], "setting": name, "care": care, "abstain_cost": A,
                             "destination": destination, "at_rest": reset.get("at_rest", True), "menus": menus})
            print(f"{o} {idx}: {time.time() - t0:.1f} s", file=sys.stderr, flush=True)
    finally:
        prims.prior = base
    return rows


def _setting(text: str) -> tuple[str, str | float, float]:
    """name:care:A, the care a number or low, normal, high."""
    name, care, A = text.split(":")
    return name, (care if care in ("low", "normal", "high") else float(care)), float(A)


def run(a: argparse.Namespace) -> str:
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "records.jsonl")
    if os.path.exists(path):
        raise SystemExit(f"{path} exists: pick a new --out")
    vlm = dict(v.split("=", 1) for v in a.vlm or [])
    tables, spreads = prior_tables(a.bundle, a.scenes, vlm)
    folds = json.load(open(os.path.join(a.bundle, "folds", "folds.json")))
    objects = a.objects.split(",") if a.objects else sorted(folds)
    settings = [_setting(t) for t in a.settings.split(",")] if a.settings else [(n, c, A) for n, c, A in SETTINGS]
    from robots.mujoco_lab.estimation_check import _revision
    with open(os.path.join(a.out, "run_config.json"), "w") as f:
        json.dump({**vars(a), **_revision(), "settings_parsed": settings, "spreads": spreads, "tables": tables}, f, indent=1)
    jobs = [(o, list(range(a.repeats)), {"bundle": os.path.join(a.bundle, "folds", f"fold_{folds[o]}"), "fold": folds[o], "seed": a.seed,
                                         "settings": settings, "tables": {n: {o: t[o]} for n, t in tables.items()}}) for o in objects]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool, open(path, "w") as out:
        for rows in pool.imap_unordered(run_object, jobs):
            for r in rows:
                out.write(json.dumps(r) + "\n")
            out.flush()
    print(f"done in {time.time() - t0:.0f} s: {path}", flush=True)
    return path


def summary(a: argparse.Namespace) -> dict[str, Any]:
    recs = [json.loads(line) for line in open(os.path.join(a.dir, "records.jsonl"))]
    out = {"n_records": len(recs), "rates": change_rates(recs, a.n_boot, a.seed)}
    with open(os.path.join(a.dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    lines = ["| setting | prior | n | recommendation changed [95 % CI] | recommended row changed [95 % CI] | transitions |",
             "|---|---|---|---|---|---|"]
    for setting, block in out["rates"].items():
        for p, v in block.items():
            r, o = v["recommendation"], v["option"]
            lines.append(f"| {setting} | {p} | {v['n']} | {r['rate']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] | "
                         f"{o['rate']:.3f} [{o['ci'][0]:.3f}, {o['ci'][1]:.3f}] | {json.dumps(v['transitions'])} |")
    md = "# First-menu decisions under each prior, against the class prior\n\n" + "\n".join(lines) + "\n"
    with open(os.path.join(a.dir, "summary.md"), "w") as f:
        f.write(md)
    print(md, flush=True)
    if a.wandb:
        cfg = {k: v for k, v in json.load(open(os.path.join(a.dir, "run_config.json"))).items() if k != "tables"}
        flat = {f"{setting}/{p}/{m}": v[m]["rate"] for setting, block in out["rates"].items() for p, v in block.items()
                for m in ("recommendation", "option")}
        out["wandb"] = tracking.log_run("first_menus", cfg, flat)
        with open(os.path.join(a.dir, "summary.json"), "w") as f:
            json.dump(out, f, indent=1)
    return out


def main(argv: list[str] | None = None) -> Any:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bundle", required=True, help="the m2b2 bundle (its folds/ hold the fold bundles)")
    r.add_argument("--scenes", required=True)
    r.add_argument("--vlm", action="append", help="name=records.jsonl (vlm_query.py output); repeatable")
    r.add_argument("--settings", default=None, help="name:care:A,... (default SETTINGS)")
    r.add_argument("--objects", default=None)
    r.add_argument("--seed", type=int, default=11)
    r.add_argument("--repeats", type=int, default=11)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--out", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--dir", required=True)
    s.add_argument("--n-boot", type=int, default=N_BOOT)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--wandb", action="store_true")
    a = ap.parse_args(argv)
    return run(a) if a.cmd == "run" else summary(a)


if __name__ == "__main__":
    main()
    sys.exit(0)
