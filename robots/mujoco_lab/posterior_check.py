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


"""Posterior check: does one attempt's update give honest 80 % intervals?

For each corpus row of the chosen objects: the lab prior's particles (the generator, so the truth
is a prior draw), one update with the row's outcome and readouts (the bundle's outcome model and
readout model), then the 80 % intervals of mass, friction and the CoM fraction along the pivot.
Coverage is reported overall and by label, beside the prior's.

The truth is the row's theta in the prior's hypothesis space (``generator_truth``): the mass
nominal x scale x (1 + f), the friction drawn, and the generator's CoM fraction. The particles live
in that space. The sim's CoM differs from the generator's by a model error (corpus v3: median 4e-4,
max 0.031 of the pivot extent), and the prior puts about 0.7 of its weight on one point (the
unshifted CoM), so a posterior on that point has a zero-width interval that a sub-millimetre error
falls outside. Coverage against the sim's values (``sim_truth``) is reported beside it. With a temper grid and
``--choose-on``, the temper whose coverage of all three lies in [0.75, 0.85] and is closest to 0.80
(worst of the three; ties: the smaller temper) goes into the bundle (manifest ``tempers.readout``);
when no temper is feasible the bundle keeps 1 and the result says so.

``--probes A|B`` checks the update after the probes instead, on the bundle's probe corpus: B, the
weigh probe alone; A, the push and then the weigh from the state the push left (the push alone when
it threw the object). The grid then tempers the probe models (manifest ``tempers.probe``).

    python -P -m robots.mujoco_lab.posterior_check --bundle robots/mujoco_lab/data/m2 --split val \\
        --temper-grid 0.7,0.85,1,1.25,1.5,2,3 --choose-on val --out robots/mujoco_lab/data/m2/posterior_check.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from robots.mujoco_lab import tracking
from robots.mujoco_lab.prior import com_hypothesis, lab_prior
from robots.mujoco_lab.robot_spec import load_bundle
from robots.mujoco_lab.train_models import load_rows, write_manifest
from rpent.skills.pick_place.belief import ParticleBelief
from rpent.skills.pick_place.evaluate import rows_of
from rpent.skills.pick_place.features import model_names, particle_matrix, particle_sides

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COVER = (0.75, 0.85)
TARGET = 0.80
KEYS = (("mass", "mass"), ("mu", "mu"), ("com", "com_frac"))


def true_com_frac(row: dict[str, Any]) -> float:
    """The row's true CoM as a fraction along the pivot axis (the particles' com_frac)."""
    pv = row["pose"]["pivot"]
    c, ax = np.asarray(pv["centre"], dtype=float), np.asarray(pv["axis"], dtype=float)
    return float(((np.asarray(row["com_obj_true"], dtype=float) - c) @ ax - float(pv["lo"])) / float(pv["L"]))


def sim_truth(row: dict[str, Any]) -> dict[str, float]:
    """The sim's values: the row's truth mass and friction, and the CoM fraction of ``com_obj_true``."""
    return {"mass": float(row["truth"]["mass"]), "mu": float(row["truth"]["mu"]), "com": true_com_frac(row)}


def generator_truth(row: dict[str, Any]) -> dict[str, float]:
    """The row's theta as a point of the prior's space: mass nominal x scale x (1 + f), the friction drawn,
    the CoM fraction of ``com_hypothesis`` (the particles' arithmetic)."""
    pose, th = row["pose"], row["theta"]
    com, f = com_hypothesis(pose["pivot"], pose["com_obj_nominal"], th["com_s"], int(row["com_end"]))
    pv = pose["pivot"]
    frac = ((np.asarray(com) - np.asarray(pv["centre"], dtype=float)) @ np.asarray(pv["axis"], dtype=float) - float(pv["lo"])) / float(pv["L"])
    return {"mass": float(pose["nominal_mass"]) * float(th["mass_scale"]) * (1.0 + f), "mu": float(th["mu"]), "com": float(frac)}


def _hits(b: ParticleBelief, truth: dict[str, float]) -> dict[str, bool]:
    return {k: bool(b.quantile(key, 0.1) <= truth[k] <= b.quantile(key, 0.9)) for k, key in KEYS}


def _widths(b: ParticleBelief) -> dict[str, float]:
    return {k: b.quantile(key, 0.9) - b.quantile(key, 0.1) for k, key in KEYS}


def check_rows(rows: list[dict[str, Any]], bundle: dict[str, Any], grid: list[float], n: int, seed: int) -> list[dict[str, Any]]:
    """One case per row: the prior's hits and widths, and per temper the posterior's (``*_sim``: against the sim's values)."""
    model, readout = bundle["model"], bundle["readout_model"]
    rng = np.random.default_rng(seed)
    cases = []
    for r in rows:
        b = lab_prior(r["pose"], n, rng)
        a = b.arrays()
        X = particle_matrix(r["features"], a, r["pose"], r["force"], model_names(model))
        P = model.predict_matrix(X, r.get("planned"))[0]
        obs = {"label": r["label"], "readouts": r["readouts"], "force": r["force"]}
        comp, info = readout.components(obs, P, X, a, particle_sides(a, r["pose"]))
        truth, sim = generator_truth(r), sim_truth(r)
        case = {"object": r["object"], "label": r["label"], "info": info, "prior": _hits(b, truth), "prior_sim": _hits(b, sim),
                "prior_width": _widths(b), "post": {}, "post_sim": {}, "post_width": {}}
        for t in grid:
            post = b.reweight(readout.combine(comp, t))
            case["post"][t], case["post_sim"][t], case["post_width"][t] = _hits(post, truth), _hits(post, sim), _widths(post)
        cases.append(case)
    return cases


PROBE_TAGS = {"A": ("Apush", "Aweigh"), "B": ("Bweigh",)}


def probe_cases(rows: list[dict[str, Any]], probes: str) -> list[list[dict[str, Any]]]:
    """One case per draw: the scene's probe rows in order (A: push, then weigh when the push left the
    object on the table; B: weigh)."""
    tags = PROBE_TAGS[probes]
    by: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        tag = r["key"].rsplit("__", 1)[1]
        if tag in tags:
            by[(r["object"], int(r["draw"]))][tag] = r
    return [[d[t] for t in tags if t in d] for _, d in sorted(by.items()) if tags[0] in d]


def check_probe_cases(cases: list[list[dict[str, Any]]], bundle: dict[str, Any], grid: list[float], n: int, seed: int) -> list[dict[str, Any]]:
    """Like check_rows, with the probe models' likelihood of each case's probe results."""
    pms = bundle["probe_models"]
    rng = np.random.default_rng(seed)
    out = []
    for case in cases:
        first = case[0]
        b = lab_prior(first["pose"], n, rng)
        a = b.arrays()
        terms = [(pms[r["probe"]], {"kind": r["kind"], "reading": r["reading"]},
                  particle_matrix(r["base"], a, r["pose"], r["force"], model_names(pms[r["probe"]])), float(r["force"])) for r in case]
        truth, sim = generator_truth(first), sim_truth(first)
        c = {"object": first["object"], "label": "+".join(r["kind"] for r in case), "n_probes": len(case),
             "info": {"used": [], "rejected": []}, "prior": _hits(b, truth), "prior_sim": _hits(b, sim), "prior_width": _widths(b),
             "post": {}, "post_sim": {}, "post_width": {}}
        for j, t in enumerate(grid):
            ll = np.zeros(len(b.weights))
            for pm, obs, X, force in terms:
                lt, info = pm.loglik(obs, X, a, force, temper=t)
                ll = ll + lt
                if j == 0:
                    c["info"]["used"] += [f"{pm.probe}_{k}" for k in info["used"]]
                    c["info"]["rejected"] += [f"{pm.probe}_{k}" for k in info["rejected"]]
            post = b.reweight(ll)
            c["post"][t], c["post_sim"][t], c["post_width"][t] = _hits(post, truth), _hits(post, sim), _widths(post)
        out.append(c)
    return out


def coverage(hits: list[dict[str, bool]]) -> dict[str, float | None]:
    return {k: (float(np.mean([h[k] for h in hits])) if hits else None) for k, _ in KEYS}


def _inside(cov: dict[str, float | None]) -> bool:
    return all(cov[k] is not None and COVER[0] <= cov[k] <= COVER[1] for k, _ in KEYS)


def choose_temper(grid_cov: dict[float, dict[str, float | None]]) -> tuple[float, bool]:
    """The feasible temper closest to 0.80 (worst of the three; ties: the smaller), else (1.0, False)."""
    feasible = [t for t, c in grid_cov.items() if _inside(c)]
    if not feasible:
        return 1.0, False
    return min(feasible, key=lambda t: (max(abs(grid_cov[t][k] - TARGET) for k, _ in KEYS), t)), True


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--split", default="val", choices=("train", "val", "test"))
    ap.add_argument("--objects", default=None, help="comma-separated ids in place of the split's")
    ap.add_argument("--n-particles", type=int, default=2000)
    ap.add_argument("--temper-grid", default=None, help="comma-separated tempers (always with 1 and the bundle's)")
    ap.add_argument("--choose-on", default=None, choices=("train", "val"), help="choose the temper on this split; write it into the bundle")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probes", default=None, choices=tuple(PROBE_TAGS), help="check the update after the probes (A: push then weigh; B: weigh)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    B = load_bundle(a.bundle)
    man = B["manifest"]
    metrics = json.load(open(os.path.join(a.bundle, man["metrics"])))

    def path_of(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    if a.probes:
        if not B["probe_models"]:
            raise SystemExit("the bundle has no probe models: train_models --probe-corpus PATH --out-dir BUNDLE")
        rows = [r for r in load_rows(path_of(man["probe_corpus"])) if "pose" in r and "com_end" in r]
        temper0 = float(next(iter(B["probe_models"].values())).temper)
    else:
        if B["readout_model"] is None:
            raise SystemExit("the bundle has no readout model: train it on corpus v2 rows (they carry readouts)")
        paths = metrics["corpus"] if isinstance(metrics["corpus"], list) else [metrics["corpus"]]
        rows = [r for p in paths for r in load_rows(path_of(p))]
        rows = [r for r in rows if r.get("readouts") and "pose" in r and "com_obj_true" in r and "com_end" in r]
        temper0 = float(B["readout_model"].temper)

    def rows_for(split: str) -> list[dict[str, Any]]:
        ids = a.objects.split(",") if (a.objects and split == a.split) else metrics["split"][split]
        return rows_of(rows, ids)

    def check(split: str, seed: int) -> list[dict[str, Any]]:
        if a.probes:
            return check_probe_cases(probe_cases(rows_for(split), a.probes), B, grid, a.n_particles, seed)
        return check_rows(rows_for(split), B, grid, a.n_particles, seed)

    grid = sorted({float(t) for t in (a.temper_grid.split(",") if a.temper_grid else [])} | {1.0, temper0})
    t0 = time.time()
    cases = check(a.split, a.seed)
    if not cases:
        raise SystemExit(f"no rows for split {a.split!r}" + (f" with probes {a.probes}" if a.probes else " with readouts"))
    pick = cases if a.choose_on in (None, a.split) else check(a.choose_on, a.seed + 1)
    grid_cov = {t: coverage([c["post"][t] for c in pick]) for t in grid}
    if a.choose_on:
        temper, feasible = choose_temper(grid_cov)
    else:
        temper = temper0
        feasible = _inside(coverage([c["post"][temper] for c in cases]))
    cov = coverage([c["post"][temper] for c in cases])
    by_label: dict[str, list[dict[str, bool]]] = defaultdict(list)
    for c in cases:
        by_label[c["label"]].append(c["post"][temper])
    out = {
        "bundle": a.bundle, "split": a.split, "probes": a.probes, "objects": sorted({c["object"] for c in cases}), "n": len(cases),
        "n_particles": a.n_particles, "temper": temper, "feasible": feasible, "chosen_on": a.choose_on,
        "mass_cov80": cov["mass"], "mu_cov80": cov["mu"], "com_cov80": cov["com"],
        "acceptance_b": {"bounds": list(COVER), "pass": _inside(cov)},
        "prior_cov80": coverage([c["prior"] for c in cases]), "truth": "generator",
        "sim_truth": {"cov80": coverage([c["post_sim"][temper] for c in cases]), "prior_cov80": coverage([c["prior_sim"] for c in cases])},
        "by_kind" if a.probes else "by_label": {lbl: {"n": len(h), **{f"{k}_cov80": v for k, v in coverage(h).items()}}
                                                for lbl, h in sorted(by_label.items())},
        "grid": {str(t): grid_cov[t] for t in grid},
        "width80_median": {"prior": {k: float(np.median([c["prior_width"][k] for c in cases])) for k, _ in KEYS},
                           "posterior": {k: float(np.median([c["post_width"][temper][k] for c in cases])) for k, _ in KEYS}},
        "terms": {"used": dict(Counter(t for c in cases for t in c["info"]["used"])),
                  "rejected": dict(Counter(t for c in cases for t in c["info"]["rejected"]))},
        "seconds": round(time.time() - t0, 1),
    }
    if a.probes == "A":
        out["n_with_weigh"] = sum(c["n_probes"] == 2 for c in cases)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    if a.choose_on:
        write_manifest(a.bundle, tempers={**(man.get("tempers") or {}), ("probe" if a.probes else "readout"): temper})
    out["wandb"] = tracking.log_run("posterior_check", {"bundle": a.bundle, "split": a.split, "probes": a.probes, "n_particles": a.n_particles,
                                                        "grid": grid, "choose_on": a.choose_on, "seed": a.seed},
                                    {"cov80": cov, "temper": temper, "feasible": feasible, "prior_cov80": out["prior_cov80"],
                                     "sim_cov80": out["sim_truth"]["cov80"],
                                     "grid": out["grid"], "width80_median": out["width80_median"], "n": len(cases)})
    print(json.dumps({k: out[k] for k in ("n", "temper", "feasible", "mass_cov80", "mu_cov80", "com_cov80", "prior_cov80",
                                          "sim_truth", "acceptance_b", "width80_median", "terms", "seconds")}, indent=1))
    return out


if __name__ == "__main__":
    main()
    sys.exit(0)
