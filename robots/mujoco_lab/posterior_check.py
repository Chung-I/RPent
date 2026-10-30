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

``--probes A|B|C`` checks the update after the probes instead, on the bundle's probe corpus: B, the
weigh probe alone; A, the push and then the weigh from the state the push left (the push alone when
it threw the object); C, the squeeze alone (crushable and bruisable objects). The probe models see each
row's public fragility entry (``row_pose``). The grid then tempers the probe models (manifest ``tempers.probe``).

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
from robots.mujoco_lab.prior import com_hypothesis, joint_hidden, lab_prior
from robots.mujoco_lab.robot_spec import load_bundle
from robots.mujoco_lab.train_models import load_rows, repo_path, write_manifest
from rpent.skills.pick_place.belief import ParticleBelief
from rpent.skills.pick_place.evaluate import _ece, ece_ci, object_weights, row_base, row_pose, rows_of
from rpent.skills.pick_place.features import model_names, particle_matrix, particle_sides
from rpent.skills.pick_place.likelihood import as_law

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COVER = (0.75, 0.85)
TARGET = 0.80
KEYS = (("mass", "mass"), ("mu", "mu"), ("com", "com_frac"))
MIN_CASES, MIN_OBJECTS = 30, 4    # plan 2b2 decision 9: a coverage cell below either is insufficient
MIN_ECE_CASES = 50                # and a P(filled) ECE below this
P_FILLED_ECE = 0.05
N_BOOT = 2000
JUDGED = {None: ("mass", "mu", "com"), "A": ("mass", "mu"), "B": ("mass", "mu"), "C": ()}


def true_com_frac(row: dict[str, Any]) -> float:
    """The row's true CoM as a fraction along the pivot axis (the particles' com_frac)."""
    pv = row["pose"]["pivot"]
    c, ax = np.asarray(pv["centre"], dtype=float), np.asarray(pv["axis"], dtype=float)
    return float(((np.asarray(row["com_obj_true"], dtype=float) - c) @ ax - float(pv["lo"])) / float(pv["L"]))


def sim_truth(row: dict[str, Any]) -> dict[str, float]:
    """The sim's values: the row's truth mass and friction, and the CoM fraction of ``com_obj_true``."""
    return {"mass": float(row["truth"]["mass"]), "mu": float(row["truth"]["mu"]), "com": true_com_frac(row)}


def generator_truth(row: dict[str, Any], classes: dict[str, dict[str, Any]] | None = None) -> dict[str, float]:
    """The row's theta as a point of the prior's space: mass nominal x scale x (1 + f), the friction drawn,
    the CoM fraction of ``com_hypothesis`` (the particles' arithmetic). A filled row adds its liquid as the m2b
    particles do (prior.joint_hidden: the joint mass and CoM), from the object's cavity in ``classes``."""
    pose, th = row["pose"], row["theta"]
    pv = pose["pivot"]
    fill = float(row.get("fill") or 0.0)
    if fill > 0.0:
        entry = (classes or {}).get(row.get("object")) or {}
        if not entry.get("cavity"):
            raise ValueError(f"{row.get('object')}: a filled row's truth needs the object's cavity (the fragility classes)")
        joint = joint_hidden(pose, entry, th, int(row["com_end"]), fill)
        mass, com = float(joint["mass"]), joint["com_obj"]
    else:
        com, f = com_hypothesis(pv, pose["com_obj_nominal"], th["com_s"], int(row["com_end"]))
        mass = float(pose["nominal_mass"]) * float(th["mass_scale"]) * (1.0 + f)
    frac = ((np.asarray(com) - np.asarray(pv["centre"], dtype=float)) @ np.asarray(pv["axis"], dtype=float) - float(pv["lo"])) / float(pv["L"])
    return {"mass": mass, "mu": float(th["mu"]), "com": float(frac)}


def _hits(b: ParticleBelief, truth: dict[str, float]) -> dict[str, bool]:
    return {k: bool(b.quantile(key, 0.1) <= truth[k] <= b.quantile(key, 0.9)) for k, key in KEYS}


def _widths(b: ParticleBelief) -> dict[str, float]:
    return {k: b.quantile(key, 0.9) - b.quantile(key, 0.1) for k, key in KEYS}


def check_rows(rows: list[dict[str, Any]], bundle: dict[str, Any], grid: list[float], n: int, seed: int) -> list[dict[str, Any]]:
    """One case per row: the prior's hits and widths, and per temper the posterior's (``*_sim``: against the sim's values)."""
    model, readout = bundle["model"], bundle["readout_model"]
    prior = bundle.get("prior") or lab_prior                  # the bundle's manifest names it
    rng = np.random.default_rng(seed)
    cases = []
    for r in rows:
        b = prior(r["pose"], n, rng)
        a = b.arrays()
        X = particle_matrix(row_base(r), a, row_pose(r), r["force"], model_names(model))
        P = model.predict_matrix(X, r.get("planned"))[0]
        obs = {"label": r["label"], "readouts": r["readouts"], "force": r["force"]}
        comp, info = readout.components(obs, P, X, a, particle_sides(a, r["pose"]))
        truth, sim = generator_truth(r, bundle.get("classes")), sim_truth(r)
        case = {"object": r["object"], "label": r["label"], "info": info, "prior": _hits(b, truth), "prior_sim": _hits(b, sim),
                "prior_width": _widths(b), "post": {}, "post_sim": {}, "post_width": {}}
        for t in grid:
            post = b.reweight(readout.combine(comp, t))
            case["post"][t], case["post_sim"][t], case["post_width"][t] = _hits(post, truth), _hits(post, sim), _widths(post)
        cases.append(case)
    return cases


PROBE_TAGS = {"A": ("Apush", "Aweigh"), "B": ("Bweigh",), "C": ("Csqueeze",)}


def probe_cases(rows: list[dict[str, Any]], probes: str) -> list[list[dict[str, Any]]]:
    """One case per draw: the scene's probe rows in order (A: push, then weigh when the push left the
    object on the table; B: weigh; C: squeeze)."""
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
    prior = bundle.get("prior") or lab_prior
    rng = np.random.default_rng(seed)
    out = []
    for case in cases:
        first = case[0]
        b = prior(first["pose"], n, rng)
        a = b.arrays()
        terms = [(pms[r["probe"]], {"kind": r["kind"], "reading": r["reading"]},
                  particle_matrix(r["base"], a, row_pose(r), r["force"], model_names(pms[r["probe"]])), float(r["force"]), r.get("thrown"))
                 for r in case]
        truth, sim = generator_truth(first, bundle.get("classes")), sim_truth(first)
        c = {"object": first["object"], "label": "+".join(r["kind"] for r in case), "n_probes": len(case),
             "info": {"used": [], "rejected": []}, "prior": _hits(b, truth), "prior_sim": _hits(b, sim), "prior_width": _widths(b),
             "post": {}, "post_sim": {}, "post_width": {}}
        squeeze = next((r for r in case if r["probe"] == "squeeze" and (r.get("theta_frag") or {}).get("f_damage_n") is not None), None)
        weigh = next((r for r in case if r["probe"] == "weigh" and (r.get("entry") or {}).get("fillable")), None)
        if squeeze is not None:
            c["squeeze_f_damage"] = {"prior": _fragility_interval(b, squeeze), "post": {}}
        if weigh is not None:
            c["weigh_fill"] = {"filled": float(weigh.get("fill") or 0.0) > 0,
                               "prior": _fill_interval(b, weigh), "post": {}}
        for j, t in enumerate(grid):
            ll = np.zeros(len(b.weights))
            for pm, obs, X, force, thrown in terms:
                lt, info = pm.loglik(obs, X, a, force, temper=t, thrown=thrown)
                ll = ll + lt
                if j == 0:
                    c["info"]["used"] += [f"{pm.probe}_{k}" for k in info["used"]]
                    c["info"]["rejected"] += [f"{pm.probe}_{k}" for k in info["rejected"]]
            post = b.reweight(ll)
            c["post"][t], c["post_sim"][t], c["post_width"][t] = _hits(post, truth), _hits(post, sim), _widths(post)
            if squeeze is not None:
                c["squeeze_f_damage"]["post"][t] = _fragility_interval(post, squeeze)
            if weigh is not None:
                c["weigh_fill"]["post"][t] = _fill_interval(post, weigh)
        out.append(c)
    return out


def _interval(b: ParticleBelief, key: str, positive: bool = False) -> tuple[float, float] | None:
    v = np.array([np.nan if p.get(key) is None else float(p[key]) for p in b.particles])
    ok = np.isfinite(v) & (v > 0 if positive else True)
    v, w = v[ok], np.asarray(b.weights)[ok]
    if not len(v) or not w.sum() > 0:
        return None
    order = np.argsort(v, kind="stable")
    cw = np.cumsum(w[order]) / w.sum()
    return tuple(float(v[order[min(int(np.searchsorted(cw, q)), len(v) - 1)]]) for q in (0.1, 0.9))


def _fragility_interval(b: ParticleBelief, row: dict[str, Any]) -> dict[str, Any]:
    interval = _interval(b, "f_damage_n", positive=True)
    truth = float(row["theta_frag"]["f_damage_n"])
    return {"hit": bool(interval is not None and interval[0] <= truth <= interval[1]),
            "width": float(np.log(interval[1] / interval[0])) if interval is not None else None}


def _fill_interval(b: ParticleBelief, row: dict[str, Any]) -> dict[str, Any]:
    filled = np.array([float(p.get("fill") or 0.0) > 0 for p in b.particles])
    truth = float(row.get("fill") or 0.0)
    interval = _interval(b, "fill", positive=True) if truth > 0 else None
    return {"p_filled": float(np.asarray(b.weights)[filled].sum() / np.sum(b.weights)),
            "hit": bool(interval is not None and interval[0] <= truth <= interval[1]) if truth > 0 else None,
            "width": interval[1] - interval[0] if interval is not None else None}


def probe_property_report(cases: list[dict[str, Any]], temper: float, constant: float | None) -> dict[str, Any]:
    """Fragility intervals and fill classification; fill intervals condition on positive fill.

    Empty truths enter the binary fill scores only. A filled truth without positive posterior
    support misses its interval. The binary constant is the training fillable weigh rate.
    """
    out = {}
    for key in ("squeeze_f_damage", "weigh_fill"):
        records = [c[key] for c in cases if key in c]
        if not records:
            continue
        block = {"n": len(records)}
        y = np.array([r["filled"] for r in records], dtype=float) if key == "weigh_fill" else None
        if y is not None:
            block.update(n_filled=int(y.sum()), constant_p_filled=constant)
        for name in ("prior", "post"):
            scores = [r[name][temper] if name == "post" else r[name] for r in records]
            if y is None:
                widths = [v["width"] for v in scores if v["width"] is not None]
                block[name] = {"coverage": float(np.mean([v["hit"] for v in scores])),
                               "log_width_median": float(np.median(widths)) if widths else None}
            else:
                p = np.array([v["p_filled"] for v in scores])
                widths = [v["width"] for v, filled in zip(scores, y) if filled and v["width"] is not None]
                brier = float(np.mean((p - y) ** 2))
                bc = float(np.mean((constant - y) ** 2)) if constant is not None else None
                ece = _ece(p, y)
                block[name] = {"filled_only_coverage": float(np.mean([v["hit"] for v, filled in zip(scores, y) if filled])) if y.any() else None,
                               "filled_only_width_median": float(np.median(widths)) if widths else None,
                               "p_filled_when_filled": float(p[y > 0].mean()) if y.any() else None,
                               "p_filled_when_empty": float(p[y == 0].mean()) if (y == 0).any() else None,
                               "brier": brier, "brier_const": bc, "ece": ece,
                               "acceptance": {"brier_below_const": bool(brier < bc) if bc is not None else None,
                                              "ece_le_005": bool(ece <= 0.05)}}
        out[key] = block
    return out


def cell_verdict(hits: list[bool], objects: list[str], rng: np.random.Generator, n_boot: int = N_BOOT) -> dict[str, Any]:
    """One coverage cell (plan 2b2 decision 9): pass when the coverage lies in COVER; fail when its 95 % object-bootstrap
    interval lies wholly outside COVER; otherwise undecided. Under MIN_CASES cases or MIN_OBJECTS objects it is
    insufficient, with no interval."""
    objs = sorted(set(objects))
    out = {"n": len(hits), "n_objects": len(objs), "coverage": float(np.mean(hits)) if hits else None, "ci": None}
    if len(hits) < MIN_CASES or len(objs) < MIN_OBJECTS:
        return {**out, "verdict": "insufficient"}
    inv = np.searchsorted(objs, objects)
    W = object_weights(len(objs), n_boot, rng)
    boot = (W @ np.bincount(inv, weights=np.asarray(hits, dtype=float), minlength=len(objs))) / (W @ np.bincount(inv, minlength=len(objs)))
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    cov = out["coverage"]
    verdict = "pass" if COVER[0] <= cov <= COVER[1] else "fail" if ci[0] > COVER[1] or ci[1] < COVER[0] else "undecided"
    return {**out, "ci": ci, "verdict": verdict}


def judged_keys(probes: str | None) -> tuple[str, ...]:
    """The property cells a check judges: those its evidence reads (one attempt: all three; the push and the weigh:
    the mass and the friction; the squeeze: none, only the damage force). The others are reported."""
    return JUDGED[probes]


def acceptance_b2(cases: list[dict[str, Any]], temper: float, probes: str | None, rng: np.random.Generator) -> dict[str, Any]:
    """Plan 2b2 decision 9's (b): per property cell the coverage, its interval and verdict ("reported" for a cell the
    probes do not read); after a squeeze, the damage force; after a weigh of a fillable object, the fill (filled
    cases) and the P(filled) ECE (calibrate's plug-in rule at P_FILLED_ECE, from MIN_ECE_CASES cases)."""
    from robots.mujoco_lab.calibrate import ece_rule

    def cell(sel: list[dict[str, Any]], hit) -> dict[str, Any]:
        return cell_verdict([bool(hit(c)) for c in sel], [c["object"] for c in sel], rng)

    out: dict[str, Any] = {"bounds": list(COVER), "min_cases": MIN_CASES, "min_objects": MIN_OBJECTS}
    for k, _ in KEYS:
        v = cell(cases, lambda c: c["post"][temper][k])
        out[k] = v if k in judged_keys(probes) else {**v, "verdict": "reported"}
    squeezed = [c for c in cases if "squeeze_f_damage" in c]
    if squeezed:
        out["f_damage"] = cell(squeezed, lambda c: c["squeeze_f_damage"]["post"][temper]["hit"])
    weighed = [c for c in cases if "weigh_fill" in c]
    if weighed:
        out["fill"] = cell([c for c in weighed if c["weigh_fill"]["filled"]], lambda c: c["weigh_fill"]["post"][temper]["hit"])
        p = np.array([c["weigh_fill"]["post"][temper]["p_filled"] for c in weighed])
        y = np.array([float(c["weigh_fill"]["filled"]) for c in weighed])
        ece = {"n": len(weighed), "ece": _ece(p, y), "bound": P_FILLED_ECE, "ci": None}
        if len(weighed) < MIN_ECE_CASES:
            out["p_filled_ece"] = {**ece, "verdict": "insufficient"}
        else:
            ci = ece_ci(p, y, [c["object"] for c in weighed], rng, N_BOOT)
            out["p_filled_ece"] = {**ece, "ci": ci, "verdict": ece_rule(ci, P_FILLED_ECE)}
    return out


def laws_of(bundle: dict[str, Any], probes: str | None) -> dict[str, dict[str, dict[str, Any]]]:
    """The reading laws the check used: term -> label -> (kind, core weight, sd1, sd2), or probe -> kind -> the same."""
    if probes:
        return {p: {k: as_law(law).summary() for k, law in pm.reading.items()} for p, pm in bundle["probe_models"].items()}
    return {t: {g: law.summary() for g, law in laws.items()} for t, laws in bundle["readout_model"].terms.items()}


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
    ap.add_argument("--probes", default=None, choices=tuple(PROBE_TAGS), help="check the update after the probes (A: push then weigh; B: weigh; C: squeeze)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    B = load_bundle(a.bundle)
    man = B["manifest"]
    metrics = json.load(open(os.path.join(a.bundle, man["metrics"])))

    if a.probes:
        if not B["probe_models"]:
            raise SystemExit("the bundle has no probe models: train_models --probe-corpus PATH --out-dir BUNDLE")
        rows = [r for r in load_rows(repo_path(man["probe_corpus"])) if "pose" in r and "com_end" in r]
        temper0 = float(next(iter(B["probe_models"].values())).temper)
    else:
        if B["readout_model"] is None:
            raise SystemExit("the bundle has no readout model: train it on corpus v2 rows (they carry readouts)")
        paths = metrics["corpus"] if isinstance(metrics["corpus"], list) else [metrics["corpus"]]
        rows = [r for p in paths for r in load_rows(repo_path(p))]
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
        "laws": laws_of(B, a.probes),
        "seconds": round(time.time() - t0, 1),
    }
    if a.probes == "A":
        out["n_with_weigh"] = sum(c["n_probes"] == 2 for c in cases)
    if a.probes:
        tags = set(PROBE_TAGS[a.probes])
        train_weigh = [r for r in rows_for("train") if r["probe"] == "weigh" and r["key"].rsplit("__", 1)[1] in tags
                       and (r.get("entry") or {}).get("fillable")]
        constant = float(np.mean([float(r.get("fill") or 0.0) > 0 for r in train_weigh])) if train_weigh else None
        out.update(probe_property_report(cases, temper, constant))
    out["acceptance_b2"] = acceptance_b2(cases, temper, a.probes, np.random.default_rng(a.seed))
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
                                          "sim_truth", "acceptance_b", "acceptance_b2", "width80_median", "terms", "seconds")}, indent=1))
    return out


if __name__ == "__main__":
    main()
    sys.exit(0)
