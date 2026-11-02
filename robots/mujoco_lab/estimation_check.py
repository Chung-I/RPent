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


"""Mass and CoM estimation check: how close the belief's posterior mean comes to the truth, and how wide and how honest
its 80 % interval is, at four stages: the prior; after the first attempt (its outcome and readouts, as
``posterior_check.check_rows``); after a weigh (probe corpus B); after a push (probe corpus A, the push alone).

Held-out objects by cross-fitting (``--crossfit``): each object's cases use the bundle of the m2b2 fold that never saw the
object (``folds/folds.json``). Without it, one bundle on ``--objects`` (the regression check against posterior_check).

Per case and property: the truth (``posterior_check.generator_truth``, the prior's own space), the posterior mean, the
10 % and 90 % quantiles and the hit. Errors: the mass relative to the truth, |m̂ − m| / m; the friction absolute; the CoM
fraction along the pivot absolute, and in cm (times the pivot length). Widths: the mass interval relative to the posterior
mean; the friction's absolute; the CoM's as a fraction and in cm. Per stage and property: the error's median and IQR, the
width's median and the coverage, each with a 95 % object-bootstrap interval (objects drawn with replacement, all of a
drawn object's cases kept). The prior row is the attempt cases' prior.

    python -P -m robots.mujoco_lab.estimation_check run --bundle robots/mujoco_lab/data/m2b2 --crossfit --workers 5 --out DIR
    python -P -m robots.mujoco_lab.estimation_check summary --dir DIR [--wandb]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable

import numpy as np

from robots.mujoco_lab import tracking
from robots.mujoco_lab.posterior_check import generator_truth
from robots.mujoco_lab.prior import lab_prior
from robots.mujoco_lab.robot_spec import load_bundle
from robots.mujoco_lab.train_models import load_rows, repo_path
from rpent.skills.pick_place.belief import ParticleBelief
from rpent.skills.pick_place.evaluate import object_weights, row_base, row_pose, rows_of
from rpent.skills.pick_place.features import model_names, particle_matrix, particle_sides

STAGES = ("prior", "attempt", "weigh", "push")
PROPS = (("mass", "mass"), ("mu", "mu"), ("com", "com_frac"))
TAGS = {"weigh": ("Bweigh",), "push": ("Apush",)}
N_BOOT = 2000


def estimates(b: ParticleBelief, truth: dict[str, float], length: float) -> dict[str, dict[str, Any]]:
    """Per property the truth, the posterior mean, the 80 % interval, its hit, the error and the width (see the module)."""
    a = b.arrays()
    out = {}
    for prop, key in PROPS:
        mean = float(np.dot(b.weights, a[key]))
        lo, hi = b.quantile(key, 0.1), b.quantile(key, 0.9)
        t = float(truth[prop])
        e = {"truth": t, "mean": mean, "lo": lo, "hi": hi, "hit": bool(lo <= t <= hi)}
        if prop == "mass":
            e["err"], e["width"] = abs(mean - t) / t, (hi - lo) / mean
        else:
            e["err"], e["width"] = abs(mean - t), hi - lo
        if prop == "com":
            e["err_cm"], e["width_cm"] = 100.0 * length * e["err"], 100.0 * length * e["width"]
        out[prop] = e
    return out


def stage_cases(rows: list[dict[str, Any]], stage: str) -> list[list[dict[str, Any]]]:
    """One case per scene draw: its probe rows of the stage's tags (weigh: B's weigh; push: A's push alone)."""
    tags = TAGS[stage]
    by: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        tag = r["key"].rsplit("__", 1)[1]
        if tag in tags:
            by[(r["object"], int(r["draw"]))][tag] = r
    return [[d[t] for t in tags if t in d] for _, d in sorted(by.items()) if tags[0] in d]


def _boot(values: np.ndarray, inv: np.ndarray, n_objects: int, rng: np.random.Generator, n_boot: int, stat: Callable) -> list[float]:
    W = object_weights(n_objects, n_boot, rng)
    reps = [stat(np.repeat(values, W[b, inv].astype(int))) for b in range(n_boot)]
    return [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]


def cell(values, objects, rng: np.random.Generator, n_boot: int = N_BOOT) -> dict[str, Any]:
    """The median, the IQR and the 95 % object-bootstrap interval of the median."""
    v = np.asarray(values, dtype=float)
    objs, inv = np.unique(np.asarray(objects), return_inverse=True)
    return {"n": len(v), "n_objects": len(objs), "median": float(np.median(v)), "q25": float(np.percentile(v, 25)),
            "q75": float(np.percentile(v, 75)), "ci": _boot(v, inv, len(objs), rng, n_boot, np.median)}


def coverage_cell(hits, objects, rng: np.random.Generator, n_boot: int = N_BOOT) -> dict[str, Any]:
    """The hit rate of the 80 % intervals and its 95 % object-bootstrap interval."""
    h = np.asarray(hits, dtype=float)
    objs, inv = np.unique(np.asarray(objects), return_inverse=True)
    W = object_weights(len(objs), n_boot, rng)
    boot = (W @ np.bincount(inv, weights=h, minlength=len(objs))) / (W @ np.bincount(inv, minlength=len(objs)))
    return {"n": len(h), "n_objects": len(objs), "coverage": float(h.mean()),
            "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def summarize(records: list[dict[str, Any]], n_boot: int = N_BOOT, seed: int = 0) -> dict[str, Any]:
    """Per stage and property: the error and width cells and the coverage. The prior row is the attempt cases' prior."""
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    for stage in STAGES:
        sel = [(r, r["prior"]) for r in records if r["stage"] == "attempt"] if stage == "prior" else \
              [(r, r["post"]) for r in records if r["stage"] == stage]
        if not sel:
            continue
        objects = [r["object"] for r, _ in sel]
        block = {}
        for prop, _ in PROPS:
            ests = [e[prop] for _, e in sel]
            p = {"n": len(ests), "coverage": coverage_cell([e["hit"] for e in ests], objects, rng, n_boot)}
            for m in ("err", "err_cm", "width", "width_cm"):
                if m in ests[0]:
                    p[m] = cell([e[m] for e in ests], objects, rng, n_boot)
            block[prop] = p
        out[stage] = block
    return out


def attempt_rows(bundle_dir: str) -> list[dict[str, Any]]:
    """The bundle's corpus rows that carry readouts (posterior_check's filter)."""
    man = json.load(open(os.path.join(bundle_dir, "manifest.json")))
    metrics = json.load(open(os.path.join(bundle_dir, man["metrics"])))
    paths = metrics["corpus"] if isinstance(metrics["corpus"], list) else [metrics["corpus"]]
    rows = [r for p in paths for r in load_rows(repo_path(p))]
    return [r for r in rows if r.get("readouts") and "pose" in r and "com_obj_true" in r and "com_end" in r]


def probe_rows(bundle_dir: str) -> list[dict[str, Any]]:
    man = json.load(open(os.path.join(bundle_dir, "manifest.json")))
    return [r for r in load_rows(repo_path(man["probe_corpus"])) if "pose" in r and "com_end" in r]


def _record(stage: str, first: dict[str, Any], b: ParticleBelief, post: ParticleBelief, classes, label: str) -> dict[str, Any]:
    truth = generator_truth(first, classes)
    length = float(first["pose"]["pivot"]["L"])
    return {"object": first["object"], "key": first.get("key"), "stage": stage, "label": label, "length": length,
            "fillable": bool((first.get("entry") or {}).get("fillable")), "filled": float(first.get("fill") or 0.0) > 0,
            "prior": estimates(b, truth, length), "post": estimates(post, truth, length), "ess": post.ess()}


def run_attempt(rows: list[dict[str, Any]], B: dict[str, Any], n: int, seed: int, prior=None) -> list[dict[str, Any]]:
    """posterior_check.check_rows at the bundle's readout temper, with the estimates of each case's prior and posterior."""
    model, readout = B["model"], B["readout_model"]
    prior = prior or B.get("prior") or lab_prior
    rng = np.random.default_rng(seed)
    out = []
    for r in rows:
        b = prior(r["pose"], n, rng)
        a = b.arrays()
        X = particle_matrix(row_base(r), a, row_pose(r), r["force"], model_names(model))
        P = model.predict_matrix(X, r.get("planned"))[0]
        obs = {"label": r["label"], "readouts": r["readouts"], "force": r["force"]}
        comp, _ = readout.components(obs, P, X, a, particle_sides(a, r["pose"]))
        out.append(_record("attempt", r, b, b.reweight(readout.combine(comp)), B.get("classes"), r["label"]))
    return out


def run_probes(cases: list[list[dict[str, Any]]], stage: str, B: dict[str, Any], n: int, seed: int, prior=None) -> list[dict[str, Any]]:
    """posterior_check.check_probe_cases at the probe models' temper, with the estimates."""
    pms = B["probe_models"]
    prior = prior or B.get("prior") or lab_prior
    rng = np.random.default_rng(seed)
    out = []
    for case in cases:
        first = case[0]
        b = prior(first["pose"], n, rng)
        a = b.arrays()
        ll = np.zeros(len(b.weights))
        for r in case:
            pm = pms[r["probe"]]
            X = particle_matrix(r["base"], a, row_pose(r), r["force"], model_names(pm))
            lt, _ = pm.loglik({"kind": r["kind"], "reading": r["reading"]}, X, a, float(r["force"]), thrown=r.get("thrown"))
            ll = ll + lt
        out.append(_record(stage, first, b, b.reweight(ll), B.get("classes"), "+".join(r["kind"] for r in case)))
    return out


def job(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """One bundle's stages on its objects (a fold's, or the ``--objects``); each stage seeded by (seed, fold, stage)."""
    B = load_bundle(spec["bundle"])
    objs = spec["objects"]
    out = []
    for i, stage in enumerate(spec["stages"]):
        rng_seed = [int(spec["seed"]), int(spec["fold"]), i]
        if stage == "attempt":
            recs = run_attempt(rows_of(attempt_rows(spec["source"]), objs), B, spec["n"], rng_seed if spec["fold"] >= 0 else spec["seed"])
        else:
            recs = run_probes(stage_cases(rows_of(probe_rows(spec["source"]), objs), stage), stage, B, spec["n"], rng_seed)
        for r in recs:
            r["fold"] = spec["fold"]
        out += recs
        print(f"fold {spec['fold']} {stage}: {len(recs)} cases", flush=True)
    return out


def jobs_for(bundle_dir: str, crossfit: bool, objects: list[str] | None, stages: list[str], n: int, seed: int) -> list[dict[str, Any]]:
    if not crossfit:
        return [{"bundle": bundle_dir, "source": bundle_dir, "fold": -1, "objects": objects, "stages": stages, "n": n, "seed": seed}]
    folds = json.load(open(os.path.join(bundle_dir, "folds", "folds.json")))
    return [{"bundle": os.path.join(bundle_dir, "folds", f"fold_{k}"), "source": bundle_dir, "fold": k,
             "objects": sorted(o for o, f in folds.items() if f == k), "stages": stages, "n": n, "seed": seed}
            for k in sorted(set(folds.values()))]


def _revision() -> dict[str, Any]:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rev = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    return {"code_revision": rev, "dirty": dirty}


def run(a: argparse.Namespace) -> str:
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "records.jsonl")
    if os.path.exists(path):
        raise SystemExit(f"{path} exists: pick a new --out")
    objects = a.objects.split(",") if a.objects else None
    specs = jobs_for(a.bundle, a.crossfit, objects, a.stages.split(","), a.n_particles, a.seed)
    with open(os.path.join(a.out, "run_config.json"), "w") as f:
        json.dump({**vars(a), **_revision(), "jobs": [{k: v for k, v in s.items() if k != "objects"} | {"n_objects": len(s["objects"] or [])}
                                                      for s in specs]}, f, indent=1)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(a.workers, len(specs))) as ex, open(path, "w") as out:
        for recs in ex.map(job, specs):
            for r in recs:
                out.write(json.dumps(r) + "\n")
    print(f"done in {time.time() - t0:.0f} s: {path}", flush=True)
    return path


def markdown(s: dict[str, Any]) -> str:
    def fmt(c: dict[str, Any], k: str = "median", d: int = 3) -> str:
        return f"{c[k]:.{d}f} [{c['ci'][0]:.{d}f}, {c['ci'][1]:.{d}f}]"

    lines = ["| stage | property | n (objects) | error median (IQR) [95 % CI] | 80 % width median [95 % CI] | coverage [95 % CI] |",
             "|---|---|---|---|---|---|"]
    for stage, block in s.items():
        for prop, p in block.items():
            e, w, c = p["err"], p["width"], p["coverage"]
            err = f"{fmt(e)} ({e['q25']:.3f}–{e['q75']:.3f})"
            wid = fmt(w)
            if prop == "com":
                ec, wc = p["err_cm"], p["width_cm"]
                err += f"; {fmt(ec, d=2)} cm ({ec['q25']:.2f}–{ec['q75']:.2f})"
                wid += f"; {fmt(wc, d=2)} cm"
            lines.append(f"| {stage} | {prop} | {c['n']} ({c['n_objects']}) | {err} | {wid} | {fmt(c, 'coverage')} |")
    return "\n".join(lines) + "\n"


def summary(a: argparse.Namespace) -> dict[str, Any]:
    recs = [json.loads(line) for line in open(os.path.join(a.dir, "records.jsonl"))]
    s = summarize(recs, a.n_boot, a.seed)
    out = {"n_records": len(recs), "n_boot": a.n_boot, "stages": s}
    with open(os.path.join(a.dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    md = ("# Mass and CoM estimation check\n\nErrors: mass |m̂ − m| / m, friction |μ̂ − μ|, CoM along the pivot |ĉ − c| "
          "(fraction of the length; cm). Widths: the 80 % interval (mass relative to the posterior mean). 95 % object-bootstrap "
          f"intervals ({a.n_boot} draws). The prior row is the attempt cases' prior.\n\n" + markdown(s))
    with open(os.path.join(a.dir, "summary.md"), "w") as f:
        f.write(md)
    print(md, flush=True)
    if a.wandb:
        cfg = json.load(open(os.path.join(a.dir, "run_config.json")))
        flat = {f"{st}/{p}/{m}": (c["coverage"] if m == "coverage" else c["median"])
                for st, blk in s.items() for p, pb in blk.items() for m, c in pb.items() if isinstance(c, dict)}
        out["wandb"] = tracking.log_run("estimation_check", cfg, flat)
    return out


def main(argv: list[str] | None = None) -> Any:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bundle", required=True)
    r.add_argument("--crossfit", action="store_true", help="each fold's objects through the fold's bundle")
    r.add_argument("--objects", default=None, help="comma-separated ids (without --crossfit)")
    r.add_argument("--stages", default="attempt,weigh,push")
    r.add_argument("--n-particles", type=int, default=2000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--workers", type=int, default=5)
    r.add_argument("--out", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--dir", required=True)
    s.add_argument("--n-boot", type=int, default=N_BOOT)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--wandb", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "run" and not a.crossfit and not a.objects:
        raise SystemExit("--objects is needed without --crossfit")
    return run(a) if a.cmd == "run" else summary(a)


if __name__ == "__main__":
    main()
    sys.exit(0)
