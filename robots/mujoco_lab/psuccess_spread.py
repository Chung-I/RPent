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


"""P(success) spread check: does a weigh narrow the spread of P(success) across the belief, and which property drives
what is left? Analysis only: no probe or attempt runs.

Per case and row, each particle's P(placed_stable) and P(irreversible) come from the menu's own model, with its
calibration and its damage-flag masks (``SuccessModel.predict_matrix``, as ``menu.estimate``). The spread across the
belief is the weighted SD and the 10-90 % range of those chances (``ParticleBelief.quantile``'s quantiles).

Weigh cases: estimation_check's (probe corpus B's weigh, one case per scene draw, cross-fit over m2b2's folds), with the
exact particles of its run (``prior(pose, n, rng)`` on the stream [seed, fold, 1], the cases in its order) and its
posterior (the weigh reading's likelihood, no refresh). The lab is reset once per case to the recorded scene B (the probe
corpus's reset: its seed, no fixture; a settle only), and the reset's pose is checked against the recorded one. Rows:
commit_std, and per setting (care, abstain cost) the menu's best attempt row at that stage: the row it commits to when it
commits (``build_menu`` on the stage's belief without the probe rows and the guard, which do not move the attempt rows).
Destination table_centre: scene B has no fixture, and the probe rows' base reads the table centre.

Attempt cases: estimation_check's (the corpus rows with readouts), its particles (stream [seed, fold, 0]) and its
posterior after the attempt's outcome and readouts. Only the attempted row itself is scored: commit_std on those scenes
would need a reset per corpus scene.

Driver split: the same particles with the mass, the CoM or both at the generator truth (``posterior_check.generator_truth``),
and with every model input at truth (mass, CoM, friction, fill, spill angle and fragility: the row's hidden draws).

Tags (commit_std's P(placed_stable), fixed before the results): weigh resolves (the weigh's SD at most half the prior's),
CoM-limited (the SD stays above half, and the CoM at truth leaves less than half of it), unavoidable (P at every input's
truth below 0.2), bad-luck zone (P at every input's truth in [0.3, 0.7]; that point has no spread, so the small-spread
condition holds by construction; ``bad_luck_mc_small`` adds an SD of at most 0.05 with only the mass and CoM at truth).

    python -P -m robots.mujoco_lab.psuccess_spread run --bundle robots/mujoco_lab/data/m2b2_v11c --scenes S --workers 2 --out DIR
    python -P -m robots.mujoco_lab.psuccess_spread summary --dir DIR [--check class=PARTA_DIR] [--wandb]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from robots.mujoco_lab.posterior_check import generator_truth
from rpent.skills.pick_place.evaluate import object_weights
from rpent.skills.pick_place.features import model_names, particle_matrix
from rpent.skills.pick_place.protocol import IRREVERSIBLE, OUTCOMES

SETTINGS = (("k100_A3", 100.0, 3.0), ("k30_A3", 30.0, 3.0), ("k100_A10", 100.0, 10.0))   # the ladder's (care, abstain cost)
DESTINATION = "table_centre"
PRIORS = ("class", "generic")
TRUTH_SETS = {"mass_true": ("mass",), "com_true": ("com_obj",), "both_true": ("mass", "com_obj"),
              "all_true": ("mass", "com_obj", "mu", "fill", "spill_angle_deg", "f_damage_n", "v_bruise_ms", "v_break_ms",
                           "k_n_per_mm")}
VARIANTS = ("belief",) + tuple(TRUTH_SETS)
FRAG = ("k_n_per_mm", "f_damage_n", "v_bruise_ms", "v_break_ms")
HALF = 0.5                 # weigh resolves: the SD falls by at least half; CoM-limited: the CoM at truth removes more than half
SMALL_SD = 0.05            # a small spread (bad_luck_mc_small)
LOW_P = 0.2                # unavoidable: P(placed_stable) at every input's truth below this
BAD_LUCK = (0.3, 0.7)      # bad-luck zone: P at every input's truth in this range
MIN_PRIOR_SD = 0.02        # the extra counts: cases whose prior SD is at least this
N_BOOT = 2000
STAGE_INDEX = {"attempt": 0, "weigh": 1}   # estimation_check's stage order (attempt, weigh, push): the stream's third key


def spread(p: np.ndarray, w: np.ndarray) -> dict[str, float]:
    """The weighted mean, SD, 10 % and 90 % quantiles (ParticleBelief.quantile's rule) and the 10-90 % range."""
    p, w = np.asarray(p, dtype=float), np.asarray(w, dtype=float)
    w = w / w.sum()
    mean = float(w @ p)
    sd = float(np.sqrt(max(0.0, float(w @ (p - mean) ** 2))))
    order = np.argsort(p, kind="stable")
    cw = np.cumsum(w[order])

    def q(x: float) -> float:
        return float(p[order][min(int(np.searchsorted(cw, x - 1e-12, side="left")), len(p) - 1)])

    lo, hi = q(0.1), q(0.9)
    return {"mean": mean, "sd": sd, "q10": lo, "q90": hi, "range": hi - lo}


def with_truth(arrays: dict[str, np.ndarray], truth: dict[str, Any], keys) -> dict[str, np.ndarray]:
    """A copy of the particle arrays with ``keys`` at the truth (None reads NaN, as ParticleBelief.arrays); the input
    and its other arrays are kept."""
    out = dict(arrays)
    n = len(arrays["mass"])
    for k in keys:
        if k == "com_obj":
            out[k] = np.tile(np.asarray(truth[k], dtype=float).reshape(1, 3), (n, 1))
        else:
            out[k] = np.full(n, np.nan if truth.get(k) is None else float(truth[k]))
    return out


def truth_point(row: dict[str, Any], classes: dict[str, Any] | None) -> dict[str, Any]:
    """Every model input at the row's truth: the mass, friction and CoM of generator_truth (the prior's space; a filled
    row's joint mass and CoM), the fill, the spill angle and the fragility draw (theta_frag)."""
    g = generator_truth(row, classes)
    frag = row.get("theta_frag") or {}
    return {"mass": float(g["mass"]), "mu": float(g["mu"]), "com": float(g["com"]), "com_obj": [float(x) for x in g["com_obj"]],
            "fill": float(row.get("fill") or 0.0), "spill_angle_deg": row.get("spill_angle_deg"),
            **{k: frag.get(k) for k in FRAG}}


def row_chances(model: Any, arrays: dict[str, np.ndarray], base: dict[str, float], pose: dict[str, Any], force: float,
                planned: list[dict[str, Any]] | None) -> tuple[np.ndarray, np.ndarray]:
    """Per particle P(placed_stable) and P(irreversible) of one row through the menu's model (menu.estimate's matrix)."""
    X = particle_matrix(base, arrays, pose, force, model_names(model))
    hazard = getattr(model, "kind", "classifier") == "hazard"
    P = (model.predict_matrix(X, planned) if hazard else model.predict_matrix(X))[0]
    return P[:, OUTCOMES.index("placed_stable")].copy(), P[:, [OUTCOMES.index(k) for k in IRREVERSIBLE]].sum(axis=1)


def tags(r: dict[str, float]) -> dict[str, bool]:
    """The fixed exploratory tags of one case from its commit_std P(placed_stable) spreads (see the module)."""
    resolves = r["sd_prior"] > 0 and r["sd_weigh"] <= HALF * r["sd_prior"]
    bad = BAD_LUCK[0] <= r["p_all_true"] <= BAD_LUCK[1]
    return {"weigh_resolves": bool(resolves),
            "com_limited": bool(not resolves and r["sd_weigh"] > 0 and r["sd_com_true"] < HALF * r["sd_weigh"]),
            "unavoidable": bool(r["p_all_true"] < LOW_P), "bad_luck": bool(bad),
            "bad_luck_mc_small": bool(bad and r["sd_both_true"] <= SMALL_SD), "prior_sd_ok": bool(r["sd_prior"] >= MIN_PRIOR_SD)}


def paired(before, after, objects, n_boot: int = N_BOOT, seed: int = 0) -> dict[str, Any]:
    """The mean before and after, the mean change (after - before) and its 95 % object-bootstrap interval (objects drawn
    with replacement, all of a drawn object's cases kept)."""
    a, b = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    d = b - a
    objs, inv = np.unique(np.asarray(objects), return_inverse=True)
    W = object_weights(len(objs), n_boot, np.random.default_rng(seed))
    boot = (W @ np.bincount(inv, weights=d, minlength=len(objs))) / (W @ np.bincount(inv, minlength=len(objs)))
    return {"n": len(d), "n_objects": len(objs), "before": float(a.mean()), "after": float(b.mean()), "change": float(d.mean()),
            "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def _maxdiff(a: Any, b: Any) -> float:
    """The largest absolute difference over the numeric leaves of two nested structures (inf where they differ in shape)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return float("inf")
        return max([_maxdiff(a[k], b[k]) for k in a] or [0.0])
    if isinstance(a, (list, tuple, np.ndarray)) and isinstance(b, (list, tuple, np.ndarray)):
        if len(a) != len(b):
            return float("inf")
        return max([_maxdiff(x, y) for x, y in zip(a, b)] or [0.0])
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)) or a is None or b is None or isinstance(a, str):
        return 0.0 if a == b else float("inf")
    return abs(float(a) - float(b))


def _variant_chances(model, arrays, truth, base, pose, force, planned) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {"belief": row_chances(model, arrays, base, pose, force, planned)}
    for name, keys in TRUTH_SETS.items():
        out[name] = row_chances(model, with_truth(arrays, truth, keys), base, pose, force, planned)
    return out


def _stats(ch: dict[str, tuple[np.ndarray, np.ndarray]], weights: dict[str, np.ndarray]) -> dict[str, Any]:
    return {q: {v: {st: spread(c[i], w) for st, w in weights.items()} for v, c in ch.items()} for i, q in enumerate(("placed", "irr"))}


def _means(b: Any) -> dict[str, float]:
    a, w = b.arrays(), b.weights
    return {"mass": float(w @ a["mass"]), "mu": float(w @ a["mu"]), "com": float(w @ a["com_frac"])}


def _priors(spec: dict[str, Any], B: dict[str, Any]) -> dict[str, Any]:
    """Each prior of the job as estimation_check.job_prior gives it (the class prior: the bundle's)."""
    from robots.mujoco_lab.estimation_check import job_prior
    from robots.mujoco_lab.prior import lab_prior

    out = {}
    for name in spec["priors"]:
        fn, _ = job_prior({**spec, "prior": name}, B)
        out[name] = fn or B.get("prior") or lab_prior
    return out


def _memo(fn):
    cache: dict[tuple, Any] = {}

    def call(prop, v_lift, v_transport, destination, release_height=None, **kw):
        key = (float(prop.s), tuple(map(float, prop.point)), float(prop.yaw), v_lift, v_transport, destination, release_height,
               tuple(sorted(kw.items())))
        if key not in cache:
            cache[key] = fn(prop, v_lift, v_transport, destination, release_height, **kw)
        return cache[key]

    return call


class _Memo:
    """The menu's model with predict_matrix cached by the feature matrix's bytes. A case's menus at both stages and
    every setting rebuild the same particle matrices (the reweight moves only the weights), and the belief variant
    repeats the menu's own. Exact for a classifier, which reads no plan; a hazard model's calls pass through."""

    def __init__(self, model: Any):
        self._model, self._cache = model, {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._model, name)

    def predict_matrix(self, X: np.ndarray, planned: list[dict[str, Any]] | None = None) -> tuple[np.ndarray, Any, Any]:
        if getattr(self._model, "kind", "classifier") == "hazard":
            return self._model.predict_matrix(X, planned)
        X = np.ascontiguousarray(X, dtype=float)
        key = (X.shape, hashlib.blake2b(X.tobytes(), digest_size=16).digest())
        if key not in self._cache:
            self._cache[key] = self._model.predict_matrix(X)
        P, DH, LT = self._cache[key]
        return P.copy(), DH, LT


def weigh_job(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """One fold's weigh cases under each prior: the exact estimation_check beliefs, one reset per case to scene B, the
    menus at each stage and setting, and the rows' spreads per variant (see the module)."""
    from robots.mujoco_lab import calibrate as cal
    from robots.mujoco_lab.estimation_check import probe_rows, stage_cases
    from robots.mujoco_lab.prior import fragility_world
    from robots.mujoco_lab.robot_spec import load_bundle
    from rpent.skills.pick_place.evaluate import row_pose, rows_of
    from rpent.skills.pick_place.menu import build_menu, mid_proposal, pinch_limit, usable_proposals
    from rpent.skills.pick_place.tools import standard_force

    B = load_bundle(spec["bundle"])
    priors = _priors(spec, B)
    pms, classes = B["probe_models"], B.get("classes")
    rngs = {p: np.random.default_rng([int(spec["seed"]), int(spec["fold"]), STAGE_INDEX["weigh"]]) for p in priors}
    cases = stage_cases(rows_of(probe_rows(spec["source"]), spec["objects"]), "weigh")
    prims = cal._primitives(spec["bundle"], None, None)
    env = prims.env
    env.force_noise = cal.job_force_noise(None, prims.manifest)
    world = fragility_world(prims.manifest)
    keep = set(spec.get("only") or spec["objects"])
    out = []
    for case in cases:
        first = case[0]
        drawn = {p: fn(first["pose"], spec["n"], rngs[p]) for p, fn in priors.items()}   # every case draws: the streams stay
        if first["object"] not in keep:                                                    # estimation_check's
            continue
        beliefs = {}
        for p, b in drawn.items():
            a = b.arrays()
            ll = np.zeros(len(b.weights))
            for r in case:
                pm = pms[r["probe"]]
                X = particle_matrix(r["base"], a, row_pose(r), r["force"], model_names(pm))
                lt, _ = pm.loglik({"kind": r["kind"], "reading": r["reading"]}, X, a, float(r["force"]), thrown=r.get("thrown"))
                ll = ll + lt
            beliefs[p] = (b, b.reweight(ll))
        t0 = time.time()
        o = first["object"]
        if world is not None:
            env.slosh_zeta, env.slosh_scale = world[1].slosh_zeta, world[1].slosh_scale
        env.classes = None if world is None else world[0]
        env.reset(o, first["theta"], seed=int(first["seed"]) * 1000 + int(first["draw"]), com_end=first["com_end"], fixture=None,
                  fill=first["fill"], frag=first["theta_frag"])
        loc = env.locate("object")
        geometry, proposals, lim = env.geometry(), env.proposals(), pinch_limit(env)
        usable = usable_proposals(proposals, lim)
        mid = mid_proposal(usable) if usable else None
        checks = {"usable": bool(usable), "pose_maxdiff": _maxdiff(env.proposal_pose(mid), first["pose"]) if mid else None,
                  "s_equal": bool(mid is not None and abs(float(mid.s) - float(first["s"])) < 1e-12)}
        f_std = standard_force(env.nominal_mass(loc["object_id"]))
        checks["f_std_equal"] = abs(f_std - float(first["f_std"])) < 1e-9
        plan, model = _memo(env.plan_trajectory), _Memo(prims.model)
        truth = truth_point(first, classes)
        L = float(first["pose"]["pivot"]["L"])
        for p, (b, post) in beliefs.items():
            weights = {"prior": b.weights, "weigh": post.weights}
            arrays = b.arrays()
            cache: dict[tuple, Any] = {}

            def chances(opt):
                k = (opt.id, float(opt.action.get("s", -1.0)), float(opt.action["force"]))
                if k not in cache:
                    cache[k] = _variant_chances(model, arrays, truth, opt._base, opt._pose, opt.action["force"], opt._planned)
                return cache[k]

            rec = {"stage": "weigh", "prior": p, "object": o, "key": first["key"], "draw": int(first["draw"]), "fold": spec["fold"],
                   "class": (first.get("entry") or {}).get("class"), "fillable": bool((first.get("entry") or {}).get("fillable")),
                   "filled": float(first.get("fill") or 0.0) > 0, "length": L, "truth": truth, "ess": float(post.ess()),
                   "means": {"prior": _means(b), "weigh": _means(post)}, "checks": dict(checks), "best": {}}
            menu_diff = 0.0
            for stage, belief in (("prior", b), ("weigh", post)):
                for name, care, A in SETTINGS:
                    options, _ = build_menu(model, belief, geometry, proposals, f_std, care, DESTINATION, plan,
                                            np.random.default_rng(0), pose_fn=env.proposal_pose, max_open=lim,
                                            threshold=getattr(prims, "threshold", None), probe_models=None, guard=None,
                                            abstain_cost=A, evoi=getattr(prims, "evoi", "plugin"),
                                            probe_time_scale=getattr(prims, "probe_time_scale", 1.0))
                    same = [x for x in options if x.kind in ("commit", "lower_risk") and x.action["destination"] == DESTINATION]
                    if not same:
                        rec["best"].setdefault(name, {})[stage] = None
                        continue
                    best, std = same[0], next(x for x in options if x.id == "commit_std")
                    for x in (best, std):
                        menu_diff = max(menu_diff, abs(spread(chances(x)["belief"][0], belief.weights)["mean"] - x.pred.p["placed_stable"]))
                    ch = chances(best)
                    rec["best"].setdefault(name, {})[stage] = {
                        "row": best.id, "s": float(best.action.get("s", -1.0)), "expected_cost": float(best.expected_cost),
                        **{q: {v: spread(c[i], belief.weights) for v, c in ch.items()} for i, q in enumerate(("placed", "irr"))}}
                    if "commit_std" not in rec:
                        rec["commit_std"] = _stats(chances(std), weights)
            rec["checks"]["menu_maxdiff"] = menu_diff
            out.append(rec)
        print(f"fold {spec['fold']} weigh {first['key']}: {time.time() - t0:.1f} s", flush=True)
    return out


def attempt_job(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """One fold's attempt cases under each prior: estimation_check.run_attempt's beliefs, and the attempted row's spreads
    per variant at the prior and after the attempt."""
    from robots.mujoco_lab.estimation_check import attempt_rows
    from robots.mujoco_lab.robot_spec import load_bundle
    from rpent.skills.pick_place.evaluate import row_base, row_pose, rows_of
    from rpent.skills.pick_place.features import particle_sides

    B = load_bundle(spec["bundle"])
    priors = _priors(spec, B)
    model, readout, classes = B["model"], B["readout_model"], B.get("classes")
    rngs = {p: np.random.default_rng([int(spec["seed"]), int(spec["fold"]), STAGE_INDEX["attempt"]]) for p in priors}
    keep = set(spec.get("only") or spec["objects"])
    out = []
    for r in rows_of(attempt_rows(spec["source"]), spec["objects"]):
        for p, fn in priors.items():
            b = fn(r["pose"], spec["n"], rngs[p])          # every row draws: the streams stay estimation_check's
            if r["object"] not in keep:
                continue
            a, m = b.arrays(), _Memo(model)
            X = particle_matrix(row_base(r), a, row_pose(r), r["force"], model_names(model))
            P = m.predict_matrix(X, r.get("planned"))[0]
            obs = {"label": r["label"], "readouts": r["readouts"], "force": r["force"]}
            comp, _ = readout.components(obs, P, X, a, particle_sides(a, r["pose"]))
            post = b.reweight(readout.combine(comp))
            truth = truth_point(r, classes)
            ch = _variant_chances(m, a, truth, row_base(r), row_pose(r), r["force"], r.get("planned"))
            out.append({"stage": "attempt", "prior": p, "object": r["object"], "key": r.get("key"), "fold": spec["fold"],
                        "label": r["label"], "class": (r.get("entry") or {}).get("class"), "truth": truth, "ess": float(post.ess()),
                        "means": {"prior": _means(b), "attempt": _means(post)},
                        "checks": {"model_maxdiff": float(np.max(np.abs(ch["belief"][0] - P[:, OUTCOMES.index("placed_stable")])))},
                        "row": _stats(ch, {"prior": b.weights, "attempt": post.weights})})
    print(f"fold {spec['fold']} attempt: {len(out)} records", flush=True)
    return out


def _run_job(spec: dict[str, Any]) -> str:
    path = os.path.join(spec["out"], f"records_{spec['stage']}_fold{spec['fold']}.jsonl")
    recs = (weigh_job if spec["stage"] == "weigh" else attempt_job)(spec)
    with open(path + ".tmp", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    os.replace(path + ".tmp", path)
    return path


def _revision() -> dict[str, Any]:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rev = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "--no-optional-locks", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True).stdout.strip())
    from robots.mujoco_lab.corpus import lab_generator

    return {"code_revision": rev, "dirty": dirty, "generator": lab_generator()}


def run(a: argparse.Namespace) -> str:
    from robots.mujoco_lab.estimation_check import tuning_objects

    os.makedirs(a.out, exist_ok=True)
    folds = json.load(open(os.path.join(a.bundle, "folds", "folds.json")))
    only = a.objects.split(",") if a.objects else None
    stages = a.stages.split(",")
    specs = []
    for stage in stages:
        for k in sorted(set(folds.values())):
            objs = sorted(o for o, f in folds.items() if f == k)
            if only and not set(only) & set(objs):
                continue
            specs.append({"stage": stage, "bundle": os.path.join(a.bundle, "folds", f"fold_{k}"), "source": a.bundle, "fold": k,
                          "objects": objs, "only": only, "tuning": tuning_objects(folds, k), "scenes": a.scenes, "vlm": None,
                          "priors": a.priors.split(","), "n": a.n, "seed": a.seed, "out": a.out})
    todo = [s for s in specs if not os.path.exists(os.path.join(a.out, f"records_{s['stage']}_fold{s['fold']}.jsonl"))]
    cfg = {"cmd": "run", **{k: v for k, v in vars(a).items() if k != "func"}, "settings": SETTINGS, "destination": DESTINATION,
           "thresholds": {"half": HALF, "small_sd": SMALL_SD, "low_p": LOW_P, "bad_luck": BAD_LUCK, "min_prior_sd": MIN_PRIOR_SD},
           **_revision(), "jobs": [{k: s[k] for k in ("stage", "fold", "bundle")} for s in specs], "todo": len(todo)}
    with open(os.path.join(a.out, f"run_config_{time.strftime('%Y%m%dT%H%M%S')}.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    if a.workers <= 1:
        for s in todo:
            print(_run_job(s), flush=True)
    else:
        import multiprocessing as mp

        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn"), max_tasks_per_child=1) as ex:
            for path in ex.map(_run_job, todo):
                print(path, flush=True)
    return a.out


# ---------------------------------------------------------------- the summary

def load_records(d: str) -> list[dict[str, Any]]:
    out = []
    for name in sorted(os.listdir(d)):
        if name.startswith("records_") and name.endswith((".jsonl", ".jsonl.gz")):
            with (gzip.open if name.endswith(".gz") else open)(os.path.join(d, name), "rt") as f:
                out += [json.loads(line) for line in f if line.strip()]
    return out


def _cell_rows(recs, prior, row, q, stage_after):
    """(before, after, objects, before-variants at after) for one prior, row and quantity: commit_std reads one row's
    two weightings; a best row reads each stage's own best row."""
    out = []
    for r in recs:
        if r["prior"] != prior:
            continue
        if row == "commit_std" or row == "attempted":
            src = r["commit_std" if row == "commit_std" else "row"][q]
            out.append((src["belief"]["prior"], src["belief"][stage_after], {v: src[v][stage_after] for v in src}, r["object"]))
        else:
            bb = r["best"].get(row[len("best_"):]) or {}
            if not bb.get("prior") or not bb.get("weigh"):
                continue
            out.append((bb["prior"][q]["belief"], bb["weigh"][q]["belief"], {v: bb["weigh"][q][v] for v in bb["weigh"][q]}, r["object"]))
    return out


def summarize(recs: list[dict[str, Any]], n_boot: int = N_BOOT, seed: int = 0) -> dict[str, Any]:
    """Per stage, prior, row and quantity: the spread before and after (mean P, SD, 10-90 % range) with the paired
    changes; the driver split after; the tags of commit_std's P(placed_stable); the per-object table; the checks."""
    out: dict[str, Any] = {"n_boot": n_boot, "stages": {}}
    for stage, rows in (("weigh", ["commit_std"] + [f"best_{n}" for n, _, _ in SETTINGS]), ("attempt", ["attempted"])):
        sr = [r for r in recs if r["stage"] == stage]
        if not sr:
            continue
        S: dict[str, Any] = {"n": len(sr), "priors": {}}
        for prior in sorted({r["prior"] for r in sr}):
            pr = [r for r in sr if r["prior"] == prior]
            ess = np.array([r["ess"] for r in pr])
            P: dict[str, Any] = {"n": len(pr), "n_objects": len({r["object"] for r in pr}),
                                 "ess": {"quantiles": dict(zip(("0", "10", "25", "50", "75", "90", "100"),
                                                               np.quantile(ess, [0, .1, .25, .5, .75, .9, 1]).round(1).tolist())),
                                         "below_100": int((ess < 100).sum())}, "rows": {}}
            ck = [r["checks"] for r in pr]
            P["checks"] = ({"pose_maxdiff": max(c["pose_maxdiff"] for c in ck), "s_unequal": sum(not c["s_equal"] for c in ck),
                            "f_std_unequal": sum(not c["f_std_equal"] for c in ck), "menu_maxdiff": max(c["menu_maxdiff"] for c in ck)}
                           if stage == "weigh" else {"model_maxdiff": max(c["model_maxdiff"] for c in ck)})
            for row in rows:
                R: dict[str, Any] = {}
                for q in ("placed", "irr"):
                    cells = _cell_rows(pr, prior, row, q, stage)
                    if not cells:
                        continue
                    objs = [c[3] for c in cells]
                    k = {m: paired([c[0][m] for c in cells], [c[1][m] for c in cells], objs, n_boot, seed) for m in ("mean", "sd", "range")}
                    sd_after = [c[2]["belief"]["sd"] for c in cells]
                    split = {v: paired(sd_after, [c[2][v]["sd"] for c in cells], objs, n_boot, seed) for v in TRUTH_SETS if v != "all_true"}
                    for v in split:
                        split[v]["share_removed"] = (1.0 - split[v]["after"] / split[v]["before"]) if split[v]["before"] > 0 else None
                    R[q] = {"spread": k, "split_after": split, "p_all_true": float(np.mean([c[2]["all_true"]["mean"] for c in cells]))}
                P["rows"][row] = R
            if stage == "weigh":
                T = []
                for r in pr:
                    c = r["commit_std"]["placed"]
                    T.append({"object": r["object"], "class": r.get("class"), **tags(
                        {"sd_prior": c["belief"]["prior"]["sd"], "sd_weigh": c["belief"]["weigh"]["sd"], "sd_com_true": c["com_true"]["weigh"]["sd"],
                         "sd_both_true": c["both_true"]["weigh"]["sd"], "p_all_true": c["all_true"]["weigh"]["mean"]})})
                names = [k for k in T[0] if k not in ("object", "class", "prior_sd_ok")]
                P["tags"] = {"all": {"n": len(T), **{k: sum(t[k] for t in T) for k in names}},
                             "prior_sd_ok": {"n": sum(t["prior_sd_ok"] for t in T), **{k: sum(t[k] and t["prior_sd_ok"] for t in T) for k in names}}}
                per = defaultdict(list)
                for r, t in zip(pr, T):
                    per[r["object"]].append((r, t))
                P["objects"] = {}
                for o, lst in sorted(per.items()):
                    cs = [r["commit_std"]["placed"] for r, _ in lst]
                    P["objects"][o] = {"n": len(lst), "class": lst[0][0].get("class"),
                                       "sd_prior": float(np.mean([c["belief"]["prior"]["sd"] for c in cs])),
                                       "sd_weigh": float(np.mean([c["belief"]["weigh"]["sd"] for c in cs])),
                                       "sd_mass_true": float(np.mean([c["mass_true"]["weigh"]["sd"] for c in cs])),
                                       "sd_com_true": float(np.mean([c["com_true"]["weigh"]["sd"] for c in cs])),
                                       "sd_both_true": float(np.mean([c["both_true"]["weigh"]["sd"] for c in cs])),
                                       "p_weigh": float(np.mean([c["belief"]["weigh"]["mean"] for c in cs])),
                                       "p_all_true": float(np.mean([c["all_true"]["weigh"]["mean"] for c in cs])),
                                       **{k: sum(t[k] for _, t in lst) for k in names}}
            S["priors"][prior] = P
        out["stages"][stage] = S
    return out


def replication(recs: list[dict[str, Any]], checks: dict[str, str]) -> dict[str, Any]:
    """The rebuilt beliefs against estimation_check's records (its posterior means of mass, friction and CoM fraction,
    per prior): the largest absolute difference, the cases compared and the label mismatches. The corpus and fixtures
    files reuse keys for different rollouts, so the records of one stage and key pair in file order."""
    out = {}
    for prior, d in checks.items():
        path = os.path.join(d, "records.jsonl.gz") if os.path.exists(os.path.join(d, "records.jsonl.gz")) else os.path.join(d, "records.jsonl")
        ref: dict[tuple[str, str], list[dict[str, Any]]] = {}
        with (gzip.open if path.endswith(".gz") else open)(path, "rt") as f:
            for r in map(json.loads, f):
                if r.get("stage") in ("attempt", "weigh"):
                    ref.setdefault((r["stage"], r["key"]), []).append(r)
        mine = [r for r in recs if r["prior"] == prior]
        diffs, missing, mismatch, used = [], 0, 0, {}
        for r in mine:
            k = (r["stage"], r["key"])
            i = used.get(k, 0)
            if i >= len(ref.get(k, ())):
                missing += 1
                continue
            used[k] = i + 1
            e = ref[k][i]
            mismatch += int("label" in r and r["label"] != e.get("label"))
            m = r["means"][r["stage"]]
            diffs.append(max(abs(m["mass"] - e["post"]["mass"]["mean"]), abs(m["mu"] - e["post"]["mu"]["mean"]),
                             abs(m["com"] - e["post"]["com"]["mean"])))
        out[prior] = {"compared": len(diffs), "missing": missing, "label_mismatch": mismatch,
                      "maxdiff": max(diffs) if diffs else None}
    return out


def _iv(c: dict[str, Any], d: int = 3) -> str:
    return f"{c['change']:+.{d}f} [{c['ci'][0]:+.{d}f}, {c['ci'][1]:+.{d}f}]"


def markdown(s: dict[str, Any], rep: dict[str, Any]) -> str:
    L = ["# P(success) spread check", "",
         "Per case the weighted spread across the belief of each particle's chance (the menu's model, its calibration and "
         f"masks). Changes are after - before, mean over cases, with 95 % object-bootstrap intervals ({s['n_boot']} draws).", ""]
    for stage, S in s["stages"].items():
        after = "weigh" if stage == "weigh" else "attempt"
        for prior, P in S["priors"].items():
            L += [f"## {stage} cases, {prior} prior ({P['n']} cases, {P['n_objects']} objects)", "",
                  f"ESS after: quantiles {P['ess']['quantiles']}, below 100: {P['ess']['below_100']}. Checks: {P['checks']}.", "",
                  f"| row | chance | mean P before | after | change | SD before | after | change | 10-90 % range before | after | change |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for row, R in P["rows"].items():
                for q, Q in R.items():
                    k = Q["spread"]
                    L.append(f"| {row} | {q} | {k['mean']['before']:.3f} | {k['mean']['after']:.3f} | {_iv(k['mean'])} | "
                             f"{k['sd']['before']:.3f} | {k['sd']['after']:.3f} | {_iv(k['sd'])} | {k['range']['before']:.3f} | "
                             f"{k['range']['after']:.3f} | {_iv(k['range'])} |")
            L += ["", f"Driver split after the {after}: the mean SD with the belief's particles, and with the mass, the CoM or "
                  "both at truth (the rest: friction, fill, fragility). Change = variant - belief.", "",
                  "| row | chance | SD | mass at truth | change | share removed | CoM at truth | change | share removed | both | change | "
                  "share removed | mean P, every input at truth |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for row, R in P["rows"].items():
                for q, Q in R.items():
                    sp = Q["split_after"]
                    cells = " | ".join(f"{sp[v]['after']:.3f} | {_iv(sp[v])} | {sp[v]['share_removed']:.2f}" if sp[v]["share_removed"] is not None
                                       else f"{sp[v]['after']:.3f} | {_iv(sp[v])} | -" for v in ("mass_true", "com_true", "both_true"))
                    L.append(f"| {row} | {q} | {sp['mass_true']['before']:.3f} | {cells} | {Q['p_all_true']:.3f} |")
            if "tags" in P:
                t = P["tags"]
                L += ["", "Tags (commit_std, P(placed_stable); exploratory leads, not claims): "
                      + "; ".join(f"{k} {t['all'][k]} of {t['all']['n']} ({t['prior_sd_ok'][k]} of {t['prior_sd_ok']['n']} with a prior SD "
                                  f"of at least {MIN_PRIOR_SD})" for k in t["all"] if k != "n") + ".", "",
                      "| object | class | cases | SD prior | SD weigh | mass at truth | CoM at truth | both | P weigh | P every input at truth | "
                      "resolves | CoM-limited | unavoidable | bad luck |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
                for o, v in sorted(P["objects"].items(), key=lambda kv: -kv[1]["sd_weigh"]):
                    L.append(f"| {o} | {v['class']} | {v['n']} | {v['sd_prior']:.3f} | {v['sd_weigh']:.3f} | {v['sd_mass_true']:.3f} | "
                             f"{v['sd_com_true']:.3f} | {v['sd_both_true']:.3f} | {v['p_weigh']:.3f} | {v['p_all_true']:.3f} | "
                             f"{v['weigh_resolves']} | {v['com_limited']} | {v['unavoidable']} | {v['bad_luck']} |")
            L.append("")
    if rep:
        L += ["Replication of estimation_check's posterior means (mass, friction, CoM fraction): "
              + "; ".join(f"{p}: {v['compared']} compared, {v['missing']} missing, {v['label_mismatch']} label mismatches, max diff {v['maxdiff']}" for p, v in rep.items()), ""]
    return "\n".join(L)


def summary(a: argparse.Namespace) -> dict[str, Any]:
    recs = load_records(a.dir)
    s = summarize(recs, a.n_boot, a.seed)
    rep = replication(recs, dict(c.split("=", 1) for c in a.check)) if a.check else {}
    s["replication"] = rep
    with open(os.path.join(a.dir, "summary.json"), "w") as f:
        json.dump(s, f, indent=1)
    md = markdown(s, rep)
    with open(os.path.join(a.dir, "summary.md"), "w") as f:
        f.write(md)
    print(md, flush=True)
    if a.wandb:
        from robots.mujoco_lab import tracking

        cfgs = sorted(n for n in os.listdir(a.dir) if n.startswith("run_config_"))
        cfg = json.load(open(os.path.join(a.dir, cfgs[-1]))) if cfgs else {}
        flat = {}
        for stage, S in s["stages"].items():
            for prior, P in S["priors"].items():
                for row, R in P["rows"].items():
                    for q, Q in R.items():
                        for m, c in Q["spread"].items():
                            flat[f"{stage}/{prior}/{row}/{q}/{m}_change"] = c["change"]
                            flat[f"{stage}/{prior}/{row}/{q}/{m}_before"] = c["before"]
                            flat[f"{stage}/{prior}/{row}/{q}/{m}_after"] = c["after"]
                        for v, c in Q["split_after"].items():
                            flat[f"{stage}/{prior}/{row}/{q}/sd_{v}"] = c["after"]
                if "tags" in P:
                    flat.update({f"{stage}/{prior}/tags/{k}": v for k, v in P["tags"]["all"].items()})
        s["wandb"] = tracking.log_run("psuccess_spread", cfg, flat)
        with open(os.path.join(a.dir, "summary.json"), "w") as f:
            json.dump(s, f, indent=1)
    return s


def main(argv: list[str] | None = None) -> Any:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bundle", required=True)
    r.add_argument("--scenes", required=True, help="scenes.json (flat or upright; the generic prior's CoM)")
    r.add_argument("--priors", default=",".join(PRIORS))
    r.add_argument("--stages", default="weigh,attempt")
    r.add_argument("--objects", default=None, help="score only these objects (every case still draws its particles)")
    r.add_argument("--n", type=int, default=2000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--out", required=True)
    r.set_defaults(func=run)
    s = sub.add_parser("summary")
    s.add_argument("--dir", required=True)
    s.add_argument("--check", action="append", default=[], help="prior=DIR: estimation_check's run of that prior")
    s.add_argument("--n-boot", type=int, default=N_BOOT)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--wandb", action="store_true")
    s.set_defaults(func=summary)
    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    main()
