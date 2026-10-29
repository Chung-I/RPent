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


"""Calibration of the declared outcome probabilities against executed outcomes, by protocol.

Each case draws hidden properties and a destination from ``case_rng(seed, object, idx)`` (so the
protocols are paired case by case), resets the lab, and runs one protocol through the skill tools
(assess_pick, then pick_and_place with explicit option ids):

* ``commit``: the best row, whatever the recommendation (``--follow-recommendation`` abstains on
  decline instead: milestone 1's check);
* ``probe_first``: the probes the recommendation asks for (at most two), then the best row;
* ``repair``: the best row, then, when it did not place the object stably and the object is still in
  reach, the best row of the fresh assessment (the belief after the failure);
* ``random_row``: a uniformly drawn commit or lower-risk row to the requested destination (the
  control for the selection effect of executing the best row);
* ``probe_all`` (a diagnostic): the push, then the weigh (each when the menu offers it), whatever the
  recommendation, then the best row; it scores the probes' declared kinds when the recommendation
  rarely asks for a probe.

Every assigned case is kept: a thrown probe or an object out of reach ends it with its cost. The
calibration blocks score the executed attempts (``first_attempt``, and ``second_attempt`` for repair):
the seven-class Brier score against the training class rates, the ECE on P(placed_stable) with its
object-bootstrap 95 % interval and the plug-in rule (pass when the upper bound is under 0.05, fail
when the lower bound is over, else undecided), the one-vs-rest ECE per outcome, the iid floor, and the
simulated coverage of the interval. ``realized_cost`` (the cost table at care normal, probe time and
risk included) covers every case. ``probe_kind`` and ``thrown`` score the probes' declared kinds and
risk against the probe models' training rates.

Models: ``--bundle DIR`` (a deployed bundle; the objects default to its test split), ``--crossfit DIR``
(``train_models --folds``: each object is scored by the fold bundle trained without it; ``nested``
adds an isotonic map of P(placed_stable) fitted on the other folds' executions), or ``--metrics FILE``
(milestone 1: the default bundle on the metrics' test split).

    python -P -m robots.mujoco_lab.calibrate --crossfit robots/mujoco_lab/data/m2/folds --protocol commit \\
        --repeats 11 --seed 11 --workers 8 --out robots/mujoco_lab/data/m2/calibration_crossfit_commit.json
    python -P -m robots.mujoco_lab.calibrate --compare A.json B.json --out A_minus_B.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import zlib
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from robots.mujoco_lab import tracking
from robots.mujoco_lab.corpus import case_rng, draw_theta, trainable
from rpent.skills.pick_place.costs import CARE_SCALE, COST, TIME_COST_PER_S
from rpent.skills.pick_place.destinations import MESH_IDS, PRIMITIVE_IDS
from rpent.skills.pick_place.evaluate import (_ece, coverage_design, ece_ci, ece_from_sums, ece_iid_floor,
                                              object_sums, object_weights, per_class, score)
from rpent.skills.pick_place.features import extract
from rpent.skills.pick_place.probes import PROBE_KINDS, PROBE_SECONDS
from rpent.skills.pick_place.protocol import IRREVERSIBLE, OUTCOMES, Prediction
from rpent.skills.pick_place.tools import MAX_PROBES, assess_pick, pick_and_place

PROTOCOLS = ("commit", "probe_first", "repair", "random_row", "probe_all")
PROBE_ORDER = ("probe_push", "probe_weigh")     # probe_all: the order of scene A of the probe corpus
CARE = "normal"                   # the realized cost uses the cost table at care normal
ECE_TARGET = 0.05                 # the plug-in rule's bound on the ECE of P(placed_stable)
N_BOOT = 2000
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------- costs

def attempt_cost(outcome: str, cost_s: float, care: str = CARE) -> float:
    """An executed row: its outcome's cost (care scales the irreversible ones) and its time."""
    return float(COST[outcome] * (CARE_SCALE[care] if outcome in IRREVERSIBLE else 1.0) + TIME_COST_PER_S * float(cost_s))


def probe_cost(probe: str, kind: str, thrown: bool, care: str = CARE) -> float:
    """An executed probe: its time, a tipped push and a thrown object (the probe row's expected-cost terms)."""
    c = TIME_COST_PER_S * PROBE_SECONDS[probe]
    if probe == "push" and kind == "tipped":
        c += COST["tipped"]
    if thrown:
        c += COST["dropped_high"] * CARE_SCALE[care]
    return float(c)


# ---------------------------------------------------------------- one case

def _rows_to(a: dict[str, Any], destination: str) -> list[dict[str, Any]]:
    """The menu's commit and lower-risk rows to the requested destination (best() chooses among these)."""
    return [o for o in a["options"] if o["kind"] in ("commit", "lower_risk") and o["action"]["destination"] == destination]


def _probe_record(d: dict[str, Any], prims: Any, care: str) -> dict[str, Any]:
    pm = (getattr(prims, "probe_models", None) or {}).get(d["probed"])
    stats = getattr(pm, "stats", None) or {}
    n = int(stats.get("n") or 0)
    return {"probe": d["probed"], "kind": d["kind"], "reading": d["reading"], "thrown": bool(d["thrown"]),
            "displacement_m": d.get("displacement_m"), "tilt_deg": d.get("tilt_deg"), "declared": d.get("declared"),
            "const_kind": {k: v / n for k, v in stats["kinds"].items()} if n and stats.get("kinds") else None,
            "const_thrown": stats["thrown"] / n if n and "thrown" in stats else None,
            "cost": probe_cost(d["probed"], d["kind"], d["thrown"], care)}


def _attempt_record(d: dict[str, Any], row: dict[str, Any], a: dict[str, Any], features: Any, care: str) -> dict[str, Any]:
    return {"option": d["option"], "kind": row["kind"], "action": row["action"], "recommendation": a.get("recommendation"),
            "declared": {"p": row["p"], "p_drop_height": row["p_drop_height"], "expected_cost": row["expected_cost"]},
            "outcome": d["outcome"], "label": d["outcome"], "placed_at_requested": d["placed_at_requested"],
            "cost_s": row["cost_s"], "cost": attempt_cost(d["outcome"], row["cost_s"], care), "features": features,
            "diagnosis": d.get("diagnosis")}


def run_protocol(prims: Any, protocol: str, destination: str, rng: np.random.Generator, care: str = CARE, features_fn=None,
                 follow: bool = False, max_probes: int = MAX_PROBES) -> dict[str, Any]:
    """One case through the skill tools under ``protocol`` (see the module docstring). ``rng`` draws the
    random row; ``features_fn(row)`` gives the features of a row about to run (for the subgroups)."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}, not {protocol!r}")
    store: dict[str, Any] = {}
    case: dict[str, Any] = {"protocol": protocol, "recommendation": None, "best_option": None, "probes": [], "attempts": [],
                            "cost": 0.0, "ended": None}

    def end(reason: str, error: str | None = None) -> dict[str, Any]:
        case["ended"] = reason
        if error is not None:
            case["error"] = error
        case["cost"] = float(sum(p["cost"] for p in case["probes"]) + sum(t["cost"] for t in case["attempts"]))
        return case

    def call(**kw) -> dict[str, Any]:
        return pick_and_place(object="object", destination=destination, care=care, primitives=prims, store=store, **kw).data

    a = assess_pick(object="object", destination=destination, care=care, primitives=prims, store=store).data
    if "error" in a:
        return end("error", a["error"])
    if not a.get("assessed", True):
        return end("unreachable")
    case["recommendation"], case["best_option"] = a.get("recommendation"), a.get("best_option")
    if follow and a.get("recommendation") == "decline":
        return end("abstained")
    def probe(option_id: str) -> str | None:
        """Run one probe row; the reason the case ends, or None (``a`` is then the fresh assessment)."""
        nonlocal a
        d = call(option_id=option_id)
        if "error" in d:
            case["error"] = d["error"]
            return "error"
        case["probes"].append(_probe_record(d, prims, care))
        if d["thrown"]:
            return "thrown"
        a = d["assessment"]
        if "error" in a:
            case["error"] = a["error"]
            return "error"
        return None if a.get("assessed", False) else "unreachable"

    if protocol == "probe_first":
        while a.get("recommendation") == "probe" and len(case["probes"]) < max_probes:
            reason = probe(a["recommended_option"])
            if reason is not None:
                return end(reason)
    if protocol == "probe_all":
        for option_id in PROBE_ORDER:
            if any(o["id"] == option_id for o in a["options"]):
                reason = probe(option_id)
                if reason is not None:
                    return end(reason)
    for attempt in range(2 if protocol == "repair" else 1):
        if protocol == "random_row":
            rows = _rows_to(a, destination)
            row = rows[int(rng.integers(len(rows)))] if rows else None
        else:
            row = next((o for o in a["options"] if o["id"] == a.get("best_option")), None)
        if row is None:
            return end("no_row")
        features = features_fn(row) if features_fn is not None else None
        d = call(option_id=row["id"])
        if "outcome" not in d:
            return end("error", str(d.get("error") or d.get("reason")))
        case["attempts"].append(_attempt_record(d, row, a, features, care))
        if d["outcome"] == "placed_stable":
            return end("placed")
        a = d["assessment"]
        if protocol == "repair" and attempt == 0:
            if "error" in a:
                return end("error", a["error"])
            if not a.get("assessed", False):
                return end("unreachable")
    return end("failed")


def case_draw(seed: int, object_id: str, idx: int, nominal_mass: float, use_fixtures: bool = False) -> tuple[dict[str, Any], int, str]:
    """The case's hidden properties, CoM end and destination: one stream per (seed, object, idx)."""
    rng = case_rng(seed, object_id, idx)
    theta, com_end = draw_theta(rng, nominal_mass)
    if use_fixtures and MESH_IDS and rng.random() < 0.5:
        destination = str(rng.choice(sorted(MESH_IDS)))
    else:
        destination = str(rng.choice(sorted(PRIMITIVE_IDS)))
    return theta, com_end, destination


def _stream(seed: int, object_id: str, idx: int, k: int) -> np.random.Generator:
    """A second and third stream per case (the menu's draws; the random row), independent of the case order."""
    return np.random.default_rng([int(seed), zlib.crc32(object_id.encode()), int(idx), int(k)])


def run_case(prims: Any, object_id: str, idx: int, seed: int, protocol: str = "commit", follow: bool = False,
             use_fixtures: bool = False) -> dict[str, Any]:
    """Reset the lab to the case's draw and run the protocol; the first attempt's fields sit at the top
    level (milestone 1's row)."""
    from robots.mujoco_lab.adapter import object_props

    env = prims.env
    theta, com_end, destination = case_draw(seed, object_id, idx, object_props(object_id)["mass"], use_fixtures)
    env.reset(object_id, theta, seed=seed * 1000 + idx, com_end=com_end, fixture=destination if destination in MESH_IDS else None)
    reset_info = dict(getattr(env, "reset_info", None) or {})
    prims.rng = _stream(seed, object_id, idx, 1)
    truth = env.truth()

    def features_fn(row: dict[str, Any]) -> dict[str, float]:
        act = row["action"]
        prop = next(p for p in env.proposals() if abs(p.s - act["s"]) < 1e-9)
        return extract(env.geometry(), np.asarray(prop.point), prop.yaw, truth["mass"], truth["mu"], act["force"], act["v_lift"],
                       act["v_transport"], act["destination"], act.get("release_height"))

    case = run_protocol(prims, protocol, destination, _stream(seed, object_id, idx, 2), features_fn=features_fn, follow=follow)
    row = {"object": object_id, "idx": idx, "destination": destination, "theta": theta, "com_end": com_end,
           "truth": {k: truth[k] for k in ("mass", "mu", "com_s")}, "reset": reset_info, **case}
    first = case["attempts"][0] if case["attempts"] else None
    row.update(executed=first is not None, option=first and first["option"], outcome=first and first["outcome"],
               label=first and first["outcome"], placed_at_requested=first and first["placed_at_requested"],
               declared=first["declared"] if first else {"p": None})
    return row


# ---------------------------------------------------------------- running many cases

_STATE: dict[str, Any] = {}


def _primitives(bundle: str | None, threshold: float | None, model_kind: str | None) -> Any:
    """One lab env per process, and the primitives of each bundle on it (cached)."""
    from robots.mujoco_lab.adapter import MuJoCoLabEnv
    from robots.mujoco_lab.robot_spec import load_bundle, load_model
    from robots.mujoco_lab.toolkit import MuJoCoLabPrimitives

    if "env" not in _STATE:
        _STATE.update(env=MuJoCoLabEnv(), prims={})
    key = (bundle, threshold, model_kind)
    if key not in _STATE["prims"]:
        env = _STATE["env"]
        if bundle is None:
            prims = MuJoCoLabPrimitives(env=env, model=load_model(None, model_kind), threshold=threshold)
        else:
            B = load_bundle(bundle, model_kind)
            prims = MuJoCoLabPrimitives(env=env, model=B["model"], prior=B["prior"], readout_model=B["readout_model"],
                                        probe_models=B["probe_models"], threshold=threshold, manifest=B["manifest"])
        _STATE["prims"][key] = prims
    return _STATE["prims"][key]


def _run_object(job: tuple[str, list[int], dict[str, Any]]) -> list[dict[str, Any]]:
    object_id, idxs, s = job
    prims = _primitives(s["bundle"], s["threshold"], s["model_kind"])
    rows = []
    for idx in idxs:
        t0 = time.time()
        try:
            row = run_case(prims, object_id, idx, s["seed"], s["protocol"], s["follow"], s["fixtures"])
        except Exception as exc:  # noqa: BLE001  one bad case must not stop the run
            row = {"object": object_id, "idx": idx, "error": str(exc)[:200], "executed": False}
        row.update(fold=s["fold"], bundle=s["bundle"], const=s["const"], seconds=round(time.time() - t0, 2))
        rows.append(row)
        print(f"{object_id} {idx}: {row.get('error') or row.get('ended')} {row.get('outcome') or ''}", file=sys.stderr, flush=True)
    return rows


_LABELS: dict[str, list[tuple[str, str]]] = {}


def _corpus_labels(path: str) -> list[tuple[str, str]]:
    path = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if path not in _LABELS:
        out = []
        if os.path.exists(path):
            for line in open(path):
                r = json.loads(line)
                if trainable(r):
                    out.append((r["object"], r["label"]))
        _LABELS[path] = out
    return _LABELS[path]


def class_rates(metrics: dict[str, Any]) -> list[float]:
    """The training objects' class rates in OUTCOMES order (the constant baseline), uniform without the corpus."""
    paths = metrics["corpus"] if isinstance(metrics["corpus"], list) else [metrics["corpus"]]
    train = set(metrics["split"]["train"])
    labels = [lbl for p in paths for o, lbl in _corpus_labels(p) if o in train]
    if not labels:
        return [1.0 / len(OUTCOMES)] * len(OUTCOMES)
    c = Counter(labels)
    return [c[k] / len(labels) for k in OUTCOMES]


def _bundle_metrics(bundle: str) -> tuple[dict[str, Any], dict[str, Any]]:
    man = json.load(open(os.path.join(bundle, "manifest.json")))
    return man, json.load(open(os.path.join(bundle, man["metrics"])))


def plan_cases(a: argparse.Namespace) -> dict[str, Any]:
    """The objects, and per object the bundle that scores it, its fold and its constant baseline."""
    from robots.mujoco_lab.robot_spec import resolve_bundle

    assign: dict[str, tuple[str | None, int | None]] = {}
    consts: dict[str | None, list[float]] = {}
    kinds: dict[str, str] = {}
    if a.crossfit:
        folds = {o: int(k) for o, k in json.load(open(os.path.join(a.crossfit, "folds.json"))).items()}
        objects = a.objects.split(",") if a.objects else sorted(folds)
        missing = [o for o in objects if o not in folds]
        if missing:
            raise SystemExit(f"no fold for {missing} in {a.crossfit}/folds.json")
        for k in sorted({folds[o] for o in objects}):
            bundle = os.path.join(a.crossfit, f"fold_{k}")
            man, metrics = _bundle_metrics(bundle)
            trained = set(metrics["split"]["train"]) | set(metrics["split"]["val"])
            bad = sorted(o for o in objects if folds[o] == k and o in trained)
            if bad:
                raise SystemExit(f"fold {k} was trained on its held-out object(s) {bad}: folds.json and the fold bundles disagree")
            consts[bundle], kinds[str(k)] = class_rates(metrics), a.model_kind or man.get("kind", "classifier")
        for o in objects:
            assign[o] = (os.path.join(a.crossfit, f"fold_{folds[o]}"), folds[o])
        return {"objects": objects, "assign": assign, "consts": consts, "kind": kinds, "folds": {o: folds[o] for o in objects}}
    if a.bundle:
        man, metrics = _bundle_metrics(a.bundle)
        bundle, kind = a.bundle, a.model_kind or man.get("kind", "classifier")
    else:
        metrics = json.load(open(a.metrics))
        bundle = resolve_bundle(None)
        kind = a.model_kind or (_bundle_metrics(bundle)[0].get("kind", "classifier") if bundle else metrics.get("kind", "classifier"))
    objects = a.objects.split(",") if a.objects else list(metrics["split"]["test"])
    consts[bundle] = class_rates(metrics)
    for o in objects:
        assign[o] = (bundle, None)
    return {"objects": objects, "assign": assign, "consts": consts, "kind": kind, "folds": None}


def run_cases(a: argparse.Namespace, plan: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = []
    for o in plan["objects"]:
        bundle, fold = plan["assign"][o]
        jobs.append((o, list(range(a.repeats)), {"bundle": bundle, "fold": fold, "const": plan["consts"][bundle], "seed": a.seed,
                                                 "protocol": a.protocol, "follow": a.follow_recommendation, "fixtures": a.fixtures,
                                                 "threshold": a.threshold, "model_kind": a.model_kind}))
    rows: list[dict[str, Any]] = []
    if a.workers <= 1:
        for job in jobs:
            rows += _run_object(job)
    else:
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for part in pool.imap_unordered(_run_object, jobs):
                rows += part
    return sorted(rows, key=lambda r: (r["object"], r["idx"]))


# ---------------------------------------------------------------- scoring

def _attempt_row(case: dict[str, Any], j: int) -> dict[str, Any]:
    t = case["attempts"][j]
    return dict(t, object=case["object"], idx=case.get("idx"), destination=case["destination"], truth=case["truth"],
                fold=case.get("fold"), const=case.get("const"))


def ece_rule(ci: list[float] | None, bound: float = ECE_TARGET) -> str | None:
    """The plug-in rule: pass when the interval lies under the bound, fail when it lies over, else undecided."""
    if ci is None:
        return None
    return "pass" if ci[1] < bound else "fail" if ci[0] > bound else "undecided"


def calibration_block(att: list[dict[str, Any]], rng: np.random.Generator, coverage_sims: int = 0,
                      preds: list[Prediction] | None = None) -> dict[str, Any]:
    """Scores of executed attempts: Brier and its constant, the ECE of P(placed_stable) with its object
    interval and rule, per-class ECE, the iid floor, subgroups, and (``coverage_sims``) the interval's
    simulated coverage at the bound and at the observed ECE."""
    if not att:
        return {"n": 0, "ece": None, "ece_ci95_object_bootstrap": None, "brier": None, "brier_const": None, "by_subgroup": {},
                "per_class": None, "ece_iid_floor": None, "outcomes": {}, "by_option": {}, "by_destination": {}, "acceptance": None}
    preds = preds or [Prediction(p={k: float(t["declared"]["p"][k]) for k in OUTCOMES}) for t in att]
    const = np.array([t["const"] for t in att], dtype=float) if all(t.get("const") is not None for t in att) else None
    sc = score(preds, att, [], const=const)
    P = np.array([[x.p[k] for k in OUTCOMES] for x in preds])
    labels = [t["label"] for t in att]
    p, y, objects = P[:, 0], np.array([lbl == "placed_stable" for lbl in labels], dtype=float), [t["object"] for t in att]
    ci = ece_ci(p, y, objects, rng)
    by_opt: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "placed": 0})
    by_dest: dict[str, Counter] = defaultdict(Counter)
    for t in att:
        by_opt[t["option"]]["n"] += 1
        by_opt[t["option"]]["placed"] += t["label"] == "placed_stable"
        by_dest[t["destination"]][t["label"]] += 1
    block = {"n": len(att), "ece": sc["ece"], "ece_ci95_object_bootstrap": ci, "brier": sc["brier"], "brier_const": sc["brier_const"],
             "mean_p_placed": float(p.mean()), "rate_placed": float(y.mean()), "n_per_class": sc["n_per_class"],
             "per_class": per_class(P, labels), "ece_iid_floor": ece_iid_floor(p, rng), "by_subgroup": sc["by_subgroup"],
             "outcomes": dict(Counter(labels)), "by_option": dict(by_opt), "by_destination": {d: dict(c) for d, c in by_dest.items()},
             "acceptance": {"brier_below_const": bool(sc["brier"] < sc["brier_const"]), "ece_rule": ece_rule(ci)}}
    if coverage_sims:
        block["ci_coverage"] = {"at_bound": coverage_design(p, objects, ECE_TARGET, coverage_sims, rng, y=y),
                                "at_observed": coverage_design(p, objects, max(float(sc["ece"]), 1e-3), coverage_sims, rng, y=y)}
    return block


def _brier(P: np.ndarray, Y: np.ndarray) -> float:
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def probe_kind_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per probe: the declared kind probabilities of the executed probes against the kinds seen."""
    out = {}
    for probe, classes in PROBE_KINDS.items():
        recs = [p for r in rows for p in (r.get("probes") or []) if p["probe"] == probe and (p.get("declared") or {}).get("p_kind")]
        if not recs:
            out[probe] = {"n": 0, "kinds": {}, "brier": None, "brier_const": None, "below_const": None}
            continue
        Y = np.array([[float(p["kind"] == k) for k in classes] for p in recs])
        P = np.array([[float(p["declared"]["p_kind"].get(k, 0.0)) for k in classes] for p in recs])
        have_const = all(p.get("const_kind") for p in recs)
        C = np.array([[float(p["const_kind"].get(k, 0.0)) for k in classes] for p in recs]) if have_const else None
        b, bc = _brier(P, Y), (_brier(C, Y) if C is not None else None)
        out[probe] = {"n": len(recs), "kinds": dict(Counter(p["kind"] for p in recs)), "brier": b, "brier_const": bc,
                      "below_const": None if bc is None else bool(b < bc)}
    return out


def thrown_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The declared probability that a probe throws the object against the throws seen."""
    recs = [p for r in rows for p in (r.get("probes") or []) if (p.get("declared") or {}).get("p_thrown") is not None]
    if not recs:
        return {"n": 0, "n_thrown": 0, "brier": None, "brier_const": None, "below_const": None}
    y = np.array([float(p["thrown"]) for p in recs])
    q = np.array([float(p["declared"]["p_thrown"]) for p in recs])
    have_const = all(p.get("const_thrown") is not None for p in recs)
    b = float(np.mean((q - y) ** 2))
    bc = float(np.mean((np.array([p["const_thrown"] for p in recs]) - y) ** 2)) if have_const else None
    return {"n": len(recs), "n_thrown": int(y.sum()), "brier": b, "brier_const": bc, "below_const": None if bc is None else bool(b < bc)}


def probes_run(rows: list[dict[str, Any]]) -> dict[str, int]:
    c = Counter(p["probe"] for r in rows for p in (r.get("probes") or []))
    return {"push": c["push"], "weigh": c["weigh"], "cases_with_probes": sum(1 for r in rows if r.get("probes")), "cases": len(rows)}


def _mean_ci(values: np.ndarray, inv: np.ndarray, n_objects: int, rng: np.random.Generator, n_boot: int = N_BOOT) -> list[float]:
    W = object_weights(n_objects, n_boot, rng)
    S, N = np.bincount(inv, weights=values, minlength=n_objects), np.bincount(inv, minlength=n_objects).astype(float)
    m = (W @ S) / np.maximum(W @ N, 1e-12)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def realized_cost(rows: list[dict[str, Any]], rng: np.random.Generator, n_boot: int = N_BOOT) -> dict[str, Any]:
    """The mean realized cost over every case (errors excepted and counted), its object-bootstrap 95 %
    interval, and the mean by how the case ended."""
    ok = [r for r in rows if "error" not in r]
    if not ok:
        return {"n": 0, "n_errors": len(rows), "mean": None, "ci95_object_bootstrap": None, "by_end": {}}
    c = np.array([float(r["cost"]) for r in ok])
    _, inv = np.unique([r["object"] for r in ok], return_inverse=True)
    by_end: dict[str, list[float]] = defaultdict(list)
    for r, v in zip(ok, c):
        by_end[str(r.get("ended"))].append(float(v))
    return {"n": len(ok), "n_errors": len(rows) - len(ok), "mean": float(c.mean()),
            "ci95_object_bootstrap": _mean_ci(c, inv, int(inv.max()) + 1, rng, n_boot),
            "probe_part": float(np.mean([sum(p["cost"] for p in r.get("probes") or []) for r in ok])),
            "by_end": {e: {"n": len(v), "mean": float(np.mean(v))} for e, v in sorted(by_end.items())}}


def _rescale(P: dict[str, float], q: float) -> Prediction:
    """P(placed_stable) set to q; the other classes rescaled to sum to 1 - q."""
    rest = float(sum(float(P[k]) for k in OUTCOMES[1:]))
    out = {"placed_stable": float(q)}
    for k in OUTCOMES[1:]:
        out[k] = float(P[k]) * (1.0 - q) / rest if rest > 1e-12 else (1.0 - q) / (len(OUTCOMES) - 1)
    return Prediction(p=out)


def nested_isotonic(att: list[dict[str, Any]]) -> list[Prediction]:
    """For each fold, an isotonic map of P(placed_stable) fitted on the executions of the objects outside
    it (scored by their own fold bundles), applied to the fold's executions; the other classes rescale."""
    from sklearn.isotonic import IsotonicRegression

    p = np.array([float(t["declared"]["p"]["placed_stable"]) for t in att])
    y = np.array([float(t["label"] == "placed_stable") for t in att])
    folds = np.array([t["fold"] for t in att])
    out: list[Prediction | None] = [None] * len(att)
    for k in sorted(set(folds.tolist())):
        tr, te = folds != k, np.where(folds == k)[0]
        if tr.any():
            q = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p[tr], y[tr]).predict(p[te])
        else:
            q = p[te]
        for i, qi in zip(te, q):
            out[i] = _rescale(att[i]["declared"]["p"], float(qi))
    return out  # type: ignore[return-value]


def summarize(rows: list[dict[str, Any]], protocol: str, rng: np.random.Generator, coverage_sims: int = 0,
              nested: bool = False) -> dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    first = [_attempt_row(r, 0) for r in ok if r.get("attempts")]
    second = [_attempt_row(r, 1) for r in ok if len(r.get("attempts") or []) > 1]
    b1 = calibration_block(first, rng, coverage_sims)
    rec = [r["recommendation"] for r in ok if r.get("recommendation")]
    out = {"protocol": protocol, "n": len(rows), "n_executed": len(first),
           "recommendation_decline_rate": float(np.mean([x == "decline" for x in rec])) if rec else None,
           "recommendations": dict(Counter(rec)), "abstained": sum(1 for r in ok if r.get("ended") == "abstained"),
           "errors": len(rows) - len(ok), "ended": dict(Counter(str(r.get("ended")) for r in ok)),
           "first_attempt": b1, "second_attempt": calibration_block(second, rng) if protocol == "repair" else None}
    out.update({k: b1[k] for k in ("ece", "ece_ci95_object_bootstrap", "brier", "brier_const", "by_subgroup", "by_destination",
                                   "by_option", "outcomes", "per_class", "ece_iid_floor")})
    out.update(probe_kind=probe_kind_block(ok), thrown=thrown_block(ok), probes_run=probes_run(ok), realized_cost=realized_cost(rows, rng))
    if nested and first and protocol != "random_row":
        out["nested"] = calibration_block(first, rng, preds=nested_isotonic(first))
    out["rows"] = rows
    return out


def compare(A: dict[str, Any], B: dict[str, Any], rng: np.random.Generator, n_boot: int = N_BOOT) -> dict[str, Any]:
    """A − B over the paired cases (same object and idx): the mean realized cost and the first attempts'
    ECE on P(placed_stable), each with a 95 % interval that resamples objects jointly for both."""
    ka = {(r["object"], r["idx"]): r for r in A["rows"] if "error" not in r}
    kb = {(r["object"], r["idx"]): r for r in B["rows"] if "error" not in r}
    keys = sorted(set(ka) & set(kb))
    objs = sorted({o for o, _ in keys})
    oi = {o: i for i, o in enumerate(objs)}
    K, inv = len(objs), np.array([oi[o] for o, _ in keys], dtype=int)
    W = object_weights(K, n_boot, rng)
    ca, cb = np.array([float(ka[k]["cost"]) for k in keys]), np.array([float(kb[k]["cost"]) for k in keys])
    N = np.bincount(inv, minlength=K).astype(float)
    boot = (W @ np.bincount(inv, weights=ca - cb, minlength=K)) / np.maximum(W @ N, 1e-12)

    def sums(rows: dict) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], float | None, int]:
        sel = [(oi[k[0]], rows[k]["attempts"][0]) for k in keys if rows[k].get("attempts")]
        p = np.array([float(t["declared"]["p"]["placed_stable"]) for _, t in sel])
        y = np.array([float(t["label"] == "placed_stable") for _, t in sel])
        idx = np.array([i for i, _ in sel], dtype=int)
        return object_sums(p, y, idx, K), (_ece(p, y) if len(sel) else None), len(sel)

    (sa, ea, na), (sb, eb, nb) = sums(ka), sums(kb)
    d = ece_from_sums(W, *sa) - ece_from_sums(W, *sb)
    return {"protocol_a": A.get("protocol"), "protocol_b": B.get("protocol"), "n_paired": len(keys), "n_objects": K,
            "unpaired": {"a_only": len(set(ka) - set(kb)), "b_only": len(set(kb) - set(ka))},
            "realized_cost": {"a": float(ca.mean()), "b": float(cb.mean()), "diff": float((ca - cb).mean()),
                              "ci95_object_bootstrap": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]},
            "ece_first_attempt": {"a": ea, "b": eb, "n_a": na, "n_b": nb, "diff": None if ea is None or eb is None else ea - eb,
                                  "ci95_object_bootstrap": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}}


# ---------------------------------------------------------------- CLI

def _write(out: dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=float)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--metrics", default=None, help="milestone 1: success_model_metrics.json (its test split; the default bundle)")
    src.add_argument("--bundle", default=None, help="a model bundle (its test split)")
    src.add_argument("--crossfit", default=None, help="train_models --folds output: folds.json and fold_<k>/ bundles")
    src.add_argument("--compare", nargs=2, default=None, metavar=("A", "B"), help="two result files: A - B over the paired cases")
    ap.add_argument("--protocol", default="commit", choices=PROTOCOLS)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--objects", default=None, help="comma-separated ids (default: the test split, or every object of the folds)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--follow-recommendation", action="store_true", help="commit protocol: abstain on a decline recommendation")
    ap.add_argument("--threshold", type=float, default=None, help="override the commit threshold (tests)")
    ap.add_argument("--model-kind", default=None)
    ap.add_argument("--fixtures", action="store_true", help="draw half the destinations from the inspected mesh fixtures")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--coverage-sims", type=int, default=200, help="simulations for the interval coverage (0: skip)")
    a = ap.parse_args(argv)
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    if a.compare:
        out = dict(compare(json.load(open(a.compare[0])), json.load(open(a.compare[1])), rng), a=a.compare[0], b=a.compare[1])
        _write(out, a.out)
        out["wandb"] = tracking.log_run("calibrate_compare", {"a": a.compare[0], "b": a.compare[1]}, out)
        print(json.dumps({k: out[k] for k in ("protocol_a", "protocol_b", "n_paired", "realized_cost", "ece_first_attempt")}, indent=1))
        return out
    if not (a.metrics or a.bundle or a.crossfit):
        ap.error("give --metrics, --bundle, --crossfit or --compare")
    if a.follow_recommendation and a.protocol != "commit":
        ap.error("--follow-recommendation applies to the commit protocol")
    plan = plan_cases(a)
    rows = run_cases(a, plan)
    out = summarize(rows, a.protocol, rng, a.coverage_sims, nested=bool(a.crossfit))
    out.update(kind=plan["kind"], objects=plan["objects"], folds=plan["folds"], seed=a.seed, repeats=a.repeats,
               source={"crossfit": a.crossfit, "bundle": a.bundle, "metrics": a.metrics}, seconds=round(time.time() - t0, 1))
    _write(out, a.out)
    summary = {k: out[k] for k in ("protocol", "kind", "n", "n_executed", "abstained", "errors", "ece", "ece_ci95_object_bootstrap",
                                   "brier", "brier_const", "ece_iid_floor", "outcomes", "probes_run", "probe_kind", "thrown")}
    summary.update(realized_cost={k: out["realized_cost"][k] for k in ("n", "mean", "ci95_object_bootstrap")},
                   acceptance=out["first_attempt"]["acceptance"], seconds=out["seconds"])
    out["wandb"] = tracking.log_run("calibrate", {"protocol": a.protocol, "seed": a.seed, "repeats": a.repeats, "source": out["source"],
                                                  "n_objects": len(plan["objects"]), "threshold": a.threshold}, summary)
    print(json.dumps(summary, indent=1, default=float))
    return out


if __name__ == "__main__":
    main()
    sys.exit(0)
