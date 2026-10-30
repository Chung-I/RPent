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
  rarely asks for a probe;
* ``policy``: the recommendation at every step: decline abstains, probe runs the recommended probe (at
  most ``MAX_PROBES``; past that, the menu's rule on the best row alone), commit runs the best row, and a
  recoverable failure is reassessed and followed again (at most ``POLICY_ATTEMPTS`` attempts).

Each case keeps the novelty guard's check of its first assessment (``guard``, None without a guard), and the
summary counts the cases it checked and flagged; the commit protocol runs the best row after a guard decline too.

Every assigned case is kept: a thrown probe or an object out of reach ends it with its cost. A case
with a flagged probe or attempt, or whose reset did not reach rest (``artifact``, adapter.artifact: pad
contacts that were not physical), stays in ``rows`` but every metric leaves it out and counts it. The
calibration blocks score the executed attempts (``first_attempt``, and ``second_attempt`` for repair):
the eleven-class Brier score against the training class rates, the ECE on P(placed_stable) with its
object-bootstrap 95 % interval and the plug-in rule (pass when the upper bound is under 0.05, fail
when the lower bound is over, else undecided), the one-vs-rest ECE per outcome, the iid floor, and the
simulated coverage of the interval. ``realized_cost`` (the cost table at the run's care, probe time and
risk included, and the abstain cost A of the primitives for a case that ends without the object placed:
``--abstain-cost``, by default the bundle's decision, none for m2) covers every case. ``probe_kind`` and
``thrown`` score the probes' declared kinds and risk against the probe models' training rates.

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
from robots.mujoco_lab.corpus import apply_damage, case_rng, damage_rng, draw_damage, draw_theta, trainable
from rpent.skills.pick_place.costs import COST, TIME_COST_PER_S, care_scale
from rpent.skills.pick_place.destinations import MESH_IDS, PRIMITIVE_IDS
from rpent.skills.pick_place.evaluate import (_ece, coverage_design, ece_ci, ece_from_sums, ece_iid_floor,
                                              object_sums, object_weights, per_class, score)
from rpent.skills.pick_place.features import extract
from rpent.skills.pick_place.probes import DAMAGED, PROBE_KINDS, PROBE_SECONDS
from rpent.skills.pick_place.protocol import DAMAGE, IRREVERSIBLE, MOTION_OUTCOMES, OUTCOMES, RECOVERABLE, Prediction
from rpent.skills.pick_place.tools import MAX_PROBES, _threshold, assess_pick, pick_and_place

PROTOCOLS = ("commit", "probe_first", "repair", "random_row", "probe_all", "policy")
POLICY_ATTEMPTS = 3               # policy: the attempts per case (a recoverable failure is reassessed)
PROBE_ORDER = ("probe_push", "probe_weigh")     # probe_all: the order of scene A of the probe corpus
CARE = "normal"                   # the default care of the realized cost
ECE_TARGET = 0.05                 # the plug-in rule's bound on the ECE of P(placed_stable)
N_BOOT = 2000
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------- costs

def attempt_cost(outcome: str, cost_s: float, care: str | float = CARE) -> float:
    """An executed row: its outcome's cost (care scales the irreversible ones) and its time."""
    return float(COST[outcome] * (care_scale(care) if outcome in IRREVERSIBLE else 1.0) + TIME_COST_PER_S * float(cost_s))


def probe_cost(probe: str, kind: str, thrown: bool, care: str | float = CARE) -> float:
    """An executed probe: its time, a tipped push, and a thrown or damaged object, charged once (the probe row's
    expected-cost terms; the case adds A when it ends, see run_protocol)."""
    c = TIME_COST_PER_S * PROBE_SECONDS[probe]
    if probe == "push" and kind == "tipped":
        c += COST["tipped"]
    if thrown or kind == DAMAGED:
        c += COST["dropped_high"] * care_scale(care)
    return float(c)


# ---------------------------------------------------------------- one case

def _rows_to(a: dict[str, Any], destination: str) -> list[dict[str, Any]]:
    """The menu's commit and lower-risk rows to the requested destination (best() chooses among these)."""
    return [o for o in a["options"] if o["kind"] in ("commit", "lower_risk") and o["action"]["destination"] == destination]


def _without_probe(a: dict[str, Any], prims: Any) -> str:
    """policy, once it may not probe again: the menu's rule on the best row alone (the cost rule: commit when the
    row costs at most A and reaches the floor, if one is set; the milestone 2a rule: P(placed_stable) at the
    threshold). It reads the rounded payload, so a tie within 0.0005 can go either way."""
    row = next((o for o in a["options"] if o["id"] == a.get("best_option")), None)
    if row is None:
        return "decline"
    A, floor, p = getattr(prims, "abstain_cost", None), getattr(prims, "threshold", None), float(row["p"]["placed_stable"])
    if A is None:
        return "commit" if p >= _threshold(prims) else "decline"
    return "commit" if float(row["expected_cost"]) <= float(A) and (floor is None or p >= float(floor)) else "decline"


def _probe_record(d: dict[str, Any], prims: Any, care: str | float) -> dict[str, Any]:
    pm = (getattr(prims, "probe_models", None) or {}).get(d["probed"])
    stats = getattr(pm, "stats", None) or {}
    n, kinds = int(stats.get("n") or 0), stats.get("kinds") or {}
    n_kinds = sum(int(v) for v in kinds.values())                     # the kind model's rows (2b: the results not thrown)
    return {"probe": d["probed"], "kind": d["kind"], "reading": d["reading"], "thrown": bool(d["thrown"]), "damage": d.get("damage"),
            "artifact": bool(d.get("artifact", False)), "displacement_m": d.get("displacement_m"), "tilt_deg": d.get("tilt_deg"),
            "declared": d.get("declared"),
            "const_kind": {k: v / n_kinds for k, v in kinds.items()} if n_kinds else None,
            "kinds_not_thrown": bool(getattr(pm, "kinds_not_thrown", True)),
            "const_thrown": stats["thrown"] / n if n and "thrown" in stats else None,
            "cost": probe_cost(d["probed"], d["kind"], d["thrown"], care)}


def _attempt_record(d: dict[str, Any], row: dict[str, Any], a: dict[str, Any], features: Any, care: str | float) -> dict[str, Any]:
    return {"option": d["option"], "kind": row["kind"], "action": row["action"], "recommendation": a.get("recommendation"),
            "declared": {"p": row["p"], "p_drop_height": row["p_drop_height"], "expected_cost": row["expected_cost"]},
            "outcome": d["outcome"], "label": d["outcome"], "placed_at_requested": d["placed_at_requested"],
            "cost_s": row["cost_s"], "cost": attempt_cost(d["outcome"], row["cost_s"], care), "features": features,
            "diagnosis": d.get("diagnosis"), "artifact": bool(d.get("artifact", False))}


def run_protocol(prims: Any, protocol: str, destination: str, rng: np.random.Generator, care: str | float = CARE, features_fn=None,
                 follow: bool = False, max_probes: int = MAX_PROBES) -> dict[str, Any]:
    """One case through the skill tools under ``protocol`` (see the module docstring). ``rng`` draws the
    random row; ``features_fn(row)`` gives the features of a row about to run (for the subgroups). The case
    costs its probes, its attempts and ``prims.abstain_cost`` (when set) if it ends without the object placed."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}, not {protocol!r}")
    store: dict[str, Any] = {}
    case: dict[str, Any] = {"protocol": protocol, "recommendation": None, "best_option": None, "guard": None, "probes": [],
                            "attempts": [], "cost": 0.0, "ended": None}

    def end(reason: str, error: str | None = None) -> dict[str, Any]:
        case["ended"] = reason
        if error is not None:
            case["error"] = error
        A = getattr(prims, "abstain_cost", None)
        case["abstain_part"] = 0.0 if A is None or reason in ("placed", "error") else float(A)
        case["cost"] = float(sum(p["cost"] for p in case["probes"]) + sum(t["cost"] for t in case["attempts"]) + case["abstain_part"])
        case["artifact"] = any(x["artifact"] for x in case["probes"] + case["attempts"])
        return case

    def call(**kw) -> dict[str, Any]:
        return pick_and_place(object="object", destination=destination, care=care, primitives=prims, store=store, **kw).data

    a = assess_pick(object="object", destination=destination, care=care, primitives=prims, store=store).data
    if "error" in a:
        return end("error", a["error"])
    if not a.get("assessed", True):
        return end("unreachable")
    case["recommendation"], case["best_option"], case["guard"] = a.get("recommendation"), a.get("best_option"), a.get("guard")
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
        if d["kind"] == DAMAGED or d.get("damage"):
            return "damaged"
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
    if protocol == "policy":
        while True:
            rec = a.get("recommendation")
            if rec == "probe" and len(case["probes"]) >= max_probes:
                rec = _without_probe(a, prims)
            if rec == "decline":
                return end("abstained")
            if rec == "probe":
                reason = probe(a["recommended_option"])
                if reason is not None:
                    return end(reason)
                continue
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
            if d["outcome"] not in RECOVERABLE or len(case["attempts"]) >= POLICY_ATTEMPTS:
                return end("failed")
            a = d["assessment"]
            if "error" in a:
                return end("error", a["error"])
            if not a.get("assessed", False):
                return end("unreachable")
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


def case_draw(seed: int, object_id: str, idx: int, nominal_mass: float, use_fixtures: bool = False,
              fragility: tuple[dict[str, Any], Any] | None = None) -> tuple[dict[str, Any], int, str, dict[str, Any]]:
    """The case's hidden properties, CoM end, destination and hidden damage state {fill, frag, spill_angle_deg}: one
    stream per (seed, object, idx). In an m2b world (``fragility`` = (classes, thresholds)) the fill and the fragility
    come from the corpus's damage stream with the corpus rule (corpus.draw_damage, apply_damage), so a case hides what
    the corpus row of that (seed, object, idx) hides; an m2 case is empty and cannot be damaged."""
    from robots.mujoco_lab.fragility import entry_of

    rng = case_rng(seed, object_id, idx)
    theta, com_end = draw_theta(rng, nominal_mass)
    if use_fixtures and MESH_IDS and rng.random() < 0.5:
        destination = str(rng.choice(sorted(MESH_IDS)))
    else:
        destination = str(rng.choice(sorted(PRIMITIVE_IDS)))
    hidden: dict[str, Any] = {"fill": 0.0, "frag": None, "spill_angle_deg": None}
    if fragility is not None:
        classes, th = fragility
        d = draw_damage(damage_rng(seed, object_id, idx), entry_of(object_id, classes)[0], th, 1.0)
        theta, _ = apply_damage(theta, {}, d)
        hidden = {"fill": d["fill"], "frag": d["theta_frag"], "spill_angle_deg": d["spill_angle_deg"]}
    return theta, com_end, destination, hidden


def case_artifact(case: dict[str, Any], reset_info: dict[str, Any]) -> bool:
    """A case the metrics leave out: a probe or an attempt was flagged, or the reset did not reach rest."""
    return bool(case.get("artifact")) or not reset_info.get("at_rest", True)


def _stream(seed: int, object_id: str, idx: int, k: int) -> np.random.Generator:
    """A second and third stream per case (the menu's draws; the random row), independent of the case order."""
    return np.random.default_rng([int(seed), zlib.crc32(object_id.encode()), int(idx), int(k)])


def run_case(prims: Any, object_id: str, idx: int, seed: int, protocol: str = "commit", follow: bool = False,
             use_fixtures: bool = False, care: str | float = CARE) -> dict[str, Any]:
    """Reset the lab to the case's draw and run the protocol; the first attempt's fields sit at the top
    level (milestone 1's row)."""
    from robots.mujoco_lab.adapter import object_props
    from robots.mujoco_lab.prior import fragility_world

    env = prims.env
    env.force_noise = getattr(prims, "force_noise", None)       # the job's sensor (job_force_noise); None: the lab default
    world = fragility_world(getattr(prims, "manifest", None))
    theta, com_end, destination, hidden = case_draw(seed, object_id, idx, object_props(object_id)["mass"], use_fixtures,
                                                    fragility=world)
    if world is not None:
        env.slosh_zeta = world[1].slosh_zeta
        env.slosh_scale = world[1].slosh_scale
    env.reset(object_id, theta, seed=seed * 1000 + idx, com_end=com_end, fixture=destination if destination in MESH_IDS else None,
              fill=hidden["fill"], frag=hidden["frag"])
    reset_info = dict(getattr(env, "reset_info", None) or {})
    prims.rng = _stream(seed, object_id, idx, 1)
    truth = env.truth()

    def features_fn(row: dict[str, Any]) -> dict[str, float]:
        act = row["action"]
        prop = next(p for p in env.proposals() if abs(p.s - act["s"]) < 1e-9)
        return extract(env.geometry(), np.asarray(prop.point), prop.yaw, truth["mass"], truth["mu"], act["force"], act["v_lift"],
                       act["v_transport"], act["destination"], act.get("release_height"))

    case = run_protocol(prims, protocol, destination, _stream(seed, object_id, idx, 2), care=care, features_fn=features_fn,
                        follow=follow)
    case["artifact"] = case_artifact(case, reset_info)
    row = {"object": object_id, "idx": idx, "destination": destination, "theta": theta, "com_end": com_end,
           "fill": hidden["fill"], "theta_frag": hidden["frag"], "force_noise": env.force_noise,
           "evoi": getattr(prims, "evoi", "plugin"),
           "truth": {k: truth[k] for k in ("mass", "mu", "com_s")}, "reset": reset_info, **case}
    first = case["attempts"][0] if case["attempts"] else None
    row.update(executed=first is not None, option=first and first["option"], outcome=first and first["outcome"],
               label=first and first["outcome"], placed_at_requested=first and first["placed_at_requested"],
               declared=first["declared"] if first else {"p": None})
    return row


# ---------------------------------------------------------------- running many cases

_STATE: dict[str, Any] = {}


def job_force_noise(override: float | None, manifest: dict[str, Any] | None) -> float | None:
    """A job's force noise (plan 2b2 decision 8): ``--force-noise`` when given, else the bundle's declared sensor
    (robot_spec.manifest_force_noise: 0 for a bundle from before plan 2b2), else None (no bundle: the lab default)."""
    if override is not None:
        return float(override)
    if manifest is None:
        return None
    from robots.mujoco_lab.robot_spec import manifest_force_noise

    return manifest_force_noise(manifest)


def run_evoi(rows: list[dict[str, Any]]) -> str | list[str] | None:
    """The probe-value estimator a run's cases used: one name, the sorted names when they differ, None without cases."""
    names = sorted({r["evoi"] for r in rows if r.get("evoi") is not None})
    return None if not names else names[0] if len(names) == 1 else names


def run_force_noise(rows: list[dict[str, Any]]) -> float | list[float] | None:
    """The force noise a run's cases read with: one level, the sorted levels when they differ, None without a case."""
    levels = sorted({float(r["force_noise"]) for r in rows if r.get("force_noise") is not None})
    return levels[0] if len(levels) == 1 else (levels or None)


def _primitives(bundle: str | None, threshold: float | None, model_kind: str | None) -> Any:
    """One lab env per process, and the primitives of each bundle on it (cached). Every call returns them with the
    bundle's abstain cost and force noise: a job sets its own after the call, and none stays for the next caller."""
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
                                        probe_models=B["probe_models"], threshold=threshold, manifest=B["manifest"],
                                        guard=B["guard"], abstain_cost=B["abstain_cost"], force_noise=B["force_noise"],
                                        evoi=B["evoi"])
        _STATE["prims"][key] = (prims, {"abstain_cost": prims.abstain_cost, "force_noise": prims.force_noise})
    prims, own = _STATE["prims"][key]
    for name, value in own.items():
        setattr(prims, name, value)
    return prims


def _run_object(job: tuple[str, list[int], dict[str, Any]]) -> list[dict[str, Any]]:
    object_id, idxs, s = job
    prims = _primitives(s["bundle"], s["threshold"], s["model_kind"])
    prims.abstain_cost = s["abstain_cost"]       # every job sets it: the cached primitives keep no earlier run's value
    prims.force_noise = job_force_noise(s.get("force_noise"), prims.manifest)
    rows = []
    for idx in idxs:
        t0 = time.time()
        try:
            row = run_case(prims, object_id, idx, s["seed"], s["protocol"], s["follow"], s["fixtures"], s["care"])
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
    from robots.mujoco_lab.robot_spec import manifest_abstain_cost, resolve_bundle

    assign: dict[str, tuple[str | None, int | None]] = {}
    decision: dict[str | None, float | None] = {}
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
            decision[bundle] = manifest_abstain_cost(man)
        for o in objects:
            assign[o] = (os.path.join(a.crossfit, f"fold_{folds[o]}"), folds[o])
        return {"objects": objects, "assign": assign, "consts": consts, "kind": kinds, "folds": {o: folds[o] for o in objects},
                "decision": decision}
    if a.bundle:
        man, metrics = _bundle_metrics(a.bundle)
        bundle, kind = a.bundle, a.model_kind or man.get("kind", "classifier")
    else:
        metrics = json.load(open(a.metrics))
        bundle = resolve_bundle(None)
        man = _bundle_metrics(bundle)[0] if bundle else {}
        kind = a.model_kind or (man.get("kind", "classifier") if bundle else metrics.get("kind", "classifier"))
    objects = a.objects.split(",") if a.objects else list(metrics["split"]["test"])
    consts[bundle], decision[bundle] = class_rates(metrics), manifest_abstain_cost(man)
    for o in objects:
        assign[o] = (bundle, None)
    return {"objects": objects, "assign": assign, "consts": consts, "kind": kind, "folds": None, "decision": decision}


def run_abstain_cost(a: argparse.Namespace, plan: dict[str, Any]) -> float | None:
    """The run's abstain cost: ``--abstain-cost`` (a number, or "none"), else the bundles' decision, which must agree."""
    if getattr(a, "abstain_cost", None) is not None:
        return None if a.abstain_cost == "none" else float(a.abstain_cost)
    values = set(plan["decision"].values())
    if len(values) > 1:
        raise SystemExit(f"the bundles disagree on the abstain cost: {plan['decision']}; pass --abstain-cost")
    return values.pop()


def run_cases(a: argparse.Namespace, plan: dict[str, Any]) -> list[dict[str, Any]]:
    jobs, A = [], run_abstain_cost(a, plan)
    for o in plan["objects"]:
        bundle, fold = plan["assign"][o]
        jobs.append((o, list(range(a.repeats)), {"bundle": bundle, "fold": fold, "const": plan["consts"][bundle], "seed": a.seed,
                                                 "protocol": a.protocol, "follow": a.follow_recommendation, "fixtures": a.fixtures,
                                                 "threshold": a.threshold, "model_kind": a.model_kind, "care": a.care,
                                                 "abstain_cost": A, "force_noise": getattr(a, "force_noise", None)}))
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
    preds = preds or [Prediction(p={k: float(t["declared"]["p"].get(k, 0.0)) for k in OUTCOMES}) for t in att]
    const = np.array([t["const"] for t in att], dtype=float) if all(t.get("const") is not None for t in att) else None
    sc = score(preds, att, [], const=const)
    P = np.array([[x.p.get(k, 0.0) for k in OUTCOMES] for x in preds])
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


def irreversible_block(att: list[dict[str, Any]], rng: np.random.Generator) -> dict[str, Any]:
    """Binary calibration of the sum over IRREVERSIBLE; damage ECE needs at least 50 positives.

    The constant is each attempt's training class rates (its fold's for cross-fit), or the
    uniform outcome baseline when no rates are available, as in calibration_block.
    """
    if not att:
        return {"n": 0, "n_irreversible": 0, "ece": None, "ece_ci95_object_bootstrap": None,
                "brier": None, "brier_const": None, "per_class": None, "acceptance": None}
    P = np.array([[float(t["declared"]["p"].get(k, 0.0)) for k in OUTCOMES] for t in att])
    labels = [t["label"] for t in att]
    p = P[:, [OUTCOMES.index(k) for k in IRREVERSIBLE]].sum(axis=1)
    y = np.array([k in IRREVERSIBLE for k in labels], dtype=float)
    if all(t.get("const") is not None for t in att):
        c = np.array([sum(float(v) for k, v in zip(OUTCOMES, t["const"]) if k in IRREVERSIBLE) for t in att])
    else:
        c = np.full(len(att), len(IRREVERSIBLE) / len(OUTCOMES))
    ci = ece_ci(p, y, [t["object"] for t in att], rng)
    brier, baseline = float(np.mean((p - y) ** 2)), float(np.mean((c - y) ** 2))
    classes = per_class(P, labels)
    return {"n": len(att), "n_irreversible": int(y.sum()), "mean_p_irreversible": float(p.mean()),
            "rate_irreversible": float(y.mean()), "ece": _ece(p, y), "ece_ci95_object_bootstrap": ci,
            "brier": brier, "brier_const": baseline, "per_class": {k: classes[k] for k in DAMAGE},
            "acceptance": {"brier_below_const": bool(brier < baseline), "ece_rule": ece_rule(ci)}}


def _brier(P: np.ndarray, Y: np.ndarray) -> float:
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def probe_kind_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per probe: the declared kind probabilities of the executed probes against the kinds seen ("damaged" among
    them; a model that never saw it declares 0). A 2b model declares the kinds of a result not thrown
    (``kinds_not_thrown`` in the record), so its thrown probes are left out here and scored by thrown_block; a record
    without the flag (a milestone 2a run) keeps them."""
    out = {}
    for probe, kinds in PROBE_KINDS.items():
        classes = kinds + (DAMAGED,)
        recs = [p for r in rows for p in (r.get("probes") or [])
                if p["probe"] == probe and not (p.get("thrown") and p.get("kinds_not_thrown", False))
                and (p.get("declared") or {}).get("p_kind")]
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


def guard_block(rows: list[dict[str, Any]]) -> dict[str, int] | None:
    """The cases whose first assessment the novelty guard checked, and those it flagged; None without a guard."""
    checks = [r["guard"] for r in rows if r.get("guard") is not None]
    return {"n_checked": len(checks), "n_flagged": sum(bool(c["flag"]) for c in checks)} if checks else None


def probes_run(rows: list[dict[str, Any]]) -> dict[str, int]:
    c = Counter(p["probe"] for r in rows for p in (r.get("probes") or []))
    return {"push": c["push"], "weigh": c["weigh"], "squeeze": c["squeeze"], "cases_with_probes": sum(1 for r in rows if r.get("probes")), "cases": len(rows)}


def _mean_ci(values: np.ndarray, inv: np.ndarray, n_objects: int, rng: np.random.Generator, n_boot: int = N_BOOT) -> list[float]:
    W = object_weights(n_objects, n_boot, rng)
    S, N = np.bincount(inv, weights=values, minlength=n_objects), np.bincount(inv, minlength=n_objects).astype(float)
    m = (W @ S) / np.maximum(W @ N, 1e-12)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def _counted(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The cases the metrics count: no error and no artifact flag."""
    return [r for r in rows if "error" not in r and not r.get("artifact")]


def _excluded(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """(errors, artifact cases without an error)."""
    return sum("error" in r for r in rows), sum("error" not in r and bool(r.get("artifact")) for r in rows)


def realized_cost(rows: list[dict[str, Any]], rng: np.random.Generator, n_boot: int = N_BOOT) -> dict[str, Any]:
    """The mean realized cost over every case (errors and artifact cases excepted and counted), its
    object-bootstrap 95 % interval, and the mean by how the case ended."""
    ok = _counted(rows)
    n_err, n_art = _excluded(rows)
    if not ok:
        return {"n": 0, "n_errors": n_err, "n_artifacts": n_art, "mean": None, "ci95_object_bootstrap": None, "by_end": {}}
    c = np.array([float(r["cost"]) for r in ok])
    _, inv = np.unique([r["object"] for r in ok], return_inverse=True)
    by_end: dict[str, list[float]] = defaultdict(list)
    for r, v in zip(ok, c):
        by_end[str(r.get("ended"))].append(float(v))
    return {"n": len(ok), "n_errors": n_err, "n_artifacts": n_art, "mean": float(c.mean()),
            "ci95_object_bootstrap": _mean_ci(c, inv, int(inv.max()) + 1, rng, n_boot),
            "probe_part": float(np.mean([sum(p["cost"] for p in r.get("probes") or []) for r in ok])),
            "abstain_part": float(np.mean([float(r.get("abstain_part") or 0.0) for r in ok])),
            "by_end": {e: {"n": len(v), "mean": float(np.mean(v))} for e, v in sorted(by_end.items())}}


def _rescale(P: dict[str, float], q: float) -> Prediction:
    """P(placed_stable) set to q; the other classes rescaled to sum to 1 - q (spread over the motion classes when
    they all have probability 0: no damage class gets mass the model did not give it)."""
    rest = float(sum(float(P.get(k, 0.0)) for k in OUTCOMES[1:]))
    out = {"placed_stable": float(q)}
    for k in OUTCOMES[1:]:
        if rest > 1e-12:
            out[k] = float(P.get(k, 0.0)) * (1.0 - q) / rest
        else:
            out[k] = (1.0 - q) / (len(MOTION_OUTCOMES) - 1) if k in MOTION_OUTCOMES else 0.0
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
    ok = _counted(rows)
    n_err, n_art = _excluded(rows)
    first = [_attempt_row(r, 0) for r in ok if r.get("attempts")]
    second = [_attempt_row(r, 1) for r in ok if len(r.get("attempts") or []) > 1]
    b1 = calibration_block(first, rng, coverage_sims)
    rec = [r["recommendation"] for r in ok if r.get("recommendation")]
    out = {"protocol": protocol, "n": len(rows), "n_executed": len(first),
           "recommendation_decline_rate": float(np.mean([x == "decline" for x in rec])) if rec else None,
           "recommendations": dict(Counter(rec)), "abstained": sum(1 for r in ok if r.get("ended") == "abstained"),
           "errors": n_err, "artifacts": n_art, "ended": dict(Counter(str(r.get("ended")) for r in ok)),
           "irreversible": irreversible_block(first, rng),
           "first_attempt": b1, "second_attempt": calibration_block(second, rng) if protocol == "repair" else None}
    out.update({k: b1[k] for k in ("ece", "ece_ci95_object_bootstrap", "brier", "brier_const", "by_subgroup", "by_destination",
                                   "by_option", "outcomes", "per_class", "ece_iid_floor")})
    out.update(probe_kind=probe_kind_block(ok), thrown=thrown_block(ok), probes_run=probes_run(ok), realized_cost=realized_cost(rows, rng),
               guard=guard_block(ok))
    if nested and first and protocol != "random_row":
        out["nested"] = calibration_block(first, rng, preds=nested_isotonic(first))
    out["rows"] = rows
    return out


def compare(A: dict[str, Any], B: dict[str, Any], rng: np.random.Generator, n_boot: int = N_BOOT) -> dict[str, Any]:
    """A − B over the paired cases (same object and idx, counted in both: no error, no artifact): the mean
    realized cost and the first attempts' ECE on P(placed_stable), each with a 95 % interval that resamples
    objects jointly for both. The two runs must use the same costs (care scale, abstain cost; files from before
    these fields read as care normal and no abstain cost)."""
    costs_a, costs_b = ((R.get("care_scale", 1.0), R.get("abstain_cost")) for R in (A, B))
    if costs_a != costs_b:
        raise ValueError(f"the runs use different costs (care_scale, abstain_cost): a {costs_a}, b {costs_b}")
    ka = {(r["object"], r["idx"]): r for r in _counted(A["rows"])}
    kb = {(r["object"], r["idx"]): r for r in _counted(B["rows"])}
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
            "artifacts": {"a": _excluded(A["rows"])[1], "b": _excluded(B["rows"])[1]},
            "realized_cost": {"a": float(ca.mean()), "b": float(cb.mean()), "diff": float((ca - cb).mean()),
                              "ci95_object_bootstrap": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]},
            "ece_first_attempt": {"a": ea, "b": eb, "n_a": na, "n_b": nb, "diff": None if ea is None or eb is None else ea - eb,
                                  "ci95_object_bootstrap": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}}


# ---------------------------------------------------------------- CLI

def _write(out: dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=float)


def _care_arg(text: str) -> str | float:
    """--care: low, normal, high or a number above 0."""
    care = text if text in ("low", "normal", "high") else _number(text)
    try:
        care_scale(care)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return care


def _abstain_arg(text: str) -> str | float:
    """--abstain-cost: a finite number above 0, or none (the milestone 2a rule)."""
    if text == "none":
        return text
    x = _number(text)
    if not (isinstance(x, float) and np.isfinite(x) and x > 0):
        raise argparse.ArgumentTypeError(f"abstain cost must be a number above 0 or none, not {text!r}")
    return x


def _noise_arg(text: str) -> float:
    """--force-noise: a finite number at least 0."""
    x = _number(text)
    if not (isinstance(x, float) and np.isfinite(x) and x >= 0):
        raise argparse.ArgumentTypeError(f"force noise must be a finite number at least 0, not {text!r}")
    return x


def _number(text: str) -> str | float:
    try:
        return float(text)
    except ValueError:
        return text


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
    ap.add_argument("--care", type=_care_arg, default=CARE, help="low, normal, high or a number above 0 (scales the irreversible costs)")
    ap.add_argument("--abstain-cost", type=_abstain_arg, default=None,
                    help="the cost of a case that ends without the object placed: a number, or none (the 2a rule); "
                         "default: the bundle's decision")
    ap.add_argument("--force-noise", type=_noise_arg, default=None,
                    help="the sd of the force readings' error (0: clean); default: each bundle's declared sensor")
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
               care=a.care, care_scale=care_scale(a.care), abstain_cost=run_abstain_cost(a, plan), force_noise=run_force_noise(rows),
               evoi=run_evoi(rows),
               source={"crossfit": a.crossfit, "bundle": a.bundle, "metrics": a.metrics}, seconds=round(time.time() - t0, 1))
    _write(out, a.out)
    summary = {k: out[k] for k in ("protocol", "kind", "care", "abstain_cost", "n", "n_executed", "abstained", "errors", "artifacts", "ece",
                                   "ece_ci95_object_bootstrap", "brier", "brier_const", "ece_iid_floor", "outcomes", "probes_run",
                                   "probe_kind", "thrown")}
    summary.update(realized_cost={k: out["realized_cost"][k] for k in ("n", "mean", "ci95_object_bootstrap")},
                   acceptance=out["first_attempt"]["acceptance"], seconds=out["seconds"])
    out["wandb"] = tracking.log_run("calibrate", {"protocol": a.protocol, "seed": a.seed, "repeats": a.repeats, "source": out["source"],
                                                  "n_objects": len(plan["objects"]), "threshold": a.threshold,
                                                  "care": a.care, "abstain_cost": out["abstain_cost"]}, summary)
    print(json.dumps(summary, indent=1, default=float))
    return out


if __name__ == "__main__":
    main()
    sys.exit(0)
