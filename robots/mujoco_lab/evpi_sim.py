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


"""Experiment 1, step 7: the simulator check of the EVPI per hidden property (daily-logs
researches/property-belief-manipulation/experiments/2026-11-05-evpi-decomposition, "Step 7"; the coordinator's plan of
2026-11-05 with its five changes).

evpi_props's outcome model chose a_prior and a_S and also scored them (J at the truth). Here the simulator scores them.
Sample (``draw_sample``): up to PER_CELL θ-critical records (evpi_props.groups) per cell (CELLS: κ30 A3 and κ100 A10
under both priors), a simple random sample with SAMPLE_SEED from the run's counted records, drawn before any simulation.
Per record, a_prior and every a_S that differs from it run from the case's reset at θ_true, through the skill, at
N_SEEDS reset seeds that every action of the case shares (paired, ``seeds_of``): the case's own reset seed, then
calibrate._stream k SEED_STREAM. The reset seed sets the sensor noise streams only.
  decline, unreachable  A, no simulation;
  a row                 pick_and_place with its id; the realized cost is particle_costs at the realized outcome
                        (``row_cost``: calibrate.attempt_cost, plus A unless placed_stable);
  a probe               pick_and_place with its id and the choosing belief in the store; the skill's next menu under that
                        belief's posterior, no second probe (calibrate._without_probe), then decline (A) or the best row
                        in the simulator; a thrown or damaged object ends the case (``probe_case_cost``), as
                        calibrate.run_protocol counts a case.
A row runs once per case, action key and seed and is scored under every record that chose it; a probe runs per record
and belief, because its continuation follows that belief and setting (``jobs_of``). Each record's beliefs and first
menus are rebuilt first with evpi_props's functions and streams (``replay_case``), and each simulation's first menu again:
an action key that differs from the record's stops the run (the replay gate, ``stop_reason``). While a belief with S
revealed runs, the skill's prior is that prior with S revealed (``revealed_prior``), so a refresh after a probe keeps
the reveal, and the store holds the belief without its reveal entries (``store_belief``). A case whose replay or reset
raises becomes a plan with an ``error``: all its pairs drop.
The summary (``summarize``; it refuses a sampled record that did not run or failed the replay), per cell and S, on all
records and on those that pass evpi_props.ess_ok, over the records whose a_S differs from a_prior (``unchanged`` counts
the others; they are not simulated, and for a row their realized difference is 0 by construction):
  realized  the primary quantity: the mean over the records of cost(a_prior) - cost(a_S) (each record: the mean over its
            kept seeds), with its object-bootstrap interval; read for S = all only (``reading``: one reading per cell);
            the single properties are descriptive; no multiplicity correction;
  gap       realized - EVPI_S: the realized minus predicted gain on cases selected for a predicted gain (it includes the
            selection on a predicted gain, so it is not a calibration test);
  level     the mean J(a; θ_true) and the mean realized cost of a_prior and of a_S on the same records.
A pair with a probe on either side gets its own rows and no reading; a (record, seed) pair one of whose actions ends as an
artifact or an error drops from both, and a cell whose drops pass MAX_DROP gets counts only; so does a cell and S with
fewer than MIN_RECORDS records. A counts-only block shows no mean and no interval. The drops and the outcome counts come
for both record sets.

    python -P -m robots.mujoco_lab.evpi_sim sample --dirs D1,D2,... --out FILE
    python -P -m robots.mujoco_lab.evpi_sim run --sample FILE --out DIR [--smoke 5] [--objects O,...]
    python -P -m robots.mujoco_lab.evpi_sim project --sample FILE --dirs D
    python -P -m robots.mujoco_lab.evpi_sim summary --sample FILE --dirs D1,D2,... --out DIR [--wandb]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Callable

import numpy as np

from robots.mujoco_lab import calibrate as cal
from robots.mujoco_lab import evpi_props as ev
from robots.mujoco_lab import tracking
from robots.mujoco_lab.com_gap import _mean_ci

CELLS = (("k30_A3", "class"), ("k30_A3", "generic"), ("k100_A10", "class"), ("k100_A10", "generic"))
PER_CELL = 50
SAMPLE_SEED = 20261106
N_SEEDS = 5
SEED_STREAM = 8                           # calibrate._stream k (1 menus, 2 random row, 3 and 4 com_gap, 6 reveals, 7 truth)
MIN_RECORDS = 10                          # a cell and S with fewer records gets counts only
MAX_DROP = 0.10                           # a cell whose (record, seed) pairs drop more than this share gets counts only
MAIN = "all"                              # the main reading's reveal; the single properties are descriptive
N_BOOT = 2000
POSITIVE = "the realized gain from revealing S is positive in this cell"
WORSE = "revealing S makes the realized cost worse in this cell"
NONE = "no detectable realized gain"
COUNTS = "counts only"
SECONDARY = "realized minus predicted gain on cases selected for a predicted gain"


# ---------------------------------------------------------------- the seeds and the sample

def seeds_of(seed: int, o: str, idx: int, n: int = N_SEEDS) -> list[int]:
    """The case's reset seeds: its own (seed x 1000 + idx, the run's reset), then n - 1 from calibrate._stream k
    SEED_STREAM."""
    extra = cal._stream(seed, o, idx, SEED_STREAM).integers(2 ** 31, size=n - 1)
    return [int(seed) * 1000 + int(idx)] + [int(x) for x in extra]


def draw_sample(recs: list[dict[str, Any]], per_cell: int = PER_CELL, seed: int = SAMPLE_SEED) -> list[dict[str, Any]]:
    """Up to ``per_cell`` θ-critical counted records per cell, without replacement, each cell's records in (object, idx)
    order and its own stream ([seed, cell index]); every record when the cell has no more."""
    out = []
    for i, cell in enumerate(CELLS):
        rs = sorted((r for r in ev._counted(recs) if (r["setting"], r["prior"]) == cell and ev.groups(r)["critical"]),
                    key=lambda r: (r["object"], r["idx"]))
        if len(rs) > per_cell:
            pick = np.sort(np.random.default_rng([int(seed), i]).choice(len(rs), int(per_cell), replace=False))
            rs = [rs[int(j)] for j in pick]
        out += rs
    return out


def smoke_order(sample: list[dict[str, Any]], seed: int = SAMPLE_SEED + 1) -> list[dict[str, Any]]:
    """The sample in the smoke test's order: each cell's records shuffled (its own stream), then the cells in turn."""
    cells = []
    for i, cell in enumerate(CELLS):
        rs = [r for r in sample if (r["setting"], r["prior"]) == cell]
        cells.append([rs[int(j)] for j in np.random.default_rng([int(seed), i]).permutation(len(rs))])
    return [c[j] for j in range(max((len(c) for c in cells), default=0)) for c in cells if j < len(c)]


# ---------------------------------------------------------------- the realized costs

def row_cost(label: str, cost_s: float, care: float, A: float) -> float:
    """An executed row: particle_costs at the realized outcome (calibrate.attempt_cost, absorbing, plus A unless the
    object is placed)."""
    return cal.attempt_cost(label, float(cost_s), care) + (0.0 if label == "placed_stable" else float(A))


def probe_case_cost(x: dict[str, Any], care: float, A: float) -> float:
    """An executed probe and how its case ended: calibrate.probe_cost (its time, a tipped push, a thrown or damaged
    object), then the row its belief picked (``row_cost``), or A when no row ran (thrown, damaged, abstained, unreachable,
    no_row)."""
    p = x["probe"]
    c = cal.probe_cost(p["probe"], p["kind"], p["thrown"], care, x["time_scale"])
    if x["end"] == "row":
        return c + row_cost(x["cont"]["label"], x["cont"]["cost_s"], care, A)
    return c + float(A)


# ---------------------------------------------------------------- the simulations of a case

def job_key(prior: str, setting: str, key: str, b: dict[str, Any]) -> tuple | None:
    """The simulation that scores belief ``key``'s choice ``b``: none for an abstain, one per action key for a row, one
    per prior, setting and belief for a probe."""
    if b["kind"] == "abstain":
        return None
    return ("row", b["action"]) if b["kind"] != "probe" else ("probe", prior, setting, key)


def jobs_of(plans: list[dict[str, Any]]) -> dict[tuple, tuple[str, str, str, dict[str, Any]]]:
    """A case's simulations, each with the (prior, setting, belief, choice) that runs it: the first that needs it."""
    jobs: dict[tuple, tuple[str, str, str, dict[str, Any]]] = {}
    for p in plans:
        for key, b in p["beliefs"].items():
            jk = job_key(p["prior"], p["setting"], key, b)
            if jk is not None:
                jobs.setdefault(jk, (p["prior"], p["setting"], key, b))
    return jobs


def _store(env: Any, belief: Any) -> dict[str, Any]:
    """A fresh store that holds ``belief`` for this episode (as evpi_props.first_menu builds it)."""
    oid = env.locate("object")["object_id"]
    return {"beliefs": {oid: {"episode": int(getattr(env, "episode", 0)), "belief": belief, "probes_run": set(), "probe_seq": [],
                              "attempted": False}}}


def _menu(store: dict[str, Any]) -> dict[str, Any]:
    cache = list(store.get("pick_place", {}).values())
    if len(cache) != 1:
        raise RuntimeError(f"{len(cache)} cached assessments (one expected)")
    return cache[0]["menu"]


def store_belief(b: Any) -> Any:
    """The belief the store holds: its history without the reveal entries. A refresh (tools.update_belief) replays every
    history entry with its used terms, and a reveal entry has none; the revealed prior carries the reveal instead
    (``revealed_prior``). Nothing in the menu path reads the history, and the replay gate checks every first menu."""
    return b._replace(history=[e for e in b.history if e.get("type") != "reveal"])


def revealed_prior(fn: Callable, S: str, truth: dict[str, Any], entry: dict[str, Any], thresholds: Any,
                   pivot_length: float) -> Callable:
    """The prior a belief with S revealed refreshes from (``primitives.prior`` while its simulations run): fn's draw with
    S revealed as evpi_props.reveal reveals it, at equal weights (belief.refresh ignores a draw's weights) and with no
    reveal entry; fn itself for the prior's belief. Under fn a probe that leaves fewer than MIN_ESS effective particles
    would refresh without the reveal (S = all is one particle, so its probes always refresh)."""
    if S == ev.PRIOR:
        return fn

    def prior(pose: dict[str, Any], n: int, rng: np.random.Generator, **kw: Any) -> Any:
        b, _ = ev.reveal(fn(pose, n, rng, **kw), S, truth, entry, thresholds, lambda m, r: fn(pose, m, r, **kw), rng, pivot_length)
        if not np.allclose(b.weights, b.weights[0]):
            b = b._replace(particles=b.resample(int(n), rng), weights=np.ones(int(n)))
        return store_belief(b)

    return prior


def replay_case(prims: Any, o: str, idx: int, recs: list[dict[str, Any]], s: dict[str, Any], priors: dict[str, Callable],
                world: Any, all_keys: bool = False) -> dict[str, Any]:
    """The case as evpi_props.run_case builds it (its reset, truth, prior draws on calibrate._stream k 1, reveals on
    REVEAL_STREAM in REVEALS order, first menus on the stream after the prior's draw), and per record its plan: each
    needed belief's choice, action key, kind and row time (a_prior and the a_S that differ; every belief with
    ``all_keys``), with ``mismatch`` naming the beliefs whose choice or action key differs from the record's. Returns
    the plans, the beliefs' contexts, their revealed priors (``prior_of``), the destination and ``reset(seed)``."""
    from robots.mujoco_lab.adapter import object_props
    from robots.mujoco_lab.com_gap import fingerprint
    from robots.mujoco_lab.fragility import entry_of
    from rpent.skills.pick_place.tools import _particles, _spread_kw

    t0 = time.time()
    env, seed = prims.env, int(s["seed"])
    theta, com_end, destination, hidden = cal.case_draw(seed, o, idx, object_props(o)["mass"], False, fragility=world)
    if world is not None:
        env.slosh_zeta, env.slosh_scale = world[1].slosh_zeta, world[1].slosh_scale
    env.classes = None if world is None else world[0]
    fixture = destination if destination in cal.MESH_IDS else None

    def reset(reset_seed: int) -> None:
        env.reset(o, theta, seed=int(reset_seed), com_end=com_end, fixture=fixture, fill=hidden["fill"], frag=hidden["frag"])

    reset(seed * 1000 + idx)
    fp = fingerprint(env)
    pose = env.pose_info()
    entry = entry_of(o, world[0])[0] if world is not None else {"class": "none", "fillable": False}
    th = None if world is None else world[1]
    truth = ev.truth_particle(pose, theta, com_end, hidden, entry)
    n, spread = _particles(prims), _spread_kw(prims)
    plans, ctx, prior_of = [], {}, {}
    L = float(pose["pivot"]["L"])
    base = prims.prior
    try:
        for pname in sorted({r["prior"] for r in recs}):
            fn = priors[pname]
            prims.prior = fn
            g = cal._stream(seed, o, idx, 1)
            b0 = fn(pose, n, g, **spread)
            state = copy.deepcopy(g.bit_generator.state)
            rr = cal._stream(seed, o, idx, ev.REVEAL_STREAM)

            def draw(m: int, r: np.random.Generator, _fn=fn) -> Any:
                return _fn(pose, m, r, **spread)

            beliefs = {ev.PRIOR: b0}
            for S in ev.REVEALS:
                beliefs[S] = ev.reveal(b0, S, truth, entry, th, draw, rr, L)[0]
            for r in (r for r in recs if r["prior"] == pname):
                care, A = float(r["care"]), float(r["abstain_cost"])
                keys = [ev.PRIOR] + [S for S in ev.REVEALS if all_keys or r["action"][S] != r["action"][ev.PRIOR]]
                plan = {"object": o, "idx": idx, "prior": pname, "setting": r["setting"], "beliefs": {}, "mismatch": [],
                        "fingerprint_ok": fp == r.get("fingerprint"), "n_particles_ok": n == r.get("n_particles", n)}
                for key in keys:
                    m = ev.first_menu(prims, beliefs[key], destination, care, A, state)
                    c = ev.choice_of(m["payload"])
                    opt = m["menu"].get(c)
                    kind = "abstain" if c in ev.ABSTAINS else (opt.kind if opt is not None else "none")
                    plan["beliefs"][key] = {"choice": c, "action": ev.action_key(c, m["menu"]), "kind": kind,
                                            "cost_s": float(opt.cost_s) if opt is not None and kind != "probe" else None}
                    if c != r["choice"][key] or plan["beliefs"][key]["action"] != r["action"][key]:
                        plan["mismatch"].append(key)
                    ctx[(pname, r["setting"], key)] = (beliefs[key], care, A, state)
                    prior_of[(pname, key)] = revealed_prior(fn, key, truth, entry, th, L)
                if not (plan["fingerprint_ok"] and plan["n_particles_ok"]):
                    plan["mismatch"].append("case")
                plans.append(plan)
    finally:
        prims.prior = base
    return {"plans": plans, "ctx": ctx, "prior_of": prior_of, "destination": destination, "reset": reset, "fingerprint": fp,
            "seconds": round(time.time() - t0, 2)}


def execute(prims: Any, belief: Any, destination: str, care: float, A: float, state: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """One simulation from the lab's reset: the first menu of ``belief`` (it must give ``b``'s choice and action key),
    then the row, or the probe, its next menu under the posterior, no second probe, and the row it picks."""
    from rpent.skills.pick_place.probes import DAMAGED
    from rpent.skills.pick_place.tools import assess_pick, pick_and_place

    t0 = time.time()
    env = prims.env
    store = _store(env, belief)
    prims.abstain_cost, prims.rng = float(A), ev._rng_at(state)
    x: dict[str, Any] = {"time_scale": getattr(prims, "probe_time_scale", 1.0), "artifact": False}

    def done(**kw) -> dict[str, Any]:
        return dict(x, seconds=round(time.time() - t0, 2), **kw)

    def call(**kw) -> dict[str, Any]:
        try:
            return pick_and_place(object="object", destination=destination, care=care, primitives=prims, store=store, **kw).data
        except Exception as exc:  # noqa: BLE001  a simulator fault ends this simulation as an error
            return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}

    a = assess_pick(object="object", destination=destination, care=care, primitives=prims, store=store).data
    c = ev.choice_of(a)
    menu = _menu(store) if "error" not in a else {}
    x["gate"] = c == b["choice"] and ev.action_key(c, menu) == b["action"]
    if not x["gate"]:
        return done(error="the first menu differs from the replay's")
    if menu[c].kind != "probe":
        d = call(option_id=c)
        if "outcome" not in d:
            return done(error=str(d.get("error") or d.get("reason")))
        return done(end="row", label=d["outcome"], artifact=bool(d.get("artifact")), cost_s=float(menu[c].cost_s))
    d = call(option_id=c)
    if "error" in d:
        return done(error=str(d["error"]))
    x["probe"] = {"probe": d["probed"], "kind": d["kind"], "thrown": bool(d["thrown"]), "damage": d.get("damage"),
                  "artifact": bool(d.get("artifact"))}
    x["artifact"] = x["probe"]["artifact"]
    if d["kind"] == DAMAGED or d.get("damage"):
        return done(end="damaged")
    if d["thrown"]:
        return done(end="thrown")
    a2 = d["assessment"]
    if "error" in a2:
        return done(error=str(a2["error"]))
    if not a2.get("assessed", False):
        return done(end="unreachable")
    rec = a2.get("recommendation")
    if rec == "probe":
        rec = cal._without_probe(a2, prims)
    if rec == "decline":
        return done(end="abstained")
    row = next((o for o in a2["options"] if o["id"] == a2.get("best_option")), None)
    if row is None:
        return done(end="no_row")
    menu2 = _menu(store)
    opt = menu2.get(row["id"])
    d2 = call(option_id=row["id"])
    if "outcome" not in d2:
        return done(error=str(d2.get("error") or d2.get("reason")))
    x["artifact"] = x["artifact"] or bool(d2.get("artifact"))
    return done(end="row", cont={"option": row["id"], "action": ev.action_key(row["id"], menu2), "kind": row["kind"],
                                 "cost_s": float(opt.cost_s if opt is not None else row["cost_s"]), "label": d2["outcome"],
                                 "artifact": bool(d2.get("artifact"))})


def stop_reason(o: str, idx: int, runs: list[dict[str, Any]]) -> str | None:
    """The replay gate on the simulations: a first menu that differs from the replay's stops the run."""
    bad = next((x for x in runs if x.get("gate") is False), None)
    if bad is None:
        return None
    return f"simulation gate: {o} {idx}: the first menu differs from the replay's (seed {bad.get('seed_k')}, job {bad.get('job')})"


def sim_case(prims: Any, o: str, idx: int, recs: list[dict[str, Any]], s: dict[str, Any], priors: dict[str, Callable],
             world: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    """One case: the replay, then (when every plan passes the gate) each of its simulations at every seed, under the
    belief's revealed prior and with the store's belief (``store_belief``). Returns (plans, runs, stop): ``stop`` is the
    simulation gate's reason (``stop_reason``), and the case ends at that simulation."""
    from robots.mujoco_lab.com_gap import fingerprint

    rc = replay_case(prims, o, idx, recs, s, priors, world)
    plans = [dict(p, replay_seconds=rc["seconds"]) for p in rc["plans"]]
    if any(p["mismatch"] for p in plans):
        return plans, [], None
    runs: list[dict[str, Any]] = []
    base = prims.prior
    try:
        for k, reset_seed in enumerate(seeds_of(int(s["seed"]), o, idx)):
            for jk, (pname, setting, key, b) in jobs_of(plans).items():
                belief, care, A, state = rc["ctx"][(pname, setting, key)]
                prims.prior = rc["prior_of"][(pname, key)]
                t0, fp_ok = time.time(), None
                try:
                    rc["reset"](reset_seed)
                    fp_ok = fingerprint(prims.env) == rc["fingerprint"]
                    x = execute(prims, store_belief(belief), rc["destination"], care, A, state, b)
                except Exception as exc:  # noqa: BLE001  one bad simulation must not stop the case
                    x = {"gate": None, "artifact": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                         "seconds": round(time.time() - t0, 2)}
                runs.append(dict(x, object=o, idx=idx, job=list(jk), seed_k=k, reset_seed=reset_seed, prior=pname,
                                 setting=setting, key=key, fingerprint_ok=fp_ok, wall_seconds=round(time.time() - t0, 2)))
                stop = stop_reason(o, idx, runs[-1:])
                if stop:
                    return plans, runs, stop
    finally:
        prims.prior = base
    return plans, runs, None


def run_object(job: tuple[str, list[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    """One object's sampled records on its fold's bundle (evpi_props.run_object's sensor, world and priors)."""
    from robots.mujoco_lab.prior import fragility_world
    from robots.mujoco_lab.vision_prior import shape_prior

    o, recs, s = job
    prims = cal._primitives(s["bundle"], None, None)
    base = prims.prior
    priors = {"class": base, "generic": shape_prior(base, {o: s["generic"][o]})}
    prims.force_noise = cal.job_force_noise(None, prims.manifest)
    prims.env.force_noise = prims.force_noise
    world = fragility_world(prims.manifest)
    if getattr(prims, "reset_s", None) is not None:
        raise RuntimeError("evpi_sim scores the absorbing menu only")
    out: dict[str, Any] = {"plans": [], "runs": [], "stop": None}
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        by[int(r["idx"])].append(r)
    try:
        for idx in sorted(by):
            t0 = time.time()
            try:
                plans, runs, stop = sim_case(prims, o, idx, by[idx], s, priors, world)
            except Exception as exc:  # noqa: BLE001  a case whose replay or reset raises drops all its pairs (summarize)
                err = f"{type(exc).__name__}: {str(exc)[:200]}"
                plans = [{"object": o, "idx": idx, "prior": r["prior"], "setting": r["setting"], "beliefs": {}, "mismatch": [],
                          "error": err, "replay_seconds": round(time.time() - t0, 2)} for r in by[idx]]
                runs, stop = [], None
            out["plans"] += plans
            out["runs"] += runs
            print(f"{o} {idx}: {time.time() - t0:.1f} s, {len(plans)} records, {len(runs)} simulations"
                  + (f", error {plans[0]['error']}" if plans and "error" in plans[0] else ""), file=sys.stderr, flush=True)
            bad = [(p["prior"], p["setting"], p["mismatch"]) for p in plans if p["mismatch"]]
            if bad:
                out["stop"] = f"replay gate: {o} {idx} {bad}"
                break
            if stop:
                out["stop"] = stop
                break
    finally:
        prims.prior = base
    return out


# ---------------------------------------------------------------- the run

def _sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def sample(a: argparse.Namespace) -> str:
    recs, cfg = ev.load(a.dirs.split(","))
    picked = draw_sample(recs, a.per_cell, a.sample_seed)
    keep = ("bundle", "scenes", "seed", "code_revision", "generator", "settings_parsed", "priors_parsed")
    objects = sorted({r["object"] for r in picked})
    out = {"config": dict({k: cfg.get(k) for k in keep}, generic={o: cfg["generic"][o] for o in objects}, run_dirs=a.dirs.split(",")),
           "cells": [list(c) for c in CELLS], "per_cell": a.per_cell, "sample_seed": a.sample_seed,
           "available": {f"{c[0]}/{c[1]}": sum(1 for r in ev._counted(recs) if (r["setting"], r["prior"]) == c and ev.groups(r)["critical"])
                         for c in CELLS},
           "records": picked}
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(json.dumps({"out": a.out, "available": out["available"], "sampled": dict(Counter(f"{r['setting']}/{r['prior']}" for r in picked)),
                      "objects": len(objects), "cases": len({(r['object'], r['idx']) for r in picked})}), flush=True)
    return a.out


def run(a: argparse.Namespace) -> str:
    from robots.mujoco_lab.estimation_check import _revision

    os.makedirs(a.out, exist_ok=True)
    paths = {k: os.path.join(a.out, f"{k}.jsonl") for k in ("plans", "runs")}
    if any(os.path.exists(p) for p in paths.values()):
        raise SystemExit(f"{a.out} holds a run: pick a new --out")
    smp = json.load(open(a.sample))
    cfg = smp["config"]
    recs = smoke_order(smp["records"])[: a.smoke] if a.smoke else smp["records"]
    if a.objects:
        keep = set(a.objects.split(","))
        recs = [r for r in recs if r["object"] in keep]
    folds = json.load(open(os.path.join(cfg["bundle"], "folds", "folds.json")))
    with open(os.path.join(a.out, "run_config.json"), "w") as f:
        json.dump({**vars(a), **_revision(), "sample_sha256": _sha(a.sample), "n_seeds": N_SEEDS, "seed_stream": SEED_STREAM,
                   "records": [[r["object"], r["idx"], r["prior"], r["setting"]] for r in recs]}, f, indent=1)
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        by[r["object"]].append(r)
    jobs = [(o, rs, {"bundle": os.path.join(cfg["bundle"], "folds", f"fold_{folds[o]}"), "seed": cfg["seed"],
                     "generic": {o: cfg["generic"][o]}}) for o, rs in sorted(by.items())]
    t0, stop = time.time(), None
    with mp.get_context("spawn").Pool(a.workers) as pool, open(paths["plans"], "w") as fp, open(paths["runs"], "w") as fr:
        for res in pool.imap_unordered(run_object, jobs):
            for p in res["plans"]:
                fp.write(json.dumps(p) + "\n")
            for x in res["runs"]:
                fr.write(json.dumps(x, default=str) + "\n")
            fp.flush()
            fr.flush()
            if res["stop"]:
                stop = res["stop"]
                pool.terminate()
                break
    if stop:
        raise SystemExit(f"stopped by the {stop}")
    print(f"done in {time.time() - t0:.0f} s: {a.out}", flush=True)
    return a.out


def load_runs(dirs: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """The plans and simulations of one step-7 run's shard dirs, which must share a clean code revision and the sample."""
    plans, runs, cfg = [], [], None
    for d in dirs:
        c = json.load(open(os.path.join(d, "run_config.json")))
        if c.get("dirty"):
            raise SystemExit(f"{d}: dirty code")
        if cfg is not None:
            bad = [k for k in ("code_revision", "sample_sha256") if c.get(k) != cfg.get(k)]
            if bad:
                raise SystemExit(f"{d}: run config differs in {bad}")
        cfg = cfg or c
        for name, into in (("plans", plans), ("runs", runs)):
            for line in open(os.path.join(d, f"{name}.jsonl")):
                if line.strip():
                    into.append(json.loads(line))
    return plans, runs, cfg or {}


def project(a: argparse.Namespace) -> dict[str, Any]:
    """The smoke test's report: the replay gate, the time per simulation (its reset and run, a raised one too) and per
    replay, and the full sample's projected CPU-h (its simulations counted from the records' choices: one per row action
    of a case, one per probe choice, N_SEEDS each). Without a probe simulation in the smoke test a probe is taken as two
    rows (a probe, its next menu and a row), and ``probe_seconds_assumed`` says so."""
    smp = json.load(open(a.sample))
    plans, runs, _ = load_runs(a.dirs.split(","))
    sims: Counter = Counter()
    cases = set()
    for r in smp["records"]:
        cases.add((r["object"], r["idx"]))
        for key in [ev.PRIOR] + [S for S in ev.REVEALS if r["action"][S] != r["action"][ev.PRIOR]]:
            c = r["choice"][key]
            if c in ev.ABSTAINS:
                continue
            sims[("probe", r["object"], r["idx"], r["prior"], r["setting"], key) if c.startswith("probe_")
                 else ("row", r["object"], r["idx"], r["action"][key])] += 1
    kinds = Counter(k[0] for k in sims)
    sec: dict[str, list[float]] = {"row": [], "probe": []}
    for x in runs:
        v = x.get("wall_seconds", x.get("seconds"))
        if v is not None and x["job"][0] in sec:
            sec[x["job"][0]].append(float(v))
    rep = {(p["object"], p["idx"]): float(p["replay_seconds"]) for p in plans if p.get("replay_seconds") is not None}
    mean = {k: float(np.mean(v)) if v else None for k, v in sec.items()}
    assumed = mean["probe"] is None and kinds["probe"] > 0
    if mean["probe"] is None and mean["row"] is not None:
        mean["probe"] = 2.0 * mean["row"]
    per_case = float(np.mean(list(rep.values()))) if rep else 0.0
    total = len(cases) * per_case + sum(N_SEEDS * kinds[k] * (mean[k] or 0.0) for k in kinds)
    out = {"records": len(plans), "mismatch": [[p["object"], p["idx"], p["prior"], p["setting"], p["mismatch"]] for p in plans if p.get("mismatch")],
           "case_errors": sum(1 for p in plans if "error" in p),
           "simulations": len(runs), "gate_failures": sum(1 for x in runs if x.get("gate") is False),
           "errors": sum(1 for x in runs if "error" in x), "artifacts": sum(1 for x in runs if x.get("artifact")),
           "fingerprint_differs": sum(1 for x in runs if x.get("fingerprint_ok") is False),
           "seconds_per_simulation": mean, "probe_seconds_assumed": assumed, "replay_seconds_per_case": per_case,
           "full": {"cases": len(cases), "row_simulations": N_SEEDS * kinds["row"], "probe_simulations": N_SEEDS * kinds["probe"],
                    "projected_cpu_h": total / 3600.0}}
    print(json.dumps(out, indent=1), flush=True)
    return out


# ---------------------------------------------------------------- the summary

def _cost(r: dict[str, Any], b: dict[str, Any], x: dict[str, Any] | None) -> float:
    care, A = float(r["care"]), float(r["abstain_cost"])
    if b["kind"] == "abstain":
        return A
    if b["kind"] == "probe":
        return probe_case_cost(x, care, A)
    return row_cost(x["label"], b["cost_s"], care, A)


def _ok(b: dict[str, Any], x: dict[str, Any] | None) -> bool:
    return b["kind"] == "abstain" or (x is not None and "error" not in x and not x.get("artifact"))


def record_pairs(r: dict[str, Any], plan: dict[str, Any], runs: dict[tuple, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per reveal S whose action differs from a_prior's: the (record, seed) pairs, a pair kept when both actions ran
    without an error or an artifact (``drops`` by the action that failed), and over the kept seeds the mean realized
    difference ``d`` = cost(a_prior) - cost(a_S), the mean costs ``rp`` and ``rs``; ``probe_pair`` when a probe is on
    either side. ``runs``: the case's simulations by (job key, seed). A plan with an ``error`` (its case raised) drops
    every pair of the record's changed reveals from both actions."""
    if "error" in plan:
        return {S: {"kept": 0, "drops": {"prior": 0, "S": 0, "both": N_SEEDS}, "d": None, "rp": None, "rs": None,
                    "probe_pair": any(str(r["choice"][k]).startswith("probe_") for k in (ev.PRIOR, S))}
                for S in ev.REVEALS if r["action"][S] != r["action"][ev.PRIOR]}
    bs = plan["beliefs"]
    b0 = bs[ev.PRIOR]

    def sim(key: str, k: int) -> dict[str, Any] | None:
        jk = job_key(plan["prior"], plan["setting"], key, bs[key])
        return None if jk is None else runs.get((jk, k))

    out = {}
    for S in (S for S in ev.REVEALS if S in bs):
        drops = {"prior": 0, "S": 0, "both": 0}
        d, rp, rs = [], [], []
        for k in range(N_SEEDS):
            x0, xs = sim(ev.PRIOR, k), sim(S, k)
            ok0, oks = _ok(b0, x0), _ok(bs[S], xs)
            if not (ok0 and oks):
                drops["both" if not (ok0 or oks) else ("prior" if not ok0 else "S")] += 1
                continue
            c0, cs = _cost(r, b0, x0), _cost(r, bs[S], xs)
            d.append(c0 - cs)
            rp.append(c0)
            rs.append(cs)
        out[S] = {"kept": len(d), "drops": drops, "probe_pair": "probe" in (b0["kind"], bs[S]["kind"]),
                  "d": float(np.mean(d)) if d else None, "rp": float(np.mean(rp)) if rp else None,
                  "rs": float(np.mean(rs)) if rs else None}
    return out


def reading(ci: list[float] | None, n: int, drop_share: float) -> str:
    """The main reading of the mean realized gain's interval (counts only below MIN_RECORDS records or above MAX_DROP
    dropped pairs)."""
    if ci is None or n < MIN_RECORDS or drop_share > MAX_DROP:
        return COUNTS
    if ci[0] > 0:
        return POSITIVE
    if ci[1] < 0:
        return WORSE
    return NONE


def _token(b: dict[str, Any], x: dict[str, Any] | None) -> str:
    """One simulation's outcome for the counts."""
    if b["kind"] == "abstain":
        return b["choice"]
    if x is None:
        return "missing"
    if "error" in x:
        return "error"
    if x.get("artifact"):
        return "artifact"
    if b["kind"] != "probe":
        return x["label"]
    return f"{x['probe']['probe']}:{x['end']}" + (f":{x['cont']['label']}" if x["end"] == "row" else "")


def _counts(items: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
            by_case: dict[tuple, dict[tuple, dict[str, Any]]]) -> dict[str, Any]:
    """The outcome per belief and seed (``_token``), and after a probe the continuation the simulator ran and the picks
    the model scored."""
    tok: dict[str, Counter] = defaultdict(Counter)
    picks_j: dict[str, Counter] = defaultdict(Counter)
    picks_s: dict[str, Counter] = defaultdict(Counter)
    for r, p, _ in items:
        sims = by_case[(r["object"], r["idx"])]
        for key, b in p["beliefs"].items():
            jk = job_key(p["prior"], p["setting"], key, b)
            for k in range(N_SEEDS):
                x = None if jk is None else sims.get((jk, k))
                tok[key][_token(b, x)] += 1
                if b["kind"] == "probe" and x is not None and "error" not in x:
                    picks_s[key][x["cont"]["option"] if x.get("end") == "row" else x.get("end")] += 1
            if b["kind"] == "probe":
                picks_j[key].update((r.get("probe_detail") or {}).get(key, {}).get("picks") or {})
    return {"outcomes": {k: dict(v) for k, v in tok.items()},
            "continuation_model": {k: dict(v) for k, v in picks_j.items()},
            "continuation_simulator": {k: dict(v) for k, v in picks_s.items()}}


def summarize(sample_recs: list[dict[str, Any]], plans: list[dict[str, Any]], runs: list[dict[str, Any]], n_boot: int = N_BOOT,
              seed: int = 0) -> dict[str, Any]:
    """The tables, drops and outcome counts per record set ("all", "ess_ok"), cell and S (module docstring). A sampled
    record with no plan, or whose replay differed, stops the summary (SystemExit)."""
    rng = np.random.default_rng(seed)
    pl = {(p["object"], p["idx"], p["prior"], p["setting"]): p for p in plans}
    by_case: dict[tuple, dict[tuple, dict[str, Any]]] = defaultdict(dict)
    for x in runs:
        by_case[(x["object"], x["idx"])][(tuple(x["job"]), int(x["seed_k"]))] = x
    rows: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    missing, mismatch = [], []
    for r in sample_recs:
        p = pl.get((r["object"], r["idx"], r["prior"], r["setting"]))
        if p is None:
            missing.append([r["object"], r["idx"], r["prior"], r["setting"]])
            continue
        if p["mismatch"]:
            mismatch.append([r["object"], r["idx"], r["prior"], r["setting"], p["mismatch"]])
            continue
        rows[(r["setting"], r["prior"])].append((r, p, record_pairs(r, p, by_case[(r["object"], r["idx"])])))
    if missing:
        raise SystemExit(f"{len(missing)} sampled records not run, for example {missing[:3]}")
    if mismatch:
        raise SystemExit(f"{len(mismatch)} sampled records failed the replay, for example {mismatch[:3]}")

    def block(items: list[tuple[dict[str, Any], dict[str, Any]]], S: str, main: bool, share: float) -> dict[str, Any]:
        objs = [r["object"] for r, _ in items]
        n = len(items)
        only = n < MIN_RECORDS or share > MAX_DROP

        def ci(v: list[float]) -> dict[str, Any]:
            return _mean_ci(np.array(v, dtype=float), objs, rng, n_boot)

        b: dict[str, Any] = {"n": n, "n_objects": len(set(objs)), "counts_only": only, "realized": None, "predicted": None,
                             "gap": None, "level": None}
        if not only:
            b.update(realized=ci([q["d"] for _, q in items]), predicted=ci([r["evpi"][S] for r, _ in items]),
                     gap=ci([q["d"] - r["evpi"][S] for r, q in items]),
                     level={"J_prior": ci([r["J"][ev.PRIOR] for r, _ in items]), "realized_prior": ci([q["rp"] for _, q in items]),
                            "J_S": ci([r["J"][S] for r, _ in items]), "realized_S": ci([q["rs"] for _, q in items])})
        if not main:
            b["reading"] = None
        elif S == MAIN:
            b["reading"] = reading(None if only else b["realized"]["ci"], n, share)
        else:
            b["reading"] = COUNTS if only else "descriptive"
        return b

    tables: dict[str, dict[str, dict[str, Any]]] = {"all": {}, "ess_ok": {}}
    drops: dict[str, dict[str, dict[str, Any]]] = {"all": defaultdict(dict), "ess_ok": defaultdict(dict)}
    counts: dict[str, dict[str, dict[str, Any]]] = {"all": defaultdict(dict), "ess_ok": defaultdict(dict)}
    for (setting, prior), items in rows.items():
        for subset in ("all", "ess_ok"):
            its = [t for t in items if subset == "all" or ev.ess_ok(t[0])]
            pairs = [q for _, _, qs in its for q in qs.values()]
            total = N_SEEDS * len(pairs)
            by_action = {k: sum(q["drops"][k] for q in pairs) for k in ("prior", "S", "both")}
            share = sum(by_action.values()) / total if total else 0.0
            cell: dict[str, Any] = {}
            for S in ev.REVEALS:
                kept = [(r, qs[S]) for r, _, qs in its if S in qs and qs[S]["kept"] > 0]
                cell[S] = {"main": block([t for t in kept if not t[1]["probe_pair"]], S, True, share),
                           "probe": block([t for t in kept if t[1]["probe_pair"]], S, False, share),
                           "records": sum(1 for _, _, qs in its if S in qs),
                           "dropped_records": sum(1 for _, _, qs in its if S in qs and qs[S]["kept"] == 0),
                           "unchanged": sum(1 for r, _, _ in its if r["action"][S] == r["action"][ev.PRIOR])}
            tables[subset].setdefault(setting, {})[prior] = cell
            drops[subset][setting][prior] = {"pairs": total, "by_action": by_action, "share": share, "counts_only": share > MAX_DROP,
                                             "case_errors": sum(1 for _, p, _ in its if "error" in p),
                                             "probe_pairs": sum(1 for q in pairs if q["probe_pair"])}
            counts[subset][setting][prior] = _counts(its, by_case)
    labels: dict[tuple, set] = defaultdict(set)
    n_valid: Counter = Counter()
    for x in runs:
        if x["job"][0] == "row" and "error" not in x and not x.get("artifact"):
            labels[(x["object"], x["idx"], tuple(x["job"]))].add(x["label"])
            n_valid[(x["object"], x["idx"], tuple(x["job"]))] += 1
    multi = [k for k in labels if n_valid[k] >= 2]
    return {"tables": tables, "drops": {k: dict(v) for k, v in drops.items()},
            "probe_pairs": {st: {pr: d["probe_pairs"] for pr, d in v.items()} for st, v in drops["all"].items()},
            "counts": {k: dict(v) for k, v in counts.items()},
            "rows_label_changes": {"n": len(multi), "changed": sum(1 for k in multi if len(labels[k]) > 1)},
            "gates": {"records": len(sample_recs), "missing": missing, "mismatch": mismatch,
                      "case_errors": sum(1 for p in plans if "error" in p),
                      "simulation_gate_failures": sum(1 for x in runs if x.get("gate") is False),
                      "fingerprint_differs": sum(1 for x in runs if x.get("fingerprint_ok") is False),
                      "errors": sum(1 for x in runs if "error" in x), "artifacts": sum(1 for x in runs if x.get("artifact"))},
            "n_main_readings": len(CELLS), "main": MAIN, "min_records": MIN_RECORDS, "max_drop": MAX_DROP, "n_seeds": N_SEEDS}


def _f(c: dict[str, Any] | None) -> str:
    if not c or c.get("mean") is None:
        return "–"
    return f"{c['mean']:+.3f}" + (f" [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]" if c.get("ci") else "")


def markdown(s: dict[str, Any]) -> str:
    out = [f"# Experiment 1, step 7: the simulator check ({s['n_seeds']} paired seeds per case)", "",
           f"Main reading: S = {s['main']}, non-probe pairs, one per cell ({s['n_main_readings']} readings); the single "
           "properties are descriptive; no multiplicity correction. The secondary quantity is the " + SECONDARY + ": it "
           "includes the selection on a predicted gain, so it is not a calibration test. a_S is the one-step rule after "
           "the reveal. Each table is over the records whose a_S differs from a_prior; \"unchanged\" counts the others. "
           f"A block with fewer than {s['min_records']} records, or in a cell whose pairs drop more than "
           f"{s['max_drop']:.0%}, shows counts only.", ""]
    for subset, title in (("all", "all records"), ("ess_ok", f"records with ESS >= {ev.ESS_TABLE:g}")):
        t = s["tables"][subset]
        out += [f"## {title}", "", "### Main reading (S = all, non-probe pairs)", "",
                "| cell | prior | records (objects) | unchanged | realized gain [95 %] | reading | predicted EVPI | " + SECONDARY + " |",
                "|---|---|---|---|---|---|---|---|"]
        for setting, prior in CELLS:
            c = (t.get(setting) or {}).get(prior)
            if c is None:
                continue
            m = c[MAIN]["main"]
            out.append(f"| {setting} | {prior} | {m['n']} ({m['n_objects']}) | {c[MAIN]['unchanged']} | {_f(m['realized'])} | "
                       f"{m['reading']} | {_f(m['predicted'])} | {_f(m['gap'])} |")
        out += ["", "### Single properties (descriptive), non-probe pairs", "",
                "| cell | prior | S | records | unchanged | realized gain | predicted EVPI | " + SECONDARY + " |",
                "|---|---|---|---|---|---|---|---|"]
        for setting, prior in CELLS:
            c = (t.get(setting) or {}).get(prior)
            for S in (ev.SINGLE if c else ()):
                m = c[S]["main"]
                out.append(f"| {setting} | {prior} | {S} | {m['n']} | {c[S]['unchanged']} | "
                           + (f"{COUNTS} | | |" if m["counts_only"] else f"{_f(m['realized'])} | {_f(m['predicted'])} | {_f(m['gap'])} |"))
        out += ["", "### Probe pairs (a probe on either side: their own rows, not read)", "",
                "| cell | prior | S | records | realized gain | predicted EVPI |", "|---|---|---|---|---|---|"]
        for setting, prior in CELLS:
            c = (t.get(setting) or {}).get(prior)
            for S in (ev.REVEALS if c else ()):
                m = c[S]["probe"]
                if m["n"]:
                    out.append(f"| {setting} | {prior} | {S} | {m['n']} | "
                               + (f"{COUNTS} | |" if m["counts_only"] else f"{_f(m['realized'])} | {_f(m['predicted'])} |"))
        out += ["", "### Level check (same records): J(a; θ_true) and the realized cost", "",
                "| cell | prior | S | pairs | records | J(a_prior) | realized(a_prior) | J(a_S) | realized(a_S) |",
                "|---|---|---|---|---|---|---|---|---|"]
        for setting, prior in CELLS:
            c = (t.get(setting) or {}).get(prior)
            for S in (ev.REVEALS if c else ()):
                for part in ("main", "probe"):
                    m = c[S][part]
                    if not m["n"]:
                        continue
                    lv = m["level"]
                    out.append(f"| {setting} | {prior} | {S} | {part} | {m['n']} | "
                               + (f"{COUNTS} | | | |" if lv is None else
                                  f"{_f(lv['J_prior'])} | {_f(lv['realized_prior'])} | {_f(lv['J_S'])} | {_f(lv['realized_S'])} |"))
        out += ["", "### Dropped pairs (an artifact or an error on either side)", "",
                "| cell | prior | pairs | a_prior failed | a_S failed | both | case errors | share | counts only | probe pairs |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for setting, prior in CELLS:
            d = (s["drops"][subset].get(setting) or {}).get(prior)
            if d:
                ba = d["by_action"]
                out.append(f"| {setting} | {prior} | {d['pairs']} | {ba['prior']} | {ba['S']} | {ba['both']} | {d['case_errors']} | "
                           f"{d['share']:.3f} | {'yes' if d['counts_only'] else 'no'} | {d['probe_pairs']} |")
        out += ["", "### Outcome counts per seed", ""]
        for setting, prior in CELLS:
            c = (s["counts"][subset].get(setting) or {}).get(prior)
            if not c:
                continue
            out.append(f"- {setting} / {prior}:")
            for key, v in c["outcomes"].items():
                out.append(f"  - {key}: " + ", ".join(f"{k} {n}" for k, n in sorted(v.items(), key=lambda kv: -kv[1])))
            for key in c["continuation_simulator"]:
                out.append(f"  - continuation after the probe of {key}: simulator " + json.dumps(c["continuation_simulator"][key], sort_keys=True)
                           + ", model " + json.dumps(c["continuation_model"].get(key, {}), sort_keys=True))
        out.append("")
    rl = s["rows_label_changes"]
    g = s["gates"]
    out += [f"Rows whose label changes across the seeds: {rl['changed']} of {rl['n']} (a row simulated once per case, "
            "action and seed).",
            f"Gates: {g['records']} sampled records, {len(g['missing'])} not run, {len(g['mismatch'])} replay mismatches, "
            f"{g['case_errors']} records whose case raised, {g['simulation_gate_failures']} simulations whose first menu "
            f"differed, {g['fingerprint_differs']} resets whose scene differed, {g['errors']} errors, {g['artifacts']} artifacts."]
    return "\n".join(out) + "\n"


def summary(a: argparse.Namespace) -> dict[str, Any]:
    smp = json.load(open(a.sample))
    plans, runs, cfg = load_runs(a.dirs.split(","))
    if cfg.get("sample_sha256") != _sha(a.sample):
        raise SystemExit("the runs used another sample file")
    out = summarize(smp["records"], plans, runs, a.n_boot, a.seed)
    os.makedirs(a.out, exist_ok=True)
    md = markdown(out)
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write(md)
    print(md, flush=True)
    if a.wandb:
        conf = {"sample": a.sample, "sample_sha256": cfg.get("sample_sha256"), "code_revision": cfg.get("code_revision"),
                "n_seeds": N_SEEDS, "seed_stream": SEED_STREAM, "cells": [list(c) for c in CELLS], "per_cell": smp.get("per_cell"),
                "sample_seed": smp.get("sample_seed"), "evpi_run": smp["config"]}
        flat: dict[str, Any] = {}
        for subset, t in out["tables"].items():
            for setting, block in t.items():
                for prior, c in block.items():
                    for S in ev.REVEALS:
                        for part in ("main", "probe"):
                            m = c[S][part]
                            pre = f"{subset}/{setting}/{prior}/{S}/{part}"
                            flat[f"{pre}/n"] = m["n"]
                            flat[f"{pre}/unchanged"] = c[S]["unchanged"]
                            if m["counts_only"]:
                                continue
                            ci = m["realized"]["ci"]
                            flat[f"{pre}/realized"] = m["realized"]["mean"]
                            flat[f"{pre}/realized_lo"] = None if ci is None else ci[0]
                            flat[f"{pre}/realized_hi"] = None if ci is None else ci[1]
                            flat[f"{pre}/predicted"] = m["predicted"]["mean"]
                            flat[f"{pre}/gap"] = m["gap"]["mean"]
        for subset, t in out["drops"].items():
            for setting, block in t.items():
                for prior, d in block.items():
                    flat[f"drops/{subset}/{setting}/{prior}/share"] = d["share"]
                    flat[f"drops/{subset}/{setting}/{prior}/case_errors"] = d["case_errors"]
        flat.update({f"gates/{k}": (len(v) if isinstance(v, list) else v) for k, v in out["gates"].items()})
        flat["rows_label_changes"] = out["rows_label_changes"]["changed"]
        out["wandb"] = tracking.log_run("evpi_sim", conf, flat)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def main(argv: list[str] | None = None) -> Any:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--dirs", required=True, help="the evpi_props run's shard dirs, comma-separated")
    p.add_argument("--out", required=True, help="the sample file (json)")
    p.add_argument("--per-cell", type=int, default=PER_CELL)
    p.add_argument("--sample-seed", type=int, default=SAMPLE_SEED)
    r = sub.add_parser("run")
    r.add_argument("--sample", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--smoke", type=int, default=None, help="only the first N records of the smoke order")
    r.add_argument("--objects", default=None)
    r.add_argument("--workers", type=int, default=1)
    j = sub.add_parser("project")
    j.add_argument("--sample", required=True)
    j.add_argument("--dirs", required=True, help="the smoke run's dir")
    s = sub.add_parser("summary")
    s.add_argument("--sample", required=True)
    s.add_argument("--dirs", required=True, help="the run's shard dirs, comma-separated")
    s.add_argument("--out", required=True)
    s.add_argument("--n-boot", type=int, default=N_BOOT)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--wandb", action="store_true")
    a = ap.parse_args(argv)
    return {"sample": sample, "run": run, "project": project, "summary": summary}[a.cmd](a)


if __name__ == "__main__":
    main()
