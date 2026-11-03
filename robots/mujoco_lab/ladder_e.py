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

"""E, the receding-horizon reference planner of the help ladder (item 1, 2026-11-04; no language model).

At each real decision point E starts from the skill's actual menu and belief (they already hold every reading) at the
true setting (kappa, A, absorbing), and scores every row the agents may choose (every row but e1.HIDDEN_ROWS, decline at
A) with a lookahead of H attempts. It runs the row with the lowest score, and replans at the next real menu.

* H counts attempts. H = 1 is the menu's own prices: an attempt row's one-step expected cost (``Option._costs``, every
  unplaced outcome priced as ending the case at A) and a probe row's own expected cost. At H >= 2 a recoverable outcome
  (tipped, missed, dropped_low, not_lifted) leads to an imagined menu with one attempt fewer: the same rows and the same
  per-particle predictions (the menu's own "menu unchanged" convention), the belief updated by the outcome's probability
  alone (an imagined step has no readouts), decline at A. An irreversible outcome ends the case at 5 kappa + A. The
  horizon never passes the attempt cap: min(H, 3 - attempts so far).
* An imagined choice is the bundle's probe-value estimator applied to the attempt rows: "heldout" (the m2b2 bundles)
  scores each particle with the row chosen by the belief without that particle's draw (menu.heldout_rows), so a lookahead
  finds no value in a chance correlation; "plugin" scores the chosen row on the particles that chose it.
* Probes keep the lab's rules (only before the first attempt; at most the push and then the weigh): E scores the probe
  rows the menu offers. A probe row's score is its time and push-tip charge (the menu's) plus the expected cost after it
  (menu.probe_value's preposterior, the same simulated results), with the continuation at H attempts and every later
  choice made under the belief given the result (``probe_after``). A later probe is not imagined after an imagined one;
  the real loop offers it at the next real menu.

E is a reference for regret: its first-decision scores are never called optimal. The first decision of E at H = 1 is
the lowest-cost row of the true menu; it differs from the one-step rule (oracle_rule) only where the rule declines on a
guard flag, where a probe row is cheapest by less than the rule's EVOI_Z margin, or at a tie.

The run (``run``): per case and setting, the default rule's first choice (its own menu, e1_gap.rule_config("default"))
and E's run at each horizon on the true menu (rule_config("oracle"), a fresh reset each; e1.planner_loop with the agents'
caps). Each run of E at H scores every decision at the horizons 1 to H, so its first decision holds the tables the
summary compares. The summary (``summarize``): E at H = 1 against the oracle rule's first choice (every mismatch with its
reason), the default rule's first-decision regret at each horizon's table (an object-bootstrap 95 % interval and e1_gap's
n50), how often H 2 and 3 change E's choice (the first decision, and every decision of the longest run), every case
where a policy's first choice scores below E's on E's own table, every case where the lowest score rises with H, and
the wall time per case. A case whose reset did not reach rest is left out of the metrics (as the gap check does), not of
the H = 1 check.

    python -P -m robots.mujoco_lab.ladder_e run --cases FILE --crossfit DIR [--settings ...] [--horizons 1,2,3] --workers 8 --out DIR
    python -P -m robots.mujoco_lab.ladder_e summary --dir DIR [--wandb]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from rpent.skills.pick_place.belief import ParticleBelief
from rpent.skills.pick_place.costs import COST, TIME_COST_PER_S, care_scale, cost_vector, prediction_matrix
from rpent.skills.pick_place.features import model_names, particle_matrix
from rpent.skills.pick_place.menu import EVOI_EPS, EVOI_K, HELDOUT_REST, estimate
from rpent.skills.pick_place.probes import DAMAGED
from rpent.skills.pick_place.protocol import OUTCOMES, RECOVERABLE

from robots.mujoco_lab import e1, e1_gap

ESTIMATORS = ("plugin", "heldout")
ATTEMPT_KINDS = ("commit", "lower_risk")
TOL = 1e-9                # a score within this of the lowest ties with it (ladder.TOL); the menu's order breaks a tie
CHUNK = 8                 # simulated probe results scored per batch (memory: about 100 MB per batch at H = 3, 2000 particles)
PLACED = OUTCOMES.index("placed_stable")
REC = [OUTCOMES.index(k) for k in RECOVERABLE]


@dataclass
class AttemptRows:
    """The attempt rows of one menu for the lookahead: per-particle outcome probabilities ``P`` (rows x particles x
    OUTCOMES), each row's seconds, the care and A. ``C1`` is each row's one-step per-particle cost (costs.particle_costs:
    every unplaced outcome ends the case at A), ``c_imm`` the part a lookahead keeps (C1 less A x P(recoverable)) and
    ``P_rec`` the recoverable outcomes' probabilities (rows x RECOVERABLE x particles). ``C1`` given replaces the formula
    (the menu's own numbers)."""

    ids: list[str]
    P: np.ndarray
    cost_s: np.ndarray
    care: float | str
    abstain_cost: float
    C1: np.ndarray | None = None
    c_imm: np.ndarray = field(init=False)
    P_rec: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.P = np.asarray(self.P, dtype=float)
        self.cost_s = np.asarray(self.cost_s, dtype=float).reshape(-1)
        A = float(self.abstain_cost)
        if self.C1 is None:
            self.C1 = (self.P @ cost_vector(self.care) + TIME_COST_PER_S * self.cost_s[:, None]
                       + A * (1.0 - self.P[..., PLACED]))
        self.C1 = np.asarray(self.C1, dtype=float)
        self.P_rec = np.moveaxis(self.P[..., REC], -1, 1)
        self.c_imm = self.C1 - A * self.P_rec.sum(axis=1)

    @staticmethod
    def from_options(options: list[Any], belief: ParticleBelief, model: Any, care: str | float, abstain_cost: float,
                     tol: float = TOL) -> AttemptRows:
        """A live menu's attempt rows: each row's per-particle predictions from its own inputs (menu.estimate, which
        draws nothing), C1 the rows' own ``_costs``. A ValueError when those predictions do not give the rows' costs at
        (care, abstain_cost): the menu is priced at another setting."""
        P = np.stack([prediction_matrix(estimate(model, belief, o._base, o._pose, o.action["force"], o._planned)[1])
                      for o in options])
        own = np.vstack([np.asarray(o._costs, dtype=float) for o in options])
        rows = AttemptRows([o.id for o in options], P, [o.cost_s for o in options], care, abstain_cost)
        gap = float(np.max(np.abs(rows.C1 - own)))
        if gap > tol * max(1.0, float(np.max(np.abs(own)))):
            raise ValueError(f"the menu's row costs are not this setting's (care {care}, A {abstain_cost}): off by {gap:.3g}")
        return AttemptRows(rows.ids, P, rows.cost_s, care, abstain_cost, C1=own)


@dataclass
class Groups:
    """The held-out groups of a belief's particles (the ``ancestor`` key: copies of one draw are held out together, as
    menu.heldout_rows); none when every particle is its own draw."""

    inv: np.ndarray | None = None
    order: np.ndarray | None = None
    starts: np.ndarray | None = None

    @staticmethod
    def of(groups: Any) -> Groups:
        if groups is None:
            return Groups()
        g = np.asarray(groups).reshape(-1)
        if len(np.unique(g)) == len(g):
            return Groups()
        inv = np.unique(g, return_inverse=True)[1].reshape(-1)
        order = np.argsort(inv, kind="stable")
        counts = np.bincount(inv)
        return Groups(inv, order, np.concatenate([[0], np.cumsum(counts)[:-1]]))

    def sum_back(self, X: np.ndarray) -> np.ndarray:
        """Per particle, the sum of X (beliefs x particles) over its group."""
        if self.inv is None:
            return X
        return np.add.reduceat(X[:, self.order], self.starts, axis=1)[:, self.inv]


def _estimator(estimator: str) -> str:
    if estimator not in ESTIMATORS:
        raise ValueError(f"unknown estimator {estimator!r}; known: {', '.join(ESTIMATORS)}")
    return estimator


def choose(C: np.ndarray, U: np.ndarray, estimator: str, groups: Groups, fallback: np.ndarray | None = None) -> np.ndarray:
    """Per belief b and particle l, the cost under l of the row chosen at belief U[b] among the rows C[b] (beliefs x rows x
    particles): "plugin" the row cheapest under U[b]; "heldout" the row cheapest under U[b] without l's group
    (menu.heldout_rows), or row ``fallback[b]`` (default: the plug-in row) where the rest of U[b] is (numerically) 0."""
    _estimator(estimator)
    B, R, N = C.shape
    S = np.einsum("brn,bn->br", C, U)
    best = np.argmin(S, axis=1)
    if estimator == "plugin":
        return np.array(C[np.arange(B), best])
    fall = best if fallback is None else np.broadcast_to(np.asarray(fallback, dtype=int), (B,))
    total = U.sum(axis=1, keepdims=True)
    rest = total - groups.sum_back(U)
    low = np.full((B, N), np.inf)
    K = np.empty((B, N))
    for r in range(R):
        val = S[:, r, None] - groups.sum_back(U * C[:, r])
        better = val < low
        low = np.where(better, val, low)
        K = np.where(better, C[:, r], K)
    return np.where(rest > HELDOUT_REST * total, K, C[np.arange(B)[:, None], fall[:, None], np.arange(N)[None, :]])


def node_costs(rows: AttemptRows, V: np.ndarray, h: int, estimator: str, groups: Groups | None = None) -> np.ndarray:
    """Per belief V[b] (beliefs x particles, any scale) with ``h`` attempts left: each attempt row's per-particle cost
    with the lookahead, and decline (A) as the last row (beliefs x rows + 1 x particles). At h = 1 the rows' C1; at h >= 2
    a recoverable outcome o of row r costs COST[o] and then the cost under each particle of the action chosen at the
    belief V[b] x P_r(o) with h - 1 attempts left."""
    groups = groups or Groups()
    V = np.atleast_2d(np.asarray(V, dtype=float))
    B, N = V.shape
    A = float(rows.abstain_cost)
    R = len(rows.ids)
    if h <= 1:
        C = np.vstack([rows.C1, np.full((1, N), A)])
        return np.broadcast_to(C, (B,) + C.shape)
    children = (V[:, None, None, :] * rows.P_rec[None]).reshape(-1, N)
    K = choose(node_costs(rows, children, h - 1, estimator, groups), children, estimator, groups)
    out = np.empty((B, R + 1, N))
    out[:, :R] = rows.c_imm[None] + np.einsum("ron,bron->brn", rows.P_rec, K.reshape(B, R, len(REC), N))
    out[:, R] = A
    return out


def probe_after(belief: Any, cont_fn: Callable[[np.ndarray], np.ndarray], probe_model: Any, X_probe: np.ndarray, force: float,
                care: str | float, abstain_cost: float, rng: np.random.Generator, k: int = EVOI_K, estimator: str = "heldout",
                chunk: int = CHUNK) -> dict[str, Any]:
    """menu.probe_value (absorbing, with an abstain cost A) with a continuation that follows the result: ``cont_fn(W)``
    gives, per belief W[b], every continuation row's per-particle cost and decline's (beliefs x rows + 1 x particles), its
    later choices made under that belief. Row b is the cheapest at the belief now; a result's gain is the saving of the plan
    chosen under the belief given the result (and the object kept) over row b's plan under the belief now. The same
    simulated results and the same draws from ``rng`` as probe_value: with a ``cont_fn`` that ignores its argument the two
    agree. Returns evoi, evoi_se, expected_cost_after (and plugin_evoi under "heldout")."""
    _estimator(estimator)
    w = belief.weights
    C0 = np.array(cont_fn(w[None, :])[0])
    best = int(np.argmin(C0 @ w))
    risk = np.asarray(probe_model.terminal_risk(X_probe), dtype=float)
    P = np.asarray(probe_model.p_kind(X_probe), dtype=float)
    kept = P.copy()
    if DAMAGED in probe_model.classes:
        kept[:, list(probe_model.classes).index(DAMAGED)] = 0.0
    total = kept.sum(axis=1, keepdims=True)
    Q = np.divide(kept, total, out=np.zeros_like(kept), where=total > 0)
    a = belief.arrays()
    keep = 1.0 - risk
    terminal = COST["dropped_high"] * care_scale(care) + float(abstain_cost)
    unused = float(w @ (risk * terminal + keep * C0[best]))
    groups = Groups.of(a.get("ancestor"))
    W, J = [], []
    for j, i in enumerate(rng.choice(len(w), size=int(k), p=w)):
        if not total[i, 0] > 0:
            continue
        obs = probe_model.sample(X_probe[i], belief.particles[i], force, rng, p=Q[i])
        ll, _ = probe_model.loglik(obs, X_probe, a, force, P=Q)
        W.append(belief.reweight(ll).weights * keep)
        J.append(j)
    gains, held = np.zeros(int(k)), np.zeros(int(k))
    for s in range(0, len(W), int(chunk)):
        Wc = np.asarray(W[s:s + int(chunk)])
        Ck = np.asarray(cont_fn(Wc))
        pick = np.argmin(np.einsum("brn,bn->br", Ck, Wc), axis=1)
        Kh = choose(Ck, Wc, "heldout", groups, fallback=np.full(len(Wc), best)) if estimator == "heldout" else None
        for b, j in enumerate(J[s:s + int(chunk)]):
            gains[j] = float(np.dot(Wc[b], C0[best]) - np.dot(Wc[b], Ck[b, pick[b]]))
            if Kh is not None:
                held[j] = float(Wc[b] @ (C0[best] - Kh[b]))
    mean = lambda g: float(np.mean(g)) if len(g) else 0.0  # noqa: E731
    err = lambda g: float(np.std(g, ddof=1) / np.sqrt(len(g))) if len(g) > 1 else 0.0  # noqa: E731
    gain = mean(gains)
    if estimator == "heldout":
        return {"estimator": estimator, "evoi": mean(held), "evoi_se": err(held), "expected_cost_after": unused - mean(held),
                "plugin_evoi": gain if gain > EVOI_EPS else 0.0}
    return {"estimator": estimator, "evoi": gain if gain > EVOI_EPS else 0.0, "evoi_se": err(gains),
            "expected_cost_after": unused - gain}


# ---------------------------------------------------------------- E on a live menu

def lowest(table: dict[str, float], order: list[str]) -> str:
    """The row with the lowest score; among scores within TOL of it, the first in ``order`` (the menu's)."""
    lo = min(table.values())
    return next(i for i in order if i in table and table[i] - lo <= TOL)


def decide(entry: dict[str, Any], prims: Any, ids: list[str], horizons: tuple[int, ...], care: str | float, abstain_cost: float,
           attempts_done: int, seed: tuple[int, ...], max_attempts: int = e1.MAX_ATTEMPTS, chunk: int = CHUNK) -> dict[str, Any]:
    """E's scores of the rows ``ids`` (the agents' list, decline included) of one live menu (``entry``: the skill tools'
    cached {"menu": {id: Option}, "belief"}) at each horizon in ``horizons``, the effective horizon min(H, max_attempts -
    attempts_done). At an effective horizon of 1 every score is the menu's own (decline A). Each probe row's results are
    drawn from its own stream (``seed``, the row id): the same draws at every horizon. Returns {"order", "h", "tables",
    "probes" (each probe's value, its error, its cost after it and its charge, per horizon above 1), "seconds"}."""
    menu, belief = entry["menu"], entry["belief"]
    if not isinstance(belief, ParticleBelief):
        raise ValueError(f"E needs a particle belief, not {type(belief).__name__}")
    missing = [i for i in ids if i != "decline" and i not in menu]
    if missing:
        raise ValueError(f"rows not on the menu: {missing}")
    left = int(max_attempts) - int(attempts_done)
    if left < 1:
        raise ValueError(f"no attempt left ({attempts_done} of {max_attempts} done)")
    shown = [o for o in menu.values() if o.id in ids and o.kind != "decline"]
    attempt = [o for o in shown if o.kind in ATTEMPT_KINDS]
    probes = [o for o in shown if o.kind == "probe"]
    A = float(abstain_cost)
    own = {o.id: float(o.expected_cost) for o in shown}
    own["decline"] = A
    est = getattr(prims, "evoi", "plugin")
    rows = AttemptRows.from_options(attempt, belief, prims.model, care, A) if attempt else None
    groups = Groups.of(belief.arrays().get("ancestor"))
    w = belief.weights
    out: dict[str, Any] = {"order": [o.id for o in shown] + ["decline"], "h": {}, "tables": {}, "probes": {}, "seconds": {}}
    for H in horizons:
        t0 = time.time()
        h = min(int(H), left)
        out["h"][H] = h
        if h <= 1 or rows is None:
            out["tables"][H] = dict(own)
        else:
            cont = lambda W, h=h: node_costs(rows, W, h, est, groups)  # noqa: E731
            C = cont(w[None])[0]
            table = {rid: float(C[r] @ w) for r, rid in enumerate(rows.ids)}
            pv = {}
            for o in probes:
                pm = prims.probe_models[o.action["probe"]]
                X = particle_matrix(o._base, belief.arrays(), o._pose, o.action["force"], model_names(pm))
                rng = np.random.default_rng([*map(int, seed), zlib.crc32(o.id.encode())])
                v = probe_after(belief, cont, pm, X, o.action["force"], care, A, rng, estimator=est, chunk=chunk)
                charge = float(o.expected_cost) - float(o.probe["expected_cost_after"])
                table[o.id] = charge + v["expected_cost_after"]
                pv[o.id] = {"evoi": v["evoi"], "evoi_se": v["evoi_se"], "expected_cost_after": v["expected_cost_after"],
                            "charge": charge}
            table["decline"] = A
            out["tables"][H] = table
            out["probes"][H] = pv
        out["seconds"][H] = round(time.time() - t0, 3)
    return out


def live_entry(store: dict[str, Any]) -> dict[str, Any]:
    """The skill tools' one cached menu of the case (a reassessment clears the cache and caches the fresh menu)."""
    cache = store.get("pick_place") or {}
    if len(cache) != 1:
        raise ValueError(f"the store holds {len(cache)} menus, not one")
    return next(iter(cache.values()))


class EPlanner:
    """e1.planner_loop's model: E at horizon ``H`` on the live menu of ``tools`` (its store and primitives), priced at
    (care, abstain_cost). Each call scores the listed rows at every horizon of ``horizons`` (default 1 to H) and runs
    the lowest score at H; ``log`` keeps every call's scores, effective horizons, probe values, choices per horizon and
    seconds. ``attempts`` counts the attempt rows it ran (the effective horizon's cap)."""

    def __init__(self, tools: Any, H: int, care: str | float, abstain_cost: float, seed: tuple[int, ...],
                 horizons: tuple[int, ...] | None = None, max_attempts: int = e1.MAX_ATTEMPTS, chunk: int = CHUNK):
        self.tools, self.H, self.care, self.A = tools, int(H), care, float(abstain_cost)
        self.horizons = tuple(sorted(set(horizons or range(1, int(H) + 1)) | {int(H)}))
        self.seed, self.max_attempts, self.chunk = tuple(int(x) for x in seed), int(max_attempts), int(chunk)
        self.attempts, self.log = 0, []

    def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        t0 = time.time()
        ids = list(tools[0]["function"]["parameters"]["properties"]["option_id"]["enum"])
        entry = live_entry(self.tools.store)
        d = decide(entry, self.tools.prims, ids, self.horizons, self.care, self.A, self.attempts, (*self.seed, len(self.log)),
                   self.max_attempts, self.chunk)
        choices = {H: lowest(t, d["order"]) for H, t in d["tables"].items()}
        choice = choices[self.H]
        self.log.append({"step": len(self.log), "attempts": self.attempts, "h": d["h"], "tables": d["tables"],
                         "probes": d["probes"], "choices": choices, "seconds": d["seconds"]})
        if choice != "decline" and entry["menu"][choice].kind in ATTEMPT_KINDS:
            self.attempts += 1
        return {"tool": "choose", "args": {"option_id": choice, "reason": f"E at H = {self.H}: the lowest score, "
                                           f"{d['tables'][self.H][choice]:.3f}"},
                "content": "", "reasoning": "", "seconds": round(time.time() - t0, 3), "finish_reason": "tool_calls", "usage": {},
                "error": None}


# ---------------------------------------------------------------- the run

def _set(prims: Any, conf: dict[str, Any]) -> None:
    prims.abstain_cost, prims.menu_recovery = conf["abstain_cost"], conf["menu_recovery"]
    prims.reset_s, prims.reset_price = conf["lab_recovery"]


def e_job(item: tuple[dict[str, Any], dict[str, Any]]) -> list[dict[str, Any]]:
    """One case under one setting: the default rule's first choice and E's run at each horizon (a row per horizon)."""
    from robots.mujoco_lab import calibrate, e1_expected, ladder

    (case, name), cfg = item
    setting = cfg["settings"][name]
    if e1.setting_recovery(setting)[0] is not None:
        raise ValueError(f"{name}: E prices the absorbing setting only")
    obj, idx, seed = case["object"], int(case["idx"]), int(case["seed"])
    prims = e1_expected._prims(cfg, obj)
    conf = e1_gap.rule_config("default", setting)
    _set(prims, conf)
    destination, reset_info, truth = e1.reset_case(prims, obj, idx, seed)
    try:
        a = e1.LabTools(prims, destination, conf["care"]).assess()
        ok = "error" not in a and a.get("assessed", True)
        default = {"assessed": bool(ok), "choice": ladder.rule_choice(a) if ok else None, "recommendation": a.get("recommendation"),
                   "guard_flag": bool((a.get("guard") or {}).get("flag")), "reason": None if ok else a.get("error") or a.get("reason")}
    except Exception as exc:  # noqa: BLE001  one bad case must not stop the run
        default = {"assessed": False, "choice": None, "error": f"{type(exc).__name__}: {exc}"[:300]}
    default["reset_at_rest"] = bool(reset_info.get("at_rest", True))
    conf = e1_gap.rule_config("oracle", setting)
    out = []
    for H in cfg["horizons"]:
        t0 = time.time()
        _set(prims, conf)
        destination, reset_info, truth = e1.reset_case(prims, obj, idx, seed)
        tools = e1.LabTools(prims, destination, conf["care"])
        planner = EPlanner(tools, H, conf["care"], conf["abstain_cost"], seed=(seed, zlib.crc32(obj.encode()), idx, zlib.crc32(name.encode())),
                           horizons=tuple(h for h in cfg["horizons"] if h <= H))
        rng = np.random.default_rng([seed, zlib.crc32(obj.encode()), idx, 7])
        try:
            rec = e1.planner_loop(tools, planner, "menu_planner", f"Please put the {e1.display_object(obj)} at the "
                                  f"{e1.display_destination(destination)}.", e1.display_object(obj), destination, rng)
        except Exception as exc:  # noqa: BLE001  one bad case must not stop the run
            rec = {"ended": "error", "error": f"{type(exc).__name__}: {exc}"[:300], "probes": [], "attempts": [], "decisions": []}
        artifact = calibrate.case_artifact(rec, reset_info)
        oracle = None if rec.get("recommendation") is None else ("decline" if rec["recommendation"] == "decline"
                                                                 else (rec.get("first_menu") or {}).get("recommended_option"))
        out.append({"object": obj, "idx": idx, "seed": seed, "setting": name, "H": int(H), "destination": destination, "truth": truth,
                    "reset": reset_info, "reset_artifact": not reset_info.get("at_rest", True), "artifact": bool(artifact),
                    "default": default,
                    "first": {"recommendation": rec.get("recommendation"), "best_option": rec.get("best_option"),
                              "oracle_choice": oracle, "guard_flag": bool((rec.get("guard") or {}).get("flag"))},
                    "ended": rec.get("ended"), "error": rec.get("error"), "probes": rec.get("probes"), "attempts": rec.get("attempts"),
                    "decisions": rec.get("decisions"), "e_log": planner.log, "cost": e1.realized_cost(rec, setting),
                    "e_seconds": round(sum(x["seconds"][H] for x in planner.log), 3), "seconds": round(time.time() - t0, 2)})
    prims.menu_recovery = None
    print(f"{obj} {idx} {name}: " + ", ".join(f"H{r['H']} {r['ended']} {r['e_seconds']:.0f}s/{r['seconds']:.0f}s" for r in out),
          file=sys.stderr, flush=True)
    return out


def _key(r: dict[str, Any]) -> tuple[str, int, int, str]:
    return r["object"], int(r["idx"]), int(r["seed"]), r["setting"]


def _pool(workers: int, max_tasks: int | None = None) -> Any:
    """The run's worker pool (spawn). ``max_tasks``: a worker is replaced after that many jobs, so its memory cannot grow
    without bound (a worker's caches grow by about 0.07 GB per job; a job's results do not depend on the worker's earlier
    jobs, so the replacement changes no result); None keeps each worker for the whole run."""
    return mp.get_context("spawn").Pool(workers, maxtasksperchild=max_tasks)


def run(cases: list[dict[str, Any]], settings: dict[str, dict[str, Any]], crossfit: str, horizons: tuple[int, ...], workers: int,
        out_dir: str, cases_path: str, max_tasks: int | None = None) -> str:
    """Every (case, setting) job into ``out_dir``/rows.jsonl (resuming: a job with every horizon's row is skipped; a directory
    of another bundle, generator, setting list or horizon list is an error); returns the path. ``max_tasks``: ``_pool``."""
    from robots.mujoco_lab.calibrate import code_revision
    from robots.mujoco_lab.corpus import lab_generator

    generator = e1.run_generator(out_dir)
    e1.check_bundle(out_dir, crossfit, generator)
    for held in sorted(p for p in os.listdir(out_dir) if p.startswith("run_config_")) if os.path.isdir(out_dir) else []:
        c = json.load(open(os.path.join(out_dir, held)))
        if c.get("settings") != settings or c.get("horizons") != list(horizons):
            raise ValueError(f"{out_dir} holds a run of other settings or horizons ({held}); give a new directory")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"run_config_{time.strftime('%Y%m%dT%H%M%S')}.json"), "w") as f:
        json.dump({"cases": cases_path, "n_cases": len(cases), "settings": settings, "horizons": list(horizons), "crossfit": crossfit,
                   "generator": lab_generator(), "workers": workers, "max_tasks_per_worker": max_tasks, "code": code_revision()},
                  f, indent=1)
    path = os.path.join(out_dir, "rows.jsonl")
    done: Counter = Counter()
    if os.path.exists(path):
        with open(path) as f:
            done = Counter(_key(json.loads(line)) for line in f if line.strip())
    cfg = {"crossfit": crossfit, "folds": json.load(open(os.path.join(crossfit, "folds.json"))), "settings": settings,
           "horizons": list(horizons)}
    todo = [((c, n), cfg) for c in cases for n in ([c["setting"]] if c.get("setting") else list(settings))
            if done[(c["object"], c["idx"], c["seed"], n)] < len(horizons)]
    print(f"ladder E: {len(todo)} (case, setting) jobs to run, horizons {list(horizons)}", file=sys.stderr, flush=True)
    with open(path, "a") as f:
        def write(parts) -> None:
            for part in parts:
                for r in part:
                    f.write(json.dumps(r, default=str) + "\n")
                f.flush()

        if workers <= 1:
            write(map(e_job, todo))
        else:
            with _pool(workers, max_tasks) as pool:
                write(pool.imap_unordered(e_job, todo))
    return path


# ---------------------------------------------------------------- the summary

def _stats(v: list[float]) -> dict[str, Any]:
    if not v:
        return {"n": 0, "mean": None, "p90": None, "max": None}
    a = np.asarray(v, dtype=float)
    return {"n": len(a), "mean": float(a.mean()), "p90": float(np.quantile(a, 0.9)), "max": float(a.max())}


def _regret(values: list[float], objects: list[str], name: str, n_boot: int) -> dict[str, Any]:
    out = e1_gap._interval(values, objects, name, n_boot)
    out["n50"] = e1_gap.power_n(values, objects) if values else None
    out["above_0"] = int(sum(v > TOL for v in values))
    return out


def summarize(rows: list[dict[str, Any]], n_boot: int = 2000) -> dict[str, Any]:
    """Per setting: E at H = 1 against the oracle rule's first choice; the default rule's (and the oracle rule's) first-decision
    regret on each horizon's table; how often a horizon above 1 changes E's choice; the policies scored below E on E's
    own table; the cases whose lowest score rises with H; the first tables' agreement across the runs; the wall times."""
    groups: dict[tuple, dict[int, dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(_key(r), {})[int(r["H"])] = r
    Hs = sorted({int(r["H"]) for r in rows})
    out: dict[str, Any] = {"horizons": Hs, "settings": {}}
    acc: dict[str, dict[str, Any]] = {}
    for key in sorted(groups):
        runs = groups[key]
        name = key[3]
        s = out["settings"].setdefault(name, {
            "cases": 0, "incomplete": 0, "no_menu": 0, "artifact": 0,
            "h1_vs_oracle": {"compared": 0, "match": 0, "mismatch": Counter(), "cases": []},
            "regret": {}, "oracle_regret": {}, "differ_first": {str(H): {"n": 0, "differ": 0, "pairs": Counter()} for H in Hs if H > 1},
            "differ_all": {str(H): {"n": 0, "differ": 0} for H in Hs if H > 1}, "below_e": [], "not_monotone": [],
            "consistency": {"compared": 0, "differ": []}, "wall": {}})
        a = acc.setdefault(name, {"regret": {H: ([], []) for H in Hs}, "oracle": {H: ([], []) for H in Hs},
                                  "e_s": {H: [] for H in Hs}, "case_s": {H: [] for H in Hs}})
        s["cases"] += 1
        tag = f"{key[0]}/{key[1]}"
        if set(runs) != set(Hs):
            s["incomplete"] += 1
            continue
        full = runs[Hs[-1]]
        if not full.get("e_log") or any(not runs[H].get("e_log") for H in Hs):
            s["no_menu"] += 1
            continue
        for H in Hs:
            a["e_s"][H].append(float(runs[H]["e_seconds"]))
            a["case_s"][H].append(float(runs[H]["seconds"]))
        first = full["e_log"][0]
        T = {H: first["tables"][str(H)] for H in Hs}
        s["consistency"]["compared"] += 1
        if any(runs[H]["e_log"][0]["tables"][str(h)] != T[h] for H in Hs for h in Hs if h <= H):
            s["consistency"]["differ"].append(tag)
        cE = {H: runs[H]["e_log"][0]["choices"][str(H)] for H in Hs}
        oc, dc = full["first"].get("oracle_choice"), (full.get("default") or {}).get("choice")
        if 1 in T and oc is not None:
            h1 = s["h1_vs_oracle"]
            h1["compared"] += 1
            if cE[1] == oc:
                h1["match"] += 1
            else:
                if oc in T[1] and abs(T[1][oc] - T[1][cE[1]]) <= TOL:
                    reason = "tie"
                elif oc == "decline" and full["first"].get("guard_flag"):
                    reason = "guard"
                elif cE[1].startswith("probe_") and not oc.startswith("probe_"):
                    reason = "probe_margin"
                else:
                    reason = "other"
                h1["mismatch"][reason] += 1
                h1["cases"].append({"case": tag, "reason": reason, "e": cE[1], "oracle": oc, "e_value": T[1][cE[1]],
                                    "oracle_value": T[1].get(oc)})
        if any(runs[H].get("reset_artifact") for H in Hs) or not (full.get("default") or {}).get("reset_at_rest", True):
            s["artifact"] += 1
            continue
        for H in Hs:
            lo = min(T[H].values())
            for pol, c, sink in (("default", dc, a["regret"]), ("oracle_rule", oc, a["oracle"])):
                if c in T[H]:
                    sink[H][0].append(T[H][c] - lo)
                    sink[H][1].append(key[0])
            pols = {"default": dc, "oracle_rule": oc, **{f"E{h}": cE[h] for h in Hs}}
            for pol, c in pols.items():
                if c in T[H] and T[H][c] < T[H][cE[H]] - TOL:
                    s["below_e"].append({"case": tag, "H": H, "policy": pol, "choice": c, "value": T[H][c], "e_choice": cE[H],
                                         "e_value": T[H][cE[H]]})
            if H > 1:
                d = s["differ_first"][str(H)]
                d["n"] += 1
                if cE[H] != cE[1]:
                    d["differ"] += 1
                    d["pairs"][f"{cE[1]} -> {cE[H]}"] += 1
        V = [min(T[H].values()) for H in Hs]
        for k in range(1, len(Hs)):
            if V[k] > V[k - 1] + TOL:
                s["not_monotone"].append({"case": tag, "H": Hs[k], "value": V[k], "below": V[k - 1]})
        for x in full["e_log"]:
            for H in Hs[1:]:
                if str(H) in x["choices"]:
                    s["differ_all"][str(H)]["n"] += 1
                    s["differ_all"][str(H)]["differ"] += x["choices"][str(H)] != x["choices"]["1"]
    for name, s in out["settings"].items():
        a = acc[name]
        for H in Hs:
            s["regret"][str(H)] = _regret(*a["regret"][H], f"{name}/default/{H}", n_boot)
            s["oracle_regret"][str(H)] = _regret(*a["oracle"][H], f"{name}/oracle/{H}", n_boot)
            s["wall"][str(H)] = {"e_seconds": _stats(a["e_s"][H]), "case_seconds": _stats(a["case_s"][H])}
        s["h1_vs_oracle"]["mismatch"] = dict(s["h1_vs_oracle"]["mismatch"])
        for d in s["differ_first"].values():
            d["pairs"] = dict(d["pairs"])
    return out


def _f(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:+.{nd}f}" if nd < 0 else f"{x:.{nd}f}"


def markdown(summ: dict[str, Any], title: str = "E: the receding-horizon reference planner") -> str:
    Hs = summ["horizons"]
    L = [f"# {title}", "",
         "E scores every row the agents may choose with a lookahead of H attempts on the true menu, and replans at each real "
         "menu. H = 1 is the menu's own prices. The default rule's regret is its first choice's score on E's table minus the "
         "lowest score of that table (mean, object-bootstrap 95 % interval; n50: e1_gap.power_n, the cases that show half of "
         "this gap at 80 % power). A case whose reset did not reach rest in any of its runs is left out of the regrets, the "
         "horizon changes and the two lists in Checks; the H = 1 comparison, the table agreement and the wall times count "
         "every case.", "",
         "## E at H = 1 against the oracle rule (first decision)", "",
         "| setting | compared | match | mismatch by reason |", "|---|---|---|---|"]
    for name, s in summ["settings"].items():
        h = s["h1_vs_oracle"]
        L.append(f"| {name} | {h['compared']} | {h['match']} | {json.dumps(h['mismatch']) if h['mismatch'] else '0'} |")
    for name, s in summ["settings"].items():
        for x in s["h1_vs_oracle"]["cases"]:
            L.append(f"- {name} {x['case']}: {x['reason']}: E {x['e']} ({_f(x['e_value'])}), oracle rule {x['oracle']} "
                     f"({_f(x['oracle_value'])})")
    L += ["", "## The default rule's first-decision regret against E", "",
          "| setting | H | cases (objects) | regret | 95 % interval | n50 | cases above 0 | oracle rule's regret |", "|---|---|---|---|---|---|---|---|"]
    for name, s in summ["settings"].items():
        for H in Hs:
            r, o = s["regret"][str(H)], s["oracle_regret"][str(H)]
            ci = "n/a" if r["mean"] is None else f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]"
            oc = "n/a" if o["mean"] is None else f"{o['mean']:+.3f} [{o['ci'][0]:+.3f}, {o['ci'][1]:+.3f}]"
            L.append(f"| {name} | {H} | {r['n']} ({r['objects']}) | {_f(r['mean'])} | {ci} | {r['n50'] if r['n50'] else 'n/a'} | "
                     f"{r['above_0']} | {oc} |")
    L += ["", "## How often a longer horizon changes E's choice", "",
          "| setting | H | first decisions | changed | changes | every decision of the H = " + str(Hs[-1]) + " run | changed |",
          "|---|---|---|---|---|---|---|"]
    for name, s in summ["settings"].items():
        for H in Hs[1:]:
            d, e = s["differ_first"][str(H)], s["differ_all"][str(H)]
            pairs = ", ".join(f"{k} ({v})" for k, v in sorted(d["pairs"].items(), key=lambda kv: -kv[1])) or "none"
            L.append(f"| {name} | {H} | {d['n']} | {d['differ']} | {pairs} | {e['n']} | {e['differ']} |")
    L += ["", "## Checks", ""]
    for name, s in summ["settings"].items():
        c = s["consistency"]
        L.append(f"- {name}: {s['cases']} cases, {s['no_menu']} without a menu, {s['incomplete']} incomplete, {s['artifact']} reset "
                 f"artifacts; the first tables agree across the runs on {c['compared'] - len(c['differ'])} of {c['compared']}"
                 + (f" (differ: {', '.join(c['differ'])})" if c["differ"] else "") + ".")
        L.append(f"  - A policy's first choice below E's on E's own table: {len(s['below_e'])}"
                 + ("".join(f"; {x['case']} H {x['H']} {x['policy']} {x['choice']} {_f(x['value'])} < E {x['e_choice']} "
                            f"{_f(x['e_value'])}" for x in s["below_e"]) if s["below_e"] else "") + ".")
        L.append(f"  - The lowest score rises with H: {len(s['not_monotone'])}"
                 + ("".join(f"; {x['case']} H {x['H']} {_f(x['value'])} > {_f(x['below'])}" for x in s["not_monotone"])
                    if s["not_monotone"] else "") + ".")
    L += ["", "## Wall time per case (seconds)", "", "| setting | H | E's own scoring: mean, p90, max | the whole run: mean, p90, max |",
          "|---|---|---|---|"]
    for name, s in summ["settings"].items():
        for H in Hs:
            w = s["wall"][str(H)]
            e, c = w["e_seconds"], w["case_seconds"]
            L.append(f"| {name} | {H} | {_f(e['mean'], 1)}, {_f(e['p90'], 1)}, {_f(e['max'], 1)} | "
                     f"{_f(c['mean'], 1)}, {_f(c['p90'], 1)}, {_f(c['max'], 1)} |")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    from robots.mujoco_lab import ladder

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="the default rule's first choice and E's runs at each horizon")
    p.add_argument("--cases", required=True)
    p.add_argument("--settings", default=",".join(ladder.SETTINGS))
    p.add_argument("--crossfit", required=True)
    p.add_argument("--horizons", default="1,2,3")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-tasks-per-worker", type=int, default=None, help="replace a worker after this many jobs (memory)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("summary", help="the report")
    p.add_argument("--dir", required=True)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--wandb", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        horizons = tuple(int(x) for x in a.horizons.split(",") if x)
        if not horizons or min(horizons) < 1 or 1 not in horizons:
            ap.error("horizons: positive integers, 1 among them")
        print(run(ladder.load_cases(a.cases), ladder.parse_settings(a.settings), a.crossfit, horizons, a.workers, a.out, a.cases,
                  a.max_tasks_per_worker))
        return 0
    with open(os.path.join(a.dir, "rows.jsonl")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    summ = summarize(rows, a.n_boot)
    with open(os.path.join(a.dir, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    md = markdown(summ)
    with open(os.path.join(a.dir, "summary.md"), "w") as f:
        f.write(md)
    print(md)
    if a.wandb:
        import wandb

        run_ = wandb.init(project="rpent-pick-place", job_type="ladder_e", config={"dir": a.dir, "horizons": summ["horizons"]})
        for name, s in summ["settings"].items():
            run_.summary[f"{name}/h1_match"] = s["h1_vs_oracle"]["match"]
            run_.summary[f"{name}/h1_compared"] = s["h1_vs_oracle"]["compared"]
            for H in summ["horizons"]:
                r = s["regret"][str(H)]
                run_.summary[f"{name}/H{H}/default_regret"] = r["mean"]
                run_.summary[f"{name}/H{H}/n50"] = r["n50"]
                run_.summary[f"{name}/H{H}/e_seconds_mean"] = s["wall"][str(H)]["e_seconds"]["mean"]
            for H, d in s["differ_first"].items():
                run_.summary[f"{name}/H{H}/first_changed"] = d["differ"]
            run_.summary[f"{name}/below_e"] = len(s["below_e"])
            run_.summary[f"{name}/not_monotone"] = len(s["not_monotone"])
        print(f"wandb: {run_.url}")
        run_.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
