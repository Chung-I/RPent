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


"""Experiment 1: the value of perfect information per hidden property at the first decision (daily-logs
researches/property-belief-manipulation/experiments/2026-11-05-evpi-decomposition).

Per com_gap case (calibrate's cross-fit cases on first_menus' reset), prior and setting (care κ, abstain cost A): the
one-step rule's first decision as com_gap.first_attempt reads its first menu (``choice_of``: decline, the recommended
probe, or the best row) under the prior b0 (a_prior) and under b0 with a set S of hidden properties revealed (a_S), S in
REVEALS:
  mass    the mass;
  mu      the friction;
  com     the centre of mass (com_obj, com_frac);
  damage  the crush force and the impact speeds (f_damage_n, v_bruise_ms, v_break_ms) and, for a fillable object, the fill
          and its spill angle; the stiffness keeps its prior given the crush force (the fragility latent's rho);
  all     every property: a point mass at the truth.
Reveal (``reveal``): every particle takes the truth's values of S, and the other keys keep their prior. Under the v11
generator the mass, the friction, the CoM and the fragility of an object without liquid are independent, so that is the
conditional. A fillable object's mass, CoM and fill are coupled: the revealed ones among them first weight the particles
(narrow kernels: the log mass at EXACT_SD["mass"], the CoM point at EXACT_SD["com"] times the pivot length, the fill at
FILL_SD), refreshed from the prior under MIN_ESS (com_gap.condition), and are then set.
J(a; θ_true) (``true_cost``): the expected cost of choice a at the truth from the outcome model: a row's particle cost at
the truth particle, A for decline, and for a probe its one-step preposterior at the truth (``probe_true_cost``: results
drawn at the truth, the row the choosing belief then picks, scored at the truth). EVPI_S = J(a_prior) - J(a_S).
The menus of one belief under the settings share their setting-free work (``SharedMenus``: each memo returns what the
call returns for the same inputs; the replay of com_gap's first decisions in the summary checks it).

    python -P -m robots.mujoco_lab.evpi_props run --bundle B --scenes S --out DIR [--objects O,...] [--sample 20]
    python -P -m robots.mujoco_lab.evpi_props summary --dirs D1,D2,... --out DIR [--comgap RECORDS] [--wandb]
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
import types
from collections import Counter, defaultdict
from typing import Any, Callable

import numpy as np

from robots.mujoco_lab import tracking
from robots.mujoco_lab.com_gap import _mean_ci, condition, sample_cases
from robots.mujoco_lab.vision_prior import EXACT_SD
from rpent.skills.pick_place import menu as menu_mod
from rpent.skills.pick_place.belief import ParticleBelief
from rpent.skills.pick_place.costs import COST, TIME_COST_PER_S, care_scale, particle_costs
from rpent.skills.pick_place.features import FRAG_KEYS, model_names
from rpent.skills.pick_place.menu import EVOI_K, push_tip_charge
from rpent.skills.pick_place.probes import DAMAGED, PROBE_SECONDS

SETTINGS = (("k100_A3", 100.0, 3.0), ("k30_A3", 30.0, 3.0), ("k100_A10", 100.0, 10.0))
LOW_STAKES = (("k10_A10", 10.0, 10.0), ("k1_A10", 1.0, 10.0))      # the low-stakes controls (coordinator, 2026-11-05)
PRIORS = ("class", "generic")
PRIOR = "prior"
REVEALS = ("mass", "mu", "com", "damage", "all")
SINGLE = ("mass", "mu", "com", "damage")
ABSTAINS = ("decline", "unreachable")    # an unreachable case (no pinch the arm reaches) ends undone, as a decline does
DAMAGE_KEYS = ("f_damage_n", "v_bruise_ms", "v_break_ms")
COUPLED = ("mass", "com", "damage")       # the reveals that weight a fillable object's particles first
FILL_SD = 0.01                            # the fill kernel (the fill is U[0.3, 0.9] when filled)
BAR = 0.02                                # a property matters when the 95 % lower bound of its mean EVPI is above this
ESS_TABLE = 100.0                         # the second table keeps the cases whose revealed beliefs all have this ESS
N_BOOT = 2000
REVEAL_STREAM = 6                         # calibrate._stream k (1 menus, 2 random row, 3 and 4 com_gap's refreshes)
TRUE_STREAM = 7                           # calibrate._stream k of a probe's results at the truth
LOAD_BINS = (0.25, 0.5, 1.0)              # the weight over the grip's friction capacity 2 mu F_std
G = 9.81


# ---------------------------------------------------------------- the truth and the reveals

def truth_particle(pose: dict[str, Any], theta: dict[str, Any], com_end: int, hidden: dict[str, Any], entry: dict[str, Any],
                   generator: str = "v11") -> dict[str, Any]:
    """The case's generator truth as one particle of the prior's space (prior.make_prior's keys): mass, friction and CoM of
    posterior_check.generator_truth, the CoM branch keys of the case's theta, the fill and its liquid, and the hidden
    fragility (its latent z as frag_z)."""
    from robots.mujoco_lab.posterior_check import generator_truth
    from robots.mujoco_lab.prior import com_hypothesis, joint_hidden

    o = str(pose.get("object_id"))
    fill = float(hidden.get("fill") or 0.0)
    g = generator_truth({"object": o, "pose": pose, "theta": theta, "com_end": com_end, "fill": fill, "generator": generator},
                        {o: entry})
    com_s = theta.get("com_s")
    f = com_hypothesis(pose["pivot"], pose["com_obj_nominal"], com_s, int(com_end))[1]
    m_l = float(joint_hidden(pose, entry, theta, int(com_end), fill, generator)["liquid_mass"]) if fill > 0 else 0.0
    frag = hidden.get("frag") or {}
    return {"mass": float(g["mass"]), "mu": float(g["mu"]), "com_obj": [float(x) for x in g["com_obj"]],
            "mass_scale": float(theta["mass_scale"]), "com_branch": com_s is not None, "com_s": None if com_s is None else float(com_s),
            "com_end": int(com_end), "f": float(f), "com_frac": float(g["com"]), "ancestor": 0, "fill": fill, "liquid_mass": m_l,
            "spill_angle_deg": None if fill <= 0 or hidden.get("spill_angle_deg") is None else float(hidden["spill_angle_deg"]),
            **{k: None if frag.get(k) is None else float(frag[k]) for k in FRAG_KEYS},
            "frag_z": [float(z) for z in (frag.get("z") or [0.0, 0.0, 0.0])]}


def reveal_entry(S: str, truth: dict[str, Any], pivot_length: float) -> dict[str, Any]:
    """The kernel entry of a coupled reveal (a fillable object): the keys it weights and their truth."""
    keys = {"mass": ["mass"], "com": ["com"], "damage": ["fill"]}[S]
    return {"type": "reveal", "keys": keys, "truth": {"mass": truth["mass"], "com_obj": truth["com_obj"], "fill": truth["fill"]},
            "com_sd_m": float(EXACT_SD["com"]) * float(pivot_length)}


def reveal_loglik(b: ParticleBelief, e: dict[str, Any]) -> np.ndarray:
    """A reveal entry's log-likelihood on any particles: N(log mass; log truth, EXACT_SD["mass"]^2), the CoM point's
    isotropic normal at ``com_sd_m``, N(fill; truth, FILL_SD^2)."""
    a, t = b.arrays(), e["truth"]
    ll = np.zeros(len(b))
    if "mass" in e["keys"]:
        ll -= 0.5 * ((np.log(a["mass"]) - math.log(float(t["mass"]))) / EXACT_SD["mass"]) ** 2
    if "com" in e["keys"]:
        d = a["com_obj"] - np.asarray(t["com_obj"], dtype=float)[None, :]
        ll -= 0.5 * (d ** 2).sum(axis=1) / float(e["com_sd_m"]) ** 2
    if "fill" in e["keys"]:
        ll -= 0.5 * ((a["fill"] - float(t["fill"])) / FILL_SD) ** 2
    return ll


def _values(S: str, truth: dict[str, Any], fillable: bool) -> dict[str, Any]:
    """The keys a reveal sets in every particle. A fillable object keeps its particles' own scale, liquid and CoM-branch
    keys (the kernel weighting made them agree with the truth)."""
    if S == "mass":
        return {"mass": truth["mass"]} if fillable else {"mass": truth["mass"], "mass_scale": truth["mass_scale"]}
    if S == "mu":
        return {"mu": truth["mu"]}
    if S == "com":
        keys = ("com_obj", "com_frac") if fillable else ("com_obj", "com_frac", "com_branch", "com_s", "com_end", "f")
        return {k: copy.deepcopy(truth[k]) for k in keys}
    if S == "damage":
        out = {k: truth[k] for k in DAMAGE_KEYS}
        if fillable:
            out.update(fill=truth["fill"], spill_angle_deg=truth["spill_angle_deg"], liquid_mass=truth["liquid_mass"])
        return out
    raise ValueError(f"unknown reveal {S!r}")


def reveal(b0: ParticleBelief, S: str, truth: dict[str, Any], entry: dict[str, Any], thresholds: Any, draw: Callable,
           rng: np.random.Generator, pivot_length: float) -> tuple[ParticleBelief, dict[str, Any]]:
    """b0 with the properties of S revealed (the module docstring); "all" is the truth as a point mass. Returns (belief,
    info): the weighting's ess, refreshed, importance_ess, ess_final, ancestors, degenerate and weighted."""
    if S == "all":
        b = b0._replace(particles=[dict(truth)], weights=np.ones(1), history=[], degenerate=False, importance_ess=None)
        return b, {"weighted": False, "ess": 1.0, "refreshed": False, "importance_ess": None, "ess_final": 1.0, "ancestors": 1,
                   "degenerate": False}
    fillable = bool(entry.get("fillable"))
    if fillable and S in COUPLED:
        b, info = condition(b0, [reveal_entry(S, truth, pivot_length)], reveal_loglik, draw, rng)
        info["weighted"] = True
    else:
        b, info = b0, {"weighted": False, "ess": b0.effective_size(), "refreshed": False, "importance_ess": None,
                       "ess_final": b0.effective_size(), "ancestors": b0.n_ancestors(), "degenerate": bool(b0.degenerate)}
    vals = _values(S, truth, fillable)
    parts = [dict(p, **copy.deepcopy(vals)) for p in b.particles]
    if S == "damage":                     # the stiffness: the truth's crush latent, each particle's own second latent
        z = np.array([p.get("frag_z") or [0.0, 0.0, 0.0] for p in parts], dtype=float).reshape(-1, 3)
        z[:, 0], z[:, 2] = truth["frag_z"][0], truth["frag_z"][2]
        k = _stiffness(z, entry, thresholds)
        for i, p in enumerate(parts):
            p["frag_z"] = z[i].tolist()
            p["k_n_per_mm"] = None if k is None else float(k[i])
    return b._replace(particles=parts), info


def _stiffness(z: np.ndarray, entry: dict[str, Any], thresholds: Any) -> np.ndarray | None:
    from robots.mujoco_lab.fragility import map_latent_arrays

    if thresholds is None:
        return None
    return map_latent_arrays(z, entry, thresholds)["k_n_per_mm"]


# ---------------------------------------------------------------- shared menus

def _canon(x: Any) -> Any:
    """A hashable, exact key for nested dicts, lists, arrays and scalars."""
    if isinstance(x, dict):
        return ("d",) + tuple(sorted((str(k), _canon(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple)):
        return ("l",) + tuple(_canon(v) for v in x)
    if isinstance(x, np.ndarray):
        return ("a", x.dtype.str, x.shape, x.tobytes())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if hasattr(x, "to_dict"):
        return _canon(x.to_dict())
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return ("o", type(x).__name__, _canon({f.name: getattr(x, f.name) for f in dataclasses.fields(x)}))
    if hasattr(x, "__dict__"):
        return ("o", type(x).__name__, _canon(vars(x)))
    return x


def _digest(x: Any) -> bytes:
    if x is None:
        return b""
    return hashlib.blake2b(np.ascontiguousarray(np.asarray(x, dtype=float)).tobytes(), digest_size=16).digest()


class SharedMenus:
    """While active (a context manager): memos for the work of one belief's menus that no setting changes (the outcome
    model's predictions per row, the feature matrices, the probe models' kind, risk and likelihood calls, the guard's
    check) and for the lab's reach and plans of one case. A memo returns the object the call returned for the same
    inputs: arrays and beliefs are keyed by identity while the memo holds them, everything else by value. ``clear``
    drops the belief memos (call it between beliefs), ``clear_case`` the lab's too (call it after a reset)."""

    def __init__(self, prims: Any):
        self.prims = prims
        self._undo: list[Callable[[], None]] = []
        self.hits: Counter = Counter()
        self.clear_case()

    def clear(self) -> None:
        self.memo: dict[Any, Any] = {}
        self.alive: dict[int, Any] = {}
        self.pdig: dict[int, bytes] = {}
        self.blocks: dict[int, int] = {}          # a guard query's id -> its row count (guard_queries)

    def clear_case(self) -> None:
        self.clear()
        self.lab: dict[Any, Any] = {}

    def _hold(self, x: Any) -> int:
        self.alive[id(x)] = x
        return id(x)

    def _held(self, x: Any) -> bool:
        return self.alive.get(id(x)) is x

    def _cached(self, key: Any, fn: Callable[[], Any], kind: str) -> Any:
        if key in self.memo:
            self.hits[kind] += 1
            return self.memo[key]
        self.hits[kind + "_miss"] += 1
        out = self.memo[key] = fn()
        return out

    def _patch(self, obj: Any, name: str, new: Callable) -> None:
        own = isinstance(obj, types.ModuleType) or name in vars(obj)
        old = vars(obj).get(name) if own else None
        setattr(obj, name, new)
        self._undo.append((lambda: setattr(obj, name, old)) if own else (lambda: delattr(obj, name)))

    def __enter__(self) -> SharedMenus:
        est, pmat = menu_mod.estimate, menu_mod.particle_matrix

        def estimate(model, belief, base, pose, force, planned, rng=None):
            key = ("est", id(model), self._hold(belief), _canon(base), _canon(pose), float(force), _canon(planned))
            return self._cached(key, lambda: est(model, belief, base, pose, force, planned, rng), "estimate")

        def particle_matrix(base, arrays, pose, force, names):
            key = ("pm", self._hold(arrays), _canon(base), _canon(pose), float(force), tuple(names))
            X = self._cached(key, lambda: pmat(base, arrays, pose, force, names), "particle_matrix")
            self._hold(X)
            return X

        self._patch(menu_mod, "estimate", estimate)
        self._patch(menu_mod, "particle_matrix", particle_matrix)
        for name, pm in (getattr(self.prims, "probe_models", None) or {}).items():
            self._patch_probe(name, pm)
        gq = menu_mod.guard_queries

        def guard_queries(options, belief, *args, **kw):
            X, w = gq(options, belief, *args, **kw)
            self._hold(X)
            self.blocks[id(X)] = sum(o.kind in ("commit", "lower_risk") for o in options)
            return X, w

        self._patch(menu_mod, "guard_queries", guard_queries)
        guard = getattr(self.prims, "guard", None)
        if guard is not None:
            check = guard.check

            def guard_check(X, weights=None):
                R = self.blocks.get(id(X)) if self._held(X) else None
                if R:                     # the menu ranks the rows by cost, so a setting can stack them in another order:
                    n = len(X) // R       # the score (a weighted median over the rows, equal weights per block) is not
                    key = ("guard", tuple(sorted(_digest(X[i * n:(i + 1) * n]) for i in range(R))),   # changed by it
                           _digest(None if weights is None else np.asarray(weights)[:n]))
                else:
                    key = ("guard", _digest(X), _digest(weights))
                return dict(self._cached(key, lambda: check(X, weights), "guard"))

            self._patch(guard, "check", guard_check)
        env = self.prims.env
        for meth in ("reach", "plan_trajectory"):
            fn = getattr(env, meth, None)
            if fn is None:
                continue

            def lab(*args, _fn=fn, _m=meth, **kw):
                key = (_m, _canon(args), _canon(kw))
                if key in self.lab:
                    self.hits[_m] += 1
                    return self.lab[key]
                self.hits[_m + "_miss"] += 1
                out = self.lab[key] = _fn(*args, **kw)
                return out

            self._patch(env, meth, lab)
        return self

    def _patch_probe(self, name: str, pm: Any) -> None:
        for meth in ("p_kind", "risk", "terminal_risk"):
            fn = getattr(pm, meth)

            def one(X, _fn=fn, _m=meth):
                if not self._held(X):
                    return _fn(X)
                return self._cached((_m, name, id(X)), lambda: _fn(X), _m)

            self._patch(pm, meth, one)
        loglik = pm.loglik

        def ll(obs, X, arrays, force, temper=None, terms=None, P=None, thrown=None):
            call = lambda: loglik(obs, X, arrays, force, temper=temper, terms=terms, P=P, thrown=thrown)  # noqa: E731
            if not (self._held(X) and self._held(arrays)):
                return call()
            if P is not None and id(P) not in self.pdig:
                self._hold(P)
                self.pdig[id(P)] = _digest(P)
            key = ("ll", name, id(X), id(arrays), _canon(obs), float(force), temper, _canon(terms),
                   None if P is None else self.pdig[id(P)], thrown)
            return self._cached(key, call, "loglik")

        self._patch(pm, "loglik", ll)

    def __exit__(self, *exc: Any) -> bool:
        while self._undo:
            self._undo.pop()()
        self.clear_case()
        return False


# ---------------------------------------------------------------- menus, choices and costs at the truth

def _rng_at(state: dict[str, Any]) -> np.random.Generator:
    g = np.random.default_rng()
    g.bit_generator.state = copy.deepcopy(state)
    return g


def choice_of(a: dict[str, Any]) -> str:
    """The first decision of a first menu as com_gap.first_attempt takes it: decline, the recommended probe, or the best
    row; "error" or "unreachable" when the menu has none."""
    if "error" in a:
        return "error"
    if not a.get("assessed", True):
        return "unreachable"
    rec = a.get("recommendation")
    if rec == "decline":
        return "decline"
    if rec == "probe":
        return a.get("recommended_option") or "error"
    return a.get("best_option") or "no_row"


def action_key(choice: str, menu: dict[str, Any] | None) -> str:
    """The choice and its action (rounded), so that the same row id on two menus (regrasp_com's pinch, the squeeze's force)
    compares by what it does."""
    o = (menu or {}).get(choice)
    if o is None or not o.action:
        return choice
    act = {k: (round(float(v), 9) if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in sorted(o.action.items())}
    return f"{choice}|{json.dumps(act, sort_keys=True)}"


def first_menu(prims: Any, b: ParticleBelief, destination: str, care: float, A: float, state: dict[str, Any]) -> dict[str, Any]:
    """The first menu of belief b under (care, A): the assessment payload, the menu's options by id and the assessment,
    with the menu stream at ``state`` (the stream after the prior's draw) and a fresh store holding b."""
    from rpent.skills.pick_place.tools import assess_pick

    env = prims.env
    oid = env.locate("object")["object_id"]
    store = {"beliefs": {oid: {"episode": int(getattr(env, "episode", 0)), "belief": b, "probes_run": set(), "probe_seq": [],
                               "attempted": False}}}
    prims.abstain_cost, prims.rng = float(A), _rng_at(state)
    a = assess_pick(object="object", destination=destination, care=care, primitives=prims, store=store).data
    cache = list(store.get("pick_place", {}).values())
    if len(cache) > 1:
        raise RuntimeError(f"{len(cache)} cached assessments for one menu")
    return {"payload": a, "menu": cache[0]["menu"] if cache else {}, "assessment": cache[0]["assessment"] if cache else None}


def row_true_cost(prims: Any, o: Any, tb: ParticleBelief, care: float, A: float, reset_s: float | None, price: str) -> float:
    """A commit or lower-risk row's expected cost at the truth particle (the menu's particle cost of that particle)."""
    per = menu_mod.estimate(prims.model, tb, o._base, o._pose, o.action["force"], o._planned)[1]
    return float(particle_costs(per, o.cost_s, care, A, reset_s, price)[0])


def _kept(pm: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P = np.asarray(pm.p_kind(X), dtype=float)
    kept = P.copy()
    if DAMAGED in pm.classes:
        kept[:, list(pm.classes).index(DAMAGED)] = 0.0
    total = kept.sum(axis=1, keepdims=True)
    return np.divide(kept, total, out=np.zeros_like(kept), where=total > 0), total


def probe_true_cost(prims: Any, o: Any, menu: dict[str, Any], b: ParticleBelief, tb: ParticleBelief, destination: str, care: float,
                    A: float, time_scale: float, rng: np.random.Generator, k: int = EVOI_K) -> dict[str, Any]:
    """A probe row's expected cost at the truth, one step as probe_value prices it under the belief: its time, the push's
    tip charge at the truth, the truth's terminal risk r at the irreversible cost and A, and with 1 - r the mean over k
    results drawn at the truth of the continuation row (or abstain) that b, reweighted by the result and its keep
    probability, then picks, scored at the truth. Absorbing only (a recovery-mode menu raises)."""
    from rpent.skills.pick_place.features import particle_matrix

    reset_s, price = None, "time"
    probe = o.action["probe"]
    pm = prims.probe_models[probe]
    cont = [r for r in menu.values() if r.kind in ("commit", "lower_risk") and r.action.get("destination") == destination
            and r.id != "inboard_place"]
    C = np.vstack([r._costs for r in cont] + [np.full(len(b), float(A))])
    Ct = np.array([row_true_cost(prims, r, tb, care, A, reset_s, price) for r in cont] + [float(A)])
    force, names = float(o.action["force"]), model_names(pm)
    Xb = particle_matrix(o._base, b.arrays(), o._pose, force, names)
    Xt = particle_matrix(o._base, tb.arrays(), o._pose, force, names)
    Qb, _ = _kept(pm, Xb)
    Qt, total_t = _kept(pm, Xt)
    keep_b = 1.0 - np.asarray(pm.terminal_risk(Xb), dtype=float)
    r_t = float(np.asarray(pm.terminal_risk(Xt), dtype=float)[0])
    terminal = COST["dropped_high"] * care_scale(care) + float(A)
    tip = push_tip_charge(tb, pm, Xt) if probe == "push" else 0.0
    picks: list[int] = []
    if total_t[0, 0] > 0:
        a = b.arrays()
        for _ in range(int(k)):
            obs = pm.sample(Xt[0], tb.particles[0], force, rng, p=Qt[0])
            ll, _ = pm.loglik(obs, Xb, a, force, P=Qb)
            wk = b.reweight(ll).weights * keep_b
            picks.append(int(np.argmin(C @ wk)))
    after = float(np.mean(Ct[picks])) if picks else 0.0
    ids = [r.id for r in cont] + ["decline"]
    J = time_scale * TIME_COST_PER_S * PROBE_SECONDS[probe] + tip + r_t * terminal + (1.0 - r_t) * after
    return {"J": float(J), "risk": r_t, "tip": float(tip), "after": after, "picks": dict(Counter(ids[i] for i in picks))}


def true_cost(prims: Any, choice: str, m: dict[str, Any], b: ParticleBelief, tb: ParticleBelief, destination: str, care: float,
              A: float, rng_fn: Callable[[], np.random.Generator]) -> dict[str, Any]:
    """J(choice; θ_true) for the choice of menu ``m`` (first_menu's) under belief b: A for decline, a row's cost at the
    truth, a probe's one-step cost at the truth (``rng_fn`` gives its results' stream); A for an unreachable case (the
    skill cannot pinch it for this destination under any belief, so every EVPI is 0, as com_gap counts it); None for no
    choice."""
    if choice in ABSTAINS:
        return {"J": float(A)}
    o = m["menu"].get(choice)
    if o is None:
        return {"J": None}
    asm = m["assessment"]
    if asm.reset_s is not None:
        raise NotImplementedError("evpi_props prices the absorbing menu only")
    if o.kind == "probe":
        return probe_true_cost(prims, o, m["menu"], b, tb, destination, care, A, float(asm.probe_time_scale), rng_fn())
    return {"J": row_true_cost(prims, o, tb, care, A, None, "time")}


def belief_evpi(m: dict[str, Any], b: ParticleBelief, destination: str, A: float, without_probe: str | None, best: str | None) -> float | None:
    """The value of perfect information under the belief itself: the belief's mean, over its particles, of the cost of
    the choice without probes (the best same-destination row, or decline) less the particle's cheapest same-destination
    row or abstain. EVOI is at most this quantity, so a probe's EVOI is read against it."""
    same = [o for o in m["menu"].values() if o.kind in ("commit", "lower_risk") and o.action.get("destination") == destination]
    if not same:
        return None
    C = np.vstack([o._costs for o in same] + [np.full(len(b), float(A))])
    if without_probe == "decline":
        c = np.full(len(b), float(A))
    else:
        row = m["menu"].get(best)
        if row is None or row.kind not in ("commit", "lower_risk"):
            return None
        c = row._costs
    return float(b.weights @ (c - C.min(axis=0)))


# ---------------------------------------------------------------- one case

def _load_bin(x: float) -> str:
    edges = (0.0,) + LOAD_BINS + (math.inf,)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= x < hi:
            return f"[{lo:g}, {hi:g})"
    return "nan"


def run_case(prims: Any, o: str, idx: int, s: dict[str, Any], priors: dict[str, Callable], world: Any,
             shared: SharedMenus) -> list[dict[str, Any]]:
    """One com_gap case: first_menus' reset, the truth particle, then per prior the prior draw on the menu stream
    (calibrate._stream k 1), the reveals (REVEAL_STREAM), every belief's first menu under every setting (the stream
    after the prior's draw, as com_gap's first menu), the choices, J at the truth (TRUE_STREAM) and the EVPIs."""
    from robots.mujoco_lab import calibrate as cal
    from robots.mujoco_lab.adapter import object_props
    from robots.mujoco_lab.com_gap import fingerprint
    from robots.mujoco_lab.fragility import entry_of
    from rpent.skills.pick_place.tools import _particles, _spread_kw, standard_force

    env, seed = prims.env, int(s["seed"])
    theta, com_end, destination, hidden = cal.case_draw(seed, o, idx, object_props(o)["mass"], False, fragility=world)
    if world is not None:
        env.slosh_zeta, env.slosh_scale = world[1].slosh_zeta, world[1].slosh_scale
    env.classes = None if world is None else world[0]
    env.reset(o, theta, seed=seed * 1000 + idx, com_end=com_end, fixture=destination if destination in cal.MESH_IDS else None,
              fill=hidden["fill"], frag=hidden["frag"])
    shared.clear_case()
    fp = fingerprint(env)
    reset_info = dict(getattr(env, "reset_info", None) or {})
    pose = env.pose_info()
    entry = entry_of(o, world[0])[0] if world is not None else {"class": "none", "fillable": False}
    th = None if world is None else world[1]
    truth = truth_particle(pose, theta, com_end, hidden, entry)
    n, spread = _particles(prims), _spread_kw(prims)
    f_std = standard_force(env.nominal_mass(o))
    load = truth["mass"] * G / (2.0 * truth["mu"] * f_std)
    case = {"object": o, "idx": idx, "fold": s["fold"], "destination": destination, "flat": s.get("flat"),
            "at_rest": reset_info.get("at_rest", True), "fingerprint": fp, "class": entry.get("class", "none"),
            "fillable": bool(entry.get("fillable")), "fill": truth["fill"], "n_particles": n, "f_std": f_std, "load_ratio": load,
            "load_bin": _load_bin(load), "truth": {k: truth[k] for k in ("mass", "mu", "com_frac", "fill", *DAMAGE_KEYS)}}
    recs = []
    for pname, fn in priors.items():
        t0 = time.time()
        prims.prior = fn
        mine: list[dict[str, Any]] = []
        try:
            g = cal._stream(seed, o, idx, 1)
            b0 = fn(pose, n, g, **spread)
            state = copy.deepcopy(g.bit_generator.state)
            rr = cal._stream(seed, o, idx, REVEAL_STREAM)

            def draw(m: int, r: np.random.Generator, _fn=fn) -> ParticleBelief:
                return _fn(pose, m, r, **spread)

            beliefs = {PRIOR: (b0, {"weighted": False, "ess": b0.effective_size(), "ess_final": b0.effective_size(),
                                    "refreshed": False, "ancestors": b0.n_ancestors()})}
            for S in REVEALS:
                beliefs[S] = reveal(b0, S, truth, entry, th, draw, rr, float(pose["pivot"]["L"]))
            tb = beliefs["all"][0]
            menus: dict[str, dict[str, Any]] = {}
            for key, (b, _) in beliefs.items():
                menus[key] = {name: first_menu(prims, b, destination, care, A, state) for name, care, A in s["settings"]}
                shared.clear()
            for name, care, A in s["settings"]:
                mine.append(_record(case, pname, name, care, A, prims, beliefs, menus, tb, destination, seed, o, idx))
            for r in mine:
                r["prior_seconds"] = round(time.time() - t0, 2)
        except Exception as exc:  # noqa: BLE001  one bad case must not stop the run
            mine = [dict(case, prior=pname, setting=name, care=care, abstain_cost=A, error=f"{type(exc).__name__}: {str(exc)[:300]}")
                    for name, care, A in s["settings"]]
        recs += mine
        shared.clear()
    for r in recs:
        r["artifact"] = cal.case_artifact(r, reset_info)
    return recs


def _record(case: dict[str, Any], pname: str, name: str, care: float, A: float, prims: Any, beliefs: dict[str, Any],
            menus: dict[str, Any], tb: ParticleBelief, destination: str, seed: int, o: str, idx: int) -> dict[str, Any]:
    from robots.mujoco_lab import calibrate as cal

    rec = dict(case, prior=pname, setting=name, care=care, abstain_cost=A)
    choice, akey, J, detail, side = {}, {}, {}, {}, {}
    for key, (b, _) in beliefs.items():
        m = menus[key][name]
        a = m["payload"]
        choice[key] = c = choice_of(a)
        akey[key] = action_key(c, m["menu"])
        t = true_cost(prims, c, m, b, tb, destination, care, A, lambda: cal._stream(seed, o, idx, TRUE_STREAM))
        J[key] = t["J"]
        if len(t) > 1:
            detail[key] = t
        guard = a.get("guard") or {}
        side[key] = {"recommendation": a.get("recommendation"), "best_option": a.get("best_option"),
                     "recommended_option": a.get("recommended_option"), "without_probe": a.get("without_probe"),
                     "guard_flag": bool(guard.get("flag")), "first_menu": cal._first_menu(a)}
    m0 = menus[PRIOR][name]
    evoi = {oid: {"evoi": o.probe["evoi"], "evoi_se": o.probe["evoi_se"], "plugin_evoi": o.probe.get("plugin_evoi"),
                  "expected_cost": o.expected_cost} for oid, o in m0["menu"].items() if o.kind == "probe"}
    rec.update(choice=choice, action=akey, J=J, probe_detail=detail, menu=side,
               evpi={S: (None if J[PRIOR] is None or J[S] is None else J[PRIOR] - J[S]) for S in REVEALS},
               ess={k: v[1] for k, v in beliefs.items()}, evoi=evoi,
               evpi_belief=belief_evpi(m0, beliefs[PRIOR][0], destination, A, side[PRIOR]["without_probe"], side[PRIOR]["best_option"]),
               all_rows={oid: o.expected_cost for oid, o in menus["all"][name]["menu"].items()
                         if o.kind in ("commit", "lower_risk") and o.action.get("destination") == destination})
    return rec


def run_object(job: tuple[str, list[int], dict[str, Any]]) -> list[dict[str, Any]]:
    """One object's cases on its fold's bundle (com_gap.run_object's sensor, world and priors)."""
    from robots.mujoco_lab import calibrate as cal
    from robots.mujoco_lab.prior import fragility_world
    from robots.mujoco_lab.vision_prior import shape_prior

    o, idxs, s = job
    prims = cal._primitives(s["bundle"], None, None)
    base = prims.prior
    priors = {"class": base, "generic": shape_prior(base, {o: s["generic"][o]})}
    priors = {k: priors[k] for k in s.get("priors", PRIORS)}
    prims.force_noise = cal.job_force_noise(None, prims.manifest)
    prims.env.force_noise = prims.force_noise
    world = fragility_world(prims.manifest)
    rows = []
    try:
        with SharedMenus(prims) as shared:
            for idx in idxs:
                t0 = time.time()
                rows += run_case(prims, o, idx, s, priors, world, shared)
                print(f"{o} {idx}: {time.time() - t0:.1f} s, memo {dict(shared.hits)}", file=sys.stderr, flush=True)
                shared.hits.clear()
    finally:
        prims.prior = base
    return rows


# ---------------------------------------------------------------- the run

def _setting(text: str) -> tuple[str, float, float]:
    name, care, A = text.split(":")
    return name, float(care), float(A)


def run(a: argparse.Namespace) -> str:
    from robots.mujoco_lab.estimation_check import _revision
    from robots.mujoco_lab.first_menus import prior_tables
    from robots.mujoco_lab.identify_fit import require_bundle_generator

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "records.jsonl")
    if os.path.exists(path):
        raise SystemExit(f"{path} exists: pick a new --out")
    generator = require_bundle_generator(a.bundle)
    if generator != "v11":
        raise SystemExit(f"evpi_props reveals under the v11 generator's couplings; the bundle's generator is {generator}")
    tables, spreads = prior_tables(a.bundle, a.scenes, {})
    folds = json.load(open(os.path.join(a.bundle, "folds", "folds.json")))
    flat = {o: bool(v["flat"]) for o, v in json.load(open(a.scenes)).items()}
    objects = a.objects.split(",") if a.objects else sorted(folds)
    cases = sample_cases([(o, i) for o in objects for i in range(a.repeats)], a.sample, a.sample_seed)
    by: dict[str, list[int]] = defaultdict(list)
    for o, i in cases:
        by[o].append(i)
    settings = [_setting(t) for t in a.settings.split(",")] if a.settings else [tuple(x) for x in SETTINGS + LOW_STAKES]
    priors = a.priors.split(",") if a.priors else list(PRIORS)
    with open(os.path.join(a.out, "run_config.json"), "w") as f:
        json.dump({**vars(a), **_revision(), "generator": generator, "settings_parsed": settings, "priors_parsed": priors,
                   "cases": cases, "spreads": spreads, "generic": tables["generic"], "exact_sd": EXACT_SD, "fill_sd": FILL_SD,
                   "bar": BAR, "ess_table": ESS_TABLE, "reveals": REVEALS}, f, indent=1)
    jobs = [(o, idxs, {"bundle": os.path.join(a.bundle, "folds", f"fold_{folds[o]}"), "fold": folds[o], "seed": a.seed,
                       "settings": settings, "priors": priors, "generic": {o: tables["generic"][o]}, "flat": flat.get(o)})
            for o, idxs in by.items()]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool, open(path, "w") as out:
        for rows in pool.imap_unordered(run_object, jobs):
            for r in rows:
                out.write(json.dumps(r) + "\n")
            out.flush()
    print(f"done in {time.time() - t0:.0f} s: {path}", flush=True)
    return path


# ---------------------------------------------------------------- the summary

CONFIG_KEYS = ("bundle", "scenes", "seed", "repeats", "sample", "sample_seed", "code_revision", "generator", "settings_parsed",
               "priors_parsed", "spreads", "exact_sd", "fill_sd", "bar", "ess_table", "reveals")


def load(dirs: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The records of one run's shard dirs, which must agree on CONFIG_KEYS and not repeat a case."""
    recs, cfg = [], None
    for d in dirs:
        c = json.load(open(os.path.join(d, "run_config.json")))
        if c.get("dirty"):
            raise SystemExit(f"{d}: dirty code")
        if cfg is not None:
            bad = [k for k in CONFIG_KEYS if c.get(k) != cfg.get(k)]
            if bad:
                raise SystemExit(f"{d}: config differs in {bad}")
        cfg = cfg or c
        for line in open(os.path.join(d, "records.jsonl")):
            if line.strip():
                recs.append(json.loads(line))
    seen = Counter((r["object"], r["idx"], r["prior"], r["setting"]) for r in recs)
    dup = [k for k, v in seen.items() if v > 1]
    if dup:
        raise SystemExit(f"{len(dup)} repeated case rows, e.g. {dup[0]}")
    return recs, cfg or {}


def groups(r: dict[str, Any]) -> dict[str, bool]:
    """Step 5's groups from the choices alone (no EVPI value): invariant (every reveal keeps a_prior's action), critical
    (some reveal changes it), and the overlay abstain_dominated (decline under the prior and under the truth)."""
    a, c = r["action"], r["choice"]
    invariant = all(a[S] == a[PRIOR] for S in REVEALS)
    return {"invariant": invariant, "critical": not invariant, "invariant_id": all(c[S] == c[PRIOR] for S in REVEALS),
            "abstain_dominated": c[PRIOR] in ABSTAINS and c["all"] in ABSTAINS}


def ess_ok(r: dict[str, Any]) -> bool:
    """The second table keeps a case when the prior and every single-property reveal keep ESS_TABLE effective particles."""
    return all(float(r["ess"][k]["ess_final"]) >= ESS_TABLE for k in (PRIOR,) + SINGLE)


def _counted(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in recs if "error" not in r and not r.get("artifact") and all(r["J"][k] is not None for k in r["J"])]


def evpi_table(recs: list[dict[str, Any]], rng: np.random.Generator, n_boot: int) -> dict[str, Any]:
    objs = [r["object"] for r in recs]
    out = {}
    for S in REVEALS:
        c = _mean_ci(np.array([r["evpi"][S] for r in recs], dtype=float), objs, rng, n_boot)
        c["reading"] = None if c["ci"] is None else ("matters" if c["ci"][0] > BAR else "no detectable value")
        c["share_positive"] = float(np.mean([r["evpi"][S] > 1e-12 for r in recs])) if recs else None
        out[S] = c
    return out


def _shares(recs: list[dict[str, Any]], by: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
    cells: dict[str, list[dict[str, bool]]] = defaultdict(list)
    for r in recs:
        cells[by(r)].append(groups(r))
    return {k: {"n": len(v), **{g: float(np.mean([x[g] for x in v])) for g in ("invariant", "critical", "abstain_dominated", "invariant_id")}}
            for k, v in sorted(cells.items())}


def checks(recs: list[dict[str, Any]], comgap: dict[tuple, dict[str, Any]] | None) -> dict[str, Any]:
    """The checks in place of step 4's E1 check: the prior's first decision replays com_gap's (recommendation, best row,
    guard, first-menu costs); EVPI_all is never below 0 without a guard flag under the truth; a_all costs
    min(the truth menu's cheapest same-destination row, A) at the truth unless the guard flags."""
    out: dict[str, Any] = {"n": len(recs)}
    if comgap is not None:
        keys = ("recommendation", "best_option", "guard_flag", "first_menu")
        found = [(r, comgap.get((r["object"], r["idx"], r["prior"], r["setting"]))) for r in recs]
        found = [(r, c) for r, c in found if c is not None]
        bad = [(r, c) for r, c in found if any(r["menu"][PRIOR][k] != c.get(k) for k in keys)]
        out["replay"] = {"compared": len(found), "mismatch": len(bad),
                         "examples": [{"case": [r["object"], r["idx"], r["prior"], r["setting"]],
                                       "here": {k: r["menu"][PRIOR][k] for k in keys}, "com_gap": {k: c.get(k) for k in keys}}
                                      for r, c in bad[:5]]}
    neg = [r for r in recs if r["evpi"]["all"] < -1e-9]
    out["evpi_all_negative"] = {"n": len(neg), "guard_under_truth": sum(r["menu"]["all"]["guard_flag"] for r in neg)}
    off = []
    for r in recs:
        if r["menu"]["all"]["guard_flag"] or not r["all_rows"]:
            continue
        lo = min(min(r["all_rows"].values()), float(r["abstain_cost"]))
        if abs(r["J"]["all"] - lo) > 1e-6:
            off.append([r["object"], r["idx"], r["prior"], r["setting"], r["J"]["all"], lo])
    out["a_all_is_min"] = {"off": len(off), "examples": off[:5]}
    return out


def probe_gap(recs: list[dict[str, Any]], rng: np.random.Generator, n_boot: int) -> dict[str, Any] | None:
    """Step 6 on the cases whose first menu offers a probe row: the best probe's held-out EVOI against EVPI_all (at the
    truth) and against the belief's own EVPI (belief_evpi), with the cross-count at BAR."""
    rows = [r for r in recs if r["evoi"]]
    if not rows:
        return None
    objs = [r["object"] for r in rows]
    evoi = np.array([max(v["evoi"] for v in r["evoi"].values()) for r in rows], dtype=float)
    allv = np.array([r["evpi"]["all"] for r in rows], dtype=float)
    bel = [r.get("evpi_belief") for r in rows]
    out = {"n": len(rows), "evoi_max": _mean_ci(evoi, objs, rng, n_boot), "evpi_all": _mean_ci(allv, objs, rng, n_boot),
           "cross": {"evpi_all>bar, evoi<=bar": int(((allv > BAR) & (evoi <= BAR)).sum()),
                     "evpi_all>bar, evoi>bar": int(((allv > BAR) & (evoi > BAR)).sum()),
                     "evpi_all<=bar, evoi>bar": int(((allv <= BAR) & (evoi > BAR)).sum()),
                     "evpi_all<=bar, evoi<=bar": int(((allv <= BAR) & (evoi <= BAR)).sum())}}
    ok = [i for i, v in enumerate(bel) if v is not None]
    if ok:
        out["evpi_belief"] = _mean_ci(np.array([bel[i] for i in ok], dtype=float), [objs[i] for i in ok], rng, n_boot)
    return out


def summarize(recs: list[dict[str, Any]], comgap: dict[tuple, dict[str, Any]] | None = None, n_boot: int = N_BOOT,
              seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    counted = _counted(recs)
    cells: dict[str, dict[str, Any]] = defaultdict(dict)
    for (setting, prior), rs in sorted(_by(counted).items()):
        sub = [r for r in rs if ess_ok(r)]
        cells[setting][prior] = {
            "n": len(rs), "n_ess_ok": len(sub), "groups": _shares(rs, lambda r: "all")["all"] if rs else None,
            "evpi": evpi_table(rs, rng, n_boot), "evpi_ess_ok": evpi_table(sub, rng, n_boot),
            "groups_by_class": _shares(rs, lambda r: r["class"] + (" (fillable)" if r["fillable"] else "")),
            "groups_by_load": _shares(rs, lambda r: r["load_bin"]),
            "groups_ess_ok": _shares(sub, lambda r: "all")["all"] if sub else None,
            "choices": dict(Counter(r["choice"][PRIOR] for r in rs)), "probe_gap": probe_gap(rs, rng, n_boot),
            "probe_gap_ess_ok": probe_gap(sub, rng, n_boot)}
    return {"n_records": len(recs), "n_counted": len(counted), "errors": sum("error" in r for r in recs),
            "artifacts": sum(bool(r.get("artifact")) and "error" not in r for r in recs), "cells": cells,
            "checks": checks(counted, comgap), "bar": BAR, "ess_table": ESS_TABLE}


def _by(recs: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        out[(r["setting"], r["prior"])].append(r)
    return out


def _f(c: dict[str, Any] | None) -> str:
    if not c or c.get("mean") is None:
        return "–"
    return f"{c['mean']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]"


def markdown(s: dict[str, Any]) -> str:
    L = [f"# evpi_props summary", "", f"records {s['n_records']}, counted {s['n_counted']}, errors {s['errors']}, "
         f"artifacts {s['artifacts']}; a property matters when the 95 % object-bootstrap lower bound of its mean EVPI "
         f"is above {s['bar']}", ""]
    for table, label in (("evpi", "all cases"), ("evpi_ess_ok", f"cases with ESS >= {s['ess_table']:g}")):
        L += [f"## Mean EVPI per reveal ({label})", "", "| setting | prior | n | " + " | ".join(REVEALS) + " |",
              "|---|---|---|" + "---|" * len(REVEALS)]
        for setting, block in s["cells"].items():
            for prior, c in block.items():
                n = c["n"] if table == "evpi" else c["n_ess_ok"]
                L.append(f"| {setting} | {prior} | {n} | " + " | ".join(f"{_f(c[table][S])} {c[table][S]['reading'] or ''}"
                                                                     for S in REVEALS) + " |")
        L.append("")
    L += ["## Groups (from the choices)", "", "| setting | prior | n | invariant | critical | abstain-dominated (overlay) |",
          "|---|---|---|---|---|---|"]
    for setting, block in s["cells"].items():
        for prior, c in block.items():
            g = c["groups"] or {}
            L.append(f"| {setting} | {prior} | {c['n']} | {g.get('invariant', 0):.3f} | {g.get('critical', 0):.3f} | "
                     f"{g.get('abstain_dominated', 0):.3f} |")
    L += ["", "## Checks", "", "```", json.dumps(s["checks"], indent=1), "```", ""]
    return "\n".join(L)


def _comgap(path: str | None) -> dict[tuple, dict[str, Any]] | None:
    if not path:
        return None
    op = gzip.open if path.endswith(".gz") else open
    out = {}
    with op(path, "rt") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[(r["object"], r["idx"], r["prior"], r["setting"])] = r
    return out


def summary(a: argparse.Namespace) -> dict[str, Any]:
    recs, cfg = load(a.dirs.split(","))
    out = summarize(recs, _comgap(a.comgap), a.n_boot, a.seed)
    os.makedirs(a.out, exist_ok=True)
    md = markdown(out)
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write(md)
    print(md, flush=True)
    if a.wandb:
        conf = {k: v for k, v in cfg.items() if k not in ("generic", "cases", "spreads")}
        flat = {}
        for setting, block in out["cells"].items():
            for prior, c in block.items():
                for S in REVEALS:
                    flat[f"{setting}/{prior}/evpi_{S}"] = c["evpi"][S]["mean"]
                    flat[f"{setting}/{prior}/evpi_{S}_lo"] = None if c["evpi"][S]["ci"] is None else c["evpi"][S]["ci"][0]
                flat.update({f"{setting}/{prior}/share_{g}": (c["groups"] or {}).get(g) for g in ("invariant", "critical", "abstain_dominated")})
        flat["replay_mismatch"] = (out["checks"].get("replay") or {}).get("mismatch")
        out["wandb"] = tracking.log_run("evpi_props", conf, flat)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def main(argv: list[str] | None = None) -> Any:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bundle", required=True, help="the m2b2 bundle (its folds/ hold the fold bundles)")
    r.add_argument("--scenes", required=True, help="the scenes file of the generic prior's table (first_menus)")
    r.add_argument("--settings", default=None, help="name:care:A,... (default SETTINGS and LOW_STAKES)")
    r.add_argument("--priors", default=None, help="class,generic (default PRIORS)")
    r.add_argument("--objects", default=None)
    r.add_argument("--seed", type=int, default=11)
    r.add_argument("--repeats", type=int, default=11)
    r.add_argument("--sample", type=int, default=None, help="a fixed draw of this many cases (the timing run)")
    r.add_argument("--sample-seed", type=int, default=0)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--out", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--dirs", required=True, help="the run's shard dirs, comma-separated")
    s.add_argument("--out", required=True)
    s.add_argument("--comgap", default=None, help="com_gap's records (jsonl or jsonl.gz): the replay check")
    s.add_argument("--n-boot", type=int, default=N_BOOT)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--wandb", action="store_true")
    a = ap.parse_args(argv)
    return {"run": run, "summary": summary}[a.cmd](a)


if __name__ == "__main__":
    main()
    sys.exit(0)
