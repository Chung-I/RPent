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


"""TODO 6 step 1: the oracle gap for the centre of mass at point (b), the end of the first grasp's hold (daily-logs
researches/property-belief-manipulation/experiments/2026-11-05-com-oracle-gap, with its Amendments 1 and 2).

Per W0 case (calibrate's cross-fit cases; first_menus' reset), prior (class, generic) and setting (care κ, abstain cost
A): the rule's own policy protocol (calibrate.run_protocol "policy": decline, probe, or execute the best row) up to its
first attempt (first_attempt). The attempt runs as tools._attempt runs it (env.execute) with a second readout beside the
adapter's own: the joint fit on the lift and hold steps alone (GraspTap). A case that ends before an attempt, or whose
attempt does not lift the object and hold it through the hold, gets EVPI 0.

Beliefs at (b), each particle one draw of the generator (no key is overwritten):
  b1    the belief before the attempt (the prior and the probes' evidence) times the lift-and-hold fit
        (identify_fit.fit_loglik); no outcome label and no transport data;
  b2    b1 times N(CoM fraction; truth, EXACT_SD["com"]^2), as importance weights;
  yard  b1 times N(log mass; log truth, EXACT_SD["mass"]^2) times the CoM term: the primary yardstick;
  w0    W0's exact_both prior (first_menus.exact_priors), drawn on W0's menu stream: a secondary yardstick.
A belief under MIN_ESS effective particles is refreshed from the prior with all its evidence (belief.refresh), as
tools.update_belief refreshes; the ESS of every conditioned belief is recorded.

Choices at (b) (finishing_rows): the executed pinch, force and lift speed times the carry {standard, slow (slow_lift's
carry), gentle (gentle_carry's a_transport)} times the places the menu offers (the requested place, inboard_place,
drop_release), priced as build_menu prices a row (price_rows) on the plans and poses captured before the attempt
(Scene). EVPI_i = yard(b1's choice) - yard(b2's choice). Checks: w0 as the yardstick, the choices without inboard_place,
and the full menu at (b) (full_menu). Summary: per setting and prior the mean EVPI over all cases, H(c) = mean
max(0, EVPI_i - c), object-bootstrap intervals, the stop rule on the class prior (verdict), the share of cases that reach
(b), the ESS distribution and the table of the cases with an ESS of at least ESS_TABLE.

    python -P -m robots.mujoco_lab.com_gap run --bundle robots/mujoco_lab/data/m2b2_v12 --scenes S --out DIR [--sample 20]
    python -P -m robots.mujoco_lab.com_gap summary --dir DIR [--wandb]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from robots.mujoco_lab import tracking
from robots.mujoco_lab.adapter import JOINT_RISE_M
from robots.mujoco_lab.identify_fit import fit_loglik
from robots.mujoco_lab.vision_prior import EXACT_SD
from rpent.skills.pick_place.belief import MIN_ESS, REFRESH_DRAWS, ParticleBelief, refresh
from rpent.skills.pick_place.costs import expected_cost
from rpent.skills.pick_place.destinations import DESTINATIONS, edge_distance
from rpent.skills.pick_place.features import extract
from rpent.skills.pick_place.menu import (A_GENTLE, EDGE_RISK_M, PLACEHOLDER, STD, _rows, build_menu, estimate, mid_proposal,
                                         pinch_limit, usable_proposals)
from rpent.skills.pick_place.probes import DAMAGED
from rpent.skills.pick_place.tools import MAX_PROBES, assess_pick, entry_loglik, pick_and_place

SETTINGS = (("k100_A3", 100.0, 3.0), ("k30_A3", 30.0, 3.0), ("k100_A10", 100.0, 10.0))
PRIORS = ("class", "generic")
PRIMARY_PRIOR = "class"
H_COSTS = (0.05, 0.10, 0.30)      # 1 s, 2 s and the 6 s identify charge at 0.05 per second
STOP_BOUND = 0.02                 # the stop rule's bar on H(0.05) per case (pre-registered)
ESS_TABLE = 100.0                 # Amendment 2's secondary table: the cases with at least this ESS in b2 and in the yardstick
N_BOOT = 2000
GRASP_FIT, TRUTH = "grasp_fit", "truth"     # the history entries com_gap adds: the lift-and-hold fit, a truth term
REFRESH_STREAM = 3                # calibrate._stream k of the refreshes at (b) (1: the menus, 2: the random row)
HELD_TO_B = ("lift", "hold")      # a loss in these phases: the object is not in the hand at (b)
BASE_S = 8.0                      # commit_std's time; the executed grasp's own charge is the same for every choice at (b)
SLOW_LIFT = {"v_lift": 0.05, "v_transport": 0.2}   # slow_lift's lift and carry
SLOW_EXTRA_S = 6.0                # slow_lift's time over commit_std's (14 s against 8 s)
CARRIES = ("standard", "slow", "gentle")
CARRY_S = {"standard": 0.0, "gentle": 1.0}         # gentle_carry's 9 s; the slow carry's charge is slow_carry_s
PLACE_S = {"requested": 0.0, "inboard_place": 1.0, "drop_release": -2.0}   # inboard_place 9 s, drop_release 6 s


# ---------------------------------------------------------------- the lift-and-hold fit

def grasp_steps(trace: list[dict[str, Any]]) -> list[int]:
    """The attempt's joint window (adapter._attempt_joint) without the carry: the lift once the hand rose JOINT_RISE_M
    above where it closed, and the hold; none without a close."""
    closes = [x["hand_z"] for x in trace if x["phase"] == "close"]
    if not closes:
        return []
    z0 = closes[-1]
    return [i for i, x in enumerate(trace) if x["phase"] == "hold" or (x["phase"] == "lift" and x["hand_z"] - z0 >= JOINT_RISE_M)]


class GraspTap:
    """While open, each call of ``env._attempt_joint`` (execute makes it once the object rose) also records the joint fit
    on the lift and hold steps alone (``env._joint_fit(log, grasp_steps(trace), 4)``) as ``fit`` and its hidden record as
    ``record``, and counts the call; the attempt's own readout comes back unchanged. The fit draws no random numbers
    (ident.segment adds no noise at noise_nm 0). On exit the env's own method is back."""

    def __init__(self, env: Any):
        self.env, self.fit, self.record, self.calls = env, None, None, 0

    def __enter__(self) -> GraspTap:
        env, inner = self.env, self.env._attempt_joint
        self._had, self._prev = "_attempt_joint" in vars(env), vars(env).get("_attempt_joint")

        def tapped(log, trace):
            self.fit, self.record = env._joint_fit(log, grasp_steps(trace), 4)
            self.calls += 1
            return inner(log, trace)

        env._attempt_joint = tapped
        return self

    def __exit__(self, *exc: Any) -> bool:
        if self._had:
            self.env._attempt_joint = self._prev
        else:
            del self.env._attempt_joint
        return False


# ---------------------------------------------------------------- the conditioned beliefs

def fit_entry(fit: dict[str, Any]) -> dict[str, Any]:
    return {"type": GRASP_FIT, "fit": fit}


def truth_entry(truth: dict[str, Any], keys: tuple[str, ...] | list[str]) -> dict[str, Any]:
    """A truth term on ``keys`` (mass, com) at the case's generator truth {mass, com_frac}."""
    return {"type": TRUTH, "truth": {"mass": float(truth["mass"]), "com_frac": float(truth["com_frac"])}, "keys": list(keys)}


def truth_loglik(arrays: dict[str, np.ndarray], truth: dict[str, Any], keys) -> np.ndarray:
    """The tight likelihood around the truth, W0's exact spreads: N(log mass; log truth, EXACT_SD["mass"]^2) for "mass",
    N(CoM fraction; truth, EXACT_SD["com"]^2) for "com"; another key is a ValueError."""
    bad = set(keys) - {"mass", "com"}
    if bad:
        raise ValueError(f"unknown truth keys {sorted(bad)}")
    ll = np.zeros(len(arrays["mass"]))
    if "mass" in keys:
        ll = ll - 0.5 * ((np.log(arrays["mass"]) - np.log(float(truth["mass"]))) / EXACT_SD["mass"]) ** 2
    if "com" in keys:
        ll = ll - 0.5 * ((np.asarray(arrays["com_frac"], dtype=float) - float(truth["com_frac"])) / EXACT_SD["com"]) ** 2
    return ll


def replay_fn(prims: Any) -> Callable[[ParticleBelief, dict[str, Any]], np.ndarray]:
    """The log-likelihood of one history entry on any particles: a lift-and-hold fit, a truth term, or the skill's own
    entry with its recorded terms (tools.entry_loglik, as update_belief's refresh replays it; needs ``prims``)."""

    def replay(b: ParticleBelief, e: dict[str, Any]) -> np.ndarray:
        if e["type"] == GRASP_FIT:
            return fit_loglik(e["fit"], b.arrays())
        if e["type"] == TRUTH:
            return truth_loglik(b.arrays(), e["truth"], e["keys"])
        if prims is None:
            raise ValueError(f"a {e['type']} entry needs the primitives' models")
        return entry_loglik(b, e, prims, terms=e["used_terms"])[0]

    return replay


def condition(belief: ParticleBelief, entries: list[dict[str, Any]], replay: Callable, prior_fn: Callable | None = None,
              rng: np.random.Generator | None = None, n_draws: tuple[int, ...] = REFRESH_DRAWS) -> tuple[ParticleBelief, dict[str, Any]]:
    """``belief`` times the entries' likelihoods as importance weights, the entries appended to its history. Under MIN_ESS
    effective particles, with ``prior_fn(n, rng)``, the belief is refreshed from the prior with all its history
    (belief.refresh), as tools.update_belief refreshes. Returns (belief, info): ess (the effective particles of the
    weights, before any refresh), refreshed, importance_ess (the refresh's), ess_final, ancestors, degenerate."""
    ll = np.zeros(len(belief))
    for e in entries:
        ll = ll + np.asarray(replay(belief, e), dtype=float)
    post = belief.reweight(ll) if entries else belief._replace()
    post = post._replace(history=list(post.history) + list(entries))
    ess = post.effective_size()
    refreshed = bool(ess < MIN_ESS and prior_fn is not None)
    out = refresh(post, prior_fn, replay, rng, n_draws=n_draws, n_keep=len(belief)) if refreshed else post
    return out, {"ess": float(ess), "refreshed": refreshed, "importance_ess": out.importance_ess if refreshed else None,
                 "ess_final": float(out.effective_size()), "ancestors": int(out.n_ancestors()), "degenerate": bool(out.degenerate)}


# ---------------------------------------------------------------- the choices at (b)

def _key(x: Any) -> float | None:
    return None if x is None else round(float(x), 9)


def place_options(destination: str) -> list[tuple[str, str, float | None]]:
    """(place, destination, release height) of the places the menu offers for ``destination`` (menu._rows' rule): the
    requested one; inboard_place when it is none or a surface within EDGE_RISK_M of the table's edge; drop_release for a
    rim, cavity or open bin."""
    dest = DESTINATIONS[destination]
    out = [("requested", destination, None)]
    if dest["kind"] == "none" or (dest["kind"] == "surface" and edge_distance(dest) < EDGE_RISK_M):
        out.append(("inboard_place", "table_centre", None))
    if dest["kind"] in ("rim", "cavity", "open_bin"):
        out.append(("drop_release", destination, 0.05 + float(dest["rim_height"])))
    return out


CARRY_ACTIONS = {"standard": {"v_transport": STD["v_transport"], "a_transport": STD["a_transport"]},
                 "slow": {"v_transport": SLOW_LIFT["v_transport"], "a_transport": STD["a_transport"]},
                 "gentle": {"v_transport": STD["v_transport"], "a_transport": A_GENTLE}}


def finishing_rows(executed: dict[str, Any], destination: str, slow_s: dict[str, float]) -> list[dict[str, Any]]:
    """The choices at (b): the executed pinch (s), force and lift speed times CARRIES times place_options. Each row: id
    "carry|place", action, cost_s = BASE_S plus the carry's charge (``slow_s[the place's destination]`` for the slow
    carry) and the place's (PLACE_S)."""
    rows = []
    for place, dest_id, rh in place_options(destination):
        for carry in CARRIES:
            charge = float(slow_s[dest_id]) if carry == "slow" else CARRY_S[carry]
            action = {"force": float(executed["force"]), "v_lift": float(executed["v_lift"]), **CARRY_ACTIONS[carry],
                      "destination": dest_id, "release_height": rh, "s": float(executed["s"])}
            rows.append({"id": f"{carry}|{place}", "action": action, "cost_s": BASE_S + charge + PLACE_S[place]})
    return rows


def _phase_steps(planned: list[Any]) -> Counter:
    c: Counter = Counter()
    for seg in planned:
        d = seg.to_dict() if hasattr(seg, "to_dict") else seg
        c[d["phase"]] += int(d["n_steps"])
    return c


def slow_carry_s(plan_fn: Callable, prop: Any, destination: str, release_height: float | None = None) -> float:
    """The slow carry's charge: slow_lift's SLOW_EXTRA_S times the planned carry's share of slow_lift's planned extra time
    over commit_std (lift and carry), at ``prop`` and ``destination``."""
    std = _phase_steps(plan_fn(prop, STD["v_lift"], STD["v_transport"], destination, release_height, a_transport=STD["a_transport"]))
    slow = _phase_steps(plan_fn(prop, SLOW_LIFT["v_lift"], SLOW_LIFT["v_transport"], destination, release_height,
                                a_transport=STD["a_transport"]))
    d_lift, d_carry = slow["lift"] - std["lift"], slow["transport"] - std["transport"]
    if d_lift + d_carry <= 0:
        raise ValueError(f"slow_lift plans no extra time at {destination}: lift {d_lift}, carry {d_carry} steps")
    return SLOW_EXTRA_S * d_carry / (d_lift + d_carry)


def gap(c1: dict[str, float], c2: dict[str, float], yard: dict[str, float], exclude=()) -> dict[str, Any]:
    """Each belief's choice (its cheapest row, the first on a tie) priced on the yardstick: evpi = yard(choice1) -
    yard(choice2); it is negative when b2's choice costs more there."""
    ids = [k for k in c1 if k not in set(exclude)]
    a, b = min(ids, key=lambda k: c1[k]), min(ids, key=lambda k: c2[k])
    return {"choice1": a, "choice2": b, "evpi": float(yard[a] - yard[b]), "changed": a != b}


# ---------------------------------------------------------------- the scene captured before the attempt

def _plan_key(s, v_lift, v_transport, destination, release_height, a_transport) -> tuple:
    return (_key(s), _key(v_lift), _key(v_transport), destination, _key(release_height), _key(a_transport))


@dataclass
class Scene:
    """What point (b) reads from the lab, captured before the attempt moves the object (plan_trajectory and
    proposal_pose read the current rest): the geometry, the pose of the case's start (the prior's input), the reachable
    proposals, each usable one's pose and the plans of every action that the choices at (b) and the full menu need. A
    plan, pose or proposal not captured is a KeyError: nothing at (b) reads the lab after the attempt."""

    geometry: Any
    pose_info: dict[str, Any]
    proposals: list[Any]
    max_open: float
    f_std: float
    poses: dict[float, dict[str, Any]]
    plans: dict[tuple, list[dict[str, Any]]]

    def proposal(self, s: float) -> Any:
        for p in self.proposals:
            if abs(float(p.s) - float(s)) < 1e-9:
                return p
        raise KeyError(f"no captured proposal at s {s}")

    def pose(self, prop: Any) -> dict[str, Any]:
        return self.poses[_key(prop.s)]

    def plan(self, prop: Any, v_lift: float, v_transport: float, destination: str, release_height: float | None = None,
             a_transport: float = STD["a_transport"]) -> list[dict[str, Any]]:
        return self.plans[_plan_key(prop.s, v_lift, v_transport, destination, release_height, a_transport)]


def capture_scene(env: Any, destination: str, f_std: float, executed: dict[str, Any]) -> Scene:
    """The Scene of the case's state now, for the row ``executed`` (an action with s): the proposals that reach
    ``destination`` (as tools._assess filters them), the poses of the usable ones and the plans of (1) every menu row at
    the middle pinch (menu._rows with the damage rows), (2) the standard action at every usable pinch (regrasp_com's,
    under any belief), (3) every choice at (b) and (4) the slow carry's two plans per place, at the executed pinch."""
    geometry, pose_info = env.geometry(), env.pose_info()
    proposals = env.proposals()
    reach = getattr(env, "reach", None)
    if reach is not None:
        proposals = [p for p in proposals if reach(p, destination) is None]
    max_open = pinch_limit(env)
    usable = usable_proposals(proposals, max_open)
    poses = {_key(p.s): env.proposal_pose(p) for p in usable}
    need: dict[tuple, tuple] = {}

    def add(p, v_lift, v_transport, dest, rh, a):
        need.setdefault(_plan_key(p.s, v_lift, v_transport, dest, rh, a), (p, v_lift, v_transport, dest, rh, a))

    if usable:
        mid = mid_proposal(usable)
        for _, _, prop, act, _, _ in _rows(usable, f_std, destination, np.asarray(mid.point[:2], dtype=float), damage=True):
            add(prop, act["v_lift"], act["v_transport"], act["destination"], act.get("release_height"), act["a_transport"])
        for p in usable:
            add(p, STD["v_lift"], STD["v_transport"], destination, None, STD["a_transport"])
    ex = next((p for p in usable if abs(float(p.s) - float(executed["s"])) < 1e-9), None)
    if ex is None:
        raise KeyError(f"the executed pinch s {executed['s']} is not among the usable proposals")
    for r in finishing_rows(executed, destination, {d: 0.0 for _, d, _ in place_options(destination)}):
        a = r["action"]
        add(ex, a["v_lift"], a["v_transport"], a["destination"], a["release_height"], a["a_transport"])
    for _, dest_id, _ in place_options(destination):
        add(ex, STD["v_lift"], STD["v_transport"], dest_id, None, STD["a_transport"])
        add(ex, SLOW_LIFT["v_lift"], SLOW_LIFT["v_transport"], dest_id, None, STD["a_transport"])
    plans = {k: [s.to_dict() if hasattr(s, "to_dict") else s for s in env.plan_trajectory(p, vl, vt, d, rh, a_transport=a)]
             for k, (p, vl, vt, d, rh, a) in need.items()}
    return Scene(geometry=geometry, pose_info=pose_info, proposals=list(proposals), max_open=float(max_open), f_std=float(f_std),
                 poses=poses, plans=plans)


def scene_slow_s(scene: Scene, executed: dict[str, Any], destination: str) -> dict[str, float]:
    """slow_carry_s at the executed pinch for each place's destination."""
    prop = scene.proposal(executed["s"])
    return {d: slow_carry_s(scene.plan, prop, d, None) for _, d, _ in place_options(destination)}


def menu_row(o: Any) -> dict[str, Any]:
    """A menu Option as a row of price_rows: id, action (with s), cost_s."""
    return {"id": o.id, "action": dict(o.action), "cost_s": float(o.cost_s)}


def price_rows(model: Any, belief: ParticleBelief, scene: Scene, rows: list[dict[str, Any]], care: str | float, A: float | None,
               reset_s: float | None = None, reset_price: str = "time") -> dict[str, float]:
    """{row id: expected cost} under ``belief``, as build_menu prices a row: the row's plan and pose, the base features at
    menu.PLACEHOLDER, menu.estimate, costs.expected_cost with the abstain cost and the recovery mode."""
    out = {}
    for r in rows:
        act = r["action"]
        prop = scene.proposal(act["s"])
        planned = scene.plan(prop, act["v_lift"], act["v_transport"], act["destination"], act.get("release_height"),
                             a_transport=act["a_transport"])
        base = extract(scene.geometry, prop.point, prop.yaw, PLACEHOLDER["mass"], PLACEHOLDER["mu"], act["force"], act["v_lift"],
                       act["v_transport"], act["destination"], act.get("release_height"), belief.visibility,
                       a_transport=act["a_transport"])
        pred, _ = estimate(model, belief, base, scene.pose(prop), act["force"], planned)
        out[r["id"]] = expected_cost(pred, r["cost_s"], care, A, reset_s, reset_price)
    return out


def point_b(model: Any, b1: ParticleBelief, scene: Scene, rows: list[dict[str, Any]], truth: dict[str, Any], care: str | float,
            A: float | None, replay: Callable, prior_fn: Callable | None = None, rng: np.random.Generator | None = None,
            w0: ParticleBelief | None = None, reset_s: float | None = None, reset_price: str = "time",
            n_draws: tuple[int, ...] = REFRESH_DRAWS, beliefs: dict[str, Any] | None = None) -> dict[str, Any]:
    """The choices at (b) under b1, b2 (b1 and the true CoM), the yardstick (b1, the true mass and CoM) and W0's exact_both
    (``w0``, when given): every row's cost under each, the gap on the yardstick (primary), the same without inboard_place,
    the gap on w0, and the conditioning's ESS. ``beliefs`` (a dict) receives b2 and the yardstick."""
    b2, i2 = condition(b1, [truth_entry(truth, ("com",))], replay, prior_fn, rng, n_draws)
    yard, iy = condition(b1, [truth_entry(truth, ("mass", "com"))], replay, prior_fn, rng, n_draws)
    if beliefs is not None:
        beliefs.update(b2=b2, yard=yard)
    named = {"b1": b1, "b2": b2, "yard": yard, **({"w0": w0} if w0 is not None else {})}
    costs = {k: price_rows(model, b, scene, rows, care, A, reset_s, reset_price) for k, b in named.items()}
    inboard = [r["id"] for r in rows if r["id"].endswith("|inboard_place")]
    out = {"costs": costs, "ess": {"b2": i2, "yard": iy}, "primary": gap(costs["b1"], costs["b2"], costs["yard"]),
           "no_inboard": gap(costs["b1"], costs["b2"], costs["yard"], exclude=inboard)}
    if w0 is not None:
        out["w0"] = gap(costs["b1"], costs["b2"], costs["w0"])
    return out


def full_menu(model: Any, b1: ParticleBelief, b2: ParticleBelief, yard: ParticleBelief, scene: Scene, destination: str,
              f_std: float, care: str | float, A: float | None, reset_s: float | None = None, reset_price: str = "time",
              rng: np.random.Generator | None = None) -> dict[str, Any]:
    """Amendment 1's secondary check: the full menu at (b) (build_menu on the captured scene, no probe rows, no guard)
    under b1 and under b2, each belief's cheapest commit or lower-risk row (inboard_place among them) priced on the
    yardstick as an action (its own pinch and action, not the yardstick's row of that id)."""
    rng = rng if rng is not None else np.random.default_rng(0)

    def best(b: ParticleBelief) -> dict[str, Any]:
        opts, _ = build_menu(model, b, scene.geometry, scene.proposals, f_std, care, destination, scene.plan, rng, pose_fn=scene.pose,
                             max_open=scene.max_open, abstain_cost=A, reset_s=reset_s, reset_price=reset_price, n=len(b))
        return menu_row(min((o for o in opts if o.kind in ("commit", "lower_risk")), key=lambda o: o.expected_cost))

    r1, r2 = best(b1), best(b2)
    y = price_rows(model, yard, scene, [dict(r1, id="choice1"), dict(r2, id="choice2")], care, A, reset_s, reset_price)
    return {"choice1": r1["id"], "choice2": r2["id"], "s1": r1["action"]["s"], "s2": r2["action"]["s"],
            "evpi": float(y["choice1"] - y["choice2"]), "changed": (r1["id"], r1["action"]) != (r2["id"], r2["action"])}


# ---------------------------------------------------------------- the policy up to its first attempt

def first_attempt(prims: Any, destination: str, care: str | float, store: dict[str, Any], max_probes: int = MAX_PROBES) -> dict[str, Any]:
    """calibrate.run_protocol's "policy" loop up to its first attempt, which it does not run: the first recommendation,
    best option, guard and first menu, the probes it runs (pick_and_place; recorded as run_protocol records them), and how
    it ends: "attempt" (with the row and the assessment payload it came from) or run_protocol's own end (abstained,
    no_row, error, unreachable, damaged, thrown). ``moved``: a probe ran, so the lab is no longer at the case's reset."""
    from robots.mujoco_lab import calibrate as cal

    out: dict[str, Any] = {"ended": None, "recommendation": None, "best_option": None, "guard": None, "first_menu": None,
                           "probes": [], "restores": [], "moved": False, "row": None, "assessment": None}

    def end(reason: str, error: str | None = None) -> dict[str, Any]:
        out["ended"] = reason
        if error is not None:
            out["error"] = error
        return out

    def call(**kw) -> dict[str, Any]:
        try:
            return pick_and_place(object="object", destination=destination, care=care, primitives=prims, store=store, **kw).data
        except Exception as exc:  # noqa: BLE001  one bad case must not stop the run
            return {"error": str(exc)[:200]}

    a = assess_pick(object="object", destination=destination, care=care, primitives=prims, store=store).data
    if "error" in a:
        return end("error", a["error"])
    if not a.get("assessed", True):
        return end("unreachable")
    out["recommendation"], out["best_option"], out["guard"] = a.get("recommendation"), a.get("best_option"), a.get("guard")
    out["first_menu"] = cal._first_menu(a)
    while True:
        rec = a.get("recommendation")
        if rec == "probe" and len(out["probes"]) >= max_probes:
            rec = cal._without_probe(a, prims)
        if rec == "decline":
            return end("abstained")
        if rec == "probe":
            out["moved"] = True
            d = call(option_id=a["recommended_option"])
            if "error" in d:
                return end("error", d["error"])
            out["probes"].append(cal._probe_record(d, prims, care, len(out["probes"]), 0))
            if d.get("restored") is not None:
                out["restores"].append(dict(d["restored"], probe=d["probed"]))
            elif d["kind"] == DAMAGED or d.get("damage"):
                return end("damaged")
            elif d["thrown"]:
                return end("thrown")
            a = d["assessment"]
            if "error" in a:
                return end("error", a["error"])
            if not a.get("assessed", False):
                return end("unreachable")
            continue
        row = next((o for o in a["options"] if o["id"] == a.get("best_option")), None)
        if row is None:
            return end("no_row")
        out["row"], out["assessment"] = row, a
        return end("attempt")


# ---------------------------------------------------------------- one case

def fingerprint(env: Any) -> str:
    """A hash of the lab's state (qpos, qvel, the pose record): two resets to the same draw must agree."""
    h = hashlib.sha256()
    d = getattr(getattr(env, "sim", None), "d", None)
    if d is not None:
        h.update(np.round(np.asarray(d.qpos, dtype=float), 9).tobytes())
        h.update(np.round(np.asarray(d.qvel, dtype=float), 9).tobytes())
    h.update(json.dumps(env.pose_info(), sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def _fit_summary(fit: dict[str, Any] | None) -> dict[str, Any] | None:
    if not fit:
        return None
    return {"window_s": fit.get("window_s"), "n": fit.get("n"), "phases": fit.get("phases"), "phi4": list(fit["phi"][:4])}


def case_b(prims: Any, env: Any, o: str, destination: str, care: float, A: float, prior_fn: Callable, truth: dict[str, Any],
           w0: ParticleBelief | None, rng: np.random.Generator, full: bool = True) -> dict[str, Any]:
    """One case under one prior and setting (``prims`` holds them, a fresh menu stream, the lab at the case's reset): the
    policy to its first attempt, the attempt with the lift-and-hold fit, and point (b)."""
    from robots.mujoco_lab.adapter import A_TRANSPORT
    from rpent.skills.pick_place.tools import _menu_recovery, _spread_kw, standard_force

    store: dict[str, Any] = {}
    fa = first_attempt(prims, destination, care, store)
    out: dict[str, Any] = {"ended": fa["ended"], "recommendation": fa["recommendation"], "best_option": fa["best_option"],
                           "guard_flag": bool((fa["guard"] or {}).get("flag")), "first_menu": fa["first_menu"],
                           "probes": [p["probe"] for p in fa["probes"]], "probe_kinds": [p["kind"] for p in fa["probes"]],
                           "restores": len(fa["restores"]), "moved": fa["moved"], "reached": False, "evpi": 0.0, "evpi_w0": 0.0,
                           "evpi_no_inboard": 0.0, "evpi_full": 0.0, "changed": False}
    if "error" in fa:
        out["error"] = fa["error"]
    if fa["ended"] != "attempt":
        return out
    cache = list(store.get("pick_place", {}).values())
    if len(cache) != 1:
        raise RuntimeError(f"{len(cache)} cached assessments at the first attempt (one expected)")
    option, pre = cache[0]["menu"][fa["row"]["id"]], cache[0]["belief"]
    act = option.action
    f_std = standard_force(env.nominal_mass(o))
    scene = capture_scene(env, destination, f_std, act)
    prop = next(p for p in env.proposals() if abs(p.s - act["s"]) < 1e-9)       # as tools._attempt finds it
    out["moved"] = True
    with GraspTap(env) as tap:
        r = env.execute(prop, act["force"], act["v_lift"], act["v_transport"], act["destination"], act.get("release_height"),
                        a_transport=act.get("a_transport", A_TRANSPORT))
    reached = tap.calls > 0 and r.loss_phase not in HELD_TO_B
    out.update(executed=option.id, action=act, label=r.label, loss_phase=r.loss_phase, rose=tap.calls > 0,
               artifact=bool(getattr(r, "artifact", False)), fit=_fit_summary(tap.fit), reached=bool(reached),
               pre_ess=float(pre.effective_size()))
    if not reached:
        return out
    replay = replay_fn(prims)
    spread = _spread_kw(prims)

    def draw(n: int, g: np.random.Generator) -> ParticleBelief:
        return prior_fn(scene.pose_info, n, g, **spread)

    b1, i1 = condition(pre, [fit_entry(tap.fit)] if tap.fit else [], replay, draw, rng)
    rows = finishing_rows(act, destination, scene_slow_s(scene, act, destination))
    reset_s, price = _menu_recovery(prims)
    keep: dict[str, Any] = {}
    pb = point_b(prims.model, b1, scene, rows, truth, care, A, replay, draw, rng, w0=w0, reset_s=reset_s, reset_price=price,
                 beliefs=keep)
    out.update(evpi=pb["primary"]["evpi"], choice1=pb["primary"]["choice1"], choice2=pb["primary"]["choice2"],
               changed=pb["primary"]["changed"], evpi_no_inboard=pb["no_inboard"]["evpi"], no_inboard=pb["no_inboard"],
               evpi_w0=pb["w0"]["evpi"] if "w0" in pb else None, w0=pb.get("w0"), costs=pb["costs"],
               ess_b1=i1, ess_b2=pb["ess"]["b2"]["ess_final"], ess_yard=pb["ess"]["yard"]["ess_final"], ess=pb["ess"],
               slow_s=scene_slow_s(scene, act, destination))
    if full:
        fm = full_menu(prims.model, b1, keep["b2"], keep["yard"], scene, destination, f_std, care, A, reset_s, price, rng)
        out.update(full=fm, evpi_full=fm["evpi"])
    return out


def run_case(prims: Any, o: str, idx: int, s: dict[str, Any], priors: dict[str, Callable], base: Callable,
             world: Any) -> list[dict[str, Any]]:
    """One W0 case: first_menus' reset, the case's truth and W0's exact_both (over the class prior ``base``), then per
    prior and setting case_b on a fresh menu stream (calibrate._stream k 1) and refresh stream (REFRESH_STREAM); the lab
    is reset again after a run that moved it, and every reset is checked against the first (reset_match)."""
    from robots.mujoco_lab import calibrate as cal
    from robots.mujoco_lab.adapter import object_props
    from robots.mujoco_lab.first_menus import exact_priors
    from rpent.skills.pick_place.tools import _particles, _spread_kw

    env, seed = prims.env, int(s["seed"])
    theta, com_end, destination, hidden = cal.case_draw(seed, o, idx, object_props(o)["mass"], False, fragility=world)

    def reset() -> str:
        if world is not None:
            env.slosh_zeta, env.slosh_scale = world[1].slosh_zeta, world[1].slosh_scale
        env.classes = None if world is None else world[0]
        env.reset(o, theta, seed=seed * 1000 + idx, com_end=com_end, fixture=destination if destination in cal.MESH_IDS else None,
                  fill=hidden["fill"], frag=hidden["frag"])
        return fingerprint(env)

    fp0 = reset()
    reset_info = dict(getattr(env, "reset_info", None) or {})
    t, exact = exact_priors(o, base, s["generic"][o], env, theta, com_end, hidden, world)
    truth = {"mass": t["mass"], "com_frac": t["com_frac"]}
    n = _particles(prims)
    w0 = exact["exact_both"](env.pose_info(), n, cal._stream(seed, o, idx, 1), **_spread_kw(prims))
    recs, dirty = [], False
    for pname, fn in priors.items():
        for name, care, A in s["settings"]:
            t0 = time.time()
            match = True
            if dirty:
                match = reset() == fp0
            prims.prior, prims.abstain_cost, prims.rng = fn, A, cal._stream(seed, o, idx, 1)
            rec = {"object": o, "idx": idx, "fold": s["fold"], "prior": pname, "setting": name, "care": care, "abstain_cost": A,
                   "destination": destination, "at_rest": reset_info.get("at_rest", True), "reset_match": match,
                   "flat": s.get("flat"), "truth": t, "com_frac": truth["com_frac"], "n_particles": n,
                   "fill": hidden["fill"]}
            try:
                rec.update(case_b(prims, env, o, destination, care, A, fn, truth, w0, cal._stream(seed, o, idx, REFRESH_STREAM),
                                  full=bool(s.get("full_menu", True))))
            except Exception as exc:  # noqa: BLE001  one bad case must not stop the run
                rec.update(error=f"{type(exc).__name__}: {str(exc)[:300]}", moved=True)
            dirty = bool(rec.get("moved", True))
            rec["seconds"] = round(time.time() - t0, 2)
            recs.append(rec)
    return recs


def run_object(job: tuple[str, list[int], dict[str, Any]]) -> list[dict[str, Any]]:
    """One object's cases on its fold's bundle (calibrate._run_object's sensor and world)."""
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
        for idx in idxs:
            t0 = time.time()
            rows += run_case(prims, o, idx, s, priors, base, world)
            print(f"{o} {idx}: {time.time() - t0:.1f} s", file=sys.stderr, flush=True)
    finally:
        prims.prior = base
    return rows


# ---------------------------------------------------------------- the run

def sample_cases(cases: list[tuple[str, int]], n: int | None, seed: int) -> list[tuple[str, int]]:
    """A fixed draw of n cases without replacement (in the case set's order), or every case for n None."""
    if n is None:
        return list(cases)
    pick = np.sort(np.random.default_rng(seed).choice(len(cases), int(n), replace=False))
    return [cases[int(i)] for i in pick]


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
    tables, spreads = prior_tables(a.bundle, a.scenes, {})
    folds = json.load(open(os.path.join(a.bundle, "folds", "folds.json")))
    flat = {o: bool(v["flat"]) for o, v in json.load(open(a.scenes)).items()}
    objects = a.objects.split(",") if a.objects else sorted(folds)
    cases = sample_cases([(o, i) for o in objects for i in range(a.repeats)], a.sample, a.sample_seed)
    by: dict[str, list[int]] = defaultdict(list)
    for o, i in cases:
        by[o].append(i)
    settings = [_setting(t) for t in a.settings.split(",")] if a.settings else [tuple(x) for x in SETTINGS]
    priors = a.priors.split(",") if a.priors else list(PRIORS)
    with open(os.path.join(a.out, "run_config.json"), "w") as f:
        json.dump({**vars(a), **_revision(), "generator": generator, "settings_parsed": settings, "priors_parsed": priors,
                   "cases": cases, "spreads": spreads, "generic": tables["generic"], "exact_sd": EXACT_SD,
                   "stop_bound": STOP_BOUND, "h_costs": H_COSTS, "ess_table": ESS_TABLE}, f, indent=1)
    jobs = [(o, idxs, {"bundle": os.path.join(a.bundle, "folds", f"fold_{folds[o]}"), "fold": folds[o], "seed": a.seed,
                       "settings": settings, "priors": priors, "generic": {o: tables["generic"][o]}, "flat": flat.get(o),
                       "full_menu": not a.no_full_menu}) for o, idxs in by.items()]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool, open(path, "w") as out:
        for rows in pool.imap_unordered(run_object, jobs):
            for r in rows:
                out.write(json.dumps(r) + "\n")
            out.flush()
    print(f"done in {time.time() - t0:.0f} s: {path}", flush=True)
    return path


# ---------------------------------------------------------------- the summary

def _mean_ci(values: np.ndarray, objects: list[str], rng: np.random.Generator, n_boot: int) -> dict[str, Any]:
    from robots.mujoco_lab.calibrate import _mean_ci as boot

    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return {"mean": None, "ci": None}
    _, inv = np.unique(objects, return_inverse=True)
    return {"mean": float(v.mean()), "ci": boot(v, inv, int(inv.max()) + 1, rng, n_boot)}


def _ess_stats(values: list[float]) -> dict[str, Any] | None:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return None
    return {"n": int(len(v)), "min": float(v.min()), "p10": float(np.percentile(v, 10)), "median": float(np.median(v)),
            "max": float(v.max()), "below_100": int((v < ESS_TABLE).sum())}


def _ess_ok(r: dict[str, Any]) -> bool:
    """Amendment 2's table keeps a case unless it reached (b) with fewer than ESS_TABLE effective particles in b2 or in
    the yardstick (a case that never reached (b) counts 0, as in the primary result)."""
    if not r.get("reached"):
        return True
    return all(r.get(k) is not None and float(r[k]) >= ESS_TABLE for k in ("ess_b2", "ess_yard"))


def _h(evpi: np.ndarray, objects: list[str], rng: np.random.Generator, n_boot: int) -> dict[str, Any]:
    return {str(c): _mean_ci(np.maximum(0.0, evpi - c), objects, rng, n_boot) for c in H_COSTS}


def cell(recs: list[dict[str, Any]], rng: np.random.Generator, n_boot: int = N_BOOT) -> dict[str, Any]:
    """One setting and prior: the mean EVPI over every case without an error (a case that never reached (b) counts 0),
    H(c), the checks (w0, no inboard_place, the full menu), the share that reached (b) and the mean over those, the
    shares with EVPI above 0 and with a changed choice, the ESS distribution and Amendment 2's ESS table."""
    ok = [r for r in recs if "error" not in r]
    objects = [r["object"] for r in ok]
    evpi = np.array([float(r["evpi"]) for r in ok])
    reached = np.array([bool(r["reached"]) for r in ok])
    tab = [r for r in ok if _ess_ok(r)]
    tab_evpi = np.array([float(r["evpi"]) for r in tab])
    return {"n": len(ok), "n_errors": len(recs) - len(ok), "n_artifacts": sum(bool(r.get("artifact")) for r in ok),
            "n_reset_mismatch": sum(not r.get("reset_match", True) for r in ok),
            "n_reached": int(reached.sum()), "share_reached": float(reached.mean()) if len(ok) else None,
            "ended": dict(Counter(str(r.get("ended")) for r in ok)),
            "evpi": _mean_ci(evpi, objects, rng, n_boot), "evpi_reached": float(evpi[reached].mean()) if reached.any() else None,
            "share_positive": float((evpi > 0).mean()) if len(ok) else None,
            "share_changed": float(np.mean([bool(r.get("changed")) for r in ok])) if len(ok) else None,
            "H": _h(evpi, objects, rng, n_boot),
            "checks": {k: _mean_ci(np.array([float(r.get(f"evpi_{k}") or 0.0) for r in ok]), objects, rng, n_boot)
                       for k in ("w0", "no_inboard", "full")},
            "ess": {k: _ess_stats([r.get(f"ess_{k}") for r in ok if r.get("reached")]) for k in ("b2", "yard")},
            "ess_raw": {k: _ess_stats([((r.get("ess") or {}).get(k) or {}).get("ess") for r in ok if r.get("reached")])
                        for k in ("b2", "yard")},
            "ess_table": {"n": len(tab), "evpi": _mean_ci(tab_evpi, [r["object"] for r in tab], rng, n_boot),
                          "H": _h(tab_evpi, [r["object"] for r in tab], rng, n_boot)}}


def verdict(h05: dict[str, dict[str, Any]]) -> str:
    """The pre-registered stop rule on H(0.05) per setting: go when any setting has H(0.05) at least STOP_BOUND with its
    lower bound above 0; stop when every setting's upper bound is below STOP_BOUND; else undecided."""
    if any(c["mean"] is not None and c["mean"] >= STOP_BOUND and c["ci"][0] > 0 for c in h05.values()):
        return "go"
    if all(c["ci"] is not None and c["ci"][1] < STOP_BOUND for c in h05.values()):
        return "stop"
    return "undecided"


def _groups(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """The mean EVPI by group and true CoM: upright objects in terciles of the CoM fraction, flat ones in terciles of
    |c - 0.5|."""
    out = {}
    for group, flat in (("upright", False), ("flat", True)):
        rs = [r for r in recs if "error" not in r and bool(r.get("flat")) == flat]
        if not rs:
            continue
        x = np.array([abs(r["com_frac"] - 0.5) if flat else r["com_frac"] for r in rs], dtype=float)
        edges = np.quantile(x, [1 / 3, 2 / 3])
        bins = np.digitize(x, edges)
        out[group] = [{"tercile": k + 1, "range": [float(x[bins == k].min()), float(x[bins == k].max())], "n": int((bins == k).sum()),
                       "mean_evpi": float(np.mean([rs[i]["evpi"] for i in np.flatnonzero(bins == k)]))}
                      for k in range(3) if (bins == k).any()]
    return out


def _top(recs: list[dict[str, Any]], k: int = 10) -> list[dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        if "error" not in r:
            by[r["object"]].append(r)
    rows = [{"object": o, "n": len(v), "mean_evpi": float(np.mean([r["evpi"] for r in v])),
             "changes": dict(Counter(f"{r['choice1']} -> {r['choice2']}" for r in v if r.get("changed")))} for o, v in by.items()]
    return sorted(rows, key=lambda x: -x["mean_evpi"])[:k]


def summarize(records: list[dict[str, Any]], n_boot: int = N_BOOT, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    by: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        by[r["setting"]][r["prior"]].append(r)
    cells = {s: {p: cell(v, rng, n_boot) for p, v in block.items()} for s, block in by.items()}
    verdicts = {}
    for p in sorted({p for block in cells.values() for p in block}):
        h = {s: block[p]["H"][str(H_COSTS[0])] for s, block in cells.items() if p in block}
        verdicts[p] = verdict(h)
    primary = {s: by[s].get(PRIMARY_PRIOR, []) for s in by}
    return {"n_records": len(records), "cells": cells, "verdict": verdicts, "primary_prior": PRIMARY_PRIOR,
            "groups": {s: _groups(v) for s, v in primary.items()}, "top": {s: _top(v) for s, v in primary.items()}}


def _fmt(c: dict[str, Any]) -> str:
    return "n/a" if c["mean"] is None else f"{c['mean']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]"


def markdown(s: dict[str, Any]) -> str:
    lines = ["| setting | prior | n | reached | EVPI [95 % CI] | H(0.05) | H(0.10) | H(0.30) | w0 check | no inboard | full menu | "
             "ESS table n | ESS table H(0.05) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for setting, block in s["cells"].items():
        for p, c in block.items():
            lines.append(f"| {setting} | {p} | {c['n']} | {c['n_reached']} | {_fmt(c['evpi'])} | {_fmt(c['H']['0.05'])} | "
                         f"{_fmt(c['H']['0.1'])} | {_fmt(c['H']['0.3'])} | {_fmt(c['checks']['w0'])} | {_fmt(c['checks']['no_inboard'])} | "
                         f"{_fmt(c['checks']['full'])} | {c['ess_table']['n']} | {_fmt(c['ess_table']['H']['0.05'])} |")
    v = ", ".join(f"{p}: {x}" for p, x in s["verdict"].items())
    return f"# The CoM oracle gap at point (b)\n\nStop rule (on {s['primary_prior']}): {v}\n\n" + "\n".join(lines) + "\n"


def summary(a: argparse.Namespace) -> dict[str, Any]:
    recs = [json.loads(line) for line in open(os.path.join(a.dir, "records.jsonl"))]
    out = summarize(recs, a.n_boot, a.seed)
    md = markdown(out)
    with open(os.path.join(a.dir, "summary.md"), "w") as f:
        f.write(md)
    print(md, flush=True)
    if a.wandb:
        cfg = {k: v for k, v in json.load(open(os.path.join(a.dir, "run_config.json"))).items() if k not in ("generic", "cases")}
        flat = {f"{setting}/{p}/{m}": v for setting, block in out["cells"].items() for p, c in block.items()
                for m, v in (("evpi", c["evpi"]["mean"]), ("H05", c["H"]["0.05"]["mean"]), ("share_reached", c["share_reached"]))}
        flat.update({f"verdict/{p}": x for p, x in out["verdict"].items()})
        out["wandb"] = tracking.log_run("com_gap", cfg, flat)
    with open(os.path.join(a.dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def main(argv: list[str] | None = None) -> Any:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bundle", required=True, help="the m2b2 bundle (its folds/ hold the fold bundles)")
    r.add_argument("--scenes", required=True, help="the scenes file of the generic prior's table (first_menus)")
    r.add_argument("--settings", default=None, help="name:care:A,... (default SETTINGS)")
    r.add_argument("--priors", default=None, help="class,generic (default PRIORS)")
    r.add_argument("--objects", default=None)
    r.add_argument("--seed", type=int, default=11)
    r.add_argument("--repeats", type=int, default=11)
    r.add_argument("--sample", type=int, default=None, help="a fixed draw of this many cases (the timing run)")
    r.add_argument("--sample-seed", type=int, default=0)
    r.add_argument("--no-full-menu", action="store_true", help="skip the full menu check at (b)")
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
