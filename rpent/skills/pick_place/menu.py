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

"""The option menu: never a bare decline.

Each row is one concrete action (pinch, force, speeds, destination, release height) with
its predicted outcome vector under the belief and its expected cost. Rows that change
the destination (``inboard_place``) are shown but never chosen by default. Predictions are
made for every particle of the belief (the model is conditional on the CoM) and averaged
with the belief's weights; each row keeps its per-particle costs for the probe values. A proposal
whose exact pinch width is over the substrate's opening (``env.max_pinch_width``, else
MAX_OPEN_WIDTH) or nan (the pads close on nothing) gets no row; when no proposal is left the
menu declines and says why ("every proposal is wider than the gripper opening").

Probe rows (push, weigh, and the squeeze for a crushable or bruisable object at ``squeeze_cap``; when the substrate has
probe models) are valued by preposterior analysis: the expected cost of the best continuation row after the probe's
result, over results simulated from the belief (``probe_value``). Every probe may end the case, thrown or damaged
(``terminal_risk``), at the irreversible cost and A. A probe row's expected cost is its time, the push's tip risk and
that expected cost after it; it competes with the other rows in the same ranking. The plug-in value is
estimated per simulated result against the best row now, so it is exactly 0 when no result can
change that row; the held-out value (``evoi="heldout"``, plan 2b2) scores each particle with the row chosen without
its draw (every copy of its ``ancestor``), so it is not biased upward by a chance correlation and can be below 0. A probe is recommended only when its value is
above 0 and its row beats the best row by more than EVOI_Z Monte Carlo errors.

The cost rule (milestone 2b, ``abstain_cost`` a number A): every row's expected cost adds A x (1 - P(placed_stable)),
the cost of leaving the task undone, and abstaining costs A. The recommendation is the cheapest of the best row
("commit"), the cheapest probe row ("probe", when it beats the cheaper of the best row and A by more than EVOI_Z errors)
and abstaining ("decline"); ``threshold`` is then an optional floor on P(placed_stable). A probe's preposterior may
abstain after the probe, and a probe that ends the case costs the irreversible cost and A. Under the recovery mode
(``reset_s``, plan H3 decision 3) an irreversible outcome costs the reset's time, and a probe that throws or damages the
object costs the reset's time plus the best continuation row now (the object is restored; its result is not valued);
the menu stays one-step. ``abstain_cost`` None keeps
the milestone 2a rule: commit when the best row reaches the threshold (default DEFAULT_THRESHOLD), and a decline costs
nothing.

The novelty guard (``guard``, guard.KnnGuard) sees every commit and lower-risk row for every particle. When it flags
the object as unlike the training objects, the menu declines and says why, whatever the threshold and the probe rows
say; the rows stay in the menu, and the decline row keeps the guard's check either way.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from rpent.skills.pick_place.belief import N_PARTICLES, Belief, ParticleBelief
from rpent.skills.pick_place.costs import COST, TIME_COST_PER_S, care_scale, expected_cost, irreversible_cost, particle_costs
from rpent.skills.pick_place.costs import probe_time_scale as _probe_time_scale  # build_menu has a parameter of that name
from rpent.skills.pick_place.destinations import DESTINATIONS, edge_distance
from rpent.skills.pick_place.features import (A_TRANSPORT, DAMAGE_FEATURES, FEATURES_M2, Geometry, entry_flags, extract,
                                              model_names, particle_features, particle_matrix)
from rpent.skills.pick_place.guard import flag_reason
from rpent.skills.pick_place.probes import DAMAGED, PROBE_SECONDS, squeeze_cap, weigh_force
from rpent.skills.pick_place.protocol import IRREVERSIBLE, Option, Prediction

MAX_OPEN_WIDTH = 0.090                       # m: the widest exact pinch (a 9.4 cm open gap less 2 mm a side); a substrate
                                             # with another gripper sets env.max_pinch_width
TOO_WIDE = "wider than the gripper opening"  # a wider pinch lands the pads on the object (the wedge)
OFF_OBJECT = "off the object"                # a nan width: no hull lies between the pads
DEFAULT_THRESHOLD = 0.7
EDGE_RISK_M = 0.08
STD = {"v_lift": 0.15, "v_transport": 0.5, "a_transport": A_TRANSPORT}
A_GENTLE = 0.5                               # m/s^2: the gentle carry (milestone 2b)
PLACEHOLDER = {"mass": 1.0, "mu": 0.5}       # base features carry these; every particle overwrites them
PROBE_V_LIFT = 0.03                          # the probe features, as the probe corpus extracted them
PROBE_DESTINATION = "table_centre"
EVOI_K = 200                                 # simulated probe results per probe row
EVOI_Z = 2.0                                 # a probe is recommended when it beats the best row by more than this many errors
EVOI_EPS = 1e-9
ESTIMATORS = ("plugin", "heldout")           # probe_value's estimators (plan 2b2 decision 2); a bundle declares one
HELDOUT_REST = 1e-12                         # a held-out chooser whose rest of the weight is below this share keeps row b


def unusable_reason(width: float, max_open: float = MAX_OPEN_WIDTH) -> str | None:
    """Why the skill will not pinch a proposal of this exact width (features.pinch_width), or None."""
    if not np.isfinite(width):
        return OFF_OBJECT
    return TOO_WIDE if width > max_open else None


def usable_proposals(proposals: list[Any], max_open: float = MAX_OPEN_WIDTH) -> list[Any]:
    return [p for p in proposals if unusable_reason(p.width, max_open) is None]


def mid_proposal(usable: list[Any]) -> Any:
    """The usable proposal nearest the middle (s 0.5): the pose the belief maps through and the probes' grasp."""
    return min(usable, key=lambda p: abs(p.s - 0.5))


def decline_reason(proposals: list[Any], max_open: float = MAX_OPEN_WIDTH) -> str:
    """The decline when no proposal is usable, e.g. "every proposal is wider than the gripper opening"."""
    why = sorted({r for r in (unusable_reason(p.width, max_open) for p in proposals) if r})
    return "every proposal is " + " or ".join(why)


def pinch_limit(env: Any) -> float:
    """The substrate's widest exact pinch (env.max_pinch_width), else MAX_OPEN_WIDTH."""
    return float(getattr(env, "max_pinch_width", MAX_OPEN_WIDTH))


def geometry_pose(geometry: Geometry, proposal: Any) -> dict[str, Any]:
    """The pose of a substrate without ``proposal_pose``: identity rotation, the CoM given in the world frame."""
    com = np.asarray(geometry.com, dtype=float)
    return {"R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], "p": [0.0, 0.0, 0.0],
            "low_z": float(com[2] - geometry.com_height), "base_extent": float(geometry.base_extent),
            "length": float(geometry.length), "com_obj_nominal": com.tolist(),
            "pt": [float(x) for x in proposal.point], "yaw": float(proposal.yaw)}


def estimate(model: Any, belief: ParticleBelief, base: dict[str, float], pose: dict[str, Any], force: float,
             planned: list[dict[str, Any]], rng: np.random.Generator | None = None) -> tuple[Prediction, list[Prediction]]:
    """(Weighted mean prediction, one prediction per particle) for one action on one pose."""
    hazard = getattr(model, "kind", "classifier") == "hazard"
    if hasattr(model, "predict_matrix"):
        from rpent.skills.pick_place.hazard_model import predictions_from_matrix

        X = particle_matrix(base, belief.arrays(), pose, force, model_names(model))
        P, DH, LT = model.predict_matrix(X, planned) if hazard else model.predict_matrix(X)
        per = predictions_from_matrix(P, DH, LT)
    else:
        feats = [particle_features(base, part, pose, force) for part in belief.particles]
        per = model.predict_many(feats, planned) if hazard else model.predict_many(feats)
    return Prediction.weighted_mean(per, belief.weights), per


def believed_com_world(belief: ParticleBelief, pose: dict[str, Any]) -> np.ndarray:
    R = np.asarray(pose["R"], dtype=float).reshape(3, 3)
    C = belief.arrays()["com_obj"] @ R.T + np.asarray(pose["p"], dtype=float)
    return belief.weights @ C


def _check_estimator(estimator: str) -> str:
    if estimator not in ESTIMATORS:
        raise ValueError(f"unknown probe-value estimator {estimator!r}; known: {', '.join(ESTIMATORS)}")
    return estimator


def heldout_rows(C: np.ndarray, u: np.ndarray, best: int, groups: np.ndarray | None = None) -> np.ndarray:
    """Per particle l, the row chosen by the weights u without l's draw: argmin_r (C_r . u - sum_{l' in g(l)} u_l' C_r,l'),
    one (rows x particles) argmin, where g(l) holds the particles with l's label in ``groups`` (its ancestor: a refresh
    resamples copies of one prior draw, and a copy left with the chooser brings l's chance correlation back); without
    groups each particle is its own draw. Row ``best`` where the rest of u is (numerically) 0, so a chooser with no
    weight never falls back to a prior."""
    own, rest = C * u[None, :], u.sum() - u
    if groups is not None:
        inv = np.unique(np.asarray(groups), return_inverse=True)[1].reshape(-1)
        sums = np.zeros((int(inv.max()) + 1, C.shape[0]))
        np.add.at(sums, inv, own.T)
        own, rest = sums[inv].T, u.sum() - np.bincount(inv, weights=u)[inv]
    a = np.argmin((C @ u)[:, None] - own, axis=0)
    return np.where(rest > HELDOUT_REST * u.sum(), a, best)


def probe_value(belief: ParticleBelief, continuation_costs: list[np.ndarray], probe_model: Any, X_probe: np.ndarray, force: float,
                care: str | float, rng: np.random.Generator, k: int = EVOI_K, abstain_cost: float | None = None,
                estimator: str = "plugin", reading_value: bool = False, reset_s: float | None = None) -> dict[str, Any]:
    """The preposterior value of one probe against the cheapest continuation row now (row b, per-particle
    costs C_b). With an abstain cost A, abstaining is a continuation row that costs A for every particle, and T
    below adds A. The probe ends the case, thrown or damaged, with each particle's risk r (``terminal_risk``; the case
    ends at the irreversible cost T), so the exact cost after the probe when its result is not used is
    sum_i w_i (r_i T + (1 - r_i) C_b,i). For k particles drawn by weight one result is sampled from the
    probe model among the results that keep the object (P(kind | kept): no "damaged" kind), and the belief w' is
    reweighted by the result's likelihood given the object is kept; the result's gain is the posterior saving of the
    row it then picks (a, the cheapest under w'(1 - r), the belief given the result and the object kept) over row b
    where the object is kept: sum_l w'_l (1 - r_l) (C_b,l - C_a,l), 0 whenever the result leaves row b cheapest (and
    for a particle the probe damages for sure). Each particle's survival enters once, in the weight (1 - r_l): the
    likelihood is that of a kept result, so the mean gain is the exact value of the result at temper 1. Returns the belief's P(kind) and P(thrown), ``evoi`` (the mean
    gain, floored at 0), ``evoi_se`` (its Monte Carlo error) and ``expected_cost_after`` (the cost when the result is
    not used, minus the mean gain).

    That plug-in value chooses row a and scores it on the same particles, so it finds value in a chance correlation
    between a reading and the costs (the optimizer's curse). ``estimator`` "heldout" (plan 2b2 decision 2) scores each
    particle l with the row chosen by the posterior without l's draw (``heldout_rows`` over the belief's ``ancestor``
    key: a refreshed belief holds copies of its draws): the result's gain is sum_l w'_l (1 - r_l) (C_b,l - C_a-l,l).
    The target is the population the draws sample: for equally weighted draws from it, the held-out cost after the
    probe is in expectation at least the population's optimal cost after it (up to the self-normalized importance
    bias, O(1/ESS), for a weighted belief). Its ``evoi`` is the raw
    mean (it can be below 0) and ``plugin_evoi`` the plug-in's on the same draws; no extra randomness is drawn.
    ``reading_value`` adds ``reading_gain`` and ``reading_gain_se``: per result, the saving of the row chosen from the
    whole result over the row chosen from its kind and survival only (the likelihood replayed without its reading term;
    0 when the reading term was not used), scored as the estimator scores, with ``plugin_reading_gain`` beside a held-out
    one. With ``reset_s`` (the recovery mode, plan H3 decision 3) the probe that ends the case does not end it: the
    object is restored and the case goes on, so T_i is the reset's time plus C_b,i (row b now, under the pre-probe
    belief). An unknown estimator raises ValueError."""
    _check_estimator(estimator)
    C = np.asarray(continuation_costs, dtype=float).reshape(len(continuation_costs), -1)
    if abstain_cost is not None:
        C = np.vstack([C, np.full(C.shape[1], float(abstain_cost))])
    w = belief.weights
    best = int(np.argmin(C @ w))
    thrown = np.asarray(probe_model.risk(X_probe), dtype=float)
    risk = np.asarray(probe_model.terminal_risk(X_probe), dtype=float)
    P = np.asarray(probe_model.p_kind(X_probe), dtype=float)
    kept = P.copy()                                  # the results that keep the object: the damaged kind is in the risk
    if DAMAGED in probe_model.classes:
        kept[:, list(probe_model.classes).index(DAMAGED)] = 0.0
    total = kept.sum(axis=1, keepdims=True)
    Q = np.divide(kept, total, out=np.zeros_like(kept), where=total > 0)     # P(kind | the object is kept)
    a = belief.arrays()
    keep = 1.0 - risk
    if reset_s is None:
        terminal = COST["dropped_high"] * care_scale(care) + (0.0 if abstain_cost is None else float(abstain_cost))
    else:
        terminal = irreversible_cost(care, reset_s) + C[best]          # restored: the reset, then row b (per particle)
    unused = float(w @ (risk * terminal + keep * C[best]))
    heldout = estimator == "heldout"
    groups = a.get("ancestor")                       # copies of one draw are held out together (heldout_rows);
    if groups is not None and len(np.unique(groups)) == len(groups):
        groups = None                                # distinct draws (a first menu): the per-particle rows, exactly
    cols = np.arange(C.shape[1])
    gains, held = np.zeros(int(k)), np.zeros(int(k))
    read, read_held = np.zeros(int(k)), np.zeros(int(k))
    for j, i in enumerate(rng.choice(len(w), size=int(k), p=w)):
        if not total[i, 0] > 0:
            continue
        obs = probe_model.sample(X_probe[i], belief.particles[i], force, rng, p=Q[i])
        ll, info = probe_model.loglik(obs, X_probe, a, force, P=Q)
        wk = belief.reweight(ll).weights * keep      # w'(1 - r): the belief given the result and the object kept
        pick = int(np.argmin(C @ wk))
        if pick != best:
            gains[j] = float(wk @ (C[best] - C[pick]))
        if heldout:
            rows_l = heldout_rows(C, wk, best, groups)
            held[j] = float(wk @ (C[best] - C[rows_l, cols]))
        if reading_value and "reading" in info["used"]:
            terms = [t for t in info["used"] if t != "reading"]
            uk = belief.reweight(probe_model.loglik(obs, X_probe, a, force, P=Q, terms=terms)[0]).weights * keep
            read[j] = float(wk @ (C[int(np.argmin(C @ uk))] - C[pick]))
            if heldout:
                read_held[j] = float(wk @ (C[heldout_rows(C, uk, best, groups), cols] - C[rows_l, cols]))
    mean = lambda g: float(np.mean(g)) if len(g) else 0.0  # noqa: E731
    err = lambda g: float(np.std(g, ddof=1) / np.sqrt(len(g))) if len(g) > 1 else 0.0  # noqa: E731
    gain = mean(gains)
    out = {"p_kind": dict(zip(probe_model.classes, (w @ P).tolist())), "p_thrown": float(w @ thrown), "estimator": estimator}
    if heldout:
        out.update(evoi=mean(held), evoi_se=err(held), expected_cost_after=unused - mean(held),
                   plugin_evoi=gain if gain > EVOI_EPS else 0.0)
    else:
        out.update(evoi=gain if gain > EVOI_EPS else 0.0, evoi_se=err(gains), expected_cost_after=unused - gain)
    if reading_value:
        r = read_held if heldout else read
        out.update(reading_gain=mean(r), reading_gain_se=err(r))
        if heldout:
            out["plugin_reading_gain"] = mean(read)
    return out


def push_tip_charge(belief: ParticleBelief, probe_model: Any, X_probe: np.ndarray) -> float:
    """The expected tipping charge of a push: P(the push keeps the object and tips it) times the tip's cost. For a kind
    model of the pushes not thrown that is per particle (1 - P(thrown)) P(tipped | not thrown); a milestone 2a model's
    P(tipped) is already that (it learned on every push, and no thrown push of its corpus tipped)."""
    if "tipped" not in probe_model.classes:
        return 0.0
    P = np.asarray(probe_model.p_kind(X_probe), dtype=float)[:, list(probe_model.classes).index("tipped")]
    if getattr(probe_model, "kinds_not_thrown", True):
        P = (1.0 - np.asarray(probe_model.risk(X_probe), dtype=float)) * P
    return float(belief.weights @ P) * COST["tipped"]


def _probe_rows(probe_models: dict[str, Any], belief: ParticleBelief, geometry: Geometry, mid: Any, pose: dict[str, Any],
                f_std: float, care: str | float, continuation: list[Option], rng: np.random.Generator,
                abstain_cost: float | None = None, estimator: str = "plugin", time_scale: float = 1.0,
                reset_s: float | None = None) -> list[Option]:
    """One row per probe model, at the middle proposal: the push and the weigh with the weigh squeeze, the squeeze
    (only for a crushable or bruisable object, by the pose's public entry) at ``squeeze_cap``. The continuation costs
    are those of the current pose (the push moves the object up to 4 cm, the weigh lifts it 2 cm, the squeeze
    presses in place; each is treated as leaving the menu unchanged). ``time_scale`` multiplies each probe's time charge
    (plan 2b2 decision 4)."""
    flags = entry_flags(pose.get("entry")) or {}
    a = belief.arrays()
    costs = [o._costs for o in continuation]
    rows = []
    for probe, pm in probe_models.items():
        if probe == "squeeze" and not (flags.get("frag_crush") or flags.get("frag_bruise")):
            continue
        force = squeeze_cap(belief, f_std) if probe == "squeeze" else weigh_force(f_std)
        base = extract(geometry, mid.point, mid.yaw, PLACEHOLDER["mass"], PLACEHOLDER["mu"], force, PROBE_V_LIFT, 0.0,
                       PROBE_DESTINATION, None, belief.visibility)
        X = particle_matrix(base, a, pose, force, model_names(pm))
        v = probe_value(belief, costs, pm, X, force, care, rng, abstain_cost=abstain_cost, estimator=estimator, reset_s=reset_s)
        tip = push_tip_charge(belief, pm, X) if probe == "push" else 0.0
        cost = time_scale * TIME_COST_PER_S * PROBE_SECONDS[probe] + tip + v["expected_cost_after"]
        reason = f"{probe} first: value of information {v['evoi']:.2f}, expected cost of the best row after it {v['expected_cost_after']:.2f}"
        rows.append(Option(id=f"probe_{probe}", kind="probe", action={"probe": probe, "s": mid.s, "force": force}, pred=None,
                           cost_s=PROBE_SECONDS[probe], expected_cost=cost, reason=reason, probe=v, _base=base, _pose=pose))
    return rows


def guard_queries(options: list[Option], belief: ParticleBelief, names=FEATURES_M2) -> tuple[np.ndarray, np.ndarray]:
    """The novelty guard's query: every commit and lower-risk row for every particle (particle_matrix in the guard's
    ``names``, as the model sees them), with the belief's weights within a row and the rows weighted equally."""
    rows = [o for o in options if o.kind in ("commit", "lower_risk")]
    a = belief.arrays()
    w = np.asarray(belief.weights, dtype=float)
    X = np.vstack([particle_matrix(o._base, a, o._pose, o.action["force"], names) for o in rows])
    return X, np.tile(w / w.sum(), len(rows)) / len(rows)


def _rows(proposals: list[Any], f_std: float, destination: str, com_xy: np.ndarray, damage: bool = False):
    """Candidate rows: (id, kind, proposal, action, cost_s, reason). ``damage`` (a model on the damage features) adds
    the rows that lower a damage risk: light_grip and gentle_carry."""
    dest = DESTINATIONS[destination]
    mid = mid_proposal(proposals)
    base = {"force": f_std, "v_lift": STD["v_lift"], "v_transport": STD["v_transport"], "a_transport": STD["a_transport"],
            "destination": destination, "release_height": None}
    rows = [
        ("commit_std", "commit", mid, dict(base), 8.0, ""),
        ("slow_lift", "lower_risk", mid, dict(base, v_lift=0.05, v_transport=0.2), 14.0, "slower lift and carry"),
        ("firm_grip", "lower_risk", mid, dict(base, force=2 * f_std), 8.0, "twice the squeeze"),
    ]
    if damage:
        rows += [
            ("light_grip", "lower_risk", mid, dict(base, force=0.5 * f_std), 8.0, "half the squeeze: less crush and bruise risk, more slip"),
            ("gentle_carry", "lower_risk", mid, dict(base, a_transport=A_GENTLE), 9.0, "a gentle carry: less spill risk, slower"),
        ]
    nearest = min(proposals, key=lambda p: float(np.linalg.norm(np.asarray(p.point[:2]) - com_xy)))
    if nearest is not mid:
        rows.append(("regrasp_com", "lower_risk", nearest, dict(base), 10.0, "pinch nearer the believed centre of mass"))
    else:
        rows.append(("regrasp_com", "lower_risk", mid, dict(base), 10.0, "the standard pinch is already nearest the believed centre of mass"))
    if dest["kind"] == "none" or (dest["kind"] == "surface" and edge_distance(dest) < EDGE_RISK_M):
        rows.append(("inboard_place", "lower_risk", mid, dict(base, destination="table_centre"), 9.0,
                     "safer place, not the requested one"))
    if dest["kind"] in ("rim", "cavity", "open_bin"):
        rows.append(("drop_release", "lower_risk", mid, dict(base, release_height=0.05 + float(dest["rim_height"])), 6.0,
                     "release 5 cm above the rim: faster, more tip risk"))
    return rows


def build_menu(model: Any, belief: ParticleBelief | Belief, geometry: Geometry, proposals: list[Any], f_std: float, care: str | float,
               destination: str, plan_fn: Callable[..., list[Any]], rng: np.random.Generator,
               pose_fn: Callable[[Any], dict[str, Any]] | None = None, max_open: float = MAX_OPEN_WIDTH,
               threshold: float | None = None, n: int = N_PARTICLES,
               probe_models: dict[str, Any] | None = None, guard: Any = None,
               abstain_cost: float | None = None, evoi: str = "plugin", probe_time_scale: float = 1.0,
               reset_s: float | None = None) -> tuple[list[Option], str]:
    """The ranked menu and the recommendation. ``pose_fn(proposal)`` gives the pose the particles map
    through (default: ``geometry_pose``); a milestone 1 ``Belief`` becomes n particles at the
    nominal CoM; ``threshold`` None reads DEFAULT_THRESHOLD now, not at import; ``probe_models``
    ({probe: model}) adds a probe row per model. The recommendation is "probe" when the first row
    that is a probe or the best same-destination row is a probe with a positive value of information
    whose expected cost is below the best row's by more than EVOI_Z Monte Carlo errors (read at call
    time), else "commit" when the best row reaches the threshold, else "decline". A ``guard`` (guard.KnnGuard) that
    flags the rows makes it "decline" with the guard's reason; the decline row keeps the guard's check. With
    ``abstain_cost`` a number, the cost rule of the module docstring decides instead, and the decline row costs it.
    ``evoi`` names probe_value's estimator ("plugin" or "heldout"; another name raises ValueError);
    ``probe_time_scale`` multiplies every probe row's time charge (a finite number at least 0, else ValueError);
    ``reset_s`` (plan H3) prices every row's irreversible outcomes at the reset's time (costs.irreversible_cost)."""
    _check_estimator(evoi)
    time_scale = _probe_time_scale(probe_time_scale)
    pose_fn = pose_fn or (lambda prop: geometry_pose(geometry, prop))
    usable = usable_proposals(proposals, max_open)
    options: list[Option] = []
    if usable:
        mid = mid_proposal(usable)
        mid_pose = pose_fn(mid)
        if isinstance(belief, Belief):
            belief = ParticleBelief.from_parametric(belief, mid_pose["com_obj_nominal"], n, rng)
        com_xy = believed_com_world(belief, mid_pose)[:2]
        damage = bool(set(DAMAGE_FEATURES) & set(model_names(model)))
        for rid, kind, prop, action, cost_s, reason in _rows(usable, f_std, destination, com_xy, damage):
            planned = plan_fn(prop, action["v_lift"], action["v_transport"], action["destination"], action.get("release_height"),
                              a_transport=action["a_transport"])
            planned = [s.to_dict() if hasattr(s, "to_dict") else s for s in planned]
            pose = pose_fn(prop)
            base = extract(geometry, prop.point, prop.yaw, PLACEHOLDER["mass"], PLACEHOLDER["mu"], action["force"], action["v_lift"],
                           action["v_transport"], action["destination"], action.get("release_height"), belief.visibility,
                           a_transport=action["a_transport"])
            pred, per = estimate(model, belief, base, pose, action["force"], planned, rng)
            act = dict(action, s=prop.s)
            options.append(Option(id=rid, kind=kind, action=act, pred=pred, cost_s=cost_s,
                                  expected_cost=expected_cost(pred, cost_s, care, abstain_cost, reset_s), reason=reason,
                                  _costs=particle_costs(per, cost_s, care, abstain_cost, reset_s), _base=base, _pose=pose,
                                  _planned=planned))
        continuation = [o for o in options if o.action["destination"] == destination and o.id != "inboard_place"]
        if probe_models and continuation:
            options += _probe_rows(probe_models, belief, geometry, mid, mid_pose, f_std, care, continuation, rng, abstain_cost,
                                   estimator=evoi, time_scale=time_scale, reset_s=reset_s)
        options.sort(key=lambda o: o.expected_cost)
    same = [o for o in options if o.kind in ("commit", "lower_risk") and o.action["destination"] == destination]
    check = None
    if same:
        best = same[0]
        if abstain_cost is None:
            floor = DEFAULT_THRESHOLD if threshold is None else float(threshold)
            worst = max(((k, best.pred.p[k] * COST[k]) for k in best.pred.p), key=lambda kv: kv[1])
            reason = f"{worst[0]} carries the largest expected cost in the best row ({best.id})"
            recommendation = "commit" if best.pred.p["placed_stable"] >= floor else "decline"
            first = next(o for o in options if o.kind == "probe" or o is best)
            if (first.kind == "probe" and first.probe["evoi"] > 0
                    and best.expected_cost - first.expected_cost > EVOI_Z * first.probe["evoi_se"]):
                recommendation = "probe"
        else:
            recommendation, reason = _cost_rule(options, best, care, float(abstain_cost), threshold, reset_s=reset_s)
        if guard is not None:
            check = guard.check(*guard_queries(options, belief, getattr(guard, "feature_names", None) or FEATURES_M2))
            if check["flag"]:
                recommendation, reason = "decline", flag_reason(check)
    else:
        reason = decline_reason(proposals, max_open) if proposals and not usable else "no row reaches the requested destination"
        recommendation = "decline"
    options.append(Option(id="decline", kind="decline", action={}, pred=None, cost_s=0.0,
                          expected_cost=None if abstain_cost is None else float(abstain_cost), reason=reason, guard=check))
    return options, recommendation


def without_probe(options: list[Option], recommendation: str, destination: str, care: str | float, abstain_cost: float | None,
                  threshold: float | None, reset_s: float | None = None) -> str:
    """The recommendation of the same menu without its probe rows, from the unrounded costs: the recommendation itself
    unless it is probe (a guard flag has already declined); else the cost rule on the best row (commit when it costs at
    most A and reaches the floor, if one is set), or without an abstain cost the milestone 2a rule (P(placed_stable) at
    the threshold). The policy that never probes follows it (plan m4 decision 2)."""
    if recommendation != "probe":
        return recommendation
    same = [o for o in options if o.kind in ("commit", "lower_risk") and o.action.get("destination") == destination]
    if not same:
        return "decline"
    best = min(same, key=lambda o: o.expected_cost)
    if abstain_cost is None:
        floor = DEFAULT_THRESHOLD if threshold is None else float(threshold)
        return "commit" if best.pred.p["placed_stable"] >= floor else "decline"
    return _cost_rule([o for o in options if o.kind != "probe"], best, care, float(abstain_cost), threshold, reset_s=reset_s)[0]


def _cost_rule(options: list[Option], best: Option, care: str | float, A: float, floor: float | None,
               reset_s: float | None = None) -> tuple[str, str]:
    """(recommendation, decline reason) under the cost rule: commit when the best row costs at most A (and reaches
    the floor, if one is set), a probe that beats the cheaper of the two by more than EVOI_Z errors, else decline. The
    reason names the outcome with the largest expected cost under the costs in force (``reset_s``: the recovery mode)."""
    scale = care_scale(care)
    reset = None if reset_s is None else irreversible_cost(care, reset_s)

    def charge(k: str) -> float:
        if reset is not None and k in IRREVERSIBLE:
            return best.pred.p[k] * reset
        return best.pred.p[k] * COST[k] * (scale if k in IRREVERSIBLE else 1.0)

    worst = max(best.pred.p, key=charge)
    c, p_ok = float(best.expected_cost), float(best.pred.p["placed_stable"])
    if floor is not None and c <= A and p_ok < float(floor):
        recommendation, reason = "decline", f"best row's P(placed_stable) {p_ok:.2f} is below the floor {float(floor):.2f}"
    else:
        recommendation = "commit" if c <= A else "decline"
        reason = f"best row costs {c:.2f} {'>' if c > A else '<='} abstain {A:.2f}: {worst} carries the largest expected cost"
    probes = [o for o in options if o.kind == "probe"]
    if probes:
        pr = min(probes, key=lambda o: o.expected_cost)
        if pr.probe["evoi"] > 0 and min(c, A) - pr.expected_cost > EVOI_Z * pr.probe["evoi_se"]:
            recommendation = "probe"
    return recommendation, reason
