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

Probe rows (push, weigh; when the substrate has probe models) are valued by preposterior analysis:
the expected cost of the best continuation row after the probe's result, over results simulated
from the belief (``probe_value``). A probe row's expected cost is its time, the push's tip risk and
that expected cost after it; it competes with the other rows in the same ranking. The value is
estimated per simulated result against the best row now, so it is exactly 0 when no result can
change that row; a probe is recommended only when its row beats the best row by more than EVOI_Z
Monte Carlo errors.

The cost rule (milestone 2b, ``abstain_cost`` a number A): every row's expected cost adds A x (1 - P(placed_stable)),
the cost of leaving the task undone, and abstaining costs A. The recommendation is the cheapest of the best row
("commit"), the cheapest probe row ("probe", when it beats the cheaper of the best row and A by more than EVOI_Z errors)
and abstaining ("decline"); ``threshold`` is then an optional floor on P(placed_stable). A probe's preposterior may
abstain after the probe, and a probe that ends the case costs the irreversible cost and A. ``abstain_cost`` None keeps
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
from rpent.skills.pick_place.costs import COST, TIME_COST_PER_S, care_scale, expected_cost, particle_costs
from rpent.skills.pick_place.destinations import DESTINATIONS, edge_distance
from rpent.skills.pick_place.features import (A_TRANSPORT, DAMAGE_FEATURES, FEATURES_M2, Geometry, extract, model_names,
                                              particle_features, particle_matrix)
from rpent.skills.pick_place.guard import flag_reason
from rpent.skills.pick_place.probes import PROBE_SECONDS, weigh_force
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


def probe_value(belief: ParticleBelief, continuation_costs: list[np.ndarray], probe_model: Any, X_probe: np.ndarray, force: float,
                care: str | float, rng: np.random.Generator, k: int = EVOI_K, abstain_cost: float | None = None) -> dict[str, Any]:
    """The preposterior value of one probe against the cheapest continuation row now (row b, per-particle
    costs C_b). With an abstain cost A, abstaining is a continuation row that costs A for every particle, and T
    below adds A. The probe throws the object with each particle's risk r (the case ends at the
    irreversible cost T), so the exact cost after the probe when its result is not used is
    sum_i w_i (r_i T + (1 - r_i) C_b,i). For k particles drawn by weight one result is sampled from the
    probe model and the belief w' is reweighted by its likelihood; the result's gain is the posterior
    saving of the row it then picks (a, the cheapest under w') over row b where the object is kept:
    sum_l w'_l (1 - r_l) (C_b,l - C_a,l), 0 whenever the result leaves row b cheapest. Returns the
    belief's P(kind) and P(thrown), ``evoi`` (the mean gain, floored at 0), ``evoi_se`` (its Monte
    Carlo error) and ``expected_cost_after`` (the cost when the result is not used, minus the mean gain)."""
    C = np.asarray(continuation_costs, dtype=float).reshape(len(continuation_costs), -1)
    if abstain_cost is not None:
        C = np.vstack([C, np.full(C.shape[1], float(abstain_cost))])
    w = belief.weights
    best = int(np.argmin(C @ w))
    risk = np.asarray(probe_model.risk(X_probe), dtype=float)
    P = np.asarray(probe_model.p_kind(X_probe), dtype=float)
    a = belief.arrays()
    keep = 1.0 - risk
    terminal = COST["dropped_high"] * care_scale(care) + (0.0 if abstain_cost is None else float(abstain_cost))
    unused = float(w @ (risk * terminal + keep * C[best]))
    gains = np.zeros(int(k))
    for j, i in enumerate(rng.choice(len(w), size=int(k), p=w)):
        obs = probe_model.sample(X_probe[i], belief.particles[i], force, rng, p=P[i])
        ll, _ = probe_model.loglik(obs, X_probe, a, force, P=P)
        wp = belief.reweight(ll).weights
        pick = int(np.argmin(C @ wp))
        if pick != best:
            gains[j] = float((wp * keep) @ (C[best] - C[pick]))
    gain = float(np.mean(gains)) if len(gains) else 0.0
    se = float(np.std(gains, ddof=1) / np.sqrt(len(gains))) if len(gains) > 1 else 0.0
    return {"p_kind": dict(zip(probe_model.classes, (w @ P).tolist())), "p_thrown": float(w @ risk),
            "evoi": gain if gain > EVOI_EPS else 0.0, "evoi_se": se, "expected_cost_after": unused - gain}


def _probe_rows(probe_models: dict[str, Any], belief: ParticleBelief, geometry: Geometry, mid: Any, pose: dict[str, Any],
                f_std: float, care: str | float, continuation: list[Option], rng: np.random.Generator,
                abstain_cost: float | None = None) -> list[Option]:
    """One row per probe model, at the middle proposal with the weigh squeeze. The continuation costs
    are those of the current pose (the push moves the object up to 4 cm, the weigh lifts it 2 cm;
    both are treated as leaving the menu unchanged)."""
    force = weigh_force(f_std)
    base = extract(geometry, mid.point, mid.yaw, PLACEHOLDER["mass"], PLACEHOLDER["mu"], force, PROBE_V_LIFT, 0.0,
                   PROBE_DESTINATION, None, belief.visibility)
    a = belief.arrays()
    costs = [o._costs for o in continuation]
    rows = []
    for probe, pm in probe_models.items():
        X = particle_matrix(base, a, pose, force, model_names(pm))
        v = probe_value(belief, costs, pm, X, force, care, rng, abstain_cost=abstain_cost)
        tip = v["p_kind"].get("tipped", 0.0) * COST["tipped"] if probe == "push" else 0.0
        cost = TIME_COST_PER_S * PROBE_SECONDS[probe] + tip + v["expected_cost_after"]
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
               abstain_cost: float | None = None) -> tuple[list[Option], str]:
    """The ranked menu and the recommendation. ``pose_fn(proposal)`` gives the pose the particles map
    through (default: ``geometry_pose``); a milestone 1 ``Belief`` becomes n particles at the
    nominal CoM; ``threshold`` None reads DEFAULT_THRESHOLD now, not at import; ``probe_models``
    ({probe: model}) adds a probe row per model. The recommendation is "probe" when the first row
    that is a probe or the best same-destination row is a probe with a positive value of information
    whose expected cost is below the best row's by more than EVOI_Z Monte Carlo errors (read at call
    time), else "commit" when the best row reaches the threshold, else "decline". A ``guard`` (guard.KnnGuard) that
    flags the rows makes it "decline" with the guard's reason; the decline row keeps the guard's check. With
    ``abstain_cost`` a number, the cost rule of the module docstring decides instead, and the decline row costs it."""
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
                                  expected_cost=expected_cost(pred, cost_s, care, abstain_cost), reason=reason,
                                  _costs=particle_costs(per, cost_s, care, abstain_cost), _base=base, _pose=pose,
                                  _planned=planned))
        continuation = [o for o in options if o.action["destination"] == destination and o.id != "inboard_place"]
        if probe_models and continuation:
            options += _probe_rows(probe_models, belief, geometry, mid, mid_pose, f_std, care, continuation, rng, abstain_cost)
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
            recommendation, reason = _cost_rule(options, best, care, float(abstain_cost), threshold)
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


def _cost_rule(options: list[Option], best: Option, care: str | float, A: float, floor: float | None) -> tuple[str, str]:
    """(recommendation, decline reason) under the cost rule: commit when the best row costs at most A (and reaches
    the floor, if one is set), a probe that beats the cheaper of the two by more than EVOI_Z errors, else decline."""
    scale = care_scale(care)
    worst = max(best.pred.p, key=lambda k: best.pred.p[k] * COST[k] * (scale if k in IRREVERSIBLE else 1.0))
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
