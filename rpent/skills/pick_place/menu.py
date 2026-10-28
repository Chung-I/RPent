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
with the belief's weights; each row keeps its per-particle costs for the probe values.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from rpent.skills.pick_place.belief import N_PARTICLES, Belief, ParticleBelief
from rpent.skills.pick_place.costs import COST, expected_cost, particle_costs
from rpent.skills.pick_place.destinations import DESTINATIONS, edge_distance
from rpent.skills.pick_place.features import Geometry, extract, particle_features, particle_matrix
from rpent.skills.pick_place.protocol import Option, Prediction

MAX_OPEN_WIDTH = 0.11
DEFAULT_THRESHOLD = 0.7
EDGE_RISK_M = 0.08
STD = {"v_lift": 0.15, "v_transport": 0.5}
PLACEHOLDER = {"mass": 1.0, "mu": 0.5}       # base features carry these; every particle overwrites them


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

        X = particle_matrix(base, belief.arrays(), pose, force)
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


def _rows(proposals: list[Any], f_std: float, destination: str, com_xy: np.ndarray):
    """Candidate rows: (id, kind, proposal, action, cost_s, reason)."""
    dest = DESTINATIONS[destination]
    mid = min(proposals, key=lambda p: abs(p.s - 0.5))
    base = {"force": f_std, "v_lift": STD["v_lift"], "v_transport": STD["v_transport"], "destination": destination, "release_height": None}
    rows = [
        ("commit_std", "commit", mid, dict(base), 8.0, ""),
        ("slow_lift", "lower_risk", mid, dict(base, v_lift=0.05, v_transport=0.2), 14.0, "slower lift and carry"),
        ("firm_grip", "lower_risk", mid, dict(base, force=2 * f_std), 8.0, "twice the squeeze"),
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


def build_menu(model: Any, belief: ParticleBelief | Belief, geometry: Geometry, proposals: list[Any], f_std: float, care: str,
               destination: str, plan_fn: Callable[..., list[Any]], rng: np.random.Generator,
               pose_fn: Callable[[Any], dict[str, Any]] | None = None, max_open: float = MAX_OPEN_WIDTH,
               threshold: float | None = None, n: int = N_PARTICLES) -> tuple[list[Option], str]:
    """The ranked menu and the recommendation. ``pose_fn(proposal)`` gives the pose the particles map
    through (default: ``geometry_pose``); a milestone 1 ``Belief`` becomes n particles at the
    nominal CoM; ``threshold`` None reads DEFAULT_THRESHOLD now, not at import."""
    threshold = DEFAULT_THRESHOLD if threshold is None else float(threshold)
    pose_fn = pose_fn or (lambda prop: geometry_pose(geometry, prop))
    usable = [p for p in proposals if not (np.isfinite(p.width) and p.width > max_open)]
    skipped = len(proposals) - len(usable)
    options: list[Option] = []
    if usable:
        mid_pose = pose_fn(min(usable, key=lambda p: abs(p.s - 0.5)))
        if isinstance(belief, Belief):
            belief = ParticleBelief.from_parametric(belief, mid_pose["com_obj_nominal"], n, rng)
        com_xy = believed_com_world(belief, mid_pose)[:2]
        for rid, kind, prop, action, cost_s, reason in _rows(usable, f_std, destination, com_xy):
            planned = plan_fn(prop, action["v_lift"], action["v_transport"], action["destination"], action.get("release_height"))
            planned = [s.to_dict() if hasattr(s, "to_dict") else s for s in planned]
            pose = pose_fn(prop)
            base = extract(geometry, prop.point, prop.yaw, PLACEHOLDER["mass"], PLACEHOLDER["mu"], action["force"], action["v_lift"],
                           action["v_transport"], action["destination"], action.get("release_height"), belief.visibility)
            pred, per = estimate(model, belief, base, pose, action["force"], planned, rng)
            act = dict(action, s=prop.s)
            options.append(Option(id=rid, kind=kind, action=act, pred=pred, cost_s=cost_s,
                                  expected_cost=expected_cost(pred, cost_s, care), reason=reason,
                                  _costs=particle_costs(per, cost_s, care), _base=base, _pose=pose, _planned=planned))
        options.sort(key=lambda o: o.expected_cost)
    same = [o for o in options if o.action["destination"] == destination]
    if same:
        best = same[0]
        worst = max(((k, best.pred.p[k] * COST[k]) for k in best.pred.p), key=lambda kv: kv[1])
        reason = f"{worst[0]} carries the largest expected cost in the best row ({best.id})"
        recommendation = "commit" if best.pred.p["placed_stable"] >= threshold else "decline"
    else:
        reason = "every proposal is too wide for the gripper" if skipped and not usable else "no row reaches the requested destination"
        recommendation = "decline"
    options.append(Option(id="decline", kind="decline", action={}, pred=None, cost_s=0.0, expected_cost=None, reason=reason))
    return options, recommendation
