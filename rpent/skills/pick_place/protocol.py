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

"""Result protocol of the pick_place skill.

Every probability is conditional on one object and one menu row (a grasp pose,
a squeeze force, a lift and transport speed, a destination and a release
height). The top-level numbers of an assessment are the best row's numbers,
repeated so a planner can read one line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

OUTCOMES: tuple[str, ...] = (
    "placed_stable",
    "tipped",
    "missed",
    "dropped_low",
    "dropped_high",
    "rolled_off",
    "not_lifted",
)
IRREVERSIBLE: tuple[str, ...] = ("dropped_high", "rolled_off")
RECOVERABLE: tuple[str, ...] = ("tipped", "missed", "dropped_low", "not_lifted")
DROP_HEIGHT_BINS: tuple[str, ...] = ("0.00-0.05", "0.05-0.10", "0.10-0.20", "0.20+")


def _r(x: float | None) -> float | None:
    return None if x is None else round(float(x), 3)


@dataclass
class Prediction:
    """Outcome probabilities for one row, with an optional drop-height distribution."""

    p: dict[str, float]
    drop_height: dict[str, float] | None = None
    expected_loss_t: float | None = None

    @staticmethod
    def mean(preds: list[Prediction]) -> Prediction:
        n = len(preds)
        p = {k: sum(x.p.get(k, 0.0) for x in preds) / n for k in OUTCOMES}
        if any(x.drop_height is None for x in preds):
            return Prediction(p=p, drop_height=None, expected_loss_t=None)
        dh = {b: sum(x.drop_height.get(b, 0.0) for x in preds) / n for b in DROP_HEIGHT_BINS}
        lt = [x.expected_loss_t for x in preds if x.expected_loss_t is not None]
        return Prediction(p=p, drop_height=dh, expected_loss_t=sum(lt) / len(lt) if lt else None)

    @staticmethod
    def weighted_mean(preds: list[Prediction], weights) -> Prediction:
        """The belief-weighted mean of per-particle predictions (the expected loss time over the
        particles that have one)."""
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        P = np.array([[x.p.get(k, 0.0) for k in OUTCOMES] for x in preds])
        p = dict(zip(OUTCOMES, (w @ P).tolist()))
        if any(x.drop_height is None for x in preds):
            return Prediction(p=p, drop_height=None, expected_loss_t=None)
        DH = np.array([[x.drop_height.get(b, 0.0) for b in DROP_HEIGHT_BINS] for x in preds])
        lt = [(wi, x.expected_loss_t) for wi, x in zip(w, preds) if x.expected_loss_t is not None]
        wl = sum(wi for wi, _ in lt)
        return Prediction(p=p, drop_height=dict(zip(DROP_HEIGHT_BINS, (w @ DH).tolist())),
                          expected_loss_t=(sum(wi * t for wi, t in lt) / wl) if wl > 0 else None)

    def held_to_release(self) -> float:
        return 1.0 - self.p["not_lifted"] - self.p["dropped_low"] - self.p["dropped_high"]


def _probe_data(probe: dict[str, Any] | None) -> dict[str, Any] | None:
    if probe is None:
        return None
    return {"p_kind": {k: _r(v) for k, v in probe["p_kind"].items()}, "p_thrown": _r(probe["p_thrown"]),
            "evoi": _r(probe["evoi"]), "evoi_se": _r(probe["evoi_se"]), "expected_cost_after": _r(probe["expected_cost_after"])}


@dataclass
class Option:
    """One menu row: an action, its predicted outcomes and its expected cost. A probe row (kind
    ``probe``) has no outcome prediction; ``probe`` holds its predicted kinds, its risk of throwing the
    object, its value of information and the expected cost of the best row after it."""

    id: str
    kind: Literal["commit", "lower_risk", "decline", "probe"]
    action: dict[str, Any]
    pred: Prediction | None
    cost_s: float
    expected_cost: float | None
    reason: str = ""
    probe: dict[str, Any] | None = None
    _costs: Any = field(default=None, repr=False, compare=False)       # per-particle expected costs (not serialised)
    _base: dict[str, Any] | None = field(default=None, repr=False, compare=False)
    _pose: dict[str, Any] | None = field(default=None, repr=False, compare=False)
    _planned: list[dict[str, Any]] | None = field(default=None, repr=False, compare=False)

    def to_data(self) -> dict[str, Any]:
        pred = self.pred
        return {
            "id": self.id,
            "kind": self.kind,
            "action": dict(self.action),
            "p": None if pred is None else {k: _r(v) for k, v in pred.p.items()},
            "p_drop_height": None if pred is None or pred.drop_height is None
            else {k: _r(v) for k, v in pred.drop_height.items()},
            "expected_loss_t": None if pred is None else _r(pred.expected_loss_t),
            "cost_s": _r(self.cost_s),
            "expected_cost": _r(self.expected_cost),
            "reason": self.reason,
            "probe": _probe_data(self.probe),
        }


@dataclass
class Assessment:
    """What assess_pick returns: a ranked menu and the best row's numbers."""

    object: str
    destination: str
    options: list[Option]
    evidence: list[str]
    recommendation: Literal["commit", "decline", "probe"]
    care: str
    visibility: float
    episode: int
    extras: dict[str, Any] = field(default_factory=dict)

    def best(self) -> Option | None:
        """The first commit or lower-risk row to the requested destination (never a probe, never decline)."""
        for row in self.options:
            if row.kind in ("commit", "lower_risk") and row.action.get("destination") == self.destination:
                return row
        return None

    def best_overall(self) -> Option | None:
        """The first row that is a probe or the best() row."""
        best = self.best()
        for row in self.options:
            if row.kind == "probe" or row is best:
                return row
        return None

    def recommended(self) -> Option | None:
        """The row the recommendation names: the best_overall() probe, the best() row on commit, none on decline."""
        if self.recommendation == "probe":
            return self.best_overall()
        return self.best() if self.recommendation == "commit" else None

    def decline_reason(self) -> str:
        for row in self.options:
            if row.kind == "decline":
                return row.reason
        return ""

    def to_data(self) -> dict[str, Any]:
        best = self.best()
        pred = None if best is None else best.pred
        if pred is None:
            p_success = p_held = p_irr = p_rec = p_dh = None
        else:
            p_success = _r(pred.p["placed_stable"])
            p_held = _r(pred.held_to_release())
            p_irr = {k: _r(pred.p[k]) for k in IRREVERSIBLE}
            p_rec = {k: _r(pred.p[k]) for k in RECOVERABLE}
            p_dh = None if pred.drop_height is None else {k: _r(v) for k, v in pred.drop_height.items()}
        return {
            "assessed": True,
            "object": self.object,
            "destination": self.destination,
            "best_option": None if best is None else best.id,
            "p_success": p_success,
            "p_held_to_release": p_held,
            "p_irreversible": p_irr,
            "p_recoverable": p_rec,
            "p_drop_height": p_dh,
            "evidence": list(self.evidence),
            "options": [row.to_data() for row in self.options],
            "recommendation": self.recommendation,
            "recommended_option": None if self.recommended() is None else self.recommended().id,
            "decline_reason": self.decline_reason(),
            "care": self.care,
            "visibility": _r(self.visibility),
        }
