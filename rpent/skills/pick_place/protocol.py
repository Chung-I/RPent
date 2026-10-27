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

    def held_to_release(self) -> float:
        return 1.0 - self.p["not_lifted"] - self.p["dropped_low"] - self.p["dropped_high"]


@dataclass
class Option:
    """One menu row: an action, its predicted outcomes and its expected cost."""

    id: str
    kind: Literal["commit", "lower_risk", "decline"]
    action: dict[str, Any]
    pred: Prediction | None
    cost_s: float
    expected_cost: float | None
    reason: str = ""

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
        }


@dataclass
class Assessment:
    """What assess_pick returns: a ranked menu and the best row's numbers."""

    object: str
    destination: str
    options: list[Option]
    evidence: list[str]
    recommendation: Literal["commit", "decline"]
    care: str
    visibility: float
    episode: int
    extras: dict[str, Any] = field(default_factory=dict)

    def best(self) -> Option | None:
        for row in self.options:
            if row.kind != "decline" and row.action.get("destination") == self.destination:
                return row
        return None

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
            "decline_reason": self.decline_reason(),
            "care": self.care,
            "visibility": _r(self.visibility),
        }
