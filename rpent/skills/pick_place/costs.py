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

"""Cost of each outcome and of time. ``care`` scales only the irreversible outcomes: a name (low, normal, high) or a
number above 0 (the care level kappa of the paper's sweep).

The abstain cost (milestone 2b) is the value of the task: a case that ends without the object placed costs it, so a
row's expected cost adds ``abstain_cost * (1 - P(placed_stable))`` and abstaining costs ``abstain_cost``. None keeps the
milestone 2a costs (no such term, and a decline costs nothing)."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np

from rpent.skills.pick_place.protocol import IRREVERSIBLE, OUTCOMES, Prediction

COST: dict[str, float] = {
    "placed_stable": 0.0,
    "tipped": 1.0,
    "missed": 1.0,
    "dropped_low": 1.0,
    "not_lifted": 0.5,
    "dropped_high": 5.0,
    "rolled_off": 5.0,
    "crushed": 5.0,
    "bruised": 5.0,
    "broken": 5.0,
    "spilled": 5.0,
}
TIME_COST_PER_S = 0.05
CARE_SCALE: dict[str, float] = {"low": 0.5, "normal": 1.0, "high": 3.0}
ABSTAIN_COST = 1.0           # milestone 2b: the value of the task (as much as a recoverable failure, at no time cost)


def care_scale(care: str | float) -> float:
    """The factor on the irreversible costs: CARE_SCALE for a name, the number itself when finite and above 0."""
    if isinstance(care, str) and care in CARE_SCALE:
        return CARE_SCALE[care]
    if isinstance(care, Real) and not isinstance(care, bool) and math.isfinite(float(care)) and float(care) > 0:
        return float(care)
    raise ValueError(f"care must be low, normal, high or a number above 0, not {care!r}")


def probe_time_scale(scale: float) -> float:
    """The factor on every probe's time charge (plan 2b2 decision 4): a finite number at least 0 (0 removes the
    charge)."""
    if isinstance(scale, Real) and not isinstance(scale, bool) and math.isfinite(float(scale)) and float(scale) >= 0:
        return float(scale)
    raise ValueError(f"the probe time scale must be a finite number at least 0, not {scale!r}")


def _unfinished(p_placed, abstain_cost: float | None):
    return 0.0 if abstain_cost is None else float(abstain_cost) * (1.0 - p_placed)


def expected_cost(pred: Prediction, cost_s: float, care: str | float = "normal", abstain_cost: float | None = None) -> float:
    scale = care_scale(care)
    total = 0.0
    for k, p in pred.p.items():
        total += p * COST[k] * (scale if k in IRREVERSIBLE else 1.0)
    return float(total + TIME_COST_PER_S * cost_s + _unfinished(pred.p.get("placed_stable", 0.0), abstain_cost))


def cost_vector(care: str | float = "normal") -> np.ndarray:
    """The outcome costs in OUTCOMES order, the irreversible ones scaled by care."""
    scale = care_scale(care)
    return np.array([COST[k] * (scale if k in IRREVERSIBLE else 1.0) for k in OUTCOMES])


def prediction_matrix(preds: list[Prediction]) -> np.ndarray:
    return np.array([[x.p.get(k, 0.0) for k in OUTCOMES] for x in preds])


def particle_costs(preds: list[Prediction], cost_s: float, care: str | float = "normal",
                   abstain_cost: float | None = None) -> np.ndarray:
    """Expected cost of one row under each particle (the row's expected cost is their weighted mean)."""
    P = prediction_matrix(preds)
    return P @ cost_vector(care) + TIME_COST_PER_S * cost_s + _unfinished(P[:, OUTCOMES.index("placed_stable")], abstain_cost)
