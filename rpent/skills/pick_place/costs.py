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
milestone 2a costs (no such term, and a decline costs nothing).

The recovery mode (plan H3 decision 3, ``reset_s`` a number of seconds): an irreversible outcome costs the reset's time,
TIME_COST_PER_S x reset_s, whatever the care, in place of 5 x the care factor (the object is restored and the case goes
on). None keeps the absorbing costs of milestone 4. ``reset_price`` (plan H3 Addendum A) names that price: "time" (the
reset's time) or "care" (rec_hi: absorbing's 5 x the care factor, the reset's seconds left out), so rec_hi differs from
absorbing in the restore alone."""

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


RESET_PRICES = ("time", "care")    # plan H3 Addendum A: the recovery mode's price of an irreversible event


def probe_time_scale(scale: float) -> float:
    """The factor on every probe's time charge (plan 2b2 decision 4): a finite number at least 0 (0 removes the
    charge)."""
    if isinstance(scale, Real) and not isinstance(scale, bool) and math.isfinite(float(scale)) and float(scale) >= 0:
        return float(scale)
    raise ValueError(f"the probe time scale must be a finite number at least 0, not {scale!r}")


def reset_time(reset_s: float) -> float:
    """A reset's seconds under the recovery mode (plan H3 decision 3): a finite number at least 0."""
    if isinstance(reset_s, Real) and not isinstance(reset_s, bool) and math.isfinite(float(reset_s)) and float(reset_s) >= 0:
        return float(reset_s)
    raise ValueError(f"the reset time must be a finite number of seconds at least 0, not {reset_s!r}")


def _reset_price(price: str) -> str:
    """The recovery mode's price of an irreversible event (plan H3 Addendum A): "time" or "care" (RESET_PRICES)."""
    if isinstance(price, str) and price in RESET_PRICES:
        return price
    raise ValueError(f"the reset price must be one of {RESET_PRICES}, not {price!r}")


reset_price = _reset_price


def irreversible_cost(care: str | float, reset_s: float | None = None, reset_price: str = "time") -> float:
    """The cost of an irreversible outcome: 5 x the care factor (absorbing, or with ``reset_s`` at the care price), or with
    ``reset_s`` at the time price the reset's time."""
    price = _reset_price(reset_price)
    if reset_s is None:
        return COST["dropped_high"] * care_scale(care)
    seconds = reset_time(reset_s)
    if price == "care":
        return COST["dropped_high"] * care_scale(care)
    care_scale(care)
    return TIME_COST_PER_S * seconds


def reset_cost(reset_s: float | None, care: str | float, reset_price: str) -> float | None:
    """The one price of every irreversible outcome under the time price; None where absorbing's per-outcome costs apply
    (no reset, or the care price: rec_hi then prices every outcome exactly as absorbing does)."""
    if _reset_price(reset_price) == "care" or reset_s is None:
        if reset_s is not None:
            reset_time(reset_s)
        return None
    return irreversible_cost(care, reset_s, reset_price)


def _unfinished(p_placed, abstain_cost: float | None):
    return 0.0 if abstain_cost is None else float(abstain_cost) * (1.0 - p_placed)


def expected_cost(pred: Prediction, cost_s: float, care: str | float = "normal", abstain_cost: float | None = None,
                  reset_s: float | None = None, reset_price: str = "time") -> float:
    scale, reset = care_scale(care), reset_cost(reset_s, care, reset_price)
    total = 0.0
    for k, p in pred.p.items():
        if reset is not None and k in IRREVERSIBLE:
            total += p * reset
        else:
            total += p * COST[k] * (scale if k in IRREVERSIBLE else 1.0)
    return float(total + TIME_COST_PER_S * cost_s + _unfinished(pred.p.get("placed_stable", 0.0), abstain_cost))


def cost_vector(care: str | float = "normal", reset_s: float | None = None, reset_price: str = "time") -> np.ndarray:
    """The outcome costs in OUTCOMES order, the irreversible ones scaled by care (with ``reset_s``: the reset's price)."""
    scale, reset = care_scale(care), reset_cost(reset_s, care, reset_price)
    return np.array([reset if reset is not None and k in IRREVERSIBLE else COST[k] * (scale if k in IRREVERSIBLE else 1.0)
                     for k in OUTCOMES])


def prediction_matrix(preds: list[Prediction]) -> np.ndarray:
    return np.array([[x.p.get(k, 0.0) for k in OUTCOMES] for x in preds])


def particle_costs(preds: list[Prediction], cost_s: float, care: str | float = "normal",
                   abstain_cost: float | None = None, reset_s: float | None = None, reset_price: str = "time") -> np.ndarray:
    """Expected cost of one row under each particle (the row's expected cost is their weighted mean)."""
    P = prediction_matrix(preds)
    return P @ cost_vector(care, reset_s, reset_price) + TIME_COST_PER_S * cost_s + _unfinished(P[:, OUTCOMES.index("placed_stable")], abstain_cost)
