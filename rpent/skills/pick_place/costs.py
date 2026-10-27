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

"""Cost of each outcome and of time. ``care`` scales only the irreversible outcomes."""

from __future__ import annotations

from rpent.skills.pick_place.protocol import IRREVERSIBLE, Prediction

COST: dict[str, float] = {
    "placed_stable": 0.0,
    "tipped": 1.0,
    "missed": 1.0,
    "dropped_low": 1.0,
    "not_lifted": 0.5,
    "dropped_high": 5.0,
    "rolled_off": 5.0,
}
TIME_COST_PER_S = 0.05
CARE_SCALE: dict[str, float] = {"low": 0.5, "normal": 1.0, "high": 3.0}


def expected_cost(pred: Prediction, cost_s: float, care: str = "normal") -> float:
    scale = CARE_SCALE[care]
    total = 0.0
    for k, p in pred.p.items():
        total += p * COST[k] * (scale if k in IRREVERSIBLE else 1.0)
    return float(total + TIME_COST_PER_S * cost_s)
