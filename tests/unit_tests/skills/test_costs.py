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

"""The cost rule of milestone 2b: numeric care and the abstain cost of an unfinished case."""

import numpy as np
import pytest

from rpent.skills.pick_place.costs import ABSTAIN_COST, care_scale, expected_cost, particle_costs
from rpent.skills.pick_place.protocol import OUTCOMES, Prediction


def _pred(**p):
    out = {k: 0.0 for k in OUTCOMES}
    out.update(p)
    return Prediction(p=out)


def test_care_scale_accepts_names_and_numbers():
    assert care_scale("low") == 0.5 and care_scale("normal") == 1.0 and care_scale("high") == 3.0
    assert care_scale(10) == 10.0 and care_scale(0.25) == 0.25
    for bad in ("extreme", 0, -1, float("nan"), float("inf"), None, True):
        with pytest.raises(ValueError):
            care_scale(bad)


def test_expected_cost_charges_the_abstain_cost_when_unfinished():
    p = _pred(placed_stable=0.6, not_lifted=0.4)
    assert ABSTAIN_COST == 1.0
    assert expected_cost(p, 8.0, "normal", abstain_cost=1.0) == pytest.approx(0.4 * 0.5 + 0.4 + 0.4)
    assert expected_cost(p, 8.0, "normal", abstain_cost=None) == pytest.approx(0.6)
    assert expected_cost(p, 8.0, "normal") == pytest.approx(0.6)            # the default is the milestone 2a rule


def test_irreversible_costs_scale_with_numeric_care():
    p = _pred(placed_stable=0.8, dropped_high=0.1, rolled_off=0.05, missed=0.05)
    assert expected_cost(p, 0.0, 10.0) == pytest.approx(0.1 * 50 + 0.05 * 50 + 0.05 * 1)
    assert expected_cost(p, 0.0, 10.0, abstain_cost=2.0) == pytest.approx(0.1 * 50 + 0.05 * 50 + 0.05 * 1 + 2.0 * 0.2)


def test_particle_costs_match_expected_cost_per_particle():
    preds = [_pred(placed_stable=0.9, not_lifted=0.1), _pred(placed_stable=0.2, dropped_high=0.8)]
    c = particle_costs(preds, 8.0, 3.0, abstain_cost=1.0)
    assert c == pytest.approx(np.array([expected_cost(x, 8.0, 3.0, abstain_cost=1.0) for x in preds]))
