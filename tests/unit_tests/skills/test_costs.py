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

from rpent.skills.pick_place.costs import (ABSTAIN_COST, RESET_PRICES, TIME_COST_PER_S, care_scale, cost_vector, expected_cost,
                                          irreversible_cost, particle_costs, reset_price)
from rpent.skills.pick_place.protocol import IRREVERSIBLE, OUTCOMES, Prediction


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


# ---------------------------------------------------------------- the recovery mode (plan H3 decision 3)
def test_absorbing_is_the_default_cost():
    """Without reset_s every cost is milestone 4's: an irreversible outcome costs 5 x the care factor."""
    p = _pred(placed_stable=0.5, dropped_high=0.2, crushed=0.1, tipped=0.2)
    for care in ("normal", 10.0, 30.0):
        assert irreversible_cost(care) == irreversible_cost(care, None) == 5.0 * care_scale(care)
        assert cost_vector(care, reset_s=None) == pytest.approx(cost_vector(care))
        assert expected_cost(p, 8.0, care, 10.0, reset_s=None) == expected_cost(p, 8.0, care, 10.0) == pytest.approx(
            0.3 * 5.0 * care_scale(care) + 0.2 + 0.4 + 10.0 * 0.5)
        assert particle_costs([p], 8.0, care, 10.0, reset_s=None) == pytest.approx(particle_costs([p], 8.0, care, 10.0))


def test_a_reset_prices_every_irreversible_kind_at_its_time():
    """reset_s 20: each of the six irreversible outcomes costs 0.05 x 20 = 1.0 whatever the care, the recoverable ones
    keep 1, 1, 1 and 0.5; reset_s 0 makes them free."""
    assert len(IRREVERSIBLE) == 6
    for care in ("normal", 10.0, 30.0):
        v = dict(zip(OUTCOMES, cost_vector(care, reset_s=20.0)))
        assert [v[k] for k in IRREVERSIBLE] == pytest.approx([1.0] * 6)
        assert [v[k] for k in ("tipped", "missed", "dropped_low", "not_lifted")] == [1.0, 1.0, 1.0, 0.5]
        assert v["placed_stable"] == 0.0
        zero = dict(zip(OUTCOMES, cost_vector(care, reset_s=0.0)))
        assert [zero[k] for k in IRREVERSIBLE] == [0.0] * 6
        assert irreversible_cost(care, 20.0) == pytest.approx(TIME_COST_PER_S * 20.0) and irreversible_cost(care, 0.0) == 0.0
    p = _pred(placed_stable=0.5, dropped_high=0.2, crushed=0.1, tipped=0.2)
    assert expected_cost(p, 8.0, 10.0, 10.0, reset_s=20.0) == pytest.approx(0.3 * 1.0 + 0.2 + 0.4 + 10.0 * 0.5)
    assert particle_costs([p], 8.0, 10.0, 10.0, reset_s=20.0)[0] == pytest.approx(expected_cost(p, 8.0, 10.0, 10.0, reset_s=20.0))


def test_a_bad_reset_time_is_an_error():
    for bad in (-1.0, float("nan"), float("inf"), True, "20"):
        with pytest.raises(ValueError):
            irreversible_cost("normal", bad)


def test_the_care_price_charges_the_absorbing_penalty():
    """rec_hi (plan H3 Addendum A): reset_price "care" charges an irreversible outcome 5 x the care factor, absorbing's
    penalty, whatever the reset time, so every cost equals absorbing's; "time" (the default) keeps the reset's time."""
    p = _pred(placed_stable=0.5, dropped_high=0.2, crushed=0.1, tipped=0.2)
    for care in ("normal", 10.0, 30.0):
        for reset in (0.0, 20.0):
            assert irreversible_cost(care, reset, "care") == irreversible_cost(care) == 5.0 * care_scale(care)
            assert cost_vector(care, reset_s=reset, reset_price="care") == pytest.approx(cost_vector(care))
            assert expected_cost(p, 8.0, care, 10.0, reset_s=reset, reset_price="care") == pytest.approx(expected_cost(p, 8.0, care, 10.0))
            assert particle_costs([p], 8.0, care, 10.0, reset_s=reset, reset_price="care") == pytest.approx(
                particle_costs([p], 8.0, care, 10.0))
    assert irreversible_cost(10.0, 20.0, "time") == irreversible_cost(10.0, 20.0) == pytest.approx(TIME_COST_PER_S * 20.0)
    assert RESET_PRICES == ("time", "care") and reset_price("care") == "care" and reset_price("time") == "time"


def test_a_bad_reset_price_is_an_error():
    for bad in ("Care", "", None, 1):
        with pytest.raises(ValueError):
            reset_price(bad)
        with pytest.raises(ValueError):
            irreversible_cost(10.0, 20.0, bad)
