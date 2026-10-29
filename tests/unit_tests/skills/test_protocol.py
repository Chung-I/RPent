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

"""The pick_place result protocol: Prediction, Option, Assessment."""

import json

from rpent.skills.pick_place.protocol import (
    IRREVERSIBLE,
    OUTCOMES,
    Assessment,
    Option,
    Prediction,
)

EXPECTED_KEYS = {
    "assessed", "object", "destination", "best_option", "p_success", "p_held_to_release",
    "p_irreversible", "p_recoverable", "p_drop_height", "evidence", "options",
    "recommendation", "recommended_option", "decline_reason", "care", "visibility", "guard",
}


def _pred(**over):
    p = {"placed_stable": 0.71234, "tipped": 0.05, "missed": 0.0, "dropped_low": 0.05,
         "dropped_high": 0.1, "rolled_off": 0.08766, "not_lifted": 0.0}
    p.update(over)
    return Prediction(p=p, drop_height={"0.00-0.05": 0.05, "0.05-0.10": 0.0, "0.10-0.20": 0.1, "0.20+": 0.0},
                      expected_loss_t=1.2)


def _row(id="commit_std", destination="plate", pred=None):
    return Option(id=id, kind="commit",
                  action={"s": 0.5, "force": 20.0, "v_lift": 0.15, "v_transport": 0.5,
                          "destination": destination, "release_height": 0.065},
                  pred=pred or _pred(), cost_s=8.0, expected_cost=1.0)


def _decline(reason="dropped_high carries the cost"):
    return Option(id="decline", kind="decline", action={}, pred=None, cost_s=0.0,
                  expected_cost=None, reason=reason)


def test_outcomes_and_irreversible():
    assert OUTCOMES == ("placed_stable", "tipped", "missed", "dropped_low", "dropped_high",
                        "rolled_off", "not_lifted")
    assert IRREVERSIBLE == ("dropped_high", "rolled_off")


def test_to_data_keys_and_best_row():
    a = Assessment(object="mug", destination="plate", options=[_row(), _decline()],
                   evidence=["prior: ceramic"], recommendation="commit", care="normal",
                   visibility=1.0, episode=3)
    d = a.to_data()
    assert set(d) == EXPECTED_KEYS
    assert d["assessed"] is True
    assert d["best_option"] == "commit_std"
    assert d["p_success"] == 0.712
    assert d["p_irreversible"] == {"dropped_high": 0.1, "rolled_off": 0.088}
    assert d["p_recoverable"] == {"tipped": 0.05, "missed": 0.0, "dropped_low": 0.05, "not_lifted": 0.0}
    assert abs(d["p_held_to_release"] - 0.85) < 1e-3
    assert d["p_drop_height"] == {"0.00-0.05": 0.05, "0.05-0.10": 0.0, "0.10-0.20": 0.1, "0.20+": 0.0}
    assert d["decline_reason"] == "dropped_high carries the cost"
    assert d["options"][0]["p"]["placed_stable"] == 0.712
    assert d["options"][0]["expected_loss_t"] == 1.2
    assert d["options"][1]["p"] is None
    json.dumps(d)


def test_decline_only_assessment():
    a = Assessment(object="box", destination="bin", options=[_decline("every proposal is too wide")],
                   evidence=[], recommendation="decline", care="high", visibility=1.0, episode=0)
    d = a.to_data()
    assert d["best_option"] is None
    assert d["p_success"] is None and d["p_held_to_release"] is None
    assert d["p_irreversible"] is None and d["p_drop_height"] is None
    assert d["decline_reason"] == "every proposal is too wide"
    json.dumps(d)


def test_the_assessment_reports_the_guard_check():
    rec = {"guard": "knn", "score": 4.21234, "p": 0.02439, "flag": True, "alpha": 0.05, "k": 10, "n_calibration": 40}
    decline = _decline("the object is unlike the training objects")
    decline.guard = rec
    a = Assessment(object="box", destination="bin", options=[_row(destination="bin"), decline], evidence=[],
                   recommendation="decline", care="normal", visibility=1.0, episode=0)
    d = a.to_data()
    assert d["guard"] == {"guard": "knn", "score": 4.212, "p": 0.0244, "flag": True, "alpha": 0.05, "k": 10, "n_calibration": 40}
    assert d["recommended_option"] is None and "guard" not in d["options"][-1]
    json.dumps(d)
    plain = Assessment(object="mug", destination="plate", options=[_row(), _decline()], evidence=[], recommendation="commit",
                       care="normal", visibility=1.0, episode=0)
    assert plain.to_data()["guard"] is None


def test_best_ignores_rows_for_other_destinations():
    inboard = _row(id="inboard_place", destination="table_centre")
    same = _row(id="slow_lift", destination="table_edge")
    a = Assessment(object="mug", destination="table_edge", options=[inboard, same, _decline()],
                   evidence=[], recommendation="commit", care="normal", visibility=1.0, episode=0)
    assert a.best().id == "slow_lift"
    assert a.to_data()["best_option"] == "slow_lift"


def test_prediction_mean_averages_and_drops_height_when_any_none():
    a = _pred(placed_stable=0.8, dropped_high=0.01234)
    b = Prediction(p={k: 1 / 7 for k in OUTCOMES}, drop_height=None, expected_loss_t=None)
    m = Prediction.mean([a, b])
    assert abs(m.p["placed_stable"] - (0.8 + 1 / 7) / 2) < 1e-9
    assert m.drop_height is None and m.expected_loss_t is None
    m2 = Prediction.mean([_pred(), _pred()])
    assert m2.drop_height == _pred().drop_height and m2.expected_loss_t == 1.2


def _probe_row(id="probe_push", expected_cost=0.5, evoi=0.2):
    return Option(id=id, kind="probe", action={"probe": id.split("_", 1)[1], "s": 0.5, "force": 20.0}, pred=None, cost_s=2.0,
                  expected_cost=expected_cost, reason="push first",
                  probe={"p_kind": {"slid": 0.9, "tipped": 0.1}, "p_thrown": 0.02, "evoi": evoi, "evoi_se": 0.01, "expected_cost_after": 0.4})


def test_best_never_returns_a_probe():
    rows = [_probe_row(), _row(id="slow_lift"), _decline()]
    a = Assessment(object="mug", destination="plate", options=rows, evidence=[], recommendation="probe", care="normal",
                   visibility=1.0, episode=0)
    assert a.best().id == "slow_lift" and a.best_overall().id == "probe_push"
    d = a.to_data()
    assert d["best_option"] == "slow_lift" and d["p_success"] == 0.712            # the top-level numbers stay the commit row's
    assert d["recommendation"] == "probe" and d["recommended_option"] == "probe_push"
    probe = d["options"][0]
    assert probe["kind"] == "probe" and probe["p"] is None
    assert probe["probe"] == {"p_kind": {"slid": 0.9, "tipped": 0.1}, "p_thrown": 0.02, "evoi": 0.2, "evoi_se": 0.01, "expected_cost_after": 0.4}
    assert d["options"][1]["probe"] is None
    json.dumps(d)
    commit = Assessment(object="mug", destination="plate", options=[_row(id="slow_lift"), _probe_row(), _decline()],
                        evidence=[], recommendation="commit", care="normal", visibility=1.0, episode=0)
    assert commit.best_overall().id == "slow_lift" and commit.to_data()["recommended_option"] == "slow_lift"
    decline = Assessment(object="mug", destination="plate", options=[_row(id="slow_lift"), _decline()], evidence=[],
                         recommendation="decline", care="normal", visibility=1.0, episode=0)
    assert decline.to_data()["recommended_option"] is None
