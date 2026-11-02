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


"""First-menu decisions under each prior: the change rate against the class prior."""

import os

import pytest


def _m(rec, opt, guard=False):
    return {"recommendation": rec, "recommended_option": opt, "guard_flag": guard}


def test_change_rates_count_changed_recommendations_against_the_class_prior():
    from robots.mujoco_lab.first_menus import change_rates

    recs = [{"object": "a", "idx": 0, "setting": "s", "menus": {"class": _m("commit", "x"), "vlm": _m("commit", "y")}},
            {"object": "b", "idx": 0, "setting": "s", "menus": {"class": _m("probe", "p"), "vlm": _m("decline", "decline")}},
            {"object": "b", "idx": 1, "setting": "t", "menus": {"class": _m("commit", "x"), "vlm": _m("commit", "x")}}]
    r = change_rates(recs, n_boot=50, seed=0)
    v = r["s"]["vlm"]
    assert v["n"] == 2 and v["recommendation"]["rate"] == 0.5 and v["option"]["rate"] == 1.0
    assert v["transitions"] == {"probe->decline": 1} and len(v["recommendation"]["ci"]) == 2
    assert r["t"]["vlm"]["recommendation"]["rate"] == 0.0 and "class" not in r["s"]


def test_change_rates_leave_out_a_case_with_an_error_under_either_prior():
    from robots.mujoco_lab.first_menus import change_rates

    recs = [{"object": "a", "idx": 0, "setting": "s", "menus": {"class": _m("commit", "x"), "vlm": {"error": "boom"}}},
            {"object": "b", "idx": 0, "setting": "s", "menus": {"class": _m("commit", "x"), "vlm": _m("decline", "decline")}}]
    v = change_rates(recs, n_boot=20, seed=0)["s"]["vlm"]
    assert v["n"] == 1 and v["n_left_out"] == 1 and v["recommendation"]["rate"] == 1.0


M2B2 = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "robots", "mujoco_lab", "data", "m2b2")
POLICY = os.path.join(M2B2, "calibration_crossfit_policy_care_10_abstain_10_time_1.json")


@pytest.mark.skipif(not os.path.exists(POLICY) or not os.environ.get("RPENT_MUJOCO_SIM_DIR"),
                    reason="the m2b2 bundle's crossfit run or the lab's sim dir is not here")
def test_class_menus_repeat_the_recorded_crossfit_policy_run():
    """Same fold bundle, case draw and menu stream: the class prior's first menu at care 10, abstain cost 10 equals the
    first menu of calibrate's crossfit policy run (seed 11)."""
    import json

    from robots.mujoco_lab.first_menus import run_object

    folds = json.load(open(os.path.join(M2B2, "folds", "folds.json")))
    o = "handal__hammer_2"
    s = {"bundle": os.path.join(M2B2, "folds", f"fold_{folds[o]}"), "fold": folds[o], "seed": 11,
         "settings": [("k10_A10", 10.0, 10.0)], "tables": {}}
    rows = run_object((o, [1, 2], s))
    recorded = {r["idx"]: r for r in json.load(open(POLICY))["rows"] if r["object"] == o}
    assert len(rows) == 2
    for r in rows:
        m, c = r["menus"]["class"], recorded[r["idx"]]
        assert (m["recommendation"], m["best_option"]) == (c["recommendation"], c["best_option"])
        assert r["destination"] == c["destination"]
