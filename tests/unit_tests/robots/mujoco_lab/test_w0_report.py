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


"""W0: the decision value of an exact CoM at the first menu (first_menus --exact records)."""

import pytest


def _m(rec, opt, costs):
    return {"recommendation": rec, "recommended_option": opt, "costs": costs}


COSTS_B = {"x": 5.0, "y": 7.0, "probe_push": 6.0, "inboard_place": 1.0, "decline": 3.0}   # the answer's menu (exact_both)
COSTS_C = {"x": 4.0, "y": 2.0, "probe_push": 6.0, "decline": 3.0}                         # the class menu


def _recs():
    return [{"object": "a", "idx": 0, "setting": "s", "menus": {
                "generic": _m("commit", "y", {}), "exact_com": _m("commit", "x", {}), "class": _m("commit", "y", COSTS_C),
                "exact_both": _m("decline", None, COSTS_B)}},
            {"object": "b", "idx": 0, "setting": "s", "menus": {
                "generic": _m("decline", None, {}), "exact_com": _m("probe", "probe_push", {}), "class": _m("commit", "x", COSTS_C),
                "exact_both": _m("commit", "x", COSTS_B)}},
            {"object": "b", "idx": 1, "setting": "s", "menus": {
                "generic": _m("commit", "z", {}), "exact_com": _m("commit", "x", {}), "class": {"error": "boom"},
                "exact_both": _m("commit", "x", COSTS_B)}}]


def test_paired_cost_prices_the_prior_and_the_reference_on_the_named_menu():
    """Each case: the prior's first choice minus the generic prior's, both priced on one menu (decline at its row); a choice
    that menu has no row for is unmatched, a menu with an error is left out."""
    from robots.mujoco_lab.w0_report import paired_cost

    on_b = paired_cost(_recs(), menu="exact_both", ref="generic", n_boot=50, seed=0)["s"]
    v = on_b["exact_com"]
    assert v["n"] == 2 and v["n_unmatched"] == 1 and v["n_left_out"] == 0      # case b 1: generic's row z is not on the menu
    assert v["mean"] == pytest.approx(((5.0 - 7.0) + (6.0 - 3.0)) / 2) and len(v["ci"]) == 2
    assert on_b["exact_both"]["mean"] == pytest.approx(((3.0 - 7.0) + (5.0 - 3.0)) / 2)
    on_c = paired_cost(_recs(), menu="class", ref="generic", n_boot=50, seed=0)["s"]
    assert on_c["exact_com"]["n"] == 2 and on_c["exact_com"]["n_left_out"] == 1
    assert on_c["exact_com"]["mean"] == pytest.approx(((4.0 - 2.0) + (6.0 - 3.0)) / 2)
    assert "generic" not in on_b


def test_paired_cost_splits_the_cases_by_group():
    from robots.mujoco_lab.w0_report import paired_cost

    g = paired_cost(_recs(), menu="exact_both", ref="generic", n_boot=50, seed=0, group={"a": "flat", "b": "upright"})
    by = g["s"]["exact_com"]["groups"]
    assert by["flat"]["n"] == 1 and by["flat"]["mean"] == pytest.approx(-2.0)
    assert by["upright"]["n"] == 1 and by["upright"]["mean"] == pytest.approx(3.0)


def test_object_gains_rank_the_objects_by_the_cost_the_exact_com_saves():
    """Gain: the generic choice's cost minus the exact-CoM choice's on the answer's menu, per object and setting, with
    the choices that changed."""
    from robots.mujoco_lab.w0_report import object_gains

    recs = _recs()
    recs[0]["truth"] = {"mass": 1.0, "com_frac": 0.8, "sim_mass": 1.0, "sim_com_frac": 0.8}
    g = object_gains(recs, menu="exact_both", prior="exact_com", ref="generic")
    assert [o for o, _ in g] == ["a", "b"]                                     # a saves 2, b loses 3
    a = dict(g)["a"]
    assert a["mean_gain"] == pytest.approx(2.0) and a["settings"]["s"]["gain"] == pytest.approx(2.0)
    assert a["settings"]["s"]["changes"] == {"y->x": 1} and a["com_frac_mean"] == pytest.approx(0.8)
    assert dict(g)["b"]["settings"]["s"]["n"] == 1                              # its case 1 is unmatched
