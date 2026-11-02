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
"""Part B of the mass/CoM check: the priors side by side, and each prior's first choice priced on the class prior's menu."""

import json

import pytest


def _m(rec, opt, costs, guard=False):
    return {"recommendation": rec, "recommended_option": opt, "guard_flag": guard, "costs": costs}


def test_choice_of_a_menu_is_its_recommended_row_or_decline():
    from robots.mujoco_lab.masscom_report import choice_of

    assert choice_of(_m("commit", "firm_grip", {})) == "firm_grip"
    assert choice_of(_m("decline", None, {})) == "decline"
    assert choice_of({"error": "boom"}) is None


def test_regret_prices_each_priors_choice_on_the_class_menu():
    from robots.mujoco_lab.masscom_report import regret_vs_class

    costs = {"decline": 10.0, "firm_grip": 4.0, "probe_weigh": 6.0, "inboard_place": None}
    recs = [{"object": "a", "idx": 0, "setting": "s", "menus": {"class": _m("commit", "firm_grip", costs),
                                                                 "vlm": _m("decline", None, {"decline": 10.0})}},
            {"object": "b", "idx": 0, "setting": "s", "menus": {"class": _m("commit", "firm_grip", costs),
                                                                 "vlm": _m("probe", "probe_weigh", {})}},
            {"object": "c", "idx": 0, "setting": "s", "menus": {"class": _m("commit", "firm_grip", costs),
                                                                 "vlm": _m("commit", "slow_lift", {})}},
            {"object": "d", "idx": 0, "setting": "s", "menus": {"class": _m("commit", "firm_grip", costs),
                                                                 "vlm": {"error": "boom"}}}]
    v = regret_vs_class(recs, n_boot=50, seed=0)["s"]["vlm"]
    assert v["n"] == 2 and v["mean"] == pytest.approx((6.0 + 2.0) / 2)       # decline 10 − 4; probe 6 − 4
    assert v["n_unmatched"] == 1 and v["n_left_out"] == 1 and len(v["ci"]) == 2
    assert v["class_own"] == pytest.approx(0.0)                             # the class rule's own choice is its cheapest row


def test_compare_puts_the_priors_side_by_side(tmp_path):
    from robots.mujoco_lab.masscom_report import compare

    cell = {"median": 0.5, "q25": 0.4, "q75": 0.6, "ci": [0.45, 0.55]}
    stages = {st: {p: {"n": 10, "n_objects": 2, "err": cell, "width": cell, "coverage": {"n": 10, "n_objects": 2, "coverage": 0.8, "ci": [0.7, 0.9]}}
                   for p in ("mass", "mu", "com")} for st in ("prior", "attempt", "weigh", "push")}
    down = {"n": 10, "n_objects": 2, "brier": {"value": 0.3, "ci": [0.2, 0.4]}, "ece_irreversible": {"value": 0.05, "ci": [0.01, 0.1]},
            "ece_placed": {"value": 0.1, "ci": [0.05, 0.2]}, "mean_p_irreversible": 0.1, "rate_irreversible": 0.09}
    dirs = {}
    for name in ("class", "generic"):
        d = tmp_path / name
        d.mkdir()
        (d / "summary.json").write_text(json.dumps({"stages": stages, "downstream": down}))
        dirs[name] = str(d)
    out, md = compare(dirs, None, n_boot=20, seed=0)
    assert set(out["priors"]) == {"class", "generic"} and out["priors"]["generic"]["downstream"]["brier"]["value"] == 0.3
    assert "| generic |" in md and "Brier" in md


def test_regret_leaves_out_the_row_that_changes_the_destination():
    """inboard_place sets the object down elsewhere: the menu shows it but never chooses it (menu.py), so it is not the
    cheapest row of a menu here either (as in e1_expected)."""
    from robots.mujoco_lab.masscom_report import regret_vs_class

    costs = {"decline": 10.0, "firm_grip": 4.0, "inboard_place": 1.0}
    recs = [{"object": "a", "idx": 0, "setting": "s", "menus": {"class": _m("commit", "firm_grip", costs),
                                                                 "vlm": _m("decline", None, {})}}]
    v = regret_vs_class(recs, n_boot=20, seed=0)["s"]["vlm"]
    assert v["mean"] == pytest.approx(6.0) and v["class_own"] == pytest.approx(0.0)


def test_regret_difference_is_paired_by_case():
    """diff: the prior's regret minus the class rule's own regret on the same case (the decision cost of the swap)."""
    from robots.mujoco_lab.masscom_report import regret_vs_class

    costs = {"decline": 10.0, "firm_grip": 4.0}
    recs = [{"object": "a", "idx": 0, "setting": "s", "menus": {"class": _m("commit", "firm_grip", costs), "vlm": _m("decline", None, {})}},
            {"object": "b", "idx": 0, "setting": "s", "menus": {"class": _m("decline", None, costs), "vlm": _m("commit", "firm_grip", {})}},
            {"object": "b", "idx": 1, "setting": "s", "menus": {"class": _m("decline", None, costs), "vlm": _m("commit", "firm_grip", {})}}]
    v = regret_vs_class(recs, n_boot=50, seed=0)["s"]["vlm"]
    assert v["mean"] == pytest.approx(2.0) and v["class_own"] == pytest.approx(4.0)      # (6 + 0 + 0) / 3; (0 + 6 + 6) / 3
    assert v["diff"]["mean"] == pytest.approx(-2.0) and len(v["diff"]["ci"]) == 2         # (+6 − 6 − 6) / 3
