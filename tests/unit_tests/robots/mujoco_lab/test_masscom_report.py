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


def test_regret_vs_generic_is_paired_by_case():
    """vs_generic: a prior's regret minus the generic prior's on the same case and setting (both on the class menu)."""
    from robots.mujoco_lab.masscom_report import regret_vs_class

    costs = {"decline": 10.0, "firm_grip": 4.0}
    recs = [{"object": "a", "idx": 0, "setting": "s", "menus": {"class": _m("commit", "firm_grip", costs),
                                                                 "generic": _m("decline", None, {}), "vlm": _m("commit", "firm_grip", {})}},
            {"object": "b", "idx": 0, "setting": "s", "menus": {"class": _m("commit", "firm_grip", costs),
                                                                 "generic": _m("commit", "firm_grip", {}), "vlm": _m("commit", "firm_grip", {})}},
            {"object": "c", "idx": 0, "setting": "s", "menus": {"class": _m("commit", "firm_grip", costs),
                                                                 "generic": {"error": "boom"}, "vlm": _m("decline", None, {})}}]
    out = regret_vs_class(recs, n_boot=50, seed=0)["s"]
    v = out["vlm"]["vs_generic"]
    assert v["n"] == 2 and v["mean"] == pytest.approx(-3.0) and len(v["ci"]) == 2     # (0 − 6 + 0 − 0) / 2; c has no generic regret
    assert "vs_generic" not in out["generic"]


def _att(o, k, p_placed, truth=1.0, label="placed_stable"):
    from rpent.skills.pick_place.protocol import OUTCOMES

    return {"stage": "attempt", "object": o, "key": k, "label": label, "prior": {"mass": {"truth": truth}},
            "p_prior": {OUTCOMES[0]: p_placed, OUTCOMES[1]: 1.0 - p_placed}}


def test_paired_brier_pairs_rows_by_object_key_and_occurrence():
    """Keys repeat across the corpus files, so the k-th row of an (object, key) pairs with the k-th of the other run."""
    from robots.mujoco_lab.masscom_report import paired_brier

    a = [_att("x", "k1", 1.0, truth=1.0), _att("x", "k1", 0.5, truth=2.0), _att("y", "k2", 0.5, truth=3.0)]
    b = [_att("y", "k2", 1.0, truth=3.0), _att("x", "k1", 1.0, truth=1.0), _att("x", "k1", 1.0, truth=2.0)]
    d = paired_brier(a, b, n_boot=50, seed=0)
    assert d["n"] == 3 and d["mean"] == pytest.approx((0.0 + 0.5 + 0.5) / 3) and len(d["ci"]) == 2   # Brier of 0.5 / 0.5 is 0.5


def test_paired_brier_refuses_rows_that_do_not_pair():
    from robots.mujoco_lab.masscom_report import paired_brier

    with pytest.raises(ValueError):
        paired_brier([_att("x", "k1", 1.0, truth=1.0)], [_att("x", "k1", 1.0, truth=1.5)], n_boot=20, seed=0)


def test_compare_pairs_each_prior_with_the_generic_one(tmp_path):
    """With the Part A records at hand, each prior but class and generic gets its Brier minus the generic prior's."""
    from robots.mujoco_lab.masscom_report import compare

    cell = {"median": 0.5, "q25": 0.4, "q75": 0.6, "ci": [0.45, 0.55]}
    stages = {st: {p: {"n": 10, "n_objects": 2, "err": cell, "width": cell, "coverage": {"n": 10, "n_objects": 2, "coverage": 0.8, "ci": [0.7, 0.9]}}
                   for p in ("mass", "mu", "com")} for st in ("prior", "attempt", "weigh", "push")}
    down = {"n": 10, "n_objects": 2, "brier": {"value": 0.3, "ci": [0.2, 0.4]}, "ece_irreversible": {"value": 0.05, "ci": [0.01, 0.1]},
            "ece_placed": {"value": 0.1, "ci": [0.05, 0.2]}, "mean_p_irreversible": 0.1, "rate_irreversible": 0.09}
    rows = {"generic": [_att("x", "k1", 1.0), _att("y", "k2", 1.0)], "vlm": [_att("x", "k1", 0.5), _att("y", "k2", 1.0)]}
    dirs = {}
    for name in ("class", "generic", "vlm"):
        d = tmp_path / name
        d.mkdir()
        (d / "summary.json").write_text(json.dumps({"stages": stages, "downstream": down}))
        if name in rows:
            (d / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[name]))
        dirs[name] = str(d)
    out, md = compare(dirs, None, n_boot=20, seed=0)
    v = out["vs_generic"]["vlm"]["brier"]
    assert v["n"] == 2 and v["mean"] == pytest.approx(0.25)          # (0.5 + 0) / 2
    assert set(out["vs_generic"]) == {"vlm"} and "Paired against the generic prior" in md


def test_compare_refuses_runs_of_two_generators_or_bundles(tmp_path):
    """masscom-v11 review: compare reads each input's run_config.json and refuses a mix of generators (a config without
    the field, or no config, ran v10) or of bundles, so a v10 row never sits beside v11 rows unlabelled; paired_brier
    refuses runs whose attempt rows do not pair at all (v10 keys o__d0 against v11 keys o__v11d0 gave n 0 and nan)."""
    from robots.mujoco_lab.masscom_report import compare, paired_brier

    down = {"n": 10, "n_objects": 2, "brier": {"value": 0.3, "ci": [0.2, 0.4]}, "ece_irreversible": {"value": 0.05, "ci": [0.01, 0.1]},
            "ece_placed": {"value": 0.1, "ci": [0.05, 0.2]}, "mean_p_irreversible": 0.1, "rate_irreversible": 0.09}

    def run(name, generator, bundle="robots/mujoco_lab/data/m2b2_v11"):
        d = tmp_path / name
        d.mkdir()
        (d / "summary.json").write_text(json.dumps({"stages": {}, "downstream": down}))
        (d / "run_config.json").write_text(json.dumps({"bundle": bundle} | ({} if generator is None else {"generator": generator})))
        return str(d)

    v11 = {"class": run("c11", "v11"), "generic": run("g11", "v11")}
    out, _ = compare(v11, None, n_boot=20, seed=0)
    assert out["generator"] == "v11"
    with pytest.raises(ValueError, match="more than one generator"):
        compare({"class": run("c10", None, "robots/mujoco_lab/data/m2b2"), "generic": v11["generic"]}, None, n_boot=20, seed=0)
    with pytest.raises(ValueError, match="more than one bundle"):
        compare({"class": v11["class"], "generic": run("g11b", "v11", "robots/mujoco_lab/data/other")}, None, n_boot=20, seed=0)
    with pytest.raises(ValueError, match="more than one generator"):
        compare(v11, run("m10", "v10", "robots/mujoco_lab/data/m2b2"), n_boot=20, seed=0)
    with pytest.raises(ValueError, match="no attempt rows pair"):
        paired_brier([_att("x", "x__v11d0", 1.0)], [_att("x", "x__d0", 1.0)], n_boot=20, seed=0)
