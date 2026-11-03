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

"""The help ladder's inputs (robots.mujoco_lab.ladder): the posterior, the predictions, the values, the lowest-value
check against the one-step rule, the rung views and the case list."""

from __future__ import annotations

import json

import numpy as np
import pytest

from robots.mujoco_lab import e1, ladder
from rpent.skills.pick_place.belief import ParticleBelief


def _row(rid, kind="lower_risk", cost=2.5, p_ok=0.6, dest="table_edge"):
    p = {k: 0.0 for k in e1.OUTCOMES}
    p.update(placed_stable=p_ok, crushed=0.1, not_lifted=1 - p_ok - 0.1)
    return {"id": rid, "kind": kind, "action": {"force": 10.0, "v_lift": 0.15, "v_transport": 0.5, "a_transport": 2.0,
                                                "destination": dest, "release_height": None, "s": 0.5},
            "p": p, "cost_s": 8.0, "expected_cost": cost, "reason": "twice the squeeze", "probe": None}


def _probe_row(rid="probe_weigh", cost=2.0, evoi=0.7, se=0.5):
    return {"id": rid, "kind": "probe", "action": {"probe": rid.split("_")[1], "s": 0.5, "force": 20.0}, "p": None, "cost_s": 4.0,
            "expected_cost": cost, "reason": f"weigh first: value of information {evoi:.2f}",
            "probe": {"p_kind": {"measured": 0.6, "lower_bound": 0.4}, "p_thrown": 0.01, "evoi": evoi, "evoi_se": se,
                      "expected_cost_after": 1.8}}


def _menu(rows, rec="commit", recommended="light_grip", guard=False, A=3.0):
    rows = sorted(rows, key=lambda o: o["expected_cost"]) + [
        {"id": "decline", "kind": "decline", "action": {}, "p": None, "cost_s": 0.0, "expected_cost": A,
         "reason": "best row costs 2.50 <= abstain 3.00: crushed carries the largest expected cost", "probe": None}]
    return {"assessed": True, "destination": "table_edge", "options": rows, "recommendation": rec,
            "recommended_option": recommended, "best_option": "light_grip", "without_probe": "commit",
            "decline_reason": rows[-1]["reason"], "evidence": [], "guard": {"flag": guard}}


def _a():
    return _menu([_row("light_grip", cost=2.5), _row("inboard_place", cost=1.0, dest="table_centre"), _probe_row(cost=2.0)])


def test_values_are_each_shown_rows_expected_cost_and_decline_at_A():
    d = ladder.values(_a())
    assert d["rows"] == {"probe_weigh": 2.0, "light_grip": 2.5, "decline": 3.0}
    assert d["probes"] == {"probe_weigh": {"evoi": 0.7, "evoi_se": 0.5, "expected_cost_after": 1.8}}


def test_predictions_carry_the_probabilities_and_the_time_but_no_cost():
    c = ladder.predictions(_a())
    assert [r["id"] for r in c] == ["probe_weigh", "light_grip"]          # the menu's order; inboard_place and decline left out
    pr, att = c
    assert att["kind"] == "lower_risk" and att["p"]["placed_stable"] == 0.6 and att["time_s"] == 8.0
    assert att["action"]["force"] == 10.0 and att["what"] == "twice the squeeze"
    assert pr["p_kind"] == {"measured": 0.6, "lower_bound": 0.4} and pr["p_thrown"] == 0.01 and pr["time_s"] == 4.0
    text = json.dumps(c)
    assert "expected_cost" not in text and "evoi" not in text and "value of information" not in text


def test_the_rule_choice_is_the_recommended_row_or_decline():
    assert ladder.rule_choice(_a()) == "light_grip"
    assert ladder.rule_choice(_menu([_row("light_grip"), _probe_row()], "probe", "probe_weigh")) == "probe_weigh"
    assert ladder.rule_choice(_menu([_row("light_grip", cost=4.0)], "decline", None)) == "decline"


def test_the_lowest_value_check_matches_the_rule_and_names_each_mismatch():
    # the rule probes: the probe is the lowest value
    ok = ladder.lowest_value_check(_menu([_row("light_grip", cost=2.5), _probe_row(cost=1.0, se=0.1)], "probe", "probe_weigh"))
    assert ok["lowest"] == "probe_weigh" and ok["rule"] == "probe_weigh" and ok["match"] and ok["reason"] is None
    # the probe is the lowest value, but it beats the best row by less than EVOI_Z errors: the rule commits
    pm = ladder.lowest_value_check(_a())
    assert pm["lowest"] == "probe_weigh" and pm["rule"] == "light_grip" and not pm["match"] and pm["reason"] == "probe_margin"
    # the guard declines a row that costs less than A
    g = ladder.lowest_value_check(_menu([_row("light_grip", cost=2.5)], "decline", None, guard=True))
    assert g["lowest"] == "light_grip" and g["rule"] == "decline" and g["reason"] == "guard"
    # two rows show the same value; the rule's unrounded costs chose the other one
    t = ladder.lowest_value_check(_menu([_row("light_grip", cost=2.5), _row("slow_lift", cost=2.5)], "commit", "slow_lift"))
    assert t["lowest"] == "light_grip" and t["rule"] == "slow_lift" and t["ties"] == ["light_grip", "slow_lift"]
    assert t["reason"] == "tie"
    # anything else is a ladder bug
    o = ladder.lowest_value_check(_menu([_row("light_grip", cost=2.5), _row("slow_lift", cost=2.7)], "commit", "slow_lift"))
    assert o["reason"] == "other"


def _belief():
    rows = [(0.2, 0.3, 0.50, 0.0, 30.0, 0.1), (0.4, 0.5, 0.55, 0.2, 40.0, 0.1), (0.6, 0.7, 0.70, 0.3, 50.0, 0.1),
            (0.8, 0.9, 0.40, 0.0, 60.0, 0.1)]
    parts = [{"mass": m, "mu": mu, "com_obj": [0.0, 0.0, 0.0], "com_frac": cf, "f": f, "f_damage_n": fd, "v_bruise_ms": None,
              "fill": fill, "spill_angle_deg": None, "k_n_per_mm": None} for m, mu, cf, f, fd, fill in rows]
    return ParticleBelief(parts, np.array([0.1, 0.2, 0.3, 0.4]))


def test_the_posterior_summarizes_the_weighted_particles_and_drops_absent_keys():
    b = _belief()
    post = ladder.posterior(b)
    for key in ("mass", "mu", "com_frac", "f_damage_n", "fill"):
        assert post[key] == {"median": b.quantile(key, 0.5), "p10": b.quantile(key, 0.1), "p90": b.quantile(key, 0.9)}
    assert "v_bruise_ms" not in post and "spill_angle_deg" not in post and "k_n_per_mm" not in post
    assert post["p_com_shifted"] == pytest.approx(0.2 + 0.3)
    assert post["n_observations"] == 0 and post["ess"] == pytest.approx(b.effective_size())


def test_each_rung_adds_its_part_to_the_rung_below():
    a = _a()
    inputs = {"B": ladder.posterior(_belief()), "C": ladder.predictions(a), "D": ladder.values(a)}
    views = {r: ladder.render_state(r, inputs, "table_edge", np.random.default_rng(0)) for r in ladder.RUNGS}
    for text, ids in views.values():
        assert set(ids) == {"light_grip", "probe_weigh", "decline"} and ids[-1] == "decline"
        assert "inboard_place" not in text and "Recommendation" not in text and "guard" not in text
        assert "pinch at 0.50 of the long axis" in text and "about 8 s" in text
    A, B, C, D = (views[r][0] for r in ladder.RUNGS)
    assert "belief" not in A and "P(placed_stable)" not in A and "expected cost" not in A
    assert "The skill's belief" in B and "mass 0.6" in B and "crush" in B and "P(placed_stable)" not in B
    assert "P(placed_stable) 0.60" in C and "result kinds" in C and "expected cost" not in C and "value of information" not in C
    assert "expected cost 2.50" in D and "expected cost 2.00" in D and "value of information 0.70" in D and "cost 3.00" in D


def test_the_rows_order_follows_the_rng_and_decline_stays_last():
    a = _menu([_row(f"r{i}", cost=1.0 + i) for i in range(6)], "commit", "r0", A=10.0)
    inputs = {"B": ladder.posterior(_belief()), "C": ladder.predictions(a), "D": ladder.values(a)}
    orders = {tuple(ladder.render_state("D", inputs, "table_edge", np.random.default_rng(s))[1]) for s in range(8)}
    assert len(orders) > 1 and all(o[-1] == "decline" for o in orders)


def test_cases_come_from_a_gap_cases_file_or_a_list(tmp_path):
    p = tmp_path / "cases.json"
    p.write_text(json.dumps({"seed": 13, "cases": [["ycb__mustard", 2], ["hope__butter", 0]]}))
    assert ladder.load_cases(str(p)) == [{"object": "ycb__mustard", "idx": 2, "seed": 13}, {"object": "hope__butter", "idx": 0, "seed": 13}]
    q = tmp_path / "list.json"
    q.write_text(json.dumps([{"object": "ycb__mustard", "idx": 2, "seed": 14, "setting": "k100_A3_abs"}]))
    assert ladder.load_cases(str(q)) == [{"object": "ycb__mustard", "idx": 2, "seed": 14, "setting": "k100_A3_abs"}]


def test_settings_are_named_as_in_the_gap_grid():
    s = ladder.parse_settings("k100_A3_abs,k30_A3_abs")
    assert s == {"k100_A3_abs": {"care_scale": 100.0, "abstain_cost": 3.0, "recovery": "absorbing"},
                 "k30_A3_abs": {"care_scale": 30.0, "abstain_cost": 3.0, "recovery": "absorbing"}}
    with pytest.raises(ValueError):
        ladder.parse_settings("k7_A3_abs")


def test_smoke_cases_leave_out_the_excluded_cases_and_cycle_the_settings():
    never = {"seed": 13, "rows": [{"object": f"obj{o}", "idx": i, "recommendation": "commit"} for o in range(8) for i in range(3)]}
    exclude = {(13, "obj0", 0), (13, "obj1", 0), (13, "obj1", 1), (13, "obj1", 2)}
    picked = ladder.pick_smoke(never, 6, 4, exclude, ("k100_A3_abs", "k30_A3_abs", "k100_A10_abs"))
    assert len(picked) == 6 and not {(13, c["object"], c["idx"]) for c in picked} & exclude
    assert len({c["object"] for c in picked}) == 6 and all(c["seed"] == 13 for c in picked)
    assert [c["setting"] for c in picked] == ["k100_A3_abs", "k30_A3_abs", "k100_A10_abs"] * 2
    assert picked == ladder.pick_smoke(never, 6, 4, exclude, ("k100_A3_abs", "k30_A3_abs", "k100_A10_abs"))
