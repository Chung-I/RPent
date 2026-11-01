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

"""The E1 pilot (robots.mujoco_lab.e1): display names, the case plan, the scoring, the menu views and the planner loop."""

from __future__ import annotations

import json

import numpy as np
import pytest

from robots.mujoco_lab import e1

TEMPLATES = {
    "default_rule": {"care_scale": 10, "abstain_cost": 10, "recovery": "absorbing"},
    "settings": {"hi_abs": {"care_scale": 30, "abstain_cost": 10, "recovery": "absorbing"},
                 "hi_rec": {"care_scale": 30, "abstain_cost": 10, "recovery": "rec20"},
                 "lo_abs": {"care_scale": 1, "abstain_cost": 10, "recovery": "absorbing"},
                 "lo_rec": {"care_scale": 1, "abstain_cost": 10, "recovery": "rec20"}},
    "families": {f: {"split": "heldout" if f == "F3" else "tune",
                     "templates": {s: [f"{f} {s} a: put the {{object}} on the {{destination}}.",
                                       f"{f} {s} b: move the {{object}} to the {{destination}}."]
                                   for s in ("hi_abs", "hi_rec", "lo_abs", "lo_rec")}}
                 for f in ("F1", "F2", "F3")},
}


def test_display_names_drop_the_dataset_and_the_number():
    assert e1.display_object("hope__orange_juice_carton") == "orange juice carton"
    assert e1.display_object("fruits_veggies__lemon1") == "lemon"
    assert e1.display_object("handal__hammer_6") == "hammer"
    assert e1.display_object("ycb__mug") == "mug"
    assert e1.display_destination("table_edge") == "table edge"
    assert e1.display_destination("mouse_pad") == "mouse pad"


def test_spread_cases_cover_every_object_before_a_second_case():
    cases = {"a": [1, 2, 3, 4], "b": [5], "c": [6, 7]}
    picked = e1.spread_cases(cases, 4, seed=0)
    assert len(picked) == 4 and {o for o, _ in picked} == {"a", "b", "c"}
    assert len(set(picked)) == 4 and all(i in cases[o] for o, i in picked)
    assert e1.spread_cases(cases, 4, seed=0) == picked


def test_the_final_plan_pairs_two_settings_per_case_and_uses_the_held_out_family():
    cases = {f"o{k}": list(range(5)) for k in range(6)}
    jobs = e1.plan_jobs(cases, TEMPLATES, n_cases=24, split="final", seed=12, plan_seed=0)
    assert len(jobs) == 48
    per = {s: sum(j["setting"] == s for j in jobs) for s in TEMPLATES["settings"]}
    assert per == {"hi_abs": 12, "hi_rec": 12, "lo_abs": 12, "lo_rec": 12}
    by_case = {}
    for j in jobs:
        by_case.setdefault((j["object"], j["idx"]), []).append(j["setting"])
    assert all(sorted(v) in (["hi_abs", "lo_rec"], ["hi_rec", "lo_abs"]) for v in by_case.values())
    assert all(j["family"] == "F3" and j["seed"] == 12 and j["split"] == "final" for j in jobs)
    assert all(j["template"] in TEMPLATES["families"]["F3"]["templates"][j["setting"]] for j in jobs)
    assert e1.instruction(jobs[0], "table_edge").endswith(".") and "{" not in e1.instruction(jobs[0], "table_edge")


def test_the_tuning_plan_uses_one_setting_per_case_and_only_the_tuning_families():
    cases = {f"o{k}": list(range(5)) for k in range(6)}
    jobs = e1.plan_jobs(cases, TEMPLATES, n_cases=8, split="tune", seed=13, plan_seed=0)
    assert len(jobs) == 8 and {j["family"] for j in jobs} <= {"F1", "F2"}
    assert {j["setting"] for j in jobs} == set(TEMPLATES["settings"])


def test_realized_cost_follows_the_hidden_setting():
    case = {"ended": "placed", "probes": [{"probe": "weigh", "kind": "measured", "thrown": False}],
            "attempts": [{"outcome": "dropped_high", "cost_s": 8.0}, {"outcome": "placed_stable", "cost_s": 8.0}]}
    s = TEMPLATES["settings"]
    # rec20: the drop costs the reset (20 s at 0.05) and its time; the weigh 4 s; the success its time
    assert e1.realized_cost(case, s["hi_rec"]) == pytest.approx(0.2 + (1.0 + 0.4) + 0.4)
    assert e1.realized_cost(case, s["lo_rec"]) == pytest.approx(0.2 + 1.4 + 0.4)
    failed = {"ended": "failed", "probes": [], "attempts": [{"outcome": "spilled", "cost_s": 9.0}]}
    assert e1.realized_cost(failed, s["hi_abs"]) == pytest.approx(5 * 30 + 0.45 + 10)
    assert e1.realized_cost(failed, s["lo_abs"]) == pytest.approx(5 * 1 + 0.45 + 10)
    assert e1.realized_cost({"ended": "abstained", "probes": [], "attempts": []}, s["lo_abs"]) == 10.0
    thrown = {"ended": "thrown", "probes": [{"probe": "push", "kind": "slid", "thrown": True}], "attempts": []}
    assert e1.realized_cost(thrown, s["hi_abs"]) == pytest.approx(0.1 + 150 + 10)


def _row(rid, kind="lower_risk", cost=12.0, p_ok=0.6, dest="table_edge"):
    p = {k: 0.0 for k in e1.OUTCOMES}
    p.update(placed_stable=p_ok, spilled=0.2, not_lifted=1 - p_ok - 0.2)
    return {"id": rid, "kind": kind, "action": {"force": 10.0, "v_lift": 0.15, "v_transport": 0.5, "a_transport": 2.0,
                                                "destination": dest, "release_height": None, "s": 0.5},
            "p": p, "cost_s": 8.0, "expected_cost": cost, "reason": "twice the squeeze", "probe": None}


def _probe_row(rid="probe_weigh", cost=9.0):
    return {"id": rid, "kind": "probe", "action": {"probe": rid.split("_")[1], "s": 0.5, "force": 20.0}, "p": None, "cost_s": 4.0,
            "expected_cost": cost, "reason": "weigh first: value of information 1.51",
            "probe": {"p_kind": {"measured": 0.6, "lower_bound": 0.4}, "p_thrown": 0.01, "evoi": 1.5, "evoi_se": 0.2,
                      "expected_cost_after": 8.8}}


def _menu(rec="probe", recommended="probe_weigh"):
    rows = [_probe_row(), _row("firm_grip", cost=17.6), _row("commit_std", "commit", cost=19.5),
            _row("inboard_place", cost=19.1, dest="table_centre"),
            {"id": "decline", "kind": "decline", "action": {}, "p": None, "cost_s": 0.0, "expected_cost": 10.0,
             "reason": "best row costs 17.57 > abstain 10.00: spilled carries the largest expected cost", "probe": None}]
    return {"assessed": True, "destination": "table_edge", "best_option": "firm_grip", "options": rows, "recommendation": rec,
            "recommended_option": recommended, "without_probe": "decline", "decline_reason": rows[-1]["reason"],
            "evidence": ["mass median 0.268 kg, 80 % interval 0.094 to 0.403 kg"], "guard": {"flag": False}}


def test_the_menu_view_shows_every_row_but_inboard_place_with_its_numbers():
    text, ids = e1.render_menu(_menu(), "menu_planner", "table_edge", np.random.default_rng(0))
    assert set(ids) == {"probe_weigh", "firm_grip", "commit_std", "decline"} and "inboard_place" not in text
    assert "17.60" in text and "probe_weigh" in text and "Recommendation: probe" in text


def test_the_rec_view_shows_the_recommended_row_and_decline_only():
    text, ids = e1.render_menu(_menu(), "rec_planner", "table_edge", np.random.default_rng(0))
    assert ids == ["probe_weigh", "decline"] and "firm_grip" not in text
    text, ids = e1.render_menu(_menu("decline", None), "rec_planner", "table_edge", np.random.default_rng(0))
    assert ids == ["firm_grip", "decline"] and "declines" in text


def test_the_nomenu_view_shows_actions_without_numbers():
    text, ids = e1.render_menu(_menu(), "nomenu_planner", "table_edge", np.random.default_rng(0))
    assert set(ids) == {"probe_weigh", "firm_grip", "commit_std", "decline"}
    assert "17.6" not in text and "0.60" not in text and "Recommendation" not in text and "value of information" not in text


def test_parse_choice_reads_a_tool_call_or_a_json_object():
    assert e1.parse_choice({"tool": "choose", "args": {"option_id": "firm_grip", "reason": "x"}}, ["firm_grip", "decline"]) == "firm_grip"
    assert e1.parse_choice({"tool": None, "args": None, "content": 'I pick {"option_id": "decline"}'}, ["firm_grip", "decline"]) == "decline"
    assert e1.parse_choice({"tool": "choose", "args": {"option_id": "slow_lift"}}, ["firm_grip", "decline"]) is None


class FakeTools:
    """A scripted lab: the first menu recommends decline; an attempt places the object."""

    def __init__(self, outcome="placed_stable"):
        self.outcome, self.calls = outcome, []

    def assess(self):
        return _menu("decline", None)

    def act(self, option_id):
        self.calls.append(option_id)
        return {"outcome": self.outcome, "placed": self.outcome == "placed_stable", "placed_at_requested": self.outcome == "placed_stable",
                "option": option_id, "damage": None, "readouts": {"rose": True}, "diagnosis": [], "artifact": False,
                "assessment": _menu("decline", None)}


class ScriptModel:
    def __init__(self, choices):
        self.choices, self.n = list(choices), 0

    def __call__(self, messages, tools):
        self.n += 1
        c = self.choices.pop(0)
        if c is None:
            return {"tool": None, "args": None, "content": "no idea", "reasoning": "", "seconds": 0.0, "usage": {}, "error": None}
        return {"tool": "choose", "args": {"option_id": c, "reason": "because"}, "content": "", "reasoning": "r", "seconds": 0.0,
                "usage": {}, "error": None}


def test_the_planner_loop_counts_an_override_and_ends_when_placed():
    tools, model = FakeTools(), ScriptModel(["firm_grip"])
    case = e1.planner_loop(tools, model, "menu_planner", "Put it there.", "object", "table_edge", np.random.default_rng(0))
    assert case["ended"] == "placed" and tools.calls == ["firm_grip"] and case["overrides"] == 1 and len(case["calls"]) == 1
    assert case["attempts"][0]["outcome"] == "placed_stable" and case["attempts"][0]["cost_s"] == 8.0


def test_the_planner_loop_retries_a_malformed_call_and_stops_at_the_cap():
    tools, model = FakeTools(outcome="tipped"), ScriptModel([None, "firm_grip", "firm_grip", "firm_grip"])
    case = e1.planner_loop(tools, model, "menu_planner", "Put it there.", "object", "table_edge", np.random.default_rng(0),
                           max_calls=6)
    assert case["malformed"] == 1 and case["ended"] == "failed" and len(case["attempts"]) == 3
    tools, model = FakeTools(outcome="tipped"), ScriptModel([None] * 6)
    case = e1.planner_loop(tools, model, "menu_planner", "Put it there.", "object", "table_edge", np.random.default_rng(0),
                           max_calls=6)
    assert case["ended"] == "call_cap" and case["malformed"] == 6 and tools.calls == []


def test_the_planner_loop_declines():
    tools, model = FakeTools(), ScriptModel(["decline"])
    case = e1.planner_loop(tools, model, "rec_planner", "Put it there.", "object", "table_edge", np.random.default_rng(0))
    assert case["ended"] == "abstained" and tools.calls == [] and case["overrides"] == 0


def test_object_bootstrap_interval_contains_the_mean():
    rng = np.random.default_rng(0)
    rows = [{"object": f"o{k % 5}", "d": float(v)} for k, v in enumerate(rng.normal(1.0, 0.5, 50))]
    mean, lo, hi = e1.object_bootstrap([r["d"] for r in rows], [r["object"] for r in rows], n_boot=500, seed=0)
    assert lo < mean < hi and mean == pytest.approx(np.mean([r["d"] for r in rows]))
