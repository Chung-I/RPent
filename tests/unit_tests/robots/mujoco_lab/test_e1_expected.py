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

"""The E1 expected-cost rescoring (robots.mujoco_lab.e1_expected): a choice's regret on a true menu, the recorded choices,
the replay of a recorded run, the true menu on copies and the two summaries."""

from __future__ import annotations

import numpy as np
import pytest

from robots.mujoco_lab import e1_expected as ex


def _menu(costs, A=10.0, rec="commit", recommended=None, best=None, guard=False, assessed=True):
    """A payload with the rows ``costs`` ({id: expected cost}; probe_* rows are probes, commit_std a commit row, the rest
    lower-risk rows) and decline at A."""
    kind = {k: "probe" if k.startswith("probe_") else "commit" if k == "commit_std" else "lower_risk" for k in costs}
    options = [{"id": k, "kind": kind[k], "expected_cost": c} for k, c in costs.items()]
    options.append({"id": "decline", "kind": "decline", "expected_cost": A})
    attempt = [k for k in costs if kind[k] != "probe" and k != "inboard_place"]
    best = best or (min(attempt, key=costs.get) if attempt else None)
    return {"assessed": assessed, "options": options, "recommendation": rec, "recommended_option": recommended,
            "best_option": best, "without_probe": None, "guard": {"flag": guard}}


def test_the_choosable_rows_leave_out_inboard_place_and_price_decline_at_A():
    a = _menu({"probe_squeeze": 3.15, "light_grip": 14.289, "inboard_place": 1.0}, A=3.0)
    assert ex.choosable(a) == {"probe_squeeze": 3.15, "light_grip": 14.289, "decline": 3.0}


def test_a_choice_scores_its_true_cost_minus_the_cheapest_row():
    a = _menu({"probe_squeeze": 3.15, "light_grip": 14.289, "commit_std": 60.72}, A=3.0)
    s = ex.score(a, "light_grip")
    assert (s["choice"], s["min_row"]) == ("light_grip", "decline")
    assert (s["cost"], s["min"], s["regret"]) == pytest.approx((14.289, 3.0, 11.289))
    assert ex.score(a, "decline")["regret"] == 0.0
    assert ex.score(a, "inboard_place") is None and ex.score(a, "slow_lift") is None


def test_a_rule_run_lists_its_probes_then_its_attempts_then_decline():
    run = {"ended": "abstained", "probes": [{"probe": "squeeze", "after_attempts": 0}], "attempts": [{"option": "light_grip"}]}
    assert ex.rule_actions(run) == ["probe_squeeze", "light_grip", "decline"]
    assert ex.rule_actions({"ended": "placed", "probes": [], "attempts": [{"option": "commit_std"}]}) == ["commit_std"]
    assert ex.rule_actions({"ended": "unreachable", "probes": [], "attempts": []}) == []
    with pytest.raises(ValueError):
        ex.rule_actions({"ended": "placed", "probes": [{"probe": "weigh", "after_attempts": 1}], "attempts": [{"option": "x"}]})


def test_a_planner_run_lists_its_choices_and_a_call_cap_declines_at_its_last_menu():
    run = {"ended": "placed", "decisions": [{"choice": "probe_weigh"}, {"choice": "firm_grip"}]}
    assert ex.planner_actions(run) == ["probe_weigh", "firm_grip"]
    assert ex.planner_actions({"ended": "call_cap", "decisions": [{"choice": "probe_weigh"}]}) == ["probe_weigh", "decline"]
    assert ex.planner_actions({"ended": "call_cap", "decisions": []}) == ["decline"]


def test_the_miss_reason_names_a_guard_decline_and_a_probe_below_the_margin():
    a = _menu({"probe_weigh": 4.0, "firm_grip": 6.0})
    assert ex.miss_reason(a, "probe_weigh") is None
    assert ex.miss_reason(a, "firm_grip") == "probe_margin"
    assert ex.miss_reason(_menu({"firm_grip": 6.0}, guard=True, rec="decline"), "decline") == "guard"
    assert ex.miss_reason(_menu({"firm_grip": 6.0}), "decline") == "other"
    assert ex.miss_reason(_menu({"firm_grip": 6.0}), "slow_lift") == "not_a_row"


def test_a_rule_record_gives_each_points_choice_menu_fields_and_outcome():
    run = {"ended": "abstained", "recommendation": "probe", "best_option": "light_grip", "guard": {"flag": False},
           "first_menu": {"best_cost": 14.289, "recommended_cost": 3.15},
           "probes": [{"probe": "squeeze", "kind": "measured", "thrown": False, "reading": 0.4, "after_attempts": 0}],
           "attempts": [{"option": "light_grip", "recommendation": "commit", "declared": {"expected_cost": 1.2},
                         "outcome": "slipped"}]}
    pts = ex.expectations(run, "default")
    assert [p["choice"] for p in pts] == ["probe_squeeze", "light_grip", "decline"]
    assert pts[0]["menu"] == {"recommendation": "probe", "best_option": "light_grip", "guard_flag": False, "best_cost": 14.289,
                              "recommended_cost": 3.15}
    assert pts[0]["outcome"] == {"probed": "squeeze", "kind": "measured", "thrown": False, "reading": 0.4}
    assert pts[1]["menu"] == {"recommendation": "commit", "cost:light_grip": 1.2} and pts[1]["outcome"] == {"outcome": "slipped"}
    assert pts[2] == {"choice": "decline", "menu": {}, "outcome": None}


def test_a_planner_record_gives_each_decisions_menu_fields_and_outcome():
    dec = [{"recommendation": "probe", "recommended_option": "probe_weigh", "best_option": "firm_grip", "guard_flag": False,
            "choice": "probe_weigh"},
           {"recommendation": "commit", "recommended_option": "firm_grip", "best_option": "firm_grip", "guard_flag": False,
            "choice": "firm_grip"},
           {"recommendation": "decline", "recommended_option": None, "best_option": "firm_grip", "guard_flag": True,
            "choice": "decline"}]
    run = {"ended": "abstained", "first_menu": {"without_probe": "commit", "recommended_option": "probe_weigh"}, "decisions": dec,
           "probes": [{"probe": "weigh", "option": "probe_weigh", "kind": "measured", "thrown": False, "reading": 0.3}],
           "attempts": [{"option": "firm_grip", "declared": {"expected_cost": 6.0}, "outcome": "slipped"}]}
    pts = ex.expectations(run, "menu_planner")
    assert [p["choice"] for p in pts] == ["probe_weigh", "firm_grip", "decline"]
    assert pts[0]["menu"] == {"recommendation": "probe", "recommended_option": "probe_weigh", "best_option": "firm_grip",
                              "guard_flag": False, "without_probe": "commit"}
    assert pts[0]["outcome"] == {"probed": "weigh", "kind": "measured", "thrown": False, "reading": 0.3}
    assert pts[1]["menu"]["cost:firm_grip"] == 6.0 and pts[1]["outcome"] == {"outcome": "slipped"}
    assert pts[2]["menu"]["guard_flag"] is True and pts[2]["outcome"] is None


class FakeLab:
    """A scripted lab: menu k after k actions; act returns the k-th outcome with the next menu."""

    def __init__(self, menus, outcomes):
        self.menus, self.outcomes, self.calls, self.store, self.destination = menus, outcomes, [], {"beliefs": 0}, "bin"

    def assess(self):
        return self.menus[0]

    def act(self, option_id):
        self.calls.append(option_id)
        k = len(self.calls)
        return dict(self.outcomes[k - 1], assessment=self.menus[k] if k < len(self.menus) else {"assessed": False})


WEIGHED = {"probed": "weigh", "kind": "measured", "thrown": False, "reading": 0.3}


def _points(*choices):
    out = []
    for c in choices:
        outcome = None if c == "decline" else dict(WEIGHED) if c.startswith("probe_") else {"outcome": "slipped"}
        out.append({"choice": c, "menu": {}, "outcome": outcome})
    return out


def test_the_replay_acts_in_order_and_builds_true_menus_after_the_first_point():
    m = [_menu({"probe_weigh": 4.0, "firm_grip": 6.0}, rec="probe"), _menu({"firm_grip": 5.0}), _menu({"firm_grip": 7.0})]
    lab = FakeLab(m, [WEIGHED, {"outcome": "slipped"}])
    built = []
    rep = ex.replay(lab, _points("probe_weigh", "firm_grip", "decline"),
                    build=lambda tools: built.append(list(lab.calls)) or {"s": "true menu"})
    assert lab.calls == ["probe_weigh", "firm_grip"] and built == [["probe_weigh"], ["probe_weigh", "firm_grip"]]
    assert rep["mismatch"] == [] and rep["menus"] == m and rep["true"] == [None, {"s": "true menu"}, {"s": "true menu"}]
    assert rep["first"] is m[0]


def test_the_replay_checks_the_menu_fields_and_stops_at_the_first_divergence():
    m = [_menu({"probe_weigh": 4.0, "firm_grip": 6.0}, rec="probe"), _menu({"firm_grip": 5.0})]
    lab = FakeLab(m, [dict(WEIGHED, thrown=True), {"outcome": "slipped"}])
    rep = ex.replay(lab, _points("probe_weigh", "firm_grip"))
    assert lab.calls == ["probe_weigh"] and len(rep["menus"]) == 1 and rep["true"] == [None]
    assert rep["mismatch"] == ["point 0 outcome thrown: recorded false, replayed true"]
    pts = _points("probe_weigh")
    pts[0]["menu"] = {"recommendation": "commit", "cost:firm_grip": 6.0}
    rep = ex.replay(FakeLab(m, [WEIGHED]), pts)
    assert rep["mismatch"] == ['point 0 menu recommendation: recorded "commit", replayed "probe"'] and rep["menus"] == []
    rep = ex.replay(FakeLab(m[:1], [WEIGHED]), _points("probe_weigh", "firm_grip"))
    assert rep["mismatch"] == ["point 1: no menu"]


class FakePrims:
    def __init__(self):
        self.rng, self.abstain_cost, self.menu_recovery = np.random.default_rng(5), 10.0, (None, "time")


def test_the_true_menu_runs_on_copies_and_leaves_the_run_as_it_was():
    prims, store = FakePrims(), {"beliefs": {"episode": 1}, "pick_place": {"cached": True}}
    seen = {}

    def assess(object, destination, care, primitives, store):
        seen.update(care=care, A=primitives.abstain_cost, recovery=primitives.menu_recovery, keys=sorted(store), dest=destination)
        primitives.rng.random(10)
        store["beliefs"]["episode"] = 99

        class R:
            data = {"assessed": True}
        return R()

    before = prims.rng.bit_generator.state
    out = ex.true_menu(prims, store, "bin", {"care": 30.0, "abstain_cost": 3.0, "menu_recovery": (20.0, "time")}, assess=assess)
    assert out == {"assessed": True}
    assert seen == {"care": 30.0, "A": 3.0, "recovery": (20.0, "time"), "keys": ["beliefs"], "dest": "bin"}
    assert prims.rng.bit_generator.state == before and (prims.abstain_cost, prims.menu_recovery) == (10.0, (None, "time"))
    assert store == {"beliefs": {"episode": 1}, "pick_place": {"cached": True}}


def test_a_run_scores_its_first_choice_on_the_fresh_true_menu_and_later_ones_on_its_own_belief():
    first = _menu({"probe_weigh": 4.0, "firm_grip": 6.0})
    s = ex.arm_score(["probe_weigh", "firm_grip"], first, [_menu({"firm_grip": 12.0})])
    assert s["regret_first"] == 0.0 and s["regret_later"] == pytest.approx(2.0) and s["regret_total"] == pytest.approx(2.0)
    assert s["points"] == 2 and s["first"]["min_row"] == "probe_weigh" and s["later"][0]["min_row"] == "decline"
    none = ex.arm_score([], {"assessed": False}, [])
    assert none["no_menu"] and none["regret_first"] == 0.0 and none["regret_total"] == 0.0
    lost = ex.arm_score(["probe_weigh", "firm_grip"], first, None)
    assert lost["regret_first"] == 0.0 and lost["regret_later"] is None and lost["regret_total"] is None
    assert ex.arm_score(["slow_lift"], first, [])["regret_first"] is None


def _rec(arm, setting, obj="mug", idx=0, first=0.0, total=0.0, rec="probe", ended="placed", artifact=False, miss=None,
         noguard=None):
    return {"object": obj, "idx": idx, "seed": 13, "arm": arm, "setting": setting, "rec": rec, "ended": ended,
            "artifact": artifact, "regret_first": first, "regret_later": None if total is None else total - first,
            "regret_total": total, "regret_noguard": total if noguard is None else noguard, "miss": miss, "misses_later": [],
            "points": 1, "mismatch": []}


def test_the_gap_pairs_take_default_minus_oracle_and_drop_errors_and_artifacts():
    recs = [_rec("default", "k1_A3_abs", first=2.0, total=3.0), _rec("oracle", "k1_A3_abs", first=0.5, total=0.5),
            _rec("default", "k1_A3_abs", idx=1, first=1.0, total=1.0, artifact=True), _rec("oracle", "k1_A3_abs", idx=1),
            _rec("default", "k1_A3_abs", idx=2, first=1.0, total=1.0), _rec("oracle", "k1_A3_abs", idx=2, ended="error")]
    prs = ex.gap_pairs(recs)
    assert [(p["idx"], p["kept"], p["drop"]) for p in prs] == [(0, True, None), (1, False, "artifact"), (2, False, "error")]
    assert prs[0]["first"] == pytest.approx(1.5) and prs[0]["total"] == pytest.approx(2.5) and prs[0]["later"] == pytest.approx(1.0)


def test_the_gap_summary_ranks_by_the_lower_bound_of_the_first_decision_interval():
    recs = []
    for k in range(6):
        recs += [_rec("default", "k1_A3_abs", obj=f"o{k}", first=1.0 + k % 2, total=2.0 + k % 2),
                 _rec("oracle", "k1_A3_abs", obj=f"o{k}", miss="guard" if k == 0 else None),
                 _rec("default", "k30_A10_abs", obj=f"o{k}", first=0.0, rec="decline"),
                 _rec("oracle", "k30_A10_abs", obj=f"o{k}", first=0.5 * (k % 2), rec="decline")]
    s = ex.summarize_gap(ex.gap_pairs(recs), n_boot=200)
    assert s["ranking"] == ["k1_A3_abs", "k30_A10_abs"]
    x = s["settings"]["k1_A3_abs"]
    assert x["first"]["all"]["mean"] == pytest.approx(1.5) and x["first"]["probe"]["n"] == 6 and x["first"]["decline"]["n"] == 0
    assert x["total"]["all"]["mean"] == pytest.approx(2.5) and x["later"]["all"]["mean"] == pytest.approx(1.0)
    assert x["power"]["n_half"] >= 1 and x["oracle_miss"] == {"guard": 1}
    assert s["settings"]["k30_A10_abs"]["power"]["n_half"] is None


def test_the_pilot_contrasts_pair_each_arm_with_the_default_rule_per_setting_and_pooled():
    recs = []
    for k, setting in enumerate(["hi_abs", "hi_abs", "lo_rec", "lo_rec"]):
        recs += [_rec("default_rule", setting, obj=f"o{k}", first=1.0, total=2.0),
                 _rec("menu_planner", setting, obj=f"o{k}", first=0.0 if setting == "hi_abs" else 1.0, total=1.0),
                 _rec("oracle_rule", setting, obj=f"o{k}", artifact=k == 3)]
    s = ex.summarize_pilot(recs, n_boot=200)
    m = s["contrasts"]["menu_planner"]
    assert m["first"]["hi_abs"]["mean"] == pytest.approx(-1.0) and m["first"]["lo_rec"]["mean"] == pytest.approx(0.0)
    assert m["first"]["all"]["mean"] == pytest.approx(-0.5) and m["first"]["all"]["n"] == 4
    assert m["total"]["all"]["mean"] == pytest.approx(-1.0)
    assert s["contrasts"]["oracle_rule"]["first"]["all"]["n"] == 3 and s["contrasts"]["oracle_rule"]["dropped"] == 1
    assert s["arms"]["default_rule"]["hi_abs"]["first"] == pytest.approx(1.0)


def test_a_planner_run_that_ended_in_an_error_lists_only_the_points_with_a_result():
    dec = {"recommendation": "commit", "recommended_option": "commit_std", "best_option": "commit_std", "guard_flag": False}
    run = {"ended": "error", "first_menu": {"without_probe": "commit"},
           "decisions": [dict(dec, choice="probe_weigh"), dict(dec, choice="commit_std")],
           "probes": [{"probe": "weigh", "option": "probe_weigh", "kind": "measured", "thrown": False, "reading": 0.3}], "attempts": []}
    assert [p["choice"] for p in ex.expectations(run, "menu_planner")] == ["probe_weigh"]


def test_the_record_checks_the_attempted_rows_action_probabilities_and_artifact_flags():
    action = {"force": 10.0, "destination": "bin"}
    run = {"ended": "placed", "recommendation": "commit", "best_option": "light_grip", "guard": {"flag": False},
           "first_menu": {"best_cost": 1.2, "recommended_cost": 1.2},
           "probes": [], "attempts": [{"option": "light_grip", "recommendation": "commit", "action": action,
                                       "declared": {"p": {"placed_stable": 0.9}, "expected_cost": 1.2}, "outcome": "placed_stable",
                                       "artifact": False}]}
    pt = ex.expectations(run, "default")[0]
    assert pt["menu"]["action:light_grip"] == action and pt["menu"]["p:light_grip"] == {"placed_stable": 0.9}
    assert pt["outcome"] == {"outcome": "placed_stable", "artifact": False}
    menu = _menu({"light_grip": 1.2})
    menu["options"][0].update(action=dict(action, force=20.0), p={"placed_stable": 0.9})
    lab = FakeLab([menu], [{"outcome": "placed_stable", "artifact": False}])
    pts = [{"choice": "light_grip", "menu": {"action:light_grip": action}, "outcome": {"outcome": "placed_stable"}}]
    assert ex.replay(lab, pts)["mismatch"] == ['point 0 menu action:light_grip: recorded {"destination": "bin", "force": 10.0}, '
                                              'replayed {"destination": "bin", "force": 20.0}']
    lab = FakeLab([_menu({"light_grip": 1.2})], [{"outcome": "placed_stable", "artifact": True}])
    pts = [{"choice": "light_grip", "menu": {}, "outcome": {"outcome": "placed_stable", "artifact": False}}]
    assert ex.replay(lab, pts)["mismatch"] == ["point 0 outcome artifact: recorded false, replayed true"]
    probe = {"probe": "weigh", "option": "probe_weigh", "kind": "measured", "thrown": False, "reading": 0.3, "artifact": False,
             "declared": {"evoi": 1.5}, "after_attempts": 0}
    assert ex.expectations({"ended": "abstained", "probes": [probe], "attempts": []}, "default")[0]["outcome"] == {
        "probed": "weigh", "kind": "measured", "thrown": False, "reading": 0.3, "artifact": False, "declared": {"evoi": 1.5}}


def test_a_run_whose_reset_or_replay_differs_gets_no_later_score_and_a_bad_true_first_menu_no_first_score():
    first = _menu({"probe_weigh": 4.0, "firm_grip": 6.0})
    row = {"object": "mug", "idx": 0, "seed": 13, "ended": "placed", "recommendation": "probe",
           "probes": [{"probe": "weigh", "after_attempts": 0}], "attempts": [{"option": "firm_grip"}]}
    rep = {"first": first, "menus": [first, _menu({"firm_grip": 5.0})], "true": [None, {"s": _menu({"firm_grip": 5.0})}], "mismatch": []}
    ok = ex.score_run(row, "default", "s", first, rep, "s")
    assert ok["regret_first"] == 0.0 and ok["regret_total"] == 0.0 and ok["first_ok"]
    moved = ex.score_run(row, "default", "s", first, rep, "s", start=["reset truth: recorded 1, replayed 2"])
    assert moved["regret_first"] == 0.0 and moved["regret_later"] is None and moved["regret_total"] is None
    bad = ex.score_run(row, "default", "s", first, rep, "s", m0_ok=False)
    assert bad["regret_first"] is None and bad["regret_total"] is None and not bad["m0_ok"]


def test_the_regret_without_guard_points_leaves_out_the_points_whose_true_menu_flags():
    first = _menu({"probe_weigh": 4.0, "firm_grip": 6.0})
    flagged = _menu({"firm_grip": 2.0}, guard=True, rec="decline")
    s = ex.arm_score(["probe_weigh", "decline"], first, [flagged])
    assert s["first"]["guard"] is False and s["later"][0]["guard"] is True
    assert s["regret_total"] == pytest.approx(8.0) and s["regret_noguard"] == 0.0 and s["guard_points"] == 1
    assert ex.arm_score(["probe_weigh", "decline"], first, None)["regret_noguard"] is None


def test_the_summaries_report_the_regret_without_guard_points():
    recs = ([_rec("default", "k1_A3_abs", obj=f"o{k}", first=1.0, total=3.0, noguard=1.0) for k in range(4)]
            + [_rec("oracle", "k1_A3_abs", obj=f"o{k}") for k in range(4)])
    x = ex.summarize_gap(ex.gap_pairs(recs), n_boot=100)["settings"]["k1_A3_abs"]
    assert x["noguard"]["all"]["mean"] == pytest.approx(1.0) and x["total"]["all"]["mean"] == pytest.approx(3.0)
    recs = ([_rec("default_rule", "lo_rec", obj=f"o{k}", total=3.0, noguard=1.0) for k in range(4)]
            + [_rec("menu_planner", "lo_rec", obj=f"o{k}", total=0.0) for k in range(4)])
    c = ex.summarize_pilot(recs, n_boot=100)["contrasts"]["menu_planner"]
    assert c["noguard"]["all"]["mean"] == pytest.approx(-1.0) and c["total"]["all"]["mean"] == pytest.approx(-3.0)


def test_the_rescore_takes_the_bundle_and_generator_of_its_source_runs(tmp_path, monkeypatch):
    """e1-v11: the rescore rebuilds the menus with the bundle its source runs used (their run configs' crossfit), under
    their generator (no field: v10); sources that name two bundles, or a generator other than the process's, are an
    error."""
    import json

    def source(name, **cfg):
        d = tmp_path / name
        d.mkdir()
        for i, c in enumerate(cfg.get("configs", [{}])):
            (d / f"run_config_{i}.json").write_text(json.dumps(c))
        return str(d)

    v11 = source("v11", configs=[{"crossfit": "data/m2b2_v11/folds", "generator": "v11"}] * 2)
    v10 = source("v10", configs=[{"crossfit": "data/m2b2/folds"}])
    two = source("two", configs=[{"crossfit": "data/m2b2/folds"}, {"crossfit": "data/other/folds"}])
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    assert ex.source_bundle([v11]) == ("data/m2b2_v11/folds", "v11")
    with pytest.raises(ValueError, match="generator v10"):
        ex.source_bundle([v10])
    with pytest.raises(ValueError, match="2 "):
        ex.source_bundle([v10, v11])
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v10")
    assert ex.source_bundle([v10]) == ("data/m2b2/folds", "v10")
    with pytest.raises(ValueError, match="2 "):
        ex.source_bundle([two])
    with pytest.raises(ValueError, match="no run config"):
        ex.source_bundle([source("empty", configs=[])])


def test_the_rescore_compares_the_bundles_as_one_path(tmp_path, monkeypatch):
    """Review fix: one folder spelled relative and absolute (or through a link) is one bundle (e1.bundle_path), so a
    resume that spells its --crossfit another way is not refused."""
    import json

    (tmp_path / "folds").mkdir()
    src = tmp_path / "src"
    src.mkdir()
    (src / "run_config_0.json").write_text(json.dumps({"crossfit": str(tmp_path / "folds"), "generator": "v11"}))
    (src / "run_config_1.json").write_text(json.dumps({"crossfit": "folds", "generator": "v11"}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    crossfit, generator = ex.source_bundle([str(src)])
    assert generator == "v11" and ex.e1.bundle_path(crossfit) == str((tmp_path / "folds").resolve())


def test_the_rescore_summary_records_the_generator_and_the_bundle(tmp_path, monkeypatch):
    """Review fix: summary.json and the wandb config name the rescore's generator and bundle (its run configs)."""
    import json

    from robots.mujoco_lab import tracking

    d = tmp_path / "rescore"
    d.mkdir()
    (d / "records.jsonl").write_text("")
    (d / "source.json").write_text(json.dumps({"kind": "gap", "src": "s"}))
    (d / "run_config_20261103T170000.json").write_text(json.dumps({"crossfit": str(tmp_path / "folds"), "generator": "v11"}))
    monkeypatch.setattr(ex, "gap_pairs", lambda recs: [])
    monkeypatch.setattr(ex, "summarize_gap", lambda prs, n_boot: {"settings": {}, "ranking": []})
    monkeypatch.setattr(ex, "checks", lambda recs: {})
    monkeypatch.setattr(ex, "markdown_gap", lambda summ: "")
    seen = {}
    monkeypatch.setattr(tracking, "log_run", lambda name, config, metrics: seen.update(config) or "url")
    assert ex.main(["summary", "--dir", str(d), "--wandb"]) == 0
    summ = json.load(open(d / "summary.json"))
    assert summ["generator"] == "v11" and summ["bundle"] == ex.e1.bundle_path(str(tmp_path / "folds"))
    assert seen["generator"] == "v11" and seen["bundle"] == summ["bundle"]
