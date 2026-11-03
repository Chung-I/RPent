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

"""The E1 oracle-gap check (robots.mujoco_lab.e1_gap): the setting grid, the rules' configurations, the cost parts, the
pairs, the power line, the summary, the run checks and the case draw."""

from __future__ import annotations

import json
import math

import pytest

from robots.mujoco_lab import e1, e1_gap

Z2 = (1.959964 + 0.841621) ** 2


def test_the_grid_has_every_care_task_value_and_recovery_but_not_the_default_point():
    g = e1_gap.grid()
    assert len(g) == 24
    assert g["k30_A10_abs"] == {"care_scale": 30.0, "abstain_cost": 10.0, "recovery": "absorbing"}
    assert g["k1_A3_rec20"] == {"care_scale": 1.0, "abstain_cost": 3.0, "recovery": "rec20"}
    assert "k10_A10_abs" not in g
    assert len(e1_gap.grid(kappas=(1, 30, 100))) == 18


def test_the_default_rule_ignores_the_setting_but_the_lab_restores_under_its_recovery():
    s = {"care_scale": 100.0, "abstain_cost": 30.0, "recovery": "rec20"}
    assert e1_gap.rule_config("default", s) == {"care": 10.0, "abstain_cost": 10.0, "menu_recovery": (None, "time"),
                                                "lab_recovery": (20.0, "time")}
    assert e1_gap.rule_config("oracle", s) == {"care": 100.0, "abstain_cost": 30.0, "menu_recovery": (20.0, "time"),
                                               "lab_recovery": (20.0, "time")}
    with pytest.raises(ValueError):
        e1_gap.rule_config("menu_planner", s)


def test_the_cost_parts_add_up_to_the_realized_cost():
    s = {"care_scale": 30.0, "abstain_cost": 30.0, "recovery": "absorbing"}
    failed = {"ended": "failed", "probes": [{"probe": "weigh", "kind": "measured", "thrown": False}],
              "attempts": [{"outcome": "spilled", "cost_s": 9.0}]}
    p = e1_gap.cost_parts(failed, s)
    assert p == pytest.approx({"probes": 0.2, "attempts": 5 * 30 + 0.45, "A": 30.0, "total": 0.2 + 150.45 + 30.0})
    assert p["total"] == pytest.approx(e1.realized_cost(failed, s))
    placed = {"ended": "placed", "probes": [],
              "attempts": [{"outcome": "dropped_high", "cost_s": 8.0}, {"outcome": "placed_stable", "cost_s": 8.0}]}
    rec = dict(s, recovery="rec20")
    assert e1_gap.cost_parts(placed, rec) == pytest.approx({"probes": 0.0, "attempts": 1.4 + 0.4, "A": 0.0, "total": 1.8})


def _run(obj, role, recovery="absorbing", setting=None, ended="placed", attempts=(), rec="commit", artifact=False, idx=0):
    return {"object": obj, "idx": idx, "seed": 13, "role": role, "recovery": recovery, "setting": setting, "ended": ended,
            "attempts": [{"outcome": o, "cost_s": 8.0} for o in attempts], "probes": [], "restores": [],
            "recommendation": rec, "artifact": artifact}


def test_a_pair_rescores_the_default_run_of_the_settings_recovery():
    settings = {"k1_A30_abs": {"care_scale": 1.0, "abstain_cost": 30.0, "recovery": "absorbing"},
                "k1_A30_rec20": {"care_scale": 1.0, "abstain_cost": 30.0, "recovery": "rec20"}}
    rows = [_run("a", "default", "absorbing", ended="abstained", rec="decline"),
            _run("a", "default", "rec20", ended="failed", attempts=["dropped_high"] * 3, rec="decline"),
            _run("a", "oracle", "absorbing", "k1_A30_abs", attempts=["placed_stable"]),
            _run("a", "oracle", "rec20", "k1_A30_rec20", attempts=["placed_stable"])]
    p = {x["setting"]: x for x in e1_gap.pairs(rows, settings)}
    assert p["k1_A30_abs"]["rec"] == "decline" and p["k1_A30_abs"]["kept"]
    assert p["k1_A30_abs"]["default"]["total"] == pytest.approx(30.0)
    assert p["k1_A30_abs"]["oracle"]["total"] == pytest.approx(0.4)
    assert p["k1_A30_rec20"]["default"]["total"] == pytest.approx(3 * 1.4 + 30.0)
    assert p["k1_A30_rec20"]["default"]["events"] == 3 and p["k1_A30_abs"]["default"]["abstained"]


def test_a_pair_with_an_error_or_an_artifact_in_either_run_is_dropped():
    settings = {"k30_A10_abs": {"care_scale": 30.0, "abstain_cost": 10.0, "recovery": "absorbing"}}
    rows = [_run("a", "default", attempts=["placed_stable"], artifact=True),
            _run("a", "oracle", setting="k30_A10_abs", attempts=["placed_stable"]),
            _run("b", "default", attempts=["placed_stable"]),
            _run("b", "oracle", setting="k30_A10_abs", ended="error"),
            _run("c", "default", attempts=["placed_stable"]),
            _run("c", "oracle", setting="k30_A10_abs", attempts=["placed_stable"])]
    p = {x["object"]: x for x in e1_gap.pairs(rows, settings)}
    assert (p["a"]["kept"], p["a"]["drop"]) == (False, "artifact")
    assert (p["b"]["kept"], p["b"]["drop"]) == (False, "error")
    assert (p["c"]["kept"], p["c"]["drop"]) == (True, None)


def test_the_power_line_matches_the_closed_form():
    objs = [f"o{k}" for k in range(10)]
    # every case gains 2: a planner with the oracle's cost on a random half gains 1 on average, with SD 1 per case
    assert e1_gap.power_n([2.0] * 10, objs, share=0.5) == math.ceil(Z2)
    # the oracle itself (share 1), gains 1 and 3: mean 2, SD 1
    assert e1_gap.power_n([1.0, 3.0] * 5, objs, share=1.0) == math.ceil(Z2 / 4)
    # the same gains on two objects (5 cases each): the object clusters raise the variance five-fold
    assert e1_gap.power_n([1.0] * 5 + [3.0] * 5, ["a"] * 5 + ["b"] * 5, share=1.0) == math.ceil(10 * Z2 * 0.5 / 4)
    assert e1_gap.power_n([0.0, -1.0] * 5, objs) is None


def test_the_power_line_needs_at_least_one_case():
    assert e1_gap.power_n([2.0, 2.0], ["a", "b"], share=1.0) == 1


def test_a_run_refuses_a_directory_that_holds_another_cases_file(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "cases.json").write_text(json.dumps({"seed": 13, "n": 1, "cases": [["a", 0]]}))
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"seed": 13, "n": 1, "cases": [["b", 0]]}))
    with pytest.raises(ValueError, match="another cases file"):
        e1_gap.run(str(other), workers=1, out_dir=str(out))
    assert json.loads((out / "cases.json").read_text())["cases"] == [["a", 0]]


def _pair(obj, setting, rec, d_total, o_total, kept=True, drop=None):
    def arm(t):
        return {"total": t, "probes": 0.0, "attempts": t, "A": 0.0, "probed": False, "abstained": False, "events": 0,
                "ended": "placed"}
    return {"object": obj, "idx": 0, "seed": 13, "setting": setting, "rec": rec, "kept": kept, "drop": drop,
            "default": arm(d_total), "oracle": arm(o_total)}


def test_the_summary_ranks_the_settings_by_the_lower_bound_and_splits_by_the_recommendation():
    prs = []
    for k in range(12):
        rec = e1_gap.RECS[k % 3]
        prs.append(_pair(f"o{k}", "none", rec, 5.0 + k % 2, 5.0 + (k + 1) % 2))
        prs.append(_pair(f"o{k}", "big", rec, 10.0 + k % 2, 4.0))
    prs.append(_pair("o99", "big", "commit", 0.0, 100.0, kept=False, drop="artifact"))
    s = e1_gap.summarize(prs, n_boot=200)
    assert s["ranking"] == ["big", "none"]
    big = s["settings"]["big"]
    assert big["n"] == 12 and big["dropped"] == {"artifact": 1, "error": 0}
    assert big["gap"]["all"]["mean"] == pytest.approx(6.5) and big["gap"]["all"]["ci"][0] > 0
    assert [big["gap"][r]["n"] for r in e1_gap.RECS] == [4, 4, 4]
    assert big["arms"]["oracle"]["total"] == pytest.approx(4.0) and big["arms"]["default"]["total"] == pytest.approx(10.5)
    assert big["power"]["n_half"] is not None
    assert s["settings"]["none"]["gap"]["all"]["mean"] == pytest.approx(0.0)
    assert "| 1 | big |" in e1_gap.markdown(s)


def test_the_record_comparison_ignores_the_role_and_the_timing():
    a = _run("a", "default", attempts=["placed_stable"])
    assert e1_gap.same_record(a, dict(a, seconds=9.0, role="default_repeat"))
    assert not e1_gap.same_record(a, _run("a", "default", attempts=["tipped", "placed_stable"]))


def test_the_checks_count_a_repeat_that_differs():
    rows = [_run("a", "default", attempts=["placed_stable"]),
            _run("a", "default_repeat", attempts=["tipped", "placed_stable"]),
            _run("b", "default", attempts=["placed_stable"]),
            _run("b", "default_repeat", attempts=["placed_stable"]),
            _run("b", "oracle_default", attempts=["placed_stable"])]
    c = e1_gap.checks(rows)
    assert (c["repeat"]["n"], c["repeat"]["mismatch"], c["repeat"]["examples"]) == (2, 1, ["a/0"])
    assert (c["oracle_at_default"]["n"], c["oracle_at_default"]["mismatch"]) == (1, 0)


def test_the_cases_spread_over_the_cohort_and_leave_out_the_excluded_ones():
    never = {"seed": 13, "rows": [{"object": f"o{k}", "idx": i, "recommendation": e1_gap.RECS[(k + i) % 3]}
                                  for k in range(30) for i in range(11)]}
    c = e1_gap.pick_cases(never, n=20, plan_seed=1)
    assert c["seed"] == 13 and len(c["cases"]) == 20 and len({o for o, _ in c["cases"]}) == 20
    assert set(c["h3_recommendation"]) == {f"{o}/{i}" for o, i in c["cases"]}
    assert c == e1_gap.pick_cases(never, n=20, plan_seed=1)
    only = {"seed": 13, "rows": [{"object": "o0", "idx": i, "recommendation": "commit"} for i in range(3)]}
    assert e1_gap.pick_cases(only, n=5, plan_seed=1, exclude={(13, "o0", 1)})["cases"] in ([["o0", 0], ["o0", 2]],
                                                                                         [["o0", 2], ["o0", 0]])


def test_a_run_records_its_generator_and_refuses_a_directory_of_another(tmp_path, monkeypatch):
    """e1-v11: the run config records the lab's generator, and a directory that holds a run of another generator (a run
    config with no field is v10) is an error, so a resumed run never mixes generators."""
    import glob

    crossfit = tmp_path / "folds"
    crossfit.mkdir()
    (crossfit / "folds.json").write_text(json.dumps({"a": 0}))
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"seed": 13, "n": 1, "cases": [["a", 0]]}))
    monkeypatch.setattr(e1_gap, "run_case", lambda item: [])
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    out = tmp_path / "v11"
    e1_gap.run(str(cases), workers=1, out_dir=str(out), crossfit=str(crossfit))
    [held] = glob.glob(str(out / "run_config_*.json"))
    assert json.load(open(held))["generator"] == "v11"
    e1_gap.run(str(cases), workers=1, out_dir=str(out), crossfit=str(crossfit))          # a resume under v11
    old = tmp_path / "v10"
    old.mkdir()
    (old / "run_config_20261102T141816.json").write_text(json.dumps({"crossfit": str(crossfit)}))
    with pytest.raises(ValueError, match="generator v10"):
        e1_gap.run(str(cases), workers=1, out_dir=str(old), crossfit=str(crossfit))


def test_the_run_command_takes_the_bundle(tmp_path, monkeypatch):
    """e1-v11: `run --crossfit` names the bundle (the v11 check runs on data/m2b2_v11/folds); the run config records it."""
    import glob

    crossfit = tmp_path / "folds"
    crossfit.mkdir()
    (crossfit / "folds.json").write_text(json.dumps({"a": 0}))
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"seed": 13, "n": 1, "cases": [["a", 0]]}))
    monkeypatch.setattr(e1_gap, "run_case", lambda item: [])
    e1_gap.main(["run", "--cases", str(cases), "--workers", "1", "--out", str(tmp_path / "out"), "--crossfit", str(crossfit)])
    [held] = glob.glob(str(tmp_path / "out" / "run_config_*.json"))
    assert json.load(open(held))["crossfit"] == str(crossfit)
