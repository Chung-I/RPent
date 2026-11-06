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

"""The success-damage curve (curve; daily-logs designs/2026-11-06-success-damage-curve-design.md): Stage 0's points from
the m4 sweep."""

import gzip
import json
import os

import pytest

from robots.mujoco_lab import curve as cv


def _case(ended, labels=(), probe_damage=False, artifact=False, error=False):
    r = {"ended": ended, "artifact": artifact, "attempts": [{"label": x} for x in labels],
         "probes": [{"kind": "damaged" if probe_damage else "measured", "damage": "crushed" if probe_damage else None}]}
    if error:
        r["error"] = "boom"
    return r


def test_m4_point_counts_success_damage_and_give_up_over_counted_cases():
    rows = [_case("placed", ["placed_stable"]), _case("placed", ["tipped", "placed_stable"]),
            _case("failed", ["tipped", "crushed"]), _case("failed", ["not_lifted"] * 3), _case("damaged", probe_damage=True),
            _case("abstained"), _case("no_row"), _case("unreachable"),
            _case("placed", ["placed_stable"], artifact=True), _case("error", error=True)]
    p = cv.m4_point(rows)
    assert p["n"] == 8                                     # the artifact and the error case are not counted
    assert p["success"] == pytest.approx(2 / 8)
    assert p["irreversible"] == pytest.approx(2 / 8)       # an irreversible attempt label or a damaging probe
    assert p["probe_damage"] == pytest.approx(1 / 8)
    assert p["give_up"] == pytest.approx(3 / 8)            # abstained, no_row, unreachable
    assert p["second_attempt"] == pytest.approx(3 / 8)     # cases with more than one attempt


def test_stage0_reads_both_protocols_per_cell_at_one_time_scale(tmp_path):
    def write(name, rows):
        with gzip.open(os.path.join(tmp_path, name), "wt") as f:
            json.dump({"rows": rows}, f)

    write("policy_k10_A10_s1_seed12.json.gz", [_case("placed", ["placed_stable"]), _case("abstained")])
    write("commit_k10_A10_s1_seed12.json.gz", [_case("failed", ["rolled_off"]), _case("placed", ["placed_stable"])])
    write("policy_k10_A10_s0_seed12.json.gz", [_case("abstained")])              # another time scale: not read
    write("policy_k10_A10_s1_seed13.json.gz", [_case("abstained")])              # another seed: not read
    write("policy_no_probe_k10_A10_s1_seed12.json.gz", [_case("abstained")])     # another protocol: not read
    pts = cv.stage0_points(str(tmp_path))
    assert [(p["protocol"], p["care"], p["A"]) for p in pts] == [("commit", 10.0, 10.0), ("policy", 10.0, 10.0)]
    pol = pts[1]
    assert pol["r"] == pytest.approx(10.0 / 50.0)          # r = A / (5 kappa)
    assert pol["success"] == pytest.approx(0.5) and pol["give_up"] == pytest.approx(0.5)
    assert pts[0]["irreversible"] == pytest.approx(0.5)


def test_label_groups_merge_points_whose_labels_would_overlap():
    groups = cv._label_groups([(2.5, 50.0, 4.0), (2.5, 50.0, 2.0), (1.5, 47.9, 0.2), (1.6, 47.9, 0.6), (0.0, 7.0, 0.00667)])
    assert groups == [((0.0, 7.0), [0.00667]), ((1.6, 47.9), [0.2, 0.6]), ((2.5, 50.0), [2.0, 4.0])]


# ---------------------------------------------------------------- the object split

def _entries():
    objs = [f"ycb__o{i}" for i in range(7)] + [f"hope__o{i}" for i in range(6)] + [f"hot3d__o{i}" for i in range(5)]
    ent = {o: {"class": "crushable" if i % 3 == 0 else "none", "fillable": i % 4 == 1} for i, o in enumerate(objs)}
    return objs, ent


def test_split_halves_each_stratum_and_balances_the_odd_extras():
    objs, ent = _entries()
    sp = cv.make_split(objs, ent, 20261107)
    val, test = set(sp["validation"]), set(sp["test"])
    assert val | test == set(objs) and not val & test
    assert abs(len(val) - len(test)) <= 1
    for key, st in sp["strata"].items():
        members = [o for o in objs if cv.stratum(o, ent[o]) == tuple(key.split("|"))]
        assert sorted(st["validation"] + st["test"]) == sorted(members)
        assert abs(len(st["validation"]) - len(st["test"])) <= 1
    odd = [st for st in sp["strata"].values() if (len(st["validation"]) + len(st["test"])) % 2]
    extras = [len(st["validation"]) > len(st["test"]) for st in odd]
    assert extras == [i % 2 == 0 for i in range(len(odd))]        # the odd extras go to validation, test, validation, ...


def test_split_is_fixed_by_its_seed():
    objs, ent = _entries()
    assert cv.make_split(objs, ent, 20261107) == cv.make_split(list(reversed(objs)), ent, 20261107)
    assert cv.make_split(objs, ent, 20261107)["validation"] != cv.make_split(objs, ent, 1)["validation"]


def test_stratum_is_the_source_and_whether_the_object_can_be_damaged():
    assert cv.stratum("ycb__mug", {"class": "none", "fillable": True}) == ("ycb", "damageable")
    assert cv.stratum("hope__butter", {"class": "crushable", "fillable": False}) == ("hope", "damageable")
    assert cv.stratum("handal__hammer", {"class": "none", "fillable": False}) == ("handal", "rigid")


# ---------------------------------------------------------------- Stage 1: the per-particle statistics

import numpy as np  # noqa: E402

from rpent.skills.pick_place.costs import TIME_COST_PER_S, expected_cost  # noqa: E402
from rpent.skills.pick_place.protocol import IRREVERSIBLE, OUTCOMES, RECOVERABLE, Prediction  # noqa: E402


def test_weighted_quantile_is_the_lower_inverse_cdf():
    x = np.arange(10.0)
    assert cv.weighted_quantile(x, np.ones(10), 0.9) == 8.0
    assert cv.weighted_quantile(x[::-1].copy(), np.ones(10), 0.9) == 8.0
    w = np.array([0.0] * 9 + [1.0])
    assert cv.weighted_quantile(x, w, 0.9) == 9.0
    assert cv.weighted_quantile(np.array([3.0, 1.0]), np.array([0.85, 0.15]), 0.9) == 3.0


def test_cvar_is_the_mean_of_the_lowest_share_of_weight_with_the_boundary_fractional():
    x = np.arange(10.0)
    assert cv.cvar_lower(x, np.ones(10), 0.1) == pytest.approx(0.0)
    assert cv.cvar_lower(x, np.ones(10), 0.15) == pytest.approx((0.1 * 0 + 0.05 * 1) / 0.15)
    assert cv.cvar_lower(x, np.array([1.0] + [0.0] * 9), 0.1) == pytest.approx(0.0)
    assert cv.cvar_lower(np.array([5.0, 2.0]), np.array([0.5, 0.5]), 0.1) == pytest.approx(2.0)


def _P(rows):
    """An n x 11 matrix from per-particle dicts of outcome probabilities (the rest on placed_stable)."""
    out = []
    for d in rows:
        v = np.array([d.get(k, 0.0) for k in OUTCOMES])
        v[OUTCOMES.index("placed_stable")] = 1.0 - sum(v[i] for i, k in enumerate(OUTCOMES) if k != "placed_stable")
        out.append(v)
    return np.array(out)


def test_stats_from_matrix_means_tail_and_cvar():
    P = _P([{"crushed": 0.0}, {"crushed": 0.1}, {"crushed": 0.4, "tipped": 0.2}, {"not_lifted": 0.5}])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    s = cv.stats_from_matrix(P, w, 8.0, keys=((10.0, 0.5),))
    assert s["p_irr"] == pytest.approx((0.0 + 0.1 + 0.4 + 0.0) / 4)
    assert s["p_placed"] == pytest.approx((1.0 + 0.9 + 0.4 + 0.5) / 4)
    assert s["rec"] == pytest.approx((0.2 * 1.0 + 0.5 * 0.5) / 4)
    assert s["q_irr"] == pytest.approx(0.4)                        # the 0.9 quantile of (0, 0.1, 0.4, 0)
    score = [1.0 - 0 - 0.4 / 10, 0.9 - 2 * 0.1 - 0.4 / 10, 0.4 - 2 * 0.4 - (0.2 + 0.4) / 10, 0.5 - 0 - (0.25 + 0.4) / 10]
    assert s["cvar"]["10|0.5"] == pytest.approx(min(score))        # CVaR 0.1 of 4 equal particles = the worst one
    assert sum(s["p"].values()) == pytest.approx(1.0)


# ---------------------------------------------------------------- Stage 1: the rules

def _row(rid, placed, irr=0.0, rec_label="tipped", rec=None, cost_s=8.0, q_irr=None, cvar=None):
    """A row's statistics: P(placed), P(irr) on crushed, the rest on ``rec_label``."""
    rec = 1.0 - placed - irr if rec is None else rec
    p = {k: 0.0 for k in OUTCOMES}
    p.update({"placed_stable": placed, "crushed": irr, rec_label: rec})
    from rpent.skills.pick_place.costs import COST
    return {"id": rid, "action": f"{rid}|{{}}", "p": p, "p_placed": placed, "p_irr": irr, "rec": rec * COST[rec_label], "cost_s": cost_s,
            "q_irr": irr if q_irr is None else q_irr, "cvar": cvar or {}}


def test_J_is_the_menus_expected_cost():
    r = _row("commit_std", 0.6, irr=0.05, rec_label="not_lifted", cost_s=9.0)
    pred = Prediction(p=dict(r["p"]))
    for care, A in ((100.0, 10.0), (0.2, 10.0), (3.0, 3.0)):
        assert cv.J(r, care, A) == pytest.approx(expected_cost(pred, 9.0, care, A))


def test_r0_takes_the_cheapest_row_when_it_costs_at_most_A():
    rows = [_row("commit_std", 0.5, irr=0.02), _row("firm_grip", 0.8, irr=0.05)]
    assert cv.decide("R0", rows, 0.02) == "give_up"          # kappa 100: 5 kappa P(irr) = 10 and 25
    assert cv.decide("R0", rows, 10.0) == "firm_grip"        # kappa 0.2
    tie = [_row("regrasp_com", 0.5, cost_s=8.0), _row("commit_std", 0.5, cost_s=8.0)]
    assert cv.decide("R0", tie, 10.0) == "commit_std"         # a tie goes to the menu's row order


def test_r1_is_r0_when_every_continuation_costs_at_least_A():
    rows = [_row("commit_std", 0.3, irr=0.0, rec_label="tipped")]
    high = {k: 1e9 for k in RECOVERABLE}
    for r in (0.02, 0.2, 2.0):
        assert cv.decide("R1", rows, r, V=lambda care, A: high) == cv.decide("R0", rows, r)
    low = lambda care, A: {k: (0.0 if k == "tipped" else 1e9) for k in RECOVERABLE}  # noqa: E731
    rows = [_row("commit_std", 0.05, irr=0.0, rec_label="tipped")]
    assert cv.decide("R0", rows, 2.0) == "give_up"           # J = 0.95 (1 + 10) + 0.4 > 10
    assert cv.decide("R1", rows, 2.0, V=low) == "commit_std"  # J1 = 0.95 (1 + 0) + 0.4 < 10


def test_r2_and_r5_cap_the_damage_then_take_the_likeliest_success():
    rows = [_row("commit_std", 0.5, irr=0.004, q_irr=0.03), _row("firm_grip", 0.9, irr=0.02, q_irr=0.02),
            _row("light_grip", 0.6, irr=0.009, q_irr=0.009)]
    assert cv.decide("R2", rows, 0.005) == "commit_std"
    assert cv.decide("R2", rows, 0.01) == "light_grip"
    assert cv.decide("R2", rows, 0.002) == "give_up"
    assert cv.decide("R5", rows, 0.01) == "light_grip"       # commit_std's tail is 0.03
    assert cv.decide("R5", rows, 0.02) == "firm_grip"


def test_r3_prices_the_damage_tail_and_keeps_the_mean_success():
    rows = [_row("commit_std", 0.6, irr=0.01, q_irr=0.10), _row("light_grip", 0.55, irr=0.01, q_irr=0.01)]
    assert cv.decide("R0", rows, 1.0) == "commit_std"
    assert cv.decide("R3", rows, 1.0) == "light_grip"
    r = rows[0]
    assert cv.J3(r, 2.0, 10.0) == pytest.approx(cv.J(r, 2.0, 10.0) + 5 * 2.0 * (0.10 - 0.01))


def test_r4_always_grasps_the_likeliest_success():
    rows = [_row("commit_std", 0.2, irr=0.5), _row("regrasp_com", 0.2, irr=0.5)]
    assert cv.decide("R4", rows, None) == "commit_std"
    assert cv.decide("R4", [], None) == "give_up"


def test_r6_grasps_the_best_cvar_only_above_zero():
    rows = [_row("commit_std", 0.6, cvar={"10|0.5": -0.1}), _row("firm_grip", 0.7, cvar={"10|0.5": 0.2})]
    assert cv.decide("R6", rows, 0.5) == "firm_grip"
    rows[1]["cvar"]["10|0.5"] = 0.0
    assert cv.decide("R6", rows, 0.5) == "give_up"


def test_care_of_r_at_A():
    assert cv.care_of(10.0, 0.02) == pytest.approx(100.0)
    assert cv.care_of(10.0, 10.0) == pytest.approx(0.2)
    assert cv.care_of(3.0, 0.2) == pytest.approx(3.0)


def test_retry_continuation_from_the_retry_rows():
    rows = [{"attempts": [{"label": "tipped", "cost_s": 8.0}, {"label": "placed_stable", "cost_s": 10.0}]},
            {"attempts": [{"label": "tipped", "cost_s": 8.0}, {"label": "crushed", "cost_s": 8.0}]},
            {"attempts": [{"label": "not_lifted", "cost_s": 8.0}]},
            {"attempts": [{"label": "placed_stable", "cost_s": 8.0}]},
            {"attempts": [{"label": "tipped", "cost_s": 8.0}], "artifact": True}]
    c = cv.retry_continuation(rows)
    assert c["tipped"] == {"n": 2, "q": 0.5, "d": 0.5, "t": pytest.approx(TIME_COST_PER_S * (10.0 + 8.0) / 2)}
    assert c["not_lifted"] == {"n": 1, "q": 0.0, "d": 0.0, "t": 0.0}
    V = cv.continuation_values(c, care=2.0, A=10.0)
    assert V["tipped"] == pytest.approx(0.5 * 10.0 + 0.5 * 10.0 + c["tipped"]["t"])
    assert V["missed"] == float("inf")                          # no data: never below A


# ---------------------------------------------------------------- Stage 1: the run's records

from types import SimpleNamespace  # noqa: E402


def test_row_record_holds_the_rows_statistics_its_action_and_the_gap_to_the_menus_mean():
    per = [Prediction(p={"placed_stable": 0.9, "crushed": 0.1}), Prediction(p={"placed_stable": 0.5, "tipped": 0.5})]
    w = np.array([0.25, 0.75])
    menu_mean = Prediction.weighted_mean(per, w)
    o = SimpleNamespace(id="firm_grip", kind="lower_risk", action={"force": 20.0000000001, "destination": "plate"}, cost_s=8.0,
                        pred=menu_mean)
    r = cv.row_record(o, per, w)
    assert r["id"] == "firm_grip" and r["kind"] == "lower_risk"
    assert r["action"] == 'firm_grip|{"destination": "plate", "force": 20.0}'
    assert r["p_placed"] == pytest.approx(0.25 * 0.9 + 0.75 * 0.5)
    assert r["p_irr"] == pytest.approx(0.025)
    assert set(r["cvar"]) == {f"{A:g}|{x:g}" for A, x in cv.CVAR_KEYS}
    assert r["mean_gap"] == pytest.approx(0.0, abs=1e-12)
    o.pred = Prediction(p=dict(menu_mean.p, placed_stable=menu_mean.p["placed_stable"] + 0.01))
    assert cv.row_record(o, per, w)["mean_gap"] == pytest.approx(0.01)


def test_case_status():
    assert cv.case_status({"assessed": False}, []) == "unreachable"
    assert cv.case_status({"assessed": True}, []) == "no_rows"
    assert cv.case_status({}, [{"id": "commit_std"}]) == "ok"


def test_shards_split_the_sorted_objects_round_robin():
    objs = ["d", "a", "c", "b", "e"]
    parts = [cv.shard_objects(objs, k, 3) for k in range(3)]
    assert parts == [["a", "d"], ["b", "e"], ["c"]]
    with pytest.raises(ValueError):
        cv.shard_objects(objs, 3, 3)


def test_decision_keys_cover_every_rule_and_knob_and_the_r_check():
    keys = cv.decision_keys()
    assert len(keys) == 4 * 9 + 2 * 7 + 1 + 4 * 2
    assert ("R4", 10.0, None) in keys and ("R6", 3.0, 2.0) in keys and ("R2", 10.0, 0.002) in keys
    assert ("R2", 3.0, 0.2) not in keys
    assert cv.dkey("R0", 10.0, 0.02) == "R0|10|0.02" and cv.dkey("R4", 10.0, None) == "R4|10|-"


def test_decisions_of_a_record_follow_decide():
    rows = [_row("commit_std", 0.5, irr=0.02, cvar={k: 0.1 for k in ("10|0.02", "10|0.5", "3|0.2", "3|2")}),
            _row("firm_grip", 0.8, irr=0.05, cvar={k: 0.3 for k in ("10|0.02", "10|0.5", "3|0.2", "3|2")})]
    for row in rows:
        row["cvar"].update({f"10|{x:g}": 0.0 for x in cv.R_GRID if f"10|{x:g}" not in row["cvar"]})
    V = lambda care, A: {k: 1e9 for k in RECOVERABLE}  # noqa: E731
    d = cv.decisions({"rows": rows}, V)
    assert d["R0|10|0.02"] == cv.decide("R0", rows, 0.02) == "give_up"
    assert d["R4|10|-"] == "firm_grip"
    assert d["R6|3|0.2"] == "firm_grip"
    assert cv.decisions({"rows": []}, V)["R4|10|-"] == "give_up"


# ---------------------------------------------------------------- Stage 1: the model-only report

def _rec(o, rows, dec, side="validation", **kw):
    return dict({"object": o, "idx": 0, "side": side, "status": "ok" if rows else "no_rows", "rows": rows, "decisions": dec,
                 "artifact": False, "ess": 20000.0, "thin": False}, **kw)


def test_predicted_point_counts_a_give_up_as_zero_success_and_zero_damage():
    rows = [_row("commit_std", 0.6, irr=0.1), _row("firm_grip", 0.1, irr=0.0)]
    recs = [_rec("a", rows, {"K": "commit_std"}), _rec("b", rows, {"K": "firm_grip"}), _rec("c", rows, {"K": "give_up"}),
            _rec("d", [], {"K": "give_up"})]
    p = cv.predicted_point(recs, "K")
    assert p["n"] == 4
    assert p["success"] == pytest.approx((0.6 + 0.1) / 4)
    assert p["irr"] == pytest.approx(0.1 / 4)
    assert p["give_up"] == pytest.approx(0.5)
    assert p["low_success"] == pytest.approx(0.5)             # firm_grip's 0.1 among the 2 chosen rows
    assert p["placed_q"] == pytest.approx([0.1, 0.225, 0.35, 0.475, 0.6])    # min, quartiles, max of the chosen rows
    assert p["irr_q"] == pytest.approx([0.0, 0.025, 0.05, 0.075, 0.1])


def test_counted_drops_errors_artifacts_and_other_sides():
    recs = [_rec("a", [], {}), _rec("b", [], {}, side="test"), _rec("c", [], {}, artifact=True),
            dict(_rec("d", [], {}), status="error", error="x")]
    assert [r["object"] for r in cv.counted(recs, "validation")] == ["a"]


def test_at_equal_success_interpolates_the_other_curves_damage():
    a = [{"knob": 1.0, "success": 0.3, "irr": 0.02}, {"knob": 2.0, "success": 0.9, "irr": 0.2}]
    b = [{"knob": 0.1, "success": 0.2, "irr": 0.01}, {"knob": 0.2, "success": 0.4, "irr": 0.03}, {"knob": 0.5, "success": 0.5, "irr": 0.05}]
    out = cv.at_equal_success(a, b)
    assert out[0]["irr_other"] == pytest.approx(0.02)
    assert out[1]["irr_other"] is None                         # 0.9 is outside b's success range


def test_only_r_share_compares_A_3_with_A_10_at_the_same_r():
    recs = [_rec("a", [], {"R0|3|0.2": "give_up", "R0|10|0.2": "give_up"}),
            _rec("b", [], {"R0|3|0.2": "commit_std", "R0|10|0.2": "firm_grip"})]
    assert cv.only_r(recs, "R0", 0.2) == {"n": 2, "differ": 1, "share": 0.5}


def test_e1_check_compares_the_menu_and_R0_with_experiment_1s_decision_without_probes():
    rows = [_row("commit_std", 0.97, irr=0.0)]                 # J at kappa 100, A 10: 0.03 + 0.4 + 0.3 <= 10
    mine = [dict(_rec("a", rows, {"R0|10|0.02": "commit_std"}), choice="commit_std", best_option="commit_std",
                 guard_flag=False, fingerprint="f1"),
            dict(_rec("b", rows, {"R0|10|0.02": "commit_std"}), choice="decline", best_option="commit_std",
                 guard_flag=True, fingerprint="f2"),
            dict(_rec("c", [], {"R0|10|0.02": "give_up"}), choice="unreachable", best_option=None, guard_flag=False,
                 fingerprint="f3")]

    def e1(o, choice, best, without, flag, fp):
        return {"object": o, "idx": 0, "prior": "class", "setting": "k100_A10", "choice": {"prior": choice}, "fingerprint": fp,
                "menu": {"prior": {"best_option": best, "without_probe": without, "guard_flag": flag}}}

    theirs = [e1("a", "weigh_probe", "commit_std", "commit", False, "f1"), e1("b", "decline", "commit_std", "decline", True, "f2"),
              e1("c", "unreachable", None, None, False, "f3"), e1("a", "commit_std", "x", "commit", False, "f1") | {"prior": "generic"}]
    c = cv.e1_check(mine, theirs)
    assert c == {"cases": 3, "missing": 0, "fingerprint_differs": 0, "guard_differs": 0, "menu_differs": 0,
                 "no_guard": 2, "r0_vs_menu_differs": 0, "r0_vs_e1_differs": 0}


def _cv(x):
    return {f"{A:g}|{r:g}": x for A, r in cv.CVAR_KEYS}


def _s1rec(o, side="validation", thin=False, rows=None, **kw):
    rows = [_row("commit_std", 0.6, irr=0.01, cvar=_cv(0.1)), _row("firm_grip", 0.8, irr=0.05, cvar=_cv(0.2))] if rows is None else rows
    rows = [dict(r, mean_gap=0.0) for r in rows]
    return dict({"object": o, "idx": 0, "side": side, "status": "ok" if rows else "no_rows", "rows": rows, "artifact": False,
                 "ess": 20000.0, "thin": thin, "choice": "commit_std", "best_option": "commit_std", "guard_flag": False,
                 "fingerprint": "f", "n_particles": 20000}, **kw)


def test_stage1_report_reads_validation_only_and_reports_thin_plastic_and_low_ess_apart():
    recs = [_s1rec("a"), _s1rec("b", thin=True), _s1rec("e", ess=50.0), _s1rec("c", side="test"),
            _s1rec("d", rows=[], status="error", error="x")]
    V = lambda care, A: {k: 1e9 for k in RECOVERABLE}  # noqa: E731
    s0 = [{"protocol": "policy", "care": 1.5, "A": 3.0, "r": 0.4, "success": 0.45, "irreversible": 0.02},
          {"protocol": "policy", "care": 7.5, "A": 3.0, "r": 0.08, "success": 0.3, "irreversible": 0.0},
          {"protocol": "policy", "care": 10.0, "A": 10.0, "r": 0.2, "success": 0.48, "irreversible": 0.02}]
    rep = cv.stage1_report(recs, V, s0, [])
    assert rep["counts"]["validation"] == 3 and rep["counts"]["errors"] == 1 and rep["counts"]["thin_validation"] == 1
    assert rep["counts"]["ess_low_validation"] == 1
    assert rep["curves"]["R4"][0]["n"] == 3
    assert rep["curves_no_thin"]["R4"][0]["n"] == 2
    assert rep["curves_ess_ok"]["R4"][0]["n"] == 2
    assert [p["r"] for p in rep["stage0_at_check"]] == [0.2]
    assert set(rep["only_r"]) == {cv.dkey(rule, 3.0, r) for rule in cv.CHECK_RULES for _, r in cv.R_CHECK}
    assert rep["e1"]["cases"] == 4 and rep["e1"]["missing"] == 4
    assert rep["mean_gap_max"] == 0.0
    md = cv.stage1_markdown(rep)
    assert "without the 9 thin-plastic objects" in md and "| R4 |" in md


def _write_run(d, recs, **cfg):
    import json as _json
    import os as _os

    _os.makedirs(d)
    base = {"bundle": "B", "seed": 11, "split_sha256": "s", "scenes_sha256": "c", "manifest_sha256": "m", "code_revision": "x",
            "generator": "v11", "r_grid": list(cv.R_GRID), "delta_grid": list(cv.DELTA_GRID), "a_main": 10.0,
            "r_check": [list(x) for x in cv.R_CHECK], "quantile": 0.9, "cvar_level": 0.1, "stage1_care": 100.0}
    _json.dump(dict(base, **cfg), open(_os.path.join(d, "run_config.json"), "w"))
    with open(_os.path.join(d, "records.jsonl"), "w") as f:
        for r in recs:
            f.write(_json.dumps(r) + "\n")
    _json.dump({"cases": len(recs)}, open(_os.path.join(d, "run_metrics.json"), "w"))


def test_load_stage1_joins_finished_runs_that_agree(tmp_path):
    _write_run(str(tmp_path / "a"), [_s1rec("a")])
    _write_run(str(tmp_path / "b"), [_s1rec("b")])
    recs, cfg = cv.load_stage1([str(tmp_path / "a"), str(tmp_path / "b")])
    assert [r["object"] for r in recs] == ["a", "b"] and cfg["seed"] == 11
    _write_run(str(tmp_path / "c"), [_s1rec("c")], seed=12)
    with pytest.raises(SystemExit, match="seed"):
        cv.load_stage1([str(tmp_path / "a"), str(tmp_path / "c")])
    _write_run(str(tmp_path / "d"), [_s1rec("a")])
    with pytest.raises(SystemExit, match="twice"):
        cv.load_stage1([str(tmp_path / "a"), str(tmp_path / "d")])
    _write_run(str(tmp_path / "e"), [_s1rec("e")])
    (tmp_path / "e" / "run_metrics.json").unlink()
    with pytest.raises(SystemExit, match="not finished"):
        cv.load_stage1([str(tmp_path / "e")])


def test_stage1_summary_end_to_end_on_synthetic_runs(tmp_path, monkeypatch):
    import gzip as _gzip
    import json as _json

    monkeypatch.setenv("WANDB_MODE", "disabled")
    _write_run(str(tmp_path / "a"), [_s1rec("a"), _s1rec("t", side="test")])
    _write_run(str(tmp_path / "b"), [_s1rec("b", thin=True)])
    retry = tmp_path / "retry.json.gz"
    with _gzip.open(retry, "wt") as f:
        _json.dump({"rows": [{"attempts": [{"label": "tipped", "cost_s": 8.0}, {"label": "placed_stable", "cost_s": 8.0}]}]}, f)
    s0 = tmp_path / "s0.json"
    s0.write_text(_json.dumps([{"protocol": "policy", "care": 10.0, "A": 10.0, "r": 0.2, "success": 0.48, "irreversible": 0.02,
                                "give_up": 0.3}]))
    (tmp_path / "e1" / "shard_0").mkdir(parents=True)
    with _gzip.open(tmp_path / "e1" / "shard_0" / "records.jsonl.gz", "wt") as f:
        f.write(_json.dumps({"object": "a", "idx": 0, "prior": "class", "setting": "k100_A10", "choice": {"prior": "commit_std"},
                             "fingerprint": "f", "menu": {"prior": {"best_option": "commit_std", "without_probe": "commit",
                                                                     "guard_flag": False}}}) + "\n")
    out = tmp_path / "out"
    rep = cv.main(["stage1-summary", "--runs", f"{tmp_path / 'a'},{tmp_path / 'b'}", "--retry", str(retry), "--stage0", str(s0),
                   "--e1", str(tmp_path / "e1"), "--split-commit", "723125a5", "--out", str(out)])
    assert rep["counts"]["validation"] == 2 and rep["e1"]["cases"] == 3 and rep["e1"]["missing"] == 2
    assert rep["provenance"]["retry_continuation"]["tipped"]["n"] == 1
    assert (out / "stage1_summary.md").read_text().startswith("# Success-damage curve, Stage 1")
    assert (out / "stage1_curves.png").stat().st_size > 10000
    with pytest.raises(SystemExit, match="exists"):
        cv.main(["stage1-summary", "--runs", str(tmp_path / "a"), "--retry", str(retry), "--stage0", str(s0),
                 "--e1", str(tmp_path / "e1"), "--split-commit", "x", "--out", str(out)])


# ---------------------------------------------------------------- Stage 2: the episodes

def test_first_rows_are_the_distinct_grasps_of_the_rules_at_A_main_in_menu_order():
    dec = {"R0|10|0.02": "give_up", "R0|10|10": "firm_grip", "R4|10|-": "commit_std", "R2|10|0.2": "firm_grip",
           "R6|10|0.5": "light_grip", "R0|3|0.2": "slow_lift"}
    assert cv.first_rows(dec, ("R0", "R2", "R4")) == ["commit_std", "firm_grip"]
    assert cv.first_rows(dec, ("R0", "R2", "R4", "R6")) == ["commit_std", "firm_grip", "light_grip"]


def test_second_decisions_price_no_continuation_for_R1():
    rows = [dict(_row("commit_std", 0.05, irr=0.0, rec_label="tipped"), cvar=_cv(0.1))]
    d = cv.second_decisions(rows, ("R0", "R1", "R4"))
    assert set(d) == {cv.dkey(rule, 10.0, k) for rule in ("R0", "R1", "R4") for k in cv.KNOBS[rule]}
    assert d["R1|10|2"] == d["R0|10|2"] == "give_up"           # no third attempt: R1 prices a recoverable label as R0 does
    assert d["R4|10|-"] == "commit_std"


def test_menu_matches_needs_the_same_rows_actions_and_means():
    a = [{"id": "commit_std", "action": "commit_std|{}", "p": {"placed_stable": 0.5, "tipped": 0.5}}]
    assert cv.menu_matches(a, [dict(a[0], p={"placed_stable": 0.5 + 1e-15, "tipped": 0.5})]) == (True, "")
    assert cv.menu_matches(a, [dict(a[0], action="commit_std|{x}")])[0] is False
    assert cv.menu_matches(a, [dict(a[0], p={"placed_stable": 0.6, "tipped": 0.4})])[0] is False
    assert cv.menu_matches(a, a + [dict(a[0], id="firm_grip")])[0] is False


def _sim(first, label, menu2=None, second=None, seed_k=0, **kw):
    return dict({"seed_k": seed_k, "first": first, "label": label, "artifact": False, "error": None,
                 "menu2": menu2, "second": second or {}}, **kw)


def test_episode_follows_the_rule_through_two_attempts():
    dec = {"R0|10|1": "commit_std", "R0|10|2": "give_up", "R2|10|0.01": "firm_grip", "R4|10|-": "firm_grip"}
    m2 = {"status": "ok", "rows": [], "decisions": {"R0|10|1": "slow_lift", "R4|10|-": "give_up"}}
    sims = [_sim("commit_std", "tipped", menu2=m2, second={"slow_lift": {"label": "placed_stable", "artifact": False, "error": None}}),
            _sim("firm_grip", "crushed")]
    e = cv.episode(dec, sims, "R0|10|1")
    assert e == {"attempts": 2, "success": True, "irr": False, "give_up": False, "labels": ["tipped", "placed_stable"],
                 "rows": ["commit_std", "slow_lift"]}
    assert cv.episode(dec, sims, "R0|10|2") == {"attempts": 0, "success": False, "irr": False, "give_up": True, "labels": [],
                                                 "rows": []}
    assert cv.episode(dec, sims, "R2|10|0.01")["irr"] is True
    m2["decisions"]["R2|10|0.01"] = "give_up"
    e4 = cv.episode(dict(dec, **{"R4|10|-": "commit_std"}), sims, "R4|10|-")
    assert e4["give_up"] is True and e4["attempts"] == 1          # the second decision gave up


def test_case_dropped_by_any_artifact_or_error_at_the_main_seed():
    good = [_sim("commit_std", "placed_stable")]
    assert cv.case_dropped(good) is False
    assert cv.case_dropped(good + [_sim("firm_grip", "tipped", artifact=True)]) is True
    assert cv.case_dropped([_sim("commit_std", "tipped", second={"x": {"label": None, "artifact": False, "error": "boom"}})]) is True
    assert cv.case_dropped([_sim("commit_std", "tipped", menu2={"status": "error", "rows": [], "decisions": {}})]) is True
    assert cv.case_dropped(good + [_sim("firm_grip", "tipped", artifact=True, seed_k=1)]) is False


class _FakeLab:
    """simulate_case's lab: first attempts give first[row], second attempts second[row]; ``replay`` overrides the
    first attempt's label on fresh episodes after the first."""

    def __init__(self, first, second, rows2, stop_at=None, replay=None):
        self.first, self.second, self.rows2, self.stop_at, self.replay = first, second, rows2, stop_at, replay
        self.fresh_calls, self.attempts = 0, []

    def fresh(self, reset_seed, rec1):
        self.fresh_calls += 1
        if self.stop_at == self.fresh_calls:
            return None, "replay gate: rows differ", True
        return {"n": 0, "seed": reset_seed, "k": self.fresh_calls}, "", True

    def attempt(self, store, row):
        store["n"] += 1
        self.attempts.append((store["seed"], store["n"], row))
        if store["n"] == 1:
            label = self.replay if (self.replay and store["k"] > 1) else self.first[row]
        else:
            label = self.second[row]
        return {"outcome": label, "artifact": False}

    def second_menu(self, store, d):
        return {"status": "ok", "rows": [dict(r) for r in self.rows2]}


def _dec_for(rows, rules=("R0", "R4")):
    V = lambda care, A: {k: 1e9 for k in RECOVERABLE}  # noqa: E731
    return cv.decisions({"rows": rows}, V)


def test_simulate_case_runs_each_first_row_and_each_distinct_second_row():
    rows1 = [dict(_row("commit_std", 0.9, irr=0.0), cvar=_cv(0.1)), dict(_row("firm_grip", 0.95, irr=0.02), cvar=_cv(0.1))]
    rows2 = [dict(_row("commit_std", 0.9, irr=0.0), cvar=_cv(0.1)), dict(_row("slow_lift", 0.95, irr=0.01), cvar=_cv(0.1))]
    dec = _dec_for(rows1)                                      # R0 grasps commit_std at low r, firm_grip at high r; R4 firm_grip
    lab = _FakeLab({"commit_std": "tipped", "firm_grip": "placed_stable"}, {"commit_std": "placed_stable", "slow_lift": "missed"}, rows2)
    res = cv.simulate_case(lab, {"rows": rows1}, dec, ("R0", "R4"), [11000])
    assert res["stop"] is None
    sims = {s["first"]: s for s in res["sims"]}
    assert set(sims) == set(cv.first_rows(dec, ("R0", "R4")))
    want2 = cv.first_rows(cv.second_decisions([dict(r) for r in rows2], ("R0", "R4")), ("R0", "R4"))
    assert sorted(sims["commit_std"]["second"]) == sorted(want2) and len(want2) == 2
    assert sims["commit_std"]["second"]["slow_lift"]["label"] == "missed"
    assert sims["firm_grip"]["menu2"] is None and sims["firm_grip"]["second"] == {}
    # one fresh episode per first row, plus one per later second row (its first attempt replayed)
    assert lab.fresh_calls == len(sims) + len(want2) - 1
    assert cv.episode(dec, res["sims"], "R4|10|-")["rows"][0] == "firm_grip"


def test_simulate_case_stops_on_a_gate_or_a_replayed_first_attempt_that_differs():
    rows1 = [dict(_row("commit_std", 0.9, irr=0.0), cvar=_cv(0.1))]
    rows2 = [dict(_row("commit_std", 0.9, irr=0.0), cvar=_cv(0.1)), dict(_row("slow_lift", 0.95, irr=0.01), cvar=_cv(0.1))]
    dec = _dec_for(rows1)
    lab = _FakeLab({"commit_std": "tipped"}, {"commit_std": "placed_stable", "slow_lift": "missed"}, rows2, stop_at=1)
    assert "replay gate" in cv.simulate_case(lab, {"rows": rows1}, dec, ("R0", "R4"), [11000])["stop"]
    lab = _FakeLab({"commit_std": "tipped"}, {"commit_std": "placed_stable", "slow_lift": "missed"}, rows2, replay="missed")
    assert "first attempt" in cv.simulate_case(lab, {"rows": rows1}, dec, ("R0", "R4"), [11000])["stop"]


def test_simulate_case_repeats_everything_at_each_seed():
    rows1 = [dict(_row("commit_std", 0.9, irr=0.0), cvar=_cv(0.1))]
    lab = _FakeLab({"commit_std": "placed_stable"}, {}, [])
    res = cv.simulate_case(lab, {"rows": rows1}, _dec_for(rows1), ("R0", "R4"), [11000, 5, 7])
    assert [(s["seed_k"], s["reset_seed"]) for s in res["sims"]] == [(0, 11000), (1, 5), (2, 7)]


def _s1(o="ycb__a", idx=0, rows=None, **kw):
    rec = {"object": o, "idx": idx, "fold": 0, "side": "validation", "class": "rigid", "law": None, "fillable": False,
           "thin": False, "destination": "bin", "ess": 900.0, "guard_flag": False, "status": "ok" if rows else "no_rows",
           "rows": rows or [], "artifact": False, "fingerprint": "f"}
    return dict(rec, **kw)


def test_stage2_record_skips_stage1_errors_and_artifacts_and_keeps_row_stats():
    rows1 = [dict(_row("commit_std", 0.9, irr=0.01, q_irr=0.03), cvar=_cv(0.1))]
    dec = _dec_for(rows1)
    sims = [_sim("commit_std", "placed_stable", reset_seed=11000)]
    r = cv.stage2_record(_s1(rows=rows1), dec, ("R0", "R4"), {"sims": sims, "stop": None}, 1)
    assert r["skipped"] is None and r["dropped"] is False and r["first_rows"] == ["commit_std"] and r["sims"] == sims
    assert r["row_stats"] == {"commit_std": {"p_placed": 0.9, "p_irr": 0.01, "q_irr": 0.03}}
    assert r["decisions"] == dec and r["object"] == "ycb__a" and r["side"] == "validation" and r["n_seeds"] == 1
    bad = cv.stage2_record(_s1(rows=rows1, status="error", error="E"), dec, ("R0", "R4"), None, 1)
    assert bad["skipped"] == "stage1 error" and bad["dropped"] is True and bad["sims"] == [] and bad["first_rows"] == []
    art = cv.stage2_record(_s1(rows=rows1, artifact=True), dec, ("R0", "R4"), None, 1)
    assert art["skipped"] == "stage1 artifact" and art["dropped"] is True
    none = cv.stage2_record(_s1(), _dec_for([]), ("R0", "R4"), {"sims": [], "stop": None}, 1)
    assert none["skipped"] is None and none["dropped"] is False and none["first_rows"] == []
    assert cv.stage2_skip(_s1(rows=rows1)) is None and cv.stage2_skip(_s1(rows=rows1, artifact=True)) == "stage1 artifact"
    err = cv.stage2_record(_s1(rows=rows1), dec, ("R0", "R4"), {"sims": [], "stop": None, "error": "RuntimeError: E"}, 1)
    assert err["dropped"] is True and err["error"] == "RuntimeError: E" and r["error"] is None


def test_smoke_summary_counts_label_changes_across_seeds_and_seconds_per_seed():
    def sim(k, first, label, second=None, seconds=10.0):
        return _sim(first, label, menu2={"status": "ok"} if second else None, second=second or {}, seed_k=k, seconds=seconds)
    a = [sim(0, "commit_std", "tipped", {"slow_lift": {"label": "placed_stable"}}, 30.0), sim(1, "commit_std", "tipped",
         {"slow_lift": {"label": "missed"}}), sim(0, "firm_grip", "placed_stable"), sim(1, "firm_grip", "placed_stable")]
    b = [sim(0, "commit_std", "placed_stable"), sim(1, "commit_std", "tipped", {"slow_lift": {"label": "missed"}})]
    recs = [dict(_s1("ycb__a", 0), sims=a, n_seeds=2), dict(_s1("ycb__b", 1), sims=b, n_seeds=2)]
    sm = cv.smoke_summary(recs)
    assert sm["first_pairs"] == 3 and sm["first_changed"] == 1            # ycb__b commit_std: placed_stable vs tipped
    assert sm["second_triples"] == 2 and sm["second_changed"] == 1        # ycb__a slow_lift changed; ycb__b's ran at one seed
    assert sm["second_presence_changed"] == 1                             # ycb__b slow_lift ran at seed 1 only
    assert sm["seconds_per_seed"] == {"0": 50.0, "1": 30.0}
    assert sm["first_rows_seed0"] == 3 and sm["seconds_per_first_row_seed0"] == pytest.approx(50.0 / 3)
    assert sm["errors"] == 0 and sm["artifacts"] == 0


def test_projection_scales_the_seed0_seconds_per_first_row_to_the_run():
    sm = {"seconds_per_first_row_seed0": 36.0}
    assert cv.projection(sm, 1000) == pytest.approx(10.0)


def test_stage2_writes_records_and_stops_the_run_on_a_gate(tmp_path, monkeypatch):
    import gzip as _gzip
    import json as _json

    from robots.mujoco_lab import identify_fit

    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(identify_fit, "require_bundle_generator", lambda b: "v11")
    rows1 = [dict(_row("commit_std", 0.9, irr=0.0), cvar=_cv(0.1))]
    _write_run(str(tmp_path / "a"), [_s1("ycb__a", 0, rows=rows1), _s1("ycb__a", 1, rows=rows1), _s1("ycb__t", 0, rows=rows1, side="test")])
    _write_run(str(tmp_path / "b"), [_s1("ycb__b", 0, rows=rows1, artifact=True)])
    retry = tmp_path / "retry.json.gz"
    with _gzip.open(retry, "wt") as f:
        _json.dump({"rows": [{"attempts": [{"label": "tipped", "cost_s": 8.0}, {"label": "placed_stable", "cost_s": 8.0}]}]}, f)
    seen = []

    def fake(job):
        o, items, s = job
        seen.append((o, [int(r["idx"]) for r, _ in items], s["n_seeds"], s["bundle"]))
        recs = [cv.stage2_record(r, d, s["rules"], {"sims": [_sim("commit_std", "placed_stable", seconds=1.0)], "stop": None}, s["n_seeds"])
                for r, d in items]
        for r in recs:
            r["seconds"] = 1.0
        return {"records": recs, "stop": None}

    monkeypatch.setattr(cv, "stage2_object", fake)
    runs = f"{tmp_path / 'a'},{tmp_path / 'b'}"
    path = cv.main(["stage2", "--stage1", runs, "--retry", str(retry), "--out", str(tmp_path / "s2"), "--sides", "validation",
                    "--seeds", "5"])
    recs = [_json.loads(x) for x in open(path)]
    assert sorted((r["object"], r["idx"]) for r in recs) == [("ycb__a", 0), ("ycb__a", 1), ("ycb__b", 0)]
    assert {r["object"]: r["skipped"] for r in recs}["ycb__b"] == "stage1 artifact"
    assert sorted(seen) == [("ycb__a", [0, 1], 5, "B/folds/fold_0"), ("ycb__b", [0], 5, "B/folds/fold_0")]
    cfg = _json.load(open(tmp_path / "s2" / "run_config.json"))
    assert cfg["rules"] == list(cv.SIM_RULES) and cfg["n_seeds"] == 5 and set(cfg["stage1_records_sha256"]) == set(runs.split(","))
    m = _json.load(open(tmp_path / "s2" / "run_metrics.json"))
    assert m["cases"] == 3 and m["stop"] is None and m["skipped"] == 1
    with pytest.raises(SystemExit, match="exists"):
        cv.main(["stage2", "--stage1", runs, "--retry", str(retry), "--out", str(tmp_path / "s2")])
    monkeypatch.setattr(cv, "stage2_object", lambda job: dict(fake(job), stop="ycb__a 0: replay gate: rows differ"))
    with pytest.raises(SystemExit) as e:
        cv.main(["stage2", "--stage1", runs, "--retry", str(retry), "--out", str(tmp_path / "s3"), "--cases", "ycb__a:1"])
    assert e.value.code == 3
    m = _json.load(open(tmp_path / "s3" / "run_metrics.json"))
    assert m["stop"] == "ycb__a 0: replay gate: rows differ" and m["of_cases"] == 1
    sm = cv.main(["stage2-smoke", "--runs", str(tmp_path / "s2"), "--out", str(tmp_path / "sm")])
    assert sm["cases"] == 3 and sm["run_first_rows"] == 3 and sm["run_cases"] == 3
    assert sm["seconds_per_first_row_seed0"] == 1.0                      # 2 simulated cases (the artifact is skipped), 1 s each
    assert sm["projection_cpu_h"] == pytest.approx(3 * 1.0 / 3600.0)   # 3 non-skipped Stage 1 cases, one first row each
