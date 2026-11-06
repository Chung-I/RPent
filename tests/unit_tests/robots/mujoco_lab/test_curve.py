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
    return {"id": rid, "p": p, "p_placed": placed, "p_irr": irr, "rec": rec * COST[rec_label], "cost_s": cost_s,
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
