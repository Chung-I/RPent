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

"""Tests for the exploratory step 7 levels check (evpi_sim_levels): the model's outcome probabilities at the truth next
to the simulator's outcomes, J's parts, the sensitivity without one object and the transitions."""

import math

import numpy as np
import pytest

from robots.mujoco_lab import evpi_props as ev
from robots.mujoco_lab import evpi_sim as es
from robots.mujoco_lab import evpi_sim_levels as lv
from rpent.skills.pick_place.costs import particle_costs
from rpent.skills.pick_place.protocol import OUTCOMES, Prediction

ROW = 'commit_std|{"force": 20.0}'
LOW = 'commit_low|{"force": 10.0}'
WEIGH = 'probe_weigh|{"force": 20.0, "probe": "weigh"}'
P_S = {"placed_stable": 0.9, "tipped": 0.05, "dropped_high": 0.05}
P_0 = {"placed_stable": 0.8, "not_lifted": 0.1, "dropped_high": 0.1}


def test_j_parts_sum_to_the_rows_cost_at_one_particle():
    p = {"placed_stable": 0.9, "tipped": 0.04, "not_lifted": 0.03, "dropped_high": 0.02, "crushed": 0.01}
    parts = lv.j_parts(p, 8.0, 30.0, 3.0)
    assert parts["total"] == pytest.approx(particle_costs([Prediction(p=p)], 8.0, 30.0, 3.0, None, "time")[0])
    assert parts["irreversible"] == pytest.approx((0.02 + 0.01) * 5.0 * 30.0)
    assert parts["recoverable"] == pytest.approx(0.04 * 1.0 + 0.03 * 0.5)
    assert parts["unfinished"] == pytest.approx(3.0 * 0.1)
    assert parts["time"] == pytest.approx(0.05 * 8.0)


@pytest.mark.parametrize("label", OUTCOMES)
def test_the_realized_parts_sum_to_the_realized_row_cost(label):
    parts = lv.realized_parts(label, 10.0, 100.0, 10.0)
    assert parts["total"] == pytest.approx(es.row_cost(label, 10.0, 100.0, 10.0))
    assert parts == pytest.approx(lv.j_parts({label: 1.0}, 10.0, 100.0, 10.0))


def test_poisson_le():
    assert lv.poisson_le(0, 1.0) == pytest.approx(math.exp(-1.0))
    assert lv.poisson_le(1, 2.0) == pytest.approx(3.0 * math.exp(-2.0))
    assert lv.poisson_le(0, 0.0) == 1.0


def test_the_sensitivity_drops_the_object_with_the_largest_absolute_gain():
    assert lv.top_object(["a", "b", "c"], [2.0, -9.5, 3.0]) == "b"
    assert lv.top_object(["b", "a"], [2.0, -2.0]) == "a"          # a tie goes to the first object in sorted order


def _rec(obj, idx, prior_action="decline", all_action=ROW):
    action = {k: prior_action for k in (ev.PRIOR,) + ev.REVEALS}
    action["all"] = all_action
    choice = {k: a.split("|")[0] for k, a in action.items()}
    J = {k: 3.0 for k in action}
    return {"object": obj, "idx": idx, "prior": "class", "setting": "k30_A3", "care": 30.0, "abstain_cost": 3.0, "choice": choice,
            "action": action, "J": J, "evpi": {S: 0.0 for S in ev.REVEALS}, "ess": {k: {"ess_final": 500.0} for k in action}}


def _kind(c):
    return "abstain" if c in ev.ABSTAINS else ("probe" if c.startswith("probe_") else "commit")


def _case_set():
    """Cell κ30 A3 / class: o0..o3 decline -> ROW (o0 tips), o4 LOW (dropped_high) -> decline, o5 LOW -> ROW (both place),
    o6 decline -> a weigh probe (a probe pair, left out), o7 unchanged (left out). The model's p at the truth: P_S for
    ROW, P_0 for LOW; each record's J is the model's J at those p, as evpi_props stores it."""
    recs = [_rec(f"o{i}", i) for i in range(4)] + [_rec("o4", 4, LOW, "decline"), _rec("o5", 5, LOW, ROW),
                                                  _rec("o6", 6, "decline", WEIGH), _rec("o7", 7, "decline", "decline")]
    plans, runs, levels = [], [], []
    labels = {("o0", ROW): "tipped", ("o4", LOW): "dropped_high"}
    for r in recs:
        beliefs = {}
        for key in (ev.PRIOR, "all"):
            a = r["action"][key]
            beliefs[key] = {"choice": r["choice"][key], "action": a, "kind": _kind(r["choice"][key]),
                            "cost_s": {ROW: 8.0, LOW: 6.0}.get(a)}
            if a in (ROW, LOW):
                p = P_S if a == ROW else P_0
                r["J"][key] = lv.j_parts(p, beliefs[key]["cost_s"], 30.0, 3.0)["total"]
                levels.append({"object": r["object"], "idx": r["idx"], "prior": "class", "setting": "k30_A3", "key": key,
                               "choice": r["choice"][key], "cost_s": beliefs[key]["cost_s"], "p": p, "J": r["J"][key],
                               "J_record": r["J"][key]})
                for k in range(es.N_SEEDS):
                    runs.append({"end": "row", "label": labels.get((r["object"], a), "placed_stable"), "artifact": False,
                                 "object": r["object"], "idx": r["idx"], "job": ["row", a], "seed_k": k})
            elif a == WEIGH:
                for k in range(es.N_SEEDS):
                    runs.append({"end": "row", "probe": {"probe": "weigh", "kind": "measured", "thrown": False}, "time_scale": 1.0,
                                 "cont": {"label": "placed_stable", "cost_s": 5.0}, "artifact": False, "object": r["object"],
                                 "idx": r["idx"], "job": ["probe", "class", "k30_A3", "all"], "seed_k": k})
        r["evpi"]["all"] = r["J"][ev.PRIOR] - r["J"]["all"]
        plans.append({"object": r["object"], "idx": r["idx"], "prior": "class", "setting": "k30_A3", "beliefs": beliefs, "mismatch": []})
    return recs, plans, runs, levels


def test_the_model_next_to_the_simulator_per_side():
    recs, plans, runs, levels = _case_set()
    t = lv.tables(recs, plans, runs, levels, n_boot=200)
    s = t["sides"]["k30_A3"]["class"]
    a_s, a_p = s["a_S"], s["a_prior"]
    assert a_s["rows"] == 5 and a_p["rows"] == 2
    assert a_s["observed"] == {"placed_stable": 4, "tipped": 1} and a_p["observed"] == {"placed_stable": 1, "dropped_high": 1}
    assert a_s["p_mean"]["dropped_high"] == pytest.approx(0.05) and a_s["expected"]["dropped_high"] == pytest.approx(0.25)
    assert a_s["poisson_le"]["dropped_high"] == pytest.approx(math.exp(-0.25))
    assert a_p["poisson_le"]["dropped_high"] == pytest.approx(lv.poisson_le(1, 0.2))
    assert a_s["poisson_le"]["irreversible_any"] == pytest.approx(math.exp(-0.25))
    assert a_s["model"] == pytest.approx(lv.j_parts(P_S, 8.0, 30.0, 3.0))
    want = {k: np.mean([lv.realized_parts(x, 8.0, 30.0, 3.0)[k] for x in ["tipped"] + ["placed_stable"] * 4])
            for k in lv.PARTS}
    assert a_s["realized"] == pytest.approx(want)
    assert a_s["p_placed_mean"] == pytest.approx(0.9) and a_s["placed_observed"] == 4


def test_the_transitions_and_the_sensitivity():
    recs, plans, runs, levels = _case_set()
    t = lv.tables(recs, plans, runs, levels, n_boot=200)
    tr = t["transitions"]["k30_A3"]["class"]
    assert tr["decline → row"]["n"] == 4 and tr["row → decline"]["n"] == 1 and tr["row → row"]["n"] == 1
    gain_row = 3.0 - es.row_cost("placed_stable", 8.0, 30.0, 3.0)
    want = np.mean([3.0 - es.row_cost("tipped", 8.0, 30.0, 3.0)] + [gain_row] * 3)
    assert tr["decline → row"]["mean"] == pytest.approx(want)
    assert tr["row → decline"]["mean"] == pytest.approx(es.row_cost("dropped_high", 6.0, 30.0, 3.0) - 3.0)
    sv = t["sensitivity"]["k30_A3"]["class"]
    assert sv["dropped_object"] == "o4" and sv["n"] == 5 and sv["n_objects"] == 5
    assert sv["mean"] == pytest.approx(np.mean([3.0 - es.row_cost("tipped", 8.0, 30.0, 3.0)] + [gain_row] * 3
                                               + [es.row_cost("placed_stable", 6.0, 30.0, 3.0) - es.row_cost("placed_stable", 8.0, 30.0, 3.0)]))
    assert sv["ci"] is not None and t["j_check"] == {"rows": 7, "max_abs_diff": 0.0}


def test_a_j_that_differs_from_the_records_stops_the_tables():
    recs, plans, runs, levels = _case_set()
    levels[0]["J"] += 1e-3
    with pytest.raises(SystemExit, match="J differs"):
        lv.tables(recs, plans, runs, levels, n_boot=50)


def test_a_rows_label_that_changes_across_seeds_stops_the_tables():
    recs, plans, runs, levels = _case_set()
    runs[0]["label"] = "tipped" if runs[0]["label"] != "tipped" else "placed_stable"
    with pytest.raises(SystemExit, match="label changes across seeds"):
        lv.tables(recs, plans, runs, levels, n_boot=50)


def test_a_missing_level_stops_the_tables():
    recs, plans, runs, levels = _case_set()
    with pytest.raises(SystemExit, match="no level"):
        lv.tables(recs, plans, runs, levels[1:], n_boot=50)
