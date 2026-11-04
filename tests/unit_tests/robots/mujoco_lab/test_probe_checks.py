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


"""Checks 1 and 2 on the milestone 4 sweep: success is the last attempt placed_stable, the pairing mirrors sweep.pairs,
paired rates and accuracy scores with a joint object bootstrap."""

import numpy as np
import pytest

from rpent.skills.pick_place.protocol import IRREVERSIBLE, OUTCOMES


def _row(obj, idx, rec="commit", outcomes=(), probes=(), ended=None, **kw):
    return {"object": obj, "idx": idx, "recommendation": rec, "probes": list(probes),
            "attempts": [{"outcome": o, "option": "commit_std"} for o in outcomes], "ended": ended, **kw}


def test_success_is_the_last_attempt_placed_stable():
    from robots.mujoco_lab.probe_checks import success

    assert success(_row("o", 0, outcomes=("tipped", "placed_stable")))
    assert not success(_row("o", 0, outcomes=()))
    assert not success(_row("o", 0, outcomes=("placed_stable", "dropped_high")))


def test_row_metrics_read_success_damage_giving_up_and_a_probe():
    from robots.mujoco_lab.probe_checks import METRICS, row_metrics

    assert METRICS == ("success", "irreversible", "abstained", "probed")
    np.testing.assert_array_equal(row_metrics(_row("o", 0, outcomes=("placed_stable",), ended="placed")), [1, 0, 0, 0])
    np.testing.assert_array_equal(row_metrics(_row("o", 0, outcomes=("dropped_high",), ended="failed")), [0, 1, 0, 0])
    np.testing.assert_array_equal(row_metrics(_row("o", 0, ended="abstained", probes=({"kind": "damaged"},))), [0, 1, 1, 1])


def test_paired_rows_self_pair_a_subset_runs_cases_without_a_probe_recommendation():
    from robots.mujoco_lab.probe_checks import paired_rows

    a = {"rows": [_row("o", 0, rec="probe"), _row("o", 1)]}
    b = {"rows": [_row("o", 0, rec="commit")]}
    got = paired_rows(a, b, subset=True)
    assert [(x is a["rows"][0], y is b["rows"][0]) for x, y in got[:1]] == [(True, True)]
    assert got[1][0] is a["rows"][1] and got[1][1] is a["rows"][1]
    with pytest.raises(ValueError):
        paired_rows(a, b, subset=False)
    with pytest.raises(ValueError):
        paired_rows({"rows": [_row("o", 2, rec="probe")]}, b, subset=True)


def test_contrast_gives_paired_rate_differences_and_object_intervals():
    from robots.mujoco_lab.probe_checks import METRICS, contrast, object_sums

    pa = [(_row("a", 0, outcomes=("placed_stable",)), _row("a", 0)), (_row("b", 0, outcomes=("placed_stable",)), _row("b", 0, outcomes=("placed_stable",))),
          (_row("c", 0), _row("c", 0))]
    W = np.random.default_rng(0).integers(0, 3, size=(500, 3)).astype(float) + np.eye(3)[np.arange(500) % 3]
    c = contrast(object_sums(pa, ["a", "b", "c"]), W)
    assert c["n"] == 3 and set(c["metrics"]) == set(METRICS)
    s = c["metrics"]["success"]
    assert s["a"] == pytest.approx(2 / 3) and s["b"] == pytest.approx(1 / 3) and s["diff"] == pytest.approx(1 / 3)
    assert 0.0 <= s["diff_ci"][0] <= s["diff"] <= s["diff_ci"][1] <= 1.0
    assert c["metrics"]["irreversible"]["diff_ci"] == [0.0, 0.0]


def test_attempt_scores_give_the_brier_and_the_two_chances():
    from robots.mujoco_lab.probe_checks import attempt_scores

    p = {k: 0.0 for k in OUTCOMES}
    p.update(placed_stable=0.6, tipped=0.1, dropped_high=0.2, rolled_off=0.1)
    brier, pp, yp, pi, yi = attempt_scores({"declared": {"p": p}, "outcome": "dropped_high"})
    assert brier == pytest.approx(0.6 ** 2 + 0.1 ** 2 + 0.8 ** 2 + 0.1 ** 2)
    assert (pp, yp) == (pytest.approx(0.6), 0.0)
    assert pi == pytest.approx(sum(p[k] for k in IRREVERSIBLE)) and yi == 1.0


def test_accuracy_contrast_of_equal_arms_is_zero():
    from robots.mujoco_lab.probe_checks import accuracy_contrast, accuracy_sums

    p = {k: 0.0 for k in OUTCOMES}
    p.update(placed_stable=0.7, tipped=0.3)
    t = {"declared": {"p": p}, "outcome": "placed_stable", "option": "commit_std"}
    rows = [{"object": o, "probe_one": t, "never": t, "same": True} for o in ("a", "b", "a")]
    W = np.random.default_rng(1).integers(0, 3, size=(200, 2)).astype(float) + 1.0
    c = accuracy_contrast(accuracy_sums(rows, ["a", "b"]), W)
    assert c["n"] == 3
    for q in ("brier", "ece_placed", "ece_irreversible"):
        assert c[q]["diff"] == 0.0 and c[q]["diff_ci"] == [0.0, 0.0]
    assert c["brier"]["a"] == pytest.approx(0.3 ** 2 + 0.3 ** 2)


def test_accuracy_pairs_keep_probed_pairs_where_both_attempted_and_count_the_rest():
    from robots.mujoco_lab.probe_checks import accuracy_pairs

    probe = ({"kind": "measured"},)
    F = {"rows": [_row("o", 0, probes=probe), _row("o", 1, outcomes=("placed_stable",), probes=probe),
                  _row("o", 2, probes=probe), _row("o", 3, outcomes=("tipped",)), _row("o", 4, outcomes=("tipped",), probes=probe)]}
    N = {"rows": [_row("o", 0, outcomes=("tipped",)), _row("o", 1), _row("o", 2), _row("o", 3, outcomes=("tipped",)),
                  _row("o", 4, outcomes=("placed_stable",))]}
    rows, left = accuracy_pairs(F, N)
    assert [(r["idx"], r["same"], r["probe_one"]["outcome"], r["never"]["outcome"]) for r in rows] == [(4, True, "tipped", "placed_stable")]
    assert left == {"probe_one_did_not_attempt": 1, "never_did_not_attempt": 1, "neither_attempted": 1}
