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


"""The P(success) spread check: the weighted spread of per-particle chances, the properties set to the truth, the
row's chances through the menu's model, the fixed tags and the paired object bootstrap."""

import numpy as np
import pytest

from rpent.skills.pick_place.protocol import IRREVERSIBLE, OUTCOMES


def test_spread_gives_the_weighted_mean_sd_and_the_belief_quantiles():
    from robots.mujoco_lab.psuccess_spread import spread

    s = spread(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
    assert (s["mean"], s["sd"]) == (pytest.approx(0.5), pytest.approx(0.5))
    assert (s["q10"], s["q90"], s["range"]) == (0.0, 1.0, 1.0)
    s = spread(np.array([0.2, 0.4, 0.9]), np.array([2.0, 1.0, 1.0]))          # weights need not sum to 1
    assert s["mean"] == pytest.approx(0.425)
    assert s["sd"] == pytest.approx(np.sqrt((2 * 0.225 ** 2 + 0.025 ** 2 + 0.475 ** 2) / 4))
    assert (s["q10"], s["q90"]) == (0.2, 0.9)      # ParticleBelief.quantile: the first value whose weight sum reaches q


def test_spread_of_equal_chances_is_zero():
    from robots.mujoco_lab.psuccess_spread import spread

    s = spread(np.full(4, 0.3), np.array([0.1, 0.2, 0.3, 0.4]))
    assert (s["mean"], s["sd"], s["range"]) == (pytest.approx(0.3), pytest.approx(0.0, abs=1e-15), 0.0)


def test_with_truth_sets_only_the_named_properties_and_keeps_the_input():
    from robots.mujoco_lab.psuccess_spread import with_truth

    a = {"mass": np.array([1.0, 2.0]), "mu": np.array([0.3, 0.6]), "com_obj": np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
         "fill": np.array([0.0, 0.5])}
    truth = {"mass": 1.5, "mu": 0.4, "com_obj": [0.02, 0.01, 0.0], "fill": 0.2}
    b = with_truth(a, truth, ("mass", "com_obj"))
    assert b["mass"].tolist() == [1.5, 1.5] and b["com_obj"].tolist() == [[0.02, 0.01, 0.0]] * 2
    assert b["mu"] is a["mu"] and b["fill"] is a["fill"]
    assert a["mass"].tolist() == [1.0, 2.0] and a["com_obj"][1].tolist() == [0.1, 0.0, 0.0]


def test_with_truth_sets_a_key_the_particles_lack_and_reads_none_as_nan():
    from robots.mujoco_lab.psuccess_spread import with_truth

    a = {"mass": np.array([1.0, 2.0]), "f_damage_n": np.array([30.0, 40.0])}
    b = with_truth(a, {"f_damage_n": None, "fill": 0.0}, ("f_damage_n", "fill"))
    assert np.isnan(b["f_damage_n"]).all() and b["fill"].tolist() == [0.0, 0.0]


def test_truth_point_takes_the_generator_truth_and_the_rows_hidden_draws(monkeypatch):
    from robots.mujoco_lab import psuccess_spread as ps

    monkeypatch.setattr(ps, "generator_truth", lambda row, classes: {"mass": 0.8, "mu": 0.5, "com": 0.4, "com_obj": [0.0, 0.01, 0.0]})
    row = {"fill": 0.3, "spill_angle_deg": 35.0,
           "theta_frag": {"k_n_per_mm": None, "f_damage_n": 25.0, "v_bruise_ms": 0.6, "v_break_ms": None, "z": [0.1, 0.2, 0.3]}}
    t = ps.truth_point(row, None)
    assert (t["mass"], t["mu"], t["com_obj"], t["fill"], t["spill_angle_deg"]) == (0.8, 0.5, [0.0, 0.01, 0.0], 0.3, 35.0)
    assert (t["f_damage_n"], t["v_bruise_ms"], t["k_n_per_mm"], t["v_break_ms"]) == (25.0, 0.6, None, None)
    t = ps.truth_point({"fill": 0.0, "spill_angle_deg": None, "theta_frag": None}, None)     # class none: no fragility draw
    assert all(t[k] is None for k in ("f_damage_n", "v_bruise_ms", "v_break_ms", "k_n_per_mm")) and t["fill"] == 0.0


def test_every_truth_set_names_the_model_inputs():
    from robots.mujoco_lab.psuccess_spread import TRUTH_SETS

    assert TRUTH_SETS["mass_true"] == ("mass",) and TRUTH_SETS["com_true"] == ("com_obj",)
    assert TRUTH_SETS["both_true"] == ("mass", "com_obj")
    assert set(TRUTH_SETS["all_true"]) == {"mass", "com_obj", "mu", "fill", "spill_angle_deg", "f_damage_n", "v_bruise_ms",
                                            "v_break_ms", "k_n_per_mm"}


class _Model:
    """predict_matrix's outcome columns from a fixed matrix; it records the call."""
    kind = "classifier"
    feature_names = ("mass",)

    def __init__(self, P):
        self.P, self.calls = P, []

    def predict_matrix(self, X, planned=None):
        self.calls.append((X, planned))
        return self.P, None, None


def test_row_chances_read_placed_stable_and_sum_the_irreversible_outcomes(monkeypatch):
    from robots.mujoco_lab import psuccess_spread as ps

    P = np.zeros((2, len(OUTCOMES)))
    P[0, OUTCOMES.index("placed_stable")], P[0, OUTCOMES.index(IRREVERSIBLE[0])], P[0, OUTCOMES.index("tipped")] = 0.7, 0.2, 0.1
    P[1, OUTCOMES.index("placed_stable")], P[1, OUTCOMES.index(IRREVERSIBLE[-1])], P[1, OUTCOMES.index("missed")] = 0.1, 0.5, 0.4
    seen = {}
    monkeypatch.setattr(ps, "particle_matrix", lambda base, arrays, pose, force, names: seen.setdefault("args", (base, pose, force, names)) and np.zeros((2, 1)))
    m = _Model(P)
    placed, irr = ps.row_chances(m, {"mass": np.ones(2)}, {"b": 1}, {"p": 2}, 12.0, [{"t": 0}])
    assert placed.tolist() == [0.7, 0.1] and irr.tolist() == pytest.approx([0.2, 0.5])
    assert seen["args"] == ({"b": 1}, {"p": 2}, 12.0, ("mass",)) and m.calls[0][1] is None    # a classifier takes no plan


def test_tags_follow_the_fixed_thresholds():
    from robots.mujoco_lab.psuccess_spread import tags

    base = {"sd_prior": 0.20, "sd_weigh": 0.10, "sd_com_true": 0.06, "sd_both_true": 0.04, "p_all_true": 0.5}
    t = tags(base)
    assert t == {"weigh_resolves": True, "com_limited": False, "unavoidable": False, "bad_luck": True,
                 "bad_luck_mc_small": True, "prior_sd_ok": True}
    t = tags(dict(base, sd_weigh=0.15))                      # the SD stays; the CoM at truth leaves 0.06 of 0.15: most removed
    assert t["weigh_resolves"] is False and t["com_limited"] is True
    assert tags(dict(base, sd_weigh=0.15, sd_com_true=0.075))["com_limited"] is False    # half removed is not most
    assert tags(dict(base, p_all_true=0.19))["unavoidable"] is True and tags(dict(base, p_all_true=0.2))["unavoidable"] is False
    assert tags(dict(base, p_all_true=0.71))["bad_luck"] is False and tags(dict(base, p_all_true=0.3))["bad_luck"] is True
    assert tags(dict(base, sd_both_true=0.051))["bad_luck_mc_small"] is False
    assert tags(dict(base, sd_prior=0.019))["prior_sd_ok"] is False
    flat = tags(dict(base, sd_prior=0.0, sd_weigh=0.0, sd_com_true=0.0))
    assert flat["weigh_resolves"] is False and flat["com_limited"] is False     # no spread: nothing to resolve or to limit


def test_paired_gives_the_mean_change_and_an_object_bootstrap_interval():
    from robots.mujoco_lab.psuccess_spread import paired

    before, after = [0.4, 0.5, 0.2, 0.3], [0.1, 0.3, 0.2, 0.1]
    c = paired(before, after, ["a", "a", "b", "c"], n_boot=500, seed=0)
    assert c["n"] == 4 and c["n_objects"] == 3
    assert c["before"] == pytest.approx(0.35) and c["after"] == pytest.approx(0.175)
    assert c["change"] == pytest.approx(-0.175)
    assert c["ci"][0] <= c["change"] <= c["ci"][1] and c["ci"][1] <= 0.0
    same = paired(before, before, ["a", "a", "b", "c"], n_boot=200, seed=1)
    assert same["change"] == 0.0 and same["ci"] == [0.0, 0.0]


def test_paired_is_reproducible_for_a_seed():
    from robots.mujoco_lab.psuccess_spread import paired

    x, y, o = [0.1, 0.4, 0.3, 0.8, 0.2], [0.2, 0.1, 0.3, 0.5, 0.0], ["a", "b", "c", "d", "e"]
    assert paired(x, y, o, n_boot=300, seed=4) == paired(x, y, o, n_boot=300, seed=4)


def test_replication_matches_records_that_share_a_key_in_file_order(tmp_path):
    """The corpus and fixtures files reuse keys for different rollouts, so a key's records pair by their order."""
    import json

    from robots.mujoco_lab.psuccess_spread import replication

    def post(mass):
        return {"mass": {"mean": mass}, "mu": {"mean": 0.5}, "com": {"mean": 0.4}}

    def mine(mass, label):
        return {"prior": "class", "stage": "attempt", "key": "k", "label": label,
                "means": {"attempt": {"mass": mass, "mu": 0.5, "com": 0.4}}}

    ref = [{"stage": "attempt", "key": "k", "label": "missed", "post": post(1.3)},
           {"stage": "attempt", "key": "k", "label": "placed_stable", "post": post(0.4)}]
    (tmp_path / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ref))
    out = replication([mine(1.3, "missed"), mine(0.4, "placed_stable"), mine(0.9, "tipped")], {"class": str(tmp_path)})
    assert out["class"] == {"compared": 2, "missing": 1, "label_mismatch": 0, "maxdiff": 0.0}


def test_replication_compares_labels_only_where_the_record_has_one(tmp_path):
    """A weigh record carries no label, and estimation_check's weigh label is the probe's outcome."""
    import json

    from robots.mujoco_lab.psuccess_spread import replication

    ref = {"stage": "weigh", "key": "w", "label": "measured",
           "post": {"mass": {"mean": 0.2}, "mu": {"mean": 0.5}, "com": {"mean": 0.4}}}
    (tmp_path / "records.jsonl").write_text(json.dumps(ref) + "\n")
    mine = {"prior": "class", "stage": "weigh", "key": "w", "means": {"weigh": {"mass": 0.2, "mu": 0.5, "com": 0.4}}}
    assert replication([mine], {"class": str(tmp_path)})["class"]["label_mismatch"] == 0


class _Counting:
    """A model that counts its predict_matrix calls and returns the row sums as P."""

    def __init__(self, kind="classifier"):
        self.kind, self.feature_names, self.calls = kind, ("a", "b"), 0

    def predict_matrix(self, X, planned=None):
        self.calls += 1
        X = np.asarray(X, dtype=float)
        return np.stack([X.sum(axis=1), X[:, 0]], axis=1), None, planned


def test_memo_predicts_each_feature_matrix_once_and_returns_a_copy():
    from robots.mujoco_lab.psuccess_spread import _Memo

    model = _Counting()
    memo = _Memo(model)
    X = np.array([[1.0, 2.0], [3.0, np.nan]])
    P1 = memo.predict_matrix(X.copy())[0]
    P1[:] = -1.0                                   # a caller that writes into its result must not change the cache
    P2 = memo.predict_matrix(X.copy())[0]
    assert model.calls == 1
    np.testing.assert_array_equal(P2, model.predict_matrix(X)[0])
    memo.predict_matrix(X + 1.0)
    memo.predict_matrix(X[:1])
    assert model.calls == 2 + 2
    assert memo.feature_names == ("a", "b") and memo.kind == "classifier"


def test_memo_passes_a_hazard_models_plan_through_uncached():
    from robots.mujoco_lab.psuccess_spread import _Memo

    model = _Counting("hazard")
    memo = _Memo(model)
    X = np.ones((2, 2))
    assert memo.predict_matrix(X, ["plan"])[2] == ["plan"]
    memo.predict_matrix(X, ["plan"])
    assert model.calls == 2
