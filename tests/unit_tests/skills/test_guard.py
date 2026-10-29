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

"""The knn novelty guard: row distances, the object score, the calibration and the conformal p-value."""

import numpy as np
import pytest

from rpent.skills.pick_place.features import FEATURES, as_vector
from rpent.skills.pick_place.guard import (ALPHA, K, KnnGuard, calibration_groups, calibration_scores, fit_guard, flag_reason,
                                           rows_matrix, weighted_median)


def _ref():
    """Four reference rows on a line, and a constant second feature."""
    return np.array([[0.0, 5.0], [1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])


def _clusters(seed=1, n_objects=40, n_rows=12):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0.0, 1.0, size=(n_objects, 3))
    X = np.vstack([rng.normal(c, 0.3, size=(n_rows, 3)) for c in centres])
    return X, np.repeat([f"o{i}" for i in range(n_objects)], n_rows), centres, rng


def test_the_study_constants():
    assert K == 10 and ALPHA == 0.05


def test_a_row_distance_is_the_mean_standardized_distance_to_the_k_nearest_reference_rows():
    g = KnnGuard.from_reference(_ref(), calibration=[1.0], k=2)
    # mean 1.5 and sd sqrt(1.25); the query 1.5 lies 0.5 from rows 1 and 2
    assert g.distances(np.array([[1.5, 5.0]]))[0] == pytest.approx(0.5 / np.sqrt(1.25))


def test_a_constant_training_feature_is_dropped():
    g = KnnGuard.from_reference(_ref(), calibration=[1.0], k=2)
    assert g.distances(np.array([[1.5, 9.0]]))[0] == pytest.approx(g.distances(np.array([[1.5, 5.0]]))[0])


def test_the_weighted_median_is_the_median_under_equal_weights():
    rng = np.random.default_rng(0)
    for n in (1, 2, 5, 8):
        v = rng.normal(size=n)
        assert weighted_median(v) == pytest.approx(np.median(v))
        assert weighted_median(v, np.full(n, 3.0)) == pytest.approx(np.median(v))
    assert weighted_median([1.0, 2.0, 10.0], [0.2, 0.2, 0.6]) == 10.0
    assert weighted_median([10.0, 1.0, 2.0], [0.6, 0.2, 0.2]) == 10.0


def test_the_p_value_counts_the_calibration_scores_at_or_above_the_score():
    g = KnnGuard.from_reference(_ref(), calibration=[1.0, 2.0, 3.0, 4.0], k=2)
    assert g.p_value(3.0) == pytest.approx(3 / 5)
    assert g.p_value(3.5) == pytest.approx(2 / 5)
    assert g.p_value(9.0) == pytest.approx(1 / 5)
    assert g.p_value(0.0) == pytest.approx(1.0)


def test_the_guard_flags_at_p_at_most_alpha():
    far = np.array([[30.0, 5.0]])
    at = KnnGuard.from_reference(_ref(), calibration=np.zeros(19), k=2)           # p = 1 / 20, exactly alpha
    assert at.check(far)["flag"] is True
    above = KnnGuard.from_reference(_ref(), calibration=np.zeros(18), k=2)        # p = 1 / 19
    rec = above.check(far)
    assert rec["flag"] is False and rec["p"] == pytest.approx(1 / 19)
    assert rec["guard"] == "knn" and rec["alpha"] == ALPHA and rec["k"] == 2 and rec["n_calibration"] == 18
    assert rec["score"] == pytest.approx(above.distances(far)[0])


def test_the_calibration_groups_are_seeded_and_hold_every_object_once():
    objs = [f"o{i}" for i in range(23)]
    g = calibration_groups(objs, n_groups=11, seed=0)
    assert sorted(g) == sorted(objs) and set(g.values()) == set(range(11))
    assert calibration_groups(list(reversed(objs)), n_groups=11, seed=0) == g
    assert calibration_groups(objs, n_groups=11, seed=1) != g


def test_each_calibration_score_uses_only_the_other_groups():
    """Three objects, each its own group, k = 1: an object's score is the median distance of its rows to the
    nearest row of the other two objects, standardized by their rows."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 1.0, size=(5, 2)) for c in (0.0, 0.5, 6.0)])
    obj = np.repeat(["a", "b", "c"], 5)
    s = calibration_scores(X, obj, k=1, n_groups=3, seed=0)
    for o in "abc":
        ref = X[obj != o]
        d = np.sqrt((((X[obj == o][:, None, :] - ref[None]) / ref.std(0)) ** 2).sum(-1)).min(1)
        assert s[o] == pytest.approx(np.median(d))
    assert s["c"] > max(s["a"], s["b"])


def test_an_object_far_from_the_training_objects_is_flagged_and_a_familiar_one_is_not():
    X, obj, centres, rng = _clusters()
    g = KnnGuard.fit(X, obj, seed=0)
    assert g.n_calibration == 40 and len(g.calibration_objects) == 40
    far = rng.normal(8.0, 0.3, size=(12, 3))
    near = rng.normal(centres[3], 0.3, size=(12, 3))
    assert g.check(far)["flag"] is True and g.check(far)["p"] == pytest.approx(1 / 41)
    assert g.check(near)["flag"] is False


def test_the_particle_weights_choose_which_rows_count():
    X, obj, centres, rng = _clusters()
    g = KnnGuard.fit(X, obj, seed=0)
    Q = np.vstack([rng.normal(8.0, 0.3, size=(12, 3)), rng.normal(centres[3], 0.3, size=(12, 3))])
    w = np.r_[np.full(12, 1e-6), np.ones(12)]
    assert g.check(Q, w)["flag"] is False and g.check(Q, w[::-1])["flag"] is True


def test_save_and_load_give_the_same_checks(tmp_path):
    X, obj, centres, rng = _clusters()
    g = KnnGuard.fit(X, obj, seed=0)
    g.save(tmp_path / "knn_guard.npz", features=("a", "b", "c"))
    h = KnnGuard.load(tmp_path / "knn_guard.npz", features=("a", "b", "c"))
    Q = rng.normal(0.0, 1.5, size=(30, 3))
    assert h.check(Q) == g.check(Q) and h.calibration_objects == g.calibration_objects


def test_a_guard_for_other_features_does_not_load(tmp_path):
    X, obj, _, _ = _clusters()
    g = KnnGuard.fit(X, obj, seed=0)
    g.save(tmp_path / "knn_guard.npz", features=("a", "b", "c"))
    with pytest.raises(ValueError, match="features"):
        KnnGuard.load(tmp_path / "knn_guard.npz")              # the skill's FEATURES
    with pytest.raises(ValueError, match="features"):
        g.save(tmp_path / "other.npz", features=("a", "b"))     # one name per column


def test_the_flag_reason_names_the_guard_the_score_and_p():
    r = flag_reason({"guard": "knn", "score": 4.123, "p": 0.0244, "flag": True, "alpha": 0.05, "k": 10, "n_calibration": 40})
    assert "unlike the training objects" in r and "4.12" in r and "0.024" in r and "0.05" in r


def test_the_reference_rows_are_the_features_the_models_see():
    from rpent.skills.pick_place.evaluate import true_features
    from tests.unit_tests.skills.synthetic import synthetic_rows, with_pose

    rows = with_pose(synthetic_rows(40))
    assert np.allclose(rows_matrix(rows, "true"), [as_vector(true_features(r)) for r in rows])
    assert np.allclose(rows_matrix(rows, "nominal"), [as_vector(r["features"]) for r in rows])
    assert rows_matrix(rows, "true").shape == (40, len(FEATURES))


def test_fit_guard_scores_each_training_object_once():
    from tests.unit_tests.skills.synthetic import synthetic_rows, with_pose

    rows = with_pose(synthetic_rows(300, n_objects=30))
    g = fit_guard(rows, "true", seed=0)
    assert sorted(g.calibration_objects) == sorted({r["object"] for r in rows}) and g.reference.shape[0] == 300


def test_a_particle_without_weight_does_not_move_the_score():
    """Codex review: at a cumulative weight of exactly 0.5 the median took the next value even when its weight was 0."""
    assert weighted_median([1.0, 2.0, 5.0], [0.5, 0.0, 0.5]) == 3.0
    g = KnnGuard.from_reference([[0.0], [1.0]], np.full(19, 3.0), k=1)
    two = g.check(np.array([[1.0], [5.0]]), [0.5, 0.5])
    three = g.check(np.array([[1.0], [2.0], [5.0]]), [0.5, 0.0, 0.5])
    assert two["score"] == pytest.approx(4.0) and two["flag"] is True and three == two


def test_a_feature_constant_at_an_inexact_value_is_dropped():
    """Codex review: np.std of a constant 0.1 column is about 1e-17, not 0, so `sd > 0` kept it and blew up the scale."""
    g = KnnGuard.from_reference(np.column_stack([np.arange(3.0), np.full(3, 0.1)]), np.ones(19), k=1)
    assert g.use.tolist() == [0] and g.distances(np.array([[1.0, 0.2]]))[0] == pytest.approx(0.0)


def test_a_reference_without_a_varying_feature_puts_every_row_at_distance_zero():
    """Codex review: a reference whose features are all constant (one row, or identical rows) left no feature, and the
    neighbour search failed. With no feature every row is at distance 0."""
    assert KnnGuard.from_reference(np.ones((3, 2)), [1.0]).distances(np.zeros((4, 2))).tolist() == [0.0] * 4
    g = KnnGuard.fit([[0.0], [1.0]], ["a", "b"])
    assert g.calibration.tolist() == [0.0, 0.0] and g.check(np.array([[7.0]]))["flag"] is False


def test_guard_stores_its_feature_names(tmp_path):
    from rpent.skills.pick_place.features import FEATURES_M2, FEATURES_M2B
    from tests.unit_tests.skills.synthetic import synthetic_rows, with_pose

    rows = with_pose(synthetic_rows(200))
    assert fit_guard(rows, "true", seed=0).feature_names == FEATURES_M2
    g = fit_guard(rows, "true", seed=0, names=FEATURES_M2B)
    assert g.feature_names == FEATURES_M2B and g.n_features == len(FEATURES_M2B)
    g.save(tmp_path / "g.npz")
    h = KnnGuard.load(tmp_path / "g.npz")
    assert h.feature_names == FEATURES_M2B
    with pytest.raises(ValueError, match="other features"):
        KnnGuard.load(tmp_path / "g.npz", features=FEATURES_M2)
