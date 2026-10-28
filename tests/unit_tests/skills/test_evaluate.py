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

"""Object split, belief-mode features and the scoring report."""

import numpy as np

from rpent.skills.pick_place.evaluate import belief_features, score, split_objects
from rpent.skills.pick_place.protocol import OUTCOMES, Prediction
from tests.unit_tests.skills.synthetic import synthetic_rows


def test_split_is_by_object_and_disjoint():
    rows = synthetic_rows(300)
    sp = split_objects(rows, seed=0)
    assert set(sp) == {"train", "val", "test"}
    parts = [set(sp[k]) for k in ("train", "val", "test")]
    assert not (parts[0] & parts[1]) and not (parts[1] & parts[2]) and not (parts[0] & parts[2])
    assert parts[0] | parts[1] | parts[2] == {r["object"] for r in rows}
    assert len(parts[0]) >= len(parts[1]) >= 3 and len(parts[2]) >= 3
    assert split_objects(rows, seed=0) == sp                      # deterministic


def test_belief_features_vary_mass_and_mu_only():
    row = synthetic_rows(1)[0]
    feats = belief_features(row, np.random.default_rng(0), n=16)
    assert len(feats) == 16
    masses = {f["mass"] for f in feats}
    assert len(masses) == 16 and all(0.1 <= f["mu"] <= 0.9 for f in feats)
    for f in feats:
        assert f["width"] == row["features"]["width"] and f["dest_opening_ratio"] == row["features"]["dest_opening_ratio"]
        assert abs(f["load_ratio"] - f["mass"] * 9.81 / (2 * f["mu"] * row["force"])) < 1e-9


def test_score_keys_and_constant_baseline():
    rows = synthetic_rows(200)
    train = synthetic_rows(500, seed=1)
    freq = {k: sum(r["label"] == k for r in train) / len(train) for k in OUTCOMES}
    preds = [Prediction(p=dict(freq)) for _ in rows]
    s = score(preds, rows, train)
    assert set(s) >= {"brier", "brier_const", "ece", "by_subgroup", "n_per_class", "unsupported_classes", "n"}
    assert abs(s["brier"] - s["brier_const"]) < 1e-9
    assert {f"{k}_t{i}" for k in ("mass", "mu", "lever") for i in range(3)} <= set(s["by_subgroup"])
    assert "dest_table_centre" in s["by_subgroup"] and s["by_subgroup"]["vis_0.9-1.0"]["n"] == len(rows)
    assert s["by_subgroup"]["vis_0.0-0.3"] == "untested"


def test_score_ece_of_perfect_and_bad_predictions():
    rows = synthetic_rows(400)
    train = rows
    perfect = [Prediction(p={k: 1.0 if r["label"] == k else 0.0 for k in OUTCOMES}) for r in rows]
    bad = [Prediction(p={k: (0.9 if k == "placed_stable" else 0.1 / 6) for k in OUTCOMES}) for _ in rows]
    assert score(perfect, rows, train)["ece"] < 1e-9 and score(perfect, rows, train)["brier"] < 1e-9
    assert score(bad, rows, train)["ece"] > 0.3


def test_true_features_differ_from_nominal_for_a_shifted_com():
    from rpent.skills.pick_place.evaluate import true_features
    from tests.unit_tests.skills.synthetic import synthetic_pose

    row = dict(synthetic_rows(1)[0])
    row["features"] = dict(row["features"], lever=0.0)
    row.update(pose=synthetic_pose(), com_obj_true=[0.0, 0.0, 0.03])
    assert abs(true_features(row)["lever"]) < 1e-12
    row["com_obj_true"] = [0.0, 0.05, 0.03]                                  # 5 cm along the hand's ey
    f = true_features(row)
    assert abs(f["lever"] - row["features"]["lever"] - 0.05) < 1e-9
    assert f["mass"] == row["truth"]["mass"] and f["mu"] == row["truth"]["mu"]


# ---------------------------------------------------------------- cross-fitted calibration (milestone 2, task 8)
import pytest  # noqa: E402


def _P(rows):
    return np.array([[r.get(k, 0.0) for k in OUTCOMES] for r in rows])


def test_iid_floor_of_calibrated_outcomes():
    from rpent.skills.pick_place.evaluate import _ece, ece_iid_floor

    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 1.0, 1000)
    y = (rng.random(1000) < p).astype(float)                    # calibrated outcomes
    floor = ece_iid_floor(p, np.random.default_rng(1))
    assert 0.0 < floor < 0.1 and abs(floor - _ece(p, y)) < 0.02
    half = np.full(1000, 0.5)
    assert _ece(half, np.ones(1000)) == pytest.approx(0.5)
    assert ece_iid_floor(half, np.random.default_rng(2)) < 0.05     # the floor depends on the declared p only


def test_ci_coverage_of_a_known_miscalibration():
    from rpent.skills.pick_place.evaluate import ci_coverage

    rng = np.random.default_rng(0)
    objects = np.repeat([f"o{i}" for i in range(95)], 11)
    p = np.clip(rng.beta(2.0, 5.0, len(objects)), 0.01, 0.99)
    y = (rng.random(len(p)) < p).astype(float)
    cov = ci_coverage(p, objects, 0.05, n_sim=400, rng=np.random.default_rng(1), y=y)
    assert 0.85 <= cov <= 1.0


def test_ci_coverage_matches_the_between_object_variance():
    from rpent.skills.pick_place.evaluate import coverage_design

    rng = np.random.default_rng(3)
    objects = np.repeat([f"o{i}" for i in range(95)], 11)
    p = np.clip(rng.beta(2.0, 5.0, len(objects)), 0.01, 0.99)
    inv = np.repeat(np.arange(95), 11)
    flat = (rng.random(len(p)) < p).astype(float)
    u = rng.uniform(-0.9, 0.9, 95)[inv]
    clustered = (rng.random(len(p)) < p + u * np.minimum(p, 1 - p)).astype(float)
    d0 = coverage_design(p, objects, 0.05, n_sim=50, rng=np.random.default_rng(4), y=flat)
    d1 = coverage_design(p, objects, 0.05, n_sim=50, rng=np.random.default_rng(4), y=clustered)
    assert d1["object_effect"] > d0["object_effect"] + 0.2
    assert d1["population_ece"] == pytest.approx(0.05, abs=1e-6) and {"coverage", "mean_width", "n_sim"} <= set(d1)


def test_per_class_marks_insufficient_classes():
    from rpent.skills.pick_place.evaluate import per_class

    labels = ["placed_stable"] * 60 + ["tipped"] * 49 + ["not_lifted"] * 91
    P = _P([{"placed_stable": 0.3, "tipped": 0.245, "not_lifted": 0.455}] * 200)
    out = per_class(P, labels)
    assert set(out) == set(OUTCOMES)
    assert out["placed_stable"]["n_pos"] == 60 and out["placed_stable"]["ece"] == pytest.approx(0.0, abs=1e-9)
    assert out["not_lifted"]["n_pos"] == 91 and out["not_lifted"]["ece"] == pytest.approx(0.0, abs=1e-9)
    assert out["tipped"] == {"n_pos": 49, "ece": "insufficient"}
    assert out["missed"] == {"n_pos": 0, "ece": "insufficient"}


def test_ece_ci_resamples_objects():
    from rpent.skills.pick_place.evaluate import _ece, ece_ci

    rng = np.random.default_rng(5)
    objects = np.repeat([f"o{i}" for i in range(60)], 10)
    p = rng.uniform(0.1, 0.9, 600)
    y = (rng.random(600) < p).astype(float)
    lo, hi = ece_ci(p, y, objects, np.random.default_rng(6))
    assert lo <= _ece(p, y) <= hi and hi - lo > 0.005
    y_same = np.repeat((rng.random(60) < 0.5).astype(float), 10)          # every case of an object has one outcome
    lo2, hi2 = ece_ci(p, y_same, objects, np.random.default_rng(6))
    assert hi2 - lo2 > hi - lo                                          # clustered outcomes widen the interval


def test_score_takes_a_per_row_constant_baseline():
    rows = synthetic_rows(100)
    preds = [Prediction(p={k: 1.0 / len(OUTCOMES) for k in OUTCOMES}) for _ in rows]
    const = np.tile(np.eye(len(OUTCOMES))[0], (len(rows), 1))           # "always placed_stable"
    s = score(preds, rows, [], const=const)
    Y = np.array([[1.0 if r["label"] == k else 0.0 for k in OUTCOMES] for r in rows])
    assert s["brier_const"] == pytest.approx(float(np.mean(np.sum((const - Y) ** 2, axis=1))))
