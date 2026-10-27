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
