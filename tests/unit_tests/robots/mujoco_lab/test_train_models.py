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


"""Training on true-CoM features, the versioned model bundle and its manifest (needs the sim)."""

import json
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")

M1_MODEL = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",
                                         "robots", "mujoco_lab", "data", "success_model.joblib"))


def _write(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_bundle_manifest_round_trip(tmp_path):
    from robots.mujoco_lab.robot_spec import build_lab
    from robots.mujoco_lab.train_models import main
    from tests.unit_tests.skills.synthetic import synthetic_rows, with_pose

    before = os.path.getmtime(M1_MODEL) if os.path.exists(M1_MODEL) else None
    corpus = tmp_path / "c.jsonl"
    _write(with_pose(synthetic_rows(300)), corpus)
    out = main(["--corpus", str(corpus), "--out-dir", str(tmp_path / "m2"), "--features", "true", "--n-belief", "4"])
    man = json.load(open(tmp_path / "m2" / "manifest.json"))
    assert man["features"] == "true" and man["prior"] == "lab_generator" and man["kind"] == out["kind"]
    assert os.path.exists(tmp_path / "m2" / man["success_model"]) and os.path.exists(tmp_path / "m2" / man["hazard_model"])
    prims = build_lab(bundle=str(tmp_path / "m2"))
    assert prims.model.kind == man["kind"] and prims.prior is not None and prims.manifest == man
    if before is not None:
        assert os.path.getmtime(M1_MODEL) == before                          # milestone 1's files are not touched


def test_train_models_refuses_rows_without_pose_under_true_features(tmp_path):
    from robots.mujoco_lab.train_models import main
    from tests.unit_tests.skills.synthetic import synthetic_rows

    corpus = tmp_path / "c.jsonl"
    _write(synthetic_rows(50), corpus)
    with pytest.raises(SystemExit, match="pose"):
        main(["--corpus", str(corpus), "--out-dir", str(tmp_path / "m2"), "--features", "true"])


def test_train_models_on_true_features(tmp_path):
    from robots.mujoco_lab.corpus import main as corpus_main
    from robots.mujoco_lab.train_models import main

    corpus = tmp_path / "c.jsonl"
    assert corpus_main(["--out", str(corpus), "--draws", "8", "--workers", "1", "--objects", "ycb__banana,ycb__brick,ycb__mug"]) == 24
    out = main(["--corpus", str(corpus), "--out-dir", str(tmp_path / "m2"), "--n-belief", "4"])
    assert out["features"] == "true" and out["prior"] == "lab_generator"
    assert out["classifier"]["features"] == "true" and out["classifier"]["belief"]["n"] == out["n_val"]


def test_train_models_folds_hold_out_each_fold(tmp_path):
    """--folds K: one bundle per fold, trained and tuned on the other folds' objects (an inner split)."""
    from robots.mujoco_lab.train_models import main
    from tests.unit_tests.skills.synthetic import synthetic_rows, with_pose

    corpus = tmp_path / "c.jsonl"
    _write(with_pose(synthetic_rows(300)), corpus)
    out = main(["--corpus", str(corpus), "--out-dir", str(tmp_path / "folds"), "--folds", "3", "--n-belief", "4"])
    folds = json.load(open(tmp_path / "folds" / "folds.json"))
    objects = {f"o{i}" for i in range(30)}
    assert set(folds) == objects and sorted(list(folds.values()).count(k) for k in range(3)) == [10, 10, 10]
    for k in range(3):
        d = tmp_path / "folds" / f"fold_{k}"
        man = json.load(open(d / "manifest.json"))
        split = json.load(open(d / man["metrics"]))["split"]
        held = {o for o, f in folds.items() if f == k}
        assert set(split["test"]) == held and not (set(split["train"]) | set(split["val"])) & held
        assert set(split["train"]) | set(split["val"]) | held == objects and not set(split["train"]) & set(split["val"])
        assert man["fold"] == k and len(split["val"]) >= 1
    assert out["folds"] == folds


def test_load_rows_keeps_only_trainable_rows(tmp_path):
    """An error row and a declined row (a pinch the skill never offers) do not reach the fit."""
    from robots.mujoco_lab.train_models import load_rows

    path = tmp_path / "c.jsonl"
    _write([{"key": "a__d0", "error": "boom"}, {"key": "b__d0", "declined": "wider than the gripper opening"},
            {"key": "c__d0", "label": "placed_stable"}], path)
    assert [r["key"] for r in load_rows(str(path))] == ["c__d0"]
