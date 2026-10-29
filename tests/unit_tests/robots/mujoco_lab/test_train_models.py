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
    assert corpus_main(["--out", str(corpus), "--draws", "8", "--workers", "1", "--objects", "ycb__banana,ycb__brick,ycb__mug",
                        "--mode", "m2"]) == 24
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


def test_the_split_and_the_folds_keep_an_object_whose_rows_are_all_filtered(tmp_path):
    """The split and the folds are drawn over the corpus objects (the rows without an error), so an object whose
    pinches are all declined keeps its part and the other objects keep theirs (corpus v8 declines every pinch of
    11 objects; drawn over the trainable rows, the v7 and v8 validation sets shared 4 of 14 objects)."""
    from robots.mujoco_lab.train_models import corpus_objects, main
    from rpent.skills.pick_place.evaluate import split_objects
    from tests.unit_tests.skills.synthetic import synthetic_rows, with_pose

    rows = with_pose(synthetic_rows(300))
    cut = tmp_path / "c.jsonl"
    _write([dict(r, declined="wider than the gripper opening") if r["object"] == "o3" else r for r in rows]
           + [{"key": "z__d0", "object": "z", "error": "boom"}], cut)
    objects = sorted({r["object"] for r in rows})
    assert corpus_objects([str(cut)]) == objects
    out = main(["--corpus", str(cut), "--out-dir", str(tmp_path / "m2"), "--n-belief", "4"])
    assert out["split"] == split_objects(rows, seed=0)
    folds = main(["--corpus", str(cut), "--out-dir", str(tmp_path / "folds"), "--folds", "3", "--n-belief", "4"])["folds"]
    assert sorted(folds) == objects


def test_a_bundle_carries_the_novelty_guard(tmp_path):
    """train_models fits the knn guard on the training rows: every training object with rows is a calibration
    object, and the bundle's loaders hand the guard to the skill."""
    from robots.mujoco_lab.robot_spec import build_lab, load_bundle
    from robots.mujoco_lab.train_models import main
    from rpent.skills.pick_place.guard import KnnGuard
    from tests.unit_tests.skills.synthetic import synthetic_rows, with_pose

    rows = with_pose(synthetic_rows(300))
    corpus = tmp_path / "c.jsonl"
    _write([dict(r, declined="wider than the gripper opening") if r["object"] == "o3" else r for r in rows], corpus)
    out = main(["--corpus", str(corpus), "--out-dir", str(tmp_path / "m2"), "--n-belief", "4"])
    man = json.load(open(tmp_path / "m2" / "manifest.json"))
    assert man["guard"] == "knn_guard.npz"
    g = KnnGuard.load(tmp_path / "m2" / man["guard"])
    assert sorted(g.calibration_objects) == sorted(set(out["split"]["train"]) - {"o3"})
    assert out["guard"]["n_calibration"] == g.n_calibration and out["guard"]["n_reference"] == len(g.reference)
    assert isinstance(load_bundle(str(tmp_path / "m2"))["guard"], KnnGuard)
    assert isinstance(build_lab(bundle=str(tmp_path / "m2")).guard, KnnGuard)


def test_the_guard_fits_into_an_existing_bundle_and_into_each_fold(tmp_path):
    """--guard alone: a bundle trained before the guard gets one from its manifest's corpus and split; a folds
    directory gets one per fold bundle."""
    from robots.mujoco_lab.train_models import main
    from rpent.skills.pick_place.guard import KnnGuard
    from tests.unit_tests.skills.synthetic import synthetic_rows, with_pose

    corpus = tmp_path / "c.jsonl"
    _write(with_pose(synthetic_rows(300)), corpus)

    def strip(d):
        man = json.load(open(d / "manifest.json"))
        os.remove(d / man.pop("guard"))
        json.dump(man, open(d / "manifest.json", "w"))

    main(["--corpus", str(corpus), "--out-dir", str(tmp_path / "m2"), "--n-belief", "4"])
    strip(tmp_path / "m2")
    out = main(["--guard", "--out-dir", str(tmp_path / "m2")])
    man = json.load(open(tmp_path / "m2" / "manifest.json"))
    split = json.load(open(tmp_path / "m2" / man["metrics"]))["split"]
    assert man["guard"] == "knn_guard.npz" and out["guard"]["n_calibration"] == len(split["train"])
    main(["--corpus", str(corpus), "--out-dir", str(tmp_path / "folds"), "--folds", "3", "--n-belief", "4"])
    for k in range(3):
        strip(tmp_path / "folds" / f"fold_{k}")
    out = main(["--guard", "--out-dir", str(tmp_path / "folds")])
    assert sorted(out["guard"]) == [0, 1, 2]
    for k in range(3):
        d = tmp_path / "folds" / f"fold_{k}"
        man = json.load(open(d / "manifest.json"))
        split = json.load(open(d / man["metrics"]))["split"]
        g = KnnGuard.load(d / man["guard"])
        assert sorted(g.calibration_objects) == sorted(split["train"]) and not set(g.calibration_objects) & set(split["test"])


def test_a_bundle_with_one_training_object_has_no_guard(tmp_path):
    """The guard's calibration needs two training objects; a bundle with one has no guard (the skill runs without)."""
    from robots.mujoco_lab.train_models import fit_bundle_guard
    from tests.unit_tests.skills.synthetic import synthetic_rows, with_pose

    rows = with_pose(synthetic_rows(60))
    assert fit_bundle_guard([r for r in rows if r["object"] == "o1"], str(tmp_path), "true") is None
    assert os.listdir(tmp_path) == []
    assert fit_bundle_guard([r for r in rows if r["object"] in ("o1", "o2")], str(tmp_path), "true")["n_calibration"] == 2


def test_hazard_model_not_eligible_with_damage_labels(tmp_path):
    """--feature-schema m2b on a corpus with damage labels: no hazard model, the classifier on FEATURES_M2B, and a
    manifest with the schemas and the decision; load_bundle refuses a manifest whose schema the models do not match."""
    from robots.mujoco_lab.robot_spec import load_bundle
    from robots.mujoco_lab.train_models import main
    from rpent.skills.pick_place.features import FEATURES_M2B
    from tests.unit_tests.skills.synthetic import synthetic_rows, with_pose

    rows = with_pose(synthetic_rows(300))
    rows = [dict(r, label="crushed", features=dict(r["features"], frag_crush=1.0)) if i % 10 == 0 else r for i, r in enumerate(rows)]
    corpus = tmp_path / "c.jsonl"
    _write(rows, corpus)
    out = main(["--corpus", str(corpus), "--out-dir", str(tmp_path / "b"), "--n-belief", "4", "--feature-schema", "m2b"])
    assert out["hazard"] == "not eligible: damage labels" and out["kind"] == "classifier"
    man = json.load(open(tmp_path / "b" / "manifest.json"))
    assert man["feature_schema"] == "m2b" and man["outcome_schema"] == "m2b" and man["decision"] == {"abstain_cost": 1.0}
    assert man.get("hazard_model") is None and not os.path.exists(tmp_path / "b" / "hazard_model.joblib")
    B = load_bundle(str(tmp_path / "b"))
    assert B["model"].feature_names == FEATURES_M2B and B["guard"].feature_names == FEATURES_M2B
    assert B["readout_model"] is None or B["readout_model"].rot_clf is None or B["readout_model"].rot_clf.feature_names == FEATURES_M2B
    assert (B["feature_schema"], B["outcome_schema"], B["abstain_cost"]) == ("m2b", "m2b", 1.0)
    json.dump(dict(man, feature_schema="m2"), open(tmp_path / "b" / "manifest.json", "w"))
    with pytest.raises(ValueError, match="feature names"):
        load_bundle(str(tmp_path / "b"))


def test_the_default_schema_is_m2(tmp_path):
    from robots.mujoco_lab.train_models import main
    from tests.unit_tests.skills.synthetic import synthetic_rows, with_pose

    corpus = tmp_path / "c.jsonl"
    _write(with_pose(synthetic_rows(300)), corpus)
    main(["--corpus", str(corpus), "--out-dir", str(tmp_path / "b"), "--n-belief", "4"])
    man = json.load(open(tmp_path / "b" / "manifest.json"))
    assert man["feature_schema"] == "m2" and man["outcome_schema"] == "m2" and man.get("decision") is None
    assert man["hazard_model"] == "hazard_model.joblib"


def test_the_m2_bundle_reads_the_m2_schema():
    """The m2 bundle's models were saved before the schemas: they read FEATURES_M2, and the rotation classifier keeps
    its seven label columns after OUTCOMES grows to eleven."""
    import numpy as np

    from robots.mujoco_lab.robot_spec import DEFAULT_BUNDLE, load_bundle
    from rpent.skills.pick_place.features import FEATURES_M2
    from rpent.skills.pick_place.protocol import MOTION_OUTCOMES

    B = load_bundle(str(DEFAULT_BUNDLE))
    assert (B["feature_schema"], B["outcome_schema"]) == ("m2", "m2")
    assert B["model"].feature_names == FEATURES_M2 and B["guard"].feature_names == FEATURES_M2
    assert all(pm.feature_names == FEATURES_M2 for pm in B["probe_models"].values())
    rot = B["readout_model"].rot_clf
    assert rot.feature_names == FEATURES_M2 and rot.label_classes == MOTION_OUTCOMES
    assert rot.predict(np.zeros((2, len(FEATURES_M2))), "placed_stable").shape == (2,)


def test_probe_metrics_serialise_mixture_laws(tmp_path):
    """Weigh readings with a 25 % wide tail: fit_probes selects the measured law's kind on an inner split of the
    training objects, writes a mixture as a JSON object and a normal law as [b, sd], and the saved model holds the Law."""
    import numpy as np

    from robots.mujoco_lab.train_models import fit_probes
    from rpent.skills.pick_place.likelihood import Law
    from rpent.skills.pick_place.probes import ProbeModel
    from tests.unit_tests.skills.test_probes import _probe_rows

    rng = np.random.default_rng(11)
    rows = _probe_rows(1200, sd=0.05)
    for r in rows:
        if r["kind"] == "measured" and rng.random() < 0.25:
            r["reading"] = {"m_hat": r["reading"]["m_hat"] * float(np.exp(rng.normal(0.0, 0.6)))}
    _write(rows, tmp_path / "p.jsonl")
    objects = sorted({r["object"] for r in rows})
    os.makedirs(tmp_path / "b")
    out = fit_probes(str(tmp_path / "p.jsonl"), str(tmp_path / "b"), {"train": objects[:17], "val": objects[17:]})
    metrics = json.load(open(tmp_path / "b" / "probe_metrics.json"))
    measured = metrics["probes"]["weigh"]["reading"]["measured"]
    assert measured["kind"] == "mix2" and set(measured) == {"kind", "w", "b1", "sd1", "b2", "sd2"}
    assert out["probes"]["weigh"]["reading"]["measured"] == measured
    assert Law.from_json(measured) == ProbeModel.load(str(tmp_path / "b" / "probe_weigh.joblib")).reading["measured"]
    slid = metrics["probes"]["push"]["reading"]["slid"]
    assert isinstance(slid, list) and len(slid) == 2


def test_load_bundle_takes_the_prior_from_the_manifest(tmp_path):
    """An m2 manifest gives today's lab prior; a manifest naming the m2b prior gives the joint prior on its class and
    threshold files, and a file that changed since training stops the load."""
    from robots.mujoco_lab.fragility import CLASSES, THRESHOLDS_V0
    from robots.mujoco_lab.prior import M2B_PRIOR, sha256_of
    from robots.mujoco_lab.robot_spec import lab_prior_fn, load_bundle
    from robots.mujoco_lab.train_models import main
    from tests.unit_tests.skills.synthetic import synthetic_rows, with_pose

    corpus = tmp_path / "c.jsonl"
    _write(with_pose(synthetic_rows(300)), corpus)
    main(["--corpus", str(corpus), "--out-dir", str(tmp_path / "b"), "--features", "true", "--n-belief", "4"])
    assert load_bundle(str(tmp_path / "b"))["prior"] is lab_prior_fn
    path = tmp_path / "b" / "manifest.json"
    man = {**json.load(open(path)), "prior": M2B_PRIOR, "classes": {"path": CLASSES, "sha256": sha256_of(CLASSES)},
           "thresholds": {"path": THRESHOLDS_V0, "sha256": sha256_of(THRESHOLDS_V0)}}
    json.dump(man, open(path, "w"))
    prior = load_bundle(str(tmp_path / "b"))["prior"]
    assert callable(prior) and prior is not lab_prior_fn
    json.dump({**man, "thresholds": {"path": THRESHOLDS_V0, "sha256": "0" * 64}}, open(path, "w"))
    with pytest.raises(ValueError, match="sha256"):
        load_bundle(str(tmp_path / "b"))
