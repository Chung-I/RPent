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

"""The feature classifier: training, calibration, degenerate heads."""

from rpent.skills.pick_place.evaluate import split_objects
from rpent.skills.pick_place.protocol import OUTCOMES
from rpent.skills.pick_place.success_model import SuccessModel, train
from tests.unit_tests.skills.synthetic import synthetic_rows


def _split(rows):
    sp = split_objects(rows, seed=0)
    return [r for r in rows if r["object"] in sp["train"]], [r for r in rows if r["object"] in sp["val"]]


def test_train_then_predict_sums_to_one(tmp_path):
    tr, va = _split(synthetic_rows(400))
    train(tr, va, str(tmp_path / "m.joblib"))
    sm = SuccessModel.load(str(tmp_path / "m.joblib"))
    p = sm.predict(va[0]["features"])
    assert set(p.p) == set(OUTCOMES) and abs(sum(p.p.values()) - 1) < 1e-6 and p.drop_height is None
    assert len(sm.predict_many([r["features"] for r in va[:5]])) == 5


def test_missing_class_gets_zero_and_single_class_is_constant(tmp_path):
    rows = [r for r in synthetic_rows(400) if r["label"] != "rolled_off"]
    tr, va = _split(rows)
    train(tr, va, str(tmp_path / "m.joblib"))
    assert SuccessModel.load(str(tmp_path / "m.joblib")).predict(va[0]["features"]).p["rolled_off"] == 0.0
    one = [dict(r, label="placed_stable") for r in synthetic_rows(60)]
    tr1, va1 = _split(one)
    train(tr1, va1, str(tmp_path / "one.joblib"))
    p = SuccessModel.load(str(tmp_path / "one.joblib")).predict(va1[0]["features"]).p
    assert p["placed_stable"] > 0.99


def test_calibrated_on_val_in_belief_mode(tmp_path):
    tr, va = _split(synthetic_rows(2000, noise=0.03))
    m = train(tr, va, str(tmp_path / "m.joblib"))
    assert set(m) >= {"oracle", "belief", "kind", "held_out_objects"}
    assert m["oracle"]["brier"] < m["oracle"]["brier_const"]
    assert m["belief"]["brier"] < m["belief"]["brier_const"] and m["belief"]["ece"] < 0.08


import numpy as np  # noqa: E402


class _RecordingPrior:
    """A prior_fn stub: records its calls, returns n particles at the nominal CoM."""

    def __init__(self):
        self.calls = []

    def __call__(self, pose, n, rng):
        from rpent.skills.pick_place.belief import ParticleBelief

        self.calls.append(pose)
        parts = [{"mass": float(m), "mu": float(u), "com_obj": [0.0, 0.0, 0.03]}
                 for m, u in zip(rng.uniform(0.05, 1.0, n), rng.uniform(0.1, 0.9, n))]
        return ParticleBelief(parts, np.ones(n))


def test_evaluate_model_scores_with_the_prior_fn(tmp_path):
    from rpent.skills.pick_place.success_model import evaluate_model, fit_success_model
    from tests.unit_tests.skills.synthetic import with_pose

    rows = with_pose(synthetic_rows(400))
    tr, va = _split(rows)
    model = fit_success_model(tr)
    prior = _RecordingPrior()
    m = evaluate_model(model, va, tr, prior_fn=prior, features="true")
    assert len(prior.calls) == len(va) and m["belief"]["n"] == len(va)
    plain = evaluate_model(model, [{k: v for k, v in r.items() if k not in ("pose", "com_obj_true")} for r in va], tr)
    assert plain["belief"]["n"] == len(va)                                   # without it: the milestone 1 path, no pose needed


def test_predict_matrix_equals_predict_many():
    from rpent.skills.pick_place.features import as_vector
    from rpent.skills.pick_place.protocol import OUTCOMES
    from rpent.skills.pick_place.success_model import fit_success_model

    rows = synthetic_rows(300)
    model = fit_success_model(rows)
    feats = [r["features"] for r in rows[:20]]
    P, dh, lt = model.predict_matrix(np.array([as_vector(f) for f in feats]))
    want = np.array([[p.p[k] for k in OUTCOMES] for p in model.predict_many(feats)])
    assert dh is None and lt is None and np.allclose(P, want)
