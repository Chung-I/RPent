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


def _damage_rows(n: int = 600, seed: int = 3) -> list[dict]:
    """Synthetic rows of three kinds of object (by object id): steel (no flag), crushable boxes (frag_crush, half the
    rows crushed) and fillable cups (fill 0.5, a quarter of the rows spilled)."""
    out = []
    for i, r in enumerate(synthetic_rows(n, seed=seed)):
        kind = i % 3
        f = dict(r["features"], frag_crush=float(kind == 1), fillable=float(kind == 2), fill=0.5 if kind == 2 else 0.0)
        label = "crushed" if kind == 1 and i % 2 == 0 else ("spilled" if kind == 2 and i % 4 == 0 else r["label"])
        out.append(dict(r, features=f, label=label))
    return out


def test_models_store_their_feature_names(tmp_path):
    from rpent.skills.pick_place.features import FEATURES_M2, FEATURES_M2B
    from rpent.skills.pick_place.success_model import fit_success_model

    m = fit_success_model(_damage_rows(), names=FEATURES_M2B)
    assert m.feature_names == FEATURES_M2B
    m.save(str(tmp_path / "b.joblib"))
    assert SuccessModel.load(str(tmp_path / "b.joblib")).feature_names == FEATURES_M2B
    old = fit_success_model(synthetic_rows(300))
    assert old.feature_names == FEATURES_M2
    vars(old).pop("feature_names", None)                     # a model saved before the schemas
    old.save(str(tmp_path / "old.joblib"))
    assert SuccessModel.load(str(tmp_path / "old.joblib")).feature_names == FEATURES_M2


def test_masks_zero_inapplicable_damage_classes():
    from rpent.skills.pick_place.features import FEATURES_M2B, as_vector
    from rpent.skills.pick_place.success_model import fit_success_model

    rows = _damage_rows()
    m = fit_success_model(rows, names=FEATURES_M2B)
    assert {"crushed", "spilled"} <= set(m.classes)
    steel = [r["features"] for i, r in enumerate(rows) if i % 3 == 0][:20]
    cup = [r["features"] for i, r in enumerate(rows) if i % 3 == 2][:20]
    for p in m.predict_many(steel):
        assert p.p["crushed"] == 0.0 and p.p["spilled"] == 0.0 and abs(sum(p.p.values()) - 1.0) < 1e-9
    for p in m.predict_many(cup):
        assert p.p["spilled"] > 0.0 and p.p["crushed"] == 0.0
    P, _, _ = m.predict_matrix(np.array([as_vector(f, FEATURES_M2B) for f in steel]))
    assert (P[:, OUTCOMES.index("crushed")] == 0.0).all() and np.allclose(P.sum(axis=1), 1.0)


def _shared_point_rows(labels):
    """1200 rows split at x0 = 0, then five rare rows and one common row at one point that no split can divide."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1200, 3))
    y = np.where(X[:, 0] > 0, labels[0], labels[1]).astype(object)
    X[:6] = 3.0
    y[:5], y[5] = labels[2], labels[0]
    return X, y.astype(type(labels[0]))


def test_boosting_stays_bounded_when_a_rare_class_shares_a_point():
    """P(rare) at the shared point is the rare rows' share there, 5/6. An unregularized Newton leaf (the sum of
    the residuals over the sum of p(1 - p)) diverges on this data: P(rare) goes to 1 and the leaf values to 1e34.
    The m2b corpora have rare classes of 9 to 81 rows, and that divergence corrupted the out-of-fold calibration."""
    from rpent.skills.pick_place.success_model import _fit_gbm

    for labels in (("a", "b", "rare"), (0, 0, 1)):
        X, y = _shared_point_rows(labels)
        clf = _fit_gbm(X, y, 0)
        cls = list(clf.classes_)
        P = clf.predict_proba(np.array([[3.0, 3.0, 3.0], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]))
        assert abs(P[0, cls.index(labels[2])] - 5 / 6) < 0.05, (labels, P[0])
        assert P[1, cls.index(labels[0])] > 0.9 and P[2, cls.index(labels[1])] > 0.9, (labels, P)


def test_boosting_runs_openmp_at_one_thread(monkeypatch):
    """Fit and predict at one OpenMP thread. The results do not depend on the count, and at 20 threads the idle
    workers spin between the many small predictions and slowed this file from 12 s to 288 s."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

    from rpent.skills.pick_place.success_model import _fit_gbm

    seen = []
    fit, raw = HistGradientBoostingClassifier.fit, HistGradientBoostingClassifier._raw_predict

    def fit_counting(self, *a, **k):
        seen.append(("fit", _openmp_effective_n_threads()))
        return fit(self, *a, **k)

    def raw_counting(self, *a, **k):
        seen.append(("predict", _openmp_effective_n_threads()))
        return raw(self, *a, **k)

    monkeypatch.setattr(HistGradientBoostingClassifier, "fit", fit_counting)
    monkeypatch.setattr(HistGradientBoostingClassifier, "_raw_predict", raw_counting)
    X, y = _shared_point_rows(("a", "b", "rare"))
    _fit_gbm(X, y, 0).predict_proba(X[:4])
    assert {k for k, _ in seen} == {"fit", "predict"} and {n for _, n in seen} == {1}, seen
