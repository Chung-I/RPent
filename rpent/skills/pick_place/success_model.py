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

"""The feature classifier: gradient boosting over the seven outcomes, calibrated per class
on out-of-fold predictions grouped by object (isotonic when every class has enough rows,
else a sigmoid), with constant predictors for degenerate targets."""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from rpent.skills.pick_place.evaluate import MIN_CLASS_ROWS, belief_features, score
from rpent.skills.pick_place.features import as_vector
from rpent.skills.pick_place.protocol import OUTCOMES, Prediction

N_ESTIMATORS = 200
MAX_DEPTH = 2
N_FOLDS = 5
EPS = 1e-4


class _Calibrator:
    """Per-class monotone map from a raw probability to a calibrated one."""

    def __init__(self, kind: str):
        self.kind = kind
        self.maps: dict[str, Any] = {}

    def fit(self, raw: np.ndarray, Y: np.ndarray, classes: list[str]) -> _Calibrator:
        for j, c in enumerate(classes):
            y = Y[:, j]
            if y.min() == y.max():
                self.maps[c] = float(y.mean())
                continue
            if self.kind == "isotonic":
                m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw[:, j], y)
            else:
                x = np.log(np.clip(raw[:, j], EPS, 1 - EPS) / (1 - np.clip(raw[:, j], EPS, 1 - EPS))).reshape(-1, 1)
                m = LogisticRegression(C=1.0).fit(x, y)
            self.maps[c] = m
        return self

    def transform(self, raw: np.ndarray, classes: list[str]) -> np.ndarray:
        out = np.zeros_like(raw)
        for j, c in enumerate(classes):
            m = self.maps[c]
            if isinstance(m, float):
                out[:, j] = m
            elif self.kind == "isotonic":
                out[:, j] = m.predict(raw[:, j])
            else:
                x = np.log(np.clip(raw[:, j], EPS, 1 - EPS) / (1 - np.clip(raw[:, j], EPS, 1 - EPS))).reshape(-1, 1)
                out[:, j] = m.predict_proba(x)[:, 1]
        return out


class SuccessModel:
    kind = "classifier"

    def __init__(self, clf: Any, classes: list[str], calibrator: _Calibrator | None, const: np.ndarray | None):
        self.clf, self.classes, self.calibrator, self.const = clf, classes, calibrator, const

    @staticmethod
    def load(path: str) -> SuccessModel:
        return joblib.load(path)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    def _proba(self, X: np.ndarray) -> np.ndarray:
        if self.const is not None:
            return np.tile(self.const, (len(X), 1))
        raw = self.clf.predict_proba(X)
        if self.calibrator is not None:
            raw = self.calibrator.transform(raw, self.classes)
        raw = np.clip(raw, EPS, None)
        return raw / raw.sum(axis=1, keepdims=True)

    def predict_many(self, features: list[dict[str, float]]) -> list[Prediction]:
        X = np.array([as_vector(f) for f in features])
        P = self._proba(X)
        out = []
        for row in P:
            p = {k: 0.0 for k in OUTCOMES}
            for j, c in enumerate(self.classes):
                p[c] = float(row[j])
            out.append(Prediction(p=p))
        return out

    def predict(self, features: dict[str, float]) -> Prediction:
        return self.predict_many([features])[0]


def _fit_gbm(X: np.ndarray, y: np.ndarray, seed: int) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=seed).fit(X, y)


def fit_success_model(rows: list[dict[str, Any]], seed: int = 0) -> SuccessModel:
    X = np.array([as_vector(r["features"]) for r in rows])
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["object"] for r in rows])
    classes = sorted(set(y))
    if len(classes) == 1:
        const = np.array([1.0 if c == classes[0] else 0.0 for c in OUTCOMES])
        return SuccessModel(None, list(OUTCOMES), None, const)
    counts = {c: int((y == c).sum()) for c in classes}
    kind = "isotonic" if min(counts.values()) >= MIN_CLASS_ROWS else "sigmoid"
    n_groups = len(set(groups))
    calibrator = None
    if n_groups >= N_FOLDS:
        oof = np.zeros((len(rows), len(classes)))
        for tr, te in GroupKFold(N_FOLDS).split(X, y, groups):
            clf = _fit_gbm(X[tr], y[tr], seed)
            P = clf.predict_proba(X[te])
            for j, c in enumerate(classes):
                if c in clf.classes_:
                    oof[te, j] = P[:, list(clf.classes_).index(c)]
        Y = np.array([[1.0 if lbl == c else 0.0 for c in classes] for lbl in y])
        calibrator = _Calibrator(kind).fit(oof, Y, classes)
    clf = _fit_gbm(X, y, seed)
    return SuccessModel(clf, list(clf.classes_), calibrator, None)


def evaluate_model(model: Any, rows_val: list[dict[str, Any]], rows_train: list[dict[str, Any]], seed: int = 0,
                   n_belief: int = 32) -> dict[str, Any]:
    """Oracle-mode (true θ) and belief-mode (prior samples) scores on the validation rows."""
    rng = np.random.default_rng(seed)
    oracle = model.predict_many([r["features"] for r in rows_val]) if hasattr(model, "predict_many") else [
        model.predict(r["features"]) for r in rows_val]
    belief = []
    for r in rows_val:
        feats = belief_features(r, rng, n_belief)
        preds = model.predict_many(feats) if hasattr(model, "predict_many") else [model.predict(f) for f in feats]
        belief.append(Prediction.mean(preds))
    return {"oracle": score(oracle, rows_val, rows_train), "belief": score(belief, rows_val, rows_train)}


def train(rows_train: list[dict[str, Any]], rows_val: list[dict[str, Any]], out_path: str, seed: int = 0) -> dict[str, Any]:
    model = fit_success_model(rows_train, seed)
    model.save(out_path)
    metrics = evaluate_model(model, rows_val, rows_train, seed)
    metrics.update({"kind": "classifier", "held_out_objects": sorted({r["object"] for r in rows_val}),
                    "classes": model.classes, "n_train": len(rows_train), "path": out_path})
    return metrics
