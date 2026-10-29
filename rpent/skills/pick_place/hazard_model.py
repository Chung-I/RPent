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

"""The loss-time hazard model.

Three heads: (a) not lifted at all; (b) a discrete-time hazard of losing the object per
control step, constant within a phase, as a function of the features and the phase; (c) the
release outcome among placed_stable / tipped / missed / rolled_off for an object that reaches
the release. The drop height is never predicted: it is the planned height of the object at
the step the loss is sampled at. The release phase is never at risk (the pads open on purpose).
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from rpent.skills.pick_place.evaluate import belief_features, score, true_features
from rpent.skills.pick_place.features import FEATURES_M2, as_vector
from rpent.skills.pick_place.protocol import DROP_HEIGHT_BINS, MOTION_OUTCOMES, OUTCOMES, Prediction

AT_RISK = ("approach", "close", "lift", "hold", "transport", "lower")
RELEASE_CLASSES = ("placed_stable", "tipped", "missed", "rolled_off")
HIGH_DROP_M = 0.10
EPS = 1e-6


def hazard_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-phase Bernoulli records for the lifted rows: one negative record weighted by the steps
    survived in the phase, plus one positive record (weight 1) in the phase of the loss."""
    recs = []
    for r in rows:
        if r["label"] == "not_lifted":
            continue
        lost = r.get("loss_phase") is not None and r.get("loss_step") is not None
        offset = 0
        for seg in r["planned"]:
            phase, n = seg["phase"], int(seg["n_steps"])
            if phase not in AT_RISK:
                break
            if lost and r["loss_phase"] == phase:
                k = max(1, min(n, int(r["loss_step"]) - offset))
                if k - 1 > 0:
                    recs.append({"features": r["features"], "phase": phase, "event": 0, "weight": k - 1})
                recs.append({"features": r["features"], "phase": phase, "event": 1, "weight": 1})
                break
            if n > 0:
                recs.append({"features": r["features"], "phase": phase, "event": 0, "weight": n})
            offset += n
    return recs


def _phase_onehot(phase: str) -> np.ndarray:
    return np.array([1.0 if phase == p else 0.0 for p in AT_RISK])


def _X_hazard(features: dict[str, float], phase: str, names=FEATURES_M2) -> np.ndarray:
    return np.concatenate([as_vector(features, names), _phase_onehot(phase)])


class _Head:
    """A classifier head with a constant fallback."""

    def __init__(self, clf: Any, classes: list[str], const: dict[str, float] | None):
        self.clf, self.classes, self.const = clf, classes, const

    def proba(self, X: np.ndarray) -> list[dict[str, float]]:
        M = self.proba_matrix(X)
        return [dict(zip(self.classes, map(float, row))) for row in M]

    def proba_matrix(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities for the rows of X, columns in ``self.classes`` order."""
        X = np.asarray(X, dtype=float)
        if self.const is not None:
            return np.tile([self.const.get(c, 0.0) for c in self.classes], (len(X), 1))
        P = self.clf.predict_proba(X)
        out = np.zeros((len(X), len(self.classes)))
        for j, c in enumerate(self.clf.classes_):
            if str(c) in self.classes:
                out[:, self.classes.index(str(c))] = P[:, j]
        return out


def _fit_head(X: np.ndarray, y: np.ndarray, classes: list[str], seed: int, weight: np.ndarray | None = None,
              logistic: bool = False) -> _Head:
    present = sorted(set(y.tolist()))
    if len(present) <= 1 or len(X) < 5:
        if len(present) == 0:
            const = {c: 1.0 / len(classes) for c in classes}
        else:
            const = {c: (1.0 - EPS * (len(classes) - 1)) if c == present[0] else EPS for c in classes}
        return _Head(None, classes, const)
    if logistic:
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(X, y, sample_weight=weight)
    else:
        clf = GradientBoostingClassifier(n_estimators=150, max_depth=2, random_state=seed).fit(X, y, sample_weight=weight)
    return _Head(clf, classes, None)


class HazardModel:
    kind = "hazard"
    feature_names: tuple[str, ...] = FEATURES_M2           # the class default: a model saved before the schemas

    def __init__(self, not_lifted: _Head, hazard: _Head, release: _Head, hz: float = 50.0, feature_names=FEATURES_M2):
        self.not_lifted, self.hazard, self.release, self.hz = not_lifted, hazard, release, hz
        self.feature_names = tuple(feature_names)

    @staticmethod
    def supports(labels) -> bool:
        """Whether the outcome walk can model these labels: motion outcomes only (no damage class)."""
        return all(lbl in MOTION_OUTCOMES for lbl in labels)

    @staticmethod
    def load(path: str) -> HazardModel:
        return joblib.load(path)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    def loss_hazard(self, features: dict[str, float], phase: str) -> float:
        if phase not in AT_RISK:
            return 0.0
        return float(self.hazard.proba(_X_hazard(features, phase, self.feature_names)[None, :])[0]["1"])

    def p_not_lifted(self, features: dict[str, float]) -> float:
        return float(self.not_lifted.proba(as_vector(features, self.feature_names)[None, :])[0]["1"])

    def predict_matrix(self, X: np.ndarray, planned: list[dict[str, Any]] | None = None
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The outcome walk for every feature row at once: (P (n x len(OUTCOMES), OUTCOMES order), drop-height
        bins (n x 4), expected loss time (n,), nan where no loss mass)."""
        if planned is None:
            raise ValueError("the hazard model needs the planned trajectory")
        X = np.asarray(X, dtype=float)
        n = len(X)
        p_nl = self.not_lifted.proba_matrix(X)[:, self.not_lifted.classes.index("1")]
        surv, low, high, t_mass = np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n)
        bins = np.zeros((n, len(DROP_HEIGHT_BINS)))
        t = 0.0
        for seg in planned:
            phase, ns = seg["phase"], int(seg["n_steps"])
            if phase not in AT_RISK or ns <= 0:
                t += ns / self.hz
                continue
            Xh = np.hstack([X, np.tile(_phase_onehot(phase), (n, 1))])
            h = np.clip(self.hazard.proba_matrix(Xh)[:, self.hazard.classes.index("1")], 0.0, 1.0)
            ks = np.arange(1, ns + 1)
            p_loss = surv[:, None] * (1 - h[:, None]) ** (ks - 1)[None, :] * h[:, None]
            z = seg["z_start"] + (seg["z_end"] - seg["z_start"]) * (ks / ns)
            high_mask = (z >= HIGH_DROP_M) | (not seg.get("has_surface", True))
            high += p_loss[:, high_mask].sum(1)
            low += p_loss[:, ~high_mask].sum(1)
            for j, m in enumerate((z < 0.05, (z >= 0.05) & (z < 0.10), (z >= 0.10) & (z < 0.20), z >= 0.20)):
                bins[:, j] += p_loss[:, m].sum(1)
            t_mass += (p_loss * (t + ks / self.hz)[None, :]).sum(1)
            surv *= (1 - h) ** ns
            t += ns / self.hz
        rel = self.release.proba_matrix(X)
        lifted = 1.0 - p_nl
        P = np.zeros((n, len(OUTCOMES)))
        P[:, OUTCOMES.index("not_lifted")] = p_nl
        P[:, OUTCOMES.index("dropped_low")] = lifted * low
        P[:, OUTCOMES.index("dropped_high")] = lifted * high
        for c in RELEASE_CLASSES:
            P[:, OUTCOMES.index(c)] = lifted * surv * rel[:, self.release.classes.index(c)]
        total = P.sum(1)
        P /= total[:, None]
        DH = lifted[:, None] * bins / total[:, None]
        loss_mass = low + high
        LT = np.where(loss_mass > 1e-9, t_mass / np.maximum(loss_mass, 1e-300), np.nan)
        return P, DH, LT

    def outcome_distribution(self, features: dict[str, float], planned: list[dict[str, Any]]) -> Prediction:
        return self.predict_many([features], planned)[0]

    def predict_many(self, features: list[dict[str, float]], planned: list[dict[str, Any]] | None = None) -> list[Prediction]:
        if planned is None:
            raise ValueError("the hazard model needs the planned trajectory")
        P, DH, LT = self.predict_matrix(np.array([as_vector(f, self.feature_names) for f in features]), planned)
        return predictions_from_matrix(P, DH, LT)


def predictions_from_matrix(P: np.ndarray, DH: np.ndarray | None = None, LT: np.ndarray | None = None) -> list[Prediction]:
    """Per-row Prediction objects from predict_matrix's arrays."""
    out = []
    for i in range(len(P)):
        dh = None if DH is None else dict(zip(DROP_HEIGHT_BINS, map(float, DH[i])))
        lt = None if LT is None or not np.isfinite(LT[i]) else float(LT[i])
        out.append(Prediction(p=dict(zip(OUTCOMES, map(float, P[i]))), drop_height=dh, expected_loss_t=lt))
    return out


def fit_hazard_model(rows: list[dict[str, Any]], seed: int = 0, names=FEATURES_M2) -> HazardModel:
    """The hazard model on the rows' features in ``names`` order; motion labels only (HazardModel.supports)."""
    if not HazardModel.supports(r["label"] for r in rows):
        raise ValueError("the hazard model needs motion labels only")
    X_all = np.array([as_vector(r["features"], names) for r in rows])
    y_nl = np.array(["1" if r["label"] == "not_lifted" else "0" for r in rows])
    not_lifted = _fit_head(X_all, y_nl, ["0", "1"], seed, logistic=True)
    recs = hazard_records(rows)
    if recs:
        Xh = np.array([_X_hazard(r["features"], r["phase"], names) for r in recs])
        yh = np.array([str(r["event"]) for r in recs])
        wh = np.array([float(r["weight"]) for r in recs])
        hazard = _fit_head(Xh, yh, ["0", "1"], seed, weight=wh)
    else:
        hazard = _Head(None, ["0", "1"], {"0": 1.0 - EPS, "1": EPS})
    rel_rows = [r for r in rows if r["label"] in RELEASE_CLASSES]
    if rel_rows:
        Xr = np.array([as_vector(r["features"], names) for r in rel_rows])
        yr = np.array([r["label"] for r in rel_rows])
        release = _fit_head(Xr, yr, list(RELEASE_CLASSES), seed)
    else:
        release = _Head(None, list(RELEASE_CLASSES), {c: 1.0 / len(RELEASE_CLASSES) for c in RELEASE_CLASSES})
    return HazardModel(not_lifted, hazard, release, feature_names=names)


def _score_hazard(model: HazardModel, rows_val: list[dict[str, Any]], rows_train: list[dict[str, Any]], seed: int,
                  n_belief: int, prior_fn=None, features: str = "nominal") -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    oracle, belief = [], []
    for r in rows_val:
        oracle.append(model.outcome_distribution(true_features(r) if features == "true" else r["features"], r["planned"]))
        belief.append(Prediction.mean(model.predict_many(belief_features(r, rng, n_belief, prior_fn), r["planned"])))
    out = {"oracle": score(oracle, rows_val, rows_train), "belief": score(belief, rows_val, rows_train)}
    for name, preds in (("oracle", oracle), ("belief", belief)):
        errs = [abs(p.expected_loss_t - r["loss_t"]) for p, r in zip(preds, rows_val)
                if r.get("loss_t") is not None and p.expected_loss_t is not None]
        out[name]["loss_time_mae"] = float(np.mean(errs)) if errs else None
    return out


def train_hazard(rows_train: list[dict[str, Any]], rows_val: list[dict[str, Any]], out_path: str, seed: int = 0,
                 n_belief: int = 16, prior_fn=None, features: str = "nominal", names=FEATURES_M2) -> dict[str, Any]:
    fit_rows = [dict(r, features=true_features(r)) for r in rows_train] if features == "true" else rows_train
    model = fit_hazard_model(fit_rows, seed, names)
    model.save(out_path)
    metrics = _score_hazard(model, rows_val, rows_train, seed, n_belief, prior_fn=prior_fn, features=features)
    metrics.update({"kind": "hazard", "held_out_objects": sorted({r["object"] for r in rows_val}),
                    "n_train": len(rows_train), "n_records": len(hazard_records(fit_rows)), "path": out_path,
                    "features": features})
    return metrics
