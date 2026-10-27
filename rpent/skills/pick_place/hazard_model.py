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

from rpent.skills.pick_place.evaluate import belief_features, score
from rpent.skills.pick_place.features import as_vector
from rpent.skills.pick_place.protocol import DROP_HEIGHT_BINS, OUTCOMES, Prediction

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


def _X_hazard(features: dict[str, float], phase: str) -> np.ndarray:
    return np.concatenate([as_vector(features), _phase_onehot(phase)])


class _Head:
    """A classifier head with a constant fallback."""

    def __init__(self, clf: Any, classes: list[str], const: dict[str, float] | None):
        self.clf, self.classes, self.const = clf, classes, const

    def proba(self, X: np.ndarray) -> list[dict[str, float]]:
        if self.const is not None:
            return [dict(self.const) for _ in range(len(X))]
        P = self.clf.predict_proba(X)
        out = []
        for row in P:
            d = {c: 0.0 for c in self.classes}
            for j, c in enumerate(self.clf.classes_):
                d[str(c)] = float(row[j])
            out.append(d)
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

    def __init__(self, not_lifted: _Head, hazard: _Head, release: _Head, hz: float = 50.0):
        self.not_lifted, self.hazard, self.release, self.hz = not_lifted, hazard, release, hz

    @staticmethod
    def load(path: str) -> HazardModel:
        return joblib.load(path)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    def loss_hazard(self, features: dict[str, float], phase: str) -> float:
        if phase not in AT_RISK:
            return 0.0
        return float(self.hazard.proba(_X_hazard(features, phase)[None, :])[0]["1"])

    def p_not_lifted(self, features: dict[str, float]) -> float:
        return float(self.not_lifted.proba(as_vector(features)[None, :])[0]["1"])

    def outcome_distribution(self, features: dict[str, float], planned: list[dict[str, Any]]) -> Prediction:
        p_nl = self.p_not_lifted(features)
        surv = 1.0
        low = high = 0.0
        bins = {b: 0.0 for b in DROP_HEIGHT_BINS}
        t_mass = 0.0
        t = 0.0
        for seg in planned:
            phase, n = seg["phase"], int(seg["n_steps"])
            if phase not in AT_RISK or n <= 0:
                t += n / self.hz
                continue
            h = self.loss_hazard(features, phase)
            if h <= 0:
                t += n / self.hz
                continue
            ks = np.arange(1, n + 1)
            p_loss = surv * (1 - h) ** (ks - 1) * h
            z = seg["z_start"] + (seg["z_end"] - seg["z_start"]) * (ks / n)
            high_mask = (z >= HIGH_DROP_M) | (not seg.get("has_surface", True))
            high += float(p_loss[high_mask].sum())
            low += float(p_loss[~high_mask].sum())
            bins["0.00-0.05"] += float(p_loss[z < 0.05].sum())
            bins["0.05-0.10"] += float(p_loss[(z >= 0.05) & (z < 0.10)].sum())
            bins["0.10-0.20"] += float(p_loss[(z >= 0.10) & (z < 0.20)].sum())
            bins["0.20+"] += float(p_loss[z >= 0.20].sum())
            t_mass += float((p_loss * (t + ks / self.hz)).sum())
            surv *= (1 - h) ** n
            t += n / self.hz
        rel = self.release.proba(as_vector(features)[None, :])[0]
        lifted = 1.0 - p_nl
        p = {k: 0.0 for k in OUTCOMES}
        p["not_lifted"] = p_nl
        p["dropped_low"] = lifted * low
        p["dropped_high"] = lifted * high
        for c in RELEASE_CLASSES:
            p[c] = lifted * surv * rel.get(c, 0.0)
        total = sum(p.values())
        p = {k: v / total for k, v in p.items()}
        dh = {b: lifted * v / total for b, v in bins.items()}
        loss_mass = low + high
        expected_t = (t_mass / loss_mass) if loss_mass > 1e-9 else None
        return Prediction(p=p, drop_height=dh, expected_loss_t=expected_t)

    def predict_many(self, features: list[dict[str, float]], planned: list[dict[str, Any]] | None = None) -> list[Prediction]:
        if planned is None:
            raise ValueError("the hazard model needs the planned trajectory")
        return [self.outcome_distribution(f, planned) for f in features]


def fit_hazard_model(rows: list[dict[str, Any]], seed: int = 0) -> HazardModel:
    X_all = np.array([as_vector(r["features"]) for r in rows])
    y_nl = np.array(["1" if r["label"] == "not_lifted" else "0" for r in rows])
    not_lifted = _fit_head(X_all, y_nl, ["0", "1"], seed, logistic=True)
    recs = hazard_records(rows)
    if recs:
        Xh = np.array([_X_hazard(r["features"], r["phase"]) for r in recs])
        yh = np.array([str(r["event"]) for r in recs])
        wh = np.array([float(r["weight"]) for r in recs])
        hazard = _fit_head(Xh, yh, ["0", "1"], seed, weight=wh)
    else:
        hazard = _Head(None, ["0", "1"], {"0": 1.0 - EPS, "1": EPS})
    rel_rows = [r for r in rows if r["label"] in RELEASE_CLASSES]
    if rel_rows:
        Xr = np.array([as_vector(r["features"]) for r in rel_rows])
        yr = np.array([r["label"] for r in rel_rows])
        release = _fit_head(Xr, yr, list(RELEASE_CLASSES), seed)
    else:
        release = _Head(None, list(RELEASE_CLASSES), {c: 1.0 / len(RELEASE_CLASSES) for c in RELEASE_CLASSES})
    return HazardModel(not_lifted, hazard, release)


def _score_hazard(model: HazardModel, rows_val: list[dict[str, Any]], rows_train: list[dict[str, Any]], seed: int,
                  n_belief: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    oracle, belief = [], []
    for r in rows_val:
        oracle.append(model.outcome_distribution(r["features"], r["planned"]))
        belief.append(Prediction.mean([model.outcome_distribution(f, r["planned"]) for f in belief_features(r, rng, n_belief)]))
    out = {"oracle": score(oracle, rows_val, rows_train), "belief": score(belief, rows_val, rows_train)}
    for name, preds in (("oracle", oracle), ("belief", belief)):
        errs = [abs(p.expected_loss_t - r["loss_t"]) for p, r in zip(preds, rows_val)
                if r.get("loss_t") is not None and p.expected_loss_t is not None]
        out[name]["loss_time_mae"] = float(np.mean(errs)) if errs else None
    return out


def train_hazard(rows_train: list[dict[str, Any]], rows_val: list[dict[str, Any]], out_path: str, seed: int = 0,
                 n_belief: int = 16) -> dict[str, Any]:
    model = fit_hazard_model(rows_train, seed)
    model.save(out_path)
    metrics = _score_hazard(model, rows_val, rows_train, seed, n_belief)
    metrics.update({"kind": "hazard", "held_out_objects": sorted({r["object"] for r in rows_val}),
                    "n_train": len(rows_train), "n_records": len(hazard_records(rows_train)), "path": out_path})
    return metrics
