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


"""The readout likelihood: how probable one attempt's outcome and pad readings are under each
belief particle.

A composite likelihood, validated by posterior coverage, not assumed exact. Terms of particle i:

* outcome: log max(floor_p, P_i[label]) / temper;
* hold (a hold reading): log N(log(fz / (m_i g)); b, sd * temper);
* slip (a slip reading, not lifted) and loss (the force just before a loss):
  log N(log(fz / (2 mu_i F)); b, sd * temper), F the squeeze per pad;
* rot (always, with a rotation classifier): log P(rotated or not | X_i, label) / temper;
* side (when the readouts tell a heavy side): log(1 - eps) on that side, log eps on the other,
  log 0.5 for a CoM on the pinch; divided by temper like the other discrete terms.

The residual laws (b, sd) are fitted per label on training rows at the true mass and friction;
a label with fewer than 30 rows uses the pooled law of all rows. A continuous term is rejected
(listed, not used) when its best particle lies more than 5 sd off, or when the reading is not
positive; the outcome, rot and side terms are never rejected.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

from rpent.skills.pick_place.costs import prediction_matrix
from rpent.skills.pick_place.evaluate import MIN_CLASS_ROWS, true_features
from rpent.skills.pick_place.features import FEATURES_M2, G, as_vector, particle_sides
from rpent.skills.pick_place.protocol import DAMAGE, MOTION_OUTCOMES, OUTCOMES
from rpent.skills.pick_place.success_model import N_FOLDS, _Calibrator, _fit_gbm

ROT_DEG = 10.0                      # rot_deg at or above this counts as rotated (the lab's ROT_SIDE_DEG)
REJECT_Z = 5.0                      # a continuous term whose best particle is this many sd off is rejected
MIN_LABEL_ROWS = 30                 # rows a label needs for its own residual law
SD_FLOOR = 0.05
SIDE_ERROR_FLOOR = 0.02
MAD_TO_SD = 1.4826
POOLED = "_pooled"
CONTINUOUS = ("hold", "slip", "loss")
READING = {"hold": "fz_hold_n", "slip": "fz_slip_max_n", "loss": "fz_at_loss_n"}
_LOG_SQRT_2PI = 0.5 * float(np.log(2.0 * np.pi))


def _log_ratio(term: str, fz: float, mass, mu, force: float):
    """The term's residual before the law: log of the reading over its physical prediction."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if term == "hold":
            return np.log(fz / (np.asarray(mass, dtype=float) * G))
        return np.log(fz / (2.0 * np.asarray(mu, dtype=float) * force))


def _law(x: np.ndarray) -> tuple[float, float]:
    b = float(np.median(x))
    return b, float(max(SD_FLOOR, MAD_TO_SD * np.median(np.abs(x - b))))


def _onehot(labels, n: int, classes=OUTCOMES) -> np.ndarray:
    """The labels one-hot over ``classes`` (a label outside them reads all zeros)."""
    labels = [labels] * n if isinstance(labels, str) else list(labels)
    out = np.zeros((n, len(classes)))
    for i, lbl in enumerate(labels):
        if lbl in classes:
            out[i, classes.index(lbl)] = 1.0
    return out


class RotClassifier:
    """P(rot_deg >= ROT_DEG | features, label): gradient boosting with the success model's
    out-of-fold calibration grouped by object. The label is one-hot over ``label_classes`` (the motion
    outcomes, or every outcome when the training labels hold a damage class)."""

    feature_names: tuple[str, ...] = FEATURES_M2           # the class defaults: a classifier saved before the schemas
    label_classes: tuple[str, ...] = MOTION_OUTCOMES

    def __init__(self, clf: Any, calibrator: _Calibrator | None, const: float | None, feature_names=FEATURES_M2,
                 label_classes=MOTION_OUTCOMES):
        self.clf, self.calibrator, self.const = clf, calibrator, const
        self.feature_names, self.label_classes = tuple(feature_names), tuple(label_classes)

    def design(self, X: np.ndarray, labels) -> np.ndarray:
        X = np.asarray(X, dtype=float).reshape(len(X), -1)
        return np.hstack([X, _onehot(labels, len(X), self.label_classes)])

    @classmethod
    def fit(cls, X: np.ndarray, labels, y: np.ndarray, groups: np.ndarray, seed: int = 0,
            feature_names=FEATURES_M2) -> RotClassifier:
        labels = list(labels)
        classes = OUTCOMES if any(lbl in DAMAGE for lbl in labels) else MOTION_OUTCOMES
        y = np.asarray(y, dtype=int)
        if y.min() == y.max():
            return cls(None, None, float(y.mean()), feature_names, classes)
        D = cls(None, None, None, feature_names, classes).design(X, labels)
        kind = "isotonic" if min(int(y.sum()), int((1 - y).sum())) >= MIN_CLASS_ROWS else "sigmoid"
        calibrator = None
        if len(set(groups)) >= N_FOLDS:
            oof = np.zeros(len(y))
            for tr, te in GroupKFold(N_FOLDS).split(D, y, groups):
                if y[tr].min() == y[tr].max():
                    oof[te] = float(y[tr].mean())
                    continue
                oof[te] = _fit_gbm(D[tr], y[tr], seed).predict_proba(D[te])[:, 1]
            calibrator = _Calibrator(kind).fit(oof[:, None], y[:, None].astype(float), ["rot"])
        return cls(_fit_gbm(D, y, seed), calibrator, None, feature_names, classes)

    def predict(self, X: np.ndarray, label: str) -> np.ndarray:
        if self.const is not None:
            return np.full(len(X), self.const)
        raw = self.clf.predict_proba(self.design(X, label))[:, 1]
        if self.calibrator is not None:
            raw = self.calibrator.transform(raw[:, None], ["rot"])[:, 0]
        return np.clip(raw, 0.0, 1.0)


@dataclass
class ReadoutModel:
    terms: dict[str, dict[str, tuple[float, float]]]
    side_error: float
    rot_clf: Any = None
    temper: float = 1.0
    floor_p: float = 1e-4
    stats: dict[str, Any] = field(default_factory=dict)

    def law(self, term: str, label: str) -> tuple[float, float] | None:
        laws = self.terms.get(term)
        if not laws:
            return None
        return laws.get(label) or laws.get(POOLED)

    def components(self, obs: dict[str, Any], preds, X: np.ndarray | None, arrays: dict[str, np.ndarray],
                   sides: np.ndarray | None, terms: list[str] | None = None) -> tuple[dict[str, Any], dict[str, list[str]]]:
        """The untempered terms: discrete log-likelihoods and standardised residuals, plus which terms
        were used and rejected. ``combine`` applies a temper. ``terms`` (an entry's recorded used terms)
        replays exactly those terms, without the rejection rule (a belief refresh)."""
        want = (lambda term: True) if terms is None else (lambda term: term in terms)
        label, r = obs["label"], obs.get("readouts") or {}
        force = float(obs.get("force", np.nan))
        m, mu = np.asarray(arrays["mass"], dtype=float), np.asarray(arrays["mu"], dtype=float)
        cat: dict[str, np.ndarray] = {}
        norm: dict[str, tuple[np.ndarray, float]] = {}
        used: list[str] = []
        rejected: list[str] = []
        if preds is not None and want("outcome"):
            P = preds if isinstance(preds, np.ndarray) else prediction_matrix(preds)
            cat["outcome"] = np.log(np.maximum(self.floor_p, P[:, OUTCOMES.index(label)]))
            used.append("outcome")
        for term in CONTINUOUS:
            fz, law = r.get(READING[term]), self.law(term, label)
            if fz is None or law is None or not want(term):
                continue
            if terms is None and not float(fz) > 0:
                rejected.append(term)
                continue
            b, sd = law
            z = (_log_ratio(term, float(fz), m, mu, force) - b) / sd
            z = np.where(np.isfinite(z), z, np.inf)
            if terms is None and (not np.isfinite(z).any() or float(np.min(np.abs(z))) > REJECT_Z):
                rejected.append(term)
                continue
            norm[term] = (z, float(sd))
            used.append(term)
        if self.rot_clf is not None and X is not None and want("rot"):
            p = self.rot_clf.predict(X, label)
            rotated = float(r.get("rot_deg") or 0.0) >= ROT_DEG
            cat["rot"] = np.log(np.maximum(self.floor_p, p if rotated else 1.0 - p))
            used.append("rot")
        heavy = int(r.get("heavy_side") or 0)
        if sides is not None and heavy != 0 and want("side"):
            s, e = np.asarray(sides), float(self.side_error)
            cat["side"] = np.log(np.where(s == heavy, 1.0 - e, np.where(s == 0, 0.5, e)))
            used.append("side")
        return {"n": len(m), "cat": cat, "norm": norm}, {"used": used, "rejected": rejected}

    def combine(self, comp: dict[str, Any], temper: float | None = None) -> np.ndarray:
        """Per-particle log-likelihood of the components at a temper (default: the model's)."""
        t = float(self.temper if temper is None else temper)
        ll = np.zeros(comp["n"])
        for v in comp["cat"].values():
            ll += v / t
        for z, sd in comp["norm"].values():
            with np.errstate(invalid="ignore", over="ignore"):
                ll += -0.5 * (z / t) ** 2 - np.log(sd * t) - _LOG_SQRT_2PI
        return ll

    def loglik(self, obs: dict[str, Any], preds, X: np.ndarray | None, arrays: dict[str, np.ndarray],
               sides: np.ndarray | None) -> tuple[np.ndarray, dict[str, list[str]]]:
        """Per-particle log-likelihood of one observation ``{"label", "readouts", "force"}`` and
        ``{"used": [...], "rejected": [...]}``. ``preds``: per-particle predictions (an (n, 7) matrix in
        OUTCOMES order or a list of Prediction); ``X``: the particles' feature rows (for rot);
        ``sides``: particle_sides of the particles (for side). A None input skips its term."""
        comp, info = self.components(obs, preds, X, arrays, sides)
        return self.combine(comp), info

    @classmethod
    def fit(cls, rows: list[dict[str, Any]], seed: int = 0, names=FEATURES_M2) -> ReadoutModel:
        """Fit the residual laws, the side error and the rotation classifier (on ``names``) on (training) corpus
        rows with readouts, at each row's true mass, friction and CoM."""
        rows = [r for r in rows if r.get("readouts")]
        terms: dict[str, dict[str, tuple[float, float]]] = {}
        stats: dict[str, Any] = {"n_rows": len(rows), "terms": {}}
        for term in CONTINUOUS:
            xs, labels = [], []
            for r in rows:
                fz = r["readouts"].get(READING[term])
                if fz is None or not float(fz) > 0:
                    continue
                x = float(_log_ratio(term, float(fz), float(r["truth"]["mass"]), float(r["truth"]["mu"]), float(r["force"])))
                if np.isfinite(x):
                    xs.append(x)
                    labels.append(r["label"])
            if not xs:
                continue
            x_arr, lab = np.array(xs), np.array(labels)
            laws = {POOLED: _law(x_arr)}
            for lbl in sorted(set(labels)):
                if int((lab == lbl).sum()) >= MIN_LABEL_ROWS:
                    laws[lbl] = _law(x_arr[lab == lbl])
            terms[term] = laws
            stats["terms"][term] = {"n": len(xs), "by_label": {lbl: int((lab == lbl).sum()) for lbl in sorted(set(labels))}}
        seen, truth = [], []
        for r in rows:
            heavy = int(r["readouts"].get("heavy_side") or 0)
            if heavy == 0 or "pose" not in r or "com_obj_true" not in r:
                continue
            true_side = int(particle_sides({"com_obj": np.array([r["com_obj_true"]], dtype=float)}, r["pose"])[0])
            if true_side != 0:
                seen.append(heavy)
                truth.append(true_side)
        mismatch = float(np.mean(np.array(seen) != np.array(truth))) if seen else None
        side_error = 0.5 if mismatch is None else float(np.clip(mismatch, SIDE_ERROR_FLOOR, 0.5))
        stats["side"] = {"n": len(seen), "mismatch": mismatch}
        rot_rows = [r for r in rows if "pose" in r and "com_obj_true" in r and r["readouts"].get("rot_deg") is not None]
        rot_clf = None
        if rot_rows:
            X = np.array([as_vector(true_features(r), names) for r in rot_rows])
            y = np.array([float(r["readouts"]["rot_deg"]) >= ROT_DEG for r in rot_rows], dtype=int)
            rot_clf = RotClassifier.fit(X, [r["label"] for r in rot_rows], y, np.array([r["object"] for r in rot_rows]), seed,
                                        names)
            stats["rot"] = {"n": len(rot_rows), "rate": float(y.mean())}
        return cls(terms=terms, side_error=side_error, rot_clf=rot_clf, stats=stats)

    def to_json(self) -> dict[str, Any]:
        return {"version": 1, "terms": {t: {g: [float(v[0]), float(v[1])] for g, v in laws.items()} for t, laws in self.terms.items()},
                "side_error": float(self.side_error), "temper": float(self.temper), "floor_p": float(self.floor_p), "stats": self.stats}

    @staticmethod
    def from_json(d: dict[str, Any], rot_clf: Any = None) -> ReadoutModel:
        terms = {t: {g: (float(v[0]), float(v[1])) for g, v in laws.items()} for t, laws in d["terms"].items()}
        return ReadoutModel(terms=terms, side_error=float(d["side_error"]), rot_clf=rot_clf, temper=float(d.get("temper", 1.0)),
                            floor_p=float(d.get("floor_p", 1e-4)), stats=dict(d.get("stats", {})))

    def save(self, json_path: str, joblib_path: str | None = None) -> None:
        """The laws as JSON; the rotation classifier in a joblib file the JSON names (same directory)."""
        d = self.to_json()
        if self.rot_clf is not None and joblib_path:
            joblib.dump(self.rot_clf, joblib_path)
            d["rot_clf"] = os.path.basename(joblib_path)
        with open(json_path, "w") as f:
            json.dump(d, f, indent=1)

    @staticmethod
    def load(json_path: str) -> ReadoutModel:
        d = json.load(open(json_path))
        clf = joblib.load(os.path.join(os.path.dirname(os.path.abspath(json_path)), d["rot_clf"])) if d.get("rot_clf") else None
        return ReadoutModel.from_json(d, clf)
