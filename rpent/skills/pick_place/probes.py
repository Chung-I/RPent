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


"""Probes: short actions that reveal a hidden property at a small cost, and the models that say
what each probe will show under each belief particle.

* weigh: close with a firm squeeze, lift 2 cm slowly and hold; while the object is risen the pads'
  net vertical force is its weight (kind ``measured``, reading ``m_hat``). An object that never rises
  tells the friction from the largest vertical pad force while the pads slide up it (``lower_bound``:
  ``mu_hat`` = fz / 2F and ``m_lower`` = fz / g; no reading when they never slide up a still object);
  one that rises and slips out is ``lost``.
* push: slide the object with the open hand at 3 cm/s; at breakaway the push force is mu m g
  (``slid``, reading ``f_break``); the object may tip instead, not break away, or move without a
  hand force (``no_contact``).

A ProbeModel gives, per particle, the probability of each kind (gradient boosting with out-of-fold
calibration by object), the law of the reading's log ratio to its physics (per kind, fitted at the truth on
training rows: a normal law, or a narrow core with a wide tail when that predicts held-out objects better;
likelihood.fit_law), and the risk that the probe throws the object.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

from rpent.skills.pick_place.evaluate import MIN_CLASS_ROWS
from rpent.skills.pick_place.features import FEATURES_M2, G, particle_matrix
from rpent.skills.pick_place.likelihood import Law, as_law, fit_law
from rpent.skills.pick_place.success_model import N_FOLDS, _Calibrator, _fit_gbm

PROBE_SECONDS = {"weigh": 4.0, "push": 2.0}
PROBE_KINDS = {"weigh": ("measured", "lower_bound", "lost"), "push": ("slid", "tipped", "no_breakaway", "no_contact")}
READING = {"measured": "m_hat", "lower_bound": "mu_hat", "slid": "f_break"}     # the reading each kind's law is on


def weigh_force(f_std: float) -> float:
    """The weigh probe's squeeze per pad: twice the standard, at least 10 N more, at most 70 N."""
    return float(min(70.0, max(2.0 * f_std, f_std + 10.0)))


@dataclass
class ProbeResult:
    probe: str
    kind: str
    reading: dict[str, float]
    displacement_m: float
    tilt_deg: float
    thrown: bool
    seconds: float
    trace: list[dict[str, Any]] = field(default_factory=list, repr=False)
    rev_steps: int = 0              # a substrate's contact monitor (mujoco_lab: adapter.artifact): artifact is True
    deepest_mm: float = 0.0         # when the run's contacts were not physical; a substrate without one leaves the
    artifact: bool = False          # defaults

    def to_data(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("trace")
        return d


def log_ratio(kind: str, reading: dict[str, float], mass, mu) -> np.ndarray | None:
    """The kind's reading over its physical prediction, in log: log(m_hat/m), log(mu_hat/mu),
    log(f_break/(mu m g)); None when the kind has no reading or the reading is absent."""
    key = READING.get(kind)
    if key is None or reading.get(key) is None:
        return None
    v = float(reading[key])
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "measured":
            return np.log(v / np.asarray(mass, dtype=float))
        if kind == "lower_bound":
            return np.log(v / np.asarray(mu, dtype=float))
        return np.log(v / (np.asarray(mu, dtype=float) * np.asarray(mass, dtype=float) * G))


class CalibratedClassifier:
    """Gradient boosting over fixed classes with per-class out-of-fold calibration grouped by object
    (isotonic when every present class has 30 rows, else sigmoid); rows renormalised to sum 1."""

    def __init__(self, classes: tuple[str, ...], clf: Any, calibrator: _Calibrator | None, const: np.ndarray | None):
        self.classes, self.clf, self.calibrator, self.const = tuple(classes), clf, calibrator, const

    @classmethod
    def fit(cls, X: np.ndarray, y, groups, classes: tuple[str, ...], seed: int = 0) -> CalibratedClassifier:
        y = np.asarray(y)
        present = sorted(set(y.tolist()))
        if len(present) == 1:
            return cls(classes, None, None, np.array([1.0 if c == present[0] else 0.0 for c in classes]))
        kind = "isotonic" if min(int((y == c).sum()) for c in present) >= MIN_CLASS_ROWS else "sigmoid"
        calibrator = None
        if len(set(np.asarray(groups).tolist())) >= N_FOLDS:
            oof = np.zeros((len(y), len(classes)))
            for tr, te in GroupKFold(N_FOLDS).split(X, y, groups):
                if len(set(y[tr].tolist())) == 1:
                    oof[te, classes.index(y[tr][0])] = 1.0
                    continue
                oof[te] = cls._raw(_fit_gbm(X[tr], y[tr], seed), X[te], classes)
            Y = np.array([[1.0 if lbl == c else 0.0 for c in classes] for lbl in y])
            calibrator = _Calibrator(kind).fit(oof, Y, list(classes))
        return cls(classes, _fit_gbm(X, y, seed), calibrator, None)

    @staticmethod
    def _raw(clf: Any, X: np.ndarray, classes: tuple[str, ...]) -> np.ndarray:
        out = np.zeros((len(X), len(classes)))
        P = clf.predict_proba(X)
        for j, c in enumerate(clf.classes_):
            out[:, classes.index(c)] = P[:, j]
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float).reshape(len(X), -1)
        if self.const is not None:
            return np.tile(self.const, (len(X), 1))
        P = self._raw(self.clf, X, self.classes)
        if self.calibrator is not None:
            P = self.calibrator.transform(P, list(self.classes))
        P = np.clip(P, 0.0, 1.0)
        s = P.sum(axis=1, keepdims=True)
        return np.where(s > 0, P / np.where(s > 0, s, 1.0), 1.0 / len(self.classes))


@dataclass
class ProbeModel:
    probe: str
    classes: tuple[str, ...]
    clf: Any
    reading: dict[str, Law]                                 # a model pickled before the mixture laws holds (b, sd) pairs
    p_thrown: Any
    temper: float = 1.0
    floor_p: float = 1e-4
    stats: dict[str, Any] = field(default_factory=dict)
    reading_rate: dict[str, float] = field(default_factory=dict)
    feature_names: tuple[str, ...] = FEATURES_M2           # the class default: a model saved before the schemas

    def p_kind(self, X: np.ndarray) -> np.ndarray:
        """P(kind) per particle row of X, columns in ``classes`` order."""
        return self.clf.predict(X)

    def risk(self, X: np.ndarray) -> np.ndarray:
        """P(the probe throws the object: off the table or more than 10 cm away) per row."""
        X = np.asarray(X, dtype=float).reshape(len(X), -1)
        if isinstance(self.p_thrown, (int, float)):
            return np.full(len(X), float(self.p_thrown))
        return self.p_thrown.predict(X)[:, 1]

    def loglik(self, obs: dict[str, Any], X: np.ndarray, arrays: dict[str, np.ndarray], force: float,
               temper: float | None = None, terms: list[str] | None = None,
               P: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, list[str]]]:
        """Per-particle log-likelihood of one probe result ``{"kind", "reading"}``: the kind term
        log max(floor_p, P(kind)) / temper, and the reading term log p(ratio), p the kind's reading law with
        every sd times temper, rejected (listed, not used) when the reading is not positive or its best particle
        lies 5 sd from the centre of every component of the law.
        ``temper`` overrides the model's (the posterior check's grid). ``terms`` (an entry's recorded
        used terms) replays exactly those terms, without the rejection rule (a belief refresh). ``P``:
        ``p_kind(X)`` when the caller has it already (the preposterior loop)."""
        t = self.temper if temper is None else float(temper)
        want = (lambda term: True) if terms is None else (lambda term: term in terms)
        kind = obs["kind"]
        used, rejected = [], []
        ll = np.zeros(np.asarray(X).shape[0])
        if want("kind"):
            P = self.p_kind(X) if P is None else np.asarray(P, dtype=float)
            ll = np.log(np.maximum(self.floor_p, P[:, self.classes.index(kind)])) / t
            used.append("kind")
        law = as_law(self.reading.get(kind))
        key = READING.get(kind)
        value = (obs.get("reading") or {}).get(key) if key else None
        if law is not None and value is not None and want("reading"):
            if terms is None and not float(value) > 0:
                rejected.append("reading")
            else:
                x = log_ratio(kind, obs["reading"], arrays["mass"], arrays["mu"])
                if terms is None and law.outside(x).all():
                    rejected.append("reading")
                else:
                    ll = ll + law.logpdf(x, t)
                    used.append("reading")
        return ll, {"used": used, "rejected": rejected}

    def sample(self, x_row: np.ndarray, particle: dict[str, Any], force: float, rng: np.random.Generator,
               p: np.ndarray | None = None) -> dict[str, Any]:
        """One simulated result of the probe for one particle (its feature row and properties); ``p``:
        the row's P(kind) when the caller has it already. A kind's reading is present at the rate the
        training rows had a usable one (``reading_rate``)."""
        x = np.asarray(x_row, dtype=float).reshape(1, -1)
        p = self.p_kind(x)[0] if p is None else np.asarray(p, dtype=float)
        kind = self.classes[int(rng.choice(len(self.classes), p=p / p.sum()))]
        reading: dict[str, float] = {}
        law = as_law(self.reading.get(kind))
        if law is not None and rng.random() < float(getattr(self, "reading_rate", {}).get(kind, 1.0)):
            e = float(np.exp(law.sample(rng)))
            m, mu = float(particle["mass"]), float(particle["mu"])
            if kind == "measured":
                reading = {"m_hat": m * e}
            elif kind == "lower_bound":
                reading = {"mu_hat": mu * e, "m_lower": 2.0 * mu * e * float(force) / G}
            else:
                reading = {"f_break": mu * m * G * e}
        return {"probe": self.probe, "kind": kind, "reading": reading, "thrown": bool(rng.random() < float(self.risk(x)[0]))}

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> ProbeModel:
        return joblib.load(path)


def true_matrix(rows: list[dict[str, Any]], names=FEATURES_M2) -> np.ndarray:
    """Probe features of each row at its truth: particle_matrix of the stored base, pose and force, in ``names`` order."""
    out = []
    for r in rows:
        a = {"mass": np.array([float(r["truth"]["mass"])]), "mu": np.array([float(r["truth"]["mu"])]),
             "com_obj": np.array([r["com_obj_true"]], dtype=float)}
        out.append(particle_matrix(r["base"], a, r["pose"], float(r["force"]), names)[0])
    return np.array(out)


def fit_probe_model(rows: list[dict[str, Any]], probe: str, seed: int = 0, names=FEATURES_M2, val_objects=None) -> ProbeModel:
    """The kind classifier, the reading law per kind and the thrown model of one probe. The rows of ``val_objects``
    are the held-out readings that choose each reading law's kind (fit_law); every row enters the kept law."""
    classes = PROBE_KINDS[probe]
    val = set(val_objects or ())
    X = true_matrix(rows, names)
    y = np.array([r["kind"] for r in rows])
    groups = np.array([r["object"] for r in rows])
    clf = CalibratedClassifier.fit(X, y, groups, classes, seed)
    reading, n_reading, rate = {}, {}, {}
    for kind in classes:
        of = [r for r in rows if r["kind"] == kind]
        pairs = [(log_ratio(kind, r["reading"], r["truth"]["mass"], r["truth"]["mu"]), r["object"] in val) for r in of]
        pairs = [(float(v), h) for v, h in pairs if v is not None and np.isfinite(v)]
        xs, held = np.array([v for v, _ in pairs]), np.array([h for _, h in pairs], dtype=bool)
        n_reading[kind] = int(len(xs))
        if len(xs):
            reading[kind] = fit_law(xs[~held], xs[held], seed)
            rate[kind] = len(xs) / len(of)
    thrown = np.array([bool(r["thrown"]) for r in rows])
    if int(thrown.sum()) < MIN_CLASS_ROWS:
        p_thrown: Any = float(thrown.mean()) if len(thrown) else 0.0
    else:
        p_thrown = CalibratedClassifier.fit(X, np.where(thrown, "thrown", "kept"), groups, ("kept", "thrown"), seed)
    stats = {"n": len(rows), "kinds": {k: int((y == k).sum()) for k in classes}, "n_reading": n_reading,
             "reading_rate": rate, "thrown": int(thrown.sum())}
    return ProbeModel(probe=probe, classes=classes, clf=clf, reading=reading, p_thrown=p_thrown, stats=stats, reading_rate=rate,
                      feature_names=tuple(names))


def fit_probe_models(rows: list[dict[str, Any]], seed: int = 0, names=FEATURES_M2, val_objects=None) -> dict[str, ProbeModel]:
    """One ProbeModel per probe present in the (training) probe-corpus rows, on ``names``; ``val_objects`` as in
    fit_probe_model."""
    return {probe: fit_probe_model([r for r in rows if r["probe"] == probe], probe, seed, names, val_objects)
            for probe in PROBE_KINDS if any(r["probe"] == probe for r in rows)}
