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
* squeeze (milestone 2b): close at the proposal and ramp each pad to the cap ``squeeze_cap``, no lift; the
  reading is the stiffness (``measured``, reading ``k_hat``, a law on log(k_hat / k_n_per_mm)).

Every probe can damage the object (kind ``damaged``, milestone 2b): its kind classifier learns the kind where the
probe corpus has it, and ``terminal_risk`` (thrown or damaged) is the chance the probe ends the case.

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

from rpent.skills.pick_place.evaluate import MIN_CLASS_ROWS, row_pose
from rpent.skills.pick_place.features import FEATURES_M2, FRAG_KEYS, G, particle_matrix
from rpent.skills.pick_place.likelihood import Law, as_law, fit_law
from rpent.skills.pick_place.success_model import DAMAGE_FLAG, N_FOLDS, _Calibrator, _fit_gbm

PROBE_SECONDS = {"weigh": 4.0, "push": 2.0, "squeeze": 3.0}
PROBE_KINDS = {"weigh": ("measured", "lower_bound", "lost"), "push": ("slid", "tipped", "no_breakaway", "no_contact"),
               "squeeze": ("measured",)}
DAMAGED = "damaged"                          # a probe's damage kind (milestone 2b): learned where the probe corpus has it
READING = {"measured": "m_hat", "lower_bound": "mu_hat", "slid": "f_break"}     # the reading each kind's law is on
SQUEEZE_READING = "k_hat"                    # the squeeze's measured reading: the stiffness, not the weigh's mass


def weigh_force(f_std: float) -> float:
    """The weigh probe's squeeze per pad: twice the standard, at least 10 N more, at most 70 N."""
    return float(min(70.0, max(2.0 * f_std, f_std + 10.0)))


def squeeze_cap(belief: Any, f_std: float) -> float:
    """The squeeze probe's largest force per pad: half the standard squeeze, or less, the weighted 10 % quantile of the
    particles' damage force (``f_damage_n``; particles without one do not count). So the probe crushes under the
    belief's own law with probability 0.1 at most."""
    v = np.array([np.nan if p.get("f_damage_n") is None else float(p["f_damage_n"]) for p in belief.particles], dtype=float)
    ok = np.isfinite(v)
    cap = 0.5 * float(f_std)
    if not ok.any():
        return cap
    w = np.asarray(belief.weights, dtype=float)[ok]
    vals = v[ok]
    order = np.argsort(vals, kind="stable")
    cw = np.cumsum(w[order]) / float(w.sum())
    q10 = float(vals[order][min(int(np.searchsorted(cw, 0.1 - 1e-12, side="left")), len(vals) - 1)])
    return float(min(cap, q10))


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
    meter: dict[str, Any] = field(default_factory=dict)      # a substrate's damage meter (mujoco_lab: meter.py) and its
    damage: dict[str, Any] | None = None                      # first threshold crossing; kind "damaged" when there is one

    def to_data(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("trace")
        return d


def reading_key(probe: str | None, kind: str) -> str | None:
    """The reading key of a probe's kind: k_hat for the squeeze's measured kind, else READING's."""
    return SQUEEZE_READING if probe == "squeeze" and kind == "measured" else READING.get(kind)


def log_ratio(kind: str, reading: dict[str, float], mass, mu, probe: str | None = None, k=None) -> np.ndarray | None:
    """The kind's reading over its physical prediction, in log: log(m_hat/m), log(mu_hat/mu),
    log(f_break/(mu m g)), and for the squeeze log(k_hat/k) with ``k`` the stiffness; None when the kind has no
    reading, the reading is absent, or the squeeze has no stiffness to compare with."""
    key = reading_key(probe, kind)
    if key is None or reading.get(key) is None:
        return None
    v = float(reading[key])
    with np.errstate(divide="ignore", invalid="ignore"):
        if probe == "squeeze":
            return None if k is None else np.log(v / np.asarray(k, dtype=float))
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
        """P(kind) per particle row of X, columns in ``classes`` order. The probe cannot damage an object whose public
        entry has every damage flag off (a model on the four flags: no crush, bruise or break threshold, not
        fillable), so that row's "damaged" column is 0 and the row renormalised, as the success model masks."""
        P = self.clf.predict(X)
        names = list(self.feature_names)
        flags = [names.index(f) for f in DAMAGE_FLAG.values() if f in names]
        if DAMAGED not in self.classes or len(flags) < len(DAMAGE_FLAG):
            return P
        safe = np.all(np.asarray(X, dtype=float).reshape(len(P), -1)[:, flags] <= 0.0, axis=1)
        if not safe.any():
            return P
        j = self.classes.index(DAMAGED)
        P = np.array(P, dtype=float)
        P[safe, j] = 0.0
        s = P[safe].sum(axis=1, keepdims=True)
        rest = np.where(np.arange(len(self.classes)) == j, 0.0, 1.0 / (len(self.classes) - 1))
        P[safe] = np.where(s > 0, P[safe] / np.where(s > 0, s, 1.0), rest)
        return P

    def risk(self, X: np.ndarray) -> np.ndarray:
        """P(the probe throws the object: off the table or more than 10 cm away) per row."""
        X = np.asarray(X, dtype=float).reshape(len(X), -1)
        if isinstance(self.p_thrown, (int, float)):
            return np.full(len(X), float(self.p_thrown))
        return self.p_thrown.predict(X)[:, 1]

    def terminal_risk(self, X: np.ndarray) -> np.ndarray:
        """P(the probe ends the case) per row: P(thrown) + (1 - P(thrown)) P(damaged), the kind model being that of
        the results not thrown, so a throw that damages counts once (P(damaged) is 0 for a model whose corpus had no
        damaged rows)."""
        r = self.risk(X)
        if DAMAGED in self.classes:
            r = r + (1.0 - r) * self.p_kind(X)[:, self.classes.index(DAMAGED)]
        return np.minimum(r, 1.0)

    def loglik(self, obs: dict[str, Any], X: np.ndarray, arrays: dict[str, np.ndarray], force: float,
               temper: float | None = None, terms: list[str] | None = None,
               P: np.ndarray | None = None, thrown: bool | None = None) -> tuple[np.ndarray, dict[str, list[str]]]:
        """Per-particle log-likelihood of one probe result ``{"kind", "reading"}``: the kind term
        log max(floor_p, P(kind)) / temper, and the reading term log p(ratio), p the kind's reading law with
        every sd times temper, rejected (listed, not used) when the reading is not positive or its best particle
        lies 5 sd from the centre of every component of the law. A kind the model never saw (a damaged result from a
        model whose corpus had none) tells nothing: its kind term is rejected.
        ``temper`` overrides the model's (the posterior check's grid). ``terms`` (an entry's recorded
        used terms) replays exactly those terms, without the rejection rule (a belief refresh). ``P``:
        ``p_kind(X)`` when the caller has it already (the preposterior loop). ``thrown``: whether a real result threw the
        object. The kinds are those of the results not thrown, so a result that kept the object (False) adds the term
        log max(floor_p, 1 - P(thrown)) / temper ("not_thrown"). None (the preposterior, which weighs each particle's
        survival itself) adds nothing, and so does a throw, which ends the case."""
        t = self.temper if temper is None else float(temper)
        want = (lambda term: True) if terms is None else (lambda term: term in terms)
        kind = obs["kind"]
        used, rejected = [], []
        ll = np.zeros(np.asarray(X).shape[0])
        if want("kind") and kind not in self.classes:
            rejected.append("kind")
        elif want("kind"):
            P = self.p_kind(X) if P is None else np.asarray(P, dtype=float)
            ll = np.log(np.maximum(self.floor_p, P[:, self.classes.index(kind)])) / t
            used.append("kind")
        law = as_law(self.reading.get(kind))
        key = reading_key(self.probe, kind)
        value = (obs.get("reading") or {}).get(key) if key else None
        if law is not None and value is not None and want("reading"):
            if terms is None and not float(value) > 0:
                rejected.append("reading")
            else:
                x = log_ratio(kind, obs["reading"], arrays["mass"], arrays["mu"], self.probe, arrays.get("k_n_per_mm"))
                if terms is None and law.outside(x).all():
                    rejected.append("reading")
                else:
                    ll = ll + law.logpdf(x, t)
                    used.append("reading")
        if thrown is not None and not thrown and want("not_thrown"):
            ll = ll + np.log(np.maximum(self.floor_p, 1.0 - self.risk(X))) / t
            used.append("not_thrown")
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
            if self.probe == "squeeze":
                reading = {} if particle.get("k_n_per_mm") is None else {SQUEEZE_READING: float(particle["k_n_per_mm"]) * e}
            elif kind == "measured":
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


def truth_arrays(row: dict[str, Any]) -> dict[str, np.ndarray]:
    """The row's truth as a one-particle belief's arrays: mass, friction, CoM and, for a milestone 2b row, the
    fragility (NaN for a threshold the class lacks), the fill and the spill angle."""
    a = {"mass": np.array([float(row["truth"]["mass"])]), "mu": np.array([float(row["truth"]["mu"])]),
         "com_obj": np.array([row["com_obj_true"]], dtype=float)}
    if row.get("theta_frag") is not None:
        a.update({k: np.array([np.nan if row["theta_frag"].get(k) is None else float(row["theta_frag"][k])]) for k in FRAG_KEYS})
    if "fill" in row:
        a["fill"] = np.array([float(row.get("fill") or 0.0)])
        a["spill_angle_deg"] = np.array([np.nan if row.get("spill_angle_deg") is None else float(row["spill_angle_deg"])])
    return a


def true_matrix(rows: list[dict[str, Any]], names=FEATURES_M2) -> np.ndarray:
    """Probe features of each row at its truth: particle_matrix of the stored base, the pose with the row's public
    entry (row_pose) and the force, in ``names`` order."""
    return np.array([particle_matrix(r["base"], truth_arrays(r), row_pose(r), float(r["force"]), names)[0] for r in rows])


def fit_probe_model(rows: list[dict[str, Any]], probe: str, seed: int = 0, names=FEATURES_M2, val_objects=None) -> ProbeModel:
    """The kind classifier (with "damaged" when a row has it), the reading law per kind and the thrown model of one
    probe. The kinds and the readings are those of the results not thrown (P(kind | not thrown); terminal_risk adds
    the throw once); the thrown model reads every row. The rows of ``val_objects`` are the held-out readings that
    choose each reading law's kind (fit_law); every row enters the kept law."""
    every = rows
    rows = [r for r in every if not r["thrown"]]
    classes = PROBE_KINDS[probe] + ((DAMAGED,) if any(r["kind"] == DAMAGED for r in rows) else ())
    val = set(val_objects or ())
    X = true_matrix(rows, names)
    y = np.array([r["kind"] for r in rows])
    groups = np.array([r["object"] for r in rows])
    clf = CalibratedClassifier.fit(X, y, groups, classes, seed)
    reading, n_reading, rate = {}, {}, {}
    for kind in classes:
        of = [r for r in rows if r["kind"] == kind]
        pairs = [(log_ratio(kind, r["reading"], r["truth"]["mass"], r["truth"]["mu"], probe, (r.get("theta_frag") or {}).get("k_n_per_mm")),
                  r["object"] in val) for r in of]
        pairs = [(float(v), h) for v, h in pairs if v is not None and np.isfinite(v)]
        xs, held = np.array([v for v, _ in pairs]), np.array([h for _, h in pairs], dtype=bool)
        n_reading[kind] = int(len(xs))
        if len(xs):
            reading[kind] = fit_law(xs[~held], xs[held], seed)
            rate[kind] = len(xs) / len(of)
    thrown = np.array([bool(r["thrown"]) for r in every])
    if int(thrown.sum()) < MIN_CLASS_ROWS:
        p_thrown: Any = float(thrown.mean()) if len(thrown) else 0.0
    else:
        p_thrown = CalibratedClassifier.fit(true_matrix(every, names), np.where(thrown, "thrown", "kept"),
                                            np.array([r["object"] for r in every]), ("kept", "thrown"), seed)
    stats = {"n": len(every), "n_not_thrown": len(rows), "kinds": {k: int((y == k).sum()) for k in classes}, "n_reading": n_reading,
             "reading_rate": rate, "thrown": int(thrown.sum())}
    return ProbeModel(probe=probe, classes=classes, clf=clf, reading=reading, p_thrown=p_thrown, stats=stats, reading_rate=rate,
                      feature_names=tuple(names))


def fit_probe_models(rows: list[dict[str, Any]], seed: int = 0, names=FEATURES_M2, val_objects=None) -> dict[str, ProbeModel]:
    """One ProbeModel per probe present in the (training) probe-corpus rows, on ``names``; ``val_objects`` as in
    fit_probe_model."""
    return {probe: fit_probe_model([r for r in rows if r["probe"] == probe], probe, seed, names, val_objects)
            for probe in PROBE_KINDS if any(r["probe"] == probe for r in rows)}
