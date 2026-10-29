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

"""The novelty guard: the skill declines an object unlike the training objects.

The family hold-out study (vlm-grasp-limits/family_study, corpora v7 and v8) compared ten checks, and its rule chose
the nearest-neighbour distance (knn). A row's distance is the mean Euclidean distance to the K nearest training rows,
the features standardized by the training rows (constant features dropped). An object's score is the median of its
rows' distances; at an assessment the rows are the menu's rows for every particle, weighted by the belief. The
threshold is a conformal outlier p-value (Bates et al. 2023) against calibration scores: the training objects in
N_GROUPS seeded groups, each object scored against the rows of the other groups. p = (1 + number of calibration scores
>= s) / (n + 1); the guard flags at p <= ALPHA, and a flag makes the menu decline with a reason. The guard draws no
random numbers.

It sees only what the features show. In the study it caught objects that the model got wrong without committing
(the v7 birdhouse, the v8 spatula); no check caught an object with confident false commits (the v7 storage box).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from rpent.skills.pick_place.features import FEATURE_SCHEMAS, FEATURES_M2, as_vector

K = 10                 # training rows per row distance
ALPHA = 0.05           # the guard flags at p <= ALPHA
N_GROUPS = 11          # calibration groups of training objects (the study's 11 hold-outs)
NAME = "knn"
TIE = 1e-12            # a p-value equal to ALPHA flags; a cumulative weight equal to 0.5 splits the median


def weighted_median(values, weights=None) -> float:
    """The weighted median; equal weights give np.median (the mean of the two middle values for an even count). Values
    without weight take no part."""
    v = np.asarray(values, dtype=float).ravel()
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype=float).ravel()
    v, w = v[w > 0], w[w > 0]
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    c = np.cumsum(w) / w.sum()
    i = int(np.searchsorted(c, 0.5 - TIE))
    if i < len(v) - 1 and abs(c[i] - 0.5) <= TIE:
        return float((v[i] + v[i + 1]) / 2)
    return float(v[i])


def calibration_groups(objects, n_groups: int = N_GROUPS, seed: int = 0) -> dict[str, int]:
    """Object -> group in 0..n_groups-1: a seeded shuffle of the sorted ids, dealt round-robin."""
    order = sorted(set(objects))
    np.random.default_rng(seed).shuffle(order)
    return {o: i % n_groups for i, o in enumerate(order)}


class KnnGuard:
    """The fitted guard: the standardized reference rows, the sorted calibration scores (and their objects), K and
    ALPHA. ``use`` are the kept columns of the n_features input columns."""

    def __init__(self, mean, scale, use, reference, calibration, k: int = K, alpha: float = ALPHA,
                 calibration_objects: Sequence[str] = (), n_features: int | None = None,
                 feature_names: Sequence[str] | None = None):
        self.mean = np.asarray(mean, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.use = np.asarray(use, dtype=int)
        self.reference = np.asarray(reference, dtype=float)
        order = np.argsort(np.asarray(calibration, dtype=float), kind="stable")
        self.calibration = np.asarray(calibration, dtype=float)[order]
        objs = list(calibration_objects)
        self.calibration_objects = [objs[i] for i in order] if objs else []
        self.k = int(k)
        self.alpha = float(alpha)
        self.n_features = int(n_features if n_features is not None else (self.use.max() + 1 if len(self.use) else 0))
        self.feature_names = None if feature_names is None else tuple(feature_names)    # the input columns' names
        self._nn = None

    @property
    def n_calibration(self) -> int:
        return len(self.calibration)

    @classmethod
    def from_reference(cls, X_ref, calibration, k: int = K, alpha: float = ALPHA,
                       calibration_objects: Sequence[str] = (), feature_names: Sequence[str] | None = None) -> "KnnGuard":
        """A guard whose reference rows are X_ref (standardized by their own mean and sd, constant columns dropped: equal
        minimum and maximum, as the sd of a constant 0.1 column is about 1e-17)."""
        X_ref = np.asarray(X_ref, dtype=float)
        use = np.flatnonzero(np.ptp(X_ref, axis=0) > 0)
        mean, scale = X_ref[:, use].mean(0), X_ref[:, use].std(0)
        return cls(mean, scale, use, (X_ref[:, use] - mean) / scale, calibration, k, alpha, calibration_objects, X_ref.shape[1],
                   feature_names)

    @classmethod
    def fit(cls, X, objects, k: int = K, alpha: float = ALPHA, n_groups: int = N_GROUPS, seed: int = 0,
            feature_names: Sequence[str] | None = None) -> "KnnGuard":
        """The guard on the training rows X (one object id per row), calibrated by calibration_scores."""
        cal = calibration_scores(X, objects, k=k, n_groups=n_groups, seed=seed)
        names = sorted(cal)
        return cls.from_reference(X, [cal[o] for o in names], k, alpha, names, feature_names)

    def _neighbours(self):
        if self._nn is None:
            from sklearn.neighbors import NearestNeighbors

            self._nn = NearestNeighbors(n_neighbors=min(self.k, len(self.reference)), algorithm="brute").fit(self.reference)
        return self._nn

    def distances(self, X) -> np.ndarray:
        """Each row's mean distance to its K nearest reference rows (0 when no reference feature varies)."""
        X = np.asarray(X, dtype=float).reshape(-1, self.n_features)
        if len(self.use) == 0:
            return np.zeros(len(X))
        return self._neighbours().kneighbors((X[:, self.use] - self.mean) / self.scale)[0].mean(1)

    def score(self, X, weights=None) -> float:
        """The weighted median of the rows' distances."""
        return weighted_median(self.distances(X), weights)

    def p_value(self, score: float) -> float:
        """(1 + number of calibration scores >= score) / (n + 1)."""
        n_ge = self.n_calibration - int(np.searchsorted(self.calibration, float(score), side="left"))
        return (1 + n_ge) / (self.n_calibration + 1)

    def check(self, X, weights=None) -> dict[str, Any]:
        """The guard's record for rows X: its score, p-value and flag."""
        s = self.score(X, weights)
        p = self.p_value(s)
        return {"guard": NAME, "score": s, "p": p, "flag": bool(p <= self.alpha + TIE), "alpha": self.alpha, "k": self.k,
                "n_calibration": self.n_calibration}

    def save(self, path, features: Sequence[str] | None = None) -> None:
        """Write the guard (``features`` names the input columns, one name per column; by default its own
        feature_names, else FEATURES_M2)."""
        features = features if features is not None else (self.feature_names or FEATURES_M2)
        if len(features) != self.n_features:
            raise ValueError(f"{len(features)} feature names for a guard on {self.n_features} features")
        np.savez_compressed(path, mean=self.mean, scale=self.scale, use=self.use, reference=self.reference,
                            calibration=self.calibration, calibration_objects=np.array(self.calibration_objects, dtype=str),
                            k=self.k, alpha=self.alpha, features=np.array(list(features), dtype=str))

    @classmethod
    def load(cls, path, features: Sequence[str] | None = None) -> "KnnGuard":
        """Read a guard with the feature names it was fit on: those of a feature schema (features.FEATURE_SCHEMAS), or
        with ``features``, exactly those, in order."""
        z = np.load(path, allow_pickle=False)
        names = tuple(z["features"].tolist())
        known = names == tuple(features) if features is not None else names in FEATURE_SCHEMAS.values()
        if not known:
            raise ValueError(f"{path} was fit on other features: {list(names)}")
        return cls(z["mean"], z["scale"], z["use"], z["reference"], z["calibration"], int(z["k"]), float(z["alpha"]),
                   z["calibration_objects"].tolist(), len(names), names)


def object_scores(X_ref, X, objects, k: int = K) -> dict[str, float]:
    """Each object's score against the reference rows X_ref: the median distance of its rows."""
    d = KnnGuard.from_reference(X_ref, calibration=(), k=k).distances(X)
    objects = np.asarray(objects)
    return {o: float(np.median(d[objects == o])) for o in sorted(set(objects.tolist()))}


def calibration_scores(X, objects, k: int = K, n_groups: int = N_GROUPS, seed: int = 0) -> dict[str, float]:
    """Each object's score against the rows of the other calibration groups (calibration_groups)."""
    X, objects = np.asarray(X, dtype=float), np.asarray(objects)
    groups = calibration_groups(objects.tolist(), n_groups, seed)
    gid = np.array([groups[o] for o in objects.tolist()])
    if len(set(gid.tolist())) < 2:
        raise ValueError("the calibration needs at least two groups of objects")
    out: dict[str, float] = {}
    for g in sorted(set(gid.tolist())):
        held = gid == g
        out.update(object_scores(X[~held], X[held], objects[held], k))
    return out


def rows_matrix(rows: list[dict[str, Any]], features: str = "true", names: Sequence[str] = FEATURES_M2) -> np.ndarray:
    """The rows' features as the models see them, in ``names`` order: at the true mass, friction and CoM
    (evaluate.true_features) or the stored nominal ones; a nan reads 0 or its sentinel (as_vector)."""
    from rpent.skills.pick_place.evaluate import true_features

    get = true_features if features == "true" else (lambda r: r["features"])
    return np.array([as_vector(get(r), names) for r in rows], dtype=float).reshape(len(rows), len(names))


def fit_guard(rows: list[dict[str, Any]], features: str = "true", seed: int = 0, k: int = K,
              n_groups: int = N_GROUPS, names: Sequence[str] = FEATURES_M2) -> KnnGuard:
    """The guard on a bundle's training rows (corpus.trainable rows of its training objects), in ``names`` order."""
    return KnnGuard.fit(rows_matrix(rows, features, names), [r["object"] for r in rows], k=k, n_groups=n_groups, seed=seed,
                        feature_names=names)


def flag_reason(rec: dict[str, Any]) -> str:
    """The decline's reason when the guard flags."""
    return (f"the object is unlike the training objects (knn distance {rec['score']:.2f}, p {rec['p']:.3f} <= "
            f"{rec['alpha']:g}): the predictions are not trusted")
