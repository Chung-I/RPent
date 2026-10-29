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

"""Object splits, belief-mode features, and the calibration report.

The deployed skill never sees the true hidden properties: it samples them from
the belief prior. So the acceptance score is computed on predictions made from
belief samples ("belief mode"); the true-θ score ("oracle mode") is an upper bound.
Milestone 2 models are conditional on the true CoM (``true_features``), and belief
mode draws whole particles from a prior function (``prior_fn(pose, n, rng)``).

The interval tools resample objects, the unit of generalisation: ``ece_ci`` (the object-bootstrap
95 % interval of the ECE), ``ci_coverage`` (how often that interval covers a known population ECE
in simulation), ``ece_iid_floor`` (the ECE that sampling noise alone gives at the declared p) and
``per_class`` (one-vs-rest ECE per outcome).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rpent.skills.pick_place.belief import Belief
from rpent.skills.pick_place.features import particle_features
from rpent.skills.pick_place.protocol import MOTION_OUTCOMES, OUTCOMES, Prediction

G = 9.81
MIN_CLASS_ROWS = 30
MIN_CLASS_POS = 50                # per_class reports an ECE only for outcomes with at least this many positives
ECE_BINS = 10
VIS_BINS = (("vis_0.0-0.3", 0.0, 0.3), ("vis_0.3-0.6", 0.3, 0.6), ("vis_0.6-0.9", 0.6, 0.9), ("vis_0.9-1.0", 0.9, 1.0001))


def split_objects(rows: list[dict[str, Any]], seed: int = 0, fractions=(0.7, 0.15, 0.15)) -> dict[str, list[str]]:
    """Train / val / test object ids, disjoint, deterministic for a seed."""
    objects = sorted({r["object"] for r in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(objects)
    n = len(objects)
    n_val = max(1, int(round(fractions[1] * n)))
    n_test = max(1, int(round(fractions[2] * n)))
    n_train = n - n_val - n_test
    return {"train": sorted(objects[:n_train]), "val": sorted(objects[n_train:n_train + n_val]),
            "test": sorted(objects[n_train + n_val:])}


def rows_of(rows: list[dict[str, Any]], object_ids) -> list[dict[str, Any]]:
    ids = set(object_ids)
    return [r for r in rows if r["object"] in ids]


def true_features(row: dict[str, Any]) -> dict[str, float]:
    """The row's features at its true mass, friction and CoM (corpus v2: ``pose`` and ``com_obj_true``)."""
    truth = {"mass": row["truth"]["mass"], "mu": row["truth"]["mu"], "com_obj": row["com_obj_true"]}
    return particle_features(row["features"], truth, row["pose"], row["force"])


def belief_features(row: dict[str, Any], rng: np.random.Generator, n: int = 64, prior_fn=None) -> list[dict[str, float]]:
    """The row's features under n draws of the prior the skill would hold: with ``prior_fn`` whole
    particles (mass, friction, CoM) through the row's pose; without it the milestone 1 prior
    (mass, friction, lever noise)."""
    if prior_fn is not None:
        b = prior_fn(row["pose"], n, rng)
        return [particle_features(row["features"], part, row["pose"], row["force"]) for part in b.particles]
    g = row["geometry"]
    belief = Belief.prior(g["volume"], g["material"], g.get("hollow", False), row["features"].get("visibility", 1.0),
                          length=g.get("length"))
    out = []
    for s in belief.sample(n, rng):
        f = dict(row["features"])
        f["mass"], f["mu"] = s["mass"], s["mu"]
        f["lever"] = abs(row["features"]["lever"] + s["lever_noise"])
        f["load_ratio"] = s["mass"] * G / (2 * s["mu"] * row["force"])
        out.append(f)
    return out


def _brier(P: np.ndarray, Y: np.ndarray) -> float:
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def _bin_index(p: np.ndarray, bins: int = ECE_BINS) -> np.ndarray:
    edges = np.linspace(0, 1, bins + 1)
    return np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)


def _ece(p: np.ndarray, y: np.ndarray, bins: int = ECE_BINS) -> float:
    idx = _bin_index(p, bins)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def _terciles(values: np.ndarray) -> np.ndarray:
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    return np.where(values <= q1, 0, np.where(values <= q2, 1, 2))


def score(preds: list[Prediction], rows: list[dict[str, Any]], train_rows: list[dict[str, Any]],
          const: np.ndarray | None = None) -> dict[str, Any]:
    """Brier, constant-baseline Brier, ECE on P(placed_stable), and the same by subgroup. The constant
    baseline is the training rows' class frequencies, or ``const`` (one row per scored row, OUTCOMES
    order) when the rows come from models trained on different data (the cross-fit's folds)."""
    P = np.array([[p.p.get(k, 0.0) for k in OUTCOMES] for p in preds])
    Y = np.array([[1.0 if r["label"] == k else 0.0 for k in OUTCOMES] for r in rows])
    if const is not None:
        Pc = np.asarray(const, dtype=float).reshape(len(rows), -1)
        if Pc.shape[1] == len(MOTION_OUTCOMES) < len(OUTCOMES):             # a result file from before the damage classes
            Pc = np.hstack([Pc, np.zeros((len(rows), len(OUTCOMES) - len(MOTION_OUTCOMES)))])
    else:
        freq = np.array([np.mean([t["label"] == k for t in train_rows]) if train_rows else 1 / len(OUTCOMES) for k in OUTCOMES])
        Pc = np.tile(freq, (len(rows), 1))
    ps, ys = P[:, 0], Y[:, 0]

    def block(m: np.ndarray) -> dict[str, Any]:
        return {"n": int(m.sum()), "brier": _brier(P[m], Y[m]), "brier_const": _brier(Pc[m], Y[m]), "ece": _ece(ps[m], ys[m])}

    by: dict[str, Any] = {}
    mass = np.array([r["truth"]["mass"] for r in rows])
    mu = np.array([r["truth"]["mu"] for r in rows])
    lever = np.array([r["features"]["lever"] for r in rows])
    for name, vals in (("mass", mass), ("mu", mu), ("lever", lever)):
        t = _terciles(vals)
        for i in range(3):
            by[f"{name}_t{i}"] = block(t == i) if (t == i).any() else "empty"
    dests = np.array([r["destination"] for r in rows])
    for d in sorted(set(dests)):
        by[f"dest_{d}"] = block(dests == d)
    vis = np.array([r["features"].get("visibility", 1.0) for r in rows])
    for name, lo, hi in VIS_BINS:
        m = (vis >= lo) & (vis < hi)
        by[name] = block(m) if m.any() else "untested"
    n_train = {k: int(sum(t["label"] == k for t in train_rows)) for k in OUTCOMES}
    return {
        "n": len(rows),
        "brier": _brier(P, Y),
        "brier_const": _brier(Pc, Y),
        "ece": _ece(ps, ys),
        "by_subgroup": by,
        "n_per_class": {k: int(sum(r["label"] == k for r in rows)) for k in OUTCOMES},
        "n_train_per_class": n_train,
        "unsupported_classes": [k for k, v in n_train.items() if v < MIN_CLASS_ROWS],
    }


def per_class(P: np.ndarray, labels, min_pos: int = MIN_CLASS_POS) -> dict[str, dict[str, Any]]:
    """One-vs-rest ECE of each outcome (P's columns in OUTCOMES order) with its count of positives;
    "insufficient" in place of the ECE under ``min_pos`` positives."""
    P = np.asarray(P, dtype=float).reshape(len(labels), len(OUTCOMES))
    labels = np.asarray(labels)
    out: dict[str, dict[str, Any]] = {}
    for j, k in enumerate(OUTCOMES):
        y = (labels == k).astype(float)
        n_pos = int(y.sum())
        out[k] = {"n_pos": n_pos, "ece": _ece(P[:, j], y) if n_pos >= min_pos else "insufficient"}
    return out


def ece_iid_floor(p, rng: np.random.Generator, n_sim: int = 200) -> float:
    """The mean ECE of ``n_sim`` outcome draws y ~ Bernoulli(p) at the declared p: what sampling noise
    alone gives for calibrated predictions of this design (a diagnostic, not a bound)."""
    p = np.asarray(p, dtype=float)
    if not len(p):
        return float("nan")
    onehot = np.eye(ECE_BINS)[_bin_index(p)]                       # n x bins
    sp = p @ onehot
    Y = (rng.random((n_sim, len(p))) < p).astype(float)
    return float(np.mean(np.abs(sp - Y @ onehot).sum(axis=1) / len(p)))


def object_sums(p: np.ndarray, y: np.ndarray, inv: np.ndarray, n_objects: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per object (integer index ``inv``) and ECE bin: the sums of p and y, and the case counts per object."""
    p, y, inv = np.asarray(p, dtype=float), np.asarray(y, dtype=float), np.asarray(inv, dtype=int)
    b = _bin_index(p)
    SP, SY = np.zeros((n_objects, ECE_BINS)), np.zeros((n_objects, ECE_BINS))
    np.add.at(SP, (inv, b), p)
    np.add.at(SY, (inv, b), y)
    return SP, SY, np.bincount(inv, minlength=n_objects).astype(float)


def object_weights(n_objects: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap multiplicities: n_boot rows of how often each object is drawn (n_objects draws with replacement)."""
    draws = rng.integers(0, n_objects, size=(n_boot, n_objects))
    W = np.zeros((n_boot, n_objects))
    np.add.at(W, (np.repeat(np.arange(n_boot), n_objects), draws.ravel()), 1.0)
    return W


def ece_from_sums(W: np.ndarray, SP: np.ndarray, SY: np.ndarray, N: np.ndarray) -> np.ndarray:
    """The ECE of each weighted resample (rows of W) from the per-object bin sums."""
    return np.abs(W @ SP - W @ SY).sum(axis=1) / np.maximum(W @ N, 1e-12)


def ece_ci(p, y, objects, rng: np.random.Generator, n_boot: int = 300) -> list[float]:
    """95 % percentile interval of the ECE on the declared p by resampling objects with replacement."""
    _, inv = np.unique(np.asarray(objects), return_inverse=True)
    SP, SY, N = object_sums(p, y, inv, int(inv.max()) + 1)
    e = ece_from_sums(object_weights(len(N), n_boot, rng), SP, SY, N)
    return [float(np.percentile(e, 2.5)), float(np.percentile(e, 97.5))]


def _shift_to(p: np.ndarray, target: float, sign: float) -> tuple[np.ndarray, float]:
    """q = sigmoid(logit p + c) with c of the given sign and mean |q - p| = target (bisection). A
    one-signed shift moves every bin the same way, so the binned population ECE equals mean |q - p|."""
    lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1.0))
    lo, hi = 0.0, 20.0
    for _ in range(100):
        c = 0.5 * (lo + hi)
        q = 1.0 / (1.0 + np.exp(-(lp + sign * c)))
        lo, hi = (c, hi) if np.mean(np.abs(q - p)) < target else (lo, c)
    c = sign * 0.5 * (lo + hi)
    return 1.0 / (1.0 + np.exp(-(lp + c))), float(c)


def _draw(q0: np.ndarray, inv: np.ndarray, a: float, rng: np.random.Generator, n_sim: int) -> np.ndarray:
    """n_sim outcome vectors: per object an effect u ~ U(-a, a) moves q by u * min(q, 1 - q), which keeps
    q in [0, 1] and its mean over objects at q0 (the population map stays the known one)."""
    u = rng.uniform(-a, a, size=(n_sim, int(inv.max()) + 1))[:, inv]
    q = q0 + u * np.minimum(q0, 1.0 - q0)
    return (rng.random(q.shape) < q).astype(float)


def _between_var(p: np.ndarray, Y: np.ndarray, inv: np.ndarray) -> np.ndarray:
    """Variance across objects of the per-object mean residual y - p (one value per row of Y)."""
    N = np.bincount(inv).astype(float)
    R = np.zeros((Y.shape[0], len(N)))
    np.add.at(R.T, inv, (Y - p).T)
    return np.var(R / N, axis=1, ddof=1)


def coverage_design(p, objects, ece_target: float, n_sim: int = 200, rng: np.random.Generator | None = None, y=None,
                    n_boot: int = 200) -> dict[str, Any]:
    """Simulated coverage of ``ece_ci`` at the design (p, objects): outcomes from a known map of p with
    population ECE ``ece_target`` (a logit shift, toward the observed residual's sign when ``y`` is
    given, else down: overconfidence), with per-object effects whose size matches the observed
    between-object variance of the residuals y - p (none without ``y``)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    p = np.asarray(p, dtype=float)
    _, inv = np.unique(np.asarray(objects), return_inverse=True)
    sign = -1.0 if y is None or float(np.mean(np.asarray(y, dtype=float) - p)) < 0 else 1.0
    q0, c = _shift_to(p, ece_target, sign)
    a, observed = 0.0, None
    if y is not None and int(inv.max()) >= 1:                      # a between-object variance needs two objects
        observed = float(_between_var(p, np.asarray(y, dtype=float)[None, :], inv)[0])
        grid = np.linspace(0.0, 1.0, 21)
        sims = [float(np.mean(_between_var(p, _draw(q0, inv, g, rng, 40), inv))) for g in grid]
        a = float(grid[int(np.argmin([abs(v - observed) for v in sims]))])
    Y = _draw(q0, inv, a, rng, n_sim)
    covered, widths = [], []
    for ys in Y:
        lo, hi = ece_ci(p, ys, objects, rng, n_boot)
        covered.append(lo <= ece_target <= hi)
        widths.append(hi - lo)
    return {"coverage": float(np.mean(covered)), "n_sim": int(n_sim), "n_boot": int(n_boot), "ece_target": float(ece_target),
            "population_ece": float(np.mean(np.abs(q0 - p))), "logit_shift": c, "object_effect": a,
            "between_object_var_observed": observed, "mean_width": float(np.mean(widths)), "n": int(len(p)),
            "n_objects": int(inv.max()) + 1}


def ci_coverage(p, objects, ece_target: float, n_sim: int = 200, rng: np.random.Generator | None = None, y=None,
                n_boot: int = 200) -> float:
    """The simulated coverage of the object-bootstrap ECE interval (see coverage_design)."""
    return coverage_design(p, objects, ece_target, n_sim, rng, y, n_boot)["coverage"]
