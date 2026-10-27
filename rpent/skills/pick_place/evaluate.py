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
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rpent.skills.pick_place.belief import Belief
from rpent.skills.pick_place.protocol import OUTCOMES, Prediction

G = 9.81
MIN_CLASS_ROWS = 30
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


def belief_features(row: dict[str, Any], rng: np.random.Generator, n: int = 64) -> list[dict[str, float]]:
    """The row's features with mass, friction and lever drawn from the prior the skill would hold."""
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


def _ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def _terciles(values: np.ndarray) -> np.ndarray:
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    return np.where(values <= q1, 0, np.where(values <= q2, 1, 2))


def score(preds: list[Prediction], rows: list[dict[str, Any]], train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Brier, constant-baseline Brier, ECE on P(placed_stable), and the same by subgroup."""
    P = np.array([[p.p[k] for k in OUTCOMES] for p in preds])
    Y = np.array([[1.0 if r["label"] == k else 0.0 for k in OUTCOMES] for r in rows])
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
