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

"""Measurable features of one grasp-and-place action, for the success and hazard models.

Ported from the experiment's ``t3g_features.features_for`` (pinch width, contact-face
slope, CoM lever and height, load over pad capacity) and extended with the object's
stability at rest and the destination's properties. Substrate independent: the
caller supplies hull face centres and normals, and each convex hull's vertices, in the
world frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from rpent.skills.pick_place.destinations import DESTINATIONS, edge_distance, has_surface
from rpent.skills.pick_place.destinations import release_height as default_release_height

G = 9.81
PAD_HALF = (0.004, 0.00875, 0.012)   # pad half-sizes: thickness, width, height (the lab gripper)
NO_OPENING_RATIO = 10.0              # "opening over footprint" for places without an opening

FEATURES_M2: tuple[str, ...] = (
    "width", "slope_deg", "lever", "com_h", "mass", "mu", "load_ratio", "v_lift", "v_transport", "length",
    "visibility", "base_ratio", "com_height_ratio", "dest_mu", "dest_rim_height", "dest_opening_ratio",
    "dest_edge_dist", "dest_has_surface", "dest_release_height",
)
FEATURES = FEATURES_M2                 # the default schema (a model saved without feature names reads it)
# milestone 2b: the public fragility flags, the damage margins (computed by the skill, Task 6), the fill and the
# transport acceleration. A feature dict without one (or with a non-finite one) reads its sentinel: no flag,
# far from any threshold, empty, the standard carry.
DAMAGE_SENTINELS: dict[str, float] = {
    "frag_crush": 0.0, "frag_bruise": 0.0, "frag_break": 0.0, "fillable": 0.0,
    "crush_margin": -5.0, "log_v_damage": 5.0, "fill": 0.0, "spill_margin": 5.0, "a_transport": 2.0,
}
DAMAGE_FEATURES: tuple[str, ...] = tuple(DAMAGE_SENTINELS)
FEATURES_M2B: tuple[str, ...] = FEATURES_M2 + DAMAGE_FEATURES
FEATURE_SCHEMAS: dict[str, tuple[str, ...]] = {"m2": FEATURES_M2, "m2b": FEATURES_M2B}
A_TRANSPORT = DAMAGE_SENTINELS["a_transport"]          # m/s^2: the standard carry's acceleration
FRAG_KEYS: tuple[str, ...] = ("k_n_per_mm", "f_damage_n", "v_bruise_ms", "v_break_ms")   # a particle's hidden fragility
DAMAGE_PARTICLE_KEYS: tuple[str, ...] = FRAG_KEYS + ("fill", "spill_angle_deg")          # what the damage features read
FLAG_CLASS = {"frag_crush": "crushable", "frag_bruise": "bruisable", "frag_break": "breakable"}


def model_names(model: Any) -> tuple[str, ...]:
    """The feature names a fitted model reads (FEATURES_M2 for a model without them, or for None)."""
    return tuple(getattr(model, "feature_names", None) or FEATURES_M2)


@dataclass
class Geometry:
    """What the features need from the object: hull faces, CoM, extents, all in the world frame. ``hulls``: each
    convex hull's vertices, for the exact pinch width (without them the width comes from the face centres in
    the pad's band, a lower bound)."""

    centres: np.ndarray
    normals: np.ndarray
    com: np.ndarray
    length: float
    base_extent: float
    com_height: float
    footprint_half_width: float
    hulls: tuple[np.ndarray, ...] = ()


def pinch_width(hulls: Sequence[np.ndarray], point, yaw: float, pad_half=PAD_HALF) -> float:
    """The pinch width: the extent along the closing axis of the convex hulls inside the pads' swept slab
    (|ey . (p - point)| <= pad_half[1] and |z - pad_z| <= pad_half[2], pad_z 4 mm above the point); nan when no
    hull meets the slab. ``hulls``: each convex hull's vertices in the world frame.

    The slab cuts each hull into a convex polytope, and a linear function is extreme at one of its vertices: a
    hull vertex inside the slab, a hull edge crossing a slab face, or one of the slab's four corner lines (they
    run along the closing axis) entering or leaving the hull. Face centres in the band miss a large face: on
    760 corpus v7 rows that width was short by more than 5 mm, or nan, on 233 (31 %)."""
    from scipy.spatial import ConvexHull, QhullError

    pt = np.asarray(point, dtype=float)
    ex = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    ey = np.array([-ex[1], ex[0], 0.0])
    ez = np.array([0.0, 0.0, 1.0])
    c0 = np.array([pt[0], pt[1], pt[2] + 0.004])          # the slab's centre line passes here
    a, b, tol = float(pad_half[1]), float(pad_half[2]), 1e-12
    vals: list[np.ndarray] = []
    for V in hulls:
        V = np.asarray(V, dtype=float)
        try:
            h = ConvexHull(V)
        except QhullError:                                 # a flat hull: joggle it
            h = ConvexHull(V, qhull_options="QJ")
        R = V - c0
        x, u, w = R @ ex, R @ ey, R @ ez
        vals.append(x[(np.abs(u) <= a + tol) & (np.abs(w) <= b + tol)])
        tri = h.simplices
        E = np.unique(np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1), axis=0)
        i, j = E[:, 0], E[:, 1]
        for q, o, lim, olim in ((u, w, a, b), (w, u, b, a)):
            den = q[j] - q[i]
            safe = np.where(np.abs(den) > 1e-15, den, np.inf)
            for d in (lim, -lim):
                t = (d - q[i]) / safe
                ok = (np.abs(den) > 1e-15) & (t >= -tol) & (t <= 1 + tol) & (np.abs(o[i] + t * (o[j] - o[i])) <= olim + tol)
                vals.append((x[i] + t * (x[j] - x[i]))[ok])
        n, off = h.equations[:, :3], h.equations[:, 3]      # inside: n . p + off <= 0
        ne = n @ ex
        par = np.abs(ne) < 1e-15
        for su in (a, -a):
            for sw in (b, -b):
                r = n @ (c0 + su * ey + sw * ez) + off          # along the line p0 + t ex: r + t ne <= 0
                if np.any(r[par] > tol):
                    continue
                tb = -r[~par] / ne[~par]
                up = ne[~par] > 0
                if up.all() or not up.any():
                    continue
                hi, lo = tb[up].min(), tb[~up].max()
                if lo <= hi + tol:
                    vals.append(np.array([lo, hi]))
    allv = np.concatenate(vals) if vals else np.zeros(0)
    return float(allv.max() - allv.min()) if allv.size else float("nan")


def extract(geometry: Geometry, point, yaw: float, mass: float, mu: float, force: float, v_lift: float,
            v_transport: float, destination: str, release_height: float | None = None, visibility: float = 1.0,
            pad_half=PAD_HALF, a_transport: float = A_TRANSPORT) -> dict[str, float]:
    """Features of pinching `geometry` at `point` with the hand yaw `yaw`, squeezing `force` N per
    pad, lifting at `v_lift`, carrying at `v_transport` (accelerating at `a_transport`) to `destination`."""
    C, N = np.asarray(geometry.centres), np.asarray(geometry.normals)
    pt = np.asarray(point, dtype=float)
    ex = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    ey = np.array([-ex[1], ex[0], 0.0])
    pad_z = pt[2] + 0.004
    band = np.abs(C[:, 2] - pad_z) < pad_half[2]
    along = np.abs((C - pt) @ ey) < pad_half[1]
    sel = band & along
    if sel.any():
        s = (C[sel] - pt) @ ex
        width = pinch_width(geometry.hulls, pt, yaw, pad_half) if len(geometry.hulls) else float(s.max() - s.min())
        cosang = np.abs(N[sel] @ ex)
        facing = cosang > 0.3
        slope = float(np.degrees(np.arccos(np.clip(cosang[facing], 0, 1))).mean()) if facing.any() else 90.0
    else:
        width = pinch_width(geometry.hulls, pt, yaw, pad_half) if len(geometry.hulls) else float("nan")
        slope = float("nan")
    com = np.asarray(geometry.com, dtype=float)
    dest = DESTINATIONS[destination]
    com_height = float(geometry.com_height)
    opening = dest.get("opening_half_width")
    if opening is None or dest["kind"] in ("surface", "none"):
        opening_ratio = NO_OPENING_RATIO
    else:
        opening_ratio = float(opening) / max(float(geometry.footprint_half_width), 1e-6)
    return {
        "width": width,
        "slope_deg": slope,
        "lever": float(abs((com - pt) @ ey)),
        "com_h": float(com[2] - pt[2]),
        "mass": float(mass),
        "mu": float(mu),
        "load_ratio": float(mass * G / (2 * mu * force)),
        "v_lift": float(v_lift),
        "v_transport": float(v_transport),
        "length": float(geometry.length),
        "visibility": float(visibility),
        "base_ratio": float(geometry.base_extent / com_height) if com_height > 0 else float("nan"),
        "com_height_ratio": float(com_height / geometry.length) if geometry.length > 0 else float("nan"),
        "dest_mu": float(dest["mu"]),
        "dest_rim_height": float(dest["rim_height"]),
        "dest_opening_ratio": float(opening_ratio),
        "dest_edge_dist": float(edge_distance(dest)),
        "dest_has_surface": 1.0 if has_surface(dest) else 0.0,
        "dest_release_height": float(default_release_height(dest) if release_height is None else release_height),
        "a_transport": float(a_transport),
    }


def as_vector(f: dict[str, float], names: Sequence[str] = FEATURES_M2) -> np.ndarray:
    """Features in ``names`` order; a missing or non-finite damage feature reads its sentinel, any other nan (no
    face under the pads, no height) becomes 0."""
    v = np.array([float(f.get(k, float("nan"))) for k in names], dtype=float)
    bad = np.flatnonzero(~np.isfinite(v))
    if len(bad):
        v[bad] = [DAMAGE_SENTINELS.get(names[i], 0.0) for i in bad]
    return v


# ---------------------------------------------------------------- per-particle features (milestone 2)
SIDE_DEADBAND_M = 0.002


def _pose_arrays(pose: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    R = np.asarray(pose["R"], dtype=float).reshape(3, 3)
    p = np.asarray(pose["p"], dtype=float)
    pt = np.asarray(pose["pt"], dtype=float)
    yaw = float(pose["yaw"])
    return R, p, pt, np.array([-np.sin(yaw), np.cos(yaw), 0.0])


def carry_deg(a_transport: float) -> float:
    """The carry's surface step overshoot (decision 8): 2 atan(a_transport / g), in degrees."""
    return float(np.degrees(2.0 * np.arctan(float(a_transport) / G)))


def entry_flags(entry: dict | None) -> dict[str, float] | None:
    """The four public flags of a fragility entry ({class, law, fillable}), or None without an entry."""
    if not entry:
        return None
    out = {k: 1.0 if entry.get("class") == c else 0.0 for k, c in FLAG_CLASS.items()}
    out["fillable"] = 1.0 if entry.get("fillable") else 0.0
    return out


def damage_columns(arrays: dict[str, np.ndarray], force: float, a_transport: float, n: int) -> dict[str, np.ndarray]:
    """The damage margins (milestone 2b) that the particles' keys determine: crush_margin log(F / f_damage_n) from
    ``f_damage_n``, log_v_damage log(min(v_bruise, v_break)) from either speed, the fill, and spill_margin
    log(spill_angle_deg / carry_deg) from the spill angle (for a filled particle). A value that is NaN (None) gets the
    sentinel; a column whose keys the particles lack is not returned (the base's value stands)."""
    S = DAMAGE_SENTINELS

    def col(k: str) -> np.ndarray:
        return np.asarray(arrays[k], dtype=float).reshape(-1) if k in arrays else np.full(n, np.nan)

    out: dict[str, np.ndarray] = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        if "f_damage_n" in arrays:
            f_d = col("f_damage_n")
            out["crush_margin"] = np.where(np.isfinite(f_d) & (f_d > 0), np.log(float(force) / f_d), S["crush_margin"])
        if "v_bruise_ms" in arrays or "v_break_ms" in arrays:
            vb, vk = col("v_bruise_ms"), col("v_break_ms")
            v = np.fmin(np.where(vb > 0, vb, np.nan), np.where(vk > 0, vk, np.nan))
            out["log_v_damage"] = np.where(np.isfinite(v), np.log(v), S["log_v_damage"])
        if "fill" in arrays:
            out["fill"] = np.nan_to_num(col("fill"), nan=0.0)
        if "spill_angle_deg" in arrays:
            spill, cd = col("spill_angle_deg"), carry_deg(a_transport)
            filled = out["fill"] > 0 if "fill" in out else np.isfinite(spill)
            out["spill_margin"] = np.where(filled & np.isfinite(spill) & (spill > 0) & (cd > 0), np.log(spill / cd), S["spill_margin"])
    return out


def particle_features(base: dict[str, float], particle: dict, pose: dict, force: float) -> dict[str, float]:
    """``base`` (the action's ``extract`` features) with one particle's mass, friction and CoM:
    the CoM maps from the object frame through the pose (R, p); lever, CoM height, base ratio and
    CoM height ratio follow it. The damage features come from the pose's ``entry`` (the public flags; without one
    the base's stand), the particle's fragility, fill and spill angle, the squeeze and the base's ``a_transport``."""
    R, p, pt, ey = _pose_arrays(pose)
    c = R @ np.asarray(particle["com_obj"], dtype=float) + p
    m, mu = float(particle["mass"]), float(particle["mu"])
    h = float(c[2] - float(pose["low_z"]))
    length = float(pose["length"])
    out = dict(base)
    out.update({
        "mass": m,
        "mu": mu,
        "load_ratio": float(m * G / (2 * mu * force)),
        "lever": float(abs((c - pt) @ ey)),
        "com_h": float(c[2] - pt[2]),
        "base_ratio": float(pose["base_extent"]) / h if h > 0 else float("nan"),
        "com_height_ratio": h / length if length > 0 else float("nan"),
    })
    out.update(entry_flags(pose.get("entry")) or {})
    one = {k: np.array([np.nan if particle[k] is None else particle[k]], dtype=float) for k in DAMAGE_PARTICLE_KEYS if k in particle}
    out.update({k: float(v[0]) for k, v in damage_columns(one, force, float(base.get("a_transport", A_TRANSPORT)), 1).items()})
    for k, v in DAMAGE_SENTINELS.items():              # a damage feature nothing determines reads its sentinel
        if not np.isfinite(float(out.get(k, float("nan")))):
            out[k] = v
    return out


def particle_matrix(base: dict[str, float], arrays: dict[str, np.ndarray], pose: dict, force: float,
                    names: Sequence[str] = FEATURES_M2) -> np.ndarray:
    """``particle_features`` for every particle, as rows in ``names`` order (a non-finite damage feature reads its
    sentinel, any other becomes 0, as in as_vector)."""
    R, p, pt, ey = _pose_arrays(pose)
    m, mu = np.asarray(arrays["mass"], dtype=float), np.asarray(arrays["mu"], dtype=float)
    C = np.asarray(arrays["com_obj"], dtype=float).reshape(-1, 3) @ R.T + p
    X = np.tile(as_vector(base, names), (len(m), 1))
    idx = {k: i for i, k in enumerate(names)}
    h = C[:, 2] - float(pose["low_z"])
    length = float(pose["length"])
    with np.errstate(divide="ignore", invalid="ignore"):
        X[:, idx["mass"]] = m
        X[:, idx["mu"]] = mu
        X[:, idx["load_ratio"]] = m * G / (2 * mu * force)
        X[:, idx["lever"]] = np.abs((C - pt) @ ey)
        X[:, idx["com_h"]] = C[:, 2] - pt[2]
        X[:, idx["base_ratio"]] = np.where(h > 0, float(pose["base_extent"]) / h, np.nan)
        X[:, idx["com_height_ratio"]] = h / length if length > 0 else np.nan
    if any(k in idx for k in DAMAGE_FEATURES):
        flags = entry_flags(pose.get("entry")) or {}
        cols = damage_columns(arrays, force, float(base.get("a_transport", A_TRANSPORT)), len(m))
        for k, v in list(flags.items()) + list(cols.items()):      # the rest are the base's (as_vector: or the sentinels)
            if k in idx:
                X[:, idx[k]] = v
    X[~np.isfinite(X)] = 0.0
    return X


def particle_sides(arrays: dict[str, np.ndarray], pose: dict) -> np.ndarray:
    """Which side of the pinch each particle's CoM lies on along the hand's ey (+1, -1; 0 within 2 mm)."""
    R, p, pt, ey = _pose_arrays(pose)
    s = (np.asarray(arrays["com_obj"], dtype=float).reshape(-1, 3) @ R.T + p - pt) @ ey
    return np.where(np.abs(s) < SIDE_DEADBAND_M, 0, np.sign(s)).astype(int)
