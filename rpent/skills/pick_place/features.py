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
caller supplies hull face centres and normals in the world frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rpent.skills.pick_place.destinations import DESTINATIONS, edge_distance, has_surface
from rpent.skills.pick_place.destinations import release_height as default_release_height

G = 9.81
PAD_HALF = (0.004, 0.00875, 0.012)   # pad half-sizes: thickness, width, height (the lab gripper)
NO_OPENING_RATIO = 10.0              # "opening over footprint" for places without an opening

FEATURES: tuple[str, ...] = (
    "width", "slope_deg", "lever", "com_h", "mass", "mu", "load_ratio", "v_lift", "v_transport", "length",
    "visibility", "base_ratio", "com_height_ratio", "dest_mu", "dest_rim_height", "dest_opening_ratio",
    "dest_edge_dist", "dest_has_surface", "dest_release_height",
)


@dataclass
class Geometry:
    """What the features need from the object: hull faces, CoM, extents, all in the world frame."""

    centres: np.ndarray
    normals: np.ndarray
    com: np.ndarray
    length: float
    base_extent: float
    com_height: float
    footprint_half_width: float


def extract(geometry: Geometry, point, yaw: float, mass: float, mu: float, force: float, v_lift: float,
            v_transport: float, destination: str, release_height: float | None = None, visibility: float = 1.0,
            pad_half=PAD_HALF) -> dict[str, float]:
    """Features of pinching `geometry` at `point` with the hand yaw `yaw`, squeezing `force` N per
    pad, lifting at `v_lift`, carrying at `v_transport` to `destination`."""
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
        width = float(s.max() - s.min())
        cosang = np.abs(N[sel] @ ex)
        facing = cosang > 0.3
        slope = float(np.degrees(np.arccos(np.clip(cosang[facing], 0, 1))).mean()) if facing.any() else 90.0
    else:
        width, slope = float("nan"), float("nan")
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
    }


def as_vector(f: dict[str, float]) -> np.ndarray:
    """Features in FEATURES order; nan (no face under the pads, no height) becomes 0."""
    v = np.array([float(f.get(k, float("nan"))) for k in FEATURES], dtype=float)
    v[~np.isfinite(v)] = 0.0
    return v
