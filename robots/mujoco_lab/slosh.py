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


"""The linear first-mode slosh model of a liquid in an upright cavity (plan 2b, decision 8).

The free surface of a liquid carried with a horizontal acceleration a tilts toward a by atan(a / g) when the
acceleration is held; a sudden step overshoots to twice that. The first sloshing mode of a cylindrical tank of
radius R filled to h has the natural frequency omega = sqrt(g xi1 / R tanh(xi1 h / R)), xi1 = 1.8412 (the first root
of J1'). Per horizontal axis the surface slope s (radians) follows

    s'' + 2 zeta omega s' + omega^2 s = omega^2 a / g,

integrated with semi-implicit Euler. ``step`` returns the angle between the free surface's normal and the cavity's
axis, |s - tilt|, tilt being the axis's deviation from vertical: a container that leans into the acceleration holds
its liquid level with the rim. The model is linear (small angles); Genesis checks it in the annex (Task 8), which
fits a factor on the forcing a per profile class (carry or jolt) and fill: the thresholds' slosh_scale.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

G = 9.81
XI1 = 1.8412


class Slosh:
    def __init__(self, cavity: dict[str, Any], fill: float, zeta: float, g: float = G, scale: dict[str, float] | None = None):
        from robots.mujoco_lab.fragility import cavity_radius, liquid

        h = float(liquid(cavity, fill)["height_m"])
        R = float(cavity_radius(cavity, h))
        self.omega = math.sqrt(g * XI1 / R * math.tanh(XI1 * h / R)) if h > 0 else 0.0
        self.zeta, self.g = float(zeta), float(g)
        self.s = np.zeros(2)
        self.v = np.zeros(2)
        self.scale = dict(scale or {})      # the forcing's factor per profile class at this fill (scale_at)
        self.profile = "jolt"               # the class of the motion from the next step on: carry or jolt

    def step(self, a_h, tilt, dt: float) -> float:
        """Advance by dt under the horizontal acceleration a_h (m/s^2, world x and y) and return the surface angle in
        degrees, the angle between the free surface's normal and the cavity axis tilted by ``tilt`` (radians, x and y);
        0 for an empty cavity. The current profile's factor scales a_h, not the tilt."""
        if self.omega == 0.0:
            return 0.0
        w = self.omega
        a = self.scale.get(self.profile, 1.0) * np.asarray(a_h, dtype=float)
        self.v += float(dt) * (w * w * a / self.g - 2.0 * self.zeta * w * self.v - w * w * self.s)
        self.s += float(dt) * self.v
        return float(np.degrees(np.linalg.norm(self.s - np.asarray(tilt, dtype=float))))


def scale_at(slosh_scale: dict[str, dict[str, float]] | None, fill: float) -> dict[str, float]:
    """Each profile class's forcing factor at ``fill`` from slosh_scale ({class: {str(fill): factor}} at the annex's
    fills): linear in the fill between them, the end values outside; {} for none."""
    out = {}
    for profile, by_fill in (slosh_scale or {}).items():
        fills = sorted(by_fill, key=float)
        out[profile] = float(np.interp(float(fill), [float(f) for f in fills], [float(by_fill[f]) for f in fills]))
    return out
