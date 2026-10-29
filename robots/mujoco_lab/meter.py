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



"""The damage meter (plan 2b, Task 5): three channels measured in every physics substep of an attempt or a probe,
with times from its start. The sim feeds it (adapter.MuJoCoLabEnv._damage_substep); it never writes the state.

* ``fn_pad_max``: the larger pad's normal force on the object, each pad averaged over the last 10 ms, as a running
  maximum kept as a step list: an entry when the maximum first passes FN_FLOOR_N and each time it grows 1 % over the
  last entry, and the final maximum. A threshold's first crossing then reads late by at most one 1 % growth.
* ``surface_max``: for a filled cavity, the angle between the liquid's free surface and the cavity (slosh.Slosh),
  driven by the CoM's horizontal acceleration (its velocity difference over 10 ms) and the cavity axis's tilt, kept
  the same way from SURFACE_FLOOR_DEG.
* ``impacts``: once armed (after a release or a loss), the CoM speed one substep before each new contact of an object
  hull with a non-pad geom, at MIN_IMPACT_MS or more.

The floors lie below every damage threshold of the laws (thresholds_v0: the lowest damage force 3 sd below its
median is 5.8 N, the lowest bruise speed 0.39 m/s, the smallest spill angle of a fill in [0.3, 0.9] 2.9 deg) and keep
the rows small. Values keep four significant digits.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

WINDOW_S = 0.010
GROW = 0.01
FN_FLOOR_N = 1.0
SURFACE_FLOOR_DEG = 0.5
MIN_IMPACT_MS = 0.1
CHANNELS = ("fn_pad_max", "surface_max", "impacts")


def _r(v: float) -> float:
    return float(f"{float(v):.4g}")


class StepMax:
    """A running maximum as a step list [(t, value)]: an entry when the maximum first exceeds ``floor`` and each time
    it exceeds the last entry by ``grow``; ``steps`` ends with the final maximum."""

    def __init__(self, floor: float, grow: float = GROW):
        self.floor, self.grow = float(floor), float(grow)
        self.entries: list[list[float]] = []
        self.best, self.t_best = -math.inf, 0.0

    def add(self, t: float, v: float) -> None:
        v = float(v)
        if v <= self.best:
            return
        self.best, self.t_best = v, float(t)
        if v > self.floor and (not self.entries or v > (1.0 + self.grow) * self.entries[-1][1]):
            self.entries.append([float(t), v])

    def steps(self) -> list[list[float]]:
        out = [[round(t, 4), _r(v)] for t, v in self.entries]
        if self.best > self.floor and (not self.entries or self.entries[-1][1] < self.best):
            out.append([round(self.t_best, 4), _r(self.best)])
        return out


def tilt_of(axis_world) -> np.ndarray:
    """The cavity axis's deviation from vertical as a horizontal vector: its angle (radians) along the axis's
    horizontal direction; small-angle slopes (slosh.Slosh) and the static tilt at any angle agree on it."""
    a = np.asarray(axis_world, dtype=float)
    h = a[:2]
    nh = float(np.linalg.norm(h))
    if nh < 1e-12:
        return np.zeros(2)
    return math.atan2(nh, float(a[2])) * h / nh


class DamageMeter:
    def __init__(self, dt: float, slosh: Any = None, window_s: float = WINDOW_S):
        self.dt, self.slosh, self.n = float(dt), slosh, 0
        k = max(1, int(round(window_s / self.dt)))
        self._pads: deque = deque(maxlen=k)
        self._vel: deque = deque(maxlen=k + 1)
        self.force = StepMax(FN_FLOOR_N)
        self.surface = StepMax(SURFACE_FLOOR_DEG)
        self.impacts: list[list[float]] = []
        self._touching: set | None = None
        self._speed = 0.0
        self._drive = 0.0
        self.jam_n = 0.0          # the largest excess of the window's pad force over the largest drive so far

    @property
    def t(self) -> float:
        return self.n * self.dt

    def substep(self, pad_n, touching: set, com_vel, axis_world=None, armed: bool = False, drive: float = 0.0) -> None:
        """One physics substep: the pads' normal forces on the object (left, right), the set of (hull, geom) contacts
        with non-pad geoms, the CoM velocity, the cavity axis in the world (None: no surface channel), whether a
        release or a loss has armed the impact channel, and the grip drive's squeeze per pad (0 for an open grip)."""
        self.n += 1
        t = self.t
        self._pads.append((float(pad_n[0]), float(pad_n[1])))
        fn = float(max(np.mean(self._pads, axis=0)))
        self.force.add(t, fn)
        self._drive = max(self._drive, float(drive))
        self.jam_n = max(self.jam_n, fn - self._drive)
        v = np.asarray(com_vel, dtype=float)
        self._vel.append(v)
        if self.slosh is not None and axis_world is not None:
            # a difference over the full window only: a shorter span at the start turns contact chatter into a kick
            full = len(self._vel) == self._vel.maxlen
            a = (self._vel[-1] - self._vel[0]) / ((len(self._vel) - 1) * self.dt) if full else np.zeros(3)
            self.surface.add(t, self.slosh.step(a[:2], tilt_of(axis_world), self.dt))
        if armed and self._touching is not None and set(touching) - self._touching and self._speed >= MIN_IMPACT_MS:
            self.impacts.append([round(t, 4), _r(self._speed)])
        self._touching = set(touching)
        self._speed = float(np.linalg.norm(v))

    def to_data(self) -> dict[str, list[list[float]]]:
        return {"fn_pad_max": self.force.steps(), "surface_max": self.surface.steps() if self.slosh is not None else [],
                "impacts": [list(x) for x in self.impacts]}


def maxima(meter: dict[str, Any] | None) -> dict[str, float]:
    """The channels' maxima of a meter record (0 for an empty channel)."""
    return {k: max((float(v) for _, v in (meter or {}).get(k) or []), default=0.0) for k in CHANNELS}
