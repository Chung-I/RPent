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


"""The linear first-mode slosh model: the natural frequency, the step overshoot, the quasi-static ramp and the tilt."""

import math

import numpy as np
import pytest

CYL = {"axis_obj": [0.0, 0.0, 1.0], "bottom_obj": [0.0, 0.0, 0.0], "r_bottom_m": 0.04, "r_top_m": 0.04, "height_m": 0.09}
G = 9.81


def test_natural_frequency():
    from robots.mujoco_lab.slosh import XI1, Slosh

    s = Slosh(CYL, 0.6, 0.01)
    h, R = 0.054, 0.04
    assert XI1 == 1.8412 and s.omega == pytest.approx(math.sqrt(G * XI1 / R * math.tanh(XI1 * h / R)), rel=1e-9)


def test_slosh_step_overshoots_twice_the_static():
    from robots.mujoco_lab.slosh import Slosh

    s = Slosh(CYL, 0.6, 0.0)
    dt = 0.0005
    peak = max(s.step([1.0, 0.0], [0.0, 0.0], dt) for _ in range(int(2.0 / dt)))
    assert peak == pytest.approx(math.degrees(2.0 * math.atan(1.0 / G)), rel=0.05)


def test_slosh_slow_ramp_is_quasi_static():
    from robots.mujoco_lab.slosh import Slosh

    s = Slosh(CYL, 0.6, 0.01)
    dt, T = 0.002, 10.0
    for i in range(int(T / dt)):
        a = 1.0 * (i + 1) * dt / T
        out = s.step([0.0, a], [0.0, 0.0], dt)
    assert out == pytest.approx(math.degrees(1.0 / G), rel=0.05)


def test_tilt_alone_is_the_surface_angle():
    from robots.mujoco_lab.slosh import Slosh

    s = Slosh(CYL, 0.6, 0.01)
    assert s.step([0.0, 0.0], [0.1, 0.0], 0.002) == pytest.approx(math.degrees(0.1), rel=1e-9)


def test_leaning_into_the_carry_lowers_the_surface_angle():
    """A carry at a constant 1 m/s^2 tilts the free surface 1/g toward the acceleration; a container whose axis leans the
    same way by 1/g holds its liquid level with the rim, one that leans away doubles the angle."""
    from robots.mujoco_lab.slosh import Slosh

    dt = 0.002
    out = {}
    for lean in (-1.0, 0.0, 1.0):
        s = Slosh(CYL, 0.6, 0.2)
        for _ in range(int(5.0 / dt)):
            out[lean] = s.step([1.0, 0.0], [lean / G, 0.0], dt)
    assert out[1.0] < 0.05 * out[0.0] and out[-1.0] == pytest.approx(2.0 * out[0.0], rel=0.02)


def test_empty_container_has_no_slosh():
    from robots.mujoco_lab.slosh import Slosh

    s = Slosh(CYL, 0.0, 0.01)
    assert s.omega == 0.0 and s.step([5.0, 0.0], [0.0, 0.0], 0.002) == 0.0
