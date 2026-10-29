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



"""The damage meter's channels without the sim: the running-maximum step lists, the 10 ms force window, the impacts
after a release, and the surface angle of a filled cavity that tilts."""

import numpy as np
import pytest

DT = 0.002


def _cup():
    return {"axis_obj": [0.0, 0.0, 1.0], "bottom_obj": [0.0, 0.0, 0.0], "r_bottom_m": 0.03, "r_top_m": 0.04, "height_m": 0.08}


def test_step_max_keeps_one_percent_steps_and_the_final_maximum():
    from robots.mujoco_lab.meter import StepMax

    s = StepMax(floor=1.0)
    for k in range(1, 501):
        s.add(k * DT, 0.5 + 0.04 * k)                # 0.54 to 20.5, rising
    s.add(1.1, 3.0)                                   # below the maximum: no entry
    steps = s.steps()
    assert steps[0][1] > 1.0 and steps[-1] == [pytest.approx(1.0), pytest.approx(20.5)]
    assert all(b[1] > 1.01 * a[1] for a, b in zip(steps[:-2], steps[1:-1]))
    assert all(b[0] > a[0] for a, b in zip(steps, steps[1:]))
    assert StepMax(floor=1.0).steps() == []


def test_force_window_averages_ten_ms():
    """One substep of 50 N among 5 N substeps reads (50 + 4 x 5) / 5 = 14 N: the channel is the larger pad's force
    over the last 10 ms (5 substeps)."""
    from robots.mujoco_lab.meter import DamageMeter

    m = DamageMeter(DT)
    for k in range(20):
        m.substep((50.0 if k == 10 else 5.0, 4.0), set(), np.zeros(3))
    assert m.to_data()["fn_pad_max"][-1][1] == pytest.approx(14.0)


def test_impacts_need_the_release_and_a_new_contact():
    """A new (hull, geom) contact records the CoM speed one substep before it, once the meter is armed; a contact that
    stays records nothing, and one before the arming records nothing."""
    from robots.mujoco_lab.meter import DamageMeter

    m = DamageMeter(DT)
    v = np.array([0.0, 0.0, -1.5])
    m.substep((0.0, 0.0), set(), v)
    m.substep((0.0, 0.0), {(3, 7)}, v)                               # not armed: no impact
    m.substep((0.0, 0.0), set(), v * 1.1, armed=True)
    m.substep((0.0, 0.0), {(3, 7)}, np.zeros(3), armed=True)        # new: the speed before, 1.65
    m.substep((0.0, 0.0), {(3, 7)}, np.zeros(3), armed=True)        # stays
    m.substep((0.0, 0.0), {(3, 7), (4, 7)}, np.array([0.0, 0.0, -0.05]), armed=True)   # new, under 0.1 m/s
    assert m.to_data()["impacts"] == [[pytest.approx(4 * DT), pytest.approx(1.65)]]


def test_surface_angle_of_a_still_cavity_is_its_tilt():
    """A cavity held still and tilted 30 deg: the liquid stays level, so the surface makes 30 deg with the cavity."""
    from robots.mujoco_lab.meter import DamageMeter
    from robots.mujoco_lab.slosh import Slosh

    m = DamageMeter(DT, slosh=Slosh(_cup(), 0.6, 0.01))
    t = np.radians(30.0)
    axis = np.array([np.sin(t) * np.cos(0.7), np.sin(t) * np.sin(0.7), np.cos(t)])
    for _ in range(50):
        m.substep((0.0, 0.0), set(), np.zeros(3), axis_world=axis)
    assert m.to_data()["surface_max"][-1][1] == pytest.approx(30.0, abs=0.01)
    assert DamageMeter(DT).to_data()["surface_max"] == []          # no cavity, no surface channel


def test_surface_angle_has_no_start_up_kick():
    """Contact chatter of a resting object (its velocity flips +-4 mm/s every substep) has no net acceleration: the
    acceleration is a velocity difference over a full 10 ms window, never over the shorter span at the meter's start,
    so the lightly damped slosh never rings up from the start (the shorter span gave a 1.4 deg kick)."""
    from robots.mujoco_lab.meter import DamageMeter, maxima
    from robots.mujoco_lab.slosh import Slosh

    m = DamageMeter(DT, slosh=Slosh(_cup(), 0.6, 0.01))
    for k in range(500):
        m.substep((0.0, 0.0), set(), np.array([0.004 * (-1) ** k, 0.0, 0.0]), axis_world=np.array([0.0, 0.0, 1.0]))
    assert maxima(m.to_data())["surface_max"] == 0.0               # below the 0.5 deg floor all along


def test_a_tilt_past_the_spill_angle_is_a_spill():
    """The meter's surface channel and the damage rule: a mug at fill 0.8 turned slowly to 40 deg spills, and the
    spill replaces the motion label."""
    from robots.mujoco_lab.fragility import damage_event, spill_angle_deg
    from robots.mujoco_lab.labels import final_label
    from robots.mujoco_lab.meter import DamageMeter
    from robots.mujoco_lab.slosh import Slosh

    entry = {"class": "breakable", "law": "ceramic", "fillable": True, "cavity": _cup()}
    m = DamageMeter(DT, slosh=Slosh(_cup(), 0.8, 0.01))
    for k in range(2000):                                            # 4 s, 10 deg/s
        t = np.radians(min(40.0, 10.0 * k * DT))
        m.substep((0.0, 0.0), set(), np.zeros(3), axis_world=np.array([np.sin(t), 0.0, np.cos(t)]))
    meter = m.to_data()
    spill = spill_angle_deg(_cup(), 0.8)
    label, channel, t_spill = damage_event(meter, {"v_break_ms": 3.0}, entry, 0.8)
    assert (label, channel) == ("spilled", "surface") and t_spill == pytest.approx(spill / 10.0, abs=0.02)
    assert final_label("placed_stable", {"label": label, "channel": channel, "t": t_spill}) == "spilled"


def test_meter_keeps_the_pad_force_above_the_drive():
    """jam_n: the largest excess of the 10 ms pad force over the largest drive so far. An open grip (drive 0) pressed
    at 60 N gives 60; a squeeze read at 25 N under a drive of 20 N gives 5."""
    from robots.mujoco_lab.meter import DamageMeter

    m = DamageMeter(DT)
    for _ in range(10):
        m.substep((60.0, 0.0), set(), (0.0, 0.0, 0.0))
    assert abs(m.jam_n - 60.0) < 1e-9
    s = DamageMeter(DT)
    for _ in range(10):
        s.substep((25.0, 24.0), set(), (0.0, 0.0, 0.0), drive=20.0)
    assert abs(s.jam_n - 5.0) < 1e-9
