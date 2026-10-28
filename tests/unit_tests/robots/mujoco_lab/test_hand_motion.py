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

"""The hand follows its mocap target smoothly (needs the sim).

The target used to jump once per 20 ms control step. The 4 ms hand weld then yanked the hand at 18.8 g in
every step of a 0.15 m/s lift (49.8 g at 0.40 m/s), and a brick whose grip limit 2μF was 16 times its
weight slid 33 mm down the pads in a 0.40 m/s lift."""

import os

import numpy as np

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")

SHORT = "ycb__brick"
G = 9.81


def _env():
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    return MuJoCoLabEnv()


def test_set_hand_moves_the_target_across_the_substeps():
    """A new target is reached at the end of the control step, on a straight line from the old one."""
    env = _env()
    env.reset(SHORT, seed=0)
    sim, sub = env.sim, env.s["scene"].SUB
    p0 = np.array([0.0, 0.0, 0.5])
    sim.teleport_hand(p0, 0.3)
    sim.set_hand(p0 + [0.0, 0.0, 0.003], 0.3)
    dz = []
    for _ in range(sub):
        sim.substep()
        dz.append(float(sim.d.mocap_pos[sim.mocap][2] - p0[2]))
    assert np.allclose(dz, 0.003 * np.arange(1, sub + 1) / sub)


def test_teleport_hand_leaves_no_motion_pending():
    env = _env()
    env.reset(SHORT, seed=0)
    sim = env.sim
    sim.set_hand([0.0, 0.0, 0.6], 0.0)
    sim.teleport_hand([0.1, 0.0, 0.5], 0.0)
    sim.step_ctrl()
    assert np.allclose(sim.d.mocap_pos[sim.mocap], [0.1, 0.0, 0.5])


def _execute_recorded(env, p, **kw):
    """execute() with the hand's height after every physics substep, and the phase each substep is in."""
    sim = env.sim
    heights, phase_of = [], []
    plain_sub, plain_tracer = sim.substep, env._tracer

    def substep():
        plain_sub()
        heights.append(float(sim.d.xpos[sim.hand][2]))

    def tracer(pt, yaw):
        trace, on_step = plain_tracer(pt, yaw)

        def on_step2(phase):
            on_step(phase)
            phase_of.extend([phase] * (len(heights) - len(phase_of)))
        return trace, on_step2

    sim.substep, env._tracer = substep, tracer
    r = env.execute(p, **kw)
    return r, np.array(heights), phase_of


@pytest.mark.parametrize("v_lift", [0.15, 0.8])
def test_carried_hand_accelerates_under_2g(v_lift):
    """While the hand carries the object up and sets it down, its vertical acceleration stays under 2 g, at
    a middle and at the fastest lift speed of the corpus."""
    env = _env()
    env.reset(SHORT, seed=0)
    p = env.proposals()[1]
    f_std = env.s["t3_make"].std_force(env.nominal_mass())
    r, h, phase_of = _execute_recorded(env, p, force=2 * f_std, v_lift=v_lift, v_transport=0.5, destination="table_centre")
    dt = env.s["scene"].DT
    a = np.abs(h[2:] - 2 * h[1:-1] + h[:-2]) / dt ** 2
    for phase in ("lift", "lower"):
        idx = [i for i in range(len(a)) if phase_of[i + 1] == phase]
        assert idx, phase
        assert a[idx].max() < 2 * G, (phase, round(float(a[idx].max()) / G, 1))


def test_brick_rides_up_with_the_hand():
    """A grip limit 2μF of 16 times the weight holds: the brick does not slide in the pads in the lift."""
    env = _env()
    env.reset(SHORT, {"mu": 0.8, "mass_scale": 1.0, "com_s": None}, seed=0)
    assert 2 * 0.8 * 20.0 > 7 * 9.81 * env.truth()["mass"]
    r = env.execute(env.proposals()[1], force=20.0, v_lift=0.4, v_transport=0.5, destination="table_centre")
    close = [x for x in r.trace if x["phase"] == "close"][-1]
    lift = [x for x in r.trace if x["phase"] == "lift"]
    slip = (lift[-1]["hand_z"] - close["hand_z"]) - (lift[-1]["z"] - close["z"])
    assert all(all(x["pads"]) for x in lift) and slip < 0.002, slip
