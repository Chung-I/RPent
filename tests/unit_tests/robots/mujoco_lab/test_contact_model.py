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

"""The gripper's close and the hulls' contact parameters (needs the sim).

The grip used to apply its squeeze as a step on fingers with only 4 N s/m of damping, so a free finger hit the
object at 1 to 2 m/s and threw light objects; a real parallel gripper closes at 20 to 150 mm/s (Robotiq 2F-85).
The hulls' torsional friction was 17.5 mm for every friction coefficient: a pinched block resisted turning 2.8 to
11.4 times the Coulomb answer (daily-logs vlm-grasp-limits/contact_tests, fix_impact)."""

import os

import numpy as np

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")

OBJ = "ycb__brick"
V_CLOSE = 0.15          # m/s
TORS_C = 0.003          # m: the torsional coefficient per unit of sliding friction


def _env(**theta):
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    env = MuJoCoLabEnv()
    env.reset(OBJ, theta or None, seed=0)
    return env


def _finger_dofs(sim):
    return [int(sim.m.joint(j).dofadr[0]) for j in ("jl", "jr")]


@pytest.mark.parametrize("force", [10.0, 70.0])
def test_a_free_finger_closes_at_the_speed_limit(force):
    env = _env()
    sim = env.sim
    sim.teleport_hand([0.0, 0.0, 1.3], 0.0)          # the pads far above the object: nothing to touch
    sim.grip(force)
    speeds = []
    for _ in range(int(0.15 * env.s["scene"].CTRL_HZ) * env.s["scene"].SUB):
        sim.substep()
        speeds.append(max(abs(float(sim.d.qvel[k])) for k in _finger_dofs(sim)))
    assert max(speeds) <= 1.02 * V_CLOSE
    assert speeds[-1] >= 0.95 * V_CLOSE              # it closes at the limit, not slower


def test_a_stopped_finger_squeezes_with_the_full_force():
    """The speed limit is a damping: a finger that the object stops presses with the whole motor force. (A finger
    that still moves inward at v presses with force * (1 - v / V_CLOSE): at the end of the corpus's 0.6 s close the
    brick still settles, and the fingers move at about 6 mm/s.)"""
    env = _env()
    p = env.proposals()[1]
    gr, sim = env.s["grasp"], env.sim
    gr.run_grasp(sim, np.asarray(p.point), p.yaw, force=30.0, lift_h=0.0, hold_s=0.0, close_s=1.5)
    assert max(abs(float(sim.d.qvel[k])) for k in _finger_dofs(sim)) < 5e-4
    assert np.allclose(env._pad_forces(), 30.0, rtol=0.03)


def test_the_open_is_not_slowed():
    env = _env()
    sim = env.sim
    sim.grip(40.0)
    sim.grip(-20.0)
    assert all(float(sim.m.dof_damping[k]) == 4.0 for k in _finger_dofs(sim))


@pytest.mark.parametrize("mu", [0.1, 0.6, 0.9])
def test_the_hull_torsional_friction_is_3_mm_times_mu(mu):
    env = _env(mu=mu)
    m = env.sim.m
    for g in env.sim.hull_ids:
        assert m.geom_friction[g, 0] == pytest.approx(mu)
        assert m.geom_friction[g, 1] == pytest.approx(TORS_C * mu)
