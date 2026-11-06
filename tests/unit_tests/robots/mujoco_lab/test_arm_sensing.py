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

"""Joint-torque sensing on the arm port's scene (daily-logs experiments/2026-11-04-arm-port/sim): the attempt's
joint readout and the identify probe. Runs only when RPENT_MUJOCO_SIM_DIR is an arm sim (it has arm.py and ident.py)
and its step-1 robot model exists (RPENT_ARM_ROBOT_MODEL, default <sim>/../out/r1_robot.npz)."""

import os

import numpy as np
import pytest

SIM = os.environ.get("RPENT_MUJOCO_SIM_DIR", "")
pytestmark = pytest.mark.skipif(not (SIM and os.path.exists(os.path.join(SIM, "ident.py"))), reason="not the arm sim")


@pytest.fixture()
def env(monkeypatch):
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    e = MuJoCoLabEnv(damage_meter=False)
    e.reset("ycb__wood_block", {"mass_scale": 1.0, "mu": 0.8, "com_s": None, "v_lift": 0.1}, seed=3)
    return e


def _truth_mass(env):
    return float(env.sim.m.body_subtreemass[env.sim.obj])


def test_the_attempt_gives_a_joint_mass_readout(env):
    prop = env.proposals()[1]
    r = env.execute(prop, 15.0, 0.1, 0.5, "mouse_pad")
    j = r.readouts["joint"]
    assert j["params"] == 4 and j["n"] > 0
    assert abs(j["phi"][0] - _truth_mass(env)) / _truth_mass(env) < 0.05
    info = np.array(j["info"])
    assert info.shape == (4, 4) and np.allclose(info, info.T)
    assert len(j["obj_l7"]["p"]) == 3 and len(j["obj_l7"]["R"]) == 9
    # the log's tags: every traced step is marked, and the fitted window holds lift, hold and transport only
    assert set(j["phases"]) <= {"lift", "hold", "transport"} and "hold" in j["phases"]
    assert "joint_truth_l7" in r.sensor and "joint_truth_l7" not in j


def test_the_identify_probe_reads_mass_and_com(env):
    prop = env.proposals()[1]
    res = env.probe_identify(prop, 15.0, duration=3.0, n_candidates=6)
    assert res.probe == "identify" and res.kind == "measured", res.kind
    j = res.reading
    m = _truth_mass(env)
    assert abs(j["phi"][0] - m) / m < 0.03 and j["params"] == 10
    assert res.seconds > 3.0 and j["duration_s"] == 3.0
    tr = res.sensor["joint_truth_l7"]                # hidden: the checks' truth, never a planner's
    assert np.linalg.norm(np.array(j["phi"][1:4]) / j["phi"][0] - np.array(tr[1:4]) / tr[0]) < 0.01
    # the agents' keys (probes.identify_summary): mass and the CoM in the object frame with 80 % ranges
    assert j["m_hat"] == j["phi"][0] and j["m_range80"][0] < j["m_hat"] < j["m_range80"][1]
    c = np.asarray(env.truth()["com_obj"], dtype=float)
    assert np.linalg.norm(np.asarray(j["com_obj_m"]) - c) < 0.01 and len(j["com_obj_range80_m"]) == 3


def test_a_reset_draws_the_same_torque_noise(env):
    a = env.torque_rng.standard_normal(3)
    env.reset("ycb__wood_block", {"mass_scale": 1.0, "mu": 0.8, "com_s": None, "v_lift": 0.1}, seed=3)
    assert np.array_equal(a, env.torque_rng.standard_normal(3))
