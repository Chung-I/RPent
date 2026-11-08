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

"""The 6-DoF executor of the sixdof rerun (design 2026-11-08-sixdof-rerun-design.md, level A): a pinch with a rotation,
its angle bin, and the routing; a Proposal or a grasp within 30° of straight down runs on today's executor unchanged."""

import math

import numpy as np
import pytest

from robots.mujoco_lab import curve_graspgen as gg
from robots.mujoco_lab import sixdof
from robots.mujoco_lab.adapter import Proposal


def _grasp(approach=(0.0, 0.0, -1.0), closing=(1.0, 0.0, 0.0), t=(0.0, 0.0, 0.0)):
    """A 4x4 grasp: +Z the approach, +X the closing axis (made square to it), Y = Z x X."""
    z = np.asarray(approach, dtype=float) / np.linalg.norm(approach)
    x = np.asarray(closing, dtype=float)
    x = x - (x @ z) * z
    x /= np.linalg.norm(x)
    g = np.eye(4)
    g[:3, 0], g[:3, 1], g[:3, 2], g[:3, 3] = x, np.cross(z, x), z, t
    return g


def _tilted(deg, yaw=0.3, t=(0.1, 0.2, 0.9)):
    """A grasp whose approach tilts ``deg`` from straight down about its closing axis, which lies at ``yaw``."""
    a = math.radians(deg)
    ex = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    ey = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    return _grasp(-math.cos(a) * np.array([0.0, 0.0, 1.0]) + math.sin(a) * ey, ex, t)


def _random_within(rng, deg):
    """A grasp with its approach within ``deg`` of straight down, any closing axis square to it, anywhere."""
    a, b = math.radians(rng.uniform(0.0, deg)), rng.uniform(-math.pi, math.pi)
    z = np.array([math.sin(a) * math.cos(b), math.sin(a) * math.sin(b), -math.cos(a)])
    return _grasp(z, rng.normal(size=3), rng.uniform(-0.6, 1.2, 3))


def test_a_graspgen_grasp_becomes_its_pad_centre_and_its_own_frame():
    g = _tilted(20.0)
    p = sixdof.from_graspgen(g, s=0.4, width=0.05)
    assert p.tcp == [float(x) for x in g[:3, 3] + gg.DEPTH * g[:3, 2]]
    assert np.array_equal(np.array(p.R), g[:3, :3]) and (p.s, p.width) == (0.4, 0.05)
    assert sixdof.approach_deg(p) == gg.approach_deg(g) and sixdof.approach_deg(p) == pytest.approx(20.0)


def test_the_angle_bins_are_the_designs_and_a_grasp_from_below_has_none():
    degs = (0.0, 30.0, 30.5, 45.0, 45.1, 60.0, 75.0, 75.1, 105.0, 105.1, 180.0)
    assert [sixdof.angle_bin(d) for d in degs] == ["≤ 30°", "≤ 30°", "30–45°", "30–45°", "45–60°", "45–60°", "60–75°",
                                                   "75–105°", "75–105°", None, None]
    assert sixdof.BINS[:2] == ("≤ 30°", "30–45°") and sixdof.LEVEL_A_DEG == 45.0


def test_a_vertical_grasp_maps_to_the_pilots_point_and_yaw_bit_for_bit():
    rng = np.random.default_rng(20261108)
    for _ in range(300):
        g = _random_within(rng, 30.0)
        m = gg.map_grasp(g)
        q = sixdof.as_vertical(sixdof.from_graspgen(g, s=0.5, width=0.04))
        assert isinstance(q, Proposal) and q.point == m["point"] and q.yaw == m["yaw"] and (q.s, q.width) == (0.5, 0.04)


def test_a_grasp_past_30_degrees_has_no_vertical_mapping():
    with pytest.raises(ValueError, match="30"):
        sixdof.as_vertical(sixdof.from_graspgen(_tilted(30.5)))


class RecEnv:
    """Today's executor, hand path and reach check, recording the proposal and the arguments each one gets."""

    def __init__(self):
        self.calls = []

    def execute(self, proposal, force, v_lift, v_transport=0.5, destination="bin", release_height=None, a_transport=2.0):
        self.calls.append(("execute", proposal, force, v_lift, v_transport, destination, release_height, a_transport))
        return "rollout"

    def hand_path(self, proposal, v_lift, v_transport, destination, release_height=None, a_transport=2.0):
        self.calls.append(("hand_path", proposal, v_lift, v_transport, destination, release_height, a_transport))
        return "path"

    def reach(self, proposal, destination, v_lift=0.1, v_transport=0.5, release_height=None, a_transport=2.0):
        self.calls.append(("reach", proposal, destination, v_lift, v_transport, release_height, a_transport))
        return None


def test_a_proposal_goes_to_todays_executor_as_it_is():
    env, p = RecEnv(), Proposal(s=0.5, point=[0.1, 0.2, 0.8], yaw=0.3, width=0.04)
    assert sixdof.execute(env, p, 12.0, 0.15, 0.5, "bin", None, a_transport=1.5) == "rollout"
    assert sixdof.hand_path(env, p, 0.15, 0.5, "bin", None, a_transport=1.5) == "path"
    assert sixdof.reach(env, p, "bin") is None
    assert [c[1] is p for c in env.calls] == [True, True, True]
    assert env.calls[0][2:] == (12.0, 0.15, 0.5, "bin", None, 1.5)
    assert env.calls[1][2:] == (0.15, 0.5, "bin", None, 1.5)
    assert env.calls[2][2:] == ("bin", 0.1, 0.5, None, 2.0)


def test_a_vertical_grasp_goes_to_todays_executor_with_its_derived_yaw():
    env, g = RecEnv(), _tilted(25.0, yaw=2.9)
    p6 = sixdof.from_graspgen(g, s=0.5, width=0.04)
    sixdof.execute(env, p6, 12.0, 0.15, 0.5, "bin", None, a_transport=1.5)
    sixdof.hand_path(env, p6, 0.15, 0.5, "bin", None, a_transport=1.5)
    sixdof.reach(env, p6, "bin")
    m = gg.map_grasp(g)
    assert [c[0] for c in env.calls] == ["execute", "hand_path", "reach"]
    assert all(isinstance(c[1], Proposal) and c[1].point == m["point"] and c[1].yaw == m["yaw"] for c in env.calls)


def test_a_tilted_grasp_never_reaches_todays_executor():
    env = RecEnv()
    with pytest.raises(NotImplementedError):
        sixdof.execute(env, sixdof.from_graspgen(_tilted(40.0)), 12.0, 0.15)
    assert env.calls == []


SIM = pytest.mark.skipif(not __import__("os").environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")


@SIM
def test_on_the_lab_a_vertical_grasp_has_todays_hand_path_and_rollout():
    from robots.mujoco_lab.adapter import MuJoCoLabEnv
    from robots.mujoco_lab.curve_poseset import trace_sha

    env = MuJoCoLabEnv()
    env.reset("ycb__sugar_box", seed=0)
    p = env.proposals((0.5,))[0]
    pad = np.asarray(p.point) + [0.0, 0.0, gg.PAD_ABOVE]
    g = _grasp((0.0, 0.0, -1.0), (math.cos(p.yaw), math.sin(p.yaw), 0.0), pad + [0.0, 0.0, gg.DEPTH])
    p6 = sixdof.from_graspgen(g, s=p.s, width=p.width)
    q = sixdof.as_vertical(p6)
    a, b = sixdof.hand_path(env, p6, 0.15, 0.5, "bin"), env.hand_path(q, 0.15, 0.5, "bin")
    assert len(a) == len(b) and all(x[0] == y[0] and np.array_equal(x[1], y[1]) and x[2] == y[2] for x, y in zip(a, b))
    out = []
    for run in (lambda: env.execute(q, 12.0, 0.15, 0.5, "bin"), lambda: sixdof.execute(env, p6, 12.0, 0.15, 0.5, "bin")):
        env.reset("ycb__sugar_box", seed=0)
        ro = run()
        out.append((ro.label, trace_sha(ro.trace), np.round(np.asarray(env.sim.d.qpos, dtype=float), 9).tobytes()))
    assert out[0] == out[1]
