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

from robots.mujoco_lab import adapter
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


# ------------------------------------------------------------------ the tilted path's geometry

def _yaw_frame(yaw):
    """The lab hand's rotation at ``yaw`` (pads down): Rz(yaw)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_the_hand_sits_0_048_m_back_along_the_approach_with_its_z_against_it():
    p6 = sixdof.from_graspgen(_grasp((0.0, 0.0, -1.0), (math.cos(0.7), math.sin(0.7), 0.0), (0.1, 0.2, 0.9)))
    pos, R = sixdof.hand_frame(p6)
    assert sixdof.HAND_OFF == 0.048
    assert np.allclose(pos, np.asarray(p6.tcp) + [0.0, 0.0, 0.048], atol=1e-15) and np.allclose(R, _yaw_frame(0.7), atol=1e-15)
    p6 = sixdof.from_graspgen(_tilted(40.0, yaw=0.7))
    pos, R = sixdof.hand_frame(p6)
    a = np.array(p6.R)[:, 2]
    assert np.allclose(pos, np.asarray(p6.tcp) - 0.048 * a) and np.allclose(R[:, 2], -a) and np.allclose(R[:, 0], np.array(p6.R)[:, 0])
    assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)


def test_the_twin_is_the_same_pinch_turned_a_half_turn_about_the_approach():
    p6 = sixdof.from_graspgen(_tilted(40.0, yaw=0.7), s=0.3, width=0.05)
    t = sixdof.twin(p6)
    R, Rt = np.array(p6.R), np.array(t.R)
    assert t.tcp == p6.tcp and (t.s, t.width) == (0.3, 0.05) and sixdof.twin(t).R == p6.R
    assert np.array_equal(Rt[:, 2], R[:, 2]) and np.array_equal(Rt[:, :2], -R[:, :2])
    assert np.allclose(sixdof.hand_frame(t)[1], sixdof.hand_frame(p6)[1] @ np.diag([-1.0, -1.0, 1.0]))
    assert np.allclose(sixdof.hand_frame(t)[0], sixdof.hand_frame(p6)[0])


def _sugar_box():
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    env = MuJoCoLabEnv()
    env.reset("ycb__sugar_box", seed=0)
    return env, env.proposals((0.5,))[0]


def _at(p, tilt_deg=0.0):
    """A grasp whose pad centre is the lab's pad centre at the proposal ``p``, tilted about its closing axis."""
    a = math.radians(tilt_deg)
    ex = np.array([math.cos(p.yaw), math.sin(p.yaw), 0.0])
    ey = np.array([-math.sin(p.yaw), math.cos(p.yaw), 0.0])
    z = -math.cos(a) * np.array([0.0, 0.0, 1.0]) + math.sin(a) * ey
    pad = np.asarray(p.point) + [0.0, 0.0, gg.PAD_ABOVE]
    return _grasp(z, ex, pad - gg.DEPTH * z)


def test_the_bearing_turn_is_the_shortest_signed_angle_about_the_base():
    assert sixdof.bearing_turn((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)) == pytest.approx(math.pi / 2)
    assert sixdof.bearing_turn((1.0, 1.0), (1.0, 2.0), (2.0, 1.0)) == pytest.approx(-math.pi / 2)
    assert sixdof.bearing_turn((0.0, 0.0), (-1.0, 0.01), (-1.0, -0.01)) == pytest.approx(2 * math.atan(0.01))     # across the -x axis
    assert sixdof.bearing_turn((0.0, 0.0), (0.3, 0.4), (0.6, 0.8)) == pytest.approx(0.0, abs=1e-15)


@SIM
def test_the_transport_turn_is_the_pad_centres_bearing_change_from_the_arm_base():
    env, p = _sugar_box()
    p6 = sixdof.from_graspgen(_at(p, 40.0), s=p.s, width=p.width)
    base = np.asarray(sixdof._armlib(env.sim.arm).BASE_POS[:2])
    goal, tcp = np.asarray(env._dest("bin")["xy"]), np.asarray(p6.tcp)[:2]
    b = [math.atan2(v[1] - base[1], v[0] - base[0]) for v in (tcp, goal)]
    assert sixdof.turn_deg(env, p6, "bin") == pytest.approx(math.degrees(math.remainder(b[1] - b[0], 2 * math.pi)), abs=1e-9)
    assert abs(sixdof.turn_deg(env, p6, "bin")) > 30.0


@SIM
def test_at_no_tilt_the_tilted_hand_path_is_todays_but_for_the_transport_turn():
    env, p = _sugar_box()
    p6 = sixdof.from_graspgen(_at(p), s=p.s, width=p.width)
    a = sixdof.hand_path6(env, p6, 0.15, 0.5, "bin")
    b = env.hand_path(sixdof.as_vertical(p6), 0.15, 0.5, "bin")
    turn = math.radians(sixdof.turn_deg(env, p6, "bin"))
    assert [x[0] for x in a] == [x[0] for x in b]
    assert all(np.allclose(x[1], y[1], rtol=0.0, atol=1e-12) for x, y in zip(a, b))
    k0 = [x[0] for x in a].index("transport")
    k1 = len(a) - [x[0] for x in a][::-1].index("transport")
    start, goal = b[k0 - 1][1][:2], b[k1 - 1][1][:2]
    assert all(np.allclose(x[2], _yaw_frame(y[2]), atol=1e-12) for x, y in zip(a[:k0], b[:k0]))
    fr = [float(np.linalg.norm(y[1][:2] - start) / np.linalg.norm(goal - start)) for y in b[k0:k1]]
    assert all(np.allclose(x[2], _yaw_frame(y[2] + turn * f), atol=1e-9) for x, y, f in zip(a[k0:k1], b[k0:k1], fr))
    assert all(np.allclose(x[2], _yaw_frame(y[2] + turn), atol=1e-12) for x, y in zip(a[k1:], b[k1:]))


def test_the_return_rotation_takes_the_approach_to_straight_down_about_a_horizontal_axis():
    rng = np.random.default_rng(3)
    down = np.array([0.0, 0.0, -1.0])
    for _ in range(50):
        g = _random_within(rng, 45.0)
        a, th = g[:3, 2], math.radians(gg.approach_deg(g))
        U = sixdof.return_rotation(a, 1.0)
        assert np.allclose(U @ a, down, rtol=0.0, atol=1e-12)
        assert np.allclose(U @ U.T, np.eye(3), rtol=0.0, atol=1e-12) and np.linalg.det(U) == pytest.approx(1.0)
        assert abs((U - U.T)[1, 0]) < 1e-12                                     # the axis is horizontal: no turn about z
        assert math.acos(np.clip((np.trace(U) - 1.0) / 2.0, -1.0, 1.0)) == pytest.approx(th, abs=1e-7)
        for f in (0.25, 0.5):
            assert math.acos(np.clip(-(sixdof.return_rotation(a, f) @ a)[2], -1.0, 1.0)) == pytest.approx((1.0 - f) * th, abs=1e-7)
    assert np.array_equal(sixdof.return_rotation(down, 0.7), np.eye(3))


@SIM
def test_a_tilted_hand_path_approaches_along_the_approach_lifts_straight_up_returns_to_vertical_and_retreats_up():
    from robots.mujoco_lab import adapter

    env, p = _sugar_box()
    p6 = sixdof.from_graspgen(_at(p, 40.0), s=p.s, width=p.width)
    path = sixdof.hand_path6(env, p6, 0.15, 0.5, "bin")
    hp, Rh = sixdof.hand_frame(p6)
    a = np.array(p6.R)[:, 2]
    by = {ph: [x for x in path if x[0] == ph] for ph in ("wait", "approach", "close", "lift", "hold", "transport", "lower", "release", "retreat")}
    flat = sixdof.hand_path6(env, sixdof.from_graspgen(_at(p), s=p.s, width=p.width), 0.15, 0.5, "bin")
    for ph in ("wait", "approach", "close", "lift", "hold", "release", "retreat"):
        assert len(by[ph]) == sum(x[0] == ph for x in flat) > 0
    assert [x[0] for x in path] == sorted((x[0] for x in path), key=list(by).index)
    k0 = [x[0] for x in path].index("transport")
    assert all(np.array_equal(x[2], Rh) for x in path[:k0])
    assert np.allclose(by["wait"][0][1], hp - adapter.APPROACH_M * a)
    s = [float((x[1] - by["wait"][0][1]) @ a) for x in by["approach"]]
    assert all(np.allclose(x[1], by["wait"][0][1] + si * a) for x, si in zip(by["approach"], s)) and s == sorted(s)
    assert np.array_equal(by["approach"][-1][1], hp) and all(np.array_equal(x[1], hp) for x in by["close"])
    assert all(np.allclose(x[1][:2], hp[:2]) for x in by["lift"]) and by["lift"][-1][1][2] == pytest.approx(hp[2] + adapter.LIFT_H)
    # the transport: the pad centre on a straight line at the lifted height, the hand turned about the vertical and
    # returned toward vertical by the same fraction (amendments 5 and 7)
    turn = math.radians(sixdof.turn_deg(env, p6, "bin"))
    start, goal = np.asarray(p6.tcp)[:2], np.asarray(env._dest("bin")["xy"])
    z_top = float(np.asarray(p6.tcp)[2] + adapter.LIFT_H)
    for x in by["transport"]:
        c = x[1] - sixdof.HAND_OFF * x[2][:, 2]                                     # the pad centre
        f = float((c[:2] - start) @ (goal - start) / ((goal - start) @ (goal - start)))
        assert c[2] == pytest.approx(z_top, abs=1e-12) and np.allclose(c[:2], start + f * (goal - start), rtol=0.0, atol=1e-12)
        assert np.allclose(x[2], _yaw_frame(turn * f) @ sixdof.return_rotation(a, f) @ Rh, rtol=0.0, atol=1e-9)
    Re = _yaw_frame(turn) @ sixdof.return_rotation(a, 1.0) @ Rh
    assert np.allclose(Re[:, 2], [0.0, 0.0, 1.0], rtol=0.0, atol=1e-12)                # vertical, the pads down
    assert all(np.allclose(x[2], Re, rtol=0.0, atol=1e-12) for x in by["transport"][-1:] + by["lower"] + by["release"] + by["retreat"])
    assert np.allclose(by["transport"][-1][1][:2], goal, rtol=0.0, atol=1e-12)            # the pad centre under the hand, over the goal
    assert all(np.allclose(x[1][:2], goal, rtol=0.0, atol=1e-12) for x in by["lower"] + by["release"])
    zs = [x[1][2] for x in by["lower"]]
    assert zs == sorted(zs, reverse=True)
    rel = by["release"][0][1]
    V = np.concatenate(env._hull_vertex_list())                     # the object rigid in the hand: its rest pose turned and returned
    Q = _yaw_frame(turn) @ sixdof.return_rotation(a, 1.0)
    low = rel[2] - sixdof.HAND_OFF + float(((V - np.asarray(p6.tcp)) @ Q.T)[:, 2].min())
    assert low == pytest.approx(env._release_target_low_z(env._dest("bin"), None), abs=1e-9)   # the release height's rule
    assert np.allclose(by["retreat"][-1][1], rel + [0.0, 0.0, adapter.RETREAT_M], rtol=0.0, atol=1e-12)


# ------------------------------------------------------------------ the rotation IK, the candidates and the path check

def test_the_rows_generator_is_fixed_by_its_key():
    a, b, c = sixdof.row_rng("hammer:3:g17"), sixdof.row_rng("hammer:3:g17"), sixdof.row_rng("hammer:3:g18")
    x = a.uniform(size=4)
    assert np.array_equal(x, b.uniform(size=4)) and not np.array_equal(x, c.uniform(size=4))
    assert sixdof.row_key({"object": "hammer", "idx": 3}, "g17") == "hammer:3:g17"


@SIM
def test_the_rotation_ik_at_a_vertical_target_is_the_arms_yaw_ik():
    env, p = _sugar_box()
    arm = env.sim.arm
    A = sixdof._armlib(arm)
    pos, R = sixdof.hand_frame(sixdof.from_graspgen(_at(p)))
    qa, ea, ra = arm.ik(pos, p.yaw, q_init=A.Q_REST)
    qb, eb, rb = sixdof.ik6(arm, pos, _yaw_frame(p.yaw), q_init=A.Q_REST)
    assert np.allclose(qa, qb, atol=1e-9) and eb < 1e-4 and rb < 1e-3


@SIM
def test_the_rotation_ik_reaches_a_tilted_pose():
    import mujoco

    env, p = _sugar_box()
    arm = env.sim.arm
    pos, R = sixdof.hand_frame(sixdof.from_graspgen(_at(p, 40.0)))
    q, ep, er = sixdof.ik6(arm, pos, R, q_init=arm.seeds(pos)[0])
    arm.kd.qpos[arm.q0:arm.q0 + 7] = q
    mujoco.mj_kinematics(arm.m, arm.kd)
    assert ep < 1e-4 and er < 1e-3 and np.allclose(arm.kd.xpos[arm.hand], pos, atol=1e-4)
    assert np.allclose(arm.kd.xmat[arm.hand].reshape(3, 3), R, atol=1e-3)


@SIM
def test_the_candidates_come_in_a_fixed_order_from_the_rows_key_and_are_distinct():
    env, p = _sugar_box()
    p6 = sixdof.from_graspgen(_at(p, 40.0))
    a, b = sixdof.candidates(env, p6, "box:0:g1"), sixdof.candidates(env, p6, "box:0:g1")
    assert len(a) >= 2 and [c["index"] for c in a] == list(range(len(a)))
    assert [(c["twin"], c["seed"]) for c in a] == [(c["twin"], c["seed"]) for c in b] and all(np.array_equal(x["q"], y["q"]) for x, y in zip(a, b))
    assert a[0]["seed"] == "current" and {c["twin"] for c in a} == {0, 1}
    assert all(np.max(np.abs(x["q"] - y["q"])) > 1e-3 for i, x in enumerate(a) for y in a[:i])
    order = [s for s in ("current", "aimed", "rest") + tuple(f"random{k}" for k in range(sixdof.N_RANDOM_SEEDS))]
    assert [c["seed"] for c in a] == sorted((c["seed"] for c in a), key=order.index)


@SIM
def test_at_no_tilt_the_path_check_passes_where_reach_does():
    env, p = _sugar_box()
    assert env.reach(p, "bin") is None
    p6 = sixdof.from_graspgen(_at(p), s=p.s, width=p.width)
    path = env.hand_path(sixdof.as_vertical(p6), 0.1, 0.5, "bin")
    plan = env.sim.plan_free(path[0][1], path[0][2])
    same = abs(math.remainder(plan["yaw_rep"] - sixdof.as_vertical(p6).yaw, 2 * math.pi)) < 1e-9     # plan_free's yaw is p6's or its twin's
    assert sixdof.reach6(env, p6 if same else sixdof.twin(p6), plan["q"], "bin") is None
    with pytest.raises(ValueError):
        sixdof.reach6(env, sixdof.twin(p6) if same else p6, plan["q"], "bin")          # a start configuration not at the pregrasp


def test_the_choice_is_the_first_candidate_whose_path_passes_on_its_own_twin(monkeypatch):
    p6 = sixdof.from_graspgen(_tilted(40.0))
    cands = [{"index": k, "twin": k % 2, "seed": s, "q": np.full(7, float(k))} for k, s in enumerate(("current", "current", "aimed"))]
    seen = []

    def fake_reach6(env, p, q, destination, *args):
        seen.append((p.R, float(q[0])))
        return None if q[0] == 2.0 else {"reason": "ik", "phase": "transport", "step": 7}

    monkeypatch.setattr(sixdof, "candidates", lambda env, p, key: cands)
    monkeypatch.setattr(sixdof, "reach6", fake_reach6)
    ch = sixdof.choose(None, p6, "k", "bin")
    assert ch["goal"] and ch["candidate"] is cands[2]
    assert ch["failed"] == [{"index": 0, "reason": "ik", "phase": "transport", "step": 7}, {"index": 1, "reason": "ik", "phase": "transport", "step": 7}]
    assert seen == [(p6.R, 0.0), (sixdof.twin(p6).R, 1.0), (p6.R, 2.0)]
    monkeypatch.setattr(sixdof, "reach6", lambda *a: {"reason": "collision"})
    assert sixdof.choose(None, p6, "k", "bin") == {"goal": True, "candidate": None, "failed": [{"index": k, "reason": "collision"} for k in range(3)]}
    monkeypatch.setattr(sixdof, "candidates", lambda env, p, key: [])
    assert sixdof.choose(None, p6, "k", "bin") == {"goal": False, "candidate": None, "failed": []}


@SIM
def test_a_tilted_pinch_gets_the_first_candidate_whose_path_passes():
    env, p = _sugar_box()
    p6 = sixdof.from_graspgen(_at(p, 40.0), s=p.s, width=p.width)
    ch = sixdof.choose(env, p6, "box:0:g1", "box")
    assert ch["goal"] and ch["candidate"] is not None and ch["candidate"]["index"] == len(ch["failed"])
    assert sixdof.reach6(env, sixdof.twin(p6) if ch["candidate"]["twin"] else p6, ch["candidate"]["q"], "box") is None
    again = sixdof.choose(env, p6, "box:0:g1", "box")
    assert again["candidate"]["index"] == ch["candidate"]["index"] and np.array_equal(again["candidate"]["q"], ch["candidate"]["q"])


@SIM
def test_a_pinch_in_the_table_fails_the_path_check_and_one_out_of_reach_has_no_goal():
    env, p = _sugar_box()
    low = Proposal(s=p.s, point=[p.point[0], p.point[1], 0.75 - 0.02], yaw=p.yaw, width=p.width)
    ch = sixdof.choose(env, sixdof.from_graspgen(_at(low, 40.0)), "box:0:g2", "bin")
    assert ch["goal"] and ch["candidate"] is None and ch["failed"] and all(f["reason"] in ("collision", "ik") for f in ch["failed"])
    far = Proposal(s=p.s, point=[2.5, 0.0, 0.9], yaw=p.yaw, width=p.width)
    ch = sixdof.choose(env, sixdof.from_graspgen(_at(far, 40.0)), "box:0:g3", "bin")
    assert not ch["goal"] and ch["candidate"] is None and ch["failed"] == []


@SIM
def test_the_rotation_error_counts_a_half_turn_as_a_half_turn():
    env, p = _sugar_box()
    arm = env.sim.arm
    pos, R = sixdof.hand_frame(sixdof.from_graspgen(_at(p, 40.0)))
    q, ep, er = sixdof.ik6(arm, pos, R, q_init=arm.seeds(pos)[0])
    for flip in (np.diag([-1.0, -1.0, 1.0]), np.diag([1.0, -1.0, -1.0])):
        assert abs(sixdof.pose_error(arm, q, pos, R @ flip)[1] - math.pi) < 1e-3
    assert sixdof.pose_error(arm, q, pos, R) == pytest.approx((ep, er), abs=1e-12)


# ------------------------------------------------------------------ the clearance check (ruling 1)

def test_a_closed_finger_puts_its_pad_face_on_the_width_within_the_joint_range():
    from types import SimpleNamespace

    sc = SimpleNamespace(MAX_OPEN=0.055, PAD_HALF=(0.004, 0.00875, 0.012))
    assert sixdof.finger_q(sc, 0.05) == pytest.approx(0.055 - 0.008 - 0.025)
    assert sixdof.finger_q(sc, 0.2) == 0.0
    assert sixdof.finger_q(sc, -1.0) == pytest.approx(0.055 - 0.002)


@SIM
def test_the_bodies_are_the_palm_the_wrist_and_the_two_finger_boxes_and_the_obstacles_are_by_kind():
    env, p = _sugar_box()
    sim = env.sim
    m = sim.m
    hg = sixdof.hand_geoms(sim)
    assert hg["palm"] == [m.geom("palm").id] and hg["wrist"] == [m.geom("wrist").id] and len(hg["finger"]) == 2
    assert all(m.geom_contype[g] == 0 and np.allclose(m.geom_size[g], [0.004, 0.008, 0.03]) for g in hg["finger"])
    assert np.allclose(m.jnt_range[m.joint("jl").id], [0.0, env.s["scene"].MAX_OPEN - 0.002])
    ob = sixdof.obstacle_sets(sim, "bin")
    assert ob["object"] == set(sim.hull_ids) and ob["table"] == {sim.table} and ob["destination"] == set(sim.bin_geoms)
    assert ob["fixture"] == set().union(*sim.fixture_geoms.values())
    ob = sixdof.obstacle_sets(sim, "box")
    assert ob["destination"] == sim.fixture_geoms["box"] and set(sim.bin_geoms) <= ob["fixture"] and not ob["fixture"] & sim.fixture_geoms["box"]
    assert sixdof.obstacle_sets(sim, "table_centre")["destination"] == set()


@SIM
def test_the_hand_geoms_sit_on_the_hand_pose_with_the_fingers_at_their_joint():
    import mujoco

    env, p = _sugar_box()
    sim, arm = env.sim, env.sim.arm
    q = sixdof.candidates(env, sixdof.from_graspgen(_at(p, 40.0)), "box:0:g1")[0]["q"]
    geoms = [g for gs in sixdof.hand_geoms(sim).values() for g in gs]
    for jl in (0.0, 0.02):
        loc = sixdof.local_poses(sim, geoms, jl)
        arm.kd.qpos[:] = sim.d.qpos
        arm.kd.qpos[arm.q0:arm.q0 + 7] = q
        arm.kd.qpos[sim.finger_q] = jl
        mujoco.mj_kinematics(sim.m, arm.kd)
        ph, Rh = arm.kd.xpos[arm.hand].copy(), arm.kd.xmat[arm.hand].reshape(3, 3).copy()
        for g, (lp, lR) in loc.items():
            assert np.allclose(ph + Rh @ lp, arm.kd.geom_xpos[g], atol=1e-12)
            assert np.allclose(Rh @ lR, arm.kd.geom_xmat[g].reshape(3, 3), atol=1e-12)


@SIM
def test_the_open_fingers_rest_at_open_q_past_the_soft_stop():
    env, _ = _sugar_box()
    sim, sc = env.sim, env.s["scene"]
    sim.grip(-20.0)                                   # run_grasp's open command, given before the free move
    for _ in range(int(0.5 * sc.CTRL_HZ)):
        sim.step_ctrl()
    assert np.allclose(sim.d.qpos[sim.finger_q], sixdof.OPEN_Q, atol=2e-5)
    assert 2 * (sc.MAX_OPEN - sixdof.OPEN_Q) == pytest.approx(0.119, abs=1e-4)      # the open hand: 119.0 mm, not 110 mm


def _pinch(env, f):
    return env.proposals((f,))[0]


@SIM
def test_a_palm_inside_the_object_fails_the_clearance_against_the_object():
    env, p = _sugar_box()                   # upright, 17.5 cm tall: a vertical pinch at mid-height puts the palm inside it
    c = sixdof.clearance(env, sixdof.from_graspgen(_at(p), s=p.s, width=p.width), 0.05, "table_centre")
    assert c is not None and "palm/object" in c["pairs"] and c["by_pair"]["palm/object"]["phase"] in ("approach", "close")
    assert c["first"]["pair"] in c["pairs"] and c["by_pair"]["palm/object"]["depth"] > 0.01
    assert all(pr.split("/")[0] != "finger" or pr.split("/")[1] != "object" for pr in c["pairs"])      # the width rule's


@SIM
def test_the_clearance_of_a_pinch_is_the_clearance_on_its_tilted_path_with_each_pairs_phases():
    env, p = _sugar_box()
    p6 = sixdof.from_graspgen(_at(p), s=p.s, width=p.width)
    c = sixdof.clearance(env, p6, 0.05, "table_centre")
    assert c is not None and c == sixdof.clearance_on_path(env, sixdof.hand_path6(env, p6, 0.1, 0.5, "table_centre"), 0.05, "table_centre")
    for b in c["by_pair"].values():
        assert b["phase"] in b["phases"] and sum(x["steps"] for x in b["phases"].values()) == b["steps"]
        assert max(x["depth"] for x in b["phases"].values()) == b["depth"] > 0.0


@SIM
def test_the_finger_boxes_meet_the_object_only_when_asked_and_only_on_the_way_to_the_close():
    env, _ = _sugar_box()
    p = env.proposals((0.9,), turn=-math.pi / 2)[0]          # the box's 6 cm side, 1.8 cm below its top: the palm is clear
    path = [(ph, pos, sixdof._rz(y)) for ph, pos, y in env.hand_path(p, 0.15, 0.5, "table_centre")]
    assert sixdof.clearance_on_path(env, path, p.width, "table_centre", fingers_on_object=True) is None   # closed on the pinch
    narrow = sixdof.clearance_on_path(env, path, 0.02, "table_centre", fingers_on_object=True)           # closed on 2 cm: inside
    assert "finger/object" in narrow["pairs"] and set(narrow["by_pair"]["finger/object"]["phases"]) == {"close"}
    assert sixdof.clearance_on_path(env, path, 0.02, "table_centre") is None                             # the default skips them


@SIM
def test_the_closed_fingers_can_sit_each_on_its_own_side():
    env, _ = _sugar_box()
    sc = env.s["scene"]
    p = env.proposals((0.9,), turn=-math.pi / 2)[0]
    path = [(ph, pos, sixdof._rz(y)) for ph, pos, y in env.hand_path(p, 0.15, 0.5, "table_centre")]
    lo, hi = env.pinch_extent(p.point, p.yaw)
    q_l, q_r = lo + sc.MAX_OPEN - 2 * sc.PAD_HALF[0], sc.MAX_OPEN - 2 * sc.PAD_HALF[0] - hi    # each pad on its side's face
    assert sixdof.clearance_on_path(env, path, p.width, "table_centre", fingers_on_object=True, closed_q=(q_l, q_r)) is None
    c = sixdof.clearance_on_path(env, path, p.width, "table_centre", fingers_on_object=True, closed_q=(q_l + 0.01, q_r - 0.01))
    b = c["by_pair"]["finger/object"]               # the left pad 1 cm into the box: its finger box up to 2 mm into it
    assert set(b["phases"]) == {"close"} and 0.0005 < b["depth"] < 0.0025


@SIM
def test_a_pinch_near_the_top_set_down_on_the_table_is_clear():
    env, _ = _sugar_box()
    p = _pinch(env, 0.9)
    assert sixdof.clearance(env, sixdof.from_graspgen(_at(p), s=p.s, width=p.width), 0.05, "table_centre") is None


@SIM
def test_the_fingers_are_closed_on_the_width_from_the_close_to_the_lower_and_open_otherwise(monkeypatch):
    env, p = _sugar_box()
    seen = {}

    def fake_local(sim, geoms, jl):          # open: the bodies 0.5 m above the hand; closed: 0.5 m below it, into the table
        seen[jl > 0] = jl
        return {g: (np.array([0.0, 0.0, -0.5 if jl > 0 else 0.5]), np.eye(3)) for g in geoms}

    phases = ("wait", "approach", "close", "lift", "hold", "transport", "lower", "release", "retreat")
    P = np.array([-0.3, 0.15, env.s["scene"].TABLE_TOP + 0.49])
    monkeypatch.setattr(sixdof, "local_poses", fake_local)
    monkeypatch.setattr(sixdof, "hand_path6", lambda *a, **k: [(ph, P, np.eye(3)) for ph in phases])
    c = sixdof.clearance(env, sixdof.from_graspgen(_at(p)), 0.02, "table_centre")
    assert seen == {False: sixdof.OPEN_Q, True: pytest.approx(sixdof.finger_q(env.s["scene"], 0.02))}
    t = c["by_pair"]["palm/table"]
    assert (t["phase"], t["step"], t["steps"]) == ("close", 2, 5) and c["first"]["pair"] == "palm/table"
    seen.clear()
    sixdof.clearance_on_path(env, [(ph, P, np.eye(3)) for ph in phases], 0.02, "table_centre", open_q=0.0)
    assert seen[False] == 0.0                # a caller can place the open fingers elsewhere (the palm audit: at the stop)


# ------------------------------------------------------------------ width and centring on the tilted closing axis (unit c)

def _box(l, w, h, centre=(0.0, 0.0, 0.0)):
    return np.array([[x, y, z] for x in (-l / 2, l / 2) for y in (-w / 2, w / 2) for z in (-h / 2, h / 2)]) + np.asarray(centre)


def _to_vertical(hulls, p):
    """The hulls moved rigidly so the pinch ``p`` becomes a vertical pinch at yaw 0 with its pad centre at the origin (the
    lab point 4 mm below it): v -> diag(1, -1, -1) R^T (v - tcp)."""
    M = np.diag([1.0, -1.0, -1.0]) @ np.asarray(p.R, dtype=float).T
    return [(np.asarray(V, dtype=float) - np.asarray(p.tcp, dtype=float)) @ M.T for V in hulls], np.array([0.0, 0.0, -gg.PAD_ABOVE])


def test_the_slab_the_column_and_the_open_pads_of_a_pinch_turn_with_its_frame():
    from rpent.skills.pick_place import features
    from rpent.skills.pick_place.features import PAD_HALF

    rng = np.random.default_rng(3)
    for _ in range(12):
        p = sixdof.from_graspgen(_random_within(rng, 60.0))
        tcp = np.asarray(p.tcp)
        hulls = [rng.normal(size=(14, 3)) * 0.03 + tcp + rng.normal(size=3) * 0.01, _box(0.05, 0.02, 0.04, tcp + rng.normal(size=3) * 0.02)]
        hv, pt = _to_vertical(hulls, p)
        assert sixdof.pinch_extent6(hulls, p) == pytest.approx(features.pinch_extent(hv, pt, 0.0, PAD_HALF), abs=1e-7)
        assert sixdof.column_extent6(hulls, p) == pytest.approx(features.column_extent(hv, pt, 0.0, PAD_HALF), abs=1e-7, nan_ok=True)
        assert sixdof.bounded_width6(hulls, p) == pytest.approx(gg.bounded_width(hv, pt, 0.0), abs=1e-7, nan_ok=True)
        assert sixdof.pads_free6(hulls, p) == gg.pads_free(hv, pt, 0.0)
        lo, hi = sixdof.pinch_extent6(hulls, p)
        assert sixdof.pinch_width6(hulls, p) == pytest.approx(hi - lo, nan_ok=True)


def test_a_vertical_pinchs_slab_is_the_labs_slab_at_its_mapped_point_and_yaw():
    from rpent.skills.pick_place import features
    from rpent.skills.pick_place.features import PAD_HALF

    rng = np.random.default_rng(4)
    hulls = [rng.normal(size=(12, 3)) * 0.03 + [0.1, 0.2, 0.9], _box(0.06, 0.03, 0.05, (0.11, 0.2, 0.9))]
    for yaw in (0.0, 0.7, -2.0):
        p = sixdof.from_graspgen(_tilted(0.0, yaw=yaw, t=(0.1, 0.2, 0.9 + gg.DEPTH)))
        q = sixdof.as_vertical(p)
        assert sixdof.pinch_extent6(hulls, p) == pytest.approx(features.pinch_extent(hulls, q.point, q.yaw, PAD_HALF), abs=1e-7)
        assert sixdof.column_extent6(hulls, p) == pytest.approx(features.column_extent(hulls, q.point, q.yaw, PAD_HALF), abs=1e-7)


def test_the_closing_axis_is_the_pinchs_own_when_it_leaves_the_horizontal():
    # a 6 cm cube; the closing axis 30° above the horizontal, the approach square to it: the slab (2.4 cm thick along the
    # approach) holds the cube's far edges (x = z = +-3 cm), so each end lies 3 cm (cos 30° + sin 30°) out along the axis
    a = math.radians(30.0)
    x = np.array([math.cos(a), 0.0, math.sin(a)])
    z = np.array([math.sin(a), 0.0, -math.cos(a)])
    g = np.eye(4)
    g[:3, 0], g[:3, 1], g[:3, 2] = x, np.cross(z, x), z
    p = sixdof.from_graspgen(g, depth=0.0)
    lo, hi = sixdof.pinch_extent6([_box(0.06, 0.06, 0.06)], p)
    assert hi == pytest.approx(0.03 * (math.cos(a) + math.sin(a)), abs=1e-9) and lo == pytest.approx(-hi, abs=1e-9)
    assert sixdof.pinch_width6([_box(0.06, 0.06, 0.06)], p) > 0.06 + 0.02        # a level-axis rule would read 6 cm


class WidthEnv:
    """The hulls and the lab's widest pinch."""

    max_pinch_width = 0.09

    def __init__(self, hulls):
        self.hulls = hulls

    def _hull_vertex_list(self):
        return self.hulls


def test_the_width_rules_on_a_tilted_pinch_give_the_pilots_reasons():
    from rpent.skills.pick_place.menu import OFF_OBJECT, TOO_WIDE

    from robots.mujoco_lab.curve_poseset import OFF_CENTRE, OFF_CENTRE_COLUMN

    p = sixdof.from_graspgen(_tilted(40.0, yaw=0.0, t=(0.0, 0.0, 0.0)), depth=0.0)     # the pad centre at the origin
    R = np.asarray(p.R)
    ex, a = R[:, 0], R[:, 2]

    def slab_box(lo, hi, back=0.0, depth=0.02):
        """A box along the closing axis from lo to hi, ``depth`` deep along the approach, its centre ``back`` behind the pad."""
        c = 0.5 * (lo + hi) * ex - back * a
        corners = [c + s1 * 0.5 * (hi - lo) * ex + s2 * 0.01 * R[:, 1] + s3 * 0.5 * depth * a
                   for s1 in (-1, 1) for s2 in (-1, 1) for s3 in (-1, 1)]
        return np.array(corners)

    ok = sixdof.width_rule6(WidthEnv([slab_box(-0.02, 0.02)]), p)
    assert ok["width"] == pytest.approx(0.04) and ok["why"] is None and ok["centred"] and ok["width_b"] == pytest.approx(0.04)
    assert ok["pads_free"] and ok["why_b"] is None and ok["slab"] == pytest.approx([-0.02, 0.02])
    wide = sixdof.width_rule6(WidthEnv([slab_box(-0.06, 0.06)]), p)    # 12 cm: 9.4 cm between the open pads, both too wide
    assert wide["why"] == TOO_WIDE and wide["why_b"] == TOO_WIDE and wide["width_b"] == pytest.approx(2 * gg.OPEN_GAP)
    long = sixdof.width_rule6(WidthEnv([slab_box(-0.02, 0.08)]), p)     # 10 cm: too wide; 6.7 cm between the pads, a pad lands
    assert long["why"] == TOO_WIDE and long["width_b"] == pytest.approx(0.067) and long["why_b"] == gg.PAD_LANDS
    gone = sixdof.width_rule6(WidthEnv([slab_box(-0.02, 0.02) + 0.5]), p)
    assert gone["why"] == OFF_OBJECT and gone["why_b"] == OFF_OBJECT
    off = sixdof.width_rule6(WidthEnv([slab_box(-0.01, 0.05)]), p)                # 6 cm wide, one end 5 cm out: past the 4.5 cm
    assert off["why"] == OFF_CENTRE and off["width"] == pytest.approx(0.06) and not off["centred"]
    col = sixdof.width_rule6(WidthEnv([slab_box(-0.02, 0.02), slab_box(-0.06, 0.06, back=0.2)]), p)   # wide 20 cm back
    assert col["why"] == OFF_CENTRE_COLUMN and col["width"] == pytest.approx(0.04) and col["column"] == pytest.approx([-0.06, 0.06])
    assert col["why_b"] == gg.PAD_LANDS and not col["pads_free"]
    behind = sixdof.width_rule6(WidthEnv([slab_box(-0.02, 0.02), slab_box(-0.06, 0.06, back=-0.2)]), p)  # 20 cm past the pad
    assert behind["why"] is None and behind["why_b"] is None


# ------------------------------------------------------------------ the tilted executor (unit d): the grasp phase

def _chosen(env, p6, key="box:0:g1", destination="box"):
    """The pinch (or its twin) of the first candidate whose path passes, and that candidate's configuration."""
    ch = sixdof.choose(env, p6, key, destination)
    c = ch["candidate"]
    return (sixdof.twin(p6) if c["twin"] else p6), c["q"]


@SIM
def test_the_rotation_hand_command_at_a_yaw_rotation_is_the_sims_yaw_command():
    env, p = _sugar_box()
    sim = env.sim
    pos = np.asarray(p.point) + [0.0, 0.0, 0.3]
    sim.yaw_ref = p.yaw                       # set_hand folds the yaw toward this: the pinch itself, not its half turn
    sim.set_hand(pos, p.yaw)
    q_ref = sim.arm.q_to.copy()
    env.reset("ycb__sugar_box", seed=0)
    sixdof.set_hand6(env.sim, pos, _yaw_frame(p.yaw))
    assert np.allclose(env.sim.arm.q_to, q_ref, rtol=0.0, atol=1e-9) and env.sim.in_episode
    assert env.sim.ik_error[0] < 1e-6 and env.sim.ik_error[1] < 1e-5


@SIM
def test_the_free_move_waypoints_are_the_sims_quintic():
    env, p = _sugar_box()
    sim = env.sim
    pt = np.asarray(p.point)
    start = [pt[0], pt[1], env.s["grasp"].hand_start_z(pt) + adapter.APPROACH_M]
    plan = sim.plan_free(start, p.yaw)
    way = sixdof.quintic(sim.arm, plan["q"], env.s["scene"].CTRL_HZ)
    seen, move = [], sim.arm.move
    sim.arm.move = lambda q, n: (seen.append(np.asarray(q, dtype=float).copy()), move(q, n))[1]
    sim.free_move(start, p.yaw, plan=plan)
    assert len(way) == len(seen) == plan["steps"] and all(np.array_equal(a, b) for a, b in zip(way, seen))
    A = sixdof._armlib(sim.arm)
    assert len(sixdof.quintic(sim.arm, sim.arm.desired() + 1e-3, env.s["scene"].CTRL_HZ)) == math.ceil(A.FREE_MIN_S * env.s["scene"].CTRL_HZ)


def _record(env, out):
    def on_step(ph):
        sim = env.sim
        out.append((ph, sim.obj_pos().copy(), sim.d.xpos[sim.hand].copy(),
                    0.5 * (sim.d.geom_xpos[sim.pad_l] + sim.d.geom_xpos[sim.pad_r])))
    return on_step


@SIM
def test_at_no_tilt_the_grasp_phase_is_todays_run_grasp():
    env, p = _sugar_box()
    sim, gr, hz = env.sim, env.s["grasp"], env.s["scene"].CTRL_HZ
    pt = np.asarray(p.point)
    plan = sim.plan_free([pt[0], pt[1], gr.hand_start_z(pt) + adapter.APPROACH_M], p.yaw)
    today: list = []
    gr.run_grasp(sim, pt, p.yaw, force=12.0, v_lift=0.1, lift_h=adapter.LIFT_H, hold_s=adapter.HOLD_S,
                 on_step=_record(env, today), lift_profile=adapter.ramp_profile(adapter.LIFT_H, 0.1, hz))
    env.reset("ycb__sugar_box", seed=0)
    g = sixdof.from_graspgen(_at(p), s=p.s, width=p.width)
    if abs(math.remainder(plan["yaw_rep"] - p.yaw, 2 * math.pi)) > 1.0:        # the free move took the half turn
        g = sixdof.twin(g)
    new: list = []
    out = sixdof.grasp_phase6(env, g, sixdof.quintic(env.sim.arm, plan["q"], hz), 12.0, 0.1, _record(env, new))
    assert [x[0] for x in new] == [x[0] for x in today] and {x[0] for x in new} == {"approach", "close", "lift", "hold"}
    for k in (1, 2):
        assert max(float(np.abs(a[k] - b[k]).max()) for a, b in zip(new, today)) < 1e-6
    assert out["free_steps"] == plan["steps"] and [c["check"] for c in out["g2"]] == ["pregrasp", "grasp"]
    assert all(c["pos_m"] < adapter.REACH_TOL_M and c["rot_rad"] < adapter.REACH_TOL_RAD for c in out["g2"])


@SIM
def test_a_tilted_approach_keeps_the_pad_centre_on_the_approach_line_at_v_down():
    env, p = _sugar_box()
    g, q = _chosen(env, sixdof.from_graspgen(_at(p, 40.0), s=p.s, width=p.width))
    hz = env.s["scene"].CTRL_HZ
    rec: list = []
    sixdof.grasp_phase6(env, g, sixdof.quintic(env.sim.arm, q, hz), 12.0, 0.1, _record(env, rec))
    a, tcp = np.asarray(g.R)[:, 2], np.asarray(g.tcp)
    mids = [x[3] for x in rec if x[0] == "approach"]
    off = [float(np.linalg.norm((m - tcp) - ((m - tcp) @ a) * a)) for m in mids]
    assert max(off) < 1e-3                                             # within 1 mm of the approach line
    s = np.array([float((m - tcp) @ a) for m in mids])
    assert abs(s[-1]) < 1e-3 and float(np.max(np.diff(s))) * hz == pytest.approx(adapter.V_DOWN, rel=0.05)


@SIM
def test_a_free_move_that_ends_off_the_pregrasp_fails_the_tracking_check():
    env, p = _sugar_box()
    g, q = _chosen(env, sixdof.from_graspgen(_at(p, 40.0), s=p.s, width=p.width))
    off = np.asarray(q, dtype=float) + np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])     # 0.02 rad on joint 1: about 1 cm
    rec: list = []
    with pytest.raises(sixdof.Unreachable6) as e:
        sixdof.grasp_phase6(env, g, sixdof.quintic(env.sim.arm, off, env.s["scene"].CTRL_HZ), 12.0, 0.1, _record(env, rec))
    assert e.value.record["check"] == "pregrasp" and e.value.record["pos_m"] > adapter.REACH_TOL_M and rec == []


# ------------------------------------------------------------------ the tracer and readouts in the grasp frame (unit d)

def _turn_object(sim, axis, angle):
    """Turn the object ``angle`` rad about the world ``axis`` through its body origin, and update the kinematics."""
    import mujoco

    v = sim.m.joint("obj_free").qposadr[0]
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.array([math.cos(angle / 2), *(math.sin(angle / 2) * np.asarray(axis, dtype=float))]),
                       sim.d.qpos[v + 3:v + 7].copy())
    sim.d.qpos[v + 3:v + 7] = out
    mujoco.mj_forward(sim.m, sim.d)


@SIM
def test_the_pitch_meter_in_a_vertical_grasp_frame_is_todays():
    env, p = _sugar_box()
    pt = np.asarray(p.point)
    ex = np.array([math.cos(p.yaw), math.sin(p.yaw), 0.0])
    today = env._pitch_meter(pt, p.yaw)
    framed = env._pitch_meter(pt, p.yaw, (np.cross([0.0, 0.0, 1.0], ex), np.array([0.0, 0.0, 1.0])))
    _turn_object(env.sim, ex, 0.2)
    assert today() == framed() and today() == pytest.approx(math.sin(0.2), abs=1e-9)


@SIM
def test_the_pitch_meter_in_a_tilted_grasp_frame_reads_the_turn_about_the_closing_axis():
    env, p = _sugar_box()
    g = sixdof.from_graspgen(_at(p, 40.0))
    R = np.asarray(g.R)
    ex, up = R[:, 0], -R[:, 2]
    pitch = env._pitch_meter(np.asarray(g.tcp), 0.0, (np.cross(up, ex), up))
    assert pitch() == pytest.approx(0.0, abs=1e-12)
    _turn_object(env.sim, ex, 0.1)                                      # (ex, ey, up) is right-handed: the +ey side rises
    assert pitch() == pytest.approx(math.sin(0.1), abs=1e-9)


@SIM
def test_the_pads_contact_normals_read_along_the_grasp_frames_up():
    env, p = _sugar_box()
    out, _, _ = env.s["grasp"].run_grasp(env.sim, np.asarray(p.point), p.yaw, force=12.0, v_lift=0.1, lift_h=adapter.LIFT_H,
                                         hold_s=adapter.HOLD_S)
    assert out["label"] == "held"
    plain = env._pad_hull_sums()
    assert plain[1] > 0.0 and plain[2] > 0.0                                 # both pads press on the box
    env._pad_up = np.array([0.0, 0.0, 1.0])
    assert np.array_equal(env._pad_hull_sums(), plain)                      # the vertical frame: today's sums
    env._pad_up = np.array([math.cos(p.yaw), math.sin(p.yaw), 0.0])         # the closing axis: the normals lie along it
    tilted = env._pad_hull_sums()
    assert np.array_equal(tilted[:3], plain[:3]) and not np.array_equal(tilted[3:], plain[3:])
    env._tracer(np.asarray(p.point), p.yaw)
    assert env._pad_up is None                                               # a tracer without a frame: today's


@SIM
def test_the_tracer_reads_the_pitch_in_its_frame():
    env, p = _sugar_box()
    seen = []
    env._pitch_meter = lambda pt, yaw, frame=None: (seen.append(frame), lambda: 0.0)[1]
    frame = (np.array([0.0, 1.0, 0.0]), np.array([0.6, 0.0, 0.8]))
    env._tracer(np.asarray(p.point), 0.0, frame)
    env._tracer(np.asarray(p.point), p.yaw)
    assert seen[0] is frame and seen[1] is None


# ------------------------------------------------------------------ the tilted re-plan at the hold (rule (c), unit d)

def _cloud(p, seed=0):
    """50 points of an object below the pinch's pad centre, at rest."""
    return np.asarray(p.tcp) + np.random.default_rng(seed).uniform(-0.04, 0.04, (50, 3)) - [0.0, 0.0, 0.03]


def test_at_no_tilt_the_tilted_replan_reads_todays_d_and_raises_only_past_the_band_when_held_and_lifted():
    p = sixdof.from_graspgen(_grasp((0.0, 0.0, -1.0), (math.cos(0.4), math.sin(0.4), 0.0), (0.1, 0.2, 0.9)))
    hp, Rh = sixdof.hand_frame(p)
    V, top = _cloud(p), hp + [0.0, 0.0, adapter.LIFT_H]
    for drop, held, rose, rise in ((0.005, True, True, 0.005), (0.002, True, True, 0.0), (0.005, False, True, 0.0),
                                   (0.005, True, False, 0.0), (-0.004, True, True, 0.0)):
        Vn = V + [0.0, 0.0, adapter.LIFT_H - drop]                    # lifted with the hand, then slipped down by drop
        rec, r = sixdof.replan_at_hold6(p, V, Vn, top, Rh, [0.0, 0.5, 1.0], held, rose)
        today, _ = adapter.replan_at_hold([{"phase": "hold", "low_z": float(Vn[:, 2].min()), "hand_z": float(top[2]),
                                            "pads": [held, held]}], float(V[:, 2].min() - hp[2]), rose)
        assert rec["d_mm"] == pytest.approx(today["d_mm"], abs=1e-9) and rec["d_mm"] == pytest.approx(1e3 * drop, abs=1e-9)
        assert r == pytest.approx(rise, abs=1e-12)
        assert (rec["held"], rec["lifted"], rec["raise_mm"], rec["unreachable"], rec["frame"]) == (held, rose, 0.0, False, "tcp")


def test_the_tilted_replan_reads_the_in_hand_pose_not_the_arms_sag_or_droop():
    p = sixdof.from_graspgen(_tilted(40.0))
    hp, Rh = sixdof.hand_frame(p)
    V = _cloud(p)
    c, s = math.cos(0.05), math.sin(0.05)
    droop = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])           # the hand 0.05 rad off, 8 mm low
    hand = hp + [0.0, 0.0, adapter.LIFT_H - 0.008]
    Vn = hand + (V - hp) @ droop.T                                          # the object rigid in the drooped hand
    rec, r = sixdof.replan_at_hold6(p, V, Vn, hand, droop @ Rh, [0.0, 0.25, 0.5, 1.0], True, True)
    assert abs(rec["d_mm"]) < 1e-9 and r == 0.0


def test_the_tilted_replan_takes_the_lowest_point_over_the_return_to_vertical():
    p = sixdof.from_graspgen(_tilted(40.0))
    hp, Rh = sixdof.hand_frame(p)
    a, tcp = np.asarray(p.R)[:, 2], np.asarray(p.tcp)
    V = _cloud(p)
    fr = [0.0, 0.2, 0.6, 1.0]
    low = min(float(((V - tcp) @ sixdof.return_rotation(a, f).T)[:, 2].min()) for f in fr)
    assert sixdof.carry_low6(a, V - tcp, fr) == pytest.approx(low, abs=1e-15)
    ex = np.asarray(p.R)[:, 0]
    k = ex / np.linalg.norm(ex)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    turn = np.eye(3) + math.sin(0.3) * K + (1.0 - math.cos(0.3)) * (K @ K)    # the object turned 0.3 rad in the pads
    Vn = tcp + [0.0, 0.0, adapter.LIFT_H] + (V - tcp) @ turn.T
    rec, r = sixdof.replan_at_hold6(p, V, Vn, hp + [0.0, 0.0, adapter.LIFT_H], Rh, fr, True, True)
    d = sixdof.carry_low6(a, V - tcp, fr) - sixdof.carry_low6(a, (V - tcp) @ turn.T, fr)
    assert rec["d_mm"] == pytest.approx(1e3 * d, abs=1e-9) and r == (d if d > adapter.REPLAN_BAND_M else 0.0)


# ------------------------------------------------------------------ the tilted attempt (unit d)

def _spy(monkeypatch):
    seen, real = [], sixdof.set_hand6

    def spy(sim, pos, R):
        seen.append((np.asarray(pos, dtype=float).copy(), np.asarray(R, dtype=float).copy()))
        real(sim, pos, R)
    monkeypatch.setattr(sixdof, "set_hand6", spy)
    return seen


def _phase_runs(path, phases):
    return [x for x in path if x[0] in phases]


@SIM
def test_a_tilted_attempt_runs_hand_path6_to_the_end_of_the_transport_and_flags_its_readouts(monkeypatch):
    env, p = _sugar_box()
    g, q = _chosen(env, sixdof.from_graspgen(_at(p, 40.0), s=p.s, width=p.width))
    hz = env.s["scene"].CTRL_HZ
    path = sixdof.hand_path6(env, g, 0.1, 0.5, "box")
    free = sixdof.quintic(env.sim.arm, q, hz)
    seen = _spy(monkeypatch)
    r = sixdof.execute6(env, g, free, 12.0, 0.1, 0.5, "box")
    phases = [x["phase"] for x in r.trace]
    assert sorted(set(phases), key=phases.index) == ["approach", "close", "lift", "hold", "transport", "lower", "release", "retreat"]
    planned = _phase_runs(path, ("approach", "lift", "transport"))
    assert len(seen) > len(planned)
    for (pos, R), (_, hp, Rp) in zip(seen, planned):
        assert np.allclose(pos, hp, rtol=0.0, atol=1e-12) and np.allclose(R, Rp, rtol=0.0, atol=1e-12)
    Re = path[-1][2]
    assert all(np.allclose(R, Re, rtol=0.0, atol=1e-12) for _, R in seen[len(planned):])     # lower and retreat: returned
    rel = seen[len(planned) + phases.count("lower")][0]
    assert all(np.allclose(pos[:2], rel[:2], atol=1e-12) for pos, _ in seen[len(planned) + phases.count("lower"):])
    assert r.sixdof["readouts_valid"] is False and r.sixdof["tilt_deg"] == pytest.approx(40.0, abs=1e-9)
    assert r.sixdof["turn_deg"] == pytest.approx(sixdof.turn_deg(env, g, "box")) and [c["check"] for c in r.sixdof["g2"]] == ["pregrasp", "grasp"]
    assert r.time_s == pytest.approx((len(free) + len(r.trace)) / hz) and not hasattr(r, "replan")
    assert [s.phase for s in r.planned] == ["approach", "close", "lift", "hold", "transport", "lower", "release", "retreat"]
    for s in r.planned:
        if s.phase not in ("lower",):                                     # the lower follows the end-of-transport read
            assert phases.count(s.phase) == s.n_steps
    assert env._metering is False and getattr(env.sim, "log", None) is None
    assert np.array_equal(env._pad_up, -np.asarray(g.R)[:, 2])               # the trace read along the grasp frame's up
    tr = [x for x in r.trace if x["phase"] == "transport"][-1]                # the end-of-transport read of the lowest point
    z_rel = env._release_target_low_z(env._dest("box"), None) - (tr["low_z"] - tr["hand_z"])
    assert seen[len(planned) + phases.count("lower") - 1][0][2] == pytest.approx(z_rel, abs=1e-12)


@SIM
def test_a_tilted_raise_lifts_the_hand_straight_up_then_carries_the_planned_path_that_much_higher(monkeypatch):
    env, p = _sugar_box()
    env.replan_hold = True
    g, q = _chosen(env, sixdof.from_graspgen(_at(p, 40.0), s=p.s, width=p.width))
    hz = env.s["scene"].CTRL_HZ
    path = sixdof.hand_path6(env, g, 0.1, 0.5, "box")
    real = sixdof.replan_at_hold6
    monkeypatch.setattr(sixdof, "replan_at_hold6", lambda *a, **k: (real(*a, **k)[0], 0.007))
    seen = _spy(monkeypatch)
    r = sixdof.execute6(env, g, sixdof.quintic(env.sim.arm, q, hz), 12.0, 0.1, 0.5, "box")
    n = len(_phase_runs(path, ("approach", "lift")))
    up = adapter.ramp_profile(0.007, 0.1, hz)
    _, top, Rh = _phase_runs(path, ("lift",))[-1]
    for (pos, R), f in zip(seen[n:n + len(up)], up):
        assert np.allclose(pos, top + [0.0, 0.0, 0.007 * f], atol=1e-12) and np.allclose(R, Rh, atol=1e-12)
    carry = _phase_runs(path, ("transport",))
    for (pos, R), (_, hp, Rp) in zip(seen[n + len(up):n + len(up) + len(carry)], carry, strict=True):
        assert np.allclose(pos, hp + [0.0, 0.0, 0.007], atol=1e-12) and np.allclose(R, Rp, atol=1e-12)
    assert r.replan["raise_mm"] == pytest.approx(7.0) and r.replan["unreachable"] is False and r.replan["frame"] == "tcp"
    hold = [x for x in r.trace if x["phase"] == "hold"]
    assert r.replan["held"] == all(hold[-1]["pads"]) and r.replan["lifted"] is True
    assert [x["phase"] for x in r.trace].count("transport") == len(up) + len(carry)


@SIM
def test_a_raise_the_arm_cannot_carry_keeps_the_planned_height_and_says_so(monkeypatch):
    env, p = _sugar_box()
    env.replan_hold = True
    g, q = _chosen(env, sixdof.from_graspgen(_at(p, 40.0), s=p.s, width=p.width))
    path = sixdof.hand_path6(env, g, 0.1, 0.5, "box")
    real = sixdof.replan_at_hold6
    monkeypatch.setattr(sixdof, "replan_at_hold6", lambda *a, **k: (real(*a, **k)[0], 0.007))
    monkeypatch.setattr(sixdof, "carry_reaches6", lambda *a, **k: False)
    seen = _spy(monkeypatch)
    r = sixdof.execute6(env, g, sixdof.quintic(env.sim.arm, q, env.s["scene"].CTRL_HZ), 12.0, 0.1, 0.5, "box")
    planned = _phase_runs(path, ("approach", "lift", "transport"))
    assert all(np.allclose(pos, hp, atol=1e-12) for (pos, _), (_, hp, _) in zip(seen, planned))
    assert r.replan["unreachable"] is True and r.replan["raise_mm"] == 0.0


class _Body:
    def __init__(self, pos, R):
        self.xpos, self.xmat = {7: np.asarray(pos, dtype=float)}, {7: np.asarray(R, dtype=float).ravel()}


class _Sim:
    hand = 7

    def __init__(self, pos, R):
        self.d = _Body(pos, R)


def test_the_tracking_check_measures_the_tcp_and_the_hands_turn():
    p = sixdof.from_graspgen(_tilted(40.0))
    hp, Rh = sixdof.hand_frame(p)
    tcp = np.asarray(p.tcp)
    k = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    for ang, ok in ((0.005, True), (0.02, False)):
        T = np.eye(3) + math.sin(ang) * K + (1.0 - math.cos(ang)) * (K @ K)     # the hand turned about the TCP
        c = sixdof.tracking(_Sim(tcp + T @ (hp - tcp), T @ Rh), "grasp", hp, Rh)
        assert c["pos_m"] < 1e-12 and c["rot_rad"] == pytest.approx(ang, abs=1e-12) and c["ok"] is ok and c["check"] == "grasp"
    c = sixdof.tracking(_Sim(hp + [0.0, 0.0, 0.002], Rh), "pregrasp", hp, Rh)
    assert c["pos_m"] == pytest.approx(0.002, abs=1e-12) and c["rot_rad"] == 0.0 and c["ok"] is False


@SIM
def test_the_raised_carry_check_tests_the_poses_the_attempt_commands(monkeypatch):
    env, p = _sugar_box()
    g = sixdof.from_graspgen(_at(p, 40.0), s=p.s, width=p.width)
    hz = env.s["scene"].CTRL_HZ
    path = sixdof.hand_path6(env, g, 0.1, 0.5, "box")
    _, top, Rh = _phase_runs(path, ("lift",))[-1]
    want = [(top + [0.0, 0.0, 0.007 * f], Rh) for f in adapter.ramp_profile(0.007, 0.1, hz)]
    want += [(hp + [0.0, 0.0, 0.007], Rp) for _, hp, Rp in _phase_runs(path, ("transport",))]
    seen = []
    monkeypatch.setattr(sixdof, "ik6", lambda arm, pos, R, q_init=None: (seen.append((np.asarray(pos), np.asarray(R))), np.asarray(q_init), 0.0, 0.0)[1:])
    assert sixdof.carry_reaches6(env, g, 0.007, 0.1, 0.5, "box") is True
    assert len(seen) == len(want)
    assert all(np.allclose(a, b, atol=1e-12) and np.allclose(Ra, Rb, atol=1e-12) for (a, Ra), (b, Rb) in zip(seen, want))


@SIM
def test_the_raised_carry_check_passes_the_chosen_path_and_fails_a_raise_out_of_reach():
    env, p = _sugar_box()
    g, q = _chosen(env, sixdof.from_graspgen(_at(p, 40.0), s=p.s, width=p.width))
    hz = env.s["scene"].CTRL_HZ
    sixdof.grasp_phase6(env, g, sixdof.quintic(env.sim.arm, q, hz), 12.0, 0.1)
    q_to = env.sim.arm.q_to.copy()
    assert sixdof.carry_reaches6(env, g, 0.0, 0.1, 0.5, "box") is True
    assert sixdof.carry_reaches6(env, g, 1.0, 0.1, 0.5, "box") is False
    assert np.array_equal(env.sim.arm.q_to, q_to)                            # a check only: nothing moves


@SIM
def test_a_tilted_attempt_that_fails_the_tracking_check_stops_its_meters_and_raises():
    env, p = _sugar_box()
    g, q = _chosen(env, sixdof.from_graspgen(_at(p, 40.0), s=p.s, width=p.width))
    off = np.asarray(q, dtype=float) + np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(sixdof.Unreachable6):
        sixdof.execute6(env, g, sixdof.quintic(env.sim.arm, off, env.s["scene"].CTRL_HZ), 12.0, 0.1, 0.5, "box")
    assert env._metering is False and env._dmeter is None and getattr(env.sim, "log", None) is None


@SIM
def test_a_tilted_pinch_that_lifts_nothing_opens_at_the_lift_top_and_backs_out_along_the_approach(monkeypatch):
    env, p = _sugar_box()
    env.replan_hold = True
    g = sixdof.from_graspgen(_at(p, 40.0), s=p.s, width=p.width)
    g, q = _chosen(env, sixdof.Grasp6(s=g.s, tcp=list(np.asarray(g.tcp) + [0.0, 0.0, 0.15]), R=g.R, width=g.width))   # in the air
    path = sixdof.hand_path6(env, g, 0.1, 0.5, "box")
    seen = _spy(monkeypatch)
    r = sixdof.execute6(env, g, sixdof.quintic(env.sim.arm, q, env.s["scene"].CTRL_HZ), 12.0, 0.1, 0.5, "box")
    phases = [x["phase"] for x in r.trace]
    assert sorted(set(phases), key=phases.index) == ["approach", "close", "lift", "hold", "release", "retreat"]
    assert r.label == "not_lifted"
    assert (r.replan["held"], r.replan["lifted"], r.replan["raise_mm"]) == (False, False, 0.0)
    n = len(_phase_runs(path, ("approach", "lift")))
    _, top, Rh = _phase_runs(path, ("lift",))[-1]
    a = np.asarray(g.R)[:, 2]
    back = adapter.ramp_profile(adapter.RETREAT_M, adapter.V_LOWER, env.s["scene"].CTRL_HZ)
    assert len(seen) == n + len(back)
    for (pos, R), f in zip(seen[n:], back):
        assert np.allclose(pos, top - adapter.RETREAT_M * f * a, atol=1e-12) and np.allclose(R, Rh, atol=1e-12)


@SIM
def test_at_no_tilt_the_tilted_plan_is_todays_plan():
    env, p = _sugar_box()
    g = sixdof.from_graspgen(_at(p), s=p.s, width=p.width)
    a = sixdof.plan_trajectory6(env, g, 0.1, 0.5, "box")
    b = env.plan_trajectory(sixdof.as_vertical(g), 0.1, 0.5, "box")
    assert [(s.phase, s.n_steps, s.has_surface, s.speed) for s in a] == [(s.phase, s.n_steps, s.has_surface, s.speed) for s in b]
    assert np.allclose([[s.z_start, s.z_end] for s in a], [[s.z_start, s.z_end] for s in b], rtol=0.0, atol=1e-9)


@SIM
def test_the_tilted_plan_counts_hand_path6s_steps_and_ends_the_transport_at_the_returned_lowest_point():
    env, p = _sugar_box()
    g = sixdof.from_graspgen(_at(p, 40.0), s=p.s, width=p.width)
    plan = sixdof.plan_trajectory6(env, g, 0.1, 0.5, "box")
    path = sixdof.hand_path6(env, g, 0.1, 0.5, "box")
    hz = env.s["scene"].CTRL_HZ
    n = {s.phase: s.n_steps for s in plan}
    for ph in ("approach", "close", "lift", "hold", "transport", "lower", "release"):
        assert n[ph] == sum(x[0] == ph for x in path)
    assert n["retreat"] == sum(x[0] == "retreat" for x in path) + int(adapter.RETREAT_SETTLE_S * hz)
    a, tcp = np.asarray(g.R)[:, 2], np.asarray(g.tcp)
    V = np.concatenate(env._hull_vertex_list()) - tcp
    Q = sixdof._rz(math.radians(sixdof.turn_deg(env, g, "box"))) @ sixdof.return_rotation(a, 1.0)
    dest = env._dest("box")
    low_end = tcp[2] + adapter.LIFT_H + float((V @ Q.T)[:, 2].min())
    tr = next(s for s in plan if s.phase == "transport")
    assert tr.z_end == pytest.approx(low_end - float(dest["surface_z"]), abs=1e-12)
