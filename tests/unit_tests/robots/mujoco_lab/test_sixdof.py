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


@SIM
def test_at_no_tilt_the_tilted_hand_path_is_todays():
    env, p = _sugar_box()
    p6 = sixdof.from_graspgen(_at(p), s=p.s, width=p.width)
    a = sixdof.hand_path6(env, p6, 0.15, 0.5, "bin")
    b = env.hand_path(sixdof.as_vertical(p6), 0.15, 0.5, "bin")
    assert [x[0] for x in a] == [x[0] for x in b]
    assert all(np.allclose(x[1], y[1], rtol=0.0, atol=1e-12) and np.allclose(x[2], _yaw_frame(y[2]), atol=1e-12) for x, y in zip(a, b))


@SIM
def test_a_tilted_hand_path_approaches_along_the_approach_lifts_straight_up_and_retreats_back_along_it():
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
    assert all(np.array_equal(x[2], Rh) for x in path)
    assert np.allclose(by["wait"][0][1], hp - adapter.APPROACH_M * a)
    s = [float((x[1] - by["wait"][0][1]) @ a) for x in by["approach"]]
    assert all(np.allclose(x[1], by["wait"][0][1] + si * a) for x, si in zip(by["approach"], s)) and s == sorted(s)
    assert np.array_equal(by["approach"][-1][1], hp) and all(np.array_equal(x[1], hp) for x in by["close"])
    assert all(np.allclose(x[1][:2], hp[:2]) for x in by["lift"]) and by["lift"][-1][1][2] == pytest.approx(hp[2] + adapter.LIFT_H)
    goal = np.asarray(env._dest("bin")["xy"]) + (hp - np.asarray(p6.tcp))[:2]
    assert np.allclose(by["transport"][-1][1][:2], goal)
    rel = by["release"][0][1]
    assert np.allclose(by["retreat"][-1][1], rel - adapter.RETREAT_M * a)


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
@pytest.mark.xfail(strict=True, reason="the transport with the rotation kept has no IK solution above the bin (0 of 200 seeds at 20 and 40 "
                   "degrees, both twins); design question sent 2026-11-09")
def test_a_tilted_pinch_gets_the_first_candidate_whose_path_passes():
    env, p = _sugar_box()
    p6 = sixdof.from_graspgen(_at(p, 40.0), s=p.s, width=p.width)
    ch = sixdof.choose(env, p6, "box:0:g1", "bin")
    assert ch["goal"] and ch["candidate"] is not None and ch["candidate"]["index"] == len(ch["failed"])
    assert sixdof.reach6(env, sixdof.twin(p6) if ch["candidate"]["twin"] else p6, ch["candidate"]["q"], "bin") is None
    again = sixdof.choose(env, p6, "box:0:g1", "bin")
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
