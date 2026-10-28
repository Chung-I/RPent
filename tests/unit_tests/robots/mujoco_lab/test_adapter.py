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

"""MuJoCo lab adapter: grasp, transport, set-down and the outcome labels (needs the sim)."""

import os

import numpy as np

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")

LONG = "ycb__banana"  # 19.6 cm: fits plate, mouse pad, bin, table; too long for the 16 cm box
SHORT = "ycb__brick"  # 7.8 cm block, control grasp held 10/10 in t3 (a sphere like the orange is squeezed out of the pads)


def _env(**kw):
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    return MuJoCoLabEnv(**kw)


def _control(env, object_id=LONG, **theta):
    env.reset(object_id, theta or None, seed=0)
    return env.proposals()[1]


def test_control_grasp_on_banana_is_placed_in_bin():
    env = _env()
    p = _control(env)
    assert len(env.proposals()) == 3 and 0.03 < p.width < 0.05
    r = env.execute(p, force=20.0, v_lift=0.15)
    assert r.label == "placed_stable" and r.zone == "bin"


@pytest.mark.parametrize(
    "dest,expect",
    [("table_centre", "placed_stable"), ("mouse_pad", "placed_stable"), ("plate", "placed_stable"), ("table_gap", "rolled_off")],
)
def test_set_down_per_destination(dest, expect):
    env = _env()
    p = _control(env)
    assert env.execute(p, force=20.0, v_lift=0.15, destination=dest).label == expect


def test_short_object_into_box_is_placed():
    env = _env()
    from robots.mujoco_lab.adapter import object_length

    assert object_length(SHORT) < 0.12
    p = _control(env, SHORT)
    r = env.execute(p, force=20.0, v_lift=0.15, destination="box")
    assert r.label == "placed_stable" and r.zone == "box"


def test_long_object_into_box_is_never_placed_stable():
    env = _env()
    p = _control(env)
    r = env.execute(p, force=20.0, v_lift=0.15, destination="box")
    assert r.label in ("missed", "tipped", "dropped_low")   # the rim strips a 19.6 cm object out of the pads or it bridges the opening


def test_unknown_destination_raises():
    env = _env()
    p = _control(env)
    with pytest.raises(ValueError, match="destination"):
        env.execute(p, 20.0, 0.15, destination="shelf")


def test_slippery_heavy_grasp_drops_and_records_loss():
    env = _env()
    p = _control(env, mu=0.05, mass_scale=3.0, com_s=None, v_lift=0.15)
    r = env.execute(p, force=10.0, v_lift=0.15)
    assert r.label in ("not_lifted", "dropped_low", "dropped_high")
    if r.label != "not_lifted":
        assert r.loss_phase in ("lift", "hold", "transport", "lower")
        assert r.loss_speed is not None and r.loss_step is not None and r.drop_z is not None


def test_pinch_beside_object_is_not_lifted():
    from robots.mujoco_lab.adapter import Proposal

    env = _env()
    p = _control(env)
    beside = Proposal(s=p.s, point=[p.point[0], p.point[1] + 0.15, p.point[2]], yaw=p.yaw, width=p.width)
    r = env.execute(beside, force=20.0, v_lift=0.15)
    assert r.label == "not_lifted" and r.drop_z is None and r.loss_phase is None


def test_intended_release_is_not_a_loss():
    env = _env()
    p = _control(env)
    r = env.execute(p, force=20.0, v_lift=0.15, destination="table_centre")
    assert r.label == "placed_stable" and r.loss_phase is None and r.release is not None


def test_plan_trajectory_matches_execution():
    env = _env()
    p = _control(env)
    planned = env.plan_trajectory(p, 0.15, 0.5, "plate")
    phases = [s.phase for s in planned]
    assert phases == ["approach", "close", "lift", "hold", "transport", "lower", "release", "retreat"]
    r = env.execute(p, force=20.0, v_lift=0.15, v_transport=0.5, destination="plate")
    total = sum(s.n_steps for s in planned)
    assert abs(total - len(r.trace)) <= 0.05 * total   # the held object hangs a little differently from its rest pose
    assert [s.phase for s in r.planned] == phases


def test_com_end_negative_moves_com_to_other_end():
    env = _env()
    env.reset(LONG, {"mu": 0.8, "mass_scale": 1.0, "com_s": 0.8, "v_lift": 0.15}, seed=0, com_end=+1)
    hi = env.truth()["com_s"]
    env.reset(LONG, {"mu": 0.8, "mass_scale": 1.0, "com_s": 0.8, "v_lift": 0.15}, seed=0, com_end=-1)
    lo = env.truth()["com_s"]
    assert hi > 0.65 and lo < 0.5


def test_locate_geometry_and_truth():
    env = _env()
    env.reset(LONG)
    loc = env.locate("banana")
    assert loc["object_id"] == LONG and env.locate("spoon") is None
    assert loc["volume"] > 0 and loc["material"] == "fruit" and loc["hollow"] is False
    g = env.geometry()
    assert 0.15 < g.length < 0.25 and g.com_height > 0 and g.footprint_half_width >= g.base_extent > 0
    assert 0.1 < env.truth()["mass"] < 0.3


def test_unset_sim_dir_raises(monkeypatch):
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    monkeypatch.delenv("RPENT_MUJOCO_SIM_DIR")
    with pytest.raises(RuntimeError, match="RPENT_MUJOCO_SIM_DIR"):
        MuJoCoLabEnv()


def _mesh_ids():
    from rpent.skills.pick_place.destinations import MESH_IDS

    return list(MESH_IDS)


@pytest.mark.skipif(not _mesh_ids(), reason="no inspected mesh fixtures in fixtures.json")
@pytest.mark.parametrize("fixture", [f for f in ("plate_small", "serving_bowl", "bin_b03", "tray_b02") if f in _mesh_ids()] or ["none"])
def test_set_down_on_mesh_fixture(fixture):
    if fixture == "none":
        pytest.skip("none of the four reference fixtures is inspected")
    env = _env()
    env.reset(SHORT, fixture=fixture)
    assert fixture in env.fixtures_in_scene
    r = env.execute(env.proposals()[1], force=20.0, v_lift=0.15, destination=fixture)
    assert r.label in ("placed_stable", "tipped", "missed", "dropped_low")
    if r.label in ("placed_stable", "tipped"):
        assert r.zone == fixture


@pytest.mark.skipif(not _mesh_ids(), reason="no inspected mesh fixtures in fixtures.json")
def test_mesh_fixture_not_in_scene_is_error():
    env = _env()
    env.reset(SHORT)
    with pytest.raises(ValueError, match="fixture"):
        env.execute(env.proposals()[1], 20.0, 0.15, destination=_mesh_ids()[0])
    with pytest.raises(ValueError, match="fixture"):
        env.reset(SHORT, fixture="not_a_fixture")


def test_hand_retreats_after_release_and_pads_let_go():
    env = _env()
    p = _control(env, SHORT)
    r = env.execute(p, force=20.0, v_lift=0.15, destination="table_centre")
    assert r.label == "placed_stable"
    assert [s.phase for s in r.planned][-1] == "retreat" and r.trace[-1]["phase"] == "retreat"
    assert not any(r.trace[-1]["pads"])
    hand_z = env.sim.d.xpos[env.sim.hand][2]
    assert hand_z > r.trace[-1]["low_z"] + 0.08


def test_geometry_com_is_nominal_not_the_hidden_shift():
    env = _env()
    env.reset(LONG, {"mu": 0.8, "mass_scale": 1.0, "com_s": None, "v_lift": 0.15})
    body_plain = env.sim.obj_mat().T @ (env.geometry().com - env.sim.obj_pos())      # nominal CoM in the object frame
    env.reset(LONG, {"mu": 0.8, "mass_scale": 1.0, "com_s": 0.8, "v_lift": 0.15})
    body_shifted = env.sim.obj_mat().T @ (env.geometry().com - env.sim.obj_pos())
    truth = env.truth()
    assert np.linalg.norm(body_plain - body_shifted) < 1e-6                        # the hidden shift is not in the geometry
    assert np.linalg.norm(np.asarray(truth["com"]) - env.geometry().com) > 0.02    # but it is in the truth


def test_trace_records_pad_forces_during_the_hold():
    env = _env()
    env.reset(SHORT, {"mu": 0.8, "mass_scale": 1.0, "com_s": None, "v_lift": 0.15}, seed=0)
    f_std = env.s["t3_make"].std_force(env.nominal_mass())
    r = env.execute(env.proposals()[1], force=f_std, v_lift=0.15, v_transport=0.5, destination="table_centre")
    assert all({"fz", "fn", "pitch_ey"} <= set(x) for x in r.trace)
    hold = [x["fz"] for x in r.trace if x["phase"] == "hold"]
    m = env.truth()["mass"]
    assert hold and 0.8 <= float(np.median(hold)) / (m * 9.81) <= 1.25
    assert r.readouts["rose"] is True and r.readouts["fz_hold_n"] is not None


def _roll_about_long_axis(env, deg=90.0):
    """Turn the object onto another face (a knock-over without the randomness of a fall) and let it settle."""
    import mujoco

    sim = env.sim
    R = sim.obj_mat()
    a = R @ np.asarray(env.props["axis"], dtype=float)
    a /= np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    th = np.radians(deg)
    Rn = (np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K) @ R
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, Rn.ravel())
    adr, dof = sim.m.joint("obj_free").qposadr[0], sim.m.joint("obj_free").dofadr[0]
    sim.d.qpos[adr + 3:adr + 7] = q
    sim.d.qvel[dof:dof + 6] = 0
    mujoco.mj_forward(sim.m, sim.d)
    sim.d.qpos[adr + 2] += env.s["scene"].TABLE_TOP + 0.003 - sim.obj_low_z()
    mujoco.mj_forward(sim.m, sim.d)
    sim.settle(1.5)


def test_retry_after_the_object_fell_plans_from_the_new_pose():
    env = _env()
    env.reset("ycb__spam_can", {"mu": 0.8, "mass_scale": 1.0, "com_s": None, "v_lift": 0.15}, seed=0)
    sim = env.sim
    up0, h0 = sim.obj_mat()[:, 2].copy(), float(sim.d.subtree_com[sim.obj][2] - sim.obj_low_z())
    _roll_about_long_axis(env)
    up1, h1 = sim.obj_mat()[:, 2], float(sim.d.subtree_com[sim.obj][2] - sim.obj_low_z())
    assert np.degrees(np.arccos(np.clip(up0 @ up1, -1, 1))) > 60 and abs(h1 - h0) > 0.2 * h0   # another face, another CoM height
    p = env.proposals()[1]
    lift = next(s for s in env.plan_trajectory(p, 0.15, 0.5, "table_centre") if s.phase == "lift")
    surf, _ = env._surface_below(np.asarray(p.point[:2]))
    assert abs(lift.z_start - (sim.obj_low_z() - surf)) < 0.005
    f_std = env.s["t3_make"].std_force(env.nominal_mass())
    r = env.execute(p, force=2 * f_std, v_lift=0.15, v_transport=0.5, destination="table_centre")
    assert r.label == "placed_stable"                                           # not "tipped" against the old face
