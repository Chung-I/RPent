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


def test_at_rest_needs_a_still_window():
    """At rest: over the last REST_WINDOW_S the object moved under 1 mm and turned under 1 deg."""
    from robots.mujoco_lab.adapter import REST_WINDOW_S, at_rest

    hz = 50
    w = int(REST_WINDOW_S * hz)
    eye = np.eye(3)
    still = [(np.zeros(3), eye)] * (w + 1)
    assert at_rest(still, hz)
    assert not at_rest(still[:w], hz)                                                   # shorter than the window
    assert not at_rest([(np.array([0.002 * k / hz, 0.0, 0.0]), eye) for k in range(w + 1)], hz)   # creeps 2 mm/s

    def rz(deg):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    assert not at_rest([(np.zeros(3), rz(2.0 * k / w)) for k in range(w + 1)], hz)      # turns 2 deg in the window
    rock = [(np.array([0.003 * np.sin(2 * np.pi * k / w), 0.0, 0.0]), eye) for k in range(w + 1)]
    assert not at_rest(rock, hz)                                    # rocks 3 mm and is back where it was at the window's ends


def test_reset_leaves_the_object_at_rest_near_the_rest_xy():
    """Case (11, lizard, 0): the lizard's rest-pose drop drifts 26 cm. The old reset placed it at that drift and
    returned while it tumbled off the table edge (case (11, lizard, 5) until the calibrated torsion, which
    now rocks: see the next test)."""
    from robots.mujoco_lab.adapter import OBJECT_REST_XY, object_props
    from robots.mujoco_lab.calibrate import case_draw

    env = _env()
    obj = "hot3d__lizard_figurine"
    theta, com_end, _ = case_draw(11, obj, 0, object_props(obj)["mass"])
    env.reset(obj, theta, seed=11000, com_end=com_end)
    info = env.reset_info
    assert info["at_rest"] and info["zone"] in ("table", *env.fixtures_in_scene) and 0.2 <= info["settle_s"] <= 3.0
    p0 = env.sim.obj_pos().copy()
    assert np.linalg.norm(p0[:2] - np.array(OBJECT_REST_XY)) < 0.05
    for _ in range(50):
        env.sim.step_ctrl()
    assert np.linalg.norm(env.sim.obj_pos() - p0) < 0.002



def test_a_reset_that_does_not_reach_rest_says_so():
    """Case (11, lizard, 5): with the hulls' torsional friction at 3 mm times mu the lizard rocks on the table and
    does not settle (an at-rest artifact of the native collider; the corpus leaves such rows out)."""
    from robots.mujoco_lab.adapter import REST_CAP_S, object_props
    from robots.mujoco_lab.calibrate import case_draw

    env = _env()
    obj = "hot3d__lizard_figurine"
    theta, com_end, _ = case_draw(11, obj, 5, object_props(obj)["mass"])
    env.reset(obj, theta, seed=11005, com_end=com_end)
    assert env.reset_info == {"settle_s": REST_CAP_S, "at_rest": False, "zone": "table"}

def test_reset_puts_an_object_that_left_the_table_in_its_drop_on_the_table():
    """Corpus draw (0, pitcher, 9): under its own shifted CoM the pitcher topples in the rest-pose drop
    and rolls off the drop's table, so the drop's height is the floor's."""
    from robots.mujoco_lab.adapter import object_props
    from robots.mujoco_lab.corpus import case_rng, draw_theta

    env = _env()
    obj = "ycb__pitcher"
    theta, com_end = draw_theta(case_rng(0, obj, 9), object_props(obj)["mass"])
    env.reset(obj, theta, seed=9, com_end=com_end)
    assert env.reset_info["zone"] in ("table", *env.fixtures_in_scene)
    assert env.sim.obj_low_z() > env.s["scene"].TABLE_TOP - 0.01


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
    # 0.25 m/s: at 0.5 m/s the brick swings 30 deg in the pinch and the braking ejects it or not with a 5 mm start shift
    r = env.execute(p, force=20.0, v_lift=0.15, v_transport=0.25, destination="box")
    assert r.label == "placed_stable" and r.zone == "box"


def test_long_object_into_box_is_never_placed_stable():
    env = _env()
    p = _control(env)
    r = env.execute(p, force=20.0, v_lift=0.15, destination="box")
    # the 12 cm rim strips a 19.6 cm object out of the pads (about 11 cm above the box floor) or it bridges the opening
    assert r.label in ("missed", "tipped", "dropped_low", "dropped_high")
    assert r.loss_phase in (None, "lower")


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



def test_lift_trace_reads_the_weight_and_the_ramps():
    """The trace's force is the mean over the step's physics substeps: the weight while the lift runs at
    constant speed, and the weight times 1 ± a/g while it speeds up and brakes (a = v_lift / RAMP_S). While
    the hand target jumped once per control step, a force sampled at the end of a step read 0.9 of the weight."""
    from robots.mujoco_lab.adapter import RAMP_S

    env = _env()
    env.reset("ycb__brick", {"mu": 0.8, "mass_scale": 1.0, "com_s": None, "v_lift": 0.15}, seed=0)
    f_std = env.s["t3_make"].std_force(env.nominal_mass())
    r = env.execute(env.proposals()[1], force=2 * f_std, v_lift=0.15, v_transport=0.5, destination="table_centre")
    hz, mg, ag = env.s["scene"].CTRL_HZ, env.truth()["mass"] * 9.81, 0.15 / RAMP_S / 9.81
    lift = [x for x in r.trace if x["phase"] == "lift"]
    assert all(all(x["pads"]) for x in lift)
    v = [(b["hand_z"] - c["hand_z"]) * hz for c, b in zip(lift, lift[1:])]
    steady = [x for x, v0, v1 in zip(lift[2:], v, v[1:]) if abs(v0 / 0.15 - 1) < 0.01 and abs(v1 / 0.15 - 1) < 0.01]
    assert len(steady) >= 20
    assert max(abs(x["fz"] / mg - 1.0) for x in steady) <= 0.02
    fz = [x["fz"] / mg for x in lift]
    assert abs(max(fz) - (1 + ag)) <= 0.01 and abs(min(fz) - (1 - ag)) <= 0.01


def test_trace_records_the_pad_normals_the_hand_height_and_the_turn_since_the_lift():
    env = _env()
    env.reset("ycb__brick", {"mu": 0.8, "mass_scale": 1.0, "com_s": None, "v_lift": 0.15}, seed=0)
    f_std = env.s["t3_make"].std_force(env.nominal_mass())
    r = env.execute(env.proposals()[1], force=2 * f_std, v_lift=0.15, v_transport=0.5, destination="table_centre")
    assert all({"hand_z", "pad_nz", "rot_lift_deg"} <= set(x) for x in r.trace)
    assert all(x["rot_lift_deg"] is None for x in r.trace if x["phase"] in ("approach", "close"))
    hold = [x for x in r.trace if x["phase"] == "hold"]
    assert hold and all(n is not None and abs(n) < 0.2 for x in hold for n in x["pad_nz"])      # the brick's sides are vertical
    assert all(x["rot_lift_deg"] < 3.0 for x in hold)
    lift = [x for x in r.trace if x["phase"] == "lift"]
    assert lift[-1]["hand_z"] - lift[0]["hand_z"] > 0.15


def test_trace_follows_the_object_point_between_the_pads():
    """grip_dz: the rise since the lift began of the object point between the pads at the end of the close.
    It is None before the lift, the hand's rise while the pads carry the brick, and about 0 while the pads
    slide up a brick they cannot lift (2 mu F = 1 N against a 1.96 N weight), until they run off its top edge
    and tip it (the last two steps: 0.6 and 2.4 mm)."""
    env = _env()
    for mu, force, carried in ((0.8, 20.0, True), (0.1, 5.0, False)):
        env.reset(SHORT, {"mu": mu, "mass_scale": 1.0, "com_s": None, "v_lift": 0.15}, seed=0)
        r = env.execute(env.proposals()[1], force=force, v_lift=0.15, v_transport=0.5, destination="table_centre")
        close = [x for x in r.trace if x["phase"] == "close"][-1]
        lift = [x for x in r.trace if x["phase"] == "lift" and all(x["pads"])]
        assert close["grip_dz"] is None and len(lift) >= 10
        rise = np.array([x["hand_z"] - close["hand_z"] for x in lift])
        grip = np.array([x["grip_dz"] for x in lift])
        if carried:
            assert np.abs(grip - rise).max() < 0.001
        else:
            assert rise.max() > 0.005 and np.abs(grip[:-2]).max() < 0.0005


@pytest.mark.parametrize("oid", ["ycb__spam_can", "hope__ketchup_bottle"])
def test_hold_force_is_the_weight_under_a_firm_squeeze(oid):
    """The hold reading is the pads' net vertical force on the object: the weight, whatever the squeeze
    (a per-contact absolute sum counts the normal force's vertical parts on curved faces)."""
    env = _env()
    env.reset(oid, {"mu": 0.8, "mass_scale": 1.0, "com_s": None, "v_lift": 0.15}, seed=0)
    f_std = env.s["t3_make"].std_force(env.nominal_mass())
    r = env.execute(env.proposals()[1], force=2 * f_std, v_lift=0.15, v_transport=0.5, destination="table_centre")
    assert r.readouts["fz_hold_n"] is not None
    assert r.readouts["fz_hold_n"] / (env.truth()["mass"] * 9.81) == pytest.approx(1.0, abs=0.02)

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


def test_reset_frees_the_previous_sim():
    """The metered control step must not tie a sim to itself: a corpus worker resets once per row, and
    sims kept alive until the cycle collector ran filled the 14 GB limit of 16 workers in 19 s."""
    import gc
    import weakref

    env = _env()
    env.reset("ycb__brick", seed=0)
    old = weakref.ref(env.sim)
    gc.disable()
    try:
        env.reset("ycb__brick", seed=1)
        assert old() is None
    finally:
        gc.enable()
