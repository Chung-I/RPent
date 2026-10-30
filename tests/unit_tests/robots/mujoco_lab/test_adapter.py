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
    theta, com_end, _, _ = case_draw(11, obj, 0, object_props(obj)["mass"])
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
    theta, com_end, _, _ = case_draw(11, obj, 5, object_props(obj)["mass"])
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
    (a per-contact absolute sum counts the normal force's vertical parts on curved faces). The physics is the clean
    reading; the declared force error (plan 2b2) is checked in test_force_noise."""
    env = _env()
    env.reset(oid, {"mu": 0.8, "mass_scale": 1.0, "com_s": None, "v_lift": 0.15}, seed=0)
    f_std = env.s["t3_make"].std_force(env.nominal_mass())
    r = env.execute(env.proposals()[1], force=2 * f_std, v_lift=0.15, v_transport=0.5, destination="table_centre")
    assert r.readouts["fz_hold_n"] is not None
    assert r.sensor["clean"]["fz_hold_n"] / (env.truth()["mass"] * 9.81) == pytest.approx(1.0, abs=0.02)

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


# --- milestone 2b: the damage meter, the liquid, the carry acceleration and the squeeze probe

MUG = "ycb__mug"            # fillable: a 276 ml frustum cavity (fragility_classes.json); its s 0.5 and 0.7 pinches fit
CRUSHABLE = "ycb__jello"    # cardboard, 7.3 cm across at s 0.5


def _mug(env, fill, seed=0):
    env.reset(MUG, {"mass_scale": 1.0, "com_s": None}, seed=seed, fill=fill)
    return env


def _mid(env):
    from rpent.skills.pick_place.menu import mid_proposal, pinch_limit, usable_proposals

    return mid_proposal(usable_proposals(env.proposals(), pinch_limit(env)))


def test_damage_meter_does_not_change_the_motion():
    """The meter reads the contacts, the pad forces and the CoM velocity in every physics substep and writes nothing,
    so a rollout with it on moves the object exactly as one with it off."""
    runs = []
    for on in (True, False):
        env = _env(damage_meter=on)
        p = _control(env, SHORT)
        r = env.execute(p, force=20.0, v_lift=0.15, v_transport=0.25, destination="plate")
        runs.append((r, [(x["z"], x["rot_deg"], x["low_z"]) for x in r.trace], env.sim.obj_pos().tolist(), env.sim.obj_mat().tolist()))
    (r_on, *on), (r_off, *off) = runs
    assert on == off
    assert r_on.meter["fn_pad_max"] and r_on.motion_label == r_off.motion_label == r_on.label == r_off.label
    assert r_off.meter == {} and r_off.damage is None


def test_filled_container_weighs_its_liquid():
    from robots.mujoco_lab.fragility import entry_of, liquid, load_classes

    entry, _ = entry_of(MUG, load_classes())
    env = _env()
    m_empty = _mug(env, 0.0).truth()["mass"]
    t = _mug(env, 0.6).truth()
    m_l = liquid(entry["cavity"], 0.6)["mass_kg"]
    assert m_l > 0.1                                                    # 0.6 of 276 ml
    assert t["mass"] == pytest.approx(m_empty + m_l, rel=0.01)
    assert t["fill"] == 0.6 and t["liquid_mass"] == pytest.approx(m_l, rel=1e-6) and t["frag"] is None


def test_liquid_moves_the_com_to_the_weighted_centroid():
    from robots.mujoco_lab.fragility import entry_of, liquid, load_classes

    entry, _ = entry_of(MUG, load_classes())
    env = _env()
    t0 = _mug(env, 0.0).truth()
    t1 = _mug(env, 0.6).truth()
    liq = liquid(entry["cavity"], 0.6)
    want = (t0["mass"] * np.asarray(t0["com_obj"]) + liq["mass_kg"] * np.asarray(liq["centroid_obj"])) / (t0["mass"] + liq["mass_kg"])
    assert np.allclose(t1["com_obj"], want, atol=2e-4)
    # not vacuous: the liquid moves the CoM by more than five tolerances (1.5 mm for this mug at 0.6)
    assert np.linalg.norm(np.asarray(t1["com_obj"]) - np.asarray(t0["com_obj"])) > 5 * 2e-4


def test_free_fall_impact_speed_matches_the_height():
    """The brick released with its lowest point 0.18 m above the table: the first impact after the release is the CoM
    speed one substep before the new table contact, sqrt(2 g 0.18) = 1.879 m/s."""
    env = _env()
    p = _control(env, SHORT)
    r = env.execute(p, force=20.0, v_lift=0.15, v_transport=0.25, destination="table_centre", release_height=0.18)
    assert r.meter["impacts"]
    t, v = r.meter["impacts"][0]
    assert v == pytest.approx(1.879, rel=0.10)


def test_force_channel_first_crossing():
    """A squeeze ramp to 20 N on the brick: the force channel (the larger pad's normal force over 10 ms, a running
    maximum kept at every 1 % growth) first passes 10 N within 50 ms of the step in which the ramp commands 10 N."""
    from robots.mujoco_lab.fragility import _first_above

    env = _env()
    p = _control(env, SHORT)
    res = env.probe_squeeze(p, 20.0)
    t_cmd = next(x["t_meter"] for x in res.trace if x["phase"] == "ramp" and x["f_cmd"] >= 10.0)
    t_meas = _first_above(res.meter["fn_pad_max"], 10.0)
    assert t_meas is not None and abs(t_meas - t_cmd) <= 0.05
    assert res.meter["fn_pad_max"][-1][1] == pytest.approx(20.0, rel=0.05)
    assert res.probe == "squeeze" and res.kind == "measured" and res.damage is None


def test_slow_carry_lowers_the_surface_angle():
    """A mug filled to 0.8 carried to the table centre: the surface angle's maximum at a 0.5 m/s^2 carry is below that
    at 2.0 m/s^2 (a step in acceleration overshoots the static slope atan(a / g), 2.9 and 11.5 deg)."""
    out = {}
    for a in (0.5, 2.0):
        env = _env()
        _mug(env, 0.8)
        r = env.execute(_mid(env), force=20.0, v_lift=0.15, v_transport=0.5, destination="table_centre", a_transport=a)
        out[a] = r.meter["surface_max"][-1][1]
    assert out[0.5] < out[2.0]


def test_spill_by_tilt_overrides_the_motion_label():
    """A mug filled to 0.8 pinched off its centre with 2.5 N turns in the pads during the lift past its spill angle,
    before any carry (a slow one, 0.5 m/s^2): it spills when the tilt crosses the spill angle, and the spill replaces
    the motion label, whatever that says (placed_stable here)."""
    from robots.mujoco_lab.fragility import entry_of, load_classes, spill_angle_deg

    env = _env()
    _mug(env, 0.8)
    r = env.execute(env.proposals()[0], force=2.5, v_lift=0.15, v_transport=0.25, destination="table_centre",
                    a_transport=0.5)
    spill = spill_angle_deg(entry_of(MUG, load_classes())[0]["cavity"], 0.8)
    assert max(x["rot_lift_deg"] or 0.0 for x in r.trace) > spill                 # the tilt alone passes it
    cross = next(x["t_meter"] for x in r.trace if (x["rot_lift_deg"] or 0.0) > spill)
    carry = next(x["t_meter"] for x in r.trace if x["phase"] == "transport")
    assert r.damage["label"] == "spilled" and r.damage["channel"] == "surface"
    assert r.damage["t"] < carry and abs(r.damage["t"] - cross) < 0.15
    assert r.label == "spilled" and r.motion_label != "spilled"


def _crushable(env, f_damage_n):
    from robots.mujoco_lab.fragility import THRESHOLDS_V0, Thresholds, entry_of, load_classes, map_latent

    entry, _ = entry_of(CRUSHABLE, load_classes())
    frag = {**map_latent([0.0, 0.0, 0.0], entry, Thresholds.from_json(THRESHOLDS_V0)), "f_damage_n": float(f_damage_n)}
    env.reset(CRUSHABLE, seed=0, frag=frag)
    return frag


def test_weigh_probe_can_crush():
    """The jello box with f_damage_n 5 N under the weigh squeeze (weigh_force(10) = 20 N): the probe crushes it, so its
    kind is "damaged" and its damage "crushed"; at 60 N it survives."""
    from rpent.skills.pick_place.probes import weigh_force

    env = _env()
    _crushable(env, 5.0)
    res = env.probe_weigh(_mid(env), weigh_force(10.0))
    assert res.kind == "damaged" and res.damage["label"] == "crushed" and res.damage["channel"] == "force"
    assert not res.artifact                                # the drive crushed it, not a jam
    _crushable(env, 60.0)
    res = env.probe_weigh(_mid(env), weigh_force(10.0))
    assert res.kind != "damaged" and res.damage is None and res.meter["fn_pad_max"][-1][1] > 15.0


def test_a_crush_by_a_strong_drive_is_not_a_jam():
    """The jello box with f_damage_n 40 N under a 60 N weigh squeeze: the drive crushes it. The pad force exceeds 50 N,
    but not the drive by 50 N, so the run is no artifact."""
    env = _env()
    _crushable(env, 40.0)
    res = env.probe_weigh(_mid(env), 60.0)
    assert res.damage and res.damage["label"] == "crushed" and res.meter["fn_pad_max"][-1][1] > 55.0
    assert not res.artifact

def test_a_jam_makes_the_damage_an_artifact():
    """Probe corpus v8_t0, the carton's draw 14 (seed 7): the open hand's push wedges the carton under a pad at
    thousands of newtons, far above any drive (the position-driven hand has no force limit). The crush is the jam's,
    not a grip's, so the run is an artifact."""
    env = _env()
    theta = {"mass_scale": 0.983188789301263, "mu": 0.6498445278095558, "com_s": 0.3995646448976271, "v_lift": 0.4}
    frag = {"k_n_per_mm": 1.7798343355862833, "f_damage_n": 19.128204436226216, "v_bruise_ms": None, "v_break_ms": None,
            "z": [-0.7648911627712235, 0.46448873853987255, -0.9416817358216752]}
    env.reset("hope__orange_juice_carton", theta, seed=7014, com_end=-1, fixture=None, fill=0.0, frag=frag)
    res = env.probe_push()
    assert res.damage and res.damage["label"] == "crushed" and res.meter["fn_pad_max"][-1][1] > 1000.0
    assert res.artifact

def test_push_meters_the_surface_angle():
    """The push moves a mug filled to 0.6: the surface channel records the slosh it starts."""
    env = _env()
    _mug(env, 0.6)
    res = env.probe_push()
    assert res.meter["surface_max"] and res.meter["surface_max"][-1][1] > 0.5


def test_squeeze_reading_noise_is_ten_percent():
    """500 seeded readings: log(k_hat / k) has sd 0.10; the probe keeps its standard normal."""
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    r = [MuJoCoLabEnv.squeeze_reading(2.0, np.random.default_rng(i)) for i in range(500)]
    lr = np.log([x["k_hat"] / 2.0 for x in r])
    assert abs(float(np.std(lr)) - 0.10) < 0.01
    assert all(x["k_hat"] == pytest.approx(2.0 * np.exp(0.10 * x["eps"])) for x in r)
    assert MuJoCoLabEnv.squeeze_reading(None, np.random.default_rng(0)).get("k_hat") is None


def test_squeeze_probe_reads_the_stiffness():
    """The jello box under a 10 N squeeze (its f_damage_n 60 N): measured, k_hat = k exp(0.10 eps) with the eps the
    reading keeps, no lift, the box left where it was."""
    env = _env()
    frag = _crushable(env, 60.0)
    res = env.probe_squeeze(_mid(env), 10.0)
    assert res.kind == "measured" and res.damage is None
    assert res.reading["k_hat"] == pytest.approx(frag["k_n_per_mm"] * np.exp(0.10 * res.reading["eps"]))
    assert res.displacement_m < 0.02 and not res.thrown and max(x["z"] for x in res.trace) < 0.01


def test_squeeze_noise_is_drawn_per_object():
    """The corpus, the probe corpus and calibrate reset every object of a draw with one seed (seed x 1000 + idx), so
    the squeeze's noise stream must include the object: two objects at one seed draw different eps; one object at its
    seed repeats its own."""
    env = _env()
    eps = []
    for oid in (SHORT, CRUSHABLE, SHORT):
        env.reset(oid, seed=5)
        eps.append(env.probe_squeeze(_mid(env), 5.0).reading["eps"])
    assert eps[0] != eps[1] and eps[0] == eps[2]


def test_the_transport_slosh_takes_the_carry_scale(monkeypatch):
    """slosh_scale (plan 2b, Task 8): the slosh model runs with the carry class over the transport's physics substeps
    and the jolt class everywhere else, at the factors of the mug's fill."""
    import robots.mujoco_lab.slosh as slosh

    made = []

    class Spy(slosh.Slosh):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.log = []
            made.append(self)

        def step(self, a_h, tilt, dt):
            self.log.append(self.profile)
            return super().step(a_h, tilt, dt)

    monkeypatch.setattr(slosh, "Slosh", Spy)
    env = _env()
    env.slosh_scale = {"carry": {"0.6": 0.5, "1.0": 0.7}, "jolt": {"0.6": 0.9}}
    _mug(env, 0.8)
    r = env.execute(_mid(env), force=20.0, v_lift=0.15, v_transport=0.5, destination="table_centre")
    s = made[-1]
    n = sum(x["phase"] == "transport" for x in r.trace) * int(env.s["scene"].SUB)
    assert s.scale == pytest.approx({"carry": 0.6, "jolt": 0.9})
    assert n > 0 and s.log.count("carry") == n and s.log[0] == s.log[-1] == "jolt"
    i = s.log.index("carry")
    assert s.log[i:i + n] == ["carry"] * n
