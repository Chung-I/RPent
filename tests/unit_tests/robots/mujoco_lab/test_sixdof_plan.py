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

"""The lab side of the sixdof rerun's free move (design 2026-11-08-sixdof-rerun-design.md, unit (e), gate G3): the
world cuRobo sees, the robot's sphere refit, and the cached plan as the executor's joint waypoints."""

import json
import math
import os

import numpy as np
import pytest

from robots.mujoco_lab import curobo_plan as cp
from robots.mujoco_lab import sixdof_plan as sp

SIM = pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")


def _box(c, h):
    return np.array([[c[0] + a * h[0], c[1] + b * h[1], c[2] + z * h[2]] for a in (-1, 1) for b in (-1, 1) for z in (-1, 1)])


# ------------------------------------------------------------------ the sphere cover

def test_surface_points_lie_on_the_hull_with_outward_normals():
    P, N = sp.surface_points([_box((0.1, 0.0, 0.0), (0.03, 0.01, 0.05))], spacing=0.002, seed=0)
    assert len(P) > 1000
    d = np.abs(np.abs(P - [0.1, 0.0, 0.0]) - [0.03, 0.01, 0.05]).min(axis=1)
    assert d.max() < 1e-12                                          # every point on a face
    assert np.all(np.einsum("ij,ij->i", N, P - [0.1, 0.0, 0.0]) > 0)  # normals point out
    np.testing.assert_allclose(np.linalg.norm(N, axis=1), 1.0)


def test_outside_is_a_lower_bound_of_the_distance_to_the_union():
    hulls = [_box((0, 0, 0), (0.01, 0.01, 0.01)), _box((0.05, 0, 0), (0.01, 0.01, 0.01))]
    x = np.array([[0.0, 0.0, 0.0], [0.025, 0.0, 0.0], [0.0, 0.0, 0.03], [0.07, 0.02, 0.0]])
    out = sp.outside(x, hulls)
    assert out[0] < 0 and out[1] == pytest.approx(0.015) and out[2] == pytest.approx(0.02)
    assert out[3] <= math.hypot(0.01, 0.01) + 1e-12


def test_cover_reaches_the_gap_keeps_the_start_spheres_and_caps_the_protrusion():
    hulls = [_box((0.0, 0.0, 0.0), (0.03, 0.01, 0.05))]
    C0, R0 = np.array([[0.0, 0.0, 0.03]]), np.array([0.012])
    fit = sp.cover(hulls, C0, R0, gap=0.0008, protrusion=0.005)
    np.testing.assert_array_equal(fit["C"][:1], C0)
    np.testing.assert_array_equal(fit["R"][:1], R0)
    assert list(fit["added"]) == [False] + [True] * (len(fit["R"]) - 1)
    P, _ = sp.surface_points(hulls, spacing=0.0005, seed=1)         # an independent, denser sample
    assert sp.gaps(P, fit["C"], fit["R"]).max() <= 0.001
    a = fit["added"]
    assert sp.outside(sp.sphere_points(fit["C"][a], fit["R"][a]), hulls).max() <= 0.005 + 1e-12
    assert fit["gap_m"] <= 0.0008 and fit["protrusion_m"] <= 0.005


def test_cover_ignores_protrusion_outside_the_region():
    hulls = [_box((0.0, 0.0, 0.0), (0.01, 0.01, 0.01))]
    fit = sp.cover(hulls, np.zeros((0, 3)), np.zeros(0), gap=0.0008, protrusion=0.0, region=lambda X: X[:, 2] < -0.02)
    assert len(fit["R"]) > 0                                        # nothing protrudes below z = -0.02, so any sphere fits


# ------------------------------------------------------------------ the cached plan as waypoints

START = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.0])
GOAL = np.array([0.1, -0.2, 0.05, -2.0, 0.1, 1.8, 0.3])
IDENT = {"robot_yml_sha256": "00" * 32, "settings": cp.SETTINGS, "curobo": "0.9.0", "torch": "2.14.1"}
WORLD = {"cuboid": {"t": {"dims": [1.0, 1.0, 0.1], "pose": [0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]}}, "mesh": {}}


def _cache(tmp_path, **entry):
    os.makedirs(tmp_path / "c", exist_ok=True)
    json.dump(IDENT, open(tmp_path / "c" / "identity.json", "w"))
    if entry:
        cp.write_entry(str(tmp_path / "c"), cp.plan_key(WORLD, START, GOAL, IDENT), entry)
    return str(tmp_path / "c")


def test_a_cache_miss_stops_the_row(tmp_path):
    with pytest.raises(sp.PlanCacheMiss):
        sp.free_from_cache(_cache(tmp_path), WORLD, START, GOAL, 50.0)


def test_a_cached_failure_is_a_failed_free_move(tmp_path):
    c = _cache(tmp_path, success=False, status="trajopt failed", dt=0.02, q=None, wall_s=1.0)
    with pytest.raises(sp.PlanFailed, match="trajopt failed"):
        sp.free_from_cache(c, WORLD, START, GOAL, 50.0)


def test_a_cached_plan_gives_one_waypoint_per_control_step_after_the_start(tmp_path):
    q = np.linspace(START, GOAL, 30).astype(np.float32).astype(float)          # cuRobo plans in float32
    c = _cache(tmp_path, success=True, status="ok", dt=0.02, q=q, wall_s=0.1)
    free = sp.free_from_cache(c, WORLD, START, GOAL, 50.0)
    assert len(free) == 29
    np.testing.assert_array_equal(np.array(free), q[1:])


def test_a_cached_plan_at_another_step_or_start_is_refused(tmp_path):
    q = np.linspace(START, GOAL, 30)
    c = _cache(tmp_path, success=True, status="ok", dt=0.025, q=q, wall_s=0.1)
    with pytest.raises(ValueError, match="dt"):
        sp.free_from_cache(c, WORLD, START, GOAL, 50.0)
    q2 = q.copy()
    q2[0, 0] += 1e-4
    c2 = _cache(tmp_path / "x", success=True, status="ok", dt=0.02, q=q2, wall_s=0.1)
    with pytest.raises(ValueError, match="start"):
        sp.free_from_cache(c2, WORLD, START, GOAL, 50.0)


# ------------------------------------------------------------------ on the lab

@SIM
def test_world_export_is_the_scene_without_the_robot_in_the_base_frame():
    import mujoco

    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    env = MuJoCoLabEnv()
    env.reset("ycb__sugar_box", seed=0)
    sim = env.sim
    m, d = sim.m, sim.d
    w = sp.world_export(sim)
    assert cp.canonical(w) == cp.canonical(sp.world_export(sim))
    robot = sp.robot_subtree_geoms(sim)
    assert {m.geom("palm").id, m.geom("pad_l").id, m.geom("arm_link4_c").id} <= robot
    want = {sp.geom_name(m, g) for g in range(m.ngeom) if g not in robot and (m.geom_contype[g] or m.geom_conaffinity[g])}
    got = set(w["cuboid"]) | set(w["mesh"])
    assert got == want and not ({sp.geom_name(m, g) for g in robot} & got)
    b0 = m.body("link0").id
    R0, p0 = d.xmat[b0].reshape(3, 3), d.xpos[b0]
    for g in sim.hull_ids:
        V = np.asarray(w["mesh"][sp.geom_name(m, g)]["vertices"])
        mid = m.geom_dataid[g]
        Vw = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        Vb = (Vw - p0) @ R0
        assert {tuple(np.round(v, 12)) for v in V} <= {tuple(np.round(v, 12)) for v in Vb}
        F = np.asarray(w["mesh"][sp.geom_name(m, g)]["faces"]).reshape(-1, 3)
        c = V.mean(axis=0)
        n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        assert np.all(np.einsum("ij,ij->i", n, V[F].mean(axis=1) - c) > 0)       # faces wound outward
    boxes = [g for g in range(m.ngeom) if g not in robot and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and (m.geom_contype[g] or m.geom_conaffinity[g])]
    assert boxes
    for g in boxes:
        e = w["cuboid"][sp.geom_name(m, g)]
        assert e["dims"] == [2.0 * float(x) for x in m.geom_size[g]]
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, (R0.T @ d.geom_xmat[g].reshape(3, 3)).ravel())
        np.testing.assert_allclose(e["pose"][:3], R0.T @ (d.geom_xpos[g] - p0), atol=1e-12)
        assert abs(abs(float(np.dot(e["pose"][3:], q))) - 1.0) < 1e-12


@SIM
def test_the_committed_fit_covers_the_arm_and_the_open_gripper_within_1_mm():
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    env = MuJoCoLabEnv()
    env.reset("ycb__sugar_box", seed=0)
    lab = json.load(open(sp.LAB_ROBOT))
    A = sp._armlib(env.sim)
    assert lab["mount_pos"] == [float(x) for x in A.MOUNT_POS] and lab["free_v"] == float(A.FREE_V)
    assert lab["pad_centre_z"] == -sp.HAND_OFF
    for link in [f"panda_link{i}" for i in range(1, 8)]:
        P, _ = sp.surface_points(sp.link_hulls(env.sim, link), spacing=0.001, seed=7)
        s = lab["arm_spheres"][link]
        assert sp.gaps(P, np.array([x["center"] for x in s]), np.array([x["radius"] for x in s])).max() <= 0.001, link
    hand = ("lab_hand", "lab_finger_l", "lab_finger_r")
    P, _ = sp.surface_points([h for n in hand for h in sp.link_hulls(env.sim, n)], spacing=0.001, seed=7)
    s = [x for n in hand for x in lab["gripper_spheres"][n]]
    assert sp.gaps(P, np.array([x["center"] for x in s]), np.array([x["radius"] for x in s])).max() <= 0.001


@SIM
def test_free_hits_reads_the_waypoints_against_the_scene():
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    env = MuJoCoLabEnv()
    env.reset("ycb__sugar_box", seed=0)
    q0 = np.asarray(env.sim.arm.desired(), dtype=float)
    assert sp.free_hits(env.sim, [q0, q0]) == []
    low = q0.copy()
    low[1], low[3] = 1.4, -1.6                                      # the arm reaches down into the table
    hits = sp.free_hits(env.sim, [q0, low])
    assert hits and all(h["step"] == 1 for h in hits) and any(h["other"] == "table" for h in hits)
    robot = sp.robot_subtree_geoms(env.sim)
    assert all(h["robot_id"] in robot and h["other_id"] not in robot for h in hits)


GPU = pytest.mark.skipif(not (os.path.exists(sp.CUROBO_PY) and os.path.exists("/dev/nvidia0")), reason="needs cuRobo's venv and a GPU")


@SIM
@GPU
def test_plan_case_plans_once_and_serves_the_free_move(tmp_path):
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    env = MuJoCoLabEnv()
    env.reset("ycb__sugar_box", seed=0)
    q0 = np.asarray(env.sim.arm.desired(), dtype=float)
    goal = q0 + np.array([0.4, 0.25, -0.3, 0.35, 0.2, -0.25, 0.5])
    world = sp.world_export(env.sim)
    req = sp.request(("ycb__sugar_box", 0), world, q0, [{"row": "a", "q": goal}, {"row": "b", "q": goal}])
    assert sp.plan_case(req, str(tmp_path)) == {"goals": 2, "planned": 1, "cached": 0}
    assert sp.plan_case(req, str(tmp_path)) == {"goals": 2, "planned": 0, "cached": 2}    # the second run plans nothing
    free = sp.free_from_cache(str(tmp_path), world, q0, goal, 50)
    q = np.array([q0] + free)
    assert q.shape[1] == 7 and len(free) > 10
    assert np.abs(q[-1] - goal).max() < 1e-3                                             # ends at the goal (ruling 7)
    assert np.abs(np.diff(q, axis=0)).max() * 50 <= 1.0 + 1e-3                           # FREE_V cap on every joint
    assert sp.free_hits(env.sim, free) == []


def test_plan_case_keeps_one_request_per_case_and_reports_a_crash(tmp_path):
    a = sp.request(("obj", 0), {"cuboid": {}}, [0.0] * 7, [{"row": "r", "q": [0.1] * 7}])
    b = sp.request(("obj", 0), {"cuboid": {}}, [0.0] * 7, [{"row": "r", "q": [0.2] * 7}])
    with pytest.raises(RuntimeError, match="exited 1 on obj_0"):
        sp.plan_case(a, str(tmp_path), python="/bin/false")
    with pytest.raises(RuntimeError, match="exited 1"):
        sp.plan_case(a, str(tmp_path), python="/bin/false")                       # the same request again is fine
    with pytest.raises(ValueError, match="another request"):
        sp.plan_case(b, str(tmp_path), python="/bin/false")
