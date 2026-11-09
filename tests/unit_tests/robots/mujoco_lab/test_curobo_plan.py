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

"""The plan cache of the sixdof rerun (design 2026-11-08-sixdof-rerun-design.md, unit (e), gate G3): keys, entries and
the cuRobo robot config, without cuRobo (the module imports it only to plan)."""

import hashlib
import json
import math
import os

import numpy as np
import pytest

from robots.mujoco_lab import curobo_plan as cp

WORLD = {"cuboid": {"table": {"dims": [1.0, 1.2, 0.05], "pose": [0.5, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]}},
         "mesh": {"hull0": {"vertices": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
                            "faces": [0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3], "pose": [0, 0, 0, 1, 0, 0, 0]}}}
START = [0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.0]
GOAL = [0.1, -0.2, 0.05, -2.0, 0.1, 1.8, 0.3]
IDENT = {"robot_yml_sha256": "ab" * 32, "settings": {"interpolation_dt": 0.02}, "curobo": "0.9.0", "torch": "2.14.1"}


def test_plan_key_is_the_sha256_of_the_canonical_request():
    k = cp.plan_key(WORLD, START, GOAL, IDENT)
    body = {"world": WORLD, "start": START, "goal": GOAL, "identity": IDENT}
    assert k == hashlib.sha256(cp.canonical(body)).hexdigest()
    assert len(k) == 64


def test_plan_key_ignores_dict_order_and_takes_numpy():
    w2 = {"mesh": WORLD["mesh"], "cuboid": {"table": {"pose": WORLD["cuboid"]["table"]["pose"], "dims": WORLD["cuboid"]["table"]["dims"]}}}
    assert cp.plan_key(w2, np.array(START), np.array(GOAL, dtype=np.float64), IDENT) == cp.plan_key(WORLD, START, GOAL, IDENT)


@pytest.mark.parametrize("where", ["start", "goal", "world", "identity"])
def test_plan_key_changes_with_one_ulp(where):
    k0 = cp.plan_key(WORLD, START, GOAL, IDENT)
    w, s, g, i = json.loads(json.dumps(WORLD)), list(START), list(GOAL), json.loads(json.dumps(IDENT))
    if where == "start":
        s[3] = float(np.nextafter(s[3], 0.0))
    elif where == "goal":
        g[6] = float(np.nextafter(g[6], 1.0))
    elif where == "world":
        w["mesh"]["hull0"]["vertices"][1][0] = float(np.nextafter(0.1, 1.0))
    else:
        i["settings"]["interpolation_dt"] = float(np.nextafter(0.02, 1.0))
    assert cp.plan_key(w, s, g, i) != k0


def test_canonical_refuses_nan():
    with pytest.raises(ValueError):
        cp.canonical({"x": math.nan})


def test_entries_round_trip_and_never_change(tmp_path):
    key = cp.plan_key(WORLD, START, GOAL, IDENT)
    assert cp.read_entry(str(tmp_path), key) is None
    q = np.array([START, GOAL])
    e = {"success": True, "status": "ok", "dt": 0.02, "q": q, "wall_s": 0.1}
    cp.write_entry(str(tmp_path), key, e)
    got = cp.read_entry(str(tmp_path), key)
    assert got["success"] is True and got["status"] == "ok" and got["dt"] == 0.02 and got["key"] == key
    np.testing.assert_array_equal(got["q"], q)
    cp.write_entry(str(tmp_path), key, dict(e))                 # the same entry again: allowed, nothing changes
    with pytest.raises(ValueError, match="differs"):
        cp.write_entry(str(tmp_path), key, dict(e, success=False, q=None))
    left = [f for _, _, fs in os.walk(tmp_path) for f in fs]
    assert left == [os.path.basename(cp.entry_path(str(tmp_path), key))]


def test_failures_are_cached(tmp_path):
    key = cp.plan_key(WORLD, START, GOAL, IDENT)
    cp.write_entry(str(tmp_path), key, {"success": False, "status": "trajopt failed", "dt": 0.02, "q": None, "wall_s": 1.0})
    got = cp.read_entry(str(tmp_path), key)
    assert got["success"] is False and got["q"] is None and got["status"] == "trajopt failed"


FRANKA = {"robot_cfg": {"kinematics": {
    "collision_link_names": [f"panda_link{i}" for i in range(8)] + ["panda_hand", "panda_leftfinger", "panda_rightfinger", "attached_object"],
    "collision_spheres": {**{f"panda_link{i}": [{"center": [0.0, 0.0, 0.01 * i], "radius": 0.05}] for i in range(8)},
                          "panda_hand": [{"center": [0, 0, 0], "radius": 0.04}], "panda_leftfinger": [{"center": [0, 0, 0], "radius": 0.01}],
                          "panda_rightfinger": [{"center": [0, 0, 0], "radius": 0.01}], "attached_object": [{"center": [0, 0, 0], "radius": -10.0}]},
    "self_collision_buffer": {"panda_link0": 0.1, "panda_hand": 0.02, "panda_leftfinger": 0.01, "panda_rightfinger": 0.01},
    "self_collision_ignore": {"panda_link5": ["panda_link6", "panda_hand"], "panda_hand": ["panda_leftfinger", "panda_rightfinger"]},
    "cspace": {"joint_names": [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]},
    "grasp_contact_link_names": ["panda_hand"], "mesh_link_names": [f"panda_link{i}" for i in range(8)] + ["panda_hand"]}}}
LAB = {"mount_pos": [0.0, 0.0, 0.217], "mount_quat": [0.0, 1.0, 0.0, 0.0], "pad_centre_z": -0.048, "free_v": 1.0,
       "arm_spheres": {f"panda_link{i}": [{"center": [0.0, 0.0, 0.0], "radius": 0.06}, {"center": [0.0, 0.01, 0.0], "radius": 0.02}]
                       for i in range(1, 8)},
       "gripper_spheres": {"lab_hand": [{"center": [0.0, 0.0, 0.0], "radius": 0.02}],
                           "lab_finger_l": [{"center": [-0.06, 0.0, -0.03], "radius": 0.009}],
                           "lab_finger_r": [{"center": [0.06, 0.0, -0.03], "radius": 0.009}]}}


def test_robot_yml_mounts_our_gripper_and_takes_the_refit():
    y = cp.robot_yml(json.loads(json.dumps(FRANKA)), LAB)
    k = y["robot_cfg"]["kinematics"]
    assert k["tool_frames"] == ["lab_tcp"]
    ex = k["extra_links"]
    assert ex["lab_hand"]["parent_link_name"] == "panda_link7"
    assert ex["lab_hand"]["fixed_transform"] == [0.0, 0.0, 0.217, 0.0, 1.0, 0.0, 0.0]
    assert ex["lab_tcp"]["parent_link_name"] == "lab_hand" and ex["lab_tcp"]["fixed_transform"] == [0, 0, -0.048, 0, 1, 0, 0]
    sph = k["collision_spheres"]
    assert sph["panda_link0"] == FRANKA["robot_cfg"]["kinematics"]["collision_spheres"]["panda_link0"]    # the base is kept
    for i in range(1, 8):
        assert sph[f"panda_link{i}"] == LAB["arm_spheres"][f"panda_link{i}"]
    for n in ("lab_hand", "lab_finger_l", "lab_finger_r"):
        assert sph[n] == LAB["gripper_spheres"][n]
    assert not {"panda_hand", "panda_leftfinger", "panda_rightfinger"} & (set(sph) | set(k["collision_link_names"]))
    assert {"lab_hand", "lab_finger_l", "lab_finger_r", "attached_object"} <= set(k["collision_link_names"])
    for l in ("panda_link5", "panda_link6", "panda_link7"):         # the gripper never collides with the wrist links
        assert {"lab_hand", "lab_finger_l", "lab_finger_r"} <= set(k["self_collision_ignore"][l])
    assert k["self_collision_ignore"]["lab_hand"] == sorted(["attached_object", "lab_finger_l", "lab_finger_r"])
    assert "panda_hand" not in k["self_collision_buffer"] and k["self_collision_buffer"]["lab_hand"] == 0.02


def test_robot_yml_caps_the_joint_speed_at_the_lab_free_move():
    k = cp.robot_yml(json.loads(json.dumps(FRANKA)), LAB)["robot_cfg"]["kinematics"]
    vs = k["cspace"]["velocity_scale"]
    assert len(vs) == 9
    for v, lim in zip(vs[:7], cp.PANDA_V_MAX, strict=True):
        assert v * lim == pytest.approx(LAB["free_v"])
    assert vs[7:] == [1.0, 1.0]
