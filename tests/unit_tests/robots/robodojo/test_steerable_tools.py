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

"""Offline checks for the steerable RoboDojo tools (no Isaac, no GraspGen)."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robots.robodojo import steerable

HOME = [0.3, -0.2, 1.1, 1.0, 0.0, 0.0, 0.0]


def _rot(quat_wxyz):
    w, x, y, z = quat_wxyz
    return Rotation.from_quat([x, y, z, w]).as_matrix()


class FakeEnv:
    """Joint values equal the requested pose, so a reached joint goal is the pose."""

    def __init__(self, unreachable=()):
        self.state = {
            "right_ee_pose": list(HOME),
            "right_arm_joint_state": list(HOME),
            "left_arm_joint_state": [0.0] * 7,
            "right_ee_joint_state": [1.0],
            "left_ee_joint_state": [1.0],
        }
        self.unreachable = [np.asarray(p) for p in unreachable]
        self.actions = []
        self.ik_calls = []

    def get_obs(self):
        return {"state": dict(self.state), "vision": {}}

    def solve_ik_pose(self, arm, pose):
        self.ik_calls.append((arm, list(pose)))
        if any(np.allclose(pose[:3], p, atol=1e-3) for p in self.unreachable):
            return {"status": "Fail", "error": "no IK solution for this pose"}
        return {
            "status": "Success",
            "joint_value": list(pose),
            "joint_displacement": 0.1,
        }

    def step(self, action):
        self.actions.append(action)
        joints = action["right_arm_joint_state"]
        self.state["right_arm_joint_state"] = list(joints)
        self.state["right_ee_pose"] = list(joints)
        self.state["right_ee_joint_state"] = action["right_ee_joint_state"]
        return (
            self.get_obs(),
            0.0,
            False,
            {"status": {"step": len(self.actions), "step_limit": 500}},
        )


def _primitives(env, sam3=None):
    return SimpleNamespace(
        env=env, sam3_client=sam3, _last_obs=None, _check_cancelled=lambda: None
    )


def test_top_down_grasp_maps_to_link6_pose_with_tcp_offset():
    grasp = np.eye(4)
    grasp[:3, 0] = [1, 0, 0]  # closing axis
    grasp[:3, 1] = [0, -1, 0]
    grasp[:3, 2] = [0, 0, -1]  # approach straight down
    grasp[:3, 3] = [0.5, 0.0, 1.0 + steerable.GRASPGEN_DEPTH]

    first, flipped = steerable.grasp_to_ee_targets(grasp)

    assert first["tcp_xyz"] == pytest.approx([0.5, 0.0, 1.0])
    assert first["grasp_xyz"] == pytest.approx(
        [0.5, 0.0, 1.0 + steerable.ARX_TCP_OFFSET]
    )
    assert first["pregrasp_xyz"] == pytest.approx([0.5, 0.0, 1.245])
    for target, closing in ((first, [1, 0, 0]), (flipped, [-1, 0, 0])):
        rot = _rot(target["quat_wxyz"])
        assert rot[:, 0] == pytest.approx(
            [0, 0, -1], abs=1e-4
        )  # link6 +x along the approach
        assert rot[:, 1] == pytest.approx(
            closing, abs=1e-4
        )  # fingers close along link6 y
        assert target["grasp_xyz"] == first["grasp_xyz"]


def test_mask_points_back_projects_like_back_project():
    depth = np.full((3, 3), 2.0)
    cam = {
        "distance_to_image_plane": depth,
        "intrinsic_matrix": [[100.0, 0, 1.0], [0, 100.0, 1.0], [0, 0, 1]],
        "extrinsic_matrix": np.eye(4),
    }
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True

    assert steerable.mask_points(cam, mask) == pytest.approx(
        np.array([[0.0, 0.0, -2.0]])
    )


def test_move_pose_follows_straight_waypoints_to_the_pose():
    env = FakeEnv()
    target_quat = [0.0, 0.0, 1.0, 0.0]
    target_xyz = [0.3, -0.2, 0.95]

    result = steerable.move_pose(
        primitives=_primitives(env),
        state=None,
        xyz=target_xyz,
        quat_wxyz=target_quat,
        gripper=-1,
    ).data

    assert result["reached"] and result["moved"]
    assert result["waypoints"] == 5  # 15 cm in <=3 cm segments
    xs = [a["right_arm_joint_state"][:3] for a in env.actions]
    assert all(x[0] == pytest.approx(0.3) and x[1] == pytest.approx(-0.2) for x in xs)
    assert all(a["right_ee_joint_state"] == [1.0] for a in env.actions)  # -1 = open
    assert all(a["left_arm_joint_state"] == [0.0] * 7 for a in env.actions)


def test_move_pose_does_not_move_when_the_final_pose_is_unreachable():
    target_xyz = [0.9, 0.0, 0.5]
    env = FakeEnv(unreachable=[target_xyz])

    result = steerable.move_pose(
        primitives=_primitives(env), state=None, xyz=target_xyz, quat_wxyz=[1, 0, 0, 0]
    ).data

    assert result["moved"] is False and env.actions == []


def test_plan_grasp_keeps_reachable_grasps_and_rejects_ones_from_below(monkeypatch):
    depth = np.full((8, 8), 2.0)
    cam = {
        "color": np.zeros((8, 8, 3), dtype=np.uint8),
        "distance_to_image_plane": depth,
        "intrinsic_matrix": [[100.0, 0, 4.0], [0, 100.0, 4.0], [0, 0, 1]],
        "extrinsic_matrix": np.eye(4),
    }
    env = FakeEnv()
    env.get_obs = lambda: {"state": dict(env.state), "vision": {"cam_head": cam}}
    sam3 = SimpleNamespace(
        segment=lambda color, text_prompt, min_score: SimpleNamespace(
            found=True, mask=np.ones((8, 8), dtype=bool), score=0.9
        )
    )
    down = np.eye(4)
    down[:3, :3] = np.column_stack([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    down[:3, 3] = [
        0.0,
        0.0,
        -1.9 + steerable.GRASPGEN_DEPTH,
    ]  # TCP 10 cm above the object floor
    up = np.eye(4)
    up[:3, :3] = np.column_stack([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    monkeypatch.setattr(
        steerable,
        "graspgen_grasps",
        lambda points: (np.stack([up, down]), np.array([0.9, 0.8])),
    )

    result = steerable.plan_grasp(
        primitives=_primitives(env, sam3),
        state=None,
        text_prompt="the bottle",
        arm="right",
    ).data

    assert result["rejected"]["from_below"] == 1
    assert len(result["candidates"]) == 1
    best = result["candidates"][0]
    assert best["arm"] == "right" and best["graspgen_score"] == pytest.approx(0.8)
    assert best["pregrasp_xyz"][2] > best["grasp_xyz"][2]
