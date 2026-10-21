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

"""Offline checks for the RoboDojo skills (no Isaac, no SAM3)."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robots.robodojo import skills, steerable

# Camera at the origin looking along -z (identity extrinsic): depth d maps to z = -d.
# At d = 2 m with fx = 400, the 100 px image spans x, y in [-0.25, 0.25].
TABLE_Z = -2.0


def _tcp(pose):
    w, x, y, z = pose[3:7]
    rot = Rotation.from_quat([x, y, z, w]).as_matrix()
    return np.asarray(pose[:3]) + steerable.ARX_TCP_OFFSET * rot[:, 0]


class TwoArmEnv:
    """Joint values equal the requested pose; arms reach instantly unless frozen."""

    def __init__(self, cam=None, reach=lambda arm, tcp: True, frozen=()):
        self.cam = cam
        self.reach = reach
        self.frozen = set(frozen)
        self.state = {}
        for arm, x in (("left", -0.1), ("right", 0.1)):
            pose = [
                x - steerable.ARX_TCP_OFFSET,
                0.0,
                TABLE_Z + 0.2,
                1.0,
                0.0,
                0.0,
                0.0,
            ]
            self.state[f"{arm}_ee_pose"] = list(pose)
            self.state[f"{arm}_arm_joint_state"] = list(pose)
            self.state[f"{arm}_ee_joint_state"] = [0.0]  # closed, holding
        self.events = []  # ("move", arm, pose) or ("gripper", arm, value)

    def get_obs(self):
        vision = {} if self.cam is None else {"cam_head": self.cam}
        return {"state": {k: list(v) for k, v in self.state.items()}, "vision": vision}

    def solve_ik_pose(self, arm, pose):
        if not self.reach(arm, _tcp(pose)):
            return {"status": "Fail", "error": "no IK solution"}
        return {
            "status": "Success",
            "joint_value": list(pose),
            "joint_displacement": 0.1,
        }

    def get_status(self):
        return {"step": len(self.events), "step_limit": 10_000}

    def step(self, action):
        for arm in ("left", "right"):
            joints = action.get(f"{arm}_arm_joint_state")
            if joints is not None and arm not in self.frozen:
                if list(joints) != self.state[f"{arm}_arm_joint_state"]:
                    self.events.append(("move", arm, list(joints)))
                self.state[f"{arm}_arm_joint_state"] = list(joints)
                self.state[f"{arm}_ee_pose"] = list(joints)
            grip = action.get(f"{arm}_ee_joint_state")
            if grip is not None and grip != self.state[f"{arm}_ee_joint_state"]:
                self.events.append(("gripper", arm, grip[0]))
                self.state[f"{arm}_ee_joint_state"] = list(grip)
        return self.get_obs(), 0.0, False, {"status": self.get_status()}


def _segmenter(masks):
    return SimpleNamespace(
        segment=lambda color, text_prompt, min_score: SimpleNamespace(
            found=text_prompt in masks, mask=masks.get(text_prompt), score=0.9
        )
    )


def _camera(depth):
    return {
        "color": np.zeros((*depth.shape, 3), dtype=np.uint8),
        "distance_to_image_plane": depth,
        "intrinsic_matrix": [[400.0, 0, 50.0], [0, 400.0, 50.0], [0, 0, 1]],
        "extrinsic_matrix": np.eye(4),
    }


def _primitives(env, sam3=None):
    return SimpleNamespace(
        env=env, sam3_client=sam3, _last_obs=None, _check_cancelled=lambda: None
    )


def _place_plan(monkeypatch, candidates):
    data = {"found": True, "target_top_z": TABLE_Z, "candidates": candidates}
    monkeypatch.setattr(
        steerable,
        "plan_place",
        lambda *args, **kwargs: SimpleNamespace(data=data, images=[]),
    )


CANDIDATE = {
    "rank": 1,
    "release_xyz": [-0.2, 0.0, TABLE_Z + 0.1],
    "approach_xyz": [-0.2, 0.0, TABLE_Z + 0.2],
    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    "drop_height_m": 0.03,
    "edge_clearance_m": 0.1,
}


def test_move_pose_checks_every_via_pose_before_moving():
    blocked = [0.0, 0.0, TABLE_Z + 0.3]
    env = TwoArmEnv(reach=lambda arm, tcp: not np.allclose(tcp, blocked))
    via = [*(np.asarray(blocked) - [steerable.ARX_TCP_OFFSET, 0, 0]), 1.0, 0, 0, 0]

    data = steerable.move_pose(
        _primitives(env),
        None,
        xyz=[0.0, 0.0, TABLE_Z + 0.1],
        quat_wxyz=[1.0, 0.0, 0.0, 0.0],
        arm="left",
        via_poses=[via],
    ).data

    assert data["moved"] is False and data["failed_pose_index"] == 0
    assert env.events == []


def test_move_pose_passes_through_via_poses_in_order():
    env = TwoArmEnv()
    via = [-0.3, 0.1, TABLE_Z + 0.3, 1.0, 0.0, 0.0, 0.0]

    data = steerable.move_pose(
        _primitives(env),
        None,
        xyz=[-0.3, 0.1, TABLE_Z + 0.2],
        quat_wxyz=[1.0, 0.0, 0.0, 0.0],
        arm="left",
        via_poses=[via],
    ).data

    assert data["reached"] and data["via_poses"] == 1
    poses = [e[2] for e in env.events if e[0] == "move"]
    via_at = next(i for i, p in enumerate(poses) if np.allclose(p, via))
    assert via_at < len(poses) - 1 and np.allclose(poses[-1][:3], data["target_xyz"])


def test_place_releases_only_at_the_planned_pose_then_retreats(monkeypatch):
    _place_plan(monkeypatch, [CANDIDATE])
    env = TwoArmEnv()

    result = skills.place(_primitives(env), None, target_prompt="the bin", arm="left")

    assert result.data["placed"] is True
    opened = env.events.index(("gripper", "left", 1.0))
    before = [e[2] for e in env.events[:opened] if e[0] == "move"]
    after = [e[2] for e in env.events[opened:] if e[0] == "move"]
    assert before[-1][:3] == pytest.approx(CANDIDATE["release_xyz"])
    assert after[-1][:3] == pytest.approx(CANDIDATE["approach_xyz"])


def test_place_keeps_holding_when_the_carry_does_not_arrive(monkeypatch):
    _place_plan(monkeypatch, [CANDIDATE])
    env = TwoArmEnv(frozen={"left"})

    data = skills.place(
        _primitives(env), None, target_prompt="the bin", arm="left"
    ).data

    assert data["placed"] is False and "still holding" in data["reason"]
    assert not any(e[0] == "gripper" for e in env.events)


def test_place_points_to_handoff_when_no_release_pose_is_reachable(monkeypatch):
    _place_plan(monkeypatch, [])

    data = skills.place(
        _primitives(TwoArmEnv()), None, target_prompt="the bin", arm="left"
    ).data

    assert data["placed"] is False and "no reachable release pose" in data["reason"]


def _handoff_scene():
    depth = np.full((100, 100), 2.0)
    held = np.zeros((100, 100), dtype=bool)
    held[46:54, 26:34] = True  # under the left TCP at x = -0.1
    depth[held] = 1.9  # 10 cm above the table, hanging from the gripper
    table = ~held
    cam = _camera(depth)
    # Left reaches x < 0.05, right reaches x > -0.05: they share a 10 cm band.
    env = TwoArmEnv(
        cam, reach=lambda arm, tcp: tcp[0] < 0.05 if arm == "left" else tcp[0] > -0.05
    )
    env.state["left_ee_pose"][2] = env.state["left_arm_joint_state"][2] = -1.85
    return env, _segmenter({"the bottle": held, "the table": table})


def test_handoff_sets_the_object_down_where_both_arms_reach():
    env, sam3 = _handoff_scene()

    data = skills.handoff(
        _primitives(env, sam3), None, object_prompt="the bottle", from_arm="left"
    ).data

    assert data["handed_off"] is True and data["receiver"] == "right"
    assert -0.05 <= data["planned_xy"][0] <= 0.05
    opened = env.events.index(("gripper", "left", 1.0))
    release = [e[2] for e in env.events[:opened] if e[0] == "move"][-1]
    # The object's bottom (5 cm below the TCP) ends 1 cm above the table.
    assert _tcp(release)[2] - 0.05 == pytest.approx(TABLE_Z + 0.01, abs=2e-3)
    assert data["retreat"]["moved"] is True


def test_handoff_reports_when_no_spot_is_shared():
    env, sam3 = _handoff_scene()
    env.reach = lambda arm, tcp: tcp[0] < -0.1 if arm == "left" else tcp[0] > 0.1

    data = skills.handoff(
        _primitives(env, sam3), None, object_prompt="the bottle", from_arm="left"
    ).data

    assert data["handed_off"] is False and "both arms" in data["reason"]
    assert not any(e[0] == "gripper" for e in env.events)


def _push_scene(reach=lambda arm, tcp: True):
    depth = np.full((100, 100), 2.0)
    car = np.zeros((100, 100), dtype=bool)
    car[45:55, 45:55] = True  # 5 cm square at the table centre
    depth[car] = 1.95
    env = TwoArmEnv(_camera(depth), reach=reach)
    return env, _segmenter({"the car": car, "the table": ~car})


def test_push_starts_behind_the_object_and_moves_along_the_direction():
    env, sam3 = _push_scene()

    data = skills.push(
        _primitives(env, sam3),
        None,
        object_prompt="the car",
        arm="right",
        direction_xy=[1.0, 0.0],
        distance_m=0.1,
    ).data

    assert data["pushed"] is True
    tcps = [_tcp(e[2]) for e in env.events if e[0] == "move"]
    low = [t for t in tcps if t[2] == pytest.approx(TABLE_Z + 0.02)]
    assert low[0][0] < -0.025 and low[-1][0] - low[0][0] == pytest.approx(
        0.14, abs=1e-3
    )
    assert all(t[1] == pytest.approx(0.0, abs=5e-3) for t in low)
    assert env.state["right_ee_joint_state"] == [0.0]  # closed while pushing


def test_push_reports_the_unreachable_part_of_the_path():
    env, sam3 = _push_scene(reach=lambda arm, tcp: tcp[0] < 0.05)

    data = skills.push(
        _primitives(env, sam3),
        None,
        object_prompt="the car",
        arm="right",
        direction_xy=[1.0, 0.0],
        distance_m=0.1,
    ).data

    assert data["pushed"] is False and data["reason"] == "no IK for push_end"
    assert env.events == []
