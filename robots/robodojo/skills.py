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

"""Semantic RoboDojo skills on top of the steerable planners: place, hand-off, push.

Each skill searches IK internally and checks its whole path before the arm
moves, then acts or reports why it cannot, so the planner does not probe
reachability itself. Registered with the steerable tools.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field

from robots.robodojo import steerable
from rpent.tools import ToolResult, tool

Camera = Literal["cam_head", "cam_left_wrist", "cam_right_wrist"]
TABLE_PROMPT = "the table"
ROBOT_PROMPT = "robot"  # SAM3 masks the most visible arm, usually the extended one
RELEASE_TOLERANCE_M = 0.02  # release only when the end-effector is this close


def _tilted(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array(
        [[1.0, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
    )


# Receiver grasps: link6 +x (approach) points down, fingers closing along world y or
# x, and the same tilted 30 or 45 degrees toward +y (away from the robot), as
# GraspGen proposes for objects near the far edge of the reach.
RECEIVER_ROTS = tuple(
    _tilted(tilt) @ base
    for tilt in (0.0, 30.0, 45.0)
    for base in (
        np.column_stack([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
        np.column_stack([[0, 0, -1], [1, 0, 0], [0, -1, 0]]),
    )
)
LEG_KEYS = (
    "moved",
    "reached",
    "pos_error_m",
    "stopped_because",
    "steps_used",
    "error",
    "failed_pose_index",
    "stopped_at_via_index",
)


def _leg(data: dict) -> dict:
    return {k: data[k] for k in LEG_KEYS if k in data}


def _arrived(data: dict) -> bool:
    return (
        bool(data.get("moved"))
        and (data.get("stopped_because") == "converged" or bool(data.get("reached")))
        and float(data.get("pos_error_m", 1.0)) <= RELEASE_TOLERANCE_M
    )


def _pose(xyz: np.ndarray, rot: np.ndarray) -> list[float]:
    return [float(v) for v in xyz] + steerable._quat_wxyz(rot)


def _ok(primitives: Any, arm: str, pose: list[float]) -> bool:
    return primitives.env.solve_ik_pose(arm, pose).get("status") == "Success"


def _transit(
    primitives: Any, arm: str, start: list[float], end: list[float], ee_z: float
) -> list[list[float]]:
    """Lift above `start`, cross at `ee_z`, arrive above `end`; [] without IK for both."""
    vias = [[*start[:2], ee_z, *start[3:7]], [*end[:2], ee_z, *end[3:7]]]
    return vias if all(_ok(primitives, arm, v) for v in vias) else []


def _scene(primitives: Any, camera: str) -> tuple[dict, dict] | ToolResult:
    """Fresh observation and camera, or an error result."""
    from robots.robodojo.tools import _refresh_obs

    obs = _refresh_obs(primitives)
    cam = obs.get("vision", {}).get(camera)
    if cam is None or cam.get("color") is None:
        return ToolResult(
            data={"error": f"camera {camera!r} color image not in observation"}
        )
    if getattr(primitives, "sam3_client", None) is None:
        return ToolResult(data={"error": "sam3_client not configured"})
    return obs, cam


def _tcp(obs: dict, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """(TCP xyz, link6 rotation) of an arm."""
    from scipy.spatial.transform import Rotation

    ee = np.asarray(obs["state"][f"{arm}_ee_pose"], dtype=np.float64)[:7]
    w, x, y, z = ee[3:7]
    rot = Rotation.from_quat([x, y, z, w]).as_matrix()
    return ee[:3] + steerable.ARX_TCP_OFFSET * rot[:, 0], rot


def _locate(primitives: Any, cam: dict, prompt: str) -> np.ndarray:
    seg = primitives.sam3_client.segment(
        cam["color"], text_prompt=prompt, min_score=0.2
    )
    return steerable.mask_points(cam, seg.mask) if seg.found else np.zeros((0, 3))


def _head_image(primitives: Any, camera: str) -> list[bytes]:
    from robots.robodojo.tools import _png

    cam = (primitives._last_obs or {}).get("vision", {}).get(camera) or {}
    return [] if cam.get("color") is None else [_png(cam["color"])]


@tool(readonly=False, exclude=("primitives", "state"))
def place(
    primitives: Any,
    state: Any,
    target_prompt: str,
    arm: Literal["left", "right"],
    held_object_prompt: str | None = None,
    camera: Camera = "cam_head",
) -> ToolResult:
    """Put the held object on or into a target and let go: plan IK-checked release poses (`plan_place`), carry the object over the target, lower it, open the gripper and lift back to the approach pose.

    It releases only after the release pose is reached; otherwise it keeps holding and
    reports why. With no reachable release pose the target is beyond this arm: use
    `handoff`. The attached image is the camera frame after the place; check that the
    object is on or inside the target.

    Args:
        target_prompt: SAM3 prompt for the place target, e.g. 'the black dustbin'
        arm: Arm holding the object
        held_object_prompt: SAM3 prompt for the held object, e.g. 'the yellow bottle'
        camera: Camera for the masks and depth"""
    from robots.robodojo.tools import _refresh_obs, gripper_report, set_gripper

    status = (
        gripper_report(_refresh_obs(primitives)["state"]).get(arm, {}).get("status")
    )
    if status != "closed_on_object":
        return ToolResult(
            data={
                "placed": False,
                "reason": f"the {arm} gripper holds nothing ({status})",
            }
        )
    plan = steerable.plan_place(
        primitives,
        state,
        target_prompt=target_prompt,
        arm=arm,
        held_object_prompt=held_object_prompt,
        camera=camera,
    )
    summary = {
        k: plan.data[k]
        for k in ("found", "target_top_z", "held_object", "rejected", "error")
        if k in plan.data
    }
    if not plan.data.get("candidates"):
        reason = plan.data.get("error") or (
            "target not found"
            if plan.data.get("found") is False
            else "no reachable release pose over the target for this arm"
        )
        return ToolResult(
            data={"placed": False, "reason": reason, "plan": summary},
            images=plan.images,
        )
    carry = None
    start = [float(v) for v in _refresh_obs(primitives)["state"][f"{arm}_ee_pose"][:7]]
    for candidate in plan.data["candidates"]:
        quat = candidate["quat_wxyz"]
        approach = candidate["approach_xyz"] + quat
        # Cross no lower than the start or the approach, so the carry does not dip.
        cross_z = max(start[2], approach[2])
        carry = steerable.move_pose(
            primitives,
            state,
            xyz=candidate["release_xyz"],
            quat_wxyz=quat,
            arm=arm,
            gripper=1,
            via_poses=[*_transit(primitives, arm, start, approach, cross_z), approach],
        ).data
        if carry.get("moved"):
            break
    if not _arrived(carry):
        return ToolResult(
            data={
                "placed": False,
                "reason": "release pose not reached; still holding the object",
                "plan": summary,
                "carry": _leg(carry),
            },
            images=_head_image(primitives, camera),
        )
    gripper = set_gripper(primitives, state, arm, -1).data["gripper"]
    retreat = steerable.move_pose(
        primitives,
        state,
        xyz=candidate["approach_xyz"],
        quat_wxyz=quat,
        arm=arm,
        gripper=-1,
    ).data
    return ToolResult(
        data={
            "placed": True,
            "plan": summary,
            "release": {
                k: candidate[k]
                for k in ("rank", "release_xyz", "drop_height_m", "edge_clearance_m")
            },
            "carry": _leg(carry),
            "gripper": gripper,
            "retreat": _leg(retreat),
        },
        images=_head_image(primitives, camera),
    )


def _table(primitives: Any, cam: dict) -> tuple[float, np.ndarray, np.ndarray] | None:
    """(top z, footprint grid at 1 cm, grid origin) of the table, or None."""
    pts = _locate(primitives, cam, TABLE_PROMPT)
    if len(pts) < 64:
        return None
    grid, origin = steerable._footprint(pts, 0.01)
    return float(np.median(pts[:, 2])), grid, origin


@tool(readonly=False, exclude=("primitives", "state", "max_ik_checks"))
def handoff(
    primitives: Any,
    state: Any,
    object_prompt: str,
    from_arm: Literal["left", "right"],
    camera: Camera = "cam_head",
    max_ik_checks: Any = 1200,
) -> ToolResult:
    """Pass the held object to the other arm through the table: put it down where both arms reach, then move the holding arm away.

    It searches free table spots where the holding arm can set the object down as held
    and the other arm can reach a grasp from above, and uses the middle of that shared
    region. The object is carried above the objects on the table and lowered until its
    bottom is 1 cm above the table or it touches the table. Afterwards grasp it with the
    other arm (`plan_grasp`). Use it when the target is beyond the holding arm's reach.

    Args:
        object_prompt: SAM3 prompt for the held object, e.g. 'the green bottle'
        from_arm: Arm holding the object
        camera: Camera for the masks and depth"""
    import cv2

    from robots.robodojo.tools import set_gripper

    scene = _scene(primitives, camera)
    if isinstance(scene, ToolResult):
        return scene
    obs, cam = scene
    to_arm = "right" if from_arm == "left" else "left"
    tcp_now, rot_now = _tcp(obs, from_arm)
    offsets, held = steerable.held_object_offsets(
        primitives.sam3_client, cam, object_prompt, tcp_now
    )
    table = _table(primitives, cam)
    if table is None:
        return ToolResult(data={"handed_off": False, "reason": "table not found"})
    table_z, grid, origin = table
    # Unmeasured: assume a short hang, so the descent ends on contact, not in a drop.
    hang = -float(offsets[:, 2].min()) if held["measured"] else 0.02
    height = max(float(np.ptp(offsets[:, 2])), 0.03)
    shape_xy = offsets[:, :2] - offsets[:, :2].mean(axis=0)
    radius = float(np.linalg.norm(shape_xy, axis=1).max())

    # Obstacles: anything standing on the table except the robot, away from both grippers.
    h, w = grid.shape
    valid = np.zeros(cam["color"].shape[:2], dtype=bool)
    valid[::4, ::4] = True
    robot = primitives.sam3_client.segment(
        cam["color"], text_prompt=ROBOT_PROMPT, min_score=0.2
    )
    if robot.found:
        valid &= ~np.asarray(robot.mask, dtype=bool)
    cloud = steerable.pixel_points(cam, valid)
    tcps = [_tcp(obs, a)[0] for a in ("left", "right")]
    keep = (cloud[:, 2] > table_z + 0.015) & (cloud[:, 2] < table_z + 0.50)
    for t in tcps:
        keep &= np.linalg.norm(cloud[:, :2] - t[:2], axis=1) > 0.15
    obstacles = cloud[keep]
    ij = np.floor((obstacles[:, :2] - origin) / 0.01).astype(int)
    ij = ij[(ij >= 0).all(axis=1) & (ij < [h, w]).all(axis=1)]
    free = np.ones((h, w), dtype=np.uint8)
    free[ij[:, 0], ij[:, 1]] = 0
    clearance = cv2.distanceTransform(free, cv2.DIST_L2, 3) * 0.01
    inside = cv2.distanceTransform(grid, cv2.DIST_L2, 3) * 0.01
    ok_cells = (inside > radius + 0.03) & (clearance > radius + 0.02)
    ok_cells[np.arange(h) % 5 != 0, :] = False  # 5 cm lattice
    ok_cells[:, np.arange(w) % 5 != 0] = False
    spots = origin + (np.argwhere(ok_cells) + 0.5) * 0.01

    def corridor_top(xy: np.ndarray) -> float:
        """Highest obstacle the carried object passes on the way to `xy`."""
        a, ab = tcp_now[:2], xy - tcp_now[:2]
        t = np.clip((obstacles[:, :2] - a) @ ab / max(float(ab @ ab), 1e-9), 0, 1)
        near = np.linalg.norm(obstacles[:, :2] - (a + t[:, None] * ab), axis=1)
        hit = obstacles[near < radius + 0.05]
        return float(hit[:, 2].max()) if len(hit) else table_z

    grip_z = max(table_z + 0.015, table_z + 0.01 + height - 0.03)
    ik_checks = 0
    rejected = {
        "receiver_cannot_grasp": 0,
        "giver_cannot_release": 0,
        "giver_cannot_carry": 0,
    }
    feasible = []
    for xy in spots:
        if ik_checks >= int(max_ik_checks):
            break
        tcp = np.array([xy[0], xy[1], grip_z])
        receivable = False
        for rot in RECEIVER_ROTS:
            ee = tcp - steerable.ARX_TCP_OFFSET * rot[:, 0]
            ik_checks += 2
            if _ok(primitives, to_arm, _pose(ee, rot)) and _ok(
                primitives, to_arm, _pose(ee - 0.10 * rot[:, 0], rot)
            ):
                receivable = True
                break
        if not receivable:
            rejected["receiver_cannot_grasp"] += 1
            continue
        top = corridor_top(xy)
        # Turns about the vertical keep the hang; the held outline turns with the wrist.
        for yaw in (0.0, 30.0, -30.0, 60.0, -60.0, 90.0):
            turn = steerable._rot_z(yaw)
            rot = turn @ rot_now
            centre_xy = (turn @ offsets.T).T[:, :2].mean(axis=0)
            tcp = np.array([*(xy - centre_xy), table_z + 0.01 + hang])
            release = tcp - steerable.ARX_TCP_OFFSET * rot[:, 0]
            give = [_pose(release, rot), _pose(release + [0.0, 0.0, 0.10], rot)]
            ik_checks += 2
            if not all(_ok(primitives, from_arm, p) for p in give):
                rejected["giver_cannot_release"] += 1
                continue
            cross_tcp_z = max(float(tcp_now[2]), top + hang + 0.05, tcp[2] + 0.10)
            cross = [*release[:2], cross_tcp_z - steerable.ARX_TCP_OFFSET * rot[2, 0]]
            ik_checks += 1
            if not _ok(primitives, from_arm, _pose(cross, rot)):
                rejected["giver_cannot_carry"] += 1
                continue
            feasible.append((xy, [*give, _pose(cross, rot)], cross_tcp_z))
            break
    if not feasible:
        return ToolResult(
            data={
                "handed_off": False,
                "reason": "no free table spot where both arms reach",
                "free_spots_checked": len(spots),
                "rejected": rejected,
                "ik_checks": ik_checks,
                "held_object": held,
            }
        )
    centre = np.mean([f[0] for f in feasible], axis=0)
    xy, (release, approach, cross), cross_tcp_z = min(
        feasible, key=lambda f: float(np.linalg.norm(f[0] - centre))
    )
    lift_now = _pose(
        [
            *(tcp_now - steerable.ARX_TCP_OFFSET * rot_now[:, 0])[:2],
            cross_tcp_z - steerable.ARX_TCP_OFFSET * float(rot_now[2, 0]),
        ],
        rot_now,
    )
    vias = [lift_now] if _ok(primitives, from_arm, lift_now) else []
    lower = steerable.move_pose(
        primitives,
        state,
        xyz=release[:3],
        quat_wxyz=release[3:],
        arm=from_arm,
        gripper=1,
        via_poses=[*vias, cross, approach],
    ).data
    # A stall on the final descent means the object rests on the table.
    landed = (
        bool(lower.get("moved"))
        and "stopped_at_via_index" not in lower
        and lower.get("stopped_because") in ("converged", "stalled")
    )
    if not landed:
        return ToolResult(
            data={
                "handed_off": False,
                "reason": "release pose not reached; still holding the object",
                "lower": _leg(lower),
            },
            images=_head_image(primitives, camera),
        )
    gripper = set_gripper(primitives, state, from_arm, -1).data["gripper"]
    # Clear the object for the receiver and the camera: lift and move outward to the
    # giver's side, else return to where the hand-off started, else the approach pose.
    lift = list(approach)
    lift[2] += 0.05
    away = list(lift)
    away[0] += -0.15 if from_arm == "left" else 0.15
    start = [float(v) for v in obs["state"][f"{from_arm}_ee_pose"][:7]]
    for target, vias in ((away, [lift]), (start, [approach]), (approach, [])):
        retreat = steerable.move_pose(
            primitives,
            state,
            xyz=target[:3],
            quat_wxyz=target[3:],
            arm=from_arm,
            gripper=-1,
            via_poses=vias,
        ).data
        if retreat.get("moved"):
            break
    from robots.robodojo.tools import _refresh_obs

    now = _locate(primitives, _refresh_obs(primitives)["vision"][camera], object_prompt)
    return ToolResult(
        data={
            "handed_off": True,
            "receiver": to_arm,
            "planned_xy": [round(float(v), 4) for v in xy],
            "object_centroid_xyz": [round(float(v), 4) for v in now.mean(axis=0)]
            if len(now)
            else None,
            "table_z": round(table_z, 4),
            "shared_spots": len(feasible),
            "lower": _leg(lower),
            "gripper": gripper,
            "retreat": _leg(retreat),
            "next": f"grasp it with the {to_arm} arm (plan_grasp)",
        },
        images=_head_image(primitives, camera),
    )


@tool(readonly=False, exclude=("primitives", "state"))
def push(
    primitives: Any,
    state: Any,
    object_prompt: str,
    arm: Literal["left", "right"],
    target_xy: Annotated[list[float], Field(min_length=2, max_length=2)] | None = None,
    direction_xy: Annotated[list[float], Field(min_length=2, max_length=2)]
    | None = None,
    distance_m: float = 0.10,
    camera: Camera = "cam_head",
) -> ToolResult:
    """Slide an object along the table with the closed gripper pointing down, fingertips 2 cm above the table.

    The gripper starts behind the object and moves straight toward `target_xy` (the
    object's new centre), or along `direction_xy` for `distance_m` (at most 0.30 m). The
    whole path is IK-checked before moving; the result reports where the object ended up.
    A gripper that was open is opened again after the push. Use it to bring an object
    within reach or away from an obstacle.

    Args:
        object_prompt: SAM3 prompt for the object, e.g. 'the toy car'
        arm: Arm that pushes
        target_xy: World xy for the object's centre after the push
        direction_xy: World xy push direction, used with distance_m
        distance_m: Push distance along direction_xy
        camera: Camera for the masks and depth"""
    if (target_xy is None) == (direction_xy is None):
        return ToolResult(
            data={"error": "give exactly one of target_xy or direction_xy"}
        )
    scene = _scene(primitives, camera)
    if isinstance(scene, ToolResult):
        return scene
    obs, cam = scene
    from robots.robodojo.tools import _refresh_obs, gripper_report, set_gripper

    status = gripper_report(obs["state"]).get(arm, {}).get("status")
    if status == "closed_on_object":
        return ToolResult(
            data={"pushed": False, "reason": f"the {arm} arm holds an object"}
        )
    was_open = status == "open"
    pts = _locate(primitives, cam, object_prompt)
    if len(pts) < 32:
        return ToolResult(data={"pushed": False, "reason": "object not found"})
    centre = pts[:, :2].mean(axis=0)
    if target_xy is not None:
        delta = np.asarray(target_xy, dtype=np.float64) - centre
        distance = float(np.linalg.norm(delta))
    else:
        delta = np.asarray(direction_xy, dtype=np.float64)
        distance = float(distance_m)
    norm = float(np.linalg.norm(delta))
    if norm < 1e-6 or not 0.0 < distance <= 0.30:
        return ToolResult(data={"error": "push distance must be in (0, 0.30] m"})
    d = delta / norm
    table = _table(primitives, cam)
    table_z = table[0] if table is not None else float(pts[:, 2].min())
    back = float(((pts[:, :2] - centre) @ -d).max())
    start = centre - d * (back + 0.04)
    end = start + d * (0.04 + distance)
    x_axis = np.array([0.0, 0.0, -1.0])
    y_axis = np.array([-d[1], d[0], 0.0])
    rot = np.column_stack([x_axis, y_axis, np.cross(x_axis, y_axis)])
    lift = np.array([0.0, 0.0, 0.10])

    def ee(xy: np.ndarray) -> np.ndarray:
        return np.array([xy[0], xy[1], table_z + 0.02 + steerable.ARX_TCP_OFFSET])

    path = [ee(start) + lift, ee(start), ee(end), ee(end) + lift]
    names = ["above_start", "contact_start", "push_end", "lift"]
    move = steerable.move_pose(
        primitives,
        state,
        xyz=path[-1],
        quat_wxyz=steerable._quat_wxyz(rot),
        arm=arm,
        gripper=1,
        via_poses=[_pose(p, rot) for p in path[:-1]],
    ).data
    if not move.get("moved"):
        index = move.get("failed_pose_index")
        reason = f"no IK for {names[index]}" if index is not None else move.get("error")
        return ToolResult(data={"pushed": False, "reason": reason, "move": _leg(move)})
    if was_open:
        set_gripper(primitives, state, arm, -1)
    after = _locate(
        primitives, _refresh_obs(primitives)["vision"][camera], object_prompt
    )
    result: dict[str, Any] = {
        "pushed": True,
        "start_centroid_xy": [round(float(v), 4) for v in centre],
        "planned_distance_m": round(distance, 3),
        "move": _leg(move),
    }
    if len(after):
        moved = after[:, :2].mean(axis=0) - centre
        result["end_centroid_xy"] = [round(float(v), 4) for v in centre + moved]
        result["moved_along_push_m"] = round(float(moved @ d), 3)
    else:
        result["note"] = "object not found after the push; check the image"
    return ToolResult(data=result, images=_head_image(primitives, camera))
