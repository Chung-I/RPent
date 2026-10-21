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

"""Steerable RoboDojo tools: reachability checks, full-pose motion, GraspGen grasps.

Frames (ARX X5): the observed ``<arm>_ee_pose`` is the ``link6`` pose (xyz +
wxyz). Its +x axis points out through the fingers, the fingers close along y,
and the tool centre point (TCP) lies ``gripper_bias`` (0.145 m) along +x.
GraspGen's convention: approach along +z, fingers close along x, and the
fingertips lie ``GRASPGEN_DEPTH`` along +z from the grasp frame origin.

The toolkit registers these tools only when ``RPENT_ROBODOJO_STEERABLE=1``;
``plan_grasp`` also needs a GraspGen server at ``RPENT_GRASPGEN_URL``.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field

from rpent.tools import ToolResult, tool

GRASPGEN_DEPTH = 0.1034  # franka_panda gripper_depth in GraspGen's checkpoint config
ARX_TCP_OFFSET = 0.145  # ARX X5 gripper_bias along link6 +x


def steerable_enabled() -> bool:
    """Whether the toolkit registers these tools (``RPENT_ROBODOJO_STEERABLE=1``)."""
    return os.environ.get("RPENT_ROBODOJO_STEERABLE") == "1"


def mask_points(
    cam: dict, mask: np.ndarray, max_points: int = 2048, seed: int = 0
) -> np.ndarray:
    """World points under a mask, using the ``back_project`` camera convention.

    Isaac cameras look along -Z and ``distance_to_image_plane`` is the
    optical-axis distance; image rows grow downward while the camera +Y is up.
    Far outliers (mask bleed onto the background) are removed by distance to
    the median point.
    """
    depth = cam.get("distance_to_image_plane")
    if depth is None:
        depth = cam.get("depth")
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[..., 0]
    K = np.asarray(cam["intrinsic_matrix"], dtype=np.float64)
    T = np.asarray(cam["extrinsic_matrix"], dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != depth.shape:
        raise ValueError(f"mask shape {mask.shape} != depth shape {depth.shape}")
    rows, cols = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    if rows.size == 0:
        return np.zeros((0, 3))
    d = depth[rows, cols]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    p_cam = np.stack(
        [(cols - cx) / fx * d, -(rows - cy) / fy * d, -d, np.ones_like(d)], axis=1
    )
    pts = (T @ p_cam.T).T[:, :3]
    centre = np.median(pts, axis=0)
    dist = np.linalg.norm(pts - centre, axis=1)
    mad = np.median(np.abs(dist - np.median(dist))) + 1e-6
    pts = pts[dist <= np.median(dist) + 4.0 * mad]
    if len(pts) > max_points:
        pts = pts[
            np.random.default_rng(seed).choice(len(pts), max_points, replace=False)
        ]
    return pts


def _encode(a: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, np.asarray(a), allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode()


def _decode(s: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(s)), allow_pickle=False)


def graspgen_grasps(
    points: np.ndarray, url: str | None = None, timeout_s: float = 60.0
):
    """Query a GraspGen server; returns (poses [N,4,4] in the cloud's frame, scores [N])."""
    url = url or os.environ.get("RPENT_GRASPGEN_URL")
    if not url:
        raise RuntimeError("GraspGen server not configured (set RPENT_GRASPGEN_URL)")
    body = json.dumps({"points": _encode(points.astype(np.float32))}).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/grasps",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        out = json.loads(resp.read())
    return _decode(out["poses"]), _decode(out["scores"])


def _quat_wxyz(rot: np.ndarray) -> list[float]:
    from scipy.spatial.transform import Rotation as _R

    x, y, z, w = _R.from_matrix(rot).as_quat()
    return [float(w), float(x), float(y), float(z)]


def grasp_to_ee_targets(
    grasp: np.ndarray, pregrasp_m: float = 0.10
) -> list[dict[str, Any]]:
    """ARX ``link6`` grasp and pre-grasp poses for one GraspGen grasp.

    Returns both antipodal variants (rotated 180 degrees about the approach
    axis); either one closes the fingers across the same contact points.
    """
    grasp = np.asarray(grasp, dtype=np.float64)
    x_g, y_g, z_g = grasp[:3, 0], grasp[:3, 1], grasp[:3, 2]
    tcp = grasp[:3, 3] + GRASPGEN_DEPTH * z_g
    ee_xyz = tcp - ARX_TCP_OFFSET * z_g
    targets = []
    for sign in (1.0, -1.0):
        rot = np.stack(
            [z_g, sign * x_g, sign * y_g], axis=1
        )  # ee x=approach, y=closing, z=x×y
        targets.append(
            {
                "tcp_xyz": [round(float(v), 4) for v in tcp],
                "grasp_xyz": [round(float(v), 4) for v in ee_xyz],
                "pregrasp_xyz": [round(float(v), 4) for v in ee_xyz - pregrasp_m * z_g],
                "quat_wxyz": [round(v, 5) for v in _quat_wxyz(rot)],
                "approach_dir": [round(float(v), 3) for v in z_g],
                "closing_axis": [round(float(v), 3) for v in sign * x_g],
            }
        )
    return targets


def rotation_error_deg(q_a: list[float], q_b: list[float]) -> float:
    """Angle between two wxyz quaternions in degrees."""
    a = np.asarray(q_a, dtype=np.float64)
    b = np.asarray(q_b, dtype=np.float64)
    dot = abs(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))))
    return float(np.degrees(2.0 * np.arccos(min(1.0, dot))))


def _hold_action(obs: dict, arm: str, joints: list[float], gripper: float) -> dict:
    """Joint action for ``arm``; the other arm and its gripper hold their state."""
    state = obs.get("state", {})
    action: dict = {f"{arm}_arm_joint_state": [float(v) for v in joints]}
    for a in ("left", "right"):
        other = state.get(f"{a}_arm_joint_state")
        if a != arm and other is not None:
            action[f"{a}_arm_joint_state"] = [
                float(v) for v in np.asarray(other).reshape(-1)
            ]
        joint = state.get(f"{a}_ee_joint_state")
        if joint is None:
            continue
        if a == arm and gripper != 0:
            # Env convention: normalized 1.0 = OPEN, 0.0 = CLOSED; tool: 1=close, -1=open.
            action[f"{a}_ee_joint_state"] = [0.0 if gripper > 0 else 1.0]
        else:
            action[f"{a}_ee_joint_state"] = [float(np.asarray(joint).reshape(-1)[0])]
    return action


def slerp_waypoints(
    start: np.ndarray, target: np.ndarray, step_m: float = 0.03, max_n: int = 8
) -> list[list[float]]:
    """Straight-line xyz + slerped wxyz poses from ``start`` (exclusive) to ``target``."""
    from scipy.spatial.transform import Rotation as _R
    from scipy.spatial.transform import Slerp

    start = np.asarray(start, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n = int(
        np.clip(
            np.ceil(np.linalg.norm(target[:3] - start[:3]) / step_m - 1e-9), 1, max_n
        )
    )
    rots = _R.from_quat([[q[1], q[2], q[3], q[0]] for q in (start[3:7], target[3:7])])
    slerp = Slerp([0.0, 1.0], rots)
    out = []
    for t in np.linspace(0.0, 1.0, n + 1)[1:]:
        x, y, z, w = slerp([t]).as_quat()[0]
        xyz = start[:3] + t * (target[:3] - start[:3])
        out.append([float(v) for v in xyz] + [float(w), float(x), float(y), float(z)])
    return out


@tool(readonly=True, exclude=("primitives", "state"))
def check_reach(
    primitives: Any,
    state: Any,
    xyz: Annotated[list[float], Field(min_length=3, max_length=3)],
    quat_wxyz: Annotated[list[float], Field(min_length=4, max_length=4)] | None = None,
    arm: Literal["left", "right", "both"] = "both",
) -> ToolResult:
    """Check, without moving, whether an end-effector (link6) target is reachable by CuRobo IK.

    Without `quat_wxyz` it tests position only (several wrist orientations, like `move_to`);
    with it, it tests the exact pose (like `move_pose`).

    Args:
        xyz: World xyz of the end-effector link
        quat_wxyz: Optional end-effector orientation (w, x, y, z)
        arm: Arm to test, or both"""
    out = {}
    for a in ("left", "right") if arm == "both" else (arm,):
        if quat_wxyz is None:
            res = primitives.env.solve_ik_position(a, [float(v) for v in xyz])
        else:
            pose = [float(v) for v in xyz] + [float(v) for v in quat_wxyz]
            res = primitives.env.solve_ik_pose(a, pose)
        out[a] = {
            "reachable": res.get("status") == "Success",
            "joint_displacement": res.get("joint_displacement"),
            "error": res.get("error"),
        }
    return ToolResult(data={"xyz": list(xyz), "quat_wxyz": quat_wxyz, "arms": out})


@tool(readonly=False, exclude=("primitives", "state", "settle_steps"))
def move_pose(
    primitives: Any,
    state: Any,
    xyz: Annotated[list[float], Field(min_length=3, max_length=3)],
    quat_wxyz: Annotated[list[float], Field(min_length=4, max_length=4)],
    arm: Literal["left", "right"] = "right",
    gripper: float = 0,
    settle_steps: Any = 6,
) -> ToolResult:
    """Move an end-effector (link6) to a full world pose (position + wrist orientation) with CuRobo IK.

    The path is a straight line split into ~3 cm IK waypoints, so a pre-grasp -> grasp move
    approaches along the gripper axis. Use the poses returned by `plan_grasp`. The arm does
    not move when the final pose has no IK solution.

    Args:
        xyz: World xyz of the end-effector link
        quat_wxyz: End-effector orientation (w, x, y, z)
        arm: Arm to move
        gripper: 1=close, -1=open, 0=keep current"""
    from robots.robodojo.tools import _refresh_obs

    target = np.asarray(
        [float(v) for v in xyz] + [float(v) for v in quat_wxyz], dtype=np.float64
    )
    norm = float(np.linalg.norm(target[3:]))
    if not np.isfinite(norm) or norm < 1e-6:
        return ToolResult(data={"error": "quat_wxyz has zero norm"})
    target[3:] /= norm
    obs = _refresh_obs(primitives)
    start_ee = obs.get("state", {}).get(f"{arm}_ee_pose")
    if start_ee is None:
        return ToolResult(data={"error": f"{arm} ee_pose not in state"})
    start = np.asarray(start_ee, dtype=np.float64)[:7]
    final_ik = primitives.env.solve_ik_pose(arm, [float(v) for v in target])
    if final_ik.get("status") != "Success":
        error = final_ik.get("error", "no IK solution for target pose")
        return ToolResult(data={"arm": arm, "moved": False, "error": error})
    plan = []
    skipped = 0
    for waypoint in slerp_waypoints(start, target)[:-1]:
        ik = primitives.env.solve_ik_pose(arm, waypoint)
        if ik.get("status") == "Success":
            plan.append(ik["joint_value"])
        else:
            skipped += 1
    n_waypoints = len(plan) + 1
    plan += [final_ik["joint_value"]] * (1 + int(settle_steps))
    goal = np.asarray(final_ik["joint_value"], dtype=np.float64)
    final = start
    steps_used = 0
    for i, joints in enumerate(plan):
        primitives._check_cancelled()
        obs, _reward, _done, info = primitives.env.step(
            _hold_action(obs, arm, joints, gripper)
        )
        primitives._last_obs = obs
        steps_used += 1
        st = obs.get("state", {})
        if st.get(f"{arm}_ee_pose") is not None:
            final = np.asarray(st[f"{arm}_ee_pose"], dtype=np.float64)[:7]
        joints_now = st.get(f"{arm}_arm_joint_state")
        if i >= n_waypoints - 1 and joints_now is not None:
            q = np.asarray(joints_now, dtype=np.float64).reshape(-1)[: len(goal)]
            if float(np.max(np.abs(q - goal))) < 0.01:
                break
        if info["status"].get("step", 0) >= info["status"].get("step_limit", 1):
            break
    pos_err = float(np.linalg.norm(final[:3] - target[:3]))
    rot_err = rotation_error_deg(list(final[3:]), list(target[3:]))
    return ToolResult(
        data={
            "arm": arm,
            "moved": True,
            "target_xyz": [round(float(v), 4) for v in target[:3]],
            "final_xyz": [round(float(v), 4) for v in final[:3]],
            "final_quat_wxyz": [round(float(v), 4) for v in final[3:]],
            "pos_error_m": round(pos_err, 4),
            "rot_error_deg": round(rot_err, 2),
            "reached": pos_err <= 0.01 and rot_err <= 5.0,
            "waypoints": n_waypoints,
            "waypoints_without_ik": skipped,
            "steps_used": steps_used,
        }
    )


@tool(readonly=True, exclude=("primitives", "state", "max_ik_checks"))
def plan_grasp(
    primitives: Any,
    state: Any,
    text_prompt: str,
    camera: Literal["cam_head", "cam_left_wrist", "cam_right_wrist"] = "cam_head",
    arm: Literal["left", "right", "both"] = "both",
    top_k: int = 3,
    min_score: float = 0.2,
    max_ik_checks: Any = 160,
) -> ToolResult:
    """Plan 6-DoF grasps for one object: SAM3 mask -> depth point cloud -> GraspGen -> IK-reachable arm poses.

    Returns up to `top_k` candidates ranked by GraspGen confidence. Each gives the arm, a
    `pregrasp_xyz` (10 cm back along the approach), a `grasp_xyz` and the shared `quat_wxyz`
    for the end-effector link, so they feed `move_pose` directly. Both poses passed an IK
    check. The cloud is single-view, so confirm the approach side in the camera image.

    Args:
        text_prompt: SAM3 prompt naming the object, e.g. 'the red bottle'
        camera: Camera for the mask and depth
        arm: Arm(s) to plan for
        top_k: Number of candidates to return
        min_score: Minimum SAM3 mask score"""
    from robots.robodojo.tools import _refresh_obs

    obs = _refresh_obs(primitives)
    cam = obs.get("vision", {}).get(camera)
    if cam is None or cam.get("color") is None:
        return ToolResult(
            data={"error": f"camera {camera!r} color image not in observation"}
        )
    sam3 = getattr(primitives, "sam3_client", None)
    if sam3 is None:
        return ToolResult(data={"error": "sam3_client not configured"})
    seg = sam3.segment(cam["color"], text_prompt=text_prompt, min_score=min_score)
    if not seg.found:
        return ToolResult(
            data={"found": False, "text_prompt": text_prompt, "camera": camera}
        )
    points = mask_points(cam, seg.mask)
    if len(points) < 64:
        error = f"only {len(points)} valid depth points under the mask"
        return ToolResult(data={"error": error, "camera": camera})
    try:
        poses, scores = graspgen_grasps(points)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        return ToolResult(data={"error": f"GraspGen HTTP {exc.code}: {detail}"})
    except (OSError, RuntimeError) as exc:
        return ToolResult(data={"error": f"GraspGen unavailable: {exc}"})
    lo, hi = points.min(axis=0), points.max(axis=0)
    floor_z = float(lo[2])
    arms = ("left", "right") if arm == "both" else (arm,)
    candidates: list[dict[str, Any]] = []
    rejected = {"from_below": 0, "table_collision": 0, "unreachable": 0}
    ik_checks = 0
    for idx in np.argsort(-np.asarray(scores)):
        if len(candidates) >= int(top_k) or ik_checks >= int(max_ik_checks):
            break
        grasp = np.asarray(poses[idx], dtype=np.float64)
        z_g, x_g = grasp[:3, 2], grasp[:3, 0]
        if z_g[2] > 0.2:  # approach points up: it comes from under the support surface
            rejected["from_below"] += 1
            continue
        targets = grasp_to_ee_targets(grasp)
        # The object's lowest point approximates the support surface; keep the lower
        # finger (half the ~8 cm opening along the closing axis) and the wrist above it.
        lowest_finger = targets[0]["tcp_xyz"][2] - abs(float(x_g[2])) * 0.04 - 0.01
        if (
            lowest_finger < floor_z + 0.005
            or targets[0]["grasp_xyz"][2] < floor_z + 0.03
        ):
            rejected["table_collision"] += 1
            continue
        best = None
        for a in arms:
            for variant in targets:
                ik_checks += 1
                g_ik = primitives.env.solve_ik_pose(
                    a, variant["grasp_xyz"] + variant["quat_wxyz"]
                )
                if g_ik.get("status") != "Success":
                    continue
                ik_checks += 1
                p_ik = primitives.env.solve_ik_pose(
                    a, variant["pregrasp_xyz"] + variant["quat_wxyz"]
                )
                if p_ik.get("status") != "Success":
                    continue
                disp = float(p_ik.get("joint_displacement") or 0.0)
                if best is None or disp < best["pregrasp_joint_displacement"]:
                    best = {
                        "arm": a,
                        **variant,
                        "pregrasp_joint_displacement": round(disp, 3),
                    }
        if best is None:
            rejected["unreachable"] += 1
            continue
        score = round(float(scores[idx]), 4)
        candidates.append(
            {"rank": len(candidates) + 1, "graspgen_score": score, **best}
        )
    return ToolResult(
        data={
            "found": True,
            "text_prompt": text_prompt,
            "camera": camera,
            "mask_score": seg.score,
            "n_points": int(len(points)),
            "object_centroid_xyz": [round(float(v), 4) for v in points.mean(axis=0)],
            "object_bbox_min_xyz": [round(float(v), 4) for v in lo],
            "object_bbox_max_xyz": [round(float(v), 4) for v in hi],
            "n_graspgen_grasps": int(len(poses)),
            "rejected": rejected,
            "ik_checks": ik_checks,
            "candidates": candidates,
        }
    )
