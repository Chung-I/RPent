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

"""Lab-side tools and state capture for the MuJoCo lab."""

from __future__ import annotations

from typing import Any

from rpent.tools.base import ToolResult, tool

CAMERA_ARTIFACT = "cam_main.png"


def capture_state(primitives: Any, state: Any, *, command: dict | None = None, result: dict | None = None,
                  elapsed_s: float | None = None) -> tuple[Any, list[bytes]]:
    """Record one step (observation, command, result) and return (record, images)."""
    obs = primitives.env.get_obs()
    images: list[bytes] = []
    with state.record_step(state=dict(obs["state"]), command=command, result=result, elapsed_s=elapsed_s) as step_idx:
        rgb = (obs.get("vision") or {}).get("cam_main")
        if rgb is not None:
            state.save(CAMERA_ARTIFACT, rgb, step=step_idx)
            images.append(state.load_bytes(CAMERA_ARTIFACT, step=step_idx))
    return state.get(step_idx), images


@tool(readonly=True, exclude=("primitives", "state"))
def view_env_state(step: int = -1, *, primitives, state) -> ToolResult:
    """Read the lab state: the object's position and orientation, the hand position, the gripper
    opening, the pad forces, the fixtures in the scene, and the camera image when rendering is on.
    Step -1 reads the current state (and records it); another step reads a past record.

    Args:
        step: Recorded step to read (-1 = current).
    """
    if step == -1:
        record, images = capture_state(primitives, state, command={"action": "view_env_state"})
    else:
        try:
            record = state.get(step)
        except Exception as error:  # noqa: BLE001
            return ToolResult(data={"error": f"state step not available: {error}"})
        images = [state.load_bytes(CAMERA_ARTIFACT, step=record.step_idx)] if CAMERA_ARTIFACT in record.artifacts else []
    obj = getattr(primitives.env, "obj", None)
    data = {"step": record.step_idx, "state": dict(record.state),
            "object": None if obj is None else {"name": obj["name"], "id": obj["id"], "material": obj["material"]}}
    return ToolResult(data=data, images=images)
