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

"""XPolicyLab VLA client — shared ``BaseVLAClient`` RPC surface."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from rpent.robots.components.vla_client_base import BaseVLAClient

#: Cameras and state keys a streamed observation keeps (RoboDojo EE16 policies).
_STREAM_CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
_STREAM_STATE = (
    "left_ee_pose",
    "left_ee_joint_state",
    "right_ee_pose",
    "right_ee_joint_state",
)


def encode_stream_obs(obs: dict[str, Any]) -> dict[str, Any]:
    """Keep the RGB views (lossless PNG), EE state and instruction of one observation."""
    import cv2

    vision = {}
    for camera in _STREAM_CAMERAS:
        rgb = np.ascontiguousarray(
            np.asarray(obs["vision"][camera]["color"], dtype=np.uint8)
        )
        ok, png = cv2.imencode(".png", rgb[..., ::-1])
        if not ok:
            raise ValueError(f"PNG encoding failed for {camera}")
        vision[camera] = {"color_png": png.reshape(-1)}
    state = {key: np.asarray(obs["state"][key]) for key in _STREAM_STATE}
    return {"vision": vision, "state": state, "instruction": obs.get("instruction")}


class XPolicyLabVLAClient(BaseVLAClient):
    """Client for the shared XPolicyLab WebSocket policy adapter.

    Uses the shared :class:`BaseVLAClient` wire protocol (``vla.predict``);
    the shared XPolicyLab adapter translates WebSocket Pi_05 calls into
    that protocol, including the backend's stateful ``reset`` operation.

    With ``RPENT_VLA_STREAM_PNG=1`` (bench), the client streams every executed
    step's observation to history-conditioned policies such as VPP2: primitives
    call :meth:`observe` after each step, and :meth:`predict` uploads the
    buffered frames through ``vla.predict_stream`` before requesting actions.
    """

    def __init__(self, client):
        super().__init__(client)
        self.stream_observations = os.environ.get("RPENT_VLA_STREAM_PNG") == "1"
        self._pending: list[dict[str, Any]] = []

    def healthz(self) -> dict[str, Any]:
        return self._client.call("healthz")

    def reset(self) -> None:
        self._pending.clear()
        self._client.call("reset")

    def observe(self, obs: dict[str, Any]) -> None:
        """Buffer one executed step's observation for the next prediction."""
        self._pending.append(encode_stream_obs(obs))

    def predict(self, obs, options=None):
        if not self.stream_observations:
            return super().predict(obs, options)
        # The latest buffered frame is the current observation; upload the
        # current one only when nothing has executed since the last reset.
        frames = self._pending or [encode_stream_obs(obs)]
        if self._pending:
            frames[-1]["instruction"] = obs.get("instruction")
        self._pending = []
        return self._client.call(
            "vla.predict_stream",
            args=(frames, options),
            timeout_s=self._TIMEOUT_S["predict"],
        )
