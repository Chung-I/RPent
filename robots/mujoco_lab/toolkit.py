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

"""Toolkit for the MuJoCo lab: the pick_place skill tools plus state capture."""

from __future__ import annotations

import json
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np

from rpent.dashboard.events import DashboardEventSink
from rpent.session import EnvState
from rpent.tools import Toolkit, ToolResult, iter_tools
from rpent.utils.logging import get_output_dir

LAST_PICK_ARTIFACT = "pick_place_last.json"    # the last pick_and_place result, at the run's root for the finaliser

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager


class MuJoCoLabPrimitives:
    """The lab adapter, the success model and a random stream, as the skill tools expect them."""

    def __init__(self, env: Any, model: Any, rng: np.random.Generator | None = None):
        self.env = env
        self.model = model
        self.rng = rng if rng is not None else np.random.default_rng(0)


class MuJoCoLabToolkit(Toolkit):
    def __init__(self, *, memory: MemoryManager, primitives: MuJoCoLabPrimitives, dashboard_events: DashboardEventSink,
                 state: EnvState | None = None) -> None:
        self._primitives = primitives
        self._store: dict[str, Any] = {}
        super().__init__(dashboard_events=dashboard_events, state=state if state is not None else EnvState(get_output_dir()),
                         memory=memory)
        self._register_lab_tools()

    def _register_lab_tools(self) -> None:
        from robots.mujoco_lab import tools as lab_tools
        from rpent.skills.pick_place import tools as skill_tools

        for definition in iter_tools(skill_tools):
            self.add_tool(definition.with_handler(partial(definition, primitives=self._primitives, store=self._store)))
        for definition in iter_tools(lab_tools):
            self.add_tool(definition.with_handler(partial(definition, primitives=self._primitives, state=self._state)))

    def get_env_state(self, *, command: dict[str, Any], result: dict[str, Any], elapsed_s: float) -> ToolResult:
        """Capture the observation after an action; keep the action's own result fields."""
        from robots.mujoco_lab.tools import capture_state

        record, images = capture_state(self._primitives, self._state, command=command, result=result, elapsed_s=elapsed_s)
        data = dict(result)
        data["env_state"] = {"step": record.step_idx, **record.state}
        if command.get("action") == "pick_and_place" and "outcome" in result:
            self._state.save(LAST_PICK_ARTIFACT, {"step": record.step_idx, **result}, step=None)
        return ToolResult(data=data, images=images)

    def solved(self) -> bool:
        """The runner's ground truth: the last pick_and_place set the object down at the requested place."""
        path = self._state.artifact_path(LAST_PICK_ARTIFACT, step=None) if hasattr(self._state, "artifact_path") else None
        try:
            last = json.loads(path.read_text()) if path is not None and path.exists() else None
        except (OSError, ValueError):
            last = None
        return bool(last and last.get("placed_at_requested"))

    def close(self) -> None:
        return None
