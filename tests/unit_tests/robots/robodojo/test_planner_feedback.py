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

"""What RoboDojo actions report back to the planner."""

import json
from types import SimpleNamespace

import numpy as np

from robots.robodojo import tools
from robots.robodojo.toolkit import RoboDojoToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory.manager import MemoryManager


def _obs(left_grip=1.0, right_grip=1.0, z=1.0, shade=0):
    state = {
        "left_ee_pose": [0.0, 0.0, z, 1, 0, 0, 0],
        "right_ee_pose": [0.0, 0.0, z, 1, 0, 0, 0],
        "left_ee_joint_state": [left_grip],
        "right_ee_joint_state": [right_grip],
    }
    vision = {
        name: {
            "color": np.full((4, 4, 3), shade, dtype=np.uint8),
            "intrinsic_matrix": np.eye(3),
            "extrinsic_matrix": np.eye(4),
        }
        for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")
    }
    return {"state": state, "vision": vision}


class StepCountingEnv:
    """Grippers move halfway to their command each step; the counter is the episode step."""

    def __init__(self):
        self.step_count = 0
        self.obs = _obs()

    def get_obs(self):
        return self.obs

    def get_status(self):
        return {"step": self.step_count, "step_limit": 700}

    def get_task_language(self):
        return "pick"

    def step(self, action):
        self.step_count += 1
        state = dict(self.obs["state"])
        for arm in ("left", "right"):
            cur = state[f"{arm}_ee_joint_state"][0]
            goal = action[f"{arm}_ee_joint_state"][0]
            new = goal if abs(goal - cur) < 0.004 else cur + 0.5 * (goal - cur)
            state[f"{arm}_ee_joint_state"] = [new]
        self.obs = {"state": state, "vision": self.obs["vision"]}
        return self.obs, 0.0, False, {"status": self.get_status()}


def test_actions_report_their_step_cost_and_grippers_but_not_the_limit(
    monkeypatch, tmp_path
):
    from rpent.utils import logging

    monkeypatch.setattr(logging, "_output_dir", tmp_path)
    env = StepCountingEnv()
    toolkit = RoboDojoToolkit(
        primitives_kwargs={"env": env, "task": "pick"},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )

    data = toolkit.execute_tool("set_gripper", {"arm": "left", "gripper": 1}).data

    assert data["env_steps_used"] == env.step_count > 1  # waited for the fingers
    assert data["grippers"]["left"]["status"] == "closed_empty"
    text = json.dumps(data)
    assert "step_limit" not in text
    assert "intrinsic_matrix" not in text  # calibration is not repeated per action


def test_no_vla_runs_omit_pi0_pick_from_tools_and_prompt(monkeypatch, tmp_path):
    from robots.robodojo.prompt_bundle import system_prompt
    from rpent.prompt.utils import format_prompt
    from rpent.utils import logging

    monkeypatch.setattr(logging, "_output_dir", tmp_path)
    monkeypatch.setenv("RPENT_ROBODOJO_STEERABLE", "1")
    monkeypatch.setenv("RPENT_ROBODOJO_NO_VLA", "1")
    toolkit = RoboDojoToolkit(
        primitives_kwargs={"env": StepCountingEnv(), "task": "pick"},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )
    names = {spec.name for spec in toolkit.list_tools()}
    prompt = format_prompt(system_prompt({}), variables={"output_dir": "out"})

    assert "pi0_pick" not in names and {"plan_grasp", "move_to"} <= names
    assert "pi0" not in prompt and "Pi_05" not in prompt


def test_gripper_opening_is_interpreted():
    report = tools.gripper_report(
        {"left_ee_joint_state": [0.0], "right_ee_joint_state": [0.45]}
    )
    assert report["left"]["status"] == "closed_empty"
    assert report["right"]["status"] == "closed_on_object"
    assert (
        tools.gripper_report({"left_ee_joint_state": [0.97]})["left"]["status"]
        == "open"
    )


def test_pi0_pick_logs_gripper_events_with_keyframes():
    # The policy closes the right gripper, then opens it again (for example a release).
    frames = iter(
        [
            _obs(right_grip=0.9, shade=10),
            _obs(right_grip=0.3, shade=20),
            _obs(right_grip=0.95, shade=30),
        ]
    )
    primitives = SimpleNamespace(
        env=SimpleNamespace(
            get_obs=lambda: _obs(),
            step=lambda action: (
                next(frames),
                0,
                False,
                {"status": {"step": 1, "step_limit": 99}},
            ),
        ),
        vla_client=SimpleNamespace(predict=lambda _: np.zeros((3, 14))),
        _check_cancelled=lambda: None,
        _last_obs=None,
    )

    result = tools.pi0_pick(primitives, None, "pick bottle", arm="right", max_chunks=1)

    events = [e["event"] for e in result.data["events"]]
    assert events == ["right_gripper_closed", "right_gripper_opened"]
    assert [e["keyframe"]["camera"] for e in result.data["events"]] == [
        "cam_right_wrist",
        "cam_head",
    ]
    assert len(result.images) == 2


def test_move_to_stops_and_says_why_when_blocked():
    obs = _obs()
    steps = []

    def step(action):
        steps.append(action)
        return obs, 0, False, {"status": {"step": len(steps), "step_limit": 700}}

    primitives = SimpleNamespace(
        env=SimpleNamespace(
            get_obs=lambda: obs,
            step=step,
            solve_ik_position=lambda arm, xyz: {
                "status": "Success",
                "joint_value": [0.0] * 6,
            },
        ),
        _check_cancelled=lambda: None,
        _last_obs=None,
    )

    result = tools.move_to(primitives, None, [0.3, 0.0, 1.0], arm="right").data

    assert result["stopped_because"] == "stalled"
    assert len(steps) == 3
    assert result["final_quat_wxyz"] == [1, 0, 0, 0]


def test_compaction_settings_follow_the_environment(monkeypatch):
    from rpent.planner.api_loop import _compaction_settings

    assert _compaction_settings() == {
        "max_messages": 80,
        "max_fraction": 0.8,
        "keep_messages": 40,
    }
    monkeypatch.setenv("RPENT_COMPACTION_MAX_MESSAGES", "220")
    monkeypatch.setenv("RPENT_COMPACTION_MAX_FRACTION", "0.85")
    monkeypatch.setenv("RPENT_CONTEXT_WINDOW", "262144")
    monkeypatch.setenv("RPENT_COMPACTION_KEEP_TOKENS", "150000")
    settings = _compaction_settings()
    assert settings["max_messages"] == 220 and settings["context_window"] == 262144
    assert settings["max_fraction"] == 0.85 and settings["keep_tokens"] == 150000


def test_segment_returns_an_outlined_frame_and_a_grounding_box():
    obs = _obs()
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 2:4] = True
    primitives = SimpleNamespace(
        _last_obs=obs,
        sam3_client=SimpleNamespace(
            segment=lambda *a, **kw: SimpleNamespace(
                found=True, score=0.9, box=[2, 1, 3, 2], mask=mask
            )
        ),
    )

    result = tools.segment(primitives, None, "the bottle")

    assert result.data["bbox_2d"] == [500, 250, 750, 500]
    assert "annotation" in result.data["image"]
    assert len(result.images) == 1 and result.images[0][:4] == b"\x89PNG"


def test_file_tools_stay_inside_the_run_folders(monkeypatch, tmp_path):
    from rpent.utils import logging

    out = tmp_path / "run"
    out.mkdir()
    monkeypatch.setattr(logging, "_output_dir", out)
    (out / "notes.txt").write_text("mine")
    secret = tmp_path / "cases" / "case.json"
    secret.parent.mkdir()
    secret.write_text('{"members": []}')
    toolkit = RoboDojoToolkit(
        primitives_kwargs={"env": StepCountingEnv(), "task": "pick"},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )

    assert (
        toolkit.execute_tool("read_text_file", {"path": str(out / "notes.txt")}).data[
            "content"
        ]
        == "mine"
    )
    for name, args in (
        ("read_text_file", {"path": str(secret)}),
        ("list_dir", {"path": str(secret.parent)}),
        ("write_text_file", {"path": str(secret), "content": "x"}),
    ):
        assert (
            "outside this run's folders"
            in toolkit.execute_tool(name, args).data["error"]
        )
    assert secret.read_text() == '{"members": []}'


def test_history_trimming_counts_an_image_as_hundreds_of_tokens(monkeypatch):
    # The compaction estimator stringifies tool results, raw image bytes included;
    # one camera frame then looks like ~290K tokens and every request is trimmed.
    from pydantic_ai import BinaryContent
    from pydantic_ai.messages import ModelRequest, ToolReturnPart
    from pydantic_ai_harness.compaction._shared import estimate_token_count

    from rpent.planner.api_loop import _compaction_settings

    monkeypatch.setenv("RPENT_CONTEXT_WINDOW", "262144")
    image = BinaryContent(
        data=bytes(range(256)) * 1600, media_type="image/png"
    )  # ~400 KB
    message = ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name="view_env_state",
                content=['{"step": 0}', image],
                tool_call_id="x",
            )
        ]
    )

    tokens = estimate_token_count([message], _compaction_settings()["tokenizer"])

    assert 400 <= tokens < 1000
