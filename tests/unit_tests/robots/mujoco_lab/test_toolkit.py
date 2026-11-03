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

"""The mujoco_lab toolkit through RPent's own dispatch (execute_tool, EnvState), with a fake env."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.robots.robot_spec import RunConfig
from rpent.session import EnvState
from tests.unit_tests.skills.test_tools import FakeEnv, StubModel


def _toolkit(tmp_path, env=None):
    from robots.mujoco_lab.toolkit import MuJoCoLabPrimitives, MuJoCoLabToolkit
    from rpent.utils.logging import init_output_dir

    init_output_dir(tmp_path / "out")          # RPent substitutes the output dir into tool descriptions
    prims = MuJoCoLabPrimitives(env=env or FakeEnv(), model=StubModel(), rng=np.random.default_rng(0))
    return MuJoCoLabToolkit(memory=MemoryManager(tmp_path / "memory"), primitives=prims,
                            dashboard_events=NullDashboardEventSink(), state=EnvState(tmp_path / "state"))


def test_toolkit_lists_tools_with_schemas(tmp_path):
    tk = _toolkit(tmp_path)
    names = {t.name for t in tk.list_tools()}
    assert {"assess_pick", "pick_and_place", "view_env_state"} <= names
    schema = next(t for t in tk.list_tools() if t.name == "pick_and_place").input_schema
    assert set(schema["properties"]) == {"object", "destination", "mode", "option_id", "care"}
    assert schema["required"] == ["object"]
    assert set(next(t for t in tk.list_tools() if t.name == "assess_pick").input_schema["properties"]) == {"object", "destination", "care"}


def test_execute_tool_keeps_result_fields_and_records_step(tmp_path):
    tk = _toolkit(tmp_path)
    a = tk.execute_tool("assess_pick", {"object": "mug", "destination": "table_centre"})
    assert a.data["assessed"] is True and a.data["best_option"] == "commit_std"
    r = tk.execute_tool("pick_and_place", {"object": "mug", "destination": "table_centre"})
    assert r.data["placed"] is True and r.data["outcome"] == "placed_stable" and "declared" in r.data
    assert "env_state" in r.data and r.data["env_state"]["step"] == 0
    rec = tk.state.latest_record()
    assert rec is not None and rec.command["action"] == "pick_and_place" and rec.result["outcome"] == "placed_stable"
    v = tk.execute_tool("view_env_state", {})
    assert v.data["step"] == 1 and "object_pos" in v.data["state"] and v.data["state"]["fixtures_in_scene"]   # -1 captures and records the current state
    assert tk.execute_tool("view_env_state", {"step": 0}).data["step"] == 0


def test_execute_tool_semantic_failure_survives_capture(tmp_path):
    tk = _toolkit(tmp_path)
    r = tk.execute_tool("pick_and_place", {"object": "spoon"})
    assert r.data["assessed"] is False and r.data["reason"] == "object not found"


def test_runner_factory_signature(tmp_path, monkeypatch):
    from robots.mujoco_lab import get_robot_spec, get_toolkit
    from rpent.utils.logging import init_output_dir

    init_output_dir(tmp_path / "out")
    prims = SimpleNamespace(env=FakeEnv(), model=StubModel(), rng=np.random.default_rng(0))
    cfg = RunConfig(recipe_tag="mug_bin_s0", output_dir=tmp_path / "out",
                    prompt_vars={"object": "mug", "destination": "bin", "memory_dir": str(tmp_path / "memory")},
                    task_desc={"object": "mug", "destination": "bin", "seed": 0})
    tk = get_toolkit(runtime_kwargs={"primitives": prims}, dashboard_events=NullDashboardEventSink(), config=cfg)
    assert {"assess_pick", "pick_and_place"} <= {t.name for t in tk.list_tools()}
    spec = get_robot_spec()
    assert spec.name == "mujoco_lab" and spec.finalize_run is not None
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)      # global RPent arguments the robot must not redefine
    parser.add_argument("--memory-dir", default=None)
    spec.add_cli_args(parser, False)
    args = parser.parse_args(["--object", "mug", "--destination", "plate", "--seed", "3", "--output-dir", str(tmp_path / "o")])
    cfg2 = spec.parse_config(args)
    assert cfg2.recipe_tag == "mug_plate_s3" and cfg2.prompt_vars["destination"] == "plate"
    assert spec.prompts.render("system", variables=cfg2.prompt_vars)


def test_system_prompt_names_the_fields():
    from robots.mujoco_lab.prompts.system import RESULT_FIELDS, DESTINATIONS_TEXT

    for k in ("p_success", "p_held_to_release", "p_irreversible", "p_recoverable", "p_drop_height", "options",
              "expected_cost", "recommendation", "decline_reason", "care", "inboard_place", "`guard`"):
        assert k in RESULT_FIELDS, k
    assert "unlike the training objects" in RESULT_FIELDS
    for d in ("table_centre", "table_edge", "table_gap", "mouse_pad", "plate", "box", "bin"):
        assert d in DESTINATIONS_TEXT, d


def test_system_prompt_explains_probes_and_the_belief():
    from robots.mujoco_lab.prompts.system import system_prompt

    text = system_prompt()
    for k in ("probe_push", "probe_weigh", "probe_first", "`probe`", "recommended_option", "evoi", "evoi_se",
              "expected_cost_after", "p_kind", "p_thrown", "`belief`", "`belief_before`", "`diagnosis`",
              "`assessment`", "mass_80", "mu_80", "p_com_shifted"):
        assert k in text, k
    assert "retry" in text.lower()
    assert "once per episode" in text


def test_user_prompt_allows_probes_and_a_retry():
    from robots.mujoco_lab.prompts.user import user_prompt

    text = user_prompt({"object": "ycb__brick", "destination": "plate"})
    assert "ycb__brick" in text and "plate" in text
    assert "probe" in text and "diagnosis" in text and "retry" in text.lower()


def test_user_prompt_accepts_a_decline_as_an_end():
    """Under the cost rule the recommendation is often decline. The task text lets the planner end there, instead of
    asking it to execute a row: in smoke run 2 of plan 2b the planner ran a declined row and spilled the mug (final
    review, plan 2b)."""
    from robots.mujoco_lab.prompts.user import user_prompt

    text = user_prompt({"object": "ycb__mug", "destination": "table_centre"})
    assert "decline" in text and "when you decline" in text
    assert "choose a menu row, execute it" not in text


def test_finalize_run_writes_outcome(tmp_path):
    from robots.mujoco_lab.robot_spec import finalize_run
    from rpent.evaluation.result import RunFinalizationContext

    tk = _toolkit(tmp_path)
    tk.execute_tool("pick_and_place", {"object": "mug", "destination": "table_centre"})
    ctx = RunFinalizationContext(output_dir=tmp_path / "state", robot_name="mujoco_lab",
                                 task_desc={"object": "mug", "destination": "table_centre", "seed": 0},
                                 environment_success=None, agent_error=None, elapsed_s=1.0, planner="x", model=None,
                                 reasoning_effort="low", max_turns=1, planner_timeout_s=None, finish_result=None, stats={})
    path = finalize_run(ctx)
    assert Path(path).exists()
    import json

    out = json.load(open(path))
    assert out["outcome"] == "placed_stable" and out["placed_at_requested"] is True and out["object"] == "mug"


def test_solved_reads_the_last_pick_result(tmp_path):
    tk = _toolkit(tmp_path)
    assert tk.solved() is False
    tk.execute_tool("pick_and_place", {"object": "mug", "destination": "table_centre"})
    assert tk.solved() is True
    tk2 = _toolkit(tmp_path / "second", env=FakeEnv(label="dropped_high"))
    tk2.execute_tool("pick_and_place", {"object": "mug", "destination": "table_centre"})
    assert tk2.solved() is False
    assert tk2.exploration_continuation() is None


def test_system_prompt_states_the_cost_rule_and_damage():
    """Milestone 2b: the cost rule with the abstain cost, numeric care, the eleven outcomes, the squeeze probe and a
    damaged object."""
    from robots.mujoco_lab.prompts.system import system_prompt
    from rpent.skills.pick_place.protocol import DAMAGE, OUTCOMES

    text = system_prompt()
    assert len(OUTCOMES) == 11 and "eleven outcomes" in text
    for k in OUTCOMES:
        assert f"`{k}`" in text or k in text, k
    assert text.count("Outcomes (eleven outcomes):") == 1
    for k in DAMAGE:
        assert k in text.split("Outcomes (eleven outcomes):")[1], k
    held = next(line for line in text.splitlines() if line.startswith("- `p_held_to_release`"))
    assert "damage outcomes are left out" in held, held
    for k in ("abstain cost", "`abstain_cost`", "`care_scale`", "a number above 0", "probe_squeeze", "k_hat",
              "light_grip", "gentle_carry", "`damage`", "`meter_max`", "`damaged`", "object is damaged"):
        assert k in text, k


def test_cli_care_takes_a_number(tmp_path):
    """--care takes low, normal, high or a number above 0 (the care factor kappa); a number stays a number."""
    import pytest

    from robots.mujoco_lab import get_robot_spec

    spec = get_robot_spec()

    def parse(*argv):
        parser = argparse.ArgumentParser()
        parser.add_argument("--output-dir", default=None)
        parser.add_argument("--memory-dir", default=None)
        spec.add_cli_args(parser, False)
        return spec.parse_config(parser.parse_args(["--object", "mug", "--output-dir", str(tmp_path / "o"), *argv]))

    cfg = parse("--care", "10")
    assert cfg.prompt_vars["care"] == 10.0 and cfg.task_desc["care"] == 10.0
    assert parse("--care", "high").prompt_vars["care"] == "high"
    for bad in ("0", "-1", "nan", "very"):
        with pytest.raises(SystemExit):
            parse("--care", bad)


def test_build_lab_passes_the_fill_and_the_fragility(monkeypatch):
    """theta's fill and frag go to the lab's reset (milestone 2b) and the rest stays theta; an m2b bundle's
    thresholds set the lab's slosh model, as in the calibration."""
    from robots.mujoco_lab import adapter, robot_spec
    from robots.mujoco_lab.fragility import DATA, Thresholds

    calls = {}

    class Env:
        def __init__(self, sim_dir=None, render=False):
            self.slosh_zeta, self.slosh_scale = 0.5, None

        def reset(self, object_id, theta=None, seed=0, **kw):
            calls.update(object_id=object_id, theta=theta, seed=seed, **kw)

    manifest = {"prior": "lab_generator_m2b", "classes": {"path": "fragility_classes.json"},
                "thresholds": {"path": "thresholds_v1.json"}}
    bundle = {"model": StubModel(), "prior": None, "readout_model": None, "probe_models": {}, "guard": None,
              "abstain_cost": 1.0, "force_noise": 0.05, "evoi": "heldout", "probe_time_scale": 0.25, "manifest": manifest}
    monkeypatch.setattr(adapter, "MuJoCoLabEnv", Env)
    monkeypatch.setattr(robot_spec, "load_bundle", lambda b, kind=None: bundle)
    frag = {"k_n_per_mm": 2.0, "f_damage_n": 12.0, "v_bruise_ms": None, "v_break_ms": None}
    prims = robot_spec.build_lab(object_id="ycb__mug", theta={"mu": 0.4, "fill": 0.6, "frag": frag}, seed=3, bundle="b")
    assert calls == {"object_id": "ycb__mug", "theta": {"mu": 0.4}, "seed": 3, "fill": 0.6, "frag": frag}
    th = Thresholds.from_json(f"{DATA}/thresholds_v1.json")
    assert prims.env.slosh_zeta == th.slosh_zeta and prims.env.slosh_scale == th.slosh_scale and th.slosh_scale
    assert prims.force_noise == prims.env.force_noise == 0.05                 # the bundle's sensor (plan 2b2)
    assert prims.evoi == "heldout"                                            # the bundle's estimator (plan 2b2)
    assert prims.probe_time_scale == 0.25                                     # the bundle's probe time scale (plan 2b2)
    calls.clear()
    robot_spec.build_lab(object_id="ycb__mug", theta={"mu": 0.4}, seed=3, bundle="b")
    assert calls["fill"] == 0.0 and calls["frag"] is None and calls["theta"] == {"mu": 0.4}


def test_system_prompt_states_the_force_error():
    """Plan 2b2 decision 1: the hand's force readings carry a declared 5 % error, so a slide's mass bound is approximate."""
    from robots.mujoco_lab.prompts.system import system_prompt

    text = system_prompt()
    assert "the measured mass in kg, within about 5 %" in text
    assert "can exceed the mass by a few percent" in text
    assert "started to slide, within about 5 %" in text
    assert "the pads' force readings carry an error of about 5 %" in text


def test_primitives_keep_the_plugin_estimator_by_default():
    from robots.mujoco_lab.toolkit import MuJoCoLabPrimitives

    assert MuJoCoLabPrimitives(env=None, model=None).evoi == "plugin"
    assert MuJoCoLabPrimitives(env=None, model=None, evoi="heldout").evoi == "heldout"


def test_system_prompt_says_the_value_of_information_is_an_estimate():
    """Plan 2b2 decision 2: the held-out estimate can be below 0, and the plug-in's is not exactly 0 for every useless
    probe any more than it is exact for a useful one."""
    from robots.mujoco_lab.prompts.system import system_prompt

    text = system_prompt()
    assert "an estimate that can be below 0" in text and "0 when no result can change the best row" not in text


def test_primitives_keep_probe_time_scale_1_by_default():
    from robots.mujoco_lab.toolkit import MuJoCoLabPrimitives

    assert MuJoCoLabPrimitives(env=None, model=None).probe_time_scale == 1.0
    assert MuJoCoLabPrimitives(env=None, model=None, probe_time_scale=0.25).probe_time_scale == 0.25


def test_system_prompt_names_the_probe_time_scale():
    """The final review of plan 2b2: a probe row's expected_cost charges its seconds at probe_time_scale x 0.05."""
    from robots.mujoco_lab.prompts.system import system_prompt

    text = system_prompt()
    assert "`probe_time_scale`" in text and "`probe_time_scale` times 0.05" in text


def test_system_prompt_says_probes_come_before_the_first_attempt():
    """Plan m4 decision 1: probe rows only where the probe models have data."""
    from robots.mujoco_lab.prompts.system import system_prompt

    text = " ".join(system_prompt().split())
    assert "Probe rows appear only before the first attempt of an episode; after the push only the weigh may follow." in text


def test_system_prompt_states_the_reset_rule():
    """plan H3 Task 1: the cost rule says what `reset_s` changes (the price of an irreversible outcome, the restore, no
    probe after it)."""
    from robots.mujoco_lab.prompts.system import system_prompt

    text = system_prompt()
    assert "`reset_s`" in text and "restored" in text and "no probe runs after a restore" in text
