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


"""What an agent sees under generator v11: the material of the real object (data/materials_v11.json, pinned by its
sha256), one name format for every object, the corrected names of the six renamed objects, and a prior that follows the
v11 classes and masses (the bundle's make_prior; the milestone 1 prior reads the lab's densities, so v11 refuses it)."""

import argparse
import hashlib
import json
import math
import os

import numpy as np
import pytest

MATERIALS_V11_SHA256 = "68efd51603df33f21f572e21b654684caa7b643be281b63aa93e6c37bb5736dd"
RENAMED = {"hot3d__foam_roller": "foam roller", "hot3d__megaphone": "megaphone", "hot3d__clay_plates": "clay plates",
           "hot3d__lizard_figurine": "lizard figurine", "hot3d__storage_box": "storage box", "hot3d__pitcher": "pitcher"}
RECLASSED = ("ycb__mug", "ycb__bowl", "ycb__bowl2", "hot3d__clay_plates", "hot3d__pitcher")
SIM = pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")


def test_the_materials_table_is_pinned(tmp_path):
    from robots.mujoco_lab.masses import load_masses
    from robots.mujoco_lab.names import MATERIALS_V11, MATERIALS_V11_SHA256 as PIN, load_materials

    assert PIN == MATERIALS_V11_SHA256
    assert hashlib.sha256(open(MATERIALS_V11, "rb").read()).hexdigest() == MATERIALS_V11_SHA256
    M = load_materials()
    assert set(M) == set(load_masses()["objects"]) and len(M) == 95
    assert all({"material", "lab", "changed", "basis", "evidence"} <= set(v) for v in M.values())
    assert M["ycb__mug"]["material"] == "enamelled metal" and M["hot3d__clay_plates"]["material"] == "spun bamboo"
    bad = tmp_path / "materials_v11.json"
    bad.write_text(open(MATERIALS_V11).read().replace("spun bamboo", "ceramic"))
    with pytest.raises(ValueError, match="sha256"):
        load_materials(str(bad))


def test_v11_agent_object_shows_the_real_material_and_one_name_format():
    """Under v11 every object shows the table's material and a spaced name; v10 shows the lab's record unchanged."""
    from robots.mujoco_lab.names import agent_object

    mug = {"id": "ycb__mug", "name": "mug", "material": "ceramic"}
    assert agent_object(mug, "v11") == {"name": "mug", "id": "ycb__mug", "material": "enamelled metal"}
    assert agent_object(mug, "v10") == {"name": "mug", "id": "ycb__mug", "material": "ceramic"}
    eraser = {"id": "hot3d__foam_roller", "name": "foam_roller", "material": "foam"}
    assert agent_object(eraser, "v11") == {"name": "whiteboard eraser", "id": "hot3d__whiteboard_eraser",
                                           "material": "plastic, felt pad"}
    box = {"id": "ycb__sugar_box", "name": "sugar_box", "material": "cardboard"}
    assert agent_object(box, "v11") == {"name": "sugar box", "id": "ycb__sugar_box", "material": "cardboard box"}
    assert agent_object(box, "v10") == box


@SIM
def test_v11_agent_text_shows_no_old_name_and_no_lab_material(monkeypatch, tmp_path):
    """Every text an agent reads about the object under v11 (the task text, the audit file name and view_env_state's
    entry), for all 95 objects: a renamed object's old name never appears, the id is the agent id, and the material is
    the table's."""
    from robots.mujoco_lab.adapter import load_sim
    from robots.mujoco_lab.names import agent_object, load_materials
    from robots.mujoco_lab.prompts.user import user_prompt
    from robots.mujoco_lab.robot_spec import _parse_config

    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    M = load_materials()
    objects = {o["id"]: o for o in load_sim()["objects"].OBJECTS}
    for oid, m in M.items():
        args = argparse.Namespace(object=oid, destination="bin", care="normal", seed=0, output_dir=str(tmp_path),
                                  memory_dir=str(tmp_path))
        cfg = _parse_config(args)
        entry = agent_object(objects[oid], "v11")
        text = (user_prompt(cfg.prompt_vars) + " " + cfg.recipe_tag + " " + json.dumps(entry)).lower()
        assert entry["material"] == m["material"], oid
        if oid in RENAMED:
            old = RENAMED[oid]
            assert old not in text.replace("_", " ") and oid not in text, (oid, text)


@SIM
def test_v11_locate_takes_any_separator(monkeypatch):
    """The agent may write a name with spaces, hyphens or underscores under v11; v10 keeps the lab's names."""
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    env = MuJoCoLabEnv()
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    env.reset("hot3d__clay_plates", seed=0)
    assert all(env.locate(p)["object_id"] == "hot3d__clay_plates"
               for p in ("spun bamboo tray", "Spun-Bamboo Tray", "spun_bamboo_tray", "hot3d__spun_bamboo_tray"))
    env.reset("ycb__sugar_box", seed=0)
    assert all(env.locate(p)["object_id"] == "ycb__sugar_box" for p in ("sugar box", "sugar-box", "sugar_box"))
    assert env.locate("toothbrush holder") is None
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v10")
    env.reset("hot3d__clay_plates", seed=0)
    assert env.locate("spun bamboo tray") is None and env.locate("clay plates")["object_id"] == "hot3d__clay_plates"


def test_v11_primitives_need_the_bundles_prior(monkeypatch):
    """Under v11 the skill's prior is the bundle's (make_prior: the v11 classes and masses); without one the skill would
    fall back to the milestone 1 prior, which reads the lab's densities."""
    from robots.mujoco_lab.toolkit import MuJoCoLabPrimitives

    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    with pytest.raises(ValueError, match="prior"):
        MuJoCoLabPrimitives(env=None, model=None)
    assert MuJoCoLabPrimitives(env=None, model=None, prior=lambda pose, n, rng: None).prior is not None
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v10")
    assert MuJoCoLabPrimitives(env=None, model=None).prior is None


def test_the_milestone_1_prior_refuses_v11_rows():
    from rpent.skills.pick_place.evaluate import belief_features

    row = {"generator": "v11", "geometry": {"volume": 1e-4, "material": "ceramic"}, "features": {"lever": 0.0}}
    with pytest.raises(ValueError, match="v11"):
        belief_features(row, np.random.default_rng(0))


@SIM
def test_the_v11_prior_draws_the_v11_classes_and_masses():
    """The prior a v11 agent holds (make_prior under the v11 classes and law): no damage law on the 5 reclassed
    objects, and the empty particles' median mass at the real mass (the scale is log-uniform on [1/sqrt(6), sqrt(6)],
    median 1)."""
    from robots.mujoco_lab.adapter import MuJoCoLabEnv
    from robots.mujoco_lab.fragility import CLASSES_V11, DATA, Thresholds, load_classes
    from robots.mujoco_lab.masses import load_masses
    from robots.mujoco_lab.prior import make_prior

    env = MuJoCoLabEnv()
    prior = make_prior(load_classes(CLASSES_V11), Thresholds.from_json(os.path.join(DATA, "thresholds_v1.json")), generator="v11")
    real = load_masses()["objects"]
    for oid in RECLASSED + ("ycb__sugar_box",):
        os.environ["RPENT_LAB_GENERATOR"] = "v11"
        try:
            env.reset(oid, seed=0)
            b = prior(env.pose_info(), 4000, np.random.default_rng(1))
        finally:
            os.environ.pop("RPENT_LAB_GENERATOR")
        if oid in RECLASSED:
            assert all(p["v_break_ms"] is None and p["f_damage_n"] is None for p in b.particles), oid
        med = float(np.median([p["mass"] for p in b.particles if p["fill"] == 0.0]))
        assert math.isclose(med, real[oid]["mass_kg"], rel_tol=0.06), (oid, med, real[oid]["mass_kg"])


def test_an_assessment_names_the_located_agent_name_not_the_callers_spelling(tmp_path):
    """Review (Codex, agent-facing-v11): an assessment echoed the caller's spelling, and its cache kept the first one, so
    under v11 an agent that wrote the lab id ("hot3d__foam_roller") read the old id back. An env whose locate gives the
    agent's name (the lab env under v11) names the assessment with it; without one (v10) the caller's spelling stays."""
    from tests.unit_tests.robots.mujoco_lab.test_toolkit import _toolkit
    from tests.unit_tests.skills.test_tools import FakeEnv

    class Renamed(FakeEnv):
        def locate(self, prompt):
            loc = super().locate("mug" if prompt in ("hot3d__foam_roller", "whiteboard eraser") else prompt)
            return None if loc is None else dict(loc, name="whiteboard eraser")

    (tmp_path / "v11").mkdir()
    tk = _toolkit(tmp_path / "v11", env=Renamed())
    first = tk.execute_tool("assess_pick", {"object": "hot3d__foam_roller", "destination": "table_centre"})
    again = tk.execute_tool("assess_pick", {"object": "whiteboard eraser", "destination": "table_centre"})
    assert first.data["object"] == again.data["object"] == "whiteboard eraser"
    (tmp_path / "v10").mkdir()
    plain = _toolkit(tmp_path / "v10", env=FakeEnv())
    assert plain.execute_tool("assess_pick", {"object": "object", "destination": "table_centre"}).data["object"] == "object"


@SIM
def test_v11_locate_gives_the_agent_name(monkeypatch):
    """The lab env's locate names the located object as the agent sees it under v11 (any accepted spelling); v10's
    locate record is unchanged (no name)."""
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    env = MuJoCoLabEnv()
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    env.reset("hot3d__foam_roller", seed=0)
    spellings = ("hot3d__foam_roller", "foam roller", "Whiteboard-Eraser", "hot3d__whiteboard_eraser")
    assert {env.locate(p)["name"] for p in spellings} == {"whiteboard eraser"}
    env.reset("ycb__sugar_box", seed=0)
    assert env.locate("sugar_box")["name"] == "sugar box"
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v10")
    env.reset("hot3d__foam_roller", seed=0)
    assert "name" not in env.locate("foam_roller")


def test_v11_refuses_an_output_dir_that_holds_a_renamed_objects_lab_id(tmp_path, monkeypatch):
    """Review (Opus, agent-facing-v11): the dashboard names the task folder from the id the user typed, and RPent puts the
    output dir into list_directory's schema, so a v11 run of a renamed object showed the old id to the agent. Under v11
    _parse_config refuses an output dir that holds a renamed object's lab id; v10 and the other objects keep theirs."""
    from robots.mujoco_lab.robot_spec import _parse_config

    def cfg(obj, out, gen):
        monkeypatch.setenv("RPENT_LAB_GENERATOR", gen)
        return _parse_config(argparse.Namespace(object=obj, destination="bin", care="normal", seed=0,
                                                output_dir=str(tmp_path / out), memory_dir=str(tmp_path)))

    with pytest.raises(ValueError, match="hot3d__clay_plates"):
        cfg("hot3d__clay_plates", "tasks/0001_hot3d__clay_plates_bin_s0", "v11")
    assert cfg("hot3d__clay_plates", "tasks/0001_hot3d__spun_bamboo_tray_bin_s0", "v11").prompt_vars["object"] == "hot3d__spun_bamboo_tray"
    assert cfg("ycb__mug", "tasks/0001_ycb__mug_bin_s0", "v11").output_dir.name == "0001_ycb__mug_bin_s0"
    assert cfg("hot3d__clay_plates", "tasks/0001_hot3d__clay_plates_bin_s0", "v10").output_dir.name == "0001_hot3d__clay_plates_bin_s0"
