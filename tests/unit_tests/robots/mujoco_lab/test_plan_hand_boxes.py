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

"""The planner without the hand boxes (daily-logs designs/2026-11-09-stage2-contacts-rerun-design.md, the coordinator's
ruling A of 2026-11-09): with the executor's flag on, the lab's reach and free-move checks test the published collision
set (the arm links and the pads), so on the contacts sim (daily-logs ee882f9b) the planner answers as on the published
sim (fe46f6c), while the physics keeps the boxes; with the flag off the contacts sim's planner tests the boxes too."""

import os

import pytest

from tests.unit_tests.robots.mujoco_lab.test_replan_hold import SIMS, _sim_sha

SIM = os.environ.get("RPENT_MUJOCO_SIM_DIR", "")
ARM = pytest.mark.skipif(not (SIM and os.path.exists(os.path.join(SIM, "ident.py"))), reason="not the arm sim")
THETA = {"mass_scale": 1.0, "mu": 0.8, "com_s": None, "v_lift": 0.1}
PUBLISHED = ["arm_link0_c", "arm_link1_c", "arm_link2_c", "arm_link3_c", "arm_link4_c", "arm_link5_c0", "arm_link5_c1",
             "arm_link5_c2", "arm_link6_c", "arm_link7_c", "pad_l", "pad_r"]       # fe46f6c's Arm.robot_geoms, by name
BOXES = ["palm", "wrist", "finger_l#box", "finger_r#box"]
CASES = ("hope__mayonnaise_bottle", "hot3d__ceramic_mug", "ycb__tuna_can")
# reach to table_centre of each proposal after the reset (THETA, seed 3, no fixture), compact (see _compact): on fe46f6c
# every proposal is reachable; on ee882f9b the planner that tests the boxes finds the palm inside the tall objects
REACH_FE = {o: [None, None, None] for o in CASES}
REACH_EE_BOXES = {"hope__mayonnaise_bottle": [["collision", "approach", 50, "palm", "hull0"], ["collision", "approach", 65, "palm", "hull0"],
                                              ["collision", "approach", 75, "palm", "hull0"]],
                  "hot3d__ceramic_mug": [["collision", "approach", 80, "palm", "hull5"], ["collision", "approach", 80, "palm", "hull2"],
                                         ["collision", "approach", 80, "palm", "hull2"]],
                  "ycb__tuna_can": [None, None, None]}


def _sim():
    sim = SIMS.get(_sim_sha(SIM)) if SIM else None
    if sim is None:
        pytest.skip("neither fe46f6c nor ee882f9b")
    return sim


def _name(m, g):
    return m.geom(g).name or f"{m.body(int(m.geom_bodyid[g])).name}#box"


def _planner(env):
    m = env.sim.m
    return sorted(_name(m, g) for g in env.sim.arm.robot_geoms())


def _compact(r):
    return None if r is None else [r["reason"], r["phase"], r.get("step"), *(r.get("hit") or [None, None])[:2]]


def _reach(env, o):
    env.reset(o, THETA, seed=3)
    return [_compact(env.reach(p, "table_centre")) for p in env.proposals()]


@pytest.fixture()
def lab(monkeypatch):
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    return MuJoCoLabEnv(damage_meter=False)


@ARM
def test_the_flag_is_off_by_default(lab):
    assert lab.plan_without_hand_boxes is False


@ARM
def test_with_the_flag_on_the_planner_tests_the_published_collision_set(lab):
    _sim()
    lab.plan_without_hand_boxes = True
    lab.reset("ycb__tuna_can", THETA, seed=3)
    assert _planner(lab) == sorted(PUBLISHED)


@ARM
def test_with_the_flag_off_the_planner_tests_what_the_sim_makes_collide(lab):
    lab.reset("ycb__tuna_can", THETA, seed=3)
    assert _planner(lab) == sorted(PUBLISHED + (BOXES if _sim() == "ee882f9b" else []))


@ARM
def test_with_the_flag_on_reach_equals_reach_on_the_published_sim(lab):
    _sim()
    lab.plan_without_hand_boxes = True
    assert {o: _reach(lab, o) for o in CASES} == REACH_FE


@ARM
def test_with_the_flag_off_the_contacts_sims_planner_finds_the_palm_in_tall_objects(lab):
    want = REACH_EE_BOXES if _sim() == "ee882f9b" else REACH_FE
    assert {o: _reach(lab, o) for o in CASES} == want


@ARM
def test_the_flag_applies_to_the_scene_now_and_to_every_later_reset_and_turns_off(lab):
    if _sim() != "ee882f9b":
        pytest.skip("the boxes collide only on ee882f9b")
    o = "hope__mayonnaise_bottle"
    lab.reset(o, THETA, seed=3)
    lab.plan_without_hand_boxes = True                   # set after the reset: the scene that is built now
    assert [_compact(lab.reach(p, "table_centre")) for p in lab.proposals()] == REACH_FE[o]
    assert _reach(lab, o) == REACH_FE[o]                  # and the next reset's scene
    lab.plan_without_hand_boxes = False
    assert [_compact(lab.reach(p, "table_centre")) for p in lab.proposals()] == REACH_EE_BOXES[o]


@ARM
def test_the_flag_leaves_the_physics_alone(lab):
    if _sim() != "ee882f9b":
        pytest.skip("the boxes collide only on ee882f9b")
    lab.plan_without_hand_boxes = True
    lab.reset("ycb__tuna_can", THETA, seed=3)
    m = lab.sim.m
    got = {_name(m, g): (int(m.geom_contype[g]), int(m.geom_conaffinity[g])) for g in range(m.ngeom) if _name(m, g) in BOXES}
    assert got == {b: (2, 1) for b in BOXES}
