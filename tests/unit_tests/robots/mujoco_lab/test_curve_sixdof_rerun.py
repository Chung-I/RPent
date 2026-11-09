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

"""The level-A rerun driver (design 2026-11-08-sixdof-rerun-design.md, unit (f)): the funnel of the 30–45° rows. The
tests target the contacts sim (armsim ee882f9b)."""

import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

from robots.mujoco_lab import curobo_plan as cp
from robots.mujoco_lab import curve_graspgen as gg
from robots.mujoco_lab import curve_sixdof_rerun as rr
from robots.mujoco_lab import sixdof
from robots.mujoco_lab import sixdof_plan as sp
from tests.unit_tests.robots.mujoco_lab.test_curve_graspgen import BASE147, _entry, _rec1, lab  # noqa: F401  lab: the fake case

SIM = pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")
GPU = pytest.mark.skipif(not (os.path.exists(sp.CUROBO_PY) and os.path.exists("/dev/nvidia0")), reason="needs cuRobo's venv and a GPU")


@SIM
def test_the_level_a_funnel_keeps_the_30_to_45_degree_rows_and_says_where_each_stops():
    from tests.unit_tests.robots.mujoco_lab.test_sixdof import _at, _sugar_box

    env, _ = _sugar_box()
    top = env.proposals((0.9,), turn=-math.pi / 2)[0]            # the box's 6 cm side, 1.8 cm below its top: palm clear
    mid = env.proposals((0.5,), turn=-math.pi / 2)[0]            # the 6 cm side at mid-height: the palm inside the box
    G = np.stack([_at(top, 10.0), _at(top, 36.0), _at(mid, 40.0), _at(top, 55.0)])
    rows, counts = rr.funnel6(env, G, np.array([0.9, 0.8, 0.7, 0.6]), "box", {"object": "ycb__sugar_box", "idx": 0})
    assert counts["generated"] == 4 and counts["level_a"] == 2 and [r["id"] for r in rows] == ["g1", "g2"]
    assert all(r["bin"] == "30–45°" and 30.0 < r["approach_deg"] <= 45.0 for r in rows)
    ok, palm = rows
    assert ok["why"] is None and ok["candidate"]["index"] == 0 and len(ok["candidate"]["q"]) == 7
    assert palm["why"] == "clearance: palm/object" and palm["clearance"]["first"]["pair"] == "palm/object" and palm["candidate"] is None
    assert counts["chosen"] == 1


def test_clearance_and_reach6_disagree_only_when_one_passes_and_the_other_fails_on_a_hand_box():
    boxes = {"palm": "palm", "wrist": "wrist", "": "finger"}            # reach6's names of the boxes (the fingers' are unnamed)
    finger = {"index": 1, "reason": "collision", "phase": "close", "step": 60, "hit": ["", "hull2", -0.001]}
    link = {"index": 0, "reason": "collision", "phase": "approach", "step": 30, "hit": ["arm_link7_c", "hull2", -0.002]}
    ik = {"index": 0, "reason": "ik", "phase": "approach", "step": 54, "ik_error": [0.0013, 0.02]}
    assert rr.box_hit([link, finger], boxes) == {"index": 1, "box": "finger", "phase": "close", "other": "hull2"}
    assert rr.box_hit([link, ik], boxes) is None and rr.box_hit([], boxes) is None
    clear = {"pairs": ["palm/object"], "first": {"pair": "palm/object", "phase": "approach", "step": 40}}
    assert rr.disagree(None, None, rr.box_hit([finger], boxes)) == "clearance pass, reach6 box hit"
    assert rr.disagree(clear, {"index": 3}, None) == "clearance fail, reach6 pass"
    assert rr.disagree(None, None, None) is None                        # reach6 fails on an arm link or the IK, or has no goal
    assert rr.disagree(clear, None, rr.box_hit([finger], boxes)) is None   # both fail
    assert rr.disagree(None, {"index": 0}, None) is None                # both pass



def test_a_row_stops_at_its_first_failing_step_and_its_fields_record_both_checks():
    boxes = {"palm": "palm", "wrist": "wrist", "": "finger"}
    clear = {"pairs": ["palm/object"], "first": {"pair": "palm/object", "phase": "approach", "step": 40}, "by_pair": {}}
    ik = {"index": 0, "reason": "ik", "phase": "approach", "step": 54, "ik_error": [0.0013, 0.02]}
    finger = {"index": 1, "reason": "collision", "phase": "close", "step": 60, "hit": ["", "hull2", -0.001]}
    cand = {"index": 1, "twin": 1, "seed": "aimed", "q": np.arange(7.0)}
    # the clearance fails, reach6 passes: the row stops at the clearance and keeps no candidate
    out, stop = rr.assess(clear, {"goal": True, "candidate": cand, "failed": [ik]}, boxes)
    assert stop == "clearance: palm/object" and out["candidate"] is None and out["disagree"] == "clearance fail, reach6 pass"
    assert out["clearance"] == {"first": clear["first"], "pairs": ["palm/object"]}
    assert out["reach"] == {"goal": True, "candidate": 1, "box_hit": None, "failed": [ik]}
    # the clearance passes, reach6 fails on a finger box: the row stops at the path check
    out, stop = rr.assess(None, {"goal": True, "candidate": None, "failed": [ik, finger]}, boxes)
    assert stop == "path: ik (approach)" and out["clearance"] is None and out["candidate"] is None
    assert out["reach"]["box_hit"] == {"index": 1, "box": "finger", "phase": "close", "other": "hull2"}
    assert out["disagree"] == "clearance pass, reach6 box hit"
    # both pass: the candidate goes to the planner
    out, stop = rr.assess(None, {"goal": True, "candidate": cand, "failed": [ik]}, boxes)
    assert stop is None and out["disagree"] is None
    assert out["candidate"] == {"index": 1, "twin": True, "seed": "aimed", "q": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    out, stop = rr.assess(None, {"goal": False, "candidate": None, "failed": []}, boxes)
    assert stop == "path: no candidate" and out["reach"]["goal"] is False

@SIM
def test_the_funnel_runs_reach6_also_after_a_clearance_failure_and_the_row_stops_at_the_clearance():
    from tests.unit_tests.robots.mujoco_lab.test_sixdof import _at, _sugar_box

    env, _ = _sugar_box()
    sim, m = env.sim, env.sim.m
    # reach6 names a geom by its MJCF name: the finger boxes have none, and they are the only robot geoms without one
    assert {g for g in sim.arm.robot_geoms() if not m.geom(g).name} == set(sixdof.hand_geoms(sim)["finger"])
    assert rr.hand_boxes(sim) == {"palm": "palm", "wrist": "wrist", "": "finger"}
    top, low, mid = (env.proposals((s,), turn=-math.pi / 2)[0] for s in (0.9, 0.7, 0.5))
    G = np.stack([_at(top, 36.0), _at(low, 36.0), _at(mid, 40.0)])
    rows, counts = rr.funnel6(env, G, np.array([0.9, 0.8, 0.7]), "box", {"object": "ycb__sugar_box", "idx": 0})
    ok, palm, ik = rows
    assert ok["why"] is None and ok["reach"]["candidate"] == ok["candidate"]["index"] and ok["reach"]["box_hit"] is None
    # the palm inside the box in the approach: the clearance and reach6 both see it, the row stops at the clearance
    assert palm["why"] == "clearance: palm/object" and palm["candidate"] is None and palm["reach"]["candidate"] is None
    assert palm["reach"]["box_hit"] == {"index": 0, "box": "palm", "phase": "approach", "other": "hull0"}
    # the palm inside the box at mid-height: reach6's IK fails in the approach before any collision check
    assert ik["why"] == "clearance: palm/object" and ik["reach"]["goal"] and ik["reach"]["box_hit"] is None
    assert [f["reason"] for f in ik["reach"]["failed"]] == ["ik", "ik"]
    assert [r["disagree"] for r in rows] == [None, None, None] and counts["disagree"] == 0


@SIM
def test_the_vertical_rows_run_the_pilots_funnel_with_the_published_planner_set_and_a_flag_for_the_clearance(monkeypatch):
    from robots.mujoco_lab import curve_graspgen as gg
    from tests.unit_tests.robots.mujoco_lab.test_sixdof import _at, _sugar_box

    monkeypatch.setattr(gg, "KEEP_ROWS", 1)          # the pilot's cap, lowered so that the two distinct rows exceed it
    env, _ = _sugar_box()
    top, mid = (env.proposals((s,), turn=-math.pi / 2)[0] for s in (0.9, 0.5))
    G = np.stack([_at(top, 10.0), _at(mid, 0.0), _at(top, 20.0), _at(top, 36.0)])
    rows, counts = rr.funnel30(env, G, np.array([0.9, 0.8, 0.7, 0.6]), "box")
    assert [r["id"] for r in rows] == ["g0", "g1", "g2"] and all(r["bin"] == "≤ 30°" for r in rows)
    clear, palm, dup = rows
    # the palm 36 mm inside the box: the published planner set (ruling A) passes it, the clearance flags it, and it runs
    assert palm["reach"] == "ok" and palm["clearance"]["first"]["pair"] == "palm/object" and palm["kept_all"] and palm["kept_all_b"]
    assert clear["clearance"] is None and clear["kept_all"] and dup["dup_of"] == "g0" and not dup["kept_all"]
    assert [r["kept"] for r in rows] == [True, False, False] and [r["kept_b"] for r in rows] == [True, False, False]   # the pilot's cap
    assert counts["kept"] == 1 and counts["distinct"] == 2 and counts["clearance_fail"] == 1
    assert env.plan_without_hand_boxes is False      # the caller's planner set again


@SIM
def test_the_tilted_rows_check_the_path_with_the_hand_boxes_whatever_the_callers_planner_set():
    from tests.unit_tests.robots.mujoco_lab.test_sixdof import _at, _sugar_box

    env, _ = _sugar_box()
    env.plan_without_hand_boxes = True
    low = env.proposals((0.7,), turn=-math.pi / 2)[0]
    rows, _ = rr.funnel6(env, np.stack([_at(low, 36.0)]), np.array([0.9]), "box", {"object": "ycb__sugar_box", "idx": 0})
    assert rows[0]["reach"]["box_hit"]["box"] == "palm" and rows[0]["disagree"] is None
    assert env.plan_without_hand_boxes is True


# ------------------------------------------------------------------ step 4: the prepass (the funnels at the reset, the free-move plans)

def test_the_goals_are_the_tilted_rows_with_a_candidate_in_their_order():
    rows = [{"id": "g3", "candidate": {"q": [0.1] * 7}}, {"id": "g1", "candidate": None}, {"id": "g7", "candidate": {"q": [0.2] * 7}}]
    assert rr.goals6(rows) == [{"row": "g3", "q": [0.1] * 7}, {"row": "g7", "q": [0.2] * 7}]
    assert rr.goals6([]) == []


@SIM
def test_a_cases_funnels_run_at_its_reset_and_its_request_holds_the_world_the_start_and_the_chosen_configurations():
    from robots.mujoco_lab.palm_audit import state_sha
    from tests.unit_tests.robots.mujoco_lab.test_sixdof import _at, _sugar_box

    env, _ = _sugar_box()
    world, start, sha = sp.world_export(env.sim), [float(x) for x in env.sim.arm.desired()], state_sha(env)
    top, low = (env.proposals((s,), turn=-math.pi / 2)[0] for s in (0.9, 0.7))
    G, S, case = np.stack([_at(top, 10.0), _at(top, 36.0), _at(low, 36.0)]), np.array([0.9, 0.8, 0.7]), {"object": "ycb__sugar_box", "idx": 0}
    out = rr.case_funnel(env, case, G, S, "box")
    assert [r["id"] for r in out["rows30"]] == ["g0"] and [r["id"] for r in out["rows6"]] == ["g1", "g2"]
    assert out["counts30"]["vertical"] == 1 and out["counts6"]["level_a"] == 2
    assert state_sha(env) == sha                                         # the funnels leave the sim as the reset left it
    req = out["request"]
    assert req["case"] == ["ycb__sugar_box", 0] and req["start"] == start and cp.canonical(req["world"]) == cp.canonical(world)
    assert req["goals"] == [{"row": "g1", "q": out["rows6"][0]["candidate"]["q"]}]
    assert rr.case_funnel(env, case, G[[0, 2]], S[[0, 2]], "box")["request"] is None     # no tilted row passes: nothing to plan


def test_the_prepass_of_a_case_checks_its_reset_like_the_pilot_and_runs_the_funnels_once_there(lab, monkeypatch):
    seen = []
    monkeypatch.setattr(rr, "case_funnel", lambda env, case, G, S, dest: seen.append((case, len(S), dest, list(lab.resets))) or
                        {"rows30": [], "counts30": {}, "rows6": [], "counts6": {}, "request": None})
    rec = rr.prepass_case(SimpleNamespace(env=lab), _rec1(), 3, None, _entry(lab))
    assert rec["stop"] is None and rec["error"] is None and seen == [({"object": "ycb__sugar_box", "idx": 3}, 4, "bin", [3003])]
    assert rec["base"] == BASE147 and rec["destination"] == "bin" and rec["hull_sha"] == gg.hulls_sha(lab) and rec["request"] is None
    assert "cloud" in rr.prepass_case(SimpleNamespace(env=lab), _rec1(), 3, None, _entry(lab, cloud_sha="0" * 64))["stop"]
    assert "fingerprint" in rr.prepass_case(SimpleNamespace(env=lab), _rec1(fingerprint="other"), 3, None, _entry(lab))["stop"]
    assert "no grasps" in rr.prepass_case(SimpleNamespace(env=lab), _rec1(), 3, None, None)["stop"]
    assert "generator" in rr.prepass_case(SimpleNamespace(env=lab), _rec1(), 3, None, dict(_entry(lab), error="boom"))["stop"]
    assert rr.prepass_case(SimpleNamespace(env=lab), _rec1(status="error"), 3, None, _entry(lab))["dropped"]
    assert len(seen) == 1                                                # no stopped case reaches the funnels


def test_the_plan_step_sends_a_cases_request_to_one_planner_process_and_records_its_answer_or_its_failure():
    calls = []
    rec = {"object": "o", "idx": 0, "stop": None, "request": {"case": ["o", 0], "goals": [{"row": "g1", "q": [0.0] * 7}]}}
    out = rr.plan_prepass(rec, "/cache", plan=lambda req, cache: calls.append((req, cache)) or {"goals": 1, "planned": 1, "cached": 0})
    assert calls == [(rec["request"], "/cache")] and out["plan"] == {"goals": 1, "planned": 1, "cached": 0} and out["plan_error"] is None

    def boom(req, cache):
        raise RuntimeError("cuRobo exited 1 on o_0 (log x)")

    assert rr.plan_prepass(rec, "/cache", plan=boom)["plan_error"] == "RuntimeError: cuRobo exited 1 on o_0 (log x)"
    assert rr.plan_prepass(dict(rec, request=None), "/cache", plan=boom) == dict(rec, request=None, plan=None, plan_error=None)
    assert rr.plan_prepass(dict(rec, stop="x"), "/cache", plan=boom) == dict(rec, stop="x", plan=None, plan_error=None)
    assert rr.plan_prepass(dict(rec, dropped=True), "/cache", plan=boom) == dict(rec, dropped=True, plan=None, plan_error=None)


@SIM
@GPU
def test_the_prepass_plans_the_chosen_configuration_and_the_cache_serves_it_after_the_reset(tmp_path):
    from tests.unit_tests.robots.mujoco_lab.test_sixdof import _at, _sugar_box

    env, _ = _sugar_box()
    top = env.proposals((0.9,), turn=-math.pi / 2)[0]
    out = rr.case_funnel(env, {"object": "ycb__sugar_box", "idx": 0}, np.stack([_at(top, 36.0)]), np.array([0.9]), "box")
    rec = rr.plan_prepass({"stop": None, "request": out["request"]}, str(tmp_path))
    assert rec["plan"] == {"goals": 1, "planned": 1, "cached": 0} and rec["plan_error"] is None
    env.reset("ycb__sugar_box", seed=0)                                  # the execute pass: its own reset, the same world and start
    goal = out["rows6"][0]["candidate"]["q"]
    free = sp.free_from_cache(str(tmp_path), sp.world_export(env.sim), [float(x) for x in env.sim.arm.desired()], goal,
                              env.s["scene"].CTRL_HZ)
    assert len(free) > 10 and np.abs(np.asarray(free[-1]) - goal).max() < 1e-3
