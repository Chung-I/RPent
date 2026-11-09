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

import json
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


# ------------------------------------------------------------------ step 5: the execute pass (lab only, cached plans only)

IDENT = {"robot_yml_sha256": "00" * 32, "settings": cp.SETTINGS, "curobo": "test", "torch": "test"}


def _cache_at(path, world, start, goal, **entry):
    """A plan cache at ``path`` with IDENT and, given ``entry``, that entry under the key of (world, start, goal)."""
    import json

    os.makedirs(path, exist_ok=True)
    json.dump(IDENT, open(os.path.join(path, "identity.json"), "w"))
    if entry:
        cp.write_entry(str(path), cp.plan_key(world, start, goal, IDENT), entry)
    return str(path)


def _tilted_row(env):
    """The s 0.9 / 36° sugar-box row through funnel6, its pinch (the twin when its candidate is the twin's) and a reset."""
    from tests.unit_tests.robots.mujoco_lab.test_sixdof import _at

    g = _at(env.proposals((0.9,), turn=-math.pi / 2)[0], 36.0)
    r = rr.funnel6(env, np.stack([g]), np.array([0.9]), "box", {"object": "ycb__sugar_box", "idx": 0})[0][0]
    p = sixdof.from_graspgen(g, width=r["width"])

    def reset():
        env.reset("ycb__sugar_box", seed=0)
        return True
    return r, sixdof.twin(p) if r["candidate"]["twin"] else p, reset


def test_the_straight_joint_line_ends_at_the_goal_in_steps_within_the_free_moves_speed():
    a, b = np.zeros(7), np.array([0.5, -0.1, 0.0, 0.2, 0.0, 0.0, 0.05])
    pts = rr.straight_path(a, b, 50.0, 1.0)
    assert len(pts) == 25 and np.allclose(pts[-1], b) and np.abs(np.diff([a] + pts, axis=0)).max() <= 0.02 + 1e-12
    assert len(rr.straight_path(a, a, 50.0, 1.0)) == 1


@SIM
def test_a_cached_planning_failure_is_the_planners_spheres_only_when_the_straight_line_touches_nothing():
    from tests.unit_tests.robots.mujoco_lab.test_sixdof import _sugar_box

    env, _ = _sugar_box()
    q0 = np.asarray(env.sim.arm.desired(), dtype=float)
    low = q0.copy()
    low[1], low[3] = 1.4, -1.6                                         # the arm reaches down into the table
    assert rr.plan_failure(env.sim, q0, q0 + np.array([0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.3]), 50.0) == \
        {"reason": "planner spheres", "hits": 0, "first": None}
    f = rr.plan_failure(env.sim, q0, low, 50.0)
    assert f["reason"] == "path blocked" and f["hits"] > 0 and f["first"]["robot_id"] in sp.robot_subtree_geoms(env.sim)


@SIM
def test_a_tilted_row_takes_its_free_move_from_the_cache_only_and_a_miss_or_a_failure_stops_it_before_any_motion(tmp_path, monkeypatch):
    from robots.mujoco_lab.curve_poseset import commit_std_base
    from tests.unit_tests.robots.mujoco_lab.test_sixdof import _sugar_box

    env, _ = _sugar_box()
    r, p, reset = _tilted_row(env)
    reset()
    world, start, goal, hz = sp.world_export(env.sim), [float(x) for x in env.sim.arm.desired()], r["candidate"]["q"], env.s["scene"].CTRL_HZ
    base, fp = commit_std_base(env, "ycb__sugar_box", "box"), (lambda e: "fp")
    new, stop = rr.run_row6(env, reset, r, p, _cache_at(tmp_path / "miss", world, start, goal), base, fp, "fp", None)
    assert stop is None and new["free"] == "miss" and new["label"] is None and "box_contacts" not in new     # G3: counted
    failed = _cache_at(tmp_path / "failed", world, start, goal, success=False, status="trajopt failed", dt=1.0 / hz, q=None)
    new, _ = rr.run_row6(env, reset, r, p, failed, base, fp, "fp", None)
    assert new["free"] == "failed" and new["free_failure"]["reason"] in ("planner spheres", "path blocked") and new["label"] is None
    assert "trajopt failed" in new["free_failure"]["status"]
    env.replan_hold, env.plan_without_hand_boxes = True, True
    seen, execute6 = [], sixdof.execute6
    monkeypatch.setattr(sixdof, "execute6", lambda *a, **k: seen.append((env.plan_without_hand_boxes, a[1], a[3:], k)) or execute6(*a, **k))
    plan = [start] + [list(q) for q in sixdof.quintic(env.sim.arm, goal, hz)]   # the lab's own free move, as a cached plan
    ok = _cache_at(tmp_path / "ok", world, start, goal, success=True, status="ok", dt=1.0 / hz, q=plan)
    new, _ = rr.run_row6(env, reset, r, p, ok, base, fp, "fp", None)
    assert new["free"] == "ok" and new["error"] is None and new["unreachable"] is None and new["label"] is not None
    assert [c["ok"] for c in new["sixdof"]["g2"]] == [True, True] and new["box_contacts"]["observed"] and "replan" in new
    assert set(new["free_hits"]) == {"n", "first", "pairs"}
    assert seen == [(False, p, (base["force"], base["v_lift"], base["v_transport"], base["destination"], base["release_height"]),
                     {"a_transport": base["a_transport"]})]           # the hand boxes in the planner's set, the pinch at the base
    assert rr.run_row6(env, reset, r, p, ok, base, fp, "fp", None)[0]["free"] == "ok"   # the key is read after the row's own reset
    assert env.plan_without_hand_boxes is True                         # the caller's planner set again


@SIM
def test_a_tilted_row_whose_free_move_stops_short_fails_g2_and_records_the_failed_check_not_an_outcome(tmp_path):
    from robots.mujoco_lab.curve_poseset import commit_std_base
    from tests.unit_tests.robots.mujoco_lab.test_sixdof import _sugar_box

    env, _ = _sugar_box()
    r, p, reset = _tilted_row(env)
    reset()
    world, start, goal, hz = sp.world_export(env.sim), [float(x) for x in env.sim.arm.desired()], r["candidate"]["q"], env.s["scene"].CTRL_HZ
    stay = _cache_at(tmp_path, world, start, goal, success=True, status="ok", dt=1.0 / hz, q=[start, start])   # the arm stays at rest
    new, stop = rr.run_row6(env, reset, r, p, stay, commit_std_base(env, "ycb__sugar_box", "box"), lambda e: "fp", "fp", None)
    assert stop is None and new["free"] == "ok" and new["error"] is None and new["label"] is None and "sixdof" not in new
    assert new["unreachable"]["check"] == "pregrasp" and new["unreachable"]["ok"] is False and new["unreachable"]["pos_m"] > 0.01


def test_the_free_moves_contact_summary_counts_every_hit_and_names_each_pair_once():
    hits = [{"step": 3, "robot": "palm", "other": "table", "robot_id": 1, "other_id": 2, "dist_m": -0.001},
            {"step": 4, "robot": "palm", "other": "table", "robot_id": 1, "other_id": 2, "dist_m": -0.002},
            {"step": 4, "robot": "link7", "other": "ycb__sugar_box", "robot_id": 5, "other_id": 9, "dist_m": -0.003}]
    assert rr.hits_summary(hits) == {"n": 3, "first": hits[0], "pairs": ["link7/ycb__sugar_box", "palm/table"]}
    assert rr.hits_summary([]) == {"n": 0, "first": None, "pairs": []}


@SIM
def test_a_vertical_row_runs_todays_executor_with_ruling_As_planner_set_and_records_the_box_contacts(monkeypatch):
    from robots.mujoco_lab.curve_poseset import commit_std_base
    from tests.unit_tests.robots.mujoco_lab.test_sixdof import _at, _sugar_box

    env, _ = _sugar_box()
    mid = env.proposals((0.5,), turn=-math.pi / 2)[0]
    r = rr.funnel30(env, np.stack([_at(mid, 0.0)]), np.array([0.9]), "box")[0][0]
    seen, execute = [], env.execute
    monkeypatch.setattr(env, "execute", lambda *a, **k: seen.append(env.plan_without_hand_boxes) or execute(*a, **k))

    def reset():
        env.reset("ycb__sugar_box", seed=0)
        return True
    new, stop = rr.run_row30(env, reset, r, commit_std_base(env, "ycb__sugar_box", "box"), lambda e: "fp", "fp", None)
    assert stop is None and new["error"] is None and new["label"] is not None and seen == [True]
    assert new["box_contacts"]["observed"] and any(k.endswith("palm/object") for k in new["box_contacts"]["by"])   # the palm meets the box
    assert env.plan_without_hand_boxes is False


def _prep(lab):
    """A prepass record of the fake lab's case 3: two kept ≤ 30° rows and one not kept, a planned 30–45° row (on its
    twin) and one without a candidate."""
    return {"object": "ycb__sugar_box", "idx": 3, "fingerprint": "fp", "stop": None, "dropped": False, "base": BASE147,
            "destination": "bin", "hull_sha": gg.hulls_sha(lab), "request": {"goals": [{"row": "g3", "q": [0.0] * 7}]},
            "rows30": [{"id": "g0", "kept_all": True, "kept_all_b": False}, {"id": "g1", "kept_all": False, "kept_all_b": True},
                       {"id": "g2", "kept_all": False, "kept_all_b": False}],
            "rows6": [{"id": "g3", "width": 0.05, "candidate": {"index": 1, "twin": True, "seed": "aimed", "q": [0.0] * 7}},
                      {"id": "g9", "width": 0.05, "candidate": None}]}


def test_the_execute_pass_runs_the_kept_vertical_rows_then_the_planned_tilted_rows_each_with_the_re_plan_on(lab, monkeypatch):
    seen = []
    monkeypatch.setattr(rr, "run_row30", lambda env, reset, r, base, fp, want, hull:
                        seen.append(("30", r["id"], env.replan_hold, base)) or ({"label": "placed_stable"}, None))
    monkeypatch.setattr(rr, "run_row6", lambda env, reset, r, p, cache, base, fp, want, hull:
                        seen.append(("6", r["id"], env.replan_hold, cache, p.R)) or ({"label": "tipped"}, None))
    entry, prep = _entry(lab), _prep(lab)
    out = rr.execute_case(SimpleNamespace(env=lab), _rec1(), 3, None, prep, entry, "/cache")
    twin = sixdof.twin(sixdof.from_graspgen(entry["grasps"][3], width=0.05))
    assert [s[:3] for s in seen] == [("30", "g0", True), ("30", "g1", True), ("6", "g3", True)] and seen[2][3] == "/cache"
    assert seen[0][3] == BASE147 and seen[2][4] == twin.R
    assert [r.get("run") for r in out["rows30"]] == [{"label": "placed_stable"}, {"label": "placed_stable"}, None]
    assert [r.get("run") for r in out["rows6"]] == [{"label": "tipped"}, None] and out["exec_stop"] is None and out["exec_error"] is None
    assert lab.replan_hold is False                                    # the executor's flag is the caller's again
    assert rr.execute_case(SimpleNamespace(env=lab), _rec1(), 3, None, dict(prep, stop="x"), entry, "/cache")["exec_stop"] is None
    assert rr.execute_case(SimpleNamespace(env=lab), _rec1(), 3, None, dict(prep, dropped=True), entry, "/cache")["exec_stop"] is None
    assert "fingerprint" in rr.execute_case(SimpleNamespace(env=lab), _rec1(fingerprint="other"), 3, None, prep, entry, "/cache")["exec_stop"]
    assert "base" in rr.execute_case(SimpleNamespace(env=lab), _rec1(), 3, None, dict(prep, base=dict(BASE147, force=1.0)), entry,
                                     "/cache")["exec_stop"]
    assert len(seen) == 3 and prep == _prep(lab)                       # a stopped case runs no row, the runs go into copies


def test_a_row_that_stops_stops_its_case_and_an_exception_is_the_cases_error_and_the_prepass_record_stays_as_it_was(lab, monkeypatch):
    seen, prep, entry = [], _prep(lab), _entry(lab)
    monkeypatch.setattr(rr, "run_row30", lambda env, reset, r, *a:
                        (None, "the fingerprint differs") if r["id"] == "g1" else ({"label": "placed_stable"}, None))
    monkeypatch.setattr(rr, "run_row6", lambda *a: seen.append("6") or ({"label": "tipped"}, None))
    out = rr.execute_case(SimpleNamespace(env=lab), _rec1(), 3, None, prep, entry, "/cache")
    assert out["exec_stop"] == "g1: the fingerprint differs" and seen == [] and lab.replan_hold is False
    monkeypatch.setattr(rr, "run_row30", lambda *a: ({"label": "placed_stable"}, None))
    monkeypatch.setattr(rr, "run_row6", lambda *a: (None, "the hulls differ"))
    assert rr.execute_case(SimpleNamespace(env=lab), _rec1(), 3, None, prep, entry, "/cache")["exec_stop"] == "g3: the hulls differ"
    monkeypatch.setattr(rr, "run_row6", lambda *a: (_ for _ in ()).throw(RuntimeError("the sim broke")))
    out = rr.execute_case(SimpleNamespace(env=lab), _rec1(), 3, None, prep, entry, "/cache")
    assert out["exec_error"] == "RuntimeError: the sim broke" and out["exec_stop"] is None and lab.replan_hold is False
    assert prep == _prep(lab)                                          # the runs go into copies of the rows


# ------------------------------------------------------------------ step 6: G3's replay, the unit's counts, the stop rule, the smoke report

def _stub_rows(monkeypatch, differ=()):
    """run_row30 / run_row6 stubs: each resets (the fake lab records the seed) and returns fixed fields; a row in
    ``differ`` returns another label at its second run. Returns the row ids in call order."""
    seen = []

    def fields(r, label):
        seen.append(r["id"])
        return {"label": "dropped_high" if r["id"] in differ and seen.count(r["id"]) == 2 else label, "error": None, "seconds": 1.0}

    monkeypatch.setattr(rr, "run_row30", lambda env, reset, r, *a: (reset(), (fields(r, "placed_stable"), None))[1])
    monkeypatch.setattr(rr, "run_row6", lambda env, reset, r, *a: (reset(), (dict(fields(r, "tipped"), free="ok", unreachable=None), None))[1])
    return seen


def test_the_replay_runs_every_row_that_ran_again_at_the_second_seed_in_reverse_and_lists_what_changed(lab, monkeypatch):
    seen = _stub_rows(monkeypatch)
    out = rr.execute_case(SimpleNamespace(env=lab), _rec1(), 3, None, _prep(lab), _entry(lab), "/cache", replay=True)
    assert seen == ["g0", "g1", "g3", "g3", "g1", "g0"] and lab.resets == [3003] * 4 + [777] * 3
    assert out["replay"] == {"rows": 3, "mismatches": [], "stop": None} and out["exec_stop"] is None and out["replay_seconds"] >= 0
    lab.resets.clear()
    _stub_rows(monkeypatch, differ=("g1",))
    out = rr.execute_case(SimpleNamespace(env=lab), _rec1(), 3, None, _prep(lab), _entry(lab), "/cache", replay=True)
    assert [m["id"] for m in out["replay"]["mismatches"]] == ["g1"] and out["exec_stop"] == "replay: 1 rows differ at the second seed (G3)"
    assert out["rows30"][1]["run"]["label"] == "placed_stable" and lab.replan_hold is False   # the first pass's fields stay
    calls = []
    monkeypatch.setattr(rr, "run_row6", lambda env, reset, r, *a:
                        (calls.append(r["id"]), ({"label": "tipped", "free": "ok"}, None) if len(calls) == 1 else (None, "the hulls differ"))[1])
    out = rr.execute_case(SimpleNamespace(env=lab), _rec1(), 3, None, _prep(lab), _entry(lab), "/cache", replay=True)
    assert out["exec_stop"] == "replay: g3: the hulls differ" and out["replay"]["stop"] == "g3: the hulls differ"
    assert rr.execute_case(SimpleNamespace(env=lab), _rec1(), 3, None, _prep(lab), _entry(lab), "/cache")["replay"] is None


def _done_records():
    """Three executed records of real shape: one with every kind of row (its replay included), one stopped in the
    prepass, one dropped with an error."""
    c30 = dict({k: 2 for k in gg.FUNNEL + gg.FUNNEL_B}, clearance_fail=1)
    ok = {"object": "ycb__bowl", "idx": 0, "skipped": None, "dropped": False, "stop": None, "error": None, "exec_stop": None,
          "exec_error": None, "seconds": 30.0, "exec_seconds": 50.0, "replay_seconds": 20.0, "counts30": c30,
          "counts6": {"generated": 10, "level_a": 5, "chosen": 4, "disagree": 1},
          "rows30": [{"id": "g0", "kept_all": True, "kept_all_b": False, "clearance": {"first": "palm/object", "pairs": ["palm/object"]},
                      "run": {"label": "placed_stable", "error": None}},
                     {"id": "g1", "kept_all": False, "kept_all_b": True, "clearance": None, "run": {"label": "tipped", "error": "RuntimeError: x"}},
                     {"id": "g2", "kept_all": False, "kept_all_b": False, "clearance": None}],
          "rows6": [{"id": "g3", "disagree": None, "candidate": {"index": 0}, "run": {"free": "ok", "unreachable": None, "error": None, "label": "tipped"}},
                    {"id": "g4", "disagree": None, "candidate": {"index": 0},
                     "run": {"free": "ok", "unreachable": {"check": "pregrasp", "ok": False}, "error": None, "label": None}},
                    {"id": "g5", "disagree": None, "candidate": {"index": 1},
                     "run": {"free": "failed", "free_failure": {"reason": "planner spheres"}, "error": None, "label": None}},
                    {"id": "g6", "disagree": None, "candidate": {"index": 0}, "run": {"free": "miss", "error": None, "label": None}},
                    {"id": "g7", "disagree": "clearance fail, reach6 pass", "candidate": None}],
          "replay": {"rows": 5, "mismatches": [{"id": "g3"}], "stop": None}}
    stop = dict(ok, idx=1, stop="the reset's fingerprint differs from Stage 1's", counts30=None, counts6=None, rows30=[], rows6=[],
                replay=None, exec_seconds=0.0, replay_seconds=None)
    return [ok, stop, dict(stop, idx=2, stop=None, dropped=True, error="ValueError: y")]


def test_the_units_counts_hold_the_funnels_the_free_moves_g2_and_g3_and_no_label():
    m = rr.exec_metrics(_done_records())
    assert m == {"cases": 3, "skipped": 0, "dropped": 1, "stops": 1, "errors": 1,
                 "funnel30": dict({k: 2 for k in gg.FUNNEL + gg.FUNNEL_B}, clearance_fail=1),
                 "funnel6": {"generated": 10, "level_a": 5, "chosen": 4, "disagree": 1},
                 "rows30_run": 2, "rows30_flagged_run": 1, "rows30_errors": 1,
                 "path_passed": 4, "free": {"ok": 2, "miss": 1, "failed": 1}, "failed_by_reason": {"planner spheres": 1},
                 "g2_fail": 1, "g2_by_check": {"pregrasp": 1}, "rows6_errors": 0,
                 "replay_rows": 5, "replay_mismatches": 1, "case_seconds": 140.0, "case_seconds_max": 80.0}
    assert "placed_stable" not in json.dumps(m) and "tipped" not in json.dumps(m)
    late = rr.exec_metrics([dict(_done_records()[0], exec_stop="g3: x", exec_error="RuntimeError: z",
                                 rows30=[{"id": "g9", "kept_all": False, "kept_all_b": False, "clearance": {"first": "palm/object"}}])])
    assert (late["stops"], late["errors"], late["rows30_run"], late["rows30_flagged_run"]) == (1, 1, 0, 0)   # an execute-pass stop counts


def test_the_stop_rule_fires_above_ten_percent_of_tracking_or_plan_failures_and_at_any_miss_mismatch_stop_or_error():
    base = {"stops": 0, "errors": 0, "path_passed": 20, "free": {"ok": 10, "miss": 0, "failed": 2}, "g2_fail": 1, "replay_mismatches": 0}
    assert rr.stop_rule(base) == []                                    # 1 of 10 and 2 of 20 are 10 %, not more
    assert rr.stop_rule(dict(base, g2_fail=2)) == ["tracking (G2) fails on 2 of 10 planned rows (more than 10 %)"]
    assert rr.stop_rule(dict(base, free={"ok": 10, "miss": 0, "failed": 3})) == \
        ["cuRobo fails on 3 of 20 rows that passed the lab's path check (more than 10 %)"]
    assert rr.stop_rule(dict(base, free={"ok": 10, "miss": 1, "failed": 2}, replay_mismatches=2, stops=1, errors=1)) == \
        ["G3: 1 plan-cache misses (0 required)", "G3: 2 rows differ at the second seed (0 required)", "1 case stops and 1 case errors"]
    assert rr.stop_rule(dict(base, path_passed=0, free={"ok": 0, "miss": 0, "failed": 0}, g2_fail=0)) == []
    assert rr.stop_rule(dict(base, errors=1)) == ["0 case stops and 1 case errors"]


def test_the_smoke_report_projects_the_full_run_from_each_objects_mean_times_and_holds_no_label():
    plans = [{"object": "ycb__bowl", "idx": 0, "seconds": 40.0, "plan": {"goals": 4, "planned": 4, "cached": 0}, "plan_error": None},
             {"object": "ycb__bowl", "idx": 1, "seconds": 0.0, "plan": None, "plan_error": None},
             {"object": "ycb__bowl", "idx": 2, "seconds": 0.5, "plan": None, "plan_error": None}]
    sm = rr.smoke_report(_done_records(), plans, {"ycb__bowl": 11, "ycb__pitcher": 11})
    assert sm["projection_cpu_h"] == pytest.approx(((30 + 30) + 30 + 30) / 3 * 22 / 3600)   # first pass only; no smoke case: the largest mean
    assert sm["projection_gpu_h"] == pytest.approx((40.0 + 0.0 + 0.5) / 3 * 22 / 3600)
    assert sm["counts"] == rr.exec_metrics(_done_records()) and sm["stop"] == rr.stop_rule(sm["counts"]) and sm["stop"]
    assert sm["disagree"] == {"ycb__bowl": 1} and sm["funnel6"]["ycb__bowl"]["level_a"] == 5 and sm["funnel30"]["all"]["kept"] == 2
    assert sm["plans"] == {"cases": 3, "with_plan": 1, "goals": 4, "planned": 4, "cached": 0, "errors": 0}
    bad = rr.smoke_report(_done_records(), [dict(plans[0], plan=None, plan_error="RuntimeError: cuRobo exited 1")], {"ycb__bowl": 11})
    assert bad["plans"]["errors"] == 1 and bad["plans"]["with_plan"] == 0
    text = json.dumps(sm) + rr.smoke_markdown(sm)
    assert "placed_stable" not in text and "tipped" not in text and '"label"' not in text and "stop rule" in text


# ------------------------------------------------------------------ step 6: the commands (the heavy parts stubbed)

@pytest.fixture
def quiet(monkeypatch):
    """The commands' outside calls stubbed: the code revision, the bundle's generator and wandb."""
    from robots.mujoco_lab import estimation_check, identify_fit, tracking

    monkeypatch.setattr(estimation_check, "_revision", lambda: {"code_revision": "r"})
    monkeypatch.setattr(identify_fit, "require_bundle_generator", lambda b: "v11")
    monkeypatch.setattr(tracking, "log_run", lambda name, config, metrics: f"wandb:{name}")


def _stage1(tmp_path, monkeypatch, objects=("ycb__bowl", "ycb__pitcher")):
    """A Stage 1 dir and its records (3 cases per object, fold 0), load_stage1 stubbed to return them."""
    from robots.mujoco_lab import curve

    d = tmp_path / "stage1"
    d.mkdir()
    (d / "records.jsonl").write_text("{}\n")
    recs1 = [_rec1(object=o, idx=i) for o in objects for i in (0, 1, 2)]
    monkeypatch.setattr(curve, "load_stage1", lambda dirs: (recs1, {"bundle": "/b", "seed": 3, "code_revision": "c"}))
    return str(d), recs1


def _finished_dir(path, name, lines, config=None):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, name), "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in lines)
    json.dump(config or {}, open(os.path.join(path, "run_config.json"), "w"))
    json.dump({}, open(os.path.join(path, "run_metrics.json"), "w"))
    return str(path)


def test_the_prepass_command_runs_the_chosen_cases_on_the_pilots_grasps_and_refuses_other_grasps_or_an_existing_out(tmp_path, monkeypatch, quiet):
    from robots.mujoco_lab.curve import _sha

    stage, _ = _stage1(tmp_path, monkeypatch)
    g = tmp_path / "grasps"
    g.mkdir()
    (g / "index.jsonl").write_text("{}\n")
    pilot = {"grasps_index_sha256": _sha(str(g / "index.jsonl")), "bundle": "/b", "seed": 3,
             "stage1_records_sha256": {stage: _sha(os.path.join(stage, "records.jsonl"))}}
    _finished_dir(tmp_path / "pilot", "records_test.jsonl", [], pilot)
    jobs = []
    monkeypatch.setattr(rr, "prepass_object", lambda job: jobs.append(job) or
                        {"records": [dict(r, seconds=1.0, stop=None, request={"goals": [1, 2]} if r["idx"] == 0 else None) for r in job[1]],
                         "stop": None})
    args = ["prepass", "--stage1", stage, "--grasps", str(g), "--pilot", str(tmp_path / "pilot"), "--objects", "ycb__bowl,ycb__pitcher",
            "--sides", "test", "--per-object", "2", "--out", str(tmp_path / "out")]
    rr.main(args)
    recs = [json.loads(line) for line in open(tmp_path / "out" / "prepass_records.jsonl")]
    assert [(r["object"], r["idx"]) for r in recs] == [("ycb__bowl", 0), ("ycb__bowl", 1), ("ycb__pitcher", 0), ("ycb__pitcher", 1)]
    assert [j[0] for j in jobs] == ["ycb__bowl", "ycb__pitcher"] and jobs[0][2] == {"bundle": "/b/folds/fold_0", "seed": 3, "grasps": str(g)}
    m = json.load(open(tmp_path / "out" / "run_metrics.json"))
    assert m["cases"] == 4 and m["requests"] == 2 and m["goals"] == 4 and m["stop"] is None and m["wandb"] == "wandb:sixdof-rerun-prepass"
    assert json.load(open(tmp_path / "out" / "run_config.json"))["cases"] == [["ycb__bowl", 0], ["ycb__bowl", 1], ["ycb__pitcher", 0], ["ycb__pitcher", 1]]
    with pytest.raises(SystemExit, match="exists"):
        rr.main(args)
    json.dump(dict(pilot, grasps_index_sha256="0" * 64), open(tmp_path / "pilot" / "run_config.json", "w"))
    with pytest.raises(SystemExit, match="grasps_index_sha256"):
        rr.main(args[:-1] + [str(tmp_path / "out2")])
    json.dump(pilot, open(tmp_path / "pilot" / "run_config.json", "w"))
    monkeypatch.setattr(rr, "prepass_object", lambda job: {"records": [dict(job[1][0], stop="x")], "stop": "ycb__bowl 0: x"})
    with pytest.raises(SystemExit) as e:
        rr.main(args[:-1] + [str(tmp_path / "out3")])
    assert e.value.code == 3 and json.load(open(tmp_path / "out3" / "run_metrics.json"))["stop"] == "ycb__bowl 0: x"


def test_the_plan_command_plans_each_request_in_turn_and_exits_3_when_a_planner_process_fails(tmp_path, monkeypatch, quiet):
    recs = [{"object": "ycb__bowl", "idx": 0, "stop": None, "dropped": False, "request": {"case": ["ycb__bowl", 0]}},
            {"object": "ycb__bowl", "idx": 1, "stop": None, "dropped": False, "request": None},
            {"object": "ycb__bowl", "idx": 2, "stop": "x", "dropped": False, "request": {"case": ["ycb__bowl", 2]}}]
    pre = _finished_dir(tmp_path / "pre", "prepass_records.jsonl", recs)
    calls = []
    monkeypatch.setattr(sp, "plan_case", lambda req, cache: calls.append((req["case"], cache)) or {"goals": 2, "planned": 2, "cached": 0})
    rr.main(["plan", "--prepass", pre, "--cache", str(tmp_path / "cache"), "--out", str(tmp_path / "out")])
    lines = [json.loads(line) for line in open(tmp_path / "out" / "plan_records.jsonl")]
    assert calls == [(["ycb__bowl", 0], str(tmp_path / "cache"))] and [x["idx"] for x in lines] == [0, 1, 2]
    assert lines[0]["plan"] == {"goals": 2, "planned": 2, "cached": 0} and lines[1]["plan"] is None and set(lines[0]) == {
        "object", "idx", "plan", "plan_error", "seconds"}
    m = json.load(open(tmp_path / "out" / "run_metrics.json"))
    assert (m["cases"], m["with_plan"], m["goals"], m["planned"], m["errors"]) == (3, 1, 2, 2, 0) and m["wandb"] == "wandb:sixdof-rerun-plan"
    monkeypatch.setattr(sp, "plan_case", lambda req, cache: (_ for _ in ()).throw(RuntimeError("cuRobo exited 1")))
    with pytest.raises(SystemExit) as e:
        rr.main(["plan", "--prepass", pre, "--cache", str(tmp_path / "cache"), "--out", str(tmp_path / "out2")])
    assert e.value.code == 3
    os.remove(os.path.join(pre, "run_metrics.json"))
    with pytest.raises(SystemExit, match="not finished"):
        rr.main(["plan", "--prepass", pre, "--cache", str(tmp_path / "cache"), "--out", str(tmp_path / "out3")])


def test_the_execute_command_runs_each_prepass_record_with_its_stage1_case_after_a_finished_plan_step(tmp_path, monkeypatch, quiet):
    from robots.mujoco_lab.curve import _sha

    stage, recs1 = _stage1(tmp_path, monkeypatch)
    preps = [{"object": o, "idx": i, "stop": None, "dropped": False, "request": {"goals": []} if i == 0 else None, "seconds": 1.0}
             for o in ("ycb__bowl", "ycb__pitcher") for i in (0, 1)]
    cfg = {"stage1_records_sha256": {stage: _sha(os.path.join(stage, "records.jsonl"))}, "bundle": "/b", "seed": 3, "grasps": "/g"}
    pre = _finished_dir(tmp_path / "pre", "prepass_records.jsonl", preps, cfg)
    plan = _finished_dir(tmp_path / "plan", "plan_records.jsonl", [{"object": p["object"], "idx": p["idx"], "plan": None, "plan_error": None,
                                                                    "seconds": 0.0} for p in preps])
    jobs = []
    monkeypatch.setattr(rr, "execute_object", lambda job: jobs.append(job) or
                        {"records": [dict(prep, exec_stop=None, exec_error=None, replay=None, exec_seconds=2.0) for _, prep in job[1]], "stop": None})
    args = ["execute", "--prepass", pre, "--plan", plan, "--cache", str(tmp_path / "cache"), "--replay", "--out", str(tmp_path / "out")]
    rr.main(args)
    assert [j[0] for j in jobs] == ["ycb__bowl", "ycb__pitcher"] and [(r1["idx"], p["idx"]) for r1, p in jobs[0][1]] == [(0, 0), (1, 1)]
    assert jobs[0][1][0][0] is recs1[0] and jobs[1][2] == {"bundle": "/b/folds/fold_0", "seed": 3, "grasps": "/g",
                                                           "cache": str(tmp_path / "cache"), "replay": True}
    assert len(open(tmp_path / "out" / "exec_records.jsonl").readlines()) == 4
    m = json.load(open(tmp_path / "out" / "run_metrics.json"))
    assert m["cases"] == 4 and m["stop"] is None and m["wandb"] == "wandb:sixdof-rerun-execute"
    _finished_dir(tmp_path / "plan2", "plan_records.jsonl", [])
    with pytest.raises(SystemExit, match="no plan record"):
        rr.main(args[:4] + [str(tmp_path / "plan2")] + args[5:-1] + [str(tmp_path / "out2")])
    os.remove(os.path.join(plan, "run_metrics.json"))
    with pytest.raises(SystemExit, match="not finished"):
        rr.main(args[:-1] + [str(tmp_path / "out3")])


FE46 = "e823247052a2bf4d1b97d5077a51f0ead91fd80c"   # the published sim's last commit (the fe46f6c worktree): hand boxes contype 0


def _provenance(d, code=None, sim=None, code_dirty=0, sim_dirty=0, sim_dir="/w/armsim-contacts-ee882f9b/sim"):
    """A run dir's provenance.txt as the launcher (run_curve.sh) writes it: the code line, then the sim line."""
    with open(os.path.join(d, "provenance.txt"), "w") as f:
        f.write(f"code {code or rr.LEVEL_A_CODE} (detached run worktree), dirty files {code_dirty}\n"
                f"sim {sim_dir}, last commit {sim or rr.LEVEL_A_SIM}, dirty files {sim_dirty}\n"
                "command python -P -u -m robots.mujoco_lab.curve_sixdof_rerun execute\n")
    return d


def _gate_dirs(tmp_path, records=(), plans=()):
    """One prepass unit, two execute units that read it, and a plan step whose sim line names the published sim."""
    pre = _provenance(_finished_dir(tmp_path / "pre", "prepass_records.jsonl", []))
    ex = [_provenance(_finished_dir(tmp_path / f"ex{k}", "exec_records.jsonl", list(records) if k == 0 else [], {"prepass": pre}))
          for k in (0, 1)]
    plan = _provenance(_finished_dir(tmp_path / "plan", "plan_records.jsonl", list(plans)), sim=FE46, sim_dir="/w/v12checks-armsim-fe46f6c/sim")
    return {"pre": pre, "ex0": ex[0], "ex1": ex[1], "plan": plan}


def test_the_provenance_gate_passes_units_on_the_level_a_code_and_sim_and_records_the_plans_sim_without_checking_it(tmp_path):
    d = _gate_dirs(tmp_path)
    g = rr.provenance_gate([d["ex0"], d["ex1"]], d["plan"])
    assert g == {"code": rr.LEVEL_A_CODE, "sim": rr.LEVEL_A_SIM, "execute_units": 2, "prepass_units": 1,
                 "plan": {"code": rr.LEVEL_A_CODE, "sim": FE46, "sim_dir": "/w/v12checks-armsim-fe46f6c/sim"}}
    assert rr.LEVEL_A_SIM == "ee882f9b5574a4adc30559fd92c9df3bf69afdf3" and rr.LEVEL_A_CODE == "f63ff35ce11cca646650c99c14e5eb9c95a1fa77"


@pytest.mark.parametrize("where, change", [("ex1", {"sim": FE46}), ("pre", {"sim": FE46}), ("pre", {"code": "c2cfaac" + "0" * 33}),
                                           ("ex0", {"sim_dirty": 1}), ("ex1", {"code_dirty": 2}), ("plan", {"code": "c2cfaac" + "0" * 33}),
                                           ("plan", {"code_dirty": 1}), ("ex0", None), ("pre", None)])
def test_the_provenance_gate_refuses_a_unit_on_another_sim_or_code_with_dirty_files_or_without_provenance(tmp_path, where, change):
    d = _gate_dirs(tmp_path)
    if change is None:
        os.remove(os.path.join(d[where], "provenance.txt"))
    elif where == "plan":
        _provenance(d["plan"], **dict({"sim": FE46, "sim_dir": "/w/v12checks-armsim-fe46f6c/sim"}, **change))
    else:
        _provenance(d[where], **change)
    with pytest.raises(SystemExit) as e:
        rr.provenance_gate([d["ex0"], d["ex1"]], d["plan"])
    msg = str(e.value)
    assert msg.startswith("provenance gate") and f"{d[where]}:" in msg
    assert [k for k in d if f"{d[k]}:" in msg] == [where]


def test_the_provenance_gate_refuses_an_execute_unit_that_names_no_prepass_run(tmp_path):
    d = _gate_dirs(tmp_path)
    json.dump({}, open(os.path.join(d["ex1"], "run_config.json"), "w"))
    with pytest.raises(SystemExit, match="no prepass"):
        rr.provenance_gate([d["ex0"], d["ex1"]], d["plan"])


def test_the_smoke_command_writes_the_report_and_exits_3_when_the_stop_rule_fires(tmp_path, monkeypatch, quiet):
    d = _gate_dirs(tmp_path, records=_done_records(), plans=[{"object": "ycb__bowl", "idx": 0, "plan": None, "plan_error": None, "seconds": 1.0}])
    ex, plan = f"{d['ex0']},{d['ex1']}", d["plan"]
    full = tmp_path / "cases.txt"
    full.write_text(",".join(f"ycb__bowl:{i}" for i in range(11)) + "\n")
    with pytest.raises(SystemExit) as e:
        rr.main(["smoke-report", "--execute", ex, "--plan", plan, "--full-cases", str(full), "--out", str(tmp_path / "out")])
    sm = json.load(open(tmp_path / "out" / "smoke_summary.json"))
    assert e.value.code == 3 and sm["stop"] and sm["full_cases"] == {"ycb__bowl": 11} and sm["wandb"] == "wandb:sixdof-rerun-smoke"
    md = (tmp_path / "out" / "smoke_summary.md").read_text()
    assert md == rr.smoke_markdown(sm) and sm["provenance"] == rr.provenance_gate(ex.split(","), plan)
    assert f"code {rr.LEVEL_A_CODE}" in md and f"sim {rr.LEVEL_A_SIM}" in md and FE46 in md


def test_the_smoke_command_refuses_before_it_writes_or_logs_anything_when_a_unit_ran_on_another_sim(tmp_path, monkeypatch, quiet):
    from robots.mujoco_lab import tracking

    d = _gate_dirs(tmp_path, records=_done_records())
    _provenance(d["ex1"], sim=FE46, sim_dir="/w/v12checks-armsim-fe46f6c/sim")
    monkeypatch.setattr(tracking, "log_run", lambda *a: (_ for _ in ()).throw(AssertionError("wandb called")))
    full = tmp_path / "cases.txt"
    full.write_text("ycb__bowl:0\n")
    with pytest.raises(SystemExit, match="provenance gate"):
        rr.main(["smoke-report", "--execute", f"{d['ex0']},{d['ex1']}", "--plan", d["plan"], "--full-cases", str(full),
                 "--out", str(tmp_path / "out")])
    assert not os.path.exists(tmp_path / "out" / "smoke_summary.json") and not os.path.exists(tmp_path / "out" / "smoke_summary.md")
