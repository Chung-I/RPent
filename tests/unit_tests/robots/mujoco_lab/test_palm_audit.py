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

import dataclasses
import math

import numpy as np
import pytest

from robots.mujoco_lab import palm_audit as pa

SIM = pytest.mark.skipif(not __import__("os").environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")
COMMIT = ('commit_std|{"a_transport": 2.0, "destination": "table_centre", "force": 10.0, "release_height": null, "s": 0.5, '
          '"v_lift": 0.15, "v_transport": 0.5}')


def test_an_action_string_is_its_row_id_and_its_fields():
    rid, act = pa.parse_action(COMMIT)
    assert rid == "commit_std"
    assert act == {"a_transport": 2.0, "destination": "table_centre", "force": 10.0, "release_height": None, "s": 0.5, "v_lift": 0.15,
                   "v_transport": 0.5}


def test_a_rows_action_is_found_by_its_id_and_an_absent_or_repeated_id_raises():
    rows = [{"id": "commit_std", "action": COMMIT}, {"id": "slow_lift", "action": COMMIT.replace("commit_std", "slow_lift", 1)}]
    assert pa.action_of(rows, "commit_std")["s"] == 0.5
    with pytest.raises(ValueError):
        pa.action_of(rows, "firm_grip")
    with pytest.raises(ValueError):
        pa.action_of(rows + rows[:1], "commit_std")
    with pytest.raises(ValueError):
        pa.action_of([{"id": "slow_lift", "action": COMMIT}], "slow_lift")       # the string names another row


def test_the_flags_are_the_deepest_overlap_against_each_threshold():
    clear = pa.flags_of(None)
    assert clear["depth_mm"] == 0.0 and not any(clear["flags"].values()) and clear["pairs"] == {}
    c = {"pairs": ["palm/object", "finger/table"], "first": {},
         "by_pair": {"palm/object": {"phase": "approach", "step": 3, "steps": 4, "depth": 0.0031, "geoms": ["palm", "hull0"],
                                     "phases": {"approach": {"steps": 3, "depth": 0.0031}, "close": {"steps": 1, "depth": 0.002}}},
                     "finger/table": {"phase": "lower", "step": 90, "steps": 1, "depth": 0.0004, "geoms": ["finger50", "table"],
                                      "phases": {"lower": {"steps": 1, "depth": 0.0004}}}}}
    f = pa.flags_of(c)
    assert f["depth_mm"] == pytest.approx(3.1)
    assert f["flags"] == {"any": True, "gt1": True, "gt2": True, "gt5": False}
    assert f["pairs"]["palm/object"]["body"] == "palm" and f["pairs"]["palm/object"]["kind"] == "object"
    assert f["pairs"]["palm/object"]["phases"]["close"]["depth_mm"] == pytest.approx(2.0)
    assert f["pairs"]["finger/table"]["first_phase"] == "lower" and f["pairs"]["finger/table"]["depth_mm"] == pytest.approx(0.4)


def test_a_replay_differs_on_the_label_the_artifact_or_the_error():
    sim = {"label": "dropped_low", "artifact": False, "error": None}
    assert pa.replay_differs(sim, {"outcome": "dropped_low"}, True) == {}
    assert pa.replay_differs(sim, {"outcome": "tipped"}, True) == {"label": ["dropped_low", "tipped"]}
    assert pa.replay_differs(sim, {"outcome": "dropped_low"}, False) == {"artifact": [False, True]}       # not at rest
    assert pa.replay_differs(sim, {"error": "boom"}, True) == {"label": ["dropped_low", None], "error": [None, "boom"]}


def _rec(decisions, sims):
    return {"object": "o", "idx": 0, "decisions": decisions, "sims": sims, "dropped": False}


def test_an_episodes_executed_rows_are_its_attempts_with_their_labels():
    s1 = {"seed_k": 0, "first": "commit_std", "label": "dropped_low", "menu2": {"decisions": {"R0|10|2": "slow_lift", "R4|10|-": "give_up"}},
          "second": {"slow_lift": {"label": "placed_stable"}}}
    s2 = {"seed_k": 0, "first": "firm_grip", "label": "placed_stable", "menu2": None, "second": {}}
    r = _rec({"R0|10|2": "commit_std", "R4|10|-": "firm_grip", "R1|10|2": "give_up"}, [s1, s2])
    assert pa.episode_rows(r, "R0|10|2") == [(1, "commit_std", "commit_std", "dropped_low"), (2, "commit_std", "slow_lift", "placed_stable")]
    assert pa.episode_rows(r, "R4|10|-") == [(1, "firm_grip", "firm_grip", "placed_stable")]
    assert pa.episode_rows(r, "R1|10|2") == []
    r2 = _rec({"R4|10|-": "commit_std"}, [s1])
    assert pa.episode_rows(r2, "R4|10|-") == [(1, "commit_std", "commit_std", "dropped_low")]      # gave up at the second


def test_a_ceiling_without_the_flagged_rows_drops_them_from_the_union():
    rows = [{"id": "across_0.5", "feasible": True, "current": True, "label": "placed_stable", "artifact": False, "error": None},
            {"id": "along_0.1", "feasible": True, "current": False, "label": "placed_stable", "artifact": False, "error": None},
            {"id": "across_0.3", "feasible": True, "current": True, "label": "tipped", "artifact": False, "error": None}]
    assert pa.ceiling_without(rows, set(), False)["success"] and pa.ceiling_without(rows, set(), True)["success"]
    c10, c3 = pa.ceiling_without(rows, {"across_0.5"}, False), pa.ceiling_without(rows, {"across_0.5"}, True)
    assert c10["success"] and c10["rows"] == ["along_0.1"] and not c3["success"] and c3["rows"] == ["across_0.3"]
    gone = pa.ceiling_without(rows, {"across_0.5", "along_0.1", "across_0.3"}, False)
    assert gone["give_up"] and not gone["success"]


# ---------------------------------------------------------------- on the lab

def _sugar_box():
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    env = MuJoCoLabEnv()
    env.reset("ycb__sugar_box", seed=0)
    return env


@SIM
def test_the_pinch_of_an_action_is_the_default_pinch_at_its_fraction():
    env = _sugar_box()
    assert pa.proposal_at(env, 0.5) == env.proposals()[1]
    with pytest.raises(ValueError):
        pa.proposal_at(env, 0.4)


@SIM
def test_todays_path_is_the_hand_path_with_each_yaw_as_a_rotation_about_the_vertical():
    env = _sugar_box()
    p, act = env.proposals()[1], pa.parse_action(COMMIT)[1]
    got = pa.today_path(env, p, act)
    want = env.hand_path(p, 0.15, 0.5, "table_centre", None, a_transport=2.0)
    assert [x[0] for x in got] == [x[0] for x in want] and all(np.array_equal(x[1], y[1]) for x, y in zip(got, want))
    assert all(np.allclose(x[2], [[math.cos(y[2]), -math.sin(y[2]), 0.0], [math.sin(y[2]), math.cos(y[2]), 0.0], [0.0, 0.0, 1.0]],
                           rtol=0.0, atol=1e-15) for x, y in zip(got, want))


@SIM
def test_a_vertical_pinch_at_mid_height_of_an_upright_box_is_flagged_on_the_palm():
    env = _sugar_box()                     # upright, 17.5 cm tall: the palm sits inside it at a mid-height pinch
    p, act = env.proposals()[1], pa.parse_action(COMMIT)[1]
    f = pa.audit_path(env, p, act)
    assert f["flags"]["gt5"] and f["pairs"]["palm/object"]["first_phase"] == "approach"
    assert set(f["pairs"]["palm/object"]["phases"]) <= {"approach", "close"} and "close" in f["pairs"]["palm/object"]["phases"]
    high = max(env.proposals((0.1, 0.9)), key=lambda q: q.point[2])  # 1.8 cm below the top: the palm is above the box
    top = pa.audit_path(env, high, act)
    assert top["depth_mm"] == 0.0 and not top["flags"]["any"] and top["pairs"] == {}


@SIM
def test_a_rows_audit_carries_its_key_and_pinch_and_leaves_the_sim_as_it_found_it():
    env = _sugar_box()
    act = pa.parse_action(COMMIT)[1]
    before = pa.state_sha(env)
    r = pa.audit_row(env, act, first="commit_std", attempt=1, row="commit_std")
    assert pa.state_sha(env) == before
    assert r["first"] == "commit_std" and r["attempt"] == 1 and r["s"] == 0.5 and r["error"] is None
    assert r["point"] == [float(x) for x in env.proposals()[1].point] and r["flags"]["gt5"]
    bad = pa.audit_row(env, dict(act, s=0.4), row="x")                 # no default pinch at 0.4: the row's error
    assert bad["error"].startswith("ValueError") and "flags" not in bad
    q = env.proposals((0.9,))[0]
    assert pa.audit_row(env, act, prop=q, id="across_0.9")["s"] == 0.9    # a given pinch (the pose set's, the pilot's)


@SIM
def test_the_state_hash_sees_a_moved_object():
    env = _sugar_box()
    a = pa.state_sha(env)
    env.sim.d.qpos[0] += 1e-6
    assert pa.state_sha(env) != a


@SIM
def test_the_audits_closed_fingers_put_each_pad_on_its_own_side_of_the_slab():
    from robots.mujoco_lab import sixdof

    env = _sugar_box()
    sc = env.s["scene"]
    p = env.proposals((0.5,), turn=-math.pi / 2)[0]                  # the 6 cm side (the 9.75 cm side is wider than the hand opens)
    lo, hi = env.pinch_extent(p.point, p.yaw)
    q = pa.pads_on_extent(env, p)
    assert 0.0 < min(q) and max(q) < sc.MAX_OPEN - 0.002
    L = sixdof.local_poses(env.sim, [env.sim.pad_l, env.sim.pad_r], q)
    assert L[env.sim.pad_l][0][0] + sc.PAD_HALF[0] == pytest.approx(lo, abs=1e-9)
    assert L[env.sim.pad_r][0][0] - sc.PAD_HALF[0] == pytest.approx(hi, abs=1e-9)


@SIM
def test_the_audit_checks_the_finger_boxes_against_the_object_with_the_pads_on_the_slabs_faces(monkeypatch):
    from robots.mujoco_lab import sixdof

    env = _sugar_box()
    p = env.proposals((0.9,), turn=-math.pi / 2)[0]
    act = pa.parse_action(COMMIT)[1]
    assert pa.audit_path(env, dataclasses.replace(p, width=0.02), act)["pairs"] == {}   # the stored width does not place the pads
    seen = {}
    real = sixdof.clearance_on_path

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(sixdof, "clearance_on_path", spy)
    pa.audit_path(env, p, act)
    assert seen["fingers_on_object"] is True and seen["closed_q"] == pa.pads_on_extent(env, p)


# ---------------------------------------------------------------- the summary

def _row(depth_mm, pairs=None, **k):
    f = pa.flags_of(None) if depth_mm == 0 else {"depth_mm": depth_mm, "flags": {n: depth_mm > v for n, v in pa.FLAG_MM.items()},
                                                    "pairs": pairs or {}}
    return dict(k, error=None, **f)


def test_a_rows_key_names_its_attempt_for_the_lab_arms_and_its_id_for_the_pose_rows():
    assert pa.row_key("stage2", "test", "o", 3, {"first": "a", "attempt": 2, "row": "b"}) == ("stage2", "test", "o", 3, "a", 2, "b")
    assert pa.row_key("poseset", "validation", "o", 3, {"id": "along_0.1"}) == ("poseset", "validation", "o", 3, "along_0.1")


def test_the_row_index_refuses_a_row_audited_twice():
    rec = {"source": "pilot", "side": "test", "object": "o", "idx": 0, "rows": [_row(0, id="x"), _row(2.0, id="y")]}
    assert set(pa.index_rows([rec])) == {("pilot", "test", "o", 0, "x"), ("pilot", "test", "o", 0, "y")}
    with pytest.raises(ValueError):
        pa.index_rows([rec, rec])


def test_a_rows_status_is_clear_flagged_or_not_audited():
    assert pa.status(_row(0)) == "clear" and pa.status(_row(0.5)) == "clear" and pa.status(_row(0.5), "any") == "flagged"
    assert pa.status(_row(1.5)) == "flagged" and pa.status(_row(1.5), "gt2") == "clear"
    assert pa.status(None) == pa.status({"error": "ValueError: x"}) == pa.status({"not_audited": "differs", "error": None}) == "not audited"
    assert pa.status(dict(_row(1.5), error="ValueError: x")) == pa.status(dict(_row(0), error="ValueError: x")) == "not audited"


def test_an_episode_is_clear_only_when_every_executed_row_is_audited_and_clear():
    rec = {"object": "o", "idx": 0, "decisions": {"K": "a", "G": "give_up"},
           "sims": [{"seed_k": 0, "first": "a", "label": "tipped", "menu2": {"decisions": {"K": "b"}}, "second": {"b": {"label": "placed_stable"}}}]}
    rows = {("stage2", "test", "o", 0, "a", 1, "a"): _row(0), ("stage2", "test", "o", 0, "a", 2, "b"): _row(3.0)}
    assert pa.episode_status(rows, "stage2", "test", rec, "K") == "flagged"
    assert pa.episode_status(rows, "stage2", "test", rec, "K", "gt5") == "clear"
    assert pa.episode_status(rows, "stage2", "test", rec, "G") == "clear"                  # a give-up executes nothing
    del rows[("stage2", "test", "o", 0, "a", 2, "b")]
    assert pa.episode_status(rows, "stage2", "test", rec, "K") == "not audited"
    assert pa.worst(["clear", "not audited", "flagged"]) == "flagged" and pa.worst([]) == "clear"


def test_the_flagged_shares_count_placed_and_irreversible_rows_apart():
    got = pa.label_shares([("placed_stable", "flagged"), ("placed_stable", "clear"), ("dropped_high", "flagged"), ("tipped", "clear"),
                           ("crushed", "not audited")])
    assert got["placed"] == {"rows": 2, "audited": 2, "flagged": 1, "share": 0.5}
    assert got["irreversible"] == {"rows": 2, "audited": 1, "flagged": 1, "share": 1.0}
    assert got["all"] == {"rows": 5, "audited": 4, "flagged": 2, "share": 0.5}


def test_reading_one_counts_a_row_under_each_body_obstacle_and_phase_deeper_than_the_threshold():
    deep = {"palm/object": {"body": "palm", "kind": "object", "depth_mm": 6.0, "phases": {"approach": {"depth_mm": 6.0}, "close": {"depth_mm": 0.5}}},
            "finger/table": {"body": "finger", "kind": "table", "depth_mm": 0.4, "phases": {"lower": {"depth_mm": 0.4}}}}
    rows = [("o1", _row(6.0, deep)), ("o1", _row(0)), ("o2", _row(1.5, {"wrist/fixture": {"body": "wrist", "kind": "fixture", "depth_mm": 1.5,
                                                                                          "phases": {"transport": {"depth_mm": 1.5}}}})),
            ("o2", {"not_audited": "differs", "error": None})]
    r = pa.reading1(rows)
    assert r["rows"] == 4 and r["audited"] == 3 and r["flagged"] == {"any": 2, "gt1": 2, "gt2": 1, "gt5": 1}
    assert r["by_body"] == {"palm": 1, "wrist": 1} and r["by_obstacle"] == {"object": 1, "fixture": 1}
    assert r["by_phase"] == {"approach": 1, "transport": 1}
    assert r["by_object"]["o1"] == {"rows": 2, "flagged": 1, "depth_median_mm": 6.0, "depth_max_mm": 6.0}
    assert r["by_object"]["o2"] == {"rows": 1, "flagged": 1, "depth_median_mm": 1.5, "depth_max_mm": 1.5}


def test_the_gate_names_every_field_that_differs_from_the_published_numbers():
    want = {"cases": 10, "success": {"diff": -0.0125, "lo": -0.03, "hi": 0.0}, "irr": {"diff": -0.13, "lo": -0.17, "hi": -0.09}}
    assert pa.differs(want, want) == []
    got = {"cases": 11, "success": dict(want["success"], lo=-0.031), "irr": want["irr"]}
    assert pa.differs(got, want) == ["cases 11 vs 10", "success.lo -0.031 vs -0.03"]
    assert pa.differs(got, want, ("irr",)) == ["cases 11 vs 10"]
    assert pa.differs(dict(want, irr=dict(want["irr"], hi=-0.09 + 1e-12)), want) == [f"irr.hi {-0.09 + 1e-12!r} vs -0.09"]    # exact


def test_the_pose_views_without_their_flagged_rows_recompute_both_ceilings():
    rows = [{"id": "across_0.5", "feasible": True, "current": True, "label": "placed_stable", "artifact": False, "error": None, "seconds": 1},
            {"id": "along_0.3", "feasible": True, "current": False, "label": "placed_stable", "artifact": False, "error": None, "seconds": 1},
            {"id": "along_0.9", "feasible": False, "current": False, "why": "width"}]
    from robots.mujoco_lab.curve_poseset import ceiling
    v = {"object": "o", "idx": 0, "rows": rows, "c10": ceiling(rows), "c3": ceiling(rows, True), "upright": True}
    idx = {("poseset", "test", "o", 0, "across_0.5"): _row(4.0), ("poseset", "test", "o", 0, "along_0.3"): _row(0)}
    (w,) = pa.poseset_without([v], idx)
    assert w["c10"]["success"] and w["c10"]["rows"] == ["along_0.3"] and not w["c3"]["success"] and w["c3"]["give_up"]
    assert w["removed"] == ["across_0.5"] and w["kept_unaudited"] == []
    assert pa.poseset_without([v], idx, level="gt5")[0]["c3"] == v["c3"]
    del idx[("poseset", "test", "o", 0, "along_0.3")]
    (w,) = pa.poseset_without([v], idx)
    assert w["kept_unaudited"] == ["along_0.3"] and w["c10"]["success"] and w["c10"]["rows"] == ["along_0.3"]    # kept, not removed


def test_a_files_sha256_is_read_from_its_reference_with_or_without_a_commit():
    h = "0123456789abcdef" * 4
    assert pa.ref_sha(f"/x/locked.json (sha256 {h}, commit 21778cdc)") == pa.ref_sha(f"/x/locked.json (sha256 {h})") == h
    assert pa.ref_sha("/x/locked.json") is None and pa.ref_sha(None) is None


def _published(lk="a" * 64, s2="b" * 64, osh="c" * 64):
    s2sum = {"q1": {"boot": {"a": "K0", "b": "K4"}}, "provenance": {"runs": "s/0,s/1", "locked": f"/x/locked.json (sha256 {lk}, commit 21778cdc)"}}
    common = {"keys": ["K0", "K4"], "locked": f"/y/locked.json (sha256 {lk})", "stage2": "s/1,s/0", "stage2_summary": f"/s.json (sha256 {s2})",
              "stage1": ["t/1", "t/0"]}
    osum = {"provenance": dict(common, runs="o/0,o/1")}
    psum = {"provenance": dict(common, runs="p/0", oracle="o/1,o/0", oracle_summary=f"/o.json (sha256 {osh})")}
    return s2sum, osum, psum, {"locked": lk, "stage2_summary": s2, "oracle_summary": osh}


def test_the_published_summaries_must_link_the_same_lock_keys_and_inputs():
    s2sum, osum, psum, sha = _published()
    assert pa.provenance_errors(s2sum, osum, psum, ("K0", "K4"), sha, ["t/0", "t/1"]) == []
    assert pa.provenance_errors(s2sum, osum, psum, ("K0", "K9"), sha, ["t/0", "t/1"]) == [
        "the Stage 2 summary compared other keys than the lock's", "the oracle summary read other keys", "the pose-set summary read other keys"]
    assert pa.provenance_errors(s2sum, osum, psum, ("K0", "K4"), dict(sha, locked="d" * 64), ["t/0", "t/1"]) == [
        "the Stage 2 summary read another lock", "the oracle summary read another lock", "the pose-set summary read another lock"]
    bad = {"provenance": dict(psum["provenance"], oracle="o/0", oracle_summary=f"/o.json (sha256 {'e' * 64})", stage2="s/0")}
    assert pa.provenance_errors(s2sum, osum, bad, ("K0", "K4"), sha, ["t/0", "t/1"]) == [
        "the pose-set summary read other Stage 2 runs", "the pose-set summary read other oracle runs", "the pose-set summary read another oracle summary"]
    assert pa.provenance_errors(s2sum, osum, psum, ("K0", "K4"), dict(sha, stage2_summary="f" * 64), ["t/0", "t/1"]) == [
        "the oracle summary read another Stage 2 summary", "the pose-set summary read another Stage 2 summary"]
    assert pa.provenance_errors(s2sum, osum, psum, ("K0", "K4"), sha, ["t/0"]) == [
        "the oracle summary read another Stage 1", "the pose-set summary read another Stage 1"]


def test_the_audit_must_hold_every_stored_row_and_no_other():
    idx = {("pilot", "test", "o", 0, "x"): _row(0), ("pilot", "test", "o", 0, "y"): _row(0)}
    want = {("pilot", "test", "o", 0, "x"): "placed_stable", ("pilot", "test", "o", 0, "z"): "tipped"}
    assert pa.row_coverage(idx, want) == {"missing": [("pilot", "test", "o", 0, "z")], "extra": [("pilot", "test", "o", 0, "y")]}
    assert pa.row_coverage(idx, dict.fromkeys(idx)) == {"missing": [], "extra": []}


def test_the_gate_compares_an_empty_view_without_raising():
    empty = {"cases": 0, "success": None, "irr": None}
    want = {"cases": 10, "success": {"diff": -0.0125, "lo": -0.03, "hi": 0.0}, "irr": {"diff": -0.13, "lo": -0.17, "hi": -0.09}}
    assert pa.differs(empty, dict(empty)) == []
    assert pa.differs(empty, want) == ["cases 0 vs 10", f"success None vs {want['success']!r}", f"irr None vs {want['irr']!r}"]
