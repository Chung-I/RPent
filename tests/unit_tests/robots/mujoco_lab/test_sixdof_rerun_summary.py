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

"""The level-A summary (design 2026-11-08-sixdof-rerun-design.md, section 4 and the level-A summary rulings of 9 Nov), on
synthetic records only."""

import json
import os

import pytest

from robots.mujoco_lab import curve_sixdof_rerun as rr
from robots.mujoco_lab import sixdof_rerun_summary as ss

V, A = rr.VERTICAL_BIN, rr.LEVEL_A_BIN
FLAG = {"first": {"pair": "palm|object", "phase": "approach", "depth": 2.5, "step": 3, "steps": 40}, "pairs": ["palm|object"]}


def _run(label, **k):
    return dict({"label": label, "artifact": False, "error": None, "damage": None, "loss_phase": None, "max_z": 0.2,
                 "box_contacts": {"first": None}, "replan": None}, **k)


def _r30(i, rank, label, kept=False, kept_b=False, kept_all=True, kept_all_b=True, flagged=False, **run):
    r = {"id": f"g{i}", "rank": rank, "approach_deg": 10.0, "bin": V, "kept": kept, "kept_b": kept_b, "kept_all": kept_all,
         "kept_all_b": kept_all_b, "clearance": FLAG if flagged else None, "why": None if kept_all else "wider than the gripper opening",
         "why_b": None if kept_all_b else "wider than the gripper opening", "reach": "ok"}
    if kept_all or kept_all_b:
        r["run"] = _run(label, **run)
    return r


def _r6(i, rank, label, why=None, why_b=None, free="ok", g2=False, deg=40.0, candidate=True, **run):
    r = {"id": f"g{i}", "rank": rank, "approach_deg": deg, "bin": A, "why": why, "why_b": why_b, "why_width": None,
         "why_width_b": None, "clearance": None, "disagree": None, "reach": {"candidate": 0 if candidate else None, "box_hit": None},
         "candidate": {"index": 0, "twin": False, "seed": "aimed", "q": [0.0] * 7} if candidate else None}
    if candidate:
        ok = free == "ok" and not g2
        r["run"] = _run(label if ok else None, free=free, unreachable={"check": "grasp", "mm": 18.0} if g2 else None,
                        free_failure={"reason": "planner spheres", "hits": [], "first": None, "status": "x"} if free == "failed" else None, **run)
    return r


def _ex(o="ycb__bowl", idx=0, side="test", rows30=(), rows6=(), **k):
    return dict({"object": o, "idx": idx, "side": side, "hammer": "hammer" in o, "upright": False, "kind": "other", "thin": False,
                 "com_f": 0.5, "length": 0.2, "skipped": None, "dropped": False, "stop": None, "error": None, "exec_stop": None,
                 "exec_error": None, "rows30": list(rows30), "rows6": list(rows6), "n_grasps": 200,
                 "counts30": {"generated": 200, "vertical": len(rows30)}, "counts6": {"generated": 200, "level_a": len(rows6)}}, **k)


def _counts(ex):
    """The record's prepass counts as the prepass writes them, from its rows."""
    r30, r6 = ex["rows30"], ex["rows6"]
    n = len(r30)
    ex["counts30"] = {"generated": ex["n_grasps"], "vertical": n, "mapped": n, "width": n, "centred": n, "reach": n,
                      "distinct": sum(r["kept_all"] for r in r30), "kept": sum(r["kept"] for r in r30), "width_b": n, "pads_free": n,
                      "reach_b": n, "distinct_b": sum(r["kept_all_b"] for r in r30), "kept_b": sum(r["kept_b"] for r in r30),
                      "clearance_fail": sum(r["clearance"] is not None for r in r30)}
    ex["counts6"] = {"generated": ex["n_grasps"], "level_a": len(r6), "chosen": sum(r["candidate"] is not None for r in r6),
                     "disagree": sum(r["disagree"] is not None for r in r6)}
    return ex


def _tg(ex, f=None, h=None, near=None):
    """The tags record of an executed record: every row on the axis at f 0.5 (or ``f``: {id: f}), h 0.5 (``h``) and no
    pose-set row near (``near``)."""
    f, h, near = f or {}, h or {}, near or {}

    def t(r):
        return {"id": r["id"], "f": f.get(r["id"], 0.5), "r_axis": 0.0, "h": h.get(r["id"], 0.5), "near": near.get(r["id"]),
                "near_current": False}
    return {"object": ex["object"], "idx": ex["idx"], "rows30": [t(r) for r in ex["rows30"]],
            "rows6": [dict(t(r), point=[0.0, 0.0, 0.0], pad=[0.0, 0.0, 0.004], yaw=0.0, approach_deg=r["approach_deg"]) for r in ex["rows6"]]}


# ------------------------------------------------------------------ the rows and the ceilings

def test_a_cases_rows_carry_each_arms_flag_the_tags_and_the_run_and_merge_the_bins_by_rank():
    ex = _ex(rows30=[_r30(0, 0, "dropped_low", kept=True, kept_b=True), _r30(1, 3, "placed_stable", flagged=True),
                     _r30(2, 5, None, kept_all=False, kept_all_b=False)],
             rows6=[_r6(3, 1, "placed_stable"), _r6(4, 2, "tipped", why_b="the pads land on the object"), _r6(5, 4, None, free="failed"),
                    _r6(6, 6, None, g2=True), _r6(7, 7, None, candidate=False, why="clearance: palm|object")])
    rows = ss.level_a_rows(ex, _tg(ex, f={"g3": 0.9}))
    assert [r["id"] for r in rows] == ["g0", "g3", "g4", "g1", "g5", "g2", "g6", "g7"]      # one rank order over both bins
    by = {r["id"]: r for r in rows}
    assert by["g0"]["cap"] and by["g0"]["a30"] and by["g0"]["a30n"] and by["g0"]["a45"] and not by["g0"]["x6"]
    assert by["g1"]["flagged"] and by["g1"]["a30n"] and not by["g1"]["a30"] and not by["g1"]["a45"]
    assert not by["g2"]["ran"] and not by["g2"]["a30n"]
    assert by["g3"]["x6"] and by["g3"]["a6"] and by["g3"]["a6_b"] and by["g3"]["a45"] and by["g3"]["f"] == 0.9 and by["g3"]["bin"] == A
    assert by["g4"]["a6"] and not by["g4"]["a6_b"] and by["g4"]["a45"] and not by["g4"]["a45_b"]
    assert not any(by[k]["x6"] for k in ("g5", "g6", "g7"))     # a cuRobo failure, a G2 failure, no candidate: funnel steps
    c = ss.level_a_ceilings(rows)
    assert c["a30"]["success"] is False and c["a30"]["rows"] == ["g0"] and c["a30n"]["success"] and c["a30n"]["rows"] == ["g1"]
    assert c["a45"]["success"] and c["a45"]["rows"] == ["g3"] and c["a45_b"]["rows"] == ["g3"] and c["cap"]["rows"] == ["g0"]


def test_a_row_without_its_tags_is_refused():
    ex = _ex(rows30=[_r30(0, 0, "placed_stable")], rows6=[_r6(1, 1, "tipped")])
    tg = _tg(ex)
    tg["rows6"] = []
    with pytest.raises(ValueError, match="tags"):
        ss.level_a_rows(ex, tg)


def test_a_case_drops_when_a_pass_stopped_it_or_a_level_a_arm_is_undetermined_but_not_for_the_capped_check():
    ok = _ex(rows30=[_r30(0, 0, "placed_stable", kept=True)], rows6=[_r6(1, 1, "tipped")])
    assert ss.case_view(ok, _tg(ok))["dropped"] is None
    for bad in (dict(ok, stop="x"), dict(ok, dropped=True), dict(ok, exec_stop="y"), dict(ok, exec_error="z")):
        assert ss.case_view(bad, _tg(bad))["dropped"]
    und = _ex(rows30=[_r30(0, 0, "tipped", kept=True)], rows6=[_r6(1, 1, "tipped", artifact=True)])
    v = ss.case_view(und, _tg(und))
    assert "a45" in v["dropped"] and "a30" not in v["dropped"].replace("a30n", "")
    cap = _ex(rows30=[_r30(0, 0, "tipped", kept=True, artifact=True, kept_all=False, kept_all_b=False)])
    cap["rows30"][0]["run"] = _run("tipped", artifact=True)
    assert ss.case_view(cap, _tg(cap))["dropped"] is None                   # only the capped check arm is undetermined


# ------------------------------------------------------------------ reading 2: the headline and its labels

def _b(diff, lo, hi, n=10):
    return {"cases": n, "success": {"diff": diff, "lo": lo, "hi": hi}}


def test_the_headline_reads_the_pose_sets_three_labels_against_the_2_point_margin():
    assert ss.level_a_reading(_b(0.05, 0.01, 0.09)) == ss.ABOVE
    assert ss.level_a_reading(_b(0.0, -0.03, 0.019)) == ss.BELOW
    assert ss.level_a_reading(_b(0.03, -0.01, 0.07, n=81)) == "undecided at n = 81"
    assert ss.level_a_reading(_b(0.015, 0.001, 0.03)) == "undecided at n = 10"          # above 0 but not above the margin


def _view(o, idx, a30, a45, side="test", a30n=None, **k):
    def ep(s):
        return {"success": s, "irr": False, "give_up": False, "attempts": 1, "determined": True, "rows": [], "labels": []}
    a = {"a30": ep(a30), "a30_b": ep(a30), "a30n": ep(a30 if a30n is None else a30n), "a30n_b": ep(a30 if a30n is None else a30n),
         "a45": ep(a45), "a45_b": ep(a45), "cap": ep(a30), "cap_b": ep(a30)}
    return dict({"object": o, "idx": idx, "side": side, "a": a, "dropped": None, "rows": [], "hammer": False}, **k)


def test_the_headline_pairs_by_case_with_the_object_bootstrap_and_the_pitcher_8_line_drops_that_case_only():
    vs = [_view(o, i, False, i % 2 == 0) for o in ("ycb__bowl", "ycb__pitcher", "hope__milk_carton") for i in range(9)]
    h = ss.headline(vs)
    assert h["A"]["cases"] == 27 and h["A"]["objects"] == 3 and h["A"]["draws"] == 2000 and h["A"]["seed"] == 20261108
    assert h["A"]["success"]["diff"] == pytest.approx(15 / 27) and h["A"]["reading"] == ss.ABOVE and "B" in h
    p8 = ss.headline(vs, without=("ycb__pitcher", 8))
    assert p8["A"]["cases"] == 26 and p8["A"]["success"]["diff"] == pytest.approx(14 / 26)
    assert ss.headline(vs, without=("ycb__pitcher", 99))["A"]["cases"] == 27


# ------------------------------------------------------------------ reading 1: the funnels

def test_the_funnel_counts_each_bins_steps_by_object_and_the_shares_of_the_generator():
    ex = _ex(rows30=[_r30(0, 0, "placed_stable", kept=True), _r30(1, 2, "tipped", flagged=True),
                     _r30(2, 5, None, kept_all=False, kept_all_b=False, flagged=True)],
             rows6=[_r6(3, 1, "placed_stable"), _r6(4, 3, None, free="failed"), _r6(5, 4, None, g2=True),
                    _r6(6, 6, None, candidate=False, why="clearance: palm|object")])
    ex["rows6"][3]["clearance"] = FLAG
    ex["rows6"][3]["disagree"] = "clearance fail, reach6 pass"
    ex["counts30"] = {"generated": 200, "vertical": 3, "kept": 1, "kept_b": 0}
    ex["counts6"] = {"generated": 200, "level_a": 4, "chosen": 3, "disagree": 1}
    v = ss.case_view(ex, _tg(ex))
    f = ss.funnel([v])
    b = f["by_object"]["ycb__bowl"]
    assert b["30"]["within"] == 3 and b["30"]["uncapped"] == 2 and b["30"]["ran"] == 2
    assert b["30"]["clearance_fail"] == 2 and b["30"]["flagged"] == 1             # flagged rows that ran
    assert b["45"]["within"] == 4 and b["45"]["chosen"] == 3 and b["45"]["planned"] == 2 and b["45"]["g2_passed"] == 1
    assert b["45"]["labelled"] == 1 and b["45"]["rule_a"] == 1 and b["45"]["distinct"] == 1 and b["45"]["disagree"] == 1
    assert f["all"]["generated"] == 200 and f["all"]["share_within_30"] == pytest.approx(3 / 200)
    assert f["all"]["share_within_45"] == pytest.approx(7 / 200) and f["all"]["share_every_step"] == pytest.approx(2 / 200)
    assert f["clearance_pairs"]["30"] == {"palm|object": 2} and f["clearance_pairs"]["45"] == {"palm|object": 1}
    assert f["free_failed"] == 1 and f["free_failed_by"] == {"planner spheres": 1} and f["g2_failed"] == {"grasp": 1}
    assert f["why6"]["ycb__bowl"] == {"a": {"None": 3, "clearance: palm|object": 1}, "b": {"None": 4}}


def _funnel_record():
    ex = _ex(rows30=[_r30(0, 0, "placed_stable", kept=True), _r30(1, 2, "tipped", flagged=True), _r30(2, 5, None, kept_all=False, kept_all_b=False)],
             rows6=[_r6(3, 1, "placed_stable"), _r6(4, 3, None, free="failed"), _r6(5, 4, None, g2=True),
                    _r6(6, 6, None, candidate=False, why="clearance: palm|object")])
    ex["rows6"][3]["clearance"] = FLAG
    ex["rows6"][3]["disagree"] = "clearance fail, reach6 pass"
    return _counts(ex)


def test_the_counts_check_agrees_with_the_counts_reports_own_code_on_the_same_records_and_lists_each_difference():
    ex = _funnel_record()
    f = ss.funnel([ss.case_view(ex, _tg(ex))])
    rep = rr.smoke_report([ex], [], {})
    assert ss.counts_check(f, rep) == []
    rep["counts"]["free"]["miss"] = 1
    rep["why6"]["ycb__bowl"]["None"] = 9
    rep["funnel30"]["ycb__bowl"]["kept"] = 7
    bad = ss.counts_check(f, rep)
    assert len(bad) == 3 and any(b.startswith("free.miss: 0") for b in bad) and any(b.startswith("why6.ycb__bowl") for b in bad)
    assert any(b.startswith("funnel30.ycb__bowl.kept: 1") for b in bad)


# ------------------------------------------------------------------ readings 3 to 5 and the additions

def test_the_hammer_line_classes_each_rule_at_each_cumulative_angle_and_lists_every_placing_row_with_its_place():
    ex = _ex(o="handal__hammer_1", com_f=0.4, length=0.3,
             rows30=[_r30(0, 0, "not_lifted")], rows6=[_r6(1, 1, "placed_stable", deg=38.0), _r6(2, 2, "tipped", why="x")])
    v = ss.case_view(ex, _tg(ex, f={"g1": 0.6}, h={"g1": 0.9}))
    h = ss.hammers([v])
    assert h["classes"]["a30"]["not lifted by any row"] == 1 and h["classes"]["a45"]["placed by a row"] == 1
    p = h["per_case"][0]["placed"]
    assert p == [{"id": "g1", "bin": A, "approach_deg": 38.0, "f": 0.6, "balance_m": pytest.approx(0.06), "head_m": pytest.approx(0.18),
                  "tag": "rim or top edge", "rules": ["a45", "a45_b"]}]
    assert h["per_case"][0]["lifted"] == ["g1", "g2"]


def test_the_tilted_placing_rows_are_counted_by_tag_rule_and_object_with_the_gain_cases():
    ex = _ex(rows30=[_r30(0, 0, "tipped")], rows6=[_r6(1, 1, "placed_stable"), _r6(2, 2, "placed_stable", why="x")])
    v = ss.case_view(ex, _tg(ex, h={"g1": 0.9}, near={"g2": "across_0.5"}))
    t = ss.tilted_tags([v])
    assert t["a"]["rows"] == {"rim or top edge": 1} and t["b"]["rows"] == {"rim or top edge": 1, "an axis-rule pose": 1}
    assert t["a"]["gain"] == {"cases": 1, "tags": {"rim or top edge": 1}} and t["a"]["loss"] == {"cases": 0}
    o = ss.objects_line([v], ("ycb__bowl",))["ycb__bowl"]
    assert o["a"]["cases_with_tilted_row"] == 1 and o["a"]["tilted_rows"] == 1 and o["b"]["tilted_rows"] == 2
    assert o["a"]["tilted_placing_tags"] == {"rim or top edge": 1} and o["a"]["placed_45"] == 1 and o["a"]["placed_30"] == 0
    assert o["b"]["tilted_labels"] == {"placed_stable": 2}
    ex2 = _ex(idx=1, rows30=[_r30(0, 0, "placed_stable")], rows6=[_r6(1, 1, "placed_stable")])
    t2 = ss.tilted_tags([v, ss.case_view(ex2, _tg(ex2))])
    assert t2["a"]["gain"]["cases"] == 1 and t2["a"]["vertical_rows"] == {"on the axis at another fraction or yaw": 1}


def test_the_tilted_rows_distinct_count_drops_each_rules_duplicates_on_the_mapped_point_and_yaw():
    ex = _ex(rows6=[_r6(1, 1, "placed_stable"), _r6(2, 2, "tipped"), _r6(3, 3, "tipped", why="x")])
    tg = _tg(ex)
    tg["rows6"][1]["point"] = [0.0, 0.05, 0.0]                 # 5 cm from the others: not a duplicate
    f = ss.funnel([ss.case_view(ex, tg)])["by_object"]["ycb__bowl"]["45"]
    assert (f["rule_a"], f["distinct"], f["rule_b"], f["distinct_b"]) == (2, 2, 3, 2)


def test_the_loss_phase_split_counts_the_lifted_rows_that_did_not_place_by_phase_with_the_vertical_rows_beside():
    ex = _ex(rows30=[_r30(0, 0, "dropped_low", loss_phase="transport"), _r30(1, 1, "placed_stable")],
             rows6=[_r6(2, 2, "dropped_low", loss_phase="transport"), _r6(3, 3, "tipped"), _r6(4, 4, "not_lifted"),
                    _r6(5, 5, "dropped_high", loss_phase="lower"), _r6(6, 6, "placed_stable"), _r6(7, 7, "crushed", artifact=True)])
    s = ss.loss_phase_split([ss.case_view(ex, _tg(ex))])
    assert s["45"] == {"rows": 3, "phases": {"transport": 1, "none": 1, "lower": 1}, "labels": {"dropped_low": 1, "tipped": 1, "dropped_high": 1}}
    assert s["30"]["rows"] == 1 and s["30"]["phases"] == {"transport": 1}


def test_the_tilt_at_release_is_the_approach_angle_of_each_executed_tilted_row_by_object_and_by_outcome():
    ex = _ex(rows6=[_r6(1, 1, "placed_stable", deg=33.0), _r6(2, 2, "not_lifted", deg=41.0), _r6(3, 3, "tipped", deg=44.0),
                    _r6(4, 4, None, g2=True, deg=35.0)])
    t = ss.release_tilt([ss.case_view(ex, _tg(ex))])
    assert t["by_object"]["ycb__bowl"] == {"rows": 3, "median": 41.0, "min": 33.0, "max": 44.0}
    assert t["by_outcome"]["placed"]["rows"] == 1 and t["by_outcome"]["not lifted"]["median"] == 41.0
    assert t["by_outcome"]["lifted, not placed"]["median"] == 44.0


def test_the_by_cause_table_counts_each_bins_executed_rows_by_box_contact_and_raise_with_placed_and_damage():
    hit = {"first": {"phase": "lift", "box": "palm", "partner": "object", "mm": 1.0}}
    ex = _ex(rows30=[_r30(0, 0, "placed_stable", box_contacts=hit), _r30(1, 1, "crushed", replan={"raise_mm": 4.0, "d_mm": 4.0})],
             rows6=[_r6(2, 2, "crushed", box_contacts=hit, replan={"raise_mm": 2.0, "d_mm": 12.0}), _r6(3, 3, "placed_stable")])
    v = ss.case_view(ex, _tg(ex))
    c = ss.by_cause([v])
    assert c["30"]["contact only"] == {"rows": 1, "placed": 1, "damage": 0} and c["30"]["raise only"] == {"rows": 1, "placed": 0, "damage": 1}
    assert c["45"]["both"] == {"rows": 1, "placed": 0, "damage": 1} and c["45"]["neither"] == {"rows": 1, "placed": 1, "damage": 0}


def test_the_replan_d_bins_take_g1ps_rule_rows_that_rose_and_were_held_by_d_with_the_others_counted():
    def rp(d, held=True, lifted=True, up=0.0):
        return {"d_mm": d, "held": held, "lifted": lifted, "raise_mm": up, "unreachable": False}
    ex = _ex(rows30=[_r30(0, 0, "placed_stable", replan=rp(0.5)), _r30(1, 1, "dropped_low", replan=rp(4.0, up=4.0)),
                     _r30(2, 2, "not_lifted", replan=rp(None, held=False, lifted=False))],
             rows6=[_r6(3, 3, "tipped", replan=rp(12.0, up=12.0)), _r6(4, 4, "dropped_low", replan=rp(2.0, held=False)), _r6(5, 5, "placed_stable")])
    d = ss.replan_d_bins([ss.case_view(ex, _tg(ex))])
    assert d["30"]["bins"]["≤ 1 mm"] == {"rows": 1, "raised": 0, "unreachable": 0}
    assert d["30"]["bins"]["3 to 10 mm"] == {"rows": 1, "raised": 1, "unreachable": 0} and d["30"]["not_lifted"] == 1
    assert d["45"]["bins"]["> 10 mm"] == {"rows": 1, "raised": 1, "unreachable": 0} and d["45"]["bins"]["1 to 3 mm"]["rows"] == 0
    assert d["45"]["lifted_not_held"] == 1 and d["45"]["no_record"] == 1 and d["30"]["no_record"] == 0


def test_ruling_3_compares_the_capped_rows_rerun_labels_with_the_corrected_ones_and_checks_capped_within_uncapped():
    ex = _ex(rows30=[_r30(0, 0, "placed_stable", kept=True, kept_b=True), _r30(1, 1, "tipped", kept=True),
                     _r30(2, 2, "dropped_low", kept_b=True, box_contacts={"first": {"phase": "lift", "box": "palm", "partner": "object", "mm": 1.0}})])
    v = ss.case_view(ex, _tg(ex))
    corr = {("ycb__bowl", 0): {"rows": [{"id": "g0", "label": "placed_stable", "kept": True, "kept_b": True, "seconds": 1.0},
                                        {"id": "g1", "label": "tipped", "kept": True, "seconds": 1.0},
                                        {"id": "g2", "label": "placed_stable", "kept_b": True, "seconds": 1.0}]}}
    r3 = ss.ruling3([v], corr)
    assert r3["rows"] == {"compared": 3, "agree": 2, "differ": 1, "missing": 0}
    assert r3["differences"] == [{"object": "ycb__bowl", "idx": 0, "row": "g2", "corrected": "placed_stable", "rerun": "dropped_low",
                                  "cause": "contact only"}]
    assert r3["capped_places_uncapped_not"] == 0


# ------------------------------------------------------------------ the tags units' gate and the command

def _unit(path, name, lines, cfg, code=None, sim=None):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, name), "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in lines)
    json.dump(cfg, open(os.path.join(path, "run_config.json"), "w"))
    json.dump({}, open(os.path.join(path, "run_metrics.json"), "w"))
    open(os.path.join(path, "run.exit"), "w").write("EXIT 0\n")
    with open(os.path.join(path, "provenance.txt"), "w") as f:
        f.write(f"code {code or rr.LEVEL_A_CODE} (detached run worktree), dirty files 0\n"
                f"sim /w/armsim-contacts-ee882f9b/sim, last commit {sim or rr.LEVEL_A_SIM}, dirty files 0\n")
    return str(path)


def _record(tmp_path):
    """Two prepass units, their execute units and tags units, and a plan step."""
    from robots.mujoco_lab.curve import _sha

    pre = [_unit(tmp_path / f"pre{k}", "prepass_records.jsonl", [{"object": "ycb__bowl", "idx": k}], {}) for k in (0, 1)]
    ex = [_unit(tmp_path / f"ex{k}", "exec_records.jsonl", [], {"prepass": pre[k]}) for k in (0, 1)]
    tg = [_unit(tmp_path / f"tg{k}", "tags_records.jsonl", [], {"prepass": pre[k],
                "prepass_records_sha256": _sha(os.path.join(pre[k], "prepass_records.jsonl"))}, code=ss.TAGS_CODE) for k in (0, 1)]
    plan = _unit(tmp_path / "plan", "plan_records.jsonl", [], {})
    return {"pre": pre, "ex": ex, "tg": tg, "plan": plan}


def test_the_tags_gate_passes_one_tags_unit_per_prepass_unit_on_the_tags_code_and_the_level_a_sim(tmp_path):
    d = _record(tmp_path)
    g = ss.tags_gate(d["tg"], d["pre"])
    assert g == {"code": ss.TAGS_CODE, "sim": rr.LEVEL_A_SIM, "tags_units": 2}


@pytest.mark.parametrize("change", ["code", "sim", "exit", "sha", "twice", "other"])
def test_the_tags_gate_refuses_a_unit_off_its_code_or_sim_on_other_records_or_a_prepass_unit_without_exactly_one(tmp_path, change):
    d = _record(tmp_path)
    t1, cfg = d["tg"][1], json.load(open(os.path.join(d["tg"][1], "run_config.json")))
    if change == "code":
        _unit(t1, "tags_records.jsonl", [], cfg, code=rr.LEVEL_A_CODE)
        want = f"{t1}: code {rr.LEVEL_A_CODE}"
    elif change == "sim":
        _unit(t1, "tags_records.jsonl", [], cfg, code=ss.TAGS_CODE, sim="e8" * 20)
        want = f"{t1}: sim {'e8' * 20}"
    elif change == "exit":
        open(os.path.join(t1, "run.exit"), "w").write("EXIT 3\n")
        want = f"{t1}: EXIT 3"
    elif change == "sha":
        json.dump(dict(cfg, prepass_records_sha256="0" * 64), open(os.path.join(t1, "run_config.json"), "w"))
        want = f"{t1}: it read other prepass records"
    elif change == "twice":
        json.dump(json.load(open(os.path.join(d["tg"][0], "run_config.json"))), open(os.path.join(t1, "run_config.json"), "w"))
        want = f"{d['pre'][1]}: not one tags unit"
    else:
        json.dump({"prepass": str(tmp_path / "elsewhere")}, open(os.path.join(t1, "run_config.json"), "w"))
        want = f"{t1}: its prepass run is not one of the execute units'"
    with pytest.raises(SystemExit) as e:
        ss.tags_gate(d["tg"], d["pre"])
    assert str(e.value).startswith("tags gate") and want in str(e.value)


def test_the_command_runs_both_provenance_gates_before_it_opens_a_record_or_writes(tmp_path, monkeypatch):
    from robots.mujoco_lab import tracking

    d = _record(tmp_path)
    _unit(d["ex"][1], "exec_records.jsonl", [], {"prepass": d["pre"][1]}, sim="e8" * 20)
    opened = []
    monkeypatch.setattr(ss, "_open_inputs", lambda *a, **k: opened.append("inputs"))
    monkeypatch.setattr(ss, "_open_level_a", lambda *a, **k: opened.append("level A"))
    monkeypatch.setattr(tracking, "log_run", lambda *a: (_ for _ in ()).throw(AssertionError("wandb called")))
    args = ["--execute", ",".join(d["ex"]), "--plan", d["plan"], "--tags", ",".join(d["tg"]), "--poseset-summary", "p",
            "--pilot-summary", "g", "--readings-summary", "r", "--contacts", "c", "--replan", "q", "--stage2c-summary", "s",
            "--oracle-c-summary", "o", "--counts", "n", "--smoke", "m", "--g1", "x", "--out", str(tmp_path / "out")]
    with pytest.raises(SystemExit, match="provenance gate"):
        ss.main(args)
    _unit(d["ex"][1], "exec_records.jsonl", [], {"prepass": d["pre"][1]})
    _unit(d["tg"][1], "tags_records.jsonl", [], json.load(open(os.path.join(d["tg"][1], "run_config.json"))), code=rr.LEVEL_A_CODE)
    with pytest.raises(SystemExit, match="tags gate"):
        ss.main(args)
    assert opened == [] and not os.path.exists(tmp_path / "out")


# ------------------------------------------------------------------ the level-A readings end to end, and the command's order

K0, K4 = "R0|10|2", "R4|10|-"
CASES = (("ycb__bowl", 0, "test"), ("ycb__bowl", 1, "test"), ("handal__hammer_1", 0, "test"), ("handal__hammer_1", 1, "test"),
         ("ycb__pitcher", 8, "test"), ("ycb__pitcher", 3, "test"), ("hot3d__mug", 0, "validation"))


def _ep(success):
    return {"success": success, "irr": False, "give_up": False, "attempts": 1, "determined": True, "rows": [], "labels": []}


def _pview(o, idx, side, ok):
    return {"object": o, "idx": idx, "side": side, **{k: _ep(ok) for k in ("gg", "ggb", "c10", "c3", "oracle", "r4")}}


def _s2(o, idx, label, dropped=False):
    from robots.mujoco_lab.curve import GIVE_UP_ROW

    return {"object": o, "idx": idx, "side": "test", "dropped": dropped, "decisions": {K0: "r1", K4: "r1"},
            "sims": [{"seed_k": 0, "first": "r1", "label": label, "menu2": {"decisions": {K0: GIVE_UP_ROW, K4: GIVE_UP_ROW}}}]}


def _level_a():
    """Seven pilot cases: five paired test cases (pitcher:8 among them), one the execute pass stopped, one validation
    case; the published views lack bowl:1, the corrected ones hammer_1:1; oracle-c dropped bowl:0."""
    from robots.mujoco_lab import curve_graspgen as gg

    rp = {"d_mm": 5.0, "held": True, "lifted": True, "raise_mm": 5.0, "unreachable": False}
    exs = []
    for n, (o, i, side) in enumerate(CASES):
        ex = _ex(o=o, idx=i, side=side, com_f=0.4, length=0.3,
                 rows30=[_r30(0, 0, "placed_stable" if n % 2 else "tipped", kept=True, kept_b=True),
                         _r30(1, 2, "placed_stable" if n == 0 else "dropped_low", flagged=True)],
                 rows6=[_r6(2, 1, "rolled_off" if n == 1 else "placed_stable", replan=rp), _r6(3, 3, None, free="failed"), _r6(4, 4, None, g2=True)])
        if (o, i) == ("ycb__pitcher", 3):
            ex["exec_stop"] = "g0: the hulls after the reset differ from the record's"
        exs.append(_counts(ex))
    tgs = [_tg(e) for e in exs]
    grecs = [{"object": o, "idx": i, "side": side, "rows": [dict(t, gate=False) for t in tg["rows30"]]} for (o, i, side), tg in zip(CASES, tgs)]
    grecs[0]["rows"][1]["f"] = 0.6                               # one pilot tag differs from the tags pass
    gv = [_pview(o, i, side, True) for o, i, side in CASES if (o, i) not in (("ycb__bowl", 1), ("ycb__pitcher", 3))]
    cv = [_pview(o, i, side, i == 0) for o, i, side in CASES if (o, i) not in (("handal__hammer_1", 1), ("ycb__pitcher", 3))]
    cg = [{"object": e["object"], "idx": e["idx"], "side": e["side"],
           "rows": [{"id": r["id"], "label": r["run"]["label"], "kept": r["kept"], "kept_b": r["kept_b"], "seconds": 1.0} for r in e["rows30"]]}
          for e in exs]
    test = [c for c in CASES if c[2] == "test"]
    D = {"grecs": grecs, "k0": K0, "k4": K4, "s2c": [_s2(o, i, "placed_stable") for o, i, _ in test],
         "ocrecs": [_s2(o, i, "tipped", dropped=(o, i) == ("ycb__bowl", 0)) for o, i, _ in test],
         "side": {"counts": rr.smoke_report(exs, [], {}), "smoke": {"counts": {"replay_rows": 616, "replay_mismatches": 0, "free": {"miss": 0}}},
                  "g1": {"rows": 6565, "mismatches": 0, "passed": True}, "g1p": {"contacts": {"passed": True}, "replan": {"passed": True}}}}
    P = {"pub": {"pilot": {"test": gg.pilot_block([v for v in gv if v["side"] == "test"])}}, "gv": gv, "cv": cv, "cg": cg,
         "corr_test": gg.pilot_block([v for v in cv if v["side"] == "test"]), "g_pub": {"pilot_gg_vs_c10": []}, "g_corr": [], "stop": False}
    L = {"exs": exs, "tgs": tgs, "shas": {"x/exec_records.jsonl": "0" * 64}}
    gate = {"code": rr.LEVEL_A_CODE, "sim": rr.LEVEL_A_SIM, "execute_units": 2, "prepass_units": 2,
            "plan": {"code": rr.LEVEL_A_CODE, "sim": "e8", "sim_dir": "/w"}}
    return D, P, L, gate, {"code": ss.TAGS_CODE, "sim": rr.LEVEL_A_SIM, "tags_units": 2}


def test_the_level_a_build_puts_each_column_on_its_own_cases_and_the_extra_on_the_set_no_arm_drops(tmp_path):
    D, P, L, gate, tgate = _level_a()
    rep = ss.level_a_build(D, P, L, gate, tgate)
    R = rep["readings"]
    assert rep["counts"]["paired"] == {"test": 5, "validation": 1} and rep["counts"]["dropped"] == {"the execute pass stopped the case": 1}
    la = R["2_level_a"]
    assert la["cases"] == 5 and la["headline"]["A"]["cases"] == 5 and la["pitcher_8"]["in_the_set"]
    assert la["pitcher_8"]["headline"]["A"]["cases"] == 4
    assert R["2_published"]["cases"] == 4 and R["2_corrected"]["cases"] == 4 and la["extras_cases"] == {"level_a": 5, "with_corrected": 4}
    x = R["2_extra"]
    assert x["cases"] == 3 and x["exploratory"] and x["of_pilot_cases"] == 7
    assert x["n_per_column"] == {"level_a": 5, "corrected": 4, "oracle_c": 5, "r4_c": 6}      # each arm's own test cases
    assert x["rates"]["oracle_c"]["n"] == 3 and x["rates"]["r4_c"]["success"] == 1.0 and x["rates"]["oracle_c"]["success"] == 0.0
    b = R["2_by_object"]["ycb__bowl"]
    assert (b["level_a"]["n"], b["published"]["n"], b["corrected"]["n"]) == (2, 1, 2) and R["2_by_object"]["hot3d__mug"]["side"] == "validation"
    g = R["7_gates"]
    assert g["counts_check"] == [] and g["tags_vs_pilot"]["equal"] == 13 and g["tags_vs_pilot"]["differ"] == 1
    assert la["headline"]["A"]["success"]["diff"] == pytest.approx(0.6)
    assert la["extras"]["A_vs_30n"]["success"]["diff"] == pytest.approx(0.4)        # the flagged row places bowl:0
    assert la["extras"]["A_vs_c10"]["cases"] == 4 and la["extras"]["A_vs_c10"]["success"]["diff"] == pytest.approx(0.5)
    assert g["cache_misses"] == 0 and g["g2"]["45"] == {"planned": 14, "failed": 7, "share": 0.5} and isinstance(g["g2"]["30"], str)
    assert R["1_funnel"]["all"]["45"]["within"] == 21                     # the funnel keeps the stopped case's prepass rows
    assert R["ruling3"]["rows"] == {"compared": 6, "agree": 6, "differ": 0, "missing": 0}
    assert R["release_tilt"]["all"]["rows"] == 5 and R["replan_d"]["45"]["bins"]["3 to 10 mm"]["rows"] == 5
    json.dumps(rep)
    rep["provenance"] = {"bootstrap": "2000 object draws, seed 20261108, 95 % percentile interval"}     # summary_cmd adds it
    md = ss.summary_markdown(rep)
    assert "## Reading 2" in md and "headline" in md and "ycb__pitcher:8" in md and "exploratory" in md
    ss.figure(rep, str(tmp_path / "f.png"))
    assert os.path.getsize(tmp_path / "f.png") > 1000


def _cmd_args(d, tmp_path):
    return ["--execute", ",".join(d["ex"]), "--plan", d["plan"], "--tags", ",".join(d["tg"]), "--poseset-summary", "p",
            "--pilot-summary", "g", "--readings-summary", "r", "--contacts", "c", "--replan", "q", "--stage2c-summary", "s",
            "--oracle-c-summary", "o", "--counts", "n", "--smoke", "m", "--g1", "x", "--out", str(tmp_path / "out")]


def test_a_published_or_corrected_gate_that_fails_stops_before_any_level_a_record_opens(tmp_path, monkeypatch):
    d = _record(tmp_path)
    monkeypatch.setattr(ss, "_open_inputs", lambda a: {"prov": {}})
    monkeypatch.setattr(ss, "inputs_gate", lambda D: {"stop": True, "g_pub": {"pilot_gg_vs_c10": ["success.diff 0.1 vs 0.2"]}, "g_corr": []})
    monkeypatch.setattr(ss, "_open_level_a", lambda *a: (_ for _ in ()).throw(AssertionError("a level-A record opened")))
    with pytest.raises(SystemExit) as e:
        ss.main(_cmd_args(d, tmp_path))
    assert e.value.code == 3
    stop = json.load(open(tmp_path / "out" / "rerun_a_summary_stop.json"))
    assert stop["published_gate"]["pilot_gg_vs_c10"] and not os.path.exists(tmp_path / "out" / "rerun_a_summary.json")


def test_the_command_opens_the_level_a_records_once_after_both_gates_and_writes_the_summary(tmp_path, monkeypatch):
    d = _record(tmp_path)
    D, P, L, _, _ = _level_a()
    calls = []
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(ss, "_open_inputs", lambda a: dict(D, prov={"inputs_sha256": {}}))
    monkeypatch.setattr(ss, "inputs_gate", lambda D_: P)
    monkeypatch.setattr(ss, "_open_level_a", lambda ex, tg: calls.append((ex, tg)) or L)
    rep = ss.main(_cmd_args(d, tmp_path))
    assert calls == [(d["ex"], d["tg"])] and rep["readings"]["2_level_a"]["cases"] == 5
    out = tmp_path / "out"
    assert json.load(open(out / "rerun_a_summary.json"))["provenance"]["level_a_records_sha256"] == L["shas"]
    assert os.path.exists(out / "rerun_a_summary.md") and os.path.exists(out / "rerun_a_ceiling.png")
    with pytest.raises(SystemExit, match="a summary exists"):
        ss.main(_cmd_args(d, tmp_path))


def test_the_inputs_gate_stops_on_a_published_difference_or_a_corrected_one(monkeypatch):
    from robots.mujoco_lab import curve_graspgen as gg
    from robots.mujoco_lab import sixdof_readings as sr

    iv = {"diff": 0.1, "lo": 0.0, "hi": 0.2}
    test = {"cases": 2, "rates": {k: {"n": 2, "success": 0.5} for k in gg.ARMS},
            "gg_vs_c10": {"cases": 2, "success": dict(iv), "irr": dict(iv), "give_up": dict(iv)},
            "ggb_vs_c10": {"cases": 2, "success": dict(iv), "irr": dict(iv), "give_up": dict(iv)}}
    got = {"test": test, "paired": {"test": 2, "validation": 0}}
    for name, fn in (("pose_views", lambda *a: ([], {})), ("pilot_views", lambda *a: ([], {})), ("pose_reading", lambda *a: {}),
                     ("pilot_reading", lambda *a: got), ("correct_pilot", lambda r, p: r), ("correct_poseset", lambda r, p: r),
                     ("gate", lambda *a: {"pilot_gg_vs_c10": []})):
        monkeypatch.setattr(sr, name, fn)
    D = {"k0": K0, "k4": K4, "passes": {"replan": {}}, "precs": [], "pvrecs": [], "grecs": [], "s2t": [], "s2v": [], "orecs": [],
         "ovrecs": [], "psum": {}, "gsum": {}, "side": {"readings": {"replan": {"pilot": json.loads(json.dumps(got))}}}}
    assert ss.inputs_gate(D)["stop"] is False
    monkeypatch.setattr(sr, "gate", lambda *a: {"pilot_gg_vs_c10": ["success.diff 0.1 vs 0.2"]})
    assert ss.inputs_gate(D)["stop"] is True
    monkeypatch.setattr(sr, "gate", lambda *a: {"pilot_gg_vs_c10": []})
    want = D["side"]["readings"]["replan"]["pilot"]
    want["test"]["gg_vs_c10"]["success"]["lo"] = -0.1
    want["test"]["rates"]["c3"]["success"] = 0.25
    want["paired"]["validation"] = 1
    g = ss.inputs_gate(D)
    assert g["stop"] and g["g_corr"][0] == "gg_vs_c10: success.lo 0.0 vs -0.1" and g["g_corr"][1].startswith("rates.c3")
    assert g["g_corr"][2].startswith("paired") and len(g["g_corr"]) == 3


def test_the_level_a_loader_reads_each_named_file_once_with_its_sha256(tmp_path):
    from robots.mujoco_lab.curve import _sha

    ex = [_unit(tmp_path / f"ex{k}", "exec_records.jsonl", [{"object": "ycb__bowl", "idx": k}], {}) for k in (0, 1)]
    tg = [_unit(tmp_path / f"tg{k}", "tags_records.jsonl", [{"object": "ycb__bowl", "idx": k}], {}) for k in (0, 1)]
    L = ss._open_level_a(ex, tg)
    assert [e["idx"] for e in L["exs"]] == [0, 1] and len(L["tgs"]) == 2
    assert L["shas"][os.path.join(ex[1], "exec_records.jsonl")] == _sha(os.path.join(ex[1], "exec_records.jsonl")) and len(L["shas"]) == 4
    with pytest.raises(SystemExit, match="named twice"):
        ss._open_level_a([ex[0], ex[0]], tg)


def test_the_level_a_build_refuses_records_that_do_not_hold_the_pilots_cases_once_each():
    D, P, L, gate, tgate = _level_a()
    with pytest.raises(SystemExit, match="same cases"):
        ss.level_a_build(dict(D, grecs=D["grecs"][1:]), P, L, gate, tgate)
    with pytest.raises(SystemExit, match="same cases"):
        ss.level_a_build(D, P, dict(L, exs=L["exs"] + L["exs"][:1]), gate, tgate)


@pytest.mark.parametrize("what", ["exit", "metrics"])
def test_the_command_refuses_an_execute_or_plan_unit_that_did_not_finish_with_exit_0(tmp_path, monkeypatch, what):
    d = _record(tmp_path)
    monkeypatch.setattr(ss, "_open_inputs", lambda *a: (_ for _ in ()).throw(AssertionError("an input opened")))
    if what == "exit":
        open(os.path.join(d["ex"][1], "run.exit"), "w").write("EXIT 3\n")
        want = f"{d['ex'][1]}: not EXIT 0"
    else:
        os.remove(os.path.join(d["plan"], "run_metrics.json"))
        want = "has not finished"
    with pytest.raises(SystemExit, match=want):
        ss.main(_cmd_args(d, tmp_path))
    assert not os.path.exists(tmp_path / "out")
