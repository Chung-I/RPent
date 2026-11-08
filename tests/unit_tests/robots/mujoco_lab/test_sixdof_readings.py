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

"""The readings of the G1′ write-up (design 2026-11-08-sixdof-rerun-design.md, amendment 8, readings (i) to (iii)):
each stored row with its replay's values of the replay-gate fields, the ceilings and the drop recomputed by the
published rules, placed and damage counts per object and pass, and the changed rows by cause."""

import json

import pytest

from robots.mujoco_lab import curve_graspgen as gg
from robots.mujoco_lab import curve_poseset as cp
from robots.mujoco_lab import curve_sixdof as cs
from robots.mujoco_lab import sixdof_readings as sr


def _row(i, label, current=True, sim=True, **k):
    r = dict({"id": i, "feasible": True, "current": current, "label": label if sim else None, "artifact": None, "error": None,
              "damage": None, "drop_z": None, "loss_phase": None, "max_z": 0.2, "qpos_sha": f"q{i}", "trace_sha": f"t{i}"}, **k)
    if sim:
        r["seconds"] = 1.0
    return r


def _rep(i, stored, diffs=None, first=None, raise_mm=None):
    d = diffs or {}
    r = {"id": i, "mapping": None, "hand_path": None, "diffs": d, "seconds": 1.0, "same": not d, "stored_label": stored,
         "label": d.get("label", [stored, stored])[1], "error": None, "box_contacts": {"observed": True, "first": first, "by": {}}}
    if raise_mm is not None:
        r["replan"] = {"raise_mm": raise_mm}
    return r


PALM = {"phase": "lift", "t": 2.1, "box": "palm", "partner": "object", "mm": 1.2}


# ------------------------------------------------------------------ the stored rows with the replay's values

def test_a_replayed_row_takes_the_replays_value_of_each_differing_field_and_keeps_the_others():
    stored = [_row("across_0.5", "placed_stable")]
    rep = [_rep("across_0.5", "placed_stable", {"label": ["placed_stable", "crushed"], "max_z": [0.2, 0.19], "damage": [None, {"f_n": 41.0}]})]
    out = sr.correct_rows(stored, rep)
    assert out[0]["label"] == "crushed" and out[0]["max_z"] == 0.19 and out[0]["damage"] == {"f_n": 41.0}
    keep = ("id", "feasible", "current", "qpos_sha", "trace_sha", "seconds", "artifact")
    assert {k: out[0][k] for k in keep} == {k: stored[0][k] for k in keep}
    assert stored[0]["label"] == "placed_stable" and stored[0]["max_z"] == 0.2 and stored[0]["damage"] is None


def test_a_row_that_never_ran_keeps_its_record_and_takes_no_replay():
    stored = [_row("across_0.1", None, sim=False, feasible=False), _row("across_0.5", "bruised")]
    assert sr.correct_rows(stored, [_rep("across_0.5", "bruised")]) == stored


def test_a_simulated_row_without_its_replay_a_replay_without_its_row_or_a_row_replayed_twice_is_refused():
    stored = [_row("across_0.5", "bruised"), _row("across_0.3", "bruised")]
    with pytest.raises(ValueError, match="across_0.3"):
        sr.correct_rows(stored, [_rep("across_0.5", "bruised")])
    with pytest.raises(ValueError, match="along_0.5"):
        sr.correct_rows(stored, [_rep("across_0.5", "bruised"), _rep("across_0.3", "bruised"), _rep("along_0.5", "bruised")])
    with pytest.raises(ValueError, match="twice"):
        sr.correct_rows(stored, [_rep("across_0.5", "bruised"), _rep("across_0.3", "bruised"), _rep("across_0.3", "bruised")])


def test_a_replay_of_another_label_than_the_stored_rows_is_refused():
    with pytest.raises(ValueError, match="stored label"):
        sr.correct_rows([_row("across_0.5", "bruised")], [_rep("across_0.5", "placed_stable")])


# ------------------------------------------------------------------ the ceilings by the published rules

def _pose(rows, **k):
    return dict({"object": "ycb__mug", "idx": 3, "side": "test", "skipped": False, "error": None, "stop": None, "rows": rows,
                 "ceiling10": cp.ceiling(rows), "ceiling3": cp.ceiling(rows, current_only=True), "dropped": False}, **k)


def test_the_pose_sets_ceilings_and_drop_follow_the_corrected_labels():
    rows = [_row("across_0.5", "placed_stable"), _row("across_0.3", "not_lifted"), _row("along_0.1", "placed_stable", current=False)]
    rec = _pose(rows)
    crushed = _rep("across_0.5", "placed_stable", {"label": ["placed_stable", "crushed"]})
    out = sr.correct_poseset(rec, {"rows": [crushed, _rep("across_0.3", "not_lifted"), _rep("along_0.1", "placed_stable")]})
    assert out["ceiling10"] == cp.ceiling(out["rows"]) and out["ceiling10"]["success"] and out["ceiling10"]["rows"] == ["along_0.1"]
    assert out["ceiling3"] == cp.ceiling(out["rows"], current_only=True) and not out["ceiling3"]["success"]
    assert out["ceiling3"]["labels"] == ["not_lifted"] and out["dropped"] is False
    assert rec["ceiling3"]["success"] and rec["rows"][0]["label"] == "placed_stable"          # the record is not changed
    # an artifact on the only placing row leaves the 10-row ceiling undetermined: the case drops (poseset_case's rule)
    art = _rep("along_0.1", "placed_stable", {"artifact": [None, cp.OFF_CENTRE]})
    out = sr.correct_poseset(rec, {"rows": [crushed, _rep("across_0.3", "not_lifted"), art]})
    assert out["ceiling10"]["determined"] is False and out["ceiling3"]["determined"] is True and out["dropped"] is True
    # an error on a current row that did not place leaves only the 3-row ceiling undetermined: the case drops too
    err = _rep("across_0.3", "not_lifted", {"error": [None, "RuntimeError: x"]})
    out = sr.correct_poseset(rec, {"rows": [crushed, err, _rep("along_0.1", "placed_stable")]})
    assert out["ceiling10"]["determined"] is True and out["ceiling3"]["determined"] is False and out["dropped"] is True


def test_a_skipped_or_failed_pose_set_case_is_kept_as_published_and_a_simulated_one_needs_its_replay():
    for rec in (_pose([], skipped=True, dropped=True, ceiling10=None, ceiling3=None),
                _pose([], error="ValueError: x", dropped=True, ceiling10=None, ceiling3=None)):
        assert sr.correct_poseset(rec, None) == rec
    rec = _pose([_row("across_0.1", None, sim=False, feasible=False)])
    assert sr.correct_poseset(rec, {"rows": []}) == rec and sr.correct_poseset(rec, None) == rec
    with pytest.raises(ValueError, match="no replay"):
        sr.correct_poseset(_pose([_row("across_0.5", "bruised")]), None)


def _gg_row(i, label, rank, kept=True, kept_b=True, **k):
    return _row(i, label, kept=kept, kept_b=kept_b, rank=rank, **k)


def test_a_pilot_case_takes_both_rules_ceilings_from_its_corrected_kept_rows_and_drops_on_the_design_rule():
    rows = [_gg_row("g0", "placed_stable", 0), _gg_row("g1", "tipped", 1, kept=False),
            _gg_row("gate_across_0.5", "placed_stable", -1, kept=False, kept_b=False, gate=True)]
    rec = {"object": "ycb__bowl", "idx": 1, "side": "test", "skipped": False, "error": None, "stop": None, "rows": rows,
           "ceiling": gg.ceiling(rows, "kept"), "ceiling_b": gg.ceiling(rows, "kept_b"), "dropped": False}
    reps = [_rep("g0", "placed_stable", {"label": ["placed_stable", "dropped_high"]}), _rep("g1", "tipped", {"label": ["tipped", "placed_stable"]}),
            _rep("gate_across_0.5", "placed_stable", {"label": ["placed_stable", "crushed"]})]
    out = sr.correct_pilot(rec, {"rows": reps})
    assert out["ceiling"] == gg.ceiling(out["rows"], "kept") and not out["ceiling"]["success"] and out["ceiling"]["irr"]
    assert out["ceiling_b"]["success"] and out["ceiling_b"]["rows"] == ["g1"] and out["dropped"] is False
    assert out["rows"][2]["label"] == "crushed"                                          # the gate row is corrected too
    reps[0] = _rep("g0", "placed_stable", {"error": [None, "RuntimeError: x"], "label": ["placed_stable", None]})
    out = sr.correct_pilot(rec, {"rows": reps})
    assert out["ceiling"]["determined"] is False and out["dropped"] is True


# ------------------------------------------------------------------ placed and damage per object and pass, by cause

def test_object_counts_give_placed_and_damage_per_object_and_pass_with_the_hammers_and_all_on_their_own_lines():
    rows = [{"object": "handal__hammer_1", "labels": ["not_lifted", "placed_stable", "placed_stable"]},
            {"object": "handal__hammer_2", "labels": ["crushed", "crushed", "placed_stable"]},
            {"object": "ycb__bowl", "labels": ["placed_stable", "bruised", "bruised"]},
            {"object": "ycb__bowl", "labels": ["tipped", "tipped", "tipped"]}]
    out = sr.object_counts(rows)
    assert out["by_object"]["handal__hammer_1"] == {"rows": 1, "placed": [0, 1, 1], "damage": [0, 0, 0]}
    assert out["by_object"]["ycb__bowl"] == {"rows": 2, "placed": [1, 0, 0], "damage": [0, 1, 1]}
    assert out["hammers"] == {"rows": 2, "placed": [0, 1, 2], "damage": [1, 1, 0]}
    assert out["all"] == {"rows": 4, "placed": [1, 1, 2], "damage": [1, 2, 1]}
    assert list(out["by_object"]) == ["handal__hammer_1", "handal__hammer_2", "ycb__bowl"]


def test_row_cause_reads_the_first_box_contact_and_the_raise():
    assert cs.row_cause(_rep("a", "x", first=PALM)) == "contact only"
    assert cs.row_cause(_rep("a", "x", raise_mm=4.2)) == "raise only"
    assert cs.row_cause(_rep("a", "x", first=PALM, raise_mm=4.2)) == "both"
    assert cs.row_cause(_rep("a", "x", raise_mm=0.0)) is None and cs.row_cause(_rep("a", "x")) is None


def test_by_cause_counts_the_changed_rows_with_their_placed_and_damage_before_and_after():
    rows = [{"object": "o", "labels": ["placed_stable", "placed_stable", "crushed"], "changed": True, "cause": "both"},
            {"object": "o", "labels": ["tipped", "tipped", "placed_stable"], "changed": True, "cause": "raise only"},
            {"object": "o", "labels": ["dropped_low", "dropped_low", "placed_stable"], "changed": True, "cause": "raise only"},
            {"object": "o", "labels": ["placed_stable", "crushed", "placed_stable"], "changed": True, "cause": "contact only"},
            {"object": "o", "labels": ["bruised", "bruised", "bruised"], "changed": False, "cause": None}]
    out = sr.by_cause(rows)
    assert out["raise only"] == {"rows": 2, "placed": [0, 2], "damage": [0, 0],
                                 "transitions": {"tipped → placed_stable": 1, "dropped_low → placed_stable": 1}}
    assert out["both"] == {"rows": 1, "placed": [1, 0], "damage": [0, 1], "transitions": {"placed_stable → crushed": 1}}
    assert out["contact only"] == {"rows": 1, "placed": [1, 1], "damage": [0, 0], "transitions": {}}
    assert list(out) == list(cs.CAUSES)
    with pytest.raises(ValueError, match="no cause"):
        sr.by_cause([{"object": "o", "labels": ["a", "b"], "changed": True, "cause": None}])


def test_the_hammer_cases_by_object_count_the_cases_a_pose_placed():
    def v(o, labels):
        return {"object": o, "hammer": "hammer" in o, "rows": [_row(f"r{k}", x) for k, x in enumerate(labels)]}

    vs = [v("handal__hammer_1", ["not_lifted", "placed_stable"]), v("handal__hammer_1", ["tipped", "not_lifted"]),
          v("handal__hammer_7", ["not_lifted"]), v("ycb__bowl", ["placed_stable"])]
    out = sr.hammer_by_object(vs)
    assert list(out) == ["handal__hammer_1", "handal__hammer_7"]
    assert out["handal__hammer_1"] == {"cases": 2, "placed": 1, "classes": {"placed by a pose": 1, "lifted, never placed": 1}}
    assert out["handal__hammer_7"] == {"cases": 1, "placed": 0, "classes": {"not lifted by any pose": 1}}


# ------------------------------------------------------------------ the replays of one pass

def _unit(tmp_path, name, cases, cfg=None, exit_="EXIT 0"):
    d = tmp_path / name
    d.mkdir()
    (d / "g1_records.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
    (d / "run_config.json").write_text(json.dumps(dict({"box_contacts": True, "allow_diffs": True}, **(cfg or {}))))
    (d / "run.exit").write_text(exit_ + "\n")
    return d


def _g1case(o, idx, rows, side="test", stop=None, error=None):
    return {"source": "poseset", "side": side, "object": o, "idx": idx, "rows": rows, "stop": stop, "error": error}


def _queue(tmp_path, cases_by_unit, cfg=None, passed=True, exit_="EXIT 0"):
    q = tmp_path / "q"
    q.mkdir()
    for k, cases in enumerate(cases_by_unit):
        _unit(q, f"unit_{k}", cases, cfg=cfg)
    (q / "report").mkdir()
    (q / "report" / "g1p_summary.json").write_text(json.dumps({"passed": passed}))
    (q / "report" / "run.exit").write_text(exit_ + "\n")
    return str(q)


def test_a_pass_loads_its_replays_by_case_and_checks_the_flag(tmp_path):
    a, b = _g1case("ycb__mug", 0, [_rep("across_0.5", "bruised")]), _g1case("ycb__mug", 1, [], side="validation")
    q = _queue(tmp_path, [[a], [b]])
    got = sr.load_pass(q, replan=False)
    assert set(got) == {("poseset", "test", "ycb__mug", 0), ("poseset", "validation", "ycb__mug", 1)} and got[("poseset", "test", "ycb__mug", 0)] == a
    with pytest.raises(SystemExit, match="replan"):
        sr.load_pass(q, replan=True)


@pytest.mark.parametrize("bad, match", [("exit", "EXIT 0"), ("report", "did not pass"), ("report_exit", "did not pass"), ("twice", "twice"), ("stop", "stop"),
                                        ("error", "error"), ("contacts", "box contacts")])
def test_a_pass_with_a_failed_unit_a_failed_report_a_case_twice_a_stop_an_error_or_no_contacts_is_refused(tmp_path, bad, match):
    a = _g1case("ycb__mug", 0, [_rep("across_0.5", "bruised")], stop="x" if bad == "stop" else None,
                error="ValueError: y" if bad == "error" else None)
    units = [[a], [a]] if bad == "twice" else [[a]]
    q = _queue(tmp_path, units, cfg={"box_contacts": bad != "contacts"}, passed=bad != "report", exit_="EXIT 1" if bad == "report_exit" else "EXIT 0")
    if bad == "exit":
        (tmp_path / "q" / "unit_0" / "run.exit").write_text("EXIT 3\n")
    with pytest.raises(SystemExit, match=match):
        sr.load_pass(q, replan=False)


# ------------------------------------------------------------------ one line per replayed row

def test_the_row_table_has_one_line_per_simulated_row_with_each_pass_label_and_the_last_passs_change_and_cause():
    pose = {"object": "ycb__mug", "idx": 0, "side": "test", "rows": [_row("across_0.5", "placed_stable"), _row("across_0.1", None, sim=False)]}
    pilot = {"object": "ycb__mug", "idx": 0, "side": "test",
             "rows": [_gg_row("g0", "tipped", 0), _gg_row("gate_across_0.5", "placed_stable", -1, kept=False, kept_b=False, gate=True)]}
    c1 = {("poseset", "test", "ycb__mug", 0): {"rows": [_rep("across_0.5", "placed_stable", {"label": ["placed_stable", "crushed"]}, first=PALM)]},
          ("pilot", "test", "ycb__mug", 0): {"rows": [_rep("g0", "tipped"), _rep("gate_across_0.5", "placed_stable")]}}
    c2 = {("poseset", "test", "ycb__mug", 0): {"rows": [_rep("across_0.5", "placed_stable", raise_mm=6.0)]},
          ("pilot", "test", "ycb__mug", 0): {"rows": [_rep("g0", "tipped", {"label": ["tipped", "placed_stable"]}, raise_mm=40.0),
                                                      _rep("gate_across_0.5", "placed_stable")]}}
    rows = sr.row_table([pose], "poseset", [c1, c2]) + sr.row_table([pilot], "pilot", [c1, c2])
    assert [(r["source"], r["id"]) for r in rows] == [("poseset", "across_0.5"), ("pilot", "g0")]          # never-ran and gate rows out
    assert rows[0]["labels"] == ["placed_stable", "crushed", "placed_stable"] and rows[0]["changed"] is False
    assert rows[0]["cause"] == "raise only" and rows[0]["first"] == [PALM, None]
    assert rows[1]["labels"] == ["tipped", "tipped", "placed_stable"] and rows[1]["changed"] is True and rows[1]["cause"] == "raise only"
    with pytest.raises(ValueError, match="not replayed"):
        sr.row_table([pose], "poseset", [c1, {}])
    with pytest.raises(ValueError, match="not replayed"):
        sr.row_table([pose], "poseset", [c1, {("poseset", "test", "ycb__mug", 0): {"rows": []}}])


def test_the_hammer_rows_whose_label_changed_are_counted_per_pass_by_first_box_contact_and_transition():
    lift = {"phase": "lift", "box": "palm", "partner": "object", "mm": 2.0, "t": 2.0}
    rows = [{"object": "handal__hammer_1", "labels": ["not_lifted", "placed_stable", "placed_stable"], "first": [lift, lift], "cause": "contact only"},
            {"object": "handal__hammer_1", "labels": ["tipped", "tipped", "placed_stable"], "first": [None, None], "cause": "raise only"},
            {"object": "handal__hammer_7", "labels": ["not_lifted", "not_lifted", "not_lifted"], "first": [lift, lift], "cause": "contact only"},
            {"object": "ycb__bowl", "labels": ["placed_stable", "crushed", "crushed"], "first": [lift, lift], "cause": "contact only"}]
    out = sr.hammer_label_changes(rows)
    assert out["contacts"] == {"rows": 1, "first_contact": {"lift palm/object": 1}, "transitions": {"not_lifted → placed_stable": 1}}
    assert out["replan"] == {"rows": 2, "first_contact": {"lift palm/object": 1, "none": 1},
                             "transitions": {"not_lifted → placed_stable": 1, "tipped → placed_stable": 1},
                             "by_cause": {"contact only": 1, "raise only": 1}}


def _ep(s, irr=False, give_up=False):
    return {"success": s, "irr": irr, "give_up": give_up, "attempts": 0 if give_up else 1}


def test_the_markdown_puts_each_pass_in_its_column_with_the_audits_line_beside_the_published():
    from robots.mujoco_lab.curve_oracle import paired_outcomes

    one = paired_outcomes([("a", _ep(True), _ep(False)), ("b", _ep(True), _ep(True))])
    two = paired_outcomes([("a", _ep(False), _ep(False)), ("b", _ep(True), _ep(True), )])

    def a_pass(x, placed):
        return {"poseset": {"views": {v: x for v in sr.VIEWS}, "hammer_classes": {c: 0 for c in cp.HAMMER_CLASSES},
                            "hammer_by_object": {"handal__hammer_1": {"cases": 2, "placed": placed, "classes": {}}}, "paired": 2,
                            "paired_dropped": {}, "pose_dropped": {"test": 0, "validation": 0}},
                "pilot": {"test": {"cases": 2, "objects": 2, "rates": {k: x["rate_a"] for k in gg.ARMS}, "gg_vs_c10": x, "ggb_vs_c10": x,
                                   "rule": "r", "rule_b": "rb"},
                          "hammer_classes": {n: {c: 0 for c in gg.HAMMER_ROW_CLASSES} for n, _ in gg.RULES},
                          "hammer_c10_classes": {c: 0 for c in cp.HAMMER_CLASSES}, "paired": {"test": 2, "validation": 0},
                          "paired_dropped": {}, "pilot_dropped": 0}}

    rows = [{"object": "handal__hammer_1", "labels": ["not_lifted", "placed_stable", "placed_stable"], "changed": True,
             "cause": "contact only", "first": [PALM, PALM]}]
    rep = {"published": a_pass(one, 0), "contacts": a_pass(two, 1), "replan": a_pass(two, 1), "gate": {"x": []},
           "audit": {"all": one, "upright": one}, "counts": {s: sr.object_counts(rows) for s in ("poseset", "pilot")},
           "by_cause": {s: {"all": sr.by_cause(rows), "hammers": sr.by_cause(rows)} for s in ("poseset", "pilot")},
           "hammer_label_changes": {s: sr.hammer_label_changes(rows) for s in ("poseset", "pilot")},
           "provenance": {"bootstrap": "2000 object draws", "audit_level": "gt1"}}
    md = sr.readings_markdown(rep)
    assert "- The published numbers recomputed from the stored records: all equal." in md
    c1, c2 = sr._ci(one["success"]), sr._ci(two["success"])
    assert f"| all | 10-row − 3-row, success | {c1} | {c1} | {c2} | {c2} |" in md
    assert f"| lying | 10-row − 3-row, success | {c1} | – | {c2} | {c2} |" in md
    assert "| handal__hammer_1 | 0 / 2 | 1 / 2 | 1 / 2 |" in md
    assert "| **hammers** | 1 | 0 → 1 → 1 | 0 → 0 → 0 |" in md
    assert "| contact only | 1 | 0 → 1 | 0 → 0 | not_lifted → placed_stable 1 |" in md
    assert f"| GraspGen (the design's rule) − 10-pinch, success | {c1} | {c2} | {c2} |" in md
