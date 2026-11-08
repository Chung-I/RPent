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

"""The sixdof rerun's driver (design 2026-11-08-sixdof-rerun-design.md): gate G1 replays every simulated row of the
GraspGen pilot and of the pose set through the new executor path and compares it with its record."""

import copy
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from robots.mujoco_lab import curve_graspgen as gg
from robots.mujoco_lab import curve_poseset as cp
from robots.mujoco_lab import curve_sixdof as cs
from robots.mujoco_lab.adapter import Proposal
from tests.unit_tests.robots.mujoco_lab.test_curve_graspgen import BASE, CaseEnv, SimEnv, _box, _entry, _rec1


class G1Env(CaseEnv):
    """CaseEnv with today's hand path (the proposal's point and yaw at three steps) and ``hand_calls``."""

    def __init__(self, labels, **kw):
        super().__init__(labels, **kw)
        self.hand_calls = []

    def hand_path(self, proposal, v_lift, v_transport, destination, release_height=None, a_transport=2.0):
        self.hand_calls.append((round(float(proposal.point[0]), 4), v_lift, v_transport, destination, release_height, a_transport))
        return [(ph, np.asarray(proposal.point, dtype=float), float(proposal.yaw)) for ph in ("wait", "approach", "close")]


LABELS = {-0.1: "tipped", 0.0003: "placed_stable", 0.1: "placed_stable", 0.0: "placed_stable", -0.12: "tipped", -0.06: "tipped",
          0.06: "placed_stable", 0.12: "dropped_high"}


@pytest.fixture
def lab(monkeypatch):
    from robots.mujoco_lab import adapter, calibrate, com_gap, evpi_sim

    monkeypatch.setattr(calibrate, "case_draw", lambda seed, o, idx, m, flag, fragility=None:
                        ({"mu": 0.5}, 1, "bin", {"fill": 0.0, "frag": None}))
    monkeypatch.setattr(adapter, "object_props", lambda o: {"mass": 1.0})
    monkeypatch.setattr(com_gap, "fingerprint", lambda env: "fp")
    monkeypatch.setattr(evpi_sim, "seeds_of", lambda seed, o, idx: [seed * 1000 + idx, 777, 778])
    return G1Env(LABELS)


def _stored(rec):
    """A record as its run wrote it and the summary reads it: through JSON."""
    return json.loads(json.dumps(rec))


def _pilot(lab):
    entry = _entry(lab)
    rec = _stored(gg.graspgen_case(SimpleNamespace(env=lab), _rec1(), 3, None, entry))
    lab.calls.clear(), lab.resets.clear()
    return rec, entry


# ------------------------------------------------------------------ one row

def test_a_row_runs_after_its_own_reset_and_checks_through_the_new_path():
    env, order = SimEnv([_box(0.30, 0.04, 0.03)], {0.1: "placed_stable"}), []
    p = Proposal(s=0.5, point=[0.1, 0.0, 0.0], yaw=math.pi / 2, width=0.04)

    def make(e):
        order.append("make")
        return p, None

    new, stop = cs.run_row(env, lambda: order.append("reset") or True, make, BASE, lambda e: order.append("fp") or "fp", "fp")
    assert stop is None and order == ["reset", "fp", "make"]
    assert env.calls == [(0.1, math.pi / 2, 12.0, 0.15, 0.5, "bin", None, 2.0)]
    assert new["label"] == "placed_stable" and new["artifact"] is False and new["error"] is None and new["seconds"] >= 0
    assert new["trace_sha"] == cp.trace_sha([{"phase": "lift", "z": 0.1, "fz_joint": 3.0}]) and new["qpos_sha"] is None


def test_a_row_has_the_pilots_outcome_fields():
    env = SimEnv([_box(0.30, 0.04, 0.03)], {0.1: "placed_stable"})
    r = {"id": "g0", "kept": True, "kept_b": True, "f": 0.5, "point": [0.1, 0.0, 0.0], "yaw": math.pi / 2, "width": 0.04}
    gg.simulate_rows(env, lambda: False, [r], BASE, lambda e: "fp", "fp")
    new, _ = cs.run_row(env, lambda: False, lambda e: (gg._proposal(r), None), BASE, lambda e: "fp", "fp")
    assert {k: new.get(k) for k in cp.OUTCOME if k != "seconds"} == {k: r.get(k) for k in cp.OUTCOME if k != "seconds"}
    assert new["artifact"] is True                                   # the reset was not at rest


def test_a_row_stops_on_a_fingerprint_a_hull_change_or_a_pinch_it_cannot_make():
    env = SimEnv([_box(0.30, 0.04, 0.03)], {0.1: "placed_stable"})
    p = Proposal(s=0.5, point=[0.1, 0.0, 0.0], yaw=0.0, width=0.04)
    assert "fingerprint" in cs.run_row(env, lambda: True, lambda e: (p, None), BASE, lambda e: "x", "fp")[1]
    assert "hulls" in cs.run_row(env, lambda: True, lambda e: (p, None), BASE, lambda e: "fp", "fp", hull_sha="0" * 16)[1]
    assert cs.run_row(env, lambda: True, lambda e: (None, "moved"), BASE, lambda e: "fp", "fp")[1] == "moved"
    assert env.calls == []


def test_an_exception_in_the_rollout_is_the_rows_error():
    env = SimEnv([_box(0.30, 0.04, 0.03)], {0.1: "placed_stable"}, fail={0.1})
    p = Proposal(s=0.5, point=[0.1, 0.0, 0.0], yaw=0.0, width=0.04)
    new, stop = cs.run_row(env, lambda: True, lambda e: (p, None), BASE, lambda e: "fp", "fp")
    assert stop is None and new["error"].startswith("RuntimeError") and new["label"] is None


def test_the_fields_compare_as_their_json():
    a = {"label": "tipped", "damage": [1, 2.5], "drop_z": float("nan"), "max_z": 0.2, "qpos_sha": "q", "trace_sha": "t"}
    assert cs.same_fields(a, dict(a, damage=(1, 2.5), drop_z=float("nan"))) == {}
    assert cs.same_fields(a, dict(a, trace_sha="u")) == {"trace_sha": ["t", "u"]}
    assert cs.same_fields(a, {k: v for k, v in a.items() if k != "max_z"}) == {"max_z": [0.2, None]}


# ------------------------------------------------------------------ one case

def test_every_pilot_row_replays_through_the_new_path_and_matches(lab):
    rec, entry = _pilot(lab)
    out = cs.g1_case(SimpleNamespace(env=lab), _rec1(), 3, None, rec, "pilot", entry)
    assert out["stop"] is None and out["error"] is None
    assert [r["id"] for r in out["rows"]] == ["g0", "g1", "g2", "gate_across_0.5"]
    assert all(r["same"] and r["diffs"] == {} for r in out["rows"])
    assert [r["mapping"] for r in out["rows"]] == [True, True, True, None]
    assert [r["hand_path"] for r in out["rows"]] == [True, True, True, None]
    assert [c[0] for c in lab.calls] == [-0.1, 0.0003, 0.1, 0.0] and {c[2] for c in lab.calls} == {14.7}
    assert lab.resets == [3003] * 8                                  # the case's reset, then the hand path's and the row's
    assert lab.hand_calls == [(x, 0.15, 0.5, "bin", None, 2.0) for x in (-0.1, -0.1, 0.0003, 0.0003, 0.1, 0.1)]


def test_a_pilot_row_whose_record_differs_is_a_mismatch(lab):
    rec, entry = _pilot(lab)
    rec["rows"][1]["trace_sha"] = "0" * 16
    rec["rows"][0]["yaw"] += 1e-12                                   # its grasp no longer maps to its record's yaw
    out = cs.g1_case(SimpleNamespace(env=lab), _rec1(), 3, None, rec, "pilot", entry)
    g0, g1 = out["rows"][0], out["rows"][1]
    assert not g0["same"] and g0["mapping"] is False and g0["diffs"] == {}
    assert not g1["same"] and list(g1["diffs"]) == ["trace_sha"] and out["rows"][2]["same"]


def test_a_replayed_row_keeps_its_stored_label_its_new_label_and_its_error(lab):
    rec, entry = _pilot(lab)
    rec["rows"][1]["label"] = "tipped"                               # the record's; the replay places it
    rec["rows"][1]["error"] = "stale"
    out = cs.g1_case(SimpleNamespace(env=lab), _rec1(), 3, None, rec, "pilot", entry)
    g1 = out["rows"][1]
    assert (g1["stored_label"], g1["label"], g1["error"]) == ("tipped", "placed_stable", None)
    assert g1["diffs"]["label"] == ["tipped", "placed_stable"] and not g1["same"]
    assert all(r["stored_label"] == r["label"] for r in out["rows"] if r["id"] != "g1")


def test_a_pilot_case_stops_on_a_base_a_fingerprint_or_hulls_that_are_not_the_records(lab):
    rec, entry = _pilot(lab)
    bad = copy.deepcopy(rec)
    bad["base"]["force"] = 12.0
    assert "base" in cs.g1_case(SimpleNamespace(env=lab), _rec1(), 3, None, bad, "pilot", entry)["stop"]
    assert "fingerprint" in cs.g1_case(SimpleNamespace(env=lab), _rec1(fingerprint="other"), 3, None, rec, "pilot", entry)["stop"]
    bad = dict(copy.deepcopy(rec), hull_sha="0" * 16)
    assert "hulls" in cs.g1_case(SimpleNamespace(env=lab), _rec1(), 3, None, bad, "pilot", entry)["stop"]
    assert lab.calls == []


def test_every_pose_set_row_replays_with_its_proposal_fetched_after_the_reset(lab):
    rec = _stored(cp.poseset_case(SimpleNamespace(env=lab), _rec1(), 3, None))
    sim = [r["id"] for r in rec["rows"] if "seconds" in r]
    lab.calls.clear(), lab.resets.clear()
    out = cs.g1_case(SimpleNamespace(env=lab), _rec1(), 3, None, rec, "poseset")
    assert out["stop"] is None and [r["id"] for r in out["rows"]] == sim and len(sim) >= 3
    assert all(r["same"] and r["mapping"] is None and r["hand_path"] is None for r in out["rows"])
    assert lab.resets == [3003] * (1 + len(sim)) and lab.hand_calls == []
    moved = copy.deepcopy(rec)
    next(r for r in moved["rows"] if "seconds" in r)["point"][0] += 0.001
    assert "pinch" in cs.g1_case(SimpleNamespace(env=lab), _rec1(), 3, None, moved, "poseset")["stop"]


def test_a_case_without_simulated_rows_replays_nothing(lab):
    rec = _stored(cp.poseset_case(SimpleNamespace(env=lab), _rec1(status="error"), 3, None))
    out = cs.g1_case(SimpleNamespace(env=lab), _rec1(status="error"), 3, None, rec, "poseset")
    assert out["rows"] == [] and out["stop"] is None and lab.resets == []


def test_the_metrics_count_rows_mismatches_and_stops_by_source_and_hold_no_label():
    rows = [{"id": "g0", "same": True, "diffs": {}, "mapping": True, "hand_path": True, "seconds": 1.0},
            {"id": "g1", "same": False, "diffs": {"label": ["tipped", "placed_stable"], "trace_sha": ["a", "b"]}, "mapping": True,
             "hand_path": True, "seconds": 1.0}]
    done = [{"source": "pilot", "side": "test", "rows": rows, "stop": None, "error": None, "seconds": 3.0},
            {"source": "poseset", "side": "validation", "rows": rows[:1], "stop": "x", "error": None, "seconds": 1.0}]
    m = cs.g1_metrics(done)
    assert m["rows"] == 3 and m["mismatches"] == 1 and m["stops"] == 1 and m["passed"] is False
    assert m["by_field"] == {"label": 1, "trace_sha": 1} and m["mapping_failures"] == 0 and m["hand_path_failures"] == 0
    assert m["by_source"]["pilot"]["test"] == {"cases": 1, "rows": 2, "mismatches": 1}
    assert "tipped" not in json.dumps(m) and "placed_stable" not in json.dumps(m)
    assert cs.g1_metrics([dict(done[0], rows=rows[:1])])["passed"] is True and cs.g1_metrics([])["passed"] is False


# ------------------------------------------------------------------ the report over G1's units

def _unit(tmp_path, k, cases, rows_of, exit_="EXIT 0"):
    d = tmp_path / f"unit_{k}"
    d.mkdir()
    (d / "g1_records.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
    (d / "run_config.json").write_text(json.dumps({"source": cases[0]["source"], "rows": rows_of, "code_revision": "abc"}))
    (d / "run.exit").write_text(exit_ + "\n")
    return str(d)


def _case(src, side, o, idx, n, bad=0, stop=None):
    rows = [{"id": f"g{i}", "same": i >= bad, "diffs": {} if i >= bad else {"trace_sha": ["a", "b"]}, "mapping": True,
             "hand_path": True, "seconds": 1.0} for i in range(n)]
    return {"source": src, "side": side, "object": o, "idx": idx, "rows": rows, "stop": stop, "error": None, "seconds": n + 1.0}


def test_the_report_adds_the_units_and_passes_only_when_every_stored_row_matched(tmp_path):
    units = [_unit(tmp_path, 0, [_case("pilot", "test", "a", 0, 3), _case("pilot", "test", "a", 1, 2)], 5),
             _unit(tmp_path, 1, [_case("poseset", "validation", "b", 0, 4)], 4)]
    rep = cs.g1_report(units, expected={"pilot": 5, "poseset": 4})
    assert rep["passed"] and rep["rows"] == 9 and rep["mismatches"] == 0 and rep["units"] == 2
    assert rep["of_rows"] == {"pilot": 5, "poseset": 4} and rep["missing_rows"] == {"pilot": 0, "poseset": 0}
    assert rep["code_revisions"] == ["abc"]
    rep = cs.g1_report(units, expected={"pilot": 6, "poseset": 4})
    assert not rep["passed"] and rep["missing_rows"]["pilot"] == 1


def test_the_report_fails_on_a_mismatch_a_stop_or_a_unit_that_did_not_end_with_exit_0(tmp_path):
    a = _unit(tmp_path, 0, [_case("pilot", "test", "a", 0, 3, bad=1)], 3, exit_="EXIT 3")
    b = _unit(tmp_path, 1, [_case("poseset", "test", "b", 0, 2, stop="g1: the hulls after the reset differ from the record's")], 2)
    rep = cs.g1_report([a, b], expected={"pilot": 3, "poseset": 2})
    assert not rep["passed"] and rep["mismatches"] == 1 and rep["stops"] == 1 and rep["units_not_exit_0"] == [a]
    assert rep["mismatched_rows"] == [{"source": "pilot", "side": "test", "object": "a", "idx": 0, "id": "g0", "fields": ["trace_sha"]}]
    assert rep["stopped_cases"] == [{"source": "poseset", "side": "test", "object": "b", "idx": 0,
                                     "stop": "g1: the hulls after the reset differ from the record's"}]


def test_the_report_command_writes_the_summary_and_exits_3_when_g1_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(cs, "G1_EXPECTED", {"pilot": 3, "poseset": 0})
    q = tmp_path / "q"
    q.mkdir()
    _unit(q, 0, [_case("pilot", "test", "a", 0, 3)], 3)
    rep = cs.main(["g1-report", "--queue", str(q), "--out", str(tmp_path / "s")])
    md = (tmp_path / "s" / "g1_summary.md").read_text()
    assert rep["passed"] and "Result: PASSED." in md and "| pilot | test | 1 | 3 | 0 |" in md
    assert json.loads((tmp_path / "s" / "g1_summary.json").read_text())["rows"] == 3
    _unit(q, 1, [_case("pilot", "test", "b", 0, 1, bad=1)], 1)
    with pytest.raises(SystemExit) as e:
        cs.main(["g1-report", "--queue", str(q), "--out", str(tmp_path / "s2")])
    assert e.value.code == 3 and "Result: FAILED." in (tmp_path / "s2" / "g1_summary.md").read_text()


# ------------------------------------------------------------------ gate G1′ (amendment 8: contacts on)

def test_a_g1_prime_unit_fails_only_on_a_stop_an_error_or_a_geometry_check_while_rows_may_differ():
    m = {"rows": 3, "mismatches": 2, "stops": 0, "case_errors": 0, "mapping_failures": 0, "hand_path_failures": 0, "passed": False}
    assert cs.g1_unit_ok(m, allow_diffs=True) and not cs.g1_unit_ok(m, allow_diffs=False)
    assert cs.g1_unit_ok(dict(m, mismatches=0, passed=True), allow_diffs=False)
    for k in ("stops", "case_errors", "mapping_failures", "hand_path_failures"):
        assert not cs.g1_unit_ok(dict(m, **{k: 1}), allow_diffs=True)
    assert not cs.g1_unit_ok(dict(m, rows=0), allow_diffs=True)


def _row(i, stored="placed_stable", label="placed_stable", diffs=None, error=None, mapping=None):
    return {"id": i, "same": not diffs and mapping is not False, "diffs": diffs or {}, "mapping": mapping, "hand_path": None,
            "seconds": 1.0, "stored_label": stored, "label": label, "error": error}


def _g1p_units(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    a = {"source": "poseset", "side": "test", "object": "box", "idx": 0, "stop": None, "error": None, "seconds": 4.0,
         "rows": [_row("across_0.1"), _row("across_0.5", label="not_lifted", diffs={"label": ["placed_stable", "not_lifted"]}),
                  _row("across_0.7")]}
    b = {"source": "pilot", "side": "validation", "object": "mug", "idx": 2, "stop": None, "error": None, "seconds": 2.0,
         "rows": [_row("g0", label=None, error="Unreachable: free move: palm - fx_shelf_0",
                       diffs={"label": ["tipped", None], "error": [None, "Unreachable: free move: palm - fx_shelf_0"]}, stored="tipped")]}
    return [_unit(tmp_path, 0, [a], 3), _unit(tmp_path, 1, [b], 1)]


def test_g1_prime_sorts_the_rows_by_the_audits_list_and_names_every_changed_row_the_list_lacks(tmp_path):
    flagged = {("poseset", "test", "box", 0, "across_0.5"), ("poseset", "test", "box", 0, "across_0.7"), ("poseset", "test", "box", 0, "along_0.9")}
    rep = cs.g1p_report(_g1p_units(tmp_path), flagged, {"pilot": 1, "poseset": 3})
    assert rep["unflagged"] == {"rows": 2, "identical": 1, "changed": 1} and rep["flagged"] == {"rows": 2, "changed": 1, "identical": 1}
    assert rep["flagged_by_transition"] == {"placed_stable → not_lifted": 1} and rep["flagged_by_object"] == {"box": 1}
    assert rep["changed_not_in_list"] == [{"source": "pilot", "side": "validation", "object": "mug", "idx": 2, "id": "g0",
                                           "fields": ["error", "label"], "stored_label": "tipped", "label": None,
                                           "error": "Unreachable: free move: palm - fx_shelf_0"}]
    assert rep["flagged_not_replayed"] == [["poseset", "test", "box", 0, "along_0.9"]] and rep["passed"] is False
    flagged.add(("pilot", "validation", "mug", 2, "g0"))
    rep = cs.g1p_report(_g1p_units(tmp_path / "x"), flagged, {"pilot": 1, "poseset": 3})
    assert rep["passed"] is True and rep["unflagged"] == {"rows": 1, "identical": 1, "changed": 0}
    assert rep["flagged_by_transition"] == {"placed_stable → not_lifted": 1, "tipped → None": 1}
    assert not cs.g1p_report(_g1p_units(tmp_path / "y"), flagged, {"pilot": 2, "poseset": 3})["passed"]      # a row was not replayed


def test_the_g1_command_exits_0_on_differing_rows_with_allow_diffs_and_3_without(tmp_path, monkeypatch):
    import robots.mujoco_lab.curve as cv

    monkeypatch.setenv("WANDB_MODE", "disabled")
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_metrics.json").write_text("{}")
    (run / "run_config.json").write_text(json.dumps({"stage1": "s1", "seed": 0, "bundle": str(tmp_path / "b")}))
    (run / "records_test.jsonl").write_text(json.dumps({"object": "box", "idx": 0, "rows": [{"id": "a", "seconds": 1.0}]}) + "\n")
    monkeypatch.setattr(cv, "load_stage1", lambda dirs: ([{"object": "box", "idx": 0, "fold": 0}], {}))
    case = {"source": "poseset", "side": "test", "object": "box", "idx": 0, "stop": None, "error": None, "seconds": 2.0,
            "rows": [_row("a", label="not_lifted", diffs={"label": ["placed_stable", "not_lifted"]})]}
    monkeypatch.setattr(cs, "g1_object", lambda job: iter([case]))
    args = ["g1", "--source", "poseset", "--runs", str(run), "--sides", "test"]
    assert cs.main(args + ["--allow-diffs", "--out", str(tmp_path / "o1")]) == str(tmp_path / "o1")
    with pytest.raises(SystemExit) as e:
        cs.main(args + ["--out", str(tmp_path / "o2")])
    assert e.value.code == 3


def test_g1_prime_fails_on_a_stop_a_unit_without_exit_0_or_a_geometry_check_even_when_the_list_holds_every_changed_row(tmp_path):
    flagged = {("poseset", "test", "box", 0, "across_0.5"), ("pilot", "validation", "mug", 2, "g0")}
    units = _g1p_units(tmp_path / "a")
    assert cs.g1p_report(units, flagged, {"pilot": 1, "poseset": 3})["passed"]
    stop = {"source": "poseset", "side": "test", "object": "can", "idx": 1, "stop": "g1: the hulls after the reset differ from the record's",
            "error": None, "seconds": 1.0, "rows": []}
    rep = cs.g1p_report(units + [_unit(tmp_path / "a", 2, [stop], 0)], flagged, {"pilot": 1, "poseset": 3})
    assert not rep["passed"] and rep["stops"] == 1 and rep["missing_rows"] == {"pilot": 0, "poseset": 0}
    units = _g1p_units(tmp_path / "b")
    (tmp_path / "b" / "unit_1" / "run.exit").write_text("EXIT 1\n")
    rep = cs.g1p_report(units, flagged, {"pilot": 1, "poseset": 3})
    assert not rep["passed"] and rep["units_not_exit_0"] == [units[1]]
    geo = {"source": "pilot", "side": "test", "object": "can", "idx": 4, "stop": None, "error": None, "seconds": 1.0,
           "rows": [_row("g3", mapping=False)]}
    rep = cs.g1p_report(_g1p_units(tmp_path / "c") + [_unit(tmp_path / "c", 2, [geo], 1)], flagged, {"pilot": 2, "poseset": 3})
    assert not rep["passed"] and rep["mapping_failures"] == 1
    assert [x["fields"] for x in rep["changed_not_in_list"]] == [["mapping"]]


def test_the_g1_prime_report_command_reads_the_audits_list_and_exits_3_when_an_unflagged_row_changed(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(cs, "G1_EXPECTED", {"pilot": 1, "poseset": 3})
    q = tmp_path / "q"
    q.mkdir()
    _g1p_units(q)
    lst = tmp_path / "flagged_rows.jsonl"
    lines = [{"source": "poseset", "side": "test", "object": "box", "idx": 0, "id": i, "label": "placed_stable", "depth_mm": 3.0}
             for i in ("across_0.5", "across_0.7")]
    lines.append({"source": "stage2", "side": "test", "object": "box", "idx": 0, "first": "a", "attempt": 1, "row": "a", "depth_mm": 2.0})
    lst.write_text("".join(json.dumps(x) + "\n" for x in lines))
    with pytest.raises(SystemExit) as e:
        cs.main(["g1p-report", "--queue", str(q), "--flagged", str(lst), "--out", str(tmp_path / "s")])
    md = (tmp_path / "s" / "g1p_summary.md").read_text()
    assert e.value.code == 3 and "Result: FAILED." in md and "mug" in md and "free move" in md
    lst.write_text(lst.read_text() + json.dumps({"source": "pilot", "side": "validation", "object": "mug", "idx": 2, "id": "g0"}) + "\n")
    rep = cs.main(["g1p-report", "--queue", str(q), "--flagged", str(lst), "--out", str(tmp_path / "s2")])
    assert rep["passed"] and "Result: PASSED." in (tmp_path / "s2" / "g1p_summary.md").read_text()
    assert json.loads((tmp_path / "s2" / "g1p_summary.json").read_text())["unflagged"]["identical"] == 1
