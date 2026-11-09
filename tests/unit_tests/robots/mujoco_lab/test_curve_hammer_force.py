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

"""The hammer squeeze x CoM-pinch ceiling (daily-logs design 2026-11-09-hammer-force-com-design.md): two across pinches
(the menu's mid pinch and the pinch at the true balance fraction) at four squeezes per finger, each row run once at
the case's own reset seed; G0 (the control row replays the corrected pose-set record) and G1 (a second reset seed)."""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from robots.mujoco_lab import curve_hammer_force as hf
from rpent.skills.pick_place import features

BASE = {"force": 12.0, "v_lift": 0.15, "v_transport": 0.5, "a_transport": 2.0, "destination": "bin", "release_height": None}


def _box(l, w, h):
    return np.array([[x, y, z] for x in (-l / 2, l / 2) for y in (-w / 2, w / 2) for z in (-h / 2, h / 2)])


class FakeEnv:
    """A box lying along x (length L, width W): the across pinch at fraction f sits at x = -L/2 + f L and closes along y
    (yaw pi/2). ``narrow``: fractions at which the pinch is wider than the hand (a width refusal); ``unreachable``:
    fractions reach() refuses."""

    max_pinch_width = 0.09

    def __init__(self, L=0.3, W=0.03, narrow=(), unreachable=()):
        self.hulls, self.L, self.W, self.x0 = [_box(L, W, 0.03)], L, W, 0.0
        self.narrow, self.unreachable, self.reach_calls = set(narrow), set(unreachable), []

    def proposals(self, fractions=(0.3, 0.5, 0.7), turn=0.0):
        out = []
        for f in fractions:
            p = np.array([self.x0 - self.L / 2 + f * self.L, 0.0, 0.0])
            yaw = np.pi / 2 + float(turn)
            w = 0.2 if f in self.narrow else features.pinch_width(self.hulls, p, yaw)
            out.append(SimpleNamespace(s=float(f), point=[float(x) for x in p], yaw=yaw, width=w))
        return out

    def reach(self, p, destination):
        self.reach_calls.append(p.s)
        return {"reason": "ik", "phase": "transport"} if p.s in self.unreachable else None


def _trace(hold_fn=(30.0, 30.0), hold_pads=((True, True), (True, True)), close_fn=(0.0, 10.0, 20.0, 24.0)):
    t = [{"phase": "approach", "fn": 0.0, "pads": [False, False]}]
    t += [{"phase": "close", "fn": f, "pads": [f > 0, f > 0]} for f in close_fn]
    t += [{"phase": "lift", "fn": 20.0, "pads": [True, True]}]
    t += [{"phase": "hold", "fn": f, "pads": list(p)} for f, p in zip(hold_fn, hold_pads)]
    return t + [{"phase": "transport", "fn": 99.0, "pads": [True, True]}]


class SimEnv(FakeEnv):
    """FakeEnv with execute: ``labels`` by (fraction, force), the trace's hold reads the force; ``fail``: (fraction,
    force) pairs that raise."""

    def __init__(self, labels=None, fail=(), **kw):
        super().__init__(**kw)
        self.labels, self.fail, self.calls = labels or {}, set(fail), []
        self.sim = SimpleNamespace(d=None)

    def execute(self, proposal, force, v_lift, v_transport, destination, release_height, a_transport):
        self.calls.append((proposal.s, force, v_lift, v_transport, destination, release_height, a_transport))
        if (proposal.s, force) in self.fail:
            raise RuntimeError("the sim fell over")
        return SimpleNamespace(label=self.labels.get((proposal.s, force), "not_lifted"), artifact=False, damage=None,
                               drop_z=None, loss_phase=None, max_z=0.0, trace=_trace(hold_fn=(2 * force, 2 * force)))


# ------------------------------------------------------------------ the rows

def test_the_levels_are_the_designs_four_squezes_per_finger():
    assert hf.LEVELS == ("f_std", "2f_std", "40N", "70N") and hf.PINCHES == ("across_0.5", "across_com")
    assert [hf.level_force(k, 12.5) for k in hf.LEVELS] == [12.5, 25.0, 40.0, 70.0]
    assert hf.CONTROL == "across_0.5@f_std" and hf.row_id("across_com", "70N") == "across_com@70N"


def test_the_rows_are_both_pinches_at_every_level_the_control_first():
    env = FakeEnv()
    rows = hf.pinch_rows(env, "bin", 0.37, 12.0)
    assert [r["id"] for r in rows] == [hf.row_id(p, k) for p in hf.PINCHES for k in hf.LEVELS] and rows[0]["id"] == hf.CONTROL
    assert [r["force"] for r in rows] == [12.0, 24.0, 40.0, 70.0] * 2
    by = {r["id"]: r for r in rows}
    assert by["across_0.5@40N"]["f"] == 0.5 and by["across_com@40N"]["f"] == 0.37 and all(r["feasible"] for r in rows)
    assert by["across_com@f_std"]["point"] == pytest.approx([-0.15 + 0.37 * 0.3, 0.0, 0.0]) and by["across_com@f_std"]["yaw"] == np.pi / 2
    assert env.reach_calls == [0.5, 0.37]                                          # reach() once per pinch


def test_an_infeasible_pinch_is_infeasible_at_every_level_with_its_reason_and_no_substitute():
    env = FakeEnv(narrow={0.37})
    rows = hf.pinch_rows(env, "bin", 0.37, 12.0)
    com = [r for r in rows if r["pinch"] == "across_com"]
    assert all(not r["feasible"] and r["why"] and r["reach"] is None for r in com) and env.reach_calls == [0.5]
    assert all(r["feasible"] for r in rows if r["pinch"] == "across_0.5")
    env = FakeEnv(unreachable={0.37})
    com = [r for r in hf.pinch_rows(env, "bin", 0.37, 12.0) if r["pinch"] == "across_com"]
    assert all(not r["feasible"] and r["why"] == "unreachable: ik (transport)" for r in com)


# ------------------------------------------------------------------ the squeeze reader

def test_the_squeeze_reads_the_hold_steps_with_both_pads_on_the_object_per_pad():
    t = _trace(hold_fn=(140.0, 136.0, 20.0, 144.0), hold_pads=((True, True), (True, True), (True, False), (True, True)))
    s = hf.squeeze(t)
    assert s == {"hold_steps": 4, "hold_contact": 3, "hold": 70.0, "close": 11.0}  # close: the second half's both-pad steps
    s = hf.squeeze(_trace(hold_fn=(0.0, 0.0), hold_pads=((False, False), (False, False)), close_fn=(0.0, 0.0)))
    assert s == {"hold_steps": 2, "hold_contact": 0, "hold": None, "close": None}


# ------------------------------------------------------------------ the simulations

def test_each_feasible_row_runs_once_after_its_own_reset_at_its_squeeze():
    env, resets = SimEnv({(0.37, 70.0): "placed_stable"}), []
    rows = hf.pinch_rows(env, "bin", 0.37, 12.0)
    stop = hf.simulate(env, lambda: resets.append(1), rows, BASE, lambda e: "fp", "fp")
    assert stop is None and len(resets) == 8 == len(env.calls)
    assert [c[:2] for c in env.calls] == [(0.5, 12.0), (0.5, 24.0), (0.5, 40.0), (0.5, 70.0), (0.37, 12.0), (0.37, 24.0),
                                          (0.37, 40.0), (0.37, 70.0)]
    assert {c[2:] for c in env.calls} == {(0.15, 0.5, "bin", None, 2.0)}
    by = {r["id"]: r for r in rows}
    assert by["across_com@70N"]["label"] == "placed_stable" and by[hf.CONTROL]["label"] == "not_lifted"
    assert by["across_com@70N"]["squeeze"]["hold"] == 70.0 and by[hf.CONTROL]["squeeze"]["hold"] == 12.0
    assert all(k in by[hf.CONTROL] for k in ("qpos_sha", "trace_sha", "seconds", "artifact", "error", "max_z"))


def test_an_infeasible_row_is_not_run_and_an_exception_is_the_rows_error():
    env = SimEnv(fail={(0.5, 40.0)}, unreachable={0.37})
    rows = hf.pinch_rows(env, "bin", 0.37, 12.0)
    assert hf.simulate(env, lambda: None, rows, BASE, lambda e: "fp", "fp") is None and len(env.calls) == 4
    by = {r["id"]: r for r in rows}
    assert by["across_0.5@40N"]["error"].startswith("RuntimeError: the sim fell over") and by["across_0.5@70N"]["label"]
    assert "label" not in by["across_com@40N"] and "seconds" not in by["across_com@40N"]


def test_a_fingerprint_or_a_moved_pinch_stops_the_case():
    env = SimEnv()
    rows = hf.pinch_rows(env, "bin", 0.37, 12.0)
    assert "fingerprint" in hf.simulate(env, lambda: None, rows, BASE, lambda e: "other", "fp") and env.calls == []
    rows = hf.pinch_rows(env, "bin", 0.37, 12.0)
    env.x0 = 0.01                                                                    # the reset moved the object
    assert "pinch" in hf.simulate(env, lambda: None, rows, BASE, lambda e: "fp", "fp") and env.calls == []


def test_a_reset_that_is_not_at_rest_marks_the_row_an_artifact():
    env = SimEnv()
    rows = hf.pinch_rows(env, "bin", 0.37, 12.0)
    hf.simulate(env, lambda: False, rows, BASE, lambda e: "fp", "fp")
    assert all(r["artifact"] for r in rows)


# ------------------------------------------------------------------ G0 and the second seed

def test_g0_names_the_fields_that_differ_and_no_value():
    row = {"label": "tipped", "qpos_sha": "a", "trace_sha": "b", "max_z": 0.1}
    assert hf.g0_diff(row, {"label": "tipped", "qpos_sha": "a", "trace_sha": "b"}) == []
    assert hf.g0_diff(row, {"label": "placed_stable", "qpos_sha": "a", "trace_sha": "c"}) == ["label", "trace_sha"]
    assert hf.g0_diff(dict(row, label=None, error="x"), {"label": "tipped", "qpos_sha": "a", "trace_sha": "b"}) == ["label"]


def test_the_replay_runs_the_rows_in_reverse_on_copies_and_counts_label_mismatches():
    env = SimEnv({(0.37, 70.0): "placed_stable"})
    rows = hf.pinch_rows(env, "bin", 0.37, 12.0)
    hf.simulate(env, lambda: True, rows, BASE, lambda e: "fp", "fp")
    keep = json.dumps(rows, sort_keys=True)
    env.calls.clear()
    rep = hf.replay_rows(env, lambda: True, rows, BASE, lambda e: "fp", "fp")
    assert rep == {"rows": 8, "mismatches": [], "label_mismatches": 0, "stop": None} and json.dumps(rows, sort_keys=True) == keep
    assert [c[:2] for c in env.calls] == [(0.37, 70.0), (0.37, 40.0), (0.37, 24.0), (0.37, 12.0), (0.5, 70.0), (0.5, 40.0),
                                          (0.5, 24.0), (0.5, 12.0)]
    env.labels = {(0.37, 40.0): "tipped", (0.37, 70.0): "placed_stable"}
    rep = hf.replay_rows(env, lambda: True, rows, BASE, lambda e: "fp", "fp")
    assert rep["label_mismatches"] == 1 and [m["id"] for m in rep["mismatches"]] == ["across_com@40N"]
    assert rep["mismatches"][0]["fields"] == ["label"]                                 # the field names, no value


# ------------------------------------------------------------------ one case, end to end on a fake lab

class CaseEnv(SimEnv):
    """SimEnv with the reset, truth and pose fields force_case reads; ``resets`` holds the reset seeds in order."""

    def __init__(self, labels=None, **kw):
        super().__init__(labels, **kw)
        self.resets, self.reset_info = [], {"at_rest": True}
        self.props = {"axis": [1.0, 0.0, 0.0], "centre": [0.0, 0.0, 0.0], "axis_lo": -0.15, "length": 0.3}
        self.sim = SimpleNamespace(obj_mat=lambda: np.eye(3), d=None)
        self.replan_hold = False

    def reset(self, o, theta, seed, com_end, fixture, fill, frag):
        self.resets.append(int(seed))

    def truth(self):
        return {"mass": 1.7, "com_s": 0.4, "com_obj": [-0.039, 0.0, 0.0], "fill": 0.0, "mu": 0.5}   # com_f 0.37

    def nominal_mass(self, o):
        return 1.2                                                    # standard_force: 14.7 N


KEY147 = 'commit_std|{"a_transport": 2.0, "destination": "bin", "force": 14.7, "release_height": null, "s": 0.5, "v_lift": 0.15, "v_transport": 0.5}'


@pytest.fixture
def lab(monkeypatch):
    from robots.mujoco_lab import adapter, calibrate, com_gap, evpi_sim

    monkeypatch.setattr(calibrate, "case_draw", lambda seed, o, idx, m, flag, fragility=None:
                        ({"mu": 0.5}, 1, "bin", {"fill": 0.0, "frag": None}))
    monkeypatch.setattr(adapter, "object_props", lambda o: {"mass": 1.0})
    monkeypatch.setattr(com_gap, "fingerprint", lambda env: "fp")
    monkeypatch.setattr(evpi_sim, "seeds_of", lambda seed, o, idx: [seed * 1000 + idx, 777, 778])
    monkeypatch.setattr(hf, "_outcome", lambda env, ro, at_rest: {
        "label": ro.label, "artifact": at_rest is False, "damage": None, "drop_z": None, "loss_phase": None, "max_z": 0.0,
        "qpos_sha": f"q{ro.label}", "trace_sha": f"t{ro.label}"})
    return CaseEnv({(0.37, 70.0): "placed_stable"})


def _rec1(**kw):
    return dict({"object": "handal__hammer_7", "idx": 3, "fold": 0, "side": "validation", "destination": "bin",
                 "fingerprint": "fp", "status": "ok", "thin": False, "rows": [{"id": "commit_std", "action": KEY147}]}, **kw)


def _entry(env=None, **kw):
    env = env or FakeEnv()
    p = env.proposals((0.5,))[0]
    pinch = {"point": p.point, "yaw": p.yaw, "width": p.width, "feasible": True}
    return dict({"object": "handal__hammer_7", "idx": 3, "side": "validation", "base": dict(BASE, force=14.7), "pinch": pinch,
                 "ref": {"label": "not_lifted", "qpos_sha": "qnot_lifted", "trace_sha": "tnot_lifted"}}, **kw)


def test_a_case_resets_at_its_own_seed_before_each_row_and_passes_g0(lab):
    rec = hf.force_case(SimpleNamespace(env=lab), _rec1(), 3, None, _entry())
    assert lab.resets == [3003] * 9 and rec["stop"] is None and rec["error"] is None and rec["g0"] == []
    assert rec["com_f"] == pytest.approx(0.37) and rec["sim_truth"]["mass"] == 1.7 and rec["base"] == dict(BASE, force=14.7)
    assert [c[1] for c in lab.calls] == [14.7, 29.4, 40.0, 70.0] * 2 and len(rec["rows"]) == 8 and rec["replay"] is None
    assert rec["hammer"] and rec["skipped"] is None
    assert lab.replan_hold is hf.REPLAN_HOLD is True                    # Stage 2c's flag (ruling 7): the re-plan at the hold on


def test_a_g0_mismatch_stops_the_case_and_names_the_fields_only(lab):
    rec = hf.force_case(SimpleNamespace(env=lab), _rec1(), 3, None, _entry(ref={"label": "tipped", "qpos_sha": "qnot_lifted",
                                                                                     "trace_sha": "tnot_lifted"}))
    assert rec["g0"] == ["label"] and rec["stop"] == "G0: the control row differs from the corrected pose-set record in label"
    assert "tipped" not in rec["stop"] and "not_lifted" not in rec["stop"]


def test_the_case_stops_on_a_control_pinch_or_a_base_that_is_not_the_pose_sets(lab):
    e = _entry()
    rec = hf.force_case(SimpleNamespace(env=lab), _rec1(), 3, None, dict(e, pinch=dict(e["pinch"], width=0.5)))
    assert "across_0.5" in rec["stop"] and lab.calls == []
    rec = hf.force_case(SimpleNamespace(env=lab), _rec1(), 3, None, dict(e, base=BASE))
    assert "base" in rec["stop"] and lab.calls == []
    rec = hf.force_case(SimpleNamespace(env=lab), _rec1(fingerprint="other"), 3, None, e)
    assert "fingerprint" in rec["stop"] and lab.calls == []


def test_the_replay_resets_at_the_second_seed_and_a_label_mismatch_stops_the_case(lab):
    rec = hf.force_case(SimpleNamespace(env=lab), _rec1(), 3, None, _entry(), replay=True)
    assert lab.resets == [3003] * 9 + [777] * 8 and rec["replay"]["label_mismatches"] == 0 and rec["stop"] is None
    lab.resets.clear()
    calls = {"n": 0}
    orig = lab.execute

    def flaky(p, force, *a, **kw):                                  # the second seed turns one row
        calls["n"] += 1
        ro = orig(p, force, *a, **kw)
        return SimpleNamespace(**dict(vars(ro), label="tipped")) if calls["n"] == 12 else ro
    lab.execute = flaky
    rec = hf.force_case(SimpleNamespace(env=lab), _rec1(), 3, None, _entry(), replay=True)
    assert rec["replay"]["label_mismatches"] == 1 and "second seed" in rec["stop"] and "tipped" not in rec["stop"]


def test_a_skipped_case_runs_nothing(lab):
    rec = hf.force_case(SimpleNamespace(env=lab), _rec1(status="error"), 3, None, _entry())
    assert rec["skipped"] == "stage1 error" and rec["rows"] == [] and lab.resets == [] and rec["stop"] is None


# ------------------------------------------------------------------ the sim gate

def test_the_sim_gate_refuses_another_sim_or_a_dirty_one(monkeypatch):
    out = {"log": hf.CONTACTS_SIM + "\n", "status": ""}
    monkeypatch.setattr(hf, "_git", lambda d, *args: out["log"] if args[0] == "log" else out["status"])
    assert hf.sim_problem("/sim") is None
    out["status"] = " M scene.py\n"
    assert "dirty" in hf.sim_problem("/sim")
    out.update(log="e823247052a2bf4d1b97d5077a51f0ead91fd80c\n", status="")
    assert "e823247052a2bf4d1b97d5077a51f0ead91fd80c" in hf.sim_problem("/sim")
    assert "not set" in hf.sim_problem(None)


# ------------------------------------------------------------------ the case entries (published inputs)

def _prow(rid, label=None, feasible=True, qpos="q0", trace="t0"):
    d, f = rid.split("_")
    r = {"id": rid, "dir": d, "f": float(f), "current": d == "across" and float(f) in (0.3, 0.5, 0.7), "point": [float(f), 0.0, 0.7],
         "yaw": 1.5, "width": 0.03, "feasible": feasible, "why": None if feasible else "too wide"}
    if feasible:
        r.update(label=label, artifact=False, error=None, qpos_sha=qpos, trace_sha=trace, seconds=1.0)
    return r


def _prec(**kw):
    rows = [_prow("across_0.5", "not_lifted"), _prow("across_0.3", "not_lifted"), _prow("across_0.7", "tipped"),
            _prow("along_0.5", feasible=False)]
    return dict({"object": "handal__hammer_7", "idx": 3, "side": "test", "fold": 0, "skipped": None, "error": None, "dropped": False,
                 "fingerprint": "fp", "base": BASE, "com_f": 0.41, "sim_truth": {"mass": 1.7}, "rows": rows}, **kw)


def _rep(diffs=None):
    d = diffs or {}
    labels = {"across_0.5": "not_lifted", "across_0.3": "not_lifted", "across_0.7": "tipped"}
    return {"rows": [{"id": k, "diffs": d.get(k, {}), "stored_label": v} for k, v in labels.items()]}


def test_a_case_entry_takes_the_control_row_as_the_pass_corrected_it():
    e = hf.case_entry(_prec(), _rep({"across_0.5": {"label": ["not_lifted", "tipped"], "trace_sha": ["t0", "t1"]},
                                     "across_0.3": {"label": ["not_lifted", "placed_stable"]}}), r4=False)
    assert e["ref"] == {"label": "tipped", "qpos_sha": "q0", "trace_sha": "t1"}
    assert e["pinch"] == {"point": [0.5, 0.0, 0.7], "yaw": 1.5, "width": 0.03, "feasible": True} and e["base"] == BASE
    assert e["ceiling10"] is True and e["r4"] is False and e["mass"] == 1.7 and e["com_f"] == 0.41     # the corrected ceiling
    assert (e["object"], e["idx"], e["side"], e["fold"], e["fingerprint"]) == ("handal__hammer_7", 3, "test", 0, "fp")
    assert hf.case_entry(_prec(), _rep(), r4=None)["ceiling10"] is False


def test_a_skipped_case_has_no_reference_and_a_simulated_case_needs_its_replay():
    e = hf.case_entry(_prec(skipped="stage1 artifact", dropped=True, rows=[]), None, r4=None)
    assert e["skipped"] == "stage1 artifact" and e["ref"] is None and e["pinch"] is None and e["ceiling10"] is None
    with pytest.raises(ValueError):
        hf.case_entry(_prec(), None, r4=None)


def test_an_undetermined_corrected_ceiling_is_none():
    rows = [_prow("across_0.5", "not_lifted"), dict(_prow("across_0.3", "not_lifted"), artifact=True)]
    e = hf.case_entry(_prec(rows=rows), {"rows": [{"id": "across_0.5", "diffs": {}, "stored_label": "not_lifted"},
                                                  {"id": "across_0.3", "diffs": {}, "stored_label": "not_lifted"}]}, r4=None)
    assert e["ceiling10"] is None


def test_the_smoke_takes_the_first_two_simulated_cases_by_idx_of_each_object():
    es = [{"object": o, "idx": i, "skipped": "stage1 artifact" if (o, i) == ("b", 0) else None} for o in ("a", "b") for i in (3, 0, 2, 1)]
    assert hf.smoke_cases(es) == [("a", 0), ("a", 1), ("b", 1), ("b", 2)]


# ------------------------------------------------------------------ the run: gates, metrics and the log line

def test_the_run_refuses_another_sim_before_it_reads_or_writes_anything(monkeypatch, tmp_path):
    monkeypatch.setattr(hf, "sim_problem", lambda d, want=hf.CONTACTS_SIM: "the sim /x is at e823247, not the contacts sim")
    a = SimpleNamespace(cases_file=str(tmp_path / "missing.json"), out=str(tmp_path / "u"), cases=None, objects=None, workers=1,
                        replay=False)
    with pytest.raises(SystemExit, match="contacts sim"):
        hf.run(a)
    assert not os.path.exists(a.out)


def _done_rec(labels=("not_lifted",) * 8, **kw):
    rows = [dict(_r8(i, lab)) for i, lab in enumerate(labels)]
    return dict({"object": "handal__hammer_7", "idx": 3, "side": "test", "skipped": None, "stop": None, "error": None, "g0": [],
                 "rows": rows, "replay": None, "seconds": 10.0}, **kw)


def _r8(i, label):
    pinch, level = hf.PINCHES[i // 4], hf.LEVELS[i % 4]
    return {"id": hf.row_id(pinch, level), "pinch": pinch, "level": level, "feasible": True, "label": label, "artifact": False,
            "error": None, "seconds": 1.0, "squeeze": {"hold_steps": 50, "hold_contact": 0, "hold": None, "close": 10.0 + i}}


def test_the_metrics_and_the_log_line_hold_counts_and_no_label():
    done = [_done_rec(), _done_rec(labels=("placed_stable",) * 8, idx=4, g0=["label"], stop="G0: ... in label"),
            _done_rec(idx=5, skipped="stage1 artifact", rows=[], g0=None)]
    m = hf.metrics(done)
    assert m["cases"] == 3 and m["skipped"] == 1 and m["rows_simulated"] == 16 and m["g0_compared"] == 2 and m["g0_mismatches"] == 1
    assert m["stops"] == 1 and "placed_stable" not in json.dumps(m) and "not_lifted" not in json.dumps(m)
    line = hf.case_line(done[1])
    assert "placed_stable" not in line and "G0 differs" in line and "STOP" in line


# ------------------------------------------------------------------ the smoke report

def test_the_smoke_report_counts_both_gates_and_the_squeeze_without_a_label_or_the_hold_counts():
    a = _done_rec(replay={"rows": 8, "mismatches": [{"id": "across_com@70N", "fields": ["max_z"]}], "label_mismatches": 0, "stop": None},
                  replay_seconds=9.0)
    for r in a["rows"]:
        r["squeeze"] = dict(r["squeeze"], hold=r["force"] if "force" in r else 2.0 * (1 + hf.LEVELS.index(r["level"])), hold_contact=40)
    b = _done_rec(idx=4, labels=("placed_stable",) * 8, replay={"rows": 8, "mismatches": [], "label_mismatches": 0, "stop": None},
                  replay_seconds=9.0)
    sm = hf.smoke_report([a, b], full_cases=99)
    assert sm["cases"] == 2 and sm["g0"] == {"compared": 2, "mismatches": 0}
    assert sm["g1"] == {"cases": 2, "rows": 16, "label_mismatches": 0, "other_mismatches": 1, "other_fields": {"max_z": 1}}
    assert sm["squeeze"]["70N"]["hold_median"] == 8.0 and "hold_rows" not in json.dumps(sm["squeeze"])
    assert sm["squeeze"]["f_std"]["close_median"] == pytest.approx(np.median([10.0, 14.0, 10.0, 14.0]))
    assert sm["squeeze"]["f_std"]["close_rows"] == 4
    assert sm["seconds_per_case"] == pytest.approx(1.0) and sm["projection_cpu_h"] == pytest.approx(99 / 3600.0)
    text = json.dumps(sm) + hf.smoke_markdown(sm)
    assert "placed_stable" not in text and "not_lifted" not in text
