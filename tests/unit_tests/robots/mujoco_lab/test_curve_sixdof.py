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


# ------------------------------------------------------------------ the box-contact recorder (revised G1′)

BOXES_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="table" type="box" size="0.5 0.5 0.02" pos="0 0 -0.02"/>
    <body name="obj" pos="0 0 0.05"><freejoint/><geom name="hull0" type="box" size="0.03 0.03 0.05" mass="0.2"/></body>
    <body name="shelf" pos="0.3 0 0.05"><freejoint/><geom name="fx_shelf_0" type="box" size="0.03 0.03 0.05" mass="0.2"/></body>
    <body name="hand" mocap="true" pos="0 0 0.3">
      <geom name="palm" type="box" size="0.045 0.02 0.01" contype="2" conaffinity="1"/>
      <geom name="wrist" type="cylinder" size="0.02 0.03" pos="0 0 0.05" contype="2" conaffinity="1"/>
      <body name="finger_l" pos="0.3 0 0.01">
        <geom type="box" size="0.004 0.008 0.02" contype="2" conaffinity="1"/>
        <geom name="pad_l" type="box" size="0.002 0.008 0.008" pos="0.006 0 -0.014"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class _BoxSim:
    """A free object (hull0) and a free fixture on a table; a mocap hand with the palm, the wrist and a finger (its box
    unnamed, as in the lab, and its pad). At hand height 0.105 m the palm is 5 mm into the object's top, the finger box
    5 mm and the pad 7 mm into the fixture's top; at 0.3 m nothing touches."""

    def __init__(self):
        import mujoco

        self.m = mujoco.MjModel.from_xml_string(BOXES_XML)
        self.d = mujoco.MjData(self.m)
        self.hull_ids = [self.m.geom("hull0").id]
        mujoco.mj_forward(self.m, self.d)

    def substep(self):
        import mujoco

        mujoco.mj_step(self.m, self.d)


class _BoxEnv:
    def __init__(self):
        self.sim = _BoxSim()
        self.s = {"scene": SimpleNamespace(CTRL_HZ=10, SUB=4)}

    def _tracer(self, pt, yaw):
        trace = []
        return trace, lambda phase: trace.append({"phase": phase})

    def at(self, z):
        self.sim.d.mocap_pos[0][2] = z

    def run(self, script):
        """script: [(phase or None, hand z)]: SUB substeps at that height, then the traced step's on_step (None: untraced,
        as the free move to the pregrasp)."""
        trace, on_step = self._tracer(None, 0.0)
        for phase, z in script:
            self.at(z)
            for _ in range(self.s["scene"].SUB):
                self.sim.substep()
            if phase is not None:
                on_step(phase)
        return trace


SCRIPT = [(None, 0.105), ("approach", 0.3), ("close", 0.105), ("lift", 0.3)]


def test_the_recorder_reads_every_substeps_box_contacts_by_phase_and_partner():
    env = _BoxEnv()
    with cs.BoxContacts(env) as bc:
        env.run(SCRIPT)
        env.at(0.105)
        for _ in range(3):                                           # after the last traced step
            env.sim.substep()
    s = bc.summary()
    assert s["observed"] is True and s["first"]["phase"] == "untraced" and s["first"]["t"] is None
    assert s["first"]["box"] in ("palm", "finger") and s["first"]["partner"] == ("object" if s["first"]["box"] == "palm" else "fx_shelf_0")
    assert set(s["by"]) == {f"{ph}|{pair}" for ph in ("untraced", "close", "after") for pair in ("palm/object", "finger/fx_shelf_0")}
    assert s["by"]["close|palm/object"]["substeps"] == 4 and 3.0 < s["by"]["close|palm/object"]["deepest_mm"] < 6.0
    assert 3.0 < s["by"]["close|finger/fx_shelf_0"]["deepest_mm"] < 6.0                       # the box, not the deeper pad
    assert s["by"]["untraced|palm/object"]["substeps"] == 4 and s["by"]["after|palm/object"]["substeps"] == 3


def test_the_recorder_dates_a_traced_first_contact_and_keeps_the_deepest():
    env = _BoxEnv()
    with cs.BoxContacts(env) as bc:
        env.run([("approach", 0.3), ("approach", 0.3), ("close", 0.105), ("close", 0.103)])
    s = bc.summary()
    assert s["first"]["phase"] == "close" and s["first"]["t"] == pytest.approx(0.2) and s["first"]["mm"] > 4.0
    assert s["by"]["close|palm/object"]["substeps"] == 8 and s["by"]["close|palm/object"]["deepest_mm"] > 6.0


def test_a_second_rollout_in_one_context_counts_each_substep_once():
    env = _BoxEnv()
    wrapped = env.sim.substep
    env.sim.substep = wrapped
    with cs.BoxContacts(env) as bc:
        env.run([("close", 0.105)])
        env.run([("close", 0.105)])
    assert bc.summary()["by"]["close|palm/object"]["substeps"] == 8 and env.sim.substep is wrapped


def test_the_recorder_reads_only():
    a, b = _BoxEnv(), _BoxEnv()
    a.run(SCRIPT)
    with cs.BoxContacts(b):
        b.run(SCRIPT)
    assert np.array_equal(a.sim.d.qpos, b.sim.d.qpos) and np.array_equal(a.sim.d.qvel, b.sim.d.qvel)


def test_the_recorder_restores_the_tracer_and_the_substep_and_sees_nothing_without_a_rollout():
    env = _BoxEnv()
    wrapped = env.sim.substep
    env.sim.substep = wrapped                                        # the adapter's meter: an instance attribute
    with cs.BoxContacts(env) as bc:
        pass
    assert bc.summary() == {"observed": False, "first": None, "by": {}}
    with cs.BoxContacts(env):
        env.run([("close", 0.105)])
    assert "_tracer" not in vars(env) and env.sim.substep is wrapped
    with cs.BoxContacts(SimpleNamespace()) as bc:                    # an env without a tracer
        pass
    assert bc.summary() == {"observed": False, "first": None, "by": {}}


def test_with_the_recorder_on_every_replayed_row_carries_its_box_contacts(lab):
    rec, entry = _pilot(lab)
    out = cs.g1_case(SimpleNamespace(env=lab), _rec1(), 3, None, rec, "pilot", entry, record=True)
    assert [r["box_contacts"] for r in out["rows"]] == [{"observed": False, "first": None, "by": {}}] * 4   # the fake has no tracer
    assert all(r["same"] for r in out["rows"])
    assert all("box_contacts" not in r for r in cs.g1_case(SimpleNamespace(env=lab), _rec1(), 3, None, rec, "pilot", entry)["rows"])


def test_with_the_recorder_on_each_row_hands_its_env_to_the_recorder(lab, monkeypatch):
    got = []

    class Spy(cs.BoxContacts):
        def __init__(self, env):
            got.append(env)
            super().__init__(env)

    monkeypatch.setattr(cs, "BoxContacts", Spy)
    rec, entry = _pilot(lab)
    cs.g1_case(SimpleNamespace(env=lab), _rec1(), 3, None, rec, "pilot", entry, record=True)
    cs.g1_case(SimpleNamespace(env=lab), _rec1(), 3, None, rec, "pilot", entry)
    assert got == [lab] * 4 + [None] * 4


def test_an_objects_job_hands_the_recorder_flag_to_each_case(monkeypatch):
    from robots.mujoco_lab import calibrate as cal
    from robots.mujoco_lab import prior

    seen = []
    monkeypatch.setattr(cal, "_primitives", lambda b, t, k: SimpleNamespace(env=SimpleNamespace(), manifest={}))
    monkeypatch.setattr(cal, "job_force_noise", lambda o, m: None)
    monkeypatch.setattr(prior, "fragility_world", lambda m: None)
    monkeypatch.setattr(cs, "g1_case", lambda *a, record=False, **k: seen.append(record) or
                        {"source": "poseset", "object": "box", "idx": 0, "rows": [], "stop": None, "error": None})
    item = ({"object": "box", "idx": 0}, {"object": "box", "idx": 0})
    for flag in (True, False):
        cs.g1_object(("box", [item, item], {"bundle": "b", "seed": 0, "source": "poseset", "grasps": None, "record": flag}))
    assert seen == [True, True, False, False]


def test_the_metrics_count_the_rows_the_recorder_saw_and_the_changed_rows_without_a_box_contact():
    seen, hit = {"observed": True, "first": None, "by": {}}, {"observed": True, "first": _contact("release"), "by": {}}
    rows = [{"id": "g0", "same": True, "diffs": {}, "mapping": None, "hand_path": None, "box_contacts": seen},
            {"id": "g1", "same": False, "diffs": {"trace_sha": ["a", "b"]}, "mapping": None, "hand_path": None, "box_contacts": hit},
            {"id": "g2", "same": False, "diffs": {"trace_sha": ["a", "b"]}, "mapping": None, "hand_path": None, "box_contacts": seen},
            {"id": "g3", "same": False, "diffs": {"trace_sha": ["a", "b"]}, "mapping": None, "hand_path": None}]
    m = cs.g1_metrics([{"source": "pilot", "side": "test", "rows": rows, "stop": None, "error": None}])
    assert (m["rows_recorded"], m["rows_with_box_contact"], m["changed_without_box_contact"]) == (3, 1, 1)


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


def _contact(phase, box="finger", partner="object", mm=0.1, t=1.0):
    return {"phase": phase, "t": t, "box": box, "partner": partner, "mm": mm}


def _row(i, stored="placed_stable", label="placed_stable", diffs=None, error=None, mapping=None, contact=None, recorded=True):
    r = {"id": i, "same": not diffs and mapping is not False, "diffs": diffs or {}, "mapping": mapping, "hand_path": None,
         "seconds": 1.0, "stored_label": stored, "label": label, "error": error}
    if recorded:
        by = {f"{contact['phase']}|{contact['box']}/{contact['partner']}": {"substeps": 3, "deepest_mm": contact["mm"]}} if contact else {}
        r["box_contacts"] = {"observed": True, "first": contact, "by": by}
    return r


def _g1p_units(tmp_path, mug_g0=None):
    """A pose-set box case (across_0.5 changed after a palm contact at the approach; across_0.1 and across_0.7 identical
    without one) and a pilot mug case (g0 changed after a finger contact at the release; g1 identical with a contact)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    a = {"source": "poseset", "side": "test", "object": "box", "idx": 0, "stop": None, "error": None, "seconds": 4.0,
         "rows": [_row("across_0.1"), _row("across_0.5", label="not_lifted", diffs={"label": ["placed_stable", "not_lifted"]},
                                            contact=_contact("approach", "palm", mm=2.0)), _row("across_0.7")]}
    b = {"source": "pilot", "side": "validation", "object": "mug", "idx": 2, "stop": None, "error": None, "seconds": 2.0,
         "rows": [mug_g0 or _row("g0", diffs={"trace_sha": ["a", "b"]}, contact=_contact("release", t=10.5)),
                  _row("g1", contact=_contact("release", mm=0.001, t=10.4))]}
    return [_unit(tmp_path, 0, [a], 3), _unit(tmp_path, 1, [b], 2)]


FLAGGED = {("poseset", "test", "box", 0, "across_0.5"), ("poseset", "test", "box", 0, "across_0.7"), ("poseset", "test", "box", 0, "along_0.9")}


def test_revised_g1_prime_passes_when_every_changed_row_shows_a_box_contact_and_reads_the_audits_list_beside_it(tmp_path):
    rep = cs.g1p_report(_g1p_units(tmp_path), FLAGGED, {"pilot": 2, "poseset": 3})
    assert rep["passed"] is True and rep["rows_not_recorded"] == 0
    assert rep["gate"] == {"changed": 2, "changed_with_contact": 2, "changed_without_contact": []}
    assert rep["flagged"] == {"rows": 2, "changed": 1, "identical": 1}
    assert rep["unflagged"] == {"rows": 3, "identical": 2, "changed": 1, "changed_by_first_phase": {"release": 1}}
    assert rep["unflagged_changed_by_first_contact"] == {"release finger/object": 1}
    assert rep["identical_with_contact"] == [{"source": "pilot", "side": "validation", "object": "mug", "idx": 2, "id": "g1",
                                              "first_contact": _contact("release", mm=0.001, t=10.4)}]
    assert rep["transitions_by_object"] == {"box": {"placed_stable → not_lifted": 1}, "mug": {"placed_stable → placed_stable": 1}}
    assert rep["flagged_by_transition"] == {"placed_stable → not_lifted": 1} and rep["flagged_by_object"] == {"box": 1}
    assert [(x["object"], x["id"], x["fields"], x["first_contact"]) for x in rep["changed_not_in_list"]] == [
        ("mug", "g0", ["trace_sha"], _contact("release", t=10.5))]
    assert rep["flagged_not_replayed"] == [["poseset", "test", "box", 0, "along_0.9"]]


def test_revised_g1_prime_fails_on_a_changed_row_without_a_box_contact_whatever_the_list_says(tmp_path):
    bare = _row("g0", stored="tipped", label=None, error="Unreachable: x", diffs={"label": ["tipped", None]})
    flagged = FLAGGED | {("pilot", "validation", "mug", 2, "g0")}
    rep = cs.g1p_report(_g1p_units(tmp_path, mug_g0=bare), flagged, {"pilot": 2, "poseset": 3})
    assert rep["passed"] is False and rep["gate"]["changed_with_contact"] == 1
    assert rep["gate"]["changed_without_contact"] == [{"source": "pilot", "side": "validation", "object": "mug", "idx": 2, "id": "g0",
                                                       "fields": ["label"], "stored_label": "tipped", "label": None,
                                                       "error": "Unreachable: x", "observed": True}]
    bare["box_contacts"]["observed"] = False                                          # the recorder ran, no rollout did
    rep = cs.g1p_report(_g1p_units(tmp_path / "b", mug_g0=bare), flagged, {"pilot": 2, "poseset": 3})
    assert rep["passed"] is False and rep["gate"]["changed_without_contact"][0]["observed"] is False


def test_revised_g1_prime_fails_on_a_row_the_recorder_did_not_see_a_stop_a_missing_row_a_bad_exit_or_a_geometry_check(tmp_path):
    rep = cs.g1p_report(_g1p_units(tmp_path / "a", mug_g0=_row("g0", recorded=False)), FLAGGED, {"pilot": 2, "poseset": 3})
    assert rep["passed"] is False and rep["rows_not_recorded"] == 1 and rep["gate"]["changed_without_contact"] == []
    units = _g1p_units(tmp_path / "b")
    assert cs.g1p_report(units, FLAGGED, {"pilot": 2, "poseset": 3})["passed"] is True
    stop = {"source": "poseset", "side": "test", "object": "can", "idx": 1, "stop": "g1: the hulls after the reset differ from the record's",
            "error": None, "seconds": 1.0, "rows": []}
    rep = cs.g1p_report(units + [_unit(tmp_path / "b", 2, [stop], 0)], FLAGGED, {"pilot": 2, "poseset": 3})
    assert rep["passed"] is False and rep["stops"] == 1 and rep["missing_rows"] == {"pilot": 0, "poseset": 0}
    assert cs.g1p_report(units, FLAGGED, {"pilot": 3, "poseset": 3})["passed"] is False                 # a row was not replayed
    units = _g1p_units(tmp_path / "c")
    (tmp_path / "c" / "unit_1" / "run.exit").write_text("EXIT 1\n")
    rep = cs.g1p_report(units, FLAGGED, {"pilot": 2, "poseset": 3})
    assert rep["passed"] is False and rep["units_not_exit_0"] == [units[1]]
    geo = {"source": "pilot", "side": "test", "object": "can", "idx": 4, "stop": None, "error": None, "seconds": 1.0,
           "rows": [_row("g3", mapping=False, contact=_contact("approach"))]}
    rep = cs.g1p_report(_g1p_units(tmp_path / "d") + [_unit(tmp_path / "d", 2, [geo], 1)], FLAGGED, {"pilot": 3, "poseset": 3})
    assert rep["passed"] is False and rep["mapping_failures"] == 1
    assert [x["fields"] for x in rep["changed_not_in_list"] if x["id"] == "g3"] == [["mapping"]]


def _g1_run(tmp_path, monkeypatch, seen=None):
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
    monkeypatch.setattr(cs, "g1_object", lambda job: (seen.append(job[2]["record"]) if seen is not None else None) or iter([case]))
    return ["g1", "--source", "poseset", "--runs", str(run), "--sides", "test"]


def test_the_g1_command_exits_0_on_differing_rows_with_allow_diffs_and_3_without(tmp_path, monkeypatch):
    args = _g1_run(tmp_path, monkeypatch)
    assert cs.main(args + ["--allow-diffs", "--out", str(tmp_path / "o1")]) == str(tmp_path / "o1")
    with pytest.raises(SystemExit) as e:
        cs.main(args + ["--out", str(tmp_path / "o2")])
    assert e.value.code == 3


def test_the_g1_command_hands_the_recorder_flag_to_every_object_and_records_it(tmp_path, monkeypatch):
    seen = []
    args = _g1_run(tmp_path, monkeypatch, seen)
    cs.main(args + ["--allow-diffs", "--box-contacts", "--out", str(tmp_path / "o1")])
    cs.main(args + ["--allow-diffs", "--out", str(tmp_path / "o2")])
    assert seen == [True, False]
    assert json.loads((tmp_path / "o1" / "run_config.json").read_text())["box_contacts"] is True


def test_the_g1_prime_report_command_writes_the_gate_and_the_audit_reading_and_exits_3_on_a_changed_row_without_contact(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(cs, "G1_EXPECTED", {"pilot": 2, "poseset": 3})
    lst = tmp_path / "flagged_rows.jsonl"
    lines = [{"source": "poseset", "side": "test", "object": "box", "idx": 0, "id": i, "label": "placed_stable", "depth_mm": 3.0}
             for i in ("across_0.5", "across_0.7")]
    lines.append({"source": "stage2", "side": "test", "object": "box", "idx": 0, "first": "a", "attempt": 1, "row": "a", "depth_mm": 2.0})
    lst.write_text("".join(json.dumps(x) + "\n" for x in lines))
    _g1p_units(tmp_path / "q")
    rep = cs.main(["g1p-report", "--queue", str(tmp_path / "q"), "--flagged", str(lst), "--out", str(tmp_path / "s")])
    md = (tmp_path / "s" / "g1p_summary.md").read_text()
    assert rep["passed"] and "Result: PASSED." in md and "| release | 1 |" in md and "| mug | placed_stable → placed_stable | 1 |" in md
    assert json.loads((tmp_path / "s" / "g1p_summary.json").read_text())["unflagged"]["changed"] == 1
    _g1p_units(tmp_path / "q2", mug_g0=_row("g0", diffs={"trace_sha": ["a", "b"]}))
    with pytest.raises(SystemExit) as e:
        cs.main(["g1p-report", "--queue", str(tmp_path / "q2"), "--flagged", str(lst), "--out", str(tmp_path / "s2")])
    md = (tmp_path / "s2" / "g1p_summary.md").read_text()
    assert e.value.code == 3 and "Result: FAILED." in md
    assert md.index("mug 2 g0: fields trace_sha; no box contact") < md.index("## Beside the gate")   # listed under the gate
