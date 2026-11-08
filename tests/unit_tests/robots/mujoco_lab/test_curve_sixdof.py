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
