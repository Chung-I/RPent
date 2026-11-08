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

"""The pose-set ablation (design 2026-11-08-pose-set-ablation-design.md, section 3.1): ten pose rows (pinch fractions
0.1 to 0.9, across and along the long axis) at commit_std, each simulated once at the case's own reset seed; the
capability ceilings over the ten rows and over the three current rows."""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

SIM = pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")
FRACS = (0.1, 0.3, 0.5, 0.7, 0.9)


@SIM
def test_the_turned_proposals_keep_the_points_and_turn_the_yaw():
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    env = MuJoCoLabEnv()
    env.reset("ycb__sugar_box", seed=0)
    a, b = env.proposals(FRACS), env.proposals(FRACS, turn=-np.pi / 2)
    assert [p.point for p in a] == [p.point for p in b] and [p.s for p in b] == list(FRACS)
    assert all(q.yaw == p.yaw - np.pi / 2 for p, q in zip(a, b))
    assert all(q.width == env.pinch_width(np.asarray(q.point), q.yaw) for q in b)
    assert any(abs(q.width - p.width) > 0.01 for p, q in zip(a, b))       # the turned pinch closes on another side


@SIM
def test_the_default_proposals_are_unchanged():
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    env = MuJoCoLabEnv()
    env.reset("ycb__sugar_box", seed=0)
    p = env.proposals()
    assert [x.s for x in p] == [0.3, 0.5, 0.7] and all(x.yaw == env._pinch_frame()[0] for x in p)
    assert [x.point for x in p] == [x.point for x in env.proposals((0.3, 0.5, 0.7), turn=0.0)]


@SIM
def test_the_adapter_extents_are_the_features_extents():
    from robots.mujoco_lab.adapter import MuJoCoLabEnv
    from rpent.skills.pick_place import features

    env = MuJoCoLabEnv()
    env.reset("ycb__sugar_box", seed=0)
    ph = env.s["scene"].PAD_HALF
    for p in env.proposals(FRACS, turn=-np.pi / 2):
        lo, hi = env.pinch_extent(p.point, p.yaw)
        assert (lo, hi) == features.pinch_extent(env._hull_vertex_list(), p.point, p.yaw, ph) and hi - lo == p.width
        assert env.column_extent(p.point, p.yaw) == features.column_extent(env._hull_vertex_list(), p.point, p.yaw, ph)


# ------------------------------------------------------------------ the pose rows (no sim)

from robots.mujoco_lab import curve_poseset as ps  # noqa: E402
from rpent.skills.pick_place import features  # noqa: E402
from rpent.skills.pick_place.protocol import IRREVERSIBLE  # noqa: E402


def _box(l, w, h, centre=(0.0, 0.0, 0.0)):
    return np.array([[x, y, z] for x in (-l / 2, l / 2) for y in (-w / 2, w / 2) for z in (-h / 2, h / 2)]) + np.asarray(centre)


class FakeEnv:
    """A box lying along x (length L, width W, height H) centred at (0, y0, 0): the across pinch closes along y (yaw
    pi/2), the along pinch along x (yaw 0). ``unreachable``: the (direction, fraction) pairs reach() refuses."""

    max_pinch_width = 0.09

    def __init__(self, L=0.06, W=0.04, H=0.03, y0=0.0, extra=(), unreachable=()):
        self.hulls = [_box(L, W, H, (0.0, y0, 0.0))] + list(extra)
        self.L, self.unreachable, self.reach_calls = L, set(unreachable), []

    def proposals(self, fractions=(0.3, 0.5, 0.7), turn=0.0):
        yaw = np.pi / 2 + float(turn)
        pts = [np.array([-self.L / 2 + f * self.L, 0.0, 0.0]) for f in fractions]
        return [SimpleNamespace(s=float(f), point=[float(x) for x in p], yaw=yaw, width=features.pinch_width(self.hulls, p, yaw))
                for f, p in zip(fractions, pts)]

    def pinch_extent(self, point, yaw):
        return features.pinch_extent(self.hulls, point, yaw)

    def column_extent(self, point, yaw):
        return features.column_extent(self.hulls, point, yaw)

    def reach(self, p, destination):
        d = "across" if abs(p.yaw - np.pi / 2) < 1e-9 else "along"
        self.reach_calls.append((d, p.s))
        return {"reason": "ik", "phase": "transport"} if (d, p.s) in self.unreachable else None


def _by_id(rows):
    return {r["id"]: r for r in rows}


def test_the_row_order_puts_the_current_across_rows_first_nearest_the_middle():
    assert ps.ROW_ORDER == (("across", 0.5), ("across", 0.3), ("across", 0.7), ("across", 0.1), ("across", 0.9),
                            ("along", 0.5), ("along", 0.3), ("along", 0.7), ("along", 0.1), ("along", 0.9))
    assert [ps.row_id(d, f) for d, f in ps.ROW_ORDER[:2]] == ["across_0.5", "across_0.3"]


def test_centred_needs_both_ends_within_the_bound():
    assert ps.centred((-0.045, 0.045), 0.045) and not ps.centred((-0.0451, 0.0), 0.045)
    assert not ps.centred((0.0, 0.0451), 0.045) and not ps.centred((float("nan"), float("nan")), 0.045)


def test_a_lying_box_gets_every_across_row_and_the_centred_along_rows():
    """L 6 cm: the along pinch at f spans [-f L, (1 - f) L], so 0.1 and 0.9 put an end 5.4 cm out, past the 4.5 cm bound."""
    rows = ps.pose_rows(FakeEnv(), "bin")
    assert [r["id"] for r in rows] == [ps.row_id(d, f) for d, f in ps.ROW_ORDER]
    by = _by_id(rows)
    assert [r["id"] for r in rows if r["feasible"]] == ["across_0.5", "across_0.3", "across_0.7", "across_0.1", "across_0.9",
                                                          "along_0.5", "along_0.3", "along_0.7"]
    assert by["along_0.1"]["why"] == ps.OFF_CENTRE and by["along_0.9"]["why"] == ps.OFF_CENTRE
    assert by["along_0.1"]["slab"] == [pytest.approx(-0.006, abs=1e-9), pytest.approx(0.054, abs=1e-9)]
    assert [r["current"] for r in rows] == [True, True, True] + [False] * 7
    assert by["across_0.5"]["width"] == pytest.approx(0.04, abs=1e-9) and by["across_0.5"]["reach"] == "ok"


def test_the_current_rows_keep_the_current_rule_and_the_new_rows_need_centring():
    """An across extent from -1 to +5 cm (6 cm wide, usable): the current rows stay feasible and are marked off-centre
    (the disclosure); the new across rows at 0.1 and 0.9 are not feasible."""
    by = _by_id(ps.pose_rows(FakeEnv(W=0.06, y0=0.02), "bin"))
    for f in (0.3, 0.5, 0.7):
        r = by[f"across_{f}"]
        assert r["feasible"] and r["current"] and not r["centred"] and r["why"] is None
    assert by["across_0.1"]["why"] == ps.OFF_CENTRE and by["across_0.9"]["why"] == ps.OFF_CENTRE
    assert by["along_0.5"]["feasible"]


def test_reach_runs_only_on_rows_that_pass_width_and_centring_and_can_refuse_one():
    env = FakeEnv(unreachable={("along", 0.5)})
    by = _by_id(ps.pose_rows(env, "bin"))
    assert not by["along_0.5"]["feasible"] and by["along_0.5"]["why"] == "unreachable: ik (transport)"
    assert by["along_0.5"]["reach"] == "ik (transport)"
    assert ("along", 0.1) not in env.reach_calls and ("along", 0.9) not in env.reach_calls and len(env.reach_calls) == 8


def test_the_column_check_refuses_a_new_row_whose_object_reaches_past_a_pad_above_the_pinch():
    """A 10 cm cap 3 to 4 cm up, above the pad slab: the along pinch at 0.5 is centred in the slab, not in the column."""
    cap = _box(0.10, 0.04, 0.01, (0.0, 0.0, 0.035))
    by = _by_id(ps.pose_rows(FakeEnv(extra=[cap]), "bin"))
    assert by["along_0.5"]["why"] == ps.OFF_CENTRE_COLUMN and ps.centred(by["along_0.5"]["slab"], 0.045)
    assert by["across_0.1"]["feasible"]                                 # across, the cap is 4 cm wide


def test_a_wide_box_loses_every_across_row_to_the_width_rule():
    by = _by_id(ps.pose_rows(FakeEnv(W=0.10), "bin"))
    assert all(by[f"across_{f}"]["why"] == "wider than the gripper opening" for f in FRACS)
    assert by["along_0.5"]["feasible"]


# ------------------------------------------------------------------ the ceilings

def _r(rid, label=None, feasible=True, artifact=False, error=None):
    d, f = rid.split("_")
    out = {"id": rid, "dir": d, "f": float(f), "current": d == "across" and float(f) in (0.3, 0.5, 0.7), "feasible": feasible}
    if feasible and (label or error):
        out.update(label=label, artifact=artifact, error=error)
    return out


def test_the_ceiling_succeeds_when_any_row_places_and_selects_the_first_in_order():
    rows = [_r("across_0.5", "tipped"), _r("across_0.3", "placed_stable"), _r("along_0.5", "placed_stable")]
    c = ps.ceiling(rows)
    assert c["success"] and c["rows"] == ["across_0.3"] and c["labels"] == ["placed_stable"] and not c["irr"]
    assert c["attempts"] == 1 and not c["give_up"] and c["determined"] and c["n_rows"] == 3 and c["n_success"] == 2


def test_the_three_row_ceiling_reads_the_current_rows_only():
    rows = [_r("across_0.5", "tipped"), _r("along_0.5", "placed_stable")]
    c3, c10 = ps.ceiling(rows, current_only=True), ps.ceiling(rows)
    assert c10["success"] and not c3["success"] and c3["rows"] == ["across_0.5"] and c3["n_rows"] == 1


def test_without_a_success_the_selector_avoids_irreversible_damage():
    irr = IRREVERSIBLE[2]                                                 # crushed
    c = ps.ceiling([_r("across_0.5", irr), _r("across_0.3", "tipped")])
    assert not c["success"] and c["rows"] == ["across_0.3"] and not c["irr"]
    c = ps.ceiling([_r("across_0.5", irr), _r("across_0.3", IRREVERSIBLE[0])])
    assert c["irr"] and c["rows"] == ["across_0.5"]


def test_no_feasible_row_is_a_give_up():
    c = ps.ceiling([_r("across_0.5", feasible=False), _r("along_0.5", feasible=False)])
    assert c["give_up"] and c["attempts"] == 0 and not c["success"] and not c["irr"] and c["determined"] and c["rows"] == []


def test_an_artifact_or_an_error_leaves_the_ceiling_undetermined_unless_a_clean_row_succeeds():
    assert ps.ceiling([_r("across_0.5", "tipped", artifact=True), _r("along_0.5", "placed_stable")])["determined"]
    c = ps.ceiling([_r("across_0.5", "placed_stable", artifact=True), _r("along_0.5", "tipped")])
    assert not c["determined"] and not c["success"]
    assert not ps.ceiling([_r("across_0.5", error="RuntimeError: x"), _r("along_0.5", "tipped")])["determined"]
    assert ps.ceiling([_r("across_0.5", "tipped", artifact=True, feasible=False), _r("along_0.5", "tipped")])["determined"]


# ------------------------------------------------------------------ the simulations

class SimEnv(FakeEnv):
    def __init__(self, labels, fail=(), **kw):
        super().__init__(**kw)
        self.labels, self.fail, self.calls = labels, set(fail), []

    def execute(self, proposal, force, v_lift, v_transport, destination, release_height, a_transport):
        d = "across" if abs(proposal.yaw - np.pi / 2) < 1e-9 else "along"
        self.calls.append((d, proposal.s, force, v_lift, v_transport, destination, release_height, a_transport))
        if (d, proposal.s) in self.fail:
            raise RuntimeError("the sim fell over")
        return SimpleNamespace(label=self.labels[(d, proposal.s)], artifact=False, damage=None, drop_z=None, loss_phase=None,
                               trace=[{"phase": "lift", "z": 0.1}])


BASE = {"force": 12.0, "v_lift": 0.15, "v_transport": 0.5, "a_transport": 2.0, "destination": "bin", "release_height": None}


def test_each_feasible_row_runs_once_after_its_own_reset_with_the_commit_std_action():
    env = SimEnv({("across", f): "placed_stable" for f in FRACS} | {("along", f): "tipped" for f in FRACS})
    rows, resets = ps.pose_rows(env, "bin"), []
    stop = ps.simulate_rows(env, lambda: resets.append(1), rows, BASE, lambda e: "fp", "fp")
    assert stop is None and len(resets) == 8 == len(env.calls)
    assert {c[2:] for c in env.calls} == {(12.0, 0.15, 0.5, "bin", None, 2.0)}
    by = _by_id(rows)
    assert by["across_0.5"]["label"] == "placed_stable" and by["along_0.5"]["label"] == "tipped"
    assert "label" not in by["along_0.1"] and by["across_0.5"]["seconds"] >= 0.0 and by["across_0.5"]["error"] is None


def test_a_fingerprint_mismatch_stops_the_case_and_an_exception_is_the_rows_error():
    env = SimEnv({("across", f): "placed_stable" for f in FRACS} | {("along", f): "tipped" for f in FRACS}, fail={("across", 0.3)})
    rows = ps.pose_rows(env, "bin")
    assert ps.simulate_rows(env, lambda: None, rows, BASE, lambda e: "fp", "fp") is None
    assert _by_id(rows)["across_0.3"]["error"].startswith("RuntimeError: the sim fell over")
    assert _by_id(rows)["across_0.7"]["label"] == "placed_stable"
    rows = ps.pose_rows(env, "bin")
    stop = ps.simulate_rows(env, lambda: None, rows, BASE, lambda e: "other", "fp")
    assert "fingerprint" in stop and all("label" not in r for r in rows)


# ------------------------------------------------------------------ the kinds

def test_every_kind_is_one_of_the_designs_kinds():
    assert set(ps.KIND_OF.values()) <= set(ps.KINDS)
    assert ps.KIND_OF["handal__hammer_1"] == "hammers and tools" and ps.KIND_OF["ycb__sugar_box"] == "boxes"
    assert ps.KIND_OF["ycb__spam_can"] == "cans and jars" and ps.KIND_OF["hope__mayonnaise_bottle"] == "bottles"
    assert ps.KIND_OF["fruits_veggies__avocado"] == "fruit" and ps.KIND_OF["ycb__bowl"] == "bowls and containers"
    assert ps.KIND_OF["hot3d__birdhouse"] == "other" and len(ps.KIND_OF) == 95
    assert sum("hammer" in o for o in ps.KIND_OF) == 9
