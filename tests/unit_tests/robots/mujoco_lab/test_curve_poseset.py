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


# ------------------------------------------------------------------ the case's fields and gates

KEY = 'commit_std|{"a_transport": 2.0, "destination": "bin", "force": 12.0, "release_height": null, "s": 0.5, "v_lift": 0.15, "v_transport": 0.5}'


def test_a_stage1_rows_action_is_read_from_its_key():
    rec1 = {"rows": [{"id": "commit_std", "action": KEY}]}
    assert ps.stage1_action(rec1, "commit_std") == dict(BASE, s=0.5) and ps.stage1_action(rec1, "slow_lift") is None
    assert ps.stage1_action({"rows": None}, "commit_std") is None


def test_the_base_gate_needs_stage1s_commit_std_action_at_a_current_pinch():
    rec1 = {"rows": [{"id": "commit_std", "action": KEY}]}
    assert ps.base_gate(rec1, BASE) is None and ps.base_gate({"rows": []}, BASE) is None      # a no-row case has no row to match
    assert "commit_std" in ps.base_gate(rec1, dict(BASE, force=10.0))
    assert "pinch" in ps.base_gate({"rows": [{"id": "commit_std", "action": KEY.replace('"s": 0.5', '"s": 0.1')}]}, BASE)


def test_a_reset_that_is_not_at_rest_marks_every_row_an_artifact():
    env = SimEnv({("across", f): "placed_stable" for f in FRACS} | {("along", f): "tipped" for f in FRACS})
    rows = ps.pose_rows(env, "bin")
    assert ps.simulate_rows(env, lambda: False, rows, BASE, lambda e: "fp", "fp") is None
    assert all(r["artifact"] for r in rows if r["feasible"]) and not ps.ceiling(rows)["determined"]


def test_the_trace_hash_reads_every_field_to_nine_digits():
    a = [{"phase": "lift", "z": 0.1, "fz": 1.0, "pads": [True, False], "pad_nz": [None, 0.5]}]
    assert ps.trace_sha(a) == ps.trace_sha([dict(a[0], z=0.1 + 1e-12)])
    assert ps.trace_sha(a) != ps.trace_sha([dict(a[0], fz=1.000001)]) != ps.trace_sha([dict(a[0], pads=[True, True])])


def test_the_replay_runs_the_rows_in_reverse_on_a_copy_and_lists_what_changed():
    labels = {("across", f): "placed_stable" for f in FRACS} | {("along", f): "tipped" for f in FRACS}
    env = SimEnv(labels)
    rows = ps.pose_rows(env, "bin")
    ps.simulate_rows(env, lambda: True, rows, BASE, lambda e: "fp", "fp")
    keep = json.dumps(rows, sort_keys=True)
    env.calls.clear()
    same = ps.replay_rows(env, lambda: True, rows, BASE, lambda e: "fp", "fp")
    assert same == {"rows": 8, "mismatches": [], "stop": None} and json.dumps(rows, sort_keys=True) == keep
    assert [c[:2] for c in env.calls] == [(d, f) for d, f in reversed(ps.ROW_ORDER) if (d, f) not in (("along", 0.1), ("along", 0.9))]
    env.labels = labels | {("across", 0.3): "tipped"}
    bad = ps.replay_rows(env, lambda: True, rows, BASE, lambda e: "fp", "fp")
    assert [m["id"] for m in bad["mismatches"]] == ["across_0.3"]
    assert bad["mismatches"][0]["first"][0] == "placed_stable" and bad["mismatches"][0]["replay"][0] == "tipped"


def test_the_balance_fraction_and_the_tilt():
    props = {"axis": [1.0, 0.0, 0.0], "centre": [0.0, 0.01, 0.0], "axis_lo": -0.05, "length": 0.1}
    assert ps.balance_fraction(props, [0.02, 0.0, 0.0]) == pytest.approx(0.7)
    assert ps.tilt_deg(np.eye(3), [1.0, 0.0, 0.0]) == pytest.approx(0.0)
    R = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])            # the x axis turned onto -z
    assert ps.tilt_deg(R, [1.0, 0.0, 0.0]) == pytest.approx(90.0)


def _rows(**labels):
    """Clean simulated rows in ROW_ORDER; ``labels`` by row id (across_0_5 for across_0.5); the others not feasible."""
    out = []
    for d, f in ps.ROW_ORDER:
        rid = ps.row_id(d, f)
        lab = labels.get(rid.replace(".", "_"))
        out.append(_r(rid, lab, feasible=lab is not None))
    return out


def test_the_pose_change_class_takes_the_smallest_change_that_succeeds():
    assert ps.pose_change(_rows(across_0_5="placed_stable"), None) == "the oracle gives up"
    assert ps.pose_change(_rows(across_0_5="tipped"), 0.5) == "no pose succeeds"
    assert ps.pose_change(_rows(across_0_5="placed_stable", along_0_5="placed_stable"), 0.5) == "the oracle's pose"
    assert ps.pose_change(_rows(across_0_5="tipped", across_0_9="placed_stable", along_0_5="placed_stable"), 0.5) == \
        "fraction change only"
    assert ps.pose_change(_rows(across_0_5="tipped", along_0_5="placed_stable", along_0_3="placed_stable"), 0.5) == \
        "direction change only"
    assert ps.pose_change(_rows(across_0_5="tipped", along_0_3="placed_stable"), 0.5) == "fraction and direction change"


def test_the_nearest_row_to_the_balance_point_is_a_clean_feasible_row():
    rows = _rows(across_0_5="tipped", across_0_3="placed_stable", along_0_7="placed_stable")
    assert ps.nearest_row(rows, 0.32)["id"] == "across_0.3" and ps.nearest_row(rows, 0.62)["id"] == "along_0.7"
    assert ps.nearest_row(rows, 0.4)["id"] == "across_0.5"                      # a tie goes to the first in ROW_ORDER
    assert ps.nearest_row(_rows(), 0.5) is None


def test_the_hammer_class():
    assert ps.hammer_class(_rows()) == "no feasible pose"
    assert ps.hammer_class(_rows(across_0_5="not_lifted", along_0_5="missed")) == "not lifted by any pose"
    assert ps.hammer_class(_rows(across_0_5="not_lifted", across_0_3="dropped_high")) == "lifted, never placed"
    assert ps.hammer_class(_rows(across_0_5="not_lifted", across_0_9="placed_stable")) == "placed by a pose"


def _s2(first, label, artifact=False, seed_k=0):
    return {"first": first, "seed_k": seed_k, "label": label, "artifact": artifact}


def test_the_stage2_gate_compares_the_rows_with_commit_stds_action_at_a_pinch():
    pose = {"object": "o", "idx": 1, "base": BASE, "rows": _rows(across_0_5="tipped", across_0_3="placed_stable")}
    s1 = {"rows": [{"id": "commit_std", "action": KEY}, {"id": "regrasp_com", "action": KEY.replace("commit_std", "regrasp_com").replace('"s": 0.5', '"s": 0.3')},
                   {"id": "firm_grip", "action": KEY.replace("commit_std", "firm_grip").replace("12.0", "24.0")}]}
    s2 = {"sims": [_s2("commit_std", "tipped"), _s2("regrasp_com", "placed_stable"), _s2("firm_grip", "crushed"),
                   _s2("commit_std", "placed_stable", seed_k=1)]}
    g = ps.stage2_gate([{"pose": pose, "stage1": s1, "stage2": s2}])
    assert g == {"compared": 2, "missing": 0, "mismatches": []}
    s2["sims"][1]["label"] = "tipped"
    g = ps.stage2_gate([{"pose": pose, "stage1": s1, "stage2": s2}])
    assert g["compared"] == 2 and g["mismatches"] == [{"object": "o", "idx": 1, "row": "across_0.3", "stage2_row": "regrasp_com",
                                                       "pose": ["placed_stable", False], "stage2": ["tipped", False]}]


def test_the_stage2_gate_counts_a_matching_row_the_pose_set_did_not_simulate_as_missing():
    pose = {"object": "o", "idx": 1, "base": BASE, "rows": _rows(across_0_3="placed_stable")}
    s1 = {"rows": [{"id": "commit_std", "action": KEY}]}
    g = ps.stage2_gate([{"pose": pose, "stage1": s1, "stage2": {"sims": [_s2("commit_std", "tipped")]}}])
    assert g == {"compared": 0, "missing": 1, "mismatches": []}


def test_the_off_centre_disclosure_counts_feasible_current_rows():
    def rec(o, flags):
        rows = [dict(_r(ps.row_id("across", f), "placed_stable"), centred=c) for f, c in zip((0.5, 0.3, 0.7), flags)]
        return {"object": o, "rows": rows + [dict(_r("along_0.5", "tipped"), centred=False)]}
    d = ps.off_centre_current([rec("a", (False, False, False)), rec("a", (True, False, True)), rec("b", (True, True, True))])
    assert d == {"rows": 4, "objects": {"a": 4}, "cases_all_flagged": 1, "cases": 3}


# ------------------------------------------------------------------ one case, end to end on a fake lab

class CaseEnv(SimEnv):
    """SimEnv with the reset, truth and pose fields poseset_case reads; ``resets`` holds the reset seeds in order."""

    def __init__(self, labels, **kw):
        super().__init__(labels, **kw)
        self.resets, self.reset_info = [], {"at_rest": True}
        self.props = {"axis": [1.0, 0.0, 0.0], "centre": [0.0, 0.0, 0.0], "axis_lo": -0.03, "length": 0.06}
        self.sim = SimpleNamespace(obj_mat=lambda: np.eye(3), d=None)

    def reset(self, o, theta, seed, com_end, fixture, fill, frag):
        self.resets.append(int(seed))

    def truth(self):
        return {"mass": 0.3, "com_s": 0.6, "com_obj": [0.006, 0.0, 0.0], "fill": 0.0, "mu": 0.5}

    def nominal_mass(self, o):
        return 1.2                                                    # standard_force: 14.7 N


KEY147 = KEY.replace("12.0", "14.7")


@pytest.fixture
def lab(monkeypatch):
    from robots.mujoco_lab import adapter, calibrate, com_gap, evpi_sim

    monkeypatch.setattr(calibrate, "case_draw", lambda seed, o, idx, m, flag, fragility=None:
                        ({"mu": 0.5}, 1, "bin", {"fill": 0.0, "frag": None}))
    monkeypatch.setattr(adapter, "object_props", lambda o: {"mass": 1.0})
    monkeypatch.setattr(com_gap, "fingerprint", lambda env: "fp")
    monkeypatch.setattr(evpi_sim, "seeds_of", lambda seed, o, idx: [seed * 1000 + idx, 777, 778])
    labels = {("across", f): "tipped" for f in FRACS} | {("along", f): "placed_stable" for f in FRACS}
    return CaseEnv(labels)


def _rec1(**kw):
    return dict({"object": "ycb__sugar_box", "idx": 3, "fold": 0, "side": "validation", "destination": "bin", "fingerprint": "fp",
                 "status": "ok", "thin": False, "rows": [{"id": "commit_std", "action": KEY147}]}, **kw)


def test_a_case_resets_at_its_own_seed_before_each_row_and_reads_both_ceilings(lab):
    rec = ps.poseset_case(SimpleNamespace(env=lab), _rec1(), 3, None)
    assert lab.resets == [3003] * 9 and rec["stop"] is None and rec["error"] is None and not rec["dropped"]
    assert rec["kind"] == "boxes" and not rec["hammer"] and not rec["upright"] and rec["tilt_deg"] == pytest.approx(0.0)
    assert rec["com_f"] == pytest.approx(0.6) and rec["sim_truth"]["com_s"] == 0.6 and rec["length"] == 0.06
    assert rec["base"] == dict(BASE, force=14.7) and {c[2] for c in lab.calls} == {14.7}
    assert rec["ceiling10"]["success"] and rec["ceiling10"]["rows"] == ["along_0.5"] and not rec["ceiling3"]["success"]
    assert rec["replay"] is None and len(rec["rows"]) == 10


def test_the_replay_resets_at_the_second_seed(lab):
    rec = ps.poseset_case(SimpleNamespace(env=lab), _rec1(), 3, None, replay=True)
    assert lab.resets == [3003] * 9 + [777] * 8 and rec["replay"] == {"rows": 8, "mismatches": [], "stop": None}
    assert rec["replay_seconds"] >= 0.0


def test_a_case_stops_on_a_fingerprint_or_a_base_that_is_not_stage1s(lab):
    rec = ps.poseset_case(SimpleNamespace(env=lab), _rec1(fingerprint="other"), 3, None)
    assert "fingerprint" in rec["stop"] and lab.calls == []
    rec = ps.poseset_case(SimpleNamespace(env=lab), _rec1(rows=[{"id": "commit_std", "action": KEY}]), 3, None)
    assert "commit_std" in rec["stop"] and lab.calls == []


def test_a_skipped_case_is_dropped_without_a_reset(lab):
    rec = ps.poseset_case(SimpleNamespace(env=lab), _rec1(status="error"), 3, None)
    assert rec["skipped"] == "stage1 error" and rec["dropped"] and rec["rows"] == [] and lab.resets == []


def test_an_undetermined_ceiling_drops_the_case(lab):
    lab.fail = {("along", f) for f in FRACS}                           # every success raises: no clean success
    rec = ps.poseset_case(SimpleNamespace(env=lab), _rec1(), 3, None)
    assert rec["dropped"] and not rec["ceiling10"]["determined"] and rec["ceiling3"]["determined"]


# ------------------------------------------------------------------ the smoke report

def test_the_smoke_report_counts_both_gates_and_projects_the_first_pass():
    rows = _rows(across_0_5="tipped", across_0_3="placed_stable")
    for r in rows:
        if r["feasible"]:
            r["seconds"] = 1.0
    pose = {"object": "o", "idx": 1, "side": "validation", "skipped": None, "dropped": False, "stop": None, "error": None,
            "base": BASE, "rows": rows, "seconds": 10.0, "replay_seconds": 4.0, "upright": False,
            "replay": {"rows": 2, "mismatches": [{"id": "across_0.3"}], "stop": None},
            "ceiling10": ps.ceiling(rows), "ceiling3": ps.ceiling(rows, current_only=True)}
    s1 = {"object": "o", "idx": 1, "side": "validation", "rows": [{"id": "commit_std", "action": KEY}], "status": "ok"}
    t1 = [{"object": "t", "idx": i, "side": "test", "status": "ok"} for i in range(3)] + [{"object": "t", "idx": 9, "side": "test", "status": "error"}]
    s2 = {"object": "o", "idx": 1, "sims": [_s2("commit_std", "tipped")]}
    sm = ps.smoke_report([pose], [s2], [s1] + t1)
    assert sm["cases"] == 1 and sm["rows_simulated"] == 2 and sm["row_errors"] == 0 and sm["stops"] == []
    assert sm["replay"] == {"cases": 1, "rows": 2, "mismatches": 1} and sm["stage2_gate"]["compared"] == 1
    assert sm["run_cases"] == 3 and sm["seconds_per_case"] == 6.0 and sm["projection_cpu_h"] == pytest.approx(18.0 / 3600)
    assert sm["table"][0]["labels"] == {"across_0.5": "tipped", "across_0.3": "placed_stable"}


def test_the_smoke_markdown_states_both_gates_and_the_projection():
    sm = {"cases": 1, "dropped": 0, "case_errors": 0, "rows_simulated": 2, "row_errors": 0, "stops": [], "skipped": 0,
          "replay": {"cases": 1, "rows": 2, "mismatches": 0}, "stage2_gate": {"compared": 1, "missing": 0, "mismatches": []},
          "run_cases": 3, "seconds_per_case": 6.0, "seconds_per_case_max": 6.0, "projection_cpu_h": 0.005, "projection_cpu_h_at_max": 0.005,
          "table": [{"object": "o", "idx": 1, "upright": False, "skipped": None, "dropped": False, "stop": None,
                     "feasible": ["across_0.5"], "labels": {"across_0.5": "tipped"}, "artifacts": [], "errors": [],
                     "ceiling10": ps.ceiling([_r("across_0.5", "tipped")]), "ceiling3": ps.ceiling([_r("across_0.5", "tipped")], True),
                     "seconds": 6.0, "replay_seconds": 3.0}]}
    md = ps.smoke_markdown(sm)
    assert "2-seed gate: 1 cases, 2 rows replayed, 0 mismatches" in md and "Stage 2 gate: 1 rows compared, 0 mismatches" in md
    assert "0.01 CPU-h" in md and "| o 1 | lying |" in md and "across_0.5 tipped" in md
