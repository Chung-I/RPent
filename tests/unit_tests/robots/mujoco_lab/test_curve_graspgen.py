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

"""The GraspGen menu pilot (design 2026-11-08-graspgen-menu-pilot-design.md): GraspGenX's Panda grasps on the hull cloud
at the case's rest pose, the executability funnel to lab pinches, and the capability ceiling over the kept rows."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from robots.mujoco_lab import curve_graspgen as gg
from rpent.skills.pick_place import features
from rpent.skills.pick_place.features import PAD_HALF


def _box(l, w, h, centre=(0.0, 0.0, 0.0)):
    return np.array([[x, y, z] for x in (-l / 2, l / 2) for y in (-w / 2, w / 2) for z in (-h / 2, h / 2)]) + np.asarray(centre)


def _inside(P, V, tol=1e-9):
    """The largest signed distance of each point to the faces of the convex hull of V (<= 0: inside or on it)."""
    from scipy.spatial import ConvexHull

    eq = ConvexHull(V).equations
    return (P @ eq[:, :3].T + eq[:, 3]).max(1)


def _grasp(approach=(0.0, 0.0, -1.0), closing=(1.0, 0.0, 0.0), t=(0.0, 0.0, 0.0)):
    """A 4x4 grasp: +Z the approach, +X the closing axis (made square to it), Y = Z x X."""
    z = np.asarray(approach, dtype=float) / np.linalg.norm(approach)
    x = np.asarray(closing, dtype=float)
    x = x - (x @ z) * z
    x /= np.linalg.norm(x)
    g = np.eye(4)
    g[:3, 0], g[:3, 1], g[:3, 2], g[:3, 3] = x, np.cross(z, x), z, t
    return g


# ------------------------------------------------------------------ the cloud

def test_the_cloud_lies_on_the_hulls_surface_and_has_n_float32_points():
    V = _box(0.06, 0.04, 0.03)
    P = gg.surface_cloud([V], n=500, seed=3)
    assert P.shape == (500, 3) and P.dtype == np.float32
    assert np.abs(_inside(P.astype(float), V)).max() < 1e-6


def test_the_cloud_is_the_same_for_the_same_seed_and_differs_for_another():
    V = [_box(0.06, 0.04, 0.03)]
    assert np.array_equal(gg.surface_cloud(V, n=200, seed=1), gg.surface_cloud(V, n=200, seed=1))
    assert not np.array_equal(gg.surface_cloud(V, n=200, seed=1), gg.surface_cloud(V, n=200, seed=2))


def test_the_cloud_drops_points_inside_another_hull():
    A, B = _box(0.06, 0.04, 0.03), _box(0.06, 0.04, 0.03, (0.03, 0.0, 0.0))
    P = gg.surface_cloud([A, B], n=1000, seed=0).astype(float)
    assert ((_inside(P, A) > -1e-4) & (_inside(P, B) > -1e-4)).all()     # no point strictly inside either box


def test_the_cloud_weights_the_faces_by_area():
    V = _box(0.10, 0.01, 0.01)                       # four long faces of 10 cm2 and two ends of 1 cm2
    P = gg.surface_cloud([V], n=4000, seed=0)
    ends = (np.abs(np.abs(P[:, 0]) - 0.05) < 1e-6).mean()
    assert abs(ends - 2 / 42) < 0.02


def test_the_cloud_seed_is_fixed_per_case():
    assert gg.cloud_seed("ycb__bowl", 3) == gg.cloud_seed("ycb__bowl", 3)
    assert len({gg.cloud_seed("ycb__bowl", 3), gg.cloud_seed("ycb__bowl", 4), gg.cloud_seed("ycb__pitcher", 3)}) == 3


# ------------------------------------------------------------------ the frame and the mapping

def test_the_approach_angle_is_measured_from_straight_down():
    assert gg.approach_deg(_grasp((0, 0, -1))) == pytest.approx(0.0, abs=1e-9)
    assert gg.approach_deg(_grasp((0, 0, 1), (1, 0, 0))) == pytest.approx(180.0)
    a = math.radians(30)
    assert gg.approach_deg(_grasp((math.sin(a), 0, -math.cos(a)), (0, 1, 0))) == pytest.approx(30.0)


def test_the_mapping_moves_the_palm_origin_to_the_pad_centre_and_projects_the_closing_axis():
    a = math.radians(20)
    g = _grasp((0.0, math.sin(a), -math.cos(a)), (math.cos(math.radians(30)), math.sin(math.radians(30)), 0.0), (0.1, 0.2, 1.0))
    m = gg.map_grasp(g)
    pad = np.array([0.1, 0.2, 1.0]) + gg.DEPTH * g[:3, 2]
    assert gg.DEPTH == 0.1034 and np.allclose(m["pad"], pad)
    assert np.allclose(m["point"], pad - [0.0, 0.0, 0.004])                 # the lab's pad centre sits 4 mm above its point
    assert m["yaw"] == pytest.approx(math.atan2(g[1, 0], g[0, 0])) and m["approach_deg"] == pytest.approx(20.0)
    assert np.allclose(gg.map_grasp(g, depth=0.0)["pad"], [0.1, 0.2, 1.0])   # the other candidate: the origin between the tips


def test_the_yaw_gap_is_taken_modulo_a_half_turn():
    assert gg.yaw_gap_deg(0.0, math.pi) == pytest.approx(0.0, abs=1e-9)
    assert gg.yaw_gap_deg(0.0, math.pi / 2) == pytest.approx(90.0)
    assert gg.yaw_gap_deg(0.1, math.pi - 0.1) == pytest.approx(math.degrees(0.2))
    assert gg.yaw_gap_deg(-3.0, 3.0) == pytest.approx(math.degrees(2 * math.pi - 6.0))


# ------------------------------------------------------------------ the exact extents

def test_the_box_extent_of_a_box_hull():
    V = [_box(0.06, 0.04, 0.03)]
    lo, hi = gg.box_extent(V, np.zeros(3), np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0]), (np.inf, 0.005, 0.005))
    assert (lo, hi) == pytest.approx((-0.02, 0.02))
    assert all(np.isnan(gg.box_extent(V, np.array([0.0, 0.0, 0.5]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), (np.inf, 0.01, 0.01))))


def test_an_unbounded_box_extent_is_the_features_pinch_extent():
    rng = np.random.default_rng(0)
    hulls = [rng.normal(size=(12, 3)) * 0.03 + [0.01, -0.02, 0.0], _box(0.05, 0.02, 0.04, (0.04, 0.01, 0.01))]
    for yaw in (0.0, 0.4, 1.3, -2.2):
        pt = np.array([0.02, -0.005, 0.003])
        ex, ey = np.array([math.cos(yaw), math.sin(yaw), 0.0]), np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        got = gg.box_extent(hulls, pt + [0, 0, 0.004], ex, ey, (np.inf, PAD_HALF[1], PAD_HALF[2]))
        assert got == pytest.approx(features.pinch_extent(hulls, pt, yaw, PAD_HALF), abs=1e-7)


def _bowl_rim():
    """A bowl's two walls 16 cm apart (5 mm thick, 5 cm tall) on x = +-0.08: a rim pinch on the near wall."""
    return [_box(0.005, 0.10, 0.05, (0.08, 0.0, 0.0)), _box(0.005, 0.10, 0.05, (-0.08, 0.0, 0.0))]


def test_a_rim_pinch_is_too_wide_by_the_slab_and_five_mm_between_the_open_pads():
    hulls, pt = _bowl_rim(), np.array([0.08, 0.0, 0.01])
    assert features.pinch_width(hulls, pt, 0.0, PAD_HALF) == pytest.approx(0.165)
    assert gg.bounded_width(hulls, pt, 0.0) == pytest.approx(0.005)
    assert gg.pads_free(hulls, pt, 0.0)


def test_an_open_pad_on_the_object_is_not_free():
    pt = np.zeros(3)
    assert not gg.pads_free([_box(0.10, 0.03, 0.03)], pt, 0.0)               # 10 cm: the open pads (4.7 to 5.5 cm) land on it
    assert gg.pads_free([_box(0.09, 0.03, 0.03)], pt, 0.0)
    assert not gg.pads_free([_box(0.01, 0.03, 0.03, (0.05, 0.0, 0.3))], pt, 0.0)   # high above the pad, in its column
    assert gg.pads_free([_box(0.01, 0.03, 0.03, (0.05, 0.0, -0.05))], pt, 0.0)    # below the pad's bottom edge
    assert gg.OPEN_GAP == pytest.approx(0.047)


# ------------------------------------------------------------------ the funnel

class FunnelEnv:
    """Hulls and the lab's width, extents and reach(); reach() refuses the points in ``unreachable`` (by x)."""

    max_pinch_width = 0.09

    def __init__(self, hulls, unreachable=()):
        self.hulls, self.unreachable, self.reach_calls = hulls, set(unreachable), []

    def _hull_vertex_list(self):
        return self.hulls

    def pinch_width(self, point, yaw):
        return features.pinch_width(self.hulls, point, yaw, PAD_HALF)

    def pinch_extent(self, point, yaw):
        return features.pinch_extent(self.hulls, point, yaw, PAD_HALF)

    def column_extent(self, point, yaw):
        return features.column_extent(self.hulls, point, yaw, PAD_HALF)

    def reach(self, p, destination):
        self.reach_calls.append(round(float(p.point[0]), 4))
        return {"reason": "ik", "phase": "transport"} if round(float(p.point[0]), 4) in self.unreachable else None


def _down(x, y=0.0, z=0.0, yaw=0.0, tilt_deg=0.0):
    """A grasp whose pad centre is at (x, y, z + 0.004) (the lab point (x, y, z)), tilted about its closing axis."""
    a = math.radians(tilt_deg)
    ex = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    ey = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    z_ax = -math.cos(a) * np.array([0, 0, 1.0]) + math.sin(a) * ey
    pad = np.array([x, y, z + 0.004])
    return _grasp(z_ax, ex, pad - gg.DEPTH * z_ax)


def test_the_funnel_counts_each_step_and_keeps_rows_by_confidence():
    env = FunnelEnv([_box(0.30, 0.04, 0.03)], unreachable={0.1})
    G = np.array([_down(-0.1, yaw=math.pi / 2), _down(0.0, yaw=math.pi / 2, tilt_deg=40.0), _down(0.05, yaw=0.0),
                  _down(0.1, yaw=math.pi / 2), _down(-0.105, yaw=math.pi / 2 + 0.05), _down(0.02, yaw=math.pi / 2)])
    conf = np.array([0.9, 0.95, 0.8, 0.7, 0.6, 0.5])
    rows, c = gg.funnel(env, G, conf, "bin")
    assert c["generated"] == 6 and c["vertical"] == 5 and c["mapped"] == 5
    assert [r["id"] for r in rows] == ["g0", "g2", "g3", "g4", "g5"]          # the vertical ones, by confidence
    by = {r["id"]: r for r in rows}
    assert not by["g2"]["feasible"] and by["g2"]["why"] == "wider than the gripper opening"   # along the 30 cm axis
    assert by["g3"]["why"].startswith("unreachable") and not by["g3"]["feasible_b"]
    assert by["g4"]["feasible"] and by["g4"]["dup_of"] == "g0" and not by["g4"]["kept"]   # 5 mm and 2.9 deg from g0
    assert [r["id"] for r in rows if r["kept"]] == ["g0", "g5"]
    assert c["width"] == 4 and c["centred"] == 4 and c["reach"] == 3 and c["distinct"] == 2 and c["kept"] == 2
    assert sorted(env.reach_calls) == sorted([-0.1, 0.1, -0.105, 0.02])        # reach() once per row that passes a geometry


def test_the_funnel_variant_admits_a_rim_pinch_the_slab_rule_refuses():
    env = FunnelEnv(_bowl_rim())
    rows, c = gg.funnel(env, np.array([_down(0.08, z=0.01, yaw=0.0)]), np.array([0.7]), "bin")
    r = rows[0]
    assert not r["feasible"] and r["why"] == "wider than the gripper opening" and r["width"] == pytest.approx(0.165)
    assert r["feasible_b"] and r["width_b"] == pytest.approx(0.005) and r["pads_free"] and r["kept_b"] and not r["kept"]
    assert (c["width"], c["width_b"], c["pads_free"], c["reach_b"], c["kept_b"]) == (0, 1, 1, 1, 1)


def test_the_funnel_keeps_at_most_twenty_rows():
    env = FunnelEnv([_box(0.60, 0.04, 0.03)])
    G = np.array([_down(-0.28 + 0.02 * k, yaw=math.pi / 2) for k in range(25)])
    rows, c = gg.funnel(env, G, np.linspace(0.9, 0.1, 25), "bin")
    assert c["distinct"] == 25 and c["kept"] == 20 and [r["kept"] for r in rows] == [True] * 20 + [False] * 5


def test_dedup_needs_both_the_distance_and_the_angle():
    rows = [{"id": "a", "point": [0.0, 0, 0], "yaw": 0.0, "ok": True}, {"id": "b", "point": [0.009, 0, 0], "yaw": math.pi + 0.1, "ok": True},
            {"id": "c", "point": [0.0, 0.011, 0], "yaw": 0.0, "ok": True}, {"id": "d", "point": [0.0, 0.0, 0.005], "yaw": 0.2, "ok": True},
            {"id": "e", "point": [0.0, 0, 0], "yaw": 0.0, "ok": False}]
    assert gg.dedup(rows, "ok") == {"b": "a"}          # c is 1.1 cm away, d is 11.5 deg off, e is not a candidate


# ------------------------------------------------------------------ the frame-mapping check

def test_the_frame_check_finds_the_object_between_the_fingers_only_at_the_palm_depth():
    hulls = [_box(0.06, 0.04, 0.03)]
    g = _down(0.0, yaw=math.pi / 2)                    # pads across the 4 cm side, palm 10.34 cm above the pads
    out = gg.frame_check(hulls, np.array([g, _down(0.0, yaw=0.0, tilt_deg=45.0)]))
    assert out["n"] == 1                                # the near-vertical grasps only
    assert out["palm"]["opening_found"] == 1 and out["palm"]["width_found"] == 1 and out["palm"]["within_5mm"] == 1
    assert out["palm"]["median_abs_mm"] == pytest.approx(0.0, abs=1e-6)
    assert out["tip"]["opening_found"] == 0 and out["tip"]["within_5mm"] == 0
    row = out["rows"][0]
    assert row["palm"]["opening"] == pytest.approx(0.04) and row["palm"]["width"] == pytest.approx(0.04)


def test_the_frame_checks_opening_is_bounded_by_the_panda_aperture():
    hulls = _bowl_rim()
    g = _down(0.08, z=0.01, yaw=0.0)
    row = gg.frame_check(hulls, np.array([g]))["rows"][0]["palm"]
    assert row["opening"] == pytest.approx(0.005) and row["width"] == pytest.approx(0.165) and row["width_b"] == pytest.approx(0.005)


# ------------------------------------------------------------------ the simulations and the ceiling

class SimEnv(FunnelEnv):
    """FunnelEnv with execute(): the label of the row whose point's x is the key of ``labels``."""

    def __init__(self, hulls, labels, fail=(), **kw):
        super().__init__(hulls, **kw)
        self.labels, self.fail, self.calls = labels, set(fail), []

    def execute(self, proposal, force, v_lift, v_transport, destination, release_height, a_transport):
        x = round(float(proposal.point[0]), 4)
        self.calls.append((x, proposal.yaw, force, v_lift, v_transport, destination, release_height, a_transport))
        if x in self.fail:
            raise RuntimeError("the sim fell over")
        return SimpleNamespace(label=self.labels[x], artifact=False, damage=None, drop_z=None, loss_phase=None, max_z=0.2,
                               trace=[{"phase": "lift", "z": 0.1, "fz_joint": 3.0}])


BASE = {"force": 12.0, "v_lift": 0.15, "v_transport": 0.5, "a_transport": 2.0, "destination": "bin", "release_height": None}


def _sim_rows(env, xs, kept=None, kept_b=None):
    G = np.array([_down(x, yaw=math.pi / 2) for x in xs])
    rows, _ = gg.funnel(env, G, np.linspace(0.9, 0.5, len(xs)), "bin")
    for r, k, kb in zip(rows, kept or [r["kept"] for r in rows], kept_b or [r["kept_b"] for r in rows]):
        r["kept"], r["kept_b"] = k, kb
    return rows


def test_each_kept_row_runs_once_after_its_own_reset_at_its_own_point_and_yaw():
    env = SimEnv([_box(0.30, 0.04, 0.03)], {-0.1: "placed_stable", 0.0: "tipped", 0.1: "dropped_high"})
    rows = _sim_rows(env, [-0.1, 0.0, 0.1], kept=[True, False, True], kept_b=[True, True, False])
    resets = []
    stop = gg.simulate_rows(env, lambda: resets.append(1) or True, rows, BASE, lambda e: "fp", "fp")
    assert stop is None and len(resets) == 3 and [c[0] for c in env.calls] == [-0.1, 0.0, 0.1]
    assert all(c[1:] == (math.pi / 2, 12.0, 0.15, 0.5, "bin", None, 2.0) for c in env.calls)
    assert [r["label"] for r in rows] == ["placed_stable", "tipped", "dropped_high"]
    assert all(r["trace_sha"] and r["seconds"] >= 0 and r["artifact"] is False for r in rows)


def test_a_row_neither_rule_keeps_is_not_simulated_and_an_exception_is_the_rows_error():
    env = SimEnv([_box(0.30, 0.04, 0.03)], {-0.1: "placed_stable", 0.1: "tipped"}, fail={0.1})
    rows = _sim_rows(env, [-0.1, 0.0, 0.1], kept=[True, False, True], kept_b=[False, False, False])
    assert gg.simulate_rows(env, lambda: True, rows, BASE, lambda e: "fp", "fp") is None
    assert "label" not in rows[1] and rows[2]["error"].startswith("RuntimeError") and rows[2]["label"] is None


def test_a_fingerprint_or_hull_change_after_a_reset_stops_the_case():
    env = SimEnv([_box(0.30, 0.04, 0.03)], {-0.1: "placed_stable"})
    rows = _sim_rows(env, [-0.1])
    assert "fingerprint" in gg.simulate_rows(env, lambda: True, rows, BASE, lambda e: "other", "fp")
    sha = gg.hulls_sha(env)
    env.hulls = [_box(0.30, 0.04, 0.031)]
    assert "hulls" in gg.simulate_rows(env, lambda: True, rows, BASE, lambda e: "fp", "fp", hull_sha=sha)


def test_a_reset_that_is_not_at_rest_marks_the_row_an_artifact():
    env = SimEnv([_box(0.30, 0.04, 0.03)], {-0.1: "placed_stable"})
    rows = _sim_rows(env, [-0.1])
    gg.simulate_rows(env, lambda: False, rows, BASE, lambda e: "fp", "fp")
    assert rows[0]["artifact"] is True


def _r(rid, rank, label=None, kept=True, kept_b=True, artifact=False, error=None):
    return {"id": rid, "rank": rank, "kept": kept, "kept_b": kept_b, "label": label, "artifact": artifact, "error": error}


def test_the_ceiling_reads_each_rules_kept_rows_in_confidence_order():
    rows = [_r("g3", 0, "tipped"), _r("g1", 1, "placed_stable", kept=False), _r("g7", 2, "dropped_high"), _r("g2", 3, "placed_stable")]
    c, cb = gg.ceiling(rows, "kept"), gg.ceiling(rows, "kept_b")
    assert c["success"] and c["rows"] == ["g2"] and c["n_rows"] == 3 and c["n_success"] == 1
    assert cb["success"] and cb["rows"] == ["g1"] and cb["n_rows"] == 4 and cb["n_success"] == 2
    nos = [_r("g3", 0, "dropped_high"), _r("g1", 1, "tipped")]
    assert gg.ceiling(nos, "kept")["labels"] == ["tipped"] and not gg.ceiling(nos, "kept")["irr"]
    assert gg.ceiling([], "kept")["give_up"] and gg.ceiling([_r("g1", 0, kept=False)], "kept")["give_up"]
    und = [_r("g3", 0, artifact=True, label="tipped"), _r("g1", 1, "tipped")]
    assert not gg.ceiling(und, "kept")["determined"]


def test_the_replay_runs_the_simulated_rows_in_reverse_and_lists_what_changed():
    env = SimEnv([_box(0.30, 0.04, 0.03)], {-0.1: "placed_stable", 0.1: "tipped"})
    rows = _sim_rows(env, [-0.1, 0.0, 0.1], kept=[True, False, True], kept_b=[False, False, False])
    gg.simulate_rows(env, lambda: True, rows, BASE, lambda e: "fp", "fp")
    env.calls.clear()
    env.labels[0.1] = "dropped_high"
    rep = gg.replay_rows(env, lambda: True, rows, BASE, lambda e: "fp", "fp")
    assert [c[0] for c in env.calls] == [0.1, -0.1] and rep["rows"] == 2 and rep["stop"] is None
    assert [m["id"] for m in rep["mismatches"]] == [rows[2]["id"]]
    assert rows[2]["label"] == "tipped"                 # the first pass's record is untouched


# ------------------------------------------------------------------ one case

KEY147 = ('commit_std|{"a_transport": 2.0, "destination": "bin", "force": 14.7, "release_height": null, "s": 0.5, "v_lift": 0.15, '
          '"v_transport": 0.5}')
BASE147 = dict(BASE, force=14.7)


class CaseEnv(SimEnv):
    """A box lying along x (30 cm by 4 by 3) at the origin, with the reset, truth, proposals and pose fields the case
    reads; ``resets`` holds the reset seeds in order."""

    def __init__(self, labels, **kw):
        super().__init__([_box(0.30, 0.04, 0.03)], labels, **kw)
        self.resets, self.reset_info = [], {"at_rest": True}
        self.props = {"axis": [1.0, 0.0, 0.0], "centre": [0.0, 0.0, 0.0], "axis_lo": -0.15, "length": 0.30}
        self.sim = SimpleNamespace(obj_mat=lambda: np.eye(3), obj_pos=lambda: np.zeros(3), d=None)

    def reset(self, o, theta, seed, com_end, fixture, fill, frag):
        self.resets.append(int(seed))

    def truth(self):
        return {"mass": 0.3, "com_s": 0.6, "com_obj": [0.03, 0.0, 0.0], "fill": 0.0, "mu": 0.5}

    def nominal_mass(self, o):
        return 1.2                                                    # standard_force: 14.7 N

    def proposals(self, fractions=(0.3, 0.5, 0.7), turn=0.0):
        yaw = math.pi / 2 + float(turn)
        return [SimpleNamespace(s=float(f), point=[-0.15 + f * 0.30, 0.0, 0.0], yaw=yaw,
                                width=features.pinch_width(self.hulls, np.array([-0.15 + f * 0.30, 0.0, 0.0]), yaw, PAD_HALF))
                for f in fractions]


@pytest.fixture
def lab(monkeypatch):
    from robots.mujoco_lab import adapter, calibrate, com_gap, evpi_sim

    monkeypatch.setattr(calibrate, "case_draw", lambda seed, o, idx, m, flag, fragility=None:
                        ({"mu": 0.5}, 1, "bin", {"fill": 0.0, "frag": None}))
    monkeypatch.setattr(adapter, "object_props", lambda o: {"mass": 1.0})
    monkeypatch.setattr(com_gap, "fingerprint", lambda env: "fp")
    monkeypatch.setattr(evpi_sim, "seeds_of", lambda seed, o, idx: [seed * 1000 + idx, 777, 778])
    return CaseEnv({-0.1: "tipped", 0.0003: "placed_stable", 0.1: "placed_stable", 0.0: "placed_stable"})


def _rec1(**kw):
    return dict({"object": "ycb__sugar_box", "idx": 3, "fold": 0, "side": "test", "destination": "bin", "fingerprint": "fp",
                 "status": "ok", "thin": False, "rows": [{"id": "commit_std", "action": KEY147}]}, **kw)


def _entry(env, xs=(-0.1, 0.0003, 0.1), conf=(0.9, 0.8, 0.7), cloud_sha=None):
    import hashlib

    G = np.array([_down(x, yaw=math.pi / 2) for x in xs] + [_down(0.05, yaw=0.0, tilt_deg=60.0)])
    sha = cloud_sha or hashlib.sha256(gg.surface_cloud(env.hulls, seed=gg.cloud_seed("ycb__sugar_box", 3)).tobytes()).hexdigest()
    return {"grasps": G, "conf": np.array(list(conf) + [0.95]), "cloud_sha256": sha, "error": None}


def test_a_case_maps_the_grasps_and_simulates_the_kept_rows_and_the_gate_row(lab):
    rec = gg.graspgen_case(SimpleNamespace(env=lab), _rec1(), 3, None, _entry(lab))
    assert rec["stop"] is None and rec["error"] is None and not rec["dropped"]
    assert rec["counts"]["generated"] == 4 and rec["counts"]["vertical"] == 3 and rec["n_grasps"] == 4
    assert [r["id"] for r in rec["rows"]] == ["g0", "g1", "g2", "gate_across_0.5"]
    gate = rec["rows"][-1]
    assert gate["gate"] and not gate["kept"] and not gate["kept_b"] and gate["point"] == [0.0, 0.0, 0.0] and gate["label"] == "placed_stable"
    assert lab.resets == [3003] * 5 and [c[0] for c in lab.calls] == [-0.1, 0.0003, 0.1, 0.0]
    assert rec["base"] == BASE147 and {c[2] for c in lab.calls} == {14.7}
    assert rec["ceiling"]["success"] and rec["ceiling"]["rows"] == ["g1"] and rec["ceiling_b"]["rows"] == ["g1"]
    assert rec["com_f"] == pytest.approx(0.6) and rec["kind"] == "boxes" and not rec["upright"]
    g0, g1 = rec["rows"][0], rec["rows"][1]
    assert g0["f"] == pytest.approx(0.05 / 0.30) and g0["r_axis"] == pytest.approx(0.0, abs=1e-12)
    assert g0["h"] == pytest.approx((0.004 + 0.015) / 0.03) and g0["near"] is None
    assert g1["near"] == "across_0.5" and g1["near_current"]
    assert rec["frame"]["n"] == 3 and rec["frame"]["palm"]["within_5mm"] == 3 and rec["object_z"] == pytest.approx([-0.015, 0.015])


def test_a_cloud_that_is_not_the_generators_stops_the_case(lab):
    rec = gg.graspgen_case(SimpleNamespace(env=lab), _rec1(), 3, None, _entry(lab, cloud_sha="0" * 64))
    assert "cloud" in rec["stop"] and lab.calls == []


def test_a_case_without_grasps_or_with_a_generator_error_stops(lab):
    assert "no grasps" in gg.graspgen_case(SimpleNamespace(env=lab), _rec1(), 3, None, None)["stop"]
    e = dict(_entry(lab), error="TimeoutError: no reply")
    assert "generator" in gg.graspgen_case(SimpleNamespace(env=lab), _rec1(), 3, None, e)["stop"] and lab.resets == []


def test_a_skipped_case_is_dropped_without_a_reset(lab):
    rec = gg.graspgen_case(SimpleNamespace(env=lab), _rec1(status="error"), 3, None, _entry(lab))
    assert rec["skipped"] == "stage1 error" and rec["dropped"] and rec["rows"] == [] and lab.resets == []


def test_a_case_stops_on_a_fingerprint_or_a_base_that_is_not_stage1s(lab):
    rec = gg.graspgen_case(SimpleNamespace(env=lab), _rec1(fingerprint="other"), 3, None, _entry(lab))
    assert "fingerprint" in rec["stop"] and lab.calls == []
    rec = gg.graspgen_case(SimpleNamespace(env=lab), _rec1(rows=[{"id": "commit_std", "action": KEY147.replace("14.7", "12.0")}]),
                           3, None, _entry(lab))
    assert "commit_std" in rec["stop"] and lab.calls == []


def test_the_replay_resets_at_the_second_seed_for_every_simulated_row(lab):
    rec = gg.graspgen_case(SimpleNamespace(env=lab), _rec1(), 3, None, _entry(lab), replay=True)
    assert lab.resets == [3003] * 5 + [777] * 4 and rec["replay"]["rows"] == 4 and rec["replay"]["mismatches"] == []


def test_an_undetermined_design_ceiling_drops_the_case(lab):
    lab.fail = {0.0003, 0.1}                                          # every success raises: no clean success
    rec = gg.graspgen_case(SimpleNamespace(env=lab), _rec1(), 3, None, _entry(lab))
    assert rec["dropped"] and not rec["ceiling"]["determined"]


# ------------------------------------------------------------------ the cloud step and the request step

def test_the_cloud_of_a_case_is_the_cloud_the_case_checks(lab):
    import hashlib

    out = gg.cloud_case(SimpleNamespace(env=lab), _rec1(), 3, None)
    assert out["stop"] is None and lab.resets == [3003] and out["cloud"].shape == (gg.N_POINTS, 3)
    assert out["cloud_sha256"] == hashlib.sha256(out["cloud"].tobytes()).hexdigest() == _entry(lab)["cloud_sha256"]
    assert out["n_hulls"] == 1 and out["fingerprint"] == "fp"
    assert gg.cloud_case(SimpleNamespace(env=lab), _rec1(fingerprint="other"), 3, None)["stop"]
    skip = gg.cloud_case(SimpleNamespace(env=lab), _rec1(status="error"), 3, None)
    assert skip["skipped"] == "stage1 error" and skip["cloud"] is None


def test_the_cloud_files_and_their_index(tmp_path):
    cloud = np.arange(30, dtype=np.float32).reshape(10, 3)
    line = gg.write_cloud(str(tmp_path), {"object": "ycb__bowl", "idx": 4, "side": "test", "fold": 0}, cloud, {"n_hulls": 2})
    assert line["file"] == "ycb__bowl__4.npz" and np.array_equal(np.load(tmp_path / "ycb__bowl__4.npz")["cloud"], cloud)
    assert line["cloud_sha256"] == gg._sha_bytes(cloud) and line["n_hulls"] == 2 and line["object"] == "ycb__bowl"


def _fake_infer(cloud):
    k = 3
    return np.tile(np.eye(4, dtype=np.float32), (k, 1, 1)), np.linspace(0.9, 0.1, k).astype(np.float32), 12.5


def test_the_request_step_writes_one_grasp_file_per_cloud_and_refuses_a_second_run(tmp_path):
    from robots.mujoco_lab import graspgen_request as gr

    cdir, gdir = tmp_path / "clouds", tmp_path / "grasps"
    cdir.mkdir()
    lines = [gg.write_cloud(str(cdir), {"object": "a", "idx": i, "side": "test", "fold": 0},
                            np.full((5, 3), i, dtype=np.float32), {}) for i in (0, 1)]
    (cdir / "index.jsonl").write_text("".join(__import__("json").dumps(x) + "\n" for x in lines))
    m = gr.request_all(str(cdir), str(gdir), _fake_infer, {"gripper": "franka_panda"})
    idx = [__import__("json").loads(x) for x in (gdir / "index.jsonl").read_text().splitlines()]
    assert m["cases"] == 2 and m["errors"] == 0 and [x["k"] for x in idx] == [3, 3] and idx[0]["cloud_sha256"] == lines[0]["cloud_sha256"]
    g = np.load(gdir / idx[1]["file"])
    assert g["grasps"].shape == (3, 4, 4) and idx[1]["grasps_sha256"] == gr.sha_arrays(g["grasps"], g["conf"])
    with pytest.raises(SystemExit):
        gr.request_all(str(cdir), str(gdir), _fake_infer, {})


def test_the_request_step_records_a_failed_request_and_a_cloud_that_changed(tmp_path):
    import json

    from robots.mujoco_lab import graspgen_request as gr

    cdir = tmp_path / "clouds"
    cdir.mkdir()
    lines = [gg.write_cloud(str(cdir), {"object": "a", "idx": i, "side": "test", "fold": 0},
                            np.full((5, 3), i, dtype=np.float32), {}) for i in (0, 1)]
    lines[1]["cloud_sha256"] = "0" * 64
    (cdir / "index.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))

    def boom(cloud):
        raise TimeoutError("no reply")

    m = gr.request_all(str(cdir), str(tmp_path / "g"), boom, {})
    idx = [json.loads(x) for x in (tmp_path / "g" / "index.jsonl").read_text().splitlines()]
    assert m["errors"] == 2 and idx[0]["error"].startswith("TimeoutError") and "cloud" in idx[1]["error"]


def test_the_grasps_load_by_case_and_a_changed_file_is_an_error(tmp_path):
    import json

    from robots.mujoco_lab import graspgen_request as gr

    cdir = tmp_path / "clouds"
    cdir.mkdir()
    lines = [gg.write_cloud(str(cdir), {"object": "a", "idx": 0, "side": "test", "fold": 0}, np.zeros((5, 3), np.float32), {})]
    (cdir / "index.jsonl").write_text(json.dumps(lines[0]) + "\n")
    gr.request_all(str(cdir), str(tmp_path / "g"), _fake_infer, {})
    e = gg.load_grasps(str(tmp_path / "g"))
    assert set(e) == {("a", 0)} and e[("a", 0)]["error"] is None and e[("a", 0)]["grasps"].shape == (3, 4, 4)
    np.savez(tmp_path / "g" / "a__0.npz", grasps=np.zeros((3, 4, 4), np.float32), conf=np.zeros(3, np.float32))
    assert "hash" in gg.load_grasps(str(tmp_path / "g"))[("a", 0)]["error"]


# ------------------------------------------------------------------ the case selection, the log line and the smoke report

def _r1(o, idx, side="test"):
    return {"object": o, "idx": idx, "side": side, "fold": 0}


def test_the_cases_are_chosen_by_name_or_by_object_side_and_count():
    recs = [_r1("a", 5), _r1("a", 2), _r1("a", 9), _r1("b", 1, "validation"), _r1("b", 0, "validation"), _r1("c", 3)]
    assert [(r["object"], r["idx"]) for r in gg.select_cases(recs, objects="a,b", sides="test,validation", per_object=2)] == \
        [("a", 2), ("a", 5), ("b", 0), ("b", 1)]
    assert [(r["object"], r["idx"]) for r in gg.select_cases(recs, objects="a", sides="test")] == [("a", 2), ("a", 5), ("a", 9)]
    assert [(r["object"], r["idx"]) for r in gg.select_cases(recs, cases="c:3,a:9")] == [("a", 9), ("c", 3)]
    with pytest.raises(SystemExit):
        gg.select_cases(recs, cases="c:4")
    with pytest.raises(SystemExit):
        gg.select_cases(recs, objects="a,d", sides="test")                # an object with no case on these sides


LABELS = ("placed_stable", "tipped", "dropped_high", "dropped_low", "rolled_off", "crushed", "not_lifted", "missed")


def _srec(o, idx, side="test", secs=10.0, gate_sha=("q1", "t1"), mismatches=(), kept=(True, False), label="tipped"):
    rows = [{"id": "g0", "rank": 0, "kept": kept[0], "kept_b": True, "label": label, "seconds": 2.0, "qpos_sha": "x", "trace_sha": "y"},
            {"id": "g4", "rank": 1, "kept": kept[1], "kept_b": False, "label": None, "error": "RuntimeError: x", "seconds": 1.0},
            {"id": "gate_across_0.5", "rank": -1, "gate": True, "kept": False, "kept_b": False, "label": "placed_stable",
             "seconds": 3.0, "qpos_sha": gate_sha[0], "trace_sha": gate_sha[1]}]
    counts = {"generated": 200, "vertical": 30, "mapped": 30, "width": 20, "centred": 15, "reach": 12, "distinct": 9, "kept": 9,
              "width_b": 25, "pads_free": 22, "reach_b": 18, "distinct_b": 14, "kept_b": 14}
    frame = {"n": 30, "tol_mm": 5.0, "rows": [], "palm": {"opening_found": 28, "width_found": 27, "both": 27, "within_5mm": 20,
                                                          "median_abs_mm": 2.0},
             "tip": {"opening_found": 0, "width_found": 0, "both": 0, "within_5mm": 0, "median_abs_mm": None}}
    return {"object": o, "idx": idx, "side": side, "rows": rows, "counts": counts, "frame": frame, "seconds": secs, "skipped": None,
            "stop": None, "error": None, "dropped": False, "replay": {"rows": 3, "mismatches": list(mismatches), "stop": None},
            "ceiling": {"success": False}, "ceiling_b": {"success": True}}


def _prec(o, idx, side="test", sha=("q1", "t1")):
    return {"object": o, "idx": idx, "side": side, "rows": [{"id": "across_0.5", "label": "placed_stable", "qpos_sha": sha[0],
                                                             "trace_sha": sha[1], "seconds": 3.0}]}


def test_the_smoke_report_counts_the_funnel_both_gates_and_the_projection_and_holds_no_label():
    import json

    recs = [_srec("hope__milk_carton", 0), _srec("hope__milk_carton", 1, secs=30.0, gate_sha=("q9", "t1")),
            _srec("ycb__bowl", 0, mismatches=[{"id": "g0"}])]
    pose = [_prec("hope__milk_carton", 0), _prec("hope__milk_carton", 1), _prec("ycb__bowl", 0)]
    sm = gg.smoke_report(recs, pose, {"hope__milk_carton": 11, "ycb__bowl": 11})
    assert sm["cases"] == 3 and sm["funnel"]["hope__milk_carton"]["generated"] == 400 and sm["funnel"]["all"]["kept_b"] == 42
    assert sm["frame"]["all"]["palm"]["within_5mm"] == 60 and sm["frame"]["hope__milk_carton"]["palm"]["n"] == 60
    assert sm["frame"]["all"]["tip"]["opening_found"] == 0
    assert sm["gate"] == {"cases": 3, "equal": 2, "differ": 1, "missing": 0}
    assert sm["replay"] == {"rows": 9, "mismatches": 1, "stops": 0}
    assert sm["rows_simulated"] == 9 and sm["row_errors"] == 3
    assert sm["projection_cpu_h"] == pytest.approx((20.0 * 11 + 10.0 * 11) / 3600)
    text = json.dumps(sm) + gg.smoke_markdown(sm)
    assert not any(lab in text for lab in LABELS)                    # no early look: the report carries no label


def test_the_case_log_line_has_counts_and_no_label():
    line = gg.case_line(_srec("ycb__bowl", 0, label="crushed"))
    assert "ycb__bowl 0" in line and "rows simulated" in line and not any(lab in line for lab in LABELS)


SIM = pytest.mark.skipif(not __import__("os").environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")


@SIM
def test_a_grasp_at_a_menu_pinch_maps_to_that_pinch_on_the_real_adapter():
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    env = MuJoCoLabEnv()
    env.reset("ycb__sugar_box", seed=0)
    p = min((env.proposals((0.5,), turn=t)[0] for t in (0.0, -math.pi / 2)), key=lambda q: q.width)   # the narrower side
    assert p.width < 0.09
    ex = np.array([math.cos(p.yaw), math.sin(p.yaw), 0.0])
    pad = np.asarray(p.point) + [0.0, 0.0, gg.PAD_ABOVE]
    g = _grasp((0.0, 0.0, -1.0), ex, pad - gg.DEPTH * np.array([0.0, 0.0, -1.0]))
    rows, c = gg.funnel(env, np.array([g]), np.array([0.5]), "bin")
    r = rows[0]
    assert np.allclose(r["point"], p.point) and gg.yaw_gap_deg(r["yaw"], p.yaw) < 1e-9
    assert r["width"] == pytest.approx(p.width) and r["width_b"] == pytest.approx(p.width, abs=1e-6)
    assert r["reach"] == "ok" and r["feasible"] and r["feasible_b"] and c["kept"] == c["kept_b"] == 1
