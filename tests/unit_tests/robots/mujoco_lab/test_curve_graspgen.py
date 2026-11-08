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


def test_the_gate_row_runs_only_where_the_pose_set_could_run_its_across_pinch(lab):
    lab.hulls = [_box(0.30, 0.12, 0.03)]                              # 12 cm across: the across pinch is wider than the opening
    rec = gg.graspgen_case(SimpleNamespace(env=lab), _rec1(), 3, None, _entry(lab))
    gate = rec["rows"][-1]
    assert gate["gate"] and not gate["feasible"] and gate["why"] == "wider than the gripper opening" and "seconds" not in gate
    assert rec["stop"] is None and all(c[0] != 0.0 for c in lab.calls)


# ------------------------------------------------------------------ the summary's pieces (synthetic records only)

K0, K4 = "R0|10|2", "R4|10|-"
CURRENT_IDS = ("across_0.3", "across_0.5", "across_0.7")


def _g(rid, rank, label=None, kept=True, kept_b=True, f=0.5, h=0.5, r_axis=0.0, near=None, near_current=None, artifact=False, error=None):
    """A pilot row: tags and, when simulated, its outcome."""
    r = {"id": rid, "rank": rank, "kept": kept, "kept_b": kept_b, "f": f, "h": h, "r_axis": r_axis, "near": near,
         "near_current": near in CURRENT_IDS if near_current is None else near_current}
    if label is not None or error is not None:
        r.update(label=label, artifact=artifact, error=error, seconds=1.0)
    return r


def _gate(sha=("q1", "t1")):
    return {"id": "gate_across_0.5", "rank": -1, "gate": True, "feasible": True, "kept": False, "kept_b": False, "label": "tipped",
            "artifact": False, "error": None, "qpos_sha": sha[0], "trace_sha": sha[1], "seconds": 1.0}


def _prows(**labels):
    """The pose set's ten rows; ``labels`` by id (across_0_5 for across_0.5), the others not feasible."""
    from robots.mujoco_lab import curve_poseset as ps

    out = []
    for d, f in ps.ROW_ORDER:
        rid = ps.row_id(d, f)
        lab = labels.get(rid.replace(".", "_"))
        r = {"id": rid, "dir": d, "f": f, "feasible": lab is not None, "current": rid in CURRENT_IDS}
        if lab is not None:
            r.update(label=lab, artifact=False, error=None, qpos_sha="q1", trace_sha="t1", seconds=1.0)
        out.append(r)
    return out


def _ep(success=False, irr=False, give_up=False, labels=()):
    return {"success": success, "irr": irr, "give_up": give_up, "labels": list(labels), "rows": [], "attempts": len(labels)}


def _sims(key, label):
    return [] if label is None else [{"seed_k": 0, "first": "commit_std", "label": label, "artifact": False,
                                      "menu2": {"decisions": {K0: "give_up", K4: "give_up"}}, "second": {}}]


def _jcase(o="ycb__sugar_box", idx=0, rows=(), prows=None, side="test", oracle="tipped", r4="tipped", com_f=0.5, length=0.3,
           dropped=(False, False, False, False), gate=("q1", "t1"), undetermined_b=False):
    """A joined case: the pilot record ("g"), the pose-set record, the oracle record (its first attempt's label, None: it
    gives up) and the Stage 2 record (R4's first label, None: it gives up)."""
    from robots.mujoco_lab import curve_poseset as ps

    rows = list(rows) + ([_gate(gate)] if gate is not None else [])
    c, cb = gg.ceiling(rows, "kept"), gg.ceiling(rows, "kept_b")
    if undetermined_b:
        cb = dict(cb, determined=False)
    counts = dict(zip(gg.FUNNEL + gg.FUNNEL_B, (200, 30, 30, 20, 15, 15, 12, 10, 25, 22, 22, 18, 14)))
    frame = {"n": 30, "tol_mm": 5.0, "rows": [{"k": 0, "approach_deg": 5.0, "palm": {"opening": 0.06, "width": 0.062, "width_b": 0.061},
                                               "tip": {"opening": float("nan"), "width": float("nan"), "width_b": float("nan")}}],
             "palm": {"opening_found": 30, "width_found": 30, "both": 30, "within_5mm": 25, "median_abs_mm": 2.0},
             "tip": {"opening_found": 0, "width_found": 0, "both": 0, "within_5mm": 0, "median_abs_mm": None}}
    g = {"object": o, "idx": idx, "side": side, "thin": False, "kind": ps.KIND_OF.get(o, "other"), "hammer": "hammer" in o,
         "upright": False, "com_f": com_f, "length": length, "rows": rows, "counts": counts, "frame": frame, "ceiling": c, "ceiling_b": cb,
         "dropped": dropped[0] or not c["determined"], "skipped": None, "stop": None, "error": None, "seconds": 5.0}
    prows = prows if prows is not None else _prows(across_0_5="tipped")
    pose = {"object": o, "idx": idx, "side": side, "thin": False, "rows": prows, "ceiling10": ps.ceiling(prows),
            "ceiling3": ps.ceiling(prows, current_only=True), "dropped": dropped[1], "skipped": None, "stop": None, "error": None}
    orec = {"object": o, "idx": idx, "side": side, "decisions": {K0: "give_up" if oracle is None else "commit_std"}, "sims": _sims(K0, oracle),
            "dropped": dropped[2], "stop": None, "skipped": None}
    s2 = {"object": o, "idx": idx, "side": side, "thin": False, "decisions": {K0: "give_up", K4: "give_up" if r4 is None else "commit_std"},
          "sims": _sims(K4, r4), "dropped": dropped[3], "stop": None, "skipped": None}
    return {"g": g, "pose": pose, "oracle": orec, "stage2": s2}


def test_the_angle_histogram_bins_the_approach_from_straight_down():
    G = [_grasp(approach=(math.sin(math.radians(a)), 0.0, -math.cos(math.radians(a))), closing=(0.0, 1.0, 0.0))
         for a in (0.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 180.0)]
    assert gg.angle_histogram(np.array(G)) == [2, 2, 2, 2]
    assert gg.ANGLE_BINS == ("within 30° of down", "30–60°", "side (60–120°)", "from below (> 120°)")


def test_a_rows_tag_puts_an_axis_rule_pose_first_then_the_rim_then_off_the_axis():
    assert gg.row_tag(_g("a", 0, near="along_0.5", h=0.95, r_axis=0.05)) == "an axis-rule pose"
    assert gg.row_tag(_g("b", 0, h=0.85, r_axis=0.05)) == "rim or top edge"
    assert gg.row_tag(_g("c", 0, h=0.84, r_axis=0.0101)) == "off the axis"
    assert gg.row_tag(_g("d", 0, h=0.5, r_axis=0.01)) == "on the axis at another fraction or yaw"
    assert gg.TAGS == ("an axis-rule pose", "rim or top edge", "off the axis", "on the axis at another fraction or yaw")


def test_the_exact_gate_compares_the_gate_row_with_the_pose_sets_across_pinch():
    recs = [_jcase(idx=0)["g"], _jcase(idx=1, gate=("q9", "t1"))["g"], _jcase(idx=2)["g"], _jcase(idx=3, gate=None)["g"],
            _jcase(idx=4, gate=("q1", "t9"))["g"]]
    pose = [_jcase(idx=0)["pose"], _jcase(idx=1)["pose"], _jcase(idx=2, prows=_prows(across_0_3="tipped"))["pose"], _jcase(idx=3)["pose"],
            _jcase(idx=4)["pose"]]
    g = gg.exact_gate(recs, pose)
    assert {k: g[k] for k in ("cases", "equal", "differ", "missing")} == {"cases": 4, "equal": 1, "differ": 2, "missing": 1}
    assert g["differences"] == [{"object": "ycb__sugar_box", "idx": 1}, {"object": "ycb__sugar_box", "idx": 4}]


def test_the_paired_set_drops_a_case_any_arm_drops_or_an_undetermined_variant():
    js = [_jcase(idx=0), _jcase(idx=1, dropped=(True, False, False, False)), _jcase(idx=2, undetermined_b=True),
          _jcase(idx=3, dropped=(False, True, False, False)), _jcase(idx=4, dropped=(False, False, True, False)),
          _jcase(idx=5, dropped=(False, False, False, True))]
    kept, n = gg.pilot_paired(js)
    assert [j["g"]["idx"] for j in kept] == [0]
    assert n == {"dropped": 5, "graspgen": 1, "variant": 1, "pose": 1, "oracle": 1, "stage2": 1}


def test_the_views_carry_both_graspgen_ceilings_and_the_four_arms_without_the_gate_row():
    j = _jcase(rows=[_g("g0", 0, "tipped"), _g("g1", 1, "placed_stable", kept=False)], prows=_prows(across_0_5="placed_stable"),
               oracle="placed_stable", r4=None)
    v = gg.pilot_views([j], K0, K4)[0]
    assert not v["gg"]["success"] and v["ggb"]["success"] and v["c10"]["success"] and v["c3"]["success"]
    assert v["oracle"]["success"] and v["r4"]["give_up"] and [r["id"] for r in v["rows"]] == ["g0", "g1"]
    assert v["side"] == "test" and v["com_f"] == 0.5 and v["length"] == 0.3 and v["prows"][0]["id"] == "across_0.5"


def test_the_pilot_rule_is_the_two_point_rule_with_no_verdict():
    b = lambda d, lo, hi: {"cases": 88, "success": {"diff": d, "lo": lo, "hi": hi}}  # noqa: E731
    assert gg.pilot_rule(b(0.05, 0.01, 0.09)) == gg.G_ABOVE
    assert gg.pilot_rule(b(0.005, -0.01, 0.019)) == gg.G_NOT
    assert gg.pilot_rule(b(0.04, -0.005, 0.085)) == "undecided at n = 88"
    assert "10-pinch" in gg.G_ABOVE and "10-pinch" in gg.G_NOT


def test_the_pilot_block_pairs_graspgen_with_the_ten_pinch_ceiling_and_counts_the_two_by_two():
    placed, tipped = _prows(across_0_5="placed_stable"), _prows(across_0_5="tipped")
    vs = gg.pilot_views([_jcase(o="a", idx=0, rows=[_g("g0", 0, "placed_stable")], prows=placed),
                         _jcase(o="a", idx=1, rows=[_g("g0", 0, "placed_stable")], prows=tipped),
                         _jcase(o="b", idx=2, rows=[_g("g0", 0, "tipped", kept=False)], prows=tipped)], K0, K4)
    b = gg.pilot_block(vs)
    assert b["cases"] == 3 and b["objects"] == 2 and b["gg_vs_c10"]["success"]["diff"] == pytest.approx(1 / 3)
    assert b["rates"]["gg"]["success"] == pytest.approx(2 / 3) and b["rates"]["gg"]["give_up"] == pytest.approx(1 / 3)
    assert b["success_2x2"] == {"both": 1, "graspgen_only": 1, "c10_only": 0, "neither": 1}
    assert b["success_2x2_b"] == {"both": 1, "graspgen_only": 1, "c10_only": 0, "neither": 1}
    assert b["rule"] == gg.pilot_rule(b["gg_vs_c10"]) and b["rule_b"] == gg.pilot_rule(b["ggb_vs_c10"])
    assert set(b["rates"]) == set(gg.ARMS) and gg.pilot_block([])["rule"] is None


def test_the_by_object_table_has_each_objects_side_count_and_arm_rates():
    vs = gg.pilot_views([_jcase(o="a", idx=0, rows=[_g("g0", 0, "placed_stable")]), _jcase(o="b", idx=1, side="validation")], K0, K4)
    t = gg.by_object(vs)
    assert list(t) == ["a", "b"] and t["a"]["n"] == 1 and t["a"]["side"] == "test" and t["a"]["gg"]["success"] == 1.0
    assert t["b"]["side"] == "validation" and t["b"]["gg"]["give_up"] == 1.0 and set(t["a"]) == {"side", "n", *gg.ARMS}


def test_the_hammer_line_classes_each_rule_and_measures_the_placing_rows_from_the_balance_point_and_the_head():
    h1 = _jcase(o="handal__hammer_1", idx=0, com_f=0.2, length=0.3,
                rows=[_g("g0", 0, "placed_stable", f=0.25), _g("g1", 1, "not_lifted", kept=False, f=0.8)], prows=_prows(across_0_5="tipped"))
    h2 = _jcase(o="handal__hammer_7", idx=1, com_f=0.9, length=0.4, rows=[_g("g0", 0, "tipped", kept=False, f=0.6)],
                prows=_prows(across_0_5="not_lifted"))
    box = _jcase(o="ycb__sugar_box", idx=2, rows=[_g("g0", 0, "placed_stable")])
    h = gg.hammer_line(gg.pilot_views([h1, h2, box], K0, K4))
    assert h["cases"] == 2 and h["objects"] == ["handal__hammer_1", "handal__hammer_7"]
    assert h["classes"]["gg"]["placed by a row"] == 1 and h["classes"]["gg"]["no kept row"] == 1
    assert h["classes"]["ggb"]["placed by a row"] == 1 and h["classes"]["ggb"]["lifted, never placed"] == 1
    assert h["c10_classes"]["lifted, never placed"] == 1 and h["c10_classes"]["not lifted by any pose"] == 1
    assert h["placed"]["balance_m"] == [pytest.approx(0.015)] and h["placed"]["head_m"] == [pytest.approx(0.075)]
    assert sorted(h["not_placed"]["balance_m"]) == [pytest.approx(0.12), pytest.approx(0.18)]
    assert sorted(h["not_placed"]["head_m"]) == [pytest.approx(0.16), pytest.approx(0.24)]     # hammer_7's head is at its f = 1 end
    pc = h["per_case"][0]
    assert pc["placed"] == [{"id": "g0", "f": 0.25, "balance_m": pytest.approx(0.015), "head_m": pytest.approx(0.075),
                             "tag": "on the axis at another fraction or yaw", "rules": ["gg", "ggb"]}]
    assert pc["classes"] == {"gg": "placed by a row", "ggb": "placed by a row"} and pc["c10_class"] == "lifted, never placed"


def test_the_object_line_counts_the_bowls_rows_by_tag_and_rule_and_their_labels():
    bowl = _jcase(o="ycb__bowl", idx=0, rows=[_g("g0", 0, "placed_stable", kept=False, h=0.9), _g("g1", 1, "tipped")], prows=_prows())
    pitcher = _jcase(o="ycb__pitcher", idx=1, rows=[], prows=_prows())
    x = gg.object_line(gg.pilot_views([bowl, pitcher], K0, K4))
    on = "on the axis at another fraction or yaw"
    b, bb = x["ycb__bowl"]["gg"], x["ycb__bowl"]["ggb"]
    assert (b["cases"], b["cases_with_row"], b["rows"], b["clean"], b["placing"], b["cases_placed"]) == (1, 1, 1, 1, 0, 0)
    assert b["tags"] == {on: 1} and b["labels"] == {"tipped": 1} and b["placing_tags"] == {}
    assert bb["rows"] == 2 and bb["tags"] == {"rim or top edge": 1, on: 1} and bb["placing_tags"] == {"rim or top edge": 1}
    assert bb["cases_placed"] == 1 and bb["labels"] == {"placed_stable": 1, "tipped": 1}
    p = x["ycb__pitcher"]["gg"]
    assert (p["cases"], p["cases_with_row"], p["rows"]) == (1, 0, 0) and x["ycb__pitcher"]["c10_placed"] == 0
    assert list(x) == list(gg.ROW_OBJECTS)


def test_the_placing_tags_count_rows_cases_and_the_gain_cases():
    a = _jcase(o="a", idx=0, rows=[_g("g0", 0, "placed_stable", near="across_0.5"), _g("g1", 1, "placed_stable", h=0.9),
                                   _g("g2", 2, "tipped", r_axis=0.05)], prows=_prows(along_0_5="placed_stable"))   # c10 places, c3 not
    b = _jcase(o="b", idx=1, rows=[_g("g0", 0, "placed_stable", r_axis=0.05), _g("g1", 1, "placed_stable", r_axis=0.06, kept=False)])
    c = _jcase(o="b", idx=2, rows=[_g("g0", 0, "tipped")], prows=_prows(across_0_5="placed_stable"))
    t = gg.placing_tags(gg.pilot_views([a, b, c], K0, K4), "gg")
    assert t["rows"] == {"an axis-rule pose": 1, "rim or top edge": 1, "off the axis": 1}
    assert t["cases"] == {"an axis-rule pose": 1, "rim or top edge": 1, "off the axis": 1}
    assert t["by_object"] == {"a": {"an axis-rule pose": 1, "rim or top edge": 1}, "b": {"off the axis": 1}}
    assert t["gain"] == {"cases": 1, "tags": {"off the axis": 1}} and t["loss"] == {"cases": 1}
    tb = gg.placing_tags(gg.pilot_views([a, b, c], K0, K4), "ggb")
    assert tb["rows"]["off the axis"] == 2 and tb["cases"]["off the axis"] == 1


def test_the_label_gate_compares_a_kept_row_at_a_current_pinch_with_the_pose_sets_label():
    j = _jcase(rows=[_g("g0", 0, "tipped", near="across_0.5"), _g("g1", 1, "placed_stable", near="across_0.3"),
                     _g("g2", 2, "tipped", near="along_0.5"), _g("g3", 3, "tipped", near="across_0.7"),
                     _g("g4", 4, "tipped", near="across_0.5", kept=False, kept_b=False)],
               prows=_prows(across_0_5="tipped", across_0_3="tipped", along_0_5="placed_stable"))
    g = gg.label_gate(gg.pilot_views([j], K0, K4))
    assert {k: g[k] for k in ("compared", "agree", "differ", "missing")} == {"compared": 2, "agree": 1, "differ": 1, "missing": 1}
    assert g["differences"] == [{"object": "ycb__sugar_box", "idx": 0, "row": "g1", "pose_row": "across_0.3",
                                 "graspgen": "placed_stable", "pose": "tipped"}]


def test_the_funnel_table_sums_each_object_and_the_share_lost_at_the_approach_step():
    recs = [_jcase(o="a", idx=0)["g"], _jcase(o="a", idx=1)["g"], _jcase(o="b", idx=2)["g"], dict(_jcase(o="b", idx=3)["g"], counts=None)]
    t = gg.funnel_table(recs)
    assert t["a"]["generated"] == 400 and t["a"]["vertical"] == 60 and t["all"]["kept_b"] == 42 and t["b"]["cases"] == 1
    assert t["a"]["lost_at_approach"] == pytest.approx(0.85) and list(t) == ["a", "b", "all"]


# ------------------------------------------------------------------ the summary command, end to end on synthetic records

def _sha256(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _write(path, obj=None, lines=None):
    import json
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        if lines is not None:
            f.write("".join(json.dumps(x) + "\n" for x in lines))
        else:
            json.dump(obj, f)


def _done(d):
    _write(f"{d}/run_metrics.json", {"stop": None})
    open(f"{d}/run.exit", "w").write("EXIT 0\n")


def _pilot_inputs(tmp_path, cases):
    """Every input of the summary for the joined ``cases``: Stage 1, Stage 2 and its summary, the oracle arm (test runs
    with its summary, validation runs), the pose set (the same), the grasps (2 per case: straight down and from the side),
    one pilot run, the smoke report and the lock."""
    from robots.mujoco_lab.curve import STAGE1_CONFIG_KEYS, STAGE2_CONFIG_KEYS
    from robots.mujoco_lab.graspgen_request import sha_arrays

    t = str(tmp_path)
    s1, s2, gr = f"{t}/stage1/shard_0", f"{t}/stage2/shard_0", f"{t}/grasps"
    pt, pv, ot, ov, gd = f"{t}/pose/test_0", f"{t}/pose/val_0", f"{t}/oracle/test_0", f"{t}/oracle/val_0", f"{t}/pilot/unit_0"
    _write(f"{s1}/records.jsonl", lines=[{"object": c["g"]["object"], "idx": c["g"]["idx"], "side": c["g"]["side"], "status": "ok"} for c in cases])
    _write(f"{s1}/run_config.json", {k: 1 for k in STAGE1_CONFIG_KEYS})
    _write(f"{s1}/run_metrics.json", {})
    h1 = {s1: _sha256(f"{s1}/records.jsonl")}

    def by(side, part):
        return [dict(c[part], side=side) for c in cases if c["g"]["side"] == side]

    for side in ("test", "validation"):
        _write(f"{s2}/records_{side}.jsonl", lines=by(side, "stage2"))
    _write(f"{s2}/run_config.json", dict({k: 1 for k in STAGE2_CONFIG_KEYS}, stage1_records_sha256=h1, rules=["R4"]))
    _write(f"{s2}/run_metrics.json", {"stop": None})
    _write(f"{t}/stage2_summary.json", {"provenance": {"records_sha256": {f"{s2} {side}": _sha256(f"{s2}/records_{side}.jsonl")
                                                                          for side in ("test", "validation")}}})
    cfg = {"bundle": "b", "seed": 1, "stage1_records_sha256": h1}
    for d, side, rev in ((ot, "test", "o"), (ov, "validation", "o"), (pt, "test", "p"), (pv, "validation", "p")):
        _write(f"{d}/records_{side}.jsonl", lines=by(side, "oracle" if rev == "o" else "pose"))
        _write(f"{d}/run_config.json", dict(cfg, code_revision=rev))
        _done(d)
    _write(f"{t}/oracle_summary.json", {"provenance": {"oracle_records_sha256": {ot: _sha256(f"{ot}/records_test.jsonl")}, "keys": [K0, K4],
                                                       "stage2": s2, "stage1": [s1]}})
    _write(f"{t}/poseset_summary.json", {"provenance": {"runs": pt, "records_sha256": {pt: _sha256(f"{pt}/records_test.jsonl")}}})
    lines = []
    for c in cases:
        G = np.array([_grasp(), _grasp(approach=(1.0, 0.0, 0.0), closing=(0.0, 1.0, 0.0))], dtype=np.float32)
        conf = np.array([0.9, 0.5], dtype=np.float32)
        name = f"{c['g']['object']}__{c['g']['idx']}.npz"
        __import__("os").makedirs(gr, exist_ok=True)
        np.savez(f"{gr}/{name}", grasps=G, conf=conf)
        lines.append({"object": c["g"]["object"], "idx": c["g"]["idx"], "side": c["g"]["side"], "file": name, "cloud_sha256": "c",
                      "grasps_sha256": sha_arrays(G, conf), "error": None})
    _write(f"{gr}/index.jsonl", lines=lines)
    for side in ("test", "validation"):
        _write(f"{gd}/records_{side}.jsonl", lines=by(side, "g"))
    _write(f"{gd}/run_config.json", dict(cfg, code_revision="g", grasps=gr, grasps_index_sha256=_sha256(f"{gr}/index.jsonl"), replay=False,
                                         clouds_index_sha256="ci", depth=gg.DEPTH, vertical_deg=gg.VERTICAL_DEG, keep_rows=gg.KEEP_ROWS))
    _done(gd)
    _write(f"{t}/smoke_summary.json", {"replay": {"rows": 312, "mismatches": 0, "stops": 0}, "gate": {"cases": 10, "equal": 10, "differ": 0,
                                                                                                     "missing": 0}, "wandb": "w"})
    _write(f"{t}/locked.json", {"lock": {"R0": {"key": K0, "knob": 2.0}, "R4": {"key": K4, "knob": None}}})
    return SimpleNamespace(runs=gd, poseset=pt, poseset_validation=pv, poseset_summary=f"{t}/poseset_summary.json", oracle=ot,
                           oracle_validation=ov, oracle_summary=f"{t}/oracle_summary.json", stage2=s2, stage2_summary=f"{t}/stage2_summary.json",
                           locked=f"{t}/locked.json", smoke=f"{t}/smoke_summary.json", out=f"{t}/summary")


def _five_cases(gate=("q1", "t1")):
    placed = _prows(across_0_5="placed_stable")
    return [_jcase(o="handal__hammer_1", idx=0, com_f=0.2, gate=gate, prows=_prows(across_0_5="tipped", across_0_3="tipped"),
                   rows=[_g("g0", 0, "placed_stable", f=0.25, near="across_0.3"), _g("g1", 1, "tipped", kept=False, f=0.8)]),
            _jcase(o="ycb__bowl", idx=1, rows=[_g("g0", 0, "placed_stable", kept=False, h=0.9)], prows=_prows(), gate=None),
            _jcase(o="ycb__pitcher", idx=2, rows=[], prows=_prows(), gate=None),
            _jcase(o="hope__milk_carton", idx=3, rows=[_g("g0", 0, "tipped")], prows=placed, oracle="placed_stable"),
            _jcase(o="fruits_veggies__lemon1", idx=4, side="validation", rows=[_g("g0", 0, "placed_stable")], prows=placed)]


def test_the_summary_reads_every_input_and_writes_the_readings(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("WANDB_MODE", "disabled")
    a = _pilot_inputs(tmp_path, _five_cases())
    rep = gg.summary(a)
    for f in ("graspgen_summary.json", "graspgen_summary.md", "graspgen_points.png"):
        assert os.path.exists(os.path.join(a.out, f))
    R, c = rep["readings"], rep["counts"]
    assert c["cases"] == 5 and c["paired"] == {"test": 4, "validation": 1} and c["paired_dropped"]["dropped"] == 0
    assert R["2_test"]["cases"] == 4 and R["2_test"]["objects"] == 4 and R["2_validation"]["cases"] == 1
    assert R["2_test"]["gg_vs_c10"]["success"]["diff"] == pytest.approx(0.0) and R["2_test"]["ggb_vs_c10"]["success"]["diff"] == pytest.approx(0.25)
    assert R["2_test_no_thin"]["cases"] == 4 and rep["disclosures"]["thin_cases"] == 0
    assert R["1_funnel"]["all"]["generated"] == 1000 and R["1_angles"]["all"] == [5, 0, 5, 0] and R["1_angles"]["ycb__bowl"] == [1, 0, 1, 0]
    assert R["3_hammer"]["cases"] == 1 and R["4_objects"]["ycb__bowl"]["ggb"]["cases_placed"] == 1
    assert R["5_tags"]["ggb"]["gain"] == {"cases": 2, "tags": {"an axis-rule pose": 1, "rim or top edge": 1}}
    assert R["5_tags"]["gg"]["loss"] == {"cases": 1} and R["5_tags_validation"]["gg"]["rows"] == {"on the axis at another fraction or yaw": 1}
    assert list(R["2_by_object"]) == ["handal__hammer_1", "hope__milk_carton", "ycb__bowl", "ycb__pitcher", "fruits_veggies__lemon1"]
    G = rep["gates"]
    assert {k: G["exact"][k] for k in ("cases", "equal", "differ", "missing")} == {"cases": 3, "equal": 3, "differ": 0, "missing": 0}
    assert (G["label"]["compared"], G["label"]["differ"]) == (1, 1) and G["smoke"]["replay"] == {"rows": 312, "mismatches": 0, "stops": 0}
    assert rep["disclosures"]["frame"]["all"]["palm"]["n"] == 150 and rep["disclosures"]["frame"]["all"]["tip"]["opening_found"] == 0
    with pytest.raises(SystemExit):
        gg.summary(a)                                                          # an existing summary is refused


def test_an_exact_gate_difference_stops_the_summary_before_any_reading(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("WANDB_MODE", "disabled")
    a = _pilot_inputs(tmp_path, _five_cases(gate=("q9", "t1")))
    with pytest.raises(SystemExit) as e:
        gg.summary(a)
    assert e.value.code == 3 and os.path.exists(os.path.join(a.out, "graspgen_summary_stop.json"))
    assert not os.path.exists(os.path.join(a.out, "graspgen_summary.json"))


@pytest.mark.parametrize("what", ["pose", "oracle", "Stage 2", "grasps", "Stage 1"])
def test_the_summary_refuses_inputs_that_changed_since_they_were_read(tmp_path, monkeypatch, what):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    a = _pilot_inputs(tmp_path, _five_cases())
    path = {"pose": f"{a.poseset}/records_test.jsonl", "oracle": f"{a.oracle}/records_test.jsonl", "Stage 2": f"{a.stage2}/records_test.jsonl",
            "grasps": f"{tmp_path}/grasps/index.jsonl", "Stage 1": f"{tmp_path}/stage1/shard_0/records.jsonl"}[what]
    with open(path, "a") as f:
        f.write("\n")
    with pytest.raises(SystemExit) as e:
        gg.summary(a)
    assert what in str(e.value)


def test_the_summary_refuses_a_pilot_run_without_exit_0_and_a_case_an_arm_lacks(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    a = _pilot_inputs(tmp_path / "exit", _five_cases())
    open(f"{a.runs}/run.exit", "w").write("EXIT 1\n")
    with pytest.raises(SystemExit) as e:
        gg.summary(a)
    assert "EXIT 0" in str(e.value)
    cases = _five_cases()
    a = _pilot_inputs(tmp_path / "lack", cases)
    lines = open(f"{a.oracle_validation}/records_validation.jsonl").read().splitlines()
    open(f"{a.oracle_validation}/records_validation.jsonl", "w").write("".join(x + "\n" for x in lines[:-1]))
    with pytest.raises(SystemExit) as e:
        gg.summary(a)
    assert "oracle" in str(e.value) and "fruits_veggies__lemon1" in str(e.value)


def test_the_summary_opens_the_pilot_records_once(tmp_path, monkeypatch):
    import builtins
    import os

    monkeypatch.setenv("WANDB_MODE", "disabled")
    a = _pilot_inputs(tmp_path, _five_cases())
    paths = {os.path.abspath(f"{a.runs}/records_{s}.jsonl") for s in ("test", "validation")}
    real, n = builtins.open, []

    def counting(f, *x, **kw):
        n.append(os.path.abspath(str(f)) in paths)
        return real(f, *x, **kw)

    monkeypatch.setattr(builtins, "open", counting)
    gg.summary(a)
    assert sum(n) == 2


def test_the_markdown_has_a_section_per_reading_and_claims_no_verdict(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("WANDB_MODE", "disabled")
    a = _pilot_inputs(tmp_path, _five_cases())
    gg.summary(a)
    md = open(os.path.join(a.out, "graspgen_summary.md")).read()
    for sec in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6."):
        assert sec in md
    assert "no verdict" in md and "thin plastic" in md and all(b in md for b in gg.ANGLE_BINS) and "lemon1" in md.split("## 2.")[1]


def test_the_figure_is_redrawn_from_the_summary_json_alone(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("WANDB_MODE", "disabled")
    a = _pilot_inputs(tmp_path, _five_cases())
    gg.summary(a)
    out = str(tmp_path / "again.png")
    gg.figure(SimpleNamespace(summary=os.path.join(a.out, "graspgen_summary.json"), out=out))
    assert os.path.getsize(out) > 0
    with pytest.raises(SystemExit):
        gg.figure(SimpleNamespace(summary=os.path.join(a.out, "graspgen_summary.json"), out=out))
