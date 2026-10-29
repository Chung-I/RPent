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

"""Feature extraction for the pick_place success model (no sim: synthetic box geometry)."""

import numpy as np

import pytest

from rpent.skills.pick_place.features import FEATURES, PAD_HALF, Geometry, as_vector, extract, pinch_width


def _box_geometry(w=0.04, l=0.10, h=0.03, com=(0.0, 0.0, 0.0)):
    """An axis-aligned box centred at the origin: x = length l, y = width w, z = height h.
    Face centres and outward normals, each face split into a 5 x 5 grid so the pad band finds faces."""
    C, N = [], []
    hx, hy, hz = l / 2, w / 2, h / 2
    grid = np.linspace(-0.8, 0.8, 5)
    for axis, sign, (a, b) in ((0, 1, (hy, hz)), (0, -1, (hy, hz)), (1, 1, (hx, hz)), (1, -1, (hx, hz)), (2, 1, (hx, hy)), (2, -1, (hx, hy))):
        for u in grid:
            for v in grid:
                c = np.zeros(3)
                c[axis] = sign * (hx, hy, hz)[axis]
                others = [i for i in range(3) if i != axis]
                c[others[0]] = u * a
                c[others[1]] = v * b
                n = np.zeros(3)
                n[axis] = sign
                C.append(c)
                N.append(n)
    return Geometry(centres=np.array(C), normals=np.array(N), com=np.asarray(com, float), length=l,
                    base_extent=w / 2, com_height=h / 2, footprint_half_width=np.hypot(hx, hy))


COMMON = dict(mass=0.2, mu=0.8, force=20.0, v_lift=0.15, v_transport=0.5)


def test_features_order_and_count():
    assert FEATURES[:4] == ("width", "slope_deg", "lever", "com_h") and len(FEATURES) == 19


def test_box_width_and_zero_slope():
    g = _box_geometry()
    f = extract(g, point=np.zeros(3), yaw=np.pi / 2, destination="bin", **COMMON)   # pinch across y (the 4 cm side)
    assert abs(f["width"] - 0.04) < 1e-6
    assert f["slope_deg"] < 1e-6
    assert abs(f["load_ratio"] - 0.2 * 9.81 / (2 * 0.8 * 20)) < 1e-9
    assert f["length"] == 0.10 and f["visibility"] == 1.0


def test_no_faces_in_band_gives_nan_then_zero():
    g = _box_geometry()
    f = extract(g, point=np.array([0.0, 0.0, 0.5]), yaw=0.0, destination="bin", **COMMON)
    assert np.isnan(f["width"]) and np.isnan(f["slope_deg"])
    v = as_vector(f)
    assert v.shape == (len(FEATURES),) and v[0] == 0.0 and v[1] == 0.0


def test_lever_is_abs_offset_along_hand_y_and_com_h():
    g = _box_geometry(com=(0.03, 0.0, 0.01))
    f = extract(g, point=np.zeros(3), yaw=np.pi / 2, destination="bin", **COMMON)   # hand y = world x
    assert abs(f["lever"] - 0.03) < 1e-9 and abs(f["com_h"] - 0.01) < 1e-9


def test_stability_and_destination_features():
    g = _box_geometry()
    f = extract(g, point=np.zeros(3), yaw=0.0, destination="box", **COMMON)
    assert abs(f["base_ratio"] - (0.02 / 0.015)) < 1e-9
    assert abs(f["com_height_ratio"] - 0.15) < 1e-9
    assert abs(f["dest_opening_ratio"] - 0.08 / g.footprint_half_width) < 1e-9
    assert f["dest_mu"] == 0.6 and f["dest_rim_height"] == 0.12 and f["dest_has_surface"] == 1.0
    assert f["dest_release_height"] == 0.02
    gap = extract(g, point=np.zeros(3), yaw=0.0, destination="table_gap", **COMMON)
    assert gap["dest_has_surface"] == 0.0 and gap["dest_opening_ratio"] == 10.0 and gap["dest_edge_dist"] < 0
    drop = extract(g, point=np.zeros(3), yaw=0.0, destination="plate", release_height=0.065, **COMMON)
    assert drop["dest_release_height"] == 0.065


def test_visibility_passes_through():
    g = _box_geometry()
    assert extract(g, point=np.zeros(3), yaw=0.0, destination="bin", visibility=0.4, **COMMON)["visibility"] == 0.4


# ---------------------------------------------------------------- per-particle features (milestone 2)
import pytest  # noqa: E402

from rpent.skills.pick_place.features import particle_features, particle_matrix, particle_sides  # noqa: E402


def _pose(th=0.4, yaw=0.3):
    R = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
    return {"R": R.ravel().tolist(), "p": [0.10, -0.05, 0.78], "low_z": 0.75, "base_extent": 0.02, "length": 0.10,
            "pt": [0.12, -0.04, 0.765], "yaw": yaw}


def test_particle_matrix_equals_particle_features():
    rng = np.random.default_rng(0)
    base = extract(_box_geometry(), [0.0, 0.0, 0.0], 0.0, destination="plate", **COMMON)
    arrays = {"mass": rng.uniform(0.05, 1.0, 50), "mu": rng.uniform(0.1, 0.9, 50), "com_obj": rng.normal(0.0, 0.03, (50, 3))}
    pose = _pose()
    M = particle_matrix(base, arrays, pose, 20.0)
    assert M.shape == (50, len(FEATURES))
    for i in range(50):
        part = {"mass": arrays["mass"][i], "mu": arrays["mu"][i], "com_obj": arrays["com_obj"][i]}
        f = particle_features(base, part, pose, 20.0)
        assert np.allclose(M[i], as_vector(f), atol=1e-12)
        assert f["load_ratio"] == pytest.approx(part["mass"] * 9.81 / (2 * part["mu"] * 20.0))
        assert f["width"] == base["width"] and f["dest_rim_height"] == base["dest_rim_height"]


def test_particle_sides_has_a_dead_zone():
    pose = _pose(th=0.0, yaw=0.0)                    # ey = world +y; the object frame equals the world frame shifted by p
    pt = np.asarray(pose["pt"]) - np.asarray(pose["p"])
    com = np.array([pt + [0.0, 0.03, 0.0], pt + [0.0, -0.03, 0.0], pt + [0.05, 0.001, 0.0]])
    assert particle_sides({"mass": np.ones(3), "mu": np.ones(3), "com_obj": com}, pose).tolist() == [1, -1, 0]


# ---------------------------------------------------------------- the exact pinch width
def _box_vertices(w=0.04, l=0.10, h=0.03, yaw=0.0, centre=(0.0, 0.0, 0.0)):
    """The 8 corners of a box (x = length l, y = width w, z = height h), turned by yaw about z."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    corners = np.array([[x, y, z] for x in (-l / 2, l / 2) for y in (-w / 2, w / 2) for z in (-h / 2, h / 2)])
    return corners @ R.T + np.asarray(centre, dtype=float)


def _lp_width(hulls, point, yaw, pad_half=PAD_HALF):
    """Reference: per hull, two linear programs for its extreme points along the closing axis in the pad's slab."""
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull

    pt = np.asarray(point, dtype=float)
    ex = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    ey = np.array([-ex[1], ex[0], 0.0])
    pad_z, a, b = pt[2] + 0.004, pad_half[1], pad_half[2]
    S = np.array([ey, -ey, [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    s_rhs = np.array([a + ey @ pt, a - ey @ pt, pad_z + b, b - pad_z])
    lo, hi = np.inf, -np.inf
    for V in hulls:
        eq = ConvexHull(V).equations
        A, rhs = np.vstack([eq[:, :3], S]), np.concatenate([-eq[:, 3], s_rhs])
        for sign in (1.0, -1.0):
            r = linprog(sign * ex, A_ub=A, b_ub=rhs, bounds=[(None, None)] * 3, method="highs")
            if r.status != 0:
                break
            v = float(ex @ (r.x - pt))
            lo, hi = min(lo, v), max(hi, v)
    return hi - lo if hi >= lo else float("nan")


def test_exact_width_across_a_box():
    assert pinch_width([_box_vertices()], np.zeros(3), np.pi / 2) == pytest.approx(0.04, abs=1e-9)


def test_exact_width_of_a_tall_box_whose_face_centres_miss_the_pads():
    """A tall box with one triangle pair per side: every side triangle's centre lies outside the pad's band, so the
    face-centre width found nothing (Cheez-It d0 in corpus v7: 0.0 cm against 17.0 cm)."""
    from scipy.spatial import ConvexHull

    V = _box_vertices(h=0.20)
    tri = V[ConvexHull(V).simplices]
    C = tri.mean(1)
    N = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    g = Geometry(centres=C, normals=N, com=np.zeros(3), length=0.10, base_extent=0.02, com_height=0.10,
                 footprint_half_width=float(np.hypot(0.05, 0.02)), hulls=(V,))
    f = extract(g, point=np.zeros(3), yaw=np.pi / 2, destination="bin", **COMMON)
    assert f["width"] == pytest.approx(0.04, abs=1e-9)


def test_exact_width_of_two_hulls_spans_both():
    """The pads close on the outer walls of a two-wall object (a mug's or a box's walls)."""
    walls = [_box_vertices(w=0.01, l=0.05, h=0.03, centre=(0.0, y, 0.0)) for y in (-0.025, 0.025)]
    assert pinch_width(walls, np.zeros(3), np.pi / 2) == pytest.approx(0.06, abs=1e-9)


def test_exact_width_without_a_hull_in_the_slab_is_nan():
    assert np.isnan(pinch_width([_box_vertices()], np.array([0.0, 0.0, 0.5]), 0.0))


def test_exact_width_matches_a_linear_program():
    rng = np.random.default_rng(3)
    for _ in range(40):
        hulls = []
        for _ in range(int(rng.integers(1, 5))):
            from scipy.spatial import ConvexHull

            P = rng.uniform(-0.03, 0.03, size=(12, 3)) + rng.uniform(-0.02, 0.02, size=3)
            hulls.append(P[ConvexHull(P).vertices])
        point, yaw = rng.uniform(-0.01, 0.01, size=3), float(rng.uniform(-np.pi, np.pi))
        want, got = _lp_width(hulls, point, yaw), pinch_width(hulls, point, yaw)
        assert (np.isnan(want) and np.isnan(got)) or abs(got - want) < 1e-7, (want, got)


def test_feature_schemas_and_damage_sentinels():
    from rpent.skills.pick_place.features import DAMAGE_FEATURES, FEATURE_SCHEMAS, FEATURES_M2, FEATURES_M2B

    assert FEATURES_M2 == (
        "width", "slope_deg", "lever", "com_h", "mass", "mu", "load_ratio", "v_lift", "v_transport", "length",
        "visibility", "base_ratio", "com_height_ratio", "dest_mu", "dest_rim_height", "dest_opening_ratio",
        "dest_edge_dist", "dest_has_surface", "dest_release_height") and FEATURES == FEATURES_M2
    assert DAMAGE_FEATURES == ("frag_crush", "frag_bruise", "frag_break", "fillable", "crush_margin", "log_v_damage", "fill",
                               "spill_margin", "a_transport")
    assert FEATURES_M2B == FEATURES_M2 + DAMAGE_FEATURES and FEATURE_SCHEMAS == {"m2": FEATURES_M2, "m2b": FEATURES_M2B}
    f = {"mass": 0.2, "mu": 0.5}
    v = as_vector(f, FEATURES_M2B)
    assert np.array_equal(v[:19], as_vector(f)) and v[19:].tolist() == [0.0, 0.0, 0.0, 0.0, -5.0, 5.0, 0.0, 5.0, 2.0]
    w = as_vector({"crush_margin": float("nan"), "fill": 0.4, "frag_crush": 1.0}, FEATURES_M2B)
    assert w[FEATURES_M2B.index("crush_margin")] == -5.0 and w[FEATURES_M2B.index("fill")] == 0.4
    assert w[FEATURES_M2B.index("frag_crush")] == 1.0


def test_particle_matrix_follows_the_names():
    from rpent.skills.pick_place.features import FEATURES_M2B
    from tests.unit_tests.skills.synthetic import synthetic_pose, synthetic_rows

    base = dict(synthetic_rows(1)[0]["features"], fill=0.6, fillable=1.0)
    arrays = {"mass": np.array([0.1, 0.4]), "mu": np.array([0.3, 0.6]), "com_obj": np.zeros((2, 3))}
    X2 = particle_matrix(base, arrays, synthetic_pose(), 20.0)
    X = particle_matrix(base, arrays, synthetic_pose(), 20.0, FEATURES_M2B)
    assert X.shape == (2, len(FEATURES_M2B)) and np.array_equal(X[:, :19], X2)
    assert X[:, FEATURES_M2B.index("fill")].tolist() == [0.6, 0.6] and X[:, FEATURES_M2B.index("crush_margin")].tolist() == [-5.0, -5.0]


def _frag_particle(f_damage_n=None, v_bruise=None, v_break=None, fill=0.0, spill=None, mass=0.3):
    return {"mass": mass, "mu": 0.5, "com_obj": [0.0, 0.0, 0.03], "k_n_per_mm": None if f_damage_n is None else 2.0,
            "f_damage_n": f_damage_n, "v_bruise_ms": v_bruise, "v_break_ms": v_break, "fill": fill, "spill_angle_deg": spill}


def test_damage_features_per_particle():
    """f_damage_n 20 and 40 N at a 30 N squeeze: crush_margin log(30 / 20) = 0.405 and log(30 / 40) = -0.288; a spill
    angle of 20 deg at 2.0 m/s^2 (carry 2 atan(2.0 / 9.81) = 23.05 deg): spill_margin log(20 / 23.05) = -0.142; an empty
    particle reads 5. The flags come from the pose's entry, and particle_matrix agrees with particle_features."""
    from rpent.skills.pick_place.belief import ParticleBelief
    from rpent.skills.pick_place.features import FEATURES_M2B
    from tests.unit_tests.skills.synthetic import synthetic_pose

    pose = dict(synthetic_pose(), entry={"class": "crushable", "law": "cardboard", "fillable": True})
    base = extract(_box_geometry(), [0.0, 0.0, 0.0], 0.0, 0.3, 0.5, 30.0, 0.15, 0.5, "table_centre", a_transport=2.0)
    parts = [_frag_particle(20.0, v_bruise=1.2, v_break=2.0, fill=0.5, spill=20.0), _frag_particle(40.0)]
    f = [particle_features(base, p, pose, 30.0) for p in parts]
    assert f[0]["crush_margin"] == pytest.approx(0.405, abs=1e-3) and f[1]["crush_margin"] == pytest.approx(-0.288, abs=1e-3)
    assert f[0]["spill_margin"] == pytest.approx(-0.142, abs=1e-3) and f[1]["spill_margin"] == 5.0
    assert f[0]["log_v_damage"] == pytest.approx(np.log(1.2)) and f[1]["log_v_damage"] == 5.0
    assert (f[0]["frag_crush"], f[0]["frag_bruise"], f[0]["frag_break"], f[0]["fillable"]) == (1.0, 0.0, 0.0, 1.0)
    assert (f[0]["fill"], f[1]["fill"], f[0]["a_transport"]) == (0.5, 0.0, 2.0)
    X = particle_matrix(base, ParticleBelief(parts, np.ones(2)).arrays(), pose, 30.0, FEATURES_M2B)
    for j, k in enumerate(FEATURES_M2B):
        assert X[:, j] == pytest.approx([f[0][k] if np.isfinite(f[0][k]) else 0.0, f[1][k] if np.isfinite(f[1][k]) else 0.0]), k


def test_true_features_fill_the_damage_features():
    """A corpus v9 row: true_features reads its entry, theta_frag, fill, spill angle, force and carry acceleration; a v8
    row (none of them) reads the sentinels."""
    from rpent.skills.pick_place.evaluate import true_features
    from tests.unit_tests.skills.synthetic import synthetic_pose

    base = extract(_box_geometry(), [0.0, 0.0, 0.0], 0.0, 0.3, 0.5, 30.0, 0.15, 0.5, "table_centre")
    row = {"features": base, "truth": {"mass": 0.3, "mu": 0.5}, "com_obj_true": [0.0, 0.0, 0.03], "pose": synthetic_pose(),
           "force": 30.0, "a_transport": 0.5, "entry": {"class": "bruisable", "law": "fruit", "fillable": False},
           "theta_frag": {"k_n_per_mm": 2.0, "f_damage_n": 20.0, "v_bruise_ms": 1.2, "v_break_ms": None, "z": [0.0, 0.0, 0.0]},
           "fill": 0.0, "spill_angle_deg": None}
    f = true_features(row)
    assert (f["frag_bruise"], f["frag_crush"], f["fillable"]) == (1.0, 0.0, 0.0)
    assert f["crush_margin"] == pytest.approx(np.log(30 / 20)) and f["log_v_damage"] == pytest.approx(np.log(1.2))
    assert (f["a_transport"], f["spill_margin"], f["fill"]) == (0.5, 5.0, 0.0)
    v8 = {k: v for k, v in row.items() if k not in ("a_transport", "entry", "theta_frag", "fill", "spill_angle_deg")}
    g = true_features(v8)
    assert (g["crush_margin"], g["log_v_damage"], g["spill_margin"], g["a_transport"], g["frag_bruise"]) == (-5.0, 5.0, 5.0, 2.0, 0.0)
