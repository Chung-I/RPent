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

from rpent.skills.pick_place.features import FEATURES, Geometry, as_vector, extract


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
