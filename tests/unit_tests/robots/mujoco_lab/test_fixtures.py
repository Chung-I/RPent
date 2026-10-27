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

"""Mesh fixtures: hull analysis, table layout, and the inspected-only registry (no sim)."""

import json

import numpy as np
import pytest
import trimesh

from robots.mujoco_lab.fixtures import analyse, layout, write_registry
from rpent.skills.pick_place.destinations import OBJECT_REST_XY, TABLE_HALF, load_mesh_destinations


def _box(center, size):
    return trimesh.creation.box(extents=size, transform=trimesh.transformations.translation_matrix(center))


def _save_hulls(path, meshes):
    np.savez(path, n=len(meshes), **{f"v{k}": np.asarray(m.vertices) for k, m in enumerate(meshes)},
             **{f"f{k}": np.asarray(m.faces) for k, m in enumerate(meshes)})


def _cup_dir(tmp_path, inner=0.05, wall=0.08, t=0.005, closed=False):
    d = tmp_path / "fx__cup"
    d.mkdir()
    parts = [_box((0, 0, t / 2), (2 * inner + 4 * t, 2 * inner + 4 * t, t))]          # floor
    for dx, dy, sx, sy in ((1, 0, t, 2 * inner + 4 * t), (-1, 0, t, 2 * inner + 4 * t), (0, 1, 2 * inner + 4 * t, t), (0, -1, 2 * inner + 4 * t, t)):
        parts.append(_box((dx * (inner + t), dy * (inner + t), t + wall / 2), (sx, sy, wall)))
    if closed:
        parts.append(_box((0, 0, t + wall + t / 2), (2 * inner + 4 * t, 2 * inner + 4 * t, t)))
    _save_hulls(d / "hulls.npz", parts)
    return d


def test_analyse_primitive_cup(tmp_path):
    a = analyse(str(_cup_dir(tmp_path)))
    assert a["kind"] == "cavity" and a["open_top"] is True
    assert abs(a["opening_half_width"] - 0.05) < 0.006
    assert abs(a["rim_height"] - 0.08) < 0.006
    assert abs(a["surface_z"] - 0.005) < 0.004
    assert abs(a["footprint"][0] - 0.12) < 0.005 and abs(a["footprint"][1] - 0.12) < 0.005   # inner 10 cm plus the walls


def test_closed_box_fails_open_top(tmp_path):
    a = analyse(str(_cup_dir(tmp_path, closed=True)))
    assert a["open_top"] is False


def test_shallow_dish_is_rim_and_slab_is_surface(tmp_path):
    a = analyse(str(_cup_dir(tmp_path, inner=0.1, wall=0.015)))
    assert a["kind"] == "rim"
    d = tmp_path / "fx__slab"
    d.mkdir()
    _save_hulls(d / "hulls.npz", [_box((0, 0, 0.002), (0.2, 0.12, 0.004))])
    assert analyse(str(d))["kind"] == "surface"


def test_layout_fits_table_and_clears_object():
    for fp in ((0.07, 0.07), (0.30, 0.30), (0.40, 0.30), (0.17, 0.10)):
        lay = layout({"footprint": fp})
        x, y = lay["xy"]
        ox, oy = lay["object_rest_xy"]
        hx, hy = fp[0] / 2, fp[1] / 2
        assert abs(x) + hx <= TABLE_HALF[0] - 0.02 + 1e-9 and abs(y) + hy <= TABLE_HALF[1] - 0.02 + 1e-9, fp
        box = (ox - 0.19, ox + 0.19, oy - 0.09, oy + 0.09)                     # the longest object plus 2 cm
        assert not (x - hx < box[1] and x + hx > box[0] and y - hy < box[3] and y + hy > box[2]), fp
    assert layout({"footprint": (0.07, 0.07)})["object_rest_xy"] == OBJECT_REST_XY


def test_registry_and_loader_skip_uninspected_and_rejected(tmp_path):
    cup = _cup_dir(tmp_path)
    out = tmp_path / "fixtures.json"
    entries = write_registry([(str(cup), "cup", "ceramic")], str(out))
    assert entries[0]["id"] == "cup" and entries[0]["fragile"] is True and entries[0]["mu"] == 0.4
    assert entries[0]["inspected"] is False
    assert load_mesh_destinations(str(out)) == {}
    data = json.load(open(out))
    data["fixtures"][0]["inspected"] = True
    data["fixtures"].append(dict(data["fixtures"][0], id="bridged", inspected=True, rejected="a hull bridges the cavity"))
    json.dump(data, open(out, "w"))
    loaded = load_mesh_destinations(str(out))
    assert set(loaded) == {"cup"} and loaded["cup"]["kind"] == "cavity" and loaded["cup"]["mesh_dir"] == str(cup)
    assert set(loaded["cup"]) >= {"id", "kind", "xy", "surface_z", "mu", "rim_height", "opening_half_width", "fragile", "object_rest_xy"}


def test_reexport_drops_the_approval(tmp_path):
    cup = _cup_dir(tmp_path)
    out = tmp_path / "fixtures.json"
    write_registry([(str(cup), "cup", "ceramic")], str(out))
    data = json.load(open(out))
    data["fixtures"][0]["inspected"] = True
    json.dump(data, open(out, "w"))
    assert write_registry([(str(cup), "cup", "ceramic")], str(out))[0]["inspected"] is True      # same hulls: approval kept
    parts = [_box((0, 0, 0.005), (0.12, 0.12, 0.01))] + [_box((dx * 0.055, dy * 0.055, 0.045), (0.01 if dx else 0.12, 0.01 if dy else 0.12, 0.08)) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    _save_hulls(cup / "hulls.npz", parts)                                                            # re-exported: same count, different hulls
    assert write_registry([(str(cup), "cup", "ceramic")], str(out))[0]["inspected"] is False
