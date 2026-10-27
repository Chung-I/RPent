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

"""Mesh fixtures: measure a decomposed container's floor, rim and opening from its hulls, lay it
out on the table clear of the object, and write the registry.

A registry entry is loaded by the skill only when a person has looked at the hull sheet and set
``inspected: true`` (and no ``rejected`` reason): a convex decomposition can bridge or fill a
cavity, and no ray test catches every such case.

    python -m robots.mujoco_lab.fixtures --meshes <sim>/meshes --out data/fixtures.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import trimesh

from rpent.skills.pick_place.destinations import OBJECT_REST_XY, TABLE_HALF, TABLE_TOP

GRID_M = 0.01
RIM_SURFACE_M = 0.01
RIM_CAVITY_M = 0.04
FLOOR_TOL_M = 0.01
MARGIN_M = 0.02
OBJECT_HALF = (0.17, 0.07)                     # the longest object's half-extents at rest
SMALL_SLOT = (0.21, 0.12)                      # the plate's slot: fixtures up to 28 x 22 cm
LARGE_REST_XY = (-0.12, -0.16)                 # the object moves to the -y side for a wide fixture
MU_BY_MATERIAL = {"ceramic": 0.4, "glass": 0.4, "wood": 0.6, "metal": 0.5, "steel": 0.5, "metal_can": 0.5}
FRAGILE = {"ceramic", "glass"}


def load_hulls(mesh_dir: str) -> list[trimesh.Trimesh]:
    z = np.load(os.path.join(mesh_dir, "hulls.npz"))
    return [trimesh.Trimesh(z[f"v{k}"], z[f"f{k}"], process=False) for k in range(int(z["n"]))]


def _height_map(hulls: list[trimesh.Trimesh], xs: np.ndarray, ys: np.ndarray, z_top: float) -> np.ndarray:
    """Highest hull surface under each (x, y) seen from above; -inf where nothing is below."""
    mesh = trimesh.util.concatenate(hulls)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    origins = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z_top + 0.05)], axis=1)
    dirs = np.tile([0.0, 0.0, -1.0], (len(origins), 1))
    locs, idx, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=True)
    H = np.full(X.size, -np.inf)
    for p, i in zip(locs, idx):
        H[i] = max(H[i], p[2])
    return H.reshape(X.shape)


def analyse(mesh_dir: str) -> dict[str, Any]:
    """Floor height, rim height, opening half-width, kind and open-top flag, in the fixture frame
    shifted so its lowest point is at z = 0."""
    hulls = load_hulls(mesh_dir)
    V = np.concatenate([h.vertices for h in hulls])
    z0 = V[:, 2].min()
    hulls = [trimesh.Trimesh(h.vertices - [0, 0, z0], h.faces, process=False) for h in hulls]
    V = V - [0, 0, z0]
    lo, hi = V.min(0), V.max(0)
    top = float(hi[2])
    cx, cy = float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)
    xs = np.arange(lo[0] + GRID_M / 2, hi[0], GRID_M)
    ys = np.arange(lo[1] + GRID_M / 2, hi[1], GRID_M)
    H = _height_map(hulls, xs, ys, top)
    ix, iy = int(np.argmin(np.abs(xs - cx))), int(np.argmin(np.abs(ys - cy)))
    floor = float(H[ix, iy]) if np.isfinite(H[ix, iy]) else 0.0
    open_top = floor < top - RIM_SURFACE_M
    rim_height = max(0.0, top - floor)
    interior = np.isfinite(H) & (H < floor + FLOOR_TOL_M)

    def chord(line: np.ndarray, centre: int, coords: np.ndarray) -> float:
        a = b = centre
        while a - 1 >= 0 and line[a - 1]:
            a -= 1
        while b + 1 < len(line) and line[b + 1]:
            b += 1
        return float(coords[b] - coords[a] + GRID_M) if line[centre] else 0.0

    chord_x = chord(interior[:, iy], ix, xs)
    chord_y = chord(interior[ix, :], iy, ys)
    opening = min(chord_x, chord_y) / 2
    kind = "surface" if rim_height < RIM_SURFACE_M else ("rim" if rim_height < RIM_CAVITY_M else "cavity")
    return {
        "surface_z": floor, "rim_height": float(rim_height), "opening_half_width": float(opening), "kind": kind,
        "open_top": bool(open_top), "footprint": (float(hi[0] - lo[0]), float(hi[1] - lo[1])), "centre_xy": (cx, cy),
        "z_offset": float(-z0), "hull_count": len(hulls),
    }


def layout(entry: dict[str, Any]) -> dict[str, Any]:
    """Where the fixture stands and where the object rests so neither overlaps the other or leaves
    the table: small fixtures take the plate's slot; wide ones sit along +y with the object moved to -y."""
    hx, hy = entry["footprint"][0] / 2, entry["footprint"][1] / 2
    if hx <= 0.14 and hy <= 0.11:
        return {"xy": SMALL_SLOT, "object_rest_xy": OBJECT_REST_XY}
    if hx > TABLE_HALF[0] - MARGIN_M or hy > 0.15:
        raise ValueError(f"fixture footprint {entry['footprint']} does not fit the table beside an object")
    return {"xy": (0.0, TABLE_HALF[1] - MARGIN_M - hy), "object_rest_xy": LARGE_REST_XY}


def entry_for(mesh_dir: str, name: str, material: str) -> dict[str, Any]:
    a = analyse(mesh_dir)
    lay = layout(a)
    return {
        "id": name, "kind": a["kind"], "xy": lay["xy"], "object_rest_xy": lay["object_rest_xy"],
        "surface_z": TABLE_TOP + a["surface_z"], "mu": MU_BY_MATERIAL.get(material, 0.5), "rim_height": a["rim_height"],
        "opening_half_width": a["opening_half_width"] if a["kind"] != "surface" else None, "fragile": material in FRAGILE,
        "material": material, "mesh_dir": mesh_dir, "footprint": a["footprint"], "centre_xy": a["centre_xy"],
        "z_offset": a["z_offset"], "open_top": a["open_top"], "hull_count": a["hull_count"],
        "inspected": False, "rejected": None if a["open_top"] else "closed top: the centre ray does not reach below the rim",
    }


def write_registry(items: list[tuple[str, str, str]], out_path: str) -> list[dict[str, Any]]:
    entries = [entry_for(d, n, m) for d, n, m in items]
    existing = {}
    if os.path.exists(out_path):
        try:
            existing = {e["id"]: e for e in json.load(open(out_path)).get("fixtures", [])}
        except (OSError, ValueError):
            existing = {}
    for e in entries:                       # keep a person's verdict across re-runs
        old = existing.get(e["id"])
        if old and old.get("hull_count") == e["hull_count"]:
            e["inspected"], e["rejected"] = old.get("inspected", False), old.get("rejected", e["rejected"])
    with open(out_path, "w") as f:
        json.dump({"fixtures": entries}, f, indent=1)
    return entries


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meshes", required=True, help="the sim's meshes/ folder (fx__<name> subfolders)")
    ap.add_argument("--spec", required=True, help="JSON list of [name, material] pairs, or the sim's objects.py FIXTURE_USDS")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spec = json.load(open(a.spec)) if a.spec.endswith(".json") else None
    if spec is None:
        import importlib
        import sys

        sys.path.insert(0, os.path.dirname(a.meshes))
        spec = [(f["name"], f["material"]) for f in importlib.import_module("objects").FIXTURE_USDS]
    items = [(os.path.join(a.meshes, f"fx__{name}"), name, material) for name, material in spec
             if os.path.exists(os.path.join(a.meshes, f"fx__{name}", "hulls.npz"))]
    for e in write_registry(items, a.out):
        print(f"{e['id']}: {e['kind']} opening {e['opening_half_width']} rim {e['rim_height']:.3f} hulls {e['hull_count']} open_top {e['open_top']}")


if __name__ == "__main__":
    main()
