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


"""Cavities of open containers from their visual meshes (milestone 2b), and a render to check each estimate.

A cavity is a frustum in the object frame: ``axis_obj`` (unit, out of the opening), ``bottom_obj`` (the floor's
centre), ``r_bottom_m`` and ``r_top_m`` (the radii at the floor and the rim) and ``height_m`` (floor to rim).
``estimate_mesh``:

1. For each candidate axis (the object frame's +-x, +-y, +-z), rays along the axis from above a grid over the
   footprint find the first surface. The deep hits (more than 30 % of the extent below the top) form a footprint
   (their largest connected group).
2. The rim is the highest slice across the axis in which rays from the footprint's centre, in 36 directions, meet
   the inner wall in 90 % of the directions; a circle fit to the hits recentres the axis. The floor is the first
   surface a ray down the axis meets at the centre, at least 5 mm below the rim.
3. The inner radius at 10 heights between the floor and the rim (circle fits) gives the cavity's volume (the
   midpoint rule). The frustum keeps the rim's radius and that volume: a line through the radii would overshoot
   the rim of a curved wall (a round bowl), and the rim sets the spill angle.

Steps 2 and 3 run along every axis with a footprint, and the cavity is the largest frustum that passes them. The
deepest footprint is not the cavity's: the band along a round body's outline (a flared wall seen from below, the
sides seen sideways) is deep too, but no floor lies under its rim.

``render`` draws two sections through the axis and one across it, with the frustum in red and the liquid level of
a 0.6 fill in blue; every fillable entry is checked on its render before it loads (fragility.load_classes).

    python -m robots.mujoco_lab.cavities --objects ycb__mug,ycb__bowl --out-dir robots/mujoco_lab/data/cavities
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np

AXES = tuple(np.eye(3)[i] * s for i in range(3) for s in (1.0, -1.0))
GRID = 25                         # rays per side of the footprint grid
DEEP = 0.3                        # a deep hit lies more than this share of the extent below the top
N_DIRS = 36
HIT_SHARE = 0.9                   # the share of directions that must meet the wall
N_HEIGHTS = 10
STEP = 0.001                      # m, the rim scan
DIRS = np.stack([np.cos(np.linspace(0.0, 2.0 * np.pi, N_DIRS, endpoint=False)),
                 np.sin(np.linspace(0.0, 2.0 * np.pi, N_DIRS, endpoint=False))], axis=1)


def load_mesh(object_id: str):
    """The object's visual mesh (every part) in the object frame."""
    import trimesh

    from robots.mujoco_lab.adapter import load_sim

    d = os.path.join(load_sim()["scene"].MESHES, object_id)
    meta = json.load(open(os.path.join(d, "meta.json")))
    return trimesh.util.concatenate([trimesh.load(os.path.join(d, p["obj"]), force="mesh", process=False) for p in meta["parts"]])


def basis(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors that complete the axis a to a right-handed frame (u, w, a)."""
    a = np.asarray(a, dtype=float) / np.linalg.norm(a)
    u = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    return u, np.cross(a, u)


def _first_hits(mesh, origins: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """The distance along ``direction`` to the first surface per origin (inf for a miss)."""
    origins = np.atleast_2d(origins)
    locs, ray, _ = mesh.ray.intersects_location(origins, np.tile(direction, (len(origins), 1)), multiple_hits=False)
    out = np.full(len(origins), np.inf)
    if len(ray):
        out[ray] = (locs - origins[ray]) @ direction
    return out


def _slice(mesh, a, u, w, h: float) -> np.ndarray:
    """The mesh's section by the plane across the axis at height h, as 2D segments in (u, w)."""
    from trimesh.intersections import mesh_plane

    seg = mesh_plane(mesh, plane_normal=a, plane_origin=a * h)
    return np.stack([seg @ u, seg @ w], axis=-1) if len(seg) else np.zeros((0, 2, 2))


def _ray2d(seg: np.ndarray, c: np.ndarray, dirs: np.ndarray = DIRS) -> np.ndarray:
    """The nearest segment hit per direction from c (inf where a ray meets none)."""
    if len(seg) == 0:
        return np.full(len(dirs), np.inf)
    p, e = seg[:, 0] - c, seg[:, 1] - seg[:, 0]
    dx, dy = dirs[:, 0:1], dirs[:, 1:2]
    with np.errstate(divide="ignore", invalid="ignore"):
        den = dx * e[None, :, 1] - dy * e[None, :, 0]
        t = (p[None, :, 0] * e[None, :, 1] - p[None, :, 1] * e[None, :, 0]) / den
        s = (p[None, :, 0] * dy - p[None, :, 1] * dx) / den
    ok = np.isfinite(t) & (t > 1e-9) & (s >= 0.0) & (s <= 1.0)
    return np.where(ok, t, np.inf).min(axis=1)


def _circle(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """A least-squares circle through 2D points (the algebraic fit): its centre and radius."""
    x, y = pts[:, 0], pts[:, 1]
    (D, E, F), *_ = np.linalg.lstsq(np.column_stack([x, y, np.ones_like(x)]), -(x * x + y * y), rcond=None)
    c = np.array([-D / 2.0, -E / 2.0])
    return c, float(np.sqrt(max(0.0, c @ c - F)))


def _wall(seg: np.ndarray, c: np.ndarray, refits: int = 2) -> tuple[np.ndarray, float] | None:
    """(centre, radius) of the inner wall around c in a slice: rays in N_DIRS directions, a circle fit, refitted from
    its own centre; None when the rays meet the wall in fewer than HIT_SHARE of the directions."""
    r = 0.0
    for _ in range(refits + 1):
        t = _ray2d(seg, c)
        hit = np.isfinite(t)
        if hit.mean() < HIT_SHARE:
            return None
        c, r = _circle(c + t[hit, None] * DIRS[hit])
    return c, r


def _footprint(mesh, a, u, w) -> dict[str, Any] | None:
    V = np.asarray(mesh.vertices)
    hv, pu, pw = V @ a, V @ u, V @ w
    top, extent = float(hv.max()), float(hv.max() - hv.min())
    gu, gw = np.meshgrid(np.linspace(pu.min(), pu.max(), GRID), np.linspace(pw.min(), pw.max(), GRID), indexing="ij")
    origins = gu.ravel()[:, None] * u + gw.ravel()[:, None] * w + (top + 0.01) * a
    z1 = (top + 0.01) - _first_hits(mesh, origins, -a)
    deep = np.isfinite(z1) & (z1 < top - DEEP * extent)
    from scipy.ndimage import label

    lab, n = label(deep.reshape(GRID, GRID))
    if n == 0:
        return None
    k = 1 + int(np.argmax(np.bincount(lab.ravel())[1:]))
    m = (lab == k).ravel()
    return {"top": top, "extent": extent, "centre": np.array([gu.ravel()[m].mean(), gw.ravel()[m].mean()]),
            "depth": float(top - np.median(z1[m])), "n": int(m.sum())}


def _along(mesh, a, u, w, fp: dict[str, Any]) -> dict[str, Any]:
    """Steps 2 and 3 along the axis a from its footprint; ValueError (the reason) without a cavity there."""
    c, rim, h = fp["centre"], None, fp["top"] - 0.5 * STEP
    while h > fp["top"] - fp["extent"]:
        wall = _wall(_slice(mesh, a, u, w, h), c)
        if wall is not None:
            rim, (c, r_rim) = h, wall
            break
        h -= STEP
    if rim is None:
        raise ValueError("no slice closes around the axis")
    floor = (fp["top"] + 0.01) - float(_first_hits(mesh, c[0] * u + c[1] * w + (fp["top"] + 0.01) * a, -a)[0])
    if not np.isfinite(floor) or rim - floor < 0.005:
        raise ValueError("the floor is not below the rim")
    hs, rs, cs = [], [], []
    for i in range(N_HEIGHTS):
        hi = floor + (i + 0.5) / N_HEIGHTS * (rim - floor)
        wall = _wall(_slice(mesh, a, u, w, hi), c)
        if wall is not None:
            hs.append(hi - floor)
            cs.append(wall[0])
            rs.append(wall[1])
    if len(hs) < N_HEIGHTS // 2:
        raise ValueError(f"the wall closes at {len(hs)} of {N_HEIGHTS} heights")
    cm = np.median(np.array(cs), axis=0)
    H = rim - floor
    k = 3.0 * float(np.mean(np.square(rs)))        # r_b^2 + r_b r_t + r_t^2 of the frustum with the measured volume
    r_b = max(0.0, 0.5 * (-r_rim + np.sqrt(max(0.0, 4.0 * k - 3.0 * r_rim * r_rim))))
    return {"axis_obj": [float(v) for v in a], "bottom_obj": [float(v) for v in cm[0] * u + cm[1] * w + floor * a],
            "r_bottom_m": float(r_b), "r_top_m": float(r_rim), "height_m": float(H),
            "profile": [[float(x), float(y)] for x, y in zip(hs, rs)]}


def _volume(c: dict[str, Any]) -> float:
    rb, rt = c["r_bottom_m"], c["r_top_m"]
    return np.pi * c["height_m"] * (rb * rb + rb * rt + rt * rt) / 3.0


def estimate_mesh(mesh) -> dict[str, Any]:
    """The cavity frustum of an open container's mesh (see the module docstring), plus ``profile`` [(height above the
    floor, radius)]. ValueError without a cavity, with the reason per axis."""
    found, why = [], []
    for a in AXES:
        u, w = basis(a)
        fp = _footprint(mesh, a, u, w)
        if fp is None or fp["n"] < 4:
            continue
        try:
            found.append(_along(mesh, a, u, w, fp))
        except ValueError as e:
            why.append(f"{np.round(a).astype(int).tolist()} {e}")
    if not found:
        raise ValueError("no cavity: " + ("; ".join(why) if why else "no axis has a deep footprint"))
    return max(found, key=_volume)


def estimate(object_id: str) -> dict[str, Any]:
    return estimate_mesh(load_mesh(object_id))


def render(mesh, cavity: dict[str, Any], path: str, title: str = "") -> str:
    """Two sections of the mesh through the cavity's axis and one across it at mid-depth (black), the frustum (red),
    the measured radii (red dots) and the liquid level of a 0.6 fill (blue)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from trimesh.intersections import mesh_plane

    from robots.mujoco_lab.fragility import cavity_radius, liquid

    a = np.asarray(cavity["axis_obj"], dtype=float)
    b = np.asarray(cavity["bottom_obj"], dtype=float)
    u, w = basis(a)
    rb, rt, H = cavity["r_bottom_m"], cavity["r_top_m"], cavity["height_m"]
    h6 = liquid(cavity, 0.6)["height_m"]
    r6 = float(cavity_radius(cavity, h6))
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    for k, (e1, e2, name) in enumerate(((u, w, "u"), (w, u, "w"))):
        seg = mesh_plane(mesh, plane_normal=e2, plane_origin=b)
        ax[k].add_collection(LineCollection(np.stack([(seg - b) @ e1, (seg - b) @ a], axis=-1) * 100.0, colors="k", linewidths=0.6))
        ax[k].plot(np.array([-rb, -rt]) * 100, [0, H * 100], "r-", np.array([rb, rt]) * 100, [0, H * 100], "r-", lw=1.2)
        ax[k].plot(np.array([-rb, rb]) * 100, [0, 0], "r-", np.array([-rt, rt]) * 100, [H * 100, H * 100], "r--", lw=1.2)
        ax[k].plot(np.array([-r6, r6]) * 100, [h6 * 100, h6 * 100], "b-", lw=1.5)
        for z, r in cavity.get("profile", []):
            ax[k].plot([-r * 100, r * 100], [z * 100, z * 100], "r.", ms=4)
        ax[k].autoscale()
        ax[k].set_aspect("equal")
        ax[k].set_title(f"section through the axis, along {name} (cm)")
    seg = mesh_plane(mesh, plane_normal=a, plane_origin=b + 0.5 * H * a)
    ax[2].add_collection(LineCollection(np.stack([(seg - b) @ u, (seg - b) @ w], axis=-1) * 100.0, colors="k", linewidths=0.6))
    th = np.linspace(0, 2 * np.pi, 200)
    rm = float(cavity_radius(cavity, 0.5 * H))
    ax[2].plot(rm * 100 * np.cos(th), rm * 100 * np.sin(th), "r-", lw=1.2)
    ax[2].autoscale()
    ax[2].set_aspect("equal")
    ax[2].set_title("section across the axis at mid-depth (cm)")
    fig.suptitle(f"{title}  r_bottom {rb * 100:.2f} cm, r_top {rt * 100:.2f} cm, depth {H * 100:.2f} cm, "
                 f"volume {liquid(cavity, 1.0)['volume_l'] * 1000:.0f} ml, axis {np.round(a, 2).tolist()}")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objects", required=True, help="comma-separated object ids")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    out = {}
    for oid in a.objects.split(","):
        mesh = load_mesh(oid)
        try:
            out[oid] = estimate_mesh(mesh)
        except ValueError as e:
            out[oid] = {"error": str(e)}
            print(f"{oid}: {e}", flush=True)
            continue
        render(mesh, out[oid], os.path.join(a.out_dir, f"{oid}.png"), oid)
        c = out[oid]
        print(f"{oid}: axis {np.round(c['axis_obj'], 2).tolist()} r_b {c['r_bottom_m']:.4f} r_t {c['r_top_m']:.4f} "
              f"H {c['height_m']:.4f}", flush=True)
    return out


if __name__ == "__main__":
    main()
