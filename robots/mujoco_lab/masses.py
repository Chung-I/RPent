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


"""The lab's real-object masses (generator v11): ``data/masses_v11b.json``, the frozen table of the real-masses study
(daily-logs experiments/2026-11-03-real-masses), pinned by its sha256. Under generator v11 ``adapter.object_props``
gives each object its real mass (``real_props``): every hull mass times real / props.json's mass, so the CoM and the
ratio of the part densities stay as they were. Seven tools carry their own hull masses instead (table v11b: a hammer's
head takes its rated head weight and the handle the rest; the nylon potato masher and spatula have one density); their
dense flags follow from them as in the lab's prepare.py. Under v11 every object's CoM and com_s are the hull-mass CoM
of the hulls' volume centroids, the sim's (``volume_centroid``): prepare.py took trimesh's surface centroid per hull,
which misses the sim's CoM by up to 14 mm (hammer_1), so the sim did not reach a drawn com_s. The table has the 95
objects whose mesh compiles (the pomegranate's does not: it has no row). ``masses_v11.json``, the first table (the
totals alone), made no corpus."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import numpy as np

MASSES_V11 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "masses_v11b.json")
MASSES_V11_SHA256 = "fcfab59ef2ee9168f83f93f9b2017dd1ca6f663a954fbc4b5b5a869a44287ade"
NOMINAL_ATOL = 5e-5          # the table keeps the v10 nominal to 0.1 g
_TABLES: dict[str, dict[str, Any]] = {}


def load_masses(path: str | None = None) -> dict[str, Any]:
    """The v11 table (cached per path); a file whose sha256 is not the pinned one is an error."""
    path = path or MASSES_V11
    if path not in _TABLES:
        raw = open(path, "rb").read()
        if hashlib.sha256(raw).hexdigest() != MASSES_V11_SHA256:
            raise ValueError(f"{path}: its sha256 is not the pinned v11 table's {MASSES_V11_SHA256}")
        _TABLES[path] = json.loads(raw)
    return _TABLES[path]


def volume_centroid(vertices: Any) -> np.ndarray:
    """The volume centroid of the convex hull of ``vertices``: MuJoCo's CoM of a mesh given by its vertices alone (it
    takes their convex hull). Each hull face spans a tetrahedron with an inner point."""
    from scipy.spatial import ConvexHull

    v = np.asarray(vertices, dtype=float)
    hull = ConvexHull(v)
    p0 = v[hull.vertices].mean(0)
    tri = v[hull.simplices] - p0
    vol = np.abs(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))) / 6.0
    return p0 + (tri.sum(1) / 4.0 * vol[:, None]).sum(0) / vol.sum()


def real_props(object_id: str, props: dict[str, Any], centroids: Any) -> dict[str, Any]:
    """``props`` (props.json of ``object_id``) at the real mass: the table's mass, every hull mass times real /
    props["mass"], or for a tool the table's hull masses (one per hull, summing to the mass) with the dense flags and
    heavy end that follow from them; the CoM and com_s of the hull masses at ``centroids``, the hulls' volume centroids
    (adapter.hull_centroids). The table's v10 nominal must be props.json's mass, so a changed mesh cannot change the
    factor."""
    entry = load_masses()["objects"].get(object_id)
    if entry is None:
        raise ValueError(f"{object_id} has no real mass in {os.path.basename(MASSES_V11)} (generator v11)")
    nominal = float(props["mass"])
    if abs(float(entry["nominal_kg_v10"]) - nominal) > NOMINAL_ATOL:
        raise ValueError(f"{object_id}: props.json's mass {nominal} is not the table's v10 nominal {entry['nominal_kg_v10']}")
    c = np.asarray(centroids, dtype=float)
    if c.shape != (len(props["hull_mass"]), 3):
        raise ValueError(f"{object_id}: {c.shape} hull centroids for {len(props['hull_mass'])} hulls")
    out = {**props, "mass": float(entry["mass_kg"])}
    if "hull_mass_kg" in entry:
        m = np.asarray(entry["hull_mass_kg"], dtype=float)
        if len(m) != len(props["hull_mass"]) or abs(float(m.sum()) - float(entry["mass_kg"])) > 1e-9:
            raise ValueError(f"{object_id}: the table's hull masses ({len(m)}, sum {m.sum()}) do not fit its mesh "
                             f"({len(props['hull_mass'])} hulls) and its mass {entry['mass_kg']}")
        head = set(entry["head_hulls"])
        out.update(hull_mass=m.tolist(), dense=[i in head for i in range(len(m))], heavy_end=entry["heavy_end"])
    else:
        k = float(entry["mass_kg"]) / nominal
        out["hull_mass"] = [float(m) * k for m in props["hull_mass"]]
    m = np.asarray(out["hull_mass"])
    com = (c * m[:, None]).sum(0) / m.sum()
    out["com"] = com.tolist()
    out["com_s"] = float(((com - np.asarray(props["centre"])) @ np.asarray(props["axis"]) - float(props["axis_lo"])) / float(props["length"]))
    return out
