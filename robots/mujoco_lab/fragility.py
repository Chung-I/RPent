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


"""Fragility of the lab's objects (milestone 2b): each object's damage class, material law and cavity, the
provisional thresholds, the draws of the hidden fragility, the liquid a fill puts into a cavity, the static
spill angle, and the damage rule that turns the damage meter's channels into a label.

* ``fragility_classes.json``: per object ``class`` (none, crushable, bruisable, breakable), ``law`` (cardboard,
  thin_plastic, fruit, ceramic, glass or null), ``fillable``, ``cavity`` (an open container's frustum in the object
  frame: ``axis_obj``, ``bottom_obj``, ``r_bottom_m``, ``r_top_m``, ``height_m``), ``cavity_checked`` and ``reason``.
  Written by ``write_classes`` from the corpus material and hollow fields with the hand corrections below; every
  cavity is estimated from the visual mesh (cavities.estimate) and checked on its render before it loads.
* ``Thresholds``: per law a bivariate log-normal of the stiffness k (N/mm) and the damage force F (N) with
  correlation rho (cardboard, thin_plastic, fruit), the bruise speed (fruit) and the break speed (ceramic, glass),
  and the slosh damping. ``draw_fragility`` keeps the three standard normals z that produced a draw:
  log F = mu_F + sd_F z1, log k = mu_k + sd_k (rho z1 + sqrt(1 - rho^2) z2), log v = mu_v + sd_v z3, so
  ``map_latent`` maps kept draws through another law (thresholds v1).
* ``damage_event``: the first threshold crossing in time over the force, impact and surface-angle channels;
  equal times resolve crushed, bruised, broken, spilled.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CLASSES = os.path.join(DATA, "fragility_classes.json")
THRESHOLDS_V0 = os.path.join(DATA, "thresholds_v0.json")
RENDERS = os.path.join(DATA, "cavities")
LAWS = {"crushable": ("cardboard", "thin_plastic"), "bruisable": ("fruit",), "breakable": ("ceramic", "glass")}
ORDER = ("crushed", "bruised", "broken", "spilled")              # the tie rule
CAVITY_KEYS = ("axis_obj", "bottom_obj", "r_bottom_m", "r_top_m", "height_m")
NZ = 2001                                                         # heights of the spill angle's volume integral
SCAN_DEG = 2.0                                                    # the spill angle's first sign change is looked for on this grid
FLAGS = ("class", "law", "fillable")


def load_classes(path: str = CLASSES) -> dict[str, dict[str, Any]]:
    """The fragility entries by object id. A fillable entry needs a cavity with every key and ``cavity_checked`` true."""
    classes = json.load(open(path))
    for oid, e in classes.items():
        if e.get("fillable"):
            c = e.get("cavity")
            if not c or any(k not in c for k in CAVITY_KEYS):
                raise ValueError(f"{oid} is fillable without a cavity ({', '.join(CAVITY_KEYS)})")
            if e.get("cavity_checked") is not True:
                raise ValueError(f"{oid}: its cavity has not been checked on a render (cavities.render)")
    return classes


def entry_of(object_id: str, classes: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """(the object's entry, evidence lines): an unknown id is treated as class none and says so."""
    if object_id in classes:
        return classes[object_id], []
    return ({"class": "none", "law": None, "fillable": False, "cavity": None, "reason": "no entry"},
            [f"no fragility class for {object_id}: treated as none"])


@dataclass
class Thresholds:
    laws: dict[str, dict[str, Any]]
    slosh_zeta: float
    version: str = ""
    status: str = ""
    slosh_scale: dict[str, dict[str, float]] | None = None   # the annex's forcing factors, {class: {str(fill): factor}}

    @staticmethod
    def from_json(path: str) -> Thresholds:
        d = json.load(open(path))
        return Thresholds(laws=d["laws"], slosh_zeta=float(d["slosh_zeta"]), version=str(d.get("version", "")),
                          status=str(d.get("status", "")), slosh_scale=_slosh_scale(d.get("slosh_scale")))


def _slosh_scale(sc: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, float]] | None:
    """A file's slosh_scale as floats (None: none): every class has fills in [0, 1], each with a finite positive factor."""
    if sc is None:
        return None
    out = {str(c): {str(f): float(v) for f, v in by_fill.items()} for c, by_fill in sc.items()}
    if any(not by_fill or any(not 0.0 <= float(f) <= 1.0 or not (math.isfinite(v) and v > 0.0) for f, v in by_fill.items())
           for by_fill in out.values()):
        raise ValueError(f"slosh_scale must give each class {{fill in [0, 1]: a finite positive factor}}: {sc}")
    return out


def _lognormal(p: dict[str, float], z: float) -> float:
    return float(p["median"]) * math.exp(float(p["log_sd"]) * z)


def map_latent(z, entry: dict[str, Any], th: Thresholds) -> dict[str, Any]:
    """The fragility of the latent standard normals z = (z1, z2, z3) under the entry's law in ``th``; keys the law has
    no part for are None."""
    z1, z2, z3 = (float(v) for v in z)
    out: dict[str, Any] = {"k_n_per_mm": None, "f_damage_n": None, "v_bruise_ms": None, "v_break_ms": None, "z": [z1, z2, z3]}
    law = th.laws.get(entry.get("law")) if entry.get("law") and entry.get("class", "none") != "none" else None
    if law is None:
        return out
    if "f_damage_n" in law:
        rho = float(law["rho"])
        out["f_damage_n"] = _lognormal(law["f_damage_n"], z1)
        out["k_n_per_mm"] = _lognormal(law["k_n_per_mm"], rho * z1 + math.sqrt(1.0 - rho * rho) * z2)
    for key in ("v_bruise_ms", "v_break_ms"):
        if key in law:
            out[key] = _lognormal(law[key], z3)
    return out


def draw_fragility(rng: np.random.Generator, entry: dict[str, Any], th: Thresholds) -> dict[str, Any]:
    """One draw of the hidden fragility: three standard normals for every class (one stream for all objects)."""
    return map_latent(rng.standard_normal(3), entry, th)


def _frustum(cavity: dict[str, Any]) -> tuple[float, float, float]:
    return float(cavity["r_bottom_m"]), float(cavity["r_top_m"]), float(cavity["height_m"])


def cavity_radius(cavity: dict[str, Any], z):
    """The cavity's radius at height z above its floor (clipped to the cavity)."""
    r_b, r_t, H = _frustum(cavity)
    return r_b + (r_t - r_b) * np.clip(z, 0.0, H) / H


def _volume_to(cavity: dict[str, Any], h: float) -> float:
    r_b, r_t, H = _frustum(cavity)
    k = (r_t - r_b) / H
    return math.pi * (r_b * r_b * h + r_b * k * h * h + k * k * h ** 3 / 3.0)


def _moment_to(cavity: dict[str, Any], h: float) -> float:
    r_b, r_t, H = _frustum(cavity)
    k = (r_t - r_b) / H
    return math.pi * (r_b * r_b * h * h / 2.0 + 2.0 * r_b * k * h ** 3 / 3.0 + k * k * h ** 4 / 4.0)


def liquid(cavity: dict[str, Any], fill: float) -> dict[str, Any]:
    """The liquid of a fill (a fraction of the cavity's volume) of water: the cavity's volume in litres, the liquid's
    mass (1000 kg/m^3), its height above the floor and its centroid in the object frame (on the axis)."""
    f = float(np.clip(fill, 0.0, 1.0))
    H = _frustum(cavity)[2]
    V = _volume_to(cavity, H)
    if f == 0.0:
        h = 0.0
    elif f == 1.0:
        h = H
    else:
        h = float(brentq(lambda x: _volume_to(cavity, x) - f * V, 0.0, H, xtol=1e-12))
    zc = _moment_to(cavity, h) / _volume_to(cavity, h) if h > 0 else 0.0
    axis, bottom = np.asarray(cavity["axis_obj"], dtype=float), np.asarray(cavity["bottom_obj"], dtype=float)
    return {"volume_l": 1000.0 * V, "mass_kg": 1000.0 * V * f, "height_m": h, "centroid_obj": (bottom + zc * axis).tolist()}


def _volume_under_plane(z: np.ndarray, r: np.ndarray, z0: float, tan_t: float) -> float:
    """The cavity's volume below the plane z = z0 + x tan_t (x across the cavity, z along its axis): the disk segment
    x > (z - z0) / tan_t at each height, integrated over the heights."""
    u = np.clip((z - z0) / (tan_t * r), -1.0, 1.0)
    return float(np.trapezoid(r * r * (np.arccos(u) - u * np.sqrt(1.0 - u * u)), z))


def spill_angle_deg(cavity: dict[str, Any], fill: float) -> float:
    """The static spill angle: the largest tilt of the cavity at which the flat liquid surface that holds the fill's
    volume stays below the rim's lowest point. The plane's height at the axis comes from the volume below a tilted
    plane, integrated over the frustum; the first sign change of (surface at the low rim - rim) is bracketed on a
    2 degree grid and refined. 90 for an empty cavity, 0 for a full one."""
    f = float(np.clip(fill, 0.0, 1.0))
    if f <= 0.0:
        return 90.0
    r_b, r_t, H = _frustum(cavity)
    if f >= 1.0:
        return 0.0
    z = np.linspace(0.0, H, NZ)
    r = r_b + (r_t - r_b) * z / H
    target = f * _volume_to(cavity, H)
    r_max = max(r_b, r_t)

    def above_rim(t: float) -> float:
        if t <= 0.0:
            return float(liquid(cavity, f)["height_m"]) - H
        tt = math.tan(t)
        z0 = brentq(lambda c: _volume_under_plane(z, r, c, tt) - target, -r_max * tt - 1e-9, H + r_max * tt + 1e-9,
                    xtol=1e-12)
        return z0 + r_t * tt - H

    grid = np.radians(np.append(np.arange(0.0, 90.0, SCAN_DEG), 89.999))
    prev, g_prev = grid[0], above_rim(grid[0])
    for t in grid[1:]:
        g = above_rim(float(t))
        if g >= 0.0:
            return float(np.degrees(brentq(above_rim, prev, float(t), xtol=1e-10))) if g_prev < 0.0 else float(np.degrees(prev))
        prev, g_prev = float(t), g
    return 90.0


SPILL_GRID = np.linspace(0.0, 1.0, 201)       # the fills of a cavity's spill-angle table
_SPILL_TABLES: dict[tuple, np.ndarray] = {}


def spill_angle(cavity: dict[str, Any], fill):
    """spill_angle_deg from a table over SPILL_GRID (linear between fills 0.005 apart, about 0.7 s per cavity once a
    process): the corpus rows, the sim's damage rule and the prior read the same angle."""
    key = tuple(float(cavity[k]) for k in ("r_bottom_m", "r_top_m", "height_m"))
    if key not in _SPILL_TABLES:
        _SPILL_TABLES[key] = np.array([spill_angle_deg(cavity, float(f)) for f in SPILL_GRID])
    out = np.interp(np.clip(fill, 0.0, 1.0), SPILL_GRID, _SPILL_TABLES[key])
    return float(out) if np.ndim(out) == 0 else out


def liquid_arrays(cavity: dict[str, Any], fills) -> tuple[np.ndarray, np.ndarray]:
    """liquid() for many fills: (mass in kg (n,), centroid in the object frame (n, 3)); the height solves the frustum's
    volume by Newton steps (dV/dh = pi r(h)^2)."""
    f = np.clip(np.asarray(fills, dtype=float), 0.0, 1.0)
    r_b, r_t, H = _frustum(cavity)
    k = (r_t - r_b) / H
    V = _volume_to(cavity, H)
    target = f * V
    h = f * H
    for _ in range(40):
        vol = np.pi * (r_b * r_b * h + r_b * k * h * h + k * k * h ** 3 / 3.0)
        h = np.clip(h - (vol - target) / np.maximum(np.pi * (r_b + k * h) ** 2, 1e-12), 0.0, H)
    vol = np.pi * (r_b * r_b * h + r_b * k * h * h + k * k * h ** 3 / 3.0)
    mom = np.pi * (r_b * r_b * h * h / 2.0 + 2.0 * r_b * k * h ** 3 / 3.0 + k * k * h ** 4 / 4.0)
    zc = np.where(vol > 0, mom / np.maximum(vol, 1e-300), 0.0)
    axis, bottom = np.asarray(cavity["axis_obj"], dtype=float), np.asarray(cavity["bottom_obj"], dtype=float)
    return 1000.0 * V * f, bottom[None, :] + zc[:, None] * axis[None, :]


def map_latent_arrays(Z, entry: dict[str, Any], th: Thresholds) -> dict[str, Any]:
    """map_latent for many latent draws Z (n, 3): arrays per key, None for the keys the law has no part for."""
    Z = np.asarray(Z, dtype=float).reshape(-1, 3)
    out: dict[str, Any] = {"k_n_per_mm": None, "f_damage_n": None, "v_bruise_ms": None, "v_break_ms": None, "z": Z}
    law = th.laws.get(entry.get("law")) if entry.get("law") and entry.get("class", "none") != "none" else None
    if law is None:
        return out

    def ln(p, z):
        return float(p["median"]) * np.exp(float(p["log_sd"]) * z)

    if "f_damage_n" in law:
        rho = float(law["rho"])
        out["f_damage_n"] = ln(law["f_damage_n"], Z[:, 0])
        out["k_n_per_mm"] = ln(law["k_n_per_mm"], rho * Z[:, 0] + math.sqrt(1.0 - rho * rho) * Z[:, 1])
    for key in ("v_bruise_ms", "v_break_ms"):
        if key in law:
            out[key] = ln(law[key], Z[:, 2])
    return out


def _first_above(steps, threshold: float) -> float | None:
    for t, v in steps or []:
        if float(v) > float(threshold):
            return float(t)
    return None


def damage_event(meter: dict[str, Any], frag: dict[str, Any], entry: dict[str, Any], fill: float,
                 spill_deg: float | None = None) -> tuple[str, str, float] | None:
    """(label, channel, time) of the first threshold crossing, or None. ``meter``: ``fn_pad_max`` and ``surface_max``
    (running-maximum step lists [(t, value)]) and ``impacts`` ([(t, speed)]). A pad force above f_damage_n crushes a
    crushable object and bruises a bruisable one; an impact above v_bruise_ms bruises, above v_break_ms breaks; a
    surface angle above the spill angle (``spill_deg``, else spill_angle_deg of the cavity) spills a filled cavity.
    Equal times resolve crushed, bruised, broken, spilled."""
    cls = entry.get("class", "none")
    events = []
    if frag.get("f_damage_n") is not None and cls in ("crushable", "bruisable"):
        t = _first_above(meter.get("fn_pad_max"), frag["f_damage_n"])
        if t is not None:
            events.append((t, "crushed" if cls == "crushable" else "bruised", "force"))
    for key, label, wanted in (("v_bruise_ms", "bruised", "bruisable"), ("v_break_ms", "broken", "breakable")):
        if frag.get(key) is not None and cls == wanted:
            t = _first_above(meter.get("impacts"), frag[key])
            if t is not None:
                events.append((t, label, "impact"))
    if entry.get("fillable") and fill is not None and float(fill) > 0.0:
        angle = spill_angle_deg(entry["cavity"], fill) if spill_deg is None else float(spill_deg)
        t = _first_above(meter.get("surface_max"), angle)
        if t is not None:
            events.append((t, "spilled", "surface"))
    if not events:
        return None
    t, label, channel = min(events, key=lambda e: (e[0], ORDER.index(e[1])))
    return label, channel, t


# --- the class file. The corpus material gives the class and law; the corrections below were each seen on a render
# of the textured mesh. Fruit means fruit, not the cans and cartons with fruit names.
FRUIT = ("fruits_veggies__avocado", "ycb__banana", "fruits_veggies__lemon1", "fruits_veggies__lime",
         "fruits_veggies__orange1", "fruits_veggies__red_onion", "fruits_veggies__pomegranate")
BY_MATERIAL = {"cardboard": ("crushable", "cardboard"), "thin_plastic": ("crushable", "thin_plastic"),
               "fruit": ("bruisable", "fruit"), "ceramic": ("breakable", "ceramic"), "glass": ("breakable", "glass")}
OVERRIDES = {
    "hope__canned_peaches": ("none", None, "a metal can: the corpus material 'fruit' comes from its name"),
    "hope__pineapple_slices_can": ("none", None, "a metal can: the corpus material 'fruit' comes from its name"),
    "hope__tomato_sauce_can": ("none", None, "a metal can: the corpus material 'fruit' comes from its name"),
    "hope__orange_juice_carton": ("crushable", "cardboard",
                                  "a paperboard carton: the corpus material 'fruit' comes from its name"),
    "handal__measuring_cups": ("none", None,
                               "a plastic measuring cup (render): the corpus material 'ceramic' comes from the name 'cup'"),
    "handal__measuring_cups_1": ("none", None,
                                 "a plastic measuring cup (render): the corpus material 'ceramic' comes from the name 'cup'"),
    "hot3d__coffee_pot": ("none", None,
                          "a metal moka pot (render): the corpus material 'ceramic' comes from the name 'pot'"),
    "hot3d__storage_box": ("none", None,
                           "a bamboo box (render): the corpus material 'cardboard' comes from the name 'box'"),
}
# the open containers, each cavity checked on its render (data/cavities/<id>.png): what the check saw
FILLABLE = {
    "ycb__mug": "the frustum follows the inner wall; the scan fills the inner bottom corner below 3.5 cm (a "
                "reconstruction artifact), so the frustum narrows toward the floor",
    "ycb__bowl": "the frustum follows the curved wall; the rim on its inner edge",
    "ycb__bowl2": "the same mesh as ycb__bowl",
    "hot3d__ceramic_mug": "the frustum follows the tapered wall",
    "hot3d__mug": "the frustum follows the tapered wall",
    "hot3d__wooden_bowl": "a round wall: the frustum lies inside it near the rim and outside it near the floor, with "
                          "the rim's radius and the measured volume",
    "hot3d__pitcher": "a jug that narrows to a neck: the frustum averages the body and the neck, with the lip's radius; "
                      "the spout is not modelled",
}
NOT_FILLABLE = {
    "ycb__pitcher": "the scanned mesh is closed at the top (a lid), so it has no open cavity",
    "hot3d__coffee_pot": "a closed moka pot: no open cavity",
    "hope__yogurt_cup": "a sealed cup: the mesh is closed at the top",
    "handal__measuring_cups": "it rests opening-down at the corpus reset pose (the mesh opens toward -z)",
    "handal__measuring_cups_1": "it rests opening-down at the corpus reset pose (the mesh opens toward -z)",
    "hot3d__storage_box": "an open box, not a liquid container",
}
UP_MIN = 0.95                     # a fillable cavity's axis points up at the corpus reset pose: its world z above this


def write_classes(path: str = CLASSES, renders: str | None = RENDERS) -> dict[str, dict[str, Any]]:
    """Write ``fragility_classes.json`` for every corpus object (adapter.object_ids): class and law from the corpus
    material and OVERRIDES, and for each FILLABLE object the cavity from its mesh (cavities.estimate_mesh), rendered
    to ``renders``. ValueError when a fillable cavity does not face up at the corpus reset pose."""
    from robots.mujoco_lab import cavities
    from robots.mujoco_lab.adapter import MuJoCoLabEnv, load_sim, object_ids

    objects = load_sim()["objects"].BY_ID
    env = MuJoCoLabEnv()
    if renders:
        os.makedirs(renders, exist_ok=True)
    out: dict[str, dict[str, Any]] = {}
    for oid in object_ids():
        mat = objects[oid]["material"]
        if oid in OVERRIDES:
            cls, law, reason = OVERRIDES[oid]
        elif mat == "fruit" and oid not in FRUIT:
            raise ValueError(f"{oid}: material fruit, not in FRUIT and without a correction")
        elif mat in BY_MATERIAL:
            (cls, law), reason = BY_MATERIAL[mat], f"corpus material {mat}"
        else:
            cls, law, reason = "none", None, f"corpus material {mat}: no damage class"
        e: dict[str, Any] = {"class": cls, "law": law, "fillable": oid in FILLABLE, "cavity": None, "reason": reason}
        if oid in FILLABLE:
            mesh = cavities.load_mesh(oid)
            c = cavities.estimate_mesh(mesh)
            env.reset(oid, seed=0)
            up = float((env.sim.obj_mat() @ np.asarray(c["axis_obj"]))[2])
            if up < UP_MIN:
                raise ValueError(f"{oid}: the cavity's axis has world z {up:.3f} at the corpus reset pose")
            e.update(cavity={k: c[k] for k in CAVITY_KEYS}, cavity_checked=True, cavity_check=FILLABLE[oid])
            if renders:
                cavities.render(mesh, c, os.path.join(renders, f"{oid}.png"), oid)
        elif oid in NOT_FILLABLE:
            e["reason"] += f"; not fillable: {NOT_FILLABLE[oid]}"
        out[oid] = e
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Write fragility_classes.json and the cavity renders.")
    ap.add_argument("--out", default=CLASSES)
    ap.add_argument("--renders", default=RENDERS)
    a = ap.parse_args(argv)
    classes = write_classes(a.out, a.renders)
    n = {c: sum(e["class"] == c for e in classes.values()) for c in ("none", "crushable", "bruisable", "breakable")}
    print(f"{len(classes)} objects: {n}; fillable {sorted(o for o, e in classes.items() if e['fillable'])}", flush=True)


if __name__ == "__main__":
    main()
