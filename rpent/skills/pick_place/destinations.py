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

"""Destination registry: where an object can be set down, and what each place is like.

The seven primitive destinations live in the MuJoCo lab scene. Positions are in
the scene frame (table centre at the origin, table top at ``TABLE_TOP``). The
``surface_z`` of a table place is the table top; of the bin it is the bin
floor on the ground.
"""

from __future__ import annotations

from typing import Any

TABLE_TOP = 0.75
TABLE_HALF = (0.35, 0.25)
BIN_FLOOR_Z = 0.0
BIN_H = 0.30


def _entry(
    id: str,
    kind: str,
    xy: tuple[float, float],
    *,
    surface_z: float = TABLE_TOP,
    mu: float = 0.8,
    rim_height: float = 0.0,
    opening_half_width: float | None = None,
    fragile: bool = False,
    object_rest_xy: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any]:
    return {
        "id": id,
        "kind": kind,
        "xy": tuple(xy),
        "surface_z": surface_z,
        "mu": mu,
        "rim_height": rim_height,
        "opening_half_width": opening_half_width,
        "fragile": fragile,
        "object_rest_xy": tuple(object_rest_xy),
    }


DESTINATIONS: dict[str, dict[str, Any]] = {
    "table_centre": _entry("table_centre", "surface", (0.0, 0.0), mu=0.8),
    "table_edge": _entry("table_edge", "surface", (-0.29, 0.0), mu=0.8),
    "table_gap": _entry("table_gap", "none", (-0.45, 0.0), mu=0.0),
    "mouse_pad": _entry("mouse_pad", "surface", (0.15, -0.15), surface_z=TABLE_TOP + 0.004, mu=1.2),
    "plate": _entry("plate", "rim", (0.12, 0.15), surface_z=TABLE_TOP + 0.005, mu=0.4,
                    rim_height=0.015, opening_half_width=0.11, fragile=True),
    "box": _entry("box", "cavity", (-0.15, 0.15), surface_z=TABLE_TOP + 0.005, mu=0.6,
                  rim_height=0.12, opening_half_width=0.08),
    "bin": _entry("bin", "open_bin", (-0.62, 0.0), surface_z=BIN_FLOOR_Z, mu=0.6,
                  rim_height=BIN_H, opening_half_width=0.15),
}


def release_height(dest: dict[str, Any]) -> float:
    """Height of the object's lowest point above ``surface_z`` when the pads open."""
    if dest["kind"] in ("rim", "cavity", "open_bin"):
        return 0.05 + float(dest["rim_height"])
    return 0.02


def has_surface(dest: dict[str, Any]) -> bool:
    return dest["kind"] != "none"


def edge_distance(dest: dict[str, Any]) -> float:
    """Distance from the release point to the nearest table edge; negative off the table."""
    x, y = dest["xy"]
    dx = TABLE_HALF[0] - abs(x)
    dy = TABLE_HALF[1] - abs(y)
    return float(min(dx, dy))
