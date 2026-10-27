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

"""Outcome labels from a recorded rollout trace. Pure functions, so the rules are testable
without the simulator.

Rules, in order:
1. ``not_lifted``: the object never rose 2 cm while both pads held it in a lifted phase
   (an object knocked over by the descending pads also raises its centre; that is not a lift).
2. Lost before the intended release (no pad touching in a held phase): ``dropped_high`` when
   the lowest point was 10 cm or more above the surface under it, or there was no surface;
   else ``dropped_low``. What happens after the drop does not change the label.
3. After the release: ``rolled_off`` when the object ends on the floor and the place is not
   the bin; ``missed`` when the place has an opening and the object's CoM is outside it or its
   lowest point sits more than 2 cm above the place's floor (bridging); ``tipped`` when the
   place reports it; else ``placed_stable``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

RISE_M = 0.02
HIGH_DROP_M = 0.10
CONTAIN_TOL_M = 0.02
HELD_PHASES = ("lift", "hold", "transport", "lower")


@dataclass
class Final:
    """The object's state after the rollout."""

    zone: str | None
    pos: list[float]
    low_z: float
    tipped: bool


def inside_opening(dest: dict[str, Any], pos) -> bool:
    fx, fy = dest["xy"]
    r = float(dest["opening_half_width"] or 0.0)
    if dest["kind"] == "rim":
        return math.hypot(pos[0] - fx, pos[1] - fy) <= r
    return abs(pos[0] - fx) <= r and abs(pos[1] - fy) <= r


def label_rollout(trace: list[dict[str, Any]], dest: dict[str, Any], release: dict[str, Any] | None,
                  final: Final) -> dict[str, Any]:
    """Return ``{"label", "max_z", "drop_z", "loss_phase", "loss_step", "loss_t", "loss_speed"}``."""
    max_z = max((x["z"] for x in trace), default=0.0)
    none = {"drop_z": None, "loss_phase": None, "loss_step": None, "loss_t": None, "loss_speed": None}
    first_rise = next((i for i, x in enumerate(trace)
                       if x["z"] > RISE_M and all(x["pads"]) and x["phase"] in HELD_PHASES), None)
    if first_rise is None:
        return {"label": "not_lifted", "max_z": max_z, **none}
    loss = next((i for i in range(first_rise, len(trace))
                 if trace[i]["phase"] in HELD_PHASES and not any(trace[i]["pads"])), None)
    if loss is not None:
        x = trace[loss]
        drop_z = float(x["low_z"] - (x["surface_z_below"] if x["has_surface"] else 0.0))
        label = "dropped_high" if (drop_z >= HIGH_DROP_M or not x["has_surface"]) else "dropped_low"
        return {"label": label, "max_z": max_z, "drop_z": drop_z, "loss_phase": x["phase"], "loss_step": loss,
                "loss_t": x["t"], "loss_speed": x["speed"]}
    if final.zone == "floor" and dest["id"] != "bin":
        label = "rolled_off"
    elif dest["kind"] in ("rim", "cavity", "open_bin") and not (
            inside_opening(dest, final.pos) and final.low_z <= float(dest["surface_z"]) + CONTAIN_TOL_M):
        label = "missed"
    elif release is not None and final.tipped:
        label = "tipped"
    else:
        label = "placed_stable"
    return {"label": label, "max_z": max_z, **none}
