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
1. ``not_lifted``: in no held phase step did the hand hold the object up off its support (``lifted``:
   both pads press and push it up, its origin rose 2 cm and its lowest point is 5 mm clear of where it
   rested). The origin alone does not tell a lift: an object knocked over by the descending pads, turned
   about one end that stays on the table (a pivot) or tilted by pads pressing its top also raises it.
2. Lost before the intended release (no pad touching in a held phase): ``dropped_high`` when
   the lowest point was 10 cm or more above the surface under it, or there was no surface;
   else ``dropped_low``. What happens after the drop does not change the label. A loss in the
   lower while the object touches the place itself is a set-down, judged by the release rules (the
   place took the object and the descending pads slid off it), unless the object then fell 10 cm
   or more before the lower ended.
3. After the release: ``missed`` when the object is still in the pads or rests on nothing
   (a wedged object the open gripper did not let go of is not placed); ``rolled_off`` when the
   object ends on the floor and the place is not the bin; ``missed`` when the place has an
   opening and the object's centre of mass is outside it or its lowest point sits more than
   2 cm above the place's floor (bridging); ``tipped`` when the place reports it; else
   ``placed_stable``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

RISE_M = 0.02
CLEAR_M = 0.005            # lifted: the lowest point this far above where the object rested (a pivot keeps it at 0)
HIGH_DROP_M = 0.10
CONTAIN_TOL_M = 0.02
HELD_PHASES = ("lift", "hold", "transport", "lower")
TABLE_PLACES = ("table_centre", "table_edge")


@dataclass
class Final:
    """The object's state after the rollout."""

    zone: str | None
    pos: list[float]
    low_z: float
    tipped: bool
    held: bool = False                      # a pad still touches the object after the release and retreat
    com_xy: list[float] | None = None       # the centre of mass (containment is judged by it, not the body origin)


def lifted(x: dict[str, Any], rest_low_z: float, rise: float = RISE_M) -> bool:
    """In trace step x the hand holds the object up off its support: both pads press, their net vertical force
    pushes it up, its origin has risen ``rise`` and its lowest point is CLEAR_M above ``rest_low_z``, where it
    rested. A pivot keeps the lowest point on the table (hammer_7 d55 of corpus v6: the origin rose 12.8 cm, the
    head 0.2 mm); pads pressing a top face push down, and the object pops up as they slide off its edge."""
    return bool(all(x["pads"]) and x["fz"] > 0.0 and x["z"] > rise and x["low_z"] - rest_low_z > CLEAR_M)


def first_lifted(trace: list[dict[str, Any]], phases=HELD_PHASES, rise: float = RISE_M) -> int | None:
    """The first step in ``phases`` in which the object is ``lifted``; it rested where the trace began."""
    if not trace:
        return None
    rest = float(trace[0]["low_z"])
    return next((i for i, x in enumerate(trace) if x["phase"] in phases and lifted(x, rest, rise)), None)


def place_zone(dest: dict[str, Any]) -> str | None:
    """The contact zone (Sim.zone_of) of the place's own surface: the table for a table place, none past the
    table's edge, else the fixture, whose zone carries the place's id (the bin's is "bin")."""
    if dest["kind"] == "none":
        return None
    return "table" if dest["id"] in TABLE_PLACES else dest["id"]


def _set_down(trace: list[dict[str, Any]], i: int, dest: dict[str, Any]) -> bool:
    """The loss at step i is a set-down: it is in the lower, the object touches the place itself (a ladle
    across the box wall) and its lowest point does not then fall HIGH_DROP_M before the lower ends (potato_masher
    d30 of corpus v6 fell 24 cm off the bin's rim to its floor: a drop)."""
    x = trace[i]
    if x["phase"] != "lower" or x["zone"] is None or x["zone"] != place_zone(dest):
        return False
    rest = [float(y["low_z"]) for y in trace[i:] if y["phase"] == "lower"]
    return float(x["low_z"]) - min(rest) < HIGH_DROP_M


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
    first_rise = first_lifted(trace)
    if first_rise is None:
        return {"label": "not_lifted", "max_z": max_z, **none}
    loss = next((i for i in range(first_rise, len(trace))
                 if trace[i]["phase"] in HELD_PHASES and not any(trace[i]["pads"])), None)
    if loss is not None and not _set_down(trace, loss, dest):
        x = trace[loss]
        drop_z = float(x["low_z"] - (x["surface_z_below"] if x["has_surface"] else 0.0))
        label = "dropped_high" if (drop_z >= HIGH_DROP_M or not x["has_surface"]) else "dropped_low"
        return {"label": label, "max_z": max_z, "drop_z": drop_z, "loss_phase": x["phase"], "loss_step": loss,
                "loss_t": x["t"], "loss_speed": x["speed"]}
    com_xy = final.com_xy if final.com_xy is not None else final.pos[:2]
    if final.held or final.zone is None:
        label = "missed"
    elif final.zone == "floor" and dest["id"] != "bin":
        label = "rolled_off"
    elif dest["kind"] in ("rim", "cavity", "open_bin") and not (
            inside_opening(dest, com_xy) and final.low_z <= float(dest["surface_z"]) + CONTAIN_TOL_M):
        label = "missed"
    elif release is not None and final.tipped:
        label = "tipped"
    else:
        label = "placed_stable"
    return {"label": label, "max_z": max_z, **none}


def final_label(motion_label: str, damage: dict[str, Any] | None) -> str:
    """The rollout's label (plan 2b): the damage meter's first threshold crossing (``damage``: label, channel, t; see
    fragility.damage_event) replaces the motion label of the rules above."""
    return str(damage["label"]) if damage else motion_label
