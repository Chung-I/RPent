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

"""Outcome labelling from a recorded trace (pure; no sim)."""

from robots.mujoco_lab.labels import Final, label_rollout
from rpent.skills.pick_place.destinations import DESTINATIONS

HZ = 50.0


def _step(t, phase, z, pads, low=0.0, surf=0.0, has=True, speed=0.0, zone=None, fz=1.0):
    return {"t": t, "phase": phase, "z": z, "low_z": low, "surface_z_below": surf, "has_surface": has,
            "speed": speed, "pads": list(pads), "zone": zone, "rot_deg": 0.0, "fz": fz}


def _trace(lift_steps=20, lost_at=None, lost_phase="transport", z_rise=0.2, has=True, surf=0.75):
    """The pads lift the object z_rise off a surface at height surf (its lowest point rises with it), hold,
    carry, lower and release it."""
    tr = [_step(i / HZ, "approach", 0.0, (False, False), low=surf, surf=surf, has=has) for i in range(5)]
    tr += [_step(len(tr) / HZ, "close", 0.0, (True, True), low=surf, surf=surf, has=has) for _ in range(3)]
    for i in range(lift_steps):
        z = z_rise * (i + 1) / lift_steps
        tr.append(_step(len(tr) / HZ, "lift", z, (True, True), low=surf + z, surf=surf, has=has))
    for phase, n in (("hold", 3), ("transport", 6), ("lower", 6), ("release", 4)):
        for _ in range(n):
            tr.append(_step(len(tr) / HZ, phase, z_rise, (True, True) if phase != "release" else (False, False),
                            low=surf + z_rise, surf=surf, has=has))
    if lost_at is not None:
        idx = next(i for i, x in enumerate(tr) if x["phase"] == lost_phase) + lost_at
        for x in tr[idx:]:
            x["pads"] = [False, False]
        tr[idx]["speed"] = 0.4
    return tr


def _final(zone="table", pos=(0.0, 0.0, 0.76), low=0.751, tipped=False, held=False, com_xy=None):
    return Final(zone=zone, pos=list(pos), low_z=low, tipped=tipped, held=held, com_xy=list(com_xy or pos[:2]))


def test_knocked_over_during_approach_is_not_lifted():
    tr = [_step(i / HZ, "approach", 0.03 if i > 2 else 0.0, (False, False)) for i in range(5)]   # the pads tip it: centre rises 3 cm
    tr += [_step(len(tr) / HZ, "close", 0.03, (True, False)) for _ in range(3)]
    tr += [_step(len(tr) / HZ, "lift", 0.03, (False, False)) for _ in range(10)]
    out = label_rollout(tr, DESTINATIONS["bin"], None, _final())
    assert out["label"] == "not_lifted" and out["loss_phase"] is None


def test_loss_in_transport_high_is_dropped_high():
    out = label_rollout(_trace(lost_at=2), DESTINATIONS["table_centre"], None, _final(zone="table"))
    assert out["label"] == "dropped_high" and out["loss_phase"] == "transport" and out["loss_speed"] == 0.4
    assert abs(out["drop_z"] - 0.2) < 1e-9 and out["loss_step"] is not None


def test_loss_low_over_the_table_is_dropped_low_even_if_it_rolls_to_the_floor():
    out = label_rollout(_trace(z_rise=0.05, lost_at=1, lost_phase="lower"), DESTINATIONS["table_edge"], None, _final(zone="floor"))
    assert out["label"] == "dropped_low"


def test_loss_without_surface_is_dropped_high():
    out = label_rollout(_trace(z_rise=0.05, lost_at=1, lost_phase="lower", has=False), DESTINATIONS["table_gap"], None, _final(zone="floor"))
    assert out["label"] == "dropped_high"


def test_release_outcomes():
    rel = {"tilt_deg": 5.0, "displacement_xy_m": 0.0}
    plate = DESTINATIONS["plate"]
    inside = _final(zone="plate", pos=(plate["xy"][0], plate["xy"][1], 0.8), low=plate["surface_z"] + 0.005)
    assert label_rollout(_trace(), plate, rel, inside)["label"] == "placed_stable"
    bridged = _final(zone="plate", pos=(plate["xy"][0], plate["xy"][1], 0.8), low=plate["surface_z"] + 0.05)
    assert label_rollout(_trace(), plate, rel, bridged)["label"] == "missed"
    outside = _final(zone="table", pos=(plate["xy"][0] + 0.2, plate["xy"][1], 0.8), low=0.751)
    assert label_rollout(_trace(), plate, rel, outside)["label"] == "missed"
    assert label_rollout(_trace(), DESTINATIONS["table_edge"], rel, _final(zone="floor"))["label"] == "rolled_off"
    assert label_rollout(_trace(), DESTINATIONS["table_edge"], rel, _final(zone="table", tipped=True))["label"] == "tipped"
    assert label_rollout(_trace(), DESTINATIONS["table_edge"], rel, _final(zone="table"))["label"] == "placed_stable"


def test_release_phase_pad_loss_is_not_a_loss():
    out = label_rollout(_trace(), DESTINATIONS["table_centre"], {"tilt_deg": 0.0, "displacement_xy_m": 0.0}, _final())
    assert out["loss_phase"] is None and out["label"] == "placed_stable"


def test_object_still_in_the_pads_after_release_is_not_placed():
    rel = {"tilt_deg": 0.0, "displacement_xy_m": 0.0}
    held = Final(zone=None, pos=[0.0, 0.0, 0.8], low_z=0.77, tipped=False, held=True, com_xy=[0.0, 0.0])
    assert label_rollout(_trace(), DESTINATIONS["table_centre"], rel, held)["label"] == "missed"
    gap_held = Final(zone=None, pos=[0.0, -0.33, 0.8], low_z=0.77, tipped=False, held=True, com_xy=[0.0, -0.33])
    assert label_rollout(_trace(has=False), DESTINATIONS["table_gap"], rel, gap_held)["label"] == "missed"


def test_containment_uses_the_com_not_the_body_origin():
    rel = {"tilt_deg": 0.0, "displacement_xy_m": 0.0}
    plate = DESTINATIONS["plate"]
    px, py = plate["xy"]
    origin_far_com_inside = Final(zone="plate", pos=[px + 0.2, py, 0.8], low_z=plate["surface_z"] + 0.005, tipped=False,
                                  held=False, com_xy=[px, py])
    assert label_rollout(_trace(), plate, rel, origin_far_com_inside)["label"] == "placed_stable"
    origin_inside_com_far = Final(zone="plate", pos=[px, py, 0.8], low_z=plate["surface_z"] + 0.005, tipped=False,
                                  held=False, com_xy=[px + 0.2, py])
    assert label_rollout(_trace(), plate, rel, origin_inside_com_far)["label"] == "missed"


def _pivot(surf=0.75, pad_fz=0.6, lost=True):
    """A hammer turning about its head: the pads raise the handle and the body origin 10 cm while the head stays
    on the table (the lowest point within 0.2 mm of where it rested); the end-of-step contact flickers between
    the table and nothing. Then the pads let go in the lift, or the hand holds it so through the hold."""
    tr = [_step(i / HZ, "approach", 0.0, (False, False), low=surf, surf=surf, zone="table") for i in range(5)]
    tr += [_step(len(tr) / HZ, "close", 0.0, (True, True), low=surf, surf=surf, zone="table") for _ in range(3)]
    for i in range(20):
        tr.append(_step(len(tr) / HZ, "lift", 0.1 * (i + 1) / 20, (True, True), low=surf + 0.0002, surf=surf,
                        zone=None if i % 2 else "table", fz=pad_fz))
    if lost:
        tr += [_step(len(tr) / HZ, "lift", 0.08, (False, False), low=surf, surf=surf, zone="table", fz=0.0) for _ in range(3)]
    else:
        tr += [_step(len(tr) / HZ, "hold", 0.1, (True, True), low=surf + 0.0002, surf=surf, zone="table", fz=pad_fz)
               for _ in range(5)]
    return tr


def test_pivot_that_keeps_one_end_on_the_table_is_not_lifted():
    """The body origin rose 10 cm with both pads pushing up, but the object never left the table (handal
    hammer_7 d55 of corpus v6; the pads carried 0.6 of its weight). The loss is not a drop."""
    out = label_rollout(_pivot(), DESTINATIONS["table_centre"], None, _final())
    assert out["label"] == "not_lifted" and out["loss_phase"] is None


def test_pads_pressing_on_top_are_not_a_lift():
    """The pads land on a box's top and press it into the table (pad force down, -60 m g); the box tilts, so
    its origin rises 2.4 cm, and it pops up 6 mm as the pads slide off its edge (cheez_it d75 of corpus v6)."""
    surf = 0.75
    tr = [_step(i / HZ, "approach", 0.0, (False, False), low=surf, surf=surf, zone="table") for i in range(5)]
    tr += [_step(len(tr) / HZ, "close", 0.0, (True, True), low=surf - 0.004, surf=surf, zone="table", fz=-60.0)
           for _ in range(3)]
    tr += [_step(len(tr) / HZ, "lift", 0.024, (True, True), low=surf - 0.003, surf=surf, zone="table", fz=-60.0)
           for _ in range(5)]
    tr.append(_step(len(tr) / HZ, "lift", 0.024, (True, True), low=surf + 0.006, surf=surf, fz=-60.0))
    tr += [_step(len(tr) / HZ, "lift", 0.0, (False, False), low=surf, surf=surf, zone="table", fz=0.0) for _ in range(3)]
    out = label_rollout(tr, DESTINATIONS["table_centre"], None, _final())
    assert out["label"] == "not_lifted"


def test_object_hanging_low_in_the_pads_is_lifted():
    """The hammer turns 74 deg in the pads and hangs with its head 1.5 cm over the table while the origin rose
    15 cm (handal hammer_3 d12 of corpus v6): it left the table, so it was lifted."""
    tr = _trace(z_rise=0.15)
    for x in tr:
        if x["phase"] != "approach" and x["phase"] != "close":
            x["low_z"] = 0.75 + min(x["z"], 0.015)
    rel = {"tilt_deg": 5.0, "displacement_xy_m": 0.0}
    assert label_rollout(tr, DESTINATIONS["table_centre"], rel, _final())["label"] == "placed_stable"


def test_first_lifted_needs_the_object_clear_of_where_it_rested():
    from robots.mujoco_lab.labels import first_lifted

    assert first_lifted(_pivot(lost=False), ("lift", "hold")) is None
    tr = _trace()
    i = first_lifted(tr, ("lift", "hold"))
    assert tr[i]["phase"] == "lift" and tr[i]["z"] > 0.02 and tr[i - 1]["z"] <= 0.02


def _rim_top(dest):
    return float(dest["surface_z"]) + float(dest["rim_height"])


def _lowered(place, zone, fall=0.0, low_at_loss=None):
    """Lifted 20 cm off the table, carried over the place and lowered; at the third lower step the object
    touches `zone` and the pads let go; the lowest point then falls `fall` before the lower ends."""
    dest = DESTINATIONS[place]
    floor = float(dest["surface_z"])
    tr = _trace(z_rise=0.2)
    for x in tr:
        if x["phase"] in ("transport", "lower", "release"):
            x["surface_z_below"] = floor
    lower = [i for i, x in enumerate(tr) if x["phase"] == "lower"]
    i0 = lower[2]
    low = 0.95 if low_at_loss is None else low_at_loss
    for i in range(i0, len(tr)):
        tr[i]["pads"] = [False, False]
        tr[i]["zone"] = zone
        k = lower.index(i) - 2 if i in lower else len(lower) - 3
        tr[i]["low_z"] = low - fall * k / (len(lower) - 3)
    return tr, dest, floor


def test_set_down_on_the_place_goes_to_the_release_rules():
    """A long object meets the box wall while the hand lowers it: the wall takes it and the pads slide off.
    That is a set-down, not a 12 cm drop; the release rules judge where it ends (on the wall top: missed;
    wood_block d10 and 24 more rows of corpus v6 read dropped_high)."""
    tr, dest, floor = _lowered("box", "box", low_at_loss=_rim_top(DESTINATIONS["box"]))
    bx, by = dest["xy"]
    on_wall = Final(zone="box", pos=[bx + 0.08, by, 0.9], low_z=_rim_top(dest), tipped=False, held=False, com_xy=[bx + 0.09, by])
    out = label_rollout(tr, dest, {"tilt_deg": 0.0, "displacement_xy_m": 0.0}, on_wall)
    assert out["label"] == "missed" and out["loss_phase"] is None
    inside = Final(zone="box", pos=[bx, by, 0.8], low_z=floor + 0.005, tipped=False, held=False, com_xy=[bx, by])
    assert label_rollout(tr, dest, {"tilt_deg": 0.0, "displacement_xy_m": 0.0}, inside)["label"] == "placed_stable"


def test_set_down_that_then_falls_ten_cm_is_a_drop():
    """The pads lose the object on the bin's rim and it falls 25 cm to the bin floor before the lower ends
    (potato_masher d30 of corpus v6): a drop from the rim."""
    tr, dest, floor = _lowered("bin", "bin", fall=0.25, low_at_loss=0.30)
    final = Final(zone="bin", pos=[dest["xy"][0], dest["xy"][1], 0.03], low_z=0.05, tipped=False, held=False,
                  com_xy=list(dest["xy"]))
    out = label_rollout(tr, dest, {"tilt_deg": 0.0, "displacement_xy_m": 0.0}, final)
    assert out["label"] == "dropped_high" and out["loss_phase"] == "lower"


def test_lower_loss_touching_the_table_past_the_gap_is_a_drop():
    """Over the gap the object meets the table edge and the pads let go: the table is not the place."""
    tr, dest, floor = _lowered("table_gap", "table", low_at_loss=0.76)
    for x in tr:
        if x["phase"] in ("transport", "lower", "release"):
            x["has_surface"] = False
    out = label_rollout(tr, dest, {"tilt_deg": 0.0, "displacement_xy_m": 0.0}, _final(zone="floor", low=0.0))
    assert out["label"] == "dropped_high" and out["loss_phase"] == "lower"
