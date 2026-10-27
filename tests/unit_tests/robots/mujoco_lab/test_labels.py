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


def _step(t, phase, z, pads, low=0.0, surf=0.0, has=True, speed=0.0, zone=None):
    return {"t": t, "phase": phase, "z": z, "low_z": low, "surface_z_below": surf, "has_surface": has,
            "speed": speed, "pads": list(pads), "zone": zone, "rot_deg": 0.0}


def _trace(lift_steps=20, lost_at=None, lost_phase="transport", z_rise=0.2, has=True, surf=0.75):
    tr = [_step(i / HZ, "approach", 0.0, (False, False)) for i in range(5)]
    tr += [_step(len(tr) / HZ, "close", 0.0, (True, True)) for _ in range(3)]
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


def _final(zone="table", pos=(0.0, 0.0, 0.76), low=0.751, tipped=False):
    return Final(zone=zone, pos=list(pos), low_z=low, tipped=tipped)


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
