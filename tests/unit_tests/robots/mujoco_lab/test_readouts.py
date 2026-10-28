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


"""Readouts of one rollout: pad forces at the hold, at a slip and at a loss, the rotation and its side (pure)."""

import pytest

from robots.mujoco_lab.readouts import summarize


def _s(t, phase, z, pads=(True, True), fz=0.0, rot=0.0, pitch=0.0, hand_z=None, nz=(0.0, 0.0), rot_lift=None):
    """One trace step; by default the hand carries the object (hand_z = 0.3 + z) and the pads press side faces."""
    if rot_lift is None and phase not in ("approach", "close"):
        rot_lift = 0.0
    return {"t": t, "phase": phase, "z": z, "pads": list(pads), "fz": fz, "fn": 10.0, "rot_deg": rot, "pitch_ey": pitch,
            "hand_z": 0.3 + z if hand_z is None else hand_z, "pad_nz": list(nz), "rot_lift_deg": rot_lift}


def _label(label, loss_phase=None, loss_step=None, loss_t=None):
    return {"label": label, "max_z": 0.0, "drop_z": None, "loss_phase": loss_phase, "loss_step": loss_step,
            "loss_t": loss_t, "loss_speed": None}


def _held_trace(hold_fz=(1.9, 2.0, 2.1, 2.0, 1.9, 2.0)):
    tr, t = [], 0.0
    for _ in range(3):
        tr.append(_s(t, "approach", 0.0, pads=(False, False))); t += 0.02
    for k in range(3):
        tr.append(_s(t, "close", 0.0, fz=0.3 * k)); t += 0.02
    for k in range(5):
        tr.append(_s(t, "lift", 0.04 * (k + 1), fz=2.0)); t += 0.02
    for fz in hold_fz:
        tr.append(_s(t, "hold", 0.2, fz=fz)); t += 0.02
    return tr


def test_summarize_hold_reading():
    tr = _held_trace()
    out = summarize(tr, _label("placed_stable"), v_lift=0.15)
    assert out["rose"] is True and out["max_rise_m"] == pytest.approx(0.2)
    assert out["fz_hold_n"] == pytest.approx(2.0)
    assert out["fz_slip_max_n"] is None and out["fz_at_loss_n"] is None
    assert out["loss_since_lift_s"] is None and out["v_lift"] == 0.15
    assert summarize(_held_trace(hold_fz=(2.0, 2.0, 2.0, 2.0)), _label("placed_stable"), 0.15)["fz_hold_n"] is None  # under 5 steps


def _slide_trace(lift_fz=(1.3, 1.5, 1.4), close_fz=(0.0, 0.4, 0.8), rise=(0.0, 0.0, 0.0), nz=(0.0, 0.0), rot=(0.0, 0.0, 0.0),
                 pads_lift=None):
    """A not-lifted attempt: the hand rises 3 mm per lift step while the object stays (the pads slide up it)."""
    tr, t = [], 0.0
    for fz in close_fz:
        tr.append(_s(t, "close", 0.0, fz=fz, hand_z=0.30)); t += 0.02
    for k, fz in enumerate(lift_fz):
        tr.append(_s(t, "lift", rise[k], fz=fz, hand_z=0.30 + 0.003 * (k + 1), nz=nz, rot_lift=rot[k],
                     pads=(True, True) if pads_lift is None else pads_lift[k])); t += 0.02
    for _ in range(6):
        tr.append(_s(t, "hold", rise[-1], fz=0.1, hand_z=0.30 + 0.003 * len(lift_fz), pads=(False, False))); t += 0.02
    return tr


def test_slip_reading_is_the_largest_force_while_the_pads_slide_up_a_still_object():
    out = summarize(_slide_trace(), _label("not_lifted"), v_lift=0.15)
    assert out["rose"] is False and out["fz_slip_max_n"] == pytest.approx(1.5)
    assert out["fz_hold_n"] is None and out["fz_at_loss_n"] is None
    held = summarize(_held_trace(), _label("placed_stable"), 0.15)
    assert held["fz_slip_max_n"] is None                                     # a slip reading only for not_lifted


def test_slip_reading_ignores_the_close_phase():
    """The pads strike the object at several times the squeeze while the table still holds it."""
    out = summarize(_slide_trace(close_fz=(0.0, 9.0, 4.0)), _label("not_lifted"), v_lift=0.15)
    assert out["fz_slip_max_n"] == pytest.approx(1.5)


def test_slip_reading_needs_the_pads_on_side_faces():
    """Pads that land on the object's top press it into the table (their contact normals point down)."""
    tr = _slide_trace(lift_fz=(-40.0, -35.0, -30.0), close_fz=(-50.0, -45.0, -42.0), nz=(-0.99, -0.98))
    assert summarize(tr, _label("not_lifted"), v_lift=0.15)["fz_slip_max_n"] is None


def test_slip_reading_needs_the_object_to_stay_while_the_hand_rises():
    """A pivot: the grasp rises with the hand, or the object turns, so the force is a share of the weight."""
    carried = _slide_trace(lift_fz=(16.5, 17.6, 18.5), rise=(0.0018, 0.0036, 0.0054))
    assert summarize(carried, _label("not_lifted"), v_lift=0.15)["fz_slip_max_n"] is None
    turned = _slide_trace(lift_fz=(16.5, 17.6, 18.5), rot=(2.0, 4.0, 6.0))
    assert summarize(turned, _label("not_lifted"), v_lift=0.15)["fz_slip_max_n"] is None


def test_slip_reading_needs_two_sliding_steps():
    tr = _slide_trace(pads_lift=[(True, True), (True, False), (False, False)])
    assert summarize(tr, _label("not_lifted"), v_lift=0.15)["fz_slip_max_n"] is None


def test_slip_reading_counts_the_first_lift_step():
    out = summarize(_slide_trace(lift_fz=(1.6, 1.5, 1.4)), _label("not_lifted"), v_lift=0.15)
    assert out["fz_slip_max_n"] == pytest.approx(1.6)


def test_slip_reading_starts_at_the_first_lift_step():
    """A grasp that has lost a pad when the lift begins gives no reading, though the pads slide later."""
    tr = _slide_trace(lift_fz=(1.3, 1.5, 1.4), pads_lift=[(True, False), (True, True), (True, True)])
    assert summarize(tr, _label("not_lifted"), v_lift=0.15)["fz_slip_max_n"] is None


def test_slip_reading_stops_at_the_first_step_that_fails():
    tr = _slide_trace(lift_fz=(1.3, 1.5, 7.0, 9.0), rise=(0.0, 0.0, 0.0, 0.0), rot=(0.0, 0.0, 4.0, 0.0))
    assert summarize(tr, _label("not_lifted"), v_lift=0.15)["fz_slip_max_n"] == pytest.approx(1.5)


def test_loss_reading_is_the_mean_of_the_last_three_two_pad_steps():
    tr = _held_trace()
    t = tr[-1]["t"] + 0.02
    for fz in (2.0, 1.6, 0.8):
        tr.append(_s(t, "transport", 0.2, fz=fz)); t += 0.02
    k = len(tr)
    tr.append(_s(t, "transport", 0.19, pads=(False, False))); t += 0.02
    tr.append(_s(t, "transport", 0.15, pads=(False, False)))
    first_lift_t = next(x["t"] for x in tr if x["phase"] == "lift")
    out = summarize(tr, _label("dropped_low", "transport", k, tr[k]["t"]), v_lift=0.15)
    assert out["fz_at_loss_n"] == pytest.approx((2.0 + 1.6 + 0.8) / 3)      # the mean of the last three two-pad steps
    assert out["loss_phase"] == "transport" and out["loss_t"] == pytest.approx(tr[k]["t"])
    assert out["loss_since_lift_s"] == pytest.approx(tr[k]["t"] - first_lift_t)


def test_loss_reading_skips_steps_with_one_pad():
    tr = _held_trace()
    t = tr[-1]["t"] + 0.02
    for fz, pads in ((2.0, (True, True)), (1.6, (True, True)), (9.0, (True, False)), (0.8, (True, True))):
        tr.append(_s(t, "transport", 0.2, fz=fz, pads=pads)); t += 0.02
    k = len(tr)
    tr.append(_s(t, "transport", 0.19, pads=(False, False)))
    out = summarize(tr, _label("dropped_low", "transport", k, tr[k]["t"]), v_lift=0.15)
    assert out["fz_at_loss_n"] == pytest.approx((2.0 + 1.6 + 0.8) / 3)


def test_summarize_heavy_side():
    def lost(rot, pitch):
        tr = _held_trace()
        t = tr[-1]["t"] + 0.02
        tr.append(_s(t, "transport", 0.2, fz=1.0, rot=rot - 1, pitch=pitch))
        k = len(tr)
        tr.append(_s(t + 0.02, "transport", 0.18, pads=(False, False), rot=rot, pitch=pitch))
        return summarize(tr, _label("dropped_low", "transport", k, tr[k]["t"]), v_lift=0.4)

    a = lost(25.0, -0.3)
    assert a["rot_deg"] == pytest.approx(25.0) and a["heavy_side"] == 1
    assert lost(25.0, 0.3)["heavy_side"] == -1
    assert lost(5.0, -0.3)["heavy_side"] == 0
    tr = _held_trace()
    tr[-1].update(rot_deg=12.0, pitch_ey=0.2)                                # no loss: the last hold step
    out = summarize(tr, _label("placed_stable"), 0.15)
    assert out["rot_deg"] == pytest.approx(12.0) and out["heavy_side"] == -1
