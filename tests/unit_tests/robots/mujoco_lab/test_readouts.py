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


def _s(t, phase, z, pads=(True, True), fz=0.0, rot=0.0, pitch=0.0):
    return {"t": t, "phase": phase, "z": z, "pads": list(pads), "fz": fz, "fn": 10.0, "rot_deg": rot, "pitch_ey": pitch}


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


def test_summarize_slip_reading():
    tr, t = [], 0.0
    for k in range(4):
        tr.append(_s(t, "close", 0.0, fz=0.4 * k)); t += 0.02
    for k, fz in enumerate((1.3, 1.5, 1.1, 0.2)):
        tr.append(_s(t, "lift", 0.003 * k, fz=fz, pads=(True, k < 3))); t += 0.02
    for _ in range(6):
        tr.append(_s(t, "hold", 0.004, fz=0.1)); t += 0.02
    out = summarize(tr, _label("not_lifted"), v_lift=0.15)
    assert out["rose"] is False and out["fz_slip_max_n"] == pytest.approx(1.5)
    assert out["fz_hold_n"] is None and out["fz_at_loss_n"] is None
    held = summarize(_held_trace(), _label("placed_stable"), 0.15)
    assert held["fz_slip_max_n"] is None                                     # a slip reading only for not_lifted


def test_summarize_loss_reading():
    tr = _held_trace()
    t = tr[-1]["t"] + 0.02
    for fz in (2.0, 1.6, 0.8):
        tr.append(_s(t, "transport", 0.2, fz=fz)); t += 0.02
    k = len(tr)
    tr.append(_s(t, "transport", 0.19, pads=(False, False))); t += 0.02
    tr.append(_s(t, "transport", 0.15, pads=(False, False)))
    first_lift_t = next(x["t"] for x in tr if x["phase"] == "lift")
    out = summarize(tr, _label("dropped_low", "transport", k, tr[k]["t"]), v_lift=0.15)
    assert out["fz_at_loss_n"] == pytest.approx(0.8)
    assert out["loss_phase"] == "transport" and out["loss_t"] == pytest.approx(tr[k]["t"])
    assert out["loss_since_lift_s"] == pytest.approx(tr[k]["t"] - first_lift_t)


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
