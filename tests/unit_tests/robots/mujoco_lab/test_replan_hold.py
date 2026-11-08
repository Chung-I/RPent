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

"""The re-plan at the hold (design 2026-11-08-sixdof-rerun-design.md, amendment 8, rules (a), (d), (e), (f)): with the
executor's flag on, the carry rises by d when both pads hold the object at the end of the hold and it hangs more than
3 mm below its planned offset (the dead band, the coordinator's ruling of 2026-11-09); with the flag off, and with the flag on but no raise, a rollout is the executor's before
the re-plan byte for byte."""

import hashlib
import os

import numpy as np
import pytest

from robots.mujoco_lab import adapter
from robots.mujoco_lab import curve_sixdof as cs
from robots.mujoco_lab.adapter import LIFT_H, ramp_profile, replan_at_hold

SIM = os.environ.get("RPENT_MUJOCO_SIM_DIR", "")
ARM = pytest.mark.skipif(not (SIM and os.path.exists(os.path.join(SIM, "ident.py"))), reason="not the arm sim")
KEYS = {"held", "lifted", "planned_mm", "measured_mm", "d_mm", "raise_mm", "unreachable"}


# ------------------------------------------------------------------ the reading at the end of the hold

def _step(low_z, hand_z, pads=(True, True), phase="hold"):
    return {"phase": phase, "low_z": low_z, "hand_z": hand_z, "pads": list(pads)}


def test_the_carry_rises_by_d_when_both_pads_hold_and_the_object_hangs_more_than_3_mm_below_its_planned_offset():
    rec, rise = replan_at_hold([_step(0.10, 0.20, phase="lift"), _step(0.0955, 0.20)], -0.100, True)
    assert rise == pytest.approx(0.0045, abs=1e-12)
    assert rec == {"held": True, "lifted": True, "planned_mm": pytest.approx(-100.0), "measured_mm": pytest.approx(-104.5),
                   "d_mm": pytest.approx(4.5), "raise_mm": 0.0, "unreachable": False}


def test_within_the_3_mm_dead_band_or_with_the_object_higher_than_planned_the_carry_stays():
    for low, d_mm in ((0.0989, 1.1), (0.0971, 2.9), (0.1150, -15.0)):  # a firm grasp's turn; the band's edge; higher
        rec, rise = replan_at_hold([_step(low, 0.20)], -0.100, True)
        assert rise == 0.0 and rec["d_mm"] == pytest.approx(d_mm) and rec["raise_mm"] == 0.0
    rec, rise = replan_at_hold([_step(0.0969, 0.20)], -0.100, True)
    assert rise == pytest.approx(0.0031, abs=1e-12)


def test_without_both_pads_on_the_object_at_the_end_of_the_hold_or_an_object_that_rose_the_carry_stays():
    rec, rise = replan_at_hold([_step(0.0955, 0.20), _step(0.0955, 0.20, pads=(True, False))], -0.100, True)
    assert rise == 0.0 and rec["held"] is False and rec["d_mm"] == pytest.approx(4.5)
    rec, rise = replan_at_hold([_step(0.0955, 0.20)], -0.100, False)
    assert rise == 0.0 and rec["lifted"] is False and rec["held"] is True


def test_the_reading_is_the_last_hold_steps():
    rec, rise = replan_at_hold([_step(0.050, 0.20, pads=(False, True)), _step(0.0995, 0.20)], -0.100, True)
    assert rise == 0.0 and rec["held"] is True and rec["measured_mm"] == pytest.approx(-100.5)


def test_a_trace_without_a_hold_has_no_reading_and_no_raise():
    rec, rise = replan_at_hold([_step(0.10, 0.20, phase="lift")], -0.100, True)
    assert rise == 0.0 and set(rec) == KEYS
    assert rec["held"] is False and rec["measured_mm"] is None and rec["d_mm"] is None and rec["planned_mm"] == pytest.approx(-100.0)


# ------------------------------------------------------------------ the executor on the arm sim

def _sim_sha(d):
    h = hashlib.sha256()
    for root, dirs, files in sorted(os.walk(d)):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        for f in sorted(files):
            p = os.path.join(root, f)
            h.update(os.path.relpath(p, d).encode())
            h.update(open(p, "rb").read())
    return h.hexdigest()[:16]


SIMS = {"f49d17664ec1f66d": "fe46f6c", "f7d6964ef8bc1a35": "ee882f9b"}    # the sim dir's file hash: its daily-logs commit
ROWS = {"jello": ("ycb__jello", 0, 40.0), "spam": ("ycb__spam_can", 1, 3.0), "soup": ("ycb__tomato_soup_can", 0, 40.0),
        "tuna": ("ycb__tuna_can", 0, 3.0)}
_JELLO = ("placed_stable", "4ed96739d9aa3d58", "b7980d390904bd95")      # d -0.12 mm on both sims
_SPAM = ("placed_stable", "f7978460352b8fd9", "35ff9aab25c99ea8")       # d 4.74 mm on both sims
_TUNA = ("not_lifted", "96391592284e066f", "5f4bd5c6d052c197")          # does not rise
# (label, trace_sha, qpos_sha) through the executor before the re-plan (RPent sixdof 62c2089), by row and sim; the soup
# can sits 14.2 mm higher than planned on fe46f6c, and with the boxes' contacts (ee882f9b) it hangs 45.9 mm lower
GOLDEN = {"jello": {"fe46f6c": _JELLO, "ee882f9b": _JELLO}, "spam": {"fe46f6c": _SPAM, "ee882f9b": _SPAM},
          "tuna": {"fe46f6c": _TUNA, "ee882f9b": _TUNA},
          "soup": {"fe46f6c": ("placed_stable", "79fa7950bd91a745", "c1acdbd45341ba93")}}


def _golden(row):
    sim = SIMS.get(_sim_sha(SIM)) if SIM else None
    if sim is None or sim not in GOLDEN[row]:
        pytest.skip(f"no golden for {row} on this sim")
    return GOLDEN[row][sim]


@pytest.fixture()
def lab(monkeypatch):
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    return MuJoCoLabEnv(damage_meter=False)


def _run(env, row, replan=None):
    """The row after its reset (flag set when ``replan`` is given): (rollout, (label, trace_sha, qpos_sha), the commanded
    hand heights in order, the hand's start height)."""
    o, i, f = ROWS[row]
    env.reset(o, {"mass_scale": 1.0, "mu": 0.8, "com_s": None, "v_lift": 0.1}, seed=3)
    if replan is not None:
        env.replan_hold = replan
    calls, plain = [], env.sim.set_hand

    def set_hand(pos, yaw):
        calls.append(float(pos[2]))
        plain(pos, yaw)

    env.sim.set_hand = set_hand
    p = env.proposals()[i]
    r = env.execute(p, f, 0.1, 0.5, "mouse_pad")
    out = cs.outcome(env, r, True)
    return r, (out["label"], out["trace_sha"], out["qpos_sha"]), calls, env.s["grasp"].hand_start_z(np.asarray(p.point, dtype=float))


@ARM
def test_the_flag_is_off_by_default(lab):
    assert lab.replan_hold is False


@ARM
@pytest.mark.parametrize("row", sorted(ROWS))
def test_with_the_flag_off_a_rollout_is_the_executors_before_the_re_plan_byte_for_byte(lab, row):
    want = _golden(row)
    r, got, _, _ = _run(lab, row)
    assert got == want and not hasattr(r, "replan")


@ARM
@pytest.mark.parametrize("row", ["jello", "soup", "tuna"])
def test_with_the_flag_on_a_row_without_a_raise_is_byte_identical_and_records_its_reading(lab, row):
    want = _golden(row)
    r, got, _, _ = _run(lab, row, replan=True)
    assert got == want
    rec = r.replan
    assert set(rec) == KEYS and rec["raise_mm"] == 0.0 and rec["unreachable"] is False
    assert rec["d_mm"] == pytest.approx(rec["planned_mm"] - rec["measured_mm"])
    assert rec["d_mm"] <= 3.0 or not (rec["held"] and rec["lifted"])


@ARM
def test_with_the_flag_on_a_held_object_hanging_more_than_3_mm_low_is_carried_d_higher(lab):
    want = _golden("spam")
    off, _, off_calls, hz0 = _run(lab, "spam")
    r, got, calls, _ = _run(lab, "spam", replan=True)
    rec = r.replan
    d = rec["d_mm"] / 1e3
    assert rec["held"] and rec["lifted"] and d > 3e-3 and rec["raise_mm"] == rec["d_mm"] and rec["unreachable"] is False
    assert max(off_calls) == pytest.approx(hz0 + LIFT_H, abs=1e-12)
    assert max(calls) == pytest.approx(hz0 + LIFT_H + d, abs=1e-12)
    # the raise: "transport" steps at the lift's speed right after the hold; the carry then runs at the raised height
    hz = lab.s["scene"].CTRL_HZ
    n_raise = len(ramp_profile(d, 0.1, hz))
    phases = [s["phase"] for s in r.trace]
    assert phases.count("transport") == [s["phase"] for s in off.trace].count("transport") + n_raise
    k = phases.index("transport")
    assert phases[k - 1] == "hold" and phases[k:k + n_raise] == ["transport"] * n_raise
    assert sum(abs(z - (hz0 + LIFT_H + d)) < 1e-12 for z in calls) >= phases.count("transport") - n_raise + 1
    assert got[1] != want[1]


@ARM
def test_a_raise_the_arm_cannot_reach_keeps_the_planned_carry_and_records_raise_unreachable(lab, monkeypatch):
    want = _golden("spam")
    monkeypatch.setattr(adapter, "REACH_TOL_M", -1.0)                  # every step of the raised carry fails the reach check
    r, got, calls, hz0 = _run(lab, "spam", replan=True)
    assert got == want and max(calls) == pytest.approx(hz0 + LIFT_H, abs=1e-12)
    assert r.replan["unreachable"] is True and r.replan["raise_mm"] == 0.0 and r.replan["d_mm"] > 3.0


@ARM
def test_the_reach_check_covers_the_transport_at_the_raised_height_not_only_the_raise(lab):
    want = _golden("spam")
    o, i, f = ROWS["spam"]
    lab.reset(o, {"mass_scale": 1.0, "mu": 0.8, "com_s": None, "v_lift": 0.1}, seed=3)
    lab.replan_hold = True
    p = lab.proposals()[i]
    x0, top = np.asarray(p.point, dtype=float)[:2], lab.s["grasp"].hand_start_z(np.asarray(p.point, dtype=float)) + LIFT_H
    arm, plain = lab.sim.arm, lab.sim.arm.ik

    def ik(pos, yaw, q_init=None, **k):                 # the arm cannot carry the object sideways above the planned height
        q, ep, er = plain(pos, yaw, q_init=q_init, **k)
        moved = float(np.hypot(pos[0] - x0[0], pos[1] - x0[1])) > 1e-9
        return q, (1.0 if moved and pos[2] > top + 1e-6 else ep), er

    arm.ik = ik
    r = lab.execute(p, f, 0.1, 0.5, "mouse_pad")
    out = cs.outcome(lab, r, True)
    assert (out["label"], out["trace_sha"], out["qpos_sha"]) == want
    assert r.replan["unreachable"] is True and r.replan["raise_mm"] == 0.0
