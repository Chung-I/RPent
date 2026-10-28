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


"""Readouts of one rollout: what the pads felt and how the object turned, for the belief update.

Pure functions over the trace that ``MuJoCoLabEnv.execute`` records (one dict per control step
with ``t``, ``phase``, ``z``, ``pads``, ``fz`` (a mean over the step's physics substeps), ``pad_nz``,
``hand_z``, ``rot_deg``, ``rot_lift_deg``, ``grip_dz``, ``pitch_ey``) and the label record.

A slip force tells the friction (2 mu F on vertical faces) only while the pads slide up an object that
stays put. It is read on the lift, not the close (the pads strike the object at several times the
squeeze while the table still holds it), and not while the object point between the pads rises with
the hand or the object turns (a pivot: the force is a share of the weight) or a pad presses a top face
(the pads landed on it). The body origin does not tell a pivot: it can sit next to the table contact
the object turns about (potato_masher d57: it rose 0.1 mm while the pads rose 10 mm).
"""

from __future__ import annotations

from typing import Any

import numpy as np

RISEN_M = 0.012              # the weigh probe's "risen" height: the pads carry the object above it
MIN_HOLD_STEPS = 5
ROT_SIDE_DEG = 10.0          # a rotation this large tells the heavy side
PITCH_DEADBAND = 0.05        # |pitch_ey| below this is a turn about another axis: no side
LOSS_PHASES = ("lift", "hold", "transport")
SLIDE_REL = 0.10             # a slide: since the lift began the object point between the pads rose under a tenth
SLIDE_SLACK_M = 0.0005       # of the hand's rise, plus 0.5 mm of contact give,
SLIDE_DEG = 3.0              # and the object turned at most 3 deg
SLIDE_MIN_M = 0.001          # a step counts once the pads have slid 1 mm up that point (past the give)
SIDE_NZ = 0.5                # a pad on a side face: its contact normal's vertical part at most 0.5 (30 deg)
MIN_SLIDE_STEPS = 2          # a slide reading needs two counted steps
LOSS_STEPS = 3               # the loss reading: the mean force over the last three two-pad steps before it


def _both(x: dict[str, Any]) -> bool:
    return bool(x["pads"][0] and x["pads"][1])


def _sides(x: dict[str, Any]) -> bool:
    nz = x.get("pad_nz") or (None, None)
    return all(n is not None and abs(float(n)) <= SIDE_NZ for n in nz)


def slide_force(trace: list[dict[str, Any]]) -> float | None:
    """The largest net vertical pad force while the pads slide up a still object. The run starts at the
    first lift step and lasts while both pads touch side faces, the object point between the pads has
    risen less than SLIDE_REL of the hand's rise (plus SLIDE_SLACK_M) and the object has turned at most
    SLIDE_DEG since the lift began. A step counts once the pads have slid SLIDE_MIN_M up that point. None
    when fewer than MIN_SLIDE_STEPS steps count: the object rose with the hand or turned about a far
    contact (a pivot), a pad pressed a top face, or the pads lost it."""
    i0 = next((i for i, x in enumerate(trace) if x["phase"] == "lift"), None)
    if not i0 or trace[i0 - 1].get("hand_z") is None:
        return None
    ref = trace[i0 - 1]
    run: list[float] = []
    for x in trace[i0:]:
        turn, grip = x.get("rot_lift_deg"), x.get("grip_dz")
        if (x["phase"] != "lift" or not _both(x) or not _sides(x) or x.get("hand_z") is None or turn is None
                or grip is None):
            break
        hand = float(x["hand_z"]) - float(ref["hand_z"])
        if float(turn) > SLIDE_DEG or float(grip) > SLIDE_REL * max(hand, 0.0) + SLIDE_SLACK_M:
            break
        if hand - float(grip) >= SLIDE_MIN_M:
            run.append(float(x["fz"]))
    return max(run) if len(run) >= MIN_SLIDE_STEPS else None


def summarize(trace: list[dict[str, Any]], label_out: dict[str, Any], v_lift: float) -> dict[str, Any]:
    """The readouts of one rollout (see the keys below); None where a reading does not exist."""
    label = label_out["label"]
    lifted = [x for x in trace if x["phase"] in ("lift", "hold")]
    hold = [x["fz"] for x in trace if x["phase"] == "hold" and _both(x) and x["z"] > RISEN_M]
    slip = slide_force(trace) if label == "not_lifted" else None
    loss_step, loss_phase = label_out.get("loss_step"), label_out.get("loss_phase")
    fz_loss = None
    if loss_step is not None and loss_phase in LOSS_PHASES:
        prev = [trace[i]["fz"] for i in range(int(loss_step) - 1, -1, -1) if _both(trace[i])][:LOSS_STEPS]
        fz_loss = float(np.mean(prev)) if prev else None
    if loss_step is not None:
        ref = trace[int(loss_step)]
    else:
        holds = [x for x in trace if x["phase"] == "hold"]
        ref = holds[-1] if holds else None
    rot = 0.0 if ref is None else float(ref["rot_deg"])
    pitch = 0.0 if ref is None else float(ref.get("pitch_ey", 0.0))
    side = int(-np.sign(pitch)) if (rot >= ROT_SIDE_DEG and abs(pitch) >= PITCH_DEADBAND) else 0
    first_lift = next((x["t"] for x in trace if x["phase"] == "lift"), None)
    loss_t = label_out.get("loss_t")
    return {
        "rose": label != "not_lifted",
        "max_rise_m": float(max((x["z"] for x in lifted), default=0.0)),
        "fz_hold_n": float(np.median(hold)) if len(hold) >= MIN_HOLD_STEPS else None,
        "fz_slip_max_n": slip,
        "fz_at_loss_n": fz_loss,
        "rot_deg": rot,
        "heavy_side": side,
        "loss_phase": loss_phase,
        "loss_t": loss_t,
        "loss_since_lift_s": None if (loss_t is None or first_lift is None) else float(loss_t - first_lift),
        "v_lift": float(v_lift),
    }
