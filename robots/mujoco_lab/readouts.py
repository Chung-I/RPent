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
with ``t``, ``phase``, ``z``, ``pads``, ``fz``, ``rot_deg``, ``pitch_ey``) and the label record.
"""

from __future__ import annotations

from typing import Any

import numpy as np

RISEN_M = 0.012              # the weigh probe's "risen" height: the pads carry the object above it
MIN_HOLD_STEPS = 5
ROT_SIDE_DEG = 10.0          # a rotation this large tells the heavy side
PITCH_DEADBAND = 0.05        # |pitch_ey| below this is a turn about another axis: no side
LOSS_PHASES = ("lift", "hold", "transport")


def _both(x: dict[str, Any]) -> bool:
    return bool(x["pads"][0] and x["pads"][1])


def summarize(trace: list[dict[str, Any]], label_out: dict[str, Any], v_lift: float) -> dict[str, Any]:
    """The readouts of one rollout (see the keys below); None where a reading does not exist."""
    label = label_out["label"]
    lifted = [x for x in trace if x["phase"] in ("lift", "hold")]
    hold = [x["fz"] for x in trace if x["phase"] == "hold" and _both(x) and x["z"] > RISEN_M]
    slip = [x["fz"] for x in trace if x["phase"] in ("close", "lift") and _both(x) and x["z"] <= RISEN_M]
    loss_step, loss_phase = label_out.get("loss_step"), label_out.get("loss_phase")
    fz_loss = None
    if loss_step is not None and loss_phase in LOSS_PHASES:
        prev = next((trace[i] for i in range(int(loss_step) - 1, -1, -1) if _both(trace[i])), None)
        fz_loss = None if prev is None else float(prev["fz"])
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
        "fz_slip_max_n": float(max(slip)) if (label == "not_lifted" and slip) else None,
        "fz_at_loss_n": fz_loss,
        "rot_deg": rot,
        "heavy_side": side,
        "loss_phase": loss_phase,
        "loss_t": loss_t,
        "loss_since_lift_s": None if (loss_t is None or first_lift is None) else float(loss_t - first_lift),
        "v_lift": float(v_lift),
    }
