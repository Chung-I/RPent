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

"""The failed-attempt rule table: what the readouts say went wrong, in words for the planner.

Rules over one attempt's label and readouts (see ``robots.mujoco_lab.readouts.summarize`` for the
keys; a missing key is a missing reading):

* not_lifted: never left the table (too heavy or too slippery for this squeeze), with the friction
  the slip force reads;
* a loss in lift or hold without rotation: rose then slid (slippery or heavy at this squeeze), with
  the friction the force before the loss reads;

  (both friction numbers are fz / 2F, corrected by exp(-b) when the readout model's fitted law
  (b, sd) for the term is given: the lab's forces read low, b about -1 for slip and -2.3 for a loss;
  the law's sd gives the 80 % range of that one reading. A force that is not positive, or a friction
  outside the belief's range, gives no number: the line says why);
* a rotation of 10 degrees or more: the CoM is off the pinch, toward the side the readouts tell
  (before the fall, or in the hand when nothing fell);
* a loss within 0.3 s of the lift start at a lift speed of 0.4 m/s or more: lift slower;
* tipped or missed: a set-down failure.

The last line is always the belief before and after the update. The diagnosis explains; the belief
update and the fresh menu decide.
"""

from __future__ import annotations

from typing import Any

import numpy as np

ROT_DEG = 10.0
FAST_LIFT = 0.4               # m/s
EARLY_LOSS_S = 0.3
Z80 = 1.2816                  # the two-sided 80 % normal quantile
UNUSABLE = "not usable"


def _friction(fz: float, force: float, law: tuple[float, float] | None,
              mu_range: tuple[float, float] | None = None) -> tuple[float | None, str]:
    """(friction read from a vertical pad force, the note that goes with it). The fitted law (b, sd)
    corrects the reading by exp(-b), and its sd gives the 80 % range of this one reading. (None,
    UNUSABLE) when the force is not positive; (None, "outside the belief's range ...") when the
    friction lies outside ``mu_range``."""
    if not float(fz) > 0:
        return None, UNUSABLE
    mu = float(fz) / (2.0 * float(force))
    note = ""
    if law is not None:
        mu *= float(np.exp(-float(law[0])))
        f = float(np.exp(Z80 * float(law[1])))
        note = f" (corrected by the fitted readout law; 80 % range from this one reading: {mu / f:.2f} to {mu * f:.2f})"
    if mu_range is not None and not float(mu_range[0]) <= mu <= float(mu_range[1]):
        return None, f"outside the belief's range {float(mu_range[0]):.2f} to {float(mu_range[1]):.2f}"
    return mu, note


def _num(x: Any, fmt: str) -> str:
    return "?" if x is None else format(float(x), fmt)


def belief_line(before: dict[str, Any], after: dict[str, Any]) -> str:
    """One line: mass and friction medians, the CoM-shift probability and the effective particles, before → after."""
    parts = [f"mass median {_num(before.get('mass_median'), '.3f')} → {_num(after.get('mass_median'), '.3f')} kg",
             f"friction median {_num(before.get('mu_median'), '.2f')} → {_num(after.get('mu_median'), '.2f')}"]
    if before.get("p_com_shifted") is not None or after.get("p_com_shifted") is not None:
        parts.append(f"P(CoM shifted) {_num(before.get('p_com_shifted'), '.2f')} → {_num(after.get('p_com_shifted'), '.2f')}")
    if before.get("ess") is not None or after.get("ess") is not None:
        parts.append(f"effective particles {_num(before.get('ess'), '.0f')} → {_num(after.get('ess'), '.0f')}")
    return "belief: " + ", ".join(parts)


def diagnose(label: str, readouts: dict[str, Any], action: dict[str, Any], before: dict[str, Any], after: dict[str, Any],
             laws: dict[str, tuple[float, float] | None] | None = None,
             mu_range: tuple[float, float] | None = None) -> list[str]:
    """The rule table's lines for one attempt (label, readouts, the executed action with ``force`` and
    ``v_lift``), then the belief line (``ParticleBelief.to_data`` before and after the update).
    ``laws``: the readout model's fitted (b, sd) per term ("slip", "loss") for this label, if any.
    ``mu_range``: the friction range the belief holds; a friction reading outside it gives no number."""
    r = readouts or {}
    laws = laws or {}
    force = float(action.get("force") or 0.0)
    rot = float(r.get("rot_deg") or 0.0)
    phase = r.get("loss_phase")
    lines: list[str] = []
    if label == "not_lifted":
        lines.append("never left the table: too heavy or too slippery for this squeeze")
        fz = r.get("fz_slip_max_n")
        if fz is not None and force > 0:
            mu, note = _friction(fz, force, laws.get("slip"), mu_range)
            if mu is not None:
                lines.append(f"the pads slipped at {float(fz):.1f} N of lift force: friction about {mu:.2f} at {force:.0f} N per pad"
                             f"{note}, or a weight above {float(fz):.1f} N")
            elif note == UNUSABLE:
                lines.append("the pads' lift-force reading before the slip is not usable (not positive): the belief line carries the update")
            else:
                lines.append(f"the pads slipped at {float(fz):.1f} N of lift force: the friction this reads at {force:.0f} N per pad lies "
                             f"{note}, so it gives no number")
    if phase in ("lift", "hold") and rot < ROT_DEG:
        lines.append("rose then slid without rotating: slippery or heavy at this squeeze")
        fz = r.get("fz_at_loss_n")
        if fz is not None and force > 0:
            mu, note = _friction(fz, force, laws.get("loss"), mu_range)
            if mu is not None:
                lines.append(f"the pads held {float(fz):.1f} N up before the loss: friction about {mu:.2f} at {force:.0f} N per pad{note}")
            elif note == UNUSABLE:
                lines.append("the pads' lift-force reading before the loss is not usable (not positive): the belief line carries the update")
            else:
                lines.append(f"the pads held {float(fz):.1f} N up before the loss: the friction this reads at {force:.0f} N per pad lies "
                             f"{note}, so it gives no number")
    if rot >= ROT_DEG:
        side = int(r.get("heavy_side") or 0)
        where = "the CoM is off the pinch" + ("" if side == 0 else f" toward the {'+' if side > 0 else '−'} end")
        lines.append(f"rotated {rot:.0f}° {'before the fall' if phase is not None else 'in the hand'}: {where}")
    since = r.get("loss_since_lift_s")
    if since is not None and float(since) < EARLY_LOSS_S and float(action.get("v_lift") or 0.0) >= FAST_LIFT:
        lines.append("lost at the start of a fast lift: lift slower")
    if label in ("tipped", "missed"):
        lines.append(f"set-down: {label}")
    lines.append(belief_line(before, after))
    return lines
