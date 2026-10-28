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


"""The lab prior over the hidden properties: the corpus generator itself.

``com_hypothesis`` repeats the sim's CoM-shift arithmetic (``t3_make.com_shift_for``): a point mass
at 0.97 (end +1) or 0.03 (end −1) of the pivot extent, with the mass fraction f that moves the
CoM to the target fraction measured from the other end; f clamps at 0 when the target lies on
the near side of the nominal CoM.
"""

from __future__ import annotations

from typing import Any

import numpy as np

END_FRACTION = 0.97


def com_hypothesis(pivot: dict[str, Any], com_obj_nominal, target_s: float | None, end: int) -> tuple[list[float], float]:
    """(CoM in the object frame, point-mass fraction f) of one generator hypothesis."""
    com_nom = np.asarray(com_obj_nominal, dtype=float)
    if target_s is None:
        return com_nom.tolist(), 0.0
    c, ax = np.asarray(pivot["centre"], dtype=float), np.asarray(pivot["axis"], dtype=float)
    lo, L = float(pivot["lo"]), float(pivot["L"])
    s_nom = float(((com_nom - c) @ ax - lo) / L)
    t = float(target_s)
    if end >= 0:
        end_pt = c + ax * (lo + END_FRACTION * L)
        f = max(0.0, (t - s_nom) / (END_FRACTION - t))
    else:
        end_pt = c + ax * (lo + (1.0 - END_FRACTION) * L)
        f = max(0.0, (t - (1.0 - s_nom)) / (END_FRACTION - t))
    return ((com_nom + f * end_pt) / (1.0 + f)).tolist(), float(f)
