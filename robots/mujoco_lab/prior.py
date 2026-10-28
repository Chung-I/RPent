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

from robots.mujoco_lab.corpus import COM_S_RANGE, MASS_SCALE_RANGE, MU_RANGE, P_COM_SHIFT
from rpent.skills.pick_place.belief import ParticleBelief

END_FRACTION = 0.97


def _com_hypotheses(pivot: dict[str, Any], com_obj_nominal, target: np.ndarray, end: np.ndarray,
                    branch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised hypotheses: (CoM in the object frame (n, 3), f (n,), CoM fraction along the pivot (n,))."""
    com_nom = np.asarray(com_obj_nominal, dtype=float)
    c, ax = np.asarray(pivot["centre"], dtype=float), np.asarray(pivot["axis"], dtype=float)
    lo, L = float(pivot["lo"]), float(pivot["L"])
    s_nom = float(((com_nom - c) @ ax - lo) / L)
    t = np.asarray(target, dtype=float)
    plus = np.asarray(end) >= 0
    end_pt = np.where(plus[:, None], c + ax * (lo + END_FRACTION * L), c + ax * (lo + (1.0 - END_FRACTION) * L))
    with np.errstate(invalid="ignore"):
        f = np.where(plus, (t - s_nom) / (END_FRACTION - t), (t - (1.0 - s_nom)) / (END_FRACTION - t))
    f = np.where(np.asarray(branch, dtype=bool), np.maximum(0.0, np.nan_to_num(f, nan=0.0)), 0.0)
    com = (com_nom + f[:, None] * end_pt) / (1.0 + f)[:, None]
    return com, f, ((com - c) @ ax - lo) / L


def com_hypothesis(pivot: dict[str, Any], com_obj_nominal, target_s: float | None, end: int) -> tuple[list[float], float]:
    """(CoM in the object frame, point-mass fraction f) of one generator hypothesis."""
    branch = target_s is not None
    com, f, _ = _com_hypotheses(pivot, com_obj_nominal, np.array([target_s if branch else 0.5]), np.array([end]), np.array([branch]))
    return com[0].tolist(), float(f[0])


def lab_prior(pose: dict[str, Any], n: int, rng: np.random.Generator) -> ParticleBelief:
    """n draws of the corpus generator (``corpus.draw_theta``) as equally weighted particles: mass scale
    log-uniform, friction uniform, with p 0.5 the CoM-shift branch (a target fraction and an end),
    mass = nominal x scale x (1 + f)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    lo, hi = MASS_SCALE_RANGE
    scale = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    mu = rng.uniform(*MU_RANGE, n)
    branch = rng.random(n) >= P_COM_SHIFT                     # draw_theta: com_s is None with p P_COM_SHIFT
    target = rng.uniform(*COM_S_RANGE, n)
    end = np.where(rng.random(n) < 0.5, 1, -1)
    com, f, frac = _com_hypotheses(pose["pivot"], pose["com_obj_nominal"], target, end, branch)
    mass = float(pose["nominal_mass"]) * scale * (1.0 + f)
    parts = [{"mass": float(mass[i]), "mu": float(mu[i]), "com_obj": com[i].tolist(), "mass_scale": float(scale[i]),
              "com_branch": bool(branch[i]), "com_s": float(target[i]) if branch[i] else None, "com_end": int(end[i]),
              "f": float(f[i]), "com_frac": float(frac[i]), "ancestor": i} for i in range(n)]
    return ParticleBelief(parts, np.ones(n))
