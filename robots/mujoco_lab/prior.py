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

``lab_prior`` mirrors corpus v8's draws (milestone 2a). ``make_prior`` mirrors the joint generator of corpus v9
(plan 2b): v8's draws, then per object (its entry in the fragility classes, read from ``pose["object_id"]``) the
fill, the fragility's latent normals and the filled container's own mass scale (corpus.draw_damage and
apply_damage). The joint mass is the container's (nominal x scale x (1 + f)) plus the liquid's, the CoM their
mass-weighted mean (``joint_hidden``, the generator's arithmetic). A bundle's manifest names its prior
(``prior_from_manifest``): "lab_generator" is ``lab_prior``; "lab_generator_m2b" is ``make_prior`` on the class
and threshold files the manifest names, whose sha256 it checks.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Callable

import numpy as np

from robots.mujoco_lab.corpus import (COM_S_RANGE, FILL_RANGE, FILLED_MASS_SCALE_RANGE, MASS_SCALE_RANGE, MU_RANGE, P_COM_SHIFT,
                                      P_EMPTY)
from robots.mujoco_lab.fragility import (DATA, Thresholds, entry_of, liquid, liquid_arrays, load_classes, map_latent_arrays,
                                         spill_angle)
from rpent.skills.pick_place.belief import ParticleBelief

END_FRACTION = 0.97
M2B_PRIOR = "lab_generator_m2b"
FRAG_KEYS = ("k_n_per_mm", "f_damage_n", "v_bruise_ms", "v_break_ms")


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


def joint_hidden(pose: dict[str, Any], entry: dict[str, Any], theta: dict[str, Any], com_end: int, fill: float) -> dict[str, Any]:
    """The generator's joint mass and CoM of one draw (the sim puts the liquid as a point mass at its centroid): the
    container nominal x scale x (1 + f) at its CoM hypothesis, plus the liquid of ``fill``."""
    com, f = com_hypothesis(pose["pivot"], pose["com_obj_nominal"], theta.get("com_s"), com_end)
    m_c = float(pose["nominal_mass"]) * float(theta["mass_scale"]) * (1.0 + f)
    m_l, cen = 0.0, np.zeros(3)
    if float(fill or 0.0) > 0.0:
        liq = liquid(entry["cavity"], float(fill))
        m_l, cen = float(liq["mass_kg"]), np.asarray(liq["centroid_obj"], dtype=float)
    joint = (m_c * np.asarray(com, dtype=float) + m_l * cen) / (m_c + m_l)
    return {"mass": m_c + m_l, "com_obj": joint.tolist(), "liquid_mass": m_l, "f": f}


def make_prior(classes: dict[str, dict[str, Any]], thresholds: Thresholds) -> Callable[..., ParticleBelief]:
    """The m2b lab prior: ``prior_fn(pose, n, rng)`` draws n particles of the joint generator for the object
    ``pose["object_id"]`` (an id with no entry, or none, is class none and says so in the belief's flags)."""

    def prior_fn(pose: dict[str, Any], n: int = 2000, rng: np.random.Generator | None = None) -> ParticleBelief:
        rng = rng if rng is not None else np.random.default_rng(0)
        entry, evidence = entry_of(str(pose.get("object_id")), classes)
        lo, hi = MASS_SCALE_RANGE
        scale = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
        mu = rng.uniform(*MU_RANGE, n)
        branch = rng.random(n) >= P_COM_SHIFT
        target = rng.uniform(*COM_S_RANGE, n)
        end = np.where(rng.random(n) < 0.5, 1, -1)
        fill = np.zeros(n)
        if entry.get("fillable"):
            filled = rng.random(n) >= P_EMPTY
            fill = np.where(filled, rng.uniform(*FILL_RANGE, n), 0.0)
            lo, hi = FILLED_MASS_SCALE_RANGE
            scale = np.where(filled, np.exp(rng.uniform(np.log(lo), np.log(hi), n)), scale)
            branch = branch & ~filled
        frag = map_latent_arrays(rng.standard_normal((n, 3)), entry, thresholds)
        com, f, _ = _com_hypotheses(pose["pivot"], pose["com_obj_nominal"], target, end, branch)
        m_c = float(pose["nominal_mass"]) * scale * (1.0 + f)
        m_l, cen = (liquid_arrays(entry["cavity"], fill) if entry.get("fillable") else (np.zeros(n), np.zeros((n, 3))))
        mass = m_c + m_l
        com = (m_c[:, None] * com + m_l[:, None] * cen) / mass[:, None]
        pv = pose["pivot"]
        frac = ((com - np.asarray(pv["centre"], dtype=float)) @ np.asarray(pv["axis"], dtype=float) - float(pv["lo"])) / float(pv["L"])
        spill = spill_angle(entry["cavity"], fill) if entry.get("fillable") else np.zeros(n)
        cols = {k: (frag[k].tolist() if frag[k] is not None else [None] * n) for k in FRAG_KEYS}
        parts = [{"mass": float(mass[i]), "mu": float(mu[i]), "com_obj": com[i].tolist(), "mass_scale": float(scale[i]),
                  "com_branch": bool(branch[i]), "com_s": float(target[i]) if branch[i] else None, "com_end": int(end[i]),
                  "f": float(f[i]), "com_frac": float(frac[i]), "ancestor": i, "fill": float(fill[i]), "liquid_mass": float(m_l[i]),
                  "spill_angle_deg": float(spill[i]) if fill[i] > 0 else None, **{k: cols[k][i] for k in FRAG_KEYS},
                  "frag_z": frag["z"][i].tolist()} for i in range(n)]
        return ParticleBelief(parts, np.ones(n), flags=list(evidence))

    return prior_fn


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(DATA, path)


def sha256_of(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def fragility_world(manifest: dict[str, Any] | None) -> tuple[dict[str, dict[str, Any]], Thresholds] | None:
    """(classes, thresholds) of an m2b manifest ({"path", "sha256"} each; a relative path is under robots/mujoco_lab/data),
    or None for a manifest without them (m2: no liquid, no fragility). A file whose sha256 differs is an error."""
    if not manifest or manifest.get("prior") != M2B_PRIOR:
        return None
    out = []
    for key in ("classes", "thresholds"):
        spec = manifest.get(key)
        if not spec or not spec.get("path"):
            raise ValueError(f"an {M2B_PRIOR} manifest names its {key} file: {{'path': ..., 'sha256': ...}}")
        path = _resolve(spec["path"])
        if spec.get("sha256") and sha256_of(path) != spec["sha256"]:
            raise ValueError(f"{path} changed since the bundle was trained (sha256 differs from the manifest's)")
        out.append(path)
    return load_classes(out[0]), Thresholds.from_json(out[1])


def prior_from_manifest(manifest: dict[str, Any] | None) -> Callable[..., ParticleBelief] | None:
    """The prior a bundle's manifest names: lab_prior, make_prior on its fragility world, or None."""
    p = (manifest or {}).get("prior")
    if p == "lab_generator":
        return lab_prior
    if p == M2B_PRIOR:
        return make_prior(*fragility_world(manifest))
    return None
