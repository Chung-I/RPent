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


"""Generic and VLM priors for the mass and CoM check: drop-in replacements of a bundle's class prior
``prior(pose, n, rng)``.

Each draws the class prior's particles first, so the friction (uniform 0.1 to 0.9: generic already), the fill and the
fragility stay the class prior's, and then replaces the mass and the CoM:

- mass: log-normal with median m0 and log spread sigma_mass;
- CoM fraction along the pivot: a normal around c0 with spread sigma_com, truncated to COM_RANGE (the generator's
  point-mass ends), the CoM point on the pivot's centre line (the shape's centre off the axis).

Centres. VLM: m0 is the median of the model's mass samples; c0 is the median marker's fraction (markers at FRACS) for an
object laid flat, and 0.5 for an upright container (the model is not asked there). Generic: m0 is exp of the
object-weighted mean log truth mass of the tuning objects; c0 is 0.5. Real (the exact-answer prior, 2026-11-04): m0 is
the object's real mass (the generator v11 table), c0 is 0.5: what a model that names the real mass exactly would give.
Spreads: fitted on the tuning objects only, the maximum-likelihood spread with the centre fixed (the root mean square of
the residuals, each object weighted equally); the CoM spread separately for objects laid flat and for upright containers.

``mass_only`` (2026-11-04) replaces the mass alone: the class prior's CoM draws stay, so a swap of the mass prior holds
everything else fixed.

Exact keys (W0, 2026-11-04): ``truth`` (the case's generator truth: mass, com_obj, com_frac) centres the keys in ``exact``
instead of the table: the mass log-normal with log spread EXACT_SD["mass"], the CoM fraction normal with spread
EXACT_SD["com"] (not truncated), the point moved along the pivot axis from the true CoM, so the truth's off-axis part
stays. The draws are the table's (the same n normals, then n uniforms), so a key not in ``exact`` keeps the table's draws.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.special import ndtr, ndtri

from rpent.skills.pick_place.belief import ParticleBelief

FRACS = (0.1, 0.3, 0.5, 0.7, 0.9)
COM_RANGE = (0.03, 0.97)
MIDDLE = 0.5
EXACT_SD = {"mass": 0.01, "com": 0.005}    # W0: the exact prior's log-sd of the mass and sd of the CoM fraction
EXACT_KEYS = ("mass", "com")


def _number(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def vlm_centres(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per object: the median mass answer (kg, positive answers only) and the median CoM marker's fraction (markers 1 to 5;
    objects laid flat only), with the counts of usable samples. None when no sample is usable."""
    out = {}
    for r in records:
        masses = [v for v in (_number(s.get("mass_kg")) if isinstance(s, dict) else None for s in r.get("mass") or []) if v and v > 0]
        marks = []
        for s in r.get("structured") or []:
            m = _number(s.get("com_marker")) if isinstance(s, dict) else None
            if m is not None and m == int(m) and 1 <= m <= len(FRACS):
                marks.append(FRACS[int(m) - 1])
        out[r["object"]] = {"mass": float(np.median(masses)) if masses else None, "n_mass": len(masses),
                            "com": float(np.median(marks)) if (marks and r.get("flat")) else None, "n_com": len(marks)}
    return out


def _rms(per_object: list[list[float]]) -> float | None:
    """The root of the mean over objects of each object's mean square residual."""
    ms = [float(np.mean(np.square(v))) for v in per_object if len(v)]
    return float(np.sqrt(np.mean(ms))) if ms else None


def fit_spreads(truths: dict[str, list[tuple[float, float]]], centres: dict[str, dict[str, Any]],
                flat: dict[str, bool]) -> dict[str, float | None]:
    """The spreads on the tuning objects ``truths`` (object -> [(mass, CoM fraction)] of its cases) around ``centres``."""
    mass, com_flat, com_up = [], [], []
    for o, cases in truths.items():
        c = centres[o]
        mass.append([np.log(m / c["mass"]) for m, _ in cases])
        (com_flat if flat[o] else com_up).append([s - c["com"] for _, s in cases])
    return {"mass": _rms(mass), "com_flat": _rms(com_flat), "com_upright": _rms(com_up)}


def generic_centres(truths: dict[str, list[tuple[float, float]]], objects: list[str]) -> dict[str, dict[str, float]]:
    """The same centres for every object: the tuning objects' object-weighted mean log mass, and the pivot's middle."""
    m0 = float(np.exp(np.mean([np.mean([np.log(m) for m, _ in cases]) for cases in truths.values() if cases])))
    return {o: {"mass": m0, "com": MIDDLE} for o in objects}


def shape_prior(base: Callable[..., ParticleBelief], table: dict[str, dict[str, float]], mass_only: bool = False,
                truth: dict[str, Any] | None = None, exact: tuple[str, ...] = ()) -> Callable[..., ParticleBelief]:
    """``prior(pose, n, rng)``: the base prior's particles with the mass and the CoM of ``table[pose["object_id"]]``
    (keys mass, com, sigma_mass, sigma_com); the draws come after the base prior's, from the same stream. ``mass_only``
    replaces the mass alone and keeps the base particles' CoM keys. The keys in ``exact`` (mass, com) centre on ``truth``
    instead (module doc); an exact CoM with ``mass_only``, an exact key without a truth or an unknown key is a ValueError."""
    exact = tuple(exact)
    if set(exact) - set(EXACT_KEYS):
        raise ValueError(f"unknown exact keys {sorted(set(exact) - set(EXACT_KEYS))}")
    if exact and truth is None:
        raise ValueError("an exact key needs the case's truth")
    if mass_only and "com" in exact:
        raise ValueError("mass_only keeps the base CoM: no exact CoM")

    def prior_fn(pose: dict[str, Any], n: int = 2000, rng: np.random.Generator | None = None) -> ParticleBelief:
        rng = rng if rng is not None else np.random.default_rng(0)
        t = table[str(pose["object_id"])]
        b = base(pose, n, rng)
        m0, sig_m = (float(truth["mass"]), EXACT_SD["mass"]) if "mass" in exact else (float(t["mass"]), float(t["sigma_mass"]))
        mass = m0 * np.exp(sig_m * rng.standard_normal(n))
        if mass_only:
            parts = [{**p, "mass": float(mass[i]), "mass_scale": float("nan")} for i, p in enumerate(b.particles)]
            return ParticleBelief(parts, np.ones(n), flags=list(b.flags))
        pv = pose["pivot"]
        c, ax = np.asarray(pv["centre"], dtype=float), np.asarray(pv["axis"], dtype=float)
        if "com" in exact:
            c0 = float(truth["com_frac"])
            frac = c0 + EXACT_SD["com"] * ndtri(np.clip(rng.random(n), 1e-12, 1.0 - 1e-12))
            com = np.asarray(truth["com_obj"], dtype=float) + np.outer((frac - c0) * float(pv["L"]), ax)
        else:
            lo, hi = (ndtr((x - float(t["com"])) / float(t["sigma_com"])) for x in COM_RANGE)
            frac = np.clip(float(t["com"]) + float(t["sigma_com"]) * ndtri(lo + (hi - lo) * rng.random(n)), *COM_RANGE)
            com = c + np.outer(float(pv["lo"]) + frac * float(pv["L"]), ax)
        parts = [{**p, "mass": float(mass[i]), "com_obj": com[i].tolist(), "com_frac": float(frac[i]), "mass_scale": float("nan"),
                  "com_branch": False, "com_s": None, "f": 0.0} for i, p in enumerate(b.particles)]
        return ParticleBelief(parts, np.ones(n), flags=list(b.flags))

    return prior_fn


def prior_table(kind: str, objects: list[str], tuning: list[str], truths: dict[str, list[tuple[float, float]]],
                flat: dict[str, bool], records: list[dict[str, Any]] | None = None,
                real: dict[str, float] | None = None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """The table of ``shape_prior`` for the held-out ``objects`` (``kind`` generic, vlm or real) and the spreads, fitted on
    the ``tuning`` objects' truths only. A VLM object without a usable mass answer takes the generic mass centre, one laid
    flat without a usable marker the pivot's middle (listed under ``fallback``); an upright one is not asked (0.5). The
    real prior centres each object on ``real[object]`` (kg); an object without one is a ValueError."""
    tune = {o: truths[o] for o in tuning if truths.get(o)}
    generic = generic_centres(tune, sorted(set(objects) | set(tune)))
    if kind == "generic":
        centres = {o: dict(generic[o], fallback=[]) for o in generic}
    elif kind == "vlm":
        answers = vlm_centres(records or [])
        centres = {}
        for o in generic:
            a = answers.get(o) or {}
            fb = [k for k in ("mass", "com") if a.get(k) is None and (k == "mass" or flat[o])]
            centres[o] = {"mass": a.get("mass") or generic[o]["mass"], "com": a.get("com") if a.get("com") is not None else MIDDLE,
                          "fallback": fb}
    elif kind == "real":
        missing = sorted(o for o in generic if (real or {}).get(o) is None)
        if missing:
            raise ValueError(f"no real mass for {', '.join(missing)}")
        centres = {o: {"mass": float(real[o]), "com": MIDDLE, "fallback": []} for o in generic}
    else:
        raise ValueError(f"unknown prior {kind!r}")
    s = fit_spreads(tune, centres, flat)
    table = {}
    for o in objects:
        sig_c = s["com_flat" if flat[o] else "com_upright"]
        if not s["mass"] or not sig_c:
            raise ValueError(f"{o}: no tuning spread for its group (spreads {s})")
        table[o] = {**centres[o], "sigma_mass": s["mass"], "sigma_com": sig_c}
    return table, s
