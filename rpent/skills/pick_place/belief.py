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

"""Belief over the hidden properties (mass, friction, centre of mass).

``ParticleBelief`` is the belief the skill holds: weighted particles, each a full hypothesis
(mass in kg, friction, CoM in the object frame, and the generator's keys when the prior is the
lab's corpus generator). Attempts and probes reweight it; see ``tools.update_belief``. When the
weights collapse, ``refresh`` redraws fresh prior particles and weights them by the whole history.
``Belief`` is the milestone 1 parametric prior (volume times material density, uniform
friction), kept as the fallback for substrates without a generator
(``ParticleBelief.from_parametric``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from rpent.skills.pick_place.features import DAMAGE_PARTICLE_KEYS

# kg/m^3, a copy of the sim's MATERIAL table (sim/objects.py; a test checks they agree);
# hollow objects are shells at a quarter of the density, as prepare.py applies it
RHO: dict[str, float] = {
    "steel": 7800.0,
    "wood": 700.0,
    "hard_plastic": 1000.0,
    "thin_plastic": 300.0,
    "ceramic": 2300.0,
    "cardboard": 250.0,
    "metal_can": 900.0,
    "fruit": 950.0,
    "foam": 100.0,
    "rubber": 1100.0,
    "electronics": 1200.0,
    "glass": 2400.0,
    "unknown": 1000.0,
}
HOLLOW_FACTOR = 0.25
MASS_LOG_SIGMA = 0.5
MU_RANGE = (0.1, 0.9)
LEVER_SIGMA = 0.02
LEVER_SIGMA_PER_LENGTH = 0.15     # the corpus shifts the CoM to 0.3 to 0.8 of the extent with p 0.5
N_PARTICLES = 2000
MIN_ESS = 100.0                   # fewer effective particles than this: refresh, then report degenerate
REFRESH_DRAWS = (20000, 100000)


@dataclass
class Belief:
    mass_log_mu: float
    mass_log_sigma: float
    mu_lo: float
    mu_hi: float
    lever_sigma: float
    visibility: float

    @staticmethod
    def prior(volume_m3: float, material: str, hollow: bool = False, visibility: float = 1.0,
              length: float | None = None) -> Belief:
        rho = RHO.get(material, RHO["unknown"])
        median = max(1e-4, float(volume_m3) * rho * (HOLLOW_FACTOR if hollow else 1.0))
        sigma = MASS_LOG_SIGMA + 0.5 * (1.0 - float(np.clip(visibility, 0.0, 1.0)))
        lever_sigma = LEVER_SIGMA_PER_LENGTH * float(length) if length else LEVER_SIGMA
        return Belief(mass_log_mu=float(np.log(median)), mass_log_sigma=float(sigma), mu_lo=MU_RANGE[0],
                      mu_hi=MU_RANGE[1], lever_sigma=float(lever_sigma), visibility=float(visibility))

    def sample(self, n: int, rng: np.random.Generator) -> list[dict[str, float]]:
        mass = np.exp(rng.normal(self.mass_log_mu, self.mass_log_sigma, n))
        mu = rng.uniform(self.mu_lo, self.mu_hi, n)
        lever = rng.normal(0.0, self.lever_sigma, n)
        return [{"mass": float(m), "mu": float(u), "lever_noise": float(l)} for m, u, l in zip(mass, mu, lever)]

    def summary(self) -> list[str]:
        median = float(np.exp(self.mass_log_mu))
        factor = float(np.exp(self.mass_log_sigma))
        return [
            f"prior mass {median:.3f} kg (x/ {factor:.1f})",
            f"prior friction uniform {self.mu_lo:.1f} to {self.mu_hi:.1f}",
            f"prior CoM lever noise sd {self.lever_sigma * 100:.0f} cm",
            f"visibility {self.visibility:.2f}",
        ]


ARRAY_KEYS = ("mass_scale", "f", "com_frac", "com_end", "ancestor")


@dataclass
class ParticleBelief:
    """Weighted particles over (mass, friction, CoM in the object frame)."""

    particles: list[dict[str, Any]]
    weights: np.ndarray
    visibility: float = 1.0
    history: list[dict[str, Any]] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    degenerate: bool = False
    importance_ess: float | None = None      # the last refresh's importance-sampling ESS (before resampling)
    _arrays: dict[str, np.ndarray] | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=float).reshape(-1)
        if len(w) != len(self.particles) or len(w) == 0:
            raise ValueError(f"{len(w)} weights for {len(self.particles)} particles")
        total = float(w.sum())
        self.weights = w / total if total > 0 and np.isfinite(total) else np.full(len(w), 1.0 / len(w))

    def __len__(self) -> int:
        return len(self.particles)

    def ess(self) -> float:
        """Effective sample size of the weights, 1 / sum(w^2)."""
        return float(1.0 / np.sum(self.weights ** 2))

    def effective_size(self) -> float:
        """The effective number of particles: the weights' ESS; after a refresh, the refresh's importance
        ESS times the share of the resampled set the later weights keep (ESS / n)."""
        if self.importance_ess is None:
            return self.ess()
        return float(self.importance_ess) * self.ess() / len(self)

    def reweight(self, loglik: np.ndarray) -> ParticleBelief:
        """A new belief with weights times exp(loglik); non-finite entries get weight 0, and when no
        entry is finite the weights stay and a flag says so."""
        ll = np.asarray(loglik, dtype=float).reshape(-1)
        if len(ll) != len(self.particles):
            raise ValueError(f"{len(ll)} log-likelihoods for {len(self.particles)} particles")
        ok = np.isfinite(ll) & (self.weights > 0)
        if not ok.any():
            return self._replace(weights=self.weights.copy(), flags=self.flags + ["no finite likelihood: weights kept"])
        logw = np.full(len(ll), -np.inf)
        logw[ok] = np.log(self.weights[ok]) + ll[ok]
        logw -= logw[ok].max()
        return self._replace(weights=np.exp(logw))

    def resample(self, n: int, rng: np.random.Generator) -> list[dict[str, Any]]:
        """Systematic resampling: n particle copies (the ``ancestor`` key travels with them)."""
        cw = np.cumsum(self.weights)
        cw[-1] = 1.0
        idx = np.searchsorted(cw, (rng.random() + np.arange(n)) / n, side="left")
        return [dict(self.particles[int(i)]) for i in np.minimum(idx, len(self.particles) - 1)]

    def arrays(self) -> dict[str, np.ndarray]:
        """The particles as arrays: mass, mu (n,), com_obj (n, 3), and the generator keys present in all (the fragility
        keys, the fill and the spill angle among them, NaN where a particle's value is None)."""
        if self._arrays is None:
            P = self.particles
            out = {"mass": np.array([p["mass"] for p in P], dtype=float), "mu": np.array([p["mu"] for p in P], dtype=float),
                   "com_obj": np.array([p["com_obj"] for p in P], dtype=float).reshape(-1, 3)}
            for k in ARRAY_KEYS:
                if all(k in p for p in P):
                    out[k] = np.array([p[k] for p in P], dtype=float)
            if all("com_branch" in p for p in P):
                out["com_branch"] = np.array([bool(p["com_branch"]) for p in P])
            if all("com_s" in p for p in P):
                out["com_s"] = np.array([np.nan if p["com_s"] is None else p["com_s"] for p in P], dtype=float)
            for k in DAMAGE_PARTICLE_KEYS:              # milestone 2b: the fragility, the fill and the spill angle
                if all(k in p for p in P):
                    out[k] = np.array([np.nan if p[k] is None else p[k] for p in P], dtype=float)
            self._arrays = out
        return self._arrays

    def quantile(self, key: str, q: float) -> float:
        """Weighted quantile of a scalar key (mass, mu, com_frac, ...)."""
        v = self.arrays()[key]
        order = np.argsort(v, kind="stable")
        cw = np.cumsum(self.weights[order])
        i = int(np.searchsorted(cw, q - 1e-12, side="left"))
        return float(v[order][min(i, len(v) - 1)])

    def p_com_shifted(self) -> float | None:
        """Weight of the particles whose CoM the hidden point mass moved (f > 0); None without generator keys."""
        a = self.arrays()
        return None if "f" not in a else float(self.weights[a["f"] > 0].sum())

    def n_ancestors(self) -> int:
        a = self.arrays()
        if "ancestor" not in a:
            return int((self.weights > 0).sum())
        return int(len(np.unique(a["ancestor"][self.weights > 0])))

    def _interval(self, key: str) -> list[float]:
        return [round(self.quantile(key, 0.1), 4), round(self.quantile(key, 0.9), 4)]

    def to_data(self) -> dict[str, Any]:
        a = self.arrays()
        has_frac = "com_frac" in a
        return {
            "mass_median": round(self.quantile("mass", 0.5), 4), "mass_80": self._interval("mass"),
            "mu_median": round(self.quantile("mu", 0.5), 3), "mu_80": self._interval("mu"),
            "com_frac_median": round(self.quantile("com_frac", 0.5), 3) if has_frac else None,
            "com_frac_80": self._interval("com_frac") if has_frac else None,
            "p_com_shifted": None if self.p_com_shifted() is None else round(self.p_com_shifted(), 3),
            "ess": round(self.effective_size(), 1), "n_ancestors": self.n_ancestors(), "n_observations": len(self.history),
            "flags": list(self.flags), "degenerate": bool(self.degenerate),
        }

    def summary(self) -> list[str]:
        d = self.to_data()
        lines = [
            f"mass median {d['mass_median']:.3f} kg, 80 % interval {d['mass_80'][0]:.3f} to {d['mass_80'][1]:.3f} kg",
            f"friction median {d['mu_median']:.2f}, 80 % interval {d['mu_80'][0]:.2f} to {d['mu_80'][1]:.2f}",
        ]
        if d["p_com_shifted"] is not None:
            lines.append(f"P(CoM shifted toward one end) {d['p_com_shifted']:.2f}")
        lines.append(f"{d['ess']:.0f} effective particles of {len(self)}, {d['n_observations']} observations")
        lines += [f"belief flag: {f}" for f in self.flags]
        for e in self.history:
            for term in e.get("rejected_terms") or ():
                name = f"{(e.get('obs') or {}).get('probe', 'probe')} {term}" if e.get("type") == "probe" else term
                lines.append(f"reading outside the belief: {name}")
        if self.degenerate:
            lines.append("belief degenerate: fewer than 100 effective particles or ancestors after a refresh")
        return lines

    def _replace(self, **kw: Any) -> ParticleBelief:
        args = {"particles": self.particles, "weights": self.weights, "visibility": self.visibility,
                "history": list(self.history), "flags": list(self.flags), "degenerate": self.degenerate,
                "importance_ess": self.importance_ess}
        args.update(kw)
        out = ParticleBelief(**args)
        if args["particles"] is self.particles:
            out._arrays = self._arrays
        return out

    @staticmethod
    def from_parametric(belief: Belief, com_obj, n: int, rng: np.random.Generator) -> ParticleBelief:
        """The fallback for substrates without a generator: mass and friction from the milestone 1
        prior, the CoM at the nominal point."""
        com = [float(x) for x in com_obj]
        parts = [{"mass": s["mass"], "mu": s["mu"], "com_obj": list(com), "ancestor": i} for i, s in enumerate(belief.sample(n, rng))]
        return ParticleBelief(parts, np.ones(n), visibility=belief.visibility)


def refresh(belief: ParticleBelief, prior_fn: Callable[[int, np.random.Generator], ParticleBelief],
            loglik_fn: Callable[[ParticleBelief, dict[str, Any]], np.ndarray], rng: np.random.Generator,
            n_draws: tuple[int, ...] = REFRESH_DRAWS, n_keep: int = N_PARTICLES) -> ParticleBelief:
    """Rebuild a collapsed belief: draw fresh prior particles, weight them by the summed log-likelihood
    of every history entry (``loglik_fn(fresh, entry)``, which replays the entry's recorded terms), and
    resample ``n_keep``. The importance-sampling ESS is measured before resampling; the next, larger
    draw runs while it is under MIN_ESS. Each fresh particle's ``ancestor`` is its index in the draw, so
    ``n_ancestors`` counts the distinct draws that survive. The result is ``degenerate`` when the ESS
    or the ancestor count stays under MIN_ESS after the largest draw."""
    for n in n_draws:
        fresh = prior_fn(int(n), rng)
        parts = [dict(p, ancestor=i) for i, p in enumerate(fresh.particles)]
        fresh = ParticleBelief(parts, np.ones(len(parts)), visibility=belief.visibility)
        ll = np.zeros(len(fresh))
        for entry in belief.history:
            ll = ll + np.asarray(loglik_fn(fresh, entry), dtype=float)
        weighted = fresh.reweight(ll)
        ess = weighted.ess()
        if ess >= MIN_ESS:
            break
    out = ParticleBelief(weighted.resample(int(n_keep), rng), np.ones(int(n_keep)), visibility=belief.visibility,
                         history=list(belief.history), importance_ess=float(ess))
    n_anc = out.n_ancestors()
    flags = [f for f in belief.flags if not f.startswith("refreshed")] + [f for f in weighted.flags if f not in belief.flags]
    flags.append(f"refreshed from {len(fresh)} prior draws: importance ESS {ess:.0f}, {n_anc} ancestors")
    return out._replace(flags=flags, degenerate=bool(ess < MIN_ESS or n_anc < MIN_ESS))
