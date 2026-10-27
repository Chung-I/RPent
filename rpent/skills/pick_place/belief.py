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

"""Belief over the hidden properties: a prior from what can be seen (volume, material).

Milestone 1 holds the prior only. Milestone 2 adds the updates from probes and
failed attempts. The prior's ranges match the corpus draws, so a model calibrated
under this belief is calibrated for the population it is deployed on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# kg/m^3 by the sim's material names (objects.py); hollow objects are shells at a quarter of the density
RHO: dict[str, float] = {
    "cardboard": 300.0,
    "ceramic": 2400.0,
    "electronics": 1500.0,
    "foam": 100.0,
    "fruit": 950.0,
    "glass": 2500.0,
    "hard_plastic": 1000.0,
    "metal_can": 7800.0,
    "steel": 7800.0,
    "thin_plastic": 950.0,
    "wood": 700.0,
    "unknown": 1000.0,
}
HOLLOW_FACTOR = 0.25
MASS_LOG_SIGMA = 0.5
MU_RANGE = (0.1, 0.9)
LEVER_SIGMA = 0.02


@dataclass
class Belief:
    mass_log_mu: float
    mass_log_sigma: float
    mu_lo: float
    mu_hi: float
    lever_sigma: float
    visibility: float

    @staticmethod
    def prior(volume_m3: float, material: str, hollow: bool = False, visibility: float = 1.0) -> Belief:
        rho = RHO.get(material, RHO["unknown"])
        median = max(1e-4, float(volume_m3) * rho * (HOLLOW_FACTOR if hollow else 1.0))
        sigma = MASS_LOG_SIGMA + 0.5 * (1.0 - float(np.clip(visibility, 0.0, 1.0)))
        return Belief(mass_log_mu=float(np.log(median)), mass_log_sigma=float(sigma), mu_lo=MU_RANGE[0],
                      mu_hi=MU_RANGE[1], lever_sigma=LEVER_SIGMA, visibility=float(visibility))

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
