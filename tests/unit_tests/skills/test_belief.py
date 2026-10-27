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

"""The belief prior over the hidden properties (mass, friction, CoM lever)."""

import numpy as np

from rpent.skills.pick_place.belief import RHO, Belief


def _median_mass(b, n=4000):
    return float(np.median([x["mass"] for x in b.sample(n, np.random.default_rng(0))]))


def test_prior_mass_from_material_volume_and_hollow():
    assert abs(_median_mass(Belief.prior(1e-4, "wood")) - 0.07) < 0.004
    assert abs(_median_mass(Belief.prior(1e-4, "metal_can", hollow=True)) - 0.0225) < 0.002
    assert RHO["steel"] == 7800 and RHO["foam"] == 100


def test_mu_range_matches_corpus_and_lever_noise_is_zero_mean():
    s = Belief.prior(1e-4, "wood").sample(2000, np.random.default_rng(1))
    mus = [x["mu"] for x in s]
    assert 0.1 <= min(mus) and max(mus) <= 0.9 and 0.4 < np.mean(mus) < 0.6
    assert abs(np.mean([x["lever_noise"] for x in s])) < 0.003 and 0.015 < np.std([x["lever_noise"] for x in s]) < 0.025   # default sd 2 cm without a length


def test_occlusion_widens_mass():
    assert Belief.prior(1e-4, "wood", visibility=0.4).mass_log_sigma > Belief.prior(1e-4, "wood").mass_log_sigma
    assert Belief.prior(1e-4, "wood").mass_log_sigma == 0.5


def test_unknown_material_falls_back():
    assert Belief.prior(1e-4, "playdough").mass_log_mu == Belief.prior(1e-4, "unknown").mass_log_mu


def test_summary_lines():
    lines = Belief.prior(1e-4, "wood").summary()
    assert any("mass" in line for line in lines) and any("friction" in line for line in lines)


def test_lever_prior_scales_with_object_length():
    short = Belief.prior(1e-4, "wood", length=0.08)
    long_ = Belief.prior(1e-4, "wood", length=0.30)
    assert long_.lever_sigma > short.lever_sigma
    assert abs(long_.lever_sigma - 0.15 * 0.30) < 1e-9
    assert Belief.prior(1e-4, "wood").lever_sigma == 0.02


def test_densities_mirror_the_sim(monkeypatch):
    import os
    import sys

    sim = os.environ.get("RPENT_MUJOCO_SIM_DIR")
    if not sim:
        import pytest

        pytest.skip("sim dir not set")
    sys.path.insert(0, sim)
    from objects import MATERIAL

    for k, v in MATERIAL.items():
        assert RHO[k] == v, k
