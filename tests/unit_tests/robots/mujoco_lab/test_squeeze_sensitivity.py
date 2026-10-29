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


"""Squeeze sensitivity replays the stored physical and sensor latents without the sim."""

import copy
import importlib.util

import numpy as np
import pytest


def test_sensitivity_recomputes_readings_from_kept_latents():
    from robots.mujoco_lab.fragility import THRESHOLDS_V0, Thresholds, map_latent

    spec = importlib.util.find_spec("robots.mujoco_lab.squeeze_sensitivity")
    assert spec is not None, "missing squeeze sensitivity replay module"
    from robots.mujoco_lab.squeeze_sensitivity import recompute_readings

    th = Thresholds.from_json(THRESHOLDS_V0)
    entry = {"class": "crushable", "law": "cardboard", "fillable": False}
    rows = []
    for z, eps in (([0.3, -0.7, 0.1], 1.2), ([-0.4, 0.9, -0.3], -0.8)):
        frag = map_latent(z, entry, th)
        rows.append({"probe": "squeeze", "kind": "measured", "entry": entry, "theta_frag": frag,
                     "reading": {"k_hat": frag["k_n_per_mm"] * np.exp(0.10 * eps), "eps": eps}})
    before = copy.deepcopy(rows)
    own = recompute_readings(rows, th, 0.10)
    assert [r["reading"]["k_hat"] for r in own] == pytest.approx([r["reading"]["k_hat"] for r in rows], rel=0.0, abs=1e-9)
    changed = copy.deepcopy(th)
    changed.laws["cardboard"]["rho"] = 0.5
    for sigma, law in ((0.4, th), (0.10, changed)):
        replay = recompute_readings(rows, law, sigma)
        for original, got in zip(rows, replay):
            z1, z2, _ = original["theta_frag"]["z"]
            params = law.laws["cardboard"]
            rho = params["rho"]
            expected = params["k_n_per_mm"]["median"] * np.exp(params["k_n_per_mm"]["log_sd"] *
                        (rho * z1 + np.sqrt(1.0 - rho ** 2) * z2) + sigma * original["reading"]["eps"])
            assert got["reading"]["k_hat"] == pytest.approx(expected, rel=0.0, abs=1e-9)
            assert abs(got["reading"]["k_hat"] - original["reading"]["k_hat"]) > 1e-6
            assert got["theta_frag"]["f_damage_n"] == original["theta_frag"]["f_damage_n"]
            assert got["kind"] == original["kind"]
    assert rows == before
