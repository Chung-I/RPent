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


def test_squeeze_sensitivity_reports_the_reading_value():
    """Plan 2b2 decision 3: per cell the mean reading value and total value of the cell's estimator, each with its
    between-case error (the sd of the per-case values over sqrt(cases)) and its conditional Monte Carlo error, the
    plug-in's means on the same draws, and at rho 0 the rule: the mean reading value is at most 2 between-case errors
    above 0."""
    from robots.mujoco_lab.squeeze_sensitivity import cell_summary

    e, r = np.array([0.1, 0.3, 0.2, 0.0]), np.array([-0.01, 0.01, 0.03, -0.03])
    vals = [{"estimator": "heldout", "evoi": a, "evoi_se": 0.01, "reading_gain": b, "reading_gain_se": 0.02,
             "plugin_evoi": a + 0.01, "plugin_reading_gain": b + 0.005} for a, b in zip(e, r)]
    c = cell_summary(vals, "0")
    assert c["estimator"] == "heldout" and c["n_cases"] == 4
    assert c["mean_reading_value"] == pytest.approx(r.mean()) and c["reading_between_case_se"] == pytest.approx(r.std(ddof=1) / 2)
    assert c["mean_evoi"] == pytest.approx(e.mean()) and c["evoi_between_case_se"] == pytest.approx(e.std(ddof=1) / 2)
    assert c["reading_mc_se"] == pytest.approx(0.01) and c["evoi_mc_se"] == pytest.approx(0.005)
    assert c["plugin_mean_evoi"] == pytest.approx(e.mean() + 0.01)
    assert c["plugin_mean_reading_value"] == pytest.approx(r.mean() + 0.005)
    assert c["reading_value_at_most_2_se_above_0"] is True
    assert cell_summary([dict(v, reading_gain=v["reading_gain"] + 0.1) for v in vals], "0")["reading_value_at_most_2_se_above_0"] is False
    assert cell_summary(vals, "final")["reading_value_at_most_2_se_above_0"] is None
    assert cell_summary(vals, "0.5")["reading_value_at_most_2_se_above_0"] is None
    plug = [dict({k: v[k] for k in ("evoi", "evoi_se", "reading_gain", "reading_gain_se")}, estimator="plugin") for v in vals]
    p = cell_summary(plug, "0")
    assert p["plugin_mean_evoi"] == pytest.approx(e.mean()) and p["plugin_mean_reading_value"] == pytest.approx(r.mean())
