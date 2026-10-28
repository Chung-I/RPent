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


# ---------------------------------------------------------------- particle belief (milestone 2)
import pytest  # noqa: E402

from rpent.skills.pick_place.belief import ParticleBelief  # noqa: E402


def _pb(weights, mus=None):
    n = len(weights)
    mus = np.linspace(0.1, 0.9, n) if mus is None else mus
    parts = [{"mass": 0.2 + 0.01 * i, "mu": float(mus[i]), "com_obj": [0.0, 0.0, 0.01 * i], "ancestor": i} for i in range(n)]
    return ParticleBelief(parts, np.asarray(weights, dtype=float))


def test_reweight_normalises_and_flags_non_finite():
    b = _pb(np.ones(4))
    nb = b.reweight(np.log([1.0, 2.0, 3.0, 4.0]))
    assert nb is not b and np.allclose(nb.weights, [0.1, 0.2, 0.3, 0.4]) and np.allclose(b.weights, 0.25)
    assert np.allclose(b.reweight(np.array([0.0, -np.inf, np.nan, 0.0])).weights, [0.5, 0.0, 0.0, 0.5])
    none = b.reweight(np.full(4, -np.inf))
    assert np.allclose(none.weights, 0.25) and any("likelihood" in f for f in none.flags)
    big = b.reweight(np.array([-1e6, -1e6 + 1.0, -1e6, -1e6]))           # log-space: no underflow
    assert np.isfinite(big.weights).all() and big.weights[1] == pytest.approx(np.e / (3 + np.e))


def test_resample_follows_weights():
    b = _pb([0.9, 0.1])
    draws = b.resample(10000, np.random.default_rng(0))
    assert len(draws) == 10000 and abs(np.mean([d["ancestor"] == 0 for d in draws]) - 0.9) < 0.02
    assert {d["ancestor"] for d in draws} == {0, 1}


def test_ess_of_uniform_and_point_weights():
    assert _pb(np.ones(100)).ess() == pytest.approx(100)
    assert _pb([1.0] + [0.0] * 99).ess() == pytest.approx(1)


def test_weighted_quantiles_and_data():
    b = _pb([0.0, 0.0, 1.0, 0.0])
    assert b.quantile("mu", 0.5) == pytest.approx(b.particles[2]["mu"]) and b.quantile("mass", 0.1) == pytest.approx(0.22)
    d = _pb(np.ones(10)).to_data()
    assert {"mass_median", "mass_80", "mu_median", "mu_80", "p_com_shifted", "ess", "n_ancestors", "n_observations",
            "flags", "degenerate"} <= set(d)
    assert d["p_com_shifted"] is None and d["n_ancestors"] == 10 and d["ess"] == pytest.approx(10)
    assert any("mass" in line for line in _pb(np.ones(10)).summary())


def test_from_parametric_keeps_the_nominal_com():
    pb = ParticleBelief.from_parametric(Belief.prior(1e-4, "wood"), [0.01, 0.0, 0.02], 500, np.random.default_rng(0))
    a = pb.arrays()
    assert a["mass"].shape == (500,) and a["com_obj"].shape == (500, 3)
    assert np.allclose(a["com_obj"], [0.01, 0.0, 0.02]) and 0.1 <= a["mu"].min() and a["mu"].max() <= 0.9


# ---------------------------------------------------------------- refresh (milestone 2, task 6)
from rpent.skills.pick_place.belief import refresh  # noqa: E402


def _prior(n, rng):
    """A synthetic generator: mass log-uniform 0.05 to 2 kg, friction uniform 0.1 to 0.9."""
    m = np.exp(rng.uniform(np.log(0.05), np.log(2.0), n))
    u = rng.uniform(0.1, 0.9, n)
    return ParticleBelief([{"mass": float(m[i]), "mu": float(u[i]), "com_obj": [0.0, 0.0, 0.0], "ancestor": -1} for i in range(n)],
                          np.ones(n))


def _sharp(fresh, entry):
    """A hold-like reading of the mass: log N(log(m_read / m); 0, sd)."""
    return -0.5 * (np.log(entry["obs"]["m"] / fresh.arrays()["mass"]) / entry["obs"]["sd"]) ** 2


def _history(k, m=0.3, sd=0.05):
    return [{"type": "attempt", "obs": {"m": m, "sd": sd}, "used_terms": ["hold"], "rejected_terms": []} for _ in range(k)]


def _counting(calls, seed):
    def prior_fn(n, rng):
        calls.append(n)
        return _prior(n, np.random.default_rng(seed))
    return prior_fn


def test_refresh_reports_importance_ess_and_ancestors():
    """Six sharp readings: the reported ESS is the importance ESS of the fresh draws before resampling
    (not the 2,000 of the resampled weights); ESS and ancestors reach 100, else the belief is degenerate."""
    b = _prior(2000, np.random.default_rng(0))._replace(history=_history(6))
    calls = []
    out = refresh(b, _counting(calls, 1), _sharp, np.random.default_rng(2))
    fresh = _prior(20000, np.random.default_rng(1))                  # the same draws, weighted here
    ll = sum(_sharp(fresh, e) for e in b.history)
    w = np.exp(ll - ll.max())
    w /= w.sum()
    ess = float(1.0 / np.sum(w ** 2))
    assert calls == [20000] and ess >= 100                          # the first draw was enough
    assert len(out) == 2000 and out.importance_ess == pytest.approx(ess) and out.ess() == pytest.approx(2000)
    d = out.to_data()
    assert d["ess"] == pytest.approx(round(ess, 1)) and d["n_ancestors"] == out.n_ancestors()
    assert (d["ess"] >= 100 and d["n_ancestors"] >= 100) or d["degenerate"]
    assert not d["degenerate"] and d["n_ancestors"] >= 100
    assert {p["ancestor"] for p in out.particles} <= set(range(20000))   # ancestors index the fresh draws
    assert out.history == b.history and abs(out.quantile("mass", 0.5) - 0.3) < 0.02
    assert [f for f in out.flags if f.startswith("refreshed")] == [f"refreshed from 20000 prior draws: importance ESS {ess:.0f}, "
                                                                  f"{out.n_ancestors()} ancestors"]
    again = refresh(out, _counting([], 3), _sharp, np.random.default_rng(4))
    assert len([f for f in again.flags if f.startswith("refreshed")]) == 1     # the newest refresh replaces the old flag


def test_refresh_draws_more_when_the_first_ess_is_low():
    """Readings too sharp for the first draw: the larger draw runs; still under 100, the belief is degenerate."""
    b = _prior(500, np.random.default_rng(0))._replace(history=_history(6, sd=0.004))
    calls = []
    out = refresh(b, _counting(calls, 1), _sharp, np.random.default_rng(2), n_draws=(2000, 8000), n_keep=500)
    assert calls == [2000, 8000] and len(out) == 500
    assert out.importance_ess < 100 and out.degenerate and out.to_data()["degenerate"] is True
    assert any(s.startswith("belief degenerate") for s in out.summary())


def test_later_reweights_scale_the_importance_ess():
    """After a refresh the reported ESS is the importance ESS times the share the later weights keep."""
    b = _prior(2000, np.random.default_rng(0))._replace(history=_history(6))
    out = refresh(b, _counting([], 1), _sharp, np.random.default_rng(2))
    post = out.reweight(_sharp(out, {"obs": {"m": 0.3, "sd": 0.05}}))
    assert post.importance_ess == out.importance_ess
    assert post.effective_size() == pytest.approx(out.importance_ess * post.ess() / len(post))
    assert post.to_data()["ess"] == pytest.approx(round(post.effective_size(), 1))
    assert _pb(np.ones(10)).effective_size() == pytest.approx(10)          # no refresh: the weights' ESS


def test_summary_names_readings_outside_the_belief():
    b = _pb(np.ones(10))._replace(history=[
        {"type": "attempt", "used_terms": ["outcome"], "rejected_terms": ["hold"]},
        {"type": "probe", "obs": {"probe": "weigh"}, "used_terms": ["kind"], "rejected_terms": ["reading"]}])
    lines = b.summary()
    assert "reading outside the belief: hold" in lines and "reading outside the belief: weigh reading" in lines
