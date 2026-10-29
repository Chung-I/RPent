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


"""Probe models: the weigh force, the kind classifier, the reading laws, sampling and the likelihood."""

import numpy as np
import pytest

from rpent.skills.pick_place.belief import ParticleBelief
from rpent.skills.pick_place.features import particle_matrix
from rpent.skills.pick_place.probes import PROBE_KINDS, PROBE_SECONDS, ProbeResult, fit_probe_models, weigh_force

G = 9.81


def test_weigh_force():
    assert weigh_force(5.0) == 15.0 and weigh_force(10.0) == 20.0 and weigh_force(30.0) == 60.0 and weigh_force(40.0) == 70.0
    assert PROBE_SECONDS == {"weigh": 4.0, "push": 2.0}
    assert PROBE_KINDS == {"weigh": ("measured", "lower_bound", "lost"), "push": ("slid", "tipped", "no_breakaway", "no_contact")}


def test_probe_result_data_has_no_trace():
    r = ProbeResult(probe="push", kind="slid", reading={"f_break": 1.0}, displacement_m=0.04, tilt_deg=2.0, thrown=False,
                    seconds=1.5, trace=[{"t": 0.0}])
    d = r.to_data()
    assert "trace" not in d and d["kind"] == "slid" and d["reading"] == {"f_break": 1.0}


def _probe_rows(n: int = 400, bias: float = 0.2, sd: float = 0.1, seed: int = 0) -> list[dict]:
    """Synthetic probe-corpus rows: weigh probes that measure (load ratio under 1) with log(m_hat/m)
    ~ N(bias, sd), or read the friction (lower_bound) with log(mu_hat/mu) ~ N(-bias, sd); push probes
    that slide with log(f_break/(mu m g)) ~ N(bias, sd)."""
    from tests.unit_tests.skills.synthetic import synthetic_rows, with_pose

    rng = np.random.default_rng(seed)
    out = []
    for r in with_pose(synthetic_rows(n, seed=seed, n_objects=20)):
        m, mu, F = r["truth"]["mass"], r["truth"]["mu"], 20.0
        base = dict(r["features"], force=F, v_lift=0.03, v_transport=0.0)
        common = {"object": r["object"], "pose": r["pose"], "base": base, "force": F, "f_std": 10.0,
                  "com_obj_true": r["com_obj_true"], "truth": r["truth"]}
        if m * G < 2 * mu * F:
            weigh = {"kind": "measured", "reading": {"m_hat": m * float(np.exp(bias + rng.normal(0, sd)))}}
        else:
            mu_hat = mu * float(np.exp(-bias + rng.normal(0, sd)))
            weigh = {"kind": "lower_bound", "reading": {"mu_hat": mu_hat, "m_lower": 2 * mu_hat * F / G}}
        out.append({**common, "key": r["key"] + "__Bweigh", "probe": "weigh", "thrown": False, "displacement_m": 0.0,
                    "tilt_deg": 0.0, "seconds": 4.0, **weigh})
        push = {"kind": "slid", "reading": {"f_break": mu * m * G * float(np.exp(bias + rng.normal(0, sd)))}} \
            if rng.random() < 0.7 else {"kind": "tipped", "reading": {}}
        out.append({**common, "key": r["key"] + "__Apush", "probe": "push", "thrown": bool(rng.random() < 0.02),
                    "displacement_m": 0.04, "tilt_deg": 3.0, "seconds": 1.5, **push})
    return out


def test_probe_model_recovers_a_planted_bias():
    models = fit_probe_models(_probe_rows())
    assert set(models) == {"weigh", "push"}
    assert models["weigh"].reading["measured"].b1 == pytest.approx(0.2, abs=0.03)
    assert models["weigh"].reading["lower_bound"].b1 == pytest.approx(-0.2, abs=0.03)
    assert models["push"].reading["slid"].b1 == pytest.approx(0.2, abs=0.03)
    assert models["weigh"].reading["measured"].sd1 == pytest.approx(0.1, rel=0.3)


def _particles(n: int = 2000, seed: int = 1) -> ParticleBelief:
    rng = np.random.default_rng(seed)
    m = np.exp(rng.uniform(np.log(0.05), np.log(1.5), n))
    mu = rng.uniform(0.1, 0.9, n)
    return ParticleBelief([{"mass": float(m[i]), "mu": float(mu[i]), "com_obj": [0.0, 0.0, 0.03]} for i in range(n)], np.ones(n))


def test_probe_model_sample_is_in_support():
    rows = _probe_rows()
    models = fit_probe_models(rows)
    rng = np.random.default_rng(3)
    b = _particles(50)
    X = particle_matrix(rows[0]["base"], b.arrays(), rows[0]["pose"], rows[0]["force"])
    for probe, pm in models.items():
        for i in range(50):
            s = pm.sample(X[i], b.particles[i], rows[0]["force"], rng)
            assert s["probe"] == probe and s["kind"] in PROBE_KINDS[probe] and isinstance(s["thrown"], bool)
            assert all(np.isfinite(v) and v > 0 for v in s["reading"].values())
            want = {"measured": {"m_hat"}, "lower_bound": {"mu_hat", "m_lower"}, "slid": {"f_break"}}.get(s["kind"], set())
            assert set(s["reading"]) == want


def test_probe_model_samples_a_missing_reading_at_its_rate():
    """Some lower_bound results carry no reading (the pads never slid up a still object). The model keeps
    the rate, and its simulated results lack the reading as often as the real ones."""
    rows = _probe_rows()
    lb = [r for r in rows if r["probe"] == "weigh" and r["kind"] == "lower_bound"]
    for r in lb[::2]:
        r["reading"] = {}
    pm = fit_probe_models(rows)["weigh"]
    rate = len(lb[1::2]) / len(lb)
    assert pm.reading_rate["lower_bound"] == pytest.approx(rate) and pm.reading_rate["measured"] == 1.0
    rng = np.random.default_rng(5)
    b = _particles(1)
    X = particle_matrix(rows[0]["base"], b.arrays(), rows[0]["pose"], rows[0]["force"])
    p = np.array([0.0, 1.0, 0.0])                                           # always lower_bound
    got = [pm.sample(X[0], b.particles[0], rows[0]["force"], rng, p=p)["reading"] for _ in range(2000)]
    assert all(g == {} or set(g) == {"mu_hat", "m_lower"} for g in got)
    assert np.mean([bool(g) for g in got]) == pytest.approx(rate, abs=0.04)


def test_weigh_reading_concentrates_the_mass():
    rows = _probe_rows()
    pm = fit_probe_models(rows)["weigh"]
    b = _particles()
    X = particle_matrix(rows[0]["base"], b.arrays(), rows[0]["pose"], rows[0]["force"])
    obs = {"probe": "weigh", "kind": "measured", "reading": {"m_hat": 0.3 * float(np.exp(0.2))}}
    ll, info = pm.loglik(obs, X, b.arrays(), rows[0]["force"])
    post = b.reweight(ll)
    assert abs(post.quantile("mass", 0.5) - 0.3) <= 0.15 * 0.3
    assert "reading" in info["used"] and "kind" in info["used"]


def test_far_probe_reading_is_rejected():
    rows = _probe_rows()
    pm = fit_probe_models(rows)["push"]
    b = _particles(500)
    X = particle_matrix(rows[0]["base"], b.arrays(), rows[0]["pose"], rows[0]["force"])
    ll, info = pm.loglik({"probe": "push", "kind": "slid", "reading": {"f_break": 500.0}}, X, b.arrays(), rows[0]["force"])
    assert info["rejected"] == ["reading"] and np.isfinite(ll).all()
    ref, _ = pm.loglik({"probe": "push", "kind": "slid", "reading": {}}, X, b.arrays(), rows[0]["force"])
    np.testing.assert_allclose(ll, ref)


def test_risk_and_p_kind_shapes():
    rows = _probe_rows()
    pm = fit_probe_models(rows)["push"]
    b = _particles(30)
    X = particle_matrix(rows[0]["base"], b.arrays(), rows[0]["pose"], rows[0]["force"])
    P = pm.p_kind(X)
    assert P.shape == (30, len(PROBE_KINDS["push"])) and np.allclose(P.sum(axis=1), 1.0)
    r = pm.risk(X)
    assert r.shape == (30,) and ((r >= 0) & (r <= 1)).all()


def test_probe_model_round_trips(tmp_path):
    from rpent.skills.pick_place.probes import ProbeModel

    pm = fit_probe_models(_probe_rows())["weigh"]
    pm.save(str(tmp_path / "probe_weigh.joblib"))
    back = ProbeModel.load(str(tmp_path / "probe_weigh.joblib"))
    assert back.reading == pm.reading and back.classes == pm.classes


def test_loglik_temper_argument_matches_the_model_temper():
    rows = _probe_rows()
    pm = fit_probe_models(rows)["weigh"]
    b = _particles(300)
    X = particle_matrix(rows[0]["base"], b.arrays(), rows[0]["pose"], rows[0]["force"])
    obs = {"probe": "weigh", "kind": "measured", "reading": {"m_hat": 0.3}}
    by_arg, _ = pm.loglik(obs, X, b.arrays(), rows[0]["force"], temper=2.0)
    pm.temper = 2.0
    by_attr, _ = pm.loglik(obs, X, b.arrays(), rows[0]["force"])
    np.testing.assert_allclose(by_arg, by_attr)
    pm.temper = 1.0
    plain, _ = pm.loglik(obs, X, b.arrays(), rows[0]["force"])
    assert not np.allclose(plain, by_arg)


def test_probe_loglik_replays_listed_terms():
    """A refresh replays the terms a probe entry recorded: a listed reading is used without the
    rejection rule, an unlisted one is left out."""
    rows = _probe_rows()
    pm = fit_probe_models(rows)["push"]
    b = _particles(500)
    X = particle_matrix(rows[0]["base"], b.arrays(), rows[0]["pose"], rows[0]["force"])
    far = {"probe": "push", "kind": "slid", "reading": {"f_break": 500.0}}
    ll, info = pm.loglik(far, X, b.arrays(), rows[0]["force"], terms=["kind", "reading"])
    assert info == {"used": ["kind", "reading"], "rejected": []}
    kind_only, _ = pm.loglik(far, X, b.arrays(), rows[0]["force"], terms=["kind"])
    assert not np.allclose(ll, kind_only)
    near = {"probe": "push", "kind": "slid", "reading": {"f_break": 1.0}}
    ll_k, info_k = pm.loglik(near, X, b.arrays(), rows[0]["force"], terms=["kind"])
    np.testing.assert_allclose(ll_k, kind_only)
    assert info_k == {"used": ["kind"], "rejected": []}


def test_probe_models_store_their_feature_names():
    from rpent.skills.pick_place.features import FEATURES_M2, FEATURES_M2B

    assert all(pm.feature_names == FEATURES_M2 for pm in fit_probe_models(_probe_rows(200)).values())
    models = fit_probe_models(_probe_rows(200), names=FEATURES_M2B)
    assert models and all(pm.feature_names == FEATURES_M2B for pm in models.values())
    X = np.zeros((3, len(FEATURES_M2B)))
    assert all(pm.p_kind(X).shape == (3, len(pm.classes)) for pm in models.values())


def test_preposterior_sampling_with_a_mixture_law():
    """A weigh reading law with a narrow core and a wide tail: the sampled readings follow the mixture (about 14 % lie
    beyond 3 core sd, against 0.3 % under the core alone), and probe_value gives a finite, positive EVOI when the
    best row depends on the mass."""
    from rpent.skills.pick_place.likelihood import Law
    from rpent.skills.pick_place.menu import probe_value

    rows = _probe_rows()
    pm = fit_probe_models(rows)["weigh"]
    pm.reading["measured"] = Law("mix2", 0.8, 0.0, 0.05, 0.0, 0.4)
    b = _particles(400)
    X = particle_matrix(rows[0]["base"], b.arrays(), rows[0]["pose"], rows[0]["force"])
    rng = np.random.default_rng(2)
    p = np.array([1.0, 0.0, 0.0])                                           # always measured
    m = float(b.particles[0]["mass"])
    x = np.array([np.log(pm.sample(X[0], b.particles[0], rows[0]["force"], rng, p=p)["reading"]["m_hat"] / m) for _ in range(4000)])
    assert np.mean(np.abs(x) > 0.15) == pytest.approx(0.2 * 0.7077 + 0.8 * 0.0027, abs=0.025)
    mass = b.arrays()["mass"]
    C = [np.where(mass < 0.3, 0.2, 5.2), np.full(len(mass), 0.9)]
    v = probe_value(b, C, pm, X, rows[0]["force"], "normal", np.random.default_rng(0), k=300)
    assert np.isfinite(v["evoi"]) and np.isfinite(v["evoi_se"]) and v["evoi"] > 0.0


def test_an_old_tuple_law_reads_as_a_normal_law():
    """A probe model pickled before the mixture laws holds (b, sd) tuples: loglik and sample read them as normal laws,
    with the same random stream."""
    from rpent.skills.pick_place.likelihood import Law

    rows = _probe_rows()
    pm = fit_probe_models(rows)["weigh"]
    b = _particles(300)
    X = particle_matrix(rows[0]["base"], b.arrays(), rows[0]["pose"], rows[0]["force"])
    obs = {"probe": "weigh", "kind": "measured", "reading": {"m_hat": 0.3}}
    pm.reading["measured"] = Law.normal(0.2, 0.1)
    new, _ = pm.loglik(obs, X, b.arrays(), rows[0]["force"])
    s_new = pm.sample(X[0], b.particles[0], rows[0]["force"], np.random.default_rng(9), p=np.array([1.0, 0.0, 0.0]))
    pm.reading["measured"] = (0.2, 0.1)
    old, _ = pm.loglik(obs, X, b.arrays(), rows[0]["force"])
    s_old = pm.sample(X[0], b.particles[0], rows[0]["force"], np.random.default_rng(9), p=np.array([1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(new, old)
    assert s_new == s_old


def test_probe_reading_laws_select_a_mixture_on_validation_objects():
    """Weigh readings with a 25 % wide tail: with validation objects the measured law is a mixture, without them it is
    normal; the push readings, without a tail, stay normal."""
    rng = np.random.default_rng(11)
    rows = _probe_rows(1200, sd=0.05)
    for r in rows:
        if r["kind"] == "measured" and rng.random() < 0.25:
            r["reading"] = {"m_hat": r["reading"]["m_hat"] * float(np.exp(rng.normal(0.0, 0.6)))}
    val = sorted({r["object"] for r in rows})[::5]
    models = fit_probe_models(rows, val_objects=val)
    assert models["weigh"].reading["measured"].kind == "mix2" and models["push"].reading["slid"].kind == "normal"
    assert fit_probe_models(rows)["weigh"].reading["measured"].kind == "normal"
