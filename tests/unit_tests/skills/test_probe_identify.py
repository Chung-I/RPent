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


"""The identify probe's model (arm port): the kinds and time, the analytic joint-torque likelihood on (mass, mass x
CoM) in the object frame, its sampled readings and the fit from probe-corpus rows."""

import numpy as np
import pytest

from rpent.skills.pick_place.belief import ParticleBelief
from rpent.skills.pick_place.probes import (IDENTIFY, PROBE_KINDS, PROBE_SECONDS, IdentifyModel, fit_probe_models, identify_obs,
                                            identify_summary)


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    R = q * np.sign(np.linalg.det(q))
    return rng.normal(0, 0.05, 3), R


def _reading(m, c_obj, cov, p, R, rng=None):
    """A fit's reading in link 7's frame: phi = (m, m (p + R c)), with noise of covariance ``cov`` (link 7 frame)."""
    phi = np.concatenate([[m], m * (p + R @ c_obj)])
    if rng is not None:
        phi = phi + rng.multivariate_normal(np.zeros(4), cov)
    return {"phi": phi.tolist(), "cov": cov.tolist(), "info": np.linalg.inv(cov).tolist(),
            "obj_l7": {"p": p.tolist(), "R": R.ravel().tolist()}}


COV = np.diag([1e-6, 2.5e-7, 2.5e-7, 2.5e-7])


def test_identify_kinds_and_time():
    assert PROBE_KINDS[IDENTIFY] == ("measured", "not_lifted", "lost", "unplanned")
    assert PROBE_SECONDS[IDENTIFY] == 18.0             # a 6 s excitation in an 18 s probe (R1 decision)


def test_identify_obs_maps_the_fit_into_the_object_frame():
    p, R = _frame(1)
    c = np.array([0.01, -0.02, 0.03])
    phi_obj, info_obj = identify_obs(_reading(0.4, c, COV, p, R))
    assert phi_obj == pytest.approx([0.4, *(0.4 * c)], abs=1e-12)
    T = np.eye(4)
    T[1:4, 0], T[1:4, 1:4] = p, R
    assert info_obj == pytest.approx(T.T @ np.linalg.inv(COV) @ T)
    assert identify_obs({"phi": [0.4, 0, 0, 0]}) is None


def _model(classes=("measured", "not_lifted", "lost", "unplanned"), p=(0.9, 0.05, 0.03, 0.02), covs=(COV,)):
    from rpent.skills.pick_place.probes import CalibratedClassifier

    clf = CalibratedClassifier(classes, None, None, np.array(p))
    return IdentifyModel(probe=IDENTIFY, classes=classes, clf=clf, reading={}, p_thrown=0.0, noise_covs=[np.asarray(c) for c in covs])


def _belief(n=3000, seed=2):
    rng = np.random.default_rng(seed)
    parts = [{"mass": float(m), "mu": 0.5, "com_obj": [float(x) for x in rng.uniform(-0.04, 0.04, 3)]}
             for m in rng.uniform(0.2, 0.6, n)]
    return ParticleBelief(parts, np.full(n, 1.0 / n))


def test_a_measured_identify_concentrates_mass_and_com():
    b = _belief()
    a = b.arrays()
    p, R = _frame(3)
    m, c = 0.4, np.array([0.01, -0.02, 0.03])
    pm = _model()
    X = np.zeros((len(b), 1))
    ll, info = pm.loglik({"kind": "measured", "reading": _reading(m, c, COV, p, R)}, X, a, 20.0, thrown=False)
    assert info["used"] == ["kind", "reading", "not_thrown"] or info["used"] == ["kind", "reading"]
    post = b.reweight(ll)
    pa = post.arrays()
    assert float(post.weights @ pa["mass"]) == pytest.approx(m, abs=0.01)
    assert post.weights @ pa["com_obj"] == pytest.approx(c, abs=0.01)


def test_the_reading_term_is_not_tempered_but_the_kind_term_is():
    b = _belief(500)
    a = b.arrays()
    p, R = _frame(4)
    pm = _model()
    obs = {"kind": "measured", "reading": _reading(0.4, np.zeros(3), COV, p, R)}
    X = np.zeros((len(b), 1))
    l1, _ = pm.loglik(obs, X, a, 20.0, temper=1.0, terms=["reading"])
    l2, _ = pm.loglik(obs, X, a, 20.0, temper=2.0, terms=["reading"])
    assert l1 == pytest.approx(l2)
    k1, _ = pm.loglik(obs, X, a, 20.0, temper=1.0, terms=["kind"])
    k2, _ = pm.loglik(obs, X, a, 20.0, temper=2.0, terms=["kind"])
    assert k1 == pytest.approx(2.0 * k2)


def test_a_result_without_a_fit_has_no_reading_term():
    b = _belief(200)
    pm = _model()
    ll, info = pm.loglik({"kind": "lost", "reading": {}}, np.zeros((len(b), 1)), b.arrays(), 20.0)
    assert "reading" not in info["used"]
    ll, info = pm.loglik({"kind": "measured", "reading": {}}, np.zeros((len(b), 1)), b.arrays(), 20.0)
    assert "reading" in info["rejected"]


def test_identify_samples_a_reading_the_likelihood_reads():
    b = _belief(400)
    a = b.arrays()
    pm = _model(p=(1.0, 0.0, 0.0, 0.0))
    rng = np.random.default_rng(5)
    obs = pm.sample(np.zeros(1), b.particles[7], 20.0, rng)
    assert obs["kind"] == "measured" and obs["probe"] == IDENTIFY
    phi_obj, _ = identify_obs(obs["reading"])
    part = b.particles[7]
    assert phi_obj[0] == pytest.approx(part["mass"], abs=0.006)
    ll, info = pm.loglik(obs, np.zeros((len(b), 1)), a, 20.0, terms=["reading"])
    assert int(np.argmax(ll)) == 7 or ll[7] > np.quantile(ll, 0.95)


def _rows(n_obj=12, per=10, seed=0):
    from tests.unit_tests.skills.synthetic import synthetic_rows, with_pose

    rng = np.random.default_rng(seed)
    out = []
    for i, r in enumerate(with_pose(synthetic_rows(n_obj * per, seed=seed, n_objects=n_obj))):
        m = r["truth"]["mass"]
        c = np.asarray(r["com_obj_true"], dtype=float)
        p, R = _frame(i)
        base = dict(r["features"], force=20.0, v_lift=0.03, v_transport=0.0)
        kind = "measured" if rng.random() < 0.8 else "lost"
        rd = _reading(m, c, COV, p, R, rng) if kind == "measured" else {}
        if kind == "measured":
            rd["m_hat"] = rd["phi"][0]             # the agents' keys ride along; no law may read them
        out.append({"object": r["object"], "pose": r["pose"], "base": base, "force": 20.0, "f_std": 10.0,
                    "com_obj_true": r["com_obj_true"], "truth": r["truth"], "key": r["key"] + "__Didentify",
                    "probe": IDENTIFY, "thrown": False, "kind": kind, "reading": rd})
    return out


def test_fit_probe_models_makes_an_identify_model_without_reading_laws():
    pms = fit_probe_models(_rows())
    pm = pms[IDENTIFY]
    assert isinstance(pm, IdentifyModel)
    assert pm.reading == {} and len(pm.noise_covs) > 0
    assert set(pm.stats["kinds"]) >= {"measured", "lost"}
    # the noise covariances are in the object frame: the planted link-7 covariance, mapped back
    c0 = np.asarray(pm.noise_covs[0])
    assert c0.shape == (4, 4) and np.all(np.linalg.eigvalsh(c0) > 0)


def test_identify_summary_gives_the_agents_their_keys():
    p, R = _frame(6)
    c = np.array([0.01, -0.02, 0.03])
    s = identify_summary(_reading(0.4, c, COV, p, R))
    assert s["m_hat"] == pytest.approx(0.4)
    lo, hi = s["m_range80"]
    assert lo < 0.4 < hi and hi - lo == pytest.approx(2 * 1.2815515655446004 * 1e-3, rel=1e-6)
    assert s["com_obj_m"] == pytest.approx(c.tolist(), abs=1e-12)
    assert len(s["com_obj_range80_m"]) == 3 and all(a < b for a, b in s["com_obj_range80_m"])
