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


"""The readout likelihood: each term moves the belief the way its physics says, far readings are
rejected, and the residual laws are fitted per label."""

import numpy as np
import pytest

from rpent.skills.pick_place.belief import ParticleBelief
from rpent.skills.pick_place.likelihood import ReadoutModel
from rpent.skills.pick_place.protocol import OUTCOMES, Prediction

G = 9.81
N = 2000


def _flat(n: int) -> np.ndarray:
    return np.full((n, len(OUTCOMES)), 1.0 / len(OUTCOMES))


def _belief(n: int = N, seed: int = 0) -> ParticleBelief:
    rng = np.random.default_rng(seed)
    m = np.exp(rng.uniform(np.log(0.05), np.log(2.0), n))
    u = rng.uniform(0.1, 0.9, n)
    return ParticleBelief([{"mass": float(m[i]), "mu": float(u[i]), "com_obj": [0.0, 0.0, 0.0]} for i in range(n)], np.ones(n))


def _model(**kw) -> ReadoutModel:
    terms = {t: {"_pooled": (0.0, 0.1)} for t in ("hold", "slip", "loss")}
    return ReadoutModel(terms=kw.pop("terms", terms), side_error=kw.pop("side_error", 0.1), **kw)


def _obs(label: str = "placed_stable", force: float = 10.0, **readouts) -> dict:
    r = {"fz_hold_n": None, "fz_slip_max_n": None, "fz_at_loss_n": None, "rot_deg": 0.0, "heavy_side": 0}
    r.update(readouts)
    return {"label": label, "readouts": r, "force": force}


def _width(b: ParticleBelief, key: str) -> float:
    return b.quantile(key, 0.9) - b.quantile(key, 0.1)


def test_hold_reading_concentrates_the_mass():
    b = _belief()
    ll, info = _model().loglik(_obs(fz_hold_n=0.5 * G), _flat(N), None, b.arrays(), None)
    post = b.reweight(ll)
    assert abs(post.quantile("mass", 0.5) - 0.5) <= 0.15 * 0.5
    assert _width(post, "mass") < 0.4 * _width(b, "mass")
    assert "hold" in info["used"] and info["rejected"] == []


def test_slip_reading_concentrates_mu():
    b, F = _belief(), 10.0
    ll, info = _model().loglik(_obs("not_lifted", F, fz_slip_max_n=2 * 0.4 * F), _flat(N), None, b.arrays(), None)
    post = b.reweight(ll)
    assert abs(post.quantile("mu", 0.5) - 0.4) <= 0.15 * 0.4
    assert _width(post, "mu") < 0.4 * _width(b, "mu")
    assert "slip" in info["used"]


def test_loss_reading_concentrates_mu():
    b, F = _belief(), 20.0
    ll, info = _model().loglik(_obs("dropped_high", F, fz_at_loss_n=2 * 0.6 * F), _flat(N), None, b.arrays(), None)
    post = b.reweight(ll)
    assert abs(post.quantile("mu", 0.5) - 0.6) <= 0.15 * 0.6
    assert _width(post, "mu") < 0.4 * _width(b, "mu")
    assert "loss" in info["used"]


def test_heavy_side_moves_weight_to_that_side():
    b = _belief()
    sides = np.array([1] * (N // 2) + [-1] * (N // 2))
    ll, info = _model(side_error=0.1).loglik(_obs(rot_deg=20.0, heavy_side=1), _flat(N), None, b.arrays(), sides)
    post = b.reweight(ll)
    assert post.weights[sides == 1].sum() == pytest.approx(0.9, abs=1e-9)
    assert "side" in info["used"]


def test_side_term_is_one_half_on_the_pinch():
    b = _belief(n=3)
    ll, _ = _model(side_error=0.1).loglik(_obs(rot_deg=20.0, heavy_side=-1), _flat(3), None, b.arrays(), np.array([-1, 0, 1]))
    np.testing.assert_allclose(np.exp(ll - ll.max()) * 0.9, [0.9, 0.5, 0.1])


def test_outcome_term_uses_the_prediction():
    b = _belief(n=2)
    P = np.zeros((2, len(OUTCOMES)))
    j = OUTCOMES.index("placed_stable")
    P[0, j], P[1, j] = 0.9, 0.1
    for preds in (P, [Prediction(p=dict(zip(OUTCOMES, row))) for row in P]):
        ll, info = _model().loglik(_obs(), preds, None, b.arrays(), None)
        post = b.reweight(ll)
        assert post.weights[0] / post.weights[1] == pytest.approx(9.0)
        assert "outcome" in info["used"]


def test_far_reading_is_rejected_and_listed():
    b = _belief()
    sides = np.array([1] * (N // 2) + [-1] * (N // 2))
    m = _model()
    ll, info = m.loglik(_obs(fz_hold_n=100.0 * G, rot_deg=20.0, heavy_side=1), _flat(N), None, b.arrays(), sides)
    assert "hold" in info["rejected"] and "hold" not in info["used"]
    assert np.isfinite(ll).all()
    ref, _ = m.loglik(_obs(rot_deg=20.0, heavy_side=1), _flat(N), None, b.arrays(), sides)
    np.testing.assert_allclose(ll, ref)


def test_a_non_positive_reading_is_rejected():
    b = _belief(n=10)
    ll, info = _model().loglik(_obs("not_lifted", fz_slip_max_n=-0.3), _flat(10), None, b.arrays(), None)
    assert info["rejected"] == ["slip"] and np.isfinite(ll).all()


def _readout_rows() -> list[dict]:
    """Synthetic corpus v2 rows with readouts: hold log ratios biased +0.2 (placed_stable), -0.1
    (dropped_low) and +0.5 (tipped, 10 rows: pooled), sd 0.05."""
    from tests.unit_tests.skills.synthetic import synthetic_rows, with_pose

    rng = np.random.default_rng(1)
    rows = with_pose(synthetic_rows(170, n_objects=10))
    out = []
    for i, r in enumerate(rows):
        label, bias = ("placed_stable", 0.2) if i < 80 else (("dropped_low", -0.1) if i < 160 else ("tipped", 0.5))
        fz = r["truth"]["mass"] * G * float(np.exp(bias + rng.normal(0.0, 0.05)))
        rot = float(rng.uniform(0.0, 30.0))
        out.append(dict(r, label=label, readouts={"fz_hold_n": fz, "fz_slip_max_n": None, "fz_at_loss_n": None,
                                                  "rot_deg": rot, "heavy_side": 0}))
    return out


def test_residuals_are_fitted_per_label():
    m = ReadoutModel.fit(_readout_rows())
    assert m.terms["hold"]["placed_stable"][0] == pytest.approx(0.2, abs=0.03)
    assert m.terms["hold"]["dropped_low"][0] == pytest.approx(-0.1, abs=0.03)
    assert m.terms["hold"]["placed_stable"][1] == pytest.approx(0.05, rel=0.35)
    assert "tipped" not in m.terms["hold"] and m.law("hold", "tipped") == m.terms["hold"]["_pooled"]
    assert "slip" not in m.terms                                   # no slip readings in these rows
    assert m.rot_clf is not None


def test_side_error_is_the_mismatch_rate():
    from tests.unit_tests.skills.synthetic import synthetic_rows, with_pose

    rows = []
    for i, r in enumerate(with_pose(synthetic_rows(100, n_objects=10))):
        true_side = 1 if i % 2 else -1
        observed = -true_side if i % 10 == 0 else true_side             # 10 % mismatches
        rows.append(dict(r, com_obj_true=[0.0, 0.01 * true_side, 0.03],
                         readouts={"fz_hold_n": None, "fz_slip_max_n": None, "fz_at_loss_n": None, "rot_deg": 20.0,
                                   "heavy_side": observed}))
    assert ReadoutModel.fit(rows).side_error == pytest.approx(0.1, abs=1e-9)


def test_readout_model_round_trips(tmp_path):
    m = ReadoutModel.fit(_readout_rows())
    m.save(str(tmp_path / "readout_model.json"), str(tmp_path / "readout_rot.joblib"))
    back = ReadoutModel.load(str(tmp_path / "readout_model.json"))
    assert back.terms == m.terms and back.side_error == m.side_error and back.temper == m.temper
    b = _belief(n=50)
    from rpent.skills.pick_place.features import FEATURES

    X = np.zeros((50, len(FEATURES)))
    obs = _obs(fz_hold_n=0.4 * G, rot_deg=15.0)
    np.testing.assert_allclose(m.loglik(obs, _flat(50), X, b.arrays(), None)[0], back.loglik(obs, _flat(50), X, b.arrays(), None)[0])
