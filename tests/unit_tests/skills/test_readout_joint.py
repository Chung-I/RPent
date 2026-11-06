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


"""The attempt's joint-torque term (arm port, m2b2_v12b): the readout model reads the attempt's Real2Sim fit
(``readouts["joint"]``) with R1's Gaussian likelihood and the model-error floor, through the same code as the identify
probe; the term replaces the hold reading's mass evidence (the same joint torques); slip, loss, rot and side stay."""

import json
import os
import sys
import zlib

import numpy as np
import pytest

from rpent.skills.pick_place.belief import ParticleBelief
from rpent.skills.pick_place.likelihood import Law, ReadoutModel, joint_loglik, joint_obs
from rpent.skills.pick_place.probes import identify_loglik, identify_obs

COV = np.diag([1e-6, 2.5e-7, 2.5e-7, 2.5e-7])


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return rng.normal(0, 0.05, 3), q * np.sign(np.linalg.det(q))


def _fit(m, c_obj, p, R, cov=COV):
    phi = np.concatenate([[m], m * (p + R @ c_obj)])
    return {"params": 4, "phi": phi.tolist(), "cov": cov.tolist(), "info": np.linalg.inv(cov).tolist(),
            "obj_l7": {"p": p.tolist(), "R": R.ravel().tolist()}}


def _belief(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    parts = [{"mass": float(m), "mu": float(rng.uniform(0.3, 0.9)), "com_obj": [float(x) for x in rng.uniform(-0.04, 0.04, 3)]}
             for m in rng.uniform(0.2, 0.6, n)]
    return ParticleBelief(parts, np.full(n, 1.0 / n))


def _model(joint=True):
    law = Law.normal(0.0, 0.1)
    return ReadoutModel(terms={"hold": {"_pooled": law}, "slip": {"_pooled": law}, "loss": {"_pooled": law}},
                        side_error=0.1, joint=joint)


def _obs(m=0.4, mu=0.6, force=20.0, joint=True, seed=2):
    p, R = _frame(seed)
    r = {"fz_hold_n": m * 9.81, "fz_slip_max_n": 2 * mu * force, "fz_at_loss_n": 2 * mu * force, "rot_deg": 0.0, "heavy_side": 0}
    if joint:
        r["joint"] = _fit(m, np.array([0.01, -0.02, 0.0]), p, R)
    return {"label": "placed_stable", "readouts": r, "force": force}


def test_the_identify_probe_and_the_attempt_share_one_likelihood():
    assert identify_obs is joint_obs and identify_loglik is joint_loglik


def test_the_joint_term_replaces_the_hold_reading():
    b = _belief()
    _, info = _model().components(_obs(), None, None, b.arrays(), None)
    assert "joint" in info["used"] and "hold" not in info["used"]
    assert {"slip", "loss"} <= set(info["used"])


def test_a_model_without_the_joint_term_keeps_the_hold_reading():
    b = _belief()
    _, info = _model(joint=False).components(_obs(), None, None, b.arrays(), None)
    assert "joint" not in info["used"] and "hold" in info["used"]


def test_an_attempt_without_a_fit_keeps_the_hold_reading():
    b = _belief()
    _, info = _model().components(_obs(joint=False), None, None, b.arrays(), None)
    assert "joint" not in info["used"] and "hold" in info["used"]
    obs = _obs()
    obs["readouts"]["joint"] = {"phi": [0.4, 0, 0, 0]}                 # no information, no frame
    _, info = _model().components(obs, None, None, b.arrays(), None)
    assert "joint" in info["rejected"] and "hold" in info["used"]


def test_the_joint_term_is_not_tempered():
    b = _belief()
    rm = _model()
    comp, _ = rm.components(_obs(), None, None, b.arrays(), None, terms=["joint"])
    assert rm.combine(comp, 1.0) == pytest.approx(rm.combine(comp, 3.0))
    a = b.arrays()
    assert rm.combine(comp, 1.0) == pytest.approx(joint_loglik(joint_obs(_obs()["readouts"]["joint"]), a))


def test_a_replay_of_the_recorded_terms_gives_the_same_likelihood():
    b = _belief()
    rm = _model()
    comp, info = rm.components(_obs(), None, None, b.arrays(), None)
    comp2, info2 = rm.components(_obs(), None, None, b.arrays(), None, terms=info["used"])
    assert info2["used"] == info["used"]
    assert rm.combine(comp) == pytest.approx(rm.combine(comp2))


def test_the_joint_flag_round_trips_and_is_set_by_a_joint_corpus():
    rm = _model()
    assert ReadoutModel.from_json(rm.to_json()).joint is True
    assert ReadoutModel.from_json({k: v for k, v in rm.to_json().items() if k != "joint"}).joint is False   # an old bundle
    from tests.unit_tests.skills.synthetic import synthetic_rows, with_pose

    rows = [dict(r, readouts={"fz_hold_n": r["truth"]["mass"] * 9.81, "rot_deg": 0.0, "heavy_side": 0})
            for r in with_pose(synthetic_rows(60, seed=0, n_objects=6))]
    assert ReadoutModel.fit(rows).joint is False
    p, R = _frame(3)
    for r in rows:
        r["readouts"]["joint"] = _fit(r["truth"]["mass"], np.asarray(r["com_obj_true"], dtype=float), p, R)
    fitted = ReadoutModel.fit(rows)
    assert fitted.joint is True and fitted.stats["joint"]["n"] == len(rows)


# R1's attempt rows: the bundle's attempt update on R1's attempt fits (with the model-error floor) gives R1's posterior.
SIM = os.environ.get("RPENT_MUJOCO_SIM_DIR", "")
R1 = os.path.join(os.path.dirname(SIM), "out", "r1_accuracy.json") if SIM else ""
DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..", "robots", "mujoco_lab", "data")
R1_KEYS = ("fruits_veggies__avocado__v11d1", "hope__canned_mushrooms__v11d1", "hot3d__mug__v11d1")


@pytest.mark.skipif(not (R1 and os.path.exists(R1) and os.path.exists(os.path.join(DATA, "corpus_v11c.jsonl"))),
                    reason="needs the arm sim's R1 output and corpus v11c")
def test_the_attempt_posterior_matches_r1s_attempt_rows():
    from robots.mujoco_lab.prior import prior_from_manifest

    sys.path.insert(0, SIM)
    import ident as I

    cases = {c["key"]: c for c in json.load(open(R1))["cases"] if c["key"] in R1_KEYS}
    rows = {}
    with open(os.path.join(DATA, "corpus_v11c.jsonl")) as f:
        for line in f:
            if any(k in line for k in R1_KEYS):
                r = json.loads(line)
                if r.get("key") in R1_KEYS:
                    rows[r["key"]] = r
    assert set(cases) == set(R1_KEYS) == set(rows)
    prior = prior_from_manifest(json.load(open(os.path.join(DATA, "m2b2_v11c", "manifest.json"))))
    rm = ReadoutModel(terms={}, side_error=0.5, joint=True)
    for key in R1_KEYS:
        est, want = cases[key]["attempt"]["est"], cases[key]["attempt_floor"]["belief"]
        cov = I.with_floor(np.asarray(est["cov"], dtype=float))          # as the adapter now records it
        fit = {**est, "cov": cov.tolist(), "info": np.linalg.inv(cov[:4, :4]).tolist()}
        b = prior(rows[key]["pose"], 20000, np.random.default_rng(zlib.crc32(key.encode())))
        obs = {"label": "placed_stable", "readouts": {"joint": fit, "fz_hold_n": 1.0}, "force": float(rows[key]["force"])}
        comp, info = rm.components(obs, None, None, b.arrays(), None)
        assert info["used"] == ["joint"]
        post = b.reweight(rm.combine(comp))
        a = post.arrays()
        assert float(post.weights @ a["mass"]) == pytest.approx(want["post"]["mass"]["mean"], rel=1e-6)
        assert float(post.weights @ a["com_frac"]) == pytest.approx(want["post"]["com"]["mean"], rel=1e-6, abs=1e-9)
        assert post.ess() == pytest.approx(want["ess"], rel=1e-6)
