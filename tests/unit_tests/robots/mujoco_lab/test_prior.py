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


"""The lab prior: CoM hypotheses of the corpus generator against the sim (needs the sim)."""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")

LONG = "ycb__banana"


def test_com_hypothesis_matches_the_sim():
    from robots.mujoco_lab.adapter import MuJoCoLabEnv
    from robots.mujoco_lab.prior import com_hypothesis

    env = MuJoCoLabEnv()
    for end in (+1, -1):
        env.reset(LONG, {"mu": 0.8, "mass_scale": 1.3, "com_s": 0.8, "v_lift": 0.15}, seed=0, com_end=end)
        pose, truth = env.pose_info(), env.truth()
        com, f = com_hypothesis(pose["pivot"], pose["com_obj_nominal"], 0.8, end)
        assert f > 0
        assert np.linalg.norm(np.asarray(com) - np.asarray(truth["com_obj"])) < 1e-3
        assert abs(pose["nominal_mass"] * 1.3 * (1 + f) - truth["mass"]) < 0.01 * truth["mass"]
    piv = pose["pivot"]
    s_nom = float(((np.asarray(pose["com_obj_nominal"]) - np.asarray(piv["centre"])) @ np.asarray(piv["axis"]) - piv["lo"]) / piv["L"])
    com, f = com_hypothesis(piv, pose["com_obj_nominal"], s_nom - 0.05, +1)         # a target below the nominal fraction
    assert f == 0.0 and np.allclose(com, pose["com_obj_nominal"])
    com, f = com_hypothesis(piv, pose["com_obj_nominal"], None, +1)
    assert f == 0.0 and np.allclose(com, pose["com_obj_nominal"])


def test_lab_prior_matches_the_generator():
    from scipy.stats import ks_2samp

    from robots.mujoco_lab.adapter import MuJoCoLabEnv
    from robots.mujoco_lab.corpus import draw_theta
    from robots.mujoco_lab.prior import lab_prior

    env = MuJoCoLabEnv()
    env.reset(LONG, seed=0)
    pose = env.pose_info()
    b = lab_prior(pose, 20000, np.random.default_rng(1))
    A = b.arrays()
    rng = np.random.default_rng(2)
    draws = [draw_theta(rng, pose["nominal_mass"])[0] for _ in range(20000)]
    assert ks_2samp(np.log(A["mass_scale"]), np.log([t["mass_scale"] for t in draws])).pvalue > 0.01
    assert ks_2samp(A["mu"], [t["mu"] for t in draws]).pvalue > 0.01
    assert abs(A["com_branch"].mean() - 0.5) < 0.02
    assert b.p_com_shifted() == pytest.approx(float((A["f"] > 0).mean()))
    assert b.p_com_shifted() < A["com_branch"].mean()                      # some banana targets lie below the nominal CoM
    assert np.allclose(A["mass"], pose["nominal_mass"] * A["mass_scale"] * (1 + A["f"]), rtol=0, atol=1e-9)
    assert b.ess() == pytest.approx(20000) and b.n_ancestors() == 20000


def test_particle_features_match_extract_with_the_true_com():
    from dataclasses import replace

    from robots.mujoco_lab.adapter import MuJoCoLabEnv
    from rpent.skills.pick_place.features import extract, particle_features

    env = MuJoCoLabEnv()
    env.reset(LONG, {"mu": 0.6, "mass_scale": 1.4, "com_s": 0.8, "v_lift": 0.15}, seed=0, com_end=-1)
    p = env.proposals()[1]
    pose, truth, g = env.proposal_pose(p), env.truth(), env.geometry()
    F = 15.0
    base = extract(g, np.asarray(p.point), p.yaw, 0.1, 0.5, F, 0.15, 0.5, "plate")      # placeholder mass and friction
    got = particle_features(base, {"mass": truth["mass"], "mu": truth["mu"], "com_obj": truth["com_obj"]}, pose, F)
    com = np.asarray(truth["com"])
    g_true = replace(g, com=com, com_height=float(com[2] - env.sim.obj_low_z()))
    want = extract(g_true, np.asarray(p.point), p.yaw, truth["mass"], truth["mu"], F, 0.15, 0.5, "plate")
    for k in ("lever", "com_h", "base_ratio", "com_height_ratio", "mass", "mu", "load_ratio"):
        assert got[k] == pytest.approx(want[k], abs=1e-6), k
    assert abs(got["lever"] - base["lever"]) > 0.005                        # the shift moved the lever
