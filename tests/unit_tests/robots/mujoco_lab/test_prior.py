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
