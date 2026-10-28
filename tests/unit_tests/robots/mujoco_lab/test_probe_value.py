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

"""The probe recommendation on a recorded deployed case does not depend on the random stream (needs
the sim and the m2 bundle's models)."""

import os

import numpy as np
import pytest

SIM = os.environ.get("RPENT_MUJOCO_SIM_DIR")
BUNDLE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "robots", "mujoco_lab", "data", "m2"))
pytestmark = pytest.mark.skipif(not (SIM and os.path.exists(os.path.join(BUNDLE, "probe_weigh.joblib"))),
                                reason="sim dir or m2 bundle missing")


def test_probe_recommendation_is_stable_across_streams():
    """hope__pitted_cherries, case 21 of seed 2: the deployed run recommended probe_weigh there. With the
    case's belief and menu fixed, the case's own stream and 20 other streams give one recommendation."""
    from robots.mujoco_lab.adapter import object_props
    from robots.mujoco_lab.calibrate import _stream, case_draw
    from robots.mujoco_lab.robot_spec import build_lab
    from rpent.skills.pick_place import tools
    from rpent.skills.pick_place.destinations import MESH_IDS
    from rpent.skills.pick_place.menu import build_menu

    obj, idx, seed = "hope__pitted_cherries", 21, 2
    prims = build_lab(bundle=BUNDLE)
    env = prims.env
    theta, com_end, dest = case_draw(seed, obj, idx, object_props(obj)["mass"])
    env.reset(obj, theta, seed=seed * 1000 + idx, com_end=com_end, fixture=dest if dest in MESH_IDS else None)
    prims.rng = _stream(seed, obj, idx, 1)
    store = {}
    err, a = tools._assess(obj, dest, "normal", prims, store)
    assert err is None
    loc = env.locate(obj)
    belief = store["beliefs"][loc["object_id"]]["belief"]
    f_std = tools.standard_force(env.nominal_mass(loc["object_id"]))
    recs = [a.recommendation]
    for s in range(20):
        _, rec = build_menu(prims.model, belief, env.geometry(), env.proposals(), f_std, "normal", dest, env.plan_trajectory,
                            np.random.default_rng([s, 77]), pose_fn=env.proposal_pose, probe_models=prims.probe_models)
        recs.append(rec)
    assert len(set(recs)) == 1, {r: recs.count(r) for r in set(recs)}
