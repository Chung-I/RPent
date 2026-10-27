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

"""The ported feature extractor must reproduce the stored t3 features (needs the sim and t3/features.json)."""

import json
import os

import numpy as np
import pytest

SIM = os.environ.get("RPENT_MUJOCO_SIM_DIR")
T3 = os.path.join(os.path.dirname(SIM or "."), "t3", "features.json")
pytestmark = pytest.mark.skipif(not (SIM and os.path.exists(T3)), reason="sim dir or t3/features.json missing")

KEYS = ("ycb__banana__control", "ycb__banana__heavy", "ycb__mug__offcom", "ycb__wood_block__slippery", "ycb__tomato_soup_can__fast")


@pytest.mark.parametrize("key", KEYS)
def test_parity_with_t3_features(key):
    from robots.mujoco_lab.adapter import MuJoCoLabEnv, object_props
    from rpent.skills.pick_place.features import extract

    stored = {r["key"]: r for r in json.load(open(T3))}
    if key not in stored:
        pytest.skip(f"{key} not in t3/features.json")
    row = stored[key]
    env = MuJoCoLabEnv()
    props = object_props(row["object"])
    cause = env.s["t3_make"].causes_for(props["mass"])[row["cause"]]
    env.reset(row["object"], cause, seed=0, com_end=+1)
    p = env.proposals()[1]
    truth = env.truth()
    f = extract(env.geometry(), np.asarray(p.point), p.yaw, truth["mass"], cause["mu"], row["F"], cause["v_lift"], 0.5, "bin")
    for k in ("width", "slope_deg", "lever", "com_h", "mass", "load_ratio"):
        if np.isnan(row[k]):
            assert np.isnan(f[k]), (k, f[k], row[k])     # no face under the pads in both (the 10 cm wood block)
        else:
            assert abs(f[k] - row[k]) < 1e-5, (k, f[k], row[k])
