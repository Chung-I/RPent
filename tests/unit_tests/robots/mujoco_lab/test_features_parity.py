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

"""The ported feature extractor must reproduce the stored t3 features (needs the sim and t3/features.json).

t3 took the pinch width from the hull face centres in the pads' band, a lower bound; the features now use the
exact width (``features.pinch_width``). The face-centre rule is still what ``extract`` does for a geometry
without hulls, so parity is checked on that rule, and the exact width against a linear program."""

import dataclasses
import json
import os

import numpy as np
import pytest

from tests.unit_tests.skills.test_features import _lp_width

SIM = os.environ.get("RPENT_MUJOCO_SIM_DIR")
T3 = os.path.join(os.path.dirname(SIM or "."), "t3", "features.json")
pytestmark = pytest.mark.skipif(not (SIM and os.path.exists(T3)), reason="sim dir or t3/features.json missing")

KEYS = ("ycb__banana__control", "ycb__banana__heavy", "ycb__mug__offcom", "ycb__wood_block__slippery", "ycb__tomato_soup_can__fast")


@pytest.mark.parametrize("key", KEYS)
def test_parity_with_t3_features(key):
    from robots.mujoco_lab.adapter import MuJoCoLabEnv, object_props
    from rpent.skills.pick_place.features import FEATURES, PAD_HALF, extract

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
    g = env.geometry()
    args = (np.asarray(p.point), p.yaw, truth["mass"], cause["mu"], row["F"], cause["v_lift"], 0.5, "bin")
    f, f0 = extract(g, *args), extract(dataclasses.replace(g, hulls=()), *args)      # exact width; t3's face-centre rule
    keys = ("width", "slope_deg", "mass", "load_ratio") + (("lever", "com_h") if cause["com_s"] is None else ())
    for k in keys:                                   # t3 measured lever and com_h from the true CoM; the skill uses the nominal one
        if np.isnan(row[k]):
            assert np.isnan(f0[k]), (k, f0[k], row[k])   # no face centre under the pads in both (the 10 cm wood block)
        else:
            tol = 5e-4 if k in ("lever", "com_h") else 1e-5     # the nominal CoM (props.json) and MuJoCo's subtree CoM differ by ~1e-4
            if k == "slope_deg":
                tol = 0.05      # the reset settles until at rest (at least 0.5 s), t3 settled 0.2 s: the pose tilts up to 0.013 deg more
            assert abs(f0[k] - row[k]) < tol, (k, f0[k], row[k])
    other = [k for k in FEATURES if k != "width"]
    np.testing.assert_equal([f[k] for k in other], [f0[k] for k in other])          # the exact width changes nothing else
    assert abs(f["width"] - _lp_width(g.hulls, p.point, p.yaw, PAD_HALF)) < 1e-6
    assert np.isnan(row["width"]) or f["width"] >= row["width"] - 1e-9             # the face-centre width is a lower bound
