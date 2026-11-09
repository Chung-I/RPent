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

"""The lab side of the sixdof rerun's free move (design 2026-11-08-sixdof-rerun-design.md, unit (e), gate G3): the
world cuRobo sees, the robot's sphere refit, and the cached plan as the executor's joint waypoints."""
"""The level-A rerun driver (design 2026-11-08-sixdof-rerun-design.md, unit (f)): the funnel of the 30–45° rows. The
tests target the contacts sim (armsim ee882f9b)."""

import math
import os

import numpy as np
import pytest

from robots.mujoco_lab import curve_sixdof_rerun as rr
from robots.mujoco_lab import sixdof

SIM = pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")


@SIM
def test_the_level_a_funnel_keeps_the_30_to_45_degree_rows_and_says_where_each_stops():
    from tests.unit_tests.robots.mujoco_lab.test_sixdof import _at, _sugar_box

    env, _ = _sugar_box()
    top = env.proposals((0.9,), turn=-math.pi / 2)[0]            # the box's 6 cm side, 1.8 cm below its top: palm clear
    mid = env.proposals((0.5,), turn=-math.pi / 2)[0]            # the 6 cm side at mid-height: the palm inside the box
    G = np.stack([_at(top, 10.0), _at(top, 36.0), _at(mid, 40.0), _at(top, 55.0)])
    rows, counts = rr.funnel6(env, G, np.array([0.9, 0.8, 0.7, 0.6]), "box", {"object": "ycb__sugar_box", "idx": 0})
    assert counts["generated"] == 4 and counts["level_a"] == 2 and [r["id"] for r in rows] == ["g1", "g2"]
    assert all(r["bin"] == "30–45°" and 30.0 < r["approach_deg"] <= 45.0 for r in rows)
    ok, palm = rows
    assert ok["why"] is None and ok["candidate"]["index"] == 0 and len(ok["candidate"]["q"]) == 7
    assert palm["why"] == "clearance: palm/object" and palm["clearance"]["first"]["pair"] == "palm/object" and palm["candidate"] is None
    assert counts["chosen"] == 1
