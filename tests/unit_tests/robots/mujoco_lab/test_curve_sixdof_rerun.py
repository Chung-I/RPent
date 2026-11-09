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


def test_clearance_and_reach6_disagree_only_when_one_passes_and_the_other_fails_on_a_hand_box():
    boxes = {"palm": "palm", "wrist": "wrist", "": "finger"}            # reach6's names of the boxes (the fingers' are unnamed)
    finger = {"index": 1, "reason": "collision", "phase": "close", "step": 60, "hit": ["", "hull2", -0.001]}
    link = {"index": 0, "reason": "collision", "phase": "approach", "step": 30, "hit": ["arm_link7_c", "hull2", -0.002]}
    ik = {"index": 0, "reason": "ik", "phase": "approach", "step": 54, "ik_error": [0.0013, 0.02]}
    assert rr.box_hit([link, finger], boxes) == {"index": 1, "box": "finger", "phase": "close", "other": "hull2"}
    assert rr.box_hit([link, ik], boxes) is None and rr.box_hit([], boxes) is None
    clear = {"pairs": ["palm/object"], "first": {"pair": "palm/object", "phase": "approach", "step": 40}}
    assert rr.disagree(None, None, rr.box_hit([finger], boxes)) == "clearance pass, reach6 box hit"
    assert rr.disagree(clear, {"index": 3}, None) == "clearance fail, reach6 pass"
    assert rr.disagree(None, None, None) is None                        # reach6 fails on an arm link or the IK, or has no goal
    assert rr.disagree(clear, None, rr.box_hit([finger], boxes)) is None   # both fail
    assert rr.disagree(None, {"index": 0}, None) is None                # both pass



def test_a_row_stops_at_its_first_failing_step_and_its_fields_record_both_checks():
    boxes = {"palm": "palm", "wrist": "wrist", "": "finger"}
    clear = {"pairs": ["palm/object"], "first": {"pair": "palm/object", "phase": "approach", "step": 40}, "by_pair": {}}
    ik = {"index": 0, "reason": "ik", "phase": "approach", "step": 54, "ik_error": [0.0013, 0.02]}
    finger = {"index": 1, "reason": "collision", "phase": "close", "step": 60, "hit": ["", "hull2", -0.001]}
    cand = {"index": 1, "twin": 1, "seed": "aimed", "q": np.arange(7.0)}
    # the clearance fails, reach6 passes: the row stops at the clearance and keeps no candidate
    out, stop = rr.assess(clear, {"goal": True, "candidate": cand, "failed": [ik]}, boxes)
    assert stop == "clearance: palm/object" and out["candidate"] is None and out["disagree"] == "clearance fail, reach6 pass"
    assert out["clearance"] == {"first": clear["first"], "pairs": ["palm/object"]}
    assert out["reach"] == {"goal": True, "candidate": 1, "box_hit": None, "failed": [ik]}
    # the clearance passes, reach6 fails on a finger box: the row stops at the path check
    out, stop = rr.assess(None, {"goal": True, "candidate": None, "failed": [ik, finger]}, boxes)
    assert stop == "path: ik (approach)" and out["clearance"] is None and out["candidate"] is None
    assert out["reach"]["box_hit"] == {"index": 1, "box": "finger", "phase": "close", "other": "hull2"}
    assert out["disagree"] == "clearance pass, reach6 box hit"
    # both pass: the candidate goes to the planner
    out, stop = rr.assess(None, {"goal": True, "candidate": cand, "failed": [ik]}, boxes)
    assert stop is None and out["disagree"] is None
    assert out["candidate"] == {"index": 1, "twin": True, "seed": "aimed", "q": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    out, stop = rr.assess(None, {"goal": False, "candidate": None, "failed": []}, boxes)
    assert stop == "path: no candidate" and out["reach"]["goal"] is False

@SIM
def test_the_funnel_runs_reach6_also_after_a_clearance_failure_and_the_row_stops_at_the_clearance():
    from tests.unit_tests.robots.mujoco_lab.test_sixdof import _at, _sugar_box

    env, _ = _sugar_box()
    sim, m = env.sim, env.sim.m
    # reach6 names a geom by its MJCF name: the finger boxes have none, and they are the only robot geoms without one
    assert {g for g in sim.arm.robot_geoms() if not m.geom(g).name} == set(sixdof.hand_geoms(sim)["finger"])
    assert rr.hand_boxes(sim) == {"palm": "palm", "wrist": "wrist", "": "finger"}
    top, low, mid = (env.proposals((s,), turn=-math.pi / 2)[0] for s in (0.9, 0.7, 0.5))
    G = np.stack([_at(top, 36.0), _at(low, 36.0), _at(mid, 40.0)])
    rows, counts = rr.funnel6(env, G, np.array([0.9, 0.8, 0.7]), "box", {"object": "ycb__sugar_box", "idx": 0})
    ok, palm, ik = rows
    assert ok["why"] is None and ok["reach"]["candidate"] == ok["candidate"]["index"] and ok["reach"]["box_hit"] is None
    # the palm inside the box in the approach: the clearance and reach6 both see it, the row stops at the clearance
    assert palm["why"] == "clearance: palm/object" and palm["candidate"] is None and palm["reach"]["candidate"] is None
    assert palm["reach"]["box_hit"] == {"index": 0, "box": "palm", "phase": "approach", "other": "hull0"}
    # the palm inside the box at mid-height: reach6's IK fails in the approach before any collision check
    assert ik["why"] == "clearance: palm/object" and ik["reach"]["goal"] and ik["reach"]["box_hit"] is None
    assert [f["reason"] for f in ik["reach"]["failed"]] == ["ik", "ik"]
    assert [r["disagree"] for r in rows] == [None, None, None] and counts["disagree"] == 0
