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

import math

import numpy as np
import pytest

from robots.mujoco_lab import palm_audit as pa

SIM = pytest.mark.skipif(not __import__("os").environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")
COMMIT = ('commit_std|{"a_transport": 2.0, "destination": "table_centre", "force": 10.0, "release_height": null, "s": 0.5, '
          '"v_lift": 0.15, "v_transport": 0.5}')


def test_an_action_string_is_its_row_id_and_its_fields():
    rid, act = pa.parse_action(COMMIT)
    assert rid == "commit_std"
    assert act == {"a_transport": 2.0, "destination": "table_centre", "force": 10.0, "release_height": None, "s": 0.5, "v_lift": 0.15,
                   "v_transport": 0.5}


def test_a_rows_action_is_found_by_its_id_and_an_absent_or_repeated_id_raises():
    rows = [{"id": "commit_std", "action": COMMIT}, {"id": "slow_lift", "action": COMMIT.replace("commit_std", "slow_lift", 1)}]
    assert pa.action_of(rows, "commit_std")["s"] == 0.5
    with pytest.raises(ValueError):
        pa.action_of(rows, "firm_grip")
    with pytest.raises(ValueError):
        pa.action_of(rows + rows[:1], "commit_std")
    with pytest.raises(ValueError):
        pa.action_of([{"id": "slow_lift", "action": COMMIT}], "slow_lift")       # the string names another row


def test_the_flags_are_the_deepest_overlap_against_each_threshold():
    clear = pa.flags_of(None)
    assert clear["depth_mm"] == 0.0 and not any(clear["flags"].values()) and clear["pairs"] == {}
    c = {"pairs": ["palm/object", "finger/table"], "first": {},
         "by_pair": {"palm/object": {"phase": "approach", "step": 3, "steps": 4, "depth": 0.0031, "geoms": ["palm", "hull0"],
                                     "phases": {"approach": {"steps": 3, "depth": 0.0031}, "close": {"steps": 1, "depth": 0.002}}},
                     "finger/table": {"phase": "lower", "step": 90, "steps": 1, "depth": 0.0004, "geoms": ["finger50", "table"],
                                      "phases": {"lower": {"steps": 1, "depth": 0.0004}}}}}
    f = pa.flags_of(c)
    assert f["depth_mm"] == pytest.approx(3.1)
    assert f["flags"] == {"any": True, "gt1": True, "gt2": True, "gt5": False}
    assert f["pairs"]["palm/object"]["body"] == "palm" and f["pairs"]["palm/object"]["kind"] == "object"
    assert f["pairs"]["palm/object"]["phases"]["close"]["depth_mm"] == pytest.approx(2.0)
    assert f["pairs"]["finger/table"]["first_phase"] == "lower" and f["pairs"]["finger/table"]["depth_mm"] == pytest.approx(0.4)


def test_a_replay_differs_on_the_label_the_artifact_or_the_error():
    sim = {"label": "dropped_low", "artifact": False, "error": None}
    assert pa.replay_differs(sim, {"outcome": "dropped_low"}, True) == {}
    assert pa.replay_differs(sim, {"outcome": "tipped"}, True) == {"label": ["dropped_low", "tipped"]}
    assert pa.replay_differs(sim, {"outcome": "dropped_low"}, False) == {"artifact": [False, True]}       # not at rest
    assert pa.replay_differs(sim, {"error": "boom"}, True) == {"label": ["dropped_low", None], "error": [None, "boom"]}


def _rec(decisions, sims):
    return {"object": "o", "idx": 0, "decisions": decisions, "sims": sims, "dropped": False}


def test_an_episodes_executed_rows_are_its_attempts_with_their_labels():
    s1 = {"seed_k": 0, "first": "commit_std", "label": "dropped_low", "menu2": {"decisions": {"R0|10|2": "slow_lift", "R4|10|-": "give_up"}},
          "second": {"slow_lift": {"label": "placed_stable"}}}
    s2 = {"seed_k": 0, "first": "firm_grip", "label": "placed_stable", "menu2": None, "second": {}}
    r = _rec({"R0|10|2": "commit_std", "R4|10|-": "firm_grip", "R1|10|2": "give_up"}, [s1, s2])
    assert pa.episode_rows(r, "R0|10|2") == [(1, "commit_std", "commit_std", "dropped_low"), (2, "commit_std", "slow_lift", "placed_stable")]
    assert pa.episode_rows(r, "R4|10|-") == [(1, "firm_grip", "firm_grip", "placed_stable")]
    assert pa.episode_rows(r, "R1|10|2") == []
    r2 = _rec({"R4|10|-": "commit_std"}, [s1])
    assert pa.episode_rows(r2, "R4|10|-") == [(1, "commit_std", "commit_std", "dropped_low")]      # gave up at the second


def test_a_ceiling_without_the_flagged_rows_drops_them_from_the_union():
    rows = [{"id": "across_0.5", "feasible": True, "current": True, "label": "placed_stable", "artifact": False, "error": None},
            {"id": "along_0.1", "feasible": True, "current": False, "label": "placed_stable", "artifact": False, "error": None},
            {"id": "across_0.3", "feasible": True, "current": True, "label": "tipped", "artifact": False, "error": None}]
    assert pa.ceiling_without(rows, set(), False)["success"] and pa.ceiling_without(rows, set(), True)["success"]
    c10, c3 = pa.ceiling_without(rows, {"across_0.5"}, False), pa.ceiling_without(rows, {"across_0.5"}, True)
    assert c10["success"] and c10["rows"] == ["along_0.1"] and not c3["success"] and c3["rows"] == ["across_0.3"]
    gone = pa.ceiling_without(rows, {"across_0.5", "along_0.1", "across_0.3"}, False)
    assert gone["give_up"] and not gone["success"]


# ---------------------------------------------------------------- on the lab

def _sugar_box():
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    env = MuJoCoLabEnv()
    env.reset("ycb__sugar_box", seed=0)
    return env


@SIM
def test_the_pinch_of_an_action_is_the_default_pinch_at_its_fraction():
    env = _sugar_box()
    assert pa.proposal_at(env, 0.5) == env.proposals()[1]
    with pytest.raises(ValueError):
        pa.proposal_at(env, 0.4)


@SIM
def test_todays_path_is_the_hand_path_with_each_yaw_as_a_rotation_about_the_vertical():
    env = _sugar_box()
    p, act = env.proposals()[1], pa.parse_action(COMMIT)[1]
    got = pa.today_path(env, p, act)
    want = env.hand_path(p, 0.15, 0.5, "table_centre", None, a_transport=2.0)
    assert [x[0] for x in got] == [x[0] for x in want] and all(np.array_equal(x[1], y[1]) for x, y in zip(got, want))
    assert all(np.allclose(x[2], [[math.cos(y[2]), -math.sin(y[2]), 0.0], [math.sin(y[2]), math.cos(y[2]), 0.0], [0.0, 0.0, 1.0]],
                           rtol=0.0, atol=1e-15) for x, y in zip(got, want))


@SIM
def test_a_vertical_pinch_at_mid_height_of_an_upright_box_is_flagged_on_the_palm():
    env = _sugar_box()                     # upright, 17.5 cm tall: the palm sits inside it at a mid-height pinch
    p, act = env.proposals()[1], pa.parse_action(COMMIT)[1]
    f = pa.audit_path(env, p, act)
    assert f["flags"]["gt5"] and f["pairs"]["palm/object"]["first_phase"] == "approach"
    assert set(f["pairs"]["palm/object"]["phases"]) <= {"approach", "close"} and "close" in f["pairs"]["palm/object"]["phases"]
    high = max(env.proposals((0.1, 0.9)), key=lambda q: q.point[2])  # 1.8 cm below the top: the palm is above the box
    top = pa.audit_path(env, high, act)
    assert top["depth_mm"] == 0.0 and not top["flags"]["any"] and top["pairs"] == {}


@SIM
def test_a_rows_audit_carries_its_key_and_pinch_and_leaves_the_sim_as_it_found_it():
    env = _sugar_box()
    act = pa.parse_action(COMMIT)[1]
    before = pa.state_sha(env)
    r = pa.audit_row(env, act, first="commit_std", attempt=1, row="commit_std")
    assert pa.state_sha(env) == before
    assert r["first"] == "commit_std" and r["attempt"] == 1 and r["s"] == 0.5 and r["error"] is None
    assert r["point"] == [float(x) for x in env.proposals()[1].point] and r["flags"]["gt5"]
    bad = pa.audit_row(env, dict(act, s=0.4), row="x")                 # no default pinch at 0.4: the row's error
    assert bad["error"].startswith("ValueError") and "flags" not in bad
    q = env.proposals((0.9,))[0]
    assert pa.audit_row(env, act, prop=q, id="across_0.9")["s"] == 0.9    # a given pinch (the pose set's, the pilot's)


@SIM
def test_the_state_hash_sees_a_moved_object():
    env = _sugar_box()
    a = pa.state_sha(env)
    env.sim.d.qpos[0] += 1e-6
    assert pa.state_sha(env) != a
