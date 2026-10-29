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

"""The contact monitor: a rollout whose pad contacts were not physical is flagged (needs the sim).

MuJoCo's native convex collider can give a pad-hull contact a normal that pushes the object against the pad's
closing direction (thin hull walls), and a pinch wider than the opening drives the pads deep into the object.
While the grip squeezes, the adapter counts the substeps with such a reversed contact and keeps the deepest pad
penetration; a reset that did not reach rest is flagged too (fix_impact/README.md)."""

import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")


@pytest.fixture(scope="module")
def env():
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    return MuJoCoLabEnv()


def _run(env, key):
    """The corpus draw of `key` (seed 0, no fixtures), executed even when the corpus declines its pinch."""
    from robots.mujoco_lab.adapter import object_props
    from robots.mujoco_lab.corpus import case_rng, draw_action, draw_theta

    obj, idx = key.rsplit("__d", 1)
    rng = case_rng(0, obj, int(idx))
    theta, com_end = draw_theta(rng, object_props(obj)["mass"])
    act = draw_action(rng, env.s["t3_make"].std_force(object_props(obj)["mass"]), False)
    env.reset(obj, theta, seed=int(idx), com_end=com_end)
    prop = next(p for p in env.proposals() if abs(p.s - act["s"]) < 1e-9)
    return env.execute(prop, act["force"], theta["v_lift"], act["v_transport"], act["destination"], act["release_height"])


def test_the_artifact_rule():
    from robots.mujoco_lab.adapter import DEEP_M, REV_STEPS, artifact

    assert (REV_STEPS, DEEP_M) == (5, 0.003)
    assert artifact(5, 0.0) and not artifact(4, 3.0) and artifact(0, 3.01)
    assert artifact(0, 0.0, at_rest=False) and not artifact(0, 0.0, at_rest=True)


def test_a_wedged_storage_box_is_flagged(env):
    """storage_box d7: the pinch is wider than the opening, and the pads land on the box's thin walls."""
    r = _run(env, "hot3d__storage_box__d7")
    assert r.artifact and (r.rev_steps >= 5 or r.deepest_mm > 3.0)


def test_a_soup_can_pinch_is_clean(env):
    r = _run(env, "ycb__tomato_soup_can__d0")
    assert not r.artifact and r.rev_steps == 0 and r.deepest_mm < 3.0


def test_a_reset_that_did_not_reach_rest_flags_the_rollout(env):
    """butter d0 chatters into a slow spin on the table (the native collider at rest)."""
    r = _run(env, "hope__butter__d0")
    assert env.reset_info["at_rest"] is False and r.artifact


def test_the_corpus_row_and_the_weigh_probe_carry_the_monitor(env):
    from robots.mujoco_lab.corpus import rollout_row
    from rpent.skills.pick_place.probes import weigh_force

    row = rollout_row(env, "ycb__tomato_soup_can", 0, 0)
    assert {"rev_steps", "deepest_mm", "artifact"} <= set(row) and row["artifact"] is False
    env.reset("ycb__brick", {"mu": 0.8, "mass_scale": 1.0, "com_s": None, "v_lift": 0.15}, seed=0)
    mid = next(p for p in env.proposals() if abs(p.s - 0.5) < 1e-9)
    res = env.probe_weigh(mid, weigh_force(env.s["t3_make"].std_force(env.nominal_mass())))
    d = res.to_data()
    assert d["rev_steps"] == 0 and d["deepest_mm"] < 3.0 and d["artifact"] is False
    assert env.probe_push().to_data()["artifact"] is False
