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


"""The weigh and push probes in the lab (needs the sim)."""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")
G = 9.81


def _env():
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    return MuJoCoLabEnv()


def _mid(env):
    return next(p for p in env.proposals() if abs(p.s - 0.5) < 1e-9)


def _weigh(env):
    from rpent.skills.pick_place.probes import weigh_force

    return env.probe_weigh(_mid(env), weigh_force(env.s["t3_make"].std_force(env.nominal_mass())))


def test_weigh_probe_measures_the_brick():
    env = _env()
    env.reset("ycb__brick", {"mu": 0.8, "mass_scale": 1.0, "com_s": None, "v_lift": 0.15}, seed=0)
    res = _weigh(env)
    m = env.truth()["mass"]
    assert res.probe == "weigh" and res.kind == "measured" and not res.thrown
    assert abs(res.reading["m_hat"] - m) <= 0.15 * m
    assert res.displacement_m < 0.03 and res.seconds > 0 and res.trace


def test_weigh_probe_reads_mu_when_the_object_cannot_rise():
    """mu 0.1 at the mass-scale cap 3.0: the brick weighs 1.47x the pads' capacity 2 mu F_hi and does not
    rise (checked: 0.04 cm), so the probe reads the friction from the slip force."""
    env = _env()
    env.reset("ycb__brick", {"mu": 0.1, "mass_scale": 3.0, "com_s": None, "v_lift": 0.15}, seed=0)
    res = _weigh(env)
    assert res.kind == "lower_bound" and set(res.reading) == {"mu_hat", "m_lower"}
    assert 0.05 <= res.reading["mu_hat"] <= 0.2


def test_push_probe_slides_the_brick():
    env = _env()
    env.reset("ycb__brick", {"mu": 0.5, "mass_scale": 1.0, "com_s": None, "v_lift": 0.15}, seed=0)
    res = env.probe_push()
    t = env.truth()
    assert res.probe == "push" and res.kind == "slid"
    assert 0.5 <= res.reading["f_break"] / (t["mu"] * t["mass"] * G) <= 2.0


def test_push_force_counts_every_hand_geom():
    """A slippery brick: the pads alone read 0 N here, so a slide must carry a hand force."""
    env = _env()
    env.reset("ycb__brick", {"mu": 0.15, "mass_scale": 1.0, "com_s": None, "v_lift": 0.15}, seed=0)
    res = env.probe_push()
    assert res.kind in ("slid", "no_contact")
    if res.kind == "slid":
        assert res.reading["f_break"] > 0.05
    if res.kind == "no_contact":
        assert res.reading == {}


def test_probes_refresh_the_baselines():
    env = _env()
    env.reset("ycb__brick", {"mu": 0.5, "mass_scale": 1.0, "com_s": None, "v_lift": 0.15}, seed=0)
    env.probe_push()
    low_after = env.sim.obj_low_z()
    res = _weigh(env)
    assert res.kind == "measured" and env.rest_low_z == pytest.approx(low_after, abs=1e-6)


@pytest.mark.parametrize("obj", ["ycb__brick", "ycb__mug"])
def test_weigh_probe_reads_the_weight_within_two_percent(obj):
    """The hand follows its mocap target in jumps, so a force sampled at the end of a lift step reads
    0.9 of the weight; the reading comes from the still hold."""
    env = _env()
    env.reset(obj, {"mu": 0.8, "mass_scale": 1.0, "com_s": None, "v_lift": 0.15}, seed=0)
    res = _weigh(env)
    assert res.kind == "measured" and res.reading["m_hat"] == pytest.approx(env.truth()["mass"], rel=0.02)


def test_push_reads_a_slippery_brick():
    """mu 0.15: each jump of the hand strikes the brick, which glides off between the samples; the
    force averaged over the physics substeps still carries the push."""
    env = _env()
    env.reset("ycb__brick", {"mu": 0.15, "mass_scale": 1.0, "com_s": None, "v_lift": 0.15}, seed=0)
    res = env.probe_push()
    t = env.truth()
    assert res.kind == "slid" and 0.5 <= res.reading["f_break"] / (t["mu"] * t["mass"] * G) <= 2.0


def test_a_probe_that_leaves_the_object_off_the_table_throws_it():
    """The mug of calibrate seed 2, case 1: after the push, the weigh flips it onto the left pad and the
    rising hand carries it 10 cm up. It rests on nothing the skill can reach, so the probe threw it."""
    from robots.mujoco_lab.adapter import object_props
    from robots.mujoco_lab.calibrate import case_draw

    env = _env()
    theta, com_end, _ = case_draw(2, "ycb__mug", 1, object_props("ycb__mug")["mass"])
    env.reset("ycb__mug", theta, seed=2001, com_end=com_end)
    assert not env.probe_push().thrown
    res = _weigh(env)
    assert not env.locate("object")["reachable"]               # the case: the object hangs on the hand
    assert res.thrown
