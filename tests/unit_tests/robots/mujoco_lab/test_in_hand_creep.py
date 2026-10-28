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


"""The pads hold what friction can hold: a firm grip does not creep (needs the sim).

MuJoCo's friction is a soft constraint. Under a steady load well inside the friction cone a held object
drifted along the pads at a few millimetres a second, so in a slow lift a grip far stronger than the
weight crept off the object. The noslip solver removes the drift."""

import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")


@pytest.fixture(scope="module")
def env():
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    return MuJoCoLabEnv()


def _lift(env, key: str):
    """The corpus row of `key` (seed 0, no fixtures), and (hand rise, grip_dz) on its lift steps with both pads."""
    from robots.mujoco_lab import corpus

    got, plain = {}, env.execute

    def execute(*a, **k):
        got["r"] = plain(*a, **k)
        return got["r"]

    env.execute = execute
    try:
        obj, idx = key.rsplit("__d", 1)
        row = corpus.rollout_row(env, obj, int(idx), 0, False)
    finally:
        env.execute = plain
    tr = got["r"].trace
    i0 = next(i for i, x in enumerate(tr) if x["phase"] == "lift")
    lift = [x for x in tr[i0:] if x["phase"] == "lift" and all(x["pads"])]
    return row, [(x["hand_z"] - tr[i0 - 1]["hand_z"], x["grip_dz"]) for x in lift]


@pytest.mark.parametrize("key", ["hope__yogurt_cup__d34", "ycb__scissors__d91"])
def test_a_firm_grip_does_not_creep(env, key):
    """yogurt_cup d34: 2 mu F is 154 times the weight. In the 4 s lift at 0.05 m/s the tapered cup crept 5 mm
    up and 15 mm sideways out of the pads (dropped_high). scissors d91: 2 mu F is 6.3 times the weight. They
    crept along the pads for 1.4 s, then slid 35 mm (dropped_low)."""
    row, lift = _lift(env, key)
    assert len(lift) >= 200 and max(abs(g - h) for h, g in lift) < 0.001
    assert row["label"] == "placed_stable"
