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


"""The friction readouts on corpus attempts that read impossible values with end-of-step samples
over the close and lift (corpus v4 keys, seed 0, no fixtures; needs the sim)."""

import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")
G = 9.81


@pytest.fixture(scope="module")
def env():
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    return MuJoCoLabEnv()


def _row(env, key: str) -> dict:
    return _row_and_trace(env, key)[0]


def _row_and_trace(env, key: str) -> tuple[dict, list[dict]]:
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
    return row, got["r"].trace


def test_pads_on_top_give_no_slip_reading(env):
    """cheez_it d33 (pinch width 0): the open pads land on the box's top and press it into the table
    (the end-of-step reading was -9,597 N)."""
    r = _row(env, "ycb__cheez_it__d33")
    assert r["label"] == "not_lifted" and r["readouts"]["fz_slip_max_n"] is None


def test_a_pivot_gives_no_slip_reading(env):
    """hammer_1 d77: the head stays on the table while the pads hold the handle and the hammer turns
    (it read 17.5 N against 2 mu F = 55.6 N)."""
    r = _row(env, "handal__hammer_1__d77")
    assert r["label"] == "not_lifted" and r["readouts"]["fz_slip_max_n"] is None


def test_a_slide_reads_twice_mu_f(env):
    """hammer d66: the pads slide up a still hammer, so the reading is their friction capacity 2 mu F."""
    r = _row(env, "handal__hammer__d66")
    assert r["label"] == "not_lifted"
    assert r["readouts"]["fz_slip_max_n"] == pytest.approx(2 * r["truth"]["mu"] * r["force"], rel=0.1)


def test_a_drop_in_the_lift_reads_the_carried_weight(env):
    """lemon1 d36: the end-of-step sample before the loss read -0.61 N; the weight is 1.78 N."""
    r = _row(env, "fruits_veggies__lemon1__d36")
    assert r["loss_phase"] == "lift"
    assert 0.7 <= r["readouts"]["fz_at_loss_n"] / (r["truth"]["mass"] * G) <= 1.3


def test_a_loss_is_a_step_in_which_no_pad_pressed(env):
    """wood_block d78: the block turns in the pads and chatters. At the end of lift step 105 neither pad touched
    it, but over that step the pads pressed with 25 N and carried 13.8 N, and a pad held it at step 106."""
    r, tr = _row_and_trace(env, "ycb__wood_block__d78")
    assert r["loss_step"] is None or tr[r["loss_step"]]["fn"] == 0.0


def test_weigh_probe_gives_no_friction_when_the_pads_never_slid_on_the_sides(env):
    """pitcher d15 (probe corpus v3, scene B): the pads did not pinch the sides and read mu_hat -11.97."""
    from robots.mujoco_lab.probe_corpus import probe_rows

    b = next(r for r in probe_rows(env, "ycb__pitcher", 15, 7) if r["key"].endswith("__Bweigh"))
    assert b["kind"] == "lower_bound" and b["reading"] == {}
