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


"""The friction readouts and the loss step on corpus attempts that once read impossible values: end-of-step
samples over the close and lift (corpus v4), a pivot taken for a slide once the hand moved smoothly, and a
loss in a step over which the pads still carried the object (keys of the corpus draws, seed 0, no fixtures;
needs the sim)."""

import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")


@pytest.fixture(scope="module")
def env():
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    return MuJoCoLabEnv()


@pytest.fixture
def any_width(env, monkeypatch):
    """Run pinches wider than the 9.0 cm limit, which the corpora decline since v8: a substrate may still run
    one, and its readouts must still be right."""
    monkeypatch.setattr(type(env), "max_pinch_width", float("inf"))


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
        row = corpus.rollout_row(env, obj, int(idx), 0, False, mode="m2")     # a v8 row: v8's world
    finally:
        env.execute = plain
    return row, got["r"].trace


def test_pads_on_top_give_no_slip_reading(env, any_width):
    """cheez_it d33 (17 cm across the box; the face-centre width read 0): the open pads land on the box's top and
    press it into the table (the end-of-step reading was -9,597 N)."""
    r = _row(env, "ycb__cheez_it__d33")
    assert r["label"] == "not_lifted" and r["readouts"]["fz_slip_max_n"] is None


def test_a_pivot_gives_no_slip_reading(env):
    """potato_masher d57: the masher turns about a table contact 18 cm from the pads, next to its body
    origin. The origin rose 0.1 mm and the turn stayed under 3 deg for 15 steps, but the object point
    between the pads rose with the hand (it read 6.4 N against 2 mu F = 31.2 N)."""
    r = _row(env, "hot3d__potato_masher__d57")
    assert r["label"] == "not_lifted" and r["readouts"]["fz_slip_max_n"] is None


def test_a_slide_reads_twice_mu_f(env):
    """hammer d66: the pads slide up a still hammer, so the reading is their friction capacity 2 mu F."""
    r = _row(env, "handal__hammer__d66")
    assert r["label"] == "not_lifted"
    assert r["readouts"]["fz_slip_max_n"] == pytest.approx(2 * r["truth"]["mu"] * r["force"], rel=0.1)


def test_a_slide_out_in_the_lift_reads_the_pads_friction(env):
    """tomato_soup_can d78: the grip limit 2 mu F is 1.11 times the weight, and the speed-up of the 0.8 m/s
    lift asks 1.41 times. The pads slide up the can at that limit while it rises 3.6 cm, then run off its
    top. The loss reading, the mean over the last three two-pad steps, is the sliding friction less the
    exit step (0.74 of 2 mu F; an end-of-step sample before a loss once read -0.61 N, lemon1 d36)."""
    r = _row(env, "ycb__tomato_soup_can__d78")
    assert r["label"] == "dropped_low" and r["loss_phase"] == "lift"
    assert 0.6 <= r["readouts"]["fz_at_loss_n"] / (2 * r["truth"]["mu"] * r["force"]) <= 1.1


def test_a_loss_is_a_step_in_which_no_pad_pressed(env, any_width):
    """wood_block d78: the block turns in the pads and chatters. At the end of lift step 105 neither pad touched
    it, but over that step the pads pressed with 25 N and carried 13.8 N, and a pad held it at step 106."""
    r, tr = _row_and_trace(env, "ycb__wood_block__d78")
    assert r["loss_step"] is None or tr[r["loss_step"]]["fn"] == 0.0


def test_weigh_probe_gives_no_friction_when_the_pads_never_slid_on_the_sides(env, any_width):
    """pitcher d15 (probe corpus v3, scene B): the pads did not pinch the sides and read mu_hat -11.97."""
    from robots.mujoco_lab.probe_corpus import probe_rows

    b = next(r for r in probe_rows(env, "ycb__pitcher", 15, 7) if r["key"].endswith("__Bweigh"))
    assert b["kind"] == "lower_bound" and b["reading"] == {}


def test_a_pivot_is_not_lifted_and_the_hand_sets_it_back(env):
    """hammer_7 d55: the hammer turns about its head on the table while the pads raise the handle, and its
    origin rises 12.8 cm; the head never leaves the table. The hand does not carry it off (corpus v6 dragged
    it to the table centre and read tipped): it opens where it is and rises."""
    r, tr = _row_and_trace(env, "handal__hammer_7__d55")
    assert r["label"] == "not_lifted" and r["release"] is None
    assert not any(x["phase"] in ("transport", "lower") for x in tr)
    assert not any(env.sim.contacts()[1]) and r["zone"] not in (None, "floor")       # it lies on the table top
    assert abs(env.sim.obj_low_z() - 0.75) < 0.01


def test_a_set_down_on_the_box_wall_is_judged_by_the_release_rules(env):
    """green_serving_spoon d91: lowered into the box, the 31 cm spoon lands across the walls and stays on the wall
    tops, 12 cm above the box floor, so it missed the box. (wood_block d10, the earlier case, which corpus v6
    read as a 12 cm drop, is declined since corpus v8: its pinch is 9.1 cm.)"""
    r = _row(env, "handal__green_serving_spoon__d91")
    assert r["label"] == "missed" and r["loss_phase"] is None
