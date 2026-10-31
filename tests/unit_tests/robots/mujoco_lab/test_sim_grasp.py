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

"""The sim's grasp.run_grasp outcome label (needs the sim).

The squeeze probe calls run_grasp with no lift and no hold. When the close raised the object more than 2 cm and it
ended in both pads, the label took a max() over the trace's lift and hold steps, of which there were none, and the
case ended in an error row: three orange-juice-carton cases in milestone 4 (seeds 14, 18 and 19). A trace with a lift
or a hold step labels as before."""

import json
import os

import numpy as np

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")

FOLDS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "robots", "mujoco_lab", "data", "m2b2", "folds")
needs_folds = pytest.mark.skipif(not os.path.exists(os.path.join(FOLDS, "crossfit.json")), reason="the m2b2 folds are not built")
POINT = np.array([0.0, 0.0, 0.05])


class _ScriptedSim:
    """Stands in for scene.Sim inside run_grasp. The object does what ``motion(stage, k, lift)`` says: the stage is
    "open" until the grip closes, "close" while the hand stays at the grasp height and "lift" once it rises (the hold
    included); k counts the stage's control steps and lift is the hand's height above the grasp. motion returns the
    object's rise (m), its turn (degrees) and the pads (left, right) it touches."""

    def __init__(self, motion):
        self.motion, self.hand_z, self.z_grasp, self.stage, self.k = motion, 0.0, None, "open", 0
        self.rise, self.turn, self.pads = 0.0, 0.0, (False, False)

    def teleport_hand(self, pos, yaw):
        self.hand_z = float(pos[2])

    def set_hand(self, pos, yaw):
        self.hand_z = float(pos[2])

    def grip(self, force):
        if force > 0 and self.z_grasp is None:
            self.z_grasp = self.hand_z

    def step_ctrl(self):
        lift = 0.0 if self.z_grasp is None else self.hand_z - self.z_grasp
        stage = "open" if self.z_grasp is None else ("lift" if lift > 1e-9 else "close")
        self.k = self.k + 1 if stage == self.stage else 0
        self.stage = stage
        self.rise, self.turn, self.pads = self.motion(stage, self.k, lift)

    def obj_pos(self):
        return POINT + np.array([0.0, 0.0, self.rise])

    def obj_mat(self):
        a = np.radians(self.turn)
        return np.array([[1.0, 0.0, 0.0], [0.0, np.cos(a), -np.sin(a)], [0.0, np.sin(a), np.cos(a)]])

    def contacts(self):
        return [], self.pads

    def zone_of(self, touching):
        return None


def _grasp():
    from robots.mujoco_lab.adapter import load_sim

    return load_sim()["grasp"]


def test_a_close_that_raises_the_object_with_no_lift_and_no_hold_labels_held():
    """The squeeze probe's call: the close raises the object 3 cm and it stays in both pads. There is no lift or hold
    step, so no turn during a lift or a hold: the label is held, not an error."""
    def squeezed_up(stage, k, lift):
        if stage == "open":
            return 0.0, 0.0, (False, False)
        return (0.03 if k >= 5 else 0.0), 0.0, (True, True)

    out, _, trace = _grasp().run_grasp(_ScriptedSim(squeezed_up), POINT, 0.0, force=1.0, v_lift=0.03, lift_h=0.0, hold_s=0.0)
    assert {x["phase"] for x in trace} == {"approach", "close"}
    assert out["label"] == "held" and out["max_z"] == pytest.approx(0.03)


def _held(stage, k, lift):
    return (0.0, 0.0, (False, False)) if stage == "open" else (lift, 0.0, (True, True))


def _turned_in_the_lift(stage, k, lift):
    return (0.0, 0.0, (False, False)) if stage == "open" else (lift, 40.0 if stage == "lift" and k >= 2 else 0.0, (True, True))


def _turned_in_the_close_only(stage, k, lift):
    return (0.0, 0.0, (False, False)) if stage == "open" else (lift, 40.0 if stage == "close" else 0.0, (True, True))


def _slipped(stage, k, lift):
    if stage == "lift" and k >= 20:
        return 0.0, 0.0, (False, False)
    return _held(stage, k, lift)


@pytest.mark.parametrize("motion, label", [(_held, "held"), (_turned_in_the_lift, "rotated"),
                                           (_turned_in_the_close_only, "held"), (_slipped, "slipped_out")])
def test_a_lift_with_no_hold_step_labels_as_before(motion, label):
    """A commit-style attempt with a lift and no hold step: the turn that counts is still the largest over the lift
    and hold steps only (a turn in the close alone does not make it rotated). These labels are the ones run_grasp gave
    before the fix (the test passed on the old code)."""
    out, _, trace = _grasp().run_grasp(_ScriptedSim(motion), POINT, 0.0, force=40.0, v_lift=0.15, lift_h=0.20, hold_s=0.0)
    assert "lift" in {x["phase"] for x in trace} and "hold" not in {x["phase"] for x in trace}
    assert out["label"] == label


@needs_folds
def test_the_orange_juice_carton_case_that_crashed_in_milestone_4_completes(tmp_path):
    """Milestone 4's seed 14, case 8 of hope__orange_juice_carton (policy, care 10, A 10, s 1, the m2b2 crossfit) ended
    in the error row "max() arg is an empty sequence" from its squeeze probe. It now completes with that probe."""
    from robots.mujoco_lab.calibrate import main

    cases, out = tmp_path / "cases.json", tmp_path / "out.json"
    cases.write_text(json.dumps({"seed": 14, "cases": {"hope__orange_juice_carton": [8]}}))
    main(["--crossfit", FOLDS, "--protocol", "policy", "--repeats", "11", "--seed", "14", "--workers", "1", "--care", "10",
          "--abstain-cost", "10", "--probe-time-scale", "1", "--cases", str(cases), "--out", str(out)])
    (row,) = json.load(open(out))["rows"]
    assert "error" not in row, row.get("error")
    assert "squeeze" in [p["probe"] for p in row["probes"]]
