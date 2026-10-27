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

"""End-to-end calibration check: assess, execute, compare the declared probability with the outcome."""

import json
import os

import pytest

SIM = os.environ.get("RPENT_MUJOCO_SIM_DIR")
METRICS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "..", "..", "robots", "mujoco_lab", "data", "success_model_metrics.json")
METRICS = os.path.normpath(METRICS)
pytestmark = pytest.mark.skipif(not (SIM and os.path.exists(METRICS)), reason="sim dir or trained model missing")

KEYS = {"n", "n_executed", "recommendation_decline_rate", "abstained", "ece", "ece_ci95_object_bootstrap", "brier", "brier_const", "by_subgroup",
        "by_destination", "by_option", "rows", "kind", "objects"}


def test_calibrate_two_objects_one_repeat(tmp_path):
    from robots.mujoco_lab.calibrate import main

    out = tmp_path / "cal.json"
    res = main(["--metrics", METRICS, "--repeats", "1", "--out", str(out), "--objects", "ycb__banana,ycb__brick", "--seed", "1"])
    assert set(res) >= KEYS and res["n"] == 2 and res["n_executed"] == 2
    saved = json.load(open(out))
    assert saved["n"] == 2 and len(saved["rows"]) == 2
    row = saved["rows"][0]
    assert {"object", "destination", "option", "declared", "outcome", "placed_at_requested", "truth"} <= set(row)
    assert set(row["declared"]["p"]) == {"placed_stable", "tipped", "missed", "dropped_low", "dropped_high", "rolled_off", "not_lifted"}


def test_follow_recommendation_can_abstain(tmp_path):
    from robots.mujoco_lab.calibrate import main

    res = main(["--metrics", METRICS, "--repeats", "1", "--out", str(tmp_path / "c.json"), "--objects", "ycb__banana", "--seed", "2", "--follow-recommendation", "--threshold", "1.01"])
    assert res["n"] == 1 and res["n_executed"] == 0 and res["abstained"] == 1 and res["recommendation_decline_rate"] == 1.0
