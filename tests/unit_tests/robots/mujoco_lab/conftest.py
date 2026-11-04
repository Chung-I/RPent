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

"""The lab tests' sim kind: ``@pytest.mark.hand_sim`` marks a test of the hand sim's own API or numbers (teleports,
the scripted hand, pad force readings, frozen hand-sim references); it skips on the arm sim (a sim folder with
ident.py, the arm port's), whose tests are test_arm_port.py and test_arm_sensing.py."""
from __future__ import annotations

import os

import pytest


def arm_sim() -> bool:
    sim = os.environ.get("RPENT_MUJOCO_SIM_DIR", "")
    return bool(sim) and os.path.exists(os.path.join(sim, "ident.py"))


def pytest_configure(config):
    config.addinivalue_line("markers", "hand_sim: a test of the hand sim (skipped on the arm sim)")


def pytest_collection_modifyitems(config, items):
    if not arm_sim():
        return
    skip = pytest.mark.skip(reason="hand sim only (RPENT_MUJOCO_SIM_DIR is the arm sim)")
    for item in items:
        if item.get_closest_marker("hand_sim"):
            item.add_marker(skip)
