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


"""Experiment tracking: one wandb run per job, off in tests."""

from robots.mujoco_lab.tracking import log_run


def test_log_run_is_off_when_disabled(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    assert log_run("unit-test", {"a": 1}, {"ece": 0.1, "nested": {"brier": 0.5}}) is None
