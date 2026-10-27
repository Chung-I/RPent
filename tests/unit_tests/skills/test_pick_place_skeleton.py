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

"""Skeleton checks for the pick_place skill and the mujoco_lab robot package."""


def test_packages_import_and_robot_is_listed():
    import rpent.skills.pick_place  # noqa: F401
    from rpent.robots.base import enumerate_robots

    assert "mujoco_lab" in enumerate_robots()
