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

"""The Stage 2c summary (daily-logs designs/2026-11-09-stage2-contacts-rerun-design.md, readings 1 to 4): the case-level
change against the published records, the three-column tables, and the summary command on synthetic runs."""

from robots.mujoco_lab import curve as cv
from robots.mujoco_lab import stage2c as s2c
from tests.unit_tests.robots.mujoco_lab.test_curve_oracle import _s2, _sim

K = "R0|10|2"


def _tipped_then(second_row, label2=None):
    m2 = {"status": "ok", "rows": [], "decisions": {K: second_row}}
    sec = {second_row: {"label": label2, "artifact": False, "error": None}} if label2 else {}
    return _sim("commit_std", "tipped", menu2=m2, second=sec)


def _pub_new():
    pub = [_s2("a", 0, {K: "commit_std"}, [_sim("commit_std", "placed_stable")]),
           _s2("a", 1, {K: "commit_std"}, [_tipped_then("slow_lift", "placed_stable")]),
           _s2("b", 0, {K: cv.GIVE_UP_ROW}, []),
           _s2("b", 1, {K: "commit_std"}, [_sim("commit_std", "placed_stable")]),
           _s2("c", 0, {K: "commit_std"}, [_sim("commit_std", "placed_stable")])]
    new = [_s2("a", 0, {K: "commit_std"}, [_sim("commit_std", "crushed")]),
           _s2("a", 1, {K: "commit_std"}, [_tipped_then(cv.GIVE_UP_ROW)]),
           _s2("b", 0, {K: cv.GIVE_UP_ROW}, []),
           _s2("b", 1, {K: "commit_std"}, [_sim("commit_std", "placed_stable")], dropped=True)]
    return pub, new


def test_the_case_change_counts_final_labels_outcomes_and_decisions_on_the_cases_kept_in_both():
    pub, new = _pub_new()
    ch = s2c.case_change(pub, new, [K])[K]
    assert ch["cases"] == 3 and ch["changed"] == 2
    assert ch["by_transition"] == {"placed_stable -> crushed": 1, "placed_stable -> give_up": 1}
    assert ch["by_object"] == {"a": {"cases": 2, "changed": 2}, "b": {"cases": 1, "changed": 0}}
    assert ch["outcomes"] == {"success -> irr": 1, "success -> give_up": 1, "give_up -> give_up": 1}
    assert ch["first_agree"] == 3 and ch["second_both"] == 1 and ch["second_agree"] == 0
    assert ch["second_attempts"] == {"published": 1, "new": 0} and ch["give_ups"] == {"published": 1, "new": 2}
    assert ch["dropped_new_only"] == [["b", 1]] and ch["dropped_published_only"] == [] and ch["missing"] == [["c", 0]]


def test_a_changed_first_decision_is_counted_against_the_agreement():
    pub, new = _pub_new()
    new[2] = _s2("b", 0, {K: "commit_std"}, [_sim("commit_std", "placed_stable")])
    ch = s2c.case_change(pub, new, [K])[K]
    assert ch["first_agree"] == 2 and ch["outcomes"]["give_up -> success"] == 1
