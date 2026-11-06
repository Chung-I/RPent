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

"""The success-damage curve (curve; daily-logs designs/2026-11-06-success-damage-curve-design.md): Stage 0's points from
the m4 sweep."""

import gzip
import json
import os

import pytest

from robots.mujoco_lab import curve as cv


def _case(ended, labels=(), probe_damage=False, artifact=False, error=False):
    r = {"ended": ended, "artifact": artifact, "attempts": [{"label": x} for x in labels],
         "probes": [{"kind": "damaged" if probe_damage else "measured", "damage": "crushed" if probe_damage else None}]}
    if error:
        r["error"] = "boom"
    return r


def test_m4_point_counts_success_damage_and_give_up_over_counted_cases():
    rows = [_case("placed", ["placed_stable"]), _case("placed", ["tipped", "placed_stable"]),
            _case("failed", ["tipped", "crushed"]), _case("failed", ["not_lifted"] * 3), _case("damaged", probe_damage=True),
            _case("abstained"), _case("no_row"), _case("unreachable"),
            _case("placed", ["placed_stable"], artifact=True), _case("error", error=True)]
    p = cv.m4_point(rows)
    assert p["n"] == 8                                     # the artifact and the error case are not counted
    assert p["success"] == pytest.approx(2 / 8)
    assert p["irreversible"] == pytest.approx(2 / 8)       # an irreversible attempt label or a damaging probe
    assert p["probe_damage"] == pytest.approx(1 / 8)
    assert p["give_up"] == pytest.approx(3 / 8)            # abstained, no_row, unreachable
    assert p["second_attempt"] == pytest.approx(3 / 8)     # cases with more than one attempt


def test_stage0_reads_both_protocols_per_cell_at_one_time_scale(tmp_path):
    def write(name, rows):
        with gzip.open(os.path.join(tmp_path, name), "wt") as f:
            json.dump({"rows": rows}, f)

    write("policy_k10_A10_s1_seed12.json.gz", [_case("placed", ["placed_stable"]), _case("abstained")])
    write("commit_k10_A10_s1_seed12.json.gz", [_case("failed", ["rolled_off"]), _case("placed", ["placed_stable"])])
    write("policy_k10_A10_s0_seed12.json.gz", [_case("abstained")])              # another time scale: not read
    write("policy_k10_A10_s1_seed13.json.gz", [_case("abstained")])              # another seed: not read
    write("policy_no_probe_k10_A10_s1_seed12.json.gz", [_case("abstained")])     # another protocol: not read
    pts = cv.stage0_points(str(tmp_path))
    assert [(p["protocol"], p["care"], p["A"]) for p in pts] == [("commit", 10.0, 10.0), ("policy", 10.0, 10.0)]
    pol = pts[1]
    assert pol["r"] == pytest.approx(10.0 / 50.0)          # r = A / (5 kappa)
    assert pol["success"] == pytest.approx(0.5) and pol["give_up"] == pytest.approx(0.5)
    assert pts[0]["irreversible"] == pytest.approx(0.5)


def test_label_groups_merge_points_whose_labels_would_overlap():
    groups = cv._label_groups([(2.5, 50.0, 4.0), (2.5, 50.0, 2.0), (1.5, 47.9, 0.2), (1.6, 47.9, 0.6), (0.0, 7.0, 0.00667)])
    assert groups == [((0.0, 7.0), [0.00667]), ((1.6, 47.9), [0.2, 0.6]), ((2.5, 50.0), [2.0, 4.0])]
