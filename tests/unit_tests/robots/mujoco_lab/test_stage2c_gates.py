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

"""The gates of Stage 2c and oracle-c (daily-logs designs/2026-11-09-stage2-contacts-rerun-design.md, reading 5): the
records of the new code path with the flags off on the published sim equal the published records in every field but
the seconds, and the case lists of the new runs equal the published runs'."""

import copy
import json
import os

import pytest

from robots.mujoco_lab import curve as cv
from robots.mujoco_lab import stage2c as s2c


def _rec(o, idx, side="validation", label="placed_stable", second=None):
    sim = {"seed_k": 0, "reset_seed": 11000, "first": "commit_std", "label": label, "artifact": False, "error": None,
           "menu2": None, "second": second or {}, "seconds": 3.2}
    return {"object": o, "idx": idx, "side": side, "decisions": {"R0|10|2": "commit_std"}, "first_rows": ["commit_std"],
            "row_stats": {"commit_std": {"p_placed": 0.9, "p_irr": 0.01, "q_irr": 0.02}}, "sims": [sim], "skipped": None,
            "error": None, "dropped": False, "stop": None, "n_seeds": 1, "seconds": 7.5}


def test_records_equal_but_for_the_seconds_pass():
    pub = [_rec("a", 0), _rec("a", 1), _rec("b", 0)]
    new = copy.deepcopy(pub[:2])
    new[0]["seconds"], new[0]["sims"][0]["seconds"] = 99.0, 1.0
    g = s2c.compare_published(new, pub)
    assert g == {"compared": 2, "equal": 2, "missing": [], "differ": []}


def test_a_changed_field_anywhere_but_the_seconds_fails_and_is_named():
    pub = [_rec("a", 0, second={"slow_lift": {"label": "placed_stable", "artifact": False, "error": None}}), _rec("a", 1)]
    new = copy.deepcopy(pub)
    new[0]["sims"][0]["second"]["slow_lift"]["label"] = "tipped"
    new[1]["decisions"]["R0|10|2"] = "give_up"
    g = s2c.compare_published(new, pub)
    assert g["equal"] == 0 and g["differ"] == [{"object": "a", "idx": 0, "fields": ["sims"]}, {"object": "a", "idx": 1, "fields": ["decisions"]}]


def test_a_new_field_fails_the_gate():
    pub = [_rec("a", 0)]
    new = copy.deepcopy(pub)
    new[0]["sims"][0]["replan"] = {"d_mm": 0.0}
    assert s2c.compare_published(new, pub)["differ"] == [{"object": "a", "idx": 0, "fields": ["sims"]}]


def test_a_case_the_published_records_lack_is_missing():
    g = s2c.compare_published([_rec("a", 0), _rec("z", 3)], [_rec("a", 0)])
    assert g["missing"] == [["z", 3]] and g["equal"] == 1 and g["compared"] == 1


def _run(d, recs, cases=None):
    os.makedirs(d)
    json.dump({"cases": cases if cases is not None else [[r["object"], r["idx"]] for r in recs]}, open(os.path.join(d, "run_config.json"), "w"))
    for side in cv.SIDES:
        with open(cv.records_path(d, side), "w") as f:
            for r in recs:
                if r["side"] == side:
                    f.write(json.dumps(r) + "\n")
    open(os.path.join(d, "run.exit"), "w").write("EXIT 0\n")
    return d


def test_the_published_gate_command_writes_its_report_and_exits_3_on_a_difference(tmp_path):
    pub = [_rec("a", 0), _rec("a", 1, side="test")]
    p = _run(str(tmp_path / "pub"), pub)
    n = _run(str(tmp_path / "new"), [copy.deepcopy(pub[0])])
    g = s2c.main(["gate-published", "--runs", n, "--published", p, "--out", str(tmp_path / "g1")])
    assert g["pass"] and g["compared"] == 1 and json.load(open(tmp_path / "g1" / "gate_published.json"))["pass"]
    bad = copy.deepcopy(pub[0])
    bad["sims"][0]["label"] = "crushed"
    n2 = _run(str(tmp_path / "new2"), [bad])
    with pytest.raises(SystemExit) as e:
        s2c.main(["gate-published", "--runs", n2, "--published", p, "--out", str(tmp_path / "g2")])
    assert e.value.code == 3 and not json.load(open(tmp_path / "g2" / "gate_published.json"))["pass"]
    with pytest.raises(SystemExit, match="exists"):
        s2c.main(["gate-published", "--runs", n, "--published", p, "--out", str(tmp_path / "g1")])


def test_the_published_gate_refuses_a_run_without_exit_0(tmp_path):
    p = _run(str(tmp_path / "pub"), [_rec("a", 0)])
    n = _run(str(tmp_path / "new"), [_rec("a", 0)])
    open(os.path.join(n, "run.exit"), "w").write("EXIT 1\n")
    with pytest.raises(SystemExit, match="EXIT 0"):
        s2c.main(["gate-published", "--runs", n, "--published", p, "--out", str(tmp_path / "g")])


def test_the_case_list_gate_compares_the_runs_cases_with_the_published_runs(tmp_path):
    p = [_run(str(tmp_path / "p0"), [], cases=[["a", 0], ["a", 1]]), _run(str(tmp_path / "p1"), [], cases=[["b", 0]])]
    same = [_run(str(tmp_path / "n0"), [], cases=[["b", 0], ["a", 1]]), _run(str(tmp_path / "n1"), [], cases=[["a", 0]])]
    g = s2c.main(["gate-cases", "--runs", ",".join(same), "--published", ",".join(p), "--out", str(tmp_path / "g1")])
    assert g["pass"] and g["cases"] == 3 and g["missing"] == [] and g["extra"] == [] and g["twice"] == []
    short = [_run(str(tmp_path / "s0"), [], cases=[["a", 0], ["a", 0]]), _run(str(tmp_path / "s1"), [], cases=[["c", 9]])]
    with pytest.raises(SystemExit) as e:
        s2c.main(["gate-cases", "--runs", ",".join(short), "--published", ",".join(p), "--out", str(tmp_path / "g2")])
    g = json.load(open(tmp_path / "g2" / "gate_cases.json"))
    assert e.value.code == 3 and g["missing"] == [["a", 1], ["b", 0]] and g["extra"] == [["c", 9]] and g["twice"] == [["a", 0]]


def test_a_new_top_level_field_fails_the_gate():
    new = [dict(_rec("a", 0), oracle=True)]
    assert s2c.compare_published(new, [_rec("a", 0)])["differ"] == [{"object": "a", "idx": 0, "fields": ["oracle"]}]


def test_a_case_listed_twice_fails_the_case_list_gate_alone(tmp_path):
    p = [_run(str(tmp_path / "p0"), [], cases=[["a", 0], ["a", 1]])]
    twice = [_run(str(tmp_path / "n0"), [], cases=[["a", 0], ["a", 1]]), _run(str(tmp_path / "n1"), [], cases=[["a", 1]])]
    with pytest.raises(SystemExit) as e:
        s2c.main(["gate-cases", "--runs", ",".join(twice), "--published", ",".join(p), "--out", str(tmp_path / "g")])
    g = json.load(open(tmp_path / "g" / "gate_cases.json"))
    assert e.value.code == 3 and g["twice"] == [["a", 1]] and g["missing"] == [] and g["extra"] == []
