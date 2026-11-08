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

"""The skip list of Stage 2c and oracle-c (daily-logs designs/2026-11-09-stage2-contacts-rerun-design.md, the megaphone
ruling): a run leaves out the listed cases and keeps the list with its reason in its config; runs that disagree on it
do not join; the Stage 2 checks accept exactly the listed cases as absent; the oracle summary refuses oracle runs whose
list is not Stage 2's; and the case-list gate passes only when the published cases the new runs lack are exactly the
list's."""

import gzip
import json
import os

import pytest

from robots.mujoco_lab import curve as cv
from robots.mujoco_lab import curve_oracle as co
from robots.mujoco_lab import stage2c as s2c
from tests.unit_tests.robots.mujoco_lab.test_curve import _cv, _fake_stage2_object, _row, _s1, _write_run
from tests.unit_tests.robots.mujoco_lab.test_curve_oracle import _fake_oracle_case, _summary, _world

REASON = "the reset differs from Stage 1's on the contacts sim (finger_l box inside the object at the home pose)"


def _skip(path, cases, reason=REASON):
    json.dump({"reason": reason, "cases": cases}, open(path, "w"))
    return str(path)


# ------------------------------------------------------------------ the list

def test_a_skip_list_keeps_its_reason_its_cases_in_order_and_its_sha256(tmp_path):
    p = _skip(tmp_path / "skip.json", ["hot3d__megaphone:3", "hot3d__megaphone:0"])
    assert cv.read_skip_list(p) == {"path": p, "sha256": cv._sha(p), "reason": REASON,
                                    "cases": [["hot3d__megaphone", 0], ["hot3d__megaphone", 3]]}
    assert cv.read_skip_list(None) is None


@pytest.mark.parametrize("reason,cases,match", [("", ["a:0"], "no reason"), ("  ", ["a:0"], "no reason"),
                                                (REASON, [], "no cases"), (REASON, ["a:0", "a:0"], "twice")])
def test_a_skip_list_without_a_reason_or_cases_or_with_a_case_twice_is_refused(tmp_path, reason, cases, match):
    with pytest.raises(SystemExit, match=match):
        cv.read_skip_list(_skip(tmp_path / "skip.json", cases, reason))


# ------------------------------------------------------------------ the runs

def _stage2_world(tmp_path):
    rows1 = [dict(_row("commit_std", 0.9, irr=0.0), cvar=_cv(0.1))]
    _write_run(str(tmp_path / "a"), [_s1("ycb__a", 0, rows=rows1), _s1("ycb__a", 1, rows=rows1), _s1("ycb__t", 0, rows=rows1, side="test")])
    retry = tmp_path / "retry.json.gz"
    with gzip.open(retry, "wt") as f:
        json.dump({"rows": [{"attempts": [{"label": "tipped", "cost_s": 8.0}, {"label": "placed_stable", "cost_s": 8.0}]}]}, f)
    return ["stage2", "--stage1", str(tmp_path / "a"), "--retry", str(retry)]


@pytest.fixture()
def stage2_env(tmp_path, monkeypatch):
    from robots.mujoco_lab import identify_fit

    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(identify_fit, "require_bundle_generator", lambda b: "v11")
    seen = []
    monkeypatch.setattr(cv, "stage2_object", lambda job: seen.append(job) or {"records": [], "stop": None})
    return _stage2_world(tmp_path), seen


def test_stage2_leaves_out_the_skip_lists_cases_and_records_the_list(tmp_path, stage2_env):
    argv, seen = stage2_env
    p = _skip(tmp_path / "skip.json", ["ycb__a:1"])
    cv.main(argv + ["--out", str(tmp_path / "s2"), "--skip-list", p])
    assert sorted((rec1["object"], rec1["idx"]) for o, items, s in seen for rec1, _ in items) == [("ycb__a", 0), ("ycb__t", 0)]
    cfg = json.load(open(tmp_path / "s2" / "run_config.json"))
    assert cfg["cases"] == [["ycb__a", 0], ["ycb__t", 0]] and cfg["skip_list"] == cv.read_skip_list(p)
    cv.main(argv + ["--out", str(tmp_path / "plain")])
    assert json.load(open(tmp_path / "plain" / "run_config.json"))["skip_list"] is None


def test_a_skip_list_case_that_stage1_lacks_or_that_cases_names_is_refused_before_any_job(tmp_path, stage2_env):
    argv, seen = stage2_env
    with pytest.raises(SystemExit, match="not in the Stage 1 records"):
        cv.main(argv + ["--out", str(tmp_path / "x"), "--skip-list", _skip(tmp_path / "s1.json", ["ycb__zz:3"])])
    with pytest.raises(SystemExit, match="on the skip list"):
        cv.main(argv + ["--out", str(tmp_path / "y"), "--skip-list", _skip(tmp_path / "s2.json", ["ycb__a:1"]),
                        "--cases", "ycb__a:0,ycb__a:1"])
    assert seen == [] and not os.path.exists(tmp_path / "x" / "run_config.json") and not os.path.exists(tmp_path / "y" / "run_config.json")


def test_oracle_run_leaves_out_the_skip_lists_cases_and_records_the_list(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    _world(tmp_path)
    seen = []
    monkeypatch.setattr(co, "oracle_object", lambda job: seen.append(job) or {"records": [], "stop": None})
    monkeypatch.setattr(co, "require_bundle_generator", lambda b: "v11")
    p = _skip(tmp_path / "skip.json", ["o3:0", "o3:1"])
    co.main(["run", "--stage1", str(tmp_path / "s1"), "--locked", str(tmp_path / "locked.json"), "--out", str(tmp_path / "o"),
             "--sides", "test", "--skip-list", p])
    got = sorted((r["object"], r["idx"]) for o, items, s in seen for r in items)
    assert len(got) == 14 and ("o3", 0) not in got and ("o3", 1) not in got
    cfg = json.load(open(tmp_path / "o" / "run_config.json"))
    assert cfg["skip_list"] == cv.read_skip_list(p) and ["o3", 0] not in cfg["cases"] and len(cfg["cases"]) == 14


def test_stage2_runs_that_differ_in_the_skip_list_do_not_join(tmp_path):
    dirs = []
    for name, v in (("x", {"path": "s.json", "sha256": "h", "reason": REASON, "cases": [["a", 0]]}), ("y", None)):
        d = tmp_path / name
        os.makedirs(d)
        json.dump({"bundle": "B", "skip_list": v}, open(d / "run_config.json", "w"))
        json.dump({"stop": None}, open(d / "run_metrics.json", "w"))
        for side in cv.SIDES:
            open(cv.records_path(str(d), side), "w").close()
        dirs.append(str(d))
    with pytest.raises(SystemExit, match="differ in skip_list"):
        cv.load_stage2(dirs)


# ------------------------------------------------------------------ the Stage 2 checks

def _stage2_runs(tmp_path, monkeypatch):
    from robots.mujoco_lab import identify_fit

    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(identify_fit, "require_bundle_generator", lambda b: "v11")
    monkeypatch.setattr(cv, "stage2_object", _fake_stage2_object)
    rows_a = [dict(_row("commit_std", 0.6, irr=0.0), cvar=_cv(0.1)), dict(_row("firm_grip", 0.95, irr=0.03), cvar=_cv(0.1))]
    rows_b = [dict(_row("commit_std", 0.9, irr=0.001), cvar=_cv(0.1))]
    s1 = [_s1(f"ycb__v{i}", j, rows=rows_a if j else rows_b) for i in range(4) for j in range(2)]
    s1 += [_s1(f"hope__t{i}", j, rows=rows_a if j else rows_b, side="test") for i in range(4) for j in range(2)]
    _write_run(str(tmp_path / "s1"), s1)
    retry = tmp_path / "retry.json.gz"
    with gzip.open(retry, "wt") as f:
        json.dump({"rows": [{"attempts": [{"label": "tipped", "cost_s": 8.0}, {"label": "placed_stable", "cost_s": 8.0}]}]}, f)
    argv = ["stage2", "--stage1", str(tmp_path / "s1"), "--retry", str(retry)]
    p = _skip(tmp_path / "skip.json", ["ycb__v0:0", "ycb__v0:1"])
    cv.main(argv + ["--out", str(tmp_path / "skip"), "--skip-list", p])
    cv.main(argv + ["--out", str(tmp_path / "full")])
    return str(tmp_path / "skip"), str(tmp_path / "full")


def _lines(d, side):
    return open(cv.records_path(d, side)).read().splitlines()


def test_the_stage2_checks_accept_exactly_the_skip_lists_cases_as_absent(tmp_path, monkeypatch):
    skip, full = _stage2_runs(tmp_path, monkeypatch)
    lk = cv.main(["stage2-lock", "--runs", skip, "--out", str(tmp_path / "lock")])
    assert lk["lock"]["n"] == 6 and lk["checks"]["missing_cases"] == 0
    assert lk["checks"]["skip_list_cases"] == 2 and lk["checks"]["skip_list_present"] == 0
    # a listed case that is there anyway fails the checks
    v0 = [x for x in _lines(full, "validation") if json.loads(x)["object"] == "ycb__v0"][0]
    with open(cv.records_path(skip, "validation"), "a") as f:
        f.write(v0 + "\n")
    with pytest.raises(SystemExit, match="skip_list_present"):
        cv.main(["stage2-lock", "--runs", skip, "--out", str(tmp_path / "lock2")])


def test_the_stage2_checks_refuse_a_missing_case_that_is_not_on_the_skip_list(tmp_path, monkeypatch):
    skip, _ = _stage2_runs(tmp_path, monkeypatch)
    keep = [x for x in _lines(skip, "validation") if json.loads(x)["object"] != "ycb__v1"]
    with open(cv.records_path(skip, "validation"), "w") as f:
        f.write("".join(x + "\n" for x in keep))
    with pytest.raises(SystemExit, match="missing_cases"):
        cv.main(["stage2-lock", "--runs", skip, "--out", str(tmp_path / "lock")])


# ------------------------------------------------------------------ the oracle summary

def _oracle(tmp_path, monkeypatch, out, extra=()):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(co, "oracle_object", lambda job: {"records": [_fake_oracle_case(None, r, job[2], None, job[2]["knob"])
                                                                      for r in job[1]], "stop": None})
    monkeypatch.setattr(co, "require_bundle_generator", lambda b: "v11")
    co.main(["run", "--stage1", str(tmp_path / "s1"), "--locked", str(tmp_path / "locked.json"), "--out", str(out),
             "--sides", "test", *extra])
    (out / "run.exit").write_text("EXIT 0\n")


def test_oracle_runs_that_differ_in_the_skip_list_do_not_join(tmp_path, monkeypatch):
    _world(tmp_path)
    cases = [f"o{i}:{j}" for i in range(8) for j in range(2)]
    _oracle(tmp_path, monkeypatch, tmp_path / "o1", ["--cases", ",".join(cases[:8])])
    _oracle(tmp_path, monkeypatch, tmp_path / "o2", ["--cases", ",".join(cases[8:])])
    cfg = json.load(open(tmp_path / "o2" / "run_config.json"))
    json.dump(dict(cfg, skip_list={"path": "s.json", "sha256": "h", "reason": REASON, "cases": [["o9", 0]]}),
              open(tmp_path / "o2" / "run_config.json", "w"))
    with pytest.raises(SystemExit, match="differ in skip_list"):
        _summary(tmp_path, tmp_path / "sum", runs=f"{tmp_path / 'o1'},{tmp_path / 'o2'}")


def test_the_oracle_summary_refuses_oracle_runs_whose_skip_list_is_not_stage2s(tmp_path, monkeypatch):
    _world(tmp_path)
    _oracle(tmp_path, monkeypatch, tmp_path / "o1")
    cfg = json.load(open(tmp_path / "o1" / "run_config.json"))
    json.dump(dict(cfg, skip_list={"path": "s.json", "sha256": "h", "reason": REASON, "cases": [["o9", 0]]}),
              open(tmp_path / "o1" / "run_config.json", "w"))
    with pytest.raises(SystemExit, match="skip lists"):
        _summary(tmp_path, tmp_path / "sum")


# ------------------------------------------------------------------ the case-list gate

def _run(d, cases, skip_list=None):
    os.makedirs(d)
    json.dump({"cases": cases, "skip_list": skip_list}, open(os.path.join(d, "run_config.json"), "w"))
    return str(d)


def _gate(tmp_path, runs, pub, out, skip=None):
    argv = ["gate-cases", "--runs", ",".join(runs), "--published", ",".join(pub), "--out", str(tmp_path / out)]
    return s2c.main(argv + (["--skip-list", skip] if skip else []))


def test_the_case_list_gate_passes_when_the_new_runs_lack_exactly_the_skip_lists_cases(tmp_path):
    p = _skip(tmp_path / "skip.json", ["a:1"])
    sl = cv.read_skip_list(p)
    pub = [_run(tmp_path / "p0", [["a", 0], ["a", 1]]), _run(tmp_path / "p1", [["b", 0]])]
    new = [_run(tmp_path / "n0", [["b", 0]], sl), _run(tmp_path / "n1", [["a", 0]], sl)]
    g = _gate(tmp_path, new, pub, "g", p)
    assert g["pass"] and g["skipped"] == [["a", 1]] and g["missing"] == [] and g["cases"] == 2 and g["skip_list"] == sl


def test_the_case_list_gate_fails_on_a_missing_case_off_the_list_a_listed_case_present_or_a_run_without_the_list(tmp_path):
    p = _skip(tmp_path / "skip.json", ["a:1"])
    sl = cv.read_skip_list(p)
    pub = [_run(tmp_path / "p0", [["a", 0], ["a", 1], ["b", 0]])]
    for name, runs, field, want in (
            ("off", [_run(tmp_path / "n0", [["a", 0]], sl)], "missing", [["b", 0]]),
            ("present", [_run(tmp_path / "n1", [["a", 0], ["a", 1], ["b", 0]], sl)], "skip_list_present", [["a", 1]]),
            ("nolist", [_run(tmp_path / "n2", [["a", 0], ["b", 0]], None)], "runs_off_list", [str(tmp_path / "n2")])):
        with pytest.raises(SystemExit) as e:
            _gate(tmp_path, runs, pub, name, p)
        g = json.load(open(tmp_path / name / "gate_cases.json"))
        assert e.value.code == 3 and not g["pass"] and g[field] == want, name


def test_the_case_list_gate_fails_on_a_listed_case_that_is_not_a_published_case(tmp_path):
    p = _skip(tmp_path / "skip.json", ["a:1", "z:7"])
    sl = cv.read_skip_list(p)
    pub = [_run(tmp_path / "p0", [["a", 0], ["a", 1]])]
    with pytest.raises(SystemExit) as e:
        _gate(tmp_path, [_run(tmp_path / "n0", [["a", 0]], sl)], pub, "g", p)
    g = json.load(open(tmp_path / "g" / "gate_cases.json"))
    assert e.value.code == 3 and g["skip_list_not_published"] == [["z", 7]] and g["skipped"] == [["a", 1]]
