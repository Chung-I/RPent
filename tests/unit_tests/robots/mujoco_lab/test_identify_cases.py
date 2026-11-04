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


"""identify_cases: gate R1's firm grip, the weigh check of each identify row against its weigh case, the timing run's
pick, the summary, and (on the arm sim) one identify probe on a scene-B draw."""

import argparse
import json
import os

import numpy as np
import pytest

SIM = os.environ.get("RPENT_MUJOCO_SIM_DIR", "")
DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "robots", "mujoco_lab", "data")


def test_the_firm_grip_is_the_larger_of_the_weigh_force_and_the_standard_force():
    from robots.mujoco_lab.identify_cases import firm_force

    assert firm_force(3.0, 5.0) == 5.0 and firm_force(7.5, 5.0) == 7.5


def test_maxdiff_compares_nested_values_and_flags_a_structure_mismatch():
    from robots.mujoco_lab.identify_cases import _maxdiff

    assert _maxdiff({"a": [1.0, 2.0], "b": "x"}, {"a": [1.0, 2.5], "b": "x"}) == 0.5
    assert _maxdiff({"a": 1.0}, {"a": 1.0, "b": 2.0}) == float("inf")
    assert _maxdiff([1.0], [1.0, 2.0]) == float("inf")
    assert _maxdiff({"a": None}, {"a": None}) == 0.0 and _maxdiff("x", "y") == float("inf")


def _weigh(**kw):
    return {"key": "a__v11s0__d1__Bweigh", "object": "a", "draw": 1, "seed": 0, "theta": {"mass_scale": 1.1, "com_s": None},
            "com_end": 1, "fill": 0.0, "theta_frag": None, "s": 0.5, "f_std": 6.0, "pose": {"pivot": {"L": 0.2, "lo": 0.0}}, **kw}


def test_the_weigh_check_reports_each_way_an_identify_scene_can_differ():
    from robots.mujoco_lab.identify_cases import weigh_check

    w = _weigh()
    assert weigh_check(_weigh(), w) == {"draws_equal": True, "s_equal": True, "pose_maxdiff": 0.0, "f_std_equal": True}
    c = weigh_check(_weigh(theta={"mass_scale": 1.2, "com_s": None}, s=0.45, f_std=6.5, pose={"pivot": {"L": 0.2, "lo": 0.01}}), w)
    assert c == {"draws_equal": False, "s_equal": False, "pose_maxdiff": pytest.approx(0.01), "f_std_equal": False}
    declined = {k: v for k, v in _weigh().items() if k not in ("s", "pose")}
    assert weigh_check(declined, w)["s_equal"] is False and weigh_check(declined, w)["pose_maxdiff"] is None


def test_the_timing_run_picks_cases_spread_over_the_sorted_cases():
    from robots.mujoco_lab.identify_cases import pick

    cases = [{"i": i} for i in range(10)]
    assert pick(cases, None) == cases and pick(cases, 10) == cases and pick(cases, 20) == cases
    assert [c["i"] for c in pick(cases, 3)] == [0, 4, 9]
    assert [c["i"] for c in pick(cases, 1)] == [0]


def test_the_summary_counts_kinds_declines_and_errors_and_the_checks():
    from robots.mujoco_lab.identify_cases import summarize

    ok = {"draws_equal": True, "s_equal": True, "pose_maxdiff": 0.0, "f_std_equal": True}
    rows = [{"kind": "measured", "case_wall_s": 10.0, "weigh_check": ok},
            {"kind": "lost", "case_wall_s": 20.0, "weigh_check": {**ok, "s_equal": False, "pose_maxdiff": 0.002}},
            {"declined": "no proposal", "case_wall_s": 1.0, "weigh_check": {**ok, "s_equal": False, "pose_maxdiff": None}},
            {"error": "RuntimeError: x", "case_wall_s": 3.0}]
    s = summarize(rows, 5)
    assert s["n_rows"] == 4 and s["n_weigh_cases"] == 5
    assert s["kinds"] == {"declined": 1, "error": 1, "lost": 1, "measured": 1}
    assert s["wall_s"]["total"] == 34.0 and s["wall_s"]["max"] == 20.0 and s["wall_s"]["median"] == 6.5
    assert s["checks"] == {"draws_unequal": 0, "s_unequal": 2, "f_std_unequal": 0, "pose_maxdiff": 0.002}
    assert summarize([], 0)["wall_s"] is None


def test_numpy_values_in_a_probe_record_are_written_as_json():
    from robots.mujoco_lab.identify_cases import _plain

    rec = {"a": np.float32(1.5), "b": np.arange(3), "c": np.int64(2), "d": np.bool_(True)}
    assert json.loads(json.dumps(rec, default=_plain)) == {"a": 1.5, "b": [0, 1, 2], "c": 2, "d": True}
    with pytest.raises(TypeError):
        json.dumps({"x": object()}, default=_plain)


def test_a_class_file_that_is_not_the_bundles_is_refused(tmp_path):
    from robots.mujoco_lab.identify_cases import run

    (tmp_path / "manifest.json").write_text(json.dumps({"classes": {"path": "c.json", "sha256": "0" * 64}}))
    c = tmp_path / "c.json"
    c.write_text("{}")
    a = argparse.Namespace(bundle=str(tmp_path), classes=str(c), thresholds=str(c), objects=None, limit=None, duration=6.0,
                           workers=1, out=str(tmp_path / "out"))
    with pytest.raises(SystemExit, match="classes"):
        run(a)


ARM = SIM and os.path.exists(os.path.join(SIM, "ident.py"))
MODEL = os.environ.get("RPENT_ARM_ROBOT_MODEL") or os.path.join(os.path.dirname(SIM), "out", "r1_robot.npz")


@pytest.mark.skipif(not (ARM and os.path.exists(MODEL)), reason="not the arm sim with its step-1 robot model")
def test_one_identify_probe_on_a_scene_b_draw(monkeypatch):
    """A short excitation (1 s) on one draw: a probe-corpus row (key, probe, kind) at the firm grip; a measured kind's
    reading is a floored fit that scores particles; the same draw again gives the same row."""
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    from robots.mujoco_lab.adapter import MuJoCoLabEnv
    from robots.mujoco_lab.identify_cases import identify_row
    from robots.mujoco_lab.identify_fit import fit_loglik

    env = MuJoCoLabEnv()
    args = (env, "hope__canned_tuna", 0, 0, os.path.join(DATA, "fragility_classes_v11c.json"), os.path.join(DATA, "thresholds_v1.json"))
    row = identify_row(*args, duration=1.0)
    assert row["key"].startswith("hope__canned_tuna__") and row["key"].endswith("s0__d0__Bidentify")
    assert row.get("kind") == "measured"             # this draw is measured (2026-11-04); the checks below need a reading
    assert row["scene"] == "B" and row["force"] == max(row["weigh_force"], row["f_std"])
    assert "declined" in row or (row["probe"] == "identify" and row["kind"] in
                                 ("measured", "not_lifted", "lost", "unplanned", "damaged", "unreachable"))
    if row.get("kind") == "measured":
        fit = row["reading"]
        assert fit["floor"] == [0.001, 0.0005] and fit["duration_s"] == 1.0
        ll = fit_loglik(fit, {"mass": np.array([0.2, 0.4]), "com_obj": np.zeros((2, 3))})
        assert np.all(np.isfinite(ll))
    again = identify_row(*args, duration=1.0)
    assert again.get("kind") == row.get("kind") and again.get("s") == row.get("s")
    if row.get("kind") == "measured":
        np.testing.assert_array_equal(again["reading"]["phi"], row["reading"]["phi"])


def test_the_git_state_of_a_folder_names_its_last_commit_and_its_changes(tmp_path):
    import subprocess

    from robots.mujoco_lab.identify_cases import git_state

    g = ["git", "-C", str(tmp_path)]
    subprocess.run(g + ["init", "-q"], check=True)
    (tmp_path / "sim").mkdir()
    (tmp_path / "sim" / "a.py").write_text("x = 1\n")
    subprocess.run(g + ["add", "sim/a.py"], check=True)
    subprocess.run(g + ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "a"], check=True)
    head = subprocess.run(g + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert git_state(str(tmp_path / "sim")) == {"commit": head, "dirty": False}
    (tmp_path / "sim" / "a.py").write_text("x = 2\n")
    assert git_state(str(tmp_path / "sim")) == {"commit": head, "dirty": True}
    assert git_state(None) is None
