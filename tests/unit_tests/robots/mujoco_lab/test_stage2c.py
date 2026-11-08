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

"""Stage 2c and oracle-c (daily-logs designs/2026-11-09-stage2-contacts-rerun-design.md): the runs' executor flags (the
re-plan at the hold, the planner without the hand boxes, both off in a published run), each attempt's re-plan reading,
the sim the runs agree on, and the model lock (the bundle's files as the published runs read them, checked before the
first case)."""

import gzip
import json
import os
import time
from types import SimpleNamespace

import pytest

from robots.mujoco_lab import curve as cv
from robots.mujoco_lab import curve_oracle as co
from robots.mujoco_lab import model_lock as ml
from tests.unit_tests.robots.mujoco_lab.test_curve import _cv, _dec_for, _FakeLab, _row, _s1, _write_run
from tests.unit_tests.robots.mujoco_lab.test_curve_oracle import _world

FLAGS = ("replan_hold", "plan_without_hand_boxes")
READING = {"held": True, "lifted": True, "planned_mm": -100.0, "measured_mm": -104.5, "d_mm": 4.5, "raise_mm": 4.5,
           "unreachable": False}


# ------------------------------------------------------------------ the executor flags of a job

def test_the_executor_flags_come_from_the_job_and_are_off_when_absent():
    env = SimpleNamespace(replan_hold=None, plan_without_hand_boxes=None)
    cv.executor_flags(env, {"replan_hold": True, "plan_without_hand_boxes": True})
    assert env.replan_hold is True and env.plan_without_hand_boxes is True
    cv.executor_flags(env, {"bundle": "B"})                       # a published job: the published executor
    assert env.replan_hold is False and env.plan_without_hand_boxes is False


def _fake_prims(monkeypatch):
    from robots.mujoco_lab import calibrate as cal
    from robots.mujoco_lab import prior

    prims = SimpleNamespace(env=SimpleNamespace(replan_hold=None, plan_without_hand_boxes=None, force_noise=None), manifest={},
                            probe_models=None, reset_s=None)
    monkeypatch.setattr(cal, "_primitives", lambda b, x, y: prims)
    monkeypatch.setattr(cal, "job_force_noise", lambda x, m: 0.05)
    monkeypatch.setattr(prior, "fragility_world", lambda m: None)
    return prims


def test_stage2_object_sets_the_jobs_flags_before_its_first_case(monkeypatch):
    prims, seen = _fake_prims(monkeypatch), []
    monkeypatch.setattr(cv, "LabCase", lambda p, o, idx, seed, world: SimpleNamespace(env=p.env))
    monkeypatch.setattr(cv, "simulate_case", lambda lab, rec1, dec, rules, seeds: seen.append(
        (lab.env.replan_hold, lab.env.plan_without_hand_boxes)) or {"sims": [], "stop": None})
    rows1 = [dict(_row("commit_std", 0.9), cvar=_cv(0.1))]
    items = [(_s1("ycb__a", 0, rows=rows1), _dec_for(rows1)), (_s1("ycb__a", 1, rows=rows1), _dec_for(rows1))]
    s = {"bundle": "B", "seed": 11, "rules": ("R0", "R4"), "n_seeds": 1}
    cv.stage2_object(("ycb__a", items, dict(s, replan_hold=True, plan_without_hand_boxes=True)))
    assert seen == [(True, True), (True, True)]
    cv.stage2_object(("ycb__a", items, s))
    assert seen[2:] == [(False, False), (False, False)]


def test_oracle_object_sets_the_jobs_flags_before_its_first_case(monkeypatch):
    prims, seen = _fake_prims(monkeypatch), []

    def oracle_case(p, rec1, s, world, knob):
        seen.append((p.env.replan_hold, p.env.plan_without_hand_boxes))
        return {"sims": [], "skipped": None, "stop": None}

    monkeypatch.setattr(co, "oracle_case", oracle_case)
    s = {"bundle": "B", "seed": 11, "knob": 2.0}
    co.oracle_object(("ycb__a", [_s1("ycb__a", 0)], dict(s, replan_hold=True, plan_without_hand_boxes=True)))
    co.oracle_object(("ycb__a", [_s1("ycb__a", 0)], s))
    assert seen == [(True, True), (False, False)]


# ------------------------------------------------------------------ each attempt's re-plan reading

class _Env:
    """An executor whose rollout carries the re-plan reading when ``on``."""

    on = True

    def execute(self, *a, **k):
        return SimpleNamespace(replan=dict(READING)) if self.on else SimpleNamespace()


def _lab_case(env):
    lab = cv.LabCase.__new__(cv.LabCase)
    lab.prims, lab.destination = SimpleNamespace(env=env), "bin"
    return lab


def test_an_attempt_keeps_the_rollouts_re_plan_reading_and_restores_the_executor(monkeypatch):
    from rpent.skills.pick_place import tools

    def pick_and_place(**kw):
        kw["primitives"].env.execute()
        return SimpleNamespace(data={"outcome": "placed_stable"})

    monkeypatch.setattr(tools, "pick_and_place", pick_and_place)
    env = _Env()
    lab = _lab_case(env)
    assert lab.attempt({}, "commit_std") == {"outcome": "placed_stable", "replan": READING}
    assert "execute" not in vars(env)
    env.on = False                                               # the published executor: the payload as before
    assert lab.attempt({}, "commit_std") == {"outcome": "placed_stable"}
    assert "execute" not in vars(env)


def test_an_attempt_that_raises_restores_the_executor(monkeypatch):
    from rpent.skills.pick_place import tools

    def pick_and_place(**kw):
        kw["primitives"].env.execute()
        raise RuntimeError("boom")

    monkeypatch.setattr(tools, "pick_and_place", pick_and_place)
    env = _Env()
    assert _lab_case(env).attempt({}, "commit_std") == {"error": "RuntimeError: boom"}
    assert "execute" not in vars(env)


class _ReplanLab(_FakeLab):
    def attempt(self, store, row):
        return dict(super().attempt(store, row), replan={"d_mm": float(store["n"]), "raise_mm": 0.0})


def test_simulate_case_keeps_each_attempts_re_plan_reading_and_nothing_without_one():
    rows1 = [dict(_row("commit_std", 0.9, irr=0.0), cvar=_cv(0.1))]
    rows2 = [dict(_row("slow_lift", 0.95, irr=0.01), cvar=_cv(0.1))]
    dec = _dec_for(rows1)
    res = cv.simulate_case(_ReplanLab({"commit_std": "tipped"}, {"slow_lift": "placed_stable"}, rows2), {"rows": rows1}, dec,
                           ("R0", "R4"), [11000])
    s = res["sims"][0]
    assert s["replan"] == {"d_mm": 1.0, "raise_mm": 0.0} and s["second"]
    assert all(x["replan"] == {"d_mm": 2.0, "raise_mm": 0.0} for x in s["second"].values())
    plain = cv.simulate_case(_FakeLab({"commit_std": "tipped"}, {"slow_lift": "placed_stable"}, rows2), {"rows": rows1}, dec,
                             ("R0", "R4"), [11000])["sims"][0]
    assert set(plain) == {"seed_k", "reset_seed", "first", "label", "artifact", "error", "menu2", "second", "seconds"}
    assert all(set(x) == {"label", "artifact", "error"} for x in plain["second"].values())


# ------------------------------------------------------------------ the runs: flags, sim and lock in the config

def _stage2_world(tmp_path):
    rows1 = [dict(_row("commit_std", 0.9, irr=0.0), cvar=_cv(0.1))]
    _write_run(str(tmp_path / "a"), [_s1("ycb__a", 0, rows=rows1), _s1("ycb__t", 0, rows=rows1, side="test")])
    retry = tmp_path / "retry.json.gz"
    with gzip.open(retry, "wt") as f:
        json.dump({"rows": [{"attempts": [{"label": "tipped", "cost_s": 8.0}, {"label": "placed_stable", "cost_s": 8.0}]}]}, f)
    return ["stage2", "--stage1", str(tmp_path / "a"), "--retry", str(retry)]


def test_stage2_hands_the_flags_to_every_job_and_records_them_with_the_sim(tmp_path, monkeypatch):
    from robots.mujoco_lab import identify_fit

    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("RPENT_MUJOCO_SIM_DIR", "/sims/contacts")
    monkeypatch.setattr(identify_fit, "require_bundle_generator", lambda b: "v11")
    seen = []
    monkeypatch.setattr(cv, "stage2_object", lambda job: seen.append(job[2]) or {"records": [], "stop": None})
    argv = _stage2_world(tmp_path)
    cv.main(argv + ["--out", str(tmp_path / "on"), "--replan-hold", "--plan-without-hand-boxes"])
    assert len(seen) == 2 and all(s["replan_hold"] is True and s["plan_without_hand_boxes"] is True for s in seen)
    cfg = json.load(open(tmp_path / "on" / "run_config.json"))
    assert cfg["replan_hold"] is True and cfg["plan_without_hand_boxes"] is True and cfg["sim_dir"] == "/sims/contacts"
    cv.main(argv + ["--out", str(tmp_path / "off")])
    assert all(s["replan_hold"] is False and s["plan_without_hand_boxes"] is False for s in seen[2:])
    cfg = json.load(open(tmp_path / "off" / "run_config.json"))
    assert cfg["replan_hold"] is False and cfg["plan_without_hand_boxes"] is False and cfg["model_lock"] is None


def test_runs_that_differ_in_a_flag_or_the_sim_do_not_join(tmp_path):
    for k, (x, y) in {"replan_hold": (True, False), "plan_without_hand_boxes": (True, None), "sim_dir": ("/a", "/b")}.items():
        dirs = []
        for name, v in (("x", x), ("y", y)):
            d = tmp_path / k / name
            os.makedirs(d)
            json.dump({"bundle": "B", k: v}, open(d / "run_config.json", "w"))
            json.dump({"stop": None}, open(d / "run_metrics.json", "w"))
            for side in cv.SIDES:
                open(cv.records_path(str(d), side), "w").close()
            dirs.append(str(d))
        with pytest.raises(SystemExit, match=f"differ in {k}"):
            cv.load_stage2(dirs)


def test_oracle_run_hands_the_flags_to_every_job_and_records_them_with_the_sim(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("RPENT_MUJOCO_SIM_DIR", "/sims/contacts")
    _world(tmp_path)
    seen = []
    monkeypatch.setattr(co, "oracle_object", lambda job: seen.append(job[2]) or {"records": [], "stop": None})
    monkeypatch.setattr(co, "require_bundle_generator", lambda b: "v11")
    argv = ["run", "--stage1", str(tmp_path / "s1"), "--locked", str(tmp_path / "locked.json")]
    co.main(argv + ["--out", str(tmp_path / "on"), "--replan-hold", "--plan-without-hand-boxes"])
    assert seen and all(s["replan_hold"] is True and s["plan_without_hand_boxes"] is True for s in seen)
    cfg = json.load(open(tmp_path / "on" / "run_config.json"))
    assert cfg["replan_hold"] is True and cfg["plan_without_hand_boxes"] is True and cfg["sim_dir"] == "/sims/contacts"
    n = len(seen)
    co.main(argv + ["--out", str(tmp_path / "off")])
    assert all(s["replan_hold"] is False and s["plan_without_hand_boxes"] is False for s in seen[n:])


# ------------------------------------------------------------------ the model lock

def _bundle(tmp_path):
    b = tmp_path / "bundle"
    os.makedirs(b / "folds" / "fold_0")
    for rel, text in (("manifest.json", "{}"), ("folds/folds.json", "[0]"), ("folds/fold_0/manifest.json", "{\"f\": 0}"),
                      ("success_model.joblib", "model bytes")):
        (b / rel).write_text(text)
    return b


def _provenance(path, bundle, start, hashes=None):
    hashes = hashes or {rel: ml.sha256(os.path.join(bundle, rel)) for rel in ("manifest.json", "folds/folds.json",
                                                                               "folds/fold_0/manifest.json")}
    lines = ["code abc (detached run worktree), dirty files 0", f"bundle {bundle}:"]
    lines += [f"{h}  {rel}" for rel, h in hashes.items()]
    lines += ["env NUMPY_MADVISE_HUGEPAGE=0", "command python -P -u -m robots.mujoco_lab.curve stage2", f"start {start}",
              "end 2026-11-07 03:12:34"]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _past(b, when="2026-11-05 13:47:32"):
    t = time.mktime(time.strptime(when, "%Y-%m-%d %H:%M:%S"))
    for root, _, files in os.walk(b):
        for f in files:
            os.utime(os.path.join(root, f), (t, t))


def test_the_lock_holds_every_bundle_file_and_the_provenances_it_was_checked_against(tmp_path):
    b = _bundle(tmp_path)
    _past(b)
    p1 = _provenance(tmp_path / "p1.txt", b, "2026-11-07 02:35:01")
    p2 = _provenance(tmp_path / "p2.txt", b, "2026-11-07 01:02:52")
    lock = ml.write(str(b), [p1, p2], str(tmp_path / "lock.json"))
    assert lock == json.load(open(tmp_path / "lock.json"))
    assert lock["bundle"] == os.path.realpath(b) and lock["n_files"] == 4
    assert lock["files"] == {rel: ml.sha256(os.path.join(b, rel)) for rel in
                             ("folds/fold_0/manifest.json", "folds/folds.json", "manifest.json", "success_model.joblib")}
    assert lock["earliest_start"] == "2026-11-07 01:02:52"
    assert lock["provenance_sha256"] == {p1: ml.sha256(p1), p2: ml.sha256(p2)}
    assert ml.check(str(b), str(tmp_path / "lock.json")) == lock
    with pytest.raises(SystemExit, match="exists"):
        ml.write(str(b), [p1], str(tmp_path / "lock.json"))


def test_the_lock_refuses_a_manifest_that_is_not_the_published_runs(tmp_path):
    b = _bundle(tmp_path)
    _past(b)
    p = _provenance(tmp_path / "p.txt", b, "2026-11-07 02:35:01", {"manifest.json": "0" * 64})
    with pytest.raises(SystemExit, match="manifest.json"):
        ml.write(str(b), [p], str(tmp_path / "lock.json"))
    assert not os.path.exists(tmp_path / "lock.json")


def test_the_lock_refuses_a_file_changed_after_the_earliest_published_start(tmp_path):
    b = _bundle(tmp_path)
    _past(b)
    t = time.mktime(time.strptime("2026-11-07 01:30:00", "%Y-%m-%d %H:%M:%S"))
    os.utime(b / "success_model.joblib", (t, t))                   # after Stage 1's start, before Stage 2's
    p1 = _provenance(tmp_path / "p1.txt", b, "2026-11-07 02:35:01")
    assert ml.write(str(b), [p1], str(tmp_path / "ok.json"))["n_files"] == 4
    p2 = _provenance(tmp_path / "p2.txt", b, "2026-11-07 01:02:52")
    with pytest.raises(SystemExit, match="success_model.joblib"):
        ml.write(str(b), [p1, p2], str(tmp_path / "lock.json"))


def test_the_lock_refuses_provenances_of_another_bundle_or_without_a_start(tmp_path):
    b = _bundle(tmp_path)
    _past(b)
    p = _provenance(tmp_path / "p.txt", tmp_path / "elsewhere", "2026-11-07 02:35:01",
                    {"manifest.json": ml.sha256(os.path.join(b, "manifest.json"))})
    with pytest.raises(SystemExit, match="bundle"):
        ml.write(str(b), [p], str(tmp_path / "lock.json"))
    q = tmp_path / "q.txt"
    q.write_text(f"bundle {b}:\n{ml.sha256(os.path.join(b, 'manifest.json'))}  manifest.json\n")
    with pytest.raises(SystemExit, match="start"):
        ml.write(str(b), [str(q)], str(tmp_path / "lock.json"))


@pytest.mark.parametrize("change", ["byte", "added", "removed"])
def test_the_check_names_every_file_that_differs_from_the_lock(tmp_path, change):
    b = _bundle(tmp_path)
    _past(b)
    ml.write(str(b), [_provenance(tmp_path / "p.txt", b, "2026-11-07 02:35:01")], str(tmp_path / "lock.json"))
    if change == "byte":
        (b / "success_model.joblib").write_text("model byteS")
    elif change == "added":
        (b / "folds" / "fold_0" / "extra.joblib").write_text("x")
    else:
        os.remove(b / "folds" / "folds.json")
    name = {"byte": "success_model.joblib", "added": "folds/fold_0/extra.joblib", "removed": "folds/folds.json"}[change]
    with pytest.raises(SystemExit, match=name):
        ml.check(str(b), str(tmp_path / "lock.json"))


def test_the_check_refuses_a_lock_of_another_bundle(tmp_path):
    b = _bundle(tmp_path)
    _past(b)
    ml.write(str(b), [_provenance(tmp_path / "p.txt", b, "2026-11-07 02:35:01")], str(tmp_path / "lock.json"))
    other = tmp_path / "other"
    os.rename(b, other)
    with pytest.raises(SystemExit, match="bundle"):
        ml.check(str(other), str(tmp_path / "lock.json"))


def test_stage2_checks_the_model_lock_before_any_job_and_records_it(tmp_path, monkeypatch):
    from robots.mujoco_lab import identify_fit

    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(identify_fit, "require_bundle_generator", lambda b: "v11")
    b = _bundle(tmp_path)
    _past(b)
    lock = str(tmp_path / "lock.json")
    ml.write(str(b), [_provenance(tmp_path / "p.txt", b, "2026-11-07 02:35:01")], lock)
    argv = _stage2_world(tmp_path)
    cfg = json.load(open(tmp_path / "a" / "run_config.json"))
    json.dump(dict(cfg, bundle=str(b)), open(tmp_path / "a" / "run_config.json", "w"))
    seen = []
    monkeypatch.setattr(cv, "stage2_object", lambda job: seen.append(job) or {"records": [], "stop": None})
    cv.main(argv + ["--out", str(tmp_path / "ok"), "--model-lock", lock])
    assert len(seen) == 2
    cfg = json.load(open(tmp_path / "ok" / "run_config.json"))
    assert cfg["model_lock"] == lock and cfg["model_lock_sha256"] == ml.sha256(lock)
    (b / "success_model.joblib").write_text("retrained")
    with pytest.raises(SystemExit, match="success_model.joblib"):
        cv.main(argv + ["--out", str(tmp_path / "bad"), "--model-lock", lock])
    assert len(seen) == 2 and not os.path.exists(tmp_path / "bad" / "run_config.json")


def test_oracle_run_checks_the_model_lock_before_any_job(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    _world(tmp_path)
    b = _bundle(tmp_path)
    _past(b)
    lock = str(tmp_path / "lock.json")
    ml.write(str(b), [_provenance(tmp_path / "p.txt", b, "2026-11-07 02:35:01")], lock)
    cfg = json.load(open(tmp_path / "s1" / "run_config.json"))
    json.dump(dict(cfg, bundle=str(b)), open(tmp_path / "s1" / "run_config.json", "w"))
    seen = []
    monkeypatch.setattr(co, "oracle_object", lambda job: seen.append(job) or {"records": [], "stop": None})
    monkeypatch.setattr(co, "require_bundle_generator", lambda b: "v11")
    argv = ["run", "--stage1", str(tmp_path / "s1"), "--locked", str(tmp_path / "locked.json"), "--model-lock", lock]
    co.main(argv + ["--out", str(tmp_path / "ok")])
    assert seen and json.load(open(tmp_path / "ok" / "run_config.json"))["model_lock_sha256"] == ml.sha256(lock)
    n = len(seen)
    (b / "folds" / "fold_0" / "manifest.json").write_text("{\"f\": 1}")
    with pytest.raises(SystemExit, match="folds/fold_0/manifest.json"):
        co.main(argv + ["--out", str(tmp_path / "bad")])
    assert len(seen) == n


@pytest.mark.parametrize("key,values", [("replan_hold", (True, False)), ("plan_without_hand_boxes", (True, False)),
                                        ("sim_dir", ("/sims/contacts", "/sims/published"))])
def test_oracle_runs_that_differ_in_a_flag_or_the_sim_do_not_join(tmp_path, monkeypatch, key, values):
    from tests.unit_tests.robots.mujoco_lab.test_curve_oracle import _run_oracle, _summary

    monkeypatch.setenv("WANDB_MODE", "disabled")
    _world(tmp_path)
    cases = [f"o{i}:{j}" for i in range(8) for j in range(2)]
    for name, v, part in (("o1", values[0], cases[:8]), ("o2", values[1], cases[8:])):
        _run_oracle(tmp_path, monkeypatch, tmp_path / name, cases=",".join(part))
        cfg = json.load(open(tmp_path / name / "run_config.json"))
        json.dump(dict(cfg, **{key: v}), open(tmp_path / name / "run_config.json", "w"))
    with pytest.raises(SystemExit, match=f"differ in {key}"):
        _summary(tmp_path, tmp_path / "sum", runs=f"{tmp_path / 'o1'},{tmp_path / 'o2'}")
