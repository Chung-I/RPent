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


"""The v12 checks on the real bundles (daily-logs 2026-11-05-v12-masscom-spread, the v12b rerun): m2b2_v12b's own
identify rows and its attempt's joint term reach the stages, and m2b2_v12's records reproduce on this code. Each test
reads bundles and records outside this checkout and skips without them: RPENT_V12B_BUNDLE (m2b2_v12b), RPENT_V12_BUNDLE
(m2b2_v12), RPENT_V12_EC_RECORDS (the v12 class run's records.jsonl) and RPENT_V12_IDENTIFY_ROWS (its option-A rows)."""

import json
import os

import pytest

V12B = os.environ.get("RPENT_V12B_BUNDLE", "")
V12 = os.environ.get("RPENT_V12_BUNDLE", "")
V12_RECORDS = os.environ.get("RPENT_V12_EC_RECORDS", "")
V12_IDENTIFY = os.environ.get("RPENT_V12_IDENTIFY_ROWS", "")

need_v12b = pytest.mark.skipif(not os.path.isdir(V12B), reason="RPENT_V12B_BUNDLE is not a bundle")
need_v12 = pytest.mark.skipif(not (os.path.isdir(V12) and os.path.isfile(V12_RECORDS) and os.path.isfile(V12_IDENTIFY)),
                              reason="RPENT_V12_BUNDLE, RPENT_V12_EC_RECORDS or RPENT_V12_IDENTIFY_ROWS missing")


@need_v12b
def test_v12bs_own_identify_rows_reach_every_weigh_case():
    """The bundle's scene-D rows (96 objects x 24 draws, the 24 pomegranate error rows without a scene tag left out) are
    its identify rows, and each scene-B weigh case finds the row of its object and draw."""
    from robots.mujoco_lab.estimation_check import probe_rows, stage_cases
    from robots.mujoco_lab.identify_fit import MISSING, bundle_identify_rows, identify_cases

    rows = bundle_identify_rows(V12B)
    assert len(rows) == 2280
    cases = identify_cases(stage_cases(probe_rows(V12B), "weigh"), rows)
    no_row = [c["key"] for (c,) in cases if c["kind"] == MISSING and c.get("missing_reason") is None]
    assert len(cases) > 1000 and no_row == []
    assert sum(c["kind"] == "measured" for (c,) in cases) > 0


def _joint_rows(bundle):
    from robots.mujoco_lab.estimation_check import attempt_rows

    rows = [r for r in attempt_rows(bundle) if r["readouts"].get("joint") and (r["readouts"].get("fz_hold_n") or 0) > 0]
    assert len(rows) >= 3
    return rows[:3]


@need_v12b
@pytest.mark.skipif(not os.path.isdir(V12), reason="RPENT_V12_BUNDLE is not a bundle")
def test_v12bs_attempt_update_reads_the_joint_fit_where_v12s_read_the_hold(monkeypatch):
    """The same rows and code: m2b2_v12b's readout model uses the attempt's Real2Sim fit and drops the hold force (the
    same joint torques), m2b2_v12's uses the hold force."""
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    from robots.mujoco_lab.estimation_check import run_attempt
    from robots.mujoco_lab.robot_spec import load_bundle

    rows = _joint_rows(V12B)
    for bundle, has, lacks in ((V12B, "joint", "hold"), (V12, "hold", "joint")):
        for rec in run_attempt(rows, load_bundle(bundle), 200, 0):
            used = rec.get("terms", {}).get("used", [])
            assert has in used and lacks not in used, (bundle, rec["key"], used)


@need_v12
def test_the_v12_class_records_reproduce_on_this_code(monkeypatch):
    """m2b2_v12 is unchanged: fold 0's first three attempt and identify cases (option A's rows, gate R1's likelihood)
    give the v12 class run's records (--crossfit --n-particles 20000 --seed 0)."""
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    from robots.mujoco_lab.estimation_check import (attempt_rows, probe_rows, run_attempt, run_probes, stage_cases,
                                                    stream_index)
    from robots.mujoco_lab.identify_fit import identify_cases, identify_model, identify_source
    from robots.mujoco_lab.robot_spec import load_bundle
    from rpent.skills.pick_place.evaluate import rows_of

    folds = json.load(open(os.path.join(V12, "folds", "folds.json")))
    objs = sorted(o for o, f in folds.items() if f == 0)
    B = load_bundle(os.path.join(V12, "folds", "fold_0"))
    got = run_attempt(rows_of(attempt_rows(V12), objs)[:3], B, 20000, [0, 0, stream_index("attempt")])
    rows, src = identify_source(V12, V12_IDENTIFY)
    assert src == V12_IDENTIFY
    Bi = {**B, "probe_models": {**B["probe_models"], "identify": identify_model(B)}}
    cases = identify_cases(stage_cases(rows_of(probe_rows(V12), objs), "weigh"), rows)[:3]
    got += run_probes(cases, "identify", Bi, 20000, [0, 0, stream_index("identify")])
    recs = [r for r in map(json.loads, open(V12_RECORDS)) if r.get("fold") == 0]
    want = [r for r in recs if r["stage"] == "attempt"][:3] + [r for r in recs if r["stage"] == "identify"][:3]
    assert [g["key"] for g in got] == [w["key"] for w in want]   # by order: corpus and fixtures rows share keys
    for g, w in zip(got, want):
        for part in ("prior", "post"):
            for prop in ("mass", "mu", "com"):
                for k in ("mean", "lo", "hi"):
                    assert g[part][prop][k] == pytest.approx(w[part][prop][k], rel=1e-9, abs=1e-12), (g["key"], part, prop, k)
        assert g["ess"] == pytest.approx(w["ess"], rel=1e-9)
