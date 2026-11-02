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


"""The mass and CoM estimation check: per case the posterior mean, the 80 % interval and its hit; per stage the
error's median and IQR, the width, the coverage, each with a 95 % object-bootstrap interval."""

import os

import numpy as np
import pytest

from rpent.skills.pick_place.belief import ParticleBelief


def _belief(mass, mu, frac, w=None):
    parts = [{"mass": m, "mu": u, "com_obj": [0.0, 0.0, 0.0], "com_frac": c} for m, u, c in zip(mass, mu, frac)]
    return ParticleBelief(parts, np.ones(len(parts)) if w is None else np.asarray(w, dtype=float))


def test_estimates_give_the_mean_the_interval_the_hit_and_the_errors():
    from robots.mujoco_lab.estimation_check import estimates

    b = _belief([1.0, 2.0, 3.0, 4.0], [0.2, 0.4, 0.6, 0.8], [0.4, 0.5, 0.5, 0.6])
    e = estimates(b, {"mass": 2.0, "mu": 0.5, "com": 0.45}, length=0.2)
    assert e["mass"]["mean"] == pytest.approx(2.5)
    assert e["mass"]["err"] == pytest.approx(0.25)                      # |2.5 - 2| / 2
    assert (e["mass"]["lo"], e["mass"]["hi"], e["mass"]["hit"]) == (1.0, 4.0, True)
    assert e["mass"]["width"] == pytest.approx(3.0 / 2.5)               # relative to the posterior mean
    assert e["mu"]["err"] == pytest.approx(0.0) and e["mu"]["width"] == pytest.approx(0.6)
    assert e["com"]["err"] == pytest.approx(0.05)
    assert e["com"]["err_cm"] == pytest.approx(0.05 * 0.2 * 100)
    assert e["com"]["width"] == pytest.approx(0.2) and e["com"]["width_cm"] == pytest.approx(0.2 * 0.2 * 100)


def test_estimates_use_the_weights_and_report_a_miss():
    from robots.mujoco_lab.estimation_check import estimates

    b = _belief([1.0, 3.0], [0.2, 0.6], [0.3, 0.7], w=[3, 1])
    e = estimates(b, {"mass": 5.0, "mu": 0.2, "com": 0.3}, length=0.1)
    assert e["mass"]["mean"] == pytest.approx(1.5) and e["com"]["mean"] == pytest.approx(0.4)
    assert e["mass"]["hit"] is False and e["mass"]["err"] == pytest.approx(0.7)


def test_stage_cases_take_the_push_alone_and_the_weigh_alone():
    from robots.mujoco_lab.estimation_check import stage_cases

    rows = [{"key": "o__d0__Apush", "object": "o", "draw": 0}, {"key": "o__d0__Aweigh", "object": "o", "draw": 0},
            {"key": "o__d0__Bweigh", "object": "o", "draw": 0}, {"key": "o__d1__Csqueeze", "object": "o", "draw": 1},
            {"key": "p__d3__Apush", "object": "p", "draw": 3}]
    assert [[r["key"] for r in c] for c in stage_cases(rows, "push")] == [["o__d0__Apush"], ["p__d3__Apush"]]
    assert [[r["key"] for r in c] for c in stage_cases(rows, "weigh")] == [["o__d0__Bweigh"]]
    with pytest.raises(KeyError):
        stage_cases(rows, "attempt")


def test_cell_reports_median_iqr_and_an_object_bootstrap_interval():
    from robots.mujoco_lab.estimation_check import cell

    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    objects = ["a", "a", "b", "b", "c", "c", "d", "d"]
    c = cell(values, objects, np.random.default_rng(0), n_boot=500)
    assert (c["n"], c["n_objects"]) == (8, 4)
    assert c["median"] == pytest.approx(4.5)
    assert (c["q25"], c["q75"]) == (pytest.approx(2.75), pytest.approx(6.25))
    assert c["ci"][0] < c["median"] < c["ci"][1] and c["ci"][0] >= 1.0 and c["ci"][1] <= 8.0
    same = cell([2.0] * 6, ["a", "a", "b", "b", "c", "c"], np.random.default_rng(0), n_boot=200)
    assert same["ci"] == [2.0, 2.0]


def test_coverage_cell_reports_the_rate_and_an_object_bootstrap_interval():
    from robots.mujoco_lab.estimation_check import coverage_cell

    hits = [True, True, False, True, True, True, False, True]
    c = coverage_cell(hits, ["a", "a", "b", "b", "c", "c", "d", "d"], np.random.default_rng(0), n_boot=500)
    assert c["coverage"] == pytest.approx(0.75) and c["n"] == 8 and c["n_objects"] == 4
    assert c["ci"][0] <= 0.75 <= c["ci"][1]


def test_summarize_takes_the_prior_row_from_the_attempt_cases_and_the_stages_from_their_posteriors():
    from robots.mujoco_lab.estimation_check import summarize

    def est(err, hit):
        return {"mass": {"err": err, "width": 1.0, "hit": hit}, "mu": {"err": err, "width": 0.5, "hit": hit},
                "com": {"err": err, "err_cm": 10 * err, "width": 0.2, "width_cm": 2.0, "hit": hit}}

    recs = [{"object": o, "stage": "attempt", "prior": est(1.0, True), "post": est(0.5, False)} for o in "abcd"]
    recs += [{"object": o, "stage": "weigh", "prior": est(1.0, True), "post": est(0.1, True)} for o in "abcd"]
    s = summarize(recs, n_boot=100, seed=0)
    assert s["prior"]["mass"]["err"]["median"] == 1.0 and s["prior"]["mass"]["coverage"]["coverage"] == 1.0
    assert s["attempt"]["mass"]["err"]["median"] == 0.5 and s["attempt"]["mass"]["coverage"]["coverage"] == 0.0
    assert s["weigh"]["com"]["err_cm"]["median"] == pytest.approx(1.0)
    assert "push" not in s and s["weigh"]["mass"]["n"] == 4


def test_jobs_for_crossfit_gives_each_fold_its_bundle_and_objects(tmp_path):
    import json

    from robots.mujoco_lab.estimation_check import jobs_for

    (tmp_path / "folds").mkdir()
    json.dump({"a": 0, "b": 1, "c": 0, "d": 1}, open(tmp_path / "folds" / "folds.json", "w"))
    specs = jobs_for(str(tmp_path), True, None, ["attempt"], 100, 3)
    assert [(s["fold"], s["objects"], os.path.basename(s["bundle"])) for s in specs] == [(0, ["a", "c"], "fold_0"), (1, ["b", "d"], "fold_1")]
    assert all(s["source"] == str(tmp_path) for s in specs)
    one = jobs_for(str(tmp_path), False, ["a"], ["attempt"], 100, 3)
    assert one == [{"bundle": str(tmp_path), "source": str(tmp_path), "fold": -1, "objects": ["a"], "stages": ["attempt"], "n": 100, "seed": 3}]


M2B2 = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "robots", "mujoco_lab", "data", "m2b2")


@pytest.mark.skipif(not os.path.exists(os.path.join(M2B2, "success_model.joblib")), reason="the m2b2 bundle's models are not here")
def test_run_attempt_and_run_probes_repeat_the_posterior_checks_hits():
    """Same rows, same seed: the attempt stage's and the push-then-weigh stage's hits equal posterior_check's at temper 1."""
    import json

    from robots.mujoco_lab import posterior_check as pc
    from robots.mujoco_lab.estimation_check import attempt_rows, probe_rows, run_attempt, run_probes
    from robots.mujoco_lab.robot_spec import load_bundle
    from rpent.skills.pick_place.evaluate import rows_of

    B = load_bundle(M2B2)
    val = json.load(open(os.path.join(M2B2, json.load(open(os.path.join(M2B2, "manifest.json")))["metrics"])))["split"]["val"]
    rows = rows_of(attempt_rows(M2B2), val)[:3]
    ours = run_attempt(rows, B, 300, 0)
    theirs = pc.check_rows(rows, B, [1.0], 300, 0)
    assert [{k: r["post"][k]["hit"] for k in ("mass", "mu", "com")} for r in ours] == [c["post"][1.0] for c in theirs]
    assert [{k: r["prior"][k]["hit"] for k in ("mass", "mu", "com")} for r in ours] == [c["prior"] for c in theirs]
    cases = pc.probe_cases(rows_of(probe_rows(M2B2), val), "A")[:3]
    ours = run_probes(cases, "push_weigh", B, 300, 0)
    theirs = pc.check_probe_cases(cases, B, [1.0], 300, 0)
    assert [{k: r["post"][k]["hit"] for k in ("mass", "mu", "com")} for r in ours] == [c["post"][1.0] for c in theirs]


def test_downstream_scores_the_prior_predictive_against_the_labels():
    from robots.mujoco_lab.estimation_check import downstream
    from rpent.skills.pick_place.protocol import OUTCOMES

    def p(**kw):
        v = {k: 0.0 for k in OUTCOMES}
        v.update(kw)
        return v

    recs = [{"object": "a", "stage": "attempt", "label": "placed_stable", "p_prior": p(placed_stable=1.0)},
            {"object": "b", "stage": "attempt", "label": "rolled_off", "p_prior": p(placed_stable=0.5, rolled_off=0.5)},
            {"object": "c", "stage": "weigh", "label": "light", "p_prior": None}]
    d = downstream(recs, n_boot=50, seed=0)
    assert d["n"] == 2 and d["brier"]["value"] == pytest.approx((0.0 + 0.5) / 2)      # b: 0.25 + 0.25
    assert d["ece_irreversible"]["value"] == pytest.approx(0.25)                       # bins: p 0 vs y 0; p 0.5 vs y 1
    assert len(d["brier"]["ci"]) == 2


def test_case_truths_and_tuning_objects_for_a_fold():
    from robots.mujoco_lab.estimation_check import case_truths, tuning_objects

    rows = [{"object": "a", "key": "a__0", "pose": {"pivot": {"centre": [0, 0, 0], "axis": [1, 0, 0], "lo": 0.0, "L": 1.0},
                                                     "com_obj_nominal": [0.25, 0, 0], "nominal_mass": 2.0},
             "theta": {"mass_scale": 1.5, "mu": 0.3, "com_s": None}, "com_end": 1}]
    t = case_truths(rows, None)
    assert t == {"a": [(pytest.approx(3.0), pytest.approx(0.25))]}
    assert tuning_objects({"a": 0, "b": 1, "c": 1, "d": 2}, 1) == ["a", "d"]


def test_job_prior_of_a_class_spec_is_the_bundle_prior():
    """A class-prior spec (jobs_for gives it no "prior" key) keeps the bundle's prior: no table, no shape prior."""
    from robots.mujoco_lab.estimation_check import job_prior

    spec = {"bundle": "b", "source": "b", "fold": 0, "objects": ["a"], "stages": ["attempt"], "n": 10, "seed": 0}
    assert job_prior(spec, {}) == (None, None)


def _seed_spy(monkeypatch):
    from robots.mujoco_lab import estimation_check as ec

    seeds = []
    monkeypatch.setattr(ec, "load_bundle", lambda b: {})
    monkeypatch.setattr(ec, "attempt_rows", lambda s: [])
    monkeypatch.setattr(ec, "probe_rows", lambda s: [])
    monkeypatch.setattr(ec, "run_attempt", lambda rows, B, n, seed, prior=None: seeds.append(seed) or [])
    monkeypatch.setattr(ec, "run_probes", lambda cases, stage, B, n, seed, prior=None: seeds.append(seed) or [])
    return ec, seeds


def test_a_single_bundle_job_seeds_every_stage_with_the_run_seed(monkeypatch):
    """Without folds (fold -1), every stage takes the run's seed, as posterior_check's runs do."""
    ec, seeds = _seed_spy(monkeypatch)
    ec.job({"bundle": "b", "source": "b", "fold": -1, "objects": ["a"], "stages": ["attempt", "weigh", "push"], "n": 10, "seed": 3})
    assert seeds == [3, 3, 3]


def test_a_crossfit_job_seeds_each_stage_by_seed_fold_and_stage(monkeypatch):
    ec, seeds = _seed_spy(monkeypatch)
    ec.job({"bundle": "b", "source": "b", "fold": 2, "objects": ["a"], "stages": ["attempt", "weigh", "push"], "n": 10, "seed": 3})
    assert seeds == [[3, 2, 0], [3, 2, 1], [3, 2, 2]]


def test_summary_reports_the_downstream_scores(tmp_path, monkeypatch):
    """summary.json and summary.md carry the prior predictive's Brier and ECE next to the accuracy table."""
    import argparse
    import json

    from robots.mujoco_lab import estimation_check as ec
    from rpent.skills.pick_place.protocol import OUTCOMES

    p = {k: 0.0 for k in OUTCOMES}
    p["placed_stable"] = 1.0
    recs = [{"object": o, "stage": "attempt", "label": "placed_stable", "p_prior": p} for o in ("a", "b")]
    (tmp_path / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    monkeypatch.setattr(ec, "summarize", lambda recs, n_boot, seed: {})
    out = ec.summary(argparse.Namespace(dir=str(tmp_path), n_boot=20, seed=0, wandb=False))
    saved = json.load(open(tmp_path / "summary.json"))
    assert saved["downstream"]["n"] == 2 and saved["downstream"]["brier"]["value"] == 0.0
    assert out["downstream"] == saved["downstream"]
    assert "Brier" in (tmp_path / "summary.md").read_text()
