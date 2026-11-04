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

import json
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


def test_the_run_config_records_the_generator(monkeypatch):
    """masscom-v11: the provenance in both run configs (estimation_check and first_menus share _revision) names the lab's
    generator, so a v11 run says which corpus version its truth came from."""
    from robots.mujoco_lab.estimation_check import _revision

    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    assert _revision()["generator"] == "v11"
    monkeypatch.delenv("RPENT_LAB_GENERATOR")
    assert _revision()["generator"] == "v10"


def test_jobs_for_carries_the_mass_only_swap_to_every_shape_prior_spec(tmp_path):
    import json

    from robots.mujoco_lab.estimation_check import jobs_for

    (tmp_path / "folds").mkdir()
    json.dump({"a": 0, "b": 1}, open(tmp_path / "folds" / "folds.json", "w"))
    specs = jobs_for(str(tmp_path), True, None, ["attempt"], 100, 3, prior="real", scenes="s.json", mass_only=True)
    assert [(s["prior"], s["mass_only"]) for s in specs] == [("real", True), ("real", True)]
    assert "mass_only" not in jobs_for(str(tmp_path), True, None, ["attempt"], 100, 3)[0]    # the class prior swaps nothing


def test_job_prior_real_mass_only_takes_the_real_masses_and_keeps_the_class_com(tmp_path, monkeypatch):
    import json

    import robots.mujoco_lab.estimation_check as ec
    from robots.mujoco_lab.masses import load_masses
    from rpent.skills.pick_place.belief import ParticleBelief

    real = load_masses()["objects"]
    o, t1, t2 = sorted(real)[:3]
    scenes = tmp_path / "scenes.json"
    json.dump({k: {"flat": True} for k in (o, t1, t2)}, open(scenes, "w"))
    monkeypatch.setattr(ec, "attempt_rows", lambda src: [])
    monkeypatch.setattr(ec, "probe_rows", lambda src: [])
    monkeypatch.setattr(ec, "case_truths", lambda rows, classes: {t1: [(1.0, 0.5)], t2: [(2.0, 0.4)]})

    def base(pose, n, rng):
        return ParticleBelief([{"mass": 1.0, "mu": 0.5, "com_obj": [0.01, 0.0, 0.0], "com_frac": 0.75} for _ in range(n)], np.ones(n))

    spec = {"prior": "real", "objects": [o], "tuning": [t1, t2], "scenes": str(scenes), "source": "x", "fold": 0, "mass_only": True}
    prior, info = ec.job_prior(spec, {"prior": base})
    assert info["table"][o]["mass"] == real[o]["mass_kg"]
    a = prior({"object_id": o, "pivot": {"centre": [0, 0, 0], "axis": [1, 0, 0], "lo": -0.1, "L": 0.2}}, 20,
              np.random.default_rng(0)).arrays()
    assert np.allclose(a["com_frac"], 0.75) and np.allclose(a["com_obj"], [[0.01, 0.0, 0.0]] * 20)


def test_stream_keys_are_fixed_by_stage_name():
    """Each stage's stream key is fixed by its name, the 3 Nov places (attempt 0, weigh 1, push 2), in any order of the
    job's stages; attempt_fit takes the attempt's key and identify the weigh's, so each pair scores the same particles,
    and psuccess_spread's keys are the same."""
    from robots.mujoco_lab.estimation_check import stream_index
    from robots.mujoco_lab.psuccess_spread import STAGE_INDEX

    want = {"attempt": 0, "attempt_fit": 0, "weigh": 1, "identify": 1, "push": 2}
    for st in (["attempt", "weigh", "attempt_fit", "identify"], ["attempt", "attempt_fit", "weigh", "identify"],
               ["identify", "push", "weigh", "attempt_fit", "attempt"], ["attempt", "weigh", "push"]):
        assert {s: stream_index(s) for s in st} == {s: want[s] for s in st}
    assert all(stream_index(s) == k for s, k in STAGE_INDEX.items())


def test_twin_stages_draw_their_twins_particles_and_identify_reads_the_fit(monkeypatch):
    from robots.mujoco_lab.identify_fit import IdentifyFit

    ec, seeds = _seed_spy(monkeypatch)
    models = []
    monkeypatch.setattr(ec, "load_bundle", lambda b: {"probe_models": {"weigh": "w"}})
    monkeypatch.setattr(ec, "run_attempt_fit", lambda rows, B, n, seed, prior=None: seeds.append(("fit", seed)) or [])
    monkeypatch.setattr(ec, "run_probes", lambda cases, stage, B, n, seed, prior=None:
                        seeds.append(seed) or models.append((stage, B["probe_models"])) or [])
    sources = []
    monkeypatch.setattr(ec, "identify_source", lambda b, e: sources.append((b, e)) or ([], e))
    ec.job({"bundle": "b", "source": "b", "fold": 2, "objects": ["a"], "stages": ["attempt", "weigh", "attempt_fit", "identify"],
            "n": 10, "seed": 3, "identify": "rows.jsonl"})
    assert seeds == [[3, 2, 0], [3, 2, 1], ("fit", [3, 2, 0]), [3, 2, 1]]
    assert sources == [("b", "rows.jsonl")]
    assert models[0] == ("weigh", {"weigh": "w"})
    assert models[1][0] == "identify" and models[1][1]["weigh"] == "w" and isinstance(models[1][1]["identify"], IdentifyFit)
    mine = object()                                      # a bundle with its own identify model (m2b2_v12b on) keeps it
    monkeypatch.setattr(ec, "load_bundle", lambda b: {"probe_models": {"weigh": "w", "identify": mine}})
    ec.job({"bundle": "b", "source": "b", "fold": 2, "objects": ["a"], "stages": ["identify"], "n": 10, "seed": 3})
    assert models[-1][1]["identify"] is mine and sources[-1] == ("b", None)


def test_the_identify_stage_needs_the_identify_rows_and_every_job_carries_them(tmp_path):
    import json

    from robots.mujoco_lab.estimation_check import jobs_for

    (tmp_path / "pc.jsonl").write_text(json.dumps({"key": "a__v11s0__d0__Bweigh"}) + "\n")
    (tmp_path / "manifest.json").write_text(json.dumps({"probe_corpus": str(tmp_path / "pc.jsonl")}))
    with pytest.raises(ValueError, match="identify"):
        jobs_for(str(tmp_path), False, ["a"], ["weigh", "identify"], 10, 0)
    (tmp_path / "folds").mkdir()
    json.dump({"a": 0, "b": 1}, open(tmp_path / "folds" / "folds.json", "w"))
    specs = jobs_for(str(tmp_path), True, None, ["weigh", "identify"], 10, 0, identify="rows.jsonl")
    assert [s["identify"] for s in specs] == ["rows.jsonl", "rows.jsonl"]
    assert jobs_for(str(tmp_path), False, ["a"], ["identify"], 10, 0, identify="rows.jsonl")[0]["identify"] == "rows.jsonl"
    assert "identify" not in jobs_for(str(tmp_path), False, ["a"], ["weigh"], 10, 0)[0]


def _attempt_row(obj, fit=None):
    return {"object": obj, "key": f"{obj}__0", "label": "placed_stable", "readouts": {"joint": fit} if fit else {"hold": {}},
            "pose": {"pivot": {"centre": [0, 0, 0], "axis": [1, 0, 0], "lo": 0.0, "L": 1.0}, "com_obj_nominal": [0.5, 0, 0],
                     "nominal_mass": 1.0},
            "theta": {"mass_scale": 1.0, "mu": 0.5, "com_s": None}, "com_end": 1}


def _uniform_prior(pose, n, rng):
    m, c = rng.uniform(0.5, 1.5, n), rng.uniform(0.0, 1.0, n)
    parts = [{"mass": x, "mu": 0.5, "com_obj": [f, 0.0, 0.0], "com_frac": f} for x, f in zip(m, c)]
    return ParticleBelief(parts, np.ones(n))


def test_the_attempt_fit_stage_draws_the_attempts_particles_and_a_row_without_a_fit_keeps_its_prior():
    from robots.mujoco_lab.estimation_check import estimates, run_attempt_fit

    fit = {"obj_l7": {"p": [0.0, 0.0, 0.0], "R": np.eye(3).ravel().tolist()}, "phi": [1.0, 0.5, 0.0, 0.0] + [0.0] * 6,
           "info": np.diag([400.0, 400.0, 400.0, 400.0]).tolist(), "floor": [0.001, 0.0005]}
    rows = [_attempt_row("a"), _attempt_row("b", fit)]
    recs = run_attempt_fit(rows, {"prior": _uniform_prior}, 400, [3, 1, 0])
    rng = np.random.default_rng([3, 1, 0])                 # run_attempt's loop: one prior draw per row, in order
    for r, rec in zip(rows, recs):
        b = _uniform_prior(r["pose"], 400, rng)
        from robots.mujoco_lab.posterior_check import generator_truth

        assert rec["prior"] == estimates(b, generator_truth(r), 1.0)
    assert [r["has_fit"] for r in recs] == [False, True] and [r["stage"] for r in recs] == ["attempt_fit"] * 2
    assert recs[0]["post"] == recs[0]["prior"]
    for k in ("mass", "com"):                              # the fit (sd 0.05 on m and m x c) narrows both
        assert recs[1]["post"][k]["width"] < 0.5 * recs[1]["prior"][k]["width"]


def _wrow(draw, **kw):
    return {"key": f"a__v11s0__d{draw}__Bweigh", "object": "a", "draw": draw, "seed": 0, "probe": "weigh", "kind": "hold",
            "reading": {"fz_hold_n": 3.0}, "force": 4.0, "s": 0.5, "pose": {"tag": "weigh"}, "base": {"b": 1.0},
            "theta": {"mass_scale": 1.0}, "com_end": 1, "fill": 0.0, "theta_frag": None, **kw}


def test_a_cases_likelihood_skips_a_missing_row_and_reads_an_identify_row_at_its_own_pinch(monkeypatch):
    from robots.mujoco_lab import estimation_check as ec
    from robots.mujoco_lab.identify_fit import identify_cases

    seen = []

    class Model:
        def loglik(self, obs, X, arrays, force, thrown=None, **_):
            seen.append((obs["kind"], X, force))
            return np.full(len(arrays["mass"]), 1.0), {}

    monkeypatch.setattr(ec, "particle_matrix", lambda base, a, pose, force, names: (base["b"], pose["tag"], force))
    ident = {**_wrow(1), "key": "a__v11s0__d1__Bidentify", "probe": "identify", "kind": "measured", "reading": {"x": 1},
             "force": 6.0, "s": 0.45, "pose": {"tag": "identify"}, "base": {"b": 2.0}}
    cases = identify_cases([[_wrow(0)], [_wrow(1)]], [ident])
    a = {"mass": np.ones(3), "com_obj": np.zeros((3, 3))}
    pms = {"identify": Model(), "weigh": Model()}
    np.testing.assert_array_equal(ec.case_loglik(cases[0], pms, a), [0.0, 0.0, 0.0])     # missing: keeps the prior
    np.testing.assert_array_equal(ec.case_loglik(cases[1], pms, a), [1.0, 1.0, 1.0])
    assert seen == [("measured", (2.0, "identify", 6.0), 6.0)]
    seen.clear()
    ec.case_loglik([_wrow(0)], pms, a)                                                       # a weigh row: as before
    assert seen == [("hold", (1.0, "weigh", 4.0), 4.0)]


def test_run_refuses_a_generator_other_than_the_bundles(tmp_path, monkeypatch):
    """A bundle built under v11 refuses a run under v10 (RPENT_LAB_GENERATOR unset) before any job: the truths follow
    the generator."""
    from robots.mujoco_lab.estimation_check import main

    (tmp_path / "manifest.json").write_text(json.dumps({"generator": "v11"}))
    monkeypatch.delenv("RPENT_LAB_GENERATOR", raising=False)
    with pytest.raises(SystemExit, match="generator"):
        main(["run", "--bundle", str(tmp_path), "--crossfit", "--out", str(tmp_path / "out")])
