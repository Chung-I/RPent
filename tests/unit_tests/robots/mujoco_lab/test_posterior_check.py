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


"""The posterior check: prior draws, one update with each row's readouts, 80 % interval coverage
(needs the sim for the corpus rows)."""

import json
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")


@pytest.fixture(scope="module")
def tiny_bundle(tmp_path_factory):
    from robots.mujoco_lab.corpus import main as corpus_main
    from robots.mujoco_lab.train_models import main as train_main

    d = tmp_path_factory.mktemp("pc")
    corpus = d / "c.jsonl"
    assert corpus_main(["--out", str(corpus), "--draws", "8", "--workers", "1", "--objects", "ycb__banana,ycb__brick,ycb__mug"]) == 24
    train_main(["--corpus", str(corpus), "--out-dir", str(d / "m2"), "--n-belief", "4"])
    return d / "m2"


def test_train_models_writes_the_readout_model(tiny_bundle):
    man = json.load(open(tiny_bundle / "manifest.json"))
    assert man["readout_model"] == "readout_model.json" and (tiny_bundle / man["readout_model"]).exists()
    metrics = json.load(open(tiny_bundle / man["metrics"]))
    assert "hold" in metrics["readout"]["terms"] and "side_error" in metrics["readout"]


def test_posterior_check_reports_coverage(tiny_bundle, tmp_path):
    from robots.mujoco_lab.posterior_check import main
    from robots.mujoco_lab.robot_spec import build_lab

    out = main(["--bundle", str(tiny_bundle), "--split", "val", "--n-particles", "200", "--temper-grid", "1,2",
                "--choose-on", "val", "--out", str(tmp_path / "pc.json")])
    assert {"mass_cov80", "mu_cov80", "com_cov80", "by_label", "temper", "grid", "feasible", "prior_cov80"} <= set(out)
    assert out["n"] == 8 and all(0.0 <= out[k] <= 1.0 for k in ("mass_cov80", "mu_cov80", "com_cov80"))
    assert sum(v["n"] for v in out["by_label"].values()) == 8 and set(out["grid"]) == {"1.0", "2.0"}
    assert json.load(open(tmp_path / "pc.json"))["n"] == 8
    man = json.load(open(tiny_bundle / "manifest.json"))
    assert man["tempers"]["readout"] == out["temper"]
    prims = build_lab(bundle=str(tiny_bundle))
    assert prims.readout_model is not None and prims.readout_model.temper == out["temper"]
    assert out["truth"] == "generator" and {"cov80", "prior_cov80"} <= set(out["sim_truth"])
    assert all(0.0 <= out["sim_truth"]["cov80"][k] <= 1.0 for k in ("mass", "mu", "com"))


def _row(com_s, end, offset=0.0005):
    """A synthetic row: pivot along x over [-0.1, 0.1] m, nominal CoM at the centre; the sim's CoM sits
    ``offset`` m past the generator's along the pivot."""
    from robots.mujoco_lab.prior import com_hypothesis

    pose = {"pivot": {"centre": [0.0, 0.0, 0.0], "axis": [1.0, 0.0, 0.0], "lo": -0.1, "L": 0.2},
            "com_obj_nominal": [0.0, 0.0, 0.0], "nominal_mass": 0.4}
    com, f = com_hypothesis(pose["pivot"], pose["com_obj_nominal"], com_s, end)
    row = {"pose": pose, "theta": {"mass_scale": 1.5, "mu": 0.4, "com_s": com_s}, "com_end": end,
           "com_obj_true": [com[0] + offset, com[1], com[2]], "truth": {"mass": 0.4 * 1.5 * (1 + f) * 1.001, "mu": 0.4}}
    return row, f


def test_generator_truth_is_the_row_theta_in_the_prior_space():
    from robots.mujoco_lab.posterior_check import generator_truth, sim_truth

    row, f = _row(0.7, 1)
    g, s = generator_truth(row), sim_truth(row)
    assert f > 0 and g["mass"] == pytest.approx(0.4 * 1.5 * (1 + f), rel=1e-12) and g["mu"] == 0.4
    assert g["com"] == pytest.approx(0.7, abs=1e-12) and s["com"] - g["com"] == pytest.approx(0.0005 / 0.2, abs=1e-12)
    assert s["mass"] == pytest.approx(g["mass"] * 1.001, rel=1e-12)


def test_an_atom_posterior_covers_the_generator_truth_not_the_sim_truth():
    """All weight on the unshifted hypotheses (f = 0): a zero-width CoM interval at the nominal fraction.
    The generator's CoM is that point; the sim's sits 0.5 mm away and falls outside."""
    import numpy as np

    from robots.mujoco_lab.posterior_check import _hits, generator_truth, sim_truth
    from robots.mujoco_lab.prior import lab_prior

    row, _ = _row(None, 1)
    b = lab_prior(row["pose"], 2000, np.random.default_rng(0))
    post = b.reweight(np.where(b.arrays()["f"] == 0, 0.0, -np.inf))
    assert post.quantile("com_frac", 0.1) == post.quantile("com_frac", 0.9) == pytest.approx(0.5)
    assert _hits(post, generator_truth(row))["com"] and not _hits(post, sim_truth(row))["com"]


def test_rotation_threshold_matches_the_readouts():
    from robots.mujoco_lab.readouts import ROT_SIDE_DEG
    from rpent.skills.pick_place.likelihood import ROT_DEG

    assert ROT_DEG == ROT_SIDE_DEG


@pytest.fixture(scope="module")
def probe_bundle(tiny_bundle, tmp_path_factory):
    """The tiny bundle plus probe models fitted on a probe corpus of the same three objects."""
    from robots.mujoco_lab.probe_corpus import main as probe_main
    from robots.mujoco_lab.train_models import main as train_main

    pc = tmp_path_factory.mktemp("pp") / "p.jsonl"
    assert probe_main(["--out", str(pc), "--draws", "8", "--workers", "1", "--objects", "ycb__banana,ycb__brick,ycb__mug", "--seed", "7"]) >= 48
    out = train_main(["--probe-corpus", str(pc), "--out-dir", str(tiny_bundle)])
    rows = [r for r in map(json.loads, open(pc)) if "error" not in r]
    return tiny_bundle, rows, out


def test_train_models_fits_the_probe_models(probe_bundle):
    from robots.mujoco_lab.robot_spec import load_bundle
    from rpent.skills.pick_place.probes import PROBE_KINDS

    bundle, rows, out = probe_bundle
    man = json.load(open(bundle / "manifest.json"))
    assert man["probe_models"] == {"weigh": "probe_weigh.joblib", "push": "probe_push.joblib"}
    assert man["readout_model"] == "readout_model.json"          # a probe-only fit leaves the main models alone
    metrics = json.load(open(bundle / man["probe_metrics"]))
    assert metrics["split"] == json.load(open(bundle / man["metrics"]))["split"] and metrics["probe_corpus"] == man["probe_corpus"]
    for probe in ("weigh", "push"):
        m = metrics["probes"][probe]
        assert set(m["kind_rates"]) == set(PROBE_KINDS[probe]) and sum(m["kind_rates"].values()) == pytest.approx(1.0)
        assert m["n"] == sum(r["probe"] == probe for r in rows) and {"brier", "brier_const", "n"} <= set(m["val"])
    B = load_bundle(str(bundle))
    assert set(B["probe_models"]) == {"weigh", "push"} and B["probe_models"]["push"].probe == "push"


@pytest.mark.parametrize("probes", ["A", "B"])
def test_posterior_check_after_the_probes(probe_bundle, tmp_path, probes):
    from robots.mujoco_lab.posterior_check import main

    bundle, rows, _ = probe_bundle
    val = set(json.load(open(bundle / "probe_metrics.json"))["split"]["val"])
    tags = [r["key"].rsplit("__", 1)[1] for r in rows if r["object"] in val]
    out = main(["--bundle", str(bundle), "--split", "val", "--probes", probes, "--n-particles", "200", "--temper-grid", "1,2",
                "--out", str(tmp_path / f"pc_{probes}.json")])
    assert out["probes"] == probes and out["n"] == tags.count("Bweigh" if probes == "B" else "Apush") > 0
    assert {"mass_cov80", "mu_cov80", "com_cov80", "prior_cov80", "by_kind", "grid", "sim_truth", "terms"} <= set(out)
    if probes == "A":
        assert out["n_with_weigh"] == tags.count("Aweigh")


def test_probe_temper_reaches_the_probe_models(probe_bundle, tmp_path):
    from robots.mujoco_lab.posterior_check import main
    from robots.mujoco_lab.robot_spec import load_bundle

    bundle, _, _ = probe_bundle
    before = json.load(open(bundle / "manifest.json"))["tempers"].get("readout")
    out = main(["--bundle", str(bundle), "--split", "val", "--probes", "B", "--n-particles", "200", "--temper-grid", "1,2",
                "--choose-on", "val", "--out", str(tmp_path / "pc.json")])
    tempers = json.load(open(bundle / "manifest.json"))["tempers"]
    assert tempers["probe"] == out["temper"] and tempers.get("readout") == before
    assert all(pm.temper == out["temper"] for pm in load_bundle(str(bundle))["probe_models"].values())
