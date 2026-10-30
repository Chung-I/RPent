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

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")


@pytest.fixture(scope="module")
def tiny_bundle(tmp_path_factory):
    from robots.mujoco_lab.corpus import main as corpus_main
    from robots.mujoco_lab.train_models import main as train_main

    d = tmp_path_factory.mktemp("pc")
    corpus = d / "c.jsonl"
    assert corpus_main(["--out", str(corpus), "--draws", "8", "--workers", "1", "--objects", "ycb__banana,ycb__brick,ycb__mug",
                        "--mode", "m2"]) == 24                              # an m2 bundle trains on the m2 world
    train_main(["--corpus", str(corpus), "--out-dir", str(d / "m2"), "--n-belief", "4"])
    return d / "m2"


def test_train_models_writes_the_readout_model(tiny_bundle):
    man = json.load(open(tiny_bundle / "manifest.json"))
    assert man["readout_model"] == "readout_model.json" and (tiny_bundle / man["readout_model"]).exists()
    metrics = json.load(open(tiny_bundle / man["metrics"]))
    assert "hold" in metrics["readout"]["terms"] and "side_error" in metrics["readout"]


def test_posterior_check_draws_from_the_bundle_prior(tiny_bundle, tmp_path, monkeypatch):
    """The check draws each case's prior particles from the bundle's prior (its manifest's), not a fixed one."""
    import robots.mujoco_lab.posterior_check as pc
    from robots.mujoco_lab.prior import lab_prior

    calls = []

    def recording(pose, n=2000, rng=None):
        calls.append(n)
        return lab_prior(pose, n, rng)

    plain = pc.load_bundle
    monkeypatch.setattr(pc, "load_bundle", lambda *a, **k: {**plain(*a, **k), "prior": recording})
    out = pc.main(["--bundle", str(tiny_bundle), "--split", "val", "--n-particles", "50", "--temper-grid", "1",
                   "--choose-on", "val", "--out", str(tmp_path / "pc.json")])
    assert calls and set(calls) == {50} and len(calls) >= out["n"]


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
    laws = out["laws"]                                         # term -> label -> the law's kind, core weight and widths
    assert "hold" in laws and all(set(v) == {"kind", "w", "sd1", "sd2"} for t in laws.values() for v in t.values())
    assert all(v["kind"] in ("normal", "mix2") for t in laws.values() for v in t.values())


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
    """The tiny bundle plus probe models fitted on a probe corpus of the same three objects; the rows are the
    trainable ones, which the fit and the check use (the mug creeps at rest in some draws: those rows are flagged)."""
    from robots.mujoco_lab.corpus import trainable
    from robots.mujoco_lab.probe_corpus import main as probe_main
    from robots.mujoco_lab.train_models import main as train_main

    pc = tmp_path_factory.mktemp("pp") / "p.jsonl"
    assert probe_main(["--out", str(pc), "--draws", "8", "--workers", "1", "--objects", "ycb__banana,ycb__brick,ycb__mug", "--seed", "7",
                       "--mode", "m2"]) >= 48
    out = train_main(["--probe-corpus", str(pc), "--out-dir", str(tiny_bundle)])
    rows = [r for r in map(json.loads, open(pc)) if trainable(r)]
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
    assert set(out["laws"]) == {"weigh", "push"} and "measured" in out["laws"]["weigh"]       # probe -> kind -> law summary


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


class _RecordingProbe:
    """A probe model that tells nothing and keeps the feature rows it was given."""

    probe, classes, temper = "squeeze", ("measured",), 1.0

    def __init__(self):
        from rpent.skills.pick_place.features import FEATURES_M2B

        self.feature_names, self.seen, self.thrown = FEATURES_M2B, [], []

    def loglik(self, obs, X, arrays, force, temper=None, terms=None, P=None, thrown=None):
        self.seen.append(X)
        self.thrown.append(thrown)
        return np.zeros(len(X)), {"used": ["kind"], "rejected": []}


def test_probe_check_reads_the_entry_and_the_squeeze_scene():
    """Scene C (the squeeze) is a case of its own, and the probe models see the row's public entry: the probe corpus's
    poses carry none."""
    from robots.mujoco_lab.posterior_check import check_probe_cases, probe_cases
    from rpent.skills.pick_place.belief import ParticleBelief
    from rpent.skills.pick_place.features import FEATURES_M2B
    from tests.unit_tests.skills.synthetic import synthetic_pose, synthetic_rows

    base = dict(synthetic_rows(1)[0]["features"], force=5.0)
    row = {"key": "o__s0__d0__Csqueeze", "object": "o", "draw": 0, "probe": "squeeze", "kind": "measured", "reading": {"k_hat": 2.0},
           "base": base, "pose": synthetic_pose(), "entry": {"class": "crushable", "law": "cardboard", "fillable": False}, "force": 5.0,
           "theta": {"mass_scale": 1.0, "mu": 0.5, "com_s": None}, "com_end": 1, "truth": {"mass": 0.2, "mu": 0.5},
           "com_obj_true": [0.0, 0.0, 0.03]}
    cases = probe_cases([row, dict(row, key="o__s0__d0__Bweigh", probe="weigh")], "C")
    assert [[r["probe"] for r in c] for c in cases] == [["squeeze"]]
    pm = _RecordingProbe()
    prior = lambda pose, n, rng: ParticleBelief([{"mass": 0.2, "mu": 0.5, "com_obj": [0.0, 0.0, 0.03], "com_frac": 0.5,  # noqa: E731
                                                  "f_damage_n": 20.0}] * n, np.ones(n))
    check_probe_cases(cases, {"probe_models": {"squeeze": pm}, "prior": prior}, [1.0], 5, 0)
    assert np.all(pm.seen[0][:, FEATURES_M2B.index("frag_crush")] == 1.0)


def test_probe_check_tells_the_likelihood_whether_the_probe_threw_the_object():
    """The posterior check scores the belief the skill builds: a probe row's likelihood hears whether that probe threw
    the object, so a result that kept it weighs each particle's survival (Codex review of the fixes, plan 2b)."""
    from robots.mujoco_lab.posterior_check import check_probe_cases, probe_cases
    from rpent.skills.pick_place.belief import ParticleBelief
    from tests.unit_tests.skills.synthetic import synthetic_pose, synthetic_rows

    base = dict(synthetic_rows(1)[0]["features"], force=5.0)
    row = {"key": "o__s0__d0__Csqueeze", "object": "o", "draw": 0, "probe": "squeeze", "kind": "measured", "reading": {"k_hat": 2.0},
           "base": base, "pose": synthetic_pose(), "entry": {"class": "crushable", "law": "cardboard", "fillable": False}, "force": 5.0,
           "theta": {"mass_scale": 1.0, "mu": 0.5, "com_s": None}, "com_end": 1, "truth": {"mass": 0.2, "mu": 0.5},
           "com_obj_true": [0.0, 0.0, 0.03], "thrown": False}
    pm = _RecordingProbe()
    prior = lambda pose, n, rng: ParticleBelief([{"mass": 0.2, "mu": 0.5, "com_obj": [0.0, 0.0, 0.03], "com_frac": 0.5,  # noqa: E731
                                                  "f_damage_n": 20.0}] * n, np.ones(n))
    check_probe_cases(probe_cases([row], "C"), {"probe_models": {"squeeze": pm}, "prior": prior}, [1.0, 2.0], 5, 0)
    assert pm.thrown == [False, False]


def _fragility_check(tmp_path, monkeypatch):
    """A squeeze case and four weigh cases (two validation, two training) on a three-particle prior."""
    from robots.mujoco_lab import posterior_check as pc
    from rpent.skills.pick_place.belief import ParticleBelief
    from tests.unit_tests.skills.synthetic import synthetic_pose, synthetic_rows

    row = {"object": "v", "draw": 0, "pose": synthetic_pose(), "base": synthetic_rows(1)[0]["features"], "force": 5.0,
           "kind": "measured", "reading": {}, "theta_frag": {"f_damage_n": 20.0}, "fill": 0.7,
           "entry": {"class": "crushable", "law": "cardboard", "fillable": True},
           "theta": {"mass_scale": 1.0, "mu": 0.5, "com_s": None}, "com_end": 1,
           "truth": {"mass": 0.2, "mu": 0.5}, "com_obj_true": [0.0, 0.0, 0.03]}
    rows = [dict(row, key="v__d0__Csqueeze", probe="squeeze"),
            dict(row, key="v__d0__Bweigh", probe="weigh"),
            dict(row, key="v__d1__Bweigh", probe="weigh", draw=1, fill=0.0),
            dict(row, object="tr", key="tr__d0__Bweigh", probe="weigh", fill=0.0),
            dict(row, object="tr", key="tr__d1__Bweigh", probe="weigh", draw=1)]
    particles = [{"mass": 0.2, "mu": 0.5, "com_obj": [0.0, 0.0, 0.03], "com_frac": 0.5,
                  "f_damage_n": f, "fill": fill} for f, fill in zip((10.0, 20.0, 30.0), (0.0, 0.6, 0.9))]
    pm = _RecordingProbe()
    pm.reading = {}
    from robots.mujoco_lab.fragility import load_classes

    mug = load_classes()["ycb__mug"]                           # the filled rows' truth reads a cavity
    bundle = {"manifest": {"metrics": "metrics.json", "probe_corpus": "unused"}, "probe_models": {"squeeze": pm, "weigh": pm},
              "prior": lambda pose, n, rng: ParticleBelief(particles, np.array([0.25, 0.5, 0.25])), "classes": {"v": mug, "tr": mug}}
    (tmp_path / "metrics.json").write_text(json.dumps({"split": {"train": ["tr"], "val": ["v"]}}))
    monkeypatch.setattr(pc, "load_bundle", lambda path: bundle)
    monkeypatch.setattr(pc, "load_rows", lambda path: rows)
    monkeypatch.setattr(pc.tracking, "log_run", lambda *a, **kw: None)
    common = ["--bundle", str(tmp_path), "--split", "val"]
    return pc, common


def test_posterior_check_reports_fragility_and_fill_coverage(tmp_path, monkeypatch):
    pc, common = _fragility_check(tmp_path, monkeypatch)
    squeeze = pc.main(common + ["--probes", "C", "--out", str(tmp_path / "c.json")])["squeeze_f_damage"]
    assert squeeze["n"] == 1 and squeeze["post"]["coverage"] == 1.0
    assert squeeze["prior"]["log_width_median"] == pytest.approx(np.log(3.0))
    fill = pc.main(common + ["--probes", "B", "--out", str(tmp_path / "b.json")])["weigh_fill"]
    assert fill["n"] == 2 and fill["n_filled"] == 1 and fill["constant_p_filled"] == 0.5
    for name in ("prior", "post"):
        assert fill[name]["filled_only_coverage"] == 1.0
        assert fill[name]["filled_only_width_median"] == pytest.approx(0.3)
        assert fill[name]["p_filled_when_empty"] == fill[name]["p_filled_when_filled"] == 0.75
        assert fill[name]["brier"] == pytest.approx(0.3125) and fill[name]["brier_const"] == 0.25
        assert fill[name]["ece"] == pytest.approx(0.25)


def test_generator_truth_adds_the_liquid():
    """A filled row's generator truth is the joint generator (prior.joint_hidden), the container and its liquid, as the
    m2b particles carry them; an empty row is the container alone; a filled row needs the classes (its cavity)."""
    from robots.mujoco_lab.fragility import load_classes
    from robots.mujoco_lab.posterior_check import generator_truth
    from robots.mujoco_lab.prior import joint_hidden

    classes = load_classes()
    row, _ = _row(None, 1)
    row = dict(row, object="ycb__mug", fill=0.8)
    joint = joint_hidden(row["pose"], classes["ycb__mug"], row["theta"], 1, 0.8)
    pv = row["pose"]["pivot"]
    frac = (np.asarray(joint["com_obj"]) @ np.asarray(pv["axis"]) - pv["lo"]) / pv["L"]
    g = generator_truth(row, classes)
    assert joint["liquid_mass"] > 0.05 and g["mass"] == pytest.approx(joint["mass"], rel=1e-12)
    assert g["com"] == pytest.approx(frac, abs=1e-12) and g["mu"] == 0.4
    assert generator_truth(dict(row, fill=0.0), classes)["mass"] == pytest.approx(0.4 * 1.5, rel=1e-12)
    with pytest.raises(ValueError, match="cavity"):
        generator_truth(row)


def _hits(pattern):
    """(hits, n) per object -> the hit list and its objects."""
    hits = [h for k, n in pattern for h in [True] * k + [False] * (n - k)]
    objects = [f"o{i}" for i, (_, n) in enumerate(pattern) for _ in range(n)]
    return hits, objects


def test_cell_verdict_rule():
    """Plan 2b2 decision 9: pass when the coverage lies in [0.75, 0.85], fail when its 95 % object-bootstrap interval
    lies wholly outside the band, otherwise undecided; under 30 cases or 4 objects, insufficient."""
    from robots.mujoco_lab.posterior_check import cell_verdict

    rng = np.random.default_rng(0)
    v = cell_verdict(*_hits([(16, 20)] * 10), rng)
    assert (v["verdict"], v["n"], v["n_objects"]) == ("pass", 200, 10) and v["coverage"] == pytest.approx(0.80)
    v = cell_verdict(*_hits([(38, 40)] * 10), rng)
    assert v["verdict"] == "fail" and v["ci"][0] > 0.85
    v = cell_verdict(*_hits([(10, 10), (10, 10), (8, 10), (8, 10)]), rng)
    assert v["coverage"] == pytest.approx(0.90) and v["ci"][0] <= 0.85 and v["verdict"] == "undecided"
    assert cell_verdict(*_hits([(8, 10)] * 2), rng)["verdict"] == "insufficient"           # 20 cases
    assert cell_verdict(*_hits([(8, 10)] * 3), rng)["verdict"] == "insufficient"           # 30 cases, 3 objects


def test_judged_cells_are_what_each_probe_set_reads():
    """A probe that reads no property cannot calibrate it: the squeeze is judged on the damage force only, the push
    and the weigh on the mass and the friction, one attempt on all three."""
    from robots.mujoco_lab.posterior_check import judged_keys

    assert judged_keys(None) == ("mass", "mu", "com")
    assert judged_keys("A") == judged_keys("B") == ("mass", "mu")
    assert judged_keys("C") == ()


def test_squeeze_cells_judge_only_the_damage_force(tmp_path, monkeypatch):
    """Plan 2b2 decision 9 in the check's output: after a squeeze, the mass, the friction and the CoM are reported, not
    judged, and the damage force is judged (here too few cases); after a weigh, the CoM is reported and the fill and
    the P(filled) ECE need their minimums."""
    pc, common = _fragility_check(tmp_path, monkeypatch)
    b2 = pc.main(common + ["--probes", "C", "--out", str(tmp_path / "c.json")])["acceptance_b2"]
    assert [b2[k]["verdict"] for k in ("mass", "mu", "com")] == ["reported"] * 3
    assert b2["f_damage"]["verdict"] == "insufficient" and b2["f_damage"]["n"] == 1
    b2 = pc.main(common + ["--probes", "B", "--out", str(tmp_path / "b.json")])["acceptance_b2"]
    assert b2["com"]["verdict"] == "reported" and b2["mass"]["verdict"] == "insufficient"
    assert b2["fill"]["verdict"] == "insufficient" and b2["p_filled_ece"]["verdict"] == "insufficient"
