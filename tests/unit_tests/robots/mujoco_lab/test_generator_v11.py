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


"""Generator v11 (corpus v11): the lab's nominal masses are real-object masses (data/masses_v11.json, pinned by its
sha256), the mass scale is log-uniform on [1/sqrt(6), sqrt(6)], and the CoM point mass stays inside the total (the hulls
and the point mass both take 1 / (1 + f)). RPENT_LAB_GENERATOR picks the generator (v10 by default)."""

import hashlib
import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

V11_SHA256 = "d1cd4d7dde6c90d813dfb873da6a181c84c0c1b590b8651b3fd5fc64a776ead0"
PAIR = 2.0 / math.sqrt(6.0)        # a v11 scale over the v10 scale of the same draw: (1/sqrt(6)) / 0.5
SIM = pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")
LONG, BRICK, CRUSHABLE = "ycb__banana", "ycb__brick", "ycb__jello"
POSE = {"object_id": "ycb__brick", "nominal_mass": 0.5, "com_obj_nominal": [0.0, 0.0, 0.0],
        "pivot": {"centre": [0.0, 0.0, 0.0], "axis": [1.0, 0.0, 0.0], "lo": -0.1, "L": 0.2}}


def test_the_v11_table_is_pinned():
    from robots.mujoco_lab.masses import MASSES_V11, MASSES_V11_SHA256, load_masses

    assert MASSES_V11_SHA256 == V11_SHA256
    assert hashlib.sha256(open(MASSES_V11, "rb").read()).hexdigest() == V11_SHA256
    t = load_masses()
    assert t["version"] == "v11" and t["unit"] == "kg" and len(t["objects"]) == 95
    assert all(o["mass_kg"] > 0 and o["nominal_kg_v10"] > 0 for o in t["objects"].values())
    assert t["objects"]["hot3d__dumbbell"]["mass_kg"] == 2.268 and t["objects"]["ycb__banana"]["mass_kg"] == 0.1844


def test_a_changed_table_is_refused(tmp_path):
    from robots.mujoco_lab.masses import MASSES_V11, load_masses

    changed = tmp_path / "masses_v11.json"
    changed.write_text(open(MASSES_V11).read().replace('"mass_kg": 2.268', '"mass_kg": 2.269'))
    with pytest.raises(ValueError, match="sha256"):
        load_masses(str(changed))


def test_the_generator_comes_from_the_environment(monkeypatch):
    from robots.mujoco_lab.corpus import GENERATOR_ENV, lab_generator

    assert GENERATOR_ENV == "RPENT_LAB_GENERATOR"
    monkeypatch.delenv(GENERATOR_ENV, raising=False)
    assert lab_generator() == "v10"
    monkeypatch.setenv(GENERATOR_ENV, "v11")
    assert lab_generator() == "v11" and lab_generator("v10") == "v10"
    monkeypatch.setenv(GENERATOR_ENV, "v12")
    with pytest.raises(ValueError, match="v12"):
        lab_generator()


def test_the_v11_scale_range_is_centred_on_one():
    from robots.mujoco_lab.corpus import MASS_SCALE_RANGE, mass_scale_range

    assert mass_scale_range("v10") == MASS_SCALE_RANGE == (0.5, 3.0)
    lo, hi = mass_scale_range("v11")
    assert lo == pytest.approx(1 / math.sqrt(6), rel=1e-15) and hi == pytest.approx(math.sqrt(6), rel=1e-15)
    assert math.log(hi / lo) == pytest.approx(math.log(6.0), rel=1e-12)          # the width stays 6x
    with pytest.raises(ValueError):
        mass_scale_range("v9")


def test_v11_draws_pair_with_v10_draws(monkeypatch):
    """The same stream gives the same draw, except the scale, which is the v10 scale x 2/sqrt(6); with no argument
    draw_theta follows RPENT_LAB_GENERATOR."""
    from robots.mujoco_lab.corpus import case_rng, draw_theta

    for idx in range(50):
        t10, e10 = draw_theta(case_rng(0, "ycb__banana", idx), 0.184, generator="v10")
        t11, e11 = draw_theta(case_rng(0, "ycb__banana", idx), 0.184, generator="v11")
        assert e10 == e11 and {k: t10[k] for k in ("mu", "com_s", "v_lift")} == {k: t11[k] for k in ("mu", "com_s", "v_lift")}
        assert t11["mass_scale"] == pytest.approx(t10["mass_scale"] * PAIR, rel=1e-12)
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    assert draw_theta(case_rng(0, "ycb__banana", 3), 0.184) == draw_theta(case_rng(0, "ycb__banana", 3), 0.184, generator="v11")


def _priors():
    from robots.mujoco_lab.fragility import THRESHOLDS_V0, Thresholds, load_classes
    from robots.mujoco_lab.prior import make_prior

    classes, th = load_classes(), Thresholds.from_json(THRESHOLDS_V0)
    return make_prior(classes, th), make_prior(classes, th, generator="v11")


def test_v11_prior_keeps_the_point_mass_inside_the_total():
    """An empty object: a v11 particle's mass is nominal x scale; v10 adds (1 + f). The CoM and the friction are the
    same particles in both (one stream), and the v11 scale is the v10 scale x 2/sqrt(6)."""
    p10, p11 = _priors()
    P10 = p10(POSE, 4000, np.random.default_rng(5)).particles
    P11 = p11(POSE, 4000, np.random.default_rng(5)).particles
    assert any(p["f"] > 0 for p in P11)
    for a, b in zip(P10, P11):
        assert b["mass"] == pytest.approx(0.5 * b["mass_scale"], rel=1e-12)
        assert a["mass"] == pytest.approx(0.5 * a["mass_scale"] * (1.0 + a["f"]), rel=1e-12)
        assert b["mass_scale"] == pytest.approx(a["mass_scale"] * PAIR, rel=1e-12)
        assert (b["mu"], b["f"]) == (a["mu"], a["f"])
        assert np.allclose(b["com_obj"], a["com_obj"], rtol=0, atol=1e-12)     # (m com) / m: the same up to rounding


def test_v11_prior_of_a_filled_container():
    """A filled mug keeps its own scale range (0.8, 1.25) and has no CoM shift, so its particles match v10's; an empty
    one is the container's nominal x scale (no 1 + f)."""
    p10, p11 = _priors()
    pose = {**POSE, "object_id": "ycb__mug"}
    P10 = p10(pose, 2000, np.random.default_rng(6)).particles
    P11 = p11(pose, 2000, np.random.default_rng(6)).particles
    filled = [i for i, p in enumerate(P11) if p["fill"] > 0]
    assert 0 < len(filled) < 2000
    for i in filled:
        assert P11[i]["mass"] == pytest.approx(P10[i]["mass"], rel=1e-12) and P11[i]["f"] == 0.0
    for i in set(range(2000)) - set(filled):
        assert P11[i]["mass"] == pytest.approx(0.5 * P11[i]["mass_scale"], rel=1e-12)


def test_joint_hidden_follows_the_generator():
    from robots.mujoco_lab.prior import com_hypothesis, joint_hidden

    theta = {"mass_scale": 1.2, "mu": 0.5, "com_s": 0.8}
    _, f = com_hypothesis(POSE["pivot"], POSE["com_obj_nominal"], 0.8, +1)
    assert f > 0
    j10 = joint_hidden(POSE, {}, theta, +1, 0.0)
    j11 = joint_hidden(POSE, {}, theta, +1, 0.0, generator="v11")
    assert j10["mass"] == pytest.approx(0.5 * 1.2 * (1 + f), rel=1e-12)
    assert j11["mass"] == pytest.approx(0.5 * 1.2, rel=1e-12) and j11["com_obj"] == j10["com_obj"]


def test_generator_truth_follows_the_row():
    from robots.mujoco_lab.posterior_check import generator_truth
    from robots.mujoco_lab.prior import com_hypothesis

    row = {"object": "ycb__brick", "pose": POSE, "theta": {"mass_scale": 1.2, "mu": 0.5, "com_s": 0.7}, "com_end": 1, "fill": 0.0}
    _, f = com_hypothesis(POSE["pivot"], POSE["com_obj_nominal"], 0.7, +1)
    t10, t11 = generator_truth(row), generator_truth({**row, "generator": "v11"})
    assert t10["mass"] == pytest.approx(0.5 * 1.2 * (1 + f), rel=1e-12)
    assert t11["mass"] == pytest.approx(0.5 * 1.2, rel=1e-12) and t11["com"] == t10["com"] and t11["mu"] == 0.5
    with pytest.raises(ValueError):
        generator_truth({**row, "generator": "v9"})


def test_a_v11_manifest_names_the_v11_prior(tmp_path):
    """"lab_generator_v11" is the m2b joint prior under generator v11, on the class and threshold files the manifest
    names (sha256 checked); the m2b prior stays v10's."""
    from robots.mujoco_lab.fragility import THRESHOLDS_V0
    from robots.mujoco_lab.prior import DATA, M2B_PRIOR, V11_PRIOR, fragility_world, prior_from_manifest, sha256_of

    assert V11_PRIOR == "lab_generator_v11"
    world = {"classes": {"path": "fragility_classes.json", "sha256": sha256_of(f"{DATA}/fragility_classes.json")},
             "thresholds": {"path": THRESHOLDS_V0, "sha256": sha256_of(THRESHOLDS_V0)}}
    assert fragility_world({"prior": V11_PRIOR, **world}) is not None
    P10 = prior_from_manifest({"prior": M2B_PRIOR, **world})(POSE, 500, np.random.default_rng(2)).particles
    P11 = prior_from_manifest({"prior": V11_PRIOR, **world})(POSE, 500, np.random.default_rng(2)).particles
    assert all(b["mass"] == pytest.approx(0.5 * b["mass_scale"], rel=1e-12) for b in P11)
    assert all(b["mass_scale"] == pytest.approx(a["mass_scale"] * PAIR, rel=1e-12) for a, b in zip(P10, P11))


def test_manifest_generator():
    """A manifest's generator: its "generator" field, v10 without one; the joint priors fix it (v11's prior needs v11,
    the m2a and m2b priors need v10)."""
    from robots.mujoco_lab.robot_spec import manifest_generator

    assert manifest_generator({}) == "v10" and manifest_generator({"prior": "lab_generator_m2b"}) == "v10"
    assert manifest_generator({"prior": "lab_generator_v11", "generator": "v11"}) == "v11"
    assert manifest_generator({"prior": "m1_parametric", "generator": "v11"}) == "v11"
    for bad in ({"prior": "lab_generator_v11"}, {"prior": "lab_generator_m2b", "generator": "v11"},
                {"prior": "lab_generator", "generator": "v11"}, {"generator": "v12"}):
        with pytest.raises(ValueError):
            manifest_generator(bad)


@SIM
def test_object_props_follow_the_generator(monkeypatch):
    """v10: props.json as it is. v11: the table's mass and every hull mass x real / nominal, the rest unchanged; the
    pomegranate (its mesh does not compile, so it has no table row) is an error."""
    from robots.mujoco_lab.adapter import _props_path, load_sim, object_ids, object_props
    from robots.mujoco_lab.masses import load_masses

    monkeypatch.delenv("RPENT_LAB_GENERATOR", raising=False)
    table = load_masses()["objects"]
    assert set(object_ids()) - set(table) == {"fruits_veggies__pomegranate"} and set(table) <= set(object_ids())
    v10 = {oid: object_props(oid) for oid in table}
    assert all(v10[oid] == json.load(open(_props_path(load_sim(), oid))) for oid in table)
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    for oid, e in table.items():
        p10, p11 = v10[oid], object_props(oid)
        assert p11["mass"] == e["mass_kg"] and sum(p11["hull_mass"]) == pytest.approx(e["mass_kg"], rel=1e-9)
        assert np.allclose(p11["hull_mass"], np.asarray(p10["hull_mass"]) * e["mass_kg"] / p10["mass"], rtol=1e-12, atol=0)
        assert {k: v for k, v in p11.items() if k not in ("mass", "hull_mass")} == {k: v for k, v in p10.items() if k not in ("mass", "hull_mass")}
    assert object_props(LONG, generator="v10") == v10[LONG]
    with pytest.raises(ValueError, match="no real mass"):
        object_props("fruits_veggies__pomegranate")


@SIM
def test_v11_reset_keeps_the_com_mass_inside_the_total(monkeypatch):
    """A banana at scale 1.3 with its CoM moved to 0.8: under v10 the sim's mass is nominal x 1.3 x (1 + f); under v11
    it is real x 1.3, with v10's CoM in the object frame. The case keeps its own scale (the sim's is 1.3 / (1 + f)).
    scene.build_xml writes each hull mass to 0.01 g and the point mass to 0.1 g: the sim's mass is exact to that rounding,
    and the CoM to the 0.1 mm it moves (0.09 mm measured on the marker, 0.007 mm on the banana)."""
    from robots.mujoco_lab.adapter import MuJoCoLabEnv, object_props

    theta = {"mu": 0.8, "mass_scale": 1.3, "com_s": 0.8, "v_lift": 0.15}
    monkeypatch.delenv("RPENT_LAB_GENERATOR", raising=False)
    env = MuJoCoLabEnv()
    env.reset(LONG, theta, seed=0)
    t10 = env.truth()
    f = env.s["t3_make"].com_shift_for(env.obj, env.props, 0.8)[3]
    rounding = len(env.props["hull_mass"]) * 5e-6 + 5e-5
    assert f > 0.1 and env.generator == "v10"
    assert abs(t10["mass"] - object_props(LONG)["mass"] * 1.3 * (1 + f)) <= rounding
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    env.reset(LONG, theta, seed=0)
    t11 = env.truth()
    assert env.generator == "v11" and env.theta["mass_scale"] == 1.3
    assert abs(t11["mass"] - 0.1844 * 1.3) <= rounding
    assert np.allclose(t11["com_obj"], t10["com_obj"], rtol=0, atol=2e-4)
    assert env.pose_info()["nominal_mass"] == 0.1844 and env.nominal_mass() == 0.1844


@SIM
def test_v11_corpus_rows_pair_with_v10_rows(tmp_path, monkeypatch):
    """A v11 row records its generator and has a key no v10 row has; its draw is the v10 row's but the scale
    (x 2/sqrt(6)), and the sim's mass is real x scale. Resuming a file of the other generator is refused."""
    from robots.mujoco_lab.corpus import main as corpus_main

    out10, out11 = tmp_path / "c10.jsonl", tmp_path / "c11.jsonl"
    monkeypatch.delenv("RPENT_LAB_GENERATOR", raising=False)
    corpus_main(["--out", str(out10), "--draws", "2", "--workers", "1", "--objects", LONG])
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    corpus_main(["--out", str(out11), "--draws", "2", "--workers", "1", "--objects", LONG])
    r10, r11 = [json.loads(x) for x in out10.open()], [json.loads(x) for x in out11.open()]
    assert [r["key"] for r in r10] == [f"{LONG}__d0", f"{LONG}__d1"] and [r["key"] for r in r11] == [f"{LONG}__v11d0", f"{LONG}__v11d1"]
    assert {r["generator"] for r in r10} == {"v10"} and {r["generator"] for r in r11} == {"v11"}
    for a, b in zip(r10, r11):
        assert b["theta"]["mass_scale"] == pytest.approx(a["theta"]["mass_scale"] * PAIR, rel=1e-12)
        assert {k: b["theta"][k] for k in ("mu", "com_s", "v_lift")} == {k: a["theta"][k] for k in ("mu", "com_s", "v_lift")}
        assert [b[k] for k in ("com_end", "destination", "s", "v_transport", "fill")] == [a[k] for k in ("com_end", "destination", "s", "v_transport", "fill")]
        assert abs(b["truth"]["mass"] - 0.1844 * b["theta"]["mass_scale"]) <= 1e-4 and b["pose"]["nominal_mass"] == 0.1844
    with pytest.raises(ValueError, match="generator"):
        corpus_main(["--out", str(out10), "--draws", "3", "--workers", "1", "--objects", LONG, "--resume"])
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v10")
    with pytest.raises(ValueError, match="generator"):
        corpus_main(["--out", str(out11), "--draws", "3", "--workers", "1", "--objects", LONG, "--resume"])


@SIM
def test_v11_probe_rows(tmp_path, monkeypatch):
    """v11 probe rows record their generator, carry v11 keys and the real mass x scale; scene C's squeeze cap is the v11
    prior's, which is v10's: the cap reads only the particles' damage force, and no generator changes it. Resuming a
    file of the other generator is refused."""
    from robots.mujoco_lab.adapter import MuJoCoLabEnv
    from robots.mujoco_lab.corpus import world
    from robots.mujoco_lab.probe_corpus import SQUEEZE_PRIOR_N, SQUEEZE_STREAM, main as probe_main
    from robots.mujoco_lab.prior import make_prior
    from rpent.skills.pick_place.probes import squeeze_cap
    import zlib

    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    out = tmp_path / "p.jsonl"
    probe_main(["--out", str(out), "--draws", "1", "--workers", "1", "--objects", CRUSHABLE])
    rows = [json.loads(x) for x in out.open()]
    assert rows and all(r["generator"] == "v11" and r["key"].startswith(f"{CRUSHABLE}__v11s7__d0") for r in rows)
    assert all(abs(r["truth"]["mass"] - 0.097 * r["theta"]["mass_scale"]) <= 1e-4 for r in rows)     # build_xml's rounding
    squeeze = [r for r in rows if r.get("scene") == "C"]
    assert squeeze
    env = MuJoCoLabEnv()
    env.reset(CRUSHABLE, seed=0)
    C, th = world()
    cap = {g: float(squeeze_cap(make_prior(C, th, generator=g)(env.pose_info(), SQUEEZE_PRIOR_N, np.random.default_rng(
        [7, zlib.crc32(CRUSHABLE.encode()), 0, SQUEEZE_STREAM])), squeeze[0]["f_std"])) for g in ("v10", "v11")}
    assert squeeze[0]["force"] == cap["v11"] == cap["v10"]
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v10")
    with pytest.raises(ValueError, match="generator"):
        probe_main(["--out", str(out), "--draws", "2", "--workers", "1", "--objects", CRUSHABLE, "--resume"])


def test_corpus_generator():
    """The generator of a corpus's rows: a row without the field is v10; rows of two generators are an error."""
    from robots.mujoco_lab.train_models import corpus_generator

    assert corpus_generator([{"key": "a"}, {"key": "b", "generator": "v10"}]) == "v10"
    assert corpus_generator([{"generator": "v11"}, {"generator": "v11"}]) == "v11"
    with pytest.raises(ValueError, match="generator"):
        corpus_generator([{"generator": "v11"}, {"key": "a"}])


def test_training_records_the_generator(tmp_path, monkeypatch):
    """A bundle declares its corpus's generator; training under another RPENT_LAB_GENERATOR is refused, and so is the
    m2a lab prior (v10's) on v11 rows; the m2b world under v11 names the v11 prior."""
    from robots.mujoco_lab.train_models import m2b_world, main
    from tests.unit_tests.skills.synthetic import synthetic_rows, with_pose

    corpus = tmp_path / "c.jsonl"
    corpus.write_text("".join(json.dumps(dict(r, generator="v11")) + "\n" for r in with_pose(synthetic_rows(300))))
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    main(["--corpus", str(corpus), "--out-dir", str(tmp_path / "b"), "--n-belief", "4", "--features", "nominal"])
    assert json.load(open(tmp_path / "b" / "manifest.json"))["generator"] == "v11"
    with pytest.raises(SystemExit, match="v10"):
        main(["--corpus", str(corpus), "--out-dir", str(tmp_path / "t"), "--n-belief", "4", "--features", "true"])
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v10")
    with pytest.raises(SystemExit, match="RPENT_LAB_GENERATOR"):
        main(["--corpus", str(corpus), "--out-dir", str(tmp_path / "x"), "--n-belief", "4", "--features", "nominal"])
    assert m2b_world(None, None, generator="v11")[1] == "lab_generator_v11" and m2b_world(None, None)[1] == "lab_generator_m2b"


def test_probe_models_refuse_another_generator(tmp_path):
    """The probe corpus carries the bundle's generator (a manifest without one declares v10)."""
    from robots.mujoco_lab.train_models import fit_probes, write_manifest
    from tests.unit_tests.skills.test_probes import _probe_rows

    rows = [dict(r, generator="v11") for r in _probe_rows(400)]
    (tmp_path / "p.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    objects = sorted({r["object"] for r in rows})
    split = {"train": objects[:17], "val": objects[17:]}
    os.makedirs(tmp_path / "b")
    write_manifest(str(tmp_path / "b"), split_seed=0)
    with pytest.raises(ValueError, match="generator"):
        fit_probes(str(tmp_path / "p.jsonl"), str(tmp_path / "b"), split)
    write_manifest(str(tmp_path / "b"), generator="v11")
    fit_probes(str(tmp_path / "p.jsonl"), str(tmp_path / "b"), split)


def test_load_bundle_refuses_another_generator(tmp_path, monkeypatch):
    """A job runs a bundle under its generator: load_bundle checks RPENT_LAB_GENERATOR before it loads a model, and
    returns the generator."""
    from robots.mujoco_lab.robot_spec import load_bundle

    (tmp_path / "manifest.json").write_text(json.dumps({"prior": "lab_generator_v11", "generator": "v11", "success_model": "none.joblib"}))
    monkeypatch.delenv("RPENT_LAB_GENERATOR", raising=False)
    with pytest.raises(ValueError, match="RPENT_LAB_GENERATOR=v11"):
        load_bundle(str(tmp_path))
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    with pytest.raises(FileNotFoundError):                    # past the check: the model file is missing
        load_bundle(str(tmp_path))


@SIM
def test_a_case_records_its_generator(monkeypatch):
    """A calibrate case draws under the process's generator and records it; a run reports its cases' generator."""
    import robots.mujoco_lab.calibrate as cal

    class Env:
        force_noise, reset_info = 0.0, {"at_rest": True}

        def reset(self, object_id, theta, **kw):
            self.theta = dict(theta)

        def truth(self):
            return {"mass": 1.0, "mu": 0.5, "com_s": 0.5}

    monkeypatch.setattr(cal, "run_protocol", lambda *a, **kw: {"attempts": []})
    rows = {}
    for g in ("v10", "v11"):
        monkeypatch.setenv("RPENT_LAB_GENERATOR", g)
        rows[g] = cal.run_case(SimpleNamespace(env=Env(), manifest={}), BRICK, 3, 5)
        assert rows[g]["generator"] == g
    assert rows["v11"]["theta"]["mass_scale"] == pytest.approx(rows["v10"]["theta"]["mass_scale"] * PAIR, rel=1e-12)
    assert cal.run_generator([rows["v11"], {"error": "x"}]) == "v11" and cal.run_generator(list(rows.values())) == ["v10", "v11"]


def test_sweep_terciles_follow_the_generator():
    """A case's scale tercile is within its own generator's range: 1.6 is mid in v10's [0.5, 3] and high in v11's."""
    from robots.mujoco_lab.fragility import load_classes
    from robots.mujoco_lab.sweep import subgroup

    case = {"object": BRICK, "theta": {"mass_scale": 1.6}, "fill": 0.0}
    assert subgroup(case, load_classes())[1] == "mid" and subgroup({**case, "generator": "v11"}, load_classes())[1] == "high"


def test_squeeze_sensitivity_uses_the_bundle_generator():
    """The sensitivity's prior under each law is the bundle's generator's."""
    from robots.mujoco_lab.fragility import THRESHOLDS_V0, Thresholds
    from robots.mujoco_lab.prior import DATA, V11_PRIOR, sha256_of
    from robots.mujoco_lab.squeeze_sensitivity import law_prior

    man = {"prior": V11_PRIOR, "generator": "v11",
           "classes": {"path": "fragility_classes.json", "sha256": sha256_of(f"{DATA}/fragility_classes.json")},
           "thresholds": {"path": THRESHOLDS_V0, "sha256": sha256_of(THRESHOLDS_V0)}}
    prior = law_prior({"manifest": man, "generator": "v11"}, Thresholds.from_json(THRESHOLDS_V0))
    P = prior(POSE, 300, np.random.default_rng(0)).particles
    assert all(p["mass"] == pytest.approx(0.5 * p["mass_scale"], rel=1e-12) for p in P)
