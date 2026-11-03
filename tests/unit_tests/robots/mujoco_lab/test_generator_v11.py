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


"""Generator v11 (corpus v11): the lab's nominal masses are real-object masses (data/masses_v11b.json, pinned by its
sha256; 7 tools with their head and handle masses), the mass scale is log-uniform on [1/sqrt(6), sqrt(6)], and the CoM point mass stays inside the total (the hulls
and the point mass both take 1 / (1 + f)). RPENT_LAB_GENERATOR picks the generator (v10 by default)."""

import hashlib
import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

V11_SHA256 = "fcfab59ef2ee9168f83f93f9b2017dd1ca6f663a954fbc4b5b5a869a44287ade"      # table v11b
CLASSES_V11_SHA256 = "79608f95d3221e7413456b1cdcd143a5121c126c3fe3f53843547f3173603528"
# "breakable ceramic" in the lab, by its material rule; their real material (the products the v11 masses use)
REAL_MATERIAL = {"ycb__mug": "metal", "ycb__bowl": "metal", "ycb__bowl2": "metal", "hot3d__clay_plates": "bamboo",
                 "hot3d__pitcher": "steel"}
NAMES_V11_SHA256 = "6559c899e65d8a44994c6b51819efa1dd0242a2730f660b1641b10632fdc7062"
RENAMED = {"hot3d__foam_roller": "whiteboard eraser", "hot3d__megaphone": "vacuum jug", "hot3d__clay_plates": "spun-bamboo tray",
           "hot3d__lizard_figurine": "dinosaur figure", "hot3d__storage_box": "toothbrush holder", "hot3d__pitcher": "vase"}
TOOLS = {"handal__hammer", "handal__hammer_4", "handal__hammer_5", "handal__hammer_7", "handal__hammer_8", "hot3d__potato_masher",
         "hot3d__spatula"}
OZ = 0.028349523125
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
    assert t["version"] == "v11b" and t["unit"] == "kg" and len(t["objects"]) == 95 and MASSES_V11.endswith("masses_v11b.json")
    assert {k for k, o in t["objects"].items() if "hull_mass_kg" in o} == TOOLS
    assert all(o["mass_kg"] > 0 and o["nominal_kg_v10"] > 0 for o in t["objects"].values())
    assert t["objects"]["hot3d__dumbbell"]["mass_kg"] == 2.268 and t["objects"]["ycb__banana"]["mass_kg"] == 0.1844


def test_a_changed_table_is_refused(tmp_path):
    from robots.mujoco_lab.masses import MASSES_V11, load_masses

    changed = tmp_path / "masses_v11b.json"
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
    """v10: props.json as it is. v11: the table's mass, every hull mass x real / nominal, and the CoM and com_s of the
    hulls' volume centroids (the sim's), the rest unchanged (the 7 tools have their own hull masses:
    test_tool_part_masses); the pomegranate (its mesh does not compile, so it has no table row) is an error."""
    from robots.mujoco_lab.adapter import _props_path, hull_centroids, load_sim, object_ids, object_props
    from robots.mujoco_lab.masses import load_masses

    monkeypatch.delenv("RPENT_LAB_GENERATOR", raising=False)
    table = load_masses()["objects"]
    assert set(object_ids()) - set(table) == {"fruits_veggies__pomegranate"} and set(table) <= set(object_ids())
    v10 = {oid: object_props(oid) for oid in table}
    assert all(v10[oid] == json.load(open(_props_path(load_sim(), oid))) for oid in table)
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    for oid, e in table.items():
        if oid in TOOLS:
            continue
        p10, p11 = v10[oid], object_props(oid)
        assert p11["mass"] == e["mass_kg"] and sum(p11["hull_mass"]) == pytest.approx(e["mass_kg"], rel=1e-9)
        assert np.allclose(p11["hull_mass"], np.asarray(p10["hull_mass"]) * e["mass_kg"] / p10["mass"], rtol=1e-12, atol=0)
        keep = ("mass", "hull_mass", "com", "com_s")
        assert {k: v for k, v in p11.items() if k not in keep} == {k: v for k, v in p10.items() if k not in keep}
        m = np.asarray(p11["hull_mass"])
        com = (hull_centroids(oid) * m[:, None]).sum(0) / m.sum()
        assert np.allclose(p11["com"], com, rtol=0, atol=1e-12)
        assert p11["com_s"] == pytest.approx(((com - np.asarray(p10["centre"])) @ np.asarray(p10["axis"]) - p10["axis_lo"]) / p10["length"], abs=1e-12)
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


# ---------------------------------------------------------------- review fixes: no step mixes the two generators

def test_training_refuses_a_bundle_of_another_generator(tmp_path, monkeypatch):
    """Review (Codex 1, Opus 5): a retrain updates the manifest in place, so a v11 retrain into a v10 bundle would keep
    the v10 probe models under a v11 manifest; training refuses a bundle directory of another generator."""
    from robots.mujoco_lab.train_models import main, write_manifest
    from tests.unit_tests.skills.synthetic import synthetic_rows, with_pose

    corpus = tmp_path / "c.jsonl"
    corpus.write_text("".join(json.dumps(dict(r, generator="v11")) + "\n" for r in with_pose(synthetic_rows(300))))
    os.makedirs(tmp_path / "b")
    write_manifest(str(tmp_path / "b"), probe_models={"weigh": "probe_weigh.joblib"})        # a v10 bundle: no field
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    args = ["--corpus", str(corpus), "--out-dir", str(tmp_path / "b"), "--n-belief", "4", "--features", "nominal"]
    with pytest.raises(SystemExit, match="generator v10"):
        main(args)
    write_manifest(str(tmp_path / "b"), generator="v11")
    main(args)
    assert json.load(open(tmp_path / "b" / "manifest.json"))["generator"] == "v11"


def test_sweep_reuses_only_outputs_of_its_generator(tmp_path, monkeypatch):
    """Review (Codex 2): a cached calibrate output is reusable only under the generator it ran (no field: v10)."""
    from robots.mujoco_lab.sweep import valid
    from tests.unit_tests.robots.mujoco_lab.test_sweep import _case, _run

    expect = {"revision": "a" * 40, "sim_revision": "s" * 40, "protocol": "policy", "care": 10.0, "abstain": 3.0, "scale": 1.0,
              "seed": 12, "n": 1}
    run = _run("policy", 10.0, 3.0, 1.0, 12, [_case("a", 0, 1.0)])
    files = {}
    for name, extra in (("old", {}), ("v10", {"generator": "v10"}), ("v11", {"generator": "v11"})):
        files[name] = tmp_path / f"{name}.json"
        files[name].write_text(json.dumps(dict(run, **extra)))
    monkeypatch.delenv("RPENT_LAB_GENERATOR", raising=False)
    assert valid(str(files["old"]), expect) == [] and valid(str(files["v10"]), expect) == []
    assert any("generator" in w for w in valid(str(files["v11"]), expect))
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    assert valid(str(files["v11"]), expect) == [] and any("generator" in w for w in valid(str(files["old"]), expect))
    assert valid(str(files["v10"]), dict(expect, generator="v10")) == []


def test_paired_runs_share_a_generator():
    """Review (Codex 3): pairing a v11 run with a v10 run would read the change of masses as a protocol effect."""
    from robots.mujoco_lab.calibrate import compare
    from robots.mujoco_lab.sweep import pairs
    from tests.unit_tests.robots.mujoco_lab.test_sweep import _case, _run

    a = _run("policy", 10, 3, 1.0, 12, [_case("a", 0, 1.0)])
    b = _run("policy_no_probe", 10, 3, 1.0, 12, [_case("a", 0, 10.0)])
    assert pairs(dict(a, generator="v10"), b)["n"] == 1                                    # no field: v10
    with pytest.raises(ValueError, match="generator"):
        pairs(dict(a, generator="v11"), b)
    row = {"object": "o", "idx": 0, "cost": 1.0, "attempts": []}
    A = {"protocol": "policy", "care_scale": 1.0, "abstain_cost": 1.0, "rows": [row]}
    assert compare(A, dict(A, protocol="commit", generator="v10"), np.random.default_rng(0))["n_paired"] == 1
    with pytest.raises(ValueError, match="generator"):
        compare(dict(A, generator="v11"), A, np.random.default_rng(0))


def test_posterior_check_needs_the_bundles_prior():
    """Review (Codex 4, Opus 6): a bundle that names no prior (m1) is checked with the m2a lab prior, which mirrors
    generator v10, so a v11 bundle without a prior is refused."""
    from robots.mujoco_lab.posterior_check import check_probe_cases, check_rows

    B = {"model": None, "readout_model": None, "prior": None, "generator": "v11", "probe_models": {}}
    with pytest.raises(SystemExit, match="v10"):
        check_rows([], B, [1.0], 10, 0)
    with pytest.raises(SystemExit, match="v10"):
        check_probe_cases([], B, [1.0], 10, 0)
    assert check_rows([], dict(B, generator="v10"), [1.0], 10, 0) == []


def test_posterior_check_refuses_rows_of_another_generator(tmp_path, monkeypatch):
    """Review (Codex 5): the rows a check loads (the bundle's corpus, or its probe corpus) carry the bundle's generator."""
    import robots.mujoco_lab.posterior_check as pc

    (tmp_path / "metrics.json").write_text(json.dumps({"corpus": "c.jsonl", "split": {"val": ["a"], "train": ["a"]}}))
    B = {"manifest": {"metrics": "metrics.json", "probe_corpus": "p.jsonl"}, "generator": "v11", "prior": None,
         "readout_model": SimpleNamespace(temper=1.0), "probe_models": {"weigh": SimpleNamespace(temper=1.0)}}
    row = {"object": "a", "draw": 0, "pose": {}, "com_end": 1, "com_obj_true": [0, 0, 0], "readouts": {"rose": True}}
    monkeypatch.setattr(pc, "load_bundle", lambda path: B)
    monkeypatch.setattr(pc, "load_rows", lambda path: [row])                                # no field: v10
    for extra in ([], ["--probes", "B"]):
        with pytest.raises(SystemExit, match="generator"):
            pc.main(["--bundle", str(tmp_path), "--out", str(tmp_path / "o.json"), *extra])


def test_squeeze_sensitivity_refuses_a_probe_corpus_of_another_generator(tmp_path, monkeypatch, capsys):
    """Review (Codex 6, Opus 3): the sensitivity's probe corpus carries the bundle's generator."""
    import robots.mujoco_lab.squeeze_sensitivity as ss
    from robots.mujoco_lab.fragility import THRESHOLDS_V0

    (tmp_path / "metrics.json").write_text(json.dumps({"split": {"val": ["a"], "train": ["a"]}}))
    B = {"manifest": {"metrics": "metrics.json"}, "probe_models": {"squeeze": None}, "model": SimpleNamespace(kind="classifier"),
         "evoi": "heldout", "generator": "v11", "abstain_cost": 1.0}
    monkeypatch.setattr(ss, "load_bundle", lambda path: B)
    monkeypatch.setattr(ss, "fragility_world", lambda man: ({}, None))
    monkeypatch.setattr(ss, "load_rows", lambda path: [{"probe": "squeeze", "object": "a"}])    # no field: v10
    with pytest.raises(SystemExit):
        ss.main(["--bundle", str(tmp_path), "--probe-corpus", "p.jsonl", "--thresholds", THRESHOLDS_V0, "--out", str(tmp_path / "o.json")])
    assert "generator" in capsys.readouterr().err


def test_tool_part_masses(monkeypatch):
    """Table v11b (the user's yes to flag 1): a hammer's head hulls carry its rated head weight and the other hulls the
    rest, one density within a part; the nylon masher and spatula have one density. The CoM, com_s and the dense flags
    follow the hull masses as in the lab's prepare.py; the other fields stay."""
    from robots.mujoco_lab.adapter import hull_centroids, object_props
    from robots.mujoco_lab.masses import load_masses

    table = load_masses()["objects"]
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    for oid in TOOLS:
        e, p10, p = table[oid], object_props(oid, generator="v10"), object_props(oid)
        m, vol = np.asarray(p["hull_mass"]), np.array([h["volume"] for h in p10["hulls"]])
        head = np.zeros(len(m), dtype=bool)
        head[e["head_hulls"]] = True
        assert p["mass"] == e["mass_kg"] and m.sum() == pytest.approx(e["mass_kg"], rel=1e-12) and len(m) == len(p10["hull_mass"])
        assert p["dense"] == head.tolist() and p["heavy_end"] == e["heavy_end"]
        for part in (head, ~head):
            if part.any():
                assert np.allclose(m[part] / vol[part], (m[part] / vol[part])[0], rtol=1e-12, atol=0)
        com_s = lambda c: float(((c - np.asarray(p10["centre"])) @ np.asarray(p10["axis"]) - p10["axis_lo"]) / p10["length"])
        com = (hull_centroids(oid) * m[:, None]).sum(0) / m.sum()
        assert np.allclose(p["com"], com, rtol=0, atol=1e-12) and p["com_s"] == pytest.approx(com_s(com), abs=1e-12)
        surface = (np.array([h["centroid"] for h in p10["hulls"]]) * m[:, None]).sum(0) / m.sum()
        assert round(com_s(surface), 4) == e["com_s_v11b"]          # the table's record: prepare.py's surface centroids
        keep = ("mass", "hull_mass", "com", "com_s", "dense", "heavy_end")
        assert {k: v for k, v in p.items() if k not in keep} == {k: v for k, v in p10.items() if k not in keep}
    hammer = table["handal__hammer"]
    assert hammer["head_kg"] == round(19 * OZ, 6) and sum(np.asarray(hammer["hull_mass_kg"])[hammer["head_hulls"]]) == pytest.approx(19 * OZ)
    assert table["hot3d__spatula"]["head_hulls"] == [] and table["hot3d__spatula"]["heavy_end"] == "none"


def test_a_tool_row_must_match_its_mesh(monkeypatch):
    """A tool's hull masses must have one entry per hull of its mesh and sum to the table's mass."""
    import robots.mujoco_lab.masses as masses
    from robots.mujoco_lab.adapter import hull_centroids, object_props

    p = object_props("handal__hammer", generator="v10")
    centroids = hull_centroids("handal__hammer")
    good = masses.load_masses()
    for bad in ({"hull_mass_kg": good["objects"]["handal__hammer"]["hull_mass_kg"][:-1]},
                {"hull_mass_kg": [x * 1.01 for x in good["objects"]["handal__hammer"]["hull_mass_kg"]]}):
        table = {**good, "objects": {**good["objects"], "handal__hammer": {**good["objects"]["handal__hammer"], **bad}}}
        monkeypatch.setattr(masses, "load_masses", lambda path=None, t=table: t)
        with pytest.raises(ValueError, match="hull"):
            masses.real_props("handal__hammer", p, centroids)


@SIM
def test_v11_reset_builds_the_tool_part_masses(monkeypatch):
    """The sim's hammer has the v11b masses: with no CoM shift (the drawn com_s below the nominal), its mass is the
    table's and its CoM the CoM of the table's hull masses, 8 cm from v10's toward the head. The reference is MuJoCo's
    hull centres (volume centroids), not props["com"]: prepare.py takes trimesh's surface centroid per hull, which
    misses the sim's CoM in v10 too (0.26 mm on this hammer, up to 14 mm on others)."""
    import mujoco

    from robots.mujoco_lab.adapter import MuJoCoLabEnv, object_props

    theta = {"mu": 0.8, "mass_scale": 1.0, "com_s": 0.2, "v_lift": 0.15}
    env = MuJoCoLabEnv()
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    env.reset("handal__hammer", theta, seed=0)
    t11, p = env.truth(), object_props("handal__hammer")
    assert env.s["t3_make"].com_shift_for(env.obj, env.props, 0.2)[3] == 0.0
    assert abs(t11["mass"] - 0.837) <= len(p["hull_mass"]) * 5e-6 + 5e-5
    m = env.sim.m
    pos = np.array([m.geom_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"hull{k}")] for k in range(len(p["hull_mass"]))])
    w = np.asarray(p["hull_mass"])
    assert np.allclose(t11["com_obj"], (pos * w[:, None]).sum(0) / w.sum(), rtol=0, atol=1e-5)
    monkeypatch.delenv("RPENT_LAB_GENERATOR", raising=False)
    env.reset("handal__hammer", theta, seed=0)
    assert np.linalg.norm(np.asarray(env.truth()["com_obj"]) - np.asarray(t11["com_obj"])) > 0.05


def test_v11_classes_follow_the_real_material():
    """fragility_classes_v11.json (pinned) is fragility_classes.json but five "breakable ceramic" objects whose real
    material cannot break: the YCB mug and bowls (metal), the spun-bamboo tray and the enamelled-steel vase. Class none
    and no law; their cavities stay, so the adapter's fill check (default classes) reads the same either way."""
    from robots.mujoco_lab.fragility import CLASSES, CLASSES_V11, load_classes

    assert hashlib.sha256(open(CLASSES_V11, "rb").read()).hexdigest() == CLASSES_V11_SHA256
    old, new = load_classes(CLASSES), load_classes(CLASSES_V11)
    assert set(old) == set(new) and {k for k in old if old[k] != new[k]} == set(REAL_MATERIAL)
    for k, material in REAL_MATERIAL.items():
        assert old[k]["class"] == "breakable" and old[k]["law"] == "ceramic"
        assert new[k]["class"] == "none" and new[k]["law"] is None and material in new[k]["reason"]
        assert all(new[k].get(f) == old[k].get(f) for f in ("fillable", "cavity", "cavity_checked"))   # the tray has no cavity


def test_the_classes_default_follows_the_generator(monkeypatch):
    """With no classes file given, the corpus, probe corpus and training read the generator's: v11's under v11."""
    from robots.mujoco_lab.corpus import world
    from robots.mujoco_lab.fragility import CLASSES, CLASSES_V11, classes_for
    from robots.mujoco_lab.train_models import m2b_world

    assert classes_for("v10") == CLASSES and classes_for("v11") == CLASSES_V11
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    assert world()[0]["ycb__mug"]["class"] == "none"
    assert m2b_world(None, None, "v11")[2]["classes"]["path"] == "fragility_classes_v11.json"
    monkeypatch.delenv("RPENT_LAB_GENERATOR")
    assert world()[0]["ycb__mug"]["class"] == "breakable"
    assert m2b_world(None, None, "v10")[2]["classes"]["path"] == "fragility_classes.json"


@SIM
def test_v11_mug_rows_cannot_break(tmp_path, monkeypatch):
    """A v11 mug row draws the v10 row's latent fragility, but metal maps it to no damage force or speed."""
    from robots.mujoco_lab.corpus import main as corpus_main

    out10, out11 = tmp_path / "c10.jsonl", tmp_path / "c11.jsonl"
    monkeypatch.delenv("RPENT_LAB_GENERATOR", raising=False)
    corpus_main(["--out", str(out10), "--draws", "2", "--workers", "1", "--objects", "ycb__mug"])
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    corpus_main(["--out", str(out11), "--draws", "2", "--workers", "1", "--objects", "ycb__mug"])
    r10, r11 = [json.loads(x) for x in out10.open()], [json.loads(x) for x in out11.open()]
    assert len(r10) == len(r11) == 2
    for a, b in zip(r10, r11):
        assert a["theta_frag"]["v_break_ms"] is not None and b["theta_frag"]["z"] == a["theta_frag"]["z"]
        assert all(b["theta_frag"][k] is None for k in ("k_n_per_mm", "f_damage_n", "v_bruise_ms", "v_break_ms"))
        assert b["fill"] == a["fill"] and b.get("label") != "broken"          # a declined row has no label
        assert a["entry"]["class"] == "breakable" and b["entry"]["class"] == "none"   # the flags a planner reads


def test_the_names_map_is_pinned(tmp_path):
    """object_names_v11.json maps each of the six misnamed HOT3D ids to its old and its corrected name, with evidence."""
    from robots.mujoco_lab.names import NAMES_V11, load_names

    assert hashlib.sha256(open(NAMES_V11, "rb").read()).hexdigest() == NAMES_V11_SHA256
    m = load_names()
    assert {k: v["new"] for k, v in m.items()} == RENAMED
    assert all(v["old"] == k.split("__", 1)[1].replace("_", " ") and v["evidence"] for k, v in m.items())
    changed = tmp_path / "object_names_v11.json"
    changed.write_text(open(NAMES_V11).read().replace("vacuum jug", "thermos"))
    with pytest.raises(ValueError, match="sha256"):
        load_names(str(changed))


def test_object_names_follow_the_generator():
    """Text for an agent or a VLM: v11 uses the corrected name and an id that carries it; v10 text is the old one."""
    from robots.mujoco_lab.names import agent_object, object_name

    assert object_name("hot3d__foam_roller", "v11") == "whiteboard eraser" and object_name("hot3d__foam_roller", "v10") == "foam roller"
    assert object_name("handal__hammer_6", "v11") == object_name("handal__hammer_6", "v10") == "hammer"
    obj = {"id": "hot3d__megaphone", "name": "megaphone", "material": "hard_plastic"}
    assert agent_object(obj, "v10") == {"name": "megaphone", "id": "hot3d__megaphone", "material": "hard_plastic"}
    assert agent_object(obj, "v11") == {"name": "vacuum jug", "id": "hot3d__vacuum_jug", "material": "hard_plastic"}
    banana = {"id": "ycb__banana", "name": "banana", "material": "fruit"}
    assert agent_object(banana, "v11") == agent_object(banana, "v10") == {"name": "banana", "id": "ycb__banana", "material": "fruit"}


@SIM
def test_v11_locate_takes_the_corrected_name(monkeypatch):
    """Under v11 the agent names the object as view_env_state shows it; v10 does not know the new names."""
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    env = MuJoCoLabEnv()
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    env.reset("hot3d__foam_roller", seed=0)
    assert all(env.locate(p)["object_id"] == "hot3d__foam_roller"
               for p in ("whiteboard eraser", "hot3d__whiteboard_eraser", "foam_roller", "hot3d__foam_roller"))
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v10")
    env.reset("hot3d__foam_roller", seed=0)
    assert env.locate("whiteboard eraser") is None and env.locate("foam roller")["object_id"] == "hot3d__foam_roller"


def test_volume_centroid():
    """MuJoCo's CoM of a mesh given by its vertices: the volume centroid of their convex hull (not the surface centroid
    that prepare.py took from trimesh)."""
    from robots.mujoco_lab.masses import volume_centroid

    pyramid = [(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0), (0, 0, 1)]
    assert np.allclose(volume_centroid(pyramid), (0, 0, 0.25), rtol=0, atol=1e-12)
    cube = [(x, y, z) for x in (0, 1) for y in (0, 2) for z in (0, 3)] + [(0.5, 1, 1.5)]       # an inner point changes nothing
    assert np.allclose(volume_centroid(cube), (0.5, 1, 1.5), rtol=0, atol=1e-12)


@SIM
def test_v11_props_com_is_the_sim_com(monkeypatch):
    """Under v11 props com and com_s are the sim's (the hulls' volume centroids), so the sim reaches a drawn com_s to
    1 mm. hammer_1 and hammer_6 missed by 14 and 13 mm with prepare.py's surface centroids, which v10 keeps."""
    from robots.mujoco_lab.adapter import MuJoCoLabEnv, object_props

    env = MuJoCoLabEnv()
    th = {"mu": 0.8, "mass_scale": 1.0, "com_s": None, "v_lift": 0.15}
    for oid in ("handal__hammer_1", "handal__hammer_6"):
        monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
        p = object_props(oid)
        env.reset(oid, th, seed=0)
        t = env.truth()
        assert np.linalg.norm(np.asarray(t["com_obj"]) - np.asarray(p["com"])) < 1e-3
        assert abs(t["com_s"] - p["com_s"]) * p["length"] < 1e-3
        target = min(0.9, p["com_s"] + 0.2)
        env.reset(oid, {**th, "com_s": target}, seed=0)
        assert abs(env.truth()["com_s"] - target) * p["length"] < 1e-3
        monkeypatch.setenv("RPENT_LAB_GENERATOR", "v10")
        env.reset(oid, th, seed=0)
        assert np.linalg.norm(np.asarray(env.truth()["com_obj"]) - np.asarray(object_props(oid)["com"])) > 0.01


@SIM
def test_the_pose_carries_the_corpus_classes(tmp_path, monkeypatch):
    """The flags in a row's pose (the features and the menu read them) are the corpus's classes, as in the row: under v11
    the mug is class none in both. A classes file given to the corpus reaches the pose too, and with no corpus an agent's
    env reads the generator's file."""
    from robots.mujoco_lab.adapter import MuJoCoLabEnv
    from robots.mujoco_lab.corpus import main as corpus_main
    from robots.mujoco_lab.fragility import CLASSES
    from robots.mujoco_lab.probe_corpus import main as probe_main

    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    none = {"class": "none", "law": None, "fillable": True}
    for name, run in (("c.jsonl", corpus_main), ("p.jsonl", probe_main)):
        run(["--out", str(tmp_path / name), "--draws", "1", "--workers", "1", "--objects", "ycb__mug"])
        rows = [json.loads(x) for x in (tmp_path / name).open()]
        assert rows and all(r["entry"] == r["pose"]["entry"] == none for r in rows), name
    corpus_main(["--out", str(tmp_path / "c10.jsonl"), "--draws", "1", "--workers", "1", "--objects", "ycb__mug", "--classes", CLASSES])
    r = json.loads((tmp_path / "c10.jsonl").read_text().splitlines()[0])
    assert r["entry"] == r["pose"]["entry"] == {"class": "breakable", "law": "ceramic", "fillable": True}
    env = MuJoCoLabEnv()
    env.reset("ycb__mug", seed=0)
    assert env.pose_info()["entry"] == none
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v10")
    env.reset("ycb__mug", seed=0)
    assert env.pose_info()["entry"]["class"] == "breakable"


@SIM
def test_a_case_resets_with_the_bundles_classes(monkeypatch):
    """calibrate.run_case gives the env the bundle's classes (its manifest's fragility world), the ones its hidden
    fragility is drawn from; with no fragility world the env reads the generator's file."""
    import robots.mujoco_lab.prior as prior
    from robots.mujoco_lab.calibrate import run_case
    from robots.mujoco_lab.fragility import CLASSES, THRESHOLDS_V0, Thresholds, load_classes

    class Stop(Exception):
        pass

    class Env:
        classes = "unset"

        def reset(self, *a, **kw):
            raise Stop

    C = load_classes(CLASSES)
    for world, want in (((C, Thresholds.from_json(THRESHOLDS_V0)), C), (None, None)):
        monkeypatch.setattr(prior, "fragility_world", lambda manifest, w=world: w)
        prims = SimpleNamespace(env=Env(), manifest={})
        with pytest.raises(Stop):
            run_case(prims, "ycb__mug", 0, 0)
        assert prims.env.classes is want


def test_v11_agent_prompt_uses_the_corrected_name(tmp_path, monkeypatch):
    """The run's task text and recipe tag (the audit file name the agent sees) carry the agent id under v11; the record
    keeps the internal id; v10 is unchanged."""
    import argparse

    from robots.mujoco_lab.prompts.user import TASK
    from robots.mujoco_lab.robot_spec import _parse_config

    args = argparse.Namespace(object="hot3d__foam_roller", destination="bin", care="normal", seed=0, output_dir=str(tmp_path),
                              memory_dir=str(tmp_path))
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    cfg = _parse_config(args)
    assert cfg.prompt_vars["object"] == "hot3d__whiteboard_eraser" and cfg.recipe_tag == "hot3d__whiteboard_eraser_bin_s0"
    assert cfg.task_desc["object"] == "hot3d__foam_roller"
    text = TASK.format(**cfg.prompt_vars)
    assert "foam" not in text and "whiteboard eraser" in text.replace("_", " ")
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v10")
    cfg = _parse_config(args)
    assert cfg.prompt_vars["object"] == "hot3d__foam_roller" and cfg.recipe_tag == "hot3d__foam_roller_bin_s0"


@SIM
def test_v11_spread_shrinks_around_the_v11_range(monkeypatch):
    """The H3 spread under v11 (e1-v11, the merge of real-masses): the truth's mass scale shrinks around the log-midpoint
    of V11_MASS_SCALE_RANGE, the range the v11 prior shrinks, so the narrow prior stays quantile-coupled to the narrow
    truth (a filled container's range is the same under both generators)."""
    from robots.mujoco_lab.adapter import MuJoCoLabEnv
    from robots.mujoco_lab.corpus import FILLED_MASS_SCALE_RANGE, V11_MASS_SCALE_RANGE, shrink_theta, shrunk_range
    from robots.mujoco_lab.fragility import CLASSES_V11, DATA, Thresholds, load_classes
    from robots.mujoco_lab.prior import make_prior

    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")

    def scale(x, filled=False):
        return shrink_theta({"mass_scale": x, "mu": 0.5, "com_s": None}, 0.25, filled)["mass_scale"]

    assert [scale(x) for x in V11_MASS_SCALE_RANGE] == pytest.approx(list(shrunk_range(V11_MASS_SCALE_RANGE, 0.25, log=True)))
    assert scale(1.0) == pytest.approx(1.0)                                   # the v11 log-midpoint stays
    assert [scale(x, True) for x in FILLED_MASS_SCALE_RANGE] == pytest.approx(list(shrunk_range(FILLED_MASS_SCALE_RANGE, 0.25, log=True)))
    prior = make_prior(load_classes(CLASSES_V11), Thresholds.from_json(os.path.join(DATA, "thresholds_v1.json")), generator="v11")
    env = MuJoCoLabEnv()
    for oid in ("ycb__banana", "ycb__mug"):
        env.reset(oid, seed=0)
        wide, narrow = (prior(env.pose_info(), 500, np.random.default_rng(2), spread=f).particles for f in (1.0, 0.25))
        for w, n in zip(wide, narrow, strict=True):
            assert n["mass_scale"] == pytest.approx(scale(w["mass_scale"], w["fill"] > 0), rel=1e-12), oid
