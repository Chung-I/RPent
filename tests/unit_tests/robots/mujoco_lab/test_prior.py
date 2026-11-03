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


"""The lab prior: CoM hypotheses of the corpus generator against the sim (needs the sim)."""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")

LONG = "ycb__banana"


def test_com_hypothesis_matches_the_sim():
    from robots.mujoco_lab.adapter import MuJoCoLabEnv
    from robots.mujoco_lab.prior import com_hypothesis

    env = MuJoCoLabEnv()
    for end in (+1, -1):
        env.reset(LONG, {"mu": 0.8, "mass_scale": 1.3, "com_s": 0.8, "v_lift": 0.15}, seed=0, com_end=end)
        pose, truth = env.pose_info(), env.truth()
        com, f = com_hypothesis(pose["pivot"], pose["com_obj_nominal"], 0.8, end)
        assert f > 0
        assert np.linalg.norm(np.asarray(com) - np.asarray(truth["com_obj"])) < 1e-3
        assert abs(pose["nominal_mass"] * 1.3 * (1 + f) - truth["mass"]) < 0.01 * truth["mass"]
    piv = pose["pivot"]
    s_nom = float(((np.asarray(pose["com_obj_nominal"]) - np.asarray(piv["centre"])) @ np.asarray(piv["axis"]) - piv["lo"]) / piv["L"])
    com, f = com_hypothesis(piv, pose["com_obj_nominal"], s_nom - 0.05, +1)         # a target below the nominal fraction
    assert f == 0.0 and np.allclose(com, pose["com_obj_nominal"])
    com, f = com_hypothesis(piv, pose["com_obj_nominal"], None, +1)
    assert f == 0.0 and np.allclose(com, pose["com_obj_nominal"])


def test_lab_prior_matches_the_generator():
    from scipy.stats import ks_2samp

    from robots.mujoco_lab.adapter import MuJoCoLabEnv
    from robots.mujoco_lab.corpus import draw_theta
    from robots.mujoco_lab.prior import lab_prior

    env = MuJoCoLabEnv()
    env.reset(LONG, seed=0)
    pose = env.pose_info()
    b = lab_prior(pose, 20000, np.random.default_rng(1))
    A = b.arrays()
    rng = np.random.default_rng(2)
    draws = [draw_theta(rng, pose["nominal_mass"])[0] for _ in range(20000)]
    assert ks_2samp(np.log(A["mass_scale"]), np.log([t["mass_scale"] for t in draws])).pvalue > 0.01
    assert ks_2samp(A["mu"], [t["mu"] for t in draws]).pvalue > 0.01
    assert abs(A["com_branch"].mean() - 0.5) < 0.02
    assert b.p_com_shifted() == pytest.approx(float((A["f"] > 0).mean()))
    assert b.p_com_shifted() < A["com_branch"].mean()                      # some banana targets lie below the nominal CoM
    assert np.allclose(A["mass"], pose["nominal_mass"] * A["mass_scale"] * (1 + A["f"]), rtol=0, atol=1e-9)
    assert b.ess() == pytest.approx(20000) and b.n_ancestors() == 20000


def test_particle_features_match_extract_with_the_true_com():
    from dataclasses import replace

    from robots.mujoco_lab.adapter import MuJoCoLabEnv
    from rpent.skills.pick_place.features import extract, particle_features

    env = MuJoCoLabEnv()
    env.reset(LONG, {"mu": 0.6, "mass_scale": 1.4, "com_s": 0.8, "v_lift": 0.15}, seed=0, com_end=-1)
    p = env.proposals()[1]
    pose, truth, g = env.proposal_pose(p), env.truth(), env.geometry()
    F = 15.0
    base = extract(g, np.asarray(p.point), p.yaw, 0.1, 0.5, F, 0.15, 0.5, "plate")      # placeholder mass and friction
    got = particle_features(base, {"mass": truth["mass"], "mu": truth["mu"], "com_obj": truth["com_obj"]}, pose, F)
    com = np.asarray(truth["com"])
    g_true = replace(g, com=com, com_height=float(com[2] - env.sim.obj_low_z()))
    want = extract(g_true, np.asarray(p.point), p.yaw, truth["mass"], truth["mu"], F, 0.15, 0.5, "plate")
    for k in ("lever", "com_h", "base_ratio", "com_height_ratio", "mass", "mu", "load_ratio"):
        assert got[k] == pytest.approx(want[k], abs=1e-6), k
    assert abs(got["lever"] - base["lever"]) > 0.005                        # the shift moved the lever


def test_prior_matches_the_generator():
    """make_prior against the joint generator (draw_theta, draw_action, then draw_damage and apply_damage, the mass and
    CoM of prior.joint_hidden) on 20,000 draws each: the mug's mass (container and liquid), fill and CoM height along
    its cavity axis; the jello box's damage force."""
    from scipy.stats import ks_2samp

    from robots.mujoco_lab.adapter import MuJoCoLabEnv
    from robots.mujoco_lab.corpus import apply_damage, case_rng, damage_rng, draw_action, draw_damage, draw_theta
    from robots.mujoco_lab.fragility import THRESHOLDS_V0, Thresholds, entry_of, load_classes, map_latent
    from robots.mujoco_lab.prior import joint_hidden, make_prior

    classes, th = load_classes(), Thresholds.from_json(THRESHOLDS_V0)
    prior = make_prior(classes, th)
    env = MuJoCoLabEnv()
    n = 20000
    for oid in ("ycb__mug", "ycb__jello"):
        env.reset(oid, seed=0)
        pose = env.pose_info()
        assert pose["object_id"] == oid
        entry, _ = entry_of(oid, classes)
        b = prior(pose, n, np.random.default_rng(1))
        P = b.particles
        gen = []
        for i in range(n):
            rng = case_rng(3, oid, i)
            theta, com_end = draw_theta(rng, pose["nominal_mass"])
            act = draw_action(rng, 10.0)
            d = draw_damage(damage_rng(3, oid, i), entry, th, 10.0)
            theta, act = apply_damage(theta, act, d)
            gen.append((joint_hidden(pose, entry, theta, com_end, d["fill"]), d))
        if oid == "ycb__mug":
            axis = np.asarray(entry["cavity"]["axis_obj"])
            assert ks_2samp([p["mass"] for p in P], [h["mass"] for h, _ in gen]).pvalue > 0.01
            assert ks_2samp([p["fill"] for p in P], [d["fill"] for _, d in gen]).pvalue > 0.01
            assert ks_2samp([np.asarray(p["com_obj"]) @ axis for p in P], [np.asarray(h["com_obj"]) @ axis for h, _ in gen]).pvalue > 0.01
            assert np.allclose([p["mass"] for p in P], [pose["nominal_mass"] * p["mass_scale"] * (1 + p["f"]) + p["liquid_mass"] for p in P])
            assert all((p["spill_angle_deg"] is None) == (p["fill"] == 0.0) for p in P)
        else:
            assert ks_2samp(np.log([p["f_damage_n"] for p in P]), np.log([d["theta_frag"]["f_damage_n"] for _, d in gen])).pvalue > 0.01
            assert all(p["fill"] == 0.0 for p in P)
        for p in P[:20]:                                   # every particle's fragility is its latent draw under the law
            want = map_latent(p["frag_z"], entry, th)
            assert all(want[k] == pytest.approx(p[k]) if want[k] is not None else p[k] is None
                       for k in ("k_n_per_mm", "f_damage_n", "v_bruise_ms", "v_break_ms"))


def test_prior_from_manifest(tmp_path):
    """A bundle's manifest names its prior: "lab_generator" gives lab_prior; the m2b prior gives the joint prior on the
    named class and threshold files (relative paths under data/), each checked by its sha256; no prior gives None."""
    import shutil

    from robots.mujoco_lab.adapter import MuJoCoLabEnv
    from robots.mujoco_lab.fragility import THRESHOLDS_V0
    from robots.mujoco_lab.prior import DATA, M2B_PRIOR, lab_prior, prior_from_manifest, sha256_of

    assert prior_from_manifest({"prior": "lab_generator"}) is lab_prior and prior_from_manifest({}) is None
    man = {"prior": M2B_PRIOR, "classes": {"path": "fragility_classes.json", "sha256": sha256_of(f"{DATA}/fragility_classes.json")},
           "thresholds": {"path": THRESHOLDS_V0, "sha256": sha256_of(THRESHOLDS_V0)}}
    env = MuJoCoLabEnv()
    env.reset("ycb__mug", seed=0)
    P = prior_from_manifest(man)(env.pose_info(), 400, np.random.default_rng(0)).particles
    assert 0 < sum(p["fill"] > 0 for p in P) < 400 and all("f_damage_n" in p for p in P)
    changed = tmp_path / "thresholds.json"
    shutil.copy(THRESHOLDS_V0, changed)
    changed.write_text(changed.read_text().replace('"slosh_zeta": 0.01', '"slosh_zeta": 0.02'))
    with pytest.raises(ValueError, match="sha256"):
        prior_from_manifest({**man, "thresholds": {"path": str(changed), "sha256": sha256_of(THRESHOLDS_V0)}})


def test_weigh_narrows_the_fill():
    """A mug filled to 0.6 and a toy weigh law (log sd 0.05), 200 seeded cases under the m2b prior: after one weigh the
    80 % interval of the fill among the filled particles is narrower than 0.3 and holds 0.6 in at least 75 % of the
    cases, and P(filled) rises from the prior's 0.5. An empty mug with a mass scale up to 3 weighs what a filled light
    one weighs, so an interval over every particle spans the empty atom: P(filled) and the filled cases are scored
    apart (plan 2b, Task 9 (b))."""
    from robots.mujoco_lab.adapter import MuJoCoLabEnv
    from robots.mujoco_lab.fragility import THRESHOLDS_V0, Thresholds, entry_of, liquid, load_classes
    from robots.mujoco_lab.prior import FILLED_MASS_SCALE_RANGE, make_prior
    from rpent.skills.pick_place.belief import ParticleBelief
    from rpent.skills.pick_place.likelihood import Law
    from rpent.skills.pick_place.probes import PROBE_KINDS, CalibratedClassifier, ProbeModel

    classes = load_classes()
    prior = make_prior(classes, Thresholds.from_json(THRESHOLDS_V0))
    env = MuJoCoLabEnv()
    env.reset("ycb__mug", seed=0)
    pose = env.pose_info()
    m_liquid = liquid(entry_of("ycb__mug", classes)[0]["cavity"], 0.6)["mass_kg"]
    clf = CalibratedClassifier(PROBE_KINDS["weigh"], None, None, np.array([1.0, 0.0, 0.0]))
    pm = ProbeModel(probe="weigh", classes=PROBE_KINDS["weigh"], clf=clf, reading={"measured": Law.normal(0.0, 0.05)}, p_thrown=0.0)
    lo, hi = FILLED_MASS_SCALE_RANGE
    widths, hits, p_filled = [], [], []
    for seed in range(200):
        rng = np.random.default_rng(seed)
        b = prior(pose, 2000, rng)
        m_true = pose["nominal_mass"] * float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) + m_liquid
        obs = {"probe": "weigh", "kind": "measured", "reading": {"m_hat": m_true * float(np.exp(rng.normal(0.0, 0.05)))}}
        a = b.arrays()
        ll, _ = pm.loglik(obs, np.zeros((len(b), 3)), a, 20.0)
        post = b.reweight(ll)
        filled = a["fill"] > 0
        p_filled.append(float(post.weights[filled].sum() / post.weights.sum()))
        sub = ParticleBelief([p for p, f in zip(post.particles, filled) if f], post.weights[filled] / post.weights[filled].sum())
        q10, q90 = sub.quantile("fill", 0.1), sub.quantile("fill", 0.9)
        widths.append(q90 - q10)
        hits.append(q10 <= 0.6 <= q90)
    assert max(widths) < 0.3 and np.mean(hits) >= 0.75, (max(widths), np.mean(hits))
    assert np.median(p_filled) > 0.6


def test_the_narrow_prior_matches_the_narrow_truth():
    """plan H3 decision 4, quantile-coupled, on a fillable object (the mug) and a rigid one (the banana): the m2b prior's
    particles at f = 0.25 are shrink_theta of its particles at f = 1 from the same stream (mass scale, mu, com_s; the
    fill, fragility latent, CoM end and branch unchanged), and 2,000 narrow case draws are shrink_theta of the wide ones
    (test_shrink_keeps_the_case_stream), so each KS comparison of the prior with the draws (mass scale empty and filled
    apart, mu, com_s) equals milestone 4's comparison of the wide ones, statistic and p, whose agreement
    test_prior_matches_the_generator checks; every narrow particle and draw lies in the narrow ranges; lab_prior too."""
    from scipy.stats import ks_2samp

    from robots.mujoco_lab.adapter import MuJoCoLabEnv, object_props
    from robots.mujoco_lab.calibrate import case_draw
    from robots.mujoco_lab.corpus import shrink_theta
    from robots.mujoco_lab.fragility import THRESHOLDS_V0, Thresholds, load_classes
    from robots.mujoco_lab.prior import lab_prior, make_prior

    hidden = ("mass_scale", "mu", "com_s")

    def coupled(wide, narrow):
        for w, n in zip(wide, narrow, strict=True):
            want = shrink_theta({k: w[k] for k in hidden}, 0.25, w.get("fill", 0.0) > 0)
            assert all(n[k] is None if want[k] is None else n[k] == pytest.approx(want[k], rel=1e-12) for k in hidden)
            assert all(n[k] == w[k] for k in ("fill", "frag_z", "com_end", "com_branch") if k in w)

    def inside(x, filled):
        lo, hi = (0.9457, 1.0574) if filled else (0.9789, 1.5323)
        com = x["com_s"] is None or 0.4875 <= x["com_s"] <= 0.6125
        return lo <= x["mass_scale"] <= hi and 0.4 <= x["mu"] <= 0.6 and com

    classes, th = load_classes(), Thresholds.from_json(THRESHOLDS_V0)
    prior, env = make_prior(classes, th), MuJoCoLabEnv()
    for oid in ("ycb__mug", "ycb__banana"):
        env.reset(oid, seed=0)
        pose, mass = env.pose_info(), object_props(oid)["mass"]
        wide, narrow = (prior(pose, 4000, np.random.default_rng(2), spread=f).particles for f in (1.0, 0.25))
        coupled(wide, narrow)
        tw, tn = ([case_draw(3, oid, i, mass, False, fragility=(classes, th), spread=f) for i in range(2000)] for f in (1.0, 0.25))
        compared = 0
        for full in (False, True):
            for k in hidden:
                pw, pn = ([x[k] for x in P if (x["fill"] > 0) == full and x[k] is not None] for P in (wide, narrow))
                cw, cn = ([c[0][k] for c in T if (c[3]["fill"] > 0) == full and c[0][k] is not None] for T in (tw, tn))
                if not (pw and cw):
                    assert not (pn or cn)
                    continue
                a, b = ks_2samp(pn, cn), ks_2samp(pw, cw)
                assert (a.statistic, a.pvalue) == pytest.approx((b.statistic, b.pvalue), rel=1e-9)
                compared += 1
        assert compared == (5 if oid == "ycb__mug" else 3)
        assert all(inside(x, x["fill"] > 0) for x in narrow) and all(inside(c[0], c[3]["fill"] > 0) for c in tn)
    wide, narrow = (lab_prior(env.pose_info(), 2000, np.random.default_rng(3), spread=f).particles for f in (1.0, 0.25))
    coupled(wide, narrow)
    assert all(inside(x, False) for x in narrow)
