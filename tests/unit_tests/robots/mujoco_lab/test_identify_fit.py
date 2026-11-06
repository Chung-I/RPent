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


"""identify_fit: the Real2Sim fit's likelihood on the particles (gate R1's update), the identify probe model, and the
identify rows mapped onto the weigh cases."""

import json
import os

import numpy as np
import pytest

SIM = os.environ.get("RPENT_MUJOCO_SIM_DIR", "")


def _fit(p=(0.1, 0.0, 0.0), R=None, phi=(2.0, 0.16, 0.02, 0.06), info=None, floor=(0.001, 0.0005)):
    R = np.eye(3) if R is None else np.asarray(R, dtype=float)
    f = {"obj_l7": {"p": list(p), "R": R.ravel().tolist()}, "phi": list(phi) + [0.0] * 6,
         "info": (np.eye(4) if info is None else np.asarray(info, dtype=float)).tolist()}
    if floor is not None:
        f["floor"] = list(floor)
    return f


RZ90 = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]      # (x, y, z) -> (-y, x, z)


def test_fit_parts_put_each_particles_com_in_link_7s_frame():
    from robots.mujoco_lab.identify_fit import fit_parts

    arrays = {"mass": np.array([2.0, 1.0]), "com_obj": np.array([[0.01, 0.02, 0.03], [0.0, 0.0, 0.0]])}
    parts = fit_parts(_fit(R=RZ90), arrays)
    assert parts.shape == (2, 4)
    np.testing.assert_allclose(parts[0], [2.0, 2.0 * (0.1 - 0.02), 2.0 * 0.01, 2.0 * 0.03])
    np.testing.assert_allclose(parts[1], [1.0, 0.1, 0.0, 0.0])


def test_fit_loglik_is_the_information_form_gaussian_on_mass_and_first_moments():
    from robots.mujoco_lab.identify_fit import fit_loglik

    info = np.diag([4.0, 100.0, 100.0, 100.0])
    arrays = {"mass": np.array([2.0, 3.0]), "com_obj": np.array([[-0.02, 0.01, 0.03], [-0.02, 0.01, 0.03]])}
    ll = fit_loglik(_fit(info=info), arrays)       # particle 0: (2, 0.16, 0.02, 0.06), the fit's phi
    assert ll[0] == pytest.approx(0.0, abs=1e-12)
    e = np.array([1.0, 3.0 * 0.08 - 0.16, 3.0 * 0.01 - 0.02, 3.0 * 0.03 - 0.06])
    assert ll[1] == pytest.approx(-0.5 * e @ info @ e)


def test_a_fit_without_the_model_error_floor_is_refused():
    from robots.mujoco_lab.identify_fit import fit_loglik

    arrays = {"mass": np.array([1.0]), "com_obj": np.zeros((1, 3))}
    with pytest.raises(ValueError, match="floor"):
        fit_loglik(_fit(floor=None), arrays)


@pytest.mark.skipif(not (SIM and os.path.exists(os.path.join(SIM, "ident.py"))), reason="not the arm sim")
def test_fit_loglik_equals_ident_loglik_on_gate_r1s_particle_moments():
    from robots.mujoco_lab.adapter import load_sim
    from robots.mujoco_lab.identify_fit import fit_loglik

    ident = load_sim()["ident"]
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4))
    fit = _fit(p=(0.01, -0.02, 0.1), R=RZ90, phi=(0.5, 0.01, 0.0, 0.05), info=A @ A.T + np.eye(4))
    arrays = {"mass": rng.uniform(0.2, 1.0, 50), "com_obj": rng.normal(scale=0.03, size=(50, 3))}
    p, R = np.asarray(fit["obj_l7"]["p"]), np.asarray(fit["obj_l7"]["R"]).reshape(3, 3)     # r1_accuracy.belief_estimates
    c_l7 = p[None, :] + arrays["com_obj"] @ R.T
    parts = np.column_stack([arrays["mass"], arrays["mass"][:, None] * c_l7])
    want = ident.loglik(np.asarray(fit["phi"])[:4], np.asarray(fit["info"]), parts)
    np.testing.assert_allclose(fit_loglik(fit, arrays), want, rtol=0, atol=1e-12)


def test_an_attempt_without_a_fit_keeps_the_prior():
    from robots.mujoco_lab.identify_fit import attempt_fit_loglik

    arrays = {"mass": np.array([1.0, 2.0]), "com_obj": np.zeros((2, 3))}
    for row in ({"readouts": {}}, {"readouts": None}, {}, {"readouts": {"joint": None}}):
        np.testing.assert_array_equal(attempt_fit_loglik(row, arrays), [0.0, 0.0])
    ll = attempt_fit_loglik({"readouts": {"joint": _fit()}}, arrays)
    assert ll[0] != ll[1]


def test_the_identify_model_reads_a_measured_fit_and_nothing_else():
    from robots.mujoco_lab.identify_fit import MISSING, IdentifyFit, fit_loglik

    arrays = {"mass": np.array([2.0, 3.0]), "com_obj": np.zeros((2, 3))}
    pm = IdentifyFit()
    for obs in ({"kind": "lost", "reading": _fit()}, {"kind": "not_lifted", "reading": {}}, {"kind": "damaged", "reading": _fit()},
                {"kind": "unplanned", "reading": {}}, {"kind": MISSING, "reading": None}, {"kind": "measured", "reading": {}}):
        ll, info = pm.loglik(obs, None, arrays, 5.0)
        np.testing.assert_array_equal(ll, [0.0, 0.0])
        assert info == {"used": []}
    ll, info = pm.loglik({"kind": "measured", "reading": _fit()}, None, arrays, 5.0, thrown=False)
    np.testing.assert_array_equal(ll, fit_loglik(_fit(), arrays))
    assert info == {"used": ["fit"]}


def _weigh(obj, draw, s=0.5):
    return {"key": f"{obj}__v11s0__d{draw}__Bweigh", "object": obj, "draw": draw, "seed": 0, "probe": "weigh", "kind": "hold",
            "reading": {"fz_hold_n": 3.0}, "force": 4.0, "s": s, "pose": {"pivot": {"L": 0.2}, "tag": "weigh"},
            "base": {"b": 1.0}, "theta": {"mass_scale": 1.1}, "com_end": 1, "fill": 0.0, "theta_frag": None, "f_std": 6.0}


def _ident(obj, draw, s=0.45, **kw):
    return {**_weigh(obj, draw), "key": f"{obj}__v11s0__d{draw}__Bidentify", "probe": "identify", "kind": "measured",
            "reading": _fit(), "force": 6.0, "s": s, "pose": {"pivot": {"L": 0.2}, "tag": "identify"}, "base": {"b": 2.0}, **kw}


def test_identify_cases_follow_the_weigh_cases_and_fill_the_gaps():
    from robots.mujoco_lab.identify_fit import MISSING, identify_cases

    weigh = [[_weigh("a", 0)], [_weigh("a", 1)], [_weigh("b", 0)]]
    declined = {k: v for k, v in _ident("b", 0).items() if k not in ("pose", "s", "base", "reading", "kind", "probe")}
    out = identify_cases(weigh, [{**declined, "declined": "no proposal"}, _ident("a", 1)])
    assert [len(c) for c in out] == [1, 1, 1]
    a0, a1, b0 = (c[0] for c in out)
    assert a0["kind"] == MISSING and a0["reading"] is None and a0["missing_reason"] is None
    assert a0["key"] == "a__v11s0__d0__Bidentify" and a0["probe"] == "identify" and a0["pose"] == weigh[0][0]["pose"]
    assert b0["kind"] == MISSING and b0["missing_reason"] == "no proposal"
    assert a1["kind"] == "measured" and a1["reading"] == _fit() and a1["key"].endswith("__Bidentify")


def test_an_identify_case_keeps_the_weigh_cases_scene_and_its_own_pinch_apart():
    """The prior's input and the scene of the menus (pose, pinch, base) are the weigh case's; the identify's own stay."""
    from robots.mujoco_lab.identify_fit import identify_cases

    w = _weigh("a", 1, s=0.5)
    (c,), = identify_cases([[w]], [_ident("a", 1, s=0.45)])
    assert c["pose"] == w["pose"] and c["s"] == 0.5 and c["base"] == w["base"]
    assert c["pose_identify"]["tag"] == "identify" and c["s_identify"] == 0.45 and c["base_identify"] == {"b": 2.0}


def test_an_identify_row_with_other_draws_than_its_weigh_case_is_refused():
    from robots.mujoco_lab.identify_fit import identify_cases

    for change in ({"theta": {"mass_scale": 1.2}}, {"com_end": 0}, {"fill": 0.3}, {"theta_frag": {"k": 1.0}}, {"seed": 1}):
        with pytest.raises(ValueError, match="draws"):
            identify_cases([[_weigh("a", 1)]], [_ident("a", 1, **change)])


def test_identify_rows_read_the_jsonl(tmp_path):
    from robots.mujoco_lab.identify_fit import identify_rows

    p = tmp_path / "identify_rows.jsonl"
    p.write_text(json.dumps(_ident("a", 0)) + "\n\n" + json.dumps(_ident("a", 1)) + "\n")
    assert [r["draw"] for r in identify_rows(str(p))] == [0, 1]


def _bundle(tmp_path, rows):
    """A bundle directory whose manifest names a probe corpus of ``rows``."""
    (tmp_path / "pc.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "manifest.json").write_text(json.dumps({"probe_corpus": str(tmp_path / "pc.jsonl")}))
    return str(tmp_path)


def _scene_d(r):
    """A probe corpus's identify row (probe_corpus scene D, m2b2_v12b): the key tag Didentify."""
    return {**r, "key": r["key"].rsplit("__", 1)[0] + "__Didentify"}


def test_a_bundles_own_identify_rows_come_first_and_keep_their_declines(tmp_path):
    """A bundle built with the identify probe (v12b) brings its own rows, tagged as its probe corpus's scene D tags them,
    so its rerun is a bundle-path change only."""
    from robots.mujoco_lab.identify_fit import bundle_identify_rows, identify_source

    declined = {"key": "a__v11s0__d0__Didentify", "object": "a", "draw": 0, "declined": "no proposal"}
    own = _bundle(tmp_path, [_weigh("a", 0), declined, _scene_d(_ident("a", 1)), _ident("a", 2)])   # d2: scene B's tag
    assert [r["draw"] for r in bundle_identify_rows(own)] == [0, 1]
    ext = tmp_path / "ext.jsonl"
    ext.write_text(json.dumps(_ident("a", 0)) + "\n")
    rows, src = identify_source(own, str(ext))
    assert src == "bundle" and [r["draw"] for r in rows] == [0, 1]


def test_the_bundle_tag_is_the_posterior_checks_tag_of_probe_set_d():
    """One tag for a bundle's identify rows: the one posterior_check judges as probe set D."""
    from robots.mujoco_lab.identify_fit import BUNDLE_TAG
    from robots.mujoco_lab.posterior_check import PROBE_TAGS

    assert PROBE_TAGS["D"] == (BUNDLE_TAG,)


def test_a_bundle_without_identify_rows_reads_the_simulated_rows_or_refuses(tmp_path):
    from robots.mujoco_lab.identify_fit import bundle_identify_rows, identify_source

    (tmp_path / "b").mkdir()
    b = _bundle(tmp_path / "b", [_weigh("a", 0), _weigh("a", 1)])
    assert bundle_identify_rows(b) == []
    ext = tmp_path / "ext.jsonl"
    ext.write_text(json.dumps(_ident("a", 0)) + "\n")
    rows, src = identify_source(b, str(ext))
    assert src == str(ext) and [r["draw"] for r in rows] == [0]
    with pytest.raises(ValueError, match="--identify"):
        identify_source(b, None)


def test_the_bundles_identify_model_comes_before_gate_r1s_fit_likelihood():
    from robots.mujoco_lab.identify_fit import IdentifyFit, identify_model

    mine = object()
    assert identify_model({"probe_models": {"weigh": 1, "identify": mine}}) is mine
    assert isinstance(identify_model({"probe_models": {"weigh": 1}}), IdentifyFit)


def test_a_probe_model_reads_an_identify_case_at_its_own_pinch():
    """The prior and the menus take the weigh case's scene; a probe model's features take the identify's own."""
    from robots.mujoco_lab.identify_fit import identify_cases, own_scene

    (c,), = identify_cases([[_weigh("a", 1, s=0.5)]], [_ident("a", 1, s=0.45)])
    r = own_scene(c)
    assert r["pose"]["tag"] == "identify" and r["s"] == 0.45 and r["base"] == {"b": 2.0} and r["force"] == 6.0
    w = _weigh("a", 1)
    assert own_scene(w) == w


def test_a_run_under_another_generator_than_the_bundles_is_refused(tmp_path, monkeypatch):
    """The draws and the truths follow the lab generator: a bundle built under v11 refuses a run under v10 (an unset
    RPENT_LAB_GENERATOR); a bundle that records no generator is not checked."""
    from robots.mujoco_lab.identify_fit import require_bundle_generator

    (tmp_path / "manifest.json").write_text(json.dumps({"generator": "v11"}))
    monkeypatch.delenv("RPENT_LAB_GENERATOR", raising=False)
    with pytest.raises(SystemExit, match="generator v11"):
        require_bundle_generator(str(tmp_path))
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    assert require_bundle_generator(str(tmp_path)) == "v11"
    (tmp_path / "manifest.json").write_text(json.dumps({}))
    monkeypatch.delenv("RPENT_LAB_GENERATOR", raising=False)
    assert require_bundle_generator(str(tmp_path)) == "v10"
