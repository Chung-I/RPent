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


"""First-menu decisions under each prior: the change rate against the class prior."""

import os

import pytest


def _m(rec, opt, guard=False):
    return {"recommendation": rec, "recommended_option": opt, "guard_flag": guard}


def test_change_rates_count_changed_recommendations_against_the_class_prior():
    from robots.mujoco_lab.first_menus import change_rates

    recs = [{"object": "a", "idx": 0, "setting": "s", "menus": {"class": _m("commit", "x"), "vlm": _m("commit", "y")}},
            {"object": "b", "idx": 0, "setting": "s", "menus": {"class": _m("probe", "p"), "vlm": _m("decline", "decline")}},
            {"object": "b", "idx": 1, "setting": "t", "menus": {"class": _m("commit", "x"), "vlm": _m("commit", "x")}}]
    r = change_rates(recs, n_boot=50, seed=0)
    v = r["s"]["vlm"]
    assert v["n"] == 2 and v["recommendation"]["rate"] == 0.5 and v["option"]["rate"] == 1.0
    assert v["transitions"] == {"probe->decline": 1} and len(v["recommendation"]["ci"]) == 2
    assert r["t"]["vlm"]["recommendation"]["rate"] == 0.0 and "class" not in r["s"]


def test_change_rates_leave_out_a_case_with_an_error_under_either_prior():
    from robots.mujoco_lab.first_menus import change_rates

    recs = [{"object": "a", "idx": 0, "setting": "s", "menus": {"class": _m("commit", "x"), "vlm": {"error": "boom"}}},
            {"object": "b", "idx": 0, "setting": "s", "menus": {"class": _m("commit", "x"), "vlm": _m("decline", "decline")}}]
    v = change_rates(recs, n_boot=20, seed=0)["s"]["vlm"]
    assert v["n"] == 1 and v["n_left_out"] == 1 and v["recommendation"]["rate"] == 1.0


M2B2 = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "robots", "mujoco_lab", "data", "m2b2")
POLICY = os.path.join(M2B2, "calibration_crossfit_policy_care_10_abstain_10_time_1.json")


@pytest.mark.skipif(not os.path.exists(POLICY) or not os.environ.get("RPENT_MUJOCO_SIM_DIR"),
                    reason="the m2b2 bundle's crossfit run or the lab's sim dir is not here")
def test_class_menus_repeat_the_recorded_crossfit_policy_run():
    """Same fold bundle, case draw and menu stream: the class prior's first menu at care 10, abstain cost 10 equals the
    first menu of calibrate's crossfit policy run (seed 11)."""
    import json

    from robots.mujoco_lab.first_menus import run_object

    folds = json.load(open(os.path.join(M2B2, "folds", "folds.json")))
    o = "handal__hammer_2"
    s = {"bundle": os.path.join(M2B2, "folds", f"fold_{folds[o]}"), "fold": folds[o], "seed": 11,
         "settings": [("k10_A10", 10.0, 10.0)], "tables": {}}
    rows = run_object((o, [1, 2], s))
    recorded = {r["idx"]: r for r in json.load(open(POLICY))["rows"] if r["object"] == o}
    assert len(rows) == 2
    for r in rows:
        m, c = r["menus"]["class"], recorded[r["idx"]]
        assert (m["recommendation"], m["best_option"]) == (c["recommendation"], c["best_option"])
        assert r["destination"] == c["destination"]


@pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")
def test_a_case_resets_with_the_bundles_classes(monkeypatch):
    """masscom-v11: run_object resets as calibrate.run_case does: the env takes the bundle's classes (its manifest's
    fragility world), the ones the case's hidden fragility is drawn from; with no fragility world the env reads the
    generator's file."""
    from types import SimpleNamespace

    import robots.mujoco_lab.calibrate as cal
    import robots.mujoco_lab.prior as prior
    from robots.mujoco_lab import first_menus
    from robots.mujoco_lab.fragility import CLASSES, THRESHOLDS_V0, Thresholds, load_classes

    class Stop(Exception):
        pass

    class Env:
        classes = "unset"

        def reset(self, *a, **kw):
            raise Stop

    C = load_classes(CLASSES)
    for world, want in (((C, Thresholds.from_json(THRESHOLDS_V0)), C), (None, None)):
        prims = SimpleNamespace(env=Env(), manifest={}, prior=None)
        monkeypatch.setattr(cal, "_primitives", lambda *a, p=prims: p)
        monkeypatch.setattr(cal, "job_force_noise", lambda *a: None)
        monkeypatch.setattr(prior, "fragility_world", lambda manifest, w=world: w)
        with pytest.raises(Stop):
            first_menus.run_object(("ycb__mug", [0], {"bundle": "b", "tables": {}, "seed": 11, "settings": [], "fold": 0}))
        assert prims.env.classes is want


def test_run_object_swaps_the_mass_alone_when_asked(monkeypatch):
    """--mass-only: every shape prior of a case replaces the mass alone (the class prior's CoM stays)."""
    from types import SimpleNamespace

    import robots.mujoco_lab.adapter as adapter
    import robots.mujoco_lab.calibrate as cal
    import robots.mujoco_lab.prior as prior
    import robots.mujoco_lab.vision_prior as vp
    from robots.mujoco_lab import first_menus

    class Stop(Exception):
        pass

    class Env:
        def reset(self, *a, **kw):
            raise Stop

    monkeypatch.setattr(adapter, "object_props", lambda o: {"mass": 1.0})

    seen = []
    monkeypatch.setattr(vp, "shape_prior", lambda base, table, mass_only=False: seen.append(mass_only) or base)
    prims = SimpleNamespace(env=Env(), manifest={}, prior=None)
    monkeypatch.setattr(cal, "_primitives", lambda *a: prims)
    monkeypatch.setattr(cal, "job_force_noise", lambda *a: None)
    monkeypatch.setattr(prior, "fragility_world", lambda manifest: None)
    monkeypatch.setattr(cal, "case_draw", lambda *a, **kw: ({}, 1, "bin", {"fill": 0.0, "frag": None}))
    s = {"bundle": "b", "tables": {"generic": {"o": {}}, "real": {"o": {}}}, "seed": 11, "settings": [], "fold": 0, "mass_only": True}
    with pytest.raises(Stop):
        first_menus.run_object(("o", [0], s))
    assert seen == [True, True]
