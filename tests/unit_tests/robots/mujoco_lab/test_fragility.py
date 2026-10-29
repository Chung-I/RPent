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


"""Fragility entries, thresholds v0, the fragility draws, the liquid in a cavity, the spill angle and the damage rule."""

import math
import os

import numpy as np
import pytest

CYL = {"axis_obj": [0.0, 0.0, 1.0], "bottom_obj": [0.0, 0.0, 0.01], "r_bottom_m": 0.04, "r_top_m": 0.04, "height_m": 0.09}
CARD = {"class": "crushable", "law": "cardboard", "fillable": False, "cavity": None, "reason": "test"}
CUP = {"class": "crushable", "law": "thin_plastic", "fillable": True, "cavity": CYL, "cavity_checked": True, "reason": "test"}
MUG = {"class": "breakable", "law": "ceramic", "fillable": True, "cavity": CYL, "cavity_checked": True, "reason": "test"}
FRUIT_IDS = {"fruits_veggies__avocado", "ycb__banana", "fruits_veggies__lemon1", "fruits_veggies__lime", "fruits_veggies__orange1",
             "fruits_veggies__red_onion", "fruits_veggies__pomegranate"}
needs_sim = pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")


def _th():
    from robots.mujoco_lab.fragility import THRESHOLDS_V0, Thresholds

    return Thresholds.from_json(THRESHOLDS_V0)


@needs_sim
def test_every_object_has_an_entry():
    from robots.mujoco_lab.adapter import object_ids
    from robots.mujoco_lab.fragility import CLASSES, LAWS, load_classes

    classes = load_classes(CLASSES)
    assert set(classes) == set(object_ids())
    for oid, e in classes.items():
        assert e["class"] in ("none", "crushable", "bruisable", "breakable") and e["reason"], oid
        assert (e["law"] is None) == (e["class"] == "none") and (e["law"] is None or e["law"] in LAWS[e["class"]]), oid
        assert isinstance(e["fillable"], bool) and (e["cavity"] is not None) == e["fillable"], oid
        if e["fillable"]:
            c = e["cavity"]
            assert set(c) == {"axis_obj", "bottom_obj", "r_bottom_m", "r_top_m", "height_m"} and e["cavity_checked"] is True, oid
            assert 0.01 < c["r_bottom_m"] < 0.15 and 0.01 < c["r_top_m"] < 0.15 and 0.01 < c["height_m"] < 0.3, oid
    assert {o for o, e in classes.items() if e["law"] == "fruit"} == FRUIT_IDS
    for oid in ("hope__canned_peaches", "hope__pineapple_slices_can", "hope__tomato_sauce_can", "hope__orange_juice_carton"):
        assert classes[oid]["law"] != "fruit" and "name" in classes[oid]["reason"], oid
    for oid in ("handal__measuring_cups", "handal__measuring_cups_1", "hot3d__coffee_pot", "hot3d__storage_box"):
        assert classes[oid]["class"] == "none" and "render" in classes[oid]["reason"], oid
    assert {o for o, e in classes.items() if e["fillable"]} == {
        "ycb__mug", "ycb__bowl", "ycb__bowl2", "hot3d__ceramic_mug", "hot3d__mug", "hot3d__wooden_bowl", "hot3d__pitcher"}
    for oid in ("ycb__pitcher", "hot3d__coffee_pot", "hope__yogurt_cup"):
        assert "closed" in classes[oid]["reason"], oid
    for oid in ("handal__measuring_cups", "handal__measuring_cups_1"):
        assert "opening-down" in classes[oid]["reason"], oid


def test_unknown_object_is_none_with_evidence():
    from robots.mujoco_lab.fragility import entry_of

    e, ev = entry_of("no__such_object", {"ycb__mug": MUG})
    assert e["class"] == "none" and e["law"] is None and e["fillable"] is False and e["cavity"] is None
    assert ev == ["no fragility class for no__such_object: treated as none"]
    e, ev = entry_of("ycb__mug", {"ycb__mug": MUG})
    assert e == MUG and ev == []


def test_an_unchecked_cavity_does_not_load(tmp_path):
    import json

    from robots.mujoco_lab.fragility import load_classes

    p = tmp_path / "c.json"
    json.dump({"ycb__mug": dict(MUG, cavity_checked=False)}, open(p, "w"))
    with pytest.raises(ValueError, match="ycb__mug"):
        load_classes(str(p))
    json.dump({"ycb__mug": dict(MUG, cavity=None)}, open(p, "w"))
    with pytest.raises(ValueError, match="ycb__mug"):
        load_classes(str(p))


def test_thresholds_v0_values():
    th = _th()
    card = th.laws["cardboard"]
    assert card["f_damage_n"] == {"median": 25.0, "log_sd": 0.35} and card["k_n_per_mm"] == {"median": 2.0, "log_sd": 0.35}
    assert card["rho"] == 0.8 and th.laws["fruit"]["v_bruise_ms"] == {"median": 1.0, "log_sd": 0.30}
    assert th.laws["glass"]["v_break_ms"] == {"median": 2.5, "log_sd": 0.30} and th.slosh_zeta == 0.01
    assert "thresholds_v1" in th.status and all(law.get("source") for law in th.laws.values())


def test_draws_follow_the_law():
    from robots.mujoco_lab.fragility import draw_fragility

    rng = np.random.default_rng(0)
    d = [draw_fragility(rng, CARD, _th()) for _ in range(20000)]
    lf, lk = np.log([x["f_damage_n"] for x in d]), np.log([x["k_n_per_mm"] for x in d])
    assert abs(lf.mean() - math.log(25.0)) < 0.01 and abs(np.corrcoef(lk, lf)[0, 1] - 0.8) < 0.02
    assert abs(lf.std() - 0.35) < 0.01 and abs(lk.mean() - math.log(2.0)) < 0.01
    assert all(x["v_bruise_ms"] is None and x["v_break_ms"] is None and len(x["z"]) == 3 for x in d[:10])


def test_draws_by_class():
    from robots.mujoco_lab.fragility import draw_fragility

    rng = np.random.default_rng(1)
    fruit = draw_fragility(rng, {"class": "bruisable", "law": "fruit", "fillable": False, "cavity": None}, _th())
    assert fruit["f_damage_n"] > 0 and fruit["k_n_per_mm"] > 0 and fruit["v_bruise_ms"] > 0 and fruit["v_break_ms"] is None
    mug = draw_fragility(rng, MUG, _th())
    assert mug["v_break_ms"] > 0 and mug["f_damage_n"] is None and mug["k_n_per_mm"] is None and mug["v_bruise_ms"] is None
    steel = draw_fragility(rng, {"class": "none", "law": None, "fillable": False, "cavity": None}, _th())
    assert all(steel[k] is None for k in ("k_n_per_mm", "f_damage_n", "v_bruise_ms", "v_break_ms")) and len(steel["z"]) == 3
    a, b = np.random.default_rng(2), np.random.default_rng(2)
    draw_fragility(a, {"class": "none", "law": None, "fillable": False, "cavity": None}, _th())
    draw_fragility(b, CARD, _th())
    assert a.random() == b.random()                                     # every class takes the same three normals


def test_kept_latents_map_through_a_new_law():
    import copy

    from robots.mujoco_lab.fragility import draw_fragility, map_latent

    th = _th()
    rng = np.random.default_rng(3)
    d = [draw_fragility(rng, CARD, th) for _ in range(100)]
    th2 = copy.deepcopy(th)
    th2.laws["cardboard"]["f_damage_n"]["median"] = 50.0
    for x in d:
        y = map_latent(x["z"], CARD, th2)
        assert y["f_damage_n"] == pytest.approx(2.0 * x["f_damage_n"], rel=1e-12)
        assert y["k_n_per_mm"] == pytest.approx(x["k_n_per_mm"], rel=1e-12) and y["z"] == x["z"]


def test_liquid_mass_from_the_cavity():
    from robots.mujoco_lab.fragility import liquid

    q = liquid(CYL, 0.6)
    assert q["volume_l"] == pytest.approx(0.4524, abs=1e-4) and q["mass_kg"] == pytest.approx(0.2714, abs=1e-4)
    assert q["height_m"] == pytest.approx(0.054, abs=1e-6)
    assert np.allclose(q["centroid_obj"], [0.0, 0.0, 0.01 + 0.027], atol=1e-6)
    e = liquid(CYL, 0.0)
    assert e["mass_kg"] == 0.0 and e["height_m"] == 0.0 and np.allclose(e["centroid_obj"], [0.0, 0.0, 0.01])
    cone = dict(CYL, r_bottom_m=0.03, r_top_m=0.05)
    v = math.pi * 0.09 * (0.03 ** 2 + 0.03 * 0.05 + 0.05 ** 2) / 3.0
    assert liquid(cone, 1.0)["volume_l"] == pytest.approx(1000.0 * v, rel=1e-9)
    assert liquid(cone, 0.5)["height_m"] > 0.045                       # a cone wide at the top fills slowly at first


def test_spill_angle_of_a_cylinder():
    from robots.mujoco_lab.fragility import spill_angle_deg

    for f in (0.5, 0.6, 0.8, 0.95):
        want = math.degrees(math.atan((0.09 - f * 0.09) / 0.04))
        assert abs(spill_angle_deg(CYL, f) - want) < 0.2, f
    a = [spill_angle_deg(CYL, f) for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert all(x > y for x, y in zip(a, a[1:]))
    assert spill_angle_deg(CYL, 0.0) == 90.0 and spill_angle_deg(CYL, 1.0) == pytest.approx(0.0, abs=0.2)


def test_first_crossing_orders_events():
    from robots.mujoco_lab.fragility import damage_event, spill_angle_deg

    meter = {"fn_pad_max": [(0.5, 10.0), (1.0, 25.5), (2.0, 30.0), (4.0, 40.0)], "surface_max": [(1.0, 5.0), (3.0, 60.0)],
             "impacts": []}
    frag = {"k_n_per_mm": 2.0, "f_damage_n": 25.0, "v_bruise_ms": None, "v_break_ms": None}
    assert spill_angle_deg(CYL, 0.6) < 60.0
    label, channel, t = damage_event(meter, frag, CUP, 0.6)
    assert (label, channel) == ("crushed", "force") and t == pytest.approx(1.0)
    label, channel, t = damage_event(meter, dict(frag, f_damage_n=50.0), CUP, 0.6)
    assert (label, channel) == ("spilled", "surface") and t == pytest.approx(3.0)
    assert damage_event(meter, dict(frag, f_damage_n=50.0), CARD, 0.0) is None
    fruit = {"class": "bruisable", "law": "fruit", "fillable": False, "cavity": None}
    m2 = {"fn_pad_max": [(0.2, 5.0)], "surface_max": [], "impacts": [(1.5, 0.4), (2.5, 1.4)]}
    assert damage_event(m2, {"f_damage_n": 30.0, "v_bruise_ms": 1.0, "v_break_ms": None}, fruit, 0.0) == ("bruised", "impact", 2.5)


def test_tie_rule():
    from robots.mujoco_lab.fragility import damage_event

    meter = {"fn_pad_max": [(2.0, 30.0)], "surface_max": [(2.0, 89.0)], "impacts": [(2.0, 9.0)]}
    assert damage_event(meter, {"f_damage_n": 25.0, "v_bruise_ms": None, "v_break_ms": None}, CUP, 0.6)[0] == "crushed"
    assert damage_event(meter, {"f_damage_n": None, "v_bruise_ms": None, "v_break_ms": 3.0}, MUG, 0.6)[0] == "broken"


def test_empty_container_never_spills():
    from robots.mujoco_lab.fragility import damage_event

    meter = {"fn_pad_max": [(1.0, 5.0)], "surface_max": [(1.0, 89.9)], "impacts": []}
    assert damage_event(meter, {"f_damage_n": None, "v_bruise_ms": None, "v_break_ms": 3.0}, MUG, 0.0) is None
    assert damage_event(meter, {"f_damage_n": None, "v_bruise_ms": None, "v_break_ms": 3.0}, MUG, 0.5)[0] == "spilled"
