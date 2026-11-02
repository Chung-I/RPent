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


"""The generic and VLM priors of the mass and CoM check: the VLM's centres, the spreads fitted on tuning objects, and
the particles (the class prior's, with the mass and the CoM replaced)."""

import numpy as np
import pytest

from rpent.skills.pick_place.belief import ParticleBelief

PIVOT = {"centre": [0.0, 0.0, 0.0], "axis": [1.0, 0.0, 0.0], "lo": -0.1, "L": 0.2}


def _base(pose, n, rng):
    """A stand-in class prior: it draws from rng and carries a fill and a fragility key the replacement must keep."""
    u = rng.random(n)
    parts = [{"mass": 1.0, "mu": 0.1 + 0.8 * u[i], "com_obj": [0.05, 0.01, 0.02], "com_frac": 0.75, "mass_scale": 1.0,
              "com_branch": True, "com_s": 0.7, "com_end": 1, "f": 0.3, "fill": 0.5, "f_damage_n": 20.0 + i, "ancestor": i}
             for i in range(n)]
    return ParticleBelief(parts, np.ones(n))


def test_vlm_centres_take_the_median_answers_and_drop_bad_samples():
    from robots.mujoco_lab.vision_prior import vlm_centres

    recs = [{"object": "a", "flat": True, "mass": [{"mass_kg": 0.5}, {"mass_kg": "0.7"}, {"mass_kg": 0.6}, None, {"mass_kg": -1}],
             "structured": [{"com_marker": 1}, {"com_marker": 2}, {"com_marker": "1"}, {"com_marker": 9}, None]},
            {"object": "b", "flat": False, "mass": [{"mass_kg": 2.0}]},
            {"object": "c", "flat": True, "mass": [None], "structured": [{"com_marker": None}]}]
    c = vlm_centres(recs)
    assert c["a"]["mass"] == pytest.approx(0.6) and c["a"]["n_mass"] == 3
    assert c["a"]["com"] == pytest.approx(0.1) and c["a"]["n_com"] == 3          # markers 1, 2, 1 -> fraction 0.1
    assert c["b"]["mass"] == 2.0 and c["b"]["com"] is None
    assert c["c"]["mass"] is None and c["c"]["com"] is None


def test_fit_spreads_weights_each_object_equally():
    from robots.mujoco_lab.vision_prior import fit_spreads

    truths = {"a": [(1.0, 0.5), (np.e, 0.5)], "b": [(1.0, 0.9)]}            # a: log residuals 0, 1; b: 0
    centres = {"a": {"mass": 1.0, "com": 0.5}, "b": {"mass": 1.0, "com": 0.5}}
    s = fit_spreads(truths, centres, {"a": True, "b": True})
    assert s["mass"] == pytest.approx(np.sqrt((0.5 + 0.0) / 2))                # a's mean square 0.5, b's 0
    assert s["com_flat"] == pytest.approx(np.sqrt((0.0 + 0.16) / 2)) and s["com_upright"] is None


def test_generic_centres_are_the_tuning_population_and_the_middle():
    from robots.mujoco_lab.vision_prior import generic_centres

    truths = {"a": [(1.0, 0.2), (1.0, 0.4)], "b": [(np.e ** 2, 0.6)]}
    c = generic_centres(truths, ["x", "y"])
    assert c["x"] == {"mass": pytest.approx(np.e), "com": 0.5} and c["y"] == c["x"]   # exp of the object-weighted mean log mass


def test_shape_prior_replaces_the_mass_and_the_com_and_keeps_the_rest():
    from robots.mujoco_lab.vision_prior import shape_prior

    prior = shape_prior(_base, {"o": {"mass": 0.4, "com": 0.3, "sigma_mass": 1e-6, "sigma_com": 1e-6}})
    pose = {"object_id": "o", "pivot": PIVOT}
    b = prior(pose, 50, np.random.default_rng(0))
    ref = _base(pose, 50, np.random.default_rng(0))
    a = b.arrays()
    assert np.allclose(a["mass"], 0.4) and np.allclose(a["com_frac"], 0.3)
    assert np.allclose(a["com_obj"], [[-0.1 + 0.3 * 0.2, 0.0, 0.0]] * 50, atol=1e-5)   # on the pivot's centre line (sigma 1e-6)
    assert np.allclose(a["mu"], ref.arrays()["mu"]) and np.allclose(a["fill"], 0.5) and np.allclose(a["f_damage_n"], ref.arrays()["f_damage_n"])
    assert not a["com_branch"].any() and np.allclose(a["f"], 0.0)


def test_shape_prior_spreads_follow_sigma_and_stay_in_the_pivot():
    from robots.mujoco_lab.vision_prior import COM_RANGE, shape_prior

    prior = shape_prior(_base, {"o": {"mass": 1.0, "com": 0.9, "sigma_mass": 0.5, "sigma_com": 0.3}})
    a = prior({"object_id": "o", "pivot": PIVOT}, 4000, np.random.default_rng(1)).arrays()
    assert np.std(np.log(a["mass"])) == pytest.approx(0.5, rel=0.05) and abs(np.mean(np.log(a["mass"]))) < 0.03
    assert a["com_frac"].min() >= COM_RANGE[0] and a["com_frac"].max() <= COM_RANGE[1]


def test_shape_prior_refuses_an_object_it_has_no_centre_for():
    from robots.mujoco_lab.vision_prior import shape_prior

    with pytest.raises(KeyError):
        shape_prior(_base, {})({"object_id": "o", "pivot": PIVOT}, 5, np.random.default_rng(0))


def test_prior_table_generic_and_vlm_fit_on_the_tuning_objects_only():
    from robots.mujoco_lab.vision_prior import prior_table

    truths = {"t1": [(1.0, 0.5)], "t2": [(np.e ** 2, 0.7)], "h": [(100.0, 0.1)]}       # h is held out: never fitted on
    flat = {"t1": True, "t2": True, "h": True}
    gen, s = prior_table("generic", ["h"], ["t1", "t2"], truths, flat)
    assert gen["h"]["mass"] == pytest.approx(np.e) and gen["h"]["com"] == 0.5
    assert s["mass"] == pytest.approx(1.0) and gen["h"]["sigma_mass"] == pytest.approx(1.0)   # log residuals -1, +1
    recs = [{"object": "t1", "flat": True, "mass": [{"mass_kg": 1.0}], "structured": [{"com_marker": 3}]},
            {"object": "t2", "flat": True, "mass": [{"mass_kg": 1.0}], "structured": [{"com_marker": 3}]},
            {"object": "h", "flat": True, "mass": [{"mass_kg": 50.0}], "structured": [{"com_marker": 1}]}]
    vlm, s = prior_table("vlm", ["h"], ["t1", "t2"], truths, flat, recs)
    assert vlm["h"]["mass"] == 50.0 and vlm["h"]["com"] == pytest.approx(0.1)
    assert s["mass"] == pytest.approx(np.sqrt((0.0 + 4.0) / 2)) and vlm["h"]["sigma_com"] == pytest.approx(np.sqrt((0.0 + 0.04) / 2))


def test_prior_table_vlm_falls_back_to_the_generic_centre_and_says_so():
    from robots.mujoco_lab.vision_prior import prior_table

    truths = {"t1": [(1.0, 0.5)], "t2": [(np.e ** 2, 0.7)], "t3": [(1.0, 0.6)], "h": [(2.0, 0.5)], "u": [(2.0, 0.5)]}
    flat = {"t1": True, "t2": True, "t3": False, "h": True, "u": False}
    recs = [{"object": "t1", "flat": True, "mass": [{"mass_kg": 1.0}], "structured": [{"com_marker": 3}]},
            {"object": "t2", "flat": True, "mass": [{"mass_kg": np.e}], "structured": [{"com_marker": 3}]},
            {"object": "t3", "flat": False, "mass": [{"mass_kg": 1.0}]},
            {"object": "h", "flat": True, "mass": [None], "structured": [None]},
            {"object": "u", "flat": False, "mass": [{"mass_kg": 3.0}]}]
    vlm, s = prior_table("vlm", ["h", "u"], ["t1", "t2", "t3"], truths, flat, recs)
    assert vlm["u"]["sigma_com"] == pytest.approx(0.1) and s["mass"] == pytest.approx(np.sqrt(1 / 3))
    assert vlm["h"]["mass"] == pytest.approx(np.exp(2 / 3)) and vlm["h"]["com"] == 0.5 and vlm["h"]["fallback"] == ["mass", "com"]
    assert vlm["u"]["mass"] == 3.0 and vlm["u"]["com"] == 0.5 and vlm["u"]["fallback"] == []


def test_prior_table_refuses_a_spread_it_cannot_fit():
    from robots.mujoco_lab.vision_prior import prior_table

    truths = {"t1": [(1.0, 0.5)], "h": [(2.0, 0.5)]}
    with pytest.raises(ValueError):
        prior_table("generic", ["h"], ["t1"], truths, {"t1": True, "h": False})     # no upright tuning object
