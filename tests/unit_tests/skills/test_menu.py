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

"""Cost table and the option menu (stub model, stub trajectory planner)."""

import numpy as np

from rpent.skills.pick_place.belief import Belief
from rpent.skills.pick_place.costs import CARE_SCALE, COST, TIME_COST_PER_S, expected_cost
from rpent.skills.pick_place.features import Geometry
from rpent.skills.pick_place.menu import build_menu, estimate
from rpent.skills.pick_place.protocol import IRREVERSIBLE, OUTCOMES, Prediction


class StubModel:
    """P(placed_stable) and P(dropped_high) depend only on v_lift and force, so rows are distinguishable."""

    kind = "classifier"

    def __init__(self, table=None):
        self.table = table or {}
        self.calls = []

    def predict_many(self, feats):
        out = []
        for f in feats:
            self.calls.append(f)
            key = (round(f["v_lift"], 3), round(f["load_ratio"] * f["mu"] * 2 * 20, 2))   # second item ~ mass-driven
            ps, dh = self.table.get(round(f["v_lift"], 3), (0.6, 0.3))
            rest = 1 - ps - dh
            p = {k: 0.0 for k in OUTCOMES}
            p.update({"placed_stable": ps, "dropped_high": dh, "dropped_low": rest * 0.5, "not_lifted": rest * 0.5})
            out.append(Prediction(p=p))
        return out


def _geometry():
    C = np.array([[0.02, 0.0, 0.76], [-0.02, 0.0, 0.76]])
    N = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    return Geometry(centres=C, normals=N, com=np.array([0.0, 0.0, 0.77]), length=0.1, base_extent=0.02,
                    com_height=0.02, footprint_half_width=0.05)


class P:
    def __init__(self, s, width=0.04):
        self.s, self.point, self.yaw, self.width = s, [0.0, 0.0, 0.76], 0.0, width


def plan_fn(proposal, v_lift, v_transport, destination, release_height=None):
    return [{"phase": "lift", "n_steps": int(0.2 / v_lift * 50), "z_start": 0.0, "z_end": 0.2, "has_surface": True, "speed": v_lift}]


def _menu(model=None, care="normal", destination="table_centre", proposals=None, threshold=0.7):
    model = model or StubModel({0.15: (0.75, 0.2), 0.05: (0.72, 0.05)})
    belief = Belief.prior(1e-4, "wood")
    return build_menu(model, belief, _geometry(), proposals or [P(0.3), P(0.5), P(0.7)], 20.0, care, destination,
                      plan_fn, np.random.default_rng(0), threshold=threshold)


def test_cost_table_and_care_scales_only_irreversible():
    p_tip = {k: 0.0 for k in OUTCOMES}
    p_tip.update({"placed_stable": 0.5, "tipped": 0.5})
    p_high = {k: 0.0 for k in OUTCOMES}
    p_high.update({"placed_stable": 0.5, "dropped_high": 0.5})
    assert expected_cost(Prediction(p=p_tip), 0.0, "high") == expected_cost(Prediction(p=p_tip), 0.0, "low") == 0.5 * COST["tipped"]
    assert expected_cost(Prediction(p=p_high), 0.0, "high") == 0.5 * COST["dropped_high"] * CARE_SCALE["high"]
    assert expected_cost(Prediction(p=p_high), 10.0, "normal") == 0.5 * COST["dropped_high"] + 10 * TIME_COST_PER_S
    assert set(IRREVERSIBLE) == {"dropped_high", "rolled_off"} and COST["placed_stable"] == 0.0


def _base(destination="bin", v_lift=0.15):
    from rpent.skills.pick_place.features import extract

    return extract(_geometry(), [0.0, 0.0, 0.76], 0.0, 0.2, 0.5, 20.0, v_lift, 0.5, destination)


def _pose(com_y=0.0):
    from tests.unit_tests.skills.synthetic import synthetic_pose

    return dict(synthetic_pose(), com_obj_nominal=[0.0, com_y, 0.02])


def _particles(mus, weights, com=(0.0, 0.0, 0.02)):
    from rpent.skills.pick_place.belief import ParticleBelief

    parts = [{"mass": 0.2 + 0.001 * i, "mu": float(m), "com_obj": list(com)} for i, m in enumerate(mus)]
    return ParticleBelief(parts, np.asarray(weights, dtype=float))


def test_estimate_averages_over_belief_samples():
    model = StubModel()
    belief = _particles(np.linspace(0.2, 0.8, 16), np.ones(16))
    pred, per = estimate(model, belief, _base(), _pose(), 20.0, [], np.random.default_rng(0))
    assert len(model.calls) == 16 and len(per) == 16 and abs(sum(pred.p.values()) - 1) < 1e-9
    assert len({round(c["mass"], 6) for c in model.calls}) == 16


class MuStub:
    """P(placed_stable) = 1 for mu > 0.5 else 0 (the rest not_lifted); the hazard kind also needs the trajectory."""

    def __init__(self, kind="classifier"):
        self.kind, self.planned = kind, []

    def predict_many(self, feats, planned=None):
        if self.kind == "hazard":
            assert planned is not None
            self.planned.append(planned)
        out = []
        for f in feats:
            p = {k: 0.0 for k in OUTCOMES}
            p["placed_stable" if f["mu"] > 0.5 else "not_lifted"] = 1.0
            dh = {"0.00-0.05": 0.0, "0.05-0.10": 0.0, "0.10-0.20": 0.0, "0.20+": 0.0} if self.kind == "hazard" else None
            out.append(Prediction(p=p, drop_height=dh))
        return out


def test_estimate_weights_particles():
    belief = _particles([0.7, 0.2], [0.8, 0.2])
    planned = plan_fn(P(0.5), 0.15, 0.5, "bin")
    for kind in ("classifier", "hazard"):
        model = MuStub(kind)
        pred, per = estimate(model, belief, _base(), _pose(), 20.0, planned, np.random.default_rng(0))
        assert abs(pred.p["placed_stable"] - 0.8) < 1e-12 and abs(pred.p["not_lifted"] - 0.2) < 1e-12
        assert [x.p["placed_stable"] for x in per] == [1.0, 0.0]
    assert model.planned and model.planned[0] == planned


class LeverStub:
    """P(dropped_low) rises with the lever: min(1, lever / 0.1); the rest placed_stable."""

    kind = "classifier"

    def predict_many(self, feats):
        out = []
        for f in feats:
            p = {k: 0.0 for k in OUTCOMES}
            p["dropped_low"] = min(1.0, f["lever"] / 0.1)
            p["placed_stable"] = 1.0 - p["dropped_low"]
            out.append(Prediction(p=p))
        return out


class Q:
    """A proposal along the object's axis (world y), 4 cm apart."""

    def __init__(self, s):
        self.s, self.point, self.yaw, self.width = s, [0.0, (s - 0.5) * 0.2, 0.76], 0.0, 0.04


def test_menu_uses_the_particle_com():
    def menu_for(com_y):
        pose = _pose()
        belief = _particles(np.full(50, 0.6), np.ones(50), com=(0.0, com_y, 0.02))
        options, _ = build_menu(LeverStub(), belief, _geometry(), [Q(0.3), Q(0.5), Q(0.7)], 20.0, "normal", "table_centre",
                                plan_fn, np.random.default_rng(0), pose_fn=lambda q: dict(pose, pt=list(q.point), yaw=q.yaw))
        return {o.id: o for o in options}

    shifted = menu_for(0.08)                                                  # every CoM 8 cm toward the +y end
    assert shifted["regrasp_com"].action["s"] == 0.7
    assert shifted["regrasp_com"].expected_cost < shifted["commit_std"].expected_cost
    centred = menu_for(0.0)
    assert centred["regrasp_com"].action["s"] == 0.5
    assert centred["regrasp_com"].expected_cost > centred["commit_std"].expected_cost
    o = shifted["commit_std"]
    assert o._costs.shape == (50,) and abs(float(np.mean(o._costs)) - o.expected_cost) < 1e-9
    assert o._pose["pt"] == list(Q(0.5).point) and o._base["v_lift"] == 0.15


def test_menu_order_and_decline_last():
    options, rec = _menu()
    assert options[-1].id == "decline" and options[-1].kind == "decline"
    costs = [o.expected_cost for o in options[:-1]]
    assert costs == sorted(costs)
    ids = {o.id for o in options}
    assert {"commit_std", "slow_lift", "firm_grip", "regrasp_com", "decline"} <= ids
    assert all(o.action["destination"] == "table_centre" for o in options[:-1])


def test_too_wide_everything_gives_decline_only():
    options, rec = _menu(proposals=[P(0.3, width=0.12), P(0.5, width=0.13), P(0.7, width=0.12)])
    assert [o.id for o in options] == ["decline"] and rec == "decline"
    assert "too wide" in options[0].reason


def test_recommendation_threshold():
    assert _menu(StubModel({0.15: (0.8, 0.1), 0.05: (0.8, 0.1)}))[1] == "commit"
    assert _menu(StubModel({0.15: (0.6, 0.1), 0.05: (0.6, 0.1)}))[1] == "decline"
    assert _menu(StubModel({0.15: (0.6, 0.1), 0.05: (0.6, 0.1)}), threshold=0.5)[1] == "commit"


def test_care_high_prefers_slow_lift():
    model = StubModel({0.15: (0.75, 0.2), 0.05: (0.72, 0.05)})
    low, _ = _menu(model, care="low")
    high, _ = _menu(model, care="high")
    assert low[0].id == "commit_std" and high[0].id == "slow_lift"


def test_edge_destination_offers_inboard_place_and_gap_too():
    options, _ = _menu(destination="table_edge")
    inboard = next(o for o in options if o.id == "inboard_place")
    assert inboard.action["destination"] == "table_centre" and inboard.kind == "lower_risk" and "requested" in inboard.reason
    assert all(o.id != "inboard_place" for o in _menu(destination="bin")[0])
    assert any(o.id == "inboard_place" for o in _menu(destination="table_gap")[0])


def test_drop_release_only_for_rim_cavity_bin():
    plate, _ = _menu(destination="plate")
    drop = next(o for o in plate if o.id == "drop_release")
    assert abs(drop.action["release_height"] - 0.065) < 1e-9 and drop.cost_s < next(o for o in plate if o.id == "commit_std").cost_s
    assert all(o.id != "drop_release" for o in _menu(destination="table_centre")[0])
