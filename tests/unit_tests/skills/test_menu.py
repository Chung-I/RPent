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
from rpent.skills.pick_place.menu import (MAX_OPEN_WIDTH, OFF_OBJECT, TOO_WIDE, build_menu, decline_reason, estimate, mid_proposal,
                                         unusable_reason, usable_proposals)
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


def plan_fn(proposal, v_lift, v_transport, destination, release_height=None, a_transport=2.0):
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
    assert set(IRREVERSIBLE) == {"dropped_high", "rolled_off", "crushed", "bruised", "broken", "spilled"} and COST["placed_stable"] == 0.0


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
    assert options[0].reason == "every proposal is wider than the gripper opening"


def test_a_pinch_is_usable_when_its_exact_width_fits_the_opening():
    """The pads are 9.4 cm apart when open; the skill keeps 2 mm a side, so it pinches at most 9.0 cm. A nan width
    means no hull lies between the pads."""
    assert MAX_OPEN_WIDTH == 0.090
    assert unusable_reason(0.090) is None and unusable_reason(0.0901) == TOO_WIDE == "wider than the gripper opening"
    assert unusable_reason(float("nan")) == OFF_OBJECT and unusable_reason(0.05, max_open=0.04) == TOO_WIDE
    assert decline_reason([P(0.3, width=0.12), P(0.5, width=float("nan"))], MAX_OPEN_WIDTH) == \
        "every proposal is off the object or wider than the gripper opening"


def test_a_pinch_wider_than_the_limit_is_not_offered():
    options, _ = _menu(proposals=[P(0.3, width=0.095), P(0.5, width=0.04), P(0.7, width=0.091)])
    assert {o.action["s"] for o in options if o.kind != "decline"} == {0.5}


def test_a_pinch_that_closes_on_nothing_is_not_offered():
    options, _ = _menu(proposals=[P(0.3, width=float("nan")), P(0.5, width=0.04), P(0.7, width=float("nan"))])
    assert {o.action["s"] for o in options if o.kind != "decline"} == {0.5}


def test_the_decline_names_every_reason():
    options, rec = _menu(proposals=[P(0.3, width=float("nan")), P(0.5, width=0.12), P(0.7, width=float("nan"))])
    assert [o.id for o in options] == ["decline"] and rec == "decline"
    assert options[0].reason == "every proposal is off the object or wider than the gripper opening"


def test_the_mid_proposal_is_the_usable_one_nearest_the_middle():
    assert mid_proposal(usable_proposals([P(0.3), P(0.5), P(0.7)])).s == 0.5
    assert mid_proposal(usable_proposals([P(0.3, width=0.12), P(0.5, width=0.12), P(0.7)])).s == 0.7


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


# ---------------------------------------------------------------- probe rows and their value (milestone 2, task 7)
import pytest  # noqa: E402

from rpent.skills.pick_place.costs import CARE_SCALE as _CARE  # noqa: E402
from rpent.skills.pick_place.menu import probe_value  # noqa: E402
from rpent.skills.pick_place.probes import PROBE_SECONDS, weigh_force  # noqa: E402
from tests.unit_tests.skills.probe_stubs import FlatWeigh, MuGate, PerfectPush  # noqa: E402


def _pmenu(model=None, probe_models=None, destination="table_centre", belief=None, care="normal", guard=None, proposals=None,
           rng=None):
    belief = belief if belief is not None else _particles(np.random.default_rng(0).uniform(0.1, 0.9, 400), np.ones(400))
    return build_menu(model or MuGate(f_std=20.0), belief, _geometry(), proposals or [P(0.3), P(0.5), P(0.7)], 20.0, care,
                      destination, plan_fn, rng if rng is not None else np.random.default_rng(1), threshold=0.7,
                      probe_models=probe_models, guard=guard)


def _by_id(options):
    return {o.id: o for o in options}


def test_probe_rows_present_when_models_exist():
    options, _ = _pmenu(probe_models={"push": PerfectPush(p_tip=0.1), "weigh": FlatWeigh()})
    rows = _by_id(options)
    for pid, probe in (("probe_push", "push"), ("probe_weigh", "weigh")):
        o = rows[pid]
        assert o.kind == "probe" and o.pred is None and o.action == {"probe": probe, "s": 0.5, "force": weigh_force(20.0)}
        assert set(o.probe) == {"p_kind", "p_thrown", "evoi", "evoi_se", "expected_cost_after"}
        tip = o.probe["p_kind"]["tipped"] * COST["tipped"] if probe == "push" else 0.0
        assert o.expected_cost == pytest.approx(TIME_COST_PER_S * PROBE_SECONDS[probe] + tip + o.probe["expected_cost_after"])
    assert rows["probe_push"].probe["p_kind"]["tipped"] == pytest.approx(0.1)
    assert [o.expected_cost for o in options[:-1]] == sorted(o.expected_cost for o in options[:-1])


def test_no_probe_rows_without_models():
    options, _ = _pmenu()
    assert not any(o.kind == "probe" for o in options)
    options, _ = _pmenu(probe_models={})
    assert not any(o.kind == "probe" for o in options)


def test_evoi_is_zero_for_a_point_belief():
    point = _particles(np.linspace(0.1, 0.9, 50), [1.0] + [0.0] * 49)
    options, rec = _pmenu(probe_models={"push": PerfectPush(), "weigh": FlatWeigh()}, belief=point)
    rows = _by_id(options)
    assert rows["probe_push"].probe["evoi"] == 0.0 and rows["probe_weigh"].probe["evoi"] == 0.0 and rec != "probe"
    C = [o._costs for o in options if o.kind in ("commit", "lower_risk")]
    v = probe_value(point, C, PerfectPush(), np.zeros((50, 3)), 20.0, "normal", np.random.default_rng(0))
    assert v["evoi"] == 0.0


def test_evoi_positive_when_the_best_row_depends_on_mu():
    """firm_grip places only above mu 0.3 and slow_lift always places at a higher time cost: a push that
    tells the side of 0.3 is worth about 0.75 x (0.7 - 0.4) = 0.23, and its row ranks first."""
    options, rec = _pmenu(probe_models={"push": PerfectPush(), "weigh": FlatWeigh()})
    rows = _by_id(options)
    assert rows["slow_lift"].expected_cost == pytest.approx(0.7) and options[0].id == "probe_push" and rec == "probe"
    assert 0.1 < rows["probe_push"].probe["evoi"] < 0.35 and rows["probe_weigh"].probe["evoi"] == 0.0


def test_evoi_ignores_a_cheaper_inboard_row():
    """At the table edge an inboard row that always places is cheapest; it is not a continuation (it
    changes the destination), so the push keeps its value."""
    options, rec = _pmenu(model=MuGate(f_std=20.0, inboard=True), probe_models={"push": PerfectPush()}, destination="table_edge")
    rows = _by_id(options)
    assert options[0].id == "inboard_place" and rows["probe_push"].probe["evoi"] > 0.1 and rec == "probe"


def test_thrown_probe_ends_the_case_in_the_preposterior():
    b = _particles(np.linspace(0.1, 0.9, 40), np.ones(40))
    C = [np.where(b.arrays()["mu"] > 0.3, 0.4, 5.4), np.full(40, 0.7)]
    for care in ("normal", "high"):
        v = probe_value(b, C, PerfectPush(p_thrown=1.0), np.zeros((40, 3)), 20.0, care, np.random.default_rng(0))
        assert v["expected_cost_after"] == pytest.approx(COST["dropped_high"] * _CARE[care]) and v["evoi"] == 0.0
        assert v["p_thrown"] == pytest.approx(1.0)


class RiskyPush(PerfectPush):
    """A perfect push that throws the object with P 0.5 when mu (column 0 of X) is at most the cut."""

    def risk(self, X):
        return np.where(np.asarray(X, dtype=float)[:, 0] <= self.cut, 0.5, 0.0)


def test_evoi_is_zero_on_every_stream_when_no_result_changes_the_best_row():
    """Row 0 is cheaper than row 1 for every particle, so no push result can change the best row,
    however much it tells about mu: the value of information is 0 on every random stream, and the
    cost after the probe is row 0's cost now."""
    b = _particles(np.random.default_rng(0).uniform(0.1, 0.9, 400), np.ones(400))
    c0 = np.where(b.arrays()["mu"] > 0.3, 0.5, 0.9)
    vs = [probe_value(b, [c0, c0 + 0.2], PerfectPush(), np.zeros((400, 3)), 20.0, "normal", np.random.default_rng(s))
          for s in range(20)]
    assert [v["evoi"] for v in vs] == [0.0] * 20
    assert all(v["evoi_se"] == 0.0 for v in vs)
    assert all(v["expected_cost_after"] == pytest.approx(float(b.weights @ c0), abs=1e-12) for v in vs)


def test_evoi_counts_each_particles_throw_risk():
    """mu above 0.3 (share f): never thrown, firm_grip (0.4) after the push. mu at most 0.3: thrown
    with P 0.5 (cost 5), else slow_lift (0.7), the best row now. Exact: cost after
    f x 0.4 + (1 - f) x (2.5 + 0.35); value f x (0.7 - 0.4), counted where the probe keeps the object."""
    b = _particles(np.random.default_rng(0).uniform(0.1, 0.9, 400), np.ones(400))
    mu = b.arrays()["mu"]
    f = float(np.mean(mu > 0.3))
    C = [np.where(mu > 0.3, 0.4, 5.4), np.full(400, 0.7)]
    X = np.zeros((400, 3))
    X[:, 0] = mu
    v = probe_value(b, C, RiskyPush(), X, 20.0, "normal", np.random.default_rng(3), k=4000)
    assert v["evoi"] == pytest.approx(0.3 * f, abs=0.012)
    assert v["expected_cost_after"] == pytest.approx(f * 0.4 + (1 - f) * (2.5 + 0.35), abs=0.012)
    assert v["p_thrown"] == pytest.approx(0.5 * (1 - f))


def test_evoi_se_is_the_monte_carlo_error_of_the_mean_gain():
    """The perfect push: each result gains 0.3 (mu above 0.3) or 0, so the error of the mean over k
    draws is 0.3 x sqrt(f (1 - f) / k)."""
    b = _particles(np.random.default_rng(0).uniform(0.1, 0.9, 400), np.ones(400))
    mu = b.arrays()["mu"]
    f = float(np.mean(mu > 0.3))
    C = [np.where(mu > 0.3, 0.4, 5.4), np.full(400, 0.7)]
    v = probe_value(b, C, PerfectPush(), np.zeros((400, 3)), 20.0, "normal", np.random.default_rng(5), k=400)
    assert v["evoi_se"] == pytest.approx(0.3 * np.sqrt(f * (1 - f) / 400), rel=0.25)


def test_probe_recommendation_needs_a_margin_beyond_the_monte_carlo_error(monkeypatch):
    """The perfect push ranks first and is recommended; with the margin rule at 10^6 errors it is
    still ranked first but no longer recommended."""
    from rpent.skills.pick_place import menu

    options, rec = _pmenu(probe_models={"push": PerfectPush(), "weigh": FlatWeigh()})
    assert options[0].id == "probe_push" and rec == "probe"
    monkeypatch.setattr(menu, "EVOI_Z", 1e6)
    options, rec = _pmenu(probe_models={"push": PerfectPush(), "weigh": FlatWeigh()})
    assert options[0].id == "probe_push" and rec != "probe"


# ---------------------------------------------------------------- the novelty guard
from rpent.skills.pick_place.features import FEATURES, particle_matrix  # noqa: E402
from rpent.skills.pick_place.guard import fit_guard, flag_reason  # noqa: E402


class StubGuard:
    """Records the rows and weights the menu passes; flags when told to."""

    def __init__(self, flag):
        self.flag, self.seen = flag, []

    def check(self, X, weights=None):
        self.seen.append((np.asarray(X), np.asarray(weights)))
        return {"guard": "knn", "score": 4.2, "p": 0.02 if self.flag else 0.4, "flag": self.flag, "alpha": 0.05, "k": 10,
                "n_calibration": 60}


SURE = StubModel({0.15: (0.8, 0.1), 0.05: (0.8, 0.1)})       # every row reaches the threshold: the menu would commit


def _belief400():
    return _particles(np.random.default_rng(0).uniform(0.1, 0.9, 400), np.ones(400))


def test_a_guard_flag_declines_with_the_guard_reason_and_keeps_the_rows():
    options, rec = _pmenu(SURE, guard=StubGuard(flag=True))
    decline = options[-1]
    assert rec == "decline" and decline.id == "decline" and decline.guard["flag"] is True
    assert decline.reason == flag_reason(decline.guard)
    assert {"commit_std", "slow_lift", "firm_grip", "regrasp_com"} <= {o.id for o in options}
    assert all(o.pred is not None for o in options if o.kind in ("commit", "lower_risk"))


def test_without_a_flag_the_recommendation_stands_and_the_check_is_reported():
    options, rec = _pmenu(SURE, guard=StubGuard(flag=False))
    assert rec == "commit" and options[-1].guard["flag"] is False and "unlike the training objects" not in options[-1].reason
    options, rec = _pmenu(SURE)
    assert rec == "commit" and options[-1].guard is None


def test_a_guard_flag_overrides_a_probe_recommendation():
    probes = {"push": PerfectPush(), "weigh": FlatWeigh()}
    assert _pmenu(probe_models=probes)[1] == "probe"
    options, rec = _pmenu(probe_models=probes, guard=StubGuard(flag=True))
    assert rec == "decline" and options[0].id == "probe_push"


def test_the_guard_sees_every_commit_and_lower_risk_row_for_every_particle():
    g, belief = StubGuard(flag=False), _belief400()
    options, _ = _pmenu(SURE, destination="table_edge", belief=belief, guard=g)
    rows = [o for o in options if o.kind in ("commit", "lower_risk")]
    assert "inboard_place" in {o.id for o in rows}
    (X, w), = g.seen
    expect = np.vstack([particle_matrix(o._base, belief.arrays(), o._pose, o.action["force"]) for o in rows])
    assert X.shape == (len(rows) * 400, len(FEATURES)) and np.array_equal(X, expect)
    assert np.allclose(w, np.tile(belief.weights / belief.weights.sum(), len(rows)) / len(rows))


def test_the_guard_takes_no_random_draws():
    a, b = np.random.default_rng(1), np.random.default_rng(1)
    probes = {"push": PerfectPush(), "weigh": FlatWeigh()}
    _pmenu(probe_models=probes, rng=a)
    _pmenu(probe_models=probes, rng=b, guard=StubGuard(flag=True))
    assert a.bit_generator.state == b.bit_generator.state


def test_without_rows_the_guard_is_not_asked():
    g = StubGuard(flag=True)
    options, rec = _pmenu(SURE, guard=g, proposals=[P(0.3, width=0.12), P(0.5, width=0.13), P(0.7, width=0.12)])
    assert [o.id for o in options] == ["decline"] and rec == "decline" and g.seen == [] and options[0].guard is None
    assert options[0].reason == "every proposal is wider than the gripper opening"


def test_a_fitted_guard_checks_the_menu():
    from tests.unit_tests.skills.synthetic import synthetic_rows, with_pose

    guard = fit_guard(with_pose(synthetic_rows(300)), "true", seed=0)
    options, _ = _pmenu(SURE, guard=guard)
    rec = options[-1].guard
    assert set(rec) == {"guard", "score", "p", "flag", "alpha", "k", "n_calibration"} and np.isfinite(rec["score"])
    assert 0.0 < rec["p"] <= 1.0 and rec["n_calibration"] == 30


# ---------------------------------------------------------------- the cost rule (milestone 2b)
class FixedModel:
    """The same outcome vector for every row, so the best row is the fastest one (8 s)."""

    kind = "classifier"

    def __init__(self, **p):
        self.p = {k: 0.0 for k in OUTCOMES}
        self.p.update(p)

    def predict_many(self, feats):
        return [Prediction(p=dict(self.p)) for _ in feats]


def _cmenu(care="normal", abstain_cost=1.0, threshold=None, **p):
    return build_menu(FixedModel(**p), Belief.prior(1e-4, "wood"), _geometry(), [P(0.3), P(0.5), P(0.7)], 20.0, care,
                      "table_centre", plan_fn, np.random.default_rng(0), threshold=threshold, abstain_cost=abstain_cost)


def _best(options):
    return min((o for o in options if o.kind in ("commit", "lower_risk")), key=lambda o: o.expected_cost)


def test_certain_harmless_failure_declines():
    options, rec = _cmenu(not_lifted=1.0)
    assert _best(options).expected_cost == pytest.approx(0.5 + 0.4 + 1.0) and rec == "decline"
    assert options[-1].kind == "decline" and options[-1].expected_cost == pytest.approx(1.0)


def test_commit_when_the_best_row_costs_less_than_abstain():
    options, rec = _cmenu(placed_stable=0.9, not_lifted=0.1)
    assert _best(options).expected_cost == pytest.approx(0.05 + 0.4 + 0.1) and rec == "commit"


def test_threshold_is_a_floor_when_set():
    options, rec = _cmenu(placed_stable=0.65, not_lifted=0.35)
    assert _best(options).expected_cost == pytest.approx(0.925) and rec == "commit"
    assert _cmenu(threshold=0.7, placed_stable=0.65, not_lifted=0.35)[1] == "decline"


def test_numeric_care_changes_the_recommendation():
    p = {"placed_stable": 0.85, "not_lifted": 0.10, "dropped_high": 0.05}
    options, rec = _cmenu(care=1.0, **p)
    assert _best(options).expected_cost == pytest.approx(0.85) and rec == "commit"
    options, rec = _cmenu(care=30.0, **p)
    assert _best(options).expected_cost == pytest.approx(8.10) and rec == "decline"
    reason = options[-1].reason
    assert reason.startswith("best row costs 8.10 > abstain 1.00") and "dropped_high" in reason


def test_abstain_cost_none_keeps_the_2a_rule():
    p = {"placed_stable": 0.75, "dropped_high": 0.25}
    options, rec = _cmenu(abstain_cost=None, **p)
    assert rec == "commit" and options[-1].expected_cost is None
    assert _cmenu(abstain_cost=1.0, **p)[1] == "decline"


def test_probe_preposterior_counts_abstain_after_the_probe():
    """mu 0.2 (row cost 3.0) or 0.5 (row cost 0.2), half each, and a push that tells which. Now the cheapest
    choice is to abstain (1.0 against 1.6). After a high reading the row (0.2) beats abstaining, after a low one
    abstaining stays: cost after the probe 0.5 x 1.0 + 0.5 x 0.2 = 0.6, value 0.4. Without an abstain cost the row
    is the only choice, and no reading changes it."""
    b = _particles([0.2, 0.5], [0.5, 0.5])
    C = [np.array([3.0, 0.2])]
    v = probe_value(b, C, PerfectPush(), np.zeros((2, 3)), 20.0, "normal", np.random.default_rng(0), k=400, abstain_cost=1.0)
    assert v["expected_cost_after"] == pytest.approx(0.6, abs=0.1) and v["evoi"] == pytest.approx(0.4, abs=0.1)
    v = probe_value(b, C, PerfectPush(), np.zeros((2, 3)), 20.0, "normal", np.random.default_rng(0), k=400, abstain_cost=None)
    assert v["evoi"] == 0.0


class _RiskyFlat(FlatWeigh):
    """A weigh that tells nothing and throws the object with P 0.1."""

    def risk(self, X):
        return np.full(len(X), 0.1)


def test_probe_terminal_risk_costs_irreversible_plus_abstain():
    b = _particles(np.linspace(0.1, 0.9, 20), np.ones(20))
    v = probe_value(b, [np.full(20, 2.0)], _RiskyFlat(), np.zeros((20, 3)), 20.0, "normal", np.random.default_rng(0),
                    abstain_cost=1.0)
    assert v["expected_cost_after"] == pytest.approx(0.1 * (5.0 + 1.0) + 0.9 * 1.0) and v["evoi"] == 0.0
    v = probe_value(b, [np.full(20, 2.0)], _RiskyFlat(), np.zeros((20, 3)), 20.0, 10.0, np.random.default_rng(0),
                    abstain_cost=1.0)
    assert v["expected_cost_after"] == pytest.approx(0.1 * (50.0 + 1.0) + 0.9 * 1.0)


# ---------------------------------------------------------------- milestone 2b: the damage rows
CRUSHABLE = {"class": "crushable", "law": "cardboard", "fillable": False}
FILLABLE = {"class": "breakable", "law": "ceramic", "fillable": True}


def _damage_rows(feature, label, lo=-1.5, hi=1.5, flag="frag_crush", n=400):
    """Synthetic m2b rows: ``label`` when ``feature`` is above 0 (crush_margin) or below 0 (spill_margin), else placed."""
    from rpent.skills.pick_place.features import FEATURES_M2B
    from tests.unit_tests.skills.synthetic import synthetic_rows

    rng = np.random.default_rng(3)
    rows = []
    for i, r in enumerate(synthetic_rows(n, n_objects=20)):
        x = float(rng.uniform(lo, hi))
        f = {k: r["features"].get(k, 0.0) for k in FEATURES_M2B}
        f.update({flag: 1.0, feature: x})
        hit = x > 0 if feature == "crush_margin" else x < 0
        rows.append(dict(r, features=f, label=label if hit else "placed_stable"))
    return rows


def _frag_belief(n=64, f_damage=None, spill=None, fill=0.0):
    from rpent.skills.pick_place.belief import ParticleBelief

    f_damage = np.full(n, np.nan) if f_damage is None else np.asarray(f_damage, dtype=float)
    spill = np.full(n, np.nan) if spill is None else np.asarray(spill, dtype=float)
    parts = [{"mass": 0.2, "mu": 0.6, "com_obj": [0.0, 0.0, 0.02], "k_n_per_mm": 2.0, "v_bruise_ms": None, "v_break_ms": None,
              "f_damage_n": None if not np.isfinite(f_damage[i]) else float(f_damage[i]),
              "fill": float(fill), "spill_angle_deg": None if not np.isfinite(spill[i]) else float(spill[i])} for i in range(n)]
    return ParticleBelief(parts, np.ones(n))


def _dmenu(model, belief, entry, care="normal", abstain_cost=None, threshold=0.7):
    from rpent.skills.pick_place.menu import geometry_pose

    g = _geometry()
    return build_menu(model, belief, g, [P(0.3), P(0.5), P(0.7)], 20.0, care, "table_centre", plan_fn, np.random.default_rng(0),
                      pose_fn=lambda prop: dict(geometry_pose(g, prop), entry=entry), threshold=threshold, abstain_cost=abstain_cost)


def test_light_grip_lowers_crush_risk_on_a_toy_model():
    """A gbm on synthetic rows, crushed iff crush_margin > 0; the belief's f_damage_n spans 0.7 to 1.5 f_std: P(crushed) of
    light_grip (0.5 f_std) < commit_std < firm_grip (2 f_std)."""
    from rpent.skills.pick_place.features import FEATURES_M2B
    from rpent.skills.pick_place.success_model import fit_success_model

    model = fit_success_model(_damage_rows("crush_margin", "crushed"), names=FEATURES_M2B)
    options, _ = _dmenu(model, _frag_belief(f_damage=np.linspace(14.0, 30.0, 64)), CRUSHABLE)
    o = _by_id(options)
    assert o["light_grip"].kind == "lower_risk" and o["light_grip"].action["force"] == 10.0
    assert o["light_grip"].reason == "half the squeeze: less crush and bruise risk, more slip"
    assert o["light_grip"].pred.p["crushed"] < o["commit_std"].pred.p["crushed"] < o["firm_grip"].pred.p["crushed"]


def test_gentle_carry_lowers_spill_risk_on_a_toy_model():
    """A gbm on synthetic rows, spilled iff spill_margin < 0; a mug filled to 0.6 with spill angles of 10 to 40 deg: the
    standard carry (2.0 m/s^2, 23.05 deg) spills some, the gentle carry (0.5 m/s^2, 5.83 deg) none."""
    from rpent.skills.pick_place.features import FEATURES_M2B
    from rpent.skills.pick_place.success_model import fit_success_model

    model = fit_success_model(_damage_rows("spill_margin", "spilled", flag="fillable"), names=FEATURES_M2B)
    options, _ = _dmenu(model, _frag_belief(spill=np.linspace(10.0, 40.0, 64), fill=0.6), FILLABLE)
    o = _by_id(options)
    assert o["gentle_carry"].kind == "lower_risk" and o["gentle_carry"].action["a_transport"] == 0.5
    assert o["gentle_carry"].reason == "a gentle carry: less spill risk, slower"
    assert all(r.action["a_transport"] == 2.0 for k, r in o.items() if k not in ("gentle_carry", "decline"))
    assert o["gentle_carry"].pred.p["spilled"] < o["commit_std"].pred.p["spilled"]


class WindowModel:
    """An empty force window: a squeeze that holds (crush_margin above log 0.75) crushes with P 0.15; a lighter one
    does not lift (P(not_lifted) 0.8) and still crushes with P 0.02."""

    kind = "classifier"

    def __init__(self):
        from rpent.skills.pick_place.features import FEATURES_M2B

        self.feature_names = FEATURES_M2B

    def predict_matrix(self, X):
        j = list(self.feature_names).index("crush_margin")
        P = np.zeros((len(X), len(OUTCOMES)))
        col = {k: i for i, k in enumerate(OUTCOMES)}
        holds = X[:, j] > np.log(0.75)
        P[holds, col["placed_stable"]], P[holds, col["crushed"]] = 0.85, 0.15
        P[~holds, col["placed_stable"]], P[~holds, col["not_lifted"]], P[~holds, col["crushed"]] = 0.18, 0.8, 0.02
        return P, None, None


def test_empty_window_declines_at_high_care():
    """Review Focus 1: every squeeze that holds crushes sometimes, every lighter one fails to lift. At care 30 the best row
    costs more than abstaining and the reason names crushed; at care 0.5 the rule commits."""
    belief = _frag_belief(f_damage=np.full(64, 20.0))
    options, rec = _dmenu(WindowModel(), belief, CRUSHABLE, care=30.0, abstain_cost=1.0)
    assert rec == "decline" and "crushed carries the largest expected cost" in options[-1].reason
    _, rec = _dmenu(WindowModel(), belief, CRUSHABLE, care=0.5, abstain_cost=1.0)
    assert rec == "commit"


def test_m2_models_get_no_damage_rows():
    """A model without the damage features (the m2 bundle) gets the milestone 2a rows only: the m2 menus stay as recorded."""
    options, _ = _menu()
    ids = {o.id for o in options}
    assert "light_grip" not in ids and "gentle_carry" not in ids and "commit_std" in ids
