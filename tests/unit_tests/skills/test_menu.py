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
           rng=None, **kw):
    belief = belief if belief is not None else _particles(np.random.default_rng(0).uniform(0.1, 0.9, 400), np.ones(400))
    return build_menu(model or MuGate(f_std=20.0), belief, _geometry(), proposals or [P(0.3), P(0.5), P(0.7)], 20.0, care,
                      destination, plan_fn, rng if rng is not None else np.random.default_rng(1), threshold=0.7,
                      probe_models=probe_models, guard=guard, **kw)


def _by_id(options):
    return {o.id: o for o in options}


def test_probe_rows_present_when_models_exist():
    options, _ = _pmenu(probe_models={"push": PerfectPush(p_tip=0.1), "weigh": FlatWeigh()})
    rows = _by_id(options)
    for pid, probe in (("probe_push", "push"), ("probe_weigh", "weigh")):
        o = rows[pid]
        assert o.kind == "probe" and o.pred is None and o.action == {"probe": probe, "s": 0.5, "force": weigh_force(20.0)}
        assert set(o.probe) == {"p_kind", "p_thrown", "evoi", "evoi_se", "expected_cost_after", "estimator"}
        tip = o.probe["p_kind"]["tipped"] * COST["tipped"] if probe == "push" else 0.0
        assert o.expected_cost == pytest.approx(TIME_COST_PER_S * PROBE_SECONDS[probe] + tip + o.probe["expected_cost_after"])
    assert rows["probe_push"].probe["p_kind"]["tipped"] == pytest.approx(0.1)
    assert [o.expected_cost for o in options[:-1]] == sorted(o.expected_cost for o in options[:-1])


def test_push_row_charges_a_tip_only_for_a_push_that_keeps_the_object():
    """The push's kinds are those of a push not thrown, so its tipping charge is P(not thrown) P(tipped | not thrown):
    a push that tips every object it keeps and throws half of them costs half a tip, not a whole one (Codex review
    of the fixes, plan 2b)."""
    options, _ = _pmenu(probe_models={"push": PerfectPush(p_tip=1.0, p_thrown=0.5)})
    o = _by_id(options)["probe_push"]
    tip = 0.5 * COST["tipped"]
    assert o.expected_cost == pytest.approx(TIME_COST_PER_S * PROBE_SECONDS["push"] + tip + o.probe["expected_cost_after"])


def test_a_milestone_2a_push_charges_its_whole_tip():
    """A milestone 2a push model learned its kinds on every push, and no thrown push of its corpus tipped, so its
    P(tipped) is already that of a push that keeps the object: the whole tip is charged (the m2 menus stay pinned)."""
    push = PerfectPush(p_tip=1.0, p_thrown=0.5)
    push.kinds_not_thrown = False
    o = _by_id(_pmenu(probe_models={"push": push})[0])["probe_push"]
    assert o.expected_cost == pytest.approx(TIME_COST_PER_S * PROBE_SECONDS["push"] + COST["tipped"] + o.probe["expected_cost_after"])


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


# ---------------------------------------------------------------- milestone 2b: damage in every probe, the squeeze row
from rpent.skills.pick_place.probes import PROBE_KINDS as _KINDS  # noqa: E402
from tests.unit_tests.skills.probe_stubs import FlatSqueeze  # noqa: E402

STEEL = {"class": "none", "law": None, "fillable": False}
BRUISABLE = {"class": "bruisable", "law": "fruit", "fillable": False}


def _const_probe(probe, classes, p, p_thrown=0.0, reading=None):
    from rpent.skills.pick_place.probes import CalibratedClassifier, ProbeModel

    clf = CalibratedClassifier(tuple(classes), None, None, np.array(p, dtype=float))
    return ProbeModel(probe=probe, classes=tuple(classes), clf=clf, reading=reading or {}, p_thrown=p_thrown)


def test_probe_value_counts_damage_risk():
    """A weigh that damages with P 0.2 and tells nothing, at care 10 and A 1: unused, the probe costs 0.2 x (5 x 10 + 1)
    = 10.2 and 0.8 x the best continuation (abstaining, 1.0)."""
    pm = _const_probe("weigh", _KINDS["weigh"] + ("damaged",), [0.8, 0.0, 0.0, 0.2])
    b = _particles(np.linspace(0.1, 0.9, 20), np.ones(20))
    v = probe_value(b, [np.full(20, 2.0)], pm, np.zeros((20, 3)), 20.0, 10.0, np.random.default_rng(0), abstain_cost=1.0)
    assert v["expected_cost_after"] == pytest.approx(0.2 * 51.0 + 0.8 * 1.0) and v["evoi"] == 0.0
    assert v["p_kind"]["damaged"] == pytest.approx(0.2)


class _HalfDamagingPush(PerfectPush):
    """PerfectPush that damages the object with P 0.5 whatever its friction."""

    classes = _KINDS["push"] + ("damaged",)

    def p_kind(self, X):
        row = np.zeros(len(self.classes))
        row[0], row[-1] = 0.5, 0.5
        return np.tile(row, (len(X), 1))

    def terminal_risk(self, X):
        return np.full(len(X), 0.5)

    def sample(self, x_row, particle, force, rng, p=None):
        p = self.p_kind(np.zeros((1, 1)))[0] if p is None else np.asarray(p, dtype=float)
        if self.classes[int(rng.choice(len(self.classes), p=p / p.sum()))] == "damaged":
            return {"probe": "push", "kind": "damaged", "reading": {}, "thrown": False}
        return super().sample(x_row, particle, force, rng)

    def loglik(self, obs, X, arrays, force, temper=None, terms=None, P=None):
        if obs["kind"] == "damaged":
            return np.zeros(len(X)), {"used": ["kind"], "rejected": []}
        return super().loglik(obs, X, arrays, force, temper, terms, P)


def test_probe_value_draws_results_that_keep_the_object():
    """The push of test_probe_preposterior_counts_abstain_after_the_probe (value 0.4), now damaging with P 0.5: a
    result is drawn among those that keep the object, so the value is 0.5 x 0.4, not 0.25 x 0.4, and the cost
    after it is 0.5 x (5 + 1) + 0.5 x 1.0 - 0.2."""
    b = _particles([0.2, 0.5], [0.5, 0.5])
    v = probe_value(b, [np.array([3.0, 0.2])], _HalfDamagingPush(), np.zeros((2, 3)), 20.0, "normal", np.random.default_rng(0),
                    k=400, abstain_cost=1.0)
    assert v["evoi"] == pytest.approx(0.2, abs=0.05) and v["expected_cost_after"] == pytest.approx(3.5 - 0.2, abs=0.05)


def _sqmenu(entry, belief, probe_models):
    from rpent.skills.pick_place.menu import geometry_pose

    g = _geometry()
    return build_menu(MuGate(f_std=20.0), belief, g, [P(0.3), P(0.5), P(0.7)], 20.0, "normal", "table_centre", plan_fn,
                      np.random.default_rng(0), pose_fn=lambda prop: dict(geometry_pose(g, prop), entry=entry), threshold=0.7,
                      probe_models=probe_models)


def test_no_squeeze_row_for_non_fragile_objects():
    """The squeeze row needs a crushable or bruisable object, and it presses at the cap squeeze_cap(belief, f_std): here
    the 10 % quantile of f_damage_n (about 8.4 N), under half the standard squeeze (10 N)."""
    from rpent.skills.pick_place.probes import squeeze_cap

    belief = _frag_belief(f_damage=np.linspace(6.0, 30.0, 64))
    for entry, want in ((STEEL, False), (FILLABLE, False), (CRUSHABLE, True), (BRUISABLE, True)):
        rows = _by_id(_sqmenu(entry, belief, {"squeeze": FlatSqueeze(), "weigh": FlatWeigh()})[0])
        assert ("probe_squeeze" in rows) == want and "probe_weigh" in rows
        if want:
            cap = squeeze_cap(belief, 20.0)
            assert cap < 10.0 and rows["probe_squeeze"].action == {"probe": "squeeze", "s": 0.5, "force": cap}
            from rpent.skills.pick_place.features import G
            from rpent.skills.pick_place.menu import PLACEHOLDER

            load = PLACEHOLDER["mass"] * G / (2 * PLACEHOLDER["mu"] * cap)        # the base is extracted at the cap
            assert rows["probe_squeeze"]._base["load_ratio"] == pytest.approx(load)
            assert rows["probe_weigh"].action["force"] == weigh_force(20.0)


def _squeeze_world(rho: float, n: int = 1000, seed: int = 0):
    """Particles whose log stiffness has correlation rho with the log damage force (each sd 0.4), a squeeze that reads
    the stiffness (log sd 0.10) and never damages, and two rows: commit_std crushes an object weaker than 15 N
    (cost 5 + 0.4), light_grip costs 0.9 whatever the object."""
    from rpent.skills.pick_place.belief import ParticleBelief
    from rpent.skills.pick_place.likelihood import Law

    z1, z2 = np.random.default_rng(seed).standard_normal((2, n))
    lf, lk = np.log(20.0) + 0.4 * z1, np.log(2.0) + 0.4 * (rho * z1 + np.sqrt(1.0 - rho ** 2) * z2)
    b = ParticleBelief([{"mass": 0.3, "mu": 0.5, "com_obj": [0.0, 0.0, 0.03], "k_n_per_mm": float(np.exp(a)),
                         "f_damage_n": float(np.exp(c)), "v_bruise_ms": None, "v_break_ms": None} for a, c in zip(lk, lf)], np.ones(n))
    pm = _const_probe("squeeze", ("measured",), [1.0], reading={"measured": Law.normal(0.0, 0.10)})
    return b, pm, [np.where(np.exp(lf) < 15.0, 5.0, 0.0) + 0.4, np.full(n, 0.9)]


def test_squeeze_is_worthless_without_correlation():
    """The squeeze reads only the stiffness. At rho 0 it tells nothing about the damage force: its value is within 2
    Monte Carlo errors of 0. At rho 0.9 the same squeeze is worth more than 2 errors."""
    for rho, worth in ((0.0, False), (0.9, True)):
        b, pm, C = _squeeze_world(rho)
        v = probe_value(b, C, pm, np.zeros((len(b), 3)), 5.0, "normal", np.random.default_rng(1), k=400, abstain_cost=1.0)
        assert (v["evoi"] > 2.0 * v["evoi_se"]) == worth, (rho, v)


class _SurvivalWeigh:
    """A weigh whose reading tells nothing and that damages particle i with P X[i, 0]: only its survival informs. Its
    kind term reads ``P`` when the caller passes it, as ProbeModel.loglik does."""

    probe = "weigh"
    classes = _KINDS["weigh"] + ("damaged",)

    def p_kind(self, X):
        d = np.asarray(X, dtype=float)[:, 0]
        P = np.zeros((len(d), len(self.classes)))
        P[:, 0], P[:, -1] = 1.0 - d, d
        return P

    def risk(self, X):
        return np.zeros(len(X))

    def terminal_risk(self, X):
        return np.asarray(X, dtype=float)[:, 0]

    def sample(self, x_row, particle, force, rng, p=None):
        return {"probe": "weigh", "kind": "measured", "reading": {}, "thrown": False}

    def loglik(self, obs, X, arrays, force, temper=None, terms=None, P=None):
        P = self.p_kind(X) if P is None else np.asarray(P, dtype=float)
        return np.log(np.maximum(1e-4, P[:, list(self.classes).index(obs["kind"])])), {"used": ["kind"], "rejected": []}


def test_probe_value_weighs_each_particles_survival_once():
    """Two particles; the weigh damages the first with P 0.9 and never the second, and its reading tells nothing. With
    A 1 the best row now is abstaining (1.0 against 1.6). A kept object is the second particle with P 0.5/0.55, so the
    row (3.0, 0.2) is picked after it: cost after 0.45 T + 0.05 x 3.0 + 0.5 x 0.2, value 0.3 (Codex review, plan 2b)."""
    from rpent.skills.pick_place.costs import COST, care_scale

    b = _particles([0.2, 0.5], [0.5, 0.5])
    X = np.array([[0.9, 0.0, 0.0], [0.0, 0.0, 0.0]])
    v = probe_value(b, [np.array([3.0, 0.2])], _SurvivalWeigh(), X, 20.0, "normal", np.random.default_rng(0), k=50,
                    abstain_cost=1.0)
    T = COST["dropped_high"] * care_scale("normal") + 1.0
    assert v["evoi"] == pytest.approx(0.3) and v["expected_cost_after"] == pytest.approx(0.45 * T + 0.15 + 0.1)


# ---------------------------------------------------------------- plan 2b2 decision 2: the held-out preposterior
_TYPE_COSTS = np.array([[0.2, 3.0], [1.0, 1.0], [0.6, 2.0]])     # three rows x two types (no abstain row)


class _GaussReading:
    """A weigh that never damages or throws, whose kind tells nothing and whose reading is X[i, 0] + sd x N(0, 1). The
    kind-only replay (``terms`` without "reading") is flat."""

    probe = "weigh"
    classes = ("measured",)

    def __init__(self, sd):
        self.sd = sd

    def p_kind(self, X):
        return np.ones((len(X), 1))

    def risk(self, X):
        return np.zeros(len(X))

    def terminal_risk(self, X):
        return np.zeros(len(X))

    def sample(self, x_row, particle, force, rng, p=None):
        return {"probe": "weigh", "kind": "measured", "reading": {"y": float(x_row[0]) + self.sd * float(rng.standard_normal())},
                "thrown": False}

    def loglik(self, obs, X, arrays, force, temper=None, terms=None, P=None):
        if terms is not None and "reading" not in terms:
            return np.zeros(len(X)), {"used": ["kind"], "rejected": []}
        z = (obs["reading"]["y"] - np.asarray(X, dtype=float)[:, 0]) / self.sd
        return -0.5 * z ** 2, {"used": ["kind", "reading"], "rejected": []}


def _typed(n, rng, null=False):
    """n equally weighted particles of two types (half each in the population), the rows' costs by type, and X with the
    reading's centre in column 0: the type, or with ``null`` a draw independent of it."""
    t = rng.integers(0, 2, n)
    X = np.zeros((n, 3))
    X[:, 0] = rng.standard_normal(n) if null else t
    return _particles(np.full(n, 0.5), np.ones(n)), list(_TYPE_COSTS[:, t]), X


def test_heldout_value_is_not_positive_for_an_uninformative_result():
    """Two particles, a flat weigh, rows (0, 2) and (0.9, 0.9): row b is the second (0.9 against 1.0), and the plug-in
    keeps it (value 0). Held out, particle 1 is scored with the row chosen from particle 0 alone (the first, cost 2
    for particle 1): 0.5 x (0.9 - 2) = -0.55 for every result, and the cost after the probe is 0.9 + 0.55."""
    b = _particles([0.2, 0.5], [0.5, 0.5])
    C = [np.array([0.0, 2.0]), np.array([0.9, 0.9])]
    v = probe_value(b, C, FlatWeigh(), np.zeros((2, 3)), 20.0, "normal", np.random.default_rng(0), k=50, estimator="heldout")
    assert v["estimator"] == "heldout" and v["evoi"] == pytest.approx(-0.55) and v["plugin_evoi"] == 0.0
    assert v["expected_cost_after"] == pytest.approx(0.9 + 0.55) and v["evoi_se"] == pytest.approx(0.0, abs=1e-12)
    p = probe_value(b, C, FlatWeigh(), np.zeros((2, 3)), 20.0, "normal", np.random.default_rng(0), k=50)
    assert p["estimator"] == "plugin" and p["evoi"] == 0.0 and "plugin_evoi" not in p


def test_heldout_value_finds_no_value_in_a_reading_independent_of_the_costs():
    """Each particle's reading centre is independent of its type, so the reading is worth 0 in the population. The
    plug-in finds value in the particles' chance correlation (the optimizer's curse); the held-out value does not.
    Over 80 beliefs of 50 particles (sigma 0.5), the held-out mean is at most 2 between-belief errors above 0, and the
    plug-in's mean on the same draws is more than 2 above."""
    rng = np.random.default_rng(11)
    vals = []
    for _ in range(80):
        b, C, X = _typed(50, rng, null=True)
        v = probe_value(b, C, _GaussReading(0.5), X, 20.0, "normal", rng, k=200, estimator="heldout")
        vals.append((v["evoi"], v["plugin_evoi"]))
    m, se = np.mean(vals, axis=0), np.std(vals, axis=0, ddof=1) / np.sqrt(len(vals))
    assert m[0] <= 2 * se[0] and m[1] > 2 * se[1], (m, se)


def test_heldout_value_matches_the_exact_value_of_a_perfect_reading():
    """The throw-risk push of test_evoi_counts_each_particles_throw_risk reads the side of mu 0.3 exactly: value 0.3 f.
    Removing one particle never changes the row chosen for a side, so the held-out value equals the plug-in's on the
    same draws and lies within 3 Monte Carlo errors of the exact value (the cost after it likewise)."""
    b = _particles(np.random.default_rng(0).uniform(0.1, 0.9, 400), np.ones(400))
    mu = b.arrays()["mu"]
    f = float(np.mean(mu > 0.3))
    C = [np.where(mu > 0.3, 0.4, 5.4), np.full(400, 0.7)]
    X = np.zeros((400, 3))
    X[:, 0] = mu
    v = probe_value(b, C, RiskyPush(), X, 20.0, "normal", np.random.default_rng(3), k=4000, estimator="heldout")
    assert v["evoi"] == pytest.approx(v["plugin_evoi"], abs=1e-12)
    assert abs(v["evoi"] - 0.3 * f) <= 3 * v["evoi_se"]
    assert abs(v["expected_cost_after"] - (f * 0.4 + (1 - f) * (2.5 + 0.35))) <= 3 * v["evoi_se"]


def test_heldout_value_matches_the_quadrature_value_of_a_noisy_reading():
    """Two types (half each), a Gaussian reading of the type (sigma 1): the population's value is 0.0571 by quadrature
    (plans/2026-10-30-grasp-place-milestone2b2-evoi-check.py in daily-logs). Over 20 beliefs of 2,000 particles the
    held-out mean is within 3 between-belief errors of it."""
    rng = np.random.default_rng(12)
    vals = []
    for _ in range(20):
        b, C, X = _typed(2000, rng)
        vals.append(probe_value(b, C, _GaussReading(1.0), X, 20.0, "normal", rng, k=200, estimator="heldout")["evoi"])
    assert abs(np.mean(vals) - 0.0571) <= 3 * np.std(vals, ddof=1) / np.sqrt(len(vals)), vals


def test_reading_value_isolates_the_reading_from_survival():
    """The weigh of test_probe_value_weighs_each_particles_survival_once: only the survival informs (plug-in value 0.3).
    Its reading tells nothing, so the reading value is 0 for both estimators. Held out, the total is -0.1: particle 0,
    the one the weigh damages with P 0.9, is scored at weight 0.05 with the row chosen from particle 1 alone (3.0
    against abstaining's 1.0); particle 1 is scored with row b."""
    b = _particles([0.2, 0.5], [0.5, 0.5])
    X = np.array([[0.9, 0.0, 0.0], [0.0, 0.0, 0.0]])
    for est in ("plugin", "heldout"):
        v = probe_value(b, [np.array([3.0, 0.2])], _SurvivalWeigh(), X, 20.0, "normal", np.random.default_rng(0), k=50,
                        abstain_cost=1.0, estimator=est, reading_value=True)
        assert v["reading_gain"] == 0.0 and v["reading_gain_se"] == 0.0
    assert v["evoi"] == pytest.approx(-0.1) and v["plugin_evoi"] == pytest.approx(0.3) and v["plugin_reading_gain"] == 0.0


def test_reading_value_is_the_reading_s_share_of_the_value():
    """The Gaussian reading of the quadrature test with a flat kind: all of the value is the reading's, so the reading
    value equals the total for both estimators (the kind-only rule keeps row b)."""
    b, C, X = _typed(400, np.random.default_rng(13))
    for est in ("plugin", "heldout"):
        v = probe_value(b, C, _GaussReading(0.5), X, 20.0, "normal", np.random.default_rng(14), k=200, estimator=est,
                        reading_value=True)
        assert v["reading_gain"] == pytest.approx(v["evoi"], abs=1e-12) and v["evoi"] > 0.1
    assert v["plugin_reading_gain"] == pytest.approx(v["plugin_evoi"], abs=1e-12)


class _KindTellsReadingRejected:
    """A weigh whose kind tells the type in X[:, 0] (P(measured) 0.1 for type 0, 0.9 for type 1) and whose reading is
    always rejected; it never damages or throws."""

    probe = "weigh"
    classes = ("measured", "lower_bound")

    def p_kind(self, X):
        p = np.where(np.asarray(X, dtype=float)[:, 0] > 0.5, 0.9, 0.1)
        return np.column_stack([p, 1.0 - p])

    def risk(self, X):
        return np.zeros(len(X))

    def terminal_risk(self, X):
        return np.zeros(len(X))

    def sample(self, x_row, particle, force, rng, p=None):
        p = self.p_kind(np.asarray(x_row, dtype=float).reshape(1, -1))[0] if p is None else np.asarray(p, dtype=float)
        return {"probe": "weigh", "kind": self.classes[int(rng.choice(2, p=p / p.sum()))], "reading": {"m_hat": -1.0},
                "thrown": False}

    def loglik(self, obs, X, arrays, force, temper=None, terms=None, P=None):
        P = self.p_kind(X) if P is None else np.asarray(P, dtype=float)
        return np.log(P[:, self.classes.index(obs["kind"])]), {"used": ["kind"], "rejected": ["reading"]}


def test_rejected_reading_term_has_no_reading_value():
    """The kind is worth about 0.5 x (1.0 - 0.48) = 0.26 (a lower_bound result makes the first row the cheapest); a
    rejected reading adds nothing, so its value is exactly 0."""
    b, C, X = _typed(400, np.random.default_rng(15))
    for est in ("plugin", "heldout"):
        v = probe_value(b, C, _KindTellsReadingRejected(), X, 20.0, "normal", np.random.default_rng(16), k=200, estimator=est,
                        reading_value=True)
        assert v["evoi"] > 0.15 and v["reading_gain"] == 0.0 and v["reading_gain_se"] == 0.0


def test_heldout_estimator_handles_a_belief_on_one_particle():
    """A belief whose weight sits on one particle, and a perfect push whose result names the particle when it is
    alone on its side: the held-out value is 0 (a chooser with no posterior weight keeps row b), with no exception and
    no nan; the plug-in finds value in the lone particle."""
    point = _particles(np.linspace(0.1, 0.9, 50), [1.0] + [0.0] * 49)
    C = [np.where(point.arrays()["mu"] > 0.3, 0.4, 5.4), np.full(50, 0.7)]
    for rv in (False, True):
        v = probe_value(point, C, PerfectPush(), np.zeros((50, 3)), 20.0, "normal", np.random.default_rng(0), estimator="heldout",
                        reading_value=rv)
        assert v["evoi"] == 0.0 and v["evoi_se"] == 0.0 and np.isfinite(v["expected_cost_after"])
    lone = _particles([0.9] + [0.1] * 9, np.ones(10))
    C = [np.where(lone.arrays()["mu"] > 0.3, 0.4, 5.4), np.full(10, 0.7)]
    v = probe_value(lone, C, PerfectPush(), np.zeros((10, 3)), 20.0, "normal", np.random.default_rng(0), k=400, estimator="heldout",
                    reading_value=True)
    assert v["evoi"] == 0.0 and v["reading_gain"] == 0.0 and v["plugin_evoi"] > 0.0 and np.isfinite(v["expected_cost_after"])


def test_unknown_estimator_is_rejected():
    b = _particles([0.2, 0.5], [0.5, 0.5])
    with pytest.raises(ValueError, match="estimator"):
        probe_value(b, [np.array([0.0, 2.0])], FlatWeigh(), np.zeros((2, 3)), 20.0, "normal", np.random.default_rng(0),
                    estimator="split")
    with pytest.raises(ValueError, match="estimator"):
        build_menu(MuGate(f_std=20.0), b, _geometry(), [P(0.3), P(0.5), P(0.7)], 20.0, "normal", "table_centre", plan_fn,
                   np.random.default_rng(1), probe_models={"push": PerfectPush()}, evoi="split")


def test_menu_values_its_probe_rows_with_the_named_estimator():
    """build_menu's evoi names the probe rows' estimator; the plug-in is the default (m2 and m2b reproduce)."""
    for kw, want in (({}, "plugin"), ({"evoi": "heldout"}, "heldout")):
        options, _ = build_menu(MuGate(f_std=20.0), _particles(np.random.default_rng(0).uniform(0.1, 0.9, 400), np.ones(400)),
                                _geometry(), [P(0.3), P(0.5), P(0.7)], 20.0, "normal", "table_centre", plan_fn,
                                np.random.default_rng(1), threshold=0.7, probe_models={"push": PerfectPush(), "weigh": FlatWeigh()},
                                **kw)
        assert {o.probe["estimator"] for o in options if o.kind == "probe"} == {want}


# ---------------------------------------------------------------- plan 2b2 decision 4: the probe time scale
def test_probe_time_scale_scales_only_the_probe_time_charge():
    """A probe row's expected cost moves by (1 - s) x TIME_COST_PER_S x its seconds; no other row moves, and the probe
    values and seconds do not (the same random stream)."""
    pms = {"push": PerfectPush(), "weigh": FlatWeigh()}
    base = _by_id(_pmenu(probe_models=pms)[0])
    for s in (0.25, 3.0):
        rows = _by_id(_pmenu(probe_models=pms, probe_time_scale=s)[0])
        assert set(rows) == set(base)
        for rid, o in rows.items():
            want = base[rid].expected_cost
            if o.kind == "probe":
                want -= (1 - s) * TIME_COST_PER_S * PROBE_SECONDS[o.action["probe"]]
                assert o.probe == base[rid].probe and o.cost_s == base[rid].cost_s
            assert o.expected_cost is None if want is None else o.expected_cost == pytest.approx(want)


def test_zero_probe_time_scale_removes_the_charge():
    """s = 0: a probe row costs its tip charge and the expected cost after it, nothing for its time."""
    rows = _by_id(_pmenu(probe_models={"push": PerfectPush(p_tip=0.1), "weigh": FlatWeigh()}, probe_time_scale=0.0)[0])
    for pid in ("probe_push", "probe_weigh"):
        o = rows[pid]
        tip = o.probe["p_kind"]["tipped"] * COST["tipped"] if pid == "probe_push" else 0.0
        assert o.expected_cost == pytest.approx(tip + o.probe["expected_cost_after"])


@pytest.mark.parametrize("bad", [-0.5, float("nan"), float("inf"), "x", None, True])
def test_bad_probe_time_scale_is_an_error(bad):
    with pytest.raises(ValueError, match="probe time scale"):
        _pmenu(probe_models={"push": PerfectPush()}, probe_time_scale=bad)


# ---------------------------------------------------------------- the final review of plan 2b2: copies of one draw
def _copies(b, C, X, m):
    """The belief b as m copies of each particle with its index as ``ancestor`` (a refresh resamples copies of its prior
    draws), with the costs and the probe features repeated to match."""
    from rpent.skills.pick_place.belief import ParticleBelief

    parts = [dict(p, ancestor=i) for i, p in enumerate(b.particles) for _ in range(m)]
    return ParticleBelief(parts, np.ones(len(parts))), [np.repeat(c, m) for c in C], np.repeat(X, m, axis=0)


def test_heldout_value_of_copies_equals_the_value_of_their_draws():
    """A refreshed belief holds copies of its prior draws. Leaving out one copy keeps its twins with the chooser and
    brings the chance correlation back; the held-out value leaves out every copy of the particle's draw. So 10 copies of
    each of 50 null draws have the values of the 50 draws on the same random stream: the plug-in's, the held-out value
    and the reading's held-out value."""
    rng = np.random.default_rng(5)
    for _ in range(4):
        b, C, X = _typed(50, rng, null=True)
        bc, Cc, Xc = _copies(b, C, X, 10)
        seed = int(rng.integers(1 << 30))
        one, ten = (probe_value(bb, CC, _GaussReading(0.5), XX, 20.0, "normal", np.random.default_rng(seed), k=200,
                                estimator="heldout", reading_value=True) for bb, CC, XX in ((b, C, X), (bc, Cc, Xc)))
        for key in ("plugin_evoi", "evoi", "plugin_reading_gain", "reading_gain"):
            assert ten[key] == pytest.approx(one[key], abs=1e-9), (key, ten[key], one[key])


def test_heldout_rows_of_distinct_draws_are_the_per_particle_rows():
    """Every particle its own draw (a first menu: the prior's particles carry distinct ancestors): the rows equal the
    per-particle rows exactly, so plan 2b2's first menus and its squeeze sensitivity keep their values."""
    from rpent.skills.pick_place.menu import heldout_rows

    rng = np.random.default_rng(3)
    C, u = rng.random((7, 300)), rng.random(300)
    groups = rng.permutation(300)
    assert np.array_equal(heldout_rows(C, u, 2, groups=groups), heldout_rows(C, u, 2))
    u[5:] = 0.0                                                     # the first five keep all the weight
    assert np.array_equal(heldout_rows(C, u, 2, groups=groups), heldout_rows(C, u, 2))


# ---------------------------------------------------------------- the exact no-probe recommendation (milestone 4)
from rpent.skills.pick_place.protocol import Option  # noqa: E402


def _row(id_, kind, cost, p_ok=0.9):
    pred = None if kind == "probe" else Prediction(p={**{k: 0.0 for k in OUTCOMES}, "placed_stable": p_ok, "not_lifted": 1.0 - p_ok})
    action = {"probe": "push"} if kind == "probe" else {"destination": "table_centre"}
    return Option(id=id_, kind=kind, action=action, pred=pred, cost_s=8.0, expected_cost=cost)


def test_without_probe_uses_the_unrounded_cost():
    """Plan m4 decision 2: the recommendation without the probe rows comes from the unrounded costs (the payload rounds
    to 0.001, so a best row within 0.0005 of A read back from it can go either way)."""
    from rpent.skills.pick_place.menu import without_probe

    probe = _row("probe_push", "probe", 0.5)
    assert without_probe([probe, _row("commit_std", "commit", 1.0004)], "probe", "table_centre", "normal", 1.0, None) == "decline"
    assert without_probe([probe, _row("commit_std", "commit", 0.9996)], "probe", "table_centre", "normal", 1.0, None) == "commit"
    assert without_probe([_row("commit_std", "commit", 0.9996)], "decline", "table_centre", "normal", 1.0, None) == "decline"
    assert without_probe([_row("commit_std", "commit", 5.0)], "commit", "table_centre", "normal", 1.0, None) == "commit"
    assert without_probe([probe, _row("commit_std", "commit", 0.9, p_ok=0.6)], "probe", "table_centre", "normal", 1.0, 0.7) == "decline"
    assert without_probe([probe, _row("commit_std", "commit", 5.0, p_ok=0.75)], "probe", "table_centre", "normal", None, 0.7) == "commit"
    assert without_probe([probe, _row("commit_std", "commit", 5.0, p_ok=0.65)], "probe", "table_centre", "normal", None, 0.7) == "decline"
    assert without_probe([probe], "probe", "table_centre", "normal", 1.0, None) == "decline"


# ---------------------------------------------------------------- the recovery mode (plan H3 decision 3)
def test_the_probe_value_under_a_reset():
    """Under a reset a probe that throws or damages the object costs the reset's time plus the best continuation row
    under the pre-probe belief (row 1, 0.7 here), in place of 5 kappa + A; the kept results keep their value. A sure
    throw: 1.0 + 0.7 against 5 x 10 + A. RiskyPush (thrown with P 0.5 where mu is at most 0.3): the cost after is
    f x 0.4 + (1 - f) x (0.5 x 1.7 + 0.5 x 0.7), the value 0.3 f as without the reset."""
    b = _particles(np.linspace(0.1, 0.9, 40), np.ones(40))
    C = [np.where(b.arrays()["mu"] > 0.3, 0.4, 5.4), np.full(40, 0.7)]
    for A in (None, 10.0):
        args = (b, C, PerfectPush(p_thrown=1.0), np.zeros((40, 3)), 20.0, 10.0)
        absorbing = probe_value(*args, np.random.default_rng(0), abstain_cost=A)
        reset = probe_value(*args, np.random.default_rng(0), abstain_cost=A, reset_s=20.0)
        assert absorbing["expected_cost_after"] == pytest.approx(5 * 10.0 + (A or 0.0))
        assert reset["expected_cost_after"] == pytest.approx(TIME_COST_PER_S * 20.0 + 0.7) and reset["evoi"] == 0.0
    b = _particles(np.random.default_rng(0).uniform(0.1, 0.9, 400), np.ones(400))
    mu = b.arrays()["mu"]
    f = float(np.mean(mu > 0.3))
    X = np.zeros((400, 3))
    X[:, 0] = mu
    v = probe_value(b, [np.where(mu > 0.3, 0.4, 5.4), np.full(400, 0.7)], RiskyPush(), X, 20.0, "normal",
                    np.random.default_rng(3), k=4000, reset_s=20.0)
    assert v["evoi"] == pytest.approx(0.3 * f, abs=0.012)
    assert v["expected_cost_after"] == pytest.approx(f * 0.4 + (1 - f) * (0.5 * 1.7 + 0.5 * 0.7), abs=0.012)


def test_the_menu_prices_every_row_with_the_reset(monkeypatch):
    """build_menu's reset_s reaches every row: the attempt rows' expected and per-particle costs, and each probe row's
    preposterior (probe_value)."""
    import rpent.skills.pick_place.menu as menu_mod

    seen = []
    real = menu_mod.probe_value
    monkeypatch.setattr(menu_mod, "probe_value", lambda *a, **k: seen.append(k.get("reset_s")) or real(*a, **k))
    b = _particles(np.random.default_rng(0).uniform(0.1, 0.9, 400), np.ones(400))
    options, _ = _pmenu(probe_models={"push": PerfectPush(p_tip=0.1), "weigh": FlatWeigh()}, belief=b, care=10.0,
                        abstain_cost=10.0, reset_s=20.0)
    assert seen == [20.0, 20.0]
    rows = [o for o in options if o.kind in ("commit", "lower_risk")]
    assert any(sum(o.pred.p[k] for k in IRREVERSIBLE) > 0 for o in rows)       # MuGate drops high: the reset matters
    for o in rows:
        assert o.expected_cost == pytest.approx(expected_cost(o.pred, o.cost_s, 10.0, 10.0, reset_s=20.0))
        assert float(b.weights @ o._costs) == pytest.approx(o.expected_cost)


def test_the_cost_rule_names_the_costliest_outcome_under_the_reset():
    """The decline reason names the outcome that carries the largest expected cost under the costs in force: dropped_high
    (0.1 x 50) when absorbing at care 10, tipped (0.4 x 1 against 0.1 x 1.0) under a 20 s reset."""
    from rpent.skills.pick_place.menu import _cost_rule
    from rpent.skills.pick_place.protocol import Option

    p = {k: 0.0 for k in OUTCOMES}
    p.update(placed_stable=0.5, dropped_high=0.1, tipped=0.4)
    best = Option(id="commit_std", kind="commit", action={"destination": "table_centre"}, pred=Prediction(p=p), cost_s=8.0,
                  expected_cost=5.0, reason="")
    assert "dropped_high carries" in _cost_rule([best], best, 10.0, 0.1, None)[1]
    assert "tipped carries" in _cost_rule([best], best, 10.0, 0.1, None, reset_s=20.0)[1]
    assert "dropped_high carries" in _cost_rule([best], best, 10.0, 0.1, None, reset_s=0.0, reset_price="care")[1]


def test_the_probe_value_under_the_care_price():
    """rec_hi (plan H3 Addendum A): a probe that throws costs 5 kappa plus the best continuation row now (0.7), in place
    of 5 kappa + A (absorbing) or the reset's time plus that row (rec20)."""
    b = _particles(np.linspace(0.1, 0.9, 40), np.ones(40))
    C = [np.where(b.arrays()["mu"] > 0.3, 0.4, 5.4), np.full(40, 0.7)]
    args = (b, C, PerfectPush(p_thrown=1.0), np.zeros((40, 3)), 20.0, 10.0)
    hi = probe_value(*args, np.random.default_rng(0), abstain_cost=10.0, reset_s=0.0, reset_price="care")
    assert hi["expected_cost_after"] == pytest.approx(5 * 10.0 + 0.7) and hi["evoi"] == 0.0


def test_the_care_price_prices_the_attempt_rows_as_absorbing(monkeypatch):
    """build_menu under rec_hi: every attempt row's expected and per-particle costs equal absorbing's (the one-step menu
    adds A after every unplaced outcome in every mode), and each probe row's preposterior gets the care price."""
    import rpent.skills.pick_place.menu as menu_mod

    seen = []
    real = menu_mod.probe_value
    monkeypatch.setattr(menu_mod, "probe_value",
                        lambda *a, **k: seen.append((k.get("reset_s"), k.get("reset_price"))) or real(*a, **k))
    b = _particles(np.random.default_rng(0).uniform(0.1, 0.9, 400), np.ones(400))
    kw = dict(probe_models={"push": PerfectPush(p_tip=0.1), "weigh": FlatWeigh()}, belief=b, care=10.0, abstain_cost=10.0)
    hi, _ = _pmenu(**kw, reset_s=0.0, reset_price="care")
    assert seen == [(0.0, "care"), (0.0, "care")]
    absorbing, _ = _pmenu(**kw)
    a = {o.id: o for o in absorbing if o.kind in ("commit", "lower_risk")}
    h = {o.id: o for o in hi if o.kind in ("commit", "lower_risk")}
    assert set(a) == set(h) and any(sum(o.pred.p[k] for k in IRREVERSIBLE) > 0 for o in a.values())
    for i in a:
        assert h[i].expected_cost == pytest.approx(a[i].expected_cost)
        assert h[i]._costs == pytest.approx(a[i]._costs)
