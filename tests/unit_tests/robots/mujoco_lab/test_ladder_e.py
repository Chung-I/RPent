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

"""E, the receding-horizon reference planner (ladder_e): the attempt recursion, the held-out choice and the probe value."""

import operator
import os

import numpy as np
import pytest

from robots.mujoco_lab import ladder_e
from rpent.skills.pick_place.belief import ParticleBelief
from rpent.skills.pick_place.costs import COST, TIME_COST_PER_S, particle_costs
from rpent.skills.pick_place.menu import heldout_rows, probe_value
from rpent.skills.pick_place.protocol import IRREVERSIBLE, OUTCOMES, RECOVERABLE, Prediction

from tests.unit_tests.skills.probe_stubs import FlatWeigh, PerfectPush

KAPPA, A = 100.0, 3.0


def _random_P(rng, R, N):
    P = rng.dirichlet(np.ones(len(OUTCOMES)) * 0.4, size=(R, N))
    return P


def _rows(P, cost_s, kappa=KAPPA, A=A):
    return ladder_e.AttemptRows(ids=[f"r{i}" for i in range(len(P))], P=P, cost_s=np.asarray(cost_s, float), care=kappa,
                                abstain_cost=A)


def _preds(P_r):
    return [Prediction(p=dict(zip(OUTCOMES, map(float, p)))) for p in P_r]


def test_one_attempt_costs_are_the_menus_particle_costs():
    rng = np.random.default_rng(0)
    P = _random_P(rng, 3, 50)
    rows = _rows(P, [8.0, 14.0, 10.0])
    C = ladder_e.node_costs(rows, np.full((1, 50), 1 / 50), 1, "plugin")[0]
    for r in range(3):
        assert np.allclose(C[r], particle_costs(_preds(P[r]), [8.0, 14.0, 10.0][r], KAPPA, A), atol=1e-12)
    assert np.all(C[-1] == A)              # the decline row


def test_two_attempts_by_hand():
    """One row, two particles: particle 0 places or tips (0.5 each), particle 1 places (0.2) or drops high (0.8)."""
    P = np.zeros((1, 2, len(OUTCOMES)))
    i = OUTCOMES.index
    P[0, 0, i("placed_stable")], P[0, 0, i("tipped")] = 0.5, 0.5
    P[0, 1, i("placed_stable")], P[0, 1, i("dropped_high")] = 0.2, 0.8
    rows = _rows(P, [8.0])
    t = TIME_COST_PER_S * 8.0
    C1 = np.array([t + 0.5 * (COST["tipped"] + A), t + 0.8 * (5 * KAPPA + A)])
    w = np.array([0.5, 0.5])
    assert np.allclose(ladder_e.node_costs(rows, w[None], 1, "plugin")[0, 0], C1)
    # after a tip only particle 0 keeps weight: the row costs C1[0] = 1.2 + ... below A = 3, so it runs again
    assert C1[0] < A
    C2 = np.array([t + 0.5 * (COST["tipped"] + C1[0]), C1[1]])
    assert np.allclose(ladder_e.node_costs(rows, w[None], 2, "plugin")[0, 0], C2)


def test_the_plugin_value_does_not_rise_with_the_horizon():
    rng = np.random.default_rng(1)
    for _ in range(5):
        P = _random_P(rng, 4, 80)
        rows = _rows(P, rng.uniform(6, 14, 4), kappa=float(rng.choice([1.0, 30.0, 100.0])), A=float(rng.choice([3.0, 10.0])))
        w = rng.dirichlet(np.ones(80))
        V = [float(np.min(ladder_e.node_costs(rows, w[None], h, "plugin")[0] @ w)) for h in (1, 2, 3)]
        assert V[0] >= V[1] - 1e-12 and V[1] >= V[2] - 1e-12


def test_the_batched_choice_is_the_menus_held_out_choice():
    rng = np.random.default_rng(2)
    C = rng.uniform(0, 5, size=(4, 60))
    U = rng.dirichlet(np.ones(60), size=3)
    U[2, 1:] = 0.0                                   # one belief holds all its weight on one particle: row b
    groups = rng.integers(0, 20, size=60)
    for g in (None, groups):
        K = ladder_e.choose(np.broadcast_to(C, (3,) + C.shape), U, "heldout", ladder_e.Groups.of(g))
        for b in range(3):
            best = int(np.argmin(C @ U[b]))
            rows = heldout_rows(C, U[b], best, g)
            assert np.allclose(K[b], C[rows, np.arange(60)])


def test_the_plugin_choice_is_the_cheapest_row_for_every_particle():
    rng = np.random.default_rng(3)
    C = rng.uniform(0, 5, size=(2, 4, 30))
    U = rng.dirichlet(np.ones(30), size=2)
    K = ladder_e.choose(C, U, "plugin", ladder_e.Groups.of(None))
    for b in range(2):
        assert np.allclose(K[b], C[b, int(np.argmin(C[b] @ U[b]))])


def _belief(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return ParticleBelief([{"mass": 0.3, "mu": float(m), "com_obj": [0.0, 0.0, 0.02]} for m in rng.uniform(0.1, 0.9, n)], np.ones(n))


@pytest.mark.parametrize("estimator", ["plugin", "heldout"])
def test_the_probe_value_with_a_fixed_continuation_is_the_menus(estimator):
    b = _belief()
    mu = b.arrays()["mu"]
    C = np.vstack([np.where(mu > 0.3, 1.0, 9.0), np.full(len(mu), 4.0)])   # a row good only above mu 0.3, a flat row
    X = np.zeros((len(mu), 3))
    for pm in (PerfectPush(cut=0.3, p_tip=0.1, p_thrown=0.05), FlatWeigh()):
        ref = probe_value(b, list(C), pm, X, 20.0, KAPPA, np.random.default_rng(7), abstain_cost=A, estimator=estimator)
        got = ladder_e.probe_after(b, lambda W: np.broadcast_to(np.vstack([C, np.full(len(mu), A)]), (len(W),) + (3, len(mu))),
                                   pm, X, 20.0, KAPPA, A, np.random.default_rng(7), estimator=estimator)
        for k in ("evoi", "evoi_se", "expected_cost_after"):
            assert got[k] == pytest.approx(ref[k], abs=1e-12)


def test_the_probe_value_lets_the_later_choices_follow_the_result():
    """Two attempts after a perfect push: the plan after the result is cheaper than the plan made without it."""
    b = _belief()
    mu = b.arrays()["mu"]
    N = len(mu)
    P = np.zeros((2, N, len(OUTCOMES)))
    i = OUTCOMES.index
    hi = mu > 0.3
    P[0, :, i("placed_stable")] = np.where(hi, 0.9, 0.3)              # row 0: good above mu 0.3, tips below
    P[0, :, i("tipped")] = 1 - P[0, :, i("placed_stable")]
    P[1, :, i("placed_stable")] = np.where(hi, 0.4, 0.8)              # row 1: the other way round
    P[1, :, i("not_lifted")] = 1 - P[1, :, i("placed_stable")]
    rows = _rows(P, [8.0, 8.0], kappa=KAPPA, A=10.0)
    X = np.zeros((N, 3))
    cont = lambda W: ladder_e.node_costs(rows, W, 2, "plugin")     # noqa: E731
    v = ladder_e.probe_after(b, cont, PerfectPush(cut=0.3), X, 20.0, KAPPA, 10.0, np.random.default_rng(0), estimator="plugin")
    fixed = cont(b.weights[None])[0]
    ref = probe_value(b, list(fixed[:-1]), PerfectPush(cut=0.3), X, 20.0, KAPPA, np.random.default_rng(0), abstain_cost=10.0)
    assert v["evoi"] > 0
    assert v["expected_cost_after"] <= ref["expected_cost_after"] + 1e-12


# ---------------------------------------------------------------- E on a menu

from types import SimpleNamespace  # noqa: E402

from rpent.skills.pick_place.menu import build_menu  # noqa: E402

from tests.unit_tests.skills.test_menu import P as Proposal, _geometry, plan_fn  # noqa: E402


class MuModel:
    """slow_lift (v_lift 0.05) places with P 0.9 above mu 0.4 and tips otherwise; the other rows place with P 0.6, drop
    high with P 0.02 and are not lifted otherwise."""

    kind = "classifier"

    def predict_many(self, feats):
        out = []
        for f in feats:
            p = {k: 0.0 for k in OUTCOMES}
            if f["v_lift"] < 0.1:
                ok = 0.9 if f["mu"] > 0.4 else 0.3
                p.update(placed_stable=ok, tipped=1.0 - ok)
            else:
                p.update(placed_stable=0.6, dropped_high=0.02, not_lifted=0.38)
            out.append(Prediction(p=p))
        return out


def _menu_entry(estimator="heldout", A=A, kappa=KAPPA):
    b = _belief(300, seed=4)
    probes = {"push": PerfectPush(cut=0.4, p_tip=0.05), "weigh": FlatWeigh()}
    options, rec = build_menu(MuModel(), b, _geometry(), [Proposal(0.3), Proposal(0.5), Proposal(0.7)], 20.0, kappa,
                              "table_centre", plan_fn, np.random.default_rng(0), probe_models=probes, abstain_cost=A,
                              evoi=estimator)
    prims = SimpleNamespace(model=MuModel(), probe_models=probes, evoi=estimator)
    entry = {"assessment": None, "menu": {o.id: o for o in options}, "belief": b}
    ids = [o.id for o in options if o.kind != "decline" and o.id != "inboard_place"] + ["decline"]
    return entry, prims, ids


def test_the_rows_one_attempt_costs_are_the_menus_own():
    entry, prims, ids = _menu_entry()
    menu = entry["menu"]
    attempt = [menu[i] for i in ids if i != "decline" and menu[i].kind != "probe"]
    rows = ladder_e.AttemptRows.from_options(attempt, entry["belief"], prims.model, KAPPA, A)
    assert rows.ids == [o.id for o in attempt]
    assert np.array_equal(rows.C1, np.vstack([o._costs for o in attempt]))


def test_the_rows_refuse_a_menu_priced_at_another_setting():
    entry, prims, ids = _menu_entry()
    attempt = [entry["menu"][i] for i in ids if i != "decline" and entry["menu"][i].kind != "probe"]
    with pytest.raises(ValueError, match="setting"):
        ladder_e.AttemptRows.from_options(attempt, entry["belief"], prims.model, KAPPA, 10.0)


def test_e_at_one_attempt_scores_the_menus_own_costs():
    entry, prims, ids = _menu_entry()
    t = ladder_e.decide(entry, prims, ids, (1,), KAPPA, A, attempts_done=0, seed=(1,))
    menu = entry["menu"]
    assert t["tables"][1] == {i: (A if i == "decline" else menu[i].expected_cost) for i in ids}
    assert t["h"] == {1: 1}


def test_e_looks_ahead_over_the_attempts_left_only():
    entry, prims, ids = _menu_entry()
    ids = [i for i in ids if not i.startswith("probe_")]       # no probe after an attempt
    t = ladder_e.decide(entry, prims, ids, (1, 2, 3), KAPPA, A, attempts_done=2, seed=(1,))
    assert t["h"] == {1: 1, 2: 1, 3: 1}
    assert t["tables"][3] == t["tables"][2] == t["tables"][1]


def test_e_at_two_attempts_prices_a_retry_after_a_recoverable_failure():
    entry, prims, ids = _menu_entry(estimator="plugin", A=10.0)
    t = ladder_e.decide(entry, prims, ids, (1, 2), KAPPA, 10.0, attempts_done=0, seed=(1,))
    one, two = t["tables"][1], t["tables"][2]
    for i in ids:
        if i != "decline" and not i.startswith("probe_"):
            assert two[i] <= one[i] + 1e-12          # a retry is never worse than ending at A (plug-in choices)
    assert two["slow_lift"] < one["slow_lift"] - 1e-6    # it tips (recoverable) below mu 0.4: a retry has value
    assert two["decline"] == one["decline"] == 10.0


def test_e_scores_a_probe_with_the_same_draws_at_every_horizon():
    entry, prims, ids = _menu_entry()
    a = ladder_e.decide(entry, prims, ids, (2, 3), KAPPA, A, attempts_done=0, seed=(5,))
    b = ladder_e.decide(entry, prims, ids, (3,), KAPPA, A, attempts_done=0, seed=(5,))
    assert a["tables"][3] == b["tables"][3]
    assert set(a["probes"][2]) == {"probe_push", "probe_weigh"}


def test_the_planner_runs_the_lowest_score_and_counts_its_attempts():
    entry, prims, ids = _menu_entry(estimator="plugin", A=10.0)
    tools = SimpleNamespace(store={"pick_place": {("object", "table_centre", KAPPA, 0): entry}}, prims=prims)
    schema = __import__("robots.mujoco_lab.e1", fromlist=["tool_schema"]).tool_schema(list(reversed(ids)))
    e = ladder_e.EPlanner(tools, 2, KAPPA, 10.0, seed=(3,))
    r = e([], schema)
    table = e.log[0]["tables"][2]
    lo = min(table.values())
    assert r["tool"] == "choose" and table[r["args"]["option_id"]] == lo
    first = next(i for i in entry["menu"] if i in table and table[i] == lo)      # ties: the menu's order
    assert r["args"]["option_id"] == first
    assert e.attempts == (0 if first.startswith("probe_") or first == "decline" else 1)


# ---------------------------------------------------------------- the run

def test_a_pool_that_recycles_after_one_job_runs_each_job_in_a_new_process():
    # stdlib callables only: a spawned worker cannot import this module under --import-mode=importlib
    with ladder_e._pool(2, max_tasks=1) as pool:
        pids = list(pool.imap_unordered(operator.call, [os.getpid] * 4))
    assert len(set(pids)) == 4


# ---------------------------------------------------------------- the summary

def _row(obj, H, tables, choices, oracle, default, setting="k100_A3_abs", guard=False, artifact=False, idx=0, later=()):
    first = {"step": 0, "attempts": 0, "h": {str(h): h for h in tables}, "tables": {str(h): dict(t) for h, t in tables.items()},
             "choices": {str(h): c for h, c in choices.items()}, "seconds": {str(h): 0.5 * h for h in tables}, "probes": {}}
    log = [first] + [dict(first, step=k + 1, choices={str(h): c for h, c in x.items()}) for k, x in enumerate(later)]
    return {"object": obj, "idx": idx, "seed": 13, "setting": setting, "H": H, "reset_artifact": artifact, "ended": "placed",
            "default": {"assessed": True, "choice": default}, "first": {"oracle_choice": oracle, "guard_flag": guard},
            "e_log": log, "e_seconds": 0.5 * H, "seconds": 10.0 + H}


def _case(obj, T, oracle, default, idx=0, **kw):
    """E's runs at H = 1, 2 and 3 of one case; T: the first decision's table per horizon."""
    choice = {h: min(T[h], key=T[h].get) for h in T}
    return [_row(obj, H, {h: T[h] for h in T if h <= H}, {h: choice[h] for h in T if h <= H}, oracle, default, idx=idx, **kw)
            for H in (1, 2, 3)]


def test_the_summary_names_each_mismatch_of_e_at_one_attempt():
    T1 = {"commit_std": 2.0, "probe_push": 1.9, "decline": 3.0}
    rows = (_case("a", {1: T1, 2: T1, 3: T1}, oracle="commit_std", default="commit_std")
            + _case("b", {1: T1, 2: T1, 3: T1}, oracle="decline", default="commit_std", guard=True)
            + _case("c", {1: {"commit_std": 2.0, "slow_lift": 2.0, "decline": 3.0}} | {2: T1, 3: T1}, oracle="slow_lift",
                    default="commit_std")
            + _case("d", {1: {"commit_std": 2.0, "slow_lift": 2.4, "decline": 3.0}} | {2: T1, 3: T1}, oracle="slow_lift",
                    default="commit_std")
            + _case("e", {1: {"commit_std": 2.0, "decline": 3.0}} | {2: T1, 3: T1}, oracle="commit_std", default="commit_std"))
    s = ladder_e.summarize(rows, n_boot=200)["settings"]["k100_A3_abs"]
    assert s["h1_vs_oracle"]["match"] == 1
    assert s["h1_vs_oracle"]["mismatch"] == {"probe_margin": 1, "guard": 1, "tie": 1, "other": 1}
    assert {(x["case"], x["reason"]) for x in s["h1_vs_oracle"]["cases"]} == {("a/0", "probe_margin"), ("b/0", "guard"),
                                                                                ("c/0", "tie"), ("d/0", "other")}


def test_the_summary_takes_the_default_rules_regret_on_each_horizons_table():
    rows = []
    for k in range(6):
        T = {1: {"commit_std": 2.0, "slow_lift": 2.5, "decline": 3.0}, 2: {"commit_std": 1.5, "slow_lift": 1.0, "decline": 3.0},
             3: {"commit_std": 1.4, "slow_lift": 0.8, "decline": 3.0}}
        rows += _case(f"o{k % 3}", T, oracle="commit_std", default="decline", idx=k)
    s = ladder_e.summarize(rows, n_boot=200)["settings"]["k100_A3_abs"]
    assert s["regret"]["1"]["mean"] == pytest.approx(1.0)
    assert s["regret"]["2"]["mean"] == pytest.approx(2.0)
    assert s["regret"]["3"]["mean"] == pytest.approx(2.2)
    assert s["regret"]["3"]["n"] == 6 and s["regret"]["3"]["objects"] == 3
    assert s["oracle_regret"]["3"]["mean"] == pytest.approx(0.6)
    assert s["differ_first"]["2"] == {"n": 6, "differ": 6, "pairs": {"commit_std -> slow_lift": 6}}
    assert s["below_e"] == [] and s["not_monotone"] == []


def test_the_summary_leaves_out_an_artifact_and_lists_a_policy_below_e():
    T = {1: {"commit_std": 2.0, "decline": 3.0}, 2: {"commit_std": 1.0, "decline": 3.0}, 3: {"commit_std": 1.2, "decline": 3.0}}
    rows = _case("a", T, oracle="commit_std", default="commit_std") + _case("b", T, oracle="decline", default="decline",
                                                                            artifact=True)
    rows[2]["e_log"][0]["choices"]["3"] = "decline"          # a planted E(3) choice above another policy's
    s = ladder_e.summarize(rows, n_boot=200)["settings"]["k100_A3_abs"]
    assert s["artifact"] == 1 and s["regret"]["1"]["n"] == 1
    assert {(x["case"], x["H"], x["policy"]) for x in s["below_e"]} >= {("a/0", 3, "default"), ("a/0", 3, "oracle_rule")}
    assert [(x["case"], x["H"]) for x in s["not_monotone"]] == [("a/0", 3)]


def test_the_summary_counts_the_horizons_differences_at_every_decision():
    T = {1: {"commit_std": 2.0, "decline": 3.0}, 2: {"commit_std": 1.0, "decline": 3.0}, 3: {"commit_std": 1.0, "decline": 3.0}}
    rows = _case("a", T, oracle="commit_std", default="commit_std")
    full = rows[2]
    full["e_log"].append(dict(full["e_log"][0], step=1, choices={"1": "commit_std", "2": "slow_lift", "3": "slow_lift"}))
    s = ladder_e.summarize(rows, n_boot=200)["settings"]["k100_A3_abs"]
    assert s["differ_all"]["2"] == {"n": 2, "differ": 1} and s["differ_all"]["3"] == {"n": 2, "differ": 1}


def test_the_summary_checks_that_every_run_saw_the_same_first_tables():
    T = {1: {"commit_std": 2.0, "decline": 3.0}, 2: {"commit_std": 1.0, "decline": 3.0}, 3: {"commit_std": 1.0, "decline": 3.0}}
    rows = _case("a", T, oracle="commit_std", default="commit_std")
    rows[1]["e_log"][0]["tables"]["2"]["commit_std"] = 1.1
    s = ladder_e.summarize(rows, n_boot=200)["settings"]["k100_A3_abs"]
    assert s["consistency"] == {"compared": 1, "differ": ["a/0"]}


def test_the_summary_leaves_out_a_case_whose_reset_failed_in_any_run():
    T = {1: {"commit_std": 2.0, "decline": 3.0}, 2: {"commit_std": 1.0, "decline": 3.0}, 3: {"commit_std": 1.0, "decline": 3.0}}
    rows = _case("a", T, oracle="commit_std", default="commit_std") + _case("b", T, oracle="commit_std", default="decline")
    rows[3]["reset_artifact"] = True          # case b: only its H = 1 run's reset did not reach rest
    s = ladder_e.summarize(rows, n_boot=200)["settings"]["k100_A3_abs"]
    assert s["artifact"] == 1 and s["regret"]["1"]["n"] == 1
