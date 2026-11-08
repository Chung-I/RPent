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

"""The oracle arm (curve_oracle; daily-logs designs/2026-11-08-oracle-arm-design.md): the belief at the truth, the hooks
it uses in curve, R0 at the locked knob, the smoke report and the pre-registered readings."""

import copy
import json
import os
import subprocess
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from robots.mujoco_lab import curve as cv
from robots.mujoco_lab import curve_oracle as co
from robots.mujoco_lab import evpi_props as ev
from robots.mujoco_lab.fragility import Thresholds, map_latent
from rpent.skills.pick_place.belief import ParticleBelief
from rpent.skills.pick_place.costs import COST, TIME_COST_PER_S
from rpent.skills.pick_place.protocol import IRREVERSIBLE, OUTCOMES, RECOVERABLE, Prediction

K = cv.dkey("R0", cv.A_MAIN, 2.0)
K4 = cv.dkey("R4", cv.A_MAIN, None)
LAW = {"f_damage_n": {"median": 50.0, "log_sd": 0.5}, "k_n_per_mm": {"median": 10.0, "log_sd": 0.4}, "rho": 0.6,
       "v_bruise_ms": {"median": 1.0, "log_sd": 0.3}}
TH = Thresholds(laws={"soft": LAW}, slosh_zeta=0.1)
CRUSH = {"class": "crushable", "law": "soft", "fillable": False, "cavity": None}
POSE = {"object_id": "toy", "pivot": {"centre": [0.0, 0.0, 0.0], "axis": [1.0, 0.0, 0.0], "lo": -0.1, "L": 0.2},
        "com_obj_nominal": [0.0, 0.0, 0.01], "nominal_mass": 0.5}
THETA = {"mass_scale": 1.3, "mu": 0.42, "com_s": 0.7, "v_lift": 0.1}         # cal.case_draw's keys
HIDDEN = {"fill": 0.0, "frag": map_latent((0.3, -1.2, 0.8), CRUSH, TH), "spill_angle_deg": None}
P = dict({k: 0.0 for k in OUTCOMES}, placed_stable=0.62, crushed=0.05, tipped=0.2, missed=0.13)


def _prior_b0(n=30, seed=0):
    from robots.mujoco_lab.prior import make_prior

    return make_prior({"toy": CRUSH}, TH, generator="v11")(POSE, n, np.random.default_rng(seed))


def _row(rid, placed, irr=0.0, rec_label="tipped", rec=None, cost_s=8.0, action=None):
    rec = 1.0 - placed - irr if rec is None else rec
    p = {k: 0.0 for k in OUTCOMES}
    p.update({"placed_stable": placed, "crushed": irr, rec_label: rec})
    return {"id": rid, "action": action or f"{rid}|{{}}", "p": p, "p_placed": placed, "p_irr": irr, "rec": rec * COST[rec_label],
            "cost_s": cost_s, "q_irr": irr, "cvar": {}}


def _option(rid, **act):
    a = dict({"force": 20.0, "v_lift": 0.1, "v_transport": 0.5, "a_transport": 2.0, "destination": "bin", "release_height": None,
              "s": 0.5}, **act)
    return SimpleNamespace(id=rid, kind="commit" if rid == "commit_std" else "lower_risk", action=a, cost_s=8.0,
                           pred=Prediction(p=dict(P)), _base={}, _pose={}, _planned=[])


class _Env:
    def __init__(self):
        self.reset_info, self.resets = {"at_rest": True}, []

    def reset(self, o, theta, **kw):
        self.resets.append((o, kw["seed"]))

    def pose_info(self):
        return copy.deepcopy(POSE)

    def truth(self):
        return {"mass": 0.65, "com_obj": [0.0, 0.0, 0.01], "mu": 0.42, "fill": 0.0, "liquid_mass": 0.0}

    def locate(self, name):
        return {"object_id": "toy"}


def _prior(pose, n, g):
    g.random(3)                                     # the draw moves the stream, as the prior's does
    return ParticleBelief([{"mass": 0.1 * (i + 1)} for i in range(n)], np.ones(n), flags=["class crushable"])


# ---------------------------------------------------------------- the belief at the truth

def test_the_oracle_belief_is_one_particle_at_the_truth_with_weight_1():
    b0, truth = _prior_b0(), ev.truth_particle(POSE, THETA, 1, HIDDEN, CRUSH)
    b = co.oracle_belief(b0, truth)
    assert len(b) == 1 and b.particles[0] == truth and b.particles[0] is not truth
    np.testing.assert_array_equal(b.weights, [1.0])
    assert b.effective_size() == pytest.approx(1.0) and b.history == [] and b.flags == b0.flags and not b.degenerate
    r, _ = ev.reveal(b0, "all", truth, CRUSH, TH, None, None, 0.2)          # experiment 1's EVPI_all belief
    assert b.particles == r.particles and list(b.weights) == list(r.weights)


def test_the_truth_belief_holds_the_cases_draw_in_the_priors_particle_format():
    b0 = _prior_b0()
    b = co.truth_belief(b0, POSE, THETA, 1, HIDDEN, CRUSH)
    assert len(b) == 1 and set(b.particles[0]) == set(b0.particles[0])
    assert b.particles[0] == ev.truth_particle(POSE, THETA, 1, HIDDEN, CRUSH)
    assert b.particles[0]["frag_z"] == [0.3, -1.2, 0.8] and b.particles[0]["mu"] == 0.42


def test_row_record_over_the_oracle_belief_returns_the_particles_own_probabilities():
    per = [Prediction(p=dict(P))]
    o = SimpleNamespace(id="commit_std", kind="commit", action={"force": 20.0, "destination": "bin"}, cost_s=8.0, pred=per[0])
    r = cv.row_record(o, per, co.oracle_belief(_prior_b0(), {"mass": 0.4}).weights)
    irr, rec = sum(P[k] for k in IRREVERSIBLE), sum(P[k] * COST[k] for k in RECOVERABLE)
    assert r["p"] == {k: P[k] for k in OUTCOMES} and r["p_placed"] == 0.62 and r["mean_gap"] == 0.0
    assert r["p_irr"] == pytest.approx(irr, abs=1e-15) and r["q_irr"] == pytest.approx(irr, abs=1e-15)
    assert r["rec"] == pytest.approx(rec, rel=1e-12)
    for A, x in cv.CVAR_KEYS:
        assert r["cvar"][f"{A:g}|{x:g}"] == pytest.approx(0.62 - irr / x - (rec + TIME_COST_PER_S * 8.0) / A, rel=1e-12)


# ---------------------------------------------------------------- the hooks in curve

def _patch_first_menu(monkeypatch, seen):
    from robots.mujoco_lab import adapter, com_gap
    from robots.mujoco_lab import calibrate as cal
    from rpent.skills.pick_place import menu as menu_mod

    monkeypatch.setattr(cal, "case_draw", lambda *a, **k: (dict(THETA), 1, "bin", copy.deepcopy(HIDDEN)))
    monkeypatch.setattr(adapter, "object_props", lambda o, *a: {"mass": 0.5})
    monkeypatch.setattr(com_gap, "fingerprint", lambda env: "fp")

    def first_menu(prims, b, destination, care, A, state):
        seen.append((b, copy.deepcopy(state)))
        return {"payload": {"assessed": True, "best_option": "commit_std"}, "menu": {"commit_std": _option("commit_std")}}

    monkeypatch.setattr(ev, "first_menu", first_menu)
    monkeypatch.setattr(menu_mod, "estimate", lambda model, b, *a, **k: (None, [Prediction(p=dict(P))] * len(b)))


def test_stage1_case_hands_the_priors_draw_and_the_cases_draw_to_belief_fn(monkeypatch):
    seen, got = [], {}
    _patch_first_menu(monkeypatch, seen)
    prims = SimpleNamespace(env=_Env(), prior=_prior, model=None, n_particles=4)
    shared = SimpleNamespace(clear_case=lambda: None, clear=lambda: None)
    s = {"seed": 11, "fold": 0, "side": "validation"}
    plain = cv.stage1_case(prims, "toy", 3, s, None, shared)

    def fn(b0, pose, theta, com_end, hidden, entry):
        got.update(n=len(b0), pose=pose, theta=theta, com_end=com_end, hidden=hidden, entry=entry)
        return co.oracle_belief(b0, {"mass": 0.65})

    oracle = cv.stage1_case(prims, "toy", 3, s, None, shared, belief_fn=fn)
    (b_plain, st_plain), (b_oracle, st_oracle) = seen
    assert len(b_plain) == 4 and len(b_oracle) == 1 and b_oracle.particles == [{"mass": 0.65}]
    assert st_plain == st_oracle                         # the menu stream after the prior's draw, as at the prior
    assert got["n"] == 4 and got["pose"] == POSE and got["theta"] == THETA and got["com_end"] == 1 and got["hidden"] == HIDDEN
    assert got["entry"]["class"] == "none"               # world None: stage1_case's own entry
    assert plain["ess"] == pytest.approx(4.0) and oracle["ess"] == pytest.approx(1.0) and oracle["status"] == "ok"
    assert oracle["rows"][0]["p_placed"] == P["placed_stable"] and oracle["rows"][0]["mean_gap"] == 0.0


def _oracle_lab(prims=None):
    lab = object.__new__(co.OracleLabCase)
    lab.prims = prims or SimpleNamespace(env=_Env(), prior=_prior, model=None, n_particles=4)
    lab.o, lab.idx, lab.seed, lab.fixture, lab.destination = "toy", 3, 11, None, "bin"
    lab.theta, lab.com_end, lab.hidden, lab.entry, lab.point, lab.truth = dict(THETA), 1, copy.deepcopy(HIDDEN), CRUSH, None, None
    return lab


def test_the_lab_case_keeps_the_prior_and_the_oracle_lab_case_takes_the_truth():
    b0 = _prior_b0()
    assert object.__new__(cv.LabCase).first_belief(b0, POSE) is b0
    lab = _oracle_lab()
    b = lab.first_belief(b0, POSE)
    assert b.particles == [ev.truth_particle(POSE, THETA, 1, HIDDEN, CRUSH)] and lab.point is b
    assert lab.truth == b.particles[0] and lab.truth is not b.particles[0]


def test_fresh_builds_the_first_menu_on_first_belief(monkeypatch):
    from robots.mujoco_lab import com_gap, evpi_sim
    from rpent.skills.pick_place import tools

    monkeypatch.setattr(com_gap, "fingerprint", lambda env: "fp")
    stores = []
    monkeypatch.setattr(evpi_sim, "_store", lambda env, b: stores.append({"beliefs": {"toy": {"belief": b}}}) or stores[-1])

    def assess(object, destination, care, primitives, store):
        b = store["beliefs"]["toy"]["belief"]
        store["pick_place"] = {("toy", destination, care, 0): {"menu": {"commit_std": _option("commit_std")}, "belief": b}}

    monkeypatch.setattr(tools, "assess_pick", assess)
    lab = _oracle_lab()
    rec1 = {"fingerprint": "fp", "rows": [cv.mean_row(_option("commit_std"))]}
    store, stop, at_rest = lab.fresh(11003, rec1)
    assert stop == "" and at_rest is True and store is stores[0]
    assert store["beliefs"]["toy"]["belief"] is lab.point and len(lab.point) == 1
    assert lab.point.particles == [ev.truth_particle(POSE, THETA, 1, HIDDEN, CRUSH)]
    _, stop, _ = lab.fresh(11003, dict(rec1, fingerprint="other"))
    assert "fingerprint" in stop


def test_the_oracle_second_menu_sets_the_belief_back_to_the_truth_and_reassesses(monkeypatch):
    from rpent.skills.pick_place import menu as menu_mod
    from rpent.skills.pick_place import tools

    lab = _oracle_lab()
    lab.first_belief(_prior_b0(), POSE)
    refreshed = _prior_b0(n=1, seed=7)                    # what the attempt's update left: a refresh from the prior
    store = {"beliefs": {"toy": {"episode": 0, "belief": refreshed, "attempted": True}}, "pick_place": {"stale": {}}}
    seen = {}

    def reassess(object, destination, care, primitives, store):
        b = store["beliefs"]["toy"]["belief"]
        seen.update(belief=b, care=care, destination=destination)
        store["pick_place"] = {("toy", destination, care, 0): {"menu": {"commit_std": _option("commit_std")}, "belief": b}}
        return {"assessed": True}

    monkeypatch.setattr(tools, "_reassess", reassess)
    monkeypatch.setattr(menu_mod, "estimate", lambda model, b, *a, **k: (None, [Prediction(p=dict(P))] * len(b)))
    m = lab.second_menu(store, {"assessment": {"assessed": True}, "outcome": "tipped"})
    assert seen["care"] == cv.STAGE1_CARE and seen["destination"] == "bin"
    assert seen["belief"].particles == [ev.truth_particle(POSE, THETA, 1, HIDDEN, CRUSH)] and len(seen["belief"]) == 1
    assert m["status"] == "ok" and m["ess"] == pytest.approx(1.0) and m["rows"][0]["p_placed"] == P["placed_stable"]
    monkeypatch.setattr(tools, "_reassess", lambda *a, **k: {"error": "reassessment failed: boom"})
    assert lab.second_menu(store, {"assessment": {"assessed": True}})["status"] == "error"


class _Lab:
    """simulate_case's lab: first attempts give first[row], later ones second[row]; the second menu has rows2."""

    def __init__(self, first, second, rows2):
        self.first, self.second, self.rows2, self.attempts = first, second, rows2, []

    def fresh(self, reset_seed, rec1):
        return {"n": 0}, "", True

    def attempt(self, store, row):
        store["n"] += 1
        self.attempts.append(row)
        return {"outcome": self.first[row] if store["n"] == 1 else self.second[row], "artifact": False}

    def second_menu(self, store, d):
        return {"status": "ok", "rows": [dict(r) for r in self.rows2], "ess": 1.0}


def test_simulate_case_takes_the_second_decisions_it_is_given():
    rows2 = [_row("commit_std", 0.55), _row("slow_lift", 0.75, irr=0.002, cost_s=14.0), _row("firm_grip", 0.93, irr=0.06)]
    full = cv.second_decisions([dict(r) for r in rows2], ("R0",))
    assert len(set(full.values()) - {cv.GIVE_UP_ROW}) > 1              # the default would run more than one second row
    lab = _Lab({"commit_std": "tipped"}, {r["id"]: "placed_stable" for r in rows2}, rows2)
    res = cv.simulate_case(lab, {"rows": []}, {K: "commit_std"}, ("R0",), [11003], second=co.oracle_second(2.0))
    s = res["sims"][0]
    want = cv.decide("R0", rows2, 2.0, cv.A_MAIN, cv.no_continuation)
    assert res["stop"] is None and s["menu2"]["decisions"] == {K: want} and list(s["second"]) == [want]
    assert lab.attempts == ["commit_std", want]


# ---------------------------------------------------------------- the decisions and the case

def test_the_oracle_decides_r0_at_every_knob_and_simulates_the_locked_one():
    rows = [_row("commit_std", 0.9), _row("firm_grip", 0.95, irr=0.03)]
    cur = co.curve_decisions(rows)
    assert cur == {cv.dkey("R0", cv.A_MAIN, r): cv.decide("R0", rows, r, cv.A_MAIN) for r in cv.R_GRID}
    assert co.oracle_second(2.0)(rows, ("R0",)) == {K: cv.decide("R0", rows, 2.0, cv.A_MAIN, cv.no_continuation)}
    assert co.oracle_second(2.0)([], ("R0",)) == {K: cv.GIVE_UP_ROW}


def test_the_locked_knob_is_r0s_from_the_lock_file_and_must_be_the_designs(tmp_path):
    p = tmp_path / "locked.json"
    p.write_text(json.dumps({"lock": {"R0": {"knob": 2.0, "key": "R0|10|2"}}}))
    assert co.locked_knob(str(p)) == 2.0
    p.write_text(json.dumps({"lock": {"R0": {"knob": 5.0, "key": "R0|10|5"}}}))
    with pytest.raises(SystemExit, match="design"):
        co.locked_knob(str(p))


class _NoMemo:
    def __init__(self, prims):
        pass

    def __enter__(self):
        return SimpleNamespace(clear=lambda: None, clear_case=lambda: None, hits=Counter())

    def __exit__(self, *exc):
        return False


def _s1rec(o="toy", idx=3, side="test", rows=None, **kw):
    rec = {"object": o, "idx": idx, "fold": 0, "side": side, "class": "crushable", "law": "soft", "fillable": False, "thin": False,
           "destination": "bin", "fingerprint": "fp", "ess": 900.0, "guard_flag": False, "status": "ok" if rows is None or rows else "no_rows",
           "rows": [_row("commit_std", 0.5)] if rows is None else rows, "artifact": False}
    return dict(rec, **kw)


def test_oracle_case_simulates_r0_at_the_locked_knob_on_the_truths_first_menu(monkeypatch):
    rows = [_row("commit_std", 0.9), _row("firm_grip", 0.95, irr=0.03)]
    calls = {}

    def stage1(prims, o, idx, s, world, shared, belief_fn=None):
        calls["s"], calls["belief"] = s, belief_fn(_prior_b0(), POSE, THETA, 1, HIDDEN, CRUSH)
        return dict(_s1rec(o, idx, rows=rows, side=s["side"], ess=1.0), fold=s["fold"])

    def sim(lab, rec1, dec, rules, seeds, second=None):
        calls["sim"] = (lab, dec, rules, seeds, second(rows, rules))
        return {"sims": [dict(_sim(dec[K], "placed_stable"), reset_seed=seeds[0])], "stop": None}

    monkeypatch.setattr(co, "stage1_case", stage1)
    monkeypatch.setattr(co, "SharedMenus", _NoMemo)
    monkeypatch.setattr(co, "OracleLabCase", lambda prims, o, idx, seed, world: ("lab", o, idx, seed))
    monkeypatch.setattr(co, "seeds_of", lambda seed, o, idx: [seed * 1000 + idx, 7, 8])
    monkeypatch.setattr(co, "simulate_case", sim)
    prims = SimpleNamespace(env=_Env())
    r = co.oracle_case(prims, _s1rec(), {"seed": 11}, None, 2.0)
    want = cv.decide("R0", rows, 2.0, cv.A_MAIN)
    assert calls["sim"] == (("lab", "toy", 3, 11), {K: want}, ("R0",), [11003], {K: want})
    assert calls["s"]["fold"] == 0 and calls["s"]["side"] == "test" and len(calls["belief"]) == 1
    assert r["decisions"] == {K: want} and r["first_rows"] == [want] and r["oracle"] is True and r["knob"] == 2.0
    assert r["curve_decisions"] == co.curve_decisions(rows) and r["rows"] == rows and r["ess"] == 1.0
    assert r["dropped"] is False and r["stop"] is None and r["skipped"] is None and r["sims"][0]["first"] == want
    assert r["truth"] == ev.truth_particle(POSE, THETA, 1, HIDDEN, CRUSH) and r["sim_truth"]["mass"] == 0.65
    calls.clear()
    sk = co.oracle_case(prims, _s1rec(artifact=True), {"seed": 11}, None, 2.0)          # Stage 2's skip rule
    assert sk["skipped"] == "stage1 artifact" and sk["dropped"] is True and sk["sims"] == [] and calls == {}
    st = co.oracle_case(prims, _s1rec(fingerprint="other"), {"seed": 11}, None, 2.0)    # the oracle's reset is not Stage 1's
    assert "fingerprint" in st["stop"] and "sim" not in calls
    monkeypatch.setattr(co, "stage1_case", lambda *a, **k: dict(_s1rec(rows=[]), status="error", error="ValueError: x"))
    er = co.oracle_case(prims, _s1rec(), {"seed": 11}, None, 2.0)
    assert er["skipped"] == "oracle first menu error" and er["dropped"] is True and er["stop"] is None


# ---------------------------------------------------------------- the readings

def _sim(first, label, menu2=None, second=None, seed_k=0, **kw):
    return dict({"seed_k": seed_k, "first": first, "label": label, "artifact": False, "error": None, "menu2": menu2,
                 "second": second or {}}, **kw)


def _ep(rows=(), labels=(), give_up=False):
    lab = list(labels)
    return {"attempts": len(rows), "success": bool(lab) and lab[-1] == "placed_stable", "irr": any(x in IRREVERSIBLE for x in lab),
            "give_up": give_up, "labels": lab, "rows": list(rows)}


def _s2(o, idx, dec, sims, side="test", **kw):
    rec = {"object": o, "idx": idx, "side": side, "class": "none", "law": None, "fillable": False, "thin": False, "decisions": dec,
           "sims": sims, "dropped": False, "skipped": None, "error": None, "stop": None, "n_seeds": 1, "row_stats": {},
           "first_rows": sorted({d for d in dec.values() if d != cv.GIVE_UP_ROW})}
    return dict(rec, **kw)


def _boot_recs():
    recs = []
    for i in range(6):
        o = f"o{i}"
        recs.append(_s2(o, 0, {K: "commit_std", K4: "firm_grip"}, [_sim("commit_std", "placed_stable"), _sim("firm_grip", "crushed")]))
        m2 = {"status": "no_rows", "rows": [], "decisions": {K: cv.GIVE_UP_ROW, K4: cv.GIVE_UP_ROW}}
        recs.append(_s2(o, 1, {K: cv.GIVE_UP_ROW if i % 2 else "commit_std", K4: "commit_std"},
                        [_sim("commit_std", "tipped", menu2=m2) if i < 3 else _sim("commit_std", "placed_stable")]))
    return recs


def test_paired_outcomes_resample_objects_exactly_as_the_stage2_bootstrap():
    recs = _boot_recs()
    b = cv.paired_bootstrap(recs, K, K4, draws=300, seed=5)
    pairs = [(r["object"], cv.episode(r["decisions"], r["sims"], K), cv.episode(r["decisions"], r["sims"], K4)) for r in recs]
    o = co.paired_outcomes(pairs, draws=300, seed=5)
    assert o["success"] == b["success"] and o["irr"] == b["irr"] and o["objects"] == b["objects"] == 6 and o["cases"] == 12
    gd = [float(a["give_up"]) - float(c["give_up"]) for _, a, c in pairs]
    assert o["give_up"]["diff"] == pytest.approx(sum(gd) / len(gd)) and o["give_up"]["lo"] <= o["give_up"]["diff"] <= o["give_up"]["hi"]
    assert co.paired_outcomes(pairs, draws=300, seed=5) == o and o["draws"] == 300 and o["seed"] == 5


@pytest.mark.parametrize("d,lo,hi,want", [
    (0.05, 0.01, 0.09, "the belief is a limit"),                           # (i) g above +2, the interval above 0
    (0.025, 0.005, 0.05, "the belief is a limit"),
    (0.01, 0.002, 0.019, "the belief is not what limits success"),         # (ii) the interval below +2
    (0.01, -0.01, 0.019, "the belief is not what limits success"),         # (ii) though the interval includes 0
    (-0.005, -0.012, 0.0, "the belief is not what limits success"),        # (ii) hi 0 is not below 0
    (-0.03, -0.05, -0.01, "oracle below R0"),                              # (iii) before (ii)
    (0.015, 0.002, 0.029, "undecided at n = 481"),                         # (iv) the design's +1.5 [+0.2, +2.9]
    (0.04, -0.005, 0.085, "undecided at n = 481"),                         # (iv) the design's +4 [-0.5, +8.5]
    (0.01, -0.01, 0.03, "undecided at n = 481"),                           # (iv) straddles +2, g at most +2
    (0.02, 0.001, 0.04, "undecided at n = 481"),                           # (iv) g of exactly +2 is not above it
    (0.03, 0.0, 0.06, "undecided at n = 481"),                             # (iv) lo of exactly 0 is not above it
])
def test_the_reading_rule_is_the_designs_section_3_1(d, lo, hi, want):
    assert co.oracle_reading({"success": {"diff": d, "lo": lo, "hi": hi}, "cases": 481}) == want


def test_an_undecided_reading_names_its_views_case_count():
    assert co.oracle_reading({"success": {"diff": 0.04, "lo": -0.005, "hi": 0.085}, "cases": 12}) == "undecided at n = 12"


def test_the_first_decision_groups_partition_the_cases():
    cases = [
        {"oracle": _ep(["commit_std"], ["placed_stable"]), "r0": _ep(give_up=True), "same": False},
        {"oracle": _ep(["firm_grip"], ["crushed"]), "r0": _ep(["commit_std"], ["placed_stable"]), "same": False},
        {"oracle": _ep(["commit_std"], ["tipped", "placed_stable"]), "r0": _ep(["commit_std"], ["placed_stable"]), "same": True,
         "same_id": True},
        {"oracle": _ep(["regrasp_com"], ["placed_stable"]), "r0": _ep(["commit_std"], ["placed_stable"]), "same": True,
         "same_id": False},
        {"oracle": _ep(give_up=True), "r0": _ep(["commit_std"], ["tipped"]), "same": False},
        {"oracle": _ep(give_up=True), "r0": _ep(give_up=True), "same": False},
        {"oracle": _ep(["commit_std"], ["missed"]), "r0": _ep(give_up=True), "same": False},
    ]
    g = co.first_decision_groups(cases)
    assert list(g) == ["R0 gave up and the oracle grasped", "both grasped, same action", "both grasped, different action",
                       "the oracle gave up and R0 grasped", "both gave up"]
    assert {k: v["n"] for k, v in g.items()} == {"R0 gave up and the oracle grasped": 2, "both grasped, different action": 1,
                                                 "both grasped, same action": 2, "the oracle gave up and R0 grasped": 1,
                                                 "both gave up": 1}
    assert g["both grasped, same action"]["equal_action_different_id"] == 1
    first = g["R0 gave up and the oracle grasped"]
    assert first["oracle_first_labels"] == {"placed_stable": 1, "missed": 1} and first["oracle"]["n_success"] == 1
    diff = g["both grasped, different action"]
    assert diff["oracle"]["n_irr"] == 1 and diff["r0"]["n_success"] == 1 and diff["oracle"]["irr"] == 1.0
    assert sum(v["n"] for v in g.values()) == len(cases)


def _key(rid, **kw):
    a = dict({"a_transport": 2.0, "destination": "bin", "force": 20.0, "release_height": None, "s": 0.5, "v_lift": 0.1,
              "v_transport": 0.5}, **kw)
    return f"{rid}|{json.dumps(a, sort_keys=True)}"


def test_the_row_change_compares_the_two_actions():
    assert co.row_change(_key("commit_std"), _key("commit_std")) == "same action"
    assert co.row_change(_key("regrasp_com"), _key("commit_std")) == "same action"        # regrasp_com at the standard pinch
    assert co.row_change(_key("slow_lift", v_lift=0.05, v_transport=0.2), _key("commit_std")) == "force or speed change"
    assert co.row_change(_key("firm_grip", force=40.0), _key("light_grip", force=10.0)) == "force or speed change"
    assert co.row_change(_key("gentle_carry", a_transport=0.5), _key("commit_std")) == "force or speed change"
    assert co.row_change(_key("regrasp_com", s=0.7), _key("firm_grip", force=40.0)) == "pose change"   # pose and force
    assert co.row_change(_key("regrasp_com", s=0.7, force=40.0), _key("commit_std")) == "pose change"
    assert co.row_change(_key("drop_release", release_height=0.1), _key("commit_std")) == "other change"
    assert co.row_change(None, _key("commit_std")) == "the oracle gives up"


def test_the_views_mark_an_equal_action_pair_with_different_row_ids():
    rows1 = [_row("commit_std", 0.5, action=_key("commit_std")), _row("regrasp_com", 0.5, action=_key("regrasp_com"))]
    s1 = _s1rec("toy", 3, rows=rows1)
    s2 = _s2("toy", 3, {K: "commit_std", K4: "commit_std"}, [_sim("commit_std", "placed_stable")])
    o = dict(_s2("toy", 3, {K: "regrasp_com"}, [_sim("regrasp_com", "placed_stable")]), rows=rows1, oracle=True)
    v = co._views([{"oracle": o, "stage2": s2, "stage1": s1}], K, K4)[0]
    assert (v["change"], v["same"], v["same_id"], v["group"]) == ("same action", True, False, "c")
    o2 = dict(o, decisions={K: "commit_std"}, sims=[_sim("commit_std", "placed_stable")])
    v2 = co._views([{"oracle": o2, "stage2": s2, "stage1": s1}], K, K4)[0]
    assert (v2["same"], v2["same_id"]) == (True, True)
    g = co.first_decision_groups([v, v2, v])
    assert g["both grasped, same action"]["n"] == 3 and g["both grasped, same action"]["equal_action_different_id"] == 2


def test_the_paired_set_drops_a_case_either_arm_drops_and_counts_them():
    j = [{"oracle": {"dropped": o}, "stage2": {"dropped": s}} for s, o in ((False, False), (True, False), (False, True),
                                                                          (True, True), (False, False))]
    paired, n = co.paired_set(j)
    assert paired == [j[0], j[4]]
    assert n == {"dropped": 3, "stage2_only": 1, "oracle_only": 1, "both": 1}


def test_the_menu_limit_groups_come_from_stage2s_r0_and_r4_decisions():
    assert co.menu_group({K: cv.GIVE_UP_ROW, K4: cv.GIVE_UP_ROW}, K, K4) == "a"
    assert co.menu_group({K: cv.GIVE_UP_ROW, K4: "firm_grip"}, K, K4) == "b"
    assert co.menu_group({K: "commit_std", K4: "firm_grip"}, K, K4) == "c"
    with pytest.raises(ValueError):
        co.menu_group({K: "commit_std", K4: cv.GIVE_UP_ROW}, K, K4)       # R4 gives up only without rows


def test_reliability_bins_by_decile_and_scores_brier():
    p, y = [i / 20 for i in range(20)], [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]
    c = co.reliability(p, y)
    assert [b["n"] for b in c["bins"]] == [2] * 10
    assert c["bins"][0]["predicted"] == pytest.approx(0.025) and c["bins"][0]["observed"] == 0.0
    assert c["bins"][9]["predicted"] == pytest.approx(0.925) and c["bins"][9]["observed"] == 1.0
    assert c["brier"] == pytest.approx(float(np.mean((np.array(p) - np.array(y)) ** 2)))
    assert c["n"] == 20 and c["predicted_sum"] == pytest.approx(sum(p)) and c["observed_sum"] == 11
    assert [b["n"] for b in co.reliability([0.1, 0.1, 0.2], [0, 1, 0])["bins"]] == [1, 1, 1]
    assert co.reliability([], [])["n"] == 0


def test_the_attempts_at_the_truth_take_each_rows_own_prediction():
    m2 = {"status": "ok", "rows": [dict(_row("slow_lift", 0.7, irr=0.05))], "decisions": {K: "slow_lift"}}
    rec = _s2("o", 0, {K: "commit_std"}, [_sim("commit_std", "tipped", menu2=m2,
                                                    second={"slow_lift": {"label": "placed_stable", "artifact": False, "error": None}})],
              row_stats={"commit_std": {"p_placed": 0.6, "p_irr": 0.1, "q_irr": 0.1}})
    assert co.attempts_at_truth([rec], K) == [(0.6, 0.1, "tipped", 1), (0.7, 0.05, "placed_stable", 2)]
    assert co.attempts_at_truth([dict(rec, dropped=True)], K) == []


def _joined(o_first, o_label, s2_dec, s2_sims, o_rows, s1_rows, idx=0):
    orec = _s2("o", idx, {K: o_first}, [_sim(o_first, o_label)] if o_first != cv.GIVE_UP_ROW else [], rows=o_rows)
    return {"oracle": orec, "stage2": _s2("o", idx, s2_dec, s2_sims), "stage1": _s1rec("o", idx, rows=s1_rows)}


def test_the_determinism_check_compares_labels_only_where_the_first_actions_coincide():
    std, firm = _key("commit_std"), _key("firm_grip", force=40.0)
    rc_prior, rc_truth = _key("regrasp_com", s=0.3), _key("regrasp_com", s=0.7)
    s1 = [_row("commit_std", 0.5, action=std), _row("firm_grip", 0.5, action=firm), _row("regrasp_com", 0.5, action=rc_prior)]
    orows = [_row("commit_std", 0.5, action=std), _row("firm_grip", 0.5, action=firm), _row("regrasp_com", 0.5, action=rc_truth)]
    cases = [
        _joined("commit_std", "tipped", {K: "commit_std", K4: "firm_grip"}, [_sim("commit_std", "tipped"), _sim("firm_grip", "crushed")], orows, s1, 0),
        _joined("regrasp_com", "placed_stable", {K: "regrasp_com", K4: "regrasp_com"}, [_sim("regrasp_com", "missed")], orows, s1, 1),
        _joined("firm_grip", "placed_stable", {K: "commit_std", K4: "firm_grip"}, [_sim("commit_std", "tipped"), _sim("firm_grip", "crushed")], orows, s1, 2),
        _joined(cv.GIVE_UP_ROW, None, {K: "commit_std", K4: "firm_grip"}, [_sim("commit_std", "tipped"), _sim("firm_grip", "crushed")], orows, s1, 3),
    ]
    d = co.determinism(cases, K, K4)
    assert d["compared"] == 2 and len(d["mismatches"]) == 1
    assert d["mismatches"][0] == {"object": "o", "idx": 2, "row": "firm_grip", "oracle": ["placed_stable", False], "stage2": ["crushed", False]}


def test_the_model_only_curves_read_r0s_first_decision_at_each_knob():
    rows_o, rows_p = [_row("commit_std", 0.9), _row("firm_grip", 0.95, irr=0.03)], [_row("commit_std", 0.4, irr=0.05)]
    orecs = [dict(_s2("o", 0, {K: "commit_std"}, []), rows=rows_o, curve_decisions=co.curve_decisions(rows_o))]
    mc = co.model_curves(orecs, [_s1rec("o", 0, rows=rows_p)])
    assert [p["knob"] for p in mc["oracle"]] == list(cv.R_GRID) == [p["knob"] for p in mc["r0_stage1"]]
    for i, r in enumerate(cv.R_GRID):
        k = cv.dkey("R0", cv.A_MAIN, r)
        assert mc["oracle"][i] == dict(knob=r, **cv.predicted_point([{"rows": rows_o, "decisions": co.curve_decisions(rows_o)}], k))
        assert mc["r0_stage1"][i] == dict(knob=r, **cv.predicted_point([{"rows": rows_p, "decisions": co.curve_decisions(rows_p)}], k))


# ---------------------------------------------------------------- the commands on synthetic runs

def _write_stage1(d, recs):
    os.makedirs(d)
    json.dump({"bundle": "B", "seed": 11, "split_sha256": "s", "scenes_sha256": "c", "manifest_sha256": "m", "code_revision": "x",
               "generator": "v11", "r_grid": list(cv.R_GRID), "delta_grid": list(cv.DELTA_GRID), "a_main": 10.0,
               "r_check": [list(x) for x in cv.R_CHECK], "quantile": 0.9, "cvar_level": 0.1, "stage1_care": 100.0},
              open(os.path.join(d, "run_config.json"), "w"))
    with open(os.path.join(d, "records.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    json.dump({"cases": len(recs)}, open(os.path.join(d, "run_metrics.json"), "w"))


def _write_stage2(d, recs, stage1):
    os.makedirs(d)
    json.dump({"bundle": "B", "seed": 11, "rules": list(cv.RULES), "n_seeds": 1, "stage1": stage1,
               "stage1_records_sha256": {stage1: cv._sha(os.path.join(stage1, "records.jsonl"))},
               "retry_sha256": "r", "split_sha256": "s", "code_revision": "x"}, open(os.path.join(d, "run_config.json"), "w"))
    for side in cv.SIDES:
        with open(cv.records_path(d, side), "w") as f:
            for r in recs:
                if r["side"] == side:
                    f.write(json.dumps(r) + "\n")
    json.dump({"stop": None}, open(os.path.join(d, "run_metrics.json"), "w"))


STD, FIRM = _key("commit_std"), _key("firm_grip", force=40.0)


def _all_keys(d0, d4):
    """Every rule's first decision at A_MAIN: R4's is d4, every other rule's d0."""
    return {cv.dkey(rule, cv.A_MAIN, k): d4 if rule == "R4" else d0 for rule in cv.RULES for k in cv.KNOBS[rule]}


def _world(tmp_path):
    """Stage 1, Stage 2 and their lock and summary files for 8 test objects (o0 and o1 thin plastic) and one validation
    object. o7 has no rows (every rule gives up); R0 gives up on o6 and grasps commit_std on o0 to o5: placed at idx 0,
    tipped at idx 1, where the second decision gives up. R4 grasps firm_grip: crushed on thin plastic, placed elsewhere."""
    s1, s2 = [], []
    for i in range(8):
        o = f"o{i}"
        for idx in range(2):
            rows = [] if i == 7 else [_row("commit_std", 0.6, action=STD), _row("firm_grip", 0.8, irr=0.02, action=FIRM)]
            s1.append(_s1rec(o, idx, rows=rows, thin=i < 2))
            dec = _all_keys(cv.GIVE_UP_ROW if i >= 6 else "commit_std", cv.GIVE_UP_ROW if i == 7 else "firm_grip")
            sims = []
            if i < 6:
                m2 = {"status": "no_rows", "rows": [], "decisions": {k: cv.GIVE_UP_ROW for k in dec}}
                sims.append(_sim("commit_std", "tipped", menu2=m2) if idx else _sim("commit_std", "placed_stable"))
            if i < 7:
                sims.append(_sim("firm_grip", "crushed" if i < 2 else "placed_stable"))
            s2.append(_s2(o, idx, dec, sims, thin=i < 2))
    s1.append(_s1rec("v0", 0, side="validation", rows=[_row("commit_std", 0.6, action=STD)]))
    s2.append(_s2("v0", 0, _all_keys("commit_std", "commit_std"), [_sim("commit_std", "placed_stable")], side="validation"))
    _write_stage1(str(tmp_path / "s1"), s1)
    _write_stage2(str(tmp_path / "s2"), s2, str(tmp_path / "s1"))
    lock = {"lock": {r: {"knob": k, "key": cv.dkey(r, cv.A_MAIN, k)} for r, k in
                     (("R0", 2.0), ("R1", 2.0), ("R2", 0.2), ("R3", 5.0), ("R4", None), ("R5", 0.2), ("R6", 10.0))}}
    json.dump(lock, open(tmp_path / "locked.json", "w"))
    json.dump({"provenance": {"records_sha256": {f"{tmp_path / 's2'} {side}": cv._sha(cv.records_path(str(tmp_path / "s2"), side))
                                                 for side in cv.SIDES}}}, open(tmp_path / "stage2_summary.json", "w"))
    return s1, s2


def _fake_oracle_case(prims, rec1, s, world, knob):
    """The oracle grasps firm_grip wherever Stage 1 has rows; it crushes on thin objects and places elsewhere."""
    rows = rec1["rows"]
    skip = cv.stage2_skip(rec1)
    dec = co.oracle_second(knob)(rows, ("R0",))
    dec = {K: "firm_grip" if rows else cv.GIVE_UP_ROW}
    sims = [] if not rows or skip else [_sim("firm_grip", "crushed" if rec1["thin"] else "placed_stable")]
    rec = cv.stage2_record(rec1, dec, ("R0",), None if skip else {"sims": sims, "stop": None}, 1)
    return dict(rec, oracle=True, knob=knob, curve_decisions=co.curve_decisions(rows), rows=rows, truth={}, sim_truth={},
                seconds=2.0)


def _run_oracle(tmp_path, monkeypatch, out, sides="test", cases=None):
    monkeypatch.setattr(co, "oracle_object", lambda job: {"records": [_fake_oracle_case(None, r, job[2], None, job[2]["knob"])
                                                                      for r in job[1]], "stop": None})
    monkeypatch.setattr(co, "require_bundle_generator", lambda b: "v11")
    argv = ["run", "--stage1", str(tmp_path / "s1"), "--locked", str(tmp_path / "locked.json"), "--out", str(out), "--sides", sides]
    co.main(argv + (["--cases", cases] if cases else []))
    (out / "run.exit").write_text("EXIT 0\n")


def test_run_writes_the_records_and_refuses_an_existing_out(tmp_path, monkeypatch):
    _world(tmp_path)
    _run_oracle(tmp_path, monkeypatch, tmp_path / "o1")
    recs = [json.loads(x) for x in open(tmp_path / "o1" / "records_test.jsonl")]
    assert len(recs) == 16 and all(r["oracle"] and r["knob"] == 2.0 for r in recs)
    cfg = json.load(open(tmp_path / "o1" / "run_config.json"))
    assert cfg["knob"] == 2.0 and cfg["key"] == K and cfg["locked_sha256"] == cv._sha(str(tmp_path / "locked.json"))
    assert json.load(open(tmp_path / "o1" / "run_metrics.json"))["stop"] is None
    with pytest.raises(SystemExit, match="exists"):
        _run_oracle(tmp_path, monkeypatch, tmp_path / "o1")


def test_smoke_reports_times_labels_and_the_determinism_check_on_validation_only(tmp_path, monkeypatch):
    _world(tmp_path)
    _run_oracle(tmp_path, monkeypatch, tmp_path / "sm_run", sides="validation", cases="v0:0")
    sm = co.main(["smoke", "--runs", str(tmp_path / "sm_run"), "--stage2", str(tmp_path / "s2"), "--out", str(tmp_path / "sm")])
    assert sm["cases"] == 1 and sm["run_cases"] == 16 and sm["errors"] == 0 and sm["stops"] == []
    assert sm["table"][0]["object"] == "v0" and sm["table"][0]["labels"] == ["placed_stable"]
    assert sm["determinism"]["compared"] == 0                        # firm_grip is not Stage 2's first row on v0
    assert sm["projection_cpu_h"] == pytest.approx(16 * 2.0 / 3600.0)
    assert os.path.exists(tmp_path / "sm" / "smoke_summary.md")
    _run_oracle(tmp_path, monkeypatch, tmp_path / "bad_run", sides="test", cases="o0:0")
    with pytest.raises(SystemExit, match="validation"):
        co.main(["smoke", "--runs", str(tmp_path / "bad_run"), "--stage2", str(tmp_path / "s2"), "--out", str(tmp_path / "sm2")])


def _summary(tmp_path, out, runs=None):
    return co.main(["summary", "--runs", runs or str(tmp_path / "o1"), "--stage2", str(tmp_path / "s2"),
                    "--stage2-summary", str(tmp_path / "stage2_summary.json"), "--locked", str(tmp_path / "locked.json"),
                    "--out", str(out)])


def test_summary_end_to_end_on_synthetic_runs(tmp_path, monkeypatch):
    _world(tmp_path)
    _run_oracle(tmp_path, monkeypatch, tmp_path / "o1")
    rep = _summary(tmp_path, tmp_path / "sum")
    r1 = rep["readings"]["1_oracle_vs_r0"]["all"]
    assert r1["cases"] == 16 and r1["objects"] == 8
    # the oracle: placed on the 10 non-thin cases with rows, crushed on the 4 thin ones, gives up on o7's 2. R0: placed on
    # the 6 idx-0 cases of o0 to o5, gives up on the other 10 (6 of them at the second decision)
    assert r1["success"]["diff"] == pytest.approx(4 / 16) and r1["irr"]["diff"] == pytest.approx(4 / 16)
    assert r1["give_up"]["diff"] == pytest.approx(-8 / 16)
    nt = rep["readings"]["1_oracle_vs_r0"]["no_thin"]
    assert nt["cases"] == 12 and nt["success"]["diff"] == pytest.approx(6 / 12) and nt["irr"]["diff"] == 0.0
    r2 = rep["readings"]["2_oracle_vs_r4"]["all"]
    assert (r2["success"]["diff"], r2["irr"]["diff"], r2["give_up"]["diff"]) == (0.0, 0.0, 0.0)
    assert rep["readings"]["3_reading"] == co.oracle_reading(r1)
    assert rep["readings"]["3_reading_no_thin"] == co.oracle_reading(nt)
    assert rep["counts"]["paired_dropped"] == {"dropped": 0, "stage2_only": 0, "oracle_only": 0, "both": 0}
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(co.__file__))))
    head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert rep["provenance"]["summary_code_revision"] == head and isinstance(rep["provenance"]["summary_dirty"], bool)
    g4 = rep["readings"]["4_groups"]["all"]
    assert {k: v["n"] for k, v in g4.items()} == {"R0 gave up and the oracle grasped": 2, "both grasped, different action": 12,
                                                  "both grasped, same action": 0, "the oracle gave up and R0 grasped": 0,
                                                  "both gave up": 2}
    assert g4["both grasped, same action"]["equal_action_different_id"] == 0
    assert g4["R0 gave up and the oracle grasped"]["oracle_first_labels"] == {"placed_stable": 2}
    assert rep["readings"]["5_calibration"]["all"]["attempts"] == 14
    assert rep["readings"]["6_determinism"]["mismatches"] == [] and rep["readings"]["6_determinism"]["compared"] == 14
    g8 = rep["readings"]["8_menu_limit"]
    assert {k: g8[k]["n"] for k in "abc"} == {"a": 2, "b": 2, "c": 12}
    assert g8["c"]["row_change"] == {"force or speed change": 12}
    assert g8["a"]["oracle_rows"] == 0
    assert set(rep["readings"]["7_model_curves"]) == {"oracle", "r0_stage1"}
    for f in ("oracle_summary.json", "oracle_summary.md", "oracle_points.png"):
        assert os.path.exists(tmp_path / "sum" / f)
    with pytest.raises(SystemExit, match="exists"):
        _summary(tmp_path, tmp_path / "sum")


def test_summary_refuses_unfinished_runs_and_stops_on_a_determinism_mismatch(tmp_path, monkeypatch):
    _world(tmp_path)
    _run_oracle(tmp_path, monkeypatch, tmp_path / "o1")
    (tmp_path / "o1" / "run.exit").write_text("EXIT 1\n")
    with pytest.raises(SystemExit, match="EXIT 0"):
        _summary(tmp_path, tmp_path / "sum")
    (tmp_path / "o1" / "run.exit").write_text("EXIT 0\n")
    recs = [json.loads(x) for x in open(tmp_path / "o1" / "records_test.jsonl")]
    with open(tmp_path / "o1" / "records_test.jsonl", "w") as f:
        for r in recs[1:]:
            f.write(json.dumps(r) + "\n")
    with pytest.raises(SystemExit, match="missing"):
        _summary(tmp_path, tmp_path / "sum")
    with open(tmp_path / "o1" / "records_test.jsonl", "w") as f:            # o2 0: firm_grip placed in Stage 2, crushed here
        for r in recs:
            if (r["object"], r["idx"]) == ("o2", 0):
                r["sims"][0]["label"] = "crushed"
            f.write(json.dumps(r) + "\n")
    with pytest.raises(SystemExit) as e:
        _summary(tmp_path, tmp_path / "sum")
    assert e.value.code == 3
    stop = json.load(open(tmp_path / "sum" / "oracle_summary_stop.json"))
    assert stop["determinism"]["mismatches"] and not os.path.exists(tmp_path / "sum" / "oracle_summary.json")


def test_summary_refuses_inputs_that_changed_since_the_runs(tmp_path, monkeypatch):
    _world(tmp_path)
    _run_oracle(tmp_path, monkeypatch, tmp_path / "o1")
    prov = json.load(open(tmp_path / "stage2_summary.json"))
    bad = tmp_path / "stage2_summary_bad.json"
    json.dump({"provenance": {"records_sha256": {k: "0" * 64 for k in prov["provenance"]["records_sha256"]}}}, open(bad, "w"))
    args = ["summary", "--runs", str(tmp_path / "o1"), "--stage2", str(tmp_path / "s2"), "--locked", str(tmp_path / "locked.json")]
    with pytest.raises(SystemExit, match="Stage 2 summary"):
        co.main(args + ["--stage2-summary", str(bad), "--out", str(tmp_path / "x1")])
    cfg_path = tmp_path / "o1" / "run_config.json"
    cfg = json.load(open(cfg_path))
    json.dump(dict(cfg, stage1_records_sha256={k: "0" * 64 for k in cfg["stage1_records_sha256"]}), open(cfg_path, "w"))
    with pytest.raises(SystemExit, match="Stage 1"):
        _summary(tmp_path, tmp_path / "x2")
    json.dump(cfg, open(cfg_path, "w"))
    lock = json.load(open(tmp_path / "locked.json"))
    json.dump(lock, open(tmp_path / "locked.json", "w"), indent=2)          # the same knob, another file
    with pytest.raises(SystemExit, match="lock file"):
        _summary(tmp_path, tmp_path / "x3")
