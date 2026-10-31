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


"""Plan H3, recoverability × uncertainty (robots/mujoco_lab/h3.py) on synthetic calibrate outputs."""

import copy
import hashlib
import json
import math
from collections import Counter

import numpy as np
import pytest

from robots.mujoco_lab.calibrate import FORCED_RULE
from robots.mujoco_lab import h3 as h3m
from robots.mujoco_lab.corpus import COM_S_RANGE, MASS_SCALE_RANGE, MU_RANGE
from robots.mujoco_lab.h3 import (CONDITIONS, CONDITIONS6, RECOVERY, SPREAD, X_M_WEIGHTS, X_T_WEIGHTS, X_WEIGHTS, cohort, contrasts,
                                  full_policy, gates, in_narrow, index, interaction, power, restore_ok, row_diff, summary, valid)
from robots.mujoco_lab.sweep import bootstrap_weights, variance_split

ARMS = {"never": "policy_no_probe", "probe_once": "probe_once", "sham_once": "sham_once", "policy": "policy"}
TRACE = {"probe": "weigh", "kind": "measured", "reading": {"mass": 1.0}, "thrown": False, "displacement_m": 0.0, "tilt_deg": 0.0,
         "cost": 0.2, "position": 0, "after_attempts": 0}
CHECK = {"pos_err_mm": 0.0, "rot_err_deg": 0.0, "qpos_err": 0.0, "qvel_err": 0.0, "mocap_err": 0.0, "same_case": True,
         "episode": True, "at_rest": True}


def _row(obj, idx, cost, rec="commit", error=None, artifact=False, probes=(), restores=(), **kw):
    row = {"object": obj, "idx": idx, "cost": float(cost), "recommendation": rec, "ended": "placed", "artifact": artifact,
           "attempts": [{"option": "commit_std", "outcome": "placed_stable"}], "probes": [dict(p) for p in probes],
           "restores": [dict(r) for r in restores], "first_menu": {"best_cost": 0.6, "recommended_cost": None},
           "theta": {"mass_scale": 1.2, "mu": 0.5, "com_s": None, "v_lift": 0.15}, "fill": 0.0,
           "bundle": "/root/x/robots/mujoco_lab/data/m2b2/folds/fold_0", **kw}
    if error:
        row["error"] = error
    return row


def _run(arm, cond, seed, rows, cases=None):
    (reset_s, price), (truth, belief) = RECOVERY[cond[0]], SPREAD[cond[1]]
    return {"protocol": ARMS[arm], "care_scale": 10.0, "abstain_cost": 10.0, "probe_time_scale": 1.0, "seed": seed,
            "force_noise": 0.05, "reset_s": reset_s, "reset_price": price, "truth_spread": truth, "prior_spread": belief,
            "n": len(rows), "rows": rows,
            "code": {"revision": "a" * 40, "dirty": False}, "cases": cases,
            "forced_rule": FORCED_RULE if arm in ("probe_once", "sham_once") else None}


def _design(costs, seeds=(12,), objects=("a", "b", "c"), per_object=2):
    """Every arm in every condition (the policy in full); costs(arm, cond, object, idx, seed) gives each row's cost."""
    return [_run(arm, cond, seed, [_row(o, i, costs(arm, cond, o, i, seed)) for o in objects for i in range(per_object)])
            for seed in seeds for cond in CONDITIONS for arm in ARMS]


def _of(runs, arm, cond, seed):
    return next(r for r in runs if r["protocol"] == ARMS[arm] and h3m.condition(r) == cond and r["seed"] == seed)


def _random_costs(seed):
    rng, table = np.random.default_rng(seed), {}
    return table, lambda *k: table.setdefault(k, float(rng.uniform(0.0, 10.0)))


def test_the_conditions_are_the_six_and_addendum_as_four():
    """Decisions 3 and 4 and Addendum A: a recovery level is (reset_s, reset price) and a spread (truth, prior); rec_hi
    runs with wide and narrow, narrowT_wideB with absorbing and rec20."""
    assert RECOVERY == {"absorbing": (None, "time"), "rec20": (20.0, "time"), "rec0": (0.0, "time"), "rec_hi": (0.0, "care")}
    assert SPREAD == {"wide": (1.0, 1.0), "narrow": (0.25, 0.25), "narrowT_wideB": (0.25, 1.0)}
    assert set(CONDITIONS6) == {(r, u) for r in ("absorbing", "rec20", "rec0") for u in ("wide", "narrow")} and len(CONDITIONS6) == 6
    assert len(CONDITIONS) == len(set(CONDITIONS)) == 10 and set(CONDITIONS) - set(CONDITIONS6) == {
        ("rec_hi", "wide"), ("rec_hi", "narrow"), ("absorbing", "narrowT_wideB"), ("rec20", "narrowT_wideB")}


def test_condition_reads_the_price_and_the_two_spreads():
    """A run's condition comes from its reset time and price and its truth and prior spreads; a pair outside the design
    is refused; milestone 4's files, which carry none of the fields, read as absorbing and wide."""
    for cond in CONDITIONS:
        assert h3m.condition(_run("never", cond, 12, [])) == cond
    base = _run("never", ("rec0", "wide"), 12, [])
    for edit in ({"truth_spread": 0.25, "prior_spread": 1.0},          # (rec0, narrowT_wideB)
                 {"truth_spread": 1.0, "prior_spread": 0.25},          # a narrow belief around a wide truth (rejected oracle-narrow)
                 {"reset_s": 20.0, "reset_price": "care"}, {"reset_s": None, "reset_price": "care"}):
        with pytest.raises(ValueError, match="not an H3 condition"):
            h3m.condition(dict(base, **edit))
    bare = {k: v for k, v in _run("never", ("absorbing", "wide"), 11, []).items()
            if k not in ("reset_s", "reset_price", "truth_spread", "prior_spread")}
    assert h3m.condition(bare) == ("absorbing", "wide")


def test_the_interaction_on_a_two_by_two_toy():
    """X_q = [q(absorbing, wide) − q(rec20, wide)] − [q(absorbing, narrow) − q(rec20, narrow)]; equal per-case values give
    a zero-width interval."""
    D = {("absorbing", "wide"): 4.0, ("rec20", "wide"): 1.0, ("absorbing", "narrow"): 2.0, ("rec20", "narrow"): 1.5,
         ("rec0", "wide"): 0.5, ("rec0", "narrow"): 0.5, ("rec_hi", "wide"): 0.5, ("rec_hi", "narrow"): 0.5,
         ("absorbing", "narrowT_wideB"): 0.5, ("rec20", "narrowT_wideB"): 0.5}
    runs = _design(lambda arm, cond, o, i, s: {"never": 10.0, "probe_once": 10.0 - D[cond], "sham_once": 10.0, "policy": 10.0}[arm])
    idx, objects = index(runs), ["a", "b", "c"]
    c = contrasts(idx, bootstrap_weights(objects, 200, np.random.default_rng(0)), objects, cohort(idx))
    x = interaction(c, "D")
    assert x["value"] == pytest.approx(2.5) and x["ci"] == pytest.approx([2.5, 2.5]) and x["verdict"] == "pass"
    assert interaction(c, "I")["value"] == pytest.approx(2.5) and interaction(c, "M")["value"] == pytest.approx(0.0)
    assert interaction(c, "P")["value"] == pytest.approx(0.0) and interaction(c, "M")["verdict"] == "undecided"
    D[("absorbing", "narrow")] = 6.0
    runs = _design(lambda arm, cond, o, i, s: {"never": 10.0, "probe_once": 10.0 - D[cond], "sham_once": 10.0, "policy": 10.0}[arm])
    idx = index(runs)
    c = contrasts(idx, bootstrap_weights(objects, 200, np.random.default_rng(0)), objects, cohort(idx))
    assert interaction(c, "D")["value"] == pytest.approx(-1.5) and interaction(c, "D")["verdict"] == "fail"


def test_d_is_i_plus_m_per_case_on_the_common_cohort():
    table, cost = _random_costs(3)
    runs = _design(cost, seeds=(12, 13))
    idx, objects = index(runs), ["a", "b", "c"]
    co = cohort(idx)
    c = contrasts(idx, bootstrap_weights(objects, 100, np.random.default_rng(0)), objects, co)
    keys = sorted(co)
    for cond in CONDITIONS:
        q = c["conditions"][cond]
        assert q["n"] == 12
        assert q["D"]["value"] == pytest.approx(q["I"]["value"] + q["M"]["value"])
        assert np.allclose(q["D"]["draws"], q["I"]["draws"] + q["M"]["draws"])
        never, probe = (np.array([table[(arm, cond, o, i, s)] for o, i, s in keys]) for arm in ("never", "probe_once"))
        policy = np.array([table[("policy", cond, o, i, s)] for o, i, s in keys])
        assert q["D"]["value"] == pytest.approx(float((never - probe).mean()))
        assert q["P"]["value"] == pytest.approx(float((never - policy).mean()))


def test_a_case_invalid_in_one_arm_leaves_every_contrast():
    """An error in the sham arm of one condition, or an artifact in the policy arm of another, removes that case (object,
    idx, seed) from every arm and every condition."""
    table, cost = _random_costs(4)
    runs = _design(cost, seeds=(12, 13))
    _of(runs, "sham_once", ("rec20", "narrow"), 13)["rows"][0]["error"] = "boom"
    _of(runs, "policy", ("absorbing", "wide"), 12)["rows"][3]["artifact"] = True
    idx, objects = index(runs), ["a", "b", "c"]
    co = cohort(idx)
    assert ("a", 0, 13) not in co and ("b", 1, 12) not in co and ("a", 0, 12) in co and len(co) == 10
    c = contrasts(idx, bootstrap_weights(objects, 100, np.random.default_rng(0)), objects, co)
    keys = sorted(co)
    for cond in CONDITIONS:
        assert c["conditions"][cond]["n"] == 10
        sham, probe = (np.array([table[(arm, cond, o, i, s)] for o, i, s in keys]) for arm in ("sham_once", "probe_once"))
        assert c["conditions"][cond]["I"]["value"] == pytest.approx(float((sham - probe).mean()))


def test_the_rebuilt_policy_run_and_its_errors():
    """The verdict seeds run the policy on the never arm's probe cases only: every probe case takes its subset row, every
    other case its never row; a probe case without a subset row, or a subset row of another case, is an error."""
    cond = ("absorbing", "wide")
    never = _run("never", cond, 12, [_row("a", 0, 5.0, rec="probe"), _row("a", 1, 2.0), _row("b", 0, 7.0, rec="probe")])
    sub = _run("policy", cond, 12, [_row("a", 0, 3.0, rec="probe"), _row("b", 0, 9.0, rec="probe")],
               cases={"path": "x.json", "n": 2, "sha256": "0" * 64})
    full = full_policy(never, sub)
    assert [(r["object"], r["idx"], r["cost"]) for r in full["rows"]] == [("a", 0, 3.0), ("a", 1, 2.0), ("b", 0, 9.0)]
    assert full["protocol"] == "policy" and full["n"] == 3 and full["seed"] == 12 and full["rebuilt"] == {"subset": 2, "filled": 1}
    with pytest.raises(ValueError, match="no row in the policy subset"):
        full_policy(never, dict(sub, rows=sub["rows"][:1]))
    with pytest.raises(ValueError, match="not a probe case"):
        full_policy(never, dict(sub, rows=sub["rows"] + [_row("a", 1, 2.0)]))
    assert index([never, sub])[("absorbing", "wide", "policy", 12)]["rows"] == full["rows"]
    with pytest.raises(ValueError, match="never"):
        index([sub])
    with pytest.raises(ValueError, match="two runs"):
        index([never, never])


def test_the_joint_draw_is_shared_by_all_conditions():
    """The interaction's draws equal one weighted mean of the per-case contributions per draw, which holds only when every
    condition uses the same object multiplicities."""
    table, cost = _random_costs(5)
    objects = ["a", "b", "c", "d"]
    runs = _design(cost, objects=tuple(objects))
    idx = index(runs)
    W = bootstrap_weights(objects, 300, np.random.default_rng(1))
    co = cohort(idx)
    x = interaction(contrasts(idx, W, objects, co), "D")

    def d(cond, o, i, s):
        return table[("never", cond, o, i, s)] - table[("probe_once", cond, o, i, s)]

    keys = sorted(co)
    u = [d(("absorbing", "wide"), *k) - d(("rec20", "wide"), *k) - d(("absorbing", "narrow"), *k) + d(("rec20", "narrow"), *k) for k in keys]
    S, n = np.zeros(len(objects)), np.zeros(len(objects))
    for (o, _, _), v in zip(keys, u):
        S[objects.index(o)] += v
        n[objects.index(o)] += 1
    assert np.allclose(x["draws"], (W @ S) / (W @ n)) and x["value"] == pytest.approx(float(np.mean(u)))


def _pilot_with_contributions(u):
    """Pilot runs (seed 11) whose per-case contribution to X_D is u[(object, idx)]: D(absorbing, wide) = u, D = 0 elsewhere."""
    objects = sorted({o for o, _ in u})
    per = max(i for _, i in u) + 1
    return _design(lambda arm, cond, o, i, s: 10.0 - (u[(o, i)] if (arm, cond) == ("probe_once", ("absorbing", "wide")) else 0.0),
                   seeds=(11,), objects=tuple(objects), per_object=per)


def test_power_counts_k_verdict_seeds_and_flags_a_missed_target():
    """Decision 9: SE(k)² = (σ_b² + σ_w² / (c k)) / n with k verdict seeds (the pilot does not count), k the smallest of 2,
    4 and 6 with 1.96 SE(k) ≤ 0.7 X̂_D; k = 6 and the target not met when none qualifies or X̂_D ≤ 0."""
    rng = np.random.default_rng(7)
    objs = [f"o{j:02d}" for j in range(40)]
    eff = {o: float(rng.normal(0.0, 0.3)) for o in objs}
    u = {(o, i): 1.0 + eff[o] + float(rng.normal(0.0, 1.0)) for o in objs for i in range(3)}
    p = power(index(_pilot_with_contributions(u)))
    keys = sorted(u)
    vals, groups = np.array([u[k] for k in keys]), np.array([k[0] for k in keys])
    sb, sw = variance_split(vals, groups)
    assert p["x_d"] == pytest.approx(float(vals.mean())) and p["sigma_b2"] == pytest.approx(sb) and p["sigma_w2"] == pytest.approx(sw)
    assert p["cases_per_object"] == pytest.approx(3.0) and p["n_objects"] == 40
    se = {k: math.sqrt((sb + sw / (3.0 * k)) / 40) for k in (2, 4, 6)}
    assert all(p["se"][str(k)] == pytest.approx(se[k]) for k in (2, 4, 6))
    want = next((k for k in (2, 4, 6) if 1.96 * se[k] <= 0.7 * vals.mean()), None)
    assert p["k"] == (want or 6) and p["target_met"] is (want is not None)
    low = power(index(_pilot_with_contributions({k: v - 1.0 - float(vals.mean()) for k, v in u.items()})))
    assert low["x_d"] <= 0 and low["k"] == 6 and low["target_met"] is False
    tiny = power(index(_pilot_with_contributions({k: 0.02 + 5.0 * (v - float(vals.mean())) for k, v in u.items()})))
    assert tiny["x_d"] > 0 and tiny["k"] == 6 and tiny["target_met"] is False


def test_valid_rejects_a_wrong_reset_spread_rule_or_cases_hash(tmp_path):
    cf = tmp_path / "cases.json"
    cf.write_text(json.dumps({"seed": 12, "cases": {"a": [0, 1]}}))
    sha = hashlib.sha256(cf.read_bytes()).hexdigest()
    run = _run("policy", ("rec20", "narrow"), 12, [_row("a", 0, 1.0), _row("a", 1, 1.0)], cases={"path": str(cf), "n": 2, "sha256": sha})
    good = tmp_path / "good.json"
    good.write_text(json.dumps(run))
    expect = {"revision": "a" * 40, "protocol": "policy", "reset_s": 20.0, "reset_price": "time", "truth_spread": 0.25,
              "prior_spread": 0.25, "seed": 12, "cases": str(cf)}
    assert valid(str(good), expect) == []
    for k, v in (("reset_s", None), ("reset_s", 0.0), ("reset_price", "care"), ("truth_spread", 1.0), ("prior_spread", 1.0), ("seed", 13),
                 ("protocol", "probe_once"), ("revision", "b" * 40)):
        assert valid(str(good), dict(expect, **{k: v})), k
    assert valid(str(good), {k: v for k, v in expect.items() if k != "cases"} | {"n": 2})      # a subset run where a full one is due
    cf.write_text(json.dumps({"seed": 12, "cases": {"a": [0, 1]}}) + "\n")                    # the same cases, other bytes
    assert any("sha256" in w for w in valid(str(good), expect))
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"seed": 12, "cases": {"a": [0, 2]}}))
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps(dict(run, cases={"path": str(other), "n": 2, "sha256": hashlib.sha256(other.read_bytes()).hexdigest()})))
    assert any("cases" in w for w in valid(str(wrong), dict(expect, cases=str(other))))
    forced = _run("probe_once", ("absorbing", "wide"), 12, [_row("a", 0, 1.0)])
    f = tmp_path / "forced.json"
    f.write_text(json.dumps(forced))
    fexp = {"revision": "a" * 40, "protocol": "probe_once", "reset_s": None, "reset_price": "time", "truth_spread": 1.0,
            "prior_spread": 1.0, "seed": 12, "n": 1}
    assert valid(str(f), fexp) == []
    f.write_text(json.dumps(dict(forced, forced_rule="squeeze always")))
    assert any("rule" in w for w in valid(str(f), fexp))


def test_in_narrow_reads_the_narrow_ranges():
    """The matched-truth check's cases (decision 4): the mass scale, mu and a non-None com_s inside the ranges at f = 0.25,
    a filled container on its own mass range."""
    def row(mass, mu=0.5, com_s=None, fill=0.0):
        return {"theta": {"mass_scale": mass, "mu": mu, "com_s": com_s}, "fill": fill}

    assert in_narrow(row(1.2)) and in_narrow(row(1.0, fill=0.5)) and in_narrow(row(1.2, com_s=0.55))
    assert not in_narrow(row(0.9)) and not in_narrow(row(1.6)) and not in_narrow(row(1.2, mu=0.65))
    assert not in_narrow(row(1.2, com_s=0.7)) and not in_narrow(row(1.2, fill=0.5))


def _pilot():
    """A passing pilot: every arm in every condition at seed 11; the forced arms' probe traces equal; the sham's hashes
    equal; one restore with a passing check."""
    runs = _design(lambda *k: 5.0, seeds=(11,))
    for cond in CONDITIONS:
        for r in _of(runs, "probe_once", cond, 11)["rows"]:
            r.update(probes=[dict(TRACE)], forced_probe="weigh", post_probe="commit_std", changed="same")
        for r in _of(runs, "sham_once", cond, 11)["rows"]:
            r.update(probes=[dict(TRACE, sham=True, belief_hash_before="h", belief_hash_after="h", belief_hash_next="h")],
                     forced_probe="weigh", post_probe="commit_std", changed="same")
    _of(runs, "never", ("rec20", "wide"), 11)["rows"][0]["restores"] = [{"after": "attempt", "event": "dropped_high", "check": dict(CHECK)}]
    return runs


def _m4(tmp_path, runs):
    """Milestone 4's seed-11 outputs of the absorbing, wide condition, from another checkout root."""
    def strip(r):
        out = {k: v for k, v in r.items() if k not in ("restores", "reset_s", "reset_price", "truth_spread", "prior_spread")}
        out["bundle"] = r["bundle"].replace("/root/x/", "/root/m4/")
        return out

    policy = _of(runs, "policy", ("absorbing", "wide"), 11)
    never = _of(runs, "never", ("absorbing", "wide"), 11)
    (tmp_path / "policy_k10_A10_s1_seed11.json").write_text(json.dumps(dict(policy, rows=[strip(r) for r in policy["rows"]])))
    (tmp_path / "policy_no_probe_k10_A10_s1_seed11_cases.json").write_text(
        json.dumps(dict(never, rows=[strip(r) for r in never["rows"][:2]], n=2)))
    return str(tmp_path)


def test_a_restore_check_passes_only_within_the_limits():
    assert restore_ok(CHECK) and restore_ok(dict(CHECK, pos_err_mm=0.9, rot_err_deg=0.09))
    for k, v in (("pos_err_mm", 1.5), ("rot_err_deg", 0.2), ("qvel_err", 1e-3), ("mocap_err", 1e-3), ("qpos_err", 1e-6), ("pos_err_mm", None),
                 ("same_case", False), ("episode", False), ("at_rest", None)):
        assert not restore_ok(dict(CHECK, **{k: v})), k
    assert not restore_ok({k: v for k, v in CHECK.items() if k != "qvel_err"})


def test_a_restore_matches_the_start_rest_state(tmp_path):
    """Gr compares the restore's at_rest with the start's (the row's reset record): a start that never settled is
    rebuilt as it was (milestone 4's seed-11 run had 39 such starts, all artifacts)."""
    assert restore_ok(dict(CHECK, at_rest=False), start_at_rest=False)
    assert not restore_ok(CHECK, start_at_rest=False) and not restore_ok(dict(CHECK, at_rest=False))
    runs = _pilot()
    row = _of(runs, "never", ("rec20", "wide"), 11)["rows"][0]
    row["reset"] = {"settle_s": 3.0, "at_rest": False, "zone": "table"}
    row["restores"][0]["check"]["at_rest"] = False
    g = gates(index(runs), _m4(tmp_path, runs))
    assert g["Gr"]["pass"] and g["Gr"]["restores"] == 1


def test_ga_compares_every_milestone4_key_but_the_wall_time():
    m4 = _row("a", 0, 1.0, seconds=3.2, bundle="/home/u/Codes/RPent/robots/mujoco_lab/data/m2b2/folds/fold_0")
    h3 = dict(copy.deepcopy(m4), seconds=5.9, bundle="/home/u/Codes/RPent-h3/robots/mujoco_lab/data/m2b2/folds/fold_0",
              reset_s=None, reset_price="time", truth_spread=1.0, prior_spread=1.0)
    assert row_diff(h3, m4) == []                                # a rerun never reproduces its wall time
    assert row_diff(dict(h3, cost=1.5), m4) == ["cost"]
    assert row_diff(dict(h3, attempts=[{"option": "light_grip", "outcome": "placed_stable"}]), m4) == ["attempts"]
    assert row_diff({k: v for k, v in h3.items() if k != "theta"}, m4) == ["theta"]


def test_the_gates_pass_a_clean_pilot_and_catch_each_failure(tmp_path):
    runs = _pilot()
    m4 = _m4(tmp_path, runs)
    g = gates(index(runs), m4)
    assert all(g[k]["pass"] for k in ("G1", "Gr", "Gs", "Gp", "Ga", "first_menus")) and g["pass"] and g["Gr"]["restores"] == 1

    def broken(edit):
        rs = copy.deepcopy(runs)
        edit(rs)
        return gates(index(rs), m4)

    assert not broken(lambda rs: _of(rs, "policy", ("rec0", "narrow"), 11)["rows"][1].update(cost=6.0))["G1"]["pass"]
    assert not broken(lambda rs: _of(rs, "policy", ("rec0", "wide"), 11)["rows"][1].update(recommendation="probe"))["first_menus"]["pass"]
    assert not broken(lambda rs: _of(rs, "never", ("rec20", "wide"), 11)["rows"][0]["restores"][0]["check"].update(pos_err_mm=2.0))["Gr"]["pass"]
    assert not broken(lambda rs: _of(rs, "never", ("rec20", "wide"), 11)["rows"][0]["restores"][0]["check"].update(at_rest=False))["Gr"]["pass"]
    assert not broken(lambda rs: _of(rs, "sham_once", ("rec0", "wide"), 11)["rows"][0]["probes"][0].update(belief_hash_after="g"))["Gs"]["pass"]
    assert not broken(lambda rs: _of(rs, "sham_once", ("rec0", "wide"), 11)["rows"][0]["probes"][0].update(belief_hash_next="g"))["Gs"]["pass"]
    assert not broken(lambda rs: _of(rs, "sham_once", ("rec20", "narrow"), 11)["rows"][1]["probes"][0].update(reading={"mass": 1.1}))["Gp"]["pass"]
    assert not broken(lambda rs: _of(rs, "policy", ("absorbing", "wide"), 11)["rows"][2].update(cost=6.0))["Ga"]["pass"]
    assert not broken(lambda rs: _of(rs, "never", ("absorbing", "wide"), 11)["rows"][1].update(cost=6.0))["Ga"]["pass"]
    assert broken(lambda rs: _of(rs, "never", ("absorbing", "wide"), 11)["rows"][4].update(cost=6.0))["Ga"]["pass"]   # outside m4's cases
    def crash(rs):                   # the same case raised in both forced arms (calibrate's exception row)
        for arm in ("probe_once", "sham_once"):
            rows = _of(rs, arm, ("rec0", "wide"), 11)["rows"]
            rows[3] = {"object": rows[3]["object"], "idx": rows[3]["idx"], "error": "boom", "executed": False}

    ex = broken(crash)
    assert ex["Gp"]["pass"] and ex["Gs"]["pass"] and ex["Gp"]["pairs"] == 60      # 10 conditions x 6 cases
    thrown = broken(lambda rs: [r["probes"][0].update(thrown=True, belief_hash_next=None)
                                for arm in ("probe_once", "sham_once") for r in _of(rs, arm, ("absorbing", "narrow"), 11)["rows"][:1]])
    assert thrown["Gs"]["pass"] and thrown["Gp"]["pass"]         # a sham that ended the case has no next menu


def test_summary_runs_on_a_small_design():
    table, cost = _random_costs(9)
    runs = _design(cost, seeds=(12, 13))
    for r in _of(runs, "probe_once", ("absorbing", "wide"), 12)["rows"]:
        r.update(probes=[dict(TRACE)], changed="same")
    _of(runs, "probe_once", ("absorbing", "wide"), 12)["rows"][0]["changed"] = "decline"
    out = summary(runs, n_boot=50)
    assert out["h3"]["verdict"] in ("pass", "fail", "undecided") and out["h3_info"]["verdict"] in ("pass", "fail", "undecided")
    assert set(out["secondary"]) >= {"x_m", "x_p", "q", "rec0_order", "matched_truth", "x_pen", "x_i_term", "x_i_matched", "x_truth",
                                     "x_t_by_event"}
    assert all(out[k]["verdict"] in ("pass", "fail", "undecided") for k in ("h3_term", "h3_matched"))
    assert set(out["holm"]) == {"X_D", "X_I", "X_T", "X_M*"} and set(out["sensitivity"]) >= {"cohort6", "pairwise", "artifacts_counted"}
    assert out["n_cohort"] == 12 and out["seeds"] == [12, 13]
    shares = out["decision_change"]["absorbing/wide"]
    assert shares["n_probed"] == 6 and shares["share"]["decline"] == pytest.approx(1 / 6) and shares["share"]["same"] == pytest.approx(5 / 6)
    json.dumps(out)
    assert "H3" in out["markdown"]


def _with_forced_probes(runs, seeds):
    """The forced arms of ``seeds`` with equal probe traces and a sham whose hashes are equal."""
    for seed in seeds:
        for cond in CONDITIONS:
            for r in _of(runs, "probe_once", cond, seed)["rows"]:
                r.update(probes=[dict(TRACE)], forced_probe="weigh", post_probe="commit_std", changed="same")
            for r in _of(runs, "sham_once", cond, seed)["rows"]:
                r.update(probes=[dict(TRACE, sham=True, belief_hash_before="h", belief_hash_after="h", belief_hash_next="h")],
                         forced_probe="weigh", post_probe="commit_std", changed="same")
    return runs


def test_a_condition_without_probe_cases_takes_the_never_run_as_its_policy():
    """Decision 5 at an empty subset: at a verdict seed whose never run has no probe case, the runner skips the policy
    job and index() rebuilds the policy run from the never run alone; a never run with a probe case, or the pilot seed,
    keeps a missing policy run missing."""
    cond = ("rec0", "narrow")
    full = _design(lambda *k: 5.0, seeds=(11, 12))
    runs = [r for r in full if r is not _of(full, "policy", cond, 12)]
    idx = index(runs)
    pol, nev = idx[(*cond, "policy", 12)], idx[(*cond, "never", 12)]
    assert pol["protocol"] == "policy" and pol["rows"] == nev["rows"] and pol["rebuilt"] == {"subset": 0, "filled": 6}
    assert len(cohort({k: v for k, v in idx.items() if k[3] == 12})) == 6
    probed = copy.deepcopy(runs)
    _of(probed, "never", cond, 12)["rows"][0]["recommendation"] = "probe"
    with pytest.raises(ValueError, match="no policy run"):
        cohort(index(probed))
    pilot = [r for r in full if r is not _of(full, "policy", cond, 11)]
    assert (*cond, "policy", 11) not in index(pilot)


def test_power_reports_the_probe_cases_per_condition():
    u = {(o, i): 1.0 + 0.1 * i for o in ("a", "b", "c") for i in range(3)}
    runs = _pilot_with_contributions(u)
    _of(runs, "never", ("absorbing", "narrow"), 11)["rows"][4]["recommendation"] = "probe"
    p = power(index(runs))
    assert p["probe_cases"] == {h3m.label(c): int(c == ("absorbing", "narrow")) for c in CONDITIONS}
    assert p["cohort_share"] == pytest.approx(1.0) and p["n_cases"] == 9


def test_the_summary_checks_every_verdict_seed_and_refuses_on_a_failure(tmp_path, monkeypatch):
    """Decision 8 holds in every sham case: at each verdict seed the summary checks Gs, Gp, Gr and the first menus on the
    cohort's cases, and the summary command exits 1 when a check fails."""
    monkeypatch.setattr(h3m.tracking, "log_run", lambda *a, **k: None)
    runs = _with_forced_probes(_design(lambda *k: 5.0, seeds=(12, 13)), (12, 13))
    s = summary(runs, n_boot=20)
    assert s["checks"]["pass"] and s["checks"]["Gs"]["shams"] == 120 and s["checks"]["Gp"]["pairs"] == 120   # 2 seeds x 10 conditions x 6

    def checks(edit):
        rs = copy.deepcopy(runs)
        edit(rs)
        return summary(rs, n_boot=20)["checks"]

    assert not checks(lambda rs: _of(rs, "sham_once", ("rec20", "wide"), 13)["rows"][0]["probes"][0].update(belief_hash_after="g"))["Gs"]["pass"]
    assert not checks(lambda rs: _of(rs, "sham_once", ("rec0", "narrow"), 12)["rows"][2]["probes"][0].update(reading={"mass": 1.1}))["Gp"]["pass"]
    assert not checks(lambda rs: _of(rs, "never", ("rec20", "wide"), 12)["rows"][0].update(
        restores=[{"after": "attempt", "event": "bruised", "check": dict(CHECK, pos_err_mm=2.0)}]))["Gr"]["pass"]
    assert not checks(lambda rs: _of(rs, "policy", ("rec0", "wide"), 13)["rows"][1].update(first_menu={"best_cost": 9.0}))["first_menus"]["pass"]
    out = tmp_path / "out"
    out.mkdir()
    bad = copy.deepcopy(runs)
    _of(bad, "probe_once", ("absorbing", "narrow"), 12)["rows"][5]["probes"][0]["sham"] = True
    for j, r in enumerate(bad):
        (out / f"run{j}.json").write_text(json.dumps(r))
    argv = ["summary", "--dir", str(out), "--out", str(tmp_path / "s.json"), "--markdown", str(tmp_path / "s.md"), "--n-boot", "20"]
    assert h3m.main(argv) == 1 and not json.load(open(tmp_path / "s.json"))["checks"]["Gs"]["pass"]


def test_the_arm_table_counts_irreversible_events_and_cases():
    """Decision 6: irreversible events per 1,000 cases (an attempt with an irreversible outcome, a probe that threw or
    damaged the object), next to milestone 4's irreversible cases per 1,000; under recovery a case holds several."""
    runs = _with_forced_probes(_design(lambda *k: 5.0, seeds=(12,)), (12,))
    _of(runs, "never", ("rec20", "wide"), 12)["rows"][0]["attempts"] = [
        {"option": "commit_std", "outcome": "bruised"}, {"option": "light_grip", "outcome": "bruised"},
        {"option": "light_grip", "outcome": "placed_stable"}]
    r = _of(runs, "probe_once", ("rec0", "narrow"), 12)["rows"][3]
    r["probes"][0]["thrown"] = True
    r["attempts"] = [{"option": "commit_std", "outcome": "dropped_high"}, {"option": "commit_std", "outcome": "placed_stable"}]
    s = summary(runs, n_boot=20)
    never, forced = s["arms"]["rec20/wide/never"], s["arms"]["rec0/narrow/probe_once"]
    assert never["irreversible_per_1000"] == pytest.approx(1000 / 6) and never["irreversible_events_per_1000"] == pytest.approx(2000 / 6)
    assert forced["irreversible_per_1000"] == pytest.approx(1000 / 6) and forced["irreversible_events_per_1000"] == pytest.approx(2000 / 6)
    assert len(never["irreversible_events_per_1000_ci"]) == 2 and "irreversible events per 1,000" in s["markdown"]


def test_the_matched_truth_check_weights_each_case_by_its_selection_probability():
    """Decision 4's check, corrected: a matched case enters with weight 1 / P(selected | its branch) (f² without a CoM
    shift or for a filled container, f³ with one), so the matched set has the narrow class's mixture; the raw n and
    Kish's effective n are reported."""
    mid, mu, cs = math.sqrt(MASS_SCALE_RANGE[0] * MASS_SCALE_RANGE[1]), sum(MU_RANGE) / 2, sum(COM_S_RANGE) / 2
    d = {("a", 0): 1.0, ("a", 1): 3.0}
    runs = _design(lambda arm, cond, o, i, s: 5.0 - (d.get((o, i), 0.0) if (arm, cond) == ("probe_once", ("absorbing", "wide")) else 0.0),
                   seeds=(12,))
    for arm in ARMS:
        for r in _of(runs, arm, ("absorbing", "wide"), 12)["rows"]:
            r["theta"] = {"mass_scale": 2.9, "mu": mu, "com_s": None, "v_lift": 0.15}
        rows = _of(runs, arm, ("absorbing", "wide"), 12)["rows"]
        rows[0]["theta"] = {"mass_scale": mid, "mu": mu, "com_s": None, "v_lift": 0.15}
        rows[1]["theta"] = {"mass_scale": mid, "mu": mu, "com_s": cs, "v_lift": 0.15}
    m = summary(runs, n_boot=50)["secondary"]["matched_truth"]
    assert m["n"] == 2 and m["n_eff"] == pytest.approx(25 / 17)
    assert m["absorbing/wide"]["D"]["value"] == pytest.approx((1.0 + 4 * 3.0) / 5) and m["Q"]["value"] == pytest.approx(2.6)
    assert h3m.selection_p(rows[0]) == pytest.approx(0.25 ** 2) and h3m.selection_p(rows[1]) == pytest.approx(0.25 ** 3)
    assert h3m.selection_p({"theta": {"mass_scale": 1.0, "mu": mu, "com_s": None}, "fill": 0.5}) == pytest.approx(0.25 ** 2)


def test_the_selection_probability_is_the_generators_mass_on_the_narrow_ranges():
    """The weights' premise, on the truth generator: the share of draws inside the narrow ranges is selection_p per
    branch (no shift, shift, filled), within 4 standard errors."""
    from robots.mujoco_lab.corpus import apply_damage, draw_damage, draw_theta
    from robots.mujoco_lab.fragility import THRESHOLDS_V0, Thresholds, entry_of, load_classes

    rng = np.random.default_rng(0)
    hits, n, p = Counter(), Counter(), {}

    def add(branch, row):
        n[branch] += 1
        hits[branch] += in_narrow(row)
        p[branch] = h3m.selection_p(row)

    for _ in range(100_000):
        th, _ = draw_theta(rng, 1.0)
        add("none" if th["com_s"] is None else "shift", {"theta": th, "fill": 0.0})
    classes, thr = load_classes(), Thresholds.from_json(THRESHOLDS_V0)
    entry = entry_of("ycb__mug", classes)[0]
    for _ in range(20_000):
        th, _ = draw_theta(rng, 1.0)
        dmg = draw_damage(rng, entry, thr, 1.0)
        if dmg["fill"] > 0:
            add("filled", {"theta": apply_damage(th, {}, dmg)[0], "fill": dmg["fill"]})
    assert min(n.values()) > 5000
    for b in ("none", "shift", "filled"):
        assert abs(hits[b] / n[b] - p[b]) < 4 * math.sqrt(p[b] * (1 - p[b]) / n[b]), b


def test_gates_and_power_log_to_wandb(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(h3m.tracking, "log_run", lambda job, config, metrics: calls.append((job, config, metrics)))
    runs = _pilot()
    (tmp_path / "m4").mkdir()
    m4 = _m4(tmp_path / "m4", runs)
    out = tmp_path / "out"
    out.mkdir()
    for j, r in enumerate(runs):
        (out / f"run{j}.json").write_text(json.dumps(r))
    assert h3m.main(["power", "--dir", str(out), "--out", str(tmp_path / "power.json")]) == 0
    assert h3m.main(["gates", "--dir", str(out), "--m4-dir", m4, "--out", str(tmp_path / "gates.json")]) == 0
    assert [c[0] for c in calls] == ["h3_power", "h3_gates"]
    assert calls[0][2]["k"] == 6 and calls[0][2]["target_met"] is False and calls[1][2]["pass"] is True


# ---------------------------------------------------------------- Addendum A: X_T, X_M*, Holm, the event split, the cohorts
def test_x_d_splits_into_termination_and_penalty_and_into_matched_belief_and_truth():
    """Addendum A: X_D = X_T + X_pen and X_D = X_M* + X_truth case by case, and each interaction reads its four cells."""
    table, cost = _random_costs(21)
    runs = _design(cost, seeds=(12, 13))
    idx = index(runs)
    keys, vals = h3m.case_values(idx, cohort(idx))
    xd, xt, xm, xpen, xtruth = (h3m.case_x(vals, "D", w) for w in (X_WEIGHTS, X_T_WEIGHTS, X_M_WEIGHTS, h3m.X_PEN_WEIGHTS,
                                                                   h3m.X_TRUTH_WEIGHTS))
    assert np.allclose(xd, xt + xpen) and np.allclose(xd, xm + xtruth) and len(keys) == 12

    def d(rec, u, k):
        return table[("never", (rec, u), *k)] - table[("probe_once", (rec, u), *k)]

    for j, k in enumerate(keys):
        assert xt[j] == pytest.approx(d("absorbing", "wide", k) - d("rec_hi", "wide", k) - d("absorbing", "narrow", k) + d("rec_hi", "narrow", k))
        assert xm[j] == pytest.approx(d("absorbing", "narrowT_wideB", k) - d("rec20", "narrowT_wideB", k)
                                      - d("absorbing", "narrow", k) + d("rec20", "narrow", k))
    s = summary(runs, n_boot=50)
    assert s["h3_term"]["value"] == pytest.approx(float(xt.mean())) and s["h3_matched"]["value"] == pytest.approx(float(xm.mean()))
    assert s["h3"]["value"] == pytest.approx(s["h3_term"]["value"] + s["secondary"]["x_pen"]["value"])
    assert s["h3"]["value"] == pytest.approx(s["h3_matched"]["value"] + s["secondary"]["x_truth"]["value"])
    xi_t = h3m.case_x(vals, "I", X_T_WEIGHTS)
    assert s["secondary"]["x_i_term"]["value"] == pytest.approx(float(xi_t.mean()))


def test_the_x_t_split_by_the_ending_event_sums_to_x_t():
    """Addendum A correction 2: each arm's absorbing − rec_hi cost difference goes to the event that ended its absorbing
    case (the forced probe's own, a commit attempt's, none), signed as in X_T; the parts sum to X_T."""
    runs = _design(lambda *k: 5.0, seeds=(12,))
    thrown = _of(runs, "probe_once", ("absorbing", "wide"), 12)["rows"][0]            # case (a, 0): the forced probe threw it
    thrown.update(cost=65.0, probes=[dict(TRACE, thrown=True)], attempts=[])
    dropped = _of(runs, "never", ("absorbing", "narrow"), 12)["rows"][3]              # case (b, 1): a commit dropped it
    dropped.update(cost=63.0, attempts=[{"option": "commit_std", "outcome": "dropped_high"}])
    _of(runs, "never", ("rec_hi", "narrow"), 12)["rows"][3].update(cost=55.0)         # rec_hi: 5 kappa, restored, placed
    s = summary(runs, n_boot=50)
    e = s["secondary"]["x_t_by_event"]
    assert e["probe"]["value"] == pytest.approx(-60.0 / 6) and e["attempt"]["value"] == pytest.approx(-8.0 / 6)
    assert e["none"]["value"] == pytest.approx(0.0)
    assert sum(e[k]["value"] for k in h3m.EVENTS) == pytest.approx(s["h3_term"]["value"])
    assert h3m.ending_event(thrown) == "probe" and h3m.ending_event(dropped) == "attempt" and h3m.ending_event(_row("c", 0, 1.0)) == "none"


def test_the_bootstrap_p_and_holm_on_known_values():
    """Addendum A correction 5: p = 2 min(share of draws <= 0, share >= 0), at most 1, over the finite draws; Holm at 0.05
    steps down over the four verdicts."""
    assert h3m.boot_p(np.array([1.0, 2.0, 3.0, -1.0])) == pytest.approx(0.5)
    assert h3m.boot_p(np.array([1.0, 2.0])) == 0.0 and h3m.boot_p(np.array([0.0, 0.0])) == 1.0
    assert h3m.boot_p(np.array([np.nan, 1.0, -1.0])) == pytest.approx(1.0) and math.isnan(h3m.boot_p(np.array([np.nan])))
    h = h3m.holm({"X_D": 0.01, "X_I": 0.04, "X_T": 0.03, "X_M*": 0.005})
    assert {k: v["p_holm"] for k, v in h.items()} == pytest.approx({"X_M*": 0.02, "X_D": 0.03, "X_T": 0.06, "X_I": 0.06})
    assert {k for k, v in h.items() if v["reject"]} == {"X_M*", "X_D"} and h["X_T"]["p"] == 0.03
    n = h3m.holm({"X_D": 0.001, "X_I": float("nan")})
    assert n["X_D"]["reject"] and n["X_D"]["p_holm"] == pytest.approx(0.002) and not n["X_I"]["reject"]


def test_power_reports_x_t_and_x_m_with_their_ses_and_both_cohort_shares():
    """Addendum A: the pilot's X_T and X_M* with SE(k) as for X_D (k still from X_D), the projected verdict at the chosen
    k, and the 10- and 6-condition cohorts' shares."""
    rng = np.random.default_rng(8)
    u = {(o, i): 1.0 + float(rng.normal(0.0, 0.5)) for o in ("a", "b", "c") for i in range(3)}
    runs = _pilot_with_contributions(u)                  # u in D(absorbing, wide): X_D = X_T = mean u, X_M* = 0
    _of(runs, "never", ("rec_hi", "wide"), 11)["rows"][0]["error"] = "boom"               # (a, 0) leaves the 10-condition cohort
    p = power(index(runs))
    rest = [v for k, v in sorted(u.items()) if k != ("a", 0)]
    assert p["x_d"] == pytest.approx(float(np.mean(rest))) and p["x_t"]["value"] == pytest.approx(p["x_d"])
    assert p["x_t"]["se"] == pytest.approx(p["se"]) and p["x_m"]["value"] == pytest.approx(0.0)
    assert all(v == pytest.approx(0.0) for v in p["x_m"]["se"].values())
    k = str(p["k"])
    assert p["se_at_k"] == pytest.approx({"x_d": p["se"][k], "x_t": p["x_t"]["se"][k], "x_m": 0.0})
    assert p["x_t"]["projected"] == h3m.verdict([p["x_d"] - 1.96 * p["se"][k], p["x_d"] + 1.96 * p["se"][k]])
    assert p["cohort_share"] == pytest.approx(8 / 9) and p["cohort6_share"] == pytest.approx(1.0) and p["n_cases"] == 8


def test_the_cohort_spans_ten_conditions_and_the_sensitivity_row_six():
    """Addendum A correction 6: an error in a new condition removes the case from every contrast; the 6-condition
    cohort's X_D and X_I keep it."""
    table, cost = _random_costs(22)
    runs = _design(cost, seeds=(12,))
    _of(runs, "sham_once", ("absorbing", "narrowT_wideB"), 12)["rows"][2]["error"] = "boom"   # case (b, 0)
    idx = index(runs)
    assert len(cohort(idx)) == 5 and ("b", 0, 12) not in cohort(idx)
    six = h3m._cohort(idx, conditions=CONDITIONS6)
    assert len(six) == 6
    keys, vals = h3m.case_values(idx, six, conditions=CONDITIONS6)
    s = summary(runs, n_boot=50)
    c6 = s["sensitivity"]["cohort6"]
    assert s["n_cohort"] == 5 and c6["n"] == 6
    assert c6["D"]["value"] == pytest.approx(float(h3m.case_x(vals, "D", X_WEIGHTS).mean()))
    assert c6["I"]["value"] == pytest.approx(float(h3m.case_x(vals, "I", X_WEIGHTS).mean()))


def test_the_arm_table_reports_the_guard_flag_share():
    """Addendum A correction 4: the share of the cohort's cases whose first menu the guard flagged, per condition and arm."""
    runs = _design(lambda *k: 5.0, seeds=(12,))
    for j in (0, 4):
        _of(runs, "never", ("absorbing", "narrowT_wideB"), 12)["rows"][j]["guard"] = {"flag": True, "score": 0.9}
    _of(runs, "never", ("absorbing", "narrowT_wideB"), 12)["rows"][1]["guard"] = {"flag": False, "score": 0.1}
    s = summary(runs, n_boot=20)
    assert s["arms"]["absorbing/narrowT_wideB/never"]["guard_flag_share"] == pytest.approx(2 / 6)
    assert s["arms"]["absorbing/narrow/never"]["guard_flag_share"] == pytest.approx(0.0)
    assert "guard flag" in s["markdown"]
