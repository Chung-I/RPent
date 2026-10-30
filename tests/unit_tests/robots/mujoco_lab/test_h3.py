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

import numpy as np
import pytest

from robots.mujoco_lab.calibrate import FORCED_RULE
from robots.mujoco_lab.h3 import (CONDITIONS, RECOVERY, SPREAD, cohort, contrasts, full_policy, gates, in_narrow, index, interaction,
                                  power, restore_ok, row_diff, summary, valid)
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
    rec, spread = cond
    return {"protocol": ARMS[arm], "care_scale": 10.0, "abstain_cost": 10.0, "probe_time_scale": 1.0, "seed": seed,
            "force_noise": 0.05, "reset_s": RECOVERY[rec], "spread": SPREAD[spread], "n": len(rows), "rows": rows,
            "code": {"revision": "a" * 40, "dirty": False}, "cases": cases,
            "forced_rule": FORCED_RULE if arm in ("probe_once", "sham_once") else None}


def _design(costs, seeds=(12,), objects=("a", "b", "c"), per_object=2):
    """Every arm in every condition (the policy in full); costs(arm, cond, object, idx, seed) gives each row's cost."""
    return [_run(arm, cond, seed, [_row(o, i, costs(arm, cond, o, i, seed)) for o in objects for i in range(per_object)])
            for seed in seeds for cond in CONDITIONS for arm in ARMS]


def _of(runs, arm, cond, seed):
    return next(r for r in runs if r["protocol"] == ARMS[arm] and (r["reset_s"], r["spread"]) == (RECOVERY[cond[0]], SPREAD[cond[1]])
                and r["seed"] == seed)


def _random_costs(seed):
    rng, table = np.random.default_rng(seed), {}
    return table, lambda *k: table.setdefault(k, float(rng.uniform(0.0, 10.0)))


def test_the_conditions_are_three_recovery_levels_by_two_spreads():
    assert RECOVERY == {"absorbing": None, "rec20": 20.0, "rec0": 0.0} and SPREAD == {"wide": 1.0, "narrow": 0.25}
    assert len(CONDITIONS) == 6 and set(CONDITIONS) == {(r, u) for r in RECOVERY for u in SPREAD}


def test_the_interaction_on_a_two_by_two_toy():
    """X_q = [q(absorbing, wide) − q(rec20, wide)] − [q(absorbing, narrow) − q(rec20, narrow)]; equal per-case values give
    a zero-width interval."""
    D = {("absorbing", "wide"): 4.0, ("rec20", "wide"): 1.0, ("absorbing", "narrow"): 2.0, ("rec20", "narrow"): 1.5,
         ("rec0", "wide"): 0.5, ("rec0", "narrow"): 0.5}
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
    expect = {"revision": "a" * 40, "protocol": "policy", "reset_s": 20.0, "spread": 0.25, "seed": 12, "cases": str(cf)}
    assert valid(str(good), expect) == []
    for k, v in (("reset_s", None), ("reset_s", 0.0), ("spread", 1.0), ("seed", 13), ("protocol", "probe_once"), ("revision", "b" * 40)):
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
    fexp = {"revision": "a" * 40, "protocol": "probe_once", "reset_s": None, "spread": 1.0, "seed": 12, "n": 1}
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
        out = {k: v for k, v in r.items() if k not in ("restores", "reset_s", "spread")}
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
    for k, v in (("pos_err_mm", 1.5), ("rot_err_deg", 0.2), ("qvel_err", 1e-3), ("mocap_err", 1e-3), ("pos_err_mm", None),
                 ("same_case", False), ("episode", False), ("at_rest", None)):
        assert not restore_ok(dict(CHECK, **{k: v})), k
    assert not restore_ok({k: v for k, v in CHECK.items() if k != "qvel_err"})


def test_ga_compares_every_milestone4_key_but_the_wall_time():
    m4 = _row("a", 0, 1.0, seconds=3.2, bundle="/home/u/Codes/RPent/robots/mujoco_lab/data/m2b2/folds/fold_0")
    h3 = dict(copy.deepcopy(m4), seconds=5.9, bundle="/home/u/Codes/RPent-h3/robots/mujoco_lab/data/m2b2/folds/fold_0",
              reset_s=None, spread=1.0)
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
    assert set(out["secondary"]) >= {"x_m", "x_p", "q", "rec0_order", "matched_truth"}
    assert out["n_cohort"] == 12 and out["seeds"] == [12, 13]
    shares = out["decision_change"]["absorbing/wide"]
    assert shares["n_probed"] == 6 and shares["share"]["decline"] == pytest.approx(1 / 6) and shares["share"]["same"] == pytest.approx(5 / 6)
    json.dumps(out)
    assert "H3" in out["markdown"]
