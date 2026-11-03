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


"""The milestone 4 sweep analysis (robots/mujoco_lab/sweep.py) on synthetic calibrate outputs."""

import gzip
import json

import numpy as np
import pytest

from robots.mujoco_lab.sweep import (CHEAP, TEST_CELLS, arm_metrics, bootstrap_weights, cell, cell_draws, check_identical, check_same,
                                     declared_saving, h1_only, h1_pays, load_dir, load_run, object_sums, pairs, probe_cases, probe_risk,
                                     seeds_needed, subgroup, summary, valid, variance_split, verdict)

OK = [{"option": "commit_std", "outcome": "placed_stable"}]
CODE = {"revision": "a" * 40, "dirty": False, "sim_revision": "s" * 40, "sim_dirty": False, "changed_during_run": False}


def _case(obj, idx, cost, rec="commit", probes=(), ended="placed", first_menu=None, artifact=False, theta=None, fill=0.0,
          attempts=None, error=None):
    row = {"object": obj, "idx": idx, "cost": cost, "recommendation": rec, "ended": ended, "artifact": artifact,
           "probes": [dict(p) for p in probes], "attempts": list(OK if attempts is None else attempts),
           "first_menu": first_menu or {"best_cost": 0.6, "recommended_cost": None}, "theta": theta or {"mass_scale": 1.0}, "fill": fill}
    if error:
        row["error"] = error
    return row


def _run(protocol, care, A, s, seed, rows, force_noise=0.05, code=None):
    return {"protocol": protocol, "care_scale": float(care), "abstain_cost": float(A), "probe_time_scale": float(s), "seed": seed,
            "force_noise": force_noise, "n": len(rows), "rows": rows, "code": code or dict(CODE), "cases": None}


def _probe(probe, thrown=False, kind="measured", position=0, p_thrown=0.1, damaged=0.0, evoi=0.2):
    return {"probe": probe, "position": position, "after_attempts": 0, "kind": kind, "thrown": thrown, "damage": None, "artifact": False,
            "declared": {"p_kind": {"measured": 1.0 - damaged, "damaged": damaged}, "p_thrown": p_thrown, "evoi": evoi},
            "kinds_not_thrown": True}


def test_probe_cases_lists_the_cases_with_a_probe_recommendation():
    p1 = _run("policy", 10, 3, 1.0, 13, [_case("a", 0, 1.0, rec="probe"), _case("a", 1, 1.0), _case("b", 0, 1.0, rec="probe", artifact=True)])
    p2 = _run("policy", 10, 3, 0.0, 13, [_case("a", 1, 1.0, rec="probe"), _case("c", 2, 1.0, rec="probe")])
    assert probe_cases([p1, p2]) == {"seed": 13, "cases": {"a": [0, 1], "b": [0], "c": [2]}}
    with pytest.raises(ValueError, match="seed"):
        probe_cases([p1, _run("policy", 10, 3, 1.0, 14, [])])


def test_check_identical_finds_a_changed_case_without_a_probe():
    pol = _run("policy", 10, 3, 0.25, 12, [_case("a", 0, 1.0), _case("a", 1, 2.0, rec="probe"), _case("b", 0, 3.0)])
    nev = _run("policy_no_probe", 10, 3, 1.0, 12, [_case("a", 0, 1.0), _case("a", 1, 9.0), _case("b", 0, 3.5)])
    assert check_identical(pol, nev) == [("b", 0)]
    nev["rows"] = nev["rows"][:2]
    assert check_identical(pol, nev) == [("b", 0)]                 # a missing row is a mismatch too


def test_check_same_finds_any_changed_case():
    a = _run("policy_no_probe", 10, 3, 1.0, 12, [_case("a", 0, 1.0), _case("a", 1, 2.0)])
    b = _run("policy_no_probe", 10, 3, 0.0, 12, [_case("a", 0, 1.0), _case("a", 1, 2.5)])
    assert check_same(a, a) == [] and check_same(a, b) == [("a", 1)]


def _pair_runs():
    rows_p = [_case("a", 0, 1.0, rec="probe", first_menu={"best_cost": 2.0, "recommended_cost": 1.5}), _case("a", 1, 2.0),
              _case("b", 0, 0.5), _case("b", 1, 0.5)]
    rows_n = [_case("a", 0, 3.0), _case("a", 1, 2.0), _case("b", 0, 0.5), _case("b", 1, 0.5)]
    return _run("policy", 10, 3, 1.0, 12, rows_p), _run("policy_no_probe", 10, 3, 1.0, 12, rows_n)


def test_saving_is_zero_without_a_probe_and_the_difference_with_one():
    P, N = _pair_runs()
    c = cell_draws(object_sums(pairs(P, N), ["a", "b"]), bootstrap_weights(["a", "b"], 200, np.random.default_rng(0)))
    assert c["delta"] == pytest.approx(0.5) and c["J"] == pytest.approx(6.0 / 4) and c["declared"] == pytest.approx(0.125)
    assert c["rel"] == pytest.approx(0.5 / 1.5) and c["n"] == 4


def test_a_subset_never_run_gives_the_same_saving_as_a_full_one():
    P, N = _pair_runs()
    sub = _run("policy_no_probe", 10, 3, 1.0, 12, [N["rows"][0]])
    np.testing.assert_allclose(object_sums(pairs(P, N), ["a", "b"]), object_sums(pairs(P, sub, subset=True), ["a", "b"]))


def test_saving_drops_pairs_uncounted_in_either_arm():
    P = _run("policy", 10, 3, 1.0, 12, [_case("a", 0, 1.0, rec="probe"), _case("a", 1, 2.0, artifact=True), _case("b", 0, 0.5),
                                        _case("b", 1, 0.5, error="boom")])
    N = _run("policy_no_probe", 10, 3, 1.0, 12, [_case("a", 0, 3.0, artifact=True), _case("a", 1, 2.0), _case("b", 0, 0.5),
                                                 _case("b", 1, 0.5)])
    p = pairs(P, N)
    assert p["n"] == 1 and p["dropped"] == 3 and int(p["no_error"].sum()) == 3


def test_saving_raises_when_a_probe_case_has_no_never_row():
    P, _ = _pair_runs()
    with pytest.raises(ValueError, match="no row"):
        pairs(P, _run("policy_no_probe", 10, 3, 1.0, 12, []), subset=True)
    with pytest.raises(ValueError, match="no row"):
        pairs(P, _run("policy_no_probe", 10, 3, 1.0, 12, []))


def test_saving_refuses_runs_with_different_costs():
    P = _run("policy", 10, 3, 0.25, 12, [_case("a", 0, 1.0)])
    for bad in (_run("policy_no_probe", 30, 3, 1.0, 12, [_case("a", 0, 1.0)]), _run("policy_no_probe", 10, 10, 1.0, 12, [_case("a", 0, 1.0)]),
                _run("policy_no_probe", 10, 3, 1.0, 12, [_case("a", 0, 1.0)], force_noise=0.0)):
        with pytest.raises(ValueError, match="different costs"):
            pairs(P, bad)
    with pytest.raises(ValueError, match="ran a probe"):
        pairs(P, _run("policy_no_probe", 10, 3, 1.0, 12, [_case("a", 0, 1.0, probes=[_probe("push")])]))
    with pytest.raises(ValueError, match="seed"):
        pairs(P, _run("policy_no_probe", 10, 3, 1.0, 13, [_case("a", 0, 1.0)]))
    assert pairs(P, _run("policy_no_probe", 10, 3, 1.0, 12, [_case("a", 0, 1.0)]))["n"] == 1


def test_declared_saving_is_min_best_and_A_minus_the_probe_cost():
    assert declared_saving({"best_cost": 7.3, "recommended_cost": 6.7}, 10.0) == pytest.approx(0.6)
    assert declared_saving({"best_cost": 12.0, "recommended_cost": 9.0}, 10.0) == pytest.approx(1.0)
    assert declared_saving({"best_cost": 12.0, "recommended_cost": None}, 10.0) == 0.0
    assert declared_saving(None, 10.0) == 0.0


def test_ratios_are_recomputed_in_each_draw():
    W = bootstrap_weights(["a", "b"], 500, np.random.default_rng(0))
    S1 = np.array([[0.1, 0.0, 1.0, 1.0], [0.1, 0.0, 1.0, 1.0]])
    S2 = np.array([[0.3, 0.0, 1.0, 1.0], [0.3, 0.0, 1.0, 1.0]])
    h = h1_pays([cell_draws(S1, W), cell_draws(S2, W)])
    assert h["R"] == pytest.approx(0.2) and h["ci"][0] <= 0.2 + 1e-12 and h["ci"][1] >= 0.2 - 1e-12 and h["verdict"] == "pass"
    z = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    h = h1_pays([cell_draws(z, W)])
    assert h["R"] == 0.0 and h["ci"] == [0.0, 0.0] and h["verdict"] == "undecided"
    c = cell_draws(np.array([[1.0, 0.0, 2.0, 1.0], [0.0, 0.0, 8.0, 1.0]]), np.array([[2.0, 0.0]]))
    assert c["draws"]["rel"][0] == pytest.approx(0.5)              # object a alone: 1/2, not the pooled 1/10


def test_h1_only_needs_all_three_conditions_in_every_cheap_cell():
    good = {"probe_rate": 0.01, "policy": {"rel": 0.0, "rel_ci": [-0.01, 0.01]}, "forced": {"rel": -0.2, "rel_ci": [-0.3, -0.1]}}
    cells = {k: dict(good) for k in CHEAP}
    assert h1_only(cells)["verdict"] == "pass"
    assert h1_only({**cells, CHEAP[0]: dict(good, probe_rate=0.05)})["verdict"] == "undecided"
    assert h1_only({**cells, CHEAP[1]: dict(good, forced={"rel": 0.01, "rel_ci": [-0.01, 0.03]})})["verdict"] == "undecided"
    assert h1_only({**cells, CHEAP[2]: dict(good, forced={"rel": 0.05, "rel_ci": [0.03, 0.07]})})["verdict"] == "fail"
    assert h1_only({**cells, CHEAP[2]: dict(good, policy={"rel": 0.05, "rel_ci": [0.03, 0.07]})})["verdict"] == "fail"
    assert h1_only({CHEAP[0]: good})["verdict"] == "undecided"      # a cheap cell without its runs


def test_probe_risk_flags_too_many_events():
    def rows(events):
        return [_case(f"o{i}", 0, 1.0, rec="probe", probes=[_probe("weigh", p_thrown=0.01, thrown=i < events)]) for i in range(31)]

    r = probe_risk([_run("policy", 1, 1, 1.0, 12, rows(3))])["first weigh"]
    assert r["n"] == 31 and r["declared"] == pytest.approx(0.31) and r["events"] == 3 and r["verdict"] == "fail"
    assert probe_risk([_run("policy", 1, 1, 1.0, 12, rows(1))])["first weigh"]["verdict"] == "pass"
    twice = probe_risk([_run("policy", 1, 1, 1.0, 12, rows(1)), _run("policy", 10, 1, 1.0, 12, rows(1))])["first weigh"]
    assert twice["n"] == 31 and twice["events"] == 1                # one per (seed, object, idx, position, probe)
    second = [_case("x", 0, 1.0, rec="probe", probes=[_probe("push"), _probe("weigh", position=1, thrown=True)])]
    r = probe_risk([_run("policy", 1, 1, 1.0, 12, second)])
    assert r["second weigh"]["events"] == 1 and r["first push"]["events"] == 0


def test_verdicts_from_intervals():
    assert verdict([0.1, 0.3]) == "pass" and verdict([-0.3, -0.1]) == "fail" and verdict([-0.1, 0.1]) == "undecided"
    assert verdict([0.03, 0.05], 0.02) == "pass" and verdict([-0.1, 0.01], 0.02) == "fail" and verdict([0.01, 0.03], 0.02) == "undecided"


def test_power_picks_the_smallest_k_that_reaches_80_percent():
    assert seeds_needed(0.1, 0.0, 8.0) == 6 and seeds_needed(0.1, 0.0, 5.0) == 4 and seeds_needed(0.1, 0.0, 20.0) == 8
    assert seeds_needed(-0.1, 0.0, 1.0) == 8 and seeds_needed(0.1, 1.0, 0.0) == 8
    sb, sw = variance_split(np.array([1.0, 3.0, 5.0, 7.0]), np.array(["a", "a", "b", "b"]))
    assert sw == pytest.approx(2.0) and sb == pytest.approx(7.0)


def test_valid_rejects_a_truncated_or_foreign_output(tmp_path):
    run = _run("policy", 10.0, 3.0, 0.25, 12, [_case("a", 0, 1.0), _case("a", 1, 1.0)])
    expect = {"revision": "a" * 40, "sim_revision": "s" * 40, "protocol": "policy", "care": 10.0, "abstain": 3.0, "scale": 0.25, "seed": 12, "n": 2}
    good = tmp_path / "good.json"
    good.write_text(json.dumps(run))
    assert valid(str(good), expect) == []
    cut = tmp_path / "cut.json"
    cut.write_text(json.dumps(run)[:50])
    assert valid(str(cut), expect) and valid(str(tmp_path / "missing.json"), expect)
    for k, v in (("revision", "b" * 40), ("scale", 1.0), ("n", 3), ("protocol", "commit"), ("seed", 13)):
        assert valid(str(good), dict(expect, **{k: v})), k
    dirty = tmp_path / "dirty.json"
    dirty.write_text(json.dumps(dict(run, code=dict(CODE, dirty=True))))
    assert valid(str(dirty), expect)


def test_load_run_reads_gzip(tmp_path):
    run = _run("policy", 10, 3, 1.0, 12, [_case("a", 0, 1.0)])
    with gzip.open(tmp_path / "r.json.gz", "wt") as f:
        json.dump(run, f)
    assert load_run(str(tmp_path / "r.json.gz")) == run
    (tmp_path / "s.json").write_text(json.dumps(run))
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "cases.json").write_text(json.dumps({"seed": 12, "cases": {}}))
    assert len(load_dir(str(tmp_path))) == 2                        # the cases file and the text file are not runs


def test_subgroup_classes_and_prior_draw_terciles():
    classes = {"cup": {"class": "breakable", "fillable": True}, "glass": {"class": "breakable", "fillable": False},
               "egg": {"class": "crushable", "fillable": False}, "box": {"class": "none", "fillable": False},
               "peach": {"class": "bruisable", "fillable": False}}

    def c(obj, ms, fill=0.0):
        return {"object": obj, "theta": {"mass_scale": ms}, "fill": fill}

    assert subgroup(c("cup", 0.85, fill=0.5), classes) == ("container", "low")       # a filled container: [0.8, 1.25]
    assert subgroup(c("cup", 0.8), classes) == ("container", "low")                  # empty: [0.5, 3]
    assert subgroup(c("glass", 1.2), classes) == ("breakable", "mid") and subgroup(c("egg", 2.0), classes) == ("crushable", "high")
    assert subgroup(c("box", 1.0), classes)[0] == "rigid" and subgroup(c("peach", 1.0), classes)[0] == "bruisable"
    assert subgroup(c("unknown", 1.0), classes)[0] == "rigid"


def test_arm_metrics_counts_rates_per_1000():
    rows = [_case("a", 0, 1.0, rec="probe", probes=[_probe("push", evoi=0.2)]), _case("a", 1, 3.0, ended="abstained", attempts=[]),
            _case("b", 0, 9.0, ended="failed", attempts=[{"option": "commit_std", "outcome": "dropped_high"}]),
            _case("b", 1, 1.0, artifact=True)]
    m = arm_metrics(_run("policy", 10, 3, 1.0, 12, rows), bootstrap_weights(["a", "b"], 100, np.random.default_rng(0)), ["a", "b"])
    assert m["n"] == 3 and m["probe_rate"] == pytest.approx(1 / 3) and m["abstain_rate"] == pytest.approx(1 / 3)
    assert m["placed_per_1000"] == pytest.approx(1000 / 3) and m["irreversible_per_1000"] == pytest.approx(1000 / 3)
    assert m["cost"] == pytest.approx(13 / 3) and m["median_evoi"] == pytest.approx(0.2)


def _sweep(seeds=(12,)):
    """Every run the summary reads, on two objects with two cases each: the policy probes case (a, 0) when κ ≥ 10 and saves 1."""
    rng = np.random.default_rng(0)
    runs = []
    for seed in seeds:
        for care in (0.5, 1, 3, 10, 30):
            for A in (1, 3, 10):
                base = [(o, i, float(rng.uniform(0.5, 1.5))) for o in ("a", "b") for i in range(2)]
                never = [_case(o, i, c) for o, i, c in base]
                for s in (1.0, 0.25, 0.0):
                    pol = [_case(o, i, c - (1.0 if (o, i) == ("a", 0) and care >= 10 else 0.0),
                                 rec="probe" if (o, i) == ("a", 0) and care >= 10 else "commit",
                                 first_menu={"best_cost": 2.0, "recommended_cost": 1.2}) for o, i, c in base]
                    if seed == 12 or (care >= 10 and A >= 3):
                        runs.append(_run("policy", care, A, s, seed, pol))
                if seed == 12:
                    runs += [_run(p, care, A, 1.0, seed, never) for p in ("policy_no_probe", "commit", "repair")]
                    runs.append(_run("policy_probe_one", care, A, 1.0, seed, [_case(o, i, c + 0.2) for o, i, c in base]))
                elif care >= 10 and A >= 3:
                    runs.append(_run("policy_no_probe", care, A, 1.0, seed, [n for n in never if (n["object"], n["idx"]) == ("a", 0)]))
    return runs


def test_summary_runs_on_a_small_sweep():
    out = summary(_sweep(seeds=(12, 13)), classes={}, n_boot=50)
    assert out["h1_pays"]["verdict"] in ("pass", "fail", "undecided") and len(out["h1_pays"]["cells"]) == len(TEST_CELLS)
    assert out["h1_pays"]["R"] > 0 and out["h1_only"]["verdict"] in ("pass", "fail", "undecided")
    assert len(out["cells"]) == 45 and out["probe_value"]["sum_delta"] > 0
    json.dumps(out)
    assert "H1-pays" in out["markdown"]


def test_the_summary_leaves_an_error_row_out_of_the_subgroups():
    """An error row carries no draw (calibrate writes object, idx, error, executed, fold, bundle, const and seconds): the
    summary labels only the rows with a draw, so it runs, and the error row adds to no subgroup, as any case that is not
    counted (milestone 4: three carton cases ended in error rows and the summary raised KeyError 'theta')."""
    def with_row(make):
        runs = _sweep(seeds=(12, 13))
        run = next(r for r in runs if r["protocol"] == "policy" and r["seed"] == 13
                   and (r["care_scale"], r["abstain_cost"], r["probe_time_scale"]) == TEST_CELLS[0])
        run["rows"] = [make(r) if (r["object"], r["idx"]) == ("a", 0) else r for r in run["rows"]]
        return summary(runs, classes={}, n_boot=50)

    err = with_row(lambda r: {"object": "a", "idx": 0, "error": "max() arg is an empty sequence", "executed": False, "fold": 0,
                              "bundle": "b", "const": [0.5], "seconds": 1.0})
    art = with_row(lambda r: dict(r, artifact=True))
    assert err["subgroups"] == art["subgroups"] and err["subgroups"]["class"]
    assert sum(g["errors"] for e in err["exclusions"].values() for g in e.values()) == 1


def test_valid_rejects_another_simulator_or_a_change_during_the_run(tmp_path):
    """Final review (Opus 1): the simulator lives outside RPent, so an output is reusable only when it records the expected
    simulator revision, no uncommitted simulator change, and the same code and simulator at its end as at its start."""
    expect = {"revision": "a" * 40, "sim_revision": "s" * 40, "protocol": "policy", "care": 10.0, "abstain": 3.0, "scale": 1.0, "seed": 12,
              "n": 1}

    def why(**code):
        f = tmp_path / "r.json"
        f.write_text(json.dumps(_run("policy", 10.0, 3.0, 1.0, 12, [_case("a", 0, 1.0)], code={k: v for k, v in dict(CODE, **code).items()
                                                                                                 if v is not None})))
        return valid(str(f), expect)

    assert why() == []
    assert why(sim_revision="t" * 40) and why(sim_dirty=True) and why(changed_during_run=True)
    assert why(sim_revision=None) and why(sim_dirty=None) and why(changed_during_run=None)        # an output from before the fix


def test_valid_checks_the_cases_of_a_subset_run(tmp_path):
    """Final review (Codex 1): `sweep valid --cases` accepts a subset run only for the cases its cases file lists; a stale
    run of the same size over other cases is rerun, not skipped."""
    from robots.mujoco_lab.sweep import main as sweep_main

    out, cases = tmp_path / "never.json", tmp_path / "cases.json"
    out.write_text(json.dumps(_run("policy_no_probe", 10.0, 3.0, 1.0, 13, [_case("a", 0, 1.0), _case("b", 1, 1.0)])))
    args = ["valid", str(out), "--revision", "a" * 40, "--sim-revision", "s" * 40, "--protocol", "policy_no_probe", "--care", "10",
            "--abstain", "3", "--scale", "1", "--seed", "13", "--cases", str(cases)]
    cases.write_text(json.dumps({"seed": 13, "cases": {"a": [0], "b": [1]}}))
    assert sweep_main(args) == 0
    cases.write_text(json.dumps({"seed": 13, "cases": {"a": [0], "b": [2]}}))
    assert sweep_main(args) == 1


def test_check_stops_when_a_gate_run_is_missing(tmp_path, capsys):
    """Final review (Codex 3): the stage-1 gates need the stage-1 grid at the seed: the policy in the 45 cells, the
    never-probe arm at s = 1 for the 15 (κ, A), and at s 0.25 and 0 for the 4 test (κ, A) (G1b). A missing run stops the
    stage, and an empty directory does not pass."""
    from robots.mujoco_lab.sweep import main as sweep_main

    runs = _sweep(seeds=(12,))
    for k, A in sorted({c[:2] for c in TEST_CELLS}):
        nv = next(r for r in runs if r["protocol"] == "policy_no_probe" and (r["care_scale"], r["abstain_cost"]) == (k, A))
        runs += [dict(nv, probe_time_scale=s) for s in (0.25, 0.0)]
    d = tmp_path / "out"
    d.mkdir()
    for i, r in enumerate(runs):
        (d / f"r{i}.json").write_text(json.dumps(r))
    assert sweep_main(["check", "--dir", str(d), "--seed", "12"]) == 0
    capsys.readouterr()
    i = next(i for i, r in enumerate(runs) if r["protocol"] == "policy_no_probe" and r["probe_time_scale"] == 0.0)
    (d / f"r{i}.json").unlink()
    assert sweep_main(["check", "--dir", str(d), "--seed", "12"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"] == "stop" and f"policy_no_probe k{runs[i]['care_scale']:g}_A{runs[i]['abstain_cost']:g}_s0" in out["missing"]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert sweep_main(["check", "--dir", str(empty), "--seed", "12"]) == 1


def test_summary_is_incomplete_when_a_seed_of_a_test_cell_is_missing():
    """Final review (Codex 2): H1-pays pools the seeds of every test cell; a missing policy or never-probe run of one seed,
    or a seed short of the power result's k, makes the verdict incomplete and names the runs, instead of a verdict over
    fewer seeds."""
    runs = [r for r in _sweep(seeds=(12, 13)) if not (r["protocol"] == "policy" and r["seed"] == 13 and cell(r) == TEST_CELLS[0])]
    out = summary(runs, classes={}, n_boot=50)
    assert out["h1_pays"]["verdict"] == "incomplete" and out["h1_pays"]["missing"] == ["policy k10_A3_s1 seed 13"]
    full = summary(_sweep(seeds=(12, 13)), classes={}, n_boot=50)
    assert full["h1_pays"]["missing"] == [] and full["h1_pays"]["verdict"] in ("pass", "fail", "undecided")
    pw = {"R": 0.1, "R_declared": 0.1, "sigma_b2": 0.0, "sigma_w2": 1.0, "se": {}, "k": 4}
    short = summary(_sweep(seeds=(12, 13)), classes={}, n_boot=50, power_result=pw)
    assert short["h1_pays"]["verdict"] == "incomplete" and len(short["h1_pays"]["missing"]) == 3 * (12 + 4)     # seeds 14 to 16
    assert "policy_no_probe k10_A3_s1 seed 16" in short["h1_pays"]["missing"]


def test_the_exclusions_count_a_row_without_a_recommendation_as_unknown():
    """Final review (Opus 2): a row that carries no first recommendation (a case that raised before its first menu) is not
    a case without a probe recommendation; the exclusions count it as unknown (milestone 4 counted its 27 carton errors,
    all probe cases, as other)."""
    runs = _sweep(seeds=(12,))
    run = next(r for r in runs if r["protocol"] == "policy" and cell(r) == TEST_CELLS[0])
    run["rows"][1] = {"object": run["rows"][1]["object"], "idx": run["rows"][1]["idx"], "error": "boom", "executed": False}
    e = summary(runs, classes={}, n_boot=50)["exclusions"]["policy_k10_A3_s1_seed12"]
    assert e["unknown"] == {"errors": 1, "artifacts": 0} and e["other"] == {"errors": 0, "artifacts": 0}
