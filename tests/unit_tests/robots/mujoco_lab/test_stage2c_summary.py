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

"""The Stage 2c summary (daily-logs designs/2026-11-09-stage2-contacts-rerun-design.md, readings 1 to 4): the case-level
change against the published records, the three-column tables, and the summary command on synthetic runs."""

from robots.mujoco_lab import curve as cv
from robots.mujoco_lab import stage2c as s2c
from tests.unit_tests.robots.mujoco_lab.test_curve_oracle import _s2, _sim

K = "R0|10|2"


def _tipped_then(second_row, label2=None):
    m2 = {"status": "ok", "rows": [], "decisions": {K: second_row}}
    sec = {second_row: {"label": label2, "artifact": False, "error": None}} if label2 else {}
    return _sim("commit_std", "tipped", menu2=m2, second=sec)


def _pub_new():
    pub = [_s2("a", 0, {K: "commit_std"}, [_sim("commit_std", "placed_stable")]),
           _s2("a", 1, {K: "commit_std"}, [_tipped_then("slow_lift", "placed_stable")]),
           _s2("b", 0, {K: cv.GIVE_UP_ROW}, []),
           _s2("b", 1, {K: "commit_std"}, [_sim("commit_std", "placed_stable")]),
           _s2("c", 0, {K: "commit_std"}, [_sim("commit_std", "placed_stable")])]
    new = [_s2("a", 0, {K: "commit_std"}, [_sim("commit_std", "crushed")]),
           _s2("a", 1, {K: "commit_std"}, [_tipped_then(cv.GIVE_UP_ROW)]),
           _s2("b", 0, {K: cv.GIVE_UP_ROW}, []),
           _s2("b", 1, {K: "commit_std"}, [_sim("commit_std", "placed_stable")], dropped=True)]
    return pub, new


def test_the_case_change_counts_final_labels_outcomes_and_decisions_on_the_cases_kept_in_both():
    pub, new = _pub_new()
    ch = s2c.case_change(pub, new, [K])[K]
    assert ch["cases"] == 3 and ch["changed"] == 2
    assert ch["by_transition"] == {"placed_stable -> crushed": 1, "placed_stable -> give_up": 1}
    assert ch["by_object"] == {"a": {"cases": 2, "changed": 2}, "b": {"cases": 1, "changed": 0}}
    assert ch["outcomes"] == {"success -> irr": 1, "success -> give_up": 1, "give_up -> give_up": 1}
    assert ch["first_agree"] == 3 and ch["second_both"] == 1 and ch["second_agree"] == 0
    assert ch["second_attempts"] == {"published": 1, "new": 0} and ch["give_ups"] == {"published": 1, "new": 2}
    assert ch["dropped_new_only"] == [["b", 1]] and ch["dropped_published_only"] == [] and ch["missing"] == [["c", 0]]


def test_a_changed_first_decision_is_counted_against_the_agreement():
    pub, new = _pub_new()
    new[2] = _s2("b", 0, {K: "commit_std"}, [_sim("commit_std", "placed_stable")])
    ch = s2c.case_change(pub, new, [K])[K]
    assert ch["first_agree"] == 2 and ch["outcomes"]["give_up -> success"] == 1


# ---------------------------------------------------------------- the summary command on synthetic runs

import gzip  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

import pytest  # noqa: E402

from robots.mujoco_lab import curve_oracle as co  # noqa: E402
from robots.mujoco_lab import estimation_check, identify_fit, model_lock  # noqa: E402
from tests.unit_tests.robots.mujoco_lab import test_curve as tc  # noqa: E402

MEGA = "hot3d__megaphone"
SIM_COMMIT = "e" * 40
NEW = ("new_0", "new_1")                                     # the Stage 2c shards
ROWS_A = [dict(tc._row("commit_std", 0.6, irr=0.0), cvar=tc._cv(0.1)), dict(tc._row("firm_grip", 0.95, irr=0.03), cvar=tc._cv(0.1))]
ROWS_B = [dict(tc._row("commit_std", 0.9, irr=0.001), cvar=tc._cv(0.1))]
# the corrected physics of the synthetic world (R0's lock is r 0.02: commit_std on both menus; R4 grasps firm_grip on
# ROWS_A): on the test side R4's first attempt on hope__t1 0 (with R0's) and hope__t2 1 and its second attempt on
# hope__t3 1 end crushed, and after R4's first attempt on hope__t1 1 the second menu has no rows (a give-up); on the
# validation side commit_std crushes ycb__v1 0 and ycb__v2 1, which moves R0's lock to r 0.1
OVER = {("hope__t1", 0, "commit_std"): "crushed", ("hope__t2", 1, "firm_grip"): "crushed",
        ("hope__t3", 1, "firm_grip", "slow_lift"): "crushed", ("hope__t1", 1, "firm_grip", None): "no_rows",
        ("ycb__v1", 0, "commit_std"): "crushed", ("ycb__v2", 1, "commit_std"): "crushed"}


def _rows2():
    return [dict(tc._row("commit_std", 0.9, irr=0.0), cvar=tc._cv(0.1)), dict(tc._row("slow_lift", 0.95, irr=0.01), cvar=tc._cv(0.1))]


def _override(rec, over, second):
    """The labels of ``over`` on a record: {(object, idx, first row): label} for a first attempt (a placed or irreversible
    label ends the episode), {(object, idx, first row, second row): label} for a second, {(object, idx, first row, None):
    "no_rows"} for a second menu without rows (``second``: rows -> the second decisions)."""
    for s in rec["sims"]:
        k = (rec["object"], int(rec["idx"]), s["first"])
        if k in over:
            s.update(label=over[k], menu2=None, second={})
        if over.get(k + (None,)) == "no_rows" and s["menu2"]:
            s.update(menu2={"status": "no_rows", "rows": [], "decisions": second([])}, second={})
        for row2, x in s["second"].items():
            x["label"] = over.get(k + (row2,), x["label"])
    return rec


def _stage2_fake(over):
    def run(job):
        out = tc._fake_stage2_object(job)
        for r in out["records"]:
            _override(r, over, lambda rows: cv.second_decisions(rows, tuple(job[2]["rules"])))
        return out

    return run


def _oracle_fake(over):
    """R0 at the knob on Stage 1's rows; commit_std places, any other first row tips and the second menu's rows place
    (tc._fake_stage2_object's labels), then the labels of ``over``."""
    def run(job):
        o, recs1, s = job
        knob, out = float(s["knob"]), []
        for rec1 in recs1:
            rows, skip = rec1["rows"], cv.stage2_skip(rec1)
            dec = {cv.dkey("R0", cv.A_MAIN, knob): cv.decide("R0", rows, knob, cv.A_MAIN)}
            first, sims = next(iter(dec.values())), []
            if not skip and first != cv.GIVE_UP_ROW:
                if first == "commit_std":
                    sims.append(_sim(first, "placed_stable"))
                else:
                    d2 = co.oracle_second(knob)(_rows2(), ("R0",))
                    sims.append(_sim(first, "tipped", menu2={"status": "ok", "rows": _rows2(), "decisions": d2},
                                     second={r: {"label": "placed_stable", "artifact": False, "error": None}
                                             for r in cv.first_rows(d2, ("R0",))}))
            rec = cv.stage2_record(rec1, dec, ("R0",), None if skip else {"sims": sims, "stop": None}, 1)
            out.append(_override(dict(rec, oracle=True, knob=knob, curve_decisions=co.curve_decisions(rows), rows=rows, truth={},
                                      sim_truth={}, seconds=1.0), over, lambda rows2: co.oracle_second(knob)(rows2, ("R0",))))
        return {"records": out, "stop": None}

    return run


def _write_json(path, x):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    json.dump(x, open(path, "w"))


def _new(T):
    return [T / d for d in NEW]


def _world(T, monkeypatch, over=OVER, rules=None):
    """Stage 1 (validation ycb__v0 to v3 and the two megaphone cases, test hope__t0 to t3 with t0 thin plastic), the
    published Stage 2 (lock and summary) and oracle runs (test and validation, summary), the new runs on the corrected
    labels ``over`` with the flags (Stage 2c in the shards NEW, with the ``rules`` of each when a tuple), the model lock
    and the skip list of the megaphone cases, and the gate files."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(identify_fit, "require_bundle_generator", lambda b: "v11")
    monkeypatch.setattr(co, "require_bundle_generator", lambda b: "v11")
    monkeypatch.setattr(model_lock, "check", lambda bundle, path: {})
    monkeypatch.setattr(estimation_check, "_revision", lambda: {"code_revision": "c0de", "dirty": False})
    s1 = [tc._s1(f"ycb__v{i}", j, rows=ROWS_A if j else ROWS_B) for i in range(4) for j in range(2)]
    s1 += [tc._s1(MEGA, j, rows=ROWS_B) for j in range(2)]
    s1 += [tc._s1(f"hope__t{i}", j, rows=ROWS_A if j else ROWS_B, side="test", thin=i == 0, **{"class": "crushable"})
           for i in range(4) for j in range(2)]
    tc._write_run(str(T / "s1"), s1)
    with gzip.open(T / "retry.json.gz", "wt") as f:
        json.dump({"rows": [{"attempts": [{"label": "tipped", "cost_s": 8.0}, {"label": "placed_stable", "cost_s": 8.0}]}]}, f)
    base = ["--stage1", str(T / "s1")]
    monkeypatch.setattr(cv, "stage2_object", _stage2_fake({}))
    cv.main(["stage2", *base, "--retry", str(T / "retry.json.gz"), "--out", str(T / "pub")])
    cv.main(["stage2-lock", "--runs", str(T / "pub"), "--out", str(T / "lock")])
    locked = str(T / "lock" / "locked.json")
    monkeypatch.setattr(co, "DESIGN_KNOB", json.load(open(locked))["lock"]["R0"]["knob"])
    cv.main(["stage2-summary", "--runs", str(T / "pub"), "--locked", locked, "--lock-commit", "abc", "--out", str(T / "pub_sum")])
    monkeypatch.setattr(co, "oracle_object", _oracle_fake({}))
    for side in ("test", "validation"):
        co.main(["run", *base, "--locked", locked, "--sides", side, "--out", str(T / f"pub_o_{side}")])
        (T / f"pub_o_{side}" / "run.exit").write_text("EXIT 0\n")
    co.main(["summary", "--runs", str(T / "pub_o_test"), "--stage2", str(T / "pub"), "--stage2-summary",
             str(T / "pub_sum" / "stage2_summary.json"), "--locked", locked, "--out", str(T / "pub_o_sum")])
    monkeypatch.setenv("RPENT_MUJOCO_SIM_DIR", "/sim")
    _write_json(T / "model_lock.json", {"bundle": "B", "files": {}})
    _write_json(T / "skip_list.json", {"reason": "the megaphone ruling", "cases": [f"{MEGA}:0", f"{MEGA}:1"]})
    flags = ["--replan-hold", "--plan-without-hand-boxes", "--model-lock", str(T / "model_lock.json"), "--skip-list",
             str(T / "skip_list.json")]
    monkeypatch.setattr(cv, "stage2_object", _stage2_fake(over))
    for k, (d, r) in enumerate(zip(NEW, rules if isinstance(rules, tuple) else (rules,) * len(NEW))):
        cv.main(["stage2", *base, "--retry", str(T / "retry.json.gz"), "--shard", str(k), "--shards", str(len(NEW)),
                 "--out", str(T / d), *flags] + (["--rules", r] if r else []))
    monkeypatch.setattr(co, "oracle_object", _oracle_fake(over))
    for side in ("test", "validation"):
        co.main(["run", *base, "--locked", locked, "--sides", side, "--out", str(T / f"new_o_{side}"), *flags])
    for d in (*NEW, "new_o_test", "new_o_validation"):
        (T / d / "run.exit").write_text("EXIT 0\n")
        (T / d / "provenance.txt").write_text(f"code c0de (detached run worktree), dirty files 0\n"
                                              f"sim /sim, last commit {SIM_COMMIT}, dirty files 0\nstart 2026-11-09 05:38:45\n")
    _write_json(T / "smoke_s2c.json", {"first_pairs": 4, "first_changed": 0, "second_triples": 2, "second_changed": 0,
                                       "second_presence_changed": 0, "errors": 0, "stops": [], "wandb": "w1"})
    _write_json(T / "smoke_o.json", {"determinism": {"compared": 3, "missing": 0, "mismatches": []}, "errors": 0, "stops": [],
                                     "wandb": "w2"})
    for g, pub in (("gp_s2", T / "pub"), ("gp_o", T / "pub_o_validation")):
        _write_json(T / g / "gate_published.json", {"pass": True, "compared": 4, "equal": 4, "differ": [], "missing": [],
                                                    "runs": str(T / "pub"), "published": str(pub)})
    sl = str(T / "skip_list.json")
    s2c.main(["gate-cases", "--runs", ",".join(str(d) for d in _new(T)), "--published", str(T / "pub"), "--skip-list", sl,
              "--out", str(T / "gc_s2")])
    s2c.main(["gate-cases", "--runs", f"{T / 'new_o_test'},{T / 'new_o_validation'}", "--published",
              f"{T / 'pub_o_test'},{T / 'pub_o_validation'}", "--skip-list", sl, "--out", str(T / "gc_o")])
    pub_q1 = json.load(open(T / "pub_sum" / "stage2_summary.json"))["q1"]["boot"]
    pub_o = json.load(open(T / "pub_o_sum" / "oracle_summary.json"))["readings"]["1_oracle_vs_r0"]["all"]
    shift = lambda b, k: dict(b, **{k: {"diff": b[k]["diff"] + 0.01, "lo": b[k]["lo"], "hi": b[k]["hi"] + 0.02}})  # noqa: E731
    gt1 = {"stage2_r0_r4": dict(shift(pub_q1, "irr"), cases=5, objects=3, excluded={"flagged": 3}),
           "oracle_r0": dict(shift(pub_o, "give_up"), cases=6, objects=3, excluded={"flagged": 2})}
    _write_json(T / "audit.json", {"published": {"stage2_r0_r4": pub_q1, "oracle_r0": pub_o}, "readings": {"3": {"gt1": gt1}}})
    return locked


def _args(T, out="sum", **kw):
    d = {"runs": ",".join(str(d) for d in _new(T)), "oracle-runs": f"{T / 'new_o_test'},{T / 'new_o_validation'}", "published": str(T / "pub"),
         "published-summary": str(T / "pub_sum" / "stage2_summary.json"),
         "published-oracle": f"{T / 'pub_o_test'},{T / 'pub_o_validation'}",
         "published-oracle-summary": str(T / "pub_o_sum" / "oracle_summary.json"), "locked": str(T / "lock" / "locked.json"),
         "audit": str(T / "audit.json"), "model-lock": str(T / "model_lock.json"), "skip-list": str(T / "skip_list.json"),
         "sim-commit": SIM_COMMIT, "smoke-stage2c": str(T / "smoke_s2c.json"), "smoke-oracle": str(T / "smoke_o.json"),
         "gate-published": f"{T / 'gp_s2' / 'gate_published.json'},{T / 'gp_o' / 'gate_published.json'}",
         "gate-cases-stage2c": str(T / "gc_s2" / "gate_cases.json"), "gate-cases-oracle": str(T / "gc_o" / "gate_cases.json"),
         "out": str(T / out)}
    d.update(kw)
    return ["summary"] + [x for k, v in d.items() for x in (f"--{k}", v)]


def _recs(dirs, side):
    return [json.loads(line) for d in dirs for line in open(cv.records_path(str(d), side))]


def test_summary_reading_1_puts_the_published_the_audit_and_the_corrected_r0_minus_r4_side_by_side(tmp_path, monkeypatch):
    T = tmp_path
    locked = _world(T, monkeypatch)
    rep = s2c.main(_args(T))
    lk = json.load(open(locked))["lock"]
    k0, k4 = lk["R0"]["key"], lk["R4"]["key"]
    test_c, test_p = _recs(_new(T), "test"), _recs([T / "pub"], "test")
    r1 = rep["readings"]["1"]
    assert r1["keys"] == [k0, k4]
    assert r1["corrected"]["boot"] == cv.paired_bootstrap(test_c, k0, k4)
    assert r1["corrected"]["no_thin"] == cv.paired_bootstrap([r for r in test_c if not r["thin"]], k0, k4)
    pairs = [(r["object"], cv.episode(r["decisions"], r["sims"], k0), cv.episode(r["decisions"], r["sims"], k4))
             for r in test_c if not r["dropped"]]
    assert r1["corrected"]["give_up"] == co.paired_outcomes(pairs)["give_up"]
    assert r1["corrected"]["rates"] == {"R0": co.paired_outcomes(pairs)["rate_a"], "R4": co.paired_outcomes(pairs)["rate_b"]}
    assert r1["corrected"]["reading"] == cv.q1_reading(r1["corrected"]["boot"])
    assert r1["corrected"]["reading_no_thin"] == cv.q1_reading(r1["corrected"]["no_thin"])
    pub = json.load(open(T / "pub_sum" / "stage2_summary.json"))
    assert r1["published"]["boot"] == pub["q1"]["boot"] and r1["published"]["no_thin"] == pub["q1_no_thin"]["boot"]
    assert r1["published"]["reading"] == pub["q1"]["reading"]
    pp = [(r["object"], cv.episode(r["decisions"], r["sims"], k0), cv.episode(r["decisions"], r["sims"], k4))
          for r in test_p if not r["dropped"]]
    assert r1["published"]["give_up"] == co.paired_outcomes(pp)["give_up"]
    audit = json.load(open(T / "audit.json"))["readings"]["3"]["gt1"]["stage2_r0_r4"]
    assert r1["audit"]["boot"] == audit and r1["audit"]["reading"] == cv.q1_reading(audit) and r1["audit"]["level"] == "gt1"
    assert r1["corrected"]["boot"]["irr"]["diff"] != r1["published"]["boot"]["irr"]["diff"]       # the corrected labels count
    assert r1["corrected"]["give_up"] != r1["published"]["give_up"]
    md = open(T / "sum" / "summary.md").read().splitlines()
    assert f"| irreversible R0 − R4 | {co._ci(pub['q1']['boot'], 'irr')} | {co._ci(audit, 'irr')} | " \
           f"{co._ci(r1['corrected']['boot'], 'irr')} |" in md
    assert f"| give-up R0 − R4 | {co._ci({'g': r1['published']['give_up']}, 'g')} | – | " \
           f"{co._ci({'g': r1['corrected']['give_up']}, 'g')} |" in md


def test_summary_repeats_the_published_stage2_readings_at_the_published_lock_on_the_new_records(tmp_path, monkeypatch):
    T = tmp_path
    locked = _world(T, monkeypatch)
    s2c.main(_args(T))
    lk = json.load(open(locked))["lock"]
    key = {r: v["key"] for r, v in lk.items() if isinstance(v, dict)}
    rep = json.load(open(T / "sum" / "stage2c" / "stage2_summary.json"))
    test_c, val_c = _recs(_new(T), "test"), _recs(_new(T), "validation")
    assert rep == dict(rep, **cv.stage2_readings(val_c, test_c, tuple(cv.RULES), key))
    assert {r: rep["lock"][r]["knob"] for r in cv.RULES} == {r: lk[r]["knob"] for r in cv.RULES}
    assert rep["lock"]["R0"]["point"] == cv.rule_point(val_c, key["R0"])
    assert rep["provenance"]["records_sha256"] == {f"{d} {side}": cv._sha(cv.records_path(str(d), side))
                                                   for d in _new(T) for side in cv.SIDES}
    assert rep["checks"]["skip_list_cases"] == 2 and rep["checks"]["missing_cases"] == 0
    for f in ("stage2_summary.md", "stage2_curves.png"):
        assert os.path.exists(T / "sum" / "stage2c" / f)
    assert open(T / "sum" / "stage2c" / "stage2_summary.md").read().startswith("# Stage 2c")


def test_summary_reading_2_runs_the_oracle_summary_on_the_new_arms_beside_the_published_oracle(tmp_path, monkeypatch):
    T = tmp_path
    _world(T, monkeypatch)
    rep = s2c.main(_args(T))
    oc = json.load(open(T / "sum" / "oracle_c" / "oracle_summary.json"))
    po = json.load(open(T / "pub_o_sum" / "oracle_summary.json"))
    assert oc["provenance"]["runs"] == str(T / "new_o_test")                  # the test runs only
    r2 = rep["readings"]["2"]
    for name in ("1_oracle_vs_r0", "2_oracle_vs_r4"):
        assert r2["corrected"][name] == oc["readings"][name] and r2["published"][name] == po["readings"][name]
    assert r2["corrected"]["3_reading"] == oc["readings"]["3_reading"] and r2["published"]["3_reading"] == po["readings"]["3_reading"]
    assert r2["audit"]["oracle_vs_r0"] == json.load(open(T / "audit.json"))["readings"]["3"]["gt1"]["oracle_r0"]
    assert r2["corrected"]["6_determinism"]["mismatches"] == []
    assert r2["model_curves_equal"] is True                    # the first menus at the truth do not see the physics
    r4 = rep["readings"]["4"]
    assert r4["oracle"]["corrected"] == oc["readings"]["5_calibration"] and r4["oracle"]["published"] == po["readings"]["5_calibration"]


def test_summary_reading_3_counts_the_case_level_change_per_arm_and_side(tmp_path, monkeypatch):
    T = tmp_path
    locked = _world(T, monkeypatch)
    rep = s2c.main(_args(T))
    lk = json.load(open(locked))["lock"]
    keys = [lk[r]["key"] for r in cv.RULES]
    r3 = rep["readings"]["3"]
    for side in cv.SIDES:
        assert r3["stage2c"][side] == s2c.case_change(_recs([T / "pub"], side), _recs(_new(T), side), keys)
        k0 = [lk["R0"]["key"]]
        assert r3["oracle_c"][side] == s2c.case_change(_recs([T / f"pub_o_{side}"], side), _recs([T / f"new_o_{side}"], side), k0)
    k4 = r3["stage2c"]["test"][lk["R4"]["key"]]
    assert k4["changed"] == 4 and k4["first_agree"] == k4["cases"] == 8          # the four R4 episodes of OVER on test
    assert k4["give_ups"] == {"published": 0, "new": 1} and k4["by_transition"]["placed_stable -> give_up"] == 1
    assert r3["stage2c"]["validation"][lk["R4"]["key"]]["missing"] == [[MEGA, 0], [MEGA, 1]]


def test_summary_reading_4_sets_the_stage2c_calibration_beside_the_published(tmp_path, monkeypatch):
    T = tmp_path
    _world(T, monkeypatch)
    rep = s2c.main(_args(T))
    pub = json.load(open(T / "pub_sum" / "stage2_summary.json"))
    new = json.load(open(T / "sum" / "stage2c" / "stage2_summary.json"))
    assert rep["readings"]["4"]["stage2c"] == {r: {"published": pub["calibration"][r], "corrected": new["calibration"][r]}
                                               for r in cv.RULES}


def test_summary_reading_5_and_the_provenance_list_the_gates_and_every_run(tmp_path, monkeypatch):
    T = tmp_path
    _world(T, monkeypatch)
    seen = _spy_records(T, monkeypatch)
    rep = s2c.main(_args(T))
    assert {os.path.dirname(p) for p in seen} == {str(T / d) for d in (*NEW, "new_o_test", "new_o_validation")}
    g = rep["readings"]["5"]
    assert g["smoke_stage2c"]["first_changed"] == 0 and g["smoke_oracle"]["determinism"]["mismatches"] == []
    assert [(x["pass"], x["arm"]) for x in g["gate_published"]] == [(True, "stage2"), (True, "oracle")]
    assert g["gate_cases"]["stage2c"]["skipped"] == [[MEGA, 0], [MEGA, 1]] and g["gate_cases"]["oracle_c"]["pass"] is True
    assert g["model_lock_sha256"] == cv._sha(str(T / "model_lock.json"))
    assert rep["r1_r6_ran"] is True and rep["sim_commit"] == SIM_COMMIT
    assert set(rep["cpu_hours"]) == {"stage2c", "oracle_c", "published_stage2", "published_oracle"}
    assert rep["provenance"]["code_revision"] == "c0de" and len(rep["provenance"]["wandb"]["stage2c"]) == len(NEW)
    assert os.path.exists(T / "sum" / "summary.json") and os.path.exists(T / "sum" / "summary.md")


def test_summary_re_locks_on_the_new_validation_records_as_an_exploratory_line(tmp_path, monkeypatch):
    T = tmp_path
    locked = _world(T, monkeypatch)
    rep = s2c.main(_args(T))
    val_c = _recs(_new(T), "validation")
    re = cv.lock(val_c, tuple(cv.RULES))
    x = rep["exploratory"]["relock"]
    assert x["knobs"] == {r: re[r]["knob"] for r in cv.RULES}
    pub = json.load(open(locked))["lock"]
    assert x["differs"] == sorted(r for r in cv.RULES if re[r]["knob"] != pub[r]["knob"]) and "R0" in x["differs"]
    if "R0" in x["differs"] or "R4" in x["differs"]:
        assert x["q1"]["boot"] == cv.paired_bootstrap(_recs(_new(T), "test"), re["R0"]["key"], re["R4"]["key"])
    else:
        assert x["q1"] is None


def test_summary_on_runs_of_r0_and_r4_only_says_r1_to_r6_did_not_run(tmp_path, monkeypatch):
    T = tmp_path
    _world(T, monkeypatch, rules="R0,R4")
    rep = s2c.main(_args(T))
    assert rep["r1_r6_ran"] is False and rep["rules"] == ["R0", "R4"]
    s = json.load(open(T / "sum" / "stage2c" / "stage2_summary.json"))
    assert s["vs_r0"] == {} and set(s["calibration"]) == {"R0", "R4"}
    assert set(rep["readings"]["3"]["stage2c"]["test"]) == {s["lock"]["R0"]["key"], s["lock"]["R4"]["key"]}
    assert set(rep["readings"]["4"]["stage2c"]) == {"R0", "R4"}


def _spy_records(T, monkeypatch):
    """The paths of the new runs' records files that are opened from now on (builtins.open)."""
    import builtins

    real, seen, new = builtins.open, [], {str(T / d) for d in (*NEW, "new_o_test", "new_o_validation")}

    def spy(path, *args, **kw):
        if os.path.basename(str(path)).startswith("records_") and os.path.dirname(str(path)) in new:
            seen.append(str(path))
        return real(path, *args, **kw)

    monkeypatch.setattr(builtins, "open", spy)
    return seen


def _break(T, what):
    """One input of the summary made wrong; returns the summary's arguments that change with it."""
    def edit(path, f):
        x = json.load(open(path))
        f(x)
        json.dump(x, open(path, "w"))

    n0, n1 = T / NEW[0], T / NEW[1]

    if what == "exit":
        (n0 / "run.exit").write_text("EXIT 1\n")
    elif what == "flag":
        edit(n1 / "run_config.json", lambda c: c.update(replan_hold=False))
    elif what == "oracle_flag":
        edit(T / "new_o_validation" / "run_config.json", lambda c: c.update(plan_without_hand_boxes=False))
    elif what == "model_lock":
        edit(T / "new_o_test" / "run_config.json", lambda c: c.update(model_lock_sha256="0" * 64))
    elif what == "sim":
        (n0 / "provenance.txt").write_text(f"code c0de, dirty files 0\nsim /sim, last commit {'f' * 40}, dirty files 0\n")
    elif what == "sim_dirty":
        (T / "new_o_test" / "provenance.txt").write_text(f"code c0de, dirty files 0\nsim /sim, last commit {SIM_COMMIT}, dirty files 2\n")
    elif what == "dirty":
        edit(n1 / "run_config.json", lambda c: c.update(dirty=True))
    elif what == "seed":
        edit(n0 / "run_config.json", lambda c: c.update(seed=12))
    elif what == "smoke":
        edit(T / "smoke_s2c.json", lambda c: c.update(first_changed=1))
    elif what == "smoke_oracle":
        edit(T / "smoke_o.json", lambda c: c["determinism"]["mismatches"].append({"object": "x"}))
    elif what == "gate_published":
        edit(T / "gp_o" / "gate_published.json", lambda c: c.update(**{"pass": False}))
    elif what == "gate_cases":
        edit(T / "gc_s2" / "gate_cases.json", lambda c: c.update(runs=str(T / "other")))
    elif what == "published_records":
        with open(cv.records_path(str(T / "pub"), "test"), "a") as f:
            f.write("\n")
    elif what == "published_oracle_records":
        with open(cv.records_path(str(T / "pub_o_test"), "test"), "a") as f:
            f.write("\n")
    elif what == "locked":
        x = json.load(open(T / "lock" / "locked.json"))
        json.dump(x, open(T / "lock" / "locked.json", "w"), indent=2)                 # the same lock, another file
    elif what == "audit":
        edit(T / "audit.json", lambda c: c["published"]["stage2_r0_r4"]["irr"].update(diff=0.5))
    elif what == "audit_cases":
        edit(T / "audit.json", lambda c: c["published"]["stage2_r0_r4"].update(cases=999))
    elif what == "audit_oracle_rates":
        edit(T / "audit.json", lambda c: c["published"]["oracle_r0"]["rate_a"].update(n=999))
    elif what == "audit_keys":
        edit(T / "audit.json", lambda c: c["readings"]["3"]["gt1"]["stage2_r0_r4"].update(a="R1|10|2"))
    elif what == "sim_dir":                                                     # every run agrees, none with its sim
        for d in (n0, n1, T / "new_o_test", T / "new_o_validation"):
            edit(d / "run_config.json", lambda c: c.update(sim_dir="/old_sim"))
    elif what == "gate_published_arm":
        return {"gate-published": str(T / "gp_s2" / "gate_published.json")}
    elif what == "gate_published_other":
        edit(T / "gp_o" / "gate_published.json", lambda c: c.update(published=str(T / "elsewhere")))
    elif what == "smoke_one_seed":
        edit(T / "smoke_s2c.json", lambda c: c.update(first_pairs=0, second_triples=0))
    elif what == "smoke_oracle_missing":
        edit(T / "smoke_o.json", lambda c: c["determinism"].update(compared=0, missing=3))
    elif what == "rules":
        edit(n1 / "run_config.json", lambda c: c.update(rules=["R0", "R4"]))
    elif what == "gate_cases_published":
        edit(T / "gc_o" / "gate_cases.json", lambda c: c.update(published=str(T / "pub_o_test")))
    elif what == "oracle_sides":
        edit(T / "new_o_validation" / "run_config.json", lambda c: c.update(sides="test"))
    return {}


@pytest.mark.parametrize("what, match", [("exit", "EXIT 0"), ("flag", "replan_hold"), ("oracle_flag", "plan_without_hand_boxes"),
                                         ("model_lock", "model lock"), ("sim", "sim commit"), ("sim_dirty", "sim"),
                                         ("dirty", "dirty"), ("seed", "seed"), ("smoke", "smoke"), ("smoke_oracle", "smoke"),
                                         ("gate_published", "gate"), ("gate_cases", "gate"),
                                         ("published_records", "published"), ("published_oracle_records", "published oracle"),
                                         ("locked", "lock"), ("audit", "audit"), ("audit_cases", "audit"),
                                         ("audit_oracle_rates", "audit"), ("audit_keys", "audit"), ("sim_dir", "sim_dir"),
                                         ("gate_published_arm", "gate"), ("gate_published_other", "gate"),
                                         ("smoke_one_seed", "smoke"), ("smoke_oracle_missing", "smoke"), ("rules", "rules"),
                                         ("gate_cases_published", "gate"), ("oracle_sides", "side")])
def test_summary_refuses_a_failed_gate_or_an_input_that_changed_before_it_opens_a_new_record(tmp_path, monkeypatch, what, match):
    T = tmp_path
    _world(T, monkeypatch)
    kw = _break(T, what)
    seen = _spy_records(T, monkeypatch)
    with pytest.raises(SystemExit, match=match):
        s2c.main(_args(T, **kw))
    assert seen == [] and not os.path.exists(T / "sum" / "summary.json")


def test_summary_refuses_a_skip_list_that_is_not_the_runs(tmp_path, monkeypatch):
    T = tmp_path
    _world(T, monkeypatch)
    _write_json(T / "other_skip.json", {"reason": "another", "cases": [f"{MEGA}:0"]})
    with pytest.raises(SystemExit, match="skip list"):
        s2c.main(_args(T, **{"skip-list": str(T / "other_skip.json")}))


def test_summary_refuses_an_out_that_is_not_empty(tmp_path, monkeypatch):
    T = tmp_path
    _world(T, monkeypatch)
    (T / "sum").mkdir()
    (T / "sum" / "x").write_text("")
    with pytest.raises(SystemExit, match="empty"):
        s2c.main(_args(T))
