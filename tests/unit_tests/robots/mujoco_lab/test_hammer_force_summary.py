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

"""The hammer squeeze x CoM-pinch summary (daily-logs design 2026-11-09-hammer-force-com-design.md and the coordinator's
rulings of 9 Nov), tested on synthetic records only."""

import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from robots.mujoco_lab import curve_hammer_force as hf
from robots.mujoco_lab import hammer_force_summary as hs

FORCE = {"f_std": 10.0, "2f_std": 20.0, "40N": 40.0, "70N": 70.0}


def _row(pinch, level, label="not_lifted", feasible=True, deep=1.0, rev=0, at_rest=True, jam=False, error=None, damage=None,
         hold=None, close=None, qpos="q", trace="t"):
    r = {"id": hf.row_id(pinch, level), "pinch": pinch, "level": level, "force": FORCE[level], "feasible": feasible,
         "why": None if feasible else "unreachable: collision (approach)"}
    if feasible:
        r.update(label=None if error else label, artifact=rev >= 5 or deep > 3.0 or at_rest is False or jam, error=error, damage=damage,
                 seconds=1.0, qpos_sha=qpos, trace_sha=trace,
                 monitor={"rev_steps": rev, "deepest_mm": deep, "at_rest": at_rest, "jam": jam, "jam_artifact": jam},
                 squeeze={"hold_steps": 50, "hold_contact": 50 if hold else 0, "hold": hold, "close": FORCE[level] if close is None else close})
    return r


def _case(obj="handal__hammer_1", idx=0, side="test", mass=0.5, labels=None, rows=None, **kw):
    """One simulated case: every row not_lifted unless ``labels`` (cell -> label) or ``rows`` (cell -> _row kwargs) say."""
    out = []
    for p in hf.PINCHES:
        for lv in hf.LEVELS:
            cid = hf.row_id(p, lv)
            k = dict((rows or {}).get(cid, {}))
            k.setdefault("label", (labels or {}).get(cid, "not_lifted"))
            out.append(_row(p, lv, **k))
    return dict({"object": obj, "idx": idx, "side": side, "skipped": None, "stop": None, "error": None, "g0": [],
                 "sim_truth": {"mass": mass}, "rows": out, "replay": None, "seconds": 8.0}, **kw)


def _entry(rec, ceiling10=False, r4=False):
    ctl = next((r for r in rec["rows"] if r["id"] == hf.CONTROL), None)
    ref = None if ctl is None else {"label": ctl.get("label"), "qpos_sha": ctl.get("qpos_sha"), "trace_sha": ctl.get("trace_sha")}
    return {"object": rec["object"], "idx": rec["idx"], "side": rec["side"], "skipped": rec["skipped"], "ref": ref,
            "ceiling10": None if rec["skipped"] else ceiling10, "r4": r4 if rec["side"] == "test" else None,
            "mass": rec["sim_truth"]["mass"] if rec.get("sim_truth") else None}


def _mon(rec, shift=2.0, turn=1.0):
    m = hf.monitor_record(rec)
    for r in m["rows"]:
        if "seconds" in next(x for x in rec["rows"] if x["id"] == r["id"]):
            r["close"] = {"close_shift_mm": shift, "close_turn_deg": 0.5, "lift_end_turn_deg": turn, "lift_end_pads": turn is not None}
    return m


def _view(rec, **kw):
    return hs.case_view(rec, _entry(rec, **kw), _mon(rec))


# ------------------------------------------------------------------ the row states (ruling B)

def test_the_row_state_takes_the_cause_and_ruling_b_lifts_only_the_penetration_at_40_and_70_n():
    assert hs.row_state(_row("across_com", "70N", label="tipped")) == "tipped"
    assert hs.row_state(_row("across_com", "70N", feasible=False)) == "infeasible"
    assert hs.row_state(_row("across_com", "70N", error="RuntimeError: x")) == "error"
    assert hs.row_state(_row("across_com", "70N", label="placed_stable", deep=4.2)) == "placed_stable"    # a monitor field
    assert hs.row_state(_row("across_com", "40N", label="placed_stable", deep=3.4)) == "placed_stable"
    assert hs.row_state(_row("across_com", "2f_std", label="placed_stable", deep=3.4)) == "artifact"     # commit_std's limit
    assert hs.row_state(_row("across_com", "f_std", deep=3.1)) == "artifact"
    for kw in ({"rev": 5}, {"at_rest": False}, {"jam": True}):                                           # artifacts at every level
        assert hs.row_state(_row("across_com", "70N", label="placed_stable", **kw)) == "artifact"
    assert hs.row_state(_row("across_com", "70N", label="crushed")) == "damage"                          # a damage label
    assert hs.PEN_MM == 3.0 and hs.PEN_LEVELS == ("40N", "70N")


def test_the_limits_are_the_adapters():
    from robots.mujoco_lab import adapter

    assert hs.PEN_MM == 1000 * adapter.DEEP_M and hs.REV_STEPS == adapter.REV_STEPS


# ------------------------------------------------------------------ the case views

def test_a_case_view_gives_placed_lifted_and_the_strict_lift_per_cell_and_infeasible_as_neither():
    rec = _case(labels={"across_com@70N": "placed_stable", "across_com@40N": "missed", "across_0.5@70N": "dropped_high"},
                rows={"across_com@f_std": {"feasible": False}})
    v = _view(rec, ceiling10=True, r4=False)
    c = v["cells"]
    assert c["across_com@70N"]["placed"] and c["across_com@70N"]["lifted"] and c["across_com@70N"]["lifted_strict"]
    assert not c["across_com@40N"]["placed"] and c["across_com@40N"]["lifted"] and not c["across_com@40N"]["lifted_strict"]
    assert c["across_0.5@70N"]["lifted"] and not c["across_0.5@70N"]["placed"]
    assert c["across_com@f_std"]["state"] == "infeasible" and not c["across_com@f_std"]["placed"] and not c["across_com@f_std"]["lifted"]
    assert not c[hf.CONTROL]["lifted"] and v["drop"] == [] and v["drop_a"] is False
    assert (v["mass"], v["ceiling10"], v["r4"], v["side"]) == (0.5, True, False, "test")
    assert c["across_com@70N"]["close"]["close_shift_mm"] == 2.0 and c["across_com@f_std"]["close"] is None


def test_a_case_drops_on_an_artifact_or_an_error_and_ruling_4_as_written_also_on_the_penetration():
    v = _view(_case(rows={"across_com@70N": {"deep": 4.0, "label": "placed_stable"}}))
    assert v["drop"] == [] and v["drop_a"] is True and v["cells"]["across_com@70N"]["pen"]
    v = _view(_case(rows={"across_0.5@2f_std": {"deep": 3.5}}))
    assert v["drop"] == ["across_0.5@2f_std: artifact"] and v["drop_a"] is True
    v = _view(_case(rows={"across_com@40N": {"error": "RuntimeError: x"}}))
    assert v["drop"] == ["across_com@40N: error"] and v["drop_a"] is True


# ------------------------------------------------------------------ the bootstrap and the verdicts

def test_the_share_and_the_paired_difference_use_a_fixed_seed_percentile_bootstrap_over_cases():
    a = [True] * 6 + [False] * 4
    s = hs.share(a)
    assert (s["n"], s["k"], s["share"]) == (10, 6, 60.0) and s["lo"] < 60.0 < s["hi"] and hs.share(a) == s
    rng = np.random.default_rng(hs.SEED)
    boot = np.asarray(a, float)[rng.integers(0, 10, size=(hs.DRAWS, 10))].mean(axis=1) * 100
    assert [s["lo"], s["hi"]] == pytest.approx(list(np.percentile(boot, [2.5, 97.5])))
    p = hs.paired([True] * 5 + [False] * 5, [False] * 10)
    assert p["n"] == 10 and p["diff"] == 50.0 and p["a"] == 50.0 and p["b"] == 0.0 and p["lo"] <= 50.0 <= p["hi"]
    assert hs.share([]) == {"n": 0, "k": 0, "share": None, "lo": None, "hi": None} and hs.paired([], [])["diff"] is None
    assert hs.DRAWS == 2000 and hs.MARGINS == {"placed": 10.0, "lifted": 20.0}


def test_a_verdict_needs_the_whole_interval_past_the_margin_or_below_it():
    assert hs.verdict({"n": 44, "diff": 30.0, "lo": 12.0, "hi": 40.0}, 10.0) == "supported"
    assert hs.verdict({"n": 44, "diff": 0.0, "lo": -5.0, "hi": 9.0}, 10.0) == "not supported"
    assert hs.verdict({"n": 44, "diff": 12.0, "lo": 4.0, "hi": 20.0}, 10.0) == "undecided at n = 44"
    assert hs.verdict({"n": 0, "diff": None, "lo": None, "hi": None}, 10.0) is None


# ------------------------------------------------------------------ the readings

def _views(n_hit=6, n=10, side="test", **kw):
    """n cases; in the first n_hit the CoM pinch places at 70 N; heavy (1.5 kg) from case 5 on."""
    out = []
    for i in range(n):
        lab = {"across_com@70N": "placed_stable"} if i < n_hit else {}
        out.append(_view(_case(idx=i, side=side, mass=0.5 if i < 5 else 1.5, labels=lab), **kw))
    return out


def test_reading_1_gives_the_shares_at_the_com_pinch_and_the_three_differences_with_the_primary_first():
    r = hs.reading1(_views())
    assert r["primary"] == "70N - f_std" and list(r["diffs"]) == ["70N - f_std", "40N - f_std", "2f_std - f_std"]
    d = r["diffs"]["70N - f_std"]
    assert d["placed"]["diff"] == 60.0 and d["placed"]["verdict"] in ("supported", "undecided at n = 10")
    assert d["lifted"]["diff"] == 60.0 and r["shares"]["70N"]["placed"]["k"] == 6 and r["shares"]["f_std"]["placed"]["k"] == 0
    assert r["diffs"]["40N - f_std"]["placed"]["diff"] == 0.0


def test_the_missed_sensitivity_counts_a_missed_row_as_a_lift_failure():
    vs = [_view(_case(idx=i, labels={"across_com@70N": "missed"})) for i in range(4)]
    assert hs.reading1(vs)["diffs"]["70N - f_std"]["lifted"]["diff"] == 100.0
    assert hs.reading1(vs, strict=True)["diffs"]["70N - f_std"]["lifted"]["diff"] == 0.0


def test_reading_2_compares_the_pinches_at_f_std_and_at_70_n():
    r = hs.reading2(_views())
    assert list(r) == ["f_std", "70N"] and r["70N"]["placed"]["diff"] == 60.0 and r["f_std"]["placed"]["diff"] == 0.0


def test_reading_3_splits_each_cell_at_1_kg_of_hidden_mass_and_states_n_per_band():
    r = hs.reading3(_views())
    assert r["light"]["n"] == 5 and r["heavy"]["n"] == 5
    assert r["light"]["cells"]["across_com@70N"]["placed"]["k"] == 5 and r["heavy"]["cells"]["across_com@70N"]["placed"]["k"] == 1


def test_the_best_cell_is_the_highest_placed_share_ties_to_the_lower_squeeze_then_the_mid_pinch():
    vs = _views()
    assert hs.best_cell(vs) == "across_com@70N"
    vs = [_view(_case(idx=i, labels={"across_com@40N": "placed_stable", "across_0.5@40N": "placed_stable",
                                     "across_com@70N": "placed_stable"})) for i in range(3)]
    assert hs.best_cell(vs) == "across_0.5@40N"


def test_reading_7_pairs_the_best_cell_with_the_corrected_ceiling_and_stage_2c():
    vs = _views(ceiling10=False, r4=False)
    vs[0]["ceiling10"] = True
    r = hs.reading7(vs, "across_com@70N")
    assert r["cell"] == "across_com@70N" and r["vs_ceiling10"]["n"] == 10 and r["vs_ceiling10"]["diff"] == 50.0
    assert r["vs_r4"]["diff"] == 60.0 and r["vs_r4"]["verdict"] is not None
    val = _views(side="validation")
    assert hs.reading7(val, "across_com@70N")["vs_r4"]["n"] == 0                  # Stage 2c holds test cases only


def test_reading_4_breaks_down_a_cell_over_every_case_of_the_side_and_reading_6_counts_damage():
    vs = [_view(_case(idx=0, labels={"across_com@70N": "tipped"})), _view(_case(idx=1, rows={"across_com@70N": {"rev": 7}})),
          _view(_case(idx=2, rows={"across_com@70N": {"feasible": False}})), _view(_case(idx=3, labels={"across_com@70N": "crushed"}))]
    b = hs.breakdown(vs, "across_com@70N")
    assert b["tipped"] == 1 and b["artifact"] == 1 and b["infeasible"] == 1 and b["damage"] == 1 and sum(b.values()) == 4
    assert list(b) == list(hs.STATES)
    assert hs.damage_counts(vs)["across_com@70N"] == 1 and hs.damage_counts(vs)[hf.CONTROL] == 0


def test_the_per_cell_table_counts_infeasible_artifacts_errors_penetration_missed_and_drops():
    vs = [_view(_case(idx=0, rows={"across_com@70N": {"deep": 4.0}, "across_com@40N": {"label": "missed"}})),
          _view(_case(idx=1, rows={"across_com@f_std": {"feasible": False}, "across_com@70N": {"feasible": False}})),
          _view(_case(idx=2, rows={"across_0.5@70N": {"error": "RuntimeError: x"}}))]
    t = hs.per_cell(vs)
    assert t["across_com@70N"] == {"cases": 3, "simulated": 2, "infeasible": 1, "artifact": 0, "error": 0, "penetration": 1,
                                   "missed": 0, "damage": 0, "drops": 0}
    assert t["across_com@40N"]["missed"] == 1 and t["across_0.5@70N"]["error"] == 1 and t["across_0.5@70N"]["drops"] == 1
    by = hs.infeasible_by_object(vs + [_view(_case(obj="handal__hammer_2", rows={"across_com@f_std": {"feasible": False}}))])
    assert by == {"handal__hammer_1": {"cases": 1, "why": {"unreachable: collision (approach)": 1}},
                  "handal__hammer_2": {"cases": 1, "why": {"unreachable: collision (approach)": 1}}}


def test_the_unpaired_lines_use_every_clean_row_of_a_cell():
    vs = [_view(_case(idx=i, labels={"across_com@70N": "placed_stable"}, rows={"across_0.5@f_std": {"rev": 9}} if i < 2 else None))
          for i in range(5)]
    u = hs.unpaired(vs)
    assert u[hf.CONTROL]["placed"]["n"] == 3 and u["across_com@70N"]["placed"]["n"] == 5 and u["across_com@70N"]["placed"]["k"] == 5


def test_the_wedge_diagnostic_sets_the_deep_rows_beside_the_shallow_ones_at_40_and_70_n():
    vs = [_view(_case(idx=0, rows={"across_com@70N": {"deep": 3.6, "label": "placed_stable"}})),
          _view(_case(idx=1, rows={"across_com@70N": {"deep": 2.0, "label": "tipped"}})),
          _view(_case(idx=2, rows={"across_com@70N": {"deep": 1.0}}))]
    w = hs.wedge(vs)
    assert set(w) == {hf.row_id(p, lv) for p in hf.PINCHES for lv in ("40N", "70N")}
    assert w["across_com@70N"]["deep"] == {"rows": 1, "lifted": 1, "placed": 1, "lifted_share": 100.0, "placed_share": 100.0}
    assert w["across_com@70N"]["shallow"] == {"rows": 2, "lifted": 1, "placed": 0, "lifted_share": 50.0, "placed_share": 0.0}


def test_the_close_report_gives_the_shift_the_rows_over_5_mm_and_the_lift_end_turn_per_cell():
    a, b, c = (_case(idx=i) for i in range(3))
    vs = [hs.case_view(a, _entry(a), _mon(a, shift=6.0, turn=10.0)), hs.case_view(b, _entry(b), _mon(b, shift=2.0, turn=None)),
          hs.case_view(c, _entry(c), _mon(c, shift=4.0, turn=2.0))]
    r = hs.close_report(vs)
    assert r["across_com@70N"] == {"rows": 3, "shift_median_mm": 4.0, "shift_max_mm": 6.0, "over_5mm": 1, "turn_rows": 2,
                                   "turn_median_deg": 6.0}


def test_the_penetration_distribution_per_level_reads_every_simulated_row():
    recs = [_case(rows={"across_com@70N": {"deep": 4.0}, "across_0.5@70N": {"deep": 2.0}, "across_com@f_std": {"feasible": False}})]
    p = hs.penetration(recs)
    assert p["70N"] == {"rows": 2, "median_mm": 3.0, "max_mm": 4.0, "over_3mm": 1, "share_over_3mm": 50.0}
    assert p["f_std"]["rows"] == 1


# ------------------------------------------------------------------ the gates after the one open

def _set(n=3):
    recs = [_case(idx=i) for i in range(n)] + [_case(idx=n, side="validation", skipped="stage1 artifact", rows=None, g0=None)]
    recs[-1]["rows"], recs[-1]["sim_truth"] = [], None
    cases = [_entry(r) for r in recs]
    mon = {(r["object"], r["idx"]): _mon(r) for r in recs}
    return recs, mon, cases


def test_the_gates_pass_on_complete_consistent_records():
    recs, mon, cases = _set()
    g = hs.post_gates(recs, mon, cases)
    assert g["problems"] == [] and g["g0"] == {"compared": 3, "mismatches": 0} and g["hash_join"] == {"rows": 24, "mismatches": 0}


def test_a_missing_or_doubled_case_or_row_stops_the_summary():
    recs, mon, cases = _set()
    assert any("missing" in p for p in hs.post_gates(recs[1:], mon, cases)["problems"])
    assert any("twice" in p for p in hs.post_gates(recs + [recs[0]], mon, cases)["problems"])
    recs[0]["rows"] = recs[0]["rows"][:7]
    assert any("rows" in p for p in hs.post_gates(recs, mon, cases)["problems"])


def test_g0_is_checked_again_and_names_the_fields_only():
    recs, mon, cases = _set()
    cases[0]["ref"] = dict(cases[0]["ref"], label="tipped")
    p = hs.post_gates(recs, mon, cases)["problems"]
    assert any("G0" in x and "label" in x for x in p) and not any("tipped" in x or "not_lifted" in x for x in p)


def test_an_artifact_flag_that_the_monitor_does_not_explain_stops_the_summary():
    recs, mon, cases = _set()
    recs[1]["rows"][3]["artifact"] = True
    assert any("artifact flag" in p for p in hs.post_gates(recs, mon, cases)["problems"])


def test_a_monitor_row_with_another_hash_stops_the_summary():
    recs, mon, cases = _set()
    mon[(recs[2]["object"], 2)]["rows"][5]["trace_sha"] = "other"
    g = hs.post_gates(recs, mon, cases)
    assert g["hash_join"]["mismatches"] == 1 and any("hash" in p for p in g["problems"])
    del mon[(recs[0]["object"], 0)]
    assert any("monitor" in p and "missing" in p for p in hs.post_gates(recs, mon, cases)["problems"])


def test_the_squeeze_gate_reads_the_hold_at_70_n_states_n_and_falls_back_to_the_close():
    recs = [_case(idx=i, rows={"across_com@70N": {"hold": 68.0}, "across_0.5@70N": {"hold": 69.0}}) for i in range(6)]
    g = hs.squeeze_gate(recs)
    assert g["gate"]["reader"] == "hold" and g["gate"]["n"] == 12 and g["gate"]["ok"] and g["gate"]["median"] == 68.5
    assert g["levels"]["70N"]["hold_n"] == 12 and g["levels"]["f_std"]["close_median"] == 10.0
    recs = [_case(idx=i, rows={"across_com@70N": {"hold": 50.0}}) for i in range(4)]
    g = hs.squeeze_gate(recs)
    assert g["gate"]["reader"] == "close" and g["gate"]["n"] == 8 and g["gate"]["ok"] and "fewer than 10" in g["gate"]["note"]
    recs = [_case(idx=i, rows={"across_com@70N": {"hold": 50.0}, "across_0.5@70N": {"hold": 52.0}}) for i in range(6)]
    assert not hs.squeeze_gate(recs)["gate"]["ok"]                                     # 51 N is more than 20 % under 70 N


# ------------------------------------------------------------------ the report, the markdown and the figure

def _report(n_test=8, n_val=4, drop_test=0):
    recs, mon, cases = [], {}, []
    for i in range(n_test + n_val):
        side = "test" if i < n_test else "validation"
        rows = {"across_com@70N": {"label": "placed_stable" if i % 2 == 0 else "tipped", "deep": 3.5 if i % 4 == 0 else 1.5, "hold": 69.5},
                "across_0.5@70N": {"hold": 70.1}}
        if i < drop_test:
            rows["across_0.5@f_std"] = {"rev": 9}
        r = _case(obj=hf.HAMMERS[side][i % 2], idx=i, side=side, mass=0.6 if i % 3 else 1.8, rows=rows)
        recs.append(r)
        mon[(r["object"], i)] = _mon(r)
        cases.append(_entry(r, ceiling10=i == 1, r4=i == 3))
    return hs.build(recs, mon, cases, pre={"note": "synthetic"})


def test_the_report_has_the_headline_the_sensitivity_the_validation_and_the_exploratory_pooled_lines():
    rep = _report()
    assert list(rep["lines"]) == ["test", "test_penetration_dropped", "validation", "pooled"]
    t = rep["lines"]["test"]
    assert t["n"] == 8 and t["r1"]["diffs"]["70N - f_std"]["placed"]["diff"] == 50.0 and rep["best"]["cell"] == "across_com@70N"
    assert rep["lines"]["test_penetration_dropped"]["n"] == 6 and rep["lines"]["pooled"]["n"] == 12
    assert rep["lines"]["pooled"]["exploratory"] is True and rep["unpaired"] is None
    assert rep["squeeze"]["gate"]["ok"] and set(rep["wedge"]) == {"test", "validation"} and set(rep["close"]) == {"test", "validation"}
    assert "r1_missed" in t and rep["r4"]["test"]["control"]["not_lifted"] == 8


def test_more_than_five_test_drops_add_the_unpaired_lines():
    rep = _report(n_test=10, drop_test=6)
    assert rep["lines"]["test"]["n"] == 4 and rep["unpaired"] is not None and rep["unpaired"][hf.CONTROL]["placed"]["n"] == 4


def test_the_markdown_states_the_lines_the_selection_and_the_independence_note():
    md = hs.markdown(_report())
    for s in ("Reading 1", "Reading 7", "exploratory", "penetration-flagged cases dropped", "validation", "independent",
              "four objects", "best cell", "missed"):
        assert s in md, s


def test_the_figure_draws_both_outcomes_in_light_and_heavy_panels(tmp_path):
    hs.figure(_report(), str(tmp_path / "fig.png"))
    assert os.path.getsize(tmp_path / "fig.png") > 10_000


# ------------------------------------------------------------------ the command: gates before the open, one open, the stop

def _write_run(d, recs, cfg, metrics=None):
    os.makedirs(d, exist_ok=True)
    for side in ("test", "validation"):
        with open(os.path.join(d, f"records_{side}.jsonl"), "w") as f:
            for r in recs:
                if r["side"] == side:
                    f.write(json.dumps(r) + "\n")
    json.dump(cfg, open(os.path.join(d, "run_config.json"), "w"))
    json.dump(dict({"stop": None, "stops": 0, "g0_mismatches": 0, "case_seconds": 36.0}, **(metrics or {})),
              open(os.path.join(d, "run_metrics.json"), "w"))
    open(os.path.join(d, "run.exit"), "w").write("EXIT 0\n")


def _fixture(tmp_path, n_test=12, bad_hold=False):
    recs, cases = [], []
    for i in range(n_test + 4):
        side = "test" if i < n_test else "validation"
        hold = 50.0 if bad_hold else 69.0
        r = _case(idx=i, side=side, rows={"across_com@70N": {"hold": hold}, "across_0.5@70N": {"hold": hold}})
        recs.append(r)
        cases.append(_entry(r))
    cf = tmp_path / "cases.json"
    cf.write_text(json.dumps({"cases": cases}))
    sha = hashlib.sha256(cf.read_bytes()).hexdigest()
    keys = [[r["object"], r["idx"]] for r in recs]
    base = {"replan_hold": True, "sim": hf.CONTACTS_SIM, "cases_sha256": sha, "dirty": False, "code_revision": "c" * 40, "keys": keys}
    _write_run(str(tmp_path / "run" / "unit_0"), recs, dict(base, close_monitor=False))
    _write_run(str(tmp_path / "mon" / "unit_0"), [hf.monitor_record(dict(r, rows=[dict(x, close={"close_shift_mm": 1.0, "close_turn_deg": 0.1,
               "lift_end_turn_deg": None, "lift_end_pads": False}) for x in r["rows"]])) for r in recs],
               dict(base, close_monitor=True, code_revision="d" * 40))
    sm = tmp_path / "smoke"
    _write_run(str(sm / "unit_0"), recs[:2], dict(base, close_monitor=False))
    os.makedirs(sm / "report")
    (sm / "report" / "run.exit").write_text("EXIT 0\n")
    (sm / "report" / "smoke_summary.json").write_text(json.dumps({"runs": str(sm / "unit_0"), "g0": {"compared": 2, "mismatches": 0},
                                                                  "g1": {"label_mismatches": 0, "other_mismatches": 0},
                                                                  "projection_cpu_h": 0.2}))
    return SimpleNamespace(runs=str(tmp_path / "run" / "unit_0"), monitor=str(tmp_path / "mon" / "unit_0"), cases_file=str(cf),
                           smoke=str(sm / "report"), out=str(tmp_path / "out"))


def test_the_command_writes_the_summary_after_the_gates_and_opens_each_records_file_once(tmp_path, monkeypatch):
    a = _fixture(tmp_path)
    opened = []
    real = hs._read_jsonl
    monkeypatch.setattr(hs, "_read_jsonl", lambda p: opened.append(p) or real(p))
    monkeypatch.setenv("WANDB_MODE", "disabled")
    rep = hs.summary_cmd(a)
    assert sorted(opened) == sorted(os.path.join(d, f"records_{s}.jsonl") for d in (a.runs, a.monitor) for s in ("test", "validation"))
    for f in ("summary.json", "summary.md", "figure.png", "figure.pdf"):
        assert os.path.exists(os.path.join(a.out, f)), f
    assert rep["lines"]["test"]["n"] == 12 and rep["cpu_h"]["projected"] == 0.2
    with pytest.raises(SystemExit, match="exists"):
        hs.summary_cmd(a)


def test_a_gate_before_the_open_refuses_without_reading_a_record(tmp_path, monkeypatch):
    a = _fixture(tmp_path)
    opened = []
    monkeypatch.setattr(hs, "_read_jsonl", lambda p: opened.append(p) or [])
    cfg = os.path.join(a.runs, "run_config.json")
    c = json.load(open(cfg))
    json.dump(dict(c, replan_hold=False), open(cfg, "w"))
    with pytest.raises(SystemExit, match="replan"):
        hs.summary_cmd(a)
    json.dump(dict(c, code_revision="e" * 40), open(cfg, "w"))
    with pytest.raises(SystemExit, match="smoke"):
        hs.summary_cmd(a)
    json.dump(c, open(cfg, "w"))
    open(os.path.join(a.monitor, "run.exit"), "w").write("EXIT 1\n")
    with pytest.raises(SystemExit, match="EXIT"):
        hs.summary_cmd(a)
    assert opened == [] and not os.path.exists(os.path.join(a.out, "summary.json"))


def test_a_failed_squeeze_gate_writes_the_stop_file_and_exits_3_before_any_reading(tmp_path, monkeypatch):
    a = _fixture(tmp_path, bad_hold=True)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    with pytest.raises(SystemExit) as e:
        hs.summary_cmd(a)
    assert e.value.code == 3
    stop = json.load(open(os.path.join(a.out, "STOP.json")))
    assert stop["gate"] == "squeeze" and stop["n"] == 32 and not os.path.exists(os.path.join(a.out, "summary.json"))
    assert "not_lifted" not in json.dumps(stop)


def test_a_mass_that_is_not_the_cases_files_or_a_monitor_case_outside_it_stops_the_summary():
    recs, mon, cases = _set()
    cases[1]["mass"] = 2.5
    assert any("mass" in p for p in hs.post_gates(recs, mon, cases)["problems"])
    recs, mon, cases = _set()
    mon[("handal__hammer_9", 0)] = dict(mon[(recs[0]["object"], 0)], object="handal__hammer_9")
    assert any("monitor pass" in p and "not in the cases file" in p for p in hs.post_gates(recs, mon, cases)["problems"])
