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

"""Experiment 1, step 7 (evpi_sim; daily-logs 2026-11-05-evpi-decomposition): the seeds, the sample, the realized costs,
the simulations a case needs, the paired exclusion, the readings and the summary."""

import json
import os
from collections import Counter

import numpy as np
import pytest

from robots.mujoco_lab import calibrate as cal
from robots.mujoco_lab import evpi_props as ev
from robots.mujoco_lab import evpi_sim as es
from rpent.skills.pick_place.costs import COST, TIME_COST_PER_S, particle_costs
from rpent.skills.pick_place.probes import PROBE_SECONDS
from rpent.skills.pick_place.protocol import OUTCOMES, Prediction

ROW = 'commit_std|{"force": 20.0}'
WEIGH = 'probe_weigh|{"force": 20.0, "probe": "weigh"}'


def _rec(obj="a", idx=0, prior="class", setting="k30_A3", changed=("all",), to=ROW, ess=500.0, **kw):
    """A run record: a_prior declines; the reveals in ``changed`` choose ``to``."""
    action = {k: "decline" for k in (ev.PRIOR,) + ev.REVEALS}
    for S in changed:
        action[S] = to
    choice = {k: a.split("|")[0] for k, a in action.items()}
    J = {k: 3.0 if a == "decline" else 2.0 for k, a in action.items()}
    r = {"object": obj, "idx": idx, "prior": prior, "setting": setting, "care": 30.0, "abstain_cost": 3.0, "choice": choice,
         "action": action, "J": J, "evpi": {S: J[ev.PRIOR] - J[S] for S in ev.REVEALS}, "probe_detail": {},
         "ess": {k: {"ess_final": ess} for k in (ev.PRIOR,) + ev.REVEALS}, "artifact": False, "fingerprint": "f"}
    r.update(kw)
    return r


def _plan(r, cost_s=4.0):
    """The run's plan of record r: each rebuilt belief's choice, as the record has it."""
    beliefs = {}
    for k in [ev.PRIOR] + [S for S in ev.REVEALS if r["action"][S] != r["action"][ev.PRIOR]]:
        c = r["choice"][k]
        kind = "abstain" if c in ev.ABSTAINS else ("probe" if c.startswith("probe_") else "commit")
        beliefs[k] = {"choice": c, "action": r["action"][k], "kind": kind, "cost_s": cost_s if kind == "commit" else None}
    return {"object": r["object"], "idx": r["idx"], "prior": r["prior"], "setting": r["setting"], "beliefs": beliefs, "mismatch": []}


def _row_run(label="placed_stable", **kw):
    return dict({"end": "row", "label": label, "artifact": False}, **kw)


def _probe_run(end, kind="measured", thrown=False, label=None, probe="weigh", scale=1.0, cont_s=5.0):
    x = {"end": end, "probe": {"probe": probe, "kind": kind, "thrown": thrown, "damage": None, "artifact": False},
         "time_scale": scale, "artifact": False}
    if end == "row":
        x["cont"] = {"option": "commit_std", "label": label, "cost_s": cont_s, "artifact": False}
    return x


# ---------------------------------------------------------------- the seeds and the sample

def test_the_seeds_are_the_cases_own_reset_seed_then_four_from_a_new_stream():
    s = es.seeds_of(11, "ycb__mug", 3)
    assert len(s) == es.N_SEEDS == 5 and s[0] == 11 * 1000 + 3 and len(set(s)) == 5
    assert s[1:] == [int(x) for x in cal._stream(11, "ycb__mug", 3, es.SEED_STREAM).integers(2 ** 31, size=4)]
    assert es.SEED_STREAM not in (1, 2, 3, 4, ev.REVEAL_STREAM, ev.TRUE_STREAM)        # a stream no earlier step reads
    assert es.seeds_of(11, "ycb__mug", 4)[1:] != s[1:]


def test_the_sample_takes_up_to_50_critical_counted_records_per_main_cell_with_a_fixed_seed():
    recs = [_rec(obj=f"o{i % 13}", idx=i) for i in range(60)]                            # k30_A3 class: 60 critical
    recs += [_rec(obj=f"o{i}", idx=i, prior="generic") for i in range(7)]               # k30_A3 generic: 7
    recs += [_rec(obj="x", idx=i, setting="k100_A10", changed=()) for i in range(9)]    # invariant: never
    recs += [_rec(obj="y", idx=i, setting="k10_A10") for i in range(9)]                  # a low-stakes control: never
    recs += [_rec(obj="z", idx=1, setting="k100_A10", artifact=True), _rec(obj="z", idx=2, setting="k100_A10", error="boom")]
    s = es.draw_sample(recs)
    assert Counter((r["setting"], r["prior"]) for r in s) == {("k30_A3", "class"): 50, ("k30_A3", "generic"): 7}
    assert s == es.draw_sample(list(reversed(recs)))          # the draw does not follow the records' order
    assert [r["idx"] for r in s] != [r["idx"] for r in es.draw_sample(recs, seed=1)]


def test_the_smoke_records_come_from_every_cell_in_turn():
    recs = [_rec(obj=f"o{i}", idx=i, prior=p, setting=c) for i in range(12) for c, p in es.CELLS]
    smoke = es.smoke_order(es.draw_sample(recs))[:5]
    assert [(r["setting"], r["prior"]) for r in smoke] == list(es.CELLS) + [es.CELLS[0]]


# ---------------------------------------------------------------- the realized costs

@pytest.mark.parametrize("label", ["placed_stable", "dropped_high", "not_lifted", "crushed"])
def test_a_rows_realized_cost_is_particle_costs_at_the_realized_outcome(label):
    pred = Prediction(p={k: float(k == label) for k in OUTCOMES})
    assert es.row_cost(label, 4.2, 30.0, 3.0) == pytest.approx(float(particle_costs([pred], 4.2, 30.0, 3.0)[0]))


def test_a_probe_case_costs_the_probe_then_A_or_the_row_its_belief_picks():
    t = TIME_COST_PER_S * PROBE_SECONDS["weigh"]
    assert es.probe_case_cost(_probe_run("abstained"), 30.0, 3.0) == pytest.approx(t + 3.0)
    assert es.probe_case_cost(_probe_run("thrown", thrown=True), 30.0, 3.0) == pytest.approx(t + COST["dropped_high"] * 30.0 + 3.0)
    assert es.probe_case_cost(_probe_run("row", label="placed_stable"), 30.0, 3.0) == pytest.approx(t + TIME_COST_PER_S * 5.0)
    assert es.probe_case_cost(_probe_run("row", label="dropped_high"), 30.0, 3.0) == pytest.approx(
        t + COST["dropped_high"] * 30.0 + TIME_COST_PER_S * 5.0 + 3.0)
    assert es.probe_case_cost(_probe_run("abstained", scale=0.5), 30.0, 3.0) == pytest.approx(0.5 * t + 3.0)
    tipped = _probe_run("row", kind="tipped", probe="push", label="placed_stable")
    assert es.probe_case_cost(tipped, 30.0, 3.0) == pytest.approx(TIME_COST_PER_S * (PROBE_SECONDS["push"] + 5.0) + COST["tipped"])


# ---------------------------------------------------------------- the simulations of a case

def test_a_row_two_records_choose_runs_once_and_a_probe_runs_per_record_and_belief():
    plans = [{"prior": "class", "setting": "k30_A3", "beliefs": {"prior": {"kind": "abstain", "action": "decline"},
                                                                 "all": {"kind": "commit", "action": ROW},
                                                                 "mass": {"kind": "probe", "action": WEIGH}}},
             {"prior": "class", "setting": "k100_A10", "beliefs": {"prior": {"kind": "commit", "action": ROW},
                                                                   "all": {"kind": "probe", "action": WEIGH}}}]
    jobs = es.jobs_of(plans)
    assert set(jobs) == {("row", ROW), ("probe", "class", "k30_A3", "mass"), ("probe", "class", "k100_A10", "all")}
    assert jobs[("row", ROW)][:3] == ("class", "k30_A3", "all")          # its first chooser runs it


def test_a_pair_with_an_artifact_or_an_error_drops_from_both_actions_and_is_counted():
    r = _rec()
    runs = {(("row", ROW), k): _row_run(artifact=k == 1) for k in range(es.N_SEEDS)}
    runs[(("row", ROW), 3)] = {"error": "sim fault"}
    out = es.record_pairs(r, _plan(r), runs)["all"]
    assert out["kept"] == 3 and out["drops"] == {"prior": 0, "S": 2, "both": 0}
    assert out["d"] == pytest.approx(3.0 - es.row_cost("placed_stable", 4.0, 30.0, 3.0))
    assert out["rp"] == pytest.approx(3.0) and not out["probe_pair"]


def test_the_pair_cost_is_paired_by_seed():
    r = _rec(changed=("all",))
    r["action"][ev.PRIOR], r["choice"][ev.PRIOR] = 'commit_low|{"force": 10.0}', "commit_low"
    plan = _plan(r)
    plan["beliefs"][ev.PRIOR]["cost_s"] = 6.0
    runs = {}
    for k in range(es.N_SEEDS):
        runs[(("row", r["action"][ev.PRIOR]), k)] = _row_run("dropped_high" if k < 2 else "placed_stable")
        runs[(("row", ROW), k)] = _row_run("placed_stable")
    out = es.record_pairs(r, plan, runs)["all"]
    want = np.mean([es.row_cost("dropped_high" if k < 2 else "placed_stable", 6.0, 30.0, 3.0) - es.row_cost("placed_stable", 4.0, 30.0, 3.0)
                    for k in range(es.N_SEEDS)])
    assert out["kept"] == 5 and out["d"] == pytest.approx(want)


@pytest.mark.parametrize("ci,n,share,want", [([0.01, 0.2], 20, 0.0, "the realized gain from revealing S is positive in this cell"),
                                             ([-0.3, -0.05], 20, 0.0, "revealing S makes the realized cost worse in this cell"),
                                             ([-0.1, 0.1], 20, 0.0, "no detectable realized gain"),
                                             ([0.01, 0.2], 9, 0.0, "counts only"),
                                             ([0.01, 0.2], 20, 0.11, "counts only")])
def test_the_reading_of_the_realized_gain(ci, n, share, want):
    assert es.reading(ci, n, share) == want


# ---------------------------------------------------------------- the summary

def _case_set():
    """12 records in κ30 A3 / class whose reveals choose a row that places (a gain of 3 - 0.2), one with a thin reveal,
    and 3 whose a_all is a weigh probe that then places (probe pairs)."""
    recs = [_rec(obj=f"o{i % 6}", idx=i, changed=("all", "mass"), ess=50.0 if i == 0 else 500.0) for i in range(12)]
    recs += [_rec(obj=f"p{i}", idx=i, changed=("all",), to=WEIGH) for i in range(3)]
    plans, runs = [], []
    for r in recs:
        plans.append(_plan(r))
        for k in range(es.N_SEEDS):
            if r["action"]["all"] == ROW:
                runs.append(dict(_row_run(), object=r["object"], idx=r["idx"], job=["row", ROW], seed_k=k))
            else:
                runs.append(dict(_probe_run("row", label="placed_stable"), object=r["object"], idx=r["idx"],
                                 job=["probe", "class", "k30_A3", "all"], seed_k=k))
    return recs, plans, runs


def test_the_summary_reads_s_all_once_per_cell_and_keeps_probe_pairs_and_single_properties_apart():
    recs, plans, runs = _case_set()
    out = es.summarize(recs, plans, runs, n_boot=200)
    c = out["tables"]["all"]["k30_A3"]["class"]
    assert c["all"]["main"]["n"] == 12 and c["all"]["probe"]["n"] == 3
    assert c["all"]["main"]["reading"] == "the realized gain from revealing S is positive in this cell"
    assert c["all"]["main"]["realized"]["mean"] == pytest.approx(3.0 - TIME_COST_PER_S * 4.0)
    assert c["all"]["main"]["predicted"]["mean"] == pytest.approx(1.0)
    assert c["all"]["main"]["gap"]["mean"] == pytest.approx(3.0 - TIME_COST_PER_S * 4.0 - 1.0)
    assert c["all"]["main"]["level"]["J_S"]["mean"] == pytest.approx(2.0)
    assert c["all"]["main"]["level"]["realized_S"]["mean"] == pytest.approx(TIME_COST_PER_S * 4.0)
    assert c["mass"]["main"]["reading"] == "descriptive" and c["all"]["probe"]["reading"] is None
    assert out["tables"]["ess_ok"]["k30_A3"]["class"]["all"]["main"]["n"] == 11
    assert out["n_main_readings"] == len(es.CELLS) and out["drops"]["k30_A3"]["class"]["share"] == 0.0
    assert out["probe_pairs"]["k30_A3"]["class"] == 3


def test_a_cell_whose_pairs_drop_more_than_a_tenth_gets_counts_only():
    recs, plans, runs = _case_set()
    for x in runs:
        if x["job"][0] == "row" and x["seed_k"] == 0:
            x["artifact"] = True                                  # seed 0 of every row: 24 of the cell's 135 pairs
    out = es.summarize(recs, plans, runs, n_boot=200)
    assert out["drops"]["k30_A3"]["class"]["share"] > 0.10
    assert out["tables"]["all"]["k30_A3"]["class"]["all"]["main"]["reading"] == "counts only"
    assert out["drops"]["k30_A3"]["class"]["by_action"]["S"] > 0 and out["drops"]["k30_A3"]["class"]["by_action"]["prior"] == 0


def test_the_markdown_names_the_main_reading_the_secondary_quantity_and_the_probe_rows():
    recs, plans, runs = _case_set()
    md = es.markdown(es.summarize(recs, plans, runs, n_boot=200))
    assert "realized minus predicted gain on cases selected for a predicted gain" in md
    assert "Probe pairs" in md and "descriptive" in md and "no multiplicity correction" in md


def test_load_runs_refuses_dirty_code_or_another_sample(tmp_path):
    def shard(name, **cfg):
        d = tmp_path / name
        d.mkdir()
        (d / "run_config.json").write_text(json.dumps(dict({"code_revision": "abc", "dirty": False, "sample_sha256": "s1"}, **cfg)))
        (d / "plans.jsonl").write_text(json.dumps({"object": name, "idx": 0, "prior": "class", "setting": "k30_A3"}) + "\n")
        (d / "runs.jsonl").write_text("")
        return str(d)

    a, b = shard("a"), shard("b")
    assert len(es.load_runs([a, b])[0]) == 2
    with pytest.raises(SystemExit, match="dirty"):
        es.load_runs([a, shard("c", dirty=True)])
    with pytest.raises(SystemExit, match="differs"):
        es.load_runs([a, shard("d", sample_sha256="s2")])


# ---------------------------------------------------------------- the replay (needs the sim and the m2b2_v12 bundle)

SIM = os.environ.get("RPENT_MUJOCO_SIM_DIR")
V12 = os.environ.get("EVPI_TEST_BUNDLE", "/home/chungyili/Codes/RPent-arm/robots/mujoco_lab/data/m2b2_v12")


@pytest.mark.skipif(not (SIM and os.path.exists(os.path.join(V12, "folds", "folds.json"))), reason="sim dir or m2b2_v12 bundle missing")
def test_the_replay_rebuilds_a_run_records_choices_and_another_reset_seed_keeps_the_scene():
    """One case at 300 particles: evpi_props.run_case's record, then the replay of every belief's first menu (no
    simulation of an action)."""
    from robots.mujoco_lab.com_gap import fingerprint
    from robots.mujoco_lab.prior import fragility_world

    o, idx = "hot3d__mug", 3
    folds = json.load(open(os.path.join(V12, "folds", "folds.json")))
    prims = cal._primitives(os.path.join(V12, "folds", f"fold_{folds[o]}"), None, None)
    prims.n_particles = 300
    world = fragility_world(prims.manifest)
    s = {"seed": 11, "fold": folds[o], "settings": [("k30_A3", 30.0, 3.0)], "flat": None}
    priors = {"class": prims.prior}
    with ev.SharedMenus(prims) as shared:
        rec = ev.run_case(prims, o, idx, s, priors, world, shared)[0]
    assert "error" not in rec
    ctx = es.replay_case(prims, o, idx, [rec], s, priors, world, all_keys=True)
    plan = ctx["plans"][0]
    assert plan["mismatch"] == [] and set(plan["beliefs"]) == {ev.PRIOR, *ev.REVEALS}
    assert all(plan["beliefs"][k]["action"] == rec["action"][k] for k in plan["beliefs"])
    ctx["reset"](es.seeds_of(11, o, idx)[1])
    assert fingerprint(prims.env) == rec["fingerprint"]
