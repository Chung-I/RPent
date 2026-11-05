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

import argparse
import json
import os
from collections import Counter

import numpy as np
import pytest

from robots.mujoco_lab import calibrate as cal
from robots.mujoco_lab import evpi_props as ev
from robots.mujoco_lab import evpi_sim as es
from robots.mujoco_lab.fragility import Thresholds
from rpent.skills.pick_place.belief import ParticleBelief
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


def _ev_shard(tmp_path, name, prior, recs, objects, **cfg):
    """An evpi_props shard dir that ran one prior: the run_config keys ev.load compares, the full generic table (every
    shard has all objects' rows) and the records."""
    d = tmp_path / name
    d.mkdir()
    c = {"seed": 11, "code_revision": "abc", "dirty": False, "priors_parsed": [prior], "generic": {o: {"mass": 1.0} for o in objects}}
    (d / "run_config.json").write_text(json.dumps(dict(c, **cfg)))
    (d / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    return str(d)


def test_the_sample_joins_shards_that_ran_one_prior_each_and_draws_as_from_each_prior_alone(tmp_path):
    cls = [_rec(obj=f"c{i % 17}", idx=i) for i in range(60)] + [_rec(obj=f"c{i % 5}", idx=i, setting="k100_A10") for i in range(55)]
    gen = [_rec(obj=f"g{i % 11}", idx=i, prior="generic") for i in range(52)]
    gen += [_rec(obj=f"g{i}", idx=i, prior="generic", setting="k100_A10") for i in range(7)]
    objs = sorted({r["object"] for r in cls + gen})
    dirs = [_ev_shard(tmp_path, "c0", "class", cls[::2], objs), _ev_shard(tmp_path, "g0", "generic", gen[::2], objs),
            _ev_shard(tmp_path, "c1", "class", cls[1::2], objs), _ev_shard(tmp_path, "g1", "generic", gen[1::2], objs)]
    with pytest.raises(SystemExit, match="priors_parsed"):
        ev.load(dirs)                                          # one load over both priors refuses (the 6 Nov stop)

    def sample(ds, name):
        return json.load(open(es.sample(argparse.Namespace(dirs=",".join(ds), out=str(tmp_path / name), per_cell=es.PER_CELL,
                                                           sample_seed=es.SAMPLE_SEED))))

    smp = sample(dirs, "s.json")
    alone = {"class": es.draw_sample(cls), "generic": es.draw_sample(gen)}
    want = [r for cell in es.CELLS for r in alone[cell[1]] if (r["setting"], r["prior"]) == cell]
    assert smp["records"] == want                              # each cell's draw: its own stream, its own records
    assert Counter((r["setting"], r["prior"]) for r in want) == {("k30_A3", "class"): 50, ("k100_A10", "class"): 50,
                                                                ("k30_A3", "generic"): 50, ("k100_A10", "generic"): 7}
    assert smp["available"] == {"k30_A3/class": 60, "k30_A3/generic": 52, "k100_A10/class": 55, "k100_A10/generic": 7}
    assert smp["config"]["priors_parsed"] == ["class", "generic"]
    assert set(smp["config"]["generic"]) == {r["object"] for r in want}
    assert sample(dirs[::-1], "s_rev.json")["records"] == want  # the dirs' order does not matter
    g12 = [_ev_shard(tmp_path, n, "generic", rs, objs, seed=12) for n, rs in (("g2", gen[::2]), ("g3", gen[1::2]))]
    with pytest.raises(SystemExit, match=r"config differs in \['seed'\]"):
        sample([dirs[0], dirs[2]] + g12, "s12.json")         # every other config key must still agree across the priors
    with pytest.raises(SystemExit, match="repeated"):
        sample(dirs + [_ev_shard(tmp_path, "c2", "class", cls[:1], objs)], "sdup.json")


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
    assert out["n_main_readings"] == len(es.CELLS) and out["drops"]["all"]["k30_A3"]["class"]["share"] == 0.0
    assert out["probe_pairs"]["k30_A3"]["class"] == 3
    assert c["mass"]["unchanged"] == 3 and c["all"]["unchanged"] == 0      # the probe records keep a_prior under mass


def test_a_cell_whose_pairs_drop_more_than_a_tenth_gets_counts_only():
    recs, plans, runs = _case_set()
    for x in runs:
        if x["job"][0] == "row" and x["seed_k"] == 0:
            x["artifact"] = True                                  # seed 0 of every row: 24 of the cell's 135 pairs
    out = es.summarize(recs, plans, runs, n_boot=200)
    d = out["drops"]["all"]["k30_A3"]["class"]
    assert d["share"] > 0.10 and d["by_action"]["S"] > 0 and d["by_action"]["prior"] == 0
    m = out["tables"]["all"]["k30_A3"]["class"]["all"]["main"]
    assert m["reading"] == "counts only" and m["counts_only"] and m["realized"] is None and m["level"] is None
    assert out["tables"]["all"]["k30_A3"]["class"]["mass"]["main"]["realized"] is None     # the whole cell


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


def test_a_block_below_ten_records_shows_counts_only():
    recs, plans, runs = _case_set()
    probe = es.summarize(recs, plans, runs, n_boot=50)["tables"]["all"]["k30_A3"]["class"]["all"]["probe"]
    assert probe["counts_only"] and probe["n"] == 3 and probe["realized"] is None and probe["predicted"] is None


def test_the_drops_and_the_outcome_counts_come_for_both_record_sets():
    recs, plans, runs = _case_set()
    out = es.summarize(recs, plans, runs, n_boot=50)
    assert set(out["drops"]) == set(out["counts"]) == {"all", "ess_ok"}
    assert out["drops"]["ess_ok"]["k30_A3"]["class"]["pairs"] == out["drops"]["all"]["k30_A3"]["class"]["pairs"] - 2 * es.N_SEEDS
    assert out["counts"]["ess_ok"]["k30_A3"]["class"]["outcomes"]["prior"]["decline"] == 14 * es.N_SEEDS


def test_the_summary_refuses_a_sampled_record_that_did_not_run_or_failed_the_replay():
    recs, plans, runs = _case_set()
    with pytest.raises(SystemExit, match="not run"):
        es.summarize(recs, plans[1:], runs, n_boot=50)
    plans[0]["mismatch"] = ["mass"]
    with pytest.raises(SystemExit, match="replay"):
        es.summarize(recs, plans, runs, n_boot=50)


def test_a_case_that_failed_drops_all_its_pairs_and_is_counted():
    recs, plans, runs = _case_set()
    plans[0] = {**{k: plans[0][k] for k in ("object", "idx", "prior", "setting")}, "beliefs": {}, "mismatch": [], "error": "reset failed"}
    out = es.summarize(recs, plans, runs, n_boot=50)
    d = out["drops"]["all"]["k30_A3"]["class"]
    assert d["by_action"]["both"] == 2 * es.N_SEEDS and d["case_errors"] == 1
    assert out["tables"]["all"]["k30_A3"]["class"]["all"]["main"]["n"] == 11


def test_a_simulation_whose_first_menu_differs_stops_the_run():
    assert es.stop_reason("o", 1, [{"gate": True}, {"gate": None, "error": "x"}]) is None
    assert "first menu" in es.stop_reason("o", 1, [{"gate": True}, {"gate": False, "seed_k": 2, "job": ["row", ROW]}])


def _smoke(tmp_path, recs, plans, runs):
    smp = tmp_path / "sample.json"
    smp.write_text(json.dumps({"records": recs}))
    d = tmp_path / "smoke"
    d.mkdir()
    (d / "run_config.json").write_text(json.dumps({"code_revision": "abc", "dirty": False, "sample_sha256": "s"}))
    (d / "plans.jsonl").write_text("".join(json.dumps(dict(p, replay_seconds=20.0)) + "\n" for p in plans[:2]))
    (d / "runs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in runs))
    return es.project(argparse.Namespace(sample=str(smp), dirs=str(d)))


def test_project_counts_a_simulation_that_raised_and_assumes_a_probe_time_when_the_smoke_ran_none(tmp_path):
    recs, plans, runs = _case_set()
    rows = [dict(x, wall_seconds=10.0) for x in runs if x["job"][0] == "row"][:4]
    rows.append({"object": "o0", "idx": 0, "job": ["row", ROW], "seed_k": 0, "gate": None, "error": "boom"})
    out = _smoke(tmp_path, recs, plans, rows)
    assert out["errors"] == 1 and out["seconds_per_simulation"]["row"] == pytest.approx(10.0)
    assert out["probe_seconds_assumed"] and out["full"]["projected_cpu_h"] > 0 and out["full"]["unavailable"] == []


def test_project_gives_no_projection_when_no_simulation_time_was_measured(tmp_path):
    recs, plans, runs = _case_set()
    out = _smoke(tmp_path, recs, plans, [{"object": "o0", "idx": 0, "job": ["row", ROW], "seed_k": 0, "gate": None, "error": "boom"}])
    assert out["full"]["projected_cpu_h"] is None and out["full"]["unavailable"] == ["probe", "row"]
    assert not out["probe_seconds_assumed"]


def test_the_summary_refuses_a_simulation_whose_first_menu_differed():
    recs, plans, runs = _case_set()
    runs[-1] = dict(runs[-1], gate=False, error="the first menu differs from the replay's")
    with pytest.raises(SystemExit, match="first menu"):
        es.summarize(recs, plans, runs, n_boot=50)


# ---------------------------------------------------------------- the refresh under a revealed belief

LAW = {"f_damage_n": {"median": 50.0, "log_sd": 0.5}, "k_n_per_mm": {"median": 10.0, "log_sd": 0.4}, "rho": 0.6,
       "v_bruise_ms": {"median": 1.0, "log_sd": 0.3}}
TH = Thresholds(laws={"soft": LAW}, slosh_zeta=0.1)
CRUSH = {"class": "crushable", "law": "soft", "fillable": False, "cavity": None}
POSE = {"object_id": "toy", "pivot": {"centre": [0.0, 0.0, 0.0], "axis": [1.0, 0.0, 0.0], "lo": -0.1, "L": 0.2},
        "com_obj_nominal": [0.0, 0.0, 0.01], "nominal_mass": 0.5}


def test_the_revealed_prior_draws_the_prior_with_the_property_revealed():
    from robots.mujoco_lab.fragility import map_latent
    from robots.mujoco_lab.prior import make_prior

    fn = make_prior({"toy": CRUSH}, TH, generator="v11")
    hidden = {"fill": 0.0, "frag": map_latent((0.3, -1.2, 0.8), CRUSH, TH), "spill_angle_deg": None}
    t = ev.truth_particle(POSE, {"mass_scale": 1.3, "mu": 0.42, "com_s": 0.7}, 1, hidden, CRUSH)
    assert es.revealed_prior(fn, ev.PRIOR, t, CRUSH, TH, 0.2) is fn
    b = es.revealed_prior(fn, "mu", t, CRUSH, TH, 0.2)(POSE, 300, np.random.default_rng(0))
    assert len(b) == 300 and all(p["mu"] == 0.42 for p in b.particles) and b.history == []
    assert len({p["mass"] for p in b.particles}) > 100                 # the other properties keep their prior


def _filled(n, rng):
    fill = np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0.3, 0.9, n))
    parts = [{"mass": 0.3 + 0.2 * f, "mu": 0.5, "com_obj": [0.0, 0.0, 0.02 - 0.02 * f], "fill": float(f), "spill_angle_deg": 40.0 * f,
              "liquid_mass": 0.2 * f, "ancestor": i, **{k: None for k in ev.FRAG_KEYS}, "frag_z": [0.0, 0.0, 0.0]}
             for i, f in enumerate(fill)]
    return ParticleBelief(parts, np.ones(n))


def test_a_coupled_revealed_prior_gives_equal_weight_draws_and_no_reveal_entry():
    entry = {"class": "none", "law": None, "fillable": True, "cavity": None}
    t = dict(_filled(1, np.random.default_rng(5)).particles[0], mass=0.42, fill=0.6, spill_angle_deg=24.0, liquid_mass=0.12,
             com_obj=[0.0, 0.0, 0.008], f_damage_n=None, v_bruise_ms=None, v_break_ms=None)
    b = es.revealed_prior(lambda pose, n, rng, **kw: _filled(n, rng), "damage", t, entry, None, 0.2)(POSE, 4000, np.random.default_rng(1))
    assert len(b) == 4000 and np.allclose(b.weights, b.weights[0]) and b.history == []
    assert all(p["fill"] == 0.6 for p in b.particles) and abs(float(np.mean(b.arrays()["mass"])) - 0.42) < 0.01


def test_the_stores_belief_leaves_out_the_reveal_entries_so_a_refresh_can_replay_its_history():
    b = ParticleBelief([{"mass": 1.0, "mu": 0.5, "com_obj": [0.0, 0.0, 0.0]}], np.ones(1),
                       history=[{"type": "reveal", "keys": ["mass"]}, {"type": "probe", "used_terms": []}])
    assert es.store_belief(b).history == [{"type": "probe", "used_terms": []}]


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


@pytest.mark.skipif(not (SIM and os.path.exists(os.path.join(V12, "folds", "folds.json"))), reason="sim dir or m2b2_v12 bundle missing")
def test_a_probe_under_a_revealed_belief_keeps_the_reveal_through_the_refresh():
    """The one-particle S = all belief refreshes after any probe: under its revealed prior it stays the truth. A coupled
    reveal's belief (a fillable object's damage) updates without a failure and keeps the revealed fill."""
    from robots.mujoco_lab.fragility import entry_of
    from robots.mujoco_lab.prior import fragility_world
    from rpent.skills.pick_place.tools import assess_pick, pick_and_place

    o, idx = "hot3d__mug", 3
    folds = json.load(open(os.path.join(V12, "folds", "folds.json")))
    prims = cal._primitives(os.path.join(V12, "folds", f"fold_{folds[o]}"), None, None)
    prims.n_particles = 300
    world = fragility_world(prims.manifest)
    s = {"seed": 11, "fold": folds[o], "settings": [("k30_A3", 30.0, 3.0)], "flat": None}
    fillable = bool(entry_of(o, world[0])[0].get("fillable"))
    keys = ["all"] + (["damage"] if fillable else [])
    rec = {"prior": "class", "setting": "k30_A3", "care": 30.0, "abstain_cost": 3.0, "fingerprint": None, "n_particles": 300,
           "choice": {k: "decline" for k in (ev.PRIOR,) + ev.REVEALS}, "action": {k: "decline" for k in (ev.PRIOR,) + ev.REVEALS}}
    for k in keys:
        rec["action"][k] = "differs"
    rc = es.replay_case(prims, o, idx, [rec], s, {"class": prims.prior}, world)
    truth = rc["ctx"][("class", "k30_A3", "all")][0].particles[0]
    for key in keys:
        belief, care, A, state = rc["ctx"][("class", "k30_A3", key)]
        prims.prior = rc["prior_of"][("class", key)]
        rc["reset"](es.seeds_of(11, o, idx)[1])
        store = es._store(prims.env, es.store_belief(belief))
        prims.abstain_cost, prims.rng = A, ev._rng_at(state)
        assert "error" not in assess_pick(object="object", destination=rc["destination"], care=care, primitives=prims, store=store).data
        d = pick_and_place(object="object", destination=rc["destination"], care=care, primitives=prims, store=store,
                           option_id="probe_weigh").data
        assert "error" not in d, d
        b = next(iter(store["beliefs"].values()))["belief"]
        assert not [f for f in b.flags if "update failed" in f], b.flags
        if key == "all":
            assert len(b) == 1 and b.particles[0]["mass"] == truth["mass"] and b.particles[0]["mu"] == truth["mu"]
            assert any(f.startswith("refreshed") for f in b.flags)
        else:
            assert all(p["fill"] == truth["fill"] for p in b.particles)
