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

"""Stage 2c and oracle-c (daily-logs designs/2026-11-09-stage2-contacts-rerun-design.md): the published arms again on
the contacts sim with the re-plan at the hold, the same models and the same cases. This module holds the gates of
reading 5 other than the model lock (model_lock.py) and the smoke's 2-seed replay (curve.smoke_summary):

- gate-published: the records of the new code path with the flags off on the published sim equal the published records
  of the same cases in every field but the seconds (the code path is unchanged);
- gate-cases: the new runs' case lists equal the published runs' (each case once) but for exactly the cases of the
  skip list (the megaphone ruling), which every new run carries in its config.

The summary command gives readings 1 to 5 of the design once every unit has ended with EXIT 0 and the gates have passed:
the published Stage 2 readings at the published lock on the Stage 2c records (stage2c/), the oracle summary on oracle-c
against Stage 2c (oracle_c/), and summary.{json,md} with the published, audit-sensitivity and corrected columns, the
case-level change, the calibration and the gates. Every check runs before a new record is opened."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any

from robots.mujoco_lab import curve_oracle as co
from robots.mujoco_lab.curve import (BOOT_DRAWS, BOOT_SEED, EPS, GIVE_UP_ROW, SIDES, _kept, _knob, _pc, _sha, episode, load_stage2,
                                     lock, paired_bootstrap, q1_reading, read_skip_list, records_path, rule_point, stage2_checks,
                                     stage2_figure, stage2_markdown, stage2_readings)


def _plain(rec: dict[str, Any]) -> dict[str, Any]:
    """A record without its seconds (the case's and each simulation's)."""
    out = {k: v for k, v in rec.items() if k != "seconds"}
    out["sims"] = [{k: v for k, v in s.items() if k != "seconds"} for s in rec.get("sims") or []]
    return out


def compare_published(new: list[dict[str, Any]], published: list[dict[str, Any]]) -> dict[str, Any]:
    """Each new record against the published record of its case, every field but the seconds: {"compared", "equal",
    "missing" (new cases the published records lack), "differ" ([{"object", "idx", "fields"}], in the new order)}."""
    pub = {(r["object"], int(r["idx"])): _plain(r) for r in published}
    out: dict[str, Any] = {"compared": 0, "equal": 0, "missing": [], "differ": []}
    for r in new:
        key = (r["object"], int(r["idx"]))
        if key not in pub:
            out["missing"].append([key[0], key[1]])
            continue
        a, b = _plain(r), pub[key]
        out["compared"] += 1
        fields = sorted(k for k in set(a) | set(b) if a.get(k, KeyError) != b.get(k, KeyError))
        if fields:
            out["differ"].append({"object": key[0], "idx": key[1], "fields": fields})
        else:
            out["equal"] += 1
    return out


def _records(dirs: list[str]) -> list[dict[str, Any]]:
    return [json.loads(line) for d in dirs for side in SIDES if os.path.exists(records_path(d, side))
            for line in open(records_path(d, side))]


def _exit0(d: str) -> None:
    p = os.path.join(d, "run.exit")
    if not os.path.exists(p) or open(p).read().split() != ["EXIT", "0"]:
        raise SystemExit(f"{d}: the run did not end with EXIT 0 (run.exit)")


def _write(out: str, name: str, rep: dict[str, Any]) -> dict[str, Any]:
    with open(os.path.join(out, name), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1), flush=True)
    if not rep["pass"]:
        raise SystemExit(3)
    return rep


def _new_out(out: str, name: str) -> None:
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, name)):
        raise SystemExit(f"{os.path.join(out, name)} exists: pick a new --out")


def gate_published(a: argparse.Namespace) -> dict[str, Any]:
    """gate_published.json in --out; exit 3 when a record differs or a case is not in the published records."""
    _new_out(a.out, "gate_published.json")
    runs = a.runs.split(",")
    for d in runs:
        _exit0(d)
    g = compare_published(_records(runs), _records(a.published.split(",")))
    rep = dict(g, runs=a.runs, published=a.published, pass_rule="every case in the published records and equal but for the seconds",
               **{"pass": not g["missing"] and not g["differ"] and g["compared"] > 0})
    return _write(a.out, "gate_published.json", rep)


def _cases(dirs: list[str]) -> list[tuple[str, int]]:
    return [(c[0], int(c[1])) for d in dirs for c in json.load(open(os.path.join(d, "run_config.json")))["cases"]]


def gate_cases(a: argparse.Namespace) -> dict[str, Any]:
    """gate_cases.json in --out; exit 3 when the new runs' cases (each once) differ from the published runs' other than by
    exactly the --skip-list's cases ("skipped": each a published case the new runs lack), or a new run's config holds
    another skip list (none without --skip-list)."""
    _new_out(a.out, "gate_cases.json")
    sl = read_skip_list(a.skip_list)
    listed = {(o, int(i)) for o, i in sl["cases"]} if sl else set()
    runs = a.runs.split(",")
    off = [d for d in runs if json.dumps(json.load(open(os.path.join(d, "run_config.json"))).get("skip_list"), sort_keys=True)
           != json.dumps(sl, sort_keys=True)]
    new, pub = _cases(runs), set(_cases(a.published.split(",")))
    twice = sorted([o, i] for (o, i), n in Counter(new).items() if n > 1)
    lack = pub - set(new)
    missing, extra = sorted([o, i] for o, i in lack - listed), sorted([o, i] for o, i in set(new) - pub)
    present, unpublished = sorted([o, i] for o, i in listed & set(new)), sorted([o, i] for o, i in listed - pub)
    rep = {"cases": len(set(new)), "published_cases": len(pub), "skipped": sorted([o, i] for o, i in lack & listed),
           "missing": missing, "extra": extra, "twice": twice, "skip_list": sl, "skip_list_present": present,
           "skip_list_not_published": unpublished, "runs_off_list": off, "runs": a.runs, "published": a.published,
           "pass": not (missing or extra or twice or present or unpublished or off)}
    return _write(a.out, "gate_cases.json", rep)


def _final(e: dict[str, Any]) -> str:
    return "give_up" if e["give_up"] else e["labels"][-1]


def _outcome(e: dict[str, Any]) -> str:
    return "success" if e["success"] else "irr" if e["irr"] else "give_up" if e["give_up"] else "fail"


def _second(e: dict[str, Any]) -> str | None:
    """The episode's second decision (a row or a give-up), None when it never reached one."""
    if e["attempts"] == 2:
        return e["rows"][1]
    return GIVE_UP_ROW if e["give_up"] and e["attempts"] == 1 else None


def case_change(pub: list[dict[str, Any]], new: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    """Reading 3 against the published records, per decision key, on the cases kept (not dropped) in both: each case's
    episode (curve.episode) before and after; the final label (the last attempt's, or "give_up") changed, by transition
    ("published -> new") and by object; the outcome transitions (success, irr, give_up, fail); the first decisions that
    agree; the second decisions that agree where both episodes reached one; second attempts and give-ups before and
    after; and the cases dropped on one side only, missing from the new records, or not published."""
    P, N = {(r["object"], int(r["idx"])): r for r in pub}, {(r["object"], int(r["idx"])): r for r in new}
    both = sorted(set(P) & set(N))
    kept = [k for k in both if not P[k]["dropped"] and not N[k]["dropped"]]
    sides = {"dropped_new_only": sorted([o, i] for o, i in both if N[(o, i)]["dropped"] and not P[(o, i)]["dropped"]),
             "dropped_published_only": sorted([o, i] for o, i in both if P[(o, i)]["dropped"] and not N[(o, i)]["dropped"]),
             "missing": sorted([o, i] for o, i in set(P) - set(N)), "extra": sorted([o, i] for o, i in set(N) - set(P))}
    out: dict[str, Any] = {}
    for key in keys:
        trans: Counter = Counter()
        outc: Counter = Counter()
        by_obj: dict[str, dict[str, int]] = defaultdict(lambda: {"cases": 0, "changed": 0})
        agree1 = both2 = agree2 = 0
        second = {"published": 0, "new": 0}
        give = {"published": 0, "new": 0}
        for k in kept:
            ep, en = episode(P[k]["decisions"], P[k]["sims"], key), episode(N[k]["decisions"], N[k]["sims"], key)
            by_obj[k[0]]["cases"] += 1
            if _final(ep) != _final(en):
                trans[f"{_final(ep)} -> {_final(en)}"] += 1
                by_obj[k[0]]["changed"] += 1
            outc[f"{_outcome(ep)} -> {_outcome(en)}"] += 1
            agree1 += P[k]["decisions"][key] == N[k]["decisions"][key]
            if _second(ep) is not None and _second(en) is not None:
                both2 += 1
                agree2 += _second(ep) == _second(en)
            for name, e in (("published", ep), ("new", en)):
                second[name] += e["attempts"] == 2
                give[name] += bool(e["give_up"])
        out[key] = {"cases": len(kept), "changed": sum(trans.values()), "by_transition": dict(trans),
                    "by_object": {o: dict(v) for o, v in sorted(by_obj.items())}, "outcomes": dict(outc), "first_agree": agree1,
                    "second_both": both2, "second_agree": agree2, "second_attempts": second, "give_ups": give, **sides}
    return out


# ---------------------------------------------------------------- the summary (readings 1 to 5)

_CODE = re.compile(r"^code (\S+).*, dirty files (\d+)$")
_SIM = re.compile(r"^sim (\S+), last commit ([0-9a-f]{40}), dirty files (\d+)$")
_SHA = re.compile(r"sha256 ([0-9a-f]{64})")
STAGE2_SAME = ("bundle", "seed", "stage1_records_sha256", "retry_sha256", "split_sha256", "n_seeds", "a_main", "r_grid", "delta_grid",
               "quantile", "stage1_care", "sides")
ORACLE_SAME = ("bundle", "seed", "stage1_records_sha256", "split_sha256", "knob", "key", "locked_sha256", "n_seeds", "a_main", "r_grid",
               "stage1_care")
R1_R6 = ("R1", "R2", "R3", "R5", "R6")


def _provenance(d: str) -> dict[str, Any]:
    """The code and sim lines of a run's provenance.txt (launch_s2c.sh): the code revision, the sim dir and commit, and
    their dirty-file counts."""
    out: dict[str, Any] = {}
    for line in open(os.path.join(d, "provenance.txt")):
        line = line.rstrip("\n")
        m = _CODE.match(line)
        if m and "code" not in out:
            out.update(code=m.group(1), code_dirty=int(m.group(2)))
        m = _SIM.match(line)
        if m:
            out.update(sim=m.group(1), sim_commit=m.group(2), sim_dirty=int(m.group(3)))
    if "code" not in out or "sim" not in out:
        raise SystemExit(f"{d}: provenance.txt lacks the code line or the sim line")
    return out


def _cfg(d: str) -> dict[str, Any]:
    return json.load(open(os.path.join(d, "run_config.json")))


def _new_runs(dirs: list[str], a: argparse.Namespace, sl: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The new runs' configs, each run checked: EXIT 0, both executor flags on, the model lock and the skip list of the
    summary, clean code, and the sim of --sim-commit, clean, the one the run's config names."""
    want = _sha(a.model_lock)
    out = []
    for d in dirs:
        _exit0(d)
        c = _cfg(d)
        for flag in ("replan_hold", "plan_without_hand_boxes"):
            if c.get(flag) is not True:
                raise SystemExit(f"{d}: the run's {flag} is not on")
        if c.get("model_lock_sha256") != want:
            raise SystemExit(f"{d}: the run's model lock is not {a.model_lock}")
        if json.dumps(c.get("skip_list"), sort_keys=True) != json.dumps(sl, sort_keys=True):
            raise SystemExit(f"{d}: the run's skip list is not {a.skip_list}")
        pv = _provenance(d)
        if c.get("dirty") is not False or pv["code_dirty"]:
            raise SystemExit(f"{d}: the run's code was dirty")
        if pv["sim_commit"] != a.sim_commit:
            raise SystemExit(f"{d}: the sim commit {pv['sim_commit']} is not --sim-commit {a.sim_commit}")
        if pv["sim_dirty"]:
            raise SystemExit(f"{d}: the sim had {pv['sim_dirty']} dirty files")
        if os.path.realpath(str(c.get("sim_dir"))) != os.path.realpath(pv["sim"]):
            raise SystemExit(f"{d}: the run's sim_dir {c.get('sim_dir')} is not the sim of its provenance, {pv['sim']}")
        out.append(dict(c, _sim=pv["sim"]))
    return out


def _same(name: str, cfgs: list[dict[str, Any]], keys: tuple[str, ...], ref: dict[str, Any] | None = None) -> None:
    """Every config agrees on ``keys`` (and with ``ref``, the published run's, when given)."""
    for k in keys:
        vals = {json.dumps(c.get(k), sort_keys=True) for c in cfgs + ([ref] if ref else [])}
        if len(vals) > 1:
            raise SystemExit(f"{name}: the runs' {k} differs" + (" from the published run's" if ref else ""))


def _split_sides(dirs: list[str]) -> dict[str, list[str]]:
    """The oracle runs by the side their config names (--sides). Refuses a run with records of another side (a run given
    --cases writes each record to its own side's file); only the files' sizes are read."""
    out: dict[str, list[str]] = {side: [] for side in SIDES}
    for d in dirs:
        sides = _cfg(d)["sides"].split(",")
        other = [s for s in SIDES if s not in sides and os.path.exists(records_path(d, s)) and os.path.getsize(records_path(d, s))]
        if other:
            raise SystemExit(f"{d}: the run's config names the side {','.join(sides)}, but it holds {other[0]} records")
        for side in sides:
            out[side].append(d)
    return out


def _side_records(dirs: list[str], side: str) -> list[dict[str, Any]]:
    return [json.loads(line) for d in dirs for line in open(records_path(d, side))]


def _code_revisions(runs: str) -> list[str | None]:
    return sorted({_cfg(d).get("code_revision") if os.path.exists(os.path.join(d, "run_config.json")) else None
                   for d in runs.split(",")}, key=str)


def _gates(a: argparse.Namespace, sl: dict[str, Any] | None) -> dict[str, Any]:
    """Reading 5 before any record: the smoke's 2-seed replay and the oracle smoke's determinism check, the gates on the
    published records (flags off, the published sim), the case-list gates of these runs with this skip list, the model
    lock (the bundle holds the locked files now)."""
    from robots.mujoco_lab import model_lock

    sm = json.load(open(a.smoke_stage2c))
    bad = {k: sm.get(k) for k in ("first_changed", "second_changed", "second_presence_changed", "errors") if sm.get(k) != 0}
    if bad or sm.get("stops"):
        raise SystemExit(f"the Stage 2c smoke failed its gate: {bad or sm['stops']}")
    if not sm.get("first_pairs") or not sm.get("second_triples"):
        raise SystemExit("the Stage 2c smoke compared no first or no second attempts across its two seeds")
    so = json.load(open(a.smoke_oracle))
    det = so["determinism"]
    if not det.get("compared") or det.get("missing") or det["mismatches"] or so.get("errors") or so.get("stops"):
        raise SystemExit(f"the oracle-c smoke failed its gate: {det}, errors {so.get('errors')}, stops {so.get('stops')}")
    arms = {"stage2": set(a.published.split(",")), "oracle": set(a.published_oracle.split(","))}
    gp = []
    for p in a.gate_published.split(","):
        g = json.load(open(p))
        if g.get("pass") is not True:
            raise SystemExit(f"{p}: the gate on the published records did not pass")
        arm = next((k for k, v in arms.items() if set(str(g.get("published")).split(",")) <= v), None)
        if arm is None:
            raise SystemExit(f"{p}: the gate's published records are not the published Stage 2 or oracle runs")
        gp.append({"file": p, "arm": arm, "pass": True, "compared": g["compared"], "equal": g["equal"],
                   "code_revisions": _code_revisions(g["runs"])})
    if {x["arm"] for x in gp} != set(arms):
        raise SystemExit(f"the gates on the published records cover {sorted({x['arm'] for x in gp})}: both arms are needed")
    gc = {}
    for name, p, runs, pub in (("stage2c", a.gate_cases_stage2c, a.runs, a.published),
                               ("oracle_c", a.gate_cases_oracle, a.oracle_runs, a.published_oracle)):
        g = json.load(open(p))
        if (g.get("pass") is not True or g.get("runs") != runs or set(str(g.get("published")).split(",")) != set(pub.split(","))
                or json.dumps(g.get("skip_list"), sort_keys=True) != json.dumps(sl, sort_keys=True)):
            raise SystemExit(f"{p}: the case-list gate did not pass on these runs against these published runs with this skip list")
        gc[name] = {k: g[k] for k in ("cases", "published_cases", "skipped", "pass")}
    bundles = {_cfg(d)["bundle"] for d in a.runs.split(",") + a.oracle_runs.split(",")}
    for b in sorted(bundles):
        model_lock.check(b, a.model_lock)
    return {"smoke_stage2c": {k: sm.get(k) for k in ("first_pairs", "first_changed", "second_triples", "second_changed",
                                                     "second_presence_changed", "errors", "stops", "wandb")},
            "smoke_oracle": {k: so.get(k) for k in ("determinism", "errors", "stops", "wandb")}, "gate_published": gp, "gate_cases": gc,
            "model_lock_sha256": _sha(a.model_lock), "model_lock_bundles": sorted(bundles)}


def _published(a: argparse.Namespace) -> dict[str, Any]:
    """The published summaries, each checked against its inputs: the Stage 2 and oracle records hash as the summaries
    read them, the lock file is the one both read, and the audit's published numbers are the summaries'."""
    ps, po = json.load(open(a.published_summary)), json.load(open(a.published_oracle_summary))
    got = {f"{d} {side}": _sha(records_path(d, side)) for d in a.published.split(",") for side in SIDES}
    if got != ps["provenance"]["records_sha256"]:
        raise SystemExit("the published Stage 2 records are not the ones the published summary read")
    otest = _split_sides(a.published_oracle.split(","))["test"]
    if {d: _sha(records_path(d, "test")) for d in otest} != po["provenance"]["oracle_records_sha256"]:
        raise SystemExit("the published oracle records are not the ones the published oracle summary read")
    for name, text in (("published summary", ps["provenance"]["locked"]), ("published oracle summary", po["provenance"]["locked"])):
        m = _SHA.search(text)
        if not m or m.group(1) != _sha(a.locked):
            raise SystemExit(f"{a.locked} is not the lock file the {name} read")
    au = json.load(open(a.audit))
    p1, o1 = au["published"]["stage2_r0_r4"], au["published"]["oracle_r0"]
    if p1 != ps["q1"]["boot"] or o1 != po["readings"]["1_oracle_vs_r0"]["all"]:
        raise SystemExit("the audit's published numbers are not the published summaries'")
    lv = au["readings"]["3"][a.audit_level]
    if ([lv["stage2_r0_r4"].get(k) for k in ("a", "b", "draws", "seed")] != [p1[k] for k in ("a", "b", "draws", "seed")]
            or [lv["oracle_r0"].get(k) for k in ("draws", "seed")] != [o1[k] for k in ("draws", "seed")]):
        raise SystemExit(f"the audit's {a.audit_level} lines are not at the published keys and bootstrap")
    return {"stage2": ps, "oracle": po, "audit": lv}


def _q1_line(recs: list[dict[str, Any]], k0: str, k4: str) -> dict[str, Any]:
    """Reading 1 on one set of records: curve.paired_bootstrap (all and without thin plastic) and the give-up difference
    and the rates from curve_oracle.paired_outcomes, whose success and irreversible lines must equal the bootstrap's
    (the same objects and draws)."""
    pairs = [(r["object"], episode(r["decisions"], r["sims"], k0), episode(r["decisions"], r["sims"], k4)) for r in _kept(recs)]
    b, o = paired_bootstrap(recs, k0, k4), co.paired_outcomes(pairs)
    if any(o[k] != b[k] for k in ("success", "irr")):
        raise SystemExit("the paired outcomes and the paired bootstrap disagree on the same cases")
    bn = paired_bootstrap([r for r in recs if not r["thin"]], k0, k4)
    return {"boot": b, "no_thin": bn, "give_up": o["give_up"], "rates": {"R0": o["rate_a"], "R4": o["rate_b"]},
            "reading": q1_reading(b), "reading_no_thin": q1_reading(bn)}


def _stage2c_lock(val: list[dict[str, Any]], locked: dict[str, Any], rules: tuple[str, ...]) -> dict[str, Any]:
    """The published lock's knobs with their points on the Stage 2c validation records (no new lock): "qualifies" is
    whether the point's success count is at least R4's on these records."""
    r4 = rule_point(val, locked["R4"]["key"])
    out: dict[str, Any] = {"r4_success": r4["success"], "r4_n_success": r4["n_success"], "n": r4["n"], "from": "the published lock"}
    for rule in rules:
        p = rule_point(val, locked[rule]["key"])
        out[rule] = {"knob": locked[rule]["knob"], "key": locked[rule]["key"], "qualifies": p["n_success"] >= r4["n_success"], "point": p}
    return out


def _hours(dirs: list[str]) -> float:
    """CPU-hours of runs of one thread each: their run_metrics.json wall seconds."""
    return round(sum(json.load(open(os.path.join(d, "run_metrics.json")))["wall_seconds"] for d in dirs) / 3600.0, 3)


def _wandb(dirs: list[str]) -> list[str | None]:
    return [json.load(open(os.path.join(d, "run_metrics.json"))).get("wandb") for d in dirs]


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"the published markdown no longer holds {old!r}")
    return text.replace(old, new)


def summary(a: argparse.Namespace) -> dict[str, Any]:
    """Readings 1 to 5 (module docstring). Refuses a --out that is not empty, a failed gate, a new run that did not end
    with EXIT 0 or lacks a flag, the model lock, the skip list or the sim commit, runs that disagree with each other or
    with the published runs, and published inputs that changed since their summaries."""
    from robots.mujoco_lab.estimation_check import _revision
    from robots.mujoco_lab.tracking import log_run

    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit(f"{a.out} is not empty: pick a new --out")
    sl = read_skip_list(a.skip_list)
    gates = _gates(a, sl)
    runs, oruns = a.runs.split(","), a.oracle_runs.split(",")
    new_s2, new_o = _new_runs(runs, a, sl), _new_runs(oruns, a, sl)
    _same("Stage 2c and oracle-c", new_s2 + new_o, ("code_revision", "sim_dir", "_sim", "bundle", "seed", "stage1_records_sha256",
                                                    "split_sha256"))
    _same("Stage 2c", new_s2, STAGE2_SAME, _cfg(a.published.split(",")[0]))
    _same("Stage 2c", new_s2, ("rules",))
    _same("oracle-c", new_o, ORACLE_SAME, _cfg(a.published_oracle.split(",")[0]))
    oside, posides = _split_sides(oruns), _split_sides(a.published_oracle.split(","))
    pub = _published(a)
    locked = json.load(open(a.locked))["lock"]

    # the records, once: Stage 2c and the published Stage 2, both sides
    val_c, cfg_c = load_stage2(runs, ("validation",))
    test_c, _ = load_stage2(runs, ("test",))
    cv_val, cv_test = stage2_checks(val_c, cfg_c, ("validation",)), stage2_checks(test_c, cfg_c, ("test",))
    checks = {k: cv_val[k] + cv_test[k] for k in cv_val}
    pdirs = a.published.split(",")
    val_p, _ = load_stage2(pdirs, ("validation",))
    test_p, _ = load_stage2(pdirs, ("test",))
    rules = tuple(cfg_c["rules"])
    if "R0" not in rules or "R4" not in rules:
        raise SystemExit(f"the Stage 2c runs hold the rules {rules}: R0 and R4 are needed")
    key = {rule: locked[rule]["key"] for rule in rules}
    k0, k4 = key["R0"], key["R4"]

    # reading 1: the published line from the published records must be the published summary's
    ps = pub["stage2"]
    r1_pub = _q1_line(test_p, k0, k4)
    if r1_pub["boot"] != ps["q1"]["boot"] or r1_pub["no_thin"] != ps["q1_no_thin"]["boot"]:
        raise SystemExit("the published records do not give the published summary's Q1")
    au = pub["audit"]["stage2_r0_r4"]
    r1 = {"keys": [k0, k4], "published": r1_pub, "corrected": _q1_line(test_c, k0, k4),
          "audit": {"level": a.audit_level, "boot": au, "reading": q1_reading(au)}}

    # the published Stage 2 readings at the published lock on the Stage 2c records
    os.makedirs(os.path.join(a.out, "stage2c"), exist_ok=True)
    s2rep: dict[str, Any] = {"rules": rules, "lock": _stage2c_lock(val_c, locked, rules), "checks": checks}
    s2rep.update(stage2_readings(val_c, test_c, rules, key))
    s2rep["provenance"] = {"runs": a.runs, "records_sha256": {f"{d} {side}": _sha(records_path(d, side)) for d in runs for side in SIDES},
                           "code_revision": cfg_c["code_revision"], "stage1": cfg_c["stage1"],
                           "locked": f"{a.locked} (sha256 {_sha(a.locked)}, the published lock; no lock is drawn from these records)",
                           "bootstrap": f"{BOOT_DRAWS} object draws, seed {BOOT_SEED}, 95 % percentile interval", "eps": EPS}
    s2path = os.path.join(a.out, "stage2c", "stage2_summary.json")
    with open(s2path, "w") as f:
        json.dump(s2rep, f, indent=1)
    md = _replace_once(stage2_markdown(s2rep), "# Success-damage curve, Stage 2: the reading (design 6)",
                       "# Stage 2c: the published Stage 2 readings on the contacts sim (design 2026-11-09-stage2-contacts-rerun)")
    md = _replace_once(md, "## 1. Locked operating points (validation objects, locked before the test reading)",
                       "## 1. The published lock's operating points (their points on the Stage 2c validation records; no new lock)")
    with open(os.path.join(a.out, "stage2c", "stage2_summary.md"), "w") as f:
        f.write(md)
    stage2_figure(s2rep, os.path.join(a.out, "stage2c", "stage2_curves.png"))

    # reading 2: the oracle summary on oracle-c against Stage 2c
    oc = co.summary(argparse.Namespace(out=os.path.join(a.out, "oracle_c"), runs=",".join(oside["test"]), locked=a.locked,
                                       stage2=a.runs, stage2_summary=s2path))
    po = pub["oracle"]

    def compact(R: dict[str, Any]) -> dict[str, Any]:
        return {"1_oracle_vs_r0": R["1_oracle_vs_r0"], "2_oracle_vs_r4": R["2_oracle_vs_r4"], "3_reading": R["3_reading"],
                "3_reading_no_thin": R["3_reading_no_thin"], "4_groups": {g: v["n"] for g, v in R["4_groups"]["all"].items()},
                "6_determinism": R["6_determinism"],
                "8_menu_limit": {g: {"n": v["n"], "oracle_rows": v["oracle_rows"], "oracle_vs_r0": v["all"]["oracle_vs_r0"],
                                     "reading": v["all"]["reading"]} for g, v in R["8_menu_limit"].items()}}

    r2 = {"published": compact(po["readings"]), "corrected": compact(oc["readings"]),
          "audit": {"level": a.audit_level, "oracle_vs_r0": pub["audit"]["oracle_r0"]},
          "counts": {"published": po["counts"], "corrected": oc["counts"]},
          "model_curves_equal": oc["readings"]["7_model_curves"] == po["readings"]["7_model_curves"]}

    # reading 3: the case-level change per arm and side
    r3 = {"stage2c": {"validation": case_change(val_p, val_c, list(key.values())), "test": case_change(test_p, test_c, list(key.values()))},
          "oracle_c": {side: case_change(_side_records(posides[side], side), _side_records(oside[side], side), [k0]) for side in SIDES}}

    # reading 4: the calibration of the unchanged model, beside the published
    r4 = {"oracle": {"published": po["readings"]["5_calibration"], "corrected": oc["readings"]["5_calibration"]},
          "stage2c": {rule: {"published": ps["calibration"][rule], "corrected": s2rep["calibration"][rule]} for rule in rules}}

    # exploratory: a lock drawn from the Stage 2c validation records
    re_lk = lock(val_c, rules)
    differs = sorted(r for r in rules if re_lk[r]["knob"] != locked[r]["knob"])
    relock: dict[str, Any] = {"knobs": {r: re_lk[r]["knob"] for r in rules}, "differs": differs, "q1": None}
    if "R0" in differs:
        b = paired_bootstrap(test_c, re_lk["R0"]["key"], re_lk["R4"]["key"])
        relock["q1"] = {"boot": b, "reading": q1_reading(b)}

    me = _revision()
    rep: dict[str, Any] = {
        "rules": list(rules), "r1_r6_ran": all(r in rules for r in R1_R6), "sim_commit": a.sim_commit,
        "readings": {"1": r1, "2": r2, "3": r3, "4": r4, "5": gates}, "exploratory": {"relock": relock},
        "cpu_hours": {"stage2c": _hours(runs), "oracle_c": _hours(oruns), "published_stage2": _hours(pdirs),
                      "published_oracle": _hours(a.published_oracle.split(","))},
        "provenance": {"code_revision": cfg_c["code_revision"], "summary_code_revision": me["code_revision"], "summary_dirty": me["dirty"],
                       "runs": a.runs, "oracle_runs": a.oracle_runs, "published": a.published, "published_oracle": a.published_oracle,
                       "published_summary": f"{a.published_summary} (sha256 {_sha(a.published_summary)})",
                       "published_oracle_summary": f"{a.published_oracle_summary} (sha256 {_sha(a.published_oracle_summary)})",
                       "locked": f"{a.locked} (sha256 {_sha(a.locked)})", "audit": f"{a.audit} (sha256 {_sha(a.audit)}), level {a.audit_level}",
                       "skip_list": sl, "sim_dir": new_s2[0].get("sim_dir"), "sim": new_s2[0]["_sim"],
                       "bootstrap": f"{BOOT_DRAWS} object draws, seed {BOOT_SEED}, 95 % percentile interval", "eps": EPS,
                       "wandb": {"stage2c": _wandb(runs), "oracle_c": _wandb(oruns), "oracle_c_summary": oc.get("wandb")}}}
    rep["wandb"] = log_run("curve-stage2c-summary", {"runs": a.runs, "oracle_runs": a.oracle_runs, "sim_commit": a.sim_commit},
                           {"q1": {"reading": r1["corrected"]["reading"], **{k: r1["corrected"]["boot"][k] for k in ("success", "irr")},
                                   "give_up": r1["corrected"]["give_up"]},
                            "oracle": {"reading": oc["readings"]["3_reading"]}, "r1_r6_ran": rep["r1_r6_ran"]})
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(rep, f, indent=1)
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write(summary_markdown(rep))
    print(f"wrote {a.out}", flush=True)
    return rep


AUDIT_LEVELS = {"any": "any palm or finger depth", "gt1": "depth > 1 mm", "gt2": "depth > 2 mm", "gt5": "depth > 5 mm"}


def _boot_cells(b: dict[str, Any] | None, names: tuple[str, ...]) -> list[str]:
    return ["–" if b is None or b.get(n) is None else co._ci(b, n) for n in names]


def _rate(r: dict[str, Any]) -> str:
    return f"{_pc(r['success'])} / {_pc(r['irr'], 2)} / {_pc(r['give_up'])}"


def summary_markdown(rep: dict[str, Any]) -> str:
    R, pv = rep["readings"], rep["provenance"]
    r1 = R["1"]
    P, C, A = r1["published"], r1["corrected"], r1["audit"]
    lev = f"{A['level']} ({AUDIT_LEVELS.get(A['level'], A['level'])}), {A['boot'].get('excluded', {}).get('flagged', '–')} flagged cases removed"
    L = ["# Stage 2c and oracle-c: readings 1 to 5 (design 2026-11-09-stage2-contacts-rerun)", "",
         f"- Rules that ran: {', '.join(rep['rules'])} (R1 to R6 ran: {'yes' if rep['r1_r6_ran'] else 'no'}). Sim commit {rep['sim_commit']}.",
         f"- Code: runs {pv['code_revision']}, summary {pv['summary_code_revision']} (dirty {pv['summary_dirty']}).",
         f"- CPU-hours (run_metrics wall seconds, one thread per unit): Stage 2c {rep['cpu_hours']['stage2c']:.2f} (published "
         f"{rep['cpu_hours']['published_stage2']:.2f}), oracle-c {rep['cpu_hours']['oracle_c']:.2f} (published "
         f"{rep['cpu_hours']['published_oracle']:.2f}).",
         f"- {pv['bootstrap']}; eps {pv['eps']}. Published columns: {pv['published_summary']}, {pv['published_oracle_summary']}.",
         f"- Audit column: {pv['audit']}.", "",
         "## 1. Stage 2c headline: R0 − R4 on the test objects (points, 95 % object bootstrap)", "",
         f"Keys {r1['keys'][0]} and {r1['keys'][1]} (the published lock).", "",
         f"| quantity | published | audit sensitivity: {lev} | corrected (Stage 2c) |", "|---|---|---|---|",
         f"| cases (objects) | {P['boot']['cases']} ({P['boot']['objects']}) | {A['boot']['cases']} ({A['boot']['objects']}) | "
         f"{C['boot']['cases']} ({C['boot']['objects']}) |"]
    for label, part, name in (("success R0 − R4", "boot", "success"), ("irreversible R0 − R4", "boot", "irr"),
                              ("success R0 − R4 without thin plastic", "no_thin", "success"),
                              ("irreversible R0 − R4 without thin plastic", "no_thin", "irr")):
        a_cell = _boot_cells(A["boot"], (name,))[0] if part == "boot" else "–"
        L.append(f"| {label} | {_boot_cells(P[part], (name,))[0]} | {a_cell} | {_boot_cells(C[part], (name,))[0]} |")
    L += [f"| give-up R0 − R4 | {co._ci({'g': P['give_up']}, 'g')} | – | {co._ci({'g': C['give_up']}, 'g')} |",
          f"| reading | {P['reading']} | {A['reading']} | {C['reading']} |",
          f"| reading without thin plastic | {P['reading_no_thin']} | – | {C['reading_no_thin']} |", "",
          "| arm (test cases kept) | published: success / irreversible / give-up % | corrected |", "|---|---|---|",
          f"| R0 at the locked knob | {_rate(P['rates']['R0'])} | {_rate(C['rates']['R0'])} |",
          f"| R4 | {_rate(P['rates']['R4'])} | {_rate(C['rates']['R4'])} |", ""]
    r2 = R["2"]
    P2, C2, A2 = r2["published"], r2["corrected"], r2["audit"]["oracle_vs_r0"]
    names = ("success", "irr", "give_up")
    L += ["## 2. Oracle-c against Stage 2c (paired test cases; points, 95 % object bootstrap)", "",
          "| quantity | published | audit sensitivity (oracle − R0 only) | corrected |", "|---|---|---|---|",
          f"| paired cases (objects) | {P2['1_oracle_vs_r0']['all']['cases']} ({P2['1_oracle_vs_r0']['all']['objects']}) | "
          f"{A2['cases']} ({A2['objects']}) | {C2['1_oracle_vs_r0']['all']['cases']} ({C2['1_oracle_vs_r0']['all']['objects']}) |"]
    for comp, k in (("oracle − R0", "1_oracle_vs_r0"), ("oracle − R4", "2_oracle_vs_r4")):
        for part, plabel in (("all", ""), ("no_thin", " without thin plastic")):
            for n, nl in zip(names, ("success", "irreversible", "give-up")):
                a_cell = _boot_cells(A2, (n,))[0] if k == "1_oracle_vs_r0" and part == "all" else "–"
                L.append(f"| {comp} {nl}{plabel} | {_boot_cells(P2[k][part], (n,))[0]} | {a_cell} | {_boot_cells(C2[k][part], (n,))[0]} |")
    d_p, d_c = P2["6_determinism"], C2["6_determinism"]
    L += [f"| reading rule (3) | {P2['3_reading']} | – | {C2['3_reading']} |",
          f"| reading rule without thin plastic | {P2['3_reading_no_thin']} | – | {C2['3_reading_no_thin']} |",
          f"| determinism (6): compared, mismatches | {d_p['compared']}, {len(d_p['mismatches'])} | – | {d_c['compared']}, {len(d_c['mismatches'])} |",
          f"| model-only curves (7) equal to the published | – | – | {'yes' if r2['model_curves_equal'] else 'no'} |", "",
          "| arm (paired cases): success / irreversible / give-up % | published | corrected |", "|---|---|---|",
          f"| oracle | {_rate(P2['1_oracle_vs_r0']['all']['rate_a'])} | {_rate(C2['1_oracle_vs_r0']['all']['rate_a'])} |",
          f"| R0 at r 2 | {_rate(P2['1_oracle_vs_r0']['all']['rate_b'])} | {_rate(C2['1_oracle_vs_r0']['all']['rate_b'])} |",
          f"| R4 | {_rate(P2['2_oracle_vs_r4']['all']['rate_b'])} | {_rate(C2['2_oracle_vs_r4']['all']['rate_b'])} |", "",
          "| first decisions (4): group | published cases | corrected cases |", "|---|---|---|"]
    for g in P2["4_groups"]:
        L.append(f"| {g} | {P2['4_groups'][g]} | {C2['4_groups'].get(g, 0)} |")
    L += ["", "| menu-limit group (8) | published: cases, oracle − R0 success, reading | corrected |", "|---|---|---|"]
    for g, name in co.MENU_GROUPS.items():
        p8, c8 = P2["8_menu_limit"][g], C2["8_menu_limit"][g]
        L.append(f"| ({g}) {name} | {p8['n']}, {co._ci(p8['oracle_vs_r0'], 'success')}, {p8['reading']} | "
                 f"{c8['n']}, {co._ci(c8['oracle_vs_r0'], 'success')}, {c8['reading']} |")
    L += ["", "## 3. Case-level change against the published records (cases kept in both)", "",
          "| arm, side, key | cases | final label changed | first decisions agree | second decisions agree (both reached one) | "
          "second attempts: published → corrected | give-ups: published → corrected | dropped in the new run only | not in the new run |",
          "|---|---|---|---|---|---|---|---|---|"]
    for arm in ("stage2c", "oracle_c"):
        for side in SIDES:
            for k, x in R["3"][arm][side].items():
                L.append(f"| {arm}, {side}, {k} | {x['cases']} | {x['changed']} ({_pc(x['changed'] / x['cases'] if x['cases'] else None)} %) | "
                         f"{x['first_agree']} | {x['second_agree']} of {x['second_both']} | {x['second_attempts']['published']} → "
                         f"{x['second_attempts']['new']} | {x['give_ups']['published']} → {x['give_ups']['new']} | "
                         f"{len(x['dropped_new_only'])} | {len(x['missing'])} |")
    for arm, k in (("stage2c", r1["keys"][0]), ("stage2c", r1["keys"][1]), ("oracle_c", r1["keys"][0])):
        x = R["3"][arm]["test"][k]
        L += ["", f"Test side, {arm} at {k}: final labels by transition, then the objects with a change.", "",
              "| transition | cases |", "|---|---|"]
        L += [f"| {t} | {n} |" for t, n in sorted(x["by_transition"].items(), key=lambda t: (-t[1], t[0]))]
        L += ["", "| object | changed of cases |", "|---|---|"]
        L += [f"| {o} | {v['changed']} of {v['cases']} |" for o, v in sorted(x["by_object"].items(), key=lambda t: (-t[1]["changed"], t[0]))
              if v["changed"]]
    r4 = R["4"]
    L += ["", "## 4. Calibration of the unchanged model against the corrected labels", "",
          "| oracle attempts at the truth | published: attempts, predicted sum, observed, Brier | corrected |", "|---|---|---|"]
    for part in ("all", "no_thin"):
        for stat in ("placed", "irr"):
            x, y = r4["oracle"]["published"][part][stat], r4["oracle"]["corrected"][part][stat]
            cell = lambda z: f"{z['n']}, {z['predicted_sum']:.1f}, {z['observed_sum']}, {'–' if z['brier'] is None else round(z['brier'], 4)}"  # noqa: E731
            L.append(f"| P({stat}), {'all' if part == 'all' else 'without thin plastic'} | {cell(x)} | {cell(y)} |")
    L += ["", "| Stage 2c rule at the locked point: P(irr) | published: attempts, predicted sum, observed | corrected |", "|---|---|---|"]
    for rule, v in r4["stage2c"].items():
        x, y = v["published"]["all"], v["corrected"]["all"]
        L.append(f"| {rule} | {x['attempts']}, {x['predicted']:.2f}, {x['observed']} | {y['attempts']}, {y['predicted']:.2f}, {y['observed']} |")
    g = R["5"]
    sm, so = g["smoke_stage2c"], g["smoke_oracle"]
    L += ["", "## 5. Gates (all passed; the summary refuses a failed one)", "",
          f"- 2-seed replay on the Stage 2c smoke: {sm['first_pairs']} (case, first row) pairs, {sm['first_changed']} changed; "
          f"{sm['second_triples']} second attempts, {sm['second_changed']} changed, {sm['second_presence_changed']} present at one seed only; "
          f"errors {sm['errors']}.",
          f"- Oracle-c smoke against the Stage 2c smoke (determinism): {so['determinism']['compared']} compared, "
          f"{len(so['determinism']['mismatches'])} mismatches."]
    L += [f"- Published records with the flags off: {x['file']}: {x['equal']} of {x['compared']} equal (code {', '.join(map(str, x['code_revisions']))})."
          for x in g["gate_published"]]
    L += [f"- Case lists, {name}: {x['cases']} cases against {x['published_cases']} published, skipped {x['skipped']}."
          for name, x in g["gate_cases"].items()]
    L += [f"- Model lock: sha256 {g['model_lock_sha256']} in every run; the bundle held the locked files at the summary."]
    x = rep["exploratory"]["relock"]
    L += ["", "## Exploratory (not a pre-registered reading): a lock drawn from the Stage 2c validation records", "",
          f"Knobs {', '.join(f'{r} {_knob(k)}' for r, k in x['knobs'].items())}; different from the published lock: "
          f"{', '.join(x['differs']) or 'none'}."
          + (f" Q1 at the drawn R0 knob: success {co._ci(x['q1']['boot'], 'success')}, irreversible {co._ci(x['q1']['boot'], 'irr')}, "
             f"{x['q1']['reading']}." if x["q1"] else "")]
    L += ["", "## wandb", ""] + [f"- {k}: {v}" for k, v in pv["wandb"].items()] + [f"- this summary: {rep['wandb']}"]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> Any:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate-published", help="new records equal the published ones but for the seconds")
    g.add_argument("--runs", required=True, help="comma-separated new run dirs (flags off, the published sim)")
    g.add_argument("--published", required=True, help="comma-separated published run dirs")
    g.add_argument("--out", required=True)
    c = sub.add_parser("gate-cases", help="the new runs' case lists equal the published runs'")
    c.add_argument("--runs", required=True)
    c.add_argument("--published", required=True)
    c.add_argument("--skip-list", default=None, help="the runs' skip list: its cases are the only published ones they may lack")
    c.add_argument("--out", required=True)
    m = sub.add_parser("summary", help="readings 1 to 5 once every unit ended EXIT 0 and the gates passed")
    m.add_argument("--runs", required=True, help="comma-separated Stage 2c run dirs")
    m.add_argument("--oracle-runs", required=True, help="comma-separated oracle-c run dirs (test and validation)")
    m.add_argument("--published", required=True, help="the published Stage 2 run dirs, as the published summary names them")
    m.add_argument("--published-summary", required=True, help="the published stage2_summary.json")
    m.add_argument("--published-oracle", required=True, help="the published oracle run dirs (test and validation)")
    m.add_argument("--published-oracle-summary", required=True, help="the published oracle_summary.json")
    m.add_argument("--locked", required=True, help="the published lock file (locked.json)")
    m.add_argument("--audit", required=True, help="the palm-contact audit's audit_summary.json")
    m.add_argument("--audit-level", default="gt1", help="the audit's sensitivity level for the audit column")
    m.add_argument("--model-lock", required=True)
    m.add_argument("--skip-list", required=True)
    m.add_argument("--sim-commit", required=True, help="the contacts sim's commit (each run's provenance.txt)")
    m.add_argument("--smoke-stage2c", required=True, help="the Stage 2c smoke_summary.json (the 2-seed replay)")
    m.add_argument("--smoke-oracle", required=True, help="the oracle-c smoke_summary.json (determinism against the Stage 2c smoke)")
    m.add_argument("--gate-published", required=True, help="comma-separated gate_published.json files")
    m.add_argument("--gate-cases-stage2c", required=True)
    m.add_argument("--gate-cases-oracle", required=True)
    m.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"gate-published": gate_published, "gate-cases": gate_cases, "summary": summary}[a.cmd](a)


if __name__ == "__main__":
    main()
