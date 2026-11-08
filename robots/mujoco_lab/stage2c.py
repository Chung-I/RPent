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
  skip list (the megaphone ruling), which every new run carries in its config."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any

from robots.mujoco_lab.curve import GIVE_UP_ROW, SIDES, episode, read_skip_list, records_path


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
    a = ap.parse_args(argv)
    return {"gate-published": gate_published, "gate-cases": gate_cases}[a.cmd](a)


if __name__ == "__main__":
    main()
