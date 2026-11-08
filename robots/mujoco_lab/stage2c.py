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
- gate-cases: the new runs' case lists equal the published runs' (each case once)."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any

from robots.mujoco_lab.curve import SIDES, records_path


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
    """gate_cases.json in --out; exit 3 when the new runs' cases (each once) differ from the published runs'."""
    _new_out(a.out, "gate_cases.json")
    new, pub = _cases(a.runs.split(",")), set(_cases(a.published.split(",")))
    twice = sorted([o, i] for (o, i), n in Counter(new).items() if n > 1)
    missing, extra = sorted([o, i] for o, i in pub - set(new)), sorted([o, i] for o, i in set(new) - pub)
    rep = {"cases": len(set(new)), "published_cases": len(pub), "missing": missing, "extra": extra, "twice": twice,
           "runs": a.runs, "published": a.published, "pass": not (missing or extra or twice)}
    return _write(a.out, "gate_cases.json", rep)


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
    c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"gate-published": gate_published, "gate-cases": gate_cases}[a.cmd](a)


if __name__ == "__main__":
    main()
