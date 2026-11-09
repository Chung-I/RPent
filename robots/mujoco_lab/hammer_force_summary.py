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

"""The hammer squeeze x CoM-pinch summary (daily-logs design 2026-11-09-hammer-force-com-design.md, approved by the user
on 9 Nov; the coordinator's rulings of 9 Nov). It runs once, after the gates.

Gates before the open (no record is read): every unit of the run and of the close monitor pass EXIT 0; their configs
(the re-plan at the hold on, the contacts sim, no dirty file, the cases file's sha256, the pass's flag) and metrics (no
stop, no G0 mismatch); the run's code is the smoke's; the smoke's G0 and G1 are clean; the configs cover the cases file.
Then each records file is opened once (_read_jsonl). Gates after the open (a failure writes STOP.json with names and
counts, no label, and exits 3): coverage (every case once, eight rows, every feasible row simulated, in both the run
and the pass), G0 again (the control row against the corrected pose-set record of the cases file), every artifact
flag explained by the contact monitor, the hash join (qpos_sha and trace_sha of every simulated row equal in the pass),
then the squeeze gate (ruling 6: the median measured squeeze per pad at 70 N within 20 % of the command, read in the
hold when 10 or more rows have a hold reading, else in the late close, with its n).

Rows and cases (rulings 2-4 and B): a row's state is infeasible, error, artifact (reversed contacts, a reset not at
rest or a jam at any level; a pad penetration over PEN_MM at f_std and 2 f_std only: at 40 N and 70 N it is a monitor
field, ruling B), its motion label, or "damage" for any other label. Placed: placed_stable. Lifted: any state of a
clean row but not_lifted (ruling 2; infeasible is neither, ruling 3). A case with an artifact or an error in any row
drops from every reading (ruling 4, with B); the sensitivity line drops the cases ruling 4 as written drops (the run's
artifact flag, which includes the penetration at every level).

Readings (the design's numbering): paired bootstrap over cases (DRAWS, SEED, 95 % percentile); verdicts against
MARGINS (supported: the whole interval past the margin; not supported: the whole interval below it; else undecided at
n = N). Lines: the 44 test cases (headline), the same with the penetration-flagged cases dropped, the validation cases,
and the pooled cases (exploratory)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from typing import Any

import numpy as np

from robots.mujoco_lab import curve_hammer_force as hf
from robots.mujoco_lab.adapter import DEEP_M, REV_STEPS

PEN_MM = 1000 * DEEP_M                 # the contact monitor's penetration limit (mm), set for the commit_std squeeze
PEN_LEVELS = ("40N", "70N")            # ruling B: at these levels a penetration over PEN_MM is a monitor field
SHIFT_MM = 5.0                         # the close monitor's count: rows whose object shifted more in the close
MARGINS = {"placed": 10.0, "lifted": 20.0}
DRAWS, SEED = 2000, 20261109           # the bootstrap's draws and stream (fixed before the one open)
SPLIT_KG = 1.0
SQUEEZE_TOL, SQUEEZE_MIN_N = 0.20, 10  # ruling 6
MOTION = ("placed_stable", "tipped", "rolled_off", "dropped_low", "dropped_high", "missed", "not_lifted")
STATES = MOTION + ("damage", "infeasible", "artifact", "error")
CELLS = tuple(hf.row_id(p, lv) for p in hf.PINCHES for lv in hf.LEVELS)
LINES = ("test", "test_penetration_dropped", "validation", "pooled")
LINE_NAMES = {"test": "test cases (headline)", "test_penetration_dropped": "test cases, penetration-flagged cases dropped (ruling 4 as written)",
              "validation": "validation cases", "pooled": "pooled cases (exploratory)"}


# ---------------------------------------------------------------- rows and cases

def causes(r: dict[str, Any]) -> dict[str, bool]:
    """The contact monitor's artifact causes of a simulated row (adapter.artifact's rule, the jam's flag)."""
    m = r.get("monitor") or {}
    return {"reversed": int(m.get("rev_steps") or 0) >= REV_STEPS, "deep": float(m.get("deepest_mm") or 0.0) > PEN_MM,
            "not_at_rest": m.get("at_rest") is False, "jam": bool(m.get("jam_artifact"))}


def row_state(r: dict[str, Any]) -> str:
    """infeasible, error, artifact (ruling B: the penetration is not one at 40 N and 70 N), the motion label, or
    damage (any other label: hammers carry no damage law)."""
    if not r["feasible"]:
        return "infeasible"
    if r.get("error") or "seconds" not in r:
        return "error"
    c = causes(r)
    if c["reversed"] or c["not_at_rest"] or c["jam"] or (c["deep"] and r["level"] not in PEN_LEVELS):
        return "artifact"
    lab = r.get("label")
    if lab is None:
        return "error"
    return lab if lab in MOTION else "damage"


def case_view(rec: dict[str, Any], entry: dict[str, Any], mon: dict[str, Any] | None) -> dict[str, Any]:
    """A simulated case: its keys, mass (the sim's truth), the published comparators (ceiling10, r4), the primary drop
    (the cells with an artifact or an error under ruling B), ``drop_a`` (ruling 4 as written: the run's artifact flag or
    an error in any row), and per cell the state, placed, lifted, lifted_strict (missed counts as a lift failure), the
    penetration flag and depth, the reason of an infeasible row and the close monitor's fields (None when unsimulated)."""
    mrows = {r["id"]: r for r in (mon or {}).get("rows", [])}
    cells, drop, drop_a = {}, [], False
    for r in rec["rows"]:
        s = row_state(r)
        clean = s not in ("infeasible", "artifact", "error")
        deep = float((r.get("monitor") or {}).get("deepest_mm") or 0.0) if "seconds" in r else None
        cells[r["id"]] = {"state": s, "placed": s == "placed_stable", "lifted": clean and s != "not_lifted",
                          "lifted_strict": clean and s not in ("not_lifted", "missed"), "pen": deep is not None and deep > PEN_MM,
                          "deep_mm": deep, "why": r.get("why"), "level": r["level"], "pinch": r["pinch"],
                          "close": mrows.get(r["id"], {}).get("close") if "seconds" in r else None}
        if s in ("artifact", "error"):
            drop.append(f"{r['id']}: {s}")
        drop_a = drop_a or bool(r.get("artifact")) or s == "error"
    if rec.get("error") or rec.get("stop"):
        drop.append("case: error or stop")
        drop_a = True
    return {"object": rec["object"], "idx": int(rec["idx"]), "side": rec["side"],
            "mass": float(rec["sim_truth"]["mass"]),
            "ceiling10": entry.get("ceiling10"), "r4": entry.get("r4"), "drop": drop, "drop_a": drop_a, "cells": cells}


# ---------------------------------------------------------------- the bootstrap and the verdicts

def _boot(x: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    b = x[rng.integers(0, len(x), size=(DRAWS, len(x)))].mean(axis=1) * 100.0
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(lo), float(hi)


def share(xs: list[bool]) -> dict[str, Any]:
    """A share in percent with its 95 % percentile bootstrap interval over cases."""
    if not xs:
        return {"n": 0, "k": 0, "share": None, "lo": None, "hi": None}
    x = np.asarray(xs, dtype=float)
    lo, hi = _boot(x)
    return {"n": len(xs), "k": int(x.sum()), "share": float(100.0 * x.mean()), "lo": lo, "hi": hi}


def paired(a: list[bool], b: list[bool]) -> dict[str, Any]:
    """The paired difference a - b in points (the same cases in both), with its interval; ``a`` and ``b`` the shares."""
    if not a:
        return {"n": 0, "diff": None, "lo": None, "hi": None, "a": None, "b": None}
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    lo, hi = _boot(d)
    return {"n": len(a), "diff": float(100.0 * d.mean()), "lo": lo, "hi": hi, "a": float(100.0 * np.mean(a)), "b": float(100.0 * np.mean(b))}


def verdict(p: dict[str, Any], margin: float) -> str | None:
    if not p["n"]:
        return None
    if p["lo"] > margin:
        return "supported"
    if p["hi"] < margin:
        return "not supported"
    return f"undecided at n = {p['n']}"


def _pv(vs: list[dict[str, Any]], a: str, b: str, out: str) -> dict[str, Any]:
    p = paired([v["cells"][a][out] for v in vs], [v["cells"][b][out] for v in vs])
    return dict(p, verdict=verdict(p, MARGINS["lifted" if out.startswith("lifted") else "placed"]))


# ---------------------------------------------------------------- the readings

def reading1(vs: list[dict[str, Any]], strict: bool = False) -> dict[str, Any]:
    """Force at the CoM pinch: the shares per level and 70 N, 40 N and 2 f_std each against f_std (70 N first: the
    pre-specified primary, ruling 8). ``strict``: missed counts as a lift failure (ruling 2's sensitivity line)."""
    lift = "lifted_strict" if strict else "lifted"
    com = {lv: hf.row_id("across_com", lv) for lv in hf.LEVELS}
    return {"primary": "70N - f_std",
            "shares": {lv: {"placed": share([v["cells"][c]["placed"] for v in vs]), "lifted": share([v["cells"][c][lift] for v in vs])}
                       for lv, c in com.items()},
            "diffs": {f"{lv} - f_std": {"placed": _pv(vs, com[lv], com["f_std"], "placed"), "lifted": _pv(vs, com[lv], com["f_std"], lift)}
                      for lv in ("70N", "40N", "2f_std")}}


def reading2(vs: list[dict[str, Any]]) -> dict[str, Any]:
    """The pinch at each force: across_com - across_0.5 at f_std and at 70 N."""
    return {lv: {out: _pv(vs, hf.row_id("across_com", lv), hf.row_id("across_0.5", lv), out) for out in ("placed", "lifted")}
            for lv in ("f_std", "70N")}


def reading3(vs: list[dict[str, Any]]) -> dict[str, Any]:
    """By hidden mass (descriptive): every cell's shares in the light (< SPLIT_KG) and the heavy band, n per band."""
    out = {}
    for band, keep in (("light", lambda m: m < SPLIT_KG), ("heavy", lambda m: m >= SPLIT_KG)):
        b = [v for v in vs if keep(v["mass"])]
        out[band] = {"n": len(b), "cells": {c: {"placed": share([v["cells"][c]["placed"] for v in b]),
                                                "lifted": share([v["cells"][c]["lifted"] for v in b])} for c in CELLS}}
    return out


def best_cell(vs: list[dict[str, Any]]) -> str:
    """Ruling 8: the highest placed share on the test cases; a tie goes to the lower squeeze, then to across_0.5."""
    def key(c: str) -> tuple[float, int, int]:
        p, lv = c.split("@")
        return (-float(np.mean([v["cells"][c]["placed"] for v in vs])) if vs else 0.0, hf.LEVELS.index(lv), hf.PINCHES.index(p))
    return min(CELLS, key=key)


def reading7(vs: list[dict[str, Any]], cell: str) -> dict[str, Any]:
    """The best cell's placed share against the corrected pose-set ceiling (ceiling10) and Stage 2c's R4, paired over
    the cases where the comparator is known."""
    out: dict[str, Any] = {"cell": cell, "share": share([v["cells"][cell]["placed"] for v in vs])}
    for k in ("ceiling10", "r4"):
        w = [v for v in vs if v[k] is not None]
        p = paired([v["cells"][cell]["placed"] for v in w], [bool(v[k]) for v in w])
        out[f"vs_{k}"] = dict(p, verdict=verdict(p, MARGINS["placed"]))
    return out


def breakdown(vs: list[dict[str, Any]], cell: str) -> dict[str, int]:
    """Reading 4: the states of one cell over every simulated case of the side (before the drop)."""
    c = Counter(v["cells"][cell]["state"] for v in vs)
    return {s: int(c.get(s, 0)) for s in STATES}


def damage_counts(vs: list[dict[str, Any]]) -> dict[str, int]:
    """Reading 6: rows with a damage label per cell (a defect of the setting: hammers carry no damage law)."""
    return {c: sum(v["cells"][c]["state"] == "damage" for v in vs) for c in CELLS}


def per_cell(vs: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Per cell over every simulated case of the side: cases, simulated rows, infeasible, artifact, error, the
    penetration flags (over PEN_MM), missed, damage, and the cases this cell's row drops (ruling 4 with B)."""
    out = {}
    for c in CELLS:
        st = [v["cells"][c]["state"] for v in vs]
        out[c] = {"cases": len(vs), "simulated": sum(s != "infeasible" for s in st), "infeasible": st.count("infeasible"),
                  "artifact": st.count("artifact"), "error": st.count("error"), "penetration": sum(v["cells"][c]["pen"] for v in vs),
                  "missed": st.count("missed"), "damage": st.count("damage"), "drops": sum(s in ("artifact", "error") for s in st)}
    return out


def infeasible_by_object(vs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Ruling 3: the cases per object whose across_com pinch is infeasible, with the reasons (cases per reason)."""
    out: dict[str, dict[str, Any]] = {}
    for v in vs:
        why = {v["cells"][c]["why"] for c in CELLS if c.startswith("across_com@") and v["cells"][c]["state"] == "infeasible"}
        if why:
            o = out.setdefault(v["object"], {"cases": 0, "why": {}})
            o["cases"] += 1
            for w in why:
                o["why"][w] = o["why"].get(w, 0) + 1
    return out


def unpaired(vs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Ruling 4's addition: per cell, the shares over every case whose row in that cell is clean or infeasible
    (unpaired across cells)."""
    out = {}
    for c in CELLS:
        w = [v for v in vs if v["cells"][c]["state"] not in ("artifact", "error")]
        out[c] = {"placed": share([v["cells"][c]["placed"] for v in w]), "lifted": share([v["cells"][c]["lifted"] for v in w])}
    return out


def wedge(vs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Ruling B, additions 1 and 3 (diagnostic): in each 40 N and 70 N cell, the simulated rows over PEN_MM beside
    those under it, lifted and placed (counts and shares)."""
    def part(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n, li, pl = len(rows), sum(r["lifted"] for r in rows), sum(r["placed"] for r in rows)
        return {"rows": n, "lifted": li, "placed": pl, "lifted_share": 100.0 * li / n if n else None,
                "placed_share": 100.0 * pl / n if n else None}
    out = {}
    for c in CELLS:
        if c.split("@")[1] in PEN_LEVELS:
            rows = [v["cells"][c] for v in vs if v["cells"][c]["deep_mm"] is not None and v["cells"][c]["state"] not in ("artifact", "error")]
            out[c] = {"deep": part([r for r in rows if r["pen"]]), "shallow": part([r for r in rows if not r["pen"]])}
    return out


def close_report(vs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The coordinator's close monitor per cell over the simulated rows: the object's shift in the close (median, max,
    rows over SHIFT_MM) and the turn in the pads at the lift's end (rows with a value, median)."""
    out = {}
    for c in CELLS:
        cl = [v["cells"][c]["close"] for v in vs if v["cells"][c]["close"] is not None]
        sh = [x["close_shift_mm"] for x in cl if x.get("close_shift_mm") is not None]
        tu = [x["lift_end_turn_deg"] for x in cl if x.get("lift_end_turn_deg") is not None]
        out[c] = {"rows": len(cl), "shift_median_mm": float(np.median(sh)) if sh else None, "shift_max_mm": float(max(sh)) if sh else None,
                  "over_5mm": sum(x > SHIFT_MM for x in sh), "turn_rows": len(tu), "turn_median_deg": float(np.median(tu)) if tu else None}
    return out


def penetration(recs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The deepest pad penetration per level over every simulated row (a monitor field, no outcome)."""
    out = {}
    for lv in hf.LEVELS:
        d = [float(r["monitor"]["deepest_mm"]) for rec in recs for r in rec["rows"] if r["level"] == lv and "seconds" in r and r.get("monitor")]
        k = sum(x > PEN_MM for x in d)
        out[lv] = {"rows": len(d), "median_mm": float(np.median(d)) if d else None, "max_mm": float(max(d)) if d else None,
                   "over_3mm": k, "share_over_3mm": 100.0 * k / len(d) if d else None}
    return out


def squeeze_gate(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """Reading 5 and ruling 6: per level the median measured squeeze per pad in the hold and in the late close (with
    n); the gate at 70 N reads the hold with SQUEEZE_MIN_N or more readings, else the close, and passes within
    SQUEEZE_TOL of the command."""
    levels = {}
    for lv in hf.LEVELS:
        sq = [r["squeeze"] for rec in recs for r in rec["rows"] if r["level"] == lv and "seconds" in r and not r.get("error") and r.get("squeeze")]
        h = [s["hold"] for s in sq if s.get("hold") is not None]
        c = [s["close"] for s in sq if s.get("close") is not None]
        levels[lv] = {"hold_median": float(np.median(h)) if h else None, "hold_n": len(h),
                      "close_median": float(np.median(c)) if c else None, "close_n": len(c)}
    g = levels["70N"]
    if g["hold_n"] >= SQUEEZE_MIN_N:
        reader, n, med, note = "hold", g["hold_n"], g["hold_median"], None
    else:
        reader, n, med = "close", g["close_n"], g["close_median"]
        note = f"fewer than {SQUEEZE_MIN_N} rows at 70 N have a hold reading ({g['hold_n']}): the gate reads the late close"
    ok = med is not None and abs(med - 70.0) <= SQUEEZE_TOL * 70.0
    return {"levels": levels, "gate": {"reader": reader, "n": n, "median": med, "command": 70.0, "tolerance": SQUEEZE_TOL, "ok": ok, "note": note}}


# ---------------------------------------------------------------- the gates after the open

def post_gates(recs: list[dict[str, Any]], mon: dict[tuple[str, int], dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Coverage, G0 again, the artifact flags, the hash join. ``problems`` names cases, rows, fields and counts, never a
    label or a value."""
    problems: list[str] = []
    want = {(e["object"], int(e["idx"])): e for e in cases}
    seen = Counter((r["object"], int(r["idx"])) for r in recs)
    problems += [f"case {k[0]} {k[1]} is in the records twice" for k, n in seen.items() if n > 1]
    problems += [f"case {k[0]} {k[1]} is missing from the records" for k in want if k not in seen]
    problems += [f"case {k[0]} {k[1]} is not in the cases file" for k in seen if k not in want]
    problems += [f"case {k[0]} {k[1]} of the monitor pass is not in the cases file" for k in mon if k not in want]
    g0 = {"compared": 0, "mismatches": 0}
    flags = 0
    hj = {"rows": 0, "mismatches": 0}
    for rec in recs:
        k = (rec["object"], int(rec["idx"]))
        e = want.get(k)
        if e is None:
            continue
        if bool(rec.get("skipped")) != bool(e.get("skipped")) or rec["side"] != e["side"]:
            problems.append(f"case {k[0]} {k[1]}: side or skip differs from the cases file")
        if rec.get("skipped"):
            continue
        mass = (rec.get("sim_truth") or {}).get("mass")
        if mass is None or e.get("mass") is None or abs(float(mass) - float(e["mass"])) > 1e-6:
            problems.append(f"case {k[0]} {k[1]}: the simulated mass is not the cases file's")
        ids = [r["id"] for r in rec["rows"]]
        if ids != list(CELLS):
            problems.append(f"case {k[0]} {k[1]}: {len(ids)} rows, not the eight cells in order")
            continue
        if any(r["feasible"] and "seconds" not in r for r in rec["rows"]):
            problems.append(f"case {k[0]} {k[1]}: a feasible row was not simulated")
        ctl = rec["rows"][0]
        diff = hf.g0_diff(ctl, e["ref"] or {}) if "seconds" in ctl else list(hf.G0_FIELDS)
        g0["compared"] += 1
        if diff or rec.get("g0"):
            g0["mismatches"] += 1
            problems.append(f"G0: case {k[0]} {k[1]}: the control row differs from the corrected pose-set record in {', '.join(diff or rec['g0'])}")
        for r in rec["rows"]:
            if "seconds" in r and not r.get("error") and bool(r.get("artifact")) != any(causes(r).values()):
                flags += 1
        m = mon.get(k)
        if m is None:
            problems.append(f"case {k[0]} {k[1]} is missing from the monitor pass")
            continue
        mrows = {r["id"]: r for r in m["rows"]}
        for r in rec["rows"]:
            if "seconds" not in r:
                continue
            hj["rows"] += 1
            x = mrows.get(r["id"])
            if x is None or "seconds" not in x or x.get("qpos_sha") != r.get("qpos_sha") or x.get("trace_sha") != r.get("trace_sha"):
                hj["mismatches"] += 1
                problems.append(f"hash join: case {k[0]} {k[1]} row {r['id']}: qpos_sha or trace_sha differs in the monitor pass")
    if flags:
        problems.append(f"{flags} rows carry an artifact flag that the contact monitor does not explain (or the reverse)")
    return {"problems": problems, "g0": g0, "artifact_flags": {"unexplained": flags}, "hash_join": hj}


# ---------------------------------------------------------------- the report

def _views(recs: list[dict[str, Any]], mon: dict[tuple[str, int], dict[str, Any]], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    want = {(e["object"], int(e["idx"])): e for e in cases}
    return [case_view(r, want[(r["object"], int(r["idx"]))], mon.get((r["object"], int(r["idx"])))) for r in recs if not r.get("skipped")]


def _line(vs: list[dict[str, Any]], cell: str, exploratory: bool = False) -> dict[str, Any]:
    return {"n": len(vs), "exploratory": exploratory, "r1": reading1(vs), "r1_missed": reading1(vs, strict=True), "r2": reading2(vs),
            "r3": reading3(vs), "r7": reading7(vs, cell)}


def build(recs: list[dict[str, Any]], mon: dict[tuple[str, int], dict[str, Any]], cases: list[dict[str, Any]],
          pre: dict[str, Any]) -> dict[str, Any]:
    """The report: the lines (test, the test with the penetration-flagged cases dropped, validation, pooled), the best
    cell, readings 4-6, the per-cell tables, the unpaired lines (more than 5 test cases dropped), the monitors."""
    allv = _views(recs, mon, cases)
    side = {s: [v for v in allv if v["side"] == s] for s in ("test", "validation")}
    keep = {s: [v for v in side[s] if not v["drop"]] for s in side}
    keep_a = [v for v in side["test"] if not v["drop"] and not v["drop_a"]]
    best = best_cell(keep["test"])
    lines = {"test": _line(keep["test"], best), "test_penetration_dropped": _line(keep_a, best),
             "validation": _line(keep["validation"], best), "pooled": _line(keep["test"] + keep["validation"], best, exploratory=True)}
    dropped = {s: len(side[s]) - len(keep[s]) for s in side}
    force = {lv: float(np.median([r["force"] for rec in recs for r in rec["rows"] if r["level"] == lv])) if recs else None for lv in hf.LEVELS}
    return {"pre": pre, "n": {s: len(side[s]) for s in side}, "dropped": dropped,
            "dropped_penetration_test": len(side["test"]) - len(keep_a), "lines": lines,
            "best": {"cell": best, "rule": "the highest placed share on the test cases; a tie goes to the lower squeeze, then to across_0.5 "
                                            "(ruling 8); across_com x 70 N stays the primary of reading 1"},
            "r4": {s: {"best": breakdown(side[s], best), "control": breakdown(side[s], hf.CONTROL)} for s in side},
            "r6": {s: damage_counts(side[s]) for s in side},
            "per_cell": {s: per_cell(side[s]) for s in side},
            "infeasible_com_by_object": infeasible_by_object(allv),
            "unpaired": unpaired(side["test"]) if dropped["test"] > 5 else None,
            "wedge": {s: wedge(keep[s]) for s in side}, "close": {s: close_report(side[s]) for s in side},
            "penetration": penetration(recs), "squeeze": squeeze_gate(recs), "force_by_level": force,
            "settings": {"draws": DRAWS, "seed": SEED, "margins": MARGINS, "split_kg": SPLIT_KG, "pen_mm": PEN_MM, "pen_levels": PEN_LEVELS,
                         "shift_mm": SHIFT_MM}}


# ---------------------------------------------------------------- markdown

def _n(x: float | None, nd: int = 1) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def _sh(s: dict[str, Any]) -> str:
    return "-" if s["share"] is None else f"{s['k']}/{s['n']} = {s['share']:.1f} % [{s['lo']:.1f}, {s['hi']:.1f}]"


def _df(p: dict[str, Any]) -> str:
    if p["diff"] is None:
        return "n = 0"
    return f"{p['diff']:+.1f} pts [{p['lo']:+.1f}, {p['hi']:+.1f}] ({p['a']:.1f} % vs {p['b']:.1f} %, n = {p['n']}): {p['verdict']}"


def markdown(rep: dict[str, Any]) -> str:
    L = rep["lines"]
    md = ["# Hammer squeeze x CoM pinch: summary", ""]
    md += [f"Cases: test {rep['n']['test']}, validation {rep['n']['validation']}. Dropped by ruling 4 (an artifact or an error in any "
           f"row, ruling B): test {rep['dropped']['test']}, validation {rep['dropped']['validation']}. The penetration sensitivity "
           f"drops {rep['dropped_penetration_test']} test cases. Paired bootstrap over cases, {DRAWS} draws, seed {SEED}, 95 % "
           f"percentile; margins placed +{MARGINS['placed']:.0f}, lifted +{MARGINS['lifted']:.0f} points. The interval treats the "
           "cases as independent, while four objects carry the test side.", ""]
    g = rep["squeeze"]["gate"]
    md += [f"Squeeze gate (reading 5): the {g['reader']} median at 70 N is {_n(g['median'])} N over n = {g['n']} rows "
           f"(command 70 N, tolerance {100 * g['tolerance']:.0f} %): {'pass' if g['ok'] else 'FAIL'}." + (f" {g['note']}." if g["note"] else ""), ""]
    md += ["## Reading 1: force at the CoM pinch (across_com), each level against f_std", "",
           "| line | n | difference | placed | lifted |", "|---|---|---|---|---|"]
    for ln in LINES:
        for d in L[ln]["r1"]["diffs"]:
            x = L[ln]["r1"]["diffs"][d]
            md.append(f"| {LINE_NAMES[ln]} | {L[ln]['n']} | {d}{' (primary)' if d == '70N - f_std' else ''} | {_df(x['placed'])} | {_df(x['lifted'])} |")
    md += ["", "Shares at across_com per level (test cases):", "", "| level | placed | lifted |", "|---|---|---|"]
    for lv, s in L["test"]["r1"]["shares"].items():
        md.append(f"| {lv} | {_sh(s['placed'])} | {_sh(s['lifted'])} |")
    md += ["", "Sensitivity (ruling 2): a missed row counts as a lift failure.", "", "| line | difference | lifted |", "|---|---|---|"]
    for ln in LINES:
        for d, x in L[ln]["r1_missed"]["diffs"].items():
            md.append(f"| {LINE_NAMES[ln]} | {d} | {_df(x['lifted'])} |")
    md += ["", "## Reading 2: the pinch at each force (across_com - across_0.5)", "", "| line | level | placed | lifted |", "|---|---|---|---|"]
    for ln in LINES:
        for lv, x in L[ln]["r2"].items():
            md.append(f"| {LINE_NAMES[ln]} | {lv} | {_df(x['placed'])} | {_df(x['lifted'])} |")
    md += ["", f"## Reading 3: by hidden mass (split at {SPLIT_KG:.0f} kg, descriptive)", ""]
    for ln in ("test", "validation"):
        r3 = L[ln]["r3"]
        md += [f"{LINE_NAMES[ln]}: light n = {r3['light']['n']}, heavy n = {r3['heavy']['n']}.", "",
               "| cell | light placed | light lifted | heavy placed | heavy lifted |", "|---|---|---|---|---|"]
        for c in CELLS:
            md.append(f"| {c} | {_sh(r3['light']['cells'][c]['placed'])} | {_sh(r3['light']['cells'][c]['lifted'])} | "
                      f"{_sh(r3['heavy']['cells'][c]['placed'])} | {_sh(r3['heavy']['cells'][c]['lifted'])} |")
        md.append("")
    md += ["## Reading 4: what remains (states over every simulated case of the side, before the drop)", "",
           f"The best cell is {rep['best']['cell']}: {rep['best']['rule']}.", "", "| side | cell | " + " | ".join(STATES) + " |",
           "|---|---|" + "---|" * len(STATES)]
    for s in ("test", "validation"):
        for which in ("best", "control"):
            b = rep["r4"][s][which]
            md.append(f"| {s} | {which} | " + " | ".join(str(b[k]) for k in STATES) + " |")
    md += ["", "## Reading 5: the measured squeeze per pad (median, N)", "", "| level | hold | n | late close | n |", "|---|---|---|---|---|"]
    for lv, x in rep["squeeze"]["levels"].items():
        md.append(f"| {lv} | {_n(x['hold_median'])} | {x['hold_n']} | {_n(x['close_median'])} | {x['close_n']} |")
    md += ["", "## Reading 6: damage labels (a defect of the setting: hammers carry no damage law)", ""]
    for s in ("test", "validation"):
        md.append(f"- {s}: {sum(rep['r6'][s].values())} rows" + (f" ({', '.join(f'{c} {k}' for c, k in rep['r6'][s].items() if k)})" if any(rep['r6'][s].values()) else ""))
    md += ["", "## Reading 7: the best cell against the published hammer line", "", "| line | n | best cell placed | vs the corrected ceiling10 | vs Stage 2c R4 |",
           "|---|---|---|---|---|"]
    for ln in LINES:
        r = L[ln]["r7"]
        md.append(f"| {LINE_NAMES[ln]} | {L[ln]['n']} | {r['cell']}: {_sh(r['share'])} | {_df(r['vs_ceiling10'])} | {_df(r['vs_r4'])} |")
    md += ["", "The best cell is selected on the test cases; its interval does not account for the selection.", ""]
    md += ["## Per cell (every simulated case of the side)", "", "| side | cell | cases | simulated | infeasible | artifact | error | "
           "penetration > 3 mm | missed | damage | drops |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in ("test", "validation"):
        for c, x in rep["per_cell"][s].items():
            md.append(f"| {s} | {c} | {x['cases']} | {x['simulated']} | {x['infeasible']} | {x['artifact']} | {x['error']} | {x['penetration']} | "
                      f"{x['missed']} | {x['damage']} | {x['drops']} |")
    md += ["", "Infeasible across_com pinches by object (cases, reasons): " +
           ("; ".join(f"{o} {x['cases']} ({', '.join(f'{w}: {k}' for w, k in x['why'].items())})" for o, x in rep["infeasible_com_by_object"].items()) or "none"), ""]
    if rep["unpaired"] is not None:
        md += [f"More than 5 test cases dropped ({rep['dropped']['test']}): per cell on every clean row, UNPAIRED.", "",
               "| cell | placed | lifted |", "|---|---|---|"]
        md += [f"| {c} | {_sh(x['placed'])} | {_sh(x['lifted'])} |" for c, x in rep["unpaired"].items()]
        md.append("")
    md += ["## Monitors", "", f"Pad penetration per level over every simulated row (the {PEN_MM:.0f} mm limit was set for the commit_std squeeze; "
           "at 40 N and 70 N it is a monitor field, ruling B):", "", "| level | rows | median mm | max mm | over 3 mm | share |", "|---|---|---|---|---|---|"]
    for lv, x in rep["penetration"].items():
        md.append(f"| {lv} | {x['rows']} | {_n(x['median_mm'], 2)} | {_n(x['max_mm'], 2)} | {x['over_3mm']} | {_n(x['share_over_3mm'])} % |")
    md += ["", "Wedge diagnostic (rows over 3 mm beside rows under it, kept cases; diagnostic only):", "",
           "| side | cell | deep rows | lifted | placed | shallow rows | lifted | placed |", "|---|---|---|---|---|---|---|---|"]
    for s in ("test", "validation"):
        for c, x in rep["wedge"][s].items():
            d, h = x["deep"], x["shallow"]
            md.append(f"| {s} | {c} | {d['rows']} | {d['lifted']} ({_n(d['lifted_share'])} %) | {d['placed']} ({_n(d['placed_share'])} %) | "
                      f"{h['rows']} | {h['lifted']} ({_n(h['lifted_share'])} %) | {h['placed']} ({_n(h['placed_share'])} %) |")
    md += ["", f"Close monitor (the object's shift in the close from the approach's end; rows over {SHIFT_MM:.0f} mm; the turn in the pads at the "
           "lift's end where both pads touch):", "", "| side | cell | rows | shift median mm | max mm | over 5 mm | turn rows | turn median deg |",
           "|---|---|---|---|---|---|---|---|"]
    for s in ("test", "validation"):
        for c, x in rep["close"][s].items():
            md.append(f"| {s} | {c} | {x['rows']} | {_n(x['shift_median_mm'], 2)} | {_n(x['shift_max_mm'], 2)} | {x['over_5mm']} | {x['turn_rows']} | "
                      f"{_n(x['turn_median_deg'])} |")
    return "\n".join(md) + "\n"


# ---------------------------------------------------------------- the figure

def figure(rep: dict[str, Any], path: str) -> None:
    """Placed and lifted against the squeeze (the median command per level), one line per pinch, light and heavy panels
    (test cases, the headline set), 95 % intervals, the control cell ringed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r3 = rep["lines"]["test"]["r3"]
    x = [rep["force_by_level"][lv] for lv in hf.LEVELS]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), sharex=True, sharey=True)
    style = {"across_0.5": ("#2a78d6", "o", "mid pinch (across_0.5)"), "across_com": ("#d6542a", "s", "pinch at the CoM (across_com)")}
    for j, band in enumerate(("light", "heavy")):
        for i, out in enumerate(("placed", "lifted")):
            ax = axes[i][j]
            for p in hf.PINCHES:
                col, mk, lab = style[p]
                s = [r3[band]["cells"][hf.row_id(p, lv)][out] for lv in hf.LEVELS]
                y = [np.nan if q["share"] is None else q["share"] for q in s]
                lo = [0 if q["share"] is None else q["share"] - q["lo"] for q in s]
                hi = [0 if q["share"] is None else q["hi"] - q["share"] for q in s]
                dx = -0.8 if p == "across_0.5" else 0.8              # a small dodge: equal shares stay visible
                ax.errorbar([xi + dx for xi in x], y, yerr=[lo, hi], color=col, marker=mk, capsize=3, lw=1.6, label=lab)
            c = r3[band]["cells"][hf.CONTROL][out]
            if c["share"] is not None:
                ax.plot([x[0] - 0.8], [c["share"]], marker="o", ms=15, mfc="none", mec="black", mew=1.4, ls="none",
                        label="control (mid pinch at f_std)")
            ax.set_ylim(-5, 105)
            ax.grid(alpha=0.3)
            name = "light (< 1 kg)" if band == "light" else "heavy (>= 1 kg)"
            ax.set_title(f"{out}, {name}, n = {r3[band]['n']}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{out} (% of test cases)")
            if i == 1:
                ax.set_xlabel("squeeze per finger (N, median command per level)")
    axes[0][0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if path.endswith(".png"):
        fig.savefig(path[:-4] + ".pdf")
    plt.close(fig)


# ---------------------------------------------------------------- the command

def _read_jsonl(path: str) -> list[dict[str, Any]]:
    """The only reader of a records file (one open per file)."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _dirs(text: str) -> list[str]:
    return [d for d in text.split(",") if d]


def pre_gates(a: Any) -> dict[str, Any]:
    """The gates before any record is read (SystemExit with the reason on a failure)."""
    cases = json.load(open(a.cases_file))["cases"]
    sha = _sha(a.cases_file)
    out: dict[str, Any] = {"cases_sha256": sha}
    keys = {(e["object"], int(e["idx"])) for e in cases}
    for name, dirs, flag in (("run", _dirs(a.runs), False), ("monitor pass", _dirs(a.monitor), True)):
        codes, cover, secs = set(), set(), 0.0
        for d in dirs:
            ex = open(os.path.join(d, "run.exit")).read().strip()
            if ex != "EXIT 0":
                raise SystemExit(f"{name} unit {d}: {ex}")
            c = json.load(open(os.path.join(d, "run_config.json")))
            m = json.load(open(os.path.join(d, "run_metrics.json")))
            if c.get("replan_hold") is not True:
                raise SystemExit(f"{name} unit {d}: replan_hold is not True (the re-plan at the hold, ruling 7)")
            if c.get("sim") != hf.CONTACTS_SIM or c.get("dirty"):
                raise SystemExit(f"{name} unit {d}: not the contacts sim at a clean code")
            if c.get("cases_sha256") != sha:
                raise SystemExit(f"{name} unit {d}: another cases file")
            if bool(c.get("close_monitor")) is not flag:
                raise SystemExit(f"{name} unit {d}: the close-monitor flag is {c.get('close_monitor')}")
            if m.get("stop") or m.get("stops") or m.get("g0_mismatches"):
                raise SystemExit(f"{name} unit {d}: a stop or a G0 mismatch in its metrics")
            codes.add(c["code_revision"])
            cover |= {(k[0], int(k[1])) for k in c["keys"]}
            secs += float(m.get("case_seconds") or 0.0)
        if len(codes) != 1:
            raise SystemExit(f"the {name}'s units ran {len(codes)} codes")
        if cover != keys:
            raise SystemExit(f"the {name}'s units do not cover the cases file ({len(cover)} of {len(keys)} cases)")
        out[name.replace(" ", "_")] = {"units": len(dirs), "code": codes.pop(), "cpu_h": secs / 3600.0}
    if open(os.path.join(a.smoke, "run.exit")).read().strip() != "EXIT 0":
        raise SystemExit("the smoke report did not exit 0")
    sm = json.load(open(os.path.join(a.smoke, "smoke_summary.json")))
    if sm["g0"]["mismatches"] or sm["g1"]["label_mismatches"]:
        raise SystemExit("the smoke's G0 or G1 is not clean")
    scodes = {json.load(open(os.path.join(d, "run_config.json")))["code_revision"] for d in _dirs(sm["runs"])}
    if scodes != {out["run"]["code"]}:
        raise SystemExit(f"the run's code {out['run']['code'][:7]} is not the smoke's {sorted(c[:7] for c in scodes)}")
    out["smoke"] = {"g0": sm["g0"], "g1": sm["g1"], "projection_cpu_h": sm.get("projection_cpu_h")}
    return out


def _stop(out: str, gate: str, **kw: Any) -> None:
    with open(os.path.join(out, "STOP.json"), "w") as f:
        json.dump(dict({"gate": gate}, **kw), f, indent=1)
    print(f"STOP at the {gate} gate: {json.dumps(kw)[:2000]}", file=sys.stderr, flush=True)
    raise SystemExit(3)


def summary_cmd(a: Any) -> dict[str, Any]:
    """The pre-open gates, the one open, the gates after it, the squeeze gate, the report: summary.json, summary.md,
    figure.png and figure.pdf in --out (refuses an existing summary or stop file); wandb."""
    from robots.mujoco_lab.tracking import log_run

    if os.path.exists(os.path.join(a.out, "summary.json")) or os.path.exists(os.path.join(a.out, "STOP.json")):
        raise SystemExit(f"{a.out}: a summary or a stop file exists")
    pre = pre_gates(a)
    os.makedirs(a.out, exist_ok=True)
    cases = json.load(open(a.cases_file))["cases"]
    recs = [r for d in _dirs(a.runs) for s in ("test", "validation") for r in _read_jsonl(os.path.join(d, f"records_{s}.jsonl"))]
    mon_list = [r for d in _dirs(a.monitor) for s in ("test", "validation") for r in _read_jsonl(os.path.join(d, f"records_{s}.jsonl"))]
    mon = {(m["object"], int(m["idx"])): m for m in mon_list}
    if len(mon) != len(mon_list):
        _stop(a.out, "coverage", problems=["a case is in the monitor pass twice"])
    g = post_gates(recs, mon, cases)
    if g["problems"]:
        _stop(a.out, "records", problems=g["problems"], g0=g["g0"], hash_join=g["hash_join"], artifact_flags=g["artifact_flags"])
    sq = squeeze_gate(recs)
    if not sq["gate"]["ok"]:
        _stop(a.out, "squeeze", **sq["gate"])
    rep = build(recs, mon, cases, pre)
    rep["gates"] = {k: g[k] for k in ("g0", "hash_join", "artifact_flags")}
    rep["cpu_h"] = {"projected": pre["smoke"]["projection_cpu_h"], "run": pre["run"]["cpu_h"], "monitor_pass": pre["monitor_pass"]["cpu_h"]}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(rep, f, indent=1)
    md = markdown(rep)
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write(md)
    figure(rep, os.path.join(a.out, "figure.png"))
    t = rep["lines"]["test"]["r1"]["diffs"]["70N - f_std"]
    rep["wandb"] = log_run("hammer-force-summary", {"cases_sha256": pre["cases_sha256"], "run_code": pre["run"]["code"],
                                                     "monitor_code": pre["monitor_pass"]["code"], "best": rep["best"]["cell"]},
                           {"n": rep["n"], "dropped": rep["dropped"], "primary_placed": t["placed"], "primary_lifted": t["lifted"]})
    print(md, flush=True)
    return rep


def main(argv: list[str] | None = None) -> Any:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", required=True, help="comma-separated run unit dirs")
    p.add_argument("--monitor", required=True, help="comma-separated close monitor pass unit dirs")
    p.add_argument("--cases-file", required=True)
    p.add_argument("--smoke", required=True, help="the smoke report dir (smoke_summary.json, run.exit)")
    p.add_argument("--out", required=True)
    return summary_cmd(p.parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
