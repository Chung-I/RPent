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
and the pooled cases (exploratory). Every reading is given on every line (the kept cases of the line).

The gates step (``gates``, the coordinator's order of 9 Nov): the same gates before the summary, on records read with
_read_fields (the case and row keys of GATE_CASE_KEYS and GATE_ROW_KEYS and the hidden mass: no label and no outcome
field is kept), G0 on the hashes; it reports the gates, the squeeze gate at 70 N, the late-close squeeze per level, and
per cell the infeasible, artifact and error rows, the penetration and the close shift (gates.json, gates.md). The
summary (``summary``) runs after the coordinator's acknowledgement and repeats every gate."""

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
COHORT = {"test": 44, "validation": 55}            # the design's 99 cases (ruling 1: 54 validation cases simulated)
SKIPPED = {("handal__hammer_3", 2)}                # the Stage 2 skip (a Stage 1 artifact)
SMOKE_PER_OBJECT = 2                               # ruling 10: the smoke runs the first two cases by idx per object
HASH_FIELDS = ("qpos_sha", "trace_sha")            # G0 in the gates step, which reads no label
GATE_CASE_KEYS = ("object", "idx", "side", "skipped", "stop", "error", "g0", "seconds")
GATE_ROW_KEYS = ("id", "pinch", "level", "feasible", "why", "force", "qpos_sha", "trace_sha", "artifact", "error", "seconds",
                 "monitor", "squeeze", "close")


# ---------------------------------------------------------------- rows and cases

def causes(r: dict[str, Any]) -> dict[str, bool]:
    """The contact monitor's artifact causes of a simulated row (adapter.artifact's rule, the jam's flag)."""
    m = r.get("monitor") or {}
    return {"reversed": int(m.get("rev_steps") or 0) >= REV_STEPS, "deep": float(m.get("deepest_mm") or 0.0) > PEN_MM,
            "not_at_rest": m.get("at_rest") is False, "jam": bool(m.get("jam_artifact"))}


def row_flag(r: dict[str, Any]) -> str | None:
    """A row's state that needs no label: infeasible, error, artifact (ruling B: the penetration is not one at 40 N and
    70 N), or None for a clean simulated row."""
    if not r["feasible"]:
        return "infeasible"
    if r.get("error") or "seconds" not in r:
        return "error"
    c = causes(r)
    if c["reversed"] or c["not_at_rest"] or c["jam"] or (c["deep"] and r["level"] not in PEN_LEVELS):
        return "artifact"
    return None


def row_state(r: dict[str, Any]) -> str:
    """row_flag, else the motion label, or damage (any other label: hammers carry no damage law); a clean row without a
    label is an error."""
    f = row_flag(r)
    if f is not None:
        return f
    lab = r.get("label")
    if lab is None:
        return "error"
    return lab if lab in MOTION else "damage"


def case_view(rec: dict[str, Any], entry: dict[str, Any], mon: dict[str, Any] | None) -> dict[str, Any]:
    """A simulated case: its keys, mass (the sim's truth), the published comparators (ceiling10, r4), the primary drop
    (the cells with an artifact or an error under ruling B), ``drop_a`` (ruling 4 as written: the run's artifact flag or
    an error in any row), and per cell the state, placed, lifted, lifted_strict (missed counts as a lift failure), the
    damage label (in any state: reading 6), the penetration flag and depth, the reason of an infeasible row, the command,
    the measured squeeze (None for an unsimulated or failed row) and the close monitor's fields (None when unsimulated)."""
    mrows = {r["id"]: r for r in (mon or {}).get("rows", [])}
    cells, drop, drop_a = {}, [], False
    for r in rec["rows"]:
        s = row_state(r)
        clean = s not in ("infeasible", "artifact", "error")
        deep = float((r.get("monitor") or {}).get("deepest_mm") or 0.0) if "seconds" in r else None
        lab = r.get("label")
        cells[r["id"]] = {"state": s, "placed": s == "placed_stable", "lifted": clean and s != "not_lifted",
                          "lifted_strict": clean and s not in ("not_lifted", "missed"), "pen": deep is not None and deep > PEN_MM,
                          "damage_label": "seconds" in r and lab is not None and lab not in MOTION,
                          "deep_mm": deep, "why": r.get("why"), "level": r["level"], "pinch": r["pinch"], "force": float(r["force"]),
                          "squeeze": r.get("squeeze") if "seconds" in r and not r.get("error") else None,
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
    """By hidden mass (descriptive): every cell's shares in the light (< SPLIT_KG) and the heavy band, n per band, and
    the band's median command per level (the figure's squeeze axis)."""
    out = {}
    for band, keep in (("light", lambda m: m < SPLIT_KG), ("heavy", lambda m: m >= SPLIT_KG)):
        b = [v for v in vs if keep(v["mass"])]
        out[band] = {"n": len(b), "cells": {c: {"placed": share([v["cells"][c]["placed"] for v in b]),
                                                "lifted": share([v["cells"][c]["lifted"] for v in b])} for c in CELLS},
                     "force": {lv: float(np.median([v["cells"][c]["force"] for v in b for c in CELLS if c.endswith("@" + lv)])) if b else None
                               for lv in hf.LEVELS}}
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


def damage_counts(vs: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Reading 6: per cell the rows with a damage label in any state (a defect of the setting: hammers carry no damage
    law), and those of them on an artifact row."""
    return {c: {"rows": sum(v["cells"][c]["damage_label"] for v in vs),
                "artifact_rows": sum(v["cells"][c]["damage_label"] and v["cells"][c]["state"] == "artifact" for v in vs)} for c in CELLS}


def per_cell(vs: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Per cell over every simulated case of the side: cases, simulated rows, infeasible, artifact, error, the
    penetration flags (over PEN_MM), missed, damage labels (any state), and the cases this cell's row drops (ruling 4
    with B)."""
    out = {}
    for c in CELLS:
        st = [v["cells"][c]["state"] for v in vs]
        out[c] = {"cases": len(vs), "simulated": sum(s != "infeasible" for s in st), "infeasible": st.count("infeasible"),
                  "artifact": st.count("artifact"), "error": st.count("error"), "penetration": sum(v["cells"][c]["pen"] for v in vs),
                  "missed": st.count("missed"), "damage": sum(v["cells"][c]["damage_label"] for v in vs),
                  "drops": sum(s in ("artifact", "error") for s in st)}
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


def squeeze_levels(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per level, the median measured squeeze per pad in the hold and in the late close, with n (``rows``: rows or
    cells with ``level`` and ``squeeze``; a row without a squeeze is left out)."""
    out = {}
    for lv in hf.LEVELS:
        sq = [r["squeeze"] for r in rows if r["level"] == lv and r.get("squeeze")]
        h = [s["hold"] for s in sq if s.get("hold") is not None]
        c = [s["close"] for s in sq if s.get("close") is not None]
        out[lv] = {"hold_median": float(np.median(h)) if h else None, "hold_n": len(h),
                   "close_median": float(np.median(c)) if c else None, "close_n": len(c)}
    return out


def squeeze_gate(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """Reading 5 and ruling 6: per level the median measured squeeze per pad in the hold and in the late close (with
    n); the gate at 70 N reads the hold with SQUEEZE_MIN_N or more readings, else the close, and passes within
    SQUEEZE_TOL of the command."""
    levels = squeeze_levels([r for rec in recs for r in rec["rows"] if "seconds" in r and not r.get("error")])
    g = levels["70N"]
    if g["hold_n"] >= SQUEEZE_MIN_N:
        reader, n, med, note = "hold", g["hold_n"], g["hold_median"], None
    else:
        reader, n, med = "close", g["close_n"], g["close_median"]
        note = f"fewer than {SQUEEZE_MIN_N} rows at 70 N have a hold reading ({g['hold_n']}): the gate reads the late close"
    ok = med is not None and abs(med - 70.0) <= SQUEEZE_TOL * 70.0
    return {"levels": levels, "gate": {"reader": reader, "n": n, "median": med, "command": 70.0, "tolerance": SQUEEZE_TOL, "ok": ok, "note": note}}


# ---------------------------------------------------------------- the gates after the open

def _monitor_problems(k: tuple[str, int], rec: dict[str, Any], m: dict[str, Any]) -> list[str]:
    """The close monitor pass's record of one simulated case against the run's: no case error or stop, the eight cells
    once in order, the run's feasibility per row, and on each simulated row no error and a close record with its shift."""
    out = []
    if m.get("error") or m.get("stop"):
        out.append(f"monitor pass case {k[0]} {k[1]}: a case error or stop")
    ids = [r["id"] for r in m.get("rows") or []]
    if ids != list(CELLS):
        return out + [f"monitor pass case {k[0]} {k[1]}: {len(ids)} rows, not the eight cells once in order"]
    for r, x in zip(rec["rows"], m["rows"]):
        if bool(r["feasible"]) != bool(x.get("feasible")):
            out.append(f"monitor pass case {k[0]} {k[1]} row {r['id']}: the feasibility differs from the run's")
        elif "seconds" in r:
            if x.get("error"):
                out.append(f"monitor pass case {k[0]} {k[1]} row {r['id']}: an error")
            if not isinstance(x.get("close"), dict) or x["close"].get("close_shift_mm") is None:
                out.append(f"monitor pass case {k[0]} {k[1]} row {r['id']}: no close record")
    return out


def post_gates(recs: list[dict[str, Any]], mon: dict[tuple[str, int], dict[str, Any]], cases: list[dict[str, Any]],
               g0_fields: tuple[str, ...] = hf.G0_FIELDS) -> dict[str, Any]:
    """Coverage of the run and of the monitor pass (every case of the cases file, skipped ones included), G0 again on
    ``g0_fields`` (the gates step, which reads no label, passes HASH_FIELDS), the artifact flags, the monitor pass's
    rows, the hash join. ``problems`` names cases, rows, fields and counts, never a label or a value."""
    problems: list[str] = []
    want = {(e["object"], int(e["idx"])): e for e in cases}
    seen = Counter((r["object"], int(r["idx"])) for r in recs)
    problems += [f"case {k[0]} {k[1]} is in the records twice" for k, n in seen.items() if n > 1]
    problems += [f"case {k[0]} {k[1]} is missing from the records" for k in want if k not in seen]
    problems += [f"case {k[0]} {k[1]} is not in the cases file" for k in seen if k not in want]
    problems += [f"case {k[0]} {k[1]} of the monitor pass is not in the cases file" for k in mon if k not in want]
    problems += [f"case {k[0]} {k[1]} is missing from the monitor pass" for k in want if k not in mon]
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
        m = mon.get(k)
        if m is not None and (bool(m.get("skipped")) != bool(e.get("skipped")) or m.get("side") != e["side"]):
            problems.append(f"case {k[0]} {k[1]}: side or skip of the monitor pass differs from the cases file")
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
        ref = e["ref"] or {}
        diff = [f for f in g0_fields if json.dumps(ctl.get(f), sort_keys=True) != json.dumps(ref.get(f), sort_keys=True)] \
            if "seconds" in ctl else list(g0_fields)
        g0["compared"] += 1
        if diff or rec.get("g0"):
            g0["mismatches"] += 1
            problems.append(f"G0: case {k[0]} {k[1]}: the control row differs from the corrected pose-set record in {', '.join(diff or rec['g0'])}")
        for r in rec["rows"]:
            if "seconds" in r and not r.get("error") and bool(r.get("artifact")) != any(causes(r).values()):
                flags += 1
        if m is None:
            continue
        problems += _monitor_problems(k, rec, m)
        mrows = {r["id"]: r for r in m.get("rows") or []} if [r["id"] for r in m.get("rows") or []] == list(CELLS) else {}
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
    """Every reading on one line's kept cases (ruling 4: a dropped case leaves every reading)."""
    return {"n": len(vs), "exploratory": exploratory, "r1": reading1(vs), "r1_missed": reading1(vs, strict=True), "r2": reading2(vs),
            "r3": reading3(vs), "r4": {"best": breakdown(vs, cell), "control": breakdown(vs, hf.CONTROL)},
            "r5": squeeze_levels([c for v in vs for c in v["cells"].values() if c["squeeze"]]), "r6": damage_counts(vs),
            "r7": reading7(vs, cell)}


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
    return {"pre": pre, "n": {s: len(side[s]) for s in side}, "dropped": dropped,
            "dropped_penetration_test": len(side["test"]) - len(keep_a), "lines": lines,
            "best": {"cell": best, "rule": "the highest placed share on the test cases; a tie goes to the lower squeeze, then to across_0.5 "
                                            "(ruling 8); across_com x 70 N stays the primary of reading 1"},
            "r4_all": {s: {"best": breakdown(side[s], best), "control": breakdown(side[s], hf.CONTROL)} for s in side},
            "r6_all": {s: damage_counts(side[s]) for s in side},
            "per_cell": {s: per_cell(side[s]) for s in side},
            "infeasible_com_by_object": infeasible_by_object(allv),
            "unpaired": unpaired(side["test"]) if dropped["test"] > 5 else None,
            "wedge": {s: wedge(keep[s]) for s in side}, "close": {s: close_report(side[s]) for s in side},
            "penetration": penetration(recs), "squeeze": squeeze_gate(recs),
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
    for ln in LINES:
        r3 = L[ln]["r3"]
        md += [f"{LINE_NAMES[ln]}: light n = {r3['light']['n']}, heavy n = {r3['heavy']['n']}.", "",
               "| cell | light placed | light lifted | heavy placed | heavy lifted |", "|---|---|---|---|---|"]
        for c in CELLS:
            md.append(f"| {c} | {_sh(r3['light']['cells'][c]['placed'])} | {_sh(r3['light']['cells'][c]['lifted'])} | "
                      f"{_sh(r3['heavy']['cells'][c]['placed'])} | {_sh(r3['heavy']['cells'][c]['lifted'])} |")
        md.append("")
    md += ["## Reading 4: what remains (states of the best and the control cell)", "",
           f"The best cell is {rep['best']['cell']}: {rep['best']['rule']}.", "", "Over each line's kept cases (ruling 4):", "",
           "| line | n | cell | " + " | ".join(STATES) + " |", "|---|---|---|" + "---|" * len(STATES)]
    for ln in LINES:
        for which in ("best", "control"):
            b = L[ln]["r4"][which]
            md.append(f"| {LINE_NAMES[ln]} | {L[ln]['n']} | {which} | " + " | ".join(str(b[k]) for k in STATES) + " |")
    md += ["", "Diagnostic: over every simulated case of the side, before the drop (the artifact and error rows of the two cells):", "",
           "| side | cell | " + " | ".join(STATES) + " |", "|---|---|" + "---|" * len(STATES)]
    for s in ("test", "validation"):
        for which in ("best", "control"):
            b = rep["r4_all"][s][which]
            md.append(f"| {s} | {which} | " + " | ".join(str(b[k]) for k in STATES) + " |")
    g = rep["squeeze"]["gate"]
    md += ["", "## Reading 5: the measured squeeze per pad (median, N)", "",
           f"The gate (every simulated row): the {g['reader']} median at 70 N is {_n(g['median'])} N over n = {g['n']} rows.", "",
           "| line | level | hold | n | late close | n |", "|---|---|---|---|---|---|"]
    for lv, x in rep["squeeze"]["levels"].items():
        md.append(f"| every simulated row | {lv} | {_n(x['hold_median'])} | {x['hold_n']} | {_n(x['close_median'])} | {x['close_n']} |")
    for ln in LINES:
        for lv, x in L[ln]["r5"].items():
            md.append(f"| {LINE_NAMES[ln]} | {lv} | {_n(x['hold_median'])} | {x['hold_n']} | {_n(x['close_median'])} | {x['close_n']} |")
    md += ["", "## Reading 6: damage labels in any state (a defect of the setting: hammers carry no damage law)", ""]

    def dmg(d: dict[str, dict[str, int]]) -> str:
        tot = sum(x["rows"] for x in d.values())
        part = ", ".join(f"{c} {x['rows']} ({x['artifact_rows']} on artifact rows)" for c, x in d.items() if x["rows"])
        return f"{tot} rows" + (f": {part}" if part else "")

    for ln in LINES:
        md.append(f"- {LINE_NAMES[ln]}: {dmg(L[ln]['r6'])}")
    for s in ("test", "validation"):
        md.append(f"- every simulated {s} case, before the drop: {dmg(rep['r6_all'][s])}")
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
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), sharex=True, sharey=True)
    style = {"across_0.5": ("#2a78d6", "o", "mid pinch (across_0.5)"), "across_com": ("#d6542a", "s", "pinch at the CoM (across_com)")}
    for j, band in enumerate(("light", "heavy")):
        x = [np.nan if r3[band]["force"][lv] is None else r3[band]["force"][lv] for lv in hf.LEVELS]   # this panel's own cases
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
                ax.set_xlabel("squeeze per finger (N)")         # the panel's median command per level (the caption says so)
    axes[0][0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if path.endswith(".png"):
        fig.savefig(path[:-4] + ".pdf")
    plt.close(fig)


# ---------------------------------------------------------------- the command

def _read_jsonl(path: str) -> list[dict[str, Any]]:
    """The summary's only reader of a records file (one open per file)."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_fields(path: str) -> list[dict[str, Any]]:
    """The gates step's only reader of a records file: each line keeps the case keys of GATE_CASE_KEYS, the hidden
    mass and the row keys of GATE_ROW_KEYS; no label and no outcome field is kept."""
    out = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            c = {k: r.get(k) for k in GATE_CASE_KEYS}
            c["sim_truth"] = {"mass": r["sim_truth"].get("mass")} if r.get("sim_truth") else None
            c["rows"] = [{k: x[k] for k in GATE_ROW_KEYS if k in x} for x in r.get("rows") or []]
            out.append(c)
    return out


def _code() -> dict[str, Any]:
    """This module's own code: the revision and whether the tree is dirty."""
    from robots.mujoco_lab.estimation_check import _revision

    me = _revision()
    return {"code_revision": me["code_revision"], "dirty": me["dirty"]}


def _clean_code() -> dict[str, Any]:
    me = _code()
    if me["dirty"]:
        raise SystemExit(f"the summary's code {me['code_revision'][:7]} is dirty: commit it first")
    return me


# ---------------------------------------------------------------- the gates step (no label)

def monitor_cells(recs: list[dict[str, Any]], mon: dict[tuple[str, int], dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per side and cell, without a label: rows simulated and infeasible, artifact rows by cause (ruling B), error rows,
    the penetration (rows over PEN_MM, median and max mm) and the close monitor (the object's shift in the close: median,
    max, rows over SHIFT_MM; the turn at the close's end). The turn at the lift's end is left to the summary."""
    out: dict[str, dict[str, Any]] = {}
    for side in ("test", "validation"):
        rs = [r for r in recs if r["side"] == side and not r.get("skipped")]
        out[side] = {}
        for i, c in enumerate(CELLS):
            rows = [rec["rows"][i] for rec in rs]
            fl = [row_flag(r) for r in rows]
            sim = [r for r in rows if "seconds" in r]
            deep = [float(r["monitor"]["deepest_mm"]) for r in sim if r.get("monitor")]
            cl = []
            for rec in rs:
                m = mon.get((rec["object"], int(rec["idx"])))
                if "seconds" in rec["rows"][i] and m is not None:
                    cl.append(m["rows"][i]["close"])
            sh = [x["close_shift_mm"] for x in cl if x.get("close_shift_mm") is not None]
            tu = [x["close_turn_deg"] for x in cl if x.get("close_turn_deg") is not None]
            art = [causes(r) for r, f in zip(rows, fl) if f == "artifact"]
            out[side][c] = {"cases": len(rows), "simulated": len(sim), "infeasible": fl.count("infeasible"), "error": fl.count("error"),
                            "artifact": len(art), "artifact_causes": {k: sum(x[k] for x in art) for k in ("reversed", "not_at_rest", "jam", "deep")},
                            "penetration": sum(x > PEN_MM for x in deep), "deepest_median_mm": float(np.median(deep)) if deep else None,
                            "deepest_max_mm": float(max(deep)) if deep else None, "close_rows": len(sh),
                            "close_shift_median_mm": float(np.median(sh)) if sh else None, "close_shift_max_mm": float(max(sh)) if sh else None,
                            "close_over_5mm": sum(x > SHIFT_MM for x in sh), "close_turn_median_deg": float(np.median(tu)) if tu else None}
    return out


def drop_counts(recs: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Per side, the simulated cases and the cases each drop rule removes (no label): ruling 4 with B (an artifact by
    row_flag or an error in any row, or a case error or stop) and ruling 4 as written (the run's artifact flag or an
    error in any row, or a case error or stop)."""
    out = {}
    for side in ("test", "validation"):
        rs = [r for r in recs if r["side"] == side and not r.get("skipped")]
        bad_case = [bool(r.get("error") or r.get("stop")) for r in rs]
        b = [bc or any(row_flag(x) in ("artifact", "error") for x in r["rows"]) for r, bc in zip(rs, bad_case)]
        a = [bb or any(bool(x.get("artifact")) or row_flag(x) == "error" for x in r["rows"]) for r, bb in zip(rs, b)]
        out[side] = {"cases": len(rs), "ruling_b": sum(b), "ruling_4_as_written": sum(a)}
    return out


def infeasible_rows(recs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The infeasible rows per object (cases and rows per reason; no label)."""
    out: dict[str, dict[str, Any]] = {}
    for rec in recs:
        rows = [r for r in rec.get("rows") or [] if not r["feasible"]]
        if rows:
            o = out.setdefault(rec["object"], {"cases": 0, "rows": 0, "why": {}})
            o["cases"] += 1
            o["rows"] += len(rows)
            for r in rows:
                o["why"][r["why"]] = o["why"].get(r["why"], 0) + 1
    return out


def gates_markdown(rep: dict[str, Any]) -> str:
    """The gates report: counts, medians, names; no outcome."""
    p, g, sq = rep["pre"], rep["gates"], rep["squeeze"]
    md = ["# Hammer squeeze x CoM pinch: the gates before the summary", "",
          "Outcome fields are not read: the records were read with a key list (identity, flags, hashes, the contact and close "
          "monitors, the squeeze, the command).", "",
          "## Before the read", "",
          f"- Cases file sha256 {p['cases_sha256'][:12]}: {COHORT['test']} test and {COHORT['validation']} validation cases, skipped "
          f"{sorted(SKIPPED)}.",
          f"- Run: {p['run']['units']} units EXIT 0, code {p['run']['code'][:7]}, {p['run']['cpu_h']:.3f} CPU-h (the smoke projected "
          f"{_n(p['smoke']['projection_cpu_h'], 2)}).",
          f"- Close monitor pass: {p['monitor_pass']['units']} units EXIT 0, code {p['monitor_pass']['code'][:7]}, {p['monitor_pass']['cpu_h']:.3f} CPU-h.",
          f"- Smoke: the first two cases by idx per object ({p['smoke']['cases']} simulated), G0 {p['smoke']['g0']['compared']} compared / "
          f"{p['smoke']['g0']['mismatches']} mismatches, G1 {p['smoke']['g1']['rows']} rows replayed on {p['smoke']['g1']['cases']} cases / "
          f"{p['smoke']['g1']['label_mismatches']} + {p['smoke']['g1'].get('other_mismatches', 0)} mismatches, at the run's code.",
          f"- This step's code: {rep['code']['code_revision'][:7]}, clean.", "",
          "## After the read", "",
          f"- Coverage: every case of the cases file once in the run and in the monitor pass, eight rows in order, side, skip and mass as "
          f"in the cases file.",
          f"- G0 on {', '.join(g['g0']['fields'])}: {g['g0']['compared']} compared, {g['g0']['mismatches']} mismatches.",
          f"- Artifact flags that the contact monitor does not explain: {g['artifact_flags']['unexplained']}.",
          f"- Hash join with the monitor pass (qpos_sha and trace_sha): {g['hash_join']['rows']} rows, {g['hash_join']['mismatches']} mismatches; "
          "each monitor row once, the run's feasibility, no error, a close record.", "",
          "## Squeeze gate (ruling 6)", "",
          f"The {sq['gate']['reader']} median per pad at 70 N is {_n(sq['gate']['median'])} N over n = {sq['gate']['n']} rows (command 70 N, "
          f"tolerance {100 * sq['gate']['tolerance']:.0f} %): {'pass' if sq['gate']['ok'] else 'FAIL'}." + (f" {sq['gate']['note']}." if sq["gate"]["note"] else ""),
          "", "Late close per level (both pads on the object, the second half of the close):", "", "| level | median N | rows |", "|---|---|---|"]
    md += [f"| {lv} | {_n(x['close_median'])} | {x['close_n']} |" for lv, x in sq["late_close"].items()]
    d = rep["drops"]
    md += ["", "## Cases per line", "",
           f"- Test: {d['test']['cases']} cases; ruling 4 with B drops {d['test']['ruling_b']} (the headline keeps "
           f"{d['test']['cases'] - d['test']['ruling_b']}); ruling 4 as written drops {d['test']['ruling_4_as_written']} (the sensitivity keeps "
           f"{d['test']['cases'] - d['test']['ruling_4_as_written']}).",
           f"- Validation: {d['validation']['cases']} cases; ruling 4 with B drops {d['validation']['ruling_b']}; as written "
           f"{d['validation']['ruling_4_as_written']}.", "",
           "## Per cell (no outcome)", "",
           "| side | cell | simulated | infeasible | artifact (rev / rest / jam / deep) | error | pen > 3 mm | deepest median mm | max mm | "
           "close shift median mm | max mm | rows > 5 mm | turn at the close end, median deg |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in ("test", "validation"):
        for c, x in rep["cells"][s].items():
            ac = x["artifact_causes"]
            md.append(f"| {s} | {c} | {x['simulated']} | {x['infeasible']} | {x['artifact']} ({ac['reversed']} / {ac['not_at_rest']} / {ac['jam']} / "
                      f"{ac['deep']}) | {x['error']} | {x['penetration']} | {_n(x['deepest_median_mm'], 2)} | {_n(x['deepest_max_mm'], 2)} | "
                      f"{_n(x['close_shift_median_mm'], 2)} | {_n(x['close_shift_max_mm'], 2)} | {x['close_over_5mm']} | {_n(x['close_turn_median_deg'])} |")
    md += ["", "Pad penetration per level over every simulated row:", "", "| level | rows | median mm | max mm | over 3 mm | share |",
           "|---|---|---|---|---|---|"]
    md += [f"| {lv} | {x['rows']} | {_n(x['median_mm'], 2)} | {_n(x['max_mm'], 2)} | {x['over_3mm']} | {_n(x['share_over_3mm'])} % |"
           for lv, x in rep["penetration"].items()]
    md += ["", "Infeasible rows by object (cases, rows, reasons): " +
           ("; ".join(f"{o} {x['cases']} cases, {x['rows']} rows ({', '.join(f'{w}: {k}' for w, k in x['why'].items())})"
                      for o, x in rep["infeasible"].items()) or "none"), "",
           "Left to the summary: the hold squeeze below 70 N and the turn in the pads at the lift's end. Both count rows with the object "
           "still between the pads."]
    return "\n".join(md) + "\n"


def gates_cmd(a: Any) -> dict[str, Any]:
    """The gates before the summary (the coordinator's order): pre_gates, the records read once with _read_fields, the
    gates after the read with G0 on HASH_FIELDS, the squeeze gate; gates.json and gates.md in --out (refuses an existing
    gates or stop file); a failure writes STOP.json and exits 3. No label is read."""
    from robots.mujoco_lab.tracking import log_run

    if os.path.exists(os.path.join(a.out, "gates.json")) or os.path.exists(os.path.join(a.out, "STOP.json")):
        raise SystemExit(f"{a.out}: a gates or a stop file exists")
    me = _clean_code()
    pre = pre_gates(a)
    os.makedirs(a.out, exist_ok=True)
    cases = json.load(open(a.cases_file))["cases"]
    recs = [r for d in _dirs(a.runs) for s in ("test", "validation") for r in _read_fields(os.path.join(d, f"records_{s}.jsonl"))]
    mon_list = [r for d in _dirs(a.monitor) for s in ("test", "validation") for r in _read_fields(os.path.join(d, f"records_{s}.jsonl"))]
    mon = {(m["object"], int(m["idx"])): m for m in mon_list}
    if len(mon) != len(mon_list):
        _stop(a.out, "coverage", problems=["a case is in the monitor pass twice"])
    g = post_gates(recs, mon, cases, g0_fields=HASH_FIELDS)
    if g["problems"]:
        _stop(a.out, "records", problems=g["problems"], g0=g["g0"], hash_join=g["hash_join"], artifact_flags=g["artifact_flags"])
    sq = squeeze_gate(recs)
    if not sq["gate"]["ok"]:
        _stop(a.out, "squeeze", **sq["gate"])
    rep = {"code": me, "pre": pre, "gates": {"g0": dict(g["g0"], fields=list(HASH_FIELDS)), "hash_join": g["hash_join"],
                                             "artifact_flags": g["artifact_flags"]},
           "squeeze": {"gate": sq["gate"], "late_close": {lv: {"close_median": x["close_median"], "close_n": x["close_n"]} for lv, x in sq["levels"].items()}},
           "drops": drop_counts(recs), "cells": monitor_cells(recs, mon), "penetration": penetration(recs), "infeasible": infeasible_rows(recs),
           "cpu_h": {"projected": pre["smoke"]["projection_cpu_h"], "run": pre["run"]["cpu_h"], "monitor_pass": pre["monitor_pass"]["cpu_h"]}}
    with open(os.path.join(a.out, "gates.json"), "w") as f:
        json.dump(rep, f, indent=1)
    md = gates_markdown(rep)
    with open(os.path.join(a.out, "gates.md"), "w") as f:
        f.write(md)
    rep["wandb"] = log_run("hammer-force-gates", {"cases_sha256": pre["cases_sha256"], "run_code": pre["run"]["code"],
                                                  "monitor_code": pre["monitor_pass"]["code"], "code": me["code_revision"]},
                           {"hash_join_rows": g["hash_join"]["rows"], "g0_compared": g["g0"]["compared"], "squeeze_n": sq["gate"]["n"],
                            "squeeze_median": sq["gate"]["median"]})
    print(md, flush=True)
    return rep


def _sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _dirs(text: str) -> list[str]:
    return [d for d in text.split(",") if d]


def check_cohort(cases: list[dict[str, Any]]) -> list[str]:
    """The cases file against the approved cohort: each case once, each object on its side (hf.HAMMERS), COHORT cases
    per side, and the skipped cases exactly SKIPPED. Names and counts only."""
    keys = [(e["object"], int(e["idx"])) for e in cases]
    out = [f"case {k[0]} {k[1]} is in the cases file twice" for k, n in Counter(keys).items() if n > 1]
    out += [f"case {e['object']} {e['idx']}: the object is not on the {e['side']} side" for e in cases
            if e["object"] not in hf.HAMMERS.get(e["side"], ())]
    n = dict(Counter(e["side"] for e in cases))
    if n != COHORT:
        out.append(f"the cases file has {n} cases per side, not {COHORT}")
    sk = {(e["object"], int(e["idx"])) for e in cases if e.get("skipped")}
    if sk != SKIPPED:
        out.append(f"the skipped cases are {sorted(sk)}, not {sorted(SKIPPED)}")
    return out


def _unit(name: str, d: str, sha: str, flag: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """One unit's gates before the open: EXIT 0, the re-plan at the hold on (ruling 7), the contacts sim at a clean code,
    the cases file, the close-monitor flag, no stop or G0 mismatch in its metrics. Returns its config and metrics."""
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
    return c, m


def _first_two(cases: list[dict[str, Any]]) -> set[tuple[str, int]]:
    by: dict[str, list[int]] = {}
    for e in cases:
        by.setdefault(e["object"], []).append(int(e["idx"]))
    return {(o, i) for o, ix in by.items() for i in sorted(ix)[:SMOKE_PER_OBJECT]}


def pre_gates(a: Any) -> dict[str, Any]:
    """The gates before any record is read (SystemExit with the reason on a failure): the cohort of the cases file;
    each unit of the run and of the monitor pass (_unit), one code per pass, the cases covered; the smoke: its report
    EXIT 0, its units (_unit, the replay on), the first two cases by idx per object, G0 compared on each of its
    simulated cases, G1 replayed on each of its simulated rows, no mismatch, and the run's code."""
    cases = json.load(open(a.cases_file))["cases"]
    bad = check_cohort(cases)
    if bad:
        raise SystemExit("the cases file: " + "; ".join(bad))
    sha = _sha(a.cases_file)
    out: dict[str, Any] = {"cases_sha256": sha}
    keys = {(e["object"], int(e["idx"])) for e in cases}
    for name, dirs, flag in (("run", _dirs(a.runs), False), ("monitor pass", _dirs(a.monitor), True)):
        codes, cover, secs = set(), set(), 0.0
        for d in dirs:
            c, m = _unit(name, d, sha, flag)
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
    scodes, skeys = set(), set()
    for d in _dirs(sm["runs"]):
        c, _ = _unit("smoke", d, sha, False)
        if c.get("replay") is not True:
            raise SystemExit(f"smoke unit {d}: the replay (G1) was off")
        scodes.add(c["code_revision"])
        skeys |= {(k[0], int(k[1])) for k in c["keys"]}
    if skeys != _first_two(cases):
        raise SystemExit(f"the smoke's {len(skeys)} cases are not the first two by idx per object (ruling 10)")
    sim = len(skeys - {(e["object"], int(e["idx"])) for e in cases if e.get("skipped")})
    if not sim or sm["g0"]["compared"] != sim:
        raise SystemExit(f"the smoke's G0 compared {sm['g0']['compared']} of its {sim} simulated cases")
    if not sm["g1"]["rows"] or sm["g1"]["cases"] != sim or sm["g1"]["rows"] != sm["rows_simulated"]:
        raise SystemExit(f"the smoke's G1 replayed {sm['g1']['rows']} of {sm['rows_simulated']} rows on {sm['g1']['cases']} of {sim} cases")
    if sm["g0"]["mismatches"] or sm["g1"]["label_mismatches"]:
        raise SystemExit("the smoke's G0 or G1 is not clean")
    if scodes != {out["run"]["code"]}:
        raise SystemExit(f"the run's code {out['run']['code'][:7]} is not the smoke's {sorted(c[:7] for c in scodes)}")
    out["smoke"] = {"cases": sim, "g0": sm["g0"], "g1": sm["g1"], "projection_cpu_h": sm.get("projection_cpu_h")}
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
    me = _clean_code()
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
    rep["code"] = me
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
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, hlp in (("gates", "the gates before the summary (no label is read)"), ("summary", "the summary (the one open)")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("--runs", required=True, help="comma-separated run unit dirs")
        s.add_argument("--monitor", required=True, help="comma-separated close monitor pass unit dirs")
        s.add_argument("--cases-file", required=True)
        s.add_argument("--smoke", required=True, help="the smoke report dir (smoke_summary.json, run.exit)")
        s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    return gates_cmd(a) if a.cmd == "gates" else summary_cmd(a)


if __name__ == "__main__":
    main(sys.argv[1:])
