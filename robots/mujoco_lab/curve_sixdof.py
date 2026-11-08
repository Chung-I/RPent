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

"""The sixdof rerun's driver (daily-logs design 2026-11-08-sixdof-rerun-design.md).

Gate G1 (design section 2 and 2.2): every simulated row of the GraspGen pilot (1,756) and of the pose-set ablation (its
test and validation sides) runs again through the new executor path (sixdof.execute), each after its own reset at the
case's own seed, and must match its record on every replay-gate field (curve_poseset.SAME: the label, the artifact flag,
the error, the damage and loss fields, the highest rise, the final state's and the trace's hashes). A pilot GraspGen
row runs as the Grasp6 of its grasp (the request step's file, sha-checked): its vertical mapping must give its record's
point and yaw bit for bit, and its hand path today's. The pilot's gate row and a pose-set row run as their Proposal (a
pose-set row's fetched after the reset, as the pose set did). One mismatch fails G1; nothing is tuned to pass it.

Gate G1′, revised (design amendment 8, ruling (b)): the same rows on the sim where the palm, wrist and finger boxes
collide. A row may change only if one of those boxes touches something during its rollout: g1 --allow-diffs
--box-contacts records each row's box contacts (BoxContacts), and g1p-report fails on a changed row without one. The
palm-contact audit's list is read beside the gate, as the prediction for box contacts before the close.

The re-plan at the hold (amendment 8, rules (a) to (g)): g1 --replan-hold sets the executor's flag (the carry rises by
d when both pads hold the object at the end of the hold and it hangs more than the dead band below its planned offset);
each row carries the executor's record, and a row may then change only through a box contact or a raise. g1p-report
counts the changed rows by cause (contact only, raise only, both), reads d in bins and, with --first-pass, checks the
rows without a raise against the contacts-only pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from robots.mujoco_lab import curve_graspgen as gg
from robots.mujoco_lab import sixdof
from robots.mujoco_lab.adapter import REPLAN_BAND_M
from robots.mujoco_lab.curve_poseset import SAME, trace_sha

SOURCES = ("pilot", "poseset")


def _js(x: Any) -> str:
    return json.dumps(x, sort_keys=True)


def outcome(env: Any, ro: Any, at_rest: Any) -> dict[str, Any]:
    """A rollout's outcome fields as the pilot and the pose set write them (simulate_rows): the label, the artifact flag
    (also when the reset was not at rest), the damage and loss fields, the highest rise, the hashes of the final state
    (qpos to 9 digits) and of the trace; with the executor's re-plan at the hold on, its record."""
    d = getattr(getattr(env, "sim", None), "d", None)
    out = {"label": ro.label, "artifact": bool(getattr(ro, "artifact", False)) or at_rest is False,
           "damage": getattr(ro, "damage", None), "drop_z": getattr(ro, "drop_z", None), "loss_phase": getattr(ro, "loss_phase", None),
           "max_z": getattr(ro, "max_z", None),
           "qpos_sha": None if d is None else hashlib.sha256(np.round(np.asarray(d.qpos, dtype=float), 9).tobytes()).hexdigest()[:16],
           "trace_sha": trace_sha(getattr(ro, "trace", []))}
    if getattr(ro, "replan", None) is not None:          # the re-plan at the hold's record (the executor's flag on)
        out["replan"] = ro.replan
    return out


def _checked_reset(env: Any, reset: Any, fingerprint: Any, want: str, hull_sha: str | None) -> tuple[Any, str | None]:
    at_rest = reset()
    if fingerprint(env) != want:
        return at_rest, "the reset's fingerprint differs from Stage 1's"
    if hull_sha is not None and gg.hulls_sha(env) != hull_sha:
        return at_rest, "the hulls after the reset differ from the record's"
    return at_rest, None


def run_row(env: Any, reset: Any, make: Any, base: dict[str, Any], fingerprint: Any, want: str,
            hull_sha: str | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """One row through the new path after its own reset: reset(), the fingerprint must be ``want`` and the hulls
    ``hull_sha`` (when given), make(env) gives the pinch (a Proposal or a Grasp6, or a reason it cannot), then
    sixdof.execute at ``base`` (commit_std's force, speeds, destination and release height). Returns (the outcome
    fields with the error and the seconds, None) or (None, the reason the case stops); an exception is the row's error."""
    at_rest, stop = _checked_reset(env, reset, fingerprint, want, hull_sha)
    if stop:
        return None, stop
    t0 = time.time()
    new: dict[str, Any] = {"label": None, "artifact": False, "error": None}
    try:
        p, why = make(env)
        if why:
            return None, why
        ro = sixdof.execute(env, p, base["force"], base["v_lift"], base["v_transport"], base["destination"], base["release_height"],
                            a_transport=base["a_transport"])
        new.update(outcome(env, ro, at_rest))
    except Exception as exc:  # noqa: BLE001  a simulator fault is this row's error
        new["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    new["seconds"] = round(time.time() - t0, 2)
    return new, None


def same_fields(stored: dict[str, Any], new: dict[str, Any]) -> dict[str, list[Any]]:
    """The replay-gate fields (SAME) whose JSON differs between a record's row and its replay: {field: [stored, new]}."""
    return {k: [stored.get(k), new.get(k)] for k in SAME if _js(stored.get(k)) != _js(new.get(k))}


def _paths_equal(a: list[tuple[str, Any, float]], b: list[tuple[str, Any, float]]) -> bool:
    return len(a) == len(b) and all(x[0] == y[0] and np.array_equal(x[1], y[1]) and float(x[2]) == float(y[2]) for x, y in zip(a, b))


class BoxContacts:
    """The contacts of the palm, wrist and finger boxes during a rollout (revised gate G1′, design amendment 8: a row
    changes only if one of those boxes touches something). The context wraps the env's tracer: each rollout's tracer
    wraps its sim's substep, which then reads that physics substep's contacts of the boxes (it reads only). A substep's
    contacts belong to the phase of the traced control step that ends with it (that step's last SUB substeps); earlier
    substeps since the last traced step (the free move to the pregrasp, the settle before the trace) are "untraced", and
    substeps after the last traced step are "after". summary(): ``observed`` (a rollout ran), ``first`` (the first
    contact: phase, t in the trace or None outside it, box (palm, wrist or finger), partner ("object" for the object's
    hulls, else the geom's or its body's name) and mm, the penetration) and ``by`` ("phase|box/partner": the substeps in
    contact and the deepest penetration in mm). An env without a tracer gives nothing."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.observed, self.first, self.by = False, None, {}
        self._buf: list[tuple[int, str, str, float]] = []
        self._last: dict[str, int] = {}
        self._n, self._traced = 0, False
        self._sims: list[tuple[Any, bool, Any]] = []
        self._wrapped, self._had, self._old = False, False, None

    def __enter__(self) -> "BoxContacts":
        env = self.env
        orig = getattr(env, "_tracer", None)
        if orig is None:
            return self
        self._wrapped, self._had, self._old = True, "_tracer" in vars(env), vars(env).get("_tracer")

        def tracer(pt: Any, yaw: float) -> tuple[list[dict[str, Any]], Any]:
            trace, on_step = orig(pt, yaw)
            self.observed = True
            self._watch(env.sim)
            sub, hz = int(env.s["scene"].SUB), float(env.s["scene"].CTRL_HZ)

            def step(phase: str) -> None:
                on_step(phase)
                self._flush(phase, sub, (len(trace) - 1) / hz)
            return trace, step

        env._tracer = tracer
        return self

    def __exit__(self, *exc: Any) -> bool:
        if self._wrapped:
            if self._had:
                self.env._tracer = self._old
            else:
                del self.env._tracer
        for sim, had, old in self._sims:
            if had:
                sim.substep = old
            else:
                del sim.substep
        return False

    def _watch(self, sim: Any) -> None:
        if any(s is sim for s, _, _ in self._sims):
            return
        m = sim.m
        kind = {m.geom("palm").id: "palm", m.geom("wrist").id: "wrist"}
        kind.update({g: "finger" for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name in ("finger_l", "finger_r")
                     and m.geom(g).name not in ("pad_l", "pad_r")})
        isbox, hulls = np.zeros(m.ngeom, dtype=bool), {int(h) for h in sim.hull_ids}
        isbox[sorted(kind)] = True

        def partner(g: int) -> str:
            return "object" if g in hulls else (m.geom(g).name or m.body(int(m.geom_bodyid[g])).name or str(g))

        had, old, plain = "substep" in vars(sim), vars(sim).get("substep"), sim.substep

        def substep() -> None:
            plain()
            self._n += 1
            d = sim.d
            n = int(d.ncon)
            if n:
                g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
                a, b = isbox[g1], isbox[g2]
                for k in np.nonzero(a | b)[0]:
                    box, oth = (int(g1[k]), int(g2[k])) if a[k] else (int(g2[k]), int(g1[k]))
                    self._buf.append((self._n, kind[box], partner(oth), -1000.0 * float(d.contact.dist[k])))

        sim.substep = substep
        self._sims.append((sim, had, old))

    def _add(self, i: int, phase: str, t: float | None, box: str, oth: str, mm: float) -> None:
        key, mm = f"{phase}|{box}/{oth}", round(mm, 3)
        e = self.by.setdefault(key, {"substeps": 0, "deepest_mm": mm})
        if self._last.get(key) != i:
            e["substeps"] += 1
            self._last[key] = i
        e["deepest_mm"] = max(e["deepest_mm"], mm)
        if self.first is None:
            self.first = {"phase": phase, "t": t, "box": box, "partner": oth, "mm": mm}

    def _flush(self, phase: str, sub: int, t: float) -> None:
        cut = self._n - sub
        for i, box, oth, mm in self._buf:
            self._add(i, *((phase, t) if i > cut else ("untraced", None)), box, oth, mm)
        self._buf.clear()
        self._traced = True

    def summary(self) -> dict[str, Any]:
        for i, box, oth, mm in self._buf:
            self._add(i, "after" if self._traced else "untraced", None, box, oth, mm)
        self._buf.clear()
        return {"observed": self.observed, "first": self.first, "by": {k: dict(v) for k, v in self.by.items()}}


def _g1_row(env: Any, reset: Any, r: dict[str, Any], source: str, entry: dict[str, Any] | None, base: dict[str, Any],
            fingerprint: Any, want: str, hull: str | None, record: bool = False) -> tuple[dict[str, Any] | None, str | None]:
    from robots.mujoco_lab.curve_poseset import FRACS, TURN

    res: dict[str, Any] = {"id": r["id"], "mapping": None, "hand_path": None}
    if source == "poseset":
        def make(e: Any) -> tuple[Any, str | None]:
            p = {q.s: q for q in e.proposals(FRACS, turn=TURN[r["dir"]])}[r["f"]]
            if [float(x) for x in p.point] != r["point"] or float(p.yaw) != r["yaw"]:
                return None, "the pinch after the reset differs from the record's"
            return p, None
    elif r.get("gate"):
        q0 = gg._proposal(r)

        def make(e: Any) -> tuple[Any, str | None]:
            return q0, None
    else:
        p6 = sixdof.from_graspgen(entry["grasps"][int(r["id"][1:])], s=float(r.get("f", math.nan)), width=float(r["width"]))
        try:
            q = sixdof.as_vertical(p6)
            res["mapping"] = q.point == r["point"] and q.yaw == r["yaw"] and sixdof.approach_deg(p6) == r["approach_deg"]
        except ValueError:
            res["mapping"] = False
        _, stop = _checked_reset(env, reset, fingerprint, want, hull)
        if stop:
            return None, stop
        args = (base["v_lift"], base["v_transport"], base["destination"], base["release_height"])
        res["hand_path"] = res["mapping"] and _paths_equal(sixdof.hand_path(env, p6, *args, a_transport=base["a_transport"]),
                                                           env.hand_path(gg._proposal(r), *args, a_transport=base["a_transport"]))

        def make(e: Any) -> tuple[Any, str | None]:
            return p6, None
    with BoxContacts(env if record else None) as bc:
        new, stop = run_row(env, reset, make, base, fingerprint, want, hull)
    if stop:
        return None, stop
    diffs = same_fields(r, new)
    res.update(diffs=diffs, seconds=new["seconds"], same=not diffs and res["mapping"] is not False and res["hand_path"] is not False,
               stored_label=r.get("label"), label=new.get("label"), error=new.get("error"))
    if record:
        res["box_contacts"] = bc.summary()
    if "replan" in new:
        res["replan"] = new["replan"]
    return res, None


def g1_case(prims: Any, rec1: dict[str, Any], seed: int, world: Any, stored: dict[str, Any], source: str,
            entry: dict[str, Any] | None = None, record: bool = False) -> dict[str, Any]:
    """G1 for one stored case of ``source`` (pilot or poseset): nothing when it has no simulated row; else the case's
    setup (curve_graspgen._setup: the draw's destination), its reset (Stage 1's fingerprint, the record's hulls for a
    pilot case), the record's base (commit_std_base, base_gate), then each simulated row in the record's order (_g1_row:
    a pilot GraspGen row's hand path after its own reset, then the row after its own reset; with ``record``, each row's
    box contacts, BoxContacts). A check that fails stops the case; an exception outside a row's rollout is the case's
    error."""
    from robots.mujoco_lab import com_gap
    from robots.mujoco_lab.curve_poseset import _err, base_gate, commit_std_base

    o, idx = stored["object"], int(stored["idx"])
    out: dict[str, Any] = {"object": o, "idx": idx, "side": stored.get("side"), "source": source, "rows": [], "stop": None, "error": None}
    rows = [r for r in stored["rows"] if "seconds" in r]
    if not rows:
        return out
    if (rec1["object"], int(rec1["idx"])) != (o, idx):
        return dict(out, stop="the Stage 1 record is another case's")
    if rec1["fingerprint"] != stored["fingerprint"]:
        return dict(out, stop="Stage 1's fingerprint is not the record's")
    if source == "pilot" and (entry is None or entry.get("error") or entry.get("grasps") is None):
        return dict(out, stop="no grasps for this case")
    env = prims.env
    st = gg._setup(prims, rec1, seed, world)
    if st["stop"]:
        return dict(out, stop=st["stop"])
    reset, want = st["reset_at"](0), rec1["fingerprint"]
    hull = stored.get("hull_sha") if source == "pilot" else None
    try:
        _, stop = _checked_reset(env, reset, com_gap.fingerprint, want, hull)
        if stop:
            return dict(out, stop=stop)
        base = commit_std_base(env, o, st["destination"])
        if _js(base) != _js(stored.get("base")):
            return dict(out, stop=f"the base {base} differs from the record's")
        why = base_gate(rec1, base)
        if why:
            return dict(out, stop=why)
        for r in rows:
            res, stop = _g1_row(env, reset, r, source, entry, base, com_gap.fingerprint, want, hull, record)
            if stop:
                out["stop"] = f"{r['id']}: {stop}"
                break
            out["rows"].append(res)
    except Exception as exc:  # noqa: BLE001  a case that raises outside a row's rollout is this case's error
        out["error"] = _err(exc)
    return out


def g1_line(c: dict[str, Any]) -> str:
    """A case's log line: counts only, no label."""
    bad = sum(not r["same"] for r in c["rows"])
    return (f"{c['source']} {c['object']} {c['idx']}: {len(c['rows'])} rows replayed, {bad} differ, {c.get('seconds', 0.0):.1f} s"
            + (f", STOP {c['stop']}" if c["stop"] else "") + (f", error {c['error']}" if c["error"] else ""))


def g1_object(job: tuple[str, list[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]]) -> list[dict[str, Any]]:
    """One object's stored cases, in the record's order, on its fold's bundle with the run's sensor and world
    (curve_graspgen.graspgen_object's set-up); the pilot's grasps from the request step's files; the executor's re-plan at
    the hold set from the job's flag for its cases."""
    from robots.mujoco_lab import calibrate as cal
    from robots.mujoco_lab.prior import fragility_world

    o, items, s = job
    entries = gg.load_grasps(s["grasps"]) if s["source"] == "pilot" else {}
    prims = cal._primitives(s["bundle"], None, None)
    if getattr(prims, "reset_s", None) is not None:
        raise RuntimeError("G1 replays the absorbing attempt only (this bundle restores the object after damage)")
    probes, prims.probe_models = getattr(prims, "probe_models", None), {}
    replan0, prims.env.replan_hold = getattr(prims.env, "replan_hold", False), bool(s.get("replan"))
    prims.force_noise = cal.job_force_noise(None, prims.manifest)
    prims.env.force_noise = prims.force_noise
    world = fragility_world(prims.manifest)
    out = []
    try:
        for rec1, stored in items:
            t0 = time.time()
            c = g1_case(prims, rec1, int(s["seed"]), world, stored, s["source"], entries.get((o, int(stored["idx"]))),
                        record=bool(s.get("record")))
            c["seconds"] = round(time.time() - t0, 2)
            out.append(c)
            print(g1_line(c), file=sys.stderr, flush=True)
    finally:
        prims.probe_models, prims.env.replan_hold = probes, replan0
    return out


def g1_metrics(done: list[dict[str, Any]]) -> dict[str, Any]:
    """G1's counts, no label: cases, rows replayed, rows that differ (by field), mapping and hand-path failures, stops,
    case errors, times, the counts by source and side, and whether G1 passed (rows replayed, none differs, no stop, no
    error)."""
    rows = [r for c in done for r in c["rows"]]
    by: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"cases": 0, "rows": 0, "mismatches": 0}))
    for c in done:
        b = by[c["source"]][c["side"]]
        b["cases"] += 1
        b["rows"] += len(c["rows"])
        b["mismatches"] += sum(not r["same"] for r in c["rows"])
    m = {"cases": len(done), "rows": len(rows), "mismatches": sum(not r["same"] for r in rows),
         "by_field": dict(Counter(k for r in rows for k in r.get("diffs") or {})),
         "mapping_failures": sum(r["mapping"] is False for r in rows), "hand_path_failures": sum(r["hand_path"] is False for r in rows),
         "stops": sum(bool(c["stop"]) for c in done), "case_errors": sum(bool(c["error"]) for c in done),
         "row_seconds": round(sum(float(r.get("seconds") or 0.0) for r in rows), 1),
         "case_seconds": round(sum(float(c.get("seconds") or 0.0) for c in done), 1),
         "by_source": {s: dict(v) for s, v in by.items()}}
    rec = [r for r in rows if "box_contacts" in r]
    m.update(rows_recorded=len(rec), rows_with_box_contact=sum(r["box_contacts"]["first"] is not None for r in rec),
             changed_without_box_contact=sum(not r["same"] and r["box_contacts"]["first"] is None for r in rec))
    m["passed"] = m["rows"] > 0 and m["mismatches"] == 0 and m["stops"] == 0 and m["case_errors"] == 0
    return m


G1_EXPECTED = {"pilot": 1756, "poseset": 2344 + 2465}   # the stored simulated rows: the pilot's; the pose set's test and validation


def g1_unit_ok(m: dict[str, Any], allow_diffs: bool) -> bool:
    """A g1 unit's exit from its metrics: G1's pass; with ``allow_diffs`` (gate G1′, contacts on, where the audit's
    flagged rows may change) rows replayed with no stop, no case error and no mapping or hand-path failure."""
    if not allow_diffs:
        return bool(m["passed"])
    return m["rows"] > 0 and not any(m[k] for k in ("stops", "case_errors", "mapping_failures", "hand_path_failures"))


def _load_units(units: list[str]) -> tuple[list[dict[str, Any]], list[str], set[str]]:
    """The cases of the units (g1 run dirs), the units whose run did not end with EXIT 0, and their code revisions."""
    done: list[dict[str, Any]] = []
    bad_exit, revs = [], set()
    for u in units:
        ex = os.path.join(u, "run.exit")
        if not os.path.exists(ex) or open(ex).read().strip() != "EXIT 0":
            bad_exit.append(u)
        rp, cp_ = os.path.join(u, "g1_records.jsonl"), os.path.join(u, "run_config.json")
        if os.path.exists(rp):
            done += [json.loads(line) for line in open(rp)]
        if os.path.exists(cp_):
            revs.add(json.load(open(cp_)).get("code_revision"))
    return done, bad_exit, revs


def _changed_fields(r: dict[str, Any]) -> list[str]:
    """A replayed row's differing fields, then its failed geometry checks."""
    return sorted(r.get("diffs") or {}) + [k for k in ("mapping", "hand_path") if r[k] is False]


def g1_report(units: list[str], expected: dict[str, int]) -> dict[str, Any]:
    """G1 over its units (g1 run dirs): g1_metrics over every case of every unit, the rows replayed against the rows each
    source holds (``expected``), the units whose run did not end with EXIT 0, the rows that differ (their fields, no
    value) and the stopped cases. Passes when every unit ended with EXIT 0, every expected row was replayed and matched,
    and no case stopped or raised."""
    done, bad_exit, revs = _load_units(units)
    m = g1_metrics(done)
    got = Counter(c["source"] for c in done for _ in c["rows"])
    m.update(units=len(units), units_not_exit_0=bad_exit, of_rows=dict(expected),
             missing_rows={s: int(expected[s]) - got.get(s, 0) for s in expected}, code_revisions=sorted(r for r in revs if r),
             mismatched_rows=[{"source": c["source"], "side": c["side"], "object": c["object"], "idx": c["idx"], "id": r["id"],
                               "fields": _changed_fields(r)}
                              for c in done for r in c["rows"] if not r["same"]],
             stopped_cases=[{k: c[k] for k in ("source", "side", "object", "idx", "stop")} for c in done if c["stop"]])
    m["passed"] = bool(m["passed"] and not bad_exit and all(v == 0 for v in m["missing_rows"].values()))
    return m


TO_CLOSE = ("untraced", "approach", "close")    # the recorder's phases up to the close: the audit's scope for the object
CAUSES = ("contact only", "raise only", "both")   # what may change a row in G1′ with the re-plan on (amendment 8, rule (g))
D_BINS = (("≤ 1 mm", -math.inf, 1.0), ("1 to 3 mm", 1.0, 3.0), ("3 to 10 mm", 3.0, 10.0), ("> 10 mm", 10.0, math.inf))  # d, (lo, hi]


def _unit_config(u: str) -> dict[str, Any]:
    p = os.path.join(u, "run_config.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def _raised(r: dict[str, Any]) -> bool:
    return float((r.get("replan") or {}).get("raise_mm") or 0.0) > 0.0


def row_cause(r: dict[str, Any]) -> str | None:
    """What may have changed a replayed row (CAUSES): its first box contact (BoxContacts), a raise of the carry, both;
    None without either."""
    c, up = (r.get("box_contacts") or {}).get("first") is not None, _raised(r)
    return "both" if c and up else "contact only" if c else "raise only" if up else None


def g1p_report(units: list[str], flagged: set[tuple[Any, ...]], expected: dict[str, int],
               first_pass: list[str] | None = None) -> dict[str, Any]:
    """Revised gate G1′ (design amendment 8, ruling (b): a row changes only if a palm, wrist or finger box touches
    something during the rollout) over its units: g1_report's counts, then the gate: the changed rows (a differing field
    or a failed geometry check), those whose recorder (BoxContacts) saw a box contact, and those it saw without one (their
    fields, labels, error and whether a rollout ran). Beside the gate, not as pass conditions, the palm-contact audit's
    list (``flagged``: (source, side, object, idx, id) of each row it flagged), read as the prediction for box contacts
    before the close: the flagged rows changed and identical; the unflagged rows identical and changed, the changed ones
    by the phase of their first box contact and by that contact (phase, box, partner); the flagged and the unflagged
    rows with a box contact up to the close (the untraced free move and settle, the approach, the close), and those of
    them deeper than 1 mm (the audit's primary threshold); the identical rows that show a box contact; the label transitions (stored → new) of the changed rows by object; the flagged rows changed by transition
    and by object; every changed row that the list lacks (its fields, labels, error and first contact) and the list's
    rows that no unit replayed. With the executor's re-plan at the hold on (the units' run_config replan_hold, rule (g)),
    a row may also change through a raise: the gate counts the changed rows by cause (CAUSES) with their label
    transitions, and ``replan`` reads the records: the units without the flag, the rows without a record (a row that
    raised has none), the rows not lifted, lifted but not held, and lifted and held (the rows the rule reads), the
    raised and the unreachable rows, the identical rows with a raise, d in D_BINS over the lifted and held rows with the
    raises and the unreachable rows per bin, the raises (mm) and the raised rows by object. With ``first_pass`` (the
    contacts-only pass's units), beside the gate: the rows without a raise whose differing fields, label and error equal
    that pass's, differ from them, or have no row there. Passes when every unit ended with EXIT 0, every expected row was
    replayed, no case stopped or raised, no mapping or hand-path check failed, the recorder saw every row, every changed
    row shows a box contact or a raise and, with the re-plan on, every unit had the flag and every row that ran carries
    its record."""
    rep = g1_report(units, expected)
    done, _, _ = _load_units(units)
    rows = [(c, r) for c in done for r in c["rows"]]

    def key(c: dict[str, Any], r: dict[str, Any]) -> tuple[Any, ...]:
        return (c["source"], c["side"], c["object"], int(c["idx"]), r["id"])

    def where(c: dict[str, Any], r: dict[str, Any]) -> dict[str, Any]:
        return {"source": c["source"], "side": c["side"], "object": c["object"], "idx": int(c["idx"]), "id": r["id"]}

    def first(r: dict[str, Any]) -> dict[str, Any] | None:
        return (r.get("box_contacts") or {}).get("first")

    def phase(r: dict[str, Any]) -> str:
        return first(r)["phase"] if first(r) else ("none" if "box_contacts" in r else "not recorded")

    def to_close(part: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, int]:
        deep = [max((v["deepest_mm"] for k, v in (r.get("box_contacts") or {}).get("by", {}).items() if k.split("|")[0] in TO_CLOSE),
                    default=None) for _, r in part]
        return {"rows": len(part), "contact": sum(x is not None for x in deep), "contact_over_1mm": sum(x is not None and x > 1.0 for x in deep)}

    changed = [(c, r) for c, r in rows if not r["same"]]
    bare = [(c, r) for c, r in changed if "box_contacts" in r and row_cause(r) is None]
    unf = [(c, r) for c, r in rows if key(c, r) not in flagged]
    fl = [(c, r) for c, r in rows if key(c, r) in flagged]
    cu, cf = [(c, r) for c, r in unf if not r["same"]], [(c, r) for c, r in fl if not r["same"]]
    trans: dict[str, Counter] = defaultdict(Counter)
    for c, r in changed:
        trans[c["object"]][f"{r.get('stored_label')} → {r.get('label')}"] += 1
    by_cause: dict[str, Counter] = defaultdict(Counter)
    for c, r in changed:
        if row_cause(r):
            by_cause[row_cause(r)][f"{r.get('stored_label')} → {r.get('label')}"] += 1
    rep.update(rows_not_recorded=sum("box_contacts" not in r for _, r in rows),
               gate={"changed": len(changed), "changed_with_contact": sum(first(r) is not None for _, r in changed),
                     "by_cause": {k: sum(by_cause[k].values()) for k in CAUSES},
                     "changed_without_cause": [dict(where(c, r), fields=_changed_fields(r), stored_label=r.get("stored_label"),
                                                    label=r.get("label"), error=r.get("error"), observed=r["box_contacts"]["observed"])
                                               for c, r in bare]},
               transitions_by_cause={k: dict(by_cause[k]) for k in CAUSES if by_cause[k]},
               flagged={"rows": len(fl), "changed": len(cf), "identical": len(fl) - len(cf)},
               unflagged={"rows": len(unf), "identical": len(unf) - len(cu), "changed": len(cu),
                          "changed_by_first_phase": dict(Counter(phase(r) for _, r in cu))},
               list_vs_contact_to_close={"flagged": to_close(fl), "unflagged": to_close(unf)},
               unflagged_changed_by_first_contact=dict(Counter(f"{f['phase']} {f['box']}/{f['partner']}" if (f := first(r)) else phase(r)
                                                               for _, r in cu)),
               identical_with_contact=[dict(where(c, r), first_contact=first(r)) for c, r in rows if r["same"] and first(r)],
               transitions_by_object={o: dict(t) for o, t in sorted(trans.items())},
               flagged_by_transition=dict(Counter(f"{r.get('stored_label')} → {r.get('label')}" for _, r in cf)),
               flagged_by_object=dict(Counter(c["object"] for c, _ in cf)),
               changed_not_in_list=[dict(where(c, r), fields=_changed_fields(r), stored_label=r.get("stored_label"), label=r.get("label"),
                                         error=r.get("error"), first_contact=first(r)) for c, r in cu],
               flagged_not_replayed=[list(k) for k in sorted(flagged - {key(c, r) for c, r in rows}, key=repr)])
    on = [u for u in units if _unit_config(u).get("replan_hold")]
    rep["replan"] = None
    if on:
        recs = [(c, r, r["replan"]) for c, r in rows if r.get("replan")]
        held = [(c, r, p) for c, r, p in recs if p["lifted"] and p["held"] and p["d_mm"] is not None]
        bins = {name: {"rows": 0, "raised": 0, "unreachable": 0} for name, _, _ in D_BINS}
        for _, r, p in held:
            b = bins[next(name for name, lo, hi in D_BINS if lo < p["d_mm"] <= hi)]
            b["rows"] += 1
            b["raised"] += int(_raised(r))
            b["unreachable"] += int(bool(p["unreachable"]))
        ups = sorted(float(p["raise_mm"]) for _, r, p in recs if _raised(r))
        rep["replan"] = {"band_mm": 1e3 * REPLAN_BAND_M, "units": len(on), "units_without_flag": [u for u in units if u not in on],
                         "rows": len(rows), "with_record": len(recs),
                         "without_record": [where(c, r) for c, r in rows if not r.get("replan") and r.get("error") is None],
                         "not_lifted": sum(not p["lifted"] for _, _, p in recs),
                         "lifted_not_held": sum(bool(p["lifted"]) and not p["held"] for _, _, p in recs), "eligible": len(held),
                         "raised": len(ups), "unreachable": sum(bool(p["unreachable"]) for _, _, p in recs),
                         "identical_with_raise": sum(bool(r["same"]) for _, r, _ in recs if _raised(r)), "bins": bins,
                         "raise_mm": {"min": ups[0], "median": float(np.median(ups)), "max": ups[-1]} if ups else None,
                         "raised_by_object": dict(Counter(c["object"] for c, r, _ in recs if _raised(r)))}
    rep["vs_first_pass"] = None
    if first_pass is not None:
        fdone, _, _ = _load_units(first_pass)
        fby = {key(c, r): r for c in fdone for r in c["rows"]}

        def sig(r: dict[str, Any]) -> str:
            return _js([r.get("diffs") or {}, r.get("label"), r.get("error")])

        nr = [(c, r) for c, r in rows if not _raised(r)]
        missing = sum(key(c, r) not in fby for c, r in nr)
        differ = [dict(where(c, r), first_fields=sorted(fby[key(c, r)].get("diffs") or {}), fields=sorted(r.get("diffs") or {}),
                       first_label=fby[key(c, r)].get("label"), label=r.get("label"))
                  for c, r in nr if key(c, r) in fby and sig(fby[key(c, r)]) != sig(r)]
        rep["vs_first_pass"] = {"rows_without_raise": len(nr), "equal": len(nr) - missing - len(differ), "differ": differ, "missing": missing}
    rp = rep["replan"]
    rep["passed"] = bool(not rep["units_not_exit_0"] and all(v == 0 for v in rep["missing_rows"].values()) and rep["rows"] > 0
                         and not any(rep[k] for k in ("stops", "case_errors", "mapping_failures", "hand_path_failures"))
                         and rep["rows_not_recorded"] == 0 and not bare
                         and (rp is None or not (rp["units_without_flag"] or rp["without_record"])))
    return rep


MD_LIST = 40      # the md lists at most this many rows of an informational list; the json holds them all


def _contact_text(f: dict[str, Any] | None) -> str:
    if not f:
        return "no box contact"
    at = f" at {f['t']:.2f} s" if f.get("t") is not None else ""
    return f"first contact {f['phase']} {f['box']}/{f['partner']} {f['mm']} mm{at}"


def _row_text(x: dict[str, Any], contact: str) -> str:
    return (f"- {x['source']} {x['side']} {x['object']} {x['idx']} {x['id']}: fields {', '.join(x['fields'])}; {contact};"
            f" {x['stored_label']} → {x['label']}" + (f"; error {x['error']}" if x["error"] else ""))


def g1p_report_cmd(a: argparse.Namespace) -> dict[str, Any]:
    """The g1p-report command: g1p_report over --units or every unit_* dir of --queue, with the audit's list --flagged
    (flagged_rows.jsonl; its pilot and pose-set lines) and, with --first-pass, every unit_* dir of the contacts-only pass's
    queue; g1p_summary.json and g1p_summary.md in --out, wandb. Exit 3 when G1′ fails."""
    import glob

    from robots.mujoco_lab.curve import _sha
    from robots.mujoco_lab.tracking import log_run

    def queue(q: str) -> list[str]:
        return sorted(glob.glob(os.path.join(q, "unit_*")), key=lambda d: int(d.rsplit("_", 1)[1]))

    units = a.units.split(",") if a.units else queue(a.queue)
    lines = [json.loads(line) for line in open(a.flagged) if line.strip()]
    flagged = {(x["source"], x["side"], x["object"], int(x["idx"]), x["id"]) for x in lines if x["source"] in SOURCES}
    rep = g1p_report(units, flagged, G1_EXPECTED, first_pass=queue(a.first_pass) if a.first_pass else None)
    rep["flagged_list"] = f"{a.flagged} (sha256 {_sha(a.flagged)}), {len(flagged)} pilot and pose-set rows"
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "g1p_summary.json"), "w") as f:
        json.dump(rep, f, indent=1)
    g, u, fl, tc, rp, vf = (rep[k] for k in ("gate", "unflagged", "flagged", "list_vs_contact_to_close", "replan", "vs_first_pass"))
    md = ["# Gate G1′, revised (amendment 8: a row changes only if a palm, wrist or finger box touches something"
          + (", or the re-plan at the hold raises its carry)" if rp else ")"), "",
          f"Result: {'PASSED' if rep['passed'] else 'FAILED'}.", "", "## The gate", "",
          f"- Changed rows (a differing label, qpos_sha or trace_sha, or a failed geometry check): {g['changed']}. With a box contact:"
          f" {g['changed_with_contact']}. By cause: {', '.join(f'{k} {n}' for k, n in g['by_cause'].items())}. Without a box contact"
          f" or a raise: {len(g['changed_without_cause'])}.",
          f"- Rows the recorder did not see: {rep['rows_not_recorded']}.",
          f"- Rows replayed {rep['rows']} of {sum(rep['of_rows'].values())}; stopped cases {rep['stops']}; case errors {rep['case_errors']};"
          f" mapping failures {rep['mapping_failures']}; hand-path failures {rep['hand_path_failures']}; units not ending with EXIT 0:"
          f" {len(rep['units_not_exit_0'])} of {rep['units']}.",
          f"- Code: {', '.join(rep['code_revisions'])}.", ""]
    if rp:
        md.insert(md.index("- Rows the recorder did not see: " + str(rep["rows_not_recorded"]) + ".") + 1,
                  f"- The re-plan: units with the flag {rp['units']} of {rep['units']}; rows that ran without a record"
                  f" {len(rp['without_record'])}.")
    md += [_row_text(x, "no box contact and no raise") + ("" if x["observed"] else "; no rollout") for x in g["changed_without_cause"]]
    md += ["", "### Label transitions of the changed rows, by cause (stored → new)", "", "| cause | transition | rows |", "|---|---|---|"]
    md += [f"| {k} | {t} | {n} |" for k, ts in rep["transitions_by_cause"].items() for t, n in sorted(ts.items(), key=lambda x: -x[1])]
    if rp:
        rm = rp["raise_mm"]
        md += ["", f"## The re-plan at the hold (dead band {rp['band_mm']:g} mm; d = planned − measured offset of the lowest point)", "",
               f"- Rows with a record: {rp['with_record']} of {rp['rows']}. Not lifted: {rp['not_lifted']}. Lifted but not held by both"
               f" pads at the end of the hold: {rp['lifted_not_held']}. Lifted and held (the rows the rule reads): {rp['eligible']}.",
               f"- Raised: {rp['raised']}" + (f" (min {rm['min']:.1f}, median {rm['median']:.1f}, max {rm['max']:.1f} mm)" if rm else "")
               + f". Raise unreachable: {rp['unreachable']}. Identical rows with a raise: {rp['identical_with_raise']}.", "",
               "| d (lifted and held rows) | rows | raised | unreachable |", "|---|---|---|---|"]
        md += [f"| {name} | {b['rows']} | {b['raised']} | {b['unreachable']} |" for name, b in rp["bins"].items()]
        md += ["", "| object | raised rows |", "|---|---|"]
        md += [f"| {o} | {n} |" for o, n in sorted(rp["raised_by_object"].items(), key=lambda x: -x[1])]
    if vf:
        md += ["", f"- Rows without a raise equal to the contacts-only pass: {vf['equal']} of {vf['rows_without_raise']} (differ"
               f" {len(vf['differ'])}, missing {vf['missing']})."]
        md += [f"- {x['source']} {x['side']} {x['object']} {x['idx']} {x['id']}: contacts-only pass fields {', '.join(x['first_fields']) or 'none'},"
               f" {x['first_label']}; now fields {', '.join(x['fields']) or 'none'}, {x['label']}" for x in vf["differ"][:MD_LIST]]
    md += ["", "## Beside the gate: the palm-contact audit's list (the prediction for box contacts before the close)", "",
           f"- Flagged rows changed: {fl['changed']} of {fl['rows']} (identical {fl['identical']}).",
           f"- Unflagged rows changed: {u['changed']} of {u['rows']} (identical {u['identical']}).",
           f"- Identical rows that show a box contact: {len(rep['identical_with_contact'])}.",
           "- The list as the prediction for box contacts up to the close (the free move, the settle, the approach, the close):"
           f" flagged rows with one {tc['flagged']['contact']} of {tc['flagged']['rows']} (deeper than 1 mm"
           f" {tc['flagged']['contact_over_1mm']}); unflagged rows with one {tc['unflagged']['contact']} of {tc['unflagged']['rows']}"
           f" (deeper than 1 mm {tc['unflagged']['contact_over_1mm']}).",
           f"- The list: {rep['flagged_list']}; rows of it not replayed: {len(rep['flagged_not_replayed'])}.", "",
           "### Unflagged rows that changed, by the phase of the first box contact", "", "| phase | rows |", "|---|---|"]
    md += [f"| {k} | {n} |" for k, n in sorted(u["changed_by_first_phase"].items(), key=lambda x: -x[1])]
    md += ["", "### Unflagged rows that changed, by the first box contact", "", "| phase box/partner | rows |", "|---|---|"]
    md += [f"| {k} | {n} |" for k, n in sorted(rep["unflagged_changed_by_first_contact"].items(), key=lambda x: -x[1])]
    md += ["", "### Label transitions of the changed rows, by object (stored → with contacts)", "", "| object | transition | rows |", "|---|---|---|"]
    md += [f"| {o} | {t} | {n} |" for o, ts in rep["transitions_by_object"].items() for t, n in sorted(ts.items(), key=lambda x: -x[1])]
    md += ["", "### Flagged rows that changed, by label transition", "", "| transition | rows |", "|---|---|"]
    md += [f"| {t} | {n} |" for t, n in sorted(rep["flagged_by_transition"].items(), key=lambda x: -x[1])]
    md += ["", "### Flagged rows that changed, by object", "", "| object | rows |", "|---|---|"]
    md += [f"| {o} | {n} |" for o, n in sorted(rep["flagged_by_object"].items(), key=lambda x: -x[1])]
    for title, xs, text in (("Identical rows that show a box contact", rep["identical_with_contact"],
                             lambda x: f"- {x['source']} {x['side']} {x['object']} {x['idx']} {x['id']}: {_contact_text(x['first_contact'])}"),
                            ("Changed rows that the list does not hold", rep["changed_not_in_list"],
                             lambda x: _row_text(x, _contact_text(x["first_contact"])))):
        md += ["", f"### {title}: {len(xs)}", ""] + [text(x) for x in xs[:MD_LIST]]
        md += [f"- … {len(xs) - MD_LIST} more in g1p_summary.json"] if len(xs) > MD_LIST else []
    with open(os.path.join(a.out, "g1p_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    rep["wandb"] = log_run("sixdof-g1p-report", {"units": units, "expected": G1_EXPECTED, "flagged": a.flagged},
                           {"passed": rep["passed"], "rows": rep["rows"], "rows_not_recorded": rep["rows_not_recorded"],
                            "changed": g["changed"], "changed_with_contact": g["changed_with_contact"], "by_cause": g["by_cause"],
                            "changed_without_cause": len(g["changed_without_cause"]), "unflagged": u, "flagged": fl,
                            "replan": None if rp is None else {k: v for k, v in rp.items() if k not in ("without_record", "units_without_flag")},
                            "vs_first_pass": None if vf is None else {k: v for k, v in vf.items() if k != "differ"},
                            "identical_with_contact": len(rep["identical_with_contact"]), "stops": rep["stops"], "case_errors": rep["case_errors"]})
    print("\n".join(md), flush=True)
    if not rep["passed"]:
        raise SystemExit(3)
    return rep


def g1_report_cmd(a: argparse.Namespace) -> dict[str, Any]:
    """The g1-report command: g1_report over --units (comma-separated) or every unit_* dir of --queue; g1_summary.json
    and g1_summary.md in --out (counts and field names only; wandb). Exit 3 when G1 fails."""
    import glob

    from robots.mujoco_lab.tracking import log_run

    units = a.units.split(",") if a.units else sorted(glob.glob(os.path.join(a.queue, "unit_*")), key=lambda d: int(d.rsplit("_", 1)[1]))
    rep = g1_report(units, G1_EXPECTED)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "g1_summary.json"), "w") as f:
        json.dump(rep, f, indent=1)
    lines = ["# Gate G1", "", f"Result: {'PASSED' if rep['passed'] else 'FAILED'}.", "",
             "| source | side | cases | rows replayed | rows that differ |", "|---|---|---|---|---|"]
    lines += [f"| {s} | {side} | {b['cases']} | {b['rows']} | {b['mismatches']} |" for s, v in sorted(rep["by_source"].items())
              for side, b in sorted(v.items())]
    lines += ["", f"Rows replayed {rep['rows']} of {sum(rep['of_rows'].values())} ({', '.join(f'{s} {n}' for s, n in rep['of_rows'].items())});"
              f" rows that differ {rep['mismatches']} (by field {rep['by_field'] or 'none'}); mapping failures {rep['mapping_failures']};"
              f" hand-path failures {rep['hand_path_failures']}; stopped cases {rep['stops']}; case errors {rep['case_errors']};"
              f" units not ending with EXIT 0: {len(rep['units_not_exit_0'])} of {rep['units']}.",
              f"Code: {', '.join(rep['code_revisions'])}. Row time {rep['row_seconds'] / 3600:.2f} CPU-h, case time"
              f" {rep['case_seconds'] / 3600:.2f} CPU-h."]
    with open(os.path.join(a.out, "g1_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    rep["wandb"] = log_run("sixdof-g1-report", {"units": units, "expected": G1_EXPECTED}, rep)
    print("\n".join(lines), flush=True)
    if not rep["passed"]:
        raise SystemExit(3)
    return rep


def g1(a: argparse.Namespace) -> str:
    """The G1 command: the stored records of --runs (one source's finished runs; the same Stage 1, seed, bundle and, for
    the pilot, grasps), their cases of --objects and --sides by object in the records' order, each replayed (g1_object);
    g1_records.jsonl (one line per case), run_config.json and g1_metrics.json (counts only; wandb). Exit 3 when G1 fails;
    with --allow-diffs (gate G1′), when a case stopped or raised or a geometry check failed. Returns --out."""
    from robots.mujoco_lab.curve import SIDES, _peak_rss_mb, _sha, load_stage1, records_path
    from robots.mujoco_lab.estimation_check import _revision
    from robots.mujoco_lab.tracking import log_run

    os.makedirs(a.out, exist_ok=True)
    rec_path = os.path.join(a.out, "g1_records.jsonl")
    if os.path.exists(rec_path):
        raise SystemExit(f"{rec_path} exists: pick a new --out")
    runs = a.runs.split(",")
    for d in runs:
        if not os.path.exists(os.path.join(d, "run_metrics.json")):
            raise SystemExit(f"{d}: the run has not finished (no run_metrics.json)")
    cfgs = [json.load(open(os.path.join(d, "run_config.json"))) for d in runs]
    keys = ("stage1", "seed", "bundle") + (("grasps",) if a.source == "pilot" else ())
    for k in keys:
        if len({_js(c.get(k)) for c in cfgs}) > 1:
            raise SystemExit(f"the runs differ in {k}")
    cfg = cfgs[0]
    sides = set(a.sides.split(","))
    objects = set(a.objects.split(",")) if a.objects else None
    paths = [records_path(d, side) for d in runs for side in SIDES if side in sides and os.path.exists(records_path(d, side))]
    stored = [json.loads(line) for p in paths for line in open(p)]
    stored = [r for r in stored if objects is None or r["object"] in objects]
    recs1, _ = load_stage1(cfg["stage1"].split(","))
    by1 = {(r["object"], int(r["idx"])): r for r in recs1}
    by: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for r in stored:
        by[r["object"]].append((by1[(r["object"], int(r["idx"]))], r))
    config = {**vars(a), **_revision(), **{k: cfg.get(k) for k in keys}, "records_sha256": {p: _sha(p) for p in paths},
              "cases": [(r["object"], int(r["idx"])) for r in stored], "rows": sum("seconds" in x for r in stored for x in r["rows"])}
    with open(os.path.join(a.out, "run_config.json"), "w") as f:
        json.dump(config, f, indent=1)
    jobs = [(o, items, {"bundle": os.path.join(cfg["bundle"], "folds", f"fold_{items[0][0]['fold']}"), "seed": cfg["seed"],
                        "source": a.source, "grasps": cfg.get("grasps"), "record": a.box_contacts, "replan": a.replan_hold})
            for o, items in by.items()]
    t0 = time.time()
    with open(rec_path, "w") as f:
        for job in jobs:
            for c in g1_object(job):
                f.write(json.dumps(c) + "\n")
                f.flush()
    done = [json.loads(line) for line in open(rec_path)]
    metrics = dict(g1_metrics(done), of_rows=config["rows"], wall_seconds=round(time.time() - t0, 1), peak_rss_mb=_peak_rss_mb())
    metrics["wandb"] = log_run("sixdof-g1", config, metrics)
    with open(os.path.join(a.out, "g1_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print(f"done: {json.dumps(metrics)}", flush=True)
    if not g1_unit_ok(metrics, a.allow_diffs):
        raise SystemExit(3)
    return a.out


def main(argv: list[str] | None = None) -> Any:
    ap = argparse.ArgumentParser(prog="curve_sixdof", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("g1", help="gate G1: the stored rows of one source's runs again through the new executor path")
    g.add_argument("--source", required=True, choices=SOURCES)
    g.add_argument("--runs", required=True, help="comma-separated finished run dirs of that source")
    g.add_argument("--objects", default=None, help="comma-separated objects (default: all)")
    g.add_argument("--sides", default="validation,test")
    g.add_argument("--allow-diffs", action="store_true", help="gate G1′: rows may differ (the exit counts stops, errors and geometry checks)")
    g.add_argument("--box-contacts", action="store_true", help="revised gate G1′: record each row's palm, wrist and finger-box contacts")
    g.add_argument("--replan-hold", action="store_true", help="the executor's re-plan at the hold on (amendment 8): each row records it")
    g.add_argument("--out", required=True)
    r = sub.add_parser("g1-report", help="G1 over its units: the summary (counts and field names only)")
    r.add_argument("--units", default=None, help="comma-separated g1 run dirs")
    r.add_argument("--queue", default=None, help="a queue dir: every unit_* dir in it")
    r.add_argument("--out", required=True)
    p = sub.add_parser("g1p-report", help="gate G1′ over its units, with the palm-contact audit's list")
    p.add_argument("--units", default=None, help="comma-separated g1 run dirs")
    p.add_argument("--queue", default=None, help="a queue dir: every unit_* dir in it")
    p.add_argument("--flagged", required=True, help="the audit's flagged_rows.jsonl")
    p.add_argument("--first-pass", default=None, help="the contacts-only pass's queue dir: the rows without a raise are checked against it")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"g1": g1, "g1-report": g1_report_cmd, "g1p-report": g1p_report_cmd}[a.cmd](a)


if __name__ == "__main__":
    main()
