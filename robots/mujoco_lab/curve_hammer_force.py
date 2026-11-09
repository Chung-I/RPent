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

"""The hammer squeeze x CoM-pinch ceiling (daily-logs design 2026-11-09-hammer-force-com-design.md, approved by the user
on 9 Nov; the coordinator's rulings of 9 Nov on the builder's ten choices).

On each of the pose set's hammer cases (its Stage 1 record, reset seed, destination and hidden draw), eight rows, each
run once (an attempt is deterministic per case: the reset seed sets only the sensor-noise streams): two across pinches,
the menu's mid pinch ``across_0.5`` (the control) and ``across_com`` at the true balance fraction com_f
(curve_poseset.balance_fraction of env.truth()'s com_obj: an oracle row), each at four squeezes per finger: f_std
(commit_std's standard_force of the nominal mass), 2 f_std (the menu's firm_grip), 40 N and 70 N. Everything else is
commit_std's (curve_poseset.commit_std_base), with Stage 2c's executor flag: the re-plan at the hold on (ruling 7). A
pinch passes the width rule and reach() as the menu's current rows do (curve_poseset.pose_rows), else it is infeasible
at every level with its reason (no substitute).

Gates: a run refuses another sim than the contacts sim (CONTACTS_SIM, no dirty file). G0: the control row (across_0.5
at f_std) must give the corrected pose-set record's label, qpos_sha and trace_sha (the published pose-set records with
the values of the G1′ pass that has this run's re-plan flag, sixdof_readings.correct_rows), else the case stops and
names the fields, no value. G1 (the smoke, --replay): every simulated row again at the case's second reset seed, 0 label
mismatches. Each row records its measured squeeze from its trace (squeeze: no hook, so no label can change). Logs and
metrics hold counts and times, no label (no early look)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np

PINCHES = ("across_0.5", "across_com")
LEVELS = ("f_std", "2f_std", "40N", "70N")
FIXED_N = {"40N": 40.0, "70N": 70.0}
CONTROL = "across_0.5@f_std"
REPLAN_HOLD = True                       # Stage 2c's executor flag (stage2c/full run_config: replan_hold True), ruling 7
CONTACTS_SIM = "ee882f9b5574a4adc30559fd92c9df3bf69afdf3"    # daily-logs: the contacts sim (amendment 8), Stage 2c's
G0_FIELDS = ("label", "qpos_sha", "trace_sha")
HAMMERS = {"test": ("handal__hammer_1", "handal__hammer_2", "handal__hammer_6", "handal__hammer_7"),
           "validation": ("handal__hammer", "handal__hammer_3", "handal__hammer_4", "handal__hammer_5", "handal__hammer_8")}


def row_id(pinch: str, level: str) -> str:
    return f"{pinch}@{level}"


def level_force(level: str, f_std: float) -> float:
    """The squeeze per finger of a level: f_std, twice it (the menu's firm_grip), 40 N or 70 N."""
    if level == "f_std":
        return float(f_std)
    if level == "2f_std":
        return 2.0 * float(f_std)
    return FIXED_N[level]


def pinch_rows(env: Any, destination: str, com_f: float, f_std: float) -> list[dict[str, Any]]:
    """The eight rows on the env's current state, PINCHES x LEVELS (the control first): per pinch the across proposal at
    its fraction (0.5, or com_f), the width rule (menu.unusable_reason), then reach() at the destination (the pose set's
    rule for its current rows); a pinch that fails either is infeasible at every level with its reason."""
    from rpent.skills.pick_place.menu import unusable_reason

    wmax = float(env.max_pinch_width)
    out = []
    for pinch in PINCHES:
        f = 0.5 if pinch == "across_0.5" else float(com_f)
        p = env.proposals((f,), turn=0.0)[0]
        why = unusable_reason(p.width, wmax)
        reach = None
        if why is None:
            r = env.reach(p, destination)
            reach = "ok" if r is None else f"{r.get('reason')} ({r.get('phase')})"
            if r is not None:
                why = f"unreachable: {reach}"
        for level in LEVELS:
            out.append({"id": row_id(pinch, level), "pinch": pinch, "level": level, "force": level_force(level, f_std), "f": f,
                        "point": [float(x) for x in p.point], "yaw": float(p.yaw), "width": float(p.width), "reach": reach,
                        "feasible": why is None, "why": why})
    return out


def _both_pads(s: dict[str, Any]) -> bool:
    return bool(all(s.get("pads") or [False]))


def squeeze(trace: list[dict[str, Any]]) -> dict[str, Any]:
    """The measured squeeze per pad from a rollout's trace (read after the rollout: no hook), as fn / 2 (fn: both pads'
    normal force on the object, the mean over the step's physics substeps). ``hold``: the median over the hold steps in
    which both pads touch the object (None without one), ``hold_contact`` those steps, ``hold_steps`` all hold steps;
    ``close``: the median over the second half of the close's steps with both pads on the object."""
    def med(steps: list[dict[str, Any]]) -> float | None:
        v = [float(s["fn"]) / 2.0 for s in steps if _both_pads(s)]
        return float(np.median(v)) if v else None

    hold = [s for s in trace if s.get("phase") == "hold"]
    close = [s for s in trace if s.get("phase") == "close"]
    return {"hold_steps": len(hold), "hold_contact": sum(_both_pads(s) for s in hold), "hold": med(hold),
            "close": med(close[len(close) // 2:])}


def monitor(ro: Any, at_rest: Any) -> dict[str, Any]:
    """The contact monitor's record of a rollout (read after it: no hook), the causes of its artifact flag
    (adapter.artifact): reversed pad contacts in squeeze substeps, the deepest pad penetration (mm), whether the reset
    reached rest, and the jam record (a jam, and whether it made the run an artifact)."""
    jam = (getattr(ro, "sensor", None) or {}).get("jam") or {}
    return {"rev_steps": int(getattr(ro, "rev_steps", 0) or 0), "deepest_mm": float(getattr(ro, "deepest_mm", 0.0) or 0.0),
            "at_rest": at_rest, "jam": bool(jam.get("jam", False)), "jam_artifact": bool(jam.get("artifact", False))}


def _outcome(env: Any, ro: Any, at_rest: Any) -> dict[str, Any]:
    from robots.mujoco_lab.curve_sixdof import outcome

    return outcome(env, ro, at_rest)


def _track(env: Any) -> tuple[list[tuple[str, list[float]]], Any]:
    """The close monitor's hook for one rollout: env._tracer wrapped so that after each control step's own on_step the
    phase and the object's position (env.sim.obj_pos) are appended to a list; it only reads (the trace, the physics and
    the hashes stay as they were: the clips' check). Returns the list and the function that removes the hook."""
    steps: list[tuple[str, list[float]]] = []
    own = "_tracer" in vars(env)
    orig = env._tracer

    def tracer(pt: Any, yaw: Any) -> Any:
        trace, on_step = orig(pt, yaw)

        def step(phase: str) -> None:
            on_step(phase)
            steps.append((phase, [float(x) for x in env.sim.obj_pos()]))
        return trace, step

    env._tracer = tracer

    def remove() -> None:
        if own:
            env._tracer = orig
        else:
            del env._tracer
    return steps, remove


def close_fields(steps: list[tuple[str, list[float]]], trace: list[dict[str, Any]]) -> dict[str, Any]:
    """The close monitor of one rollout (the coordinator's addition of 9 Nov): ``close_shift_mm``, the largest 3-D
    displacement of the object over the close's steps from where it was at the approach's end (the step before the first
    close step); ``close_turn_deg``, the object's turn since the start at the close's last step; ``lift_end_turn_deg``,
    the turn since the lift began (the hand does not turn: the turn in the pads) at the lift's last step, only when both
    pads touch the object there (``lift_end_pads``). None where the rollout has no such step."""
    close = [i for i, (ph, _) in enumerate(steps) if ph == "close"]
    shift = None
    if close:
        p0 = np.asarray(steps[max(close[0] - 1, 0)][1], dtype=float)
        shift = max(1000.0 * float(np.linalg.norm(np.asarray(steps[i][1], dtype=float) - p0)) for i in close)
    tclose = [s for s in trace if s.get("phase") == "close"]
    lift = [s for s in trace if s.get("phase") == "lift"]
    both = bool(lift) and _both_pads(lift[-1])
    return {"close_shift_mm": shift, "close_turn_deg": tclose[-1].get("rot_deg") if tclose else None,
            "lift_end_turn_deg": lift[-1].get("rot_lift_deg") if both else None, "lift_end_pads": both}


def simulate(env: Any, reset: Any, rows: list[dict[str, Any]], base: dict[str, Any], fingerprint: Any, want: str,
             close_monitor: bool = False) -> str | None:
    """Run each feasible row once, in place: reset() (the case's seed), the reset's fingerprint must be ``want`` and the
    pinch the first reset's (else the case stops: the reason is returned), then env.execute at the row's squeeze and
    ``base``'s speeds, destination and release height. Each run row gets curve_sixdof.outcome's fields (the label, the
    artifact flag, also when ``reset`` returns False, the damage and loss fields, the highest rise, the hashes of the
    final state and of the trace, the re-plan's record), its squeeze and its seconds; an exception is the row's error.
    With ``close_monitor`` each rollout runs under the read-only hook (_track) and the row gets ``close`` (close_fields)."""
    for r in rows:
        if not r["feasible"]:
            continue
        at_rest = reset()
        if fingerprint(env) != want:
            return f"{r['id']}: the reset's fingerprint differs from Stage 1's"
        p = env.proposals((r["f"],), turn=0.0)[0]
        if [float(x) for x in p.point] != r["point"] or float(p.yaw) != r["yaw"]:
            return f"{r['id']}: the pinch after the reset differs from the first reset's"
        t0 = time.time()
        r.update(label=None, artifact=False, error=None)
        try:
            steps, remove = _track(env) if close_monitor else (None, None)
            try:
                ro = env.execute(p, r["force"], base["v_lift"], base["v_transport"], base["destination"], base["release_height"],
                                 a_transport=base["a_transport"])
            finally:
                if remove is not None:
                    remove()
            r.update(_outcome(env, ro, at_rest), squeeze=squeeze(getattr(ro, "trace", None) or []), monitor=monitor(ro, at_rest))
            if close_monitor:
                r["close"] = close_fields(steps, getattr(ro, "trace", None) or [])
        except Exception as exc:  # noqa: BLE001  a simulator fault is this row's error
            r["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        r["seconds"] = round(time.time() - t0, 2)
    return None


def _js(x: Any) -> str:
    return json.dumps(x, sort_keys=True)


def g0_diff(row: dict[str, Any], ref: dict[str, Any]) -> list[str]:
    """G0: the fields of G0_FIELDS in which the control row differs from the corrected pose-set record (names only)."""
    return [k for k in G0_FIELDS if _js(row.get(k)) != _js(ref.get(k))]


def replay_rows(env: Any, reset: Any, rows: list[dict[str, Any]], base: dict[str, Any], fingerprint: Any, want: str) -> dict[str, Any]:
    """G1: the simulated rows once more, in reverse order, on copies without their outcomes (``reset`` at the second
    reset seed), each against its first run on curve_poseset.SAME. Returns the rows replayed, the mismatches ({"id",
    "fields"}: the names of the fields that differ, no value), the label mismatches and simulate's stop."""
    from robots.mujoco_lab.curve_poseset import OUTCOME, SAME

    drop = set(OUTCOME) | {"squeeze", "replan", "monitor", "close"}
    twin = [{k: v for k, v in r.items() if k not in drop} for r in reversed(rows) if "seconds" in r]
    stop = simulate(env, reset, twin, base, fingerprint, want)
    first = {r["id"]: r for r in rows}
    bad = []
    for t in twin:
        if "seconds" not in t:
            continue
        diff = [k for k in SAME if _js(first[t["id"]].get(k)) != _js(t.get(k))]
        if diff:
            bad.append({"id": t["id"], "fields": diff})
    return {"rows": sum("seconds" in t for t in twin), "mismatches": bad, "label_mismatches": sum("label" in m["fields"] for m in bad),
            "stop": stop}


def force_case(prims: Any, rec1: dict[str, Any], seed: int, world: Any, entry: dict[str, Any], replay: bool = False,
               close_monitor: bool = False) -> dict[str, Any]:
    """One case: Stage 2's skip rules on the Stage 1 record; the case's draw and reset (curve_graspgen._setup, as the G1′
    passes; Stage 1's fingerprint, else a stop); the executor's re-plan flag (REPLAN_HOLD); the balance fraction, the
    length and the truth (env.truth); commit_std's base (base_gate against Stage 1's row and equal to the pose set's,
    else a stop); the eight rows (pinch_rows: the control's pinch must be the pose set's across_0.5, else a stop); one
    simulation per feasible row after its own reset; G0 on the control row (a difference stops the case); with
    ``replay`` G1 at the second reset seed (replay_rows: a label that differs stops the case); ``close_monitor``: the
    first pass's rows run under the close monitor (simulate), the replay's do not. An exception outside a row's
    simulation is the case's error."""
    from robots.mujoco_lab import com_gap
    from robots.mujoco_lab.curve import stage2_skip
    from robots.mujoco_lab.curve_graspgen import _setup
    from robots.mujoco_lab.curve_poseset import KEEP, _err, balance_fraction, base_gate, commit_std_base

    o, idx = rec1["object"], int(rec1["idx"])
    rec: dict[str, Any] = dict({k: rec1.get(k) for k in KEEP}, hammer="hammer" in o, rows=[], base=None, com_f=None, g0=None,
                               skipped=stage2_skip(rec1), stop=None, error=None, replay=None)
    if rec["skipped"]:
        return rec
    if (entry["object"], int(entry["idx"])) != (o, idx):
        return dict(rec, stop="the case entry is another case's")
    env = prims.env
    env.replan_hold = REPLAN_HOLD
    st = _setup(prims, rec1, seed, world)
    if st["stop"]:
        return dict(rec, stop=st["stop"])
    want, reset0 = rec1["fingerprint"], st["reset_at"](0)
    try:
        reset0()
        if com_gap.fingerprint(env) != want:
            return dict(rec, stop="the reset's fingerprint differs from Stage 1's")
        t = env.truth()
        com_f = balance_fraction(env.props, t["com_obj"])
        rec.update(com_f=com_f, length=float(env.props["length"]),
                   sim_truth={k: t.get(k) for k in ("mass", "com_s", "com_obj", "fill")})
        base = commit_std_base(env, o, st["destination"])
        why = base_gate(rec1, base)
        if why:
            return dict(rec, stop=why)
        if _js(base) != _js(entry["base"]):
            return dict(rec, stop=f"the base {base} is not the pose set's {entry['base']}")
        rows = pinch_rows(env, st["destination"], com_f, base["force"])
        ctl, pin = rows[0], entry["pinch"]
        if _js([ctl["point"], ctl["yaw"], ctl["width"], ctl["feasible"]]) != _js([pin["point"], pin["yaw"], pin["width"], pin["feasible"]]):
            return dict(rec, stop="the across_0.5 pinch is not the pose set's")
        rec.update(base=base, rows=rows)
        stop = simulate(env, reset0, rows, base, com_gap.fingerprint, want, close_monitor=close_monitor)
        if stop:
            return dict(rec, stop=stop)
        rec["g0"] = g0_diff(ctl, entry["ref"]) if "seconds" in ctl else list(G0_FIELDS)
        if rec["g0"]:
            return dict(rec, stop=f"G0: the control row differs from the corrected pose-set record in {', '.join(rec['g0'])}")
        if replay:
            t1 = time.time()
            rec["replay"] = replay_rows(env, st["reset_at"](1), rows, base, com_gap.fingerprint, want)
            rec["replay_seconds"] = round(time.time() - t1, 2)
            if rec["replay"]["stop"]:
                return dict(rec, stop=f"replay: {rec['replay']['stop']}")
            if rec["replay"]["label_mismatches"]:
                return dict(rec, stop=f"replay: {rec['replay']['label_mismatches']} rows differ in the label at the second seed (G1)")
    except Exception as exc:  # noqa: BLE001  a case that raises outside a row's simulation is this case's error
        return dict(rec, error=_err(exc))
    return rec


# ---------------------------------------------------------------- the sim gate

def _git(d: str, *args: str) -> str:
    import subprocess

    return subprocess.run(["git", "--no-optional-locks", "-C", d, *args], capture_output=True, text=True, check=True).stdout


def sim_problem(sim_dir: str | None, want: str = CONTACTS_SIM) -> str | None:
    """Why the sim folder is not the contacts sim with no dirty file (None: it is)."""
    if not sim_dir:
        return "RPENT_MUJOCO_SIM_DIR is not set"
    last = _git(sim_dir, "log", "-1", "--format=%H", "--", ".").strip()
    if last != want:
        return f"the sim {sim_dir} is at {last}, not the contacts sim {want}"
    dirty = [x for x in _git(sim_dir, "status", "--porcelain", "--untracked-files=no", "--", ".").splitlines() if x.strip()]
    if dirty:
        return f"the sim {sim_dir} has {len(dirty)} dirty files"
    return None


# ---------------------------------------------------------------- the case entries (published inputs only)

def case_entry(rec: dict[str, Any], rep: dict[str, Any] | None, r4: bool | None) -> dict[str, Any]:
    """One pose-set hammer case for the run and the summary: its keys, Stage 1 fingerprint, commit_std base, balance
    fraction and hidden mass as the pose set recorded them; ``ref``: the control row's G0 fields after the G1′ pass's
    correction (sixdof_readings.correct_poseset: the stored rows with the pass's values; a simulated case without its
    replay raises ValueError); ``pinch``: the pose set's across_0.5 pinch; ``ceiling10``: the corrected 10-row ceiling's
    success (None: undetermined); ``r4``: Stage 2c's R4 success on the case (None: not a Stage 2c test case or dropped).
    A skipped case, a case with an error and a case without a simulated row carry no ref, pinch or ceiling."""
    from robots.mujoco_lab import sixdof_readings as sr

    e = {k: rec.get(k) for k in ("object", "side", "fold", "fingerprint", "skipped", "base", "com_f")}
    e.update(idx=int(rec["idx"]), mass=(rec.get("sim_truth") or {}).get("mass"), r4=r4, ref=None, pinch=None, ceiling10=None)
    if rec["skipped"] or rec.get("error") or not any("seconds" in r for r in rec.get("rows") or []):
        return e
    c = sr.correct_poseset(rec, rep)
    ctl = next(r for r in c["rows"] if r["id"] == "across_0.5")
    e.update(ref={k: ctl.get(k) for k in G0_FIELDS}, pinch={k: ctl[k] for k in ("point", "yaw", "width", "feasible")},
             ceiling10=bool(c["ceiling10"]["success"]) if c["ceiling10"]["determined"] else None)
    return e


def smoke_cases(entries: list[dict[str, Any]], k: int = 2) -> list[tuple[str, int]]:
    """The smoke's cases (ruling 10): the first ``k`` cases by idx of each object, skipped cases left out."""
    by: dict[str, list[int]] = defaultdict(list)
    for e in entries:
        if not e.get("skipped"):
            by[e["object"]].append(int(e["idx"]))
    return [(o, i) for o in sorted(by) for i in sorted(by[o])[:k]]


def _ref_path(ref: str) -> str:
    return ref.split(" (sha256 ")[0]


def cases_cmd(a: argparse.Namespace) -> str:
    """The cases file: every hammer case of the published pose set (test and validation) as case_entry, from inputs that
    must be the ones the published summaries read: the G1′ readings summary (EXIT 0) names the pose-set and pilot
    summaries and the pass (--pass, the one with REPLAN_HOLD's flag); the pose set's test records are the pose-set
    summary's, its validation records the pilot summary's; the pass's pose-set units replayed exactly those files
    (sixdof_readings.replayed_files_errors); Stage 2c's test records and lock are its summary's, its R4 key the readings'.
    Refuses an existing --out. Prints counts only."""
    from robots.mujoco_lab import curve_sixdof as cs
    from robots.mujoco_lab import sixdof_readings as sr
    from robots.mujoco_lab.curve import _sha, episode, load_stage2, records_path
    from robots.mujoco_lab.curve_poseset import _load_lines
    from robots.mujoco_lab.estimation_check import _revision
    from robots.mujoco_lab.palm_audit import ref_sha

    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    rdir = os.path.dirname(os.path.abspath(a.readings_summary))
    if open(os.path.join(rdir, "run.exit")).read().strip() != "EXIT 0":
        raise SystemExit(f"{rdir}: the G1′ readings summary did not exit 0")
    rp = json.load(open(a.readings_summary))["provenance"]
    if ref_sha(rp["poseset_summary"]) != _sha(a.poseset_summary) or ref_sha(rp["pilot_summary"]) != _sha(a.pilot_summary):
        raise SystemExit("the G1′ readings read another pose-set or pilot summary")
    name = "replan" if REPLAN_HOLD else "contacts"
    if os.path.realpath(rp["passes"][name]) != os.path.realpath(a.pass_):
        raise SystemExit(f"--pass is not the readings' {name} pass {rp['passes'][name]}")
    pp, gp = json.load(open(a.poseset_summary))["provenance"], json.load(open(a.pilot_summary))["provenance"]
    tdirs, vdirs = pp["runs"].split(","), gp["poseset_validation"].split(",")
    published = {records_path(d, "test"): pp["records_sha256"][d] for d in tdirs}
    published.update({records_path(d, "validation"): gp["validation_records_sha256"][d] for d in vdirs})
    bad = sorted(p for p, h in published.items() if _sha(p) != h)
    if bad:
        raise SystemExit(f"{len(bad)} pose-set record files are not the ones their summaries read (the first {bad[0]})")
    cfgs = [json.load(open(os.path.join(d, "run_config.json"))) for d in tdirs + vdirs]
    for k in ("bundle", "seed", "stage1_records_sha256"):
        if len({_js(c.get(k)) for c in cfgs}) > 1:
            raise SystemExit(f"the pose-set runs differ in {k}")
    passes = sr.load_pass(a.pass_, REPLAN_HOLD)
    units = [u for u in sorted(os.path.join(a.pass_, x) for x in os.listdir(a.pass_) if x.startswith("unit_"))
             if cs._unit_config(u).get("source") == "poseset"]
    errs = sr.replayed_files_errors({u: cs._unit_config(u)["records_sha256"] for u in units}, published)
    if errs:
        raise SystemExit(f"the pass: {len(errs)} replayed files are not the pose set's published records ({errs[0]})")
    s2 = json.load(open(a.stage2c_summary))["provenance"]
    lock_path = _ref_path(s2["locked"])
    if _sha(lock_path) != ref_sha(s2["locked"]):
        raise SystemExit(f"{lock_path}: not the lock Stage 2c read")
    k4 = json.load(open(lock_path))["lock"]["R4"]["key"]
    if k4 != list(rp["keys"])[1]:
        raise SystemExit(f"the lock's R4 key {k4} is not the readings' {rp['keys']}")
    c2dirs = s2["runs"].split(",")
    bad = [d for d in c2dirs if _sha(records_path(d, "test")) != s2["records_sha256"][f"{d} test"]]
    if bad:
        raise SystemExit(f"Stage 2c test records are not the ones its summary read (the first {bad[0]})")
    s2c, _ = load_stage2(c2dirs, ("test",))
    r4 = {(r["object"], int(r["idx"])): (None if r["dropped"] else bool(episode(r["decisions"], r["sims"], k4)["success"])) for r in s2c}
    recs = [r for r in _load_lines([records_path(d, "test") for d in tdirs] + [records_path(d, "validation") for d in vdirs])
            if r["object"] in HAMMERS[r["side"]]]
    entries = [case_entry(r, passes.get(("poseset", r["side"], r["object"], int(r["idx"]))), r4.get((r["object"], int(r["idx"])))
                          if r["side"] == "test" else None) for r in recs]
    entries.sort(key=lambda e: (e["side"] != "test", e["object"], e["idx"]))
    me = _revision()
    out = {"cases": entries, "bundle": cfgs[0]["bundle"], "seed": cfgs[0]["seed"], "stage1_records_sha256": cfgs[0]["stage1_records_sha256"],
           "replan_hold": REPLAN_HOLD, "pass": a.pass_, "r4_key": k4,
           "provenance": {"code_revision": me["code_revision"], "dirty": me["dirty"],
                          "readings_summary": f"{a.readings_summary} (sha256 {_sha(a.readings_summary)})",
                          "poseset_summary": f"{a.poseset_summary} (sha256 {_sha(a.poseset_summary)})",
                          "pilot_summary": f"{a.pilot_summary} (sha256 {_sha(a.pilot_summary)})",
                          "stage2c_summary": f"{a.stage2c_summary} (sha256 {_sha(a.stage2c_summary)})",
                          "locked": f"{lock_path} (sha256 {_sha(lock_path)})", "pass_units": len(units),
                          "pose_set_files": len(published)}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    n = defaultdict(int)
    for e in entries:
        n[f"{e['side']} cases"] += 1
        n[f"{e['side']} skipped"] += bool(e["skipped"])
        n[f"{e['side']} with a G0 reference"] += e["ref"] is not None
        n[f"{e['side']} with an R4 episode"] += e["r4"] is not None
    print(json.dumps({"out": a.out, "sha256": _sha(a.out), "counts": dict(n)}, indent=1), flush=True)
    return a.out


# ---------------------------------------------------------------- the run

def case_line(rec: dict[str, Any]) -> str:
    """A case's log line: counts, G0 and G1 states and stops, no label (no early look)."""
    n = sum("seconds" in r for r in rec["rows"])
    bad = sum(bool(r.get("error")) for r in rec["rows"])
    g0 = None if rec.get("g0") is None else ("G0 same" if not rec["g0"] else "G0 differs")
    rp = rec.get("replay")
    return (f"{rec['object']} {rec['idx']}: {n} rows simulated, {float(rec.get('seconds') or 0.0):.1f} s" + (f", {bad} row errors" if bad else "")
            + (f", {g0}" if g0 else "") + (f", replay {rp['rows']} rows, {rp['label_mismatches']} label mismatches" if rp else "")
            + (f", skipped ({rec['skipped']})" if rec["skipped"] else "") + (f", error {rec['error']}" if rec["error"] else "")
            + (f", STOP {rec['stop']}" if rec["stop"] else ""))


def metrics(done: list[dict[str, Any]]) -> dict[str, Any]:
    """A run's counts and times, no label: cases, skips, case errors, stops, rows simulated and infeasible, row errors
    and artifacts, G0's comparisons and mismatches, G1's rows and mismatches (label, other)."""
    rows = [r for rec in done for r in rec["rows"] if "seconds" in r]
    rep = [rec["replay"] for rec in done if rec.get("replay")]
    g0 = [rec["g0"] for rec in done if rec.get("g0") is not None]
    secs = [float(rec.get("seconds") or 0.0) for rec in done]
    return {"cases": len(done), "skipped": sum(bool(r["skipped"]) for r in done), "case_errors": sum(bool(r["error"]) for r in done),
            "stops": sum(bool(r["stop"]) for r in done), "rows_simulated": len(rows),
            "rows_infeasible": sum(not r["feasible"] for rec in done for r in rec["rows"]),
            "row_errors": sum(bool(r.get("error")) for r in rows), "row_artifacts": sum(bool(r.get("artifact")) for r in rows),
            "g0_compared": len(g0), "g0_mismatches": sum(bool(x) for x in g0), "replay_rows": sum(x["rows"] for x in rep),
            "replay_label_mismatches": sum(x["label_mismatches"] for x in rep),
            "replay_other_mismatches": sum(len(x["mismatches"]) - x["label_mismatches"] for x in rep),
            "row_seconds": round(sum(float(r["seconds"]) for r in rows), 1), "case_seconds": round(sum(secs), 1),
            "case_seconds_max": max(secs, default=0.0)}


MON_ROW_KEYS = ("id", "pinch", "level", "feasible", "qpos_sha", "trace_sha", "artifact", "error", "seconds", "monitor", "close")
MON_CASE_KEYS = ("object", "idx", "side", "skipped", "stop", "error", "g0", "seconds")


def monitor_record(rec: dict[str, Any]) -> dict[str, Any]:
    """A close-monitor pass's record (the coordinator's acceptance of 9 Nov: no label in the pass): the case's keys
    (MON_CASE_KEYS) and per row the hashes the summary joins on, the artifact flag, the error, the seconds, the contact
    monitor and the close monitor (MON_ROW_KEYS); no label, outcome, squeeze or re-plan field."""
    out = {k: rec.get(k) for k in MON_CASE_KEYS}
    out["rows"] = [{k: r.get(k) for k in MON_ROW_KEYS} for r in rec["rows"]]
    return out


def force_object(job: tuple[str, list[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]]) -> dict[str, Any]:
    """One object's cases on its fold's bundle (curve_poseset.poseset_object's sensor and world, no probe models); a
    case_line per case on stderr; a stop ends the object."""
    from robots.mujoco_lab import calibrate as cal
    from robots.mujoco_lab.prior import fragility_world

    o, items, s = job
    prims = cal._primitives(s["bundle"], None, None)
    if getattr(prims, "reset_s", None) is not None:
        raise RuntimeError("this run scores the absorbing attempt only (this bundle restores the object after damage)")
    probes, prims.probe_models = getattr(prims, "probe_models", None), {}
    prims.force_noise = cal.job_force_noise(None, prims.manifest)
    prims.env.force_noise = prims.force_noise
    world = fragility_world(prims.manifest)
    out: dict[str, Any] = {"records": [], "stop": None}
    try:
        for rec1, entry in items:
            t0 = time.time()
            rec = force_case(prims, rec1, int(s["seed"]), world, entry, replay=bool(s.get("replay")),
                             close_monitor=bool(s.get("close_monitor")))
            rec["seconds"] = round(time.time() - t0, 2)
            out["records"].append(rec)
            print(case_line(rec), file=sys.stderr, flush=True)
            if rec["stop"]:
                out["stop"] = f"{o} {rec1['idx']}: {rec['stop']}"
                break
    finally:
        prims.probe_models = probes
    return out


def _keys(text: str | None) -> list[tuple[str, int]] | None:
    if not text:
        return None
    return [(x.rsplit(":", 1)[0], int(x.rsplit(":", 1)[1])) for x in text.split(",") if x.strip()]


def run(a: argparse.Namespace) -> str:
    """The rows' simulations: the sim gate first (sim_problem: the contacts sim, no dirty file), then the cases file's
    Stage 1 records (their sha256), the cases (--cases object:idx, else --objects, else all), by object (force_object);
    records_validation.jsonl and records_test.jsonl, run_config.json (the cases file's sha256, the flag, the sim) and
    run_metrics.json (metrics; wandb). With --close-monitor (the separate monitor pass): the rows run under the close
    monitor and the records are monitor_record's (no label). Refuses an existing records file. Exit 3 at a stop. Returns
    --out."""
    import multiprocessing as mp

    from robots.mujoco_lab.curve import SIDES, _peak_rss_mb, _sha, load_stage1, records_path
    from robots.mujoco_lab.estimation_check import _revision
    from robots.mujoco_lab.tracking import log_run

    sim_dir = os.environ.get("RPENT_MUJOCO_SIM_DIR")
    why = sim_problem(sim_dir)
    if why:
        raise SystemExit(f"sim gate: {why}")
    if any(os.path.exists(records_path(a.out, side)) for side in SIDES):
        raise SystemExit(f"{a.out}: a records file exists: pick a new --out")
    C = json.load(open(a.cases_file))
    for d, h in C["stage1_records_sha256"].items():
        if _sha(os.path.join(d, "records.jsonl")) != h:
            raise SystemExit(f"{d}: the Stage 1 records changed since the cases file")
    entries = {(e["object"], int(e["idx"])): e for e in C["cases"]}
    keys = _keys(a.cases) or [k for k in entries if not a.objects or k[0] in a.objects.split(",")]
    missing = [k for k in keys if k not in entries]
    if missing:
        raise SystemExit(f"cases not in the cases file: {missing}")
    recs1, _ = load_stage1(list(C["stage1_records_sha256"]))
    by1 = {(r["object"], int(r["idx"])): r for r in recs1}
    os.makedirs(a.out, exist_ok=True)
    config = {**vars(a), **_revision(), "cases_sha256": _sha(a.cases_file), "bundle": C["bundle"], "seed": C["seed"],
              "stage1_records_sha256": C["stage1_records_sha256"], "replan_hold": REPLAN_HOLD, "sim_dir": sim_dir, "sim": CONTACTS_SIM,
              "pinches": PINCHES, "levels": LEVELS, "fixed_n": FIXED_N, "keys": [list(k) for k in keys],
              "close_monitor": bool(getattr(a, "close_monitor", False))}
    with open(os.path.join(a.out, "run_config.json"), "w") as f:
        json.dump(config, f, indent=1)
    by: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for k in keys:
        by[k[0]].append((by1[k], entries[k]))
    jobs = [(o, items, {"bundle": os.path.join(C["bundle"], "folds", f"fold_{items[0][0]['fold']}"), "seed": C["seed"],
                        "replay": bool(a.replay), "close_monitor": config["close_monitor"]}) for o, items in by.items()]
    t0, stop = time.time(), None
    outs = {side: open(records_path(a.out, side), "w") for side in SIDES}
    try:
        pool = None if a.workers == 1 else mp.get_context("spawn").Pool(a.workers)
        results = map(force_object, jobs) if pool is None else pool.imap_unordered(force_object, jobs)
        try:
            for res in results:
                for r in res["records"]:
                    outs[r["side"]].write(json.dumps(monitor_record(r) if config["close_monitor"] else r) + "\n")
                for f in outs.values():
                    f.flush()
                if res["stop"]:
                    stop = res["stop"]
                    break
        finally:
            if pool is not None:
                (pool.terminate if stop else pool.close)()
                pool.join()
    finally:
        for f in outs.values():
            f.close()
    done = [json.loads(line) for side in SIDES for line in open(records_path(a.out, side))]
    m = dict(metrics(done), of_cases=len(keys), stop=stop, wall_seconds=round(time.time() - t0, 1), peak_rss_mb=_peak_rss_mb())
    m["wandb"] = log_run("curve-hammer-force-monitor" if config["close_monitor"] else "curve-hammer-force", config, m)
    with open(os.path.join(a.out, "run_metrics.json"), "w") as f:
        json.dump(m, f, indent=1)
    print(f"done: {json.dumps(m)}", flush=True)
    if stop:
        raise SystemExit(3)
    return a.out


# ---------------------------------------------------------------- the smoke report

def _median(v: list[float]) -> float | None:
    return float(np.median(v)) if v else None


def artifact_counts(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """The artifact rows by level and cause (a row can have several: a pad penetration over adapter.DEEP_M, reversed
    contacts in adapter.REV_STEPS squeeze substeps or more, a reset not at rest, a jam), the cases with an artifact row
    and the deepest penetration over the simulated rows (mm). Counts only, no label."""
    from robots.mujoco_lab.adapter import DEEP_M, REV_STEPS

    by = {lv: {"rows": 0, "deep": 0, "reversed": 0, "not_at_rest": 0, "jam": 0} for lv in LEVELS}
    cases = 0
    for rec in recs:
        hit = False
        for r in rec["rows"]:
            if "seconds" not in r or not r.get("artifact"):
                continue
            m, x = r.get("monitor") or {}, by[r["level"]]
            x["rows"] += 1
            x["deep"] += float(m.get("deepest_mm") or 0.0) > 1000 * DEEP_M
            x["reversed"] += int(m.get("rev_steps") or 0) >= REV_STEPS
            x["not_at_rest"] += m.get("at_rest") is False
            x["jam"] += bool(m.get("jam_artifact"))
            hit = True
        cases += hit
    deep = [float((r.get("monitor") or {}).get("deepest_mm") or 0.0) for rec in recs for r in rec["rows"] if "seconds" in r]
    return {"by_level": by, "cases": cases, "deepest_mm_max": max(deep) if deep else None}


def smoke_report(recs: list[dict[str, Any]], full_cases: int) -> dict[str, Any]:
    """The smoke report, no label: metrics; G0 (compared, mismatches); G1 (cases, rows, label mismatches, other
    mismatches by field); per level the median of the rows' hold squeeze (no count of the rows with a hold reading: it
    follows the lifted count) and of their close squeeze (with its rows); the stops; the first pass's seconds per case
    (the replay's left out) and the projection to ``full_cases`` cases."""
    from collections import Counter

    m = metrics(recs)
    rep = [r["replay"] for r in recs if r.get("replay")]
    other = Counter(f for x in rep for mm in x["mismatches"] if "label" not in mm["fields"] for f in mm["fields"])
    sq = {}
    for lv in LEVELS:
        rows = [r for rec in recs for r in rec["rows"] if "seconds" in r and r["level"] == lv and r.get("squeeze")]
        close = [r["squeeze"]["close"] for r in rows if r["squeeze"]["close"] is not None]
        sq[lv] = {"hold_median": _median([r["squeeze"]["hold"] for r in rows if r["squeeze"]["hold"] is not None]),
                  "close_median": _median(close), "close_rows": len(close)}
    secs = [float(r.get("seconds") or 0.0) - float(r.get("replay_seconds") or 0.0) for r in recs if not r["skipped"]]
    sm = dict(m, g0={"compared": m["g0_compared"], "mismatches": m["g0_mismatches"]},
              g1={"cases": len(rep), "rows": sum(x["rows"] for x in rep), "label_mismatches": sum(x["label_mismatches"] for x in rep),
                  "other_mismatches": sum(len(x["mismatches"]) - x["label_mismatches"] for x in rep), "other_fields": dict(other)},
              squeeze=sq, artifacts=artifact_counts(recs), stop_texts=[r["stop"] for r in recs if r["stop"]],
              seconds_per_case=float(np.mean(secs)) if secs else None,
              seconds_per_case_max=max(secs) if secs else None, full_cases=full_cases)
    sm["projection_cpu_h"] = sm["seconds_per_case"] * full_cases / 3600.0 if secs else None
    sm["projection_cpu_h_at_max"] = sm["seconds_per_case_max"] * full_cases / 3600.0 if secs else None
    return sm


def _f(x: float | None, nd: int = 1) -> str:
    return "–" if x is None else f"{x:.{nd}f}"


def smoke_markdown(sm: dict[str, Any]) -> str:
    g0, g1 = sm["g0"], sm["g1"]
    L = ["# Hammer squeeze x CoM pinch: the smoke (counts only, no label)", "",
         f"- Cases {sm['cases']}: skipped {sm['skipped']}, case errors {sm['case_errors']}, stops {sm['stops']}; rows simulated "
         f"{sm['rows_simulated']}, infeasible {sm['rows_infeasible']}, row errors {sm['row_errors']}, row artifacts {sm['row_artifacts']}.",
         f"- G0 (the control row against the corrected pose-set record): {g0['compared']} cases compared, {g0['mismatches']} mismatches.",
         f"- G1 (a second reset seed): {g1['cases']} cases, {g1['rows']} rows, {g1['label_mismatches']} label mismatches, "
         f"{g1['other_mismatches']} rows with another field that differs {g1['other_fields']}.",
         f"- First-pass seconds per case: mean {_f(sm['seconds_per_case'])}, max {_f(sm['seconds_per_case_max'])}; projection for "
         f"{sm['full_cases']} cases {_f(sm['projection_cpu_h'], 2)} CPU-h (at the slowest case {_f(sm['projection_cpu_h_at_max'], 2)}).", "",
         "| level | median squeeze per pad in the hold (N) | median squeeze per pad, late close (N) | rows (close) |", "|---|---|---|---|"]
    for lv, x in sm["squeeze"].items():
        L.append(f"| {lv} | {_f(x['hold_median'])} | {_f(x['close_median'])} | {x['close_rows']} |")
    A = sm["artifacts"]
    L += ["", "The hold median leaves out its number of rows (rows with both pads on the object in the hold): that number follows "
          "the lifted count.", "", f"Artifact rows by level and cause (a row can have several); cases with an artifact row: {A['cases']}; "
          f"the deepest pad penetration over the simulated rows: {_f(A['deepest_mm_max'], 2)} mm.", "",
          "| level | artifact rows | penetration over 3 mm | reversed contacts | reset not at rest | jam |", "|---|---|---|---|---|---|"]
    L += [f"| {lv} | {x['rows']} | {x['deep']} | {x['reversed']} | {x['not_at_rest']} | {x['jam']} |" for lv, x in A["by_level"].items()]
    L += [""] + [f"- STOP: {s}" for s in sm["stop_texts"]]
    return "\n".join(L) + "\n"


def smoke_cmd(a: argparse.Namespace) -> dict[str, Any]:
    """The smoke report on finished smoke units (each with EXIT 0 and the contacts sim in its provenance):
    smoke_summary.{json,md} in --out (refuses an existing one), wandb. Prints the report, no label."""
    from robots.mujoco_lab.curve import SIDES, _sha, records_path
    from robots.mujoco_lab.curve_sixdof_rerun import read_provenance
    from robots.mujoco_lab.tracking import log_run

    jpath = os.path.join(a.out, "smoke_summary.json")
    if os.path.exists(jpath):
        raise SystemExit(f"{jpath} exists: pick a new --out")
    dirs = a.runs.split(",")
    for d in dirs:
        ex = os.path.join(d, "run.exit")
        if not os.path.exists(ex) or open(ex).read().strip() != "EXIT 0":
            raise SystemExit(f"{d}: no EXIT 0")
        p = read_provenance(d)
        if p["sim"] != CONTACTS_SIM or p["sim_dirty"] != 0 or p["code_dirty"] != 0:
            raise SystemExit(f"{d}: provenance {p}")
        if json.load(open(os.path.join(d, "run_config.json")))["cases_sha256"] != _sha(a.cases_file):
            raise SystemExit(f"{d}: ran on another cases file")
    recs = [json.loads(line) for d in dirs for side in SIDES for line in open(records_path(d, side)) if line.strip()]
    full = sum(1 for e in json.load(open(a.cases_file))["cases"] if not e["skipped"])
    sm = smoke_report(recs, full)
    sm["runs"] = a.runs
    os.makedirs(a.out, exist_ok=True)
    sm["wandb"] = log_run("curve-hammer-force-smoke", {"runs": a.runs, "cases_file": a.cases_file}, sm)
    with open(jpath, "w") as f:
        json.dump(sm, f, indent=1)
    with open(os.path.join(a.out, "smoke_summary.md"), "w") as f:
        f.write(smoke_markdown(sm))
    print(smoke_markdown(sm), flush=True)
    return sm


def main(argv: list[str] | None = None) -> Any:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cases", help="the cases file from the published pose set, the G1′ pass and Stage 2c")
    c.add_argument("--readings-summary", required=True, help="the G1′ readings summary (readings_summary.json)")
    c.add_argument("--poseset-summary", required=True)
    c.add_argument("--pilot-summary", required=True)
    c.add_argument("--pass", dest="pass_", required=True, help="the G1′ pass with this run's re-plan flag (g1pr when on)")
    c.add_argument("--stage2c-summary", required=True)
    c.add_argument("--out", required=True, help="the cases file to write")
    r = sub.add_parser("run", help="the rows' simulations")
    r.add_argument("--cases-file", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--cases", default=None, help="comma-separated object:idx cases")
    r.add_argument("--objects", default=None, help="comma-separated objects (all their cases)")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--replay", action="store_true", help="G1: every simulated row again at the second reset seed")
    r.add_argument("--close-monitor", action="store_true", help="the separate monitor pass: the close monitor, records without a label")
    s = sub.add_parser("smoke-report", help="the smoke's counts and gates, no label")
    s.add_argument("--runs", required=True, help="comma-separated smoke run dirs")
    s.add_argument("--cases-file", required=True)
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    return {"cases": cases_cmd, "run": run, "smoke-report": smoke_cmd}[a.cmd](a)


if __name__ == "__main__":
    main(sys.argv[1:])
