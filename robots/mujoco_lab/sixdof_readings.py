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

"""The readings of the G1′ write-up (design 2026-11-08-sixdof-rerun-design.md, amendment 8, "Readings of the G1′
write-up", fixed before the second run): (i) the pose set's ceilings (the 10-row and the 3-row unions; all, upright,
lying, hammers) and the GraspGen pilot's ceilings recomputed from the replayed labels of the same rows (each stored row
with its replay's values of the replay-gate fields), paired by case with the published object bootstrap, beside the
published numbers and the palm-contact audit's sensitivity lines; (ii) per object, the placed and the damage counts of
the replayed rows before and after, with the hammers on their own line; (iii) the same after the pass with the re-plan
at the hold, with its changed rows by cause. Readings of existing replay rows, not new experiments."""

from __future__ import annotations

import glob
import json
import os
from collections import Counter
from typing import Any

from rpent.skills.pick_place.protocol import IRREVERSIBLE

from robots.mujoco_lab import curve_graspgen as gg
from robots.mujoco_lab import curve_poseset as cp
from robots.mujoco_lab import curve_sixdof as cs

PLACED = "placed_stable"


def correct_rows(rows: list[dict[str, Any]], reps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The stored rows with each simulated row's replay-gate fields (curve_poseset.SAME) set to its replay's values (the
    replay's diffs, {field: [stored, new]}); a row that never ran is kept. Refuses a simulated row without a replay, a
    replay without its row, a row replayed twice and a replay of another stored label (another case's rows)."""
    by: dict[str, dict[str, Any]] = {}
    for x in reps:
        if x["id"] in by:
            raise ValueError(f"{x['id']}: replayed twice")
        by[x["id"]] = x
    extra = sorted(set(by) - {r["id"] for r in rows if "seconds" in r})
    if extra:
        raise ValueError(f"replays without a simulated row: {extra}")
    out = []
    for r in rows:
        if "seconds" not in r:
            out.append(r)
            continue
        x = by.get(r["id"])
        if x is None:
            raise ValueError(f"{r['id']}: a simulated row without a replay")
        if x["stored_label"] != r.get("label"):
            raise ValueError(f"{r['id']}: the replay's stored label {x['stored_label']} is not the row's {r.get('label')}")
        out.append(dict(r, **{k: v[1] for k, v in x["diffs"].items()}))
    return out


def _simulated(rec: dict[str, Any]) -> bool:
    return any("seconds" in r for r in rec.get("rows") or [])


def correct_poseset(rec: dict[str, Any], rep: dict[str, Any] | None) -> dict[str, Any]:
    """A pose-set case (curve_poseset.poseset_case's record) with its replay (a G1′ case): the rows corrected
    (correct_rows), both ceilings again (curve_poseset.ceiling) and the drop by poseset_case's rule (an undetermined
    ceiling drops the case). A skipped case, a case with an error and a case without a simulated row are kept as
    published; a simulated case without a replay is refused."""
    if rec["skipped"] or rec["error"] or not _simulated(rec):
        return rec
    if rep is None:
        raise ValueError(f"{rec['object']} {rec['idx']}: no replay of a simulated case")
    rows = correct_rows(rec["rows"], rep["rows"])
    c10, c3 = cp.ceiling(rows), cp.ceiling(rows, current_only=True)
    return dict(rec, rows=rows, ceiling10=c10, ceiling3=c3, dropped=not (c10["determined"] and c3["determined"]))


def correct_pilot(rec: dict[str, Any], rep: dict[str, Any] | None) -> dict[str, Any]:
    """A pilot case (curve_graspgen.graspgen_case's record) with its replay, as correct_poseset: both rules' ceilings
    again (curve_graspgen.ceiling over the kept rows) and the drop by graspgen_case's rule (an undetermined design-rule
    ceiling); the gate row is corrected too."""
    if rec["skipped"] or rec["error"] or not _simulated(rec):
        return rec
    if rep is None:
        raise ValueError(f"{rec['object']} {rec['idx']}: no replay of a simulated case")
    rows = correct_rows(rec["rows"], rep["rows"])
    c, cb = gg.ceiling(rows, "kept"), gg.ceiling(rows, "kept_b")
    return dict(rec, rows=rows, ceiling=c, ceiling_b=cb, dropped=not c["determined"])


def object_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per object (in first-seen order), over the hammers (an object named a hammer) and over all: the rows, and per
    pass (each row's ``labels``: the stored one, then each pass's) the rows placed and the rows with damage (an
    irreversible label)."""
    n = len(rows[0]["labels"]) if rows else 0

    def tally(x: list[dict[str, Any]]) -> dict[str, Any]:
        return {"rows": len(x), "placed": [sum(r["labels"][k] == PLACED for r in x) for k in range(n)],
                "damage": [sum(r["labels"][k] in IRREVERSIBLE for r in x) for k in range(n)]}

    objs = list(dict.fromkeys(r["object"] for r in rows))
    return {"by_object": {o: tally([r for r in rows if r["object"] == o]) for o in objs},
            "hammers": tally([r for r in rows if "hammer" in r["object"]]), "all": tally(rows)}


def by_cause(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The changed rows (``changed``) by ``cause`` (curve_sixdof.CAUSES, in order): the rows, the placed and the damage
    counts before and after (the first and the last of ``labels``) and the label transitions. Refuses a changed row
    without a cause."""
    out: dict[str, dict[str, Any]] = {k: {"rows": 0, "placed": [0, 0], "damage": [0, 0], "transitions": Counter()} for k in cs.CAUSES}
    for r in rows:
        if not r["changed"]:
            continue
        if r["cause"] not in out:
            raise ValueError(f"{r['object']} {r.get('idx')} {r.get('id')}: a changed row with no cause")
        x, (a, b) = out[r["cause"]], (r["labels"][0], r["labels"][-1])
        x["rows"] += 1
        for k, lab in enumerate((a, b)):
            x["placed"][k] += lab == PLACED
            x["damage"][k] += lab in IRREVERSIBLE
        if a != b:
            x["transitions"][f"{a} → {b}"] += 1
    return {k: dict(v, transitions=dict(v["transitions"])) for k, v in out.items()}


def hammer_by_object(vs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per hammer object (in first-seen order) of the views: the cases, the cases a pose placed (curve_poseset.
    hammer_class's first class) and the cases by class (the classes that occur)."""
    out: dict[str, dict[str, Any]] = {}
    for v in vs:
        if not v["hammer"]:
            continue
        c = cp.hammer_class(v["rows"])
        x = out.setdefault(v["object"], {"cases": 0, "placed": 0, "classes": Counter()})
        x["cases"] += 1
        x["placed"] += c == cp.HAMMER_CLASSES[0]
        x["classes"][c] += 1
    return {o: dict(x, classes=dict(x["classes"])) for o, x in out.items()}


def row_table(recs: list[dict[str, Any]], source: str, passes: list[dict[tuple[str, str, str, int], dict[str, Any]]]) -> list[dict[str, Any]]:
    """One line per simulated row of ``source``'s records (a pilot's gate row left out: it is the pose set's row): its
    case and id, ``labels`` (the stored label, then each pass's), ``changed`` and ``cause`` in the last pass
    (curve_sixdof.row_cause) and ``first``, each pass's first box contact. Refuses a simulated row a pass lacks."""
    out = []
    for rec in recs:
        rows = [r for r in rec["rows"] if "seconds" in r and not r.get("gate")]
        if not rows:
            continue
        k = (source, rec["side"], rec["object"], int(rec["idx"]))
        reps = [{x["id"]: x for x in p[k]["rows"]} if k in p else {} for p in passes]
        for r in rows:
            xs = [rp.get(r["id"]) for rp in reps]
            if any(x is None for x in xs):
                raise ValueError(f"{k} {r['id']}: not replayed in every pass")
            out.append({"source": source, "side": rec["side"], "object": rec["object"], "idx": int(rec["idx"]), "id": r["id"],
                        "labels": [r.get("label")] + [x["label"] for x in xs], "changed": not xs[-1]["same"],
                        "cause": cs.row_cause(xs[-1]), "first": [(x.get("box_contacts") or {}).get("first") for x in xs]})
    return out


def _contact_name(f: dict[str, Any] | None) -> str:
    return "none" if not f else f"{f['phase']} {f['box']}/{f['partner']}"


def hammer_label_changes(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The hammer rows (row_table's, two passes) whose label differs from the stored one, per pass: the rows, their
    first box contact in that pass (phase box/partner, "none") and the transitions; for the re-plan pass also the
    causes (curve_sixdof.row_cause). The recorder's first contact is the only cause these lines name."""
    out: dict[str, dict[str, Any]] = {}
    for k, name in ((1, "contacts"), (2, "replan")):
        x = [r for r in rows if "hammer" in r["object"] and r["labels"][k] != r["labels"][0]]
        out[name] = {"rows": len(x), "first_contact": dict(Counter(_contact_name(r["first"][k - 1]) for r in x)),
                     "transitions": dict(Counter(f"{r['labels'][0]} → {r['labels'][k]}" for r in x))}
        if name == "replan":
            out[name]["by_cause"] = dict(Counter(r["cause"] for r in x))
    return out


def load_pass(queue: str, replan: bool) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    """A G1′ pass's replays by case ((source, side, object, idx) → its G1′ case), from its units (``queue``/unit_*).
    Refuses a unit without EXIT 0, a unit without box contacts or with another re-plan flag than ``replan``, a report
    (``queue``/report) that did not pass, a case replayed twice and a case with a stop or an error."""
    units = sorted(glob.glob(os.path.join(queue, "unit_*")), key=lambda u: int(u.rsplit("_", 1)[1]))
    done, bad_exit, _ = cs._load_units(units)
    if bad_exit:
        raise SystemExit(f"units without EXIT 0: {bad_exit}")
    for u in units:
        cfg = cs._unit_config(u)
        if not cfg.get("box_contacts"):
            raise SystemExit(f"{u}: replayed without box contacts")
        if bool(cfg.get("replan_hold")) != replan:
            raise SystemExit(f"{u}: replan_hold {cfg.get('replan_hold')!r}, the pass wants {replan}")
    rp = os.path.join(queue, "report")
    ok = os.path.exists(os.path.join(rp, "run.exit")) and open(os.path.join(rp, "run.exit")).read().strip() == "EXIT 0"
    if not (ok and json.load(open(os.path.join(rp, "g1p_summary.json"))).get("passed")):
        raise SystemExit(f"{queue}: the G1′ report did not pass")
    out: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for c in done:
        k = (c["source"], c["side"], c["object"], int(c["idx"]))
        if k in out:
            raise SystemExit(f"{k}: replayed twice")
        if c["stop"] or c["error"]:
            raise SystemExit(f"{k}: a case with a stop ({c['stop']!r}) or an error ({c['error']!r})")
        out[k] = c
    return out


# ------------------------------------------------------------------ the readings over the published records

VIEWS = ("all", "upright", "lying", "hammers")
PASSES = ("published", "contacts", "replan")
PASS_NAMES = {"published": "published", "contacts": "contacts on", "replan": "contacts on + re-plan"}
THREE = ("success", "irr", "give_up")


def _by_key(recs: list[dict[str, Any]], what: str) -> dict[tuple[str, int], dict[str, Any]]:
    by = {cp._key(r): r for r in recs}
    if len(by) != len(recs):
        raise SystemExit(f"a case appears twice in the {what} records")
    return by


def pose_views(precs: list[dict[str, Any]], s2: list[dict[str, Any]], orecs: list[dict[str, Any]], k0: str,
               k4: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """The pose set's paired views as its summary joins them (per Stage 2 test case its pose-set and oracle records;
    curve_poseset.paired_set3 and views) and the cases dropped."""
    byp, byo = _by_key(precs, "pose-set"), _by_key(orecs, "oracle")
    joined = [{"pose": byp[cp._key(r)], "stage2": r, "stage1": None, "oracle": byo[cp._key(r)]} for r in s2]
    paired, n = cp.paired_set3(joined)
    return cp.views(paired, k0, k4), n


def pose_reading(vs: list[dict[str, Any]], n_dropped: dict[str, int]) -> dict[str, Any]:
    """Reading (i) on the pose set's views: per view (VIEWS) the 10-row − 3-row ceiling paired by case
    (curve_oracle.paired_outcomes: both rates and the object bootstrap), the hammer cases by class (curve_poseset.
    hammer_line) and by object (hammer_by_object), the paired cases and the cases dropped."""
    from robots.mujoco_lab.curve_oracle import paired_outcomes

    sel = {"all": vs, "upright": [v for v in vs if v["upright"]], "lying": [v for v in vs if not v["upright"]],
           "hammers": [v for v in vs if v["hammer"]]}
    return {"views": {k: paired_outcomes([(v["object"], v["c10"], v["c3"]) for v in x]) for k, x in sel.items()},
            "hammer_classes": cp.hammer_line(vs)["classes"], "hammer_by_object": hammer_by_object(vs), "paired": len(vs),
            "paired_dropped": n_dropped}


def pilot_views(grecs: list[dict[str, Any]], precs: list[dict[str, Any]], orecs: list[dict[str, Any]], s2: list[dict[str, Any]],
                k0: str, k4: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """The pilot's paired views as its summary joins them (per pilot case its pose-set, oracle and Stage 2 records on
    the same side; curve_graspgen.pilot_paired and pilot_views) and the cases dropped."""
    arms = {"pose-set": _by_key(precs, "pose-set"), "oracle": _by_key(orecs, "oracle"), "Stage 2": _by_key(s2, "Stage 2")}
    joined = []
    for r in grecs:
        k = cp._key(r)
        for name, by in arms.items():
            if k not in by or by[k]["side"] != r["side"]:
                raise SystemExit(f"{k[0]} {k[1]}: no {name} record of this case on the {r['side']} side")
        joined.append({"g": r, "pose": arms["pose-set"][k], "oracle": arms["oracle"][k], "stage2": arms["Stage 2"][k]})
    paired, n = gg.pilot_paired(joined)
    return gg.pilot_views(paired, k0, k4), n


def pilot_reading(vs: list[dict[str, Any]], n_dropped: dict[str, int]) -> dict[str, Any]:
    """Reading (i) on the pilot's test views (curve_graspgen.pilot_block: each arm's rates, GraspGen under each rule −
    the 10-pinch ceiling with the object bootstrap and the 2-point rule) and its hammer line's classes."""
    vt = [v for v in vs if v["side"] == "test"]
    b, h = gg.pilot_block(vt), gg.hammer_line(vt)
    return {"test": {k: b[k] for k in ("cases", "objects", "rates", "gg_vs_c10", "ggb_vs_c10", "rule", "rule_b")},
            "hammer_classes": h["classes"], "hammer_c10_classes": h["c10_classes"],
            "paired": {"test": len(vt), "validation": len(vs) - len(vt)}, "paired_dropped": n_dropped}


def gate(psum: dict[str, Any], gsum: dict[str, Any], pub: dict[str, Any]) -> dict[str, list[str]]:
    """The published numbers against those recomputed from the stored records: the pose set's 1′ (all, upright, lying)
    and hammer classes, the pilot's two GraspGen − 10-pinch differences and hammer classes; each entry lists what is
    not equal."""
    from robots.mujoco_lab.palm_audit import differs

    pr, gr = psum["readings"], gsum["readings"]
    g = {f"poseset_{k}": differs(pub["poseset"]["views"][k], pr[k]["1p_pose_term"], THREE) for k in ("all", "upright", "lying")}
    g["poseset_hammer_classes"] = [] if pub["poseset"]["hammer_classes"] == pr["4_hammer"]["classes"] else ["classes"]
    for k in ("gg_vs_c10", "ggb_vs_c10"):
        g[f"pilot_{k}"] = differs(pub["pilot"]["test"][k], gr["2_test"][k], THREE)
    h = pub["pilot"]
    g["pilot_hammer_classes"] = [] if (h["hammer_classes"], h["hammer_c10_classes"]) == (gr["3_hammer"]["classes"], gr["3_hammer"]["c10_classes"]) else ["classes"]
    return g


def _ref_path(ref: str) -> str:
    return ref.split(" (sha256 ")[0]


def run(a: Any) -> dict[str, Any]:
    """Readings (i) to (iii) (readings_summary.{json,md} in --out, refusing an existing summary; wandb). The records are
    the published pose-set and pilot summaries' (each file's sha256 must be the one its summary read, Stage 2's and the
    oracle arm's through their summaries); the passes (load_pass) must have replayed these files; the gate recomputes
    the published numbers from the stored records first (a difference writes readings_stop.json and exits 3)."""
    from robots.mujoco_lab.curve import BOOT_DRAWS, BOOT_SEED, SIDES, _sha, load_stage2, records_path
    from robots.mujoco_lab.curve_poseset import _load_lines
    from robots.mujoco_lab.estimation_check import _revision
    from robots.mujoco_lab.palm_audit import PRIMARY, ref_sha
    from robots.mujoco_lab.tracking import log_run

    os.makedirs(a.out, exist_ok=True)
    jpath, spath = os.path.join(a.out, "readings_summary.json"), os.path.join(a.out, "readings_stop.json")
    if os.path.exists(jpath) or os.path.exists(spath):
        raise SystemExit(f"{a.out}: a summary exists: pick a new --out")
    psum, gsum, asum = (json.load(open(x)) for x in (a.poseset_summary, a.pilot_summary, a.audit_summary))
    pp, gp = psum["provenance"], gsum["provenance"]
    if ref_sha(gp["poseset_summary"]) != _sha(a.poseset_summary):
        raise SystemExit("the pilot summary read another pose-set summary")
    if (gp["poseset"], gp["stage2"], gp["oracle"], gp["locked"]) != (pp["runs"], pp["stage2"], pp["oracle"], pp["locked"]):
        raise SystemExit("the pilot and the pose-set summaries read other runs or another lock")
    if ref_sha(asum["provenance"]["poseset_summary"]) != _sha(a.poseset_summary):
        raise SystemExit("the audit summary read another pose-set summary")
    refs = {k: _ref_path(gp[k]) for k in ("locked", "stage2_summary", "oracle_summary")}
    for k, path in refs.items():
        if _sha(path) != ref_sha(gp[k]):
            raise SystemExit(f"{path}: not the {k} the pilot summary read")
    lk = json.load(open(refs["locked"]))["lock"]
    k0, k4 = lk["R0"]["key"], lk["R4"]["key"]
    if [k0, k4] != list(gp["keys"]):
        raise SystemExit(f"the lock's keys {[k0, k4]} are not the pilot summary's {gp['keys']}")
    s2prov = json.load(open(refs["stage2_summary"]))["provenance"]["records_sha256"]
    oprov = json.load(open(refs["oracle_summary"]))["provenance"]["oracle_records_sha256"]
    pdirs, pvdirs, gdirs = pp["runs"].split(","), gp["poseset_validation"].split(","), gp["runs"].split(",")
    s2dirs, odirs, ovdirs = gp["stage2"].split(","), gp["oracle"].split(","), gp["oracle_validation"].split(",")
    want = dict(gp["records_sha256"])                                           # the pilot's records, both sides
    want.update({records_path(d, "test"): pp["records_sha256"][d] for d in pdirs})
    want.update({records_path(d, "validation"): gp["validation_records_sha256"][d] for d in pvdirs + ovdirs})
    want.update({records_path(d, "test"): oprov[d] for d in odirs})
    want.update({records_path(d, side): s2prov[f"{d} {side}"] for d in s2dirs for side in SIDES})
    bad = sorted(p for p, h in want.items() if _sha(p) != h)
    if bad:
        raise SystemExit(f"{len(bad)} record files are not the ones their summaries read (the first {bad[0]})")
    precs = _load_lines([records_path(d, "test") for d in pdirs])
    pvrecs = _load_lines([records_path(d, "validation") for d in pvdirs])
    grecs = _load_lines([records_path(d, side) for d in gdirs for side in SIDES])
    s2t, _ = load_stage2(s2dirs, ("test",))
    s2v, _ = load_stage2(s2dirs, ("validation",))
    orecs, ovrecs = _load_lines([records_path(d, "test") for d in odirs]), _load_lines([records_path(d, "validation") for d in ovdirs])

    queues = {"contacts": a.contacts, "replan": a.replan}
    passes = {"contacts": load_pass(a.contacts, False), "replan": load_pass(a.replan, True)}
    keys = ({("poseset", r["side"], r["object"], int(r["idx"])) for r in precs + pvrecs}
            | {("pilot", r["side"], r["object"], int(r["idx"])) for r in grecs})
    revs: dict[str, list[str]] = {}
    for name, q in queues.items():
        extra = sorted(set(passes[name]) - keys)
        if extra:
            raise SystemExit(f"the {name} pass replayed {len(extra)} cases no record holds (the first {extra[0]})")
        units = sorted(glob.glob(os.path.join(q, "unit_*")))
        for u in units:
            for path, h in cs._unit_config(u)["records_sha256"].items():
                if _sha(path) != h:
                    raise SystemExit(f"{u}: it replayed {path} at another sha256")
        revs[name] = sorted({cs._unit_config(u)["code_revision"] for u in units})

    def readings_of(pt: list[dict[str, Any]], pv: list[dict[str, Any]], g: list[dict[str, Any]]) -> dict[str, Any]:
        vs, n = pose_views(pt, s2t, orecs, k0, k4)
        gv, gn = pilot_views(g, pt + pv, orecs + ovrecs, s2t + s2v, k0, k4)
        return {"poseset": dict(pose_reading(vs, n), pose_dropped={"test": sum(bool(r["dropped"]) for r in pt),
                                                                     "validation": sum(bool(r["dropped"]) for r in pv)}),
                "pilot": dict(pilot_reading(gv, gn), pilot_dropped=sum(bool(r["dropped"]) for r in g))}

    def key(src: str, r: dict[str, Any]) -> tuple[str, str, str, int]:
        return (src, r["side"], r["object"], int(r["idx"]))

    rep: dict[str, Any] = {"published": readings_of(precs, pvrecs, grecs)}
    me = _revision()
    prov = {"poseset_summary": f"{a.poseset_summary} (sha256 {_sha(a.poseset_summary)})",
            "pilot_summary": f"{a.pilot_summary} (sha256 {_sha(a.pilot_summary)})",
            "audit_summary": f"{a.audit_summary} (sha256 {_sha(a.audit_summary)})", "passes": queues, "pass_code_revisions": revs,
            "code_revision": me["code_revision"], "dirty": me["dirty"], "keys": [k0, k4],
            "bootstrap": f"{BOOT_DRAWS} object draws, seed {BOOT_SEED}, 95 % percentile interval", "audit_level": PRIMARY}
    rep["gate"] = gate(psum, gsum, rep["published"])
    if any(rep["gate"].values()):
        with open(spath, "w") as f:
            json.dump({"gate": rep["gate"], "provenance": prov}, f, indent=1)
        print(f"STOP: the published numbers are not reproduced; wrote {spath}", flush=True)
        raise SystemExit(3)
    for name, p in passes.items():
        rep[name] = readings_of([correct_poseset(r, p.get(key("poseset", r))) for r in precs],
                                [correct_poseset(r, p.get(key("poseset", r))) for r in pvrecs],
                                [correct_pilot(r, p.get(key("pilot", r))) for r in grecs])
    a3 = asum["readings"]["3"][PRIMARY]
    rep["audit"] = {"all": a3["poseset_1p"], "upright": a3["poseset_1p_upright"]}
    rows = {"poseset": row_table(precs + pvrecs, "poseset", [passes["contacts"], passes["replan"]]),
            "pilot": row_table(grecs, "pilot", [passes["contacts"], passes["replan"]])}
    rep["counts"] = {src: object_counts(x) for src, x in rows.items()}
    rep["by_cause"] = {src: {"all": by_cause(x), "hammers": by_cause([r for r in x if "hammer" in r["object"]])} for src, x in rows.items()}
    rep["hammer_label_changes"] = {src: hammer_label_changes(x) for src, x in rows.items()}
    rep["provenance"] = prov
    pa = {name: rep[name]["poseset"]["views"]["all"]["success"] for name in PASSES}
    ga = {name: rep[name]["pilot"]["test"]["gg_vs_c10"]["success"] for name in PASSES}
    rep["wandb"] = log_run("sixdof-g1p-readings", {"contacts": a.contacts, "replan": a.replan, "out": a.out},
                           {"poseset_1p_success": pa, "pilot_gg_vs_c10_success": ga,
                            "placed": {src: c["all"]["placed"] for src, c in rep["counts"].items()},
                            "damage": {src: c["all"]["damage"] for src, c in rep["counts"].items()}})
    with open(jpath, "w") as f:
        json.dump(rep, f, indent=1)
    with open(os.path.join(a.out, "readings_summary.md"), "w") as f:
        f.write(readings_markdown(rep))
    print(f"wrote {a.out}", flush=True)
    return rep


def _ci(x: dict[str, Any] | None) -> str:
    return "–" if not x else f"{100 * x['diff']:+.2f} [{100 * x['lo']:+.2f}, {100 * x['hi']:+.2f}]"


def _rate(r: dict[str, Any] | None, k: str) -> str:
    return "–" if not r or r.get(k) is None else f"{100 * r[k]:.1f}"


def _arrow(xs: list[Any]) -> str:
    return " → ".join(str(x) for x in xs)


def _frac(x: dict[str, Any] | None) -> str:
    return "–" if not x else f"{x['placed']} / {x['cases']}"


SOURCES = (("poseset", "Pose set"), ("pilot", "GraspGen pilot"))


def readings_markdown(rep: dict[str, Any]) -> str:
    """readings_summary.md: the gate, reading (i) (the pose set's and the pilot's ceilings, published beside the
    corrected passes, the audit's sensitivity lines beside the published), readings (ii) and (iii) (placed and damage
    per object; the re-plan pass's changed rows by cause) and the hammer rows whose label changed (numbers only; the
    README states them)."""
    L = ["# G1′ readings: corrected ceilings, per-object counts and changes by cause", "",
         "Corrected: each stored row with its replay's values of the replay-gate fields (G1′ with the box contacts on; then "
         "also with the re-plan at the hold), the ceilings and the drop by the published rules, paired by case; "
         f"differences in points with the 95 % object bootstrap ({rep['provenance']['bootstrap']}). Stage 2 and the "
         "oracle arm are not replayed here.", "", "## Gate", "",
         "- The published numbers recomputed from the stored records: " + ("all equal." if not any(rep["gate"].values())
                                                                           else json.dumps(rep["gate"])), "",
         "## Reading (i): the pose set's ceilings (test side)", "",
         f"| view | measure | published | audit: unflagged cases ({rep['provenance']['audit_level']}) | contacts on | contacts on + re-plan |",
         "|---|---|---|---|---|---|"]
    for view in VIEWS:
        xs, a = [rep[p]["poseset"]["views"][view] for p in PASSES], rep["audit"].get(view)
        cells = [xs[0], a, xs[1], xs[2]]
        L.append(f"| {view} | cases (objects) | " + " | ".join("–" if x is None else f"{x['cases']} ({x['objects']})" for x in cells) + " |")
        for side, name in (("rate_a", "10-row"), ("rate_b", "3-row")):
            L.append(f"| {view} | {name} success % | " + " | ".join(_rate(x and x[side], "success") for x in cells) + " |")
        for n in THREE:
            L.append(f"| {view} | 10-row − 3-row, {n} | " + " | ".join(_ci(x and x[n]) for x in cells) + " |")
    L += ["", "Hammer cases by class (the pose set's reading 4, over the paired cases):", "",
          "| class | published | contacts on | contacts on + re-plan |", "|---|---|---|---|"]
    L += [f"| {c} | " + " | ".join(str(rep[p]["poseset"]["hammer_classes"].get(c, 0)) for p in PASSES) + " |" for c in cp.HAMMER_CLASSES]
    L += ["", "Hammer cases a pose placed, by object (placed / cases):", "", "| object | published | contacts on | contacts on + re-plan |",
          "|---|---|---|---|"]
    for o in dict.fromkeys(o for p in PASSES for o in rep[p]["poseset"]["hammer_by_object"]):
        L.append(f"| {o} | " + " | ".join(_frac(rep[p]["poseset"]["hammer_by_object"].get(o)) for p in PASSES) + " |")
    L += ["", "- Paired cases and drops: " + "; ".join(
        f"{PASS_NAMES[p]}: paired {rep[p]['poseset']['paired']}, dropped {json.dumps(rep[p]['poseset']['paired_dropped'])}, "
        f"pose-set records dropped {json.dumps(rep[p]['poseset']['pose_dropped'])}" for p in PASSES) + ".",
          "", "## Reading (i): the GraspGen pilot's ceilings (test side)", "", "| measure | published | contacts on | contacts on + re-plan |",
          "|---|---|---|---|"]
    ts = [rep[p]["pilot"]["test"] for p in PASSES]
    L.append("| cases (objects) | " + " | ".join(f"{t['cases']} ({t['objects']})" for t in ts) + " |")
    for arm in ("gg", "ggb", "c10", "c3"):
        L += [f"| {gg.ARM_NAMES[arm]}, {n} % | " + " | ".join(_rate(t["rates"][arm], n) for t in ts) + " |" for n in THREE]
    for k, name in (("gg_vs_c10", "GraspGen (the design's rule) − 10-pinch"), ("ggb_vs_c10", "GraspGen (the between-pad variant) − 10-pinch")):
        L += [f"| {name}, {n} | " + " | ".join(_ci(t[k][n]) for t in ts) + " |" for n in THREE]
    L += [f"| the 2-point rule, {name} | " + " | ".join(str(t[k]) for t in ts) + " |" for k, name in (("rule", "the design's rule"), ("rule_b", "the variant"))]
    L += ["", "Hammer cases by class (the pilot's reading 3; each cell: the design's rule, the variant, the 10-pinch ceiling):", "",
          "| class | published | contacts on | contacts on + re-plan |", "|---|---|---|---|"]
    for c, c10 in zip(gg.HAMMER_ROW_CLASSES, cp.HAMMER_CLASSES):
        L.append(f"| {c} / {c10} | " + " | ".join(
            f"{rep[p]['pilot']['hammer_classes']['gg'][c]}, {rep[p]['pilot']['hammer_classes']['ggb'][c]}, {rep[p]['pilot']['hammer_c10_classes'][c10]}"
            for p in PASSES) + " |")
    L += ["", "- Paired cases and drops: " + "; ".join(
        f"{PASS_NAMES[p]}: paired {json.dumps(rep[p]['pilot']['paired'])}, dropped {json.dumps(rep[p]['pilot']['paired_dropped'])}, "
        f"pilot records dropped {rep[p]['pilot']['pilot_dropped']}" for p in PASSES) + ".",
          "", "## Readings (ii) and (iii): placed and damage per object (every replayed row, both sides)", "",
          "Rows: the replayed rows (the pilot's gate row left out: it is the pose set's across pinch at 0.5). Damage: an "
          "irreversible label. Each cell: published → contacts on → contacts on + re-plan.", ""]
    for src, name in SOURCES:
        c = rep["counts"][src]
        L += [f"**{name}**", "", "| object | rows | placed | damage |", "|---|---|---|---|"]
        L += [f"| {o} | {x['rows']} | {_arrow(x['placed'])} | {_arrow(x['damage'])} |" for o, x in c["by_object"].items()]
        L += [f"| **{k}** | {c[k]['rows']} | {_arrow(c[k]['placed'])} | {_arrow(c[k]['damage'])} |" for k in ("hammers", "all")]
        L.append("")
    L += ["## Reading (iii): the re-plan pass's changed rows by cause (against the stored rows)", "",
          "A changed row: a replay-gate field differs from the stored row's. Cause: the recorder's first box contact, a raise "
          "of the carry, or both. Each cell: stored → re-plan pass.", ""]
    for src, name in SOURCES:
        for part in ("all", "hammers"):
            b = rep["by_cause"][src][part]
            L += [f"**{name}, {part}**", "", "| cause | rows | placed | damage | label transitions (the 5 most frequent) |", "|---|---|---|---|---|"]
            for k in cs.CAUSES:
                x = b[k]
                top = sorted(x["transitions"].items(), key=lambda t: (-t[1], t[0]))[:5]
                L.append(f"| {k} | {x['rows']} | {_arrow(x['placed'])} | {_arrow(x['damage'])} | " + ("; ".join(f"{t} {n}" for t, n in top) or "–") + " |")
            L.append("")
    L += ["## The hammer rows whose label changed", "", "The recorder's first box contact is the only cause named here.", ""]
    for src, name in SOURCES:
        for p in ("contacts", "replan"):
            x = rep["hammer_label_changes"][src][p]
            L.append(f"- {name}, {PASS_NAMES[p]}: {x['rows']} rows; first contact {json.dumps(x['first_contact'], ensure_ascii=False)}; "
                     f"transitions {json.dumps(x['transitions'], ensure_ascii=False)}" + (f"; causes {json.dumps(x['by_cause'])}" if "by_cause" in x else "") + ".")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> Any:
    import argparse

    ap = argparse.ArgumentParser(prog="sixdof_readings")
    ap.add_argument("--poseset-summary", required=True, help="the published pose-set summary (poseset_summary.json)")
    ap.add_argument("--pilot-summary", required=True, help="the published pilot summary (graspgen_summary.json)")
    ap.add_argument("--audit-summary", required=True, help="the palm-contact audit's summary (audit_summary.json)")
    ap.add_argument("--contacts", required=True, help="the G1′ queue with the box contacts on (its unit_k and report)")
    ap.add_argument("--replan", required=True, help="the G1′ queue with the re-plan at the hold on too")
    ap.add_argument("--out", required=True)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
