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

"""The level-A summary of the sixdof rerun (design 2026-11-08-sixdof-rerun-design.md, section 4, and the level-A summary
rulings of 9 Nov): readings 1 to 7 on the level-A record (the execute pass on the contacts sim, its prepass, the plan
step and the tags pass), each with the published pilot's numbers (recomputed from the published records as the gate)
and the corrected ones (the G1′ contacts + re-plan pass's arms: the pilot's capped rows and the 10- and 3-pinch
ceilings) beside the level-A arms; the smoke review's two additions (the loss-phase split of the 30–45° rows that lifted
and did not place; the object's tilt at release); the join with oracle-c and Stage 2c R4 as a labelled extra. Both
provenance gates run first (curve_sixdof_rerun.provenance_gate, tags_gate); then every other input loads and the
published and corrected numbers must reproduce; only then do the level-A records open, once.

Level-A arms per case and width rule (A: the design's rule, B: the between-pad variant; the pilot's ceiling rule over the
rows in confidence rank, one argsort per case, so the bins merge): ``a30`` / ``a30_b`` the ≤ 30° rows either rule keeps
without the pilot's cap (kept_all, kept_all_b) and without a clearance flag; ``a30n`` / ``a30n_b`` the same with the
flagged rows (amendment 6); ``a45`` / ``a45_b`` ``a30`` ∪ the 30–45° rows of the rule executed to a label; ``cap`` /
``cap_b`` the pilot's capped rows (kept, kept_b) with the rerun's labels (ruling 3's check). A case drops when a pass
stopped it or one of LEVEL_A_ARMS is undetermined."""

from __future__ import annotations

import glob
import json
import os
import statistics
from collections import Counter
from typing import Any

from rpent.skills.pick_place.protocol import IRREVERSIBLE

from robots.mujoco_lab import curve_graspgen as gg
from robots.mujoco_lab import curve_sixdof_rerun as rr

TAGS_CODE = "58845d895aa3425c86b3a3e29b36b54a541af14e"   # RPent sixdof: the tags pass's code (the level-A summary rulings)
PLACED = "placed_stable"
ABOVE, BELOW = "above the 2-point margin", "below the 2-point margin"   # the pose-set ablation's labels (curve_poseset.ceiling_reading)
LEVEL_A_ARMS = ("a30", "a30_b", "a30n", "a30n_b", "a45", "a45_b")
CHECK_ARMS = ("cap", "cap_b")
ARMS = CHECK_ARMS + LEVEL_A_ARMS
ARM_NAMES = {"cap": "≤ 30°, the pilot's capped rows, rule A", "cap_b": "≤ 30°, the pilot's capped rows, rule B",
             "a30": "≤ 30°, uncapped, with clearance, rule A", "a30_b": "≤ 30°, uncapped, with clearance, rule B",
             "a30n": "≤ 30°, uncapped, without clearance, rule A", "a30n_b": "≤ 30°, uncapped, without clearance, rule B",
             "a45": "≤ 45°, rule A", "a45_b": "≤ 45°, rule B"}
RULE_KEYS = {"a": {"30": "a30", "30n": "a30n", "45": "a45", "6": "a6"}, "b": {"30": "a30_b", "30n": "a30n_b", "45": "a45_b", "6": "a6_b"}}
PUB_ARMS = gg.ARMS                       # gg, ggb, c10, c3, oracle, r4: the published pilot's arms
CORR_ARMS = ("gg", "ggb", "c10", "c3")   # the G1′ readings' arms (ruling 2)
EXTRA_ARMS = ("oracle_c", "r4_c")        # the labelled extra: oracle-c (R0's key) and Stage 2c's R4
PITCHER_8 = ("ycb__pitcher", 8)
ROW_OBJECTS = ("ycb__pitcher", "ycb__bowl")
OUTCOMES = ("placed", "lifted, not placed", "not lifted", "not clean")
THREE = ("success", "irr", "give_up")
G2_30 = "not applicable: the ≤ 30° rows move on the lab hand path, not on a cuRobo plan"


# ------------------------------------------------------------------ the rows and the ceilings of one case

def _row(r: dict[str, Any], t: dict[str, Any], bin_: str) -> dict[str, Any]:
    run = r.get("run")
    out = {"id": r["id"], "rank": int(r["rank"]), "bin": bin_, "approach_deg": float(r["approach_deg"]), "why": r.get("why"),
           "why_b": r.get("why_b"), "clearance_first": (r.get("clearance") or {}).get("first"), "disagree": r.get("disagree"),
           "point": t.get("point", r.get("point")), "yaw": t.get("yaw", r.get("yaw")), "ran": run is not None}
    out.update({k: t[k] for k in ("f", "r_axis", "h", "near", "near_current")})
    out.update({k: (run or {}).get(k) for k in ("label", "artifact", "error", "damage", "loss_phase", "max_z", "free", "free_failure",
                                                 "unreachable", "box_contacts", "replan")})
    return out


def level_a_rows(ex: dict[str, Any], tg: dict[str, Any]) -> list[dict[str, Any]]:
    """One executed record's rows of both bins in confidence rank, each with its tags (the tags record), its run's
    outcome fields and each arm's flag (the module docstring), ``flagged`` (the row failed the clearance), ``x6`` (a
    30–45° row executed to a label) and ``a6`` / ``a6_b`` (and its rule passes every lab step). Refuses a tags record
    whose rows are not the executed record's."""
    t30, t6 = {r["id"]: r for r in tg["rows30"]}, {r["id"]: r for r in tg["rows6"]}
    if set(t30) != {r["id"] for r in ex["rows30"]} or set(t6) != {r["id"] for r in ex["rows6"]}:
        raise ValueError(f"{ex['object']} {ex['idx']}: the tags record's rows are not the executed record's")
    out = []
    for r in ex["rows30"]:
        x = _row(r, t30[r["id"]], rr.VERTICAL_BIN)
        flagged = r.get("clearance") is not None
        x.update(cap=bool(r.get("kept")), cap_b=bool(r.get("kept_b")), a30n=bool(r.get("kept_all")), a30n_b=bool(r.get("kept_all_b")),
                 flagged=flagged, x6=False, a6=False, a6_b=False, chosen=False)
        x.update(a30=x["a30n"] and not flagged, a30_b=x["a30n_b"] and not flagged)
        out.append(x)
    for r in ex["rows6"]:
        x = _row(r, t6[r["id"]], rr.LEVEL_A_BIN)
        x6 = x["ran"] and x["label"] is not None
        x.update(cap=False, cap_b=False, a30n=False, a30n_b=False, a30=False, a30_b=False, flagged=r.get("clearance") is not None,
                 x6=x6, a6=x6 and r.get("why") is None, a6_b=x6 and r.get("why_b") is None, chosen=r.get("candidate") is not None,
                 why_width=r.get("why_width"), why_width_b=r.get("why_width_b"), reach_candidate=(r.get("reach") or {}).get("candidate"))
        out.append(x)
    for x in out:
        x.update(a45=x["a30"] or x["a6"], a45_b=x["a30_b"] or x["a6_b"])
    return sorted(out, key=lambda x: x["rank"])


def level_a_ceilings(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each arm's ceiling (curve_graspgen.ceiling: the pilot's rule over the arm's rows in rank order)."""
    return {k: gg.ceiling(rows, k) for k in ARMS}


CASE_KEYS = ("object", "idx", "side", "hammer", "upright", "kind", "thin", "com_f", "length", "tilt_deg", "n_grasps")
DROP_PREPASS = "the prepass skipped, dropped or stopped the case"
DROP_STOP = "the execute pass stopped the case"
DROP_ERROR = "the execute pass raised on the case"


def case_view(ex: dict[str, Any], tg: dict[str, Any] | None) -> dict[str, Any]:
    """One case's level-A view: its fields, rows (level_a_rows; also for a case the execute pass stopped, for the
    funnel), ceilings (``a``: None unless the execute pass completed the case) and ``dropped`` (why the case leaves the
    paired set: a pass skipped, dropped or stopped it, or one of LEVEL_A_ARMS is undetermined; None)."""
    v: dict[str, Any] = {k: ex.get(k) for k in CASE_KEYS}
    v.update(idx=int(ex["idx"]), rows=[], a=None, counts30=ex.get("counts30"), counts6=ex.get("counts6"))
    if ex.get("skipped") or ex.get("dropped") or ex.get("stop") or ex.get("error"):
        return dict(v, dropped=DROP_PREPASS)
    if tg is None:
        raise ValueError(f"{ex['object']} {ex['idx']}: no tags record")
    rows = level_a_rows(ex, tg)
    if ex.get("exec_stop") or ex.get("exec_error"):
        return dict(v, rows=rows, dropped=DROP_STOP if ex.get("exec_stop") else DROP_ERROR)
    a = level_a_ceilings(rows)
    und = [k for k in LEVEL_A_ARMS if not a[k]["determined"]]
    return dict(v, rows=rows, a=a, dropped=f"undetermined: {', '.join(und)}" if und else None)


# ------------------------------------------------------------------ reading 2: the paired differences

def level_a_reading(b: dict[str, Any]) -> str:
    """The pose-set ablation's three labels (curve_poseset.ceiling_reading's rule) on a paired success difference g with
    its interval [lo, hi], against the 2-point margin: above when g > EPS and lo > 0; below when hi < EPS; else undecided
    at the view's case count."""
    from robots.mujoco_lab.curve import EPS

    x = b["success"]
    if x["diff"] > EPS and x["lo"] > 0.0:
        return ABOVE
    if x["hi"] < EPS:
        return BELOW
    return f"undecided at n = {b['cases']}"


def _paired(vs: list[dict[str, Any]], a: str, b: str, without: tuple[str, int] | None = None, b_from: str = "a") -> dict[str, Any]:
    from robots.mujoco_lab.curve_oracle import paired_outcomes

    return paired_outcomes([(v["object"], v["a"][a], v[b_from][b]) for v in vs if (v["object"], int(v["idx"])) != without])


def headline(vs: list[dict[str, Any]], without: tuple[str, int] | None = None) -> dict[str, dict[str, Any]]:
    """The headline per rule: ≤ 45° − ≤ 30° (with clearance), paired by case, the 95 % object bootstrap (BOOT_DRAWS,
    BOOT_SEED) and the three-label reading; ``without``: one (object, idx) left out (the pitcher:8 line)."""
    out = {}
    for name, a, b in (("A", "a45", "a30"), ("B", "a45_b", "a30_b")):
        x = _paired(vs, a, b, without)
        out[name] = dict(x, reading=level_a_reading(x) if x["cases"] else None)
    return out


def extras(vs: list[dict[str, Any]], with_corr: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The labelled extras, interval only: ≤ 45° − ≤ 30° without clearance per rule (``vs``), and ≤ 45° − the corrected
    10-pinch ceiling per rule (``with_corr``: the views that also hold the corrected arms)."""
    return {"A_vs_30n": _paired(vs, "a45", "a30n"), "B_vs_30n": _paired(vs, "a45_b", "a30n_b"),
            "A_vs_c10": _paired(with_corr, "a45", "c10", b_from="corr"), "B_vs_c10": _paired(with_corr, "a45_b", "c10", b_from="corr")}


def _rates_of(vs: list[dict[str, Any]], arms: tuple[str, ...], get: Any) -> dict[str, dict[str, Any]]:
    from robots.mujoco_lab.curve_oracle import _rates

    return {k: _rates([get(v)[k] for v in vs]) for k in arms}


# ------------------------------------------------------------------ reading 1: the funnels

def _zero(keys: tuple[str, ...]) -> dict[str, int]:
    return dict.fromkeys(keys, 0)


F30 = ("cases", "generated", "within", "mapped", "width", "centred", "reach", "distinct", "kept", "width_b", "pads_free", "reach_b",
       "distinct_b", "kept_b", "uncapped", "uncapped_b", "clearance_fail", "flagged", "ran", "a30", "a30_b")
F45 = ("within", "width", "width_b", "either_width", "clearance_ok", "reach_ok", "chosen", "disagree", "planned", "miss", "failed",
       "g2_passed", "labelled", "rule_a", "rule_b", "distinct", "distinct_b")


def funnel(vs: list[dict[str, Any]]) -> dict[str, Any]:
    """Reading 1 over every case record, as in the counts report (a case the execute pass stopped keeps its prepass
    rows; a case the prepass dropped adds its prepass counts and no row, so an object stays even when all its cases
    drop): per object and in all, the ≤ 30° funnel (the pilot's counts from the prepass, the uncapped rows, the clearance
    failures and the flagged rows that ran) and the 30–45° funnel (width rules, clearance, the lab path check, to the
    planner, cuRobo planned, missed or failed, G2 passed, executed to a label, each rule, after the pilot's de-dup of the
    rule's rows on their mapped point and yaw); each rule's first stop per 30–45° row (``why6``); the generator's shares
    within 30°, within 45° and past every step (per rule); the clearance failures by first body/obstacle pair per bin;
    cuRobo's failures by reason; G2's by check."""
    by: dict[str, dict[str, dict[str, int]]] = {}
    why6: dict[str, dict[str, Counter]] = {}
    pairs: dict[str, Counter] = {"30": Counter(), "45": Counter()}
    depth: dict[str, list[float]] = {"30": [], "45": []}
    failed, g2 = Counter(), Counter()
    for v in vs:
        x = by.setdefault(v["object"], {"30": _zero(F30), "45": _zero(F45)})
        w = why6.setdefault(v["object"], {"a": Counter(), "b": Counter()})
        c30, s30 = v.get("counts30") or {}, x["30"]
        s30["cases"] += 1
        for k in ("generated", "mapped", "width", "centred", "reach", "distinct", "kept", "width_b", "pads_free", "reach_b", "distinct_b", "kept_b"):
            s30[k] += int(c30.get(k, 0))
        rows30 = [r for r in v["rows"] if r["bin"] == rr.VERTICAL_BIN]
        rows6 = [r for r in v["rows"] if r["bin"] == rr.LEVEL_A_BIN]
        s30["within"] += len(rows30)
        s30["uncapped"] += sum(r["a30n"] for r in rows30)
        s30["uncapped_b"] += sum(r["a30n_b"] for r in rows30)
        s30["clearance_fail"] += sum(r["clearance_first"] is not None for r in rows30)
        s30["flagged"] += sum(r["ran"] and r["flagged"] for r in rows30)
        s30["ran"] += sum(r["ran"] for r in rows30)
        s30["a30"] += sum(r["a30"] for r in rows30)
        s30["a30_b"] += sum(r["a30_b"] for r in rows30)
        s6 = x["45"]
        s6["within"] += len(rows6)
        for r in rows6:
            wa, wb = r["why_width"] is None, r["why_width_b"] is None
            s6["width"] += wa
            s6["width_b"] += wb
            s6["either_width"] += wa or wb
            s6["clearance_ok"] += (wa or wb) and r["clearance_first"] is None
            s6["reach_ok"] += r["reach_candidate"] is not None
            s6["chosen"] += r["chosen"]
            s6["disagree"] += r["disagree"] is not None
            s6["planned"] += r["free"] == "ok"
            s6["miss"] += r["free"] == "miss"
            s6["failed"] += r["free"] == "failed"
            s6["g2_passed"] += r["free"] == "ok" and r["unreachable"] is None
            s6["labelled"] += r["x6"]
            s6["rule_a"] += r["a6"]
            s6["rule_b"] += r["a6_b"]
            w["a"][str(r["why"])] += 1
            w["b"][str(r["why_b"])] += 1
            if r["free"] == "failed" and r["free_failure"]:
                failed[r["free_failure"]["reason"]] += 1
            if r["unreachable"] is not None:
                g2[r["unreachable"].get("check", "unknown")] += 1
        for k, flag in (("distinct", "a6"), ("distinct_b", "a6_b")):
            s6[k] += sum(bool(r[flag]) for r in rows6) - len(gg.dedup(rows6, flag))
        for b, rows in (("30", rows30), ("45", rows6)):
            for r in rows:
                if r["clearance_first"] is not None:
                    pairs[b][r["clearance_first"].get("pair", "unknown")] += 1
                    if r["clearance_first"].get("depth") is not None:
                        depth[b].append(float(r["clearance_first"]["depth"]))
    al = {"30": {k: sum(x["30"][k] for x in by.values()) for k in F30}, "45": {k: sum(x["45"][k] for x in by.values()) for k in F45}}
    n = al["30"]["generated"]

    def share(k: int) -> float | None:
        return k / n if n else None

    al.update(generated=n, share_within_30=share(al["30"]["within"]), share_within_45=share(al["30"]["within"] + al["45"]["within"]),
              share_every_step=share(al["30"]["a30"] + al["45"]["distinct"]), share_every_step_b=share(al["30"]["a30_b"] + al["45"]["distinct_b"]))
    return {"by_object": by, "all": al, "why6": {o: {r: dict(c) for r, c in w.items()} for o, w in why6.items()},
            "clearance_pairs": {b: dict(c) for b, c in pairs.items()},
            "clearance_depth": {b: {"rows": len(d), "median": statistics.median(d) if d else None, "max": max(d) if d else None}
                                for b, d in depth.items()},
            "free_failed": al["45"]["failed"], "free_failed_by": dict(failed), "g2_failed": dict(g2)}


# ------------------------------------------------------------------ readings 3 to 5

def hammers(vs: list[dict[str, Any]]) -> dict[str, Any]:
    """Reading 3 over the hammer views: the cases by class (curve_graspgen.HAMMER_ROW_CLASSES) at ≤ 30° and ≤ 45° under
    each rule; per case every clean placing row of a ≤ 45° arm with its bin, approach angle, f, distance from the balance
    point and from the head end along the axis (|f − com_f| × length, |f − HAMMER_HEAD_F| × length), tag and arms; and
    the rows that lifted it (a clean label other than not_lifted or missed)."""
    from robots.mujoco_lab.curve_poseset import LIFT_FAILS

    keys = ("a30", "a30_b", "a45", "a45_b")
    hv = [v for v in vs if v["hammer"] and v["a"] is not None]
    out: dict[str, Any] = {"cases": len(hv), "objects": sorted({v["object"] for v in hv}),
                           "classes": {k: dict.fromkeys(gg.HAMMER_ROW_CLASSES, 0) for k in keys}, "per_case": []}
    for v in hv:
        if v["object"] not in gg.HAMMER_HEAD_F:
            raise ValueError(f"{v['object']}: no head end in HAMMER_HEAD_F")
        L, cf, head = float(v["length"]), float(v["com_f"]), gg.HAMMER_HEAD_F[v["object"]]
        cls = {k: gg.hammer_rows_class(v["rows"], k) for k in keys}
        for k, c in cls.items():
            out["classes"][k][c] += 1
        placed, lifted = [], []
        for r in v["rows"]:
            rules = [k for k in ("a45", "a45_b") if r[k]]
            if not rules or not gg.clean_row(r):
                continue
            if r["label"] == PLACED:
                placed.append({"id": r["id"], "bin": r["bin"], "approach_deg": r["approach_deg"], "f": r["f"],
                               "balance_m": abs(float(r["f"]) - cf) * L, "head_m": abs(float(r["f"]) - head) * L, "tag": gg.row_tag(r),
                               "rules": rules})
            if r["label"] not in LIFT_FAILS:
                lifted.append(r["id"])
        out["per_case"].append({"object": v["object"], "idx": v["idx"], "com_f": cf, "length": L, "classes": cls, "placed": placed,
                                "lifted": lifted})
    return out


def objects_line(vs: list[dict[str, Any]], objects: tuple[str, ...] = ROW_OBJECTS) -> dict[str, dict[str, Any]]:
    """Reading 4 per object and rule: the cases, those with an executed tilted row of the rule, those rows by tag, the
    clean ones and their label mix, the placing ones by tag, and the cases ≤ 45° and ≤ 30° (with clearance) place."""
    out: dict[str, dict[str, Any]] = {}
    for o in objects:
        x = [v for v in vs if v["object"] == o and v["a"] is not None]
        out[o] = {}
        for name, k in RULE_KEYS.items():
            t = [r for v in x for r in v["rows"] if r[k["6"]]]
            clean = [r for r in t if gg.clean_row(r)]
            pl = [r for r in clean if r["label"] == PLACED]
            out[o][name] = {"cases": len(x), "cases_with_tilted_row": sum(any(r[k["6"]] for r in v["rows"]) for v in x),
                            "tilted_rows": len(t), "tilted_tags": dict(Counter(gg.row_tag(r) for r in t)), "tilted_clean": len(clean),
                            "tilted_labels": dict(Counter(r["label"] for r in clean)), "tilted_placing": len(pl),
                            "tilted_placing_tags": dict(Counter(gg.row_tag(r) for r in pl)),
                            "placed_45": sum(bool(v["a"][k["45"]]["success"]) for v in x),
                            "placed_30": sum(bool(v["a"][k["30"]]["success"]) for v in x)}
    return out


def tilted_tags(vs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Reading 5 per rule: the clean placing 30–45° rows by tag (curve_graspgen.row_tag), the cases with one of each tag,
    the rows by object; the gain cases (≤ 45° places, ≤ 30° with clearance does not) by the tags of their placing
    tilted rows; the loss cases (the reverse; none by construction, the arm only adds rows); the clean placing ≤ 30° rows
    of the rule by tag beside them."""
    out = {}
    for name, k in RULE_KEYS.items():
        rows, cases, gain_tags, vert = Counter(), Counter(), Counter(), Counter()
        by: dict[str, Counter] = {}
        gain = loss = 0
        for v in vs:
            if v["a"] is None:
                continue
            tags = [gg.row_tag(r) for r in v["rows"] if r[k["6"]] and gg.clean_row(r) and r["label"] == PLACED]
            vert.update(gg.row_tag(r) for r in v["rows"] if r[k["30"]] and gg.clean_row(r) and r["label"] == PLACED)
            rows.update(tags)
            cases.update(set(tags))
            if tags:
                by.setdefault(v["object"], Counter()).update(tags)
            if v["a"][k["45"]]["success"] and not v["a"][k["30"]]["success"]:
                gain += 1
                gain_tags.update(set(tags))
            loss += bool(v["a"][k["30"]]["success"]) and not v["a"][k["45"]]["success"]
        out[name] = {"rows": dict(rows), "cases": dict(cases), "by_object": {o: dict(c) for o, c in by.items()},
                     "gain": {"cases": gain, "tags": dict(gain_tags)}, "loss": {"cases": loss}, "vertical_rows": dict(vert)}
    return out


# ------------------------------------------------------------------ the smoke review's additions and the by-cause tables

def _executed(vs: list[dict[str, Any]], b: str) -> list[dict[str, Any]]:
    """The rows of a bin that ran to a label: ≤ 30° rows that ran, 30–45° rows executed to a label (x6)."""
    if b == "30":
        return [r for v in vs for r in v["rows"] if r["bin"] == rr.VERTICAL_BIN and r["ran"] and r["label"] is not None]
    return [r for v in vs for r in v["rows"] if r["x6"]]


def _outcome(r: dict[str, Any]) -> str:
    from robots.mujoco_lab.curve_poseset import LIFT_FAILS

    if not gg.clean_row(r):
        return OUTCOMES[3]
    if r["label"] == PLACED:
        return OUTCOMES[0]
    return OUTCOMES[2] if r["label"] in LIFT_FAILS else OUTCOMES[1]


def loss_phase_split(vs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The clean rows that lifted the object and did not place it, by the rollout's loss phase ("none": the pads held it
    to the release) and by label: the 30–45° rows executed to a label (``45``) and the ≤ 30° rows that ran (``30``, the
    reference)."""
    out = {}
    for b in ("45", "30"):
        x = [r for r in _executed(vs, b) if _outcome(r) == OUTCOMES[1]]
        out[b] = {"rows": len(x), "phases": dict(Counter(r["loss_phase"] or "none" for r in x)), "labels": dict(Counter(r["label"] for r in x))}
    return out


def _stats(x: list[float]) -> dict[str, Any]:
    return {"rows": len(x), "median": statistics.median(x) if x else None, "min": min(x) if x else None, "max": max(x) if x else None}


RELEASE_TILT = ("planned, not measured: the tilt of the object's rest-pose vertical at the release with the object rigid in "
                "the hand, which equals the row's approach angle (hand_path6's release rotation; level_a/release_tilt.py). "
                "The records hold no object pose at the release. A row that did not lift the object has no release, and "
                "its value is its approach angle only.")


def release_tilt(vs: list[dict[str, Any]]) -> dict[str, Any]:
    """The planned tilt at the release (RELEASE_TILT: the approach angle) of the 30–45° rows executed to a label, by
    object, in all and, exploratory, by outcome (placed, lifted and not placed, not lifted, not clean)."""
    rows = [(v["object"], r) for v in vs for r in v["rows"] if r["x6"]]
    objs = list(dict.fromkeys(o for o, _ in rows))
    return {"what": RELEASE_TILT, "by_object": {o: _stats([r["approach_deg"] for p, r in rows if p == o]) for o in objs},
            "all": _stats([r["approach_deg"] for _, r in rows]),
            "by_outcome": {k: _stats([r["approach_deg"] for _, r in rows if _outcome(r) == k]) for k in OUTCOMES}}


def by_cause(vs: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    """Per bin, the rows that ran to a label by what the contacts sim and the re-plan add (curve_sixdof.row_cause: a box
    contact, a raise of the carry at the hold, both; "neither"): the rows, the placed and the damage (an irreversible
    label) counts."""
    from robots.mujoco_lab.curve_sixdof import CAUSES, row_cause

    out = {}
    for b in ("30", "45"):
        x = {k: {"rows": 0, "placed": 0, "damage": 0} for k in CAUSES + ("neither",)}
        for r in _executed(vs, b):
            c = x[row_cause(r) or "neither"]
            c["rows"] += 1
            c["placed"] += r["label"] == PLACED
            c["damage"] += r["label"] in IRREVERSIBLE
        out[b] = x
    return out


def replan_d_bins(vs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per bin, G1′'s re-plan reading (curve_sixdof.g1p_report) over the rows that ran to a label: the rows that rose
    and that both pads held at the end of the hold by d (curve_sixdof.D_BINS, (lo, hi]), each bin's rows, raised and
    unreachable; the rows without a re-plan record, that did not rise, that rose but were not held."""
    from robots.mujoco_lab.curve_sixdof import D_BINS, _raised

    out = {}
    for b in ("30", "45"):
        x: dict[str, Any] = {"bins": {name: {"rows": 0, "raised": 0, "unreachable": 0} for name, _, _ in D_BINS}, "no_record": 0,
                             "not_lifted": 0, "lifted_not_held": 0}
        for r in _executed(vs, b):
            p = r["replan"]
            if not p:
                x["no_record"] += 1
            elif not p.get("lifted"):
                x["not_lifted"] += 1
            elif not p.get("held") or p.get("d_mm") is None:
                x["lifted_not_held"] += 1
            else:
                y = x["bins"][next(name for name, lo, hi in D_BINS if lo < float(p["d_mm"]) <= hi)]
                y["rows"] += 1
                y["raised"] += int(_raised(r))
                y["unreachable"] += int(bool(p.get("unreachable")))
        out[b] = x
    return out


def ruling3(vs: list[dict[str, Any]], corr: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    """Ruling 3's check. Row level: every pilot row either rule capped (kept, kept_b) that the corrected record simulated,
    its corrected label (the G1′ contacts + re-plan pass) against the rerun's: compared, agree, differ (listed with the
    rerun row's cause, curve_sixdof.row_cause), missing (the rerun has no label). Case level: the cases where a capped
    ceiling places and the uncapped one without clearance (its superset) does not (0 required)."""
    from robots.mujoco_lab.curve_sixdof import row_cause

    out: dict[str, Any] = {"rows": {"compared": 0, "agree": 0, "differ": 0, "missing": 0}, "differences": [], "capped_places_uncapped_not": 0}
    for v in vs:
        mine = {r["id"]: r for r in v["rows"] if r["bin"] == rr.VERTICAL_BIN}
        rec = corr.get((v["object"], int(v["idx"])))
        for c in (rec or {}).get("rows") or []:
            if c.get("gate") or not (c.get("kept") or c.get("kept_b")) or "seconds" not in c:
                continue
            r = mine.get(c["id"])
            if r is None or r["label"] is None:
                out["rows"]["missing"] += 1
                continue
            out["rows"]["compared"] += 1
            if r["label"] == c.get("label"):
                out["rows"]["agree"] += 1
            else:
                out["rows"]["differ"] += 1
                out["differences"].append({"object": v["object"], "idx": int(v["idx"]), "row": c["id"], "corrected": c.get("label"),
                                           "rerun": r["label"], "cause": row_cause(r) or "neither"})
        a = v["a"]
        if a is not None:
            out["capped_places_uncapped_not"] += bool((a["cap"]["success"] and not a["a30n"]["success"])
                                                      or (a["cap_b"]["success"] and not a["a30n_b"]["success"]))
    return out


def success_2x2(vs: list[dict[str, Any]], a: str, b: str) -> dict[str, int]:
    """The level-A arm ``a`` against the corrected arm ``b`` per case (success): both, level A only, corrected only, neither."""
    x = [(bool(v["a"][a]["success"]), bool(v["corr"][b]["success"])) for v in vs]
    return {"both": sum(p and q for p, q in x), "level_a_only": sum(p and not q for p, q in x), "corrected_only": sum(q and not p for p, q in x),
            "neither": sum(not p and not q for p, q in x)}


# ------------------------------------------------------------------ the gates

def tags_gate(tags_dirs: list[str], prepass_dirs: list[str], code: str = TAGS_CODE, sim: str = rr.LEVEL_A_SIM) -> dict[str, Any]:
    """The tags units' provenance gate: each unit finished with EXIT 0 on ``code`` and ``sim`` with no dirty file, read
    one of the execute units' prepass runs (its run_config's ``prepass``) at that run's records' sha256, and each of
    those prepass runs has exactly one tags unit. Anything else raises SystemExit listing every failure."""
    from robots.mujoco_lab.curve import _sha

    bad: list[str] = []
    by: dict[str, list[str]] = {p: [] for p in prepass_dirs}
    for d in tags_dirs:
        ep = os.path.join(d, "run.exit")
        e = open(ep).read().strip() if os.path.exists(ep) else "no run.exit"
        if e != "EXIT 0":
            bad.append(f"{d}: {e}")
        if not os.path.exists(os.path.join(d, "run_metrics.json")):
            bad.append(f"{d}: the run has not finished (no run_metrics.json)")
        _, why = rr.provenance_problem(d, code, sim, True)
        if why:
            bad.append(why)
        cfg = json.load(open(os.path.join(d, "run_config.json")))
        p = cfg.get("prepass")
        if p not in by:
            bad.append(f"{d}: its prepass run is not one of the execute units'")
            continue
        by[p].append(d)
        if cfg.get("prepass_records_sha256") != _sha(os.path.join(p, "prepass_records.jsonl")):
            bad.append(f"{d}: it read other prepass records")
    bad += [f"{p}: not one tags unit ({len(t)})" for p, t in by.items() if len(t) != 1]
    if bad:
        raise SystemExit(f"tags gate: not every prepass unit has one finished tags unit on code {code} and sim {sim} that read "
                         "its records:\n" + "\n".join(bad))
    return {"code": code, "sim": sim, "tags_units": len(tags_dirs)}


TAG_FIELDS = ("f", "r_axis", "h", "near", "near_current")


def tags_vs_pilot(tgs: list[dict[str, Any]], grecs: list[dict[str, Any]], tol: float = 1e-9) -> dict[str, Any]:
    """The tags pass's ≤ 30° rows against the published pilot record's rows of the same case and id (the pilot's _tags at
    its own reset): rows compared, equal (every tag field, floats within ``tol``), differ (the first few listed), and the
    rows the pilot does not hold."""
    pil = {(r["object"], int(r["idx"])): {x["id"]: x for x in r.get("rows") or [] if not x.get("gate")} for r in grecs}
    out: dict[str, Any] = {"compared": 0, "equal": 0, "differ": 0, "missing": 0, "differences": []}
    for t in tgs:
        p = pil.get((t["object"], int(t["idx"])), {})
        for r in t["rows30"]:
            q = p.get(r["id"])
            if q is None or "f" not in q:
                out["missing"] += 1
                continue
            out["compared"] += 1
            same = all((abs(float(r[k]) - float(q[k])) <= tol) if isinstance(r[k], float) else r[k] == q.get(k) for k in TAG_FIELDS)
            out["equal" if same else "differ"] += 1
            if not same and len(out["differences"]) < 10:
                out["differences"].append({"object": t["object"], "idx": int(t["idx"]), "row": r["id"],
                                           "tags": {k: r[k] for k in TAG_FIELDS}, "pilot": {k: q.get(k) for k in TAG_FIELDS}})
    return out


def counts_check(f: dict[str, Any], counts: dict[str, Any]) -> list[str]:
    """The summary's funnel (funnel) against the gated counts report (curve_sixdof_rerun.smoke_report on the same
    records): per object both funnels' counts, rule A's first stops (why6) and the disagreements; in all the ≤ 30° rows
    run and flagged, the rows past the lab path check, the free moves by status, cuRobo's failures by reason, G2's by
    check. Lists each difference."""
    c, al, bad = counts["counts"], f["all"], []

    def cmp(name: str, mine: Any, want: Any) -> None:
        if mine != want:
            bad.append(f"{name}: {mine} vs the report's {want}")

    for o in sorted(set(f["by_object"]) | (set(counts["funnel30"]) - {"all"})):
        x, w30, w6 = f["by_object"].get(o), counts["funnel30"].get(o), counts["funnel6"].get(o)
        if x is None or w30 is None or w6 is None:
            bad.append(f"{o}: in only one of the summary's funnel and the report")
            continue
        for k in rr.FUNNEL30:
            cmp(f"funnel30.{o}.{k}", x["30"]["within" if k == "vertical" else k], w30.get(k))
        for k in rr.FUNNEL6:
            cmp(f"funnel6.{o}.{k}", x["30"]["generated"] if k == "generated" else x["45"]["within" if k == "level_a" else k], w6.get(k))
        cmp(f"why6.{o}", f["why6"].get(o, {}).get("a"), counts["why6"].get(o))
        cmp(f"disagree.{o}", x["45"]["disagree"], counts["disagree"].get(o))
    cmp("rows30_run", al["30"]["ran"], c["rows30_run"])
    cmp("rows30_flagged_run", al["30"]["flagged"], c["rows30_flagged_run"])
    cmp("path_passed", al["45"]["chosen"], c["path_passed"])
    for k, mine in (("ok", al["45"]["planned"]), ("miss", al["45"]["miss"]), ("failed", al["45"]["failed"])):
        cmp(f"free.{k}", mine, c["free"][k])
    cmp("failed_by_reason", f["free_failed_by"], c["failed_by_reason"])
    cmp("g2_fail", sum(f["g2_failed"].values()), c["g2_fail"])
    cmp("g2_by_check", f["g2_failed"], c["g2_by_check"])
    return bad


# ------------------------------------------------------------------ the inputs, the published and corrected gates, the level-A records

def readings_refs(rsum: dict[str, Any], a: Any, shas: dict[str, str], k0: str, k4: str) -> list[str]:
    """The G1′ readings summary (the corrected gate's reference) must have read the inputs that this summary reads: the
    contacts and the re-plan passes (--contacts, --replan), the pilot and the pose-set summaries (their sha256 in
    ``shas``) and the keys [k0, k4]. Lists each difference."""
    from robots.mujoco_lab.palm_audit import ref_sha

    p = rsum.get("provenance") or {}
    passes, bad = p.get("passes") or {}, []
    for n, q in (("contacts", a.contacts), ("replan", a.replan)):
        if passes.get(n) is None or os.path.realpath(passes[n]) != os.path.realpath(q):
            bad.append(f"the readings' {n} pass {passes.get(n)} is not {q}")
    for k, path in (("pilot_summary", a.pilot_summary), ("poseset_summary", a.poseset_summary)):
        if ref_sha(p.get(k)) is None or ref_sha(p.get(k)) != shas.get(path):
            bad.append(f"the readings' {k} {p.get(k)} is not {path}")
    if list(p.get("keys") or []) != [k0, k4]:
        bad.append(f"the readings' keys {p.get('keys')} are not {[k0, k4]}")
    return bad


def _open_inputs(a: Any) -> dict[str, Any]:
    """Every input but the level-A records: the published pose-set and pilot records with the G1′ readings' checks
    (sixdof_readings.run: each record file's sha256 is the one its summary read; each pass replayed only cases these
    records hold and exactly the files corrected here), the passes (sixdof_readings.load_pass), Stage 2c's and
    oracle-c's test records (their summaries' sha256, the pilot's lock and keys), and the side reports (the gated
    counts, the smoke, G1, G1′, the G1′ readings)."""
    from robots.mujoco_lab import curve_sixdof as cs
    from robots.mujoco_lab import sixdof_readings as sr
    from robots.mujoco_lab.curve import SIDES, _sha, load_stage2, records_path
    from robots.mujoco_lab.curve_poseset import _load_lines
    from robots.mujoco_lab.palm_audit import ref_sha

    psum, gsum = json.load(open(a.poseset_summary)), json.load(open(a.pilot_summary))
    pp, gp = psum["provenance"], gsum["provenance"]
    if ref_sha(gp["poseset_summary"]) != _sha(a.poseset_summary):
        raise SystemExit("the pilot summary read another pose-set summary")
    if (gp["poseset"], gp["stage2"], gp["oracle"], gp["locked"]) != (pp["runs"], pp["stage2"], pp["oracle"], pp["locked"]):
        raise SystemExit("the pilot and the pose-set summaries read other runs or another lock")
    refs = {k: gp[k].split(" (sha256 ")[0] for k in ("locked", "stage2_summary", "oracle_summary")}
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
    corrected = dict(gp["records_sha256"])
    corrected.update({records_path(d, "test"): pp["records_sha256"][d] for d in pdirs})
    corrected.update({records_path(d, "validation"): gp["validation_records_sha256"][d] for d in pvdirs})
    want = dict(corrected)
    want.update({records_path(d, "validation"): gp["validation_records_sha256"][d] for d in ovdirs})
    want.update({records_path(d, "test"): oprov[d] for d in odirs})
    want.update({records_path(d, side): s2prov[f"{d} {side}"] for d in s2dirs for side in SIDES})
    bad = sorted(p for p, h in want.items() if _sha(p) != h)
    if bad:
        raise SystemExit(f"{len(bad)} published record files are not the ones their summaries read (the first {bad[0]})")
    precs = _load_lines([records_path(d, "test") for d in pdirs])
    pvrecs = _load_lines([records_path(d, "validation") for d in pvdirs])
    grecs = _load_lines([records_path(d, side) for d in gdirs for side in SIDES])
    s2t, _ = load_stage2(s2dirs, ("test",))
    s2v, _ = load_stage2(s2dirs, ("validation",))
    orecs, ovrecs = _load_lines([records_path(d, "test") for d in odirs]), _load_lines([records_path(d, "validation") for d in ovdirs])
    passes = {"contacts": sr.load_pass(a.contacts, False), "replan": sr.load_pass(a.replan, True)}
    keys = ({("poseset", r["side"], r["object"], int(r["idx"])) for r in precs + pvrecs}
            | {("pilot", r["side"], r["object"], int(r["idx"])) for r in grecs})
    for name, q in (("contacts", a.contacts), ("replan", a.replan)):
        extra = sorted(set(passes[name]) - keys)
        if extra:
            raise SystemExit(f"the {name} pass replayed {len(extra)} cases no record holds (the first {extra[0]})")
        units = sorted(glob.glob(os.path.join(q, "unit_*")))
        errs = sr.replayed_files_errors({u: cs._unit_config(u)["records_sha256"] for u in units}, corrected)
        if errs:
            raise SystemExit(f"the {name} pass: {len(errs)} replayed files are not the records corrected here ({errs[0]})")
    s2c_sum, oc_sum = json.load(open(a.stage2c_summary)), json.load(open(a.oracle_c_summary))
    sp, op = s2c_sum["provenance"], oc_sum["provenance"]
    if ref_sha(op["stage2_summary"]) != _sha(a.stage2c_summary) or op["stage2"] != sp["runs"]:
        raise SystemExit("the oracle-c summary read another Stage 2c summary or other Stage 2c runs")
    if ref_sha(sp["locked"]) != ref_sha(gp["locked"]) or ref_sha(op["locked"]) != ref_sha(gp["locked"]) or list(op["keys"]) != [k0, k4]:
        raise SystemExit("Stage 2c or oracle-c read another lock than the pilot's, or other keys")
    c2dirs, ocdirs = sp["runs"].split(","), op["runs"].split(",")
    bad = [d for d in c2dirs if _sha(records_path(d, "test")) != sp["records_sha256"][f"{d} test"]]
    bad += [d for d in ocdirs if _sha(records_path(d, "test")) != op["oracle_records_sha256"][d]]
    if bad:
        raise SystemExit(f"Stage 2c or oracle-c test records are not the ones their summaries read (the first {bad[0]})")
    s2c, _ = load_stage2(c2dirs, ("test",))
    ocrecs = _load_lines([records_path(d, "test") for d in ocdirs])
    side = {"counts": json.load(open(a.counts)), "smoke": json.load(open(a.smoke)), "g1": json.load(open(a.g1)),
            "readings": json.load(open(a.readings_summary)),
            "g1p": {n: json.load(open(os.path.join(q, "report", "g1p_summary.json"))) for n, q in (("contacts", a.contacts), ("replan", a.replan))}}
    prov = {"inputs_sha256": {p: _sha(p) for p in (a.poseset_summary, a.pilot_summary, a.readings_summary, a.stage2c_summary,
                                                   a.oracle_c_summary, a.counts, a.smoke, a.g1)}, "keys": [k0, k4],
            "passes": {"contacts": a.contacts, "replan": a.replan}}
    return {"psum": psum, "gsum": gsum, "precs": precs, "pvrecs": pvrecs, "grecs": grecs, "s2t": s2t, "s2v": s2v, "orecs": orecs,
            "ovrecs": ovrecs, "passes": passes, "s2c": s2c, "ocrecs": ocrecs, "k0": k0, "k4": k4, "side": side, "prov": prov}


def inputs_gate(D: dict[str, Any]) -> dict[str, Any]:
    """The published gate (sixdof_readings.gate: the pose set's and the pilot's published numbers recomputed from the
    stored records) and the corrected gate (the pilot's arms corrected by the G1′ contacts + re-plan pass, as the G1′
    readings correct them, must equal the G1′ write-up's re-plan pilot block and paired counts). Returns the published
    readings and views, the corrected views and records, both gates' differences and ``stop``."""
    from robots.mujoco_lab import sixdof_readings as sr
    from robots.mujoco_lab.palm_audit import differs

    k0, k4, P = D["k0"], D["k4"], D["passes"]["replan"]
    pt, pv = D["precs"], D["pvrecs"]
    pvs, pn = sr.pose_views(pt, D["s2t"], D["orecs"], k0, k4)
    gv, gn = sr.pilot_views(D["grecs"], pt + pv, D["orecs"] + D["ovrecs"], D["s2t"] + D["s2v"], k0, k4)
    pub = {"poseset": sr.pose_reading(pvs, pn), "pilot": sr.pilot_reading(gv, gn)}
    g_pub = sr.gate(D["psum"], D["gsum"], pub)

    def key(src: str, r: dict[str, Any]) -> tuple[str, str, str, int]:
        return (src, r["side"], r["object"], int(r["idx"]))

    cg = [sr.correct_pilot(r, P.get(key("pilot", r))) for r in D["grecs"]]
    cp_ = [sr.correct_poseset(r, P.get(key("poseset", r))) for r in pt + pv]
    cv, cn = sr.pilot_views(cg, cp_, D["orecs"] + D["ovrecs"], D["s2t"] + D["s2v"], k0, k4)
    corr = sr.pilot_reading(cv, cn)
    want = D["side"]["readings"]["replan"]["pilot"]
    g_corr: list[str] = []
    for k in ("gg_vs_c10", "ggb_vs_c10"):
        g_corr += [f"{k}: {x}" for x in differs(corr["test"][k], want["test"][k], THREE)]
    g_corr += [f"rates.{k}: {corr['test']['rates'][k]} vs {want['test']['rates'][k]}" for k in CORR_ARMS
               if corr["test"]["rates"][k] != want["test"]["rates"][k]]
    g_corr += [] if corr["paired"] == want["paired"] else [f"paired {corr['paired']} vs {want['paired']}"]
    return {"pub": pub, "gv": gv, "cv": cv, "cg": cg, "corr_test": corr["test"], "g_pub": g_pub, "g_corr": g_corr,
            "stop": bool(any(g_pub.values()) or g_corr)}


def _open_level_a(ex_dirs: list[str], tg_dirs: list[str]) -> dict[str, Any]:
    """The level-A records, each file once: the execute units' exec_records.jsonl and the tags units' tags_records.jsonl,
    with each file's sha256."""
    from robots.mujoco_lab.curve_poseset import _read_once

    shas: dict[str, str] = {}
    out: dict[str, list[dict[str, Any]]] = {"exs": [], "tgs": []}
    for name, dirs, fname in (("exs", ex_dirs, "exec_records.jsonl"), ("tgs", tg_dirs, "tags_records.jsonl")):
        for d in dirs:
            path = os.path.join(d, fname)
            if path in shas:
                raise SystemExit(f"{path}: named twice")
            recs, sha = _read_once([path])            # one open: the sha256 is of the bytes that are read
            shas.update(sha)
            out[name] += recs
    return dict(out, shas=shas)


# ------------------------------------------------------------------ the readings

def by_object_table(vs: list[dict[str, Any]], la: list[dict[str, Any]], gv: list[dict[str, Any]], cv: list[dict[str, Any]]) -> dict[str, Any]:
    """Per object in the cases' order (its side from the cases): each column on its own cases with n, level A on the
    paired level-A cases (``la``), published on the published paired views, corrected on the corrected ones."""
    out = {}
    for o in dict.fromkeys(v["object"] for v in vs):
        x, p, c = ([v for v in s if v["object"] == o] for s in (la, gv, cv))
        out[o] = {"side": next(v["side"] for v in vs if v["object"] == o),
                  "level_a": dict({"n": len(x)}, **_rates_of(x, ARMS, lambda v: v["a"])),
                  "published": dict({"n": len(p)}, **_rates_of(p, PUB_ARMS, lambda v: v)),
                  "corrected": dict({"n": len(c)}, **_rates_of(c, CORR_ARMS, lambda v: v))}
    return out


def level_a_build(D: dict[str, Any], P: dict[str, Any], L: dict[str, Any], gate: dict[str, Any], tgate: dict[str, Any]) -> dict[str, Any]:
    """Readings 1 to 7, the additions, the by-cause tables, ruling 3's check, the extras and the pitcher:8 line from the
    inputs (_open_inputs), their gates (inputs_gate) and the level-A records (_open_level_a). The level-A cases must be
    the pilot's cases, each once, in the executed and the tags records."""
    from robots.mujoco_lab.curve import episode

    k0, k4 = D["k0"], D["k4"]
    tg = {(t["object"], int(t["idx"])): t for t in L["tgs"]}
    exk = [(e["object"], int(e["idx"])) for e in L["exs"]]
    pil = {(r["object"], int(r["idx"])) for r in D["grecs"]}
    if len(set(exk)) != len(exk) or len(tg) != len(L["tgs"]) or set(exk) != set(tg) or set(exk) != pil:
        raise SystemExit("the executed, the tags and the pilot's records do not hold the same cases once each")
    pubv = {(v["object"], int(v["idx"])): v for v in P["gv"]}
    corrv = {(v["object"], int(v["idx"])): v for v in P["cv"]}
    s2c = {(r["object"], int(r["idx"])): r for r in D["s2c"]}
    oc = {(r["object"], int(r["idx"])): r for r in D["ocrecs"]}
    vs = []
    for e in sorted(L["exs"], key=lambda r: (r["side"] != "test", gg._order(r["object"]), int(r["idx"]))):
        k = (e["object"], int(e["idx"]))
        v = case_view(e, tg[k])
        v["pub"] = {a: pubv[k][a] for a in PUB_ARMS} if k in pubv else None
        v["corr"] = {a: corrv[k][a] for a in CORR_ARMS} if k in corrv else None
        v["extra"] = {"oracle_c": episode(oc[k]["decisions"], oc[k]["sims"], k0) if k in oc and not oc[k]["dropped"] else None,
                      "r4_c": episode(s2c[k]["decisions"], s2c[k]["sims"], k4) if k in s2c and not s2c[k]["dropped"] else None}
        vs.append(v)
    La = [v for v in vs if v["dropped"] is None]
    Lt, Lv = [v for v in La if v["side"] == "test"], [v for v in La if v["side"] == "validation"]
    LC = [v for v in Lt if v["corr"] is not None]
    LX = [v for v in LC if v["extra"]["oracle_c"] is not None and v["extra"]["r4_c"] is not None]
    ran = [v for v in vs if v["a"] is not None]
    test = [v for v in vs if v["side"] == "test"]
    rt = test                       # the row tables: every labelled row of the test cases (a stopped case's rows that ran too)
    ct = [v for v in P["cv"] if v["side"] == "test"]
    f = funnel(vs)
    in8 = any((v["object"], int(v["idx"])) == PITCHER_8 for v in Lt)
    level_a = {"cases": len(Lt), "objects": len({v["object"] for v in Lt}), "rates": _rates_of(Lt, ARMS, lambda v: v["a"]),
               "headline": headline(Lt), "extras": extras(Lt, LC), "extras_cases": {"level_a": len(Lt), "with_corrected": len(LC)},
               "pitcher_8": {"in_the_set": in8, "headline": headline(Lt, without=PITCHER_8) if in8 else None},
               "dropped": dict(Counter(v["dropped"] for v in test if v["dropped"])),
               "validation": {"cases": len(Lv), "rates": _rates_of(Lv, ARMS, lambda v: v["a"])}}
    extra = {"cases": len(LX), "of_pilot_cases": len(vs), "exploratory": True,
             "n_per_column": {"level_a": len(Lt), "corrected": len(ct), "oracle_c": sum(v["extra"]["oracle_c"] is not None for v in test),
                              "r4_c": sum(v["extra"]["r4_c"] is not None for v in test)},
             "rates": dict(_rates_of(LX, LEVEL_A_ARMS, lambda v: v["a"]), **_rates_of(LX, CORR_ARMS, lambda v: v["corr"]),
                           **_rates_of(LX, EXTRA_ARMS, lambda v: v["extra"]))}
    side = D["side"]
    planned, g2f = f["all"]["45"]["planned"], sum(f["g2_failed"].values())
    sm = side["smoke"]["counts"]
    gates = {"provenance": gate, "tags": tgate, "published_gate": P["g_pub"], "corrected_gate": P["g_corr"],
             "tags_vs_pilot": tags_vs_pilot(L["tgs"], D["grecs"]), "counts_check": counts_check(f, side["counts"]),
             "g1": {k: side["g1"].get(k) for k in ("cases", "rows", "mismatches", "passed", "code_revisions")},
             "g1p": {n: {"passed": s.get("passed"), "rows": s.get("rows")} for n, s in side["g1p"].items()},
             "g3_smoke": {"replay_rows": sm.get("replay_rows"), "replay_mismatches": sm.get("replay_mismatches"),
                          "misses": (sm.get("free") or {}).get("miss")},
             "cache_misses": f["all"]["45"]["miss"],
             "g2": {"30": G2_30, "45": {"planned": planned, "failed": g2f, "share": g2f / planned if planned else None}},
             "curobo": {"chosen": f["all"]["45"]["chosen"], "failed": f["free_failed"], "by_reason": f["free_failed_by"]}}
    pt = P["pub"]["pilot"]["test"]
    readings = {"1_funnel": f,
                "2_published": {"cases": pt["cases"], "rates": pt["rates"], "gg_vs_c10": pt["gg_vs_c10"], "ggb_vs_c10": pt["ggb_vs_c10"]},
                "2_corrected": P["corr_test"], "2_level_a": level_a, "2_by_object": by_object_table(vs, La, P["gv"], P["cv"]),
                "2_extra": extra, "2_level_b": "not applicable at level A",
                "3_hammer": hammers(Lt), "4_objects": objects_line(Lt), "5_tags": tilted_tags(Lt),
                "6": "not applicable at level A", "7_gates": gates,
                "loss_phase": {"test": loss_phase_split(rt), "by_object": {o: loss_phase_split([v for v in rt if v["object"] == o])
                                                                           for o in dict.fromkeys(v["object"] for v in rt)}},
                "release_tilt": release_tilt(rt), "by_cause": by_cause(rt), "replan_d": replan_d_bins(rt),
                "ruling3": dict(ruling3(ran, {(r["object"], int(r["idx"])): r for r in P["cg"]}),
                                cases_2x2={"A": success_2x2(LC, "cap", "gg"), "B": success_2x2(LC, "cap_b", "ggb")})}
    return {"readings": readings, "counts": {"cases": len(vs), "executed": len(ran), "paired": {"test": len(Lt), "validation": len(Lv)},
                                             "dropped": dict(Counter(v["dropped"] for v in vs if v["dropped"]))}}


def summary_cmd(a: Any) -> dict[str, Any]:
    """The level-A summary. First the provenance gate (curve_sixdof_rerun.provenance_gate: every execute unit, its
    prepass unit and the plan step on the level-A code and sim) and the tags gate (tags_gate), before any input opens or
    any file is written; then every other input (_open_inputs), the G1′ readings' references (readings_refs: its
    passes, pilot and pose-set summaries and keys must be the ones read here) and the published and corrected gates (inputs_gate: a
    difference writes rerun_a_summary_stop.json and exits 3 before any level-A record opens); then the level-A records,
    once (_open_level_a); the readings (level_a_build); rerun_a_summary.{json,md} and rerun_a_ceiling.png in --out
    (refuses an existing summary); wandb."""
    from robots.mujoco_lab.curve import BOOT_DRAWS, BOOT_SEED, EPS
    from robots.mujoco_lab.estimation_check import _revision
    from robots.mujoco_lab.tracking import log_run

    ex_dirs, tg_dirs = a.execute.split(","), a.tags.split(",")
    for d in ex_dirs + [a.plan]:
        rr._finished(d)
        ep = os.path.join(d, "run.exit")
        if not (os.path.exists(ep) and open(ep).read().strip() == "EXIT 0"):
            raise SystemExit(f"{d}: not EXIT 0")
    gate = rr.provenance_gate(ex_dirs, a.plan)
    pre_dirs = list(dict.fromkeys(json.load(open(os.path.join(d, "run_config.json")))["prepass"] for d in ex_dirs))
    tgate = tags_gate(tg_dirs, pre_dirs)
    jpath, spath = os.path.join(a.out, "rerun_a_summary.json"), os.path.join(a.out, "rerun_a_summary_stop.json")
    if os.path.exists(jpath) or os.path.exists(spath):
        raise SystemExit(f"{a.out}: a summary exists: pick a new --out")
    os.makedirs(a.out, exist_ok=True)
    D = _open_inputs(a)
    bad = readings_refs(D["side"]["readings"], a, D["prov"]["inputs_sha256"], D["k0"], D["k4"])
    if bad:
        raise SystemExit("the G1′ readings summary read other inputs: " + "; ".join(bad) + " (no level-A record opened)")
    me = _revision()
    prov = dict(D["prov"], execute=a.execute, plan=a.plan, tags=a.tags, code_revision=me["code_revision"], dirty=me["dirty"],
                bootstrap=f"{BOOT_DRAWS} object draws, seed {BOOT_SEED}, 95 % percentile interval", eps=EPS)
    P = inputs_gate(D)
    if P["stop"]:
        with open(spath, "w") as f:
            json.dump({"published_gate": P["g_pub"], "corrected_gate": P["g_corr"], "provenance": prov}, f, indent=1)
        print(f"STOP: the published or the corrected numbers are not reproduced; no level-A record opened; wrote {spath}", flush=True)
        raise SystemExit(3)
    L = _open_level_a(ex_dirs, tg_dirs)
    prov["level_a_records_sha256"] = L["shas"]
    rep = level_a_build(D, P, L, gate, tgate)
    rep["provenance"] = prov
    json.dumps(rep)                                   # a value json cannot write fails here, before any file or wandb
    summary_markdown(rep)
    figure(rep, os.path.join(a.out, "rerun_a_ceiling.png"))
    la = rep["readings"]["2_level_a"]
    rep["wandb"] = log_run("sixdof-rerun-summary-a", {"execute": a.execute, "tags": a.tags, "out": a.out},
                           {"paired": rep["counts"]["paired"],
                            "headline": {k: {"success": x["success"], "irr": x["irr"]} for k, x in la["headline"].items()},
                            "rates": {k: {"success": x["success"], "irr": x["irr"], "give_up": x["give_up"]} for k, x in la["rates"].items()}})
    with open(jpath, "w") as f:
        json.dump(rep, f, indent=1)
    with open(os.path.join(a.out, "rerun_a_summary.md"), "w") as f:
        f.write(summary_markdown(rep))
    print(f"wrote {a.out}", flush=True)
    return rep


# ------------------------------------------------------------------ the markdown and the figure

def _pc(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def _ci(b: dict[str, Any] | None, k: str = "success") -> str:
    if not b or b.get(k) is None:
        return "–"
    x = b[k]
    return f"{100 * x['diff']:+.2f} [{100 * x['lo']:+.2f}, {100 * x['hi']:+.2f}]"


def _rate3(r: dict[str, Any] | None) -> str:
    return "–" if not r or r.get("n") == 0 else f"{_pc(r['success'])} / {_pc(r['irr'])} / {_pc(r['give_up'])}"


def summary_markdown(rep: dict[str, Any]) -> str:
    """rerun_a_summary.md: the gates, readings 1 to 7 with the published, the corrected and the level-A columns, the
    additions, the by-cause tables, ruling 3's check and the extras (numbers only; the README states them)."""
    R, c = rep["readings"], rep["counts"]
    la, pub, cor = R["2_level_a"], R["2_published"], R["2_corrected"]
    L = ["# Level A: the rerun's summary", "",
         f"Differences in points with the 95 % object bootstrap ({rep['provenance']['bootstrap']}). Rates: success / damage / "
         "give-up in %. Each column is on its own cases. Published: the pilot's numbers recomputed from its records (the gate). "
         "Corrected: the G1′ contacts + re-plan pass's arms. Level A: the rerun on the contacts sim with the re-plan on.", "",
         "## Reading 2: ceilings (test side)", "",
         "| arm | published | corrected | level A |", "|---|---|---|---|",
         f"| cases | {pub['cases']} | {cor['cases']} | {la['cases']} |",
         f"| GraspGen ≤ 30°, the pilot's capped rows, rule A | {_rate3(pub['rates']['gg'])} | {_rate3(cor['rates']['gg'])} | {_rate3(la['rates']['cap'])} |",
         f"| GraspGen ≤ 30°, the pilot's capped rows, rule B | {_rate3(pub['rates']['ggb'])} | {_rate3(cor['rates']['ggb'])} | {_rate3(la['rates']['cap_b'])} |"]
    L += [f"| {ARM_NAMES[k]} | – | – | {_rate3(la['rates'][k])} |" for k in LEVEL_A_ARMS]
    L += [f"| 10-pinch ceiling | {_rate3(pub['rates']['c10'])} | {_rate3(cor['rates']['c10'])} | – |",
          f"| 3-pinch ceiling | {_rate3(pub['rates']['c3'])} | {_rate3(cor['rates']['c3'])} | – |",
          f"| oracle arm | {_rate3(pub['rates']['oracle'])} | – | – |", f"| R4 | {_rate3(pub['rates']['r4'])} | – | – |", "",
          "| paired difference: success; damage | published | corrected | level A |", "|---|---|---|---|",
          f"| GraspGen rule A − 10-pinch | {_ci(pub['gg_vs_c10'])}; {_ci(pub['gg_vs_c10'], 'irr')} | {_ci(cor['gg_vs_c10'])}; {_ci(cor['gg_vs_c10'], 'irr')} | – |",
          f"| GraspGen rule B − 10-pinch | {_ci(pub['ggb_vs_c10'])}; {_ci(pub['ggb_vs_c10'], 'irr')} | {_ci(cor['ggb_vs_c10'])}; {_ci(cor['ggb_vs_c10'], 'irr')} | – |"]
    for r in ("A", "B"):
        h = la["headline"][r]
        L.append(f"| **headline** ≤ 45° − ≤ 30° with clearance, rule {r} | – | – | {_ci(h)}; {_ci(h, 'irr')}: **{h['reading']}** |")
    ex = la["extras"]
    L += [f"| extra: ≤ 45° − ≤ 30° without clearance, rule {r} | – | – | {_ci(ex[f'{r}_vs_30n'])} |" for r in ("A", "B")]
    L += [f"| extra: ≤ 45° − corrected 10-pinch, rule {r} (n {la['extras_cases']['with_corrected']}) | – | – | {_ci(ex[f'{r}_vs_c10'])} |"
          for r in ("A", "B")]
    p8 = la["pitcher_8"]
    if p8["in_the_set"]:
        L += [f"| exploratory: the headline without ycb__pitcher:8 (planned in pass b), rule {r} | – | – | {_ci(p8['headline'][r])} |"
              for r in ("A", "B")]
    x = R["2_extra"]
    L += ["", f"All angles (level B): {R['2_level_b']}. Level-A drops (test): {la['dropped'] or 'none'}. Validation cases "
          f"(lemon1, mug): {la['validation']['cases']}; " + ", ".join(f"{k} {_rate3(y)}" for k, y in la["validation"]["rates"].items()), "",
          f"Labelled extra, exploratory: the paired test set where no arm drops a case holds {x['cases']} of the pilot's "
          f"{x['of_pilot_cases']} cases (n per column on its own: {x['n_per_column']}). On it: "
          + ", ".join(f"{k} {_rate3(y)}" for k, y in x["rates"].items()), "",
          "| object (side) | n: level A / published / corrected | ≤ 30° uncapped, with clearance | ≤ 45° | ≤ 45°, rule B | pilot capped, "
          "published | pilot capped, corrected | 10-pinch, corrected |", "|---|---|---|---|---|---|---|---|"]
    for o, y in R["2_by_object"].items():
        a_, p_, c_ = y["level_a"], y["published"], y["corrected"]
        L.append(f"| {o} ({y['side']}) | {a_['n']} / {p_['n']} / {c_['n']} | {_pc(a_['a30']['success'])} | {_pc(a_['a45']['success'])} | "
                 f"{_pc(a_['a45_b']['success'])} | {_pc(p_['gg']['success'])} | {_pc(c_['gg']['success'])} | {_pc(c_['c10']['success'])} |")
    f = R["1_funnel"]
    L += ["", "## Reading 1: funnels", "", "| object | ≤ 30°: within, uncapped A / B, clearance failures, flagged rows run, with "
          "clearance A / B | 30–45°: within, width A / B, clearance ok, lab path ok, to the planner, planned, G2 passed, executed, "
          "rule A / B, distinct A / B |", "|---|---|---|"]
    for o, y in list(f["by_object"].items()) + [("all", f["all"])]:
        a_, b_ = y["30"], y["45"]
        L.append(f"| {o} | {a_['within']}, {a_['uncapped']} / {a_['uncapped_b']}, {a_['clearance_fail']}, {a_['flagged']}, {a_['a30']} / "
                 f"{a_['a30_b']} | {b_['within']}, {b_['width']} / {b_['width_b']}, {b_['clearance_ok']}, {b_['reach_ok']}, {b_['chosen']}, "
                 f"{b_['planned']}, {b_['g2_passed']}, {b_['labelled']}, {b_['rule_a']} / {b_['rule_b']}, {b_['distinct']} / {b_['distinct_b']} |")
    al = f["all"]
    L += ["", f"Generated {al['generated']}: within 30° {_pc(al['share_within_30'])} %, within 45° {_pc(al['share_within_45'])} %, past "
          f"every step {_pc(al['share_every_step'])} % (rule A) and {_pc(al['share_every_step_b'])} % (rule B). Clearance failures by first "
          f"pair: ≤ 30° {f['clearance_pairs']['30']}, 30–45° {f['clearance_pairs']['45']}. Disagreements (clearance vs the lab path "
          f"check): {al['45']['disagree']}. cuRobo failures {f['free_failed']} {f['free_failed_by']}; G2 failures {f['g2_failed']}.", ""]
    L += [f"- {o} first stops: rule A {w['a']}; rule B {w['b']}" for o, w in f["why6"].items()]
    h = R["3_hammer"]
    L += ["", "## Reading 3: hammers", "", f"Cases {h['cases']} ({', '.join(h['objects'])}). Classes: "
          + "; ".join(f"{k} { {c_: n for c_, n in y.items() if n} }" for k, y in h["classes"].items()), ""]
    for pc in h["per_case"]:
        if pc["placed"]:
            L.append(f"- {pc['object']} {pc['idx']}: " + "; ".join(
                f"{p['id']} {p['bin']} {p['approach_deg']:.1f}°, f {p['f']:.3f}, {100 * p['balance_m']:.1f} cm from the balance point, "
                f"{100 * p['head_m']:.1f} cm from the head end, {p['tag']}, {'/'.join(p['rules'])}" for p in pc["placed"]))
    L += ["", "## Reading 4: pitcher and bowl", ""]
    for o, y in R["4_objects"].items():
        for r in ("a", "b"):
            z = y[r]
            L.append(f"- {o}, rule {r.upper()}: {z['cases_with_tilted_row']} of {z['cases']} cases with an executed tilted row; "
                     f"{z['tilted_rows']} rows {z['tilted_tags']}; labels {z['tilted_labels']}; placing {z['tilted_placing']} "
                     f"{z['tilted_placing_tags']}; cases placed ≤ 45° {z['placed_45']}, ≤ 30° {z['placed_30']}")
    L += ["", "## Reading 5: where the tilted placing rows sit", ""]
    L += [f"- rule {r.upper()}: rows {y['rows']}; cases {y['cases']}; gain cases {y['gain']}; loss cases {y['loss']['cases']}; "
          f"≤ 30° placing rows {y['vertical_rows']}" for r, y in R["5_tags"].items()]
    g = R["7_gates"]
    pv = g["provenance"]
    L += ["", "## Reading 6", "", "Not applicable at level A.", "", "## Reading 7: gates", "",
          f"- Provenance: {pv['execute_units']} execute and {pv['prepass_units']} prepass units on code {pv['code']} and sim {pv['sim']}; "
          f"the plan on code {pv['plan']['code']} (its sim line {pv['plan']['sim']}, not checked: the plan reads no sim); "
          f"{g['tags']['tags_units']} tags units on code {g['tags']['code']} and sim {g['tags']['sim']}.",
          f"- Published gate: {'all equal' if not any(g['published_gate'].values()) else g['published_gate']}. Corrected gate: "
          f"{'equal to the G1′ write-up' if not g['corrected_gate'] else g['corrected_gate']}.",
          f"- The tags pass against the pilot's tags (≤ 30° rows): {g['tags_vs_pilot']['equal']} of {g['tags_vs_pilot']['compared']} equal, "
          f"{g['tags_vs_pilot']['differ']} differ, {g['tags_vs_pilot']['missing']} missing.",
          f"- The funnel against the gated counts report: {'equal' if not g['counts_check'] else g['counts_check']}.",
          f"- G1: {g['g1']}. G1′: {g['g1p']}. G3 smoke: {g['g3_smoke']}. Plan-cache misses: {g['cache_misses']}. G2: ≤ 30° "
          f"{g['g2']['30']}; 30–45° {g['g2']['45']['failed']} of {g['g2']['45']['planned']} planned. cuRobo: {g['curobo']}.", ""]
    lp = R["loss_phase"]["test"]
    L += ["## The loss-phase split (clean rows that lifted the object and did not place it)", "",
          f"- 30–45°: {lp['45']['rows']} rows, by phase {lp['45']['phases']}, by label {lp['45']['labels']}",
          f"- ≤ 30° (reference): {lp['30']['rows']} rows, by phase {lp['30']['phases']}, by label {lp['30']['labels']}", "",
          "## The planned tilt at the release (not measured)", "",
          f"The 30–45° rows executed to a label. {R['release_tilt']['what']}", ""]
    for o, s in list(R["release_tilt"]["by_object"].items()) + [("all", R["release_tilt"]["all"])]:
        L.append(f"- {o}: {s['rows']} rows, median {s['median']:.1f}°, {s['min']:.1f} to {s['max']:.1f}°" if s["rows"] else f"- {o}: 0 rows")
    L.append("- Exploratory, by outcome: " + "; ".join(f"{k} {s['rows']} (median {s['median']:.1f}°)" if s["rows"] else f"{k} 0"
                                                       for k, s in R["release_tilt"]["by_outcome"].items()))
    L += ["", "## By cause (rows that ran to a label: a box contact, a raise at the hold)", "", "| bin | cause | rows | placed | damage |",
          "|---|---|---|---|---|"]
    for b, y in R["by_cause"].items():
        L += [f"| {'≤ 30°' if b == '30' else '30–45°'} | {k} | {z['rows']} | {z['placed']} | {z['damage']} |" for k, z in y.items()]
    L += ["", "Re-plan at the hold (rows that rose and were held, by d): " + "; ".join(
        f"{'≤ 30°' if b == '30' else '30–45°'} {y['bins']}, no record {y['no_record']}, not lifted {y['not_lifted']}, lifted not held "
        f"{y['lifted_not_held']}" for b, y in R["replan_d"].items()), "", "## Ruling 3's check", "",
          f"- Rows: {R['ruling3']['rows']}; cases where a capped ceiling places and the uncapped one does not: "
          f"{R['ruling3']['capped_places_uncapped_not']}; capped with the rerun's labels vs corrected, success by case: {R['ruling3']['cases_2x2']}"]
    L += [f"  - {d['object']} {d['idx']} {d['row']}: corrected {d['corrected']}, rerun {d['rerun']} ({d['cause']})"
          for d in R["ruling3"]["differences"][:40]]
    L += ["", f"Cases {c['cases']}, executed {c['executed']}, paired {c['paired']}, dropped {c['dropped'] or 'none'}. wandb {rep.get('wandb')}.", ""]
    return "\n".join(L)


FIG_ARMS = (("published", "gg", "pilot ≤ 30°, capped\n(published)", "#2a78d6", "o"),
            ("corrected", "gg", "pilot ≤ 30°, capped\n(corrected)", "#2a78d6", "s"),
            ("level_a", "a30", "≤ 30°, uncapped,\nwith clearance\n(level A)", "#1b9e77", "o"),
            ("level_a", "a45", "≤ 45°\n(level A)", "#1b9e77", "D"),
            ("level_a", "a45_b", "≤ 45°, rule B\n(level A)", "#1b9e77", "d"),
            ("corrected", "c10", "10-pinch\n(corrected)", "#eb6834", "s"),
            ("published", "oracle", "oracle arm\n(published)", "#7a5195", "^"),
            ("published", "r4", "R4\n(published)", "#b8860b", "v"))


def figure(rep: dict[str, Any], path: str) -> None:
    """The ceiling by cumulative approach angle beside the pilot's capped arm, the 10-pinch ceiling, the oracle arm and
    R4 (design reading 2): success over each column's own test cases (a large marker, with n) and each test object's
    rate (small markers)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = rep["readings"]
    pooled = {"published": R["2_published"]["rates"], "corrected": R["2_corrected"]["rates"], "level_a": R["2_level_a"]["rates"]}
    by = [y for y in R["2_by_object"].values() if y["side"] == "test"]
    fig, ax = plt.subplots(figsize=(10.0, 4.4))
    labels = []
    for i, (col, k, name, color, marker) in enumerate(FIG_ARMS):
        pts = [y[col][k]["success"] for y in by if y[col]["n"] and y[col][k]["success"] is not None]
        ax.scatter([i + 0.12 * ((j % 5) - 2) / 2 for j in range(len(pts))], [100 * p for p in pts], s=16, color=color, alpha=0.45,
                   marker=marker, linewidths=0)
        r = pooled[col][k]
        labels.append(f"{name}\nn {r['n']}")
        if r["success"] is not None:
            ax.scatter([i], [100 * r["success"]], s=110, color=color, marker=marker, edgecolors="black", linewidths=0.8, zorder=3)
            ax.annotate(f"{100 * r['success']:.1f}", (i, 100 * r["success"]), xytext=(9, 0), textcoords="offset points", fontsize=8,
                        va="center")
    ax.set_xticks(range(len(FIG_ARMS)), labels, fontsize=7.5)
    ax.set_ylabel("success over the test cases (%)")
    ax.set_ylim(-3, 103)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Ceiling by cumulative approach angle; small markers: one test object each", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> Any:
    import argparse

    ap = argparse.ArgumentParser(prog="sixdof_rerun_summary", description=__doc__.split("\n")[0])
    ap.add_argument("--execute", required=True, help="comma-separated finished execute units (the level-A record)")
    ap.add_argument("--plan", required=True, help="their plan step")
    ap.add_argument("--tags", required=True, help="comma-separated finished tags units, one per prepass unit")
    ap.add_argument("--poseset-summary", required=True)
    ap.add_argument("--pilot-summary", required=True)
    ap.add_argument("--readings-summary", required=True, help="the G1′ readings' readings_summary.json (the corrected gate)")
    ap.add_argument("--contacts", required=True, help="the G1′ contacts pass's queue")
    ap.add_argument("--replan", required=True, help="the G1′ contacts + re-plan pass's queue (the corrected arms)")
    ap.add_argument("--stage2c-summary", required=True)
    ap.add_argument("--oracle-c-summary", required=True)
    ap.add_argument("--counts", required=True, help="the gated counts report's smoke_summary.json (full_a/counts_g)")
    ap.add_argument("--smoke", required=True, help="the level-A smoke's smoke_summary.json (G3)")
    ap.add_argument("--g1", required=True, help="G1's g1_summary.json")
    ap.add_argument("--out", required=True)
    return summary_cmd(ap.parse_args(argv))


if __name__ == "__main__":
    main()
