"""The post-hoc neutral-task-text check of the E1 pilot (stated in the daily-logs README before the run).

The neutral menu_planner and rec_planner rows pair with the as-run default_rule, oracle_rule and nomenu_planner rows on
(object, idx, setting). First the pairing check (truth, reset and the first assessment equal), then the same contrasts,
drop rule and object bootstrap as e1.summarize, plus neutral - v1 per planner and rec_planner - default_rule.

usage: python analyze_neutral.py AS_RUN_DIR NEUTRAL_DIR OUT_DIR
"""
from __future__ import annotations

import json
import os
import sys
import zlib

from robots.mujoco_lab import e1

FIRST = ("truth", "reset", "recommendation", "best_option", "guard", "first_menu", "destination", "instruction")


def contrast(a_rows, b_rows, name, settings, n_boot=2000):
    ka = {(r["object"], r["idx"], r["setting"]): r for r in a_rows}
    kb = {(r["object"], r["idx"], r["setting"]): r for r in b_rows}
    out = {}
    for s in settings + ["all"]:
        d, objs, dropped = [], [], 0
        for (o, i, st), r in ka.items():
            if s != "all" and st != s:
                continue
            q = kb.get((o, i, st))
            if q is None:
                continue
            if r["ended"] == "error" or q["ended"] == "error" or r.get("artifact") or q.get("artifact"):
                dropped += 1
                continue
            d.append(r["cost"] - q["cost"])
            objs.append(o)
        if d:
            mean, lo, hi = e1.object_bootstrap(d, objs, n_boot=n_boot, seed=zlib.crc32(f"{name}/{s}".encode()))
            out[s] = {"n": len(d), "objects": len(set(objs)), "mean": mean, "ci": [lo, hi], "dropped": dropped}
    return out


def main(as_run_dir, neutral_dir, out_dir):
    v1 = e1.load_rows(os.path.join(as_run_dir, "rows.jsonl"))
    nt = e1.load_rows(os.path.join(neutral_dir, "rows.jsonl"))
    key = {(r["object"], r["idx"], r["setting"], r["arm"]): r for r in v1}
    mismatches, missing = [], []
    for r in nt:
        q = key.get((r["object"], r["idx"], r["setting"], r["arm"]))
        if q is None:
            missing.append((r["object"], r["idx"], r["setting"], r["arm"]))
            continue
        bad = [f for f in FIRST if r.get(f) != q.get(f)]
        if bad:
            mismatches.append({"case": [r["object"], r["idx"], r["setting"], r["arm"]], "fields": bad})
    assert all(r.get("task") == "neutral" for r in nt), "a neutral row without task neutral"
    combined = [r for r in v1 if r["arm"] in ("default_rule", "oracle_rule", "nomenu_planner")] + nt
    summ = e1.summarize(combined)
    settings = summ["settings"]
    by = lambda rows, arm: [r for r in rows if r["arm"] == arm]
    extra = {"rec_planner - default_rule": contrast(by(nt, "rec_planner"), by(v1, "default_rule"), "rec_planner - default_rule", settings),
             "menu_planner neutral - v1": contrast(by(nt, "menu_planner"), by(v1, "menu_planner"), "menu_planner neutral - v1", settings),
             "rec_planner neutral - v1": contrast(by(nt, "rec_planner"), by(v1, "rec_planner"), "rec_planner neutral - v1", settings)}
    summ["contrasts"].update(extra)
    summ["pairing"] = {"neutral_rows": len(nt), "missing_as_run": missing, "first_assessment_mismatches": mismatches}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    md = e1.markdown(summ, "E1 pilot post-hoc check: neutral task text (menu and rec planners neutral; rules and nomenu as run)")
    md += (f"\nPairing: {len(nt)} neutral rows, {len(missing)} without an as-run row, {len(mismatches)} with a different "
           f"truth, reset or first assessment.\n")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(md)
    with open(os.path.join(out_dir, "examples.md"), "w") as f:
        f.write(e1.examples(combined))
    print(md)


if __name__ == "__main__":
    main(*sys.argv[1:4])
