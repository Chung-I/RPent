"""E1 oracle-gap check, follow-up diagnostics (post-hoc, no new runs): kappa under rec20, the pairs whose cost differs, the
default rule's irreversible events, and, at high care under absorbing, the cases the oracle declines and the default runs
(their realized share of default - oracle against the model's expected share: the default's first row's declared outcome
probabilities priced under the true setting, minus A). Writes DIR/diagnostics.json.

    python -P logs/e1/gap_diagnostics.py logs/e1/gap
"""

import json
import os
import sys

from robots.mujoco_lab import e1, e1_gap
from robots.mujoco_lab.calibrate import attempt_cost
from rpent.skills.pick_place.protocol import IRREVERSIBLE

d = sys.argv[1]
rows = e1.load_rows(os.path.join(d, "rows.jsonl"))
G = e1_gap.grid()
prs = e1_gap.pairs(rows, G)
runs = {(r["object"], r["idx"], r["role"], r["setting"] if r["role"] == "oracle" else r["recovery"]): r for r in rows}
oracle = {(r["object"], r["idx"], r["setting"]): r for r in rows if r["role"] == "oracle"}
out = {"rec20_kappa_differences": {}, "pairs": {}, "default_events": [], "caution": {}}
for A in (3, 10, 30):
    names = [f"k{k}_A{A}_rec20" for k in (1, 3, 30, 100)]
    out["rec20_kappa_differences"][f"A{A}"] = sum(not e1_gap.same_record(oracle[(o, i, names[0])], oracle[(o, i, n)])
                                                  for (o, i, s) in oracle if s == names[0] for n in names[1:])
for s in G:
    g = [p["default"]["total"] - p["oracle"]["total"] for p in prs if p["setting"] == s and p["kept"]]
    out["pairs"][s] = {"kept": len(g), "differ": sum(abs(v) > 1e-9 for v in g), "oracle_better": sum(v > 1e-9 for v in g),
                       "oracle_worse": sum(v < -1e-9 for v in g)}
out["default_events"] = [{"object": r["object"], "idx": r["idx"], "recovery": r["recovery"], "outcomes": [a["outcome"] for a in r["attempts"]],
                          "artifact": r["artifact"]} for r in rows if r["role"] == "default" and e1.irreversible_events(r)]
for s, st in G.items():
    if st["recovery"] != "absorbing" or st["care_scale"] < 30:
        continue
    k, A = st["care_scale"], st["abstain_cost"]
    kept = [p for p in prs if p["setting"] == s and p["kept"]]
    m, real, model, events = 0, 0.0, 0.0, 0.0
    for p in kept:
        dr, orr = runs[(p["object"], p["idx"], "default", "absorbing")], runs[(p["object"], p["idx"], "oracle", s)]
        if orr["ended"] == "abstained" and not orr["attempts"] and not orr["probes"] and dr["attempts"]:
            a0 = dr["attempts"][0]
            P = a0["declared"]["p"]
            m += 1
            real += p["default"]["total"] - p["oracle"]["total"]
            model += sum(P[x] * attempt_cost(x, a0["cost_s"], k, None, "time") for x in P) + A * (1 - P["placed_stable"]) - A
            events += sum(P[x] for x in IRREVERSIBLE if x in P)
    out["caution"][s] = {"cases": m, "kept": len(kept), "realized_share": real / len(kept), "model_share": model / len(kept),
                         "expected_events": events}
with open(os.path.join(d, "diagnostics.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out["rec20_kappa_differences"]), json.dumps(out["default_events"]))
for s, c in out["caution"].items():
    print(f"{s}: {c['cases']} cases; realized {c['realized_share']:+.2f}, model {c['model_share']:+.2f}, expected events {c['expected_events']:.2f}")
