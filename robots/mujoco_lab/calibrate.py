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

"""End-to-end calibration check on the untouched test objects.

For each test object and repeat: draw hidden properties and a destination, reset the lab,
call ``assess_pick`` then ``pick_and_place`` through the skill tools (the default best row,
or abstain on a decline recommendation with ``--follow-recommendation``), and record the
declared outcome vector beside the outcome. Scores with ``evaluate.score`` against the
corpus' training class frequencies.

    python -m robots.mujoco_lab.calibrate --metrics data/success_model_metrics.json --repeats 10 --out data/m1_calibration.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from robots.mujoco_lab.corpus import draw_theta
from robots.mujoco_lab.robot_spec import build_lab
from rpent.skills.pick_place import menu as menu_mod
from rpent.skills.pick_place.destinations import DESTINATIONS, MESH_IDS, PRIMITIVE_IDS
from rpent.skills.pick_place.evaluate import rows_of, score
from rpent.skills.pick_place.features import extract
from rpent.skills.pick_place.protocol import OUTCOMES, Prediction
from rpent.skills.pick_place.tools import assess_pick, pick_and_place


def _load_corpus(path: str) -> list[dict[str, Any]]:
    rows = []
    for line in open(path):
        r = json.loads(line)
        if "error" not in r:
            rows.append(r)
    return rows


def run_case(prims, object_id: str, idx: int, seed: int, rng: np.random.Generator, follow: bool,
             use_fixtures: bool = False) -> dict[str, Any]:
    env = prims.env
    from robots.mujoco_lab.adapter import object_props

    props = object_props(object_id)
    theta, com_end = draw_theta(rng, props["mass"])
    if use_fixtures and MESH_IDS and rng.random() < 0.5:
        destination = str(rng.choice(sorted(MESH_IDS)))
    else:
        destination = str(rng.choice(sorted(PRIMITIVE_IDS)))
    env.reset(object_id, theta, seed=seed * 1000 + idx, com_end=com_end, fixture=destination if destination in MESH_IDS else None)
    store: dict[str, Any] = {}
    a = assess_pick(object="object", destination=destination, primitives=prims, store=store).data
    truth = env.truth()
    row = {"object": object_id, "destination": destination, "theta": theta, "truth": {k: truth[k] for k in ("mass", "mu", "com_s")},
           "recommendation": a["recommendation"], "best_option": a["best_option"], "executed": False, "outcome": None,
           "placed_at_requested": None, "option": None, "declared": None}
    if a["best_option"] is None or (follow and a["recommendation"] == "decline"):
        row["declared"] = {"p": None}
        return row
    opt = next(o for o in a["options"] if o["id"] == a["best_option"])
    g = env.geometry()                                   # before the move: the features describe the grasp as planned
    prop = next(p for p in env.proposals() if abs(p.s - opt["action"]["s"]) < 1e-9)
    feats = extract(g, np.asarray(prop.point), prop.yaw, truth["mass"], truth["mu"], opt["action"]["force"], opt["action"]["v_lift"],
                    opt["action"]["v_transport"], opt["action"]["destination"], opt["action"].get("release_height"))
    r = pick_and_place(object="object", destination=destination, option_id=a["best_option"], primitives=prims, store=store).data
    if "outcome" not in r:
        row["error"] = r.get("error") or r.get("reason")
        return row
    row.update({"executed": True, "outcome": r["outcome"], "placed_at_requested": r["placed_at_requested"], "option": r["option"],
                "declared": {"p": opt["p"], "p_drop_height": opt["p_drop_height"], "expected_cost": opt["expected_cost"]},
                "label": r["outcome"], "features": feats})
    return row


def ece_bootstrap_ci(preds: list[Prediction], rows: list[dict[str, Any]], rng: np.random.Generator, n_boot: int = 300) -> list[float]:
    """95 % interval of the ECE on P(placed_stable) by resampling objects (the unit of generalisation)."""
    from rpent.skills.pick_place.evaluate import _ece

    objects = sorted({r["object"] for r in rows})
    by_obj = {o: [i for i, r in enumerate(rows) if r["object"] == o] for o in objects}
    p = np.array([x.p["placed_stable"] for x in preds])
    y = np.array([1.0 if r["label"] == "placed_stable" else 0.0 for r in rows])
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(objects, size=len(objects), replace=True)
        idx = np.concatenate([by_obj[o] for o in pick])
        vals.append(_ece(p[idx], y[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="success_model_metrics.json (gives the split and the corpus path)")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--objects", default=None, help="comma-separated ids (default: the metrics' test split)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--follow-recommendation", action="store_true")
    ap.add_argument("--threshold", type=float, default=None, help="override the commit threshold (tests)")
    ap.add_argument("--model-kind", default=None)
    ap.add_argument("--fixtures", action="store_true", help="draw half the destinations from the inspected mesh fixtures")
    a = ap.parse_args(argv)
    metrics = json.load(open(a.metrics))
    objects = a.objects.split(",") if a.objects else list(metrics["split"]["test"])
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    paths = metrics["corpus"] if isinstance(metrics["corpus"], list) else [metrics["corpus"]]
    train_rows = []
    for cp in paths:
        cp = cp if os.path.isabs(cp) else os.path.join(root, cp)
        if os.path.exists(cp):
            train_rows += rows_of(_load_corpus(cp), metrics["split"]["train"])
    if a.threshold is not None:
        menu_mod.DEFAULT_THRESHOLD = a.threshold
    prims = build_lab(seed=a.seed, model_kind=a.model_kind)
    rng = np.random.default_rng(a.seed)
    rows = []
    for oid in objects:
        for k in range(a.repeats):
            try:
                rows.append(run_case(prims, oid, k, a.seed, rng, a.follow_recommendation, a.fixtures))
            except Exception as exc:  # noqa: BLE001
                rows.append({"object": oid, "error": str(exc)[:200], "executed": False})
            print(f"{oid} {k}: {rows[-1].get('outcome') or rows[-1].get('error') or 'abstained'}", file=sys.stderr, flush=True)
    done = [r for r in rows if r.get("executed")]
    preds = [Prediction(p={k: float(r["declared"]["p"][k]) for k in OUTCOMES}) for r in done]
    sc = score(preds, done, train_rows) if done else {"brier": None, "brier_const": None, "ece": None, "by_subgroup": {}}
    ece_ci = ece_bootstrap_ci(preds, done, rng) if done else None
    by_dest = {}
    by_opt = defaultdict(lambda: {"n": 0, "placed": 0})
    for r in done:
        by_dest.setdefault(r["destination"], Counter())[r["outcome"]] += 1
        by_opt[r["option"]]["n"] += 1
        by_opt[r["option"]]["placed"] += r["outcome"] == "placed_stable"
    out = {
        "kind": getattr(prims.model, "kind", "classifier"), "objects": objects, "n": len(rows), "n_executed": len(done),
        "recommendation_decline_rate": float(np.mean([r.get("recommendation") == "decline" for r in rows if "recommendation" in r])) if rows else None,
        "abstained": sum(1 for r in rows if not r.get("executed") and "error" not in r),
        "errors": sum(1 for r in rows if "error" in r),
        "ece": sc["ece"], "ece_ci95_object_bootstrap": ece_ci, "brier": sc["brier"], "brier_const": sc["brier_const"],
        "by_subgroup": sc.get("by_subgroup", {}),
        "by_destination": {d: dict(c) for d, c in by_dest.items()}, "by_option": dict(by_opt),
        "outcomes": dict(Counter(r["outcome"] for r in done)), "rows": rows,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("kind", "n", "n_executed", "abstained", "errors", "ece", "ece_ci95_object_bootstrap", "brier", "brier_const", "outcomes")}, indent=1))
    return out


if __name__ == "__main__":
    main()
