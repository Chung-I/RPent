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

"""The pose-set ablation, stage A (daily-logs design 2026-11-08-pose-set-ablation-design.md, amended in its section 3.1).

Ten pose rows at commit_std's force and speeds: pinch fractions 0.1 to 0.9 of the long axis, the pads closing across
the axis (as Stage 2) or along it (the hand turned by -pi/2; on an upright object the other horizontal side). The three
current rows (across at 0.3, 0.5 and 0.7) keep the menu's rule (the width rule and reach()); the seven new rows also
need both ends of the pinch between the open pads, in the pad slab and in the column above it (the centring rule).
Each feasible row runs once at the case's own reset seed (the attempt is deterministic per case: the reset seed sets
only the sensor-noise streams). A case is achievable when any row places it: the capability ceiling, an upper bound
with a perfect selector, over the ten rows and over the three current rows (the matched baseline).
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

from rpent.skills.pick_place.protocol import IRREVERSIBLE

FRACS = (0.1, 0.3, 0.5, 0.7, 0.9)
CURRENT = (0.3, 0.5, 0.7)                  # the adapter's default proposals: the Stage 2 menu's pinches
TURN = {"across": 0.0, "along": -math.pi / 2}
# the selector's order: across first, then nearest the middle, the lower fraction first (rounded: abs(0.7 - 0.5) is
# 0.19999999999999996 in floats, so the menu's mid_proposal takes 0.7 before 0.3 when 0.5 is unusable)
ROW_ORDER = tuple((d, f) for d in TURN for f in sorted(FRACS, key=lambda f: (round(abs(f - 0.5), 9), f)))
UPRIGHT_DEG = 45.0                         # an axis tilted more than this from the table is upright
OFF_CENTRE = "an end of the pinch lies past an open pad"
OFF_CENTRE_COLUMN = "the object reaches past an open pad above the pinch"
KINDS = ("hammers and tools", "boxes", "cans and jars", "bottles", "fruit", "bowls and containers", "other")
_KINDS = {
    "hammers and tools": "handal__hammer handal__hammer_1 handal__hammer_2 handal__hammer_3 handal__hammer_4 handal__hammer_5 "
                         "handal__hammer_6 handal__hammer_7 handal__hammer_8 handal__ladle handal__measuring_spoon "
                         "handal__pink_spaghetti_spoon handal__red_serving_spoon handal__green_serving_spoon "
                         "handal__measuring_cups handal__measuring_cups_1 hot3d__spatula hot3d__wooden_spoons "
                         "hot3d__potato_masher ycb__clamp ycb__spring_clamp ycb__dry_erase_marker ycb__scissors "
                         "ycb__cordless_drill",
    "boxes": "hope__butter hope__cream_cheese hope__macaroni_and_cheese hope__milk_carton hope__oatmeal_raisin_cookies "
             "hope__orange_juice_carton hope__popcorn_box hope__raisin_box hot3d__frozen_waffles ycb__jello ycb__sugar_box "
             "hope__chocolate_pudding_mix hope__granola_bars hope__spaghetti hot3d__frozen_vegetable_block ycb__cheez_it "
             "ycb__chocolate_pudding hot3d__storage_box",
    "cans and jars": "hope__canned_mushrooms hope__canned_peaches hope__parmesan_cheese_canister hope__pineapple_slices_can "
                     "hope__tomato_sauce_can ycb__spam_can ycb__tuna_can hope__canned_tuna hope__corn_can "
                     "hope__green_beans_can hope__peas_and_carrots hope__pitted_cherries hot3d__soup_can ycb__coffee_can "
                     "ycb__tomato_soup_can",
    "bottles": "hope__mayonnaise_bottle ycb__soft_scrub hope__bbq_sauce_bottle hope__ketchup_bottle hope__mustard_bottle "
               "hope__ranch_dressing hot3d__bbq_sauce_bottle ycb__mustard",
    "fruit": "fruits_veggies__avocado fruits_veggies__orange1 fruits_veggies__lemon1 fruits_veggies__lime "
             "fruits_veggies__red_onion ycb__banana",
    "bowls and containers": "hot3d__ceramic_mug hot3d__coffee_pot hot3d__pitcher ycb__bowl ycb__bowl2 ycb__pitcher hot3d__mug "
                            "hot3d__wooden_bowl ycb__mug hope__yogurt_cup hot3d__clay_plates",
    "other": "hot3d__birdhouse hot3d__foam_roller hot3d__glasses hot3d__lizard_figurine hot3d__rubiks_cube hot3d__smartphone "
             "hot3d__computer_mouse hot3d__dumbbell hot3d__keyboard hot3d__megaphone hot3d__remote_control ycb__brick "
             "ycb__wood_block",
}
KIND_OF = {o: k for k, names in _KINDS.items() for o in names.split()}   # design reading 4's kinds, fixed before any record


def row_id(direction: str, f: float) -> str:
    return f"{direction}_{f:g}"


def centred(extent: Any, limit: float) -> bool:
    """Both ends of a signed extent (lo, hi) within +-limit of the pinch point (nan: no)."""
    lo, hi = float(extent[0]), float(extent[1])
    return bool(np.isfinite(lo) and np.isfinite(hi) and lo >= -limit and hi <= limit)


def pose_rows(env: Any, destination: str) -> list[dict[str, Any]]:
    """The ten pose rows in ROW_ORDER on the env's current state: the proposal, its width, its signed extent in the pad
    slab and in the column above it, and whether it is feasible. A current row (across at 0.3, 0.5, 0.7) needs the
    width rule and reach(), as the menu; a new row also needs the centring rule (both extents within half the widest
    pinch: the open pads' inner faces less the width rule's margin). reach() runs only on a row that passes the rest."""
    from rpent.skills.pick_place.menu import unusable_reason

    wmax = float(env.max_pinch_width)
    limit = wmax / 2
    by = {d: {p.s: p for p in env.proposals(FRACS, turn=t)} for d, t in TURN.items()}
    out = []
    for d, f in ROW_ORDER:
        p = by[d][f]
        current = d == "across" and f in CURRENT
        slab, column = env.pinch_extent(p.point, p.yaw), env.column_extent(p.point, p.yaw)
        cen_slab, cen_col = centred(slab, limit), centred(column, limit)
        why = unusable_reason(p.width, wmax)
        if why is None and not current and not (cen_slab and cen_col):
            why = OFF_CENTRE if not cen_slab else OFF_CENTRE_COLUMN
        reach = None
        if why is None:
            r = env.reach(p, destination)
            reach = "ok" if r is None else f"{r.get('reason')} ({r.get('phase')})"
            if r is not None:
                why = f"unreachable: {reach}"
        out.append({"id": row_id(d, f), "dir": d, "f": float(f), "current": current, "point": [float(x) for x in p.point],
                    "yaw": float(p.yaw), "width": float(p.width), "slab": [float(x) for x in slab],
                    "column": [float(x) for x in column], "centred": cen_slab and cen_col, "reach": reach,
                    "feasible": why is None, "why": why})
    return out


def _sha_of(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:16]


def simulate_rows(env: Any, reset: Any, rows: list[dict[str, Any]], base: dict[str, Any], fingerprint: Any,
                  want: str) -> str | None:
    """Run each feasible row once, in place: reset() (the case's own seed), the reset's fingerprint must be ``want``
    (else the case stops: the reason is returned), the proposal fetched again after the reset, then env.execute at
    ``base`` (commit_std's force, speeds, destination and release height). Each simulated row gets its label, the
    artifact flag, the damage and loss fields, hashes of the final state and of the trace, and its seconds; an
    exception is the row's error."""
    for r in rows:
        if not r["feasible"]:
            continue
        reset()
        if fingerprint(env) != want:
            return f"{r['id']}: the reset's fingerprint differs from Stage 1's"
        t0 = time.time()
        r.update(label=None, artifact=False, error=None)
        try:
            p = {q.s: q for q in env.proposals(FRACS, turn=TURN[r["dir"]])}[r["f"]]
            if [float(x) for x in p.point] != r["point"] or float(p.yaw) != r["yaw"]:
                return f"{r['id']}: the pinch after the reset differs from the first reset's"
            ro = env.execute(p, base["force"], base["v_lift"], base["v_transport"], base["destination"], base["release_height"],
                             a_transport=base["a_transport"])
            d = getattr(getattr(env, "sim", None), "d", None)
            r.update(label=ro.label, artifact=bool(getattr(ro, "artifact", False)), damage=getattr(ro, "damage", None),
                     drop_z=getattr(ro, "drop_z", None), loss_phase=getattr(ro, "loss_phase", None),
                     qpos_sha=None if d is None else hashlib.sha256(np.round(np.asarray(d.qpos, dtype=float), 9).tobytes()).hexdigest()[:16],
                     trace_sha=_sha_of([[t.get("phase"), round(float(t.get("z", 0.0)), 9)] for t in getattr(ro, "trace", [])]))
        except Exception as exc:  # noqa: BLE001  a simulator fault is this row's error
            r["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        r["seconds"] = round(time.time() - t0, 2)
    return None


def ceiling(rows: list[dict[str, Any]], current_only: bool = False) -> dict[str, Any]:
    """The capability ceiling over the feasible rows (the current ones only with ``current_only``), in episode form
    (curve.episode's keys): success when a clean row (no artifact, no error) placed the object; the selected row is the
    first clean success in ROW_ORDER, else the first clean row without an irreversible label, else the first clean row;
    no feasible row is a give-up. ``determined``: a clean row succeeds, or every row is clean."""
    cand = [r for r in rows if r["feasible"] and (r["current"] or not current_only)]
    out: dict[str, Any] = {"success": False, "irr": False, "give_up": not cand, "attempts": 0, "labels": [], "rows": [],
                           "n_rows": len(cand), "n_success": 0, "determined": True}
    if not cand:
        return out
    clean = [r for r in cand if r.get("label") is not None and not r.get("artifact") and not r.get("error")]
    succ = [r for r in clean if r["label"] == "placed_stable"]
    sel = succ[0] if succ else next((r for r in clean if r["label"] not in IRREVERSIBLE), clean[0] if clean else None)
    out.update(success=bool(succ), n_success=len(succ), determined=bool(succ) or len(clean) == len(cand))
    if sel is not None:
        out.update(attempts=1, labels=[sel["label"]], rows=[sel["id"]], irr=sel["label"] in IRREVERSIBLE)
    return out
