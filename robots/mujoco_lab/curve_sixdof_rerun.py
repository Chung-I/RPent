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
"""The level-A rerun driver (design 2026-11-08-sixdof-rerun-design.md, unit (f)): the pilot's cases with the 30–45°
rows added. The funnel of those rows (funnel6): the width rules on the pinch's own closing axis, the kinematic
clearance (ruling 1), the configuration and path check (choose), then the cached cuRobo free move and execute6."""

from __future__ import annotations

from typing import Any

import numpy as np

from robots.mujoco_lab import sixdof

LEVEL_A_BIN = sixdof.BINS[1]          # "30–45°": the rows level A adds; the pilot's vertical rows stay its own


def _path_reason(f: dict[str, Any]) -> str:
    hit = f.get("hit")
    return f"path: {f.get('reason')} ({f.get('phase')}{', ' + '/'.join(map(str, hit[:2])) if hit else ''})"


def funnel6(env: Any, grasps: Any, conf: Any, destination: str, case: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """The level-A funnel over one case's grasps, by confidence (a stable sort): the rows in the 30–45° bin, each with
    both width rules (width_rule6: ``why_width``, ``why_width_b``), and, where one passes, the clearance (its first pair
    and all pairs) and choose()'s candidate (index, twin, seed, q) or None with the failed candidates. ``why`` /
    ``why_b``: the first step each rule's row stops at (None: the row goes to the planner). Counts: generated,
    level_a, chosen."""
    G, S = np.asarray(grasps, dtype=float), np.asarray(conf, dtype=float)
    rows: list[dict[str, Any]] = []
    for rank, k in enumerate(np.argsort(-S, kind="stable")):
        p = sixdof.from_graspgen(G[k])
        deg = float(sixdof.approach_deg(p))
        if sixdof.angle_bin(deg) != LEVEL_A_BIN:
            continue
        rid = f"g{int(k)}"
        wr = sixdof.width_rule6(env, p)
        why, why_b = wr["why"], wr["why_b"]
        r: dict[str, Any] = {"id": rid, "rank": rank, "conf": float(S[k]), "approach_deg": deg, "bin": LEVEL_A_BIN,
                             "width": float(wr["width"]), "why_width": why, "why_width_b": why_b, "clearance": None,
                             "candidate": None, "failed": []}
        if why is None or why_b is None:
            p = sixdof.from_graspgen(G[k], width=float(wr["width"]))
            c = sixdof.clearance(env, p, float(wr["width"]), destination)
            if c is not None:
                r["clearance"] = {"first": c["first"], "pairs": sorted(c["pairs"])}
                stop = f"clearance: {c['first']['pair']}"
            else:
                ch = sixdof.choose(env, p, sixdof.row_key(case, rid), destination)
                r["failed"] = ch["failed"]
                cand = ch["candidate"]
                if cand is None:
                    stop = _path_reason(ch["failed"][0]) if ch["failed"] else "path: no candidate"
                else:
                    r["candidate"] = {"index": int(cand["index"]), "twin": bool(cand["twin"]), "seed": cand["seed"],
                                      "q": [float(x) for x in cand["q"]]}
                    stop = None
            why, why_b = why or stop, why_b or stop
        r.update(why=why, why_b=why_b)
        rows.append(r)
    counts = {"generated": int(len(S)), "level_a": len(rows), "chosen": sum(r["candidate"] is not None for r in rows)}
    return rows, counts
