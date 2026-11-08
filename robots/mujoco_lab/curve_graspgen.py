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

"""The GraspGen menu pilot (daily-logs design 2026-11-08-graspgen-menu-pilot-design.md).

GraspGenX (the franka_panda gripper) proposes 6-DoF grasps on a cloud of 2,048 points on the collision hulls at the
case's rest pose (world frame, full view). The executability funnel maps them to the lab's vertical pinches: the
approach within 30 degrees of straight down; the pad centre 0.1034 m along the approach from the grasp frame's origin
(the palm: the gripper's config.json "fingertip" and the centre of its sweep volume), the lab point 4 mm below it (the
lab's pad centre sits 4 mm above its point), the yaw from the closing axis projected on the table; the width rule, the
pose set's centring rule and reach(); duplicates within 1 cm and 10 degrees dropped; up to 20 rows by confidence. The
variant rule (stated beside the design's) measures the width between the open pads only and needs the open pads'
columns free of the object: it admits a rim pinch, whose slab the design's width rule reads across the whole bowl. Each
kept row runs once at commit_std on the case's own reset seed; the capability ceiling as in the pose set.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any

import numpy as np

from rpent.skills.pick_place.features import PAD_HALF

N_POINTS = 2048
DEPTH = 0.1034                 # franka_panda: the pad centre along +Z from the grasp frame's origin (config.json)
PANDA_BOX = (0.04, 0.009, 0.009)   # its open finger volume's half-sizes: 8 cm aperture, 1.8 cm fingers, 1.8 cm pads
PAD_ABOVE = 0.004              # the lab's pad centre sits this far above its pinch point (grasp.hand_start_z)
VERTICAL_DEG = 30.0
DEDUP_M, DEDUP_DEG = 0.01, 10.0
KEEP_ROWS = 20
OPEN_GAP = 0.047               # the lab pads' inner faces at the open end, from the hand's centre (MAX_OPEN less 8 mm)
PAD_LANDS = "an open pad lands on the object"


def cloud_seed(o: str, idx: int) -> int:
    return int(hashlib.sha256(f"{o}:{int(idx)}:cloud".encode()).hexdigest()[:8], 16)


def surface_cloud(hulls: list[np.ndarray], n: int = N_POINTS, seed: int = 0) -> np.ndarray:
    """``n`` points on the surface of the union of the convex hulls (world frame, float32): uniform by area over every
    hull's faces, less the points more than 0.1 mm inside another hull (a face shared inside the object)."""
    from scipy.spatial import ConvexHull, QhullError

    rng = np.random.default_rng(int(seed))
    H = []
    for V in hulls:
        V = np.asarray(V, dtype=float)
        try:
            H.append((V, ConvexHull(V)))
        except QhullError:                                 # a flat hull: joggle it
            H.append((V, ConvexHull(V, qhull_options="QJ")))
    T = np.concatenate([V[h.simplices] for V, h in H])
    owner = np.concatenate([np.full(len(h.simplices), k) for k, (_, h) in enumerate(H)])
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    out, have = [], 0
    while have < n:
        k = rng.choice(len(T), size=2 * n, p=area / area.sum())
        u, v = rng.random(2 * n), rng.random(2 * n)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        P = T[k, 0] + u[:, None] * (T[k, 1] - T[k, 0]) + v[:, None] * (T[k, 2] - T[k, 0])
        inside = np.zeros(len(P), dtype=bool)
        for j, (_, h) in enumerate(H):
            inside |= ((P @ h.equations[:, :3].T + h.equations[:, 3]) < -1e-4).all(1) & (owner[k] != j)
        out.append(P[~inside])
        have += int((~inside).sum())
    return np.concatenate(out)[:n].astype(np.float32)


def approach_deg(g: Any) -> float:
    """The angle of the grasp's approach (+Z) from straight down, in degrees."""
    return float(np.degrees(np.arccos(np.clip(-float(np.asarray(g)[2, 2]), -1.0, 1.0))))


def map_grasp(g: Any, depth: float = DEPTH) -> dict[str, Any]:
    """A grasp's lab pinch: the pad centre ``depth`` along the approach from the frame's origin, the lab point
    PAD_ABOVE below it, the yaw of the closing axis (+X) projected on the table, and the approach angle."""
    g = np.asarray(g, dtype=float)
    pad = g[:3, 3] + depth * g[:3, 2]
    return {"pad": [float(x) for x in pad], "point": [float(x) for x in pad - np.array([0.0, 0.0, PAD_ABOVE])],
            "yaw": float(math.atan2(g[1, 0], g[0, 0])), "approach_deg": approach_deg(g),
            "closing_xy": float(math.hypot(g[0, 0], g[1, 0]))}


def yaw_gap_deg(a: float, b: float) -> float:
    """The angle between two pinch yaws, a half turn being the same pinch (the pads are symmetric), in degrees."""
    d = (float(a) - float(b)) % math.pi
    return float(math.degrees(min(d, math.pi - d)))


def box_extent(hulls: list[np.ndarray], centre: Any, ex: Any, ey: Any, half: tuple[float, float, float]) -> tuple[float, float]:
    """(lo, hi): the least and the greatest coordinate along ``ex``, from ``centre``, of the convex hulls inside the box
    |ex.(p - c)| <= half[0], |ey.(p - c)| <= half[1], |ez.(p - c)| <= half[2] (ez = ex x ey; half[0] may be inf);
    (nan, nan) when no hull meets the box. Exact: two linear programs per hull over its faces and the box's planes."""
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull, QhullError

    c = np.asarray(centre, dtype=float)
    ex, ey = np.asarray(ex, dtype=float), np.asarray(ey, dtype=float)
    ez = np.cross(ex, ey)
    box_A, box_b = [], []
    for axis, h in zip((ex, ey, ez), half):
        if np.isfinite(h):
            box_A += [axis, -axis]
            box_b += [h + axis @ c, h - axis @ c]
    lo, hi = math.inf, -math.inf
    for V in hulls:
        V = np.asarray(V, dtype=float)
        try:
            eq = ConvexHull(V).equations
        except QhullError:
            eq = ConvexHull(V, qhull_options="QJ").equations
        A = np.vstack([eq[:, :3], np.asarray(box_A)]) if box_A else eq[:, :3]
        b = np.concatenate([-eq[:, 3], np.asarray(box_b)]) if box_b else -eq[:, 3]
        bounds = [(None, None)] * 3
        r1 = linprog(ex, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        if r1.status != 0:
            continue
        r2 = linprog(-ex, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        lo, hi = min(lo, float(r1.fun - ex @ c)), max(hi, float(-r2.fun - ex @ c))
    return (lo, hi) if np.isfinite(lo) else (math.nan, math.nan)


def _pinch_axes(yaw: float) -> tuple[np.ndarray, np.ndarray]:
    return np.array([math.cos(yaw), math.sin(yaw), 0.0]), np.array([-math.sin(yaw), math.cos(yaw), 0.0])


def bounded_width(hulls: list[np.ndarray], point: Any, yaw: float, pad_half: Any = PAD_HALF, gap: float = OPEN_GAP) -> float:
    """The extent along the closing axis of the hulls in the pads' slab between the open pads' inner faces: what the
    closing pads meet (nan when nothing lies between them)."""
    ex, ey = _pinch_axes(yaw)
    lo, hi = box_extent(hulls, np.asarray(point, dtype=float) + [0.0, 0.0, PAD_ABOVE], ex, ey, (gap, pad_half[1], pad_half[2]))
    return hi - lo


def pads_free(hulls: list[np.ndarray], point: Any, yaw: float, pad_half: Any = PAD_HALF, gap: float = OPEN_GAP,
              height: float = 1.0) -> bool:
    """Whether both open pads' columns are free of the object: each pad's footprint (from the inner face at ``gap``
    outwards by the pad's thickness, the pad's width) from the pad's bottom edge to ``height`` above it."""
    ex, ey = _pinch_axes(yaw)
    t = 2 * float(pad_half[0])
    for s in (-1.0, 1.0):
        c = np.asarray(point, dtype=float) + [0.0, 0.0, PAD_ABOVE - pad_half[2] + height / 2] + s * (gap + t / 2) * ex
        if np.isfinite(box_extent(hulls, c, ex, ey, (t / 2, pad_half[1], height / 2))[0]):
            return False
    return True


def dedup(rows: list[dict[str, Any]], key: str, m: float = DEDUP_M, deg: float = DEDUP_DEG) -> dict[str, str]:
    """Greedy in the rows' order over the rows with ``key`` true: a row within ``m`` (pinch points) and ``deg`` (yaws) of a
    kept row is that row's duplicate. Returns {duplicate id: kept id}."""
    kept: list[dict[str, Any]] = []
    dup: dict[str, str] = {}
    for r in rows:
        if not r[key]:
            continue
        p = np.asarray(r["point"], dtype=float)
        twin = next((k for k in kept if np.linalg.norm(p - np.asarray(k["point"], dtype=float)) <= m
                     and yaw_gap_deg(r["yaw"], k["yaw"]) <= deg), None)
        if twin is None:
            kept.append(r)
        else:
            dup[r["id"]] = twin["id"]
    return dup


def funnel(env: Any, grasps: Any, conf: Any, destination: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """The executability funnel over one case's grasps, by confidence (a stable sort): the rows of the grasps within
    VERTICAL_DEG of straight down, each with the design's rule (the width rule, the centring rule of the pose set,
    reach()) and the variant rule (the width between the open pads, both pads' columns free, reach()), each rule's
    duplicates and kept rows; and the counts left after each step."""
    from rpent.skills.pick_place.menu import unusable_reason
    from robots.mujoco_lab.curve_poseset import OFF_CENTRE, OFF_CENTRE_COLUMN, centred

    G, S = np.asarray(grasps, dtype=float), np.asarray(conf, dtype=float)
    wmax = float(env.max_pinch_width)
    hulls = env._hull_vertex_list()
    rows: list[dict[str, Any]] = []
    for rank, k in enumerate(np.argsort(-S, kind="stable")):
        m = map_grasp(G[k])
        if m["approach_deg"] > VERTICAL_DEG:
            continue
        pt, yaw = np.asarray(m["point"]), m["yaw"]
        r: dict[str, Any] = {"id": f"g{int(k)}", "rank": rank, "conf": float(S[k]), **m, "mapped": m["closing_xy"] >= 0.5}
        w, slab, column = float(env.pinch_width(pt, yaw)), env.pinch_extent(pt, yaw), env.column_extent(pt, yaw)
        wb, free = float(bounded_width(hulls, pt, yaw)), pads_free(hulls, pt, yaw)
        why = None if r["mapped"] else "the closing axis is near vertical"
        why = why or unusable_reason(w, wmax)
        why_b = None if r["mapped"] else why
        why_b = why_b or unusable_reason(wb, wmax)
        r.update(width=w, slab=[float(x) for x in slab], column=[float(x) for x in column], width_ok=why is None,
                 centred=centred(slab, wmax / 2) and centred(column, wmax / 2), width_b=wb, width_b_ok=why_b is None,
                 pads_free=free)
        if why is None and not r["centred"]:
            why = OFF_CENTRE if not centred(slab, wmax / 2) else OFF_CENTRE_COLUMN
        if why_b is None and not free:
            why_b = PAD_LANDS
        reach = None
        if why is None or why_b is None:
            q = env.reach(_proposal(r), destination)
            reach = "ok" if q is None else f"{q.get('reason')} ({q.get('phase')})"
            if q is not None:
                why, why_b = why or f"unreachable: {reach}", why_b or f"unreachable: {reach}"
        r.update(reach=reach, feasible=why is None, why=why, feasible_b=why_b is None, why_b=why_b)
        rows.append(r)
    counts = {"generated": int(len(S)), "vertical": len(rows), "mapped": sum(r["mapped"] for r in rows)}
    for sfx, ok, dkey in (("", "feasible", "dup_of"), ("_b", "feasible_b", "dup_of_b")):
        dup = dedup(rows, ok)
        n = 0
        for r in rows:
            r[dkey] = dup.get(r["id"])
            r["kept" + sfx] = bool(r[ok]) and r[dkey] is None and n < KEEP_ROWS
            n += r["kept" + sfx]
        counts["kept" + sfx] = n
        counts["distinct" + sfx] = sum(bool(r[ok]) and r[dkey] is None for r in rows)
    counts.update(width=sum(r["width_ok"] for r in rows), centred=sum(r["width_ok"] and r["centred"] for r in rows),
                  reach=sum(r["feasible"] for r in rows), width_b=sum(r["width_b_ok"] for r in rows),
                  pads_free=sum(r["width_b_ok"] and r["pads_free"] for r in rows), reach_b=sum(r["feasible_b"] for r in rows))
    return rows, counts


def _proposal(r: dict[str, Any]) -> Any:
    from robots.mujoco_lab.adapter import Proposal

    return Proposal(s=float(r.get("f", math.nan)), point=[float(x) for x in r["point"]], yaw=float(r["yaw"]), width=float(r["width"]))


def frame_check(hulls: list[np.ndarray], grasps: Any, tol: float = 0.005) -> dict[str, Any]:
    """The frame-mapping check on the grasps within VERTICAL_DEG of straight down, for both candidate origins (the palm:
    the pad centre DEPTH along the approach; the tip: the origin is the pad centre). GraspGenX reports no finger opening,
    so the opening is the exact extent of the hulls inside the Panda's open finger volume in the grasp's own frame
    (PANDA_BOX at the candidate's pad centre); the width is the lab's pinch width at the mapped (point, yaw), the
    design's, and width_b the width between the lab's open pads. Per candidate: the counts with an opening, with a
    width, with both within ``tol``, and the median gap (mm)."""
    from rpent.skills.pick_place.features import pinch_width

    G = np.asarray(grasps, dtype=float)
    rows = []
    for k, g in enumerate(G):
        if approach_deg(g) > VERTICAL_DEG:
            continue
        r: dict[str, Any] = {"k": k, "approach_deg": approach_deg(g)}
        for name, d in (("palm", DEPTH), ("tip", 0.0)):
            m = map_grasp(g, depth=d)
            lo, hi = box_extent(hulls, m["pad"], g[:3, 0], g[:3, 1], PANDA_BOX)
            r[name] = {"opening": hi - lo, "width": float(pinch_width(hulls, m["point"], m["yaw"], PAD_HALF)),
                       "width_b": float(bounded_width(hulls, m["point"], m["yaw"]))}
        rows.append(r)
    out: dict[str, Any] = {"n": len(rows), "tol_mm": tol * 1000, "rows": rows}
    for name in ("palm", "tip"):
        o = np.array([r[name]["opening"] for r in rows], dtype=float)
        w = np.array([r[name]["width"] for r in rows], dtype=float)
        both = np.isfinite(o) & np.isfinite(w)
        gap = np.abs(o - w)[both]
        out[name] = {"opening_found": int(np.isfinite(o).sum()), "width_found": int(np.isfinite(w).sum()), "both": int(both.sum()),
                     "within_5mm": int((gap <= tol + 1e-12).sum()), "median_abs_mm": float(np.median(gap) * 1000) if len(gap) else None}
    return out


def hulls_sha(env: Any) -> str:
    """A hash of the collision hulls' world vertices (to 9 digits): the rest pose the rows were mapped on."""
    return hashlib.sha256(np.round(np.concatenate([np.asarray(V, dtype=float) for V in env._hull_vertex_list()]), 9).tobytes()).hexdigest()[:16]


def _simulated(r: dict[str, Any]) -> bool:
    return bool(r.get("kept") or r.get("kept_b") or (r.get("gate") and r.get("feasible")))


def simulate_rows(env: Any, reset: Any, rows: list[dict[str, Any]], base: dict[str, Any], fingerprint: Any, want: str,
                  hull_sha: str | None = None) -> str | None:
    """Run each row either rule keeps (and the gate row) once, in place: reset() (the case's own seed), the reset's
    fingerprint must be ``want`` and its hulls ``hull_sha`` (else the case stops: the reason is returned), then
    env.execute at the row's point and yaw with ``base`` (commit_std's force, speeds, destination and release height).
    The fields are the pose set's (curve_poseset.simulate_rows); an exception is the row's error."""
    from robots.mujoco_lab.curve_poseset import trace_sha

    for r in rows:
        if not _simulated(r):
            continue
        at_rest = reset()
        if fingerprint(env) != want:
            return f"{r['id']}: the reset's fingerprint differs from Stage 1's"
        if hull_sha is not None and hulls_sha(env) != hull_sha:
            return f"{r['id']}: the hulls after the reset differ from the first reset's"
        t0 = time.time()
        r.update(label=None, artifact=False, error=None)
        try:
            ro = env.execute(_proposal(r), base["force"], base["v_lift"], base["v_transport"], base["destination"],
                             base["release_height"], a_transport=base["a_transport"])
            d = getattr(getattr(env, "sim", None), "d", None)
            r.update(label=ro.label, artifact=bool(getattr(ro, "artifact", False)) or at_rest is False,
                     damage=getattr(ro, "damage", None), drop_z=getattr(ro, "drop_z", None), loss_phase=getattr(ro, "loss_phase", None),
                     max_z=getattr(ro, "max_z", None),
                     qpos_sha=None if d is None else hashlib.sha256(np.round(np.asarray(d.qpos, dtype=float), 9).tobytes()).hexdigest()[:16],
                     trace_sha=trace_sha(getattr(ro, "trace", [])))
        except Exception as exc:  # noqa: BLE001  a simulator fault is this row's error
            r["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        r["seconds"] = round(time.time() - t0, 2)
    return None


def replay_rows(env: Any, reset: Any, rows: list[dict[str, Any]], base: dict[str, Any], fingerprint: Any, want: str,
                hull_sha: str | None = None) -> dict[str, Any]:
    """The 2-seed determinism gate: the simulated rows once more, in reverse order, on copies without their outcomes
    (``reset`` at another reset seed), each against its first run on curve_poseset.SAME."""
    from robots.mujoco_lab.curve_poseset import OUTCOME, SAME

    twin = [{k: v for k, v in r.items() if k not in OUTCOME} for r in reversed(rows) if "seconds" in r]
    stop = simulate_rows(env, reset, twin, base, fingerprint, want, hull_sha)
    first = {r["id"]: r for r in rows}
    done = [t for t in twin if "seconds" in t]
    bad = [{"id": t["id"], "first": [first[t["id"]].get(k) for k in SAME], "replay": [t.get(k) for k in SAME]}
           for t in done if [first[t["id"]].get(k) for k in SAME] != [t.get(k) for k in SAME]]
    return {"rows": len(done), "mismatches": bad, "stop": stop}


def ceiling(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """The capability ceiling over the rows ``key`` keeps ("kept": the design's rule, "kept_b": the variant), in
    confidence order: curve_poseset.ceiling's episode (a success when a clean row places; else the first clean row
    without irreversible damage, else the first clean row; no kept row is a give-up; undetermined when no clean row
    places and a row is not clean)."""
    from robots.mujoco_lab.curve_poseset import ceiling as pose_ceiling

    cand = sorted((r for r in rows if r.get(key)), key=lambda r: r["rank"])
    return pose_ceiling([dict(r, feasible=True, current=True) for r in cand])


GATE = ("across", 0.5)          # the gate row: the pose set's across pinch at 0.5, run through this module's path


def _tags(env: Any, rows: list[dict[str, Any]]) -> tuple[float, float]:
    """Each row's place on the object, in place: f (the point's fraction along the long axis, 0 at its axis_lo end),
    r_axis (the point's distance from the long axis's line), h (the pad centre's height over the hulls' height), near
    (the nearest of the pose set's ten rows within DEDUP_M and DEDUP_DEG, else None) and near_current (near is one of
    the three current rows). Returns the hulls' lowest and highest z."""
    from robots.mujoco_lab.curve_poseset import CURRENT, FRACS, TURN, row_id

    R, p = np.asarray(env.sim.obj_mat(), dtype=float), np.asarray(env.sim.obj_pos(), dtype=float)
    ax, c = np.asarray(env.props["axis"], dtype=float), np.asarray(env.props["centre"], dtype=float)
    lo, L = float(env.props["axis_lo"]), float(env.props["length"])
    V = np.concatenate([np.asarray(h, dtype=float) for h in env._hull_vertex_list()])
    z0, z1 = float(V[:, 2].min()), float(V[:, 2].max())
    pose = [(row_id(d, q.s), d == "across" and q.s in CURRENT, q) for d, t in TURN.items() for q in env.proposals(FRACS, turn=t)]
    for r in rows:
        q = R.T @ (np.asarray(r["point"], dtype=float) - p) - c
        s = float(q @ ax)
        r.update(f=(s - lo) / L, r_axis=float(np.linalg.norm(q - s * ax)), h=(float(r["pad"][2]) - z0) / (z1 - z0))
        near = [(float(np.linalg.norm(np.asarray(r["point"], dtype=float) - np.asarray(x.point, dtype=float))), rid, cur)
                for rid, cur, x in pose if yaw_gap_deg(r["yaw"], x.yaw) <= DEDUP_DEG]
        near = sorted(n for n in near if n[0] <= DEDUP_M)
        r.update(near=near[0][1] if near else None, near_current=bool(near and near[0][2]))
    return z0, z1


def _gate_row(env: Any, destination: str) -> dict[str, Any]:
    """The pose set's across pinch at 0.5 as a row of this module, feasible by the pose set's rule for a current row (the
    width rule and reach()): only then did the pose set simulate it."""
    from rpent.skills.pick_place.menu import unusable_reason

    q = env.proposals((GATE[1],), turn=0.0)[0]
    why = unusable_reason(float(q.width), float(env.max_pinch_width))
    if why is None:
        r = env.reach(q, destination)
        why = None if r is None else f"unreachable: {r.get('reason')} ({r.get('phase')})"
    return {"id": f"gate_{GATE[0]}_{GATE[1]:g}", "rank": -1, "gate": True, "feasible": why is None, "why": why, "kept": False,
            "kept_b": False, "f": float(q.s), "point": [float(x) for x in q.point], "yaw": float(q.yaw), "width": float(q.width)}


def _sha_bytes(a: Any) -> str:
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def _setup(prims: Any, rec1: dict[str, Any], seed: int, world: Any) -> dict[str, Any]:
    """The case's draw and its reset at seed k (evpi_sim.seeds_of): {"reset_at", "destination", "want"}, or {"stop"}."""
    from robots.mujoco_lab import adapter, evpi_sim
    from robots.mujoco_lab import calibrate as cal

    o, idx = rec1["object"], int(rec1["idx"])
    env = prims.env
    theta, com_end, destination, hidden = cal.case_draw(seed, o, idx, adapter.object_props(o)["mass"], False, fragility=world)
    if destination != rec1["destination"]:
        return {"stop": f"the case's draw gives the destination {destination}, Stage 1's is {rec1['destination']}"}
    if world is not None:
        env.slosh_zeta, env.slosh_scale = world[1].slosh_zeta, world[1].slosh_scale
    env.classes = None if world is None else world[0]
    seeds = evpi_sim.seeds_of(seed, o, idx)
    fixture = destination if destination in cal.MESH_IDS else None

    def reset_at(k: int) -> Any:
        def reset() -> bool:
            env.reset(o, theta, seed=int(seeds[k]), com_end=com_end, fixture=fixture, fill=hidden["fill"], frag=hidden["frag"])
            return bool(dict(getattr(env, "reset_info", None) or {}).get("at_rest", True))
        return reset

    return {"reset_at": reset_at, "destination": destination, "want": rec1["fingerprint"], "stop": None}


def cloud_case(prims: Any, rec1: dict[str, Any], seed: int, world: Any) -> dict[str, Any]:
    """The cloud step for one case: Stage 2's skip rules, the reset at the case's own seed (Stage 1's fingerprint, else
    a stop) and the cloud of its hulls at cloud_seed (its hash is what the run step checks)."""
    from robots.mujoco_lab import com_gap
    from robots.mujoco_lab.curve import stage2_skip

    out: dict[str, Any] = {k: rec1.get(k) for k in ("object", "idx", "side", "fold")}
    out.update(skipped=stage2_skip(rec1), stop=None, cloud=None)
    if out["skipped"]:
        return out
    s = _setup(prims, rec1, seed, world)
    if s["stop"]:
        return dict(out, stop=s["stop"])
    env = prims.env
    s["reset_at"](0)()
    fp = com_gap.fingerprint(env)
    if fp != s["want"]:
        return dict(out, stop="the reset's fingerprint differs from Stage 1's")
    hulls = env._hull_vertex_list()
    cloud = surface_cloud(hulls, seed=cloud_seed(rec1["object"], int(rec1["idx"])))
    return dict(out, cloud=cloud, cloud_sha256=_sha_bytes(cloud), n_hulls=len(hulls), fingerprint=fp,
                rest_R=np.asarray(env.sim.obj_mat(), dtype=float).tolist(), rest_p=np.asarray(env.sim.obj_pos(), dtype=float).tolist())


def write_cloud(out: str, rec1: dict[str, Any], cloud: np.ndarray, info: dict[str, Any]) -> dict[str, Any]:
    """Save one case's cloud as ``out``/<object>__<idx>.npz; returns its index line."""
    import os

    name = f"{rec1['object']}__{int(rec1['idx'])}.npz"
    np.savez(os.path.join(out, name), cloud=np.asarray(cloud, dtype=np.float32))
    return dict({k: rec1.get(k) for k in ("object", "idx", "side", "fold")}, file=name, cloud_sha256=_sha_bytes(np.asarray(cloud, dtype=np.float32)),
                n_points=int(len(cloud)), **info)


def load_grasps(gdir: str) -> dict[tuple[str, int], dict[str, Any]]:
    """The request step's grasps by (object, idx): {"grasps", "conf", "cloud_sha256", "error"}; a grasp file whose hash
    is not its index line's is that case's error."""
    import json
    import os

    from robots.mujoco_lab.graspgen_request import sha_arrays

    out: dict[tuple[str, int], dict[str, Any]] = {}
    for line in open(os.path.join(gdir, "index.jsonl")):
        x = json.loads(line)
        e: dict[str, Any] = {"grasps": None, "conf": None, "cloud_sha256": x["cloud_sha256"], "error": x.get("error")}
        if not e["error"]:
            z = np.load(os.path.join(gdir, x["file"]))
            if sha_arrays(z["grasps"], z["conf"]) != x["grasps_sha256"]:
                e["error"] = "the grasp file's hash differs from its index line"
            else:
                e.update(grasps=z["grasps"].astype(float), conf=z["conf"].astype(float))
        out[(x["object"], int(x["idx"]))] = e
    return out


def graspgen_case(prims: Any, rec1: dict[str, Any], seed: int, world: Any, entry: dict[str, Any] | None,
                  replay: bool = False) -> dict[str, Any]:
    """One case: Stage 2's skip rules; the reset at the case's own seed (Stage 1's fingerprint, else a stop); the cloud
    again from the hulls (its hash must be the generator's input's, else a stop); the tilt, the length and the balance
    point; commit_std's base (curve_poseset.base_gate); the funnel over the case's grasps, the rows' places, the
    frame-mapping check; one simulation per kept row and of the gate row, each after its own reset; both ceilings (an
    undetermined design ceiling drops the case); with ``replay`` the 2-seed gate (a row that differs stops the case)."""
    from robots.mujoco_lab import com_gap
    from robots.mujoco_lab.curve import stage2_skip
    from robots.mujoco_lab.curve_poseset import KEEP, KIND_OF, UPRIGHT_DEG, _err, balance_fraction, base_gate, commit_std_base, tilt_deg

    o, idx = rec1["object"], int(rec1["idx"])
    rec: dict[str, Any] = dict({k: rec1.get(k) for k in KEEP}, kind=KIND_OF.get(o, "other"), hammer="hammer" in o, rows=[],
                               counts=None, n_grasps=None, base=None, ceiling=None, ceiling_b=None, frame=None,
                               skipped=stage2_skip(rec1), dropped=False, stop=None, error=None, replay=None)
    if rec["skipped"]:
        return dict(rec, dropped=True)
    if entry is None:
        return dict(rec, stop="no grasps for this case")
    if entry.get("error"):
        return dict(rec, stop=f"the generator failed on this case: {entry['error']}")
    env = prims.env
    st = _setup(prims, rec1, seed, world)
    if st["stop"]:
        return dict(rec, stop=st["stop"])
    reset_at, destination = st["reset_at"], st["destination"]
    want = rec1["fingerprint"]
    try:
        reset_at(0)()
        if com_gap.fingerprint(env) != want:
            return dict(rec, stop="the reset's fingerprint differs from Stage 1's")
        hulls = env._hull_vertex_list()
        sha = _sha_bytes(surface_cloud(hulls, seed=cloud_seed(o, idx)))
        if sha != entry["cloud_sha256"]:
            return dict(rec, stop="the cloud at this reset is not the cloud the generator saw")
        t = env.truth()
        tilt = tilt_deg(env.sim.obj_mat(), env.props["axis"])
        rec.update(tilt_deg=tilt, upright=tilt > UPRIGHT_DEG, length=float(env.props["length"]), cloud_sha256=sha,
                   com_f=balance_fraction(env.props, t["com_obj"]), sim_truth={k: t.get(k) for k in ("mass", "com_s", "com_obj", "fill")})
        base = commit_std_base(env, o, destination)
        why = base_gate(rec1, base)
        if why:
            return dict(rec, stop=why)
        rows, counts = funnel(env, entry["grasps"], entry["conf"], destination)
        z = _tags(env, rows)
        rows.append(_gate_row(env, destination))
        hsha = hulls_sha(env)
        rec.update(base=base, rows=rows, counts=counts, n_grasps=int(len(entry["conf"])), object_z=list(z), hull_sha=hsha,
                   frame=frame_check(hulls, entry["grasps"]))
        stop = simulate_rows(env, reset_at(0), rows, base, com_gap.fingerprint, want, hsha)
        if stop:
            return dict(rec, stop=stop)
        if replay:
            t1 = time.time()
            rec["replay"] = replay_rows(env, reset_at(1), rows, base, com_gap.fingerprint, want, hsha)
            rec["replay_seconds"] = round(time.time() - t1, 2)
            if rec["replay"]["stop"]:
                return dict(rec, stop=f"replay: {rec['replay']['stop']}")
            if rec["replay"]["mismatches"]:
                return dict(rec, stop=f"replay: {len(rec['replay']['mismatches'])} rows differ at the second seed (the 2-seed gate)")
    except Exception as exc:  # noqa: BLE001  a case that raises outside a row's simulation is dropped
        return dict(rec, error=_err(exc), dropped=True)
    c, cb = ceiling(rows, "kept"), ceiling(rows, "kept_b")
    return dict(rec, ceiling=c, ceiling_b=cb, dropped=not c["determined"])


FUNNEL = ("generated", "vertical", "mapped", "width", "centred", "reach", "distinct", "kept")
FUNNEL_B = ("width_b", "pads_free", "reach_b", "distinct_b", "kept_b")


# ---------------------------------------------------------------- the summary's pieces

ANGLE_EDGES = (30.0, 60.0, 120.0)
ANGLE_BINS = ("within 30° of down", "30–60°", "side (60–120°)", "from below (> 120°)")
RIM_H, OFF_AXIS_M = 0.85, 0.01   # ruling 7's tags: a pinch at this share of the object's height or higher, this far off its axis
TAGS = ("an axis-rule pose", "rim or top edge", "off the axis", "on the axis at another fraction or yaw")
RULES = (("gg", "kept"), ("ggb", "kept_b"))
ARMS = ("gg", "ggb", "c10", "c3", "oracle", "r4")
ARM_NAMES = {"gg": "GraspGen, the design's rule", "ggb": "GraspGen, the between-pad variant", "c10": "10-pinch ceiling",
             "c3": "3-pinch ceiling", "oracle": "oracle arm", "r4": "R4 (always grasp)"}
HAMMER_ROW_CLASSES = ("placed by a row", "lifted, never placed", "not lifted by any row", "no clean simulation", "no kept row")
ROW_OBJECTS = ("ycb__bowl", "ycb__pitcher")          # reading 4
# Reading 3's head end, as a pinch fraction: the hammer's wide end on its hulls (15.1 and 15.2 cm across at f 0 to 0.2,
# 4.7 and 5.0 cm at f 0.8 to 1). The drawn CoM is not used: a CoM-shift draw can put it on the handle side.
HAMMER_HEAD_F = {"handal__hammer_1": 0.0, "handal__hammer_7": 0.0}
G_ABOVE = "more than 2 points above the 10-pinch ceiling"
G_NOT = "not more than 2 points above the 10-pinch ceiling"


def angle_histogram(grasps: Any) -> list[int]:
    """Reading 1's histogram: the grasps by approach angle from straight down (to 9 digits), in ANGLE_BINS (each bin
    holds its upper edge)."""
    a = np.round(np.array([approach_deg(g) for g in np.asarray(grasps, dtype=float)]), 9)
    e = ANGLE_EDGES
    return [int((a <= e[0]).sum()), int(((a > e[0]) & (a <= e[1])).sum()), int(((a > e[1]) & (a <= e[2])).sum()), int((a > e[2]).sum())]


def row_tag(r: dict[str, Any]) -> str:
    """Reading 5's place of a row (ruling 7), in TAGS' order: within 1 cm and 10 degrees of a pose-set row, at RIM_H of
    the object's height or higher, farther than OFF_AXIS_M from its long axis, else on the axis at another fraction or yaw."""
    if r.get("near"):
        return TAGS[0]
    if round(float(r["h"]), 9) >= RIM_H:
        return TAGS[1]
    return TAGS[2] if round(float(r["r_axis"]), 9) > OFF_AXIS_M else TAGS[3]


def clean_row(r: dict[str, Any]) -> bool:
    """A simulated row of the funnel (not the gate row) without an artifact or an error."""
    return not r.get("gate") and r.get("label") is not None and not r.get("artifact") and not r.get("error")


def kept_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return [r for r in rows if r.get(key) and not r.get("gate")]


def _has_state(x: dict[str, Any] | None) -> bool:
    """A simulated row without an error and with both its final-state and trace hashes."""
    return x is not None and "seconds" in x and not x.get("error") and bool(x.get("qpos_sha")) and bool(x.get("trace_sha"))


def exact_gate(recs: list[dict[str, Any]], pose: list[dict[str, Any]]) -> dict[str, Any]:
    """The gate row against the pose set's record of the same case: its across pinch at 0.5 (GATE) must have the same
    final state and trace hashes. Counts the cases with a simulated gate row: equal, differ (listed), missing (either
    row has no state: the pose set did not simulate it, or a row holds an error or lacks a hash)."""
    from robots.mujoco_lab.curve_poseset import _key, row_id

    pidx = {_key(p): {x["id"]: x for x in p["rows"]} for p in pose}
    out: dict[str, Any] = {"cases": 0, "equal": 0, "differ": 0, "missing": 0, "differences": []}
    for r in recs:
        g = next((x for x in r.get("rows") or [] if x.get("gate") and "seconds" in x), None)
        if g is None:
            continue
        out["cases"] += 1
        p = pidx.get(_key(r), {}).get(row_id(*GATE))
        if not (_has_state(p) and _has_state(g)):
            out["missing"] += 1
        elif (p.get("qpos_sha"), p.get("trace_sha")) == (g.get("qpos_sha"), g.get("trace_sha")):
            out["equal"] += 1
        else:
            out["differ"] += 1
            out["differences"].append({"object": r["object"], "idx": int(r["idx"])})
    return out


def pilot_paired(joined: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """The joined cases ({"g", "pose", "oracle", "stage2"}) that no arm drops: the pilot (a skip, an error or the design
    rule's undetermined ceiling), the variant's undetermined ceiling, the pose set, the oracle arm, Stage 2 (the pose set's
    ruling 2); with the numbers dropped in all and by each (a case can count for several)."""
    names = ("graspgen", "variant", "pose", "oracle", "stage2")
    flags = [(bool(j["g"]["dropped"]), not (j["g"].get("ceiling_b") or {}).get("determined", False), bool(j["pose"]["dropped"]),
              bool(j["oracle"]["dropped"]), bool(j["stage2"]["dropped"])) for j in joined]
    kept = [j for j, f in zip(joined, flags) if not any(f)]
    return kept, dict({"dropped": sum(any(f) for f in flags)}, **{n: sum(f[k] for f in flags) for k, n in enumerate(names)})


def pilot_views(paired: list[dict[str, Any]], k0: str, k4: str) -> list[dict[str, Any]]:
    """Per joined case: its object, side, strata, the pilot's rows (without the gate row), the pose set's rows and the
    six arms' episodes: both GraspGen ceilings, the 10- and the 3-pinch ceilings, the oracle arm and R4 (Stage 2)."""
    from robots.mujoco_lab.curve import episode

    out = []
    for j in paired:
        g, p, o, s2 = j["g"], j["pose"], j["oracle"], j["stage2"]
        out.append({"object": g["object"], "idx": int(g["idx"]), "side": g["side"], "thin": bool(g.get("thin")), "kind": g.get("kind"),
                    "hammer": bool(g.get("hammer")), "upright": bool(g.get("upright")), "com_f": g.get("com_f"), "length": g.get("length"),
                    "rows": [r for r in g["rows"] if not r.get("gate")], "prows": p["rows"], "gg": g["ceiling"], "ggb": g["ceiling_b"],
                    "c10": p["ceiling10"], "c3": p["ceiling3"], "oracle": episode(o["decisions"], o["sims"], k0),
                    "r4": episode(s2["decisions"], s2["sims"], k4)})
    return out


def pilot_rule(b: dict[str, Any]) -> str:
    """The design's 2-point rule on a paired success difference (GraspGen - the 10-pinch ceiling), reported with no
    verdict: above when g > EPS and the interval's bottom > 0, not above when its top < EPS, else undecided."""
    from robots.mujoco_lab.curve import EPS

    x = b["success"]
    if x["diff"] > EPS and x["lo"] > 0.0:
        return G_ABOVE
    if x["hi"] < EPS:
        return G_NOT
    return f"undecided at n = {b['cases']}"


def pilot_block(vs: list[dict[str, Any]]) -> dict[str, Any]:
    """Reading 2 on one view: each arm's rates, GraspGen (each rule) - the 10-pinch ceiling paired by case with the object
    bootstrap (curve_oracle.paired_outcomes), the 2-point rule beside each, and the success 2 x 2 against the 10-pinch."""
    from robots.mujoco_lab.curve_oracle import _rates, paired_outcomes

    def two(name: str) -> dict[str, int]:
        a = [(bool(v[name]["success"]), bool(v["c10"]["success"])) for v in vs]
        return {"both": sum(x and y for x, y in a), "graspgen_only": sum(x and not y for x, y in a),
                "c10_only": sum(y and not x for x, y in a), "neither": sum(not x and not y for x, y in a)}

    b = paired_outcomes([(v["object"], v["gg"], v["c10"]) for v in vs])
    bb = paired_outcomes([(v["object"], v["ggb"], v["c10"]) for v in vs])
    return {"cases": len(vs), "objects": len({v["object"] for v in vs}), "rates": {a: _rates([v[a] for v in vs]) for a in ARMS},
            "gg_vs_c10": b, "ggb_vs_c10": bb, "rule": pilot_rule(b) if vs else None, "rule_b": pilot_rule(bb) if vs else None,
            "success_2x2": two("gg"), "success_2x2_b": two("ggb")}


def by_object(vs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each object's side, case count and arm rates, in the views' order."""
    from robots.mujoco_lab.curve_oracle import _rates

    out = {}
    for o in dict.fromkeys(v["object"] for v in vs):
        x = [v for v in vs if v["object"] == o]
        out[o] = dict({"side": x[0]["side"], "n": len(x)}, **{a: _rates([v[a] for v in x]) for a in ARMS})
    return out


def hammer_rows_class(rows: list[dict[str, Any]], key: str) -> str:
    """Reading 3 for one case and rule, in HAMMER_ROW_CLASSES: a clean kept row placed it, lifted it (a label other than
    not_lifted or missed) but none placed it, none lifted it; no kept row was clean; the rule kept no row."""
    from robots.mujoco_lab.curve_poseset import LIFT_FAILS

    k = kept_rows(rows, key)
    if not k:
        return HAMMER_ROW_CLASSES[4]
    clean = [r for r in k if clean_row(r)]
    if not clean:
        return HAMMER_ROW_CLASSES[3]
    if any(r["label"] == "placed_stable" for r in clean):
        return HAMMER_ROW_CLASSES[0]
    return HAMMER_ROW_CLASSES[1] if any(r["label"] not in LIFT_FAILS for r in clean) else HAMMER_ROW_CLASSES[2]


def hammer_line(vs: list[dict[str, Any]]) -> dict[str, Any]:
    """Reading 3: the hammer cases by class under each rule and under the 10-pinch ceiling (curve_poseset.hammer_class);
    for the clean rows either rule keeps, the distance from the balance point along the axis (|f - com_f| x length) and
    from the head end (HAMMER_HEAD_F; a hammer without one is refused), apart for the rows that placed the hammer; per
    case its classes, the placing rows (with their tag and rules) and the lifting rows."""
    from robots.mujoco_lab.curve_poseset import HAMMER_CLASSES, LIFT_FAILS, hammer_class

    hv = [v for v in vs if v["hammer"]]
    out: dict[str, Any] = {"cases": len(hv), "objects": sorted({v["object"] for v in hv}),
                           "classes": {name: {c: 0 for c in HAMMER_ROW_CLASSES} for name, _ in RULES},
                           "c10_classes": {c: 0 for c in HAMMER_CLASSES}, "placed": {"balance_m": [], "head_m": []},
                           "not_placed": {"balance_m": [], "head_m": []}, "per_case": []}
    for v in hv:
        L, cf = float(v["length"]), float(v["com_f"])
        if v["object"] not in HAMMER_HEAD_F:
            raise ValueError(f"{v['object']}: no head end in HAMMER_HEAD_F")
        head = HAMMER_HEAD_F[v["object"]]
        cls = {name: hammer_rows_class(v["rows"], key) for name, key in RULES}
        for name, c in cls.items():
            out["classes"][name][c] += 1
        c10 = hammer_class(v["prows"])
        out["c10_classes"][c10] += 1
        placed, lifted = [], []
        for r in v["rows"]:
            rules = [name for name, key in RULES if r.get(key)]
            if not rules or not clean_row(r):
                continue
            bal, hd = abs(float(r["f"]) - cf) * L, abs(float(r["f"]) - head) * L
            side = "placed" if r["label"] == "placed_stable" else "not_placed"
            out[side]["balance_m"].append(bal)
            out[side]["head_m"].append(hd)
            if r["label"] == "placed_stable":
                placed.append({"id": r["id"], "f": r["f"], "balance_m": bal, "head_m": hd, "tag": row_tag(r), "rules": rules})
            if r["label"] not in LIFT_FAILS:
                lifted.append(r["id"])
        out["per_case"].append({"object": v["object"], "idx": v["idx"], "com_f": cf, "length": L, "classes": cls, "c10_class": c10,
                                "placed": placed, "lifted": lifted})
    return out


def object_line(vs: list[dict[str, Any]], objects: tuple[str, ...] = ROW_OBJECTS) -> dict[str, dict[str, Any]]:
    """Reading 4, per object and rule: the cases, those with a kept row, the kept rows by tag, the clean ones, the placing
    ones by tag, the cases the rule's ceiling places, and the clean rows' label mix; beside it the 10-pinch ceiling's
    placed cases and feasible rows."""
    from collections import Counter

    out: dict[str, dict[str, Any]] = {}
    for o in objects:
        x = [v for v in vs if v["object"] == o]
        out[o] = {"c10_placed": sum(bool(v["c10"]["success"]) for v in x), "c10_rows": sum(int(v["c10"].get("n_rows", 0)) for v in x)}
        for name, key in RULES:
            kr = [r for v in x for r in kept_rows(v["rows"], key)]
            clean = [r for r in kr if clean_row(r)]
            pl = [r for r in clean if r["label"] == "placed_stable"]
            out[o][name] = {"cases": len(x), "cases_with_row": sum(bool(kept_rows(v["rows"], key)) for v in x), "rows": len(kr),
                            "tags": dict(Counter(row_tag(r) for r in kr)), "clean": len(clean), "placing": len(pl),
                            "placing_tags": dict(Counter(row_tag(r) for r in pl)), "cases_placed": sum(bool(v[name]["success"]) for v in x),
                            "labels": dict(Counter(r["label"] for r in clean))}
    return out


def placing_tags(vs: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """Reading 5 for one rule ("gg" or "ggb"): the clean placing rows it keeps by tag, the cases with one of each tag, the
    rows by object; the gain cases (its ceiling places, the 10-pinch ceiling does not) by the tags of their placing rows,
    and the loss cases (the reverse)."""
    from collections import Counter

    key = dict(RULES)[name]
    rows, cases, gain_tags = Counter(), Counter(), Counter()
    by: dict[str, Counter] = {}
    gain = loss = 0
    for v in vs:
        tags = [row_tag(r) for r in kept_rows(v["rows"], key) if clean_row(r) and r["label"] == "placed_stable"]
        rows.update(tags)
        cases.update(set(tags))
        if tags:
            by.setdefault(v["object"], Counter()).update(tags)
        if v[name]["success"] and not v["c10"]["success"]:
            gain += 1
            gain_tags.update(set(tags))
        loss += bool(v["c10"]["success"]) and not v[name]["success"]
    return {"rows": dict(rows), "cases": dict(cases), "by_object": {o: dict(c) for o, c in by.items()},
            "gain": {"cases": gain, "tags": dict(gain_tags)}, "loss": {"cases": loss}}


def label_gate(vs: list[dict[str, Any]]) -> dict[str, Any]:
    """Reading 6's tolerance label gate (ruling 6, no stop): every clean kept row within the de-dup tolerance of a current
    pinch (near_current) against the pose set's clean row there: compared, agree, differ (listed), missing."""
    from robots.mujoco_lab.curve_poseset import _clean

    out: dict[str, Any] = {"compared": 0, "agree": 0, "differ": 0, "missing": 0, "differences": []}
    for v in vs:
        prow = {r["id"]: r for r in v["prows"]}
        for r in v["rows"]:
            if not (r.get("kept") or r.get("kept_b")) or not clean_row(r) or not r.get("near_current"):
                continue
            p = prow.get(r["near"])
            if p is None or not _clean(p):
                out["missing"] += 1
                continue
            out["compared"] += 1
            if p["label"] == r["label"]:
                out["agree"] += 1
            else:
                out["differ"] += 1
                out["differences"].append({"object": v["object"], "idx": v["idx"], "row": r["id"], "pose_row": r["near"],
                                           "graspgen": r["label"], "pose": p["label"]})
    return out


def funnel_table(recs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Reading 1: the funnel's counts summed by object (the records with counts) and in all, with the share of the
    generated grasps lost at the approach step."""
    out: dict[str, dict[str, Any]] = {}
    for r in recs:
        if not r.get("counts"):
            continue
        x = out.setdefault(r["object"], dict({"cases": 0}, **{k: 0 for k in FUNNEL + FUNNEL_B}))
        x["cases"] += 1
        for k in FUNNEL + FUNNEL_B:
            x[k] += int(r["counts"][k])
    out["all"] = dict({"cases": sum(x["cases"] for x in out.values())}, **{k: sum(x[k] for x in out.values()) for k in FUNNEL + FUNNEL_B})
    for x in out.values():
        x["lost_at_approach"] = 1.0 - x["vertical"] / x["generated"] if x["generated"] else None
    return out


# ---------------------------------------------------------------- the commands

def select_cases(recs1: list[dict[str, Any]], cases: str | None = None, objects: str | None = None, sides: str | None = None,
                 per_object: int | None = None) -> list[dict[str, Any]]:
    """The Stage 1 records of the run, in (object, idx) order: ``cases`` (object:idx, comma-separated), or every case of
    ``objects`` on ``sides`` (the first ``per_object`` by idx when given); an asked case or object with no record stops."""
    if cases:
        want = {(c.rsplit(":", 1)[0], int(c.rsplit(":", 1)[1])) for c in cases.split(",")}
        recs = [r for r in recs1 if (r["object"], int(r["idx"])) in want]
        if len(recs) != len(want):
            raise SystemExit(f"{len(want) - len(recs)} of the --cases are not in the Stage 1 records")
        return sorted(recs, key=lambda r: (r["object"], int(r["idx"])))
    side_set, out = set((sides or "test").split(",")), []
    for o in (objects or "").split(","):
        rs = sorted((r for r in recs1 if r["object"] == o and r["side"] in side_set), key=lambda r: int(r["idx"]))
        if not rs:
            raise SystemExit(f"{o}: no Stage 1 case on the sides {sorted(side_set)}")
        out += rs[:per_object] if per_object else rs
    return sorted(out, key=lambda r: (r["object"], int(r["idx"])))


def case_line(rec: dict[str, Any]) -> str:
    """A case's log line: counts, times and errors, no label (no early look)."""
    n = sum("seconds" in r for r in rec["rows"])
    bad = sum(bool(r.get("error")) for r in rec["rows"])
    c = rec.get("counts") or {}
    return (f"{rec['object']} {rec['idx']}: {c.get('generated', 0)} grasps, {c.get('kept', 0)} kept, {c.get('kept_b', 0)} kept by the "
            f"variant, {n} rows simulated, {float(rec.get('seconds') or 0.0):.1f} s" + (f", {bad} row errors" if bad else "")
            + (f", skipped ({rec['skipped']})" if rec.get("skipped") else "") + (f", error {rec['error']}" if rec.get("error") else "")
            + (f", STOP {rec['stop']}" if rec.get("stop") else ""))


def graspgen_object(job: tuple[str, list[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    """One object's cases on its fold's bundle (curve_poseset.poseset_object's sensor and world), with the request
    step's grasps; a gate that fails ends the object with ``stop``."""
    import sys

    from robots.mujoco_lab import calibrate as cal
    from robots.mujoco_lab.prior import fragility_world

    o, recs1, s = job
    entries = load_grasps(s["grasps"])
    prims = cal._primitives(s["bundle"], None, None)
    if getattr(prims, "reset_s", None) is not None:
        raise RuntimeError("the pilot scores the absorbing attempt only (this bundle restores the object after damage)")
    probes, prims.probe_models = getattr(prims, "probe_models", None), {}
    prims.force_noise = cal.job_force_noise(None, prims.manifest)
    prims.env.force_noise = prims.force_noise
    world = fragility_world(prims.manifest)
    out: dict[str, Any] = {"records": [], "stop": None}
    try:
        for rec1 in recs1:
            t0 = time.time()
            rec = graspgen_case(prims, rec1, int(s["seed"]), world, entries.get((o, int(rec1["idx"]))), replay=bool(s.get("replay")))
            rec["seconds"] = round(time.time() - t0, 2)
            out["records"].append(rec)
            print(case_line(rec), file=sys.stderr, flush=True)
            if rec["stop"]:
                out["stop"] = f"{o} {rec1['idx']}: {rec['stop']}"
                break
    finally:
        prims.probe_models = probes
    return out


def _metrics(done: list[dict[str, Any]]) -> dict[str, Any]:
    """A run's counts and times, no label: cases, skips, drops, stops, errors, the funnel totals, rows simulated, row
    errors, the replay gate's rows and mismatches."""
    rows = [r for rec in done for r in rec["rows"] if "seconds" in r]
    rep = [rec["replay"] for rec in done if rec.get("replay")]
    tot = {k: sum(int((rec.get("counts") or {}).get(k, 0)) for rec in done) for k in FUNNEL + FUNNEL_B}
    return {"cases": len(done), "skipped": sum(bool(r["skipped"]) for r in done), "dropped": sum(bool(r["dropped"]) for r in done),
            "stops": sum(bool(r.get("stop")) for r in done), "case_errors": sum(bool(r["error"]) for r in done),
            "funnel": tot, "rows_simulated": len(rows), "row_errors": sum(bool(r.get("error")) for r in rows),
            "row_seconds": round(sum(float(r["seconds"]) for r in rows), 1), "replay_rows": sum(x["rows"] for x in rep),
            "replay_mismatches": sum(len(x["mismatches"]) for x in rep),
            "case_seconds": round(sum(float(r.get("seconds") or 0.0) for r in done), 1),
            "case_seconds_max": max((float(r.get("seconds") or 0.0) for r in done), default=0.0)}


def _stage1(a: Any) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    from robots.mujoco_lab.curve import load_stage1

    dirs = a.stage1.split(",")
    recs1, cfg1 = load_stage1(dirs)
    return recs1, cfg1, dirs


def clouds(a: Any) -> str:
    """The cloud step: index.jsonl and one <object>__<idx>.npz per case in --out (cloud_case), run_config.json and
    run_metrics.json, wandb. A stop ends the step (exit 3)."""
    import json
    import os
    from collections import defaultdict

    from robots.mujoco_lab import calibrate as cal
    from robots.mujoco_lab.curve import _sha
    from robots.mujoco_lab.estimation_check import _revision
    from robots.mujoco_lab.identify_fit import require_bundle_generator
    from robots.mujoco_lab.prior import fragility_world
    from robots.mujoco_lab.tracking import log_run

    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(os.path.join(a.out, "index.jsonl")):
        raise SystemExit(f"{a.out}: an index exists: pick a new --out")
    recs1, cfg1, dirs = _stage1(a)
    generator = require_bundle_generator(cfg1["bundle"])
    recs = select_cases(recs1, a.cases, a.objects, a.sides, a.per_object)
    config = {**vars(a), **_revision(), "generator": generator, "bundle": cfg1["bundle"], "seed": cfg1["seed"],
              "stage1_records_sha256": {d: _sha(os.path.join(d, "records.jsonl")) for d in dirs},
              "cases": [(r["object"], int(r["idx"])) for r in recs], "n_points": N_POINTS}
    with open(os.path.join(a.out, "run_config.json"), "w") as f:
        json.dump(config, f, indent=1)
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        by[r["object"]].append(r)
    t0, stop, skipped, n = time.time(), None, [], 0
    with open(os.path.join(a.out, "index.jsonl"), "w") as idx:
        for o, items in by.items():
            prims = cal._primitives(os.path.join(cfg1["bundle"], "folds", f"fold_{items[0]['fold']}"), None, None)
            prims.force_noise = cal.job_force_noise(None, prims.manifest)
            prims.env.force_noise = prims.force_noise
            world = fragility_world(prims.manifest)
            for rec1 in items:
                c = cloud_case(prims, rec1, int(cfg1["seed"]), world)
                if c["skipped"]:
                    skipped.append([o, int(rec1["idx"]), c["skipped"]])
                    continue
                if c["stop"]:
                    stop = f"{o} {rec1['idx']}: {c['stop']}"
                    break
                info = {k: c[k] for k in ("n_hulls", "fingerprint", "rest_R", "rest_p")}
                idx.write(json.dumps(write_cloud(a.out, rec1, c["cloud"], info)) + "\n")
                idx.flush()
                n += 1
            if stop:
                break
    metrics = {"clouds": n, "skipped": skipped, "stop": stop, "wall_seconds": round(time.time() - t0, 1)}
    metrics["wandb"] = log_run("curve-graspgen-clouds", config, {k: v for k, v in metrics.items() if k != "skipped"})
    with open(os.path.join(a.out, "run_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print(f"done: {json.dumps(metrics)}", flush=True)
    if stop:
        raise SystemExit(3)
    return a.out


def log_request(a: Any) -> str | None:
    """The request step's config and metrics to wandb (its venv has no wandb)."""
    import json
    import os

    from robots.mujoco_lab.curve import _sha
    from robots.mujoco_lab.tracking import log_run

    cfg = json.load(open(os.path.join(a.grasps, "run_config.json")))
    m = json.load(open(os.path.join(a.grasps, "run_metrics.json")))
    url = log_run("curve-graspgen-request", dict(cfg, grasps=a.grasps, index_sha256=_sha(os.path.join(a.grasps, "index.jsonl"))), m)
    print(f"wandb: {url}", flush=True)
    return url


def run(a: Any) -> str:
    """The pilot's run: records_validation.jsonl and records_test.jsonl (one graspgen_case record per case, by object as
    each ends), run_config.json and run_metrics.json (times, counts and errors only; wandb). A gate that fails stops the
    run (exit 3). Returns --out."""
    import json
    import multiprocessing as mp
    import os
    from collections import defaultdict

    from robots.mujoco_lab.curve import SIDES, _peak_rss_mb, _sha, records_path
    from robots.mujoco_lab.estimation_check import _revision
    from robots.mujoco_lab.identify_fit import require_bundle_generator
    from robots.mujoco_lab.tracking import log_run

    os.makedirs(a.out, exist_ok=True)
    if any(os.path.exists(records_path(a.out, side)) for side in SIDES):
        raise SystemExit(f"{a.out}: a records file exists: pick a new --out")
    recs1, cfg1, dirs = _stage1(a)
    generator = require_bundle_generator(cfg1["bundle"])
    recs = select_cases(recs1, a.cases, a.objects, a.sides, a.per_object)
    gcfg = json.load(open(os.path.join(a.grasps, "run_config.json")))
    config = {**vars(a), **_revision(), "generator": generator, "bundle": cfg1["bundle"], "seed": cfg1["seed"],
              "stage1_code_revision": cfg1["code_revision"],
              "stage1_records_sha256": {d: _sha(os.path.join(d, "records.jsonl")) for d in dirs},
              "cases": [(r["object"], int(r["idx"])) for r in recs], "grasps_index_sha256": _sha(os.path.join(a.grasps, "index.jsonl")),
              "grasps_config": gcfg, "clouds_index_sha256": _sha(os.path.join(gcfg["clouds"], "index.jsonl")),
              "depth": DEPTH, "vertical_deg": VERTICAL_DEG, "dedup": [DEDUP_M, DEDUP_DEG], "keep_rows": KEEP_ROWS,
              "open_gap": OPEN_GAP, "pad_above": PAD_ABOVE, "gate": GATE}
    with open(os.path.join(a.out, "run_config.json"), "w") as f:
        json.dump(config, f, indent=1)
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        by[r["object"]].append(r)
    jobs = [(o, items, {"bundle": os.path.join(cfg1["bundle"], "folds", f"fold_{items[0]['fold']}"), "seed": cfg1["seed"],
                        "replay": bool(a.replay), "grasps": a.grasps}) for o, items in by.items()]
    t0, stop = time.time(), None
    outs = {side: open(records_path(a.out, side), "w") for side in SIDES}
    try:
        pool = None if a.workers == 1 else mp.get_context("spawn").Pool(a.workers)
        results = map(graspgen_object, jobs) if pool is None else pool.imap_unordered(graspgen_object, jobs)
        try:
            for res in results:
                for r in res["records"]:
                    outs[r["side"]].write(json.dumps(r) + "\n")
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
    metrics = dict(_metrics(done), of_cases=len(recs), stop=stop, wall_seconds=round(time.time() - t0, 1), peak_rss_mb=_peak_rss_mb())
    metrics["wandb"] = log_run("curve-graspgen", config, metrics)
    with open(os.path.join(a.out, "run_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print(f"done: {json.dumps(metrics)}", flush=True)
    if stop:
        raise SystemExit(3)
    return a.out


def _frame_total(recs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": sum(int(r["frame"]["n"]) for r in recs)}
    for name in ("palm", "tip"):
        out[name] = {k: sum(int(r["frame"][name][k]) for r in recs) for k in ("opening_found", "width_found", "both", "within_5mm")}
        out[name]["n"] = out["n"]
        gaps = [abs(x[name]["opening"] - x[name]["width"]) for r in recs for x in r["frame"]["rows"]
                if np.isfinite(x[name]["opening"]) and np.isfinite(x[name]["width"])]
        gaps_b = [abs(x[name]["opening"] - x[name]["width_b"]) for r in recs for x in r["frame"]["rows"]
                  if np.isfinite(x[name]["opening"]) and np.isfinite(x[name]["width_b"])]
        out[name].update(median_abs_mm=float(np.median(gaps) * 1000) if gaps else None,
                         within_5mm_b=int(sum(g <= 0.005 + 1e-12 for g in gaps_b)), both_b=len(gaps_b),
                         median_abs_b_mm=float(np.median(gaps_b) * 1000) if gaps_b else None)
    return out


def smoke_report(recs: list[dict[str, Any]], pose: list[dict[str, Any]], full: dict[str, int], one: str = "hope__milk_carton") -> dict[str, Any]:
    """The smoke's numbers, no label (the smoke runs test cases): the funnel by object and in all (both rules, and the
    reasons rows leave), the frame-mapping check in all and on ``one`` object, the 2-seed gate, the gate row's final
    state and trace against the pose set's across_0.5 row of the same case (equal, differ, missing), rows simulated,
    row errors, times, and the full run's projection (each object's mean case time times its ``full`` cases)."""
    from collections import Counter, defaultdict

    done = [r for r in recs if r.get("counts")]
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in done:
        by[r["object"]].append(r)
    funnel_t = {o: {k: sum(int(r["counts"][k]) for r in rs) for k in FUNNEL + FUNNEL_B} for o, rs in sorted(by.items())}
    funnel_t["all"] = {k: sum(v[k] for o, v in funnel_t.items()) for k in FUNNEL + FUNNEL_B}
    why = {o: dict(Counter(str(x.get("why")) for r in rs for x in r["rows"] if not x.get("gate") and x.get("why"))) for o, rs in sorted(by.items())}
    why_b = {o: dict(Counter(str(x.get("why_b")) for r in rs for x in r["rows"] if not x.get("gate") and x.get("why_b")))
             for o, rs in sorted(by.items())}
    frame = {"all": _frame_total(done)}
    if one in by:
        frame[one] = _frame_total(by[one])
    g = exact_gate(done, pose)
    gate = {k: g[k] for k in ("cases", "equal", "differ", "missing")}
    rep = [r["replay"] for r in done if r.get("replay")]
    rows = [x for r in done for x in r["rows"] if "seconds" in x]
    mean = {o: float(np.mean([float(r["seconds"]) for r in rs])) for o, rs in by.items()}
    proj = sum(mean.get(o, max(mean.values(), default=0.0)) * n for o, n in full.items()) / 3600
    return {"cases": len(recs), "with_rows": len(done), "skipped": sum(bool(r.get("skipped")) for r in recs),
            "stops": [f"{r['object']} {r['idx']}: {r['stop']}" for r in recs if r.get("stop")],
            "case_errors": sum(bool(r.get("error")) for r in recs), "funnel": funnel_t, "why": why, "why_b": why_b, "frame": frame,
            "gate": gate, "replay": {"rows": sum(x["rows"] for x in rep), "mismatches": sum(len(x["mismatches"]) for x in rep),
                                     "stops": sum(bool(x.get("stop")) for x in rep)},
            "rows_simulated": len(rows), "row_errors": sum(bool(x.get("error")) for x in rows),
            "case_seconds": {o: round(v, 1) for o, v in sorted(mean.items())}, "full_cases": full, "projection_cpu_h": proj}


def smoke_markdown(sm: dict[str, Any]) -> str:
    def cell(v: Any) -> str:
        return "–" if v is None else (f"{v:.1f}" if isinstance(v, float) else str(v))

    L = ["# GraspGen pilot: the smoke (counts only, no label)", "",
         f"{sm['cases']} cases ({sm['with_rows']} with rows, {sm['skipped']} skipped, {len(sm['stops'])} stops, "
         f"{sm['case_errors']} case errors). Rows simulated {sm['rows_simulated']}, row errors {sm['row_errors']}.", "",
         "## The funnel (rows left after each step; the variant's steps after the mapping)", "",
         "| object | " + " | ".join(FUNNEL + FUNNEL_B) + " |", "|---|" + "---|" * len(FUNNEL + FUNNEL_B)]
    L += [f"| {o} | " + " | ".join(str(v[k]) for k in FUNNEL + FUNNEL_B) + " |" for o, v in sm["funnel"].items()]
    L += ["", "## Why rows leave (the design's rule, then the variant's)", ""]
    L += [f"- {o}: {dict(sorted(sm['why'][o].items()))}; variant {dict(sorted(sm['why_b'][o].items()))}" for o in sm["why"]]
    L += ["", "## The frame-mapping check (near-vertical grasps; GraspGen's opening against the lab's width)", "",
          "| set | candidate | grasps | opening found | width found | both | within 5 mm | median gap mm | "
          "within 5 mm (between-pad width) | median gap mm (between-pad width) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s, f in sm["frame"].items():
        for name in ("palm", "tip"):
            x = f[name]
            L.append(f"| {s} | {name} | {f['n']} | {x['opening_found']} | {x['width_found']} | {x['both']} | {x['within_5mm']} | "
                     f"{cell(x['median_abs_mm'])} | {x['within_5mm_b']} of {x['both_b']} | {cell(x['median_abs_b_mm'])} |")
    g, r = sm["gate"], sm["replay"]
    L += ["", f"**Gate row** (the pose set's across_0.5 pinch through this module, against the pose set's record of the same case): "
          f"{g['cases']} cases, {g['equal']} equal final state and trace, {g['differ']} differ, {g['missing']} missing.",
          f"**2-seed gate:** {r['rows']} rows replayed, {r['mismatches']} mismatches, {r['stops']} stops.", "",
          f"**Projection:** {sm['projection_cpu_h']:.2f} CPU-h for {sum(sm['full_cases'].values())} cases "
          f"(mean case seconds by object: {sm['case_seconds']})."]
    if sm["stops"]:
        L += ["", "Stops: " + "; ".join(sm["stops"])]
    return "\n".join(L) + "\n"


def smoke(a: Any) -> dict[str, Any]:
    """The smoke report on finished smoke runs (smoke_report): the pose set's records (--poseset, test and validation
    runs) for the gate row; the full run's cases per object from Stage 1 (the smoke's objects and sides).
    smoke_summary.{json,md} in --out (refuses an existing one), wandb."""
    import json
    import os

    from robots.mujoco_lab.curve import SIDES, load_stage1, records_path, stage2_skip
    from robots.mujoco_lab.tracking import log_run

    os.makedirs(a.out, exist_ok=True)
    jpath = os.path.join(a.out, "smoke_summary.json")
    if os.path.exists(jpath):
        raise SystemExit(f"{jpath} exists: pick a new --out")
    dirs = a.runs.split(",")
    for d in dirs:
        if not os.path.exists(os.path.join(d, "run_metrics.json")):
            raise SystemExit(f"{d}: the run has not finished")
    recs = [json.loads(line) for d in dirs for side in SIDES for line in open(records_path(d, side))]
    pose = [json.loads(line) for d in a.poseset.split(",") for side in SIDES if os.path.exists(records_path(d, side))
            for line in open(records_path(d, side))]
    cfg = json.load(open(os.path.join(dirs[0], "run_config.json")))
    recs1, _ = load_stage1(list(cfg["stage1_records_sha256"]))
    sides = {(r["object"], r["side"]) for r in recs}
    full: dict[str, int] = {}
    for r in recs1:
        if (r["object"], r["side"]) in sides and not stage2_skip(r):
            full[r["object"]] = full.get(r["object"], 0) + 1
    sm = smoke_report(recs, pose, full)
    sm["wandb"] = log_run("curve-graspgen-smoke", {"runs": a.runs, "poseset": a.poseset},
                          {k: v for k, v in sm.items() if k not in ("why", "why_b", "frame", "funnel", "stops")})
    with open(jpath, "w") as f:
        json.dump(sm, f, indent=1)
    with open(os.path.join(a.out, "smoke_summary.md"), "w") as f:
        f.write(smoke_markdown(sm))
    print(smoke_markdown(sm), flush=True)
    return sm


PILOT_OBJECTS = ("handal__hammer_1", "handal__hammer_7", "hope__oatmeal_raisin_cookies", "hope__milk_carton", "ycb__bowl", "ycb__pitcher",
                 "fruits_veggies__orange1", "hot3d__ceramic_mug", "fruits_veggies__lemon1", "hot3d__mug")   # design 3.1's ten, in order
RUN_KEYS = ("bundle", "seed", "stage1_records_sha256", "code_revision", "grasps", "grasps_index_sha256", "clouds_index_sha256", "replay",
            "depth", "vertical_deg", "keep_rows")
ARM_STYLE = {"gg": ("#2a78d6", "o", True), "ggb": ("#2a78d6", "o", False), "c10": ("#eb6834", "s", True), "c3": ("#eb6834", "s", False),
             "oracle": ("#1baf7a", "D", True), "r4": ("#eda100", "*", True)}     # colour, marker, filled (validated palette slots 1-4)


def _order(o: str) -> tuple[int, str]:
    return (PILOT_OBJECTS.index(o), o) if o in PILOT_OBJECTS else (len(PILOT_OBJECTS), o)


def summary(a: Any) -> dict[str, Any]:
    """The readings (design 3 and 3.1) from finished pilot runs: graspgen_summary.{json,md} and graspgen_points.png in
    --out (refuses an existing summary), wandb. Refuses a run without EXIT 0 or with a stop, runs that differ, inputs that
    changed since the runs or the summaries that read them (Stage 1, the grasps, Stage 2, the oracle arm's and the pose
    set's test records), oracle or pose-set runs on another bundle, seed or Stage 1, and a pilot case an arm lacks. The
    pilot's records are opened once. A gate row whose final state or trace differs from the pose set's (the exact gate)
    writes graspgen_summary_stop.json and exits 3 before any reading."""
    import json
    import os

    from robots.mujoco_lab.curve import BOOT_DRAWS, BOOT_SEED, EPS, SIDES, _sha, load_stage2, records_path
    from robots.mujoco_lab.curve_oracle import _finished
    from robots.mujoco_lab.curve_poseset import _key, _load_lines, _read_once
    from robots.mujoco_lab.estimation_check import _revision
    from robots.mujoco_lab.tracking import log_run

    os.makedirs(a.out, exist_ok=True)
    jpath, spath = os.path.join(a.out, "graspgen_summary.json"), os.path.join(a.out, "graspgen_summary_stop.json")
    if os.path.exists(jpath) or os.path.exists(spath):
        raise SystemExit(f"{a.out}: a summary exists: pick a new --out")
    lk = json.load(open(a.locked))["lock"]
    k0, k4 = lk["R0"]["key"], lk["R4"]["key"]
    dirs = a.runs.split(",")
    cfgs = [_finished(d) for d in dirs]
    for k in RUN_KEYS:
        if len({json.dumps(c.get(k), sort_keys=True) for c in cfgs}) > 1:
            raise SystemExit(f"the pilot runs differ in {k}")
    cfg = cfgs[0]
    for d, h in cfg["stage1_records_sha256"].items():
        if _sha(os.path.join(d, "records.jsonl")) != h:
            raise SystemExit(f"{d}: the Stage 1 records changed since the pilot runs")
    if _sha(os.path.join(cfg["grasps"], "index.jsonl")) != cfg["grasps_index_sha256"]:
        raise SystemExit(f"{cfg['grasps']}: the grasps index is not the one the pilot runs read")
    s2dirs = a.stage2.split(",")
    prov2 = json.load(open(a.stage2_summary))["provenance"]["records_sha256"]
    for d in s2dirs:
        for side in SIDES:
            if _sha(records_path(d, side)) != prov2.get(f"{d} {side}"):
                raise SystemExit(f"{d}: the Stage 2 {side} records are not the ones the Stage 2 summary read")
    s2t, s2cfg = load_stage2(s2dirs, ("test",))
    s2v, _ = load_stage2(s2dirs, ("validation",))
    if s2cfg["stage1_records_sha256"] != cfg["stage1_records_sha256"]:
        raise SystemExit("the pilot runs and Stage 2 read different Stage 1 records")
    oprov = json.load(open(a.oracle_summary))["provenance"]
    if list(oprov.get("keys") or []) != [k0, k4]:
        raise SystemExit(f"the oracle summary's keys {oprov.get('keys')} are not the lock's {[k0, k4]}")
    odirs, ovdirs = a.oracle.split(","), [d for d in (a.oracle_validation or "").split(",") if d]
    pdirs, pvdirs = a.poseset.split(","), [d for d in (a.poseset_validation or "").split(",") if d]
    pprov = json.load(open(a.poseset_summary))["provenance"]
    for name, ds, read in (("oracle", odirs, oprov["oracle_records_sha256"]), ("pose-set", pdirs, pprov["records_sha256"])):
        if set(ds) != set(read):
            raise SystemExit(f"the {name} test runs are not the ones the {name} summary read")
        for d in ds:
            if _sha(records_path(d, "test")) != read[d]:
                raise SystemExit(f"{d}: the {name} test records are not the ones the {name} summary read")
    for d in odirs + ovdirs + pdirs + pvdirs:
        c = _finished(d)
        for k in ("bundle", "seed", "stage1_records_sha256"):
            if json.dumps(c.get(k), sort_keys=True) != json.dumps(cfg.get(k), sort_keys=True):
                raise SystemExit(f"{d}: its {k} is not the pilot runs' (an oracle or pose-set run on other cases)")
    paths = [records_path(d, side) for d in dirs for side in SIDES]
    grecs, gsha = _read_once(paths)                                             # the pilot's records, once
    if any(r.get("stop") for r in grecs):
        raise SystemExit("a pilot record holds a stop")
    if len({_key(r) for r in grecs}) != len(grecs):
        raise SystemExit("a case appears twice in the pilot records")
    grecs.sort(key=lambda r: (_order(r["object"]), int(r["idx"])))
    prec = _load_lines([records_path(d, "test") for d in pdirs] + [records_path(d, "validation") for d in pvdirs])
    orec = _load_lines([records_path(d, "test") for d in odirs] + [records_path(d, "validation") for d in ovdirs])
    arms = {"pose-set": prec, "oracle": orec, "Stage 2": s2t + s2v}
    for name, recs in arms.items():
        if len({_key(r) for r in recs}) != len(recs):
            raise SystemExit(f"a case appears twice in the {name} records")
    arms = {name: {_key(r): r for r in recs} for name, recs in arms.items()}
    joined = []
    for r in grecs:
        k = _key(r)
        for name, by in arms.items():
            if k not in by or by[k]["side"] != r["side"]:
                raise SystemExit(f"{k[0]} {k[1]}: no {name} record of this case on the {r['side']} side")
        joined.append({"g": r, "pose": arms["pose-set"][k], "oracle": arms["oracle"][k], "stage2": arms["Stage 2"][k]})
    me = _revision()
    prov = {"runs": a.runs, "records_sha256": gsha, "code_revision": cfg["code_revision"], "summary_code_revision": me["code_revision"],
            "summary_dirty": me["dirty"], "grasps": cfg["grasps"], "grasps_index_sha256": cfg["grasps_index_sha256"], "stage2": a.stage2,
            "stage2_summary": f"{a.stage2_summary} (sha256 {_sha(a.stage2_summary)})", "oracle": a.oracle, "oracle_validation": a.oracle_validation,
            "oracle_summary": f"{a.oracle_summary} (sha256 {_sha(a.oracle_summary)})", "poseset": a.poseset,
            "poseset_validation": a.poseset_validation, "poseset_summary": f"{a.poseset_summary} (sha256 {_sha(a.poseset_summary)})",
            "validation_records_sha256": {d: _sha(records_path(d, "validation")) for d in ovdirs + pvdirs},
            "locked": f"{a.locked} (sha256 {_sha(a.locked)})", "smoke": f"{a.smoke} (sha256 {_sha(a.smoke)})",
            "stage1": sorted(cfg["stage1_records_sha256"]), "bootstrap": f"{BOOT_DRAWS} object draws, seed {BOOT_SEED}, 95 % percentile interval",
            "eps": EPS, "keys": [k0, k4]}
    gate = exact_gate(grecs, prec)
    if gate["differ"]:
        with open(spath, "w") as f:
            json.dump({"exact_gate": gate, "provenance": prov}, f, indent=1)
        print(f"STOP: {gate['differ']} gate rows differ from the pose set's; wrote {spath}", flush=True)
        raise SystemExit(3)
    paired, n_dropped = pilot_paired(joined)
    vs = sorted(pilot_views(paired, k0, k4), key=lambda v: (v["side"] != "test", _order(v["object"]), v["idx"]))
    vt, vv = [v for v in vs if v["side"] == "test"], [v for v in vs if v["side"] == "validation"]
    entries = load_grasps(cfg["grasps"])
    done = [r for r in grecs if r.get("counts")]
    angles: dict[str, list[int]] = {}
    for r in done:                                                              # the cases whose grasps went through the funnel
        e = entries.get(_key(r))
        if e is None or e["error"] or e["grasps"] is None:
            raise SystemExit(f"{r['object']} {r['idx']}: its grasps do not load ({'no index line' if e is None else e['error']})")
        angles[r["object"]] = [x + y for x, y in zip(angles.get(r["object"], [0, 0, 0, 0]), angle_histogram(e["grasps"]))]
    angles["all"] = [sum(h[i] for h in angles.values()) for i in range(4)]
    sim_rows = [x for r in done for x in r["rows"] if "seconds" in x]
    counts = {"cases": len(grecs), "skipped": sum(bool(r.get("skipped")) for r in grecs), "pilot_dropped": sum(bool(r["dropped"]) for r in grecs),
              "case_errors": sum(bool(r.get("error")) for r in grecs), "rows_simulated": len(sim_rows),
              "row_errors": sum(bool(x.get("error")) for x in sim_rows), "row_artifacts": sum(bool(x.get("artifact")) for x in sim_rows),
              "paired": {"test": len(vt), "validation": len(vv)}, "paired_dropped": n_dropped,
              "objects": {"test": len({v["object"] for v in vt}), "validation": len({v["object"] for v in vv})}}
    readings = {"1_funnel": funnel_table(grecs), "1_angles": angles, "2_test": pilot_block(vt),
                "2_test_no_thin": pilot_block([v for v in vt if not v["thin"]]), "2_validation": pilot_block(vv), "2_by_object": by_object(vs),
                "3_hammer": hammer_line(vt), "4_objects": object_line(vt), "5_tags": {name: placing_tags(vt, name) for name, _ in RULES},
                "5_tags_validation": {name: placing_tags(vv, name) for name, _ in RULES}}
    smoke_rep = json.load(open(a.smoke))
    gates = {"smoke": {"replay": smoke_rep["replay"], "gate": smoke_rep.get("gate"), "wandb": smoke_rep.get("wandb")}, "exact": gate,
             "label": label_gate([{"object": j["g"]["object"], "idx": int(j["g"]["idx"]), "prows": j["pose"].get("rows") or [],
                                   "rows": [r for r in j["g"].get("rows") or [] if not r.get("gate")]} for j in joined])}
    one = "hope__milk_carton"
    disclosures = {"frame": {"all": _frame_total(done), one: _frame_total([r for r in done if r["object"] == one])},
                   "thin_cases": sum(v["thin"] for v in vs), "pad_height_m": {"lab": 2 * float(PAD_HALF[2]), "graspgen": 2 * PANDA_BOX[2]}}
    rep = {"counts": counts, "readings": readings, "gates": gates, "disclosures": disclosures,
           "figure": {"by_object": readings["2_by_object"], "test": readings["2_test"]["rates"], "validation": readings["2_validation"]["rates"]},
           "provenance": prov}
    b = readings["2_test"]
    rep["wandb"] = log_run("curve-graspgen-summary", {"runs": a.runs, "stage2": a.stage2, "oracle": a.oracle, "poseset": a.poseset},
                           {"counts": counts, "rule": b["rule"], "rule_b": b["rule_b"],
                            "gg_vs_c10": {k: b["gg_vs_c10"][k] for k in ("success", "irr", "give_up")} if vt else None,
                            "ggb_vs_c10": {k: b["ggb_vs_c10"][k] for k in ("success", "irr", "give_up")} if vt else None,
                            "exact_gate": {k: gate[k] for k in ("cases", "equal", "differ", "missing")},
                            "label_gate": {k: gates["label"][k] for k in ("compared", "agree", "differ", "missing")}})
    with open(jpath, "w") as f:
        json.dump(rep, f, indent=1)
    with open(os.path.join(a.out, "graspgen_summary.md"), "w") as f:
        f.write(summary_markdown(rep))
    pilot_figure(rep, os.path.join(a.out, "graspgen_points.png"))
    print(f"wrote {a.out}", flush=True)
    return rep


def _pc(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def _ci(b: dict[str, Any], k: str) -> str:
    x = b.get(k)
    return "–" if not x else f"{100 * x['diff']:+.2f} [{100 * x['lo']:+.2f}, {100 * x['hi']:+.2f}]"


def _med(x: list[float]) -> str:
    return "–" if not x else f"{100 * float(np.median(x)):.1f} cm (n = {len(x)})"


def _md_block(name: str, b: dict[str, Any]) -> list[str]:
    L = [f"**{name}**: {b['cases']} paired cases on {b['objects']} objects.", "",
         "| arm | success % | irreversible % | give-up % |", "|---|---|---|---|"]
    L += [f"| {ARM_NAMES[k]} | {_pc(b['rates'][k]['success'])} | {_pc(b['rates'][k]['irr'])} | {_pc(b['rates'][k]['give_up'])} |" for k in ARMS]
    if b["cases"]:
        for k, rule, lab in (("gg_vs_c10", "rule", "the design's rule"), ("ggb_vs_c10", "rule_b", "the between-pad variant")):
            x, t = b[k], b["success_2x2" if k == "gg_vs_c10" else "success_2x2_b"]
            L += ["", f"- GraspGen ({lab}) − the 10-pinch ceiling, points, 95 % object bootstrap: success {_ci(x, 'success')}, "
                  f"irreversible {_ci(x, 'irr')}, give-up {_ci(x, 'give_up')}. The 2-point rule (pilot, no verdict): {b[rule]}.",
                  f"- Success, case by case: both {t['both']}, GraspGen only {t['graspgen_only']}, the 10-pinch only {t['c10_only']}, "
                  f"neither {t['neither']}."]
    return L


def summary_markdown(rep: dict[str, Any]) -> str:
    R, G, D, c = rep["readings"], rep["gates"], rep["disclosures"], rep["counts"]
    F = FUNNEL + FUNNEL_B
    L = ["# The GraspGen menu pilot: the readings (a pilot: no verdict is claimed)", "",
         f"{c['cases']} cases ({c['skipped']} skipped, {c['case_errors']} case errors, {c['pilot_dropped']} dropped by the pilot). "
         f"Rows simulated {c['rows_simulated']}, row errors {c['row_errors']}, row artifacts {c['row_artifacts']}. Paired: "
         f"{c['paired']['test']} test cases on {c['objects']['test']} objects, {c['paired']['validation']} validation cases on "
         f"{c['objects']['validation']} objects; dropped {c['paired_dropped']}.", "",
         "## 1. The funnel and the approach angles", "", "| object | cases | " + " | ".join(F) + " | lost at the approach step |",
         "|---|---|" + "---|" * len(F) + "---|"]
    L += [f"| {o} | {x['cases']} | " + " | ".join(str(x[k]) for k in F) + f" | {_pc(x['lost_at_approach'])} % |" for o, x in R["1_funnel"].items()]
    L += ["", "| object | " + " | ".join(ANGLE_BINS) + " |", "|---|" + "---|" * len(ANGLE_BINS)]
    L += [f"| {o} | " + " | ".join(str(v) for v in h) + " |" for o, h in R["1_angles"].items()]
    L += ["", "The cloud holds no table points, so GraspGen also proposes grasps from below the object.", "",
          "## 2. The ceilings beside the four arms", ""]
    L += _md_block("Test (the pilot interval)", R["2_test"]) + [""]
    L += [f"Without thin plastic: {D['thin_cases']} thin-plastic cases, so the values equal the all-object values." if not D["thin_cases"]
          else "Without thin plastic:", ""]
    if D["thin_cases"]:
        L += _md_block("Test without thin plastic", R["2_test_no_thin"]) + [""]
    vo = [o for o, x in R["2_by_object"].items() if x["side"] == "validation"]
    L += _md_block(f"Validation, apart ({', '.join(vo) or 'none'})", R["2_validation"]) + ["", "By object (success % of each arm):", "",
          "| object (side) | n | " + " | ".join(ARM_NAMES[k] for k in ARMS) + " |", "|---|---|" + "---|" * len(ARMS)]
    L += [f"| {o} ({x['side']}) | {x['n']} | " + " | ".join(_pc(x[k]['success']) for k in ARMS) + " |" for o, x in R["2_by_object"].items()]
    h = R["3_hammer"]
    L += ["", "## 3. The hammers", "", f"{h['cases']} cases on {', '.join(h['objects']) or 'no object'}. The head end is the hammer's "
          "wide end on its hulls (" + ", ".join(f"{o} f {HAMMER_HEAD_F[o]:.0f}" for o in h["objects"]) + "), not the drawn CoM's side."]
    L += [f"- {ARM_NAMES[n]}: " + ", ".join(f"{k} {v}" for k, v in h["classes"][n].items()) for n, _ in RULES]
    L += ["- The 10-pinch ceiling: " + ", ".join(f"{k} {v}" for k, v in h["c10_classes"].items()),
          f"- Clean kept rows that placed a hammer: median distance from the balance point {_med(h['placed']['balance_m'])}, from the head end "
          f"{_med(h['placed']['head_m'])}. Rows that did not place it: {_med(h['not_placed']['balance_m'])} and {_med(h['not_placed']['head_m'])}."]
    for pc in h["per_case"]:
        rows = "; ".join(f"{r['id']} at f {r['f']:.2f} ({100 * r['balance_m']:.1f} cm from the balance point, {100 * r['head_m']:.1f} cm from "
                         f"the head end, {r['tag']}, {'+'.join(r['rules'])})" for r in pc["placed"]) or "none"
        L.append(f"  - {pc['object']} {pc['idx']} (balance point at f {pc['com_f']:.2f}): {pc['classes']['gg']} / {pc['classes']['ggb']} "
                 f"(design / variant), 10-pinch: {pc['c10_class']}; placing rows: {rows}.")
    L += ["", "## 4. The bowl and the pitcher (both rules)", ""]
    for o, x in R["4_objects"].items():
        for n, _ in RULES:
            y = x[n]
            L.append(f"- {o}, {ARM_NAMES[n]}: {y['cases_with_row']} of {y['cases']} cases with a kept row; {y['rows']} kept rows "
                     f"{y['tags']}; {y['clean']} clean, {y['placing']} placing {y['placing_tags']}; the ceiling places {y['cases_placed']} of "
                     f"{y['cases']}; labels {y['labels']}.")
        L.append(f"- {o}, the 10-pinch ceiling: places {x['c10_placed']}, {x['c10_rows']} feasible rows.")
    L += ["", "## 5. Where the placing rows sit (both rules)", ""]
    for side, key in (("test", "5_tags"), ("validation", "5_tags_validation")):
        for n, _ in RULES:
            t = R[key][n]
            L.append(f"- {side}, {ARM_NAMES[n]}: placing rows by tag {t['rows']}; cases by tag {t['cases']}; the gain cases (GraspGen places, the "
                     f"10-pinch ceiling does not) {t['gain']['cases']}, by tag {t['gain']['tags']}; the loss cases {t['loss']['cases']}.")
    s, e, lg = G["smoke"], G["exact"], G["label"]
    L += ["", "## 6. The gates", "",
          f"- The 2-seed gate (the smoke): {s['replay']['rows']} rows replayed, {s['replay']['mismatches']} mismatches, {s['replay']['stops']} stops "
          f"(wandb {s.get('wandb')}).",
          f"- The exact gate (the full run): {e['cases']} gate rows, {e['equal']} equal final state and trace, {e['differ']} differ, {e['missing']} missing.",
          f"- The tolerance label gate (no stop): {lg['compared']} kept rows within 1 cm and 10° of a current pinch, {lg['agree']} agree with the pose "
          f"set's label, {lg['differ']} differ, {lg['missing']} missing."]
    L += [f"  - {x['object']} {x['idx']}: {x['row']} {x['graspgen']} against {x['pose_row']} {x['pose']}" for x in lg["differences"]]
    L += ["", "## Disclosures", ""]
    for name, fr in D["frame"].items():
        for cand in ("palm", "tip"):
            x = fr[cand]
            L.append(f"- The frame check, {name}, {cand}: {fr['n']} grasps within 30° of down, opening found {x['opening_found']}, within 5 mm "
                     f"{x['within_5mm']} of {x['both']} (the design's width), {x['within_5mm_b']} of {x['both_b']} (the between-pad width).")
    L += [f"- The pad sizes: the lab pad is {100 * D['pad_height_m']['lab']:.1f} cm tall, GraspGen's finger {100 * D['pad_height_m']['graspgen']:.1f} cm.",
          f"- Thin plastic: {D['thin_cases']} paired cases."]
    return "\n".join(L) + "\n"


def pilot_figure(rep: dict[str, Any], path: str) -> None:
    """Each object's success and irreversible damage under the six arms (two panels, one row per object, the test objects
    above the validation ones), from the summary's figure data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = rep["figure"]["by_object"]
    objs = list(by)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 0.42 * len(objs) + 1.6), sharey=True)
    ys = {o: -(k + (1 if by[o]["side"] != "test" else 0)) for k, o in enumerate(objs)}
    off = dict(zip(ARMS, np.linspace(-0.27, 0.27, len(ARMS))))
    for ax, k, title in ((axes[0], "success", "task success (% of cases)"), (axes[1], "irr", "irreversible damage (% of cases)")):
        for arm in ARMS:
            col, m, filled = ARM_STYLE[arm]
            x = [100 * (by[o][arm][k] or 0.0) for o in objs]
            y = [ys[o] + off[arm] for o in objs]
            ax.scatter(x, y, marker=m, s=46 if m != "*" else 90, facecolors=col if filled else "white", edgecolors=col if not filled else "#1f1e1c",
                       linewidths=1.3 if not filled else 0.5, zorder=3, label=ARM_NAMES[arm])
        ax.set_xlim(-4, 104)
        ax.set_xlabel(title)
        ax.grid(True, axis="x", color="#e6e5e0", linewidth=0.8, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks([ys[o] for o in objs])
    axes[0].set_yticklabels([f"{o.split('__', 1)[-1]}{' (validation)' if by[o]['side'] != 'test' else ''}, n = {by[o]['n']}" for o in objs],
                            fontsize=8)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(1.05, 1.0 + 1.6 / (0.42 * len(objs) + 1.6)), ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure(a: Any) -> str:
    """The figure again from a finished summary's graspgen_summary.json (its figure data only; no records file is opened)."""
    import json
    import os

    from robots.mujoco_lab.curve import _sha

    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pilot_figure(json.load(open(a.summary)), a.out)
    print(f"wrote {a.out} from {a.summary} (sha256 {_sha(a.summary)})", flush=True)
    return a.out


def main(argv: list[str] | None = None) -> Any:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, hlp in (("clouds", "the cloud step: each case's hull cloud at its rest pose"), ("run", "the funnel and the simulations")):
        r = sub.add_parser(name, help=hlp)
        r.add_argument("--stage1", required=True, help="comma-separated Stage 1 run dirs")
        r.add_argument("--out", required=True)
        r.add_argument("--cases", default=None, help="comma-separated object:idx cases")
        r.add_argument("--objects", default=None, help="comma-separated objects (with --sides)")
        r.add_argument("--sides", default="test")
        r.add_argument("--per-object", type=int, default=None, help="the first N cases of each object by idx")
        if name == "run":
            r.add_argument("--grasps", required=True, help="the request step's output dir")
            r.add_argument("--workers", type=int, default=1)
            r.add_argument("--replay", action="store_true", help="the 2-seed gate: every simulated row again at the second seed")
    q = sub.add_parser("log-request", help="the request step's config and metrics to wandb")
    q.add_argument("--grasps", required=True)
    s = sub.add_parser("smoke", help="the smoke report (counts only)")
    s.add_argument("--runs", required=True, help="comma-separated smoke run dirs")
    s.add_argument("--poseset", required=True, help="comma-separated pose-set run dirs (test and validation)")
    s.add_argument("--out", required=True)
    m = sub.add_parser("summary", help="the readings from finished pilot runs (opens the pilot's records once)")
    m.add_argument("--runs", required=True, help="comma-separated pilot run dirs")
    m.add_argument("--poseset", required=True, help="comma-separated pose-set test run dirs (the pose-set summary's)")
    m.add_argument("--poseset-validation", default="", help="comma-separated pose-set validation run dirs")
    m.add_argument("--poseset-summary", required=True, help="the pose set's poseset_summary.json")
    m.add_argument("--oracle", required=True, help="comma-separated oracle-arm test run dirs (the oracle summary's)")
    m.add_argument("--oracle-validation", default="", help="comma-separated oracle-arm validation run dirs")
    m.add_argument("--oracle-summary", required=True, help="the oracle arm's oracle_summary.json")
    m.add_argument("--stage2", required=True, help="comma-separated Stage 2 run dirs")
    m.add_argument("--stage2-summary", required=True, help="Stage 2's stage2_summary.json")
    m.add_argument("--locked", required=True, help="Stage 2's locked.json")
    m.add_argument("--smoke", required=True, help="the smoke report's smoke_summary.json (the 2-seed gate)")
    m.add_argument("--out", required=True)
    g = sub.add_parser("figure", help="the figure again from a finished summary's json (reads no records)")
    g.add_argument("--summary", required=True, help="graspgen_summary.json")
    g.add_argument("--out", required=True, help="the png to write")
    a = p.parse_args(argv)
    return {"clouds": clouds, "run": run, "log-request": log_request, "smoke": smoke, "summary": summary, "figure": figure}[a.cmd](a)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
