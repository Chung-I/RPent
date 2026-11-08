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
    return bool(r.get("kept") or r.get("kept_b") or r.get("gate"))


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


def _gate_row(env: Any) -> dict[str, Any]:
    q = env.proposals((GATE[1],), turn=0.0)[0]
    return {"id": f"gate_{GATE[0]}_{GATE[1]:g}", "rank": -1, "gate": True, "kept": False, "kept_b": False, "f": float(q.s),
            "point": [float(x) for x in q.point], "yaw": float(q.yaw), "width": float(q.width)}


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
        rows.append(_gate_row(env))
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


# ---------------------------------------------------------------- the commands

FUNNEL = ("generated", "vertical", "mapped", "width", "centred", "reach", "distinct", "kept")
FUNNEL_B = ("width_b", "pads_free", "reach_b", "distinct_b", "kept_b")


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

    from robots.mujoco_lab.curve_poseset import _key

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
    pidx = {_key(p): {x["id"]: x for x in p["rows"]} for p in pose}
    gate = {"cases": 0, "equal": 0, "differ": 0, "missing": 0}
    for r in done:
        g = next((x for x in r["rows"] if x.get("gate") and "seconds" in x), None)
        if g is None:
            continue
        gate["cases"] += 1
        p = pidx.get(_key(r), {}).get("across_0.5")
        if p is None or "seconds" not in p:
            gate["missing"] += 1
        elif (p.get("qpos_sha"), p.get("trace_sha")) == (g.get("qpos_sha"), g.get("trace_sha")):
            gate["equal"] += 1
        else:
            gate["differ"] += 1
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
    a = p.parse_args(argv)
    return {"clouds": clouds, "run": run, "log-request": log_request, "smoke": smoke}[a.cmd](a)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
