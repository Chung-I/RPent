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

"""The 6-DoF grasp executor of the sixdof rerun (daily-logs design 2026-11-08-sixdof-rerun-design.md, level A).

A pinch with a rotation (Grasp6: the pad centre and the TCP frame, +X the closing axis, +Z the approach), its angle bin
by the approach's angle from straight down, and the routing. A Proposal, or a Grasp6 within VERTICAL_DEG of straight
down, runs on today's executor unchanged (MuJoCoLabEnv.execute, hand_path and reach): the Grasp6 through the pilot's
vertical mapping (curve_graspgen.map_grasp's point and yaw, the yaw derived from the closing axis). A tilted Grasp6
never reaches today's executor.
"""

from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from robots.mujoco_lab.adapter import (A_TRANSPORT, APPROACH_M, CLOSE_S, HOLD_S, LIFT_H, REACH_CHECK_EVERY, REACH_TOL_M, REACH_TOL_RAD,
                                       RELEASE_S, RETREAT_M, V_DOWN, V_LOWER, Proposal, ramp_profile, transport_profile)
from robots.mujoco_lab.curve_graspgen import DEPTH, PAD_ABOVE, VERTICAL_DEG

BIN_EDGES = (VERTICAL_DEG, 45.0, 60.0, 75.0, 105.0)     # each bin (lo, hi], the first from 0; past the last: from below
BINS = ("≤ 30°", "30–45°", "45–60°", "60–75°", "75–105°")
LEVEL_A_DEG = 45.0                                       # level A runs the rows up to this angle
HAND_OFF = 0.048          # the hand origin sits this far behind the pad centre (the sim's PAD_DROP less PAD_HALF[2])
N_RANDOM_SEEDS = 4        # the configuration candidates' IK seeds past the arm's own three (design amendment 2)
SEEDS = ("current", "aimed", "rest") + tuple(f"random{k}" for k in range(N_RANDOM_SEEDS))
GOAL_TOL_M, GOAL_TOL_RAD = 1e-4, 1e-3     # a candidate is an IK solution at the pregrasp within these (Sim.plan_free's)
DISTINCT_RAD = 1e-3       # a candidate within this of an earlier one (every joint) is the same configuration


@dataclass
class Grasp6:
    """A pinch with a rotation: ``tcp`` the pad centre and ``R`` the TCP frame (world; columns +X the closing axis,
    +Y = Z x X, +Z the approach into the object). ``s`` and ``width`` as Proposal's (the executor reads neither)."""

    s: float
    tcp: list[float]
    R: list[list[float]]
    width: float


def from_graspgen(g: Any, s: float = math.nan, width: float = math.nan, depth: float = DEPTH) -> Grasp6:
    """A GraspGen grasp (4x4, +Z the approach, +X the closing axis): its pad centre ``depth`` along the approach from the
    frame's origin (curve_graspgen.map_grasp's pad) and its own frame."""
    g = np.asarray(g, dtype=float)
    return Grasp6(s=float(s), tcp=[float(x) for x in g[:3, 3] + depth * g[:3, 2]], R=[[float(x) for x in row] for row in g[:3, :3]],
                  width=float(width))


def approach_deg(p: Grasp6) -> float:
    """The angle of the approach (+Z) from straight down, in degrees (curve_graspgen.approach_deg)."""
    return float(np.degrees(np.arccos(np.clip(-float(p.R[2][2]), -1.0, 1.0))))


def angle_bin(deg: float) -> str | None:
    """The design's angle bin of an approach ``deg`` from straight down; None past the last edge (from below)."""
    if deg <= BIN_EDGES[0]:
        return BINS[0]
    for lo, hi, name in zip(BIN_EDGES, BIN_EDGES[1:], BINS[1:]):
        if lo < deg <= hi:
            return name
    return None


def as_vertical(p: Grasp6) -> Proposal:
    """The pilot's vertical mapping of a grasp within VERTICAL_DEG: the lab point PAD_ABOVE below the pad centre, the yaw
    of the closing axis projected on the table (curve_graspgen.map_grasp, bit for bit)."""
    a = approach_deg(p)
    if a > VERTICAL_DEG:
        raise ValueError(f"the approach is {a:.2f}° from straight down, past {VERTICAL_DEG:g}°: no vertical mapping")
    return Proposal(s=p.s, point=[float(x) for x in np.asarray(p.tcp, dtype=float) - np.array([0.0, 0.0, PAD_ABOVE])],
                    yaw=float(math.atan2(p.R[1][0], p.R[0][0])), width=p.width)


def hand_frame(p: Grasp6) -> tuple[np.ndarray, np.ndarray]:
    """The lab hand's pose at a pinch: its origin HAND_OFF back along the approach from the pad centre, its rotation
    R diag(1, -1, -1) (+X the closing axis, +Z against the approach: Rz(yaw) for a vertical pinch)."""
    R = np.asarray(p.R, dtype=float)
    Rh = R @ np.diag([1.0, -1.0, -1.0])
    return np.asarray(p.tcp, dtype=float) + HAND_OFF * Rh[:, 2], Rh


def twin(p: Grasp6) -> Grasp6:
    """The same pinch with the pads swapped: the frame turned a half turn about the approach."""
    return Grasp6(s=p.s, tcp=list(p.tcp), R=[[-r[0], -r[1], r[2]] for r in p.R], width=p.width)


def pregrasp(p: Grasp6) -> tuple[np.ndarray, np.ndarray]:
    """The hand's pose at the pregrasp: APPROACH_M back along the approach from the pinch, the pinch's rotation."""
    hp, Rh = hand_frame(p)
    return hp - APPROACH_M * np.asarray(p.R, dtype=float)[:, 2], Rh


def bearing_turn(base: Any, start: Any, goal: Any) -> float:
    """The signed angle (rad, -pi to pi) from ``start``'s bearing to ``goal``'s about ``base`` in the plane, the short
    way round."""
    return math.remainder(math.atan2(goal[1] - base[1], goal[0] - base[0]) - math.atan2(start[1] - base[1], start[0] - base[0]),
                          2 * math.pi)


def _rz(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def return_rotation(a: Any, frac: float) -> np.ndarray:
    """Design amendment 7's return toward vertical: the rotation about the horizontal axis k = a x down that turns the
    approach ``a`` toward straight down by ``frac`` of its angle from it (Rodrigues; the identity at no tilt)."""
    a = np.asarray(a, dtype=float)
    c = np.cross(a, [0.0, 0.0, -1.0])
    s = float(np.linalg.norm(c))
    if s == 0.0:
        return np.eye(3)
    k = c / s
    t = float(frac) * math.atan2(s, -float(a[2]))
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(t) * K + (1.0 - math.cos(t)) * (K @ K)


def turn_deg(env: Any, p: Grasp6, destination: str) -> float:
    """Design amendment 5's transport turn, degrees: the pad centre's bearing change about the arm's base (joint 1's
    axis), from the pinch to the destination's xy."""
    arm = env.sim.arm
    base = arm.d.xpos[arm.m.body("link0").id][:2]
    return math.degrees(bearing_turn(base, np.asarray(p.tcp, dtype=float)[:2], np.asarray(env._dest(destination)["xy"], dtype=float)))


def hand_path6(env: Any, p: Grasp6, v_lift: float, v_transport: float, destination: str, release_height: float | None = None,
               a_transport: float = A_TRANSPORT) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """The hand pose per control step of the tilted path from the pregrasp on, as MuJoCoLabEnv.hand_path: the 0.3 s
    open wait at the pregrasp APPROACH_M back along the approach, the approach along it (the sim's approach_profile at
    V_DOWN), the close, the lift LIFT_H straight up, the hold, the transport of the pad centre in a straight line to the
    destination's xy, the lower, the release, and the retreat RETREAT_M back along the approach (straight up); the held
    object rigid in the hand: [(phase, hand position, hand rotation)]. The rotation is the pinch's to the end of the
    hold. During the transport it turns about the world vertical by turn_deg (design amendment 5) and returns toward
    vertical by return_rotation (amendment 7), both by the transport's fraction, about the pad centre, then holds the
    full turn and return (vertical, the pads down) through the lower, the release and the retreat. The lower puts the
    lowest hull vertex of the object, turned and returned with the hand, at the release height. At no tilt the
    positions are today's path, the yaw turns."""
    sc, gr = env.s["scene"], env.s["grasp"]
    dest = env._dest(destination)
    hz = sc.CTRL_HZ
    hp, Rh = hand_frame(p)
    a = np.asarray(p.R, dtype=float)[:, 2]
    out = [("wait", pregrasp(p)[0], Rh)] * int(0.3 * hz)
    for f in gr.approach_profile(APPROACH_M, V_DOWN, hz):
        out.append(("approach", hp - APPROACH_M * (1.0 - f) * a, Rh))       # ends at the grasp pose exactly (grasp.descent_zs)
    out += [("close", hp, Rh)] * int(CLOSE_S * hz)
    for frac in ramp_profile(LIFT_H, v_lift, hz):
        out.append(("lift", hp + np.array([0.0, 0.0, LIFT_H * frac]), Rh))
    top = hp + np.array([0.0, 0.0, LIFT_H])
    out += [("hold", top, Rh)] * int(HOLD_S * hz)
    tcp = np.asarray(p.tcp, dtype=float) + np.array([0.0, 0.0, LIFT_H])       # the pad centre at the top
    off = top - tcp
    turn = math.radians(turn_deg(env, p, destination))
    start, goal = tcp[:2], np.asarray(dest["xy"], dtype=float)
    for frac in transport_profile(float(np.linalg.norm(goal - start)), v_transport, hz, accel=a_transport):
        Q = _rz(turn * frac) @ return_rotation(a, frac)
        c = start + (goal - start) * frac
        out.append(("transport", np.array([c[0], c[1], tcp[2]]) + Q @ off, Q @ Rh))
    Q = _rz(turn) @ return_rotation(a, 1.0)
    Re, end = Q @ Rh, np.array([goal[0], goal[1], tcp[2]]) + Q @ off
    V = np.concatenate(env._hull_vertex_list()) - np.asarray(p.tcp, dtype=float)     # the object from the pad centre, at rest
    z_rel = env._release_target_low_z(dest, release_height) + float((Q @ off)[2]) - float((V @ Q.T)[:, 2].min())
    z0 = z = float(end[2])
    for frac in ramp_profile(z0 - z_rel, V_LOWER, hz):
        z = z0 + (z_rel - z0) * frac
        out.append(("lower", np.array([end[0], end[1], z]), Re))
    rel = np.array([end[0], end[1], z])
    out += [("release", rel, Re)] * int(RELEASE_S * hz)
    for frac in ramp_profile(RETREAT_M, V_LOWER, hz):
        out.append(("retreat", rel - RETREAT_M * frac * (Q @ a), Re))
    return out


# ------------------------------------------------------------------ the rotation IK, the candidates and the path check

def _armlib(arm: Any) -> Any:
    """The sim's arm module (its IK constants, Q_REST, the joint limits)."""
    return sys.modules[type(arm).__module__]


def _fk(arm: Any, q: Any) -> tuple[np.ndarray, np.ndarray]:
    """The hand's pose at arm joints ``q`` on the arm's kinematics scratch (the rest of the scene as it is now)."""
    import mujoco

    kd = arm.kd
    kd.qpos[:] = arm.d.qpos
    kd.qpos[arm.q0:arm.q0 + len(q)] = q
    mujoco.mj_kinematics(arm.m, kd)
    return kd.xpos[arm.hand].copy(), kd.xmat[arm.hand].reshape(3, 3).copy()


def pose_error(arm: Any, q: Any, pos: Any, R: Any) -> tuple[float, float]:
    """The hand's position error (m) and rotation angle (rad, 0 to pi) from (pos, R) at arm joints ``q``."""
    p, Rc = _fk(arm, np.asarray(q, dtype=float))
    M = Rc.T @ np.asarray(R, dtype=float)
    s = 0.5 * float(np.linalg.norm([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]]))
    return float(np.linalg.norm(np.asarray(pos, dtype=float) - p)), float(math.atan2(s, 0.5 * (float(np.trace(M)) - 1.0)))


def ik6(arm: Any, pos: Any, R: Any, q_init: Any = None) -> tuple[np.ndarray, float, float]:
    """The sim's Arm.ik with a rotation target: arm joints that put the hand origin at ``pos`` with rotation ``R``. The
    same damped least squares on the kinematics scratch, from ``q_init`` (default Q_REST), with the null-space pull to
    Q_REST and the joints LIMIT_MARGIN inside their ranges; at R = Rz(yaw) the joints are Arm.ik(pos, yaw)'s. Returns
    (q, position error m, rotation angle rad) of the returned q: Arm.ik's rotation error is |sin| of the angle, zero at
    a half turn, so the angle is measured again (pose_error)."""
    import mujoco

    A = _armlib(arm)
    m, kd, n = arm.m, arm.kd, A.N_ARM
    kd.qpos[:] = arm.d.qpos
    q = (A.Q_REST if q_init is None else np.asarray(q_init, float)).copy()
    target_R = np.asarray(R, dtype=float)
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    lo, hi = A.Q_LO + A.LIMIT_MARGIN, A.Q_HI - A.LIMIT_MARGIN
    for _ in range(A.IK_ITERS):
        kd.qpos[arm.q0:arm.q0 + n] = q
        mujoco.mj_kinematics(m, kd)
        p, Rc = kd.xpos[arm.hand], kd.xmat[arm.hand].reshape(3, 3)
        e_p = np.asarray(pos, float) - p
        e_r = 0.5 * (np.cross(Rc[:, 0], target_R[:, 0]) + np.cross(Rc[:, 1], target_R[:, 1]) + np.cross(Rc[:, 2], target_R[:, 2]))
        if float(np.linalg.norm(e_p)) < A.IK_TOL_M and float(np.linalg.norm(e_r)) < A.IK_TOL_RAD:
            break
        mujoco.mj_comPos(m, kd)
        mujoco.mj_jacBody(m, kd, jp, jr, arm.hand)
        J = np.vstack([jp[:, arm.v0:arm.v0 + n], jr[:, arm.v0:arm.v0 + n]])
        e = np.concatenate([e_p, e_r])
        JJt = J @ J.T + A.IK_DAMP ** 2 * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, e)
        if A.NULL_GAIN > 0:
            Jp = J.T @ np.linalg.solve(JJt, J)
            dq += (np.eye(n) - Jp) @ (A.NULL_GAIN * (A.Q_REST - q))
        q = np.clip(q + dq, lo, hi)
    return (q, *pose_error(arm, q, pos, target_R))


def row_key(case: dict[str, Any], row_id: str) -> str:
    """The (case, row) key the candidates' random seeds derive from: object:idx:row."""
    return f"{case['object']}:{int(case['idx'])}:{row_id}"


def row_rng(key: str) -> np.random.Generator:
    """The row's own generator: seeded from the sha256 of its key."""
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))


def candidates(env: Any, p: Grasp6, key: str) -> list[dict[str, Any]]:
    """The arm configurations at the pinch's pregrasp, in the fixed order of design amendment 2: per IK seed in SEEDS
    (the arm's current joint target, Q_REST turned toward the pregrasp, Q_REST, then N_RANDOM_SEEDS uniform draws
    within the joint ranges from the row's generator) the pinch, then its twin; an IK solution within GOAL_TOL and
    DISTINCT_RAD from every earlier one is a candidate: [{"index", "twin", "seed", "q"}]."""
    arm = env.sim.arm
    A = _armlib(arm)
    rng = row_rng(key)
    lo, hi = A.Q_LO + A.LIMIT_MARGIN, A.Q_HI - A.LIMIT_MARGIN
    seeds = list(arm.seeds(pregrasp(p)[0], q_init=arm.desired())) + [rng.uniform(lo, hi) for _ in range(N_RANDOM_SEEDS)]
    out: list[dict[str, Any]] = []
    for name, q0 in zip(SEEDS, seeds, strict=True):
        for t, pt in enumerate((p, twin(p))):
            q, ep, er = ik6(arm, *pregrasp(pt), q_init=q0)
            if ep < GOAL_TOL_M and er < GOAL_TOL_RAD and all(float(np.max(np.abs(q - c["q"]))) > DISTINCT_RAD for c in out):
                out.append({"index": len(out), "twin": t, "seed": name, "q": q})
    return out


def reach6(env: Any, p: Grasp6, q: Any, destination: str, v_lift: float = 0.1, v_transport: float = 0.5,
           release_height: float | None = None, a_transport: float = A_TRANSPORT) -> dict[str, Any] | None:
    """MuJoCoLabEnv.reach's path check on the tilted path from arm joints ``q`` at the pregrasp (no free move: cuRobo
    plans it): every control step of hand_path6 by ik6 warm-started at the previous step ("ik" past REACH_TOL_M or
    REACH_TOL_RAD), and the arm and pads against the scene every REACH_CHECK_EVERY steps and at each phase change, the
    pads free to touch the object and the object left out from the lift on ("collision"). None if the path passes.
    ValueError if ``q`` does not put the hand at the pregrasp (the other twin's configuration, say)."""
    sim = env.sim
    arm = sim.arm
    path = hand_path6(env, p, v_lift, v_transport, destination, release_height, a_transport)
    q = np.asarray(q, dtype=float)
    ep, er = pose_error(arm, q, path[0][1], path[0][2])
    if ep > REACH_TOL_M or er > REACH_TOL_RAD:
        raise ValueError(f"the start configuration puts the hand {ep:.2e} m and {er:.2e} rad from the pregrasp")
    hulls, pads = set(sim.hull_ids), {sim.pad_l, sim.pad_r}
    prev, lifted_ = None, False
    for k, (ph, pos, Rh) in enumerate(path):
        q, ep, er = ik6(arm, pos, Rh, q_init=q)
        if ep > REACH_TOL_M or er > REACH_TOL_RAD:
            return {"reason": "ik", "phase": ph, "step": k, "ik_error": [float(ep), float(er)]}
        lifted_ = lifted_ or ph == "lift"
        if k % REACH_CHECK_EVERY == 0 or ph != prev or k == len(path) - 1:
            hits = [h for h in arm.collisions(q, ignore=hulls if lifted_ else ()) if not (h[0] in pads and h[1] in hulls)]
            if hits:
                r_, o_, dist = hits[0]
                return {"reason": "collision", "phase": ph, "step": k, "hit": [sim.m.geom(r_).name, sim.m.geom(o_).name, float(dist)]}
        prev = ph
    return None


def choose(env: Any, p: Grasp6, key: str, destination: str, v_lift: float = 0.1, v_transport: float = 0.5,
           release_height: float | None = None, a_transport: float = A_TRANSPORT) -> dict[str, Any]:
    """The configuration choice of design amendment 2: the first candidate whose whole path passes reach6. Returns
    {"goal": any candidate, "candidate": the chosen one or None, "failed": [{"index", reach6's failure}] before it}."""
    cands = candidates(env, p, key)
    failed: list[dict[str, Any]] = []
    for c in cands:
        r = reach6(env, twin(p) if c["twin"] else p, c["q"], destination, v_lift, v_transport, release_height, a_transport)
        if r is None:
            return {"goal": True, "candidate": c, "failed": failed}
        failed.append({"index": c["index"], **r})
    return {"goal": bool(cands), "candidate": None, "failed": failed}


# ------------------------------------------------------------------ the clearance check (design ruling 1)

OPEN_PHASES = ("wait", "approach", "release", "retreat")   # the fingers open; closed on the width from the close to the lower
OBJECT_PHASES = ("wait", "approach", "close")             # the palm and wrist against the object's hulls at its rest pose
CLEAR_DISTMAX = 0.01                                      # mj_geomDistance's horizon (m): a pair this far apart is clear


def finger_q(sc: Any, w: float) -> float:
    """The finger joints' position that closes each pad's inner face onto +-w/2 (a pad is 2 PAD_HALF[0] thick and its
    finger starts MAX_OPEN out), within the joints' range [0, MAX_OPEN - 0.002]."""
    return float(np.clip(sc.MAX_OPEN - 2.0 * sc.PAD_HALF[0] - 0.5 * float(w), 0.0, sc.MAX_OPEN - 0.002))


def hand_geoms(sim: Any) -> dict[str, list[int]]:
    """The hand's bodies the sim never collides (contype 0): the palm box, the wrist cylinder and the two finger boxes."""
    m = sim.m
    fingers = [g for b in ("finger_l", "finger_r") for g in range(m.ngeom)
               if int(m.geom_bodyid[g]) == m.body(b).id and g not in (sim.pad_l, sim.pad_r)]
    return {"palm": [m.geom("palm").id], "wrist": [m.geom("wrist").id], "finger": fingers}


def obstacle_sets(sim: Any, destination: str) -> dict[str, set[int]]:
    """The obstacles by kind: the object's hulls, the table, the destination (the bin's geoms or the fixture of that
    name; none for a table spot), and every other fixture with the bin when it is not the destination."""
    dest = set(sim.bin_geoms) if destination == "bin" else set(sim.fixture_geoms.get(destination, ()))
    scene = set(sim.bin_geoms).union(*sim.fixture_geoms.values())
    return {"object": set(sim.hull_ids), "table": {sim.table}, "fixture": scene - dest, "destination": dest}


def local_poses(sim: Any, geoms: list[int], jl: float) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Each geom's pose in the hand frame with both finger joints at ``jl`` (kinematics scratch)."""
    import mujoco

    m, kd, h = sim.m, sim.arm.kd, sim.arm.hand
    kd.qpos[:] = sim.d.qpos
    kd.qpos[sim.finger_q] = jl
    mujoco.mj_kinematics(m, kd)
    ph, Rh = kd.xpos[h].copy(), kd.xmat[h].reshape(3, 3).copy()
    return {g: (Rh.T @ (kd.geom_xpos[g] - ph), Rh.T @ kd.geom_xmat[g].reshape(3, 3)) for g in geoms}


def clearance(env: Any, p: Grasp6, w: float, destination: str, v_lift: float = 0.1, v_transport: float = 0.5,
              release_height: float | None = None, a_transport: float = A_TRANSPORT) -> dict[str, Any] | None:
    """Ruling 1's kinematic clearance of the bodies the sim never collides (hand_geoms) along hand_path6, at every
    control step from the pregrasp to the end of the retreat, the boxes placed on the hand pose (they do not depend on
    the arm's joints): the palm and wrist against the object's hulls at its rest pose from the pregrasp to the close
    (the finger boxes against the object are the width rule's), every body against the table, the fixtures and the
    destination at every step; the fingers open in OPEN_PHASES, closed on ``w`` (finger_q) from the close to the lower.
    A pair fails when mj_geomDistance is below 0. None when clear, else clearance_on_path's result."""
    return clearance_on_path(env, hand_path6(env, p, v_lift, v_transport, destination, release_height, a_transport), w, destination)


def clearance_on_path(env: Any, path: list[tuple[str, Any, Any]], w: float, destination: str) -> dict[str, Any] | None:
    """clearance's check along a given hand path [(phase, hand position, hand rotation)] (hand_path6's, or today's
    MuJoCoLabEnv.hand_path with each yaw as a rotation about the vertical). None when clear, else {"pairs": the failing
    body/obstacle pairs, "first": the earliest failure, "by_pair": {pair: its first phase and step, its steps, its deepest
    overlap and its geoms, and per phase its steps and deepest overlap}}."""
    import mujoco

    sim, sc = env.sim, env.s["scene"]
    m, kd = sim.m, sim.arm.kd
    body_of = {g: b for b, gs in hand_geoms(sim).items() for g in gs}
    kind_of = {o: k for k, os_ in obstacle_sets(sim, destination).items() for o in os_}
    hand = list(body_of)
    loc = {"open": local_poses(sim, hand, 0.0), "closed": local_poses(sim, hand, finger_q(sc, w))}
    kd.qpos[:] = sim.d.qpos                     # the obstacles where they are now (the object at rest)
    mujoco.mj_kinematics(m, kd)
    obs = np.array(sorted(kind_of), dtype=int)
    centres, rb = kd.geom_xpos[obs].copy(), m.geom_rbound[obs]
    by: dict[str, dict[str, Any]] = {}
    for k, (ph, pos, R) in enumerate(path):
        L = loc["open" if ph in OPEN_PHASES else "closed"]
        for g in hand:
            lp, lR = L[g]
            gp = pos + R @ lp
            kd.geom_xpos[g] = gp
            kd.geom_xmat[g] = (R @ lR).reshape(-1)
            near = obs[np.linalg.norm(centres - gp, axis=1) <= rb + m.geom_rbound[g] + CLEAR_DISTMAX]
            for o in near:
                kind = kind_of[int(o)]
                if kind == "object" and (body_of[g] == "finger" or ph not in OBJECT_PHASES):
                    continue
                dist = float(mujoco.mj_geomDistance(m, kd, int(g), int(o), CLEAR_DISTMAX, None))
                if dist < 0.0:
                    pair = f"{body_of[g]}/{kind}"
                    b = by.setdefault(pair, {"phase": ph, "step": k, "steps": 0, "depth": 0.0, "geoms": [m.geom(g).name or body_of[g],
                                                                                                         m.geom(int(o)).name],
                                             "phases": {}})
                    b["steps"] += 1
                    b["depth"] = max(b["depth"], -dist)
                    q = b["phases"].setdefault(ph, {"steps": 0, "depth": 0.0})
                    q["steps"] += 1
                    q["depth"] = max(q["depth"], -dist)
    if not by:
        return None
    first = min(by, key=lambda pr: by[pr]["step"])
    return {"pairs": sorted(by), "first": dict(pair=first, **by[first]), "by_pair": by}


def _today(p: Proposal | Grasp6) -> Proposal:
    """The Proposal today's executor runs: a Proposal (anything but a Grasp6) as it is, a Grasp6 within VERTICAL_DEG by the
    vertical mapping."""
    if not isinstance(p, Grasp6):
        return p
    if approach_deg(p) <= VERTICAL_DEG:
        return as_vertical(p)
    raise NotImplementedError(f"the tilted path ({angle_bin(approach_deg(p))}) is not built")


def execute(env: Any, p: Proposal | Grasp6, force: float, v_lift: float, v_transport: float = 0.5, destination: str = "bin",
            release_height: float | None = None, a_transport: float = A_TRANSPORT) -> Any:
    """Run a pinch: MuJoCoLabEnv.execute for a Proposal or a vertical Grasp6."""
    return env.execute(_today(p), force, v_lift, v_transport, destination, release_height, a_transport=a_transport)


def hand_path(env: Any, p: Proposal | Grasp6, v_lift: float, v_transport: float, destination: str,
              release_height: float | None = None, a_transport: float = A_TRANSPORT) -> list[tuple[str, np.ndarray, float]]:
    """The planned hand path: MuJoCoLabEnv.hand_path for a Proposal or a vertical Grasp6."""
    return env.hand_path(_today(p), v_lift, v_transport, destination, release_height, a_transport=a_transport)


def reach(env: Any, p: Proposal | Grasp6, destination: str, v_lift: float = 0.1, v_transport: float = 0.5,
          release_height: float | None = None, a_transport: float = A_TRANSPORT) -> dict[str, Any] | None:
    """The reachability rule: MuJoCoLabEnv.reach for a Proposal or a vertical Grasp6 (None: reachable)."""
    return env.reach(_today(p), destination, v_lift, v_transport, release_height, a_transport)
