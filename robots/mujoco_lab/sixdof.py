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

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from robots.mujoco_lab.adapter import (A_TRANSPORT, APPROACH_M, CLOSE_S, HOLD_S, LIFT_H, RELEASE_S, RETREAT_M, V_DOWN, V_LOWER,
                                       Proposal, ramp_profile, transport_profile)
from robots.mujoco_lab.curve_graspgen import DEPTH, PAD_ABOVE, VERTICAL_DEG

BIN_EDGES = (VERTICAL_DEG, 45.0, 60.0, 75.0, 105.0)     # each bin (lo, hi], the first from 0; past the last: from below
BINS = ("≤ 30°", "30–45°", "45–60°", "60–75°", "75–105°")
LEVEL_A_DEG = 45.0                                       # level A runs the rows up to this angle
HAND_OFF = 0.048          # the hand origin sits this far behind the pad centre (the sim's PAD_DROP less PAD_HALF[2])


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


def hand_path6(env: Any, p: Grasp6, v_lift: float, v_transport: float, destination: str, release_height: float | None = None,
               a_transport: float = A_TRANSPORT) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """The hand pose per control step of the tilted path from the pregrasp on, as MuJoCoLabEnv.hand_path: the 0.3 s
    open wait at the pregrasp APPROACH_M back along the approach, the approach along it (the sim's approach_profile at
    V_DOWN), the close, the lift LIFT_H straight up, the hold, the transport of the pad centre to the destination's xy,
    the lower, the release, and the retreat RETREAT_M back along the approach; the rotation kept throughout, the held
    object rigid in the hand: [(phase, hand position, hand rotation)]. At no tilt it is today's path."""
    sc, gr = env.s["scene"], env.s["grasp"]
    dest = env._dest(destination)
    hz = sc.CTRL_HZ
    hp, Rh = hand_frame(p)
    a = np.asarray(p.R, dtype=float)[:, 2]
    pre = hp - APPROACH_M * a
    out = [("wait", pre, Rh)] * int(0.3 * hz)
    for f in gr.approach_profile(APPROACH_M, V_DOWN, hz):
        out.append(("approach", hp - APPROACH_M * (1.0 - f) * a, Rh))       # ends at the grasp pose exactly (grasp.descent_zs)
    out += [("close", hp, Rh)] * int(CLOSE_S * hz)
    for frac in ramp_profile(LIFT_H, v_lift, hz):
        out.append(("lift", hp + np.array([0.0, 0.0, LIFT_H * frac]), Rh))
    top = hp + np.array([0.0, 0.0, LIFT_H])
    out += [("hold", top, Rh)] * int(HOLD_S * hz)
    start, goal = top[:2], np.asarray(dest["xy"], dtype=float) + (hp - np.asarray(p.tcp, dtype=float))[:2]
    for frac in transport_profile(float(np.linalg.norm(goal - start)), v_transport, hz, accel=a_transport):
        c = start + (goal - start) * frac
        out.append(("transport", np.array([c[0], c[1], top[2]]), Rh))
    z_rel = env._release_target_low_z(dest, release_height) - (env.sim.obj_low_z() - hp[2])
    z0 = z = top[2]
    for frac in ramp_profile(z0 - z_rel, V_LOWER, hz):
        z = z0 + (z_rel - z0) * frac
        out.append(("lower", np.array([goal[0], goal[1], z]), Rh))
    rel = np.array([goal[0], goal[1], z])
    out += [("release", rel, Rh)] * int(RELEASE_S * hz)
    for frac in ramp_profile(RETREAT_M, V_LOWER, hz):
        out.append(("retreat", rel - RETREAT_M * frac * a, Rh))
    return out


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
