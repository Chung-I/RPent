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

"""Thin adapter over the standalone MuJoCo grasp scene (the experiment's ``sim/`` code).

The scene lives outside RPent; ``RPENT_MUJOCO_SIM_DIR`` names its folder. This
module is the only place that imports MuJoCo. It offers what the pick_place skill
needs: reset an object with hidden properties, locate it, propose pinches, measure
geometry, plan the hand trajectory, execute pinch → lift → transport → set-down →
release, and label the outcome.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from rpent.skills.pick_place.destinations import DESTINATIONS, OBJECT_REST_XY, PRIMITIVE_IDS, has_surface, is_mesh, release_height
from robots.mujoco_lab.labels import RISE_M, Final, label_rollout
from robots.mujoco_lab.readouts import RISEN_M, summarize
from rpent.skills.pick_place.features import G, Geometry
from rpent.skills.pick_place.probes import ProbeResult
from rpent.skills.pick_place.protocol import OUTCOMES  # noqa: F401  (label vocabulary)

LIFT_H = 0.20
HOLD_S = 0.5
V_DOWN = 0.10
V_LOWER = 0.10
A_TRANSPORT = 2.0        # m/s^2: the transport accelerates and brakes at 0.2 g (a step change in speed strips the pads)
CLOSE_S = 0.6
APPROACH_M = 0.12
RELEASE_S = 3.0
RETREAT_M = 0.10
RETREAT_SETTLE_S = 0.5
TIP_DEG = 45.0
DISPLACE_TOL_M = 0.05
WEIGH_V_LIFT = 0.03          # the weigh probe: lift 2 cm slowly, hold 0.5 s, then set down
WEIGH_LIFT_M = 0.02
WEIGH_HOLD_S = 0.5
WEIGH_OPEN_S = 0.5
WEIGH_SETTLE_S = 1.0
PUSH_V = 0.03                # the push probe (t3_probes.push_probe): 3 cm/s, stop at 4 cm or 25 deg
PUSH_MAX_S = 4.0
PUSH_STOP_M = 0.04
PUSH_STOP_DEG = 25.0
PUSH_BREAKAWAY_M = 0.003
PUSH_WINDOW_S = 0.3
PUSH_TIP_DEG = 8.0
PUSH_MIN_FORCE_N = 0.05
THROWN_M = 0.10              # a probe "throws" the object off the table top or this far from where it was
REST_MIN_S = 0.2             # reset: settle at least this long, then until the object is at rest,
REST_CAP_S = 3.0             # at most this long (an object still moving then starts flagged: reset_info at_rest False)
REST_WINDOW_S = 0.5          # at rest: over the last 0.5 s it moved under 1 mm and turned under 1 deg
REST_MOVE_M = 0.001
REST_TURN_DEG = 1.0

def at_rest(poses: list[tuple[np.ndarray, np.ndarray]], hz: int) -> bool:
    """True when every (position, rotation matrix) of the history's last REST_WINDOW_S (one entry per
    control step) lies within REST_MOVE_M and REST_TURN_DEG of the latest one: a rocking object that is
    back where it was at the window's ends is still moving."""
    w = int(round(REST_WINDOW_S * hz))
    if len(poses) <= w:
        return False
    p1, R1 = np.asarray(poses[-1][0]), np.asarray(poses[-1][1])
    for p, R in poses[-w - 1:-1]:
        cos = (np.trace(np.asarray(R).T @ R1) - 1.0) / 2.0
        if np.linalg.norm(np.asarray(p) - p1) >= REST_MOVE_M or math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0)))) >= REST_TURN_DEG:
            return False
    return True


# Primitive fixtures built into every scene (the registry gives the place; this gives the shape).
PRIMITIVE_FIXTURES: list[dict[str, Any]] = [
    {"name": "mouse_pad", "kind": "surface", "size": (0.10, 0.06, 0.002), "rgba": "0.15 0.15 0.18 1"},
    {"name": "plate", "kind": "rim", "radius": 0.11, "rim_height": 0.015, "rgba": "0.92 0.92 0.9 1"},
    {"name": "box", "kind": "cavity", "half": 0.08, "wall_height": 0.12, "rgba": "0.6 0.45 0.3 1"},
]

PRIMITIVE_HALF = {"mouse_pad": (0.10, 0.06), "plate": (0.12, 0.12), "box": (0.09, 0.09)}

_SIM: dict[str, Any] = {}


def sim_dir(explicit: str | None = None) -> str:
    d = explicit or os.environ.get("RPENT_MUJOCO_SIM_DIR")
    if not d:
        raise RuntimeError("RPENT_MUJOCO_SIM_DIR is not set (the experiment's sim/ folder)")
    return d


def load_sim(explicit: str | None = None) -> dict[str, Any]:
    """Import the sim modules once and cache them."""
    d = sim_dir(explicit)
    if _SIM.get("dir") == d:
        return _SIM
    if d not in sys.path:
        sys.path.insert(0, d)
    for name in ("scene", "grasp", "objects", "t3_make"):
        _SIM[name] = importlib.import_module(name)
    _SIM["dir"] = d
    return _SIM


def object_ids() -> list[str]:
    s = load_sim()
    return [o["id"] for o in s["objects"].OBJECTS if os.path.exists(_props_path(s, o["id"]))]


def _props_path(s: dict[str, Any], object_id: str) -> str:
    return os.path.join(s["scene"].MESHES, object_id, "props.json")


def object_props(object_id: str) -> dict[str, Any]:
    import json

    s = load_sim()
    return json.load(open(_props_path(s, object_id)))


def object_length(object_id: str) -> float:
    return float(object_props(object_id)["length"])


@dataclass
class Proposal:
    s: float
    point: list[float]
    yaw: float
    width: float


@dataclass
class Segment:
    phase: str
    n_steps: int
    z_start: float
    z_end: float
    has_surface: bool
    speed: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Rollout:
    label: str
    max_z: float
    drop_z: float | None
    loss_phase: str | None
    loss_step: int | None
    loss_t: float | None
    loss_speed: float | None
    zone: str | None
    release: dict[str, Any] | None
    planned: list[Segment]
    trace: list[dict[str, Any]] = field(default_factory=list)
    readouts: dict[str, Any] = field(default_factory=dict)


def _steps(distance: float, speed: float, hz: float) -> int:
    return int(math.ceil(max(0.0, distance) / (speed / hz) - 1e-9))


def transport_profile(distance: float, v_max: float, hz: float, accel: float = A_TRANSPORT) -> list[float]:
    """Path fractions per control step for a trapezoidal (or triangular) speed profile."""
    if distance <= 1e-9:
        return []
    dt = 1.0 / hz
    v_peak = min(v_max, math.sqrt(accel * distance))
    t_ramp = v_peak / accel
    d_ramp = 0.5 * accel * t_ramp ** 2
    t_flat = max(0.0, (distance - 2 * d_ramp) / v_peak)
    total = 2 * t_ramp + t_flat
    n = int(math.ceil(total / dt))
    out = []
    for k in range(1, n + 1):
        t = min(k * dt, total)
        if t < t_ramp:
            d = 0.5 * accel * t ** 2
        elif t < t_ramp + t_flat:
            d = d_ramp + v_peak * (t - t_ramp)
        else:
            tr = total - t
            d = distance - 0.5 * accel * tr ** 2
        out.append(min(1.0, d / distance))
    out[-1] = 1.0
    return out


class MuJoCoLabEnv:
    """One object on the table with hidden properties, the pads, and the place fixtures."""

    def __init__(self, sim_dir: str | None = None, render: bool = False):
        self.s = load_sim(sim_dir)
        self.render = render
        self.episode = 0
        self.sim = None
        self.obj: dict[str, Any] | None = None
        self.props: dict[str, Any] | None = None
        self.theta: dict[str, Any] | None = None
        self.com_end = +1
        self.up0 = np.array([0.0, 0.0, 1.0])
        self.rest_low_z = 0.0
        self.fixtures_in_scene: list[str] = [f["name"] for f in PRIMITIVE_FIXTURES]
        self.fixture: str | None = None

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _overlaps(xy_a, half_a, xy_b, half_b) -> bool:
        return abs(xy_a[0] - xy_b[0]) < half_a[0] + half_b[0] and abs(xy_a[1] - xy_b[1]) < half_a[1] + half_b[1]

    def _fixture_specs(self, fixture: str | None) -> tuple[list[dict[str, Any]], list[str]]:
        """The primitives (minus any the mesh fixture would overlap) plus the mesh fixture, and their names."""
        mesh = None
        if fixture is not None:
            if fixture not in DESTINATIONS or not is_mesh(DESTINATIONS[fixture]):
                raise ValueError(f"fixture {fixture!r} is not an inspected mesh fixture; known: {[k for k in DESTINATIONS if is_mesh(DESTINATIONS[k])]}")
            mesh = DESTINATIONS[fixture]
        out, names = [], []
        for f in PRIMITIVE_FIXTURES:
            dest = DESTINATIONS[f["name"]]
            if mesh is not None and self._overlaps(dest["xy"], PRIMITIVE_HALF[f["name"]], mesh["xy"],
                                                   (mesh["footprint"][0] / 2 + 0.01, mesh["footprint"][1] / 2 + 0.01)):
                continue
            out.append({**f, "xy": tuple(dest["xy"]), "mu": dest["mu"]})
            names.append(f["name"])
        if mesh is not None:
            out.append({"name": mesh["id"], "kind": "mesh", "mesh_dir": mesh["mesh_dir"], "xy": tuple(mesh["xy"]), "mu": mesh["mu"],
                        "z_offset": mesh.get("z_offset", 0.0), "centre_xy": tuple(mesh.get("centre_xy", (0.0, 0.0)))})
            names.append(mesh["id"])
        return out, names

    def reset(self, object_id: str, theta: dict[str, Any] | None = None, seed: int = 0, com_end: int = +1,
              fixture: str | None = None) -> None:
        sc, gr, ob, t3 = self.s["scene"], self.s["grasp"], self.s["objects"], self.s["t3_make"]
        o = ob.BY_ID[object_id]
        props = object_props(object_id)
        th = dict(t3.causes_for(props["mass"])["control"])
        th.update(theta or {})
        shift = (0.0, 0.0, 0.0, 0.0)
        if th.get("com_s") is not None:
            shift = t3.com_shift_for(o, props, float(th["com_s"]), end=com_end)
        flat = not (o["hollow"] or o["liquid"])
        # the rest pose under the case's own friction, mass and CoM (the default drop's pose rolls or tips
        # when the CoM moves); the drop's sideways drift is not kept: the object starts at the rest xy
        pos, quat = gr.rest_pose(o, props, mu=th["mu"], mass_scale=th["mass_scale"], com_shift=shift, flat=flat)
        specs, names = self._fixture_specs(fixture)
        rest_xy = tuple(DESTINATIONS[fixture]["object_rest_xy"]) if fixture is not None else OBJECT_REST_XY
        pos = np.array([rest_xy[0], rest_xy[1], pos[2]])
        xml = sc.build_xml(o, props, mu=th["mu"], mass_scale=th["mass_scale"], com_shift=shift, obj_pos=pos,
                           obj_quat=quat, cam_k=1.0, fixtures=specs)
        self.sim = sc.Sim(xml)
        self.fixtures_in_scene = names
        self.fixture = fixture
        self._onto_table()
        self.reset_info = self._settle_to_rest()
        self.obj, self.props, self.theta, self.com_end = o, props, th, com_end
        self.rng = np.random.default_rng(seed)
        self._refresh_baselines()
        self.episode += 1

    def _onto_table(self) -> None:
        """An object that left the table in its rest-pose drop (it toppled and rolled off) comes back with
        the drop's orientation and its lowest point on the table top: the drop's height was the floor's."""
        import mujoco

        sim, top = self.sim, float(self.s["scene"].TABLE_TOP)
        low = sim.obj_low_z()
        if low < top - 0.01:
            qa = sim.m.joint("obj_free").qposadr[0]
            sim.d.qpos[qa + 2] += top - low + 0.001
            mujoco.mj_forward(sim.m, sim.d)

    def _settle_to_rest(self) -> dict[str, Any]:
        """Step at least REST_MIN_S, then until at_rest, at most REST_CAP_S; the seconds taken, whether the
        object came to rest, and the zone it rests in."""
        sim, hz = self.sim, self.s["scene"].CTRL_HZ
        poses = [(sim.obj_pos().copy(), sim.obj_mat().copy())]
        k, rest = 0, False
        while k < int(round(REST_CAP_S * hz)):
            sim.step_ctrl()
            k += 1
            poses.append((sim.obj_pos().copy(), sim.obj_mat().copy()))
            if k >= int(round(REST_MIN_S * hz)) and at_rest(poses, hz):
                rest = True
                break
        return {"settle_s": k / hz, "at_rest": rest, "zone": sim.zone_of(sim.contacts()[0])}

    def _refresh_baselines(self) -> None:
        """The rest pose an attempt is planned and labelled against: the up axis, the lowest point
        and the CoM height, from the current state (after a fall or a push, not the reset pose)."""
        self.up0 = self.sim.obj_mat()[:, 2].copy()
        self.rest_low_z = self.sim.obj_low_z()
        self.rest_com_height = float(self.sim.d.subtree_com[self.sim.obj][2] - self.rest_low_z)

    def _need(self) -> None:
        if self.sim is None:
            raise RuntimeError("call reset() first")

    # ---------------------------------------------------------------- queries
    def get_obs(self) -> dict[str, Any]:
        self._need()
        sim = self.sim
        hand = sim.d.xpos[sim.hand].copy()
        jl = sim.d.qpos[sim.m.joint("jl").qposadr[0]]
        jr = sim.d.qpos[sim.m.joint("jr").qposadr[0]]
        opening = float(2 * sim.m.body("finger_r").pos[0] - jl - jr) if False else float((self.s["scene"].MAX_OPEN - jl) + (self.s["scene"].MAX_OPEN - jr))
        state = {
            "object_pos": sim.obj_pos().tolist(),
            "object_mat": sim.obj_mat().ravel().tolist(),
            "hand_pos": hand.tolist(),
            "gripper_opening": opening,
            "pad_forces": self._pad_forces(),
            "fixtures_in_scene": list(self.fixtures_in_scene),
        }
        vision = {"cam_main": sim.render("main")} if self.render else {}
        return {"state": state, "vision": vision}

    def _pad_forces(self) -> list[float]:
        import mujoco

        sim = self.sim
        f = np.zeros(6)
        out = [0.0, 0.0]
        for i in range(sim.d.ncon):
            c = sim.d.contact[i]
            for k, pad in enumerate((sim.pad_l, sim.pad_r)):
                if pad in (c.geom1, c.geom2):
                    mujoco.mj_contactForce(sim.m, sim.d, i, f)
                    out[k] += float(abs(f[0]))
        return out

    def _pad_hull_forces(self) -> tuple[float, float]:
        """(net vertical force of both pads on the object, summed normal force), pad-hull contacts only.

        The vertical force is signed (up is positive), so in a static two-pad hold it equals the
        weight whatever the squeeze; a per-contact absolute sum (t3_probes.pad_forces) also counts
        the normal force's vertical parts on curved or sloped faces. mj_contactForce gives the force
        in the contact frame, whose normal points from geom1 to geom2, as the force on geom2."""
        import mujoco

        sim = self.sim
        f = np.zeros(6)
        hull = set(sim.hull_ids)
        fz = fn = 0.0
        for i in range(sim.d.ncon):
            c = sim.d.contact[i]
            pads = (sim.pad_l, sim.pad_r)
            if c.geom1 in pads or c.geom2 in pads:
                other = c.geom2 if c.geom1 in pads else c.geom1
                if other in hull:
                    mujoco.mj_contactForce(sim.m, sim.d, i, f)
                    fw = c.frame.reshape(3, 3).T @ f[:3]
                    fz += float(fw[2]) if other == c.geom2 else -float(fw[2])
                    fn += abs(float(f[0]))
        return fz, fn

    def locate(self, prompt: str) -> dict[str, Any] | None:
        self._need()
        o, props = self.obj, self.props
        p = prompt.strip().lower()
        names = {o["id"].lower(), o["name"].lower(), "object", o["name"].lower().replace("_", " ")}
        if p not in names:
            return None
        return {
            "object_id": o["id"],
            "centroid": self.sim.obj_pos().tolist(),
            "extent": float(props["length"]),
            "volume": float(sum(h["volume"] for h in props["hulls"])),
            "material": o["material"],
            "hollow": bool(o["hollow"]),
            "reachable": self._resting(),
        }

    def _resting(self) -> bool:
        """The object touches the table or a fixture in the scene: not on the floor, in the bin, or held
        up by the hand alone."""
        zone = self.sim.zone_of(self.sim.contacts()[0])
        return zone == "table" or zone in self.fixtures_in_scene

    def nominal_mass(self, object_id: str | None = None) -> float:
        """The object's mass at mass_scale 1 (mesh volume times the material density)."""
        props = self.props if object_id is None or (self.obj and self.obj["id"] == object_id) else object_props(object_id)
        return float(props["mass"])

    def truth(self) -> dict[str, Any]:
        self._need()
        sim, t3 = self.sim, self.s["t3_make"]
        mass = float(sim.m.body_subtreemass[sim.obj])
        com = sim.d.subtree_com[sim.obj].copy()
        ax, lo, L = t3.pivot_axis(self.obj, self.props)
        com_obj = sim.obj_mat().T @ (com - sim.obj_pos())
        com_s = float(((com_obj - np.array(self.props["centre"])) @ ax - lo) / L)
        return {"mass": mass, "mu": float(self.theta["mu"]), "com_s": com_s, "com": com.tolist(), "com_obj": com_obj.tolist(),
                "theta": dict(self.theta)}

    def pose_info(self) -> dict[str, Any]:
        """The object's current pose and what a CoM hypothesis needs (object frame, nominal values)."""
        self._need()
        sim, t3 = self.sim, self.s["t3_make"]
        ax, lo, L = t3.pivot_axis(self.obj, self.props)
        return {
            "R": sim.obj_mat().ravel().tolist(),
            "p": sim.obj_pos().tolist(),
            "low_z": float(sim.obj_low_z()),
            "base_extent": float(self.geometry().base_extent),
            "length": float(self.props["length"]),
            "com_obj_nominal": [float(x) for x in self.props["com"]],
            "pivot": {"centre": [float(x) for x in self.props["centre"]], "axis": [float(x) for x in ax], "lo": float(lo), "L": float(L)},
            "nominal_mass": float(self.props["mass"]),
        }

    def proposal_pose(self, proposal: Proposal) -> dict[str, Any]:
        """pose_info plus the pinch point and the hand yaw of one proposal."""
        return {**self.pose_info(), "pt": [float(x) for x in proposal.point], "yaw": float(proposal.yaw)}

    def hull_faces_world(self) -> tuple[np.ndarray, np.ndarray]:
        self._need()
        sim = self.sim
        C, N = [], []
        for g in sim.hull_ids:
            mid = sim.m.geom_dataid[g]
            v0, nv = sim.m.mesh_vertadr[mid], sim.m.mesh_vertnum[mid]
            f0, nf = sim.m.mesh_faceadr[mid], sim.m.mesh_facenum[mid]
            Rg, pg = sim.d.geom_xmat[g].reshape(3, 3), sim.d.geom_xpos[g]
            Vw = sim.m.mesh_vert[v0:v0 + nv] @ Rg.T + pg
            tri = Vw[sim.m.mesh_face[f0:f0 + nf]]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
            C.append(tri.mean(1))
            N.append(n)
        return np.concatenate(C), np.concatenate(N)

    def _hull_vertices_world(self) -> np.ndarray:
        sim = self.sim
        out = []
        for g in sim.hull_ids:
            mid = sim.m.geom_dataid[g]
            v0, nv = sim.m.mesh_vertadr[mid], sim.m.mesh_vertnum[mid]
            out.append(sim.m.mesh_vert[v0:v0 + nv] @ sim.d.geom_xmat[g].reshape(3, 3).T + sim.d.geom_xpos[g])
        return np.concatenate(out)

    def geometry(self) -> Geometry:
        self._need()
        C, N = self.hull_faces_world()
        V = self._hull_vertices_world()
        low = V[:, 2].min()
        support = V[np.abs(V[:, 2] - low) < 0.01][:, :2]
        if len(support) < 3:
            support = V[:, :2]
        xy = V[:, :2]
        footprint_half = float(np.linalg.norm(xy - xy.mean(0), axis=1).max())
        sc = support - support.mean(0)
        if len(sc) >= 3:
            w, vec = np.linalg.eigh(np.cov(sc.T) + 1e-12 * np.eye(2))
            proj = sc @ vec
            base = float(min(np.ptp(proj[:, 0]), np.ptp(proj[:, 1])) / 2)
        else:
            base = 0.0
        com = self.nominal_com()
        return Geometry(centres=C, normals=N, com=com, length=float(self.props["length"]), base_extent=base,
                        com_height=float(com[2] - low), footprint_half_width=footprint_half)

    def nominal_com(self) -> np.ndarray:
        """The centre of mass the skill may know: the mesh-density CoM of props.json in the current
        pose. The hidden shift of the episode is not in it (that is what the belief is for)."""
        return self.sim.obj_mat() @ np.asarray(self.props["com"], dtype=float) + self.sim.obj_pos()

    def _pinch_frame(self) -> tuple[float, np.ndarray, np.ndarray]:
        axis = self.sim.obj_mat() @ np.array(self.props["axis"])
        yaw = float(np.arctan2(axis[1], axis[0]) + np.pi / 2)
        ex = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        ey = np.array([-ex[1], ex[0], 0.0])
        return yaw, ex, ey

    def pinch_width(self, point: np.ndarray, yaw: float) -> float:
        sc = self.s["scene"]
        C, _ = self.hull_faces_world()
        ex = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        ey = np.array([-ex[1], ex[0], 0.0])
        pad_z = point[2] + 0.004
        band = np.abs(C[:, 2] - pad_z) < sc.PAD_HALF[2]
        along = np.abs((C - point) @ ey) < sc.PAD_HALF[1]
        sel = band & along
        if not sel.any():
            return float("nan")
        s = (C[sel] - point) @ ex
        return float(s.max() - s.min())

    def proposals(self, fractions: tuple[float, ...] = (0.3, 0.5, 0.7)) -> list[Proposal]:
        self._need()
        gr = self.s["grasp"]
        pts = gr.axis_points(self.sim, self.props, list(fractions))
        yaw, _, _ = self._pinch_frame()
        return [Proposal(s=float(f), point=[float(x) for x in pt], yaw=yaw, width=self.pinch_width(pt, yaw))
                for f, pt in zip(fractions, pts)]

    # --------------------------------------------------------------- planning
    def _surface_below(self, xy: np.ndarray) -> tuple[float, bool]:
        """(surface height under this xy, whether there is one): a fixture, the table, the bin floor, or none."""
        x, y = float(xy[0]), float(xy[1])
        sc = self.s["scene"]
        if self.fixture is not None:
            m = DESTINATIONS[self.fixture]
            if abs(x - m["xy"][0]) <= m["footprint"][0] / 2 and abs(y - m["xy"][1]) <= m["footprint"][1] / 2:
                return float(m["surface_z"]), True
        for f in PRIMITIVE_FIXTURES:
            if f["name"] not in self.fixtures_in_scene:
                continue
            dest = DESTINATIONS[f["name"]]
            fx, fy = dest["xy"]
            if f["kind"] == "surface":
                hx, hy = f["size"][0], f["size"][1]
                if abs(x - fx) <= hx and abs(y - fy) <= hy:
                    return float(dest["surface_z"]), True
            elif f["kind"] == "rim":
                if math.hypot(x - fx, y - fy) <= f["radius"]:
                    return float(dest["surface_z"]), True
            elif f["kind"] == "cavity":
                if abs(x - fx) <= f["half"] and abs(y - fy) <= f["half"]:
                    return float(dest["surface_z"]), True
        if abs(x) <= sc.TABLE_HALF[0] and abs(y) <= sc.TABLE_HALF[1]:
            return float(sc.TABLE_TOP), True
        bx, by = sc.BIN_CENTRE
        if abs(x - bx) <= sc.BIN_INNER and abs(y - by) <= sc.BIN_INNER:
            return float(DESTINATIONS["bin"]["surface_z"]), True
        return 0.0, False

    def _dest(self, destination: str) -> dict[str, Any]:
        if destination not in DESTINATIONS:
            raise ValueError(f"unknown destination {destination!r}; known: {sorted(DESTINATIONS)}")
        dest = DESTINATIONS[destination]
        if (is_mesh(dest) or destination in PRIMITIVE_HALF) and destination not in self.fixtures_in_scene:
            raise ValueError(f"fixture {destination!r} is not in the scene (present: {self.fixtures_in_scene})")
        return dest

    def _release_target_low_z(self, dest: dict[str, Any], rel_h: float | None) -> float:
        sc = self.s["scene"]
        h = release_height(dest) if rel_h is None else float(rel_h)
        base = float(dest["surface_z"]) if has_surface(dest) else float(sc.TABLE_TOP)
        return base + h

    def plan_trajectory(self, proposal: Proposal, v_lift: float, v_transport: float, destination: str,
                        release_height: float | None = None) -> list[Segment]:
        self._need()
        sc, gr = self.s["scene"], self.s["grasp"]
        dest = self._dest(destination)
        hz = sc.CTRL_HZ
        pt = np.asarray(proposal.point, dtype=float)
        hand_z0 = gr.hand_start_z(pt)
        rest_low_z = self.sim.obj_low_z()             # the current rest: a plan after a fall starts from the new pose
        low_off = rest_low_z - hand_z0                # object's lowest point relative to the hand (held)
        surf0, _ = self._surface_below(pt[:2])
        z_rest = rest_low_z - surf0
        dxy = np.asarray(dest["xy"], dtype=float) - pt[:2]
        surf_d, has_d = self._surface_below(np.asarray(dest["xy"]))
        z_hand_top = hand_z0 + LIFT_H
        low_top = z_hand_top + low_off
        target_low = self._release_target_low_z(dest, release_height)
        z_hand_rel = target_low - low_off
        segs = [
            Segment("approach", _steps(APPROACH_M, V_DOWN, hz), z_rest, z_rest, True, V_DOWN),
            Segment("close", int(CLOSE_S * hz), z_rest, z_rest, True, 0.0),
            Segment("lift", _steps(LIFT_H, v_lift, hz), z_rest, low_top - surf0, True, v_lift),
            Segment("hold", int(HOLD_S * hz), low_top - surf0, low_top - surf0, True, 0.0),
            Segment("transport", len(transport_profile(float(np.linalg.norm(dxy)), v_transport, hz)), low_top - surf0,
                    low_top - (surf_d if has_d else sc.TABLE_TOP), has_d, v_transport),
            Segment("lower", _steps(z_hand_top - z_hand_rel, V_LOWER, hz), low_top - (surf_d if has_d else sc.TABLE_TOP),
                    target_low - (surf_d if has_d else sc.TABLE_TOP), has_d, V_LOWER),
            Segment("release", int(RELEASE_S * hz), target_low - (surf_d if has_d else sc.TABLE_TOP),
                    target_low - (surf_d if has_d else sc.TABLE_TOP), has_d, 0.0),
            Segment("retreat", _steps(RETREAT_M, V_LOWER, hz) + int(RETREAT_SETTLE_S * hz),
                    target_low - (surf_d if has_d else sc.TABLE_TOP), target_low - (surf_d if has_d else sc.TABLE_TOP), has_d, V_LOWER),
        ]
        return segs

    # -------------------------------------------------------------- execution
    def execute(self, proposal: Proposal, force: float, v_lift: float, v_transport: float = 0.5,
                destination: str = "bin", release_height: float | None = None) -> Rollout:
        self._need()
        sc, gr = self.s["scene"], self.s["grasp"]
        dest = self._dest(destination)
        self._refresh_baselines()
        planned = self.plan_trajectory(proposal, v_lift, v_transport, destination, release_height)
        sim = self.sim
        hz = sc.CTRL_HZ
        pt = np.asarray(proposal.point, dtype=float)
        trace, on_step = self._tracer(pt, proposal.yaw)
        out, _, _ = gr.run_grasp(sim, pt, proposal.yaw, force=force, v_lift=v_lift, lift_h=LIFT_H, hold_s=HOLD_S,
                                 on_step=on_step)
        rose = any(x["z"] > RISE_M and all(x["pads"]) and x["phase"] in ("lift", "hold") for x in trace)
        release = None
        if rose:
            hand_z0 = gr.hand_start_z(pt)
            z = hand_z0 + LIFT_H
            xy = pt[:2].copy()
            goal = np.asarray(dest["xy"], dtype=float)
            for frac in transport_profile(float(np.linalg.norm(goal - xy)), v_transport, hz):
                cur = xy + (goal - xy) * frac
                sim.set_hand([cur[0], cur[1], z], proposal.yaw)
                sim.step_ctrl()
                on_step("transport")
            low_off = sim.obj_low_z() - sim.d.xpos[sim.hand][2]
            target_low = self._release_target_low_z(dest, release_height)
            z_rel = target_low - low_off
            while z > z_rel + 1e-6:
                z = max(z_rel, z - V_LOWER / hz)
                sim.set_hand([goal[0], goal[1], z], proposal.yaw)
                sim.step_ctrl()
                on_step("lower")
            release = gr.run_release(sim, None, 1, [], [], len(trace), seconds=RELEASE_S, on_step=on_step, up0=self.up0)
            z_up = z + RETREAT_M                        # the open hand moves away; a wedged object comes with it
            while z < z_up - 1e-6:
                z = min(z_up, z + V_LOWER / hz)
                sim.set_hand([goal[0], goal[1], z], proposal.yaw)
                sim.step_ctrl()
                on_step("retreat")
            for _ in range(int(RETREAT_SETTLE_S * hz)):
                sim.step_ctrl()
                on_step("retreat")
        return self._label(trace, rose, dest, release, planned, v_lift)

    def _tracer(self, pt: np.ndarray, yaw: float):
        """(trace, on_step): on_step(phase) appends one dict per control step (see execute's trace)."""
        sim = self.sim
        hz = self.s["scene"].CTRL_HZ
        p0z = sim.obj_pos()[2]
        R0 = sim.obj_mat().copy()
        trace: list[dict[str, Any]] = []
        pitch_of = self._pitch_meter(pt, yaw)
        pitch0: list[float] = []

        def on_step(phase: str) -> None:
            pos = sim.obj_pos()
            low = sim.obj_low_z()
            lin, _ = sim.obj_vel()
            touching, pads = sim.contacts()
            surf, has = self._surface_below(pos[:2])
            ang = float(np.degrees(np.arccos(np.clip(((sim.obj_mat() @ R0.T).trace() - 1) / 2, -1, 1))))
            fz, fn = self._pad_hull_forces()
            pitch = pitch_of()
            if not pitch0:
                pitch0.append(pitch)
            trace.append({"t": len(trace) / hz, "phase": phase, "z": float(pos[2] - p0z), "low_z": float(low),
                          "surface_z_below": surf, "has_surface": has, "speed": float(np.linalg.norm(lin)),
                          "pads": list(pads), "zone": sim.zone_of(touching), "rot_deg": ang,
                          "fz": fz, "fn": fn, "pitch_ey": float(pitch - pitch0[0])})
        return trace, on_step

    # ---------------------------------------------------------------- probes
    def _hand_geoms(self) -> set[int]:
        """Every geom of the hand's body subtree (pads, fingers, palm)."""
        m = self.sim.m
        bodies = {self.sim.hand}
        for b in range(m.nbody):
            k = b
            while k > 0 and k not in bodies:
                k = int(m.body_parentid[k])
            if k in bodies:
                bodies.add(b)
        return {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies}

    def _hand_push_force(self, hand: set[int], ex: np.ndarray) -> float:
        """Net force of the hand's geoms on the object along ex (the push direction)."""
        import mujoco

        sim = self.sim
        f = np.zeros(6)
        hull = set(sim.hull_ids)
        total = 0.0
        for i in range(sim.d.ncon):
            c = sim.d.contact[i]
            if (c.geom1 in hand and c.geom2 in hull) or (c.geom2 in hand and c.geom1 in hull):
                mujoco.mj_contactForce(sim.m, sim.d, i, f)
                fw = c.frame.reshape(3, 3).T @ f[:3]
                total += float(fw @ ex) if c.geom2 in hull else -float(fw @ ex)
        return total

    def _push_step(self, hand: set[int], ex: np.ndarray) -> float:
        """One control step, physics substep by substep: the hand force along ex averaged over the
        step. The hand follows its mocap target in jumps, so on a light, slippery object the push is a
        train of short strikes that a force sampled at the end of the step misses."""
        import mujoco

        sim, sub = self.sim, int(self.s["scene"].SUB)
        total = 0.0
        for _ in range(sub):
            mujoco.mj_step(sim.m, sim.d)
            total += self._hand_push_force(hand, ex)
        return total / sub

    def _tilt_from(self, R0: np.ndarray) -> float:
        return float(np.degrees(np.arccos(np.clip(((self.sim.obj_mat() @ R0.T).trace() - 1) / 2, -1, 1))))

    def _probe_cost(self, p_start: np.ndarray, R_start: np.ndarray) -> tuple[float, float, bool]:
        """(displacement from where the probe started, tilt, thrown: off the table top, > 10 cm away, or
        not resting on the table or a fixture, e.g. carried up on a pad by the rising hand)."""
        sc, sim = self.s["scene"], self.sim
        pos = sim.obj_pos()
        disp = float(np.linalg.norm((pos - p_start)[:2]))
        off = (sim.obj_low_z() < sc.TABLE_TOP - 0.01 or abs(pos[0]) > sc.TABLE_HALF[0] or abs(pos[1]) > sc.TABLE_HALF[1])
        return disp, self._tilt_from(R_start), bool(off or disp > THROWN_M or not self._resting())

    def _rise_hand(self, yaw: float, steps: list[int]) -> None:
        """The open hand rises RETREAT_M from where it is."""
        sim, hz = self.sim, self.s["scene"].CTRL_HZ
        x, y, z = (float(v) for v in sim.d.mocap_pos[sim.mocap])
        z_up = z + RETREAT_M
        while z < z_up - 1e-6:
            z = min(z_up, z + V_LOWER / hz)
            sim.set_hand([x, y, z], yaw)
            sim.step_ctrl()
            steps[0] += 1

    @staticmethod
    def weigh_reading(trace: list[dict[str, Any]], force: float) -> tuple[str, dict[str, float]]:
        """The weigh probe's kind and reading from its trace (see probe_weigh)."""
        risen = [i for i, x in enumerate(trace) if x["phase"] in ("lift", "hold") and x["z"] > RISEN_M and all(x["pads"])]
        if risen:
            i0 = risen[0]
            if any(not any(x["pads"]) for x in trace[i0:] if x["phase"] in ("lift", "hold")):
                return "lost", {}
            held = [x["fz"] for x in trace[i0:] if x["phase"] == "hold" and x["z"] > RISEN_M and all(x["pads"])]
            seg = held or [trace[i]["fz"] for i in risen]
            return "measured", {"m_hat": float(np.median(seg)) / G}
        both = [x["fz"] for x in trace if x["phase"] in ("close", "lift", "hold") and all(x["pads"])]
        fmax = float(max(both)) if both else 0.0
        return "lower_bound", {"mu_hat": fmax / (2.0 * float(force)), "m_lower": fmax / G}

    def probe_weigh(self, proposal: Proposal, force: float) -> ProbeResult:
        """Close with ``force``, lift 2 cm at 3 cm/s, hold 0.5 s (the execute trace); then open, settle
        0.5 s, rise 10 cm, settle 1 s. Kinds: measured (rose 1.2 cm with both pads: m_hat = the median
        net vertical pad force over the still hold, / g), lower_bound (never rose: mu_hat = max force /
        2F, m_lower = max force / g), lost (rose, then both pads lost it). The hold, not the lift: the
        hand follows its mocap target in jumps, so a force sampled at the end of a lift step reads
        0.9 of the weight on every object."""
        self._need()
        self._refresh_baselines()
        gr, sim, hz = self.s["grasp"], self.sim, self.s["scene"].CTRL_HZ
        pt = np.asarray(proposal.point, dtype=float)
        p_start, R_start = sim.obj_pos().copy(), sim.obj_mat().copy()
        trace, on_step = self._tracer(pt, proposal.yaw)
        gr.run_grasp(sim, pt, proposal.yaw, force=force, v_lift=WEIGH_V_LIFT, lift_h=WEIGH_LIFT_M, hold_s=WEIGH_HOLD_S,
                     on_step=on_step)
        kind, reading = self.weigh_reading(trace, force)
        steps = [len(trace)]
        sim.grip(-20.0)
        for _ in range(int(WEIGH_OPEN_S * hz)):
            sim.step_ctrl()
            steps[0] += 1
        self._rise_hand(proposal.yaw, steps)
        for _ in range(int(WEIGH_SETTLE_S * hz)):
            sim.step_ctrl()
            steps[0] += 1
        disp, tilt, thrown = self._probe_cost(p_start, R_start)
        return ProbeResult("weigh", kind, reading, disp, tilt, thrown, steps[0] / hz, trace)

    def probe_push(self) -> ProbeResult:
        """The t3_probes.push_probe motion: the open hand's leading pad, at a quarter of the object's
        height, pushes across the long axis at 3 cm/s until the object moved 4 cm or turned 25 deg; then
        the hand rises 10 cm and the scene settles 0.5 s. The push force is the net force along the push
        of every hand geom touching the object, averaged over each control step's physics substeps. Kinds: slid (breakaway, 3 mm, with a force above 0.05 N
        and a tilt of at most 8 deg; reading f_break = the largest force over the 0.3 s before it),
        tipped, no_breakaway, no_contact (moved with no hand force above 0.05 N)."""
        self._need()
        self._refresh_baselines()
        sc, sim, hz = self.s["scene"], self.sim, self.s["scene"].CTRL_HZ
        R, p = sim.obj_mat(), sim.obj_pos()
        axis = R @ np.asarray(self.props["axis"], dtype=float)
        yaw = float(np.arctan2(axis[1], axis[0]) + np.pi / 2)
        ex = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        V = self._hull_vertices_world()
        lo = float((V @ ex).min())
        zlo, zhi = float(V[:, 2].min()), float(V[:, 2].max())
        pad_z = max(zlo + 0.25 * (zhi - zlo), sc.TABLE_TOP + sc.PAD_HALF[2] + 0.002)
        hand_z = pad_z + sc.PAD_DROP - sc.PAD_HALF[2]
        start_s = lo - 0.01 - sc.MAX_OPEN
        perp = p - ex * float(p @ ex)
        perp[2] = 0.0
        p_start, R_start = p.copy(), R.copy()
        sim.teleport_hand(perp + ex * start_s + np.array([0.0, 0.0, hand_z]), yaw)
        sim.grip(-20.0)
        steps = [0]
        for _ in range(int(0.3 * hz)):
            sim.step_ctrl()
            steps[0] += 1
        p0, R0 = sim.obj_pos().copy(), sim.obj_mat().copy()
        hand = self._hand_geoms()
        rec: list[dict[str, Any]] = []
        breakaway = None
        window = int(round(PUSH_WINDOW_S * hz))
        for k in range(int(PUSH_MAX_S * hz)):
            sim.set_hand(perp + ex * (start_s + PUSH_V * k / hz) + np.array([0.0, 0.0, hand_z]), yaw)
            f = self._push_step(hand, ex)
            steps[0] += 1
            disp = float((sim.obj_pos() - p0) @ ex)
            tilt = self._tilt_from(R0)
            rec.append({"t": k / hz, "f": f, "disp": disp, "tilt": tilt})
            if breakaway is None and disp > PUSH_BREAKAWAY_M:
                breakaway = (max(r["f"] for r in rec[-window:]), tilt)
            if disp > PUSH_STOP_M or tilt > PUSH_STOP_DEG:
                break
        if breakaway is None:
            kind, reading = "no_breakaway", {}
        elif breakaway[0] <= PUSH_MIN_FORCE_N:
            kind, reading = "no_contact", {}
        elif breakaway[1] > PUSH_TIP_DEG:
            kind, reading = "tipped", {}
        else:
            kind, reading = "slid", {"f_break": float(breakaway[0])}
        self._rise_hand(yaw, steps)
        for _ in range(int(RETREAT_SETTLE_S * hz)):
            sim.step_ctrl()
            steps[0] += 1
        disp, tilt, thrown = self._probe_cost(p_start, R_start)
        return ProbeResult("push", kind, reading, disp, tilt, thrown, steps[0] / hz, rec)

    def _pitch_meter(self, pt: np.ndarray, yaw: float):
        """A function giving (z+ - z-) / 0.10 for the object-fixed points pt ± 0.05 ey (ey: the hand's
        horizontal axis across the pinch), positive when the +ey side is higher."""
        sim = self.sim
        ey = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        R0, p0 = sim.obj_mat(), sim.obj_pos()
        qp, qm = R0.T @ (pt + 0.05 * ey - p0), R0.T @ (pt - 0.05 * ey - p0)

        def pitch() -> float:
            R, p = sim.obj_mat(), sim.obj_pos()
            return float(((R @ qp + p)[2] - (R @ qm + p)[2]) / 0.10)
        return pitch

    # ----------------------------------------------------------------- labels
    def _tipped(self, release: dict[str, Any], low: float) -> bool:
        """Tipped = the object rests on a different face (its CoM height changed by more than a
        fifth after a large tilt; a rolled sphere keeps its CoM height), or it moved away."""
        if release["displacement_xy_m"] > DISPLACE_TOL_M:
            return True
        if release["tilt_deg"] <= TIP_DEG:
            return False
        com_h = float(self.sim.d.subtree_com[self.sim.obj][2] - low)
        return abs(com_h - self.rest_com_height) > 0.2 * max(self.rest_com_height, 1e-3)

    def _label(self, trace, rose, dest, release, planned, v_lift: float = 0.0) -> Rollout:
        sim = self.sim
        touching, pads = sim.contacts()
        low = sim.obj_low_z()
        final = Final(zone=sim.zone_of(touching), pos=sim.obj_pos().tolist(), low_z=float(low),
                      tipped=bool(release is not None and self._tipped(release, low)), held=bool(any(pads)),
                      com_xy=sim.d.subtree_com[sim.obj][:2].tolist())
        out = label_rollout(trace, dest, release, final)
        return Rollout(out["label"], out["max_z"], out["drop_z"], out["loss_phase"], out["loss_step"], out["loss_t"],
                       out["loss_speed"], final.zone, release, planned, trace, summarize(trace, out, v_lift))
