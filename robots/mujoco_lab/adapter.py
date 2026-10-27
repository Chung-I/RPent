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

from rpent.skills.pick_place.destinations import DESTINATIONS, OBJECT_REST_XY, has_surface, release_height
from robots.mujoco_lab.labels import RISE_M, Final, label_rollout
from rpent.skills.pick_place.features import Geometry
from rpent.skills.pick_place.protocol import OUTCOMES  # noqa: F401  (label vocabulary)

LIFT_H = 0.20
HOLD_S = 0.5
V_DOWN = 0.10
V_LOWER = 0.10
A_TRANSPORT = 2.0        # m/s^2: the transport accelerates and brakes at 0.2 g (a step change in speed strips the pads)
CLOSE_S = 0.6
APPROACH_M = 0.12
RELEASE_S = 3.0
TIP_DEG = 45.0
DISPLACE_TOL_M = 0.05

# Primitive fixtures built into every scene (the registry gives the place; this gives the shape).
PRIMITIVE_FIXTURES: list[dict[str, Any]] = [
    {"name": "mouse_pad", "kind": "surface", "size": (0.10, 0.06, 0.002), "rgba": "0.15 0.15 0.18 1"},
    {"name": "plate", "kind": "rim", "radius": 0.11, "rim_height": 0.015, "rgba": "0.92 0.92 0.9 1"},
    {"name": "box", "kind": "cavity", "half": 0.08, "wall_height": 0.12, "rgba": "0.6 0.45 0.3 1"},
]

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

    # ------------------------------------------------------------------ setup
    def _fixture_specs(self) -> list[dict[str, Any]]:
        out = []
        for f in PRIMITIVE_FIXTURES:
            dest = DESTINATIONS[f["name"]]
            out.append({**f, "xy": tuple(dest["xy"]), "mu": dest["mu"]})
        return out

    def reset(self, object_id: str, theta: dict[str, Any] | None = None, seed: int = 0, com_end: int = +1) -> None:
        sc, gr, ob, t3 = self.s["scene"], self.s["grasp"], self.s["objects"], self.s["t3_make"]
        o = ob.BY_ID[object_id]
        props = object_props(object_id)
        th = dict(t3.causes_for(props["mass"])["control"])
        th.update(theta or {})
        shift = (0.0, 0.0, 0.0, 0.0)
        if th.get("com_s") is not None:
            shift = t3.com_shift_for(o, props, float(th["com_s"]), end=com_end)
        flat = not (o["hollow"] or o["liquid"])
        pos, quat = gr.rest_pose(o, props, flat=flat)      # the default rest, as the t3/t3g corpora used
        pos = np.array([pos[0] + OBJECT_REST_XY[0], pos[1] + OBJECT_REST_XY[1], pos[2]])
        xml = sc.build_xml(o, props, mu=th["mu"], mass_scale=th["mass_scale"], com_shift=shift, obj_pos=pos,
                           obj_quat=quat, cam_k=1.0, fixtures=self._fixture_specs())
        self.sim = sc.Sim(xml)
        self.sim.settle(0.2)
        self.obj, self.props, self.theta, self.com_end = o, props, th, com_end
        self.rng = np.random.default_rng(seed)
        self.up0 = self.sim.obj_mat()[:, 2].copy()
        self.rest_low_z = self.sim.obj_low_z()
        self.rest_com_height = float(self.sim.d.subtree_com[self.sim.obj][2] - self.rest_low_z)
        self.episode += 1

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
        }

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
        return {"mass": mass, "mu": float(self.theta["mu"]), "com_s": com_s, "com": com.tolist(), "theta": dict(self.theta)}

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
        com = self.sim.d.subtree_com[self.sim.obj].copy()
        return Geometry(centres=C, normals=N, com=com, length=float(self.props["length"]), base_extent=base,
                        com_height=float(com[2] - low), footprint_half_width=footprint_half)

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
        for f in PRIMITIVE_FIXTURES:
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
        if destination not in self.fixtures_in_scene and DESTINATIONS[destination]["kind"] in ("rim", "cavity") \
                and destination not in {f["name"] for f in PRIMITIVE_FIXTURES}:
            raise ValueError(f"fixture {destination!r} is not in the scene")
        return DESTINATIONS[destination]

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
        low_off = self.rest_low_z - hand_z0           # object's lowest point relative to the hand (held)
        surf0, _ = self._surface_below(pt[:2])
        z_rest = self.rest_low_z - surf0
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
        ]
        return segs

    # -------------------------------------------------------------- execution
    def execute(self, proposal: Proposal, force: float, v_lift: float, v_transport: float = 0.5,
                destination: str = "bin", release_height: float | None = None) -> Rollout:
        self._need()
        sc, gr = self.s["scene"], self.s["grasp"]
        dest = self._dest(destination)
        planned = self.plan_trajectory(proposal, v_lift, v_transport, destination, release_height)
        sim = self.sim
        hz = sc.CTRL_HZ
        pt = np.asarray(proposal.point, dtype=float)
        p0z = sim.obj_pos()[2]
        R0 = sim.obj_mat().copy()
        trace: list[dict[str, Any]] = []

        def on_step(phase: str) -> None:
            pos = sim.obj_pos()
            low = sim.obj_low_z()
            lin, _ = sim.obj_vel()
            touching, pads = sim.contacts()
            surf, has = self._surface_below(pos[:2])
            ang = float(np.degrees(np.arccos(np.clip(((sim.obj_mat() @ R0.T).trace() - 1) / 2, -1, 1))))
            trace.append({"t": len(trace) / hz, "phase": phase, "z": float(pos[2] - p0z), "low_z": float(low),
                          "surface_z_below": surf, "has_surface": has, "speed": float(np.linalg.norm(lin)),
                          "pads": list(pads), "zone": sim.zone_of(touching), "rot_deg": ang})

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
        return self._label(trace, rose, dest, release, planned)

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

    def _label(self, trace, rose, dest, release, planned) -> Rollout:
        sim = self.sim
        touching, _ = sim.contacts()
        low = sim.obj_low_z()
        final = Final(zone=sim.zone_of(touching), pos=sim.obj_pos().tolist(), low_z=float(low),
                      tipped=bool(release is not None and self._tipped(release, low)))
        out = label_rollout(trace, dest, release, final)
        return Rollout(out["label"], out["max_z"], out["drop_z"], out["loss_phase"], out["loss_step"], out["loss_t"],
                       out["loss_speed"], final.zone, release, planned, trace)
