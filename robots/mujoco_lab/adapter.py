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
release, and label the outcome. A contact monitor flags the runs whose pad contacts
were not physical, and those after a reset that did not reach rest (``artifact``).
A damaged run with a jam is an artifact too (JAM_N).

Milestone 2b: a fillable object may hold a liquid (``reset(fill=...)``: a point mass at the liquid's centroid,
fragility.liquid), the object may carry a hidden fragility (``reset(frag=...)``), and the damage meter
(meter.DamageMeter) runs in every physics substep of an attempt or a probe. Its first threshold crossing
(fragility.damage_event) replaces the motion label (``Rollout.damage``, labels.final_label) and makes a probe's kind
"damaged". The squeeze probe (``probe_squeeze``) presses without lifting and reads the stiffness.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
import zlib
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from rpent.skills.pick_place.destinations import DESTINATIONS, OBJECT_REST_XY, PRIMITIVE_IDS, has_surface, is_mesh, release_height
from robots.mujoco_lab.fragility import FLAGS, THRESHOLDS_V0, Thresholds, damage_event, entry_of, liquid, load_classes, spill_angle
from robots.mujoco_lab.labels import Final, final_label, first_lifted, label_rollout, lifted
from robots.mujoco_lab.meter import DamageMeter
from robots.mujoco_lab.readouts import RISEN_M, slide_force, summarize
from rpent.skills.pick_place.features import G, Geometry
from rpent.skills.pick_place.probes import ProbeResult
from rpent.skills.pick_place.protocol import OUTCOMES  # noqa: F401  (label vocabulary)

LIFT_H = 0.20
HOLD_S = 0.5
V_DOWN = 0.10
V_LOWER = 0.10
A_TRANSPORT = 2.0        # m/s^2: the transport accelerates and brakes at 0.2 g (a step change in speed strips the pads)
RAMP_S = 0.2             # s: the lift, the lower and the retreats reach their speed in 0.2 s and brake in 0.2 s
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
PINCH_MARGIN_M = 0.004       # the skill pinches at most the pads' open gap less 2 mm a side (see max_pinch_width)
REST_MIN_S = 0.2             # reset: settle at least this long, then until the object is at rest,
REST_CAP_S = 3.0             # at most this long (an object still moving then starts flagged: reset_info at_rest False)
REST_WINDOW_S = 0.5          # at rest: over the last 0.5 s it moved under 1 mm and turned under 1 deg
REST_MOVE_M = 0.001
REST_TURN_DEG = 1.0
REV_COS = -0.5               # the contact monitor (artifact): a pad-hull contact is reversed when its normal into the
REV_N = 1.0                  # object points against the pad's closing direction (cosine below REV_COS) with more than
REV_STEPS = 5                # REV_N newtons; a run is an artifact with reversed contacts in REV_STEPS squeeze substeps
DEEP_M = 0.003               # or more, a pad penetration over DEEP_M, or a reset that did not reach rest
JAM_N = 50.0                 # a jam: the meter's 10 ms pad force above the run's largest grip drive by more than JAM_N.
                             # The position-driven hand has no force limit: a pad that wedged the carton against the
                             # table pressed 7,373 N with the grip open. No grip drive or force-limited arm gives that,
                             # so a damaged run with a jam is an artifact
SQUEEZE_F0 = 1.0             # the squeeze probe: close with 1 N a pad, ramp to its force over 2 s, hold 0.3 s, open
SQUEEZE_RAMP_S = 2.0
SQUEEZE_HOLD_S = 0.3
SQUEEZE_NOISE = 0.10         # its reading: k_hat = k exp(0.10 eps) (plan 2b, decision 9: a synthetic sensor)
PAD_FORCE_NOISE = 0.05       # every force reading of the hand is x exp(0.05 eps) (plan 2b2, decision 1): the error the
                             # skill's laws assume (likelihood.SD_FLOOR); the sim's forces are exact to 0.01 %
FORCE_STREAM = 1             # the force readings' stream per reset: (seed, crc32(object), FORCE_STREAM)
FORCE_NOISE_ENV = "RPENT_LAB_FORCE_NOISE"
LIQUID_ANCHOR = '<freejoint name="obj_free"/>'

def force_noise_level(value: float | None = None) -> float:
    """The lab's force noise: ``value``, else RPENT_LAB_FORCE_NOISE, else PAD_FORCE_NOISE; a finite number at least 0
    (0 gives the clean readings of milestone 2b)."""
    raw = os.environ.get(FORCE_NOISE_ENV, PAD_FORCE_NOISE) if value is None else value
    try:
        sd = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"the force noise must be a number at least 0, not {raw!r}") from None
    if not (math.isfinite(sd) and sd >= 0.0):
        raise ValueError(f"the force noise must be a finite number at least 0, not {raw!r}")
    return sd


def noisy_force(value: float | None, eps: float, sd: float) -> float | None:
    """A force reading with the declared error: value x exp(sd eps); None stays None."""
    return None if value is None else float(value) * math.exp(float(sd) * float(eps))


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


def artifact(rev_steps: int, deepest_mm: float, at_rest: bool = True) -> bool:
    """True when a run's pad contacts were not physical: reversed contacts in REV_STEPS or more substeps of the
    squeeze (the native convex collider on thin hull walls), or a pad penetration over DEEP_M (pads driven into
    the object); or when the reset did not reach rest (an object that chatters on the table)."""
    return rev_steps >= REV_STEPS or deepest_mm > 1000 * DEEP_M or not at_rest


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
    rev_steps: int = 0                  # the contact monitor (see MuJoCoLabEnv._watch_contacts and artifact)
    deepest_mm: float = 0.0
    artifact: bool = False
    meter: dict[str, Any] = field(default_factory=dict)     # the damage meter's channels (meter.DamageMeter); {} when off
    motion_label: str | None = None                          # the motion rules' label; ``label`` is it unless damage
    damage: dict[str, Any] | None = None                     # the first threshold crossing: label, channel, t
    sensor: dict[str, Any] = field(default_factory=dict)     # the force noise, its eps and the clean readouts; no planner sees it


def _with_liquid(xml: str, liq: dict[str, Any]) -> str:
    """The scene with the liquid as a point mass at its centroid in the object body (like the CoM point mass): it adds
    its mass and moves the CoM, and does not slosh in the physics (the meter models the slosh)."""
    if xml.count(LIQUID_ANCHOR) != 1:
        raise RuntimeError("the object body's free joint, where the liquid goes, is not in the scene")
    c = liq["centroid_obj"]
    geom = (f'<geom name="liquid" type="sphere" size="0.005" pos="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" mass="{liq["mass_kg"]:.6f}" '
            'contype="0" conaffinity="0" rgba="0 0 0 0" group="3"/>')
    return xml.replace(LIQUID_ANCHOR, LIQUID_ANCHOR + "\n      " + geom)


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


def ramp_profile(distance: float, v_max: float, hz: float) -> list[float]:
    """Path fractions per control step for the lift, the lower and the retreats: transport_profile's
    trapezoid with the acceleration that reaches v_max in RAMP_S. A lift that started at full speed
    yanked the hand at 10 g at 0.8 m/s."""
    return transport_profile(distance, v_max, hz, accel=v_max / RAMP_S)


class MuJoCoLabEnv:
    """One object on the table with hidden properties, the pads, and the place fixtures."""

    def __init__(self, sim_dir: str | None = None, render: bool = False, damage_meter: bool = True,
                 slosh_zeta: float | None = None, force_noise: float | None = None):
        self.force_noise = force_noise_level(force_noise)
        self.s = load_sim(sim_dir)
        self.render = render
        self.damage_meter = bool(damage_meter)
        th0 = Thresholds.from_json(THRESHOLDS_V0)
        self.slosh_zeta = float(th0.slosh_zeta if slosh_zeta is None else slosh_zeta)
        self.slosh_scale = th0.slosh_scale          # the thresholds' forcing factors per profile class and fill
        self._classes: dict[str, Any] | None = None
        self.entry: dict[str, Any] = {"class": "none", "law": None, "fillable": False, "cavity": None}
        self.fill, self.frag, self.liquid, self.spill_deg = 0.0, None, None, None
        self._dmeter: DamageMeter | None = None
        self._jam = False
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

    def fragility_classes(self) -> dict[str, Any]:
        if self._classes is None:
            self._classes = load_classes()
        return self._classes

    def reset(self, object_id: str, theta: dict[str, Any] | None = None, seed: int = 0, com_end: int = +1,
              fixture: str | None = None, fill: float = 0.0, frag: dict[str, Any] | None = None) -> None:
        """``fill``: the share of a fillable object's cavity that holds water (0: empty); ``frag``: its hidden
        fragility (fragility.map_latent's keys; None: nothing can damage it)."""
        sc, gr, ob, t3 = self.s["scene"], self.s["grasp"], self.s["objects"], self.s["t3_make"]
        o = ob.BY_ID[object_id]
        props = object_props(object_id)
        entry, _ = entry_of(object_id, self.fragility_classes())
        fill = float(fill or 0.0)
        if not 0.0 <= fill <= 1.0:
            raise ValueError(f"fill {fill} is outside [0, 1]")
        if fill > 0.0 and not entry.get("fillable"):
            raise ValueError(f"{object_id} is not a fillable container (fragility_classes.json)")
        liq = liquid(entry["cavity"], fill) if fill > 0.0 else None
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
        if liq is not None:          # the rest pose above is the empty container's: the reset settles the filled one
            xml = _with_liquid(xml, liq)
        self.entry, self.fill, self.frag, self.liquid = entry, fill, (dict(frag) if frag else None), liq
        self.spill_deg = spill_angle(entry["cavity"], fill) if liq is not None else None
        self._cavity_axis = np.asarray(entry["cavity"]["axis_obj"], dtype=float) if liq is not None else None
        self.sim = sc.Sim(xml)
        self._install_meter()
        self.fixtures_in_scene = names
        self.fixture = fixture
        self._onto_table()
        self.reset_info = self._settle_to_rest()
        self.obj, self.props, self.theta, self.com_end = o, props, th, com_end
        # the reset's stream (the squeeze noise) is per seed and object: every object of a draw is reset with one seed
        self.rng = np.random.default_rng([int(seed), zlib.crc32(object_id.encode())])
        # the force readings' own stream, so the squeeze's stays as it was
        self.force_rng = np.random.default_rng([int(seed), zlib.crc32(object_id.encode()), FORCE_STREAM])
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

    def _pad_hull_sums(self) -> np.ndarray:
        """Over the pad-hull contacts now: [the net vertical force of both pads on the object, the left
        pad's normal force, the right pad's, and each pad's normal force times the vertical part of its
        contact normal into the object (-1 for a pad that presses down on a top face, 0 on a vertical side)].

        The vertical force is signed (up is positive), so in a static two-pad hold it equals the
        weight whatever the squeeze; a per-contact absolute sum (t3_probes.pad_forces) also counts
        the normal force's vertical parts on curved or sloped faces. mj_contactForce gives the force
        in the contact frame, whose normal points from geom1 to geom2, as the force on geom2."""
        import mujoco

        sim = self.sim
        f = np.zeros(6)
        hull = set(sim.hull_ids)
        pads = (sim.pad_l, sim.pad_r)
        out = np.zeros(5)
        for i in range(sim.d.ncon):
            c = sim.d.contact[i]
            if c.geom1 in pads or c.geom2 in pads:
                k = 0 if sim.pad_l in (c.geom1, c.geom2) else 1
                other = c.geom2 if c.geom1 in pads else c.geom1
                if other in hull:
                    mujoco.mj_contactForce(sim.m, sim.d, i, f)
                    frame = c.frame.reshape(3, 3)
                    sign = 1.0 if other == c.geom2 else -1.0       # the force and the normal on the object
                    out[0] += sign * float((frame.T @ f[:3])[2])
                    out[1 + k] += abs(float(f[0]))
                    out[3 + k] += abs(float(f[0])) * sign * float(frame[0, 2])
        return out

    def _watch_contacts(self) -> None:
        """The contact monitor, after a physics substep while the grip squeezes: count the substep in
        ``rev_steps`` when a pad-hull contact is reversed (REV_COS, REV_N), and keep the deepest pad
        penetration in ``deepest_mm`` (the rule of the experiment's fix_impact/impact.py)."""
        import mujoco

        sim, w = self.sim, self._watch
        hull, f, rev = set(sim.hull_ids), np.zeros(6), False
        xh = sim.d.xmat[sim.hand].reshape(3, 3)[:, 0]
        for i in range(sim.d.ncon):
            c = sim.d.contact[i]
            for pad, close_dir in ((sim.pad_l, xh), (sim.pad_r, -xh)):
                if pad in (c.geom1, c.geom2) and ({c.geom1, c.geom2} - {pad}) & hull:
                    w["deepest_mm"] = max(w["deepest_mm"], -1000 * float(c.dist))
                    n = c.frame[:3] * (1.0 if c.geom1 == pad else -1.0)            # pad -> object
                    if float(n @ close_dir) < REV_COS:
                        mujoco.mj_contactForce(sim.m, sim.d, i, f)
                        rev = rev or f[0] > REV_N
        w["rev_steps"] += int(rev)

    def _monitor(self) -> dict[str, Any]:
        """The contact monitor's record of the run since the last trace began, and the artifact flag."""
        w = self._watch
        return {"rev_steps": int(w["rev_steps"]), "deepest_mm": float(w["deepest_mm"]),
                "artifact": artifact(w["rev_steps"], w["deepest_mm"], bool(self.reset_info["at_rest"])) or self._jam}

    def _install_meter(self) -> None:
        """Wrap the sim's control step: while a trace records (``_metering``), each step also keeps the mean
        of ``_pad_hull_sums`` over its physics substeps in ``_step_forces``. The substeps go through
        ``Sim.substep``, so a metered step moves the hand target smoothly too. The forces vary within a step,
        and a force sampled at the end of one can miss most of it: while the target still jumped once per
        step, a carried lift read 0.9 of the weight and a light object's lift step read below zero (the push
        averages the same way, see _push_step). While the grip squeezes, each metered substep also feeds the
        contact monitor (_watch_contacts). The wrapper holds its sim by a weak reference: a strong one
        ties each sim to itself, and a worker's old sims then live until the cycle collector runs."""
        import weakref

        ref, plain, sub = weakref.ref(self.sim), type(self.sim).step_ctrl, int(self.s["scene"].SUB)
        plain_sub = type(self.sim).substep
        self._metering, self._step_forces = False, None
        self._watch = {"rev_steps": 0, "deepest_mm": 0.0}
        self._dmeter, self._hull_set, self._fbuf = None, set(self.sim.hull_ids), np.zeros(6)

        def substep() -> None:          # every physics step of the scene: the damage meter reads after it
            sim = ref()
            plain_sub(sim)
            if self._dmeter is not None:
                self._damage_substep()

        def step_ctrl() -> None:
            sim = ref()
            if not self._metering:
                plain(sim)
                return
            acc = np.zeros(5)
            for _ in range(sub):
                sim.substep()
                acc += self._pad_hull_sums()
                if sim.d.ctrl[0] > 0:
                    self._watch_contacts()
            self._step_forces = acc / sub

        self.sim.step_ctrl = step_ctrl
        self.sim.substep = substep

    def _damage_substep(self) -> None:
        """One physics substep of the damage meter: the pads' normal forces on the object, its (hull, geom) contacts with
        non-pad geoms, the CoM velocity and the cavity's axis. A release (the grip opens after a squeeze) or a loss
        (both pads held it with nothing under it, then neither touches it) arms the impact channel. Reads only."""
        import mujoco

        sim, f, hull = self.sim, self._fbuf, self._hull_set
        pl, pr = (sim.pad_l, sim.pad_r) if sim.has_hand else (-1, -1)
        pad_n, touch, touching = [0.0, 0.0], [False, False], set()
        for i in range(sim.d.ncon):
            c = sim.d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in hull) == (g2 in hull):
                continue
            h, other = (g1, g2) if g1 in hull else (g2, g1)
            if other == pl or other == pr:
                k = 0 if other == pl else 1
                mujoco.mj_contactForce(sim.m, sim.d, i, f)
                pad_n[k] += abs(float(f[0]))
                touch[k] = True
            else:
                touching.add((h, other))
        ctrl = float(sim.d.ctrl[0]) if sim.m.nu else 0.0
        hi = float(sim.m.actuator_ctrlrange[0, 1]) if sim.m.nu and sim.m.actuator_ctrllimited[0] else np.inf
        self._squeezed = self._squeezed or ctrl > 0.0
        self._held = self._held or (touch[0] and touch[1] and not touching)
        if not self._armed and ((self._squeezed and ctrl <= 0.0) or (self._held and not (touch[0] or touch[1]))):
            self._armed = True
        m = self._dmeter
        axis = sim.obj_mat() @ self._cavity_axis if m.slosh is not None else None
        m.substep(pad_n, touching, sim.obj_vel()[0], axis, self._armed, drive=min(max(ctrl, 0.0), hi))

    def _meter_begin(self) -> None:
        """Start the damage meter for an attempt or a probe (its times count from here); off: no meter."""
        from robots.mujoco_lab.slosh import Slosh, scale_at

        self._squeezed = self._held = self._armed = self._jam = False
        self._dmeter = None
        if self.damage_meter:
            slosh = (Slosh(self.entry["cavity"], self.fill, self.slosh_zeta, scale=scale_at(self.slosh_scale, self.fill))
                     if self.liquid is not None else None)
            self._dmeter = DamageMeter(float(self.s["scene"].DT), slosh)

    def _slosh_profile(self, profile: str) -> None:
        """The slosh model's profile class from the next physics substep on: carry over the transport, jolt elsewhere."""
        m = self._dmeter
        if m is not None and m.slosh is not None:
            m.slosh.profile = profile

    def _meter_end(self) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """(the meter's channels, the first threshold crossing as {label, channel, t} or None) and stop the meter."""
        m, self._dmeter = self._dmeter, None
        if m is None:
            return {}, None
        data = m.to_data()
        ev = damage_event(data, self.frag or {}, self.entry, self.fill, spill_deg=self.spill_deg)
        self._jam = ev is not None and m.jam_n > JAM_N
        return data, (None if ev is None else {"label": ev[0], "channel": ev[1], "t": round(float(ev[2]), 4)})

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
                "theta": dict(self.theta), "fill": self.fill, "liquid_mass": float(self.liquid["mass_kg"]) if self.liquid else 0.0,
                "frag": None if self.frag is None else dict(self.frag)}

    def pose_info(self) -> dict[str, Any]:
        """The object's current pose and what a CoM hypothesis needs (object frame, nominal values)."""
        self._need()
        sim, t3 = self.sim, self.s["t3_make"]
        ax, lo, L = t3.pivot_axis(self.obj, self.props)
        return {
            "object_id": self.obj["id"],
            "R": sim.obj_mat().ravel().tolist(),
            "p": sim.obj_pos().tolist(),
            "low_z": float(sim.obj_low_z()),
            "base_extent": float(self.geometry().base_extent),
            "length": float(self.props["length"]),
            "com_obj_nominal": [float(x) for x in self.props["com"]],
            "pivot": {"centre": [float(x) for x in self.props["centre"]], "axis": [float(x) for x in ax], "lo": float(lo), "L": float(L)},
            "nominal_mass": float(self.props["mass"]),
            "entry": {k: self.entry.get(k) for k in FLAGS},          # the public fragility flags (milestone 2b)
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

    def _hull_vertex_list(self) -> list[np.ndarray]:
        """Each collision hull's vertices in the world frame (MuJoCo collides each one's convex hull)."""
        sim = self.sim
        out = []
        for g in sim.hull_ids:
            mid = sim.m.geom_dataid[g]
            v0, nv = sim.m.mesh_vertadr[mid], sim.m.mesh_vertnum[mid]
            out.append(sim.m.mesh_vert[v0:v0 + nv] @ sim.d.geom_xmat[g].reshape(3, 3).T + sim.d.geom_xpos[g])
        return out

    def _hull_vertices_world(self) -> np.ndarray:
        return np.concatenate(self._hull_vertex_list())

    def geometry(self) -> Geometry:
        self._need()
        C, N = self.hull_faces_world()
        hulls = self._hull_vertex_list()
        V = np.concatenate(hulls)
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
                        com_height=float(com[2] - low), footprint_half_width=footprint_half, hulls=tuple(hulls))

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

    @property
    def max_pinch_width(self) -> float:
        """The widest exact pinch the skill may plan: the pads' gap at the joints' open end (two fingers at
        MAX_OPEN less two pad thicknesses: 9.4 cm) less PINCH_MARGIN_M. A wider pinch lands the pads on the
        object; the menu (and the corpora) leave it out."""
        sc = self.s["scene"]
        return 2 * (float(sc.MAX_OPEN) - 2 * float(sc.PAD_HALF[0])) - PINCH_MARGIN_M

    def pinch_width(self, point: np.ndarray, yaw: float) -> float:
        """The hulls' extent between the pads (features.pinch_width); nan when the pads close on nothing."""
        from rpent.skills.pick_place.features import pinch_width

        return pinch_width(self._hull_vertex_list(), point, yaw, self.s["scene"].PAD_HALF)

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
                        release_height: float | None = None, a_transport: float = A_TRANSPORT) -> list[Segment]:
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
            Segment("lift", len(ramp_profile(LIFT_H, v_lift, hz)), z_rest, low_top - surf0, True, v_lift),
            Segment("hold", int(HOLD_S * hz), low_top - surf0, low_top - surf0, True, 0.0),
            Segment("transport", len(transport_profile(float(np.linalg.norm(dxy)), v_transport, hz, accel=a_transport)), low_top - surf0,
                    low_top - (surf_d if has_d else sc.TABLE_TOP), has_d, v_transport),
            Segment("lower", len(ramp_profile(z_hand_top - z_hand_rel, V_LOWER, hz)), low_top - (surf_d if has_d else sc.TABLE_TOP),
                    target_low - (surf_d if has_d else sc.TABLE_TOP), has_d, V_LOWER),
            Segment("release", int(RELEASE_S * hz), target_low - (surf_d if has_d else sc.TABLE_TOP),
                    target_low - (surf_d if has_d else sc.TABLE_TOP), has_d, 0.0),
            Segment("retreat", len(ramp_profile(RETREAT_M, V_LOWER, hz)) + int(RETREAT_SETTLE_S * hz),
                    target_low - (surf_d if has_d else sc.TABLE_TOP), target_low - (surf_d if has_d else sc.TABLE_TOP), has_d, V_LOWER),
        ]
        return segs

    # -------------------------------------------------------------- execution
    def execute(self, proposal: Proposal, force: float, v_lift: float, v_transport: float = 0.5,
                destination: str = "bin", release_height: float | None = None, a_transport: float = A_TRANSPORT) -> Rollout:
        self._need()
        sc, gr = self.s["scene"], self.s["grasp"]
        dest = self._dest(destination)
        self._refresh_baselines()
        planned = self.plan_trajectory(proposal, v_lift, v_transport, destination, release_height, a_transport)
        self._meter_begin()
        sim = self.sim
        hz = sc.CTRL_HZ
        pt = np.asarray(proposal.point, dtype=float)
        trace, on_step = self._tracer(pt, proposal.yaw)
        out, _, _ = gr.run_grasp(sim, pt, proposal.yaw, force=force, v_lift=v_lift, lift_h=LIFT_H, hold_s=HOLD_S,
                                 on_step=on_step, lift_profile=ramp_profile(LIFT_H, v_lift, hz))
        rose = first_lifted(trace, ("lift", "hold")) is not None
        release = None
        if rose:
            hand_z0 = gr.hand_start_z(pt)
            z = hand_z0 + LIFT_H
            xy = pt[:2].copy()
            goal = np.asarray(dest["xy"], dtype=float)
            self._slosh_profile("carry")
            for frac in transport_profile(float(np.linalg.norm(goal - xy)), v_transport, hz, accel=a_transport):
                cur = xy + (goal - xy) * frac
                sim.set_hand([cur[0], cur[1], z], proposal.yaw)
                sim.step_ctrl()
                on_step("transport")
            self._slosh_profile("jolt")
            low_off = sim.obj_low_z() - sim.d.xpos[sim.hand][2]
            target_low = self._release_target_low_z(dest, release_height)
            z_rel = target_low - low_off
            z0 = z
            for frac in ramp_profile(z0 - z_rel, V_LOWER, hz):
                z = z0 + (z_rel - z0) * frac
                sim.set_hand([goal[0], goal[1], z], proposal.yaw)
                sim.step_ctrl()
                on_step("lower")
            release = self._open_and_rise(goal[0], goal[1], z, proposal.yaw, trace, on_step)
        else:           # not lifted (a pivot holds one end up): let it down where it is and carry nothing off
            self._open_and_rise(pt[0], pt[1], gr.hand_start_z(pt) + LIFT_H, proposal.yaw, trace, on_step)
        self._metering = False
        meter, damage = self._meter_end()
        return self._label(trace, rose, dest, release, planned, v_lift, meter, damage)

    def _open_and_rise(self, x: float, y: float, z: float, yaw: float, trace: list[dict[str, Any]], on_step) -> dict[str, Any]:
        """Open the gripper with the hand at (x, y, z) for RELEASE_S, then rise RETREAT_M and settle; the release
        record. The open hand moves away; a wedged object comes with it."""
        sim, gr, hz = self.sim, self.s["grasp"], self.s["scene"].CTRL_HZ
        release = gr.run_release(sim, None, 1, [], [], len(trace), seconds=RELEASE_S, on_step=on_step, up0=self.up0)
        for frac in ramp_profile(RETREAT_M, V_LOWER, hz):
            sim.set_hand([x, y, z + RETREAT_M * frac], yaw)
            sim.step_ctrl()
            on_step("retreat")
        for _ in range(int(RETREAT_SETTLE_S * hz)):
            sim.step_ctrl()
            on_step("retreat")
        return release

    def _tracer(self, pt: np.ndarray, yaw: float):
        """(trace, on_step): on_step(phase) appends one dict per control step (see execute's trace). ``pads``
        tells whether each pad pressed on the object in any physics substep of the step: the contact at the end
        of a step flickers while an object chatters in the pads (wood_block d78 showed no pad at the end of a
        step over which the pads carried 13.8 N). The pad forces are means over the step's physics substeps
        (``_install_meter``): ``fz`` the pads' net
        vertical force on the object, ``fn`` their normal force, ``pad_nz`` each pad's force-weighted
        vertical part of its contact normal into the object (None without contact). ``hand_z`` is the
        hand's height, ``rot_lift_deg`` the object's turn since the lift began and ``grip_dz`` the rise since
        then of the object point that sat between the pads at the end of the close (both None before it)."""
        sim = self.sim
        hz = self.s["scene"].CTRL_HZ
        p0z = sim.obj_pos()[2]
        R0 = sim.obj_mat().copy()
        trace: list[dict[str, Any]] = []
        pitch_of = self._pitch_meter(pt, yaw)
        pitch0: list[float] = []
        self._metering = True
        self._watch = {"rev_steps": 0, "deepest_mm": 0.0}

        def turn(A: np.ndarray, B: np.ndarray) -> float:
            return float(np.degrees(np.arccos(np.clip(((A @ B.T).trace() - 1) / 2, -1, 1))))

        def mid() -> np.ndarray:
            return (sim.d.geom_xpos[sim.pad_l] + sim.d.geom_xpos[sim.pad_r]) / 2

        prev, lift_R, grip = [(R0, sim.obj_pos(), mid())], [], []

        def on_step(phase: str) -> None:
            pos = sim.obj_pos()
            low = sim.obj_low_z()
            lin, _ = sim.obj_vel()
            touching, _ = sim.contacts()
            surf, has = self._surface_below(pos[:2])
            R = sim.obj_mat().copy()
            if phase == "lift" and not lift_R:
                Rc, pc, mc = prev[0]                             # the pose at the end of the close
                lift_R.append(Rc)
                grip.extend((Rc.T @ (mc - pc), float(mc[2])))   # the point between the pads, in the object frame
            prev[0] = (R, pos, mid())
            acc = self._step_forces
            pad_nz = [float(acc[3 + k] / acc[1 + k]) if acc[1 + k] > 0 else None for k in (0, 1)]
            pitch = pitch_of()
            if not pitch0:
                pitch0.append(pitch)
            trace.append({"t": len(trace) / hz, "phase": phase, "z": float(pos[2] - p0z), "low_z": float(low),
                          "surface_z_below": surf, "has_surface": has, "speed": float(np.linalg.norm(lin)),
                          "pads": [bool(acc[1 + k] > 0) for k in (0, 1)], "zone": sim.zone_of(touching), "rot_deg": turn(R, R0),
                          "fz": float(acc[0]), "fn": float(acc[1] + acc[2]), "pad_nz": pad_nz,
                          "hand_z": float(sim.d.xpos[sim.hand][2]), "rot_lift_deg": turn(R, lift_R[0]) if lift_R else None,
                          "grip_dz": float((R @ grip[0] + pos)[2] - grip[1]) if grip else None,
                          "pitch_ey": float(pitch - pitch0[0]),
                          "t_meter": None if self._dmeter is None else round(self._dmeter.t, 4)})
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
        step. A force sampled at the end of the step misses its variation within the step (while the hand
        target still jumped once per step, the push on a light, slippery object was a train of strikes)."""
        sim, sub = self.sim, int(self.s["scene"].SUB)
        total = 0.0
        for _ in range(sub):
            sim.substep()
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
        x, y, z0 = (float(v) for v in sim.d.mocap_pos[sim.mocap])
        for frac in ramp_profile(RETREAT_M, V_LOWER, hz):
            sim.set_hand([x, y, z0 + RETREAT_M * frac], yaw)
            sim.step_ctrl()
            steps[0] += 1

    @staticmethod
    def weigh_reading(trace: list[dict[str, Any]], force: float) -> tuple[str, dict[str, float]]:
        """The weigh probe's kind and reading from its trace (see probe_weigh)."""
        rest = float(trace[0]["low_z"]) if trace else 0.0
        risen = [i for i, x in enumerate(trace) if x["phase"] in ("lift", "hold") and lifted(x, rest, RISEN_M)]
        if risen:
            i0 = risen[0]
            if any(not any(x["pads"]) for x in trace[i0:] if x["phase"] in ("lift", "hold")):
                return "lost", {}
            held = [x["fz"] for x in trace[i0:] if x["phase"] == "hold" and lifted(x, rest, RISEN_M)]
            seg = held or [trace[i]["fz"] for i in risen]
            return "measured", {"m_hat": float(np.median(seg)) / G}
        fmax = slide_force(trace)
        if fmax is None:
            return "lower_bound", {}
        return "lower_bound", {"mu_hat": fmax / (2.0 * float(force)), "m_lower": fmax / G}

    def probe_weigh(self, proposal: Proposal, force: float) -> ProbeResult:
        """Close with ``force``, lift 2 cm at 3 cm/s, hold 0.5 s (the execute trace); then open, settle
        0.5 s, rise 10 cm, settle 1 s. Kinds: measured (lifted off the table, labels.lifted with a 1.2 cm rise:
        m_hat = the median net vertical pad force over the still hold while lifted, / g; a pivot keeps one end
        on the table, which carries part of the weight), lower_bound (never lifted: mu_hat = the largest
        force while the pads slid up the still object (readouts.slide_force) / 2F, m_lower = that force
        / g; no reading when the pads never did), lost (lifted, then both pads lost it). The forces are
        means over each step's physics substeps (the hand follows its mocap target in jumps)."""
        self._need()
        self._refresh_baselines()
        gr, sim, hz = self.s["grasp"], self.sim, self.s["scene"].CTRL_HZ
        pt = np.asarray(proposal.point, dtype=float)
        p_start, R_start = sim.obj_pos().copy(), sim.obj_mat().copy()
        self._meter_begin()
        trace, on_step = self._tracer(pt, proposal.yaw)
        gr.run_grasp(sim, pt, proposal.yaw, force=force, v_lift=WEIGH_V_LIFT, lift_h=WEIGH_LIFT_M, hold_s=WEIGH_HOLD_S,
                     on_step=on_step)
        self._metering = False
        kind, reading = self.weigh_reading(trace, force)
        reading, sensor = self._force_reading(reading)
        steps = [len(trace)]
        sim.grip(-20.0)
        for _ in range(int(WEIGH_OPEN_S * hz)):
            sim.step_ctrl()
            steps[0] += 1
        self._rise_hand(proposal.yaw, steps)
        for _ in range(int(WEIGH_SETTLE_S * hz)):
            sim.step_ctrl()
            steps[0] += 1
        meter, damage = self._meter_end()
        disp, tilt, thrown = self._probe_cost(p_start, R_start)
        return ProbeResult("weigh", "damaged" if damage else kind, reading, disp, tilt, thrown, steps[0] / hz, trace, **self._monitor(),
                           meter=meter, damage=damage, sensor=sensor)

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
        self._watch = {"rev_steps": 0, "deepest_mm": 0.0}             # the open hand never squeezes
        self._meter_begin()
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
        meter, damage = self._meter_end()
        disp, tilt, thrown = self._probe_cost(p_start, R_start)
        reading, sensor = self._force_reading(reading)
        return ProbeResult("push", "damaged" if damage else kind, reading, disp, tilt, thrown, steps[0] / hz, rec, **self._monitor(),
                           meter=meter, damage=damage, sensor=sensor)

    def _force_reading(self, reading: dict[str, float]) -> tuple[dict[str, float], dict[str, Any]]:
        """A probe's force reading with the declared error: one eps from the force stream scales every value (the
        weigh's m_hat, or its mu_hat and m_lower from one slide force; the push's f_break), drawn also when the
        reading is empty, so the stream stays aligned. Returns the reading and the sensor record."""
        eps = float(self.force_rng.standard_normal())
        noisy = {k: noisy_force(v, eps, self.force_noise) for k, v in reading.items()}
        return noisy, {"force_noise": self.force_noise, "eps": eps, "clean": dict(reading)}

    @staticmethod
    def squeeze_reading(k: float | None, rng: np.random.Generator) -> dict[str, float]:
        """The squeeze probe's reading: k_hat = k exp(SQUEEZE_NOISE eps), eps a standard normal it keeps (``eps``); no
        k_hat for an object without a stiffness."""
        eps = float(rng.standard_normal())
        out = {"eps": eps}
        if k is not None:
            out["k_hat"] = float(k) * math.exp(SQUEEZE_NOISE * eps)
        return out

    def probe_squeeze(self, proposal: Proposal, f_max: float) -> ProbeResult:
        """Close at the proposal with SQUEEZE_F0 a pad, ramp each pad's force to ``f_max`` over SQUEEZE_RAMP_S, hold
        SQUEEZE_HOLD_S, open, rise 10 cm and settle; no lift. Kinds: measured (reading ``squeeze_reading`` of the hidden
        stiffness, eps from the reset's stream, per seed and object) and damaged (the meter's first threshold crossing). The trace's
        steps carry the commanded force ``f_cmd`` and the meter's time ``t_meter``."""
        self._need()
        self._refresh_baselines()
        gr, sim, hz = self.s["grasp"], self.sim, self.s["scene"].CTRL_HZ
        pt = np.asarray(proposal.point, dtype=float)
        p_start, R_start = sim.obj_pos().copy(), sim.obj_mat().copy()
        f_max = max(float(f_max), SQUEEZE_F0)
        self._meter_begin()
        trace, on_step = self._tracer(pt, proposal.yaw)
        f_cmd = [SQUEEZE_F0]

        def on(phase: str) -> None:
            on_step(phase)
            trace[-1]["f_cmd"] = f_cmd[0]

        gr.run_grasp(sim, pt, proposal.yaw, force=SQUEEZE_F0, v_lift=WEIGH_V_LIFT, lift_h=0.0, hold_s=0.0, on_step=on)
        n = int(round(SQUEEZE_RAMP_S * hz))
        for k in range(1, n + 1):
            f_cmd[0] = SQUEEZE_F0 + (f_max - SQUEEZE_F0) * k / n
            sim.grip(f_cmd[0])
            sim.step_ctrl()
            on("ramp")
        for _ in range(int(round(SQUEEZE_HOLD_S * hz))):
            sim.step_ctrl()
            on("hold")
        self._metering = False
        steps = [len(trace)]
        sim.grip(-20.0)
        for _ in range(int(WEIGH_OPEN_S * hz)):
            sim.step_ctrl()
            steps[0] += 1
        self._rise_hand(proposal.yaw, steps)
        for _ in range(int(RETREAT_SETTLE_S * hz)):
            sim.step_ctrl()
            steps[0] += 1
        meter, damage = self._meter_end()
        reading = self.squeeze_reading((self.frag or {}).get("k_n_per_mm"), self.rng)
        disp, tilt, thrown = self._probe_cost(p_start, R_start)
        return ProbeResult("squeeze", "damaged" if damage else "measured", reading, disp, tilt, thrown, steps[0] / hz, trace,
                           **self._monitor(), meter=meter, damage=damage)

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

    def _label(self, trace, rose, dest, release, planned, v_lift: float = 0.0, meter: dict[str, Any] | None = None,
               damage: dict[str, Any] | None = None) -> Rollout:
        sim = self.sim
        touching, pads = sim.contacts()
        low = sim.obj_low_z()
        final = Final(zone=sim.zone_of(touching), pos=sim.obj_pos().tolist(), low_z=float(low),
                      tipped=bool(release is not None and self._tipped(release, low)), held=bool(any(pads)),
                      com_xy=sim.d.subtree_com[sim.obj][:2].tolist())
        out = label_rollout(trace, dest, release, final)
        readouts, sensor = self._force_readouts(summarize(trace, out, v_lift))
        return Rollout(final_label(out["label"], damage), out["max_z"], out["drop_z"], out["loss_phase"], out["loss_step"],
                       out["loss_t"], out["loss_speed"], final.zone, release, planned, trace, readouts,
                       **self._monitor(), meter=meter or {}, motion_label=out["label"], damage=damage, sensor=sensor)

    def _force_readouts(self, readouts: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """The attempt's force readouts with the declared error: three eps from the force stream, in the order hold,
        slip, loss, drawn also for a reading that does not exist. Returns the readouts and the sensor record."""
        eps, clean, out = {}, {}, dict(readouts)
        for term, key in (("hold", "fz_hold_n"), ("slip", "fz_slip_max_n"), ("loss", "fz_at_loss_n")):
            eps[term] = float(self.force_rng.standard_normal())
            clean[key] = readouts.get(key)
            out[key] = noisy_force(clean[key], eps[term], self.force_noise)
        return out, {"force_noise": self.force_noise, "eps": eps, "clean": clean}
