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

"""Corpus generator: pick, carry and set down every object under drawn hidden properties.

    python -m robots.mujoco_lab.corpus --out PATH --draws 48 --workers 8 [--objects a,b] [--resume]

One row per rollout (JSON lines): the hidden draw, the action, the measured
geometry, the features (true-θ, see features.extract), the planned trajectory
and the outcome label with the loss record. One worker process per object.

Corpus v9 (plan 2b, ``--mode v9``, the default) draws v8's theta and action from v8's stream and then, from a
damage stream of its own (``damage_rng``), the fill, the hidden fragility, the squeeze (log-uniform on
[0.4, 2.5] f_std) and the carry acceleration (``draw_damage``, applied by ``apply_damage``). A row carries the
public fragility flags (``entry``), the draws (``theta_frag``, ``fill``, ``spill_angle_deg``, ``a_transport``),
the damage meter (``meter``), the motion label of the rules (``motion_label``), the meter's first threshold
crossing (``damage``) and the label, which the damage replaces. ``--mode control`` reruns v8's draws and actions
(a_transport 2.0, no liquid) with the fragility drawn, to check that nothing else changed the motion;
``--mode m2`` is milestone 2a's world (v8's draws and actions, no liquid, no fragility: v8's rows and labels).
A drawn pinch the skill never offers (menu.unusable_reason: wider than the
gripper opening, or off the object) does not run: its row says why
("declined") and carries no outcome. Each rollout row carries the contact
monitor (``rev_steps``, ``deepest_mm``) and its flag: ``artifact`` is True when
the pad contacts were not physical or the reset did not reach rest
(adapter.artifact). ``trainable`` says which rows the models may learn from.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import zlib
from numbers import Real
from typing import Any

import numpy as np

from robots.mujoco_lab.fragility import CLASSES, FLAGS, THRESHOLDS_V0, Thresholds, draw_fragility, entry_of, load_classes, spill_angle
from rpent.skills.pick_place.destinations import DESTINATIONS, MESH_IDS, PRIMITIVE_IDS, release_height

MASS_SCALE_RANGE = (0.5, 3.0)
MU_RANGE = (0.1, 0.9)
COM_S_RANGE = (0.3, 0.8)
V_LIFT_CHOICES = (0.05, 0.15, 0.4, 0.8)
V_TRANSPORT_CHOICES = (0.2, 0.5, 1.0)
S_CHOICES = (0.3, 0.5, 0.7)
P_COM_SHIFT = 0.5
P_DROP_RELEASE = 0.3
RELEASE_KEYS = ("tilt_deg", "displacement_xy_m", "drop_height_m", "final_zone", "final_speed")
DAMAGE_STREAM = 9
P_EMPTY = 0.5                          # a fillable object is empty with p 0.5
FILL_RANGE = (0.3, 0.9)
FILLED_MASS_SCALE_RANGE = (0.8, 1.25)  # a filled container's own mass scale: v8's [0.5, 3.0] would hide the liquid's mass
FORCE_FACTOR_RANGE = (0.4, 2.5)        # v9's squeeze, times f_std
A_TRANSPORT_CHOICES = (0.5, 1.0, 2.0)  # m/s^2
MODES = ("v9", "control", "m2")
A_TRANSPORT_V8 = 2.0                   # adapter.A_TRANSPORT, the carry acceleration of every v8 row
_WORLDS: dict[tuple[str, str], tuple[dict[str, Any], Thresholds]] = {}


def trainable(row: dict[str, Any]) -> bool:
    """A row the models may learn from: it ran (no error), the skill would have offered its pinch, and its
    contacts were physical (no artifact flag)."""
    return "error" not in row and "declined" not in row and not row.get("artifact")


def case_rng(seed: int, object_id: str, idx: int) -> np.random.Generator:
    """One stream per (seed, object, draw index), so objects do not share hidden draws."""
    return np.random.default_rng([int(seed), zlib.crc32(object_id.encode()), int(idx)])


def draw_theta(rng: np.random.Generator, nominal_mass: float) -> tuple[dict[str, Any], int]:
    """Hidden properties for one rollout and the end the CoM mass sits at (±1)."""
    lo, hi = MASS_SCALE_RANGE
    theta = {
        "mass_scale": float(np.exp(rng.uniform(np.log(lo), np.log(hi)))),
        "mu": float(rng.uniform(*MU_RANGE)),
        "com_s": None if rng.random() < P_COM_SHIFT else float(rng.uniform(*COM_S_RANGE)),
        "v_lift": float(rng.choice(V_LIFT_CHOICES)),
    }
    com_end = 1 if rng.random() < 0.5 else -1
    return theta, com_end


def spread_factor(f: Any) -> float:
    """The spread factor of plan H3 decision 4: a finite number in (0, 1] (1: the corpus ranges)."""
    if isinstance(f, Real) and not isinstance(f, bool) and math.isfinite(float(f)) and 0.0 < float(f) <= 1.0:
        return float(f)
    raise ValueError(f"the spread factor must be a number in (0, 1], not {f!r}")


def shrunk_range(bounds: tuple[float, float], f: float, log: bool = False) -> tuple[float, float]:
    """A range shrunk by the factor f around its midpoint (on the log scale when ``log``), the range shrink_theta maps
    its draws onto; f = 1 returns ``bounds`` itself, so the draws stay bit for bit."""
    f = spread_factor(f)
    if f == 1.0:
        return bounds
    lo, hi = (math.log(b) for b in bounds) if log else bounds
    mid, half = (lo + hi) / 2.0, f * (hi - lo) / 2.0
    return (math.exp(mid - half), math.exp(mid + half)) if log else (mid - half, mid + half)


def _shrink(x: float, bounds: tuple[float, float], f: float, log: bool = False) -> float:
    lo, hi = (math.log(b) for b in bounds) if log else bounds
    mid = (lo + hi) / 2.0
    v = mid + f * ((math.log(x) if log else x) - mid)
    return math.exp(v) if log else v


def shrink_theta(theta: dict[str, Any], f: float, filled: bool) -> dict[str, Any]:
    """A case's hidden properties under the spread factor f (plan H3 decision 4, quantile-coupled): the mass scale on
    the log scale around the log-midpoint of its range (a filled container's own range when ``filled``), mu and a
    non-None com_s around their midpoints; every other key unchanged. A (log-)uniform draw keeps its quantile, so the
    shrunk value is a draw of the narrow range (shrunk_range) and the case stream stays. f = 1 returns an equal dict."""
    f = spread_factor(f)
    out = dict(theta)
    if f == 1.0:
        return out
    out["mass_scale"] = _shrink(float(theta["mass_scale"]), FILLED_MASS_SCALE_RANGE if filled else MASS_SCALE_RANGE, f, log=True)
    out["mu"] = _shrink(float(theta["mu"]), MU_RANGE, f)
    if theta.get("com_s") is not None:
        out["com_s"] = _shrink(float(theta["com_s"]), COM_S_RANGE, f)
    return out


def draw_action(rng: np.random.Generator, f_std: float, use_fixtures: bool = False) -> dict[str, Any]:
    if use_fixtures and MESH_IDS and rng.random() < 0.5:
        dest_id = str(rng.choice(sorted(MESH_IDS)))
    else:
        dest_id = str(rng.choice(sorted(PRIMITIVE_IDS)))
    dest = DESTINATIONS[dest_id]
    rel = None
    if dest["kind"] in ("rim", "cavity", "open_bin") and rng.random() < P_DROP_RELEASE:
        rel = 0.05 + float(dest["rim_height"])            # the careless "drop it in" release
    return {
        "destination": dest_id,
        "s": float(rng.choice(S_CHOICES)),
        "force": float(f_std if rng.random() < 0.5 else 2 * f_std),
        "v_transport": float(rng.choice(V_TRANSPORT_CHOICES)),
        "release_height": rel,
    }


def damage_rng(seed: int, object_id: str, idx: int) -> np.random.Generator:
    """The damage stream of a draw, apart from case_rng's, so v8's draws stay as they were."""
    return np.random.default_rng([int(seed), zlib.crc32(object_id.encode()), int(idx), DAMAGE_STREAM])


def draw_damage(rng: np.random.Generator, entry: dict[str, Any], th: Thresholds, f_std: float) -> dict[str, Any]:
    """The draw's fill (0 unless the object is fillable and not empty: then U[0.3, 0.9]), its hidden fragility (three
    standard normals for every class, fragility.draw_fragility), the squeeze, the carry acceleration and, for a filled
    draw, the container's mass scale. Every draw takes the same values from the stream."""
    u_empty, fill = float(rng.random()), float(rng.uniform(*FILL_RANGE))
    frag = draw_fragility(rng, entry, th)
    lo, hi = FORCE_FACTOR_RANGE
    force = float(f_std) * float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    a_transport = float(rng.choice(A_TRANSPORT_CHOICES))
    lo, hi = FILLED_MASS_SCALE_RANGE
    scale = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    filled = bool(entry.get("fillable")) and u_empty >= P_EMPTY
    out = {"fill": fill if filled else 0.0, "theta_frag": frag, "spill_angle_deg": spill_angle(entry["cavity"], fill) if filled else None,
           "force": force, "a_transport": a_transport}
    if filled:
        out["mass_scale"] = scale
    return out


def apply_damage(theta: dict[str, Any], action: dict[str, Any], d: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """v8's draws under a damage draw: a filled container takes its own mass scale and no CoM-shift branch; the action
    takes the squeeze and the carry acceleration."""
    th, act = dict(theta), dict(action)
    if float(d.get("fill") or 0.0) > 0.0:
        th["mass_scale"], th["com_s"] = float(d["mass_scale"]), None
    act["force"], act["a_transport"] = float(d["force"]), float(d["a_transport"])
    return th, act


def world(classes: str | dict[str, Any] | None = None, thresholds: str | Thresholds | None = None) -> tuple[dict[str, Any], Thresholds]:
    """The fragility entries and the thresholds (paths or loaded; default: the class file and thresholds v0), cached."""
    if isinstance(classes, dict) and isinstance(thresholds, Thresholds):
        return classes, thresholds
    cp = classes if isinstance(classes, str) else CLASSES
    tp = thresholds if isinstance(thresholds, str) else THRESHOLDS_V0
    if (cp, tp) not in _WORLDS:
        _WORLDS[(cp, tp)] = (load_classes(cp), Thresholds.from_json(tp))
    c, t = _WORLDS[(cp, tp)]
    return (classes if isinstance(classes, dict) else c), (thresholds if isinstance(thresholds, Thresholds) else t)


def _release_summary(rel: dict[str, Any] | None) -> dict[str, Any] | None:
    if rel is None:
        return None
    out = {k: rel.get(k) for k in RELEASE_KEYS}
    fc = rel.get("first_contact")
    out["impact_speed"] = None if fc is None else fc.get("impact_speed")
    out["first_zone"] = None if fc is None else fc.get("zone")
    return out


def rollout_row(env, object_id: str, idx: int, seed: int, use_fixtures: bool = False, mode: str = "v9",
                classes: str | dict[str, Any] | None = None, thresholds: str | Thresholds | None = None) -> dict[str, Any]:
    """One rollout of `object_id` under a fresh hidden draw; returns the corpus row (``mode``: v9, control or m2)."""
    from robots.mujoco_lab.adapter import object_props
    from rpent.skills.pick_place.features import extract
    from rpent.skills.pick_place.menu import pinch_limit, unusable_reason

    if mode not in MODES:
        raise ValueError(f"unknown corpus mode {mode!r}; known: {MODES}")
    rng = case_rng(seed, object_id, idx)
    props = object_props(object_id)
    theta, com_end = draw_theta(rng, props["mass"])
    f_std = env.s["t3_make"].std_force(props["mass"])
    act = draw_action(rng, f_std, use_fixtures)
    C, th = world(classes, thresholds)
    entry, _ = entry_of(object_id, C)
    d = draw_damage(damage_rng(seed, object_id, idx), entry, th, f_std)
    if mode == "control":                  # v8's draws and action, the fragility drawn, no liquid
        d = {"fill": 0.0, "theta_frag": d["theta_frag"], "spill_angle_deg": None, "force": act["force"], "a_transport": A_TRANSPORT_V8}
    elif mode == "m2":                     # milestone 2a's world: v8's draws and action, no liquid, no fragility
        d = {"fill": 0.0, "theta_frag": None, "spill_angle_deg": None, "force": act["force"], "a_transport": A_TRANSPORT_V8}
    theta, act = apply_damage(theta, act, d)
    fixture = act["destination"] if act["destination"] in MESH_IDS else None
    env.slosh_zeta = th.slosh_zeta
    env.slosh_scale = th.slosh_scale
    env.reset(object_id, theta, seed=seed * 1000 + idx, com_end=com_end, fixture=fixture, fill=d["fill"], frag=d["theta_frag"])
    reset_info = dict(env.reset_info)
    props_list = env.proposals()
    prop = next(p for p in props_list if abs(p.s - act["s"]) < 1e-9)
    truth = env.truth()
    loc = env.locate("object")
    g = env.geometry()
    pose = env.proposal_pose(prop)                      # before the move: the pose the action was planned in
    feats = extract(g, np.asarray(prop.point), prop.yaw, truth["mass"], truth["mu"], act["force"], theta["v_lift"],
                    act["v_transport"], act["destination"], act["release_height"])
    dest = DESTINATIONS[act["destination"]]
    row = {
        "key": f"{object_id}__d{idx}",
        "object": object_id,
        "theta": theta,
        "com_end": com_end,
        "truth": {k: truth[k] for k in ("mass", "mu", "com_s")},
        "seed": seed,
        "destination": act["destination"],
        "fixture": fixture,
        "s": act["s"],
        "force": act["force"],
        "v_lift": theta["v_lift"],
        "v_transport": act["v_transport"],
        "release_height": act["release_height"] if act["release_height"] is not None else release_height(dest),
        "width": prop.width,
        "geometry": {"base_extent": g.base_extent, "com_height": g.com_height, "footprint_half_width": g.footprint_half_width,
                     "length": g.length, "volume": loc["volume"], "material": loc["material"], "hollow": loc["hollow"]},
        "features": feats,
        "pose": pose,
        "com_obj_true": truth["com_obj"],
        "reset": reset_info,
        "a_transport": act["a_transport"],
        "entry": {k: entry.get(k) for k in FLAGS},
        "theta_frag": d["theta_frag"],
        "fill": d["fill"],
        "spill_angle_deg": d["spill_angle_deg"],
    }
    declined = unusable_reason(prop.width, pinch_limit(env))
    if declined is not None:                            # the menu never offers this pinch: no rollout
        return {**row, "declined": declined}
    r = env.execute(prop, act["force"], theta["v_lift"], act["v_transport"], act["destination"], act["release_height"],
                    a_transport=act["a_transport"])
    return {
        **row,
        "label": r.label,
        "motion_label": r.motion_label,
        "damage": r.damage,
        "meter": r.meter,
        "max_z": r.max_z,
        "drop_z": r.drop_z,
        "loss_phase": r.loss_phase,
        "loss_step": r.loss_step,
        "loss_t": r.loss_t,
        "loss_speed": r.loss_speed,
        "zone": r.zone,
        "release": _release_summary(r.release),
        "planned": [s.to_dict() for s in r.planned],
        "readouts": r.readouts,
        "sensor": r.sensor,
        "rev_steps": r.rev_steps,
        "deepest_mm": r.deepest_mm,
        "artifact": r.artifact,
    }


def _worker(args: tuple[str, int, int, list[int], bool, str, str, str]) -> list[dict[str, Any]]:
    object_id, draws, seed, skip, use_fixtures, mode, classes, thresholds = args
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    env = MuJoCoLabEnv()
    rows = []
    for idx in range(draws):
        if idx in skip:
            continue
        try:
            rows.append(rollout_row(env, object_id, idx, seed, use_fixtures, mode, classes, thresholds))
        except Exception as exc:  # noqa: BLE001  one bad rollout must not kill the object's batch
            rows.append({"key": f"{object_id}__d{idx}", "object": object_id, "error": str(exc)[:200]})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=48)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--objects", default=None, help="comma-separated object ids (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fixtures", action="store_true", help="draw half the destinations from the inspected mesh fixtures")
    ap.add_argument("--mode", default="v9", choices=MODES,
                    help="v9: the damage draws; control: v8's draws and actions with the fragility; m2: v8's world (no damage)")
    ap.add_argument("--classes", default=CLASSES, help="the fragility classes (fragility_classes.json)")
    ap.add_argument("--thresholds", default=THRESHOLDS_V0, help="the damage thresholds (thresholds_v0.json)")
    a = ap.parse_args(argv)
    from robots.mujoco_lab.adapter import object_ids

    objects = a.objects.split(",") if a.objects else object_ids()
    done: dict[str, set[int]] = {}
    if a.resume and os.path.exists(a.out):
        for line in open(a.out):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in row:
                continue
            done.setdefault(row["object"], set()).add(int(row["key"].rsplit("d", 1)[1]))
    jobs = [(oid, a.draws, a.seed, sorted(done.get(oid, set())), a.fixtures, a.mode, a.classes, a.thresholds) for oid in objects]
    written = 0
    mode = "a" if a.resume and os.path.exists(a.out) else "w"
    with open(a.out, mode) as f:
        if a.workers <= 1:
            results = map(_worker, jobs)
        else:
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(a.workers)
            results = pool.imap_unordered(_worker, jobs)
        for rows in results:
            for row in rows:
                f.write(json.dumps(row) + "\n")
                written += "error" not in row
            f.flush()
            print(f"{rows[0]['object'] if rows else '-'}: {len(rows)} rows", file=sys.stderr, flush=True)
        if a.workers > 1:
            pool.close()
            pool.join()
    return written


if __name__ == "__main__":
    sys.exit(0 if main() >= 0 else 1)
