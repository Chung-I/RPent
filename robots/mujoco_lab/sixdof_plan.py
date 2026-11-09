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

"""The lab side of the tilted rows' free move (daily-logs design 2026-11-08-sixdof-rerun-design.md, unit (e), G3).

cuRobo plans the free move from the arm's start joints to the chosen configuration at the pregrasp (curobo_plan, in its
own venv). The lab gives it the world (world_export: every colliding geom outside the robot, in the arm's base frame)
and the robot (lab_robot: the arm's spheres refit to the sim's collision hulls and the gripper's at its real open pose,
committed as data/curobo/lab_robot.json), and reads the cached plan back as execute6's joint waypoints, one per control
step (free_from_cache). A run never plans: a key not in the cache raises PlanCacheMiss (the row stops and is counted),
a cached failure raises PlanFailed (the funnel's free-move step). free_hits reads the waypoints against the scene with
the sim's exact geometry (a record, not a filter).

The sphere cover: spheres are added to a link's start spheres until every surface sample of its collision hulls lies
within the gap of the union; each added sphere protrudes at most a cap past the link's hulls. The protrusion uses each
face plane's distance, a lower bound of the distance to the hull, so a sphere can stand a little further out at an edge.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from typing import Any

import numpy as np

from robots.mujoco_lab import curobo_plan as cp
from robots.mujoco_lab import sixdof

HERE = os.path.dirname(os.path.abspath(__file__))
LAB_ROBOT = os.path.join(HERE, "data", "curobo", "lab_robot.json")
ROBOT_YML = os.path.join(HERE, "data", "curobo", "panda_lab_gripper.yml")
FRANKA_YML = os.path.expanduser("~/Codes/curobo/curobo/content/configs/robot/franka.yml")
CUROBO_PY = os.path.expanduser("~/Codes/curobo/.venv/bin/python")
HAND_OFF = sixdof.HAND_OFF
START_TOL_RAD = 1e-6            # a cached plan starts at the arm's start joints within this (cuRobo plans in float32)
FIT_GAP_M, CHECK_GAP_M, MAX_PROTRUSION_M = 0.0008, 0.001, 0.005
FIT_SPACING_M = 0.0005
RADII = (0.004, 0.006, 0.008, 0.012, 0.016, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06)
DEPTHS = (0.0, 0.002, 0.004)    # an added sphere stands this far past the face at the point it covers
CYL_SIDES = 64
HAND_PARTS = {"lab_hand": ("palm", "wrist"), "lab_finger_l": ("finger_l",), "lab_finger_r": ("finger_r",)}


class PlanCacheMiss(Exception):
    """The free move's key is not in the plan cache (G3: the row stops and is counted)."""


class PlanFailed(Exception):
    """cuRobo found no free move for this key (a cached failure)."""


def _armlib(sim: Any) -> Any:
    return sixdof._armlib(sim.arm)


def geom_name(m: Any, g: int) -> str:
    return m.geom(g).name or f"geom{g}"


def robot_subtree_geoms(sim: Any) -> set[int]:
    """Every geom of the arm's body tree (link 0 down to the fingers), colliding or not."""
    m = sim.m
    root = m.body("link0").id
    out = set()
    for g in range(m.ngeom):
        k = int(m.geom_bodyid[g])
        while k > 0 and k != root:
            k = int(m.body_parentid[k])
        if k == root:
            out.add(g)
    return out


def _quat(R: Any) -> list[float]:
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).ravel())
    return [float(x) for x in q]


def _outward(V: np.ndarray, H: Any) -> list[int]:
    """The hull's triangles, wound so that their normals point out."""
    F = []
    for tri, e in zip(H.simplices, H.equations, strict=True):
        a, b, c = (int(x) for x in tri)
        if float(np.dot(np.cross(V[b] - V[a], V[c] - V[a]), e[:3])) < 0.0:
            b, c = c, b
        F += [a, b, c]
    return F


def world_export(sim: Any) -> dict[str, Any]:
    """The scene for cuRobo in the arm's base frame (link 0): every geom outside the robot that collides; boxes as
    cuboids ({"dims", "pose": [x, y, z, qw, qx, qy, qz]}), a plane as a slab 0.1 m thick under it, a cylinder as its
    bounding cuboid, a mesh (the object's hulls, mesh fixtures) as its convex hull ({"vertices", "faces" wound outward,
    "pose": identity}), as MuJoCo collides it."""
    import mujoco
    from scipy.spatial import ConvexHull

    m, d = sim.m, sim.d
    b0 = m.body("link0").id
    R0, p0 = d.xmat[b0].reshape(3, 3).copy(), d.xpos[b0].copy()
    robot = robot_subtree_geoms(sim)
    out: dict[str, Any] = {"cuboid": {}, "mesh": {}}
    for g in range(m.ngeom):
        if g in robot or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        name, t = geom_name(m, g), int(m.geom_type[g])
        pw, Rw = d.geom_xpos[g].copy(), d.geom_xmat[g].reshape(3, 3).copy()
        pose = [*(float(x) for x in R0.T @ (pw - p0)), *_quat(R0.T @ Rw)]
        s = m.geom_size[g]
        if t == mujoco.mjtGeom.mjGEOM_BOX:
            out["cuboid"][name] = {"dims": [2.0 * float(x) for x in s], "pose": pose}
        elif t == mujoco.mjtGeom.mjGEOM_PLANE:
            c = pw - 0.05 * Rw[:, 2]
            out["cuboid"][name] = {"dims": [6.0, 6.0, 0.1], "pose": [*(float(x) for x in R0.T @ (c - p0)), *_quat(R0.T @ Rw)]}
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            out["cuboid"][name] = {"dims": [2.0 * float(s[0]), 2.0 * float(s[0]), 2.0 * float(s[1])], "pose": pose}
        elif t == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            Vm = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
            Vb = ((Vm @ Rw.T + pw) - p0) @ R0
            H = ConvexHull(Vb)
            keep = sorted(int(i) for i in H.vertices)
            V = Vb[keep]
            H = ConvexHull(V)
            out["mesh"][name] = {"vertices": V.tolist(), "faces": _outward(V, H), "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]}
        else:
            raise ValueError(f"geom {name}: type {t} has no cuRobo shape here")
    return out


# ------------------------------------------------------------------ the sphere refit

def _geom_hull(m: Any, d: Any, g: int, R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """The vertices of geom g's collision shape in the frame (R, p): a box's corners, a cylinder's two CYL_SIDES-gons,
    a mesh's vertices."""
    import mujoco

    t, s = int(m.geom_type[g]), m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        L = np.array([[a * s[0], b * s[1], c * s[2]] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        u = 2.0 * math.pi * np.arange(CYL_SIDES) / CYL_SIDES
        ring = np.stack([s[0] * np.cos(u), s[0] * np.sin(u)], axis=1)
        L = np.concatenate([np.column_stack([ring, np.full(CYL_SIDES, z)]) for z in (-s[1], s[1])])
    elif t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        L = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
    else:
        raise ValueError(f"geom {geom_name(m, g)}: type {t}")
    W = L @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
    return (W - p) @ R


def link_hulls(sim: Any, link: str) -> list[np.ndarray]:
    """A cuRobo link's collision shapes in its frame: panda_link1..7 the sim's link meshes; lab_hand the palm and the
    wrist, lab_finger_l and lab_finger_r a finger's box and pad, in the hand's frame, the fingers at their rest under
    the open command (sixdof.OPEN_Q)."""
    import mujoco

    m = sim.m
    d = mujoco.MjData(m)
    d.qpos[:] = sim.d.qpos
    if link.startswith("panda_link"):
        body = m.body("link" + link[len("panda_link"):]).id
        bodies = {body}
    else:
        for j in ("jl", "jr"):
            d.qpos[m.jnt_qposadr[m.joint(j).id]] = sixdof.OPEN_Q
        body = m.body("hand").id
        bodies = {m.body(n).id for n in HAND_PARTS[link] if n.startswith("finger")}
    mujoco.mj_kinematics(m, d)
    R, p = d.xmat[body].reshape(3, 3).copy(), d.xpos[body].copy()
    gs = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies and (m.geom_contype[g] or m.geom_conaffinity[g])]
    if link == "lab_hand":
        gs = [m.geom(n).id for n in HAND_PARTS[link]]
    return [_geom_hull(m, d, g, R, p) for g in gs]


def surface_points(hulls: list[np.ndarray], spacing: float, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Points on the hulls' surfaces: on each triangle a lattice at ``spacing`` (seed None) or about one random point
    per spacing^2 of area (a seed), one point per spacing along each triangle's edges, and each hull's vertices; with
    outward unit normals (an edge or a vertex: the mean of its faces' normals)."""
    from scipy.spatial import ConvexHull

    rng = np.random.default_rng(seed)
    P, N = [], []
    for V in hulls:
        H = ConvexHull(V)
        edges: dict[tuple[int, int], list[np.ndarray]] = {}
        for tri, e in zip(H.simplices, H.equations, strict=True):
            A, B, C = V[tri[0]], V[tri[1]], V[tri[2]]
            if seed is None:
                k = int(math.ceil(max(float(np.linalg.norm(B - A)), float(np.linalg.norm(C - A)), float(np.linalg.norm(C - B))) / spacing))
                i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
                f = (i + j) <= k
                u, v = i[f] / k, j[f] / k
            else:
                n = int(math.ceil(0.5 * float(np.linalg.norm(np.cross(B - A, C - A))) / spacing ** 2))
                u, v = rng.random(n), rng.random(n)
                f = u + v > 1.0
                u[f], v[f] = 1.0 - u[f], 1.0 - v[f]
            n = len(u)
            P.append(A + u[:, None] * (B - A) + v[:, None] * (C - A))
            N.append(np.repeat(e[None, :3], n, axis=0))
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                edges.setdefault((int(min(a, b)), int(max(a, b))), []).append(e[:3])
        for (a, b), ns in edges.items():
            k = int(math.ceil(float(np.linalg.norm(V[b] - V[a])) / spacing))
            t = (np.arange(k) + 0.5) / k
            nv = np.mean(ns, axis=0)
            P.append(V[a] + t[:, None] * (V[b] - V[a]))
            N.append(np.repeat((nv / np.linalg.norm(nv))[None], k, axis=0))
        for i in H.vertices:
            nv = H.equations[[k for k, tri in enumerate(H.simplices) if i in tri], :3].mean(axis=0)
            P.append(V[i][None])
            N.append((nv / np.linalg.norm(nv))[None])
    return np.concatenate(P), np.concatenate(N)


def gaps(P: np.ndarray, C: Any, R: Any) -> np.ndarray:
    """Each point's distance past the union of the spheres (C, R) (<= 0 inside; inf without spheres)."""
    C, R = np.asarray(C, dtype=float).reshape(-1, 3), np.asarray(R, dtype=float).ravel()
    if not len(R):
        return np.full(len(P), np.inf)
    out = np.empty(len(P))
    for a in range(0, len(P), 20000):
        X = P[a:a + 20000]
        out[a:a + 20000] = (np.linalg.norm(X[:, None, :] - C[None], axis=2) - R[None]).min(axis=1)
    return out


def outside(X: np.ndarray, hulls: list[np.ndarray]) -> np.ndarray:
    """A lower bound of each point's distance outside the union of the hulls (the largest face-plane distance per
    hull, the smallest over the hulls; < 0 inside)."""
    from scipy.spatial import ConvexHull

    X = np.asarray(X, dtype=float).reshape(-1, 3)
    return np.min([(X @ E[:, :3].T + E[:, 3]).max(axis=1) for E in (ConvexHull(V).equations for V in hulls)], axis=0)


def sphere_points(C: Any, R: Any, n: int = 200) -> np.ndarray:
    """n points on each sphere's surface (a Fibonacci lattice)."""
    k = np.arange(n) + 0.5
    z = 1.0 - 2.0 * k / n
    t = math.pi * (1.0 + math.sqrt(5.0)) * k
    u = np.column_stack([np.sqrt(1.0 - z * z) * np.cos(t), np.sqrt(1.0 - z * z) * np.sin(t), z])
    C, R = np.asarray(C, dtype=float).reshape(-1, 3), np.asarray(R, dtype=float).ravel()
    return (C[:, None, :] + R[:, None, None] * u[None]).reshape(-1, 3)


def cover(hulls: list[np.ndarray], C0: Any, R0: Any, gap: float = FIT_GAP_M, protrusion: float = MAX_PROTRUSION_M,
          region: Any = None, spacing: float = FIT_SPACING_M) -> dict[str, Any]:
    """Spheres added to (C0, R0) until every surface sample is within ``gap`` of the union: each step takes the worst
    sample p (outward normal n) and, among the spheres c = p - (r - depth) n over RADII and DEPTHS whose surface points
    in ``region`` (all if None) stand at most ``protrusion`` outside the hulls, adds the one covering the most
    uncovered samples. Returns {"C", "R", "added" (mask), "gap_m", "protrusion_m" (largest over the added spheres)}."""
    P, N = surface_points(hulls, spacing)
    C = [np.asarray(c, dtype=float) for c in np.asarray(C0, dtype=float).reshape(-1, 3)]
    R = [float(r) for r in np.asarray(R0, dtype=float).ravel()]
    added, prot_max = [False] * len(R), 0.0
    g = gaps(P, C, R)
    while (g > gap).any():
        if len(R) > 2000:
            raise RuntimeError("the cover did not converge")
        i = int(np.argmax(g))
        bad = P[g > gap]
        best = None
        for r in RADII:
            for dp in DEPTHS:
                c = P[i] - (r - dp) * N[i]
                X = sphere_points(c, r)
                if region is not None:
                    X = X[region(X)]
                pr = float(outside(X, hulls).max()) if len(X) else -math.inf
                if pr > protrusion:
                    continue
                key = (int((np.linalg.norm(bad - c, axis=1) - r <= gap).sum()), -pr)
                if best is None or key > best[0]:
                    best = (key, c, r, pr)
        if best is None:
            raise RuntimeError(f"no sphere covers {P[i]} within the protrusion cap")
        _, c, r, pr = best
        C.append(c)
        R.append(r)
        added.append(True)
        prot_max = max(prot_max, pr)
        g = np.minimum(g, np.linalg.norm(P - c, axis=1) - r)
    return {"C": np.array(C).reshape(-1, 3), "R": np.array(R), "added": np.array(added, dtype=bool), "gap_m": float(g.max()),
            "protrusion_m": prot_max}


def _spheres(C: Any, R: Any) -> list[dict[str, Any]]:
    return [{"center": [round(float(x), 6) for x in c], "radius": round(float(r), 6)} for c, r in zip(C, R, strict=True)]


def lab_robot(sim: Any, franka_yml: str = FRANKA_YML) -> dict[str, Any]:
    """The lab's part of the cuRobo robot (curobo_plan.robot_yml's ``lab``): the mount, the TCP, the lab's free-move
    speed, panda_link1..7's spheres (franka.yml's, plus the cover of the sim's link hulls) and the gripper's (the cover
    of each part at the open pose, nothing counted above the wrist's top, which is link 7's space), with the fit."""
    import yaml

    A = _armlib(sim)
    franka = yaml.safe_load(open(franka_yml))["robot_cfg"]["kinematics"]["collision_spheres"]
    out: dict[str, Any] = {"mount_pos": [float(x) for x in A.MOUNT_POS], "mount_quat": [float(x) for x in A.mount_quat()],
                           "pad_centre_z": -HAND_OFF, "free_v": float(A.FREE_V), "open_q": sixdof.OPEN_Q,
                           "franka_yml_sha256": hashlib.sha256(open(franka_yml, "rb").read()).hexdigest(),
                           "arm_spheres": {}, "gripper_spheres": {}, "fit": {}}
    jobs = [(f"panda_link{i}", "arm_spheres", franka[f"panda_link{i}"], None) for i in range(1, 8)]
    jobs += [(n, "gripper_spheres", [], lambda X: X[:, 2] <= A.WRIST_TOP) for n in HAND_PARTS]
    for link, part, start, region in jobs:
        H = link_hulls(sim, link)
        f = cover(H, [s["center"] for s in start], [s["radius"] for s in start], region=region)
        out[part][link] = _spheres(f["C"], f["R"])
        Pc, _ = surface_points(H, spacing=CHECK_GAP_M, seed=1)
        out["fit"][link] = {"start": len(start), "added": int(f["added"].sum()), "fit_gap_mm": round(1e3 * f["gap_m"], 3),
                            "check_gap_mm": round(1e3 * float(gaps(Pc, f["C"], f["R"]).max()), 3),
                            "protrusion_mm": round(1e3 * f["protrusion_m"], 3)}
    return out


# ------------------------------------------------------------------ the cached plan

def free_from_cache(cache: str, world: Any, start: Any, goal: Any, hz: float) -> list[np.ndarray]:
    """execute6's ``free``: the cached cuRobo trajectory from ``start`` to ``goal`` in ``world``, one joint waypoint per
    control step, without its first point (the start). PlanCacheMiss when the key is not cached, PlanFailed for a
    cached failure; ValueError for a plan at another step or from another start."""
    identity = json.load(open(os.path.join(cache, "identity.json")))
    key = cp.plan_key(world, start, goal, identity)
    e = cp.read_entry(cache, key)
    if e is None:
        raise PlanCacheMiss(key)
    if not e["success"]:
        raise PlanFailed(f"{key}: {e['status']}")
    if abs(float(e["dt"]) - 1.0 / hz) > 1e-12:
        raise ValueError(f"the cached plan's dt {e['dt']} is not the control step {1.0 / hz}")
    q = np.asarray(e["q"], dtype=float)
    if float(np.max(np.abs(q[0] - np.asarray(start, dtype=float)))) > START_TOL_RAD:
        raise ValueError("the cached plan does not begin at the start joints")
    return [x.copy() for x in q[1:]]


def free_hits(sim: Any, free: list[Any]) -> list[dict[str, Any]]:
    """The robot's contacts with the scene (the sim's exact geometry, the rest of the scene as it is now) at each
    waypoint: [{"step", "robot", "other", "robot_id", "other_id", "dist_m"}]."""
    m = sim.m
    out = []
    for k, q in enumerate(free):
        for r, o, dist in sim.arm.collisions(np.asarray(q, dtype=float)):
            out.append({"step": k, "robot": geom_name(m, r), "other": geom_name(m, o), "robot_id": int(r), "other_id": int(o),
                        "dist_m": float(dist)})
    return out


def request(case: Any, world: dict[str, Any], start: Any, goals: list[dict[str, Any]]) -> dict[str, Any]:
    """One case's planning request: its world, the arm's start joints and the goals ([{"row", "q"}])."""
    return cp._plain({"case": list(case), "world": world, "start": start, "goals": goals})


def plan_case(req: dict[str, Any], cache: str, python: str = CUROBO_PY, robot_yml: str = ROBOT_YML) -> dict[str, Any]:
    """Plan one request's uncached goals in one cuRobo process (curobo_plan.py in cuRobo's venv): the request goes to
    cache/requests/, the process log to cache/logs/; returns the process's summary {"planned", "cached"}. A request
    file of the same case with other content is refused."""
    name = "{}_{}".format(*req["case"])
    rq = os.path.join(cache, "requests", name + ".json")
    os.makedirs(os.path.dirname(rq), exist_ok=True)
    os.makedirs(os.path.join(cache, "logs"), exist_ok=True)
    if os.path.exists(rq):
        if cp.canonical(json.load(open(rq))) != cp.canonical(req):
            raise ValueError(f"{rq} holds another request")
    else:
        json.dump(req, open(rq, "w"))
    log = os.path.join(cache, "logs", name + ".log")
    with open(log, "a") as f:
        r = subprocess.run([python, "-P", cp.__file__, "plan", "--request", rq, "--cache", cache, "--robot", robot_yml],
                           stdout=subprocess.PIPE, stderr=f, text=True, stdin=subprocess.DEVNULL)
        f.write(r.stdout)
    if r.returncode != 0:
        raise RuntimeError(f"cuRobo exited {r.returncode} on {name} (log {log})")
    return json.loads(r.stdout.strip().splitlines()[-1])


def main(argv: list[str] | None = None) -> None:
    """fit --out PATH: the lab's part of the cuRobo robot (lab_robot) on the sim of RPENT_MUJOCO_SIM_DIR, with the sim
    dir and its last commit; refuses an existing file."""
    import argparse

    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    ap = argparse.ArgumentParser(prog="sixdof_plan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--out", default=LAB_ROBOT)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"refusing: {a.out} exists")
    env = MuJoCoLabEnv()
    env.reset("ycb__sugar_box", seed=0)
    out = lab_robot(env.sim)
    sim_dir = os.environ["RPENT_MUJOCO_SIM_DIR"]
    commit = subprocess.run(["git", "-C", sim_dir, "log", "-1", "--format=%H", "--", "."], capture_output=True, text=True).stdout.strip()
    out = {"sim_dir": sim_dir, "sim_commit": commit, **out}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out["fit"], indent=1))


if __name__ == "__main__":
    main()
