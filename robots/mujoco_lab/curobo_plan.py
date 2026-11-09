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

"""The cuRobo side of the sixdof rerun (daily-logs design 2026-11-08-sixdof-rerun-design.md, unit (e), gate G3).

cuRobo plans the free move from the arm's start joints to the chosen configuration at the pregrasp (plan_cspace). It
runs in its own venv (~/Codes/curobo/.venv) as this file run as a script, one process per case; the lab never imports
it. Both sides share the plan cache: an entry per sha256 key of the canonical request (the world, the start, the goal
and the planner's identity: the robot yml's sha256, the settings, cuRobo's version and commit, torch's), failures
included. cuRobo plans are not repeatable (the costing: 0 of 7 repeated plans bit-identical), so a run reads the cache
only and never plans: a miss stops the row (G3). This module imports cuRobo and torch inside the planning functions
only, so the lab imports it for the keys and the entries.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from typing import Any

import numpy as np

PANDA_V_MAX = (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)     # rad/s, the Panda's joint speed limits (franka_panda.urdf)
HAND_LINKS = {"panda_hand": "lab_hand", "panda_leftfinger": "lab_finger_l", "panda_rightfinger": "lab_finger_r"}
SETTINGS = {"position_tolerance": 0.001, "orientation_tolerance": 0.01, "interpolation_dt": 0.02, "random_seed": 123,
            "collision_cache": {"mesh": 24, "primitive": 48}, "use_cuda_graph": True, "warmup_iterations": 5,
            "max_attempts": 20, "enable_graph_attempt": 1, "finetune_attempts": 1}


def _plain(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_plain(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if not math.isfinite(float(x)):
            raise ValueError(f"not a finite number: {x}")
        return float(x)
    return x


def canonical(obj: Any) -> bytes:
    """The bytes a key hashes: sorted keys, no spaces, floats by repr (json's), numpy turned into lists; NaN refused."""
    return json.dumps(_plain(obj), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def plan_key(world: Any, start: Any, goal: Any, identity: Any) -> str:
    """The cache key of a free move: sha256 of the canonical {world, start, goal, identity}. No tolerance: a float that
    differs in its last bit is another key."""
    return hashlib.sha256(canonical({"world": world, "start": start, "goal": goal, "identity": identity})).hexdigest()


def entry_path(cache: str, key: str) -> str:
    return os.path.join(cache, "entries", key[:2], key + ".json")


def read_entry(cache: str, key: str) -> dict[str, Any] | None:
    """The entry of ``key`` ({"key", "success", "status", "dt", "q" (array or None), ...}), None when not cached."""
    p = entry_path(cache, key)
    if not os.path.exists(p):
        return None
    e = json.load(open(p))
    e["q"] = None if e.get("q") is None else np.asarray(e["q"], dtype=float)
    return e


def write_entry(cache: str, key: str, entry: dict[str, Any]) -> None:
    """Write ``entry`` under ``key`` atomically (a temporary file renamed). An entry is written once: the same entry
    again does nothing, a different one raises ValueError."""
    body = _plain(dict(entry, key=key))
    p = entry_path(cache, key)
    if os.path.exists(p):
        if canonical(json.load(open(p))) != canonical(body):
            raise ValueError(f"the cached entry {key} differs from the new one")
        return
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(p), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(body, f)
    os.replace(tmp, p)


def robot_yml(franka: dict[str, Any], lab: dict[str, Any]) -> dict[str, Any]:
    """cuRobo's robot config "Panda + lab gripper" from franka.yml (as a dict) and the lab's fit (``lab``: mount_pos,
    mount_quat (w, x, y, z), pad_centre_z, free_v, arm_spheres for panda_link1..7, gripper_spheres for lab_hand and the
    fingers): the Franka hand's links leave the collision model; the gripper hangs from panda_link7 at the mount; the
    tool frame lab_tcp sits at the pads' centre with +Z along the approach (Rx(pi) of the hand frame); link 0 keeps
    franka.yml's spheres; the arm's joint speed is capped at free_v (the lab's free move) by the velocity scale."""
    fy = copy.deepcopy(franka)
    k = fy["robot_cfg"]["kinematics"]

    def fix(name: str, parent: str, T: list[float]) -> dict[str, Any]:
        return {"fixed_transform": [float(x) for x in T], "joint_name": name + "_joint", "joint_type": "FIXED",
                "link_name": name, "parent_link_name": parent}

    k["extra_links"] = {
        "lab_hand": fix("lab_hand", "panda_link7", [*lab["mount_pos"], *lab["mount_quat"]]),
        "lab_finger_l": fix("lab_finger_l", "lab_hand", [0, 0, 0, 1, 0, 0, 0]),
        "lab_finger_r": fix("lab_finger_r", "lab_hand", [0, 0, 0, 1, 0, 0, 0]),
        "lab_tcp": fix("lab_tcp", "lab_hand", [0, 0, lab["pad_centre_z"], 0, 1, 0, 0]),
        "attached_object": fix("attached_object", "lab_tcp", [0, 0, 0, 1, 0, 0, 0]),
    }
    k["tool_frames"] = ["lab_tcp"]
    arm = [f"panda_link{i}" for i in range(8)]
    k["collision_link_names"] = arm + list(HAND_LINKS.values()) + ["attached_object"]
    k["grasp_contact_link_names"] = ["lab_finger_l", "lab_finger_r", "attached_object"]
    k["mesh_link_names"] = arm
    sph = {l: s for l, s in k["collision_spheres"].items() if l not in HAND_LINKS}
    sph.update(lab["arm_spheres"])
    sph.update(lab["gripper_spheres"])
    k["collision_spheres"] = sph
    k["self_collision_buffer"] = {HAND_LINKS.get(l, l): v for l, v in k["self_collision_buffer"].items()}
    ign = {HAND_LINKS.get(l, l): [HAND_LINKS.get(o, o) for o in (others or [])] for l, others in k["self_collision_ignore"].items()}
    for l in ("panda_link5", "panda_link6", "panda_link7"):
        ign[l] = ign.get(l, []) + [o for o in HAND_LINKS.values() if o not in ign.get(l, [])]
    ign["lab_hand"] = sorted(set(ign.get("lab_hand", []) + ["lab_finger_l", "lab_finger_r", "attached_object"]))
    ign["lab_finger_l"] = sorted(set(ign.get("lab_finger_l", []) + ["lab_finger_r", "attached_object"]))
    k["self_collision_ignore"] = ign
    n = len(k["cspace"]["joint_names"])
    k["cspace"]["velocity_scale"] = [float(lab["free_v"]) / v for v in PANDA_V_MAX] + [1.0] * (n - len(PANDA_V_MAX))
    return fy
