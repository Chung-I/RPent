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

"""The objects' names in any text an agent or a VLM reads. Six HOT3D ids name the wrong object (the real-masses study,
daily-logs experiments/2026-11-03-real-masses, sources/hot3d.md): ``data/object_names_v11.json``, pinned by its sha256,
maps each to its old and its corrected name. The ids stay, because they seed the draws and name the asset folders and
the splits, so v10 rows, the stored VLM answers and the E1 records still use them. Under generator v11 every text for
an agent or a VLM uses the corrected name, and the id an agent sees carries it ("hot3d__whiteboard_eraser"); v10 text
is unchanged."""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

NAMES_V11 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "object_names_v11.json")
NAMES_V11_SHA256 = "6559c899e65d8a44994c6b51819efa1dd0242a2730f660b1641b10632fdc7062"
_MAPS: dict[str, dict[str, Any]] = {}


def load_names(path: str | None = None) -> dict[str, dict[str, str]]:
    """The renamed objects by id ({old, new, evidence}, cached per path); a file whose sha256 is not the pinned one is
    an error."""
    path = path or NAMES_V11
    if path not in _MAPS:
        raw = open(path, "rb").read()
        if hashlib.sha256(raw).hexdigest() != NAMES_V11_SHA256:
            raise ValueError(f"{path}: its sha256 is not the pinned v11 names map's {NAMES_V11_SHA256}")
        _MAPS[path] = json.loads(raw)["objects"]
    return _MAPS[path]


def object_name(object_id: str, generator: str) -> str:
    """The object's plain name: under v11 a renamed object's corrected name; otherwise the id without its dataset prefix
    and a trailing number, underscores as spaces ("hot3d__orange_juice_carton" -> "orange juice carton", "handal__hammer_6"
    -> "hammer")."""
    if generator == "v11" and object_id in load_names():
        return load_names()[object_id]["new"]
    return re.sub(r"_?\d+$", "", object_id.split("__", 1)[-1]).replace("_", " ")


def agent_id(object_id: str, generator: str) -> str:
    """The id an agent sees: under v11 a renamed object's dataset prefix and corrected name ("hot3d__megaphone" ->
    "hot3d__vacuum_jug"); otherwise the id."""
    if generator == "v11" and object_id in load_names():
        prefix = object_id.split("__", 1)[0]
        return f"{prefix}__{re.sub(r'[^a-z0-9]+', '_', load_names()[object_id]['new'].lower())}"
    return object_id


def agent_object(obj: dict[str, Any], generator: str) -> dict[str, Any]:
    """view_env_state's object entry ({name, id, material}): a renamed object's corrected name and agent id under v11,
    the lab's name and id otherwise."""
    if generator == "v11" and obj["id"] in load_names():
        return {"name": object_name(obj["id"], generator), "id": agent_id(obj["id"], generator), "material": obj["material"]}
    return {"name": obj["name"], "id": obj["id"], "material": obj["material"]}


def agent_names(obj: dict[str, Any], generator: str) -> set[str]:
    """The extra names an agent may call a renamed object under v11 (lower case): its corrected name and agent id, with
    spaces or underscores; none otherwise."""
    if generator != "v11" or obj["id"] not in load_names():
        return set()
    out = {object_name(obj["id"], generator).lower(), agent_id(obj["id"], generator).lower()}
    return out | {n.replace("_", " ") for n in out} | {n.replace(" ", "_") for n in out}
