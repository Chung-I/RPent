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
is unchanged. Under v11 an agent also sees the material of the real object (``data/materials_v11.json``, pinned by its
sha256), not the lab's, which was often guessed from the name, and every object's name in one format (spaces)."""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

NAMES_V11 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "object_names_v11.json")
NAMES_V11_SHA256 = "6559c899e65d8a44994c6b51819efa1dd0242a2730f660b1641b10632fdc7062"
MATERIALS_V11 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "materials_v11.json")
MATERIALS_V11_SHA256 = "68efd51603df33f21f572e21b654684caa7b643be281b63aa93e6c37bb5736dd"
_MAPS: dict[str, dict[str, Any]] = {}
_MATERIALS: dict[str, dict[str, Any]] = {}


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


def load_materials(path: str | None = None) -> dict[str, dict[str, Any]]:
    """The material an agent is told under v11, by id ({material, lab, changed, basis, evidence}, cached per path); a
    file whose sha256 is not the pinned one is an error."""
    path = path or MATERIALS_V11
    if path not in _MATERIALS:
        raw = open(path, "rb").read()
        if hashlib.sha256(raw).hexdigest() != MATERIALS_V11_SHA256:
            raise ValueError(f"{path}: its sha256 is not the pinned v11 materials table's {MATERIALS_V11_SHA256}")
        _MATERIALS[path] = json.loads(raw)["objects"]
    return _MATERIALS[path]


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
    """view_env_state's object entry ({name, id, material}): under v11 the plain name (a renamed object's corrected
    one), the agent id and the real material (data/materials_v11.json); the lab's record otherwise."""
    if generator == "v11":
        return {"name": object_name(obj["id"], generator), "id": agent_id(obj["id"], generator),
                "material": load_materials()[obj["id"]]["material"]}
    return {"name": obj["name"], "id": obj["id"], "material": obj["material"]}


def _words(name: str) -> str:
    return " ".join(re.split(r"[\s_-]+", name.strip().lower()))


def accepts(obj: dict[str, Any], prompt: str, generator: str) -> bool:
    """locate's name check. Every generator takes the lab's names (the id, the name, the name with spaces, "object");
    v11 also takes the names an agent sees (the plain name, the agent id) and the lab's, with any mix of spaces,
    hyphens and underscores."""
    p = prompt.strip().lower()
    if p in {obj["id"].lower(), obj["name"].lower(), "object", obj["name"].lower().replace("_", " ")}:
        return True
    if generator != "v11":
        return False
    return _words(p) in {_words(n) for n in (object_name(obj["id"], generator), agent_id(obj["id"], generator), obj["name"], obj["id"])}
