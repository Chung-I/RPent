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


"""The lab's real-object masses (generator v11): ``data/masses_v11.json``, the frozen table of the real-masses study
(daily-logs experiments/2026-11-03-real-masses), pinned by its sha256. Under generator v11 ``adapter.object_props``
gives each object its real mass (``real_props``): every hull mass times real / props.json's mass, so the CoM and the
ratio of the part densities stay as they were. The table has the 95 objects whose mesh compiles (the pomegranate's
does not: it has no row)."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

MASSES_V11 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "masses_v11.json")
MASSES_V11_SHA256 = "d1cd4d7dde6c90d813dfb873da6a181c84c0c1b590b8651b3fd5fc64a776ead0"
NOMINAL_ATOL = 5e-5          # the table keeps the v10 nominal to 0.1 g
_TABLES: dict[str, dict[str, Any]] = {}


def load_masses(path: str | None = None) -> dict[str, Any]:
    """The v11 table (cached per path); a file whose sha256 is not the pinned one is an error."""
    path = path or MASSES_V11
    if path not in _TABLES:
        raw = open(path, "rb").read()
        if hashlib.sha256(raw).hexdigest() != MASSES_V11_SHA256:
            raise ValueError(f"{path}: its sha256 is not the pinned v11 table's {MASSES_V11_SHA256}")
        _TABLES[path] = json.loads(raw)
    return _TABLES[path]


def real_props(object_id: str, props: dict[str, Any]) -> dict[str, Any]:
    """``props`` (props.json of ``object_id``) at the real mass: the table's mass, every hull mass times real /
    props["mass"]. The table's v10 nominal must be props.json's mass, so a changed mesh cannot change the factor."""
    entry = load_masses()["objects"].get(object_id)
    if entry is None:
        raise ValueError(f"{object_id} has no real mass in {os.path.basename(MASSES_V11)} (generator v11)")
    nominal = float(props["mass"])
    if abs(float(entry["nominal_kg_v10"]) - nominal) > NOMINAL_ATOL:
        raise ValueError(f"{object_id}: props.json's mass {nominal} is not the table's v10 nominal {entry['nominal_kg_v10']}")
    k = float(entry["mass_kg"]) / nominal
    return {**props, "mass": float(entry["mass_kg"]), "hull_mass": [float(m) * k for m in props["hull_mass"]]}
