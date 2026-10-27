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

"""Synthetic corpus rows for the model tests: labels follow thresholds on a few features with noise."""

from __future__ import annotations

import numpy as np

from rpent.skills.pick_place.features import FEATURES

PLANNED_TEMPLATE = [
    {"phase": "approach", "n_steps": 60, "z_start": 0.0, "z_end": 0.0, "has_surface": True, "speed": 0.1},
    {"phase": "close", "n_steps": 30, "z_start": 0.0, "z_end": 0.0, "has_surface": True, "speed": 0.0},
    {"phase": "lift", "n_steps": 67, "z_start": 0.0, "z_end": 0.2, "has_surface": True, "speed": 0.15},
    {"phase": "hold", "n_steps": 25, "z_start": 0.2, "z_end": 0.2, "has_surface": True, "speed": 0.0},
    {"phase": "transport", "n_steps": 40, "z_start": 0.2, "z_end": 0.2, "has_surface": True, "speed": 0.5},
    {"phase": "lower", "n_steps": 90, "z_start": 0.2, "z_end": 0.02, "has_surface": True, "speed": 0.1},
    {"phase": "release", "n_steps": 150, "z_start": 0.02, "z_end": 0.02, "has_surface": True, "speed": 0.0},
]


def synthetic_rows(n: int = 300, seed: int = 0, n_objects: int = 30, noise: float = 0.05) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        mass = float(np.exp(rng.uniform(np.log(0.05), np.log(1.5))))
        mu = float(rng.uniform(0.1, 0.9))
        force = float(rng.choice([10.0, 20.0, 40.0]))
        lever = float(abs(rng.normal(0, 0.03)))
        base_ratio = float(rng.uniform(0.1, 1.5))
        opening = float(rng.choice([10.0, 0.8, 1.5, 2.5]))
        v_lift = float(rng.choice([0.05, 0.15, 0.4, 0.8]))
        f = {k: 0.0 for k in FEATURES}
        f.update({"width": 0.04, "slope_deg": 10.0, "lever": lever, "com_h": 0.0, "mass": mass, "mu": mu,
                  "load_ratio": mass * 9.81 / (2 * mu * force), "v_lift": v_lift, "v_transport": 0.5, "length": 0.1,
                  "visibility": 1.0, "base_ratio": base_ratio, "com_height_ratio": 0.3, "dest_mu": 0.8,
                  "dest_rim_height": 0.0, "dest_opening_ratio": opening, "dest_edge_dist": 0.2,
                  "dest_has_surface": 1.0, "dest_release_height": 0.02})
        lr = f["load_ratio"] * (1 + v_lift)
        if rng.random() < noise:
            label = str(rng.choice(["placed_stable", "tipped", "dropped_low", "dropped_high", "not_lifted", "missed", "rolled_off"]))
        elif lr > 1.6:
            label = "not_lifted"
        elif lr > 0.8:
            label = "dropped_high" if rng.random() < 0.6 else "dropped_low"
        elif lever > 0.03:
            label = "dropped_low"
        elif opening < 1.0:
            label = "missed"
        elif base_ratio <= 0.4:
            label = "tipped"
        else:
            label = "placed_stable"
        planned = [dict(s) for s in PLANNED_TEMPLATE]
        loss_phase = loss_step = None
        if label in ("dropped_low", "dropped_high"):
            loss_phase = "lift" if label == "dropped_high" and rng.random() < 0.5 else "transport"
            offset = sum(s["n_steps"] for s in planned[: [s["phase"] for s in planned].index(loss_phase)])
            loss_step = offset + int(rng.integers(1, next(s["n_steps"] for s in planned if s["phase"] == loss_phase)))
        rows.append({"key": f"o{i % n_objects}__d{i}", "object": f"o{i % n_objects}", "label": label, "features": f,
                     "planned": planned, "loss_phase": loss_phase, "loss_step": loss_step,
                     "loss_t": None if loss_step is None else loss_step / 50.0,
                     "truth": {"mass": mass, "mu": mu, "com_s": 0.5},
                     "geometry": {"volume": mass / 1000.0, "material": "hard_plastic", "hollow": False,
                                  "base_extent": base_ratio * 0.03, "com_height": 0.03, "footprint_half_width": 0.05, "length": 0.1},
                     "destination": "table_centre", "s": 0.5, "force": force, "v_lift": v_lift, "v_transport": 0.5,
                     "release_height": 0.02, "point_yaw": [0.0, 0.0, 0.76, 0.0]})
    return rows
