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

"""Corpus generator: rollouts with hidden properties, destinations and labels (needs the sim)."""

import json
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")

ROW_KEYS = {
    "key", "object", "theta", "com_end", "truth", "seed", "destination", "s", "force", "v_lift",
    "v_transport", "release_height", "width", "label", "max_z", "drop_z", "loss_phase", "loss_step",
    "loss_t", "loss_speed", "zone", "release", "planned", "geometry", "features",
}


def test_draw_theta_ranges():
    from robots.mujoco_lab.corpus import draw_theta

    rng = np.random.default_rng(0)
    draws = [draw_theta(rng, 0.2) for _ in range(500)]
    scales = [t["mass_scale"] for t, _ in draws]
    assert 0.5 <= min(scales) and max(scales) <= 3.0
    assert all(0.1 <= t["mu"] <= 0.9 for t, _ in draws)
    assert {t["v_lift"] for t, _ in draws} == {0.05, 0.15, 0.4, 0.8}
    assert {e for _, e in draws} == {-1, 1}
    com = [t["com_s"] for t, _ in draws]
    assert 0.3 < sum(c is None for c in com) / len(com) < 0.7
    assert all(0.3 <= c <= 0.8 for c in com if c is not None)


def test_corpus_two_objects_two_draws(tmp_path):
    from robots.mujoco_lab.corpus import main
    from rpent.skills.pick_place.protocol import OUTCOMES

    out = tmp_path / "c.jsonl"
    n = main(["--out", str(out), "--draws", "2", "--workers", "1", "--objects", "ycb__banana,ycb__mug"])
    rows = [json.loads(line) for line in out.open()]
    assert n == 4 and len(rows) == 4
    assert all(set(r) == ROW_KEYS for r in rows)
    assert {r["label"] for r in rows} <= set(OUTCOMES)
    assert all(set(r["theta"]) == {"mu", "mass_scale", "com_s", "v_lift"} for r in rows)
    assert all(r["planned"] and r["truth"]["mass"] > 0 for r in rows)
    assert all(r["destination"] in ("table_centre", "table_edge", "table_gap", "mouse_pad", "plate", "box", "bin") for r in rows)
    # resume: nothing new is written
    assert main(["--out", str(out), "--draws", "2", "--workers", "1", "--objects", "ycb__banana,ycb__mug", "--resume"]) == 0
    assert len([line for line in out.open()]) == 4
