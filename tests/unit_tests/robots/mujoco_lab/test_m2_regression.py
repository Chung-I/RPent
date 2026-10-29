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

"""The milestone 2a bundle (data/m2) keeps its menus while milestone 2b changes the skill: the rows, their
predictions, costs and probe values, and the recommendation under the milestone 2a rule (abstain_cost None).
The fixture was recorded at grasp-place-skill f109c7f, before any milestone 2b change:

    PYTHONPATH=. python -P -m tests.unit_tests.robots.mujoco_lab.test_m2_regression   (from the RPent root)
"""

import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "m2_menus.json")
CASES = [("hope__popcorn_box", 0), ("hope__popcorn_box", 1), ("ycb__mug", 0), ("ycb__mug", 1),
         ("ycb__tomato_soup_can", 0), ("ycb__tomato_soup_can", 1), ("handal__hammer_1", 0), ("hot3d__spatula", 0),
         ("hot3d__storage_box", 0)]     # the last three decline: low P(placed), and no usable proposal
SEED = 5
MOTION = ("placed_stable", "tipped", "missed", "dropped_low", "not_lifted", "dropped_high", "rolled_off")


def _probe(v):
    if v is None:
        return None
    return {"p_kind": v["p_kind"], "p_thrown": v["p_thrown"], "evoi": v["evoi"], "evoi_se": v["evoi_se"],
            "expected_cost_after": v["expected_cost_after"]}


def menus() -> list[dict]:
    """The m2 menus of CASES, unrounded, through the calibration path (case draw, reset, the menu's stream)."""
    from robots.mujoco_lab.adapter import object_props
    from robots.mujoco_lab.calibrate import MESH_IDS, _primitives, _stream, case_draw
    from robots.mujoco_lab.robot_spec import DEFAULT_BUNDLE
    from rpent.skills.pick_place.tools import _assess

    prims = _primitives(str(DEFAULT_BUNDLE), None, None)
    prims.abstain_cost = None
    out = []
    for object_id, idx in CASES:
        theta, com_end, destination = case_draw(SEED, object_id, idx, object_props(object_id)["mass"], False)
        prims.env.reset(object_id, theta, seed=SEED * 1000 + idx, com_end=com_end,
                        fixture=destination if destination in MESH_IDS else None)
        prims.rng = _stream(SEED, object_id, idx, 1)
        err, a = _assess("object", destination, "normal", prims, {})
        assert err is None, err
        out.append({"object": object_id, "idx": idx, "destination": destination, "recommendation": a.recommendation,
                    "rows": [{"id": o.id, "kind": o.kind, "p": None if o.pred is None else dict(o.pred.p),
                              "expected_cost": o.expected_cost, "reason": o.reason, "probe": _probe(o.probe)}
                             for o in a.options]})
    return out


def _close(a, b, path=""):
    if isinstance(a, dict):
        assert set(a) == set(b), f"{path}: keys {sorted(a)} != {sorted(b)}"
        for k in a:
            _close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), f"{path}: lengths {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and b is not None:
        assert b == pytest.approx(a, abs=1e-9), f"{path}: {b} != {a}"
    else:
        assert a == b, f"{path}: {b!r} != {a!r}"


def test_m2_bundle_menus_unchanged():
    want = json.load(open(FIXTURE))
    got = menus()
    for case in got:                       # classes added after the recording must predict 0 under the m2 models
        for row in case["rows"]:
            if row["p"] is not None:
                extra = {k: v for k, v in row["p"].items() if k not in MOTION}
                assert all(v == 0.0 for v in extra.values()), extra
                row["p"] = {k: v for k, v in row["p"].items() if k in MOTION}
    _close(want, got)


if __name__ == "__main__":
    json.dump(menus(), open(FIXTURE, "w"), indent=1)
    print("wrote", FIXTURE)
