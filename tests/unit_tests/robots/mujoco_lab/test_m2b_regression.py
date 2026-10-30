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


"""The milestone 2b bundle (data/m2b) keeps its menus while milestone 2b2 changes the skill: the rows, their
predictions, costs and probe values, and the recommendation under the bundle's cost rule (A 1.0) at care 1 and 10.
The fixture was recorded at grasp-place-skill c41cb9b, before any milestone 2b2 change. The bundle's models are local
(gitignored), so the test skips without them:

    PYTHONPATH=. python -P -m tests.unit_tests.robots.mujoco_lab.test_m2b_regression   (from the RPent root)
"""

import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "m2b_menus.json")
CASES = [("hope__popcorn_box", 0), ("hope__popcorn_box", 1), ("ycb__mug", 0), ("ycb__mug", 1), ("ycb__mug", 2),
         ("ycb__mug", 3), ("ycb__tomato_soup_can", 0), ("ycb__tomato_soup_can", 1), ("handal__hammer_1", 0),
         ("hot3d__spatula", 0), ("hot3d__storage_box", 0), ("ycb__chocolate_pudding", 0),
         ("fruits_veggies__orange1", 0)]       # boxes and the orange get a squeeze row; some mug cases are filled
CARES = (1.0, 10.0)
SEED = 5


def _bundle() -> str:
    from robots.mujoco_lab.robot_spec import DATA_DIR

    return str(DATA_DIR / "m2b")


def _probe(v):
    if v is None:
        return None
    return {"p_kind": v["p_kind"], "p_thrown": v["p_thrown"], "evoi": v["evoi"], "evoi_se": v["evoi_se"],
            "expected_cost_after": v["expected_cost_after"]}


def menus() -> list[dict]:
    """The m2b menus of CASES at each care, unrounded, through the calibration path (the case's m2b draw with its
    fill and fragility, the reset, the menu's stream)."""
    from robots.mujoco_lab.adapter import object_props
    from robots.mujoco_lab.calibrate import MESH_IDS, _primitives, _stream, case_draw
    from robots.mujoco_lab.prior import fragility_world
    from rpent.skills.pick_place.tools import _assess

    prims = _primitives(_bundle(), None, None)
    world = fragility_world(prims.manifest)
    env = prims.env
    out = []
    for care in CARES:
        for object_id, idx in CASES:
            theta, com_end, destination, hidden = case_draw(SEED, object_id, idx, object_props(object_id)["mass"], False,
                                                            fragility=world)
            env.slosh_zeta, env.slosh_scale = world[1].slosh_zeta, world[1].slosh_scale
            env.reset(object_id, theta, seed=SEED * 1000 + idx, com_end=com_end,
                      fixture=destination if destination in MESH_IDS else None, fill=hidden["fill"], frag=hidden["frag"])
            prims.rng = _stream(SEED, object_id, idx, 1)
            err, a = _assess("object", destination, care, prims, {})
            assert err is None, err
            out.append({"object": object_id, "idx": idx, "care": care, "destination": destination, "fill": hidden["fill"],
                        "recommendation": a.recommendation,
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


def test_m2b_bundle_menus_unchanged():
    if not os.path.exists(os.path.join(_bundle(), "success_model.joblib")):
        pytest.skip("the m2b bundle's models are local (gitignored)")
    _close(json.load(open(FIXTURE)), menus())


if __name__ == "__main__":
    json.dump(menus(), open(FIXTURE, "w"), indent=1)
    print("wrote", FIXTURE)
