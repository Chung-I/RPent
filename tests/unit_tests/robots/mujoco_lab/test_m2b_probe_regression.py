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


"""The milestone 2b bundle (data/m2b) keeps its probe cases while milestone 2b2 changes the lab and the skill: two
probe_all cases (the push, the weigh, then the best row) through calibrate.run_case, every field the fixture holds
(later fields may be added) but the run time. The menu pins see only first menus; this pin also covers the probe
readings, the belief updates and the attempt. The fixture was recorded at grasp-place-skill c41cb9b, before any
milestone 2b2 change, from a checkout of that commit with this bundle (its models are local, gitignored, so the test
skips without them):

    PYTHONPATH=<c41cb9b checkout> python -P tests/unit_tests/robots/mujoco_lab/test_m2b_probe_regression.py <data/m2b>
"""

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "m2b_probe_cases.json")
CASES = [("ycb__mug", 3), ("ycb__chocolate_pudding", 0), ("ycb__tomato_soup_can", 0)]   # a filled mug; a crushable box; a can
SEED = 5
VOLATILE = {"seconds"}


def _plain(x):
    return x.item() if hasattr(x, "item") else str(x)


def cases(bundle: str) -> list[dict]:
    """The probe_all cases of CASES on the bundle, as JSON values."""
    from robots.mujoco_lab.calibrate import _primitives, run_case

    prims = _primitives(bundle, None, None)
    return [json.loads(json.dumps(run_case(prims, o, i, SEED, "probe_all"), default=_plain)) for o, i in CASES]


def _within(a, b, path=""):
    """Every value of the fixture a is in b unchanged (floats to 1e-9); b may hold more keys."""
    if isinstance(a, dict):
        missing = set(a) - set(b) - VOLATILE
        assert not missing, f"{path}: missing keys {sorted(missing)}"
        for k in set(a) - VOLATILE:
            _within(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), f"{path}: lengths {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _within(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, (int, float)):
        assert b == pytest.approx(a, abs=1e-9), f"{path}: {b} != {a}"
    else:
        assert a == b, f"{path}: {b!r} != {a!r}"


def test_m2b_probe_cases_unchanged():
    from robots.mujoco_lab.robot_spec import DATA_DIR

    bundle = str(DATA_DIR / "m2b")
    if not os.path.exists(os.path.join(bundle, "success_model.joblib")):
        pytest.skip("the m2b bundle's models are local (gitignored)")
    _within(json.load(open(FIXTURE)), cases(bundle))


if __name__ == "__main__":
    json.dump(cases(sys.argv[1]), open(FIXTURE, "w"), indent=1)
    print("wrote", FIXTURE)
