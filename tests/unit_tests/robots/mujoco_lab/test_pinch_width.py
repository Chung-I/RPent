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

"""The planner's pinch width is the hulls' exact extent between the pads (needs the sim).

The width came from the hull triangles' centres in the pad's band: for ycb__cheez_it d0 of corpus v7 it was 0.0 cm,
and the pads closed on 17.0 cm."""

import os

import numpy as np

import pytest

from tests.unit_tests.skills.test_features import _lp_width

pytestmark = pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")


def _hulls(env):
    sim, out = env.sim, []
    for g in sim.hull_ids:
        mid = sim.m.geom_dataid[g]
        v0, nv = sim.m.mesh_vertadr[mid], sim.m.mesh_vertnum[mid]
        out.append(sim.m.mesh_vert[v0:v0 + nv] @ sim.d.geom_xmat[g].reshape(3, 3).T + sim.d.geom_xpos[g])
    return out


@pytest.mark.parametrize("object_id", ["ycb__cheez_it", "hot3d__keyboard", "ycb__tomato_soup_can"])
def test_proposal_width_is_the_exact_width(object_id):
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    env = MuJoCoLabEnv()
    env.reset(object_id, seed=0)
    for p in env.proposals():
        want = _lp_width(_hulls(env), p.point, p.yaw)
        assert abs(p.width - want) < 1e-7, (p.s, p.width, want)


def test_the_features_see_the_proposal_width():
    from robots.mujoco_lab.adapter import MuJoCoLabEnv
    from rpent.skills.pick_place.features import extract

    env = MuJoCoLabEnv()
    env.reset("ycb__cheez_it", seed=0)
    g = env.geometry()
    assert len(g.hulls) == len(env.sim.hull_ids)
    for p in env.proposals():
        f = extract(g, np.asarray(p.point), p.yaw, 0.4, 0.5, 20.0, 0.15, 0.5, "bin")
        assert f["width"] == pytest.approx(p.width, abs=1e-12)
