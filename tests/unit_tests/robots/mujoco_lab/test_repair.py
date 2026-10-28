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

"""The repair loop in the lab: a failed attempt moves the belief and the fresh menu, and an object on
the floor is out of reach (needs the sim and the m2 bundle's models)."""

import os

import pytest

SIM = os.environ.get("RPENT_MUJOCO_SIM_DIR")
BUNDLE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "robots", "mujoco_lab", "data", "m2"))
pytestmark = pytest.mark.skipif(not (SIM and os.path.exists(os.path.join(BUNDLE, "readout_rot.joblib"))),
                                reason="sim dir or m2 bundle missing")
THETA = {"mu": 0.12, "mass_scale": 2.8, "com_s": None, "v_lift": 0.15}


def _lab():
    from robots.mujoco_lab.robot_spec import build_lab

    return build_lab(object_id="ycb__brick", theta=THETA, seed=0, bundle=BUNDLE)


def _row(assessment, rid):
    return next(o for o in assessment["options"] if o["id"] == rid)


def test_failed_attempt_updates_the_belief_and_the_menu():
    """The brick at mu 0.12 and 2.8 times its mass (5.5 N) does not lift with commit_std (2 mu F = 2.4 N):
    the posterior friction drops below 0.3 (prior median 0.5) with at least 100 effective particles, and
    the fresh menu predicts the failure: every row's success falls, commit_std's not_lifted rises, and the
    recommendation is decline. (Not "the best row is not commit_std": at this truth every row fails, firm
    grip too, and not lifting is the cheapest failure, so commit_std ranks first under the posterior as
    under the truth.)"""
    from rpent.skills.pick_place.tools import assess_pick, pick_and_place

    prims, store = _lab(), {}
    a0 = assess_pick(object="ycb__brick", destination="table_centre", primitives=prims, store=store).data
    r = pick_and_place(object="ycb__brick", destination="table_centre", option_id="commit_std", primitives=prims, store=store).data
    assert r["outcome"] == "not_lifted" and r["diagnosis"][0].startswith("never left the table")
    assert r["belief_before"]["mu_median"] == pytest.approx(0.5, abs=0.05)
    assert r["belief"]["mu_median"] < 0.3 and r["belief"]["ess"] >= 100 and r["belief"]["n_observations"] == 1
    fresh = r["assessment"]
    assert fresh["assessed"] is True and fresh["recommendation"] == "decline"
    for rid in ("commit_std", "firm_grip", "slow_lift", "regrasp_com"):
        assert _row(fresh, rid)["p"]["placed_stable"] < _row(a0, rid)["p"]["placed_stable"], rid
    assert _row(fresh, "commit_std")["p"]["not_lifted"] > _row(a0, "commit_std")["p"]["not_lifted"]


def test_locate_reports_reachability():
    """On the table the object is reachable; dropped on the floor past the table it is not, and
    assess_pick offers no rows."""
    import mujoco

    from rpent.skills.pick_place.tools import assess_pick

    prims = _lab()
    env = prims.env
    assert env.locate("ycb__brick")["reachable"] is True
    sim = env.sim
    adr = sim.m.jnt_qposadr[sim.m.body_jntadr[sim.obj]]
    sim.d.qpos[adr:adr + 3] = [0.0, 0.6, 0.1]
    sim.d.qvel[sim.m.jnt_dofadr[sim.m.body_jntadr[sim.obj]]:][:6] = 0.0
    mujoco.mj_forward(sim.m, sim.d)
    sim.settle(1.0)
    assert env.locate("ycb__brick")["reachable"] is False
    d = assess_pick(object="ycb__brick", destination="table_centre", primitives=prims, store={}).data
    assert d == {"assessed": False, "reason": "object is out of reach (floor or bin)", "next": "report it; the skill cannot reach it"}
