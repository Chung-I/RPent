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

"""The arm port's adapter rules (daily-logs experiments/2026-11-04-arm-port, design "Settled by the paper session"):
the finger drive by name, the readouts from the joint torques (attempt, weigh, push, squeeze), the jam record, the
reachability rule, unreachable probes, the time and the 50 Hz sidecar fields. Runs only on the arm sim."""

import math
import os

import numpy as np
import pytest

SIM = os.environ.get("RPENT_MUJOCO_SIM_DIR", "")
pytestmark = pytest.mark.skipif(not (SIM and os.path.exists(os.path.join(SIM, "ident.py"))), reason="not the arm sim")

THETA = {"mass_scale": 1.0, "mu": 0.8, "com_s": None, "v_lift": 0.1}


@pytest.fixture()
def env(monkeypatch):
    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    e = MuJoCoLabEnv(damage_meter=True)
    e.reset("ycb__wood_block", THETA, seed=3)
    return e


def _mass(env):
    return float(env.sim.m.body_subtreemass[env.sim.obj])


def test_the_finger_drive_is_found_by_name(env):
    sim = env.sim
    assert env._finger_act() == sim.finger_act[0]
    assert sim.finger_act[0] not in set(sim.arm.act.tolist())
    sim.grip(7.0)
    assert env._finger_drive() == 7.0
    assert env._finger_hi() == float(sim.m.actuator_ctrlrange[sim.finger_act[0], 1])


def test_the_bin_surface_follows_its_destination(env, monkeypatch):
    from rpent.skills.pick_place.destinations import DESTINATIONS

    monkeypatch.setitem(DESTINATIONS, "bin", {**DESTINATIONS["bin"], "xy": (-0.5, 0.6), "surface_z": 0.45})
    assert env._surface_below(np.array([-0.5, 0.6])) == (0.45, True)
    assert env._surface_below(np.array([-0.62, 0.0]))[1] is False


def test_the_attempt_reads_from_the_joints(env):
    from robots.mujoco_lab.adapter import ARM_SUB

    prop = env.proposals()[1]
    r = env.execute(prop, 15.0, 0.1, 0.5, "mouse_pad")
    m = _mass(env)
    j = r.readouts["joint"]
    assert j["floor"] == [0.001, 0.0005]
    assert np.array(j["cov"])[0, 0] >= 0.001 ** 2
    assert r.sensor["source"] == "joint" and r.sensor["force_noise"] == 0.0
    assert abs(r.readouts["fz_hold_n"] - m * 9.81) / (m * 9.81) < 0.05
    held = [x for x in r.trace if x["phase"] == "hold"]
    assert all(x["fz_joint"] is not None for x in held)
    hold = r.readouts["arm_phases"]["hold"]
    assert abs(hold["wrench_mean"][2] - m * 9.81) / (m * 9.81) < 0.05 and len(hold["tau_mean"]) == 7
    assert "transport" in r.readouts["arm_phases"]
    side = r.arm
    n = len(side["t"])
    assert side["q"].shape == (n, 7) and side["tau"].shape == (n, 7) and side["wrench"].shape == (n, 6)
    assert np.allclose(np.diff(side["t"]), ARM_SUB * float(env.s["scene"].DT), atol=1e-6)
    assert (side["step"] == -1).any()                     # the free move is logged, not traced
    hz = env.s["scene"].CTRL_HZ
    assert r.time_s > len(r.trace) / hz
    assert set(r.sensor["jam"]) == {"jam", "jam_n", "arm_max_n", "artifact"} and r.sensor["jam"]["arm_max_n"] > 50
    assert abs(r.sensor["pad_readouts"]["fz_hold_n"] - r.readouts["fz_hold_n"]) / (m * 9.81) < 0.05
    sat = r.sensor["saturation"]
    assert sat["of"] > 0 and len(sat["per_joint"]) == 7
    assert side["qd"].shape == (n, 7) and side["flange_hidden"].shape == (n, 6) and side["saturated"].dtype == bool
    held = side["step"] >= 0
    assert np.allclose(side["f_act"][held & (side["f_act"] > 0)], 15.0)


def test_the_weigh_reads_the_mass_from_the_joints(env):
    prop = env.proposals()[1]
    res = env.probe_weigh(prop, 20.0)
    m = _mass(env)
    assert res.kind == "measured", res.kind
    assert res.reading["joint"]["params"] == 4
    assert abs(res.reading["m_hat"] - m) / m < 0.03
    assert res.reading["m_hat"] == res.reading["joint"]["phi"][0]
    assert res.sensor["source"] == "joint" and "hold" in res.sensor["arm_phases"]
    assert "arm" not in res.to_data() and len(res.arm["t"]) > 0


def test_the_push_moves_freely_and_reads_the_joints(env):
    res = env.probe_push()
    assert res.kind in ("slid", "tipped", "no_breakaway", "no_contact"), res.kind
    assert res.sensor["source"] == "joint"
    assert all("f" in x for x in res.trace)
    if res.kind == "slid":
        mg = _mass(env) * 9.81
        assert 0.1 * mg < res.reading["f_break"] < 2.0 * mg
    assert res.seconds > len(res.trace) / env.s["scene"].CTRL_HZ


def test_the_squeeze_reads_the_actuator_force(env):
    prop = env.proposals()[1]
    res = env.probe_squeeze(prop, 6.0)
    assert res.kind in ("measured", "damaged")
    assert abs(res.reading["f_act_n"] - 6.0) < 1e-6


def test_an_unreachable_probe_is_counted_not_run(env, monkeypatch):
    prop = env.proposals()[1]
    p0 = env.sim.obj_pos().copy()
    monkeypatch.setattr(env.sim, "plan_free", lambda *a, **k: {"ok": False, "reason": "ik", "q": None, "steps": 0})
    res = env.probe_weigh(prop, 20.0)
    assert res.kind == "unreachable" and res.seconds == 0.0 and res.sensor["reach"]["reason"] == "ik"
    assert np.allclose(env.sim.obj_pos(), p0)
    assert env.probe_push().kind == "unreachable"


def test_the_reach_rule(env, monkeypatch):
    from rpent.skills.pick_place.destinations import DESTINATIONS

    prop = env.proposals()[1]
    q0 = env.sim.d.qpos.copy()
    assert env.reach(prop, "table_centre") is None
    assert np.array_equal(env.sim.d.qpos, q0)              # nothing moved
    monkeypatch.setitem(DESTINATIONS, "bin", {**DESTINATIONS["bin"], "xy": (-0.62, 0.0), "surface_z": 0.0})
    out = env.reach(prop, "bin")
    assert out is not None and out["reason"] in ("ik", "collision")
    path = env.hand_path(prop, 0.1, 0.5, "table_centre")
    assert path[0][0] == "wait" and path[-1][0] == "retreat"


def test_the_corpus_declines_an_unreachable_row_and_keeps_the_arm_fields(env, monkeypatch, tmp_path):
    from robots.mujoco_lab import corpus
    from rpent.skills.pick_place.destinations import DESTINATIONS

    row = corpus.rollout_row(env, "hope__corn_can", 0, 0)
    assert "declined" not in row
    if True:
        assert row["time_s"] > 0 and "_arm" in row
        side = {}
        out = corpus.pop_arm(dict(row), side, "c.jsonl")
        assert "_arm" not in out and f"c.jsonl|{row['key']}|tau" in side
        path = str(tmp_path / "c_arm.npz")
        corpus.save_sidecar(path, side)
        corpus.save_sidecar(path, {"c.jsonl|x|tau": np.zeros(2)}, resume=True)
        with np.load(path) as z:
            assert f"c.jsonl|{row['key']}|tau" in z.files and "c.jsonl|x|tau" in z.files
    assert corpus.sidecar_path("a/corpus_v12.jsonl") == "a/corpus_v12_arm.npz"
    monkeypatch.setattr(env, "reach", lambda *a, **k: {"reason": "ik", "phase": "transport"})
    rows = [corpus.rollout_row(env, "hope__corn_can", i, 0) for i in range(1, 6)]
    rows = [r for r in rows if r.get("declined") != "wider than the gripper opening"]
    assert rows and all(r["declined"] == "unreachable" and r["reach"]["reason"] == "ik" and "label" not in r for r in rows)


def test_the_menu_is_unreachable_when_no_pinch_is(env, monkeypatch):
    from rpent.skills.pick_place import tools

    calls = []

    def reach(p, destination, *a, **k):
        calls.append(destination)
        return {"reason": "ik"}

    monkeypatch.setattr(env, "reach", reach)

    class Prims:
        pass

    prims = Prims()
    prims.env = env
    err, a = tools._assess("object", "table_centre", "normal", prims, {})
    assert a is None and err["assessed"] is False and err["reason"] == "unreachable"
    assert calls and set(calls) == {"table_centre"}
