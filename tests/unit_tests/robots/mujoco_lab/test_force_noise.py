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


"""Plan 2b2 decision 1: the lab's hand reads every force with a declared error. Each force reading is its clean value
times exp(PAD_FORCE_NOISE eps), one standard normal per reading from a force stream per reset (seed, object, 1). The
rollout and the probe result keep the clean values and the eps in ``sensor``, which no planner payload shows (needs
the sim)."""

import json
import math
import os
import zlib

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")

BRICK = "ycb__brick"
LONG = "ycb__banana"
CRUSHABLE = "ycb__jello"
READOUTS = {"hold": "fz_hold_n", "slip": "fz_slip_max_n", "loss": "fz_at_loss_n"}


def _env(**kw):
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    return MuJoCoLabEnv(**kw)


def _mid(env):
    from rpent.skills.pick_place.menu import mid_proposal, pinch_limit, usable_proposals

    return mid_proposal(usable_proposals(env.proposals(), pinch_limit(env)))


def _attempt(env, oid=LONG, seed=0):
    env.reset(oid, {"mu": 0.8, "mass_scale": 1.0, "com_s": None, "v_lift": 0.15}, seed=seed)
    return env.execute(env.proposals()[1], force=20.0, v_lift=0.15, destination="table_centre")


def _weigh(env):
    from rpent.skills.pick_place.probes import weigh_force

    return env.probe_weigh(_mid(env), weigh_force(env.s["t3_make"].std_force(env.nominal_mass())))


def test_force_readings_carry_the_declared_noise():
    """A placed banana: each present readout is its clean value x exp(0.05 eps); the hold is among them."""
    from robots.mujoco_lab.adapter import PAD_FORCE_NOISE

    assert PAD_FORCE_NOISE == 0.05
    r = _attempt(_env())
    s = r.sensor
    assert s["force_noise"] == 0.05 and set(s["eps"]) == set(READOUTS) and set(s["clean"]) == set(READOUTS.values())
    assert s["clean"]["fz_hold_n"] is not None
    for term, key in READOUTS.items():
        clean = s["clean"][key]
        assert (r.readouts[key] is None) == (clean is None)
        if clean is not None:
            assert r.readouts[key] == pytest.approx(clean * math.exp(0.05 * s["eps"][term]), rel=1e-12)
    assert r.readouts["fz_hold_n"] != s["clean"]["fz_hold_n"]


def test_zero_force_noise_reproduces_the_clean_readings(monkeypatch):
    """RPENT_LAB_FORCE_NOISE=0 gives the clean readings, and the noise does not move the object: the noisy run's clean
    values are the noise-free run's readings, and every other readout is the same."""
    noisy = _attempt(_env())
    monkeypatch.setenv("RPENT_LAB_FORCE_NOISE", "0")
    env = _env()
    assert env.force_noise == 0.0
    clean = _attempt(env)
    assert clean.readouts == {**noisy.readouts, **noisy.sensor["clean"]}
    assert clean.label == noisy.label and clean.max_z == noisy.max_z


def test_force_noise_differs_between_objects_of_a_draw():
    """Every object of a draw is reset with one seed, so the force stream includes the object: two objects at seed 5
    draw different eps, and one object at its seed repeats its own."""
    env = _env()
    eps = [_attempt(env, oid, seed=5).sensor["eps"]["hold"] for oid in (LONG, BRICK, LONG)]
    assert eps[0] != eps[1] and eps[0] == eps[2]


def test_force_stream_leaves_the_squeeze_noise_unchanged():
    """The squeeze keeps its own stream: after a weigh, which draws a force eps, the squeeze's eps is still the first
    draw of the reset's stream (seed, crc32(object))."""
    from robots.mujoco_lab.fragility import THRESHOLDS_V0, Thresholds, entry_of, load_classes, map_latent

    env = _env()
    entry, _ = entry_of(CRUSHABLE, load_classes())
    frag = {**map_latent([0.0, 0.0, 0.0], entry, Thresholds.from_json(THRESHOLDS_V0)), "f_damage_n": 60.0}
    env.reset(CRUSHABLE, seed=5, frag=frag)
    assert "eps" in _weigh(env).sensor
    res = env.probe_squeeze(_mid(env), 5.0)
    assert res.reading["eps"] == float(np.random.default_rng([5, zlib.crc32(CRUSHABLE.encode())]).standard_normal())


def test_weigh_reading_carries_the_noise():
    env = _env()
    env.reset(BRICK, {"mu": 0.8, "mass_scale": 1.0, "com_s": None, "v_lift": 0.15}, seed=0)
    res = _weigh(env)
    assert res.kind == "measured" and set(res.reading) == {"m_hat"}
    assert res.sensor["force_noise"] == 0.05
    assert res.reading["m_hat"] == pytest.approx(res.sensor["clean"]["m_hat"] * math.exp(0.05 * res.sensor["eps"]), rel=1e-12)


def test_weigh_slide_readings_share_one_eps():
    """A brick that cannot rise (mu 0.1, mass scale 3): mu_hat and m_lower come from one slide force, so one eps
    scales both."""
    env = _env()
    env.reset(BRICK, {"mu": 0.1, "mass_scale": 3.0, "com_s": None, "v_lift": 0.15}, seed=0)
    res = _weigh(env)
    assert res.kind == "lower_bound" and set(res.reading) == {"mu_hat", "m_lower"}
    f = math.exp(0.05 * res.sensor["eps"])
    for k in ("mu_hat", "m_lower"):
        assert res.reading[k] == pytest.approx(res.sensor["clean"][k] * f, rel=1e-12)


def test_push_break_force_carries_the_noise():
    env = _env()
    env.reset(BRICK, {"mu": 0.5, "mass_scale": 1.0, "com_s": None, "v_lift": 0.15}, seed=0)
    res = env.probe_push()
    assert res.kind == "slid" and set(res.reading) == {"f_break"}
    assert res.reading["f_break"] == pytest.approx(res.sensor["clean"]["f_break"] * math.exp(0.05 * res.sensor["eps"]),
                                                   rel=1e-12)


def test_bad_force_noise_is_rejected(monkeypatch):
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            MuJoCoLabEnv(force_noise=bad)
    monkeypatch.setenv("RPENT_LAB_FORCE_NOISE", "abc")
    with pytest.raises(ValueError):
        MuJoCoLabEnv()


def test_rows_keep_the_sensor_record(tmp_path):
    """A corpus row and a probe corpus row keep the sensor record: the noise level, the eps and the clean readings."""
    from robots.mujoco_lab.corpus import main as corpus_main
    from robots.mujoco_lab.probe_corpus import main as probe_main

    out = tmp_path / "c.jsonl"
    corpus_main(["--out", str(out), "--draws", "1", "--workers", "1", "--objects", LONG])
    row = json.loads(out.open().readline())
    assert row["sensor"]["force_noise"] == 0.05 and set(row["sensor"]["eps"]) == set(READOUTS)
    assert row["sensor"]["clean"]["fz_hold_n"] is None or row["readouts"]["fz_hold_n"] != row["sensor"]["clean"]["fz_hold_n"]
    pout = tmp_path / "p.jsonl"
    probe_main(["--out", str(pout), "--draws", "1", "--workers", "1", "--objects", BRICK])
    rows = [json.loads(line) for line in pout.open()]
    ran = [r for r in rows if "declined" not in r and r.get("probe") in ("weigh", "push")]
    assert ran and all(r["sensor"]["force_noise"] == 0.05 and "eps" in r["sensor"] for r in ran)


def test_planner_never_sees_the_sensor_record():
    """A guard (it holds before decision 1 too): the probe and attempt payloads of the tools carry the noisy readings,
    never the sensor record (the m2b bundle's models are local, so the test skips without them)."""
    from robots.mujoco_lab.calibrate import _primitives
    from robots.mujoco_lab.robot_spec import DATA_DIR
    from rpent.skills.pick_place.tools import _assess, _attempt, _run_probe

    bundle = str(DATA_DIR / "m2b")
    if not os.path.exists(os.path.join(bundle, "success_model.joblib")):
        pytest.skip("the m2b bundle's models are local (gitignored)")
    prims = _primitives(bundle, None, None)
    prims.rng = np.random.default_rng(0)
    prims.env.reset(BRICK, {"mu": 0.8, "mass_scale": 1.0, "com_s": None, "v_lift": 0.15}, seed=0)
    store: dict = {}
    err, a = _assess("object", "table_centre", "normal", prims, store)
    assert err is None
    weigh = next(o for o in a.options if o.id == "probe_weigh")
    p = _run_probe(weigh, "object", "table_centre", "normal", prims, store)
    best = next(o for o in a.options if o.kind == "commit")
    q = _attempt(best, "object", "table_centre", "normal", prims, store)
    hidden = {"sensor", "clean", "force_noise", "eps_force"}
    assert not hidden & set(p) and not hidden & set(p["reading"]) and set(p["reading"]) <= {"m_hat", "mu_hat", "m_lower"}
    assert not hidden & set(q) and not hidden & set(q["readouts"])
