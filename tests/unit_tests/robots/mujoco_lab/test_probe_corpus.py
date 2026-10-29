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


"""The probe corpus: scene A pushes then weighs, scene B weighs alone, under the same hidden draw (needs the sim)."""

import json
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")

NEED = {"key", "object", "draw", "scene", "order", "probe", "s", "pose", "base", "force", "f_std", "com_obj_true", "truth",
        "kind", "reading", "displacement_m", "tilt_deg", "thrown", "seconds", "reset", "rev_steps", "deepest_mm", "artifact"}


def test_probe_corpus_rows(tmp_path):
    from robots.mujoco_lab.probe_corpus import main

    out = tmp_path / "p.jsonl"
    n = main(["--out", str(out), "--draws", "2", "--workers", "1", "--objects", "ycb__brick", "--seed", "7"])
    rows = [json.loads(line) for line in open(out)]
    assert n == len(rows) == 6
    assert {r["key"] for r in rows} == {f"ycb__brick__s7__d{i}__{s}" for i in (0, 1) for s in ("Apush", "Aweigh", "Bweigh")}
    assert all(NEED <= set(r) and "trace" not in r for r in rows)
    for i in (0, 1):
        by = {r["key"].rsplit("__", 1)[1]: r for r in rows if r["draw"] == i}
        assert (by["Apush"]["order"], by["Aweigh"]["order"], by["Bweigh"]["order"]) == (0, 1, 0)
        assert by["Apush"]["probe"] == "push" and by["Aweigh"]["probe"] == by["Bweigh"]["probe"] == "weigh"
        assert by["Aweigh"]["truth"] == by["Bweigh"]["truth"] and by["Aweigh"]["force"] == by["Bweigh"]["force"]
        if by["Apush"]["displacement_m"] > 0.005:
            assert by["Aweigh"]["pose"]["p"] != by["Bweigh"]["pose"]["p"]


def test_probe_corpus_resume_skips_done_draws(tmp_path):
    from robots.mujoco_lab.probe_corpus import main

    out = tmp_path / "p.jsonl"
    assert main(["--out", str(out), "--draws", "1", "--workers", "1", "--objects", "ycb__brick", "--seed", "7"]) == 3
    assert main(["--out", str(out), "--draws", "2", "--workers", "1", "--objects", "ycb__brick", "--seed", "7", "--resume"]) == 3
    keys = [json.loads(line)["key"] for line in open(out)]
    assert len(keys) == len(set(keys)) == 6


def test_the_probes_use_the_fitting_proposal_nearest_the_middle():
    """ceramic_mug d0: the s 0.3 and 0.5 pinches are 9.2 and 9.1 cm, over the 9.0 cm limit; the menu would
    probe at s 0.7 (6.8 cm), so the corpus does."""
    from robots.mujoco_lab.adapter import MuJoCoLabEnv
    from robots.mujoco_lab.probe_corpus import probe_rows

    rows = probe_rows(MuJoCoLabEnv(), "hot3d__ceramic_mug", 0, 7)
    assert {r["key"].rsplit("__", 1)[1] for r in rows} <= {"Apush", "Aweigh", "Bweigh"} and len(rows) >= 2
    assert all(r["s"] == 0.7 and r["base"]["width"] < 0.090 for r in rows)


def test_a_draw_with_no_fitting_proposal_is_declined():
    """macaroni_and_cheese: every pinch is 12.5 cm across the box. The skill declines before any probe, so the
    draw gets one declined row and runs no probe."""
    from robots.mujoco_lab.adapter import MuJoCoLabEnv
    from robots.mujoco_lab.probe_corpus import probe_rows

    env = MuJoCoLabEnv()

    def probe(*a, **k):
        raise AssertionError("no probe may run")
    env.probe_push = env.probe_weigh = probe
    rows = probe_rows(env, "hope__macaroni_and_cheese", 2, 7)
    assert len(rows) == 1 and rows[0]["key"] == "hope__macaroni_and_cheese__s7__d2" and rows[0]["draw"] == 2
    assert rows[0]["declined"] == "every proposal is wider than the gripper opening" and "probe" not in rows[0]


def test_probe_rows_carry_the_damage_and_a_squeeze_for_a_crushable_object():
    """Probe corpus v8: every row carries the draw's fill and fragility, the meter and the damage; a crushable object
    also gets a squeeze from a fresh reset at the cap of squeeze_cap on the prior (at most half the standard squeeze)."""
    from robots.mujoco_lab.adapter import MuJoCoLabEnv
    from robots.mujoco_lab.probe_corpus import probe_rows

    rows = probe_rows(MuJoCoLabEnv(), "ycb__jello", 0, 7)
    by = {r["key"].rsplit("__", 1)[1]: r for r in rows}
    assert set(by) <= {"Apush", "Aweigh", "Bweigh", "Csqueeze"} and {"Bweigh", "Csqueeze"} <= set(by)
    assert all({"fill", "theta_frag", "entry", "meter", "damage"} <= set(r) for r in rows if "declined" not in r)
    sq = by["Csqueeze"]
    assert sq["probe"] == "squeeze" and sq["scene"] == "C" and sq["kind"] in ("measured", "damaged")
    assert 0.0 < sq["force"] <= 0.5 * sq["f_std"] and sq["theta_frag"] == by["Bweigh"]["theta_frag"]
    assert sq["kind"] == "damaged" or "k_hat" in sq["reading"]


def test_m2_probe_rows_have_no_damage_and_no_squeeze():
    """Mode m2 is probe corpus v7's world: v8's draws, no liquid, no fragility and no squeeze scene."""
    from robots.mujoco_lab.adapter import MuJoCoLabEnv
    from robots.mujoco_lab.corpus import case_rng, draw_theta
    from robots.mujoco_lab.adapter import object_props
    from robots.mujoco_lab.probe_corpus import probe_rows

    rows = probe_rows(MuJoCoLabEnv(), "ycb__jello", 0, 7, mode="m2")
    assert {r["key"].rsplit("__", 1)[1] for r in rows} <= {"Apush", "Aweigh", "Bweigh"}
    assert all(r["fill"] == 0.0 and r["theta_frag"] is None and r.get("damage") is None for r in rows)
    assert rows[0]["theta"] == draw_theta(case_rng(7, "ycb__jello", 0), object_props("ycb__jello")["mass"])[0]


def test_probe_rows_set_the_slosh_scale_of_their_thresholds():
    from robots.mujoco_lab.adapter import load_sim
    from robots.mujoco_lab.fragility import THRESHOLDS_V0, Thresholds, load_classes
    from robots.mujoco_lab.probe_corpus import probe_rows

    class Stop(Exception):
        pass

    class Env:
        s = load_sim(None)

        def reset(self, *a, **kw):
            raise Stop

    th = Thresholds.from_json(THRESHOLDS_V0)
    th.slosh_zeta, th.slosh_scale = 0.02, {"carry": {"0.6": 0.5}}
    env = Env()
    with pytest.raises(Stop):
        probe_rows(env, "ycb__mug", 0, 7, classes=load_classes(), thresholds=th)
    assert env.slosh_zeta == 0.02 and env.slosh_scale == {"carry": {"0.6": 0.5}}
