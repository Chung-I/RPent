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
