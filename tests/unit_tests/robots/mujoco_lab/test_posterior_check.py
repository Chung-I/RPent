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


"""The posterior check: prior draws, one update with each row's readouts, 80 % interval coverage
(needs the sim for the corpus rows)."""

import json
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")


@pytest.fixture(scope="module")
def tiny_bundle(tmp_path_factory):
    from robots.mujoco_lab.corpus import main as corpus_main
    from robots.mujoco_lab.train_models import main as train_main

    d = tmp_path_factory.mktemp("pc")
    corpus = d / "c.jsonl"
    assert corpus_main(["--out", str(corpus), "--draws", "8", "--workers", "1", "--objects", "ycb__banana,ycb__brick,ycb__mug"]) == 24
    train_main(["--corpus", str(corpus), "--out-dir", str(d / "m2"), "--n-belief", "4"])
    return d / "m2"


def test_train_models_writes_the_readout_model(tiny_bundle):
    man = json.load(open(tiny_bundle / "manifest.json"))
    assert man["readout_model"] == "readout_model.json" and (tiny_bundle / man["readout_model"]).exists()
    metrics = json.load(open(tiny_bundle / man["metrics"]))
    assert "hold" in metrics["readout"]["terms"] and "side_error" in metrics["readout"]


def test_posterior_check_reports_coverage(tiny_bundle, tmp_path):
    from robots.mujoco_lab.posterior_check import main
    from robots.mujoco_lab.robot_spec import build_lab

    out = main(["--bundle", str(tiny_bundle), "--split", "val", "--n-particles", "200", "--temper-grid", "1,2",
                "--choose-on", "val", "--out", str(tmp_path / "pc.json")])
    assert {"mass_cov80", "mu_cov80", "com_cov80", "by_label", "temper", "grid", "feasible", "prior_cov80"} <= set(out)
    assert out["n"] == 8 and all(0.0 <= out[k] <= 1.0 for k in ("mass_cov80", "mu_cov80", "com_cov80"))
    assert sum(v["n"] for v in out["by_label"].values()) == 8 and set(out["grid"]) == {"1.0", "2.0"}
    assert json.load(open(tmp_path / "pc.json"))["n"] == 8
    man = json.load(open(tiny_bundle / "manifest.json"))
    assert man["tempers"]["readout"] == out["temper"]
    prims = build_lab(bundle=str(tiny_bundle))
    assert prims.readout_model is not None and prims.readout_model.temper == out["temper"]


def test_rotation_threshold_matches_the_readouts():
    from robots.mujoco_lab.readouts import ROT_SIDE_DEG
    from rpent.skills.pick_place.likelihood import ROT_DEG

    assert ROT_DEG == ROT_SIDE_DEG
