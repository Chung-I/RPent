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

"""The loss-time hazard model: records, survival walk, release head."""

import numpy as np

from rpent.skills.pick_place.evaluate import split_objects
from rpent.skills.pick_place.hazard_model import HazardModel, hazard_records, train_hazard
from rpent.skills.pick_place.protocol import OUTCOMES
from tests.unit_tests.skills.synthetic import PLANNED_TEMPLATE, synthetic_rows


def _split(rows):
    sp = split_objects(rows, seed=0)
    return [r for r in rows if r["object"] in sp["train"]], [r for r in rows if r["object"] in sp["val"]]


def test_hazard_records_weights():
    row = synthetic_rows(1)[0]
    row["label"], row["loss_phase"] = "dropped_high", "lift"
    row["loss_step"] = 60 + 30 + 40                      # step 40 of the 67-step lift phase
    recs = hazard_records([row])
    by = {(r["phase"], r["event"]): r["weight"] for r in recs}
    assert by[("approach", 0)] == 60 and by[("close", 0)] == 30
    assert by[("lift", 0)] == 39 and by[("lift", 1)] == 1
    assert not any(r["phase"] in ("hold", "transport", "lower", "release") for r in recs)
    held = dict(synthetic_rows(1)[0], label="placed_stable", loss_phase=None, loss_step=None, loss_t=None)
    recs = hazard_records([held])
    assert {r["phase"] for r in recs} == {"approach", "close", "lift", "hold", "transport", "lower"}
    assert all(r["event"] == 0 for r in recs)
    assert hazard_records([dict(held, label="not_lifted")]) == []


def test_outcome_distribution_sums_to_one_and_has_height_bins(tmp_path):
    tr, va = _split(synthetic_rows(800))
    train_hazard(tr, va, str(tmp_path / "h.joblib"))
    hm = HazardModel.load(str(tmp_path / "h.joblib"))
    pred = hm.outcome_distribution(va[0]["features"], PLANNED_TEMPLATE)
    assert set(pred.p) == set(OUTCOMES) and abs(sum(pred.p.values()) - 1) < 1e-6
    assert pred.drop_height is not None and abs(sum(pred.drop_height.values()) - pred.p["dropped_low"] - pred.p["dropped_high"]) < 1e-6
    assert pred.expected_loss_t is None or pred.expected_loss_t >= 0


def test_slow_low_trajectory_has_less_dropped_high(tmp_path):
    tr, va = _split(synthetic_rows(800))
    train_hazard(tr, va, str(tmp_path / "h.joblib"))
    hm = HazardModel.load(str(tmp_path / "h.joblib"))
    f = dict(va[0]["features"], load_ratio=0.9)
    high = hm.outcome_distribution(f, PLANNED_TEMPLATE)
    low = [dict(s, z_start=min(s["z_start"], 0.05), z_end=min(s["z_end"], 0.05)) for s in PLANNED_TEMPLATE]
    low_pred = hm.outcome_distribution(f, low)
    assert low_pred.p["dropped_high"] < high.p["dropped_high"]
    assert abs((low_pred.p["dropped_high"] + low_pred.p["dropped_low"]) - (high.p["dropped_high"] + high.p["dropped_low"])) < 1e-6


def test_release_is_never_at_risk_and_no_loss_rows_give_survival_near_one(tmp_path):
    rows = [dict(r, label="placed_stable", loss_phase=None, loss_step=None, loss_t=None) for r in synthetic_rows(300)]
    tr, va = _split(rows)
    train_hazard(tr, va, str(tmp_path / "h.joblib"))
    hm = HazardModel.load(str(tmp_path / "h.joblib"))
    pred = hm.outcome_distribution(va[0]["features"], PLANNED_TEMPLATE)
    assert pred.p["dropped_low"] + pred.p["dropped_high"] < 0.05
    assert hm.loss_hazard(va[0]["features"], "release") == 0.0


def test_release_head_moves_mass_to_tipped(tmp_path):
    tr, va = _split(synthetic_rows(1500, noise=0.0))
    train_hazard(tr, va, str(tmp_path / "h.joblib"))
    hm = HazardModel.load(str(tmp_path / "h.joblib"))
    f = dict(va[0]["features"], load_ratio=0.2, lever=0.0, dest_opening_ratio=10.0)
    narrow = hm.outcome_distribution(dict(f, base_ratio=0.2), PLANNED_TEMPLATE)
    wide = hm.outcome_distribution(dict(f, base_ratio=1.2), PLANNED_TEMPLATE)
    assert narrow.p["tipped"] > wide.p["tipped"] and wide.p["placed_stable"] > narrow.p["placed_stable"]


def test_train_hazard_reports_both_modes(tmp_path):
    tr, va = _split(synthetic_rows(1000))
    m = train_hazard(tr, va, str(tmp_path / "h.joblib"))
    assert set(m) >= {"oracle", "belief", "kind", "held_out_objects"} and "loss_time_mae" in m["oracle"]
    assert m["belief"]["brier"] < m["belief"]["brier_const"]
