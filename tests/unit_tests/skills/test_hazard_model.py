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


def test_score_hazard_scores_with_the_prior_fn(tmp_path):
    from rpent.skills.pick_place.hazard_model import _score_hazard, fit_hazard_model
    from tests.unit_tests.skills.synthetic import with_pose
    from tests.unit_tests.skills.test_success_model import _RecordingPrior

    rows = with_pose(synthetic_rows(400))
    tr, va = _split(rows)
    model = fit_hazard_model(tr)
    prior = _RecordingPrior()
    m = _score_hazard(model, va, tr, 0, 8, prior_fn=prior, features="true")
    assert len(prior.calls) == len(va) and m["belief"]["n"] == len(va)


def _reference_walk(model, features, planned):
    """The milestone 1 scalar outcome walk, kept here as the reference for the vectorised one."""
    from rpent.skills.pick_place.features import as_vector
    from rpent.skills.pick_place.hazard_model import AT_RISK, HIGH_DROP_M, RELEASE_CLASSES
    from rpent.skills.pick_place.protocol import DROP_HEIGHT_BINS

    p_nl = float(model.not_lifted.proba(as_vector(features)[None, :])[0]["1"])
    surv, low, high, t_mass, t = 1.0, 0.0, 0.0, 0.0, 0.0
    bins = {b: 0.0 for b in DROP_HEIGHT_BINS}
    for seg in planned:
        phase, n = seg["phase"], int(seg["n_steps"])
        if phase not in AT_RISK or n <= 0:
            t += n / model.hz
            continue
        h = model.loss_hazard(features, phase)
        ks = np.arange(1, n + 1)
        p_loss = surv * (1 - h) ** (ks - 1) * h
        z = seg["z_start"] + (seg["z_end"] - seg["z_start"]) * (ks / n)
        hm = (z >= HIGH_DROP_M) | (not seg.get("has_surface", True))
        high += float(p_loss[hm].sum())
        low += float(p_loss[~hm].sum())
        bins["0.00-0.05"] += float(p_loss[z < 0.05].sum())
        bins["0.05-0.10"] += float(p_loss[(z >= 0.05) & (z < 0.10)].sum())
        bins["0.10-0.20"] += float(p_loss[(z >= 0.10) & (z < 0.20)].sum())
        bins["0.20+"] += float(p_loss[z >= 0.20].sum())
        t_mass += float((p_loss * (t + ks / model.hz)).sum())
        surv *= (1 - h) ** n
        t += n / model.hz
    rel = model.release.proba(as_vector(features)[None, :])[0]
    p = {k: 0.0 for k in OUTCOMES}
    p.update(not_lifted=p_nl, dropped_low=(1 - p_nl) * low, dropped_high=(1 - p_nl) * high)
    for c in RELEASE_CLASSES:
        p[c] = (1 - p_nl) * surv * rel.get(c, 0.0)
    total = sum(p.values())
    return ({k: v / total for k, v in p.items()}, {b: (1 - p_nl) * v / total for b, v in bins.items()},
            t_mass / (low + high) if low + high > 1e-9 else None)


def test_predict_matrix_matches_the_scalar_walk(tmp_path):
    from rpent.skills.pick_place.features import as_vector
    from rpent.skills.pick_place.hazard_model import fit_hazard_model

    rows = synthetic_rows(400)
    model = fit_hazard_model(rows)
    planned = rows[0]["planned"]
    feats = [r["features"] for r in rows[:12]]
    P, DH, LT = model.predict_matrix(np.array([as_vector(f) for f in feats]), planned)
    for i, f in enumerate(feats):
        p, dh, lt = _reference_walk(model, f, planned)
        assert np.allclose(P[i], [p[k] for k in OUTCOMES], atol=1e-12)
        assert np.allclose(DH[i], list(dh.values()), atol=1e-12)
        assert (lt is None and not np.isfinite(LT[i])) or abs(LT[i] - lt) < 1e-9


def test_hazard_model_supports_motion_labels_only_and_stores_names():
    from rpent.skills.pick_place.features import FEATURES_M2, FEATURES_M2B
    from rpent.skills.pick_place.hazard_model import fit_hazard_model

    assert HazardModel.supports(["placed_stable", "tipped", "not_lifted"])
    assert not HazardModel.supports(["placed_stable", "crushed"])
    rows = synthetic_rows(300)
    assert fit_hazard_model(rows).feature_names == FEATURES_M2
    m = fit_hazard_model(rows, names=FEATURES_M2B)
    assert m.feature_names == FEATURES_M2B
    P, _, _ = m.predict_matrix(np.zeros((3, len(FEATURES_M2B))), PLANNED_TEMPLATE)
    assert P.shape == (3, len(OUTCOMES)) and np.allclose(P.sum(axis=1), 1.0)
