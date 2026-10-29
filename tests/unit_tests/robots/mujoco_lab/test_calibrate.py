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

"""End-to-end calibration check: assess, execute, compare the declared probability with the outcome."""

import json
import os

import pytest

SIM = os.environ.get("RPENT_MUJOCO_SIM_DIR")
METRICS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "..", "..", "robots", "mujoco_lab", "data", "success_model_metrics.json")
METRICS = os.path.normpath(METRICS)
pytestmark = pytest.mark.skipif(not (SIM and os.path.exists(METRICS)), reason="sim dir or trained model missing")

KEYS = {"n", "n_executed", "recommendation_decline_rate", "abstained", "ece", "ece_ci95_object_bootstrap", "brier", "brier_const", "by_subgroup",
        "by_destination", "by_option", "rows", "kind", "objects"}


def test_calibrate_two_objects_one_repeat(tmp_path):
    from robots.mujoco_lab.calibrate import main

    out = tmp_path / "cal.json"
    res = main(["--metrics", METRICS, "--repeats", "1", "--out", str(out), "--objects", "ycb__banana,ycb__brick", "--seed", "1"])
    assert set(res) >= KEYS and res["n"] == 2 and res["n_executed"] == 2
    saved = json.load(open(out))
    assert saved["n"] == 2 and len(saved["rows"]) == 2
    row = saved["rows"][0]
    assert {"object", "destination", "option", "declared", "outcome", "placed_at_requested", "truth"} <= set(row)
    assert set(row["declared"]["p"]) == {"placed_stable", "tipped", "missed", "dropped_low", "dropped_high", "rolled_off", "not_lifted"}


def test_follow_recommendation_can_abstain(tmp_path):
    from robots.mujoco_lab.calibrate import main

    res = main(["--metrics", METRICS, "--repeats", "1", "--out", str(tmp_path / "c.json"), "--objects", "ycb__banana", "--seed", "2", "--follow-recommendation", "--threshold", "1.01"])
    assert res["n"] == 1 and res["n_executed"] == 0 and res["abstained"] == 1 and res["recommendation_decline_rate"] == 1.0


def test_threshold_override_does_not_leak(tmp_path, monkeypatch):
    """--threshold goes through the primitives; the module default stays as it was for later callers."""
    from robots.mujoco_lab.calibrate import main
    from rpent.skills.pick_place import menu

    monkeypatch.setattr(menu, "DEFAULT_THRESHOLD", 0.7)      # a known start, whatever ran before
    main(["--metrics", METRICS, "--repeats", "1", "--out", str(tmp_path / "c.json"), "--objects", "ycb__banana", "--seed", "2",
          "--follow-recommendation", "--threshold", "1.01"])
    assert menu.DEFAULT_THRESHOLD == 0.7


def test_threshold_reaches_the_menu(tmp_path):
    """--threshold 0.0 reaches the menu: no case abstains on a decline."""
    from robots.mujoco_lab.calibrate import main

    res = main(["--metrics", METRICS, "--repeats", "1", "--out", str(tmp_path / "c.json"), "--objects", "ycb__banana", "--seed", "2",
                "--follow-recommendation", "--threshold", "0.0"])
    assert res["n"] == 1 and res["n_executed"] == 1 and res["abstained"] == 0 and res["recommendation_decline_rate"] == 0.0


# ---------------------------------------------------------------- cross-fitted and deployment calibration (milestone 2, task 8)
import numpy as np  # noqa: E402

from rpent.skills.pick_place.protocol import OUTCOMES  # noqa: E402

BUNDLE = os.path.normpath(os.path.join(os.path.dirname(METRICS), "m2"))
needs_bundle = pytest.mark.skipif(not os.path.exists(os.path.join(BUNDLE, "success_model.joblib")), reason="the m2 bundle is not built")


def _stub_fold(root, k, train, val, test):
    """A fold directory: the deployed bundle's model files, with a metrics split that holds out ``test``."""
    d = root / f"fold_{k}"
    d.mkdir()
    man = json.load(open(os.path.join(BUNDLE, "manifest.json")))
    for f in os.listdir(BUNDLE):
        if f.endswith(".joblib") or f == man.get("readout_model"):
            os.symlink(os.path.join(BUNDLE, f), d / f)
    metrics = json.load(open(os.path.join(BUNDLE, man["metrics"])))
    metrics["split"] = {"train": train, "val": val, "test": test}
    (d / man["metrics"]).write_text(json.dumps(metrics))
    (d / "manifest.json").write_text(json.dumps(dict(man, fold=k)))
    return d


@needs_bundle
def test_crossfit_uses_the_fold_that_excludes_the_object(tmp_path):
    from robots.mujoco_lab.calibrate import main

    cf = tmp_path / "folds"
    cf.mkdir()
    _stub_fold(cf, 0, ["ycb__brick", "ycb__mug"], ["ycb__clamp"], ["ycb__banana"])
    _stub_fold(cf, 1, ["ycb__banana", "ycb__mug"], ["ycb__clamp"], ["ycb__brick"])
    (cf / "folds.json").write_text(json.dumps({"ycb__banana": 0, "ycb__brick": 1}))
    args = ["--crossfit", str(cf), "--protocol", "commit", "--repeats", "1", "--seed", "1", "--coverage-sims", "0",
            "--out", str(tmp_path / "c.json")]
    res = main(args)
    assert res["objects"] == ["ycb__banana", "ycb__brick"] and res["n"] == 2 and res["errors"] == 0
    assert {r["object"]: r["fold"] for r in res["rows"]} == {"ycb__banana": 0, "ycb__brick": 1}
    assert all(r["bundle"] == str(cf / f"fold_{r['fold']}") for r in res["rows"])
    (cf / "folds.json").write_text(json.dumps({"ycb__banana": 1, "ycb__brick": 0}))      # each object in the fold that trained on it
    with pytest.raises(SystemExit, match="trained on"):
        main(args)


@needs_bundle
def test_protocols_are_paired_by_case(tmp_path):
    from robots.mujoco_lab.calibrate import main

    common = ["--bundle", BUNDLE, "--objects", "ycb__banana,ycb__brick", "--repeats", "2", "--seed", "3", "--coverage-sims", "0"]
    a = main(common + ["--protocol", "commit", "--out", str(tmp_path / "a.json")])
    b = main(common + ["--protocol", "repair", "--out", str(tmp_path / "b.json")])

    def key(r):
        return r["object"], r["idx"], r["destination"], json.dumps(r["theta"], sort_keys=True), r["com_end"]

    assert len(a["rows"]) == 4 and [key(r) for r in a["rows"]] == [key(r) for r in b["rows"]]
    assert len({json.dumps(r["theta"], sort_keys=True) for r in a["rows"]}) == 4             # one draw per case
    assert all(r["executed"] for r in a["rows"])                                            # the best row runs whatever the recommendation
    assert all(set(r["reset"]) == {"settle_s", "at_rest", "zone"} for r in a["rows"])
    first = [(r["attempts"][0]["option"], r["attempts"][0]["outcome"]) for r in a["rows"]]
    assert first == [(r["attempts"][0]["option"], r["attempts"][0]["outcome"]) for r in b["rows"]]
    assert a["protocol"] == "commit" and b["protocol"] == "repair"
    assert b["second_attempt"]["n"] == sum(len(r["attempts"]) == 2 for r in b["rows"])
    cmp = main(["--compare", str(tmp_path / "b.json"), str(tmp_path / "a.json"), "--out", str(tmp_path / "cmp.json")])
    assert cmp["n_paired"] == 4
    assert cmp["realized_cost"]["diff"] == pytest.approx(b["realized_cost"]["mean"] - a["realized_cost"]["mean"])


def test_every_case_is_kept_for_the_cost():
    """A thrown probe and an unreachable object after a failed attempt end their cases with their cost."""
    from robots.mujoco_lab.calibrate import realized_cost, run_protocol
    from rpent.skills.pick_place.costs import COST, TIME_COST_PER_S
    from rpent.skills.pick_place.probes import PROBE_SECONDS
    from tests.unit_tests.skills.probe_stubs import PerfectPush
    from tests.unit_tests.skills.test_tools import _FallEnv, _ProbeEnv, _prims, _probe_prims, _push

    env = _ProbeEnv(probes=[_push(thrown=True)])
    thrown = run_protocol(_probe_prims(env, push=PerfectPush()), "probe_first", "table_centre", np.random.default_rng(0))
    assert thrown["ended"] == "thrown" and thrown["attempts"] == [] and env.calls == []
    assert [p["probe"] for p in thrown["probes"]] == ["push"] and thrown["probes"][0]["thrown"] is True
    assert thrown["cost"] == pytest.approx(TIME_COST_PER_S * PROBE_SECONDS["push"] + COST["dropped_high"])
    fall = run_protocol(_prims(env=_FallEnv()), "repair", "bin", np.random.default_rng(0))
    assert fall["ended"] == "unreachable" and [t["outcome"] for t in fall["attempts"]] == ["dropped_high"]
    assert fall["cost"] == pytest.approx(COST["dropped_high"] + TIME_COST_PER_S * fall["attempts"][0]["cost_s"])
    rows = [dict(thrown, object="a", idx=0), dict(fall, object="b", idx=0), {"object": "c", "idx": 0, "error": "boom"}]
    rc = realized_cost(rows, np.random.default_rng(0))
    assert rc["n"] == 2 and rc["n_errors"] == 1 and rc["mean"] == pytest.approx((thrown["cost"] + fall["cost"]) / 2)
    assert rc["by_end"]["thrown"]["n"] == 1 and rc["by_end"]["unreachable"]["n"] == 1


def test_probe_first_runs_the_recommended_probe_then_the_best_row():
    from robots.mujoco_lab.calibrate import run_protocol
    from rpent.skills.pick_place.costs import TIME_COST_PER_S
    from rpent.skills.pick_place.probes import PROBE_SECONDS
    from tests.unit_tests.skills.probe_stubs import FlatWeigh, PerfectPush
    from tests.unit_tests.skills.test_tools import _ProbeEnv, _probe_prims, _push

    env = _ProbeEnv(results=[("placed_stable", {})], probes=[_push(high=1.0)])
    case = run_protocol(_probe_prims(env, push=PerfectPush(), weigh=FlatWeigh()), "probe_first", "table_centre", np.random.default_rng(0))
    assert [p["probe"] for p in case["probes"]] == ["push"] and case["probes"][0]["declared"]["p_kind"]["slid"] == pytest.approx(1.0)
    assert [(t["option"], t["outcome"]) for t in case["attempts"]] == [("firm_grip", "placed_stable")] and case["ended"] == "placed"
    assert case["cost"] == pytest.approx(TIME_COST_PER_S * (PROBE_SECONDS["push"] + case["attempts"][0]["cost_s"]))


def test_each_case_keeps_the_guard_check_of_its_first_assessment():
    """The commit protocol runs the best row even after a guard decline; the case keeps the check, and the
    summary counts the cases the guard checked and flagged."""
    from robots.mujoco_lab.calibrate import guard_block, run_protocol
    from tests.unit_tests.skills.test_tools import FlagGuard, _prims

    prims = _prims()
    prims.guard = FlagGuard(True)
    case = run_protocol(prims, "commit", "table_centre", np.random.default_rng(0))
    assert case["guard"]["flag"] is True and case["recommendation"] == "decline" and len(case["attempts"]) == 1
    prims.guard = FlagGuard(False)
    calm = run_protocol(prims, "commit", "table_centre", np.random.default_rng(0))
    plain = run_protocol(_prims(), "commit", "table_centre", np.random.default_rng(0))
    assert calm["guard"]["flag"] is False and plain["guard"] is None
    assert guard_block([case, calm, plain, {"object": "c", "idx": 0, "error": "boom"}]) == {"n_checked": 2, "n_flagged": 1}
    assert guard_block([plain]) is None


def test_random_row_draws_among_the_same_destination_rows():
    from robots.mujoco_lab.calibrate import run_protocol
    from rpent.skills.pick_place.tools import assess_pick
    from tests.unit_tests.skills.test_tools import _prims

    menu = assess_pick(object="mug", destination="table_centre", primitives=_prims(), store={}).data["options"]
    allowed = {o["id"] for o in menu if o["kind"] in ("commit", "lower_risk") and o["action"]["destination"] == "table_centre"}
    seen = {run_protocol(_prims(), "random_row", "table_centre", np.random.default_rng(s))["attempts"][0]["option"] for s in range(12)}
    assert seen <= allowed and len(seen) > 1


def _att(obj, fold, p, label):
    P = {k: 0.0 for k in OUTCOMES}
    P.update(placed_stable=p, not_lifted=0.6 * (1 - p), dropped_low=0.4 * (1 - p))
    return {"object": obj, "fold": fold, "declared": {"p": P}, "label": label}


def test_nested_map_is_fitted_without_the_held_out_fold():
    from robots.mujoco_lab.calibrate import nested_isotonic

    rows = [_att(f"a{i}", 0, 0.8, "not_lifted") for i in range(20)] + [_att(f"b{i}", 1, 0.8, "placed_stable") for i in range(20)]
    preds = nested_isotonic(rows)
    assert all(x.p["placed_stable"] == pytest.approx(1.0) for x in preds[:20])        # fold 0: the map of fold 1 (always placed)
    assert all(x.p["placed_stable"] == pytest.approx(0.0) for x in preds[20:])
    assert preds[20].p["not_lifted"] == pytest.approx(0.6) and preds[20].p["dropped_low"] == pytest.approx(0.4)   # rescaled to 1 - p'
    assert all(sum(x.p.values()) == pytest.approx(1.0) for x in preds)


def test_compare_resamples_objects_jointly():
    from robots.mujoco_lab.calibrate import compare

    def case(obj, idx, cost, label):
        return {"object": obj, "idx": idx, "cost": cost, "ended": "placed", "executed": True,
                "attempts": [dict(_att(obj, 0, 0.5, label), cost=cost)]}

    A = [case(f"o{i}", j, 1.0 + 0.1 * i, "placed_stable" if j else "not_lifted") for i in range(10) for j in range(3)]
    B = [case(r["object"], r["idx"], r["cost"] - 0.5, "not_lifted") for r in A if (r["object"], r["idx"]) != ("o9", 2)]
    out = compare({"protocol": "probe_first", "rows": A}, {"protocol": "commit", "rows": B}, np.random.default_rng(0))
    assert out["n_paired"] == 29 and out["unpaired"] == {"a_only": 1, "b_only": 0} and out["n_objects"] == 10
    assert out["realized_cost"]["diff"] == pytest.approx(0.5)
    assert out["realized_cost"]["ci95_object_bootstrap"] == pytest.approx([0.5, 0.5])
    assert {"a", "b", "diff", "ci95_object_bootstrap"} <= set(out["ece_first_attempt"])


def test_probe_all_runs_both_probes_then_the_best_row():
    """The diagnostic protocol: push, then weigh, whatever the recommendation, then the best row."""
    from robots.mujoco_lab.calibrate import run_protocol
    from rpent.skills.pick_place.costs import TIME_COST_PER_S
    from rpent.skills.pick_place.probes import PROBE_SECONDS, ProbeResult
    from tests.unit_tests.skills.probe_stubs import FlatWeigh, PerfectPush
    from tests.unit_tests.skills.test_tools import _ProbeEnv, _probe_prims, _push

    weigh = ProbeResult(probe="weigh", kind="measured", reading={"m_hat": 0.3}, displacement_m=0.0, tilt_deg=0.0, thrown=False, seconds=4.0)
    env = _ProbeEnv(results=[("placed_stable", {})], probes=[_push(high=1.0), weigh])
    case = run_protocol(_probe_prims(env, push=PerfectPush(), weigh=FlatWeigh()), "probe_all", "table_centre", np.random.default_rng(0))
    assert env.probe_calls == ["push", "weigh"] and [p["kind"] for p in case["probes"]] == ["slid", "measured"]
    assert len(case["attempts"]) == 1 and case["ended"] == "placed"
    assert case["cost"] == pytest.approx(TIME_COST_PER_S * (PROBE_SECONDS["push"] + PROBE_SECONDS["weigh"] + case["attempts"][0]["cost_s"]))
    env = _ProbeEnv(probes=[_push(thrown=True)])
    thrown = run_protocol(_probe_prims(env, push=PerfectPush(), weigh=FlatWeigh()), "probe_all", "table_centre", np.random.default_rng(0))
    assert env.probe_calls == ["push"] and thrown["ended"] == "thrown" and thrown["attempts"] == []


def test_the_class_rates_come_from_trainable_rows_only(tmp_path):
    from robots.mujoco_lab.calibrate import _corpus_labels

    path = tmp_path / "c.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in ({"object": "a", "error": "boom"},
                                                       {"object": "b", "declined": "wider than the gripper opening"},
                                                       {"object": "c", "label": "tipped"})) + "\n")
    assert _corpus_labels(str(path)) == [("c", "tipped")]


def test_an_artifact_case_is_left_out_of_the_metrics_and_counted():
    """A case with a flagged attempt or probe, or whose reset did not reach rest, does not count."""
    from robots.mujoco_lab.calibrate import case_artifact, compare, realized_cost, run_protocol, summarize
    from tests.unit_tests.skills.test_tools import FakeEnv, _prims

    env = FakeEnv()
    plain = env.execute

    def flagged(*a, **k):
        r = plain(*a, **k)
        r.artifact = True
        return r

    feats = {"features_fn": lambda row: {"lever": 0.5}}
    clean = run_protocol(_prims(env=FakeEnv()), "commit", "table_centre", np.random.default_rng(0), **feats)
    env.execute = flagged
    bad = run_protocol(_prims(env=env), "commit", "table_centre", np.random.default_rng(0), **feats)
    assert clean["artifact"] is False and bad["artifact"] is True and bad["attempts"][0]["artifact"] is True
    assert case_artifact(clean, {"at_rest": False}) and not case_artifact(clean, {"at_rest": True}) and case_artifact(bad, {})
    ctx = {"idx": 0, "destination": "table_centre", "truth": {"mass": 0.3, "mu": 0.5, "com_s": 0.5}}
    rows = [dict(clean, object="a", **ctx), dict(bad, object="b", **ctx), {"object": "c", "idx": 0, "error": "boom"}]
    rc = realized_cost(rows, np.random.default_rng(0))
    assert rc["n"] == 1 and rc["n_errors"] == 1 and rc["n_artifacts"] == 1 and rc["mean"] == pytest.approx(clean["cost"])
    s = summarize(rows, "commit", np.random.default_rng(0))
    assert s["n"] == 3 and s["n_executed"] == 1 and s["errors"] == 1 and s["artifacts"] == 1
    c = compare({"rows": rows}, {"rows": rows[:1] + [dict(bad, object="b", **ctx, artifact=False)]}, np.random.default_rng(0))
    assert c["n_paired"] == 1 and c["artifacts"] == {"a": 1, "b": 0}
