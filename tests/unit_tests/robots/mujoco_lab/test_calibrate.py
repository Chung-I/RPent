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
    assert set(row["declared"]["p"]) == {"placed_stable", "tipped", "missed", "dropped_low", "dropped_high", "rolled_off", "not_lifted",
                                         "crushed", "bruised", "broken", "spilled"}
    assert all(row["declared"]["p"][k] == 0.0 for k in ("crushed", "bruised", "broken", "spilled"))     # m2: motion classes only


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
        if f.endswith(".joblib") or f in (man.get("readout_model"), man.get("guard")):
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


# ---------------------------------------------------------------- the cost rule and the policy protocol (milestone 2b)
from types import SimpleNamespace  # noqa: E402


def _row(rid="commit_std", cost_s=8.0):
    return {"id": rid, "kind": "commit", "action": {"destination": "table_centre", "s": 0.5, "force": 20.0}, "p": {"placed_stable": 0.9},
            "p_drop_height": None, "expected_cost": 0.6, "cost_s": cost_s}


def _assessment(rec, probe=False):
    opts = [_row()] + ([{"id": "probe_push", "kind": "probe", "action": {"probe": "push"}}] if probe else [])
    return {"assessed": True, "recommendation": rec, "best_option": "commit_std",
            "recommended_option": "probe_push" if rec == "probe" else ("commit_std" if rec == "commit" else None),
            "guard": None, "options": opts}


class _Script:
    """Scripted assess_pick and pick_and_place: assessments in order, and the attempt outcomes in order."""

    def __init__(self, assessments, outcomes):
        self.assessments, self.outcomes, self.calls = list(assessments), list(outcomes), []

    def assess(self, **kw):
        return SimpleNamespace(data=self.assessments.pop(0))

    def pick(self, **kw):
        self.calls.append(kw.get("option_id"))
        nxt = self.assessments.pop(0) if self.assessments else _assessment("decline")
        if str(kw.get("option_id")).startswith("probe_"):
            return SimpleNamespace(data={"probed": "push", "kind": "slid", "reading": {"f_break": 1.0}, "thrown": False,
                                         "assessment": nxt})
        return SimpleNamespace(data={"option": kw.get("option_id"), "outcome": self.outcomes.pop(0), "placed_at_requested": True,
                                     "diagnosis": None, "assessment": nxt})


def _scripted(monkeypatch, assessments, outcomes):
    from robots.mujoco_lab import calibrate

    sc = _Script(assessments, outcomes)
    monkeypatch.setattr(calibrate, "assess_pick", sc.assess)
    monkeypatch.setattr(calibrate, "pick_and_place", sc.pick)
    return sc


def test_policy_protocol_abstains_after_a_probe(monkeypatch):
    from robots.mujoco_lab.calibrate import probe_cost, run_protocol

    sc = _scripted(monkeypatch, [_assessment("probe", probe=True), _assessment("decline")], [])
    case = run_protocol(SimpleNamespace(abstain_cost=1.0), "policy", "table_centre", np.random.default_rng(0))
    assert sc.calls == ["probe_push"] and case["ended"] == "abstained" and case["attempts"] == []
    assert case["cost"] == pytest.approx(probe_cost("push", "slid", False, "normal") + 1.0)


def test_policy_protocol_retries_after_a_recoverable_failure(monkeypatch):
    from robots.mujoco_lab.calibrate import attempt_cost, run_protocol

    sc = _scripted(monkeypatch, [_assessment("commit"), _assessment("commit"), _assessment("commit")], ["missed", "placed_stable"])
    case = run_protocol(SimpleNamespace(abstain_cost=1.0), "policy", "table_centre", np.random.default_rng(0))
    assert sc.calls == ["commit_std", "commit_std"] and case["ended"] == "placed"
    assert case["cost"] == pytest.approx(attempt_cost("missed", 8.0) + attempt_cost("placed_stable", 8.0))


def test_policy_protocol_abstains_at_once_on_a_decline(monkeypatch):
    from robots.mujoco_lab.calibrate import run_protocol

    sc = _scripted(monkeypatch, [_assessment("decline")], [])
    case = run_protocol(SimpleNamespace(abstain_cost=1.0), "policy", "table_centre", np.random.default_rng(0))
    assert sc.calls == [] and case["ended"] == "abstained" and case["cost"] == pytest.approx(1.0)


def test_case_cost_charges_the_abstain_cost_when_unfinished(monkeypatch):
    from robots.mujoco_lab.calibrate import attempt_cost, run_protocol

    _scripted(monkeypatch, [_assessment("commit")], ["missed"])
    case = run_protocol(SimpleNamespace(abstain_cost=1.0), "commit", "table_centre", np.random.default_rng(0))
    assert case["ended"] == "failed" and case["cost"] == pytest.approx(attempt_cost("missed", 8.0) + 1.0)
    _scripted(monkeypatch, [_assessment("commit")], ["missed"])
    case = run_protocol(SimpleNamespace(abstain_cost=None), "commit", "table_centre", np.random.default_rng(0))
    assert case["cost"] == pytest.approx(attempt_cost("missed", 8.0))
    _scripted(monkeypatch, [_assessment("commit")], ["dropped_high"])
    case = run_protocol(SimpleNamespace(abstain_cost=1.0), "commit", "table_centre", np.random.default_rng(0), care=10.0)
    assert case["cost"] == pytest.approx(5.0 * 10.0 + 0.4 + 1.0)


def test_care_and_abstain_cost_reach_the_workers(tmp_path):
    from robots.mujoco_lab.calibrate import attempt_cost, main

    out = tmp_path / "c.json"
    main(["--metrics", METRICS, "--repeats", "1", "--out", str(out), "--objects", "ycb__banana", "--seed", "2",
          "--care", "10", "--abstain-cost", "2"])
    saved = json.load(open(out))
    assert saved["care_scale"] == 10.0 and saved["abstain_cost"] == 2.0
    row = saved["rows"][0]
    t = row["attempts"][0]
    assert t["cost"] == pytest.approx(attempt_cost(t["outcome"], t["cost_s"], 10.0))
    assert row["cost"] == pytest.approx(t["cost"] + (0.0 if row["ended"] == "placed" else 2.0))


def test_a_bundle_decision_sets_the_abstain_cost(tmp_path):
    """load_bundle reads the manifest's decision (None when absent, as in m2), and the factories put it on the primitives."""
    import os

    from robots.mujoco_lab.calibrate import _primitives
    from robots.mujoco_lab.robot_spec import DEFAULT_BUNDLE, build_lab, load_bundle

    assert load_bundle(str(DEFAULT_BUNDLE))["abstain_cost"] is None
    b = tmp_path / "b"
    b.mkdir()
    for f in os.listdir(DEFAULT_BUNDLE):
        if f != "manifest.json":
            (b / f).symlink_to(DEFAULT_BUNDLE / f)
    man = json.load(open(DEFAULT_BUNDLE / "manifest.json"))
    json.dump(dict(man, decision={"abstain_cost": 1.0}), open(b / "manifest.json", "w"))
    assert load_bundle(str(b))["abstain_cost"] == 1.0
    assert build_lab(bundle=str(b)).abstain_cost == 1.0
    assert _primitives(str(b), None, None).abstain_cost == 1.0
    assert build_lab(bundle=str(DEFAULT_BUNDLE)).abstain_cost is None


def test_compare_refuses_runs_with_different_costs():
    from robots.mujoco_lab.calibrate import compare

    row = {"object": "o", "idx": 0, "cost": 1.0, "attempts": []}
    a = {"protocol": "policy", "care_scale": 1.0, "abstain_cost": 1.0, "rows": [row]}
    with pytest.raises(ValueError, match="different costs"):
        compare(a, dict(a, abstain_cost=None), np.random.default_rng(0))
    with pytest.raises(ValueError, match="different costs"):
        compare(a, dict(a, care_scale=10.0), np.random.default_rng(0))
    assert compare(a, dict(a, protocol="commit"), np.random.default_rng(0))["n_paired"] == 1


def test_policy_protocol_without_probes_left_follows_the_cost_rule(monkeypatch):
    """Past max_probes a probe recommendation falls back to the menu's rule on the best row alone."""
    from robots.mujoco_lab.calibrate import run_protocol

    cheap = _assessment("probe", probe=True)
    sc = _scripted(monkeypatch, [cheap, cheap, _assessment("commit")], ["placed_stable"])
    case = run_protocol(SimpleNamespace(abstain_cost=1.0), "policy", "table_centre", np.random.default_rng(0), max_probes=1)
    assert sc.calls == ["probe_push", "commit_std"] and case["ended"] == "placed"
    dear = dict(cheap, options=[dict(_row(), expected_cost=1.5)] + cheap["options"][1:])
    sc = _scripted(monkeypatch, [cheap, dear], [])
    case = run_protocol(SimpleNamespace(abstain_cost=1.0), "policy", "table_centre", np.random.default_rng(0), max_probes=1)
    assert sc.calls == ["probe_push"] and case["ended"] == "abstained"


def test_policy_protocol_ends_after_an_irreversible_outcome(monkeypatch):
    from robots.mujoco_lab.calibrate import attempt_cost, run_protocol

    sc = _scripted(monkeypatch, [_assessment("commit"), _assessment("commit")], ["dropped_high"])
    case = run_protocol(SimpleNamespace(abstain_cost=1.0), "policy", "table_centre", np.random.default_rng(0))
    assert sc.calls == ["commit_std"] and case["ended"] == "failed"
    assert case["cost"] == pytest.approx(attempt_cost("dropped_high", 8.0) + 1.0)


def test_policy_protocol_stops_at_three_attempts(monkeypatch):
    from robots.mujoco_lab.calibrate import POLICY_ATTEMPTS, run_protocol

    sc = _scripted(monkeypatch, [_assessment("commit")] * 4, ["missed"] * 3)
    case = run_protocol(SimpleNamespace(abstain_cost=1.0), "policy", "table_centre", np.random.default_rng(0))
    assert POLICY_ATTEMPTS == 3 and len(sc.calls) == 3 and case["ended"] == "failed" and case["abstain_part"] == 1.0


def test_case_draw_draws_fill_and_fragility():
    """An m2b case draws the fill and the fragility with the corpus rule from the corpus's damage stream, so it hides
    what the corpus row of that (seed, object, idx) hides; an m2 case draws neither."""
    from robots.mujoco_lab.adapter import object_props
    from robots.mujoco_lab.calibrate import case_draw
    from robots.mujoco_lab.corpus import case_rng, damage_rng, draw_damage, draw_theta
    from robots.mujoco_lab.fragility import THRESHOLDS_V0, Thresholds, entry_of, load_classes

    classes, th = load_classes(), Thresholds.from_json(THRESHOLDS_V0)
    m = object_props("ycb__mug")["mass"]
    fills = []
    for idx in range(40):
        theta, com_end, dest, hidden = case_draw(5, "ycb__mug", idx, m, fragility=(classes, th))
        d = draw_damage(damage_rng(5, "ycb__mug", idx), entry_of("ycb__mug", classes)[0], th, 1.0)
        assert hidden == {"fill": d["fill"], "frag": d["theta_frag"], "spill_angle_deg": d["spill_angle_deg"]}
        if d["fill"] > 0:
            assert theta["com_s"] is None and theta["mass_scale"] == d["mass_scale"]
        fills.append(hidden["fill"])
    assert 0 < sum(f == 0 for f in fills) < 40 and all(f == 0 or 0.3 <= f <= 0.9 for f in fills)
    theta, com_end, dest, hidden = case_draw(5, "ycb__mug", 0, m)
    assert hidden == {"fill": 0.0, "frag": None, "spill_angle_deg": None}
    assert (theta, com_end) == draw_theta(case_rng(5, "ycb__mug", 0), m)


# ---------------------------------------------------------------- milestone 2b: damage in every probe (task 7)
def test_probe_cost_charges_damage():
    """A probe that damages or throws the object costs its time and 5 kappa once; the case adds A when it ends."""
    from robots.mujoco_lab.calibrate import probe_cost
    from rpent.skills.pick_place.costs import TIME_COST_PER_S
    from rpent.skills.pick_place.probes import PROBE_SECONDS

    t = TIME_COST_PER_S * PROBE_SECONDS["squeeze"]
    assert probe_cost("squeeze", "measured", False, 10.0) == pytest.approx(t)
    assert probe_cost("squeeze", "damaged", False, 10.0) == pytest.approx(t + 50.0)
    assert probe_cost("weigh", "damaged", True, 1.0) == pytest.approx(TIME_COST_PER_S * PROBE_SECONDS["weigh"] + 5.0)


class _DamagingProbe(_Script):
    """A scripted case whose probe crushes the object."""

    def pick(self, **kw):
        self.calls.append(kw.get("option_id"))
        return SimpleNamespace(data={"probed": "squeeze", "kind": "damaged", "reading": {}, "thrown": False,
                                     "damage": {"label": "crushed", "channel": "force", "t": 1.0},
                                     "assessment": {"assessed": False, "reason": "object is damaged (crushed)"}})


def test_a_damaging_probe_ends_the_case_damaged(monkeypatch):
    """Review Focus 2: the case ends "damaged" and costs the probe's time, 5 kappa and A."""
    from robots.mujoco_lab import calibrate
    from rpent.skills.pick_place.costs import TIME_COST_PER_S
    from rpent.skills.pick_place.probes import PROBE_SECONDS

    sc = _DamagingProbe([_assessment("probe", probe=True)], [])
    monkeypatch.setattr(calibrate, "assess_pick", sc.assess)
    monkeypatch.setattr(calibrate, "pick_and_place", sc.pick)
    for protocol in ("policy", "probe_first"):
        sc.assessments = [_assessment("probe", probe=True)]
        case = calibrate.run_protocol(SimpleNamespace(abstain_cost=1.0), protocol, "table_centre", np.random.default_rng(0), care=10.0)
        assert case["ended"] == "damaged" and case["probes"][0]["kind"] == "damaged" and case["attempts"] == []
        assert case["cost"] == pytest.approx(TIME_COST_PER_S * PROBE_SECONDS["squeeze"] + 50.0 + 1.0)


def test_probe_kind_block_scores_the_damaged_kind():
    """The probes' declared kinds are scored with "damaged" among the classes."""
    from robots.mujoco_lab.calibrate import probe_kind_block

    rec = {"probe": "weigh", "kind": "damaged", "declared": {"p_kind": {"measured": 0.5, "damaged": 0.5}}, "const_kind": None}
    out = probe_kind_block([{"probes": [rec]}])
    assert out["weigh"]["brier"] == pytest.approx(0.5) and out["weigh"]["kinds"] == {"damaged": 1}
    assert out["squeeze"]["n"] == 0


def test_a_case_sets_the_slosh_scale_of_the_bundle_thresholds(monkeypatch):
    import robots.mujoco_lab.prior as prior
    from robots.mujoco_lab.calibrate import run_case
    from robots.mujoco_lab.fragility import THRESHOLDS_V0, Thresholds, load_classes

    class Stop(Exception):
        pass

    class Env:
        def reset(self, *a, **kw):
            raise Stop

    th = Thresholds.from_json(THRESHOLDS_V0)
    th.slosh_zeta, th.slosh_scale = 0.02, {"carry": {"0.6": 0.5}}
    monkeypatch.setattr(prior, "fragility_world", lambda manifest: (load_classes(), th))
    prims = SimpleNamespace(env=Env(), manifest={})
    with pytest.raises(Stop):
        run_case(prims, "ycb__mug", 0, 0)
    assert prims.env.slosh_zeta == 0.02 and prims.env.slosh_scale == {"carry": {"0.6": 0.5}}


def test_irreversible_block_sums_the_irreversible_classes():
    from robots.mujoco_lab import calibrate
    from rpent.skills.pick_place.evaluate import _ece, ece_ci
    from rpent.skills.pick_place.protocol import IRREVERSIBLE

    att = [_att(f"o{i // 2}", 0, 0.0, label) for i, label in enumerate(OUTCOMES)]
    for i, r in enumerate(att):
        r["declared"]["p"] = {k: (0.02 + i * 0.003 if k in IRREVERSIBLE else 0.0) for k in OUTCOMES}
        r["declared"]["p"]["placed_stable"] = 1.0 - sum(r["declared"]["p"].values())
        r["const"] = [0.1 if k in IRREVERSIBLE else 0.08 for k in OUTCOMES]
    p = np.array([sum(r["declared"]["p"][k] for k in IRREVERSIBLE) for r in att])
    y = np.array([r["label"] in IRREVERSIBLE for r in att], dtype=float)
    ci = ece_ci(p, y, [r["object"] for r in att], np.random.default_rng(4))
    out = calibrate.irreversible_block(att, np.random.default_rng(4))
    assert out["n"] == len(att) and out["n_irreversible"] == len(IRREVERSIBLE)
    assert out["brier"] == pytest.approx(np.mean((p - y) ** 2))
    assert out["brier_const"] == pytest.approx(np.mean((0.6 - y) ** 2))
    assert out["ece"] == pytest.approx(_ece(p, y))
    assert out["ece_ci95_object_bootstrap"] == pytest.approx(ci)
    assert out["acceptance"]["ece_rule"] == calibrate.ece_rule(ci)
    assert calibrate.summarize([], "commit", np.random.default_rng(0))["irreversible"]["n"] == 0


def test_damage_class_marked_insufficient_under_50_positives():
    from robots.mujoco_lab import calibrate
    from rpent.skills.pick_place.protocol import DAMAGE

    att = [_att(f"o{i % 3}", 0, 0.5, "crushed") for i in range(49)]
    out = calibrate.irreversible_block(att, np.random.default_rng(0))
    assert set(out["per_class"]) == set(DAMAGE)
    assert out["per_class"]["crushed"] == {"n_pos": 49, "ece": "insufficient"}
    assert all(out["per_class"][k] == {"n_pos": 0, "ece": "insufficient"} for k in DAMAGE[1:])
    out = calibrate.irreversible_block(att + [att[0]], np.random.default_rng(0))
    assert out["per_class"]["crushed"] == {"n_pos": 50, "ece": 1.0}
