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


def test_calibrate_two_objects_one_repeat(tmp_path, monkeypatch):
    from robots.mujoco_lab.calibrate import main

    monkeypatch.setenv("RPENT_PICK_PLACE_BUNDLE", os.path.join(os.path.dirname(METRICS), "m2"))     # m2's rules; the default is m2b2
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


def test_threshold_reaches_the_menu(tmp_path, monkeypatch):
    """--threshold 0.0 reaches the menu: no case abstains on a decline (the m2 rule; under a cost rule the threshold is a floor)."""
    from robots.mujoco_lab.calibrate import main

    monkeypatch.setenv("RPENT_PICK_PLACE_BUNDLE", os.path.join(os.path.dirname(METRICS), "m2"))     # m2's rules; the default is m2b2
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
    from robots.mujoco_lab.robot_spec import DATA_DIR, build_lab, load_bundle

    m2 = DATA_DIR / "m2"

    assert load_bundle(str(m2))["abstain_cost"] is None
    b = tmp_path / "b"
    b.mkdir()
    for f in os.listdir(m2):
        if f != "manifest.json":
            (b / f).symlink_to(m2 / f)
    man = json.load(open(m2 / "manifest.json"))
    json.dump(dict(man, decision={"abstain_cost": 1.0}), open(b / "manifest.json", "w"))
    assert load_bundle(str(b))["abstain_cost"] == 1.0
    assert build_lab(bundle=str(b)).abstain_cost == 1.0
    assert _primitives(str(b), None, None).abstain_cost == 1.0
    assert build_lab(bundle=str(m2)).abstain_cost is None


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


def test_probe_kind_block_leaves_out_the_thrown_probes():
    """A 2b probe model declares the kinds of a result not thrown (P(kind | not thrown)), so its thrown probes are left
    out here, and the thrown block scores them. A milestone 2a model's kinds hold the throws, so its thrown probes stay
    (Codex review of the fixes, plan 2b)."""
    from robots.mujoco_lab.calibrate import probe_kind_block

    kept = {"probe": "push", "kind": "slid", "thrown": False, "declared": {"p_kind": {"slid": 1.0}}, "const_kind": {"slid": 1.0},
            "kinds_not_thrown": True}
    out = probe_kind_block([{"probes": [kept, dict(kept, kind="damaged", thrown=True)]}])["push"]
    assert out["n"] == 1 and out["kinds"] == {"slid": 1} and out["brier"] == 0.0 and out["brier_const"] == 0.0
    old = [dict(kept, kinds_not_thrown=False), dict(kept, thrown=True, kinds_not_thrown=False)]
    assert probe_kind_block([{"probes": old}])["push"]["n"] == 2


def test_probe_record_constant_is_the_kind_rate_of_the_results_not_thrown():
    """The constant baseline of the kinds is their rate among the probe model's kind rows (for a 2b model the results
    not thrown), so it sums to 1; the thrown constant is the rate among every row. The record says which kinds the
    model declares."""
    from robots.mujoco_lab.calibrate import _probe_record
    from rpent.skills.pick_place.probes import CalibratedClassifier, ProbeModel

    classes = ("slid", "tipped", "no_breakaway", "no_contact")
    clf = CalibratedClassifier(classes, None, None, np.array([0.8, 0.2, 0.0, 0.0]))
    pm = ProbeModel(probe="push", classes=classes, clf=clf, reading={}, p_thrown=0.5,
                    stats={"n": 100, "n_not_thrown": 50, "kinds": {"slid": 40, "tipped": 10}, "thrown": 50})
    d = {"probed": "push", "kind": "slid", "reading": {}, "thrown": False}
    rec = _probe_record(d, SimpleNamespace(probe_models={"push": pm}), 1.0)
    assert rec["const_kind"] == pytest.approx({"slid": 0.8, "tipped": 0.2}) and rec["const_thrown"] == pytest.approx(0.5)
    assert rec["kinds_not_thrown"] is True


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


def test_job_force_noise_follows_the_bundle():
    """Plan 2b2 decision 8: a job runs a bundle with the sensor it declares, unless --force-noise overrides it."""
    from robots.mujoco_lab.calibrate import job_force_noise

    assert job_force_noise(None, {"decision": {"abstain_cost": 1.0}}) == 0.0   # m2b: no sensor, clean corpora
    assert job_force_noise(None, {"sensor": {"force_noise": 0.05}}) == 0.05
    assert job_force_noise(0.02, {"sensor": {"force_noise": 0.05}}) == 0.02
    assert job_force_noise(None, None) is None                                  # no bundle: the env's own setting


def test_calibrate_runs_a_bundle_with_its_declared_noise():
    """The m2b bundle's job reads clean forces under the new lab default; an override reaches the rows."""
    from robots.mujoco_lab.calibrate import _run_object
    from robots.mujoco_lab.robot_spec import DATA_DIR

    bundle = str(DATA_DIR / "m2b")
    if not os.environ.get("RPENT_MUJOCO_SIM_DIR") or not os.path.exists(os.path.join(bundle, "success_model.joblib")):
        pytest.skip("needs the sim and the m2b bundle's local models")
    s = {"bundle": bundle, "fold": None, "const": None, "seed": 5, "protocol": "commit", "follow": False, "fixtures": False,
         "threshold": None, "model_kind": None, "care": "normal", "abstain_cost": 1.0, "force_noise": None}
    row = _run_object(("ycb__tomato_soup_can", [0], s))[0]
    assert row["force_noise"] == 0.0 and row["evoi"] == "plugin"             # m2b declares no estimator: the plug-in
    assert _run_object(("ycb__tomato_soup_can", [0], dict(s, force_noise=0.05)))[0]["force_noise"] == 0.05


def test_force_noise_argument_is_checked():
    """--force-noise takes a finite number >= 0 (argparse reports anything else)."""
    import argparse

    from robots.mujoco_lab.calibrate import _noise_arg

    assert _noise_arg("0.05") == 0.05 and _noise_arg("0") == 0.0
    for bad in ("-0.1", "nan", "inf", "x"):
        with pytest.raises(argparse.ArgumentTypeError):
            _noise_arg(bad)


def test_run_records_its_force_noise():
    """The output's force_noise: the one level its cases ran with (error rows carry none), the levels when they differ."""
    from robots.mujoco_lab.calibrate import run_force_noise

    assert run_force_noise([{"force_noise": 0.05}, {"error": "x"}, {"force_noise": 0.05}]) == 0.05
    assert run_force_noise([{"force_noise": 0.05}, {"force_noise": 0.0}]) == [0.0, 0.05]
    assert run_force_noise([{"error": "x"}]) is None


def test_cached_primitives_come_back_with_the_bundle_settings():
    """_primitives caches one primitives object per bundle; a job's --force-noise or --abstain-cost must not stay on
    it for the next caller (the m2b probe pin read 0.05 after a job with an override)."""
    from robots.mujoco_lab.calibrate import _primitives, _run_object
    from robots.mujoco_lab.robot_spec import DATA_DIR

    bundle = str(DATA_DIR / "m2b")
    if not os.environ.get("RPENT_MUJOCO_SIM_DIR") or not os.path.exists(os.path.join(bundle, "success_model.joblib")):
        pytest.skip("needs the sim and the m2b bundle's local models")
    s = {"bundle": bundle, "fold": None, "const": None, "seed": 5, "protocol": "commit", "follow": False, "fixtures": False,
         "threshold": None, "model_kind": None, "care": "normal", "abstain_cost": 3.0, "force_noise": 0.05,
         "probe_time_scale": 0.25}
    row = _run_object(("ycb__tomato_soup_can", [0], s))[0]
    assert row["probe_time_scale"] == 0.25
    prims = _primitives(bundle, None, None)
    assert prims.force_noise == 0.0 and prims.abstain_cost == 1.0 and prims.probe_time_scale == 1.0


def test_cached_primitives_take_the_bundle_estimator(monkeypatch):
    """Plan 2b2 decision 2: calibrate's primitives value probes with the bundle's estimator."""
    from robots.mujoco_lab import calibrate, robot_spec

    bundle = {"model": None, "prior": None, "readout_model": None, "probe_models": {}, "guard": None, "abstain_cost": 1.0,
              "force_noise": 0.05, "evoi": "heldout", "probe_time_scale": 1.0, "manifest": {}}
    monkeypatch.setattr(robot_spec, "load_bundle", lambda b, kind=None: bundle)
    monkeypatch.setattr(calibrate, "_STATE", {"env": object(), "prims": {}})
    assert calibrate._primitives("b", None, None).evoi == "heldout"


def test_run_records_its_estimator():
    """The output's evoi: the probe-value estimator its cases ran with (one name, or the names when they differ)."""
    from robots.mujoco_lab.calibrate import run_evoi

    assert run_evoi([{"evoi": "heldout"}, {"error": "x"}, {"evoi": "heldout"}]) == "heldout"
    assert run_evoi([{"evoi": "heldout"}, {"evoi": "plugin"}]) == ["heldout", "plugin"]
    assert run_evoi([{"error": "x"}]) is None


def test_policy_accounting_uses_the_probe_time_scale():
    """Plan 2b2 decision 4: an executed probe's realized time cost is scaled by the job's probe time scale (the
    bundle's, or the --probe-time-scale override; 1 without a bundle); the rows and the run record it."""
    from types import SimpleNamespace

    from robots.mujoco_lab.calibrate import _probe_record, job_probe_time_scale, probe_cost, run_probe_time_scale
    from rpent.skills.pick_place.costs import COST, TIME_COST_PER_S
    from rpent.skills.pick_place.probes import PROBE_SECONDS

    t = TIME_COST_PER_S * PROBE_SECONDS["weigh"]
    assert probe_cost("weigh", "measured", False, "normal") == pytest.approx(t)
    assert probe_cost("weigh", "measured", False, "normal", time_scale=0.25) == pytest.approx(0.25 * t)
    assert probe_cost("push", "tipped", False, "normal", time_scale=0.0) == pytest.approx(COST["tipped"])
    d = {"probed": "weigh", "kind": "measured", "reading": {}, "thrown": False}
    assert _probe_record(d, SimpleNamespace(probe_models={}, probe_time_scale=0.25), "normal")["cost"] == pytest.approx(0.25 * t)
    assert _probe_record(d, SimpleNamespace(probe_models={}), "normal")["cost"] == pytest.approx(t)
    assert job_probe_time_scale(None, {"decision": {"probe_time_scale": 0.25}}) == 0.25
    assert job_probe_time_scale(0.0, {"decision": {"probe_time_scale": 0.25}}) == 0.0
    assert job_probe_time_scale(None, {"decision": {"abstain_cost": 1.0}}) == 1.0 and job_probe_time_scale(None, None) == 1.0
    assert run_probe_time_scale([{"probe_time_scale": 0.25}, {"error": "x"}]) == 0.25
    assert run_probe_time_scale([{"probe_time_scale": 0.25}, {"probe_time_scale": 1.0}]) == [0.25, 1.0]


def test_probe_time_scale_argument_is_checked():
    import argparse

    from robots.mujoco_lab.calibrate import _scale_arg

    assert _scale_arg("0.25") == 0.25 and _scale_arg("0") == 0.0
    for bad in ("-0.1", "nan", "inf", "x"):
        with pytest.raises(argparse.ArgumentTypeError):
            _scale_arg(bad)


def test_compare_refuses_runs_with_different_probe_time_scales():
    """A file without the field reads as scale 1."""
    from robots.mujoco_lab.calibrate import compare

    row = {"object": "o", "idx": 0, "cost": 1.0, "attempts": []}
    a = {"protocol": "policy", "care_scale": 1.0, "abstain_cost": 1.0, "rows": [row]}
    with pytest.raises(ValueError, match="different costs"):
        compare(a, dict(a, probe_time_scale=0.25), np.random.default_rng(0))
    assert compare(a, dict(a, probe_time_scale=1.0), np.random.default_rng(0))["n_paired"] == 1


def test_cached_primitives_take_the_bundle_probe_time_scale(monkeypatch):
    """calibrate's primitives start each call with the bundle's probe time scale: a job's override does not stay."""
    from robots.mujoco_lab import calibrate, robot_spec

    bundle = {"model": None, "prior": None, "readout_model": None, "probe_models": {}, "guard": None, "abstain_cost": 1.0,
              "force_noise": 0.05, "evoi": "heldout", "probe_time_scale": 0.25, "manifest": {}}
    monkeypatch.setattr(robot_spec, "load_bundle", lambda b, kind=None: bundle)
    monkeypatch.setattr(calibrate, "_STATE", {"env": object(), "prims": {}})
    prims = calibrate._primitives("b", None, None)
    assert prims.probe_time_scale == 0.25
    prims.probe_time_scale = 3.0                                       # a job's --probe-time-scale
    assert calibrate._primitives("b", None, None).probe_time_scale == 0.25


def test_the_constant_baseline_reads_the_corpora_on_another_checkout():
    """The final review of plan 2b2: a bundle recorded its corpora's absolute paths. On another checkout the constant
    baseline used uniform rates without a word; a path under another checkout's robots/mujoco_lab/ now resolves on this
    one, and a corpus that is nowhere is an error."""
    from robots.mujoco_lab.calibrate import class_rates
    from robots.mujoco_lab.robot_spec import DATA_DIR

    metrics = json.load(open(DATA_DIR / "m2b2" / "success_model_metrics.json"))
    here = class_rates(metrics)
    moved = dict(metrics, corpus=["/elsewhere/RPent/robots/mujoco_lab/data/" + os.path.basename(p) for p in metrics["corpus"]])
    assert class_rates(moved) == here and len(set(here)) > 1
    with pytest.raises(FileNotFoundError, match="corpus"):
        class_rates(dict(metrics, corpus=["/nowhere/corpus_none.jsonl"]))


# ---------------------------------------------------------------- the milestone 4 arms, case lists and records
def test_policy_no_probe_follows_without_probe(monkeypatch):
    """Plan m4 decision 3: the retry baseline follows the menu's recommendation without its probe rows (the payload's
    without_probe, from the unrounded costs) at every menu."""
    from robots.mujoco_lab.calibrate import run_protocol

    sc = _scripted(monkeypatch, [dict(_assessment("probe", probe=True), without_probe="commit"), _assessment("commit")], ["placed_stable"])
    case = run_protocol(SimpleNamespace(abstain_cost=1.0), "policy_no_probe", "table_centre", np.random.default_rng(0))
    assert sc.calls == ["commit_std"] and case["probes"] == [] and case["ended"] == "placed"
    sc = _scripted(monkeypatch, [dict(_assessment("probe", probe=True), without_probe="decline")], [])
    case = run_protocol(SimpleNamespace(abstain_cost=1.0), "policy_no_probe", "table_centre", np.random.default_rng(0))
    assert sc.calls == [] and case["ended"] == "abstained" and case["cost"] == pytest.approx(1.0)


def test_policy_no_probe_equals_the_policy_without_a_probe_recommendation(monkeypatch):
    from robots.mujoco_lab.calibrate import run_protocol

    cases = []
    for protocol in ("policy", "policy_no_probe"):
        _scripted(monkeypatch, [_assessment("commit"), _assessment("commit"), _assessment("commit")], ["missed", "placed_stable"])
        cases.append(run_protocol(SimpleNamespace(abstain_cost=1.0), protocol, "table_centre", np.random.default_rng(0)))
    a, b = ({k: v for k, v in c.items() if k != "protocol"} for c in cases)
    assert a == b and len(a["attempts"]) == 2 and cases[1]["protocol"] == "policy_no_probe"


def test_policy_probe_one_runs_the_cheapest_probe_then_the_policy(monkeypatch):
    """Plan m4 decision 3: the first menu's cheapest probe row runs whatever the recommendation, then the policy loop;
    no probe after a guard flag."""
    from robots.mujoco_lab.calibrate import run_protocol

    first = _assessment("commit")
    first["options"] = first["options"] + [{"id": "probe_push", "kind": "probe", "action": {"probe": "push"}, "expected_cost": 0.9},
                                           {"id": "probe_weigh", "kind": "probe", "action": {"probe": "weigh"}, "expected_cost": 0.7}]
    sc = _scripted(monkeypatch, [first, _assessment("commit")], ["placed_stable"])
    case = run_protocol(SimpleNamespace(abstain_cost=1.0), "policy_probe_one", "table_centre", np.random.default_rng(0))
    assert sc.calls == ["probe_weigh", "commit_std"] and case["ended"] == "placed" and len(case["probes"]) == 1
    sc = _scripted(monkeypatch, [dict(first, recommendation="decline", guard={"flag": True})], [])
    case = run_protocol(SimpleNamespace(abstain_cost=1.0), "policy_probe_one", "table_centre", np.random.default_rng(0))
    assert sc.calls == [] and case["ended"] == "abstained" and case["probes"] == []


def test_probe_records_say_where_the_probe_ran(monkeypatch):
    from robots.mujoco_lab.calibrate import run_protocol

    two = _assessment("probe", probe=True)
    sc = _scripted(monkeypatch, [two, two, _assessment("commit")], ["placed_stable"])
    case = run_protocol(SimpleNamespace(abstain_cost=1.0), "policy", "table_centre", np.random.default_rng(0))
    assert sc.calls == ["probe_push", "probe_push", "commit_std"]
    assert [(p["position"], p["after_attempts"]) for p in case["probes"]] == [(0, 0), (1, 0)]


def test_the_case_records_its_first_menu_costs(monkeypatch):
    from robots.mujoco_lab.calibrate import run_protocol

    a = _assessment("probe", probe=True)
    a["options"][1]["expected_cost"] = 0.4
    _scripted(monkeypatch, [a, _assessment("decline")], [])
    case = run_protocol(SimpleNamespace(abstain_cost=1.0), "policy", "table_centre", np.random.default_rng(0))
    assert case["first_menu"] == {"best_cost": 0.6, "recommended_cost": 0.4}
    _scripted(monkeypatch, [_assessment("decline")], [])
    case = run_protocol(SimpleNamespace(abstain_cost=1.0), "policy", "table_centre", np.random.default_rng(0))
    assert case["first_menu"] == {"best_cost": 0.6, "recommended_cost": None}


@needs_bundle
def test_cases_file_runs_only_the_listed_cases(tmp_path):
    """Plan m4 decision 4: a cases file runs exactly its cases, the same cases a full run draws; it must be for the run's
    seed, and its objects must have folds; the output records the file's sha256 (plan H3 decision 11)."""
    import hashlib

    from robots.mujoco_lab.calibrate import main

    cf = tmp_path / "folds"
    cf.mkdir()
    _stub_fold(cf, 0, ["ycb__brick", "ycb__mug"], ["ycb__clamp"], ["ycb__banana"])
    _stub_fold(cf, 1, ["ycb__banana", "ycb__mug"], ["ycb__clamp"], ["ycb__brick"])
    (cf / "folds.json").write_text(json.dumps({"ycb__banana": 0, "ycb__brick": 1}))
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"seed": 13, "cases": {"ycb__banana": [2, 5]}}))
    base = ["--crossfit", str(cf), "--protocol", "commit", "--coverage-sims", "0"]
    res = main(base + ["--repeats", "11", "--cases", str(cases), "--seed", "13", "--out", str(tmp_path / "c.json")])
    assert [(r["object"], r["idx"]) for r in res["rows"]] == [("ycb__banana", 2), ("ycb__banana", 5)]
    assert res["cases"] == {"path": str(cases), "n": 2, "sha256": hashlib.sha256(cases.read_bytes()).hexdigest()}
    assert res["objects"] == ["ycb__banana"]
    full = main(base + ["--repeats", "6", "--objects", "ycb__banana", "--seed", "13", "--out", str(tmp_path / "f.json")])

    def key(r):
        return r["idx"], r["destination"], json.dumps(r["theta"], sort_keys=True), [(t["option"], t["outcome"]) for t in r["attempts"]]

    assert [key(r) for r in res["rows"]] == [key(r) for r in full["rows"] if r["idx"] in (2, 5)]
    with pytest.raises(SystemExit, match="seed"):
        main(base + ["--repeats", "11", "--cases", str(cases), "--seed", "12", "--out", str(tmp_path / "d.json")])
    cases.write_text(json.dumps({"seed": 13, "cases": {"ycb__clamp": [0]}}))
    with pytest.raises(SystemExit, match="no fold"):
        main(base + ["--repeats", "11", "--cases", str(cases), "--seed", "13", "--out", str(tmp_path / "e.json")])


def test_the_output_records_its_code_revision(tmp_path):
    """Plan m4 decision 9: every output says which code made it, and is written whole or not at all; an arm without a
    forced probe records no forced-probe rule (plan H3 decision 11)."""
    import re

    from robots.mujoco_lab.calibrate import main

    out = tmp_path / "c.json"
    main(["--metrics", METRICS, "--repeats", "1", "--out", str(out), "--objects", "ycb__banana", "--seed", "2", "--coverage-sims", "0"])
    saved = json.load(open(out))
    rev = saved["code"]["revision"]
    assert (rev is None or re.fullmatch(r"[0-9a-f]{40}", rev)) and saved["code"]["dirty"] in (True, False, None)
    assert saved["cases"] is None and not list(tmp_path.glob("*.tmp"))
    assert saved["protocol"] == "commit" and saved["forced_rule"] is None



# ---------------------------------------------------------------- the recovery mode (plan H3 decision 3)
class _RestoringScript(_Script):
    """A scripted case under the recovery mode: an irreversible attempt, or a thrown probe (``thrown``), comes back
    restored with the skill's restore record."""

    def __init__(self, assessments, outcomes, thrown=False):
        super().__init__(assessments, outcomes)
        self.thrown = thrown

    def pick(self, **kw):
        d = super().pick(**kw).data
        if "probed" in d and self.thrown:
            d.update(thrown=True, restored={"after": "probe", "event": "thrown", "check": {"same_case": True}})
        elif d.get("outcome") in ("dropped_high", "rolled_off", "crushed", "bruised", "broken", "spilled"):
            d["restored"] = {"after": "attempt", "event": d["outcome"], "check": {"same_case": True}}
        return SimpleNamespace(data=d)


def _restoring(monkeypatch, assessments, outcomes, thrown=False):
    from robots.mujoco_lab import calibrate

    sc = _RestoringScript(assessments, outcomes, thrown)
    monkeypatch.setattr(calibrate, "assess_pick", sc.assess)
    monkeypatch.setattr(calibrate, "pick_and_place", sc.pick)
    return sc


def test_recovery_retries_after_an_irreversible_event(monkeypatch):
    """Under reset_s 20 a dropped_high comes back restored and the policy retries: dropped_high then placed costs
    0.05 x (8 + 8) + 1.0 and ends placed; absorbing, the same script ends after the drop at 5 kappa + A + 0.05 x 8."""
    from robots.mujoco_lab.calibrate import run_protocol
    from rpent.skills.pick_place.costs import TIME_COST_PER_S

    sc = _restoring(monkeypatch, [_assessment("commit")] * 3, ["dropped_high", "placed_stable"])
    case = run_protocol(SimpleNamespace(abstain_cost=10.0, reset_s=20.0), "policy", "table_centre", np.random.default_rng(0), care=10.0)
    assert sc.calls == ["commit_std", "commit_std"] and case["ended"] == "placed"
    assert case["cost"] == pytest.approx(TIME_COST_PER_S * 16.0 + 1.0)
    assert [(r["after"], r["event"]) for r in case["restores"]] == [("attempt", "dropped_high")]
    sc = _scripted(monkeypatch, [_assessment("commit")] * 3, ["dropped_high", "placed_stable"])
    case = run_protocol(SimpleNamespace(abstain_cost=10.0), "policy", "table_centre", np.random.default_rng(0), care=10.0)
    assert sc.calls == ["commit_std"] and case["ended"] == "failed" and case["restores"] == []
    assert case["cost"] == pytest.approx(50.0 + 10.0 + TIME_COST_PER_S * 8.0)


def test_a_restored_probe_does_not_end_the_case(monkeypatch):
    """Under reset_s a thrown probe costs its time and the reset (0.05 x 20), the object comes back, and the case goes
    on to the next assessment's row."""
    from robots.mujoco_lab.calibrate import run_protocol
    from rpent.skills.pick_place.costs import TIME_COST_PER_S
    from rpent.skills.pick_place.probes import PROBE_SECONDS

    sc = _restoring(monkeypatch, [_assessment("probe", probe=True), _assessment("commit"), _assessment("commit")], ["placed_stable"],
                    thrown=True)
    case = run_protocol(SimpleNamespace(abstain_cost=10.0, reset_s=20.0), "policy", "table_centre", np.random.default_rng(0), care=10.0)
    assert sc.calls == ["probe_push", "commit_std"] and case["ended"] == "placed"
    assert case["probes"][0]["cost"] == pytest.approx(TIME_COST_PER_S * (PROBE_SECONDS["push"] + 20.0))
    assert case["cost"] == pytest.approx(TIME_COST_PER_S * (PROBE_SECONDS["push"] + 20.0 + 8.0))
    assert [(r["after"], r["event"]) for r in case["restores"]] == [("probe", "thrown")]


def test_the_reset_prices_the_realized_costs():
    from robots.mujoco_lab.calibrate import attempt_cost, probe_cost
    from rpent.skills.pick_place.costs import TIME_COST_PER_S
    from rpent.skills.pick_place.probes import PROBE_SECONDS

    assert attempt_cost("dropped_high", 8.0, 10.0, reset_s=20.0) == pytest.approx(TIME_COST_PER_S * 28.0)
    assert attempt_cost("dropped_high", 8.0, 10.0) == pytest.approx(50.0 + TIME_COST_PER_S * 8.0)
    assert attempt_cost("tipped", 8.0, 10.0, reset_s=20.0) == attempt_cost("tipped", 8.0, 10.0)
    assert probe_cost("squeeze", "damaged", False, 10.0, reset_s=0.0) == pytest.approx(TIME_COST_PER_S * PROBE_SECONDS["squeeze"])


def test_reset_argument_reaches_the_workers_and_the_output(tmp_path):
    import argparse

    from robots.mujoco_lab.calibrate import _reset_arg, attempt_cost, main, run_reset_s

    assert _reset_arg("20") == 20.0 and _reset_arg("0") == 0.0
    for bad in ("-1", "nan", "inf", "x"):
        with pytest.raises(argparse.ArgumentTypeError):
            _reset_arg(bad)
    assert run_reset_s([{"reset_s": 20.0}, {"error": "x"}]) == 20.0 and run_reset_s([{"reset_s": None}]) is None
    out = tmp_path / "c.json"
    main(["--metrics", METRICS, "--repeats", "1", "--out", str(out), "--objects", "ycb__banana", "--seed", "2", "--care", "10",
          "--abstain-cost", "2", "--reset-s", "20"])
    saved = json.load(open(out))
    assert saved["reset_s"] == 20.0 and saved["rows"][0]["reset_s"] == 20.0
    t = saved["rows"][0]["attempts"][0]
    assert t["cost"] == pytest.approx(attempt_cost(t["outcome"], t["cost_s"], 10.0, reset_s=20.0))


def test_compare_refuses_runs_with_different_reset_times():
    """plan H3: a file without the field reads as absorbing (no reset)."""
    from robots.mujoco_lab.calibrate import compare

    row = {"object": "o", "idx": 0, "cost": 1.0, "attempts": []}
    a = {"protocol": "policy", "care_scale": 1.0, "abstain_cost": 1.0, "rows": [row]}
    with pytest.raises(ValueError, match="different costs"):
        compare(a, dict(a, reset_s=20.0), np.random.default_rng(0))
    assert compare(a, dict(a, reset_s=None), np.random.default_rng(0))["n_paired"] == 1


# ---------------------------------------------------------------- the spread factor (plan H3 decision 4)
def test_shrink_keeps_the_case_stream():
    """A narrow case is its wide case with the shrunk theta: over 50 cases the CoM end, destination, v_lift, fill and
    fragility are equal at f = 1 and 0.25, and the narrow theta is shrink_theta of the wide one (filled or not)."""
    from robots.mujoco_lab.adapter import object_props
    from robots.mujoco_lab.calibrate import case_draw
    from robots.mujoco_lab.corpus import shrink_theta
    from robots.mujoco_lab.fragility import THRESHOLDS_V0, Thresholds, load_classes

    world, filled = (load_classes(), Thresholds.from_json(THRESHOLDS_V0)), 0
    for oid in ("ycb__mug", "ycb__banana"):
        mass = object_props(oid)["mass"]
        for i in range(25):
            wide = case_draw(3, oid, i, mass, True, fragility=world)
            narrow = case_draw(3, oid, i, mass, True, fragility=world, spread=0.25)
            assert narrow[1:] == wide[1:] and narrow[0]["v_lift"] == wide[0]["v_lift"]
            assert narrow[0] == shrink_theta(wide[0], 0.25, wide[3]["fill"] > 0)
            filled += wide[3]["fill"] > 0
    assert 0 < filled < 25                              # both branches of the mug ran


SPREAD_ONE_PINS = ("3aafbcc248fdb5b0", "7743bd5a185c308d")   # milestone 4's draws (b3136f5), before the spread factor


def _spread_one_digests(**kw):
    """Digests of 120 case draws (the mug and the banana, m2b and m2 worlds) and of 300 prior particles of each prior
    for both objects, with the prior streams' states after the draw."""
    import hashlib

    from robots.mujoco_lab.adapter import MuJoCoLabEnv, object_props
    from robots.mujoco_lab.calibrate import case_draw
    from robots.mujoco_lab.fragility import THRESHOLDS_V0, Thresholds, load_classes
    from robots.mujoco_lab.prior import lab_prior, make_prior

    classes, th = load_classes(), Thresholds.from_json(THRESHOLDS_V0)
    prior, env, cases, parts = make_prior(classes, th), MuJoCoLabEnv(), [], []
    for oid in ("ycb__mug", "ycb__banana"):
        mass = object_props(oid)["mass"]
        for world in ((classes, th), None):
            cases += [case_draw(5, oid, i, mass, True, fragility=world, **kw) for i in range(30)]
        env.reset(oid, seed=0)
        for fn in (prior, lab_prior):
            rng = np.random.default_rng(11)
            b = fn(env.pose_info(), 300, rng, **kw)
            parts.append([b.particles, rng.bit_generator.state])
    h = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:16]  # noqa: E731
    return h(cases), h(parts)


def test_spread_one_is_milestone4():
    """f = 1 (the default, or named) gives milestone 4's case draws and prior particles bit for bit, the streams'
    states included."""
    assert _spread_one_digests() == SPREAD_ONE_PINS
    assert _spread_one_digests(spread=1.0) == SPREAD_ONE_PINS


def test_spread_argument_reaches_the_workers_and_the_output(tmp_path):
    import argparse

    from robots.mujoco_lab.calibrate import _spread_arg, main, run_spread

    assert _spread_arg("0.25") == 0.25 and _spread_arg("1") == 1.0
    for bad in ("0", "-0.5", "1.5", "nan", "x"):
        with pytest.raises(argparse.ArgumentTypeError):
            _spread_arg(bad)
    assert run_spread([{"spread": 0.25}, {"error": "x"}]) == 0.25 and run_spread([{"spread": 1.0}]) == 1.0
    out = tmp_path / "c.json"
    main(["--metrics", METRICS, "--repeats", "2", "--out", str(out), "--objects", "ycb__banana", "--seed", "2", "--spread", "0.25"])
    saved = json.load(open(out))
    assert saved["spread"] == 0.25 and all(r["spread"] == 0.25 for r in saved["rows"])
    for r in saved["rows"]:
        assert 0.4 <= r["theta"]["mu"] <= 0.6 and 0.978 <= r["theta"]["mass_scale"] <= 1.533
        assert r["theta"]["com_s"] is None or 0.4875 <= r["theta"]["com_s"] <= 0.6125


def test_compare_refuses_runs_with_different_spreads():
    """plan H3: a file without the field reads as f = 1."""
    from robots.mujoco_lab.calibrate import compare

    row = {"object": "o", "idx": 0, "cost": 1.0, "attempts": []}
    a = {"protocol": "policy", "care_scale": 1.0, "abstain_cost": 1.0, "rows": [row]}
    with pytest.raises(ValueError, match="different spreads"):
        compare(a, dict(a, spread=0.25), np.random.default_rng(0))
    assert compare(a, dict(a, spread=1.0), np.random.default_rng(0))["n_paired"] == 1


# ---------------------------------------------------------------- the forced probe and the sham (plan H3 decisions 5, 6)
M2B2 = os.path.join(os.path.dirname(METRICS), "m2b2")      # the default bundle (H3 runs on it); BUNDLE stays m2
needs_m2b2 = pytest.mark.skipif(not os.path.exists(os.path.join(M2B2, "success_model.joblib")), reason="the m2b2 bundle is not built")


def _forced_menu(rec="probe", squeeze=False, guard=None, without="commit"):
    a = _assessment(rec, probe=True)
    a["options"] = a["options"] + [{"id": "probe_weigh", "kind": "probe", "action": {"probe": "weigh"}, "expected_cost": 0.7}]
    if squeeze:
        a["options"].append({"id": "probe_squeeze", "kind": "probe", "action": {"probe": "squeeze"}, "expected_cost": 0.8})
    return dict(a, guard=guard, without_probe=without)


class _ForcedScript(_Script):
    """The scripted skill for the forced-probe arms: a probe payload names its option's probe; ``damage`` crushes the
    object (the case ends) or, with ``restore``, comes back restored (the recovery mode); the sham goes through the same
    script and adds the sham record."""

    def __init__(self, assessments, outcomes, damage=False, restore=False):
        super().__init__(assessments, outcomes)
        self.damage, self.restore, self.shams = damage, restore, 0

    def pick(self, **kw):
        oid = str(kw.get("option_id"))
        if not oid.startswith("probe_"):
            return super().pick(**kw)
        self.calls.append(oid)
        nxt = self.assessments.pop(0) if self.assessments else _assessment("decline")
        d = {"probed": oid[len("probe_"):], "kind": "measured", "reading": {"k_hat": 1.0}, "thrown": False, "assessment": nxt}
        if self.damage:
            d.update(kind="damaged", reading={}, damage={"label": "crushed", "channel": "force", "t": 1.0})
            if self.restore:
                d["restored"] = {"after": "probe", "event": "crushed", "check": {"same_case": True}}
            else:
                d["assessment"] = {"assessed": False, "reason": "object is damaged (crushed)"}
        return SimpleNamespace(data=d)

    def sham(self, **kw):
        self.shams += 1
        return dict(self.pick(**kw).data, sham=True, belief_hash_before="h0", belief_hash_after="h0", belief_hash_next="h0")


def _forced(monkeypatch, assessments, outcomes, **kw):
    from robots.mujoco_lab import calibrate

    sc = _ForcedScript(assessments, outcomes, **kw)
    monkeypatch.setattr(calibrate, "assess_pick", sc.assess)
    monkeypatch.setattr(calibrate, "pick_and_place", sc.pick)
    monkeypatch.setattr(calibrate, "sham_probe", sc.sham)
    return sc


def test_probe_once_runs_one_probe_then_no_more(monkeypatch):
    """The forced probe runs at the first menu, then the policy without probes: a menu that recommends a probe again
    commits by without_probe, or declines by it; the case records the forced probe, the decision after it and the
    change, and first_menu keeps the decision without probes and the best row."""
    from robots.mujoco_lab.calibrate import run_protocol

    sc = _forced(monkeypatch, [_forced_menu(), _forced_menu(), _assessment("commit")], ["placed_stable"])
    case = run_protocol(SimpleNamespace(abstain_cost=1.0), "probe_once", "table_centre", np.random.default_rng(0))
    assert sc.calls == ["probe_weigh", "commit_std"] and len(case["probes"]) == 1 and case["ended"] == "placed"
    assert case["forced_probe"] == "weigh" and case["post_probe"] == "commit_std" and case["changed"] == "same"
    assert case["first_menu"]["without_probe"] == "commit" and case["first_menu"]["best_row"] == "commit_std"
    sc = _forced(monkeypatch, [_forced_menu(), _forced_menu(without="decline")], [])
    case = run_protocol(SimpleNamespace(abstain_cost=1.0), "probe_once", "table_centre", np.random.default_rng(0))
    assert sc.calls == ["probe_weigh"] and case["ended"] == "abstained" and len(case["probes"]) == 1
    assert case["post_probe"] == "decline" and case["changed"] == "decline" and sc.shams == 0


def test_the_forced_probe_rule(monkeypatch):
    """The squeeze when the first menu offers it, else the weigh; no probe after a guard flag or without either row
    (then the policy without probes runs: post_probe and changed "no_probe"); sham_once forces the same probe through
    the sham."""
    from robots.mujoco_lab.calibrate import forced_probe, run_protocol

    assert forced_probe(_forced_menu(squeeze=True)) == "probe_squeeze" and forced_probe(_forced_menu()) == "probe_weigh"
    assert forced_probe(_forced_menu(squeeze=True, guard={"flag": True})) is None and forced_probe(_assessment("commit")) is None
    sc = _forced(monkeypatch, [_forced_menu(rec="decline", guard={"flag": True}, without="decline")], [])
    case = run_protocol(SimpleNamespace(abstain_cost=1.0), "sham_once", "table_centre", np.random.default_rng(0))
    assert sc.calls == [] and case["forced_probe"] is None and case["post_probe"] == case["changed"] == "no_probe"
    assert case["ended"] == "abstained" and case["probes"] == []
    sc = _forced(monkeypatch, [_forced_menu(squeeze=True), _assessment("commit"), _assessment("commit")], ["placed_stable"])
    case = run_protocol(SimpleNamespace(abstain_cost=1.0), "sham_once", "table_centre", np.random.default_rng(0))
    assert sc.calls == ["probe_squeeze", "commit_std"] and sc.shams == 1 and case["forced_probe"] == "squeeze"
    assert case["probes"][0]["sham"] is True and case["probes"][0]["belief_hash_before"] == case["probes"][0]["belief_hash_next"]


def test_the_decision_change_categories(monkeypatch):
    """plan H3 decision 6: same (the first menu's decision without probes), another commit row, a decline, terminal (the
    probe crushed the object: the case ended, or the object was restored under the recovery mode) and no probe."""
    from robots.mujoco_lab.calibrate import decision_change, run_protocol

    first = {"without_probe": "commit", "best_row": "commit_std"}
    assert [decision_change(first, p) for p in ("commit_std", "commit_low", "decline", "terminal", "no_probe")] == \
        ["same", "other_row", "decline", "terminal", "no_probe"]
    assert decision_change({"without_probe": "decline", "best_row": "commit_std"}, "decline") == "same"
    assert decision_change({"without_probe": "decline", "best_row": "commit_std"}, "commit_std") == "other_row"
    assert decision_change(first, "unreachable") is None and decision_change(first, "error") is None
    other = dict(_forced_menu(rec="commit"), best_option="commit_low")
    other["options"] = other["options"] + [_row("commit_low")]
    sc = _forced(monkeypatch, [_forced_menu(), other], ["placed_stable"])
    case = run_protocol(SimpleNamespace(abstain_cost=1.0), "probe_once", "table_centre", np.random.default_rng(0))
    assert sc.calls == ["probe_weigh", "commit_low"] and case["post_probe"] == "commit_low" and case["changed"] == "other_row"
    sc = _forced(monkeypatch, [_forced_menu(squeeze=True)], [], damage=True)
    case = run_protocol(SimpleNamespace(abstain_cost=1.0), "probe_once", "table_centre", np.random.default_rng(0))
    assert sc.calls == ["probe_squeeze"] and case["ended"] == "damaged" and case["post_probe"] == case["changed"] == "terminal"
    sc = _forced(monkeypatch, [_forced_menu(squeeze=True), _assessment("commit"), _assessment("commit")], ["placed_stable"],
                 damage=True, restore=True)
    case = run_protocol(SimpleNamespace(abstain_cost=1.0, reset_s=20.0), "sham_once", "table_centre", np.random.default_rng(0))
    assert sc.calls == ["probe_squeeze", "commit_std"] and case["ended"] == "placed" and case["restores"][0]["probe"] == "squeeze"
    assert case["post_probe"] == case["changed"] == "terminal"


@needs_m2b2
def test_the_forced_probe_is_the_same_at_every_recovery_level(tmp_path):
    """The rule reads only the menu's probe rows, which follow the object's public flags: every recovery level forces
    the same physical probe, the squeeze on a crushable object (the jello box), the weigh on a rigid one (the soup can).
    Each output records the rule (plan H3 decision 11)."""
    from robots.mujoco_lab.calibrate import FORCED_RULE, main

    forced, rules = [], []
    for reset in (None, "20", "0"):
        out = tmp_path / f"c{reset}.json"
        main(["--bundle", M2B2, "--repeats", "1", "--out", str(out), "--objects", "ycb__jello,ycb__tomato_soup_can", "--seed", "2",
              "--protocol", "probe_once", "--care", "10", "--abstain-cost", "10"] + ([] if reset is None else ["--reset-s", reset]))
        forced.append({r["object"]: r["forced_probe"] for r in json.load(open(out))["rows"]})
        rules.append(json.load(open(out))["forced_rule"])
    assert forced[0] == forced[1] == forced[2] == {"ycb__jello": "squeeze", "ycb__tomato_soup_can": "weigh"}
    assert rules == [FORCED_RULE] * 3 and "squeeze" in FORCED_RULE and "weigh" in FORCED_RULE


@needs_m2b2
def test_sham_once_leaves_the_belief_unchanged(tmp_path, monkeypatch):
    """Gates Gp and Gs on real cases: the sham's probe trace (kind, reading, thrown, displacement, tilt) and its charge
    equal the real probe's on the same case, and its belief hashes before, after and at the next menu are equal (the
    next menu missing only when the probe ended the case); a real probe carries no sham record. Paired draws: the
    stream state at the probe's next menu is the same in both arms."""
    import rpent.skills.pick_place.tools as tools_mod
    from robots.mujoco_lab.calibrate import main

    run_probe, reassess, inside, states = tools_mod._run_probe, tools_mod._reassess, [], {"probe_once": [], "sham_once": []}
    ess = tools_mod.REFRESH_ESS

    def probe_spy(*a, **k):
        inside.append(json.dumps(a[4].rng.bit_generator.state, sort_keys=True))
        tools_mod.REFRESH_ESS = float("inf")       # the probe's update refreshes, so it draws from primitives.rng
        try:
            return run_probe(*a, **k)
        finally:
            tools_mod.REFRESH_ESS = ess
            inside.pop()

    def reassess_spy(*a):
        if inside:      # the probe's next menu: the stream state at the probe's start and at that menu
            states[arm].append((inside[-1], json.dumps(a[3].rng.bit_generator.state, sort_keys=True)))
        return reassess(*a)

    monkeypatch.setattr(tools_mod, "_run_probe", probe_spy)
    monkeypatch.setattr(tools_mod, "_reassess", reassess_spy)
    rows = {}
    for arm in ("probe_once", "sham_once"):
        out = tmp_path / f"{arm}.json"
        main(["--bundle", M2B2, "--repeats", "2", "--out", str(out), "--objects", "ycb__jello,ycb__tomato_soup_can", "--seed", "2",
              "--protocol", arm, "--care", "10", "--abstain-cost", "10"])
        rows[arm] = {(r["object"], r["idx"]): r for r in json.load(open(out))["rows"]}
    trace, ran = ("probe", "kind", "reading", "thrown", "displacement_m", "tilt_deg", "cost"), 0
    for key, real in rows["probe_once"].items():
        sham = rows["sham_once"][key]
        assert sham["forced_probe"] == real["forced_probe"] and len(sham["probes"]) == len(real["probes"]) == 1
        p, q = real["probes"][0], sham["probes"][0]
        assert {k: p[k] for k in trace} == {k: q[k] for k in trace} and "sham" not in p and q["sham"] is True
        assert q["belief_hash_before"] == q["belief_hash_after"]
        assert q["belief_hash_next"] == q["belief_hash_before"] or q["kind"] == "damaged" or q["thrown"]
        ran += 1
    assert ran == 4 and len(states["probe_once"]) == 4 and all(s != n for s, n in states["probe_once"])
    assert states["sham_once"] == states["probe_once"]
