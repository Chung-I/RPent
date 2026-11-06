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

"""Frozen decision regression and task-unit score contracts, using offline fixtures."""

import copy
import json

import numpy as np
import pytest

from rpent.skills.pick_place import costs
from rpent.skills.pick_place.protocol import (
    IRREVERSIBLE,
    OUTCOMES,
    RECOVERABLE,
    Prediction,
    _r,
)
from rpent.skills.pick_place.tools import assess_pick, pick_and_place
from tests.unit_tests.skills.make_score_golden import GOLDEN, build_case, snapshot
from tests.unit_tests.skills.probe_stubs import PerfectPush
from tests.unit_tests.skills.test_tools import _aprims, _probe_prims, _ProbeEnv, _push

GOLDENS = json.loads(GOLDEN.read_text())


@pytest.mark.parametrize("golden", GOLDENS, ids=lambda g: str(g["case"]))
def test_frozen_decisions_and_row_scores(golden):
    a = build_case(golden["case"])
    actual, want = snapshot(a), golden["result"]
    for key in (
        "recommendation",
        "best_option",
        "recommended_option",
        "order",
        "without_probe",
    ):
        assert actual[key] == want[key]
    for row in a.options:
        expected = want["expected_cost"][row.id]
        if expected is None:
            assert row.expected_cost is None
        else:
            assert row.expected_cost == pytest.approx(expected, abs=1e-9, rel=0)
        if a.abstain_cost is None:
            assert row.score is None
        else:
            assert row.score == pytest.approx(
                (a.abstain_cost - row.expected_cost) / a.abstain_cost, abs=1e-12, rel=0
            )
            if row.kind == "decline":
                assert row.score == 0
        if row.probe:
            for key in ("evoi", "evoi_se"):
                assert row.probe[key] == want["probes"][row.id][key]
            for new, old in (("value", "evoi"), ("value_se", "evoi_se")):
                assert row.probe[new] == (
                    None if a.abstain_cost is None else row.probe[old] / a.abstain_cost
                )
                assert row.to_data()["probe"][new] == _r(row.probe[new])
    data = a.to_data()
    assert data["alpha"] == (
        None
        if a.abstain_cost is None
        else costs.irreversible_cost(a.care, a.reset_s, a.reset_price) / a.abstain_cost
    )
    assert data["beta"] == (None if a.abstain_cost is None else 1 / a.abstain_cost)
    assert [r["score"] for r in data["options"]] == [_r(r.score) for r in a.options]
    if a.abstain_cost is None:
        # All pre-existing fields, including reasons and probe records, remain identical.
        data.pop("alpha")
        data.pop("beta")
        for row in data["options"]:
            row.pop("score")
            if row["probe"]:
                assert row["probe"].pop("value") is None
                assert row["probe"].pop("value_se") is None
        assert data == want["m2a_output"]


@pytest.mark.parametrize(
    "reset_s,price", [(None, "time"), (0, "time"), (12, "time"), (12, "care")]
)
def test_score_algebra_and_particles(reset_s, price):
    rng = np.random.default_rng(42)
    preds = [
        Prediction(dict(zip(OUTCOMES, row)))
        for row in rng.dirichlet(np.ones(len(OUTCOMES)), 50)
    ]
    for care in (0.5, 1, 3, 30, 100):
        for A in (1, 3, 10):
            alpha = costs.alpha(care, A, reset_s, price)
            beta = costs.beta(A)
            assert alpha == costs.irreversible_cost(care, reset_s, price) / A
            assert beta == 1 / A
            seconds = float(rng.uniform(0, 30))
            scores = []
            for pred in preds:
                c = sum(pred.p[k] * costs.COST[k] for k in RECOVERABLE) + 0.05 * seconds
                want = (
                    pred.p["placed_stable"]
                    - alpha * sum(pred.p[k] for k in IRREVERSIBLE)
                    - beta * c
                )
                value = costs.score(pred, seconds, care, A, reset_s, price)
                assert value == pytest.approx(want, abs=1e-9, rel=0)
                assert value == pytest.approx(
                    (A - costs.expected_cost(pred, seconds, care, A, reset_s, price))
                    / A,
                    abs=1e-12,
                    rel=0,
                )
                scores.append(value)
            assert costs.particle_scores(
                preds, seconds, care, A, reset_s, price
            ) == pytest.approx(scores, abs=1e-12, rel=0)
            assert costs.particle_scores(
                preds, seconds, care, A, reset_s, price
            ) == pytest.approx(
                (A - costs.particle_costs(preds, seconds, care, A, reset_s, price)) / A,
                abs=1e-12,
                rel=0,
            )


def _assert_planner_view(result):
    record = copy.deepcopy(result.data)
    view = json.loads(result.to_text())

    def check(value):
        if isinstance(value, dict):
            assert not {"expected_cost", "expected_cost_after"} & value.keys()
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)

    check(view)
    menu = view.get("assessment", view)
    assert menu["alpha"] == 5 and menu["beta"] == 1
    assert "give up scores 0" in menu["score_rule"]
    for row in menu["options"]:
        assert {
            "id",
            "kind",
            "action",
            "p_success",
            "p_damage",
            "seconds",
            "score",
            "p",
        } <= row.keys()
        if row["p"]:
            assert row["p_success"] == row["p"]["placed_stable"]
            assert row["p_damage"] == pytest.approx(
                sum(row["p"][k] for k in IRREVERSIBLE), abs=0.003
            )
        if row["probe"]:
            assert {"value", "value_se"} <= row["probe"].keys()
    assert result.data == record
    assert "expected_cost" in json.dumps(result.to_dict())
    return view


def test_assess_and_execution_text_hide_costs_but_records_keep_them():
    prims = _aprims()
    _assert_planner_view(
        assess_pick(
            object="mug", destination="table_centre", primitives=prims, store={}
        )
    )
    _assert_planner_view(
        pick_and_place(
            object="mug",
            destination="table_centre",
            mode="assess_only",
            primitives=prims,
            store={},
        )
    )
    _assert_planner_view(
        pick_and_place(
            object="mug",
            destination="table_centre",
            option_id="commit_std",
            primitives=prims,
            store={},
        )
    )


def test_probe_execution_text_hides_nested_costs():
    prims = _probe_prims(_ProbeEnv(probes=[_push()]), push=PerfectPush())
    prims.abstain_cost = 1.0
    _assert_planner_view(
        pick_and_place(
            object="mug",
            destination="table_centre",
            option_id="probe_push",
            primitives=prims,
            store={},
        )
    )


def test_planner_projection_survives_state_capture(tmp_path):
    from rpent.planner.utils.http_mcp_server import mcp_result
    from rpent.skills.pick_place.protocol import planner_view
    from rpent.tools import tool
    from tests.unit_tests.rpent.tools.test_toolkit_contracts import _ContractToolkit

    result = assess_pick(
        object="mug", destination="table_centre", primitives=_aprims(), store={}
    )
    toolkit = _ContractToolkit(tmp_path)

    @tool
    def assess_action():
        """Return a scored assessment as an action with state capture."""
        return result

    toolkit.add_tool(assess_action)
    observed = toolkit.execute_tool("assess_action", {})
    assert json.loads(mcp_result(observed)["content"][0]["text"]) == planner_view(
        observed.data
    )
    # The fixture's observation omits the action payload; its stored record retains it.
    assert toolkit.capture_calls[0]["result"] == result.to_dict()
    assert "expected_cost" in json.dumps(toolkit.capture_calls[0]["result"])
    assert observed.planner_view is planner_view


@pytest.mark.parametrize("A", [1.0, 3.0, 10.0])
@pytest.mark.parametrize(
    "best_cost,probe_cost,se", [(0.4, 0.2, 0.1), (0.4, 0.3, 0.05), (1.2, 0.8, 0.1)]
)
def test_strict_probe_margin_matches_original_floating_decision(
    A, best_cost, probe_cost, se
):
    from rpent.skills.pick_place.menu import EVOI_Z, _cost_rule
    from rpent.skills.pick_place.protocol import Option

    # Adjacent floats exercise both sides of the strict Monte Carlo margin, including decline as continuation.
    for error in (np.nextafter(se, 0.0), se, np.nextafter(se, np.inf)):
        best = Option(
            "best",
            "commit",
            {},
            Prediction({"placed_stable": 1.0}),
            8.0,
            best_cost,
            score=(A - best_cost) / A,
        )
        probe = Option(
            "probe",
            "probe",
            {},
            None,
            2.0,
            probe_cost,
            score=(A - probe_cost) / A,
            probe={
                "evoi": 1.0,
                "evoi_se": error,
                "value": 1.0 / A,
                "value_se": error / A,
            },
        )
        want = (
            "probe"
            if min(best_cost, A) - probe_cost > EVOI_Z * error
            else "commit"
            if best_cost <= A
            else "decline"
        )
        assert _cost_rule([best, probe], best, 1.0, A, None)[0] == want
