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

"""Capture the pre-score decision contract. Run as a module from the repository root.

The JSON is a frozen oracle; tests rebuild these inputs but never regenerate it.
"""

import itertools
import json
from pathlib import Path

import numpy as np

from rpent.skills.pick_place.menu import build_menu, without_probe
from rpent.skills.pick_place.protocol import Assessment
from tests.unit_tests.skills.probe_stubs import FlatWeigh, MuGate, PerfectPush
from tests.unit_tests.skills.test_menu import (
    FixedModel,
    P,
    StubModel,
    _geometry,
    _particles,
    plan_fn,
)
from tests.unit_tests.skills.test_tools import StubModel as ToolStub

GOLDEN = Path(__file__).with_name("data") / "score_golden.json"


def cases():
    for care, A, mode, probes, seed in itertools.product(
        [0.5, 1, 3, 30, 100],
        [None, 1, 3, 10],
        ["absorbing", "time", "care"],
        [False, True],
        [0, 7, 19],
    ):
        yield {
            "model": "mu",
            "care": care,
            "A": A,
            "mode": mode,
            "probes": probes,
            "seed": seed,
        }
    for model, care, A, seed in itertools.product(
        ["menu_stub", "tool_stub"], [0.5, 1, 3, 30, 100], [None, 1, 3, 10], [0, 7, 19]
    ):
        yield {
            "model": model,
            "care": care,
            "A": A,
            "mode": "absorbing",
            "probes": False,
            "seed": seed,
        }
    for care, A, mode, delta in itertools.product(
        [0.5, 1, 3, 30, 100],
        [1, 3, 10],
        ["absorbing", "time", "care"],
        [-1e-10, 0, 1e-10],
    ):
        yield {
            "model": "boundary",
            "care": care,
            "A": A,
            "mode": mode,
            "probes": False,
            "seed": 7,
            "delta": delta,
        }


def build_case(case):
    rng = np.random.default_rng(case["seed"])
    belief = _particles(rng.uniform(0.1, 0.9, 32), np.ones(32))
    model = {
        "mu": lambda: MuGate(f_std=20.0),
        "menu_stub": lambda: StubModel({0.15: (0.75, 0.2), 0.05: (0.72, 0.05)}),
        "tool_stub": ToolStub,
        "boundary": lambda: FixedModel(
            placed_stable=0.9 / (case["A"] + 0.5) + case["delta"],
            not_lifted=1 - 0.9 / (case["A"] + 0.5) - case["delta"],
        ),
    }[case["model"]]()
    reset_s = None if case["mode"] == "absorbing" else 12.0
    price = "care" if case["mode"] == "care" else "time"
    options, rec = build_menu(
        model,
        belief,
        _geometry(),
        [P(0.3), P(0.5), P(0.7)],
        20.0,
        case["care"],
        "table_centre",
        plan_fn,
        rng,
        abstain_cost=case["A"],
        reset_s=reset_s,
        reset_price=price,
        probe_models={"push": PerfectPush(), "weigh": FlatWeigh()}
        if case["probes"]
        else None,
    )
    no_probe = without_probe(
        options, rec, "table_centre", case["care"], case["A"], None, reset_s, price
    )
    return Assessment(
        "object",
        "table_centre",
        options,
        [],
        rec,
        case["care"],
        1.0,
        1,
        abstain_cost=case["A"],
        without_probe=no_probe,
        reset_s=reset_s,
        reset_price=price,
    )


def snapshot(a):
    data = a.to_data()
    return dict(
        recommendation=a.recommendation,
        best_option=data["best_option"],
        recommended_option=data["recommended_option"],
        order=[o.id for o in a.options],
        expected_cost={o.id: o.expected_cost for o in a.options},
        probes={
            o.id: {k: o.probe[k] for k in ("evoi", "evoi_se")}
            for o in a.options
            if o.probe
        },
        without_probe=a.without_probe,
        **({"m2a_output": data} if a.abstain_cost is None else {}),
    )


if __name__ == "__main__":
    rows = [{"case": c, "result": snapshot(build_case(c))} for c in cases()]
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(rows, indent=2) + "\n")
    print(
        f"Saved {len(rows)} menus; {sum(r['result']['recommendation'] == 'probe' for r in rows)} recommend a probe"
    )
