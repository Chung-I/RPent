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

"""assess_pick and pick_and_place called directly with a fake env and a stub model."""

from types import SimpleNamespace

import numpy as np
import pytest

from rpent.skills.pick_place.features import Geometry
from rpent.skills.pick_place.protocol import OUTCOMES, Prediction
from rpent.skills.pick_place.tools import assess_pick, pick_and_place


class FakeRollout:
    def __init__(self, label, zone="table"):
        self.label, self.zone = label, zone
        self.max_z, self.drop_z, self.loss_phase, self.loss_step, self.loss_t, self.loss_speed = 0.2, None, None, None, None, None
        self.release = {"tilt_deg": 1.0, "displacement_xy_m": 0.0}
        self.planned, self.trace = [], []


class FakeEnv:
    def __init__(self, label="placed_stable"):
        self.episode = 1
        self.label = label
        self.calls = []
        self.fixtures_in_scene = ["mouse_pad", "plate", "box"]

    def locate(self, prompt):
        if prompt not in ("mug", "object"):
            return None
        return {"object_id": "ycb__mug", "centroid": [0.0, 0.0, 0.8], "extent": 0.1, "volume": 2e-4, "material": "ceramic", "hollow": True}

    def proposals(self):
        return [SimpleNamespace(s=s, point=[0.0, 0.0, 0.8], yaw=0.0, width=0.08) for s in (0.3, 0.5, 0.7)]

    def geometry(self):
        return Geometry(centres=np.zeros((2, 3)), normals=np.array([[1.0, 0, 0], [-1.0, 0, 0]]), com=np.array([0.0, 0.0, 0.81]),
                        length=0.1, base_extent=0.03, com_height=0.04, footprint_half_width=0.05)

    def plan_trajectory(self, proposal, v_lift, v_transport, destination, release_height=None):
        return [{"phase": "lift", "n_steps": 10, "z_start": 0.0, "z_end": 0.2, "has_surface": True, "speed": v_lift}]

    def execute(self, proposal, force, v_lift, v_transport=0.5, destination="bin", release_height=None):
        self.calls.append({"s": proposal.s, "force": force, "v_lift": v_lift, "v_transport": v_transport,
                           "destination": destination, "release_height": release_height})
        return FakeRollout(self.label)

    def nominal_mass(self, object_id):
        return 0.3


class StubModel:
    kind = "classifier"

    def __init__(self, ps=0.8):
        self.ps = ps

    def predict_many(self, feats):
        out = []
        for f in feats:
            p = {k: 0.0 for k in OUTCOMES}
            ps = self.ps if f["v_lift"] > 0.1 else self.ps - 0.1
            p.update({"placed_stable": ps, "dropped_high": (1 - ps) * 0.5, "dropped_low": (1 - ps) * 0.5})
            out.append(Prediction(p=p))
        return out


def _prims(env=None, model=None):
    return SimpleNamespace(env=env or FakeEnv(), model=model or StubModel(), rng=np.random.default_rng(0))


def test_assess_unknown_object():
    d = assess_pick(object="spoon", primitives=_prims(), store={}).data
    assert d["assessed"] is False and d["reason"] == "object not found" and "next" in d


def test_assess_unknown_destination():
    d = assess_pick(object="mug", destination="shelf", primitives=_prims(), store={}).data
    assert "error" in d and "shelf" in d["error"]


def test_assess_returns_protocol_keys_and_caches():
    store = {}
    prims = _prims()
    d = assess_pick(object="mug", destination="table_centre", primitives=prims, store=store).data
    assert d["assessed"] is True and d["best_option"] == "commit_std" and d["p_success"] == 0.8
    assert d["recommendation"] == "commit" and d["options"][-1]["id"] == "decline"
    assert any("prior mass" in e for e in d["evidence"])
    key = ("mug", "table_centre", "normal", 1)
    plate = assess_pick(object="mug", destination="plate", primitives=prims, store=store).data
    assert plate["best_option"] == "drop_release"      # same probabilities from the stub, so the faster row wins
    assert key in store["pick_place"]


def test_cache_invalidates_on_episode_and_care():
    store = {}
    prims = _prims()
    assess_pick(object="mug", primitives=prims, store=store)
    assess_pick(object="mug", care="high", primitives=prims, store=store)
    prims.env.episode = 2
    assess_pick(object="mug", primitives=prims, store=store)
    assert {k[2:] for k in store["pick_place"]} == {("normal", 1), ("high", 1), ("normal", 2)}


def test_pick_executes_best_same_destination_row_by_default():
    prims = _prims()
    r = pick_and_place(object="mug", destination="table_edge", primitives=prims, store={}).data
    assert r["placed"] is True and r["placed_at_requested"] is True and r["outcome"] == "placed_stable"
    assert prims.env.calls[0]["destination"] == "table_edge" and prims.env.calls[0]["s"] == 0.5
    assert r["option"] == "commit_std" and r["declared"]["p_success"] == 0.8 and r["assessment"]["assessed"] is True


def test_pick_inboard_place_only_when_selected_and_reports_placed_at_requested_false():
    prims = _prims()
    r = pick_and_place(object="mug", destination="table_edge", option_id="inboard_place", primitives=prims, store={}).data
    assert prims.env.calls[0]["destination"] == "table_centre"
    assert r["placed"] is True and r["placed_at_requested"] is False and r["destination_used"] == "table_centre"


def test_pick_unknown_option_id_is_error_and_does_not_execute():
    prims = _prims()
    r = pick_and_place(object="mug", option_id="teleport", primitives=prims, store={}).data
    assert "error" in r and "teleport" in r["error"] and prims.env.calls == []


def test_pick_decline_row_does_not_execute():
    prims = _prims()
    r = pick_and_place(object="mug", option_id="decline", primitives=prims, store={}).data
    assert r["placed"] is False and "reason" in r and "next" in r and prims.env.calls == []


def test_pick_declines_when_no_row_reaches_threshold_and_no_option_given():
    prims = _prims(model=StubModel(ps=0.5))
    r = pick_and_place(object="mug", primitives=prims, store={}).data
    assert r["placed"] is False and r["recommendation"] == "decline" and prims.env.calls == []
    r2 = pick_and_place(object="mug", option_id="commit_std", primitives=prims, store={}).data
    assert r2["outcome"] == "placed_stable" and len(prims.env.calls) == 1


def test_declared_is_the_chosen_rows_snapshot():
    prims = _prims()
    r = pick_and_place(object="mug", option_id="slow_lift", primitives=prims, store={}).data
    assert r["option"] == "slow_lift" and abs(r["declared"]["p_success"] - 0.7) < 1e-9


def test_probe_first_is_not_available_and_assess_only_executes_nothing():
    prims = _prims()
    assert "error" in pick_and_place(object="mug", mode="probe_first", primitives=prims, store={}).data
    r = pick_and_place(object="mug", mode="assess_only", primitives=prims, store={}).data
    assert r["assessed"] is True and prims.env.calls == []


def test_failed_outcome_reports_placed_false():
    prims = _prims(env=FakeEnv(label="dropped_high"))
    r = pick_and_place(object="mug", primitives=prims, store={}).data
    assert r["placed"] is False and r["outcome"] == "dropped_high"
