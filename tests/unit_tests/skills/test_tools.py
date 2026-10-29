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

    obj = {"name": "mug", "id": "ycb__mug", "material": "ceramic"}

    def get_obs(self):
        return {"state": {"object_pos": [0.0, 0.0, 0.8], "object_mat": [1, 0, 0, 0, 1, 0, 0, 0, 1], "hand_pos": [0.0, 0.0, 1.1],
                          "gripper_opening": 0.11, "pad_forces": [0.0, 0.0], "fixtures_in_scene": list(self.fixtures_in_scene)},
                "vision": {}}


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
    assert any(e.startswith("mass median") for e in d["evidence"])      # the particle belief's summary
    key = ("ycb__mug", "table_centre", "normal", 1)
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


def test_assess_only_executes_nothing():
    prims = _prims()
    r = pick_and_place(object="mug", mode="assess_only", primitives=prims, store={}).data
    assert r["assessed"] is True and prims.env.calls == []
    assert "error" in pick_and_place(object="mug", mode="teleport", primitives=prims, store={}).data


def test_failed_outcome_reports_placed_false():
    prims = _prims(env=FakeEnv(label="dropped_high"))
    r = pick_and_place(object="mug", primitives=prims, store={}).data
    assert r["placed"] is False and r["outcome"] == "dropped_high"


def test_cache_keyed_by_object_id_and_cleared_after_execution():
    prims = _prims()
    store = {}
    a1 = assess_pick(object="mug", destination="table_centre", primitives=prims, store=store).data
    a2 = assess_pick(object="object", destination="table_centre", primitives=prims, store=store).data
    assert len(store["pick_place"]) == 1 and a1["best_option"] == a2["best_option"]   # same object under two prompts
    pick_and_place(object="object", destination="table_centre", primitives=prims, store=store)
    assert all(k[0] != "ycb__mug" or k[3] != 1 for k in store["pick_place"]) or len(store["pick_place"]) == 1
    keys_after = list(store["pick_place"])
    assert len(keys_after) == 1 and keys_after[0][:2] == ("ycb__mug", "table_centre")   # only the fresh assessment remains


class _PoseEnv(FakeEnv):
    """FakeEnv with the substrate's pose (the lab adapter's pose_info)."""

    def pose_info(self):
        from tests.unit_tests.skills.synthetic import synthetic_pose

        return synthetic_pose()


class _RecordingPrior:
    """A primitives.prior stub: records (pose, n) and returns a point-mass belief."""

    def __init__(self):
        self.calls = []

    def __call__(self, pose, n, rng):
        from rpent.skills.pick_place.belief import ParticleBelief

        self.calls.append((pose, n))
        parts = [{"mass": 0.2, "mu": 0.6, "com_obj": [0.0, 0.0, 0.03]} for _ in range(n)]
        return ParticleBelief(particles=parts, weights=np.full(n, 1.0 / n))


def test_assess_draws_the_belief_from_the_primitives_prior():
    from rpent.skills.pick_place.belief import N_PARTICLES
    from tests.unit_tests.skills.synthetic import synthetic_pose

    prior = _RecordingPrior()
    prims = SimpleNamespace(env=_PoseEnv(), model=StubModel(), rng=np.random.default_rng(0), prior=prior)
    d = assess_pick(object="mug", destination="table_centre", primitives=prims, store={}).data
    assert len(prior.calls) == 1 and prior.calls[0] == (synthetic_pose(), N_PARTICLES)
    assert any(e.startswith("mass median 0.200 kg") for e in d["evidence"])


def test_assess_without_a_prior_uses_the_parametric_prior_as_particles():
    from rpent.skills.pick_place.belief import N_PARTICLES

    d = assess_pick(object="mug", destination="table_centre", primitives=_prims(), store={}).data
    assert any(f"effective particles of {N_PARTICLES}, 0 observations" in e for e in d["evidence"])
    assert not any(e.startswith("mass median 0.200 kg") for e in d["evidence"])


# ---------------------------------------------------------------- the repair loop (milestone 2, task 6)
from rpent.skills.pick_place import tools as tools_mod  # noqa: E402
from rpent.skills.pick_place.belief import ParticleBelief  # noqa: E402
from rpent.skills.pick_place.likelihood import ReadoutModel  # noqa: E402

G = 9.81
OUT_OF_REACH = {"assessed": False, "reason": "object is out of reach (floor or bin)", "next": "report it; the skill cannot reach it"}
KEY = ("ycb__mug", "table_centre", "normal", 1)


def _gen_prior(pose, n, rng):
    """A synthetic generator prior: mass log-uniform 0.05 to 2 kg, friction uniform 0.1 to 0.9, the CoM
    2 cm either side of the pinch along the object's y and 3 cm up."""
    m = np.exp(rng.uniform(np.log(0.05), np.log(2.0), n))
    u = rng.uniform(0.1, 0.9, n)
    cy = rng.uniform(-0.02, 0.02, n)
    return ParticleBelief([{"mass": float(m[i]), "mu": float(u[i]), "com_obj": [0.0, float(cy[i]), 0.03], "ancestor": i}
                           for i in range(n)], np.ones(n))


class _ReadoutRollout(FakeRollout):
    def __init__(self, label, readouts):
        super().__init__(label)
        self.readouts = readouts


class _ReadingEnv(_PoseEnv):
    """A pose env with proposal poses whose attempts return the queued (label, readouts) in turn."""

    def __init__(self, results):
        super().__init__()
        self.results = list(results)

    def proposal_pose(self, prop):
        return {**self.pose_info(), "pt": [float(x) for x in prop.point], "yaw": float(prop.yaw)}

    def execute(self, proposal, force, v_lift, v_transport=0.5, destination="bin", release_height=None):
        super().execute(proposal, force, v_lift, v_transport, destination, release_height)
        label, readouts = self.results.pop(0)
        return _ReadoutRollout(label, readouts)


def _bprims(env, probe_models=None, model=None):
    rm = ReadoutModel(terms={t: {"_pooled": (0.0, 0.1)} for t in ("hold", "slip", "loss")}, side_error=0.1)
    return SimpleNamespace(env=env, model=model or StubModel(), rng=np.random.default_rng(0), prior=_gen_prior,
                           readout_model=rm, probe_models=probe_models)


def _hold(m):
    return {"fz_hold_n": m * G, "fz_slip_max_n": None, "fz_at_loss_n": None, "rot_deg": 0.0, "heavy_side": 0}


def _commit(prims, store, destination="table_centre", obj="mug"):
    return pick_and_place(object=obj, destination=destination, option_id="commit_std", primitives=prims, store=store).data


def test_pick_payload_carries_readouts_diagnosis_and_beliefs():
    env = _ReadingEnv([("not_lifted", {"fz_slip_max_n": 3.0, "rot_deg": 0.0, "heavy_side": 0, "loss_phase": None})])
    r = _commit(_bprims(env), {})
    assert r["readouts"]["fz_slip_max_n"] == 3.0
    assert r["diagnosis"][0].startswith("never left the table") and r["diagnosis"][-1].startswith("belief: ")
    assert r["belief_before"]["n_observations"] == 0 and r["belief"]["n_observations"] == 1
    assert r["belief"]["mu_median"] < 0.25 < r["belief_before"]["mu_median"]       # 3 N over 2 x 10 N: mu about 0.15


def test_pick_diagnosis_gives_no_friction_outside_the_belief():
    """100 N of slip force at the standard squeeze reads a friction far above the prior's 0.9: the line
    says so and gives no number."""
    env = _ReadingEnv([("not_lifted", {"fz_slip_max_n": 100.0, "rot_deg": 0.0, "heavy_side": 0, "loss_phase": None})])
    r = _commit(_bprims(env), {})
    assert not any("friction about" in s for s in r["diagnosis"]) and any("outside the belief" in s for s in r["diagnosis"])


def test_a_reading_outside_the_belief_is_reported():
    """A hold reading 10x the largest prior mass: the term is rejected, the history lists the terms used
    and rejected, the reading does not move the belief, and the evidence names it."""
    store = {}
    r = _commit(_bprims(_ReadingEnv([("placed_stable", _hold(20.0))])), store)
    e = store["beliefs"]["ycb__mug"]["belief"].history[-1]
    assert e["type"] == "attempt" and e["used_terms"] == ["outcome"] and e["rejected_terms"] == ["hold"]
    assert "reading outside the belief: hold" in r["assessment"]["evidence"]
    assert r["belief"]["mass_80"] == r["belief_before"]["mass_80"]


def test_refresh_replays_only_used_terms(monkeypatch):
    """A term rejected when its reading came in stays out of every later replay. With a refresh after each
    update, the 1.5 kg hold reading (rejected against a belief already at 0.2 kg) does not pull the
    refreshed posterior toward it, though the fresh prior draws reach 1.5 kg."""
    monkeypatch.setattr(tools_mod, "REFRESH_ESS", float("inf"))
    prims = _bprims(_ReadingEnv([("placed_stable", _hold(0.2)), ("placed_stable", _hold(1.5))]))
    store = {}
    _commit(prims, store)
    r = _commit(prims, store)
    b = store["beliefs"]["ycb__mug"]["belief"]
    assert [e["used_terms"] for e in b.history] == [["outcome", "hold"], ["outcome"]] and b.history[1]["rejected_terms"] == ["hold"]
    assert any(f.startswith("refreshed from 20000 prior draws") for f in b.flags)
    assert abs(r["belief"]["mass_median"] - 0.2) < 0.03


def test_belief_is_reset_on_a_new_episode():
    env = _ReadingEnv([("placed_stable", _hold(0.2))])
    prims, store = _bprims(env), {}
    _commit(prims, store)
    rec = store["beliefs"]["ycb__mug"]
    assert rec["episode"] == 1 and len(rec["belief"].history) == 1 and rec["probes_run"] == set()
    env.episode = 2
    d = assess_pick(object="mug", destination="table_centre", primitives=prims, store=store).data
    rec = store["beliefs"]["ycb__mug"]
    assert rec["episode"] == 2 and rec["belief"].history == [] and rec["probes_run"] == set()
    assert any("0 observations" in e for e in d["evidence"])


class _FallEnv(FakeEnv):
    """An env whose object ends on the floor after an attempt (locate says reachable False)."""

    def __init__(self, reachable=True):
        super().__init__(label="dropped_high")
        self.reachable = reachable

    def locate(self, prompt):
        loc = super().locate(prompt)
        return None if loc is None else dict(loc, reachable=self.reachable)

    def execute(self, *a, **kw):
        r = super().execute(*a, **kw)
        self.reachable = False
        return r


def test_object_out_of_reach_is_not_assessed():
    prims = _prims(env=_FallEnv(reachable=False))
    assert assess_pick(object="mug", primitives=prims, store={}).data == OUT_OF_REACH
    assert _commit(prims, {}, destination="bin") == OUT_OF_REACH and prims.env.calls == []
    prims = _prims(env=_FallEnv())
    r = _commit(prims, {}, destination="bin")
    assert r["outcome"] == "dropped_high" and r["assessment"] == OUT_OF_REACH        # the fresh assessment offers no rows


class _TipEnv(_ReadingEnv):
    """After the attempt the object lies on its side: R turns 90 degrees about x."""

    R_SIDE = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]

    def __init__(self):
        super().__init__([("tipped", {})])
        self.R = None

    def pose_info(self):
        pose = super().pose_info()
        return pose if self.R is None else dict(pose, R=list(self.R))

    def execute(self, *a, **kw):
        r = super().execute(*a, **kw)
        self.R = self.R_SIDE
        return r


class _LeverModel(StubModel):
    """StubModel that records the lever feature of every particle it predicts for."""

    def __init__(self):
        super().__init__()
        self.levers = []

    def predict_many(self, feats):
        self.levers.append(np.array([f["lever"] for f in feats]))
        return super().predict_many(feats)


def test_tipped_object_keeps_its_particles():
    """The particles' CoMs stay in the object frame; the fresh menu maps them through the new pose."""
    prims, store = _bprims(_TipEnv(), model=_LeverModel()), {}
    assess_pick(object="mug", destination="table_centre", primitives=prims, store=store)
    com0 = store["beliefs"]["ycb__mug"]["belief"].arrays()["com_obj"].copy()
    r = _commit(prims, store)
    assert r["outcome"] == "tipped"
    assert np.array_equal(store["beliefs"]["ycb__mug"]["belief"].arrays()["com_obj"], com0)
    assert store["pick_place"][KEY]["menu"]["commit_std"]._pose["R"] == _TipEnv.R_SIDE
    assert np.allclose(prims.model.levers[0], np.abs(com0[:, 1]))     # upright: the lever is the CoM's object y
    assert np.allclose(prims.model.levers[-1], com0[:, 2])           # on its side: the object z lies along the hand's y


def test_cache_is_cleared_for_every_destination_after_an_attempt():
    """`mug` and `object` name one object: one belief, and the attempt clears every cached assessment."""
    prims, store = _bprims(_ReadingEnv([("dropped_low", {})])), {}
    assess_pick(object="mug", destination="table_centre", primitives=prims, store=store)
    assess_pick(object="mug", destination="plate", care="high", primitives=prims, store=store)
    assess_pick(object="object", destination="bin", primitives=prims, store=store)
    assert len(store["pick_place"]) == 3 and list(store["beliefs"]) == ["ycb__mug"]
    _commit(prims, store, destination="bin", obj="object")
    assert list(store["pick_place"]) == [("ycb__mug", "bin", "normal", 1)]
    assert len(store["beliefs"]["ycb__mug"]["belief"].history) == 1
    d = assess_pick(object="mug", destination="plate", care="high", primitives=prims, store=store).data
    assert any("1 observations" in e for e in d["evidence"])


def test_probe_entry_updates_the_belief():
    """update_belief takes probe entries too: a measured weigh reading moves the mass, and the entry
    records its terms."""
    from rpent.skills.pick_place.probes import CalibratedClassifier, ProbeModel

    clf = CalibratedClassifier(("measured", "lower_bound", "lost"), None, None, np.array([0.8, 0.15, 0.05]))
    pm = ProbeModel(probe="weigh", classes=clf.classes, clf=clf, reading={"measured": (0.0, 0.05)}, p_thrown=0.0)
    prims, store = _bprims(_ReadingEnv([]), probe_models={"weigh": pm}), {}
    assess_pick(object="mug", destination="table_centre", primitives=prims, store=store)
    o = store["pick_place"][KEY]["menu"]["commit_std"]
    b = store["beliefs"]["ycb__mug"]["belief"]
    entry = {"type": "probe", "base": o._base, "pose": o._pose, "force": 20.0, "planned": None,
             "obs": {"probe": "weigh", "kind": "measured", "reading": {"m_hat": 0.5}}}
    post = tools_mod.update_belief(b, entry, prims)
    assert post.history[-1]["used_terms"] == ["kind", "reading"] and post.history[-1]["rejected_terms"] == []
    assert abs(post.quantile("mass", 0.5) - 0.5) < 0.03 and post.effective_size() >= 100


# ---------------------------------------------------------------- probes and the threshold (milestone 2, task 7)
from rpent.skills.pick_place.probes import ProbeResult  # noqa: E402
from tests.unit_tests.skills.probe_stubs import FlatWeigh, MuGate, PerfectPush  # noqa: E402


class _ProbeEnv(_ReadingEnv):
    """A reading env with the two probes: each returns the next queued ProbeResult; a thrown probe puts
    the object out of reach."""

    def __init__(self, results=(), probes=()):
        super().__init__(results)
        self.probe_results = list(probes)
        self.probe_calls = []
        self.reachable = True

    def locate(self, prompt):
        loc = super().locate(prompt)
        return None if loc is None else dict(loc, reachable=self.reachable)

    def _probe(self, name):
        self.probe_calls.append(name)
        res = self.probe_results.pop(0)
        assert res.probe == name
        if res.thrown:
            self.reachable = False
        return res

    def probe_push(self):
        return self._probe("push")

    def probe_weigh(self, proposal, force):
        self.weigh_args = (proposal.s, force)
        return self._probe("weigh")


def _probe_prims(env, push=None, weigh=None, model=None):
    pms = {k: v for k, v in (("push", push), ("weigh", weigh)) if v is not None}
    return _bprims(env, probe_models=pms or None, model=model or MuGate(f_std=10.0))


def _push(high=1.0, thrown=False):
    return ProbeResult(probe="push", kind="slid", reading={"high": high}, displacement_m=0.03, tilt_deg=1.0, thrown=thrown, seconds=1.5)


def test_the_substrate_pinch_limit_reaches_the_menu():
    """The substrate's max_pinch_width replaces the menu's default: FakeEnv's 8 cm pinches are wider than a 7 cm
    opening, so the skill declines and says why."""
    env = FakeEnv()
    env.max_pinch_width = 0.07
    d = assess_pick(object="mug", primitives=_prims(env), store={}).data
    assert d["recommendation"] == "decline" and [o["id"] for o in d["options"]] == ["decline"]
    assert d["options"][0]["reason"] == "every proposal is wider than the gripper opening"


def test_threshold_reaches_the_menu():
    prims = _prims(model=StubModel(ps=0.5))
    assert assess_pick(object="mug", primitives=prims, store={}).data["recommendation"] == "decline"
    prims.threshold = 0.0
    assert assess_pick(object="mug", primitives=prims, store={}).data["recommendation"] == "commit"


def test_probe_first_runs_probes_then_commits():
    """The push tells mu is above 0.3; then firm_grip is the best row and it runs."""
    env = _ProbeEnv(results=[("placed_stable", {})], probes=[_push(high=1.0)])
    prims, store = _probe_prims(env, push=PerfectPush(), weigh=FlatWeigh()), {}
    a = assess_pick(object="mug", destination="table_centre", primitives=prims, store=store).data
    assert a["recommendation"] == "probe" and a["recommended_option"] == "probe_push" and a["best_option"] == "slow_lift"
    r = pick_and_place(object="mug", destination="table_centre", mode="probe_first", primitives=prims, store=store).data
    assert env.probe_calls == ["push"] and [(p["probed"], p["kind"], p["thrown"]) for p in r["probes"]] == [("push", "slid", False)]
    assert r["option"] == "firm_grip" and r["outcome"] == "placed_stable" and env.calls[0]["force"] == 20.0
    assert store["beliefs"]["ycb__mug"]["probes_run"] == {"push"}
    assert r["belief"]["mu_80"][0] > 0.3                         # the posterior after the push and the attempt


def test_probe_first_keeps_each_probe_s_artifact_flag():
    """mode probe_first: each probe brief says whether the substrate flagged that probe; the top-level flag is the
    commit's only."""
    import dataclasses

    env = _ProbeEnv(results=[("placed_stable", {})], probes=[dataclasses.replace(_push(high=1.0), artifact=True)])
    prims = _probe_prims(env, push=PerfectPush(), weigh=FlatWeigh())
    r = pick_and_place(object="mug", destination="table_centre", mode="probe_first", primitives=prims, store={}).data
    assert [(p["probed"], p["artifact"]) for p in r["probes"]] == [("push", True)] and r["artifact"] is False


def test_probe_first_stops_after_a_thrown_probe():
    env = _ProbeEnv(probes=[_push(thrown=True)])
    prims = _probe_prims(env, push=PerfectPush())
    r = pick_and_place(object="mug", destination="table_centre", mode="probe_first", primitives=prims, store={}).data
    assert r["probed"] == "push" and r["thrown"] is True and r["placed"] is False and len(r["probes"]) == 1
    assert r["assessment"] == OUT_OF_REACH and env.calls == []


def test_probe_budget_persists_across_calls():
    env = _ProbeEnv(results=[("placed_stable", {})], probes=[_push(high=1.0), _push(high=1.0)])
    prims, store = _probe_prims(env, push=PerfectPush(), weigh=FlatWeigh()), {}
    r = pick_and_place(object="mug", destination="table_centre", option_id="probe_push", primitives=prims, store=store).data
    assert r["probed"] == "push" and r["placed"] is False and r["kind"] == "slid" and r["reading"] == {"high": 1.0}
    assert r["displacement_m"] == 0.03 and r["belief"]["n_observations"] == 1 and r["assessment"]["assessed"] is True
    assert all(o["id"] != "probe_push" for o in r["assessment"]["options"])            # a probe runs once per episode
    again = pick_and_place(object="mug", destination="table_centre", option_id="probe_push", primitives=prims, store=store).data
    assert "already ran" in again["error"] and env.probe_calls == ["push"]
    r = pick_and_place(object="mug", destination="table_centre", mode="probe_first", primitives=prims, store=store).data
    assert env.probe_calls == ["push"] and r["probes"] == [] and r["option"] == "firm_grip"
    env.episode = 2
    d = assess_pick(object="mug", destination="table_centre", primitives=prims, store=store).data
    assert any(o["id"] == "probe_push" for o in d["options"]) and store["beliefs"]["ycb__mug"]["probes_run"] == set()


def test_probe_option_without_models_returns_error():
    prims = _bprims(_ReadingEnv([("placed_stable", {})]))
    assert pick_and_place(object="mug", option_id="probe_weigh", primitives=prims, store={}).data == {"error": "probe models are not loaded"}
    d = assess_pick(object="mug", destination="table_centre", primitives=prims, store={}).data
    assert not any(o["kind"] == "probe" for o in d["options"])
    r = pick_and_place(object="mug", destination="table_centre", mode="probe_first", primitives=prims, store={}).data
    assert r["outcome"] == "placed_stable" and len(prims.env.calls) == 1               # probe_first acts as commit


def test_commit_mode_with_a_probe_recommendation():
    """mode commit never probes: it runs the best row when that reaches the threshold, else nothing runs
    and the result points to the probe."""
    env = _ProbeEnv(results=[("placed_stable", {})])
    prims = _probe_prims(env, push=PerfectPush())
    r = pick_and_place(object="mug", destination="table_centre", primitives=prims, store={}).data
    assert r["option"] == "slow_lift" and env.probe_calls == [] and len(env.calls) == 1
    prims.threshold = 1.01
    r = pick_and_place(object="mug", destination="table_centre", primitives=prims, store={}).data
    assert r["placed"] is False and r["recommendation"] == "probe" and "probe_push" in r["next"] and len(env.calls) == 1


def test_the_payloads_say_when_a_run_was_an_artifact():
    """A substrate may flag a rollout or a probe whose contacts were not physical (the mujoco_lab monitor); the
    payload carries the flag, False when the substrate gives none."""
    import dataclasses

    assert _commit(_prims(), {})["artifact"] is False
    env = FakeEnv()
    plain = env.execute

    def flagged(*a, **k):
        r = plain(*a, **k)
        r.artifact = True
        return r

    env.execute = flagged
    assert _commit(_prims(env), {})["artifact"] is True
    penv = _ProbeEnv(probes=[dataclasses.replace(_push(), artifact=True)])
    d = pick_and_place(object="mug", destination="table_centre", option_id="probe_push", primitives=_probe_prims(penv, push=PerfectPush()),
                       store={}).data
    assert d["probed"] == "push" and d["artifact"] is True
