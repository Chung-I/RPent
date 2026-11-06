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

"""TODO 6 step 1, the oracle gap for the CoM at point (b) (com_gap; daily-logs 2026-11-05-com-oracle-gap, Amendments 1
and 2): the lift-and-hold fit, the conditioned beliefs, the choices at (b), the captured scene, the policy up to its
first attempt, the summary, the shard merge and the stop rule."""

import json
import os
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from robots.mujoco_lab import com_gap as cg
from rpent.skills.pick_place.belief import ParticleBelief
from rpent.skills.pick_place.destinations import DESTINATIONS
from rpent.skills.pick_place.features import Geometry
from rpent.skills.pick_place.menu import A_GENTLE, STD, _rows, build_menu
from rpent.skills.pick_place.protocol import OUTCOMES, Prediction

DATA = "robots/mujoco_lab/data/"


# ---------------------------------------------------------------- the lift-and-hold fit

def _trace():
    z = [("approach", 0.12), ("close", 0.10), ("close", 0.10), ("lift", 0.11), ("lift", 0.125), ("lift", 0.13), ("lift", 0.16),
         ("hold", 0.30), ("hold", 0.30), ("transport", 0.30), ("transport", 0.30), ("lower", 0.2)]
    return [{"phase": p, "hand_z": h} for p, h in z]


def test_grasp_steps_are_the_attempt_window_without_the_carry():
    """The adapter's attempt window (the lift once the hand rose JOINT_RISE_M above where it closed, the hold and the
    transport) less the transport."""
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    seen = []
    fake = SimpleNamespace(torque_noise=0.0, _joint_fit=lambda log, steps, params: (seen.append(list(steps)), {})[1:] + (None,))
    MuJoCoLabEnv._attempt_joint(fake, None, _trace())
    assert seen == [[5, 6, 7, 8, 9, 10]]
    assert cg.grasp_steps(_trace()) == [5, 6, 7, 8]
    assert cg.grasp_steps([{"phase": "lift", "hand_z": 0.2}]) == []          # no close: no window


class _FitEnv:
    torque_noise = 0.1

    def __init__(self):
        self.fits = []

    def _joint_fit(self, log, steps, params):
        self.fits.append((log, list(steps), params))
        return {"steps": list(steps), "params": params, "floor": [1.0]}, {"torque_noise": self.torque_noise}

    def _attempt_joint(self, log, trace):
        return {"attempt": True}, {"torque_noise": self.torque_noise}


class _Log:
    """ident.Log's fields: per sample t, q, tau, fingers, mark, step (the trace index), tau_clean and aux; per tagged
    control step {"phase", "i", "pose", "end"}. Each trace step gets `sub` samples; `after` is added to the torques of
    the steps after the hold (the carry and what follows)."""

    def __init__(self, trace, sub=4, after=0.0):
        self.t, self.q, self.tau, self.fingers, self.mark, self.step, self.tau_clean, self.aux, self.steps = ([] for _ in range(9))
        last_hold = max(i for i, x in enumerate(trace) if x["phase"] == "hold")
        for i, x in enumerate(trace):
            for _ in range(sub):
                k = len(self.t)
                tau = np.array([np.sin(0.3 * k), np.cos(0.2 * k)]) + (after if i > last_hold else 0.0)
                self.t.append(0.002 * k)
                self.q.append(np.zeros(2))
                self.tau.append(tau)
                self.tau_clean.append(tau)
                self.fingers.append(np.zeros(2))
                self.aux.append({"qd": np.zeros(2)})
                self.mark.append(x["phase"])
                self.step.append(i)
            self.steps.append({"phase": x["phase"], "i": i, "pose": None, "end": len(self.t)})

    def arrays(self):
        return (np.asarray(self.t), np.asarray(self.q), np.asarray(self.tau), np.asarray(self.fingers),
                np.asarray(self.mark, dtype=object))

    def step_index(self):
        return np.asarray(self.step)

    def __len__(self):
        return len(self.t)


class _FilterEnv:
    """_joint_fit as ident.segment reads the log: the whole log low-passed with a zero-phase filter, then the window cut.
    The attempt's readout reports how many samples it read."""

    def _joint_fit(self, log, steps, params):
        from scipy.signal import iirfilter, sosfiltfilt

        sos = iirfilter(N=4, Wn=0.2, btype="lowpass", ftype="butter", output="sos")
        tau = sosfiltfilt(sos, log.arrays()[2], axis=0)
        return {"tau": tau[np.isin(log.step_index(), sorted(steps))].tolist(), "floor": [1.0]}, {}

    def _attempt_joint(self, log, trace):
        return {"samples": len(log)}, {}


def test_the_lift_and_hold_fit_reads_no_sample_after_the_hold():
    """ident.segment low-passes the whole log (zero phase) before it cuts the window, so a fit on the whole log would
    let the carry's torques reach the hold's last samples: point (b) comes before the carry. The fit gets the log up to
    the hold's last sample; the attempt's own readout still reads the whole log."""
    fits = []
    for after in (0.0, 5.0):
        log, env = _Log(_trace(), after=after), _FilterEnv()
        with cg.GraspTap(env) as tap:
            out = env._attempt_joint(log, _trace())
        assert out == ({"samples": len(log)}, {}) and len(log) == 48
        fits.append((np.asarray(tap.fit["tau"]), np.asarray(tap.fit_whole_log["tau"])))
    assert fits[0][0].shape == (16, 2) and np.allclose(fits[0][0], fits[1][0])
    assert not np.allclose(fits[0][1], fits[1][1])          # the check's whole-log fit reads the carry, as the adapter's does


def test_cut_log_ends_the_log_at_the_window_and_holds_its_last_sample():
    """The cut log: every sample up to the window's last one, then hold_s of copies of that sample (the arm keeps
    holding; step -1, so no window reads them), so the zero-phase filter has no edge inside the window and reads
    nothing after (b). No tagged step after the window; the log itself is unchanged."""
    log = _Log(_trace())
    cut = cg.cut_log(log, [5, 6, 7, 8], hold_s=0.01)              # 0.01 s at the fake's 2 ms samples: 5 copies
    assert len(log) == 48 and len(log.steps) == 12                    # the log itself is unchanged
    assert len(cut) == 36 + 5 and [x["i"] for x in cut.steps] == list(range(9))
    assert cut.step[:36] == log.step[:36] and cut.step[36:] == [-1] * 5 and max(cut.step) == 8
    assert np.allclose(cut.t[36:], log.t[35] + 0.002 * np.arange(1, 6))
    for name in ("q", "tau", "fingers", "tau_clean"):
        assert all(np.array_equal(x, getattr(log, name)[35]) for x in getattr(cut, name)[36:])
    assert all(np.array_equal(a["qd"], log.aux[35]["qd"]) for a in cut.aux[36:]) and len(cut.aux) == 41
    assert cut.mark[36:] == [None] * 5 and len(cut.arrays()[2]) == 41
    assert len(cg.cut_log(log, [5, 6, 7, 8])) == 36 + round(cg.CUT_HOLD_S / 0.002)
    assert cg.cut_log(log, []) is log        # no window: the log as it is (the fit then has no window either)


def test_grasp_tap_records_the_lift_and_hold_fit_and_keeps_the_attempt_readout():
    env, log = _FitEnv(), _Log(_trace())
    with cg.GraspTap(env) as tap:
        assert tap.fit is None and tap.calls == 0
        out = env._attempt_joint(log, _trace())
    assert out == ({"attempt": True}, {"torque_noise": 0.1})
    assert tap.fit == {"steps": [5, 6, 7, 8], "params": 4, "floor": [1.0]} and tap.calls == 1
    assert len(env.fits[0][0]) == 36 + round(cg.CUT_HOLD_S / 0.002)
    assert "_attempt_joint" not in vars(env) and env._attempt_joint("LOG", []) == ({"attempt": True}, {"torque_noise": 0.1})
    with pytest.raises(RuntimeError):
        with cg.GraspTap(env):
            raise RuntimeError("boom")
    assert "_attempt_joint" not in vars(env)


# ---------------------------------------------------------------- the conditioned beliefs (Amendment 2)

def test_truth_loglik_uses_the_exact_sds():
    from robots.mujoco_lab.vision_prior import EXACT_SD

    a = {"mass": np.array([1.0, np.exp(EXACT_SD["mass"]), 1.0]), "com_frac": np.array([0.5, 0.5, 0.5 + EXACT_SD["com"]])}
    t = {"mass": 1.0, "com_frac": 0.5}
    assert np.allclose(cg.truth_loglik(a, t, ("mass",)), [0.0, -0.5, 0.0])
    assert np.allclose(cg.truth_loglik(a, t, ("com",)), [0.0, 0.0, -0.5])
    assert np.allclose(cg.truth_loglik(a, t, ("mass", "com")), [0.0, -0.5, -0.5])
    with pytest.raises(ValueError):
        cg.truth_loglik(a, t, ("mu",))


def _wmed(v, w):
    o = np.argsort(v)
    c = np.cumsum(w[o]) / w[o].sum()
    return float(v[o][np.searchsorted(c, 0.5)])


def _overwrite_com(b, frac, pose):
    """The design Amendment 2 rejects: every particle's CoM set to the truth, its fill and spill angle kept."""
    pv = pose["pivot"]
    point = (np.asarray(pv["centre"]) + (float(pv["lo"]) + frac * float(pv["L"])) * np.asarray(pv["axis"])).tolist()
    return ParticleBelief([dict(p, com_obj=point, com_frac=frac) for p in b.particles], b.weights.copy())


def test_belief_two_gives_the_spill_angle_of_the_true_fill():
    """Amendment 2's unit test: for a fillable object (ycb__mug, filled 0.6, the truth's mass and CoM from that fill),
    belief 2's spill angle matches the generator's at the true fill; an overwrite of the CoM fails the same check."""
    from robots.mujoco_lab.fragility import Thresholds, entry_of, load_classes, spill_angle
    from robots.mujoco_lab.prior import joint_hidden, make_prior

    classes = load_classes(DATA + "fragility_classes_v11c.json")
    prior = make_prior(classes, Thresholds.from_json(DATA + "thresholds_v1.json"), "v11")
    entry = entry_of("ycb__mug", classes)[0]
    pose = {"object_id": "ycb__mug", "pivot": {"centre": [0.0, 0.0, 0.0], "axis": [0.0, 0.0, 1.0], "lo": -0.05, "L": 0.1},
            "com_obj_nominal": [0.0, 0.0, 0.0], "nominal_mass": 0.12}
    b0 = prior(pose, 20000, np.random.default_rng(0))
    h = joint_hidden(pose, entry, {"mass_scale": 1.2, "com_s": None}, 1, 0.6, "v11")
    frac = float((np.asarray(h["com_obj"]) @ np.array([0.0, 0.0, 1.0]) + 0.05) / 0.1)
    want = float(spill_angle(entry["cavity"], 0.6))
    a = b0.arrays()
    b1 = b0.reweight(-0.5 * ((np.log(a["mass"]) - np.log(h["mass"])) / 0.03) ** 2)     # a mass reading, as a fit gives
    b2, info = cg.condition(b1, [cg.truth_entry({"mass": h["mass"], "com_frac": frac}, ("com",))], cg.replay_fn(None))
    f = a["fill"] > 0
    assert info["ess"] > 100 and not info["refreshed"] and b2.weights[f].sum() > 0.99
    assert abs(_wmed(a["spill_angle_deg"][f], b2.weights[f]) - want) < 1.5
    assert abs(_wmed(a["spill_angle_deg"][f], b1.weights[f]) - want) > 4
    over = _overwrite_com(b1, frac, pose)
    assert abs(_wmed(over.arrays()["spill_angle_deg"][f], over.weights[f]) - want) > 4


def _toy_prior(n, rng):
    """Mass log-normal about 1 kg (log-sd 0.05), CoM fraction U(0.55, 0.65) along y, the CoM height U(0.02, 0.04)."""
    m, c, z = np.exp(0.05 * rng.standard_normal(n)), rng.uniform(0.55, 0.65, n), rng.uniform(0.02, 0.04, n)
    parts = [{"mass": float(m[i]), "mu": 0.5, "com_obj": [0.0, float(c[i]) * 0.1 - 0.05, float(z[i])], "com_frac": float(c[i]),
              "ancestor": i} for i in range(n)]
    return ParticleBelief(parts, np.ones(n))


def _fit_at(mass, com_obj, sd=(0.2, 1.0, 1.0, 0.002)):
    """A lift-and-hold fit reading in identify_fit's form: link 7's frame the object's, a diagonal information."""
    phi = [mass] + [mass * x for x in com_obj] + [0.0] * 6
    return {"params": 4, "phi": phi, "info": np.diag([1.0 / s ** 2 for s in sd]).tolist(), "floor": [1.0],
            "obj_l7": {"p": [0.0, 0.0, 0.0], "R": np.eye(3).ravel().tolist()}}


def test_condition_refreshes_a_collapsed_belief_from_the_prior_with_all_its_evidence():
    rng = np.random.default_rng(1)
    b = _toy_prior(1000, rng)
    fit = cg.fit_entry(_fit_at(1.0, [0.0, 0.01, 0.03]))           # pins the CoM height at 0.03, says little of the rest
    b1, i1 = cg.condition(b, [fit], cg.replay_fn(None), prior_fn=_toy_prior, rng=rng, n_draws=(3000, 30000))
    assert not i1["refreshed"] and i1["ess"] > 100 and [e["type"] for e in b1.history] == [cg.GRASP_FIT]
    y, info = cg.condition(b1, [cg.truth_entry({"mass": 1.0, "com_frac": 0.6}, ("mass", "com"))], cg.replay_fn(None),
                           prior_fn=_toy_prior, rng=rng, n_draws=(3000, 30000))
    assert info["ess"] < 100 and info["refreshed"] and info["importance_ess"] >= 100 and not info["degenerate"]
    assert len(y) == 1000 and [e["type"] for e in y.history] == [cg.GRASP_FIT, cg.TRUTH]
    a = y.arrays()
    assert abs(float(y.weights @ np.log(a["mass"]))) < 0.01 and abs(float(y.weights @ a["com_frac"]) - 0.6) < 0.005
    z = a["com_obj"][:, 2]
    assert np.sqrt(float(y.weights @ (z - y.weights @ z) ** 2)) < 0.003            # the refresh replayed the fit too
    kept, k = cg.condition(b1, [cg.truth_entry({"mass": 1.0, "com_frac": 0.6}, ("mass", "com"))], cg.replay_fn(None))
    assert k["ess"] < 100 and not k["refreshed"] and k["ess_final"] == pytest.approx(k["ess"])   # no prior: weights only


def test_with_no_fit_belief_one_is_the_belief_before_the_attempt():
    b = _toy_prior(200, np.random.default_rng(0))
    b1, info = cg.condition(b, [], cg.replay_fn(None))
    assert np.allclose(b1.weights, b.weights) and b1.history == [] and info["ess"] == pytest.approx(200)


# ---------------------------------------------------------------- the choices at (b)

class P:
    def __init__(self, s, width=0.04):
        self.s, self.point, self.yaw, self.width = s, [0.0, s - 0.5, 0.76], 0.0, width


def test_the_charges_are_the_menus_own():
    rows = {r[0]: r for r in _rows([P(0.3), P(0.5), P(0.7)], 20.0, "plate", np.zeros(2), damage=True)}
    edge = {r[0]: r for r in _rows([P(0.3), P(0.5), P(0.7)], 20.0, "table_edge", np.zeros(2), damage=True)}
    assert cg.BASE_S == rows["commit_std"][4]
    assert cg.SLOW_EXTRA_S == rows["slow_lift"][4] - rows["commit_std"][4]
    assert cg.SLOW_LIFT == {"v_lift": rows["slow_lift"][3]["v_lift"], "v_transport": rows["slow_lift"][3]["v_transport"]}
    assert cg.CARRY_S["gentle"] == rows["gentle_carry"][4] - rows["commit_std"][4] and cg.CARRY_S["standard"] == 0.0
    assert cg.PLACE_S["drop_release"] == rows["drop_release"][4] - rows["commit_std"][4]
    assert cg.PLACE_S["inboard_place"] == edge["inboard_place"][4] - edge["commit_std"][4] and cg.PLACE_S["requested"] == 0.0


def test_finishing_rows_cross_the_carries_with_the_places_the_menu_offers():
    ex = {"force": 30.0, "v_lift": 0.05, "v_transport": 0.2, "a_transport": STD["a_transport"], "destination": "plate",
          "release_height": 0.065, "s": 0.4}
    rows = cg.finishing_rows(ex, "plate", {"plate": 2.5})
    assert [r["id"] for r in rows] == ["standard|requested", "slow|requested", "gentle|requested",
                                       "standard|drop_release", "slow|drop_release", "gentle|drop_release"]
    by = {r["id"]: r for r in rows}
    assert all(r["action"]["force"] == 30.0 and r["action"]["v_lift"] == 0.05 and r["action"]["s"] == 0.4 for r in rows)
    assert by["standard|requested"]["action"]["v_transport"] == STD["v_transport"] and by["slow|requested"]["action"]["v_transport"] == 0.2
    assert by["gentle|requested"]["action"]["a_transport"] == A_GENTLE and by["slow|requested"]["action"]["a_transport"] == STD["a_transport"]
    assert by["standard|requested"]["action"]["release_height"] is None
    assert by["standard|drop_release"]["action"]["release_height"] == pytest.approx(0.05 + DESTINATIONS["plate"]["rim_height"])
    assert by["standard|requested"]["cost_s"] == 8.0 and by["gentle|requested"]["cost_s"] == 9.0
    assert by["standard|drop_release"]["cost_s"] == 6.0 and by["gentle|drop_release"]["cost_s"] == 7.0
    assert by["slow|drop_release"]["cost_s"] == pytest.approx(8.0 + 2.5 - 2.0)
    edge = {r["id"]: r for r in cg.finishing_rows(dict(ex, destination="table_edge"), "table_edge",
                                                  {"table_edge": 1.0, "table_centre": 2.0})}
    assert set(edge) == {f"{c}|{p}" for c in ("standard", "slow", "gentle") for p in ("requested", "inboard_place")}
    assert edge["slow|inboard_place"]["action"]["destination"] == "table_centre"
    assert edge["slow|inboard_place"]["cost_s"] == pytest.approx(8.0 + 2.0 + 1.0) and edge["slow|requested"]["cost_s"] == 9.0
    centre = cg.finishing_rows(dict(ex, destination="table_centre"), "table_centre", {"table_centre": 1.0})
    assert [r["id"] for r in centre] == ["standard|requested", "slow|requested", "gentle|requested"]


def _plan_steps(proposal, v_lift, v_transport, destination, release_height=None, a_transport=2.0):
    dist = {"plate": 0.4, "table_centre": 0.1}[destination]
    return [{"phase": "lift", "n_steps": int(round(10.0 / v_lift))}, {"phase": "hold", "n_steps": 25},
            {"phase": "transport", "n_steps": int(round(50.0 * dist / v_transport))}]


def test_the_slow_carry_gets_slow_lifts_extra_time_times_the_carrys_share():
    p = P(0.4)
    assert cg.slow_carry_s(_plan_steps, p, "plate", None) == pytest.approx(6.0 * 60 / 193)       # lift +133, carry +60
    assert cg.slow_carry_s(_plan_steps, p, "table_centre", None) == pytest.approx(6.0 * 15 / 148)  # lift +133, carry +15


def test_gap_prices_each_beliefs_choice_on_the_yardstick():
    c1, c2, y = {"a": 1.0, "b": 2.0, "c": 0.5}, {"a": 3.0, "b": 1.0, "c": 4.0}, {"a": 2.0, "b": 1.5, "c": 1.0}
    assert cg.gap(c1, c2, y) == {"choice1": "c", "choice2": "b", "evpi": pytest.approx(-0.5), "changed": True}
    assert cg.gap(c1, c2, y, exclude=("c",)) == {"choice1": "a", "choice2": "b", "evpi": pytest.approx(0.5), "changed": True}
    assert cg.gap(c1, c1, y)["evpi"] == 0.0 and not cg.gap(c1, c1, y)["changed"]


# ---------------------------------------------------------------- the captured scene and the prices

def _geometry():
    C = np.array([[0.02, 0.0, 0.76], [-0.02, 0.0, 0.76]])
    N = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    return Geometry(centres=C, normals=N, com=np.array([0.0, 0.0, 0.77]), length=0.1, base_extent=0.02,
                    com_height=0.02, footprint_half_width=0.05)


def _pose(s=0.5):
    from tests.unit_tests.skills.synthetic import synthetic_pose

    return dict(synthetic_pose(), pt=[0.0, s - 0.5, 0.76])


class _SceneEnv:
    """The lab calls a scene needs, counted: four proposals, one too wide for the gripper and one out of reach."""

    max_pinch_width = 0.09

    def __init__(self):
        self.calls = Counter()

    def geometry(self):
        self.calls["geometry"] += 1
        return _geometry()

    def proposals(self):
        self.calls["proposals"] += 1
        return [P(0.1, width=0.2), P(0.3), P(0.5), P(0.7), P(0.9)]

    def reach(self, p, destination):
        self.calls["reach"] += 1
        return None if p.s < 0.8 else "unreachable"

    def proposal_pose(self, p):
        self.calls["pose"] += 1
        return _pose(p.s)

    def pose_info(self):
        self.calls["pose_info"] += 1
        return {"object_id": "x"}

    def plan_trajectory(self, p, v_lift, v_transport, destination, release_height=None, a_transport=2.0):
        self.calls["plan"] += 1
        dist = 0.3 if destination == "table_centre" else 0.4
        return [SimpleNamespace(to_dict=lambda n=n, ph=ph: {"phase": ph, "n_steps": n, "z_start": 0.0, "z_end": 0.2,
                                                            "has_surface": True, "speed": v_lift})
                for ph, n in (("lift", int(round(10.0 / v_lift))), ("hold", 25), ("transport", int(round(50.0 * dist / v_transport))))]


class _CarryModel:
    """P(placed_stable) 1, except a fast carry (v_transport above 0.2 and a_transport above A_GENTLE) of a CoM more than
    2.5 cm from the pinch: then P(dropped_high) 0.5."""

    kind = "classifier"

    def predict_many(self, feats):
        out = []
        for f in feats:
            risky = f["lever"] > 0.025 and f["v_transport"] > 0.2 + 1e-9 and f["a_transport"] > A_GENTLE + 1e-9
            p = {k: 0.0 for k in OUTCOMES}
            p.update({"placed_stable": 0.5, "dropped_high": 0.5} if risky else {"placed_stable": 1.0})
            out.append(Prediction(p=p))
        return out


def _lever_belief(n=100):
    """Half the particles hold the CoM at the pinch of s 0.5 (CoM fraction 0.5), half 5 cm along the axis (1.0)."""
    parts = [{"mass": 0.2, "mu": 0.5, "com_obj": [0.0, 0.05 * (i % 2), 0.03], "com_frac": 0.5 + 0.5 * (i % 2)} for i in range(n)]
    return ParticleBelief(parts, np.ones(n))


EXECUTED = {"force": 20.0, "v_lift": 0.15, "v_transport": 0.5, "a_transport": STD["a_transport"], "destination": "plate",
            "release_height": None, "s": 0.5}


def test_the_scene_serves_every_plan_and_pose_without_the_lab():
    env = _SceneEnv()
    scene = cg.capture_scene(env, "plate", 20.0, EXECUTED)
    assert [p.s for p in scene.proposals] == [0.1, 0.3, 0.5, 0.7] and scene.max_open == 0.09 and scene.pose_info == {"object_id": "x"}
    n = dict(env.calls)
    rows = cg.finishing_rows(EXECUTED, "plate", cg.scene_slow_s(scene, EXECUTED, "plate"))
    for r in rows:
        act = r["action"]
        scene.plan(scene.proposal(act["s"]), act["v_lift"], act["v_transport"], act["destination"], act["release_height"],
                   a_transport=act["a_transport"])
    for b in (_lever_belief(), ParticleBelief([dict(p, com_obj=[0.0, -0.2, 0.03]) for p in _lever_belief().particles], np.ones(100))):
        build_menu(_CarryModel(), b, scene.geometry, scene.proposals, 20.0, 100.0, "plate", scene.plan,
                   np.random.default_rng(0), pose_fn=scene.pose, max_open=scene.max_open, abstain_cost=3.0)
    assert dict(env.calls) == n                                         # nothing read the lab after the capture
    with pytest.raises(KeyError):
        scene.plan(scene.proposal(0.5), 0.07, 0.5, "plate", None, a_transport=STD["a_transport"])
    with pytest.raises(KeyError):
        scene.proposal(0.9)


def test_price_rows_prices_a_row_as_the_menu_does():
    scene = cg.capture_scene(_SceneEnv(), "plate", 20.0, EXECUTED)
    b = _lever_belief()
    opts, _ = build_menu(_CarryModel(), b, scene.geometry, scene.proposals, 20.0, 100.0, "plate", scene.plan,
                         np.random.default_rng(0), pose_fn=scene.pose, max_open=scene.max_open, abstain_cost=3.0)
    rows = [cg.menu_row(o) for o in opts if o.kind in ("commit", "lower_risk")]
    priced = cg.price_rows(_CarryModel(), b, scene, rows, 100.0, 3.0)
    assert set(priced) == {r["id"] for r in rows} and len(rows) >= 5
    for o in opts:
        if o.kind in ("commit", "lower_risk"):
            assert priced[o.id] == pytest.approx(o.expected_cost)


def test_point_b_prices_both_choices_on_the_yardstick():
    """Belief 1 is split between a CoM at the pinch and one 5 cm off; the truth is at the pinch. Belief 1 pays for a gentle
    carry (+1 s), belief 2 drops it: EVPI is that second at 0.05 per second."""
    scene = cg.capture_scene(_SceneEnv(), "plate", 20.0, EXECUTED)
    rows = cg.finishing_rows(EXECUTED, "plate", cg.scene_slow_s(scene, EXECUTED, "plate"))
    out = cg.point_b(_CarryModel(), _lever_belief(), scene, rows, {"mass": 0.2, "com_frac": 0.5}, 100.0, 3.0,
                     cg.replay_fn(None), w0=_lever_belief())
    assert out["primary"]["choice1"] == "gentle|drop_release" and out["primary"]["choice2"] == "standard|drop_release"
    assert out["primary"]["evpi"] == pytest.approx(0.05) and out["primary"]["changed"]
    assert out["no_inboard"] == out["primary"]                         # no inboard row at the plate
    w0 = out["costs"]["w0"]                                            # w0 keeps both CoMs: b2's fast carry is risky there
    assert out["w0"]["evpi"] == pytest.approx(w0["gentle|drop_release"] - w0["standard|drop_release"]) and out["w0"]["evpi"] < 0
    assert out["ess"]["b2"]["ess"] == pytest.approx(50) and out["ess"]["yard"]["ess"] == pytest.approx(50)
    assert set(out["costs"]) == {"b1", "b2", "yard", "w0"} and set(out["costs"]["b1"]) == {r["id"] for r in rows}


def test_the_full_menu_check_prices_each_beliefs_best_action_on_the_yardstick():
    scene = cg.capture_scene(_SceneEnv(), "plate", 20.0, EXECUTED)
    b1 = _lever_belief()
    out = cg.full_menu(_CarryModel(), b1, b1, b1, scene, "plate", 20.0, 100.0, 3.0)
    assert out["choice1"] == out["choice2"] and out["evpi"] == 0.0
    yard = ParticleBelief(b1.particles, np.array([1.0, 0.0] * 50))          # the truth: the CoM at the pinch
    out = cg.full_menu(_CarryModel(), b1, yard, yard, scene, "plate", 20.0, 100.0, 3.0)
    assert out["choice1"] != out["choice2"] and out["evpi"] > 0


# ---------------------------------------------------------------- the policy up to its first attempt

def _row(rid="commit_std"):
    return {"id": rid, "kind": "commit", "action": {"destination": "table_centre", "s": 0.5, "force": 20.0}, "p": {"placed_stable": 0.9},
            "p_drop_height": None, "expected_cost": 0.6, "cost_s": 8.0}


def _assessment(rec, probe=False, without=None):
    opts = [_row()] + ([{"id": "probe_push", "kind": "probe", "action": {"probe": "push"}}] if probe else [])
    return {"assessed": True, "recommendation": rec, "best_option": "commit_std", "without_probe": without,
            "recommended_option": "probe_push" if rec == "probe" else ("commit_std" if rec == "commit" else None),
            "guard": None, "options": opts}


class _Script:
    """Scripted assess_pick and pick_and_place: the assessments in order, the attempts' outcomes in order."""

    def __init__(self, assessments, outcomes=()):
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


SCRIPTS = {
    "decline": [_assessment("decline")],
    "probe_then_decline": [_assessment("probe", probe=True), _assessment("decline")],
    "probe_then_commit": [_assessment("probe", probe=True), _assessment("commit")],
    "capped": [_assessment("probe", probe=True), _assessment("probe", probe=True), _assessment("probe", probe=True, without="commit")],
    "commit": [_assessment("commit")],
    "error": [{"error": "boom"}],
    "unreachable": [{"assessed": False, "reason": "unreachable"}],
}


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_first_attempt_runs_the_policy_protocol_up_to_its_first_attempt(monkeypatch, name):
    from robots.mujoco_lab import calibrate

    ref_script = _Script(SCRIPTS[name], ["placed_stable"])
    monkeypatch.setattr(calibrate, "assess_pick", ref_script.assess)
    monkeypatch.setattr(calibrate, "pick_and_place", ref_script.pick)
    ref = calibrate.run_protocol(SimpleNamespace(abstain_cost=1.0), "policy", "table_centre", np.random.default_rng(0))
    sc = _Script(SCRIPTS[name])
    monkeypatch.setattr(cg, "assess_pick", sc.assess)
    monkeypatch.setattr(cg, "pick_and_place", sc.pick)
    got = cg.first_attempt(SimpleNamespace(abstain_cost=1.0), "table_centre", "normal", {})
    probes = [c for c in ref_script.calls if str(c).startswith("probe_")]
    assert sc.calls == probes and got["probes"] == ref["probes"] and got["moved"] == bool(probes)
    assert got["recommendation"] == ref["recommendation"] and got["first_menu"] == ref["first_menu"]
    if ref["attempts"]:
        assert got["ended"] == "attempt" and got["row"]["id"] == ref["attempts"][0]["option"]
        assert got["assessment"]["best_option"] == got["row"]["id"]
    else:
        assert got["ended"] == ref["ended"] and got.get("row") is None


# ---------------------------------------------------------------- the summary and the stop rule

def _recs(evpis, setting="k100_A3", prior="class"):
    return [{"object": f"o{i % 5}", "idx": i, "prior": prior, "setting": setting, "reached": e is not None,
             "evpi": 0.0 if e is None else e, "evpi_w0": 0.0, "evpi_no_inboard": 0.0 if e is None else e, "evpi_full": 0.0,
             "changed": bool(e), "ess_b2": None if e is None else 500.0, "ess_yard": None if e is None else (50.0 if i % 2 else 500.0),
             "flat": False, "com_frac": 0.1 * (i % 10),
             "choice1": "a", "choice2": "b" if e else "a"} for i, e in enumerate(evpis)]


def test_summary_counts_the_cases_that_never_reach_b_as_zero():
    s = cg.summarize(_recs([None, None, 0.3, 0.1, -0.05, None, 0.0, None, None, None]), n_boot=200, seed=0)
    c = s["cells"]["k100_A3"]["class"]
    assert c["n"] == 10 and c["n_reached"] == 4 and c["share_reached"] == pytest.approx(0.4)
    assert c["evpi"]["mean"] == pytest.approx(0.035) and c["evpi_reached"] == pytest.approx(0.35 / 4)
    assert c["H"]["0.05"]["mean"] == pytest.approx(0.03) and c["H"]["0.3"]["mean"] == 0.0
    assert c["share_positive"] == pytest.approx(0.2) and c["share_changed"] == pytest.approx(0.3)
    lo, hi = c["H"]["0.05"]["ci"]
    assert lo <= 0.03 <= hi
    assert c["ess_table"]["n"] == 9          # less the reached case with an ESS under 100 (idx 3); the rest count as above


def test_the_stop_rule():
    def H(m, lo, hi):
        return {"mean": m, "ci": [lo, hi]}

    assert cg.verdict({"a": H(0.005, 0.0, 0.015), "b": H(0.01, 0.002, 0.019)}) == "stop"
    assert cg.verdict({"a": H(0.005, 0.0, 0.015), "b": H(0.03, 0.01, 0.05)}) == "go"
    assert cg.verdict({"a": H(0.005, 0.0, 0.025)}) == "undecided"
    assert cg.verdict({"a": H(0.03, 0.0, 0.06)}) == "undecided"          # at the bar, but the lower bound is not above 0


def test_sample_cases_is_a_fixed_draw_of_the_case_set():
    cases = [(o, i) for o in ("a", "b", "c") for i in range(11)]
    s = cg.sample_cases(cases, 20, 0)
    assert len(s) == 20 and len(set(s)) == 20 and set(s) <= set(cases) and s == cg.sample_cases(cases, 20, 0)
    assert cg.sample_cases(cases, None, 0) == cases


def test_run_case_refuses_a_lab_without_joint_sensing(monkeypatch):
    """The hand sim has no joint torques: execute never calls _attempt_joint, so every lifted case would read as one
    that never reached (b), at EVPI 0, and the run would read as a stop. The case refuses such a lab after its reset
    (env.sim, and with it joint_sensing, exists only from the first reset on), before any setting runs."""
    from robots.mujoco_lab import adapter
    from robots.mujoco_lab import calibrate as cal

    monkeypatch.setattr(cal, "case_draw", lambda *a, **k: ({}, None, "plate", {"fill": 0.0, "frag": None}))
    monkeypatch.setattr(adapter, "object_props", lambda o: {"mass": 1.0})
    monkeypatch.setattr(cg, "case_b", lambda *a, **k: pytest.fail("a setting ran"))
    resets = []
    env = SimpleNamespace(joint_sensing=False, reset=lambda *a, **k: resets.append(1), pose_info=lambda: {}, reset_info={})
    with pytest.raises(RuntimeError, match="joint sensing"):
        cg.run_case(SimpleNamespace(env=env), "x", 0, {"seed": 11, "settings": [("k100_A3", 100.0, 3.0)]}, {"class": None},
                    None, None)
    assert resets == [1]


def _ended(monkeypatch, ended, probes=()):
    fa = {"ended": ended, "recommendation": "decline", "best_option": None, "guard": None, "first_menu": None,
          "probes": list(probes), "restores": [], "moved": bool(probes), "row": None, "assessment": None}
    monkeypatch.setattr(cg, "first_attempt", lambda *a, **k: dict(fa))


def test_a_skipped_full_menu_check_is_missing_not_zero(monkeypatch):
    _ended(monkeypatch, "declined")
    args = (None, SimpleNamespace(), "x", "plate", 100.0, 3.0, None, {}, None, None)
    assert cg.case_b(*args, full=False)["evpi_full"] is None
    assert cg.case_b(*args, full=True)["evpi_full"] == 0.0           # checked: a case that never reached (b) counts 0
    recs = _recs([None, 0.3, 0.1])
    assert cg.cell(recs, np.random.default_rng(0), 50)["checks"]["full"]["mean"] == 0.0
    recs[0]["evpi_full"] = None
    c = cg.cell(recs, np.random.default_rng(0), 50)
    assert c["checks"]["full"] == {"mean": None, "ci": None} and c["checks"]["w0"]["mean"] == 0.0
    assert "| n/a |" in cg.markdown({"cells": {"k100_A3": {"class": c}}, "verdict": {}, "verdict_all_cases": {},
                                     "primary_prior": "class"})


def test_a_flagged_probe_marks_the_case_as_an_artifact(monkeypatch):
    """calibrate's case convention: a flagged probe or attempt, or a reset off rest, makes an artifact case."""
    args = (None, SimpleNamespace(), "x", "plate", 100.0, 3.0, None, {}, None, None)
    _ended(monkeypatch, "declined", [{"probe": "weigh", "kind": "measured", "artifact": True}])
    assert cg.case_b(*args)["artifact"] is True
    _ended(monkeypatch, "declined", [{"probe": "weigh", "kind": "measured", "artifact": False}])
    assert cg.case_b(*args)["artifact"] is False


def test_the_metrics_leave_out_and_count_errors_and_artifacts():
    """As calibrate counts cases (errors and artifact cases left out and counted); H over every record, with those at
    EVPI 0, is reported next to it, with the stop rule's verdict on it."""
    recs = _recs([None, 0.3, 0.1, None])
    recs += [dict(_recs([1.0])[0], object="o9", idx=9, artifact=True), {**_recs([None])[0], "object": "o8", "idx": 8, "error": "boom"}]
    for r in recs[-2:]:
        r.pop("evpi", None) if "error" in r else None
    c = cg.cell(recs, np.random.default_rng(0), 50)
    assert c["n"] == 4 and c["n_errors"] == 1 and c["n_artifacts"] == 1
    assert c["evpi"]["mean"] == pytest.approx(0.1) and c["H"]["0.05"]["mean"] == pytest.approx(0.3 / 4)
    assert c["H_all_cases"]["0.05"]["mean"] == pytest.approx(0.3 / 6)
    s = cg.summarize(recs, n_boot=50, seed=0)
    assert set(s["verdict_all_cases"]) == {"class"} and s["verdict_all_cases"]["class"] in ("go", "stop", "undecided")


def _b_case(monkeypatch, record=None):
    """A case that reaches (b) on the lever belief (CoM fraction 0.5 or 1.0): the cut log's fit says nothing, the whole
    log's fit puts the CoM at the pinch (m cy 0, sd 2 mm). ``record`` is the fits' hidden record. Returns (prims, env, b)."""

    def fit(info_cy):
        return {"floor": [0.001, 0.0005], "obj_l7": {"p": [0.0, 0.0, 0.0], "R": np.eye(3).tolist()},
                "phi": [0.2, 0.0, 0.0, 0.006], "info": np.diag([0.0, 0.0, info_cy, 0.0]).tolist(),
                "cov": np.diag([0.01, 1e-4, 1e-4, 1e-4]).tolist(), "window_s": 1.0, "n": 100, "phases": ["hold", "lift"]}

    env, b = _SceneEnv(), _lever_belief(1000)
    env.nominal_mass = lambda o: 0.2
    env._attempt_joint = lambda log, trace: (None, {})
    env._joint_fit = lambda log, steps, params: (fit(0.0) if max(log.step) == 8 else fit(1.0 / 0.002 ** 2), dict(record or {}))

    def execute(*a, **k):
        env._attempt_joint(_Log(_trace()), _trace())
        return SimpleNamespace(label="placed_stable", loss_phase=None, artifact=False)

    env.execute = execute

    def first(prims, destination, care, store):
        store["pick_place"] = {"x": {"menu": {"commit_std": SimpleNamespace(id="commit_std", action=EXECUTED)}, "belief": b}}
        return {"ended": "attempt", "recommendation": "commit", "best_option": "commit_std", "guard": None, "first_menu": {},
                "probes": [], "restores": [], "moved": False, "row": {"id": "commit_std"}}

    monkeypatch.setattr(cg, "first_attempt", first)
    return SimpleNamespace(model=_CarryModel()), env, b


def test_case_b_reports_the_whole_log_fit_as_a_check(monkeypatch):
    """The check column: the same point (b) with the fit on the whole log (the carry reaches it through the filter).
    Here the cut log's fit says nothing, so belief 1 stays split and pays for the gentle carry (EVPI 0.05, as in the
    point_b test); the whole log's fit puts the CoM at the pinch, so belief 1 takes belief 2's carry (EVPI 0). The
    primary result does not read the check's stream."""
    prims, env, b = _b_case(monkeypatch)
    truth = {"mass": 0.2, "com_frac": 0.5}
    runs = [cg.case_b(prims, env, "x", "plate", 100.0, 3.0, None, truth, b, np.random.default_rng(0), full=False,
                      rng_whole_log=np.random.default_rng(k)) for k in (1, 2)]
    r = runs[0]
    assert r["reached"] and r["evpi"] == pytest.approx(0.05) and r["choice1"] == "gentle|drop_release"
    assert r["evpi_whole_log"] == pytest.approx(0.0) and r["choice1_whole_log"] == "standard|drop_release"
    assert not r["changed_whole_log"] and r["fit_whole_log"]["phi4"] == [0.2, 0.0, 0.0, 0.006]
    assert set(r["ess_whole_log"]) == {"b1", "b2", "yard"}
    assert runs[1]["evpi"] == r["evpi"] and runs[1]["costs"] == r["costs"]     # the check's stream leaves the primary alone


def test_the_whole_log_check_gets_its_own_H_and_verdict():
    recs = _recs([None, 0.3, 0.1, None])
    for r, w in zip(recs, (0.0, 0.0, 0.2, 0.0)):
        r["evpi_whole_log"] = w
    c = cg.cell(recs, np.random.default_rng(0), 50)
    assert c["checks"]["whole_log"]["mean"] == pytest.approx(0.05)
    assert c["H_whole_log"]["0.05"]["mean"] == pytest.approx(0.15 / 4) and c["H"]["0.05"]["mean"] == pytest.approx(0.3 / 4)
    s = cg.summarize(recs, n_boot=50, seed=0)
    assert set(s["verdict_whole_log"]) == {"class"}
    md = cg.markdown(s)
    assert "whole-log fit" in md and md.count("|") > 0


def test_fit_z_scores_the_fit_against_the_window_truth():
    """Belief 1's diagnostic at (b) (coordinator, 2026-11-05): the fit's (m, m cx, m cy, m cz) against the window's hidden
    truth (joint_truth_l7), in the floored sds that the belief's likelihood uses; None without a fit or a truth."""
    fit = {"phi": [0.6, 0.01, 0.0, -0.02, 9.0], "cov": np.diag([0.01, 1e-4, 4e-4, 4e-4, 1.0]).tolist()}
    rec = {"joint_truth_l7": [0.5, 0.0, 0.0, 0.0, 0.0]}
    assert cg.fit_z(fit, rec) == pytest.approx([1.0, 1.0, 0.0, -1.0])
    assert cg.fit_z(None, rec) is None and cg.fit_z(fit, {}) is None and cg.fit_z(fit, None) is None


def test_cover80_is_the_truth_inside_the_weighted_10_to_90_percent_interval():
    b = _lever_belief(100)                                   # mass 0.2; CoM fraction 0.5 or 1.0, half each
    assert cg.cover80(b, {"mass": 0.2, "com_frac": 0.75}) == {"mass": True, "com_frac": True}
    assert cg.cover80(b, {"mass": 0.3, "com_frac": 0.4}) == {"mass": False, "com_frac": False}
    w = ParticleBelief(b.particles, np.array([1.0, 0.0] * 50))     # all the weight on CoM fraction 0.5
    assert cg.cover80(w, {"mass": 0.2, "com_frac": 0.75})["com_frac"] is False


def test_case_b_records_the_z_scores_and_belief_ones_coverage_at_b(monkeypatch):
    """Per reached case: both fits' z-scores against the window's truth, and whether the 80 % intervals of belief 1 (and
    of the whole-log check's belief) hold the case's truth. The lever belief's interval of CoM fraction is [0.5, 1.0];
    the whole-log fit collapses it to 0.5."""
    prims, env, b = _b_case(monkeypatch, {"joint_truth_l7": [0.1, 0.0, 0.0, 0.006]})
    r = cg.case_b(prims, env, "x", "plate", 100.0, 3.0, None, {"mass": 0.2, "com_frac": 0.75}, b, np.random.default_rng(0),
                  full=False, rng_whole_log=np.random.default_rng(1))
    assert r["reached"]
    assert r["fit_z"] == pytest.approx([1.0, 0.0, 0.0, 0.0]) and r["fit_z_whole_log"] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert r["cover80_b1"] == {"mass": True, "com_frac": True}
    assert r["cover80_whole_log"] == {"mass": True, "com_frac": False}


def test_the_cell_reports_the_z_scores_and_belief_ones_coverage():
    """Over the counted cases at (b) (the artifact case and the case that never reached (b) are not in it): each fit
    parameter's median z-score and its share above 0, and the share of cases whose 80 % interval holds the truth."""
    recs = _recs([None, 0.3, 0.1, 0.0, 0.2])
    z = [[1.0, 2.0, 0.5, 1.5], [-1.0, 1.0, 0.0, 2.5], [0.5, 3.0, -0.5, -1.0], [2.0, 1.0, 1.0, 1.0]]
    cover = [(True, False), (True, True), (False, False), (True, False)]
    for r, zz, (m, c) in zip(recs[1:], z, cover):
        r["fit_z"], r["cover80_b1"] = zz, {"mass": m, "com_frac": c}
    recs.append(dict(recs[1], object="o9", idx=9, artifact=True, fit_z=[9.0] * 4, cover80_b1={"mass": False, "com_frac": False}))
    c = cg.cell(recs, np.random.default_rng(0), 50)
    d = c["b1_diagnostic"]
    assert d["n_z"] == 4 and d["n_cover"] == 4
    assert d["z"]["m"] == {"median": pytest.approx(0.75), "share_positive": pytest.approx(0.75)}
    assert d["z"]["m_cx"] == {"median": pytest.approx(1.5), "share_positive": pytest.approx(1.0)}
    assert d["z"]["m_cy"] == {"median": pytest.approx(0.25), "share_positive": pytest.approx(0.5)}
    assert d["z"]["m_cz"] == {"median": pytest.approx(1.25), "share_positive": pytest.approx(0.75)}
    assert d["cover80"] == {"mass": pytest.approx(0.75), "com_frac": pytest.approx(0.25)}
    assert c["b1_diagnostic_whole_log"]["n_z"] == 0 and c["b1_diagnostic_whole_log"]["cover80"]["com_frac"] is None


def test_the_verdict_is_flagged_when_belief_one_under_covers_the_com_fraction():
    """Below 0.7, the 80 % coverage of the CoM fraction at (b) (pooled over the settings, per prior) puts the
    coordinator's flag next to that prior's verdict; the stop rule does not change."""

    def recs(cover_com):
        rs = _recs([0.3, 0.1, 0.0, 0.2]) + _recs([0.3, 0.1, 0.0, 0.2], setting="k30_A3")
        for r, c in zip(rs, cover_com):
            r["fit_z"], r["cover80_b1"] = [0.0] * 4, {"mass": True, "com_frac": c}
        return rs

    low = cg.summarize(recs([True, False, False, True, True, False, True, False]), n_boot=50, seed=0)    # 4 of 8
    ok = cg.summarize(recs([True] * 6 + [False] * 2), n_boot=50, seed=0)                                   # 6 of 8
    assert low["b1_diagnostic"]["class"]["cover80"]["com_frac"] == pytest.approx(0.5)
    assert low["flags"] == {"class": cg.UNDER_COVERS} and ok["flags"] == {}
    assert low["verdict"] == ok["verdict"]
    md = cg.markdown(low)
    assert cg.UNDER_COVERS in next(line for line in md.splitlines() if line.startswith("Stop rule"))
    assert cg.UNDER_COVERS not in cg.markdown(ok) and "80 % coverage" in md


def _shard(tmp_path, name, objects, repeats=2, settings=("k100_A3", "k30_A3"), priors=("class", "generic"), **cfg):
    """A com_gap run dir as run() writes it (run_config.json, records.jsonl, provenance.txt) for `objects`."""
    d = tmp_path / name
    d.mkdir()
    cases = [[o, i] for o in objects for i in range(repeats)]
    config = {"cmd": "run", "bundle": "B", "scenes": "S", "settings": None, "priors": None, "objects": ",".join(objects),
              "seed": 11, "repeats": repeats, "sample": None, "sample_seed": 0, "no_full_menu": False, "workers": 1,
              "out": str(d), "code_revision": "abc", "dirty": False, "generator": "v11",
              "settings_parsed": [[s, 100.0, 3.0] for s in settings], "priors_parsed": list(priors), "cases": cases,
              "spreads": {"generic": {}}, "generic": {"o": {}}, "exact_sd": {"mass": 0.01}, "stop_bound": 0.02,
              "h_costs": [0.05, 0.1, 0.3], "ess_table": 100, **cfg}
    (d / "run_config.json").write_text(json.dumps(config))
    recs = [{"object": o, "idx": i, "prior": p, "setting": s} for o, i in cases for p in priors for s in settings]
    (d / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (d / "provenance.txt").write_text(f"code abc, shard {name}\n")
    return d


def _merge(dirs, out):
    return cg.merge(SimpleNamespace(dirs=",".join(str(d) for d in dirs), out=str(out)))


def _merged(out):
    return ([json.loads(line) for line in open(out / "records.jsonl")], json.load(open(out / "run_config.json")),
            (out / "provenance.txt").read_text())


def test_merge_joins_disjoint_shards_into_one_run(tmp_path):
    a, b = _shard(tmp_path, "a", ["o1", "o3"]), _shard(tmp_path, "b", ["o2"])
    _merge([a, b], tmp_path / "m")
    recs, cfg, prov = _merged(tmp_path / "m")
    assert len(recs) == 3 * 2 * 2 * 2
    assert cfg["cases"] == [["o1", 0], ["o1", 1], ["o2", 0], ["o2", 1], ["o3", 0], ["o3", 1]]
    assert cfg["objects"] == "o1,o2,o3" and cfg["out"] == str(tmp_path / "m") and cfg["merged_from"] == [str(a), str(b)]
    assert "shard a" in prov and "shard b" in prov


def test_merge_refuses_a_case_in_two_shards(tmp_path):
    with pytest.raises(SystemExit, match="o1.*twice"):
        _merge([_shard(tmp_path, "a", ["o1"]), _shard(tmp_path, "b", ["o1", "o2"])], tmp_path / "m")


@pytest.mark.parametrize("key, value", [("seed", 12), ("code_revision", "def"), ("settings_parsed", [["k100_A3", 100.0, 10.0]]),
                                        ("bundle", "B2"), ("post_identify", True)])
def test_merge_refuses_shards_that_differ_in_what_the_records_depend_on(tmp_path, key, value):
    with pytest.raises(SystemExit, match=key):
        _merge([_shard(tmp_path, "a", ["o1"]), _shard(tmp_path, "b", ["o2"], **{key: value})], tmp_path / "m")


def test_merge_refuses_a_shard_run_on_uncommitted_code(tmp_path):
    with pytest.raises(SystemExit, match="dirty"):
        _merge([_shard(tmp_path, "a", ["o1"], dirty=True)], tmp_path / "m")


def test_merge_drops_a_stopped_shards_unfinished_object_until_it_is_rerun(tmp_path):
    a = _shard(tmp_path, "a", ["o1", "o2"], repeats=1, settings=("k100_A3",), priors=("class",))
    lines = (a / "records.jsonl").read_text().splitlines()
    (a / "records.jsonl").write_text(lines[0] + "\n" + lines[1][:12])          # o2's record cut mid-line by the stop
    with pytest.raises(SystemExit, match="missing.*o2"):
        _merge([a], tmp_path / "m")
    _merge([a, _shard(tmp_path, "b", ["o2"], repeats=1, settings=("k100_A3",), priors=("class",))], tmp_path / "m2")
    recs, cfg, _ = _merged(tmp_path / "m2")
    assert [(r["object"], r["idx"]) for r in recs] == [("o1", 0), ("o2", 0)] and cfg["dropped"] == {str(a): ["o2"]}


def test_merge_drops_an_object_with_only_some_of_its_records(tmp_path):
    a = _shard(tmp_path, "a", ["o1", "o2"])
    lines = (a / "records.jsonl").read_text().splitlines()
    (a / "records.jsonl").write_text("\n".join(lines[:-3]) + "\n")              # o2 lost its last three records
    _merge([a, _shard(tmp_path, "b", ["o2"])], tmp_path / "m")
    recs, cfg, _ = _merged(tmp_path / "m")
    assert len(recs) == 2 * 2 * 2 * 2 and cfg["dropped"] == {str(a): ["o2"]}


def test_merge_refuses_an_existing_out(tmp_path):
    a = _shard(tmp_path, "a", ["o1"])
    _merge([a], tmp_path / "m")
    with pytest.raises(SystemExit, match="exists"):
        _merge([a], tmp_path / "m")


# ---------------------------------------------------------------- post_identify (an amendment after the pre-registration, 2026-11-06)

class _IdentifyModel:
    """A probe model whose reading pins the CoM fraction at ``at`` (sd 0.01), the identify fit's role at (b); ``fail``
    raises instead. Counts its calls."""

    feature_names = None

    def __init__(self, at=0.5, fail=False):
        self.at, self.fail, self.calls = at, fail, 0

    def loglik(self, obs, X, a, force, terms=None, thrown=None):
        self.calls += 1
        if self.fail:
            raise ValueError("boom")
        assert obs["probe"] == "identify" and len(X) == len(a["mass"])
        return -0.5 * ((np.asarray(a["com_frac"], dtype=float) - self.at) / 0.01) ** 2, {"used": ["kind", "joint"], "rejected": []}


def _identify(**kw):
    """The case's identify probe as identify_probe returns it: measured at the pinch of s 0.5."""
    return {"kind": "measured", "thrown": False, "damaged": False, "damage": None, "artifact": False, "seconds": 18.2,
            "reading": {"m_hat": 0.2}, "s": 0.5, "point": [0.0, 0.0, 0.76], "yaw": 0.0, "pose": _pose(0.5), "force": 20.0,
            "geometry": _geometry(), **kw}


def _with_identify(prims, pm):
    prims.probe_models = {"identify": pm}
    return prims


def _run_b(prims, env, b, identify=None, k=5):
    return cg.case_b(prims, env, "x", "plate", 100.0, 3.0, None, {"mass": 0.2, "com_frac": 0.5}, b, np.random.default_rng(0),
                     full=False, rng_whole_log=np.random.default_rng(1), identify=identify, rng_identify=np.random.default_rng(k))


def test_post_identify_adds_the_cases_identify_probe_to_belief_one(monkeypatch):
    """Belief 1 stays split on the CoM (the cut log's fit says nothing), so it pays for the gentle carry; the identify
    probe pins the CoM fraction at 0.5, so belief 1 with it takes belief 2's carry: its gain on the yardstick is the
    whole EVPI (0.05). The probe's update is the skill's (tools.entry_loglik), with its terms recorded."""
    prims, env, b = _b_case(monkeypatch)
    pm = _IdentifyModel(at=0.5)
    r = _run_b(_with_identify(prims, pm), env, b, _identify())
    assert r["evpi"] == pytest.approx(0.05) and r["choice1"] == "gentle|drop_release"
    p = r["post_identify"]
    assert p["choice"] == "standard|drop_release" and p["changed"] and not p["by_rule"]
    assert r["gain_identify"] == pytest.approx(0.05) and p["used_terms"] == ["kind", "joint"]
    assert pm.calls >= 1 and p["ess"]["ess_final"] > 0


def test_post_identify_leaves_the_primary_columns_alone(monkeypatch):
    """The new column runs on its own stream after the primary: every key of a run without it is the same with it."""
    prims, env, b = _b_case(monkeypatch)
    without = _run_b(prims, env, b)
    with_ = _run_b(_with_identify(prims, _IdentifyModel(at=1.0)), env, b, _identify())
    assert "post_identify" not in without and "gain_identify" not in without
    assert set(with_) - set(without) == {"post_identify", "gain_identify"}
    assert {k: with_[k] for k in without} == without


def test_post_identify_is_belief_one_when_the_rule_already_ran_identify(monkeypatch):
    """No double count: when the rule's own probes held identify, belief 1 already has its evidence, so post_identify is
    belief 1 (gain 0) and the case's extra probe is not applied."""
    prims, env, b = _b_case(monkeypatch)
    first = cg.first_attempt

    def ran_identify(*a, **k):
        fa = first(*a, **k)
        fa["probes"] = [{"probe": "identify", "kind": "measured", "artifact": False}]
        return fa

    monkeypatch.setattr(cg, "first_attempt", ran_identify)
    pm = _IdentifyModel(at=0.5)
    r = _run_b(_with_identify(prims, pm), env, b, _identify())
    assert r["post_identify"]["by_rule"] and r["gain_identify"] == 0.0 and pm.calls == 0
    assert r["post_identify"]["choice"] == r["choice1"]


def test_a_failed_post_identify_is_recorded_and_leaves_the_case_alone(monkeypatch):
    """The new column cannot fail the case: an error in its update is recorded (gain None, counted apart in the
    summary) and the primary columns stay as they are."""
    prims, env, b = _b_case(monkeypatch)
    without = _run_b(prims, env, b)
    r = _run_b(_with_identify(prims, _IdentifyModel(fail=True)), env, b, _identify())
    assert r["post_identify"] == {"error": "ValueError: boom"} and r["gain_identify"] is None
    assert {k: r[k] for k in without} == without


def test_post_identify_counts_zero_for_a_case_that_never_reaches_b(monkeypatch):
    _ended(monkeypatch, "abstained")
    args = (None, SimpleNamespace(), "x", "plate", 100.0, 3.0, None, {}, None, None)
    r = cg.case_b(*args, identify=_identify())
    assert r["gain_identify"] == 0.0 and r["post_identify"] is None
    assert "gain_identify" not in cg.case_b(*args)


def test_a_late_post_identify_error_is_recorded_and_leaves_the_case_alone(monkeypatch):
    """Every step of the new column is guarded, its diagnostic (cover80) and the yardstick lookup too: an error after
    the update is recorded (gain None) and the primary columns stay as they are."""
    prims, env, b = _b_case(monkeypatch)
    real, calls = cg.cover80, []
    monkeypatch.setattr(cg, "cover80", lambda *a, **k: calls.append(1) or real(*a, **k))
    without = _run_b(prims, env, b)
    k = len(calls)                      # the primary's calls; post_identify's comes after them, last

    def late(*a, **kw):
        calls.append(1)
        if len(calls) > 2 * k:
            raise RuntimeError("late")
        return real(*a, **kw)

    monkeypatch.setattr(cg, "cover80", late)
    r = _run_b(_with_identify(prims, _IdentifyModel(at=0.5)), env, b, _identify())
    assert len(calls) == 2 * k + 1
    assert r["post_identify"] == {"error": "RuntimeError: late"} and r["gain_identify"] is None
    assert {key: r[key] for key in without} == without


def test_post_identify_reads_the_geometry_its_probe_saw_not_the_scene_at_b(monkeypatch):
    """The lab's geometry is in the world frame, and a rule probe (a push) can move the object before (b). The identify
    probe ran at the case's fresh reset, so its entry's features come from the geometry identify_probe saw there, not
    from the scene at (b)."""
    prims, env, b = _b_case(monkeypatch)
    g = _geometry()
    at_reset = Geometry(centres=g.centres + [0.05, 0.0, 0.0], normals=g.normals, com=g.com + [0.05, 0.0, 0.0],
                        length=g.length, base_extent=g.base_extent, com_height=g.com_height,
                        footprint_half_width=g.footprint_half_width)
    real, seen = cg.extract, []
    monkeypatch.setattr(cg, "extract", lambda geometry, *a, **k: seen.append(geometry) or real(geometry, *a, **k))
    r = _run_b(_with_identify(prims, _IdentifyModel(at=0.5)), env, b, _identify(geometry=at_reset))
    assert sum(x is at_reset for x in seen) == 1 and "error" not in r["post_identify"]


def test_post_identify_has_no_gain_for_a_declined_or_failed_probe_even_without_b(monkeypatch):
    """A declined or failed identify probe gives its case no post_identify belief in any run: gain None (counted apart),
    also in a run that never reaches (b), so the clean mean never counts it as 0."""
    _ended(monkeypatch, "abstained")
    args = (None, SimpleNamespace(), "x", "plate", 100.0, 3.0, None, {}, None, None)
    for probe in ({"declined": "every pinch is wider than the gripper"}, {"error": "RuntimeError: x"}):
        r = cg.case_b(*args, identify=dict(probe, wall_s=1.0))
        assert r["gain_identify"] is None and r["post_identify"] == probe


class _ProbeEnv(_SceneEnv):
    """_SceneEnv with the identify probe: records its proposal and force, returns ``res``."""

    def __init__(self, res=None):
        super().__init__()
        self.ran, self.order = [], []
        self.res = res or SimpleNamespace(kind="measured", thrown=False, damage=None, artifact=False, seconds=18.2,
                                          reading={"m_hat": 0.21, "eps": [0.1]})

    def nominal_mass(self, o):
        return 0.2

    def probe_identify(self, p, force):
        self.ran.append((p.s, force))
        self.order.append("probe")
        return self.res


def test_identify_probe_runs_the_menus_probe_identify_row():
    """At the mid usable proposal with the weigh squeeze of the standard force, as the menu's probe rows; the reading's
    hidden noise stays out, as tools._run_probe keeps it out."""
    from rpent.skills.pick_place.probes import weigh_force
    from rpent.skills.pick_place.tools import standard_force

    env = _ProbeEnv()
    p = cg.identify_probe(env, "x")
    f = weigh_force(standard_force(0.2))
    assert env.ran == [(0.5, f)] and p["s"] == 0.5 and p["force"] == f and p["pose"] == _pose(0.5)
    assert p["kind"] == "measured" and p["reading"] == {"m_hat": 0.21} and not p["thrown"] and not p["damaged"]
    assert p["seconds"] == 18.2 and not p["artifact"]


def test_identify_probe_marks_a_damaging_or_throwing_probe():
    env = _ProbeEnv(SimpleNamespace(kind="damaged", thrown=False, damage={"label": "crushed"}, artifact=True, seconds=3.0, reading={}))
    p = cg.identify_probe(env, "x")
    assert p["damaged"] and p["artifact"] and p["damage"] == {"label": "crushed"}
    env = _ProbeEnv(SimpleNamespace(kind="lost", thrown=True, damage=None, artifact=False, seconds=9.0, reading={}))
    assert cg.identify_probe(env, "x")["thrown"]


def test_identify_probe_declines_without_a_usable_proposal():
    env = _ProbeEnv()
    env.proposals = lambda: [P(0.5, width=0.2)]
    p = cg.identify_probe(env, "x")
    assert "wider" in p["declined"] and env.ran == []


def test_identify_probe_keeps_the_geometry_it_saw_before_the_probe():
    """The probe entry's features need the object's geometry at the probe (the lab's geometry is in the world frame):
    identify_probe takes it before the probe moves the object. It stays out of the record (IDENTIFY_KEYS)."""
    env = _ProbeEnv()
    env.geometry = lambda: env.order.append("geometry") or "G0"
    p = cg.identify_probe(env, "x")
    assert p["geometry"] == "G0" and env.order == ["geometry", "probe"] and "geometry" not in cg.IDENTIFY_KEYS


def test_run_case_runs_one_identify_probe_per_case_and_resets_before_the_first_setting(monkeypatch):
    """post_identify's probe runs once per case, from the case's fresh reset, after W0's exact prior is drawn; then the
    lab is reset (and checked against the first reset) before the first prior and setting, and every run gets the same
    probe and its own stream. Without the flag: no probe and no extra reset."""
    from robots.mujoco_lab import adapter, first_menus
    from robots.mujoco_lab import calibrate as cal
    from rpent.skills.pick_place import tools

    monkeypatch.setattr(cal, "case_draw", lambda *a, **k: ({}, None, "plate", {"fill": 0.0, "frag": None}))
    monkeypatch.setattr(adapter, "object_props", lambda o: {"mass": 1.0})
    monkeypatch.setattr(tools, "_particles", lambda prims: 10)
    monkeypatch.setattr(tools, "_spread_kw", lambda prims: {})
    settings = [("k100_A3", 100.0, 3.0), ("k30_A3", 30.0, 3.0)]
    for flag in (True, False):
        env, seen = _ProbeEnv(), []
        env.joint_sensing, env.reset_info, env.resets = True, {"at_rest": True}, 0
        env.reset = lambda *a, env=env, **k: setattr(env, "resets", env.resets + 1)
        monkeypatch.setattr(first_menus, "exact_priors", lambda *a, env=env, **k: (
            {"mass": 0.2, "com_frac": 0.5}, {"exact_both": lambda *a, **k: env.order.append("w0") or "w0"}))
        monkeypatch.setattr(cg, "case_b", lambda prims, e, *a, seen=seen, **k: seen.append((e.resets, k)) or {"moved": False})
        recs = cg.run_case(SimpleNamespace(env=env), "x", 0, {"seed": 11, "fold": 0, "settings": settings, "generic": {"x": None},
                                                          "post_identify": flag},
                           {"class": None, "generic": None}, None, None)
        assert len(recs) == 4 and all(r["reset_match"] for r in recs)
        if flag:
            assert env.order == ["w0", "probe"] and len(env.ran) == 1
            assert [n for n, _ in seen] == [2, 2, 2, 2] and all(k["identify"] is seen[0][1]["identify"] for _, k in seen)
            assert all(k["rng_identify"] is not None for _, k in seen)
            assert recs[0]["identify_probe"] == {"kind": "measured", "thrown": False, "damaged": False, "artifact": False,
                                                 "seconds": 18.2, "wall_s": recs[0]["identify_probe"]["wall_s"]}
        else:
            assert env.ran == [] and [n for n, _ in seen] == [1, 1, 1, 1] and all(k.get("identify") is None for _, k in seen)
            assert all("identify_probe" not in r for r in recs)


def test_the_cell_reports_post_identify_with_thrown_damaged_or_flagged_probes_apart():
    """The identify outcome counts, the mean gain over the cases whose probe was clean (with the same cases' EVPI and the
    share it closes), the cases whose probe threw, damaged or was flagged counted apart with their own mean (not
    dropped), the cases where the rule ran identify, and failed updates counted and left out."""
    recs = _recs([None, 0.3, 0.1, 0.2, None, 0.4])
    probe = {"kind": "measured", "thrown": False, "damaged": False, "artifact": False}
    for r, g in zip(recs, (0.0, 0.3, 0.05, 0.2, 0.0, 0.1)):
        r.update(gain_identify=g, identify_probe=dict(probe), post_identify={"by_rule": False} if r["reached"] else None)
    recs[2]["post_identify"] = {"by_rule": True}
    recs[3]["identify_probe"].update(kind="damaged", damaged=True)
    recs[4]["identify_probe"].update(artifact=True)
    recs[5]["identify_probe"].update(thrown=True)
    recs.append({**_recs([0.5])[0], "object": "o7", "idx": 7, "gain_identify": None, "identify_probe": dict(probe),
                 "post_identify": {"error": "ValueError: boom"}})
    c = cg.cell(recs, np.random.default_rng(0), 50)["post_identify"]
    assert c["n"] == 3 and c["gain"]["mean"] == pytest.approx(0.35 / 3)
    assert c["evpi_same_cases"] == pytest.approx(0.4 / 3) and c["share_closed"] == pytest.approx(0.875)
    assert c["apart"] == {"n": 3, "thrown": 1, "damaged": 1, "artifact": 1, "gain_mean": pytest.approx(0.1), "n_no_gain": 0}
    assert c["n_errors"] == 1 and c["n_by_rule"] == 1 and c["kinds"] == {"measured": 6, "damaged": 1}
    md = cg.markdown(cg.summarize(recs, n_boot=50, seed=0))
    assert "## post_identify" in md and "18 s" in md
    assert "post_identify" not in cg.markdown(cg.summarize(_recs([0.1]), n_boot=50, seed=0))
    assert "post_identify" not in cg.cell(_recs([0.1]), np.random.default_rng(0), 50)


def test_the_cell_counts_a_flagged_probe_apart_even_when_its_update_failed():
    """The outcome counts (thrown, damaged, flagged) cover every probe: a flagged probe whose update failed is counted
    apart, its missing gain left out of the apart mean and counted there, not among the errors."""
    recs = _recs([0.3, 0.1, 0.2])
    probe = {"kind": "measured", "thrown": False, "damaged": False, "artifact": False}
    for r in recs:
        r.update(gain_identify=0.1, identify_probe=dict(probe), post_identify={"by_rule": False})
    recs[1].update(gain_identify=None, post_identify={"error": "ValueError: boom"})
    recs[1]["identify_probe"].update(kind="lost", thrown=True)
    c = cg.cell(recs, np.random.default_rng(0), 50)["post_identify"]
    assert c["apart"] == {"n": 1, "thrown": 1, "damaged": 0, "artifact": 0, "gain_mean": None, "n_no_gain": 1}
    assert c["n_errors"] == 0 and c["n"] == 2


def test_the_post_identify_block_leaves_the_summarys_primary_numbers_alone():
    """Its bootstrap draws from its own stream: every primary number and verdict of the summary is the same with the
    post_identify fields and without them (two settings, so a shared stream would move the second cell's intervals)."""
    recs = []
    for setting in ("k100_A3", "k30_A3"):
        rs = _recs([None, 0.3, 0.1, 0.2, None, 0.4], setting=setting)
        for r, g in zip(rs, (0.0, 0.3, 0.05, 0.2, 0.0, 0.1)):
            r.update(gain_identify=g, identify_probe={"kind": "measured"}, post_identify=None)
        recs += rs
    plain = [{k: v for k, v in r.items() if k not in ("gain_identify", "identify_probe", "post_identify")} for r in recs]
    a, b = cg.summarize(plain, n_boot=50, seed=0), cg.summarize(recs, n_boot=50, seed=0)
    for setting, block in a["cells"].items():
        for p, c in block.items():
            assert "post_identify" in b["cells"][setting][p]
            assert {k: v for k, v in b["cells"][setting][p].items() if k != "post_identify"} == c
    assert {k: v for k, v in b.items() if k != "cells"} == {k: v for k, v in a.items() if k != "cells"}


def _identify_bundle(tmp_path, with_model):
    b = tmp_path / "B"
    (b / "folds" / "fold_0").mkdir(parents=True)
    (b / "folds" / "folds.json").write_text(json.dumps({"o1": 0}))
    if with_model:
        (b / "folds" / "fold_0" / "probe_identify.joblib").write_text("x")
    return b


def test_run_refuses_post_identify_on_a_bundle_without_the_identify_model(tmp_path):
    b = _identify_bundle(tmp_path, False)
    with pytest.raises(SystemExit, match="probe_identify"):
        cg.main(["run", "--bundle", str(b), "--scenes", "S", "--out", str(tmp_path / "out"), "--post-identify"])


def test_the_post_identify_flag_reaches_the_workers_and_the_config(tmp_path, monkeypatch):
    from robots.mujoco_lab import first_menus, identify_fit

    b = _identify_bundle(tmp_path, True)
    sc = tmp_path / "scenes.json"
    sc.write_text(json.dumps({"o1": {"flat": False}}))
    monkeypatch.setattr(identify_fit, "require_bundle_generator", lambda d: "v11")
    monkeypatch.setattr(first_menus, "prior_tables", lambda *a: ({"generic": {"o1": {}}}, {}))
    jobs = []

    class _Pool:
        def __init__(self, n):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def imap_unordered(self, f, js):
            jobs.extend(js)
            return iter([])

    monkeypatch.setattr(cg.mp, "get_context", lambda kind: SimpleNamespace(Pool=_Pool))
    for flag, out in ((True, "on"), (False, "off")):
        jobs.clear()
        cg.main(["run", "--bundle", str(b), "--scenes", str(sc), "--repeats", "1", "--out", str(tmp_path / out)]
                + (["--post-identify"] if flag else []))
        assert [j[2]["post_identify"] for j in jobs] == [flag]
        assert json.load(open(tmp_path / out / "run_config.json"))["post_identify"] is flag


SIM_CASE = ("ycb__tuna_can", 7, ("k30_A3", 30.0, 3.0))       # reached (b) on m2b2_v12 in 32 s


@pytest.mark.skipif(not all(os.environ.get(k) for k in ("RPENT_V12B_BUNDLE", "RPENT_MUJOCO_SIM_DIR", "RPENT_COMGAP_SCENES")),
                    reason="needs the m2b2_v12b bundle, the arm port's sim and the scenes file")
def test_post_identify_leaves_a_real_cases_primary_columns_alone(monkeypatch):
    """The amendment's condition (coordinator, 2026-11-06), on the arm sim and m2b2_v12b: one case that reaches (b), run
    with and without post_identify, has the same primary columns (the probe runs from its own fresh reset, the lab is
    reset before the first setting, and the new column draws from its own stream)."""
    from robots.mujoco_lab import calibrate as cal
    from robots.mujoco_lab.first_menus import prior_tables
    from robots.mujoco_lab.prior import fragility_world

    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    bundle, o, (idx, setting) = os.environ["RPENT_V12B_BUNDLE"], SIM_CASE[0], SIM_CASE[1:]
    tables, _ = prior_tables(bundle, os.environ["RPENT_COMGAP_SCENES"], {})
    fold = json.load(open(os.path.join(bundle, "folds", "folds.json")))[o]
    runs = []
    for flag in (False, True):
        prims = cal._primitives(os.path.join(bundle, "folds", f"fold_{fold}"), None, None)
        prims.force_noise = cal.job_force_noise(None, prims.manifest)
        prims.env.force_noise = prims.force_noise
        s = {"seed": 11, "fold": fold, "settings": [setting], "generic": {o: tables["generic"][o]}, "post_identify": flag}
        runs.append(cg.run_case(prims, o, idx, s, {"class": prims.prior}, prims.prior, fragility_world(prims.manifest))[0])
    without, with_ = runs
    assert without["reached"] and "error" not in without and "error" not in with_
    assert set(with_) - set(without) == {"post_identify", "gain_identify", "identify_probe"}
    assert "error" not in with_["post_identify"] and with_["identify_probe"]["kind"] is not None
    assert {k: with_[k] for k in without if k != "seconds"} == {k: v for k, v in without.items() if k != "seconds"}


@pytest.mark.skipif(not all(os.environ.get(k) for k in ("RPENT_V12_BUNDLE", "RPENT_MUJOCO_SIM_DIR", "RPENT_COMGAP_SCENES")),
                    reason="needs the m2b2_v12 bundle, the arm port's sim and the scenes file")
def test_the_v12_records_reproduce_on_this_code(monkeypatch):
    """The merge of arm-identify (the identify probe model, the attempt's joint term) leaves m2b2_v12's com_gap alone:
    a case of the full v12 run (data/com_gap/full_v12_n20000), rerun on this code, gives its record again (all but the
    wall time). The v12 bundle has no identify model and no joint readout term, so neither may reach its menus."""
    import gzip

    from robots.mujoco_lab import calibrate as cal
    from robots.mujoco_lab.first_menus import prior_tables
    from robots.mujoco_lab.prior import fragility_world

    monkeypatch.setenv("RPENT_LAB_GENERATOR", "v11")
    bundle, o, (idx, setting) = os.environ["RPENT_V12_BUNDLE"], SIM_CASE[0], SIM_CASE[1:]
    path = os.path.join(os.path.dirname(cg.__file__), "data", "com_gap", "full_v12_n20000", "records.jsonl.gz")
    stored = [r for r in map(json.loads, gzip.open(path, "rt"))
              if (r["object"], r["idx"], r["prior"], r["setting"]) == (o, idx, "class", setting[0])]
    tables, _ = prior_tables(bundle, os.environ["RPENT_COMGAP_SCENES"], {})
    fold = json.load(open(os.path.join(bundle, "folds", "folds.json")))[o]
    prims = cal._primitives(os.path.join(bundle, "folds", f"fold_{fold}"), None, None)
    assert "identify" not in (getattr(prims, "probe_models", None) or {})
    prims.force_noise = cal.job_force_noise(None, prims.manifest)
    prims.env.force_noise = prims.force_noise
    s = {"seed": 11, "fold": fold, "settings": [setting], "generic": {o: tables["generic"][o]}, "flat": stored[0]["flat"]}
    rec = cg.run_case(prims, o, idx, s, {"class": prims.prior}, prims.prior, fragility_world(prims.manifest))[0]
    assert len(stored) == 1 and stored[0]["reached"]
    assert {k: v for k, v in rec.items() if k != "seconds"} == {k: v for k, v in stored[0].items() if k != "seconds"}
