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

"""Experiment 1, EVPI per hidden property (evpi_props; daily-logs 2026-11-05-evpi-decomposition): the truth particle, the
reveals, the shared menus, the choices, J at the truth, the groups, the checks and the summary."""

import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

from robots.mujoco_lab import evpi_props as ev
from robots.mujoco_lab.fragility import Thresholds
from rpent.skills.pick_place.belief import ParticleBelief
from rpent.skills.pick_place.costs import COST, TIME_COST_PER_S, care_scale
from rpent.skills.pick_place.probes import PROBE_SECONDS

LAW = {"f_damage_n": {"median": 50.0, "log_sd": 0.5}, "k_n_per_mm": {"median": 10.0, "log_sd": 0.4}, "rho": 0.6,
       "v_bruise_ms": {"median": 1.0, "log_sd": 0.3}}
TH = Thresholds(laws={"soft": LAW}, slosh_zeta=0.1)
CRUSH = {"class": "crushable", "law": "soft", "fillable": False, "cavity": None}
POSE = {"object_id": "toy", "pivot": {"centre": [0.0, 0.0, 0.0], "axis": [1.0, 0.0, 0.0], "lo": -0.1, "L": 0.2},
        "com_obj_nominal": [0.0, 0.0, 0.01], "nominal_mass": 0.5}


def _prior(entry=CRUSH):
    from robots.mujoco_lab.prior import make_prior

    return make_prior({"toy": entry}, TH, generator="v11")


def _truth(entry=CRUSH, com_s=0.7, z=(0.3, -1.2, 0.8)):
    from robots.mujoco_lab.fragility import map_latent

    hidden = {"fill": 0.0, "frag": map_latent(z, entry, TH), "spill_angle_deg": None}
    return ev.truth_particle(POSE, {"mass_scale": 1.3, "mu": 0.42, "com_s": com_s}, 1, hidden, entry)


# ---------------------------------------------------------------- the truth and the reveals

def test_the_truth_particle_has_the_priors_keys_and_the_generators_values():
    b = _prior()(POSE, 5, np.random.default_rng(0))
    t = _truth()
    assert set(t) == set(b.particles[0])
    assert t["mass"] == pytest.approx(0.5 * 1.3)                      # v11: nominal x scale, the CoM point mass inside
    assert t["com_frac"] == pytest.approx(0.7)
    assert t["f_damage_n"] == pytest.approx(50.0 * math.exp(0.5 * 0.3))
    assert t["frag_z"] == [0.3, -1.2, 0.8]


def _b0(n=400, entry=CRUSH, seed=1):
    return _prior(entry)(POSE, n, np.random.default_rng(seed))


@pytest.mark.parametrize("S,keys", [("mass", ("mass", "mass_scale")), ("mu", ("mu",)),
                                    ("com", ("com_obj", "com_frac", "com_branch", "com_s", "com_end", "f"))])
def test_a_reveal_sets_its_keys_in_every_particle_and_keeps_the_rest(S, keys):
    b0, t = _b0(), _truth()
    b, info = ev.reveal(b0, S, t, CRUSH, TH, None, np.random.default_rng(0), 0.2)
    assert not info["weighted"] and info["ess_final"] == pytest.approx(len(b0))
    np.testing.assert_allclose(b.weights, b0.weights, rtol=1e-12)          # ParticleBelief renormalises
    for p, q in zip(b.particles, b0.particles):
        assert all(p[k] == t[k] for k in keys)
        assert all(p[k] == q[k] for k in p if k not in keys)


def test_the_damage_reveal_keeps_each_particles_own_stiffness_latent():
    b0, t = _b0(), _truth()
    b, _ = ev.reveal(b0, "damage", t, CRUSH, TH, None, np.random.default_rng(0), 0.2)
    rho = LAW["rho"]
    for p, q in zip(b.particles[:20], b0.particles[:20]):
        assert p["f_damage_n"] == t["f_damage_n"] and p["v_bruise_ms"] == t["v_bruise_ms"]
        assert p["frag_z"] == [t["frag_z"][0], q["frag_z"][1], t["frag_z"][2]]
        z = rho * t["frag_z"][0] + math.sqrt(1 - rho * rho) * q["frag_z"][1]
        assert p["k_n_per_mm"] == pytest.approx(10.0 * math.exp(0.4 * z))
        assert p["mass"] == q["mass"] and p["mu"] == q["mu"]


def test_all_is_the_truth_as_a_point_mass():
    b0, t = _b0(), _truth()
    b, info = ev.reveal(b0, "all", t, CRUSH, TH, None, np.random.default_rng(0), 0.2)
    assert len(b) == 1 and b.particles[0] == t and info["ess_final"] == 1.0


def _filled(n, rng):
    """A toy fillable prior: the fill 0 with p 0.5, else U[0.3, 0.9]; the mass grows with the fill."""
    fill = np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0.3, 0.9, n))
    parts = [{"mass": 0.3 + 0.2 * f, "mu": 0.5, "com_obj": [0.0, 0.0, 0.02 - 0.02 * f], "fill": float(f), "spill_angle_deg": 40.0 * f,
              "liquid_mass": 0.2 * f, "ancestor": i, **{k: None for k in ev.FRAG_KEYS}, "frag_z": [0.0, 0.0, 0.0]}
             for i, f in enumerate(fill)]
    return ParticleBelief(parts, np.ones(n))


def test_a_fillable_objects_coupled_reveal_weights_the_particles_before_it_sets_the_keys():
    entry = {"class": "none", "law": None, "fillable": True, "cavity": None}
    b0 = _filled(4000, np.random.default_rng(0))
    t = dict(b0.particles[0], mass=0.3 + 0.2 * 0.6, fill=0.6, spill_angle_deg=24.0, liquid_mass=0.12, com_obj=[0.0, 0.0, 0.008])
    b, info = ev.reveal(b0, "damage", t, entry, None, lambda n, r: _filled(n, r), np.random.default_rng(1), 0.2)
    assert info["weighted"]
    assert all(p["fill"] == 0.6 and p["spill_angle_deg"] == 24.0 for p in b.particles)
    a = b.arrays()
    assert abs(float(b.weights @ a["mass"]) - 0.42) < 0.01               # the mass follows the revealed fill


# ---------------------------------------------------------------- choices and J at the truth

def test_choice_of_reads_the_first_decision_as_first_attempt_does():
    assert ev.choice_of({"recommendation": "decline", "best_option": "commit_std"}) == "decline"
    assert ev.choice_of({"recommendation": "probe", "recommended_option": "probe_weigh", "best_option": "x"}) == "probe_weigh"
    assert ev.choice_of({"recommendation": "commit", "best_option": "slow_lift"}) == "slow_lift"
    assert ev.choice_of({"error": "boom"}) == "error"
    assert ev.choice_of({"assessed": False}) == "unreachable"


def test_the_action_key_tells_two_rows_of_one_id_apart():
    m1 = {"regrasp_com": SimpleNamespace(action={"s": 0.3, "force": 20.0})}
    m2 = {"regrasp_com": SimpleNamespace(action={"s": 0.7, "force": 20.0})}
    assert ev.action_key("regrasp_com", m1) != ev.action_key("regrasp_com", m2)
    assert ev.action_key("regrasp_com", m1) == ev.action_key("regrasp_com", {"regrasp_com": SimpleNamespace(action={"force": 20.0, "s": 0.3})})
    assert ev.action_key("decline", m1) == "decline"


class _Probe:
    """Results that name the particle exactly; the probe ends the case with probability ``r``."""
    classes = ("slipped", "held")

    def __init__(self, r=0.0):
        self.r = r

    def p_kind(self, X):
        return np.full((len(X), 2), 0.5)

    def terminal_risk(self, X):
        return np.full(len(X), self.r)

    def risk(self, X):
        return np.full(len(X), self.r)

    def sample(self, x_row, particle, force, rng, p=None):
        return {"kind": "held", "reading": particle["mass"]}

    def loglik(self, obs, X, arrays, force, P=None, **kw):
        return np.where(np.isclose(arrays["mass"], obs["reading"]), 0.0, -1e9), {"used": ["kind", "reading"]}


def _probe_case(monkeypatch, r):
    from rpent.skills.pick_place import features

    monkeypatch.setattr(features, "particle_matrix", lambda base, a, pose, force, names: np.arange(len(a["mass"]), dtype=float)[:, None])
    monkeypatch.setattr(ev, "model_names", lambda pm: ())
    truth_cost = {"commit_std": 9.0, "slow_lift": 2.0}
    monkeypatch.setattr(ev, "row_true_cost", lambda prims, o, tb, care, A, reset_s, price: truth_cost[o.id])
    parts = [{"mass": m, "mu": 0.5, "com_obj": [0.0, 0.0, 0.0]} for m in (0.2, 0.9)]
    b, tb = ParticleBelief(parts, np.ones(2)), ParticleBelief([parts[1]], np.ones(1))
    rows = {"commit_std": SimpleNamespace(id="commit_std", kind="commit", action={"destination": "bin"}, _costs=np.array([1.0, 9.0])),
            "slow_lift": SimpleNamespace(id="slow_lift", kind="lower_risk", action={"destination": "bin"}, _costs=np.array([9.0, 2.0]))}
    o = SimpleNamespace(id="probe_weigh", kind="probe", action={"probe": "weigh", "s": 0.5, "force": 10.0}, _base={}, _pose={})
    prims = SimpleNamespace(probe_models={"weigh": _Probe(r)}, model=None)
    return ev.probe_true_cost(prims, o, {**rows, "probe_weigh": o}, b, tb, "bin", 100.0, 5.0, 1.0, np.random.default_rng(0), k=20)


def test_a_probe_costs_its_time_and_the_row_its_result_picks_scored_at_the_truth(monkeypatch):
    out = _probe_case(monkeypatch, 0.0)
    assert out["picks"] == {"slow_lift": 20}
    assert out["J"] == pytest.approx(TIME_COST_PER_S * PROBE_SECONDS["weigh"] + 2.0)


def test_a_probe_that_ends_the_case_at_the_truth_costs_the_irreversible_cost_and_A(monkeypatch):
    out = _probe_case(monkeypatch, 1.0)
    assert out["J"] == pytest.approx(TIME_COST_PER_S * PROBE_SECONDS["weigh"] + COST["dropped_high"] * care_scale(100.0) + 5.0)


# ---------------------------------------------------------------- groups, checks and the summary

def _rec(choices, evpi_all=0.0, obj="o1", setting="k100_A3", prior="class", guard_all=False, J_all=None, all_rows=None,
         ess=20000.0, evpi=None, evoi=None):
    actions = {k: v for k, v in choices.items()}
    J = {k: 1.0 for k in choices}
    J["all"] = 1.0 - evpi_all if J_all is None else J_all
    menu = {k: {"recommendation": "decline" if c == "decline" else "commit", "best_option": "commit_std", "recommended_option": None,
                "without_probe": None, "guard_flag": guard_all if k == "all" else False,
                "first_menu": {"best_cost": 1.0, "recommended_cost": None}} for k, c in choices.items()}
    return {"object": obj, "idx": 0, "prior": prior, "setting": setting, "abstain_cost": 3.0, "class": "none", "fillable": False,
            "load_bin": "[0, 0.25)", "choice": dict(choices), "action": actions, "J": J, "menu": menu,
            "evpi": evpi or {S: (evpi_all if S == "all" else 0.0) for S in ev.REVEALS},
            "ess": {k: {"ess_final": ess} for k in choices}, "evoi": evoi or {}, "evpi_belief": None,
            "all_rows": all_rows if all_rows is not None else {"commit_std": J["all"]}}


SAME = {k: "decline" for k in ("prior",) + ev.REVEALS}


def test_the_groups_come_from_the_choices():
    g = ev.groups(_rec(SAME))
    assert g == {"invariant": True, "critical": False, "invariant_id": True, "abstain_dominated": True}
    g = ev.groups(_rec(dict(SAME, mass="commit_std", all="commit_std")))
    assert g["critical"] and not g["abstain_dominated"]


def test_the_checks_count_replay_mismatches_negative_evpi_and_an_all_choice_above_the_minimum():
    a = _rec(SAME, obj="a")
    b = _rec(SAME, obj="b", evpi_all=-0.5, guard_all=True)
    c = _rec(SAME, obj="c", J_all=2.0, all_rows={"commit_std": 1.5})
    cg = {("a", 0, "class", "k100_A3"): {"recommendation": "decline", "best_option": "commit_std", "guard_flag": False,
                                         "first_menu": {"best_cost": 1.0, "recommended_cost": None}},
          ("b", 0, "class", "k100_A3"): {"recommendation": "commit", "best_option": "commit_std", "guard_flag": False,
                                         "first_menu": {"best_cost": 1.0, "recommended_cost": None}}}
    out = ev.checks([a, b, c], cg)
    assert out["replay"]["compared"] == 2 and out["replay"]["mismatch"] == 1
    assert out["evpi_all_negative"] == {"n": 1, "guard_under_truth": 1}
    assert out["a_all_is_min"]["off"] == 1                    # c: J(a_all) 2.0 above min(1.5, A 3.0)


def test_a_property_matters_only_when_its_lower_bound_clears_the_bar():
    big = [_rec(SAME, obj=f"o{i}", evpi={S: (0.5 if S in ("mass", "all") else 0.0) for S in ev.REVEALS}) for i in range(12)]
    s = ev.summarize(big, n_boot=200)
    t = s["cells"]["k100_A3"]["class"]["evpi"]
    assert t["mass"]["reading"] == "matters" and t["mu"]["reading"] == "no detectable value"


def test_the_ess_table_leaves_out_a_case_with_a_thin_reveal():
    ok, thin = _rec(SAME, obj="a"), _rec(SAME, obj="b")
    thin["ess"]["mass"]["ess_final"] = 40.0
    assert ev.ess_ok(ok) and not ev.ess_ok(thin)
    s = ev.summarize([ok, thin], n_boot=50)
    assert s["cells"]["k100_A3"]["class"]["n"] == 2 and s["cells"]["k100_A3"]["class"]["n_ess_ok"] == 1


def _shard(tmp_path, name, recs, **cfg):
    d = tmp_path / name
    d.mkdir()
    (d / "run_config.json").write_text(json.dumps({"seed": 11, "code_revision": "abc", "dirty": False, **cfg}))
    (d / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    return str(d)


def test_load_joins_shards_and_refuses_a_repeated_case_or_another_config(tmp_path):
    a = _shard(tmp_path, "a", [_rec(SAME, obj="a")])
    b = _shard(tmp_path, "b", [_rec(SAME, obj="b")])
    assert len(ev.load([a, b])[0]) == 2
    with pytest.raises(SystemExit, match="repeated"):
        ev.load([a, _shard(tmp_path, "c", [_rec(SAME, obj="a")])])
    with pytest.raises(SystemExit, match="config differs"):
        ev.load([a, _shard(tmp_path, "d", [_rec(SAME, obj="d")], seed=12)])


# ---------------------------------------------------------------- the shared menus (needs the sim and the m2b2_v12 bundle)

SIM = os.environ.get("RPENT_MUJOCO_SIM_DIR")
V12 = os.environ.get("EVPI_TEST_BUNDLE", "/home/chungyili/Codes/RPent-arm/robots/mujoco_lab/data/m2b2_v12")


@pytest.mark.skipif(not (SIM and os.path.exists(os.path.join(V12, "folds", "folds.json"))), reason="sim dir or m2b2_v12 bundle missing")
def test_shared_menus_give_every_settings_menu_exactly():
    """One case's prior and its mass reveal at 300 particles: every option's expected cost, the recommendation and the
    guard under two settings, with and without the memos."""
    from robots.mujoco_lab import calibrate as cal
    from robots.mujoco_lab.adapter import object_props
    from robots.mujoco_lab.prior import fragility_world

    o = "hot3d__mug"
    folds = json.load(open(os.path.join(V12, "folds", "folds.json")))
    prims = cal._primitives(os.path.join(V12, "folds", f"fold_{folds[o]}"), None, None)
    prims.n_particles = 300
    world = fragility_world(prims.manifest)
    env = prims.env
    theta, com_end, dest, hidden = cal.case_draw(11, o, 3, object_props(o)["mass"], False, fragility=world)
    env.slosh_zeta, env.slosh_scale = world[1].slosh_zeta, world[1].slosh_scale
    env.classes = world[0]
    env.reset(o, theta, seed=11 * 1000 + 3, com_end=com_end, fixture=dest if dest in cal.MESH_IDS else None, fill=hidden["fill"],
              frag=hidden["frag"])
    g = cal._stream(11, o, 3, 1)
    b0 = prims.prior(env.pose_info(), 300, g)
    state = g.bit_generator.state
    settings = (("k100_A3", 100.0, 3.0), ("k1_A10", 1.0, 10.0))

    def menus():
        return [ev.first_menu(prims, b0, dest, care, A, state) for _, care, A in settings]

    plain = menus()
    with ev.SharedMenus(prims) as shared:
        memo = menus()
        assert shared.hits["estimate"] > 0
    for p, m in zip(plain, memo):
        assert p["payload"]["recommendation"] == m["payload"]["recommendation"]
        assert (p["payload"].get("guard") or {}).get("flag") == (m["payload"].get("guard") or {}).get("flag")
        assert {k: o.expected_cost for k, o in p["menu"].items()} == {k: o.expected_cost for k, o in m["menu"].items()}
