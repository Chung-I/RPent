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

"""The two planner-facing tools: ``assess_pick`` (read-only) and ``pick_and_place``.

Resources the toolkit injects: ``primitives`` (``.env`` with the lab adapter's method set,
``.model`` a SuccessModel or HazardModel, ``.rng``, and optionally ``.prior(pose, n, rng)``, the
substrate's prior over the hidden properties, ``.readout_model`` and ``.probe_models``, which turn
attempts and probes into belief updates) and ``store`` (a dict that holds the per-episode
assessments and beliefs; separate from RPent's EnvState).

The belief lives in ``store["beliefs"][object_id]`` (``{"episode", "belief", "probes_run"}``) for one
episode; every attempt appends a history entry ``{"type", "base", "pose", "force", "planned", "obs",
"used_terms", "rejected_terms"}`` and reweights it (``update_belief``), and every cached assessment is
then stale.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from rpent.skills.pick_place.belief import MIN_ESS, N_PARTICLES, REFRESH_DRAWS, Belief, ParticleBelief, refresh
from rpent.skills.pick_place.costs import prediction_matrix
from rpent.skills.pick_place.destinations import DESTINATIONS
from rpent.skills.pick_place.diagnose import diagnose
from rpent.skills.pick_place.features import particle_matrix, particle_sides
from rpent.skills.pick_place.menu import build_menu, estimate, geometry_pose
from rpent.skills.pick_place.protocol import OUTCOMES, Assessment, Option
from rpent.tools.base import ToolResult, tool

G = 9.81
MIN_FORCE_N = 10.0
MODES = ("commit", "assess_only")
REFRESH_ESS = MIN_ESS             # update_belief refreshes when the effective particles fall below this
FLOOR_P = 1e-4                    # the outcome term's floor without a readout model
OUT_OF_REACH = {"assessed": False, "reason": "object is out of reach (floor or bin)", "next": "report it; the skill cannot reach it"}


def standard_force(nominal_mass: float) -> float:
    """The standard squeeze per pad: twice what holds the nominal mass at mu 0.8, at least 10 N."""
    return float(max(MIN_FORCE_N, round(2 * nominal_mass * G / (2 * 0.8), 1)))


def _cache(store: dict[str, Any]) -> dict[tuple, dict[str, Any]]:
    return store.setdefault("pick_place", {})


def _beliefs(store: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return store.setdefault("beliefs", {})


def _prior_belief(loc: dict[str, Any], env: Any, primitives: Any, n: int = N_PARTICLES,
                  rng: np.random.Generator | None = None) -> ParticleBelief:
    """The substrate's prior (``primitives.prior``) when it has one, else the milestone 1 prior as
    particles at the nominal CoM."""
    rng = primitives.rng if rng is None else rng
    prior = getattr(primitives, "prior", None)
    if prior is not None and hasattr(env, "pose_info"):
        return prior(env.pose_info(), n, rng)
    parametric = Belief.prior(loc["volume"], loc["material"], loc.get("hollow", False), length=loc.get("extent"))
    com = env.pose_info()["com_obj_nominal"] if hasattr(env, "pose_info") else list(map(float, env.geometry().com))
    return ParticleBelief.from_parametric(parametric, com, n, rng)


def _laws(primitives: Any, label: str) -> dict[str, tuple[float, float] | None] | None:
    """The readout model's fitted laws of the slip and loss terms for a label (the diagnosis corrects its
    friction numbers with them); None without a readout model."""
    rm = getattr(primitives, "readout_model", None)
    return None if rm is None else {t: rm.law(t, label) for t in ("slip", "loss")}


def _belief_record(loc: dict[str, Any], env: Any, primitives: Any, store: dict[str, Any]) -> dict[str, Any]:
    """The object's belief record for this episode; a new episode starts from the prior."""
    episode = int(getattr(env, "episode", 0))
    rec = _beliefs(store).get(loc["object_id"])
    if rec is None or rec["episode"] != episode:
        rec = {"episode": episode, "belief": _prior_belief(loc, env, primitives), "probes_run": set()}
        _beliefs(store)[loc["object_id"]] = rec
    return rec


def entry_loglik(belief: ParticleBelief, entry: dict[str, Any], primitives: Any,
                 terms: list[str] | None = None) -> tuple[np.ndarray, dict[str, list[str]]]:
    """Per-particle log-likelihood of one history entry and the terms ``{"used", "rejected"}``. An
    attempt goes through the success model's per-particle predictions and the readout model (the
    outcome term alone without one); a probe through its probe model. ``terms`` replays exactly an
    entry's recorded terms (a refresh); None applies the rejection rule (a new reading)."""
    a = belief.arrays()
    X = particle_matrix(entry["base"], a, entry["pose"], float(entry["force"]))
    if entry["type"] == "probe":
        pm = (getattr(primitives, "probe_models", None) or {}).get(entry["obs"]["probe"])
        if pm is None:
            raise ValueError(f"no probe model for {entry['obs']['probe']!r}")
        return pm.loglik(entry["obs"], X, a, float(entry["force"]), terms=terms)
    obs = entry["obs"]
    _, per = estimate(primitives.model, belief, entry["base"], entry["pose"], float(entry["force"]), entry.get("planned") or [])
    rm = getattr(primitives, "readout_model", None)
    if rm is None:
        if terms is not None and "outcome" not in terms:
            return np.zeros(len(belief)), {"used": [], "rejected": []}
        P = prediction_matrix(per)
        return np.log(np.maximum(FLOOR_P, P[:, OUTCOMES.index(obs["label"])])), {"used": ["outcome"], "rejected": []}
    comp, info = rm.components(obs, per, X, a, particle_sides(a, entry["pose"]), terms=terms)
    return rm.combine(comp), info


def update_belief(belief: ParticleBelief, entry: dict[str, Any], primitives: Any,
                  prior_fn: Callable[[int, np.random.Generator], ParticleBelief] | None = None,
                  refresh_draws: tuple[int, ...] = REFRESH_DRAWS) -> ParticleBelief:
    """The belief after one attempt or probe: reweighted by the entry's log-likelihood, the entry (with
    the terms used and rejected) appended to the history, and refreshed from the prior when fewer than
    REFRESH_ESS effective particles remain. ``prior_fn(n, rng)`` draws the prior (default:
    ``primitives.prior`` at the entry's pose); without one the collapsed belief is kept and flagged."""
    ll, info = entry_loglik(belief, entry, primitives)
    rec = dict(entry, used_terms=list(info["used"]), rejected_terms=list(info["rejected"]))
    post = belief.reweight(ll)
    post = post._replace(history=post.history + [rec])
    if post.effective_size() >= REFRESH_ESS:
        return post
    if prior_fn is None and getattr(primitives, "prior", None) is not None:
        prior_fn = lambda n, rng: primitives.prior(entry["pose"], n, rng)  # noqa: E731
    if prior_fn is None:
        return post._replace(degenerate=True, flags=post.flags + ["fewer than 100 effective particles and no prior to refresh from"])
    replay = lambda fresh, e: entry_loglik(fresh, e, primitives, terms=e["used_terms"])[0]  # noqa: E731
    return refresh(post, prior_fn, replay, primitives.rng, n_draws=refresh_draws, n_keep=len(belief))


def _assess(object: str, destination: str, care: str, primitives: Any, store: dict[str, Any]) -> tuple[dict[str, Any] | None, Assessment | None]:
    """Build (or reuse) the assessment. Returns (error-or-semantic payload, None) or (None, assessment)."""
    if destination not in DESTINATIONS:
        return {"error": f"unknown destination {destination!r}; known: {sorted(DESTINATIONS)}"}, None
    if care not in ("low", "normal", "high"):
        return {"error": f"care must be low, normal or high, not {care!r}"}, None
    env = primitives.env
    loc = env.locate(object)
    if loc is None:
        return {"assessed": False, "reason": "object not found",
                "next": "call view_env_state and use the object's name as `object`"}, None
    if loc.get("reachable") is False:
        return dict(OUT_OF_REACH), None
    key = (loc["object_id"], destination, care, int(getattr(env, "episode", 0)))
    cache = _cache(store)
    if key in cache:
        return None, cache[key]["assessment"]
    geometry = env.geometry()
    proposals = env.proposals()
    pose_fn = getattr(env, "proposal_pose", None) or (lambda prop: geometry_pose(geometry, prop))
    belief = _belief_record(loc, env, primitives, store)["belief"]
    f_std = standard_force(env.nominal_mass(loc["object_id"]))
    options, recommendation = build_menu(primitives.model, belief, geometry, proposals, f_std, care, destination,
                                         env.plan_trajectory, primitives.rng, pose_fn=pose_fn,
                                         threshold=getattr(primitives, "threshold", None))
    evidence = belief.summary() + [f"standard squeeze {f_std:.0f} N per pad", f"{len(proposals)} pinch proposals along the long axis"]
    a = Assessment(object=object, destination=destination, options=options, evidence=evidence,
                   recommendation=recommendation, care=care, visibility=belief.visibility, episode=key[3])
    cache[key] = {"assessment": a, "menu": {o.id: o for o in options}, "belief": belief}
    return None, a


@tool(readonly=True, exclude=("primitives", "store"))
def assess_pick(*, object: str, destination: str = "bin", care: str = "normal", primitives, store) -> ToolResult:
    """Estimate the outcomes of picking an object and setting it down, without moving.

    Returns the best menu row's probabilities (success, held to release, irreversible and
    recoverable failures, drop-height distribution), the evidence, the ranked menu of options
    with their expected costs, and a recommendation. Every probability is conditional on the
    object and one menu row.

    Args:
        object: The object's name as shown by view_env_state.
        destination: Where to set it down (table_centre, table_edge, table_gap, mouse_pad, plate, box, bin).
        care: low, normal or high; scales the cost of irreversible outcomes in the ranking.
    """
    err, a = _assess(object, destination, care, primitives, store)
    if err is not None:
        return ToolResult(data=err)
    return ToolResult(data=a.to_data())


@tool(readonly=False, exclude=("primitives", "store"))
def pick_and_place(*, object: str, destination: str = "bin", mode: str = "commit", option_id: str | None = None,
                   care: str = "normal", primitives, store) -> ToolResult:
    """Pick the object and set it down at the destination, executing one menu row.

    With no option_id the best row for the requested destination runs, unless the
    recommendation is decline (then nothing runs; pass an option_id to override). Rows that
    change the destination (inboard_place) run only when named. mode "assess_only" returns
    the assessment without moving. After an attempt the result carries the pad readouts, a
    diagnosis of what went wrong, the belief before and after the update, and a fresh assessment
    under the updated belief and the object's new pose.

    Args:
        object: The object's name as shown by view_env_state.
        destination: Where to set it down.
        mode: commit or assess_only.
        option_id: A row id from the last assessment's menu, or none for the best row.
        care: low, normal or high.
    """
    if mode == "probe_first":
        return ToolResult(data={"error": "probe_first is not available in milestone 1"})
    if mode not in MODES:
        return ToolResult(data={"error": f"mode must be one of {MODES}, not {mode!r}"})
    err, a = _assess(object, destination, care, primitives, store)
    if err is not None:
        return ToolResult(data=err)
    if mode == "assess_only":
        return ToolResult(data=a.to_data())
    menu: dict[str, Option] = {o.id: o for o in a.options}
    if option_id is None:
        if a.recommendation == "decline":
            data = a.to_data()
            data.update({"placed": False, "reason": f"recommendation is decline: {a.decline_reason()}",
                         "next": "pass an option_id from the menu to execute anyway, or raise care"})
            return ToolResult(data=data)
        option = a.best()
    elif option_id not in menu:
        return ToolResult(data={"error": f"unknown option_id {option_id!r}; menu ids: {sorted(menu)}"})
    else:
        option = menu[option_id]
    if option.kind == "decline":
        return ToolResult(data={"placed": False, "reason": option.reason,
                                "next": "pick another option_id or raise care"})
    env = primitives.env
    loc = env.locate(object)
    rec = _belief_record(loc, env, primitives, store)
    before = rec["belief"]
    prop = next(p for p in env.proposals() if abs(p.s - option.action["s"]) < 1e-9)
    act = option.action
    r = env.execute(prop, act["force"], act["v_lift"], act["v_transport"], act["destination"], act.get("release_height"))
    declared = option.to_data()
    readouts = dict(getattr(r, "readouts", None) or {})
    entry = {"type": "attempt", "option": option.id, "base": option._base, "pose": option._pose, "force": float(act["force"]),
             "planned": option._planned, "obs": {"label": r.label, "readouts": readouts, "force": float(act["force"])}}
    try:
        after = update_belief(before, entry, primitives, prior_fn=lambda n, rng: _prior_belief(loc, env, primitives, n, rng))
    except Exception as exc:  # noqa: BLE001  the executed outcome must survive a failed update
        after = before._replace(flags=before.flags + [f"belief update failed: {exc}"])
    rec["belief"] = after
    cache = _cache(store)
    cache.clear()                                   # the scene and the belief changed: every assessment is stale
    try:
        fresh_err, fresh = _assess(object, destination, care, primitives, store)
    except Exception as exc:  # noqa: BLE001  the executed outcome must survive a failed reassessment
        fresh_err, fresh = {"error": f"reassessment failed: {exc}"}, None
    data = {
        "placed": r.label == "placed_stable",
        "placed_at_requested": r.label == "placed_stable" and act["destination"] == destination,
        "outcome": r.label,
        "drop_z": r.drop_z,
        "loss_phase": r.loss_phase,
        "zone": r.zone,
        "release": None if r.release is None else {k: r.release.get(k) for k in ("tilt_deg", "displacement_xy_m", "drop_height_m", "final_zone")},
        "option": option.id,
        "destination_used": act["destination"],
        "declared": {"p_success": declared["p"]["placed_stable"],
                     "p_irreversible": {k: declared["p"][k] for k in ("dropped_high", "rolled_off")},
                     "p_recoverable": {k: declared["p"][k] for k in ("tipped", "missed", "dropped_low", "not_lifted")},
                     "p_drop_height": declared["p_drop_height"], "expected_cost": declared["expected_cost"]},
        "readouts": readouts,
        "diagnosis": diagnose(r.label, readouts, act, before.to_data(), after.to_data(), laws=_laws(primitives, r.label)),
        "belief_before": before.to_data(),
        "belief": after.to_data(),
        "assessment": fresh.to_data() if fresh is not None else fresh_err,
    }
    return ToolResult(data=data)
