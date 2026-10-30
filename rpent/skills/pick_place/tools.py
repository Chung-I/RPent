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
attempts and probes into belief updates, and ``.guard``, the novelty guard (guard.KnnGuard)) and ``store`` (a dict that holds the per-episode
assessments and beliefs; separate from RPent's EnvState).

The belief lives in ``store["beliefs"][object_id]`` (``{"episode", "belief", "probes_run", "probe_seq", "attempted"}``)
for one episode; every attempt appends a history entry ``{"type", "base", "pose", "force", "planned", "obs",
"used_terms", "rejected_terms"}`` and reweights it (``update_belief``), and every cached assessment is
then stale. A probe or an attempt that damages the object ends the case: the record keeps the damage label
(``"damaged"``) and the object gets no menu for the rest of the episode.

The recovery mode (``primitives.reset_s``, plan H3 decision 3): an irreversible attempt, or a probe that throws or
damages the object, is followed in the same call by ``env.restore()`` (the case's start state; the episode and the
sensor streams go on), the damage latch is cleared, the record keeps ``restored`` (no probe after it), and the call
returns the restored object's fresh assessment with the payload's ``restored`` record. The belief keeps what the event
taught.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from rpent.skills.pick_place.belief import MIN_ESS, N_PARTICLES, REFRESH_DRAWS, Belief, ParticleBelief, refresh
from rpent.skills.pick_place.costs import care_scale, prediction_matrix, probe_time_scale, reset_time
from rpent.skills.pick_place.destinations import DESTINATIONS
from rpent.skills.pick_place.diagnose import diagnose
from rpent.skills.pick_place.features import A_TRANSPORT, model_names, particle_matrix, particle_sides
from rpent.skills.pick_place.menu import build_menu, estimate, geometry_pose, pinch_limit, without_probe
from rpent.skills.pick_place.probes import DAMAGED
from rpent.skills.pick_place.protocol import DAMAGE, IRREVERSIBLE, OUTCOMES, RECOVERABLE, Assessment, Option
from rpent.tools.base import ToolResult, tool

G = 9.81
MIN_FORCE_N = 10.0
MODES = ("commit", "assess_only", "probe_first")
PROBE_OPTIONS = {"probe_push": "push", "probe_weigh": "weigh", "probe_squeeze": "squeeze"}
MAX_PROBES = 2                    # probe_first runs at most this many probes per call
PROBE_FOLLOWS = {"push": ("weigh",)}   # the probes that may follow a probe: the probe models cover probes from rest
                                       # and the weigh after the push (the probe corpus's scenes); primitives.probe_follows
                                       # overrides it
AFTER_ATTEMPT = "no probe after an attempt: the probe models cover probes from rest"
AFTER_RESTORE = "no probe after a restore: the probe models cover probes from rest of an untouched object"
REFRESH_ESS = MIN_ESS             # update_belief refreshes when the effective particles fall below this
FLOOR_P = 1e-4                    # the outcome term's floor without a readout model
OUT_OF_REACH = {"assessed": False, "reason": "object is out of reach (floor or bin)", "next": "report it; the skill cannot reach it"}
DAMAGED_NEXT = "report it; the skill does not move a damaged object again"
HIDDEN_READING = ("eps",)         # the substrate's record of a reading's noise: for its own checks, never the planner's


def standard_force(nominal_mass: float) -> float:
    """The standard squeeze per pad: twice what holds the nominal mass at mu 0.8, at least 10 N."""
    return float(max(MIN_FORCE_N, round(2 * nominal_mass * G / (2 * 0.8), 1)))


def _cache(store: dict[str, Any]) -> dict[tuple, dict[str, Any]]:
    return store.setdefault("pick_place", {})


def _beliefs(store: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return store.setdefault("beliefs", {})


def _spread_kw(primitives: Any) -> dict[str, float]:
    """The prior's spread argument (plan H3 decision 4): none at f = 1 or without ``primitives.spread``, so a prior
    without the argument keeps working."""
    f = getattr(primitives, "spread", None)
    return {} if f is None or float(f) == 1.0 else {"spread": float(f)}


def _prior_belief(loc: dict[str, Any], env: Any, primitives: Any, n: int = N_PARTICLES,
                  rng: np.random.Generator | None = None) -> ParticleBelief:
    """The substrate's prior (``primitives.prior``) when it has one, else the milestone 1 prior as
    particles at the nominal CoM."""
    rng = primitives.rng if rng is None else rng
    prior = getattr(primitives, "prior", None)
    if prior is not None and hasattr(env, "pose_info"):
        return prior(env.pose_info(), n, rng, **_spread_kw(primitives))
    parametric = Belief.prior(loc["volume"], loc["material"], loc.get("hollow", False), length=loc.get("extent"))
    com = env.pose_info()["com_obj_nominal"] if hasattr(env, "pose_info") else list(map(float, env.geometry().com))
    return ParticleBelief.from_parametric(parametric, com, n, rng)


def _laws(primitives: Any, label: str) -> dict[str, Any] | None:
    """The readout model's fitted laws of the slip and loss terms for a label (the diagnosis corrects its
    friction numbers with them); None without a readout model."""
    rm = getattr(primitives, "readout_model", None)
    return None if rm is None else {t: rm.law(t, label) for t in ("slip", "loss")}


def _belief_record(loc: dict[str, Any], env: Any, primitives: Any, store: dict[str, Any]) -> dict[str, Any]:
    """The object's belief record for this episode; a new episode starts from the prior."""
    episode = int(getattr(env, "episode", 0))
    rec = _beliefs(store).get(loc["object_id"])
    if rec is None or rec["episode"] != episode:
        rec = {"episode": episode, "belief": _prior_belief(loc, env, primitives), "probes_run": set(), "probe_seq": [],
               "attempted": False}
        _beliefs(store)[loc["object_id"]] = rec
    return rec


def _probes_allowed(rec: dict[str, Any], primitives: Any) -> set[str] | None:
    """The probes the menu may offer now (plan m4 decision 1): every probe before the first one, after one probe those
    that may follow it (PROBE_FOLLOWS), none after two probes, after an attempt or after a restore (plan H3). None: no
    restriction yet."""
    if rec.get("attempted") or rec.get("restored"):
        return set()
    seq = rec.get("probe_seq") or []
    if not seq:
        return None
    follows = getattr(primitives, "probe_follows", None) or PROBE_FOLLOWS
    return set(follows.get(seq[-1], ())) if len(seq) == 1 else set()


def _probe_refusal(probe: str, rec: dict[str, Any], primitives: Any) -> str | None:
    """Why a named probe is not offered now, or None when the sequence allows it."""
    allowed = _probes_allowed(rec, primitives)
    if allowed is None or probe in allowed:
        return None
    if rec.get("restored"):
        return f"probe_{probe} is not offered: {AFTER_RESTORE}"
    if rec.get("attempted"):
        return f"probe_{probe} is not offered: probes run only before the first attempt of an episode ({AFTER_ATTEMPT})"
    return (f"probe_{probe} cannot follow the {rec['probe_seq'][-1]}: the probe models cover probes from rest and the weigh "
            "after the push")


def _damage_of(loc: dict[str, Any], env: Any, store: dict[str, Any]) -> str | None:
    """The damage label of this episode's damaging probe or attempt, if one ran (the case is then over)."""
    rec = _beliefs(store).get(loc["object_id"])
    return rec.get("damaged") if rec is not None and rec["episode"] == int(getattr(env, "episode", 0)) else None


def entry_loglik(belief: ParticleBelief, entry: dict[str, Any], primitives: Any,
                 terms: list[str] | None = None) -> tuple[np.ndarray, dict[str, list[str]]]:
    """Per-particle log-likelihood of one history entry and the terms ``{"used", "rejected"}``. An
    attempt goes through the success model's per-particle predictions and the readout model (the
    outcome term alone without one); a probe through its probe model. ``terms`` replays exactly an
    entry's recorded terms (a refresh); None applies the rejection rule (a new reading)."""
    a = belief.arrays()
    if entry["type"] == "probe":
        pm = (getattr(primitives, "probe_models", None) or {}).get(entry["obs"]["probe"])
        if pm is None:
            raise ValueError(f"no probe model for {entry['obs']['probe']!r}")
        X = particle_matrix(entry["base"], a, entry["pose"], float(entry["force"]), model_names(pm))
        return pm.loglik(entry["obs"], X, a, float(entry["force"]), terms=terms, thrown=entry["obs"].get("thrown"))
    obs = entry["obs"]
    _, per = estimate(primitives.model, belief, entry["base"], entry["pose"], float(entry["force"]), entry.get("planned") or [])
    rm = getattr(primitives, "readout_model", None)
    if rm is None:
        if terms is not None and "outcome" not in terms:
            return np.zeros(len(belief)), {"used": [], "rejected": []}
        P = prediction_matrix(per)
        return np.log(np.maximum(FLOOR_P, P[:, OUTCOMES.index(obs["label"])])), {"used": ["outcome"], "rejected": []}
    X = particle_matrix(entry["base"], a, entry["pose"], float(entry["force"]), model_names(getattr(rm, "rot_clf", None)))
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
        prior_fn = lambda n, rng: primitives.prior(entry["pose"], n, rng, **_spread_kw(primitives))  # noqa: E731
    if prior_fn is None:
        return post._replace(degenerate=True, flags=post.flags + ["fewer than 100 effective particles and no prior to refresh from"])
    replay = lambda fresh, e: entry_loglik(fresh, e, primitives, terms=e["used_terms"])[0]  # noqa: E731
    return refresh(post, prior_fn, replay, primitives.rng, n_draws=refresh_draws, n_keep=len(belief))


def _assess(object: str, destination: str, care: str | float, primitives: Any,
            store: dict[str, Any]) -> tuple[dict[str, Any] | None, Assessment | None]:
    """Build (or reuse) the assessment. Returns (error-or-semantic payload, None) or (None, assessment).
    ``primitives.abstain_cost`` (when set) selects the cost rule, ``primitives.evoi`` (when set) the probe-value
    estimator, ``primitives.probe_time_scale`` (when set) the factor on the probe rows' time charge and
    ``primitives.reset_s`` (when set, plan H3) the recovery mode's reset seconds (menu.build_menu); a bad scale or reset
    time is an error payload."""
    if destination not in DESTINATIONS:
        return {"error": f"unknown destination {destination!r}; known: {sorted(DESTINATIONS)}"}, None
    try:
        scale = care_scale(care)
        time_scale = probe_time_scale(getattr(primitives, "probe_time_scale", 1.0))
        reset_s = getattr(primitives, "reset_s", None)
        reset_s = None if reset_s is None else reset_time(reset_s)
    except ValueError as exc:
        return {"error": str(exc)}, None
    abstain = getattr(primitives, "abstain_cost", None)
    env = primitives.env
    loc = env.locate(object)
    if loc is None:
        return {"assessed": False, "reason": "object not found",
                "next": "call view_env_state and use the object's name as `object`"}, None
    if loc.get("reachable") is False:
        return dict(OUT_OF_REACH), None
    damaged = _damage_of(loc, env, store)
    if damaged:
        return {"assessed": False, "reason": f"object is damaged ({damaged})", "next": DAMAGED_NEXT}, None
    key = (loc["object_id"], destination, care, int(getattr(env, "episode", 0)))
    cache = _cache(store)
    if key in cache:
        return None, cache[key]["assessment"]
    geometry = env.geometry()
    proposals = env.proposals()
    pose_fn = getattr(env, "proposal_pose", None) or (lambda prop: geometry_pose(geometry, prop))
    belief = _belief_record(loc, env, primitives, store)["belief"]
    rec = _belief_record(loc, env, primitives, store)
    f_std = standard_force(env.nominal_mass(loc["object_id"]))
    allowed = _probes_allowed(rec, primitives)
    probes = {p: m for p, m in (getattr(primitives, "probe_models", None) or {}).items()
              if p not in rec["probes_run"] and (allowed is None or p in allowed)}
    options, recommendation = build_menu(primitives.model, belief, geometry, proposals, f_std, care, destination,
                                         env.plan_trajectory, primitives.rng, pose_fn=pose_fn, max_open=pinch_limit(env),
                                         threshold=getattr(primitives, "threshold", None), probe_models=probes or None,
                                         guard=getattr(primitives, "guard", None), abstain_cost=abstain,
                                         evoi=getattr(primitives, "evoi", "plugin"), probe_time_scale=time_scale, reset_s=reset_s)
    evidence = belief.summary() + [f"standard squeeze {f_std:.0f} N per pad", f"{len(proposals)} pinch proposals along the long axis"]
    if rec["probes_run"]:
        evidence.append(f"probes run this episode: {', '.join(sorted(rec['probes_run']))}")
    if rec.get("attempted") and getattr(primitives, "probe_models", None):
        evidence.append(AFTER_ATTEMPT)
    if rec.get("restored"):
        evidence.append(f"restored after {rec['restored']['event']} ({rec['restored']['after']}); {AFTER_RESTORE}")
    a = Assessment(object=object, destination=destination, options=options, evidence=evidence,
                   recommendation=recommendation, care=care, visibility=belief.visibility, episode=key[3], care_scale=scale,
                   abstain_cost=None if abstain is None else float(abstain), probe_time_scale=time_scale, reset_s=reset_s,
                   without_probe=without_probe(options, recommendation, destination, care, abstain, getattr(primitives, "threshold", None),
                                               reset_s=reset_s))
    cache[key] = {"assessment": a, "menu": {o.id: o for o in options}, "belief": belief}
    return None, a


@tool(readonly=True, exclude=("primitives", "store"))
def assess_pick(*, object: str, destination: str = "bin", care: str | float = "normal", primitives, store) -> ToolResult:
    """Estimate the outcomes of picking an object and setting it down, without moving.

    Returns the best menu row's probabilities (success, held to release, irreversible and
    recoverable failures, drop-height distribution), the evidence, the ranked menu of options
    with their expected costs, and a recommendation. Every probability is conditional on the
    object and one menu row. The recommendation is decline when the novelty guard finds the
    object unlike the training objects (``guard``).

    Args:
        object: The object's name as shown by view_env_state.
        destination: Where to set it down (table_centre, table_edge, table_gap, mouse_pad, plate, box, bin).
        care: low, normal, high, or a number above 0 (0.5, 1, 3 for the names); scales the cost of irreversible
            outcomes in the ranking and in the recommendation.
    """
    err, a = _assess(object, destination, care, primitives, store)
    if err is not None:
        return ToolResult(data=err)
    return ToolResult(data=a.to_data())


def _threshold(primitives: Any) -> float:
    """The commit threshold: ``primitives.threshold`` when set, else menu.DEFAULT_THRESHOLD, read now."""
    from rpent.skills.pick_place import menu as menu_mod

    t = getattr(primitives, "threshold", None)
    return float(menu_mod.DEFAULT_THRESHOLD if t is None else t)


def _reassess(object: str, destination: str, care: str, primitives: Any, store: dict[str, Any]) -> dict[str, Any]:
    """The fresh assessment's payload after the scene and the belief changed (every cached one is stale)."""
    _cache(store).clear()
    try:
        err, fresh = _assess(object, destination, care, primitives, store)
    except Exception as exc:  # noqa: BLE001  the executed action's result must survive a failed reassessment
        return {"error": f"reassessment failed: {exc}"}
    return fresh.to_data() if fresh is not None else err


def _update(rec: dict[str, Any], entry: dict[str, Any], loc: dict[str, Any], primitives: Any) -> tuple[ParticleBelief, ParticleBelief]:
    """(before, after): the record's belief updated by one entry; a failed update keeps the belief, flagged."""
    env, before = primitives.env, rec["belief"]
    try:
        after = update_belief(before, entry, primitives, prior_fn=lambda n, rng: _prior_belief(loc, env, primitives, n, rng))
    except Exception as exc:  # noqa: BLE001  the executed action's result must survive a failed update
        after = before._replace(flags=before.flags + [f"belief update failed: {exc}"])
    rec["belief"] = after
    return before, after


def _restore(rec: dict[str, Any], env: Any, after: str, event: str) -> dict[str, Any]:
    """The recovery mode (plan H3 decision 3): the lab rebuilds the case's start state (env.restore()), the damage
    latch is cleared and the record keeps ``restored``; the belief stays. The caller reassesses next (_reassess clears
    the cached assessments). Returns the payload's record: after, event and the lab's check (plan H3 gate Gr)."""
    check = env.restore()
    rec.pop("damaged", None)
    rec["restored"] = {"after": after, "event": event}
    return {"after": after, "event": event, "check": check}


def _run_probe(row: Option, object: str, destination: str, care: str, primitives: Any, store: dict[str, Any]) -> dict[str, Any]:
    """Run one probe row, update the belief with its result, and return the probe payload. A probe that damages
    the object ends the case (its fresh assessment says so); the reading's noise (HIDDEN_READING) stays out."""
    env = primitives.env
    loc = env.locate(object)
    rec = _belief_record(loc, env, primitives, store)
    probe, act = row.action["probe"], row.action
    if probe in ("weigh", "squeeze"):
        prop = next(p for p in env.proposals() if abs(p.s - act["s"]) < 1e-9)
        res = env.probe_weigh(prop, act["force"]) if probe == "weigh" else env.probe_squeeze(prop, act["force"])
    else:
        res = env.probe_push()
    rec["probes_run"].add(probe)
    rec["probe_seq"].append(probe)
    damage = getattr(res, "damage", None)
    if res.kind == DAMAGED or damage:
        rec["damaged"] = (damage or {}).get("label") or DAMAGED
    reading = {k: v for k, v in dict(res.reading).items() if k not in HIDDEN_READING}
    entry = {"type": "probe", "option": row.id, "base": row._base, "pose": row._pose, "force": float(act["force"]), "planned": None,
             "obs": {"probe": probe, "kind": res.kind, "reading": dict(reading), "thrown": bool(res.thrown)}}
    before, after = _update(rec, entry, loc, primitives)
    out = {"probed": probe, "placed": False, "kind": res.kind, "reading": reading,
           "displacement_m": res.displacement_m, "tilt_deg": res.tilt_deg, "thrown": bool(res.thrown), "seconds": res.seconds,
           "artifact": bool(getattr(res, "artifact", False)), "damage": damage,
           "declared": row.to_data()["probe"], "belief_before": before.to_data(), "belief": after.to_data()}
    if getattr(primitives, "reset_s", None) is not None and (res.thrown or rec.get("damaged")):
        out["restored"] = _restore(rec, env, "probe", rec.get("damaged") or "thrown")
    out["assessment"] = _reassess(object, destination, care, primitives, store)
    return out


def _meter_max(meter: dict[str, Any] | None) -> dict[str, float]:
    """Each damage-meter channel's maximum (its records are [t, value] pairs; 0 for an empty channel); {} without a
    meter."""
    return {k: max((float(v) for _, v in rows or []), default=0.0) for k, rows in (meter or {}).items()}


def _attempt(option: Option, object: str, destination: str, care: str, primitives: Any, store: dict[str, Any]) -> dict[str, Any]:
    """Execute one commit or lower-risk row, update the belief with the outcome and readouts, and
    return the attempt payload. An outcome in DAMAGE ends the case (its fresh assessment says so)."""
    env = primitives.env
    loc = env.locate(object)
    rec = _belief_record(loc, env, primitives, store)
    prop = next(p for p in env.proposals() if abs(p.s - option.action["s"]) < 1e-9)
    act = option.action
    r = env.execute(prop, act["force"], act["v_lift"], act["v_transport"], act["destination"], act.get("release_height"),
                    a_transport=act.get("a_transport", A_TRANSPORT))
    rec["attempted"] = True
    if r.label in DAMAGE:
        rec["damaged"] = r.label
    declared = option.to_data()
    readouts = dict(getattr(r, "readouts", None) or {})
    entry = {"type": "attempt", "option": option.id, "base": option._base, "pose": option._pose, "force": float(act["force"]),
             "planned": option._planned, "obs": {"label": r.label, "readouts": readouts, "force": float(act["force"])}}
    before, after = _update(rec, entry, loc, primitives)
    out = {
        "placed": r.label == "placed_stable",
        "placed_at_requested": r.label == "placed_stable" and act["destination"] == destination,
        "outcome": r.label,
        "drop_z": r.drop_z,
        "loss_phase": r.loss_phase,
        "zone": r.zone,
        "release": None if r.release is None else {k: r.release.get(k) for k in ("tilt_deg", "displacement_xy_m", "drop_height_m", "final_zone")},
        "option": option.id,
        "destination_used": act["destination"],
        "artifact": bool(getattr(r, "artifact", False)),       # the substrate says the run's contacts were not physical
        "damage": getattr(r, "damage", None),                  # the damage meter's first threshold crossing, if any
        "meter_max": _meter_max(getattr(r, "meter", None)),
        "declared": {"p_success": declared["p"]["placed_stable"],
                     "p_irreversible": {k: declared["p"][k] for k in IRREVERSIBLE},
                     "p_recoverable": {k: declared["p"][k] for k in RECOVERABLE},
                     "p_drop_height": declared["p_drop_height"], "expected_cost": declared["expected_cost"]},
        "readouts": readouts,
        "diagnosis": diagnose(r.label, readouts, act, before.to_data(), after.to_data(), laws=_laws(primitives, r.label),
                              mu_range=(float(np.min(before.arrays()["mu"])), float(np.max(before.arrays()["mu"])))),
        "belief_before": before.to_data(),
        "belief": after.to_data(),
    }
    if getattr(primitives, "reset_s", None) is not None and r.label in IRREVERSIBLE:
        out["restored"] = _restore(rec, env, "attempt", r.label)
    out["assessment"] = _reassess(object, destination, care, primitives, store)
    return out


def _commit(a: Assessment, option_id: str | None, object: str, destination: str, care: str, primitives: Any,
            store: dict[str, Any]) -> dict[str, Any]:
    """The commit path: the named row, or the best row unless the recommendation advises against it
    (decline; probe while the best row is below the threshold, or any probe under the cost rule)."""
    menu: dict[str, Option] = {o.id: o for o in a.options}
    if option_id is None:
        best = a.best()
        if a.recommendation == "decline" or best is None:
            data = a.to_data()
            flagged = bool((a.guard() or {}).get("flag"))
            data.update({"placed": False, "reason": f"recommendation is decline: {a.decline_reason()}",
                         "next": "pass an option_id from the menu to execute anyway" + ("" if flagged else (
                             ", or lower care" if getattr(primitives, "abstain_cost", None) is not None else ", or raise care"))})
            return data
        if a.recommendation == "probe" and (getattr(primitives, "abstain_cost", None) is not None
                                            or best.pred.p["placed_stable"] < _threshold(primitives)):
            probe = a.best_overall()
            data = a.to_data()
            why = ("a probe first costs less" if getattr(primitives, "abstain_cost", None) is not None
                   else f"the best row ({best.id}) is below the threshold")
            data.update({"placed": False, "reason": f"recommendation is probe ({probe.id}): {why}",
                         "next": f"pass option_id {probe.id!r} or mode probe_first, or an option_id from the menu to execute anyway"})
            return data
        option = best
    elif option_id not in menu:
        return {"error": f"unknown option_id {option_id!r}; menu ids: {sorted(menu)}"}
    else:
        option = menu[option_id]
    if option.kind == "decline":
        return {"placed": False, "reason": option.reason, "next": "pick another option_id or raise care"}
    return _attempt(option, object, destination, care, primitives, store)


def _probe_brief(d: dict[str, Any]) -> dict[str, Any]:
    return {k: d[k] for k in ("probed", "kind", "reading", "displacement_m", "thrown", "artifact")}


@tool(readonly=False, exclude=("primitives", "store"))
def pick_and_place(*, object: str, destination: str = "bin", mode: str = "commit", option_id: str | None = None,
                   care: str | float = "normal", primitives, store) -> ToolResult:
    """Pick the object and set it down at the destination, executing one menu row, or run a probe.

    With no option_id the best row for the requested destination runs, unless the
    recommendation is decline or probe (then nothing runs; pass an option_id to override). Rows that change the destination
    (inboard_place) run only when named. option_id "probe_push", "probe_weigh" or "probe_squeeze" runs that probe
    (once per episode, before the first attempt; after the push only the weigh): the object is pushed, lifted 2 cm and set back, or squeezed in place; the belief is updated,
    and nothing is placed. A probe or an attempt that damages the object ends the case: every later call this
    episode returns "object is damaged". mode "probe_first" runs the recommended probes (at most two, while the
    recommendation is probe), then commits as above; mode "assess_only" returns the assessment
    without moving. After an attempt the result carries the pad readouts, a diagnosis of what
    went wrong, the belief before and after the update, and a fresh assessment under the updated
    belief and the object's new pose.

    Args:
        object: The object's name as shown by view_env_state.
        destination: Where to set it down.
        mode: commit, probe_first or assess_only.
        option_id: A row id from the last assessment's menu (a probe id runs that probe), or none for the best row.
        care: low, normal, high, or a number above 0.
    """
    if mode not in MODES:
        return ToolResult(data={"error": f"mode must be one of {MODES}, not {mode!r}"})
    probe_models = getattr(primitives, "probe_models", None) or {}
    if option_id in PROBE_OPTIONS and PROBE_OPTIONS[option_id] not in probe_models:
        return ToolResult(data={"error": "probe models are not loaded"})
    err, a = _assess(object, destination, care, primitives, store)
    if err is not None:
        return ToolResult(data=err)
    if mode == "assess_only":
        return ToolResult(data=a.to_data())
    if option_id in PROBE_OPTIONS:
        loc = primitives.env.locate(object)
        rec = _belief_record(loc, primitives.env, primitives, store)
        if PROBE_OPTIONS[option_id] in rec["probes_run"]:
            return ToolResult(data={"error": f"{option_id} already ran this episode; the menu shows what is left"})
        refusal = _probe_refusal(PROBE_OPTIONS[option_id], rec, primitives)
        if refusal is not None:
            return ToolResult(data={"error": refusal})
        row = next((o for o in a.options if o.id == option_id), None)
        if row is None:
            return ToolResult(data={"error": f"{option_id} is not on the menu: no row to the requested destination to probe for"})
        return ToolResult(data=_run_probe(row, object, destination, care, primitives, store))
    probes: list[dict[str, Any]] = []
    if mode == "probe_first" and probe_models:
        for _ in range(MAX_PROBES):
            if a.recommendation != "probe":
                break
            d = _run_probe(a.best_overall(), object, destination, care, primitives, store)
            probes.append(_probe_brief(d))
            if d["thrown"] or not d["assessment"].get("assessed", False):
                return ToolResult(data=dict(d, probes=probes))
            err, a = _assess(object, destination, care, primitives, store)
            if err is not None:
                return ToolResult(data=dict(err, probes=probes))
    data = _commit(a, option_id, object, destination, care, primitives, store)
    if mode == "probe_first":
        data["probes"] = probes
    return ToolResult(data=data)
