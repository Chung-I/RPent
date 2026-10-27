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
``.model`` a SuccessModel or HazardModel, ``.rng``) and ``store`` (a dict that holds the
per-episode assessments; separate from RPent's EnvState).
"""

from __future__ import annotations

from typing import Any

from rpent.skills.pick_place.belief import Belief
from rpent.skills.pick_place.destinations import DESTINATIONS
from rpent.skills.pick_place.menu import build_menu
from rpent.skills.pick_place.protocol import Assessment, Option
from rpent.tools.base import ToolResult, tool

G = 9.81
MIN_FORCE_N = 10.0
MODES = ("commit", "assess_only")


def standard_force(nominal_mass: float) -> float:
    """The standard squeeze per pad: twice what holds the nominal mass at mu 0.8, at least 10 N."""
    return float(max(MIN_FORCE_N, round(2 * nominal_mass * G / (2 * 0.8), 1)))


def _cache(store: dict[str, Any]) -> dict[tuple, dict[str, Any]]:
    return store.setdefault("pick_place", {})


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
    key = (loc["object_id"], destination, care, int(getattr(env, "episode", 0)))
    cache = _cache(store)
    if key in cache:
        return None, cache[key]["assessment"]
    belief = Belief.prior(loc["volume"], loc["material"], loc.get("hollow", False), length=loc.get("extent"))
    geometry = env.geometry()
    proposals = env.proposals()
    f_std = standard_force(env.nominal_mass(loc["object_id"]))
    options, recommendation = build_menu(primitives.model, belief, geometry, proposals, f_std, care, destination,
                                         env.plan_trajectory, primitives.rng)
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
    the assessment without moving.

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
    prop = next(p for p in env.proposals() if abs(p.s - option.action["s"]) < 1e-9)
    act = option.action
    r = env.execute(prop, act["force"], act["v_lift"], act["v_transport"], act["destination"], act.get("release_height"))
    declared = option.to_data()
    cache = _cache(store)
    cache.clear()                                   # the scene changed: every assessment of this episode is stale
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
        "assessment": fresh.to_data() if fresh is not None else fresh_err,
    }
    return ToolResult(data=data)
