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

"""The E1 pilot (2026-11-02, throwaway; the registered E1 comes later): does a language-model planner that reads the
user's instruction and the outcome menu beat the fixed decision rule?

Each case (an H3 probe case: the default menu's first recommendation is a probe) runs under one hidden setting
(daily-logs ``experiments/2026-11-02-e1-pilot/templates.json``: care factor kappa 30 or 1, absorbing or rec20 at the time
price, A 10). The lab restores the object after an irreversible event under rec20 in every arm; the realized cost uses the
hidden setting (calibrate.attempt_cost and probe_cost at kappa, the reset, A for a case that ends unplaced). Arms, paired
(the same case draw, rng streams and instruction):

* ``default_rule``: the H3 policy (calibrate.run_protocol "policy") on the menu at the default (kappa 10, A 10, absorbing);
* ``oracle_rule``: the same policy on the menu at the true kappa and recovery;
* ``menu_planner``: the planner reads the instruction and the default menu (rows shuffled, inboard_place left out) and
  calls ``choose`` once per turn: a row, a probe or decline; after each action it reads the result and the fresh menu;
* ``rec_planner``: the same with only the recommended row (or, on decline, the best row the rule declines) and decline;
* ``nomenu_planner``: the same rows as actions, with no predictions; results show the raw readings.

A planner case ends when the object is placed, after an irreversible event without a restore, after three attempts, on
decline, on a skill error, or after ``MAX_CALLS`` model calls (then it ends unplaced).

    python -P -m robots.mujoco_lab.e1 plan --templates T.json --final-cases seed12.json --tune-cases seed13.json --out plan.json
    python -P -m robots.mujoco_lab.e1 run --plan plan.json --split tune --workers 8 --base-url http://127.0.0.1:18691/v1 --out DIR
    python -P -m robots.mujoco_lab.e1 summary --dir DIR
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import re
import sys
import time
import zlib
from typing import Any, Callable

import numpy as np

from rpent.skills.pick_place.probes import DAMAGED
from rpent.skills.pick_place.protocol import IRREVERSIBLE, OUTCOMES, RECOVERABLE

ARMS = ("default_rule", "oracle_rule", "menu_planner", "rec_planner", "nomenu_planner")
RULES = ("default_rule", "oracle_rule")
PLANNERS = ("menu_planner", "rec_planner", "nomenu_planner")
RECOVERY = {"absorbing": (None, "time"), "rec20": (20.0, "time")}
DEFAULT = {"care_scale": 10.0, "abstain_cost": 10.0, "recovery": "absorbing"}
PAIRS = (("hi_abs", "lo_rec"), ("lo_abs", "hi_rec"))      # a final case runs under one pair: opposite care and recovery
MAX_CALLS = 6
MAX_ATTEMPTS = 3                  # calibrate.POLICY_ATTEMPTS: the rule's attempts per case
HIDDEN_ROWS = ("inboard_place",)  # it sets the object down elsewhere; the rule never runs it
MODEL = "gemma4-31b-fp8"
DEST_NAMES = {"table_centre": "table centre", "table_edge": "table edge", "table_gap": "spot just past the table edge"}
PROBE_TEXT = {
    "weigh": "squeeze the object, lift it 2 cm, hold it and set it back (about 4 s); it measures the mass, or a lower bound on "
             "the mass and the friction when the object does not rise",
    "push": "push the object sideways at a quarter of its height, up to 4 cm (about 2 s); it measures the force at which it slides",
    "squeeze": "close the pads in place to a capped force, with no lift (about 3 s); it measures the stiffness",
}
READOUT_KEYS = ("rose", "max_rise_m", "fz_hold_n", "fz_slip_max_n", "fz_at_loss_n", "rot_deg", "loss_phase", "loss_since_lift_s")


# ---------------------------------------------------------------- names and the case plan

def display_object(object_id: str) -> str:
    """The object's plain name: the dataset prefix and a trailing number dropped ("hope__orange_juice_carton" -> "orange juice
    carton", "handal__hammer_6" -> "hammer")."""
    name = object_id.split("__", 1)[-1]
    return re.sub(r"_?\d+$", "", name).replace("_", " ")


def display_destination(destination: str) -> str:
    return DEST_NAMES.get(destination, destination.replace("_", " "))


def instruction(job: dict[str, Any], destination: str) -> str:
    """The job's template with its slots filled."""
    return job["template"].format(object=display_object(job["object"]), destination=display_destination(destination))


def spread_cases(cases: dict[str, list[int]], n: int, seed: int) -> list[tuple[str, int]]:
    """n cases over as many objects as possible: rounds over the objects in a seeded order, one case per object per round
    (each object's cases in a seeded order)."""
    rng = np.random.default_rng(seed)
    queues = {o: [int(i) for i in rng.permutation(sorted(v))] for o, v in sorted(cases.items())}
    order = [str(o) for o in rng.permutation(sorted(queues))]
    out: list[tuple[str, int]] = []
    while len(out) < n and any(queues.values()):
        for o in order:
            if queues[o] and len(out) < n:
                out.append((o, queues[o].pop(0)))
    return out


def plan_jobs(cases: dict[str, list[int]], templates: dict[str, Any], n_cases: int, split: str, seed: int,
              plan_seed: int = 0) -> list[dict[str, Any]]:
    """The jobs of one split. ``final``: n_cases cases, each under one PAIRS pair (alternating in the spread order), with a
    held-out family template per (case, setting); ``tune``: each case under one setting (in turn) with a tuning-family
    template. ``seed`` is the cases' H3 seed."""
    picked = spread_cases(cases, n_cases, plan_seed)
    rng = np.random.default_rng([plan_seed, 1])
    families = templates["families"]
    held = [f for f, v in families.items() if v["split"] == "heldout"]
    tune = [f for f, v in families.items() if v["split"] == "tune"]
    settings = list(templates["settings"])
    jobs = []
    for k, (o, idx) in enumerate(picked):
        if split == "final":
            pairs = [(s, held[0]) for s in PAIRS[k % len(PAIRS)]]
        else:
            pairs = [(settings[k % len(settings)], tune[(k // len(settings)) % len(tune)])]
        for setting, family in pairs:
            options = families[family]["templates"][setting]
            t = int(rng.integers(len(options)))
            jobs.append({"split": split, "object": o, "idx": int(idx), "seed": int(seed), "setting": setting, "family": family,
                         "template_idx": t, "template": options[t]})
    return jobs


# ---------------------------------------------------------------- scoring

def setting_recovery(setting: dict[str, Any]) -> tuple[float | None, str]:
    return RECOVERY[setting["recovery"]]


def realized_cost(case: dict[str, Any], setting: dict[str, Any]) -> float:
    """The case's cost under the hidden setting: its probes and attempts at kappa and the setting's recovery (calibrate's
    cost rule, probe time scale 1) and A when it ends without the object placed."""
    from robots.mujoco_lab.calibrate import attempt_cost, probe_cost

    kappa, A = float(setting["care_scale"]), float(setting["abstain_cost"])
    reset_s, price = setting_recovery(setting)
    c = sum(probe_cost(p["probe"], p["kind"], bool(p["thrown"]), kappa, 1.0, reset_s, price) for p in case.get("probes") or [])
    c += sum(attempt_cost(t["outcome"], float(t["cost_s"]), kappa, reset_s, price) for t in case.get("attempts") or [])
    return float(c + (0.0 if case.get("ended") == "placed" else A))


def irreversible_events(case: dict[str, Any]) -> int:
    return (sum(t["outcome"] in IRREVERSIBLE for t in case.get("attempts") or [])
            + sum(bool(p.get("thrown")) or p.get("kind") == DAMAGED for p in case.get("probes") or []))


def object_bootstrap(values, objects, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """(mean, 2.5 %, 97.5 %) of the case mean, resampling objects with replacement (each draw's mean over its cases)."""
    v = np.asarray(values, dtype=float)
    names = sorted(set(objects))
    inv = np.array([names.index(o) for o in objects])
    sums = np.bincount(inv, weights=v, minlength=len(names))
    counts = np.bincount(inv, minlength=len(names)).astype(float)
    draws = np.random.default_rng(seed).integers(0, len(names), size=(n_boot, len(names)))
    boot = sums[draws].sum(1) / np.maximum(counts[draws].sum(1), 1.0)
    return float(v.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


# ---------------------------------------------------------------- what the planner reads

SYSTEM = """You are the planner for a robot arm with a parallel-jaw gripper in a table-top lab. One object must be picked up and
set down at a requested place. A skill runs every motion and sets the grip force. On each turn you choose ONE action by
calling the tool `choose` with an `option_id` from the list you were given and a short `reason`:
- an attempt row (kinds commit and lower_risk) picks the object up and sets it down;
- a probe row runs a short test that measures the object; nothing is placed. Probes run only before the first attempt,
  each probe runs once, and after the push only the weigh may follow;
- `decline` abstains: the task ends without the object placed, and you report why.
After each action you read what happened and, while the task goes on, the actions you may choose next. Actions from an
older list are stale. You have at most {max_calls} turns; a task still open after them ends unplaced.

Outcomes of an attempt: placed_stable (the success); tipped, missed, dropped_low and not_lifted (recoverable: the object
stays on the table and you may try again); dropped_high and rolled_off (the object falls to the floor) and crushed,
bruised, broken and spilled (the object is damaged). Those last six are the irreversible outcomes: what one costs the user
and whether anyone puts the object back depend on the situation the user describes. A probe can also knock the object
off the table (thrown) or damage it."""

SYSTEM_MENU = """The skill's menu: every attempt row has the skill's probabilities of the eleven outcomes (from a model trained in
this lab, averaged over the skill's belief about the object's hidden mass, friction and centre of mass), its time and its
expected cost. A probe row has the probabilities of its result kinds, P(thrown), its value of information (the expected fall
in the best row's cost after the probe; it can be below 0) and its expected cost. The cost rule: an attempt row's expected
cost sums each outcome's probability times its cost (an irreversible outcome costs 5 times the care factor, a recoverable
one 1, not_lifted 0.5), 0.05 per second of time, and the abstain cost A times P(the object does not end placed_stable). A
probe row's expected cost is its time, its risk and the expected cost of the best row after it. `decline` costs A. The
recommendation: commit when the best row costs at most A; probe when a probe row costs clearly less than both; else
decline. EVERY cost on the menu assumes the default care factor 10 (an irreversible outcome costs 50 and ends the task: no
recovery) and A = 10, the value of the task. The user's instruction can say otherwise. Keep the probabilities (they are
not your guesses to replace); re-weigh what each outcome would cost this user before you choose."""

SYSTEM_REC = """The skill shows its recommendation and one row: the row it recommends, or, when it recommends decline, its best
row, which it declines. The row has the skill's outcome probabilities (from a model trained in this lab) and its expected
cost under the cost rule: each outcome's probability times its cost (an irreversible outcome costs 5 times the care factor,
a recoverable one 1, not_lifted 0.5), 0.05 per second, and the abstain cost A times P(the object does not end placed). A
probe row has its value of information and its expected cost after the probe. `decline` costs A. EVERY cost assumes the
default care factor 10 (an irreversible outcome costs 50 and ends the task: no recovery) and A = 10, the value of the task.
The user's instruction can say otherwise. Keep the probabilities; re-weigh what each outcome would cost this user before
you choose."""

SYSTEM_NOMENU = """The skill gives no predictions: you see the actions it can run, with their grip and motion. After a probe you read
its raw reading (for example the measured mass in kg); after an attempt, its outcome and the pad readouts. Ending without
the object placed costs the user the value of the task."""

TASK = {
    "menu": """Task: pick up the {object} and set it down at `{destination}`. The skill has assessed the pick: its menu is
below. If the recommendation is a probe, run it and read how the belief changed. Then choose a menu row and execute it, or
decline when the recommendation is decline: report its reason. If an attempt fails, read its diagnosis; you can retry from
the fresh menu in its result. Finish when the object has been set down, when it is out of reach, when you decline, or when
every row you are willing to run has been tried. The menu's costs assume a default care level and no recovery: weigh them
against the instruction above.""",
    "nomenu": """Task: pick up the {object} and set it down at `{destination}`. The actions the skill can run are below. You may
run a probe first and read its result. Then choose an action and execute it, or decline: report its reason. If an attempt
fails, read its result; you can retry. Finish when the object has been set down, when it is out of reach, when you
decline, or when every action you are willing to run has been tried. Weigh the risks against the instruction above.""",
    # A post-hoc variant (after the Codex review of the pilot): "menu" tells the planner to run a recommended probe and to
    # decline on a recommended decline, which the system prompt's "re-weigh" contradicts. This text leaves the choice open.
    "menu_neutral": """Task: pick up the {object} and set it down at `{destination}`. The skill has assessed the pick: its menu is
below. The recommendation is the skill's choice under the menu's default costs: advice, not an order. Choose the action
that is best for this user: a probe (then read how the belief changed), a menu row to execute, or decline (report your
reason). If an attempt fails, read its diagnosis; you can retry from the fresh menu in its result. Finish when the object
has been set down, when it is out of reach, when you decline, or when every row you are willing to run has been tried. The
menu's costs assume a default care level and no recovery: weigh them against the instruction above.""",
}
TASK_TEXTS = ("v1", "neutral")     # v1: the pilot as run; neutral: the post-hoc variant


def system_prompt(arm: str, max_calls: int = MAX_CALLS) -> str:
    part = {"menu_planner": SYSTEM_MENU, "rec_planner": SYSTEM_REC, "nomenu_planner": SYSTEM_NOMENU}[arm]
    return SYSTEM.format(max_calls=max_calls) + "\n\n" + part


def user_text(instruction_text: str, object_name: str, destination: str, arm: str, task: str = "v1") -> str:
    if task not in TASK_TEXTS:
        raise ValueError(f"unknown task text {task!r}")
    key = "nomenu" if arm == "nomenu_planner" else ("menu" if task == "v1" else "menu_neutral")
    task = TASK[key].format(object=object_name, destination=destination)
    return f"User: {instruction_text}\n\n{task}"


def _f(x: float | None, nd: int = 2) -> str:
    return "n/a" if x is None else f"{float(x):.{nd}f}"


def _row_action(o: dict[str, Any], destination: str) -> str:
    a = o["action"]
    if o["kind"] == "probe":
        return f"{PROBE_TEXT.get(a['probe'], a['probe'])}; {a['force']:.0f} N per pad"
    what = o.get("reason") or "the standard grasp and carry"
    parts = [what, f"pinch at {a['s']:.2f} of the long axis", f"{a['force']:.0f} N per pad", f"lift {a['v_lift']:g} m/s",
             f"carry {a['v_transport']:g} m/s"]
    if a.get("release_height") is not None:
        parts.append(f"release {a['release_height']:.2f} m up")
    if a.get("destination") != destination:
        parts.append(f"to {a['destination']}, not the requested place")
    return ", ".join(parts)


def _top(p: dict[str, float], keys) -> str:
    items = sorted(((k, float(p.get(k) or 0.0)) for k in keys), key=lambda kv: -kv[1])
    return ", ".join(f"{k} {v:.2f}" for k, v in items if v >= 0.005) or "none"


def _row_numbers(o: dict[str, Any]) -> str:
    if o["kind"] == "probe":
        v = o["probe"]
        kinds = ", ".join(f"{k} {float(x):.2f}" for k, x in sorted(v["p_kind"].items(), key=lambda kv: -float(kv[1] or 0)))
        return (f"result kinds {kinds}; P(thrown) {_f(v['p_thrown'])}; value of information {_f(v['evoi'])} (Monte Carlo error "
                f"{_f(v['evoi_se'])}); expected cost {_f(o['expected_cost'])} (its time and risk plus the best row's expected cost "
                f"after it, {_f(v['expected_cost_after'])})")
    p = o["p"]
    irr, rec = sum(float(p[k]) for k in IRREVERSIBLE), sum(float(p[k]) for k in RECOVERABLE)
    return (f"P(placed_stable) {_f(p['placed_stable'])}; irreversible {irr:.2f} ({_top(p, IRREVERSIBLE)}); recoverable "
            f"{rec:.2f} ({_top(p, RECOVERABLE)}); time {o['cost_s']:.0f} s; expected cost {_f(o['expected_cost'])}")


def _decline_line(arm: str, a: dict[str, Any]) -> str:
    if arm == "nomenu_planner":
        return "- decline: abstain; the task ends without the object placed."
    A = next((o["expected_cost"] for o in a["options"] if o["kind"] == "decline"), None)
    return f"- decline: abstain; the task ends without the object placed; cost {_f(A)} (A)."


def shown_rows(a: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    """The rows an arm sees (decline excluded): every row but HIDDEN_ROWS; the rec planner only the recommended row, or on
    decline the best row the rule declines (none when there is no row)."""
    rows = [o for o in a["options"] if o["kind"] != "decline" and o["id"] not in HIDDEN_ROWS]
    if arm != "rec_planner":
        return rows
    rid = a.get("recommended_option") if a.get("recommendation") != "decline" else a.get("best_option")
    return [o for o in rows if o["id"] == rid]


def render_menu(a: dict[str, Any], arm: str, destination: str, rng: np.random.Generator) -> tuple[str, list[str]]:
    """(text, option ids) of one assessment as ``arm`` sees it; the rows in a random order (``rng``), decline last."""
    rows = shown_rows(a, arm)
    if arm != "rec_planner":
        rows = [rows[i] for i in rng.permutation(len(rows))]
    ids = [o["id"] for o in rows] + ["decline"]
    lines = []
    if arm == "nomenu_planner":
        lines.append("Actions (in random order; the skill gives no predictions):")
        lines += [f"- {o['id']} ({o['kind']}): {_row_action(o, destination)}." for o in rows]
        lines.append(_decline_line(arm, a))
        return "\n".join(lines), ids
    if arm == "rec_planner":
        lines.append("The skill's recommended row and the abstain option (every cost assumes the default care factor 10 and no "
                     "recovery):")
    else:
        lines.append(f"Menu ({len(rows) + 1} rows, in random order; every cost assumes the default care factor 10 and no "
                     "recovery):")
    for o in rows:
        lines.append(f"- {o['id']} ({o['kind']}): {_row_action(o, destination)}.\n  {_row_numbers(o)}.")
    lines.append(_decline_line(arm, a))
    rec, rid = a.get("recommendation"), a.get("recommended_option")
    if rec == "decline":
        lines.append(f"Recommendation: decline. Reason: {a.get('decline_reason')}."
                     + (" The row above is the skill's best row, which it declines." if arm == "rec_planner" and rows else ""))
    else:
        lines.append(f"Recommendation: {rec} ({rid}).")
        if rec == "probe" and arm == "menu_planner":
            lines.append(f"Without the probe rows the rule would {a.get('without_probe')}: {a.get('decline_reason')}.")
    if (a.get("guard") or {}).get("flag"):
        lines.append("The novelty guard flags this object as unlike the training objects: the rows' predictions are not "
                     "trusted.")
    if arm == "menu_planner":
        belief = [e for e in a.get("evidence") or [] if e.startswith(("mass median", "friction median", "P(CoM"))]
        if belief:
            lines.append("The skill's belief: " + "; ".join(belief) + ".")
    return "\n".join(lines), ids


def render_result(d: dict[str, Any], row: dict[str, Any], arm: str) -> str:
    """What the planner reads after one action."""
    if row["kind"] == "probe":
        reading = ", ".join(f"{k} {v:.3g}" if isinstance(v, (int, float)) else f"{k} {v}" for k, v in (d.get("reading") or {}).items())
        text = (f"You ran {row['id']}: result `{d.get('kind')}`" + (f" (reading: {reading})" if reading else " (no reading)")
                + f"; displacement {1000 * float(d.get('displacement_m') or 0):.0f} mm, tilt {float(d.get('tilt_deg') or 0):.0f} deg; "
                + ("thrown off the table" if d.get("thrown") else "not thrown")
                + (f"; damage: {d['damage'].get('label')}" if d.get("damage") else "") + ".")
        if arm != "nomenu_planner" and d.get("belief"):
            b = d["belief"]
            text += (f" The belief now: mass median {b['mass_median']:.3f} kg (80 % {b['mass_80'][0]:.3f} to {b['mass_80'][1]:.3f}), "
                     f"friction median {b['mu_median']:.2f} (80 % {b['mu_80'][0]:.2f} to {b['mu_80'][1]:.2f}).")
    else:
        outcome = d.get("outcome")
        kind = "the success" if outcome == "placed_stable" else ("irreversible" if outcome in IRREVERSIBLE else "recoverable")
        text = f"You ran {row['id']}: outcome `{outcome}` ({kind})" + (f"; damage: {d['damage'].get('label')}" if d.get("damage") else "") + "."
        if arm == "nomenu_planner":
            ro = d.get("readouts") or {}
            vals = ", ".join(f"{k} {ro[k]:.3g}" if isinstance(ro[k], float) else f"{k} {ro[k]}" for k in READOUT_KEYS if k in ro)
            text += f" Pad readouts: {vals}." if vals else ""
        elif d.get("diagnosis"):
            text += " Diagnosis: " + " ".join(str(x) for x in d["diagnosis"])
    if d.get("restored") is not None:
        text += (" The object was caught and restored to its start pose (a reset of 20 s). No probe runs after a restore; you "
                 "may try again.")
    return text


# ---------------------------------------------------------------- the planner loop

def tool_schema(ids: list[str]) -> list[dict[str, Any]]:
    return [{"type": "function", "function": {
        "name": "choose",
        "description": "Run one action by its option_id (an attempt row or a probe row), or decline: the task then ends without "
                       "the object placed.",
        "parameters": {"type": "object", "properties": {
            "option_id": {"type": "string", "enum": list(ids)},
            "reason": {"type": "string", "description": "One or two sentences: why this action, given the user's instruction."}},
            "required": ["option_id", "reason"]}}}]


def parse_choice(r: dict[str, Any], ids: list[str]) -> str | None:
    """The option id of a model reply: the `choose` call's option_id, else the first {"option_id": ...} in its text; None when
    neither names an id on the list."""
    if r.get("tool") == "choose" and isinstance(r.get("args"), dict):
        oid = r["args"].get("option_id")
        return oid if oid in ids else None
    m = re.search(r'"option_id"\s*:\s*"([A-Za-z_]+)"', r.get("content") or "")
    return m.group(1) if m and m.group(1) in ids else None


def _probe_record(d: dict[str, Any], row: dict[str, Any], position: int, after_attempts: int) -> dict[str, Any]:
    return {"probe": d["probed"], "option": row["id"], "position": position, "after_attempts": after_attempts, "kind": d["kind"],
            "reading": d.get("reading"), "thrown": bool(d.get("thrown")), "damage": d.get("damage"),
            "artifact": bool(d.get("artifact", False)), "declared": d.get("declared")}


def _attempt_record(d: dict[str, Any], row: dict[str, Any], a: dict[str, Any]) -> dict[str, Any]:
    return {"option": row["id"], "kind": row["kind"], "action": row["action"], "recommendation": a.get("recommendation"),
            "declared": {"p": row["p"], "expected_cost": row["expected_cost"]}, "outcome": d["outcome"],
            "placed_at_requested": d.get("placed_at_requested"), "cost_s": row["cost_s"], "artifact": bool(d.get("artifact", False))}


def planner_loop(tools: Any, model: Callable[..., dict[str, Any]], arm: str, instruction_text: str, object_name: str,
                 destination: str, rng: np.random.Generator, max_calls: int = MAX_CALLS,
                 max_attempts: int = MAX_ATTEMPTS, task: str = "v1") -> dict[str, Any]:
    """One planner case: ``tools.assess()`` gives the first assessment payload and ``tools.act(option_id)`` runs a row
    (pick_and_place's payload); ``model(messages, tool_schema)`` returns {tool, args, content, reasoning, seconds, usage,
    error}. Returns the case record (calibrate's probe and attempt fields, plus calls, decisions, overrides, malformed,
    failed_calls and the transcript)."""
    case: dict[str, Any] = {"protocol": arm, "recommendation": None, "best_option": None, "guard": None, "probes": [],
                            "attempts": [], "restores": [], "ended": None, "calls": [], "decisions": [], "overrides": 0,
                            "malformed": 0, "failed_calls": 0, "transcript": []}

    def end(reason: str, error: str | None = None) -> dict[str, Any]:
        case["ended"] = reason
        if error is not None:
            case["error"] = error
        case["artifact"] = any(x.get("artifact") for x in case["probes"] + case["attempts"])
        return case

    a = tools.assess()
    if "error" in a:
        return end("error", a["error"])
    if not a.get("assessed", True):
        return end("unreachable")
    case.update(recommendation=a.get("recommendation"), best_option=a.get("best_option"), guard=a.get("guard"),
                first_menu={"without_probe": a.get("without_probe"), "recommended_option": a.get("recommended_option")})
    text, ids = render_menu(a, arm, destination, rng)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt(arm, max_calls)},
                                      {"role": "user", "content": user_text(instruction_text, object_name, destination, arm, task) + "\n\n" + text}]
    case["transcript"] = list(messages)
    rows = {o["id"]: o for o in a["options"]}
    while True:
        if len(case["calls"]) >= max_calls:
            return end("call_cap")
        r = model(messages, tool_schema(ids))
        case["calls"].append({k: r.get(k) for k in ("tool", "args", "content", "reasoning", "seconds", "usage", "error", "finish_reason")})
        if r.get("error"):
            case["failed_calls"] += 1
        choice = parse_choice(r, ids)
        call_id = f"call_{len(case['calls'])}"
        if choice is None:
            case["malformed"] += 1
            if r.get("tool") == "choose":
                fix = [{"role": "assistant", "content": r.get("content") or "", "tool_calls": [
                           {"id": call_id, "type": "function", "function": {"name": "choose", "arguments": json.dumps(r.get("args") or {})}}]},
                       {"role": "tool", "tool_call_id": call_id,
                        "content": f"Error: option_id must be one of {ids}. Nothing ran. Call `choose` again."}]
            else:
                fix = [{"role": "assistant", "content": r.get("content") or "(no tool call)"},
                       {"role": "user", "content": f"Nothing ran: call the tool `choose` once, with option_id one of {ids}."}]
            messages += fix
            case["transcript"] += fix
            continue
        reason = (r.get("args") or {}).get("reason") if isinstance(r.get("args"), dict) else None
        case["decisions"].append({"step": len(case["decisions"]), "recommendation": a.get("recommendation"),
                                  "recommended_option": a.get("recommended_option"), "best_option": a.get("best_option"),
                                  "guard_flag": bool((a.get("guard") or {}).get("flag")), "choice": choice, "reason": reason})
        if a.get("recommendation") == "decline" and choice != "decline":
            case["overrides"] += 1
        call = {"role": "assistant", "content": r.get("content") or "", "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "choose", "arguments": json.dumps({"option_id": choice, "reason": reason})}}]}
        if choice == "decline":
            case["transcript"].append(call)
            return end("abstained")
        row = rows[choice]
        d = tools.act(choice)
        if "error" in d and "outcome" not in d and "probed" not in d:
            case["transcript"].append(call)
            return end("error", str(d["error"]))
        ended = None
        if row["kind"] == "probe":
            case["probes"].append(_probe_record(d, row, len(case["probes"]), len(case["attempts"])))
            if d.get("restored") is not None:
                case["restores"].append(dict(d["restored"], probe=d["probed"]))
            elif d.get("kind") == DAMAGED or d.get("damage"):
                ended = "damaged"
            elif d.get("thrown"):
                ended = "thrown"
        else:
            case["attempts"].append(_attempt_record(d, row, a))
            restored = d.get("restored") is not None
            if restored:
                case["restores"].append(dict(d["restored"], option=row["id"]))
            if d["outcome"] == "placed_stable":
                ended = "placed"
            elif (d["outcome"] not in RECOVERABLE and not restored) or len(case["attempts"]) >= max_attempts:
                ended = "failed"
        result = render_result(d, row, arm)
        if ended is not None:
            case["transcript"] += [call, {"role": "tool", "tool_call_id": call_id, "content": result + " The task is over."}]
            return end(ended)
        a = d.get("assessment") or {}
        if "error" in a:
            return end("error", a["error"])
        if not a.get("assessed", False):
            case["transcript"] += [call, {"role": "tool", "tool_call_id": call_id, "content": result + f" {a.get('reason')}."}]
            return end("unreachable")
        text, ids = render_menu(a, arm, destination, rng)
        rows = {o["id"]: o for o in a["options"]}
        step = [call, {"role": "tool", "tool_call_id": call_id, "content": result + "\n\n" + text}]
        messages += step
        case["transcript"] += step


# ---------------------------------------------------------------- the model

class GemmaPlanner:
    """The served model through its OpenAI-compatible endpoint (thinking on by default); three tries per call."""

    def __init__(self, base_url: str, model: str = MODEL, max_tokens: int = 6144, temperature: float = 0.0,
                 thinking: bool = True, timeout: float = 600.0):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key="none", timeout=timeout)
        self.model, self.max_tokens, self.temperature, self.thinking = model, max_tokens, temperature, thinking

    def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        t0, last = time.time(), None
        for k in range(3):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=messages, tools=tools, tool_choice="auto", max_tokens=self.max_tokens,
                    temperature=self.temperature, extra_body={"chat_template_kwargs": {"enable_thinking": self.thinking}})
                m = r.choices[0].message
                call = (m.tool_calls or [None])[0]
                args = None
                if call is not None:
                    try:
                        args = json.loads(call.function.arguments)
                    except (TypeError, ValueError):
                        args = {"_raw": call.function.arguments}
                return {"tool": None if call is None else call.function.name, "args": args, "content": m.content or "",
                        "reasoning": getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None) or "",
                        "seconds": round(time.time() - t0, 2), "finish_reason": r.choices[0].finish_reason,
                        "usage": {"prompt": r.usage.prompt_tokens, "completion": r.usage.completion_tokens}, "error": None}
            except Exception as exc:  # noqa: BLE001  a failed call is counted, never fatal
                last = exc
                time.sleep(5 * (k + 1))
        return {"tool": None, "args": None, "content": "", "reasoning": "", "seconds": round(time.time() - t0, 2),
                "finish_reason": None, "usage": {}, "error": str(last)[:300]}


class StubPlanner:
    """A model stand-in for tests and dry runs: a seeded random option id from the tool's list (decline one time in four)."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        ids = tools[0]["function"]["parameters"]["properties"]["option_id"]["enum"]
        rows = [i for i in ids if i != "decline"]
        choice = "decline" if not rows or self.rng.random() < 0.25 else str(self.rng.choice(rows))
        return {"tool": "choose", "args": {"option_id": choice, "reason": "stub"}, "content": "", "reasoning": "", "seconds": 0.0,
                "finish_reason": "tool_calls", "usage": {}, "error": None}


# ---------------------------------------------------------------- the lab

class LabTools:
    """The skill tools on the live lab for one planner case: the menu at ``care`` (prims.menu_recovery prices it)."""

    def __init__(self, prims: Any, destination: str, care: float):
        self.prims, self.destination, self.care, self.store = prims, destination, care, {}

    def assess(self) -> dict[str, Any]:
        from rpent.skills.pick_place.tools import assess_pick

        return assess_pick(object="object", destination=self.destination, care=self.care, primitives=self.prims, store=self.store).data

    def act(self, option_id: str) -> dict[str, Any]:
        from rpent.skills.pick_place.tools import pick_and_place

        return pick_and_place(object="object", destination=self.destination, care=self.care, option_id=option_id,
                              primitives=self.prims, store=self.store).data


def reset_case(prims: Any, object_id: str, idx: int, seed: int) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Reset the lab to the case's draw as calibrate.run_case does (wide spreads, no fixtures) and reset prims.rng to the
    case's menu stream; returns (destination, reset_info, truth)."""
    from robots.mujoco_lab import calibrate
    from robots.mujoco_lab.adapter import object_props
    from robots.mujoco_lab.prior import fragility_world

    env = prims.env
    env.force_noise = getattr(prims, "force_noise", None)
    world = fragility_world(getattr(prims, "manifest", None))
    theta, com_end, destination, hidden = calibrate.case_draw(seed, object_id, idx, object_props(object_id)["mass"], False,
                                                              fragility=world, spread=1.0)
    if world is not None:
        env.slosh_zeta = world[1].slosh_zeta
        env.slosh_scale = world[1].slosh_scale
    env.reset(object_id, theta, seed=seed * 1000 + idx, com_end=com_end, fixture=None, fill=hidden["fill"], frag=hidden["frag"])
    prims.rng = calibrate._stream(seed, object_id, idx, 1)
    truth = env.truth()
    return destination, dict(getattr(env, "reset_info", None) or {}), {k: truth[k] for k in ("mass", "mu", "com_s")}


_WORKER: dict[str, Any] = {}


def _model(cfg: dict[str, Any]) -> Callable[..., dict[str, Any]]:
    if "model" not in _WORKER:
        _WORKER["model"] = StubPlanner(os.getpid()) if cfg.get("stub") else GemmaPlanner(cfg["base_url"], cfg.get("model", MODEL))
    return _WORKER["model"]


def run_job(item: tuple[dict[str, Any], dict[str, Any]]) -> list[dict[str, Any]]:
    """Every arm of one (case, setting) job, in turn on one lab; a row per arm with the case record and its realized cost."""
    from robots.mujoco_lab import calibrate

    job, cfg = item
    setting = cfg["settings"][job["setting"]]
    folds = cfg["folds"]
    prims = calibrate._primitives(os.path.join(cfg["crossfit"], f"fold_{folds[job['object']]}"), None, None)
    prims.abstain_cost = float(setting["abstain_cost"])
    prims.force_noise = calibrate.job_force_noise(None, prims.manifest)
    prims.probe_time_scale = 1.0
    prims.truth_spread = prims.prior_spread = 1.0
    prims.reset_s, prims.reset_price = setting_recovery(setting)
    rows = []
    for arm in cfg["arms"]:
        t0 = time.time()
        destination, reset_info, truth = reset_case(prims, job["object"], job["idx"], job["seed"])
        oracle = arm == "oracle_rule"
        prims.menu_recovery = setting_recovery(setting) if oracle else RECOVERY[DEFAULT["recovery"]]
        care = float(setting["care_scale"]) if oracle else float(DEFAULT["care_scale"])
        text = instruction(job, destination)
        try:
            if arm in RULES:
                case = calibrate.run_protocol(prims, "policy", destination, np.random.default_rng(0), care=care)
            else:
                rng = np.random.default_rng([job["seed"], zlib.crc32(job["object"].encode()), job["idx"], 7])
                case = planner_loop(LabTools(prims, destination, care), _model(cfg), arm, text, display_object(job["object"]), destination, rng,
                                    max_calls=cfg.get("max_calls", MAX_CALLS), task=cfg.get("task", "v1"))
        except Exception as exc:  # noqa: BLE001  one bad case must not stop the run
            case = {"ended": "error", "error": f"{type(exc).__name__}: {exc}"[:300], "probes": [], "attempts": []}
        case["artifact"] = calibrate.case_artifact(case, reset_info)
        row = dict(job, arm=arm, destination=destination, instruction=text, truth=truth, reset=reset_info, menu_care=care,
                   task=cfg.get("task", "v1"), **case)
        row["protocol_cost"] = case.get("cost")          # run_protocol's own cost (the menu's care): the oracle's equals cost
        row["cost"] = realized_cost(case, setting)
        row["irreversible"] = irreversible_events(case)
        row["seconds"] = round(time.time() - t0, 2)
        rows.append(row)
        print(f"{job['object']} {job['idx']} {job['setting']} {arm}: {row['ended']} cost {row['cost']:.2f} "
              f"calls {len(row.get('calls') or [])} {row['seconds']:.0f}s", file=sys.stderr, flush=True)
    prims.menu_recovery = None
    return rows


def _done(path: str) -> set[tuple]:
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out.add((r["object"], r["idx"], r["setting"], r["arm"]))
    return out


def run(plan_path: str, split: str, arms: list[str], workers: int, out_dir: str, base_url: str | None, stub: bool,
        limit: int | None = None, crossfit: str | None = None, max_calls: int = MAX_CALLS, task: str = "v1") -> str:
    """Run a split's jobs (resuming: a (case, setting, arm) already in rows.jsonl is skipped); returns the rows path."""
    from robots.mujoco_lab.train_models import repo_path

    plan = json.load(open(plan_path))
    crossfit = crossfit or repo_path("robots/mujoco_lab/data/m2b2/folds")
    cfg = {"arms": arms, "settings": plan["settings"], "crossfit": crossfit,
           "folds": json.load(open(os.path.join(crossfit, "folds.json"))), "base_url": base_url, "stub": stub,
           "max_calls": max_calls, "task": task}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "rows.jsonl")
    from robots.mujoco_lab.calibrate import code_revision

    with open(os.path.join(out_dir, f"run_config_{time.strftime('%Y%m%dT%H%M%S')}.json"), "w") as f:
        json.dump({"plan": os.path.abspath(plan_path), "split": split, "arms": arms, "workers": workers, "base_url": base_url,
                   "stub": stub, "limit": limit, "crossfit": crossfit, "max_calls": max_calls, "task": task, "model": MODEL,
                   "planner": {"temperature": 0.0, "max_tokens": 6144, "thinking": True}, "code": code_revision()}, f, indent=1)
    done = _done(path)
    jobs = [j for j in plan["jobs"] if j["split"] == split][: limit or None]
    todo = []
    for j in jobs:
        left = [a for a in arms if (j["object"], j["idx"], j["setting"], a) not in done]
        if left:
            todo.append((j, dict(cfg, arms=left)))
    print(f"e1 run: {len(todo)} of {len(jobs)} jobs to run ({split}), arms {arms}", file=sys.stderr, flush=True)
    with open(path, "a") as f:
        if workers <= 1:
            for item in todo:
                for r in run_job(item):
                    f.write(json.dumps(r, default=str) + "\n")
                f.flush()
        else:
            with mp.get_context("spawn").Pool(workers) as pool:
                for part in pool.imap_unordered(run_job, todo):
                    for r in part:
                        f.write(json.dumps(r, default=str) + "\n")
                    f.flush()
    return path


# ---------------------------------------------------------------- the summary

def load_rows(path: str) -> list[dict[str, Any]]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def _rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    calls = [len(r.get("calls") or []) for r in rows]
    return {"n": n, "cost": float(np.mean([r["cost"] for r in rows])) if n else None,
            "placed": float(np.mean([r["ended"] == "placed" for r in rows])) if n else None,
            "abstain": float(np.mean([r["ended"] == "abstained" for r in rows])) if n else None,
            "probe": float(np.mean([bool(r.get("probes")) for r in rows])) if n else None,
            "irreversible": float(np.mean([r.get("irreversible", 0) > 0 for r in rows])) if n else None,
            "call_cap": int(sum(r["ended"] == "call_cap" for r in rows)),
            "errors": int(sum(r["ended"] == "error" for r in rows)),
            "artifacts": int(sum(bool(r.get("artifact")) for r in rows)),
            "overrides_cases": int(sum((r.get("overrides") or 0) > 0 for r in rows)),
            "overrides": int(sum(r.get("overrides") or 0 for r in rows)),
            "guard_overrides": int(sum(d["recommendation"] == "decline" and d["choice"] != "decline" and d.get("guard_flag")
                                       for r in rows for d in r.get("decisions") or [])),
            "malformed": int(sum(r.get("malformed") or 0 for r in rows)),
            "failed_calls": int(sum(r.get("failed_calls") or 0 for r in rows)),
            "calls_mean": float(np.mean(calls)) if n else None, "calls_max": int(max(calls)) if n else None}


def summarize(rows: list[dict[str, Any]], n_boot: int = 2000) -> dict[str, Any]:
    """Per arm and setting: the mean realized cost and the rates; the paired contrasts menu_planner minus each other arm
    (and default_rule minus oracle_rule) per setting and pooled, with 95 % object-bootstrap intervals, over the (case,
    setting) pairs where both arms ran without an error or an artifact."""
    arms = [a for a in ARMS if any(r["arm"] == a for r in rows)]
    settings = sorted({r["setting"] for r in rows})
    key = {(r["object"], r["idx"], r["setting"], r["arm"]): r for r in rows}
    out: dict[str, Any] = {"n_rows": len(rows), "arms": arms, "settings": settings, "per": {}, "contrasts": {},
                           "oracle_cost_mismatches": sum(abs(float(r["protocol_cost"]) - r["cost"]) > 1e-9 for r in rows
                                                         if r["arm"] == "oracle_rule" and r.get("protocol_cost") is not None
                                                         and r["ended"] != "error")}
    for arm in arms:
        out["per"][arm] = {s: _rates([r for r in rows if r["arm"] == arm and r["setting"] == s]) for s in settings}
        out["per"][arm]["all"] = _rates([r for r in rows if r["arm"] == arm])
    pairs = [("menu_planner", b) for b in arms if b != "menu_planner"] + [("default_rule", "oracle_rule")]
    for A, B in pairs:
        if A not in arms or B not in arms:
            continue
        name = f"{A} - {B}"
        out["contrasts"][name] = {}
        for s in settings + ["all"]:
            d, objs, dropped = [], [], 0
            for (o, i, st, arm), r in key.items():
                if arm != A or (s != "all" and st != s):
                    continue
                q = key.get((o, i, st, B))
                if q is None:
                    continue
                if r["ended"] == "error" or q["ended"] == "error" or r.get("artifact") or q.get("artifact"):
                    dropped += 1
                    continue
                d.append(r["cost"] - q["cost"])
                objs.append(o)
            if d:
                mean, lo, hi = object_bootstrap(d, objs, n_boot=n_boot, seed=zlib.crc32(f"{name}/{s}".encode()))
                out["contrasts"][name][s] = {"n": len(d), "objects": len(set(objs)), "mean": mean, "ci": [lo, hi], "dropped": dropped}
    return out


def markdown(summary: dict[str, Any], title: str) -> str:
    lines = [f"## {title}", "", "| arm | setting | n | mean cost | placed | abstain | probe | irreversible | overrides (cases) | "
             "malformed | failed calls | calls/case (max) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, per in summary["per"].items():
        for s, v in per.items():
            if not v["n"]:
                continue
            lines.append(f"| {arm} | {s} | {v['n']} | {v['cost']:.3f} | {v['placed']:.3f} | {v['abstain']:.3f} | {v['probe']:.3f} | "
                         f"{v['irreversible']:.3f} | {v['overrides']} ({v['overrides_cases']}) | {v['malformed']} | {v['failed_calls']} | "
                         f"{v['calls_mean']:.2f} ({v['calls_max']}) |")
    lines += ["", "| contrast | setting | n (objects) | mean difference [95 % object-bootstrap interval] | dropped |", "|---|---|---|---|---|"]
    for name, per in summary["contrasts"].items():
        for s, v in per.items():
            lines.append(f"| {name} | {s} | {v['n']} ({v['objects']}) | {v['mean']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}] | {v['dropped']} |")
    return "\n".join(lines) + "\n"


def _transcript_md(r: dict[str, Any]) -> str:
    lines = [f"**{r['arm']}**, {r['object']} idx {r['idx']}, setting {r['setting']}, destination {r['destination']}: ended "
             f"{r['ended']}, cost {r['cost']:.2f}", ""]
    calls = iter(r.get("calls") or [])
    for m in r.get("transcript") or []:
        if m["role"] == "system":
            continue
        if m["role"] == "assistant":
            c = next(calls, {}) or {}
            if c.get("reasoning"):
                lines += ["> thinking: " + c["reasoning"][:1500].replace("\n", "\n> "), ""]
            for t in m.get("tool_calls") or []:
                lines.append(f"`choose({t['function']['arguments']})`")
            if m.get("content"):
                lines.append(m["content"][:500])
            lines.append("")
        else:
            lines += [f"*{m['role']}*:", "", "```", m["content"], "```", ""]
    return "\n".join(lines)


def examples(rows: list[dict[str, Any]], base: str = "default_rule") -> str:
    """Three menu_planner transcripts against ``base``: the largest gain, the largest loss, and an override (the one with the
    largest cost difference in either direction), over the pairs the contrasts keep (no error or artifact in either arm)."""
    key = {(r["object"], r["idx"], r["setting"], r["arm"]): r for r in rows}
    pairs = []
    for (o, i, s, arm), r in key.items():
        q = key.get((o, i, s, base))
        if (arm == "menu_planner" and q is not None and r["ended"] != "error" and q["ended"] != "error"
                and not r.get("artifact") and not q.get("artifact")):
            pairs.append((r["cost"] - q["cost"], r, q))
    if not pairs:
        return ""
    pairs.sort(key=lambda t: t[0])
    picks = [("The planner beat the rule", pairs[0]), ("The planner lost to the rule", pairs[-1])]
    over = [t for t in pairs if (t[1].get("overrides") or 0) > 0]
    if over:
        picks.append(("An override of a decline", max(over, key=lambda t: abs(t[0]))))
    out = []
    for title, (d, r, q) in picks:
        out += [f"### {title} (difference {d:+.2f})", "", f"Instruction: {r['instruction']}", "",
                f"The rule ({base}): ended {q['ended']}, cost {q['cost']:.2f}, probes "
                f"{[p['probe'] for p in q.get('probes') or []]}, attempts {[t['option'] + ':' + t['outcome'] for t in q.get('attempts') or []]}.",
                "", _transcript_md(r), ""]
    return "\n".join(out)


# ---------------------------------------------------------------- the command line

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="e1")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--templates", required=True)
    p.add_argument("--final-cases", required=True)
    p.add_argument("--tune-cases", required=True)
    p.add_argument("--n-final", type=int, default=80)
    p.add_argument("--n-tune", type=int, default=20)
    p.add_argument("--plan-seed", type=int, default=0)
    p.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--plan", required=True)
    r.add_argument("--split", choices=("tune", "final"), required=True)
    r.add_argument("--arms", default=",".join(ARMS))
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--base-url", default=None)
    r.add_argument("--stub", action="store_true")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--max-calls", type=int, default=MAX_CALLS)
    r.add_argument("--task", choices=TASK_TEXTS, default="v1")
    r.add_argument("--out", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--dir", required=True)
    s.add_argument("--n-boot", type=int, default=2000)
    s.add_argument("--title", default="E1 pilot")
    s.add_argument("--wandb", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        templates = json.load(open(a.templates))
        final, tune = (json.load(open(x)) for x in (a.final_cases, a.tune_cases))
        jobs = (plan_jobs(final["cases"], templates, a.n_final, "final", int(final["seed"]), a.plan_seed)
                + plan_jobs(tune["cases"], templates, a.n_tune, "tune", int(tune["seed"]), a.plan_seed))
        plan = {"templates": os.path.abspath(a.templates), "templates_version": templates.get("version"), "settings": templates["settings"],
                "default_rule": templates.get("default_rule"), "final_cases": os.path.abspath(a.final_cases),
                "tune_cases": os.path.abspath(a.tune_cases), "plan_seed": a.plan_seed, "jobs": jobs}
        with open(a.out, "w") as f:
            json.dump(plan, f, indent=1)
        print(f"plan: {sum(j['split'] == 'final' for j in jobs)} final and {sum(j['split'] == 'tune' for j in jobs)} tuning jobs -> {a.out}")
        return 0
    if a.cmd == "run":
        if not a.stub and not a.base_url:
            ap.error("give --base-url or --stub")
        arms = [x for x in a.arms.split(",") if x]
        bad = [x for x in arms if x not in ARMS]
        if bad:
            ap.error(f"unknown arms {bad}")
        print(run(a.plan, a.split, arms, a.workers, a.out, a.base_url, a.stub, a.limit, max_calls=a.max_calls, task=a.task))
        return 0
    rows = load_rows(os.path.join(a.dir, "rows.jsonl"))
    summ = summarize(rows, a.n_boot)
    with open(os.path.join(a.dir, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    with open(os.path.join(a.dir, "summary.md"), "w") as f:
        f.write(markdown(summ, a.title))
    with open(os.path.join(a.dir, "examples.md"), "w") as f:
        f.write(examples(rows))
    if a.wandb:
        from robots.mujoco_lab import tracking

        url = tracking.log_run("e1_pilot", {"dir": a.dir, "title": a.title},
                               {"per": {arm: {s: {k: v for k, v in x.items() if isinstance(v, (int, float))} for s, x in per.items()}
                                        for arm, per in summ["per"].items()},
                                "contrasts": {n: {s: {"mean": v["mean"], "lo": v["ci"][0], "hi": v["ci"][1], "n": v["n"]}
                                                  for s, v in per.items()} for n, per in summ["contrasts"].items()}})
        print(f"wandb: {url}")
    print(markdown(summ, a.title))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
