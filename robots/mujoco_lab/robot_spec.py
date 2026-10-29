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

"""MuJoCo lab robot extension: RobotSpec factory, toolkit factory, runtime hooks."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from robots.mujoco_lab.prompt_bundle import system_prompt, user_prompt
from rpent.dashboard.events import DashboardEventSink
from rpent.dashboard.spec import DashboardSpec
from rpent.evaluation.result import RunFinalizationContext
from rpent.memory import MemoryManager
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.utils.config import get_memory_dir, get_repo_root

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_MODEL_PATH = str(DATA_DIR / "success_model.joblib")
DEFAULT_HAZARD_PATH = str(DATA_DIR / "hazard_model.joblib")
DEFAULT_METRICS_PATH = str(DATA_DIR / "success_model_metrics.json")
DEFAULT_BUNDLE = DATA_DIR / "m2"                 # milestone 2's versioned models (manifest.json); milestone 1's stay in data/


def lab_prior_fn(pose: dict[str, Any], n: int = 2000, rng: np.random.Generator | None = None):
    """The lab prior: the corpus generator itself (robots.mujoco_lab.prior.lab_prior)."""
    from robots.mujoco_lab.prior import lab_prior

    return lab_prior(pose, n, rng)


def resolve_bundle(bundle: str | None = None) -> str | None:
    """The model bundle to load: the argument, else RPENT_PICK_PLACE_BUNDLE, else data/m2 when its
    manifest exists, else None (milestone 1's files)."""
    b = bundle or os.environ.get("RPENT_PICK_PLACE_BUNDLE")
    if b:
        return str(b)
    return str(DEFAULT_BUNDLE) if (DEFAULT_BUNDLE / "manifest.json").exists() else None


def manifest_abstain_cost(manifest: dict[str, Any]) -> float | None:
    """The abstain cost of a bundle's decision (milestone 2b: {"decision": {"abstain_cost": 1.0}}); None without
    one (m2: the milestone 2a rule)."""
    A = (manifest.get("decision") or {}).get("abstain_cost")
    return None if A is None else float(A)


def load_bundle(bundle_dir: str, kind: str | None = None) -> dict[str, Any]:
    """The models a bundle's manifest names: the success model of its kind (or ``kind``), the prior (and an m2b
    prior's fragility classes), and the readout and probe models and the novelty guard when present, the decision's
    abstain cost, and the feature and outcome schemas ("m2" when the manifest names none). Every model must read the schema's features."""
    from rpent.skills.pick_place.features import FEATURE_SCHEMAS, model_names

    from robots.mujoco_lab.prior import fragility_world, prior_from_manifest

    man = json.load(open(os.path.join(bundle_dir, "manifest.json")))
    kind = kind or os.environ.get("RPENT_PICK_PLACE_MODEL_KIND") or man.get("kind", "classifier")
    if kind == "hazard" and not man.get("hazard_model"):
        raise ValueError(f"{bundle_dir} has no hazard model (the hazard model is not eligible for its labels)")
    path = os.path.join(bundle_dir, man["hazard_model" if kind == "hazard" else "success_model"])
    if kind == "hazard":
        from rpent.skills.pick_place.hazard_model import HazardModel

        model = HazardModel.load(path)
    else:
        from rpent.skills.pick_place.success_model import SuccessModel

        model = SuccessModel.load(path)
    readout = None
    if man.get("readout_model"):
        from rpent.skills.pick_place.likelihood import ReadoutModel

        readout = ReadoutModel.load(os.path.join(bundle_dir, man["readout_model"]))
        temper = (man.get("tempers") or {}).get("readout")
        if temper is not None:
            readout.temper = float(temper)
    probe_models = None
    if man.get("probe_models"):
        from rpent.skills.pick_place.probes import ProbeModel

        probe_models = {p: ProbeModel.load(os.path.join(bundle_dir, f)) for p, f in man["probe_models"].items()}
        temper = (man.get("tempers") or {}).get("probe")
        if temper is not None:
            for pm in probe_models.values():
                pm.temper = float(temper)
    guard = None
    if man.get("guard"):
        from rpent.skills.pick_place.guard import KnnGuard

        guard = KnnGuard.load(os.path.join(bundle_dir, man["guard"]))
    schema = man.get("feature_schema", "m2")
    fitted = [("success model", model), ("guard", guard), ("readout rotation classifier", getattr(readout, "rot_clf", None))]
    fitted += [(f"probe model {p}", pm) for p, pm in (probe_models or {}).items()]
    for what, m in fitted:
        if m is not None and model_names(m) != FEATURE_SCHEMAS[schema]:
            raise ValueError(f"{bundle_dir}: the {what} reads feature names other than the manifest's {schema!r} schema")
    return {"manifest": man, "kind": kind, "model": model, "feature_schema": schema,
            "outcome_schema": man.get("outcome_schema", "m2"),
            "prior": lab_prior_fn if man.get("prior") == "lab_generator" else prior_from_manifest(man),
            "classes": (fragility_world(man) or (None,))[0],
            "readout_model": readout, "probe_models": probe_models, "guard": guard, "abstain_cost": manifest_abstain_cost(man)}


def load_model(model_path: str | None = None, kind: str | None = None) -> Any:
    """The success model the tools use: the kind chosen at training time unless overridden."""
    kind = kind or os.environ.get("RPENT_PICK_PLACE_MODEL_KIND")
    if kind is None and os.path.exists(DEFAULT_METRICS_PATH):
        kind = json.load(open(DEFAULT_METRICS_PATH)).get("kind", "classifier")
    kind = kind or "classifier"
    if model_path is None:
        model_path = os.environ.get("RPENT_PICK_PLACE_MODEL") or (DEFAULT_HAZARD_PATH if kind == "hazard" else DEFAULT_MODEL_PATH)
    if kind == "hazard":
        from rpent.skills.pick_place.hazard_model import HazardModel

        return HazardModel.load(model_path)
    from rpent.skills.pick_place.success_model import SuccessModel

    return SuccessModel.load(model_path)


def build_lab(sim_dir: str | None = None, model_path: str | None = None, render: bool = False, object_id: str | None = None,
              theta: dict[str, Any] | None = None, seed: int = 0, model_kind: str | None = None, bundle: str | None = None,
              threshold: float | None = None):
    """The convenience factory: an adapter, the models and a random stream (used by the calibration
    check and the tests; the runner goes through get_toolkit). An explicit ``model_path`` loads that
    milestone 1 model; otherwise the bundle (see resolve_bundle), else milestone 1's files. ``theta``'s ``fill`` and
    ``frag`` (milestone 2b) go to the lab's reset as the fill and the hidden fragility."""
    from robots.mujoco_lab.adapter import MuJoCoLabEnv
    from robots.mujoco_lab.toolkit import MuJoCoLabPrimitives

    env = MuJoCoLabEnv(sim_dir=sim_dir, render=render)
    if object_id is not None:
        hidden = dict(theta or {})
        fill, frag = hidden.pop("fill", 0.0), hidden.pop("frag", None)
        env.reset(object_id, hidden, seed=seed, fill=fill, frag=frag)
    rng = np.random.default_rng(seed)
    b = None if model_path else resolve_bundle(bundle)
    if b is not None:
        from robots.mujoco_lab.prior import fragility_world

        B = load_bundle(b, model_kind)
        world = fragility_world(B["manifest"])
        if world is not None:            # the bundle's slosh model, as the calibration sets it
            env.slosh_zeta, env.slosh_scale = world[1].slosh_zeta, world[1].slosh_scale
        return MuJoCoLabPrimitives(env=env, model=B["model"], rng=rng, prior=B["prior"], readout_model=B["readout_model"],
                                   probe_models=B["probe_models"], threshold=threshold, manifest=B["manifest"], guard=B["guard"],
                                   abstain_cost=B["abstain_cost"])
    return MuJoCoLabPrimitives(env=env, model=load_model(model_path, model_kind), rng=rng, threshold=threshold)


def get_toolkit(*, runtime_kwargs: dict[str, Any], dashboard_events: DashboardEventSink, config: RunConfig):
    """Return the lab toolkit (common tools + the pick_place skill tools)."""
    from robots.mujoco_lab.toolkit import MuJoCoLabToolkit

    memory = MemoryManager(root=config.prompt_vars.get("memory_dir") or get_memory_dir("mujoco_lab"))
    return MuJoCoLabToolkit(memory=memory, primitives=runtime_kwargs["primitives"], dashboard_events=dashboard_events)


def care_arg(value: str) -> str | float:
    """--care: a name (low, normal, high) or a number above 0, the care factor itself (costs.care_scale)."""
    from rpent.skills.pick_place.costs import CARE_SCALE, care_scale

    if value in CARE_SCALE:
        return value
    try:
        kappa = float(value)
        care_scale(kappa)
    except ValueError:
        raise argparse.ArgumentTypeError(f"care must be low, normal, high or a number above 0, not {value!r}") from None
    return kappa


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    parser.add_argument("--object", default=None, required=not use_dashboard, help="object id in the lab (e.g. ycb__banana)")
    parser.add_argument("--destination", default="bin", help="where to set it down (see the system prompt)")
    parser.add_argument("--care", default="normal", type=care_arg, help="low, normal, high or a number above 0 (the care factor)")
    parser.add_argument("--theta-json", default=None, help="hidden properties as JSON: mu, mass_scale, com_s, v_lift, and "
                        "fill (0 to 1) and frag (fragility.map_latent's keys) for milestone 2b")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sim-dir", default=None, help="the experiment's sim/ folder (default: RPENT_MUJOCO_SIM_DIR)")
    parser.add_argument("--model-path", default=None, help="success model .joblib (default: the trained one)")
    parser.add_argument("--render", action="store_true", help="attach the main camera image to each state")


def _parse_config(args: argparse.Namespace) -> RunConfig:
    if not args.object:
        raise ValueError("--object is required")
    recipe_tag = f"{args.object}_{args.destination}_s{args.seed}"
    output_dir = getattr(args, "output_dir", None)
    if output_dir is None:
        output_dir = get_repo_root() / "logs" / f"{datetime.now().strftime('%Y%m%d-%H:%M:%S')}_mujoco_lab_{recipe_tag}"
    memory_dir = Path(args.memory_dir) if getattr(args, "memory_dir", None) else get_memory_dir("mujoco_lab")
    prompt_vars = {"object": args.object, "destination": args.destination, "care": args.care, "seed": args.seed,
                   "recipe_tag": recipe_tag, "mode": "eval", "memory_dir": str(memory_dir)}
    return RunConfig(recipe_tag=recipe_tag, output_dir=Path(output_dir), prompt_vars=prompt_vars,
                     task_desc={"object": args.object, "destination": args.destination, "seed": args.seed, "care": args.care})


def _init_runtime(args: argparse.Namespace, output_dir: Path, dashboard_events: DashboardEventSink,
                  components: set[str] | None) -> tuple[list, dict[str, Any]]:
    theta = json.loads(args.theta_json) if getattr(args, "theta_json", None) else None
    prims = build_lab(sim_dir=args.sim_dir, model_path=args.model_path, render=bool(args.render), object_id=args.object,
                      theta=theta, seed=args.seed)
    return [], {"primitives": prims}


def finalize_run(context: RunFinalizationContext) -> Path:
    """Write the last pick_and_place outcome and its declared probabilities beside the run."""
    from robots.mujoco_lab.toolkit import LAST_PICK_ARTIFACT

    last_path = Path(context.output_dir) / LAST_PICK_ARTIFACT
    last = json.loads(last_path.read_text()) if last_path.exists() else None
    out = {
        "object": context.task_desc.get("object"),
        "destination": context.task_desc.get("destination"),
        "outcome": None if last is None else last.get("outcome"),
        "placed_at_requested": None if last is None else last.get("placed_at_requested"),
        "declared": None if last is None else last.get("declared"),
        "option": None if last is None else last.get("option"),
        "agent_error_present": context.agent_error is not None,
        "elapsed_s": round(context.elapsed_s, 1),
        "planner": {"backend": context.planner, "model": context.model, "reasoning_effort": context.reasoning_effort},
    }
    path = Path(context.output_dir) / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return path


MUJOCO_LAB_DASHBOARD_SPEC: DashboardSpec = {
    "task": {
        "command": "/rpent-task",
        "usage": "/rpent-task <object> <destination> <seed>",
        "fields": (
            {"name": "object"},
            {"name": "destination", "suggestions": ("table_centre", "table_edge", "table_gap", "mouse_pad", "plate", "box", "bin")},
            {"name": "seed", "kind": "integer", "minimum": 0},
        ),
        "display": "{object} -> {destination} / seed {seed}",
        "output_slug": "{object}_{destination}_s{seed}",
    },
    "runtime_components": ({"name": "env", "label": "LAB", "scope": "unique"},),
    "primitives": ("assess_pick", "pick_and_place", "view_env_state"),
}


def get_robot_spec() -> RobotSpec:
    return RobotSpec(
        name="mujoco_lab",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
        dashboard=MUJOCO_LAB_DASHBOARD_SPEC,
        finalize_run=finalize_run,
    )
