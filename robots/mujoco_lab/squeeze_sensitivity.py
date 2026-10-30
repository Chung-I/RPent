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


"""Offline squeeze sensitivity on probe-corpus latents; the simulator never runs.

Each cell changes sigma and the stiffness/damage-force correlation, replays k_hat from
(z, eps), and refits only the squeeze reading law on training objects. The kind and thrown
models, outcome model and temper stay fixed. Continuations use the recorded middle pinch
pose and the menu's rows to table_centre; the probe corpus records no other proposal poses.

Each case is valued with the bundle's estimator (``--estimator`` overrides it) and reports the reading's own value
beside the total (probe_value's reading_value: the saving of the whole result over its kind and survival). Per cell
(``cell_summary``, plan 2b2 decision 3): the means of both, each with its between-case error (the sd of the per-case
values over sqrt(cases), which includes the particle draws) and its Monte Carlo error conditional on these cases; the
plug-in's means on the same draws; and at rho 0 the upward-bias screen: the mean reading value is at most 2
between-case errors above 0.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np

from robots.mujoco_lab.fragility import Thresholds, map_latent
from robots.mujoco_lab.prior import fragility_world, make_prior, sha256_of
from robots.mujoco_lab.robot_spec import load_bundle
from robots.mujoco_lab.train_models import law_val_objects, load_rows
from rpent.skills.pick_place.costs import particle_costs
from rpent.skills.pick_place.evaluate import row_pose, rows_of
from rpent.skills.pick_place.features import model_names, particle_matrix
from rpent.skills.pick_place.likelihood import fit_law
from rpent.skills.pick_place.menu import ESTIMATORS, EVOI_K, _rows, estimate, probe_value
from rpent.skills.pick_place.probes import log_ratio, squeeze_cap


def recompute_readings(rows: list[dict[str, Any]], thresholds: Thresholds, sigma: float) -> list[dict[str, Any]]:
    """Replay squeeze readings at sigma from kept fragility z and sensor eps, without mutating rows.

    Changing rho changes k but leaves the damage force, the rigid motion and the kind unchanged.
    Missing or non-finite latents are errors: a sensitivity cell must use the original draws.
    """
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    out = []
    for r in rows:
        try:
            z = np.asarray(r["theta_frag"]["z"], dtype=float)
            eps = float(r["reading"]["eps"])
            if r["probe"] != "squeeze" or z.shape != (3,) or not np.isfinite(z).all() or not math.isfinite(eps):
                raise ValueError("invalid squeeze latents")
            frag = map_latent(z, r["entry"], thresholds)
            k = float(frag["k_n_per_mm"])
            reading = k * math.exp(sigma * eps)
            if not math.isfinite(reading) or reading <= 0:
                raise ValueError("invalid stiffness reading")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{r.get('key', '<row>')}: squeeze replay requires finite z, eps and a stiffness law") from exc
        out.append(dict(r, theta_frag=frag, reading={**r["reading"], "k_hat": reading}))
    return out


def cell_thresholds(thresholds: Thresholds, rho: str) -> Thresholds:
    """A copy of the final laws with every stiffness correlation set to rho (final keeps each law's)."""
    th = copy.deepcopy(thresholds)
    if rho != "final":
        value = float(rho)
        if not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("rho must be final or a finite number in [-1, 1]")
        for law in th.laws.values():
            if "rho" in law:
                law["rho"] = value
    return th


def reading_model(pm: Any, rows: list[dict[str, Any]], split: dict[str, list[str]], seed: int) -> Any:
    """Replace only the measured squeeze law, using the usual inner training-object law split."""
    train = rows_of(rows, split["train"])
    held = set(law_val_objects(split, seed))
    pairs = [(log_ratio(r["kind"], r["reading"], r["truth"]["mass"], r["truth"]["mu"], "squeeze",
                        r["theta_frag"]["k_n_per_mm"]), r["object"] in held)
             for r in train if r["kind"] == "measured"]
    pairs = [(float(x), h) for x, h in pairs if x is not None and np.isfinite(x)]
    if not pairs:
        raise ValueError("no measured training squeeze readings to refit")
    xs = np.array([x for x, _ in pairs])
    val = np.array([h for _, h in pairs], dtype=bool)
    law = fit_law(xs[~val], xs[val], seed)
    return replace(pm, reading={**pm.reading, "measured": law})


def case_value(row: dict[str, Any], bundle: dict[str, Any], prior: Any, pm: Any, n: int, seed: int,
               estimator: str = "plugin") -> dict[str, Any]:
    """The squeeze row's value against the menu continuations at the recorded middle pinch, with its reading value."""
    rng = np.random.default_rng(seed)
    pose = dict(row_pose(row), object_id=row["object"])
    belief = prior(pose, n, rng)
    prop = SimpleNamespace(s=float(row["s"]), point=np.asarray(pose["pt"]), yaw=float(pose["yaw"]))
    f_std = float(row["f_std"])
    costs = []
    for _, _, _, action, seconds, _ in _rows([prop], f_std, "table_centre", np.asarray(pose["pt"])[:2], damage=True):
        base = dict(row["base"], v_lift=action["v_lift"], v_transport=action["v_transport"], a_transport=action["a_transport"])
        _, per = estimate(bundle["model"], belief, base, pose, action["force"], [])
        costs.append(particle_costs(per, seconds, "normal", bundle["abstain_cost"]))
    force = squeeze_cap(belief, f_std)
    X = particle_matrix(row["base"], belief.arrays(), pose, force, model_names(pm))
    return probe_value(belief, costs, pm, X, force, "normal", rng, abstain_cost=bundle["abstain_cost"], estimator=estimator,
                       reading_value=True)


def cell_summary(values: list[dict[str, Any]], rho: str) -> dict[str, Any]:
    """One cell's summary of its per-case probe values (see the module docstring)."""
    n = len(values)

    def mean_se(key: str) -> tuple[float, float | None]:
        x = np.array([v[key] for v in values], dtype=float)
        return float(x.mean()), float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else None

    estimator = values[0]["estimator"]
    plugin = estimator == "plugin"
    reading, reading_se = mean_se("reading_gain")
    total, total_se = mean_se("evoi")
    screen = None
    if rho != "final" and float(rho) == 0 and reading_se is not None:
        screen = bool(reading <= 2.0 * reading_se)
    return {"estimator": estimator, "n_cases": n,
            "mean_reading_value": reading, "reading_between_case_se": reading_se,
            "reading_mc_se": float(np.sqrt(sum(v["reading_gain_se"] ** 2 for v in values)) / n),
            "mean_evoi": total, "evoi_between_case_se": total_se,
            "evoi_mc_se": float(np.sqrt(sum(v["evoi_se"] ** 2 for v in values)) / n),
            "plugin_mean_reading_value": reading if plugin else mean_se("plugin_reading_gain")[0],
            "plugin_mean_evoi": total if plugin else mean_se("plugin_evoi")[0],
            "reading_value_at_most_2_se_above_0": screen}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--probe-corpus", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--split", default="val", choices=("train", "val", "test"))
    ap.add_argument("--sigmas", default="0.05,0.10,0.20,0.40")
    ap.add_argument("--rhos", default="0,0.5,final")
    ap.add_argument("--n", type=int, default=2000, help="prior particles per case")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--estimator", choices=ESTIMATORS, default=None, help="the probe-value estimator (default: the bundle's)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.n < 2:
        ap.error("--n must be at least 2")
    try:
        sigmas = [float(s) for s in a.sigmas.split(",")]
        rhos = [s.strip() for s in a.rhos.split(",")]
        th = Thresholds.from_json(a.thresholds)
        laws = [(rho, cell_thresholds(th, rho)) for rho in rhos]
        if any(not math.isfinite(s) or s <= 0 for s in sigmas):
            raise ValueError("sigmas must be finite and positive")
    except ValueError as exc:
        ap.error(str(exc))
    bundle = load_bundle(a.bundle)
    world = fragility_world(bundle["manifest"])
    if world is None or "squeeze" not in bundle["probe_models"]:
        ap.error("sensitivity requires an m2b bundle with a squeeze model")
    if getattr(bundle["model"], "kind", "classifier") != "classifier":
        ap.error("sensitivity requires the classifier bundle")
    estimator = a.estimator or bundle["evoi"]
    metrics = json.load(open(os.path.join(a.bundle, bundle["manifest"]["metrics"])))
    split = metrics["split"]
    rows = [r for r in load_rows(a.probe_corpus) if r.get("probe") == "squeeze"]
    if not rows:
        ap.error("no trainable squeeze rows in the probe corpus")
    cells = []
    for rho, law in laws:
        prior = make_prior(world[0], law)
        for sigma in sigmas:
            replay = recompute_readings(rows, law, sigma)
            pm = reading_model(bundle["probe_models"]["squeeze"], replay, split, a.seed)
            validation = [r for r in rows_of(replay, split[a.split]) if r["entry"].get("class") == "crushable"]
            if not validation:
                ap.error(f"no crushable squeeze cases in split {a.split!r}")
            values = [case_value(r, bundle, prior, pm, a.n, a.seed + j, estimator) for j, r in enumerate(validation)]
            cells.append({"sigma": sigma, "rho": rho, "rho_by_law": {k: v["rho"] for k, v in law.laws.items() if "rho" in v},
                          **cell_summary(values, rho), "reading_law": pm.reading["measured"].to_json()})
    out = {"bundle": a.bundle, "probe_corpus": a.probe_corpus, "thresholds": {"path": a.thresholds, "sha256": sha256_of(a.thresholds)},
           "split": a.split, "n_particles": a.n, "evoi_samples_per_case": EVOI_K, "seed": a.seed, "estimator": estimator,
           "care": "normal", "abstain_cost": bundle["abstain_cost"], "continuations": "middle pinch to table_centre",
           "cells": cells}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
