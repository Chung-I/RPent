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


"""The Real2Sim fit as evidence for the v12 checks (arm port; daily-logs 2026-11-05-v12-masscom-spread): the identify
probe's rows (identify_cases) and the attempt's own fit (``readouts["joint"]``), each scored as gate R1 scored them.

The likelihood is the fit's Gaussian on each particle's (m, m x c) in link 7's frame, information form (``ident.loglik``),
with the floored covariance that ``adapter._joint_fit`` records. A particle's CoM goes into link 7's frame through the
fit's ``obj_l7`` (the object's frame at the window's first step). A row without a reading (any identify kind but
measured, an attempt without a fit) gives no evidence: a zero log-likelihood, so the belief keeps its prior.

A bundle built with the identify probe (m2b2_v12b on) brings its own identify rows and probe model; the checks take them
first (identify_source, identify_model), so a rerun on such a bundle changes only the bundle path.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

MEASURED = "measured"
TAG = "Bidentify"              # identify_cases's rows' key tag (option A's simulated rows), and a missing row's
BUNDLE_TAG = "Didentify"       # a bundle's own identify rows (m2b2_v12b on): probe_corpus scene D, posterior_check's set D
MISSING = "missing"            # a weigh case without an identify row (the run wrote none, or no pose: declined, error)
SCENE = ("pose", "s", "base")  # the weigh case's scene: the prior's input and the menus' pinch
DRAWS = ("seed", "theta", "com_end", "fill", "theta_frag")   # an identify row repeats its weigh case's


def fit_parts(fit: dict[str, Any], arrays: dict[str, np.ndarray]) -> np.ndarray:
    """(n, 4): each particle's (m, m cx, m cy, m cz), its CoM in link 7's frame through the fit's obj_l7."""
    p = np.asarray(fit["obj_l7"]["p"], dtype=float)
    R = np.asarray(fit["obj_l7"]["R"], dtype=float).reshape(3, 3)
    m = np.asarray(arrays["mass"], dtype=float)
    c = p[None, :] + np.asarray(arrays["com_obj"], dtype=float) @ R.T
    return np.column_stack([m, m[:, None] * c])


def fit_loglik(fit: dict[str, Any], arrays: dict[str, np.ndarray]) -> np.ndarray:
    """ident.loglik of the particles under one fit: -0.5 e' info e, e = parts - phi[:4]. A fit without the model-error
    floor (a reading made before the R1 decision) is refused."""
    if not fit.get("floor"):
        raise ValueError("the fit carries no model-error floor: adapter._joint_fit records one since gate R1")
    e = fit_parts(fit, arrays) - np.asarray(fit["phi"], dtype=float)[None, :4]
    return -0.5 * np.einsum("ni,ij,nj->n", e, np.asarray(fit["info"], dtype=float), e)


def attempt_fit_loglik(row: dict[str, Any], arrays: dict[str, np.ndarray]) -> np.ndarray:
    """The attempt's own fit (R1's attempt row); zeros for a row without one (the object never rose)."""
    fit = (row.get("readouts") or {}).get("joint")
    return fit_loglik(fit, arrays) if fit else np.zeros(len(arrays["mass"]))


class IdentifyFit:
    """The identify probe as a probe model (``loglik(obs, X, arrays, force)``, as the bundle's): the fit's likelihood for
    a measured reading, zeros otherwise. It reads no features: its callers pass X None."""

    def loglik(self, obs: dict[str, Any], X: Any, arrays: dict[str, np.ndarray], force: float, thrown: Any = None,
               **_: Any) -> tuple[np.ndarray, dict[str, Any]]:
        if obs.get("kind") != MEASURED or not obs.get("reading"):
            return np.zeros(len(arrays["mass"])), {"used": []}
        return fit_loglik(obs["reading"], arrays), {"used": ["fit"]}


def identify_rows(path: str) -> list[dict[str, Any]]:
    """identify_cases rows (one per scene-B weigh case)."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def bundle_identify_rows(bundle_dir: str) -> list[dict[str, Any]]:
    """The bundle's own identify rows: its probe corpus's rows tagged BUNDLE_TAG (scene D), declined ones among them (a
    bundle built with the identify probe, m2b2_v12b on); none for a bundle without them (m2b2_v12)."""
    from robots.mujoco_lab.train_models import repo_path

    man = json.load(open(os.path.join(bundle_dir, "manifest.json")))
    if not man.get("probe_corpus"):
        return []
    with open(repo_path(man["probe_corpus"])) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if str(r.get("key", "")).endswith("__" + BUNDLE_TAG)]


def require_bundle_generator(bundle_dir: str) -> str:
    """The lab generator of this run, refused when the bundle's manifest names another: the draws and the truths follow
    the generator, and RPENT_LAB_GENERATOR unset means v10."""
    from robots.mujoco_lab.corpus import lab_generator

    g = lab_generator()
    want = json.load(open(os.path.join(bundle_dir, "manifest.json"))).get("generator")
    if want and want != g:
        raise SystemExit(f"{bundle_dir} was built under generator {want}, this run is under {g} (RPENT_LAB_GENERATOR)")
    return g


def identify_source(bundle_dir: str, explicit: str | None) -> tuple[list[dict[str, Any]], str]:
    """A stage's identify rows and their source: the bundle's own when it has them, so a rerun on a bundle built with
    the identify probe is a bundle-path change only (``explicit`` is then not read); else identify_cases's rows at
    ``explicit`` (the simulated cases, option A)."""
    own = bundle_identify_rows(bundle_dir)
    if own:
        return own, "bundle"
    if not explicit:
        raise ValueError(f"the identify stage needs identify rows: {bundle_dir}'s probe corpus has none (--identify)")
    return identify_rows(explicit), explicit


def identify_model(B: dict[str, Any]) -> Any:
    """The bundle's identify probe model when it has one (m2b2_v12b on), else gate R1's fit likelihood."""
    return (B.get("probe_models") or {}).get("identify") or IdentifyFit()


def own_scene(r: dict[str, Any]) -> dict[str, Any]:
    """An identify case's row at its own pinch (SCENE from the ``_identify`` keys), for a probe model that reads
    features; any other row as it is."""
    return {**r, **{k: r[f"{k}_identify"] for k in SCENE if f"{k}_identify" in r}}


def identify_cases(weigh_cases: list[list[dict[str, Any]]], rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """One identify case per weigh case, in the weigh cases' order, so a stage on the weigh's stream draws the weigh's
    particles: the identify row of the same object and draw with the weigh row's scene (SCENE: the prior's input and the
    menus' pinch; the row's own stay as ``pose_identify``, ``s_identify`` and ``base_identify``), or, when the run has no
    usable row, a ``missing`` row without a reading. A row whose draws (DRAWS) are not its weigh case's is refused."""
    by = {(r["object"], int(r["draw"])): r for r in rows}
    out = []
    for case in weigh_cases:
        w = case[0]
        r = by.get((w["object"], int(w["draw"])))
        if r is not None and any(k in r and r[k] != w.get(k) for k in DRAWS):
            raise ValueError(f"identify row {r.get('key')} has other draws than its weigh case {w['key']}")
        if r is None or "pose" not in r:
            why = None if r is None else (r.get("declined") or r.get("error"))
            out.append([{**w, "key": w["key"].rsplit("__", 1)[0] + "__Bidentify", "probe": "identify", "kind": MISSING,
                         "reading": None, "missing_reason": why}])
        else:
            out.append([{**r, **{k: w[k] for k in SCENE}, **{f"{k}_identify": r.get(k) for k in SCENE}}])
    return out
