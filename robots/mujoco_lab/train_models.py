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

"""Train the classifier and the hazard model on a corpus, score both on the validation
objects in oracle and belief mode, and write a model bundle (models, metrics, manifest).

    python -P -m robots.mujoco_lab.train_models --corpus data/corpus_v2.jsonl data/corpus_fixtures_v2.jsonl \
        --out-dir robots/mujoco_lab/data/m2 [--features true] [--seed 0]

``--features true`` (the default) trains on the true-CoM features of corpus v2 rows and scores
belief mode with the lab prior (the corpus generator); ``--features nominal`` is milestone 1's
recipe (nominal-CoM features, the parametric prior). When the training rows carry readouts, the
readout model (likelihood.ReadoutModel) is fitted on them and written into the bundle; a retrain
clears the bundle's tempers (posterior_check chooses them again).

``--probe-corpus PATH`` (with or without ``--corpus``) fits the probe models (probes.ProbeModel) on
the probe rows of the bundle's training objects, scores each kind classifier on the validation
objects (Brier against the training kind rates), and writes the models, ``probe_metrics.json`` and
the manifest's ``probe_models``; without ``--corpus`` the split is the bundle's.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from robots.mujoco_lab import tracking
from rpent.skills.pick_place.evaluate import rows_of, split_objects
from rpent.skills.pick_place.hazard_model import train_hazard
from rpent.skills.pick_place.success_model import train

MANIFEST = "manifest.json"
METRICS = "success_model_metrics.json"
READOUT_MODEL = "readout_model.json"
READOUT_ROT = "readout_rot.joblib"
PROBE_METRICS = "probe_metrics.json"
PROBE_MODEL = "probe_{}.joblib"


def load_rows(path: str) -> list[dict]:
    rows = []
    for line in open(path):
        r = json.loads(line)
        if "error" not in r:
            rows.append(r)
    return rows


def write_manifest(out_dir: str, **fields) -> dict:
    """Create or update the bundle's manifest (paths are relative to the bundle directory)."""
    path = os.path.join(out_dir, MANIFEST)
    man = json.load(open(path)) if os.path.exists(path) else {"version": 2}
    man.update(fields)
    with open(path, "w") as f:
        json.dump(man, f, indent=1)
    return man


def fit_probes(path: str, out_dir: str, split: dict, seed: int = 0) -> dict:
    """Probe models on the training objects' probe rows; per probe the kind rates, the thrown rate, the
    reading laws and the validation Brier of the kind classifier against the training kind rates."""
    from rpent.skills.pick_place.probes import fit_probe_models, true_matrix

    rows = load_rows(path)
    tr, va = rows_of(rows, split["train"]), rows_of(rows, split["val"])
    models = fit_probe_models(tr, seed=seed)
    out: dict = {"probe_corpus": path, "split": split, "probes": {}}
    for probe, pm in models.items():
        every = [r for r in rows if r["probe"] == probe]
        tr_p, va_p = [r for r in tr if r["probe"] == probe], [r for r in va if r["probe"] == probe]
        const = np.array([sum(r["kind"] == k for r in tr_p) / len(tr_p) for k in pm.classes])
        val: dict = {"n": len(va_p), "brier": None, "brier_const": None}
        if va_p:
            Y = np.array([[float(r["kind"] == k) for k in pm.classes] for r in va_p])
            P = pm.p_kind(true_matrix(va_p))
            val.update(brier=float(np.mean(np.sum((P - Y) ** 2, axis=1))), brier_const=float(np.mean(np.sum((const - Y) ** 2, axis=1))))
        out["probes"][probe] = {"n": len(every), "n_train": len(tr_p), "kind_rates": {k: sum(r["kind"] == k for r in every) / len(every) for k in pm.classes},
                                "thrown_rate": sum(bool(r["thrown"]) for r in every) / len(every),
                                "reading": {k: list(v) for k, v in pm.reading.items()}, "stats": pm.stats, "val": val}
        pm.save(os.path.join(out_dir, PROBE_MODEL.format(probe)))
    with open(os.path.join(out_dir, PROBE_METRICS), "w") as f:
        json.dump(out, f, indent=1)
    man_path = os.path.join(out_dir, MANIFEST)
    tempers = (json.load(open(man_path)).get("tempers") or {}) if os.path.exists(man_path) else {}
    write_manifest(out_dir, probe_models={p: PROBE_MODEL.format(p) for p in models}, probe_metrics=PROBE_METRICS, probe_corpus=path,
                   tempers={k: v for k, v in tempers.items() if k != "probe"})      # new probe models: their temper is re-chosen
    out["wandb"] = tracking.log_run("train_probe_models", {"probe_corpus": path, "seed": seed, "out_dir": out_dir},
                                    {p: {"val": m["val"], "kind_rates": m["kind_rates"], "thrown_rate": m["thrown_rate"]}
                                     for p, m in out["probes"].items()})
    print(json.dumps({p: {"n": m["n"], "kind_rates": m["kind_rates"], "thrown_rate": m["thrown_rate"], "reading": m["reading"], "val": m["val"]}
                      for p, m in out["probes"].items()}, indent=1))
    return out


def bundle_split(out_dir: str) -> dict:
    """The object split of an existing bundle (its metrics file)."""
    man_path = os.path.join(out_dir, MANIFEST)
    if not os.path.exists(man_path):
        raise SystemExit(f"{out_dir} has no manifest: train the bundle with --corpus first (the probe fit uses its split)")
    man = json.load(open(man_path))
    return json.load(open(os.path.join(out_dir, man["metrics"])))["split"]


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", default=None, help="one or more corpus JSONL files (rows are pooled)")
    ap.add_argument("--probe-corpus", default=None, help="a probe corpus JSONL: fit the probe models into the bundle")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-belief", type=int, default=32)
    ap.add_argument("--features", choices=("true", "nominal"), default="true")
    a = ap.parse_args(argv)
    if not a.corpus and not a.probe_corpus:
        ap.error("give --corpus, --probe-corpus, or both")
    out = train_bundle(a) if a.corpus else {"split": bundle_split(a.out_dir)}
    if a.probe_corpus:
        out["probes"] = fit_probes(a.probe_corpus, a.out_dir, out["split"], a.seed)
    return out


def train_bundle(a: argparse.Namespace) -> dict:
    """The classifier, the hazard model and the readout model on ``a.corpus``; metrics and manifest."""
    rows = [r for path in a.corpus for r in load_rows(path)]
    if a.features == "true":
        missing = sum(1 for r in rows if "pose" not in r or "com_obj_true" not in r)
        if missing:
            raise SystemExit(f"{missing} rows have no pose or true CoM: --features true needs the corpus v2 fields "
                             "(regenerate the corpus, or pass --features nominal)")
    prior_fn, prior = (None, "m1_parametric")
    if a.features == "true":
        from robots.mujoco_lab.prior import lab_prior

        prior_fn, prior = lab_prior, "lab_generator"
    split = split_objects(rows, seed=a.seed)
    tr, va = rows_of(rows, split["train"]), rows_of(rows, split["val"])
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    clf = train(tr, va, os.path.join(a.out_dir, "success_model.joblib"), seed=a.seed, prior_fn=prior_fn, features=a.features)
    t1 = time.time()
    haz = train_hazard(tr, va, os.path.join(a.out_dir, "hazard_model.joblib"), seed=a.seed, n_belief=a.n_belief,
                       prior_fn=prior_fn, features=a.features)
    t2 = time.time()
    readout = None
    if any(r.get("readouts") for r in tr):
        from rpent.skills.pick_place.likelihood import ReadoutModel

        rm = ReadoutModel.fit(tr, seed=a.seed)
        rm.save(os.path.join(a.out_dir, READOUT_MODEL), os.path.join(a.out_dir, READOUT_ROT))
        readout = {"terms": rm.to_json()["terms"], "side_error": rm.side_error, "stats": rm.stats}
    t3 = time.time()
    kind = "hazard" if haz["belief"]["brier"] <= clf["belief"]["brier"] else "classifier"
    out = {
        "corpus": a.corpus if len(a.corpus) > 1 else a.corpus[0], "n_rows": len(rows), "split": split, "n_train": len(tr), "n_val": len(va),
        "features": a.features, "prior": prior, "classifier": clf, "hazard": haz, "kind": kind,
        "acceptance_val_belief": {
            "classifier": {"ece": clf["belief"]["ece"], "brier": clf["belief"]["brier"], "brier_const": clf["belief"]["brier_const"]},
            "hazard": {"ece": haz["belief"]["ece"], "brier": haz["belief"]["brier"], "brier_const": haz["belief"]["brier_const"]},
        },
        "readout": readout,
        "seconds": {"classifier": round(t1 - t0, 1), "hazard": round(t2 - t1, 1), "readout": round(t3 - t2, 1)},
    }
    with open(os.path.join(a.out_dir, METRICS), "w") as f:
        json.dump(out, f, indent=1)
    write_manifest(a.out_dir, features=a.features, prior=prior, kind=kind, success_model="success_model.joblib",
                   hazard_model="hazard_model.joblib", metrics=METRICS, corpus=out["corpus"], split_seed=a.seed,
                   readout_model=READOUT_MODEL if readout else None, tempers={})      # new models: tempers are re-chosen
    out["wandb"] = tracking.log_run("train_models", {"corpus": out["corpus"], "features": a.features, "prior": prior,
                                                     "seed": a.seed, "n_belief": a.n_belief, "out_dir": a.out_dir},
                                    {"acceptance_val_belief": out["acceptance_val_belief"], "kind_hazard": kind == "hazard",
                                     "seconds": out["seconds"],
                                     "classifier_by_subgroup": {k: v for k, v in clf["belief"]["by_subgroup"].items() if isinstance(v, dict)}})
    print(json.dumps({"kind": kind, "features": a.features, "acceptance_val_belief": out["acceptance_val_belief"],
                      "oracle": {"classifier": clf["oracle"]["brier"], "hazard": haz["oracle"]["brier"]},
                      "unsupported": clf["belief"]["unsupported_classes"], "seconds": out["seconds"]}, indent=1))
    return out


if __name__ == "__main__":
    main()
    sys.exit(0)
