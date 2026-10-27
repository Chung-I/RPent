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
objects in oracle and belief mode, and record the split and the chosen model kind.

    python -m robots.mujoco_lab.train_models --corpus data/corpus.jsonl --out-dir data [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from rpent.skills.pick_place.evaluate import rows_of, split_objects
from rpent.skills.pick_place.hazard_model import train_hazard
from rpent.skills.pick_place.success_model import train


def load_rows(path: str) -> list[dict]:
    rows = []
    for line in open(path):
        r = json.loads(line)
        if "error" not in r:
            rows.append(r)
    return rows


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-belief", type=int, default=32)
    a = ap.parse_args(argv)
    rows = load_rows(a.corpus)
    split = split_objects(rows, seed=a.seed)
    tr, va = rows_of(rows, split["train"]), rows_of(rows, split["val"])
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    clf = train(tr, va, os.path.join(a.out_dir, "success_model.joblib"), seed=a.seed)
    t1 = time.time()
    haz = train_hazard(tr, va, os.path.join(a.out_dir, "hazard_model.joblib"), seed=a.seed, n_belief=a.n_belief)
    t2 = time.time()
    kind = "hazard" if haz["belief"]["brier"] <= clf["belief"]["brier"] else "classifier"
    out = {
        "corpus": a.corpus, "n_rows": len(rows), "split": split, "n_train": len(tr), "n_val": len(va),
        "classifier": clf, "hazard": haz, "kind": kind,
        "acceptance_val_belief": {
            "classifier": {"ece": clf["belief"]["ece"], "brier": clf["belief"]["brier"], "brier_const": clf["belief"]["brier_const"]},
            "hazard": {"ece": haz["belief"]["ece"], "brier": haz["belief"]["brier"], "brier_const": haz["belief"]["brier_const"]},
        },
        "seconds": {"classifier": round(t1 - t0, 1), "hazard": round(t2 - t1, 1)},
    }
    with open(os.path.join(a.out_dir, "success_model_metrics.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"kind": kind, "acceptance_val_belief": out["acceptance_val_belief"],
                      "oracle": {"classifier": clf["oracle"]["brier"], "hazard": haz["oracle"]["brier"]},
                      "unsupported": clf["belief"]["unsupported_classes"], "seconds": out["seconds"]}, indent=1))
    return out


if __name__ == "__main__":
    main()
    sys.exit(0)
