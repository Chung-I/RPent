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


"""Experiment tracking: one wandb run per job (training, posterior check, calibration).

Off when ``WANDB_MODE=disabled`` (the tests) or when wandb is missing; a tracking failure never
stops the job. Long jobs run with ``python -P`` so the repo's ``wandb/`` run folder does not shadow
the package, and ``WANDB_DIR`` points at ``logs/wandb``.
"""

from __future__ import annotations

import os
import sys
from typing import Any

PROJECT = "rpent-pick-place"


def _scalars(d: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, bool):
            out[key] = float(v)
        elif isinstance(v, (int, float)):
            out[key] = float(v)
        elif isinstance(v, dict):
            out.update(_scalars(v, key + "/"))
    return out


def log_run(job: str, config: dict[str, Any], metrics: dict[str, Any]) -> str | None:
    """Log one run's config and scalar metrics; returns the run URL, or None when tracking is off."""
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        return None
    try:
        run = wandb.init(project=PROJECT, job_type=job, config=config,
                         dir=os.environ.get("WANDB_DIR") or None)
        run.summary.update(_scalars(metrics))
        url = run.url
        run.finish()
        return url
    except Exception as exc:  # noqa: BLE001  tracking must not stop the job
        print(f"wandb logging failed: {exc}", file=sys.stderr)
        return None
