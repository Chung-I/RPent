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

"""Summarize LIBERO-Pro worker results per suite; optionally log to W&B.

Every expected cell counts in the denominator: a missing result or a final
infrastructure failure counts as unsuccessful and is listed separately.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from worker import SUITES, all_cells, cell_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out_root")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--tag", default="qwen38-xhigh")
    p.add_argument("--mode", choices=["eval", "explore"], default="eval")
    args = p.parse_args()
    root = Path(args.out_root)

    rows, missing = [], []
    expected = [c for c in all_cells() if args.mode == "eval" or c.endswith(":0")]
    for cell in expected:
        path = cell_dir(root, cell) / "result.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
        else:
            missing.append(cell)
    by_suite = defaultdict(Counter)
    for row in rows:
        by_suite[row["suite"]][row["outcome"]] += 1

    print(f"{'suite':22} {'success':>8} {'done':>5} {'rate%':>6}  outcomes")
    total_success = 0
    for suite in SUITES:
        counts = by_suite[suite]
        success = counts["success"]
        total_success += success
        per_suite = len(expected) // len(SUITES)
        print(
            f"{suite:22} {success:8} {sum(counts.values()):5} {100 * success / per_suite:6.1f}  {dict(counts)}"
        )
    print(
        f"{f'overall (of {len(expected)})':22} {total_success:8} {len(rows):5} {100 * total_success / len(expected):6.2f}"
    )
    retried = sorted((root / "_infra").glob("*/result.json"))
    print(f"missing={len(missing)} retried_infra_attempts={len(retried)}")
    for row in rows:
        if row["outcome"] in ("infra_failure", "config_error"):
            print(f"  {row['outcome']}: {row['cell']} {row.get('error')}")

    if args.wandb_project:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=f"{args.tag}-{args.mode}-summary",
            group=args.tag,
            job_type="summary",
        )
        table = wandb.Table(columns=sorted({k for r in rows for k in r}))
        for row in rows:
            table.add_data(*[row.get(c) for c in table.columns])
        run.log({"episodes": table})
        run.summary.update(
            {
                f"success_rate/{s}": 100
                * by_suite[s]["success"]
                / (len(expected) // len(SUITES))
                for s in SUITES
            }
            | {
                "success_rate/overall": 100 * total_success / len(expected),
                "episodes_done": len(rows),
            }
        )
        run.finish()


if __name__ == "__main__":
    main()
