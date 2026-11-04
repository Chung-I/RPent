#!/bin/bash
# run_e_gap80_resume.sh (2026-11-04): resume run_e_gap80.sh's E run in the same folder (ladder_e run skips every job that
# has all its horizon rows). The first launch (04:43) was stopped at 04:52 with 11 of 240 jobs done, to free the box for
# the arm-port builder; those 11 jobs ran beside an 8-worker job, so their wall times are high. Refuses to run unless the
# folder holds that first launch and HEAD is the same commit with no dirty tracked file. Then the summary (wandb).
# The first resume (05:00, 8 workers, MemoryMax=16G, swap not capped) died at 05:09:32 with EXIT 137 after 72 rows:
# systemd-oomd killed the user manager (memory pressure in user@1000.service above 50 % for 20 s), and with it every
# user scope. Since then each unit gets a cap and no swap (the paper session's budget: E at most 9 GB in one unit,
# workers = floor(9 GB / the measured peak of one worker), at most 6), so the kernel kills inside the unit instead.
# A worker's memory grows by about 0.07 GB per job (15 jobs in one process: peak 1.05 to 2.16 GB), so ladder-pilot
# 8c3cfa1 replaces a worker after E_MAX_TASKS jobs (a job's results do not depend on the worker's earlier jobs: 24 of 24
# rows equal). A HEAD other than the first launch's commit must be declared (E_CODE=<HEAD>), descend from it, and change
# only ladder_e.py and its test; the provenance line records it.
# Settings: E_WORKERS (default 5), E_MEMMAX (default 9G), E_MAX_TASKS (default 4), E_CODE.
set -u
W=${E_WORKERS:-5}
CAP=${E_MEMMAX:-9G}
MT=${E_MAX_TASKS:-4}
R=/home/chungyili/Codes/RPent-ladder
O=$R/logs/ladder_e/gap80_v11c
cd "$R" || exit 1
export PYTHONPATH=$R PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl OMP_NUM_THREADS=2 WANDB_MODE=online WANDB_DIR=$R/logs/wandb
export RPENT_MUJOCO_SIM_DIR=/home/chungyili/Codes/daily-logs/researches/property-belief-manipulation/experiments/2026-10-27-vlm-grasp-limits/sim
export RPENT_LAB_GENERATOR=v11
PY="systemd-run --user --scope -q -p MemoryMax=$CAP -p MemorySwapMax=0 /home/chungyili/Codes/venvs/rpent-mujoco/bin/python -P -u -m"
systemd-run --user --scope -q -p MemoryMax=64M -p MemorySwapMax=0 true || { echo "refusing: no user manager (systemd-run --user failed)"; exit 1; }
[ -s "$O/rows.jsonl" ] && [ -e "$O/provenance.txt" ] || { echo "refusing: no first launch in $O"; exit 1; }
[ ! -e "$O/all.done" ] || { echo "refusing: $O is complete"; exit 1; }
first=$(head -1 "$O/provenance.txt" | sed -n 's/^branch \([0-9a-f]*\), dirty files 0$/\1/p')
[ -n "$first" ] || { echo "refusing: provenance line 1 is not a clean commit"; exit 1; }
head_sha=$(git -C "$R" rev-parse HEAD)
dirty=$(git --no-optional-locks -C "$R" status --porcelain --untracked-files=no | wc -l)
[ "$dirty" = 0 ] || { echo "refusing: $dirty dirty tracked files"; exit 1; }
changed="none"
if [ "$head_sha" != "$first" ]; then
  [ "${E_CODE:-}" = "$head_sha" ] || { echo "refusing: code changed since $first (HEAD $head_sha); declare it with E_CODE=<HEAD>"; exit 1; }
  git -C "$R" merge-base --is-ancestor "$first" "$head_sha" || { echo "refusing: HEAD does not descend from $first"; exit 1; }
  other=$(git --no-optional-locks -C "$R" diff --name-only "$first" "$head_sha" | grep -v -x -e robots/mujoco_lab/ladder_e.py -e tests/unit_tests/robots/mujoco_lab/test_ladder_e.py)
  [ -z "$other" ] || { echo "refusing: other files changed since $first: $other"; exit 1; }
  changed="$(git --no-optional-locks -C "$R" diff --shortstat "$first" "$head_sha" | sed 's/^ *//') (ladder_e.py and its test)"
fi
now="branch $head_sha, dirty files 0"
echo "resume $(date +%Y-%m-%dT%H:%M:%S): $now, rows before $(wc -l < "$O/rows.jsonl"), workers $W, max tasks per worker $MT, MemoryMax $CAP, MemorySwapMax 0, changed since $first: $changed" >> "$O/provenance.txt"
ts=$(date +%Y%m%dT%H%M%S); mv "$O/run.log" "$O/run_until_$ts.log"; mv "$O/run.exit" "$O/run_until_$ts.exit"   # keep each launch's log
$PY robots.mujoco_lab.ladder_e run --cases robots/mujoco_lab/data/e1/gap/cases.json --crossfit robots/mujoco_lab/data/m2b2_v11c/folds \
  --horizons 1,2,3 --workers "$W" --max-tasks-per-worker "$MT" --out "$O" > "$O/run.log" 2>&1 </dev/null
rc=$?; echo "EXIT $rc" > "$O/run.exit"; [ "$rc" = 0 ] || exit "$rc"
$PY robots.mujoco_lab.ladder_e summary --dir "$O" --wandb > "$O/summary.log" 2>&1 </dev/null
rc=$?; echo "EXIT $rc" > "$O/summary.exit"; [ "$rc" = 0 ] || exit "$rc"
echo done > "$O/all.done"
