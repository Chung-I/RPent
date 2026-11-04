#!/bin/bash
# run_e_gap80.sh (2026-11-04): item 1, E (the receding-horizon reference planner, no LLM) at H = 1, 2 and 3 on the gap
# check's 80 seed-13 cases, settings k100_A3_abs, k30_A3_abs and k100_A10_abs, bundle m2b2_v11c (5-fold cross-fit),
# generator v11; then the summary (wandb). 8 workers on the local box. Refuses an existing output folder; writes
# OUT/run.log, run.exit, summary.log, summary.exit and OUT/all.done at the end; a failed step stops the script.
set -u
R=/home/chungyili/Codes/RPent-ladder
O=$R/logs/ladder_e/gap80_v11c
cd "$R" || exit 1
export PYTHONPATH=$R PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl OMP_NUM_THREADS=2 WANDB_MODE=online WANDB_DIR=$R/logs/wandb
export RPENT_MUJOCO_SIM_DIR=/home/chungyili/Codes/daily-logs/researches/property-belief-manipulation/experiments/2026-10-27-vlm-grasp-limits/sim
export RPENT_LAB_GENERATOR=v11
PY="systemd-run --user --scope -q -p MemoryMax=32G /home/chungyili/Codes/venvs/rpent-mujoco/bin/python -P -u -m"
if [ -e "$O" ]; then echo "refusing: $O exists"; exit 1; fi
mkdir -p "$O" "$R/logs/wandb"
echo "branch $(git -C "$R" rev-parse HEAD), dirty files $(git --no-optional-locks -C "$R" status --porcelain --untracked-files=no | wc -l)" > "$O/provenance.txt"
$PY robots.mujoco_lab.ladder_e run --cases robots/mujoco_lab/data/e1/gap/cases.json --crossfit robots/mujoco_lab/data/m2b2_v11c/folds \
  --horizons 1,2,3 --workers 8 --out "$O" > "$O/run.log" 2>&1 </dev/null
rc=$?; echo "EXIT $rc" > "$O/run.exit"; [ "$rc" = 0 ] || exit "$rc"
$PY robots.mujoco_lab.ladder_e summary --dir "$O" --wandb > "$O/summary.log" 2>&1 </dev/null
rc=$?; echo "EXIT $rc" > "$O/summary.exit"; [ "$rc" = 0 ] || exit "$rc"
echo done > "$O/all.done"
