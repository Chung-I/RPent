#!/bin/bash
# run_neutral.sh: the post-hoc neutral-task-text check (menu and rec planners): the tuning split, then the final split.
# Each split runs in a memory-capped scope and writes OUTDIR/run.log and OUTDIR/run.exit.
set -u
cd /home/chungyili/Codes/RPent-e1
export PYTHONPATH=/home/chungyili/Codes/RPent-e1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl OMP_NUM_THREADS=2 WANDB_MODE=disabled
export RPENT_MUJOCO_SIM_DIR=/home/chungyili/Codes/daily-logs/researches/property-belief-manipulation/experiments/2026-10-27-vlm-grasp-limits/sim
for pair in "tune /home/chungyili/Codes/RPent-e1/logs/e1/tune_neutral" "final /home/chungyili/Codes/RPent-e1/logs/e1/final_neutral"; do
  set -- $pair; SPLIT=$1; OUT=$2
  mkdir -p "$OUT"
  systemd-run --user --scope -q -p MemoryMax=24G /home/chungyili/Codes/venvs/rpent-mujoco/bin/python -P -u -m robots.mujoco_lab.e1 run \
    --plan logs/e1/plan.json --split "$SPLIT" --arms menu_planner,rec_planner --task neutral --workers 8 \
    --base-url http://127.0.0.1:18691/v1 --out "$OUT" > "$OUT/run.log" 2>&1 </dev/null
  echo "EXIT $?" > "$OUT/run.exit"
done
