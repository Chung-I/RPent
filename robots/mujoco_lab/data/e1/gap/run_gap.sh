#!/bin/bash
# run_gap.sh: the E1 oracle-gap check (rules only) in a memory-capped scope; writes logs/e1/gap/run.log and run.exit
set -u
OUT=/home/chungyili/Codes/RPent-e1/logs/e1/gap
cd /home/chungyili/Codes/RPent-e1
export PYTHONPATH=/home/chungyili/Codes/RPent-e1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl OMP_NUM_THREADS=2 WANDB_MODE=disabled
export RPENT_MUJOCO_SIM_DIR=/home/chungyili/Codes/daily-logs/researches/property-belief-manipulation/experiments/2026-10-27-vlm-grasp-limits/sim
systemd-run --user --scope -q -p MemoryMax=24G /home/chungyili/Codes/venvs/rpent-mujoco/bin/python -P -u -m robots.mujoco_lab.e1_gap run \
  --cases "$OUT/cases.json" --workers 8 --out "$OUT" > "$OUT/run.log" 2>&1 </dev/null
echo "EXIT $?" > "$OUT/run.exit"
