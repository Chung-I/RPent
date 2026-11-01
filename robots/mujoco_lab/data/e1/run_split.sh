#!/bin/bash
# run_split.sh SPLIT OUTDIR [ARMS]: one e1 run in a memory-capped scope; writes OUTDIR/run.log and OUTDIR/run.exit
set -u
SPLIT=$1; OUT=$2; ARMS=${3:-default_rule,oracle_rule,menu_planner,rec_planner,nomenu_planner}
cd /home/chungyili/Codes/RPent-e1
export PYTHONPATH=/home/chungyili/Codes/RPent-e1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl OMP_NUM_THREADS=2 WANDB_MODE=disabled
export RPENT_MUJOCO_SIM_DIR=/home/chungyili/Codes/daily-logs/researches/property-belief-manipulation/experiments/2026-10-27-vlm-grasp-limits/sim
mkdir -p "$OUT"
systemd-run --user --scope -q -p MemoryMax=24G /home/chungyili/Codes/venvs/rpent-mujoco/bin/python -P -u -m robots.mujoco_lab.e1 run \
  --plan logs/e1/plan.json --split "$SPLIT" --arms "$ARMS" --workers 8 --base-url http://127.0.0.1:18691/v1 --out "$OUT" \
  > "$OUT/run.log" 2>&1 </dev/null
echo "EXIT $?" > "$OUT/run.exit"
