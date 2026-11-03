#!/bin/bash
# run_expected.sh: the E1 expected-cost rescoring (no new decision, no model call) in a memory-capped scope: the gap check,
# then the pilot; writes logs/e1/{gap,pilot}_expected/run.log and run.exit
set -u
cd /home/chungyili/Codes/RPent-e1
export PYTHONPATH=/home/chungyili/Codes/RPent-e1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl OMP_NUM_THREADS=2 WANDB_MODE=disabled
export RPENT_MUJOCO_SIM_DIR=/home/chungyili/Codes/daily-logs/researches/property-belief-manipulation/experiments/2026-10-27-vlm-grasp-limits/sim
D=/home/chungyili/Codes/RPent-e1/robots/mujoco_lab/data/e1
L=/home/chungyili/Codes/RPent-e1/logs/e1
PY="systemd-run --user --scope -q -p MemoryMax=24G /home/chungyili/Codes/venvs/rpent-mujoco/bin/python -P -u -m robots.mujoco_lab.e1_expected"
mkdir -p "$L/gap_expected" "$L/pilot_expected"
$PY gap --src "$D/gap" --workers 8 --out "$L/gap_expected" > "$L/gap_expected/run.log" 2>&1 </dev/null
echo "EXIT $?" > "$L/gap_expected/run.exit"
$PY pilot --final "$D/final" --neutral "$D/neutral/final" --plan "$D/plan.json" --workers 8 --out "$L/pilot_expected" \
  > "$L/pilot_expected/run.log" 2>&1 </dev/null
echo "EXIT $?" > "$L/pilot_expected/run.exit"
