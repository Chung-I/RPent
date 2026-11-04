#!/bin/bash
# run_offline.sh: the mass/CoM steps that need no model server, one after another in memory-capped scopes:
# Part A under the class and the generic prior (m2b2 5-fold cross-fit), then the first menus (class + generic).
# Each step writes OUT/run.log, OUT/run.exit and its summary (wandb); logs/masscom/all.done at the end.
set -u
cd /home/chungyili/Codes/RPent-masscom
export PYTHONPATH=/home/chungyili/Codes/RPent-masscom PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl OMP_NUM_THREADS=2
export RPENT_MUJOCO_SIM_DIR=/home/chungyili/Codes/daily-logs/researches/property-belief-manipulation/experiments/2026-10-27-vlm-grasp-limits/sim
export WANDB_DIR=/home/chungyili/Codes/RPent-masscom/logs/wandb
L=/home/chungyili/Codes/RPent-masscom/logs/masscom
SC=/home/chungyili/Codes/daily-logs/researches/property-belief-manipulation/experiments/2026-11-03-mass-com-estimation/scenes.json
PY="systemd-run --user --scope -q -p MemoryMax=24G /home/chungyili/Codes/venvs/rpent-mujoco/bin/python -P -u -m"
B=robots/mujoco_lab/data/m2b2
mkdir -p "$L/partA_class" "$L/partA_generic" "$L/menus_generic" "$WANDB_DIR"
for p in class generic; do
  O=$L/partA_$p
  $PY robots.mujoco_lab.estimation_check run --bundle $B --crossfit --prior $p --scenes $SC --workers 5 --out $O > $O/run.log 2>&1 </dev/null
  echo "EXIT $?" > $O/run.exit
  $PY robots.mujoco_lab.estimation_check summary --dir $O --wandb > $O/summary.log 2>&1 </dev/null
done
O=$L/menus_generic
$PY robots.mujoco_lab.first_menus run --bundle $B --scenes $SC --workers 8 --out $O > $O/run.log 2>&1 </dev/null
echo "EXIT $?" > $O/run.exit
$PY robots.mujoco_lab.first_menus summary --dir $O --wandb > $O/summary.log 2>&1 </dev/null
echo done > $L/all.done
