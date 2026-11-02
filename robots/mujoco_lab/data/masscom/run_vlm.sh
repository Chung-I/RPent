#!/bin/bash
# run_vlm.sh: the mass/CoM steps with the VLM priors, after both models' queries are in, one after another in
# memory-capped scopes: Part A under each VLM prior (m2b2 5-fold cross-fit), the first menus (class, generic and both
# VLM priors at six settings: the four of menus_generic plus the E1 candidates k30_A10 and k100_A30), and the
# comparison. Each step writes OUT/run.log, OUT/run.exit and its summary (wandb); logs/masscom/vlm.done at the end.
set -u
cd /home/chungyili/Codes/RPent-masscom
export PYTHONPATH=/home/chungyili/Codes/RPent-masscom PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl OMP_NUM_THREADS=2
export RPENT_MUJOCO_SIM_DIR=/home/chungyili/Codes/daily-logs/researches/property-belief-manipulation/experiments/2026-10-27-vlm-grasp-limits/sim
export WANDB_DIR=/home/chungyili/Codes/RPent-masscom/logs/wandb
L=/home/chungyili/Codes/RPent-masscom/logs/masscom
X=/home/chungyili/Codes/daily-logs/researches/property-belief-manipulation/experiments/2026-11-03-mass-com-estimation
SC=$X/scenes.json
Q=$X/out/qwen38_27b.jsonl
G=$X/out/gemma4_31b.jsonl
PY="systemd-run --user --scope -q -p MemoryMax=24G /home/chungyili/Codes/venvs/rpent-mujoco/bin/python -P -u -m"
B=robots/mujoco_lab/data/m2b2
for m in qwen38_27b=$Q gemma4_31b=$G; do
  O=$L/partA_${m%%=*}
  mkdir -p "$O"
  $PY robots.mujoco_lab.estimation_check run --bundle $B --crossfit --prior vlm --scenes $SC --vlm ${m#*=} --workers 5 --out $O > $O/run.log 2>&1 </dev/null
  echo "EXIT $?" > $O/run.exit
  $PY robots.mujoco_lab.estimation_check summary --dir $O --wandb > $O/summary.log 2>&1 </dev/null
done
O=$L/menus_all
mkdir -p "$O"
$PY robots.mujoco_lab.first_menus run --bundle $B --scenes $SC --vlm qwen38_27b=$Q --vlm gemma4_31b=$G \
  --settings k10_A10:10:10,k100_A3:100:3,k100_A10:100:10,k30_A3:30:3,k30_A10:30:10,k100_A30:100:30 \
  --workers 8 --out $O > $O/run.log 2>&1 </dev/null
echo "EXIT $?" > $O/run.exit
$PY robots.mujoco_lab.first_menus summary --dir $O --wandb > $O/summary.log 2>&1 </dev/null
$PY robots.mujoco_lab.masscom_report --part-a class=$L/partA_class --part-a generic=$L/partA_generic \
  --part-a qwen38_27b=$L/partA_qwen38_27b --part-a gemma4_31b=$L/partA_gemma4_31b --menus $O \
  --out $L/compare_all --wandb > $L/compare_all.log 2>&1 </dev/null
echo "EXIT $?" > $L/compare_all.exit
echo done > $L/vlm.done
