#!/bin/bash
# run_masscom_v11c.sh: the mass/CoM check under generator v11 with the bundle m2b2_v11c (the v11c classes; its 5-fold cross-fit), one step
# after another in memory-capped scopes: Part A under the class, generic, Qwen and Gemma priors; the first menus (all
# four priors at menus_all's six settings); the comparison. Each step writes OUT/run.log, OUT/run.exit and its summary
# (wandb); a failed step stops the script. logs/masscom_v11c/all.done at the end. Refuses an existing output folder and
# a regeneration that has not ended with EXIT 0. The bundle is a symlink into RPent-real, so bundle_provenance.txt
# records its real path, the regeneration's code, exit and corpora, the simulator's commit and the sha256 of the
# bundle's manifests and folds.
set -u
R=/home/chungyili/Codes/RPent-masscom-v11
L=$R/logs/masscom_v11c
G_LOG=/home/chungyili/Codes/RPent-real/logs/m2b2_v11c/final
cd "$R" || exit 1
export PYTHONPATH=$R PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl OMP_NUM_THREADS=2 WANDB_MODE=online WANDB_DIR=$R/logs/wandb
export RPENT_MUJOCO_SIM_DIR=/home/chungyili/Codes/daily-logs/researches/property-belief-manipulation/experiments/2026-10-27-vlm-grasp-limits/sim
export RPENT_LAB_GENERATOR=v11
X=/home/chungyili/Codes/daily-logs/researches/property-belief-manipulation/experiments/2026-11-03-mass-com-estimation
SC=$X/scenes.json
Q=$X/out/qwen38_27b.jsonl
G=$X/out/gemma4_31b.jsonl
PY="systemd-run --user --scope -q -p MemoryMax=24G /home/chungyili/Codes/venvs/rpent-mujoco/bin/python -P -u -m"
B=robots/mujoco_lab/data/m2b2_v11c
if [ -e "$L" ]; then echo "refusing: $L exists"; exit 1; fi
[ "$(cat "$G_LOG/regen.exit" 2>/dev/null)" = "EXIT 0" ] || { echo "the regeneration has not ended with EXIT 0"; exit 1; }
[ -e "$B/folds/crossfit.json" ] || { echo "no $B folds"; exit 1; }
mkdir -p "$L" "$R/logs/wandb"
{
  echo "bundle $(realpath "$B")"
  echo "regen $(cat "$G_LOG/regen.exit"), code $(cat "$G_LOG/head.txt")"
  echo "branch $(git -C "$R" rev-parse HEAD), dirty files $(git --no-optional-locks -C "$R" status --porcelain --untracked-files=no | wc -l)"
  echo "simulator folder last commit $(git -C "$RPENT_MUJOCO_SIM_DIR" log -1 --format=%H -- .), dirty files $(git --no-optional-locks -C "$RPENT_MUJOCO_SIM_DIR" status --porcelain --untracked-files=no -- . | wc -l)"
  echo "corpora (the regeneration's corpora.sha256):"
  cat "$G_LOG/corpora.sha256"
  echo "bundle files:"
  (cd "$(realpath "$B")" && sha256sum manifest.json folds/folds.json folds/crossfit.json folds/fold_*/manifest.json)
} > "$L/bundle_provenance.txt"
step() {   # step NAME OUT -- command...: run, record the exit, stop on a failure
  local out=$1; shift
  $PY "$@" > "$out" 2>&1 </dev/null
  local rc=$?
  echo "EXIT $rc" > "${out%.log}.exit"
  [ "$rc" = 0 ] || { echo "stopped: ${out} exit $rc" > "$L/stopped"; exit "$rc"; }
}
for p in class generic "qwen38_27b=$Q" "gemma4_31b=$G"; do
  n=${p%%=*}
  O=$L/partA_$n
  mkdir -p "$O"
  if [ "$n" = "$p" ]; then prior=(--prior "$p"); else prior=(--prior vlm --vlm "${p#*=}"); fi
  step "$O/run.log" robots.mujoco_lab.estimation_check run --bundle $B --crossfit "${prior[@]}" --scenes "$SC" --workers 5 --out "$O"
  step "$O/summary.log" robots.mujoco_lab.estimation_check summary --dir "$O" --wandb
done
O=$L/menus_all
mkdir -p "$O"
step "$O/run.log" robots.mujoco_lab.first_menus run --bundle $B --scenes "$SC" --vlm "qwen38_27b=$Q" --vlm "gemma4_31b=$G" \
  --settings k10_A10:10:10,k100_A3:100:3,k100_A10:100:10,k30_A3:30:3,k30_A10:30:10,k100_A30:100:30 --workers 8 --out "$O"
step "$O/summary.log" robots.mujoco_lab.first_menus summary --dir "$O" --wandb
step "$L/compare_all.log" robots.mujoco_lab.masscom_report --part-a "class=$L/partA_class" --part-a "generic=$L/partA_generic" \
  --part-a "qwen38_27b=$L/partA_qwen38_27b" --part-a "gemma4_31b=$L/partA_gemma4_31b" --menus "$O" --out "$L/compare_all" --wandb
echo done > "$L/all.done"
