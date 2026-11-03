#!/bin/bash
# run_w0_v11c.sh (2026-11-04): W0, the decision value of an exact CoM at the first menu, on m2b2_v11c. One first_menus run
# with the class and generic priors plus the exact CoM, exact mass and both priors (--exact: the case's generator truth,
# mass log-sd 0.01, CoM-fraction sd 0.005) at the three settings; its summary; w0_report. Each step writes its log and
# exit (wandb); a failed step stops the script. Refuses an existing output folder. logs/w0_v11c/all.done at the end.
set -u
R=/home/chungyili/Codes/RPent-masscom-v11
L=$R/logs/w0_v11c
G_LOG=/home/chungyili/Codes/RPent-real/logs/m2b2_v11c/final
cd "$R" || exit 1
export PYTHONPATH=$R PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl OMP_NUM_THREADS=2 WANDB_MODE=online WANDB_DIR=$R/logs/wandb
export RPENT_MUJOCO_SIM_DIR=/home/chungyili/Codes/daily-logs/researches/property-belief-manipulation/experiments/2026-10-27-vlm-grasp-limits/sim
export RPENT_LAB_GENERATOR=v11
SC=/home/chungyili/Codes/daily-logs/researches/property-belief-manipulation/experiments/2026-11-03-mass-com-estimation/scenes.json
PY="systemd-run --user --scope -q -p MemoryMax=24G /home/chungyili/Codes/venvs/rpent-mujoco/bin/python -P -u -m"
B=robots/mujoco_lab/data/m2b2_v11c
if [ -e "$L" ]; then echo "refusing: $L exists"; exit 1; fi
[ "$(cat "$G_LOG/regen.exit" 2>/dev/null)" = "EXIT 0" ] || { echo "the regeneration has not ended with EXIT 0"; exit 1; }
mkdir -p "$L" "$R/logs/wandb"
{
  echo "bundle $(realpath "$B")"
  echo "regen $(cat "$G_LOG/regen.exit"), code $(cat "$G_LOG/head.txt")"
  echo "branch $(git -C "$R" rev-parse HEAD), dirty files $(git --no-optional-locks -C "$R" status --porcelain --untracked-files=no | wc -l)"
  echo "simulator folder last commit $(git -C "$RPENT_MUJOCO_SIM_DIR" log -1 --format=%H -- .), dirty files $(git --no-optional-locks -C "$RPENT_MUJOCO_SIM_DIR" status --porcelain --untracked-files=no -- . | wc -l)"
  (cd "$(realpath "$B")" && sha256sum manifest.json folds/folds.json folds/crossfit.json folds/fold_*/manifest.json)
} > "$L/bundle_provenance.txt"
step() {   # step OUT command...: run, record the exit, stop on a failure
  local out=$1; shift
  $PY "$@" > "$out" 2>&1 </dev/null
  local rc=$?
  echo "EXIT $rc" > "${out%.log}.exit"
  [ "$rc" = 0 ] || { echo "stopped: ${out} exit $rc" > "$L/stopped"; exit "$rc"; }
}
O=$L/menus_exact
mkdir -p "$O"
step "$O/run.log" robots.mujoco_lab.first_menus run --bundle $B --scenes "$SC" --exact \
  --settings k100_A3:100:3,k30_A3:30:3,k100_A10:100:10 --workers 8 --out "$O"
step "$O/summary.log" robots.mujoco_lab.first_menus summary --dir "$O" --wandb
step "$L/w0_report.log" robots.mujoco_lab.w0_report --dir "$O" --scenes "$SC" --wandb
echo done > "$L/all.done"
