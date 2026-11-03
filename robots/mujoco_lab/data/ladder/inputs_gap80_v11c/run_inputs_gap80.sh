#!/bin/bash
# run_inputs_gap80.sh: item 2 (the B-D precompute) on the gap check's 80 seed-13 cases under the three ladder
# settings, bundle m2b2_v11c (its 5-fold cross-fit), generator v11; then the lowest-D check against the gap rescore's
# oracle records. Each step writes OUT/<step>.log and OUT/<step>.exit; a failed step stops the script; OUT/all.done
# at the end. Refuses an existing output folder.
set -u
R=/home/chungyili/Codes/RPent-ladder
O=$R/logs/ladder/inputs_gap80_v11c
cd "$R" || exit 1
export PYTHONPATH=$R PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl OMP_NUM_THREADS=2 WANDB_MODE=online WANDB_DIR=$R/logs/wandb
export RPENT_MUJOCO_SIM_DIR=/home/chungyili/Codes/daily-logs/researches/property-belief-manipulation/experiments/2026-10-27-vlm-grasp-limits/sim
export RPENT_LAB_GENERATOR=v11
PY="systemd-run --user --scope -q -p MemoryMax=24G /home/chungyili/Codes/venvs/rpent-mujoco/bin/python -P -u -m"
if [ -e "$O" ]; then echo "refusing: $O exists"; exit 1; fi
mkdir -p "$O" "$R/logs/wandb"
echo "branch $(git -C "$R" rev-parse HEAD), dirty files $(git --no-optional-locks -C "$R" status --porcelain --untracked-files=no | wc -l)" > "$O/provenance.txt"
echo "bundle $(realpath robots/mujoco_lab/data/m2b2_v11c)" >> "$O/provenance.txt"
step() {   # step NAME -- command...: run, record the exit, stop on a failure
  local name=$1; shift
  $PY "$@" > "$O/$name.log" 2>&1 </dev/null
  local rc=$?
  echo "EXIT $rc" > "$O/$name.exit"
  [ "$rc" = 0 ] || { echo "stopped: $name exit $rc" > "$O/stopped"; exit "$rc"; }
}
step inputs robots.mujoco_lab.ladder inputs --cases robots/mujoco_lab/data/e1/gap/cases.json \
  --settings k100_A3_abs,k30_A3_abs,k100_A10_abs --crossfit robots/mujoco_lab/data/m2b2_v11c/folds --workers 8 --out "$O"
step check robots.mujoco_lab.ladder check --dir "$O" --expected robots/mujoco_lab/data/e1/gap_v11c_expected --wandb
echo done > "$O/all.done"
