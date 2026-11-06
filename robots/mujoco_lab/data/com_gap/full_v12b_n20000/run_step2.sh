#!/bin/bash
# run_step2.sh NAME OUT MODULE ARGS...: one com_gap process (the v12b rerun; the v12 checks launcher, its own R) in its own systemd-run --user unit v12-NAME (MemoryMax
# ${MEMMAX:-8G}, no swap, huge pages off, one thread), with provenance; refuses an existing OUT or a live v12-NAME.
# SIM overrides the sim folder (default: the hand sim). BUNDLE (optional) adds the sha256 of its manifest and folds.
# MODULE's own --out goes in ARGS.
set -u
NAME=$1 OUT=$2; shift 2
R=/home/chungyili/Codes/RPent-comgap-v12b
[ -e "$OUT" ] && { echo "refusing: $OUT exists"; exit 1; }
systemctl --user is-active -q "v12-$NAME.scope" && { echo "refusing: v12-$NAME.scope is active"; exit 1; }
mkdir -p "$OUT" "$R/logs/wandb"
cd "$R" || exit 9
export PYTHONPATH=$R PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMPY_MADVISE_HUGEPAGE=0 WANDB_MODE=online WANDB_DIR=$R/logs/wandb RPENT_LAB_GENERATOR=v11
export RPENT_MUJOCO_SIM_DIR=${SIM:-/home/chungyili/Codes/daily-logs/researches/property-belief-manipulation/experiments/2026-10-27-vlm-grasp-limits/sim}
unset RPENT_LAB_TORQUE_NOISE RPENT_PICK_PLACE_N_PARTICLES RPENT_LAB_FORCE_NOISE RPENT_ARM_ROBOT_MODEL
M=${MEMMAX:-8G}
{
  echo "code $(git -C $R rev-parse HEAD) (branch $(git -C $R branch --show-current)), dirty files $(git --no-optional-locks -C $R status --porcelain --untracked-files=no | wc -l)"
  echo "sim $RPENT_MUJOCO_SIM_DIR, last commit $(git -C $RPENT_MUJOCO_SIM_DIR log -1 --format=%H -- .), dirty files $(git --no-optional-locks -C $RPENT_MUJOCO_SIM_DIR status --porcelain --untracked-files=no -- . | wc -l)"
  rm_=$(dirname "$RPENT_MUJOCO_SIM_DIR")/out/r1_robot.npz
  [ -e "$rm_" ] && echo "robot model $(sha256sum "$rm_")"
  if [ -n "${BUNDLE:-}" ]; then
    echo "bundle $(realpath "$BUNDLE"):"
    (cd "$(realpath "$BUNDLE")" && sha256sum manifest.json folds/folds.json folds/crossfit.json folds/fold_*/manifest.json 2>/dev/null)
  fi
  echo "env NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 RPENT_LAB_GENERATOR=v11 WANDB_MODE=online, unit v12-$NAME MemoryMax $M MemorySwapMax 0"
  echo "command python -P -u -m $*"
  echo "start $(date '+%F %T')"
} > "$OUT/provenance.txt"
systemd-run --user --scope -q --unit="v12-$NAME" -p MemoryMax=$M -p MemorySwapMax=0 /home/chungyili/Codes/venvs/rpent-mujoco/bin/python -P -u -m "$@" > "$OUT/run.log" 2>&1 </dev/null
rc=$?
echo "end $(date '+%F %T')" >> "$OUT/provenance.txt"
echo "EXIT $rc" > "$OUT/run.exit"
