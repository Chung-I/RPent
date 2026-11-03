#!/bin/bash
# run_gap_v11c.sh: the E1 oracle-gap check (rules only, the v10 check's cases file) under generator v11 with the bundle
# m2b2_v11c, then its expected-cost rescore, each in a memory-capped scope; writes logs/e1/{gap_v11c,gap_v11c_expected}/
# run.log, run.exit, summary.log and summary.exit, and logs/e1/gap_v11c.done at the end. Refuses an existing output
# folder and a regeneration that has not ended with EXIT 0. The bundle is a symlink into RPent-real, so
# logs/e1/bundle_provenance_v11c.txt records its real path, the regeneration's code, exit and corpora, the simulator's
# commit and the sha256 of the bundle's manifests and folds.
set -u
R=/home/chungyili/Codes/RPent-e1v11
L=$R/logs/e1
G_LOG=/home/chungyili/Codes/RPent-real/logs/m2b2_v11c/final
cd "$R" || exit 1
export PYTHONPATH=$R PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl OMP_NUM_THREADS=2 WANDB_MODE=online WANDB_DIR=$L/wandb
export RPENT_MUJOCO_SIM_DIR=/home/chungyili/Codes/daily-logs/researches/property-belief-manipulation/experiments/2026-10-27-vlm-grasp-limits/sim
export RPENT_LAB_GENERATOR=v11
PY="systemd-run --user --scope -q -p MemoryMax=24G /home/chungyili/Codes/venvs/rpent-mujoco/bin/python -P -u -m"
B=robots/mujoco_lab/data/m2b2_v11c
for d in gap_v11c gap_v11c_expected bundle_provenance_v11c.txt; do
    if [ -e "$L/$d" ]; then echo "refusing: $L/$d exists"; exit 1; fi
done
[ "$(cat "$G_LOG/regen.exit" 2>/dev/null)" = "EXIT 0" ] || { echo "the regeneration has not ended with EXIT 0"; exit 1; }
[ -e "$B/folds/crossfit.json" ] || { echo "no $B folds"; exit 1; }
mkdir -p "$L/gap_v11c" "$L/gap_v11c_expected" "$L/wandb"
{
  echo "bundle $(realpath "$B")"
  echo "regen $(cat "$G_LOG/regen.exit"), code $(cat "$G_LOG/head.txt")"
  echo "branch $(git -C "$R" rev-parse HEAD), dirty files $(git --no-optional-locks -C "$R" status --porcelain --untracked-files=no | wc -l)"
  echo "simulator folder last commit $(git -C "$RPENT_MUJOCO_SIM_DIR" log -1 --format=%H -- .), dirty files $(git --no-optional-locks -C "$RPENT_MUJOCO_SIM_DIR" status --porcelain --untracked-files=no -- . | wc -l)"
  echo "corpora (the regeneration's corpora.sha256):"
  cat "$G_LOG/corpora.sha256"
  echo "bundle files:"
  (cd "$(realpath "$B")" && sha256sum manifest.json folds/folds.json folds/crossfit.json folds/fold_*/manifest.json)
} > "$L/bundle_provenance_v11c.txt"
$PY robots.mujoco_lab.e1_gap run --cases robots/mujoco_lab/data/e1/gap/cases.json --crossfit $B/folds \
    --workers 8 --out "$L/gap_v11c" > "$L/gap_v11c/run.log" 2>&1 </dev/null
rc=$?; echo "EXIT $rc" > "$L/gap_v11c/run.exit"; [ "$rc" = 0 ] || exit "$rc"
$PY robots.mujoco_lab.e1_gap summary --dir "$L/gap_v11c" --wandb > "$L/gap_v11c/summary.log" 2>&1 </dev/null
echo "EXIT $?" > "$L/gap_v11c/summary.exit"
$PY robots.mujoco_lab.e1_expected gap --src "$L/gap_v11c" --workers 8 --out "$L/gap_v11c_expected" > "$L/gap_v11c_expected/run.log" 2>&1 </dev/null
rc=$?; echo "EXIT $rc" > "$L/gap_v11c_expected/run.exit"; [ "$rc" = 0 ] || exit "$rc"
$PY robots.mujoco_lab.e1_expected summary --dir "$L/gap_v11c_expected" --wandb > "$L/gap_v11c_expected/summary.log" 2>&1 </dev/null
echo "EXIT $?" > "$L/gap_v11c_expected/summary.exit"
echo done > "$L/gap_v11c.done"
