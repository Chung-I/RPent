#!/bin/bash
# Experiment 1 main summary over the 14 shards, from the evpi-summary worktree (RPent-evpi-sum), wandb on.
set -u
S=/home/chungyili/Codes/RPent-evpi-sum
E=/home/chungyili/Codes/RPent-evpi/logs/evpi
OUT=$E/summary
[ -e $OUT ] && { echo "refusing: $OUT exists"; exit 1; }
mkdir -p $OUT
DIRS=$(ls -d $E/full_class/shard_{0,1,2,3,4,5,6} $E/full_generic/shard_{0,1,2,3,4,5,6} | paste -sd,)
COMGAP=/home/chungyili/Codes/RPent-comgap/robots/mujoco_lab/data/com_gap/full_v12_n20000/records.jsonl.gz
SIM=/home/chungyili/Codes/daily-logs-worktrees/v12checks-armsim-fe46f6c/researches/property-belief-manipulation/experiments/2026-11-04-arm-port/sim
{
  echo "code $(git -C $S rev-parse HEAD) (branch $(git -C $S branch --show-current)), dirty files $(git -C $S status --porcelain | wc -l)"
  echo "comgap $(sha256sum $COMGAP)"
  for d in ${DIRS//,/ }; do echo "shard $d $(sha256sum < $d/records.jsonl | cut -c1-16) $(cat $d/run.exit)"; done
  echo "env NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 RPENT_LAB_GENERATOR=v11 WANDB_MODE=online, unit v12-ev-summary MemoryMax 8G MemorySwapMax 0"
  echo "command python -P -u -m robots.mujoco_lab.evpi_props summary --dirs $DIRS --out $OUT --comgap $COMGAP --wandb"
  echo "start $(date '+%F %T')"
} > $OUT/provenance.txt
cd $S
systemd-run --user --scope -q --unit=v12-ev-summary -p MemoryMax=8G -p MemorySwapMax=0 \
  env PYTHONPATH=$S NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 RPENT_LAB_GENERATOR=v11 WANDB_MODE=online \
  RPENT_MUJOCO_SIM_DIR=$SIM /home/chungyili/Codes/venvs/rpent-mujoco/bin/python -P -u -m robots.mujoco_lab.evpi_props \
  summary --dirs $DIRS --out $OUT --comgap $COMGAP --wandb > $OUT/run.log 2>&1
rc=$?
echo "end $(date '+%F %T')" >> $OUT/provenance.txt
echo "EXIT $rc" > $OUT/run.exit
