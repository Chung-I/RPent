#!/bin/bash
# post_v12b.sh: the merge and the summary of the com_gap rerun on m2b2_v12b (written 2026-11-06, before the shards end;
# b5). The same two commands as the v12 run's merge_job and summary_job. Refuses to start before all 7 shards ended
# with EXIT 0. Each step is one run_step2.sh unit of the v12b checkout (MemoryMax 4G, no swap, huge pages off). The
# summary has wandb on. The merge refuses a case without complete records (rerun its object as a new shard).
set -u
R=/home/chungyili/Codes/RPent-comgap-v12b
F=$R/logs/comgap/full_v12b
for k in 0 1 2 3 4 5 6; do
  grep -qx "EXIT 0" $F/shard_$k/run.exit 2>/dev/null || { echo "refusing: shard_$k has not ended with EXIT 0"; exit 1; }
done
export MEMMAX=4G
export SIM=/home/chungyili/Codes/daily-logs-worktrees/v12checks-armsim-fe46f6c/researches/property-belief-manipulation/experiments/2026-11-04-arm-port/sim
DIRS=$(echo $F/shard_{0..6} | tr ' ' ,)
$R/logs/comgap/run_step2.sh cgb-merge $F/merge_job robots.mujoco_lab.com_gap merge --dirs $DIRS --out $F/merged
[ "$(cat $F/merge_job/run.exit 2>/dev/null)" = "EXIT 0" ] || { echo "merge failed: see $F/merge_job/run.log"; exit 1; }
$R/logs/comgap/run_step2.sh cgb-summary $F/summary_job robots.mujoco_lab.com_gap summary --dir $F/merged --wandb
echo "summary $(cat $F/summary_job/run.exit 2>/dev/null) $(date '+%F %T')"
