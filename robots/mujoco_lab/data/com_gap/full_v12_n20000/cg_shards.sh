#!/bin/bash
# cg_shards.sh K [OBJECTS]: prints the queue lines (queue.sh jobs.txt) of the com_gap full run (TODO 6 step 1, 1,045
# cases on m2b2_v12, both priors, the three settings) in K shards, one worker each: the bundle's objects sorted, shard k
# gets every K-th from k. With OBJECTS (comma-separated: the unfinished objects of a stopped shard), one rerun shard of
# those objects instead (named by the time). `com_gap merge` joins the shards' dirs for the summary.
set -u
K=$1
R=/home/chungyili/Codes/RPent-comgap; RS=$R/logs/comgap/run_step2.sh; F=$R/logs/comgap/full
ARMSIM=/home/chungyili/Codes/daily-logs-worktrees/v12checks-armsim-fe46f6c/researches/property-belief-manipulation/experiments/2026-11-04-arm-port/sim
B12=/home/chungyili/Codes/RPent-arm/robots/mujoco_lab/data/m2b2_v12
SC=/home/chungyili/Codes/daily-logs/researches/property-belief-manipulation/experiments/2026-11-03-mass-com-estimation/scenes.json
run="robots.mujoco_lab.com_gap run --bundle $B12 --scenes $SC --workers 1"
if [ -n "${2:-}" ]; then
  t=$(date +%H%M)
  echo "SIM=$ARMSIM BUNDLE=$B12 $RS cg-r$t $F/rerun_$t $run --objects $2 --out $F/rerun_$t"
  exit 0
fi
python3 -c "import json, sys; o = sorted(json.load(open('$B12/folds/folds.json'))); [print(','.join(o[k::$K])) for k in range($K)]" |
  { k=0; while read -r objs; do echo "SIM=$ARMSIM BUNDLE=$B12 $RS cg-s$k $F/shard_$k $run --objects $objs --out $F/shard_$k"; k=$((k + 1)); done; }
