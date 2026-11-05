#!/bin/bash
# ev_shards.sh K PRIOR [OBJECTS]: prints the queue lines (queue2.sh jobs.txt) of experiment 1's full run (EVPI per hidden
# property, 1,045 com_gap cases on m2b2_v12, five settings) for one prior in K shards, one worker each: the bundle's
# objects sorted, shard k gets every K-th from k. With OBJECTS (comma-separated: the unfinished objects of a stopped
# shard), one rerun shard of those objects instead (named by the time). `evpi_props summary --dirs` joins the shards.
set -u
K=$1 P=$2
R=/home/chungyili/Codes/RPent-evpi; RS=$R/logs/evpi/run_step2.sh; F=$R/logs/evpi/full_$P
ARMSIM=/home/chungyili/Codes/daily-logs-worktrees/v12checks-armsim-fe46f6c/researches/property-belief-manipulation/experiments/2026-11-04-arm-port/sim
B12=/home/chungyili/Codes/RPent-arm/robots/mujoco_lab/data/m2b2_v12
SC=/home/chungyili/Codes/daily-logs/researches/property-belief-manipulation/experiments/2026-11-03-mass-com-estimation/scenes.json
run="robots.mujoco_lab.evpi_props run --bundle $B12 --scenes $SC --priors $P --workers 1"
if [ -n "${3:-}" ]; then
  t=$(date +%H%M)
  echo "SIM=$ARMSIM BUNDLE=$B12 $RS ev-${P:0:1}r$t $F/rerun_$t $run --objects $3 --out $F/rerun_$t"
  exit 0
fi
python3 -c "import json, sys; o = sorted(json.load(open('$B12/folds/folds.json'))); [print(','.join(o[k::$K])) for k in range($K)]" |
  { k=0; while read -r objs; do echo "SIM=$ARMSIM BUNDLE=$B12 $RS ev-${P:0:1}$k $F/shard_$k $run --objects $objs --out $F/shard_$k"; k=$((k + 1)); done; }
