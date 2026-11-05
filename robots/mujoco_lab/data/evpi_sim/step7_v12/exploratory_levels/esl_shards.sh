#!/bin/bash
# esl_shards.sh K [SAMPLE]: prints the queue lines (queue2.sh jobs.txt) of step 7's exploratory levels check (evpi_sim_levels
# levels: the model at the truth for the main reading's records, no simulation) in K shards, one process each: the objects
# of those records sorted, shard k gets every K-th from k. `evpi_sim_levels summary --levels` joins the shards.
set -u
K=$1
R=/home/chungyili/Codes/RPent-evpi-sim; RS=$R/logs/evpi_sim/run_step2.sh; F=$R/logs/evpi_sim_levels/full
SMP=${2:-$R/logs/evpi_sim/sample.json}
ARMSIM=/home/chungyili/Codes/daily-logs-worktrees/v12checks-armsim-fe46f6c/researches/property-belief-manipulation/experiments/2026-11-04-arm-port/sim
B12=/home/chungyili/Codes/RPent-arm/robots/mujoco_lab/data/m2b2_v12
cd $R && PYTHONPATH=$R /home/chungyili/Codes/venvs/rpent-mujoco/bin/python -P -c "
import json
from robots.mujoco_lab.evpi_sim_levels import _wanted
o = sorted({r['object'] for r in json.load(open('$SMP'))['records'] if _wanted(r)})
[print(','.join(o[k::$K])) for k in range($K)]" |
  { k=0; while read -r objs; do [ -n "$objs" ] && echo "SIM=$ARMSIM BUNDLE=$B12 $RS esl-$k $F/shard_$k robots.mujoco_lab.evpi_sim_levels levels --sample $SMP --objects $objs --out $F/shard_$k"; k=$((k + 1)); done; }
