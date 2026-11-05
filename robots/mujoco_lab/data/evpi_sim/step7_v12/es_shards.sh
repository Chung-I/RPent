#!/bin/bash
# es_shards.sh K [SAMPLE]: prints the queue lines (queue2.sh jobs.txt) of experiment 1's step 7 full run (evpi_sim, the
# simulator check of the sampled θ-critical records) in K shards, one worker each: the sample's objects sorted, shard k
# gets every K-th from k. `evpi_sim summary --dirs` joins the shards.
set -u
K=$1
R=/home/chungyili/Codes/RPent-evpi-sim; RS=$R/logs/evpi_sim/run_step2.sh; F=$R/logs/evpi_sim/full
SMP=${2:-$R/logs/evpi_sim/sample.json}
ARMSIM=/home/chungyili/Codes/daily-logs-worktrees/v12checks-armsim-fe46f6c/researches/property-belief-manipulation/experiments/2026-11-04-arm-port/sim
B12=/home/chungyili/Codes/RPent-arm/robots/mujoco_lab/data/m2b2_v12
python3 -c "import json; o = sorted({r['object'] for r in json.load(open('$SMP'))['records']}); [print(','.join(o[k::$K])) for k in range($K)]" |
  { k=0; while read -r objs; do [ -n "$objs" ] && echo "SIM=$ARMSIM BUNDLE=$B12 $RS es-$k $F/shard_$k robots.mujoco_lab.evpi_sim run --sample $SMP --workers 1 --objects $objs --out $F/shard_$k"; k=$((k + 1)); done; }
