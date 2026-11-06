#!/bin/bash
# cgb_shards.sh K [SETTINGS]: prints the queue lines (queue2.sh jobs.txt) of the com_gap rerun on m2b2_v12b (TODO 6
# step 1, 1,045 cases, both priors, with the post_identify column of Amendment 3) in K shards of one worker each: the
# bundle's objects sorted, shard k gets every K-th from k (cg_shards.sh's split). SETTINGS (name:care:A,...) replaces
# the three default settings, for a setting that the v12b E1 gap check adds. Units v12-cgb-sK, MemoryMax 6G.
set -u
K=$1; S=${2:-}
R=/home/chungyili/Codes/RPent-comgap-v12b; RS=$R/logs/comgap/run_step2.sh; F=$R/logs/comgap/full_v12b
ARMSIM=/home/chungyili/Codes/daily-logs-worktrees/v12checks-armsim-fe46f6c/researches/property-belief-manipulation/experiments/2026-11-04-arm-port/sim
B=/home/chungyili/Codes/RPent-arm-identify/robots/mujoco_lab/data/m2b2_v12b
SC=/home/chungyili/Codes/daily-logs/researches/property-belief-manipulation/experiments/2026-11-03-mass-com-estimation/scenes.json
run="robots.mujoco_lab.com_gap run --bundle $B --scenes $SC --workers 1 --post-identify${S:+ --settings $S}"
python3 -c "import json; o = sorted(json.load(open('$B/folds/folds.json'))); [print(','.join(o[k::$K])) for k in range($K)]" |
  { k=0; while read -r objs; do echo "SIM=$ARMSIM MEMMAX=6G BUNDLE=$B $RS cgb-s$k $F/shard_$k $run --objects $objs --out $F/shard_$k"; k=$((k + 1)); done; }
