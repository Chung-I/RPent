#!/bin/bash
# Full retry curve: retry vs retry_frozen, 11 repeats per object (m4 size), seed 12.
cd /home/chungyili/Codes/RPent-retry || exit 9
L=/home/chungyili/Codes/RPent-retry/logs/retry_full
PY=/home/chungyili/Codes/venvs/rpent-mujoco/bin/python
F=/home/chungyili/Codes/RPent-retry/robots/mujoco_lab/data/m2b2/folds
git rev-parse HEAD > $L/head.txt; git -C "$RPENT_MUJOCO_SIM_DIR" log -1 --format=%H -- . > $L/sim_head.txt
for p in retry retry_frozen; do
  s=$(date +%s)
  $PY -P -u -m robots.mujoco_lab.calibrate --crossfit "$F" --protocol $p --retry-attempts 5 --repeats 11 --seed 12 --workers 4 \
      --care 1 --abstain-cost 10 --probe-time-scale 1 --out $L/${p}_full.json > $L/$p.log 2>&1
  echo "$? $(( $(date +%s) - s ))" > $L/$p.exit
done
echo done > $L/all.exit
