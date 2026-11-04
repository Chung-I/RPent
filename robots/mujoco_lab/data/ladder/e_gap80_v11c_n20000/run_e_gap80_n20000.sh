#!/bin/bash
# run_e_gap80_n20000.sh (2026-11-04): E's N-sensitivity run on v11c, the 80 gap cases at H = 1 to 3 with 20,000
# particles per belief (--particles 20000, an explicit override of the bundle's 2,000). NOT STARTED: the user decides
# between the local box and cml30. Settings from the timing study (README, "E at 20,000 particles"):
# - --chunk 1 (bitwise-equal tables to chunk 8; 1.3 GB per worker instead of 7.5 GB) and a fresh process per job
#   (--max-tasks-per-worker 1, the paper session's call: a worker's growth at 20,000 is not measured);
# - NUMPY_MADVISE_HUGEPAGE=0: with numpy's huge pages on, 4 copies stalled on memory compaction within 2 s;
# - E_WORKERS (default 4) in one unit with MemoryMax E_MEMMAX (default 9G) and no swap: 4 copies peaked at 5.3 GB;
# - a pressure guard: every 2 s it reads the unit's and the system's memory pressure (some avg10) and kills the unit at
#   25 % or more in 2 samples in a row, before systemd-oomd (50 % for 20 s) can pick another unit. A killed run resumes
#   with the same command: ladder_e run skips every job that has all its horizon rows.
# Refuses without a user manager, with a dirty tracked file, on a commit without --chunk (e0cd9a2), when another launch
# holds the lock, when the folder is complete, and on a resume at another commit (unless declared with E_CODE=<HEAD>
# that descends from it and changes only ladder_e.py and its test). E_DRY=1 runs the checks and prints the command only.
# Every 60 s overlap.log records the load, E's memory and pressure, MemAvailable and the python processes outside E's
# unit (the arm-port builder runs beside E in its own 5G scope).
# About 12.5 to 13 h at 4 workers on the local box (estimated from the 4-copy throughput). Then the summary (wandb).
set -u
W=${E_WORKERS:-4}
CAP=${E_MEMMAX:-9G}
DRY=${E_DRY:-0}
R=/home/chungyili/Codes/RPent-ladder
O=$R/logs/ladder_e/gap80_v11c_n20000
cd "$R" || exit 1
export PYTHONPATH=$R PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export WANDB_MODE=online WANDB_DIR=$R/logs/wandb NUMPY_MADVISE_HUGEPAGE=0
export RPENT_MUJOCO_SIM_DIR=/home/chungyili/Codes/daily-logs/researches/property-belief-manipulation/experiments/2026-10-27-vlm-grasp-limits/sim
export RPENT_LAB_GENERATOR=v11
UNIT="systemd-run --user --scope -q -p MemoryMax=$CAP -p MemorySwapMax=0"
PYM="/home/chungyili/Codes/venvs/rpent-mujoco/bin/python -P -u -m"
ARGS="--cases robots/mujoco_lab/data/e1/gap/cases.json --crossfit robots/mujoco_lab/data/m2b2_v11c/folds --horizons 1,2,3 --workers $W --max-tasks-per-worker 1 --particles 20000 --chunk 1 --out $O"
systemd-run --user --scope -q -p MemoryMax=64M -p MemorySwapMax=0 true || { echo "refusing: no user manager (systemd-run --user failed)"; exit 1; }
head_sha=$(git -C "$R" rev-parse HEAD)
dirty=$(git --no-optional-locks -C "$R" status --porcelain --untracked-files=no | wc -l)
[ "$dirty" = 0 ] || { echo "refusing: $dirty dirty tracked files"; exit 1; }
git -C "$R" merge-base --is-ancestor e0cd9a2 "$head_sha" || { echo "refusing: HEAD $head_sha has no --chunk (e0cd9a2)"; exit 1; }
[ ! -e "$O/all.done" ] || { echo "refusing: $O is complete"; exit 1; }
kind=first changed=none
if [ -s "$O/provenance.txt" ]; then
  kind=resume
  first=$(head -1 "$O/provenance.txt" | sed -n 's/^branch \([0-9a-f]*\), dirty files 0.*$/\1/p')
  [ -n "$first" ] || { echo "refusing: provenance line 1 is not a clean commit"; exit 1; }
  if [ "$head_sha" != "$first" ]; then
    [ "${E_CODE:-}" = "$head_sha" ] || { echo "refusing: code changed since $first (HEAD $head_sha); declare it with E_CODE=<HEAD>"; exit 1; }
    git -C "$R" merge-base --is-ancestor "$first" "$head_sha" || { echo "refusing: HEAD does not descend from $first"; exit 1; }
    other=$(git --no-optional-locks -C "$R" diff --name-only "$first" "$head_sha" | grep -v -x -e robots/mujoco_lab/ladder_e.py -e tests/unit_tests/robots/mujoco_lab/test_ladder_e.py)
    [ -z "$other" ] || { echo "refusing: other files changed since $first: $other"; exit 1; }
    changed="$(git --no-optional-locks -C "$R" diff --shortstat "$first" "$head_sha" | sed 's/^ *//') (ladder_e.py and its test)"
  fi
fi
settings="workers $W, max tasks per worker 1, particles 20000, chunk 1, MemoryMax $CAP, MemorySwapMax 0, NUMPY_MADVISE_HUGEPAGE=0, pressure guard 25 % x 2"
if [ "$DRY" = 1 ]; then
  echo "dry run ($kind launch): $UNIT $PYM robots.mujoco_lab.ladder_e run $ARGS"
  echo "provenance would get: branch $head_sha, dirty files 0, $settings, changed: $changed"
  exit 0
fi
mkdir -p "$O"
exec 9> "$O/launch.lock"
flock -n 9 || { echo "refusing: another launch holds $O/launch.lock"; exit 1; }
if [ "$kind" = first ]; then
  echo "branch $head_sha, dirty files 0, $(date +%Y-%m-%dT%H:%M:%S), $settings" > "$O/provenance.txt"
else
  echo "resume $(date +%Y-%m-%dT%H:%M:%S): branch $head_sha, dirty files 0, rows before $(wc -l < "$O/rows.jsonl" 2>/dev/null || echo 0), $settings, changed since $first: $changed" >> "$O/provenance.txt"
  ts=$(date +%Y%m%dT%H%M%S)
  [ -e "$O/run.log" ] && mv "$O/run.log" "$O/run_until_$ts.log"
  [ -e "$O/run.exit" ] && mv "$O/run.exit" "$O/run_until_$ts.exit"
fi
$UNIT $PYM robots.mujoco_lab.ladder_e run $ARGS > "$O/run.log" 2>&1 </dev/null &
pid=$! cg="" hot=0 i=0
sleep 2
while kill -0 "$pid" 2>/dev/null; do
  [ -z "$cg" ] && cg=/sys/fs/cgroup$(cut -d: -f3 /proc/$pid/cgroup 2>/dev/null)
  if [ -d "$cg" ]; then
    ua=$(awk '/^some/ {split($2, a, "="); print int(a[2])}' "$cg/memory.pressure")
    sa=$(awk '/^some/ {split($2, a, "="); print int(a[2])}' /proc/pressure/memory)
    if [ $((i % 30)) = 0 ]; then
      others=$(ps -ww -eo pid=,pcpu=,rss=,comm= --sort=-pcpu | awk -v e=" $(tr '\n' ' ' < "$cg/cgroup.procs") " '$4 ~ /^python/ && index(e, " " $1 " ") == 0 {n++; c += $2; r += $3} END {printf "%d pythons, %.0f %% CPU, %.1f GB", n, c, r / 1048576}')
      echo "$(date +%Y-%m-%dT%H:%M:%S) load $(cut -d' ' -f1 /proc/loadavg), E $(( $(cat $cg/memory.current) / 1048576 )) MB (peak $(( $(cat $cg/memory.peak) / 1048576 ))), E avg10 $ua %, system avg10 $sa %, MemAvailable $(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo) MB, outside E: $others" >> "$O/overlap.log"
    fi
    i=$((i + 1))
    if [ "$ua" -ge 25 ] || [ "$sa" -ge 25 ]; then hot=$((hot + 1)); else hot=0; fi
    if [ "$hot" -ge 2 ]; then
      echo "$(date +%Y-%m-%dT%H:%M:%S) guard: unit avg10 $ua %, system avg10 $sa %, memory.current $(( $(cat $cg/memory.current) / 1048576 )) MB; killing the unit" >> "$O/guard.log"
      for p in $(cat "$cg/cgroup.procs"); do kill -KILL "$p" 2>/dev/null; done
    fi
  fi
  sleep 2
done
wait "$pid"; rc=$?
echo "EXIT $rc" > "$O/run.exit"; [ "$rc" = 0 ] || exit "$rc"
$UNIT $PYM robots.mujoco_lab.ladder_e summary --dir "$O" --wandb > "$O/summary.log" 2>&1 </dev/null
rc=$?; echo "EXIT $rc" > "$O/summary.exit"; [ "$rc" = 0 ] || exit "$rc"
echo done > "$O/all.done"
