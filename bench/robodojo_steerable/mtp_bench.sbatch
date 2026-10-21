#!/bin/bash
# Qwen3.8-27B-FP8 decode speed with and without MTP speculative decoding, using
# the production planner serve flags. Replays $RUN/requests.json (built by
# build_mtp_requests.py) once per server config; results land in $RUN.
#SBATCH --job-name=rpent-mtp-bench
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --account=mst115550
#SBATCH --time=03:00:00
#SBATCH --output=/work/roboleon1295/rpent-serve/mtp-bench-%j.log
set -euo pipefail
W=/work/roboleon1295
RUN=$W/rpent-serve/mtp-bench
REPO=$W/RPent-mtp
export HOME=$W/jobhome XDG_CACHE_HOME=$W/jobhome/.cache TRITON_CACHE_DIR=$W/jobhome/.triton
export HF_HOME=$W/hf HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_API_KEY="$(cat $W/rpent-serve/api_key)"
PORT=$((30000 + (SLURM_JOB_ID % 3000) * 10 + 7))
MODEL=$W/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a
echo "node=$(hostname) port=$PORT rpent=$(git -C $REPO rev-parse HEAD)"

run_config() {
  local label=$1; shift
  $W/.venv-vllm-minicpm/bin/vllm serve $MODEL \
    --served-model-name Qwen/Qwen3.8-27B-FP8 --host 127.0.0.1 --port $PORT \
    --max-model-len 262144 --max-num-seqs 2 --gpu-memory-utilization 0.85 --dtype bfloat16 \
    --limit-mm-per-prompt '{"image":4096}' --mm-processor-cache-gb 32 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --enable-prompt-tokens-details "$@" > $RUN/serve-$label.log 2>&1 &
  local pid=$!
  for _ in $(seq 180); do
    curl -sf http://127.0.0.1:$PORT/health >/dev/null && break
    kill -0 $pid 2>/dev/null || { echo "$label: server exited"; tail -30 $RUN/serve-$label.log; return 0; }
    sleep 10
  done
  $W/.venv-vllm-minicpm/bin/python -u $REPO/bench/robodojo_steerable/mtp_bench.py \
    http://127.0.0.1:$PORT $RUN/requests.json $RUN/result-$label.json --label $label || echo "$label: bench failed"
  kill $pid; wait $pid 2>/dev/null || true
}

run_config baseline
for n in 1 2 3; do
  run_config mtp$n --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$n}"
done
echo done
