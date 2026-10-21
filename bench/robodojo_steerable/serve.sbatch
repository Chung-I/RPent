#!/bin/bash
# Nano4 services for the steerable RPent RoboDojo condition (simulator, Pi0.5
# and GraspGen run on an RTX host): Qwen3.8-27B-FP8 planner (vLLM) and SAM3 on
# one GPU. Ports derive from the job id; the log prints the node and ports.
#SBATCH --job-name=rpent-steer-serve
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --account=mst115550
#SBATCH --time=12:00:00
#SBATCH --output=/work/roboleon1295/rpent-serve/steer-%j.log
set -euo pipefail
W=/work/roboleon1295
REPO=$W/RPent-steer  # worktree of bench/robodojo-steerable; $W/RPent serves the live LIBERO run
SVC=$W/rpent-serve/steer/$SLURM_JOB_ID
mkdir -p "$SVC"
export HOME=$W/jobhome XDG_CACHE_HOME=$W/jobhome/.cache TRITON_CACHE_DIR=$W/jobhome/.triton
export HF_HOME=$W/hf HF_HUB_OFFLINE=1
BASE=$((30000 + (SLURM_JOB_ID % 3000) * 10))
LLM_PORT=$BASE SAM3_PORT=$((BASE + 3))
IP=$(getent hosts "$(hostname)" | cut -d' ' -f1)
echo "node=$(hostname) ip=$IP llm=$LLM_PORT sam3=$SAM3_PORT rpent=$(git -C $REPO rev-parse HEAD)"

pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT

VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_API_KEY="$(cat $W/rpent-serve/api_key)" \
  $W/.venv-vllm-minicpm/bin/vllm serve $W/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --served-model-name Qwen/Qwen3.8-27B-FP8 --host 0.0.0.0 --port $LLM_PORT \
  --max-model-len 262144 --max-num-seqs 2 --gpu-memory-utilization 0.85 --dtype bfloat16 \
  --limit-mm-per-prompt '{"image":4096}' --mm-processor-cache-gb 32 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --enable-prompt-tokens-details > "$SVC/vllm.log" 2>&1 &
pids+=($!)

cd "$REPO"
SAM3_CHECKPOINT_PATH=$W/checkpoints/sam3/sam3.pt \
  $W/.venv-rpent-liberopro/bin/python -u -m rpent.robots.components.sam3_server \
  --transport http --host 0.0.0.0 --port $SAM3_PORT > "$SVC/sam3.log" 2>&1 &
pids+=($!)

for i in $(seq 1 120); do
  grep -q "Application startup complete" "$SVC/vllm.log" && break
  for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null || { echo "a service exited during startup"; tail -30 "$SVC"/*.log; exit 1; }; done
  sleep 10
done
grep -q "Application startup complete" "$SVC/vllm.log" || { echo "vLLM not ready after 20 min"; tail -30 "$SVC/vllm.log"; exit 1; }
echo "services ready at $(date -Is)"
wait
