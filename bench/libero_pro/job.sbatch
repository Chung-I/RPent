#!/bin/bash
# One self-contained LIBERO-Pro evaluation job on Nano4 (2 H200):
#   GPU 0: vLLM Qwen3.8-27B-FP8; GPU 1: Pi0.5 + SAM3 + LIBERO EGL rendering.
# The worker takes cells from the shared queue in $OUT_ROOT; resubmit to resume.
# Usage: sbatch --export=ALL,OUT_ROOT=/work/.../full,SLOTS=12[,WORKER_ARGS="--suites libero_10_task --limit 12"] job.sbatch
#SBATCH --job-name=rpent-libpro
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn001
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=400G
#SBATCH --account=mst115550
#SBATCH --time=24:00:00
#SBATCH --output=/work/roboleon1295/rpent-bench/logs/libpro-%j.log
set -euo pipefail
: "${OUT_ROOT:?set OUT_ROOT}" "${SLOTS:=12}" "${WORKER_ARGS:=}"
W=/work/roboleon1295
REPO=$W/RPent
V=$W/.venv-rpent-liberopro
SVC=$OUT_ROOT/services/$SLURM_JOB_ID
mkdir -p "$SVC"
export HOME=$W/jobhome XDG_CACHE_HOME=$W/jobhome/.cache TRITON_CACHE_DIR=$W/jobhome/.triton
export HF_HOME=$W/hf HF_HUB_OFFLINE=1
read -r GPU_LLM GPU_SIM <<<"${CUDA_VISIBLE_DEVICES//,/ }"
echo "node=$(hostname) gpus=$CUDA_VISIBLE_DEVICES llm=$GPU_LLM sim=$GPU_SIM commit=$(git -C $REPO rev-parse HEAD)"

pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT

VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_API_KEY="$(cat $W/rpent-serve/api_key)" CUDA_VISIBLE_DEVICES=$GPU_LLM \
  $W/.venv-vllm-minicpm/bin/vllm serve $W/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --served-model-name Qwen/Qwen3.8-27B-FP8 --host 127.0.0.1 --port 8001 \
  --max-model-len 262144 --max-num-seqs 12 --gpu-memory-utilization 0.9 --dtype bfloat16 \
  --limit-mm-per-prompt '{"image":4096}' --mm-processor-cache-gb 64 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --enable-prompt-tokens-details > "$SVC/vllm.log" 2>&1 &
pids+=($!)
PI05_CHECKPOINT_PATH=$W/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT \
  $V/bin/python -u -m rpent.robots.components.pi05_vla_server --embodiment libero \
  --cuda-device "$GPU_SIM" --transport http --host 127.0.0.1 --port 18220 > "$SVC/vla.log" 2>&1 &
pids+=($!)
SAM3_CHECKPOINT_PATH=$W/checkpoints/sam3/sam3.pt \
  $V/bin/python -u -m rpent.robots.components.sam3_server \
  --cuda-device "$GPU_SIM" --transport http --host 127.0.0.1 --port 18114 > "$SVC/sam3.log" 2>&1 &
pids+=($!)

for i in $(seq 1 120); do
  if grep -q "Application startup complete" "$SVC/vllm.log" && grep -q "listening" "$SVC/vla.log" \
     && grep -q "listening" "$SVC/sam3.log"; then break; fi
  for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null || { echo "a service exited during startup"; tail -20 "$SVC"/*.log; exit 1; }; done
  sleep 10
done
grep -q "Application startup complete" "$SVC/vllm.log" || { echo "vLLM not ready after 20 min"; exit 1; }
echo "services ready at $(date -Is)"

cd "$REPO"
$V/bin/python -u bench/libero_pro/worker.py \
  --out-root "$OUT_ROOT" --slots "$SLOTS" --repo "$REPO" --rpent "$V/bin/rpent" \
  --llm-url http://127.0.0.1:8001/v1 --vla-url http://127.0.0.1:18220 --sam3-url http://127.0.0.1:18114 \
  --env-cuda-device "$GPU_SIM" --api-key-file $W/rpent-serve/api_key \
  --wandb-project rpent-libero-pro $WORKER_ARGS
