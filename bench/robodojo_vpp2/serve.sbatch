#!/bin/bash
# Nano4 services for RPent + VPP2 on RoboDojo (simulator runs on an RTX host):
#   GPU 0: Qwen3.8-27B-FP8 planner (vLLM)
#   GPU 1: VPP2 joint2b_s100000 (official XPolicyLab WS server, localhost only),
#          RPent XPolicyLab facade (PNG step streaming), SAM3
# Ports derive from the job id; the log prints the node and ports to forward.
#SBATCH --job-name=rpent-vpp2-serve
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn001
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=400G
#SBATCH --account=mst115550
#SBATCH --time=12:00:00
#SBATCH --output=/work/roboleon1295/vpp2/serve-%j.log
set -euo pipefail
W=/work/roboleon1295
V2=$W/vpp2
REPO=$W/RPent
XPL=$V2/xpl
BUNDLE=$V2/weights/checkpoints/joint2b_s100000
WAN=$V2/weights/checkpoints/Wan2.1-I2V-14B-480P
SVC=$V2/services/$SLURM_JOB_ID
mkdir -p "$SVC"
export HOME=$W/jobhome XDG_CACHE_HOME=$W/jobhome/.cache TRITON_CACHE_DIR=$W/jobhome/.triton
export HF_HOME=$W/hf HF_HUB_OFFLINE=1
read -r GPU_LLM GPU_POL <<<"${CUDA_VISIBLE_DEVICES//,/ }"
BASE=$((30000 + (SLURM_JOB_ID % 3000) * 10))
LLM_PORT=$BASE VPP2_PORT=$((BASE + 1)) VLA_PORT=$((BASE + 2)) SAM3_PORT=$((BASE + 3))
IP=$(getent hosts "$(hostname)" | cut -d' ' -f1)
echo "node=$(hostname) ip=$IP llm=$LLM_PORT vla=$VLA_PORT sam3=$SAM3_PORT vpp2_ws=$VPP2_PORT(local)"
echo "vpp2_repo=$(git -C $V2/video-prediction-policy-2 rev-parse HEAD) xpolicylab=$(git -C $XPL/XPolicyLab rev-parse HEAD) rpent=$(git -C $REPO rev-parse HEAD)"

# VPP2's install_adapter.sh step, into this job's own XPolicyLab clone.
[ -f "$XPL/XPolicyLab/policy/VPP2/deploy.yml" ] || cp -r "$V2/video-prediction-policy-2/policy/VPP2" "$XPL/XPolicyLab/policy/VPP2"

pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT

VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_API_KEY="$(cat $W/rpent-serve/api_key)" CUDA_VISIBLE_DEVICES=$GPU_LLM \
  $W/.venv-vllm-minicpm/bin/vllm serve $W/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --served-model-name Qwen/Qwen3.8-27B-FP8 --host 0.0.0.0 --port $LLM_PORT \
  --max-model-len 262144 --max-num-seqs 2 --gpu-memory-utilization 0.9 --dtype bfloat16 \
  --limit-mm-per-prompt '{"image":4096}' --mm-processor-cache-gb 32 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --enable-prompt-tokens-details > "$SVC/vllm.log" 2>&1 &
pids+=($!)

# Official VPP2 serving command (vpp2.cli server) with the documented RoboDojo settings.
(cd "$XPL" && CUDA_VISIBLE_DEVICES=$GPU_POL VPP2_TORCH_LOAD_MMAP=1 \
  PYTHONPATH="$V2/video-prediction-policy-2/src:$XPL:$XPL/XPolicyLab" \
  $W/.venv-vpp2/bin/python -u "$XPL/XPolicyLab/setup_policy_server.py" \
    --config_path "$XPL/XPolicyLab/policy/VPP2/deploy.yml" \
    --overrides port=$VPP2_PORT host=127.0.0.1 \
      checkpoint_path=$BUNDLE/action.pt dataset_stats_path=$BUNDLE/dataset_stats.json \
      wan_model_dir=$WAN sigma_shift=1.0 num_inference_steps=10 seed=1 \
      ckpt_name=joint2b_s100000 action_horizon=32 replan_steps=24 \
  > "$SVC/vpp2.log" 2>&1) &
pids+=($!)

SAM3_CHECKPOINT_PATH=$W/checkpoints/sam3/sam3.pt CUDA_VISIBLE_DEVICES=$GPU_POL \
  $W/.venv-rpent-liberopro/bin/python -u -m rpent.robots.components.sam3_server \
  --transport http --host 0.0.0.0 --port $SAM3_PORT > "$SVC/sam3.log" 2>&1 &
pids+=($!)

for i in $(seq 1 180); do
  if grep -q "Application startup complete" "$SVC/vllm.log" && (echo > /dev/tcp/127.0.0.1/$VPP2_PORT) 2>/dev/null; then break; fi
  for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null || { echo "a service exited during startup"; tail -30 "$SVC"/*.log; exit 1; }; done
  sleep 10
done
(echo > /dev/tcp/127.0.0.1/$VPP2_PORT) 2>/dev/null || { echo "VPP2 not ready after 30 min"; tail -30 "$SVC/vpp2.log"; exit 1; }

cd "$REPO"
PYTHONPATH="$XPL/XPolicyLab" $W/.venv-rpent-liberopro/bin/python -u -m rpent.robots.components.xpolicylab_vla_server \
  --task robodojo-diag --policy-server-url ws://127.0.0.1:$VPP2_PORT \
  --transport http --host 0.0.0.0 --port $VLA_PORT > "$SVC/facade.log" 2>&1 &
pids+=($!)
for i in $(seq 1 60); do grep -q "listening" "$SVC/facade.log" && break; sleep 5; done
grep -q "listening" "$SVC/facade.log" || { echo "facade not ready"; tail -30 "$SVC/facade.log"; exit 1; }
echo "services ready at $(date -Is)"
wait
