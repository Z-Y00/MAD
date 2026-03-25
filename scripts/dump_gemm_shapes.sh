#!/bin/bash
###############################################################################
# Dump unique GEMM shapes for hipBLASLt on Primus Megatron training
#
# Docker image: rocm/primus-training-private:20260323_v26dot1_rebuild_hipblaslt
# Models: Llama3.1 8B, Llama3.1 70B, Llama2 70B, DeepSeek-V2-lite, Mixtral 8x22B
#
# Uses HIPBLASLT_LOG_MASK=64 to dump unique GEMM shapes in YAML format.
# Uses HIPBLASLT_LOG_FILE to write shapes to a file (%i = PID).
# hipBLASLt log directory is bind-mounted to the host for direct file access.
# Uses default mbs/gbs from Primus YAML configs inside the container.
###############################################################################

set -euo pipefail

DOCKER_IMAGE="rocm/primus-training-private:20260323_v26dot1_rebuild_hipblaslt"
CONTAINER_NAME="gemm_shape_dump"
OUTPUT_DIR="$(pwd)/gemm_shapes"
TRAIN_ITERS=5
CONTAINER_LOG_DIR="/tmp/hipblaslt_logs"
HIPBLASLT_LOG_FILE_PATTERN="${CONTAINER_LOG_DIR}/hipblaslt_%i.log"
if [ -z "${HF_TOKEN:-}" ]; then
  echo "[ERROR] HF_TOKEN is not set. Gated models (Llama) need a HuggingFace token."
  echo "  export HF_TOKEN=<your_token>"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

MODELS=(
  "Llama-3.1-8B"
  "Llama-3.1-70B"
  "Llama-2-70B"
  "DeepSeek-V2-lite"
  "Mixtral-8x22B-proxy"
)

echo "=============================================="
echo " GEMM Shape Dump Script"
echo " Image: $DOCKER_IMAGE"
echo " Output: $OUTPUT_DIR"
echo "=============================================="

cleanup_container() {
  docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
}

build_train_cmd() {
  local model="$1"

  cat <<TRAIN_EOF
set -e

# --- GPU detection ---
DEVICE=\$(/opt/rocm/bin/rocminfo | grep "AMD Instinct" | head -n1 | awk '{print \$5}')
if [[ -z "\$DEVICE" || ("\$DEVICE" != "MI300X" && "\$DEVICE" != "MI355X") ]]; then
  ARCH=\$(/opt/rocm/bin/rocminfo | grep -o 'gfx942\|gfx950' | head -n 1 | tr -d '[:space:]')
  case "\$ARCH" in
    "gfx942") DEVICE="MI300X" ;;
    "gfx950") DEVICE="MI355X" ;;
    *) DEVICE="MI300X" ;;
  esac
fi
if [ "\$DEVICE" == "MI325X" ]; then CONFIG_DEVICE="MI300X"
elif [ "\$DEVICE" == "MI350X" ]; then CONFIG_DEVICE="MI355X"
else CONFIG_DEVICE="\$DEVICE"; fi
echo "Detected GPU: \$DEVICE -> Config device: \$CONFIG_DEVICE"

if [[ "\$DEVICE" == "MI300X" || "\$DEVICE" == "MI325X" ]]; then
  export PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=1
  export NVTE_CK_IS_V3_ATOMIC_FP32=1
fi

export MOCK_DATA=1
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1

echo "HIPBLASLT_LOG_MASK=\$HIPBLASLT_LOG_MASK"
echo "HIPBLASLT_LOG_FILE=\$HIPBLASLT_LOG_FILE"

cd /workspace/Primus

case "$model" in
  Llama-3.1-8B)
    EXP=examples/megatron/configs/\$CONFIG_DEVICE/llama3.1_8B-BF16-pretrain.yaml
    echo "[INFO] Running $model with config: \$EXP"
    grep -E 'micro_batch_size|global_batch_size' \$EXP
    bash runner/primus-cli direct \
      --log_file /tmp/primus_${model}.log \
      -- train pretrain \
      --config \$EXP \
      --train_iters $TRAIN_ITERS
    ;;
  Llama-3.1-70B)
    EXP=examples/megatron/configs/\$CONFIG_DEVICE/llama3.1_70B-BF16-pretrain.yaml
    echo "[INFO] Running $model with config: \$EXP"
    grep -E 'micro_batch_size|global_batch_size' \$EXP
    bash runner/primus-cli direct \
      --log_file /tmp/primus_${model}.log \
      -- train pretrain \
      --config \$EXP \
      --train_iters $TRAIN_ITERS
    ;;
  Llama-2-70B)
    EXP=examples/megatron/configs/\$CONFIG_DEVICE/llama2_70B-BF16-pretrain.yaml
    echo "[INFO] Running $model with config: \$EXP"
    grep -E 'micro_batch_size|global_batch_size' \$EXP
    bash runner/primus-cli direct \
      --log_file /tmp/primus_${model}.log \
      -- train pretrain \
      --config \$EXP \
      --train_iters $TRAIN_ITERS
    ;;
  DeepSeek-V2-lite)
    EXP=examples/megatron/configs/\$CONFIG_DEVICE/deepseek_v2_lite-BF16-pretrain.yaml
    echo "[INFO] Running $model with config: \$EXP"
    grep -E 'micro_batch_size|global_batch_size' \$EXP
    bash runner/primus-cli direct \
      --log_file /tmp/primus_${model}.log \
      -- train pretrain \
      --config \$EXP \
      --train_iters $TRAIN_ITERS
    ;;
  Mixtral-8x22B-proxy)
    EXP=examples/megatron/configs/\$CONFIG_DEVICE/mixtral_8x22B_v0.1-BF16-pretrain.yaml
    if [[ "\$CONFIG_DEVICE" == "MI355X" ]]; then
      MBS=2; GBS=16
    else
      MBS=1; GBS=16
    fi
    echo "[INFO] Running $model (proxy, 4 layers) with config: \$EXP, MBS=\$MBS, GBS=\$GBS"
    bash runner/primus-cli direct \
      --log_file /tmp/primus_${model}.log \
      -- train pretrain \
      --config \$EXP \
      --num_layers 4 \
      --pipeline_model_parallel_size 1 \
      --micro_batch_size \$MBS \
      --global_batch_size \$GBS \
      --train_iters $TRAIN_ITERS
    ;;
  *)
    echo "[ERROR] Unknown model: $model"
    exit 1
    ;;
esac
TRAIN_EOF
}

echo ""
echo "[STEP 0] Ensuring Docker image is available: $DOCKER_IMAGE"
docker pull "$DOCKER_IMAGE" 2>/dev/null || echo "[INFO] Using locally cached image"

for model in "${MODELS[@]}"; do
  echo ""
  echo "=============================================="
  echo " Dumping GEMM shapes for: $model"
  echo "=============================================="

  cleanup_container

  OUTPUT_FILE="$OUTPUT_DIR/gemm_shapes_${model}.yaml"
  TRAIN_CMD="$(build_train_cmd "$model")"

  # Per-model host directory for hipBLASLt raw logs, mounted into container
  HOST_LOG_DIR="$OUTPUT_DIR/raw_logs_${model}"
  rm -rf "$HOST_LOG_DIR"
  mkdir -p "$HOST_LOG_DIR"

  docker run \
    --rm \
    --name "$CONTAINER_NAME" \
    --device /dev/dri \
    --device /dev/kfd \
    --network host \
    --ipc host \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    --shm-size 128G \
    -e HIPBLASLT_LOG_MASK=64 \
    -e HIPBLASLT_LOG_FILE="$HIPBLASLT_LOG_FILE_PATTERN" \
    -e HF_TOKEN="$HF_TOKEN" \
    -v "$HOST_LOG_DIR":"$CONTAINER_LOG_DIR" \
    "$DOCKER_IMAGE" \
    bash -c "$TRAIN_CMD"

  # Deduplicate per-PID logs into a single output file
  shopt -s nullglob
  log_files=("$HOST_LOG_DIR"/hipblaslt_*.log)
  shopt -u nullglob
  if [ "${#log_files[@]}" -gt 0 ]; then
    cat "${log_files[@]}" | sort -u > "$OUTPUT_FILE"
  else
    : > "$OUTPUT_FILE"
  fi

  shape_count=$(wc -l < "$OUTPUT_FILE")
  echo "[DONE] $model -> $OUTPUT_FILE ($shape_count unique shapes)"

  # Clean up raw per-PID logs
  rm -rf "$HOST_LOG_DIR"
  echo ""
done

echo "=============================================="
echo " All GEMM shapes dumped to: $OUTPUT_DIR/"
echo "=============================================="
ls -la "$OUTPUT_DIR"
