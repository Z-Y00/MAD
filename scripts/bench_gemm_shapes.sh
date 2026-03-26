#!/bin/bash
###############################################################################
# Benchmark hipBLASLt GEMM shapes across two commits
#
# Old commit: de5c1aebb6 [rocm/primus-training-private:20260323_v26dot1_rebuild_hipblaslt]
# New commit: 393404632b [rocm/mad-private:primus_ci_c313e0c_20260318]
#
# Runs hipblaslt-bench --yaml on the dumped GEMM shapes for each model
# and compares performance between the two hipBLASLt versions.
###############################################################################

set -euo pipefail

OLD_IMAGE="rocm/primus-training-private:20260323_v26dot1_rebuild_hipblaslt"
OLD_LABEL="de5c1aebb6"
NEW_IMAGE="rocm/mad-private:primus_ci_c313e0c_20260318"
NEW_LABEL="393404632b"

SHAPES_DIR="$(cd "$(dirname "$0")/gemm_shapes" && pwd)"
RESULTS_DIR="$(cd "$(dirname "$0")" && pwd)/bench_results"
CLEAN_DIR="$RESULTS_DIR/clean_yaml"

mkdir -p "$RESULTS_DIR" "$CLEAN_DIR"

MODELS=(
  "Llama-3.1-8B"
  "Llama-3.1-70B"
  "Llama-2-70B"
  "DeepSeek-V2-lite"
  "Mixtral-8x22B-proxy"
)

DOCKER_COMMON=(
  --rm
  --device /dev/dri --device /dev/kfd
  --network host --ipc host
  --group-add video --cap-add SYS_PTRACE
  --security-opt seccomp=unconfined --privileged
  --shm-size 128G
)

sanitize_yaml() {
  local src="$1" dst="$2"
  python3 -c "
import re, sys
with open(sys.argv[1]) as f:
    lines = f.readlines()
with open(sys.argv[2], 'w') as out:
    for line in lines:
        m = re.match(r'.*function:\s*(\w+),\s*M:\s*(\d+),\s*N:\s*(\d+),\s*K:\s*(\d+),\s*lda:\s*(\d+),\s*ldb:\s*(\d+),\s*ldc:\s*(\d+),\s*ldd:\s*(\d+),\s*stride_a:\s*(\d+),\s*stride_b:\s*(\d+),\s*stride_c:\s*(\d+),\s*stride_d:\s*(\d+),\s*alpha:\s*([\d.]+),\s*beta:\s*([\d.]+),\s*transA:\s*(\w+),\s*transB:\s*(\w+),\s*batch_count:\s*(\d+).*a_type:\s*(\w+),\s*b_type:\s*(\w+),\s*c_type:\s*(\w+),\s*d_type:\s*(\w+).*compute_type:\s*\w+', line)
        if m:
            g = m.groups()
            out.write('- {{ function: {}, M: {}, N: {}, K: {}, lda: {}, ldb: {}, ldc: {}, ldd: {}, stride_a: {}, stride_b: {}, stride_c: {}, stride_d: {}, alpha: {}, beta: {}, transA: {}, transB: {}, batch_count: {}, a_type: {}, b_type: {}, c_type: {}, d_type: {}, compute_type: c_f32_r, scale_type: f32_r, rotating: 512, flush: true, iters: 1000, cold_iters: 1000, initialization: trig_float, use_gpu_timer: 1 }}\n'.format(*g))
" "$src" "$dst"
}

run_bench() {
  local image="$1" label="$2" yaml="$3" out="$4"
  echo "  [bench] $label on $(basename "$yaml" .yaml)"
  docker run "${DOCKER_COMMON[@]}" \
    -v "$CLEAN_DIR":/shapes:ro \
    "$image" \
    bash -c "
      /usr/bin/python3 -c 'import yaml' 2>/dev/null || \
        apt-get update -qq && apt-get install -y -qq python3-yaml 2>/dev/null || \
        /usr/bin/python3 -m pip install pyyaml -q --break-system-packages 2>/dev/null || \
        /usr/bin/python3 -m pip install pyyaml -q 2>/dev/null
      hipblaslt-bench --yaml /shapes/$(basename "$yaml")
    " > "$out" 2>&1
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=============================================="
echo " hipBLASLt Benchmark Comparison"
echo " Old: $OLD_LABEL ($OLD_IMAGE)"
echo " New: $NEW_LABEL ($NEW_IMAGE)"
echo " Shapes: $SHAPES_DIR"
echo " Results: $RESULTS_DIR"
echo "=============================================="

echo ""
echo "[STEP 1] Sanitizing YAML files..."
for model in "${MODELS[@]}"; do
  src="$SHAPES_DIR/gemm_shapes_${model}.yaml"
  dst="$CLEAN_DIR/gemm_shapes_${model}.yaml"
  if [ -s "$src" ]; then
    sanitize_yaml "$src" "$dst"
    echo "  $model: $(wc -l < "$dst") shapes"
  else
    echo "  $model: (empty, skipping)"
  fi
done

echo ""
echo "[STEP 2] Running benchmarks..."
for model in "${MODELS[@]}"; do
  yaml="$CLEAN_DIR/gemm_shapes_${model}.yaml"
  [ ! -s "$yaml" ] && echo "  Skipping $model (no shapes)" && continue

  echo ""
  echo "--- $model ---"
  run_bench "$OLD_IMAGE" "$OLD_LABEL" "$yaml" "$RESULTS_DIR/bench_${model}_${OLD_LABEL}.csv"
  run_bench "$NEW_IMAGE" "$NEW_LABEL" "$yaml" "$RESULTS_DIR/bench_${model}_${NEW_LABEL}.csv"
done

echo ""
echo "[STEP 3] Comparing results..."
echo ""
bash "$SCRIPT_DIR/parse_bench_results.sh" "$RESULTS_DIR" "$OLD_LABEL" "$NEW_LABEL"

echo ""
echo "=============================================="
echo " Results saved to: $RESULTS_DIR"
echo "=============================================="
ls -la "$RESULTS_DIR"/*.csv 2>/dev/null
