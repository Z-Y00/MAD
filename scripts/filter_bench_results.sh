#!/bin/bash
###############################################################################
# Filter hipBLASLt benchmark CSV files to only include regressed shapes.
#
# Reads regressed_shapes_<model>.yaml files from gemm_shapes/ to determine
# which (transA, transB, M, N, K) tuples are regressed, then extracts the
# matching blocks from each bench CSV, preserving the preamble header.
#
# Usage:
#   ./filter_bench_results.sh [results_dir] [old_label] [new_label]
#
# Defaults:
#   results_dir = ./bench_results
#   old_label   = de5c1aebb6
#   new_label   = 393404632b
#
# Output:
#   bench_results/filtered_bench_<model>_<label>.csv  (one per model per label)
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${1:-$SCRIPT_DIR/bench_results}"
OLD_LABEL="${2:-de5c1aebb6}"
NEW_LABEL="${3:-393404632b}"
SHAPES_DIR="$SCRIPT_DIR/gemm_shapes"

if [ ! -d "$RESULTS_DIR" ]; then
  echo "[ERROR] Results directory not found: $RESULTS_DIR"
  exit 1
fi

SDIR="$SHAPES_DIR" RDIR="$RESULTS_DIR" OLD_L="$OLD_LABEL" NEW_L="$NEW_LABEL" python3 << 'PYEOF'
import os, re

shapes_dir = os.environ['SDIR']
results_dir = os.environ['RDIR']
old_label = os.environ['OLD_L']
new_label = os.environ['NEW_L']

models = ['Llama-3.1-8B', 'Llama-3.1-70B', 'Llama-2-70B', 'DeepSeek-V2-lite', 'Mixtral-8x22B-proxy']

def parse_regressed_shapes(yaml_path):
    """Extract (transA, transB, M, N, K) tuples from a regressed shapes YAML."""
    shapes = set()
    if not os.path.exists(yaml_path):
        return shapes
    with open(yaml_path) as f:
        for line in f:
            m_m = re.search(r'\bM:\s*(\d+)', line)
            m_n = re.search(r'\bN:\s*(\d+)', line)
            m_k = re.search(r'\bK:\s*(\d+)', line)
            m_ta = re.search(r'\btransA:\s*([TN])', line)
            m_tb = re.search(r'\btransB:\s*([TN])', line)
            if all([m_m, m_n, m_k, m_ta, m_tb]):
                shapes.add((m_ta.group(1), m_tb.group(1),
                            int(m_m.group(1)), int(m_n.group(1)), int(m_k.group(1))))
    return shapes

def parse_csv_data_key(line):
    """Extract (transA, transB, M, N, K) from a bench CSV data line."""
    s = line.strip()
    if not s or s[0] not in 'TN' or ',' not in s:
        return None
    parts = [p.strip() for p in s.split(',')]
    if len(parts) < 7:
        return None
    try:
        return (parts[0], parts[1], int(parts[4]), int(parts[5]), int(parts[6]))
    except (ValueError, IndexError):
        return None

def filter_csv(csv_path, regressed_keys, out_path):
    """Filter a bench CSV to only include blocks matching regressed shapes."""
    if not os.path.exists(csv_path):
        print("  [SKIP] {} not found".format(csv_path))
        return

    with open(csv_path) as f:
        lines = f.readlines()

    preamble = []
    blocks = []
    current_block = []
    in_preamble = True

    for line in lines:
        stripped = line.strip()
        if in_preamble:
            if stripped.startswith('Rotating buffer'):
                in_preamble = False
                current_block = [line]
            else:
                preamble.append(line)
        else:
            current_block.append(line)
            key = parse_csv_data_key(stripped)
            if key is not None:
                if key in regressed_keys:
                    blocks.append(current_block[:])
                current_block = []
            elif stripped.startswith('Rotating buffer') and len(current_block) > 1:
                current_block = [line]

    with open(out_path, 'w') as f:
        for line in preamble:
            f.write(line)
        for block in blocks:
            for line in block:
                f.write(line)

    print("  [OK] {} ({} shapes)".format(out_path, len(blocks)))

total = 0
for model in models:
    yaml_path = os.path.join(shapes_dir, "regressed_shapes_{}.yaml".format(model))
    regressed_keys = parse_regressed_shapes(yaml_path)
    if not regressed_keys:
        continue

    print("{}  ({} regressed shapes)".format(model, len(regressed_keys)))
    for label in [old_label, new_label]:
        csv_in = os.path.join(results_dir, "bench_{}_{}.csv".format(model, label))
        csv_out = os.path.join(results_dir, "filtered_bench_{}_{}.csv".format(model, label))
        filter_csv(csv_in, regressed_keys, csv_out)
    total += len(regressed_keys)

print("\nDone. Filtered {} regressed shapes across {} models.".format(
    total, sum(1 for m in models if os.path.exists(
        os.path.join(shapes_dir, "regressed_shapes_{}.yaml".format(m))))))
PYEOF
