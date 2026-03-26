#!/bin/bash
###############################################################################
# Parse and compare hipBLASLt benchmark results
#
# Usage:
#   ./parse_bench_results.sh [results_dir] [old_label] [new_label]
#
# Defaults:
#   results_dir = ./bench_results
#   old_label   = de5c1aebb6
#   new_label   = 393404632b
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${1:-$SCRIPT_DIR/bench_results}"
OLD_LABEL="${2:-de5c1aebb6}"
NEW_LABEL="${3:-393404632b}"

if [ ! -d "$RESULTS_DIR" ]; then
  echo "[ERROR] Results directory not found: $RESULTS_DIR"
  exit 1
fi

OLD_L="$OLD_LABEL" NEW_L="$NEW_LABEL" RDIR="$RESULTS_DIR" python3 << 'PYEOF'
import os

models = ['Llama-3.1-8B', 'Llama-3.1-70B', 'Llama-2-70B', 'DeepSeek-V2-lite', 'Mixtral-8x22B-proxy']
old_label = os.environ['OLD_L']
new_label = os.environ['NEW_L']
results_dir = os.environ['RDIR']

def parse_bench(filepath):
    results = []
    with open(filepath) as f:
        for line in f:
            s = line.strip()
            if s and s[0] in 'TN' and ',' in s:
                parts = [p.strip() for p in s.split(',')]
                if len(parts) >= 39:
                    results.append({
                        'transA': parts[0], 'transB': parts[1],
                        'm': int(parts[4]), 'n': int(parts[5]), 'k': int(parts[6]),
                        'gflops': float(parts[36]),
                        'gbps': float(parts[37]),
                        'us': float(parts[38]),
                    })
    return results

print("hipBLASLt Benchmark Comparison: {} (old) vs {} (new)".format(old_label, new_label))
print("Results directory: {}".format(results_dir))
print()

hdr = "{:<24s} {:<35s} {:>12s} {:>12s} {:>8s}".format(
    'Model', 'Shape (trans,M,N,K)', 'Old TFLOPS', 'New TFLOPS', 'Delta%')
print(hdr)
print('-' * len(hdr))

total_shapes = 0
regressions = []
improvements = []

for model in models:
    old_file = os.path.join(results_dir, "bench_{}_{}.csv".format(model, old_label))
    new_file = os.path.join(results_dir, "bench_{}_{}.csv".format(model, new_label))
    if not os.path.exists(old_file) or not os.path.exists(new_file):
        print("{:<24s} (missing results)".format(model))
        continue
    old_results = parse_bench(old_file)
    new_results = parse_bench(new_file)
    if not old_results and not new_results:
        print("{:<24s} (no benchmark data)".format(model))
        continue

    old_map = {(r['transA'], r['transB'], r['m'], r['n'], r['k']): r for r in old_results}
    new_map = {(r['transA'], r['transB'], r['m'], r['n'], r['k']): r for r in new_results}
    all_keys = sorted(set(old_map.keys()) | set(new_map.keys()), key=lambda x: (x[2], x[3], x[4]))

    for i, key in enumerate(all_keys):
        old_r = old_map.get(key)
        new_r = new_map.get(key)
        old_tf = old_r['gflops'] / 1000 if old_r else 0
        new_tf = new_r['gflops'] / 1000 if new_r else 0
        if old_tf > 0:
            delta = (new_tf - old_tf) / old_tf * 100
        else:
            delta = float('inf')
        shape = "{},{} M={} N={} K={}".format(key[0], key[1], key[2], key[3], key[4])
        mcol = model if i == 0 else ''
        print("{:<24s} {:<35s} {:>10.1f}T {:>10.1f}T {:>+7.1f}%".format(
            mcol, shape, old_tf, new_tf, delta))
        total_shapes += 1
        if delta < -3:
            regressions.append((model, shape, old_tf, new_tf, delta))
        elif delta > 3:
            improvements.append((model, shape, old_tf, new_tf, delta))
    print()

print("=" * len(hdr))
print("Summary: {} total shapes compared".format(total_shapes))
print()

if improvements:
    print("Improvements (>{:+.0f}%): {}".format(3, len(improvements)))
    for model, shape, old_tf, new_tf, delta in sorted(improvements, key=lambda x: -x[4]):
        print("  {:>+7.1f}%  {:<22s} {:<35s} {:>8.1f}T -> {:>8.1f}T".format(
            delta, model, shape, old_tf, new_tf))
    print()

if regressions:
    print("Regressions (<{:+.0f}%): {}".format(-3, len(regressions)))
    for model, shape, old_tf, new_tf, delta in sorted(regressions, key=lambda x: x[4]):
        print("  {:>+7.1f}%  {:<22s} {:<35s} {:>8.1f}T -> {:>8.1f}T".format(
            delta, model, shape, old_tf, new_tf))
    print()

neutral = total_shapes - len(improvements) - len(regressions)
print("Neutral (-3% to +3%): {}".format(neutral))
PYEOF
