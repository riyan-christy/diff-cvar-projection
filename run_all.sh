#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reproduce every empirical result in the paper, into results/.
# Run inside the venv created by setup_lambda.sh:
#   source .venv/bin/activate && bash run_all.sh
#
# Override scaling sizes, e.g.:  MAX_M=100000000 bash run_all.sh
# ---------------------------------------------------------------------------
set -euo pipefail
mkdir -p results results/logs
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
# name each log after the script being run (arg 2), not "python" (arg 1)
run() { local tag; tag="$(basename "${2:-$1}" .py)"; echo "==> [$(ts)] $*"; "$@" 2>&1 | tee "results/logs/${tag}.log"; }

# Sizes: CPU numpy reference goes to 1e7; the A100 GPU path can push to 1e8.
CPU_SIZES="${CPU_SIZES:-1000 3000 10000 30000 100000 300000 1000000 3000000 10000000}"
GPU_SIZES="${GPU_SIZES:-10000 100000 1000000 10000000 100000000}"

run python env_capture.py
run python test_correctness.py                       # §9.1 correctness (numpy+cvxpy)
run python bench_scaling.py $CPU_SIZES               # §9.1 CPU scaling (O(m), peak mem)
run python bench_gpu.py --sizes $GPU_SIZES           # §9.1 A100 GPU-native scaling + batched throughput
run python bench_headtohead.py                       # §9.1 vs cvxpylayers epigraph baseline
run python portfolio_e2e.py                          # §9.2 decision-focused portfolio
run python robustness.py                             # §9.4 ties/plateaus/tolerance/early-stop
run python make_results.py                           # aggregate -> results/RESULTS.md

echo "==> [$(ts)] DONE. See results/ (CSVs, PNGs) and results/RESULTS.md"

