#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reproduce every empirical result in the paper, into results/.
# Run inside the venv created by setup_lambda.sh:
#   source .venv/bin/activate && bash run_all.sh
#
# Override scaling sizes, e.g.:  MAX_M=100000000 bash run_all.sh
# ---------------------------------------------------------------------------
set -euo pipefail
PY="${PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY=python3   # portability: some hosts lack `python`
mkdir -p results results/logs
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
# name each log after the first *.py argument; stream output line-by-line.
# SKIP="tag1 tag2" skips stages by tag; ONLY="tag" runs just those stages.
run() {
  local tag="" a
  for a in "$@"; do case "$a" in *.py) tag="$(basename "$a" .py)"; break;; esac; done
  [ -z "$tag" ] && tag="$(basename "$1")"
  case " ${SKIP:-} " in *" $tag "*) echo "==> [$(ts)] SKIP $tag"; return 0;; esac
  if [ -n "${ONLY:-}" ]; then
    case " $ONLY " in *" $tag "*) : ;; *) echo "==> [$(ts)] (not in ONLY) skip $tag"; return 0;; esac
  fi
  echo "==> [$(ts)] $*"
  "$@" 2>&1 | tee "results/logs/${tag}.log"
}

# Sizes: CPU numpy reference goes to 1e7; the A100 GPU path can push to 1e8.
CPU_SIZES="${CPU_SIZES:-1000 3000 10000 30000 100000 300000 1000000 3000000 10000000}"
GPU_SIZES="${GPU_SIZES:-10000 100000 1000000 10000000 100000000 200000000}"  # 2e8 = the paper's largest point

run "$PY" -u env_capture.py
run "$PY" -u test_correctness.py                       # §9.1 correctness (numpy+cvxpy)
run "$PY" -u bench_scaling.py $CPU_SIZES               # §9.1 CPU scaling (O(m), peak mem)
run "$PY" -u bench_gpu.py --sizes $GPU_SIZES           # §9.1 A100 GPU-native scaling + batched throughput
run "$PY" -u bench_headtohead.py                       # §9.1 vs cvxpylayers epigraph baseline
run "$PY" -u portfolio_e2e.py                          # §9.2 decision-focused portfolio
run "$PY" -u robustness.py                             # §9.4 ties/plateaus/tolerance/early-stop

# ---- primary constraint-targeting experiment (EXPERIMENT_EXECUTION_SPEC.md)
# Stages: pilot (freezes kappa) -> full -> optional journal rerun.
# PRIMARY_STAGE=skip disables; smoke is the tiny CPU pipeline check.
PRIMARY_STAGE="${PRIMARY_STAGE:-full}"
DEVICE="${DEVICE:-cuda}"
if [ "$PRIMARY_STAGE" != "skip" ]; then
  run "$PY" -u exp_primary.py --self-check --device "$DEVICE"
  run "$PY" -u exp_primary.py --stage smoke
  if [ "$PRIMARY_STAGE" != "smoke" ]; then
    run "$PY" -u exp_primary.py --stage pilot --device "$DEVICE"
    run "$PY" -u exp_primary.py --stage "$PRIMARY_STAGE" --device "$DEVICE"
  fi
fi

# ---- secondary capability demo (m=1e5 end-to-end training + neg. control)
SECONDARY="${SECONDARY:-1}"
if [ "$SECONDARY" = "1" ]; then
  run "$PY" -u exp_secondary.py --device "$DEVICE"
fi

run "$PY" -u make_results.py                           # aggregate -> results/RESULTS.md

echo "==> [$(ts)] DONE. See results/ (CSVs, PNGs) and results/RESULTS.md"
