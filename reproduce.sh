#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One command to reproduce the whole paper on a fresh Lambda A100 instance:
#
#   git clone <this repo> && cd <repo> && bash reproduce.sh
#
# It builds the byte-exact environment (from requirements.lock.txt) and then
# runs every experiment into results/ (CSVs, PNGs, env.json, RESULTS.md).
# Recommended box: 1x A100 (40 GB SXM4), Lambda Stack 22.04.
# FULL PAPER (benchmarks + constraint-targeting experiment + training demo):
#   ~23 A100-hours, ~$45 at $1.99/h. Use tmux.
# Benchmark-only pass (~1 h, ~$2):
#   PRIMARY_STAGE=skip SECONDARY=0 bash run_all.sh
# Note: the epigraph negative control and head-to-head rows are EXPECTED to
# end in a recorded timeout/OOM on the reference box; that is the experiment's
# result, not a failure -- the pipeline records it and continues.
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

bash setup_lambda.sh
# shellcheck disable=SC1091
source .venv/bin/activate
bash run_all.sh

echo ""
echo "============================================================"
echo "Reproduction complete. See results/RESULTS.md and results/*.png"
echo "============================================================"
