#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One command to reproduce the whole paper on a fresh Lambda A100 instance:
#
#   git clone <this repo> && cd <repo> && bash reproduce.sh
#
# It builds the byte-exact environment (from requirements.lock.txt) and then
# runs every experiment into results/ (CSVs, PNGs, env.json, RESULTS.md).
# Recommended box: 1x A100 (40 GB SXM4), Lambda Stack 22.04. ~30-60 min, ~$1-2.
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

