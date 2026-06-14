PY ?= $(shell command -v python >/dev/null 2>&1 && echo python || echo python3)
# Convenience targets. Activate the venv first: source .venv/bin/activate
.PHONY: help setup correctness scaling gpu headtohead portfolio robustness results all clean

help:
	@echo "make setup        - create .venv and install pinned deps (run via: bash setup_lambda.sh)"
	@echo "make correctness  - VJP/forward correctness vs cvxpy + finite differences (CPU)"
	@echo "make scaling      - CPU O(m) time/memory scaling up to 1e7"
	@echo "make gpu          - A100 GPU-native projection+VJP scaling and batched throughput"
	@echo "make headtohead   - proposed VJP vs cvxpylayers epigraph differentiation"
	@echo "make portfolio    - decision-focused portfolio (end-to-end vs two-stage)"
	@echo "make robustness   - ties/plateaus/threshold sweep/grad-vs-tolerance/early-stop"
	@echo "make all          - everything (same as run_all.sh)"

setup:        ; bash setup_lambda.sh
correctness:  ; $(PY) test_correctness.py
scaling:      ; $(PY) bench_scaling.py
gpu:          ; $(PY) bench_gpu.py
headtohead:   ; $(PY) bench_headtohead.py
portfolio:    ; $(PY) portfolio_e2e.py
robustness:   ; $(PY) robustness.py
primary-smoke:   ; $(PY) exp_primary.py --stage smoke
primary-pilot:   ; $(PY) exp_primary.py --stage pilot --device cuda
primary:         ; $(PY) exp_primary.py --stage full --device cuda
primary-journal: ; $(PY) exp_primary.py --stage journal --device cuda
secondary:       ; $(PY) exp_secondary.py --device cuda
results:      ; $(PY) make_results.py
all:          ; bash run_all.sh

clean:
	rm -rf results/*.png results/*.csv results/logs results/env.json results/RESULTS.md __pycache__
