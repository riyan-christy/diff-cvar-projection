# A Differentiable CVaR Projection Primitive — experiment code

Reproducible code for the empirical sections of *A Differentiable CVaR
Projection Primitive for Risk-Constrained Optimization*.

**Certificate semantics (this revision).** The active-face certificate now
(i) merges a fully included boundary group (`q == g`, strict gap below) into
the strict set --- the classical no-tie face; averaging it was conservative,
not exact --- and (ii) takes the active/inactive decision from the forward
solve's violation status rather than re-deriving it from `z*`, which is
ambiguous exactly on the boundary.  Both are covered by new tests (5 and 6)
in `test_correctness.py` and documented in the paper.

**Contribution = the backward pass.** The forward projection onto the CVaR
(expected-shortfall) sublevel set `{ z : sum of k largest of z ≤ d }` is existing
literature (Roth & Cui 2025; Pan & Yan 2025; Luxenberg et al. 2026) and is
included only so the experiments are self-contained. What is new is the
**active-face certificate + O(m) vector–Jacobian product** (`vjp_projection`),
its grouped-plateau form, its torch/GPU batching, and its use inside a
differentiable CVQP layer.

## TL;DR — clone and run

On a fresh **1× A100 (40 GB SXM4)** with the **Lambda Stack 22.04** image:

```bash
git clone https://github.com/riyan-christy/diff-cvar-projection.git
cd diff-cvar-projection
bash reproduce.sh          # builds the byte-exact env, runs everything -> results/
```

`reproduce.sh` = `setup_lambda.sh` (isolated `.venv` from `requirements.lock.txt`)
+ `run_all.sh` (every experiment). Full paper — benchmarks, constraint-targeting
experiment, and training demo — is ≈ 23 A100-hours (≈ \$45 at \$1.99/h; the
reference full stage measured 78,355 s). For the sub-hour, ≈ \$2 benchmark-only
pass: `PRIMARY_STAGE=skip SECONDARY=0 bash run_all.sh`. Outputs land in `results/`
(CSVs, PNGs, `env.json`, `RESULTS.md`). No GPU? See *Quick start (CPU only)* below.

## Headline reproduced results (reference A100 run)

| claim | number |
|-------|--------|
| VJP correctness vs CLARABEL / finite differences | ≤ 3.5e-9 (no-tie **and** plateau) |
| GPU torch path vs numpy reference (self-check) | forward 7e-16, VJP 6e-17 |
| Differentiate a **200M-scenario** CVaR projection (1 A100) | 96 ms forward + 19 ms backward, 19.8 GB |
| Backward complexity | **O(m)**, VJP error ~1e-10 across m=10³…2×10⁸ |
| vs `cvxpylayers` epigraph differentiation | thousands× faster; baseline OOM/timeout at m=10⁵ (42 GB at 3×10⁴) |
| GPU vs CPU backward at m=10⁷ | ~**18×** (1.12 ms vs 20.4 ms); end-to-end fwd+bwd gap >2 orders of magnitude (CPU forward is sort-dominated) |
| Decision-focused CVQP layer | `gradcheck` passes; end-to-end matches two-stage within noise (Δ = −0.0025 ± 0.0062), both CVaR-feasible |
| Hard CVaR constraint vs calibrated penalty (5-seed screening) | realized/budget → 1.00 (ID) and 1.01 (shift) at m=10⁵; fixed-λ penalty stuck at 1.76× under shift; oracle-λ repairs the mean, not the dispersion |
| End-to-end training through the layer at m=10⁵ | 3 seeds, active-projection fraction 1.00, 5.8 s/step in 1.3 GB; epigraph negative control OOM-killed at 94 s |

Exact environment in `results/env.json`; exact package versions in `requirements.lock.txt`.

> **Provenance note / where the reference results live:** the numbers above
> are from the reference A100 run. This repository deliberately ships
> `results/` empty except `.gitkeep` (see `.gitignore`): `reproduce.sh`
> regenerates everything in place, and the complete reference snapshot —
> every CSV and figure behind the paper, plus `env.json` and `RESULTS.md` —
> is attached to the **tagged GitHub release** and included in the journal's
> **online supplement**. To compare a rerun against the reference, start with
> `results/RESULTS.md` and `results/primary/full/summary.csv`. A regenerated
> snapshot reflects whatever environment produced it (`results/env.json` says
> which) and may be partial — e.g. a CPU-only regeneration leaves the GPU
> sections "(not generated)" in `RESULTS.md`. Per-run CSVs are the source of
> truth.

---

## What's here

| file | what it does | needs |
|------|--------------|-------|
| `cvar_proj.py` | core library: exact forward projection, active-face certificate, O(m) VJP/JVP, batched numpy ops, **torch-native GPU projection + autograd Function**. Note: the `cvar_project()` convenience wrapper is the robust CPU-backed autograd path (host round-trip); the GPU-native no-plateau path is `get_CVaRProjectTorch()` | numpy (cvxpy for ground truth; torch optional) |
| `test_correctness.py` | §9.1 — VJP vs finite differences (no-tie + plateau), forward vs cvxpy, d/κ adjoints | numpy, cvxpy |
| `bench_scaling.py` | §9.1 — CPU forward/backward time + peak memory vs `m`, up to 1e7 | numpy, pandas, matplotlib |
| `bench_gpu.py` | §9.1 — **A100 GPU-native** projection+VJP scaling (to 2e8), batched throughput, CPU-vs-GPU; self-validates vs numpy first | + torch |
| `bench_headtohead.py` | §9.1 — proposed VJP vs `cvxpylayers` epigraph differentiation (baseline OOM/timeout) | + torch, cvxpylayers |
| `portfolio_e2e.py` | §9.2 — unrolled-ADMM CVQP layer; full-layer `gradcheck`; decision-focused training | + torch |
| `robustness.py` | §9.4 — heavy tails, plateaus, threshold sweep, gradient-vs-solve-error | numpy, cvxpy (torch optional) |
| `exp_data.py` | primary experiment: synthetic regime/shock-bank generator with named seed streams (spec-pinned defaults) | numpy |
| `exp_solvers.py` | primary experiment: batched-instance ADMM (hard CVQP, exact CVaR penalty, min-CVaR certificate, MV floor), plateau-robust projection with warm-started multiplier brackets, adaptive rho, lambda calibration | numpy (torch for GPU) |
| `exp_primary.py` | primary experiment runner: smoke/pilot/full/journal stages, append-only `solves.csv` with resume, blind kappa freeze, aggregation + headline figure | + pandas, matplotlib |
| `exp_secondary.py` | secondary demo: end-to-end training through the CVaR layer at m=1e5 (theta in operator form, VJP-use counters, gradient check) + epigraph negative control | + torch (cvxpylayers for the control) |
| `env_capture.py` | records HW/driver/CUDA/OS/pip into `results/env.json` | — |
| `make_results.py` | aggregates `results/*.csv` → `results/RESULTS.md` (paste-ready blocks) | pandas |
| `setup_lambda.sh` / `run_all.sh` / `Makefile` | one-command environment setup and full reproduction | — |

Results (CSVs + PNGs) are written to `results/` (git-ignored; regenerated by `run_all.sh`).

---

## Reproducing on Lambda Cloud (the reference run)

### Instance
- **1× A100 (40 GB SXM4)**, 30 vCPU, 200 GiB RAM, 0.5 TiB SSD — region **us-east-1**.
- The whole suite is single-GPU; no multi-node anything.

### Base image — **Lambda Stack 22.04**  ✅
Pick `Lambda Stack 22.04` at launch. Why:
- It is Lambda's **default, most-tested** image: NVIDIA driver + CUDA + cuDNN are
  preinstalled and verified against the A100, so there is **zero GPU/driver setup**.
- It is **Ubuntu 22.04 → Python 3.10**, the most reliable interpreter for this
  scientific stack. `cvxpylayers` pulls in **`diffcp`, which compiles from C++**;
  Python 3.10 has the broadest wheel/build coverage for it.
- Avoid the 24.04 images: they ship **Python 3.12** (riskier `diffcp`/`cvxpylayers`
  builds) and, as of Dec 2025, the Lambda Stack 24.04 / GPU Base 24.04 images have a
  known `apt full-upgrade` defect.
- `GPU Base 22.04` also works (we install our own pinned torch in a venv), but
  Lambda Stack 22.04 is the lowest-friction choice. Plain `Ubuntu 22.04` is not
  recommended (more manual driver/setup risk).

We do **not** rely on Lambda Stack's framework versions: `setup_lambda.sh` builds an
**isolated `.venv` with pinned deps** and records the exact lock, so the result is
reproducible regardless of what the base image ships.

### Launch tabs the UI will show
- **Filesystem (optional):** attach a small persistent filesystem (e.g. name it
  `cvar-data`, **same region us-east-1** — filesystems are region-locked) if you
  want `results/` to survive instance termination. Otherwise skip it and `scp` the
  results down before you terminate. Not required for correctness.
- **Security / SSH key:** select (or add) your SSH public key so you can log in.
  Keep the default firewall (inbound **SSH/22 only**). You do **not** need to open
  any other ports — nothing here serves a web UI. (Only open 8888 if you separately
  want remote Jupyter, which this repo doesn't use.)

### Steps
```bash
# 1) SSH in (user is `ubuntu` on Lambda)
ssh ubuntu@<instance-ip>

# 2) get the code and run everything
git clone https://github.com/riyan-christy/diff-cvar-projection.git
cd diff-cvar-projection
bash reproduce.sh

# 3) pull the results down (reviewers: this is all you need)
rsync -av ubuntu@<instance-ip>:diff-cvar-projection/results/ ./results/
```
Or run the two stages separately:
```bash
bash setup_lambda.sh          # isolated .venv from requirements.lock.txt (compiles diffcp; a few min)
source .venv/bin/activate
bash run_all.sh               # every experiment -> results/
```
`run_all.sh` writes all CSVs/PNGs, plus `results/env.json` (exact HW/driver) and
`results/RESULTS.md` (consolidated tables + ready-to-cite numbers).  The
constraint-targeting experiment runs `pilot` (freezes kappa, blind to method
separation) then `full` by default; set `PRIMARY_STAGE=journal` for the
10-seed rerun, `PRIMARY_STAGE=smoke` for a minutes-long CPU pipeline check, or
`PRIMARY_STAGE=skip` / `SECONDARY=0` to skip parts.  Every solve appends to
`results/primary/solves.csv` keyed by
`(seed, m, kappa, repeat, regime, split, method, instance, lambda)`, so an
interrupted run resumes without rework; aggregates are recomputed from the
CSV, never hand-edited.

> Developing on a local copy instead of cloning? `rsync` does **not** honor
> `.gitignore`, so sync with `--exclude results/ --exclude .venv` or a stale local
> `results/` will clobber real run outputs on the box.

### Time & cost (rough, at \$1.99/hr)
Setup ≈ 5 min. Correctness + CPU scaling + robustness ≈ 5–10 min. GPU scaling to
1e8 + batched throughput ≈ 5–15 min. The `cvxpylayers` head-to-head is **slow by
design** (the epigraph baseline is what we beat) — 15–40 min depending on the size
at which it OOMs. Portfolio ≈ 5–15 min (CPU-bound by design; its layer now
uses the vectorized plateau-robust projection — plateau inputs are generic
near a binding-CVaR optimum, and the earlier per-row fallback could stall this
stage for hours). **The Sec. 9 benchmark suite totals well
under an hour (≈ \$1–2).**

The **constraint-targeting experiment** (`exp_primary.py`, on by default with
`PRIMARY_STAGE=full`) is budgeted separately: the reference 5-seed full run
measured **21.8 A100-hours** (78,355 s, ≈ \$43; pilot included separately,
≈ 3 h), and the 10-seed `PRIMARY_STAGE=journal` rerun at the spec-full sample
sizes is budgeted at roughly twice the full stage (frozen protocol and
amendment log: `docs/EXPERIMENT_EXECUTION_SPEC.md`). The secondary training
demo (`exp_secondary.py`) adds ≈ 1 h, of which the epigraph negative control
is capped at 300 s — it is *expected* to end `killed(oom)`/`timeout` on the
reference box, and that outcome is recorded as data, not raised as an error.
Set
`PRIMARY_STAGE=skip SECONDARY=0` to reproduce only the benchmark sections.
Before the experiment stages on a fresh box, `python -u bench_solver.py
--device cuda` (~5-10 min) measures real per-solve cost and projects
per-stage hours; proceed only if `full` projects within budget.  The solver's
GPU inner loops use torch.compile/CUDA-graph capture (the bisections are
launch/sync-bound eagerly; capture is 10-30x there) --- set `CVAR_COMPILE=0`
to force eager.  The first CUDA solve per shape includes one-off compile
warmup; judge speed by the second.
`run_all.sh` streams output unbuffered and supports stage selection for
interrupted runs: `SKIP="bench_gpu bench_headtohead" bash run_all.sh` or
`ONLY="portfolio_e2e" bash run_all.sh` (tags = script basenames).
Remember to **terminate the instance** when done.

Override sizes if you want a quick pass or a bigger push:
```bash
GPU_SIZES="10000 100000 1000000 10000000" bash run_all.sh   # skip the 1e8 point
# push further WITHOUT overwriting the canonical sweep (--tag writes gpu_*_ext.csv):
python bench_gpu.py --sizes 100000000 200000000 --tag _ext   # 2e8 fits ~20 GB in float64
# NOTE: keep float64 (default). The exact active-face VJP needs the active set resolved
# to ~1e-6, which float32 cannot guarantee for O(10) values, so the self-check rejects
# float32 for the backward. float32 is fine for the forward/throughput only. float64
# covers sizes up to ~3.5e8 within 40 GB, which is past anything you need here.
```

---

## Quick start (laptop / CPU only)
You can validate the core claims with no GPU and no torch:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy cvxpy matplotlib pandas
python test_correctness.py     # forward vs cvxpy + VJP vs finite differences -> "ALL CORRECTNESS CHECKS PASSED"
python bench_scaling.py        # O(m) backward + peak memory, up to 1e7
python robustness.py           # ties/plateaus/threshold/solve-error stress
```

## Results → paper map
- **§9.1 projection microbenchmarks** — `test_correctness.py` (VJP correctness incl. plateaus),
  `bench_scaling.py` (CPU O(m), `scaling_*.png`), `bench_gpu.py` (A100 scaling to 2e8 + batched
  throughput, `gpu_*.png`), `bench_headtohead.py` (gradient agreement + epigraph OOM, `headtohead_*`).
- **§9.2 decision-focused portfolio** — `portfolio_e2e.py` (`portfolio.*`): full-layer gradcheck,
  forward validation vs cvxpy, warm-started end-to-end vs two-stage over multiple seeds, realized-CVaR
  feasibility. (Result: end-to-end matches two-stage within noise on this instance — the layer is
  correct, feasible, and trainable; it does not claim a decision-focused win.)
- **§9.4 robustness & failure analysis** — `robustness.py` (`robustness.csv`, `robustness_solveerr.png`).
- **§9.3 energy scheduling** — not implemented; future work.

## Notes / gotchas
- **Ground truth** uses **CLARABEL** at high accuracy. The cvxpy *default* solver can return
  slightly infeasible points on this QP and must not be used as ground truth.
- **Scaling instances are constructed no-plateau** (the generic, full-measure case): forward is
  one sort + O(m), backward is O(m). **Tie/plateau correctness is shown separately** at small `m`
  (where it is checked against CLARABEL and finite differences).
- **Finite-difference checks are restricted to certificate-stable cells.** The projection is
  piecewise affine; across a cell boundary it is nondifferentiable and a central difference is not
  a valid reference. `robustness.py` enforces this explicitly.
- `bench_gpu.py` **self-validates** the torch path against the numpy reference before timing, so a
  silent GPU kernel bug cannot masquerade as a result.

## Reproducibility artifacts
`requirements.lock.txt` (the exact package versions from the reference A100 run) is
committed; `setup_lambda.sh` installs from it, and a fresh run writes its own
`results/pip_freeze_thisrun.txt` + `results/env.json` (driver/CUDA/GPU) to diff
against it. Together these let a referee recreate the environment byte-for-byte.

## License & citation
MIT (`LICENSE`). If you use this code, cite the paper (`CITATION.cff`).
