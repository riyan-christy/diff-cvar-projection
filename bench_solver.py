"""
bench_solver.py
===============
Measure the batched-ADMM solver's real cost on THIS machine before committing
to the pilot/full experiment stages (~5-10 min total).  Prints per-solve wall
time, iteration counts, statuses, and projected per-cell / per-stage costs so
the go/no-go decision is data, not hope.

Usage:
    python -u bench_solver.py --device cuda          # the A100 question
    CVAR_COMPILE=0 python -u bench_solver.py --device cuda   # eager baseline
    python -u bench_solver.py --device cpu --max-m 10000     # numpy reference

Notes:
  * With CVAR_COMPILE=1 (default) the first CUDA solve triggers torch.compile
    / CUDA-graph capture (tens of seconds, once per shape bucket); the SECOND
    solve is the steady-state number that matters.
  * A full-experiment cell is ~35 batched solves (hard val/test x regimes,
    certificates, the lambda searches).  Cells per stage: pilot 12,
    full 80/40/20 for m=1e3/1e4/1e5, journal doubles N.
"""
import argparse, time
import numpy as np

import exp_data as D
import exp_solvers as S

CELL_SOLVES = 35
CELLS = {"pilot": {1000: 8, 10000: 4},
         "full": {1000: 40, 10000: 20, 100000: 5}}   # kappa_2 off at 1e5


SOLVE_KW = dict(eps_abs=1e-4, eps_rel=1e-4, max_iter=12000)  # = stage cfg


def sweep_rho0(device, m=10000, N=64, beta=0.99, gamma=1.0, w_max=0.20):
    """Empirical rho0 selection: the iteration counts in the 2026-06-10 bench
    grew ~with sqrt(m), tracking rho0 = max(gamma,.1)/sqrt(m); this sweep
    finds the best m-scaling by measurement."""
    struct = D.make_structure(0, n=50, K=5)
    L = D.make_shock_bank(struct, "id", m, 0, 0, 1)
    mus = D.make_mus(struct, "id", N, 0, 0, 1)
    ops = S.make_ops(device)
    mc = S.solve_min_cvar(L, mus[:16], beta=beta, w_max=w_max, ops=ops,
                          eps_abs=1e-4, eps_rel=1e-4, max_iter=1500)
    lo_k = float(np.median(S.predicted_cvar(mc["W"], L, mus[:16], beta)))
    W_f = S.solve_mv_floor(mus[:16], gamma=gamma, w_max=w_max)
    hi_k = float(np.median(S.predicted_cvar(W_f, L, mus[:16], beta)))
    kappa = lo_k + 0.5 * (hi_k - lo_k)
    print(f"rho0 sweep at m={m}, N={N}, eps=1e-6 (hard + penalty):",
          flush=True)
    for rho0 in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        t0 = time.perf_counter()
        r = S.solve_hard(L, mus, kappa, beta=beta, gamma=gamma, w_max=w_max,
                         ops=ops, rho0=rho0, eps_abs=1e-6, eps_rel=1e-6,
                         max_iter=8000)
        th = time.perf_counter() - t0
        t0 = time.perf_counter()
        rp = S.solve_penalty(L, mus, 1.5, beta=beta, gamma=gamma,
                             w_max=w_max, ops=ops, rho0=rho0, eps_abs=1e-6,
                             eps_rel=1e-6, max_iter=8000)
        tp = time.perf_counter() - t0
        print(f"   rho0={rho0:6.3f}: hard {th:6.1f}s iters(med/max)="
              f"{int(np.median(r['iters']))}/{int(r['iters'].max())} "
              f"conv={int((r['status']=='converged').sum())}/{N} "
              f"rho_fin={np.median(r['rho']):.4f} | "
              f"pen {tp:6.1f}s iters={int(np.median(rp['iters']))}/"
              f"{int(rp['iters'].max())} "
              f"conv={int((rp['status']=='converged').sum())}/{N}",
              flush=True)


def bench_one(m, N, device, beta=0.99, gamma=1.0, w_max=0.20, reps=2):
    struct = D.make_structure(0, n=50, K=5)
    L = D.make_shock_bank(struct, "id", m, 0, 0, 1)
    mus = D.make_mus(struct, "id", N, 0, 0, 1)
    ops = S.make_ops(device)
    # cheap binding kappa: midpoint of [min-CVaR, floor-CVaR] medians
    mc = S.solve_min_cvar(L, mus[:16], beta=beta, w_max=w_max, ops=ops,
                          eps_abs=1e-4, eps_rel=1e-4, max_iter=1500)
    lo_k = float(np.median(S.predicted_cvar(mc["W"], L, mus[:16], beta)))
    W_f = S.solve_mv_floor(mus[:16], gamma=gamma, w_max=w_max)
    hi_k = float(np.median(S.predicted_cvar(W_f, L, mus[:16], beta)))
    kappa = lo_k + 0.5 * (hi_k - lo_k)

    rows = []
    for rep in range(reps):
        t0 = time.perf_counter()
        r = S.solve_hard(L, mus, kappa, beta=beta, gamma=gamma, w_max=w_max,
                         ops=ops, **SOLVE_KW)
        wall = time.perf_counter() - t0
        conv = int((r["status"] == "converged").sum())
        rows.append((wall, int(r["iters"].max()), conv))
        tag = "(incl. compile warmup)" if rep == 0 else "(steady state)"
        print(f"   hard  m={m:>7} N={N}  rep{rep}: {wall:8.1f}s  "
              f"iters(med/max)={int(np.median(r['iters']))}/"
              f"{int(r['iters'].max()):>5}  converged={conv}/{N}  "
              f"rho_fin(med)={float(np.median(r['rho'])):.4f}  "
              f"{wall / max(int(r['iters'].max()), 1) * 1e3:7.1f} ms/iter "
              f"{tag}", flush=True)
    t0 = time.perf_counter()
    rp = S.solve_penalty(L, mus, 1.5, beta=beta, gamma=gamma, w_max=w_max,
                         ops=ops, **SOLVE_KW)
    wallp = time.perf_counter() - t0
    print(f"   pen   m={m:>7} N={N}:      {wallp:8.1f}s  "
          f"max_iters={int(rp['iters'].max()):>5}  "
          f"converged={int((rp['status'] == 'converged').sum())}/{N}",
          flush=True)
    steady = rows[-1][0]
    # structured cell model (replaces 35 x (hard+pen)/2, which ignored the
    # cheap calibration solves): 2 hard val + 4 loose certs + ~20 calibration
    # solves (eps 1e-3, max_iter 3000, <=48 of 128 instances) + 2 hard test
    # (xN_test/N) + 3 penalty test (xN_test/N)
    pen_it = max(int(rp["iters"].max()), 1)
    cal_one = wallp * min(3000.0 / pen_it, 1.0) * (48.0 / 128.0)
    ntest_scale = 192.0 / 128.0
    cell_s = (2 * steady + 4 * 0.25 * wallp + 20 * cal_one
              + 2 * ntest_scale * steady + 3 * ntest_scale * wallp)
    print(f"   -> projected per-cell cost at m={m}: ~{cell_s/60:.1f} min "
          f"(structured model; cal solve ~{cal_one:.0f}s)", flush=True)
    return cell_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-m", type=int, default=100000)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--sweep-rho0", action="store_true",
                    help="rho0 grid at m=1e4 first (~10 min); run this "
                         "when iteration counts look pathological")
    args = ap.parse_args()
    if args.sweep_rho0:
        sweep_rho0(args.device)
        return
    import exp_solvers as S_
    print(f"device={args.device}  CVAR_COMPILE="
          f"{'on' if S_._compile_enabled() else 'off'}", flush=True)
    cell = {}
    for m in (1000, 10000, 100000):
        if m > args.max_m:
            continue
        print(f"--- m={m} ---", flush=True)
        cell[m] = bench_one(m, args.N, args.device)
    print("\n=== projected stage costs (hard+penalty mix, rough) ===",
          flush=True)
    for stage, mix in CELLS.items():
        tot = sum(cell.get(m, float('nan')) * c for m, c in mix.items()
                  if m in cell)
        known = all(m in cell for m in mix)
        print(f"   {stage:7}: ~{tot/3600:.1f} h"
              + ("" if known else "  (missing m rows excluded)"), flush=True)
    print("\nDecision rule: proceed if 'full' projects under ~12 h; "
          "otherwise send this output back for the next optimization round.",
          flush=True)


if __name__ == "__main__":
    main()
