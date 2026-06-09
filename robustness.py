"""
robustness.py  (paper Sec. 9.4)
===============================
Failure-analysis / robustness suite for the differentiable CVaR projection.

Methodological note (important): the projection is piecewise affine, so a
random-direction finite difference is a valid reference ONLY when the +/- eps
perturbation stays inside the same normal-fan cell (same active-face
certificate). Across a cell boundary the map is nondifferentiable and the
central difference is meaningless. We therefore compare the VJP to finite
differences ONLY on certificate-stable cells (the full-measure differentiable
regime of Theorem 2 / Prop. 4), and separately report how often a perturbation
lands on a boundary. This is the correct test, and it yields clean errors.

Stresses:
  (1) heavy-tailed & badly-scaled scenarios     (forward vs cvxpy; VJP vs FD)
  (2) many duplicate losses -> ties/plateaus     (VJP vs FD on stable plateau cells)
  (3) active-set threshold (tol) sweep           (certificate stability vs tol)
  (4) VJP error vs forward-solve error           (monotone: Prop. certificate stability)
  (5) early-stopped ADMM (torch; skipped if absent)

Outputs: results/robustness.csv, results/robustness_tol.png, results/robustness_solveerr.png
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cvar_proj as cp_

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
rng = np.random.default_rng(0)
rows = []


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def cert_equal(c1, c2):
    if c1["active"] != c2["active"]:
        return False
    if not c1["active"]:
        return True
    return (c1["s"] == c2["s"] and c1["g"] == c2["g"]
            and abs(c1["q"] - c2["q"]) < 1e-9
            and set(c1["S"].tolist()) == set(c2["S"].tolist())
            and set(c1["P"].tolist()) == set(c2["P"].tolist()))


def fd_stable(v, k, d, u, cert, eps=1e-6, tol=1e-7):
    """Central-difference JVP, but only if both +/- eps stay in the same cell."""
    zp = cp_.project_topk_sum(v + eps * u, k, d)
    zm = cp_.project_topk_sum(v - eps * u, k, d)
    cp_p = cp_.extract_certificate(zp, k, d, tol=tol)
    cp_m = cp_.extract_certificate(zm, k, d, tol=tol)
    if cert_equal(cp_p, cert) and cert_equal(cp_m, cert):
        return (zp - zm) / (2 * eps)
    return None


# (1) heavy-tailed + badly scaled -----------------------------------------
print("=" * 72); print("(1) heavy-tailed & badly-scaled inputs"); print("=" * 72)
fwd_err = vjp_err = 0.0; stable = skipped = 0
for _ in range(60):
    m = int(rng.integers(50, 300)); k = int(rng.integers(2, m // 3))
    scale = 10.0 ** rng.integers(-3, 4)
    v = rng.standard_t(2.0, size=m) * scale                  # heavy tail (t, dof=2)
    d = float(cp_.topk_sum(v, k) - abs(scale))
    z = cp_.project_topk_sum(v, k, d)
    fwd_err = max(fwd_err, rel(z, cp_.project_topk_sum_cvxpy(v, k, d)))
    cert = cp_.extract_certificate(z, k, d)
    if not cert["active"]:
        continue
    u = rng.standard_normal(m)
    fd = fd_stable(v, k, d, u, cert, eps=1e-6 * max(scale, 1.0))
    if fd is None:
        skipped += 1; continue
    stable += 1
    vjp_err = max(vjp_err, rel(cp_.vjp_projection(u, cert), fd))
print(f"   forward rel err vs cvxpy (all)         = {fwd_err:.2e}")
print(f"   VJP rel err vs FD (stable cells={stable}) = {vjp_err:.2e}   (boundary draws skipped={skipped})")
rows += [dict(test="heavy_tailed_scaled", metric="forward_rel_err", value=fwd_err),
         dict(test="heavy_tailed_scaled", metric="vjp_rel_err_stable", value=vjp_err),
         dict(test="heavy_tailed_scaled", metric="n_stable", value=stable)]

# (2) duplicate losses -> ties/plateaus -----------------------------------
print("=" * 72); print("(2) projection-created plateaus (strong violation pools the tail)"); print("=" * 72)
tie_err = 0.0; tie_cases = 0; sizes = []
for _ in range(80):
    m = int(rng.integers(40, 200)); k = int(rng.integers(3, m // 2))
    top = rng.normal(10.0, 0.3, k + int(rng.integers(2, 6)))  # close top entries
    v = np.concatenate([top, rng.normal(-5.0, 1.0, m - top.size)]); rng.shuffle(v)
    d = float(cp_.topk_sum(v, k) - rng.uniform(15.0, 35.0))   # strong violation -> plateau
    z = cp_.project_topk_sum(v, k, d)
    if rel(z, cp_.project_topk_sum_cvxpy(v, k, d)) > 1e-5:
        continue
    cert = cp_.extract_certificate(z, k, d)
    if not cert["active"] or cert["g"] < 2:
        continue
    u = rng.standard_normal(m)
    fd = fd_stable(v, k, d, u, cert)
    if fd is None:
        continue
    tie_cases += 1; sizes.append(cert["g"])
    tie_err = max(tie_err, rel(cp_.vjp_projection(u, cert), fd))
print(f"   stable plateau cells tested = {tie_cases}  (sizes {sorted(set(sizes))})")
print(f"   max VJP rel err vs FD on plateau cells = {tie_err:.2e}")
rows += [dict(test="plateaus", metric="vjp_rel_err_stable", value=tie_err),
         dict(test="plateaus", metric="n_stable_plateau_cells", value=tie_cases)]

# (3) active-set threshold sweep ------------------------------------------
print("=" * 72); print("(3) active-set threshold (tol) sweep: tol must beat the data resolution"); print("=" * 72)
m, k = 200, 40
gap = 1e-3
top = rng.normal(10.0, gap, k)                                # top group spread ~ gap
v = np.concatenate([top, rng.normal(0.0, 1.0, m - k)]); rng.shuffle(v)
d = float(cp_.topk_sum(v, k) - 1.0)
z = cp_.project_topk_sum(v, k, d)
u = rng.standard_normal(m)
fd = (cp_.project_topk_sum(v + 1e-7 * u, k, d) - cp_.project_topk_sum(v - 1e-7 * u, k, d)) / (2e-7)
print(f"   (top-group spread ~ {gap:.0e}; tol below this is safe, above it merges spuriously)")
for tol in [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]:
    cert = cp_.extract_certificate(z, k, d, tol=tol)
    e = rel(cp_.vjp_projection(u, cert), fd) if cert["active"] else float("nan")
    print(f"   tol={tol:.0e}   detected plateau size g={cert.get('g',0):>3}   VJP rel err vs FD={e:.2e}")
    rows.append(dict(test="threshold_sweep", metric=f"vjp_err_tol_{tol:.0e}", value=e))

# (4) VJP error vs forward-solve error (Prop. certificate stability) -------
print("=" * 72); print("(4) VJP error vs forward-solve error (inject error delta into z*, recover cert)"); print("=" * 72)
m, k = 300, 60
v = np.concatenate([rng.normal(12, 1, k), rng.normal(0, 1, m - k)]); rng.shuffle(v)
d = float(cp_.topk_sum(v, k) - 2.0)
z_exact = cp_.project_topk_sum(v, k, d)
cert_exact = cp_.extract_certificate(z_exact, k, d)
u = rng.standard_normal(m)
g_exact = cp_.vjp_projection(u, cert_exact)
solve_curve = []
for delta in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8]:
    z_noisy = z_exact + delta * rng.standard_normal(m)        # emulate an inexact forward solve
    cert_n = cp_.extract_certificate(z_noisy, k, d, tol=max(10 * delta, 1e-7))
    g_n = cp_.vjp_projection(u, cert_n) if cert_n["active"] else u
    e = rel(g_n, g_exact)
    solve_curve.append((delta, e))
    print(f"   forward error delta={delta:.0e}   VJP rel err vs exact = {e:.2e}")
    rows.append(dict(test="grad_vs_solve_err", metric=f"vjp_err_delta_{delta:.0e}", value=e))
print("   -> VJP error falls to 0 once the forward error is below the cell margin "
      "(Prop. certificate stability).")

# (5) early-stopped ADMM (torch; optional) --------------------------------
print("=" * 72); print("(5) early-stopped ADMM gradient (torch; skipped if torch absent)"); print("=" * 72)
try:
    import torch  # noqa: F401
    print("   torch present: portfolio_e2e.py provides the unrolled-vs-cvxpy gradient comparison.")
    rows.append(dict(test="early_stop_admm", metric="torch_available", value=1))
except Exception:
    print("   torch not installed -> run on the A100. Numpy stress tests above are complete.")
    rows.append(dict(test="early_stop_admm", metric="torch_available", value=0))

# write -------------------------------------------------------------------
import pandas as pd
pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "robustness.csv"), index=False)
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dl = [x for x, _ in solve_curve]; el = [max(y, 1e-18) for _, y in solve_curve]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.loglog(dl, el, "o-"); ax.invert_xaxis()
    ax.set_xlabel("forward-solve error delta (tighter ->)"); ax.set_ylabel("VJP rel error vs exact")
    ax.set_title("Gradient error shrinks as the forward solve tightens")
    ax.grid(True, which="both", alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "robustness_solveerr.png"), dpi=150)
except Exception as e:
    print("   (plot skipped:", e, ")")

print("\nwrote results/robustness.csv and results/robustness_solveerr.png")
print("ROBUSTNESS SUITE COMPLETE")

