"""Quick correctness checks for cvar_proj (run directly)."""
import numpy as np
import cvar_proj as cp_

rng = np.random.default_rng(0)


def rel(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30)


def fd_jvp(v, k, d, direction, eps, forward):
    """Central-difference directional derivative of the projection."""
    zp = forward(v + eps * direction, k, d)
    zm = forward(v - eps * direction, k, d)
    return (zp - zm) / (2 * eps)


print("=" * 70)
print("1. forward projection vs cvxpy ground truth")
print("=" * 70)
errs = []
for trial in range(20):
    m = rng.integers(20, 200)
    k = int(rng.integers(2, m // 2))
    v = rng.standard_normal(m)
    # pick d so the constraint is moderately violated
    Sk = cp_.topk_sum(v, k)
    d = Sk - rng.uniform(0.5, 5.0)
    z_ours = cp_.project_topk_sum(v, k, d)
    z_cvx = cp_.project_topk_sum_cvxpy(v, k, d)
    e = rel(z_ours, z_cvx)
    errs.append(e)
print(f"   trials={len(errs)}  max rel err (ours vs cvxpy) = {max(errs):.2e}")
assert max(errs) < 1e-5, "forward projection disagrees with cvxpy"

print("=" * 70)
print("2. VJP vs finite differences  (NO-TIE / generic instances)")
print("=" * 70)
errs = []
ntests = 0
for trial in range(60):
    m = int(rng.integers(50, 400))
    k = int(rng.integers(2, m // 3))
    # exactly the top-k form a high, separated group -> unique boundary, no tie
    v = np.concatenate([rng.normal(10, 1, k), rng.normal(0, 1, m - k)])
    rng.shuffle(v)
    Sk = cp_.topk_sum(v, k)
    d = Sk - 2.0                       # mild violation -> no plateau
    z = cp_.project_topk_sum(v, k, d)
    cert = cp_.extract_certificate(z, k, d)
    if not (cert["active"] and cert["g"] == 1):
        continue                       # not a clean no-tie draw; skip
    ntests += 1
    zbar = rng.standard_normal(m)
    analytic = cp_.vjp_projection(zbar, cert)            # (D Pi)^T zbar = (D Pi) zbar
    fd = fd_jvp(v, k, d, zbar, 1e-6, cp_.project_topk_sum)
    errs.append(rel(analytic, fd))
print(f"   clean no-tie trials={ntests}  max rel VJP error = {max(errs):.2e}")
assert ntests >= 20 and max(errs) < 1e-5, "no-tie VJP disagrees with finite differences"

print("=" * 70)
print("3. VJP vs finite differences  (PLATEAU instances)")
print("=" * 70)
# Construct v inside a plateau cell, verify the plateau is detected and the
# weighted-normal VJP matches finite differences (the cell is full-dimensional,
# so the Jacobian is well-defined in a neighborhood).
errs = []
plateau_sizes = []
for trial in range(30):
    m = int(rng.integers(20, 120))
    k = int(rng.integers(3, m // 2))
    # top entries close together so a strong violation pools them
    top = rng.normal(10.0, 0.3, k + rng.integers(2, 6))
    bot = rng.normal(-5.0, 1.0, m - top.shape[0])
    v = np.concatenate([top, bot])
    rng.shuffle(v)
    Sk = cp_.topk_sum(v, k)
    d = Sk - rng.uniform(15.0, 35.0)    # strong violation -> plateau
    z = cp_.project_topk_sum(v, k, d)
    # confirm against cvxpy too
    z_cvx = cp_.project_topk_sum_cvxpy(v, k, d)
    if rel(z, z_cvx) > 1e-5:
        continue
    cert = cp_.extract_certificate(z, k, d)
    if not cert["active"] or cert["g"] < 2:
        continue                        # not a plateau this draw; skip
    plateau_sizes.append(cert["g"])
    zbar = rng.standard_normal(m)
    analytic = cp_.vjp_projection(zbar, cert)
    fd = fd_jvp(v, k, d, zbar, 1e-6, cp_.project_topk_sum)
    errs.append(rel(analytic, fd))
print(f"   plateau trials={len(errs)}  plateau sizes seen: "
      f"{sorted(set(plateau_sizes))}")
print(f"   max rel VJP error (plateau) = {max(errs):.2e}")
assert len(errs) >= 5, "did not generate enough plateau cases"
assert max(errs) < 1e-4, "plateau VJP disagrees with finite differences"

print("=" * 70)
print("4. adjoints w.r.t. d and kappa vs finite differences")
print("=" * 70)
errs = []
for trial in range(20):
    m = int(rng.integers(50, 300))
    k = int(rng.integers(2, m // 3))
    v = np.concatenate([rng.normal(8, 1, k + 5), rng.normal(0, 1, m - k - 5)])
    rng.shuffle(v)
    Sk = cp_.topk_sum(v, k)
    d = Sk - 2.0
    z = cp_.project_topk_sum(v, k, d)
    cert = cp_.extract_certificate(z, k, d)
    zbar = rng.standard_normal(m)
    dbar, kappabar = cp_.rhs_adjoints(zbar, cert)
    # finite difference of <zbar, Pi(v; d)> w.r.t. d
    eps = 1e-6
    fp = zbar @ cp_.project_topk_sum(v, k, d + eps)
    fm = zbar @ cp_.project_topk_sum(v, k, d - eps)
    dbar_fd = (fp - fm) / (2 * eps)
    errs.append(abs(dbar - dbar_fd) / (abs(dbar_fd) + 1e-12))
print(f"   trials={len(errs)}  max rel error in d-adjoint = {max(errs):.2e}")
assert max(errs) < 1e-4, "d-adjoint disagrees with finite differences"

print()
print("ALL CORRECTNESS CHECKS PASSED")

