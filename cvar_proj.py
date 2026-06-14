"""
cvar_proj.py
============
A differentiable CVaR / top-k-sum projection primitive.

Implements the Euclidean projection onto the CVaR sublevel set
    C = { z in R^m : phi_beta(z) <= kappa }
which, for integer k = (1-beta)*m, equals the top-k-sum sublevel set
    C = { z in R^m : sum of k largest entries of z <= d },   d = k*kappa.

We provide:
  * project_topk_sum(v, k, d)        exact forward projection (numpy)
  * project_topk_sum_cvxpy(v, k, d)  ground-truth forward projection (cvxpy)
  * extract_certificate(z, k, d)     active-face certificate from projected point
  * vjp_projection(zbar, cert)       reverse-mode VJP   (D Pi)^T zbar    [O(m)]
  * jvp_projection(u, cert)          forward-mode JVP   (D Pi) u         [O(m)]
  * rhs_adjoints(zbar, cert)         adjoints w.r.t. d and kappa
  * CVaRProject (torch.autograd.Function)  if torch is available

The forward routine is NOT the contribution (see Roth & Cui 2025; Pan & Yan 2025;
Luxenberg et al. 2025).  It is included only so the experiments are self-contained.
The contribution is the backward pass (certificate + VJP).
"""

import numpy as np

# --------------------------------------------------------------------------
# CVaR / top-k-sum helpers
# --------------------------------------------------------------------------

def topk_sum(z, k):
    """Sum of the k largest entries of z (last axis)."""
    z = np.asarray(z, dtype=float)
    m = z.shape[-1]
    if k <= 0:
        return np.zeros(z.shape[:-1]) if z.ndim > 1 else 0.0
    if k >= m:
        return z.sum(axis=-1)
    part = np.partition(z, m - k, axis=-1)[..., m - k:]
    return part.sum(axis=-1)


def k_from_beta(m, beta):
    """k = (1-beta)*m; returns (k_int, tau, is_integer)."""
    tau = (1.0 - beta) * m
    k = int(round(tau))
    return k, tau, abs(tau - k) < 1e-9


# --------------------------------------------------------------------------
# Forward projection (exact)
# --------------------------------------------------------------------------

def project_topk_sum(v, k, d, tol=1e-12):
    """
    Exact Euclidean projection of v onto { x : (sum of k largest) <= d }.

    Three regimes:
      (1) inactive            : topk_sum(v,k) <= d            -> return v
      (2) no-plateau active   : subtract a common lambda from the top k
      (3) plateau active      : a plateau straddles the boundary index k

    Regime (3) uses an O(m^2) block search; it is exact and is only exercised
    by the small tie/plateau tests.  Large-m scaling instances are constructed
    in the no-plateau regime (the generic, full-measure case), where the cost
    is one sort + O(m).
    """
    v = np.asarray(v, dtype=float)
    m = v.shape[0]
    assert 1 <= k <= m, "need 1<=k<=m"
    if topk_sum(v, k) <= d + tol:
        return v.copy()                                   # (1) inactive

    order = np.argsort(-v, kind="stable")
    u = v[order]
    pref = np.concatenate(([0.0], np.cumsum(u)))          # pref[i]=sum u[0:i]

    # (2) no-plateau candidate: x_i = u_i - lam for i<k
    lam = (pref[k] - d) / k
    if lam >= -tol and (k == m or (u[k - 1] - lam) >= u[k] - tol):
        x = u.copy()
        x[:k] -= lam
        z = np.empty_like(x)
        z[order] = x
        return z

    # (3) plateau: L=[0,a-1] lowered by lam, P=[a,b] flattened to theta, U unchanged
    best = None
    for a in range(0, k):                 # |L| = a, plateau-counted c = k-a >= 1
        SL = pref[a]
        nL = a
        c = k - a
        for b in range(max(a, k - 1), m):
            nP = b - a + 1
            SP = pref[b + 1] - pref[a]
            denom = c * c + nP * nL
            if denom <= 0:
                continue
            lam_ = (SP * c - nP * d + nP * SL) / denom
            if lam_ < -tol:
                continue
            theta = (d - SL + nL * lam_) / c
            ok = True
            if a > 0 and not (u[a - 1] - lam_ >= theta - 1e-9):
                ok = False
            if ok and not (u[a] <= theta + lam_ + 1e-9):       # s<=1 at top of P
                ok = False
            if ok and not (u[b] >= theta - 1e-9):              # s>=0 at bot of P
                ok = False
            if ok and (b + 1 < m) and not (u[b + 1] <= theta + 1e-9):
                ok = False
            if ok:
                best = (a, b, lam_, theta)
                break
        if best is not None:
            break
    if best is None:
        raise RuntimeError("plateau search failed (try project_topk_sum_cvxpy)")
    a, b, lam_, theta = best
    x = u.copy()
    x[:a] -= lam_
    x[a:b + 1] = theta
    z = np.empty_like(x)
    z[order] = x
    return z


def project_topk_sum_cvxpy(v, k, d, solver=None):
    """Ground-truth projection via cvxpy (uses cp.sum_largest)."""
    import cvxpy as cp
    v = np.asarray(v, dtype=float)
    m = v.shape[0]
    x = cp.Variable(m)
    prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(x - v)),
                      [cp.sum_largest(x, k) <= d])
    if solver is None:
        # high-accuracy default; CLARABEL ships with cvxpy
        try:
            prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10,
                       tol_gap_rel=1e-10, tol_feas=1e-10)
        except Exception:
            prob.solve(solver=cp.SCS, eps=1e-9)
    else:
        prob.solve(solver=solver)
    return np.array(x.value, dtype=float)


# --------------------------------------------------------------------------
# Active-face certificate (recovered from the projected point z*)
# --------------------------------------------------------------------------

def extract_certificate(z, k, d, tol=1e-7, active=None):
    """
    Recover the active-face certificate from the projected point z*.

    For the integer top-k-sum projection only ONE tied group matters: the
    plateau at the VaR boundary (the value of the k-th largest entry).  Ties
    strictly above the boundary carry weight 1 regardless; ties strictly below
    carry weight 0.  So the certificate is (S, P, q, g):
        S : indices strictly above the boundary value   (weight 1)
        P : indices equal to the boundary value          (the tied group)
        g : |P|,   q = k - |S|  (number of P counted in the top-k)

    `active`: optional violation/multiplier status from the forward solve
    (True iff the projection moved the point).  If None, falls back to a
    one-sided boundary test on z* at `tol`, which by convention selects the
    face branch at zero violation; passing the forward status is exact and is
    what the autograd wrappers do.

    Endpoint normalization (partial-inclusion rule): the face forces
    within-group averaging only for a PARTIALLY included boundary group
    (0 < q < g).  If the group is fully included (q == g, strict gap below),
    the classical derivative is the plain no-tie formula over S u P, so the
    group is merged into the strict set and the certificate degenerates to
    the no-tie face (P empty, g = q = 0).
    """
    z = np.asarray(z, dtype=float)
    m = z.shape[0]
    if active is None:
        active = topk_sum(z, k) > d - tol
    if not active:
        return {"active": False}
    bval = np.partition(z, m - k)[m - k]                 # value of k-th largest
    S = np.where(z > bval + tol)[0]
    P = np.where(np.abs(z - bval) <= tol)[0]
    s = int(S.shape[0])
    g = int(P.shape[0])
    q = float(min(max(k - s, 0.0), g))
    if g > 0 and q >= g:          # fully included boundary group -> no-tie face
        S = np.sort(np.concatenate([S, P]))
        s += g
        P = np.empty(0, dtype=int)
        g = 0
        q = 0.0
    return {"active": True, "S": S, "P": P, "s": s, "g": g, "q": q,
            "k": int(k), "d": float(d)}


# --------------------------------------------------------------------------
# Vector-Jacobian product (the contribution)
# --------------------------------------------------------------------------

def vjp_projection(zbar, cert):
    """
    Reverse-mode VJP:  vbar = (D Pi)^T zbar.
    Pi is the metric projection; its Jacobian on a fixed face is the orthogonal
    projector P_T = P_A - b b^T / ||b||^2, which is symmetric, so VJP == JVP.
    Cost: O(m), O(1) extra memory beyond the certificate index sets.
    """
    zbar = np.asarray(zbar, dtype=float)
    if not cert["active"]:
        return zbar.copy()
    S, P = cert["S"], cert["P"]
    s, g, q = cert["s"], cert["g"], cert["q"]
    vbar = zbar.copy()
    sumS = zbar[S].sum()
    if g > 0:                              # partially included boundary group
        sumP = zbar[P].sum()
        w = q / g                          # per-coordinate weight on plateau
        bnorm2 = s + (q * q) / g           # ||b||^2 = s + q^2/g
        t = sumS + w * sumP                # b^T zbar
        coef = t / bnorm2
        meanP = sumP / g                   # (P_A zbar) on the plateau
        vbar[S] = zbar[S] - coef           # P_A leaves S unchanged
        vbar[P] = meanP - coef * w
    else:                                  # no-tie face (incl. merged q==g)
        coef = sumS / s                    # ||b||^2 = s, b = 1_S
        vbar[S] = zbar[S] - coef
    return vbar


def jvp_projection(u, cert):
    """Forward-mode JVP: (D Pi) u.  Same symmetric operator as the VJP."""
    return vjp_projection(u, cert)


def rhs_adjoints(zbar, cert):
    """
    Adjoints w.r.t. the constraint level, for learnable d or kappa:
        dbar     = (a^T zbar)/||a||^2        (here a -> b, the projected normal)
        kappabar = k * dbar    (integer case, since d = k*kappa)
    Returns (dbar, kappabar).  Zero if inactive.
    """
    zbar = np.asarray(zbar, dtype=float)
    if not cert["active"]:
        return 0.0, 0.0
    S, P = cert["S"], cert["P"]
    s, g, q, k = cert["s"], cert["g"], cert["q"], cert["k"]
    if g > 0:
        w = q / g
        bnorm2 = s + (q * q) / g
        t = zbar[S].sum() + w * zbar[P].sum()
    else:                                  # no-tie face (incl. merged q==g)
        bnorm2 = float(s)
        t = zbar[S].sum()
    dbar = t / bnorm2
    kappabar = k * dbar
    return dbar, kappabar


# --------------------------------------------------------------------------
# Batched (vectorized) versions  -- forward, certificate, VJP over rows of V
# --------------------------------------------------------------------------

def project_topk_sum_batch(V, k, d, tol=1e-12):
    """
    Vectorized projection of each row of V (shape [B,m]) onto {sum_k <= d}.
    The no-plateau and inactive regimes are fully vectorized (one batched sort);
    the rare plateau rows fall back to the exact per-row routine.
    """
    V = np.asarray(V, dtype=float)
    B, m = V.shape
    order = np.argsort(-V, axis=1, kind="stable")
    U = np.take_along_axis(V, order, axis=1)            # sorted desc per row
    Sk = U[:, :k].sum(axis=1)
    active = Sk > d + tol
    lam = (Sk - d) / k                                   # per row
    if k < m:
        simple = active & (lam >= -tol) & (U[:, k - 1] - lam >= U[:, k] - tol)
    else:
        simple = active & (lam >= -tol)
    Z = V.copy()
    # vectorized no-plateau rows
    rows = np.where(simple)[0]
    if rows.size:
        Xs = U[rows].copy()
        Xs[:, :k] -= lam[rows, None]
        inv = np.argsort(order[rows], axis=1)            # inverse permutation
        Z[rows] = np.take_along_axis(Xs, inv, axis=1)
    # per-row fallback for active-but-not-simple (plateau) rows
    hard = np.where(active & ~simple)[0]
    for b in hard:
        Z[b] = project_topk_sum(V[b], k, d, tol)
    return Z


def extract_certificate_batch(Z, k, d, tol=1e-7, active=None):
    """Batched certificate: boolean masks + per-row (s,g,q). Z is [B,m].
    `active` optionally carries the forward solve's per-row violation status;
    the q==g endpoint rule merges fully included boundary groups into S
    (see extract_certificate)."""
    Z = np.asarray(Z, dtype=float)
    B, m = Z.shape
    if active is None:
        Sk = topk_sum(Z, k)
        active = Sk > d - tol
    else:
        active = np.asarray(active, dtype=bool)
    bval = np.partition(Z, m - k, axis=1)[:, m - k]      # k-th largest per row
    maskS = Z > bval[:, None] + tol
    maskP = np.abs(Z - bval[:, None]) <= tol
    maskS &= active[:, None]
    maskP &= active[:, None]
    s = maskS.sum(axis=1).astype(float)
    graw = maskP.sum(axis=1).astype(float)
    q = np.clip(k - s, 0.0, graw)
    full = active & (graw > 0) & (q >= graw)             # fully included group
    maskS = maskS | (maskP & full[:, None])
    maskP = maskP & ~full[:, None]
    s = np.where(full, s + graw, s)
    q = np.where(full, 0.0, q)
    g = np.maximum(maskP.sum(axis=1).astype(float), 1.0)  # clamp: division safety
    return {"active": active, "maskS": maskS, "maskP": maskP,
            "s": s, "g": g, "q": q, "k": int(k), "d": float(d)}


def vjp_projection_batch(Zbar, cert):
    """Batched VJP using the certificate masks. Zbar is [B,m]; returns [B,m]."""
    Zbar = np.asarray(Zbar, dtype=float)
    maskS, maskP = cert["maskS"], cert["maskP"]
    s, g, q = cert["s"], cert["g"], cert["q"]
    w = q / g                                            # (B,)
    bnorm2 = s + (q * q) / g                             # (B,)
    sumS = (Zbar * maskS).sum(axis=1)
    sumP = (Zbar * maskP).sum(axis=1)
    t = sumS + w * sumP
    coef = np.where(cert["active"], t / np.where(bnorm2 > 0, bnorm2, 1.0), 0.0)
    meanP = np.where(maskP.any(axis=1), sumP / g, 0.0)
    out = Zbar.copy()
    out = np.where(maskS, Zbar - coef[:, None], out)
    out = np.where(maskP, (meanP - coef * w)[:, None], out)
    return out


# --------------------------------------------------------------------------
# Optional torch binding
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Optional torch binding (imported lazily so numpy-only use stays light)
# --------------------------------------------------------------------------

import importlib.util
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_CVaRProject_cls = None


def _build_cvar_project():
    global _CVaRProject_cls
    import torch

    class CVaRProject(torch.autograd.Function):
        """Differentiable CVaR projection for 1-D or [B,m] tensors (same k)."""

        @staticmethod
        def forward(ctx, v, k, d, tol=1e-12):
            # violation status from the input (the forward's own activity
            # test), passed to the certificate so the boundary branch is
            # selected by the solve, not re-derived from z* (which is
            # ambiguous exactly on the boundary).
            v_np = v.detach().cpu().numpy()
            if v_np.ndim == 1:
                act = bool(topk_sum(v_np, k) > float(d) + tol)
                z = project_topk_sum(v_np, k, float(d), tol)
                ctx.cert = extract_certificate(z, k, float(d), active=act)
                ctx.batched = False
                out = z
            else:
                act = topk_sum(v_np, k) > float(d) + tol
                z = project_topk_sum_batch(v_np, k, float(d), tol)
                ctx.cert = extract_certificate_batch(z, k, float(d), active=act)
                ctx.batched = True
                out = z
            return torch.as_tensor(out, dtype=v.dtype, device=v.device)

        @staticmethod
        def backward(ctx, zbar):
            zb = zbar.detach().cpu().numpy()
            if not ctx.batched:
                vbar = vjp_projection(zb, ctx.cert)
            else:
                vbar = vjp_projection_batch(zb, ctx.cert)
            g = torch.as_tensor(vbar, dtype=zbar.dtype, device=zbar.device)
            return g, None, None, None

    _CVaRProject_cls = CVaRProject
    return CVaRProject


def get_CVaRProject():
    """Return the torch autograd Function (builds + imports torch on first use)."""
    if _CVaRProject_cls is None:
        _build_cvar_project()
    return _CVaRProject_cls


def cvar_project(v, beta=0.95, kappa=0.0, tol=1e-12):
    """Convenience wrapper: project rows of v onto the CVaR set."""
    m = v.shape[-1]
    k, tau, is_int = k_from_beta(m, beta)
    if not is_int:
        raise NotImplementedError("this demo uses integer k=(1-beta)*m")
    d = k * kappa
    return get_CVaRProject().apply(v, k, d, tol)


def __getattr__(name):                       # PEP 562: lazy module attribute
    if name == "CVaRProject":
        return get_CVaRProject()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ==========================================================================
# Torch-native, GPU-ready projection + VJP   (added for the A100 experiments)
# ==========================================================================
# These mirror the verified numpy routines above but operate on torch tensors
# end-to-end, so the forward sort/sweep AND the backward VJP both run on the
# GPU with no host round-trip.  `bench_gpu.py` self-validates them against the
# numpy reference before timing.  Plateau rows (rare; off the benchmark path)
# fall back to the exact numpy per-row routine.

def project_topk_sum_torch(V, k, d, tol=1e-12):
    """Batched projection of rows of V onto { x : sum of k largest <= d }.
    Device-agnostic torch.  V is (m,) or (B, m); returns the same shape.
    Vectorized no-plateau path (the generic, full-measure case); per-row numpy
    fallback for any plateau rows."""
    import torch
    single = (V.dim() == 1)
    if single:
        V = V.unsqueeze(0)
    B, m = V.shape
    vals, idx = torch.sort(V, dim=1, descending=True)
    Sk = vals[:, :k].sum(dim=1)
    active = Sk > d + tol
    lam = (Sk - d) / k
    if k < m:
        simple = active & (lam >= -tol) & (vals[:, k - 1] - lam >= vals[:, k] - tol)
    else:
        simple = active & (lam >= -tol)
    topk_mask = torch.zeros_like(vals, dtype=torch.bool)
    topk_mask[:, :k] = True
    Xs = torch.where(topk_mask, vals - lam.unsqueeze(1), vals)
    back = torch.empty_like(V)
    back.scatter_(1, idx, Xs)                      # undo the sort permutation
    Z = torch.where(simple.unsqueeze(1), back, V)
    hard = active & ~simple
    if bool(hard.any()):                           # rare plateau rows -> exact numpy
        Vnp = V.detach().cpu().numpy()
        for b in torch.nonzero(hard, as_tuple=False).flatten().tolist():
            znp = project_topk_sum(Vnp[b], k, float(d), tol)
            Z[b] = torch.as_tensor(znp, dtype=V.dtype, device=V.device)
    return Z.squeeze(0) if single else Z


def extract_certificate_torch(Z, k, d, tol=1e-7, active=None):
    """Batched active-face certificate (torch). Z is (m,) or (B, m).
    `active` optionally carries the forward solve's per-row violation status;
    the q==g endpoint rule merges fully included boundary groups into S
    (see extract_certificate)."""
    import torch
    single = (Z.dim() == 1)
    if single:
        Z = Z.unsqueeze(0)
    topv = torch.topk(Z, k, dim=1).values
    bval = topv[:, -1]                              # k-th largest per row
    if active is None:
        active = topv.sum(dim=1) > d - tol
    else:
        active = active.reshape(Z.shape[0]).to(torch.bool)
    maskS = (Z > bval.unsqueeze(1) + tol) & active.unsqueeze(1)
    maskP = (torch.abs(Z - bval.unsqueeze(1)) <= tol) & active.unsqueeze(1)
    s = maskS.sum(dim=1).to(Z.dtype)
    graw = maskP.sum(dim=1).to(Z.dtype)
    q = torch.minimum(torch.clamp(k - s, min=0.0), graw)
    full = active & (graw > 0) & (q >= graw)        # fully included group
    maskS = maskS | (maskP & full.unsqueeze(1))
    maskP = maskP & ~full.unsqueeze(1)
    s = torch.where(full, s + graw, s)
    q = torch.where(full, torch.zeros_like(q), q)
    g = maskP.sum(dim=1).to(Z.dtype).clamp(min=1.0)  # clamp: division safety
    return dict(active=active, maskS=maskS, maskP=maskP, s=s, g=g, q=q,
                k=int(k), d=float(d), single=single)


def vjp_projection_torch(Zbar, cert):
    """Batched reverse-mode VJP (torch), mirroring vjp_projection_batch. O(m)."""
    import torch
    if Zbar.dim() == 1:
        Zbar = Zbar.unsqueeze(0)
    maskS, maskP = cert["maskS"], cert["maskP"]
    s, g, q, active = cert["s"], cert["g"], cert["q"], cert["active"]
    w = q / g
    bnorm2 = s + (q * q) / g
    sumS = (Zbar * maskS).sum(dim=1)
    sumP = (Zbar * maskP).sum(dim=1)
    t = sumS + w * sumP
    safe = torch.where(bnorm2 > 0, bnorm2, torch.ones_like(bnorm2))
    coef = torch.where(active, t / safe, torch.zeros_like(t))
    meanP = torch.where(maskP.any(dim=1), sumP / g, torch.zeros_like(sumP))
    out = Zbar.clone()
    out = torch.where(maskS, Zbar - coef.unsqueeze(1), out)
    out = torch.where(maskP, (meanP - coef * w).unsqueeze(1), out)
    return out.squeeze(0) if cert.get("single", False) else out


def rhs_adjoints_torch(Zbar, cert):
    """Batched adjoints w.r.t. the constraint level (torch), mirroring
    rhs_adjoints: dbar = (b^T zbar)/||b||^2 per row, kappabar = k * dbar.
    Zero on inactive rows.  Used by learnable-kappa layers."""
    import torch
    single = (Zbar.dim() == 1)
    if single:
        Zbar = Zbar.unsqueeze(0)
    maskS, maskP = cert["maskS"], cert["maskP"]
    s, g, q, active = cert["s"], cert["g"], cert["q"], cert["active"]
    w = q / g                       # g is clamped >=1; q=0 when P is empty
    bnorm2 = s + (q * q) / g
    t = (Zbar * maskS).sum(dim=1) + w * (Zbar * maskP).sum(dim=1)
    safe = torch.where(bnorm2 > 0, bnorm2, torch.ones_like(bnorm2))
    dbar = torch.where(active, t / safe, torch.zeros_like(t))
    kappabar = float(cert["k"]) * dbar
    if cert.get("single", False):
        return dbar.squeeze(0), kappabar.squeeze(0)
    return dbar, kappabar


_CVaRProjectTorch_cls = None


def get_CVaRProjectTorch():
    """torch.autograd.Function with GPU-native forward + O(m) VJP backward."""
    global _CVaRProjectTorch_cls
    if _CVaRProjectTorch_cls is None:
        import torch

        class CVaRProjectTorch(torch.autograd.Function):
            @staticmethod
            def forward(ctx, V, k, d, tol=1e-12):
                # violation status from the input, not re-derived from Z
                Vb = V if V.dim() > 1 else V.unsqueeze(0)
                act = torch.topk(Vb, k, dim=1).values.sum(dim=1) > float(d) + tol
                Z = project_topk_sum_torch(V, k, d, tol)
                ctx.cert = extract_certificate_torch(Z, k, float(d), active=act)
                return Z

            @staticmethod
            def backward(ctx, Zbar):
                return vjp_projection_torch(Zbar, ctx.cert), None, None, None

        _CVaRProjectTorch_cls = CVaRProjectTorch
    return _CVaRProjectTorch_cls
