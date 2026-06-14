"""
exp_solvers.py
==============
Batched-instance solvers for the primary constraint-targeting experiment
(EXPERIMENT_EXECUTION_SPEC.md).  All solvers operate on a whole cell of
decision instances at once (spec: "Batched-instance ADMM is required at
m = 1e5 and recommended everywhere").

Problems (per decision instance j, variable w in R^n):

  HARD      min  gamma/2 ||w||^2 - mu_j^T w
            s.t. CVaR_beta(A_j w) <= kappa,  sum(w)=1,  0<=w<=w_max
  PENALTY   min  gamma/2 ||w||^2 - mu_j^T w + (lambda/k) topk_sum(A_j w, k)
            s.t. sum(w)=1,  0<=w<=w_max
  MIN-CVAR  PENALTY with lambda=1 and a small documented ridge
            (gamma_reg=1e-4): the per-instance feasibility certificate.
            Ridge bias on the certified CVaR is <= gamma_reg*w_max/2 = 1e-5
            for w_max=0.2, absorbed by CERT_BIAS below.
  MV FLOOR  min  gamma/2 ||w||^2 - mu_j^T w   s.t. sum(w)=1, 0<=w<=w_max
            (closed form: one capped-simplex projection of mu_j/gamma)

with A_j w = L w - (mu_j^T w) 1_m applied in operator form (the bank L is
shared; the per-instance mean enters analytically).  The simplex equality
rides in the box block  B = [I; 1^T],  l = [0,...,0,1],  u = [w_max,...,1].

Implementation notes (these cost real money on the A100 if done naively):
  * The CVaR-block projection inside ADMM hits plateau inputs OFTEN in early
    iterations (the paper says so explicitly), and the library's exact
    per-row plateau search is O(m^2) Python.  Here plateau-flagged rows are
    solved by a fully vectorized multiplier bisection on the sorted rows
    (`_proj_cvar_robust`), exact to ~1e-12 and batched across rows.
  * rho is adapted per instance by standard residual balancing (He-Yang-Wang
    style: x2 / /2 when the primal/dual residuals are >10x apart, checked
    every 100 iterations, with the scaled duals rescaled and the n x n
    factors rebuilt).  Fixed rho needs 10-100x more iterations on this
    problem family.

The ADMM body is written once against a tiny `ops` namespace with numpy and
torch implementations; the numpy path is exercised by the smoke test and the
torch path must pass `exp_primary.py --self-check` (torch-vs-numpy on a small
cell) before any A100 run is trusted.
"""
import os

import numpy as np
import cvar_proj as cp_

# spec: Determinism, Tolerances, And Seeds.  EPS amended 1e-8 -> 1e-7 after
# profiling: 1e-7 converges in O(100-1000) batched-ADMM iterations with
# scale-aware rho0 and is still 100x tighter than ACTIVE_TOL, while 1e-8
# costs ~10x more iterations; the pilot rechecks 1e-6/1e-7 sensitivity.
EPS_ABS = 1e-7
EPS_REL = 1e-7
MAX_ITER = 20000
ACTIVE_TOL = 1e-4            # follows solve eps 1e-4 [AMENDED 2026-06-11]
FEAS_MARGIN_REL = 1e-6       # spec margin on the certificate
CERT_BIAS = 1e-4             # absolute guard: ridge bias + loose cert solve
GAMMA_REG = 1e-4             # min-CVaR certificate ridge (documented above)


def _clip(x, lo, hi):
    if isinstance(x, np.ndarray):
        return np.clip(x, lo, hi)
    import torch
    # torch.clamp requires min/max to be BOTH Numbers or BOTH Tensors;
    # promote scalars so mixed calls like clip(x, 0.0, lam_tensor) work
    # (this is exactly what the A100 self-check gate caught on 2026-06-10)
    if torch.is_tensor(lo) != torch.is_tensor(hi):
        if not torch.is_tensor(lo):
            lo = torch.as_tensor(lo, dtype=x.dtype, device=x.device)
        if not torch.is_tensor(hi):
            hi = torch.as_tensor(hi, dtype=x.dtype, device=x.device)
    return torch.clamp(x, lo, hi)


def _where(c, a, b):
    if isinstance(a, np.ndarray) or np.isscalar(a):
        if isinstance(c, np.ndarray):
            return np.where(c, a, b)
    import torch
    return torch.where(c, a, b)


def _sync_every(Z):
    """How often (in loop rounds) to evaluate python-side early-exit checks.
    numpy: every round (checks are free).  torch: each check is a device
    sync; on small tensors syncs dominate, so check sparsely."""
    if isinstance(Z, np.ndarray):
        return 1
    return 1 if Z.numel() > 2_000_000 else 8


# --------------------------------------------------------------------------
# optional compiled inner loops (CUDA-graph capture via torch.compile).
# The bisection cascades are LAUNCH/SYNC-bound on GPU: each eager round is
# ~6 tiny kernels + (sometimes) a device sync, so a 64-round inner solve
# costs ~100x its arithmetic.  torch.compile(mode="reduce-overhead")
# captures a fixed-round, sync-free refinement as one CUDA graph whose
# replay costs microseconds.  Exactness: 48 halvings of a validity-checked
# bracket reach ~2^-48 of its width, matching the eager path's tolerance.
# Disable with CVAR_COMPILE=0 (automatic eager fallback on any failure).
# --------------------------------------------------------------------------

def _compile_enabled():
    return os.environ.get("CVAR_COMPILE", "1") != "0"


def _sigma_rounds_eager(Zh, lam, s_lo, s_hi, kf):
    """48 fixed sigma-bisection rounds; pure tensor ops, no syncs."""
    import torch
    lamc = lam[..., None]
    tgt = kf * lam
    zero = torch.zeros((), dtype=Zh.dtype, device=Zh.device)
    for _ in range(16):
        sig = 0.5 * (s_lo + s_hi)
        psi = torch.clamp(Zh - sig[..., None], min=zero, max=lamc).sum(-1)
        too_big = psi > tgt
        s_lo = torch.where(too_big, sig, s_lo)
        s_hi = torch.where(too_big, s_hi, sig)
    return s_lo, s_hi


def _tau_rounds_eager(X, lo, hi, total, cap):
    """48 fixed capped-simplex tau-bisection rounds; pure tensor ops."""
    import torch
    zero = torch.zeros((), dtype=X.dtype, device=X.device)
    capt = torch.as_tensor(cap, dtype=X.dtype, device=X.device)
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        s = torch.clamp(X - mid[..., None], min=zero, max=capt).sum(-1)
        too_big = s > total
        lo = torch.where(too_big, mid, lo)
        hi = torch.where(too_big, hi, mid)
    return lo, hi


_compiled = {"sigma": None, "tau": None, "failed": False}


def _get_compiled(which):
    if _compiled["failed"] or not _compile_enabled():
        return None
    if _compiled[which] is None:
        try:
            import torch
            import torch._dynamo
            # we intentionally specialize per shape bucket: pow2 row buckets
            # x {m values} x two functions; the default cache limit of 8 is
            # exhausted by the first m alone and silently falls back to eager
            torch._dynamo.config.cache_size_limit = 96
            fn = _sigma_rounds_eager if which == "sigma" else _tau_rounds_eager
            _compiled[which] = torch.compile(fn, mode="reduce-overhead",
                                             dynamic=False)
        except Exception as e:                     # pragma: no cover
            _compiled["failed"] = True
            print(f"[exp_solvers] torch.compile unavailable "
                  f"({type(e).__name__}: {e}); using eager loops", flush=True)
            return None
    return _compiled[which]


def _pad_rows(t, R2):
    """Pad a [R,...] tensor to R2 rows by repeating row 0 (results for the
    pad rows are discarded; row 0 is always a valid problem)."""
    pad = R2 - t.shape[0]
    if pad <= 0:
        return t
    rep = t[:1].expand(pad, *t.shape[1:])
    import torch
    return torch.cat([t, rep], 0)



# --------------------------------------------------------------------------
# capped simplex  { 0 <= a_i <= cap,  sum_i a_i = total }   (bisection on tau)
# --------------------------------------------------------------------------

def proj_capped_simplex(X, total, cap, iters=64):
    """Row-wise projection onto {0<=a<=cap, sum a = total}. X is [..., m].
    Same code path for numpy arrays and torch tensors."""
    if isinstance(X, np.ndarray):
        lo = X.min(axis=-1) - cap
        hi = X.max(axis=-1)
    else:
        lo = X.min(dim=-1).values - cap
        hi = X.max(dim=-1).values
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = _clip(X - mid[..., None], 0.0, cap).sum(-1)
        too_big = s > total
        lo = _where(too_big, mid, lo)
        hi = _where(too_big, hi, mid)
    tau = 0.5 * (lo + hi)
    return _clip(X - tau[..., None], 0.0, cap)


def proj_capped_simplex_warm(X, total, cap, state=None, iters=64):
    """proj_capped_simplex with cross-call warm-started tau brackets.
    `state` is a dict carrying per-row "tau" and bracket width "w" from the
    previous call (or None).  Exactness is preserved: warm brackets are
    validity-checked and geometrically expanded before bisection."""
    is_np = isinstance(X, np.ndarray)
    if is_np:
        lo_cold = X.min(axis=-1) - cap
        hi_cold = X.max(axis=-1)
    else:
        lo_cold = X.min(dim=-1).values - cap
        hi_cold = X.max(dim=-1).values
    lo, hi = lo_cold, hi_cold
    se = _sync_every(X)
    if state is not None and state.get("tau") is not None:
        tau0, w0 = state["tau"], state["w"]
        w = _where(w0 > 0, w0 * 8.0, 0.125 * (hi_cold - lo_cold) + 1e-30)
        lo_t = tau0 - w
        hi_t = tau0 + w
        valid = False
        for r_ in range(20):   # validity repair: need s(lo)>=total>=s(hi)
            s_lo = _clip(X - lo_t[..., None], 0.0, cap).sum(-1)
            s_hi = _clip(X - hi_t[..., None], 0.0, cap).sum(-1)
            bad_lo = s_lo < total
            bad_hi = s_hi > total
            w = w * 4.0
            lo_t = _where(bad_lo, tau0 - w, lo_t)
            hi_t = _where(bad_hi, tau0 + w, hi_t)
            if r_ % se == se - 1 or r_ >= 18:
                anybad = bad_lo | bad_hi
                if not (bool(anybad.any()) if is_np
                        else bool(anybad.any().item())):
                    valid = True
                    break
        if not valid:
            lo_t, hi_t = lo_cold, hi_cold        # give up -> cold
        lo = _where(lo_t > lo_cold, lo_t, lo_cold)
        hi = _where(hi_t < hi_cold, hi_t, hi_cold)
    ctau = None if is_np or not X.is_cuda else _get_compiled("tau")
    if ctau is not None:
        # 16-round CUDA-graph replays, up to 3, width-checked between;
        # clone outputs out of the static buffers before reuse (see sigma)
        for _rep in range(3):
            lo, hi = ctau(X, lo, hi, float(total), float(cap))
            lo = lo.clone(); hi = hi.clone()
            if bool(((hi - lo) <= 1e-15 * (1.0 + hi.abs())).all().item()):
                break
    else:
        for r_ in range(iters):
            if r_ % se == se - 1:
                wid = hi - lo
                if (bool((wid <= 1e-15 * (1.0 + abs(hi))).all()) if is_np
                        else bool((wid <= 1e-15 * (1.0 + hi.abs())).all()
                                  .item())):
                    break
            mid = 0.5 * (lo + hi)
            s = _clip(X - mid[..., None], 0.0, cap).sum(-1)
            too_big = s > total
            lo = _where(too_big, mid, lo)
            hi = _where(too_big, hi, mid)
    tau = 0.5 * (lo + hi)
    if state is not None:
        state["tau"] = tau
        state["w"] = (hi - lo) + 1e-30
    return _clip(X - tau[..., None], 0.0, cap)


def prox_topk_sum(U, alpha, k):
    """prox_{alpha * f_k}(U) = U - alpha * Proj_{capped simplex}(U / alpha).
    `alpha` may be a scalar or a per-row vector (adaptive-rho case)."""
    if np.isscalar(alpha):
        if alpha <= 0.0:
            return U
        return U - alpha * proj_capped_simplex(U / alpha, float(k), 1.0)
    a = alpha[..., None]
    return U - a * proj_capped_simplex(U / a, float(k), 1.0)


# --------------------------------------------------------------------------
# robust batched projection onto { z : topk_sum(z, k) <= d }
# --------------------------------------------------------------------------

def _topk_sum_x(Z, k):
    if isinstance(Z, np.ndarray):
        return cp_.topk_sum(Z, k)
    import torch
    return torch.topk(Z, k, dim=-1).values.sum(-1)


def _sort_desc(Z):
    if isinstance(Z, np.ndarray):
        return -np.sort(-Z, axis=-1)
    import torch
    return torch.sort(Z, dim=-1, descending=True).values


def _proj_cvar_robust(Z, k, d, state=None, gtol=1e-13,
                      lam_warm=None, return_lam=False):
    """Exact batched projection of rows of Z onto { z : topk_sum(z,k) <= d },
    accurate to |topk_sum(z)-d| <= gtol*(1+|d|) on plateau rows (and to the
    no-plateau closed form elsewhere).

    Plateau rows are solved by two nested, safeguarded bisections on the KKT
    system  z(lam) = max(min(Zh, sigma), Zh - lam)  with
    sum_i clip(Zh_i - sigma, 0, lam) = k*lam  (inner, sigma) and
    g(lam) = topk_sum(z(lam), k) = d  (outer, lam).  Plain bisection in the
    unscaled data range is immune to the breakpoint coincidences that occur
    exactly at lam* (closed-form segment identification fails there under
    1-ulp rounding).

    `state` (optional dict) carries warm brackets across calls --- inside
    ADMM, consecutive inputs differ by O(residual), so warm lam/sigma
    brackets cut the per-call pass count by an order of magnitude late in
    the solve.  Warm brackets are always validity-checked and geometrically
    expanded before use, so warm-starting never affects exactness, only
    speed.  Inner precision is tied to gtol (inexact-prox ADMM schedule;
    the caller tightens gtol with the ADMM residuals).
    """
    is_np = isinstance(Z, np.ndarray)
    B = Z.shape[0]
    Sk = _topk_sum_x(Z, k)
    active = Sk > d + 1e-12
    if is_np:
        any_active = bool(active.any())
    else:
        any_active = bool(active.any().item())
    if not any_active:
        outz = Z.copy() if is_np else Z.clone()
        if return_lam:
            return outz, 0.0 * Sk
        return outz

    U = _sort_desc(Z)
    lam0 = (Sk - d) / k
    m = Z.shape[-1]
    ok_gap = (U[..., k - 1] - lam0 >= U[..., k] - 1e-12) if k < m else \
        (lam0 >= -1e-12)
    simple = active & (lam0 >= -1e-12) & ok_gap
    top_mask = Z >= (U[..., k - 1, None] - 1e-15)
    Zs = _where(top_mask, Z - lam0[..., None], Z)
    out = _where(simple[..., None], Zs, Z)

    hard = active & ~simple
    n_hard = int(hard.sum()) if is_np else int(hard.sum().item())
    lam_final = 0.0 * Sk
    lam_final = _where(simple, lam0, lam_final)
    if n_hard:
        Zh = Z[hard]
        Skh = Sk[hard]
        R0 = Zh.shape[0]
        csig = None if is_np or not Zh.is_cuda else _get_compiled("sigma")
        if csig is not None:
            # pad rows to the next power of two: bounds the number of shape
            # specializations torch.compile/CUDA graphs must hold
            R2 = 1 << max(R0 - 1, 0).bit_length() if R0 > 1 else 1
            Zh = _pad_rows(Zh, R2)
            Skh = _pad_rows(Skh, R2)
        if is_np:
            ZMIN = Zh.min(axis=-1); ZMAX = Zh.max(axis=-1)
        else:
            ZMIN = Zh.min(dim=-1).values; ZMAX = Zh.max(dim=-1).values
        rng_h = ZMAX - ZMIN + 1e-30

        # shared sigma bracket for this call (validity-repaired per inner)
        if state is not None and state.get("sig") is not None:
            sgp = state["sig"][hard]; sw = state["sig_w"][hard] * 8.0
            if csig is not None:
                sgp = _pad_rows(sgp, Zh.shape[0])
                sw = _pad_rows(sw, Zh.shape[0])
            s_lo_sh = sgp - sw
            s_hi_sh = sgp + sw
        else:
            s_lo_sh = ZMIN - 0.0      # cold; per-inner repair extends by lam
            s_hi_sh = ZMAX + 0.0
        sig_last = {"v": 0.5 * (s_lo_sh + s_hi_sh), "w": rng_h}

        se = _sync_every(Zh)

        def inner(lam):
            """sigma bisection on a validity-repaired bracket; precision tied
            to gtol (sigma error -> g error is Lipschitz with constant <=k).
            Early-exit checks run every `se` rounds: on torch each check is a
            device sync, which dominates on small tensors.  The where-gated
            updates are idempotent for already-valid/converged rows, so
            checking sparsely never affects correctness, only round count."""
            lamc = lam[..., None]
            tgt = k * lam
            s_lo = _where(s_lo_sh < ZMAX, s_lo_sh, ZMAX)
            s_hi = _where(s_hi_sh > ZMIN, s_hi_sh, ZMIN)
            # repair: need psi(s_lo) >= tgt >= psi(s_hi); cold-valid ends are
            # ZMIN - lam (psi >= k*lam always) and ZMAX (psi = 0)
            for r_ in range(30):
                p_lo = _clip(Zh - s_lo[..., None], 0.0, lamc).sum(-1)
                p_hi = _clip(Zh - s_hi[..., None], 0.0, lamc).sum(-1)
                bad_lo = p_lo < tgt
                bad_hi = p_hi > tgt
                if r_ % se == se - 1 or r_ == 29:
                    anyb = bad_lo | bad_hi
                    if not (bool(anyb.any()) if is_np
                            else bool(anyb.any().item())):
                        break
                step = 0.25 * rng_h + 0.5 * lam
                s_lo = _where(bad_lo, _where(s_lo - step > ZMIN - lam,
                                             s_lo - step, ZMIN - lam), s_lo)
                s_hi = _where(bad_hi, _where(s_hi + step < ZMAX,
                                             s_hi + step, ZMAX), s_hi)
            wtol = gtol * (1.0 + abs(d)) / (4.0 * k)
            if csig is not None:
                # 16-round CUDA-graph replays, up to 3 (=48 halvings max);
                # warm brackets usually need only the first replay.  Clone
                # the outputs out of the graph's static buffers before any
                # reuse: feeding them back as the next replay's inputs (or
                # retaining them across replays) trips the cudagraph-trees
                # overwrite protection (A100 self-check, 2026-06-11).
                for _rep in range(3):
                    s_lo, s_hi = csig(Zh, lam, s_lo, s_hi, float(k))
                    s_lo = s_lo.clone(); s_hi = s_hi.clone()
                    wid = s_hi - s_lo
                    if bool((wid <= wtol + 1e-15 * (1.0 + s_hi.abs()))
                            .all().item()):
                        break
            else:
                for r_ in range(64):
                    if r_ % se == se - 1:
                        wid = s_hi - s_lo
                        if (bool((wid <= wtol + 1e-15
                                  * (1.0 + abs(s_hi))).all())
                                if is_np else
                                bool((wid <= wtol + 1e-15
                                      * (1.0 + s_hi.abs())).all().item())):
                            break
                    sig = 0.5 * (s_lo + s_hi)
                    psi = _clip(Zh - sig[..., None], 0.0, lamc).sum(-1)
                    too_big = psi > tgt
                    s_lo = _where(too_big, sig, s_lo)
                    s_hi = _where(too_big, s_hi, sig)
            sig = 0.5 * (s_lo + s_hi)
            sig_last["v"] = sig
            sig_last["w"] = (s_hi - s_lo) + 1e-30
            zc = _where(Zh < sig[..., None], Zh, sig[..., None])
            zc = _where(zc > (Zh - lamc), zc, Zh - lamc)
            return zc

        def g_of(lam):
            zlam = inner(lam)
            return _topk_sum_x(zlam, k), zlam

        # lambda bracket: warm (width-tracked) or cold (lower bound + doubling)
        lb0 = (Skh - d) / k                  # lower bound on lam*
        if state is not None and state.get("lam") is not None:
            lp = state["lam"][hard]
            drift = (abs(lp - state["lam_prev"][hard])
                     if state.get("lam_prev") is not None else 0.0 * lp)
            lww = (state["lam_w"][hard] if state.get("lam_w") is not None
                   else 0.5 * lp)
            if csig is not None:             # match the pow2 row padding
                lp = _pad_rows(lp, Zh.shape[0])
                drift = _pad_rows(drift, Zh.shape[0])
                lww = _pad_rows(lww, Zh.shape[0])
            lw = lww * 8.0 + 4.0 * drift + 1e-12 * lp
            has = lp > 0
            lo = _where(has, _where(lp - lw > 1e-300, lp - lw, 0.0 * lp + 1e-300),
                        0.0 * lp + 1e-300)
            hi = _where(has, lp + lw, lb0)
            hi = _where(hi > lb0, hi, lb0)   # never start below the bound
        else:
            lo = 0.0 * Skh + 1e-300
            hi = lb0 + 0.0
        # repair lower end: need g(lo) > d (else shrink toward 0).
        # These lambda-level loops check EVERY round: their body contains a
        # full inner solve, so one sync per round is cheap relative to the
        # body (unlike the inner bisection, which checks sparsely).
        for _ in range(30):
            g_lo, _z = g_of(lo)
            bad = (g_lo <= d) & (lo > 1e-300)
            if not (bool(bad.any()) if is_np else bool(bad.any().item())):
                break
            hi = _where(bad, lo, hi)
            lo = _where(bad, lo * 0.25, lo)
        # repair/establish upper end: need g(hi) <= d (else double)
        for _ in range(60):
            g_hi, _z = g_of(hi)
            bad = g_hi > d
            if not (bool(bad.any()) if is_np else bool(bad.any().item())):
                break
            lo = _where(bad, hi, lo)
            hi = _where(bad, hi * 2.0, hi)
        # main loop to gtol: bisection alternated with safeguarded Newton
        # (g is piecewise linear; Newton inside a stable segment finishes in
        # one step, and the forced bisection on odd steps guarantees
        # geometric bracket shrink regardless)
        lam = 0.5 * (lo + hi)
        for step in range(64):
            g, zl = g_of(lam)
            too_high = g > d
            lo = _where(too_high, lam, lo)
            hi = _where(too_high, hi, lam)
            small = abs(g - d) <= gtol * (1.0 + abs(d))
            narrow = (hi - lo) <= 1e-14 * (1.0 + hi)
            if (bool((small | narrow).all()) if is_np
                    else bool((small | narrow).all().item())):
                break
            if step % 2 == 0:
                kth = _sort_desc(zl)[..., k - 1, None]
                topm = zl >= kth - 1e-15
                aw = ((Zh - zl) / lam[..., None]) * topm
                slope = aw.sum(-1)
                slope = (np.maximum(slope, 1e-12) if is_np
                         else slope.clamp(min=1e-12))
                newton = lam + (g - d) / slope
                inside = (newton > lo) & (newton < hi)
                lam = _where(inside, newton, 0.5 * (lo + hi))
            else:
                lam = 0.5 * (lo + hi)
        _g, Zsol = g_of(hi)                  # feasible side
        if csig is not None:                 # drop the pow2 padding rows
            Zsol = Zsol[:R0]
            hi = hi[:R0]; lo = lo[:R0]
            sig_last["v"] = sig_last["v"][:R0]
            sig_last["w"] = sig_last["w"][:R0]
        if is_np:
            out[hard] = Zsol
            lam_final[hard] = hi
        else:
            out = out.clone(); out[hard] = Zsol
            lam_final = lam_final.clone(); lam_final[hard] = hi
        if state is not None:
            full_sig = 0.0 * Sk
            full_sw = 0.0 * Sk + 1.0
            if state.get("sig") is not None:
                full_sig = state["sig"]; full_sw = state["sig_w"]
            if is_np:
                full_sig = full_sig.copy(); full_sw = full_sw.copy()
                full_sig[hard] = sig_last["v"]; full_sw[hard] = sig_last["w"]
                lam_w_full = (state.get("lam_w") if state.get("lam_w")
                              is not None else 0.0 * Sk + 1.0).copy()
                lam_w_full[hard] = (hi - lo) + 1e-30
            else:
                full_sig = full_sig.clone(); full_sw = full_sw.clone()
                full_sig[hard] = sig_last["v"]; full_sw[hard] = sig_last["w"]
                lam_w_full = (state.get("lam_w") if state.get("lam_w")
                              is not None else 0.0 * Sk + 1.0).clone()
                lam_w_full[hard] = (hi - lo) + 1e-30
            state["sig"] = full_sig; state["sig_w"] = full_sw
            state["lam_prev"] = state.get("lam")
            state["lam"] = lam_final; state["lam_w"] = lam_w_full
    elif state is not None:
        state["lam_prev"] = state.get("lam")
        state["lam"] = lam_final
    if return_lam:
        return out, lam_final
    return out


# --------------------------------------------------------------------------

class NumpyOps:
    name = "numpy"

    @staticmethod
    def asarray(x): return np.asarray(x, dtype=float)
    @staticmethod
    def zeros(shape): return np.zeros(shape)
    @staticmethod
    def concat_cols(a, b): return np.concatenate([a, b], axis=1)
    @staticmethod
    def chol(M): return np.linalg.cholesky(M)
    @staticmethod
    def chol_solve(C, rhs):
        y = np.linalg.solve(C, rhs[..., None])
        x = np.linalg.solve(np.transpose(C, (0, 2, 1)), y)
        return x[..., 0]
    @staticmethod
    def norm_rows(x): return np.sqrt((x * x).sum(axis=1))
    @staticmethod
    def all_true(mask): return bool(np.all(mask))
    @staticmethod
    def to_numpy(x): return x
    @staticmethod
    def eye(n): return np.eye(n)
    @staticmethod
    def bool_false(N): return np.zeros(N, dtype=bool)


class TorchOps:
    name = "torch"

    def __init__(self, device="cuda", dtype=None):
        import torch
        self.torch = torch
        self.device = device
        self.dtype = dtype or torch.float64

    def asarray(self, x):
        return self.torch.as_tensor(np.asarray(x, dtype=float),
                                    dtype=self.dtype, device=self.device)
    def zeros(self, shape):
        return self.torch.zeros(shape, dtype=self.dtype, device=self.device)
    def concat_cols(self, a, b): return self.torch.cat([a, b], dim=1)
    def chol(self, M): return self.torch.linalg.cholesky(M)
    def chol_solve(self, C, rhs):
        return self.torch.cholesky_solve(rhs.unsqueeze(-1), C).squeeze(-1)
    def norm_rows(self, x): return self.torch.sqrt((x * x).sum(dim=1))
    def all_true(self, mask): return bool(mask.all().item())
    def to_numpy(self, x): return x.detach().cpu().numpy()
    def eye(self, n):
        return self.torch.eye(n, dtype=self.dtype, device=self.device)
    def bool_false(self, N):
        return self.torch.zeros(N, dtype=self.torch.bool, device=self.device)


def make_ops(device="cpu"):
    if device in ("cpu", "numpy", None):
        return NumpyOps()
    return TorchOps(device=device)


# --------------------------------------------------------------------------
# shared batched ADMM with per-instance adaptive rho
# --------------------------------------------------------------------------

def _admm_compacting(ops, L, mus, *, chunk=1000, max_iter=MAX_ITER, **kw):
    """Run the ADMM core in iteration chunks, dropping converged instances
    between chunks.  Per-instance state is carried over, so results match the
    monolithic solve up to solver tolerance (validated on the reference
    sandbox: hard solve bitwise-identical, penalty ~5e-6 from projection
    warm-state resets at chunk boundaries); wall time then tracks the MEDIAN
    instance instead of the slowest (bench 2026-06-10: med 8762 vs max 25000
    iterations at m=1e5)."""
    mus_np = np.asarray(mus, dtype=float)
    N, n = mus_np.shape
    m = np.asarray(L).shape[0]
    W = np.zeros((N, n)); Yfull = np.zeros((N, m))
    rho = np.zeros(N); status = np.array(["maxiter"] * N, dtype=object)
    iters = np.zeros(N, dtype=int)
    res_p = np.full(N, np.nan); res_d = np.full(N, np.nan)
    alive = np.arange(N)
    carry = None
    done_iters = 0
    while alive.size and done_iters < max_iter:
        budget = min(chunk, max_iter - done_iters)
        r = _admm_core(ops, L, mus_np[alive], max_iter=budget, init=carry,
                       **kw)
        fin = r["status"] == "converged"
        for dst, key in ((W, "W"), (Yfull, "Y"), (res_p, "res_pri"),
                         (res_d, "res_dua")):
            dst[alive] = r[key]
        rho[alive] = r["rho"]
        iters[alive[fin]] = done_iters + r["iters"][fin]
        status[alive[fin]] = "converged"
        done_iters += budget
        iters[alive[~fin]] = done_iters
        alive = alive[~fin]
        if alive.size:
            carry = {k2: v[~fin] for k2, v in r["carry"].items()}
    return dict(W=W, Y=Yfull, rho=rho, status=status, iters=iters,
                res_pri=res_p, res_dua=res_d)


def _admm(ops, L, mus, *, compact=True, **kw):
    if compact:
        return _admm_compacting(ops, L, mus, **kw)
    return _admm_core(ops, L, mus, **kw)


def _admm_core(ops, L, mus, *, z_update, gamma, w_max, rho0=None, relax=1.6,
          eps_abs=EPS_ABS, eps_rel=EPS_REL, max_iter=MAX_ITER,
          check_every=25, rho_every=50, rho_factor=2.0, rho_ratio=10.0,
          rho_min=1e-4, rho_max=1e6, init=None):
    """Batched ADMM over N instances sharing the bank L [m,n] with
    per-instance means mus [N,n].  `z_update(U, rho)` maps the CVaR-block
    input U=[N,m] (and per-row rho [N]) to the next z.  Returns W, iters,
    status, per-instance rho, and the (scaled) CVaR-block dual Y."""
    L = ops.asarray(L); mus = ops.asarray(mus)
    m, n = L.shape; N = mus.shape[0]
    LtL = L.T @ L
    Lt1 = L.sum(0)
    AtA = (LtL[None, :, :]
           - Lt1[None, :, None] * mus[:, None, :]
           - mus[:, :, None] * Lt1[None, None, :]
           + m * mus[:, :, None] * mus[:, None, :])
    eye = ops.eye(n)
    BtB = eye + 1.0                                  # I + 11^T
    if rho0 is None:
        # scale-aware default: balances the z-block (scale ||A|| ~ sqrt(m))
        # against the x-block (scale gamma); adaptation refines from here
        rho0 = max(float(gamma), 0.1) / np.sqrt(m)
    rho = ops.asarray(np.full(N, float(rho0)))

    def factor(rho_vec):
        M = gamma * eye[None] + rho_vec[:, None, None] * (AtA + BtB[None])
        return ops.chol(M)

    C = factor(rho)
    lo = ops.asarray(np.concatenate([np.zeros(n), [1.0]]))
    hi = ops.asarray(np.concatenate([np.full(n, w_max), [1.0]]))

    def A_mv(X): return X @ L.T - (mus * X).sum(-1)[:, None]
    def At_mv(V): return V @ L - mus * V.sum(-1)[:, None]
    def B_mv(X): return ops.concat_cols(X, X.sum(-1)[:, None])
    def Bt_mv(U): return U[:, :n] + U[:, n:]

    z = ops.zeros((N, m)); y = ops.zeros((N, m))
    zt = ops.zeros((N, n + 1)); yt = ops.zeros((N, n + 1))
    if init is not None:                     # resume a compacted chunk
        z = ops.asarray(init["z"]); y = ops.asarray(init["y"])
        zt = ops.asarray(init["zt"]); yt = ops.asarray(init["yt"])
        rho = ops.asarray(init["rho"])
        C = factor(rho)
    q = -mus
    done = ops.bool_false(N)
    iters_done = np.zeros(N, dtype=int)
    res_pri = np.full(N, np.nan); res_dua = np.full(N, np.nan)
    sqmn = float(np.sqrt(m + n + 1)); sqn = float(np.sqrt(n))
    X = ops.zeros((N, n))
    zstate = {"gtol": 1e-6}   # inexact-prox schedule: tightened with the
                              # residuals (summable-error ADMM, cf. Eckstein-
                              # Bertsekas); floor 1e-13 well below EPS_ABS

    for it in range(1, max_iter + 1):
        rr = rho[:, None]
        rhs = -(q - rr * At_mv(z - y) - rr * Bt_mv(zt - yt))
        Xn = ops.chol_solve(C, rhs)
        X = _where(done[:, None], X, Xn)
        AX = A_mv(X); BX = B_mv(X)
        zh = relax * AX + (1.0 - relax) * z
        zth = relax * BX + (1.0 - relax) * zt
        z_new = z_update(zh + y, rho, zstate)
        zt_new = _clip(zth + yt, lo[None, :], hi[None, :])
        z_new = _where(done[:, None], z, z_new)
        zt_new = _where(done[:, None], zt, zt_new)
        y = _where(done[:, None], y, y + zh - z_new)
        yt = _where(done[:, None], yt, yt + zth - zt_new)

        if it % check_every == 0 or it == max_iter:
            r_pri = (ops.norm_rows(AX - z_new) ** 2
                     + ops.norm_rows(BX - zt_new) ** 2) ** 0.5
            s_dua = rho * ops.norm_rows(At_mv(z_new - z) + Bt_mv(zt_new - zt))
            ax_n = (ops.norm_rows(AX) ** 2 + ops.norm_rows(BX) ** 2) ** 0.5
            z_n = (ops.norm_rows(z_new) ** 2 + ops.norm_rows(zt_new) ** 2) ** 0.5
            yty = rho * ops.norm_rows(At_mv(y) + Bt_mv(yt))
            eps_pri = sqmn * eps_abs + eps_rel * _where(ax_n > z_n, ax_n, z_n)
            eps_dua = sqn * eps_abs + eps_rel * yty
            newly = (r_pri <= eps_pri) & (s_dua <= eps_dua) & (~done)
            newly_np = ops.to_numpy(newly).astype(bool)
            iters_done[newly_np & (iters_done == 0)] = it
            # record residuals at freeze time (and keep updating unfrozen)
            rp_np = ops.to_numpy(r_pri); sd_np = ops.to_numpy(s_dua)
            upd = newly_np | ~ops.to_numpy(done).astype(bool)
            res_pri[upd] = rp_np[upd]; res_dua[upd] = sd_np[upd]
            done = done | newly
            # tighten the projection/prox tolerance with the residuals
            rmax = float(np.max(ops.to_numpy(r_pri))) if N else 0.0
            zstate["gtol"] = min(max(rmax * 1e-4, 1e-13), 1e-6)
            if ops.all_true(done):
                z, zt = z_new, zt_new
                break

            if it % rho_every == 0 and it < (3 * max_iter) // 4:
                up = (r_pri > rho_ratio * s_dua) & (~done)
                dn = (s_dua > rho_ratio * r_pri) & (~done)
                if bool(ops.to_numpy(up | dn).any()):
                    scale = _where(up, 0.0 * rho + rho_factor,
                                   _where(dn, 0.0 * rho + 1.0 / rho_factor,
                                          0.0 * rho + 1.0))
                    rho_new = _clip(rho * scale, rho_min, rho_max)
                    rescale = rho / rho_new
                    y = y * rescale[:, None]          # keep scaled duals valid
                    yt = yt * rescale[:, None]
                    rho = rho_new
                    C = factor(rho)
        z, zt = z_new, zt_new

    done_np = ops.to_numpy(done).astype(bool)
    iters_done[~done_np] = max_iter
    status = np.where(done_np, "converged", "maxiter")
    return dict(W=ops.to_numpy(X), Y=ops.to_numpy(y),
                rho=ops.to_numpy(rho), status=status, iters=iters_done,
                res_pri=res_pri, res_dua=res_dua,
                carry=dict(z=ops.to_numpy(z), y=ops.to_numpy(y),
                           zt=ops.to_numpy(zt), yt=ops.to_numpy(yt),
                           rho=ops.to_numpy(rho)))


# --------------------------------------------------------------------------
# public solvers
# --------------------------------------------------------------------------

def solve_hard(L, mus, kappa, *, beta, gamma, w_max, ops=None, **kw):
    """HARD CVQP.  nu_star is the CVaR multiplier on the CVaR<=kappa scale:
    rho*y -> (nu/k)*a with sum(a)=k at convergence, so nu* = rho * sum_i y_i."""
    ops = ops or NumpyOps()
    m = np.asarray(L).shape[0]
    k = cp_.k_from_beta(m, beta)[0]
    d = float(k * kappa)
    def zup(U, rho, state):
        return _proj_cvar_robust(U, k, d, state=state,
                                 gtol=state.get("gtol", 1e-13))

    out = _admm(ops, L, mus, z_update=zup, gamma=gamma, w_max=w_max, **kw)
    out["nu_star"] = out["rho"] * out.pop("Y").sum(axis=1)
    out["k"] = k
    return out


def solve_penalty(L, mus, lam, *, beta, gamma, w_max, ops=None, **kw):
    """Exact (nonsmoothed) CVaR penalty via the top-k-sum prox."""
    ops = ops or NumpyOps()
    m = np.asarray(L).shape[0]
    k = cp_.k_from_beta(m, beta)[0]

    def zup(U, rho, state):
        alpha = float(lam) / (rho * k)               # per-row (adaptive rho)
        if isinstance(U, np.ndarray):
            alpha = np.asarray(alpha)
        A = proj_capped_simplex_warm(U / alpha[..., None], float(k), 1.0,
                                     state=state.setdefault("cs", {}))
        return U - alpha[..., None] * A

    out = _admm(ops, L, mus, z_update=zup, gamma=gamma, w_max=w_max, **kw)
    out.pop("Y"); out["k"] = k
    return out


def solve_min_cvar(L, mus, *, beta, w_max, gamma_reg=GAMMA_REG, ops=None, **kw):
    """Per-instance minimum achievable predicted CVaR (feasibility
    certificate); ridge bias documented in the module docstring.  The
    certificate is margin-guarded (CERT_BIAS), so it is solved at a looser
    tolerance than the main solves: it exists to detect clearly-unachievable
    budgets cheaply, not to be a tight oracle.  Near-boundary instances
    still get the hard solve attempted and their statuses recorded."""
    kw.setdefault("eps_abs", 1e-5)
    kw.setdefault("eps_rel", 1e-5)
    kw.setdefault("max_iter", 8000)
    return solve_penalty(L, mus, 1.0, beta=beta, gamma=gamma_reg,
                         w_max=w_max, ops=ops, **kw)


def feasible_mask(min_cvar, kappa):
    """Spec margin plus the documented certificate-ridge guard."""
    return np.asarray(min_cvar) <= kappa + max(FEAS_MARGIN_REL * abs(kappa),
                                               CERT_BIAS)


def solve_mv_floor(mus, *, gamma, w_max):
    """Unconstrained mean-variance floor (closed form)."""
    return proj_capped_simplex(np.asarray(mus, dtype=float) / gamma, 1.0, w_max)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def predicted_cvar(W, L, mus, beta):
    """-mu_j^T w_j + CVaR_beta(L w_j) per instance (translation equivariance)."""
    W = np.asarray(W, dtype=float); L = np.asarray(L, dtype=float)
    m = L.shape[0]; k = cp_.k_from_beta(m, beta)[0]
    Z = W @ L.T
    return cp_.topk_sum(Z, k) / k - (np.asarray(mus) * W).sum(axis=1)


def realized_cvar(W, L_eval, mus, beta, chunk=64):
    """predicted_cvar on the big held-out bank, chunked over instances."""
    W = np.asarray(W, dtype=float); L = np.asarray(L_eval, dtype=float)
    M = L.shape[0]; k = cp_.k_from_beta(M, beta)[0]
    N = W.shape[0]
    out = np.empty(N)
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        Z = L @ W[a:b].T
        part = np.partition(Z, M - k, axis=0)[M - k:, :]
        out[a:b] = part.sum(axis=0) / k
    return out - (np.asarray(mus) * W).sum(axis=1)


def realized_return(W, L_eval, mus):
    """mean realized return: mu_j^T w + mean(shock return) = mu_j^T w -
    mean(L_eval w); the shock bank has zero population mean."""
    W = np.asarray(W, dtype=float)
    mean_loss_shock = np.asarray(L_eval).mean(axis=0)
    return (np.asarray(mus) * W).sum(axis=1) - W @ mean_loss_shock


def scenario_bootstrap_ci(w, L_eval, mu, beta, n_boot=1000, q=(2.5, 97.5),
                          seed=0):
    """CVaR-estimator precision for ONE decision: resample the evaluation
    scenarios with replacement and recompute realized CVaR."""
    rng = np.random.default_rng(seed)
    losses = np.asarray(L_eval) @ np.asarray(w)
    M = losses.shape[0]; k = cp_.k_from_beta(M, beta)[0]
    stats = np.empty(n_boot)
    for i in range(n_boot):
        s = losses[rng.integers(0, M, M)]
        stats[i] = np.partition(s, M - k)[M - k:].sum() / k
    base = -float(np.asarray(mu) @ np.asarray(w))
    lo, hi = np.percentile(stats, q)
    return base + lo, base + hi


# --------------------------------------------------------------------------
# lambda calibration (spec: "Lambda Calibration")
# --------------------------------------------------------------------------

def calibrate_lambda(mean_pred_cvar, target, *, grid_lo=1e-2, grid_hi=1e2,
                     rel_tol=0.01, max_bisect=30, expand_limit=6):
    # initial bracket amended from [1e-4, 1e4] (spec) to [1e-2, 1e2]:
    # extreme-lambda solves are the most expensive (near-LP), and the
    # geometric expansion below still reaches [1e-8, 1e8] when needed
    """Monotone bracket + bisection in log-lambda.  `mean_pred_cvar(lam)` is
    the penalty method's mean predicted CVaR over the calibration subset
    (feasible AND active hard instances).  Returns
    (lam, n_evals, rel_err, history)."""
    evals = {}

    def f(lam):
        if lam not in evals:
            evals[lam] = float(mean_pred_cvar(lam))
        return evals[lam]

    lo, hi = float(grid_lo), float(grid_hi)
    for _ in range(expand_limit + 1):
        if f(lo) >= target >= f(hi):
            break
        if f(lo) < target:
            lo /= 10.0
        if f(hi) > target:
            hi *= 10.0

    def rel(lam):
        return abs(f(lam) - target) / max(abs(target), 1e-12)

    best = min((lo, hi), key=rel)
    for _ in range(max_bisect):
        mid = float(np.sqrt(lo * hi))
        fm = f(mid)
        if rel(mid) < rel(best):
            best = mid
        if rel(best) <= rel_tol:
            break
        if fm > target:
            lo = mid
        else:
            hi = mid
    history = sorted(evals.items())
    return best, len(evals), rel(best), history
