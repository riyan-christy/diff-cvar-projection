"""
exp_secondary.py
================
Secondary capability demo (EXPERIMENT_EXECUTION_SPEC.md): end-to-end training
through the differentiable CVaR layer at m = 1e5 with the specialized VJP,
plus the epigraph negative control (CVXPYlayers/diffcp timeout/OOM).

Frozen training setup (spec):
  * theta in R^8 enters predicted losses in rank-one operator form
        L_theta w = L w + sum_p theta_p * u_p (v_p^T w),
    with fixed random unit vectors u_p in R^m, v_p in R^n (never materialize
    the dense perturbation);
  * loss(theta) = mean_j [ -mu_j^T w_j*(theta) + CVaR_beta(L_val w_j*(theta)) ]
    on the held-out validation shock bank --- the loss exists to drive
    gradients through the layer, not to claim decision quality;
  * w_j*(theta) = output of T = 50 unrolled ADMM iterations, fixed cold start;
  * Adam, lr = 1e-2, 100 steps, 32 instances per step;
  * VJP-use counters: per-step fraction of unrolled iterations whose CVaR
    projection is active, and ||grad_theta||; the demo kappa must bind
    (mean active fraction > 0.5 asserted) or the run is reconfigured, not
    reported.

The layer's forward projection is the plateau-robust batched solve from
exp_solvers (ADMM iterates hit plateaus generically); the backward is the
paper's certificate VJP (cvar_proj.extract_certificate_torch +
vjp_projection_torch), demonstrating the oracle-agnostic interface.

Requires torch + CUDA for the headline m=1e5 run (CPU works at small m for
the gradient check).  Outputs: results/secondary/*.csv.
"""
import argparse, csv, json, os, time
import numpy as np

import cvar_proj as cp_
import exp_data as D
import exp_solvers as S

HERE = os.path.dirname(os.path.abspath(__file__))


def make_layer_projection(torch):
    """Custom autograd Function: robust forward + certificate VJP backward."""

    class RobustCVaRProject(torch.autograd.Function):
        """Robust forward + certificate backward.  `d` may be a 0-dim tensor
        with requires_grad: the d-adjoint flows via rhs_adjoints_torch
        (summed over the batch, since one constraint level is shared)."""
        active_rows = 0     # VJP-use counters (read/reset by the trainer)
        total_rows = 0

        @staticmethod
        def forward(ctx, V, k, d):
            dv = float(d)
            act = torch.topk(V, k, dim=1).values.sum(1) > dv + 1e-12
            Z = S._proj_cvar_robust(V.detach(), k, dv, gtol=1e-12)
            ctx.cert = cp_.extract_certificate_torch(Z, k, dv, active=act)
            ctx.d_needs_grad = torch.is_tensor(d) and d.requires_grad
            RobustCVaRProject.active_rows += int(act.sum().item())
            RobustCVaRProject.total_rows += int(act.numel())
            return Z

        @staticmethod
        def backward(ctx, Zbar):
            vbar = cp_.vjp_projection_torch(Zbar, ctx.cert)
            dbar = None
            if ctx.d_needs_grad:
                db, _kb = cp_.rhs_adjoints_torch(Zbar, ctx.cert)
                dbar = db.sum()
            return vbar, None, dbar

    return RobustCVaRProject


class ThetaCVQPLayer:
    """Unrolled ADMM CVQP layer whose scenario bank depends on theta in
    rank-one operator form; everything differentiable in torch."""

    def __init__(self, torch, L, mus, Us, Vs, kappa, beta, gamma, w_max,
                 rho=None, relax=1.6, T=50):
        self.t = torch
        self.L, self.mus, self.Us, self.Vs = L, mus, Us, Vs
        m, n = L.shape
        self.m, self.n = m, n
        self.k = cp_.k_from_beta(m, beta)[0]
        self.d = self.k * kappa
        self.gamma, self.w_max, self.relax, self.T = gamma, w_max, relax, T
        self.rho = rho if rho is not None else max(gamma, 0.1) / np.sqrt(m)
        self.proj = make_layer_projection(torch)
        # constants for M(theta) and operator products
        self.LtL = L.T @ L                       # [n,n]
        self.Lt1 = L.sum(0)                      # [n]
        self.LtU = L.T @ Us.T                    # [n,8]
        self.UtU = Us @ Us.T                     # [8,8]
        self.Ut1 = Us.sum(1)                     # [8]
        self.eye = torch.eye(n, dtype=L.dtype, device=L.device)

    def _ops(self, theta):
        """theta-dependent operator pieces.  L_th = L + sum_p th_p u_p v_p^T;
        A_j = L_th - 1 mu_j^T  applied in operator form."""
        t = self.t
        Vth = self.Vs.T @ theta if False else (theta[:, None] * self.Vs)  # [8,n]

        def Lth_mv(X):          # [B,n] -> [B,m]
            base = X @ self.L.T
            coef = X @ self.Vs.T                       # [B,8]
            return base + (coef * theta[None, :]) @ self.Us

        def Lth_rmv(Y):         # [B,m] -> [B,n]
            base = Y @ self.L
            coef = Y @ self.Us.T                       # [B,8]
            return base + (coef * theta[None, :]) @ self.Vs

        # Lth^T Lth = LtL + sum_p th_p (v_p (L^T u_p)^T + (L^T u_p) v_p^T)
        #            + sum_pq th_p th_q (u_p . u_q) v_p v_q^T
        cross = self.LtU * theta[None, :]              # [n,8]
        quad = (theta[:, None] * self.UtU) * theta[None, :]
        LthtLth = (self.LtL + cross @ self.Vs + self.Vs.T @ cross.T
                   + self.Vs.T @ quad @ self.Vs)
        Ltht1 = self.Lt1 + self.Vs.T @ (theta * self.Ut1)
        return Lth_mv, Lth_rmv, LthtLth, Ltht1

    def _setup(self, theta, mus_b):
        """Operator algebra shared by presolve (no-grad) and solve
        (differentiable); one source of truth for M(theta) and the A/B
        maps."""
        t, m, n = self.t, self.m, self.n
        rho = self.rho
        Lmv, Lrmv, LtL, Lt1 = self._ops(theta)
        AtA = (LtL[None] - Lt1[None, :, None] * mus_b[:, None, :]
               - mus_b[:, :, None] * Lt1[None, None, :]
               + m * mus_b[:, :, None] * mus_b[:, None, :])
        M = (self.gamma * self.eye[None]
             + rho * (AtA + self.eye[None] + 1.0))
        C = t.linalg.cholesky(M)
        lo = t.cat([t.zeros(n, dtype=M.dtype, device=M.device),
                    t.ones(1, dtype=M.dtype, device=M.device)])
        hi = t.cat([t.full((n,), self.w_max, dtype=M.dtype, device=M.device),
                    t.ones(1, dtype=M.dtype, device=M.device)])

        def A_mv(X): return Lmv(X) - (mus_b * X).sum(-1)[:, None]
        def At_mv(Y): return Lrmv(Y) - mus_b * Y.sum(-1)[:, None]
        def B_mv(X): return t.cat([X, X.sum(-1)[:, None]], dim=1)
        def Bt_mv(U): return U[:, :n] + U[:, n:]
        return C, lo, hi, A_mv, At_mv, B_mv, Bt_mv

    def _state0(self, B, ref, init):
        t, m, n = self.t, self.m, self.n
        if init is None:
            return (t.zeros(B, m, dtype=ref.dtype, device=ref.device),
                    t.zeros(B, m, dtype=ref.dtype, device=ref.device),
                    t.zeros(B, n + 1, dtype=ref.dtype, device=ref.device),
                    t.zeros(B, n + 1, dtype=ref.dtype, device=ref.device))
        return (init["z"].clone(), init["y"].clone(),
                init["zt"].clone(), init["yt"].clone())

    def presolve(self, theta, mus_b, d_override=None, state=None,
                 eps=1e-4, max_iter=12000, check_every=50):
        """No-grad ADMM at the current (theta, d) to the working tolerance;
        returns a detached state for solve(init=...).

        2026-06-12 amendment: the T=50 COLD-start unroll at m=1e5 never
        reaches the constraint surface (measured mean active fraction 0.00
        over 100 steps; ~4e3 iterations are needed there), so it
        differentiated a transient and the VJP never fired.  Unrolling FROM
        the solved point keeps the iterates at the binding face (certificate
        and plateau branches live) and makes the T-step truncated adjoint
        approximate the implicit gradient.  This phase bypasses the autograd
        Function, so the VJP-use counters count only the differentiable
        unroll."""
        t = self.t
        B = mus_b.shape[0]
        rho, relax = self.rho, self.relax
        with t.no_grad():
            th = theta.detach() if t.is_tensor(theta) else theta
            C, lo, hi, A_mv, At_mv, B_mv, Bt_mv = self._setup(th, mus_b)
            d_use = float(self.d if d_override is None else d_override)
            z, y, zt, yt = self._state0(B, C, state)
            q = -mus_b
            for it in range(1, max_iter + 1):
                rhs = -(q - rho * At_mv(z - y) - rho * Bt_mv(zt - yt))
                X = t.cholesky_solve(rhs.unsqueeze(-1), C).squeeze(-1)
                AX = A_mv(X); BX = B_mv(X)
                zh = relax * AX + (1 - relax) * z
                zth = relax * BX + (1 - relax) * zt
                z_prev, zt_prev = z, zt
                z = S._proj_cvar_robust(zh + y, self.k, d_use, gtol=1e-12)
                zt = t.clamp(zth + yt, lo[None, :], hi[None, :])
                y = y + zh - z
                yt = yt + zth - zt
                if it % check_every == 0:
                    sc = 1.0 + max(float(AX.norm()), float(z.norm()),
                                   float(BX.norm()), float(zt.norm()))
                    rp = max(float((AX - z).norm()),
                             float((BX - zt).norm()))
                    rd = rho * max(float(At_mv(z - z_prev).norm()),
                                   float(Bt_mv(zt - zt_prev).norm()))
                    if rp <= eps * sc and rd <= eps * sc:
                        break
            else:
                print(f"  WARNING: presolve hit max_iter={max_iter} "
                      f"(rp={rp:.2e}, rd={rd:.2e}); proceeding", flush=True)
        return dict(z=z, y=y, zt=zt, yt=yt, iters=it)

    def solve(self, theta, mus_b, d_override=None, init=None):
        """Batched T-step differentiable unroll for instances with means
        mus_b [B,n], from `init` (a presolve state, treated as constant)
        or cold zeros."""
        t = self.t
        B = mus_b.shape[0]
        rho, relax = self.rho, self.relax
        C, lo, hi, A_mv, At_mv, B_mv, Bt_mv = self._setup(theta, mus_b)
        d_use = self.d if d_override is None else d_override
        z, y, zt, yt = self._state0(B, C, init)
        q = -mus_b
        for _ in range(self.T):
            rhs = -(q - rho * At_mv(z - y) - rho * Bt_mv(zt - yt))
            X = t.cholesky_solve(rhs.unsqueeze(-1), C).squeeze(-1)
            AX = A_mv(X); BX = B_mv(X)
            zh = relax * AX + (1 - relax) * z
            zth = relax * BX + (1 - relax) * zt
            z = self.proj.apply(zh + y, self.k, d_use)
            zt = t.clamp(zth + yt, lo[None, :], hi[None, :])
            y = y + zh - z
            yt = yt + zth - zt
        return X


def gradient_check(torch, device, m=2000, n=10, T=20, n_dirs=5, tol=1e-4):
    """Two-scale central-FD check of d loss / d theta through the unrolled
    layer at a certificate-stable point (spec, amended 2026-06-12).

    The forward runs inner bisections truncated at gtol=1e-12, so the FD
    probe carries an input-discontinuous noise term ~ gtol/(2*eps): ~1e-5
    relative at eps=1e-6 (measured 1.798e-5, A100 2026-06-12), ~1e-6 at
    eps=1e-5.  The analytic certificate VJP is eps-independent and is
    separately verified at ~2e-9..3.5e-9 in test_correctness; a WRONG
    gradient shows up here at >=1e-2.  Hence: per-direction min over
    eps in {1e-6, 1e-5} (real bias survives both scales, FD noise does
    not), gated at 1e-4."""
    cfg = dict(beta=0.95, gamma=1.0, w_max=0.30)
    struct = D.make_structure(0, n=n, K=3)
    rngs = np.random.default_rng(1234)
    L_np = D.make_shock_bank(struct, "id", m, 0, 0, 1)
    mus_np = D.make_mus(struct, "id", 4, 0, 0, 1)
    # binding kappa via the blind rule
    mc = S.solve_min_cvar(L_np, mus_np, beta=cfg["beta"], w_max=cfg["w_max"],
                          max_iter=1500, eps_abs=1e-4, eps_rel=1e-4)
    lo_k = np.median(S.predicted_cvar(mc["W"], L_np, mus_np, cfg["beta"]))
    W_f = S.solve_mv_floor(mus_np, gamma=cfg["gamma"], w_max=cfg["w_max"])
    hi_k = np.median(S.predicted_cvar(W_f, L_np, mus_np, cfg["beta"]))
    kappa = float(lo_k + 0.5 * (hi_k - lo_k))

    dt = torch.float64
    L = torch.as_tensor(L_np, dtype=dt, device=device)
    mus = torch.as_tensor(mus_np, dtype=dt, device=device)
    Us = torch.as_tensor(rngs.standard_normal((8, m)), dtype=dt, device=device)
    Us = Us / Us.norm(dim=1, keepdim=True)
    Vs = torch.as_tensor(rngs.standard_normal((8, n)), dtype=dt, device=device)
    Vs = Vs / Vs.norm(dim=1, keepdim=True)
    Lval = torch.as_tensor(D.make_shock_bank(struct, "id", m, 0, 1, 3),
                           dtype=dt, device=device)
    layer = ThetaCVQPLayer(torch, L, mus, Us, Vs, kappa, cfg["beta"],
                           cfg["gamma"], cfg["w_max"], T=T)
    kv = cp_.k_from_beta(m, cfg["beta"])[0]

    ws = layer.presolve(torch.zeros(8, dtype=dt, device=device), mus)
    print(f"  warm presolve: {ws['iters']} iters (frozen across FD evals)",
          flush=True)

    def loss_of(theta):
        W = layer.solve(theta, mus, init=ws)
        zv = W @ Lval.T
        cv = torch.topk(zv, kv, dim=1).values.sum(1) / kv
        return (-(mus * W).sum(1) + cv).mean()

    theta0 = torch.zeros(8, dtype=dt, device=device, requires_grad=True)
    loss = loss_of(theta0)
    loss.backward()
    g = theta0.grad.detach().clone()
    rng = np.random.default_rng(7)
    eps_list = (1e-6, 1e-5)
    errs = []
    for i in range(n_dirs):
        u = torch.as_tensor(rng.standard_normal(8), dtype=dt, device=device)
        u = u / u.norm()
        an = (g * u).sum()
        per_eps = []
        for eps in eps_list:
            with torch.no_grad():
                lp = loss_of(theta0.detach() + eps * u)
                lm = loss_of(theta0.detach() - eps * u)
            fd = (lp - lm) / (2 * eps)
            per_eps.append(abs(float(fd - an)) / (abs(float(fd)) + 1e-12))
        print("  dir %d: " % i + "  ".join(
            "eps=%.0e rel_err=%.3e" % (e, v)
            for e, v in zip(eps_list, per_eps)), flush=True)
        errs.append(min(per_eps))
    err = max(errs)
    print(f"gradient check: max over {n_dirs} dirs of min-over-eps "
          f"rel err = {err:.3e} (tol {tol:.0e})")
    assert err <= tol, "unrolled-layer gradient check failed"
    return err


def _negctrl_child(conn, m):
    """Child body for negative_control.  MODULE level because the 'spawn'
    start method (required: the parent holds a CUDA context) pickles the
    target by qualified name; a nested function is not picklable (the
    2026-06-12 crash)."""
    try:
        import torch, cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer
        k = max(int(round(0.01 * m)), 1)
        z = cp.Variable(m); al = cp.Variable(); tt = cp.Variable(m)
        vp = cp.Parameter(m)
        prob = cp.Problem(
            cp.Minimize(0.5 * cp.sum_squares(z - vp)),
            [k * al + cp.sum(tt) <= 0.5 * k, tt >= z - al, tt >= 0])
        layer = CvxpyLayer(prob, parameters=[vp], variables=[z])
        v = torch.randn(m, dtype=torch.double, requires_grad=True)
        zs, = layer(v)
        zs.sum().backward()
        conn.send("completed")
    except MemoryError:
        conn.send("oom")
    except Exception as e:
        conn.send(f"error:{type(e).__name__}")


def negative_control(m, outpath, timeout_s=300):
    """One epigraph backward attempt at scale via CVXPYlayers/diffcp in a
    child process; record completed/timeout/oom (spec: 300 s per phase)."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    parent, kid = ctx.Pipe()
    p = ctx.Process(target=_negctrl_child, args=(kid, m))
    t0 = time.time(); p.start(); p.join(timeout_s)
    if p.is_alive():
        p.terminate(); p.join()
        outcome = "timeout"
    else:
        outcome = parent.recv() if parent.poll() else "killed(oom)"
    wall = time.time() - t0
    new = not os.path.exists(outpath)
    with open(outpath, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["m", "outcome", "wall_s", "timeout_s"])
        w.writerow([m, outcome, f"{wall:.1f}", timeout_s])
    print(f"negative control m={m}: {outcome} ({wall:.1f}s)")
    return outcome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=100_000)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(HERE, "results",
                                                  "secondary"))
    ap.add_argument("--gradcheck-only", action="store_true")
    ap.add_argument("--skip-negative-control", action="store_true")
    ap.add_argument("--negative-control-only", action="store_true",
                    help="run only the epigraph negative control (training "
                         "already complete; avoid duplicating training.csv)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch
    device = args.device if (args.device == "cpu"
                             or torch.cuda.is_available()) else "cpu"

    if args.negative_control_only:
        negative_control(args.m, os.path.join(args.out,
                                              "negative_control.csv"))
        return

    err = gradient_check(torch, "cpu")
    json.dump(dict(gradcheck_rel_err=err),
              open(os.path.join(args.out, "gradcheck.json"), "w"))
    if args.gradcheck_only:
        return

    beta, gamma, w_max = 0.99, 1.0, 0.20
    dt = torch.float64
    tpath = os.path.join(args.out, "training.csv")
    new = not os.path.exists(tpath)
    tf = open(tpath, "a", newline=""); tw = csv.writer(tf)
    if new:
        tw.writerow(["seed", "step", "loss", "grad_norm", "active_frac",
                     "step_ms", "gpu_mem_mb"])

    for seed in range(args.seeds):
        struct = D.make_structure(seed, n=args.n)
        L_np = D.make_shock_bank(struct, "id", args.m, seed, 0, 1)
        mus_np = D.make_mus(struct, "id", args.batch, seed, 0, 1)
        # binding kappa (blind rule, midpoint)
        mc = S.solve_min_cvar(L_np, mus_np, beta=beta, w_max=w_max,
                              max_iter=2000, eps_abs=1e-4, eps_rel=1e-4)
        lo_k = np.median(S.predicted_cvar(mc["W"], L_np, mus_np, beta))
        W_f = S.solve_mv_floor(mus_np, gamma=gamma, w_max=w_max)
        hi_k = np.median(S.predicted_cvar(W_f, L_np, mus_np, beta))
        kappa = float(lo_k + 0.5 * (hi_k - lo_k))

        rngs = np.random.default_rng(9000 + seed)
        L = torch.as_tensor(L_np, dtype=dt, device=device)
        mus = torch.as_tensor(mus_np, dtype=dt, device=device)
        Us = torch.as_tensor(rngs.standard_normal((8, args.m)), dtype=dt,
                             device=device)
        Us = Us / Us.norm(dim=1, keepdim=True)
        Vs = torch.as_tensor(rngs.standard_normal((8, args.n)), dtype=dt,
                             device=device)
        Vs = Vs / Vs.norm(dim=1, keepdim=True)
        Lval = torch.as_tensor(D.make_shock_bank(struct, "id", args.m, seed,
                                                 1, 3), dtype=dt,
                               device=device)
        layer = ThetaCVQPLayer(torch, L, mus, Us, Vs, kappa, beta, gamma,
                               w_max, T=args.T)
        kv = layer.k
        theta = torch.zeros(8, dtype=dt, device=device, requires_grad=True)
        opt = torch.optim.Adam([theta], lr=args.lr)
        Proj = layer.proj
        act_fracs = []
        ws = None
        for step in range(args.steps):
            t0 = time.perf_counter()
            ws = layer.presolve(theta, mus, state=ws)
            Proj.active_rows = Proj.total_rows = 0
            opt.zero_grad()
            W = layer.solve(theta, mus, init=ws)
            zv = W @ Lval.T
            cv = torch.topk(zv, kv, dim=1).values.sum(1) / kv
            loss = (-(mus * W).sum(1) + cv).mean()
            loss.backward()
            gnorm = float(theta.grad.norm())
            afrac = Proj.active_rows / max(Proj.total_rows, 1)
            act_fracs.append(afrac)
            opt.step()
            if device != "cpu":
                torch.cuda.synchronize()
            mem = (torch.cuda.max_memory_allocated() / 2**20
                   if device != "cpu" else 0.0)
            tw.writerow([seed, step, f"{float(loss):.8g}", f"{gnorm:.6g}",
                         f"{afrac:.4f}",
                         f"{(time.perf_counter()-t0)*1e3:.1f}",
                         f"{mem:.0f}"])
            tf.flush()
            if step % 10 == 0:
                print(f"  seed {seed} step {step}: loss={float(loss):.6g} "
                      f"afrac={afrac:.2f} presolve={ws['iters']} "
                      f"({(time.perf_counter() - t0)*1e3:.0f} ms)",
                      flush=True)
            assert gnorm > 0, "zero gradient: demo does not exercise the VJP"
        mean_af = float(np.mean(act_fracs))
        print(f"seed {seed}: kappa={kappa:.4g} mean active fraction "
              f"{mean_af:.2f}")
        assert mean_af > 0.5, ("constraint not binding enough; reconfigure "
                               "kappa, do not report this run (spec)")

        # ---- learnable-kappa variant (spec: 20-step smoke test of the
        # d/kappa adjoint): kappa = kappa_min + softplus(eta)
        kpath = os.path.join(args.out, "training_kappa.csv")
        knew = not os.path.exists(kpath)
        kf = open(kpath, "a", newline=""); kw = csv.writer(kf)
        if knew:
            kw.writerow(["seed", "step", "loss", "kappa", "grad_eta"])
        kappa_min = float(lo_k)
        eta0 = float(np.log(np.expm1(max(kappa - kappa_min, 1e-3))))
        eta = torch.tensor(eta0, dtype=dt, device=device,
                           requires_grad=True)
        opt_k = torch.optim.Adam([eta], lr=1e-2)
        theta_frozen = torch.zeros(8, dtype=dt, device=device)
        ws_k = None
        for step in range(20):
            opt_k.zero_grad()
            kap_t = kappa_min + torch.nn.functional.softplus(eta)
            d_t = layer.k * kap_t
            ws_k = layer.presolve(theta_frozen, mus, d_override=d_t,
                                  state=ws_k)
            W = layer.solve(theta_frozen, mus, d_override=d_t, init=ws_k)
            zv = W @ Lval.T
            cv = torch.topk(zv, kv, dim=1).values.sum(1) / kv
            loss = (-(mus * W).sum(1) + cv).mean()
            loss.backward()
            ge = float(eta.grad.abs())
            kw.writerow([seed, step, f"{float(loss):.8g}",
                         f"{float(kap_t):.6g}", f"{ge:.6g}"])
            kf.flush()
            assert ge > 0, "zero kappa gradient: d/kappa adjoint not exercised"
            opt_k.step()
        kf.close()
        print(f"seed {seed}: kappa-adjoint variant OK (20 steps)")
    tf.close()

    if not args.skip_negative_control:
        negative_control(args.m, os.path.join(args.out,
                                              "negative_control.csv"))


if __name__ == "__main__":
    main()
