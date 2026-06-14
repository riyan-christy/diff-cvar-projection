"""
bench_headtohead.py
===================
Head-to-head comparison, for the CVaR projection map v -> Pi_C(v):

    PROPOSED : fast forward (sort+sweep) + custom active-face VJP   [this work]
    EPIGRAPH : the deterministic-equivalent CVaR epigraph QP, solved and
               differentiated by cvxpylayers / diffcp                [baseline]

The epigraph adds (m+1) auxiliary variables (alpha, t) and 2m inequalities --
exactly the expansion the specialized method avoids.

We report:
  * gradient agreement (proposed VJP vs cvxpylayers VJP) on no-tie instances,
  * forward time, backward time, peak memory vs m, each in an isolated process.

cvxpylayers is expected to slow down / run out of memory well before the
specialized method does.

Outputs: results/headtohead.csv, results/headtohead_time.png,
         results/headtohead_mem.png
"""
import os, sys, json, time, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
BETA = 0.95


def make_instance(m, seed):
    """No-tie instance: separated top-k group, mild violation."""
    rng = np.random.default_rng(seed)
    k = int(round((1 - BETA) * m))
    v = np.empty(m)
    idx = rng.permutation(m)
    v[idx[:k]] = rng.normal(12.0, 1.0, k)
    v[idx[k:]] = rng.normal(0.0, 1.0, m - k)
    import cvar_proj as cp_
    d = float(cp_.topk_sum(v, k) - 2.0)
    return v, k, d


# ----------------------------- agreement check -----------------------------
def agreement():
    import torch, cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
    import cvar_proj as cp_
    print("=" * 70)
    print("gradient agreement: proposed VJP vs cvxpylayers (no-tie)")
    print("=" * 70)
    for m in (200, 1000):
        v, k, d = make_instance(m, seed=1)
        # epigraph layer:  min .5||z-v||^2  s.t.  k*alpha + sum(t) <= d, t>=z-alpha, t>=0
        z = cp.Variable(m); al = cp.Variable(); t = cp.Variable(m)
        vp = cp.Parameter(m)
        prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(z - vp)),
                          [k * al + cp.sum(t) <= d, t >= z - al, t >= 0])
        layer = CvxpyLayer(prob, parameters=[vp], variables=[z])

        rng = np.random.default_rng(7)
        errs = []
        for _ in range(5):
            zbar = rng.standard_normal(m)
            # proposed
            zp = cp_.project_topk_sum(v, k, d)
            cert = cp_.extract_certificate(zp, k, d)
            g_ours = cp_.vjp_projection(zbar, cert)
            # cvxpylayers
            vt = torch.tensor(v, dtype=torch.double, requires_grad=True)
            (zt,) = layer(vt, solver_args={"eps": 1e-9, "max_iters": 50000})
            zt.backward(torch.tensor(zbar, dtype=torch.double))
            g_cvx = vt.grad.detach().numpy()
            errs.append(np.linalg.norm(g_ours - g_cvx) / (np.linalg.norm(g_cvx) + 1e-30))
        print(f"   m={m:5d}  max rel gradient disagreement = {max(errs):.2e}")


# ------------------------------- timing worker ------------------------------
WORKER = r'''
import sys, os, json, time, numpy as np, resource
sys.path.insert(0, "__HERE__")
import cvar_proj as cp_

method = sys.argv[1]; m = int(sys.argv[2]); seed = int(sys.argv[3])
BETA = __BETA__
rng = np.random.default_rng(seed)
k = int(round((1-BETA)*m))
v = np.empty(m); idx = rng.permutation(m)
v[idx[:k]] = rng.normal(12.0,1.0,k); v[idx[k:]] = rng.normal(0.0,1.0,m-k)
d = float(cp_.topk_sum(v,k)-2.0)
zbar = rng.standard_normal(m)

if method == "proposed":
    t=[]
    for _ in range(7):
        t0=time.perf_counter(); z=cp_.project_topk_sum(v,k,d); cert=cp_.extract_certificate(z,k,d); t.append(time.perf_counter()-t0)
    fwd=float(np.median(t))
    t=[]
    for _ in range(20):
        t0=time.perf_counter(); _=cp_.vjp_projection(zbar,cert); t.append(time.perf_counter()-t0)
    bwd=float(np.median(t))
else:  # epigraph via cvxpylayers
    import torch, cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
    z=cp.Variable(m); al=cp.Variable(); tt=cp.Variable(m); vp=cp.Parameter(m)
    prob=cp.Problem(cp.Minimize(0.5*cp.sum_squares(z-vp)),
                    [k*al+cp.sum(tt)<=d, tt>=z-al, tt>=0])
    layer=CvxpyLayer(prob, parameters=[vp], variables=[z])
    vt=torch.tensor(v, dtype=torch.double, requires_grad=True)
    t0=time.perf_counter(); (zt,)=layer(vt, solver_args={"eps":1e-7,"max_iters":20000}); fwd=time.perf_counter()-t0
    bt=torch.tensor(zbar, dtype=torch.double)
    t0=time.perf_counter(); zt.backward(bt); bwd=time.perf_counter()-t0

peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0
print(json.dumps(dict(method=method, m=m, fwd_ms=fwd*1e3, bwd_ms=bwd*1e3, peak_mb=peak_mb)))
'''.replace("__HERE__", HERE).replace("__BETA__", repr(BETA))


def run_worker(method, m, seed=1, timeout=300):
    try:
        out = subprocess.run([sys.executable, "-c", WORKER, method, str(m), str(seed)],
                             capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return dict(method=method, m=m, fwd_ms=np.nan, bwd_ms=np.nan,
                    peak_mb=np.nan, note="timeout")
    if out.returncode != 0:
        return dict(method=method, m=m, fwd_ms=np.nan, bwd_ms=np.nan,
                    peak_mb=np.nan, note=out.stderr.strip().splitlines()[-1][:80]
                    if out.stderr.strip() else "error")
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    agreement()
    print("=" * 70)
    print("timing / memory vs m (isolated processes)")
    print("=" * 70)
    sizes = [1000, 3000, 10000, 30000, 100000]
    rows = []
    for m in sizes:
        for method in ("proposed", "epigraph"):
            r = run_worker(method, m, timeout=300)
            rows.append(r)
            note = r.get("note", "")
            print(f"   {method:9s} m={m:>7d}  fwd={r['fwd_ms']:9.2f} ms  "
                  f"bwd={r['bwd_ms']:9.2f} ms  peak={r['peak_mb']:7.1f} MB  {note}")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "headtohead.csv"), index=False)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = df[df.method == "proposed"]; E = df[df.method == "epigraph"]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.loglog(P.m, P.fwd_ms + P.bwd_ms, "o-", label="proposed (fwd+bwd)")
    ax.loglog(E.m, E.fwd_ms + E.bwd_ms, "s-", label="epigraph (fwd+bwd)")
    ax.set_xlabel("scenarios m"); ax.set_ylabel("time (ms)")
    ax.set_title("Projection + VJP: proposed vs epigraph"); ax.legend()
    ax.grid(True, which="both", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "headtohead_time.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.loglog(P.m, P.peak_mb, "o-", label="proposed")
    ax.loglog(E.m, E.peak_mb, "s-", label="epigraph")
    ax.set_xlabel("scenarios m"); ax.set_ylabel("peak memory (MB)")
    ax.set_title("Projection + VJP: peak memory"); ax.legend()
    ax.grid(True, which="both", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "headtohead_mem.png"), dpi=150)

    print("\nwrote results/headtohead.csv and two PNGs")


if __name__ == "__main__":
    main()
