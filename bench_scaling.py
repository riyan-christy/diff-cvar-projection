"""
bench_scaling.py
================
Forward and backward time + peak memory of the CVaR projection primitive as a
function of the scenario count m, up to large m.  Validates correctness at each
size with a random-direction finite-difference check.

Instances are constructed in the no-plateau (generic, full-measure) regime so
the forward is one sort + O(m); the backward is O(m) in all regimes.  Peak
memory for each m is measured in an isolated subprocess.

Usage:
    python bench_scaling.py            # default sizes
    python bench_scaling.py 1000 10000 100000 1000000 10000000
Outputs:
    results/scaling.csv, results/scaling_time.png, results/scaling_mem.png
"""
import os, sys, time, json, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


# ---- the single-size worker (run in a subprocess for clean peak memory) ----
WORKER = r'''
import sys, time, json, numpy as np, resource
sys.path.insert(0, {here!r})
import cvar_proj as cp_

m = int(sys.argv[1]); reps = int(sys.argv[2]); seed = int(sys.argv[3])
rng = np.random.default_rng(seed)
beta = 0.95
k = int(round((1-beta)*m))                       # integer k
# no-plateau instance: top-k high & separated, mild violation
v = np.empty(m)
idx = rng.permutation(m)
v[idx[:k]]  = rng.normal(12.0, 1.0, k)
v[idx[k:]]  = rng.normal(0.0, 1.0, m-k)
Sk = cp_.topk_sum(v, k)
d  = Sk - 2.0

# warm up + correctness (random-direction central difference)
z = cp_.project_topk_sum(v, k, d)
cert = cp_.extract_certificate(z, k, d)
assert cert["active"], "instance must be active"
g = int(cert["g"])
du = rng.standard_normal(m)
eps = 1e-6
jvp = cp_.jvp_projection(du, cert)
fd  = (cp_.project_topk_sum(v+eps*du, k, d) - cp_.project_topk_sum(v-eps*du, k, d))/(2*eps)
vjp_err = float(np.linalg.norm(jvp-fd)/(np.linalg.norm(fd)+1e-30))

# time forward
t = []
for _ in range(reps):
    t0 = time.perf_counter(); _ = cp_.project_topk_sum(v, k, d); t.append(time.perf_counter()-t0)
fwd_ms = float(np.median(t)*1e3)

# time backward (the contribution)
zbar = rng.standard_normal(m)
t = []
for _ in range(max(reps,5)):
    t0 = time.perf_counter(); _ = cp_.vjp_projection(zbar, cert); t.append(time.perf_counter()-t0)
bwd_ms = float(np.median(t)*1e3)

peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   # KiB on Linux
print(json.dumps(dict(m=m, k=k, g=g, fwd_ms=fwd_ms, bwd_ms=bwd_ms,
                      vjp_err=vjp_err, peak_mb=peak_kb/1024.0)))
'''.format(here=HERE)


def run_one(m, reps, seed):
    code = WORKER
    out = subprocess.run([sys.executable, "-c", code, str(m), str(reps), str(seed)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stderr)
        raise RuntimeError(f"worker failed at m={m}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    sizes = [int(x) for x in sys.argv[1:]] or \
            [1000, 3000, 10000, 30000, 100000, 300000,
             1000000, 3000000, 10000000]
    rows = []
    for m in sizes:
        reps = 11 if m <= 1_000_000 else 5
        r = run_one(m, reps, seed=12345)
        rows.append(r)
        print(f"m={m:>9d}  k={r['k']:>7d}  fwd={r['fwd_ms']:8.2f} ms  "
              f"bwd={r['bwd_ms']:8.3f} ms  peakMem={r['peak_mb']:7.1f} MB  "
              f"VJPerr={r['vjp_err']:.1e}")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "scaling.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # time vs m (log-log) with O(m) reference
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.loglog(df.m, df.bwd_ms, "o-", label="backward (VJP)")
    ax.loglog(df.m, df.fwd_ms, "s--", label="forward (sort+sweep)")
    ref = df.bwd_ms.iloc[0] * df.m / df.m.iloc[0]
    ax.loglog(df.m, ref, ":", color="gray", label="O(m) reference")
    ax.set_xlabel("scenarios m"); ax.set_ylabel("time (ms)")
    ax.set_title("CVaR projection: time vs m"); ax.legend(); ax.grid(True, which="both", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "scaling_time.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.loglog(df.m, df.peak_mb, "o-", label="peak RSS")
    bytes_ref = df.peak_mb.iloc[0] + 8 * (df.m - df.m.iloc[0]) / 1e6 * 4  # ~few O(m) arrays
    ax.loglog(df.m, bytes_ref, ":", color="gray", label="linear-in-m reference")
    ax.set_xlabel("scenarios m"); ax.set_ylabel("peak memory (MB)")
    ax.set_title("CVaR projection: peak memory vs m"); ax.legend(); ax.grid(True, which="both", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "scaling_mem.png"), dpi=150)

    print("\nwrote results/scaling.csv, results/scaling_time.png, results/scaling_mem.png")
    print(f"max VJP finite-difference error across sizes: {df.vjp_err.max():.1e}")


if __name__ == "__main__":
    main()

