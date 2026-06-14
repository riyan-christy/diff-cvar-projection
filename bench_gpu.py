"""
bench_gpu.py
============
A100 GPU-native scaling for the CVaR projection primitive.

Both the forward projection (sort + sweep) and the backward VJP run entirely on
the GPU via the torch-native routines in cvar_proj.py (no host round-trip).

Before timing anything, the script SELF-VALIDATES the torch forward and VJP
against the verified numpy reference (this is the safety net, since the torch
path is exercised for the first time on the GPU). It then reports:

  (A) single-instance scaling: forward time, backward time, peak GPU memory,
      and a random-direction finite-difference VJP check, for m up to ~1e8;
  (B) batched throughput: problems/second at fixed m as the batch size grows;
  (C) CPU(numpy) vs GPU(torch) speedup at matched sizes.

Outputs: results/gpu_scaling.csv, results/gpu_batched.csv,
         results/gpu_scaling_time.png, results/gpu_scaling_mem.png,
         results/gpu_throughput.png

Usage:
    python bench_gpu.py
    python bench_gpu.py --sizes 10000 100000 1000000 10000000 100000000
    python bench_gpu.py --batch-m 10000 --batch-sizes 1 8 64 512 4096 --device cuda
"""
import argparse, os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cvar_proj as cp_

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
BETA = 0.95


def get_device(name):
    import torch
    if name == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but CUDA is unavailable; "
              "falling back to CPU. (Run this on the A100 for the real result.)")
        return torch.device("cpu")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_instance(m, device, dtype, seed=12345):
    """No-plateau (generic, full-measure) instance: separated top-k, mild violation."""
    import torch
    k = int(round((1 - BETA) * m))
    g = torch.Generator().manual_seed(seed)
    v = torch.empty(m, dtype=dtype)
    idx = torch.randperm(m, generator=g)
    v[idx[:k]] = torch.normal(12.0, 1.0, (k,), generator=g).to(dtype)
    v[idx[k:]] = torch.normal(0.0, 1.0, (m - k,), generator=g).to(dtype)
    v = v.to(device)
    d = float(torch.topk(v, k).values.sum().item() - 2.0)
    return v, k, d


def timed(fn, device, reps):
    """Median ms over `reps` calls, with warmup and CUDA-correct timing."""
    import torch
    fn()  # warmup
    if device.type == "cuda":
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return float(np.median(ts))
    else:
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
        return float(np.median(ts))


def self_validate(device, dtype):
    """Torch forward + VJP must match the numpy reference. Hard gate."""
    import torch
    print("=" * 72)
    print("self-validation: torch-native vs numpy reference (no-tie + plateau)")
    print("=" * 72)
    rng = np.random.default_rng(0)
    worst_fwd = worst_vjp = 0.0
    for _ in range(25):
        m = int(rng.integers(200, 2000))
        k = int(rng.integers(2, m // 3))
        v = np.concatenate([rng.normal(10, 1, k), rng.normal(0, 1, m - k)])
        rng.shuffle(v)
        d = float(cp_.topk_sum(v, k) - 2.0)
        z_np = cp_.project_topk_sum(v, k, d)
        vt = torch.tensor(v, dtype=dtype, device=device)
        z_t = cp_.project_topk_sum_torch(vt, k, d).cpu().numpy()
        worst_fwd = max(worst_fwd, np.linalg.norm(z_t - z_np) / (np.linalg.norm(z_np) + 1e-30))
        zbar = rng.standard_normal(m)
        cert_np = cp_.extract_certificate(z_np, k, d)
        g_np = cp_.vjp_projection(zbar, cert_np)
        cert_t = cp_.extract_certificate_torch(torch.tensor(z_t, dtype=dtype, device=device), k, d)
        g_t = cp_.vjp_projection_torch(torch.tensor(zbar, dtype=dtype, device=device), cert_t).cpu().numpy()
        worst_vjp = max(worst_vjp, np.linalg.norm(g_t - g_np) / (np.linalg.norm(g_np) + 1e-30))
    print(f"   max forward rel err (torch vs numpy) = {worst_fwd:.2e}")
    print(f"   max VJP     rel err (torch vs numpy) = {worst_vjp:.2e}")
    tolr = 1e-10 if dtype == torch.float64 else 1e-4
    if worst_fwd >= tolr:
        raise AssertionError(f"forward mismatch (torch vs numpy) {worst_fwd:.1e} > {tolr:.0e}")
    if worst_vjp >= tolr:
        raise AssertionError(
            f"VJP mismatch {worst_vjp:.1e} > {tolr:.0e}. The exact active-face VJP needs the "
            f"active set resolved to ~1e-6, which float32 cannot guarantee for O(10) values. "
            f"Re-run in float64 (the default) for exact gradients; float32 is suitable only for "
            f"the forward projection / throughput, not the exact backward.")
    print("   PASSED\n")


def scaling(device, dtype, sizes, reps):
    import torch
    rows = []
    print("=" * 72)
    print(f"(A) single-instance GPU scaling on {device} ({dtype})")
    print("=" * 72)
    for m in sizes:
        v, k, d = make_instance(m, device, dtype)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        # forward (project) and backward (VJP) as standalone primitives
        z = cp_.project_topk_sum_torch(v, k, d)
        cert = cp_.extract_certificate_torch(z, k, d)
        zbar = torch.randn(m, dtype=dtype, device=device)
        fwd = timed(lambda: cp_.project_topk_sum_torch(v, k, d), device, reps)
        bwd = timed(lambda: cp_.vjp_projection_torch(zbar, cert), device, reps)
        # random-direction finite-difference VJP check (correctness at scale)
        du = torch.randn(m, dtype=dtype, device=device)
        eps = 1e-6 if dtype == torch.float64 else 3e-4   # float32 needs a larger FD step
        jvp = cp_.vjp_projection_torch(du, cert)   # symmetric operator => JVP==VJP
        fd = (cp_.project_topk_sum_torch(v + eps * du, k, d)
              - cp_.project_topk_sum_torch(v - eps * du, k, d)) / (2 * eps)
        vjp_err = float((jvp - fd).norm() / (fd.norm() + 1e-30))
        peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else float("nan")
        rows.append(dict(m=m, k=k, fwd_ms=fwd, bwd_ms=bwd, peak_mb=peak_mb, vjp_err=vjp_err))
        print(f"   m={m:>11d}  fwd={fwd:9.3f} ms  bwd={bwd:8.3f} ms  "
              f"peakGPU={peak_mb:9.1f} MB  VJPerr={vjp_err:.1e}")
        del v, z, zbar, du, fd, jvp
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def batched(device, dtype, batch_m, batch_sizes, reps):
    import torch
    rows = []
    print("=" * 72)
    print(f"(B) batched throughput at m={batch_m} on {device}")
    print("=" * 72)
    k = int(round((1 - BETA) * batch_m))
    for B in batch_sizes:
        g = torch.Generator().manual_seed(7)
        V = torch.randn(B, batch_m, generator=g).to(dtype).to(device)
        # make each row violate the constraint
        V[:, :k] += 12.0
        d = float(torch.topk(V[0], k).values.sum().item() - 2.0)
        z = cp_.project_topk_sum_torch(V, k, d)
        cert = cp_.extract_certificate_torch(z, k, d)
        Zbar = torch.randn(B, batch_m, dtype=dtype, device=device)
        fwd = timed(lambda: cp_.project_topk_sum_torch(V, k, d), device, reps)
        bwd = timed(lambda: cp_.vjp_projection_torch(Zbar, cert), device, reps)
        thru = B / ((fwd + bwd) / 1e3)
        rows.append(dict(B=B, m=batch_m, fwd_ms=fwd, bwd_ms=bwd, probs_per_s=thru))
        print(f"   B={B:>6d}  fwd={fwd:9.3f} ms  bwd={bwd:8.3f} ms  "
              f"throughput={thru:12.0f} problems/s")
        del V, z, Zbar
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def cpu_vs_gpu(device, dtype, sizes):
    """numpy(CPU) vs torch(GPU) forward+backward at matched sizes."""
    import torch
    rows = []
    print("=" * 72)
    print("(C) CPU (numpy) vs GPU (torch) forward+backward, matched sizes")
    print("=" * 72)
    for m in sizes:
        v, k, d = make_instance(m, torch.device("cpu"), dtype)
        vnp = v.numpy()
        # CPU numpy
        t0 = time.perf_counter(); znp = cp_.project_topk_sum(vnp, k, d)
        certnp = cp_.extract_certificate(znp, k, d)
        zbar = np.random.default_rng(0).standard_normal(m)
        _ = cp_.vjp_projection(zbar, certnp); cpu_ms = (time.perf_counter() - t0) * 1e3
        # GPU torch
        vg = v.to(device)
        fn_f = lambda: cp_.project_topk_sum_torch(vg, k, d)
        zg = fn_f(); certg = cp_.extract_certificate_torch(zg, k, d)
        zbg = torch.randn(m, dtype=dtype, device=device)
        gpu_ms = timed(lambda: (cp_.vjp_projection_torch(zbg, cp_.extract_certificate_torch(
            cp_.project_topk_sum_torch(vg, k, d), k, d))), device, 5)
        rows.append(dict(m=m, cpu_ms=cpu_ms, gpu_ms=gpu_ms, speedup=cpu_ms / gpu_ms))
        print(f"   m={m:>11d}  CPU={cpu_ms:10.2f} ms  GPU={gpu_ms:10.3f} ms  "
              f"speedup={cpu_ms / gpu_ms:7.1f}x")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[10000, 100000, 1000000, 10000000, 100000000])
    ap.add_argument("--batch-m", type=int, default=10000)
    ap.add_argument("--batch-sizes", type=int, nargs="+",
                    default=[1, 8, 64, 512, 4096])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--tag", default="",
                    help="suffix for output filenames, e.g. --tag _ext, so an extra "
                         "run does NOT overwrite a previous gpu_*.csv / PNG sweep")
    args = ap.parse_args()

    import torch
    device = get_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    print(f"device={device}  dtype={dtype}  "
          f"gpu={torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}")

    self_validate(device, dtype)
    srows = scaling(device, dtype, args.sizes, args.reps)
    brows = batched(device, dtype, args.batch_m, args.batch_sizes, args.reps)
    crows = cpu_vs_gpu(device, dtype, [s for s in args.sizes if s <= 10_000_000])

    import pandas as pd
    tag = args.tag
    ds = pd.DataFrame(srows); ds.to_csv(os.path.join(RESULTS, f"gpu_scaling{tag}.csv"), index=False)
    db = pd.DataFrame(brows); db.to_csv(os.path.join(RESULTS, f"gpu_batched{tag}.csv"), index=False)
    dc = pd.DataFrame(crows); dc.to_csv(os.path.join(RESULTS, f"gpu_cpu_vs_gpu{tag}.csv"), index=False)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.loglog(ds.m, ds.fwd_ms, "s--", label="forward (sort+sweep)")
    ax.loglog(ds.m, ds.bwd_ms, "o-", label="backward (VJP)")
    ref = ds.bwd_ms.iloc[0] * ds.m / ds.m.iloc[0]
    ax.loglog(ds.m, ref, ":", color="gray", label="O(m) reference")
    ax.set_xlabel("scenarios m"); ax.set_ylabel("time (ms)")
    ax.set_title(f"CVaR projection on {('GPU' if device.type=='cuda' else 'CPU')}: time vs m")
    ax.legend(); ax.grid(True, which="both", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, f"gpu_scaling_time{tag}.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.loglog(ds.m, ds.peak_mb, "o-")
    ax.set_xlabel("scenarios m"); ax.set_ylabel("peak GPU memory (MB)")
    ax.set_title("CVaR projection: peak GPU memory vs m")
    ax.grid(True, which="both", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, f"gpu_scaling_mem{tag}.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.loglog(db.B, db.probs_per_s, "o-")
    ax.set_xlabel("batch size B"); ax.set_ylabel("problems / second")
    ax.set_title(f"Batched throughput at m={args.batch_m}")
    ax.grid(True, which="both", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, f"gpu_throughput{tag}.png"), dpi=150)

    print("\nwrote results/gpu_scaling.csv, results/gpu_batched.csv, "
          "results/gpu_cpu_vs_gpu.csv and 3 PNGs")
    print(f"max VJP finite-difference error across sizes: {ds.vjp_err.max():.1e}")


if __name__ == "__main__":
    main()
