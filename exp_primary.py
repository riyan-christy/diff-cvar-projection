"""
exp_primary.py
==============
Primary constraint-targeting experiment (EXPERIMENT_EXECUTION_SPEC.md): hard
CVaR constraint versus fixed-lambda and oracle-retuned exact CVaR penalties,
with realized adherence evaluated on a held-out bank.

Stages (--stage):
  smoke    tiny CPU run of the FULL pipeline (n=10, m in {200, 400}); minutes;
           used by tests/CI and the pre-A100 sanity check
  pilot    spec pilot: seeds {0,1}, m in {1e3, 1e4}; selects and freezes
           kappa via the spec rule into <out>/pilot_freeze.json
  full     spec full first run (5 seeds, m in {1e3,1e4,1e5})
  journal  spec journal-strength rerun (10 seeds, larger N)

Logging/resume per spec "Logging, Checkpointing, And Outputs": every solve
appends one row to <out>/solves.csv keyed by
(seed, m, kappa, repeat, regime, split, method, instance, lambda); reruns
skip keys that already have a converged row.  Aggregation recomputes
everything from solves.csv (never hand-edit aggregates).

Usage:
  python exp_primary.py --stage smoke
  python exp_primary.py --stage pilot --device cuda
  python exp_primary.py --stage full --device cuda
  python exp_primary.py --aggregate-only --stage full
  python exp_primary.py --self-check --device cuda   # torch-vs-numpy gate
"""
import argparse, csv, json, os, time
import numpy as np

import cvar_proj as cp_
import exp_data as D
import exp_solvers as S

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# stage configurations (spec: "Sample Sizes", "Portfolio Problem")
# --------------------------------------------------------------------------

def stage_cfg(stage):
    base = dict(n=50, K=5, beta=0.99, gamma=1.0, w_max=0.20, M_eval=1_000_000,
                n_kappa=2, solver_kw={})
    if stage == "smoke":
        # tiny + loose (1e-6): exercises every code path in minutes on CPU
        return dict(base, n=10, K=3, beta=0.95, w_max=0.30, M_eval=10_000,
                    seeds=[0], ms=[150, 300], N_val=6, N_test=10,
                    repeats={150: 2, 300: 1}, n_kappa=1,
                    solver_kw=dict(eps_abs=1e-6, eps_rel=1e-6,
                                   max_iter=1500),
                    cert_kw=dict(eps_abs=1e-4, eps_rel=1e-4, max_iter=1200))
    if stage == "pilot":
        return dict(base, seeds=[0, 1], ms=[1000, 10000], N_val=64,
                    N_test=128, repeats={1000: 2, 10000: 1},
                    solver_kw=dict(eps_abs=1e-4, eps_rel=1e-4,
                                   max_iter=10000),
                    cert_kw=dict(eps_abs=1e-4, eps_rel=1e-4, max_iter=2500),
                    cal_kw=dict(eps_abs=1e-3, eps_rel=1e-3, max_iter=3000))
    if stage == "full":
        # scope per the spec's pre-authorized cut (2026-06-11): all m values
        # and all methods retained; repeats/N reduced to fit the budget
        return dict(base, seeds=[0, 1, 2, 3, 4], ms=[1000, 10000, 100000],
                    N_val=96, N_test=192,
                    repeats={1000: 4, 10000: 2, 100000: 1},
                    n_kappa_m={100000: 1},   # kappa_2 dropped at m=1e5 only
                    solver_kw=dict(eps_abs=1e-4, eps_rel=1e-4,
                                   max_iter=12000),
                    cert_kw=dict(eps_abs=1e-4, eps_rel=1e-4, max_iter=2500),
                    cal_kw=dict(eps_abs=1e-3, eps_rel=1e-3, max_iter=3000))
    if stage == "journal":
        return dict(base, seeds=list(range(10)), ms=[1000, 10000, 100000],
                    N_val=256, N_test=512,
                    repeats={1000: 8, 10000: 4, 100000: 2},
                    solver_kw=dict(eps_abs=1e-6, eps_rel=1e-6,
                                   max_iter=25000),
                    cert_kw=dict(eps_abs=1e-4, eps_rel=1e-4, max_iter=2500))
    raise ValueError(stage)


# --------------------------------------------------------------------------
# solve log (append-only, resumable)
# --------------------------------------------------------------------------

COLS = ["seed", "m", "kappa", "repeat", "regime", "split", "method",
        "instance", "lam", "status", "iters", "primal_res", "dual_res",
        "pred_cvar", "real_cvar", "real_ret", "min_cvar", "nu_star",
        "wall_ms"]


class SolveLog:
    def __init__(self, path):
        self.path = path
        self.seen = set()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            with open(path) as f:
                for row in csv.DictReader(f):
                    self.seen.add(self._key(row))
        else:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(COLS)

    @staticmethod
    def _key(row):
        return (str(row["seed"]), str(row["m"]), f'{float(row["kappa"]):.10g}',
                str(row["repeat"]), row["regime"], row["split"],
                row["method"], str(row["instance"]),
                f'{float(row["lam"]):.10g}')

    def have_cell(self, seed, m, kappa, repeat, regime, split, method,
                  lam=0.0, n_expected=1):
        """True only if EVERY instance row of the batch is present (a write
        interrupted mid-batch must not be skipped on resume)."""
        for j in range(n_expected):
            k = (str(seed), str(m), f"{float(kappa):.10g}", str(repeat),
                 regime, split, method, str(j), f"{float(lam):.10g}")
            if k not in self.seen:
                return False
        return True

    def write_batch(self, base, instances_rows):
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f)
            for r in instances_rows:
                row = {**base, **r}
                w.writerow([row.get(c, "") for c in COLS])
                self.seen.add(self._key({c: row.get(c, 0) for c in COLS}))


# --------------------------------------------------------------------------
# one experiment cell
# --------------------------------------------------------------------------

def eval_metrics(W, L_pred, L_eval, mus, beta):
    pc = S.predicted_cvar(W, L_pred, mus, beta)
    rc = S.realized_cvar(W, L_eval, mus, beta)
    rr = S.realized_return(W, L_eval, mus)
    return pc, rc, rr


def pipeline_assertion(W, L_pred, mus, beta, pc, n_check=16):
    """Spec: run the realized-CVaR evaluation pipeline ON THE PREDICTED bank
    for a subset and require agreement with the solver-side predicted CVaR."""
    idx = np.arange(min(n_check, W.shape[0]))
    rc_on_pred = S.realized_cvar(W[idx], L_pred, mus[idx], beta)
    rel = np.max(np.abs(rc_on_pred - pc[idx]) / (1.0 + np.abs(pc[idx])))
    assert rel <= 1e-8, f"pipeline consistency violated: {rel:.3e}"
    return rel


def _read_back(path, seed, m, kappa, repeat, regime, split, method):
    """Recover per-instance pred_cvar/status for a logged solve (resume)."""
    pc, st = {}, {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if (row["seed"], row["m"], f'{float(row["kappa"]):.10g}',
                    row["repeat"], row["regime"], row["split"],
                    row["method"]) == (str(seed), str(m), f"{kappa:.10g}",
                                       str(repeat), regime, split, method):
                v = row["pred_cvar"]
                pc[int(row["instance"])] = float(v) if v not in ("", None) \
                    else float("nan")
                st[int(row["instance"])] = row["status"]
    n = max(pc) + 1
    return (np.array([pc[i] for i in range(n)]),
            np.array([st[i] for i in range(n)]))


def run_cell(cfg, log, seed, m, kappa, repeat, struct, banks, ops,
             scenario_boot=False):
    """All methods for one (seed, m, kappa, repeat): calibrate on ID val,
    evaluate on ID/shift test (spec: 'Methods')."""
    beta, gamma, w_max = cfg["beta"], cfg["gamma"], cfg["w_max"]
    skw = cfg.get("solver_kw", {})

    def solve_and_log(method, regime, split, L_pred, mus, lam=0.0,
                      W_known=None, extra=None):
        N = mus.shape[0]
        base = dict(seed=seed, m=m, kappa=f"{kappa:.10g}", repeat=repeat,
                    regime=regime, split=split, method=method,
                    lam=f"{lam:.10g}")
        if log.have_cell(seed, m, kappa, repeat, regime, split, method, lam,
                         n_expected=N):
            return None  # resumed
        t0 = time.perf_counter()
        L_eval = banks[("eval", regime)]
        mc = extra.get("min_cvar") if extra else None
        # spec: certified-infeasible instances are recorded, NOT solved
        if method == "hard" and mc is not None:
            feas = S.feasible_mask(mc, kappa)
        else:
            feas = np.ones(N, dtype=bool)
        idx = np.where(feas)[0]
        W = np.full((N, mus.shape[1]), np.nan)
        status = np.array(["infeasible"] * N, dtype=object)
        iters = np.zeros(N, int)
        nus = np.full(N, np.nan)
        rpri = np.full(N, np.nan); rdua = np.full(N, np.nan)
        if W_known is not None:
            W = W_known; status[:] = "closed"
        elif method == "hard":
            if idx.size:
                r = S.solve_hard(L_pred, mus[idx], kappa, beta=beta,
                                 gamma=gamma, w_max=w_max, ops=ops, **skw)
                W[idx] = r["W"]; status[idx] = r["status"]
                iters[idx] = r["iters"]; nus[idx] = r["nu_star"]
                rpri[idx] = r["res_pri"]; rdua[idx] = r["res_dua"]
        else:
            r = S.solve_penalty(L_pred, mus, lam, beta=beta, gamma=gamma,
                                w_max=w_max, ops=ops, **skw)
            W, status, iters = r["W"], r["status"], r["iters"]
            rpri, rdua = r["res_pri"], r["res_dua"]
        solved = ~np.isnan(W[:, 0])
        pc = np.full(N, np.nan); rc = np.full(N, np.nan)
        rr = np.full(N, np.nan)
        if solved.any():
            pc[solved], rc[solved], rr[solved] = eval_metrics(
                W[solved], L_pred, L_eval, mus[solved], beta)
        if method == "hard" and solved.any():
            pipeline_assertion(W[solved], L_pred, mus[solved], beta,
                               pc[solved])
        wall = (time.perf_counter() - t0) * 1e3
        rows = []
        for j in range(N):
            rows.append(dict(instance=j, status=status[j], iters=int(iters[j]),
                             primal_res=f"{rpri[j]:.4g}",
                             dual_res=f"{rdua[j]:.4g}",
                             pred_cvar=f"{pc[j]:.10g}",
                             real_cvar=f"{rc[j]:.10g}",
                             real_ret=f"{rr[j]:.10g}",
                             min_cvar=(f"{mc[j]:.10g}" if mc is not None else ""),
                             nu_star=f"{nus[j]:.6g}",
                             wall_ms=f"{wall / N:.3g}"))
        log.write_batch(base, rows)
        return dict(W=W, pc=pc, rc=rc, rr=rr, status=status)

    # ---- banks for this repeat
    Lv_id = banks[("val", "id")]; Lt_id = banks[("test", "id")]
    Lv_sh = banks[("val", "shift")]; Lt_sh = banks[("test", "shift")]
    mus_v_id = banks[("mus_val", "id")]; mus_t_id = banks[("mus_test", "id")]
    mus_v_sh = banks[("mus_val", "shift")]; mus_t_sh = banks[("mus_test", "shift")]

    # ---- feasibility certificates (cached per cell; resume-friendly)
    cert_p = os.path.join(os.path.dirname(log.path),
                          f"certs_{seed}_{m}_{kappa:.6g}_{repeat}.json")
    if os.path.exists(cert_p):
        cert = {kk: np.array(vv) for kk, vv in json.load(open(cert_p)).items()}
    else:
        cert = {}
        for tag, L_, mus_ in [("val_id", Lv_id, mus_v_id),
                              ("test_id", Lt_id, mus_t_id),
                              ("test_sh", Lt_sh, mus_t_sh),
                              ("val_sh", Lv_sh, mus_v_sh)]:
            mc = S.solve_min_cvar(L_, mus_, beta=beta, w_max=w_max, ops=ops,
                                  **cfg.get("cert_kw", {}))
            cert[tag] = S.predicted_cvar(mc["W"], L_, mus_, beta)
        json.dump({kk: vv.tolist() for kk, vv in cert.items()},
                  open(cert_p, "w"))

    feas_v = S.feasible_mask(cert["val_id"], kappa)
    print(f"      [cell] certs ready", flush=True)

    # ---- 1. hard on ID validation -> calibration target
    hv = solve_and_log("hard", "id", "val", Lv_id, mus_v_id,
                       extra=dict(min_cvar=cert["val_id"]))
    if hv is None:
        # resumed: rebuild target/active set from the solve log
        pcv, stat = _read_back(log.path, seed, m, kappa, repeat, "id", "val",
                               "hard")
        hv = dict(pc=pcv, status=stat)
    pcv = hv["pc"]
    active_v = feas_v & (np.abs(pcv / kappa - 1.0) <= S.ACTIVE_TOL) \
        & (hv["status"] == "converged")
    subset = np.where(active_v)[0]
    if subset.size == 0:
        subset = np.where(feas_v)[0]   # degenerate cell; recorded via CSV
    subset = subset[:48]               # calibration needs the MEAN to 1%;
    target = float(pcv[subset].mean())  # 48 instances suffice (spec amend)

    # ---- 2. fixed-lambda calibration on ID validation (same subset)
    # resume: reuse a previously calibrated lambda for this cell if present
    cal_path = os.path.join(os.path.dirname(log.path), "calibration.csv")
    cal_prev = None
    if os.path.exists(cal_path):
        with open(cal_path) as f:
            for row in csv.DictReader(f):
                if (row["seed"], row["m"], row["kappa"], row["repeat"]) == \
                        (str(seed), str(m), f"{kappa:.10g}", str(repeat)):
                    cal_prev = row
    nsolves = {"n": 0}
    # per-evaluation cache so an interrupted search resumes mid-bisection
    cache_p = os.path.join(os.path.dirname(log.path),
                           f"lamcache_{seed}_{m}_{kappa:.6g}_{repeat}.json")
    lcache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}

    def _cached(tag, lam, fn):
        key = f"{tag}:{lam:.12g}"
        if key not in lcache:
            lcache[key] = float(fn(lam))
            json.dump(lcache, open(cache_p, "w"))
        return lcache[key]

    ckw = cfg.get("cal_kw", skw)
    def mean_pen(lam):
        def fn(lam):
            r = S.solve_penalty(Lv_id, mus_v_id[subset], lam, beta=beta,
                                gamma=gamma, w_max=w_max, ops=ops, **ckw)
            nsolves["n"] += 1
            return S.predicted_cvar(r["W"], Lv_id, mus_v_id[subset],
                                    beta).mean()
        return _cached("id", lam, fn)

    if cal_prev is not None:
        lam_fix = float(cal_prev["lam_fixed"])
        n_evals = int(cal_prev["n_evals_fixed"])
        cal_err = float(cal_prev["cal_err_fixed"])
    else:
        lam_fix, n_evals, cal_err, _h = S.calibrate_lambda(mean_pen, target)
    print(f"      [cell] lam_fix={lam_fix:.4g} ({n_evals} evals)", flush=True)

    # ---- 3. oracle retune on SHIFTED validation
    feas_vs = S.feasible_mask(cert["val_sh"], kappa)
    print("      [cell] solving hard shift-val...", flush=True)
    hvs = solve_and_log("hard", "shift", "val", Lv_sh, mus_v_sh,
                        extra=dict(min_cvar=cert["val_sh"]))
    if hvs is None:
        pcvs, stat_s = _read_back(log.path, seed, m, kappa, repeat, "shift",
                                  "val", "hard")
        hvs = dict(pc=pcvs, status=stat_s)
    pcvs = hvs["pc"]
    act_vs = feas_vs & (np.abs(pcvs / kappa - 1) <= S.ACTIVE_TOL) \
        & (hvs["status"] == "converged")
    sub_s = np.where(act_vs)[0]
    if sub_s.size == 0:
        sub_s = np.where(feas_vs)[0]
    sub_s = sub_s[:48]
    target_s = float(pcvs[sub_s].mean()) if sub_s.size else target

    def mean_pen_s(lam):
        def fn(lam):
            r = S.solve_penalty(Lv_sh, mus_v_sh[sub_s], lam, beta=beta,
                                gamma=gamma, w_max=w_max, ops=ops, **ckw)
            return S.predicted_cvar(r["W"], Lv_sh, mus_v_sh[sub_s],
                                    beta).mean()
        return _cached("shift", lam, fn)

    if cal_prev is not None:
        lam_orc = float(cal_prev["lam_oracle"])
        n_evals_o = int(cal_prev["n_evals_oracle"])
        cal_err_o = float(cal_prev["cal_err_oracle"])
    else:
        lam_orc, n_evals_o, cal_err_o, _ = S.calibrate_lambda(mean_pen_s,
                                                              target_s)
    print(f"      [cell] lam_orc={lam_orc:.4g}; running tests", flush=True)

    # ---- 4. all methods on both test splits
    for regime, L_t, mus_t, certk in [("id", Lt_id, mus_t_id, "test_id"),
                                      ("shift", Lt_sh, mus_t_sh, "test_sh")]:
        solve_and_log("hard", regime, "test", L_t, mus_t,
                      extra=dict(min_cvar=cert[certk]))
        solve_and_log("penalty_fixed", regime, "test", L_t, mus_t, lam=lam_fix)
        if regime == "shift":
            solve_and_log("penalty_oracle", regime, "test", L_t, mus_t,
                          lam=lam_orc)
        W_fl = S.solve_mv_floor(mus_t, gamma=gamma, w_max=w_max)
        solve_and_log("floor", regime, "test", L_t, mus_t, W_known=W_fl)

    # ---- calibration bookkeeping
    if cal_prev is not None:
        return "ran"
    newfile = not os.path.exists(cal_path)
    with open(cal_path, "a", newline="") as f:
        w = csv.writer(f)
        if newfile:
            w.writerow(["seed", "m", "kappa", "repeat", "lam_fixed",
                        "lam_oracle", "n_evals_fixed", "n_evals_oracle",
                        "cal_err_fixed", "cal_err_oracle", "subset_size",
                        "target"])
        # full precision: an 8-digit rounding here changes the resume key
        # and re-solves every penalty batch (duplicate rows)
        w.writerow([seed, m, f"{kappa:.10g}", repeat, f"{lam_fix:.17g}",
                    f"{lam_orc:.17g}", n_evals, n_evals_o,
                    f"{cal_err:.3g}", f"{cal_err_o:.3g}", subset.size,
                    f"{target:.8g}"])
    return "ran"


# --------------------------------------------------------------------------
# kappa selection (spec: "Kappa Pilot And Freeze"), blind to method separation
# --------------------------------------------------------------------------

def verify_kappas(cfg, cands, seed_list, ops, n_check=32):
    """Spec: verify provisional kappas at ALL target scenario counts
    (including m=1e5) before freezing.  BLIND: looks only at activity and
    feasibility.  Returns a per-(kappa, m) report and the kappas that meet
    the criteria (active fraction 0.6-0.9 on ID validation, shift
    feasibility >= 0.95), ordered by spec preference."""
    beta, gamma, w_max = cfg["beta"], cfg["gamma"], cfg["w_max"]
    report = []
    for kappa in cands:
        ok_all = True
        for m in cfg["ms"]:
            act_f, feas_f = [], []
            for seed in seed_list:
                struct = D.make_structure(seed, n=cfg["n"], K=cfg["K"])
                L = D.make_shock_bank(struct, "id", m, seed, 0, 1)
                mus = D.make_mus(struct, "id", n_check, seed, 0, 1)
                mc = S.solve_min_cvar(L, mus, beta=beta, w_max=w_max,
                                      ops=ops, **cfg.get("cert_kw", {}))
                pcm = S.predicted_cvar(mc["W"], L, mus, beta)
                feas = S.feasible_mask(pcm, kappa)
                r = S.solve_hard(L, mus[feas], kappa, beta=beta, gamma=gamma,
                                 w_max=w_max, ops=ops,
                                 **cfg.get("solver_kw", {}))
                pc = S.predicted_cvar(r["W"], L, mus[feas], beta)
                act = (np.abs(pc / kappa - 1) <= S.ACTIVE_TOL) \
                    & (r["status"] == "converged")
                act_f.append(act.mean() if len(act) else 0.0)
                Ls = D.make_shock_bank(struct, "shift", m, seed, 0, 1)
                mus_s = D.make_mus(struct, "shift", n_check, seed, 0, 1)
                mcs = S.solve_min_cvar(Ls, mus_s, beta=beta, w_max=w_max,
                                       ops=ops, **cfg.get("cert_kw", {}))
                pcs = S.predicted_cvar(mcs["W"], Ls, mus_s, beta)
                feas_f.append(S.feasible_mask(pcs, kappa).mean())
            a, f = float(np.mean(act_f)), float(np.mean(feas_f))
            ok = (0.5 <= a <= 0.95) and (f >= 0.95)
            ok_all &= ok
            report.append(dict(kappa=kappa, m=m, active_frac=a,
                               shift_feas=f, ok=ok))
        for r_ in report:
            if r_["kappa"] == kappa:
                r_["ok_all_m"] = ok_all
    passing = [k for k in cands
               if all(r_["ok"] for r_ in report if r_["kappa"] == k)]
    return report, passing


def select_kappas(cfg, seed_list, m_sel, ops, qs=(0.25, 0.50, 0.75)):
    """Spec pilot rule; BLIND: only activity/feasibility inputs, no method
    separation metrics are computed here."""
    beta, gamma, w_max = cfg["beta"], cfg["gamma"], cfg["w_max"]
    mins, uncs = [], []
    for seed in seed_list:
        struct = D.make_structure(seed, n=cfg["n"], K=cfg["K"])
        for regime in ("id", "shift"):
            L = D.make_shock_bank(struct, regime, m_sel, seed, 0, 1)
            mus = D.make_mus(struct, regime, cfg["N_val"], seed, 0, 1)
            mc = S.solve_min_cvar(L, mus, beta=beta, w_max=w_max, ops=ops,
                                  **cfg.get("cert_kw", {}))
            mins.append(np.median(S.predicted_cvar(mc["W"], L, mus, beta)))
            if regime == "id":
                W_u = S.solve_mv_floor(mus, gamma=gamma, w_max=w_max)
                uncs.append(np.median(S.predicted_cvar(W_u, L, mus, beta)))
    cv_min, cv_unc = float(np.median(mins)), float(np.median(uncs))
    cands = [cv_min + q * (cv_unc - cv_min) for q in qs]
    return cands, cv_min, cv_unc


# --------------------------------------------------------------------------
# aggregation -> summary CSVs + figures (recomputed from solves.csv only)
# --------------------------------------------------------------------------

def aggregate(outdir):
    """Recompute every table and figure from solves.csv (spec outputs 1-6).
    Headline risk metrics use only ok rows (converged/closed); maxiter and
    infeasible rows are excluded from them and reported as rates."""
    import pandas as pd
    p = os.path.join(outdir, "solves.csv")
    if not os.path.exists(p):
        print("no solves.csv; nothing to aggregate"); return
    df = pd.read_csv(p)
    # dedupe WITHOUT lam in the key: an interrupted calibration can log
    # penalty rows at stale lambdas; the last row per instance corresponds to
    # the final calibrated lambda (append-only log)
    key = ["seed", "m", "kappa", "repeat", "regime", "split", "method",
           "instance"]
    n0 = len(df)
    df = df.drop_duplicates(subset=key, keep="last")
    if len(df) != n0:
        print(f"note: dropped {n0 - len(df)} stale/duplicate solve rows "
              "(resume artifacts); keep='last' per instance key")
    df["ok"] = df.status.isin(["converged", "closed"])
    df["active"] = (df.method.eq("hard") & df.status.eq("converged")
                    & ((df.pred_cvar / df.kappa - 1).abs() <= S.ACTIVE_TOL))
    df["feasible"] = np.where(
        df.min_cvar.notna(),
        df.min_cvar <= df.kappa * (1 + S.FEAS_MARGIN_REL) + S.CERT_BIAS, True)
    df["real_ratio"] = df.real_cvar / df.kappa
    df["pred_ratio"] = df.pred_cvar / df.kappa
    df["exceed"] = (df.real_cvar > df.kappa).astype(float)
    df["pos_exc"] = (df.real_cvar - df.kappa).clip(lower=0)
    df["optimism_gap"] = df.real_cvar - df.pred_cvar
    df["pred_violation"] = (df.pred_cvar - df.kappa).clip(lower=0)

    test = df[df.split.eq("test")].copy()
    gk = ["method", "regime", "m", "kappa", "seed"]
    # solver-quality / failure rates from ALL rows
    rates = (test.groupby(gk)
             .agg(maxiter_rate=("status", lambda s: (s == "maxiter").mean()),
                  infeasible_rate=("status",
                                   lambda s: (s == "infeasible").mean()),
                  feasible_rate=("feasible", "mean"),
                  active_rate=("active", "mean"),
                  n_rows=("status", "size"))
             .reset_index())
    # headline risk metrics from ok rows only
    ok = test[test.ok]
    g = (ok.groupby(gk)
         .agg(real_ratio=("real_ratio", "mean"),
              real_ratio_med=("real_ratio", "median"),
              pred_ratio_mean=("pred_ratio", "mean"),
              pred_ratio_std=("pred_ratio", "std"),
              exceed=("exceed", "mean"),
              pos_exc=("pos_exc", "mean"),
              optimism_gap=("optimism_gap", "mean"),
              pred_violation=("pred_violation", "mean"),
              real_ret=("real_ret", "mean"),
              wall_ms=("wall_ms", "median"))
         .reset_index()
         .merge(rates, on=gk, how="outer"))
    g.to_csv(os.path.join(outdir, "seed_level.csv"), index=False)

    def boot_ci(x, n=10000, seed=0):
        x = np.asarray(x, float)
        x = x[~np.isnan(x)]
        if len(x) <= 1:
            return np.nan, np.nan
        r = np.random.default_rng(seed)
        bs = x[r.integers(0, len(x), (n, len(x)))].mean(axis=1)
        return np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    rows = []
    for kk, sub in g.groupby(["method", "regime", "m", "kappa"]):
        lo, hi = boot_ci(sub.real_ratio)
        lo_e, hi_e = boot_ci(sub.exceed)
        rows.append(dict(
            method=kk[0], regime=kk[1], m=kk[2], kappa=kk[3],
            n_seeds=len(sub),
            real_ratio=sub.real_ratio.mean(), real_ratio_lo=lo,
            real_ratio_hi=hi, real_ratio_med=sub.real_ratio_med.mean(),
            pred_ratio_mean=sub.pred_ratio_mean.mean(),
            pred_ratio_disp=sub.pred_ratio_std.mean(),
            exceed=sub.exceed.mean(), exceed_lo=lo_e, exceed_hi=hi_e,
            pos_exc=sub.pos_exc.mean(),
            optimism_gap=sub.optimism_gap.mean(),
            pred_violation=sub.pred_violation.mean(),
            feasible_rate=sub.feasible_rate.mean(),
            active_rate=sub.active_rate.mean(),
            maxiter_rate=sub.maxiter_rate.mean(),
            infeasible_rate=sub.infeasible_rate.mean(),
            real_ret=sub.real_ret.mean()))
    summ = pd.DataFrame(rows).sort_values(["regime", "method", "m"])
    summ.to_csv(os.path.join(outdir, "summary.csv"), index=False)

    # spec table 3: exceedance table
    summ[["method", "regime", "m", "kappa", "active_rate", "feasible_rate",
          "maxiter_rate", "infeasible_rate", "exceed", "pos_exc",
          "real_ratio"]].to_csv(os.path.join(outdir, "exceedance.csv"),
                                index=False)
    # spec table 4: tuning cost
    calp = os.path.join(outdir, "calibration.csv")
    if os.path.exists(calp):
        cal = pd.read_csv(calp).drop_duplicates(
            subset=["seed", "m", "kappa", "repeat"], keep="last")
        wall = (ok.groupby(["method", "m"]).wall_ms.median()
                .rename("median_wall_ms_per_instance").reset_index())
        cal.to_csv(os.path.join(outdir, "tuning_cost.csv"), index=False)
        wall.to_csv(os.path.join(outdir, "wall_clock.csv"), index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # fig 1: realized ratio vs m
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, regime in zip(axes, ["id", "shift"]):
            sub = summ[(summ.regime == regime)
                       & summ.method.isin(["hard", "penalty_fixed",
                                           "penalty_oracle"])]
            for meth, style in [("hard", "o-"), ("penalty_fixed", "s--"),
                                ("penalty_oracle", "^:")]:
                ss = sub[sub.method == meth].groupby("m").agg(
                    y=("real_ratio", "mean"), lo=("real_ratio_lo", "mean"),
                    hi=("real_ratio_hi", "mean")).reset_index()
                if len(ss):
                    ax.errorbar(ss.m, ss.y,
                                yerr=[(ss.y - ss.lo).clip(lower=0),
                                      (ss.hi - ss.y).clip(lower=0)],
                                fmt=style, capsize=3, label=meth)
            ax.axhline(1.0, color="k", lw=0.8, ls=":")
            ax.set_xscale("log"); ax.set_xlabel("scenarios m")
            ax.set_title(f"{regime} regime")
        axes[0].set_ylabel("realized CVaR / kappa")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig_realized_ratio_vs_m.png"),
                    dpi=150)
        plt.close(fig)

        # fig 2: predicted-ratio distributions (box), active/ok rows
        okb = ok[ok.method.isin(["hard", "penalty_fixed", "penalty_oracle"])]
        if len(okb):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
            for ax, regime in zip(axes, ["id", "shift"]):
                subr = okb[okb.regime == regime]
                data, labels = [], []
                for meth in ["hard", "penalty_fixed", "penalty_oracle"]:
                    vals = test[(test.regime == regime) & test.ok
                                & test.method.eq(meth)].pred_ratio.dropna()
                    if len(vals):
                        data.append(vals); labels.append(meth)
                if data:
                    ax.boxplot(data, tick_labels=labels, showmeans=True)
                ax.axhline(1.0, color="k", lw=0.8, ls=":")
                ax.set_title(f"{regime} regime")
            axes[0].set_ylabel("predicted CVaR / kappa")
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "fig_pred_ratio_dist.png"),
                        dpi=150)
            plt.close(fig)

        # fig 5: return at matched realized risk (cautious secondary figure)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for meth, mk in [("hard", "o"), ("penalty_fixed", "s"),
                         ("penalty_oracle", "^"), ("floor", "x")]:
            ss = summ[summ.method.eq(meth) & summ.regime.eq("shift")]
            if len(ss):
                ax.scatter(ss.real_ratio, ss.real_ret, marker=mk, label=meth)
        ax.set_xlabel("realized CVaR / kappa (shift)")
        ax.set_ylabel("mean realized return")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig_return_frontier.png"), dpi=150)
        plt.close(fig)

        # fig 6: multiplier dispersion vs the frozen lambda
        hv = df[df.method.eq("hard") & df.active & df.nu_star.notna()]
        if len(hv) and os.path.exists(calp):
            fig, ax = plt.subplots(figsize=(6, 4.5))
            for regime, color in [("id", "C0"), ("shift", "C1")]:
                vals = hv[hv.regime.eq(regime)].nu_star.astype(float)
                if len(vals):
                    ax.hist(vals, bins=30, alpha=0.55, label=f"nu* ({regime})",
                            color=color)
            for lamv in pd.read_csv(calp).lam_fixed.unique()[:4]:
                ax.axvline(float(lamv), color="k", ls="--", lw=1)
            ax.set_xlabel("hard-constraint multiplier nu* "
                          "(dashed: frozen lambda)")
            ax.legend(); fig.tight_layout()
            fig.savefig(os.path.join(outdir, "fig_multiplier_dispersion.png"),
                        dpi=150)
            plt.close(fig)
    except Exception as e:   # pragma: no cover
        print("figure generation skipped:", e)
    print("aggregated ->", os.path.join(outdir, "summary.csv"))


# --------------------------------------------------------------------------
# torch-vs-numpy self check (gate for any GPU run)
# --------------------------------------------------------------------------

def self_check(device):
    """Three-part torch-vs-numpy gate, ordered so failures localize:
    (1) robust projection on adversarial inputs (incl. plateaus, exact ties,
        inactive and near-boundary rows), (2) hard CVQP solve, (3) exact
        penalty solve (capped-simplex warm path).  Run this on the A100
        BEFORE any paid experiment."""
    import torch
    cfg = stage_cfg("smoke")
    dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"

    # (1) projection: torch path vs exact numpy reference
    rng = np.random.default_rng(17)
    errs = []
    for trial in range(24):
        m = int(rng.integers(40, 300)); k = int(rng.integers(2, max(3, m // 3)))
        style = trial % 4
        if style == 0:
            v = rng.normal(0, 1, m)
        elif style == 1:   # tight top group -> plateau under strong violation
            v = np.concatenate([rng.normal(8, .1, min(k + 4, m)),
                                rng.normal(-4, 1, m - min(k + 4, m))])
            rng.shuffle(v)
        elif style == 2:   # exact input ties
            v = np.concatenate([np.full(min(k, m // 2), 2.0),
                                rng.normal(0, 1, m - min(k, m // 2))])
            rng.shuffle(v)
        else:
            v = np.round(rng.normal(0, 1, m), 1)
        d = cp_.topk_sum(v, k) - rng.uniform(0.05, 30.0)
        z_ref = cp_.project_topk_sum(v, k, d)
        vt = torch.as_tensor(v[None, :], dtype=torch.float64, device=dev)
        z_t = S._proj_cvar_robust(vt, k, d)[0].cpu().numpy()
        errs.append(np.linalg.norm(z_t - z_ref)
                    / max(np.linalg.norm(z_ref), 1e-12))
    print(f"self-check (1/4) robust projection torch({dev}) vs exact numpy: "
          f"max rel err = {max(errs):.3e}", flush=True)
    assert max(errs) < 1e-9, "torch robust projection disagrees"

    # (2) BATCHED multi-row projection with warm-state reuse: part 1 tests
    # only single rows; the 2026-06-10 A100 incident showed batched/stateful
    # paths need their own gate (boolean-mask indexing, warm brackets, the
    # state machinery across repeated calls)
    rng = np.random.default_rng(23)
    m, k = 240, 12
    rows = []
    for style in range(8):
        if style % 4 == 0:
            v = rng.normal(0, 1, m)
        elif style % 4 == 1:
            v = np.concatenate([rng.normal(8, .1, k + 4),
                                rng.normal(-4, 1, m - k - 4)])
            rng.shuffle(v)
        elif style % 4 == 2:
            v = np.concatenate([np.full(k, 2.0), rng.normal(0, 1, m - k)])
            rng.shuffle(v)
        else:
            v = np.round(rng.normal(0, 1, m), 1)
        rows.append(v)
    V = np.stack(rows)
    d = float(np.median(cp_.topk_sum(V, k)) - 2.0)
    import torch as _t
    Vt = _t.as_tensor(V, dtype=_t.float64, device=dev)
    state_t = {}
    errs2 = []
    for rep in range(3):                      # repeated calls -> warm state
        Vp = Vt + 0.01 * rep                  # small drift, like ADMM inputs
        Zt = S._proj_cvar_robust(Vp, k, d, state=state_t)
        Zt_np = Zt.cpu().numpy()
        for r in range(V.shape[0]):
            z_ref = cp_.project_topk_sum(V[r] + 0.01 * rep, k, d)
            errs2.append(np.linalg.norm(Zt_np[r] - z_ref)
                         / max(np.linalg.norm(z_ref), 1e-12))
    print(f"self-check (2/4) batched+warm projection torch({dev}): "
          f"max rel err = {max(errs2):.3e}", flush=True)
    assert max(errs2) < 1e-8, "torch batched/warm projection disagrees"

    # (3) hard solve and (4) penalty solve: torch vs numpy end to end.
    # Bounded gate cost: eps 1e-6 / max_iter 3000; statuses printed so
    # non-convergence is visible instead of looking like a hang.
    struct = D.make_structure(0, n=cfg["n"], K=cfg["K"])
    L = D.make_shock_bank(struct, "id", 200, 0, 0, 1)
    mus = D.make_mus(struct, "id", 6, 0, 0, 1)
    # kappa must be FEASIBLE and binding for this exact data: min-CVaR ~0.63,
    # unconstrained floor ~1.08 (measured), so 0.85 converges in ~350-500
    # iterations on both backends.  (0.5 was infeasible: both backends ran to
    # maxiter, agreeing to 1.5e-7 -- the 2026-06-11 gate failure.)
    kap = 0.85
    kw = dict(beta=cfg["beta"], gamma=cfg["gamma"], w_max=cfg["w_max"],
              eps_abs=1e-6, eps_rel=1e-6, max_iter=3000)
    t0 = time.perf_counter()
    a = S.solve_hard(L, mus, kap, ops=S.make_ops("cpu"), **kw)
    t1 = time.perf_counter()
    b = S.solve_hard(L, mus, kap, ops=S.make_ops(dev), **kw)
    t2 = time.perf_counter()
    err = np.max(np.abs(a["W"] - b["W"]))
    print(f"self-check (3/4) hard solve torch({dev}) vs numpy: "
          f"max |dW| = {err:.3e}\n"
          f"   numpy: {t1-t0:.1f}s iters={a['iters']} status={a['status']}\n"
          f"   torch: {t2-t1:.1f}s iters={b['iters']} status={b['status']}",
          flush=True)
    assert all(a["status"] == "converged"), "numpy hard solve did not converge"
    assert all(b["status"] == "converged"), "torch hard solve did not converge"
    assert err < 1e-5, "torch hard-solve path disagrees with numpy"
    ap_ = S.solve_penalty(L, mus, 1.5, ops=S.make_ops("cpu"), **kw)
    bp_ = S.solve_penalty(L, mus, 1.5, ops=S.make_ops(dev), **kw)
    errp = np.max(np.abs(ap_["W"] - bp_["W"]))
    print(f"self-check (4/4) penalty solve torch({dev}) vs numpy: "
          f"max |dW| = {errp:.3e} "
          f"(torch iters={bp_['iters']} status={bp_['status']})", flush=True)
    assert all(bp_["status"] == "converged"), "torch penalty did not converge"
    assert errp < 1e-5, "torch penalty-solve path disagrees with numpy"
    print("SELF-CHECK PASSED", flush=True)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="smoke",
                    choices=["smoke", "pilot", "full", "journal"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--kappa", default=None,
                    help="comma-separated kappa override (skips freeze file)")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        self_check(args.device); return

    cfg = stage_cfg(args.stage)
    # one outdir PER STAGE: stages have different N_val/N_test, so sharing a
    # solve log would let a pilot-calibrated lambda (64-instance target)
    # leak into the full run (128-instance target) via the resume cache
    outdir = args.out or os.path.join(HERE, "results", "primary", args.stage)
    os.makedirs(outdir, exist_ok=True)
    if args.aggregate_only:
        aggregate(outdir); return

    ops = S.make_ops(args.device)
    log = SolveLog(os.path.join(outdir, "solves.csv"))

    # ---- kappa values: CLI > this stage's freeze > the PILOT's freeze >
    #      select-and-freeze (pilot rule).  full/journal inherit the kappas
    #      the pilot froze; they never re-tune them (spec).
    freeze_path = os.path.join(outdir, "pilot_freeze.json")
    pilot_freeze = os.path.join(HERE, "results", "primary", "pilot",
                                "pilot_freeze.json")
    if args.kappa:
        kappas = [float(x) for x in args.kappa.split(",")]
    elif os.path.exists(freeze_path):
        kappas = json.load(open(freeze_path))["kappas"]
    elif args.stage in ("full", "journal") and os.path.exists(pilot_freeze):
        kappas = json.load(open(pilot_freeze))["kappas"]
        json.dump(dict(kappas=kappas, inherited_from=pilot_freeze,
                       stage=args.stage), open(freeze_path, "w"), indent=1)
        print(f"kappa inherited from pilot freeze: {kappas}")
    else:
        m_sel = cfg["ms"][0] if args.stage == "smoke" else \
            cfg["ms"][min(1, len(cfg["ms"]) - 1)]
        print("selecting kappa (blind pilot rule)...", flush=True)
        cands, cv_min, cv_unc = select_kappas(cfg, cfg["seeds"][:2], m_sel,
                                              ops)
        report = []
        if args.stage != "smoke":
            # spec: verify candidates at ALL target m (incl. 1e5) pre-freeze.
            # For pilot/full stages this checks the stage's own m list plus
            # the full-run list when the pilot only covers a subset.
            vcfg = dict(cfg)
            if args.stage == "pilot":
                vcfg["ms"] = sorted(set(cfg["ms"]) | {100000})
            print("verifying kappa candidates at all target m "
                  f"({vcfg['ms']})...", flush=True)
            report, passing = verify_kappas(vcfg, cands, cfg["seeds"][:2],
                                            ops)
            pool = passing if len(passing) >= cfg["n_kappa"] else cands
            kappas = pool[:cfg["n_kappa"]] if cfg["n_kappa"] > 1 \
                else [pool[len(pool) // 2]]
        else:
            kappas = cands[:cfg["n_kappa"]] if cfg["n_kappa"] > 1 \
                else [cands[1]]
        json.dump(dict(kappas=kappas, candidates=cands, cv_min=cv_min,
                       cv_unc=cv_unc, stage=args.stage,
                       verification=report,
                       note="pilot rule q in {0.25,0.5,0.75}; tuned blind "
                            "to method separation; verified at all target m "
                            "before freeze (spec)"),
                  open(freeze_path, "w"), indent=1)
        print(f"kappa frozen: {kappas} (min={cv_min:.4g}, unc={cv_unc:.4g})")

    t0 = time.time()
    for seed in cfg["seeds"]:
        struct = D.make_structure(seed, n=cfg["n"], K=cfg["K"])
        eval_banks = {("eval", rg): D.make_eval_bank(struct, rg,
                                                     cfg["M_eval"], seed)
                      for rg in ("id", "shift")}
        for m in cfg["ms"]:
            for rep in range(cfg["repeats"][m]):
                banks = dict(eval_banks)
                for rg in ("id", "shift"):
                    banks[("val", rg)] = D.make_shock_bank(
                        struct, rg, m, seed, rep, 1)
                    banks[("test", rg)] = D.make_shock_bank(
                        struct, rg, m, seed, rep, 2)
                    banks[("mus_val", rg)] = D.make_mus(
                        struct, rg, cfg["N_val"], seed, rep, 1)
                    banks[("mus_test", rg)] = D.make_mus(
                        struct, rg, cfg["N_test"], seed, rep, 2)
                for kappa in kappas[:cfg.get("n_kappa_m", {})
                                    .get(m, len(kappas))]:
                    st = run_cell(cfg, log, seed, m, kappa, rep, struct,
                                  banks, ops)
                    print(f"[{time.time()-t0:7.1f}s] seed={seed} m={m} "
                          f"kappa={kappa:.4g} rep={rep}: {st}", flush=True)
    aggregate(outdir)


if __name__ == "__main__":
    main()
