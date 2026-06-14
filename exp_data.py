"""
exp_data.py
===========
Synthetic data generator for the primary constraint-targeting experiment, as
frozen in EXPERIMENT_EXECUTION_SPEC.md ("Data Generator" + "Determinism,
Tolerances, And Seeds").

Design rules implemented here:
  * zero-mean SHOCK banks (loss shocks), never full return banks with a baked-in
    mean; each decision instance's mean mu_j is added analytically via CVaR
    translation equivariance,
  * per-(seed, regime) factor structure shared across banks,
  * named, never-reused random streams derived from SeedSequence(root).

Pinned defaults (spec):
  n=50, K=5, beta=0.99, gamma=1.0, w_max=0.20, base_mu=0.05,
  B_ij ~ N(0,1), sigma_idio ~ U[0.5,1.5],
  ID regime    (nu=8, vol=1.00, signal=1.00),
  shift regime (nu=3, vol=1.25, signal=1.75),
  pilot starting values: risk_premium_strength=0.10, alpha_strength=0.05.
"""
import numpy as np

ROOT = 20260609

REGIMES = {
    "id":    dict(nu=8.0, vol=1.00, signal=1.00),
    "shift": dict(nu=3.0, vol=1.25, signal=1.75),
}

DEFAULTS = dict(
    n=50, K=5, beta=0.99, gamma=1.0, w_max=0.20, base_mu=0.05,
    risk_premium_strength=0.10, alpha_strength=0.05,
)

_REGIME_KEY = {"id": 1, "shift": 2}
_PURPOSE_KEY = {"structure": 10, "mu": 11, "bank": 12, "eval": 13}


def rng_for(*key):
    """Named stream: rng_for('bank', seed, repeat, regime_id, m, ...).
    Streams are derived from SeedSequence([ROOT, *encoded key]) and never
    reused across banks, instances, or regimes (spec: Determinism)."""
    enc = []
    for kpart in key:
        if isinstance(kpart, str):
            enc.append(_PURPOSE_KEY.get(kpart, _REGIME_KEY.get(kpart)))
            if enc[-1] is None:
                raise ValueError(f"unknown stream key part: {kpart!r}")
        else:
            enc.append(int(kpart))
    return np.random.default_rng(np.random.SeedSequence([ROOT] + enc))


def make_structure(seed, n=None, K=None):
    """One factor/loading structure per seed (shared by both regimes)."""
    n = n or DEFAULTS["n"]; K = K or DEFAULTS["K"]
    rng = rng_for("structure", seed)
    B = rng.standard_normal((n, K))                      # B_ij ~ N(0,1)
    sigma_idio = rng.uniform(0.5, 1.5, n)                # pinned
    r = np.linalg.norm(B, axis=1) + sigma_idio           # asset risk score
    r = (r - r.mean()) / (r.std() + 1e-30)               # standardized
    return dict(B=B, sigma_idio=sigma_idio, r=r, n=n, K=K)


def _std_t(rng, nu, shape):
    """Student-t draws standardized to unit variance: t_nu / sqrt(nu/(nu-2))."""
    return rng.standard_t(nu, size=shape) / np.sqrt(nu / (nu - 2.0))


def make_mus(struct, regime, N, seed, repeat, split_id,
             base_mu=None, risk_premium_strength=None, alpha_strength=None):
    """Per-instance mean vectors mu_j [N, n]; same mu_j for all methods.
    mu_j = base_mu + signal * rps * r + alpha_strength * a_j,  a_j demeaned."""
    p = REGIMES[regime]
    base_mu = DEFAULTS["base_mu"] if base_mu is None else base_mu
    rps = DEFAULTS["risk_premium_strength"] if risk_premium_strength is None \
        else risk_premium_strength
    a_s = DEFAULTS["alpha_strength"] if alpha_strength is None else alpha_strength
    rng = rng_for("mu", seed, repeat, regime, split_id)
    n = struct["n"]
    A = rng.standard_normal((N, n))
    A -= A.mean(axis=1, keepdims=True)                   # cross-sectional demean
    return base_mu + p["signal"] * rps * struct["r"][None, :] + a_s * A


def make_shock_bank(struct, regime, m, seed, repeat, split_id, purpose="bank"):
    """Zero-mean LOSS-shock bank L [m, n]:
    shock = vol * (B f / sqrt(K) + sigma_idio * eps);  loss_shock = -shock."""
    p = REGIMES[regime]
    rng = rng_for(purpose, seed, repeat, regime, split_id, m)
    B, sig, K = struct["B"], struct["sigma_idio"], struct["K"]
    f = _std_t(rng, p["nu"], (m, K))
    eps = _std_t(rng, p["nu"], (m, struct["n"]))
    shock = p["vol"] * (f @ B.T / np.sqrt(K) + eps * sig[None, :])
    return -shock                                        # loss shocks


def make_eval_bank(struct, regime, M_eval, seed):
    """Held-out realized shock bank, fixed across methods, m, and repeats for a
    given (seed, regime). repeat=0/split_id=0 by construction (never reused)."""
    return make_shock_bank(struct, regime, M_eval, seed, 0, 0, purpose="eval")


def k_int(m, beta=None):
    beta = DEFAULTS["beta"] if beta is None else beta
    tau = (1.0 - beta) * m
    k = int(round(tau))
    assert abs(tau - k) < 1e-9, f"(1-beta)*m must be integer, got {tau}"
    return k
