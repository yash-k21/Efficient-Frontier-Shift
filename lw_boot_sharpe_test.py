"""
Ledoit-Wolf (2008) studentized circular block bootstrap test
for equality of Sharpe ratios across two independent periods.
Reference: Journal of Empirical Finance 15(5): 850-859, §3.2.2
"""

import numpy as np


# ── helpers ──────────────────────────────────────────────────────

def _hac_se_sr(e: np.ndarray) -> float:
    """
    HAC SE for SR̂ via delta method + Newey-West (§3.1).
    Used only on the ORIGINAL data to compute s(D̂).
    """
    T     = len(e)
    mu    = e.mean()
    gamma = (e ** 2).mean()
    sig2  = gamma - mu ** 2
    if sig2 <= 0:
        return np.nan
    grad  = np.array([gamma, -mu / 2]) / sig2 ** 1.5
    y     = np.column_stack([e - mu, e ** 2 - gamma])
    # Andrews (1991) AR(1) bandwidth, Bartlett kernel
    bws = []
    for j in range(2):
        rho = np.clip(np.corrcoef(y[:-1, j], y[1:, j])[0, 1], -0.99, 0.99)
        bws.append(int(np.ceil(1.1447 * (4 * rho**2 / (1-rho)**4 * T) ** (1/3))))
    m   = min(max(1, max(bws)), T // 5)
    # Newey-West LRV with d.f. correction T/(T-2) for 2-vector upsilon=(mu,gamma)
    Psi = y.T @ y / T
    for lag in range(1, m + 1):
        w    = 1 - lag / (m + 1)
        G    = y[lag:].T @ y[:-lag] / T
        Psi += w * (G + G.T)
    Psi *= T / (T - 2)
    return float(np.sqrt(max(grad @ Psi @ grad, 0) / T))


def _boot_se_sr(e: np.ndarray, b: int) -> float:
    """
    Bootstrap SE for SR̂ using the Gotze-Kunsch (1996) block LRV (§3.2.2).
    Used only on BOOTSTRAP data to compute s(D̂*).

    Paper formula:
        l   = floor(T/b)
        f_j = (1/sqrt(b)) * sum_{t=1}^{b} y*_{(j-1)b + t},  j = 1,...,l
        Psi* = (1/l) * sum_j f_j f_j'
    """
    T     = len(e)
    mu    = e.mean()
    gamma = (e ** 2).mean()
    sig2  = gamma - mu ** 2
    if sig2 <= 0:
        return np.nan
    grad  = np.array([gamma, -mu / 2]) / sig2 ** 1.5
    y     = np.column_stack([e - mu, e ** 2 - gamma])
    l     = T // b
    if l == 0:
        return np.nan
    # reshape into l complete blocks of size b, sum within each block, scale by 1/sqrt(b)
    f     = y[:l * b].reshape(l, b, 2).sum(axis=1) / np.sqrt(b)
    Psi   = f.T @ f / l
    return float(np.sqrt(max(grad @ Psi @ grad, 0) / T))


def _cbb(x: np.ndarray, b: int) -> np.ndarray:
    """Circular block bootstrap, returns array of same length as x."""
    T   = len(x)
    idx = np.concatenate([
        np.arange(s, s + b) % T
        for s in np.random.randint(0, T, int(np.ceil(T / b)))
    ])
    return x[idx[:T]]


# ── main test ─────────────────────────────────────────────────────

def lw_boot_sharpe_test(
    exc_pre:  np.ndarray,
    exc_post: np.ndarray,
    B:        int = 4999,
    seed:     int = 42,
) -> dict:
    """
    Two-sided H0: SR_pre = SR_post.
    Adapted for independent samples: Var(D_hat) = Var(SR_pre) + Var(SR_post).
    Block size: T^(1/3), capped to [5, 30].  p-value via Eq.(9) of paper.

    Original data SE  -> Newey-West HAC          (§3.1)
    Bootstrap data SE -> Gotze-Kunsch block LRV  (§3.2.2)
    """
    np.random.seed(seed)

    sr = lambda e: e.mean() / e.std(ddof=1)

    sr1, sr2 = sr(exc_pre), sr(exc_post)
    D_hat    = sr2 - sr1
    se_D     = np.sqrt(_hac_se_sr(exc_pre)**2 + _hac_se_sr(exc_post)**2)
    d_obs    = abs(D_hat) / se_D

    b1 = int(np.clip(round(len(exc_pre)  ** (1/3)), 5, 30))
    b2 = int(np.clip(round(len(exc_post) ** (1/3)), 5, 30))

    d_star = []
    for _ in range(B):
        e1s, e2s = _cbb(exc_pre, b1), _cbb(exc_post, b2)
        D_s      = sr(e2s) - sr(e1s)
        se_s     = np.sqrt(_boot_se_sr(e1s, b1)**2 + _boot_se_sr(e2s, b2)**2)
        if se_s > 0 and not np.isnan(se_s):
            d_star.append(abs(D_s - D_hat) / se_s)

    d_star  = np.array(d_star)
    p_value = (np.sum(d_star >= d_obs) + 1) / (len(d_star) + 1)

    ann = np.sqrt(252)
    return {
        "SR_pre_ann":  round(sr1 * ann, 3),
        "SR_post_ann": round(sr2 * ann, 3),
        "delta_SR":    round(D_hat * ann, 3),
        "d_obs":       round(d_obs, 3),
        "p_value":     round(p_value, 4),
        "significant": p_value < 0.05,
    }
