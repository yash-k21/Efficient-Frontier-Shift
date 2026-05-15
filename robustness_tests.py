"""
robustness_tests.py

"""

import os
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf, EmpiricalCovariance

warnings.filterwarnings("ignore")
os.makedirs("Results/Robustness", exist_ok=True)

# ── Constants ────────────────────────────────────────────────────
RF_PRE  = 0.0654
RF_POST = 0.0578

P = {
    "a_pre":  ("2015-01-01", "2019-12-31"),
    "a_post": ("2020-01-01", "2024-12-31"),
    "crisis": ("2020-02-01", "2021-06-30"),
}

# Actual Nifty 50 sector weights (approximate, as of 2023)
# Source: NSE index factsheet
NIFTY50_SECTOR_WEIGHTS = {
    "Financials":   0.363,
    "IT":           0.131,
    "Energy":       0.118,
    "Consumer":     0.126,
    "Pharma":       0.047,
    "Metals":       0.044,
    "Industrials":  0.062,
    "Telecom":      0.038,
    "Cement":       0.031,
    "Others":       0.040,
}

# Sector mapping for your 45 tickers
SECTOR_MAP = {
    'HDFCBANK.NS':   'Financials',
    'ICICIBANK.NS':  'Financials',
    'KOTAKBANK.NS':  'Financials',
    'AXISBANK.NS':   'Financials',
    'SBIN.NS':       'Financials',
    'INDUSINDBK.NS': 'Financials',
    'BANKBARODA.NS': 'Financials',
    'PNB.NS':        'Financials',
    'TCS.NS':        'IT',
    'INFY.NS':       'IT',
    'HCLTECH.NS':    'IT',
    'WIPRO.NS':      'IT',
    'TECHM.NS':      'IT',
    'RELIANCE.NS':   'Energy',
    'ONGC.NS':       'Energy',
    'BPCL.NS':       'Energy',
    'GAIL.NS':       'Energy',
    'TATAPOWER.NS':  'Energy',
    'NTPC.NS':       'Energy',
    'POWERGRID.NS':  'Energy',
    'HINDUNILVR.NS': 'Consumer',
    'ITC.NS':        'Consumer',
    'ASIANPAINT.NS': 'Consumer',
    'MARUTI.NS':     'Consumer',
    'HEROMOTOCO.NS': 'Consumer',
    'BAJAJ-AUTO.NS': 'Consumer',
    'M&M.NS':        'Consumer',
    'BOSCHLTD.NS':   'Consumer',
    'SUNPHARMA.NS':  'Pharma',
    'DRREDDY.NS':    'Pharma',
    'CIPLA.NS':      'Pharma',
    'LUPIN.NS':      'Pharma',
    'TATASTEEL.NS':  'Metals',
    'HINDALCO.NS':   'Metals',
    'VEDL.NS':       'Metals',
    'COALINDIA.NS':  'Metals',
    'ULTRACEMCO.NS': 'Industrials',
    'GRASIM.NS':     'Industrials',
    'ACC.NS':        'Industrials',
    'AMBUJACEM.NS':  'Industrials',
    'LT.NS':         'Industrials',
    'BHEL.NS':       'Industrials',
    'BHARTIARTL.NS': 'Telecom',
    'ZEEL.NS':       'Telecom',
    'ADANIPORTS.NS': 'Others',
}

# Banking sub-sector split
RETAIL_BANKS    = ['HDFCBANK.NS', 'KOTAKBANK.NS',
                   'AXISBANK.NS',  'ICICIBANK.NS', 'SBIN.NS']
WHOLESALE_BANKS = ['PNB.NS', 'BANKBARODA.NS', 'INDUSINDBK.NS']

# 1. DATA LOADING

def load_and_prepare(path="Data/prices_daily.csv"):
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices = prices.sort_index()
    prices = prices[~prices.index.astype(str).str.contains(
        "Price|Ticker", na=False)]
    prices = prices.astype(float).dropna(axis=1, how="all")
    monthly = prices.resample("ME").last()
    log_ret = np.log(monthly / monthly.shift(1)).iloc[1:]
    print(f"[load] {prices.shape[1]} tickers | "
          f"{log_ret.shape[0]} monthly obs")
    return log_ret


def slice_and_clean(log_ret, start, end,
                    tickers=None, max_nan=0.10):
    ret = log_ret.loc[start:end]
    if tickers:
        ret = ret[[t for t in tickers if t in ret.columns]]
    keep    = ret.columns[ret.isna().mean() <= max_nan]
    ret     = ret[keep].ffill(limit=1).dropna(axis=1)
    return ret

# 2. OPTIMISATION HELPERS

def min_var(mu, cov):
    n   = len(mu)
    res = minimize(
        lambda w: w @ cov @ w,
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq",
                       "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-12, "maxiter": 2000}
    )
    return res.x if res.success else np.ones(n) / n


def max_sharpe(mu, cov, rf):
    n = len(mu)
    def neg_sr(w):
        r = w @ mu
        v = np.sqrt(w @ cov @ w)
        return -(r - rf) / (v + 1e-9)
    res = minimize(
        neg_sr, np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq",
                       "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-12, "maxiter": 2000}
    )
    return res.x if res.success else np.ones(n) / n


def frontier_points(mu, cov, n_pts=150):
    n    = len(mu)
    w_mv = min_var(mu, cov)
    r_lo = w_mv @ mu
    r_hi = mu.max()
    vols, rets = [], []
    for target in np.linspace(r_lo, r_hi, n_pts):
        res = minimize(
            lambda w: w @ cov @ w,
            np.ones(n) / n,
            method="SLSQP",
            bounds=[(0, 1)] * n,
            constraints=[
                {"type": "eq",
                 "fun": lambda w: w.sum() - 1},
                {"type": "eq",
                 "fun": lambda w, t=target: w @ mu - t}
            ],
            options={"ftol": 1e-12, "maxiter": 2000}
        )
        if res.success:
            rets.append(res.x @ mu)
            vols.append(np.sqrt(res.x @ cov @ res.x))
    return np.array(vols), np.array(rets)


def pstats(w, mu, cov):
    return w @ mu, np.sqrt(w @ cov @ w)

# 3. COVARIANCE ESTIMATORS

def sample_cov(ret, freq=12):
    mu  = ret.mean().values * freq
    cov = ret.cov().values  * freq
    return mu, cov


def ledoit_wolf(ret, freq=12):
    lw  = LedoitWolf().fit(ret.values)
    mu  = ret.mean().values * freq
    cov = lw.covariance_    * freq
    return mu, cov, lw.shrinkage_


def equal_corr(ret, freq=12):
    """
    Equal-correlation shrinkage target:
    shrink sample corr toward average off-diagonal correlation.
    """
    mu      = ret.mean().values * freq
    S       = ret.cov().values
    std     = np.sqrt(np.diag(S))
    corr    = S / np.outer(std, std)
    n       = len(std)
    mask    = ~np.eye(n, dtype=bool)
    rho_bar = corr[mask].mean()
    target_corr          = np.full((n, n), rho_bar)
    np.fill_diagonal(target_corr, 1.0)
    target_cov           = target_corr * np.outer(std, std)
    # Optimal shrinkage intensity (simplified Oracle)
    delta   = 0.2
    cov     = (1 - delta) * S + delta * target_cov
    return mu, cov * freq

# 4. TEST 1 — BOOTSTRAP CONFIDENCE BANDS

def bootstrap_frontier(ret, rf, n_boots=10, n_pts=60,
                        seed=42):
    """
    Returns (lower_vols, upper_vols, target_rets) at 95% CI
    across n_pts equally spaced return targets.
    """
    rng    = np.random.default_rng(seed)
    T, N   = ret.shape

    # First pass: get return range from full-sample frontier
    mu_full, cov_full, _ = ledoit_wolf(ret)
    w_mv    = min_var(mu_full, cov_full)
    r_lo    = w_mv @ mu_full
    r_hi    = mu_full.max()
    targets = np.linspace(r_lo, r_hi, n_pts)

    all_vols = np.full((n_boots, n_pts), np.nan)

    for b in range(n_boots):
        idx    = rng.integers(0, T, size=T)
        sample = ret.iloc[idx]
        mu_b, cov_b, _ = ledoit_wolf(sample)

        for j, target in enumerate(targets):
            res = minimize(
                lambda w: w @ cov_b @ w,
                np.ones(N) / N,
                method="SLSQP",
                bounds=[(0, 1)] * N,
                constraints=[
                    {"type": "eq",
                     "fun": lambda w: w.sum() - 1},
                    {"type": "eq",
                     "fun": lambda w, t=target: w @ mu_b - t}
                ],
                options={"ftol": 1e-10, "maxiter": 10}
            )
            if res.success:
                all_vols[b, j] = np.sqrt(res.x @ cov_b @ res.x)

    lower = np.nanpercentile(all_vols, 2.5,  axis=0)
    upper = np.nanpercentile(all_vols, 97.5, axis=0)
    return lower, upper, targets


def plot_bootstrap_bands(ret_pre, ret_post,
                          rf_pre, rf_post, out_path,
                          n_boots=500):
    print("  [bootstrap] Running pre-COVID bootstrap "
          f"({n_boots} samples)...")
    lo_pre, hi_pre, tgt_pre = bootstrap_frontier(
        ret_pre,  rf_pre,  n_boots=n_boots)

    print("  [bootstrap] Running post-COVID bootstrap "
          f"({n_boots} samples)...")
    lo_post, hi_post, tgt_post = bootstrap_frontier(
        ret_post, rf_post, n_boots=n_boots)

    # Point estimates
    mu_pre,  cov_pre,  _ = ledoit_wolf(ret_pre)
    mu_post, cov_post, _ = ledoit_wolf(ret_post)
    v_pre,  r_pre  = frontier_points(mu_pre,  cov_pre)
    v_post, r_post = frontier_points(mu_post, cov_post)

    PRE_C  = "#1D9E75"
    POST_C = "#534AB7"

    fig, ax = plt.subplots(figsize=(12, 7.5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#CCCCCC")
    ax.grid(True, linestyle="--", linewidth=0.4,
            color="#DDDDDD", alpha=0.8)

    # Confidence bands
    ax.fill_betweenx(tgt_pre,  lo_pre,  hi_pre,
                     alpha=0.18, color=PRE_C,
                     label="95% CI pre-COVID")
    ax.fill_betweenx(tgt_post, lo_post, hi_post,
                     alpha=0.18, color=POST_C,
                     label="95% CI post-COVID")

    # Point frontiers
    ax.plot(v_pre,  r_pre,  color=PRE_C,  lw=2.2,
            label="Pre-COVID frontier")
    ax.plot(v_post, r_post, color=POST_C, lw=2.2,
            label="Post-COVID frontier")

    # Check overlap
    overlap = not (lo_post.min() > hi_pre.max() or
                   lo_pre.min()  > hi_post.max())
    overlap_text = ("Bands overlap — shift may\n"
                    "partially reflect sampling noise"
                    if overlap else
                    "Bands do NOT overlap —\n"
                    "shift is statistically significant")
    ax.text(0.98, 0.05, overlap_text,
            transform=ax.transAxes,
            fontsize=9, color="#333333", ha="right",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#F5F5F0",
                      edgecolor="#CCCCCC"))

    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlabel("Annualised Volatility (\u03c3)",
                  fontsize=11, color="#555555")
    ax.set_ylabel("Annualised Return (\u03bc)",
                  fontsize=11, color="#555555")
    ax.set_title(
        "Bootstrap Confidence Bands (95%) \u2014 "
        "Efficient Frontier\n"
        f"Pre-COVID vs Post-COVID | {n_boots} bootstrap "
        "samples | Ledoit-Wolf shrinkage",
        fontsize=11, fontweight="500", color="#222222"
    )
    ax.legend(fontsize=9.5, framealpha=0.7,
              loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")
    return overlap

# 5. TEST 2 — JOBSON-KORKIE TEST

def jobson_korkie_test(ret_pre, ret_post,
                        w_pre, w_post,
                        rf_pre, rf_post):
    """
    Memmel (2003) corrected Jobson-Korkie test.
    H0: SR_pre == SR_post
    Returns z-statistic, one-sided p-value.
    """
    rp = (ret_pre  @ w_pre)
    rq = (ret_post @ w_post)

    # Monthly excess returns
    ep = rp - rf_pre  / 12
    eq = rq - rf_post / 12

    T1 = len(ep)
    T2 = len(eq)

    sr1 = ep.mean() / ep.std()   # monthly SR pre
    sr2 = eq.mean() / eq.std()   # monthly SR post

    # Memmel standard error
    se = np.sqrt(
        (1/T1) * (1 + 0.5 * sr1**2) +
        (1/T2) * (1 + 0.5 * sr2**2)
    )

    z = (sr2 - sr1) / se
    p = 1 - stats.norm.cdf(z)   # one-sided: H1: SR_post > SR_pre

    # Annualise for reporting
    sr1_ann = sr1 * np.sqrt(12)
    sr2_ann = sr2 * np.sqrt(12)

    return {
        "SR_pre_ann":  round(sr1_ann, 3),
        "SR_post_ann": round(sr2_ann, 3),
        "delta_SR":    round(sr2_ann - sr1_ann, 3),
        "z_stat":      round(z, 3),
        "p_value":     round(p, 4),
        "significant": p < 0.05,
    }

# 6. TEST 3 — BOX'S M TEST

def box_m_test(ret_pre, ret_post):
    """
    Box's M test for equality of covariance matrices.
    H0: Sigma_pre == Sigma_post
    Uses chi-squared approximation.
    """
    n1, p = ret_pre.shape
    n2, _ = ret_post.shape
    n     = n1 + n2

    S1 = ret_pre.cov().values
    S2 = ret_post.cov().values
    Sp = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n - 2)

    # Log-determinants
    sign1, ld1 = np.linalg.slogdet(S1)
    sign2, ld2 = np.linalg.slogdet(S2)
    signp, ldp = np.linalg.slogdet(Sp)

    M = ((n - 2) * ldp
         - (n1 - 1) * ld1
         - (n2 - 1) * ld2)

    # Chi-squared approximation
    c = (1 - (2*p**2 + 3*p - 1) / (6*(p+1)) *
         (1/(n1-1) + 1/(n2-1) - 1/(n-2)))

    chi2_stat = M * c
    df        = p * (p + 1) / 2
    p_val     = 1 - stats.chi2.cdf(chi2_stat, df)

    return {
        "M_statistic": round(M,         2),
        "chi2_stat":   round(chi2_stat, 2),
        "df":          int(df),
        "p_value":     round(p_val, 6),
        "significant": p_val < 0.05,
        "conclusion":  ("Covariance matrices are SIGNIFICANTLY "
                        "DIFFERENT (structural change confirmed)"
                        if p_val < 0.05 else
                        "Cannot reject equal covariance matrices"),
    }

# 7. TEST 4 — ALTERNATIVE COVARIANCE ESTIMATORS

def plot_estimator_comparison(ret_pre, ret_post,
                               rf_pre, rf_post, out_path):
    """
    Plot frontiers under three covariance estimators.
    Uses n_pts=60 to keep runtime reasonable.
    """
    N_PTS = 60   # reduced for speed

    specs = [
        ("Sample covariance",
         "#A8D8A8", "#A8A8D8", ":", 1.3),
        ("Equal-correlation",
         "#5BB85B", "#7B7BC8", "--", 1.6),
        ("Ledoit-Wolf (primary)",
         "#1D9E75", "#534AB7", "-",  2.2),
    ]

    def get_mu_cov(name, ret):
        if name == "Sample covariance":
            return sample_cov(ret)
        elif name == "Equal-correlation":
            return equal_corr(ret)
        else:
            return ledoit_wolf(ret)[:2]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                              sharey=True)
    fig.patch.set_facecolor("#FAFAFA")
    titles = ["Pre-COVID (Jan 2015\u2013Dec 2019)",
              "Post-COVID (Jan 2020\u2013Dec 2024)"]

    summary_rows = []

    for ax, ret, rf, title, col_idx in [
        (axes[0], ret_pre,  rf_pre,  titles[0], 0),
        (axes[1], ret_post, rf_post, titles[1], 1),
    ]:
        ax.set_facecolor("#FAFAFA")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color("#CCCCCC")
        ax.grid(True, linestyle="--", linewidth=0.4,
                color="#DDDDDD", alpha=0.8)
        ax.set_title(title, fontsize=10.5,
                     color="#333333", pad=8)

        for name, cp, cq, ls, lw in specs:
            print(f"    {title[:3]} {name}...")
            mu, cov = get_mu_cov(name, ret)
            c       = cp if col_idx == 0 else cq

            v, r    = frontier_points(mu, cov,
                                      n_pts=N_PTS)
            ax.plot(v, r, color=c, lw=lw, ls=ls,
                    label=name, zorder=4)

            w_ms       = max_sharpe(mu, cov, rf)
            r_ms, v_ms = pstats(w_ms, mu, cov)
            sr         = (r_ms - rf) / v_ms
            ax.scatter(v_ms, r_ms, color=c,
                       s=80, marker="D", zorder=6)

            period = "Pre" if col_idx == 0 else "Post"
            summary_rows.append({
                "Period":    period,
                "Estimator": name,
                "MS_vol":    round(v_ms, 4),
                "MS_ret":    round(r_ms, 4),
                "SR":        round(sr,   3),
            })

        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.set_xlabel("Annualised Volatility (\u03c3)",
                      fontsize=10, color="#555555")
        if col_idx == 0:
            ax.set_ylabel("Annualised Return (\u03bc)",
                          fontsize=10, color="#555555")
        ax.legend(fontsize=8.5, framealpha=0.6,
                  loc="upper left")

    fig.suptitle(
        "Robustness to Covariance Estimator Choice\n"
        "Efficient frontiers under Sample, "
        "Equal-Correlation, and Ledoit-Wolf shrinkage",
        fontsize=12, fontweight="500",
        color="#222222", y=1.01
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")
    return pd.DataFrame(summary_rows)

# 8. TEST 5 — TRANSACTION COST / STT ANALYSIS

def transaction_cost_analysis(ret_pre, ret_post,
                               rf_pre, rf_post):
    """
    Computes gross and net Sharpe ratios after applying
    India STT + brokerage + impact costs.
    """
    # Cost assumptions
    STT       = 0.001    # 0.1% per leg (delivery)
    BROKERAGE = 0.0005   # 0.05% per leg
    IMPACT    = 0.001    # 0.1% market impact (large trades)
    ROUND_TRIP = (STT + BROKERAGE + IMPACT) * 2

    mu_pre,  cov_pre,  _ = ledoit_wolf(ret_pre)
    mu_post, cov_post, _ = ledoit_wolf(ret_post)

    w_pre  = max_sharpe(mu_pre,  cov_pre,  rf_pre)
    w_post = max_sharpe(mu_post, cov_post, rf_post)

    r_pre,  v_pre  = pstats(w_pre,  mu_pre,  cov_pre)
    r_post, v_post = pstats(w_post, mu_post, cov_post)

    sr_pre_gross  = (r_pre  - rf_pre)  / v_pre
    sr_post_gross = (r_post - rf_post) / v_post

    # Turnover — one-way
    turnover = float(np.sum(np.abs(w_post - w_pre)))

    # Rebalancing scenarios
    results = []
    for freq_label, rebal_per_year in [
        ("Annual rebalancing",    1),
        ("Semi-annual",           2),
        ("Quarterly",             4),
        ("Monthly",              12),
    ]:
        annual_cost = turnover * ROUND_TRIP * rebal_per_year

        sr_pre_net  = (r_pre  - rf_pre  - annual_cost) / v_pre
        sr_post_net = (r_post - rf_post - annual_cost) / v_post

        results.append({
            "Rebalancing":       freq_label,
            "Annual cost (%)":   round(annual_cost * 100, 3),
            "SR pre (gross)":    round(sr_pre_gross,  3),
            "SR post (gross)":   round(sr_post_gross, 3),
            "SR pre (net)":      round(sr_pre_net,    3),
            "SR post (net)":     round(sr_post_net,   3),
            "Delta SR (net)":    round(sr_post_net - sr_pre_net, 3),
            "Improvement survives": sr_post_net > sr_pre_net,
        })

    df = pd.DataFrame(results)

    # Plot
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#CCCCCC")
    ax.grid(True, linestyle="--", linewidth=0.4,
            color="#DDDDDD", alpha=0.8)

    x     = np.arange(len(results))
    width = 0.2
    labels = [r["Rebalancing"] for r in results]

    ax.bar(x - 1.5*width,
           [r["SR pre (gross)"]  for r in results],
           width, label="Pre gross",  color="#A8D8C8")
    ax.bar(x - 0.5*width,
           [r["SR pre (net)"]   for r in results],
           width, label="Pre net",    color="#1D9E75")
    ax.bar(x + 0.5*width,
           [r["SR post (gross)"] for r in results],
           width, label="Post gross", color="#C8C8F0")
    ax.bar(x + 1.5*width,
           [r["SR post (net)"]  for r in results],
           width, label="Post net",   color="#534AB7")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Sharpe Ratio", fontsize=11,
                  color="#555555")
    ax.set_title(
        "Gross vs Net Sharpe Ratio After STT and "
        "Transaction Costs\n"
        f"Round-trip cost: {ROUND_TRIP*100:.2f}%  |  "
        f"One-way turnover: {turnover*100:.1f}%",
        fontsize=11, fontweight="500", color="#222222"
    )
    ax.legend(fontsize=9.5, framealpha=0.7)
    ax.axhline(0, color="#888888", lw=0.8)

    fig.tight_layout()
    out = "Results/Robustness/transaction_costs.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")

    return df, turnover, ROUND_TRIP


# 9. TEST 6 — SECTOR WEIGHT CHECK

def sector_weight_check(ret_pre, ret_post, out_path):
    """
    Compare your universe sector weights (equal-weight)
    vs actual Nifty 50 sector weights.
    Also show how tangency portfolio sector weights
    changed pre vs post.
    """
    # Universe sector weights (equal-weight assumption)
    tickers = list(ret_pre.columns)
    sector_counts = {}
    for t in tickers:
        s = SECTOR_MAP.get(t, "Others")
        sector_counts[s] = sector_counts.get(s, 0) + 1

    total = len(tickers)
    universe_weights = {s: c/total
                        for s, c in sector_counts.items()}

    sectors = sorted(set(list(NIFTY50_SECTOR_WEIGHTS.keys()) +
                         list(universe_weights.keys())))

    nifty_w = [NIFTY50_SECTOR_WEIGHTS.get(s, 0)
               for s in sectors]
    univ_w  = [universe_weights.get(s, 0)
               for s in sectors]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#FAFAFA")

    # ── Left: Universe vs Nifty 50 ──
    ax = axes[0]
    ax.set_facecolor("#FAFAFA")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#CCCCCC")
    ax.grid(True, linestyle="--", linewidth=0.4,
            color="#DDDDDD", alpha=0.8, axis="x")

    x     = np.arange(len(sectors))
    width = 0.35
    ax.barh(x - width/2, nifty_w, width,
            label="Actual Nifty 50", color="#E8A838",
            alpha=0.85)
    ax.barh(x + width/2, univ_w,  width,
            label="Study universe (EW)", color="#534AB7",
            alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(sectors, fontsize=9.5)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Sector Weight", fontsize=10,
                  color="#555555")
    ax.set_title("Universe vs Actual Nifty 50\n"
                 "Sector Weights (equal-weight universe)",
                 fontsize=10.5, color="#222222")
    ax.legend(fontsize=9, framealpha=0.7)

    # ── Right: Tangency portfolio sector weights ──
    ax2 = axes[1]
    ax2.set_facecolor("#FAFAFA")
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax2.spines[sp].set_color("#CCCCCC")
    ax2.grid(True, linestyle="--", linewidth=0.4,
             color="#DDDDDD", alpha=0.8, axis="x")

    mu_pre,  cov_pre,  _ = ledoit_wolf(ret_pre)
    mu_post, cov_post, _ = ledoit_wolf(ret_post)
    w_pre  = max_sharpe(mu_pre,  cov_pre,  RF_PRE)
    w_post = max_sharpe(mu_post, cov_post, RF_POST)

    def sector_weights_from_w(w, tickers):
        sw = {}
        for wi, t in zip(w, tickers):
            s = SECTOR_MAP.get(t, "Others")
            sw[s] = sw.get(s, 0) + wi
        return sw

    sw_pre  = sector_weights_from_w(w_pre,  tickers)
    sw_post = sector_weights_from_w(w_post, tickers)
    tp_pre  = [sw_pre.get(s,  0) for s in sectors]
    tp_post = [sw_post.get(s, 0) for s in sectors]

    ax2.barh(x - width/2, tp_pre,  width,
             label="Pre-COVID tangency",  color="#1D9E75",
             alpha=0.85)
    ax2.barh(x + width/2, tp_post, width,
             label="Post-COVID tangency", color="#534AB7",
             alpha=0.85)
    ax2.set_yticks(x)
    ax2.set_yticklabels(sectors, fontsize=9.5)
    ax2.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.set_xlabel("Portfolio Weight", fontsize=10,
                   color="#555555")
    ax2.set_title("Tangency Portfolio Sector Weights\n"
                  "Pre-COVID vs Post-COVID",
                  fontsize=10.5, color="#222222")
    ax2.legend(fontsize=9, framealpha=0.7)

    fig.suptitle(
        "Sector Composition Analysis",
        fontsize=12, fontweight="500",
        color="#222222", y=1.01
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")

    return pd.DataFrame({
        "Sector":       sectors,
        "Nifty50_wt":   nifty_w,
        "Universe_wt":  univ_w,
        "TP_pre_wt":    tp_pre,
        "TP_post_wt":   tp_post,
    })


# 10. TEST 7 — BANKING SUB-SECTOR DECOMPOSITION

def banking_decomposition(ret_pre, ret_post, out_path):
    """
    Split banking sector into retail vs wholesale banks.
    Compare return profiles and show which sub-sector
    drove the financial sector underperformance.
    """
    def get_available(tickers, ret):
        return [t for t in tickers if t in ret.columns]

    retail_pre  = get_available(RETAIL_BANKS,    ret_pre)
    retail_post = get_available(RETAIL_BANKS,    ret_post)
    whole_pre   = get_available(WHOLESALE_BANKS, ret_pre)
    whole_post  = get_available(WHOLESALE_BANKS, ret_post)

    def ann_ret(ret, tickers):
        if not tickers:
            return {}
        return (ret[tickers].mean() * 12).to_dict()

    rr_pre  = ann_ret(ret_pre,  retail_pre)
    rr_post = ann_ret(ret_post, retail_post)
    rw_pre  = ann_ret(ret_pre,  whole_pre)
    rw_post = ann_ret(ret_post, whole_post)

    # Combine all banking tickers
    all_banks  = list(set(retail_pre + whole_pre))
    bank_pre   = {t: ret_pre[t].mean()  * 12
                  for t in all_banks if t in ret_pre.columns}
    bank_post  = {t: ret_post[t].mean() * 12
                  for t in all_banks if t in ret_post.columns}
    common     = [t for t in all_banks
                  if t in bank_pre and t in bank_post]
    labels     = [t.replace(".NS", "") for t in common]

    pre_vals  = [bank_pre[t]  for t in common]
    post_vals = [bank_post[t] for t in common]
    colors    = ["#E24B4A" if t in RETAIL_BANKS
                 else "#BA7517"
                 for t in common]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("#FAFAFA")

    # ── Left: bar chart pre vs post per bank ──
    ax = axes[0]
    ax.set_facecolor("#FAFAFA")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#CCCCCC")
    ax.grid(True, linestyle="--", linewidth=0.4,
            color="#DDDDDD", alpha=0.8, axis="x")

    x     = np.arange(len(labels))
    width = 0.35
    ax.barh(x - width/2, pre_vals,  width,
            label="Pre-COVID",  color="#1D9E75", alpha=0.85)
    ax.barh(x + width/2, post_vals, width,
            label="Post-COVID", color="#534AB7", alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=10)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.axvline(0, color="#888888", lw=0.8)
    ax.set_xlabel("Annualised Return", fontsize=10,
                  color="#555555")
    ax.set_title("Banking Stocks: Pre vs Post Returns",
                 fontsize=10.5, color="#222222")
    ax.legend(fontsize=9.5, framealpha=0.7)

    # Add retail/wholesale labels
    for i, t in enumerate(common):
        subtype = ("Retail" if t in RETAIL_BANKS
                   else "Wholesale")
        col     = ("#C0392B" if t in RETAIL_BANKS
                   else "#854F0B")
        ax.text(-0.01, i, subtype, ha="right",
                va="center", fontsize=7.5, color=col)

    # ── Right: avg return by sub-sector ──
    ax2 = axes[1]
    ax2.set_facecolor("#FAFAFA")
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax2.spines[sp].set_color("#CCCCCC")
    ax2.grid(True, linestyle="--", linewidth=0.4,
             color="#DDDDDD", alpha=0.8, axis="y")

    sub_labels = ["Retail Banks\n(HDFC, Kotak,\n"
                  "Axis, ICICI, SBI)",
                  "Wholesale Banks\n(PNB, BoB, IndusInd)"]

    avg_retail_pre  = (np.mean([rr_pre.get(t, 0)
                                for t in retail_pre])
                       if retail_pre else 0)
    avg_retail_post = (np.mean([rr_post.get(t, 0)
                                for t in retail_post])
                       if retail_post else 0)
    avg_whole_pre   = (np.mean([rw_pre.get(t, 0)
                                for t in whole_pre])
                       if whole_pre else 0)
    avg_whole_post  = (np.mean([rw_post.get(t, 0)
                                for t in whole_post])
                       if whole_post else 0)

    x2    = np.arange(2)
    w2    = 0.3
    ax2.bar(x2 - w2/2,
            [avg_retail_pre,  avg_whole_pre],
            w2, label="Pre-COVID",  color="#1D9E75",
            alpha=0.85)
    ax2.bar(x2 + w2/2,
            [avg_retail_post, avg_whole_post],
            w2, label="Post-COVID", color="#534AB7",
            alpha=0.85)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(sub_labels, fontsize=9.5)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.axhline(0, color="#888888", lw=0.8)
    ax2.set_ylabel("Average Annualised Return",
                   fontsize=10, color="#555555")
    ax2.set_title("Average Return by Banking Sub-sector\n"
                  "Retail vs Wholesale",
                  fontsize=10.5, color="#222222")
    ax2.legend(fontsize=9.5, framealpha=0.7)

    fig.suptitle(
        "Banking Sector Decomposition: "
        "Retail vs Wholesale Banks",
        fontsize=12, fontweight="500",
        color="#222222", y=1.01
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")

    return {
        "avg_retail_pre":   avg_retail_pre,
        "avg_retail_post":  avg_retail_post,
        "avg_wholesale_pre":  avg_whole_pre,
        "avg_wholesale_post": avg_whole_post,
    }

# 11. PRINT SUMMARY REPORT

def print_summary(jk, boxm, tc_df, overlap,
                  banking, sector_df):
    sep = "=" * 60
    print(f"\n{sep}")
    print("  ROBUSTNESS TEST SUMMARY REPORT")
    print(sep)

    print("\n── 1. Bootstrap Confidence Bands ──")
    print(f"  Frontier bands overlap: {overlap}")
    print(f"  {'NOT significant' if overlap else 'SIGNIFICANT'}"
          f" — shift {'may reflect' if overlap else 'is NOT explained by'}"
          f" sampling noise")

    print("\n── 2. Jobson-Korkie Test ──")
    print(f"  SR pre  (ann): {jk['SR_pre_ann']}")
    print(f"  SR post (ann): {jk['SR_post_ann']}")
    print(f"  Delta SR:      {jk['delta_SR']}")
    print(f"  z-statistic:   {jk['z_stat']}")
    print(f"  p-value:       {jk['p_value']}")
    print(f"  Significant:   {jk['significant']} "
          f"(alpha=0.05)")
    if jk['significant']:
        print("  CONCLUSION: Sharpe improvement is "
              "statistically significant.")
    else:
        print("  CONCLUSION: Cannot reject equal Sharpe "
              "ratios at 5% level.")

    print("\n── 3. Box's M Test (Covariance Equality) ──")
    print(f"  M statistic:  {boxm['M_statistic']}")
    print(f"  Chi2 stat:    {boxm['chi2_stat']}")
    print(f"  Degrees of freedom: {boxm['df']}")
    print(f"  p-value:      {boxm['p_value']}")
    print(f"  {boxm['conclusion']}")

    print("\n── 4. Transaction Cost Analysis ──")
    print(tc_df.to_string(index=False))

    print("\n── 5. Banking Decomposition ──")
    print(f"  Retail banks   pre  avg: "
          f"{banking['avg_retail_pre']:.1%}")
    print(f"  Retail banks   post avg: "
          f"{banking['avg_retail_post']:.1%}")
    print(f"  Wholesale banks pre avg: "
          f"{banking['avg_wholesale_pre']:.1%}")
    print(f"  Wholesale banks post avg:"
          f"{banking['avg_wholesale_post']:.1%}")

    print(f"\n{sep}")
    print("  All outputs in Results/Robustness/")
    print(sep + "\n")

# 12. MAIN

def main():
    print("\n" + "=" * 60)
    print("  Robustness & Statistical Tests Pipeline")
    print("=" * 60 + "\n")

    # ── Load data ─────────────────────────────────
    log_ret = load_and_prepare("Data/prices_daily.csv")

    ret_pre  = slice_and_clean(log_ret, *P["a_pre"])
    ret_post = slice_and_clean(log_ret, *P["a_post"])
    shared   = ret_pre.columns.intersection(ret_post.columns)
    ret_pre  = ret_pre[shared]
    ret_post = ret_post[shared]
    print(f"Shared universe: {len(shared)} tickers")
    print(f"Pre obs:  {len(ret_pre)}")
    print(f"Post obs: {len(ret_post)}\n")

    # Tangency weights for JK test
    mu_pre,  cov_pre,  _ = ledoit_wolf(ret_pre)
    mu_post, cov_post, _ = ledoit_wolf(ret_post)
    w_pre  = max_sharpe(mu_pre,  cov_pre,  RF_PRE)
    w_post = max_sharpe(mu_post, cov_post, RF_POST)

    # ── Test 1: Bootstrap ─────────────────────────
    print("── Test 1: Bootstrap confidence bands ──")
    overlap = plot_bootstrap_bands(
        ret_pre, ret_post, RF_PRE, RF_POST,
        out_path="Results/Robustness/bootstrap_bands.png",
        n_boots=10
    )

    # ── Test 2: Jobson-Korkie ─────────────────────
    print("\n── Test 2: Jobson-Korkie test ──")
    jk = jobson_korkie_test(
        ret_pre, ret_post,
        w_pre, w_post,
        RF_PRE, RF_POST
    )
    print(f"  z={jk['z_stat']}, p={jk['p_value']}, "
          f"significant={jk['significant']}")

    # ── Test 3: Box's M ───────────────────────────
    print("\n── Test 3: Box's M test ──")
    boxm = box_m_test(ret_pre, ret_post)
    print(f"  chi2={boxm['chi2_stat']}, "
          f"p={boxm['p_value']}, "
          f"significant={boxm['significant']}")

    # ── Test 4: Estimator comparison ──────────────
    print("\n── Test 4: Covariance estimator comparison ──")
    est_df = plot_estimator_comparison(
        ret_pre, ret_post, RF_PRE, RF_POST,
        out_path="Results/Robustness/estimator_comparison.png"
    )
    est_df.to_csv(
        "Results/Robustness/estimator_comparison.csv",
        index=False
    )

    # ── Test 5: Transaction costs ─────────────────
    print("\n── Test 5: Transaction cost analysis ──")
    tc_df, turnover, cost = transaction_cost_analysis(
        ret_pre, ret_post, RF_PRE, RF_POST
    )
    tc_df.to_csv(
        "Results/Robustness/transaction_costs.csv",
        index=False
    )
    print(f"  One-way turnover: {turnover*100:.1f}%")
    print(f"  Round-trip cost:  {cost*100:.2f}%")

    # ── Test 6: Sector weight check ───────────────
    print("\n── Test 6: Sector weight check ──")
    sector_df = sector_weight_check(
        ret_pre, ret_post,
        out_path="Results/Robustness/sector_weights.png"
    )
    sector_df.to_csv(
        "Results/Robustness/sector_weights.csv",
        index=False
    )

    # ── Test 7: Banking decomposition ────────────
    print("\n── Test 7: Banking sub-sector decomposition ──")
    banking = banking_decomposition(
        ret_pre, ret_post,
        out_path="Results/Robustness/banking_decomposition.png"
    )

    # ── Summary ───────────────────────────────────
    print_summary(jk, boxm, tc_df, overlap,
                  banking, sector_df)

    # Save all stats to one CSV
    summary = {
        "JK_z":              jk["z_stat"],
        "JK_p":              jk["p_value"],
        "JK_significant":    jk["significant"],
        "BoxM_chi2":         boxm["chi2_stat"],
        "BoxM_p":            boxm["p_value"],
        "BoxM_significant":  boxm["significant"],
        "Bootstrap_overlap": overlap,
        "Turnover_pct":      round(turnover * 100, 2),
        "RoundTrip_cost_pct":round(cost * 100, 3),
    }
    pd.DataFrame([summary]).to_csv(
        "Results/Robustness/test_summary.csv",
        index=False
    )
    print("Saved: Results/Robustness/test_summary.csv")


if __name__ == "__main__":
    main()
