"""
5_robustness.py
===================
Statistical tests and robustness checks for the Nifty 45
Efficient Frontier Pre/Post COVID study.

Tests included:
  1. Jobson-Korkie test (p-value on Sharpe ratio difference)
  2. Box's M test (covariance matrix equality)
  3. Alternative covariance estimators (sample, equal-corr, Ledoit-Wolf)
  4. Transaction cost / STT net Sharpe analysis (period-specific costs)
  5. Sector weight check vs actual Nifty 50
  6. Banking sub-sector decomposition (retail vs wholesale)

All outputs saved to Results/Robustness/

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
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")
os.makedirs("Results/Robustness", exist_ok=True)

# ════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════

P = {
    "a_pre":  ("2015-01-01", "2020-01-31"),
    "a_post": ("2021-07-01", "2024-12-31"),
    "crisis": ("2020-02-01", "2021-06-30"),
}

# Period-specific transaction cost assumptions
# Pre-COVID: full-service brokers dominated, lower liquidity
# Post-COVID: discount brokers mainstream, deeper liquidity
COSTS = {
    "pre": {
        "STT":        0.001,    # 0.1% per leg — unchanged pre-COVID
        "BROKERAGE":  0.003,    # 0.3% per leg — traditional brokers
        "IMPACT":     0.0012,   # 0.12% per leg — lower liquidity
    },
    "post": {
        "STT":        0.0012,   # blended 0.1% until mid-2024, 0.15% after
        "BROKERAGE":  0.0005,   # 0.05% per leg — discount brokers
        "IMPACT":     0.0007,   # 0.07% per leg — deeper liquidity
    },
}

for period in COSTS:
    c = COSTS[period]
    c["ROUND_TRIP"] = (c["STT"] + c["BROKERAGE"] + c["IMPACT"]) * 2

# Actual Nifty 50 sector weights (NSE factsheet, approx 2023)
NIFTY50_SECTOR_WEIGHTS = {
    "Financials":  0.363,
    "IT":          0.131,
    "Energy":      0.118,
    "Consumer":    0.126,
    "Pharma":      0.047,
    "Metals":      0.044,
    "Industrials": 0.062,
    "Telecom":     0.038,
    "Cement":      0.031,
    "Others":      0.040,
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


# ════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ════════════════════════════════════════════════════════════════

def load_risk_free_rates():
    xl = pd.read_excel(
        'Auctions of 91-Day Government of India Treasury Bills.xlsx',
        header=5,
        usecols=[1, 16]
    )
    xl.columns = ['Date of Auction', 'Weighted Avg Yield (per cent)']
    xl['Date of Auction'] = pd.to_datetime(xl['Date of Auction'], errors='coerce')
    xl = xl.dropna(subset=['Date of Auction', 'Weighted Avg Yield (per cent)'])
    xl = xl.set_index('Date of Auction').sort_index()
    yields = pd.to_numeric(
        xl['Weighted Avg Yield (per cent)'], errors='coerce'
    ).dropna() / 100.0

    rf_pre  = yields.loc[P["a_pre"][0]:P["a_pre"][1]].mean()
    rf_post = yields.loc[P["a_post"][0]:P["a_post"][1]].mean()
    print(f"  RF pre : {rf_pre*100:.2f}%   RF post : {rf_post*100:.2f}%")
    return float(rf_pre), float(rf_post)


def load_prices():
    prices = pd.read_csv("Data/prices_daily.csv", index_col=0, parse_dates=True)
    prices = prices[~prices.index.astype(str).str.contains("Price|Ticker", na=False)]
    return prices.astype(float).dropna(axis=1, how="all").sort_index()


def to_log_returns(prices):
    log_ret = np.log(prices / prices.shift(1)).iloc[1:]
    print(f"  {log_ret.shape[1]} tickers  {log_ret.shape[0]} daily obs")
    return log_ret


def slice_period(log_ret, start, end, max_nan=0.10):
    ret  = log_ret.loc[start:end]
    keep = ret.columns[ret.isna().mean() <= max_nan]
    return ret[keep].ffill(limit=1).dropna(axis=1)


# ════════════════════════════════════════════════════════════════
# 2. OPTIMISATION HELPERS
# ════════════════════════════════════════════════════════════════

def min_var(mu, cov):
    n = len(mu)
    res = minimize(
        lambda w: w @ cov @ w,
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq",
                       "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-12, "maxiter": 2000}
    )
    if not res.success:
        print(f"  [warn] min_var failed ({res.message}); using equal weights")
        return np.ones(n) / n
    return res.x


def max_sharpe(mu, cov, rf):
    n = len(mu)
    def neg_sr(w):
        r = w @ mu
        v = np.sqrt(w @ cov @ w)
        return -(r - rf) / (v + 1e-12)
    res = minimize(
        neg_sr,
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq",
                       "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-12, "maxiter": 2000}
    )
    if not res.success:
        print(f"  [warn] max_sharpe failed ({res.message}); using equal weights")
        return np.ones(n) / n
    return res.x


def frontier_points(mu, cov, n_pts=120):
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


# ════════════════════════════════════════════════════════════════
# 3. COVARIANCE ESTIMATORS
# ════════════════════════════════════════════════════════════════

def sample_cov(ret, freq=252):
    """Raw sample covariance — no shrinkage."""
    mu  = ret.mean().values * freq
    cov = ret.cov().values  * freq
    return mu, cov


def ledoit_wolf(ret, freq=252):
    """Ledoit-Wolf analytical shrinkage — primary estimator."""
    lw  = LedoitWolf().fit(ret.values)
    mu  = ret.mean().values * freq
    cov = lw.covariance_    * freq
    return mu, cov, lw.shrinkage_



# ════════════════════════════════════════════════════════════════
# 4. TEST 1 — JOBSON-KORKIE TEST (BASE)
# ════════════════════════════════════════════════════════════════
 
def jobson_korkie_test(ret_pre, ret_post,
                        w_pre, w_post,
                        rf_pre, rf_post):
    """
    Memmel (2003) corrected Jobson-Korkie test.
 
    H0: SR_pre == SR_post
    H1: SR_post > SR_pre  (one-sided)
    """
    rp = ret_pre  @ w_pre
    rq = ret_post @ w_post
 
    ep = rp - rf_pre  / 252
    eq = rq - rf_post / 252
 
    T1 = len(ep)
    T2 = len(eq)
 
    sr1 = ep.mean() / ep.std()
    sr2 = eq.mean() / eq.std()
 
    se = np.sqrt(
        (1 / T1) * (1 + 0.5 * sr1**2) +
        (1 / T2) * (1 + 0.5 * sr2**2)
    )
 
    z = (sr2 - sr1) / se
    p = 1 - stats.norm.cdf(z)
 
    sr1_ann = sr1 * np.sqrt(252)
    sr2_ann = sr2 * np.sqrt(252)
 
    return {
        "SR_pre_ann":   round(float(sr1_ann), 3),
        "SR_post_ann":  round(float(sr2_ann), 3),
        "delta_SR":     round(float(sr2_ann - sr1_ann), 3),
        "z_stat":       round(float(z), 3),
        "p_value":      round(float(p), 4),
        "significant":  bool(p < 0.05),
    }
 
 
# ════════════════════════════════════════════════════════════════
# 4. TEST 2 — LEDOIT-WOLF (2008) ROBUST SHARPE TEST
# Reference: Journal of Empirical Finance 15(5): 850-859, §3.2.2
# ════════════════════════════════════════════════════════════════
 
from lw_boot_sharpe_test import lw_boot_sharpe_test
 
def jobson_korkie_boot_test(ret_pre, ret_post,
                             w_pre, w_post,
                             rf_pre, rf_post,
                             B=10000):
    """
    Ledoit-Wolf (2008) studentized circular block bootstrap test.
    Robust to fat tails and GARCH effects in daily returns.
 
    H0: SR_pre == SR_post  (two-sided)
    """
    exc_pre  = ret_pre  @ w_pre  - rf_pre  / 252
    exc_post = ret_post @ w_post - rf_post / 252
 
    return lw_boot_sharpe_test(exc_pre, exc_post, B=B)

# ════════════════════════════════════════════════════════════════
# TEST — BOOTSTRAP FRONTIER CONFIDENCE BANDS
# Reference: Font, B. (2016). "Bootstrap estimation of the
#   efficient frontier." Computational Management Science 13(4).
#
# Method (Font 2016, §2):
#   1. Demean returns to get residuals û_i = x_i − μ̂
#   2. Draw B bootstrap samples: x*_i = μ̂ + û*_i  (resample û with replacement)
#   3. For each sample, re-estimate LW covariance and re-solve the frontier QP
#   4. Interpolate each bootstrap frontier at a fixed volatility grid
#   5. 95% CI = [2.5th, 97.5th] percentile across B draws at each grid point
#
# Significance: at any vol level where the post-COVID lower band
# lies above the pre-COVID upper band, the frontier shift is
# statistically significant at that point.
#
# Runtime: ~3–4 min for B=500 (both periods combined, N=45).
# ════════════════════════════════════════════════════════════════

def bootstrap_frontier_bands(ret, B=500, n_pts=40, alpha=0.05, seed=42):
    """
    Font (2016) §2 bootstrap confidence bands for the efficient frontier.

    Parameters
    ----------
    ret    : DataFrame of daily log returns for one period
    B      : bootstrap replicates (≥200; 500 recommended)
    n_pts  : QP target-return grid points per bootstrap frontier
    alpha  : confidence level → (alpha/2, 1-alpha/2) percentile band
    seed   : RNG seed

    Returns
    -------
    vol_grid   : 1-D array of annualised vol levels (common grid)
    mean_ret   : bootstrap mean return at each grid point
    lower      : lower (alpha/2) bootstrap CI
    upper      : upper (1-alpha/2) bootstrap CI
    """
    np.random.seed(seed)
    T   = len(ret)
    mu_hat  = ret.mean().values          # (N,) daily mean
    resids  = ret.values - mu_hat        # (T, N) demeaned returns

    # ── sample frontier → defines the vol grid ──────────────────
    lw0     = LedoitWolf().fit(ret.values)
    mu_ann  = mu_hat * 252
    cov_ann = lw0.covariance_ * 252
    v0, r0  = frontier_points(mu_ann, cov_ann, n_pts=n_pts)

    if len(v0) < 3:
        print("  [warn] sample frontier has < 3 points; skipping bootstrap")
        return None, None, None, None

    # Grid spans from MVP vol up to 95th-pctile of sample frontier vols
    vol_lo   = v0.min()
    vol_hi   = float(np.percentile(v0, 95))
    vol_grid = np.linspace(vol_lo, vol_hi, 60)

    # ── bootstrap loop ──────────────────────────────────────────
    boot_rets = np.full((B, len(vol_grid)), np.nan)

    for b in range(B):
        # Step 2: bootstrap sample (Font 2016, eq. 7)
        idx    = np.random.randint(0, T, size=T)
        x_star = mu_hat + resids[idx]          # (T, N)

        # Step 3: re-estimate LW, re-solve frontier
        try:
            lw_b   = LedoitWolf().fit(x_star)
            mu_b   = x_star.mean(axis=0) * 252
            cov_b  = lw_b.covariance_    * 252
            v_b, r_b = frontier_points(mu_b, cov_b, n_pts=n_pts)
        except Exception:
            continue

        if len(v_b) < 2:
            continue

        # Step 4: interpolate onto fixed vol grid (NaN outside range)
        sort_idx         = np.argsort(v_b)
        v_b, r_b         = v_b[sort_idx], r_b[sort_idx]
        in_range         = (vol_grid >= v_b.min()) & (vol_grid <= v_b.max())
        boot_rets[b, in_range] = np.interp(vol_grid[in_range], v_b, r_b)

    # Step 5: percentile bands (Font 2016, eq. 9)
    valid_frac = (~np.isnan(boot_rets)).mean(axis=0)
    lower      = np.nanpercentile(boot_rets, 100 * alpha / 2,       axis=0)
    upper      = np.nanpercentile(boot_rets, 100 * (1 - alpha / 2), axis=0)
    mean_ret   = np.nanmean(boot_rets, axis=0)

    # Mask grid points with < 50% valid bootstrap draws (edge of frontier)
    mask             = valid_frac < 0.50
    lower[mask]      = np.nan
    upper[mask]      = np.nan
    mean_ret[mask]   = np.nan

    return vol_grid, mean_ret, lower, upper


def plot_bootstrap_frontier_comparison(ret_pre, ret_post,
                                        rf_pre, rf_post,
                                        B=500, out_path=None):
    """
    Overlay pre and post-COVID efficient frontiers with 95% bootstrap
    confidence bands (Font 2016).  Highlights vol regions where the
    frontier shift is statistically significant.

    Parameters
    ----------
    ret_pre, ret_post : DataFrames of daily log returns
    rf_pre, rf_post   : annualised risk-free rates
    B                 : bootstrap replicates
    out_path          : file path for saved figure (None → show)

    Returns
    -------
    dict with significance summary
    """
    PRE_C  = "#1D9E75"
    POST_C = "#534AB7"

    print(f"  Font (2016) bootstrap frontier bands (B={B})...")
    print("    Pre-COVID period ...", flush=True)
    vg_pre,  mr_pre,  lo_pre,  hi_pre  = bootstrap_frontier_bands(
        ret_pre,  B=B, seed=42)

    print("    Post-COVID period ...", flush=True)
    vg_post, mr_post, lo_post, hi_post = bootstrap_frontier_bands(
        ret_post, B=B, seed=123)

    if vg_pre is None or vg_post is None:
        print("  [warn] bootstrap failed; skipping plot")
        return {"shift_significant": None}

    # ── plot ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#CCCCCC")
    ax.grid(True, linestyle="--", linewidth=0.4,
            color="#DDDDDD", alpha=0.8)

    # Confidence bands
    ax.fill_between(vg_pre,  lo_pre,  hi_pre,
                    alpha=0.18, color=PRE_C,
                    label="Pre-COVID 95% CI")
    ax.fill_between(vg_post, lo_post, hi_post,
                    alpha=0.18, color=POST_C,
                    label="Post-COVID 95% CI")

    # Mean bootstrap frontiers
    ax.plot(vg_pre,  mr_pre,  color=PRE_C,  lw=2.2,
            label="Pre-COVID (bootstrap mean)")
    ax.plot(vg_post, mr_post, color=POST_C, lw=2.2,
            label="Post-COVID (bootstrap mean)")

    # ── significance: where post lower > pre upper ──────────────
    common_lo   = max(np.nanmin(vg_pre),  np.nanmin(vg_post))
    common_hi   = min(np.nanmax(vg_pre),  np.nanmax(vg_post))
    common_grid = np.linspace(common_lo, common_hi, 300)

    hi_pre_i  = np.interp(common_grid, vg_pre[~np.isnan(hi_pre)],
                           hi_pre[~np.isnan(hi_pre)])
    lo_post_i = np.interp(common_grid, vg_post[~np.isnan(lo_post)],
                           lo_post[~np.isnan(lo_post)])

    sig_mask = lo_post_i > hi_pre_i
    n_sig    = int(sig_mask.sum())

    if sig_mask.any():
        ax.fill_between(
            common_grid, hi_pre_i, lo_post_i,
            where=sig_mask, alpha=0.30,
            color="#E74C3C", zorder=3,
            label="Significant gap (5% level)"
        )

    # Formatting
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlabel("Annualised Volatility (σ)",
                  fontsize=11, color="#555555")
    ax.set_ylabel("Annualised Return (μ)",
                  fontsize=11, color="#555555")
    ax.set_title(
        "Efficient Frontier: 95% Bootstrap Confidence Bands\n"
        f"Font (2016)  |  B = {B} resamples  |  "
        f"Red band = statistically significant shift",
        fontsize=11, color="#222222", pad=10
    )
    ax.legend(fontsize=9.5, framealpha=0.7, loc="upper left")
    ax.tick_params(colors="#666666", labelsize=9)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"  [saved] {out_path}")
    else:
        plt.show()

    sig_pct = round(100 * n_sig / len(common_grid), 1)
    print(f"  Significant vol points: {n_sig}/{len(common_grid)} ({sig_pct}%)")

    return {
        "B":                      B,
        "n_significant_vol_pts":  n_sig,
        "pct_significant":        sig_pct,
        "shift_significant":      bool(sig_mask.any()),
    }

# # ════════════════════════════════════════════════════════════════
# # 5. TEST 2 — BOX'S M TEST
# # ════════════════════════════════════════════════════════════════

# def box_m_test(ret_pre, ret_post):
#     """
#     Box's M test for equality of two covariance matrices.

#     H0: Sigma_pre == Sigma_post
#     """
#     n1, p = ret_pre.shape
#     n2, _ = ret_post.shape

#     S1 = ret_pre.cov().values
#     S2 = ret_post.cov().values
#     Sp = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)

#     _, ld1 = np.linalg.slogdet(S1)
#     _, ld2 = np.linalg.slogdet(S2)
#     _, ldp = np.linalg.slogdet(Sp)

#     M = ((n1 + n2 - 2) * ldp
#          - (n1 - 1) * ld1
#          - (n2 - 1) * ld2)

#     c = (1 - (2 * p**2 + 3 * p - 1) / (6 * (p + 1)) *
#          (1 / (n1 - 1) + 1 / (n2 - 1) - 1 / (n1 + n2 - 2)))

#     chi2_stat = M * c
#     df        = p * (p + 1) / 2
#     p_val     = 1 - stats.chi2.cdf(chi2_stat, df)

#     return {
#         "M_statistic": round(float(M),         2),
#         "chi2_stat":   round(float(chi2_stat), 2),
#         "df":          int(df),
#         "p_value":     round(float(p_val),     6),
#         "significant": bool(p_val < 0.05),
#         "conclusion":  (
#             "Covariance matrices are SIGNIFICANTLY DIFFERENT "
#             "(structural change in risk structure confirmed)"
#             if p_val < 0.05 else
#             "Cannot reject equal covariance matrices at 5% level"
#         ),
#     }


# # ════════════════════════════════════════════════════════════════
# # 6. TEST 3 — COVARIANCE ESTIMATOR COMPARISON
# # ════════════════════════════════════════════════════════════════

# def plot_estimator_comparison(ret_pre, ret_post,
#                                rf_pre, rf_post, out_path):
#     PRE_C  = "#1D9E75"
#     POST_C = "#534AB7"

#     specs = [
#         ("Sample covariance",
#          "#A8D8A8", "#A8A8D8", ":",  1.2),
#         ("Ledoit-Wolf (primary)",
#          PRE_C,     POST_C,    "-",  2.2),
#     ]

#     def get_mu_cov(name, ret):
#         if name == "Sample covariance":
#             return sample_cov(ret)
#         else:
#             return ledoit_wolf(ret)[:2]

#     fig, axes = plt.subplots(1, 2, figsize=(16, 7),
#                               sharey=True)
#     fig.patch.set_facecolor("#FAFAFA")

#     summary_rows = []

#     for ax, ret, rf, title, col_idx in [
#         (axes[0], ret_pre,  rf_pre,
#          "Pre-COVID (Jan 2015–Jan 2020)", 0),
#         (axes[1], ret_post, rf_post,
#          "Post-COVID (Jul 2021–Dec 2024)", 1),
#     ]:
#         ax.set_facecolor("#FAFAFA")
#         for sp in ["top", "right"]:
#             ax.spines[sp].set_visible(False)
#         for sp in ["left", "bottom"]:
#             ax.spines[sp].set_color("#CCCCCC")
#         ax.grid(True, linestyle="--", linewidth=0.4,
#                 color="#DDDDDD", alpha=0.8)
#         ax.set_title(title, fontsize=10.5,
#                      color="#333333", pad=8)

#         for name, cp, cq, ls, lw in specs:
#             print(f"    [{title[:3]}] {name}...")
#             mu, cov = get_mu_cov(name, ret)
#             c       = cp if col_idx == 0 else cq

#             v, r    = frontier_points(mu, cov, n_pts=80)
#             ax.plot(v, r, color=c, lw=lw, ls=ls,
#                     label=name, zorder=4)

#             w_ms       = max_sharpe(mu, cov, rf)
#             r_ms, v_ms = pstats(w_ms, mu, cov)
#             sr         = (r_ms - rf) / v_ms
#             ax.scatter(v_ms, r_ms, color=c,
#                        s=85, marker="D", zorder=6)

#             summary_rows.append({
#                 "Period":    "Pre" if col_idx == 0 else "Post",
#                 "Estimator": name,
#                 "MS_vol":    round(v_ms, 4),
#                 "MS_ret":    round(r_ms, 4),
#                 "SR":        round(sr,   3),
#             })

#         ax.yaxis.set_major_formatter(
#             plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
#         ax.xaxis.set_major_formatter(
#             plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
#         ax.set_xlabel("Annualised Volatility (σ)",
#                       fontsize=10, color="#555555")
#         if col_idx == 0:
#             ax.set_ylabel("Annualised Return (μ)",
#                           fontsize=10, color="#555555")
#         ax.legend(fontsize=9, framealpha=0.65,
#                   loc="upper left")
#         ax.tick_params(colors="#666666", labelsize=9)

#     fig.suptitle(
#         "Robustness to Covariance Estimator Choice\n"
#         "All three methods should show the same "
#         "directional shift if the result is genuine",
#         fontsize=12, fontweight="500",
#         color="#222222", y=1.01
#     )
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=180, bbox_inches="tight")
#     plt.close(fig)
#     print(f"  [saved] {out_path}")
#     return pd.DataFrame(summary_rows)


# # ════════════════════════════════════════════════════════════════
# # 7. TEST 4 — TRANSACTION COST ANALYSIS (period-specific costs)
# # ════════════════════════════════════════════════════════════════

# def transaction_cost_analysis(ret_pre, ret_post,
#                                rf_pre, rf_post):
#     mu_pre,  cov_pre,  _ = ledoit_wolf(ret_pre)
#     mu_post, cov_post, _ = ledoit_wolf(ret_post)

#     w_pre  = max_sharpe(mu_pre,  cov_pre,  rf_pre)
#     w_post = max_sharpe(mu_post, cov_post, rf_post)

#     r_pre,  v_pre  = pstats(w_pre,  mu_pre,  cov_pre)
#     r_post, v_post = pstats(w_post, mu_post, cov_post)

#     sr_pre_gross  = (r_pre  - rf_pre)  / v_pre
#     sr_post_gross = (r_post - rf_post) / v_post

#     turnover = float(np.sum(np.abs(w_post - w_pre)))

#     cost_pre  = COSTS["pre"]["ROUND_TRIP"]
#     cost_post = COSTS["post"]["ROUND_TRIP"]

#     rows = []
#     for freq_label, rebal_per_year in [
#         ("Annual",      1),
#         ("Semi-annual", 2),
#         ("Quarterly",   4),
#         ("Monthly",    12),
#     ]:
#         drag_pre  = turnover * cost_pre  * rebal_per_year
#         drag_post = turnover * cost_post * rebal_per_year

#         sr_pre_net  = (r_pre  - rf_pre  - drag_pre)  / v_pre
#         sr_post_net = (r_post - rf_post - drag_post) / v_post

#         rows.append({
#             "Rebalancing":          freq_label,
#             "Cost pre (%)":         round(drag_pre  * 100, 3),
#             "Cost post (%)":        round(drag_post * 100, 3),
#             "SR pre (gross)":       round(sr_pre_gross,  3),
#             "SR post (gross)":      round(sr_post_gross, 3),
#             "SR pre (net)":         round(sr_pre_net,    3),
#             "SR post (net)":        round(sr_post_net,   3),
#             "Delta SR (net)":       round(sr_post_net - sr_pre_net, 3),
#             "Improvement survives": bool(sr_post_net > sr_pre_net),
#         })

#     df = pd.DataFrame(rows)

#     fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
#     fig.patch.set_facecolor("#FAFAFA")

#     ax = axes[0]
#     ax.set_facecolor("#FAFAFA")
#     for sp in ["top", "right"]:
#         ax.spines[sp].set_visible(False)
#     for sp in ["left", "bottom"]:
#         ax.spines[sp].set_color("#CCCCCC")
#     ax.grid(True, linestyle="--", linewidth=0.4,
#             color="#DDDDDD", alpha=0.8, axis="y")

#     x     = np.arange(len(rows))
#     width = 0.2
#     labels = [r["Rebalancing"] for r in rows]

#     ax.bar(x - 1.5*width,
#            [r["SR pre (gross)"]  for r in rows],
#            width, label="Pre gross",  color="#A8D8C8",
#            alpha=0.9)
#     ax.bar(x - 0.5*width,
#            [r["SR pre (net)"]    for r in rows],
#            width, label="Pre net",    color="#1D9E75",
#            alpha=0.9)
#     ax.bar(x + 0.5*width,
#            [r["SR post (gross)"] for r in rows],
#            width, label="Post gross", color="#C8C8F0",
#            alpha=0.9)
#     ax.bar(x + 1.5*width,
#            [r["SR post (net)"]   for r in rows],
#            width, label="Post net",   color="#534AB7",
#            alpha=0.9)

#     ax.set_xticks(x)
#     ax.set_xticklabels(labels, fontsize=9.5)
#     ax.set_ylabel("Sharpe Ratio", fontsize=10,
#                   color="#555555")
#     ax.set_title(
#         "Gross vs Net Sharpe by Rebalancing Frequency\n"
#         f"Pre round-trip: {cost_pre*100:.2f}%  |  "
#         f"Post round-trip: {cost_post*100:.2f}%",
#         fontsize=10, color="#222222"
#     )
#     ax.legend(fontsize=9, framealpha=0.7)
#     ax.axhline(0, color="#888888", lw=0.8)
#     ax.tick_params(colors="#666666", labelsize=9)

#     ax2 = axes[1]
#     ax2.set_facecolor("#FAFAFA")
#     for sp in ["top", "right"]:
#         ax2.spines[sp].set_visible(False)
#     for sp in ["left", "bottom"]:
#         ax2.spines[sp].set_color("#CCCCCC")
#     ax2.grid(True, linestyle="--", linewidth=0.4,
#              color="#DDDDDD", alpha=0.8, axis="y")

#     width2 = 0.3
#     ax2.bar(x - width2/2,
#             [r["Cost pre (%)"]  for r in rows],
#             width2, label="Pre-COVID cost",
#             color="#E8A838", alpha=0.9)
#     ax2.bar(x + width2/2,
#             [r["Cost post (%)"] for r in rows],
#             width2, label="Post-COVID cost",
#             color="#534AB7", alpha=0.9)
#     ax2.set_xticks(x)
#     ax2.set_xticklabels(labels, fontsize=9.5)
#     ax2.set_ylabel("Annual Cost Drag (%)", fontsize=10,
#                    color="#555555")
#     ax2.set_title(
#         "Annual Cost Drag by Rebalancing Frequency\n"
#         "Pre-COVID costs higher due to traditional brokers",
#         fontsize=10, color="#222222"
#     )
#     ax2.legend(fontsize=9, framealpha=0.7)
#     ax2.tick_params(colors="#666666", labelsize=9)

#     fig.suptitle(
#         f"Transaction Cost Analysis  |  "
#         f"One-way turnover: {turnover*100:.1f}%  |  "
#         f"STT + brokerage + impact (period-specific)",
#         fontsize=11, fontweight="500",
#         color="#222222", y=1.01
#     )
#     fig.tight_layout()
#     out = "Results/Robustness/transaction_costs.png"
#     fig.savefig(out, dpi=180, bbox_inches="tight")
#     plt.close(fig)
#     print(f"  [saved] {out}")

#     return df, turnover, cost_pre, cost_post


# # ════════════════════════════════════════════════════════════════
# # 8. TEST 5 — SECTOR WEIGHT CHECK
# # ════════════════════════════════════════════════════════════════

# def sector_weight_check(ret_pre, ret_post, out_path, rf_pre, rf_post):
#     tickers = list(ret_pre.columns)

#     sector_counts = {}
#     for t in tickers:
#         s = SECTOR_MAP.get(t, "Others")
#         sector_counts[s] = sector_counts.get(s, 0) + 1
#     total = len(tickers)
#     universe_weights = {s: c / total
#                         for s, c in sector_counts.items()}

#     sectors = sorted(set(
#         list(NIFTY50_SECTOR_WEIGHTS.keys()) +
#         list(universe_weights.keys())
#     ))
#     nifty_w = [NIFTY50_SECTOR_WEIGHTS.get(s, 0) for s in sectors]
#     univ_w  = [universe_weights.get(s, 0)        for s in sectors]

#     mu_pre,  cov_pre,  _ = ledoit_wolf(ret_pre)
#     mu_post, cov_post, _ = ledoit_wolf(ret_post)
#     w_pre  = max_sharpe(mu_pre,  cov_pre,  rf_pre)
#     w_post = max_sharpe(mu_post, cov_post, rf_post)

#     def sector_weights_from_w(w, tickers):
#         sw = {}
#         for wi, t in zip(w, tickers):
#             s = SECTOR_MAP.get(t, "Others")
#             sw[s] = sw.get(s, 0) + wi
#         return sw

#     sw_pre  = sector_weights_from_w(w_pre,  tickers)
#     sw_post = sector_weights_from_w(w_post, tickers)
#     tp_pre  = [sw_pre.get(s,  0) for s in sectors]
#     tp_post = [sw_post.get(s, 0) for s in sectors]

#     fig, axes = plt.subplots(1, 2, figsize=(16, 6))
#     fig.patch.set_facecolor("#FAFAFA")

#     x     = np.arange(len(sectors))
#     width = 0.35

#     ax = axes[0]
#     ax.set_facecolor("#FAFAFA")
#     for sp in ["top", "right"]:
#         ax.spines[sp].set_visible(False)
#     for sp in ["left", "bottom"]:
#         ax.spines[sp].set_color("#CCCCCC")
#     ax.grid(True, linestyle="--", linewidth=0.4,
#             color="#DDDDDD", alpha=0.8, axis="x")

#     ax.barh(x - width/2, nifty_w, width,
#             label="Actual Nifty 50",
#             color="#E8A838", alpha=0.85)
#     ax.barh(x + width/2, univ_w,  width,
#             label="Study universe (equal-weight)",
#             color="#534AB7", alpha=0.85)
#     ax.set_yticks(x)
#     ax.set_yticklabels(sectors, fontsize=9.5)
#     ax.xaxis.set_major_formatter(
#         plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
#     ax.set_xlabel("Sector Weight", fontsize=10,
#                   color="#555555")
#     ax.set_title(
#         "Universe vs Actual Nifty 50 Sector Weights\n"
#         "Checks representativeness of stock selection",
#         fontsize=10.5, color="#222222"
#     )
#     ax.legend(fontsize=9.5, framealpha=0.7)
#     ax.tick_params(colors="#666666", labelsize=9)

#     ax2 = axes[1]
#     ax2.set_facecolor("#FAFAFA")
#     for sp in ["top", "right"]:
#         ax2.spines[sp].set_visible(False)
#     for sp in ["left", "bottom"]:
#         ax2.spines[sp].set_color("#CCCCCC")
#     ax2.grid(True, linestyle="--", linewidth=0.4,
#              color="#DDDDDD", alpha=0.8, axis="x")

#     ax2.barh(x - width/2, tp_pre,  width,
#              label="Pre-COVID tangency",
#              color="#1D9E75", alpha=0.85)
#     ax2.barh(x + width/2, tp_post, width,
#              label="Post-COVID tangency",
#              color="#534AB7", alpha=0.85)
#     ax2.set_yticks(x)
#     ax2.set_yticklabels(sectors, fontsize=9.5)
#     ax2.xaxis.set_major_formatter(
#         plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
#     ax2.set_xlabel("Portfolio Weight", fontsize=10,
#                    color="#555555")
#     ax2.set_title(
#         "Tangency Portfolio Sector Weights\n"
#         "Pre-COVID vs Post-COVID rotation",
#         fontsize=10.5, color="#222222"
#     )
#     ax2.legend(fontsize=9.5, framealpha=0.7)
#     ax2.tick_params(colors="#666666", labelsize=9)

#     fig.suptitle(
#         "Sector Composition Analysis",
#         fontsize=12, fontweight="500",
#         color="#222222", y=1.01
#     )
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=180, bbox_inches="tight")
#     plt.close(fig)
#     print(f"  [saved] {out_path}")

#     return pd.DataFrame({
#         "Sector":      sectors,
#         "Nifty50_wt":  nifty_w,
#         "Universe_wt": univ_w,
#         "TP_pre_wt":   tp_pre,
#         "TP_post_wt":  tp_post,
#     })


# # ════════════════════════════════════════════════════════════════
# # 9. TEST 6 — BANKING SUB-SECTOR DECOMPOSITION
# # ════════════════════════════════════════════════════════════════

# def banking_decomposition(ret_pre, ret_post, out_path):
#     def get_available(tickers, ret):
#         return [t for t in tickers if t in ret.columns]

#     retail_pre  = get_available(RETAIL_BANKS,    ret_pre)
#     retail_post = get_available(RETAIL_BANKS,    ret_post)
#     whole_pre   = get_available(WHOLESALE_BANKS, ret_pre)
#     whole_post  = get_available(WHOLESALE_BANKS, ret_post)

#     def ann_ret(ret, tickers):
#         if not tickers:
#             return {}
#         return (ret[tickers].mean() * 252).to_dict()

#     rr_pre  = ann_ret(ret_pre,  retail_pre)
#     rr_post = ann_ret(ret_post, retail_post)
#     rw_pre  = ann_ret(ret_pre,  whole_pre)
#     rw_post = ann_ret(ret_post, whole_post)

#     all_banks = list(set(retail_pre + whole_pre))
#     bank_pre  = {t: ret_pre[t].mean()  * 252
#                  for t in all_banks if t in ret_pre.columns}
#     bank_post = {t: ret_post[t].mean() * 252
#                  for t in all_banks if t in ret_post.columns}
#     common    = [t for t in all_banks
#                  if t in bank_pre and t in bank_post]
#     labels    = [t.replace(".NS", "") for t in common]

#     pre_vals  = [bank_pre[t]  for t in common]
#     post_vals = [bank_post[t] for t in common]

#     fig, axes = plt.subplots(1, 2, figsize=(15, 6))
#     fig.patch.set_facecolor("#FAFAFA")

#     ax = axes[0]
#     ax.set_facecolor("#FAFAFA")
#     for sp in ["top", "right"]:
#         ax.spines[sp].set_visible(False)
#     for sp in ["left", "bottom"]:
#         ax.spines[sp].set_color("#CCCCCC")
#     ax.grid(True, linestyle="--", linewidth=0.4,
#             color="#DDDDDD", alpha=0.8, axis="x")

#     x     = np.arange(len(labels))
#     width = 0.35
#     ax.barh(x - width/2, pre_vals,  width,
#             label="Pre-COVID",  color="#1D9E75", alpha=0.85)
#     ax.barh(x + width/2, post_vals, width,
#             label="Post-COVID", color="#534AB7", alpha=0.85)
#     ax.set_yticks(x)
#     ax.set_yticklabels(labels, fontsize=10)
#     ax.xaxis.set_major_formatter(
#         plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
#     ax.axvline(0, color="#888888", lw=0.8)
#     ax.set_xlabel("Annualised Return", fontsize=10,
#                   color="#555555")
#     ax.set_title(
#         "Individual Bank Returns: Pre vs Post\n"
#         "Red label = Retail  |  Amber label = Wholesale",
#         fontsize=10.5, color="#222222"
#     )
#     ax.legend(fontsize=9.5, framealpha=0.7)
#     ax.tick_params(colors="#666666", labelsize=9)

#     for i, t in enumerate(common):
#         subtype = ("Retail"    if t in RETAIL_BANKS
#                    else "Wholesale")
#         col     = ("#C0392B"   if t in RETAIL_BANKS
#                    else "#854F0B")
#         ax.text(-0.005, i, subtype,
#                 ha="right", va="center",
#                 fontsize=7.5, color=col,
#                 fontweight="500")

#     ax2 = axes[1]
#     ax2.set_facecolor("#FAFAFA")
#     for sp in ["top", "right"]:
#         ax2.spines[sp].set_visible(False)
#     for sp in ["left", "bottom"]:
#         ax2.spines[sp].set_color("#CCCCCC")
#     ax2.grid(True, linestyle="--", linewidth=0.4,
#              color="#DDDDDD", alpha=0.8, axis="y")

#     avg_retail_pre   = (np.mean([rr_pre.get(t, 0)
#                                  for t in retail_pre])
#                         if retail_pre else 0)
#     avg_retail_post  = (np.mean([rr_post.get(t, 0)
#                                  for t in retail_post])
#                         if retail_post else 0)
#     avg_whole_pre    = (np.mean([rw_pre.get(t, 0)
#                                  for t in whole_pre])
#                         if whole_pre else 0)
#     avg_whole_post   = (np.mean([rw_post.get(t, 0)
#                                  for t in whole_post])
#                         if whole_post else 0)

#     sub_labels = [
#         "Retail Banks\n(HDFC, Kotak, Axis,\nICICI, SBI)",
#         "Wholesale Banks\n(PNB, BoB, IndusInd)"
#     ]
#     x2    = np.arange(2)
#     w2    = 0.3
#     ax2.bar(x2 - w2/2,
#             [avg_retail_pre,  avg_whole_pre],
#             w2, label="Pre-COVID",
#             color="#1D9E75", alpha=0.85)
#     ax2.bar(x2 + w2/2,
#             [avg_retail_post, avg_whole_post],
#             w2, label="Post-COVID",
#             color="#534AB7", alpha=0.85)
#     ax2.set_xticks(x2)
#     ax2.set_xticklabels(sub_labels, fontsize=10)
#     ax2.yaxis.set_major_formatter(
#         plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
#     ax2.axhline(0, color="#888888", lw=0.8)
#     ax2.set_ylabel("Average Annualised Return",
#                    fontsize=10, color="#555555")
#     ax2.set_title(
#         "Average Return by Banking Sub-sector\n"
#         "Retail vs Wholesale banks",
#         fontsize=10.5, color="#222222"
#     )
#     ax2.legend(fontsize=9.5, framealpha=0.7)
#     ax2.tick_params(colors="#666666", labelsize=9)

#     fig.suptitle(
#         "Banking Sector Decomposition: "
#         "Retail vs Wholesale Banks",
#         fontsize=12, fontweight="500",
#         color="#222222", y=1.01
#     )
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=180, bbox_inches="tight")
#     plt.close(fig)
#     print(f"  [saved] {out_path}")

#     return {
#         "avg_retail_pre":    float(avg_retail_pre),
#         "avg_retail_post":   float(avg_retail_post),
#         "avg_wholesale_pre": float(avg_whole_pre),
#         "avg_wholesale_post":float(avg_whole_post),
#     }


# ════════════════════════════════════════════════════════════════
# 10. PRINT SUMMARY REPORT
# ════════════════════════════════════════════════════════════════

def print_summary(jk, boxm, tc_df, banking, sector_df):
    sep = "=" * 60
    print(f"\n{sep}")
    print("  ROBUSTNESS TEST SUMMARY REPORT")
    print(sep)

    print("\n── Test 1: Jobson-Korkie ──")
    print(f"  SR pre  (annualised): {jk['SR_pre_ann']}")
    print(f"  SR post (annualised): {jk['SR_post_ann']}")
    print(f"  Delta SR:             {jk['delta_SR']}")
    print(f"  z-statistic:          {jk['z_stat']}")
    print(f"  p-value:              {jk['p_value']}")
    print(f"  Significant at 5%:    {jk['significant']}")
    if jk['significant']:
        print("  --> Sharpe improvement is statistically "
              "significant.")
    else:
        print("  --> Cannot reject equal Sharpe ratios "
              "at 5% level.")

    print("\n── Test 2: Box's M (Covariance Equality) ──")
    print(f"  M statistic:        {boxm['M_statistic']}")
    print(f"  Chi2 statistic:     {boxm['chi2_stat']}")
    print(f"  Degrees of freedom: {boxm['df']}")
    print(f"  p-value:            {boxm['p_value']}")
    print(f"  --> {boxm['conclusion']}")

    print("\n── Test 3: Covariance Estimator Comparison ──")
    print("  See Results/Robustness/estimator_comparison.png")
    print("  See Results/Robustness/estimator_comparison.csv")

    print("\n── Test 4: Transaction Cost Analysis ──")
    print(tc_df.to_string(index=False))

    print("\n── Test 5: Sector Weight Check ──")
    print("  See Results/Robustness/sector_weights.png")
    top_diff = sector_df.copy()
    top_diff["gap"] = abs(
        top_diff["Nifty50_wt"] - top_diff["Universe_wt"])
    top_diff = top_diff.sort_values("gap", ascending=False)
    print("  Largest gaps vs Nifty 50:")
    for _, row in top_diff.head(3).iterrows():
        print(f"    {row['Sector']:12s}  "
              f"Nifty={row['Nifty50_wt']:.1%}  "
              f"Universe={row['Universe_wt']:.1%}  "
              f"Gap={row['gap']:.1%}")

    print("\n── Test 6: Banking Decomposition ──")
    print(f"  Retail banks    pre:  {banking['avg_retail_pre']:.1%}")
    print(f"  Retail banks    post: {banking['avg_retail_post']:.1%}")
    print(f"  Wholesale banks pre:  {banking['avg_wholesale_pre']:.1%}")
    print(f"  Wholesale banks post: {banking['avg_wholesale_post']:.1%}")

    print(f"\n{sep}")
    print("  All charts saved to Results/Robustness/")
    print(f"{sep}\n")


# ════════════════════════════════════════════════════════════════
# 11. MAIN
# ════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  Robustness & Statistical Tests Pipeline")
    print("  Pre-COVID: Jan 2015 – Jan 2020")
    print("  Post-COVID: Jul 2021 – Dec 2024 (crisis excluded)")
    print("=" * 60 + "\n")

    print("Loading RBI T-bill risk-free rates...")
    RF_PRE, RF_POST = load_risk_free_rates()

    log_ret = to_log_returns(load_prices())

    ret_pre  = slice_period(log_ret, *P["a_pre"])
    ret_post = slice_period(log_ret, *P["a_post"])
    shared   = ret_pre.columns.intersection(ret_post.columns)
    ret_pre  = ret_pre[shared]
    ret_post = ret_post[shared]

    print(f"Shared universe : {len(shared)} tickers")
    print(f"Pre obs         : {len(ret_pre)}")
    print(f"Post obs        : {len(ret_post)}\n")

    mu_pre,  cov_pre,  s_pre  = ledoit_wolf(ret_pre)
    mu_post, cov_post, s_post = ledoit_wolf(ret_post)
    print(f"LW shrinkage pre:  {s_pre:.3f}")
    print(f"LW shrinkage post: {s_post:.3f}\n")

    w_pre  = max_sharpe(mu_pre,  cov_pre,  RF_PRE)
    w_post = max_sharpe(mu_post, cov_post, RF_POST)

    # ── Test 1: Jobson-Korkie ─────────────────────
    print("── Test 1: Jobson-Korkie test ──")
    jk = jobson_korkie_test(
        ret_pre, ret_post,
        w_pre, w_post,
        RF_PRE, RF_POST
    )
    print(f"  SR pre={jk['SR_pre_ann']}  "
          f"SR post={jk['SR_post_ann']}  "
          f"z={jk['z_stat']}  "
          f"p={jk['p_value']}  "
          f"sig={jk['significant']}")

    # ── Test 1b: LW Bootstrap Sharpe test ────────
    print("\n── Test 1b: Ledoit-Wolf Bootstrap Sharpe test ──")
    jk_boot = jobson_korkie_boot_test(
        ret_pre, ret_post,
        w_pre, w_post,
        RF_PRE, RF_POST
    )
    print(f"  SR pre={jk_boot['SR_pre_ann']}  "
          f"SR post={jk_boot['SR_post_ann']}  "
          f"d_obs={jk_boot['d_obs']}  "
          f"p={jk_boot['p_value']}  "
          f"sig={jk_boot['significant']}")
    
    # ── Bootstrap frontier bands (Font 2016) ──────────────────────
    print("\n── Bootstrap Frontier Confidence Bands (Font 2016) ──")
    boot_result = plot_bootstrap_frontier_comparison(
        ret_pre, ret_post, RF_PRE, RF_POST,
        B=500,
        out_path="Results/Robustness/bootstrap_frontier_bands.png"
    )
 

    # # ── Test 2: Box's M ───────────────────────────
    # print("\n── Test 2: Box's M test ──")
    # boxm = box_m_test(ret_pre, ret_post)
    # print(f"  chi2={boxm['chi2_stat']}  "
    #       f"p={boxm['p_value']}  "
    #       f"sig={boxm['significant']}")
    # print(f"  {boxm['conclusion']}")

    # # ── Test 3: Estimator comparison ──────────────
    # print("\n── Test 3: Covariance estimator comparison ──")
    # est_df = plot_estimator_comparison(
    #     ret_pre, ret_post, RF_PRE, RF_POST,
    #     out_path="Results/Robustness/estimator_comparison.png"
    # )
    # est_df.to_csv(
    #     "Results/Robustness/estimator_comparison.csv",
    #     index=False
    # )
    # print("  Estimator SR summary:")
    # print(est_df.to_string(index=False))

    # # ── Test 4: Transaction costs ─────────────────
    # print("\n── Test 4: Transaction cost analysis ──")
    # tc_df, turnover, cost_pre, cost_post = \
    #     transaction_cost_analysis(
    #         ret_pre, ret_post, RF_PRE, RF_POST
    #     )
    # tc_df.to_csv(
    #     "Results/Robustness/transaction_costs.csv",
    #     index=False
    # )
    # print(f"  One-way turnover:   {turnover*100:.1f}%")
    # print(f"  Pre round-trip:     {cost_pre*100:.2f}%")
    # print(f"  Post round-trip:    {cost_post*100:.2f}%")

    # # ── Test 5: Sector weight check ───────────────
    # print("\n── Test 5: Sector weight check ──")
    # sector_df = sector_weight_check(
    #     ret_pre, ret_post,
    #     out_path="Results/Robustness/sector_weights.png",
    #     rf_pre=RF_PRE, rf_post=RF_POST
    # )
    # sector_df.to_csv(
    #     "Results/Robustness/sector_weights.csv",
    #     index=False
    # )

    # # ── Test 6: Banking decomposition ────────────
    # print("\n── Test 6: Banking sub-sector decomposition ──")
    # banking = banking_decomposition(
    #     ret_pre, ret_post,
    #     out_path="Results/Robustness/banking_decomposition.png"
    # )

    # # ── Summary ───────────────────────────────────
    # print_summary(jk, boxm, tc_df, banking, sector_df)

    summary = {
        "JK_SR_pre":         jk["SR_pre_ann"],
        "JK_SR_post":        jk["SR_post_ann"],
        "JK_delta_SR":       jk["delta_SR"],
        "JK_z":              jk["z_stat"],
        "JK_p":              jk["p_value"],
        "JK_significant":    jk["significant"],
        "JK_boot_d_obs":     jk_boot["d_obs"],
        "JK_boot_p":         jk_boot["p_value"],
        "JK_boot_significant": jk_boot["significant"],
        # "BoxM_chi2":         boxm["chi2_stat"],
        # "BoxM_p":            boxm["p_value"],
        # "BoxM_significant":  boxm["significant"],
        # "Turnover_pct":      round(turnover * 100, 2),
        # "Cost_pre_pct":      round(cost_pre  * 100, 3),
        # "Cost_post_pct":     round(cost_post * 100, 3),
        "LW_shrinkage_pre":  round(s_pre,  3),
        "LW_shrinkage_post": round(s_post, 3),
        "boot_B":                    boot_result["B"],
        "boot_n_sig_vol_pts":        boot_result["n_significant_vol_pts"],
        "boot_pct_significant":      boot_result["pct_significant"],
        "boot_shift_significant":    boot_result["shift_significant"]
        # "Retail_pre":        round(banking["avg_retail_pre"],    4),
        # "Retail_post":       round(banking["avg_retail_post"],   4),
        # "Wholesale_pre":     round(banking["avg_wholesale_pre"], 4),
        # "Wholesale_post":    round(banking["avg_wholesale_post"],4),
    }
    pd.DataFrame([summary]).to_csv(
        "Results/Robustness/test_summary.csv",
        index=False
    )
    print("Saved: Results/Robustness/test_summary.csv")
    print("\nAll done.\n")


if __name__ == "__main__":
    main()
