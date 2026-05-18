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
os.makedirs("../results/robustness", exist_ok=True)

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
        '../data/raw/rbi_tbill_91day_yields.xlsx',
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
    prices = pd.read_csv("../data/processed/prices_daily.csv", index_col=0, parse_dates=True)
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


def style_ax(ax):
    ax.set_facecolor("#FAFAFA")
    for sp in ["top", "right"]:   ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]: ax.spines[sp].set_color("#CCCCCC")
    ax.grid(True, linestyle="--", linewidth=0.4, color="#DDDDDD", alpha=0.9)
    ax.tick_params(colors="#666666", labelsize=9)


def hhi(w):
    return float(np.sum(w ** 2))


def _student_t_var_cvar(r, level, N=100_000, seed=42):
    """
    Student-t Monte Carlo VaR and CVaR at `level` confidence (e.g. 0.95).

    Steps:
      1. Fit Student-t (df, loc, scale) to the return series via MLE.
         The fitted df captures tail-fatness; Indian equities typically
         yield df ≈ 3-6, well below the Gaussian limit.
      2. Simulate N draws from the fitted distribution.
      3. VaR  = -(level-th lower percentile of simulated returns)
      4. CVaR = -(mean of simulated returns below the VaR cutoff)

    Returns daily figures as positive loss fractions, plus fitted df.
    """
    df, loc, scale = stats.t.fit(r)
    sim    = stats.t.rvs(df, loc=loc, scale=scale, size=N,
                         random_state=seed)
    cutoff = np.percentile(sim, (1 - level) * 100)
    var_   = float(-cutoff)
    cvar_  = float(-sim[sim <= cutoff].mean())
    return var_, cvar_, float(df)


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

    _CACHE = "../results/robustness/bootstrap_bands_cache.npz"
    if os.path.exists(_CACHE):
        print("  Loading cached bootstrap bands...")
        _c = np.load(_CACHE)
        vg_pre,  mr_pre,  lo_pre,  hi_pre  = _c["vg_pre"],  _c["mr_pre"],  _c["lo_pre"],  _c["hi_pre"]
        vg_post, mr_post, lo_post, hi_post = _c["vg_post"], _c["mr_post"], _c["lo_post"], _c["hi_post"]
    else:
        print(f"  Font (2016) bootstrap frontier bands (B={B})...")
        print("    Pre-COVID period ...", flush=True)
        vg_pre,  mr_pre,  lo_pre,  hi_pre  = bootstrap_frontier_bands(ret_pre,  B=B, seed=42)
        print("    Post-COVID period ...", flush=True)
        vg_post, mr_post, lo_post, hi_post = bootstrap_frontier_bands(ret_post, B=B, seed=123)
        if vg_pre is not None and vg_post is not None:
            np.savez(_CACHE,
                     vg_pre=vg_pre,   mr_pre=mr_pre,   lo_pre=lo_pre,   hi_pre=hi_pre,
                     vg_post=vg_post, mr_post=mr_post, lo_post=lo_post, hi_post=hi_post)
            print(f"  [saved cache] {_CACHE}")

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
# # 7. TEST 4 — TRANSACTION COST ANALYSIS (period-specific costs)
# # ════════════════════════════════════════════════════════════════

def transaction_cost_analysis(ret_pre, ret_post,
                              rf_pre, rf_post):

    # Expected returns and covariance matrices
    mu_pre,  cov_pre,  _ = ledoit_wolf(ret_pre)
    mu_post, cov_post, _ = ledoit_wolf(ret_post)

    # Tangency portfolios
    w_pre  = max_sharpe(mu_pre,  cov_pre,  rf_pre)
    w_post = max_sharpe(mu_post, cov_post, rf_post)

    # Portfolio statistics
    r_pre,  v_pre  = pstats(w_pre,  mu_pre,  cov_pre)
    r_post, v_post = pstats(w_post, mu_post, cov_post)

    # Gross Sharpe ratios
    sr_pre_gross  = (r_pre  - rf_pre)  / v_pre
    sr_post_gross = (r_post - rf_post) / v_post

    # One-time portfolio migration turnover
    turnover = 0.5 * float(
        np.sum(np.abs(w_post - w_pre))
    )

    # Round-trip implementation costs
    cost_pre  = COSTS["pre"]["ROUND_TRIP"]
    cost_post = COSTS["post"]["ROUND_TRIP"]

    # One-time implementation drag
    drag_pre  = turnover * cost_pre
    drag_post = turnover * cost_post

    # Net Sharpe ratios
    sr_pre_net  = (
        r_pre - rf_pre - drag_pre
    ) / v_pre

    sr_post_net = (
        r_post - rf_post - drag_post
    ) / v_post

    # Summary table
    df = pd.DataFrame([{
        "Turnover (%)":
            round(turnover * 100, 2),

        "Pre cost drag (%)":
            round(drag_pre * 100, 3),

        "Post cost drag (%)":
            round(drag_post * 100, 3),

        "SR pre gross":
            round(sr_pre_gross, 3),

        "SR post gross":
            round(sr_post_gross, 3),

        "SR pre net":
            round(sr_pre_net, 3),

        "SR post net":
            round(sr_post_net, 3),

        "Delta SR gross":
            round(sr_post_gross - sr_pre_gross, 3),

        "Delta SR net":
            round(sr_post_net - sr_pre_net, 3),

        "Improvement survives":
            bool(sr_post_net > sr_pre_net),
    }])

    return df


# ════════════════════════════════════════════════════════════════
# 8. TEST 5 — SECTOR WEIGHT CHECK
# ════════════════════════════════════════════════════════════════

def sector_weight_check(ret_pre, ret_post, out_path, rf_pre, rf_post):
    tickers = list(ret_pre.columns)

    sector_counts = {}
    for t in tickers:
        s = SECTOR_MAP.get(t, "Others")
        sector_counts[s] = sector_counts.get(s, 0) + 1
    total = len(tickers)
    universe_weights = {s: c / total
                        for s, c in sector_counts.items()}

    sectors = sorted(set(
        list(NIFTY50_SECTOR_WEIGHTS.keys()) +
        list(universe_weights.keys())
    ))
    nifty_w = [NIFTY50_SECTOR_WEIGHTS.get(s, 0) for s in sectors]
    univ_w  = [universe_weights.get(s, 0)        for s in sectors]

    mu_pre,  cov_pre,  _ = ledoit_wolf(ret_pre)
    mu_post, cov_post, _ = ledoit_wolf(ret_post)
    w_pre  = max_sharpe(mu_pre,  cov_pre,  rf_pre)
    w_post = max_sharpe(mu_post, cov_post, rf_post)

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

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#FAFAFA")

    x     = np.arange(len(sectors))
    width = 0.35

    ax = axes[0]
    ax.set_facecolor("#FAFAFA")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#CCCCCC")
    ax.grid(True, linestyle="--", linewidth=0.4,
            color="#DDDDDD", alpha=0.8, axis="x")

    ax.barh(x - width/2, nifty_w, width,
            label="Actual Nifty 50",
            color="#E8A838", alpha=0.85)
    ax.barh(x + width/2, univ_w,  width,
            label="Study universe (equal-weight)",
            color="#534AB7", alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(sectors, fontsize=9.5)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Sector Weight", fontsize=10,
                  color="#555555")
    ax.set_title(
        "Universe vs Actual Nifty 50 Sector Weights\n"
        "Checks representativeness of stock selection",
        fontsize=10.5, color="#222222"
    )
    ax.legend(fontsize=9.5, framealpha=0.7)
    ax.tick_params(colors="#666666", labelsize=9)

    ax2 = axes[1]
    ax2.set_facecolor("#FAFAFA")
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax2.spines[sp].set_color("#CCCCCC")
    ax2.grid(True, linestyle="--", linewidth=0.4,
             color="#DDDDDD", alpha=0.8, axis="x")

    ax2.barh(x - width/2, tp_pre,  width,
             label="Pre-COVID tangency",
             color="#1D9E75", alpha=0.85)
    ax2.barh(x + width/2, tp_post, width,
             label="Post-COVID tangency",
             color="#534AB7", alpha=0.85)
    ax2.set_yticks(x)
    ax2.set_yticklabels(sectors, fontsize=9.5)
    ax2.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.set_xlabel("Portfolio Weight", fontsize=10,
                   color="#555555")
    ax2.set_title(
        "Tangency Portfolio Sector Weights\n"
        "Pre-COVID vs Post-COVID rotation",
        fontsize=10.5, color="#222222"
    )
    ax2.legend(fontsize=9.5, framealpha=0.7)
    ax2.tick_params(colors="#666666", labelsize=9)

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
        "Sector":      sectors,
        "Nifty50_wt":  nifty_w,
        "Universe_wt": univ_w,
        "TP_pre_wt":   tp_pre,
        "TP_post_wt":  tp_post,
    })


# ════════════════════════════════════════════════════════════════
# 9. FRONTIER SHIFT DECOMPOSITION (μ vs Σ)
# ════════════════════════════════════════════════════════════════

def plot_frontier_decomposition(ret_pre, ret_post, rf_pre, rf_post,
                                out_path=None):
    mu_pre,  cov_pre,  _ = ledoit_wolf(ret_pre)
    mu_post, cov_post, _ = ledoit_wolf(ret_post)

    scenarios = {
        "Pre-COVID (μ_pre, Σ_pre)":       (mu_pre,  cov_pre,  rf_pre),
        "Only μ updated (μ_post, Σ_pre)": (mu_post, cov_pre,  rf_pre),
        "Only Σ updated (μ_pre, Σ_post)": (mu_pre,  cov_post, rf_post),
        "Post-COVID (μ_post, Σ_post)":    (mu_post, cov_post, rf_post),
    }
    colors = ["#1D9E75", "#E0A020", "#9B59B6", "#534AB7"]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#FAFAFA")
    style_ax(ax)

    srs, vols_tp, rets_tp = [], [], []
    for (label, (mu, cov, rf)), color in zip(scenarios.items(), colors):
        v, r = frontier_points(mu, cov, n_pts=150)
        w_tp = max_sharpe(mu, cov, rf)
        r_tp, v_tp = pstats(w_tp, mu, cov)
        sr = (r_tp - rf) / v_tp
        srs.append(sr); vols_tp.append(v_tp); rets_tp.append(r_tp)
        ax.plot(v, r, color=color, lw=2.2, label=f"{label}  (SR={sr:.2f})")
        ax.scatter(v_tp, r_tp, color=color, s=120, zorder=6,
                   marker="*", edgecolors="white", linewidths=0.5)

    ax.annotate("", xy=(vols_tp[1], rets_tp[1]), xytext=(vols_tp[0], rets_tp[0]),
                arrowprops=dict(arrowstyle="->", color="#E0A020", lw=1.5))
    ax.annotate("", xy=(vols_tp[2], rets_tp[2]), xytext=(vols_tp[0], rets_tp[0]),
                arrowprops=dict(arrowstyle="->", color="#9B59B6", lw=1.5))

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlabel("Annualised Volatility (σ)", fontsize=11, color="#555555")
    ax.set_ylabel("Annualised Return (μ)", fontsize=11, color="#555555")
    ax.set_title(
        "Frontier Shift Decomposition — Which input drives the shift?\n"
        "Stars = tangency  |  Yellow arrow = μ effect  |  Purple arrow = Σ effect",
        fontsize=11, color="#222222", pad=10)
    ax.legend(fontsize=9.5, framealpha=0.7, loc="upper left")

    sr_text = (f"Sharpe Ratios:\n"
               f"Pre (μ_pre, Σ_pre):     {srs[0]:.3f}\n"
               f"μ only (μ_post, Σ_pre): {srs[1]:.3f}\n"
               f"Σ only (μ_pre, Σ_post): {srs[2]:.3f}\n"
               f"Post (μ_post, Σ_post):  {srs[3]:.3f}")
    ax.text(0.98, 0.04, sr_text, transform=ax.transAxes,
            fontsize=9, va="bottom", ha="right",
            bbox=dict(facecolor="white", alpha=0.85,
                      edgecolor="#CCCCCC", boxstyle="round,pad=0.4"),
            fontfamily="monospace")

    fig.tight_layout()
    out = out_path or "../results/robustness/frontier_decomposition.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")

    return {
        "mu_effect_sr":    round(srs[1] - srs[0], 4),
        "sigma_effect_sr": round(srs[2] - srs[0], 4),
        "total_sr_gain":   round(srs[3] - srs[0], 4),
    }


# ════════════════════════════════════════════════════════════════
# 10. VAR / CVAR ANALYSIS
# ════════════════════════════════════════════════════════════════

def var_cvar_analysis(ret_pre, ret_post, rf_pre, rf_post, N=100_000):
    """
    Student-t Monte Carlo VaR and CVaR at 95% and 99%.

    For each portfolio:
      - Fit a univariate Student-t (MLE) to the daily return series
      - Simulate N=100,000 daily returns from the fitted distribution
      - Read off VaR and CVaR at 95% and 99% from the simulation
      - Annualise via sqrt-of-time scaling (×√252)

    The fitted degrees-of-freedom (t_df) column shows tail fatness;
    lower df → fatter tails → more extreme losses than Gaussian.

    Saves Results/Robustness/var_cvar.csv and returns the DataFrame.
    """
    mu_pre,  cov_pre,  _ = ledoit_wolf(ret_pre)
    mu_post, cov_post, _ = ledoit_wolf(ret_post)
    w_tp_pre  = max_sharpe(mu_pre,  cov_pre,  rf_pre)
    w_tp_post = max_sharpe(mu_post, cov_post, rf_post)

    strategies = [
        ("Pre TP",  (ret_pre  * w_tp_pre).sum(axis=1)),
        ("Post TP", (ret_post * w_tp_post).sum(axis=1)),
    ]

    rows = []
    for name, r in strategies:
        r = r.dropna().values
        seed = 42 if "Pre" in name else 123

        var95,  cvar95,  df95  = _student_t_var_cvar(r, 0.95, N=N, seed=seed)
        var99,  cvar99,  _     = _student_t_var_cvar(r, 0.99, N=N, seed=seed)

        rows.append({
            "Strategy":          name,
            "t_df (tail fit)":   round(df95, 3),
            "VaR 95% (daily)":   round(var95,  6),
            "CVaR 95% (daily)":  round(cvar95, 6),
            "VaR 99% (daily)":   round(var99,  6),
            "CVaR 99% (daily)":  round(cvar99, 6),
            "VaR 95% (ann)":     round(var95  * np.sqrt(252), 5),
            "CVaR 95% (ann)":    round(cvar95 * np.sqrt(252), 5),
            "VaR 99% (ann)":     round(var99  * np.sqrt(252), 5),
            "CVaR 99% (ann)":    round(cvar99 * np.sqrt(252), 5),
        })

    df = pd.DataFrame(rows)
    df.to_csv("../results/robustness/var_cvar.csv", index=False)
    print(f"  [saved] Results/Robustness/var_cvar.csv")
    return df


def plot_monte_carlo_trials(ret_pre, ret_post, rf_pre, rf_post,
                            N=100_000, out_path=None):
    """
    Visualise Student-t Monte Carlo simulations underlying the VaR/CVaR analysis.

    Each panel shows:
      - Histogram of N simulated daily returns (fitted Student-t draws)
      - Fitted Student-t PDF overlaid
      - Vertical lines for VaR and CVaR at 95% and 99%
      - Left tail shaded for each risk level
    """
    PRE_C  = "#1D9E75"
    POST_C = "#534AB7"

    mu_pre,  cov_pre,  _ = ledoit_wolf(ret_pre)
    mu_post, cov_post, _ = ledoit_wolf(ret_post)
    w_tp_pre  = max_sharpe(mu_pre,  cov_pre,  rf_pre)
    w_tp_post = max_sharpe(mu_post, cov_post, rf_post)

    strategies = [
        ("Pre-COVID TP",  (ret_pre  * w_tp_pre).sum(axis=1).dropna().values,  PRE_C,  42),
        ("Post-COVID TP", (ret_post * w_tp_post).sum(axis=1).dropna().values, POST_C, 123),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    fig.patch.set_facecolor("#FAFAFA")

    for ax, (name, r, color, seed) in zip(axes, strategies):
        df_t, loc, scale = stats.t.fit(r)
        sim = stats.t.rvs(df_t, loc=loc, scale=scale, size=N, random_state=seed)

        var95  = float(-np.percentile(sim, 5))
        cvar95 = float(-sim[sim <= -var95].mean())
        var99  = float(-np.percentile(sim, 1))
        cvar99 = float(-sim[sim <= -var99].mean())

        style_ax(ax)

        # histogram of simulated draws
        counts, bins, _ = ax.hist(
            sim, bins=120, density=True,
            color=color, alpha=0.30, zorder=2
        )

        # fitted Student-t PDF
        x_pdf = np.linspace(sim.min(), sim.max(), 800)
        ax.plot(x_pdf, stats.t.pdf(x_pdf, df_t, loc, scale),
                color=color, lw=2.2, zorder=4, label=f"Fitted t (df={df_t:.1f})")

        # shade 99% tail
        x_tail99 = np.linspace(sim.min(), -var99, 300)
        ax.fill_between(x_tail99,
                        stats.t.pdf(x_tail99, df_t, loc, scale),
                        alpha=0.55, color="#E74C3C", zorder=3,
                        label=f"99% tail  VaR={var99:.3%}  CVaR={cvar99:.3%}")

        # shade 95% tail (lighter, on top of 99% shade)
        x_tail95 = np.linspace(-var99, -var95, 300)
        ax.fill_between(x_tail95,
                        stats.t.pdf(x_tail95, df_t, loc, scale),
                        alpha=0.35, color="#E67E22", zorder=3,
                        label=f"95% tail  VaR={var95:.3%}  CVaR={cvar95:.3%}")

        # VaR lines
        ax.axvline(-var95,  color="#E67E22", lw=1.6, ls="--", zorder=5)
        ax.axvline(-var99,  color="#E74C3C", lw=1.6, ls="--", zorder=5)

        # CVaR lines
        ax.axvline(-cvar95, color="#E67E22", lw=1.4, ls=":",  zorder=5)
        ax.axvline(-cvar99, color="#E74C3C", lw=1.4, ls=":",  zorder=5)

        # annotation box
        ann = (f"VaR  95%: {var95:.3%}\n"
               f"CVaR 95%: {cvar95:.3%}\n"
               f"VaR  99%: {var99:.3%}\n"
               f"CVaR 99%: {cvar99:.3%}")
        ax.text(0.97, 0.97, ann, transform=ax.transAxes,
                fontsize=8.5, va="top", ha="right",
                bbox=dict(facecolor="white", alpha=0.88,
                          edgecolor="#CCCCCC", boxstyle="round,pad=0.4"),
                fontfamily="monospace")

        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
        ax.set_xlabel("Daily Return", fontsize=10, color="#555555")
        ax.set_ylabel("Probability Density", fontsize=10, color="#555555")
        ax.set_title(f"{name}\nN = {N:,} Monte Carlo draws  |  Student-t fit",
                     fontsize=10.5, color="#222222")
        ax.legend(fontsize=8.5, framealpha=0.8, loc="upper left")

    fig.suptitle(
        "Monte Carlo VaR / CVaR — Simulated Daily Return Distributions\n"
        "Dashed = VaR  |  Dotted = CVaR  |  Orange = 95% tail  |  Red = 99% tail",
        fontsize=11.5, color="#222222", y=1.02
    )
    fig.tight_layout()
    out = out_path or "../results/robustness/monte_carlo_var_cvar.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


def plot_monte_carlo_paths(ret_pre, ret_post, rf_pre, rf_post,
                           n_paths=1000, horizon=252, out_path=None):
    """
    Fan chart of simulated cumulative return paths (1-year horizon).

    Method:
      1. Fit Student-t to each tangency portfolio's daily return series.
      2. Simulate n_paths × horizon daily draws from the fitted distribution.
      3. Compound into cumulative return paths (starting at 1.0).
      4. Plot individual paths (thin, low alpha) + percentile fan bands
         (5/25/50/75/95th) to form the cone of uncertainty.

    Both periods are overlaid on one chart so the shift in the
    distribution of outcomes is immediately visible.
    """
    PRE_C  = "#1D9E75"
    POST_C = "#534AB7"

    mu_pre,  cov_pre,  _ = ledoit_wolf(ret_pre)
    mu_post, cov_post, _ = ledoit_wolf(ret_post)
    w_tp_pre  = max_sharpe(mu_pre,  cov_pre,  rf_pre)
    w_tp_post = max_sharpe(mu_post, cov_post, rf_post)

    strategies = [
        ("Pre-COVID TP",  (ret_pre  * w_tp_pre).sum(axis=1).dropna().values,  PRE_C,  42),
        ("Post-COVID TP", (ret_post * w_tp_post).sum(axis=1).dropna().values, POST_C, 123),
    ]

    PCTILES = [5, 25, 50, 75, 95]
    days    = np.arange(horizon + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)
    fig.patch.set_facecolor("#FAFAFA")

    all_bands = {}   # store for overlay reference
    for ax, (name, r, color, seed) in zip(axes, strategies):
        style_ax(ax)
        np.random.seed(seed)

        df_t, loc, scale = stats.t.fit(r)
        # (n_paths, horizon) matrix of daily draws
        draws = stats.t.rvs(df_t, loc=loc, scale=scale,
                            size=(n_paths, horizon),
                            random_state=seed)

        # cumulative return paths: shape (n_paths, horizon+1), starts at 1
        cum = np.ones((n_paths, horizon + 1))
        cum[:, 1:] = np.exp(np.cumsum(draws, axis=1))

        # individual paths (thin, transparent)
        n_show = min(300, n_paths)
        for i in range(n_show):
            ax.plot(days, cum[i], color=color, lw=0.4, alpha=0.04, zorder=2)

        # percentile bands
        bands = np.percentile(cum, PCTILES, axis=0)   # (5, horizon+1)
        all_bands[name] = bands

        alphas = [0.18, 0.28, 0.00, 0.28, 0.18]
        labels = ["5–25%", None, None, "75–95%", None]
        for i in range(len(PCTILES) - 1):
            ax.fill_between(days, bands[i], bands[i + 1],
                            alpha=alphas[i], color=color, zorder=3)

        # median path
        ax.plot(days, bands[2], color=color, lw=2.5,
                zorder=5, label="Median path")
        # 5th / 95th boundary lines
        ax.plot(days, bands[0], color=color, lw=1.1, ls="--",
                alpha=0.7, zorder=4, label="5th / 95th pctile")
        ax.plot(days, bands[4], color=color, lw=1.1, ls="--",
                alpha=0.7, zorder=4)

        ax.axhline(1.0, color="#AAAAAA", lw=0.9, ls=":", zorder=1)

        # terminal stats box
        terminal = cum[:, -1]
        med_fin  = np.median(terminal)
        p5_fin   = np.percentile(terminal, 5)
        p95_fin  = np.percentile(terminal, 95)
        prob_pos = (terminal > 1.0).mean()
        ann = (f"After {horizon} days:\n"
               f"Median:  {med_fin:.3f}×\n"
               f"P5–P95:  {p5_fin:.3f}× – {p95_fin:.3f}×\n"
               f"P(gain): {prob_pos:.1%}")
        ax.text(0.97, 0.05, ann, transform=ax.transAxes,
                fontsize=9, va="bottom", ha="right",
                bbox=dict(facecolor="white", alpha=0.88,
                          edgecolor="#CCCCCC", boxstyle="round,pad=0.4"),
                fontfamily="monospace")

        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y:.2f}×"))
        ax.set_xlabel("Trading Days Forward", fontsize=10, color="#555555")
        ax.set_ylabel("Cumulative Return (start = 1.0)", fontsize=10, color="#555555")
        ax.set_title(
            f"{name}  |  Student-t fit (df = {df_t:.1f})\n"
            f"N = {n_paths:,} paths  ·  {horizon}-day horizon",
            fontsize=10.5, color="#222222")
        ax.legend(fontsize=9, framealpha=0.8, loc="upper left")

    fig.suptitle(
        "Monte Carlo Return Path Fan Chart — Cone of Uncertainty\n"
        "Shaded bands = 5–95th and 25–75th percentiles  |  Dashed = 5th / 95th",
        fontsize=11.5, color="#222222", y=1.02
    )
    fig.tight_layout()
    out = out_path or "../results/robustness/monte_carlo_paths.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


# ════════════════════════════════════════════════════════════════
# INDIVIDUAL STOCK RETURN TESTS
# ════════════════════════════════════════════════════════════════

def individual_stock_return_tests(ret_pre, ret_post, alpha=0.05, out_path=None):
    """
    Welch's two-sample t-test on daily log returns for each stock.
    H0: mu_pre == mu_post  (two-sided)
    Multiple testing corrected via Benjamini-Hochberg FDR.

    Returns the full results DataFrame (one row per ticker).
    """
    rows = []
    for t in ret_pre.columns:
        r_pre  = ret_pre[t].dropna().values
        r_post = ret_post[t].dropna().values
        t_stat, p_val = stats.ttest_ind(r_pre, r_post, equal_var=False)
        rows.append({
            "Ticker":   t,
            "Sector":   SECTOR_MAP.get(t, "Others"),
            "mu_pre":   r_pre.mean()  * 252,
            "mu_post":  r_post.mean() * 252,
            "delta_mu": (r_post.mean() - r_pre.mean()) * 252,
            "t_stat":   round(float(t_stat), 3),
            "p_raw":    round(float(p_val),  4),
        })

    df = pd.DataFrame(rows)

    # Benjamini-Hochberg FDR correction
    n     = len(df)
    order = np.argsort(df["p_raw"].values)
    ranked_p  = df["p_raw"].values[order]
    bh_thresh = (np.arange(1, n + 1) / n) * alpha
    below   = ranked_p <= bh_thresh
    reject  = np.zeros(n, dtype=bool)
    if below.any():
        reject[order[:int(np.where(below)[0].max()) + 1]] = True
    df["sig_BH"] = reject

    df = df.sort_values("delta_mu").reset_index(drop=True)

    # ── plot ────────────────────────────────────────────────────
    n_tickers = len(df)
    fig, ax   = plt.subplots(figsize=(11, max(8, n_tickers * 0.30)))
    fig.patch.set_facecolor("#FAFAFA")
    style_ax(ax)

    bar_colors = [
        ("#1D9E75" if row["delta_mu"] > 0 else "#E74C3C") if row["sig_BH"]
        else "#CCCCCC"
        for _, row in df.iterrows()
    ]

    ax.barh(df["Ticker"], df["delta_mu"] * 100,
            color=bar_colors, alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#555555", lw=0.9, zorder=5)

    # sector labels pinned to the right margin
    x_max = df["delta_mu"].abs().max() * 100
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(x_max * 1.08, i, row["Sector"],
                va="center", ha="left", fontsize=7, color="#999999")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#1D9E75", alpha=0.85, label="Sig. increase  (BH FDR 5%)"),
        Patch(facecolor="#E74C3C", alpha=0.85, label="Sig. decrease  (BH FDR 5%)"),
        Patch(facecolor="#CCCCCC", alpha=0.85, label="Not significant"),
    ], fontsize=9, framealpha=0.8, loc="lower right")

    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0f} pp"))
    ax.set_xlabel(
        "Δ Annualised Mean Return  (Post − Pre, percentage points)",
        fontsize=10, color="#555555")
    sig_n = int(df["sig_BH"].sum())
    ax.set_title(
        f"Individual Stock Return Tests: Pre vs Post-COVID\n"
        f"Welch's t-test  |  Benjamini-Hochberg FDR  |  "
        f"{sig_n} / {n_tickers} stocks significant at {alpha:.0%}",
        fontsize=11, color="#222222", pad=10)
    ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    out = out_path or "../results/robustness/stock_return_tests.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")

    sig = df[df["sig_BH"]].copy()
    sig["delta_mu_pct"] = (sig["delta_mu"] * 100).round(1)
    print(f"  Significant stocks (BH FDR 5%): {sig_n}")
    if sig_n:
        print(sig[["Ticker", "Sector", "delta_mu_pct", "t_stat", "p_raw"]].to_string(index=False))

    return df


# ════════════════════════════════════════════════════════════════
# 11. HHI CONCENTRATION ANALYSIS
# ════════════════════════════════════════════════════════════════

def plot_hhi_concentration(ret_pre, ret_post, rf_pre, rf_post,
                           out_path=None):
    mu_pre,  cov_pre,  _ = ledoit_wolf(ret_pre)
    mu_post, cov_post, _ = ledoit_wolf(ret_post)

    w_tp_pre  = max_sharpe(mu_pre,  cov_pre,  rf_pre)
    w_tp_post = max_sharpe(mu_post, cov_post, rf_post)
    w_mv_pre  = min_var(mu_pre,  cov_pre)
    w_mv_post = min_var(mu_post, cov_post)
    w_ew      = np.ones(len(mu_pre)) / len(mu_pre)

    portfolios = {
        "Pre TP":   w_tp_pre,  "Post TP":  w_tp_post,
        "Pre GMV":  w_mv_pre,  "Post GMV": w_mv_post,
        "1/N":      w_ew,
    }
    hhi_vals = {k: hhi(v) for k, v in portfolios.items()}
    eff_n    = {k: 1 / v for k, v in hhi_vals.items()}

    colors_bar = ["#1D9E75", "#534AB7", "#1D9E75", "#534AB7", "#888888"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor("#FAFAFA")

    for ax, data, ylabel, title in zip(
        axes,
        [hhi_vals, eff_n],
        ["HHI (0 = perfectly diversified, 1 = single stock)",
         "Effective N = 1/HHI (# equivalent equal positions)"],
        ["Herfindahl-Hirschman Index (Concentration)",
         "Effective Number of Stocks (Diversification)"]
    ):
        style_ax(ax)
        bars = ax.bar(list(data.keys()), list(data.values()),
                      color=colors_bar, alpha=0.85,
                      edgecolor="white", linewidth=1.2)
        for bar, val in zip(bars, data.values()):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005 * max(data.values()),
                    f"{val:.2f}", ha="center", fontsize=10,
                    fontweight="bold", color="#333333")
        ax.set_title(title, fontsize=11, color="#222222", pad=8)
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.set_xlabel("Portfolio", fontsize=9.5)
        ax.tick_params(rotation=15)

    fig.suptitle(
        "Portfolio Concentration Analysis\n"
        "Pre vs Post-COVID tangency concentration (HHI)",
        fontsize=12, fontweight="500", color="#222222", y=1.03)
    fig.tight_layout()
    out = out_path or "../results/robustness/hhi_concentration.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")
    return hhi_vals


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
        out_path="../results/robustness/bootstrap_frontier_bands.png"
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
    #     out_path="../results/robustness/estimator_comparison.png"
    # )
    # est_df.to_csv(
    #     "../results/robustness/estimator_comparison.csv",
    #     index=False
    # )
    # print("  Estimator SR summary:")
    # print(est_df.to_string(index=False))

    # ── Test 4: Transaction costs ─────────────────
    print("\n── Test 4: Transaction cost analysis ──")
    tc_df = transaction_cost_analysis(ret_pre, ret_post, RF_PRE, RF_POST)
    tc_df.to_csv("../results/robustness/transaction_costs.csv", index=False)
    print(tc_df.to_string(index=False))

    # ── Test 5: Sector weight check ───────────────
    print("\n── Test 5: Sector weight check ──")
    sector_df = sector_weight_check(
        ret_pre, ret_post,
        out_path="../results/robustness/sector_weights.png",
        rf_pre=RF_PRE, rf_post=RF_POST
    )
    sector_df.to_csv("../results/robustness/sector_weights.csv", index=False)
    print(sector_df.to_string(index=False))

    # ── Frontier decomposition ────────────────────
    print("\n── Frontier Shift Decomposition (μ vs Σ) ──")
    decomp = plot_frontier_decomposition(
        ret_pre, ret_post, RF_PRE, RF_POST,
        out_path="../results/robustness/frontier_decomposition.png"
    )
    print(f"  μ-only SR gain:  {decomp['mu_effect_sr']:+.4f}")
    print(f"  Σ-only SR gain:  {decomp['sigma_effect_sr']:+.4f}")
    print(f"  Total SR gain:   {decomp['total_sr_gain']:+.4f}")

    # ── VaR / CVaR ────────────────────────────────
    print("\n── VaR / CVaR Analysis (95% & 99%) ──")
    vc_df = var_cvar_analysis(ret_pre, ret_post, RF_PRE, RF_POST)
    print(vc_df.to_string(index=False))

    print("\n── Monte Carlo Trial Plot ──")
    plot_monte_carlo_trials(
        ret_pre, ret_post, RF_PRE, RF_POST,
        out_path="../results/robustness/monte_carlo_var_cvar.png"
    )

    print("\n── Monte Carlo Return Path Fan Chart ──")
    plot_monte_carlo_paths(
        ret_pre, ret_post, RF_PRE, RF_POST,
        n_paths=1000, horizon=252,
        out_path="../results/robustness/monte_carlo_paths.png"
    )

    # ── Individual stock return tests ─────────────
    print("\n── Individual Stock Return Tests (Welch's t / BH FDR) ──")
    stock_test_df = individual_stock_return_tests(
        ret_pre, ret_post,
        out_path="../results/robustness/stock_return_tests.png"
    )
    stock_test_df.to_csv("../results/robustness/stock_return_tests.csv", index=False)

    # ── HHI concentration ─────────────────────────
    print("\n── HHI Concentration Analysis ──")
    hhi_vals = plot_hhi_concentration(
        ret_pre, ret_post, RF_PRE, RF_POST,
        out_path="../results/robustness/hhi_concentration.png"
    )
    print(f"  Pre TP HHI:  {hhi_vals['Pre TP']:.4f}  "
          f"(eff N = {1/hhi_vals['Pre TP']:.1f})")
    print(f"  Post TP HHI: {hhi_vals['Post TP']:.4f}  "
          f"(eff N = {1/hhi_vals['Post TP']:.1f})")

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
        "boot_shift_significant":    boot_result["shift_significant"],
        "TC_turnover_pct":           float(tc_df["Turnover (%)"].iloc[0]),
        "TC_SR_pre_gross":           float(tc_df["SR pre gross"].iloc[0]),
        "TC_SR_post_gross":          float(tc_df["SR post gross"].iloc[0]),
        "TC_SR_pre_net":             float(tc_df["SR pre net"].iloc[0]),
        "TC_SR_post_net":            float(tc_df["SR post net"].iloc[0]),
        "TC_delta_SR_gross":         float(tc_df["Delta SR gross"].iloc[0]),
        "TC_delta_SR_net":           float(tc_df["Delta SR net"].iloc[0]),
        "TC_improvement_survives":   bool(tc_df["Improvement survives"].iloc[0]),
        "sector_max_gap":            round(float(
            (sector_df["Nifty50_wt"] - sector_df["Universe_wt"]).abs().max()
        ), 4),
        "sector_max_gap_name":       str(sector_df.loc[
            (sector_df["Nifty50_wt"] - sector_df["Universe_wt"]).abs().idxmax(),
            "Sector"
        ]),
        "decomp_mu_effect_sr":       decomp["mu_effect_sr"],
        "decomp_sigma_effect_sr":    decomp["sigma_effect_sr"],
        "decomp_total_sr_gain":      decomp["total_sr_gain"],
        "HHI_pre_tp":                round(hhi_vals["Pre TP"],   4),
        "HHI_post_tp":               round(hhi_vals["Post TP"],  4),
        "HHI_pre_gmv":               round(hhi_vals["Pre GMV"],  4),
        "HHI_post_gmv":              round(hhi_vals["Post GMV"], 4),
        "HHI_eff_n_pre_tp":          round(1 / hhi_vals["Pre TP"],  2),
        "HHI_eff_n_post_tp":         round(1 / hhi_vals["Post TP"], 2),
        "VaR95_pre_tp_ann":          float(vc_df.loc[vc_df["Strategy"]=="Pre TP",  "VaR 95% (ann)"].iloc[0]),
        "CVaR95_pre_tp_ann":         float(vc_df.loc[vc_df["Strategy"]=="Pre TP",  "CVaR 95% (ann)"].iloc[0]),
        "VaR95_post_tp_ann":         float(vc_df.loc[vc_df["Strategy"]=="Post TP", "VaR 95% (ann)"].iloc[0]),
        "CVaR95_post_tp_ann":        float(vc_df.loc[vc_df["Strategy"]=="Post TP", "CVaR 95% (ann)"].iloc[0]),
        "VaR99_pre_tp_ann":          float(vc_df.loc[vc_df["Strategy"]=="Pre TP",  "VaR 99% (ann)"].iloc[0]),
        "CVaR99_pre_tp_ann":         float(vc_df.loc[vc_df["Strategy"]=="Pre TP",  "CVaR 99% (ann)"].iloc[0]),
        "VaR99_post_tp_ann":         float(vc_df.loc[vc_df["Strategy"]=="Post TP", "VaR 99% (ann)"].iloc[0]),
        "CVaR99_post_tp_ann":        float(vc_df.loc[vc_df["Strategy"]=="Post TP", "CVaR 99% (ann)"].iloc[0]),
        "n_sig_stocks_BH":           int(stock_test_df["sig_BH"].sum()),
        "sig_stocks_increased":      ", ".join(
            stock_test_df.loc[stock_test_df["sig_BH"] & (stock_test_df["delta_mu"] > 0), "Ticker"].tolist()
        ),
        "sig_stocks_decreased":      ", ".join(
            stock_test_df.loc[stock_test_df["sig_BH"] & (stock_test_df["delta_mu"] < 0), "Ticker"].tolist()
        ),
        # "Retail_pre":        round(banking["avg_retail_pre"],    4),
        # "Retail_post":       round(banking["avg_retail_post"],   4),
        # "Wholesale_pre":     round(banking["avg_wholesale_pre"], 4),
        # "Wholesale_post":    round(banking["avg_wholesale_post"],4),
    }
    pd.DataFrame([summary]).to_csv(
        "../results/robustness/test_summary.csv",
        index=False
    )
    print("Saved: Results/Robustness/test_summary.csv")
    print("\nAll done.\n")


if __name__ == "__main__":
    main()
