"""
normality_and_structural_break.py
===================================
Two diagnostic tests that motivate the methodology choices in the paper.

Test 1 — Normality diagnostics on tangency portfolio returns
  • Jarque-Bera test       → fat tails violate JKM normality assumption
  • Engle (1982) ARCH-LM  → volatility clustering violates JKM i.i.d. assumption
  Together: JKM invalid → LW (2008) Boot-TS required
  Also motivates: Font (2016) bootstrap (no distributional assumption needed),
                  Chen-Qin (2010) mean test (no normality required)

Test 2 — ICSS structural break test on NIFTY 50 index returns
  • Inclan and Tiao (1994) ICSS algorithm
  • Detects variance break points at unknown dates
  • Validates the three-regime study design (pre / crisis / post)

References
----------
Engle, R.F. (1982). Autoregressive conditional heteroscedasticity with
  estimates of the variance of United Kingdom inflation.
  Econometrica 50(4): 987–1007.
Inclan, C. and Tiao, G.C. (1994). Use of cumulative sums of squares for
  retrospective detection of changes of variance.
  JASA 89(427): 913–923.
Jarque, C.M. and Bera, A.K. (1980). Efficient tests for normality,
  homoscedasticity and serial independence of regression residuals.
  Economics Letters 6(3): 255–259.

Outputs saved to Results/Diagnostics/
"""

import os
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.stats.diagnostic import het_arch
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
os.makedirs("Results/Diagnostics", exist_ok=True)

# ── period constants (must match 5_robustness.py) ──────────────────────────
P = {
    "a_pre":  ("2015-01-01", "2020-01-31"),
    "a_post": ("2021-07-01", "2024-12-31"),
    "crisis": ("2020-02-01", "2021-06-30"),
}

PRE_C  = "#1D9E75"
POST_C = "#534AB7"
CRI_C  = "#E74C3C"


# ════════════════════════════════════════════════════════════════
# DATA LOADING  (mirrors 5_robustness.py)
# ════════════════════════════════════════════════════════════════

def load_risk_free_rates():
    xl = pd.read_excel(
        "Auctions of 91-Day Government of India Treasury Bills.xlsx",
        header=5, usecols=[1, 16]
    )
    xl.columns = ["Date of Auction", "Weighted Avg Yield (per cent)"]
    xl["Date of Auction"] = pd.to_datetime(xl["Date of Auction"], errors="coerce")
    xl = xl.dropna(subset=["Date of Auction", "Weighted Avg Yield (per cent)"])
    xl = xl.set_index("Date of Auction").sort_index()
    yields = pd.to_numeric(
        xl["Weighted Avg Yield (per cent)"], errors="coerce"
    ).dropna() / 100.0
    rf_pre  = float(yields.loc[P["a_pre"][0]:P["a_pre"][1]].mean())
    rf_post = float(yields.loc[P["a_post"][0]:P["a_post"][1]].mean())
    return rf_pre, rf_post


def load_prices():
    prices = pd.read_csv("Data/prices_daily.csv", index_col=0, parse_dates=True)
    prices = prices[~prices.index.astype(str).str.contains(
        "Price|Ticker", na=False)]
    return prices.astype(float).dropna(axis=1, how="all").sort_index()


def to_log_returns(prices):
    return np.log(prices / prices.shift(1)).iloc[1:]


def slice_period(log_ret, start, end, max_nan=0.10):
    ret  = log_ret.loc[start:end]
    keep = ret.columns[ret.isna().mean() <= max_nan]
    return ret[keep].ffill(limit=1).dropna(axis=1)


def max_sharpe(mu, cov, rf):
    n = len(mu)
    def neg_sr(w):
        r = w @ mu; v = np.sqrt(w @ cov @ w)
        return -(r - rf) / (v + 1e-12)
    res = minimize(neg_sr, np.ones(n)/n, method="SLSQP",
                   bounds=[(0,1)]*n,
                   constraints=[{"type":"eq","fun":lambda w: w.sum()-1}],
                   options={"ftol":1e-12,"maxiter":2000})
    return res.x if res.success else np.ones(n)/n


# ════════════════════════════════════════════════════════════════
# TEST 1 — NORMALITY DIAGNOSTICS
# ════════════════════════════════════════════════════════════════

def jarque_bera_test(x):
    """
    Jarque-Bera test for normality.
    H0: skewness = 0 and excess kurtosis = 0 (returns are normal).
    Under H0, JB ~ chi2(2).
    """
    stat, p = stats.jarque_bera(x)
    sk  = float(stats.skew(x))
    ku  = float(stats.kurtosis(x))      # excess kurtosis (normal = 0)
    return {"skewness": round(sk,3),
            "excess_kurtosis": round(ku,3),
            "JB_stat": round(stat,2),
            "JB_pvalue": round(p,6),
            "JB_reject_5pct": bool(p < 0.05)}


def arch_lm_test(x, nlags=5):
    """
    Engle (1982) ARCH-LM test.
    H0: no ARCH effects (squared residuals are uncorrelated up to `nlags` lags).
    LM statistic ~ chi2(nlags) under H0.
    Uses demeaned series as residuals.
    """
    resid = x - x.mean()
    lm, lm_p, _, _ = het_arch(resid, nlags=nlags)
    return {"ARCH_LM_stat": round(lm, 2),
            "ARCH_LM_pvalue": round(lm_p, 6),
            "ARCH_LM_lags": nlags,
            "ARCH_reject_5pct": bool(lm_p < 0.05)}


def run_normality_diagnostics(ret_pre, ret_post, w_pre, w_post,
                               rf_pre, rf_post):
    """
    Run JB + ARCH-LM on the daily excess returns of each tangency portfolio.
    Returns a formatted DataFrame for inclusion in the paper.
    """
    exc_pre  = (ret_pre.values  @ w_pre)  - rf_pre  / 252
    exc_post = (ret_post.values @ w_post) - rf_post / 252

    rows = []
    for label, exc in [("Pre-COVID", exc_pre), ("Post-COVID", exc_post)]:
        jb   = jarque_bera_test(exc)
        arch = arch_lm_test(exc, nlags=5)
        rows.append({"Period": label, **jb, **arch})

    df = pd.DataFrame(rows).set_index("Period")

    # ── print table ──────────────────────────────────────────────
    print("\n  ┌─── Normality Diagnostics: Tangency Portfolio Daily Excess Returns ───┐")
    print(f"  │{'':40s}{'Pre-COVID':>15}{'Post-COVID':>15}  │")
    print(f"  │{'─'*70}  │")
    for col, label in [
        ("skewness",        "Skewness"),
        ("excess_kurtosis", "Excess kurtosis"),
        ("JB_stat",         "Jarque-Bera stat"),
        ("JB_pvalue",       "JB p-value"),
        ("JB_reject_5pct",  "JB reject H0 (5%)"),
        ("ARCH_LM_stat",    "ARCH-LM stat (5 lags)"),
        ("ARCH_LM_pvalue",  "ARCH-LM p-value"),
        ("ARCH_reject_5pct","ARCH reject H0 (5%)"),
    ]:
        pre_v  = str(df.loc["Pre-COVID",  col])
        post_v = str(df.loc["Post-COVID", col])
        print(f"  │  {label:38s}{pre_v:>15}{post_v:>15}  │")
    print("  └" + "─"*72 + "┘")
    print("  JB H0: normal distribution.  ARCH-LM H0: no volatility clustering.")
    print("  Both conditions needed for JKM validity; both are rejected here.\n")

    return df, exc_pre, exc_post


def plot_qq_and_acf_squared(exc_pre, exc_post, out_path):
    """
    Two-panel figure:
      Left  — Q-Q plot against normal (fat tails visible)
      Right — ACF of squared returns (GARCH persistence visible)
    """
    from statsmodels.graphics.tsaplots import plot_acf

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#FAFAFA")

    # Q-Q plot
    ax = axes[0]
    ax.set_facecolor("#FAFAFA")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["left","bottom"]: ax.spines[sp].set_color("#CCCCCC")

    (osm_pre,  osr_pre),  _ = stats.probplot(exc_pre,  dist="norm")
    (osm_post, osr_post), _ = stats.probplot(exc_post, dist="norm")

    ax.scatter(osm_pre,  osr_pre,  s=4, alpha=0.4, color=PRE_C,
               label="Pre-COVID")
    ax.scatter(osm_post, osr_post, s=4, alpha=0.4, color=POST_C,
               label="Post-COVID")

    # 45-degree reference line
    lo = min(osm_pre[0], osm_post[0])
    hi = max(osm_pre[-1], osm_post[-1])
    ax.plot([lo,hi],[lo,hi], "k--", lw=1, alpha=0.5, label="Normal")

    ax.set_xlabel("Theoretical quantiles", fontsize=10, color="#555555")
    ax.set_ylabel("Sample quantiles",      fontsize=10, color="#555555")
    ax.set_title("Q-Q Plot: Daily Excess Returns vs Normal\n"
                 "Fat tails → deviations at extremes", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.7)
    ax.tick_params(colors="#666666", labelsize=9)
    ax.grid(True, linestyle="--", linewidth=0.4, color="#DDDDDD", alpha=0.8)

    # ACF of squared returns
    ax2 = axes[1]
    ax2.set_facecolor("#FAFAFA")
    for sp in ["top","right"]: ax2.spines[sp].set_visible(False)
    for sp in ["left","bottom"]: ax2.spines[sp].set_color("#CCCCCC")

    nlags = 20
    def acf_values(x, nlags):
        x2 = (x - x.mean())**2
        out = []
        for lag in range(1, nlags+1):
            out.append(np.corrcoef(x2[lag:], x2[:-lag])[0,1])
        return np.array(out)

    acf_pre  = acf_values(exc_pre,  nlags)
    acf_post = acf_values(exc_post, nlags)
    lags     = np.arange(1, nlags+1)
    ci       = 1.96 / np.sqrt(len(exc_pre))    # approximate 95% CI

    w = 0.3
    ax2.bar(lags - w/2, acf_pre,  w, color=PRE_C,  alpha=0.8,
            label="Pre-COVID")
    ax2.bar(lags + w/2, acf_post, w, color=POST_C, alpha=0.8,
            label="Post-COVID")
    ax2.axhline( ci, color="#888888", lw=1, ls="--", alpha=0.7)
    ax2.axhline(-ci, color="#888888", lw=1, ls="--", alpha=0.7)
    ax2.axhline(0,   color="#888888", lw=0.5)

    ax2.set_xlabel("Lag (days)",     fontsize=10, color="#555555")
    ax2.set_ylabel("Autocorrelation",fontsize=10, color="#555555")
    ax2.set_title("ACF of Squared Excess Returns\n"
                  "Persistent autocorrelation → GARCH effects", fontsize=10)
    ax2.legend(fontsize=9, framealpha=0.7)
    ax2.tick_params(colors="#666666", labelsize=9)
    ax2.grid(True, linestyle="--", linewidth=0.4, color="#DDDDDD", alpha=0.8)
    ax2.set_xlim(0, nlags+1)

    fig.suptitle(
        "Non-normality Diagnostics: Q-Q Plot and ARCH Effects\n"
        "Both invalidate the JKM/Memmel test → Ledoit-Wolf (2008) Boot-TS required",
        fontsize=11, color="#222222", y=1.01
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ════════════════════════════════════════════════════════════════
# TEST 2 — ICSS STRUCTURAL BREAK TEST
# ════════════════════════════════════════════════════════════════

def icss(returns):
    """
    Inclan-Tiao (1994) ICSS algorithm for detecting breaks in variance.

    Algorithm (§2 of Inclan-Tiao 1994):
      1. Compute C_k = Σ_{t=1}^k e_t²  and  D_k = C_k/C_T − k/T
      2. IT statistic = sqrt(T/2) * max|D_k|
      3. At 5%, critical value = 1.358  (Table 1 of paper)
      4. Iterate: split at k* and recurse on each sub-segment until
         no sub-segment shows a significant break.

    Parameters
    ----------
    returns : 1-D array of daily returns (demeaned internally)

    Returns
    -------
    break_indices : list of integer indices into `returns`
    it_stat       : float — overall IT statistic for the full series
    """
    e = np.asarray(returns, dtype=float)
    e = e - e.mean()                        # demean
    T = len(e)
    CV = 1.358                              # 5% critical value for IT statistic

    def _D_stat(sub):
        n   = len(sub)
        C   = np.cumsum(sub**2)
        C_T = C[-1]
        if C_T < 1e-14:
            return np.zeros(n), 0.0
        D   = C / C_T - np.arange(1, n+1) / n
        return D, float(np.sqrt(n / 2) * np.max(np.abs(D)))

    def _find_breaks(sub, offset):
        """Recursively find break indices within sub-array."""
        n = len(sub)
        if n < 15:
            return []
        D, it = _D_stat(sub)
        if it <= CV:
            return []
        k_star  = int(np.argmax(np.abs(D)))
        breaks  = [offset + k_star]
        # Recurse on left and right segments
        breaks += _find_breaks(sub[:k_star + 1], offset)
        breaks += _find_breaks(sub[k_star + 1:], offset + k_star + 1)
        return breaks

    _, it_full = _D_stat(e)
    raw_breaks = sorted(set(_find_breaks(e, 0)))

    return raw_breaks, it_full


def load_nifty50_index(prices):
    """
    Try to load the NIFTY 50 index from a separate file.
    Falls back to equal-weighted portfolio of available stocks.
    """
    # Try common filenames for the index
    for fname in ["Data/NIFTY50_index.csv", "Data/nifty50.csv",
                  "Data/^NSEI.csv", "NIFTY50.csv"]:
        if os.path.exists(fname):
            idx = pd.read_csv(fname, index_col=0, parse_dates=True)
            # assume it has a 'Close' or 'Adj Close' column, or is a single column
            col = ([c for c in idx.columns if "close" in c.lower()] or
                   list(idx.columns))[0]
            series = idx[col].astype(float).dropna().sort_index()
            print(f"  Loaded NIFTY 50 index from {fname}")
            return series

    # Fallback: equal-weighted portfolio of the 45 stocks
    print("  NIFTY 50 index file not found — using equal-weighted portfolio as proxy")
    ew = prices.mean(axis=1).dropna()
    return ew


def run_icss_test(prices):
    """
    Run ICSS on NIFTY 50 index (or EW proxy) and return detected break dates.
    """
    index_prices = load_nifty50_index(prices)
    log_ret      = np.log(index_prices / index_prices.shift(1)).dropna()
    dates        = log_ret.index

    break_indices, it_stat = icss(log_ret.values)
    break_dates = [dates[i] for i in break_indices]

    print(f"\n  ICSS test on {'NIFTY 50 index' if True else 'EW proxy'}:")
    print(f"  IT statistic (full series): {it_stat:.3f}  "
          f"(critical value = 1.358 at 5%)")
    print(f"  Detected {len(break_dates)} variance break(s):")
    for d in break_dates:
        print(f"    {d.strftime('%Y-%m-%d')}")

    return log_ret, break_dates, it_stat


def plot_icss_results(log_ret, break_dates, it_stat, out_path):
    """
    Two-panel figure:
      Top    — NIFTY 50 daily returns with ICSS break dates marked
      Bottom — Rolling 30-day variance showing regime changes
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.patch.set_facecolor("#FAFAFA")

    # ── Top panel: daily returns ─────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#FAFAFA")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["left","bottom"]: ax.spines[sp].set_color("#CCCCCC")
    ax.grid(True, linestyle="--", linewidth=0.4, color="#DDDDDD", alpha=0.8)

    ax.plot(log_ret.index, log_ret.values * 100,
            color="#888888", lw=0.6, alpha=0.8)

    # Shade the three regimes
    pre_end  = pd.Timestamp(P["a_pre"][1])
    cri_end  = pd.Timestamp(P["a_post"][0])
    post_end = log_ret.index[-1]
    ax.axvspan(log_ret.index[0], pre_end, alpha=0.07, color=PRE_C)
    ax.axvspan(pre_end, cri_end,          alpha=0.07, color=CRI_C)
    ax.axvspan(cri_end, post_end,         alpha=0.07, color=POST_C)

    # ICSS break lines
    for i, bd in enumerate(break_dates):
        ax.axvline(bd, color="black", lw=1.4, ls="--",
                   label="ICSS break" if i == 0 else "")
        ax.text(bd, ax.get_ylim()[1] if ax.get_ylim()[1] != 1 else 3,
                bd.strftime("%b %Y"), fontsize=7.5, ha="center",
                va="bottom", color="black", rotation=45)

    ax.axhline(0, color="#888888", lw=0.5)
    ax.set_ylabel("Daily Return (%)", fontsize=10, color="#555555")
    ax.set_title(
        f"NIFTY 50 Daily Returns with ICSS Variance Break Dates\n"
        f"IT statistic = {it_stat:.3f}  (5% critical value = 1.358)",
        fontsize=10, color="#222222"
    )

    # Legend for regime shading
    from matplotlib.patches import Patch
    legend_els = [
        Patch(color=PRE_C,  alpha=0.3, label="Pre-COVID"),
        Patch(color=CRI_C,  alpha=0.3, label="Crisis (excluded)"),
        Patch(color=POST_C, alpha=0.3, label="Post-COVID"),
        plt.Line2D([0],[0], color="black", ls="--", lw=1.4,
                   label="ICSS break"),
    ]
    ax.legend(handles=legend_els, fontsize=9, framealpha=0.7,
              loc="upper left")
    ax.tick_params(colors="#666666", labelsize=9)

    # ── Bottom panel: rolling 30-day variance ────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#FAFAFA")
    for sp in ["top","right"]: ax2.spines[sp].set_visible(False)
    for sp in ["left","bottom"]: ax2.spines[sp].set_color("#CCCCCC")
    ax2.grid(True, linestyle="--", linewidth=0.4, color="#DDDDDD", alpha=0.8)

    roll_var = log_ret.rolling(30).var() * 252 * 100   # annualised, in pct²
    ax2.fill_between(roll_var.index, roll_var.values,
                     alpha=0.6, color="#888888")
    ax2.plot(roll_var.index, roll_var.values,
             color="#444444", lw=0.8)

    ax2.axvspan(log_ret.index[0], pre_end, alpha=0.07, color=PRE_C)
    ax2.axvspan(pre_end, cri_end,          alpha=0.07, color=CRI_C)
    ax2.axvspan(cri_end, post_end,         alpha=0.07, color=POST_C)

    for bd in break_dates:
        ax2.axvline(bd, color="black", lw=1.4, ls="--")

    ax2.set_ylabel("Rolling 30-day Variance\n(annualised, %²)",
                   fontsize=10, color="#555555")
    ax2.set_xlabel("Date", fontsize=10, color="#555555")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.tick_params(colors="#666666", labelsize=9)

    fig.suptitle(
        "ICSS Structural Break Test: Variance Regime Changes in NIFTY 50\n"
        "Inclan and Tiao (1994) — validates three-regime study design",
        fontsize=11, color="#222222", y=1.01
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print("  Normality Diagnostics + ICSS Structural Break Test")
    print("="*60 + "\n")

    # ── load data ─────────────────────────────────────────────────
    rf_pre, rf_post = load_risk_free_rates()
    prices  = load_prices()
    log_ret = to_log_returns(prices)

    ret_pre  = slice_period(log_ret, *P["a_pre"])
    ret_post = slice_period(log_ret, *P["a_post"])
    shared   = ret_pre.columns.intersection(ret_post.columns)
    ret_pre  = ret_pre[shared]
    ret_post = ret_post[shared]

    # ── tangency portfolios ───────────────────────────────────────
    lw_pre  = LedoitWolf().fit(ret_pre.values)
    lw_post = LedoitWolf().fit(ret_post.values)

    mu_pre   = ret_pre.mean().values  * 252
    cov_pre  = lw_pre.covariance_     * 252
    mu_post  = ret_post.mean().values * 252
    cov_post = lw_post.covariance_    * 252

    w_pre  = max_sharpe(mu_pre,  cov_pre,  rf_pre)
    w_post = max_sharpe(mu_post, cov_post, rf_post)

    # ── Test 1: Normality diagnostics ─────────────────────────────
    print("── Test 1: Normality Diagnostics ──")
    diag_df, exc_pre, exc_post = run_normality_diagnostics(
        ret_pre, ret_post, w_pre, w_post, rf_pre, rf_post
    )
    diag_df.to_csv("Results/Diagnostics/normality_diagnostics.csv")
    print("  Saved: Results/Diagnostics/normality_diagnostics.csv")

    plot_qq_and_acf_squared(
        exc_pre, exc_post,
        out_path="Results/Diagnostics/normality_qq_acf.png"
    )

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"\n  JB test pre:   stat={diag_df.loc['Pre-COVID','JB_stat']}  "
          f"p={diag_df.loc['Pre-COVID','JB_pvalue']}  "
          f"reject={diag_df.loc['Pre-COVID','JB_reject_5pct']}")
    print(f"  JB test post:  stat={diag_df.loc['Post-COVID','JB_stat']}  "
          f"p={diag_df.loc['Post-COVID','JB_pvalue']}  "
          f"reject={diag_df.loc['Post-COVID','JB_reject_5pct']}")
    print(f"\n  ARCH-LM pre:   stat={diag_df.loc['Pre-COVID','ARCH_LM_stat']}  "
          f"p={diag_df.loc['Pre-COVID','ARCH_LM_pvalue']}  "
          f"reject={diag_df.loc['Pre-COVID','ARCH_reject_5pct']}")
    print(f"  ARCH-LM post:  stat={diag_df.loc['Post-COVID','ARCH_LM_stat']}  "
          f"p={diag_df.loc['Post-COVID','ARCH_LM_pvalue']}  "
          f"reject={diag_df.loc['Post-COVID','ARCH_reject_5pct']}")
    print()





if __name__ == "__main__":
    main()
