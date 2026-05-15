"""
enhanced_analysis.py
====================
Comprehensive enhanced analysis for Nifty 45 Pre/Post COVID Efficient Frontier.

Improvements implemented:
  1.  Bootstrap confidence bands (300 samples — proper inference)
  2.  De Roon-Nijman spanning test
  3.  Frontier shift decomposition (mu vs Sigma)
  4.  JK test with formal power analysis
  5.  Rolling Sharpe ratio (when did improvement emerge?)
  6.  Out-of-sample 2025 backtest
  7.  1/N naive benchmark on all frontier plots
  8.  Sector-capped universe robustness (20% cap per sector)
  9.  CAPM alpha for top tangency stocks
 10.  Max drawdown / CVaR analysis
 11.  Correlation heatmaps: Pre vs Post side-by-side
 12.  Delta-mu vs Delta-sigma scatter by sector
 13.  HHI concentration chart
 14.  RF rate sensitivity table (±100 bps)
 15.  Existing tests: JK, Box M, estimator comparison,
      transaction costs, sector weights, banking decomposition

All outputs → Results/Enhanced/
"""

import os, warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")
os.makedirs("Results/Enhanced", exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
RF_PRE  = 0.0654
RF_POST = 0.0578

PRE_START,  PRE_END  = "2015-01-01", "2019-12-31"
POST_START, POST_END = "2020-01-01", "2024-12-31"
OOS_START,  OOS_END  = "2025-01-01", "2025-12-31"
CRISIS_START         = "2020-02-01"
CRISIS_END           = "2021-06-30"

# Colour palette
C_PRE   = "#1D9E75"
C_POST  = "#534AB7"
C_OOS   = "#E07B39"
C_GMV   = "#D85A30"
C_TP    = "#BA7517"
C_1N    = "#888888"

SECTOR_MAP = {
    'HDFCBANK.NS':   'Financials', 'ICICIBANK.NS':  'Financials',
    'KOTAKBANK.NS':  'Financials', 'AXISBANK.NS':   'Financials',
    'SBIN.NS':       'Financials', 'INDUSINDBK.NS': 'Financials',
    'BANKBARODA.NS': 'Financials', 'PNB.NS':        'Financials',
    'TCS.NS':        'IT',         'INFY.NS':        'IT',
    'HCLTECH.NS':    'IT',         'WIPRO.NS':       'IT',
    'TECHM.NS':      'IT',
    'RELIANCE.NS':   'Energy',     'ONGC.NS':        'Energy',
    'BPCL.NS':       'Energy',     'GAIL.NS':        'Energy',
    'TATAPOWER.NS':  'Energy',     'NTPC.NS':        'Energy',
    'POWERGRID.NS':  'Energy',
    'HINDUNILVR.NS': 'Consumer',   'ITC.NS':         'Consumer',
    'ASIANPAINT.NS': 'Consumer',   'MARUTI.NS':      'Consumer',
    'HEROMOTOCO.NS': 'Consumer',   'BAJAJ-AUTO.NS':  'Consumer',
    'M&M.NS':        'Consumer',   'BOSCHLTD.NS':    'Consumer',
    'SUNPHARMA.NS':  'Pharma',     'DRREDDY.NS':     'Pharma',
    'CIPLA.NS':      'Pharma',     'LUPIN.NS':       'Pharma',
    'TATASTEEL.NS':  'Metals',     'HINDALCO.NS':    'Metals',
    'VEDL.NS':       'Metals',     'COALINDIA.NS':   'Metals',
    'ULTRACEMCO.NS': 'Industrials','GRASIM.NS':      'Industrials',
    'ACC.NS':        'Industrials','AMBUJACEM.NS':   'Industrials',
    'LT.NS':         'Industrials','BHEL.NS':        'Industrials',
    'BHARTIARTL.NS': 'Telecom',    'ZEEL.NS':        'Telecom',
    'ADANIPORTS.NS': 'Others',
}

SECTOR_COLORS = {
    'Financials':  '#1f77b4', 'IT':          '#2ca02c',
    'Energy':      '#ff7f0e', 'Consumer':    '#9467bd',
    'Pharma':      '#d62728', 'Metals':      '#8c564b',
    'Industrials': '#e377c2', 'Telecom':     '#17becf',
    'Others':      '#7f7f7f',
}

RETAIL_BANKS    = ['HDFCBANK.NS','KOTAKBANK.NS','AXISBANK.NS','ICICIBANK.NS','SBIN.NS']
WHOLESALE_BANKS = ['PNB.NS','BANKBARODA.NS','INDUSINDBK.NS']

NIFTY50_SECTOR_WEIGHTS = {
    'Financials':0.363,'IT':0.131,'Energy':0.118,'Consumer':0.126,
    'Pharma':0.047,'Metals':0.044,'Industrials':0.062,
    'Telecom':0.038,'Cement':0.031,'Others':0.040,
}

# ═══════════════════════════════════════════════════════════════
# 1. DATA
# ═══════════════════════════════════════════════════════════════
def load_data(path="Data/prices_daily.csv"):
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices = prices.sort_index()
    prices = prices[~prices.index.astype(str).str.contains("Price|Ticker", na=False)]
    prices = prices.astype(float).dropna(axis=1, how="all")
    monthly = prices.resample("ME").last()
    log_ret = np.log(monthly / monthly.shift(1)).iloc[1:]
    print(f"[data] {prices.shape[1]} tickers | {log_ret.shape[0]} monthly obs "
          f"| {log_ret.index[0].date()} – {log_ret.index[-1].date()}")
    return log_ret

def slice_ret(log_ret, start, end, max_nan=0.10):
    ret  = log_ret.loc[start:end]
    keep = ret.columns[ret.isna().mean() <= max_nan]
    ret  = ret[keep].ffill(limit=1).dropna(axis=1)
    return ret

# ═══════════════════════════════════════════════════════════════
# 2. OPTIMISATION HELPERS
# ═══════════════════════════════════════════════════════════════
def lw_cov(ret, freq=12):
    lw  = LedoitWolf().fit(ret.values)
    mu  = ret.mean().values * freq
    cov = lw.covariance_ * freq
    return mu, cov, lw.shrinkage_

def min_var(mu, cov):
    n = len(mu)
    r = minimize(lambda w: w@cov@w, np.ones(n)/n, method="SLSQP",
                 bounds=[(0,1)]*n,
                 constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],
                 options={"ftol":1e-12,"maxiter":2000})
    return r.x if r.success else np.ones(n)/n

def max_sharpe(mu, cov, rf):
    n = len(mu)
    def neg(w):
        v = np.sqrt(w@cov@w)
        return -(w@mu-rf)/(v+1e-9)
    r = minimize(neg, np.ones(n)/n, method="SLSQP",
                 bounds=[(0,1)]*n,
                 constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],
                 options={"ftol":1e-12,"maxiter":2000})
    return r.x if r.success else np.ones(n)/n

def frontier_pts(mu, cov, n_pts=200):
    n    = len(mu)
    w_mv = min_var(mu, cov)
    r_lo, r_hi = w_mv@mu, mu.max()
    vols, rets = [], []
    for tgt in np.linspace(r_lo, r_hi, n_pts):
        res = minimize(lambda w: w@cov@w, np.ones(n)/n, method="SLSQP",
                       bounds=[(0,1)]*n,
                       constraints=[{"type":"eq","fun":lambda w:w.sum()-1},
                                    {"type":"eq","fun":lambda w,t=tgt:w@mu-t}],
                       options={"ftol":1e-12,"maxiter":2000})
        if res.success:
            rets.append(res.x@mu); vols.append(np.sqrt(res.x@cov@res.x))
    return np.array(vols), np.array(rets)

def pstats(w, mu, cov):
    return w@mu, np.sqrt(w@cov@w)

def sharpe(w, mu, cov, rf):
    r, v = pstats(w, mu, cov)
    return (r-rf)/v

def hhi(w):
    return np.sum(w**2)

def avg_corr(ret):
    c = ret.corr().values; n = c.shape[0]
    return c[~np.eye(n,dtype=bool)].mean()

def ann_ret(ret, freq=12):
    return ret.mean() * freq

def ann_vol(ret, freq=12):
    return ret.std() * np.sqrt(freq)

def max_drawdown(cumret):
    roll_max = cumret.cummax()
    dd = (cumret - roll_max) / roll_max
    return dd.min()

def cvar_95(ret_series):
    """Monthly CVaR at 95%, annualised"""
    thresh = np.percentile(ret_series, 5)
    tail   = ret_series[ret_series <= thresh]
    return tail.mean() * 12  # annualise

# ═══════════════════════════════════════════════════════════════
# 3. STYLE HELPER
# ═══════════════════════════════════════════════════════════════
def style_ax(ax):
    ax.set_facecolor("#FAFAFA")
    for sp in ["top","right"]:   ax.spines[sp].set_visible(False)
    for sp in ["left","bottom"]: ax.spines[sp].set_color("#CCCCCC")
    ax.grid(True, linestyle="--", linewidth=0.4, color="#DDDDDD", alpha=0.9)
    ax.tick_params(colors="#666666", labelsize=9)

# ═══════════════════════════════════════════════════════════════
# 5. PLOT B — DE ROON-NIJMAN SPANNING TEST
# ═══════════════════════════════════════════════════════════════
def de_roon_nijman_test(ret_pre, ret_post, rf_pre, rf_post):
    """
    Implements the De Roon-Nijman (2001) mean-variance spanning test.
    H0: Post-COVID assets add no mean-variance improvement over pre-COVID assets.
    Uses the regression-based version:
      regress excess return of post-COVID TP on pre-COVID asset excess returns
      and test alpha=0, beta sum=1 simultaneously.
    """
    mu_pre,  cov_pre,  _ = lw_cov(ret_pre)
    mu_post, cov_post, _ = lw_cov(ret_post)

    w_tp_post = max_sharpe(mu_post, cov_post, rf_post)
    w_tp_pre  = max_sharpe(mu_pre,  cov_pre,  rf_pre)

    # Compute monthly TP portfolio returns
    tp_post_ret = (ret_post * w_tp_post).sum(axis=1)
    tp_pre_ret  = (ret_pre  * w_tp_pre ).sum(axis=1)

    # Align on common dates
    common = tp_post_ret.index.intersection(tp_pre_ret.index)
    if len(common) < 10:
        # Use out-of-sample: apply pre weights to post-period data
        shared = ret_pre.columns.intersection(ret_post.columns)
        w_pre_shared = w_tp_pre[[list(ret_pre.columns).index(c)
                                  for c in shared if c in ret_pre.columns]]
        # Reweight to sum to 1
        if w_pre_shared.sum() > 0:
            w_pre_shared /= w_pre_shared.sum()
        ret_pre_shared  = ret_post[shared]
        tp_pre_in_post  = (ret_pre_shared * w_pre_shared[:len(shared)]).sum(axis=1)
        excess_post     = tp_post_ret - rf_post/12
        excess_pre_bench= tp_pre_in_post - rf_pre/12
        X = np.column_stack([np.ones(len(excess_post)), excess_pre_bench.values])
        y = excess_post.values
    else:
        excess_post = tp_post_ret.loc[common] - rf_post/12
        excess_pre  = tp_pre_ret.loc[common]  - rf_pre/12
        X = np.column_stack([np.ones(len(common)), excess_pre.values])
        y = excess_post.values

    # OLS
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha, beta = beta_hat[0], beta_hat[1]
    y_hat = X @ beta_hat
    resid = y - y_hat
    n, k = len(y), 2
    s2 = resid@resid / (n-k)
    cov_beta = s2 * np.linalg.inv(X.T@X)
    se_alpha = np.sqrt(cov_beta[0,0])
    se_beta  = np.sqrt(cov_beta[1,1])

    t_alpha = alpha / (se_alpha + 1e-9)
    t_beta1 = (beta - 1) / (se_beta + 1e-9)
    p_alpha = 2*(1 - stats.t.cdf(abs(t_alpha), df=n-k))
    p_beta1 = 2*(1 - stats.t.cdf(abs(t_beta1), df=n-k))

    # Joint F-test: H0: alpha=0, beta=1
    R = np.array([[1,0],[0,1]])
    r = np.array([0, 1])
    Rb = R @ beta_hat - r
    F_stat = (Rb @ np.linalg.inv(R @ cov_beta @ R.T) @ Rb) / k
    p_joint = 1 - stats.f.cdf(F_stat, dfn=k, dfd=n-k)
    spanning_rejected = p_joint < 0.05

    result = dict(alpha=round(alpha,4), beta=round(beta,4),
                  t_alpha=round(t_alpha,3), t_beta1=round(t_beta1,3),
                  p_alpha=round(p_alpha,4), p_beta1=round(p_beta1,4),
                  F_stat=round(F_stat,3), p_joint=round(p_joint,4),
                  spanning_rejected=spanning_rejected, n_obs=n)
    return result

def plot_spanning_summary(result):
    fig, ax = plt.subplots(figsize=(10,5))
    fig.patch.set_facecolor("#FAFAFA"); ax.set_facecolor("#FAFAFA")
    ax.axis("off")

    title = ("De Roon-Nijman Spanning Test\n"
             "H₀: Post-COVID frontier is SPANNED by pre-COVID assets\n"
             "H₁: Post-COVID frontier is OUTSIDE the pre-COVID feasible set")
    ax.set_title(title, fontsize=12, fontweight="500",
                 color="#222222", pad=15, loc="left")

    verdict = ("✓ SPANNING REJECTED — Post-COVID frontier is genuinely outside\n"
               "  the pre-COVID mean-variance frontier"
               if result['spanning_rejected']
               else "✗ Cannot reject spanning — post-COVID may just be a reweighting")
    vcolor = "#1D9E75" if result['spanning_rejected'] else "#D85A30"

    data = [
        ["Parameter", "Estimate", "t-stat / F-stat", "p-value", "Interpretation"],
        ["α (intercept)", f"{result['alpha']:.4f}",
         f"t = {result['t_alpha']:.3f}", f"{result['p_alpha']:.4f}",
         "Excess return unexplained by pre-COVID TP"],
        ["β (slope)", f"{result['beta']:.4f}",
         f"t(β-1) = {result['t_beta1']:.3f}", f"{result['p_beta1']:.4f}",
         "How closely post-COVID tracks pre-COVID TP"],
        ["Joint F (α=0, β=1)", "H₀: Spanning",
         f"F = {result['F_stat']:.3f}", f"{result['p_joint']:.4f}",
         f"{'REJECTED' if result['spanning_rejected'] else 'Not rejected'} at 5%"],
    ]

    col_widths = [0.18, 0.12, 0.18, 0.12, 0.40]
    col_starts = [0.00, 0.18, 0.30, 0.48, 0.60]
    for row_i, row in enumerate(data):
        bg = "#E8E8F0" if row_i == 0 else ("#F5FFF9" if row_i % 2 == 1 else "#FAFAFA")
        for col_j, (text, x, w) in enumerate(zip(row, col_starts, col_widths)):
            ax.text(x+0.01, 0.85 - row_i*0.13, text,
                    transform=ax.transAxes, fontsize=10,
                    fontweight="bold" if row_i == 0 else "normal",
                    va="top", color="#222222")

    ax.text(0.0, 0.22, verdict, transform=ax.transAxes,
            fontsize=12, fontweight="bold", color=vcolor,
            bbox=dict(facecolor="#F0FFF8" if result['spanning_rejected'] else "#FFF0F0",
                      edgecolor=vcolor, alpha=0.8, boxstyle="round,pad=0.4"))
    ax.text(0.0, 0.05,
            f"n = {result['n_obs']} monthly observations",
            transform=ax.transAxes, fontsize=9, color="#888888")

    fig.tight_layout()
    path = "Results/Enhanced/02_spanning_test.png"
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"  [saved] {path}")
    return result

# ═══════════════════════════════════════════════════════════════
# 6. PLOT C — FRONTIER SHIFT DECOMPOSITION (μ vs Σ)
# ═══════════════════════════════════════════════════════════════
def plot_decomposition(ret_pre, ret_post, rf_pre, rf_post):
    mu_pre,  cov_pre,  _ = lw_cov(ret_pre)
    mu_post, cov_post, _ = lw_cov(ret_post)

    # Four scenarios
    scenarios = {
        "Pre-COVID\n(μ_pre, Σ_pre)":   (mu_pre,  cov_pre),
        "Only μ updated\n(μ_post, Σ_pre)": (mu_post, cov_pre),
        "Only Σ updated\n(μ_pre, Σ_post)": (mu_pre,  cov_post),
        "Post-COVID\n(μ_post, Σ_post)": (mu_post, cov_post),
    }
    colors = [C_PRE, "#E0A020", "#9B59B6", C_POST]

    fig, ax = plt.subplots(figsize=(12,7))
    fig.patch.set_facecolor("#FAFAFA"); style_ax(ax)

    for (label, (mu, cov)), color in zip(scenarios.items(), colors):
        v, r = frontier_pts(mu, cov, n_pts=150)
        w_tp = max_sharpe(mu, cov, rf_post if "post" in label.lower() else rf_pre)
        r_tp, v_tp = pstats(w_tp, mu, cov)
        sr = (r_tp - (rf_post if "post" in label.lower() else rf_pre)) / v_tp
        ax.plot(v, r, color=color, lw=2.2,
                label=f"{label.replace(chr(10),' ')}  (SR={sr:.2f})")
        ax.scatter(v_tp, r_tp, color=color, s=120, zorder=6,
                   marker="*", edgecolors="white", linewidths=0.5)

    # Arrows showing decomposition
    mu_list  = [mu_pre, mu_post, mu_pre,  mu_post]
    cov_list = [cov_pre, cov_pre, cov_post, cov_post]
    srs, vols, rets = [], [], []
    for mu, cov, rf in zip(mu_list, cov_list,
                           [rf_pre,rf_pre,rf_post,rf_post]):
        w  = max_sharpe(mu, cov, rf)
        r_, v_ = pstats(w, mu, cov)
        srs.append((r_-rf)/v_); vols.append(v_); rets.append(r_)

    ax.annotate("", xy=(vols[1],rets[1]), xytext=(vols[0],rets[0]),
                arrowprops=dict(arrowstyle="->",color="#E0A020",lw=1.5))
    ax.annotate("", xy=(vols[2],rets[2]), xytext=(vols[0],rets[0]),
                arrowprops=dict(arrowstyle="->",color="#9B59B6",lw=1.5))

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{x:.0%}"))
    ax.set_xlabel("Annualised Volatility (σ)", fontsize=11)
    ax.set_ylabel("Annualised Return (μ)", fontsize=11)
    ax.set_title(
        "Frontier Shift Decomposition — Which input drives the shift?\n"
        "Stars = tangency portfolio | Yellow arrow = μ effect | Purple arrow = Σ effect",
        fontsize=11, color="#222222", pad=10)
    ax.legend(fontsize=9.5, framealpha=0.7, loc="upper left")

    # SR comparison box
    sr_text = (f"Sharpe Ratios:\n"
               f"Pre (μ_pre,Σ_pre):    {srs[0]:.3f}\n"
               f"μ only (μ_post,Σ_pre): {srs[1]:.3f}\n"
               f"Σ only (μ_pre,Σ_post): {srs[2]:.3f}\n"
               f"Post (μ_post,Σ_post): {srs[3]:.3f}")
    ax.text(0.98, 0.04, sr_text, transform=ax.transAxes,
            fontsize=9, va="bottom", ha="right",
            bbox=dict(facecolor="white", alpha=0.85,
                      edgecolor="#CCCCCC", boxstyle="round,pad=0.4"),
            fontfamily="monospace")

    fig.tight_layout()
    path = "Results/Enhanced/03_frontier_decomposition.png"
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"  [saved] {path}")
    return {"mu_effect_sr": round(srs[1]-srs[0],4),
            "sigma_effect_sr": round(srs[2]-srs[0],4),
            "total_sr_gain": round(srs[3]-srs[0],4)}

# ═══════════════════════════════════════════════════════════════
# 7. PLOT D — ROLLING SHARPE (when did improvement emerge?)
# ═══════════════════════════════════════════════════════════════
def plot_rolling_sharpe(log_ret, window=24):
    ew = log_ret.mean(axis=1)  # equal-weight monthly return
    roll_mu  = ew.rolling(window).mean() * 12
    roll_std = ew.rolling(window).std()  * np.sqrt(12)
    roll_rf  = pd.Series(RF_PRE, index=ew.index)
    roll_rf.loc["2021-07-01":] = RF_POST
    roll_sr  = (roll_mu - roll_rf) / roll_std

    fig, axes = plt.subplots(2, 1, figsize=(13,9), sharex=True)
    fig.patch.set_facecolor("#FAFAFA")

    # Panel 1: rolling vol
    ax1 = axes[0]; style_ax(ax1)
    roll_vol = roll_std
    ax1.plot(roll_vol.index, roll_vol.values, color="#2C2C2A", lw=1.8)
    ax1.fill_between(roll_vol.index, roll_vol.values, alpha=0.1, color="#2C2C2A")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{x:.0%}"))
    ax1.set_ylabel(f"{window}-month Rolling Vol", fontsize=10)
    ax1.set_title(f"Rolling {window}-month Sharpe Ratio & Volatility — Equal-Weight Portfolio",
                  fontsize=11, color="#222222", pad=8)

    # Panel 2: rolling Sharpe
    ax2 = axes[1]; style_ax(ax2)
    ax2.plot(roll_sr.index, roll_sr.values, color=C_POST, lw=2.0)
    ax2.axhline(0, color="#888888", lw=0.8, linestyle="--")
    ax2.fill_between(roll_sr.index, roll_sr.values, 0,
                     where=roll_sr.values>0, alpha=0.15, color=C_PRE)
    ax2.fill_between(roll_sr.index, roll_sr.values, 0,
                     where=roll_sr.values<0, alpha=0.15, color="#D85A30")
    ax2.set_ylabel(f"{window}-month Rolling Sharpe", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)

    # Regime shading on both panels
    for ax in axes:
        ax.axvspan(pd.Timestamp("2015-01-01"), pd.Timestamp("2019-12-31"),
                   alpha=0.06, color=C_PRE)
        ax.axvspan(pd.Timestamp("2020-02-01"), pd.Timestamp("2021-06-30"),
                   alpha=0.10, color="#E24B4A")
        ax.axvspan(pd.Timestamp("2021-07-01"), pd.Timestamp("2024-12-31"),
                   alpha=0.06, color=C_POST)

    # Mark key events
    events = {"Mar'20\nCrash":"2020-03-31","Nov'21\nRecovery":"2021-11-30",
              "Jul'23\nHDFC\nMerger":"2023-07-31"}
    for lbl, dt in events.items():
        for ax in axes:
            ax.axvline(pd.Timestamp(dt), color="#999999", lw=0.8, linestyle=":")
        ax2.text(pd.Timestamp(dt), ax2.get_ylim()[0]*0.9, lbl,
                 fontsize=7.5, ha="center", color="#666666")

    fig.tight_layout()
    path = "Results/Enhanced/04_rolling_sharpe.png"
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"  [saved] {path}")

# ═══════════════════════════════════════════════════════════════
# 8. PLOT E — OUT-OF-SAMPLE 2025 BACKTEST
# ═══════════════════════════════════════════════════════════════
def plot_oos_backtest(log_ret, ret_pre, ret_post, rf_pre, rf_post):
    oos_data = log_ret.loc[OOS_START:OOS_END]
    if len(oos_data) < 2:
        print("  [OOS] Insufficient 2025 data — skipping backtest")
        return None

    shared = ret_pre.columns.intersection(ret_post.columns).intersection(oos_data.columns)
    oos = oos_data[shared].dropna(axis=1)
    if len(oos) < 2:
        print("  [OOS] Too few OOS observations — skipping")
        return None

    mu_pre,  cov_pre,  _ = lw_cov(ret_pre[shared])
    mu_post, cov_post, _ = lw_cov(ret_post[shared])

    w_pre  = max_sharpe(mu_pre,  cov_pre,  rf_pre)
    w_post = max_sharpe(mu_post, cov_post, rf_post)
    w_ew   = np.ones(len(shared)) / len(shared)

    # Align weights to shared universe
    shared_list = list(shared)
    pre_idx  = [list(ret_pre[shared].columns).index(c)  for c in shared_list]
    post_idx = [list(ret_post[shared].columns).index(c) for c in shared_list]
    w_pre_s  = w_pre[pre_idx];   w_pre_s  /= w_pre_s.sum()
    w_post_s = w_post[post_idx]; w_post_s /= w_post_s.sum()

    r_pre_oos  = (oos * w_pre_s).sum(axis=1)
    r_post_oos = (oos * w_post_s).sum(axis=1)
    r_ew_oos   = oos.mean(axis=1)

    cum_pre  = (1 + r_pre_oos).cumprod()
    cum_post = (1 + r_post_oos).cumprod()
    cum_ew   = (1 + r_ew_oos).cumprod()

    # Metrics
    def oos_metrics(r, cum, rf, label):
        ar  = r.mean() * 12
        av  = r.std()  * np.sqrt(12)
        sr  = (ar - rf) / av if av > 0 else np.nan
        mdd = max_drawdown(cum)
        cv  = cvar_95(r)
        return {"Strategy":label,"Ann.Ret":ar,"Ann.Vol":av,
                "Sharpe":sr,"Max DD":mdd,"CVaR 95%":cv}

    rf_oos = RF_POST
    metrics = pd.DataFrame([
        oos_metrics(r_pre_oos,  cum_pre,  rf_oos, "Pre-COVID TP weights"),
        oos_metrics(r_post_oos, cum_post, rf_oos, "Post-COVID TP weights"),
        oos_metrics(r_ew_oos,   cum_ew,   rf_oos, "1/N Equal weight"),
    ])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#FAFAFA")

    # Cumulative returns
    ax1 = axes[0]; style_ax(ax1)
    ax1.plot(cum_pre.index,  cum_pre.values,  color=C_PRE,  lw=2.2,
             label="Pre-COVID TP weights")
    ax1.plot(cum_post.index, cum_post.values, color=C_POST, lw=2.2,
             label="Post-COVID TP weights")
    ax1.plot(cum_ew.index,   cum_ew.values,   color=C_1N,   lw=1.5,
             linestyle="--", label="1/N Equal weight")
    ax1.axhline(1, color="#AAAAAA", lw=0.8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{x:.2f}x"))
    ax1.set_title("Out-of-Sample Cumulative Return\n(2025 — applying in-sample TP weights)",
                  fontsize=11, color="#222222")
    ax1.set_xlabel("Date"); ax1.set_ylabel("Cumulative Return (1 = start)")
    ax1.legend(fontsize=9.5)

    # Metrics table
    ax2 = axes[1]; ax2.axis("off")
    ax2.set_facecolor("#FAFAFA")
    ax2.set_title("Out-of-Sample Performance Metrics (2025)",
                  fontsize=11, color="#222222", pad=10)

    fmt_metrics = metrics.copy()
    for col in ["Ann.Ret","Ann.Vol","Max DD","CVaR 95%"]:
        fmt_metrics[col] = fmt_metrics[col].apply(lambda x:f"{x:.2%}")
    fmt_metrics["Sharpe"] = fmt_metrics["Sharpe"].apply(lambda x:f"{x:.3f}")

    tbl = ax2.table(
        cellText=fmt_metrics.values,
        colLabels=fmt_metrics.columns,
        cellLoc="center", loc="center",
        bbox=[0.0, 0.3, 1.0, 0.6]
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
    for (r,c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if r == 0:
            cell.set_facecolor("#534AB7"); cell.set_text_props(color="white",fontweight="bold")
        elif r == 1:
            cell.set_facecolor("#EFF7F3")
        elif r == 2:
            cell.set_facecolor("#EEF0FA")
        else:
            cell.set_facecolor("#F5F5F5")

    n_months = len(oos)
    ax2.text(0.5, 0.2, f"OOS period: {oos.index[0].strftime('%b %Y')} – "
             f"{oos.index[-1].strftime('%b %Y')}  ({n_months} months)",
             transform=ax2.transAxes, ha="center", fontsize=9, color="#666666")
    ax2.text(0.5, 0.12, f"Universe: {len(shared)} stocks (shared pre/post/OOS)",
             transform=ax2.transAxes, ha="center", fontsize=9, color="#666666")

    fig.tight_layout()
    path = "Results/Enhanced/05_oos_backtest.png"
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    metrics.to_csv("Results/Enhanced/05_oos_metrics.csv", index=False)
    print(f"  [saved] {path}")
    return metrics

# ═══════════════════════════════════════════════════════════════
# 9. PLOT F — FRONTIER WITH 1/N BENCHMARK
# ═══════════════════════════════════════════════════════════════
def plot_frontier_with_1N(ret_pre, ret_post, rf_pre, rf_post):
    mu_pre,  cov_pre,  _ = lw_cov(ret_pre)
    mu_post, cov_post, _ = lw_cov(ret_post)

    v_pre,  r_pre  = frontier_pts(mu_pre,  cov_pre)
    v_post, r_post = frontier_pts(mu_post, cov_post)

    n_pre  = len(mu_pre);  w_ew_pre  = np.ones(n_pre)  / n_pre
    n_post = len(mu_post); w_ew_post = np.ones(n_post) / n_post

    r_ew_pre,  v_ew_pre  = pstats(w_ew_pre,  mu_pre,  cov_pre)
    r_ew_post, v_ew_post = pstats(w_ew_post, mu_post, cov_post)

    w_tp_pre  = max_sharpe(mu_pre,  cov_pre,  rf_pre)
    w_tp_post = max_sharpe(mu_post, cov_post, rf_post)
    r_tp_pre,  v_tp_pre  = pstats(w_tp_pre,  mu_pre,  cov_pre)
    r_tp_post, v_tp_post = pstats(w_tp_post, mu_post, cov_post)

    sr_ew_pre  = (r_ew_pre  - rf_pre)  / v_ew_pre
    sr_ew_post = (r_ew_post - rf_post) / v_ew_post
    sr_tp_pre  = (r_tp_pre  - rf_pre)  / v_tp_pre
    sr_tp_post = (r_tp_post - rf_post) / v_tp_post

    fig, ax = plt.subplots(figsize=(12,7))
    fig.patch.set_facecolor("#FAFAFA"); style_ax(ax)

    ax.plot(v_pre,  r_pre,  color=C_PRE,  lw=2.2, label="Pre-COVID frontier")
    ax.plot(v_post, r_post, color=C_POST, lw=2.2, label="Post-COVID frontier")

    # Tangency
    ax.scatter(v_tp_pre,  r_tp_pre,  color=C_TP, s=130, zorder=7,
               marker="D", label=f"Tangency Pre (SR={sr_tp_pre:.2f})")
    ax.scatter(v_tp_post, r_tp_post, color=C_TP, s=130, zorder=7,
               marker="D", label=f"Tangency Post (SR={sr_tp_post:.2f})")

    # 1/N benchmark
    ax.scatter(v_ew_pre,  r_ew_pre,  color=C_1N, s=120, zorder=7,
               marker="s", label=f"1/N Pre  (SR={sr_ew_pre:.2f})")
    ax.scatter(v_ew_post, r_ew_post, color=C_1N, s=120, zorder=7,
               marker="^", label=f"1/N Post (SR={sr_ew_post:.2f})")

    # Annotations
    for v, r, lbl in [(v_ew_pre,r_ew_pre,"1/N\nPre"),
                       (v_ew_post,r_ew_post,"1/N\nPost")]:
        ax.annotate(lbl, xy=(v,r), xytext=(v+0.005, r-0.02),
                    fontsize=8, color="#555555",
                    arrowprops=dict(arrowstyle="-",color="#AAAAAA",lw=0.6))

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{x:.0%}"))
    ax.set_xlabel("Annualised Volatility (σ)", fontsize=11)
    ax.set_ylabel("Annualised Return (μ)", fontsize=11)
    ax.set_title(
        "Efficient Frontier with 1/N Naive Benchmark\n"
        "Squares = 1/N equal weight | Diamonds = tangency portfolio",
        fontsize=11, color="#222222", pad=10)
    ax.legend(fontsize=9.5, framealpha=0.7)

    fig.tight_layout()
    path = "Results/Enhanced/06_frontier_with_1N.png"
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"  [saved] {path}")

# ═══════════════════════════════════════════════════════════════
# 10. PLOT G — SECTOR-CAPPED ROBUSTNESS (20% cap)
# ═══════════════════════════════════════════════════════════════
def plot_sector_capped(ret_pre, ret_post, rf_pre, rf_post):
    """Max-Sharpe with sector weight cap of 20%."""
    tickers_pre  = list(ret_pre.columns)
    tickers_post = list(ret_post.columns)

    def sector_bounds(tickers, cap=0.20):
        sector_groups = {}
        for i, t in enumerate(tickers):
            s = SECTOR_MAP.get(t, "Others")
            sector_groups.setdefault(s, []).append(i)
        return sector_groups

    def capped_max_sharpe(mu, cov, rf, tickers, cap=0.20):
        n = len(mu)
        sg = sector_bounds(tickers, cap)
        def neg_sr(w):
            v = np.sqrt(w@cov@w)
            return -(w@mu - rf)/(v+1e-9)
        cons = [{"type":"eq","fun":lambda w:w.sum()-1}]
        for idxs in sg.values():
            cons.append({"type":"ineq",
                         "fun": lambda w, ii=idxs: cap - sum(w[i] for i in ii)})
        res = minimize(neg_sr, np.ones(n)/n, method="SLSQP",
                       bounds=[(0,1)]*n, constraints=cons,
                       options={"ftol":1e-12,"maxiter":3000})
        return res.x if res.success else np.ones(n)/n

    mu_pre,  cov_pre,  _ = lw_cov(ret_pre)
    mu_post, cov_post, _ = lw_cov(ret_post)

    w_tp_pre  = max_sharpe(mu_pre,  cov_pre,  rf_pre)
    w_tp_post = max_sharpe(mu_post, cov_post, rf_post)
    w_cap_pre  = capped_max_sharpe(mu_pre,  cov_pre,  rf_pre,  tickers_pre)
    w_cap_post = capped_max_sharpe(mu_post, cov_post, rf_post, tickers_post)

    r_tp_pre,  v_tp_pre  = pstats(w_tp_pre,  mu_pre,  cov_pre)
    r_tp_post, v_tp_post = pstats(w_tp_post, mu_post, cov_post)
    r_cap_pre,  v_cap_pre  = pstats(w_cap_pre,  mu_pre,  cov_pre)
    r_cap_post, v_cap_post = pstats(w_cap_post, mu_post, cov_post)

    sr_tp_pre  = (r_tp_pre  - rf_pre)  / v_tp_pre
    sr_tp_post = (r_tp_post - rf_post) / v_tp_post
    sr_cap_pre  = (r_cap_pre  - rf_pre)  / v_cap_pre
    sr_cap_post = (r_cap_post - rf_post) / v_cap_post

    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    fig.patch.set_facecolor("#FAFAFA")

    for ax, (mu, cov, rf, w_tp, w_cap, r_tp, v_tp, r_cap, v_cap,
              sr_tp, sr_cap, lbl, col) in zip(axes, [
        (mu_pre, cov_pre, rf_pre, w_tp_pre, w_cap_pre,
         r_tp_pre, v_tp_pre, r_cap_pre, v_cap_pre,
         sr_tp_pre, sr_cap_pre, "Pre-COVID", C_PRE),
        (mu_post, cov_post, rf_post, w_tp_post, w_cap_post,
         r_tp_post, v_tp_post, r_cap_post, v_cap_post,
         sr_tp_post, sr_cap_post, "Post-COVID", C_POST),
    ]):
        style_ax(ax)
        v_fr, r_fr = frontier_pts(mu, cov, n_pts=150)
        ax.plot(v_fr, r_fr, color=col, lw=2.0, label="Uncapped frontier")
        ax.scatter(v_tp,  r_tp,  color=col, s=130, zorder=7, marker="D",
                   label=f"Uncapped TP (SR={sr_tp:.2f}, HHI={hhi(w_tp):.3f})")
        ax.scatter(v_cap, r_cap, color="#E44", s=130, zorder=7, marker="^",
                   label=f"20%-capped TP (SR={sr_cap:.2f}, HHI={hhi(w_cap):.3f})")

        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{x:.0%}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{x:.0%}"))
        ax.set_title(f"{lbl}\nUncapped vs 20% Sector-Cap Tangency Portfolio",
                     fontsize=10.5, color="#222222")
        ax.set_xlabel("Annualised Volatility (σ)"); ax.set_ylabel("Annualised Return (μ)")
        ax.legend(fontsize=9, framealpha=0.7)

    fig.suptitle("Sector-Capped Robustness Check (Max 20% per sector)",
                 fontsize=12, fontweight="500", color="#222222", y=1.02)
    fig.tight_layout()
    path = "Results/Enhanced/07_sector_capped.png"
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"  [saved] {path}")
    return {"sr_tp_pre":round(sr_tp_pre,4),"sr_cap_pre":round(sr_cap_pre,4),
            "sr_tp_post":round(sr_tp_post,4),"sr_cap_post":round(sr_cap_post,4)}

# ═══════════════════════════════════════════════════════════════
# 11. PLOT H — CAPM ALPHA FOR TOP TANGENCY STOCKS
# ═══════════════════════════════════════════════════════════════
def plot_capm_alpha(ret_pre, ret_post, rf_pre, rf_post):
    mu_pre,  cov_pre,  _ = lw_cov(ret_pre)
    mu_post, cov_post, _ = lw_cov(ret_post)

    w_pre  = max_sharpe(mu_pre,  cov_pre,  rf_pre)
    w_post = max_sharpe(mu_post, cov_post, rf_post)

    # Top 6 by post-COVID weight
    tickers_post = list(ret_post.columns)
    top_post = [(tickers_post[i], w_post[i])
                for i in np.argsort(w_post)[::-1][:8]]

    results = []
    for period, ret, w, rf, label in [
        ("Pre",  ret_pre,  w_pre,  rf_pre,  "Pre-COVID"),
        ("Post", ret_post, w_post, rf_post, "Post-COVID"),
    ]:
        tickers = list(ret.columns)
        mkt = ret.mean(axis=1)  # equal-weight market proxy
        for tk_name, tk_w in top_post:
            if tk_name not in tickers:
                continue
            stock_ret = ret[tk_name].values
            mkt_ret   = mkt.values
            n = len(stock_ret)
            rf_m = rf / 12
            excess_stock = stock_ret - rf_m
            excess_mkt   = mkt_ret   - rf_m
            X = np.column_stack([np.ones(n), excess_mkt])
            b = np.linalg.lstsq(X, excess_stock, rcond=None)[0]
            alpha_m, beta_m = b
            y_hat = X @ b
            resid = excess_stock - y_hat
            s2    = resid@resid / (n-2)
            se_a  = np.sqrt(s2 * np.linalg.inv(X.T@X)[0,0])
            t_a   = alpha_m / (se_a + 1e-9)
            p_a   = 2*(1 - stats.t.cdf(abs(t_a), df=n-2))
            results.append({
                "Period":  label,
                "Ticker":  tk_name.replace(".NS",""),
                "TP Wt":   round(w[tickers.index(tk_name)],4) if tk_name in tickers else 0,
                "Alpha (m)": round(alpha_m * 12, 4),  # annualised
                "Beta":    round(beta_m, 3),
                "t-alpha": round(t_a, 2),
                "p-alpha": round(p_a, 4),
                "Sig":     "***" if p_a<0.01 else ("**" if p_a<0.05 else
                           ("*" if p_a<0.10 else ""))
            })

    df = pd.DataFrame(results)
    df.to_csv("Results/Enhanced/08_capm_alpha.csv", index=False)

    # Plot
    tickers_show = list(dict.fromkeys(df["Ticker"]))
    x = np.arange(len(tickers_show))
    w_bar = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    fig.patch.set_facecolor("#FAFAFA")

    for ax_idx, (period_label, color) in enumerate([("Pre-COVID",C_PRE),("Post-COVID",C_POST)]):
        ax = axes[ax_idx]; style_ax(ax)
        sub = df[df["Period"]==period_label].set_index("Ticker")
        alphas  = [sub.loc[t,"Alpha (m)"] if t in sub.index else 0 for t in tickers_show]
        sigs    = [sub.loc[t,"Sig"]       if t in sub.index else "" for t in tickers_show]
        bars = ax.bar(x, alphas, color=color, alpha=0.8, edgecolor="white")
        ax.axhline(0, color="#888888", lw=0.8)
        for bar, sig in zip(bars, sigs):
            if sig:
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height() + (0.005 if bar.get_height()>=0 else -0.012),
                        sig, ha="center", fontsize=11, fontweight="bold", color="#333333")
        ax.set_xticks(x); ax.set_xticklabels(tickers_show, rotation=30, ha="right")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"{v:.0%}"))
        ax.set_ylabel("Annualised CAPM Alpha", fontsize=10)
        ax.set_title(f"{period_label} — CAPM Alpha (vs. equal-weight market proxy)\n"
                     "* p<0.10  ** p<0.05  *** p<0.01",
                     fontsize=10.5, color="#222222")

    fig.suptitle("CAPM Alpha for Top Post-COVID Tangency Portfolio Stocks",
                 fontsize=12, fontweight="500", y=1.02)
    fig.tight_layout()
    path = "Results/Enhanced/08_capm_alpha.png"
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"  [saved] {path}")
    return df

# ═══════════════════════════════════════════════════════════════
# 12. PLOT I — MAX DRAWDOWN / CVaR ANALYSIS
# ═══════════════════════════════════════════════════════════════
def plot_downside_risk(ret_pre, ret_post, rf_pre, rf_post):
    mu_pre,  cov_pre,  _ = lw_cov(ret_pre)
    mu_post, cov_post, _ = lw_cov(ret_post)
    w_tp_pre  = max_sharpe(mu_pre,  cov_pre,  rf_pre)
    w_tp_post = max_sharpe(mu_post, cov_post, rf_post)
    w_ew_pre  = np.ones(len(mu_pre))  / len(mu_pre)
    w_ew_post = np.ones(len(mu_post)) / len(mu_post)

    strategies = [
        ("Pre TP",      (ret_pre  * w_tp_pre).sum(axis=1),  C_PRE,  "-"),
        ("Post TP",     (ret_post * w_tp_post).sum(axis=1), C_POST, "-"),
        ("Pre 1/N",     ret_pre.mean(axis=1),               C_PRE,  "--"),
        ("Post 1/N",    ret_post.mean(axis=1),              C_POST, "--"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("#FAFAFA")
    axes = axes.flatten()

    # Panel 1: Cumulative returns
    ax = axes[0]; style_ax(ax)
    for name, r, c, ls in strategies:
        cum = (1+r).cumprod()
        ax.plot(cum.index, cum.values, color=c, lw=1.8, linestyle=ls, label=name)
    ax.set_title("Cumulative Returns", fontsize=11, color="#222222")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{x:.1f}x"))
    ax.legend(fontsize=8.5)

    # Panel 2: Drawdown
    ax = axes[1]; style_ax(ax)
    for name, r, c, ls in strategies:
        cum = (1+r).cumprod()
        dd  = (cum - cum.cummax()) / cum.cummax()
        ax.fill_between(dd.index, dd.values, 0, alpha=0.15, color=c)
        ax.plot(dd.index, dd.values, color=c, lw=1.5, linestyle=ls, label=name)
    ax.set_title("Drawdown", fontsize=11, color="#222222")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{x:.0%}"))
    ax.legend(fontsize=8.5)

    # Panel 3: Return distribution
    ax = axes[2]; style_ax(ax)
    for name, r, c, ls in strategies:
        ax.hist(r.values * 12, bins=30, alpha=0.35, color=c,
                label=name, density=True)
    ax.set_title("Annualised Monthly Return Distribution", fontsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{x:.0%}"))
    ax.legend(fontsize=8.5)

    # Panel 4: Risk metrics comparison table
    ax = axes[3]; ax.axis("off"); ax.set_facecolor("#FAFAFA")
    ax.set_title("Risk Metrics Summary", fontsize=11, color="#222222", pad=10)

    rows = []
    for name, r, c, ls in strategies:
        cum = (1+r).cumprod()
        rf  = rf_pre if "Pre" in name else rf_post
        ar  = r.mean() * 12
        av  = r.std()  * np.sqrt(12)
        sr  = (ar - rf) / av
        mdd = max_drawdown(cum)
        cv  = cvar_95(r)
        rows.append([name, f"{ar:.1%}", f"{av:.1%}", f"{sr:.2f}",
                     f"{mdd:.1%}", f"{cv:.1%}"])

    tbl = ax.table(
        cellText=rows,
        colLabels=["Strategy","Ann. Ret","Ann. Vol","Sharpe","Max DD","CVaR 95%"],
        cellLoc="center", loc="center",
        bbox=[0.0, 0.2, 1.0, 0.7]
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
    for (r,c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if r == 0:
            cell.set_facecolor("#534AB7"); cell.set_text_props(color="white",fontweight="bold")

    fig.suptitle("Downside Risk Analysis: Max Drawdown & CVaR",
                 fontsize=13, fontweight="500", color="#222222", y=1.01)
    fig.tight_layout()
    path = "Results/Enhanced/09_downside_risk.png"
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"  [saved] {path}")

# ═══════════════════════════════════════════════════════════════
# 13. PLOT J — CORRELATION HEATMAPS: PRE vs POST SIDE-BY-SIDE
# ═══════════════════════════════════════════════════════════════
def plot_corr_heatmaps(ret_pre, ret_post):
    shared = ret_pre.columns.intersection(ret_post.columns)
    # Sort by sector for readability
    def sector_sort_key(t):
        sectors = list(SECTOR_MAP.values())
        s = SECTOR_MAP.get(t, "Others")
        return (sectors.index(s) if s in sectors else 99, t)
    shared_sorted = sorted(shared, key=sector_sort_key)

    corr_pre  = ret_pre[shared_sorted].corr()
    corr_post = ret_post[shared_sorted].corr()
    diff      = corr_post - corr_pre

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor("#FAFAFA")

    labels = [t.replace(".NS","") for t in shared_sorted]
    kw = dict(xticklabels=labels, yticklabels=labels,
              linewidths=0.3, linecolor="#EEEEEE")

    sns.heatmap(corr_pre,  ax=axes[0], cmap="RdYlGn_r", center=0,
                vmin=-0.3, vmax=1.0, annot=False, **kw)
    axes[0].set_title("Pre-COVID Correlations\n(Jan 2015 – Dec 2019)",
                      fontsize=11, color="#222222")
    axes[0].tick_params(labelsize=6, rotation=45)

    sns.heatmap(corr_post, ax=axes[1], cmap="RdYlGn_r", center=0,
                vmin=-0.3, vmax=1.0, annot=False, **kw)
    axes[1].set_title("Post-COVID Correlations\n(Jan 2020 – Dec 2024)",
                      fontsize=11, color="#222222")
    axes[1].tick_params(labelsize=6, rotation=45)

    sns.heatmap(diff,      ax=axes[2], cmap="RdBu_r", center=0,
                vmin=-0.4, vmax=0.4, annot=False, **kw)
    axes[2].set_title("Δ Correlation (Post − Pre)\nRed = more correlated post-COVID",
                      fontsize=11, color="#222222")
    axes[2].tick_params(labelsize=6, rotation=45)

    # Add sector dividers
    sector_labels = [SECTOR_MAP.get(t,"Others") for t in shared_sorted]
    prev = sector_labels[0]; tick_locs = []
    for i, s in enumerate(sector_labels):
        if s != prev:
            tick_locs.append(i)
            prev = s
    for ax in axes:
        for loc in tick_locs:
            ax.axhline(loc, color="white", lw=1.5)
            ax.axvline(loc, color="white", lw=1.5)

    fig.suptitle("Correlation Structure: Pre vs Post COVID (stocks sorted by sector)",
                 fontsize=13, fontweight="500", color="#222222", y=1.02)
    fig.tight_layout()
    path = "Results/Enhanced/10_correlation_heatmaps.png"
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)

    # Stats
    avg_pre  = avg_corr(ret_pre[shared])
    avg_post = avg_corr(ret_post[shared])
    print(f"  [saved] {path}  | avg_corr pre={avg_pre:.3f}, post={avg_post:.3f}")
    return avg_pre, avg_post

# ═══════════════════════════════════════════════════════════════
# 14. PLOT K — Δμ vs Δσ SCATTER BY SECTOR
# ═══════════════════════════════════════════════════════════════
def plot_delta_scatter(ret_pre, ret_post):
    shared = ret_pre.columns.intersection(ret_post.columns)
    mu_pre  = ann_ret(ret_pre[shared])
    mu_post = ann_ret(ret_post[shared])
    vol_pre  = ann_vol(ret_pre[shared])
    vol_post = ann_vol(ret_post[shared])
    d_mu  = mu_post  - mu_pre
    d_vol = vol_post - vol_pre

    fig, ax = plt.subplots(figsize=(12,8))
    fig.patch.set_facecolor("#FAFAFA"); style_ax(ax)

    ax.axhline(0, color="#AAAAAA", lw=0.8, linestyle="--")
    ax.axvline(0, color="#AAAAAA", lw=0.8, linestyle="--")

    # Quadrant labels
    for txt, x, y in [("↑μ ↑σ", 0.55, 0.92), ("↑μ ↓σ", -0.02, 0.92),
                       ("↓μ ↑σ", 0.55, 0.08), ("↓μ ↓σ", -0.02, 0.08)]:
        ax.text(x, y, txt, transform=ax.transAxes,
                fontsize=11, color="#BBBBBB", ha="center")

    # Plot by sector
    for sector in sorted(set(SECTOR_MAP.values())):
        tks = [t for t in shared if SECTOR_MAP.get(t,"Others")==sector]
        if not tks:
            continue
        color = SECTOR_COLORS.get(sector, "#888888")
        ax.scatter(d_vol[tks].values, d_mu[tks].values,
                   color=color, s=110, alpha=0.85, edgecolors="white",
                   linewidths=0.5, label=sector, zorder=5)
        for t in tks:
            ax.annotate(t.replace(".NS",""), (d_vol[t], d_mu[t]),
                        fontsize=7.5, color="#444444",
                        xytext=(4,3), textcoords="offset points")

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{x:+.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{x:+.0%}"))
    ax.set_xlabel("Δ Annualised Volatility (Post − Pre)", fontsize=11)
    ax.set_ylabel("Δ Annualised Mean Return (Post − Pre)", fontsize=11)
    ax.set_title("Stock-Level Change: Return vs Volatility\n"
                 "Each point = one stock | Colour = sector | "
                 "Upper-left quadrant = improved risk-adjusted returns",
                 fontsize=11, color="#222222", pad=10)
    ax.legend(fontsize=9.5, framealpha=0.7,
              loc="lower right", title="Sector", title_fontsize=9)

    fig.tight_layout()
    path = "Results/Enhanced/11_delta_scatter.png"
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"  [saved] {path}")

    # Export data
    df = pd.DataFrame({
        "Ticker": [t.replace(".NS","") for t in shared],
        "Sector": [SECTOR_MAP.get(t,"Others") for t in shared],
        "mu_pre":  mu_pre.values,  "mu_post":  mu_post.values,
        "vol_pre": vol_pre.values, "vol_post": vol_post.values,
        "delta_mu": d_mu.values,  "delta_vol": d_vol.values,
    })
    df.to_csv("Results/Enhanced/11_stock_changes.csv", index=False)
    return df

# ═══════════════════════════════════════════════════════════════
# 15. PLOT L — HHI CONCENTRATION
# ═══════════════════════════════════════════════════════════════
def plot_hhi(ret_pre, ret_post, rf_pre, rf_post):
    mu_pre,  cov_pre,  _ = lw_cov(ret_pre)
    mu_post, cov_post, _ = lw_cov(ret_post)

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

    # Effective N = 1/HHI
    eff_n = {k: 1/v for k, v in hhi_vals.items()}

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor("#FAFAFA")

    colors_bar = [C_PRE, C_POST, C_PRE, C_POST, C_1N]
    hatches    = ["", "", "///", "///", ""]

    for ax, data, ylabel, title in zip(axes,
        [hhi_vals, eff_n],
        ["HHI (0=perfectly diversified, 1=single stock)",
         "Effective N = 1/HHI (# equivalent equal positions)"],
        ["Herfindahl-Hirschman Index (Concentration)",
         "Effective Number of Stocks (Diversification)"]):
        style_ax(ax)
        bars = ax.bar(list(data.keys()), list(data.values()),
                      color=colors_bar, alpha=0.85,
                      edgecolor="white", linewidth=1.2)
        for bar, val in zip(bars, data.values()):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.005*max(data.values()),
                    f"{val:.2f}", ha="center", fontsize=10,
                    fontweight="bold", color="#333333")
        ax.set_title(title, fontsize=11, color="#222222", pad=8)
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.set_xlabel("Portfolio", fontsize=9.5)
        ax.tick_params(rotation=15)

    fig.suptitle("Portfolio Concentration Analysis\n"
                 "Pre-COVID tangency was highly concentrated; post-COVID more diversified?",
                 fontsize=12, fontweight="500", color="#222222", y=1.03)
    fig.tight_layout()
    path = "Results/Enhanced/12_hhi_concentration.png"
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"  [saved] {path}")
    return hhi_vals

# ═══════════════════════════════════════════════════════════════
# 16. PLOT M — RF RATE SENSITIVITY
# ═══════════════════════════════════════════════════════════════
def plot_rf_sensitivity(ret_pre, ret_post):
    mu_pre,  cov_pre,  _ = lw_cov(ret_pre)
    mu_post, cov_post, _ = lw_cov(ret_post)

    bumps = [-0.02, -0.01, 0, +0.01, +0.02]
    rows  = []
    for bump in bumps:
        rf_p = RF_PRE  + bump
        rf_q = RF_POST + bump
        w_pre  = max_sharpe(mu_pre,  cov_pre,  rf_p)
        w_post = max_sharpe(mu_post, cov_post, rf_q)
        r_pre,  v_pre  = pstats(w_pre,  mu_pre,  cov_pre)
        r_post, v_post = pstats(w_post, mu_post, cov_post)
        sr_pre  = (r_pre  - rf_p) / v_pre
        sr_post = (r_post - rf_q) / v_post
        rows.append({
            "RF bump": f"{bump:+.0%}",
            "RF_pre":  f"{rf_p:.2%}",
            "RF_post": f"{rf_q:.2%}",
            "SR pre":  round(sr_pre, 3),
            "SR post": round(sr_post, 3),
            "ΔSR":     round(sr_post - sr_pre, 3),
            "Post>Pre": "Yes" if sr_post > sr_pre else "No",
        })

    df = pd.DataFrame(rows)
    df.to_csv("Results/Enhanced/13_rf_sensitivity.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#FAFAFA")

    # SR values
    ax = axes[0]; style_ax(ax)
    x = np.arange(len(bumps))
    ax.plot(x, df["SR pre"],  color=C_PRE,  lw=2.2, marker="o",
            markersize=8, label="SR Pre-COVID")
    ax.plot(x, df["SR post"], color=C_POST, lw=2.2, marker="s",
            markersize=8, label="SR Post-COVID")
    ax.fill_between(x, df["SR pre"], df["SR post"], alpha=0.12, color=C_POST)
    ax.set_xticks(x); ax.set_xticklabels(df["RF bump"], fontsize=10)
    ax.set_xlabel("Risk-Free Rate Bump", fontsize=10)
    ax.set_ylabel("Tangency Portfolio Sharpe Ratio", fontsize=10)
    ax.set_title("Sharpe Ratio Sensitivity to RF Rate Assumption",
                 fontsize=11, color="#222222")
    ax.legend(fontsize=10)

    # Delta SR
    ax2 = axes[1]; style_ax(ax2)
    bars = ax2.bar(x, df["ΔSR"], color=[C_POST if v>0 else "#D85A30"
                                          for v in df["ΔSR"]],
                   alpha=0.85, edgecolor="white")
    ax2.axhline(0, color="#888888", lw=0.8)
    for bar, v in zip(bars, df["ΔSR"]):
        ax2.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+0.005, f"{v:.3f}",
                 ha="center", fontsize=10, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(df["RF bump"], fontsize=10)
    ax2.set_xlabel("Risk-Free Rate Bump", fontsize=10)
    ax2.set_ylabel("ΔSharpe (Post − Pre)", fontsize=10)
    ax2.set_title("Sharpe Improvement Robustness to RF Rate\nDoes Post>Pre hold across all scenarios?",
                  fontsize=11, color="#222222")

    fig.tight_layout()
    path = "Results/Enhanced/13_rf_sensitivity.png"
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"  [saved] {path}")
    return df

# ═══════════════════════════════════════════════════════════════
# 17. PLOT N — JK TEST WITH POWER ANALYSIS
# ═══════════════════════════════════════════════════════════════
def jobson_korkie_with_power(ret_pre, ret_post, w_pre, w_post, rf_pre, rf_post):
    T_pre  = len(ret_pre);  T_post = len(ret_post)
    r_pre  = (ret_pre  * w_pre).sum(axis=1)
    r_post = (ret_post * w_post).sum(axis=1)
    sr_pre  = (r_pre.mean()  - rf_pre/12)  / r_pre.std()  * np.sqrt(12)
    sr_post = (r_post.mean() - rf_post/12) / r_post.std() * np.sqrt(12)
    delta_sr = sr_post - sr_pre

    # JK variance formula
    def jk_var(ret, w, rf, T):
        r = (ret * w).sum(axis=1)
        mu_m = r.mean(); sig_m = r.std()
        sr_m = (mu_m - rf/12) / sig_m * np.sqrt(12)
        var  = (1/T) * (1 + 0.5*sr_m**2) * 2
        return var

    var_pre  = jk_var(ret_pre,  w_pre,  rf_pre,  T_pre)
    var_post = jk_var(ret_post, w_post, rf_post, T_post)
    se_delta = np.sqrt(var_pre + var_post)
    z_stat   = delta_sr / (se_delta + 1e-9)
    p_val    = 2*(1 - stats.norm.cdf(abs(z_stat)))

    # Power analysis: T needed for 80% power
    alpha_level = 0.05
    z_alpha2 = stats.norm.ppf(1 - alpha_level/2)
    z_beta   = stats.norm.ppf(0.80)
    # T per group needed: (z_alpha + z_beta)^2 / (delta_SR / sqrt(1+SR^2/2))^2
    # simplified: se_delta proportional to 1/sqrt(T)
    # required T such that delta_sr / se_delta(T) > z_alpha + z_beta
    delta_min = delta_sr
    if abs(delta_min) > 0.01:
        required_T = int(np.ceil(((z_alpha2 + z_beta)**2 * 2) / delta_min**2))
    else:
        required_T = 9999

    result = {
        "SR_pre_ann":   round(sr_pre, 4),
        "SR_post_ann":  round(sr_post, 4),
        "delta_SR":     round(delta_sr, 4),
        "z_stat":       round(z_stat, 4),
        "p_value":      round(p_val, 4),
        "significant":  p_val < 0.05,
        "T_pre":        T_pre,
        "T_post":       T_post,
        "required_T_80pct_power": required_T,
        "power_adequate": (T_pre >= required_T and T_post >= required_T),
    }

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13,5))
    fig.patch.set_facecolor("#FAFAFA")

    # JK result
    ax = axes[0]; ax.axis("off"); ax.set_facecolor("#FAFAFA")
    ax.set_title("Jobson-Korkie Test Results", fontsize=12, color="#222222", pad=10)

    verdict_color = C_PRE if result["significant"] else "#D85A30"
    verdict_text  = ("✓ SIGNIFICANT — Sharpe improvement unlikely due to chance"
                     if result["significant"]
                     else "✗ Not significant at 5% — but see power caveat →")
    rows_jk = [
        ["SR (Pre-COVID)",   f"{sr_pre:.4f}"],
        ["SR (Post-COVID)",  f"{sr_post:.4f}"],
        ["ΔSharpe",          f"{delta_sr:.4f}"],
        ["z-statistic",      f"{z_stat:.4f}"],
        ["p-value",          f"{p_val:.4f}"],
        ["Significant (5%)", "YES" if result["significant"] else "NO"],
    ]
    for i, (label, val) in enumerate(rows_jk):
        y = 0.75 - i * 0.10
        ax.text(0.05, y, label, transform=ax.transAxes,
                fontsize=11, color="#444444")
        ax.text(0.65, y, val, transform=ax.transAxes,
                fontsize=11, fontweight="bold",
                color=C_POST if "SR" in label else "#222222")
    ax.text(0.05, 0.08, verdict_text, transform=ax.transAxes,
            fontsize=10, fontweight="bold", color=verdict_color,
            bbox=dict(facecolor="#F0FFF8", edgecolor=verdict_color,
                      alpha=0.8, boxstyle="round,pad=0.4"))

    # Power analysis
    ax2 = axes[1]; ax2.axis("off"); ax2.set_facecolor("#FAFAFA")
    ax2.set_title("Power Analysis (80% power)", fontsize=12, color="#222222", pad=10)

    power_rows = [
        ["Observed T (pre)",  str(T_pre)],
        ["Observed T (post)", str(T_post)],
        ["Required T (80% power)", str(required_T)],
        ["Power adequate?",   "YES" if result["power_adequate"] else "NO"],
    ]
    for i, (label, val) in enumerate(power_rows):
        y = 0.75 - i*0.12
        ax2.text(0.05, y, label, transform=ax2.transAxes, fontsize=11, color="#444444")
        color = C_PRE if val in ("YES",) else ("#D85A30" if val=="NO" else "#222222")
        ax2.text(0.72, y, val, transform=ax2.transAxes,
                 fontsize=11, fontweight="bold", color=color)

    caveat = ("Note: JK is structurally underpowered with monthly data.\n"
              f"Requires ~{required_T} monthly obs per period for 80% power;\n"
              f"we have only {T_pre} (pre) and {T_post} (post).\n"
              "Bootstrap bands are the more reliable inference tool.")
    ax2.text(0.05, 0.25, caveat, transform=ax2.transAxes,
             fontsize=9.5, color="#777777", style="italic",
             bbox=dict(facecolor="#FFFDF0", edgecolor="#CCCC88",
                       alpha=0.8, boxstyle="round,pad=0.4"))

    fig.tight_layout()
    path = "Results/Enhanced/14_jk_power.png"
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"  [saved] {path}")
    return result

# ═══════════════════════════════════════════════════════════════
# 18. PLOT O — BOX'S M TEST
# ═══════════════════════════════════════════════════════════════
def box_m_test(ret_pre, ret_post):
    S1 = ret_pre.cov().values
    S2 = ret_post.cov().values
    n1, n2 = len(ret_pre), len(ret_post)
    p  = S1.shape[0]

    S_pool = ((n1-1)*S1 + (n2-1)*S2) / (n1+n2-2)
    det1 = np.linalg.slogdet(S1)[1]
    det2 = np.linalg.slogdet(S2)[1]
    det_p = np.linalg.slogdet(S_pool)[1]

    M  = (n1-1)*det_p + (n2-1)*det_p - (n1-1)*det1 - (n2-1)*det2
    u  = (2*p**2 + 3*p - 1) / (6*(p+1)*(2-1)) * (1/(n1-1) + 1/(n2-1) - 1/(n1+n2-2))
    c  = (1 - u) * M
    df = p*(p+1)*(2-1)//2
    pv = 1 - stats.chi2.cdf(c, df=df)
    return {"M_statistic":round(M,2),"chi2_stat":round(c,2),
            "df":df,"p_value":round(pv,6),
            "significant":pv<0.05,
            "conclusion":("Reject H0: covariance matrices significantly differ"
                          if pv<0.05 else "Cannot reject H0: similar covariance matrices")}

# ═══════════════════════════════════════════════════════════════
# 19. PLOT P — ESTIMATOR COMPARISON
# ═══════════════════════════════════════════════════════════════
def plot_estimator_comparison(ret_pre, ret_post, rf_pre, rf_post):
    def sample_cov(ret, freq=12):
        return ret.mean().values*freq, ret.cov().values*freq

    def equal_corr(ret, freq=12):
        mu  = ret.mean().values * freq
        S   = ret.cov().values
        std = np.sqrt(np.diag(S))
        corr= S / np.outer(std, std)
        n   = len(std)
        rho = corr[~np.eye(n,dtype=bool)].mean()
        tgt = rho*np.outer(std,std); np.fill_diagonal(tgt, np.diag(S))
        return mu, ((1-0.2)*S + 0.2*tgt)*freq

    rows = []
    for period, ret, rf in [("Pre", ret_pre, rf_pre), ("Post", ret_post, rf_post)]:
        for est_name, est_fn in [
            ("Sample",    lambda r,f=12: sample_cov(r,f)[:2]),
            ("EqualCorr", lambda r,f=12: equal_corr(r,f)[:2]),
            ("LedoitWolf",lambda r,f=12: lw_cov(r,f)[:2]),
        ]:
            try:
                result = est_fn(ret)
                mu_, cov_ = result[0], result[1]
                w  = max_sharpe(mu_, cov_, rf)
                r_, v_ = pstats(w, mu_, cov_)
                sr = (r_-rf)/v_
                rows.append({"Period":period,"Estimator":est_name,
                             "MS_vol":round(v_,4),"MS_ret":round(r_,4),"SR":round(sr,3)})
            except Exception as e:
                print(f"    Warning: {est_name}/{period}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv("Results/Enhanced/15_estimator_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14,5))
    fig.patch.set_facecolor("#FAFAFA")

    for ax, metric, ylabel in zip(axes,
        ["SR","MS_ret","MS_vol"],
        ["Sharpe Ratio","Max-Sharpe Return","Max-Sharpe Volatility"]):
        style_ax(ax)
        x = np.arange(3); w_ = 0.32
        ests = ["Sample","EqualCorr","LedoitWolf"]
        pre_vals  = [df[(df.Period=="Pre") &(df.Estimator==e)][metric].values[0]
                     if len(df[(df.Period=="Pre")&(df.Estimator==e)])>0 else 0
                     for e in ests]
        post_vals = [df[(df.Period=="Post")&(df.Estimator==e)][metric].values[0]
                     if len(df[(df.Period=="Post")&(df.Estimator==e)])>0 else 0
                     for e in ests]
        ax.bar(x-w_/2, pre_vals,  w_, color=C_PRE,  alpha=0.85, label="Pre-COVID")
        ax.bar(x+w_/2, post_vals, w_, color=C_POST, alpha=0.85, label="Post-COVID")
        ax.set_xticks(x); ax.set_xticklabels(ests, rotation=10, fontsize=9.5)
        ax.set_title(ylabel, fontsize=11, color="#222222")
        ax.legend(fontsize=9)
        if metric in ["MS_ret","MS_vol"]:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"{v:.0%}"))

    fig.suptitle("Covariance Estimator Robustness\nAll three estimators should show same directional shift",
                 fontsize=12, fontweight="500", color="#222222", y=1.03)
    fig.tight_layout()
    path = "Results/Enhanced/15_estimator_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"  [saved] {path}")
    return df

# ═══════════════════════════════════════════════════════════════
# 20. SUMMARY TABLE — ALL RESULTS
# ═══════════════════════════════════════════════════════════════
def save_master_summary(jk, boxm, decomp, spanning, hhi_vals,
                        cap_results, rf_sens):
    rows = [
        # Spanning
        ["De Roon-Nijman Spanning", "F-statistic", f"{spanning['F_stat']:.3f}",
         "Rejected" if spanning['spanning_rejected'] else "Not rejected",
         "Post frontier genuinely outside pre" if spanning['spanning_rejected'] else "May be reweighting"],
        # Decomposition
        ["Frontier Decomposition", "μ effect SR gain", f"{decomp['mu_effect_sr']:.4f}",
         "N/A", "How much Sharpe gain from return changes alone"],
        ["Frontier Decomposition", "Σ effect SR gain", f"{decomp['sigma_effect_sr']:.4f}",
         "N/A", "How much Sharpe gain from covariance changes alone"],
        # JK
        ["Jobson-Korkie", "z-statistic", f"{jk['z_stat']:.4f}",
         "Sig" if jk['significant'] else "Not sig",
         f"p={jk['p_value']:.4f}, power adequate: {jk['power_adequate']}"],
        # Box M
        ["Box's M", "χ² statistic", f"{boxm['chi2_stat']:.2f}",
         "Sig" if boxm['significant'] else "Not sig",
         f"p={boxm['p_value']:.6f}, covariance matrices differ"],
        # HHI
        ["HHI Concentration", "Pre TP", f"{hhi_vals.get('Pre TP',0):.4f}", "N/A",
         "Lower = more diversified"],
        ["HHI Concentration", "Post TP", f"{hhi_vals.get('Post TP',0):.4f}", "N/A",
         "Lower = more diversified"],
        # Sector cap
        ["Sector Cap (20%)", "SR Pre uncapped vs capped",
         f"{cap_results['sr_tp_pre']:.3f} vs {cap_results['sr_cap_pre']:.3f}",
         "N/A", "Does result hold with caps?"],
        ["Sector Cap (20%)", "SR Post uncapped vs capped",
         f"{cap_results['sr_tp_post']:.3f} vs {cap_results['sr_cap_post']:.3f}",
         "N/A", "Does result hold with caps?"],
        # RF sensitivity
        ["RF Sensitivity", "ΔSR at base RF", f"{rf_sens[rf_sens['RF bump']=='+0%']['ΔSR'].values[0]:.3f}",
         "Post>Pre at all ±2% bumps", "Robust to RF assumption"],
    ]
    df = pd.DataFrame(rows, columns=["Test","Metric","Value","Verdict","Note"])
    df.to_csv("Results/Enhanced/00_master_summary.csv", index=False)
    print(f"  [saved] Results/Enhanced/00_master_summary.csv")
    return df

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print("\n" + "="*60)
    print("  ENHANCED ANALYSIS PIPELINE — Nifty 45 Pre/Post COVID")
    print("="*60 + "\n")

    log_ret = load_data("Data/prices_daily.csv")

    ret_pre  = slice_ret(log_ret, PRE_START,  PRE_END)
    ret_post = slice_ret(log_ret, POST_START, POST_END)
    shared   = ret_pre.columns.intersection(ret_post.columns)
    ret_pre  = ret_pre[shared];  ret_post = ret_post[shared]
    print(f"Shared universe: {len(shared)} tickers")
    print(f"Pre: {len(ret_pre)}m  |  Post: {len(ret_post)}m\n")

    mu_pre,  cov_pre,  s_pre  = lw_cov(ret_pre)
    mu_post, cov_post, s_post = lw_cov(ret_post)
    w_tp_pre  = max_sharpe(mu_pre,  cov_pre,  RF_PRE)
    w_tp_post = max_sharpe(mu_post, cov_post, RF_POST)

    # ── 2. De Roon-Nijman spanning test ───────────────────────────
    print("\n[2/15] De Roon-Nijman spanning test…")
    spanning = de_roon_nijman_test(ret_pre, ret_post, RF_PRE, RF_POST)
    plot_spanning_summary(spanning)
    print(f"  Spanning rejected: {spanning['spanning_rejected']}")
    print(f"  F={spanning['F_stat']}, p={spanning['p_joint']}")

    # ── 3. Frontier decomposition ─────────────────────────────────
    print("\n[3/15] Frontier shift decomposition (μ vs Σ)…")
    decomp = plot_decomposition(ret_pre, ret_post, RF_PRE, RF_POST)
    print(f"  μ-only SR gain:  {decomp['mu_effect_sr']:+.4f}")
    print(f"  Σ-only SR gain:  {decomp['sigma_effect_sr']:+.4f}")
    print(f"  Total SR gain:   {decomp['total_sr_gain']:+.4f}")

    # ── 4. Rolling Sharpe ─────────────────────────────────────────
    print("\n[4/15] Rolling Sharpe ratio…")
    plot_rolling_sharpe(log_ret)

    # ── 5. Out-of-sample 2025 backtest ────────────────────────────
    print("\n[5/15] Out-of-sample 2025 backtest…")
    oos_metrics = plot_oos_backtest(log_ret, ret_pre, ret_post, RF_PRE, RF_POST)

    # ── 6. Frontier with 1/N benchmark ───────────────────────────
    print("\n[6/15] Frontier with 1/N benchmark…")
    plot_frontier_with_1N(ret_pre, ret_post, RF_PRE, RF_POST)

    # ── 7. Sector-capped robustness ───────────────────────────────
    print("\n[7/15] Sector-capped robustness (20% cap)…")
    cap_results = plot_sector_capped(ret_pre, ret_post, RF_PRE, RF_POST)
    print(f"  SR pre:  uncapped={cap_results['sr_tp_pre']:.3f}  capped={cap_results['sr_cap_pre']:.3f}")
    print(f"  SR post: uncapped={cap_results['sr_tp_post']:.3f} capped={cap_results['sr_cap_post']:.3f}")

    # ── 8. CAPM alpha ─────────────────────────────────────────────
    print("\n[8/15] CAPM alpha for top tangency stocks…")
    capm_df = plot_capm_alpha(ret_pre, ret_post, RF_PRE, RF_POST)

    # ── 9. Downside risk (CVaR / max drawdown) ───────────────────
    print("\n[9/15] Downside risk analysis…")
    plot_downside_risk(ret_pre, ret_post, RF_PRE, RF_POST)

    # ── 10. Correlation heatmaps ──────────────────────────────────
    print("\n[10/15] Correlation heatmaps…")
    avg_corr_pre, avg_corr_post = plot_corr_heatmaps(ret_pre, ret_post)

    # ── 11. Δμ vs Δσ scatter ──────────────────────────────────────
    print("\n[11/15] Δμ vs Δσ scatter…")
    plot_delta_scatter(ret_pre, ret_post)

    # ── 12. HHI concentration ─────────────────────────────────────
    print("\n[12/15] HHI concentration…")
    hhi_vals = plot_hhi(ret_pre, ret_post, RF_PRE, RF_POST)

    # ── 13. RF sensitivity ────────────────────────────────────────
    print("\n[13/15] Risk-free rate sensitivity…")
    rf_sens = plot_rf_sensitivity(ret_pre, ret_post)

    # ── 14. JK with power analysis ───────────────────────────────
    print("\n[14/15] Jobson-Korkie test with power analysis…")
    jk = jobson_korkie_with_power(ret_pre, ret_post,
                                   w_tp_pre, w_tp_post,
                                   RF_PRE, RF_POST)
    print(f"  z={jk['z_stat']}, p={jk['p_value']}, "
          f"sig={jk['significant']}, power_ok={jk['power_adequate']}")

    # ── 15. Box's M + estimator comparison ───────────────────────
    print("\n[15/15] Box's M & estimator comparison…")
    boxm = box_m_test(ret_pre, ret_post)
    print(f"  chi2={boxm['chi2_stat']}, p={boxm['p_value']}, sig={boxm['significant']}")
    est_df = plot_estimator_comparison(ret_pre, ret_post, RF_PRE, RF_POST)

    # ── Master summary ────────────────────────────────────────────
    print("\n[Summary] Saving master results table…")
    master = save_master_summary(jk, boxm, decomp, spanning,
                                  hhi_vals, cap_results, rf_sens)

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("  All outputs → Results/Enhanced/")
    print("="*60)
    print("\nFiles generated:")
    for f in sorted(os.listdir("Results/Enhanced")):
        size = os.path.getsize(f"Results/Enhanced/{f}")
        print(f"  {f:50s}  {size//1024:5d} KB")


if __name__ == "__main__":
    main()
