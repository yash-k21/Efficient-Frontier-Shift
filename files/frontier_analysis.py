"""
frontier_analysis.py
====================
Nifty 100 Efficient Frontier — Pre/Post COVID analysis
Builds directly on your existing prices_daily.csv and folder structure.

Outputs (all saved to Results/)
--------------------------------
  Results/frontier_optionA.png       — Primary: 60m pre vs 60m post (full universe)
  Results/frontier_optionC.png       — Sensitivity: 61m pre vs 42m post (reduced universe)
  Results/heatmap_crisis.png         — Crisis window correlation heatmap (Feb 2020–Jun 2021)
  Results/rolling_volatility.png     — Rolling 12m annualised vol through all three periods
  Results/summary_stats.csv          — GMV + Max-Sharpe stats for all four period-universe combos

Run
---
  python frontier_analysis.py

Assumes Data/prices_daily.csv exists (your existing download).
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")
os.makedirs("Results", exist_ok=True)

# ── Risk-free rate (India 10Y G-sec approx average 2015-2024) ──
RF_ANNUAL = 0.065

# ─────────────────────────────────────────────────────────────────
# Option C reduced universe — your 45 tickers mapped to ~40 usable
# This is your existing ticker list; the pipeline will auto-drop
# any with insufficient data in the post window.
# ─────────────────────────────────────────────────────────────────
OPTION_C_TICKERS = [
    # Financials
    'HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS',
    'SBIN.NS', 'INDUSINDBK.NS', 'BANKBARODA.NS', 'PNB.NS',
    # IT
    'TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'TECHM.NS',
    # Energy & Oil
    'RELIANCE.NS', 'ONGC.NS', 'BPCL.NS', 'GAIL.NS', 'TATAPOWER.NS', 'NTPC.NS', 'POWERGRID.NS',
    # Consumer
    'HINDUNILVR.NS', 'ITC.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
    'HEROMOTOCO.NS', 'BAJAJ-AUTO.NS', 'M&M.NS',
    # Pharma
    'SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'LUPIN.NS',
    # Metals & Mining
    'TATASTEEL.NS', 'HINDALCO.NS', 'VEDL.NS', 'COALINDIA.NS',
    # Cement & Industrials
    'ULTRACEMCO.NS', 'GRASIM.NS', 'ACC.NS', 'AMBUJACEM.NS',
    'LT.NS', 'BHEL.NS', 'BOSCHLTD.NS',
    # Telecom & Media
    'BHARTIARTL.NS', 'ZEEL.NS',
    # Ports
    'ADANIPORTS.NS',
]

# ─────────────────────────────────────────────────────────────────
# PERIOD DEFINITIONS
# ─────────────────────────────────────────────────────────────────
#   Option A  Pre : Jan 2015 – Dec 2019  (60m, your existing 'pre')
#   Option A  Post: Jan 2020 – Dec 2024  (60m, your existing 'post')
#   Option C  Pre : Jan 2015 – Jan 2020  (61m)
#   Option C  Post: Jul 2021 – Dec 2024  (42m)
#   Crisis        : Feb 2020 – Jun 2021  (17m, descriptive only)

P = {
    "a_pre":   ("2015-01-01", "2019-12-31"),
    "a_post":  ("2020-01-01", "2024-12-31"),
    "c_pre":   ("2015-01-01", "2020-01-31"),
    "c_post":  ("2021-07-01", "2024-12-31"),
    "crisis":  ("2020-02-01", "2021-06-30"),
    "full":    ("2015-01-01", "2024-12-31"),
}

# ═══════════════════════════════════════════════════════════════════
# 1. LOAD & PREPARE DATA
# ═══════════════════════════════════════════════════════════════════

def load_and_prepare(path="Data/prices_daily.csv"):
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices = prices.sort_index()
    # yfinance sometimes puts 'Price' or 'Ticker' as a header row — drop if so
    prices = prices[~prices.index.astype(str).str.contains("Price|Ticker", na=False)]
    prices = prices.astype(float)
    prices = prices.dropna(axis=1, how="all")

    # Month-end resample → log returns
    monthly = prices.resample("ME").last()
    log_ret  = np.log(monthly / monthly.shift(1)).iloc[1:]

    print(f"[load] {prices.shape[1]} tickers | "
          f"{prices.shape[0]} daily obs | "
          f"{log_ret.shape[0]} monthly obs | "
          f"{log_ret.index[0].date()} → {log_ret.index[-1].date()}")
    return log_ret


def slice_and_clean(log_ret, start, end, tickers=None, max_nan=0.10):
    """Slice period, optionally filter tickers, drop sparse columns, forward-fill gaps."""
    ret = log_ret.loc[start:end]
    if tickers:
        available = [t for t in tickers if t in ret.columns]
        ret = ret[available]
    # Drop tickers missing more than max_nan fraction of obs
    keep = ret.columns[ret.isna().mean() <= max_nan]
    dropped = set(ret.columns) - set(keep)
    if dropped:
        print(f"  [clean] Dropped {len(dropped)} sparse tickers: {sorted(dropped)}")
    ret = ret[keep].ffill(limit=1).dropna(axis=1)
    return ret

# ═══════════════════════════════════════════════════════════════════
# 2. COVARIANCE & OPTIMISATION
# ═══════════════════════════════════════════════════════════════════

def ledoit_wolf_cov(ret):
    """Annualised mean and covariance via Ledoit-Wolf shrinkage."""
    lw = LedoitWolf().fit(ret.values)
    mu  = ret.mean().values * 12
    cov = lw.covariance_  * 12
    n, t = ret.shape[1], ret.shape[0]
    print(f"  [cov] N={n}, T={t}, T/N={t/n:.2f}, shrinkage={lw.shrinkage_:.3f}")
    return mu, cov


def min_var(mu, cov):
    n  = len(mu)
    res = minimize(
        lambda w: w @ cov @ w,
        np.ones(n)/n,
        method="SLSQP",
        bounds=[(0,1)]*n,
        constraints=[{"type":"eq","fun":lambda w: w.sum()-1}],
        options={"ftol":1e-12,"maxiter":2000}
    )
    return res.x if res.success else np.ones(n)/n


def max_sharpe(mu, cov, rf=RF_ANNUAL):
    n = len(mu)
    def neg_sr(w):
        r = w @ mu;  v = np.sqrt(w @ cov @ w)
        return -(r-rf)/(v+1e-9)
    res = minimize(neg_sr, np.ones(n)/n, method="SLSQP",
                   bounds=[(0,1)]*n,
                   constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],
                   options={"ftol":1e-12,"maxiter":2000})
    return res.x if res.success else np.ones(n)/n


def frontier_points(mu, cov, n_pts=250):
    """Trace efficient frontier; returns (vols, rets) arrays."""
    n    = len(mu)
    w_mv = min_var(mu, cov)
    r_lo = w_mv @ mu
    r_hi = mu.max()
    vols, rets = [], []
    for target in np.linspace(r_lo, r_hi, n_pts):
        res = minimize(
            lambda w: w @ cov @ w,
            np.ones(n)/n,
            method="SLSQP",
            bounds=[(0,1)]*n,
            constraints=[
                {"type":"eq","fun":lambda w: w.sum()-1},
                {"type":"eq","fun":lambda w, t=target: w@mu - t}
            ],
            options={"ftol":1e-12,"maxiter":2000}
        )
        if res.success:
            rets.append(res.x @ mu)
            vols.append(np.sqrt(res.x @ cov @ res.x))
    return np.array(vols), np.array(rets)


def pstats(w, mu, cov):
    return w@mu, np.sqrt(w@cov@w)

# ═══════════════════════════════════════════════════════════════════
# 3. PLOT: EFFICIENT FRONTIER (two periods overlaid)
# ═══════════════════════════════════════════════════════════════════

def plot_frontier(mu_pre, cov_pre, tickers_pre,
                  mu_post, cov_post, tickers_post,
                  title, subtitle, out_path,
                  pre_label="Pre-COVID (Jan 2015–Dec 2019)",
                  post_label="Post-COVID (Jan 2020–Dec 2024)"):

    PRE_C  = "#1D9E75"   # teal
    POST_C = "#534AB7"   # purple
    GMV_C  = "#D85A30"   # coral
    MS_C   = "#BA7517"   # amber

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["left","bottom"]: ax.spines[sp].set_color("#CCCCCC")
    ax.grid(True, linestyle="--", linewidth=0.4, color="#DDDDDD", alpha=0.9)

    # ── Pre frontier ──
    v_pre, r_pre = frontier_points(mu_pre, cov_pre)
    ax.plot(v_pre, r_pre, color=PRE_C, lw=2.2, label=pre_label, zorder=4)

    w_mv_pre = min_var(mu_pre, cov_pre)
    r_mv_pre, v_mv_pre = pstats(w_mv_pre, mu_pre, cov_pre)
    ax.scatter(v_mv_pre, r_mv_pre, color=GMV_C, s=90, zorder=6, marker="o")

    w_ms_pre = max_sharpe(mu_pre, cov_pre)
    r_ms_pre, v_ms_pre = pstats(w_ms_pre, mu_pre, cov_pre)
    ax.scatter(v_ms_pre, r_ms_pre, color=MS_C, s=90, zorder=6, marker="D")

    # ── Post frontier ──
    v_post, r_post = frontier_points(mu_post, cov_post)
    ax.plot(v_post, r_post, color=POST_C, lw=2.2, label=post_label, zorder=4)

    w_mv_post = min_var(mu_post, cov_post)
    r_mv_post, v_mv_post = pstats(w_mv_post, mu_post, cov_post)
    ax.scatter(v_mv_post, r_mv_post, color=GMV_C, s=90, zorder=6, marker="o")

    w_ms_post = max_sharpe(mu_post, cov_post)
    r_ms_post, v_ms_post = pstats(w_ms_post, mu_post, cov_post)
    ax.scatter(v_ms_post, r_ms_post, color=MS_C, s=90, zorder=6, marker="D")

    # ── Individual assets ──
    ind_v_pre  = np.sqrt(np.diag(cov_pre))
    ind_v_post = np.sqrt(np.diag(cov_post))
    ax.scatter(ind_v_pre,  mu_pre,  color=PRE_C,  alpha=0.25, s=20, zorder=3)
    ax.scatter(ind_v_post, mu_post, color=POST_C, alpha=0.25, s=20, zorder=3)

    # ── Capital market line (pre) ──
    v_range = np.linspace(0, max(v_pre.max(), v_post.max()) * 1.05, 200)
    sr_pre  = (r_ms_pre  - RF_ANNUAL) / v_ms_pre
    sr_post = (r_ms_post - RF_ANNUAL) / v_ms_post
    ax.plot(v_range, RF_ANNUAL + sr_pre  * v_range, color=PRE_C,
            lw=1.0, linestyle="--", alpha=0.5)
    ax.plot(v_range, RF_ANNUAL + sr_post * v_range, color=POST_C,
            lw=1.0, linestyle="--", alpha=0.5)

    # ── Annotations on key points ──
    offset = 0.003
    ax.annotate(f"GMV pre\n({v_mv_pre:.1%}, {r_mv_pre:.1%})",
                xy=(v_mv_pre, r_mv_pre), xytext=(v_mv_pre+offset, r_mv_pre+offset),
                fontsize=8, color=GMV_C, arrowprops=dict(arrowstyle="-", color=GMV_C, lw=0.7))
    ax.annotate(f"GMV post\n({v_mv_post:.1%}, {r_mv_post:.1%})",
                xy=(v_mv_post, r_mv_post), xytext=(v_mv_post+offset, r_mv_post-offset*3),
                fontsize=8, color=GMV_C, arrowprops=dict(arrowstyle="-", color=GMV_C, lw=0.7))

    # ── Legend ──
    legend_elements = [
        Line2D([0],[0], color=PRE_C,  lw=2.2, label=pre_label),
        Line2D([0],[0], color=POST_C, lw=2.2, label=post_label),
        Line2D([0],[0], linestyle="--", color="#AAAAAA", lw=1.0, label="Capital market line"),
        plt.scatter([],[],  color=GMV_C, s=70, marker="o", label="Global min-variance (GMV)"),
        plt.scatter([],[],  color=MS_C,  s=70, marker="D", label="Max-Sharpe portfolio"),
        plt.scatter([],[],  color="#888888", s=15, alpha=0.4, label="Individual assets"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, framealpha=0.7,
              loc="upper left", frameon=True)

    # ── Axes ──
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0%}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0%}"))
    ax.set_xlabel("Annualised Volatility  (σ)", fontsize=11, color="#555555")
    ax.set_ylabel("Annualised Return  (μ)", fontsize=11, color="#555555")
    ax.tick_params(colors="#666666", labelsize=10)

    # ── Titles ──
    fig.suptitle(title, fontsize=13, fontweight="500", color="#222222", y=0.98)
    ax.set_title(subtitle, fontsize=9.5, color="#777777", pad=6)

    # ── Stats box ──
    stats_text = (
        f"Pre   GMV: σ={v_mv_pre:.1%}, μ={r_mv_pre:.1%}  |  "
        f"Max-SR: σ={v_ms_pre:.1%}, μ={r_ms_pre:.1%}, SR={sr_pre:.2f}\n"
        f"Post  GMV: σ={v_mv_post:.1%}, μ={r_mv_post:.1%}  |  "
        f"Max-SR: σ={v_ms_post:.1%}, μ={r_ms_post:.1%}, SR={sr_post:.2f}"
    )
    fig.text(0.12, 0.01, stats_text, fontsize=8.5, color="#666666",
             ha="left", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F0EE",
                       edgecolor="#CCCCCC", alpha=0.8))

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")
    return {
        "pre_gmv_vol": v_mv_pre, "pre_gmv_ret": r_mv_pre,
        "pre_ms_vol":  v_ms_pre, "pre_ms_ret":  r_ms_pre, "pre_sr": sr_pre,
        "post_gmv_vol":v_mv_post,"post_gmv_ret": r_mv_post,
        "post_ms_vol": v_ms_post,"post_ms_ret":  r_ms_post,"post_sr":sr_post,
    }

# ═══════════════════════════════════════════════════════════════════
# 4. PLOT: CORRELATION HEATMAP (crisis window)
# ═══════════════════════════════════════════════════════════════════

def plot_heatmap(ret_crisis, out_path):
    corr = ret_crisis.corr()
    # Sort by average correlation — most correlated cluster first
    order = corr.mean().sort_values(ascending=False).index
    corr  = corr.loc[order, order]

    n = corr.shape[0]
    fig_w = max(13, n * 0.20)
    fig_h = max(11, n * 0.17)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#FAFAFA")

    # Clean ticker labels (strip .NS)
    labels = [t.replace(".NS","") for t in corr.columns]
    corr_display = corr.copy()
    corr_display.columns = labels
    corr_display.index   = labels

    sns.heatmap(
        corr_display, ax=ax,
        cmap="RdYlGn", center=0, vmin=-0.3, vmax=1.0,
        linewidths=0.2, linecolor="#EEEEEE",
        annot=True, fmt=".2f", annot_kws={"size": 6},
        cbar_kws={"shrink": 0.5, "pad": 0.02, "label": "Pearson correlation"}
    )
    ax.set_title(
        "Pairwise return correlation — crisis window (Feb 2020 – Jun 2021)\n"
        "Tickers sorted by average pairwise correlation (most correlated first)",
        fontsize=11, fontweight="500", color="#222222", pad=12
    )
    ax.tick_params(labelsize=7.5, colors="#444444")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Average pairwise correlation
    mask = np.ones(corr.shape, dtype=bool)
    np.fill_diagonal(mask, False)
    avg_corr = corr.values[mask].mean()
    fig.text(
        0.01, 0.005,
        f"Average pairwise correlation: {avg_corr:.3f}  |  "
        f"N = {n} assets  |  T = {len(ret_crisis)} monthly obs  |  "
        f"Period: Feb 2020 – Jun 2021",
        fontsize=8, color="#888888"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ═══════════════════════════════════════════════════════════════════
# 5. PLOT: ROLLING VOLATILITY (across all three regimes)
# ═══════════════════════════════════════════════════════════════════

def plot_rolling_vol(log_ret, out_path, window=12):
    """
    Rolling 12-month annualised equal-weight portfolio volatility.
    Shaded bands mark the three regimes.
    """
    # Equal-weight portfolio monthly return
    ew = log_ret.mean(axis=1)

    # Rolling std → annualise
    roll_vol = ew.rolling(window).std() * np.sqrt(12)
    roll_vol = roll_vol.dropna()

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["left","bottom"]: ax.spines[sp].set_color("#CCCCCC")
    ax.grid(True, linestyle="--", linewidth=0.35, color="#DDDDDD", alpha=0.9)

    # ── Regime shading ──
    ax.axvspan(pd.Timestamp("2015-01-01"), pd.Timestamp("2019-12-31"),
               alpha=0.07, color="#1D9E75", label="Pre-COVID regime")
    ax.axvspan(pd.Timestamp("2020-02-01"), pd.Timestamp("2021-06-30"),
               alpha=0.12, color="#E24B4A", label="Crisis window")
    ax.axvspan(pd.Timestamp("2021-07-01"), pd.Timestamp("2024-12-31"),
               alpha=0.07, color="#534AB7", label="Post-COVID regime")

    # ── Period boundary lines ──
    for date, lbl, side in [
        ("2020-01-31", "Pre → Crisis",  "right"),
        ("2021-07-01", "Crisis → Post", "left"),
    ]:
        ax.axvline(pd.Timestamp(date), color="#888888", lw=0.9,
                   linestyle="--", alpha=0.7)
        ax.text(pd.Timestamp(date), roll_vol.max() * 1.02, lbl,
                fontsize=7.5, color="#777777",
                ha=side, va="bottom", rotation=0)

    # ── Rolling vol line ──
    ax.plot(roll_vol.index, roll_vol.values,
            color="#2C2C2A", lw=1.8, zorder=5, label=f"{window}-month rolling vol (EW)")

    # ── Fill under line ──
    ax.fill_between(roll_vol.index, roll_vol.values, alpha=0.12, color="#2C2C2A")

    # ── Key event annotations ──
    events = {
        "Mar 2020\nNifty −38%":  "2020-03-31",
        "Nov 2021\nRelCap insolvency": "2021-11-30",
        "Jul 2023\nHDFC merger": "2023-07-31",
    }
    for label, date in events.items():
        ts = pd.Timestamp(date)
        if ts in roll_vol.index:
            yval = roll_vol.loc[ts]
        else:
            # nearest
            idx = roll_vol.index.get_indexer([ts], method="nearest")[0]
            yval = roll_vol.iloc[idx]
        ax.annotate(label, xy=(ts, yval),
                    xytext=(ts, yval + roll_vol.max()*0.12),
                    fontsize=7.5, color="#555555", ha="center",
                    arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.7))

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0%}"))
    ax.set_ylabel(f"Annualised Volatility ({window}-month rolling)", fontsize=10.5, color="#555555")
    ax.set_xlabel("Date", fontsize=10.5, color="#555555")
    ax.tick_params(colors="#666666", labelsize=9.5)
    ax.set_title(
        "Rolling 12-month annualised volatility — equal-weight Nifty portfolio\n"
        "Shaded regions: pre-COVID (green) · crisis (red) · post-COVID (purple)",
        fontsize=11, fontweight="500", color="#222222", pad=10
    )
    ax.legend(fontsize=9, framealpha=0.6, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ═══════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("\n══════════════════════════════════════════")
    print("  Nifty 100 Efficient Frontier Pipeline")
    print("══════════════════════════════════════════\n")

    # ── Load ──────────────────────────────────────
    log_ret = load_and_prepare("Data/prices_daily.csv")

    # ── OPTION A — full universe, 60m + 60m ───────
    print("\n── Option A: full universe ──")
    ret_a_pre  = slice_and_clean(log_ret, *P["a_pre"])
    ret_a_post = slice_and_clean(log_ret, *P["a_post"])
    # Align to shared tickers only
    shared_a   = ret_a_pre.columns.intersection(ret_a_post.columns)
    ret_a_pre, ret_a_post = ret_a_pre[shared_a], ret_a_post[shared_a]
    print(f"  Shared universe: {len(shared_a)} tickers")
    print(f"  Pre obs: {len(ret_a_pre)}   Post obs: {len(ret_a_post)}")

    mu_a_pre,  cov_a_pre  = ledoit_wolf_cov(ret_a_pre)
    mu_a_post, cov_a_post = ledoit_wolf_cov(ret_a_post)

    stats_a = plot_frontier(
        mu_a_pre,  cov_a_pre,  shared_a.tolist(),
        mu_a_post, cov_a_post, shared_a.tolist(),
        title="Efficient Frontier — Option A (Primary Result)",
        subtitle=(f"Full universe ({len(shared_a)} stocks) | "
                  "Pre-COVID: Jan 2015–Dec 2019 (60m) | "
                  "Post-COVID: Jan 2020–Dec 2024 (60m) | "
                  f"Ledoit-Wolf shrinkage | RF = {RF_ANNUAL:.1%}"),
        out_path="Results/frontier_optionA.png",
        pre_label=f"Pre-COVID  (Jan 2015–Dec 2019, {len(ret_a_pre)}m)",
        post_label=f"Post-COVID  (Jan 2020–Dec 2024, {len(ret_a_post)}m)",
    )

    # ── OPTION C — reduced universe, crisis excluded ──
    print("\n── Option C: reduced universe (sensitivity) ──")
    ret_c_pre  = slice_and_clean(log_ret, *P["c_pre"],  tickers=OPTION_C_TICKERS)
    ret_c_post = slice_and_clean(log_ret, *P["c_post"], tickers=OPTION_C_TICKERS)
    shared_c   = ret_c_pre.columns.intersection(ret_c_post.columns)
    ret_c_pre, ret_c_post = ret_c_pre[shared_c], ret_c_post[shared_c]
    print(f"  Shared reduced universe: {len(shared_c)} tickers")
    print(f"  Pre obs: {len(ret_c_pre)}   Post obs: {len(ret_c_post)}")

    mu_c_pre,  cov_c_pre  = ledoit_wolf_cov(ret_c_pre)
    mu_c_post, cov_c_post = ledoit_wolf_cov(ret_c_post)

    stats_c = plot_frontier(
        mu_c_pre,  cov_c_pre,  shared_c.tolist(),
        mu_c_post, cov_c_post, shared_c.tolist(),
        title="Efficient Frontier — Option C (Sensitivity Check)",
        subtitle=(f"Reduced universe ({len(shared_c)} stocks, crisis window excluded) | "
                  "Pre: Jan 2015–Jan 2020 (61m) | "
                  "Post: Jul 2021–Dec 2024 (42m) | "
                  f"Ledoit-Wolf shrinkage | RF = {RF_ANNUAL:.1%}"),
        out_path="Results/frontier_optionC.png",
        pre_label=f"Pre-COVID  (Jan 2015–Jan 2020, {len(ret_c_pre)}m, crisis excl.)",
        post_label=f"Post-COVID  (Jul 2021–Dec 2024, {len(ret_c_post)}m, crisis excl.)",
    )

    # ── CRISIS HEATMAP ─────────────────────────────
    print("\n── Crisis window: correlation heatmap ──")
    ret_crisis = slice_and_clean(log_ret, *P["crisis"])
    print(f"  Crisis obs: {len(ret_crisis)}  Tickers: {ret_crisis.shape[1]}")
    plot_heatmap(ret_crisis, "Results/heatmap_crisis.png")

    # Average pairwise correlation comparison across periods
    def avg_pairwise(ret):
        c = ret.corr().values
        n = c.shape[0]
        mask = ~np.eye(n, dtype=bool)
        return c[mask].mean()

    avg_pre    = avg_pairwise(ret_a_pre)
    avg_post   = avg_pairwise(ret_a_post)
    avg_crisis = avg_pairwise(ret_crisis)
    print(f"\n  Avg pairwise correlation:")
    print(f"    Pre-COVID : {avg_pre:.3f}")
    print(f"    Crisis    : {avg_crisis:.3f}  ← diversification collapse")
    print(f"    Post-COVID: {avg_post:.3f}")

    # ── ROLLING VOLATILITY ──────────────────────────
    print("\n── Rolling volatility chart ──")
    plot_rolling_vol(log_ret, "Results/rolling_volatility.png")

    # ── SUMMARY CSV ────────────────────────────────
    print("\n── Summary statistics ──")
    rows = []
    for label, stats, mu_pre, cov_pre, mu_post, cov_post, n_pre, n_post, n_tkr in [
        ("Option A", stats_a,
         mu_a_pre, cov_a_pre, mu_a_post, cov_a_post,
         len(ret_a_pre), len(ret_a_post), len(shared_a)),
        ("Option C", stats_c,
         mu_c_pre, cov_c_pre, mu_c_post, cov_c_post,
         len(ret_c_pre), len(ret_c_post), len(shared_c)),
    ]:
        for period, mu, cov, n_obs in [
            ("Pre",  mu_pre,  cov_pre,  n_pre),
            ("Post", mu_post, cov_post, n_post),
        ]:
            w_mv = min_var(mu, cov)
            w_ms = max_sharpe(mu, cov)
            r_mv, v_mv = pstats(w_mv, mu, cov)
            r_ms, v_ms = pstats(w_ms, mu, cov)
            rows.append({
                "Option": label, "Period": period,
                "N_tickers": n_tkr, "T_obs": n_obs,
                "T/N": round(n_obs/n_tkr, 2),
                "GMV_vol": round(v_mv, 4), "GMV_ret": round(r_mv, 4),
                "MaxSR_vol": round(v_ms, 4), "MaxSR_ret": round(r_ms, 4),
                "Sharpe": round((r_ms-RF_ANNUAL)/v_ms, 3),
                "Avg_pairwise_corr": round(
                    avg_pairwise(ret_a_pre if period=="Pre" else ret_a_post), 3
                ),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv("Results/summary_stats.csv", index=False)
    print(summary.to_string(index=False))

    print("\n══════════════════════════════════════════")
    print("  All outputs saved to Results/")
    print("══════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
