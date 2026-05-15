"""
frontier_analysis.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")
os.makedirs("Results", exist_ok=True)

# ── Risk-free rates ──────────────────────────────────────────────
RF_PRE  = 0.0654   # RBI 91-day T-bill average Jan 2015–Jan 2020
RF_POST = 0.0578   # RBI 91-day T-bill average Jul 2021–Dec 2024

# ── Period definitions ───────────────────────────────────────────
P = {
    "a_pre":  ("2015-01-01", "2019-12-31"),   # Option A pre  — 60m
    "a_post": ("2020-01-01", "2024-12-31"),   # Option A post — 60m
    "c_pre":  ("2015-01-01", "2020-01-31"),   # Option C pre  — 61m
    "c_post": ("2021-07-01", "2024-12-31"),   # Option C post — 42m
    "crisis": ("2020-02-01", "2021-06-30"),   # Crisis window — descriptive only
}

# ── Option C reduced universe (sector-representative) ────────────
OPTION_C_TICKERS = [
    # Financials
    'HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS',
    'SBIN.NS', 'INDUSINDBK.NS', 'BANKBARODA.NS', 'PNB.NS',
    # IT
    'TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'TECHM.NS',
    # Energy & Oil
    'RELIANCE.NS', 'ONGC.NS', 'BPCL.NS', 'GAIL.NS',
    'TATAPOWER.NS', 'NTPC.NS', 'POWERGRID.NS',
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

# 1. LOAD & PREPARE

def load_and_prepare(path="Data/prices_daily.csv"):
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices = prices.sort_index()
    prices = prices[~prices.index.astype(str).str.contains(
        "Price|Ticker", na=False)]
    prices = prices.astype(float)
    prices = prices.dropna(axis=1, how="all")

    # Month-end resample → log returns
    monthly = prices.resample("ME").last()
    log_ret = np.log(monthly / monthly.shift(1)).iloc[1:]

    print(f"[load] {prices.shape[1]} tickers | "
          f"{prices.shape[0]} daily obs | "
          f"{log_ret.shape[0]} monthly obs | "
          f"{log_ret.index[0].date()} to {log_ret.index[-1].date()}")
    return log_ret


def slice_and_clean(log_ret, start, end, tickers=None, max_nan=0.10):
    ret = log_ret.loc[start:end]
    if tickers:
        available = [t for t in tickers if t in ret.columns]
        ret = ret[available]
    keep = ret.columns[ret.isna().mean() <= max_nan]
    dropped = set(ret.columns) - set(keep)
    if dropped:
        print(f"  [clean] Dropped {len(dropped)} sparse tickers: "
              f"{sorted(dropped)}")
    ret = ret[keep].ffill(limit=1).dropna(axis=1)
    return ret

# 2. COVARIANCE & OPTIMISATION

def ledoit_wolf_cov(ret, freq=12):
    """Annualised mean and covariance via Ledoit-Wolf shrinkage."""
    lw  = LedoitWolf().fit(ret.values)
    mu  = ret.mean().values * freq
    cov = lw.covariance_ * freq
    n, t = ret.shape[1], ret.shape[0]
    print(f"  [cov] N={n}, T={t}, T/N={t/n:.2f}, "
          f"shrinkage={lw.shrinkage_:.3f}")
    return mu, cov


def min_var(mu, cov):
    n = len(mu)
    res = minimize(
        lambda w: w @ cov @ w,
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
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
        neg_sr,
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-12, "maxiter": 2000}
    )
    return res.x if res.success else np.ones(n) / n


def frontier_points(mu, cov, n_pts=300):
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
                {"type": "eq", "fun": lambda w: w.sum() - 1},
                {"type": "eq", "fun": lambda w, t=target: w @ mu - t}
            ],
            options={"ftol": 1e-12, "maxiter": 2000}
        )
        if res.success:
            rets.append(res.x @ mu)
            vols.append(np.sqrt(res.x @ cov @ res.x))
    return np.array(vols), np.array(rets)


def pstats(w, mu, cov):
    return w @ mu, np.sqrt(w @ cov @ w)

# 3. PLOT: EFFICIENT FRONTIER

def plot_frontier(mu_pre, cov_pre,
                  mu_post, cov_post,
                  rf_pre, rf_post,
                  title, subtitle, out_path,
                  pre_label="Pre-COVID",
                  post_label="Post-COVID"):

    PRE_C  = "#1D9E75"
    POST_C = "#534AB7"
    GMV_C  = "#D85A30"
    MS_C   = "#BA7517"

    fig, ax = plt.subplots(figsize=(12, 7.5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#CCCCCC")
    ax.grid(True, linestyle="--", linewidth=0.4,
            color="#DDDDDD", alpha=0.9)

    # ── Pre frontier ──────────────────────────────
    v_pre, r_pre = frontier_points(mu_pre, cov_pre)
    ax.plot(v_pre, r_pre, color=PRE_C, lw=2.2,
            label=pre_label, zorder=4)

    w_mv_pre = min_var(mu_pre, cov_pre)
    r_mv_pre, v_mv_pre = pstats(w_mv_pre, mu_pre, cov_pre)
    ax.scatter(v_mv_pre, r_mv_pre, color=GMV_C,
               s=100, zorder=6, marker="o")

    w_ms_pre = max_sharpe(mu_pre, cov_pre, rf_pre)
    r_ms_pre, v_ms_pre = pstats(w_ms_pre, mu_pre, cov_pre)
    ax.scatter(v_ms_pre, r_ms_pre, color=MS_C,
               s=100, zorder=6, marker="D")

    # ── Post frontier ─────────────────────────────
    v_post, r_post = frontier_points(mu_post, cov_post)
    ax.plot(v_post, r_post, color=POST_C, lw=2.2,
            label=post_label, zorder=4)

    w_mv_post = min_var(mu_post, cov_post)
    r_mv_post, v_mv_post = pstats(w_mv_post, mu_post, cov_post)
    ax.scatter(v_mv_post, r_mv_post, color=GMV_C,
               s=100, zorder=6, marker="o")

    w_ms_post = max_sharpe(mu_post, cov_post, rf_post)
    r_ms_post, v_ms_post = pstats(w_ms_post, mu_post, cov_post)
    ax.scatter(v_ms_post, r_ms_post, color=MS_C,
               s=100, zorder=6, marker="D")

    # ── Individual assets ─────────────────────────
    ax.scatter(np.sqrt(np.diag(cov_pre)),  mu_pre,
               color=PRE_C,  alpha=0.22, s=22, zorder=3)
    ax.scatter(np.sqrt(np.diag(cov_post)), mu_post,
               color=POST_C, alpha=0.22, s=22, zorder=3)

    # ── Capital allocation lines ──────────────────
    v_max   = max(v_pre.max(), v_post.max()) * 1.08
    v_range = np.linspace(0, v_max, 300)
    sr_pre  = (r_ms_pre  - rf_pre)  / v_ms_pre
    sr_post = (r_ms_post - rf_post) / v_ms_post
    ax.plot(v_range, rf_pre  + sr_pre  * v_range,
            color=PRE_C,  lw=1.1, linestyle="--", alpha=0.55)
    ax.plot(v_range, rf_post + sr_post * v_range,
            color=POST_C, lw=1.1, linestyle="--", alpha=0.55)

    # ── GMV annotations ───────────────────────────
    off = 0.004
    ax.annotate(
        f"GMV pre\n({v_mv_pre:.1%}, {r_mv_pre:.1%})",
        xy=(v_mv_pre, r_mv_pre),
        xytext=(v_mv_pre + off, r_mv_pre - off * 4),
        fontsize=8, color=GMV_C,
        arrowprops=dict(arrowstyle="-", color=GMV_C, lw=0.7)
    )
    ax.annotate(
        f"GMV post\n({v_mv_post:.1%}, {r_mv_post:.1%})",
        xy=(v_mv_post, r_mv_post),
        xytext=(v_mv_post + off, r_mv_post + off * 3),
        fontsize=8, color=GMV_C,
        arrowprops=dict(arrowstyle="-", color=GMV_C, lw=0.7)
    )

    # ── Legend ────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], color=PRE_C,  lw=2.2, label=pre_label),
        Line2D([0], [0], color=POST_C, lw=2.2, label=post_label),
        Line2D([0], [0], linestyle="--", color="#AAAAAA",
               lw=1.1, label="Capital allocation line"),
        plt.scatter([], [], color=GMV_C, s=75, marker="o",
                    label="Global min-variance (GMV)"),
        plt.scatter([], [], color=MS_C,  s=75, marker="D",
                    label="Max-Sharpe (tangency) portfolio"),
        plt.scatter([], [], color="#888888", s=18, alpha=0.4,
                    label="Individual assets"),
    ]
    ax.legend(handles=legend_elements, fontsize=9.5,
              framealpha=0.7, loc="upper left", frameon=True)

    # ── Axes formatting ───────────────────────────
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlabel("Annualised Volatility  (\u03c3)",
                  fontsize=11, color="#555555")
    ax.set_ylabel("Annualised Return  (\u03bc)",
                  fontsize=11, color="#555555")
    ax.tick_params(colors="#666666", labelsize=10)

    # ── Titles ────────────────────────────────────
    fig.suptitle(title, fontsize=13, fontweight="500",
                 color="#222222", y=0.98)
    ax.set_title(subtitle, fontsize=9, color="#777777", pad=6)

    # ── Stats box ────────────────────────────────
    stats_text = (
        f"Pre   GMV: \u03c3={v_mv_pre:.1%}, \u03bc={r_mv_pre:.1%}  |  "
        f"Max-SR: \u03c3={v_ms_pre:.1%}, \u03bc={r_ms_pre:.1%}, "
        f"SR={sr_pre:.2f}\n"
        f"Post  GMV: \u03c3={v_mv_post:.1%}, \u03bc={r_mv_post:.1%}  |  "
        f"Max-SR: \u03c3={v_ms_post:.1%}, \u03bc={r_ms_post:.1%}, "
        f"SR={sr_post:.2f}"
    )
    fig.text(
        0.12, 0.01, stats_text,
        fontsize=8.5, color="#555555", ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F0EE",
                  edgecolor="#CCCCCC", alpha=0.85)
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")

    return {
        "pre_gmv_vol":  v_mv_pre,  "pre_gmv_ret":  r_mv_pre,
        "pre_ms_vol":   v_ms_pre,  "pre_ms_ret":   r_ms_pre,
        "pre_sr":       sr_pre,
        "post_gmv_vol": v_mv_post, "post_gmv_ret": r_mv_post,
        "post_ms_vol":  v_ms_post, "post_ms_ret":  r_ms_post,
        "post_sr":      sr_post,
    }

# 4. PLOT: CORRELATION HEATMAP

def plot_heatmap(ret_crisis, out_path):
    corr  = ret_crisis.corr()
    order = corr.mean().sort_values(ascending=False).index
    corr  = corr.loc[order, order]

    n      = corr.shape[0]
    labels = [t.replace(".NS", "") for t in corr.columns]
    corr_display         = corr.copy()
    corr_display.columns = labels
    corr_display.index   = labels

    fig, ax = plt.subplots(figsize=(20, 17))
    fig.patch.set_facecolor("#FAFAFA")

    sns.heatmap(
        corr_display, ax=ax,
        cmap="RdYlGn",
        center=0,
        vmin=-0.3, vmax=1.0,
        linewidths=0.35,
        linecolor="#FFFFFF",
        annot=False,
        cbar_kws={
            "shrink":   0.45,
            "pad":      0.02,
            "label":    "Pearson correlation"
        }
    )

    ax.set_title(
        "Pairwise return correlation \u2014 crisis window "
        "(Feb 2020\u2013Jun 2021)\n"
        "Tickers sorted by average pairwise correlation "
        "(most correlated first)",
        fontsize=13, fontweight="500", color="#222222", pad=14
    )
    ax.tick_params(labelsize=9, colors="#333333")
    ax.set_xticklabels(ax.get_xticklabels(),
                       rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # White divider lines separating correlation clusters
    for boundary in [20, 33]:
        ax.axhline(boundary, color="white", lw=2.5)
        ax.axvline(boundary, color="white", lw=2.5)

    # Footer
    mask     = np.ones(corr.shape, dtype=bool)
    np.fill_diagonal(mask, False)
    avg_corr = corr.values[mask].mean()
    fig.text(
        0.01, 0.002,
        f"Average pairwise correlation: {avg_corr:.3f}  |  "
        f"N = {n} assets  |  T = {len(ret_crisis)} monthly obs  |  "
        f"Period: Feb 2020\u2013Jun 2021",
        fontsize=9, color="#888888"
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")

# 5. PLOT: ROLLING VOLATILITY

def plot_rolling_vol(log_ret, out_path, window=12):
    ew       = log_ret.mean(axis=1)
    roll_vol = (ew.rolling(window).std() * np.sqrt(12)).dropna()

    fig, ax = plt.subplots(figsize=(14, 5.5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#CCCCCC")
    ax.grid(True, linestyle="--", linewidth=0.35,
            color="#DDDDDD", alpha=0.9)

    # Regime shading
    ax.axvspan(pd.Timestamp("2015-01-01"),
               pd.Timestamp("2019-12-31"),
               alpha=0.08, color="#1D9E75",
               label="Pre-COVID regime")
    ax.axvspan(pd.Timestamp("2020-02-01"),
               pd.Timestamp("2021-06-30"),
               alpha=0.13, color="#E24B4A",
               label="Crisis window")
    ax.axvspan(pd.Timestamp("2021-07-01"),
               pd.Timestamp("2024-12-31"),
               alpha=0.08, color="#534AB7",
               label="Post-COVID regime")

    # Boundary lines
    for date, lbl, side in [
        ("2020-01-31", "Pre \u2192 Crisis",  "right"),
        ("2021-07-01", "Crisis \u2192 Post", "left"),
    ]:
        ax.axvline(pd.Timestamp(date), color="#888888",
                   lw=1.0, linestyle="--", alpha=0.7)
        ax.text(pd.Timestamp(date), roll_vol.max() * 1.03,
                lbl, fontsize=8, color="#777777",
                ha=side, va="bottom")

    # Rolling vol line + fill
    ax.plot(roll_vol.index, roll_vol.values,
            color="#2C2C2A", lw=1.9, zorder=5,
            label=f"{window}-month rolling vol (EW)")
    ax.fill_between(roll_vol.index, roll_vol.values,
                    alpha=0.10, color="#2C2C2A")

    # Key event annotations
    events = {
        "Mar 2020\nNifty \u221238%": "2020-03-31",
        "Nov 2021\nRelCap insolvency": "2021-11-30",
        "Jul 2023\nHDFC merger": "2023-07-31",
    }
    for label, date in events.items():
        ts  = pd.Timestamp(date)
        idx = roll_vol.index.get_indexer([ts], method="nearest")[0]
        yv  = roll_vol.iloc[idx]
        ax.annotate(
            label, xy=(ts, yv),
            xytext=(ts, yv + roll_vol.max() * 0.13),
            fontsize=8, color="#444444", ha="center",
            arrowprops=dict(arrowstyle="-",
                            color="#AAAAAA", lw=0.8)
        )

    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_ylabel(
        f"Annualised Volatility ({window}-month rolling)",
        fontsize=11, color="#555555")
    ax.set_xlabel("Date", fontsize=11, color="#555555")
    ax.tick_params(colors="#666666", labelsize=9.5)
    ax.set_title(
        "Rolling 12-month annualised volatility \u2014 "
        "equal-weight Nifty portfolio\n"
        "Shaded: pre-COVID (green) \u00b7 crisis (red) "
        "\u00b7 post-COVID (purple)",
        fontsize=11, fontweight="500", color="#222222", pad=10
    )
    ax.legend(fontsize=9, framealpha=0.6, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")

# 6. HELPER — average pairwise correlation

def avg_pairwise(ret):
    c = ret.corr().values
    n = c.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return c[mask].mean()

# 7. MAIN
def main():
    print("\n" + "=" * 50)
    print("  Nifty 45 Efficient Frontier Pipeline")
    print("=" * 50 + "\n")

    # ── Load ──────────────────────────────────────
    log_ret = load_and_prepare("Data/prices_daily.csv")

    # ── OPTION A — full universe, 60m + 60m ───────
    print("\n── Option A: full universe ──")
    ret_a_pre  = slice_and_clean(log_ret, *P["a_pre"])
    ret_a_post = slice_and_clean(log_ret, *P["a_post"])
    shared_a   = ret_a_pre.columns.intersection(ret_a_post.columns)
    ret_a_pre  = ret_a_pre[shared_a]
    ret_a_post = ret_a_post[shared_a]
    print(f"  Shared universe : {len(shared_a)} tickers")
    print(f"  Pre obs         : {len(ret_a_pre)}")
    print(f"  Post obs        : {len(ret_a_post)}")

    mu_a_pre,  cov_a_pre  = ledoit_wolf_cov(ret_a_pre)
    mu_a_post, cov_a_post = ledoit_wolf_cov(ret_a_post)

    stats_a = plot_frontier(
        mu_a_pre,  cov_a_pre,
        mu_a_post, cov_a_post,
        rf_pre=RF_PRE, rf_post=RF_POST,
        title="Efficient Frontier \u2014 Option A (Primary Result)",
        subtitle=(
            f"Full universe ({len(shared_a)} stocks)  |  "
            "Pre-COVID: Jan 2015\u2013Dec 2019 (60m)  |  "
            "Post-COVID: Jan 2020\u2013Dec 2024 (60m)  |  "
            f"Ledoit-Wolf shrinkage  |  "
            f"RF pre={RF_PRE:.2%}, post={RF_POST:.2%}"
        ),
        out_path="Results/frontier_optionA.png",
        pre_label=(f"Pre-COVID  "
                   f"(Jan 2015\u2013Dec 2019, {len(ret_a_pre)}m)"),
        post_label=(f"Post-COVID  "
                    f"(Jan 2020\u2013Dec 2024, {len(ret_a_post)}m)"),
    )

    # ── OPTION C — reduced universe, crisis excl. ─
    print("\n── Option C: reduced universe (sensitivity) ──")
    ret_c_pre  = slice_and_clean(log_ret, *P["c_pre"],
                                 tickers=OPTION_C_TICKERS)
    ret_c_post = slice_and_clean(log_ret, *P["c_post"],
                                 tickers=OPTION_C_TICKERS)
    shared_c   = ret_c_pre.columns.intersection(ret_c_post.columns)
    ret_c_pre  = ret_c_pre[shared_c]
    ret_c_post = ret_c_post[shared_c]
    print(f"  Shared reduced universe : {len(shared_c)} tickers")
    print(f"  Pre obs                 : {len(ret_c_pre)}")
    print(f"  Post obs                : {len(ret_c_post)}")

    mu_c_pre,  cov_c_pre  = ledoit_wolf_cov(ret_c_pre)
    mu_c_post, cov_c_post = ledoit_wolf_cov(ret_c_post)

    stats_c = plot_frontier(
        mu_c_pre,  cov_c_pre,
        mu_c_post, cov_c_post,
        rf_pre=RF_PRE, rf_post=RF_POST,
        title="Efficient Frontier \u2014 Option C (Sensitivity Check)",
        subtitle=(
            f"Reduced universe ({len(shared_c)} stocks, "
            "crisis window excluded)  |  "
            "Pre: Jan 2015\u2013Jan 2020 (61m)  |  "
            "Post: Jul 2021\u2013Dec 2024 (42m)  |  "
            f"Ledoit-Wolf shrinkage  |  "
            f"RF pre={RF_PRE:.2%}, post={RF_POST:.2%}"
        ),
        out_path="Results/frontier_optionC.png",
        pre_label=(f"Pre-COVID  "
                   f"(Jan 2015\u2013Jan 2020, {len(ret_c_pre)}m, "
                   f"crisis excl.)"),
        post_label=(f"Post-COVID  "
                    f"(Jul 2021\u2013Dec 2024, {len(ret_c_post)}m, "
                    f"crisis excl.)"),
    )

    # ── CRISIS HEATMAP ────────────────────────────
    print("\n── Crisis window: correlation heatmap ──")
    ret_crisis = slice_and_clean(log_ret, *P["crisis"])
    print(f"  Crisis obs : {len(ret_crisis)}")
    print(f"  Tickers    : {ret_crisis.shape[1]}")
    plot_heatmap(ret_crisis, "Results/heatmap_crisis.png")

    # Correlation summary across periods
    avg_pre    = avg_pairwise(ret_a_pre)
    avg_post   = avg_pairwise(ret_a_post)
    avg_crisis = avg_pairwise(ret_crisis)
    print(f"\n  Avg pairwise correlation:")
    print(f"    Pre-COVID  : {avg_pre:.3f}")
    print(f"    Crisis     : {avg_crisis:.3f}  <- diversification collapse")
    print(f"    Post-COVID : {avg_post:.3f}")

    # ── ROLLING VOLATILITY ────────────────────────
    print("\n── Rolling volatility chart ──")
    plot_rolling_vol(log_ret, "Results/rolling_volatility.png")

    # ── SUMMARY CSV ───────────────────────────────
    print("\n── Summary statistics ──")
    rows = []
    for label, mu_pre, cov_pre, mu_post, cov_post, \
            rf_pre, rf_post, n_pre, n_post, n_tkr in [
        ("Option A",
         mu_a_pre, cov_a_pre, mu_a_post, cov_a_post,
         RF_PRE, RF_POST,
         len(ret_a_pre), len(ret_a_post), len(shared_a)),
        ("Option C",
         mu_c_pre, cov_c_pre, mu_c_post, cov_c_post,
         RF_PRE, RF_POST,
         len(ret_c_pre), len(ret_c_post), len(shared_c)),
    ]:
        for period, mu, cov, rf, n_obs in [
            ("Pre",  mu_pre,  cov_pre,  rf_pre,  n_pre),
            ("Post", mu_post, cov_post, rf_post, n_post),
        ]:
            w_mv = min_var(mu, cov)
            w_ms = max_sharpe(mu, cov, rf)
            r_mv, v_mv = pstats(w_mv, mu, cov)
            r_ms, v_ms = pstats(w_ms, mu, cov)
            sr = (r_ms - rf) / v_ms
            rows.append({
                "Option":    label,
                "Period":    period,
                "N_tickers": n_tkr,
                "T_obs":     n_obs,
                "T/N":       round(n_obs / n_tkr, 2),
                "RF":        rf,
                "GMV_vol":   round(v_mv, 4),
                "GMV_ret":   round(r_mv, 4),
                "MS_vol":    round(v_ms, 4),
                "MS_ret":    round(r_ms, 4),
                "Sharpe":    round(sr, 3),
                "Avg_corr":  round(
                    avg_pre if period == "Pre" else avg_post, 3),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv("Results/summary_stats.csv", index=False)
    print(summary.to_string(index=False))

    print("\n" + "=" * 50)
    print("  All outputs saved to Results/")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
