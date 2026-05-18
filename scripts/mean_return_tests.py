"""
mean_return_test.py
====================
Stock-level comparison of mean returns between the pre-COVID and
post-COVID estimation windows using Newey-West t-tests with three
levels of multiple testing correction at two significance levels:
  Corrections : None / Bonferroni / Benjamini-Hochberg FDR
  Alpha levels: 5% and 10%

References
----------
Newey, W.K. and West, K.D. (1987). Econometrica 55(3): 703-708.
Bonferroni, C.E. (1936). Pubblicazioni del R Istituto Superiore di
  Scienze Economiche e Commerciali di Firenze, 8: 3-62.
Benjamini, Y. and Hochberg, Y. (1995). JRSS-B 57(1): 289-300.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

os.makedirs("../results/diagnostics", exist_ok=True)

PRE_START  = "2015-01-01"
PRE_END    = "2020-01-31"
POST_START = "2021-07-01"
POST_END   = "2024-12-31"
ANN        = 252


def load_data():
    prices = pd.read_csv("../data/processed/prices_daily.csv",
                         index_col=0, parse_dates=True)
    prices = prices[~prices.index.astype(str).str.contains(
        "Price|Ticker", na=False)]
    prices = prices.astype(float).dropna(axis=1, how="all").sort_index()
    log_ret = np.log(prices / prices.shift(1)).iloc[1:]

    def slice_period(ret, start, end):
        r    = ret.loc[start:end]
        keep = r.columns[r.isna().mean() <= 0.10]
        return r[keep].ffill(limit=1).dropna(axis=1)

    ret_pre  = slice_period(log_ret, PRE_START,  PRE_END)
    ret_post = slice_period(log_ret, POST_START, POST_END)
    common   = ret_pre.columns.intersection(ret_post.columns)

    print(f"  Universe : {len(common)} stocks")
    print(f"  Pre  obs : {len(ret_pre)}   Post obs: {len(ret_post)}")
    return ret_pre[common], ret_post[common]


def _nw_se(x):
    """Newey-West HAC standard error for the sample mean of x."""
    x = np.asarray(x, dtype=float)
    T = len(x)
    e = x - x.mean()

    rho    = np.clip(np.corrcoef(e[:-1], e[1:])[0, 1], -0.99, 0.99)
    alpha1 = 4 * rho**2 / (1 - rho)**4
    m      = int(np.ceil(1.1447 * (alpha1 * T) ** (1/3)))
    m      = min(max(1, m), T // 5)

    lrv = np.dot(e, e) / T
    for lag in range(1, m + 1):
        w   = 1 - lag / (m + 1)
        lrv += 2 * w * np.dot(e[lag:], e[:-lag]) / T

    return float(np.sqrt(max(lrv, 0) / T))


def bh_reject(p_values, alpha):
    """Benjamini-Hochberg (1995) FDR correction at given alpha."""
    p     = np.asarray(p_values)
    n     = len(p)
    order = np.argsort(p)
    thresholds = alpha * (np.arange(1, n+1) / n)
    reject = np.zeros(n, dtype=bool)
    below  = np.where(p[order] <= thresholds)[0]
    if len(below):
        reject[order[:below[-1]+1]] = True
    return reject


def run_tests(ret_pre, ret_post):
    """Compute t-stats and add rejection columns for both alpha levels."""
    tickers = list(ret_pre.columns)
    n       = len(tickers)
    rows    = []

    for ticker in tickers:
        r1 = ret_pre[ticker].values
        r2 = ret_post[ticker].values

        mu1    = r1.mean()
        mu2    = r2.mean()
        diff   = mu2 - mu1
        se     = np.sqrt(_nw_se(r1)**2 + _nw_se(r2)**2)
        t_stat = diff / se if se > 0 else np.nan
        df     = min(len(r1), len(r2)) - 1
        p_val  = float(2 * stats.t.sf(abs(t_stat), df=df))

        rows.append({
            "ticker":        ticker,
            "mean_pre_ann":  round(float(mu1) * ANN, 4),
            "mean_post_ann": round(float(mu2) * ANN, 4),
            "diff_ann":      round(float(diff) * ANN, 4),
            "nw_se_ann":     round(float(se)   * ANN, 4),
            "t_stat":        round(float(t_stat),     3),
            "p_value":       round(p_val,             6),
        })

    df_out = pd.DataFrame(rows).set_index("ticker")

    for alpha in (0.05, 0.10):
        tag = "5pct" if alpha == 0.05 else "10pct"
        df_out[f"reject_none_{tag}"]       = df_out["p_value"] < alpha
        df_out[f"reject_bonferroni_{tag}"] = df_out["p_value"] < (alpha / n)
        df_out[f"reject_bh_{tag}"]         = bh_reject(df_out["p_value"].values, alpha)

    return df_out


def _print_section(df, col, label):
    sig = df[df[col]].sort_values("diff_ann", ascending=False)
    print(f"  Rejections: {len(sig)} / {len(df)}")
    if len(sig):
        print(f"  {'Ticker':15s}  {'Pre':>8}  {'Post':>8}"
              f"  {'Diff':>8}  {'t':>7}  {'p':>10}")
        print(f"  {'─'*60}")
        for ticker, row in sig.iterrows():
            print(f"  {ticker:15s}  {row['mean_pre_ann']:>7.1%}"
                  f"  {row['mean_post_ann']:>8.1%}"
                  f"  {row['diff_ann']:>+8.1%}"
                  f"  {row['t_stat']:>7.2f}"
                  f"  {row['p_value']:>10.6f}")


def main():
    print("\n" + "="*60)
    print("  Mean Return Tests: Newey-West + Multiple Testing")
    print("="*60 + "\n")

    ret_pre, ret_post = load_data()
    results = run_tests(ret_pre, ret_post)
    n = len(results)

    for alpha, tag, alpha_str in [(0.05, "5pct", "5%"), (0.10, "10pct", "10%")]:
        print(f"\n{'═'*60}")
        print(f"  SIGNIFICANCE LEVEL α = {alpha_str}")
        print(f"{'═'*60}")

        for col_sfx, label, note in [
            (f"reject_none_{tag}",
             "No correction",
             f"p < {alpha}"),
            (f"reject_bonferroni_{tag}",
             "Bonferroni",
             f"p < {alpha}/{n} = {alpha/n:.5f}"),
            (f"reject_bh_{tag}",
             "Benjamini-Hochberg FDR",
             f"FDR {alpha_str}"),
        ]:
            print(f"\n  ── {label} ({note}) ──")
            _print_section(results, col_sfx, label)

    # ── Summary grid ─────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  SUMMARY: number of stocks with significantly different means")
    print(f"{'═'*60}")
    print(f"  {'Correction':25s}  {'α = 5%':>8}  {'α = 10%':>8}")
    print(f"  {'─'*45}")
    for sfx5, sfx10, label in [
        ("reject_none_5pct",       "reject_none_10pct",
         "None"),
        ("reject_bonferroni_5pct", "reject_bonferroni_10pct",
         "Bonferroni"),
        ("reject_bh_5pct",         "reject_bh_10pct",
         "Benjamini-Hochberg"),
    ]:
        r5  = int(results[sfx5].sum())
        r10 = int(results[sfx10].sum())
        print(f"  {label:25s}  {r5:>5}/{n}   {r10:>5}/{n}")

    results.to_csv("../results/diagnostics/mean_return_tests.csv")
    print("\n  Saved: Results/Diagnostics/mean_return_tests.csv\n")


if __name__ == "__main__":
    main()