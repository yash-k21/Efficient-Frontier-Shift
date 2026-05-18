"""
structural_break_motivation.py
================================
CUSUM and CUSUM-sq structural break tests on NIFTY 50 index returns.

Test 1 — CUSUM/CUSUM-sq on full sample (2015-2024)
  Detects instability in mean and variance across the full period.
  Motivates investigating whether the efficient frontier shifted.

Test 2 — CUSUM on post-COVID sub-sample (Jul 2021 - Dec 2024)
  Tests whether the post-COVID regime is internally stable.
  If the statistic stays within bounds -> the shift is permanent,
  not transient — validating the post-COVID estimation window.

Reference
---------
Brown, R.L., Durbin, J. and Evans, J.M. (1975). Techniques for
  testing the constancy of regression relationships over time.
  JRSS-B 37(2): 149-192.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.regression.recursive_ls import RecursiveLS

os.makedirs("Results/Diagnostics", exist_ok=True)

POST_START = "2021-07-01"
POST_END   = "2024-12-31"


def load_nifty50():
    nifty = yf.download('^NSEI', start='2015-01-01', end='2025-01-01',
                        auto_adjust=True, progress=False)['Close'].squeeze()
    nifty = nifty.dropna()
    print(f"  Downloaded NIFTY 50: {nifty.index[0].date()} "
          f"to {nifty.index[-1].date()}  ({len(nifty)} obs)")
    return nifty


def _run_cusum(series, label):
    """
    Fit RecursiveLS on a constant and return CUSUM crossing results.
    Works for any sub-period passed in as series.
    """
    endog = series.values
    exog  = np.ones((len(endog), 1))

    res = RecursiveLS(endog, exog).fit()

    lo_c,  hi_c  = res._cusum_significance_bounds(alpha=0.05)
    lo_sq, hi_sq = res._cusum_squares_significance_bounds(alpha=0.05)

    hi_c_full  = np.linspace(hi_c[0],  hi_c[1],  len(res.cusum))
    hi_sq_full = np.linspace(hi_sq[0], hi_sq[1], len(res.cusum_squares))

    dates = series.index[1:]

    def first_cross(stat, hi):
        idx = np.where(np.abs(stat) > np.abs(hi))[0]
        return dates[idx[0]].strftime("%Y-%m-%d") if len(idx) else "none"

    first_cusum = first_cross(res.cusum,         hi_c_full)
    first_cusq  = first_cross(res.cusum_squares,  hi_sq_full)
    reject_c    = first_cusum != "none"
    reject_sq   = first_cusq  != "none"

    print(f"\n  [{label}]")
    print(f"  CUSUM    first crossing: {first_cusum}  reject={reject_c}")
    print(f"  CUSUM-sq first crossing: {first_cusq}   reject={reject_sq}")

    return {
        f"{label}_first_crossing_cusum":    first_cusum,
        f"{label}_first_crossing_cusum_sq": first_cusq,
        f"{label}_reject_mean_stability":   reject_c,
        f"{label}_reject_var_stability":    reject_sq,
    }


def main():
    print("\n" + "="*60)
    print("  Structural Break Tests: CUSUM / CUSUM-sq")
    print("="*60 + "\n")

    nifty   = load_nifty50()
    log_ret = np.log(nifty / nifty.shift(1)).dropna()

    # Test 1: Full sample
    print("-- Test 1: Full sample (2015-2024) --")
    r1 = _run_cusum(log_ret, "full_sample")

    if r1["full_sample_reject_mean_stability"] and r1["full_sample_reject_var_stability"]:
        print("\n  -> Both mean and variance instability detected.")
        print("     Motivates testing whether the efficient frontier shifted.")

    # Test 2: Post-COVID sub-sample
    print(f"\n-- Test 2: Post-COVID sub-sample ({POST_START}-{POST_END}) --")
    r2 = _run_cusum(log_ret.loc[POST_START:POST_END], "post_covid")

    r2_stable = (not r2["post_covid_reject_mean_stability"] and
                 not r2["post_covid_reject_var_stability"])

    if r2_stable:
        print("\n  -> Post-COVID regime is internally stable.")
        print("     The structural break identified above is permanent —")
        print("     the post-COVID window represents a settled regime.")
    else:
        print("\n  -> Post-COVID regime shows further instability.")
        print("     Interpret post-COVID frontier estimates with caution.")

    # Save
    results = {
        **r1, **r2,
        "post_regime_stable": r2_stable,
        "shift_permanent":    r2_stable,
        "test":               "Brown-Durbin-Evans (1975) CUSUM",
        "significance_level": "5%",
    }
    pd.DataFrame([results]).to_csv(
        "Results/Diagnostics/cusum_summary.csv", index=False
    )
    print("\n  Saved: Results/Diagnostics/cusum_summary.csv\n")


if __name__ == "__main__":
    main()