# Efficient Frontier Shift

**Authors:** Viraat Arora, Yash Khaitan, Dersh Savla  
**Course:** IQF Midterm Project

## Research Question

Does the efficient frontier for Indian equities shift meaningfully between the **pre-COVID (2015–2020)** and **post-COVID (2021–2024)** periods, and if so, what structural, sectoral, and macroeconomic forces drive this shift in risk-return trade-offs?

## Data

| Item | Detail |
|------|--------|
| Universe | 45 stocks from NIFTY 100 constituents as of Jan 1, 2015 (fixed — no delisted/problematic stocks) |
| Source | Bloomberg Terminal for constituents; Yahoo Finance (`.NS` tickers) for prices |
| Frequency | Daily and weekly closing prices (2015–2024) |
| Risk-free rate | 91-day Government of India Treasury Bill yield (RBI) |
| Pre-COVID window | Jan 1, 2015 – Jan 31, 2020 |
| Crisis window | Feb 1, 2020 – Jun 30, 2021 (excluded from main analysis) |
| Post-COVID window | Jul 1, 2021 – Dec 31, 2024 |

## Method

- **Covariance estimation:** Ledoit-Wolf shrinkage (better small-sample properties)
- **Optimization:** Long-only constrained mean-variance (scipy SLSQP), 250-point frontier
- **Sharpe ratio test:** Ledoit-Wolf studentized circular block bootstrap (Eq. 9, LW 2008)
- **Mean return tests:** Newey-West HAC t-tests with Bonferroni and BH-FDR corrections
- **Normality diagnostics:** Jarque-Bera + Engle ARCH-LM
- **Structural breaks:** ICSS algorithm (Inclan & Tiao 1994) on NIFTY 50 returns
- **Robustness checks:** transaction costs, estimator comparison, bootstrap frontier bands, sector weights, VaR/CVaR, downside risk

## Repository Structure

```
Efficient-Frontier-Shift/
├── scripts/
│   ├── 01_download_data.py           # Download daily/weekly/monthly prices via yfinance
│   ├── 02_compute_returns_daily.py   # Daily log returns, mu, cov → data/processed/daily/
│   ├── 02_compute_returns_weekly.py  # Weekly log returns, mu, cov → data/processed/weekly/
│   ├── 03_efficient_frontier.py      # Core analysis: frontiers, CAL, weights, correlations
│   ├── 04_nifty_index_plots.py       # NIFTY 50 index level and return plots (context)
│   ├── 05_robustness_tests.py        # Full robustness & diagnostic suite
│   ├── lw_boot_sharpe_test.py        # Utility: LW bootstrap Sharpe ratio test
│   ├── mean_return_tests.py          # Stock-level mean return HAC t-tests
│   └── normality_tests.py            # Normality + ICSS structural break tests
├── data/
│   ├── raw/
│   │   ├── nifty100_constituents_2015.xlsx   # NIFTY 100 tickers (Bloomberg → Yahoo Finance)
│   │   ├── special_stock_treatments.xlsx     # Handling of delisted/merged stocks
│   │   └── rbi_tbill_91day_yields.xlsx       # RBI 91-day T-bill auction yields
│   └── processed/
│       ├── prices_daily.csv
│       ├── prices_weekly.csv
│       ├── prices_monthly.csv
│       ├── daily/{pre-covid,post-covid}/     # returns, mu, sigma CSVs
│       ├── weekly/{pre-covid,post-covid}/
│       └── monthly/{pre-covid,post-covid}/
├── results/
│   ├── figures/          # Efficient frontier plots, tangency weights, correlation heatmap, rolling vol
│   ├── diagnostics/      # Normality tests, ICSS breaks, mean return test tables
│   └── robustness/       # Bootstrap bands, transaction costs, sector weights, VaR/CVaR, etc.
├── archives/             # Superseded scripts and earlier analysis outputs
├── README.md
└── context.md
```

## Execution Order

Run from the **repository root**:

```bash
# 1. Download price data
python scripts/01_download_data.py

# 2. Compute returns (run both)
python scripts/02_compute_returns_daily.py
python scripts/02_compute_returns_weekly.py

# 3. Main analysis — efficient frontiers, weights, correlations, rolling vol
python scripts/03_efficient_frontier.py

# 4. Market context — NIFTY 50 index and return plots
python scripts/04_nifty_index_plots.py

# 5. Statistical tests
python scripts/mean_return_tests.py
python scripts/normality_tests.py

# 6. Robustness suite (long-running — ~5–10 min)
python scripts/05_robustness_tests.py
```

## Environment

```bash
conda activate eff-frontier
```

**Packages:** `yfinance`, `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `openpyxl`, `statsmodels`
