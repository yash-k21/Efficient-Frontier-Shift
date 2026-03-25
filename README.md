# Efficient Frontier Shift

**Authors:** Viraat Arora, Yash Khaitan, Dersh Savla
**Course:** IQF Midterm Project

## Research Question

Does the efficient frontier for Indian equities shift meaningfully between the **pre-COVID (2015–2019)** and **post-COVID (2020–2024)** periods, and if so, what structural, sectoral, and macroeconomic forces drive this shift in risk-return trade-offs?

## Data

- **Universe:** 50 stocks selected from NIFTY 100 constituents as of January 1, 2015 (fixed — no problematic/delisted stocks)
- **Source:** Bloomberg Terminal for constituents; Yahoo Finance (`.NS` tickers) for price data
- **Frequency:** Daily and monthly closing prices
- **Outputs per period:** Monthly returns, mean returns vector, covariance matrix
- **Risk-free rate:** Yield on 91-day Government of India Treasury Bill (RBI)

## Method

Three measures to quantify the shift in the efficient frontier:

1. **Change in slope of the CAL** — higher slope = better risk-adjusted performance (Sharpe ratio of tangency portfolio)
2. **Return at fixed risk levels** — compare expected returns achievable at the same volatility across periods
3. **Minimum variance portfolio** — compare return and variance of the MVP across periods

Efficient frontier computed using **long-only constrained optimization** (scipy) — no short selling. Additionally: examine sectoral contributions to tangency portfolio weights across the two periods.

## Project Structure

```
Efficient-Frontier-Shift/
├── NIFTY100_2015.xlsx              # NIFTY 100 constituents with Yahoo Finance tickers (col 6)
├── Special_Stocks_Treatment.xlsx   # Treatment guide for delisted/merged/problematic stocks
├── data.py                         # Download daily & monthly prices for 50 clean tickers → Data/
├── mu_sigma.py                     # Compute monthly returns, mu, cov → Data/monthly/
├── mu_sigma_daily.py               # Compute daily returns, mu, cov → Data/daily/
├── 3_frontier.py                   # Simulate feasible set + long-only efficient frontier (combined plot)
├── Data/
│   ├── prices_daily.csv
│   ├── prices_monthly.csv
│   ├── monthly/
│   │   ├── pre-covid/              # returns_pre_monthly.csv, mu_pre_monthly.csv, sigma_pre_monthly.csv
│   │   └── post-covid/             # returns_post_monthly.csv, mu_post_monthly.csv, sigma_post_monthly.csv
│   └── daily/
│       ├── pre-covid/              # returns_pre_daily.csv, mu_pre_daily.csv, sigma_pre_daily.csv
│       └── post-covid/             # returns_post_daily.csv, mu_post_daily.csv, sigma_post_daily.csv
├── Figures/
│   ├── frontier_pre_covid.png
│   ├── frontier_post_covid.png
│   └── frontier_combined.png
├── IQF_Midterm_Project_Proposal.pdf
└── README.md
```

## Environment

```bash
conda activate eff-frontier
# run scripts directly or:
jupyter lab   # kernel: "Efficient Frontier (Python 3.11)"
```

**Packages:** `yfinance`, `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `openpyxl`

## Progress

- [x] Obtain NIFTY 100 constituents (Jan 1, 2015 and Dec 31, 2015)
- [x] Map Bloomberg tickers to Yahoo Finance `.NS` tickers
- [x] Handle problematic stocks (delisted, merged, wrong tickers) per treatment sheet
- [x] Select 50 clean tickers with full 2015-2024 coverage
- [x] Set up `eff-frontier` conda environment + Jupyter kernel
- [x] `data.py` — download daily and monthly closing prices for 50 tickers (2015-2024)
- [x] `mu_sigma.py` — monthly returns, mean vector, covariance matrix (pre and post-COVID)
- [x] `mu_sigma_daily.py` — same for daily data
- [x] `3_frontier.py` — simulate feasible set (Dirichlet) + long-only efficient frontier (scipy)
- [x] Combined frontier plot (pre vs post-COVID on same axes)
- [ ] Compute tangency portfolio and CAL slope per period
- [ ] Compare MVP across periods
- [ ] Sectoral analysis of tangency portfolio weights
- [ ] Interpret shifts via macroeconomic/structural lens
