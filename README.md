# Efficient Frontier Shift

**Authors:** Viraat Arora, Yash Khaitan, Dersh Savla
**Course:** IQF Midterm Project

## Research Question

Does the efficient frontier for Indian equities shift meaningfully between the **pre-COVID (2015–2019)** and **post-COVID (2020–2024)** periods, and if so, what structural, sectoral, and macroeconomic forces drive this shift in risk-return trade-offs?

## Data

- **Universe:** 100 stocks from NIFTY 100 constituents as of **January 1, 2015** (fixed — does not change across periods)
- **Source:** Bloomberg Terminal for constituents; Yahoo Finance for monthly price data
- **Frequency:** Monthly closing prices → monthly returns
- **Outputs per period:** Vector of mean returns, covariance matrix of returns
- **Risk-free rate:** Yield on 91-day Government of India Treasury Bill (RBI)

## Method

Three measures to quantify the shift in the efficient frontier:

1. **Change in slope of the CAL** — higher slope = better risk-adjusted performance (Sharpe ratio of tangency portfolio)
2. **Return at fixed risk levels** — compare expected returns achievable at the same volatility across periods
3. **Minimum variance portfolio** — compare return and variance of the MVP across periods

Additionally: examine sectoral contributions to tangency portfolio weights across the two periods.

## Project Structure

```
Efficient-Frontier-Shift/
├── NIFTY100_2015.xlsx       # NIFTY 100 constituents (Jan 1 & Dec 31 2015) with Yahoo Finance tickers
├── data.ipynb               # Pull monthly price data from Yahoo Finance
├── IQF_Midterm_Project_Proposal.pdf
└── README.md
```

## Environment

```bash
conda activate eff-frontier
jupyter lab
# Select kernel: "Efficient Frontier (Python 3.11)"
```

**Packages:** `yfinance`, `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `openpyxl`

## Progress

- [x] Obtain NIFTY 100 constituents (Jan 1, 2015 and Dec 31, 2015)
- [x] Map Bloomberg tickers → Yahoo Finance `.NS` tickers
- [x] Set up `eff-frontier` conda environment + Jupyter kernel
- [x] `data.ipynb` — pull monthly closing prices for all tickers (2015–2024)
- [ ] Compute monthly returns
- [ ] Split into pre-COVID (2015–2019) and post-COVID (2020–2024) periods
- [ ] Compute mean returns vector and covariance matrix per period
- [ ] Construct efficient frontier for each period
- [ ] Compute tangency portfolio and CAL slope per period
- [ ] Compare MVP across periods
- [ ] Sectoral analysis of tangency portfolio weights
- [ ] Interpret shifts via macroeconomic/structural lens
