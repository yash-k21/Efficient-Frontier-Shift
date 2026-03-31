# Risk-Free Rate Update Summary

## Changes Made

The risk-free rate data has been updated to use **actual 91-Day U.S. Treasury Bill rates** from the appropriate historical periods instead of fallback/constant values.

## Data Source

- **Ticker**: ^IRX (13-week/91-day Treasury Bill yield)
- **Period**: January 2, 2015 to March 27, 2026
- **Total Trading Days**: 2,824 days

## Key Statistics

### Period Averages (Annualized):
- **Pre-COVID (2015-2019)**: 1.05%
- **Post-COVID (2020-2024)**: 2.47%
- **Full Period (2015-2024)**: 1.76%
- **Overall Mean**: 2.00%

### Historical Rate Ranges:
- **2015**: 0.04% (near-zero rate environment)
- **2019**: 2.04% (pre-COVID normalized rates)
- **2020**: 0.34% (COVID emergency low rates)
- **2023-2024**: 5.00% (post-COVID rate hikes)
- **2026**: 3.58% (current rates)

## Files Updated

1. **1_risk_free_data.py**
   - Updated documentation to clarify using 91-Day T-Bills
   - Fixed data handling for multi-level pandas columns from yfinance
   - Enhanced output messaging to specify "91-Day T-Bill"
   - Improved error handling and data validation

2. **Data/risk_free_rates.csv**
   - Now contains actual daily 91-Day T-Bill rates from ^IRX ticker
   - Rates are in decimal format (e.g., 0.0200 = 2.00%)
   - Missing days (weekends/holidays) filled using forward-fill method

3. **Data/risk_free_summary.txt**
   - Updated to reflect the 91-Day T-Bill data source
   - Includes period statistics and descriptive statistics

## Usage

The risk-free rate data is automatically loaded by:
- `utils.py`: `load_risk_free_rate()` function
- `3_frontier_with_sharpe.py`: For Sharpe ratio calculations
- Any other analysis requiring risk-free rates

The rates are already annualized and ready to use in portfolio optimization and Sharpe ratio calculations.

## Verification

Sample verification shows the data accurately reflects market conditions:
- ✅ Near-zero rates in early 2015
- ✅ Gradual increases 2016-2019
- ✅ Emergency rate cuts in March 2020
- ✅ Rate hikes 2022-2023 reaching ~5%
- ✅ Recent stabilization around 3.6%
