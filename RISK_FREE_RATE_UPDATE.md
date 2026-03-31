# Risk-Free Rate Update Summary

## Changes Made

The risk-free rate data has been updated to use **actual Indian 91-Day Government Treasury Bill rates from RBI** for the appropriate historical periods, which is more suitable for analyzing NIFTY stocks than US rates.

## Data Source

- **Source**: Reserve Bank of India (RBI)
- **Security**: 91-Day Government of India Treasury Bills
- **Data Type**: Weekly auction results converted to daily rates (forward fill)
- **Period**: January 1, 2015 to March 26, 2025
- **Total Days**: 3,738 days
- **Total Auctions**: 526 auctions (avg. one every 7 days)

## Key Statistics

### Period Averages (Annualized):
- **Pre-COVID (2015-2019)**: 6.58%
- **Post-COVID (2020-2024)**: 5.13%
- **Full Period (2015-2024)**: 5.85%
- **Overall Mean**: 5.87%

### Historical Rate Ranges:
- **2015**: 8.31% (high inflation environment)
- **2019**: ~5.01% (pre-COVID normalized rates)
- **2020**: ~4.94% (COVID period rates)
- **2023-2024**: ~4.5-5.5% (post-COVID stabilization)
- **2025**: 6.47% (current rates)

### Why Indian Rates are Higher:
Indian T-Bill rates are approximately 3-5% higher than US rates, which reflects:
- **Inflation differential** between India and US
- **Currency risk premium** for INR vs USD
- **Economic development stage** differences
- **Monetary policy** differences between RBI and Fed

## Files Created/Updated

1. **1_risk_free_data_india.py** ⭐ NEW
   - Processes RBI 91-Day T-Bill auction data from Excel
   - Converts weekly auction data to daily rates using forward fill
   - Handles date parsing and yield conversion (% to decimal)
   - Creates continuous daily risk-free rate series

2. **1_risk_free_data.py** (Original - US data)
   - Still available for reference/comparison
   - Downloads US 91-Day T-Bill rates from Yahoo Finance (^IRX)
   - Can be used for international comparisons

3. **Data/risk_free_rates.csv**
   - Now contains actual daily Indian 91-Day T-Bill rates from RBI
   - Rates are in decimal format (e.g., 0.0585 = 5.85%)
   - Auction rates forward-filled to create continuous daily series
   - Covers January 1, 2015 to March 26, 2025

4. **Data/risk_free_summary.txt**
   - Updated to reflect the Indian 91-Day T-Bill data source
   - Includes period statistics and descriptive statistics
   - Documents the RBI auction data processing method

## Usage

The risk-free rate data is automatically loaded by:
- `utils.py`: `load_risk_free_rate()` function
- `3_frontier_with_sharpe.py`: For Sharpe ratio calculations
- Any other analysis requiring risk-free rates

The rates are already annualized and ready to use in portfolio optimization and Sharpe ratio calculations.

## Data Processing Method

The RBI provides auction results typically **once per week** for 91-Day T-Bills. To create a continuous daily series:

1. **Auction Data**: Extract weighted average yield from each auction
2. **Forward Fill**: Each auction rate is used for all days until the next auction
3. **Conversion**: Yields converted from percentage (6.5%) to decimal (0.065)
4. **Date Range**: Series extended to match stock data (2015-2025)

This approach is standard practice and reflects the real market behavior where T-Bill rates remain constant between auctions.

## Verification

Sample verification shows the data accurately reflects Indian market conditions:
- ✅ High rates in early 2015 (~8.31%) reflecting inflation concerns
- ✅ Gradual decline 2016-2019 as inflation moderated
- ✅ Relatively stable rates during COVID (~4.9-5.0%)
- ✅ Post-COVID stabilization around 5-6%
- ✅ Recent rates around 6.47% (2025)

## How to Update Data in Future

When you need to refresh the data:

1. **Download from RBI**:
   - Visit: https://dbie.rbi.org.in/DBIE/dbie.rbi?site=statistics
   - Navigate to: Financial Markets → Government Securities Market → Auctions
   - Download: "91-Day Treasury Bill" data as Excel

2. **Replace the Excel file**:
   - Save to: `/Users/viraatarora/Downloads/Auctions of 91-Day Government of India Treasury Bills.xlsx`
   - Or update the path in `1_risk_free_data_india.py`

3. **Run the script**:
   ```bash
   cd /Users/viraatarora/Documents/Efficient-Frontier-Shift
   source venv/bin/activate
   python3 1_risk_free_data_india.py
   ```

4. **Verify**: Check that `Data/risk_free_rates.csv` is updated
