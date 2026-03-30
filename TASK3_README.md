# Task 3: Risk-Free Rate, Sharpe Ratio & CAL Implementation

## Overview
This implementation adds risk-free rate integration, Sharpe ratio computation, and Capital Allocation Line (CAL) plotting to the efficient frontier pipeline. The design is modular and compatible with any time period splits or return frequencies.

## New Files Added

### 1. `utils.py`
Reusable utility functions for financial computations:
- **Annualization functions**: Convert returns/volatility/covariance across frequencies
- **Risk-free rate loader**: Load and match risk-free rates to specific periods
- **Sharpe ratio computation**: Calculate risk-adjusted returns
- **Tangency portfolio**: Find maximum Sharpe ratio portfolio using optimization
- **CAL computation**: Generate Capital Allocation Line coordinates

### 2. `1_risk_free_data.py`
Downloads and prepares risk-free rate data:
- Downloads US 3-month Treasury rates (proxy for risk-free rate)
- Fallback to reasonable historical rates if download fails
- Saves to `Data/risk_free_rates.csv` with daily data
- Computes period averages (pre-COVID: 1.40%, post-COVID: 2.78%)

### 3. `3_frontier_with_sharpe.py`
Enhanced efficient frontier analysis with:
- **Risk-free rate integration** for all frequencies
- **Annualized returns and covariances** for proper comparison
- **Tangency portfolio computation** (maximum Sharpe ratio)
- **CAL plotting** from risk-free rate through tangency portfolio
- **Sharpe ratio change analysis** between pre and post periods
- **Enhanced visualizations** with annotations

## Generated Outputs

### Figures
- `Figures/frontier_with_sharpe_daily.png` - Daily frequency with CAL
- `Figures/frontier_with_sharpe_weekly.png` - Weekly frequency with CAL
- `Figures/frontier_with_sharpe_monthly.png` - Monthly frequency with CAL
- `Figures/sharpe_ratio_summary.csv` - Summary statistics

### Data Files
- `Data/risk_free_rates.csv` - Daily risk-free rates (2015-2026)
- `Data/risk_free_summary.txt` - Summary statistics

## Key Results

### Sharpe Ratio Analysis
```
Frequency   Pre-COVID SR   Post-COVID SR   Change      % Change
Daily       1.638          1.749           +0.111      +6.8%
Weekly      1.614          1.712           +0.098      +6.1%
Monthly     1.438          1.801           +0.363      +25.2%
```

**Key Insight**: Post-COVID period shows improved risk-adjusted returns across all frequencies, with monthly data showing the most significant improvement (25.2% increase in Sharpe ratio).

## Integration with Tasks 1 & 2

### For Task 1 (Time Period Modifications)
The implementation is **period-agnostic**:

```python
# Instead of hard-coded dates, use parameters:
from utils import load_risk_free_rate

# Load risk-free rate for any period
rf_rate = load_risk_free_rate(start_date='2017', end_date='2019', frequency='daily')

# Works with any date range your teammates define
rf_bull_run = load_risk_free_rate('2017-01-01', '2018-12-31', 'weekly')
rf_crash = load_risk_free_rate('2020-03-01', '2020-06-30', 'monthly')
```

### For Task 2 (Extended Frequencies)
The implementation supports **any frequency**:

```python
from utils import get_annualization_factor, annualize_returns

# Existing frequencies
ann_factor_daily = get_annualization_factor('daily')      # 252
ann_factor_weekly = get_annualization_factor('weekly')    # 52
ann_factor_monthly = get_annualization_factor('monthly')  # 12

# Future frequencies (already supported)
ann_factor_bimonthly = get_annualization_factor('bimonthly')    # 6
ann_factor_quarterly = get_annualization_factor('quarterly')    # 4

# To add new frequencies, just update ANNUALIZATION_FACTORS in utils.py
```

## Usage

### Running the Enhanced Analysis
```bash
# Activate virtual environment
source venv/bin/activate

# Run the enhanced frontier analysis
python 3_frontier_with_sharpe.py
```

### Using Utility Functions in Your Code
```python
from utils import (
    annualize_returns,
    annualize_covariance_matrix,
    load_risk_free_rate,
    compute_tangency_portfolio,
    compute_sharpe_ratio
)

# Example: Compute tangency portfolio for your custom period
mean_returns = pd.read_csv('Data/your_custom_period/mu.csv').squeeze().values
cov_matrix = pd.read_csv('Data/your_custom_period/sigma.csv').values

# Annualize (assuming monthly data)
mean_ann = annualize_returns(mean_returns, 'monthly')
cov_ann = annualize_covariance_matrix(cov_matrix, 'monthly')

# Get risk-free rate for your period
rf_rate = load_risk_free_rate('2020-01-01', '2020-12-31', 'monthly')

# Find tangency portfolio
tangency = compute_tangency_portfolio(mean_ann, cov_ann, rf_rate)
print(f"Optimal Sharpe Ratio: {tangency['sharpe']:.4f}")
print(f"Portfolio Weights: {tangency['weights']}")
```

## Compatibility Notes

### Your teammates can:
1. **Modify time periods** in their `2_mu_sigma_*.py` scripts without changing Task 3 code
2. **Add new frequencies** by adding entries to `ANNUALIZATION_FACTORS` in `utils.py`
3. **Use the utility functions** in their own analysis scripts
4. **Run the original `3_frontier.py`** (still works) or the new `3_frontier_with_sharpe.py`

### The code will automatically:
- Load correct risk-free rates for any period using date range matching
- Annualize returns properly for any frequency
- Compute tangency portfolios and CAL for any efficient frontier
- Generate enhanced plots with Sharpe ratio annotations

## Technical Details

### Capital Allocation Line (CAL)
The CAL shows all possible portfolios combining:
- **Risk-free asset** (zero volatility, guaranteed return)
- **Tangency portfolio** (optimal risky portfolio with max Sharpe ratio)

Equation: `E[R_portfolio] = Rf + Sharpe_tangency × σ_portfolio`

Points on the CAL:
- **Left of tangency**: Partially invested in risk-free asset (lending)
- **At tangency**: 100% in risky portfolio
- **Right of tangency**: Leveraged (borrowing at risk-free rate)

### Tangency Portfolio Optimization
Maximizes Sharpe ratio subject to:
- Weights sum to 1 (fully invested)
- Weights ≥ 0 (long-only constraint)
- No short selling

Solver: SciPy's SLSQP (Sequential Least Squares Programming)

### Annualization Methodology
- **Returns**: Linear scaling (multiply by factor)
- **Volatility**: Square root scaling (multiply by √factor)
- **Covariance**: Linear scaling (multiply by factor)

## Dependencies

All dependencies installed in virtual environment:
- pandas
- numpy
- matplotlib
- scipy
- yfinance

## Files Modified
None! All existing scripts remain unchanged.

## Files Created
1. `utils.py` - Utility functions
2. `1_risk_free_data.py` - Risk-free rate data script
3. `3_frontier_with_sharpe.py` - Enhanced frontier analysis
4. `Data/risk_free_rates.csv` - Risk-free rate data
5. `Data/risk_free_summary.txt` - Risk-free rate statistics
6. `TASK3_README.md` - This documentation

## Next Steps for Integration

When your teammates complete Tasks 1 & 2:

1. **Update period definitions**: Modify date ranges in their scripts
2. **Add new frequencies**: Update `ANNUALIZATION_FACTORS` if needed
3. **Run enhanced analysis**: Execute `python 3_frontier_with_sharpe.py`
4. **Compare results**: Use the Sharpe ratio summary to analyze risk-adjusted performance across periods

No code changes needed in Task 3 files! 🎉
