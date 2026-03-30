"""
Utility functions for efficient frontier analysis with risk-free rates.

This module provides reusable utilities for:
- Annualizing returns across different frequencies
- Loading risk-free rate data
- Computing Sharpe ratios
- Finding tangency portfolios
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# Annualization factors for different return frequencies
ANNUALIZATION_FACTORS = {
    'daily': 252,      # Trading days per year
    'weekly': 52,      # Weeks per year
    'monthly': 12,     # Months per year
    'bimonthly': 6,    # Bi-monthly periods per year
    'quarterly': 4,    # Quarters per year
}


def get_annualization_factor(frequency):
    """
    Get the annualization factor for a given return frequency.
    
    Parameters:
    -----------
    frequency : str
        Return frequency ('daily', 'weekly', 'monthly', 'bimonthly', 'quarterly')
        
    Returns:
    --------
    int : Annualization factor
    
    Examples:
    ---------
    >>> get_annualization_factor('daily')
    252
    >>> get_annualization_factor('monthly')
    12
    """
    freq_lower = frequency.lower()
    if freq_lower not in ANNUALIZATION_FACTORS:
        raise ValueError(f"Unknown frequency: {frequency}. "
                       f"Supported: {list(ANNUALIZATION_FACTORS.keys())}")
    return ANNUALIZATION_FACTORS[freq_lower]


def annualize_returns(returns, frequency):
    """
    Annualize mean returns from a given frequency.
    
    Parameters:
    -----------
    returns : float, np.ndarray, or pd.Series
        Mean returns at the specified frequency (e.g., daily mean return)
    frequency : str
        Return frequency ('daily', 'weekly', 'monthly', etc.)
        
    Returns:
    --------
    Annualized returns (same type as input)
    
    Examples:
    ---------
    >>> annualize_returns(0.001, 'daily')  # 0.1% daily return
    0.252  # 25.2% annualized
    """
    factor = get_annualization_factor(frequency)
    return returns * factor


def annualize_volatility(volatility, frequency):
    """
    Annualize volatility (standard deviation) from a given frequency.
    
    Parameters:
    -----------
    volatility : float, np.ndarray, or pd.Series
        Volatility (standard deviation) at the specified frequency
    frequency : str
        Return frequency ('daily', 'weekly', 'monthly', etc.)
        
    Returns:
    --------
    Annualized volatility (same type as input)
    
    Note:
    -----
    Volatility scales with sqrt(time), not linearly like returns.
    
    Examples:
    ---------
    >>> annualize_volatility(0.02, 'daily')  # 2% daily volatility
    0.3178  # 31.78% annualized
    """
    factor = get_annualization_factor(frequency)
    return volatility * np.sqrt(factor)


def annualize_covariance_matrix(cov_matrix, frequency):
    """
    Annualize a covariance matrix from a given frequency.
    
    Parameters:
    -----------
    cov_matrix : np.ndarray or pd.DataFrame
        Covariance matrix at the specified frequency
    frequency : str
        Return frequency ('daily', 'weekly', 'monthly', etc.)
        
    Returns:
    --------
    Annualized covariance matrix (same type as input)
    
    Examples:
    ---------
    >>> cov_daily = np.array([[0.0004, 0.0001], [0.0001, 0.0009]])
    >>> cov_annual = annualize_covariance_matrix(cov_daily, 'daily')
    """
    factor = get_annualization_factor(frequency)
    return cov_matrix * factor


def load_risk_free_rate(period_start, period_end, frequency='daily', 
                        risk_free_file='Data/risk_free_rates.csv'):
    """
    Load risk-free rate for a specific period and frequency.
    
    Parameters:
    -----------
    period_start : str
        Start date (format: 'YYYY-MM-DD' or 'YYYY')
    period_end : str
        End date (format: 'YYYY-MM-DD' or 'YYYY')
    frequency : str, default 'daily'
        Return frequency to match ('daily', 'weekly', 'monthly')
    risk_free_file : str, default 'Data/risk_free_rates.csv'
        Path to risk-free rate CSV file
        
    Returns:
    --------
    float : Average annualized risk-free rate for the period
    
    Examples:
    ---------
    >>> rf_pre = load_risk_free_rate('2015', '2019', 'daily')
    >>> rf_post = load_risk_free_rate('2020', '2024', 'monthly')
    """
    try:
        rf_data = pd.read_csv(risk_free_file, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Warning: {risk_free_file} not found. Using default 6% risk-free rate.")
        return 0.06
    
    # Filter by date range
    rf_period = rf_data.loc[period_start:period_end]
    
    if rf_period.empty:
        print(f"Warning: No risk-free rate data for {period_start} to {period_end}. Using 6% default.")
        return 0.06
    
    # Get the rate column (should be annualized already)
    rate_col = rf_data.columns[0]
    avg_rate = rf_period[rate_col].mean()
    
    return avg_rate


def compute_sharpe_ratio(portfolio_return, portfolio_std, risk_free_rate):
    """
    Compute Sharpe ratio for a portfolio.
    
    Parameters:
    -----------
    portfolio_return : float or np.ndarray
        Expected portfolio return (annualized)
    portfolio_std : float or np.ndarray
        Portfolio volatility/standard deviation (annualized)
    risk_free_rate : float
        Risk-free rate (annualized)
        
    Returns:
    --------
    float or np.ndarray : Sharpe ratio
    
    Examples:
    ---------
    >>> compute_sharpe_ratio(0.15, 0.20, 0.06)  # 15% return, 20% vol, 6% rf
    0.45
    """
    return (portfolio_return - risk_free_rate) / portfolio_std


def compute_tangency_portfolio(mean_ret, cov, risk_free_rate):
    """
    Compute the tangency portfolio (maximum Sharpe ratio portfolio).
    
    This portfolio lies on the efficient frontier and has the highest
    Sharpe ratio. The Capital Allocation Line (CAL) passes through the
    risk-free asset and the tangency portfolio.
    
    Parameters:
    -----------
    mean_ret : np.ndarray or pd.Series
        Expected returns for each asset (annualized)
    cov : np.ndarray or pd.DataFrame
        Covariance matrix of returns (annualized)
    risk_free_rate : float
        Risk-free rate (annualized)
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'weights': np.ndarray of optimal portfolio weights
        - 'return': float, expected return of tangency portfolio
        - 'volatility': float, volatility of tangency portfolio
        - 'sharpe': float, Sharpe ratio of tangency portfolio
        
    Examples:
    ---------
    >>> mean_ret = np.array([0.10, 0.15, 0.12])
    >>> cov = np.array([[0.04, 0.01, 0.02],
    ...                  [0.01, 0.09, 0.03],
    ...                  [0.02, 0.03, 0.06]])
    >>> tangency = compute_tangency_portfolio(mean_ret, cov, 0.06)
    >>> print(f"Sharpe Ratio: {tangency['sharpe']:.3f}")
    """
    # Convert to numpy arrays
    mean_ret = np.array(mean_ret).flatten()
    cov = np.array(cov)
    n_assets = len(mean_ret)
    
    # Objective: minimize negative Sharpe ratio (maximize Sharpe)
    def neg_sharpe_ratio(weights):
        port_return = np.dot(weights, mean_ret)
        port_std = np.sqrt(np.dot(weights, np.dot(cov, weights)))
        if port_std == 0:
            return 1e10  # Avoid division by zero
        sharpe = (port_return - risk_free_rate) / port_std
        return -sharpe
    
    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    # Bounds: long-only portfolio (0 <= weight <= 1)
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial guess: equal weights
    w0 = np.ones(n_assets) / n_assets
    
    # Optimize
    result = minimize(neg_sharpe_ratio, w0, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    if not result.success:
        print(f"Warning: Optimization failed - {result.message}")
    
    # Compute portfolio statistics
    weights = result.x
    port_return = np.dot(weights, mean_ret)
    port_volatility = np.sqrt(np.dot(weights, np.dot(cov, weights)))
    port_sharpe = (port_return - risk_free_rate) / port_volatility
    
    return {
        'weights': weights,
        'return': port_return,
        'volatility': port_volatility,
        'sharpe': port_sharpe
    }


def compute_cal_line(risk_free_rate, tangency_portfolio, max_std=None):
    """
    Compute points along the Capital Allocation Line (CAL).
    
    The CAL shows all possible risk-return combinations when combining
    the risk-free asset with the tangency portfolio.
    
    Parameters:
    -----------
    risk_free_rate : float
        Risk-free rate (annualized)
    tangency_portfolio : dict
        Dictionary from compute_tangency_portfolio() containing
        'return' and 'volatility' keys
    max_std : float, optional
        Maximum standard deviation for the line. If None, extends to
        150% of tangency portfolio volatility
        
    Returns:
    --------
    tuple : (cal_std, cal_return) arrays for plotting
        - cal_std: np.ndarray of standard deviations along CAL
        - cal_return: np.ndarray of returns along CAL
        
    Examples:
    ---------
    >>> tangency = {'return': 0.15, 'volatility': 0.20, 'sharpe': 0.45}
    >>> cal_std, cal_return = compute_cal_line(0.06, tangency)
    """
    tang_return = tangency_portfolio['return']
    tang_vol = tangency_portfolio['volatility']
    
    if max_std is None:
        max_std = tang_vol * 1.5
    
    # CAL is a straight line from (0, rf) through (tang_vol, tang_return)
    # Slope is the Sharpe ratio
    cal_std = np.linspace(0, max_std, 100)
    cal_return = risk_free_rate + tangency_portfolio['sharpe'] * cal_std
    
    return cal_std, cal_return
