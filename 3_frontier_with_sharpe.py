"""
Enhanced efficient frontier analysis with Sharpe ratios and Capital Allocation Line.

This is an enhanced version of 3_frontier.py that adds:
- Risk-free rate integration
- Sharpe ratio computation
- Tangency portfolio identification
- Capital Allocation Line (CAL) plotting
- Sharpe ratio change analysis between periods

Usage:
    python 3_frontier_with_sharpe.py
    
The script will generate enhanced frontier plots with CAL for each frequency.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

# Import utility functions
from utils import (
    get_annualization_factor,
    annualize_returns,
    annualize_covariance_matrix,
    load_risk_free_rate,
    compute_tangency_portfolio,
    compute_cal_line
)

os.makedirs('Figures', exist_ok=True)
N = 1000000  # Number of portfolio simulations


def long_only_frontier(mean_ret, cov, n_points=100):
    """
    Compute the long-only efficient frontier.
    
    Parameters:
    -----------
    mean_ret : np.ndarray
        Expected returns for each asset
    cov : np.ndarray
        Covariance matrix
    n_points : int
        Number of points to compute along the frontier
        
    Returns:
    --------
    tuple : (frontier_std, frontier_mean) lists
    """
    n = len(mean_ret)
    w0 = np.ones(n) / n
    bounds = [(0, 1)] * n
    sum_constraint = {'type': 'eq', 'fun': lambda w: w.sum() - 1}

    # Find minimum variance portfolio
    mvp = minimize(lambda w: w @ cov @ w, w0, bounds=bounds, constraints=sum_constraint)
    ret_min = mean_ret @ mvp.x
    ret_max = mean_ret.max()

    target_returns = np.linspace(ret_min, ret_max, n_points)
    frontier_std, frontier_mean = [], []

    for target in target_returns:
        constraints = [sum_constraint, {'type': 'eq', 'fun': lambda w, t=target: mean_ret @ w - t}]
        res = minimize(lambda w: w @ cov @ w, w0, bounds=bounds, constraints=constraints)
        if res.success:
            frontier_std.append(np.sqrt(res.fun))
            frontier_mean.append(target)

    return frontier_std, frontier_mean


def simulate_portfolios(mean_ret, cov, n_portfolios=1000000, verbose=True):
    """
    Simulate random portfolios to create feasible set.
    
    Parameters:
    -----------
    mean_ret : np.ndarray
        Expected returns for each asset
    cov : np.ndarray
        Covariance matrix
    n_portfolios : int
        Number of portfolios to simulate
    verbose : bool
        Print progress updates
        
    Returns:
    --------
    tuple : (means, stds) arrays of portfolio returns and volatilities
    """
    n_assets = len(mean_ret)
    means = np.zeros(n_portfolios)
    stds = np.zeros(n_portfolios)
    
    for i in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n_assets) * 0.5)
        means[i] = mean_ret @ w
        stds[i] = np.sqrt(w @ cov @ w)
        
        if verbose and (i + 1) % (n_portfolios // 10) == 0:
            print(f"  {(i + 1) * 100 // n_portfolios}% done")
    
    return means, stds


def plot_frontier_with_cal(means_pre, stds_pre, frontier_pre, tangency_pre, rf_pre,
                           means_post, stds_post, frontier_post, tangency_post, rf_post,
                           frequency, output_file):
    """
    Create enhanced frontier plot with CAL and Sharpe ratio annotations.
    
    Parameters:
    -----------
    means_pre, stds_pre : arrays
        Pre-period feasible set points
    frontier_pre : tuple
        (frontier_std, frontier_mean) for pre-period
    tangency_pre : dict
        Tangency portfolio info for pre-period
    rf_pre : float
        Risk-free rate for pre-period
    means_post, stds_post : arrays
        Post-period feasible set points
    frontier_post : tuple
        (frontier_std, frontier_mean) for post-period
    tangency_post : dict
        Tangency portfolio info for post-period
    rf_post : float
        Risk-free rate for post-period
    frequency : str
        Data frequency ('daily', 'weekly', 'monthly')
    output_file : str
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot feasible sets (scatter)
    ax.scatter(stds_pre, means_pre, s=3, color='steelblue', linewidths=0, 
              alpha=0.2, label='Feasible Set Pre-COVID')
    ax.scatter(stds_post, means_post, s=3, color='darkorange', linewidths=0, 
              alpha=0.2, label='Feasible Set Post-COVID')
    
    # Plot efficient frontiers
    frontier_std_pre, frontier_mean_pre = frontier_pre
    frontier_std_post, frontier_mean_post = frontier_post
    
    ax.plot(frontier_std_pre, frontier_mean_pre, color='steelblue', 
           linewidth=2.5, label='Efficient Frontier Pre-COVID', zorder=5)
    ax.plot(frontier_std_post, frontier_mean_post, color='darkorange', 
           linewidth=2.5, label='Efficient Frontier Post-COVID', zorder=5)
    
    # Plot tangency portfolios
    ax.scatter([tangency_pre['volatility']], [tangency_pre['return']], 
              s=200, marker='*', color='darkblue', edgecolors='white', 
              linewidths=1.5, label=f"Tangency Pre (SR={tangency_pre['sharpe']:.3f})", 
              zorder=10)
    ax.scatter([tangency_post['volatility']], [tangency_post['return']], 
              s=200, marker='*', color='darkred', edgecolors='white', 
              linewidths=1.5, label=f"Tangency Post (SR={tangency_post['sharpe']:.3f})", 
              zorder=10)
    
    # Plot Capital Allocation Lines (CAL)
    cal_std_pre, cal_mean_pre = compute_cal_line(rf_pre, tangency_pre)
    cal_std_post, cal_mean_post = compute_cal_line(rf_post, tangency_post)
    
    ax.plot(cal_std_pre, cal_mean_pre, color='steelblue', linestyle='--', 
           linewidth=2, alpha=0.7, label='CAL Pre-COVID', zorder=7)
    ax.plot(cal_std_post, cal_mean_post, color='darkorange', linestyle='--', 
           linewidth=2, alpha=0.7, label='CAL Post-COVID', zorder=7)
    
    # Plot risk-free rate points
    ax.scatter([0], [rf_pre], s=100, marker='o', color='steelblue', 
              edgecolors='black', linewidths=1, zorder=10)
    ax.scatter([0], [rf_post], s=100, marker='o', color='darkorange', 
              edgecolors='black', linewidths=1, zorder=10)
    
    # Add text annotations for Sharpe ratios and risk-free rates
    sharpe_change = tangency_post['sharpe'] - tangency_pre['sharpe']
    sharpe_pct_change = (sharpe_change / tangency_pre['sharpe']) * 100
    
    textstr = f"Risk-Free Rates:\n  Pre: {rf_pre*100:.2f}%\n  Post: {rf_post*100:.2f}%\n\n"
    textstr += f"Sharpe Ratio Change:\n  Δ = {sharpe_change:+.3f} ({sharpe_pct_change:+.1f}%)"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)
    
    # Labels and title
    ax.set_xlabel('Standard Deviation (Annualized)', fontsize=12)
    ax.set_ylabel('Expected Return (Annualized)', fontsize=12)
    ax.set_title(f'Efficient Frontier with CAL - Pre vs Post COVID ({frequency.capitalize()})', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Save figure
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Enhanced plot saved to: {output_file}")


def main():
    """
    Main function to run enhanced efficient frontier analysis.
    """
    print("=" * 70)
    print("ENHANCED EFFICIENT FRONTIER ANALYSIS WITH SHARPE RATIOS AND CAL")
    print("=" * 70)
    
    # Store results for summary
    results = []
    
    for freq in ('daily', 'weekly', 'monthly'):
        print(f"\n{'='*70}")
        print(f"Processing {freq.upper()} data")
        print('='*70)
        
        # Get annualization factor
        ann_factor = get_annualization_factor(freq)
        
        # ===== PRE-COVID PERIOD =====
        print(f"\n[PRE-COVID] Loading {freq} data...")
        mean_ret_pre_raw = pd.read_csv(f'Data/{freq}/pre-covid/mu_pre_{freq}.csv', 
                                        index_col=0).squeeze().values
        cov_pre_raw = pd.read_csv(f'Data/{freq}/pre-covid/sigma_pre_{freq}.csv', 
                                   index_col=0).values
        
        # Annualize returns and covariance
        mean_ret_pre = annualize_returns(mean_ret_pre_raw, freq)
        cov_pre = annualize_covariance_matrix(cov_pre_raw, freq)
        
        # Load risk-free rate
        rf_pre = load_risk_free_rate('2015', '2019', freq)
        print(f"Risk-free rate (Pre): {rf_pre*100:.2f}%")
        
        # Simulate portfolios
        print(f"[PRE-COVID] Simulating {N:,} portfolios...")
        means_pre, stds_pre = simulate_portfolios(mean_ret_pre, cov_pre, N)
        
        # Compute efficient frontier
        print(f"[PRE-COVID] Computing efficient frontier...")
        frontier_std_pre, frontier_mean_pre = long_only_frontier(mean_ret_pre, cov_pre)
        
        # Compute tangency portfolio
        print(f"[PRE-COVID] Computing tangency portfolio...")
        tangency_pre = compute_tangency_portfolio(mean_ret_pre, cov_pre, rf_pre)
        print(f"  Tangency Return: {tangency_pre['return']*100:.2f}%")
        print(f"  Tangency Volatility: {tangency_pre['volatility']*100:.2f}%")
        print(f"  Sharpe Ratio: {tangency_pre['sharpe']:.4f}")
        
        # ===== POST-COVID PERIOD =====
        print(f"\n[POST-COVID] Loading {freq} data...")
        mean_ret_post_raw = pd.read_csv(f'Data/{freq}/post-covid/mu_post_{freq}.csv', 
                                         index_col=0).squeeze().values
        cov_post_raw = pd.read_csv(f'Data/{freq}/post-covid/sigma_post_{freq}.csv', 
                                    index_col=0).values
        
        # Annualize returns and covariance
        mean_ret_post = annualize_returns(mean_ret_post_raw, freq)
        cov_post = annualize_covariance_matrix(cov_post_raw, freq)
        
        # Load risk-free rate
        rf_post = load_risk_free_rate('2020', '2024', freq)
        print(f"Risk-free rate (Post): {rf_post*100:.2f}%")
        
        # Simulate portfolios
        print(f"[POST-COVID] Simulating {N:,} portfolios...")
        means_post, stds_post = simulate_portfolios(mean_ret_post, cov_post, N)
        
        # Compute efficient frontier
        print(f"[POST-COVID] Computing efficient frontier...")
        frontier_std_post, frontier_mean_post = long_only_frontier(mean_ret_post, cov_post)
        
        # Compute tangency portfolio
        print(f"[POST-COVID] Computing tangency portfolio...")
        tangency_post = compute_tangency_portfolio(mean_ret_post, cov_post, rf_post)
        print(f"  Tangency Return: {tangency_post['return']*100:.2f}%")
        print(f"  Tangency Volatility: {tangency_post['volatility']*100:.2f}%")
        print(f"  Sharpe Ratio: {tangency_post['sharpe']:.4f}")
        
        # ===== ANALYSIS =====
        sharpe_change = tangency_post['sharpe'] - tangency_pre['sharpe']
        sharpe_pct_change = (sharpe_change / tangency_pre['sharpe']) * 100
        
        print(f"\n[ANALYSIS]")
        print(f"  Sharpe Ratio Change: {sharpe_change:+.4f} ({sharpe_pct_change:+.1f}%)")
        
        results.append({
            'frequency': freq,
            'rf_pre': rf_pre,
            'rf_post': rf_post,
            'sharpe_pre': tangency_pre['sharpe'],
            'sharpe_post': tangency_post['sharpe'],
            'sharpe_change': sharpe_change,
            'sharpe_pct_change': sharpe_pct_change
        })
        
        # ===== PLOTTING =====
        print(f"\n[PLOTTING] Creating enhanced frontier plot...")
        output_file = f'Figures/frontier_with_sharpe_{freq}.png'
        plot_frontier_with_cal(
            means_pre, stds_pre, (frontier_std_pre, frontier_mean_pre), 
            tangency_pre, rf_pre,
            means_post, stds_post, (frontier_std_post, frontier_mean_post), 
            tangency_post, rf_post,
            freq, output_file
        )
    
    # ===== SUMMARY =====
    print(f"\n{'='*70}")
    print("SUMMARY OF RESULTS")
    print('='*70)
    
    results_df = pd.DataFrame(results)
    print("\nSharpe Ratio Analysis:")
    print(results_df.to_string(index=False))
    
    # Save summary to CSV
    results_df.to_csv('Figures/sharpe_ratio_summary.csv', index=False)
    print("\nSummary saved to: Figures/sharpe_ratio_summary.csv")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - Figures/frontier_with_sharpe_daily.png")
    print("  - Figures/frontier_with_sharpe_weekly.png")
    print("  - Figures/frontier_with_sharpe_monthly.png")
    print("  - Figures/sharpe_ratio_summary.csv")


if __name__ == '__main__':
    main()
