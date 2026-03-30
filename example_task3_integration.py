"""
Example script demonstrating how to use Task 3 utilities with custom time periods.

This shows how your teammates can integrate the Sharpe ratio and CAL functionality
with their modified time periods (Task 1) and extended frequencies (Task 2).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    get_annualization_factor,
    annualize_returns,
    annualize_covariance_matrix,
    load_risk_free_rate,
    compute_tangency_portfolio,
    compute_sharpe_ratio,
    compute_cal_line
)


def example_custom_period_analysis():
    """
    Example: Analyze a custom time period (e.g., bull run period).
    
    This demonstrates how to use the utilities for any time period
    your teammates define in Task 1.
    """
    print("="*60)
    print("EXAMPLE: Custom Period Analysis")
    print("="*60)
    
    # ========== SCENARIO 1: Bull Run Period ==========
    # Suppose your teammates define a bull run period as 2017-2018
    period_name = "Bull Run (2017-2018)"
    start_date = '2017'
    end_date = '2018'
    frequency = 'daily'
    
    print(f"\nAnalyzing: {period_name}")
    print(f"Frequency: {frequency}")
    
    # Load returns data (assuming your teammates create this file)
    # For demo, we'll use existing pre-COVID data
    mean_ret_raw = pd.read_csv(f'Data/{frequency}/pre-covid/mu_pre_{frequency}.csv', 
                               index_col=0).squeeze().values
    cov_raw = pd.read_csv(f'Data/{frequency}/pre-covid/sigma_pre_{frequency}.csv', 
                          index_col=0).values
    
    # Annualize the data
    mean_ret_ann = annualize_returns(mean_ret_raw, frequency)
    cov_ann = annualize_covariance_matrix(cov_raw, frequency)
    
    # Load risk-free rate for this period
    rf_rate = load_risk_free_rate(start_date, end_date, frequency)
    
    print(f"Risk-Free Rate: {rf_rate*100:.2f}%")
    print(f"Average Portfolio Return: {mean_ret_ann.mean()*100:.2f}%")
    
    # Compute tangency portfolio
    tangency = compute_tangency_portfolio(mean_ret_ann, cov_ann, rf_rate)
    
    print(f"\nTangency Portfolio:")
    print(f"  Expected Return: {tangency['return']*100:.2f}%")
    print(f"  Volatility: {tangency['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio: {tangency['sharpe']:.4f}")
    print(f"  Top 5 Holdings:")
    
    # Show top 5 weights
    weights_df = pd.read_csv(f'Data/{frequency}/pre-covid/mu_pre_{frequency}.csv', 
                            index_col=0)
    weights_with_names = pd.Series(tangency['weights'], index=weights_df.index)
    top_5 = weights_with_names.nlargest(5)
    for ticker, weight in top_5.items():
        print(f"    {ticker}: {weight*100:.2f}%")


def example_new_frequency():
    """
    Example: Add support for a new frequency (e.g., bi-monthly).
    
    This demonstrates how to extend the utilities for new frequencies
    your teammates add in Task 2.
    """
    print("\n" + "="*60)
    print("EXAMPLE: Adding New Frequency Support")
    print("="*60)
    
    # Current supported frequencies
    print("\nCurrently Supported Frequencies:")
    for freq in ['daily', 'weekly', 'monthly', 'bimonthly', 'quarterly']:
        try:
            factor = get_annualization_factor(freq)
            print(f"  {freq:12s}: {factor} periods per year")
        except ValueError:
            print(f"  {freq:12s}: NOT SUPPORTED (add to ANNUALIZATION_FACTORS)")
    
    print("\nTo add a new frequency:")
    print("1. Edit utils.py")
    print("2. Add entry to ANNUALIZATION_FACTORS dictionary")
    print("   Example: 'biweekly': 26,  # 26 two-week periods per year")
    print("3. All other functions will work automatically!")


def example_sharpe_comparison():
    """
    Example: Compare Sharpe ratios across multiple periods.
    
    This demonstrates how to analyze risk-adjusted performance changes.
    """
    print("\n" + "="*60)
    print("EXAMPLE: Sharpe Ratio Comparison Across Periods")
    print("="*60)
    
    # Load the summary results
    try:
        summary = pd.read_csv('Figures/sharpe_ratio_summary.csv')
        print("\nCurrent Results:")
        print(summary.to_string(index=False))
        
        # Identify best performing frequency
        best_freq = summary.loc[summary['sharpe_post'].idxmax(), 'frequency']
        best_sharpe = summary.loc[summary['sharpe_post'].idxmax(), 'sharpe_post']
        
        print(f"\nBest Post-COVID Sharpe Ratio:")
        print(f"  Frequency: {best_freq}")
        print(f"  Sharpe Ratio: {best_sharpe:.4f}")
        
        # Identify largest improvement
        best_improvement = summary.loc[summary['sharpe_pct_change'].idxmax()]
        print(f"\nLargest Improvement:")
        print(f"  Frequency: {best_improvement['frequency']}")
        print(f"  Change: {best_improvement['sharpe_change']:.4f} "
              f"({best_improvement['sharpe_pct_change']:.1f}%)")
        
    except FileNotFoundError:
        print("Run 3_frontier_with_sharpe.py first to generate results.")


def example_plot_single_period_with_cal():
    """
    Example: Create a custom plot with CAL for a single period.
    
    This demonstrates how to create visualizations for custom analysis.
    """
    print("\n" + "="*60)
    print("EXAMPLE: Custom Visualization with CAL")
    print("="*60)
    
    frequency = 'daily'
    
    # Load data
    mean_ret_raw = pd.read_csv(f'Data/{frequency}/pre-covid/mu_pre_{frequency}.csv',
                               index_col=0).squeeze().values
    cov_raw = pd.read_csv(f'Data/{frequency}/pre-covid/sigma_pre_{frequency}.csv',
                          index_col=0).values
    
    # Annualize
    mean_ret = annualize_returns(mean_ret_raw, frequency)
    cov = annualize_covariance_matrix(cov_raw, frequency)
    
    # Get risk-free rate
    rf = load_risk_free_rate('2015', '2019', frequency)
    
    # Compute tangency portfolio
    tangency = compute_tangency_portfolio(mean_ret, cov, rf)
    
    # Generate CAL points
    cal_std, cal_return = compute_cal_line(rf, tangency)
    
    # Create simple plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot tangency portfolio
    ax.scatter([tangency['volatility']], [tangency['return']], 
              s=200, marker='*', color='red', zorder=10,
              label=f"Tangency (SR={tangency['sharpe']:.3f})")
    
    # Plot CAL
    ax.plot(cal_std, cal_return, 'r--', linewidth=2, 
           label='Capital Allocation Line', alpha=0.7)
    
    # Plot risk-free rate
    ax.scatter([0], [rf], s=100, marker='o', color='green', 
              edgecolors='black', linewidths=1, zorder=10,
              label=f'Risk-Free Rate ({rf*100:.2f}%)')
    
    ax.set_xlabel('Volatility (Standard Deviation)')
    ax.set_ylabel('Expected Return')
    ax.set_title('Example: Tangency Portfolio and CAL')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    output_file = 'Figures/example_custom_plot.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nCustom plot saved to: {output_file}")


def main():
    """Run all examples."""
    print("\n" + "🚀 "*20)
    print("TASK 3 INTEGRATION EXAMPLES")
    print("🚀 "*20 + "\n")
    
    print("These examples show how to integrate Task 3 utilities")
    print("with your teammates' modifications (Tasks 1 & 2).\n")
    
    # Run examples
    example_custom_period_analysis()
    example_new_frequency()
    example_sharpe_comparison()
    example_plot_single_period_with_cal()
    
    print("\n" + "="*60)
    print("✅ ALL EXAMPLES COMPLETE!")
    print("="*60)
    print("\nKey Takeaways:")
    print("1. Use load_risk_free_rate() with ANY date range")
    print("2. Use annualization functions for ANY frequency")
    print("3. Use compute_tangency_portfolio() for optimal Sharpe ratio")
    print("4. Use compute_cal_line() to visualize risk-return tradeoffs")
    print("\nSee TASK3_README.md for full documentation.")


if __name__ == '__main__':
    main()
