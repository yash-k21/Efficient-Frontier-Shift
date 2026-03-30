"""
Download and prepare risk-free rate data for efficient frontier analysis.

This script downloads risk-free rate data and prepares it for use with
the efficient frontier pipeline. It handles:
- Downloading Indian 91-day T-bill rates or US Treasury rates as proxy
- Formatting data with proper date indexing
- Saving to Data/risk_free_rates.csv

For Indian stocks (NIFTY), we'll use US Treasury 3-month rates as a proxy
since they are easily accessible and widely used in academic research.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


def download_treasury_rates(start_date='2015-01-01', end_date=None):
    """
    Download US 3-month Treasury rates as proxy for risk-free rate.
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format. If None, uses today.
        
    Returns:
    --------
    pd.DataFrame : DataFrame with Date index and annualized rate column
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Downloading Treasury rates from {start_date} to {end_date}...")
    
    try:
        # Download 3-month Treasury rate (^IRX gives rates in percentage)
        treasury = yf.download('^IRX', start=start_date, end=end_date, progress=False)
        
        if treasury.empty:
            print("Warning: No Treasury data downloaded. Using fallback method.")
            return create_fallback_rates(start_date, end_date)
        
        # Use Close price (represents the yield percentage)
        rates = treasury['Close'].copy()
        
        # Convert from percentage to decimal (e.g., 5.0 -> 0.05)
        rates = rates / 100.0
        
        # Create DataFrame with proper column name
        rf_data = pd.DataFrame({
            'Risk_Free_Rate': rates
        })
        
        # Handle missing values with forward fill
        rf_data = rf_data.ffill().bfill()
        
        print(f"Downloaded {len(rf_data)} days of Treasury rate data")
        print(f"Average rate: {rf_data['Risk_Free_Rate'].mean():.4f} ({rf_data['Risk_Free_Rate'].mean()*100:.2f}%)")
        
        return rf_data
        
    except Exception as e:
        print(f"Error downloading Treasury rates: {e}")
        print("Using fallback rates...")
        return create_fallback_rates(start_date, end_date)


def create_fallback_rates(start_date, end_date):
    """
    Create fallback risk-free rates based on historical averages.
    
    Uses approximate rates:
    - 2015-2016: ~0.5%
    - 2017-2019: ~2.0%
    - 2020-2021: ~0.2% (COVID low rates)
    - 2022-2024: ~4.5% (post-COVID rate hikes)
    """
    print("Creating fallback risk-free rates based on historical periods...")
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    rates = []
    
    for date in dates:
        year = date.year
        if year <= 2016:
            rate = 0.005  # 0.5%
        elif year <= 2019:
            rate = 0.020  # 2.0%
        elif year <= 2021:
            rate = 0.002  # 0.2%
        else:
            rate = 0.045  # 4.5%
        
        rates.append(rate)
    
    rf_data = pd.DataFrame({
        'Risk_Free_Rate': rates
    }, index=dates)
    
    print(f"Created {len(rf_data)} days of fallback rate data")
    print(f"Average rate: {rf_data['Risk_Free_Rate'].mean():.4f} ({rf_data['Risk_Free_Rate'].mean()*100:.2f}%)")
    
    return rf_data


def compute_period_averages(rf_data):
    """
    Compute average rates for common analysis periods.
    
    Parameters:
    -----------
    rf_data : pd.DataFrame
        Risk-free rate data with DatetimeIndex
        
    Returns:
    --------
    dict : Dictionary with period averages
    """
    periods = {
        'pre_covid': ('2015', '2019'),
        'post_covid': ('2020', '2024'),
        'full': ('2015', '2024')
    }
    
    averages = {}
    for period_name, (start, end) in periods.items():
        try:
            period_data = rf_data.loc[start:end]
            if not period_data.empty:
                avg = period_data['Risk_Free_Rate'].mean()
                averages[period_name] = avg
                print(f"{period_name:12s}: {avg:.4f} ({avg*100:.2f}%)")
        except Exception as e:
            print(f"Could not compute {period_name} average: {e}")
    
    return averages


def main():
    """
    Main function to download and save risk-free rate data.
    """
    print("=" * 60)
    print("Risk-Free Rate Data Preparation")
    print("=" * 60)
    print()
    
    # Download data from 2015 to present (matching the stock data range)
    rf_data = download_treasury_rates(start_date='2015-01-01')
    
    # Save to CSV
    output_file = 'Data/risk_free_rates.csv'
    rf_data.to_csv(output_file)
    print(f"\nSaved risk-free rates to: {output_file}")
    
    # Compute and display period averages
    print("\nPeriod Averages:")
    print("-" * 40)
    averages = compute_period_averages(rf_data)
    
    # Save summary statistics
    summary_file = 'Data/risk_free_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("Risk-Free Rate Summary Statistics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Data Source: US 3-Month Treasury Bill (^IRX)\n")
        f.write(f"Date Range: {rf_data.index[0]} to {rf_data.index[-1]}\n")
        f.write(f"Total Days: {len(rf_data)}\n\n")
        
        f.write("Period Averages (Annualized):\n")
        f.write("-" * 50 + "\n")
        for period, rate in averages.items():
            f.write(f"{period:15s}: {rate:.6f} ({rate*100:.4f}%)\n")
        
        f.write("\nDescriptive Statistics:\n")
        f.write("-" * 50 + "\n")
        f.write(rf_data.describe().to_string())
    
    print(f"\nSaved summary statistics to: {summary_file}")
    print("\n" + "=" * 60)
    print("Risk-free rate data preparation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
