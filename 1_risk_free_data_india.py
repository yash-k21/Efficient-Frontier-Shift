"""
Process and prepare Indian 91-Day T-Bill data for efficient frontier analysis.

This script processes RBI's 91-Day Government of India Treasury Bill auction data
and prepares it for use with the efficient frontier pipeline. It handles:
- Reading RBI auction data from Excel
- Converting weekly auction data to daily rates (forward fill)
- Formatting with proper date indexing
- Saving to Data/risk_free_rates.csv

The Indian 91-Day T-Bill is the appropriate risk-free rate for Indian stocks (NIFTY),
representing the yield on short-term Indian government debt with virtually zero risk.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def process_indian_tbill_data(excel_file):
    """
    Process Indian 91-Day T-Bill auction data from RBI Excel file.
    
    Parameters:
    -----------
    excel_file : str
        Path to the RBI Excel file
        
    Returns:
    --------
    pd.DataFrame : DataFrame with Date index and daily risk-free rates
    """
    print(f"Reading Indian 91-Day T-Bill data from: {excel_file}")
    
    try:
        # Read Excel file with proper header row (skip first 5 rows)
        df = pd.read_excel(excel_file, sheet_name=0, skiprows=5)
        
        # Clean and parse dates
        df_clean = df[df['Date of Auction'].notna()].copy()
        df_clean['Date'] = pd.to_datetime(df_clean['Date of Auction'], errors='coerce')
        df_clean = df_clean[df_clean['Date'].notna()].copy()
        
        # Extract weighted average yield (already in percentage)
        yield_col = 'Weighted Avg Yield (per cent)'
        df_clean['Yield'] = pd.to_numeric(df_clean[yield_col], errors='coerce')
        df_clean = df_clean[df_clean['Yield'].notna()].copy()
        
        # Keep only Date and Yield, sort by date
        df_auction = df_clean[['Date', 'Yield']].copy()
        df_auction = df_auction.sort_values('Date').reset_index(drop=True)
        
        print(f"Loaded {len(df_auction)} auction records")
        print(f"Date range: {df_auction['Date'].min().date()} to {df_auction['Date'].max().date()}")
        print(f"Average yield: {df_auction['Yield'].mean():.4f}%")
        
        return df_auction
        
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        raise


def convert_to_daily_rates(df_auction, start_date='2015-01-01', end_date=None):
    """
    Convert weekly auction data to daily risk-free rates.
    
    Uses forward fill to propagate auction rates to all days until next auction.
    
    Parameters:
    -----------
    df_auction : pd.DataFrame
        Auction data with Date and Yield columns
    start_date : str
        Start date for daily series
    end_date : str, optional
        End date for daily series. If None, uses last auction date.
        
    Returns:
    --------
    pd.DataFrame : Daily risk-free rates with Date index
    """
    if end_date is None:
        end_date = df_auction['Date'].max()
    
    print(f"\nConverting to daily rates from {start_date} to {end_date}...")
    
    # Filter auction data to desired range
    df_period = df_auction[
        (df_auction['Date'] >= start_date) & 
        (df_auction['Date'] <= end_date)
    ].copy()
    
    # Set Date as index
    df_period = df_period.set_index('Date')
    
    # Create daily date range
    daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Reindex to daily and forward fill (use last auction rate until next auction)
    df_daily = df_period.reindex(daily_dates, method='ffill')
    
    # Back fill any initial NaN values (if data starts after start_date)
    df_daily = df_daily.bfill()
    
    # Convert yield from percentage to decimal (e.g., 6.5 -> 0.065)
    df_daily['Risk_Free_Rate'] = df_daily['Yield'] / 100.0
    
    # Keep only the Risk_Free_Rate column
    rf_data = df_daily[['Risk_Free_Rate']].copy()
    
    print(f"Created {len(rf_data)} days of daily risk-free rate data")
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
    Main function to process and save Indian 91-Day T-Bill data.
    """
    print("=" * 60)
    print("Indian 91-Day T-Bill Risk-Free Rate Data Preparation")
    print("=" * 60)
    print()
    
    # Path to the downloaded RBI Excel file
    excel_file = '/Users/viraatarora/Downloads/Auctions of 91-Day Government of India Treasury Bills.xlsx'
    
    # Process auction data
    df_auction = process_indian_tbill_data(excel_file)
    
    # Convert to daily rates (matching stock data range)
    rf_data = convert_to_daily_rates(df_auction, start_date='2015-01-01')
    
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
        f.write(f"Data Source: Reserve Bank of India (RBI)\n")
        f.write(f"Security: 91-Day Government of India Treasury Bills\n")
        f.write(f"Note: Auction data converted to daily rates using forward fill\n")
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
    
    # Display sample data
    print("\n" + "=" * 60)
    print("Sample Data Verification:")
    print("-" * 60)
    print("\nEarly 2015:")
    print(rf_data.loc['2015-01':'2015-01'].head(10))
    print("\nLate 2019 (Pre-COVID):")
    print(rf_data.loc['2019-12'].tail(10))
    print("\nMarch 2020 (COVID):")
    print(rf_data.loc['2020-03'].head(10))
    print("\nRecent 2025:")
    print(rf_data.loc['2025-03'].tail(10) if '2025' in rf_data.index.year.astype(str) else rf_data.tail(10))
    
    print("\n" + "=" * 60)
    print("Indian 91-Day T-Bill data preparation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
