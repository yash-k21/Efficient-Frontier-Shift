import pandas as pd
import yfinance as yf

TICKERS = [
    'ACC.NS', 'ADANIPORTS.NS', 'AMBUJACEM.NS', 'ASIANPAINT.NS', 'AXISBANK.NS',
    'BAJAJ-AUTO.NS', 'BANKBARODA.NS', 'BHEL.NS', 'BPCL.NS', 'BHARTIARTL.NS',
    'BOSCHLTD.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DRREDDY.NS', 'GAIL.NS',
    'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS',
    'HINDUNILVR.NS', 'ITC.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'INFY.NS',
    'KOTAKBANK.NS', 'LT.NS', 'LUPIN.NS', 'M&M.NS', 'MARUTI.NS',
    'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'PNB.NS', 'RELIANCE.NS',
    'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS', 'TATAPOWER.NS',
    'TATASTEEL.NS', 'TECHM.NS', 'ULTRACEMCO.NS', 'VEDL.NS', 'WIPRO.NS',
    'ZEEL.NS',
] ## TATAMOTORS.NS not found

prices = yf.download(TICKERS, start='2015-01-01', end='2025-01-01')['Close']
weekly_prices = prices.resample('W').last()
monthly_prices = prices.resample('ME').last()

prices.to_csv('Data/prices_daily.csv')
weekly_prices.to_csv('Data/prices_weekly.csv')
monthly_prices.to_csv('Data/prices_monthly.csv')
