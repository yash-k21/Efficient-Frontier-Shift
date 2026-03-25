import pandas as pd
import yfinance as yf

TICKERS = [
    'ACC.NS', 'AMBUJACEM.NS', 'ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS',
    'ASIANPAINT.NS', 'AUROPHARMA.NS', 'AXISBANK.NS', 'BHARTIARTL.NS', 'BHEL.NS',
    'BHARATFORG.NS', 'BAJAJ-AUTO.NS', 'BAJAJFINSV.NS', 'BAJAJHLDNG.NS', 'BANKBARODA.NS',
    'BANKINDIA.NS', 'BOSCHLTD.NS', 'BPCL.NS', 'CANBK.NS', 'CONCOR.NS',
    'CIPLA.NS', 'COLPAL.NS', 'COALINDIA.NS', 'DABUR.NS', 'DIVISLAB.NS',
    'DRREDDY.NS', 'FEDERALBNK.NS', 'GAIL.NS', 'GODREJCP.NS', 'GRASIM.NS',
    'HCLTECH.NS', 'HDFCBANK.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS',
    'ICICIBANK.NS', 'INDUSINDBK.NS', 'INFY.NS', 'ITC.NS', 'JSWSTEEL.NS',
    'KOTAKBANK.NS', 'LT.NS', 'MARUTI.NS', 'NTPC.NS', 'ONGC.NS',
    'RELIANCE.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS', 'WIPRO.NS',
]

prices = yf.download(TICKERS, start='2015-01-01', end='2025-01-01')['Close']
monthly_prices = prices.resample('ME').last()

prices.to_csv('prices_daily.csv')
monthly_prices.to_csv('prices_monthly.csv')
