import pandas as pd
import os

weekly_prices = pd.read_csv('Data/prices_weekly.csv', index_col=0, parse_dates=True)
returns = weekly_prices.pct_change().dropna()

os.makedirs('Data/weekly/pre-covid', exist_ok=True)
os.makedirs('Data/weekly/post-covid', exist_ok=True)

pre = returns.loc['2015':'2019']
pre.to_csv('Data/weekly/pre-covid/returns_pre_weekly.csv')
pre.mean().to_csv('Data/weekly/pre-covid/mu_pre_weekly.csv', header=['mean_return'])
pre.cov().to_csv('Data/weekly/pre-covid/sigma_pre_weekly.csv')

post = returns.loc['2020':]
post.to_csv('Data/weekly/post-covid/returns_post_weekly.csv')
post.mean().to_csv('Data/weekly/post-covid/mu_post_weekly.csv', header=['mean_return'])
post.cov().to_csv('Data/weekly/post-covid/sigma_post_weekly.csv')
