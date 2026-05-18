import pandas as pd
import os

daily_prices = pd.read_csv('../data/processed/prices_daily.csv', index_col=0, parse_dates=True)
returns = daily_prices.pct_change().dropna()

os.makedirs('../data/processed/daily/pre-covid', exist_ok=True)
os.makedirs('../data/processed/daily/post-covid', exist_ok=True)

pre = returns.loc['2015-01-01':'2020-01-31']
pre.to_csv('../data/processed/daily/pre-covid/returns_pre_daily.csv')
pre.mean().to_csv('../data/processed/daily/pre-covid/mu_pre_daily.csv', header=['mean_return'])
pre.cov().to_csv('../data/processed/daily/pre-covid/sigma_pre_daily.csv')

post = returns.loc['2021-07-01':'2024-12-31']
post.to_csv('../data/processed/daily/post-covid/returns_post_daily.csv')
post.mean().to_csv('../data/processed/daily/post-covid/mu_post_daily.csv', header=['mean_return'])
post.cov().to_csv('../data/processed/daily/post-covid/sigma_post_daily.csv')
