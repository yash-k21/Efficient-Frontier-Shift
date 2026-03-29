import pandas as pd
import os

monthly_prices = pd.read_csv('Data/prices_monthly.csv', index_col=0, parse_dates=True)
returns = monthly_prices.pct_change().dropna()

os.makedirs('Data/monthly/pre-covid', exist_ok=True)
os.makedirs('Data/monthly/post-covid', exist_ok=True)

pre = returns.loc['2015':'2019']
pre.to_csv('Data/monthly/pre-covid/returns_pre_monthly.csv')
pre.mean().to_csv('Data/monthly/pre-covid/mu_pre_monthly.csv', header=['mean_return'])
pre.cov().to_csv('Data/monthly/pre-covid/sigma_pre_monthly.csv')

post = returns.loc['2020':]
post.to_csv('Data/monthly/post-covid/returns_post_monthly.csv')
post.mean().to_csv('Data/monthly/post-covid/mu_post_monthly.csv', header=['mean_return'])
post.cov().to_csv('Data/monthly/post-covid/sigma_post_monthly.csv')
