import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

os.makedirs('Figures', exist_ok=True)
N = 1000000

def long_only_frontier(mean_ret, cov, n_points=100):
    n = len(mean_ret)
    w0 = np.ones(n) / n
    bounds = [(0, 1)] * n
    sum_constraint = {'type': 'eq', 'fun': lambda w: w.sum() - 1}

    # find MVP return (lower bound)
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

# Pre-COVID
print("Pre-COVID: loading data...")
mean_ret_pre = pd.read_csv('Data/monthly/pre-covid/mu_pre_monthly.csv', index_col=0).squeeze().values
cov_pre = pd.read_csv('Data/monthly/pre-covid/sigma_pre_monthly.csv', index_col=0).values
n = len(mean_ret_pre)

print(f"Pre-COVID: simulating {N} portfolios...")
means_pre, stds_pre = np.zeros(N), np.zeros(N)
for i in range(N):
    w = np.random.dirichlet(np.ones(n) * 0.5)
    means_pre[i] = mean_ret_pre @ w
    stds_pre[i] = np.sqrt(w @ cov_pre @ w)
    if (i + 1) % (N // 10) == 0:
        print(f"  Pre-COVID: {(i + 1) * 100 // N}% done")

print("Pre-COVID: computing long-only frontier...")
frontier_std_pre, frontier_mean_pre = long_only_frontier(mean_ret_pre, cov_pre)

# Post-COVID
print("Post-COVID: loading data...")
mean_ret_post = pd.read_csv('Data/monthly/post-covid/mu_post_monthly.csv', index_col=0).squeeze().values
cov_post = pd.read_csv('Data/monthly/post-covid/sigma_post_monthly.csv', index_col=0).values

print(f"Post-COVID: simulating {N} portfolios...")
means_post, stds_post = np.zeros(N), np.zeros(N)
for i in range(N):
    w = np.random.dirichlet(np.ones(n) * 0.5)
    means_post[i] = mean_ret_post @ w
    stds_post[i] = np.sqrt(w @ cov_post @ w)
    if (i + 1) % (N // 10) == 0:
        print(f"  Post-COVID: {(i + 1) * 100 // N}% done")

print("Post-COVID: computing long-only frontier...")
frontier_std_post, frontier_mean_post = long_only_frontier(mean_ret_post, cov_post)

# Combined plot
fig, ax = plt.subplots()
ax.scatter(stds_pre, means_pre, s=5, color='steelblue', linewidths=0, alpha=0.3, label='Feasible Set Pre-COVID')
ax.scatter(stds_post, means_post, s=5, color='darkorange', linewidths=0, alpha=0.3, label='Feasible Set Post-COVID')
ax.plot(frontier_std_pre, frontier_mean_pre, color='steelblue', linewidth=2, label='Efficient Frontier Pre-COVID')
ax.plot(frontier_std_post, frontier_mean_post, color='darkorange', linewidth=2, label='Efficient Frontier Post-COVID')
ax.set_xlabel('Standard Deviation')
ax.set_ylabel('Expected Return')
ax.set_title('Efficient Frontier Shift - Pre vs Post COVID')
ax.legend()
fig.savefig('Figures/frontier_combined.png', dpi=150, bbox_inches='tight')
plt.close()
print("Combined plot saved.")
