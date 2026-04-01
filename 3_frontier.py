import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import os

np.random.seed(42)
os.makedirs('Figures', exist_ok=True)

# ── Colors ──────────────────────────────────────────────────────────
PRE_C  = '#1D9E75'   # teal
POST_C = '#534AB7'   # purple

# ── Period definitions ───────────────────────────────────────────────
PRE_START,  PRE_END  = '2015-01-01', '2020-01-31'
POST_START, POST_END = '2021-07-01', '2024-12-31'
CRISIS_START, CRISIS_END = '2020-02-01', '2021-06-30'

ANNUALISE = {'daily': 252, 'weekly': 52}


# ════════════════════════════════════════════════════════════════════
# Risk-free rate
# ════════════════════════════════════════════════════════════════════

def load_risk_free_rates():
    xl = pd.read_excel(
        'Auctions of 91-Day Government of India Treasury Bills.xlsx',
        header=5,
        usecols=[1, 16]
    )
    xl.columns = ['Date of Auction', 'Weighted Avg Yield (per cent)']
    xl['Date of Auction'] = pd.to_datetime(xl['Date of Auction'], errors='coerce')
    xl = xl.dropna(subset=['Date of Auction', 'Weighted Avg Yield (per cent)'])
    xl = xl.set_index('Date of Auction').sort_index()
    yields = pd.to_numeric(xl['Weighted Avg Yield (per cent)'], errors='coerce').dropna() / 100.0

    rf_pre  = yields.loc[PRE_START:PRE_END].mean()
    rf_post = yields.loc[POST_START:POST_END].mean()
    print(f"  RF pre : {rf_pre*100:.2f}%   RF post : {rf_post*100:.2f}%")
    return rf_pre, rf_post


# ════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════

def load_prices():
    prices = pd.read_csv('Data/prices_daily.csv', index_col=0, parse_dates=True)
    prices = prices[~prices.index.astype(str).str.contains('Price|Ticker', na=False)]
    return prices.astype(float).dropna(axis=1, how='all').sort_index()


def to_log_returns(prices, freq):
    p = prices if freq == 'daily' else prices.resample('W').last()
    log_ret = np.log(p / p.shift(1)).iloc[1:]
    print(f"  [{freq}] {log_ret.shape[1]} tickers  {log_ret.shape[0]} obs")
    return log_ret


def slice_period(log_ret, start, end, max_nan=0.10):
    ret  = log_ret.loc[start:end]
    keep = ret.columns[ret.isna().mean() <= max_nan]
    return ret[keep].ffill(limit=1).dropna(axis=1)


# ════════════════════════════════════════════════════════════════════
# Covariance & optimisation
# ════════════════════════════════════════════════════════════════════

def ledoit_wolf(returns_df, freq):
    scale = ANNUALISE[freq]
    lw    = LedoitWolf().fit(returns_df.values)
    mu    = returns_df.mean().values * scale
    cov   = lw.covariance_ * scale
    n, t  = returns_df.shape[1], returns_df.shape[0]
    print(f"    N={n}  T={t}  T/N={t/n:.2f}  shrinkage={lw.shrinkage_:.3f}")
    return mu, cov


def min_var(mu, cov):
    n   = len(mu)
    res = minimize(
        lambda w: w @ cov @ w,
        np.ones(n) / n,
        method='SLSQP',
        bounds=[(0, 1)] * n,
        constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}],
        options={'ftol': 1e-12, 'maxiter': 2000}
    )
    if not res.success:
        raise RuntimeError(f'MVP failed: {res.message}')
    return res.x


def max_sharpe(mu, cov, rf):
    n = len(mu)
    def neg_sr(w):
        r = w @ mu
        v = np.sqrt(w @ cov @ w)
        return -(r - rf) / (v + 1e-12)
    res = minimize(
        neg_sr,
        np.ones(n) / n,
        method='SLSQP',
        bounds=[(0, 1)] * n,
        constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}],
        options={'ftol': 1e-12, 'maxiter': 2000}
    )
    if not res.success:
        raise RuntimeError(f'Max-Sharpe failed: {res.message}')
    return res.x


def long_only_frontier(mu, cov, n_points=250):
    n    = len(mu)
    w_mv = min_var(mu, cov)
    r_lo = w_mv @ mu
    r_hi = mu.max()
    vols, rets = [], []
    for target in np.linspace(r_lo, r_hi, n_points):
        res = minimize(
            lambda w: w @ cov @ w,
            np.ones(n) / n,
            method='SLSQP',
            bounds=[(0, 1)] * n,
            constraints=[
                {'type': 'eq', 'fun': lambda w: w.sum() - 1},
                {'type': 'eq', 'fun': lambda w, t=target: w @ mu - t}
            ],
            options={'ftol': 1e-12, 'maxiter': 2000}
        )
        if res.success:
            vols.append(np.sqrt(res.x @ cov @ res.x))
            rets.append(res.x @ mu)
    return np.array(vols), np.array(rets)


# ════════════════════════════════════════════════════════════════════
# Frontier plot
# ════════════════════════════════════════════════════════════════════

def plot_frontier(mu_pre, cov_pre, mu_post, cov_post, rf_pre, rf_post, freq, out_path):
    print(f"  Computing frontiers...")
    v_pre,  r_pre  = long_only_frontier(mu_pre,  cov_pre)
    v_post, r_post = long_only_frontier(mu_post, cov_post)

    w_mv_pre  = min_var(mu_pre,  cov_pre)
    w_mv_post = min_var(mu_post, cov_post)
    w_ms_pre  = max_sharpe(mu_pre,  cov_pre,  rf_pre)
    w_ms_post = max_sharpe(mu_post, cov_post, rf_post)

    rv_mv_pre_v  = np.sqrt(w_mv_pre  @ cov_pre  @ w_mv_pre)
    rv_mv_post_v = np.sqrt(w_mv_post @ cov_post @ w_mv_post)
    rv_ms_pre_v  = np.sqrt(w_ms_pre  @ cov_pre  @ w_ms_pre)
    rv_ms_post_v = np.sqrt(w_ms_post @ cov_post @ w_ms_post)
    rv_mv_pre    = mu_pre  @ w_mv_pre
    rv_mv_post   = mu_post @ w_mv_post
    rv_ms_pre    = mu_pre  @ w_ms_pre
    rv_ms_post   = mu_post @ w_ms_post

    sr_pre  = (rv_ms_pre  - rf_pre)  / rv_ms_pre_v
    sr_post = (rv_ms_post - rf_post) / rv_ms_post_v

    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(np.sqrt(np.diag(cov_pre)),  mu_pre,
               color=PRE_C,  alpha=0.18, s=18, zorder=2, linewidths=0)
    ax.scatter(np.sqrt(np.diag(cov_post)), mu_post,
               color=POST_C, alpha=0.18, s=18, zorder=2, linewidths=0)

    max_v = max(v_pre.max(), v_post.max()) * 1.05
    v_cal = np.linspace(0, max_v, 300)
    ax.plot(v_cal, rf_pre  + sr_pre  * v_cal, color=PRE_C,  lw=1.0, ls='--', alpha=0.5, zorder=3)
    ax.plot(v_cal, rf_post + sr_post * v_cal, color=POST_C, lw=1.0, ls='--', alpha=0.5, zorder=3)

    ax.plot(v_pre,  r_pre,  color=PRE_C,  lw=2.4, zorder=4)
    ax.plot(v_post, r_post, color=POST_C, lw=2.4, zorder=4)

    ax.scatter([rv_mv_pre_v,  rv_mv_post_v], [rv_mv_pre,  rv_mv_post],
               color='white', edgecolors=[PRE_C, POST_C], s=60, linewidths=1.8, zorder=6)
    ax.scatter([rv_ms_pre_v,  rv_ms_post_v], [rv_ms_pre,  rv_ms_post],
               color=[PRE_C, POST_C], marker='*', s=200, zorder=7, linewidths=0)
    ax.scatter([0, 0], [rf_pre, rf_post],
               color=[PRE_C, POST_C], s=40, zorder=6, linewidths=0)

    all_v = np.concatenate([v_pre, v_post])
    all_r = np.concatenate([r_pre, r_post])
    pad_x = (all_v.max() - all_v.min()) * 0.25
    pad_y = (all_r.max() - all_r.min()) * 0.25
    ax.set_xlim(0, all_v.max() + pad_x)
    ax.set_ylim(min(all_r.min() - pad_y, rf_pre - pad_y * 0.5), all_r.max() + pad_y)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.set_xlabel('Volatility  (σ, annualised)', labelpad=8)
    ax.set_ylabel('Expected Return  (μ, annualised)', labelpad=8)
    ax.tick_params(length=3, color='#AAAAAA')
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_color('#CCCCCC')

    legend_elements = [
        Line2D([0], [0], color=PRE_C,  lw=2.4, label='Pre-COVID  (Jan 2015 – Jan 2020)'),
        Line2D([0], [0], color=POST_C, lw=2.4, label='Post-COVID  (Jul 2021 – Dec 2024)'),
        Line2D([0], [0], color='#999999', lw=1.0, ls='--', label='Capital Allocation Line'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='w',
               markeredgecolor='#555555', markersize=7, lw=0, label='Global Min-Variance'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#555555',
               markersize=11, lw=0, label='Tangency Portfolio'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, frameon=False, loc='upper left', handlelength=1.6)

    ax.annotate(f'SR = {sr_pre:.2f}',  xy=(rv_ms_pre_v,  rv_ms_pre),
                xytext=(rv_ms_pre_v  + 0.01, rv_ms_pre  - 0.02),
                fontsize=8.5, color=PRE_C,  arrowprops=dict(arrowstyle='-', color=PRE_C,  lw=0.7))
    ax.annotate(f'SR = {sr_post:.2f}', xy=(rv_ms_post_v, rv_ms_post),
                xytext=(rv_ms_post_v + 0.01, rv_ms_post - 0.02),
                fontsize=8.5, color=POST_C, arrowprops=dict(arrowstyle='-', color=POST_C, lw=0.7))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ════════════════════════════════════════════════════════════════════
# Tangency portfolio weight analysis
# ════════════════════════════════════════════════════════════════════

def plot_tangency_weights(mu_pre, cov_pre, mu_post, cov_post,
                          rf_pre, rf_post, tickers, freq, out_path, top_n=20):
    w_pre  = max_sharpe(mu_pre,  cov_pre,  rf_pre)
    w_post = max_sharpe(mu_post, cov_post, rf_post)

    labels = [t.replace('.NS', '') for t in tickers]
    df = pd.DataFrame({'Pre': w_pre, 'Post': w_post}, index=labels)

    top = df.max(axis=1).nlargest(top_n).index
    df  = df.loc[top].sort_values('Pre', ascending=True)

    print(f"\n  Tangency weights [{freq}] — top {top_n} assets:")
    print(f"  {'Ticker':<12}  {'Pre':>7}  {'Post':>7}  {'Δ':>7}")
    for tk in df.sort_values('Pre', ascending=False).index:
        delta = df.loc[tk, 'Post'] - df.loc[tk, 'Pre']
        print(f"  {tk:<12}  {df.loc[tk,'Pre']:>6.1%}  {df.loc[tk,'Post']:>6.1%}  {delta:>+6.1%}")

    fig, ax = plt.subplots(figsize=(9, max(5, len(df) * 0.38)))
    y = np.arange(len(df))
    h = 0.35

    ax.barh(y + h/2, df['Pre'],  h, color=PRE_C,  alpha=0.85, label='Pre-COVID')
    ax.barh(y - h/2, df['Post'], h, color=POST_C, alpha=0.85, label='Post-COVID')

    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.set_xlabel('Portfolio weight', labelpad=8)
    ax.set_title(
        f'Tangency portfolio — top {top_n} holdings  ({freq.capitalize()} returns)\n'
        f'Pre: Jan 2015 – Jan 2020   |   Post: Jul 2021 – Dec 2024',
        fontsize=10
    )
    ax.legend(fontsize=9, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(length=3, color='#AAAAAA')
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ════════════════════════════════════════════════════════════════════
# Crisis correlation heatmap — daily log returns, independent slicing
# ════════════════════════════════════════════════════════════════════

def plot_crisis_heatmap(log_ret_daily):
    def clean(df, start, end):
        ret  = df.loc[start:end]
        keep = ret.columns[ret.isna().mean() <= 0.10]
        return ret[keep].ffill(limit=1).dropna(axis=1)

    ret_pre    = clean(log_ret_daily, PRE_START,    PRE_END)
    ret_crisis = clean(log_ret_daily, CRISIS_START, CRISIS_END)
    ret_post   = clean(log_ret_daily, POST_START,   POST_END)

    def avg_pairwise(df):
        c = df.corr().values
        mask = ~np.eye(len(c), dtype=bool)
        return c[mask].mean()

    print(f"\n  Avg pairwise correlation (daily log returns):")
    print(f"    Pre-COVID : {avg_pairwise(ret_pre):.3f}   T={len(ret_pre)}")
    print(f"    Crisis    : {avg_pairwise(ret_crisis):.3f}   T={len(ret_crisis)}  ← diversification collapse")
    print(f"    Post-COVID: {avg_pairwise(ret_post):.3f}   T={len(ret_post)}")

    corr  = ret_crisis.corr()
    order = corr.mean().sort_values(ascending=False).index
    corr  = corr.loc[order, order]
    labels = [t.replace('.NS', '') for t in corr.columns]
    corr.columns = labels
    corr.index   = labels

    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(12, n * 0.22), max(10, n * 0.18)))
    sns.heatmap(
        corr, ax=ax,
        cmap='coolwarm', center=0, vmin=-0.2, vmax=1.0,
        linewidths=0.3, linecolor='white',
        annot=True, fmt='.2f', annot_kws={'size': 6, 'color': '#333333'},
        cbar_kws={'shrink': 0.4, 'pad': 0.02}
    )
    ax.set_title(
        f'Daily log-return correlations — crisis window  (Feb 2020 – Jun 2021)\n'
        f'Avg = {avg_pairwise(ret_crisis):.3f}  vs  '
        f'pre = {avg_pairwise(ret_pre):.3f},  post = {avg_pairwise(ret_post):.3f}'
        f'   |   T={len(ret_crisis)} daily obs',
        fontsize=11, pad=14, color='#222222'
    )
    ax.tick_params(labelsize=7.5, length=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()
    fig.savefig('Figures/crisis_correlation.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: Figures/crisis_correlation.png")


# ════════════════════════════════════════════════════════════════════
# Rolling volatility
# ════════════════════════════════════════════════════════════════════

def plot_rolling_vol(log_ret_daily, out_path, window=252):
    ew       = log_ret_daily.mean(axis=1)
    roll_vol = (ew.rolling(window).std() * np.sqrt(252)).dropna()

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']: ax.spines[sp].set_color('#CCCCCC')
    ax.grid(True, linestyle='--', linewidth=0.4, color='#DDDDDD', alpha=0.9)

    # Regime shading
    ax.axvspan(pd.Timestamp('2015-01-01'), pd.Timestamp('2020-01-31'),
               alpha=0.07, color=PRE_C,     label='Pre-COVID regime')
    ax.axvspan(pd.Timestamp('2020-02-01'), pd.Timestamp('2021-06-30'),
               alpha=0.12, color='#E24B4A', label='Crisis window')
    ax.axvspan(pd.Timestamp('2021-07-01'), pd.Timestamp('2024-12-31'),
               alpha=0.07, color=POST_C,    label='Post-COVID regime')

    # Boundary lines
    for date, lbl, side in [
        ('2020-01-31', 'Pre → Crisis',  'right'),
        ('2021-07-01', 'Crisis → Post', 'left'),
    ]:
        ax.axvline(pd.Timestamp(date), color='#888888', lw=0.9, ls='--', alpha=0.7)
        ax.text(pd.Timestamp(date), roll_vol.max() * 1.02, lbl,
                fontsize=7.5, color='#777777', ha=side, va='bottom')

    ax.plot(roll_vol.index, roll_vol.values, color='#2C2C2A', lw=1.8, zorder=5,
            label=f'{window}-day rolling vol (EW)')
    ax.fill_between(roll_vol.index, roll_vol.values, alpha=0.12, color='#2C2C2A')

    # Key event annotations
    events = {
        'Mar 2020\nNifty −38%':       '2020-03-23',
        'Nov 2021\nRelCap insolvency': '2021-11-29',
        'Jul 2023\nHDFC merger':       '2023-07-01',
    }
    for label, date in events.items():
        ts  = pd.Timestamp(date)
        idx = roll_vol.index.get_indexer([ts], method='nearest')[0]
        yval = roll_vol.iloc[idx]
        ax.annotate(label, xy=(ts, yval),
                    xytext=(ts, yval + roll_vol.max() * 0.12),
                    fontsize=7.5, color='#555555', ha='center',
                    arrowprops=dict(arrowstyle='-', color='#AAAAAA', lw=0.7))

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.set_ylabel(f'Annualised Volatility  ({window}-day rolling)', fontsize=10.5, color='#555555')
    ax.set_xlabel('Date', fontsize=10.5, color='#555555')
    ax.tick_params(colors='#666666', labelsize=9.5)
    ax.set_title(
        'Rolling 252-day annualised volatility — equal-weight NIFTY portfolio\n'
        'Shaded regions: pre-COVID (green) · crisis (red) · post-COVID (purple)',
        fontsize=11, color='#222222', pad=10
    )
    ax.legend(fontsize=9, framealpha=0.6, loc='upper left')

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def run_freq(freq, log_ret, rf_pre, rf_post):
    print(f'\n=== {freq.upper()} ===')

    ret_pre  = slice_period(log_ret, PRE_START,  PRE_END)
    ret_post = slice_period(log_ret, POST_START, POST_END)

    shared = ret_pre.columns.intersection(ret_post.columns)
    ret_pre, ret_post = ret_pre[shared], ret_post[shared]
    tickers = list(shared)
    print(f'  Shared tickers: {len(shared)}  |  Pre T={len(ret_pre)}  Post T={len(ret_post)}')

    print('  Pre-COVID covariance:')
    mu_pre,  cov_pre  = ledoit_wolf(ret_pre,  freq)
    print('  Post-COVID covariance:')
    mu_post, cov_post = ledoit_wolf(ret_post, freq)

    plot_frontier(mu_pre, cov_pre, mu_post, cov_post,
                  rf_pre, rf_post, freq,
                  f'Figures/frontier_combined_{freq}.png')

    plot_tangency_weights(mu_pre, cov_pre, mu_post, cov_post,
                          rf_pre, rf_post, tickers, freq,
                          f'Figures/tangency_weights_{freq}.png')


def main():
    print('Loading RBI T-bill risk-free rates...')
    rf_pre, rf_post = load_risk_free_rates()

    print('\nLoading prices...')
    prices = load_prices()
    log_ret_daily  = to_log_returns(prices, 'daily')
    log_ret_weekly = to_log_returns(prices, 'weekly')

    run_freq('daily',  log_ret_daily,  rf_pre, rf_post)
    run_freq('weekly', log_ret_weekly, rf_pre, rf_post)

    print('\n=== CRISIS HEATMAP ===')
    plot_crisis_heatmap(log_ret_daily)

    print('\n=== ROLLING VOLATILITY ===')
    plot_rolling_vol(log_ret_daily, 'Figures/rolling_volatility.png')

    print('\nDone.')


if __name__ == '__main__':
    main()
