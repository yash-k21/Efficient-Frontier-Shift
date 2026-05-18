import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import os

os.makedirs('../results/figures', exist_ok=True)

CRISIS_START = '2020-02-01'
CRISIS_END   = '2021-06-30'

# ── Download NIFTY 50 ────────────────────────────────────────────────
nifty = yf.download('^NSEI', start='2015-01-01', end='2025-01-01',
                    auto_adjust=True, progress=False)['Close'].squeeze()
nifty = nifty.dropna()

# ── Plot ─────────────────────────────────────────────────────────────
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11})

fig, ax = plt.subplots(figsize=(11, 5))

# Crisis shading
ax.axvspan(pd.Timestamp(CRISIS_START), pd.Timestamp(CRISIS_END),
           color='#E8453C', alpha=0.10, zorder=1)
ax.axvline(pd.Timestamp(CRISIS_START), color='#E8453C',
           lw=0.9, linestyle='--', alpha=0.6, zorder=2)
ax.axvline(pd.Timestamp(CRISIS_END), color='#E8453C',
           lw=0.9, linestyle='--', alpha=0.6, zorder=2)

# Price line
ax.plot(nifty.index, nifty.values, color='#1a1a1a', lw=1.4, zorder=3)

# Crisis label
mid_crisis = pd.Timestamp('2020-10-15')
ax.text(mid_crisis, nifty.max() * 0.97, 'Crisis window\n(Feb 2020 – Jun 2021)',
        ha='center', va='top', fontsize=9, color='#E8453C')

# Period boundary annotations
for date, label, ha in [('2020-02-01', 'Feb 2020', 'right'),
                         ('2021-07-01', 'Jul 2021', 'left')]:
    ax.annotate(label,
                xy=(pd.Timestamp(date), nifty.loc[date:date].iloc[0] if date in nifty.index
                    else nifty.asof(pd.Timestamp(date))),
                xytext=(-12 if ha == 'right' else 12, -30),
                textcoords='offset points',
                fontsize=8.5, color='#E8453C', ha=ha,
                arrowprops=dict(arrowstyle='-', color='#E8453C', lw=0.7))

# Pre / Post shading labels
ax.text(pd.Timestamp('2017-06-01'), nifty.min() * 1.02,
        'Pre-COVID', fontsize=9, color='#1D9E75', alpha=0.8)
ax.text(pd.Timestamp('2022-09-01'), nifty.min() * 1.02,
        'Post-COVID', fontsize=9, color='#534AB7', alpha=0.8)

# Spines
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
for sp in ['left', 'bottom']:
    ax.spines[sp].set_color('#CCCCCC')
ax.tick_params(length=3, color='#AAAAAA')

ax.set_ylabel('Index Level', labelpad=8)
ax.set_title('NIFTY 50  (Jan 2015 – Dec 2024)', fontsize=13, pad=12, color='#222222')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))

fig.tight_layout()
out = '../results/figures/nifty50_timeline.png'
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved: {out}')

# ── Daily log returns ────────────────────────────────────────────────
ret = np.log(nifty / nifty.shift(1)).dropna()

PRE_START  = '2015-01-01'
PRE_END    = '2020-01-31'
POST_START = '2021-07-01'
POST_END   = '2024-12-31'

ret_pre  = ret.loc[PRE_START:PRE_END]
ret_post = ret.loc[POST_START:POST_END]

# Annualised stats
mean_pre  = ret_pre.mean()  * 252
vol_pre   = ret_pre.std()   * np.sqrt(252)
mean_post = ret_post.mean() * 252
vol_post  = ret_post.std()  * np.sqrt(252)

print(f"\nNIFTY 50 — Pre-COVID  ({PRE_START} – {PRE_END})")
print(f"  Annualised mean return : {mean_pre:.2%}")
print(f"  Annualised volatility  : {vol_pre:.2%}")
print(f"\nNIFTY 50 — Post-COVID ({POST_START} – {POST_END})")
print(f"  Annualised mean return : {mean_post:.2%}")
print(f"  Annualised volatility  : {vol_post:.2%}")

# ── Plot: daily returns ──────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(11, 5))

# Crisis shading
ax2.axvspan(pd.Timestamp(CRISIS_START), pd.Timestamp(CRISIS_END),
            color='#E8453C', alpha=0.10, zorder=1)
ax2.axvline(pd.Timestamp(CRISIS_START), color='#E8453C',
            lw=0.9, linestyle='--', alpha=0.6, zorder=2)
ax2.axvline(pd.Timestamp(CRISIS_END), color='#E8453C',
            lw=0.9, linestyle='--', alpha=0.6, zorder=2)

# Returns line
ax2.plot(ret.index, ret.values, color='#1a1a1a', lw=0.7, alpha=0.75, zorder=3)

# Zero reference
ax2.axhline(0, color='#AAAAAA', lw=0.8, zorder=2)

# Crisis label
ax2.text(pd.Timestamp('2020-10-15'), ret.max() * 0.92,
         'Crisis window\n(Feb 2020 – Jun 2021)',
         ha='center', va='top', fontsize=9, color='#E8453C')

# Period boundary annotations
for date, label, ha in [('2020-02-01', 'Feb 2020', 'right'),
                         ('2021-07-01', 'Jul 2021', 'left')]:
    ts = pd.Timestamp(date)
    y_val = ret.asof(ts) if ts not in ret.index else ret.loc[ts]
    ax2.annotate(label,
                 xy=(ts, y_val),
                 xytext=(-12 if ha == 'right' else 12, 20),
                 textcoords='offset points',
                 fontsize=8.5, color='#E8453C', ha=ha,
                 arrowprops=dict(arrowstyle='-', color='#E8453C', lw=0.7))

# Period stat annotations
stat_y = ret.min() * 0.75
ax2.text(pd.Timestamp('2017-06-01'), stat_y,
         f'Pre-COVID\nμ={mean_pre:.1%}  σ={vol_pre:.1%}',
         fontsize=8.5, color='#1D9E75', alpha=0.9, ha='center')
ax2.text(pd.Timestamp('2022-09-01'), stat_y,
         f'Post-COVID\nμ={mean_post:.1%}  σ={vol_post:.1%}',
         fontsize=8.5, color='#534AB7', alpha=0.9, ha='center')

# Spines
for sp in ['top', 'right']:
    ax2.spines[sp].set_visible(False)
for sp in ['left', 'bottom']:
    ax2.spines[sp].set_color('#CCCCCC')
ax2.tick_params(length=3, color='#AAAAAA')

ax2.set_ylabel('Log Return', labelpad=8)
ax2.set_title('NIFTY 50 — Daily Log Returns  (Jan 2015 – Dec 2024)',
              fontsize=13, pad=12, color='#222222')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))

fig2.tight_layout()
out2 = '../results/figures/nifty50_returns.png'
fig2.savefig(out2, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print(f'Saved: {out2}')
