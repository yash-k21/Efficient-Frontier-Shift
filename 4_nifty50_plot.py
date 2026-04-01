import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import os

os.makedirs('Figures', exist_ok=True)

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
out = 'Figures/nifty50_timeline.png'
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved: {out}')
