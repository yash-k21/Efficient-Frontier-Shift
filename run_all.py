import subprocess
import sys

steps = [
    ('1. Download prices',           '1_data.py'),
    ('2. Compute daily statistics',  '2_mu_sigma_daily.py'),
    ('2. Compute weekly statistics', '2_mu_sigma_weekly.py'),
    ('3. Frontier + CAL + heatmap',  '3_frontier.py'),
    ('4. NIFTY 50 timeline',         '4_nifty50_plot.py'),
]

for label, script in steps:
    print(f'\n{"─"*50}')
    print(f'  {label}')
    print(f'{"─"*50}')
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f'\nERROR: {script} failed. Stopping.')
        sys.exit(result.returncode)

print('\n' + '─'*50)
print('  All steps completed.')
print('─'*50)
