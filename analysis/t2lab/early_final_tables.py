"""early track (light): bootstrap tables of the final candidates from the stage C frames.
E1 recent-n (214 windows each), E2 month-start replay by n (all starts) and the Saturday-start replay (public calendar)."""
import sys
import numpy as np, pandas as pd
import lab, early_lib as el, early_engine as en
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
NAMES = {'sp4': 'sp4', 'sp8': 'sp8', 'A_w24p_k4': 'blend_w24', 'bma|w6|pi0.8|mu0.05|nu1.0': 'M8', 'bma|w6|pi0.9|mu0.05|nu1.0': 'M9',
         'bma|w4|pi0.9|mu0.05|nu1.0': 'M9w4', 'bma|w6|pi0.95|mu0.05|nu1.0': 'M95'}
R1 = pd.read_pickle(f'{el.OUT}/scanC_e1.pkl'); R2 = pd.read_pickle(f'{el.OUT}/scanC_e2.pkl')
R1 = R1[R1.design.isin(['base'] + list(NAMES))].copy(); R1['design'] = R1.design.replace(NAMES)
R2 = R2[R2.design.isin(['base'] + list(NAMES))].copy(); R2['design'] = R2.design.replace(NAMES)
ds = list(NAMES.values())
print('######## E1: evidence = the n most recent scenario days (no hidden horizons), 214 pseudo-March onset windows')
for n in (1, 2, 3, 4, 99):
    print(f'\n-- n = {n if n < 99 else "all days < d"}')
    print(en.table(R1[R1.nrec == n], ref='base', ref2='sp4', designs=['base'] + ds).round(4).to_string(index=False))
x = R1[(R1.nrec == 99) & (R1.d >= 15)]
print('\n-- April-like regime: all earlier March days, windows on days >= 16 (>= 15 scenario days)')
print(en.table(x, ref='base', ref2='sp4', designs=['base'] + ds).round(4).to_string(index=False))
print(en.panel_table(x, ['base'] + ds).round(3).to_string())
print('\n######## E2: month-start replay, all 31 start days, evidence = days d0..d-1 with the horizons of the replay picks hidden')
for n in (1, 2, 3, 4):
    print(f'\n-- n = d - d0 = {n}')
    print(en.table(R2[R2.n == n], ref='base', ref2='sp4', designs=['base'] + ds).round(4).to_string(index=False))
sat = R2[R2.d0.isin([7, 14, 21, 28])]
print('\n######## E2 Saturday starts (d0 = 7, 14, 21, 28): the public calendar (Sat, Sun, Mon, Tue, Wed ...), n = 0..6 mixed as official')
print('items by n:', sat[sat.design == 'base'].groupby('n').size().to_dict())
print(en.table(sat, ref='base', ref2='sp4', designs=['base'] + ds).round(4).to_string(index=False))
print(en.panel_table(sat, ['base'] + ds).round(3).to_string())
for lab_, m in (('n=0 (first day)', sat.n == 0), ('n=1 (Sunday, evidence Saturday)', sat.n == 1), ('n=2 (Monday, evidence Sat+Sun)', sat.n == 2),
                ('n=3 (Tuesday)', sat.n == 3), ('n>=4', sat.n >= 4)):
    y = sat[m]
    print(f'\n-- Saturday starts, {lab_}: items', int((y.design == 'base').sum()))
    print(en.table(y, ref='base', ref2='sp4', designs=['base'] + ds, boot=False).round(4).to_string(index=False))
