"""capscen_ramps.py -- do ramp demands (released ON/OFF counts attached to mainline links) shift train -> March/April
at specific locations?  Per link with ramp data: mean ON / OFF in AM (06-10h) and PM (14-19h), March/train and April/train,
null = Feb / other train months.  Also mainline flow at the same links."""
import os, sys, numpy as np, pandas as pd, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab
warnings.filterwarnings('ignore')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'capscen')
rows = []
for panel in lab.PANELS:
    z = np.load(f'{lab.DATA}/t2data_{panel}.npz')
    td = np.array([str(x)[:7] for x in z['train_days']])
    def m(a, sl): return np.nanmean(a[:, sl].astype(np.float32), axis=(0, 1))
    S = {}
    for nm, ix in (('tr', td != '2031-02'), ('feb', td == '2031-02')):
        for ch in ('ON', 'OFF', 'F'):
            a = z[f'train_{ch}'][ix]
            S[(nm, ch, 'am')] = m(a, slice(72, 120)); S[(nm, ch, 'pm')] = m(a, slice(168, 228))
    for nm, sp in (('mar', 'validation'), ('apr', 'private')):
        for ch, key in (('ON', 'ON'), ('OFF', 'OFF'), ('F', 'Fm')):
            a = z[f'{sp}_{key}']
            S[(nm, ch, 'am')] = m(a, slice(72, 120)); S[(nm, ch, 'pm')] = m(a, slice(168, 228))
    L = len(z['links'])
    for l in range(L):
        r = dict(panel=panel, link=l)
        for (nm, ch, p), v in S.items(): r[f'{nm}_{ch}_{p}'] = v[l]
        rows.append(r)
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/ramps.csv', index=False)
pd.set_option('display.width', 250)
for ch in ('ON', 'OFF', 'F'):
    for p in ('am', 'pm'):
        base = df[f'tr_{ch}_{p}']; ok = base > (30 if ch != 'F' else 300)
        for nm in ('feb', 'mar', 'apr'):
            rr = (df[f'{nm}_{ch}_{p}'] / base)[ok]
            print(f'{ch:3s} {p} {nm}/train  n={ok.sum():4d}  ratio q05 {rr.quantile(.05):.3f} q25 {rr.quantile(.25):.3f} med {rr.median():.3f} q75 {rr.quantile(.75):.3f} q95 {rr.quantile(.95):.3f}  |dev|>15%: {(abs(rr-1)>0.15).mean():.3f}')
print('\nper panel PM on-ramp total ratio (sum over links): feb / mar / apr vs train')
for panel, g in df.groupby('panel'):
    t = g.tr_ON_pm.sum(); print(panel, *(round(g[f'{n}_ON_pm'].sum() / t, 3) for n in ('feb', 'mar', 'apr')),
                                '| OFF', *(round(g[f'{n}_OFF_pm'].sum() / g.tr_OFF_pm.sum(), 3) for n in ('feb', 'mar', 'apr')),
                                '| mainline F', *(round(np.nansum(g[f'{n}_F_pm']) / np.nansum(g.tr_F_pm), 3) for n in ('feb', 'mar', 'apr')))
