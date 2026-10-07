"""TRACK ogclock: PSEUDO-MARCH gate (new scenario).  Final k = 4 models on all train days (F03 rows, lab scale 255 / 0.05 / 500).
Baselines: results/firstcross/final/model_F03_{np,id}255_500.txt (identical recipe, no ck columns).
Windows: faithful ongoing pseudo windows of the masked validation month (offlike_m, runstart_m; blank-touching windows dropped),
features: production window_features on the masked-layer history, ck features from the masked layer of the same day (slots <= T:
exactly the deployment input before T-12, sparser than the raw window history inside the 60 min), visible labels.
Usage: ogclock_pmarch.py KIND:GROUP[,KIND:GROUP ...]     e.g. np:all,id:all,np:link,np:corr"""
import os, sys
import numpy as np, pandas as pd, lightgbm as lgb
import lab
import exp_firstcross_lib as fc
import ogclock_lib as ck

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
specs = [s.split(':') for s in sys.argv[1].split(',')]
GROUPS = {'all': ck.CK_ALL, 'link': ck.CK_LINK, 'corr': ck.CK_CORR, 'none': []}
res = []
bases = {}
for kind, group in specs:
    base_kind = kind.split('+')[0]
    if base_kind not in bases:
        m0 = lgb.Booster(model_file=f'{fc.RES}/final/model_F03_{base_kind}255_500.txt')
        bases[base_kind] = m0
        res.append(ck.pm_score(m0, lab.colset(base_kind), [], f'base_{base_kind}'))
    m, pc, xc = ck.final_model(kind, GROUPS[group], tag=f'{kind}_{group}')
    res.append(ck.pm_score(m, pc, xc, f'{kind}_{group}'))
R = pd.concat(res, ignore_index=True)
tag = '_'.join(f'{k}-{g}' for k, g in specs).replace('+', 'p')
R.to_pickle(f'{ck.RES}/pmarch_{tag}.pkl')
rows = []
for kind, group in specs:
    arm = f'{kind}_{group}'; base = f'base_{kind.split("+")[0]}'
    for s in ck.PM_SETS:
        A = R[(R.arm == base) & (R.set == s)]; B = R[(R.arm == arm) & (R.set == s)]
        rows.append(ck.delta_row(A.drop(columns=['wtype']), B.drop(columns=['wtype']), f'{arm} vs {base}', s))
        if s == 'offlike_m':
            for wt in ('midrun', 'rs_old', 'rs_new'):
                a = A[A.wtype == wt].drop(columns=['wtype']); b = B[B.wtype == wt].drop(columns=['wtype'])
                rows.append(ck.delta_row(a, b, f'{arm} vs {base}', f'offlike_m:{wt}'))
            for nm, mk in (('blank_before', A.blank_before == True), ('no_blank', A.blank_before == False), ('vis<0.6', A.vis < 0.6), ('vis>=0.6', A.vis >= 0.6)):
                a = A[mk].drop(columns=['wtype']); b = B.merge(a[fc.KEY], on=fc.KEY).drop(columns=['wtype'])
                rows.append(ck.delta_row(a, b, f'{arm} vs {base}', f'offlike_m:{nm}'))
D = pd.DataFrame(rows)
print(D.round(4).to_string(index=False))
D.to_csv(f'{ck.RES}/pmarch_deltas_{tag}.csv', index=False)
lab.log('DONE')
