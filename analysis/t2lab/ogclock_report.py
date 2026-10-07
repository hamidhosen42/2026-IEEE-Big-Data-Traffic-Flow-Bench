"""TRACK ogclock: paired comparison of ck arms against their reference arm (same rows / weights / params, identical folds).
Usage: ogclock_report.py ARM:REF[,ARM:REF ...] [tag]
REF is a lab oof arm (cache/oof/<REF>/results.pkl, e.g. fc_F03_np255) or another ogclock arm (prefix ck:)."""
import sys
import numpy as np, pandas as pd
import lab
import ogclock_lib as ck

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
pairs = [p.split(':', 1) for p in sys.argv[1].split(',')]
tag = sys.argv[2] if len(sys.argv) > 2 else 'report'


def load(a):
    return ck.load_arm(a[3:]) if a.startswith('ck:') else (ck.load_arm(a) if not a.startswith('fc_') else lab.load_arm(a))


rows = []
for arm, ref in pairs:
    B = load(arm); A = load(ref)
    folds = sorted(B.fold.unique()); A = A[A.fold.isin(folds)]
    A = ck.add_wtype(A); B = ck.add_wtype(B)
    for s in ('official', 'offlike', 'pseudo'):
        a, b = A[A.set == s].drop(columns='wtype'), B[B.set == s].drop(columns='wtype')
        rows.append(dict(ck.delta_row(a, b, f'{arm} vs {ref}', s), folds=''.join(map(str, folds))))
    for wt in ('midrun', 'rs_old', 'rs_new'):
        a = A[(A.set == 'offlike') & (A.wtype == wt)].drop(columns='wtype'); b = B[(B.set == 'offlike') & (B.wtype == wt)].drop(columns='wtype')
        rows.append(dict(ck.delta_row(a, b, f'{arm} vs {ref}', f'offlike:{wt}'), folds=''.join(map(str, folds))))
    a = A[A.set == 'pseudo']; b = B[B.set == 'pseudo']
    for nm, m in (('pseudo:faithful', 'faithful'), ):
        a2 = a[a[m] == True].drop(columns='wtype'); b2 = b[b[m] == True].drop(columns='wtype')
        rows.append(dict(ck.delta_row(a2, b2, f'{arm} vs {ref}', nm), folds=''.join(map(str, folds))))
    imp = ck.load_imp(arm)
    lab.log(f'{arm}: gain share of ck / tod columns: ' + ', '.join(f'{c} {v / imp.sum():.3f}' for c, v in imp.items() if c in ck.CK_ALL + ['tod', 'dow']))
D = pd.DataFrame(rows)
print(D.round(4).to_string(index=False))
D.to_csv(f'{ck.RES}/deltas_{tag}.csv', index=False)
lab.log('DONE')
