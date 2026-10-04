"""Report for firstcross arms: aggregated IoU, paired deltas against a base arm (bootstrap SE, per fold, per panel),
per-step profile, error split, subsets.  Light (no training).
Usage: exp_firstcross_report.py <base arm> <arm[,arm...]> [out tag] [folds e.g. 0123]
Arms are oof cache names (cache/oof/<arm>/results.pkl); blends are scored by exp_firstcross_blend.py and saved as arms too."""
import json, os, sys
import numpy as np
import pandas as pd
import lab
import exp_firstcross_lib as fc

base_name = sys.argv[1]
arms = sys.argv[2].split(',')
tag = sys.argv[3] if len(sys.argv) > 3 else 'main'
folds = [int(c) for c in sys.argv[4]] if len(sys.argv) > 4 else [0, 1, 2, 3]
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 500)
SETS = ('official', 'offlike', 'pseudo')


def load(a):
    df = lab.load_arm(a)
    return df[df.fold.isin(folds)].reset_index(drop=True)


base = load(base_name)
frames = {base_name: base}
for a in arms:
    frames[a] = load(a)

# ---- 1. summary
rows = []
for a, df in frames.items():
    for s in SETS:
        g = df[df.set == s]
        if not len(g):
            continue
        es = lab.error_split(g); sp = lab.step_profile(g)
        rows.append(dict(arm=a, set=s, n=len(g), folds=''.join(str(k) for k in sorted(g.fold.unique())), iou=lab.aggregate(g), plain_mean=g.iou.mean(),
                         **{f'h{h + 1}': v for h, v in enumerate(sp)}, missed_new=es['missed_new'], kept_cleared=es['kept_cleared'],
                         dropped_old=es['dropped_old'], false_new=es['false_new']))
summ = pd.DataFrame(rows)
print(f'== SUMMARY (folds {folds}) ==')
for s in SETS:
    print(summ[summ.set == s].round(4).to_string(index=False)); print()

# ---- 2. paired deltas vs base (common windows), bootstrap, per fold
rows = []; prow = []
for a in arms:
    df = frames[a]
    for s in SETS:
        A, B = fc.paired(base[base.set == s], df[df.set == s])
        if not len(A):
            continue
        b = fc.boot(A, B)
        r = dict(arm=a, set=s, n=b['n'], base=lab.aggregate(A), arm_iou=lab.aggregate(B), diff=b['diff'], se=b['se'], lo=b['lo'], hi=b['hi'], p_pos=b['p_pos'])
        pos = 0; nf = 0
        for k in sorted(A.fold.unique()):
            dk = lab.aggregate(B[B.fold == k]) - lab.aggregate(A[A.fold == k]); r[f'f{k}'] = dk; pos += dk > 0; nf += 1
        r['folds_pos'] = f'{pos}/{nf}'
        # windows won / lost
        d = B.iou.to_numpy() - A.iou.to_numpy(); r['win'] = int((d > 1e-9).sum()); r['loss'] = int((d < -1e-9).sum())
        rows.append(r)
        pa = A.groupby('panel').iou.mean(); pb = B.groupby('panel').iou.mean(); pn = A.groupby('panel').size()
        for p in pa.index:
            prow.append(dict(arm=a, set=s, panel=p, n=int(pn[p]), base=pa[p], arm_iou=pb[p], delta=pb[p] - pa[p]))
delt = pd.DataFrame(rows); pp = pd.DataFrame(prow)
print(f'== PAIRED DELTAS vs {base_name} (aggregate(arm) - aggregate(base), bootstrap within panel, 2000 draws) ==')
print(delt.round(4).to_string(index=False)); print()
print('== PER-PANEL DELTAS ==')
for s in SETS:
    t = pp[pp.set == s].pivot(index='panel', columns='arm', values='delta').reindex(columns=arms)
    t.insert(0, 'n', pp[(pp.set == s) & (pp.arm == arms[0])].set_index('panel').n)
    t.insert(1, 'base', pp[(pp.set == s) & (pp.arm == arms[0])].set_index('panel').base)
    npos = (t[arms] > 0).sum().rename('panels_pos')
    print(s); print(t.round(4).to_string()); print('panels positive:', npos.to_dict()); print()

# ---- 3. subsets
rows = []
SUB = [('offlike', 'runstart', lambda g: g.runstart.astype(bool)), ('offlike', 'not_runstart', lambda g: ~g.runstart.astype(bool)),
       ('pseudo', 'faithful', lambda g: g.faithful.astype(bool)), ('pseudo', 'stable_pf>0.9', lambda g: g.pers_fill > 0.9),
       ('official', 'runstart', lambda g: g.runstart.astype(bool)), ('official', 'not_runstart', lambda g: ~g.runstart.astype(bool))]
for a in arms:
    df = frames[a]
    for s, nm, fn in SUB:
        A, B = fc.paired(base[base.set == s], df[df.set == s])
        if not len(A):
            continue
        m = fn(A).to_numpy()
        if not m.any():
            continue
        b = fc.boot(A[m], B[m])
        rows.append(dict(arm=a, set=s, subset=nm, n=b['n'], base=lab.aggregate(A[m]), arm_iou=lab.aggregate(B[m]), diff=b['diff'], se=b['se'], lo=b['lo'], hi=b['hi']))
sub = pd.DataFrame(rows)
print('== SUBSETS ==')
print(sub.round(4).to_string(index=False))

out = f'{fc.RES}'
summ.to_csv(f'{out}/summary_{tag}.csv', index=False); delt.to_csv(f'{out}/deltas_{tag}.csv', index=False)
pp.to_csv(f'{out}/per_panel_{tag}.csv', index=False); sub.to_csv(f'{out}/subsets_{tag}.csv', index=False)
print(f'\nsaved {out}/[summary|deltas|per_panel|subsets]_{tag}.csv')
