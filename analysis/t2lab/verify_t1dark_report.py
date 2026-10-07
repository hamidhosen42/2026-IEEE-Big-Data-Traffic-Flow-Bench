"""verify_t1dark_report.py - summary of the independent verifier runs (results/verify_t1dark/<PANEL>_f<FOLD><tag>.npz).
Per panel-month and pooled blackout speed / flow-per-lane RMSE base -> dark (single seed), event bootstrap within
panel-month, 'std' vs 'sparse' history view, per-step and per-condition breakdown, ordinary cells.
Usage: verify_t1dark_report.py [tag=] [B=1000]"""
import glob, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESV = os.path.join(HERE, 'results', 'verify_t1dark')
kw = dict(a.split('=') for a in sys.argv[1:] if '=' in a)
TAG = kw.get('tag', '')
B = int(kw.get('B', 1000))
rows = []
for f in sorted(glob.glob(f'{RESV}/*_f[01]{TAG}.npz')):
    base = os.path.basename(f)[:-4]
    if 'smoke' in base or (TAG == '' and not base[-2:] in ('f0', 'f1')):
        continue
    p, fold = base[:len(base) - len(TAG)].rsplit('_f', 1)
    z = np.load(f, allow_pickle=True)
    for sp in ('validation', 'private'):
        for view, kind in (('std', 'blk'), ('std', 'ord'), ('sparse', 'blk')):
            pre = f'{sp}|{view}|{kind}'
            if f'{pre}|vt' not in z.files:
                continue
            vt, qt, ln = z[f'{pre}|vt'], z[f'{pre}|qt'], z[f'{pre}|ln']
            rows.append(pd.DataFrame(dict(panel=p, fold=int(fold), split=sp, view=view, kind=kind,
                                          ev=z[f'{pre}|ev'] if kind == 'blk' else z[f'{pre}|d'], k=z[f'{pre}|k'], cond=z[f'{pre}|cond'],
                                          eb=(z[f'{pre}|base|v'] - vt) ** 2, ed=(z[f'{pre}|dark|v'] - vt) ** 2,
                                          qb=((z[f'{pre}|base|q'] - qt) / ln) ** 2, qd=((z[f'{pre}|dark|q'] - qt) / ln) ** 2)))
A = pd.concat(rows, ignore_index=True)
A['pm'] = A.panel + '|' + A.split
A['unit'] = A.fold.astype(str) + ':' + A.ev.astype(str)          # blackout events are unique per (fold, ev)


def boot(df, a, b, seed=0):
    g = df.groupby(['pm', 'unit']).agg(n=(a, 'size'), sa=(a, 'sum'), sb=(b, 'sum')).reset_index()
    rng = np.random.default_rng(seed)
    idx = {pm: np.flatnonzero(g.pm.to_numpy() == pm) for pm in g.pm.unique()}
    n, sa, sb = g.n.to_numpy(), g.sa.to_numpy(), g.sb.to_numpy()
    out = []
    for _ in range(B):
        w = np.zeros(len(g))
        for ii in idx.values():
            w[ii] = np.bincount(rng.integers(0, len(ii), len(ii)), minlength=len(ii))
        N = (w * n).sum()
        out.append(np.sqrt((w * sb).sum() / N) - np.sqrt((w * sa).sum() / N))
    return np.percentile(out, [2.5, 97.5])


def line(name, df, a='eb', b='ed', ci=True):
    ra, rb = np.sqrt(df[a].mean()), np.sqrt(df[b].mean())
    lo, hi = boot(df, a, b) if ci else (np.nan, np.nan)
    win = sum(np.sqrt(g[b].mean()) < np.sqrt(g[a].mean()) for _, g in df.groupby('pm'))
    print(f'  {name:40s} cells {len(df):7d} {ra:8.3f} -> {rb:8.3f} d {rb - ra:+.3f} [{lo:+.3f}, {hi:+.3f}] ({100 * (rb - ra) / ra:+.1f}%) '
          f'win {win}/{df.pm.nunique()}')


blk = A[(A.kind == 'blk') & (A.view == 'std')]
print('folds present:', sorted(set(zip(A.panel, A.fold))))
print('\n== per panel-month BLACKOUT speed RMSE (std view), base -> dark')
for pm, g in blk.groupby('pm', sort=False):
    line(pm, g)
print('\n== pooled')
line('blackout speed, all', blk)
line('blackout flow/lane, all', blk, 'qb', 'qd')
for sp in ('validation', 'private'):
    line(f'blackout speed, {sp}', blk[blk.split == sp])
for c, nm in ((0, 'onset'), (1, 'ongoing')):
    line(f'blackout speed, {nm}', blk[blk.cond == c])
for lo_, hi_ in ((1, 3), (4, 6), (7, 12), (13, 18)):
    line(f'blackout speed, k {lo_}..{hi_}', blk[blk.k.between(lo_, hi_)])
sp_ = A[(A.kind == 'blk') & (A.view == 'sparse')]
if len(sp_):
    print('\n== history-sparsified view (extra 50% of observed T-12..T cells hidden)')
    line('blackout speed, sparse view', sp_)
    line('blackout flow/lane, sparse view', sp_, 'qb', 'qd')
o = A[A.kind == 'ord']
o = o.groupby(['pm', 'ev', 'k', 'cond']).mean(numeric_only=True).reset_index() if False else o
print('\n== ordinary cells (std view; folds counted separately)')
line('ordinary speed', o.assign(unit=o.fold.astype(str) + ':' + o.ev.astype(str)))
line('ordinary flow/lane', o.assign(unit=o.fold.astype(str) + ':' + o.ev.astype(str)), 'qb', 'qd')
