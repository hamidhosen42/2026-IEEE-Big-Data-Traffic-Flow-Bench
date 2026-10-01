"""EXP_021: window-matched comparison of oofF (links in travel direction) vs oofX control (order_index), same folds/windows/seeds.
Ongoing = P255, onset = O_cat8. Competition aggregation: window -> panel x cond -> panel -> family -> mean."""
import glob, numpy as np, pandas as pd
B = 'kaggle_jobs/out'
def decode(p):
    p = p.ravel().astype(np.float64); o = np.argsort(-p); cs = np.cumsum(p[o]); m = np.arange(1, len(p) + 1)
    j = np.argmax(cs / (m + cs[-1] - cs)); out = np.zeros(len(p), bool); out[o[:j + 1]] = True; return out
def iou(t, e, pr):
    t, e = t.ravel(), e.ravel(); u = (t & e | pr & e).sum(); return 1.0 if u == 0 else ((t & pr) & e).sum() / u
rows = []
for fx in sorted(glob.glob(f'{B}/oofX/oofX_*.npz')):
    ff = fx.replace('/oofX/oofX_', '/oofF/oofF_')
    try: zf = np.load(ff, allow_pickle=True)
    except FileNotFoundError: continue
    zx = np.load(fx, allow_pickle=True)
    n = fx.split('/')[-1][5:-4].split('_'); panel = '_'.join(n[:3]); s = n[3]; fold = int(n[4][1])
    kx = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(zx['d'], zx['t']))}
    for j, (a, b) in enumerate(zip(zf['d'], zf['t'])):
        i = kx.get((int(a), int(b)))
        if i is None: continue
        c = str(zf['cond'][j]); key = 'P255' if c == 'queue_ongoing' else 'O_cat8'
        rows.append(dict(panel=panel, set=s, fold=fold, cond=c, dir='WS' if panel[-1] in 'WS' else 'EN',
                         X=iou(zx['truth'][i], zx['ev'][i], decode(zx[key][i])), F=iou(zf['truth'][j], zf['ev'][j], decode(zf[key][j]))))
df = pd.DataFrame(rows)
print('matched windows', len(df), df.groupby(['set', 'cond']).size().to_dict())
def agg(d):
    g = d.groupby(['panel', 'cond'])[['X', 'F']].mean().groupby('panel').mean(); g['fam'] = g.index.str.rsplit('_', n=1).str[0]
    return g.groupby('fam')[['X', 'F']].mean().mean()
for s in ('pseudo', 'official'):
    d = df[df.set == s]; a = agg(d)
    print(f'{s}: control {a.X:.4f}  flip {a.F:.4f}  delta {a.F - a.X:+.4f}')
    print('  by cond', d.groupby('cond')[['X', 'F']].mean().round(4).to_dict('index'))
    print('  by dir x cond', d.groupby(['dir', 'cond'])[['X', 'F']].mean().round(4).to_dict('index'))
for k in range(4):
    a = agg(df[(df.set == 'pseudo') & (df.fold == k)]); print(f'fold {k}: delta {a.F - a.X:+.4f}')
