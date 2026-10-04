"""ogfeat: do the candidate features explain the RESIDUALS of the baseline F03 np255 OOF probabilities (official-like windows)?
For every candidate column: rows binned into 12 quantile bins; score = sum_bins n_bin * |mean(y - p)| / N (calibration gap along the
feature).  A column the model already uses well scores near the noise floor (see the production columns for reference)."""
import numpy as np, pandas as pd
import lab, ogfeat_lib as og
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
names = og.static_names(); pc = lab.all_cols()
Xs, ys, ps, hs = [], [], [], []
for panel in lab.PANELS:
    rc = lab.rows(panel, 'offlike'); P6, done = lab.load_probs('fc_F03_np255', 'offlike', panel)
    FE = og.build_fe(panel, 'offlike')
    fold = rc.win.fold.to_numpy()[rc.wi]
    m = np.zeros(rc.n, bool)
    for k in range(4):
        m |= (fold == k) & rc.keep5[:, k] & rc.e
    r = np.where(m)[0]
    Xp = np.asarray(rc.X[r])                                    # production columns (prior columns unused here)
    Xe = og.extra_matrix(FE, rc.wi[r], rc.li[r], rc.h[r], names, og.OG_STEP)
    Xs.append(np.concatenate([Xp, Xe], 1)); ys.append(rc.y[r]); ps.append(P6[rc.wi[r], rc.h[r] - 1, rc.li[r]]); hs.append(rc.h[r])
X = np.concatenate(Xs); y = np.concatenate(ys).astype(np.float32); p = np.concatenate(ps); h = np.concatenate(hs)
cols = pc + names + og.OG_STEP
res = y - p
print('rows', len(y), 'pos', y.mean().round(4), 'mean p', p.mean().round(4), 'logloss', float(-(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))).mean()))
out = []
for hsel, tag in ((h > 0, 'all'), (h == 6, 'h6')):
    for j, c in enumerate(cols):
        v = X[hsel, j]; r = res[hsel]; ok = np.isfinite(v)
        if ok.sum() < 1000:
            continue
        q = np.unique(np.quantile(v[ok], np.linspace(0, 1, 13)))
        b = np.clip(np.searchsorted(q, v[ok], side='right') - 1, 0, len(q) - 2)
        g = pd.DataFrame(dict(b=b, r=r[ok])).groupby('b').r.agg(['mean', 'size'])
        gap = float((g['mean'].abs() * g['size']).sum() / len(r))
        if (~ok).sum() > 100:
            gap += abs(r[~ok].mean()) * (~ok).sum() / len(r)
        out.append(dict(step=tag, col=c, gap=gap, bins=len(g), nan_share=float((~ok).mean())))
D = pd.DataFrame(out).pivot(index='col', columns='step', values='gap').sort_values('h6', ascending=False)
D['kind'] = ['prod-np' if c in lab.colset('np') else ('prod-other' if c in pc else 'new') for c in D.index]
D.to_csv(f'{og.RES}/resid_gap.csv')
print(D.round(5).head(60).to_string())
print('\nmedian gap of production np columns (noise floor): all', D[D.kind == 'prod-np']['all'].median().round(5), 'h6', D[D.kind == 'prod-np']['h6'].median().round(5))
