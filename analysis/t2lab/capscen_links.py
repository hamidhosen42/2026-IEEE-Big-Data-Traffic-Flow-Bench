"""capscen_links.py -- per-link view of the train -> March/April change (released masked observations only).
Per link: q99 / q995 lane flow (free branch and all), queued share, and per-link ratio MAR/train, APR/train, and a train
month-vs-rest ratio as the null.  Looks for clusters at 1.2 or 1/1.2 (a capacity multiplier on a subset of links)."""
import os, sys, numpy as np, pandas as pd, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab
warnings.filterwarnings('ignore')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'capscen')
rng = np.random.default_rng(0)


def link_stats(V, F, cut, lanes):
    E = np.isfinite(V) & np.isfinite(F); fl = F / lanes
    out = []
    for l in range(V.shape[2]):
        e = E[:, :, l]; f = fl[:, :, l][e]; v = V[:, :, l][e]
        if e.sum() < 300:
            out.append((np.nan,) * 5); continue
        q = v <= cut[l]
        out.append((np.quantile(f, 0.99), np.quantile(f, 0.995), q.mean(), np.median(f[q]) if q.sum() > 10 else np.nan,
                    np.quantile(f[~q], 0.99)))
    return np.array(out)


rows = []
for panel in lab.PANELS:
    z = np.load(f'{lab.DATA}/t2data_{panel}.npz')
    vf = z['vf'].astype(float); cut = 0.6 * vf; cap = z['cap'].astype(float); lanes = z['lanes'].astype(float)
    TV = z['train_V'].astype(np.float32); TF = z['train_F'].astype(np.float32)
    keep = rng.random(TV.shape, dtype=np.float32) < 0.65
    TV = np.where(keep, TV, np.nan)
    tdays = np.array([str(x) for x in z['train_days']]); ym = np.array([d[:7] for d in tdays])
    S = {}
    S['train'] = link_stats(TV, TF, cut, lanes)
    S['feb'] = link_stats(TV[ym == '2031-02'], TF[ym == '2031-02'], cut, lanes)
    S['jan'] = link_stats(TV[ym == '2031-01'], TF[ym == '2031-01'], cut, lanes)
    S['tr_nofeb'] = link_stats(TV[ym != '2031-02'], TF[ym != '2031-02'], cut, lanes)
    for sp, tag in (('validation', 'mar'), ('private', 'apr')):
        V = z[f'{sp}_Vm'].astype(np.float32); Fm = z[f'{sp}_Fm'].astype(np.float32)
        S[tag] = link_stats(V, Fm, cut, lanes)
    for l in range(len(vf)):
        r = dict(panel=panel, link=str(z['links'][l]), order=int(z['order_index'][l]), lanes=lanes[l], capl=cap[l] / lanes[l], vf=vf[l])
        for k, a in S.items():
            for j, nm in enumerate(('q99', 'q995', 'qsh', 'qflow', 'q99free')):
                r[f'{k}_{nm}'] = a[l, j]
        rows.append(r)
    print(panel, flush=True)
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/links.csv', index=False)
pd.set_option('display.width', 250)
for a, b in (('mar', 'train'), ('apr', 'train'), ('feb', 'tr_nofeb'), ('mar', 'feb'), ('apr', 'mar')):
    rq = df[f'{a}_q995'] / df[f'{b}_q995']
    print(f'\n== per-link q995 ratio {a}/{b}: quantiles over links (all panels)')
    print(rq.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).round(3).to_string())
    print('share of links with ratio < 0.9:', round((rq < 0.9).mean(), 3), '| > 1.1:', round((rq > 1.1).mean(), 3),
          '| in [0.80,0.87]:', round(((rq > 0.80) & (rq < 0.87)).mean(), 3), '| in [1.15,1.25]:', round(((rq > 1.15) & (rq < 1.25)).mean(), 3))
    print(df.assign(r=rq).groupby('panel').r.describe(percentiles=[.05, .5, .95]).round(3).to_string())
print('\n== q995 / per-lane fd capacity, train, per panel')
print(df.assign(r=df.train_q995 / df.capl).groupby('panel').r.describe(percentiles=[.05, .5, .95]).round(3).to_string())
print('\n== links whose queued share changes most (MAR or APR vs train), top 25')
df['dq_mar'] = df.mar_qsh - df.train_qsh; df['dq_apr'] = df.apr_qsh - df.train_qsh
t = df.reindex(df[['dq_mar', 'dq_apr']].abs().max(1).sort_values(ascending=False).index).head(25)
print(t[['panel', 'link', 'order', 'train_qsh', 'feb_qsh', 'mar_qsh', 'apr_qsh', 'train_q995', 'mar_q995', 'apr_q995', 'train_q99free', 'mar_q99free', 'apr_q99free', 'capl']].round(3).to_string())
