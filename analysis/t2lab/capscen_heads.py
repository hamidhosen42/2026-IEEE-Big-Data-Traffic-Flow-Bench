"""capscen_heads.py -- bottleneck discharge test of H1.
At a queue head (link i queued, its downstream neighbour j observed and free) the flow on j is the bottleneck discharge.
If the simulator ran at capacity x c relative to the published fd capacity, discharge / fd_cap(j) would sit near c x (1 - capacity drop).
Compared per split (train months, March, April) and per bottleneck location; also the congested-branch fit with per-lane kcrit.
Released masked observations only."""
import os, sys, numpy as np, pandas as pd, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab
warnings.filterwarnings('ignore')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'capscen')
rng = np.random.default_rng(0)


def heads(V, F, cut, cap, lanes, s, kcrit_l, kjam_l):
    E = np.isfinite(V) & np.isfinite(F); Q = E & (V <= cut)
    L = V.shape[2]; i = np.arange(L); j = i + s; ok = (j >= 0) & (j < L)
    i, j = i[ok], j[ok]
    hd = Q[:, :, i] & E[:, :, j] & ~Q[:, :, j]               # head at i, downstream j free
    fj = F[:, :, j] / cap[j]                                   # discharge / fd capacity of j
    fi = F[:, :, i] / cap[i]                                   # flow inside the queue head / its capacity
    res = dict(n_head=int(hd.sum()), dis_med=float(np.median(fj[hd])) if hd.any() else np.nan,
               dis_q90=float(np.quantile(fj[hd], 0.9)) if hd.any() else np.nan,
               inq_med=float(np.median(fi[hd])) if hd.any() else np.nan)
    # wrong-direction control (upstream neighbour of a queue tail)
    j2 = np.arange(L) - s; ok2 = (j2 >= 0) & (j2 < L); i2 = np.arange(L)[ok2]; j2 = j2[ok2]
    tl = Q[:, :, i2] & E[:, :, j2] & ~Q[:, :, j2]
    res['tail_up_med'] = float(np.median((F[:, :, j2] / cap[j2])[tl])) if tl.any() else np.nan
    # congested branch per lane: q = w (kjam - k) for queued cells with k > kcrit
    fl = F / lanes; k = fl / np.maximum(V, 1)
    cg = Q & (k > kcrit_l)
    kk = (k / kcrit_l)[cg]; ff = (fl / (cap / lanes))[cg]
    if cg.sum() > 100:
        b, a = np.polyfit(kk, ff, 1); res.update(cg_slope=b, cg_icpt=a, cg_k0=-a / b if b < 0 else np.nan, n_cg=int(cg.sum()))
    # per-link head counts / discharge
    per = np.full((L, 2), np.nan);
    for n, (ii, jj) in enumerate(zip(i, j)):
        h = hd[:, :, n]
        if h.sum() >= 10:
            per[ii] = (h.sum(), np.median(fj[:, :, n][h]))
    return res, per


rows = []; prow = []
for panel in lab.PANELS:
    z = np.load(f'{lab.DATA}/t2data_{panel}.npz')
    vf = z['vf'].astype(float); cut = 0.6 * vf; cap = z['cap'].astype(float); lanes = z['lanes'].astype(float)
    kc = z['fd_kcrit'].astype(float) / lanes; kj = z['fd_kjam'].astype(float) / lanes
    TV = z['train_V'].astype(np.float32); TF = z['train_F'].astype(np.float32)
    TV = np.where(rng.random(TV.shape, dtype=np.float32) < 0.65, TV, np.nan)
    tdays = np.array([str(x) for x in z['train_days']]); ym = np.array([d[:7] for d in tdays])
    sets = [(m, TV[ym == m], TF[ym == m]) for m in sorted(set(ym))]
    sets += [(sp, z[f'{sp}_Vm'].astype(np.float32), z[f'{sp}_Fm'].astype(np.float32)) for sp in ('validation', 'private')]
    for sgn_name, s in (('down', 1 if panel[-1] in 'EN' else -1), ('flip', -1 if panel[-1] in 'EN' else 1)):
        for tag, V, F in sets:
            r, per = heads(V, F, cut, cap, lanes, s, kc, kj)
            rows.append(dict(panel=panel, dir=sgn_name, month=tag, **r))
            if sgn_name == 'down':
                for l in np.where(np.isfinite(per[:, 0]))[0]:
                    prow.append(dict(panel=panel, month=tag, link=int(l), n=per[l, 0], dis=per[l, 1]))
    print(panel, flush=True)
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/heads.csv', index=False); pd.DataFrame(prow).to_csv(f'{OUT}/heads_links.csv', index=False)
pd.set_option('display.width', 250)
df['grp'] = np.where(df.month == 'validation', 'MAR', np.where(df.month == 'private', 'APR', 'train'))
for c in ('dis_med', 'dis_q90', 'inq_med', 'tail_up_med', 'cg_slope', 'cg_k0', 'n_head'):
    t = df[(df.grp == 'train')].groupby(['panel', 'dir'])[c].agg(['min', 'median', 'max'])
    t['MAR'] = df[df.grp == 'MAR'].set_index(['panel', 'dir'])[c]; t['APR'] = df[df.grp == 'APR'].set_index(['panel', 'dir'])[c]
    print(f'\n-- {c}'); print(t.round(3).to_string())
p = pd.DataFrame(prow); p['grp'] = np.where(p.month == 'validation', 'MAR', np.where(p.month == 'private', 'APR', 'train'))
g = p.groupby(['panel', 'link', 'grp']).agg(n=('n', 'sum'), dis=('dis', 'median')).unstack('grp')
g.columns = [f'{a}_{b}' for a, b in g.columns]
print('\n== bottleneck heads (>= 10 head cells in some split): discharge / fd cap per link, train vs MAR vs APR')
print(g.round(3).to_string())
