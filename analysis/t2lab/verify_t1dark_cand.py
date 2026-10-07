"""verify_t1dark_cand.py - light re-analysis of the candidate's saved per-cell predictions (results/t1dark/*.npz).
Recomputes the headline from raw cells (own code), then: drop-one-panel, per-step k = t - T profile, per-event history
visibility bins (masked-layer obs share in T-12..T-1; real blackouts have 1.00), by regime, by month. Event bootstrap."""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(HERE, 'results', 't1dark')
CACHE = os.path.join(ROOT, 'data', 't2local', 'cache', 't1dark')
PANELS = ['D7_I405_S', 'D7_I210_W', 'D12_I5_S', 'D7_I10_E']
B = 1000


def month_obs(panel, split):
    z = np.load(f'{ROOT}/data/t2local/t2data_{panel}.npz', allow_pickle=True)
    d = z[f'{split}_days'].astype(str); o = np.argsort(d)
    Vm = z[f'{split}_Vm'][o].astype(np.float32); Fm = z[f'{split}_Fm'][o]; Em = z[f'{split}_Em'][o]
    return Em & np.isfinite(Vm) & np.isfinite(Fm), Em


rows = []
for p in PANELS:
    meta = np.load(f'{CACHE}/{p}_meta.npz', allow_pickle=True)
    for sp in ('validation', 'private'):
        obs, Em = month_obs(p, sp)
        sim = meta[f'{sp}_sim']
        hvis = np.array([obs[d, t - 12:t].sum() / max(1, Em[d, t - 12:t].sum()) for d, t, c in sim])
        for f in (0, 1):
            z = np.load(f'{RES}/{p}_f{f}.npz', allow_pickle=True)
            z2 = np.load(f'{RES}/{p}_f{f}_s10.npz', allow_pickle=True) if os.path.exists(f'{RES}/{p}_f{f}_s10.npz') else None
            k = f'{sp}_blk'
            ev = z[f'{k}|ev']; t = z[f'{k}|t']
            vt = z[f'{k}|vt']; qt = z[f'{k}|qt']; ln = z[f'{k}|ln']
            if z2 is not None:     # D7_I405_S: second seed in the _s10 file
                b1, d1 = z[f'base|{k}|v'], z[f'dark|{k}|v']; b2, d2 = z2[f'base|{k}|v'], z2[f'dark|{k}|v']
                qb1, qd1 = z[f'base|{k}|q'], z[f'dark|{k}|q']; qb2, qd2 = z2[f'base|{k}|q'], z2[f'dark|{k}|q']
            else:
                b1, d1, b2, d2 = z[f'base|{k}|v'], z[f'dark|{k}|v'], z[f'base2|{k}|v'], z[f'dark2|{k}|v']
                qb1, qd1, qb2, qd2 = z[f'base|{k}|q'], z[f'dark|{k}|q'], z[f'base2|{k}|q'], z[f'dark2|{k}|q']
            bagB = 0.5 * (b1.astype(float) + b2); bagD = 0.5 * (d1.astype(float) + d2)
            qB = 0.5 * (qb1.astype(float) + qb2); qD = 0.5 * (qd1.astype(float) + qd2)
            T = sim[ev, 1]
            assert (sim[ev, 0] == z[f'{k}|d']).all()
            df = pd.DataFrame(dict(panel=p, split=sp, fold=f, ev=ev, k=t - T, cond=sim[ev, 2], reg=z[f'{k}|reg'],
                                   hvis=hvis[ev], eB=(bagB - vt) ** 2, eD=(bagD - vt) ** 2, e1=(b1 - vt) ** 2, e1d=(d1 - vt) ** 2,
                                   qB=((qB - qt) / ln) ** 2, qD=((qD - qt) / ln) ** 2))
            rows.append(df)
A = pd.concat(rows, ignore_index=True)
assert A.k.between(1, 18).all()
A['pm'] = A.panel + '|' + A.split
print('cells', len(A), 'events', A.groupby(['pm', 'ev']).ngroups)


def pooled(df, a='eB', b='eD'):
    return np.sqrt(df[a].mean()), np.sqrt(df[b].mean())


def boot(df, a='eB', b='eD', B=B, seed=0):
    g = df.groupby(['pm', 'ev']).agg(n=(a, 'size'), sa=(a, 'sum'), sb=(b, 'sum')).reset_index()
    rng = np.random.default_rng(seed)
    out = []
    pms = g.pm.unique()
    idx = {pm: np.flatnonzero(g.pm.to_numpy() == pm) for pm in pms}
    n, sa, sb = g.n.to_numpy(), g.sa.to_numpy(), g.sb.to_numpy()
    for _ in range(B):
        w = np.zeros(len(g))
        for pm in pms:
            ii = idx[pm]; w[ii] = np.bincount(rng.integers(0, len(ii), len(ii)), minlength=len(ii))
        N = (w * n).sum(); out.append(np.sqrt((w * sb).sum() / N) - np.sqrt((w * sa).sum() / N))
    return np.percentile(out, [2.5, 97.5])


def line(name, df, a='eB', b='eD'):
    ra, rb = pooled(df, a, b); lo, hi = boot(df, a, b)
    win = sum(np.sqrt(g[b].mean()) < np.sqrt(g[a].mean()) for _, g in df.groupby('pm'))
    print(f'  {name:34s} n_ev {df.groupby(["pm", "ev"]).ngroups:4d} speed {ra:.3f} -> {rb:.3f} d {rb - ra:+.3f} [{lo:+.3f}, {hi:+.3f}] '
          f'({100 * (rb - ra) / ra:+.1f}%) win {win}/{df.pm.nunique()}')


print('\n== headline recomputed from raw cells (bagB -> bagD), speed')
line('all 8 panel-months', A)
line('single seed base -> dark', A, 'e1', 'e1d')
qa, qb = np.sqrt(A.qB.mean()), np.sqrt(A.qD.mean()); print(f'  flow/lane bagB -> bagD {qa:.2f} -> {qb:.2f} d {qb - qa:+.2f}')
print('\n== drop one panel')
for p in PANELS:
    line(f'without {p}', A[A.panel != p])
print('\n== by month')
for sp in ('validation', 'private'):
    line(sp, A[A.split == sp])
print('\n== per step k = t - T (pooled over panel-months)')
for lo_, hi_ in ((1, 3), (4, 6), (7, 12), (13, 18)):
    line(f'k {lo_}..{hi_}', A[A.k.between(lo_, hi_)])
print('\n== per-event history visibility (obs/elig in T-12..T-1; real blackouts = 1.00) -- quartiles')
q = A.groupby(['pm', 'ev']).hvis.first().quantile([0.25, 0.5, 0.75]).to_numpy()
print('  hvis quartiles', np.round(q, 3))
edges = [-1, q[0], q[1], q[2], 2]
for i in range(4):
    sub = A[(A.hvis > edges[i]) & (A.hvis <= edges[i + 1])]
    line(f'hvis bin {i} ({edges[i]:.2f},{edges[i + 1]:.2f}]', sub)
ev = A.groupby(['pm', 'ev']).agg(h=('hvis', 'first'), sB=('eB', 'sum'), sD=('eD', 'sum'), n=('eB', 'size')).reset_index()
ev['gain'] = np.sqrt(ev.sD / ev.n) - np.sqrt(ev.sB / ev.n)
ev['rel'] = ev.gain / np.sqrt(ev.sB / ev.n)
X = np.column_stack([np.ones(len(ev)), ev.h]); beta = np.linalg.lstsq(X, ev.gain, rcond=None)[0]
bR = np.linalg.lstsq(X, ev.rel, rcond=None)[0]
print(f'  OLS per-event RMSE gain = {beta[0]:+.3f} + {beta[1]:+.3f} * hvis -> extrapolated at hvis=1.0: {beta[0] + beta[1]:+.3f} '
      f'(mean hvis {ev.h.mean():.3f}); relative gain at 1.0: {100 * (bR[0] + bR[1]):+.1f}%')
print('\n== by regime')
for r in ('R1', 'R2', 'R3'):
    line(r, A[A.reg == r])
print('\n== by condition')
for c, nm in ((0, 'onset'), (1, 'ongoing')):
    line(nm, A[A.cond == c])
