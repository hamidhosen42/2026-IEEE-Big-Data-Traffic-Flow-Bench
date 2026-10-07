"""verify_pvval_scen_drift.py -- light, March + train only (no April data): within-month drift at the prior level.
For days of the second half (day index >= 15) of a month: LLR(prior of the first-half days vs leave-one-day-out prior of the other
second-half days).  ~0 = no drift (first half as good as same-period days); << 0 = the pattern moves within the month.
March (released masked layer) vs every train month (i.i.d. mask at visibility 0.65 to match the masked noise)."""
import numpy as np, pandas as pd
import lab
EPS = 0.01; S0, S1 = 60, 252
def box(a):
    c = np.cumsum(np.pad(a.astype(np.float32), ((0, 0), (3, 2), (0, 0))), 1); return c[:, 5:] - c[:, :-5]
def prior_from(Qs, Es, wk, mask):
    out = []
    for cls in (False, True):
        m = mask & (wk == cls)
        if not m.any(): m = mask
        out.append(Qs[m].sum(0) / np.maximum(Es[m].sum(0), 1e-9))
    return np.clip(np.stack(out), EPS, 1 - EPS)
def llr1(Q, E, c, pA, pB):
    q = Q[S0:S1]; e = E[S0:S1]
    a = np.where(q, np.log(pA[c, S0:S1]), np.log(1 - pA[c, S0:S1])); b = np.where(q, np.log(pB[c, S0:S1]), np.log(1 - pB[c, S0:S1]))
    return float(((a - b) * e).sum())
def month_rows(Q, E, wk, tag, panel):
    Qs, Es = box(Q), box(E); D = len(wk); first = np.arange(D) < 15; out = []
    pf = prior_from(Qs, Es, wk, first)
    for d in np.where(~first)[0]:
        keep = ~first; keep = keep.copy(); keep[d] = False
        out.append(dict(panel=panel, month=tag, day=int(d), llr=llr1(Q[d], E[d], int(wk[d]), pf, prior_from(Qs, Es, wk, keep))))
    return out
rows = []; rng = np.random.default_rng(1)
for panel in lab.PANELS:
    z = np.load(f'{lab.DATA}/t2data_{panel}.npz'); cut = 0.6 * z['vf'].astype(float)
    days = np.array([str(x) for x in z['validation_days']]); o = np.argsort(days)
    V = z['validation_Vm'][o].astype(np.float32); E = np.isfinite(V); Q = E & (V <= cut[None, None, :])
    rows += month_rows(Q, E, pd.to_datetime(pd.Series(days[o])).dt.dayofweek.to_numpy() >= 5, 'march', panel)
    TV = z['train_V'].astype(np.float32); TV = np.where(rng.random(TV.shape, dtype=np.float32) < 0.65, TV, np.nan)
    TE = np.isfinite(TV); TQ = TE & (TV <= cut[None, None, :]); del TV
    td = pd.to_datetime(pd.Series([str(x) for x in z['train_days']])); ym = (td.dt.year * 100 + td.dt.month).to_numpy(); twk = td.dt.dayofweek.to_numpy() >= 5
    for m in sorted(set(ym)):
        ix = np.where(ym == m)[0]
        rows += month_rows(TQ[ix], TE[ix], twk[ix], str(m), panel)
df = pd.DataFrame(rows); df['kind'] = np.where(df.month == 'march', 'march', 'train')
pooled = df.groupby(['kind', 'month', 'day']).llr.sum().reset_index()
print('pooled over panels, per day: median by kind', pooled.groupby('kind').llr.median().round(0).to_dict(), '| share > 0', pooled.assign(p=pooled.llr > 0).groupby('kind').p.mean().round(3).to_dict())
print('train months (median pooled):', pooled[pooled.kind == 'train'].groupby('month').llr.median().round(0).to_dict())
print('per panel median, march vs train:'); print(df.pivot_table(index='panel', columns='kind', values='llr', aggfunc='median').round(0).to_string())
