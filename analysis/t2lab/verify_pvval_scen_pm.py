"""verify_pvval_scen_pm.py -- light (analysis only): the pseudo-March split at the PRIOR level.
pm : March days >= 15 under the prior of March days < 15 vs the train prior   (what the pseudo-March test rewards)
apr: April days under the prior of March days < 15 vs the train prior          (same 15 source days, the deployment month)
apr31: April days under all 31 March days vs train;  aprpm: April days under March days < 15 + (April days before d) vs train"""
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
rows = []
for panel in lab.PANELS:
    z = np.load(f'{lab.DATA}/t2data_{panel}.npz'); cut = 0.6 * z['vf'].astype(float)
    M = {}
    for sp in ('validation', 'private'):
        days = np.array([str(x) for x in z[f'{sp}_days']]); o = np.argsort(days)
        V = z[f'{sp}_Vm'][o].astype(np.float32); E = np.isfinite(V); Q = E & (V <= cut[None, None, :])
        wk = pd.to_datetime(pd.Series(days[o])).dt.dayofweek.to_numpy() >= 5
        M[sp] = (Q, E, wk, box(Q), box(E))
    TV = z['train_V'].astype(np.float32); TE = np.isfinite(TV); TQ = TE & (TV <= cut[None, None, :]); del TV
    twk = pd.to_datetime(pd.Series([str(x) for x in z['train_days']])).dt.dayofweek.to_numpy() >= 5
    p_tr = prior_from(box(TQ), box(TE), twk, np.ones(len(twk), bool))
    Qm, Em, wkm, Qms, Ems = M['validation']; Qa, Ea, wka, Qas, Eas = M['private']
    first = np.arange(len(wkm)) < 15
    p_m15 = prior_from(Qms, Ems, wkm, first); p_m31 = prior_from(Qms, Ems, wkm, np.ones(len(wkm), bool))
    for d in np.where(~first)[0]:
        rows.append(dict(panel=panel, kind='pm', day=int(d), llr=llr1(Qm[d], Em[d], int(wkm[d]), p_m15, p_tr)))
    for d in range(len(wka)):
        rows.append(dict(panel=panel, kind='apr', day=int(d), llr=llr1(Qa[d], Ea[d], int(wka[d]), p_m15, p_tr)))
        rows.append(dict(panel=panel, kind='apr31', day=int(d), llr=llr1(Qa[d], Ea[d], int(wka[d]), p_m31, p_tr)))
        # March (31) + April days before d, as the kernel's cumulative tables
        Qc = np.concatenate([Qms, Qas[:d]]); Ec = np.concatenate([Ems, Eas[:d]]); wc = np.concatenate([wkm, wka[:d]])
        rows.append(dict(panel=panel, kind='apr_kernelprior', day=int(d), llr=llr1(Qa[d], Ea[d], int(wka[d]), prior_from(Qc, Ec, wc, np.ones(len(wc), bool)), p_tr)))
df = pd.DataFrame(rows); pd.set_option('display.width', 250)
print(df.pivot_table(index='panel', columns='kind', values='llr', aggfunc='median').round(0).to_string())
pooled = df.groupby(['kind', 'day']).llr.sum()
print('pooled over panels: median', pooled.groupby(level=0).median().round(0).to_dict(), '| share > 0', pooled.gt(0).groupby(level=0).mean().round(3).to_dict())
print('single panel-day share > 0', df.groupby('kind').llr.apply(lambda s: round(float((s > 0).mean()), 3)).to_dict())
