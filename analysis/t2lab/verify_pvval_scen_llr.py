"""verify_pvval_scen_llr.py -- light: the exp_scenario_llr.py test extended to APRIL (analysis only, no model).
Per April day and panel: LLR = sum over visible cells (05:00-21:00) of log P(q | March prior, all 31 days) - log P(q | train prior).
Same-scenario hypothesis -> April LLR > 0 like March leave-one-day-out (median pooled +2375); train-like -> < 0 like train fold 0 (-7217).
Also: April under the April leave-one-day-out prior vs train, and March prior vs April-LOO prior."""
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
def llr(Q, E, wk, pA, pB):
    out = []
    for d in range(Q.shape[0]):
        c = int(wk[d]); q = Q[d, S0:S1]; e = E[d, S0:S1]
        a = np.where(q, np.log(pA[c, S0:S1]), np.log(1 - pA[c, S0:S1])); b = np.where(q, np.log(pB[c, S0:S1]), np.log(1 - pB[c, S0:S1]))
        out.append(float(((a - b) * e).sum()))
    return np.array(out)
rows = []
for panel in lab.PANELS:
    z = np.load(f'{lab.DATA}/t2data_{panel}.npz'); cut = 0.6 * z['vf'].astype(float)
    M = {}
    for sp in ('validation', 'private'):
        days = np.array([str(x) for x in z[f'{sp}_days']]); o = np.argsort(days)
        V = z[f'{sp}_Vm'][o].astype(np.float32); E = np.isfinite(V); Q = E & (V <= cut[None, None, :])
        wk = pd.to_datetime(pd.Series(days[o])).dt.dayofweek.to_numpy() >= 5
        Qs, Es = box(Q), box(E); M[sp] = (Q, E, wk, Qs, Es)
    TV = z['train_V'].astype(np.float32); TE = np.isfinite(TV); TQ = TE & (TV <= cut[None, None, :])
    twk = pd.to_datetime(pd.Series([str(x) for x in z['train_days']])).dt.dayofweek.to_numpy() >= 5
    p_tr = prior_from(box(TQ), box(TE), twk, np.ones(len(twk), bool))
    Qm, Em, wkm, Qms, Ems = M['validation']; Qa, Ea, wka, Qas, Eas = M['private']
    p_mar = prior_from(Qms, Ems, wkm, np.ones(len(wkm), bool))
    l1 = llr(Qa, Ea, wka, p_mar, p_tr)
    for d in range(len(wka)):
        keep = np.ones(len(wka), bool); keep[d] = False
        p_aloo = prior_from(Qas, Eas, wka, keep)
        l2 = llr(Qa[d:d + 1], Ea[d:d + 1], wka[d:d + 1], p_aloo, p_tr)[0]
        l3 = llr(Qa[d:d + 1], Ea[d:d + 1], wka[d:d + 1], p_mar, p_aloo)[0]
        rows.append(dict(panel=panel, day=d, apr_march_vs_train=l1[d], apr_aprloo_vs_train=l2, apr_march_vs_aprloo=l3))
    # reference: March LOO vs train (as exp_scenario_llr) and March day under April prior vs train
    p_apr = prior_from(Qas, Eas, wka, np.ones(len(wka), bool))
    l4 = llr(Qm, Em, wkm, p_apr, p_tr)
    for d in range(len(wkm)):
        rows.append(dict(panel=panel, day=100 + d, mar_april_vs_train=l4[d]))
df = pd.DataFrame(rows); pd.set_option('display.width', 250)
cols = ['apr_march_vs_train', 'apr_aprloo_vs_train', 'apr_march_vs_aprloo', 'mar_april_vs_train']
print('per panel medians'); print(df.groupby('panel')[cols].median().round(1).to_string())
print('share > 0 (single panel-days):', {c: round(float((df[c].dropna() > 0).mean()), 3) for c in cols})
pooled = df.groupby('day')[cols].sum(min_count=1)
print('pooled over panels, median per day:', pooled.median().round(1).to_dict())
print('pooled share > 0:', {c: round(float((pooled[c].dropna() > 0).mean()), 3) for c in cols})
