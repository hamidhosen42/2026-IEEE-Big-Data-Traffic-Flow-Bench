"""verify_pvval_scen_ref.py -- light reference for verify_pvval_scen_llr.py inside the TRAIN scenario (analysis only, no model):
for each train month m (Jul..Feb) and its days: LLR(prior of month m-1 only vs prior of all train days outside m) and
LLR(month-m leave-one-day-out prior vs all train days outside m).  Optional i.i.d. mask at visibility VIS on every train cell."""
import sys
import numpy as np, pandas as pd
import lab
EPS = 0.01; S0, S1 = 60, 252
VIS = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
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
rng = np.random.default_rng(0)
for panel in lab.PANELS:
    z = np.load(f'{lab.DATA}/t2data_{panel}.npz'); cut = 0.6 * z['vf'].astype(float)
    TV = z['train_V'].astype(np.float32)
    if VIS < 1:
        TV = np.where(rng.random(TV.shape, dtype=np.float32) < VIS, TV, np.nan)
    TE = np.isfinite(TV); TQ = TE & (TV <= cut[None, None, :]); del TV
    days = pd.to_datetime(pd.Series([str(x) for x in z['train_days']]))
    wk = days.dt.dayofweek.to_numpy() >= 5; ym = (days.dt.year * 100 + days.dt.month).to_numpy()
    Qs, Es = box(TQ), box(TE); months = sorted(set(ym))
    for i in range(1, len(months)):
        m = ym == months[i]; prev = ym == months[i - 1]
        p_prev = prior_from(Qs, Es, wk, prev); p_out = prior_from(Qs, Es, wk, ~m)
        for d in np.where(m)[0]:
            keep = m.copy(); keep[d] = False
            p_loo = prior_from(Qs, Es, wk, keep)
            rows.append(dict(panel=panel, month=months[i], day=int(d), prev_vs_out=llr1(TQ[d], TE[d], int(wk[d]), p_prev, p_out),
                             loo_vs_out=llr1(TQ[d], TE[d], int(wk[d]), p_loo, p_out), prev_vs_loo=llr1(TQ[d], TE[d], int(wk[d]), p_prev, p_loo)))
df = pd.DataFrame(rows); cols = ['prev_vs_out', 'loo_vs_out', 'prev_vs_loo']
pooled = df.groupby('day')[cols].sum()
print(f'VIS {VIS}: single panel-day share > 0', {c: round(float((df[c] > 0).mean()), 3) for c in cols})
print('pooled over panels: median', pooled.median().round(1).to_dict(), '| share > 0', {c: round(float((pooled[c] > 0).mean()), 3) for c in cols})
print('pooled by month (median):'); print(pooled.join(df.groupby('day').month.first()).groupby('month')[cols].median().round(0).to_string())
