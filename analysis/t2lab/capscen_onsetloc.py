"""capscen_onsetloc.py -- diagnostic: where do new-scenario onsets happen, and what data <= T could locate them?
Onset event = first queued observation on the corridor after >= 12 slots with no queued observed cell (06-21 h).
For the onset links (queued at the event slot) report: train prior at (link +-1, slot +-6) >= 0.05 ('train-recurrent'),
same-day earlier queue at link +-1 before t-12 ('same-day'), previous same-split days queue at link +-1 within +-1 h ('prev-days', <= 4 days)."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab
rng = np.random.default_rng(0); rows = []
def events(Q, E):
    out = []
    D, T, L = Q.shape
    for d in range(D):
        anyq = Q[d].any(1); last = -99
        for t in range(72, 252):
            if anyq[t] and not anyq[max(0, t - 12):t].any() and E[d, t - 12:t].mean() > 0.3:
                out.append((d, t, np.where(Q[d, t])[0]))
    return out
for panel in lab.PANELS:
    z = np.load(f'{lab.DATA}/t2data_{panel}.npz'); cut = 0.6 * z['vf'].astype(float)
    TV = z['train_V'].astype(np.float32); TE = np.isfinite(TV); TQ = TE & (TV <= cut)
    prior = TQ.sum(0) / np.maximum(TE.sum(0), 1)
    TVm = np.where(rng.random(TV.shape, dtype=np.float32) < 0.65, TV, np.nan); TEm = np.isfinite(TVm); TQm = TEm & (TVm <= cut)
    td = np.array([str(x)[:7] for x in z['train_days']])
    sets = [('train', TQm, TEm, td)]
    for sp in ('validation', 'private'):
        days = np.array([str(x) for x in z[f'{sp}_days']]); o = np.argsort(days); V = z[f'{sp}_Vm'][o].astype(np.float32)
        E = np.isfinite(V); sets.append((sp, E & (V <= cut), E, np.array([sp] * len(o))))
    for tag, Q, E, mon in sets:
        for d, t, ls in events(Q, E):
            L = Q.shape[2]; nb = np.unique(np.clip(np.concatenate([ls - 1, ls, ls + 1]), 0, L - 1))
            rec = prior[max(0, t - 6):t + 7][:, nb].max() >= 0.05
            sd = Q[d, 60:max(60, t - 12)][:, nb].any()
            same_month = np.where(mon == mon[d])[0]; prev = [x for x in same_month if d - 4 <= x < d]
            pv = Q[prev][:, max(0, t - 12):t + 13][:, :, nb].any() if prev else np.nan
            rows.append(dict(panel=panel, split=tag, d=d, t=t, early=(d - same_month.min()) < 5, rec=rec, sameday=sd, prevdays=pv))
    print(panel, flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'capscen', 'onsetloc.csv'), index=False)
pd.set_option('display.width', 200)
print(df.groupby('split')[['rec', 'sameday', 'prevdays']].mean().round(3))
print(df.groupby(['split', 'panel'])[['rec', 'sameday', 'prevdays']].mean().unstack(0).round(2))
x = df[df.split != 'train']
print('new scenario, onset NOT train-recurrent: share', round(1 - x.rec.mean(), 3), '| of those: same-day earlier queue', round(x[~x.rec].sameday.mean(), 3),
      '| prev<=4 days', round(x[~x.rec].prevdays.mean(), 3), '| early-month (days 0-4) same-day', round(x[~x.rec & x.early].sameday.mean(), 3))
y = df[df.split == 'train']; print('train, not recurrent share', round(1 - y.rec.mean(), 3), '| same-day', round(y[~y.rec].sameday.mean(), 3), '| recurrent same-day', round(y[y.rec].sameday.mean(), 3))
