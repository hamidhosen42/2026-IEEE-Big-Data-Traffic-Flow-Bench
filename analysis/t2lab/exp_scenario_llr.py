"""Can ONE day of observations tell which scenario a month belongs to?  (basis for a per-window scenario gate in the private file:
an April window on day d may use April days < d, i.e. data before T.)
Per day and panel: LLR = sum over visible cells (05:00-21:00) of log P(q | same-scenario prior) - log P(q | train prior), with
P = smoothed (+-2 slots) queue prior by weekday class, clipped to [EPS, 1-EPS].
  March target days : 'same-scenario' prior = the OTHER March days (leave-one-day-out); train prior = all train days  -> expect > 0
  train target days : (fold 0 days) 'March' prior = all March days (wrong scenario); train prior = folds 1-3        -> expect < 0
Also reports the LLR summed over panels and over the first k days (the gate's evidence for windows on day k).  No April data."""
import numpy as np, pandas as pd
import lab

EPS = 0.01
S0, S1 = 60, 252             # 05:00 .. 21:00
pd.set_option('display.width', 220)


def box_counts(Q, E):
    def box(a):
        c = np.cumsum(np.pad(a.astype(np.float32), ((0, 0), (3, 2), (0, 0))), 1)
        return c[:, 5:] - c[:, :-5]
    return box(Q), box(E)


def prior_from(Qs, Es, wk_days, mask):
    """[2, 288, L] smoothed prior by class from days in mask."""
    out = []
    for cls in (False, True):
        m = mask & (wk_days == cls)
        if not m.any():
            m = mask
        out.append(Qs[m].sum(0) / np.maximum(Es[m].sum(0), 1e-9))
    return np.clip(np.stack(out), EPS, 1 - EPS)


def day_llr(Q, E, wk, pA, pB):
    """per day: sum over visible cells in [S0, S1) of log pA(q) - log pB(q)."""
    out = []
    for d in range(Q.shape[0]):
        c = int(wk[d]); q = Q[d, S0:S1]; e = E[d, S0:S1]
        a = np.where(q, np.log(pA[c, S0:S1]), np.log(1 - pA[c, S0:S1]))
        b = np.where(q, np.log(pB[c, S0:S1]), np.log(1 - pB[c, S0:S1]))
        out.append(float(((a - b) * e).sum()))
    return np.array(out)


rows = []
for panel in lab.PANELS:
    P = lab.load_panel(panel)
    z = np.load(P['npz'], allow_pickle=False)
    days = np.array([str(x) for x in z['validation_days']]); o = np.argsort(days)
    Vm = z['validation_Vm'][o].astype(np.float32); Em = np.isfinite(Vm); Qm = Em & (Vm <= P['cut'][None, None, :])
    wkm = pd.to_datetime(pd.Series(days[o])).dt.dayofweek.to_numpy() >= 5
    Qms, Ems = box_counts(Qm, Em)
    Qt, Et = P['Q'], P['E']; wkt = P['dow'] >= 5
    Qts, Ets = box_counts(Qt, Et)
    p_tr_all = prior_from(Qts, Ets, wkt, np.ones(len(wkt), bool))
    # March target days, leave-one-day-out same-scenario prior
    for d in range(len(days)):
        keep = np.ones(len(days), bool); keep[d] = False
        pM = prior_from(Qms, Ems, wkm, keep)
        l = day_llr(Qm[d:d + 1], Em[d:d + 1], wkm[d:d + 1], pM, p_tr_all)[0]
        rows.append(dict(panel=panel, target='march', day=d, llr=l, cells=int(Em[d, S0:S1].sum())))
    # train target days (fold 0), wrong 'March' prior vs fold-safe train prior
    f0 = P['block'] == 0
    p_tr_f = prior_from(Qts, Ets, wkt, ~f0)
    pM_all = prior_from(Qms, Ems, wkm, np.ones(len(days), bool))
    idx = np.where(f0)[0]
    ll = day_llr(Qt[idx], Et[idx], wkt[idx], pM_all, p_tr_f)
    for d, l in zip(idx, ll):
        rows.append(dict(panel=panel, target='train_f0', day=int(d - idx[0]), llr=float(l), cells=int(Et[d, S0:S1].sum())))
    lab.drop_panel(panel)

df = pd.DataFrame(rows)
df.to_csv('results/scenario_llr.csv', index=False)
print('== single-day LLR (same-scenario prior vs train prior), per panel ==')
print(df.groupby(['target', 'panel']).llr.describe()[['count', 'min', '25%', '50%', '75%', 'max']].round(1).to_string())
print('\nshare of days with the correct sign: march (LLR>0)', round(float((df[df.target == 'march'].llr > 0).mean()), 3),
      '| train (LLR<0)', round(float((df[df.target == 'train_f0'].llr < 0).mean()), 3))
# evidence pooled over the 8 panels for one day (the gate can pool panels: all panels share the scenario)
pooled = df.groupby(['target', 'day']).llr.sum().reset_index()
print('\n== LLR pooled over 8 panels, per day ==')
print(pooled.groupby('target').llr.describe()[['count', 'min', '25%', '50%', '75%', 'max']].round(1).to_string())
print('pooled correct sign: march', round(float((pooled[pooled.target == 'march'].llr > 0).mean()), 3),
      '| train', round(float((pooled[pooled.target == 'train_f0'].llr < 0).mean()), 3))
