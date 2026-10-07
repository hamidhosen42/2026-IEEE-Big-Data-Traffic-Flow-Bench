"""verify_pvval_monthdiag.py -- light model-free diagnostic (no training, train split + released March masked layer only;
no April data): does a calendar-month boundary matter for queue-location structure?

Pseudo-March measures a WITHIN-month split (March days < 15 -> days >= 15), while April is a different month "with its own demand
draws and incident schedule" (repo/docs/DATA.md).  If the generator draws demand per month, adjacent 10-day blocks inside one
month should agree more than adjacent blocks that straddle a month boundary.  Weekday queue-frequency maps (visible cells,
slots +-2), correlation of raw maps and of deviations from the all-train map.
Pairs per train month m (days 1-10 | 11-20 of m = WITHIN) and (last 10 days of m-1 | days 1-10 of m = CROSS); March halves for scale."""
import numpy as np
import pandas as pd
import lab


def fmap(Q, E, days_idx, dow):
    m = np.asarray(days_idx); m = m[dow[m] < 5]
    if len(m) < 3:
        return None
    q = Q[m].sum(0).astype(np.float64); e = E[m].sum(0).astype(np.float64)

    def box(a):
        c = np.cumsum(np.pad(a, ((3, 2), (0, 0))), 0)
        return c[5:] - c[:-5]
    q, e = box(q), box(e)
    return np.where(e > 0, q / np.maximum(e, 1e-9), np.nan)


def corr(a, b, base=None):
    if a is None or b is None:
        return np.nan
    if base is not None:
        a = a - base; b = b - base
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


rows = []
for panel in lab.PANELS:
    P = lab.load_panel(panel)
    days = pd.to_datetime(pd.Series(P['days'])); mon = days.dt.to_period('M').astype(str).to_numpy(); dom = days.dt.day.to_numpy()
    dow = P['dow']; Q, E = P['Q'], P['E']
    base = fmap(Q, E, np.arange(len(days)), dow)
    months = list(dict.fromkeys(mon))
    for i, m in enumerate(months):
        idx = np.where(mon == m)[0]
        A = idx[dom[idx] <= 10]; B = idx[(dom[idx] > 10) & (dom[idx] <= 20)]
        fA, fB = fmap(Q, E, A, dow), fmap(Q, E, B, dow)
        rows.append(dict(panel=panel, month=m, kind='within', r=corr(fA, fB), r_dev=corr(fA, fB, base)))
        if i > 0:
            prev = np.where(mon == months[i - 1])[0][-10:]
            fP = fmap(Q, E, prev, dow)
            rows.append(dict(panel=panel, month=m, kind='cross', r=corr(fP, fA), r_dev=corr(fP, fA, base)))
    # March (masked layer, visible cells): halves, and March vs the all-train map
    z = np.load(P['npz'], allow_pickle=False)
    dv = np.array([str(x) for x in z['validation_days']]); o = np.argsort(dv)
    Vm = z['validation_Vm'][o].astype(np.float32)
    with np.errstate(invalid='ignore'):
        Ev = np.isfinite(Vm); Qv = Ev & (Vm <= P['cut'][None, None, :])
    dvw = pd.to_datetime(pd.Series(dv[o])).dt.dayofweek.to_numpy()
    f1, f2 = fmap(Qv, Ev, np.arange(0, 10), dvw), fmap(Qv, Ev, np.arange(10, 20), dvw)
    fm = fmap(Qv, Ev, np.arange(len(dv)), dvw)
    rows.append(dict(panel=panel, month='2031-03', kind='march_within', r=corr(f1, f2), r_dev=corr(f1, f2, base)))
    rows.append(dict(panel=panel, month='2031-03', kind='march_vs_train', r=corr(fm, base), r_dev=np.nan))
    # train 10-day block vs the all-train map, for scale
    for m in months[1:-1]:
        idx = np.where(mon == m)[0]; A = idx[dom[idx] <= 10]
        rows.append(dict(panel=panel, month=m, kind='trainblock_vs_train', r=corr(fmap(Q, E, A, dow), base), r_dev=np.nan))
    lab.drop_panel(panel)

R = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(R.groupby(['kind', 'panel'])[['r', 'r_dev']].mean().unstack(0).round(3).to_string())
print()
print(R.groupby('kind')[['r', 'r_dev']].agg(['mean', 'count']).round(3).to_string())
W = R[R.kind == 'within'].set_index(['panel', 'month']); C = R[R.kind == 'cross'].set_index(['panel', 'month'])
j = W.join(C, lsuffix='_w', rsuffix='_c', how='inner')
d = (j.r_dev_w - j.r_dev_c).dropna(); dr = (j.r_w - j.r_c).dropna()
print(f'\npaired within - cross (same month m): raw r {dr.mean():+.3f} (pos {int((dr > 0).sum())}/{len(dr)}),'
      f' deviation r {d.mean():+.3f} (pos {int((d > 0).sum())}/{len(d)})')
