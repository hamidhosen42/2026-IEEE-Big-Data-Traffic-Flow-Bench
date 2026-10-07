"""verify_pvval_scen.py -- light: does April look like March (same scenario) rather than like train?  Weekday queue-frequency maps
(slots 05:00-21:00 x links, visible cells only) from equal-size day sets; Pearson r between maps.  Analysis only (no model)."""
import numpy as np, pandas as pd
import lab
S0, S1 = 60, 252
def qmap(V, E, cut, dmask):
    Q = E & (V <= cut[None, None, :])
    q = Q[dmask][:, S0:S1].sum(0).astype(float); e = E[dmask][:, S0:S1].sum(0).astype(float)
    return np.where(e > 0, q / np.maximum(e, 1), np.nan), e
def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b); return np.corrcoef(a[m], b[m])[0, 1]
rows = []
for panel in lab.PANELS:
    z = np.load(f'{lab.DATA}/t2data_{panel}.npz'); cut = 0.6 * z['vf'].astype(float)
    out = {}
    for sp in ('validation', 'private'):
        days = np.array([str(x) for x in z[f'{sp}_days']]); o = np.argsort(days); days = days[o]
        V = z[f'{sp}_Vm'][o].astype(np.float32); E = np.isfinite(V)
        wd = pd.to_datetime(pd.Series(days)).dt.dayofweek.to_numpy() < 5
        out[sp] = (V, E, wd, days)
    tdays = np.array([str(x) for x in z['train_days']]); TV = z['train_V'].astype(np.float32); TE = np.isfinite(TV)
    twd = pd.to_datetime(pd.Series(tdays)).dt.dayofweek.to_numpy() < 5
    mon = pd.to_datetime(pd.Series(tdays)).dt.month.to_numpy()
    Vm, Em, wm, _ = out['validation']; Va, Ea, wa, _ = out['private']
    mar, _ = qmap(Vm, Em, cut, wm); apr, _ = qmap(Va, Ea, cut, wa)
    feb, _ = qmap(TV, TE, cut, twd & (mon == 2)); jan, _ = qmap(TV, TE, cut, twd & (mon == 1)); dec, _ = qmap(TV, TE, cut, twd & (mon == 12))
    h1 = wm & (np.arange(len(wm)) < 15); h2 = wm & (np.arange(len(wm)) >= 15)
    m1, _ = qmap(Vm, Em, cut, h1); m2, _ = qmap(Vm, Em, cut, h2)
    rows.append(dict(panel=panel, mar_apr=corr(mar, apr), mar_feb=corr(mar, feb), apr_feb=corr(apr, feb), jan_feb=corr(jan, feb),
                     dec_jan=corr(dec, jan), mar1_mar2=corr(m1, m2), mar1_apr=corr(m1, apr), mar1_feb=corr(m1, feb),
                     qrate_tr=float(np.nanmean(feb)), qrate_mar=float(np.nanmean(mar)), qrate_apr=float(np.nanmean(apr))))
T = pd.DataFrame(rows); pd.set_option('display.width', 250)
print(T.round(3).to_string(index=False)); print('mean', T.drop(columns='panel').mean().round(3).to_dict())
