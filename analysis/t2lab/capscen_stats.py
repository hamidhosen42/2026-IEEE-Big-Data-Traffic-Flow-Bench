"""capscen_stats.py -- H1 test (2026-10-11): does validation/private run at capacity x1.2 relative to train?
Released masked observations only (train subsampled to the March visibility where a statistic depends on the sample size).
Per panel x month (train months 2030-06..2031-02, 2031-03 = validation, 2031-04 = private):
  q99/q999 lane flow / fd capacity per lane (median over links), fitted capacity (upper envelope of the free branch),
  critical density (density at the top 1 % flows), congested branch (w, kjam) fit, breakdown flow (flow just before a
  free->queued transition) / cap, queue discharge flow / cap, queued share by hour, demand (daily vehicle-km), peak timing,
  "unusual queue" rate (queued cells where the train prior <= 0.02).
Output: results/capscen/stats_panel_month.csv, tod_share.csv; printed summary."""
import os, sys, numpy as np, pandas as pd, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab
warnings.filterwarnings('ignore')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'capscen'); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
VIS = 0.50     # March/April finite share; train subsampled to this for sample-size-dependent stats


def month_stats(V, F, cut, vf, cap, lanes, kcrit, kjam, prior, tag, panel, days):
    """V, F: [D, 288, L] float32 with NaN = unobserved (mask applied)."""
    capl = cap / lanes; E = np.isfinite(V) & np.isfinite(F)
    fl = F / lanes[None, None, :]                       # vph per lane
    k = np.where(V > 1, fl / np.maximum(V, 1), np.nan)  # veh/km/lane
    r = dict(panel=panel, month=tag, ndays=V.shape[0], vis=float(E.mean()))
    q99 = []; q999 = []; capfit = []; kc = []; wfit = []; kjfit = []; bkd = []; dis = []; vfit = []
    Q = E & (V <= cut[None, None, :])
    for l in range(V.shape[2]):
        e = E[:, :, l]
        if e.sum() < 500:
            continue
        f = fl[:, :, l][e]; v = V[:, :, l][e]; kk = k[:, :, l][e]
        q99.append(np.quantile(f, 0.99) / capl[l]); q999.append(np.quantile(f, 0.999) / capl[l])
        top = f >= np.quantile(f, 0.99)
        kc.append(np.median(kk[top]) / kcrit[l]); capfit.append(np.median(f[top]) / capl[l])
        free = v > cut[l]
        if free.sum() > 100:
            vfit.append(np.median(v[free & (kk < 0.5 * kcrit[l])]) / vf[l] if (free & (kk < 0.5 * kcrit[l])).sum() > 20 else np.nan)
        cg = (v <= cut[l]) & (kk > kcrit[l])
        if cg.sum() > 30:   # q = w (kjam - k): linear fit f = a + b k, w = -b, kjam = a / w
            b, a = np.polyfit(kk[cg], f[cg], 1)
            if b < 0:
                wfit.append(-b); kjfit.append(a / -b / (kjam[l] / lanes[l]))
        if Q[:, :, l].sum() > 20:
            dis.append(np.median(f[v <= cut[l]]) / capl[l])
        # breakdown: queued at t, observed & free at t-1..t-3
        q = Q[:, 3:, l]; fr = np.ones_like(q)
        for s in (1, 2, 3):
            fr &= E[:, 3 - s:288 - s, l] & ~Q[:, 3 - s:288 - s, l]
        ev = q & fr
        if ev.sum() >= 3:
            fb = fl[:, 2:287, l][ev]   # flow at t-1
            bkd.append((np.median(fb) / capl[l], ev.sum()))
    r.update(q99=np.median(q99), q999=np.median(q999), capfit=np.median(capfit), kcrit_ratio=np.median(kc),
             vfree_ratio=np.nanmedian(vfit), w_kmh=np.median(wfit) if wfit else np.nan, kjam_ratio=np.median(kjfit) if kjfit else np.nan,
             discharge=np.median(dis) if dis else np.nan)
    if bkd:
        b = np.array(bkd); r.update(breakdown=float(np.average(b[:, 0], weights=b[:, 1])), n_breakdown=int(b[:, 1].sum()))
    # queued share, demand, peaks
    r['qshare'] = float(Q.sum() / max(E.sum(), 1))
    tod = Q.sum((0, 2)) / np.maximum(E.sum((0, 2)), 1)
    r['qshare_am'] = float(Q[:, 60:120].sum() / max(E[:, 60:120].sum(), 1)); r['qshare_pm'] = float(Q[:, 168:228].sum() / max(E[:, 168:228].sum(), 1))
    mf = np.nanmean(fl / capl[None, None, :], axis=(0, 2))   # mean v/c by slot
    r['vc_mean'] = float(np.nanmean(mf)); r['vc_am'] = float(np.nanmean(mf[72:108])); r['vc_pm'] = float(np.nanmean(mf[180:216]))
    r['peak_am_slot'] = int(48 + np.nanargmax(mf[48:144])); r['peak_pm_slot'] = int(144 + np.nanargmax(mf[144:240]))
    qt = Q.sum((0, 2)).astype(float)
    r['qpeak_am_slot'] = int(48 + np.argmax(qt[48:144])); r['qpeak_pm_slot'] = int(144 + np.argmax(qt[144:240]))
    # unusual queues (prior <= 0.02 at that weekday class/slot/link)
    wk = (pd.to_datetime(pd.Series(days)).dt.dayofweek.to_numpy() >= 5).astype(int)
    pr = prior[wk]                                           # [D, 288, L]
    un = Q & (pr <= 0.02)
    r['unusual_per_day'] = float(un.sum() / V.shape[0] / max(E.mean(), 1e-6) / 1.0)
    # incident-like: days with a queued run on a link whose prior <= 0.02 lasting >= 6 slots
    inc = 0
    for d in range(V.shape[0]):
        u = un[d]
        run = np.zeros(u.shape[1], int); best = 0
        for t in range(u.shape[0]):
            run = np.where(u[t], run + 1, np.where(E[d, t], 0, run)); best = max(best, run.max())
        inc += best >= 6
    r['incident_days'] = inc / V.shape[0]
    return r, tod


rows = []; tods = []
for panel in lab.PANELS:
    z = np.load(f'{lab.DATA}/t2data_{panel}.npz')
    vf = z['vf'].astype(float); cut = 0.6 * vf; cap = z['cap'].astype(float); lanes = z['lanes'].astype(float)
    kcrit = z['fd_kcrit'].astype(float); kjam = z['fd_kjam'].astype(float)
    TV = z['train_V'].astype(np.float32); TF = z['train_F'].astype(np.float32)
    TE = np.isfinite(TV); TQ = TE & (TV <= cut)
    tdays = np.array([str(x) for x in z['train_days']]); twk = (pd.to_datetime(pd.Series(tdays)).dt.dayofweek.to_numpy() >= 5)
    prior = np.stack([TQ[twk == c].sum(0) / np.maximum(TE[twk == c].sum(0), 1) for c in (0, 1)])
    keep = rng.random(TV.shape, dtype=np.float32) < (VIS / max(TE.mean(), 1e-6))
    TVm = np.where(keep, TV, np.nan); TFm = np.where(keep & np.isfinite(TF), TF, np.nan)
    ym = np.array([d[:7] for d in tdays])
    for m in sorted(set(ym)):
        ix = ym == m
        r, tod = month_stats(TVm[ix], TFm[ix], cut, vf, cap, lanes, kcrit, kjam, prior, m, panel, tdays[ix])
        rows.append(r); tods.append(dict(panel=panel, month=m, **{f'h{h:02d}': tod[h * 12:(h + 1) * 12].mean() for h in range(24)}))
    del TV, TF, TVm, TFm
    for sp in ('validation', 'private'):
        days = np.array([str(x) for x in z[f'{sp}_days']]); o = np.argsort(days)
        V = z[f'{sp}_Vm'][o].astype(np.float32); Fm = z[f'{sp}_Fm'][o].astype(np.float32)
        Fm = np.where(np.isfinite(V), Fm, np.nan)
        r, tod = month_stats(V, Fm, cut, vf, cap, lanes, kcrit, kjam, prior, days[o][0][:7], panel, days[o])
        rows.append(r); tods.append(dict(panel=panel, month=days[o][0][:7], **{f'h{h:02d}': tod[h * 12:(h + 1) * 12].mean() for h in range(24)}))
    print(panel, 'done', flush=True)
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/stats_panel_month.csv', index=False)
pd.DataFrame(tods).to_csv(f'{OUT}/tod_share.csv', index=False)
df['grp'] = np.where(df.month == '2031-03', 'MAR', np.where(df.month == '2031-04', 'APR', 'train'))
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
cols = ['q99', 'q999', 'capfit', 'kcrit_ratio', 'vfree_ratio', 'w_kmh', 'kjam_ratio', 'discharge', 'breakdown', 'n_breakdown', 'qshare',
        'qshare_am', 'qshare_pm', 'vc_mean', 'vc_am', 'vc_pm', 'peak_am_slot', 'peak_pm_slot', 'qpeak_am_slot', 'qpeak_pm_slot',
        'unusual_per_day', 'incident_days']
print('\n== per panel: train months (min / median / max) vs MAR vs APR ==')
for c in cols:
    t = df[df.grp == 'train'].groupby('panel')[c].agg(['min', 'median', 'max'])
    t['MAR'] = df[df.grp == 'MAR'].set_index('panel')[c]; t['APR'] = df[df.grp == 'APR'].set_index('panel')[c]
    t['MAR/med'] = t.MAR / t['median']; t['APR/med'] = t.APR / t['median']
    print(f'\n-- {c}'); print(t.round(4).to_string())
