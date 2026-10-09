"""Validate the footprint-matched pseudo-March forecast (FM-PM) on every ongoing LB probe with a pseudo-March per-window lab pair.
Official side: decisions of the two submitted queue files on the 40 validation ongoing windows (label-free).
Lab side: pseudo-March per-window paired deltas (offlike_m + runstart_m) of the corresponding lab arms."""
import sys
sys.path.insert(0, '/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench/analysis/t2lab')
import numpy as np, pandas as pd
import lab
DL, OUT = sys.argv[1], sys.argv[2]
R = f'{lab.ROOT}/analysis/t2lab/results'
WI = pd.concat([lab.window_index(p, 'validation') for p in lab.PANELS], ignore_index=True)
WI = WI[WI.condition == 'queue_ongoing'].copy()
org = pd.to_datetime(WI.forecast_origin, utc=True); WI['tod'] = (org.dt.hour + org.dt.minute / 60).to_numpy()
ts = (org.astype('int64') // 10 ** 9 // 300)
WIall = pd.concat([lab.window_index(p, 'validation') for p in lab.PANELS], ignore_index=True)
oa = pd.to_datetime(WIall.forecast_origin, utc=True); WIall['g'] = (oa.astype('int64') // 10 ** 9 // 300).to_numpy()
g72 = {}
for p, g in WIall.groupby('panel'):
    gg = g.sort_values('g'); gap = np.r_[10 ** 9, np.diff(gg.g.to_numpy())]
    g72.update(dict(zip(gg.window_id, gap == 72)))
WI['g72'] = WI.window_id.map(g72)


def official(fa, fb, only_g72=False):
    A = pd.read_csv(fa); B = pd.read_csv(fb)
    A = A[A.window_id.isin(WI.window_id)]; B = B[B.window_id.isin(WI.window_id)]
    m = A.merge(B, on=['window_id', 'timestamp', 'link_id'], suffixes=('_a', '_b'))
    assert len(m) == len(A) == len(B)
    if only_g72:                                    # the swap applied B only on g72 windows
        ng = ~m.window_id.map(g72)
        m.loc[ng, 'queue_pred_b'] = m.loc[ng, 'queue_pred_a']
    g = m.groupby('window_id').agg(nA=('queue_pred_a', 'sum'), nB=('queue_pred_b', 'sum'),
                                   changed=('queue_pred_a', lambda s: 0))
    g['changed'] = m.assign(c=m.queue_pred_a != m.queue_pred_b).groupby('window_id').c.sum()
    g = g.join(WI.set_index('window_id')[['panel', 'tod', 'g72']])
    g['dn'] = g.nB - g.nA
    return g.reset_index()


def labpair(f, a, b, sets=('offlike_m', 'runstart_m'), g72col=None):
    z = pd.read_pickle(f); z = z[z.set.isin(sets)]
    key = ['set', 'panel', 'd', 't']
    m = z[z.arm == a].merge(z[z.arm == b][key + ['iou', 'n_pred']], on=key, suffixes=('_a', '_b'))
    m['dI'] = m.iou_b - m.iou_a; m['dn'] = m.n_pred_b - m.n_pred_a; m['tod'] = m.t / 12
    return m


def forecast(L, off, tod_w=2.0, panel_exact=True, min_n=5):
    Lc = L[(L.dn != 0) | (L.dI != 0)]; pred = []
    for _, r in off.iterrows():
        if r.changed == 0:
            pred.append(0.0); continue
        tol = max(3, 0.5 * abs(r.dn)); sg = np.sign(r.dn)
        m = (np.abs(Lc.dn - r.dn) <= tol) & (np.sign(Lc.dn) == sg) & (np.abs(Lc.tod - r.tod) <= tod_w)
        c = Lc[m & (Lc.panel == r.panel)] if panel_exact else Lc[m]
        if len(c) < min_n:
            c = Lc[m]
        if len(c) < min_n:
            c = Lc[(np.abs(Lc.dn - r.dn) <= tol) & (np.sign(Lc.dn) == sg)]
        pred.append(c.dI.mean() if len(c) else 0.0)
    return np.array(pred)


def agg_lab(L):
    return np.mean([g.dI.mean() for _, g in L.groupby('panel')])


PROBES = [
    ('F03 fs2 (E3)', f'{DL}/tfb-t2v20/queue_submission_og8.csv', f'{DL}/tfb-t2v21/queue_submission_fs2.csv', False,
     f'{R}/firstcross/pmarch_np255_A_F03_G.pkl', 'A', 'F03', +0.00474),
    ('allfin fs4af (F5)', f'{DL}/tfb-t2v22/queue_submission_fs4.csv', f'{DL}/tfb-t2v22af/queue_submission_fs4af.csv', False,
     f'{R}/allfin/pmarch/pmarch_og_on.pkl', 'elig_npid', 'all_npid', +0.00090),
    ('occ fs4occ vs fs4af (G3-F5 approx)', f'{DL}/tfb-t2v22af/queue_submission_fs4af.csv', f'{DL}/tfb-t2occ/queue_submission_fs4occ.csv', False,
     f'{R}/occ/pmarch/pmarch_og4.pkl', 'af_bag4', 'occ_bag4', +0.00012),
    ('g72 switch sw03 (F4)', f'{DL}/tfb-t2v22/queue_submission_fs4.csv', f'{DL}/tfb-t2v24/queue_submission_sw03.csv', True,
     f'{R}/ogmodel_g72_pmarch_switch.pkl', 'F03', 'sw_F03|F10', -0.00022),
    ('ogfeat all2 fs4ogf (H4)', f'{DL}/tfb-t2occ/queue_submission_fs4occ.csv', f'{DL}/tfb-t2ogfc/queue_submission_fs4ogf.csv', False,
     f'{OUT}/ogf_pm_frame.pkl', 'og_np_base', 'og_np_all2', -0.00121),
]
# ogfeat pm frame from noise40 output
a = pd.read_pickle(f'{OUT}/ogf_perwindow_lab.pkl'); a = a[a.src == 'pmarch']
fr = pd.concat([a.assign(arm='og_np_base', n_pred=a.n_pred_a, iou=a.iou_a), a.assign(arm='og_np_all2', n_pred=a.n_pred_b, iou=a.iou_b)])
fr[['arm', 'set', 'panel', 'd', 't', 'iou', 'n_pred']].to_pickle(f'{OUT}/ogf_pm_frame.pkl')
rows = []
for name, fa, fb, og72, lf, la, lb, lbd in PROBES:
    off = official(fa, fb, og72)
    L = labpair(lf, la, lb, sets=('offlike_m',) if 'g72' in name else ('offlike_m', 'runstart_m'))
    Lo = L[L.set == 'offlike_m']
    fpan = forecast(L, off); fpool = forecast(L, off, panel_exact=False)
    fo = forecast(Lo, off)
    rows.append(dict(probe=name, lb=lbd, lb_iou=lbd / 0.15, win_changed=int((off.changed > 0).sum()), cells_changed=int(off.changed.sum()),
                     pos_A=int(off.nA.sum()), pos_B=int(off.nB.sum()),
                     pm_offlike_m=agg_lab(Lo), pm_runstart_m=agg_lab(L[L.set == 'runstart_m']) if (L.set == 'runstart_m').any() else np.nan,
                     fm_pm_panel=fpan.mean(), fm_pm_pool=fpool.mean(), fm_pm_offlike_only=fo.mean(),
                     fm_lb=0.15 * fpan.mean()))
    off.assign(fm=fpan).to_csv(f'{OUT}/fm_{name.split()[0]}.csv', index=False)
pd.set_option('display.width', 250)
D = pd.DataFrame(rows)
print(D.round(5).to_string(index=False))
D.to_csv(f'{OUT}/fm_validation.csv', index=False)
