"""PSEUDO-MARCH GATE for selection-faithful ongoing training (new-seed check; steers the PRIVATE file / motivates a public
LB probe only - it must not be used to tune the public file directly).

Models: trained on the TRAIN split only (all 273 days; results/firstcross/final/model_<ARM>_*.txt from exp_firstcross_final.py).
Windows: pseudo windows selected inside the released validation month (masked layer) with the same faithful rule as the
lab (coverage from the eligibility cube, established queue from visible history observations, persistence cap on the
time-filled visible state), away from the official windows: every window whose [T-60, T+30] touches a blanked slot
(official horizon + buffer) is dropped, so no official window's hidden horizon is involved.
Features: production window_features on the masked-layer history [T-60, T) (sparser than the released raw window_history:
this is a known mismatch that hits every arm alike), the masked slot-T value exactly as production, ramp flows < T.
Labels: visible masked-layer cells of T+5..T+30 (speed <= 0.6 v_free).  The private month is not touched.
Usage: exp_firstcross_pmarch.py <ARM[,ARM...]> [cols=np] [leaves=255] [n_trees=500]     (first arm = base)"""
import gc, json, os, sys, time, types
import numpy as np
import pandas as pd
import lightgbm as lgb
import lab
import exp_firstcross_lib as fc
from exp_firstcross_prod_snippet import fill_time36

arms = sys.argv[1].split(',')
cols = sys.argv[2] if len(sys.argv) > 2 else 'np'
leaves = int(sys.argv[3]) if len(sys.argv) > 3 else 255
n_trees = int(sys.argv[4]) if len(sys.argv) > 4 else 500
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 60)
H, F = lab.H, lab.F
cols_ = lab.colset(cols)
MD = f'{fc.RES}/final'
models = {a: lgb.Booster(model_file=f'{MD}/model_{a}_{cols}{leaves}_{n_trees}.txt') for a in arms}


def val_panel(panel):
    P = lab.load_panel(panel)
    z = np.load(P['npz'], allow_pickle=False)
    days = np.array([str(x) for x in z['validation_days']]); o = np.argsort(days)
    V = z['validation_Vm'][o].astype(np.float32); Fq = z['validation_Fm'][o].astype(np.float32); Em = z['validation_Em'][o]
    E = np.isfinite(V); Q = E & (V <= P['cut'][None, None, :])
    Pv = dict(panel=panel, links=P['links'], li=P['li'], cut=P['cut'], vf=P['vf'], cap=P['cap'], lanes=P['lanes'], length=P['length'],
              days=list(days[o]), dow=pd.to_datetime(pd.Series(days[o])).dt.dayofweek.to_numpy(), V=V, Fq=Fq, E=E, Q=Q, Em=Em,
              ON=z['validation_ON'][o].astype(np.float32), OFF=z['validation_OFF'][o].astype(np.float32))
    return P, Pv


def select(Pv):
    """faithful ongoing candidates / production-rule onset candidates on the masked validation month; blank-touching windows dropped."""
    V, E, Q, Em = Pv['V'], Pv['E'], Pv['Q'], Pv['Em']; D, _, L = V.shape
    VF = fill_time36(V); MF = np.isfinite(VF); QF = MF & (VF <= Pv['cut'][None, None, :])
    blank = (Em.sum(2) >= 5) & (E.sum(2) == 0)
    og = np.zeros((D, 288), bool); on = og.copy(); pers = np.full((D, 288), np.nan); stable = og.copy()
    for d in range(D):
        for t in range(H, 288 - F):
            if blank[d, t - H:t + F + 1].any() or Em[d, t - H:t].mean() < 0.7:
                continue
            if lab.established(Q[d, t - H:t]):
                Tf = QF[d, t + 1:t + 1 + F]; Mf = MF[d, t + 1:t + 1 + F]
                if not Tf.any():
                    continue
                S = np.repeat(QF[d, t][None], F, 0)
                u = ((Tf | S) & Mf).sum(); i = (Tf & S & Mf).sum()
                pers[d, t] = 1.0 if u == 0 else i / u
                og[d, t] = pers[d, t] <= 0.9; stable[d, t] = pers[d, t] > 0.9
            else:
                on[d, t] = Q[d, t + 1:t + 1 + F].any()
    return on, og, stable, pers, blank


def val_rows(P, Pv, wins, prior):
    d = np.array([w[0] for w in wins]); t = np.array([w[1] for w in wins])
    idx = t[:, None] + np.arange(-H, 0)[None]
    Vh = Pv['V'][d[:, None], idx]; Fh = Pv['Fq'][d[:, None], idx]; VT = Pv['V'][d, t]
    import warnings
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        Fe, qlast = lab.window_features(P, Vh, Fh, t, Pv['dow'][d], prior, VT)
        lab.add_ramp(Fe, Pv['ON'], Pv['OFF'], d, t, P['cap'])
    W, L = qlast.shape
    R = dict(d=d, t=t, cond=np.array([lab.ONGOING] * W), qlast=qlast, W=W, L=L, rows=[])
    for h in range(1, F + 1):
        feats = dict(Fe); feats['h'] = np.full((W, L), h, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = lab.prior_feats(prior, Pv['dow'][d], t, h)
        R['rows'].append((h, feats, Pv['Q'][d, np.minimum(t + h, 287)], Pv['E'][d, np.minimum(t + h, 287)]))
    return R


res = []; info = []
for panel in lab.PANELS:
    t1 = time.time()
    P, Pv = val_panel(panel)
    on, og, stable, pers, blank = select(Pv)
    D = len(Pv['days'])
    g_on, g_og = lab._glist(on), lab._glist(og); s = set()
    for d0 in range(D):
        _, b = lab.greedy(g_on, g_og, 'interleaved', start_day=d0); s.update(b)
    dd, tt = np.where(og); rs = ~og[dd, tt - 1]
    sets = {'offlike_m': sorted(s), 'runstart_m': sorted(zip(dd[rs].tolist(), tt[rs].tolist())),
            'faith3_m': sorted((a, b) for a, b in zip(dd.tolist(), tt.tolist()) if b % 3 == 0)}
    ds, ts_ = np.where(stable); k3 = ts_ % 3 == 0
    sets['stable3_m'] = sorted(zip(ds[k3].tolist(), ts_[k3].tolist()))
    vis_day = Pv['E'].sum((1, 2)) / np.maximum(Pv['Em'].sum((1, 2)), 1)
    info.append(dict(panel=panel, days=D, blank_slots=int(blank.sum()), og_cand=int(og.sum()), on_cand=int(on.sum()), stable_cand=int(stable.sum()),
                     **{k: len(v) for k, v in sets.items()}, visible_share=float(Pv['E'].sum() / Pv['Em'].sum())))
    prior = lab.prior_table(P, np.ones(len(P['days']), bool))
    for sname, wl in sets.items():
        if not wl:
            continue
        R = val_rows(P, Pv, [(a, b, lab.ONGOING) for a, b in wl], prior)
        X, y, e, meta, call = lab.to_matrix(R, lab.ONGOING, None, lab.ongoing_keep(R))
        ix = [list(call).index(c) for c in cols_]
        d_, t_ = R['d'], R['t']
        win = pd.DataFrame(dict(d=d_, t=t_, cond=lab.ONGOING, fold=0, pers_fill=pers[d_, t_], runstart=og[d_, t_] & ~og[d_, t_ - 1], faithful=og[d_, t_],
                                vis=vis_day[d_], hist_vis=[float(Pv['E'][a, b - H:b].sum() / max(Pv['Em'][a, b - H:b].sum(), 1)) for a, b in zip(d_, t_)]))
        rc = types.SimpleNamespace(truth=np.stack([Pv['Q'][d_, t_ + h] for h in range(1, F + 1)], 1), ev=np.stack([Pv['E'][d_, t_ + h] for h in range(1, F + 1)], 1),
                                   qlast=R['qlast'], win=win, name=sname, panel=panel, cond=lab.ONGOING, L=R['L'])
        P6s = {}
        for a in arms:
            P6 = np.zeros((R['W'], F, R['L']), np.float32)
            P6[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = models[a].predict(X[:, ix], num_threads=lab.THREADS)
            P6s[a] = P6
        P6s['persist'] = np.repeat(R['qlast'][:, None, :], F, 1).astype(np.float32) + 1e-6
        if len(arms) > 1:
            P6s[f'blend50_{arms[0]}_{arms[1]}'] = 0.5 * (P6s[arms[0]] + P6s[arms[1]])
        for a, P6 in P6s.items():
            df = lab.score_probs(rc, P6, arm=a, set_name=sname)
            df['vis'] = win.vis.to_numpy(); df['hist_vis'] = win.hist_vis.to_numpy()
            res.append(df)
        del X; gc.collect()
    lab.log(panel, json.dumps(info[-1]), f'{time.time() - t1:.0f}s')
    lab.drop_panel(panel)
R_ = pd.concat(res, ignore_index=True)
tag = f'{cols}{leaves}_{"_".join(arms)}'
R_.to_pickle(f'{fc.RES}/pmarch_{tag}.pkl'); pd.DataFrame(info).to_csv(f'{fc.RES}/pmarch_info_{tag}.csv', index=False)
print(pd.DataFrame(info).to_string(index=False))
allarms = list(dict.fromkeys(R_.arm)); base = arms[0]
print('\n== PSEUDO-MARCH: aggregated IoU (window -> panel -> family -> mean) ==')
rows = []
for sname in ('offlike_m', 'runstart_m', 'faith3_m', 'stable3_m'):
    for a in allarms:
        g = R_[(R_.set == sname) & (R_.arm == a)]
        if not len(g):
            continue
        es = lab.error_split(g)
        rows.append(dict(set=sname, arm=a, n=len(g), iou=lab.aggregate(g), plain=g.iou.mean(), **{f'h{h + 1}': v for h, v in enumerate(lab.step_profile(g))},
                         missed_new=es['missed_new'], kept_cleared=es['kept_cleared'], other=es['other']))
S = pd.DataFrame(rows); print(S.round(4).to_string(index=False)); S.to_csv(f'{fc.RES}/pmarch_summary_{tag}.csv', index=False)
print(f'\n== PSEUDO-MARCH: paired deltas vs {base} ==')
rows = []
for sname in ('offlike_m', 'runstart_m', 'faith3_m', 'stable3_m'):
    A = R_[(R_.set == sname) & (R_.arm == base)]
    for a in allarms:
        if a == base:
            continue
        B = R_[(R_.set == sname) & (R_.arm == a)]
        if not len(B):
            continue
        b = fc.boot(A, B); pa = A.groupby('panel').iou.mean(); pb = B.groupby('panel').iou.mean()
        r = dict(set=sname, arm=a, **b, panels_pos=f'{int((pb - pa > 0).sum())}/{len(pa)}')
        for lo, hi, nm in ((0.7, 2, 'vis_hi'), (0.55, 0.7, 'vis_mid'), (0, 0.55, 'vis_lo')):
            m = (A.vis >= lo) & (A.vis < hi)
            if m.any():
                a2, b2 = fc.paired(A[m], B)
                r[nm] = lab.aggregate(b2) - lab.aggregate(a2); r[nm + '_n'] = len(a2)
        rows.append(r)
Dl = pd.DataFrame(rows); print(Dl.round(4).to_string(index=False)); Dl.to_csv(f'{fc.RES}/pmarch_deltas_{tag}.csv', index=False)
if len(arms) > 1:
    A = R_[(R_.set == 'offlike_m') & (R_.arm == base)]; B = R_[(R_.set == 'offlike_m') & (R_.arm == arms[1])]
    a2, b2 = fc.paired(A, B)
    t = pd.DataFrame(dict(n=a2.groupby('panel').size(), base=a2.groupby('panel').iou.mean(), arm=b2.groupby('panel').iou.mean())); t['delta'] = t.arm - t.base
    print(f'\nper panel (offlike_m): {arms[1]} vs {base}'); print(t.round(4).to_string())
lab.log('DONE')
