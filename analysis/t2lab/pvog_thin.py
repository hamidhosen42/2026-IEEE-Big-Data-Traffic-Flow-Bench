"""pvog (light: prediction only): controlled visibility dose-response on the pseudo-March windows (days >= 15).
The SAME windows, labels and slot-T values are kept; only the 60-min HISTORY is thinned: every visible history cell is kept
with probability f (f = 1 as released, 0.75, 0.5), so the effective history visibility is regime x f (0.8 .. 0.25 of the
eligible cells).  Deployment (official windows) has history visibility 1.0 with slot T masked as here.  If an arm's gain over
the production-like arm grows as the history gets sparser, that part of the gain is masking adaptation and must be
extrapolated DOWN for the raw regime.  Members: train-only final models (R_*), and the March-trained lab models of exp 1
(RB_*, days < 15) / exp 2 (M_*, MB_*) when their model files exist.  Evaluation labels: visible cells (unchanged)."""
import os, sys, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import lab
import pvog_lib as pv

CUT = 15; FS = (1.0, 0.75, 0.5)
H, F = lab.H, lab.F
M = {'R_np': ('np', f'{pv.MD_FINAL}/model_F03_np255_500.txt', False), 'R_id': ('id', f'{pv.MD_FINAL}/model_F03_id255_500.txt', False),
     'R_full': ('full', f'{pv.MD_FINAL}/model_F03_full255_500.txt', True)}
for pre, tag in (('RB', 'R_15_5'), ('M', 'M_15_5'), ('MB', 'M_15_5'), ('RB10', 'R_15_10'), ('RB1', 'R_15_1')):
    for cs in ('np', 'id', 'full'):
        f = f'{pv.RES}/models/model_{tag}_{cs}{"B" if pre.endswith("B") or pre.startswith("RB") else ""}.txt'
        if os.path.exists(f):
            M[f'{pre}_{cs}'] = (cs, f, True)
M = {k: (cs, lgb.Booster(model_file=f), bl) for k, (cs, f, bl) in M.items()}
print('members:', sorted(M), flush=True)
ARMS = {'a R:np+id': {'R_np': 0.5, 'R_id': 0.5}, 'b R:npid+full_s4': {'R_np': 0.25, 'R_id': 0.25, 'R_full': 0.5}}
for pre in ('RB', 'M', 'MB', 'RB10', 'RB1'):
    if all(f'{pre}_{cs}' in M for cs in ('np', 'id', 'full')):
        ARMS[f'{pre}:np+id'] = {f'{pre}_np': 0.5, f'{pre}_id': 0.5}
        ARMS[f'{pre}:npid+full'] = {f'{pre}_np': 0.25, f'{pre}_id': 0.25, f'{pre}_full': 0.5}
res = []
for panel in lab.PANELS:
    P, Pv = pv.val_panel(panel)
    on, og = pv.select(Pv)
    D = len(Pv['days'])
    prior_tr = lab.prior_table(P, np.ones(len(P['days']), bool))
    cQ, cE = pv.cum_tables(Pv)
    pday, rday = pv.day_regime(Pv)
    dd, tt = np.where(og); rs = ~og[dd, tt - 1]
    g_on, g_og = lab._glist(on), lab._glist(og); s = set()
    for d0 in range(D):
        _, b = lab.greedy(g_on, g_og, 'interleaved', start_day=d0); s.update(b)
    sets = {'offlike_m': sorted(w for w in s if w[0] >= CUT), 'runstart_m': sorted((a, b) for a, b, r in zip(dd.tolist(), tt.tolist(), rs.tolist()) if r and a >= CUT)}
    rng = np.random.default_rng(pv.zlib.crc32(f'{panel}|thin'.encode()))
    u = rng.random(Pv['V'].shape, dtype=np.float32)
    for f_ in FS:
        Hc = dict(V=np.where(u < f_, Pv['V'], np.nan).astype(np.float32), Fq=np.where(u < f_, Pv['Fq'], np.nan).astype(np.float32))
        for sname, wl in sets.items():
            if not wl:
                continue
            R = pv.cube_rows(P, wl, prior_tr, Hc, Pv, Pv['dow'], Pv['ON'], Pv['OFF'], vt='cube', vt_cube=Pv['V'])
            X, y, e, meta, call = lab.to_matrix(R, lab.ONGOING, None, lab.ongoing_keep(R))
            Xf = pv.blended_full(X, R, meta, Pv['dow'], prior_tr, cQ, cE)
            d_, t_ = R['d'], R['t']
            win = pd.DataFrame(dict(d=d_, t=t_, cond=lab.ONGOING, fold=0, regime=rday[d_]))
            rc = pv.types.SimpleNamespace(truth=np.stack([Pv['Q'][d_, t_ + h] for h in range(1, F + 1)], 1), ev=np.stack([Pv['E'][d_, t_ + h] for h in range(1, F + 1)], 1),
                                          qlast=R['qlast'], win=win, name=sname, panel=panel, cond=lab.ONGOING, L=R['L'])
            ev = pv.Ev(rc, X, Xf, meta, R['W'], R['L'])
            P6 = pv.predict_members([ev], {k: (mdl, cs, bl) for k, (cs, mdl, bl) in M.items()})
            for nm, wts in ARMS.items():
                df = pv.score_members([ev], P6, wts, nm, win_cols=('regime',)); df['f'] = f_; res.append(df)
    lab.drop_panel(panel)
R_ = pd.concat(res, ignore_index=True); R_.to_pickle(f'{pv.RES}/thin_{"_".join(sorted(set(k.split("_")[0] for k in M)))}.pkl')
base = 'a R:np+id'
rows = []
for sname in ('offlike_m', 'runstart_m'):
    for rg in ('all', 0.8, 0.7, 0.5):
        for f_ in FS:
            g = R_[(R_.set == sname) & (R_.f == f_)]
            if rg != 'all':
                g = g[g.regime == rg]
            A = g[g.arm == base]
            r = dict(set=sname, regime=rg, f=f_, hist_vis=round(float((g.regime * f_).mean()), 3), n=len(A), iou_a=lab.aggregate(A))
            for nm in ARMS:
                if nm == base:
                    continue
                B = g[g.arm == nm]; bt = lab.paired_bootstrap(A, B)
                r[nm] = f'{bt["diff"]:+.4f} [{bt["lo"]:+.4f},{bt["hi"]:+.4f}]'
            rows.append(r)
T = pd.DataFrame(rows); T.to_csv(f'{pv.RES}/thin_table.csv', index=False)
print(T.to_string(index=False))
# within-window change of each arm when the history is thinned (f = 1 -> 0.5): how much does each arm lose?
rows = []
for sname in ('offlike_m', 'runstart_m'):
    for nm in ARMS:
        A = R_[(R_.set == sname) & (R_.f == 1.0) & (R_.arm == nm)]; B = R_[(R_.set == sname) & (R_.f == 0.5) & (R_.arm == nm)]
        bt = lab.paired_bootstrap(A.drop(columns='f'), B.drop(columns='f'))
        rows.append(dict(set=sname, arm=nm, iou_f1=lab.aggregate(A), iou_f05=lab.aggregate(B), loss=bt['diff'], lo=bt['lo'], hi=bt['hi']))
print('\nloss of each arm when the visible history is halved (same windows)'); print(pd.DataFrame(rows).round(4).to_string(index=False))
