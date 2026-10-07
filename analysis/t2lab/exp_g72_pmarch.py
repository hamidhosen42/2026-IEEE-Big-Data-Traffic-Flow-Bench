"""g72 regime switch on pseudo-March (new scenario).  Final k=4 models (np255, lr 0.05, 500 trees, train split only): F03
(production recipe, existing), F10 (all faithful windows 1.0), F01 (run-starts 1.0, rest 0.1).  Evaluation: pseudo-March offlike_m
windows (masked layer, visible labels) with the replay-derived g72 flag (results/ogmodel/g72_pmarch.pkl, g72_any).
Arms: F03 | F10 | F01 | switch F03|F10 (F10 on flagged windows) | switch F01|F10.  Decision support only."""
import gc, os, sys, time, types, warnings
import numpy as np, pandas as pd, lightgbm as lgb
import lab
import exp_firstcross_lib as fc
from exp_firstcross_prod_snippet import fill_time36

H, F = lab.H, lab.F
MD = f'{fc.RES}/final'
cols_np = lab.colset('np')
models = {}
for a in ('F03', 'F10', 'F01'):
    f = f'{MD}/model_{a}_np255_500.txt'
    if not os.path.exists(f):
        t0 = time.time(); sp = fc.ARMS[a]
        X, y, w, info = fc.train_matrix(sp['train'], 4, cols_np, sp['wfn'])
        m = fc.fit(X, y, w, lab.lgb_params(num_leaves=255, learning_rate=0.05, min_data_in_leaf=40, seed=0), 500, cols_np)
        m.save_model(f); lab.log(f'trained {a}: {len(X)} rows {time.time() - t0:.0f}s'); del X, y, w; gc.collect()
    models[a] = lgb.Booster(model_file=f)
G = pd.read_pickle('results/ogmodel/g72_pmarch.pkl')


def val_panel(panel):
    P = lab.load_panel(panel)
    z = np.load(P['npz'], allow_pickle=False)
    days = np.array([str(x) for x in z['validation_days']]); o = np.argsort(days)
    V = z['validation_Vm'][o].astype(np.float32); Fq = z['validation_Fm'][o].astype(np.float32); Em = z['validation_Em'][o]
    E = np.isfinite(V); Q = E & (V <= P['cut'][None, None, :])
    Pv = dict(panel=panel, cut=P['cut'], days=list(days[o]), dow=pd.to_datetime(pd.Series(days[o])).dt.dayofweek.to_numpy(),
              V=V, Fq=Fq, E=E, Q=Q, Em=Em, ON=z['validation_ON'][o].astype(np.float32), OFF=z['validation_OFF'][o].astype(np.float32))
    return P, Pv


def val_rows(P, Pv, wins, prior):
    d = np.array([w[0] for w in wins]); t = np.array([w[1] for w in wins])
    idx = t[:, None] + np.arange(-H, 0)[None]
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        Fe, qlast = lab.window_features(P, Pv['V'][d[:, None], idx], Pv['Fq'][d[:, None], idx], t, Pv['dow'][d], prior, Pv['V'][d, t])
        lab.add_ramp(Fe, Pv['ON'], Pv['OFF'], d, t, P['cap'])
    W, L = qlast.shape
    R = dict(d=d, t=t, cond=np.array([lab.ONGOING] * W), qlast=qlast, W=W, L=L, rows=[])
    for h in range(1, F + 1):
        feats = dict(Fe); feats['h'] = np.full((W, L), h, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = lab.prior_feats(prior, Pv['dow'][d], t, h)
        R['rows'].append((h, feats, Pv['Q'][d, np.minimum(t + h, 287)], Pv['E'][d, np.minimum(t + h, 287)]))
    return R


res = []
for panel in lab.PANELS:
    P, Pv = val_panel(panel)
    g = G[G.panel == panel]
    if not len(g):
        continue
    wl = list(zip(g.d.tolist(), g.t.tolist()))
    prior = lab.prior_table(P, np.ones(len(P['days']), bool))
    R = val_rows(P, Pv, [(a, b, lab.ONGOING) for a, b in wl], prior)
    X, y, e, meta, call = lab.to_matrix(R, lab.ONGOING, None, lab.ongoing_keep(R))
    ix = [list(call).index(c) for c in cols_np]; Xn = np.ascontiguousarray(X[:, ix])
    d_, t_ = R['d'], R['t']
    win = pd.DataFrame(dict(d=d_, t=t_, cond=lab.ONGOING, fold=0))
    rc = types.SimpleNamespace(truth=np.stack([Pv['Q'][d_, t_ + h] for h in range(1, F + 1)], 1),
                               ev=np.stack([Pv['E'][d_, t_ + h] for h in range(1, F + 1)], 1), qlast=R['qlast'], win=win,
                               name='offlike_m', panel=panel, cond=lab.ONGOING, L=R['L'])
    P6 = {}
    for a, m in models.items():
        p = np.zeros((R['W'], F, R['L']), np.float32); p[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = m.predict(Xn, num_threads=lab.THREADS); P6[a] = p
    flag = g.g72_any.to_numpy().astype(bool)[:, None, None]
    P6['sw_F03|F10'] = np.where(flag, P6['F10'], P6['F03'])
    P6['sw_F01|F10'] = np.where(flag, P6['F10'], P6['F01'])
    for a, p in P6.items():
        df = lab.score_probs(rc, p, arm=a, set_name='offlike_m'); df['g72'] = flag[:, 0, 0]; res.append(df)
    lab.drop_panel(panel); gc.collect()
Rr = pd.concat(res, ignore_index=True); Rr.to_pickle('results/ogmodel_g72_pmarch_switch.pkl')
A0 = Rr[Rr.arm == 'F03']
print('pseudo-March offlike_m, n =', len(A0), '| flagged', int(A0.g72.sum()))
for a in dict.fromkeys(Rr.arm):
    B0 = Rr[Rr.arm == a]
    bt = lab.paired_bootstrap(A0.drop(columns='g72'), B0.drop(columns='g72')) if a != 'F03' else dict(diff=0, lo=0, hi=0)
    pa = A0.groupby('panel').iou.mean(); pb = B0.groupby('panel').iou.mean()
    print(f'{a:12s} {lab.aggregate(B0):.4f} vs F03 {bt["diff"]:+.4f} [{bt["lo"]:+.4f},{bt["hi"]:+.4f}] panels +{int((pb-pa>1e-9).sum())}/-{int((pb-pa<-1e-9).sum())}'
          f' | flagged {B0[B0.g72].iou.mean():.4f} | other {B0[~B0.g72].iou.mean():.4f}')
lab.log('DONE')
