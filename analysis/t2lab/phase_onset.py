"""phase_onset.py -- EXP_T2_PHASE onset arm (PREREG results/phase/PREREG.txt).
Production-like CatBoost onset (exp_onset_splitprior.py recipe) base (69 columns) vs phase (+4 phase columns), seeds 0..4,
scored on the 214 pseudo-March onset windows (all-train prior, step 6).  Info: train OOF folds 0/1 seed 0 on offlike onset windows."""
import gc, os, time, types, warnings
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
import lab, ogfeat_lib as og
import phase_lib as ph

ph.check_prereg()
H, F = lab.H, lab.F
cols = lab.all_cols(); SEEDS = (0, 1, 2, 3, 4)
ODIR = f'{ph.ARMD}/onset'; os.makedirs(ODIR, exist_ok=True)


def fit(X, y, seed):
    m = CatBoostClassifier(iterations=600, depth=8, learning_rate=0.08, thread_count=lab.THREADS, verbose=0, random_seed=seed)
    m.fit(X, y); return m


# ---------------------------------------------------------------- training rows (dense1500 onset, as exp_onset_splitprior)
Xs, ys, Zs, fs, Xk = [], [], [], [], {0: [], 1: []}
for p in lab.PANELS:
    rc = lab.rows(p, 'dense1500', lab.ONSET); assert list(rc.cols) == cols, p
    r = np.where(rc.e)[0]
    Xs.append(np.asarray(rc.X[r])); ys.append(rc.y[r].astype(int)); fs.append(rc.fold[r])
    Zs.append(ph.phase_cols(rc.win.t.to_numpy()[rc.wi[r]], rc.h[r]))
    for k in (0, 1):
        Xk[k].append(rc.matrix(rc.e, cols, k))            # fold-k prior (OOF info only)
X = np.concatenate(Xs); y = np.concatenate(ys); Z = np.concatenate(Zs); fo = np.concatenate(fs); del Xs
sub = np.random.default_rng(0).choice(len(X), min(len(X), 1_200_000), replace=False)
lab.log('onset train rows', len(X), 'sub', len(sub), 'ph6==1 share', round(float((Z[:, 0] == 1).mean()), 3))
models = {}
for s in SEEDS:
    for arm in ('base', 'phase'):
        f = f'{ODIR}/cat_{arm}_s{s}.cbm'
        if os.path.exists(f):
            m = CatBoostClassifier(); m.load_model(f)
        else:
            t0 = time.time()
            Xa = X[sub] if arm == 'base' else np.concatenate([X[sub], Z[sub]], 1)
            m = fit(Xa, y[sub], s); m.save_model(f); lab.log(f'{arm} s{s} trained {time.time() - t0:.0f}s')
        models[(arm, s)] = m
imp = pd.Series(models[('phase', 0)].get_feature_importance(), index=cols + ph.PH)
lab.log('phase s0 importance of phase cols', imp[ph.PH].round(3).to_dict(), 'rank of ph6', int((imp > imp['ph6']).sum()) + 1)

# ---------------------------------------------------------------- pseudo-March onset windows (copy of exp_onset_splitprior)
def onset_rows(P, Pv, wins, prior):
    d = np.array([w[0] for w in wins]); t = np.array([w[1] for w in wins])
    idx = t[:, None] + np.arange(-H, 0)[None]
    Vh = Pv['V'][d[:, None], idx]; Fh = Pv['Fq'][d[:, None], idx]; VT = Pv['V'][d, t]
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        Fe, qlast = lab.window_features(P, Vh, Fh, t, Pv['dow'][d], prior, VT)
        lab.add_ramp(Fe, Pv['ON'], Pv['OFF'], d, t, P['cap'])
    W, L = qlast.shape
    R = dict(d=d, t=t, cond=np.array([lab.ONSET] * W), qlast=qlast, W=W, L=L, rows=[])
    for h in range(1, F + 1):
        feats = dict(Fe); feats['h'] = np.full((W, L), h, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = lab.prior_feats(prior, Pv['dow'][d], t, h)
        R['rows'].append((h, feats, Pv['Q'][d, np.minimum(t + h, 287)], Pv['E'][d, np.minimum(t + h, 287)]))
    return R


rows_out = []
for panel in lab.PANELS:
    P, Pv = og.val_panel(panel)
    on, ogm, _ = og.pm_select(Pv)
    D = len(Pv['days'])
    g_on, g_og = lab._glist(on), lab._glist(ogm); sset = set()
    for d0 in range(D):
        a, _ = lab.greedy(g_on, g_og, 'interleaved', start_day=d0); sset.update(a)
    wins = sorted(sset)
    if not wins:
        continue
    prior_tr = lab.prior_table(P, np.ones(len(P['days']), bool))
    R = onset_rows(P, Pv, [(a, b, lab.ONSET) for a, b in wins], prior_tr)
    Xm, _, _, meta, call = lab.to_matrix(R, lab.ONSET, None, None); assert list(call) == cols
    m6 = meta[:, 2] == F; Xm = Xm[m6]; meta = meta[m6]
    Zm = ph.phase_cols(R['t'][meta[:, 0]], meta[:, 2])
    d, t = R['d'], R['t']
    win = pd.DataFrame(dict(d=d, t=t, cond=lab.ONSET, fold=0))
    rc = types.SimpleNamespace(truth=np.stack([Pv['Q'][d, t + h] for h in range(1, F + 1)], 1),
                               ev=np.stack([Pv['E'][d, t + h] for h in range(1, F + 1)], 1),
                               qlast=R['qlast'], win=win, name='march_on', panel=panel, cond=lab.ONSET, L=R['L'])
    for (arm, s), m in models.items():
        Xa = Xm if arm == 'base' else np.concatenate([Xm, Zm], 1)
        P6 = np.zeros((R['W'], F, R['L']), np.float32); P6[meta[:, 0], F - 1, meta[:, 1]] = m.predict_proba(Xa)[:, 1]
        df = lab.score_probs(rc, P6, arm=f'{arm}_s{s}', set_name='march_on'); df['seed'] = s; df['a'] = arm
        rows_out.append(df)
    lab.log(panel, 'pm onset windows', len(wins), 'ph6==1 share', round(float((t % 6 == 1).mean()), 3))
    lab.drop_panel(panel); gc.collect()
PMR = pd.concat(rows_out, ignore_index=True); PMR.to_pickle(f'{ph.RES}/onset_pm.pkl')

# ---------------------------------------------------------------- info: train OOF (seed 0, fold-k prior, offlike onset windows)
oof = []
Xk = {k: np.concatenate(v) for k, v in Xk.items()}
for k in (0, 1):
    tr = np.where(fo != k)[0]; tr = np.random.default_rng(0).choice(tr, min(len(tr), 1_200_000), replace=False)
    mk = {'base': fit(Xk[k][tr], y[tr], 0), 'phase': fit(np.concatenate([Xk[k][tr], Z[tr]], 1), y[tr], 0)}
    for panel in lab.PANELS:
        rc = lab.rows(panel, 'offlike', lab.ONSET)
        widx = np.where(rc.win.fold.to_numpy() == k)[0]
        if not len(widx) or not rc.n:
            continue
        msk = (rc.win.fold.to_numpy() == k)[rc.wi] & (rc.h == F)
        Xe = rc.matrix(msk, cols, k); r = np.where(msk)[0]; Ze = ph.phase_cols(rc.win.t.to_numpy()[rc.wi[r]], rc.h[r])
        for arm, m in mk.items():
            P6 = np.zeros((len(rc.win), F, rc.L), np.float32)
            P6[rc.wi[r], F - 1, rc.li[r]] = m.predict_proba(Xe if arm == 'base' else np.concatenate([Xe, Ze], 1))[:, 1]
            oof.append(lab.score_probs(rc, P6, widx, arm=arm, set_name='offlike'))
    lab.log(f'onset OOF fold {k} done')
pd.concat(oof, ignore_index=True).to_pickle(f'{ph.RES}/onset_oof.pkl')
lab.log('DONE')
