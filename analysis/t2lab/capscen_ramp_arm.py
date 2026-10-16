"""capscen_ramp_arm.py -- PRE-REGISTERED (analysis/CAPACITY_SCENARIO_2026-10-11.md section 0) ongoing correction arms for the
scenario re-draw of ramp demand.  Lab np recipe identical to og_np_base (F03 rows, 255 leaves, 500 trees, lr 0.05).
  noramp : np columns minus the 5 ramp columns
  ramprel: np minus ramp columns + rel_on = ramp_on3 / (ramp_on3 - ramp_on_surge) (= on3 / on12, same window), clip [0, 5]
Per fold k in (0, 1): train, OOF score on official / offlike / pseudo (fold k windows), pseudo-March probabilities (pm_f<k>.pkl),
and probabilities on the 40 + 40 official validation / private ongoing windows (off_f<k>.pkl) -- for this arm AND for og_np_base
(model_f<k>.txt) so that the footprint is lab-model vs lab-model.
Usage (run_heavy): capscen_ramp_arm.py <noramp|ramprel> [folds=01]"""
import gc, os, sys, time, warnings
import numpy as np, pandas as pd, lightgbm as lgb
import lab, ogfeat_lib as og, exp_firstcross_lib as fc

ARM = sys.argv[1]; kw = dict(a.split('=', 1) for a in sys.argv[2:]); folds = tuple(int(c) for c in kw.get('folds', '01'))
RAMP = ['ramp_on3', 'ramp_off3', 'ramp_on_surge', 'ramp_on_pm3', 'ramp_off_pm3']
NP = lab.colset('np'); assert all(c in NP for c in RAMP)
BASE_NR = [c for c in NP if c not in RAMP]
if ARM == 'noramp':
    PC = BASE_NR; NAMES = BASE_NR
elif ARM == 'ramprel':
    PC = BASE_NR + ['ramp_on3', 'ramp_on_surge']; NAMES = BASE_NR + ['ramp_rel_on']
else:
    raise KeyError(ARM)
ALLC = lab.all_cols(); IX = [ALLC.index(c) for c in PC]; IXB = [ALLC.index(c) for c in NP]
NAME = f'og_np_{ARM}'; ODIR = f'{og.ODIR}/arms/{NAME}'; os.makedirs(ODIR, exist_ok=True)
EVALS = ('official', 'offlike', 'pseudo')


def tf(X):
    if ARM != 'ramprel':
        return X
    on3, sur = X[:, -2], X[:, -1]; den = on3 - sur
    with np.errstate(all='ignore'):
        rel = np.where(den > 1e-6, on3 / den, np.nan)
    rel = np.where(np.isfinite(rel), np.clip(rel, 0, 5), np.nan).astype(np.float32)
    return np.concatenate([X[:, :-2], rel[:, None]], 1)


def official_probs(models_new, models_base):
    """{(panel, split): dict(window_id, qlast, tod, P_new, P_base)} on the official ongoing windows (lab inputs, all-train prior)."""
    out = {}
    for panel in lab.PANELS:
        P = lab.load_panel(panel)
        for split in ('validation', 'private'):
            R = lab.split_rows(P, split)
            X, _, _, meta, cols = lab.to_matrix(R, lab.ONGOING, ALLC, lab.ongoing_keep(R))
            assert list(cols) == ALLC
            og_ix = np.where(R['cond'] == lab.ONGOING)[0]
            res = {}
            for tag, ms, ix, f in (('new', models_new, IX, tf), ('base', models_base, IXB, lambda a: a)):
                Xi = f(X[:, ix]); p = sum(m.predict(Xi, num_threads=lab.THREADS) for m in ms) / len(ms)
                P6 = np.zeros((R['W'], lab.F, R['L']), np.float32); P6[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = p
                res[tag] = P6[og_ix]
            org = pd.to_datetime(R['wi'].forecast_origin.iloc[og_ix], utc=True)
            out[(panel, split)] = dict(window_id=R['wi'].window_id.iloc[og_ix].to_numpy(), qlast=R['qlast'][og_ix],
                                       tod=(org.dt.hour + org.dt.minute / 60).to_numpy(), P_new=res['new'], P_base=res['base'])
        lab.drop_panel(panel)
    return out


params = lab.lgb_params(num_leaves=255, learning_rate=0.05, min_data_in_leaf=40, seed=0)
for k in folds:
    T0 = time.time(); mf = f'{ODIR}/model_f{k}.txt'
    if os.path.exists(mf):
        mdl = lgb.Booster(model_file=mf)
    else:
        X, y, w = og.train_matrix(k, PC, [], []); X = tf(X); n = len(X)
        mdl = fc.fit(X, y, w, params, 500, NAMES); del X, y, w; gc.collect(); mdl.save_model(mf)
        lab.log(f'{NAME} fold {k}: {n} rows x {len(NAMES)} cols trained in {time.time() - T0:.0f}s')
    res = []; probs = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for s in EVALS:
            for panel in lab.PANELS:
                rc = lab.rows(panel, s, lab.ONGOING)
                widx = np.where(rc.win.fold.to_numpy() == k)[0]
                if not len(widx) or not rc.n:
                    continue
                m = rc.keep5[:, k] & (rc.win.fold.to_numpy() == k)[rc.wi]
                P6 = np.zeros((len(rc.win), lab.F, rc.L), np.float32)
                if m.any():
                    Xe = tf(og.full_matrix(rc, None, m, PC, [], [], k))
                    P6[rc.wi[m], rc.h[m] - 1, rc.li[m]] = mdl.predict(Xe, num_threads=lab.THREADS)
                res.append(lab.score_probs(rc, P6, widx, arm=NAME, set_name=s)); probs[(s, panel)] = (widx, P6[widx].astype(np.float16))
    df = pd.concat(res, ignore_index=True); pd.to_pickle(dict(df=df, probs=probs), f'{ODIR}/fold{k}.pkl')
    lab.log(f'{NAME} fold {k}: ' + ', '.join(f'{s} {lab.aggregate(df[df.set == s]):.4f} (n={int((df.set == s).sum())})' for s in EVALS))
    pm = {}
    for panel in lab.PANELS:
        z = og.build_pm(panel)
        for s in og.PM_SETS:
            zz = z[s]; meta = zz['meta']
            if not len(zz['win']):
                continue
            P6 = np.zeros((len(zz['win']), lab.F, zz['qlast'].shape[1]), np.float32)
            P6[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = mdl.predict(tf(zz['X'][:, IX]), num_threads=lab.THREADS)
            pm[(panel, s)] = P6.astype(np.float16)
    pd.to_pickle(pm, f'{ODIR}/pm_f{k}.pkl')
    imp = pd.Series(mdl.feature_importance('gain'), index=NAMES).sort_values(ascending=False); (imp / imp.sum()).to_csv(f'{ODIR}/gain_f{k}.csv')
    mb = lgb.Booster(model_file=f'{og.ODIR}/arms/og_np_base/model_f{k}.txt'); assert mb.feature_name() == NP
    pd.to_pickle(official_probs([mdl], [mb]), f'{ODIR}/off_f{k}.pkl')
    lab.log(f'{NAME} fold {k} done in {time.time() - T0:.0f}s')
lab.log('DONE')
