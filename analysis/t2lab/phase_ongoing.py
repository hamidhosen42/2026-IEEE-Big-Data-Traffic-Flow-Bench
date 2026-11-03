"""phase_ongoing.py -- EXP_T2_PHASE ongoing arm og_np_phase = og_np_base + 4 phase columns (PREREG results/phase/PREREG.txt).
Per fold k in (0, 1): train (F03 rows, same order as ogfeat_lib.train_matrix), OOF scoring of NEW and the cached BASE model on
official / offlike / pseudo / runstart, pseudo-March probabilities (both), official validation / private probabilities (both).
Usage (run_heavy): phase_ongoing.py [folds=01]"""
import gc, os, sys, time, warnings
import numpy as np, pandas as pd, lightgbm as lgb
import lab, ogfeat_lib as og, exp_firstcross_lib as fc
import phase_lib as ph

ph.check_prereg()
kw = dict(a.split('=', 1) for a in sys.argv[1:]); folds = tuple(int(c) for c in kw.get('folds', '01'))
NP = lab.colset('np'); NAMES = NP + ph.PH
ALLC = lab.all_cols(); IXB = [ALLC.index(c) for c in NP]
ODIR = f'{ph.ARMD}/og_np_phase'; os.makedirs(ODIR, exist_ok=True)
BDIR = f'{og.ODIR}/arms/og_np_base'
EVALS = ('official', 'offlike', 'pseudo', 'runstart')


def train_phase(k):
    """phase columns of the rows of og.train_matrix(k, NP, [], []) in the same order."""
    spec = fc.ARMS['F03']; out = []
    for panel in lab.PANELS:
        for name, sw in spec['train']:
            rc = lab.rows(panel, name, lab.ONGOING)
            if not rc.n:
                continue
            ww = np.full(len(rc.win), float(sw), np.float32)
            if spec['wfn'] is not None:
                ww = ww * np.asarray(spec['wfn'](rc.win, name, panel), np.float32)
            m = rc.e & rc.keep5[:, k] & (rc.fold != k) & (ww[rc.wi] > 0)
            out.append(ph.phase_cols(rc.win.t.to_numpy()[rc.wi[m]], rc.h[m]))
    return np.concatenate(out)


def official_probs(mdl_new, mdl_base):
    out = {}
    for panel in lab.PANELS:
        P = lab.load_panel(panel)
        for split in ('validation', 'private'):
            R = lab.split_rows(P, split)
            X, _, _, meta, cols = lab.to_matrix(R, lab.ONGOING, ALLC, lab.ongoing_keep(R))
            assert list(cols) == ALLC
            og_ix = np.where(R['cond'] == lab.ONGOING)[0]
            slot = np.asarray(R['t'])
            Xb = X[:, IXB]; Xn = np.concatenate([Xb, ph.phase_cols(slot[meta[:, 0]], meta[:, 2])], 1)
            res = {}
            for tag, mdl, Xi in (('new', mdl_new, Xn), ('base', mdl_base, Xb)):
                P6 = np.zeros((R['W'], lab.F, R['L']), np.float32); P6[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = mdl.predict(Xi, num_threads=lab.THREADS)
                res[tag] = P6[og_ix]
            org = pd.to_datetime(R['wi'].forecast_origin.iloc[og_ix], utc=True)
            out[(panel, split)] = dict(window_id=R['wi'].window_id.iloc[og_ix].to_numpy(), qlast=R['qlast'][og_ix], slot=slot[og_ix],
                                       tod=(org.dt.hour + org.dt.minute / 60).to_numpy(), P_new=res['new'], P_base=res['base'])
        lab.drop_panel(panel)
    return out


params = lab.lgb_params(num_leaves=255, learning_rate=0.05, min_data_in_leaf=40, seed=0)
for k in folds:
    T0 = time.time(); mf = f'{ODIR}/model_f{k}.txt'
    mb = lgb.Booster(model_file=f'{BDIR}/model_f{k}.txt'); assert mb.feature_name() == NP
    if os.path.exists(mf):
        mdl = lgb.Booster(model_file=mf)
    else:
        X, y, w = og.train_matrix(k, NP, [], []); Z = train_phase(k); assert len(Z) == len(X)
        lab.log(f'fold {k}: phase share ph6==1 {float((Z[:, 0] == 1).mean()):.3f}, ph_on {float(Z[:, 1].mean()):.3f}')
        X = np.concatenate([X, Z], 1); n = len(X); del Z
        mdl = fc.fit(X, y, w, params, 500, NAMES); del X, y, w; gc.collect(); mdl.save_model(mf)
        lab.log(f'og_np_phase fold {k}: {n} rows x {len(NAMES)} cols trained in {time.time() - T0:.0f}s')
    res = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for s in EVALS:
            for panel in lab.PANELS:
                rc = lab.rows(panel, s, lab.ONGOING)
                widx = np.where(rc.win.fold.to_numpy() == k)[0]
                if not len(widx) or not rc.n:
                    continue
                m = rc.keep5[:, k] & (rc.win.fold.to_numpy() == k)[rc.wi]
                Pn = np.zeros((len(rc.win), lab.F, rc.L), np.float32); Pb = Pn.copy()
                if m.any():
                    Xb = og.full_matrix(rc, None, m, NP, [], [], k); r = np.where(m)[0]
                    Xn = np.concatenate([Xb, ph.phase_cols(rc.win.t.to_numpy()[rc.wi[r]], rc.h[r])], 1)
                    Pn[rc.wi[m], rc.h[m] - 1, rc.li[m]] = mdl.predict(Xn, num_threads=lab.THREADS)
                    Pb[rc.wi[m], rc.h[m] - 1, rc.li[m]] = mb.predict(Xb, num_threads=lab.THREADS)
                res.append(lab.score_probs(rc, Pn, widx, arm='og_np_phase', set_name=s))
                res.append(lab.score_probs(rc, Pb, widx, arm='og_np_base', set_name=s))
    df = pd.concat(res, ignore_index=True); pd.to_pickle(dict(df=df), f'{ODIR}/fold{k}.pkl')
    # reproduction check of the cached base fold pickle
    cb = pd.read_pickle(f'{BDIR}/fold{k}.pkl')['df']
    for s in ('official', 'offlike', 'pseudo'):
        a = lab.aggregate(cb[cb.set == s]); b = lab.aggregate(df[(df.set == s) & (df.arm == 'og_np_base')])
        assert (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9, (s, a, b)
    lab.log(f'fold {k} base reproduced on official/offlike/pseudo; ' + ', '.join(
        f'{s} base {lab.aggregate(df[(df.set == s) & (df.arm == "og_np_base")]):.4f} new {lab.aggregate(df[(df.set == s) & (df.arm == "og_np_phase")]):.4f}'
        for s in EVALS))
    pm = {}
    for panel in lab.PANELS:
        z = og.build_pm(panel)
        for s in og.PM_SETS:
            zz = z[s]; meta = zz['meta']
            if not len(zz['win']):
                continue
            Xb = zz['X'][:, IXB]; Xn = np.concatenate([Xb, ph.phase_cols(zz['win'].t.to_numpy()[meta[:, 0]], meta[:, 2])], 1)
            for tag, mm, Xi in (('new', mdl, Xn), ('base', mb, Xb)):
                P6 = np.zeros((len(zz['win']), lab.F, zz['qlast'].shape[1]), np.float32)
                P6[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = mm.predict(Xi, num_threads=lab.THREADS)
                pm[(tag, panel, s)] = P6.astype(np.float16)
    pd.to_pickle(pm, f'{ODIR}/pm_f{k}.pkl')
    imp = pd.Series(mdl.feature_importance('gain'), index=NAMES).sort_values(ascending=False); (imp / imp.sum()).to_csv(f'{ODIR}/gain_f{k}.csv')
    lab.log(f'fold {k} phase gain share ' + ', '.join(f'{c} {imp[c] / imp.sum():.4f}' for c in ph.PH))
    pd.to_pickle(official_probs(mdl, mb), f'{ODIR}/off_f{k}.pkl')
    lab.log(f'og_np_phase fold {k} done in {time.time() - T0:.0f}s')
lab.log('DONE')
