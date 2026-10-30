"""ogphys worker (run through run_heavy; design + PRE-REGISTERED gate: ogphys_lib.py).
Usage: ogphys_job.py build                      side-cars of every panel / set (+ parity vs the cached all-finite rows)
       ogphys_job.py train <occF|occG> <fold>    np then id model on fold k + OOF probabilities (official / offlike / pseudo)
       ogphys_job.py pm                          pseudo-March offlike_m / runstart_m probabilities of base + candidate fold models
       ogphys_job.py off                         probabilities on the 40 + 40 official ongoing windows (validation / private)"""
import gc, json, os, sys, time, types, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import lab
import occ_lib as oc
import ogphys_lib as gp
import exp_firstcross_lib as fc

_argv = sys.argv
sys.argv = [_argv[0], 'none']                       # occ_pmarch runs nothing at import with this argument
import occ_pmarch as opm  # noqa: E402
sys.argv = _argv

F, ONGOING = lab.F, lab.ONGOING
COLS = lab.all_cols()
CS = {'np': lab.colset('np'), 'id': lab.colset('id')}
FOLDS = (0, 1)
T0 = time.time()


def models():
    """{(arm, colset, fold): (booster, extra columns)} for base (occ track) and the candidate arms."""
    M = {}
    for k in FOLDS:
        for c in ('np', 'id'):
            M[('base', c, k)] = (lgb.Booster(model_file=f'{oc.arm_dir(f"occ_{c}255")}/model_f{k}.txt'), [])
            for arm, ex in gp.ARM_COLS.items():
                f = f'{gp.arm_dir(f"{arm}_{c}255")}/model_f{k}.txt'
                if os.path.exists(f):
                    M[(arm, c, k)] = (lgb.Booster(model_file=f), ex)
    for (arm, c, k), (b, ex) in M.items():
        assert b.feature_name() == CS[c] + oc.OCC_COLS + ex, (arm, c, k)
    return M


def train(arm, k):
    ex = gp.ARM_COLS[arm]; var = gp.ARM_VAR[arm]
    for c in ('np', 'id'):
        odir = gp.arm_dir(f'{arm}_{c}255'); os.makedirs(odir, exist_ok=True)
        ff = f'{odir}/fold{k}.pkl'
        if os.path.exists(ff):
            lab.log(f'{arm}_{c} fold {k} exists'); continue
        t1 = time.time()
        X, y, w = gp.train_matrix(k, CS[c], ex, var)
        feat = CS[c] + oc.OCC_COLS + ex
        n, pos = len(X), float(y.mean())
        mdl = fc.fit(X, y, w, lab.lgb_params(num_leaves=255, learning_rate=0.05, min_data_in_leaf=40, seed=0), 500, feat)
        del X, y, w; gc.collect()
        mdl.save_model(f'{odir}/model_f{k}.txt')
        lab.log(f'{arm}_{c}255 fold {k}: {n} rows x {len(feat)} cols, pos {pos:.3f}, trained in {time.time() - t1:.0f}s')
        res = []; probs = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for s in gp.EVAL_SETS:
                for panel in lab.PANELS:
                    rc = oc.rows_af(panel, s, ONGOING)
                    widx = np.where(rc.win.fold.to_numpy() == k)[0]
                    if not len(widx) or not rc.n:
                        continue
                    m = rc.keep5[:, k] & (rc.win.fold.to_numpy() == k)[rc.wi]
                    P6 = np.zeros((len(rc.win), F, rc.L), np.float32)
                    if m.any():
                        P6[rc.wi[m], rc.h[m] - 1, rc.li[m]] = mdl.predict(gp.eval_matrix(rc, panel, s, m, CS[c], k, ex, var), num_threads=lab.THREADS)
                    res.append(lab.score_probs(rc, P6, widx, arm=f'{arm}_{c}255', set_name=s)); probs[(s, panel)] = (widx, P6[widx].astype(np.float16))
        df = pd.concat(res, ignore_index=True)
        imp = dict(zip(mdl.feature_name(), mdl.feature_importance('gain').tolist()))
        pd.to_pickle(dict(df=df, probs=probs, nrow=n, pos=pos, imp=imp, feat=feat, secs=time.time() - t1), ff)
        lab.log(f'{arm}_{c}255 fold {k}: ' + ', '.join(f'{s} {lab.aggregate(df[df.set == s]):.4f} (n={int((df.set == s).sum())})' for s in gp.EVAL_SETS))
        del mdl; gc.collect()


def predict_all(M, X, OF, PH, meta, W, L):
    """{(arm, colset, fold): P6 float16} for one row block; X all production columns, OF {k: occ rows}, PH {var: phys rows (all 20)}."""
    out = {}
    for (arm, c, k), (b, ex) in M.items():
        ix = [COLS.index(cc) for cc in CS[c]]
        Xm = np.hstack([X[:, ix], OF[k]] + ([PH[gp.ARM_VAR[arm]][:, [gp.ALL.index(e) for e in ex]]] if ex else []))
        p6 = np.zeros((W, F, L), np.float32)
        if len(Xm):
            p6[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = b.predict(Xm, num_threads=lab.THREADS)
        out[(arm, c, k)] = p6.astype(np.float16)
    return out


def run_pm():
    M = models(); lab.log('pm models', sorted(M))
    out = {}; frames = {}; info = []
    for panel in lab.PANELS:
        t1 = time.time()
        P, Pv = opm.val_panel(panel)
        cal = np.load(f'{oc.OC}/calib/{panel}.npz'); oc5, c5 = cal['ocrit5'], cal['c5']
        on, og, stable, pers = opm.select(Pv)
        D = len(Pv['days']); g_on, g_og = lab._glist(on), lab._glist(og); s_og = set()
        for d0 in range(D):
            _, b = lab.greedy(g_on, g_og, 'interleaved', start_day=d0); s_og.update(b)
        dd, tt = np.where(og); rs = ~og[dd, tt - 1]
        sets = {'offlike_m': sorted(s_og), 'runstart_m': sorted(zip(dd[rs].tolist(), tt[rs].tolist()))}
        rsset = set(zip(dd[rs].tolist(), tt[rs].tolist()))
        prior = lab.prior_table(P, np.ones(len(P['days']), bool))
        for sname, wl in sets.items():
            if not wl:
                continue
            wins = [(a, b, ONGOING) for a, b in wl]
            R = opm.val_rows(P, Pv, wins, prior, ONGOING)
            X, y, e, meta, call = lab.to_matrix(R, ONGOING, None, lab.ongoing_keep(R))
            assert list(call) == COLS
            d, t = R['d'], R['t']; idx = t[:, None] + np.arange(-lab.H, 0)[None]
            inp = (Pv['V_all'][d[:, None], idx], Pv['F_all'][d[:, None], idx], Pv['V_all'][d, t], Pv['F_all'][d, t])
            S, ql = gp.chunked(P, *inp, panel)
            assert np.array_equal(ql, R['qlast']), (panel, sname, 'qlast differs from val_rows')
            Sd, _ = gp.chunked(P, *inp, panel, sign=1)
            PH = {'xs': S[meta[:, 0], meta[:, 1]], 'd': Sd[meta[:, 0], meta[:, 1]]}
            OF = {k: oc.occ_feats(R['raw'], oc5[k], c5[k], meta[:, 0], meta[:, 1]) for k in FOLDS}
            out[(panel, sname)] = predict_all(M, X, OF, PH, meta, R['W'], R['L'])
            rc = opm.rc_of(Pv, R, sname, panel, ONGOING, extra=dict(runstart=np.array([(a, b) in rsset for a, b in wl], bool)))
            frames[(panel, sname)] = dict(truth=rc.truth, ev=rc.ev, qlast=rc.qlast, win=rc.win)
            info.append(dict(panel=panel, set=sname, n=len(wl), fill_nb_share=float(np.isfinite(S[..., gp.ALL.index('dvT_nb')]).mean()),
                             g_ref_share=float(np.isfinite(S[..., gp.ALL.index('g_cum_km')]).mean())))
            del X, S, Sd, PH, OF; gc.collect()
        lab.log(panel, 'pm done', f'{time.time() - t1:.0f}s', json.dumps(info[-1]))
        lab.drop_panel(panel); gc.collect()
    pd.to_pickle(dict(P=out, frames=frames, info=info), f'{gp.PC}/pm/pm_probs.pkl')
    lab.log('pm saved')


def official_inputs(P, split):
    """window history / slot-T inputs of the released official windows, all-finite (= opm.split_rows_occ) + flow at T."""
    z = np.load(P['npz'], allow_pickle=False)
    x = np.load(f'{oc.XDIR}/t2x_{P["panel"]}.npz', allow_pickle=False)
    wi = lab.window_index(P['panel'], split)
    assert [str(a) for a in x[f'{split}_window_id']] == wi.window_id.tolist()
    Vh = z[f'{split}_Vh'].astype(np.float32); Fh = z[f'{split}_Fh'].astype(np.float32)
    VT = z[f'{split}_VT'].astype(np.float32); FT = z[f'{split}_FT'].astype(np.float32)
    el = np.isfinite(Vh)
    Fh = np.where(el, Fh, x[f'{split}_Fhx'].astype(np.float32)); Vh = np.where(el, Vh, x[f'{split}_Vhx'].astype(np.float32))
    VT2 = np.where(np.isfinite(VT), VT, x[f'{split}_VTx'].astype(np.float32))
    FT2 = np.where(np.isfinite(VT), FT, x[f'{split}_FTx'].astype(np.float32))
    FT2 = np.where(np.isfinite(VT2), FT2, np.nan)
    return Vh, Fh, VT2, FT2, wi


def run_off():
    M = models(); out = {}; info = []
    for panel in lab.PANELS:
        P = lab.load_panel(panel)
        cal = np.load(f'{oc.OC}/calib/{panel}.npz'); oc5, c5 = cal['ocrit5'], cal['c5']
        for split in ('validation', 'private'):
            R = opm.split_rows_occ(P, split)
            X, _, _, meta, call = lab.to_matrix(R, ONGOING, None, lab.ongoing_keep(R))
            assert list(call) == COLS
            Vh, Fh, VT, FT, wi = official_inputs(P, split)
            S, ql = gp.chunked(P, Vh, Fh, VT, FT, panel)
            assert np.array_equal(ql, R['qlast']), (panel, split, 'qlast differs from split_rows_occ')
            Sd, _ = gp.chunked(P, Vh, Fh, VT, FT, panel, sign=1)
            PH = {'xs': S[meta[:, 0], meta[:, 1]], 'd': Sd[meta[:, 0], meta[:, 1]]}
            OF = {k: oc.occ_feats(R['raw'], oc5[k], c5[k], meta[:, 0], meta[:, 1]) for k in FOLDS}
            P6 = predict_all(M, X, OF, PH, meta, R['W'], R['L'])
            og_ix = np.where(R['cond'] == ONGOING)[0]
            org = pd.to_datetime(R['wi'].forecast_origin.iloc[og_ix], utc=True)
            out[(panel, split)] = dict(window_id=R['wi'].window_id.iloc[og_ix].to_numpy(), qlast=R['qlast'][og_ix],
                                       tod=(org.dt.hour + org.dt.minute / 60).to_numpy(), P={kk: v[og_ix] for kk, v in P6.items()})
            info.append(dict(panel=panel, split=split, n_og=len(og_ix), FT_vis=float(np.isfinite(FT[og_ix]).mean()),
                             VT_vis=float(np.isfinite(VT[og_ix]).mean())))
        lab.drop_panel(panel)
    pd.to_pickle(dict(off=out, info=info), f'{gp.PC}/off/off_probs.pkl')
    lab.log('off saved', pd.DataFrame(info).groupby('split')[['n_og', 'FT_vis', 'VT_vis']].mean().round(3).to_dict())


if __name__ == '__main__':
    task = sys.argv[1]
    if task in ('build', 'build_d'):
        st = gp.ensure_side(var='xs' if task == 'build' else 'd')
        st.to_csv(f'{gp.RES}/side_stats{"" if task == "build" else "_d"}.csv', index=False)
        pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
        print(st.to_string(index=False))
        assert bool(st.par_ok.all()), 'side-car parity failed'
        print('ALL PARITY OK', len(st))
    elif task == 'train':
        train(sys.argv[2], int(sys.argv[3]))
    elif task == 'pm':
        run_pm()
    elif task == 'off':
        run_off()
    else:
        raise KeyError(task)
    lab.log('DONE', sys.argv[1:], f'{time.time() - T0:.0f}s')
