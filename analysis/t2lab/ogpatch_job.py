"""ogpatch worker (run through run_heavy; design + PRE-REGISTERED gate: ogpatch_lib.py).
Usage: ogpatch_job.py build                   strips of every panel / set + INTEGRITY (i) vs the cached all-finite rows
       ogpatch_job.py parity                  INTEGRITY (ii): the zero-patch path retrains occ_np255 fold 0 (must equal model_f0.txt)
       ogpatch_job.py train <arm> <fold>      np then id model of the arm on fold k + OOF probabilities (official / offlike / pseudo)
       ogpatch_job.py pm                      pseudo-March offlike_m / runstart_m probabilities of base + candidate fold models
       ogpatch_job.py off                     probabilities on the 40 + 40 official ongoing windows (validation / private)"""
import gc, json, os, sys, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import lab
import occ_lib as oc
import ogpatch_lib as gp
import exp_firstcross_lib as fc

_argv = sys.argv
sys.argv = [_argv[0], 'none']                       # occ_pmarch runs nothing at import with this argument
import occ_pmarch as opm  # noqa: E402
sys.argv = _argv

F, ONGOING = lab.F, lab.ONGOING
COLS = lab.all_cols()
CS = {'np': lab.colset('np'), 'id': lab.colset('id')}
FOLDS = gp.FOLDS
T0 = time.time()


def trees(s):
    return s[s.index('Tree=0'):s.index('end of trees')]


def parity_ii():
    """INTEGRITY (ii): ogpatch matrix builder + Dataset-first fit with zero patch columns vs the stored base model occ_np255 fold 0."""
    feat = CS['np'] + oc.OCC_COLS
    ref = open(f'{oc.arm_dir("occ_np255")}/model_f0.txt').read()
    t1 = time.time()
    ds, n, pos = gp.build_dataset(0, CS['np'], [], feat)
    mdl = lgb.train(gp.params(), ds, gp.N_TREES); del ds; gc.collect()
    s = mdl.model_to_string()
    out = dict(n=n, pos=pos, secs=round(time.time() - t1), trees_identical=trees(s) == trees(ref), full_identical=s == ref,
               header_identical=s.split('Tree=0')[0] == ref.split('Tree=0')[0])
    lab.log('INTEGRITY (ii) ogpatch path vs stored occ_np255 fold 0:', json.dumps(out))
    if not out['trees_identical']:                                   # pre-registered fallback: plain exp_firstcross_lib.fit on the same matrix
        X, y, w = gp.train_matrix(0, CS['np'], [])
        s2 = fc.fit(X, y, w, gp.params(), gp.N_TREES, feat).model_to_string(); del X; gc.collect()
        out.update(plain_vs_new=trees(s2) == trees(s), plain_vs_stored=trees(s2) == trees(ref))
        lab.log('INTEGRITY (ii) fallback:', json.dumps(out))
    json.dump(out, open(f'{gp.RES}/integrity_ii.json', 'w'), indent=1)
    return out


def train(arm, k):
    ex = gp.ARM_COLS[arm]
    for c in ('np', 'id'):
        odir = gp.arm_dir(f'{arm}_{c}255'); os.makedirs(odir, exist_ok=True)
        ff = f'{odir}/fold{k}.pkl'
        if os.path.exists(ff):
            lab.log(f'{arm}_{c} fold {k} exists'); continue
        t1 = time.time()
        feat = CS[c] + oc.OCC_COLS + ex
        ds, n, pos = gp.build_dataset(k, CS[c], ex, feat)
        mdl = lgb.train(gp.params(), ds, gp.N_TREES)
        del ds; gc.collect()
        mdl.save_model(f'{odir}/model_f{k}.txt')
        lab.log(f'{arm}_{c}255 fold {k}: {n} rows x {len(feat)} cols, pos {pos:.3f}, built + trained in {time.time() - t1:.0f}s')
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
                        P6[rc.wi[m], rc.h[m] - 1, rc.li[m]] = mdl.predict(gp.eval_matrix(rc, panel, s, m, CS[c], k, ex), num_threads=lab.THREADS)
                    res.append(lab.score_probs(rc, P6, widx, arm=f'{arm}_{c}255', set_name=s)); probs[(s, panel)] = (widx, P6[widx].astype(np.float16))
        df = pd.concat(res, ignore_index=True)
        imp = dict(zip(mdl.feature_name(), mdl.feature_importance('gain').tolist()))
        pd.to_pickle(dict(df=df, probs=probs, nrow=n, pos=pos, imp=imp, feat=feat, secs=time.time() - t1), ff)
        lab.log(f'{arm}_{c}255 fold {k}: ' + ', '.join(f'{s} {lab.aggregate(df[df.set == s]):.4f} (n={int((df.set == s).sum())})' for s in gp.EVAL_SETS))
        del mdl; gc.collect()


def models():
    """{(arm, colset, fold): (booster, patch columns)} for base (occ track) and the candidate arms present on disk."""
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


def predict_all(M, X, OF, PA, meta, W, L):
    """{(arm, colset, fold): P6 float16} for one row block; X all production columns, OF {k: occ rows}, PA patch rows (columns gp.ALLP)."""
    out = {}
    for (arm, c, k), (b, ex) in M.items():
        ix = [COLS.index(cc) for cc in CS[c]]
        Xm = np.hstack([X[:, ix], OF[k]] + ([PA[:, [gp.ALLP.index(e) for e in ex]]] if ex else []))
        p6 = np.zeros((W, F, L), np.float32)
        if len(Xm):
            p6[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = b.predict(Xm, num_threads=lab.THREADS)
        out[(arm, c, k)] = p6.astype(np.float16)
        del Xm
    return out


def block(P, X, meta, Vh, Fh, VT, FT, tag):
    """patch rows (gp.ALLP) of one row block + INTEGRITY (i) against its production columns."""
    Sv, Sq = gp.patch_strips(P, Vh, Fh, VT, FT)
    par = gp.parity(X, COLS, Sv, Sq, meta[:, 0], meta[:, 1])
    assert par['par_ok'], (tag, par)
    return gp.patch_rows(gp.pad_strip(Sv), gp.pad_strip(Sq), meta[:, 0], meta[:, 1], gp.ALLP), dict(
        T_vis=float(np.isfinite(Sv[:, -1]).mean()), hist_vis=float(np.isfinite(Sv[:, :-1]).mean()))


def run_pm():
    M = models(); lab.log('pm models', sorted(M))
    ref = pd.read_pickle(f'{lab.CACHE}/ogphys/pm/pm_probs.pkl')['P']          # INTEGRITY (iii)
    out = {}; frames = {}; info = []; iii = []
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
            VT = Pv['V_all'][d, t]
            FT = np.where(np.isfinite(VT), Pv['F_all'][d, t], np.nan).astype(np.float32)
            PA, vis = block(P, X, meta, Pv['V_all'][d[:, None], idx], Pv['F_all'][d[:, None], idx], VT, FT, (panel, sname))
            OF = {k: oc.occ_feats(R['raw'], oc5[k], c5[k], meta[:, 0], meta[:, 1]) for k in FOLDS}
            out[(panel, sname)] = predict_all(M, X, OF, PA, meta, R['W'], R['L'])
            rc = opm.rc_of(Pv, R, sname, panel, ONGOING, extra=dict(runstart=np.array([(a, b) in rsset for a, b in wl], bool)))
            frames[(panel, sname)] = dict(truth=rc.truth, ev=rc.ev, qlast=rc.qlast, win=rc.win)
            for kk in [x for x in out[(panel, sname)] if x[0] == 'base']:
                iii.append(bool(np.array_equal(out[(panel, sname)][kk], ref[(panel, sname)][kk])))
            info.append(dict(panel=panel, set=sname, n=len(wl), rows=len(X), **vis))
            del X, PA, OF; gc.collect()
        lab.log(panel, 'pm done', f'{time.time() - t1:.0f}s', json.dumps(info[-1]))
        lab.drop_panel(panel); gc.collect()
    lab.log(f'INTEGRITY (iii) pm base probabilities == ogphys cache: {sum(iii)}/{len(iii)}')
    assert all(iii), 'pm base probabilities differ from the ogphys cache'
    pd.to_pickle(dict(P=out, frames=frames, info=info), f'{gp.PC}/pm/pm_probs.pkl')
    lab.log('pm saved')


def official_inputs(P, split):
    """copy of ogphys_job.official_inputs: window history / slot-T inputs of the released official windows, all-finite + flow at T."""
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
    M = models(); out = {}; info = []; iii = []
    ref = pd.read_pickle(f'{lab.CACHE}/ogphys/off/off_probs.pkl')['off']     # INTEGRITY (iii)
    for panel in lab.PANELS:
        P = lab.load_panel(panel)
        cal = np.load(f'{oc.OC}/calib/{panel}.npz'); oc5, c5 = cal['ocrit5'], cal['c5']
        for split in ('validation', 'private'):
            R = opm.split_rows_occ(P, split)
            X, _, _, meta, call = lab.to_matrix(R, ONGOING, None, lab.ongoing_keep(R))
            assert list(call) == COLS
            Vh, Fh, VT, FT, wi = official_inputs(P, split)
            PA, vis = block(P, X, meta, Vh, Fh, VT, FT.astype(np.float32), (panel, split))
            OF = {k: oc.occ_feats(R['raw'], oc5[k], c5[k], meta[:, 0], meta[:, 1]) for k in FOLDS}
            P6 = predict_all(M, X, OF, PA, meta, R['W'], R['L'])
            og_ix = np.where(R['cond'] == ONGOING)[0]
            org = pd.to_datetime(R['wi'].forecast_origin.iloc[og_ix], utc=True)
            out[(panel, split)] = dict(window_id=R['wi'].window_id.iloc[og_ix].to_numpy(), qlast=R['qlast'][og_ix],
                                       tod=(org.dt.hour + org.dt.minute / 60).to_numpy(), P={kk: v[og_ix] for kk, v in P6.items()})
            rz = ref[(panel, split)]
            assert np.array_equal(rz['window_id'], out[(panel, split)]['window_id'])
            for kk in [x for x in P6 if x[0] == 'base']:
                iii.append(bool(np.array_equal(out[(panel, split)]['P'][kk], rz['P'][kk])))
            info.append(dict(panel=panel, split=split, n_og=len(og_ix), **vis))
        lab.drop_panel(panel)
    lab.log(f'INTEGRITY (iii) off base probabilities == ogphys cache: {sum(iii)}/{len(iii)}')
    assert all(iii), 'off base probabilities differ from the ogphys cache'
    pd.to_pickle(dict(off=out, info=info), f'{gp.PC}/off/off_probs.pkl')
    lab.log('off saved', pd.DataFrame(info).groupby('split')[['n_og', 'T_vis', 'hist_vis']].mean().round(3).to_dict())


if __name__ == '__main__':
    task = sys.argv[1]
    if task == 'build':
        st = gp.ensure_strips()
        st.to_csv(f'{gp.RES}/strip_stats.csv', index=False)
        pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
        print(st.to_string(index=False))
        assert bool(st.par_ok.all()), 'strip parity failed'
        print('ALL PARITY OK', len(st))
    elif task == 'parity':
        parity_ii()
    elif task == 'train':
        train(sys.argv[2], int(sys.argv[3]))
    elif task == 'pm':
        run_pm()
    elif task == 'off':
        run_off()
    else:
        raise KeyError(task)
    lab.log('DONE', sys.argv[1:], f'{time.time() - T0:.0f}s')
