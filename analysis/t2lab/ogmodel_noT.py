"""TRACK ogmodel S5 (measurement only): cost of removing the slot-T information from the ongoing model.
Contingency for an open organizer ruling (data AT T may be disallowed): features are rebuilt with NO slot-T value
(VT = NaN -> vr_last / q_last and every derived column (neighbours, band minima, distances to the nearest queued link,
queue share, keep region) come from the history T-60..T-5 only) and the columns vr_T / has_T are dropped.
Same F03 recipe, same windows, same folds and LightGBM parameters as the reference arm fc_F03_<cols>255.

Rows are rebuilt with the production functions (window_features, add_ramp, prior_table, prior_feats, to_matrix,
ongoing_keep) in the layout of lab.RowCache under data/t2local/cache/ogmodel/rows_noT/.
`check` mode rebuilds one small set WITH the production slot-T simulation and compares it with the lab cache (parity).
Usage (through run_heavy): ogmodel_noT.py build | check | run [cols=np] [folds=01234]"""
import gc, json, os, sys, time, warnings
import numpy as np
import pandas as pd
import lab
import exp_firstcross_lib as fc
import ogmodel_lib as og

H, F = lab.H, lab.F
ROWS = f'{og.CACHE}/rows_noT'
os.makedirs(ROWS, exist_ok=True)
TSETS = (('faith3', 'faith3c'), ('runstart', 'runstart'))          # (lab window set, own name); faith3c = the F03 subsample
ESETS = ('official', 'offlike', 'pseudo')
DROP_T = ('vr_T', 'has_T')


def make_rows_vt(P, wins, prior, mode):
    """production make_rows with the slot-T value simulated as production (mode 'T') or absent (mode 'noT')."""
    d = np.array([w[0] for w in wins]); t = np.array([w[1] for w in wins]); cond = [w[2] for w in wins]
    idx = t[:, None] + np.arange(-H, 0)[None]
    Vh = P['V'][d[:, None], idx]; Fh = P['Fq'][d[:, None], idx]
    if mode == 'T':
        rng = np.random.default_rng(len(wins))
        VT = np.where(rng.random((len(d), P['V'].shape[2])) < 0.62, P['V'][d, t], np.nan)
    else:
        VT = np.full((len(d), P['V'].shape[2]), np.nan, np.float32)
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        Fe, qlast = lab.window_features(P, Vh, Fh, t, P['dow'][d], prior, VT)
        lab.add_ramp(Fe, P['ON'], P['OFF'], d, t, P['cap'])
    W, L = qlast.shape
    rows = []
    for h in range(1, F + 1):
        feats = {k: v for k, v in Fe.items()}
        feats['h'] = np.full((W, L), h, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = lab.prior_feats(prior, P['dow'][d], t, h)
        rows.append((h, feats, P['Q'][d, np.minimum(t + h, 287)], P['E'][d, np.minimum(t + h, 287)]))
    return dict(d=d, t=t, cond=np.array(cond), rows=rows, qlast=qlast, W=W, L=L)


def build_rows(P, win, mode):
    """lab._build_rows with make_rows_vt (same union keep region logic, fold-safe priors, keep masks)."""
    cond = lab.ONGOING
    wins = list(zip(win.d.tolist(), win.t.tolist(), win.cond.tolist()))
    D = len(P['days'])
    priors = [lab.prior_table(P, P['block'] != k) for k in range(4)] + [lab.prior_table(P, np.ones(D, bool))]
    R = make_rows_vt(P, wins, priors[4], mode)
    d, t = R['d'], R['t']; dw = P['dow'][d]
    keeps = []
    for k in range(5):
        f0 = dict(R['rows'][0][1]); f0['prior_now'] = lab.prior_feats(priors[k], dw, t, 1)[1]
        keeps.append(lab.ongoing_keep(dict(R, rows=[(1, f0, None, None)])))
    keepU = np.logical_or.reduce(keeps)
    X, y, e, meta, cols = lab.to_matrix(R, cond, None, keepU)
    wi_, li_, h_ = meta[:, 0], meta[:, 1], meta[:, 2]
    prior5 = np.zeros((len(X), 5, 3), np.float32); keep5 = np.zeros((len(X), 5), bool)
    for k in range(5):
        for h in np.unique(h_):
            m = h_ == h
            pf = lab.prior_feats(priors[k], dw, t, int(h))
            for j in range(3):
                prior5[m, k, j] = pf[j][wi_[m], li_[m]]
        keep5[:, k] = keeps[k][wi_, li_]
    truth = np.stack([P['Q'][d, np.minimum(t + h, 287)] for h in range(1, F + 1)], 1)
    ev = np.stack([P['E'][d, np.minimum(t + h, 287)] for h in range(1, F + 1)], 1)
    return X, dict(y=y.astype(bool), e=e.astype(bool), wi=wi_.astype(np.int32), li=li_.astype(np.int16), h=h_.astype(np.int8),
                   prior5=prior5, keep5=keep5, truth=truth, ev=ev, qlast=R['qlast']), cols


def rows_noT(panel, name):
    return lab.RowCache(f'{ROWS}/{panel}__{name}__{lab.ONGOING}')


def win_of(panel, lab_set):
    win = lab.window_set(panel, lab_set); win = win[win.cond == lab.ONGOING].reset_index(drop=True)
    if lab_set == 'faith3':                                   # exactly the windows the F03 arm uses (weight > 0)
        win = win[fc.wfn_F(0.3)(win, 'faith3', panel) > 0].reset_index(drop=True)
    return win


def build_all():
    for panel in lab.PANELS:
        P = lab.load_panel(panel); t1 = time.time()
        for lab_set, name in TSETS + tuple((s, s) for s in ESETS):
            path = f'{ROWS}/{panel}__{name}__{lab.ONGOING}'
            if os.path.exists(f'{path}/meta.json'):
                continue
            win = win_of(panel, lab_set)
            os.makedirs(path, exist_ok=True)
            X, aux, cols = build_rows(P, win, 'noT')
            np.save(f'{path}/X.npy', X); np.savez_compressed(f'{path}/aux.npz', **aux); win.to_pickle(f'{path}/win.pkl')
            json.dump(dict(cols=list(cols), panel=panel, name=name, cond=lab.ONGOING, L=len(P['links']), n=int(len(X)), mode='noT'), open(f'{path}/meta.json', 'w'))
            del X, aux; gc.collect()
        lab.log(panel, f'no-T rows built {time.time() - t1:.0f}s')
        lab.drop_panel(panel)


def check():
    """parity of build_rows: mode 'T' must equal the lab cache bit for bit (official + runstart of one panel)."""
    for panel, name in (('D7_I10_W', 'official'), ('D7_I10_W', 'runstart'), ('D12_I5_S', 'offlike')):
        P = lab.load_panel(panel)
        win = lab.window_set(panel, name); win = win[win.cond == lab.ONGOING].reset_index(drop=True)
        X, aux, cols = build_rows(P, win, 'T')
        rc = lab.rows(panel, name)
        ok = list(cols) == rc.cols and np.array_equal(X, np.asarray(rc.X), equal_nan=True) and all(np.array_equal(aux[k], getattr(rc, k)) for k in ('y', 'e', 'wi', 'li', 'h', 'keep5', 'truth', 'ev', 'qlast')) \
            and np.array_equal(aux['prior5'], rc.prior5)
        print(panel, name, 'rows', len(X), 'build_rows(mode=T) == lab cache:', ok, flush=True)
        assert ok
        X2, aux2, _ = build_rows(P, win, 'noT')
        jT = [cols.index(c) for c in DROP_T]
        print('   noT rows', len(X2), '| vr_T all NaN', bool(np.isnan(X2[:, jT[0]]).all()), '| has_T all 0', bool((X2[:, jT[1]] == 0).all()),
              '| q_last changed for', round(float((aux2['qlast'] != aux['qlast']).mean()), 4), 'of cells | queued cells at T: with T', int(aux['qlast'].sum()), 'history only', int(aux2['qlast'].sum()))


def train_matrix(k, cols):
    parts = []; n = 0
    for panel in lab.PANELS:
        for (lab_set, name), wt in zip(TSETS, (0.3, 1.0)):
            rc = rows_noT(panel, name)
            if not rc.n:
                continue
            m = rc.e & rc.keep5[:, k] & (rc.fold != k)
            c = int(m.sum()); parts.append((rc, m, wt, c)); n += c
    X = np.empty((n, len(cols)), np.float32); y = np.empty(n, np.float32); w = np.empty(n, np.float32); o = 0
    for rc, m, wt, c in parts:
        X[o:o + c] = rc.matrix(m, cols, k); y[o:o + c] = rc.y[m]; w[o:o + c] = wt; o += c
    return X, y, w


def run(cols='np', folds=(0, 1, 2, 3, 4), leaves=255, lr=0.05, n_trees=500):
    cols_ = [c for c in lab.colset(cols) if c not in DROP_T]
    arm = f'og_noT_{cols}{leaves}'
    odir = og.arm_dir(arm); os.makedirs(odir, exist_ok=True)
    params = lab.lgb_params(num_leaves=leaves, learning_rate=lr, min_data_in_leaf=40, seed=0)
    spec = {}
    for k in folds:
        ff = f'{odir}/fold{k}.pkl'
        if (k < 4 and os.path.exists(ff)) or (k == 4 and os.path.exists(f'{og.MODELS}/{arm}.json')):
            lab.log(f'{arm} fold {k}: found, skipped'); continue
        t1 = time.time()
        X, y, w = train_matrix(k, cols_)
        nrow = len(X)
        mdl = fc.fit(X, y, w, params, n_trees, cols_)
        del X, y, w; gc.collect()
        lab.log(f'{arm} fold {k}: {nrow} rows x {len(cols_)} cols in {time.time() - t1:.0f}s')
        if k == 4:
            og.save_models(arm, spec, [(0, mdl)], cols_, dict(leaves=leaves, lr=lr, n_trees=n_trees, seed=0, nrow=nrow)); continue
        res = []; probs = {}
        for s in ESETS:
            for panel in lab.PANELS:
                rc = rows_noT(panel, s)
                widx = np.where(rc.win.fold.to_numpy() == k)[0]
                if not len(widx):
                    continue
                P6 = og.predict_rc(spec, [(0, mdl)], rc, k, cols_)
                res.append(lab.score_probs(rc, P6, widx, arm=arm, set_name=s))
                probs[(s, panel)] = (widx, P6[widx].astype(np.float16))
        df = pd.concat(res, ignore_index=True)
        pd.to_pickle(dict(df=df, probs=probs, nrow=nrow, secs=time.time() - t1), ff)
        lab.log(f'{arm} fold {k}: ' + ', '.join(f'{s} {lab.aggregate(df[df.set == s]):.4f} (n={int((df.set == s).sum())})' for s in ESETS))
    dfs = [pd.read_pickle(f'{odir}/fold{k}.pkl')['df'] for k in range(4) if os.path.exists(f'{odir}/fold{k}.pkl')]
    df = pd.concat(dfs, ignore_index=True); df.to_pickle(f'{odir}/results.pkl')
    ref = lab.load_arm(f'fc_F03_{cols}255')
    print(f'\n== {arm} (no slot-T information) ==')
    print(lab.summary(df).round(4).to_string(index=False))
    print(f'paired vs fc_F03_{cols}255 (with slot T; same windows):')
    print(pd.DataFrame([og.delta_row(ref[ref.set == s], df[df.set == s], arm, s) for s in ESETS]).round(4).to_string(index=False), flush=True)


if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50)
    mode = sys.argv[1]
    if mode == 'check':
        check()
    elif mode == 'build':
        build_all()
    else:
        build_all()
        run(sys.argv[2] if len(sys.argv) > 2 else 'np', tuple(int(c) for c in sys.argv[3]) if len(sys.argv) > 3 else (0, 1, 2, 3, 4))
    lab.log('DONE')
