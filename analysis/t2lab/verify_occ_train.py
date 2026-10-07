"""verify_occ_train.py folds=0,1 seed=2 : independent OOF re-run of the occ candidate (np, id, np+id) with a NEW seed.
Baseline rows = the all-finite RowCaches (cache/allfin/rows, read only), training recipe F03 (faith3F03 x 0.3, runstart x 1.0),
255 leaves, lr 0.05, min_data_in_leaf 40, 500 trees.  Occupancy features from verify_occ_feat (own builder, own calibration).
Outputs: data/t2local/cache/verify_occ/train_f<k>_s<seed>.pkl (per-window frames + parity numbers)."""
import gc, os, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import lab
import verify_occ_feat as vf

A = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
FOLDS = [int(x) for x in A.get('folds', '0').split(',')]
SEED = int(A.get('seed', 2))
ONG = lab.ONGOING; F = lab.F
TRAIN = [('faith3F03', 0.3), ('runstart', 1.0)]
EVALS = ('offlike', 'official', 'pseudo')
CID = lab.colset('id'); CNP = lab.colset('np')
assert all(c in CID for c in CNP)
IX_NP = [CID.index(c) for c in CNP]


def side_parity(panel, name, rc, mine, k, r):
    """compare my 13 features with the occ track side-car (their builder + their calibration) on the same rows."""
    import occ_lib as oc
    S = oc.Side(panel, name, ONG)
    theirs = S.rows(rc.wi[r], rc.li[r], k)
    same = np.isclose(mine, theirs, rtol=1e-5, atol=1e-6, equal_nan=True)
    return float(1 - same.mean()), float(np.nanmax(np.abs(mine - theirs))) if len(r) else 0.0


def build_train(k, cal):
    parts = []; n = 0; par = []
    for panel in lab.PANELS:
        oc5, cc5 = cal[panel]
        for name, sw in TRAIN:
            rc = vf.rows_of(panel, name)
            if not rc.n:
                continue
            m = rc.e & rc.keep5[:, k] & (rc.fold != k)
            parts.append((panel, name, rc, m, sw)); n += int(m.sum())
    X = np.empty((n, len(CID) + 13), np.float32); y = np.empty(n, np.float32); w = np.empty(n, np.float32); o = 0
    for panel, name, rc, m, sw in parts:
        oc5, cc5 = cal[panel]
        r = np.where(m)[0]; c = len(r)
        raw = vf.raw_for_rc(panel, name, rc)
        X[o:o + c, :len(CID)] = rc.matrix(m, CID, k)
        Xo = vf.feats_rows(raw, oc5[k], cc5[k], rc.wi[r], rc.li[r]); X[o:o + c, len(CID):] = Xo
        if panel in ('D12_I5_S', 'D7_I210_E'):
            par.append((panel, name, k) + side_parity(panel, name, rc, Xo, k, r) + (bool(raw.get('_par_hasT', np.array(True))),))
        y[o:o + c] = rc.y[m]; w[o:o + c] = sw; o += c
    return X, y, w, par


def fit(X, y, w, cols):
    p = lab.lgb_params(num_leaves=255, learning_rate=0.05, min_data_in_leaf=40, seed=SEED)
    return lgb.train(p, lgb.Dataset(X, y, weight=w, feature_name=list(cols), free_raw_data=True), 500)


def main():
    cal = {p: vf.calib_all(p) for p in lab.PANELS}
    import occ_lib as oc
    cdiff = []
    for p in lab.PANELS:
        z = np.load(f'{oc.OC}/calib/{p}.npz')
        cdiff.append((p, float(np.max(np.abs(z['ocrit5'] - cal[p][0]))), float(np.max(np.abs(z['c5'] - cal[p][1])))))
    lab.log('calibration: mine vs occ track, max |d ocrit|, max |d c| per panel', cdiff)
    occn = [f'occ_{c}' for c in vf.NAMES]
    for k in FOLDS:
        ff = f'{vf.VOUT}/train_f{k}_s{SEED}.pkl'
        if os.path.exists(ff):
            lab.log('fold', k, 'exists'); continue
        t0 = time.time()
        X, y, w, par = build_train(k, cal)
        lab.log(f'fold {k}: train rows {len(y)} pos {y.mean():.4f} built {time.time() - t0:.0f}s | side-car parity (frac differing, max abs, hasT ok) {par}')
        models = {}
        for arm, cols_ix, cols in (('base_np', IX_NP, CNP), ('occ_np', IX_NP + list(range(len(CID), len(CID) + 13)), CNP + occn),
                                   ('base_id', list(range(len(CID))), CID), ('occ_id', list(range(len(CID) + 13)), CID + occn)):
            t1 = time.time()
            Xa = np.ascontiguousarray(X[:, cols_ix]) if len(cols_ix) != X.shape[1] else X
            models[arm] = (fit(Xa, y, w, cols), cols_ix)
            del Xa; gc.collect()
            lab.log(f'fold {k} {arm}: {len(cols_ix)} cols trained {time.time() - t1:.0f}s')
        del X, y, w; gc.collect()
        res = []; epar = []
        for s in EVALS:
            for panel in lab.PANELS:
                oc5, cc5 = cal[panel]
                rc = vf.rows_of(panel, s)
                widx = np.where(rc.win.fold.to_numpy() == k)[0]
                if not len(widx) or not rc.n:
                    continue
                m = rc.keep5[:, k] & (rc.win.fold.to_numpy() == k)[rc.wi]
                r = np.where(m)[0]
                raw = vf.raw_for_rc(panel, s, rc)
                Xo = vf.feats_rows(raw, oc5[k], cc5[k], rc.wi[r], rc.li[r])
                if s == 'offlike':
                    epar.append((panel, s) + side_parity(panel, s, rc, Xo, k, r) + (bool(raw.get('_par_hasT', np.array(True))),))
                Xf = np.hstack([rc.matrix(m, CID, k), Xo])
                P6 = {}
                for arm, (mdl, ix) in models.items():
                    p6 = np.zeros((len(rc.win), F, rc.L), np.float32)
                    p6[rc.wi[r], rc.h[r] - 1, rc.li[r]] = mdl.predict(Xf[:, ix], num_threads=lab.THREADS)
                    P6[arm] = p6
                P6['base_npid'] = 0.5 * (P6['base_np'] + P6['base_id']); P6['occ_npid'] = 0.5 * (P6['occ_np'] + P6['occ_id'])
                for arm, p6 in P6.items():
                    res.append(lab.score_probs(rc, p6, widx, arm=arm, set_name=s))
        df = pd.concat(res, ignore_index=True)
        lab.log(f'fold {k}: eval side-car parity offlike (frac differing, max abs, hasT ok) {epar}')
        pd.to_pickle(dict(df=df, par=par, epar=epar, cdiff=cdiff), ff)
        for s in EVALS:
            g = df[df.set == s]
            for c in ('np', 'id', 'npid'):
                a = g[g.arm == f'base_{c}']; b = g[g.arm == f'occ_{c}']
                bt = lab.paired_bootstrap(a, b)
                lab.log(f'fold {k} seed {SEED} [{s}] {c}: base {lab.aggregate(a):.4f} occ {lab.aggregate(b):.4f} diff {bt["diff"]:+.4f} [{bt["lo"]:+.4f}, {bt["hi"]:+.4f}] n={bt["n_windows"]}')
        lab.log(f'fold {k} done in {time.time() - t0:.0f}s')


main()
lab.log('VERIFY TRAIN DONE', FOLDS, SEED)
