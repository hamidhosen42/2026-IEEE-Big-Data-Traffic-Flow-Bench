"""VERIFY ogclock (independent re-run): one held-out fold, reference (np or id, no ck) vs candidate (+16 ck columns), trained by
this script on F03 rows (faith3 non-run-starts capped 1500 / panel rng(0) at 0.3, runstart at 1.0) with its own matrix assembly.
ck values come from the track's cache (verified separately by verify_ogclock_feat.py).
Usage: verify_ogclock_rerun.py FOLD SEED LEAVES LR TREES TAG [KIND]"""
import gc, os, sys, time
import numpy as np, pandas as pd, lightgbm as lgb
import lab

k, seed, leaves, lr, ntree, tag = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
kind = sys.argv[7] if len(sys.argv) > 7 else 'np'
CKN = ['ck_run', 'ck_first', 'ck_clear', 'ck_frac', 'ck_minr', 'ck_nep', 'cc_first', 'cc_peak', 'cc_peak_sh', 'cc_since_peak', 'cc_rel',
       'cc_d6', 'cc_d12', 'cc_d24', 'cc_age', 'cc_area']          # layout of the K axis of cache/ogclock/fe_*.npy
FED = f'{lab.CACHE}/ogclock'
OUT = f'{lab.ROOT}/analysis/t2lab/results/verify_ogclock'
os.makedirs(OUT, exist_ok=True)
base_cols = lab.colset(kind)


def wts(win, name):
    if name == 'faith3':
        idx = np.where(~win.runstart.to_numpy().astype(bool))[0]
        if len(idx) > 1500:
            idx = np.sort(np.random.default_rng(0).choice(idx, 1500, replace=False))
        w = np.zeros(len(win), np.float32); w[idx] = 0.3
        return w
    return np.ones(len(win), np.float32)


def fe_rows(panel, name, rc, r):
    FE = np.load(f'{FED}/fe_{panel}__{name}.npy', mmap_mode='r')
    assert FE.shape[0] == len(rc.win) and FE.shape[1] == rc.L and FE.shape[2] == 16, (panel, name, FE.shape)
    return np.asarray(FE[rc.wi[r], rc.li[r], :], np.float32)


t0 = time.time()
Xs, ys, ws = [], [], []
for panel in lab.PANELS:
    for name in ('faith3', 'runstart'):
        rc = lab.rows(panel, name, lab.ONGOING)
        if not rc.n:
            continue
        w = wts(rc.win, name)
        m = rc.e & rc.keep5[:, k] & (rc.fold != k) & (w[rc.wi] > 0)
        assert not np.isin(rc.win.d.to_numpy()[rc.wi[m]], rc.win.d.to_numpy()[rc.win.fold.to_numpy() == k]).any()
        r = np.where(m)[0]
        Xs.append(np.concatenate([rc.matrix(m, base_cols, k), fe_rows(panel, name, rc, r)], 1)); ys.append(rc.y[m]); ws.append(w[rc.wi[m]])
X = np.concatenate(Xs); y = np.concatenate(ys).astype(np.float32); w = np.concatenate(ws); del Xs, ys, ws; gc.collect()
lab.log(f'{tag}: fold {k} train rows {len(X)} x {X.shape[1]} (base {len(base_cols)} + 16 ck), {time.time() - t0:.0f}s')
nb = len(base_cols)
models = {}
for arm, cols in (('ref', base_cols), ('ck', base_cols + CKN)):
    t1 = time.time()
    p = lab.lgb_params(num_leaves=leaves, learning_rate=lr, min_data_in_leaf=40, seed=seed)
    Xa = X[:, :nb] if arm == 'ref' else X
    models[arm] = lgb.train(p, lgb.Dataset(np.ascontiguousarray(Xa), y, weight=w, feature_name=cols, free_raw_data=True), ntree)
    lab.log(f'{tag}: model {arm} trained in {time.time() - t1:.0f}s')
    if arm == 'ck':
        imp = pd.Series(models[arm].feature_importance('gain'), index=cols)
        lab.log(f'{tag}: ck gain share {imp[CKN].sum() / imp.sum():.3f}; top ck', imp[CKN].sort_values(ascending=False).head(4).round(0).to_dict())
del X, y, w; gc.collect()

res = []
for s in ('official', 'offlike', 'pseudo'):
    for panel in lab.PANELS:
        rc = lab.rows(panel, s, lab.ONGOING)
        if not rc.n:
            continue
        wf = rc.win.fold.to_numpy()
        widx = np.where(wf == k)[0]
        if not len(widx):
            continue
        m = rc.keep5[:, k] & (wf[rc.wi] == k)
        r = np.where(m)[0]
        Xe = np.concatenate([rc.matrix(m, base_cols, k), fe_rows(panel, s, rc, r)], 1)
        for arm in ('ref', 'ck'):
            P6 = np.zeros((len(rc.win), lab.F, rc.L), np.float32)
            P6[rc.wi[r], rc.h[r] - 1, rc.li[r]] = models[arm].predict(Xe[:, :nb] if arm == 'ref' else Xe, num_threads=lab.THREADS)
            res.append(lab.score_probs(rc, P6, widx, arm=arm, set_name=s))
R = pd.concat(res, ignore_index=True)
R.to_pickle(f'{OUT}/{tag}.pkl')
for s in ('official', 'offlike', 'pseudo'):
    A = R[(R.arm == 'ref') & (R.set == s)]; B = R[(R.arm == 'ck') & (R.set == s)]
    if not len(A):
        continue
    bt = lab.paired_bootstrap(A, B)
    pa = A.groupby('panel').iou.mean(); pb = B.groupby('panel').iou.mean()
    lab.log(f'{tag} {s}: n {bt["n_windows"]} ref {lab.aggregate(A):.4f} ck {lab.aggregate(B):.4f} diff {bt["diff"]:+.4f} '
            f'[{bt["lo"]:+.4f}, {bt["hi"]:+.4f}] p_pos {bt["p_pos"]:.3f} panels +{int((pb - pa > 1e-12).sum())}/-{int((pb - pa < -1e-12).sum())}')
lab.log(f'{tag} DONE in {time.time() - t0:.0f}s')
