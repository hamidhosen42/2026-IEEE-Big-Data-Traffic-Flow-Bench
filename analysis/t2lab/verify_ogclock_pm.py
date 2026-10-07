"""VERIFY ogclock: pseudo-March gate with a different seed.  Final (k = 4, all train days) F03 models trained here (own assembly,
own weights), reference vs +16 ck, scored on the track's pseudo-March caches (offlike_m, runstart_m; ck values verified by
verify_ogclock_feat.py).  np first, then id; results are logged after each kind.
Usage: verify_ogclock_pm.py SEED [KINDS]"""
import gc, os, sys, time, types
import numpy as np, pandas as pd, lightgbm as lgb
import lab

seed = int(sys.argv[1]); kinds = sys.argv[2].split(',') if len(sys.argv) > 2 else ['np', 'id']
CKN = ['ck_run', 'ck_first', 'ck_clear', 'ck_frac', 'ck_minr', 'ck_nep', 'cc_first', 'cc_peak', 'cc_peak_sh', 'cc_since_peak', 'cc_rel',
       'cc_d6', 'cc_d12', 'cc_d24', 'cc_age', 'cc_area']
FED = f'{lab.CACHE}/ogclock'
OUT = f'{lab.ROOT}/analysis/t2lab/results/verify_ogclock'
os.makedirs(OUT, exist_ok=True)
K = 4


def wts(win, name):
    if name == 'faith3':
        idx = np.where(~win.runstart.to_numpy().astype(bool))[0]
        if len(idx) > 1500:
            idx = np.sort(np.random.default_rng(0).choice(idx, 1500, replace=False))
        w = np.zeros(len(win), np.float32); w[idx] = 0.3
        return w
    return np.ones(len(win), np.float32)


def bt_line(A, B, label):
    a = A.sort_values(['panel', 'd', 't']).reset_index(drop=True); b = B.sort_values(['panel', 'd', 't']).reset_index(drop=True)
    assert (a[['panel', 'd', 't']].to_numpy() == b[['panel', 'd', 't']].to_numpy()).all()
    bt = lab.paired_bootstrap(a, b)
    pa = a.groupby('panel').iou.mean(); pb = b.groupby('panel').iou.mean(); dp = (pb - pa).sort_values()
    s = f'{label}: n {bt["n_windows"]} ref {lab.aggregate(a):.4f} ck {lab.aggregate(b):.4f} diff {bt["diff"]:+.4f} [{bt["lo"]:+.4f}, {bt["hi"]:+.4f}] ' \
        f'panels +{int((dp > 1e-12).sum())}/-{int((dp < -1e-12).sum())}'
    if len(dp) > 2:
        best = dp.index[-1]; bt2 = lab.paired_bootstrap(a[a.panel != best], b[b.panel != best])
        s += f' | without best panel {best}: {bt2["diff"]:+.4f} [{bt2["lo"]:+.4f}, {bt2["hi"]:+.4f}]'
    lab.log(s)


allc = lab.all_cols()
PM = {}
for panel in lab.PANELS:
    for s in ('offlike_m', 'runstart_m'):
        f = f'{FED}/pm_{panel}__{s}'
        z = np.load(f + '.npz'); PM[(panel, s)] = (dict(X=z['X'], meta=z['meta'], FE=z['FE'], truth=z['truth'], ev=z['ev'], qlast=z['qlast']),
                                                   pd.read_pickle(f + '.pkl'))
allres = []
for kind in kinds:
    base = lab.colset(kind); nb = len(base); t0 = time.time()
    Xs, ys, ws = [], [], []
    for panel in lab.PANELS:
        for name in ('faith3', 'runstart'):
            rc = lab.rows(panel, name, lab.ONGOING)
            if not rc.n:
                continue
            w = wts(rc.win, name)
            m = rc.e & rc.keep5[:, K] & (w[rc.wi] > 0)
            r = np.where(m)[0]
            FE = np.load(f'{FED}/fe_{panel}__{name}.npy', mmap_mode='r')
            Xs.append(np.concatenate([rc.matrix(m, base, K), np.asarray(FE[rc.wi[r], rc.li[r], :], np.float32)], 1))
            ys.append(rc.y[m]); ws.append(w[rc.wi[m]])
    X = np.concatenate(Xs); y = np.concatenate(ys).astype(np.float32); w = np.concatenate(ws); del Xs, ys, ws; gc.collect()
    lab.log(f'{kind}: k=4 train rows {len(X)} x {X.shape[1]}, {time.time() - t0:.0f}s')
    models = {}
    for arm in ('ref', 'ck'):
        t1 = time.time()
        Xa = np.ascontiguousarray(X[:, :nb]) if arm == 'ref' else X
        models[arm] = lgb.train(lab.lgb_params(num_leaves=255, learning_rate=0.05, min_data_in_leaf=40, seed=seed),
                                lgb.Dataset(Xa, y, weight=w, feature_name=base + (CKN if arm == 'ck' else []), free_raw_data=True), 500)
        del Xa; gc.collect()
        lab.log(f'{kind}: model {arm} trained in {time.time() - t1:.0f}s')
    del X, y, w; gc.collect()
    ix = [allc.index(c) for c in base]
    res = []
    for (panel, s), (z, win) in PM.items():
        if not len(win):
            continue
        meta = z['meta']
        Xp = np.concatenate([z['X'][:, ix], z['FE'][meta[:, 0], meta[:, 1], :]], 1)
        rc = types.SimpleNamespace(truth=z['truth'], ev=z['ev'], qlast=z['qlast'], win=win, name=s, panel=panel, cond=lab.ONGOING, L=z['qlast'].shape[1])
        for arm in ('ref', 'ck'):
            P6 = np.zeros((len(win), lab.F, rc.L), np.float32)
            P6[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = models[arm].predict(Xp[:, :nb] if arm == 'ref' else Xp, num_threads=lab.THREADS)
            df = lab.score_probs(rc, P6, arm=f'{kind}_{arm}', set_name=s); df['wtype'] = win.wtype.to_numpy(); df['P6'] = list(P6)
            res.append(df)
    R = pd.concat(res, ignore_index=True); allres.append(R)
    for s in ('offlike_m', 'runstart_m'):
        A = R[(R.arm == f'{kind}_ref') & (R.set == s)]; B = R[(R.arm == f'{kind}_ck') & (R.set == s)]
        bt_line(A, B, f'PM seed {seed} {kind} {s}')
        if s == 'offlike_m':
            for wt in ('midrun', 'rs_old', 'rs_new'):
                bt_line(A[A.wtype == wt], B[B.wtype == wt], f'PM seed {seed} {kind} {s}:{wt}')
    del models; gc.collect()
R = pd.concat(allres, ignore_index=True)
R.drop(columns=['P6']).to_pickle(f'{OUT}/pm_seed{seed}.pkl')
if set(kinds) >= {'np', 'id'}:
    for s in ('offlike_m', 'runstart_m'):
        out = {}
        for arm in ('ref', 'ck'):
            rows = []
            for panel in lab.PANELS:
                z, win = PM[(panel, s)]
                if not len(win):
                    continue
                a = R[(R.arm == f'np_{arm}') & (R.set == s) & (R.panel == panel)].sort_values(['d', 't'])
                b = R[(R.arm == f'id_{arm}') & (R.set == s) & (R.panel == panel)].sort_values(['d', 't'])
                o = np.lexsort((win.t.to_numpy(), win.d.to_numpy()))
                P6 = 0.5 * (np.stack(a.P6.to_list()) + np.stack(b.P6.to_list()))
                rc = types.SimpleNamespace(truth=z['truth'][o], ev=z['ev'][o], qlast=z['qlast'][o], win=win.iloc[o].reset_index(drop=True), name=s,
                                           panel=panel, cond=lab.ONGOING, L=z['qlast'].shape[1])
                rows.append(lab.score_probs(rc, P6, arm=f'blend_{arm}', set_name=s))
            out[arm] = pd.concat(rows, ignore_index=True)
        bt_line(out['ref'], out['ck'], f'PM seed {seed} np+id blend {s}')
lab.log('DONE')
