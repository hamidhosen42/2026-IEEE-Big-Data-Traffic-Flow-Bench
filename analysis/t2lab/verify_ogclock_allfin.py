"""VERIFY ogclock: the SHIPPED feature configuration (ALL-FINITE inputs + ck columns, the kernel's own ck_thin / ck_train compiled
from kaggle_jobs/t2_ogclock_v25.py) at lab scale on one held-out fold.  Reference = all-finite F03 rows (allfin caches:
faith3F03 @0.3 + runstart @1.0), candidate = same rows + 16 ck columns built here with all-finite inputs (eligible cells before T-12
thinned by the kernel's regime draw, ineligible-valued cells always visible, all-finite raw history, production slot-T draw with
ineligible-valued cells visible).  Evaluation on the all-finite offlike / official / pseudo rows of the held-out fold.
Usage: verify_ogclock_allfin.py FOLD SEED KIND"""
import ast, gc, os, sys, time
import numpy as np, pandas as pd, lightgbm as lgb
import lab
import exp_firstcross_lib as fc
import allfin_lib as af

k, seed, kind = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
H = lab.H
CKN = ['ck_run', 'ck_first', 'ck_clear', 'ck_frac', 'ck_minr', 'ck_nep', 'cc_first', 'cc_peak', 'cc_peak_sh', 'cc_since_peak', 'cc_rel',
       'cc_d6', 'cc_d12', 'cc_d24', 'cc_age', 'cc_area']
KFILE = f'{lab.ROOT}/kaggle_jobs/t2_ogclock_v25.py'
OUT = f'{lab.ROOT}/analysis/t2lab/results/verify_ogclock'
os.makedirs(OUT, exist_ok=True)
ksrc = open(KFILE).read(); ktree = ast.parse(ksrc)
keep = [n for n in ktree.body if isinstance(n, ast.FunctionDef) and n.name in ('ck_ffill', 'ck_bfill', 'ck_features', 'ck_compose', 'ck_thin', 'ck_train')]
KNS = dict(np=np, pd=pd, H=12, F=6, PANELS=list(lab.PANELS), CK_REGIMES=(0.507, 0.705, 0.803), CK_REGIME_P=(0.297, 0.359, 0.344), CK_ALL=CKN)
exec(compile(ast.Module(body=keep, type_ignores=[]), KFILE, 'exec'), KNS)


def af_ck(P, name, rc):
    """[W, L, 16] all-finite ck features aligned with the allfin row cache rc (same windows, same slot-T draw)."""
    full = lab.window_set(panel, 'faith3' if name == 'faith3F03' else name); full = full[full.cond == lab.ONGOING].reset_index(drop=True)
    L = len(P['links'])
    draw = np.random.default_rng(len(full)).random((len(full), L)) < 0.62
    d = full.d.to_numpy(); t = full.t.to_numpy()
    if name == 'faith3F03':
        sub = af.f03_sub(full); d, t, draw = d[sub], t[sub], draw[sub]
    assert np.array_equal(d, rc.win.d.to_numpy()) and np.array_equal(t, rc.win.t.to_numpy()), (panel, name, 'window order')
    VA, _, Vx = af.feat_cubes(P, 'all')
    Vh = VA[d[:, None], t[:, None] + np.arange(-H, 0)[None]]
    VT = np.where(draw, P['V'][d, t], np.nan); VT = np.where(P['E'][d, t], VT, Vx[d, t])
    m1 = rc.h == 1                                                 # alignment check: slot-T value of the cached all-finite rows
    a_ = (VT / P['cut'][None, :])[rc.wi[m1], rc.li[m1]].astype(np.float32); b_ = np.asarray(rc.X[:, rc.cols.index('vr_T')])[m1]
    assert np.allclose(a_, b_, equal_nan=True, rtol=1e-6), (panel, name, 'vr_T differs')
    a_ = np.asarray(rc.X[:, rc.cols.index('vr_l11')])[m1]; b_ = (Vh[:, 0] / P['cut'][None, :])[rc.wi[m1], rc.li[m1]].astype(np.float32)
    assert np.allclose(a_, b_, equal_nan=True, rtol=1e-6), (panel, name, 'history differs')
    out = KNS['ck_train'](dict(V=P['V'], Vf=VA, E=P['E'], THIN=P['_THIN'], cut=P['cut']), d, t, Vh, VT)
    return np.stack([out[c] for c in CKN], -1)


t0 = time.time()
base = lab.colset(kind); nb = len(base)
Xs, ys, ws = [], [], []; EV = {}
for panel in lab.PANELS:
    P = lab.load_panel(panel)
    P['_THIN'] = KNS['ck_thin'](panel, len(P['days']), len(P['links']))
    for name, sw in (('faith3F03', 0.3), ('runstart', 1.0)):
        rc = af.rows_af(panel, name)
        if not rc.n:
            continue
        FE = af_ck(P, name, rc)
        m = rc.e & rc.keep5[:, k] & (rc.fold != k)
        r = np.where(m)[0]
        Xs.append(np.concatenate([rc.matrix(m, base, k), FE[rc.wi[r], rc.li[r], :]], 1)); ys.append(rc.y[m]); ws.append(np.full(len(r), sw, np.float32))
    for s in ('official', 'offlike', 'pseudo'):
        rc = af.rows_af(panel, s)
        if not rc.n or not (rc.win.fold.to_numpy() == k).any():
            continue
        FE = af_ck(P, s, rc)
        m = rc.keep5[:, k] & (rc.win.fold.to_numpy()[rc.wi] == k); r = np.where(m)[0]
        EV[(s, panel)] = (rc, r, np.concatenate([rc.matrix(m, base, k), FE[rc.wi[r], rc.li[r], :]], 1))
    lab.drop_panel(panel); gc.collect()
    lab.log(f'{panel}: all-finite ck built ({time.time() - t0:.0f}s)')
X = np.concatenate(Xs); y = np.concatenate(ys).astype(np.float32); w = np.concatenate(ws); del Xs, ys, ws; gc.collect()
lab.log(f'allfin {kind} fold {k}: train rows {len(X)} x {X.shape[1]}')
models = {}
for arm in ('ref', 'ck'):
    t1 = time.time()
    Xa = np.ascontiguousarray(X[:, :nb]) if arm == 'ref' else X
    models[arm] = lgb.train(lab.lgb_params(num_leaves=255, learning_rate=0.05, min_data_in_leaf=40, seed=seed),
                            lgb.Dataset(Xa, y, weight=w, feature_name=base + (CKN if arm == 'ck' else []), free_raw_data=True), 500)
    del Xa; gc.collect()
    lab.log(f'allfin {kind}: model {arm} trained in {time.time() - t1:.0f}s')
del X, y, w; gc.collect()
res = []
for (s, panel), (rc, r, Xe) in EV.items():
    widx = np.where(rc.win.fold.to_numpy() == k)[0]
    for arm in ('ref', 'ck'):
        P6 = np.zeros((len(rc.win), lab.F, rc.L), np.float32)
        P6[rc.wi[r], rc.h[r] - 1, rc.li[r]] = models[arm].predict(Xe[:, :nb] if arm == 'ref' else Xe, num_threads=lab.THREADS)
        rc.panel = panel
        res.append(lab.score_probs(rc, P6, widx, arm=arm, set_name=s))
R = pd.concat(res, ignore_index=True)
R.to_pickle(f'{OUT}/allfin_{kind}_f{k}_s{seed}.pkl')
for s in ('official', 'offlike', 'pseudo'):
    A = R[(R.arm == 'ref') & (R.set == s)]; B = R[(R.arm == 'ck') & (R.set == s)]
    if not len(A):
        continue
    bt = lab.paired_bootstrap(A, B)
    pa = A.groupby('panel').iou.mean(); pb = B.groupby('panel').iou.mean()
    lab.log(f'allfin {kind} f{k} s{seed} {s}: n {bt["n_windows"]} ref {lab.aggregate(A):.4f} ck {lab.aggregate(B):.4f} diff {bt["diff"]:+.4f} '
            f'[{bt["lo"]:+.4f}, {bt["hi"]:+.4f}] panels +{int((pb - pa > 1e-12).sum())}/-{int((pb - pa < -1e-12).sum())}')
lab.log(f'DONE in {time.time() - t0:.0f}s')
