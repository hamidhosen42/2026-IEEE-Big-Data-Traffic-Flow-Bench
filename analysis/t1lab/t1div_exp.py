"""t1div: CatBoost T1 member (EXP_T1_CATB) + a third LightGBM seed control on the t1bspec pseudo-holdout.

Per panel and fold: rebuild exactly the t1bspec 'base' training rows (cached train rows + 3 transductive rounds per month,
same rngs) and evaluation cells (base columns = production FD member, 94 columns, no ramp / timing columns), check them
against the saved t1bspec results, then fit
  catb   CatBoost RMSE residual models for speed and per-lane flow, depth 8 and depth 10 averaged 50/50 (anubisyy
         train_state.py: lr 0.06, l2 10, Bernoulli subsample 0.85; iterations reduced for the lab, see ITER)
  base3  LightGBM, t1lab reduced pF, seed 11 (control: what a third member of the SAME family adds at the same weight)
and store predictions at the evaluation cells (same order as the t1bspec results).
Round 2 (10-16, EXP_T1_XGB / EXP_T1_CATBFTA, gate in t1div_lib.py), same rows / cells:
  xgb      XGBoost hist (max_depth 9, eta 0.06, 1500 rounds, subsample / colsample 0.8, mcw 50, lambda 10), 94 FD columns
  catbfta  CatBoost as catb on the 74 old-FTA columns (no FD columns), random_seed 20261016
  lgbfta   LightGBM seed 13 on the 74 old-FTA columns (info: FTA-aware stand-in)
  lgbbig   POST-HOC info control (added after the round-2 gate result): LightGBM at PRODUCTION capacity (pF of
           t1_pipeline_v13a.py: 511 leaves, lr 0.04, 1500 trees), seed 12, 94 FD columns -- is xgb's margin diversity or capacity?
Round 3 (10-17, EXP_T1_DIV3, gate in t1div_lib.py ROUND 3), same rows / cells:
  xgb2     XGBoost as xgb with seed 20261017 (used as mean(xgb, xgb2) in the report)
  et       sklearn ExtraTrees per target, NaN -> -999, 200 trees, max_features 0.5, min_samples_leaf 20, 400k-row subsample
  lgblk    LightGBM (seed 14) on log-speed / log-density residual targets (flow = density x speed)
  ridge    Ridge(alpha 10) on standardised FD columns (median impute + missing indicators), 600k-row subsample
Usage: t1div_exp.py <PANEL[,PANEL]> [folds=0,1] [arms=catb,base3] [it8=1500] [it10=750] [lr=0.12] [rows=all]
                    [xgb_n=1500] [xgb_eta=0.06] [xgb_depth=9]
"""
import gc, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from t1ramp_exp import prod, View, feats, load_panel, month_cube, lgb_params, SPLITS  # noqa: E402
from t1bspec_exp import N_BS  # noqa: E402
from t1ramp_exp import RCOLS  # noqa: E402
import t1div_lib as L  # noqa: E402

THREADS = int(os.environ.get('T2LAB_THREADS', 4))
CACHE = os.path.join(L.ROOT, 'data', 't2local', 'cache', 't1div')
os.makedirs(CACHE, exist_ok=True)
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def build(panel, fold, p=0.35):
    """training rows (base columns) and evaluation matrices, identical to t1bspec run_fold's 'base' arm. Cached."""
    path = f'{CACHE}/{panel}_f{fold}.npz'
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    z, x, st = load_panel(panel)
    R = np.load(f'{L.BSPEC_CACHE}/{panel}_train_p{p}.npz', allow_pickle=True)
    meta = L.load_meta(panel)
    cols = list(R['cols']); nb = len(cols) - len(RCOLS) - N_BS; bcols = cols[:nb]
    hv, hq = meta['hv'], meta['hq']
    vf = st.vf.to_numpy(); ln = st.lanes.to_numpy()
    saved = np.load(f'{L.BSPEC_RES}/{panel}_f{fold}.npz', allow_pickle=True)
    info = json.loads(str(saved['info']))
    TX, TYV, TYQ = [], [], []
    out = {}
    for si, split in enumerate(SPLITS):
        M = month_cube(panel, z, x, split)
        obs = M['obs']; shape = obs.shape
        EM = L.eval_masks(M, meta, split, fold); EB, HO, hold = EM['EB'], EM['HO'], EM['hold']
        vf3 = np.broadcast_to(vf[None, None, :], shape); ln3 = np.broadcast_to(ln[None, None, :], shape)
        rng_keep = np.random.default_rng(900 + 10 * fold + si)
        for r in range(prod.TR_ROUNDS):
            hid = obs & ~hold & ~EB & (np.random.default_rng(10000 + 10 * r + (split == 'private')).random(shape) < prod.TR_HIDE)
            gone = hid | hold | EB
            Fd = feats(View(np.where(gone, np.nan, M['mv']), np.where(gone, np.nan, M['mq']), M['dow'], M['reg']), st, hv, hq)
            m = hid & (rng_keep.random(shape) < p)
            TX.append(np.column_stack([np.asarray(Fd[k])[m] for k in bcols]).astype(np.float32))
            TYV.append(((M['mv'] / vf3)[m] - Fd['tv'][m]).astype(np.float32)); TYQ.append(((M['mq'] / ln3)[m] - Fd['tq'][m]).astype(np.float32))
            del Fd; gc.collect()
        gone = HO | EB
        Fd = feats(View(np.where(gone, np.nan, M['mv']), np.where(gone, np.nan, M['mq']), M['dow'], M['reg']), st, hv, hq)
        for kind, m in L.eval_cells(EM, obs):
            k = f'{split}_{kind}'
            out[f'X|{k}'] = np.column_stack([np.asarray(Fd[c])[m] for c in bcols]).astype(np.float32)
            out[f'tv|{k}'] = Fd['tv'][m].astype(np.float64); out[f'tq|{k}'] = Fd['tq'][m].astype(np.float64)
            d_, t_, s_ = np.nonzero(m)
            assert np.array_equal(d_, saved[f'{k}|d']) and np.array_equal(s_, saved[f'{k}|s']) and np.array_equal(t_, saved[f'{k}|t']), k
            assert np.allclose(out[f'tv|{k}'], saved[f'{k}|tv'], equal_nan=True) and np.allclose(out[f'tq|{k}'], saved[f'{k}|tq'], equal_nan=True), k
            out[f'vf|{k}'] = vf[s_]; out[f'ln|{k}'] = ln[s_]
        del Fd; gc.collect()
    n_rand = sum(len(a) for a in TYV)
    assert n_rand == info['rand_rows'], (n_rand, info['rand_rows'])
    X = np.concatenate([R['X'][:, :nb]] + TX); YV = np.concatenate([R['yv']] + TYV); YQ = np.concatenate([R['yq']] + TYQ)
    ok = np.isfinite(YV) & np.isfinite(YQ)
    out.update(X=X[ok], yv=YV[ok], yq=YQ[ok], cols=np.array(bcols))
    np.savez(path, **out)
    log(panel, f'fold {fold}: rows {int(ok.sum())} (train {len(R["yv"])} + transductive {n_rand}) cols {nb}; eval cells and tv/tq == t1bspec')
    return out


def fta_cols(D):
    """column indices of the old-FTA feature set (= v12 features without profile columns = the 94 minus the 20 FD columns)."""
    ix = [i for i, c in enumerate(D['cols']) if not str(c).startswith('fd')]
    assert len(ix) == 74 and len(D['cols']) - len(ix) == 20, len(ix)
    return np.array(ix)


def fit_catb(D, it8, it10, lr, seed=20261005, ix=None):
    from catboost import CatBoostRegressor
    sub = (lambda A: A) if ix is None else (lambda A: np.ascontiguousarray(A[:, ix]))
    Xtr = sub(D['X']); Xev = {k: sub(D[f'X|{k}']) for k in ev_keys(D)}
    preds = {}
    for ch, key in (('v', 'yv'), ('q', 'yq')):
        P = []
        for depth, it in ((8, it8), (10, it10)):
            t = time.time()
            m = CatBoostRegressor(iterations=it, depth=depth, learning_rate=lr, l2_leaf_reg=10, loss_function='RMSE',
                                  thread_count=THREADS, random_seed=seed, bootstrap_type='Bernoulli', subsample=0.85,
                                  border_count=64, verbose=0, allow_writing_files=False)
            m.fit(Xtr, D[key])
            P.append({k: m.predict(Xev[k], thread_count=THREADS) for k in ev_keys(D)})
            log(f'  catboost {ch} depth {depth} it {it}: {time.time() - t:.0f}s')
        preds[ch] = {k: 0.5 * (P[0][k] + P[1][k]) for k in P[0]}
    return preds


def fit_lgb(D, seed, n_trees=600, lr=0.08, ix=None, leaves=255):
    import lightgbm as lgb
    prm = lgb_params(seed, leaves=leaves, lr=lr)
    sub = (lambda A: A) if ix is None else (lambda A: A[:, ix])
    Xtr = sub(D['X'])
    mv = lgb.train(prm, lgb.Dataset(Xtr, D['yv']), n_trees); mq = lgb.train(prm, lgb.Dataset(Xtr, D['yq']), n_trees)
    return {'v': {k: mv.predict(sub(D[f'X|{k}'])) for k in ev_keys(D)}, 'q': {k: mq.predict(sub(D[f'X|{k}'])) for k in ev_keys(D)}}


XGB_PRM = dict(tree_method='hist', max_depth=9, eta=0.06, subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
               reg_lambda=10.0, max_bin=256, objective='reg:squarederror', seed=20261016)


def fit_xgb(D, n_rounds=1500, eta=0.06, depth=9):
    import xgboost as xgb
    prm = dict(XGB_PRM, eta=eta, max_depth=depth, nthread=THREADS)
    preds = {}
    for ch, key in (('v', 'yv'), ('q', 'yq')):
        t = time.time()
        dtr = xgb.QuantileDMatrix(D['X'], D[key], max_bin=prm['max_bin'], nthread=THREADS)
        m = xgb.train(prm, dtr, n_rounds)
        del dtr; gc.collect()
        preds[ch] = {k: m.predict(xgb.DMatrix(D[f'X|{k}'], nthread=THREADS)) for k in ev_keys(D)}
        log(f'  xgboost {ch} depth {depth} rounds {n_rounds} eta {eta}: {time.time() - t:.0f}s')
    return preds


def fit_et(D, n_trees=200, n_rows=400_000, mf=0.5, leaf=20):
    from sklearn.ensemble import ExtraTreesRegressor
    n = len(D['yv'])
    ix = np.sort(np.random.default_rng(31).choice(n, n_rows, replace=False)) if n > n_rows else np.arange(n)
    Xtr = np.nan_to_num(D['X'][ix], nan=-999.0)
    preds = {}
    for ch, key in (('v', 'yv'), ('q', 'yq')):
        t = time.time()
        m = ExtraTreesRegressor(n_estimators=n_trees, max_features=mf, min_samples_leaf=leaf, n_jobs=THREADS,
                                random_state=20261017)
        m.fit(Xtr, D[key][ix])
        preds[ch] = {k: m.predict(np.nan_to_num(D[f'X|{k}'], nan=-999.0)) for k in ev_keys(D)}
        log(f'  extratrees {ch} trees {n_trees} rows {len(ix)}: {time.time() - t:.0f}s')
        del m; gc.collect()
    return preds


def lk_targets(X, yv, yq, cols):
    """log-speed / log-density residual targets from the residual targets (yv = v/vf - tv, yq = q/ln - tq)."""
    c = list(cols); tv, tq, vf = X[:, c.index('tv')].astype(np.float64), X[:, c.index('tq')].astype(np.float64), X[:, c.index('vf')].astype(np.float64)
    tvc = np.maximum(tv, 0.03); vr = np.maximum(yv + tv, 0.03)
    ktpl = np.maximum(tq, 1.0) / (tvc * vf)
    k = np.maximum(yq + tq, 1.0) / (vr * vf)
    return np.log(vr) - np.log(tvc), np.log(np.maximum(k, 1e-3)) - np.log(np.maximum(ktpl, 1e-3))


def fit_lgblk(D, seed=14, n_trees=600, lr=0.08):
    """returns raw residuals in the usual (yv, yq) units so that to_state applies unchanged."""
    import lightgbm as lgb
    prm = lgb_params(seed, lr=lr)
    zv, zk = lk_targets(D['X'], D['yv'], D['yq'], D['cols'])
    mv = lgb.train(prm, lgb.Dataset(D['X'], zv), n_trees); mk = lgb.train(prm, lgb.Dataset(D['X'], zk), n_trees)
    c = list(D['cols']); out = {'v': {}, 'q': {}}
    for k in ev_keys(D):
        X = D[f'X|{k}']; tv, tq, vf = D[f'tv|{k}'], D[f'tq|{k}'], X[:, c.index('vf')].astype(np.float64)
        tvc = np.maximum(tv, 0.03); ktpl = np.maximum(np.maximum(tq, 1.0) / (tvc * vf), 1e-3)
        vr = tvc * np.exp(mv.predict(X)); kl = ktpl * np.exp(mk.predict(X))
        out['v'][k] = vr - tv; out['q'][k] = kl * vr * vf - tq
    return out


def fit_ridge(D, alpha=10.0, n_rows=600_000):
    from sklearn.linear_model import Ridge
    n = len(D['yv'])
    ix = np.sort(np.random.default_rng(32).choice(n, n_rows, replace=False)) if n > n_rows else np.arange(n)
    X = D['X'][ix].astype(np.float64)
    med = np.nanmedian(X, 0); med = np.where(np.isfinite(med), med, 0.0)
    nanc = np.flatnonzero(np.isnan(X).any(0))

    def prep(A):
        A = A.astype(np.float64); M = np.isnan(A[:, nanc]).astype(np.float64)
        A = np.where(np.isnan(A), med, A)
        return np.column_stack([A, M])
    Z = prep(X); mu, sd = Z.mean(0), Z.std(0); sd = np.where(sd > 0, sd, 1.0)
    Z = (Z - mu) / sd
    preds = {}
    for ch, key in (('v', 'yv'), ('q', 'yq')):
        m = Ridge(alpha=alpha).fit(Z, D[key][ix])
        preds[ch] = {k: m.predict((prep(D[f'X|{k}']) - mu) / sd) for k in ev_keys(D)}
    return preds


def ev_keys(D):
    return [k[2:] for k in D if k.startswith('X|')]


def to_state(D, raw):
    out = {}
    for k in ev_keys(D):
        out[f'{k}|v'] = np.clip((D[f'tv|{k}'] + raw['v'][k]) * D[f'vf|{k}'], 3, 1.05 * D[f'vf|{k}']).astype(np.float32)
        out[f'{k}|q'] = np.maximum((D[f'tq|{k}'] + raw['q'][k]) * D[f'ln|{k}'], 20).astype(np.float32)
    return out


if __name__ == '__main__':
    panels = sys.argv[1].split(',')
    kw = dict(a.split('=') for a in sys.argv[2:] if '=' in a)
    folds = [int(f) for f in kw.get('folds', '0,1').split(',')]
    arms = kw.get('arms', 'catb,base3').split(',')
    it8, it10, lr = int(kw.get('it8', 1500)), int(kw.get('it10', 750)), float(kw.get('lr', 0.12))
    for pn in panels:
        for fold in folds:
            D = build(pn, fold)
            if kw.get('rows', 'all') != 'all':
                keep = np.sort(np.random.default_rng(0).choice(len(D['yv']), int(kw['rows']), replace=False))
                D.update(X=D['X'][keep], yv=D['yv'][keep], yq=D['yq'][keep])
            for arm in arms:
                f_out = f'{L.RES}/{pn}_f{fold}_{arm}{kw.get("tag", "")}.npz'
                if os.path.exists(f_out):
                    log('exists', f_out); continue
                t = time.time()
                xkw = dict(n_rounds=int(kw.get('xgb_n', 1500)), eta=float(kw.get('xgb_eta', 0.06)), depth=int(kw.get('xgb_depth', 9)))
                if arm == 'catb':
                    raw = fit_catb(D, it8, it10, lr)
                elif arm == 'catbfta':
                    raw = fit_catb(D, it8, it10, lr, seed=20261016, ix=fta_cols(D))
                elif arm == 'xgb':
                    raw = fit_xgb(D, **xkw)
                elif arm == 'xgb2':
                    XGB_PRM['seed'] = 20261017
                    raw = fit_xgb(D, **xkw)
                    XGB_PRM['seed'] = 20261016
                elif arm == 'et':
                    raw = fit_et(D, n_trees=int(kw.get('et_n', 200)), n_rows=int(kw.get('et_rows', 400000)))
                elif arm == 'lgblk':
                    raw = fit_lgblk(D)
                elif arm == 'ridge':
                    raw = fit_ridge(D)
                elif arm == 'lgbfta':
                    raw = fit_lgb(D, seed=13, ix=fta_cols(D))
                elif arm == 'lgbbig':
                    raw = fit_lgb(D, seed=12, n_trees=1500, lr=0.04, leaves=511)
                else:
                    assert arm == 'base3', arm
                    raw = fit_lgb(D, seed=11)
                inf = dict(arm=arm, it8=it8, it10=it10, lr=lr, rows=len(D['yv']), **({'xgb': xkw} if arm in ('xgb', 'xgb2') else {}))
                np.savez_compressed(f_out, **to_state(D, raw), info=json.dumps(inf))
                log(pn, f'fold {fold}', arm, f'done in {time.time() - t:.0f}s ->', os.path.basename(f_out))
            del D; gc.collect()
    log('done')
