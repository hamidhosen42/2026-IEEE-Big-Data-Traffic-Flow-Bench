"""t1div: CatBoost T1 member (EXP_T1_CATB) + a third LightGBM seed control on the t1bspec pseudo-holdout.

Per panel and fold: rebuild exactly the t1bspec 'base' training rows (cached train rows + 3 transductive rounds per month,
same rngs) and evaluation cells (base columns = production FD member, 94 columns, no ramp / timing columns), check them
against the saved t1bspec results, then fit
  catb   CatBoost RMSE residual models for speed and per-lane flow, depth 8 and depth 10 averaged 50/50 (anubisyy
         train_state.py: lr 0.06, l2 10, Bernoulli subsample 0.85; iterations reduced for the lab, see ITER)
  base3  LightGBM, t1lab reduced pF, seed 11 (control: what a third member of the SAME family adds at the same weight)
and store predictions at the evaluation cells (same order as the t1bspec results).
Usage: t1div_exp.py <PANEL[,PANEL]> [folds=0,1] [arms=catb,base3] [it8=1500] [it10=750] [lr=0.12] [rows=all]
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


def fit_catb(D, it8, it10, lr, seed=20261005):
    from catboost import CatBoostRegressor
    preds = {}
    for ch, key in (('v', 'yv'), ('q', 'yq')):
        P = []
        for depth, it in ((8, it8), (10, it10)):
            t = time.time()
            m = CatBoostRegressor(iterations=it, depth=depth, learning_rate=lr, l2_leaf_reg=10, loss_function='RMSE',
                                  thread_count=THREADS, random_seed=seed, bootstrap_type='Bernoulli', subsample=0.85,
                                  border_count=64, verbose=0, allow_writing_files=False)
            m.fit(D['X'], D[key])
            P.append({k: m.predict(D[f'X|{k}'], thread_count=THREADS) for k in ev_keys(D)})
            log(f'  catboost {ch} depth {depth} it {it}: {time.time() - t:.0f}s')
        preds[ch] = {k: 0.5 * (P[0][k] + P[1][k]) for k in P[0]}
    return preds


def fit_lgb(D, seed, n_trees=600, lr=0.08):
    import lightgbm as lgb
    prm = lgb_params(seed, lr=lr)
    mv = lgb.train(prm, lgb.Dataset(D['X'], D['yv']), n_trees); mq = lgb.train(prm, lgb.Dataset(D['X'], D['yq']), n_trees)
    return {'v': {k: mv.predict(D[f'X|{k}']) for k in ev_keys(D)}, 'q': {k: mq.predict(D[f'X|{k}']) for k in ev_keys(D)}}


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
                raw = fit_catb(D, it8, it10, lr) if arm == 'catb' else fit_lgb(D, seed=11)
                np.savez_compressed(f_out, **to_state(D, raw), info=json.dumps(dict(arm=arm, it8=it8, it10=it10, lr=lr, rows=len(D['yv']))))
                log(pn, f'fold {fold}', arm, f'done in {time.time() - t:.0f}s ->', os.path.basename(f_out))
            del D; gc.collect()
    log('done')
