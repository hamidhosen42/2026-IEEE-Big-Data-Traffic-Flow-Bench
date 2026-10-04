"""EXPERIMENT firstcross: selection-faithful ongoing training on the 8 real Task 2 panels (LightGBM, OOF over 4 day blocks).

Shared helpers for exp_firstcross_*.py.  Nothing here edits lab.py; the runner mirrors lab.run_arm but
  * builds the training matrix with preallocation (no 2x concatenate peak),
  * saves every fold to disk as soon as it is scored (resume after an interruption),
  * writes the final files in the lab's oof format (results.pkl, probs__<set>__<panel>.npz, config.json) so that
    lab.load_arm / lab.load_probs work on these arms.

Arms (ongoing only; `np` = prior-free production column set incl. ramp features):
  A    dense1500 @1.0                                   current production recipe (production replica windows)
  F01  faithful dense (faith3, non-run-start, <=1500 per panel) @0.1 + all faithful run-starts @1.0
  F03  same with the dense faithful windows @0.3
  H    dense1500 (non-run-start) @0.3 + all faithful run-starts @1.0
  RS   faithful run-starts only @1.0
  G    official-like greedy multi-start windows only (offlike) @1.0
  FD   faithful dense (faith3 incl. its run-starts, <=1500 per panel) @1.0      (no run-start emphasis; reference)
Weights are per window (every row of a window gets the window weight).
"""
import gc, json, os, sys, time
import numpy as np
import pandas as pd
import lab

RES = f'{lab.ROOT}/analysis/t2lab/results/firstcross'
os.makedirs(RES, exist_ok=True)
EVALS = ('official', 'offlike', 'pseudo')
CAP = 1500


def _cap_idx(idx, cap, seed=0):
    if len(idx) > cap:
        idx = np.sort(np.random.default_rng(seed).choice(idx, cap, replace=False))
    return idx


def wfn_F(wd, cap=CAP):
    """faith3: non-run-start faithful windows (at most `cap` per panel, rng(0)) at weight wd; runstart set at 1.0."""
    def f(win, set_name, panel):
        if set_name == 'faith3':
            idx = _cap_idx(np.where(~win.runstart.to_numpy())[0], cap)
            w = np.zeros(len(win), np.float32); w[idx] = wd
            return w
        if set_name == 'runstart':
            return np.ones(len(win), np.float32)
        raise KeyError(set_name)
    return f


def wfn_H(wd=0.3):
    """dense1500 windows that are not faithful run-starts at weight wd (the run-starts come from the runstart set at 1.0)."""
    def f(win, set_name, panel):
        if set_name == 'dense1500':
            return np.where(win.runstart.to_numpy(), 0.0, wd).astype(np.float32)
        if set_name == 'runstart':
            return np.ones(len(win), np.float32)
        raise KeyError(set_name)
    return f


def wfn_FD(cap=CAP):
    """faith3 (run-starts at their natural share) capped at `cap` windows per panel, weight 1."""
    def f(win, set_name, panel):
        idx = _cap_idx(np.arange(len(win)), cap)
        w = np.zeros(len(win), np.float32); w[idx] = 1.0
        return w
    return f


ARMS = {
    'A': dict(train=[('dense1500', 1.0)], wfn=None),
    'F01': dict(train=[('faith3', 1.0), ('runstart', 1.0)], wfn=wfn_F(0.1)),
    'F03': dict(train=[('faith3', 1.0), ('runstart', 1.0)], wfn=wfn_F(0.3)),
    'H': dict(train=[('dense1500', 1.0), ('runstart', 1.0)], wfn=wfn_H(0.3)),
    'RS': dict(train=[('runstart', 1.0)], wfn=None),
    'G': dict(train=[('offlike', 1.0)], wfn=None),
    'FD': dict(train=[('faith3', 1.0)], wfn=wfn_FD()),
    # added after the first results (response curve of the dense-faithful weight): 0 (RS) -> 0.1 -> 0.3 -> 1.0
    'F10': dict(train=[('faith3', 1.0), ('runstart', 1.0)], wfn=wfn_F(1.0)),
}


def train_matrix(train, k, cols, weight_fn=None, panels=None, cond=lab.ONGOING):
    """same rows / order as lab.train_matrix (panel outer, set inner; e & keep5[:, k] & fold != k & weight > 0), preallocated."""
    parts = []; n = 0; info = {}
    for panel in panels or lab.PANELS:
        for name, sw in train:
            rc = lab.rows(panel, name, cond)
            if not rc.n:
                continue
            ww = np.full(len(rc.win), float(sw), np.float32)
            if weight_fn is not None:
                ww = ww * np.asarray(weight_fn(rc.win, name, panel), np.float32)
            m = rc.e & rc.keep5[:, k] & (rc.fold != k) & (ww[rc.wi] > 0)
            c = int(m.sum()); parts.append((rc, m, ww, c)); n += c
            wk = (rc.win.fold.to_numpy() != k) & (ww > 0)
            a = info.setdefault(name, dict(windows=0, rows=0, wsum=0.0, runstart_windows=0))
            a['windows'] += int(wk.sum()); a['rows'] += c; a['wsum'] += float(ww[rc.wi[m]].sum())
            a['runstart_windows'] += int((wk & rc.win.runstart.to_numpy()).sum())
    X = np.empty((n, len(cols)), np.float32); y = np.empty(n, np.float32); w = np.empty(n, np.float32)
    o = 0
    for rc, m, ww, c in parts:
        X[o:o + c] = rc.matrix(m, cols, k); y[o:o + c] = rc.y[m]; w[o:o + c] = ww[rc.wi[m]]; o += c
    return X, y, w, info


def fit(X, y, w, params, n_trees, cols):
    import lightgbm as lgb
    ds = lgb.Dataset(X, y, weight=None if np.all(w == 1) else w, feature_name=list(cols), free_raw_data=True)
    return lgb.train(params, ds, n_trees)


def arm_dir(arm):
    return f'{lab.CACHE}/oof/{arm}'


def run_arm(arm, spec, cols='np', leaves=255, lr=0.05, n_trees=500, mdl=40, folds=(0, 1, 2, 3), evals=EVALS, seeds=(0,),
            extra_params=None, save_models=False):
    """OOF arm with per-fold persistence.  Returns the merged per-window frame of all folds found on disk."""
    cols_ = lab.colset(cols) if isinstance(cols, str) else list(cols)
    odir = arm_dir(arm); os.makedirs(odir, exist_ok=True)
    for k in folds:
        ff = f'{odir}/fold{k}.pkl'
        if os.path.exists(ff):
            lab.log(f'{arm} fold {k}: found on disk, skipped'); continue
        t1 = time.time()
        X, y, w, info = train_matrix(spec['train'], k, cols_, spec['wfn'])
        nrow, pos, wpos = len(X), float(y.mean()), float((y * w).sum() / w.sum())
        models = []
        for sd in seeds:
            p = lab.lgb_params(num_leaves=leaves, learning_rate=lr, min_data_in_leaf=mdl, seed=sd)
            p.update(extra_params or {})
            models.append(fit(X, y, w, p, n_trees, cols_))
        del X, y, w; gc.collect()
        lab.log(f'{arm} fold {k}: {nrow} rows x {len(cols_)} cols, pos {pos:.3f} (weighted {wpos:.3f}), {len(models)} model(s) in '
                f'{time.time() - t1:.0f}s | ' + json.dumps(info))
        if save_models:
            for sd, mdl_ in zip(seeds, models):
                mdl_.save_model(f'{odir}/model_f{k}_s{sd}.txt')
        res = []; probs = {}
        for s in evals:
            for panel in lab.PANELS:
                rc = lab.rows(panel, s, lab.ONGOING)
                widx = np.where(rc.win.fold.to_numpy() == k)[0]
                if not len(widx):
                    continue
                P6 = lab.predict_set(models, rc, k, cols_)
                res.append(lab.score_probs(rc, P6, widx, arm=arm, set_name=s))
                probs[(s, panel)] = (widx, P6[widx].astype(np.float16))
        df = pd.concat(res, ignore_index=True)
        pd.to_pickle(dict(df=df, probs=probs, info=info, nrow=nrow, pos=pos, wpos=wpos, secs=time.time() - t1), ff)
        lab.log(f'{arm} fold {k}: ' + ', '.join(f'{s} {lab.aggregate(df[df.set == s]):.4f} (n={int((df.set == s).sum())})' for s in evals))
        del models; gc.collect()
    cfg = dict(arm=arm, train=[list(x) for x in spec['train']], evals=list(evals), cols=cols_,
               params=dict(lab.lgb_params(num_leaves=leaves, learning_rate=lr, min_data_in_leaf=mdl), **(extra_params or {})),
               n_trees=n_trees, cond=lab.ONGOING, seeds=list(seeds), weighted=spec['wfn'] is not None, cap=CAP)
    return merge(arm, cfg)


def merge(arm, cfg=None):
    """merge fold files into the lab oof format (results.pkl + probs__<set>__<panel>.npz + config.json)."""
    odir = arm_dir(arm); dfs = []; P = {}; folds = []
    for k in range(4):
        ff = f'{odir}/fold{k}.pkl'
        if not os.path.exists(ff):
            continue
        z = pd.read_pickle(ff); dfs.append(z['df']); folds.append(k)
        for (s, panel), (widx, p6) in z['probs'].items():
            if (s, panel) not in P:
                rc = lab.rows(panel, s, lab.ONGOING)
                P[(s, panel)] = dict(P6=np.zeros((len(rc.win), lab.F, rc.L), np.float16), done=np.zeros(len(rc.win), bool),
                                     d=rc.win.d.to_numpy(), t=rc.win.t.to_numpy())
            P[(s, panel)]['P6'][widx] = p6; P[(s, panel)]['done'][widx] = True
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df.to_pickle(f'{odir}/results.pkl')
    for (s, panel), v in P.items():
        np.savez_compressed(f'{odir}/probs__{s}__{panel}.npz', **v)
    if cfg is not None:
        cfg = dict(cfg, folds=folds)
        json.dump(cfg, open(f'{odir}/config.json', 'w'), indent=1, default=str)
    return df


# ------------------------------------------------------------------ statistics
KEY = ['panel', 'd', 't', 'cond']


def boot(dfa, dfb, col='iou', n=2000, seed=0):
    """paired bootstrap of aggregate(B) - aggregate(A); windows resampled within panel (same scheme as lab.paired_bootstrap).
    Returns diff, se (std of the bootstrap distribution), lo, hi (95%), p_pos, n."""
    m = dfa[KEY + ['family', col]].merge(dfb[KEY + [col]], on=KEY, suffixes=('_a', '_b'))
    if not len(m):
        return dict(diff=np.nan, se=np.nan, lo=np.nan, hi=np.nan, p_pos=np.nan, n=0)
    assert not m.duplicated(KEY).any()
    rng = np.random.default_rng(seed)
    groups = [(fam, (g[f'{col}_b'] - g[f'{col}_a']).to_numpy()) for (fam, _), g in m.groupby(['family', 'panel'])]
    fams = sorted({f for f, _ in groups}); diffs = np.zeros(n)
    for b in range(n):
        acc = {f: [] for f in fams}
        for f, dv in groups:
            acc[f].append(dv[rng.integers(0, len(dv), len(dv))].mean())
        diffs[b] = np.mean([np.mean(v) for v in acc.values()])
    obs = np.mean([np.mean([dv.mean() for f, dv in groups if f == ff]) for ff in fams])
    return dict(diff=float(obs), se=float(diffs.std(ddof=1)), lo=float(np.quantile(diffs, 0.025)), hi=float(np.quantile(diffs, 0.975)),
                p_pos=float((diffs > 0).mean()), n=int(len(m)))


def paired(dfa, dfb, col='iou'):
    """windows present in both frames, aligned: returns (a, b) frames with identical key order."""
    m = dfa.merge(dfb[KEY], on=KEY)[KEY]
    a = dfa.merge(m, on=KEY).sort_values(KEY).reset_index(drop=True)
    b = dfb.merge(m, on=KEY).sort_values(KEY).reset_index(drop=True)
    return a, b


def agg_delta(dfa, dfb, col='iou'):
    a, b = paired(dfa, dfb)
    return lab.aggregate(b, col) - lab.aggregate(a, col) if len(a) else np.nan


def blend_frame(arms, weights, set_name, name, folds=None):
    """score a probability blend of cached arms on one evaluation set (windows done in every arm)."""
    out = []
    for panel in lab.PANELS:
        rc = lab.rows(panel, set_name, lab.ONGOING)
        if not len(rc.win):
            continue
        P6 = 0; done = np.ones(len(rc.win), bool)
        for a, wt in zip(arms, weights):
            p, dn = lab.load_probs(a, set_name, panel)
            P6 = P6 + wt * p; done &= dn
        if folds is not None:
            done &= np.isin(rc.win.fold.to_numpy(), list(folds))
        widx = np.where(done)[0]
        if len(widx):
            out.append(lab.score_probs(rc, P6.astype(np.float32), widx, arm=name, set_name=set_name))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
