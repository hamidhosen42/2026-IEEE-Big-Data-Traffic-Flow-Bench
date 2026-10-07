"""TRACK ogcap: do the ongoing STRUCTURE gains of track ogmodel survive at PRODUCTION capacity?  Shared helpers.

Code paths copied from ogmodel_lib.py (group_ids / fit_struct / predict_struct / predict_rc / delta_row / pm_load /
pm_predict) - nothing of theirs is edited or imported.  Same F03 training matrix as the reference arms
(exp_firstcross_lib.train_matrix: same rows, order, weights), production decode (lab.score_probs), lab.paired_bootstrap.

Own outputs only: data/t2local/cache/ogcap/{oof/<arm>/fold<k>.pkl, models/<arm>.json + __g*.txt}, results/ogcap/.
Read-only inputs: lab row caches, lab arm fc_F03_np511 (joint, full production setting, fold 0), ogmodel arm
og_qsplit_np511 (qsplit, full setting, fold 0), ogmodel pseudo-March row cache (data/t2local/cache/ogmodel/pmarch).

One job = one (structure, fold, seed): per-fold files, no shared merge step (no write races between parallel jobs).
"""
import gc, json, os, time, types
import numpy as np
import pandas as pd
import lab
import exp_firstcross_lib as fc

TRACK = 'ogcap'
RES = f'{lab.ROOT}/analysis/t2lab/results/{TRACK}'
CACHE = f'{lab.CACHE}/{TRACK}'
OOF, MODELS = f'{CACHE}/oof', f'{CACHE}/models'
for _d in (RES, CACHE, OOF, MODELS):
    os.makedirs(_d, exist_ok=True)
OG_OOF = f'{lab.CACHE}/ogmodel/oof'          # read-only (track ogmodel)
OG_PM = f'{lab.CACHE}/ogmodel/pmarch'         # read-only (track ogmodel pseudo-March rows)
H, F = lab.H, lab.F
EVALS = ('official', 'offlike', 'pseudo')
SPEC = fc.ARMS['F03']
KEY = ['panel', 'd', 't', 'cond']

STRUCTS = {
    'joint': dict(),
    'perstep': dict(part='h'),
    'qsplit': dict(part='q'),
}


# ------------------------------------------------------------------ structures (copied from ogmodel_lib)
def group_ids(spec, X, cols):
    part = spec.get('part', 'one')
    if part == 'one':
        return np.zeros(len(X), np.int8)
    h = X[:, cols.index('h')].astype(np.int8)
    if part == 'h':
        return h - 1
    if part == 'q':
        return (X[:, cols.index('q_last')] > 0.5).astype(np.int8)
    raise KeyError(part)


def fit_struct(spec, X, y, w, cols, params, n_trees):
    gid = group_ids(spec, X, cols)
    out = []
    for g in np.unique(gid):
        if spec.get('part', 'one') == 'one':
            out.append((int(g), fc.fit(X, y, w, params, n_trees, cols)))
        else:
            m = gid == g
            out.append((int(g), fc.fit(np.ascontiguousarray(X[m]), y[m], w[m], params, n_trees, cols)))
    return out


def predict_struct(spec, models, X, cols):
    gid = group_ids(spec, X, cols)
    p = np.zeros(len(X), np.float64)
    for g, mdl in models:
        m = gid == g
        if m.any():
            p[m] = mdl.predict(X[m] if not m.all() else X, num_threads=lab.THREADS)
    return p


def predict_rc(spec, models, rc, k, cols):
    """P6 [W, 6, L] for the fold-k windows of rc."""
    P6 = np.zeros((len(rc.win), F, rc.L), np.float32)
    if not rc.n:
        return P6
    wmask = rc.win.fold.to_numpy() == k
    m = rc.keep5[:, k] & wmask[rc.wi]
    if m.any():
        P6[rc.wi[m], rc.h[m] - 1, rc.li[m]] = predict_struct(spec, models, rc.matrix(m, cols, k), cols)
    return P6


def arm_name(struct, cols, leaves, lr, n_trees, seed=0):
    return f'cap_{struct}_{cols}{leaves}_l{int(round(lr * 100)):02d}_t{n_trees}' + (f'_s{seed}' if seed else '')


def run_fold(struct, k, cols='np', leaves=511, lr=0.05, n_trees=900, mdl=40, seed=0, evals=EVALS):
    """k < 4: OOF fold (train on the other day blocks, score the fold-k windows of `evals`); k = 4: final models (all days)."""
    arm = arm_name(struct, cols, leaves, lr, n_trees, seed)
    spec = dict(STRUCTS[struct])
    cols_ = lab.colset(cols)
    odir = f'{OOF}/{arm}'; os.makedirs(odir, exist_ok=True)
    ff = f'{odir}/fold{k}.pkl'
    if (k < 4 and os.path.exists(ff)) or (k == 4 and os.path.exists(f'{MODELS}/{arm}.json')):
        lab.log(f'{arm} fold {k}: found on disk, skipped'); return arm
    params = lab.lgb_params(num_leaves=leaves, learning_rate=lr, min_data_in_leaf=mdl, seed=seed)
    t1 = time.time()
    X, y, w, info = fc.train_matrix(SPEC['train'], k, cols_, SPEC['wfn'])
    nrow = len(X)
    lab.log(f'{arm} fold {k}: matrix {nrow} x {len(cols_)} in {time.time() - t1:.0f}s')
    models = fit_struct(spec, X, y, w, cols_, params, n_trees)
    del X, y, w; gc.collect()
    t_fit = time.time() - t1
    lab.log(f'{arm} fold {k}: {len(models)} model(s) fitted, {t_fit:.0f}s since start')
    cfg = dict(arm=arm, struct=struct, spec=spec, cols=cols_, leaves=leaves, lr=lr, n_trees=n_trees, mdl=mdl, seed=seed, nrow=nrow)
    if k == 4:
        for g, m_ in models:
            m_.save_model(f'{MODELS}/{arm}__g{g}.txt')
        json.dump(dict(cfg, groups=[g for g, _ in models], secs=time.time() - t1), open(f'{MODELS}/{arm}.json', 'w'), indent=1)
        lab.log(f'{arm} final models saved ({time.time() - t1:.0f}s)'); return arm
    res = []; probs = {}
    for s in evals:
        for panel in lab.PANELS:
            rc = lab.rows(panel, s, lab.ONGOING)
            widx = np.where(rc.win.fold.to_numpy() == k)[0]
            if not len(widx):
                continue
            P6 = predict_rc(spec, models, rc, k, cols_)
            res.append(lab.score_probs(rc, P6, widx, arm=arm, set_name=s))
            probs[(s, panel)] = (widx, P6[widx].astype(np.float16))
    df = pd.concat(res, ignore_index=True)
    pd.to_pickle(dict(df=df, probs=probs, cfg=cfg, secs=time.time() - t1, secs_fit=t_fit), ff)
    lab.log(f'{arm} fold {k}: ' + ', '.join(f'{s} {lab.aggregate(df[df.set == s]):.4f} (n={int((df.set == s).sum())})' for s in evals)
            + f' | {time.time() - t1:.0f}s')
    return arm


# ------------------------------------------------------------------ loading (my arms, ogmodel arms, lab arms)
def _my_folds(arm):
    d = f'{OOF}/{arm}'
    return [k for k in range(4) if os.path.exists(f'{d}/fold{k}.pkl')] if os.path.isdir(d) else []


def load_df(arm, folds=None):
    """per-window frame of one arm restricted to `folds` (None = all folds on disk)."""
    ks = _my_folds(arm)
    if ks:
        df = pd.concat([pd.read_pickle(f'{OOF}/{arm}/fold{k}.pkl')['df'] for k in ks], ignore_index=True)
    elif os.path.exists(f'{OG_OOF}/{arm}/results.pkl'):
        df = pd.read_pickle(f'{OG_OOF}/{arm}/results.pkl')
    else:
        df = lab.load_arm(arm)
    df = df.copy(); df['arm'] = arm
    return df if folds is None else df[df.fold.isin(list(folds))].reset_index(drop=True)


_PCACHE = {}


def load_probs(arm, set_name, panel):
    """(P6 float32 [W, 6, L], done [W]) over the whole window set; done marks the windows scored by this arm."""
    key = (arm, set_name, panel)
    if key in _PCACHE:
        return _PCACHE[key]
    ks = _my_folds(arm)
    if ks:
        rc = lab.rows(panel, set_name, lab.ONGOING)
        P6 = np.zeros((len(rc.win), F, rc.L), np.float32); done = np.zeros(len(rc.win), bool)
        for k in ks:
            z = pd.read_pickle(f'{OOF}/{arm}/fold{k}.pkl')
            if (set_name, panel) in z['probs']:
                widx, p6 = z['probs'][(set_name, panel)]
                P6[widx] = p6.astype(np.float32); done[widx] = True
        out = (P6, done)
    elif os.path.exists(f'{OG_OOF}/{arm}/probs__{set_name}__{panel}.npz'):
        z = np.load(f'{OG_OOF}/{arm}/probs__{set_name}__{panel}.npz'); out = (z['P6'].astype(np.float32), z['done'])
    elif os.path.exists(f'{lab.CACHE}/oof/{arm}/probs__{set_name}__{panel}.npz'):
        out = lab.load_probs(arm, set_name, panel)
    else:   # arm scored no window of this (set, panel), e.g. pseudo D7_I10_W in fold 0
        rc = lab.rows(panel, set_name, lab.ONGOING)
        out = (np.zeros((len(rc.win), F, rc.L), np.float32), np.zeros(len(rc.win), bool))
    _PCACHE[key] = out
    return out


def score_blend(parts, set_name, name, folds=None, step_w=None):
    """score a blend of cached arms on one evaluation set.  parts = [(arm, weight)]; step_w = {arm: [w1..w6]} gives a
    per-step weight to that arm (weights normalised per step).  Only windows scored by every part (and in `folds`)."""
    out = []
    for panel in lab.PANELS:
        rc = lab.rows(panel, set_name, lab.ONGOING)
        if not len(rc.win):
            continue
        num = 0; den = 0; done = np.ones(len(rc.win), bool)
        for a, wt in parts:
            p, dn = load_probs(a, set_name, panel); done = done & dn
            ws = np.asarray(step_w[a], np.float32)[None, :, None] if step_w and a in step_w else np.float32(wt)
            num = num + ws * p; den = den + ws
        P6 = (num / np.maximum(den, 1e-12)).astype(np.float32)
        if folds is not None:
            done &= np.isin(rc.win.fold.to_numpy(), list(folds))
        widx = np.where(done)[0]
        if len(widx):
            out.append(lab.score_probs(rc, P6, widx, arm=name, set_name=set_name))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ------------------------------------------------------------------ statistics (copied from ogmodel_lib)
def paired(dfa, dfb):
    m = dfa.merge(dfb[KEY], on=KEY)[KEY]
    a = dfa.merge(m, on=KEY).sort_values(KEY).reset_index(drop=True)
    b = dfb.merge(m, on=KEY).sort_values(KEY).reset_index(drop=True)
    return a, b


def delta_row(dfa, dfb, label='', set_name=''):
    """paired delta B - A on the common windows of ONE evaluation set: lab.paired_bootstrap + per-panel sign count."""
    a, b = paired(dfa, dfb)
    if not len(a):
        return dict(arm=label, set=set_name, n=0)
    bt = lab.paired_bootstrap(a, b)
    pa = a.groupby('panel').iou.mean(); pb = b.groupby('panel').iou.mean()
    d6 = lab.aggregate(b, 's6') - lab.aggregate(a, 's6')
    return dict(arm=label, set=set_name, n=bt['n_windows'], base=lab.aggregate(a), iou=lab.aggregate(b), diff=bt['diff'], lo=bt['lo'],
                hi=bt['hi'], p_pos=bt['p_pos'], panels_pos=f'{int((pb - pa > 1e-9).sum())}/{len(pa)}', panels_neg=int((pb - pa < -1e-9).sum()),
                d_h6=d6)


# ------------------------------------------------------------------ pseudo-March rows (read-only ogmodel cache)
PM_SETS = ('offlike_m', 'runstart_m')


def pm_load(panel, sname, vt='T'):
    f = f'{OG_PM}/{panel}__{sname}__{vt}.npz'
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=False)
    win = pd.DataFrame(dict(d=z['d'], t=z['t'], cond=lab.ONGOING, fold=0, pers_fill=z['pers_fill'], runstart=z['runstart'], faithful=True,
                            vis=z['vis'], hist_vis=z['hist_vis']))
    rc = types.SimpleNamespace(truth=z['truth'], ev=z['ev'], qlast=z['qlast'], win=win, name=sname, panel=panel, cond=lab.ONGOING, L=int(z['L']))
    return types.SimpleNamespace(X=z['X'], y=z['y'], e=z['e'], meta=z['meta'], rc=rc, W=len(win), L=int(z['L']), cols=[str(c) for c in z['cols']])


def pm_predict(pm, fn, cols):
    ix = [pm.cols.index(c) for c in cols]
    P6 = np.zeros((pm.W, F, pm.L), np.float32)
    P6[pm.meta[:, 0], pm.meta[:, 2] - 1, pm.meta[:, 1]] = fn(np.ascontiguousarray(pm.X[:, ix]))
    return P6


def load_models(arm):
    import lightgbm as lgb
    cfg = json.load(open(f'{MODELS}/{arm}.json'))
    return cfg['spec'], [(g, lgb.Booster(model_file=f'{MODELS}/{arm}__g{g}.txt')) for g in cfg['groups']], cfg['cols']
