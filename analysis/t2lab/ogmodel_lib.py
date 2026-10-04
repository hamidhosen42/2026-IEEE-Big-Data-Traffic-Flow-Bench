"""TRACK ogmodel: model STRUCTURE for the ongoing model (Task 2).  Shared helpers for ogmodel_*.py.

Nothing existing is modified: the F03 training matrix comes from exp_firstcross_lib.train_matrix (same rows, same order,
same weights as the reference arms fc_F03_np255 / fc_F03_id255), scoring is lab.score_probs (production decode), the
bootstrap is lab.paired_bootstrap.  Own outputs: data/t2local/cache/ogmodel/{oof,models,pmarch,rows_noT}, results/ogmodel.

A STRUCTURE = (row partition, target transform, step weights):
  part   'one' (single model) | 'h' (one model per step) | 'hg' (one model per step group) | 'q' (cells queued at T /
         cells free at T, from the feature q_last) | 'hgq' (step group x q_last)
  target 'y' (queued at T+h) | 'diff' (state at T+h differs from the state at T; p(queued) = q_last ? 1 - p : p)
  hw     per-step multiplier of the window weight (h = 1..6)
Every model of a structure uses the same LightGBM parameters as the reference arm (255 leaves, lr 0.05, 500 trees,
min_data_in_leaf 40, seed 0) unless stated; the feature `h` stays in the matrix (constant inside a per-step model).
"""
import gc, json, os, sys, time, types, warnings
import numpy as np
import pandas as pd
import lab
import exp_firstcross_lib as fc

TRACK = 'ogmodel'
RES = f'{lab.ROOT}/analysis/t2lab/results/{TRACK}'
CACHE = f'{lab.CACHE}/{TRACK}'
OOF, MODELS, PM = f'{CACHE}/oof', f'{CACHE}/models', f'{CACHE}/pmarch'
for _d in (RES, CACHE, OOF, MODELS, PM):
    os.makedirs(_d, exist_ok=True)
H, F = lab.H, lab.F
EVALS = ('official', 'offlike', 'pseudo')
SPEC = fc.ARMS['F03']
KEY = ['panel', 'd', 't', 'cond']

STRUCTS = {
    'joint': dict(),
    'hwA': dict(hw=(1, 1, 1, 1, 1.5, 2.0)),
    'hwB': dict(hw=(1, 1, 1, 1.5, 2.0, 3.0)),
    'hwC': dict(hw=(1, 1, 1, 1, 1, 2.0)),
    'perstep': dict(part='h'),
    'hg33': dict(part='hg', groups=((1, 2, 3), (4, 5, 6))),
    'hg42': dict(part='hg', groups=((1, 2, 3, 4), (5, 6))),
    'qsplit': dict(part='q'),
    'diff': dict(target='diff'),
    'hg33q': dict(part='hgq', groups=((1, 2, 3), (4, 5, 6))),
}


def group_ids(spec, X, cols):
    part = spec.get('part', 'one')
    if part == 'one':
        return np.zeros(len(X), np.int8)
    h = X[:, cols.index('h')].astype(np.int8)
    if part == 'h':
        return h - 1
    if part in ('hg', 'hgq'):
        mp = np.zeros(F + 1, np.int8)
        for g, steps in enumerate(spec['groups']):
            for s in steps:
                mp[s] = g
        gid = mp[h]
        if part == 'hgq':
            gid = gid * 2 + (X[:, cols.index('q_last')] > 0.5).astype(np.int8)
        return gid
    if part == 'q':
        return (X[:, cols.index('q_last')] > 0.5).astype(np.int8)
    raise KeyError(part)


def fit_struct(spec, X, y, w, cols, params, n_trees):
    """returns list of (group id, booster).  y is transformed for target='diff'; w is multiplied by the step weights."""
    gid = group_ids(spec, X, cols)
    if spec.get('target', 'y') == 'diff':
        y = np.abs(y - (X[:, cols.index('q_last')] > 0.5).astype(np.float32)).astype(np.float32)
    if spec.get('hw') is not None:
        w = w * np.asarray((0,) + tuple(spec['hw']), np.float32)[X[:, cols.index('h')].astype(np.int8)]
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
    if spec.get('target', 'y') == 'diff':
        q = X[:, cols.index('q_last')] > 0.5
        p = np.where(q, 1.0 - p, p)
    return p


def predict_rc(spec, models, rc, k, cols):
    """P6 [W, 6, L] for the fold-k windows of rc (lab.predict_set with a structure)."""
    P6 = np.zeros((len(rc.win), F, rc.L), np.float32)
    if not rc.n:
        return P6
    wmask = rc.win.fold.to_numpy() == k
    m = rc.keep5[:, k] & wmask[rc.wi]
    if m.any():
        P6[rc.wi[m], rc.h[m] - 1, rc.li[m]] = predict_struct(spec, models, rc.matrix(m, cols, k), cols)
    return P6


def arm_dir(arm):
    return f'{OOF}/{arm}'


def model_files(arm):
    return sorted(f for f in os.listdir(MODELS) if f.startswith(arm + '__g') and f.endswith('.txt'))


def save_models(arm, spec, models, cols, extra=None):
    for g, mdl in models:
        mdl.save_model(f'{MODELS}/{arm}__g{g}.txt')
    json.dump(dict(arm=arm, spec=spec, cols=list(cols), groups=[g for g, _ in models], **(extra or {})), open(f'{MODELS}/{arm}.json', 'w'), indent=1)


def load_models(arm):
    import lightgbm as lgb
    cfg = json.load(open(f'{MODELS}/{arm}.json'))
    spec = cfg['spec']
    if 'groups' in spec:
        spec['groups'] = tuple(tuple(g) for g in spec['groups'])
    return spec, [(g, lgb.Booster(model_file=f'{MODELS}/{arm}__g{g}.txt')) for g in cfg['groups']], cfg['cols']


def run_struct(arm, struct, cols='np', leaves=255, lr=0.05, n_trees=500, mdl=40, folds=(0, 1, 2, 3, 4), seed=0, evals=EVALS,
               spec_override=None):
    """OOF arm of one structure (folds 0..3: held-out day blocks, scored on official / offlike / pseudo; fold 4: final
    models on all train days, saved for the pseudo-March evaluation).  Per-fold files, resumable."""
    spec = dict(STRUCTS[struct]) if spec_override is None else dict(spec_override)
    cols_ = lab.colset(cols) if isinstance(cols, str) else list(cols)
    odir = arm_dir(arm); os.makedirs(odir, exist_ok=True)
    params = lab.lgb_params(num_leaves=leaves, learning_rate=lr, min_data_in_leaf=mdl, seed=seed)
    for k in folds:
        ff = f'{odir}/fold{k}.pkl'
        if k == 4 and os.path.exists(f'{MODELS}/{arm}.json'):
            lab.log(f'{arm} final models: found on disk, skipped'); continue
        if k < 4 and os.path.exists(ff):
            lab.log(f'{arm} fold {k}: found on disk, skipped'); continue
        t1 = time.time()
        X, y, w, info = fc.train_matrix(SPEC['train'], k, cols_, SPEC['wfn'])
        nrow = len(X)
        models = fit_struct(spec, X, y, w, cols_, params, n_trees)
        del X, y, w; gc.collect()
        lab.log(f'{arm} fold {k}: {nrow} rows x {len(cols_)} cols, {len(models)} model(s) in {time.time() - t1:.0f}s')
        if k == 4:
            save_models(arm, spec, models, cols_, dict(leaves=leaves, lr=lr, n_trees=n_trees, seed=seed, nrow=nrow))
            del models; gc.collect(); continue
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
        pd.to_pickle(dict(df=df, probs=probs, nrow=nrow, secs=time.time() - t1), ff)
        lab.log(f'{arm} fold {k}: ' + ', '.join(f'{s} {lab.aggregate(df[df.set == s]):.4f} (n={int((df.set == s).sum())})' for s in evals))
        del models; gc.collect()
    json.dump(dict(arm=arm, struct=struct, spec=spec, cols=cols_, leaves=leaves, lr=lr, n_trees=n_trees, mdl=mdl, seed=seed),
              open(f'{odir}/config.json', 'w'), indent=1, default=str)
    return merge(arm)


def merge(arm):
    odir = arm_dir(arm); dfs = []; P = {}
    for k in range(4):
        ff = f'{odir}/fold{k}.pkl'
        if not os.path.exists(ff):
            continue
        z = pd.read_pickle(ff); dfs.append(z['df'])
        for (s, panel), (widx, p6) in z['probs'].items():
            if (s, panel) not in P:
                rc = lab.rows(panel, s, lab.ONGOING)
                P[(s, panel)] = dict(P6=np.zeros((len(rc.win), F, rc.L), np.float16), done=np.zeros(len(rc.win), bool))
            P[(s, panel)]['P6'][widx] = p6; P[(s, panel)]['done'][widx] = True
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df.to_pickle(f'{odir}/results.pkl')
    for (s, panel), v in P.items():
        np.savez_compressed(f'{odir}/probs__{s}__{panel}.npz', **v)
    return df


def load_arm(arm):
    """per-window frame of an ogmodel arm, or of a lab arm (cache/oof/<arm>) when no ogmodel arm of that name exists."""
    f = f'{arm_dir(arm)}/results.pkl'
    return pd.read_pickle(f) if os.path.exists(f) else lab.load_arm(arm)


def load_probs(arm, set_name, panel):
    f = f'{arm_dir(arm)}/probs__{set_name}__{panel}.npz'
    if os.path.exists(f):
        z = np.load(f); return z['P6'].astype(np.float32), z['done']
    return lab.load_probs(arm, set_name, panel)


# ------------------------------------------------------------------ statistics
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
    return dict(arm=label, set=set_name, n=bt['n_windows'], base=lab.aggregate(a), iou=lab.aggregate(b), diff=bt['diff'], lo=bt['lo'], hi=bt['hi'],
                p_pos=bt['p_pos'], panels_pos=f'{int((pb - pa > 1e-9).sum())}/{len(pa)}', panels_neg=int((pb - pa < -1e-9).sum()), d_h6=d6)


def score_blend(parts, set_name, name, step_w=None, folds=None):
    """score a blend of cached arms on one evaluation set.  parts = [(arm, weight)], or with step_w = {arm: [w1..w6]} a
    per-step weight (weights are normalised per step)."""
    out = []
    for panel in lab.PANELS:
        rc = lab.rows(panel, set_name, lab.ONGOING)
        if not len(rc.win):
            continue
        num = 0; den = 0; done = np.ones(len(rc.win), bool)
        for a, wt in parts:
            p, dn = load_probs(a, set_name, panel); done &= dn
            ws = np.asarray(step_w[a], np.float32)[None, :, None] if step_w and a in step_w else np.float32(wt)
            num = num + ws * p; den = den + ws
        P6 = (num / den).astype(np.float32)
        if folds is not None:
            done &= np.isin(rc.win.fold.to_numpy(), list(folds))
        widx = np.where(done)[0]
        if len(widx):
            out.append(lab.score_probs(rc, P6, widx, arm=name, set_name=set_name))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ------------------------------------------------------------------ pseudo-March (copied from exp_firstcross_pmarch.py)
def val_panel(panel):
    P = lab.load_panel(panel)
    z = np.load(P['npz'], allow_pickle=False)
    days = np.array([str(x) for x in z['validation_days']]); o = np.argsort(days)
    V = z['validation_Vm'][o].astype(np.float32); Fq = z['validation_Fm'][o].astype(np.float32); Em = z['validation_Em'][o]
    E = np.isfinite(V); Q = E & (V <= P['cut'][None, None, :])
    Pv = dict(panel=panel, links=P['links'], li=P['li'], cut=P['cut'], vf=P['vf'], cap=P['cap'], lanes=P['lanes'], length=P['length'],
              days=list(days[o]), dow=pd.to_datetime(pd.Series(days[o])).dt.dayofweek.to_numpy(), V=V, Fq=Fq, E=E, Q=Q, Em=Em,
              ON=z['validation_ON'][o].astype(np.float32), OFF=z['validation_OFF'][o].astype(np.float32))
    return P, Pv


def select(Pv):
    """faithful ongoing candidates / production-rule onset candidates on the masked validation month; blank-touching windows dropped."""
    from exp_firstcross_prod_snippet import fill_time36
    V, E, Q, Em = Pv['V'], Pv['E'], Pv['Q'], Pv['Em']; D, _, L = V.shape
    VF = fill_time36(V); MF = np.isfinite(VF); QF = MF & (VF <= Pv['cut'][None, None, :])
    blank = (Em.sum(2) >= 5) & (E.sum(2) == 0)
    og = np.zeros((D, 288), bool); on = og.copy(); pers = np.full((D, 288), np.nan); stable = og.copy()
    for d in range(D):
        for t in range(H, 288 - F):
            if blank[d, t - H:t + F + 1].any() or Em[d, t - H:t].mean() < 0.7:
                continue
            if lab.established(Q[d, t - H:t]):
                Tf = QF[d, t + 1:t + 1 + F]; Mf = MF[d, t + 1:t + 1 + F]
                if not Tf.any():
                    continue
                S = np.repeat(QF[d, t][None], F, 0)
                u = ((Tf | S) & Mf).sum(); i = (Tf & S & Mf).sum()
                pers[d, t] = 1.0 if u == 0 else i / u
                og[d, t] = pers[d, t] <= 0.9; stable[d, t] = pers[d, t] > 0.9
            else:
                on[d, t] = Q[d, t + 1:t + 1 + F].any()
    return on, og, stable, pers, blank


def val_rows(P, Pv, wins, prior, use_T=True):
    """production-shaped rows of pseudo-March windows (features from the masked layer; use_T=False: no slot-T value)."""
    d = np.array([w[0] for w in wins]); t = np.array([w[1] for w in wins])
    idx = t[:, None] + np.arange(-H, 0)[None]
    Vh = Pv['V'][d[:, None], idx]; Fh = Pv['Fq'][d[:, None], idx]
    VT = Pv['V'][d, t] if use_T else np.full((len(d), Pv['V'].shape[2]), np.nan, np.float32)
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        Fe, qlast = lab.window_features(P, Vh, Fh, t, Pv['dow'][d], prior, VT)
        lab.add_ramp(Fe, Pv['ON'], Pv['OFF'], d, t, P['cap'])
    W, L = qlast.shape
    R = dict(d=d, t=t, cond=np.array([lab.ONGOING] * W), qlast=qlast, W=W, L=L, rows=[])
    for h in range(1, F + 1):
        feats = dict(Fe); feats['h'] = np.full((W, L), h, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = lab.prior_feats(prior, Pv['dow'][d], t, h)
        R['rows'].append((h, feats, Pv['Q'][d, np.minimum(t + h, 287)], Pv['E'][d, np.minimum(t + h, 287)]))
    return R


PM_SETS = ('offlike_m', 'runstart_m')


def pm_file(panel, sname, vt):
    return f'{PM}/{panel}__{sname}__{vt}.npz'


def pm_load(panel, sname, vt='T'):
    """cached pseudo-March rows: namespace(X [n, all cols], y, e, meta [n, 3] (window, link, step), rc (for lab.score_probs))."""
    f = pm_file(panel, sname, vt)
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=False)
    win = pd.DataFrame(dict(d=z['d'], t=z['t'], cond=lab.ONGOING, fold=0, pers_fill=z['pers_fill'], runstart=z['runstart'], faithful=True,
                            vis=z['vis'], hist_vis=z['hist_vis']))
    rc = types.SimpleNamespace(truth=z['truth'], ev=z['ev'], qlast=z['qlast'], win=win, name=sname, panel=panel, cond=lab.ONGOING, L=int(z['L']))
    return types.SimpleNamespace(X=z['X'], y=z['y'], e=z['e'], meta=z['meta'], rc=rc, W=len(win), L=int(z['L']), cols=[str(c) for c in z['cols']])


def pm_predict(pm, fn, cols):
    """P6 [W, 6, L] from fn(X[:, cols]) on cached pseudo-March rows."""
    ix = [pm.cols.index(c) for c in cols]
    P6 = np.zeros((pm.W, F, pm.L), np.float32)
    P6[pm.meta[:, 0], pm.meta[:, 2] - 1, pm.meta[:, 1]] = fn(np.ascontiguousarray(pm.X[:, ix]))
    return P6
