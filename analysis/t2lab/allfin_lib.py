"""TRACK allfin: Task 2 FEATURES from ALL finite released values (eligible + ineligible-but-valued cells).

Today's pipeline drops every cell with is_score_eligible != 1.  About 7 % of all mainline cells are ineligible but carry a
speed / flow value (pct_observed 1-74); they are exported for local work to data/t2local/x/t2x_<PANEL>.npz.
Here only the FEATURE inputs change:
    V_all = where(eligible, V, Vx), F_all likewise                               (60-min history [T-60, T))
    slot T: eligible cells keep the production visibility draw (rng(len(wins)), 62 %),
            ineligible-valued cells are always visible (they are never hidden by the Task-1 mask)
Labels, E, Q, the priors, the window sets and the scoring cells are the production ones (eligible cells), so every comparison
is paired on identical windows.  Production feature code is called, not rewritten (lab.window_features / lab.add_ramp /
lab.to_matrix / lab.ongoing_keep / lab.prior_*).  Nothing in lab.py or its caches is modified: own caches live in
data/t2local/cache/allfin/, results in analysis/t2lab/results/allfin/.
"""
import fcntl, gc, json, os, time, warnings
import numpy as np
import pandas as pd
import lab
import exp_firstcross_lib as fc

H, F = lab.H, lab.F
ONSET, ONGOING = lab.ONSET, lab.ONGOING
XDIR = f'{lab.DATA}/x'
AF = f'{lab.CACHE}/allfin'
RES = f'{lab.ROOT}/analysis/t2lab/results/allfin'
for _d in (AF, RES, f'{AF}/rows', f'{AF}/stats', f'{AF}/oof', f'{AF}/onset'):
    os.makedirs(_d, exist_ok=True)
P_VIS = 0.62
PRIOR3 = lab.PRIOR3
# (lab window set, condition) pairs that get an all-finite row cache
SETS = [('official', ONGOING), ('offlike', ONGOING), ('pseudo', ONGOING), ('runstart', ONGOING), ('faith3', ONGOING),
        ('official', ONSET), ('offlike', ONSET), ('pseudo', ONSET), ('dense1500', ONSET)]


# ------------------------------------------------------------------ cubes
def feat_cubes(P, mode):
    """(VF, FF, Vx): speed / flow cubes used for FEATURES.  mode 'elig' = production cubes; 'all' = eligible + ineligible-valued."""
    if mode == 'elig':
        return P['V'], P['Fq'], None
    if 'V_all' not in P:
        z = np.load(f'{XDIR}/t2x_{P["panel"]}.npz', allow_pickle=False)
        assert [str(x) for x in z['links']] == P['links'], 'link order differs'
        assert [str(x) for x in z['train_days']] == P['days'], 'day order differs'
        Vx = z['train_Vx'].astype(np.float32); Fx = z['train_Fx'].astype(np.float32)
        assert not (np.isfinite(Vx) & P['E']).any(), 'ineligible-valued cells overlap eligible cells'
        P['Vx'] = Vx
        P['V_all'] = np.where(P['E'], P['V'], Vx); P['F_all'] = np.where(P['E'], P['Fq'], Fx)
    return P['V_all'], P['F_all'], P['Vx']


def priors5(P):
    if '_priors5' not in P:
        D = len(P['days'])
        P['_priors5'] = [lab.prior_table(P, P['block'] != k) for k in range(4)] + [lab.prior_table(P, np.ones(D, bool))]
    return P['_priors5']


# ------------------------------------------------------------------ row builder (copy of production make_rows / lab._build_rows)
def make_rows_af(P, wins, prior, mode='all', sub=None, with_labels=True):
    """Production make_rows with the feature cubes of `mode`.  The slot-T visibility draw is the production draw of the FULL
    list `wins` (rng(len(wins)), 62 % of the cells); `sub` (sorted index array) then restricts the windows, so a subset keeps
    exactly the draw it has in the full production call.  Labels / evaluation flags always come from the eligible cube."""
    d = np.array([w[0] for w in wins]); t = np.array([w[1] for w in wins]); cond = np.array([w[2] for w in wins])
    L = P['V'].shape[2]
    rng = np.random.default_rng(len(wins))
    draw = rng.random((len(d), L)) < P_VIS
    if sub is not None:
        d, t, cond, draw = d[sub], t[sub], cond[sub], draw[sub]
    VF, FF, Vx = feat_cubes(P, mode)
    idx = t[:, None] + np.arange(-H, 0)[None]
    Vh = VF[d[:, None], idx]; Fh = FF[d[:, None], idx]
    VT = np.where(draw, P['V'][d, t], np.nan)                    # eligible cells: production simulation of the masked layer
    if mode != 'elig':
        VT = np.where(P['E'][d, t], VT, Vx[d, t])                # ineligible-valued cells: always visible in the masked layer
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
        lab_ = P['Q'][d, np.minimum(t + h, 287)] if with_labels else None
        ev = P['E'][d, np.minimum(t + h, 287)] if with_labels else None
        rows.append((h, feats, lab_, ev))
    return dict(d=d, t=t, cond=cond, rows=rows, qlast=qlast, W=W, L=L)


def build_rows_af(P, win, cond, mode='all', sub=None):
    """lab._build_rows with make_rows_af.  win = the full window frame of (set, cond) as lab.rows uses it.
    Returns X, aux, cols, kstats; aux additionally holds yx / ex = label / presence of an INELIGIBLE-valued cell at the target."""
    wins = list(zip(win.d.tolist(), win.t.tolist(), win.cond.tolist()))
    priors = priors5(P)
    R = make_rows_af(P, wins, priors[4], mode, sub)
    d, t = R['d'], R['t']; dw = P['dow'][d]
    keeps = []
    for k in range(5):
        if cond == ONGOING:
            f0 = dict(R['rows'][0][1]); f0['prior_now'] = lab.prior_feats(priors[k], dw, t, 1)[1]
            keeps.append(lab.ongoing_keep(dict(R, rows=[(1, f0, None, None)])))
        else:
            keeps.append(np.ones((R['W'], R['L']), bool))
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
    _, _, Vx = feat_cubes(P, 'all')
    vx = Vx[d[wi_], np.minimum(t[wi_] + h_, 287), li_]
    ex = np.isfinite(vx); yx = ex & (np.nan_to_num(vx, nan=1e9) <= P['cut'][li_])
    aux = dict(y=y.astype(bool), e=e.astype(bool), wi=wi_.astype(np.int32), li=li_.astype(np.int16), h=h_.astype(np.int8),
               prior5=prior5, keep5=keep5, truth=truth, ev=ev, qlast=R['qlast'], yx=yx, ex=ex)
    # keep coverage under the all-train prior (k = 4), as the production log line "ongoing keep covers ... of labeled queued cells"
    f1 = R['rows'][0][1]
    kst = dict(windows=int(R['W']), cells=int(R['W'] * R['L']), kept_share=float(keeps[4].mean()),
               pos=int((truth & ev).sum()), pos_in_keep=int((truth & ev & keeps[4][:, None, :]).sum()),
               has_T=float(np.asarray(f1['has_T']).mean()), miss_last=float(np.asarray(f1['miss_last']).mean()),
               vr_last_nan=float(np.isnan(np.asarray(f1['vr_last'])).mean()), q_last=float(np.asarray(f1['q_last']).mean()),
               q_count=float(np.asarray(f1['q_count']).mean()))
    return X, aux, cols, kst


def f03_sub(win):
    """faith3 windows used by the F03 recipe: non-run-starts, at most fc.CAP per panel, rng(0) (= fc.wfn_F weights > 0)."""
    return fc._cap_idx(np.where(~win.runstart.to_numpy())[0], fc.CAP)


def af_name(name):
    return 'faith3F03' if name == 'faith3' else name


def af_path(panel, name, cond):
    return f'{AF}/rows/{panel}__{af_name(name)}__{cond}'


def rows_af(panel, name, cond=ONGOING):
    """all-finite RowCache (same class / layout as lab.rows).  'faith3' / 'faith3F03' = the F03 subset of faith3."""
    return lab.RowCache(af_path(panel, name, cond))


def rows_any(mode):
    return (lambda p, s, c=ONGOING: lab.rows(p, 'faith3' if s == 'faith3F03' else s, c)) if mode == 'elig' else rows_af


def parity(panel, name, cond, sub, X, aux, cols):
    """compare an eligible-only build of MY builder with the cached production rows (lab.rows)."""
    rc = lab.rows(panel, name, cond)
    out = dict(cols_equal=list(cols) == list(rc.cols))
    if sub is None:
        r = np.arange(rc.n); wi_ref = rc.wi; ws = np.arange(len(rc.win))
    else:
        remap = np.full(len(rc.win), -1, np.int64); remap[sub] = np.arange(len(sub))
        r = np.where(remap[rc.wi] >= 0)[0]; wi_ref = remap[rc.wi[r]]; ws = np.asarray(sub)
    out.update(n_ref=int(len(r)), n_mine=int(len(X)))
    if len(r) != len(X):
        out.update(exact=False, max_abs_diff=np.nan, aux_equal=False)
        return out
    Xr = np.asarray(rc.X[r]) if len(r) else np.zeros((0, len(rc.cols)), np.float32)
    exact = bool(np.array_equal(Xr, X, equal_nan=True))
    if exact:
        mad = 0.0
    else:
        nn = np.isnan(Xr) != np.isnan(X)
        mad = float('inf') if nn.any() else float(np.nanmax(np.abs(Xr - X)))
    ok = [np.array_equal(aux['y'], rc.y[r]), np.array_equal(aux['e'], rc.e[r]), np.array_equal(aux['wi'], wi_ref),
          np.array_equal(aux['li'], rc.li[r]), np.array_equal(aux['h'], rc.h[r]),
          np.array_equal(aux['prior5'], rc.prior5[r], equal_nan=True), np.array_equal(aux['keep5'], rc.keep5[r]),
          np.array_equal(aux['truth'], rc.truth[ws]), np.array_equal(aux['ev'], rc.ev[ws]), np.array_equal(aux['qlast'], rc.qlast[ws])]
    out.update(exact=exact, max_abs_diff=mad, aux_equal=bool(all(ok)))
    return out


def build_one(panel, name, cond, check=True):
    """build (and cache) the all-finite rows of one (panel, set, cond); with check=True also build the eligible-only rows
    with the same builder and compare them with lab.rows.  Stats go to cache/allfin/stats/<...>.json."""
    path = af_path(panel, name, cond); sf = f'{AF}/stats/{panel}__{af_name(name)}__{cond}.json'
    if os.path.exists(f'{path}/meta.json') and os.path.exists(sf):
        return json.load(open(sf))
    t1 = time.time()
    P = lab.load_panel(panel)
    win = lab.window_set(panel, name); win = win[win.cond == cond].reset_index(drop=True)
    st = dict(panel=panel, set=af_name(name), cond=cond, windows_full=len(win))
    if not len(win):
        return st
    sub = f03_sub(win) if name == 'faith3' else None
    if check:
        Xe, auxe, cols, ke = build_rows_af(P, win, cond, 'elig', sub)
        st.update({f'par_{k}': v for k, v in parity(panel, name, cond, sub, Xe, auxe, cols).items()})
        st.update({f'elig_{k}': v for k, v in ke.items()}); st['elig_rows_e'] = int((auxe['e'] & auxe['keep5'][:, 4]).sum())
        del Xe, auxe; gc.collect()
    Xa, auxa, cols, ka = build_rows_af(P, win, cond, 'all', sub)
    st.update({f'all_{k}': v for k, v in ka.items()}); st['all_rows_e'] = int((auxa['e'] & auxa['keep5'][:, 4]).sum())
    st['all_rows_ex'] = int((auxa['ex'] & auxa['keep5'][:, 4]).sum())
    winS = win if sub is None else win.iloc[sub].reset_index(drop=True)
    if not os.path.exists(f'{path}/meta.json'):
        tmp = f'{path}.tmp{os.getpid()}'
        os.makedirs(tmp, exist_ok=True)
        np.save(f'{tmp}/X.npy', Xa); np.savez_compressed(f'{tmp}/aux.npz', **auxa); winS.to_pickle(f'{tmp}/win.pkl')
        json.dump(dict(cols=list(cols), panel=panel, name=af_name(name), cond=cond, L=len(P['links']), n=int(len(Xa)), mode='all',
                       prod_sha1=lab.prod.__source_sha1__), open(f'{tmp}/meta.json', 'w'))
        os.rename(tmp, path)
    st['secs'] = round(time.time() - t1, 1)
    json.dump(st, open(sf, 'w'))
    del Xa, auxa; gc.collect()
    return st


def ensure_built(panels=None, sets=None, check=True):
    """build every missing cache under one process-wide file lock (parallel jobs wait instead of building twice)."""
    out = []
    with open(f'{AF}/build.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for panel in panels or lab.PANELS:
            todo = [(n, c) for n, c in (sets or SETS)
                    if not (os.path.exists(f'{af_path(panel, n, c)}/meta.json') and os.path.exists(f'{AF}/stats/{panel}__{af_name(n)}__{c}.json'))]
            for n, c in sets or SETS:
                st = build_one(panel, n, c, check)
                out.append(st)
            if todo:
                lab.log(f'allfin build {panel}: {len(todo)} cache(s) built')
                lab.drop_panel(panel)
    return pd.DataFrame(out)


def load_stats():
    import glob
    return pd.DataFrame([json.load(open(f)) for f in sorted(glob.glob(f'{AF}/stats/*.json'))])


# ------------------------------------------------------------------ ongoing: F03 recipe on either feature mode
def spec(mode):
    """F03 = non-run-start faithful windows on the t % 3 grid (<= 1500 per panel) @0.3 + all faithful run-starts @1.0."""
    if mode == 'elig':
        return dict(train=[('faith3', 1.0), ('runstart', 1.0)], wfn=fc.wfn_F(0.3), rows=rows_any('elig'))
    return dict(train=[('faith3F03', 0.3), ('runstart', 1.0)], wfn=None, rows=rows_af)


def train_matrix(sp, k, cols, cond=ONGOING, xlabels=False):
    """rows / order as fc.train_matrix (panel outer, set inner; e & keep5[:, k] & fold != k & weight > 0), preallocated.
    xlabels=True additionally uses rows whose target cell is ineligible-valued (label = Vx <= cut)."""
    parts = []; n = 0; info = {}
    for panel in lab.PANELS:
        for name, sw in sp['train']:
            rc = sp['rows'](panel, name, cond)
            if not rc.n:
                continue
            ww = np.full(len(rc.win), float(sw), np.float32)
            if sp.get('wfn') is not None:
                ww = ww * np.asarray(sp['wfn'](rc.win, name, panel), np.float32)
            ok, y = rc.e, rc.y
            if xlabels:
                z = np.load(f'{rc.path}/aux.npz'); ok = rc.e | z['ex']; y = np.where(rc.e, rc.y, z['yx'])
            m = ok & rc.keep5[:, k] & (rc.fold != k) & (ww[rc.wi] > 0)
            c = int(m.sum()); parts.append((rc, m, ww, c, y)); n += c
            a = info.setdefault(name, dict(windows=0, rows=0))
            a['windows'] += int(((rc.win.fold.to_numpy() != k) & (ww > 0)).sum()); a['rows'] += c
    X = np.empty((n, len(cols)), np.float32); y = np.empty(n, np.float32); w = np.empty(n, np.float32)
    o = 0
    for rc, m, ww, c, yy in parts:
        X[o:o + c] = rc.matrix(m, cols, k); y[o:o + c] = yy[m]; w[o:o + c] = ww[rc.wi[m]]; o += c
    return X, y, w, info


def arm_dir(arm):
    return f'{AF}/oof/{arm}'


def run_arm(arm, mode, cols='np', folds=(0, 1, 2, 3), evals=fc.EVALS, extra_eval=(), leaves=255, lr=0.05, n_trees=500, xlabels=False):
    """OOF LightGBM arm (per-fold persistence).  mode = feature mode of the TRAINING rows; the held-out windows are predicted
    with the rows of `mode` and of every mode in extra_eval (cross evaluation: e.g. eligible-trained model on all-finite rows)."""
    cols_ = lab.colset(cols); odir = arm_dir(arm); os.makedirs(odir, exist_ok=True); sp = spec(mode)
    for k in folds:
        ff = f'{odir}/fold{k}.pkl'
        if os.path.exists(ff):
            lab.log(f'{arm} fold {k}: found on disk, skipped'); continue
        t1 = time.time()
        if k == 4 and os.path.exists(f'{odir}/model_f4.txt'):
            continue
        X, y, w, info = train_matrix(sp, k, cols_, xlabels=xlabels)
        nrow, pos = len(X), float(y.mean())
        p = lab.lgb_params(num_leaves=leaves, learning_rate=lr, min_data_in_leaf=40, seed=0)
        mdl = fc.fit(X, y, w, p, n_trees, cols_)
        del X, y, w; gc.collect()
        mdl.save_model(f'{odir}/model_f{k}.txt')
        lab.log(f'{arm} fold {k}: {nrow} rows x {len(cols_)} cols, pos {pos:.3f}, trained in {time.time() - t1:.0f}s | {json.dumps(info)}')
        if k == 4:                                               # final model on all train days: nothing to evaluate here
            continue
        out = {}
        for em in (mode,) + tuple(extra_eval):
            res = []; probs = {}
            for s in evals:
                for panel in lab.PANELS:
                    rc = rows_any(em)(panel, s, ONGOING)
                    widx = np.where(rc.win.fold.to_numpy() == k)[0]
                    if not len(widx):
                        continue
                    P6 = lab.predict_set([mdl], rc, k, cols_)
                    res.append(lab.score_probs(rc, P6, widx, arm=arm if em == mode else f'{arm}@{em}', set_name=s))
                    probs[(s, panel)] = (widx, P6[widx].astype(np.float16))
            df = pd.concat(res, ignore_index=True); out[em] = (df, probs)
            lab.log(f'{arm} fold {k} [{em} rows]: ' + ', '.join(f'{s} {lab.aggregate(df[df.set == s]):.4f} (n={int((df.set == s).sum())})' for s in evals))
        pd.to_pickle(dict(out=out, info=info, nrow=nrow, pos=pos, secs=time.time() - t1), ff)
        del mdl; gc.collect()
    return load_arm(arm, mode)


def load_arm(arm, em):
    dfs = []
    for k in range(4):
        ff = f'{arm_dir(arm)}/fold{k}.pkl'
        if os.path.exists(ff):
            z = pd.read_pickle(ff)
            if em in z['out']:
                dfs.append(z['out'][em][0])
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_probs(arm, em, set_name, panel):
    rc = rows_any(em)(panel, set_name, ONGOING)
    P6 = np.zeros((len(rc.win), F, rc.L), np.float32); done = np.zeros(len(rc.win), bool)
    for k in range(4):
        ff = f'{arm_dir(arm)}/fold{k}.pkl'
        if os.path.exists(ff):
            z = pd.read_pickle(ff)
            if em in z['out'] and (set_name, panel) in z['out'][em][1]:
                widx, p6 = z['out'][em][1][(set_name, panel)]
                P6[widx] = p6; done[widx] = True
    return rc, P6, done


def blend_frame(arms, em, set_name, name):
    """score the mean probability of several allfin arms (same feature mode `em` at prediction time)."""
    out = []
    for panel in lab.PANELS:
        P6 = 0; done = None; rc = None
        for a in arms:
            rc, p, dn = load_probs(a, em, set_name, panel)
            P6 = P6 + p / len(arms); done = dn if done is None else done & dn
        widx = np.where(done)[0]
        if len(widx):
            out.append(lab.score_probs(rc, P6.astype(np.float32), widx, arm=name, set_name=set_name))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ------------------------------------------------------------------ statistics
KEY = ['panel', 'd', 't', 'cond']


def compare(dfa, dfb, label=''):
    """paired B - A on the windows present in both frames (one evaluation set): lab.paired_bootstrap + per-panel signs."""
    a, b = fc.paired(dfa, dfb)
    if not len(a):
        return dict(label=label, n=0)
    bt = lab.paired_bootstrap(a, b)
    pa = a.groupby('panel').iou.mean(); pb = b.groupby('panel').iou.mean(); dl = pb - pa
    dw = b.iou.to_numpy() - a.iou.to_numpy()
    return dict(label=label, n=bt['n_windows'], base=lab.aggregate(a), new=lab.aggregate(b), diff=bt['diff'], lo=bt['lo'], hi=bt['hi'],
                p_pos=bt['p_pos'], panels_pos=int((dl > 1e-9).sum()), panels_neg=int((dl < -1e-9).sum()), panels=len(dl),
                win_up=int((dw > 1e-9).sum()), win_down=int((dw < -1e-9).sum()))


def fmt(r):
    if not r.get('n'):
        return f'{r["label"]}: n=0'
    return (f'{r["label"]}: n={r["n"]} base {r["base"]:.4f} new {r["new"]:.4f} diff {r["diff"]:+.4f} [{r["lo"]:+.4f}, {r["hi"]:+.4f}] '
            f'p_pos {r["p_pos"]:.3f} panels +{r["panels_pos"]}/-{r["panels_neg"]} of {r["panels"]} windows up/down {r["win_up"]}/{r["win_down"]}')
