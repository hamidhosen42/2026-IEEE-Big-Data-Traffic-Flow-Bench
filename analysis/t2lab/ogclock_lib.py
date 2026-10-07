"""TRACK ogclock: same-day context ("episode clock") features for the ONGOING F03 model.

Inputs of a window with origin T (all timestamped <= T):
  slots 0 .. T-13  the same-day MASKED layer (released windows: validation_Vm / private_Vm; train rows: the train cube thinned
                   i.i.d. per cell with a per-day visible share drawn from the 488 measured released days, about 0.5 / 0.7 / 0.8)
  slots T-12..T-1  the raw window history (exactly the production Vh)
  slot  T          the Task-1-masked slot-T value (exactly the production VT draw, rng(len(wins)), 62 %)
Feature groups (per (window, link); corridor values are broadcast over the links):
  link  ck_run    slots of the current queued run up to T (latest observed state queued; one isolated free observation inside a
                  run is closed), 0 if the latest observed state is free
        ck_first  slots since the link's first queued observation today (NaN: none)
        ck_clear  slots since the link's last queued -> free transition today (NaN: none)
        ck_frac   share of today's observed slots that were queued
        ck_minr   minimum speed ratio V / v_cut today
        ck_nep    number of queued episodes started today
  corr  cc_first  slots since the first slot today with >= 2 links queued (latest observed state per link; NaN: none)
        cc_peak, cc_peak_sh   peak count of queued links so far today (count, share of links)
        cc_since_peak         slots since the (latest) peak
        cc_rel                count at T / peak
        cc_d6, cc_d12, cc_d24 count at T minus count 30 / 60 / 120 min earlier
        cc_age    slots since the corridor was last free for 3 consecutive slots (0 if no link is queued at T)
        cc_area   queued link-slots today / number of links
Cache: data/t2local/cache/ogclock/fe_<PANEL>__<set>.npy [W, L, K] float32 aligned with lab.rows(panel, set).win
"""
import gc, json, os, sys, time, types, warnings
import numpy as np
import pandas as pd
import lab
import exp_firstcross_lib as fc

H, F = lab.H, lab.F
ODIR = f'{lab.CACHE}/ogclock'
RES = f'{lab.ROOT}/analysis/t2lab/results/ogclock'
for _d in (ODIR, RES, f'{ODIR}/oof'):
    os.makedirs(_d, exist_ok=True)
CK_LINK = ['ck_run', 'ck_first', 'ck_clear', 'ck_frac', 'ck_minr', 'ck_nep']
CK_CORR = ['cc_first', 'cc_peak', 'cc_peak_sh', 'cc_since_peak', 'cc_rel', 'cc_d6', 'cc_d12', 'cc_d24', 'cc_age', 'cc_area']
CK_ALL = CK_LINK + CK_CORR


# ============================================================================================ kernel-portable block
def ck_ffill(a):
    """forward fill NaN along axis 1 of [W, S, L]."""
    out = a.copy()
    for i in range(1, a.shape[1]):
        m = np.isnan(out[:, i]); out[:, i][m] = out[:, i - 1][m]
    return out


def ck_bfill(a):
    out = a.copy()
    for i in range(a.shape[1] - 2, -1, -1):
        m = np.isnan(out[:, i]); out[:, i][m] = out[:, i + 1][m]
    return out


def ck_features(Rr, T):
    """Rr [W, S, L] speed ratio V / v_cut of the day so far (NaN = not visible; every slot > T must be NaN); T [W] origin slots.
    Returns dict name -> [W, L] float32 (CK_ALL order)."""
    Rr = np.asarray(Rr, np.float32); T = np.asarray(T, np.int64)
    W, S, L = Rr.shape
    Tf = T.astype(np.float32)[:, None]
    S3 = np.arange(S)[None, :, None]
    vis = np.isfinite(Rr)
    q = vis & (np.nan_to_num(Rr, nan=9.0) <= 1.0)
    st = np.where(vis, q.astype(np.float32), np.nan).astype(np.float32)
    f = ck_ffill(st); b = ck_bfill(st)
    nanc = np.full((W, 1, L), np.nan, np.float32)
    prev = np.concatenate([nanc, f[:, :-1]], 1); nxt = np.concatenate([b[:, 1:], nanc], 1)
    q2 = q | (vis & ~q & (prev == 1) & (nxt == 1))                       # isolated free observation inside a queued run: closed
    del f, b, prev, nxt
    st2 = np.where(vis, q2.astype(np.float32), np.nan).astype(np.float32)
    f2 = ck_ffill(st2)
    prev2 = np.concatenate([nanc, f2[:, :-1]], 1)
    out = {}
    nobs = vis.sum(1); nqo = q.sum(1)
    with np.errstate(all='ignore'):
        mn = np.where(vis, Rr, np.inf).min(1)
        anyq = q.any(1); fq = np.argmax(q, 1)
        lfree = np.where(vis & ~q2, S3, -1).max(1)
        fo = np.where(vis.any(1), np.argmax(vis, 1), -1)
        last = f2[:, -1]
        start = np.where(lfree >= 0, lfree, fo - 1)
        ltr = np.where(vis & ~q2 & (prev2 == 1), S3, -1).max(1)
        out['ck_run'] = np.where(last == 1, Tf - start, np.where(np.isnan(last), np.nan, 0.0))
        out['ck_first'] = np.where(anyq, Tf - fq, np.nan)
        out['ck_clear'] = np.where(ltr >= 0, Tf - ltr, np.nan)
        out['ck_frac'] = np.where(nobs > 0, nqo / np.maximum(nobs, 1), np.nan)
        out['ck_minr'] = np.where(np.isfinite(mn), mn, np.nan)
        out['ck_nep'] = (vis & q2 & (prev2 != 1)).sum(1).astype(np.float32)
        # corridor: per slot, the number of links whose latest (closed) observed state is queued
        nqs = (f2 == 1).sum(2).astype(np.float32)                         # [W, S]
        sl = np.arange(S)[None, :]
        valid = sl <= T[:, None]
        ar = np.arange(W)
        nT = nqs[ar, T]
        nqv = np.where(valid, nqs, -1.0)
        peak = nqv.max(1)
        lastpk = S - 1 - np.argmax((nqv == peak[:, None])[:, ::-1], 1)
        m2 = (nqs >= 2) & valid
        first2 = np.where(m2.any(1), np.argmax(m2, 1), -1)
        z = nqs == 0
        zp = np.concatenate([np.ones((W, 2), bool), z], 1)
        z3 = zp[:, 2:] & zp[:, 1:-1] & zp[:, :-2]
        lz = np.where(z3 & valid, sl, -1).max(1)
        m1 = (nqs >= 1) & valid
        first1 = np.where(m1.any(1), np.argmax(m1, 1), 0)
        cc = {'cc_first': np.where(first2 >= 0, T - first2, np.nan), 'cc_peak': peak, 'cc_peak_sh': peak / L,
              'cc_since_peak': (T - lastpk).astype(np.float32), 'cc_rel': nT / np.maximum(peak, 1)}
        for k in (6, 12, 24):
            cc[f'cc_d{k}'] = nT - nqs[ar, np.maximum(T - k, 0)]
        cc['cc_age'] = np.where(nT > 0, np.where(lz >= 0, T - lz, T - first1 + 1), 0.0)
        cc['cc_area'] = np.where(valid, nqs, 0).sum(1) / L
    for k, v in cc.items():
        out[k] = np.repeat(np.asarray(v, np.float32)[:, None], L, 1)
    return {k: np.asarray(out[k], np.float32) for k in CK_ALL}


def ck_compose(Vday, T, Vh, VT, cut, vis_mask=None):
    """day-so-far ratio cube for windows: Vday [W, S, L] speed of the window's day (NaN = not visible) used for slots < T-12
    (optionally thinned by vis_mask [W, S, L] bool), Vh [W, 12, L] raw history for T-12..T-1, VT [W, L] slot-T value."""
    W, S, L = Vday.shape
    s = np.arange(S)[None, :, None]; Tt = np.asarray(T)[:, None, None]
    pre = Vday if vis_mask is None else np.where(vis_mask, Vday, np.nan)
    R = np.where(s < Tt - H, pre, np.nan).astype(np.float32)
    ar = np.arange(W)
    for j in range(H):
        R[ar, np.asarray(T) - H + j] = Vh[:, j]
    R[ar, np.asarray(T)] = VT
    return R / np.asarray(cut, np.float32)[None, None, :]
# ============================================================================================ end of kernel-portable block


# ------------------------------------------------------------------ masked-layer emulation for train rows
def vis_days():
    """per-day visible share of eligible cells (non-blank slots) of every released day (8 panels x validation / private)."""
    f = f'{RES}/vis_days.json'
    if os.path.exists(f):
        return np.array(json.load(open(f)))
    out = []
    for panel in lab.PANELS:
        z = np.load(f'{lab.DATA}/t2data_{panel}.npz', allow_pickle=False)
        for split in ('validation', 'private'):
            Vm = z[f'{split}_Vm']; Em = z[f'{split}_Em']; Vis = np.isfinite(Vm)
            blank = (Em.sum(2) >= 5) & (Vis.sum(2) == 0); nb = ~blank[:, :, None] & Em
            out += [float(Vis[d][nb[d]].mean()) for d in range(len(Vm))]
    json.dump(out, open(f, 'w'))
    return np.array(out)


def thin_cube(P):
    """[D, 288, L] bool: emulated masked-layer visibility of the train days (per-day share drawn from vis_days, i.i.d. cells)."""
    D, S, L = P['V'].shape
    pi = lab.PANELS.index(P['panel'])
    rates = np.random.default_rng(7000 + pi).choice(vis_days(), D, replace=True)
    out = np.zeros((D, S, L), bool)
    for d in range(D):
        out[d] = np.random.default_rng(10 ** 6 * (pi + 1) + d).random((S, L)) < rates[d]
    return out, rates


def train_inputs(P, win):
    """production history / slot-T draw of a lab row cache (identical to the inputs behind lab.rows)."""
    d = win.d.to_numpy(); t = win.t.to_numpy()
    idx = t[:, None] + np.arange(-H, 0)[None]
    Vh = P['V'][d[:, None], idx]; Fh = P['Fq'][d[:, None], idx]
    rng = np.random.default_rng(len(win))
    VT = np.where(rng.random((len(d), P['V'].shape[2])) < 0.62, P['V'][d, t], np.nan)
    return Vh, Fh, VT, d, t


def fe_train(P, THIN, d, t, Vh, VT, chunk=192):
    """[W, L, K] ck features of train windows (masked-layer emulation before T-12)."""
    out = np.empty((len(d), len(P['links']), len(CK_ALL)), np.float32)
    for a in range(0, len(d), chunk):
        sl = slice(a, a + chunk)
        R = ck_compose(P['V'][d[sl]], t[sl], Vh[sl], VT[sl], P['cut'], THIN[d[sl]])
        X = ck_features(R, t[sl])
        out[sl] = np.stack([X[k] for k in CK_ALL], -1)
        del R, X
    return out


def fe_path(panel, set_name):
    return f'{ODIR}/fe_{panel}__{set_name}.npy'


def build_fe(panel, set_name, check=True, refresh=False):
    f = fe_path(panel, set_name)
    if os.path.exists(f) and not refresh:
        return np.load(f, mmap_mode='r')
    rc = lab.rows(panel, set_name, lab.ONGOING)
    P = lab.load_panel(panel)
    THIN, _ = thin_cube(P)
    Vh, Fh, VT, d, t = train_inputs(P, rc.win)
    if check and len(d):
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            Fe, qlast = lab.window_features(P, Vh, Fh, t, P['dow'][d], lab.prior_table(P, np.ones(len(P['days']), bool)), VT)
        assert np.array_equal(qlast, rc.qlast), (panel, set_name, 'qlast differs')
        m = rc.h == 1
        for c in ('vr_T', 'vr_last', 'vr_l11', 'q_count'):
            a_ = np.asarray(Fe[c], np.float32)[rc.wi[m], rc.li[m]]; b_ = np.asarray(rc.X[:, rc.cols.index(c)])[m]
            assert np.array_equal(a_, b_, equal_nan=True), (panel, set_name, c)
    FE = fe_train(P, THIN, d, t, Vh, VT) if len(d) else np.zeros((0, len(P['links']), len(CK_ALL)), np.float32)
    del THIN; gc.collect()
    tmp = f'{f}.{os.getpid()}.tmp.npy'
    np.save(tmp, FE); os.replace(tmp, f)
    return np.load(f, mmap_mode='r')


# ------------------------------------------------------------------ matrices (cached production rows + ck columns)
def cols_spec(kind, xc):
    """production column list (np / id, optional '+tod', '+tod+dow') and extra ck columns."""
    base, *add = kind.split('+')
    pc = lab.colset(base) + [c for c in add if c not in lab.colset(base)]
    return pc, list(xc)


def full_matrix(rc, FE, mask, pc, xc, k):
    Xp = rc.matrix(mask, pc, k)
    if not xc:
        return Xp
    r = np.where(mask)[0]
    ix = np.array([CK_ALL.index(c) for c in xc])
    Xe = np.asarray(FE)[rc.wi[r][:, None], rc.li[r][:, None], ix[None, :]]
    return np.concatenate([Xp, Xe.astype(np.float32)], 1)


def train_matrix(k, pc, xc, spec=None, panels=None):
    """F03 rows (same rows / order / weights as exp_firstcross_lib.train_matrix) with the ck columns appended."""
    spec = spec or fc.ARMS['F03']
    parts = []; n = 0
    for panel in panels or lab.PANELS:
        for name, sw in spec['train']:
            rc = lab.rows(panel, name, lab.ONGOING)
            if not rc.n:
                continue
            ww = np.full(len(rc.win), float(sw), np.float32)
            if spec['wfn'] is not None:
                ww = ww * np.asarray(spec['wfn'](rc.win, name, panel), np.float32)
            m = rc.e & rc.keep5[:, k] & (rc.fold != k) & (ww[rc.wi] > 0)
            c = int(m.sum()); parts.append((panel, name, rc, m, ww, c)); n += c
    X = np.empty((n, len(pc) + len(xc)), np.float32); y = np.empty(n, np.float32); w = np.empty(n, np.float32); o = 0
    for panel, name, rc, m, ww, c in parts:
        FE = build_fe(panel, name) if xc else None
        X[o:o + c] = full_matrix(rc, FE, m, pc, xc, k); y[o:o + c] = rc.y[m]; w[o:o + c] = ww[rc.wi[m]]; o += c
    return X, y, w


def predict_set(models, panel, set_name, k, pc, xc):
    rc = lab.rows(panel, set_name, lab.ONGOING)
    P6 = np.zeros((len(rc.win), F, rc.L), np.float32)
    if not rc.n:
        return rc, P6
    m = rc.keep5[:, k] & (rc.win.fold.to_numpy() == k)[rc.wi]
    if m.any():
        X = full_matrix(rc, build_fe(panel, set_name) if xc else None, m, pc, xc, k)
        ms = models if isinstance(models, (list, tuple)) else [models]
        P6[rc.wi[m], rc.h[m] - 1, rc.li[m]] = sum(mdl.predict(X, num_threads=lab.THREADS) for mdl in ms) / len(ms)
    return rc, P6


def arm_dir(arm):
    return f'{ODIR}/oof/{arm}'


def run_arm(arm, kind='np', xc=CK_ALL, folds=(0, 1, 2, 3), evals=fc.EVALS, leaves=255, lr=0.05, n_trees=500, mdl=40, seed=0):
    """OOF arm with per-fold persistence (resume after interruption); returns the per-window frame of all folds on disk."""
    import lightgbm as lgb
    pc, xc = cols_spec(kind, xc); cols = pc + xc
    od = arm_dir(arm); os.makedirs(od, exist_ok=True)
    for k in folds:
        ff = f'{od}/fold{k}.pkl'
        if os.path.exists(ff):
            lab.log(f'{arm} fold {k}: on disk'); continue
        t1 = time.time()
        X, y, w = train_matrix(k, pc, xc)
        p = lab.lgb_params(num_leaves=leaves, learning_rate=lr, min_data_in_leaf=mdl, seed=seed)
        mdl_ = lgb.train(p, lgb.Dataset(X, y, weight=w, feature_name=cols, free_raw_data=True), n_trees)
        nrow = len(X); del X, y, w; gc.collect()
        imp = dict(zip(cols, mdl_.feature_importance('gain').round(1).tolist()))
        res = []
        for s in evals:
            for panel in lab.PANELS:
                rc, P6 = predict_set(mdl_, panel, s, k, pc, xc)
                widx = np.where(rc.win.fold.to_numpy() == k)[0]
                if len(widx):
                    res.append(lab.score_probs(rc, P6, widx, arm=arm, set_name=s))
        df = pd.concat(res, ignore_index=True)
        pd.to_pickle(dict(df=df, imp=imp, nrow=nrow, secs=time.time() - t1, cols=cols), ff)
        lab.log(f'{arm} fold {k}: {nrow} rows x {len(cols)} cols, {time.time() - t1:.0f}s | ' +
                ', '.join(f'{s} {lab.aggregate(df[df.set == s]):.4f} (n={int((df.set == s).sum())})' for s in evals))
        del mdl_; gc.collect()
    return load_arm(arm)


def load_arm(arm):
    dfs = [pd.read_pickle(f'{arm_dir(arm)}/fold{k}.pkl')['df'] for k in range(4) if os.path.exists(f'{arm_dir(arm)}/fold{k}.pkl')]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_imp(arm):
    out = {}
    for k in range(4):
        f = f'{arm_dir(arm)}/fold{k}.pkl'
        if os.path.exists(f):
            for c, v in pd.read_pickle(f)['imp'].items():
                out[c] = out.get(c, 0) + v
    return pd.Series(out).sort_values(ascending=False)


# ------------------------------------------------------------------ window types (offlike / official)
_WT = {}


def window_type(panel, win):
    """'midrun' (faithful, not a run-start), 'rs_old' (run-start; history at T-5 already established), 'rs_new' (run-start of a
    newly established queue), 'other' (not faithful)."""
    P = lab.load_panel(panel)
    d = win.d.to_numpy(); t = win.t.to_numpy()
    est_prev = np.array([lab.established(P['Q'][a, b - 1 - H:b - 1]) for a, b in zip(d, t)])
    rs = win.runstart.to_numpy().astype(bool); fa = win.faithful.to_numpy().astype(bool)
    return np.where(~fa, 'other', np.where(~rs, 'midrun', np.where(est_prev, 'rs_old', 'rs_new')))


def add_wtype(df):
    """attach the window type to a result frame (sets official / offlike / pseudo)."""
    key = []
    for (s, panel), g in df.groupby(['set', 'panel']):
        if (s, panel) not in _WT:
            win = lab.window_set(panel, s); win = win[win.cond == lab.ONGOING].reset_index(drop=True)
            _WT[(s, panel)] = pd.DataFrame(dict(set=s, panel=panel, d=win.d.to_numpy(), t=win.t.to_numpy(), wtype=window_type(panel, win)))
        key.append(_WT[(s, panel)])
    return df.merge(pd.concat(key), on=['set', 'panel', 'd', 't'], how='left')


# ------------------------------------------------------------------ statistics
def delta_row(A, B, label, set_name):
    a, b = fc.paired(A, B)
    if not len(a):
        return dict(cmp=label, set=set_name, n=0)
    bt = lab.paired_bootstrap(a, b)
    pa = a.groupby('panel').iou.mean(); pb = b.groupby('panel').iou.mean()
    return dict(cmp=label, set=set_name, n=bt['n_windows'], base=lab.aggregate(a), new=lab.aggregate(b), diff=bt['diff'], lo=bt['lo'],
                hi=bt['hi'], p_pos=bt['p_pos'], panels=f'+{int((pb - pa > 1e-12).sum())}/-{int((pb - pa < -1e-12).sum())}',
                d_fn_new=lab.aggregate(b, 'fn_new') - lab.aggregate(a, 'fn_new'), d_fp_old=lab.aggregate(b, 'fp_old') - lab.aggregate(a, 'fp_old'))


# ------------------------------------------------------------------ pseudo-March (masked validation month, new scenario)
PM_SETS = ('offlike_m', 'runstart_m')


def val_panel(panel):
    P = lab.load_panel(panel)
    z = np.load(P['npz'], allow_pickle=False)
    days = np.array([str(x) for x in z['validation_days']]); o = np.argsort(days)
    V = z['validation_Vm'][o].astype(np.float32); Fq = z['validation_Fm'][o].astype(np.float32); Em = z['validation_Em'][o]
    E = np.isfinite(V); Q = E & (V <= P['cut'][None, None, :])
    Pv = dict(panel=panel, cut=P['cut'], days=list(days[o]), dow=pd.to_datetime(pd.Series(days[o])).dt.dayofweek.to_numpy(), V=V, Fq=Fq,
              E=E, Q=Q, Em=Em, ON=z['validation_ON'][o].astype(np.float32), OFF=z['validation_OFF'][o].astype(np.float32))
    return P, Pv


def pm_select(Pv):
    """copy of exp_firstcross_pmarch.select (faithful ongoing candidates on the masked validation month, blank-touching dropped)."""
    from exp_firstcross_prod_snippet import fill_time36
    V, E, Q, Em = Pv['V'], Pv['E'], Pv['Q'], Pv['Em']; D, _, L = V.shape
    VF = fill_time36(V); MF = np.isfinite(VF); QF = MF & (VF <= Pv['cut'][None, None, :])
    blank = (Em.sum(2) >= 5) & (E.sum(2) == 0)
    og = np.zeros((D, 288), bool); on = og.copy(); pers = np.full((D, 288), np.nan)
    for d in range(D):
        for t in range(H, 288 - F):
            if blank[d, t - H:t + F + 1].any() or Em[d, t - H:t].mean() < 0.7:
                continue
            if lab.established(Q[d, t - H:t]):
                Tf = QF[d, t + 1:t + 1 + F]; Mf = MF[d, t + 1:t + 1 + F]
                if not Tf.any():
                    continue
                S_ = np.repeat(QF[d, t][None], F, 0)
                u = ((Tf | S_) & Mf).sum(); i = (Tf & S_ & Mf).sum()
                pers[d, t] = 1.0 if u == 0 else i / u
                og[d, t] = pers[d, t] <= 0.9
            else:
                on[d, t] = Q[d, t + 1:t + 1 + F].any()
    return on, og, pers, blank


def build_pm(panel, refresh=False):
    """pseudo-March ongoing rows (production matrix with all columns + all-train prior, meta, ck features from the masked
    validation month for the whole day <= T, truth / ev / qlast, window frame); cached per (panel, set)."""
    out = {}
    fs = {s: f'{ODIR}/pm_{panel}__{s}' for s in PM_SETS}
    if all(os.path.exists(f + '.npz') for f in fs.values()) and not refresh:
        for s, f in fs.items():
            z = np.load(f + '.npz'); out[s] = dict({k: z[k] for k in z.files}, win=pd.read_pickle(f + '.pkl'))
        return out
    P, Pv = val_panel(panel)
    on, og, pers, blank = pm_select(Pv)
    D = len(Pv['days'])
    g_on, g_og = lab._glist(on), lab._glist(og); sset = set()
    for d0 in range(D):
        _, b = lab.greedy(g_on, g_og, 'interleaved', start_day=d0); sset.update(b)
    dd, tt = np.where(og); rs = ~og[dd, tt - 1]
    sets = {'offlike_m': sorted(sset), 'runstart_m': sorted(zip(dd[rs].tolist(), tt[rs].tolist()))}
    prior = lab.prior_table(P, np.ones(len(P['days']), bool))
    vis_day = Pv['E'].sum((1, 2)) / np.maximum(Pv['Em'].sum((1, 2)), 1)
    for s, wl in sets.items():
        d = np.array([w[0] for w in wl]); t = np.array([w[1] for w in wl])
        idx = t[:, None] + np.arange(-H, 0)[None]
        Vh = Pv['V'][d[:, None], idx]; Fh = Pv['Fq'][d[:, None], idx]; VT = Pv['V'][d, t]
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
        X, y, e, meta, call = lab.to_matrix(R, lab.ONGOING, None, lab.ongoing_keep(R))
        assert list(call) == lab.all_cols()
        FE = np.empty((W, L, len(CK_ALL)), np.float32)
        for a in range(0, W, 192):                                   # masked layer for the whole day (slots < T), VT at T
            sl = slice(a, a + 192)
            Rr = ck_compose(Pv['V'][d[sl]], t[sl], Vh[sl], VT[sl], P['cut'])
            Xc = ck_features(Rr, t[sl]); FE[sl] = np.stack([Xc[k] for k in CK_ALL], -1)
        est_prev = np.array([lab.established(Pv['Q'][a, b - 1 - H:b - 1]) for a, b in zip(d, t)])
        runst = og[d, t] & ~og[d, t - 1]
        win = pd.DataFrame(dict(d=d, t=t, cond=lab.ONGOING, fold=0, pers_fill=pers[d, t], runstart=runst, faithful=og[d, t], vis=vis_day[d],
                                blank_before=[bool(blank[a, :max(b - H, 0)].any()) for a, b in zip(d, t)],
                                wtype=np.where(~runst, 'midrun', np.where(est_prev, 'rs_old', 'rs_new'))))
        dat = dict(X=X, meta=meta.astype(np.int32), FE=FE, truth=np.stack([Pv['Q'][d, t + h] for h in range(1, F + 1)], 1),
                   ev=np.stack([Pv['E'][d, t + h] for h in range(1, F + 1)], 1), qlast=qlast)
        tmp = f'{fs[s]}.{os.getpid()}.tmp.npz'
        win.to_pickle(fs[s] + '.pkl'); np.savez(tmp, **dat); os.replace(tmp, fs[s] + '.npz')
        out[s] = dict(dat, win=win)
    lab.drop_panel(panel)
    return out


def pm_score(models, pc, xc, arm, sets=PM_SETS):
    res = []; allc = lab.all_cols(); ix = [allc.index(c) for c in pc]; jx = [CK_ALL.index(c) for c in xc]
    ms = models if isinstance(models, (list, tuple)) else [models]
    for panel in lab.PANELS:
        pm = build_pm(panel)
        for s in sets:
            z = pm[s]; meta = z['meta']
            if not len(z['win']):
                continue
            X = z['X'][:, ix]
            if xc:
                X = np.concatenate([X, z['FE'][meta[:, 0][:, None], meta[:, 1][:, None], np.array(jx)[None, :]]], 1)
            P6 = np.zeros((len(z['win']), F, z['qlast'].shape[1]), np.float32)
            P6[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = sum(m.predict(X, num_threads=lab.THREADS) for m in ms) / len(ms)
            rc = types.SimpleNamespace(truth=z['truth'], ev=z['ev'], qlast=z['qlast'], win=z['win'], name=s, panel=panel, cond=lab.ONGOING,
                                       L=z['qlast'].shape[1])
            df = lab.score_probs(rc, P6, arm=arm, set_name=s)
            for c in ('vis', 'wtype', 'blank_before'):
                df[c] = z['win'][c].to_numpy()
            res.append(df)
    return pd.concat(res, ignore_index=True)


def final_model(kind, xc, leaves=255, lr=0.05, n_trees=500, tag=None):
    """k = 4 model on all train days (F03 rows), cached under results/ogclock/final/."""
    import lightgbm as lgb
    pc, xc = cols_spec(kind, xc)
    tag = tag or f'{kind}_{len(xc)}'
    f = f'{RES}/final/model_{tag}_{leaves}_{n_trees}.txt'
    os.makedirs(os.path.dirname(f), exist_ok=True)
    if os.path.exists(f):
        return lgb.Booster(model_file=f), pc, xc
    t1 = time.time()
    X, y, w = train_matrix(4, pc, xc)
    m = lgb.train(lab.lgb_params(num_leaves=leaves, learning_rate=lr, min_data_in_leaf=40, seed=0),
                  lgb.Dataset(X, y, weight=w, feature_name=pc + xc, free_raw_data=True), n_trees)
    lab.log(f'final {tag}: {len(X)} rows x {len(pc) + len(xc)} cols in {time.time() - t1:.0f}s')
    del X, y, w; gc.collect()
    m.save_model(f)
    return m, pc, xc
