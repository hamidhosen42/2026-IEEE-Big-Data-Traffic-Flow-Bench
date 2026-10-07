"""TRACK occ: OCCUPANCY features for Task 2 on top of the ALL-FINITE baseline (track allfin).

Occupancy is an independent density reading of the same detector, co-masked with speed / flow (finite exactly where the
all-finite speed is finite in train, and exactly where speed is visible in the masked month / official windows: checked).
Design (no label or post-T value enters a feature):
  * baseline rows = the all-finite RowCaches of allfin (cache/allfin/rows, read only); the occupancy features are a SIDE-CAR
    per (panel, set, cond): per-window raw arrays [W, L] computed from the same inputs as the history features
    (60-min history slots T-12..T-1 of the occupancy cube, finite wherever the all-finite speed is finite) and the slot-T value
    (visible exactly where the production slot-T speed draw is visible: eligible cells with the rng(len(wins)) 62 % draw,
    ineligible-valued cells always).  Row r gets side[wi[r], li[r]], so labels / selection / weights / keep are untouched.
  * per-link calibration constants from TRAIN days only, fold-safe (k = 0..3 without day block k, k = 4 all days):
      ocrit  = occupancy threshold that best separates queued (v <= 0.6 v_free) from free cells of the link (robust FD fit;
               near-threshold speeds are almost absent, so the median-of-band estimate is undefined for most links)
      c      = median of log(o * lanes * v / q) over free-flow cells (v > 0.85 v_free)  (density agreement offset)
  * 13 features: o_last, or_last, or_T, or_mean3, or_slope, or_max12, or_n-3..or_n+3, dens_dev   (or = o / ocrit)
Code copied from allfin_lib.py (f03_sub, the slot-T draw of make_rows_af, compare / fmt), never edited there.
Own caches: data/t2local/cache/occ/, results analysis/t2lab/results/occ/.
"""
import fcntl, gc, json, os, time, warnings
import numpy as np
import pandas as pd
import lab
import exp_firstcross_lib as fc

H, F = lab.H, lab.F
ONSET, ONGOING = lab.ONSET, lab.ONGOING
XDIR = f'{lab.DATA}/x'
ODIR = f'{lab.DATA}/occ'
AF = f'{lab.CACHE}/allfin'                       # read only
OC = f'{lab.CACHE}/occ'
RES = f'{lab.ROOT}/analysis/t2lab/results/occ'
for _d in (OC, RES, f'{OC}/side', f'{OC}/calib', f'{OC}/oof', f'{OC}/onset', f'{OC}/stats', f'{OC}/locks', f'{OC}/done'):
    os.makedirs(_d, exist_ok=True)
P_VIS = 0.62
SETS = [('official', ONGOING), ('offlike', ONGOING), ('pseudo', ONGOING), ('runstart', ONGOING), ('faith3', ONGOING),
        ('official', ONSET), ('offlike', ONSET), ('pseudo', ONSET), ('dense1500', ONSET)]
NB = (-3, -2, -1, 1, 2, 3)
OCC_COLS = ['o_last', 'or_last', 'or_T', 'or_mean3', 'or_slope', 'or_max12'] + [f'or_n{k:+d}' for k in NB] + ['dens_dev']
RAW = ('oT', 'olast', 'omean3', 'ofirst3', 'omax12', 'lr3')
MIN_Q, MIN_FREE, MIN_FF = 20, 20, 100            # calibration minimum counts (else the panel median of the estimated links)


# ------------------------------------------------------------------ copies of production / allfin helpers
def ffill_time(a):
    """production ffill_time: forward-fill NaNs along axis 1 (time) of [W, T, L]."""
    out = a.copy()
    for i in range(1, a.shape[1]):
        m = np.isnan(out[:, i]); out[:, i][m] = out[:, i - 1][m]
    return out


def shift_links(a, k):
    """production shift_links: out[..., l] = a[..., l + k] (NaN outside)."""
    out = np.full_like(a, np.nan)
    if k > 0: out[..., :-k] = a[..., k:]
    elif k < 0: out[..., -k:] = a[..., :k]
    else: out[:] = a
    return out


def f03_sub(win):
    """allfin_lib.f03_sub: faith3 windows of the F03 recipe (non-run-starts, <= fc.CAP per panel, rng(0))."""
    return fc._cap_idx(np.where(~win.runstart.to_numpy())[0], fc.CAP)


def af_name(name):
    return 'faith3F03' if name == 'faith3' else name


def rows_af(panel, name, cond=ONGOING):
    """allfin RowCache (read only).  'faith3' / 'faith3F03' = the F03 subset of faith3."""
    return lab.RowCache(f'{AF}/rows/{panel}__{af_name(name)}__{cond}')


# ------------------------------------------------------------------ cubes and calibration
def load_cubes(P):
    """all-finite speed / flow cubes and the occupancy cube of a train panel (cached in P)."""
    if 'O' not in P:
        z = np.load(f'{XDIR}/t2x_{P["panel"]}.npz', allow_pickle=False)
        assert [str(x) for x in z['links']] == P['links'] and [str(x) for x in z['train_days']] == P['days']
        Vx = z['train_Vx'].astype(np.float32); Fx = z['train_Fx'].astype(np.float32)
        assert not (np.isfinite(Vx) & P['E']).any()
        P['Vx'] = Vx; P['V_all'] = np.where(P['E'], P['V'], Vx); P['F_all'] = np.where(P['E'], P['Fq'], Fx)
        zo = np.load(f'{ODIR}/t2occ_{P["panel"]}.npz', allow_pickle=False)
        assert [str(x) for x in zo['links']] == P['links'] and [str(x) for x in zo['train_days']] == P['days'], 'occ order differs'
        O = zo['train_O'].astype(np.float32)
        assert np.array_equal(np.isfinite(O), np.isfinite(P['V_all'])), 'occupancy not co-located with the all-finite speed'
        P['O'] = O
    return P['V_all'], P['F_all'], P['O']


def _best_thr(o, q):
    """threshold minimising the misclassification of q (queued) by o > thr (midpoint of the best gap)."""
    s = np.argsort(o, kind='stable'); os_, qs = o[s], q[s]
    err = np.cumsum(qs) + ((~qs).sum() - np.cumsum(~qs))
    i = int(np.argmin(err))
    return 0.5 * (os_[i] + os_[min(i + 1, len(os_) - 1)]), err[i] / len(o)


def calibrate(P):
    """per-link ocrit5 / c5 [5, L] (k = 0..3 without day block k, k = 4 all train days) + diagnostics."""
    f = f'{OC}/calib/{P["panel"]}.npz'
    if os.path.exists(f):
        z = np.load(f)
        return z['ocrit5'], z['c5']
    Va, Fa, O = load_cubes(P)
    L = O.shape[2]; cut = P['cut']; vf = P['vf']; lanes = P['lanes']
    oc5 = np.full((5, L), np.nan, np.float32); c5 = oc5.copy(); est = np.zeros((5, L), bool); err5 = oc5.copy()
    for k in range(5):
        dm = (P['block'] != k) if k < 4 else np.ones(len(P['days']), bool)
        for l in range(L):
            v = Va[dm, :, l]; o = O[dm, :, l]; q_ = Fa[dm, :, l]
            m = np.isfinite(v)
            vv, oo, qq = v[m], o[m], q_[m]
            qd = vv <= cut[l]
            if qd.sum() >= MIN_Q and (~qd).sum() >= MIN_FREE:
                oc5[k, l], err5[k, l] = _best_thr(oo, qd); est[k, l] = True
            ff = (vv > 0.85 * vf[l]) & (qq > 0) & (oo > 0)
            if ff.sum() >= MIN_FF:
                c5[k, l] = np.median(np.log(oo[ff] * lanes[l] * vv[ff] / qq[ff]))
        oc5[k, ~est[k]] = np.nanmedian(oc5[k, est[k]])
        c5[k, ~np.isfinite(c5[k])] = np.nanmedian(c5[k])
    np.savez(f, ocrit5=oc5, c5=c5, est=est, err5=err5)
    return oc5, c5


# ------------------------------------------------------------------ occupancy features
def raw_occ(Oh, OT, Vh, Fh, lanes):
    """per-window raw occupancy quantities [W, L] from the history [W, 12, L] and the slot-T value [W, L] (NaN = not visible)."""
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        of = ffill_time(Oh)
        olast = np.where(np.isfinite(OT), OT, of[:, -1])
        ok = np.isfinite(Oh) & np.isfinite(Vh) & np.isfinite(Fh) & (Oh > 0) & (Fh > 0) & (Vh > 0)
        lr = np.where(ok, np.log(np.where(ok, Oh * lanes[None, None, :] * Vh / np.where(ok, Fh, 1.0), 1.0)), np.nan)
        out = dict(oT=OT, olast=olast, omean3=np.nanmean(Oh[:, -3:], 1), ofirst3=np.nanmean(Oh[:, :3], 1),
                   omax12=np.nanmax(Oh, 1), lr3=np.nanmean(lr[:, -3:], 1))
    return {k: np.asarray(v, np.float32) for k, v in out.items()}


def occ_feats(raw, oc, c, wi=None, li=None):
    """the 13 occupancy features from the raw arrays and one calibration (oc, c [L]).
    wi / li given: rows [n, 13]; else the [W, L, 13] cube."""
    ocb = oc[None, :].astype(np.float32)
    orl = raw['olast'] / ocb
    with np.errstate(all='ignore'):
        cube = [raw['olast'], orl, raw['oT'] / ocb, raw['omean3'] / ocb, (raw['omean3'] - raw['ofirst3']) / ocb, raw['omax12'] / ocb]
        cube += [shift_links(orl, k) for k in NB]
        cube += [raw['lr3'] - c[None, :].astype(np.float32)]
    if wi is None:
        return np.stack(cube, -1).astype(np.float32)
    return np.stack([a[wi, li] for a in cube], 1).astype(np.float32)


# ------------------------------------------------------------------ side-car builder (train window sets)
def side_path(panel, name, cond):
    return f'{OC}/side/{panel}__{af_name(name)}__{cond}.npz'


def build_side(P, name, cond, check=True):
    """raw occupancy arrays of the windows of the allfin RowCache (panel, set, cond); the slot-T draw is the production draw of
    the FULL window list (rng(len(wins)), 62 %), restricted afterwards for faith3 (as allfin make_rows_af).  check=True:
    the speed visibility / values implied by my draw must equal the cached all-finite rows (has_T, vr_T, vr_last)."""
    panel = P['panel']
    win = lab.window_set(panel, name); win = win[win.cond == cond].reset_index(drop=True)
    st = dict(panel=panel, set=af_name(name), cond=cond, windows_full=len(win))
    rc = rows_af(panel, name, cond)
    if not len(win):
        np.savez(side_path(panel, name, cond), d=np.zeros(0, int), t=np.zeros(0, int), **{k: np.zeros((0, len(P['links'])), np.float32) for k in RAW})
        return dict(st, windows=0)
    Va, Fa, O = load_cubes(P)
    d = win.d.to_numpy(); t = win.t.to_numpy(); L = O.shape[2]
    rng = np.random.default_rng(len(win))
    draw = rng.random((len(d), L)) < P_VIS
    if name == 'faith3':
        sub = f03_sub(win); d, t, draw = d[sub], t[sub], draw[sub]
    assert np.array_equal(rc.win.d.to_numpy(), d) and np.array_equal(rc.win.t.to_numpy(), t), 'window order differs from the allfin rows'
    idx = t[:, None] + np.arange(-H, 0)[None]
    Vh = Va[d[:, None], idx]; Fh = Fa[d[:, None], idx]; Oh = O[d[:, None], idx]
    VT = np.where(draw, P['V'][d, t], np.nan)
    VT = np.where(P['E'][d, t], VT, P['Vx'][d, t])                 # ineligible-valued cells: always visible
    OT = np.where(np.isfinite(VT), O[d, t], np.nan).astype(np.float32)
    raw = raw_occ(Oh, OT, Vh, Fh, P['lanes'])
    st.update(windows=len(d), rows=int(rc.n), T_vis=float(np.isfinite(OT).mean()), hist_vis=float(np.isfinite(Oh).mean()))
    if check and rc.n:
        cols = rc.cols; wi, li = rc.wi, rc.li
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            vT = VT / P['cut'][None, :]
            vrf = ffill_time(Vh / P['cut'][None, None, :])
            last = np.where(np.isfinite(vT), vT, vrf[:, -1])
        for cn, ref in (('has_T', np.isfinite(vT).astype(np.float32)), ('vr_T', vT.astype(np.float32)), ('vr_last', last.astype(np.float32))):
            x = np.asarray(rc.X[:, cols.index(cn)])
            st[f'par_{cn}'] = bool(np.array_equal(x, ref[wi, li], equal_nan=True))
        st['par_ok'] = bool(st['par_has_T'] and st['par_vr_T'] and st['par_vr_last'])
    np.savez(side_path(panel, name, cond), d=d, t=t, **raw)
    return st


def ensure_side(panels=None, sets=None, check=True):
    out = []
    with open(f'{OC}/build.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for panel in panels or lab.PANELS:
            P = None
            for n, c in sets or SETS:
                sf = f'{OC}/stats/{panel}__{af_name(n)}__{c}.json'
                if os.path.exists(side_path(panel, n, c)) and os.path.exists(sf):
                    out.append(json.load(open(sf))); continue
                if P is None:
                    P = lab.load_panel(panel, cache=False); load_cubes(P); calibrate(P)
                t1 = time.time(); st = build_side(P, n, c, check); st['secs'] = round(time.time() - t1, 1)
                json.dump(st, open(sf, 'w')); out.append(st)
            if P is not None:
                oc5, c5 = calibrate(P)
                lab.log(f'occ side-cars {panel} built | ocrit(k=4) q10/50/90 {np.quantile(oc5[4], [0.1, 0.5, 0.9]).round(3).tolist()}')
                del P; gc.collect()
    return pd.DataFrame(out)


class Side:
    """raw occupancy arrays of one (panel, set, cond) + the panel calibration."""
    _cal = {}

    def __init__(self, panel, name, cond):
        z = np.load(side_path(panel, name, cond))
        self.raw = {k: z[k] for k in RAW}; self.d = z['d']; self.t = z['t']
        if panel not in Side._cal:
            c = np.load(f'{OC}/calib/{panel}.npz'); Side._cal[panel] = (c['ocrit5'], c['c5'])
        self.oc5, self.c5 = Side._cal[panel]

    def rows(self, wi, li, k):
        return occ_feats(self.raw, self.oc5[k], self.c5[k], wi, li)


# ------------------------------------------------------------------ ongoing: F03 recipe with / without the occupancy columns
TRAIN = [('faith3F03', 0.3), ('runstart', 1.0)]


OCC_SETS = {'all': OCC_COLS,                                        # the candidate (pre-registered 13 columns)
            'noraw': [c for c in OCC_COLS if c != 'o_last']}         # mechanism diagnostic only: no raw occupancy (only o / ocrit ratios)


def matrix(rc, S, m, cols, k, use_occ, occ_set='all'):
    Xb = rc.matrix(m, cols, k)
    if not use_occ:
        return Xb
    r = np.where(m)[0]
    Xo = S.rows(rc.wi[r], rc.li[r], k)
    if occ_set != 'all':
        Xo = Xo[:, [OCC_COLS.index(c) for c in OCC_SETS[occ_set]]]
    return np.hstack([Xb, Xo])


def train_matrix(k, cols, use_occ, occ_set='all'):
    """rows / order as allfin train_matrix (panel outer, set inner; e & keep5[:, k] & fold != k & weight > 0), preallocated."""
    parts = []; n = 0; info = {}
    for panel in lab.PANELS:
        for name, sw in TRAIN:
            rc = rows_af(panel, name, ONGOING)
            if not rc.n:
                continue
            ww = np.full(len(rc.win), float(sw), np.float32)
            m = rc.e & rc.keep5[:, k] & (rc.fold != k) & (ww[rc.wi] > 0)
            c = int(m.sum()); parts.append((panel, name, rc, m, ww, c)); n += c
            a = info.setdefault(name, dict(windows=0, rows=0))
            a['windows'] += int(((rc.win.fold.to_numpy() != k) & (ww > 0)).sum()); a['rows'] += c
    nc = len(cols) + (len(OCC_SETS[occ_set]) if use_occ else 0)
    X = np.empty((n, nc), np.float32); y = np.empty(n, np.float32); w = np.empty(n, np.float32)
    o = 0
    for panel, name, rc, m, ww, c in parts:
        S = Side(panel, name, ONGOING) if use_occ else None
        X[o:o + c] = matrix(rc, S, m, cols, k, use_occ, occ_set); y[o:o + c] = rc.y[m]; w[o:o + c] = ww[rc.wi[m]]; o += c
    return X, y, w, info


def predict_set(models, rc, S, k, cols, use_occ, occ_set='all'):
    """lab.predict_set with the occupancy columns: windows of fold k, keep region under the fold-k prior."""
    P6 = np.zeros((len(rc.win), F, rc.L), np.float32)
    if not rc.n:
        return P6
    m = rc.keep5[:, k] & (rc.win.fold.to_numpy() == k)[rc.wi]
    if m.any():
        X = matrix(rc, S, m, cols, k, use_occ, occ_set)
        ms = models if isinstance(models, (list, tuple)) else [models]
        p = sum(mdl.predict(X, num_threads=lab.THREADS) for mdl in ms) / len(ms)
        P6[rc.wi[m], rc.h[m] - 1, rc.li[m]] = p
    return P6


def arm_dir(arm):
    return f'{OC}/oof/{arm}'


def run_arm(arm, cols='np', use_occ=True, folds=(0, 1, 2, 3), evals=fc.EVALS, leaves=255, lr=0.05, n_trees=500, seed=0, occ_set='all'):
    cols_ = lab.colset(cols); odir = arm_dir(arm); os.makedirs(odir, exist_ok=True)
    feat = cols_ + (OCC_SETS[occ_set] if use_occ else [])
    for k in folds:
        ff = f'{odir}/fold{k}.pkl'
        if os.path.exists(ff) or (k == 4 and os.path.exists(f'{odir}/model_f4.txt')):
            lab.log(f'{arm} fold {k}: found on disk, skipped'); continue
        t1 = time.time()
        X, y, w, info = train_matrix(k, cols_, use_occ, occ_set)
        nrow, pos = len(X), float(y.mean())
        p = lab.lgb_params(num_leaves=leaves, learning_rate=lr, min_data_in_leaf=40, seed=seed)
        mdl = fc.fit(X, y, w, p, n_trees, feat)
        del X, y, w; gc.collect()
        mdl.save_model(f'{odir}/model_f{k}.txt')
        lab.log(f'{arm} fold {k}: {nrow} rows x {len(feat)} cols, pos {pos:.3f}, trained in {time.time() - t1:.0f}s | {json.dumps(info)}')
        if k == 4:
            continue
        res = []; probs = {}
        for s in evals:
            for panel in lab.PANELS:
                rc = rows_af(panel, s, ONGOING)
                widx = np.where(rc.win.fold.to_numpy() == k)[0]
                if not len(widx):
                    continue
                S = Side(panel, s, ONGOING) if use_occ else None
                P6 = predict_set([mdl], rc, S, k, cols_, use_occ, occ_set)
                res.append(lab.score_probs(rc, P6, widx, arm=arm, set_name=s))
                probs[(s, panel)] = (widx, P6[widx].astype(np.float16))
        df = pd.concat(res, ignore_index=True)
        imp = dict(zip(mdl.feature_name(), mdl.feature_importance('gain').tolist()))
        pd.to_pickle(dict(df=df, probs=probs, info=info, nrow=nrow, pos=pos, secs=time.time() - t1, imp=imp, feat=feat), ff)
        lab.log(f'{arm} fold {k}: ' + ', '.join(f'{s} {lab.aggregate(df[df.set == s]):.4f} (n={int((df.set == s).sum())})' for s in evals))
        del mdl; gc.collect()


def load_arm(arm):
    """per-window frame of an occ arm, or of an allfin arm ('af:<arm>', all-finite rows)."""
    dfs = []
    for k in range(4):
        if arm.startswith('af:'):
            ff = f'{AF}/oof/{arm[3:]}/fold{k}.pkl'
            if os.path.exists(ff):
                dfs.append(pd.read_pickle(ff)['out']['all'][0])
        else:
            ff = f'{arm_dir(arm)}/fold{k}.pkl'
            if os.path.exists(ff):
                dfs.append(pd.read_pickle(ff)['df'])
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_probs(arm, set_name, panel, rc):
    P6 = np.zeros((len(rc.win), F, rc.L), np.float32); done = np.zeros(len(rc.win), bool)
    for k in range(4):
        if arm.startswith('af:'):
            ff = f'{AF}/oof/{arm[3:]}/fold{k}.pkl'
            if not os.path.exists(ff):
                continue
            pr = pd.read_pickle(ff)['out']['all'][1]
        else:
            ff = f'{arm_dir(arm)}/fold{k}.pkl'
            if not os.path.exists(ff):
                continue
            pr = pd.read_pickle(ff)['probs']
        if (set_name, panel) in pr:
            widx, p6 = pr[(set_name, panel)]
            P6[widx] = p6; done[widx] = True
    return P6, done


def blend_frame(arms, set_name, name):
    """score the mean probability of several arms (np + id = the production bag shape)."""
    out = []
    for panel in lab.PANELS:
        rc = rows_af(panel, set_name, ONGOING)
        P6 = 0; done = None
        for a in arms:
            p, dn = load_probs(a, set_name, panel, rc)
            P6 = P6 + p / len(arms); done = dn if done is None else done & dn
        widx = np.where(done)[0]
        if len(widx):
            out.append(lab.score_probs(rc, P6.astype(np.float32), widx, arm=name, set_name=set_name))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ------------------------------------------------------------------ statistics (copied from allfin_lib)
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
