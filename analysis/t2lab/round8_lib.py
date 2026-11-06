"""round8_lib.py -- ROUND 8 onset program (results/round8/PREREG.md, sha1 asserted on import; reading in INTERPRETATION.txt).
Shared code: PREREG check, FEAT precursor columns (data <= T only), E1 / E2 / training row assembly, decoders (production and
exact expected-IoU), per-window scoring.  Extends phase_lib.py / the EXP_T2_PHASE2 harness (phase2_onset.py)."""
import hashlib, os, types
import numpy as np, pandas as pd
import lab, ogfeat_lib as og
import phase_lib as ph

SHA = '08beede615dcb148dead77d01ba502f73c075eb8'
RES = f'{lab.ROOT}/analysis/t2lab/results/round8'
CD = f'{lab.CACHE}/round8'
MD = f'{CD}/models'
for _d in (CD, MD):
    os.makedirs(_d, exist_ok=True)
H, F = lab.H, lab.F
COLS = lab.all_cols()
FCOLS = ['f8_qT', 'f8_kT', 'f8_kT_dn1', 'f8_k_dn1_m15', 'f8_upacc3', 'f8_dq_bnd']
E1_MODEL = f'{ph.ARMD}/onset/cat_base_s{{s}}.cbm'          # cached EXP_T2_PHASE / PHASE2 full-train base models


def check_prereg():
    s = hashlib.sha1(open(f'{RES}/PREREG.md', 'rb').read()).hexdigest()
    assert s == SHA, f'PREREG.md changed: {s}'
    assert open(f'{RES}/PREREG.sha1').read().split()[0] == SHA
    return s


check_prereg()


def fit(X, y, seed):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=600, depth=8, learning_rate=0.08, thread_count=lab.THREADS, verbose=0, random_seed=seed,
                           allow_writing_files=False)
    m.fit(X, y); return m


# ---------------------------------------------------------------- FEAT columns (slots <= T only)
def panel_static(npz):
    z = np.load(npz, allow_pickle=False)
    length = z['length'].astype(np.float64); start = np.r_[0.0, np.cumsum(length)[:-1]]
    U = (start[None, :] <= start[:, None]) & (start[:, None] - start[None, :] <= 3.0)       # U[i, j]: j upstream of i within 3 km
    np.fill_diagonal(U, False)
    U &= np.arange(len(length))[None, :] < np.arange(len(length))[:, None]
    return dict(kcrit=z['fd_kcrit'].astype(np.float64), length=length, cap=z['cap'].astype(np.float64), U=U.astype(np.float64))


def feat_cols(V, Fq, d, t, mask, st):
    """V, Fq [D, 288, L] (NaN = missing); d, t [W]; mask [W, L] slot-T visibility (None = cube value at T as is, e.g. the masked
    March layer); st = panel_static.  Returns [W, L, 6] float32 (FCOLS).  Every slot read is asserted <= t."""
    d = np.asarray(d, np.int64); t = np.asarray(t, np.int64)
    cap, kc, ln, U = st['cap'][None], st['kcrit'][None], st['length'][None], st['U']

    def vals(s):
        s = np.asarray(s, np.int64)
        assert (s <= t).all() and (s >= 0).all(), 'slot after T (or < 0) read'
        v = V[d, s].astype(np.float64); q = Fq[d, s].astype(np.float64)
        q = np.where(np.isfinite(v), q, np.nan)
        if mask is not None:
            hide = (s == t)[:, None] & ~mask
            v = np.where(hide, np.nan, v); q = np.where(hide, np.nan, q)
        return v, q

    def kr(v, q):
        with np.errstate(all='ignore'):
            return np.where(v > 0, q / (v * kc), np.nan)

    with np.errstate(all='ignore'):
        vT, qT = vals(t)
        f_q = qT / cap
        krT = kr(vT, qT)
        krT_dn = lab.prod.shift_links(krT, 1)
        ks = [lab.prod.shift_links(kr(*vals(t - j)), 1) for j in (3, 2, 1, 0)]
        k_dn_m15 = np.nanmean(np.stack(ks), 0)
        veh = np.where(np.isfinite(krT), krT * kc * ln, 0.0); crit = np.where(np.isfinite(krT), kc * ln, 0.0)
        num = veh @ U.T; den = crit @ U.T
        upacc = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        b = t - t % 6
        after = []
        for j in range(3):
            s = b + j; ok = s <= t
            _, q = vals(np.minimum(s, t)); after.append(np.where(ok[:, None], q, np.nan))
        before = [vals(b - j)[1] for j in (3, 2, 1)]
        dq = (np.nanmean(np.stack(after), 0) - np.nanmean(np.stack(before), 0)) / cap
    out = np.stack([f_q, krT, krT_dn, k_dn_m15, upacc, dq], -1).astype(np.float32)
    return out


def prod_mask(W, L):
    """the production make_rows slot-T visibility draw for a window list of length W."""
    rng = np.random.default_rng(W)
    return rng.random((W, L)) < 0.62


def train_feat(panel, name):
    """FEAT columns for every window of lab.rows(panel, name, ONSET) (window order of rc.win) + mask check against has_T."""
    rc = lab.rows(panel, name, lab.ONSET)
    P = lab.load_panel(panel)
    W = len(rc.win); L = len(P['links'])
    mask = prod_mask(W, L)
    d, t = rc.win.d.to_numpy(), rc.win.t.to_numpy()
    VT = np.where(mask, P['V'][d, t], np.nan)
    ih = rc.cols.index('has_T')
    assert np.array_equal(np.isfinite(VT)[rc.wi, rc.li].astype(np.float32), np.asarray(rc.X[:, ih])), f'mask mismatch {panel} {name}'
    Z = feat_cols(P['V'], P['Fq'], d, t, mask, panel_static(P['npz']))
    return rc, Z


# ---------------------------------------------------------------- E1 (214 pseudo-March onset windows), as phase2_onset.py pm
def e1_panel(panel):
    f = f'{CD}/e1_{panel}.pkl'
    if os.path.exists(f):
        return pd.read_pickle(f)
    P, Pv = og.val_panel(panel)
    on, ogm, _ = og.pm_select(Pv)
    D = len(Pv['days'])
    g_on, g_og = lab._glist(on), lab._glist(ogm); sset = set()
    for d0 in range(D):
        a, _ = lab.greedy(g_on, g_og, 'interleaved', start_day=d0); sset.update(a)
    wins = sorted(sset)
    if not wins:
        return None
    prior_tr = lab.prior_table(P, np.ones(len(P['days']), bool))
    R = ph.onset_rows(P, Pv, [(a, b, lab.ONSET) for a, b in wins], prior_tr)
    Xm, _, _, meta, call = lab.to_matrix(R, lab.ONSET, None, None); assert list(call) == COLS
    m6 = meta[:, 2] == F; Xm = Xm[m6]; meta = meta[m6]
    d, t = R['d'], R['t']; wi, li = meta[:, 0], meta[:, 1]
    cls = (Pv['dow'][d] >= 5).astype(int); s6 = np.minimum(t + 6, 287)
    c_r, d_r, s6_r, t_r = cls[wi], d[wi], s6[wi], t[wi]
    trv = [prior_tr[c_r, s6_r, li], prior_tr[c_r, t_r, li], prior_tr[2, s6_r, li]]
    cQ, cE = ph.cum_tables(Pv, True)
    JP = [COLS.index(c) for c in lab.PRIOR3]
    Xs4 = Xm.copy()
    for j, (cc, ss) in enumerate(((c_r, s6_r), (c_r, t_r), (np.full_like(c_r, 2), s6_r))):
        q = cQ[cc, d_r, ss, li]; ee = cE[cc, d_r, ss, li]
        sp = np.where(ee > 0, q / np.maximum(ee, 1e-9), 0.0); n = ee / 5.0; a_ = n / (n + 4)
        Xs4[:, JP[j]] = (1 - a_) * trv[j] + a_ * sp
    del cQ, cE
    Zw = feat_cols(Pv['V'], Pv['Fq'], d, t, None, panel_static(P['npz']))
    out = dict(panel=panel, Xm=Xm, Xs4=Xs4, Z=Zw[wi, li], wi=wi, li=li, W=R['W'], L=R['L'],
               win=pd.DataFrame(dict(d=d, t=t, cond=lab.ONSET, fold=0)),
               truth=np.stack([Pv['Q'][d, t + h] for h in range(1, F + 1)], 1), ev=np.stack([Pv['E'][d, t + h] for h in range(1, F + 1)], 1),
               qlast=R['qlast'])
    pd.to_pickle(out, f)
    lab.drop_panel(panel)
    return out


def e1_all():
    return {p: e for p in lab.PANELS if (e := e1_panel(p)) is not None}


# ---------------------------------------------------------------- E2 rows (offlike onset, step 6, fold k) and training rows
def e2_rows(k):
    """per panel: X (fold-k prior, 69), Z (FEAT), wi, li, y, e, rc meta for the fold-k offlike onset windows (step 6)."""
    out = {}
    for panel in lab.PANELS:
        rc = lab.rows(panel, 'offlike', lab.ONSET)
        widx = np.where(rc.win.fold.to_numpy() == k)[0]
        if not len(widx) or not rc.n:
            continue
        msk = (rc.win.fold.to_numpy() == k)[rc.wi] & (rc.h == F)
        r = np.where(msk)[0]
        Zw = feat_offlike(panel)
        out[panel] = dict(X=rc.matrix(msk, COLS, k), Z=Zw[rc.wi[r], rc.li[r]], wi=rc.wi[r], li=rc.li[r], y=rc.y[r], e=rc.e[r],
                          widx=widx, rc=rc)
    return out


def feat_offlike(panel):
    f = f'{CD}/feat_offlike_{panel}.npy'
    if not os.path.exists(f):
        _, Z = train_feat(panel, 'offlike'); np.save(f, Z)
    return np.load(f)


def train_rows(prior_k=None, with_feat=True):
    """dense1500 onset training rows (eligible cells), all-train prior (prior_k None) or fold-k prior; FEAT columns aligned."""
    Xs, ys, fs, Zs = [], [], [], []
    for p in lab.PANELS:
        rc = lab.rows(p, 'dense1500', lab.ONSET); assert list(rc.cols) == COLS, p
        r = np.where(rc.e)[0]
        Xs.append(np.asarray(rc.X[r]) if prior_k is None else rc.matrix(rc.e, COLS, prior_k))
        ys.append(rc.y[r].astype(int)); fs.append(rc.fold[r])
        if with_feat:
            f = f'{CD}/feat_dense1500_{p}.npy'
            if not os.path.exists(f):
                _, Z = train_feat(p, 'dense1500'); np.save(f, Z)
            Zw = np.load(f); Zs.append(Zw[rc.wi[r], rc.li[r]])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(fs), (np.concatenate(Zs) if with_feat else None)


# ---------------------------------------------------------------- decoders
decode_prod = lab.decode


def eiou_curve(p):
    """exact E[IoU] of the top-m prefixes (descending p) for m = 1..n_pos (n_pos = #cells with p > 0, at least 1).
    I = |S n T| and O = |T \\ S| are independent Poisson-binomials -> E[I/(m+O)] = E[I] * E[1/(m+O)];
    the law of O (cells m+1..n_pos) by the exact Poisson-binomial DP, built from the end.  Returns (order, curve)."""
    p = np.asarray(p, np.float64)
    order = np.argsort(-p, kind='stable'); ps = p[order]
    npos = max(int((ps > 0).sum()), 1)
    q = ps[:npos]; cs = np.cumsum(q)
    curve = np.zeros(npos)
    g = np.ones(1)                                    # law of O for m = npos (no positive cell outside)
    for m in range(npos, 0, -1):
        curve[m - 1] = cs[m - 1] * float(np.sum(g / (m + np.arange(len(g)))))
        pm = q[m - 1]                                 # cell m (1-based) leaves S when m -> m-1
        g = np.r_[g * (1 - pm), 0.0] + np.r_[0.0, g * pm]
    return order, curve


def decode_exact(p):
    order, curve = eiou_curve(p)
    j = int(np.argmax(curve))
    out = np.zeros(len(p), bool); out[order[:j + 1]] = True
    return out


# ---------------------------------------------------------------- scoring
def p6_from_rows(W, L, wi, li, p):
    P6 = np.zeros((W, F, L), np.float32); P6[wi, F - 1, li] = p
    return P6


def score(truth, ev, P6, widx, decoder, meta):
    """per-window IoU (production iou: visible cells, empty union -> 1) for the windows widx; meta = dict(panel, family, win)."""
    out = []
    for i in widx:
        pred = decoder(P6[i].ravel()).reshape(F, -1)
        T = truth[i] & ev[i]; Pp = pred & ev[i]
        u = int((T | Pp).sum()); it = int((T & Pp).sum())
        out.append((int(meta['win'].d.iloc[i]), int(meta['win'].t.iloc[i]), int(meta['win'].fold.iloc[i]), 1.0 if u == 0 else it / u,
                    u, it, int(T.sum()), int(Pp.sum())))
    df = pd.DataFrame(out, columns=['d', 't', 'fold', 'iou', 'union', 'inter', 'n_true', 'n_pred'])
    df['panel'] = meta['panel']; df['family'] = meta['panel'].rsplit('_', 1)[0]; df['cond'] = lab.ONSET
    return df


KEY = ['family', 'panel', 'd', 't', 'cond']


def frame(df, iou):
    """window frame (KEY + iou) for lab.paired_bootstrap / lab.aggregate."""
    out = df[KEY].copy(); out['iou'] = np.asarray(iou, np.float64); out['set'] = 'x'
    return out


def delta_stats(base_df, new_df):
    """paired delta (new - base) with CI, panel deltas, non-tie panels."""
    bt = lab.paired_bootstrap(base_df, new_df)
    m = base_df.merge(new_df, on=KEY, suffixes=('_a', '_b'))
    pdl = (m.iou_b - m.iou_a).groupby(m.panel).mean()
    nt = pdl[pdl.abs() > 1e-9]
    return dict(diff=bt['diff'], lo=bt['lo'], hi=bt['hi'], n=bt['n_windows'], base=lab.aggregate(base_df), new=lab.aggregate(new_df),
                panel_delta={k: round(float(v), 5) for k, v in pdl.items()}, nontie=int(len(nt)), nontie_pos=int((nt > 0).sum()),
                nontie_min=float(nt.min()) if len(nt) else float('nan'), panels_nonneg=int((pdl >= -1e-9).sum()), panels=int(len(pdl)))
