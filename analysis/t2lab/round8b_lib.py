"""round8b_lib.py -- ROUND 8b (results/round8/PREREG_8b.md, sha1 asserted on import; reading in INTERPRETATION_8b.txt).
Production pv recipe of consolidation/tfb/stages/onset_pv.py rebuilt offline (EXP_T2_OSP_MT protocol, exp_onset_marchtrain.py):
production-exact train onset rows (slot-T draw, permutation), March onset rows of days < cut with the k = 4 scenario prior blend,
evaluation on the round-8 E1 pseudo-March onset windows with day >= cut, production decoder."""
import gc, hashlib, os, pickle, warnings
import numpy as np, pandas as pd
import lab, ogfeat_lib as og
import phase_lib as ph

SHA8B = '235618405f00af4c39e788870f8d77b76d47e635'
RES = f'{lab.ROOT}/analysis/t2lab/results/round8'
CD = f'{lab.CACHE}/round8b'
MD = f'{CD}/models'
for _d in (CD, MD):
    os.makedirs(_d, exist_ok=True)
H, F, SPACING = lab.H, lab.F, lab.SPACING
COLS69 = lab.all_cols()
COLS = [c for c in COLS69 if not c.startswith('ramp_')]            # the 64 production pv columns
KEEP = [COLS69.index(c) for c in COLS]
JP = [COLS.index(c) for c in lab.PRIOR3]
WS = (10, 20, 40)
MAX_OG = 3000
K_BLEND = 4


def check_prereg():
    s = hashlib.sha1(open(f'{RES}/PREREG_8b.md', 'rb').read()).hexdigest()
    assert s == SHA8B, f'PREREG_8b.md changed: {s}'
    assert open(f'{RES}/PREREG_8b.sha1').read().split()[0] == SHA8B
    return s


check_prereg()


def fit(X, y, w, seed):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=600, depth=8, learning_rate=0.08, thread_count=4, verbose=0, random_seed=seed,
                           allow_writing_files=False)
    m.fit(X, y, sample_weight=w); return m


# ---------------------------------------------------------------- train onset rows
def train_rows(recipe):
    """recipe 'prod': production-exact _Xo_keep (production slot-T draw over onset + capped ongoing windows, permuted);
    'prodnoperm': same rows unpermuted; 'lab': lab dense1500 onset rows (lab slot-T draw), unpermuted (= old EXP_T2_OSP_MT)."""
    f = f'{CD}/train_{recipe}.npz'
    if os.path.exists(f):
        z = np.load(f); return z['X'], z['y']
    Xs, ys = [], []
    for panel in lab.PANELS:
        if recipe == 'lab':
            rc = lab.rows(panel, 'dense1500', lab.ONSET); r = np.where(rc.e)[0]
            Xs.append(np.asarray(rc.X[r])[:, KEEP]); ys.append(rc.y[r].astype(int)); continue
        P = lab.load_panel(panel)
        tw = pickle.load(open(f'{lab.CACHE}/windows/{panel}__denseall.pkl', 'rb'))
        on = [w for w in tw if w[2] == lab.ONSET]; n_og = min(sum(w[2] == lab.ONGOING for w in tw), MAX_OG)
        L = len(P['links']); W = len(on)
        mask = np.random.default_rng(W + n_og).random((W + n_og, L))[:W] < 0.62
        d = np.array([w[0] for w in on]); t = np.array([w[1] for w in on])
        idx = t[:, None] + np.arange(-H, 0)[None]
        VT = np.where(mask, P['V'][d, t], np.nan)
        prior = lab.prior_table(P, np.ones(len(P['days']), bool))
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            Fe, qlast = lab.prod.window_features(P, P['V'][d[:, None], idx], P['Fq'][d[:, None], idx], t, P['dow'][d], prior, VT)
        feats = dict(Fe); feats['h'] = np.full((W, L), F, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = lab.prior_feats(prior, P['dow'][d], t, F)
        R = dict(d=d, t=t, cond=np.array([lab.ONSET] * W), W=W, L=L, qlast=qlast, rows=[(F, feats, P['Q'][d, t + F], P['E'][d, t + F])])
        X, y, e, meta, cols = lab.to_matrix(R, lab.ONSET, None, None); assert list(cols) == COLS, cols
        # check against the cached lab rows: same rows, identical columns that do not depend on the slot-T draw
        rc = lab.rows(panel, 'dense1500', lab.ONSET); r = np.where(rc.e)[0]
        Xl = np.asarray(rc.X[r])[:, KEEP]; assert len(Xl) == int(e.sum()) and np.array_equal(rc.y[r], y[e])
        indep = [j for j, c in enumerate(COLS) if c in ('vr_raw_last', 'vr_l1', 'vr_l6', 'fr_last', 'q_count', 'pos', 'tod', 'prior')]
        assert np.array_equal(X[e][:, indep], Xl[:, indep], equal_nan=True), panel
        Xs.append(X[e]); ys.append(y[e].astype(int))
        lab.drop_panel(panel)
    X = np.concatenate(Xs); y = np.concatenate(ys)
    if recipe == 'prod':
        sub = np.random.default_rng(0).choice(len(X), min(len(X), 1_200_000), replace=False); X, y = X[sub], y[sub]
    np.savez(f, X=X, y=y)
    return X, y


# ---------------------------------------------------------------- March rows (days < cut), production march_onset_windows + blend
def march_onset_windows(V, Em, cut_speed, day_cut):
    E = np.isfinite(V); Q = E & (np.nan_to_num(V, nan=1e9) <= cut_speed[None, None, :])
    blank = (Em.sum(2) >= 5) & (E.sum(2) == 0)
    picks, last = [], -10 ** 9
    for d in range(min(V.shape[0], day_cut)):
        for t in range(H, 288 - F):
            g = d * 288 + t
            if g - last < SPACING or blank[d, t - H:t + F + 1].any() or Em[d, t - H:t].mean() < 0.7:
                continue
            if lab.established(Q[d, t - H:t]) or not Q[d, t + 1:t + 1 + F].any():
                continue
            picks.append((d, t)); last = g
    return picks, Q, E


def blend(X, wi, li, d, t, dow, prior, cQ, cE, k=K_BLEND):
    """production blend_prior: the three prior columns blended with the scenario prior of the days < the window's day."""
    cls = (dow[d] >= 5).astype(int); s6 = np.minimum(t + 6, 287)
    cr, jr, s6r, tr_ = cls[wi], d[wi], s6[wi], t[wi]
    trv = [prior[cr, s6r, li], prior[cr, tr_, li], prior[2, s6r, li]]
    Xk = X.copy()
    for j, (cc, ss) in enumerate(((cr, s6r), (cr, tr_), (np.full_like(cr, 2), s6r))):
        q = cQ[cc, jr, ss, li]; ee = cE[cc, jr, ss, li]
        sp = np.where(ee > 0, q / np.maximum(ee, 1e-9), 0.0).astype(np.float32); n = ee / 5.0; a_ = n / (n + k)
        Xk[:, JP[j]] = (1 - a_) * trv[j] + a_ * sp
    return Xk


def march_rows(day_cut, recipe='prod'):
    f = f'{CD}/march_c{day_cut}_{recipe}.npz'
    if os.path.exists(f):
        z = np.load(f); return z['X'], z['y']
    Xs, ys, info = [], [], {}
    for panel in lab.PANELS:
        P, Pv = og.val_panel(panel)
        picks, Qm, Emv = march_onset_windows(Pv['V'], Pv['Em'], P['cut'], day_cut)
        info[panel] = len(picks)
        if not picks:
            continue
        d = np.array([w[0] for w in picks]); t = np.array([w[1] for w in picks]); idx = t[:, None] + np.arange(-H, 0)[None]
        prior = lab.prior_table(P, np.ones(len(P['days']), bool))
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            Fe, qlast = lab.prod.window_features(P, Pv['V'][d[:, None], idx], Pv['Fq'][d[:, None], idx], t, Pv['dow'][d], prior, Pv['V'][d, t])
        W, L = qlast.shape
        feats = dict(Fe); feats['h'] = np.full((W, L), F, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = lab.prior_feats(prior, Pv['dow'][d], t, F)
        R = dict(d=d, t=t, cond=np.array([lab.ONSET] * W), W=W, L=L, qlast=qlast, rows=[(F, feats, Qm[d, t + F], Emv[d, t + F])])
        X, y, e, meta, cols = lab.to_matrix(R, lab.ONSET, None, None); assert list(cols) == COLS
        cQ, cE = ph.cum_tables(Pv, True)
        Xk = blend(X, meta[:, 0], meta[:, 1], d, t, Pv['dow'], prior, cQ, cE)
        Xs.append(Xk[e]); ys.append(y[e].astype(int))
        del cQ, cE; lab.drop_panel(panel); gc.collect()
    X = np.concatenate(Xs); y = np.concatenate(ys)
    np.savez(f, X=X, y=y)
    lab.log(f'March rows cut {day_cut}: windows {info} rows {len(X)} pos rate {y.mean():.4f}')
    return X, y


# ---------------------------------------------------------------- evaluation: round-8 E1 windows with day >= cut (sp4 rows, 64 columns)
def eval_set(day_cut):
    out = {}
    for panel in lab.PANELS:
        e = pd.read_pickle(f'{lab.CACHE}/round8/e1_{panel}.pkl')
        keepw = np.where(e['win'].d.to_numpy() >= day_cut)[0]
        if not len(keepw):
            continue
        r = np.isin(e['wi'], keepw)
        out[panel] = dict(X=np.ascontiguousarray(e['Xs4'][r][:, KEEP]), wi=e['wi'][r], li=e['li'][r], widx=keepw, W=e['W'], L=e['L'],
                          win=e['win'], truth=e['truth'], ev=e['ev'])
    return out


def score(ev, probs):
    """per-window IoU (production decoder, step-6 probabilities) over the eval windows; probs: panel -> row probabilities."""
    rows = []
    for panel, s in ev.items():
        P6 = np.zeros((s['W'], F, s['L']), np.float32); P6[s['wi'], F - 1, s['li']] = probs[panel]
        for i in s['widx']:
            pred = lab.decode(P6[i].ravel()).reshape(F, -1)
            T = s['truth'][i] & s['ev'][i]; Pp = pred & s['ev'][i]
            u = int((T | Pp).sum()); it = int((T & Pp).sum())
            rows.append((panel, panel.rsplit('_', 1)[0], int(s['win'].d.iloc[i]), int(s['win'].t.iloc[i]), 1.0 if u == 0 else it / u))
    df = pd.DataFrame(rows, columns=['panel', 'family', 'd', 't', 'iou']); df['cond'] = lab.ONSET; df['set'] = 'x'
    return df


KEY = ['family', 'panel', 'd', 't', 'cond']


def frame(tmpl, iou):
    out = tmpl[KEY + ['set']].copy(); out['iou'] = np.asarray(iou, np.float64); return out


def delta_stats(base_df, new_df):
    bt = lab.paired_bootstrap(base_df, new_df)
    m = base_df.merge(new_df, on=KEY, suffixes=('_a', '_b'))
    pdl = (m.iou_b - m.iou_a).groupby(m.panel).mean()
    nt = pdl[pdl.abs() > 1e-9]
    return dict(diff=bt['diff'], lo=bt['lo'], hi=bt['hi'], n=bt['n_windows'], base=lab.aggregate(base_df), new=lab.aggregate(new_df),
                panel_delta={k: round(float(v), 5) for k, v in pdl.items()}, nontie=int(len(nt)), nontie_pos=int((nt > 0).sum()),
                nontie_min=float(nt.min()) if len(nt) else float('nan'))


def model_path(recipe, cut, seed, w):
    return f'{MD}/{recipe}_c{cut}_s{seed}_w{w}.cbm'
