"""PSEUDO-MARCH GATE (EXP_024). Trains the production Task 2 models exactly as v13, then scores them on
pseudo-windows selected from the released VALIDATION month (masked layer) with observed labels.
History masked cells are filled with Task 1 F predictions (coverage like window_history); slot T = observed only.
Labels = observed eligible horizon cells <= cut. Windows touching the official windows' blanked horizon+buffer are excluded.
The gate must reproduce the signs of known LB deltas before it is trusted. Use: decisions for the PRIVATE file only.

v13 = v12 with ongoing 511 leaves / lr 0.03 / 1800 trees and 3000 ongoing windows per panel (oofY). Task 2 model (Kaggle): LightGBM cell classifier per condition + expected-IoU mask decoding.

Only data at or before the forecast origin T is used for features (the 12-step history).
Training windows come from the replicated official window selector run over the 9 train months.
Validation: 4 contiguous day blocks (frozen); scored on builder-faithful pseudo-windows and on the
80 official train windows (models trained without those windows' days). Truth proxy = observed
eligible queue state (speed <= 0.6 * free speed); the metric is IoU on sensored-eligible cells.
Final: models refit on all train windows, applied to validation + private windows.
"""
import glob, json, os, time, gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier

T0 = time.time()
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
OUT = '/kaggle/working'
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']
SMOKE = True
if SMOKE:
    PANELS = ['D7_I10_E', 'D12_I5_S']
H, F, SPACING = 12, 6, 72
N_TREES = 60 if SMOKE else 500
MAX_OG = 300 if SMOKE else 3000


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


# ------------------------------------------------------------------ data
def load_panel(panel):
    net = f'{REL}/corridors/{panel}/network'
    topo = pd.read_csv(f'{net}/lwr_mainline_topology.csv', dtype={'link_id': str}).sort_values('order_index').reset_index(drop=True)
    links = topo.link_id.tolist(); li = {l: i for i, l in enumerate(links)}
    lk = pd.read_csv(f'{net}/links.csv', dtype={'link_id': str}).set_index('link_id').reindex(links)
    vf = lk.free_speed_kmh.fillna(105).to_numpy(); cut = 0.6 * vf
    cap = lk.capacity_vph.fillna(7200).to_numpy(); lanes = lk.lanes.fillna(4).to_numpy()
    length = lk.length_km.fillna(0.5).to_numpy()
    files = sorted(glob.glob(f'{REL}/corridors/{panel}/train/mainline_states/*/*.parquet'))
    if SMOKE:
        files = files[:90]
    days = [os.path.basename(f)[-18:-8].replace('_', '-') for f in files]
    D, L = len(files), len(links)
    V = np.full((D, 288, L), np.nan, np.float32); Fq = V.copy()
    for d, f in enumerate(files):
        x = pd.read_parquet(f, columns=['timestamp', 'link_id', 'speed_kmh', 'flow_vph', 'is_score_eligible'])
        x = x[(x.is_score_eligible == 1) & x.speed_kmh.notna()]
        ts = pd.to_datetime(x.timestamp, utc=True)
        x = x.assign(t=(ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(), l=x.link_id.astype(str).map(li)).dropna(subset=['l'])
        g = x.groupby(['t', 'l'])[['speed_kmh', 'flow_vph']].mean().reset_index()
        V[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.speed_kmh.to_numpy()
        Fq[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.flow_vph.to_numpy()
    E = np.isfinite(V); Q = E & (V <= cut[None, None, :])
    dow = pd.to_datetime(pd.Series(days)).dt.dayofweek.to_numpy()
    return dict(panel=panel, links=links, li=li, cut=cut, vf=vf, cap=cap, lanes=lanes, length=length,
                days=days, dow=dow, V=V, Fq=Fq, E=E, Q=Q)


def established(Qh):
    return Qh.sum() >= 2 and (Qh.sum(0) >= 2).any()


def iou(truth, pred, mask):
    t, p = truth[mask], pred[mask]; u = (t | p).sum()
    return 1.0 if u == 0 else (t & p).sum() / u


def build_windows(P, global_spacing=True, dense_ongoing=False):
    D = len(P['days']); out = []; taken = []
    for cond in ('queue_onset', 'queue_ongoing'):
        if not global_spacing:
            taken = []
        for d in range(D):
            for t in range(H, 288 - F):
                g = d * 288 + t
                if not (dense_ongoing and cond == 'queue_ongoing') and any(abs(g - x) < SPACING for x in taken[-400:]):
                    continue
                if dense_ongoing and cond == 'queue_ongoing' and t % 3:
                    continue
                Qf = P['Q'][d, t + 1:t + 1 + F]
                if not Qf.any() or P['E'][d, t - H:t].mean() < 0.7:
                    continue
                if (cond == 'queue_ongoing') != established(P['Q'][d, t - H:t]):
                    continue
                if cond == 'queue_ongoing' and iou(Qf, np.repeat(P['Q'][d, t][None], F, 0), P['E'][d, t + 1:t + 1 + F]) > 0.9:
                    continue
                out.append((d, t, cond)); taken.append(g); taken.sort()
    return out


# ------------------------------------------------------------------ features
def ffill_time(a):
    """forward-fill NaNs along axis 1 (time) of [W, T, L]."""
    out = a.copy()
    for i in range(1, a.shape[1]):
        m = np.isnan(out[:, i]); out[:, i][m] = out[:, i - 1][m]
    return out


def shift_links(a, k):
    out = np.full_like(a, np.nan)
    if k > 0: out[..., :-k] = a[..., k:]
    elif k < 0: out[..., -k:] = a[..., :k]
    else: out[:] = a
    return out


def dist_to_queued(q, direction):
    """distance in links to the nearest queued link at higher (+1) or lower (-1) index; 99 if none. q: [W, L] bool."""
    W, L = q.shape; out = np.full((W, L), 99.0, np.float32)
    idx = np.arange(L)
    for w in range(W):
        pos = np.where(q[w])[0]
        if len(pos) == 0:
            continue
        if direction > 0:
            j = np.searchsorted(pos, idx, side='left'); ok = j < len(pos)
            out[w, ok] = pos[j[ok]] - idx[ok]
        else:
            j = np.searchsorted(pos, idx, side='right') - 1; ok = j >= 0
            out[w, ok] = idx[ok] - pos[j[ok]]
    return out


def window_features(P, Vh, Fh, tod, dow, prior, VT):
    """Vh, Fh: [W, 12, L] history (NaN = missing). Returns dict name -> [W, L] and filled queue state at T."""
    cut = P['cut'][None, None, :]; cap = P['cap'][None, None, :]
    vr = Vh / cut; fr = Fh / cap
    vrf = ffill_time(vr); frf = ffill_time(fr)
    q_hist = np.nan_to_num(vr <= 1.0, nan=0).astype(bool) & np.isfinite(vr)
    vT = VT / P['cut'][None, :]
    last = np.where(np.isfinite(vT), vT, vrf[:, -1]); qlast = np.nan_to_num(last, nan=9) <= 1.0
    q15 = np.nan_to_num(vrf[:, -4], nan=9) <= 1.0
    W, L = last.shape
    Fe = {'vr_last': last, 'vr_raw_last': vr[:, -1], 'vr_l1': vrf[:, -2], 'vr_l2': vrf[:, -3], 'vr_l3': vrf[:, -4],
          'vr_l6': vrf[:, -7], 'vr_l11': vrf[:, 0], 'vr_min3': np.nanmin(np.where(np.isfinite(vr[:, -3:]), vr[:, -3:], 9), 1),
          'vr_mean12': np.nanmean(np.where(np.isfinite(vr), vr, np.nan), 1),
          'vr_slope': np.nanmean(vr[:, -3:], 1) - np.nanmean(vr[:, :3], 1),
          'fr_last': frf[:, -1], 'fr_mean3': np.nanmean(fr[:, -3:], 1), 'fr_slope': np.nanmean(fr[:, -3:], 1) - np.nanmean(fr[:, :3], 1),
          'q_last': qlast.astype(np.float32), 'q_count': q_hist.sum(1).astype(np.float32),
          'q_age': np.where(q_hist.any(1), 11 - np.argmax(q_hist[:, ::-1], 1) * 0 - (11 - np.argmax(q_hist[:, ::-1], 1)), 12).astype(np.float32),
          'miss_last': np.isnan(vr[:, -1]).astype(np.float32), 'vr_T': vT, 'has_T': np.isfinite(vT).astype(np.float32)}
    Fe['q_age'] = np.where(q_hist.any(1), np.argmax(q_hist[:, ::-1], 1), 12).astype(np.float32)
    for k in (-3, -2, -1, 1, 2, 3):
        Fe[f'vr_n{k:+d}'] = shift_links(last, k); Fe[f'q_n{k:+d}'] = shift_links(qlast.astype(np.float32), k)
        Fe[f'fr_n{k:+d}'] = shift_links(frf[:, -1], k)
    for a, b in ((1, 3), (4, 8), (9, 16)):
        for sgn, nm in ((1, 'dn'), (-1, 'up')):
            stack = np.stack([shift_links(np.nan_to_num(last, nan=9), sgn * k) for k in range(a, b + 1)])
            Fe[f'vr_min_{nm}{a}_{b}'] = np.nanmin(np.where(np.isfinite(stack), stack, 9), 0)
    Fe['dq_hi'] = dist_to_queued(qlast, 1); Fe['dq_lo'] = dist_to_queued(qlast, -1)
    Fe['dq_hi15'] = dist_to_queued(q15, 1); Fe['dq_lo15'] = dist_to_queued(q15, -1)
    q30 = np.nan_to_num(vrf[:, -7], nan=9) <= 1.0
    Fe['dq_hi30'] = dist_to_queued(q30, 1); Fe['dq_lo30'] = dist_to_queued(q30, -1)
    Fe['share_q'] = np.repeat(qlast.mean(1, keepdims=True), L, 1); Fe['share_q15'] = np.repeat(q15.mean(1, keepdims=True), L, 1)
    Fe['n_q_hist_win'] = np.repeat(q_hist.any(1).sum(1, keepdims=True).astype(np.float32), L, 1)
    Fe['pos'] = np.broadcast_to(np.linspace(0, 1, L, dtype=np.float32)[None], (W, L))
    Fe['lanes'] = np.broadcast_to(P['lanes'][None].astype(np.float32), (W, L))
    Fe['cap_lane'] = np.broadcast_to((P['cap'] / P['lanes'])[None].astype(np.float32), (W, L))
    Fe['cap_ratio_hi'] = np.broadcast_to((np.r_[P['cap'][1:], P['cap'][-1]] / P['cap'])[None].astype(np.float32), (W, L))
    Fe['cap_ratio_lo'] = np.broadcast_to((np.r_[P['cap'][0], P['cap'][:-1]] / P['cap'])[None].astype(np.float32), (W, L))
    Fe['length'] = np.broadcast_to(P['length'][None].astype(np.float32), (W, L))
    Fe['tod'] = np.repeat(tod[:, None].astype(np.float32), L, 1); Fe['dow'] = np.repeat(dow[:, None].astype(np.float32), L, 1)
    return Fe, qlast


def prior_table(P, day_mask):
    """P(queued | link, slot) by day class: [0]=weekday, [1]=weekend, [2]=all days (train days in day_mask)."""
    out = []
    wkend = P['dow'] >= 5
    for m in (day_mask & ~wkend, day_mask & wkend, day_mask):
        E = P['E'][m]; Q = P['Q'][m]
        with np.errstate(all='ignore'):
            out.append(Q.sum(0) / np.maximum(E.sum(0), 1))
    return np.stack(out).astype(np.float32)          # [3, 288, L]


def prior_feats(prior, dow, tod, h):
    cls = (np.asarray(dow) >= 5).astype(int); slot = np.minimum(np.asarray(tod) + h, 287)
    return prior[cls, slot], prior[cls, np.asarray(tod)], prior[2, slot]


def make_rows(P, wins, prior, with_labels=True):
    """Stack per-(window, link, step) rows. onset -> step 6 only."""
    if not wins:
        return None
    d = np.array([w[0] for w in wins]); t = np.array([w[1] for w in wins]); cond = [w[2] for w in wins]
    idx = t[:, None] + np.arange(-H, 0)[None]
    Vh = P['V'][d[:, None], idx]; Fh = P['Fq'][d[:, None], idx]
    # slot T itself is only visible through the Task-1-masked layer (~50-66% of cells): simulate that
    rng = np.random.default_rng(len(wins))
    VT = np.where(rng.random((len(d), P['V'].shape[2])) < 0.62, P['V'][d, t], np.nan)
    Fe, qlast = window_features(P, Vh, Fh, t, P['dow'][d], prior, VT)
    W, L = qlast.shape
    rows = []
    for h in range(1, F + 1):
        slot = np.minimum(t + h, 287)
        feats = {k: v for k, v in Fe.items()}
        feats['h'] = np.full((W, L), h, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = prior_feats(prior, P['dow'][d], t, h)
        lab = P['Q'][d, np.minimum(t + h, 287)] if with_labels else None
        ev = P['E'][d, np.minimum(t + h, 287)] if with_labels else None
        rows.append((h, feats, lab, ev))
    return dict(d=d, t=t, cond=np.array(cond), rows=rows, qlast=qlast, W=W, L=L)


def to_matrix(R, cond, cols=None, keep=None):
    """Flatten rows for one condition into X, y, (w_idx, l_idx, h)."""
    sel = np.where(R['cond'] == cond)[0]
    Xs, ys, es, meta = [], [], [], []
    steps = [6] if cond == 'queue_onset' else range(1, 7)
    for h, feats, lab, ev in R['rows']:
        if h not in steps:
            continue
        cols = cols or list(feats)
        M = np.stack([np.asarray(feats[c])[sel] for c in cols], -1)      # [w, L, f]
        wi, lidx = np.meshgrid(sel, np.arange(R['L']), indexing='ij')
        m = np.ones(wi.shape, bool) if keep is None else keep[sel]
        Xs.append(M[m]); meta.append(np.stack([wi[m], lidx[m], np.full(m.sum(), h)], 1))
        if lab is not None:
            ys.append(lab[sel][m]); es.append(ev[sel][m])
    X = np.concatenate(Xs).astype(np.float32); meta = np.concatenate(meta)
    y = np.concatenate(ys) if ys else None; e = np.concatenate(es) if es else None
    return X, y, e, meta, cols


def ongoing_keep(R, radius=12):
    """keep cells near queue activity: within `radius` links of a queue now / 15 / 30 min ago,
    or slow now (<= 1.3 x v_cut in the last 15 min), or historically queued at this slot (>= 5%)."""
    f = R['rows'][0][1]
    near = lambda a, b: np.minimum(f[a], f[b]) <= radius
    k = near('dq_hi', 'dq_lo') | near('dq_hi15', 'dq_lo15') | near('dq_hi30', 'dq_lo30')
    k |= np.nan_to_num(f['vr_min3'], nan=9) <= 1.3
    k |= np.nan_to_num(f['prior_now'], nan=0) >= 0.05
    return k


def decode(p):
    """choose the top-m cells maximizing sum(top p) / (m + sum(rest p)); at least one cell."""
    order = np.argsort(-p); ps = p[order]; cs = np.cumsum(ps); tot = cs[-1]
    m = np.arange(1, len(p) + 1)
    j = np.argmax(cs / (m + (tot - cs)))
    out = np.zeros(len(p), bool); out[order[:j + 1]] = True
    return out


PARAMS = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, num_threads=4)


PRIOR_COLS = {'prior', 'prior_now', 'prior_alldays', 'pos', 'tod', 'dow', 'lanes', 'cap_lane', 'length', 'cap_ratio_hi', 'cap_ratio_lo'}


def predict_windows(R, models, cols):
    """probabilities [W, 6, L] for all windows in R."""
    P6 = np.zeros((R['W'], F, R['L']), np.float32)
    for cond, mdl in models.items():
        keep = ongoing_keep(R) if cond == 'queue_ongoing' else None
        X, _, _, meta, _ = to_matrix(R, cond, cols, keep)
        if len(X):
            P6[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = mdl.predict(X)
    return P6


def score_windows(P, R, P6, name, rows_out, panel, extra=None):
    for i in range(R['W']):
        d, t, cond = R['d'][i], R['t'][i], R['cond'][i]
        truth = P['Q'][d, t + 1:t + 1 + F]; ev = P['E'][d, t + 1:t + 1 + F]
        pred = decode(P6[i].ravel()).reshape(F, -1)
        rows_out.append(dict(panel=panel, cond=cond, d=int(d), strategy=name, iou=iou(truth, pred, ev), **(extra or {})))


# ------------------------------------------------------------------ main
panels = {}
for panel in PANELS:
    P = load_panel(panel)
    tw = build_windows(P, global_spacing=False, dense_ongoing=True)
    rng = np.random.default_rng(0)
    og = [w for w in tw if w[2] == 'queue_ongoing']; on = [w for w in tw if w[2] == 'queue_onset']
    if len(og) > MAX_OG:
        og = [og[i] for i in sorted(rng.choice(len(og), MAX_OG, replace=False))]
    P['trainw'] = on + og
    P['evalw'] = build_windows(P, global_spacing=True)
    wi = pd.read_csv(f'{REL}/task2/{panel}/train/window_index.csv')
    P['official'] = [(P['days'].index(pd.Timestamp(r.forecast_origin).strftime('%Y-%m-%d')),
                      pd.Timestamp(r.forecast_origin).hour * 12 + pd.Timestamp(r.forecast_origin).minute // 5, r.condition)
                     for r in wi.itertuples() if pd.Timestamp(r.forecast_origin).strftime('%Y-%m-%d') in P['days']]
    log(panel, 'train windows', pd.Series([w[2] for w in P['trainw']]).value_counts().to_dict(),
        'eval windows', len(P['evalw']), 'official', len(P['official']))
    panels[panel] = P

# ---- CV: 4 contiguous day blocks, pooled model across panels
results = []; imp_all = []
for k in range(0):
    Xtr = {c: [] for c in ('queue_onset', 'queue_ongoing')}; ytr = {c: [] for c in Xtr}
    evalR = {}
    for panel, P in panels.items():
        D = len(P['days']); block = np.minimum(np.arange(D) * 4 // D, 3)
        test_days = set(np.where(block == k)[0])
        prior = prior_table(P, block != k)
        trw = [w for w in P['trainw'] if w[0] not in test_days]
        R = make_rows(P, trw, prior)
        kp = ongoing_keep(R); ogm = (R['cond'] == 'queue_ongoing')
        pos = sum(((lab & ev)[ogm]).sum() for _, _, lab, ev in R['rows'])
        pos_in = sum(((lab & ev & kp)[ogm]).sum() for _, _, lab, ev in R['rows'])
        log(f'fold {k} {panel}: ongoing keep covers {pos_in / max(pos, 1):.4f} of labeled queued cells')
        for c in Xtr:
            keep = ongoing_keep(R) if c == 'queue_ongoing' else None
            X, y, e, meta, cols = to_matrix(R, c, None, keep)
            Xtr[c].append(X[e]); ytr[c].append(y[e])
        evalR[panel] = (make_rows(P, [w for w in P['evalw'] if w[0] in test_days], prior),
                        make_rows(P, [w for w in P['official'] if w[0] in test_days], prior))
        del R; gc.collect()
    models = {}; models_np = {}
    cols_np = [x for x in cols if x not in PRIOR_COLS]; ix_np = [cols.index(x) for x in cols_np]
    for c in Xtr:
        X = np.concatenate(Xtr[c]); y = np.concatenate(ytr[c])
        models[c] = lgb.train(PARAMS, lgb.Dataset(X, y.astype(np.float32)), N_TREES)
        models_np[c] = lgb.train(PARAMS, lgb.Dataset(X[:, ix_np], y.astype(np.float32)), N_TREES)
        imp_all.append(pd.Series(models[c].feature_importance('gain'), index=cols).rename(f'{c}_{k}'))
        log(f'fold {k} {c}: {len(X)} rows, pos rate {y.mean():.3f}')
    for panel, (Re, Ro) in evalR.items():
        for tag, R in (('pseudo', Re), ('official', Ro)):
            if R is None:
                continue
            P6 = predict_windows(R, models, cols)
            P6n = predict_windows(R, models_np, cols_np)
            og = (R['cond'] == 'queue_ongoing')[:, None, None]
            wk = panels[panel]['dow'][R['d']] >= 5
            for name, pr in (('lgbm_decode', P6), ('og_noprior', np.where(og, P6n, 0.7 * P6 + 0.3 * P6n)),
                             ('og_noprior_on100', np.where(og, P6n, P6))):
                for i in range(R['W']):
                    d_, t_, c_ = R['d'][i], R['t'][i], R['cond'][i]
                    truth = panels[panel]['Q'][d_, t_ + 1:t_ + 1 + F]; ev = panels[panel]['E'][d_, t_ + 1:t_ + 1 + F]
                    results.append(dict(panel=panel, cond=c_, d=int(d_), strategy=name, set=tag, fold=k, weekend=bool(wk[i]),
                                        iou=iou(truth, decode(pr[i].ravel()).reshape(F, -1), ev)))
            # reference: persistence with last-valid fill for ongoing
            P6p = np.zeros_like(P6)
            for i in range(R['W']):
                if R['cond'][i] == 'queue_ongoing':
                    P6p[i] = R['qlast'][i][None].astype(np.float32)
            score_windows(panels[panel], R, P6p + 1e-6, 'persist_ffill', results, panel, dict(set=tag, fold=k))

# ---- final models on all train windows, predict validation + private
Xtr = {c: [] for c in ('queue_onset', 'queue_ongoing')}; ytr = {c: [] for c in Xtr}
for panel, P in panels.items():
    P['prior_all'] = prior_table(P, np.ones(len(P['days']), bool))
    R = make_rows(P, P['trainw'], P['prior_all'])
    for c in Xtr:
        keep = ongoing_keep(R) if c == 'queue_ongoing' else None
        X, y, e, meta, cols = to_matrix(R, c, None, keep)
        Xtr[c].append(X[e]); ytr[c].append(y[e])
cols_np = [x for x in cols if x not in PRIOR_COLS]; ix_np = [cols.index(x) for x in cols_np]
final = {c: lgb.train(PARAMS, lgb.Dataset(np.concatenate(Xtr[c]), np.concatenate(ytr[c]).astype(np.float32)), N_TREES) for c in Xtr}
final_np = {c: lgb.train(PARAMS, lgb.Dataset(np.concatenate(Xtr[c])[:, ix_np], np.concatenate(ytr[c]).astype(np.float32)), N_TREES) for c in Xtr}
_Xo = np.concatenate(Xtr['queue_onset']); _yo = np.concatenate(ytr['queue_onset']).astype(int)
_sub = np.random.default_rng(0).choice(len(_Xo), min(len(_Xo), 1_200_000), replace=False)
cat_on = CatBoostClassifier(iterations=600, depth=8, learning_rate=0.08, thread_count=4, verbose=0, random_seed=0)
cat_on.fit(_Xo[_sub], _yo[_sub]); log('catboost onset trained on', len(_sub), 'rows (v9 features)')
_Xg = np.concatenate(Xtr['queue_ongoing'])[:, ix_np]; _yg = np.concatenate(ytr['queue_ongoing']).astype(np.float32)
og_big = {'queue_ongoing': lgb.train(dict(PARAMS, num_leaves=511, min_data_in_leaf=40, learning_rate=0.03), lgb.Dataset(_Xg, _yg), 1800)}  # v13: oofY P511 +0.006 in 4/4 folds
og_big2 = {'queue_ongoing': lgb.train(dict(PARAMS, num_leaves=255, min_data_in_leaf=50), lgb.Dataset(_Xg, _yg), 1000)}
log('big ongoing models trained on', len(_Xg), 'rows')
del Xtr, ytr; gc.collect()

# ================================================================== pseudo-March gate
og_np63 = {'queue_ongoing': final_np['queue_ongoing']}; og_full63 = {'queue_ongoing': final['queue_ongoing']}
on_full = {'queue_onset': final['queue_onset']}; on_np = {'queue_onset': final_np['queue_onset']}
ST = pd.read_csv(glob.glob('/kaggle/input/**/tfb-t1v8-full/**/state_submission_F.csv', recursive=True)[0],
                 dtype={'station_id': str, 'link_id': str}, usecols=['panel', 'timestamp', 'link_id', 'speed_kmh', 'flow_vph'])
ST['timestamp'] = pd.to_datetime(ST.timestamp, utc=True)
rows = []
for panel, P in panels.items():
    fs = sorted(glob.glob(f'{REL}/corridors/{panel}/validation/mainline_states_masked/*/*.parquet'))
    days = [os.path.basename(f)[-18:-8].replace('_', '-') for f in fs]
    D, L = len(fs), len(P['links'])
    Vo = np.full((D, 288, L), np.nan, np.float32); Fo = Vo.copy(); EL = np.zeros((D, 288, L), bool)
    for d, f in enumerate(fs):
        x = pd.read_parquet(f, columns=['timestamp', 'link_id', 'speed_kmh', 'flow_vph', 'is_score_eligible'])
        ts = pd.to_datetime(x.timestamp, utc=True)
        x = x.assign(t=(ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(), l=x.link_id.astype(str).map(P['li'])).dropna(subset=['l'])
        e = x[x.is_score_eligible == 1]
        EL[d, e.t.to_numpy(), e.l.astype(int).to_numpy()] = True
        o = e[e.speed_kmh.notna()].groupby(['t', 'l'])[['speed_kmh', 'flow_vph']].mean().reset_index()
        Vo[d, o.t.to_numpy(), o.l.astype(int).to_numpy()] = o.speed_kmh.to_numpy(); Fo[d, o.t.to_numpy(), o.l.astype(int).to_numpy()] = o.flow_vph.to_numpy()
    # Task 1 fill of masked eligible cells (history only)
    s = ST[ST.panel == panel]
    s = s.assign(dd=s.timestamp.dt.strftime('%Y-%m-%d').map({dd: i for i, dd in enumerate(days)}),
                 t=(s.timestamp.dt.hour * 12 + s.timestamp.dt.minute // 5), l=s.link_id.map(P['li'])).dropna(subset=['dd', 'l'])
    Vf = Vo.copy(); Ff = Fo.copy()
    Vf[s.dd.astype(int), s.t, s.l.astype(int)] = s.speed_kmh.to_numpy(); Ff[s.dd.astype(int), s.t, s.l.astype(int)] = s.flow_vph.to_numpy()
    Vf[~EL] = np.nan; Ff[~EL] = np.nan
    OBS = np.isfinite(Vo); Qo = OBS & (Vo <= P['cut'][None, None, :])
    # official windows: horizon + buffer blanked -> exclude pseudo windows overlapping [T0+1, T0+18]
    wi = pd.read_csv(f'{REL}/task2/{panel}/validation/window_index.csv')
    blk = np.zeros((D, 288), bool)
    for r in wi.itertuples():
        o = pd.Timestamp(r.forecast_origin); dd = o.strftime('%Y-%m-%d')
        if dd in days:
            t0 = o.hour * 12 + o.minute // 5; blk[days.index(dd), t0 + 1:min(288, t0 + 19)] = True
    wins = []
    for cond in ('queue_onset', 'queue_ongoing'):
        taken = []
        for d in range(D):
            for t in range(H, 288 - F):
                g = d * 288 + t
                if blk[d, t - H:t + F + 1].any() or any(abs(g - x) < SPACING for x in taken[-50:]):
                    continue
                Qf = Qo[d, t + 1:t + 1 + F]
                if not Qf.any() or EL[d, t - H:t].mean() < 0.7:
                    continue
                if (cond == 'queue_ongoing') != established(Qo[d, t - H:t]):
                    continue
                if cond == 'queue_ongoing' and iou(Qf, np.repeat(Qo[d, t][None], F, 0), OBS[d, t + 1:t + 1 + F]) > 0.9:
                    continue
                wins.append((d, t, cond)); taken.append(g)
    if not wins:
        continue
    d = np.array([w[0] for w in wins]); t = np.array([w[1] for w in wins]); cond = np.array([w[2] for w in wins])
    idx = t[:, None] + np.arange(-H, 0)[None]
    Vh = Vf[d[:, None], idx]; Fh = Ff[d[:, None], idx]; VT = Vo[d, t]
    dow = pd.to_datetime(pd.Series(days)).dt.dayofweek.to_numpy()
    Pm = dict(P); Pm['dow'] = dow
    Fe, qlast = window_features(Pm, Vh, Fh, t, dow[d], P['prior_all'], VT)
    Rr = []
    for h in range(1, F + 1):
        feats = dict(Fe); feats['h'] = np.full(qlast.shape, h, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = prior_feats(P['prior_all'], dow[d], t, h)
        Rr.append((h, feats, None, None))
    R = dict(d=d, t=t, cond=cond, qlast=qlast, W=len(d), L=L, rows=Rr)
    log(panel, 'pseudo-March windows', pd.Series(cond).value_counts().to_dict(), 'history filled share', round(float(np.isfinite(Vh).mean()), 3))
    Pf = predict_windows(R, {'queue_onset': final['queue_onset'], 'queue_ongoing': final['queue_ongoing']}, cols)
    Pn = predict_windows(R, {'queue_onset': final_np['queue_onset'], 'queue_ongoing': final_np['queue_ongoing']}, cols_np)
    Pg511 = predict_windows(R, og_big, cols_np); Pg255 = predict_windows(R, og_big2, cols_np)
    Pc = np.zeros_like(Pf)
    Xc, _, _, meta, _ = to_matrix(R, 'queue_onset', cols, None)
    if len(Xc):
        Pc[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = cat_on.predict_proba(Xc)[:, 1]
    pers = np.repeat(qlast[:, None, :].astype(np.float32), F, 1)
    V = {'og_full63': Pf, 'og_np63': Pn, 'og_np255': Pg255, 'og_np511': Pg511, 'og_np63_pers30': 0.7 * Pn + 0.3 * pers,
         'on_70_30': 0.7 * Pf + 0.3 * Pn, 'on_30_70': 0.3 * Pf + 0.7 * Pn, 'on_full': Pf, 'on_cat': Pc, 'persistence': pers + 1e-6}
    for i in range(len(d)):
        truth = Qo[d[i], t[i] + 1:t[i] + 1 + F]; ev = OBS[d[i], t[i] + 1:t[i] + 1 + F]
        for nm, pr in V.items():
            if nm.startswith('og_') and cond[i] != 'queue_ongoing':
                continue
            if nm.startswith('on_') and cond[i] != 'queue_onset':
                continue
            rows.append(dict(panel=panel, cond=cond[i], d=int(d[i]), t=int(t[i]), variant=nm,
                             iou=iou(truth, decode(pr[i].ravel()).reshape(F, -1), ev)))
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/pseudo_march_scores.csv', index=False)
g = df.groupby(['variant', 'cond', 'panel']).iou.mean().reset_index()
g['fam'] = g.panel.str.rsplit('_', n=1).str[0]
tab = g.groupby(['variant', 'cond', 'fam']).iou.mean().groupby(['variant', 'cond']).mean().unstack()
print('\n===== pseudo-March family-mean IoU =====\n', tab.round(4).to_string())
print('windows', df.drop_duplicates(['panel', 'd', 't']).cond.value_counts().to_dict())
def m(v, c): return float(tab.loc[v, c])
print('SIGN CHECK 1 (LB +): og_np63 - og_full63 =', round(m('og_np63', 'queue_ongoing') - m('og_full63', 'queue_ongoing'), 4))
print('SIGN CHECK 2 (LB -): og_np63_pers30 - og_np63 =', round(m('og_np63_pers30', 'queue_ongoing') - m('og_np63', 'queue_ongoing'), 4))
print('SIGN CHECK 3 (LB -): on_30_70 - on_70_30 =', round(m('on_30_70', 'queue_onset') - m('on_70_30', 'queue_onset'), 4))
print('info (LB +small): on_cat - on_70_30 =', round(m('on_cat', 'queue_onset') - m('on_70_30', 'queue_onset'), 4))
print('info (LB +): og_np255 - og_np63 =', round(m('og_np255', 'queue_ongoing') - m('og_np63', 'queue_ongoing'), 4),
      '| og_np511 - og_np255 =', round(m('og_np511', 'queue_ongoing') - m('og_np255', 'queue_ongoing'), 4))
log('pseudo-March gate done')
