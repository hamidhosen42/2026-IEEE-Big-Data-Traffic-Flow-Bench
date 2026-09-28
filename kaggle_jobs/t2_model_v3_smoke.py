"""Task 2 model (Kaggle): LightGBM cell classifier per condition + expected-IoU mask decoding.

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

T0 = time.time()
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
OUT = '/kaggle/working'
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']
SMOKE = True
if SMOKE:
    PANELS = ['D7_I10_E', 'D12_I5_S']
H, F, SPACING = 12, 6, 72
N_TREES = 60 if SMOKE else 800
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


def masked_queue_cube(panel, split, links, li, cut, days=None):
    """Queue state seen through the released masked layer (what val/private also show before T)."""
    fs = sorted(glob.glob(f'{REL}/corridors/{panel}/{split}/mainline_states_masked/*/*.parquet'),
                key=lambda f: os.path.basename(f)[-18:-8])
    fdays = [os.path.basename(f)[-18:-8].replace('_', '-') for f in fs]
    if days is not None:
        keep = [i for i, dd in enumerate(fdays) if dd in set(days)]; fs = [fs[i] for i in keep]; fdays = [fdays[i] for i in keep]
    Qm = np.zeros((len(fs), 288, len(links)), np.float32); Em = np.zeros_like(Qm)
    for d, f in enumerate(fs):
        x = pd.read_parquet(f, columns=['timestamp', 'link_id', 'speed_kmh', 'is_score_eligible'])
        x = x[(x.is_score_eligible == 1) & x.speed_kmh.notna()]
        ts = pd.to_datetime(x.timestamp, utc=True)
        x = x.assign(t=(ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(), l=x.link_id.astype(str).map(li)).dropna(subset=['l'])
        g = x.groupby(['t', 'l']).speed_kmh.mean().reset_index()
        l_ = g.l.astype(int).to_numpy()
        Qm[d, g.t.to_numpy(), l_] = (g.speed_kmh.to_numpy() <= cut[l_]); Em[d, g.t.to_numpy(), l_] = 1
    return fdays, Qm, Em


def context_features(Qm, Em, d, t):
    """per-window [W, L] context from the masked layer, strictly before the history window / on earlier days."""
    W = len(d); L = Qm.shape[2]
    out = {k: np.full((W, L), np.nan, np.float32) for k in ('ctx_today', 'ctx_prev1', 'ctx_prev7', 'ctx_prev7_any')}
    for i in range(W):
        di, ti = d[i], t[i]
        lo = max(0, ti - 12 - 36)
        if ti - 12 > lo:
            e = Em[di, lo:ti - 12].sum(0); out['ctx_today'][i] = np.where(e > 0, Qm[di, lo:ti - 12].sum(0) / np.maximum(e, 1), np.nan)
        hs = slice(ti + 1, min(ti + 7, 288))
        if di >= 1:
            e = Em[di - 1, hs].sum(0); out['ctx_prev1'][i] = np.where(e > 0, Qm[di - 1, hs].sum(0) / np.maximum(e, 1), np.nan)
        if di >= 2:
            a = max(0, di - 7)
            e = Em[a:di, hs].sum((0, 1)); qq = Qm[a:di, hs].sum((0, 1))
            out['ctx_prev7'][i] = np.where(e > 0, qq / np.maximum(e, 1), np.nan)
            out['ctx_prev7_any'][i] = (Qm[a:di, hs].max(1) * 1.0).mean(0)
    return out


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
    """P(queued | link, time slot) over the given train days."""
    E = P['E'][day_mask]; Q = P['Q'][day_mask]
    with np.errstate(all='ignore'):
        p = Q.sum(0) / np.maximum(E.sum(0), 1)
    return p.astype(np.float32)          # [288, L]


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
    Fe.update(context_features(P['Qm'], P['Em'], d, t))
    W, L = qlast.shape
    rows = []
    for h in range(1, F + 1):
        slot = np.minimum(t + h, 287)
        pri = prior[slot]                                   # [W, L]
        feats = {k: v for k, v in Fe.items()}
        feats['h'] = np.full((W, L), h, np.float32); feats['prior'] = pri
        feats['prior_now'] = prior[t]
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
    """keep links within `radius` links of any link queued anywhere in the visible history."""
    return (np.minimum(R['rows'][0][1]['dq_hi'], R['rows'][0][1]['dq_lo']) <= radius) | \
           (np.minimum(R['rows'][0][1]['dq_hi15'], R['rows'][0][1]['dq_lo15']) <= radius)


def decode(p):
    """choose the top-m cells maximizing sum(top p) / (m + sum(rest p)); at least one cell."""
    order = np.argsort(-p); ps = p[order]; cs = np.cumsum(ps); tot = cs[-1]
    m = np.arange(1, len(p) + 1)
    j = np.argmax(cs / (m + (tot - cs)))
    out = np.zeros(len(p), bool); out[order[:j + 1]] = True
    return out


PARAMS = dict(objective='binary', learning_rate=0.04, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, num_threads=4)


PRIOR_COLS = {'prior', 'prior_now', 'pos', 'tod', 'dow', 'lanes', 'cap_lane', 'length', 'cap_ratio_hi', 'cap_ratio_lo'}


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
    mdays, P['Qm'], P['Em'] = masked_queue_cube(panel, 'train', P['links'], P['li'], P['cut'], P['days'])
    assert mdays == P['days'], 'masked/full day mismatch'
    wi = pd.read_csv(f'{REL}/task2/{panel}/train/window_index.csv')
    P['official'] = [(P['days'].index(pd.Timestamp(r.forecast_origin).strftime('%Y-%m-%d')),
                      pd.Timestamp(r.forecast_origin).hour * 12 + pd.Timestamp(r.forecast_origin).minute // 5, r.condition)
                     for r in wi.itertuples() if pd.Timestamp(r.forecast_origin).strftime('%Y-%m-%d') in P['days']]
    log(panel, 'train windows', pd.Series([w[2] for w in P['trainw']]).value_counts().to_dict(),
        'eval windows', len(P['evalw']), 'official', len(P['official']))
    panels[panel] = P

# ---- CV: 4 contiguous day blocks, pooled model across panels
results = []; imp_all = []
for k in range(4):
    Xtr = {c: [] for c in ('queue_onset', 'queue_ongoing')}; ytr = {c: [] for c in Xtr}
    evalR = {}
    for panel, P in panels.items():
        D = len(P['days']); block = np.minimum(np.arange(D) * 4 // D, 3)
        test_days = set(np.where(block == k)[0])
        prior = prior_table(P, block != k)
        trw = [w for w in P['trainw'] if w[0] not in test_days]
        R = make_rows(P, trw, prior)
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
            score_windows(panels[panel], R, P6, 'lgbm_decode', results, panel, dict(set=tag, fold=k))
            score_windows(panels[panel], R, P6n, 'lgbm_noprior', results, panel, dict(set=tag, fold=k))
            score_windows(panels[panel], R, 0.5 * P6 + 0.5 * P6n, 'blend_50', results, panel, dict(set=tag, fold=k))
            score_windows(panels[panel], R, 0.7 * P6 + 0.3 * P6n, 'blend_70', results, panel, dict(set=tag, fold=k))
            # reference: persistence with last-valid fill for ongoing
            P6p = np.zeros_like(P6)
            for i in range(R['W']):
                if R['cond'][i] == 'queue_ongoing':
                    P6p[i] = R['qlast'][i][None].astype(np.float32)
            score_windows(panels[panel], R, P6p + 1e-6, 'persist_ffill', results, panel, dict(set=tag, fold=k))

Rdf = pd.DataFrame(results); Rdf.to_csv(f'{OUT}/t2_cv_scores.csv', index=False)
pd.concat(imp_all, axis=1).to_csv(f'{OUT}/t2_importance.csv')


def agg(df):
    if df.empty:
        return pd.DataFrame({'S_queue': []})
    g = df.groupby(['strategy', 'panel', 'cond']).iou.mean().groupby(['strategy', 'panel']).mean().reset_index()
    g['fam'] = g.panel.str.rsplit('_', n=1).str[0]
    tot = g.groupby(['strategy', 'fam']).iou.mean().groupby('strategy').mean()
    byc = df.groupby(['strategy', 'cond']).iou.mean().unstack()
    return pd.concat([tot.rename('S_queue'), byc], axis=1).round(4)


for tag in ('pseudo', 'official'):
    print(f'\n===== {tag} windows (4 contiguous day blocks) =====')
    print(agg(Rdf[Rdf.set == tag]).to_string())
    for strat in ('lgbm_decode', 'blend_50', 'blend_70'):
        fold_scores = [agg(Rdf[(Rdf.set == tag) & (Rdf.fold == k)]).S_queue.get(strat, np.nan) for k in range(4)]
        print(strat, 'per fold:', np.round(fold_scores, 4).tolist())

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
del Xtr, ytr; gc.collect()
subs = []
for panel, P in panels.items():
    for split in ('validation', 'private'):
        wi = pd.read_csv(f'{REL}/task2/{panel}/{split}/window_index.csv')
        hist = pd.read_parquet(f'{REL}/task2/{panel}/{split}/window_history.parquet')
        tpl = pd.read_csv(f'{REL}/task2/{panel}/{split}/sample_submission_queue.csv', dtype=str)
        W, L = len(wi), len(P['links'])
        Vh = np.full((W, H, L), np.nan, np.float32); Fh = Vh.copy()
        wpos = {w: i for i, w in enumerate(wi.window_id)}
        hist = hist[(hist.is_score_eligible == 1) & hist.speed_kmh.notna()]
        org = pd.to_datetime(wi.set_index('window_id').forecast_origin, utc=True)
        ts = pd.to_datetime(hist.timestamp, utc=True)
        step = ((ts - hist.window_id.map(org)).dt.total_seconds() // 300).astype(int) + H
        ok = (step >= 0) & (step < H) & hist.link_id.isin(P['li'])
        hh = hist[ok].assign(s=step[ok].values, w=hist[ok].window_id.map(wpos).values, l=hist[ok].link_id.map(P['li']).values)
        g = hh.groupby(['w', 's', 'l'])[['speed_kmh', 'flow_vph']].mean().reset_index()
        Vh[g.w, g.s, g.l] = g.speed_kmh; Fh[g.w, g.s, g.l] = g.flow_vph
        tod = (org.dt.hour * 12 + org.dt.minute // 5).to_numpy()
        dow = org.dt.dayofweek.to_numpy()
        VT = np.full((W, L), np.nan, np.float32)
        for date, grp in wi.groupby('date'):
            fs = glob.glob(f'{REL}/corridors/{panel}/{split}/mainline_states_masked/*/*{date.replace("-", "_")}.parquet')
            if not fs:
                continue
            m = pd.read_parquet(fs[0], columns=['timestamp', 'link_id', 'speed_kmh', 'is_score_eligible'])
            m = m[(m.is_score_eligible == 1) & m.speed_kmh.notna()]
            for r in grp.itertuples():
                x = m[m.timestamp == r.forecast_origin]
                x = x[x.link_id.isin(P['li'])].groupby('link_id').speed_kmh.mean()
                VT[wpos[r.window_id], x.index.map(P['li']).to_numpy()] = x.to_numpy()
        log(panel, split, 'slot-T visible share', round(float(np.isfinite(VT).mean()), 3),
            'history filled share', round(float(np.isfinite(Vh).mean()), 3))
        Fe, qlast = window_features(P, Vh, Fh, tod, dow, P['prior_all'], VT)
        sdays, Qm_s, Em_s = masked_queue_cube(panel, split, P['links'], P['li'], P['cut'])
        dpos = {dd: i for i, dd in enumerate(sdays)}
        Fe.update(context_features(Qm_s, Em_s, np.array([dpos[x] for x in wi.date]), tod))
        R = dict(d=np.zeros(W, int), t=tod, cond=wi.condition.to_numpy(), qlast=qlast, W=W, L=L, rows=[])
        for h in range(1, F + 1):
            feats = dict(Fe); feats['h'] = np.full((W, L), h, np.float32)
            feats['prior'] = P['prior_all'][np.minimum(tod + h, 287)]; feats['prior_now'] = P['prior_all'][tod]
            R['rows'].append((h, feats, None, None))
        P6a = predict_windows(R, final, cols); P6n = predict_windows(R, final_np, cols_np)
        preds = {name: np.stack([decode(p[i].ravel()).reshape(F, L) for i in range(W)])
                 for name, p in (('main', P6a), ('blend70', 0.7 * P6a + 0.3 * P6n), ('blend50', 0.5 * P6a + 0.5 * P6n))}
        pred = preds['main']
        tpl['w'] = tpl.window_id.map(wpos); tpl['l'] = tpl.link_id.map(P['li'])
        tts = pd.to_datetime(tpl.timestamp, utc=True)
        tpl['h'] = ((tts - tpl.window_id.map(org)).dt.total_seconds() // 300).astype(int)
        okr = tpl.w.notna() & tpl.l.notna() & tpl.h.between(1, F)
        for name, pr in preds.items():
            col = 'queue_pred' if name == 'main' else f'queue_pred_{name}'
            tpl[col] = 0
            tpl.loc[okr, col] = pr[tpl.w[okr].astype(int), tpl.h[okr].astype(int) - 1, tpl.l[okr].astype(int)].astype(int)
        log(panel, split, 'windows', W, 'template rows', len(tpl), 'unmapped', int((~okr).sum()),
            'positive share', round(float(tpl.queue_pred.mean()), 4))
        subs.append(tpl[['window_id', 'timestamp', 'link_id', 'queue_pred', 'queue_pred_blend70', 'queue_pred_blend50']])
S = pd.concat(subs, ignore_index=True)
S[['window_id', 'timestamp', 'link_id', 'queue_pred']].to_csv(f'{OUT}/queue_submission.csv', index=False)
for name in ('blend70', 'blend50'):
    S[['window_id', 'timestamp', 'link_id', f'queue_pred_{name}']].rename(columns={f'queue_pred_{name}': 'queue_pred'}).to_csv(f'{OUT}/queue_submission_{name}.csv', index=False)
log('queue rows', len(S), 'dupes', int(S.duplicated(['window_id', 'timestamp', 'link_id']).sum()))
