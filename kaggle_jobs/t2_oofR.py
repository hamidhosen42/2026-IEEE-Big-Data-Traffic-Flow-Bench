"""OOF-R (EXP_030 + EXP_031): identical to OOF-X control (original link order, same windows/folds/seeds), plus
(a) released ramp flows (history slots t-3..t-1 only, i.e. before T): on/off-ramp flow attached to the link and summed over +-3 links,
(b) explicit bottleneck-propensity x live-instability interactions. P255 (ongoing, prior-free: ramp feats kept, interactions dropped
with the priors) and O_cat8 (onset, all features).
OOF-F notes: identical to OOF-X (same windows, folds, seeds) except links are ordered in TRAVEL direction in every panel:
W/S panels are reversed so index+1 = downstream everywhere (topology audit: upstream has higher milepost in W/S panels,
lower in E/N). Only P255 (ongoing) and O_cat8 (onset) are trained, to compare with the oofX controls.
"""
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
from catboost import CatBoostClassifier

T0 = time.time()
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
OUT = '/kaggle/working'
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']
SMOKE = False
if SMOKE:
    PANELS = ['D7_I10_E', 'D12_I5_S']
H, F, SPACING = 12, 6, 72
N_TREES = 60 if SMOKE else 500
MAX_OG = 300 if SMOKE else 1500
FLIP = False


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


# ------------------------------------------------------------------ data
def load_panel(panel):
    net = f'{REL}/corridors/{panel}/network'
    topo = pd.read_csv(f'{net}/lwr_mainline_topology.csv', dtype={'link_id': str}).sort_values('order_index').reset_index(drop=True)
    if FLIP and panel[-1] in 'WS':                       # travel direction = decreasing order_index in W/S panels
        topo = topo.iloc[::-1].reset_index(drop=True)
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
    # released ramp flows aggregated to attached mainline links (NaN = not observed / not attached)
    ON = np.full((D, 288, L), np.nan, np.float32); OFF = ON.copy()
    amap = pd.read_csv(f'{net}/ramp_attachment_map.csv', dtype=str)
    amap = amap[amap.nearest_mainline_link_id.isin(li)]
    rl = dict(zip(amap.ramp_link_id, amap.nearest_mainline_link_id.map(li)))
    dpos = {dd: i for i, dd in enumerate(days)}
    for f in sorted(glob.glob(f'{REL}/corridors/{panel}/train/ramp_states/**/*.parquet', recursive=True)):
        r = pd.read_parquet(f, columns=['timestamp', 'ramp_link_id', 'ramp_type', 'flow_vph', 'pct_observed', 'is_score_eligible'])
        r = r[(r.is_score_eligible == 1) & r.flow_vph.notna() & (r.pct_observed >= 75) & r.ramp_link_id.isin(rl)]
        if r.empty:
            continue
        ts = pd.to_datetime(r.timestamp, utc=True)
        r = r.assign(d=ts.dt.strftime('%Y-%m-%d').map(dpos), t=(ts.dt.hour * 12 + ts.dt.minute // 5), l=r.ramp_link_id.map(rl),
                     on=r.ramp_type.str.upper().isin(['OR', 'ON'])).dropna(subset=['d'])
        for flag, A in ((True, ON), (False, OFF)):
            g = r[r.on == flag].groupby(['d', 't', 'l']).flow_vph.sum().reset_index()
            A[g.d.astype(int).to_numpy(), g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.flow_vph.to_numpy()
    return dict(panel=panel, links=links, li=li, cut=cut, vf=vf, cap=cap, lanes=lanes, length=length,
                days=days, dow=dow, V=V, Fq=Fq, E=E, Q=Q, ON=ON, OFF=OFF)


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


def onset_prior_table(P, onset_wins, half=12):
    """share of training onset windows (same day class, origin within +-half slots) whose T+30 truth contains each link."""
    L = len(P['links']); cnt = np.zeros((2, 288, L), np.float32); n = np.zeros((2, 288, 1), np.float32)
    for d, t, _ in onset_wins:
        cls = int(P['dow'][d] >= 5); lo, hi = max(0, t - half), min(288, t + half + 1)
        cnt[cls, lo:hi] += P['Q'][d, min(t + F, 287)][None].astype(np.float32); n[cls, lo:hi] += 1
    return cnt / np.maximum(n, 1), n


def make_rows(P, wins, prior, with_labels=True, par=False):
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
    # (a) ramp features from slots t-3..t-1 (strictly before T)
    ridx = t[:, None] + np.arange(-3, 0)[None]
    capl = P['cap'][None, :]
    with np.errstate(all='ignore'):
        on3 = np.nanmean(P['ON'][d[:, None], ridx], 1) / capl; off3 = np.nanmean(P['OFF'][d[:, None], ridx], 1) / capl
        on12 = np.nanmean(P['ON'][d[:, None], t[:, None] + np.arange(-H, -3)[None]], 1) / capl
    Fe['ramp_on3'] = on3; Fe['ramp_off3'] = off3; Fe['ramp_on_surge'] = on3 - on12
    for nm, a in (('on', on3), ('off', off3)):
        a0 = np.nan_to_num(a)
        Fe[f'ramp_{nm}_pm3'] = sum(shift_links(a0, k) for k in (-3, -2, -1, 1, 2, 3)).astype(np.float32)
    rows = []
    for h in range(1, F + 1):
        slot = np.minimum(t + h, 287)
        feats = {k: v for k, v in Fe.items()}
        feats['h'] = np.full((W, L), h, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = prior_feats(prior, P['dow'][d], t, h)
        # (b) propensity x live instability (dropped with the priors in the prior-free ongoing model)
        inst = np.clip(1.6 - np.nan_to_num(Fe['vr_min3'], nan=1.6), 0, 1.6)
        feats['ix_prior_inst'] = feats['prior'] * inst
        feats['ix_prior_flow'] = feats['prior'] * np.nan_to_num(Fe['fr_last'])
        feats['ix_prior_dn'] = feats['prior'] * np.clip(1.6 - np.nan_to_num(Fe['vr_min_dn1_3'], nan=1.6), 0, 1.6)
        feats['ix_prior_up'] = feats['prior'] * np.clip(1.6 - np.nan_to_num(Fe['vr_min_up1_3'], nan=1.6), 0, 1.6)
        feats['ix_prior_ramp'] = feats['prior'] * np.nan_to_num(Fe['ramp_on_surge'])
        cls_ = (P['dow'][d] >= 5).astype(int)
        if par:   # training rows: prior from the opposite day parity only (no self-contribution)
            (o_ev, n_ev), (o_od, n_od) = P['oprior_par']
            odd = (d % 2 == 1)[:, None]
            feats['oprior'] = np.where(odd, o_ev[cls_, t], o_od[cls_, t])
            feats['oprior_n'] = np.where(odd, n_ev[cls_, t], n_od[cls_, t]).astype(np.float32) * np.ones((1, o_ev.shape[2]), np.float32)
        else:
            op, opn = P['oprior']
            feats['oprior'] = op[cls_, t]; feats['oprior_n'] = np.broadcast_to(opn[cls_, t], (len(d), op.shape[2])).astype(np.float32)
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


PRIOR_COLS = {'ix_prior_inst', 'ix_prior_flow', 'ix_prior_dn', 'ix_prior_up', 'ix_prior_ramp', 'prior', 'prior_now', 'prior_alldays', 'oprior', 'oprior_n', 'pos', 'tod', 'dow', 'lanes', 'cap_lane', 'length', 'cap_ratio_hi', 'cap_ratio_lo'}


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
for k in range(4):
    Xtr = {c: [] for c in ('queue_onset', 'queue_ongoing')}; ytr = {c: [] for c in Xtr}
    evalR = {}
    for panel, P in panels.items():
        D = len(P['days']); block = np.minimum(np.arange(D) * 4 // D, 3)
        test_days = set(np.where(block == k)[0])
        prior = prior_table(P, block != k)
        trw = [w for w in P['trainw'] if w[0] not in test_days]
        onw = [w for w in trw if w[2] == 'queue_onset']
        P['oprior'] = onset_prior_table(P, onw)                                           # fold-safe, for eval rows
        P['oprior_par'] = (onset_prior_table(P, [w for w in onw if w[0] % 2 == 0]),        # even-day table (used by odd-day rows)
                           onset_prior_table(P, [w for w in onw if w[0] % 2 == 1]))
        R = make_rows(P, trw, prior, par=True)
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
    cols_np = [x for x in cols if x not in PRIOR_COLS]; ix_np = [cols.index(x) for x in cols_np]
    X = np.concatenate(Xtr['queue_ongoing']); y = np.concatenate(ytr['queue_ongoing']).astype(np.float32)
    T = (lambda n: 40 if SMOKE else n)
    m_b255 = {'queue_ongoing': lgb.train(dict(PARAMS, num_leaves=255, min_data_in_leaf=50), lgb.Dataset(X[:, ix_np], y), T(1000))}
    # onset variants (all features)
    Xo = np.concatenate(Xtr['queue_onset']); yo = np.concatenate(ytr['queue_onset']).astype(int)
    so = np.random.default_rng(k).choice(len(Xo), min(len(Xo), 1_200_000 if not SMOKE else 100_000), replace=False)
    onsets = {}
    for nm, kw in (('cat8', dict(depth=8, iterations=600, learning_rate=0.08)),):
        cb = CatBoostClassifier(thread_count=4, verbose=0, random_seed=k, **dict(kw, iterations=40 if SMOKE else kw['iterations']))
        cb.fit(Xo[so], yo[so]); onsets[nm] = cb
    log(f'fold {k} ongoing models trained on {len(X)} rows')
    for panel, (Re, Ro) in evalR.items():
        for tag, R in (('pseudo', Re), ('official', Ro)):
            if R is None:
                continue
            og_only = dict(R); og_only['cond'] = R['cond']
            out = {'P255': predict_windows(R, m_b255, cols_np)}
            Xoo, _, _, meta, _ = to_matrix(R, 'queue_onset', cols, None)
            for nm, mdl in onsets.items():
                arr = np.zeros_like(out['P255'])
                if len(Xoo):
                    arr[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = mdl.predict_proba(Xoo)[:, 1] if nm.startswith('cat') else mdl.predict(Xoo)
                out['O_' + nm] = arr
            Pp = panels[panel]
            truth = np.stack([Pp['Q'][d_, t_ + 1:t_ + 1 + F] for d_, t_ in zip(R['d'], R['t'])])
            ev = np.stack([Pp['E'][d_, t_ + 1:t_ + 1 + F] for d_, t_ in zip(R['d'], R['t'])])
            np.savez_compressed(f'{OUT}/oofR_{panel}_{tag}_f{k}.npz', **{kk: vv.astype(np.float16) for kk, vv in out.items()},
                                truth=truth, ev=ev, cond=R['cond'], d=R['d'], t=R['t'], dow=Pp['dow'][R['d']])
            log(f'saved oofR {panel} {tag} fold {k}: {R["W"]} windows')

log('OOF-R cache done')
