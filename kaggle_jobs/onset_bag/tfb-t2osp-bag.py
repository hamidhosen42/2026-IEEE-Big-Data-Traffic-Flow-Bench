"""Task 2 onset with a SAME-SCENARIO location prior (osp).  validation (March) and private (April) come from one scenario seed
(release config: "The scenario seed for validation and private is drawn at random"); train comes from another, and the train
queue-location prior transfers badly (pseudo-March onset: train prior top-m IoU 0.72 vs 0.86 with same-month days).
Model: CatBoost onset exactly as v13 (same windows, rows, subsample, seed), trained on the train split with the all-train prior.
Inference: the three prior columns of every onset window are blended per cell with the prior of the val/private scenario
estimated from released masked-layer data of the days strictly before the window's day (validation windows: March days < d;
private windows: all of March + April days < d), i.e. only data timestamped before T:
  split prior = P(queued | link, slot +- 2, weekday class) over visible eligible cells, a = n / (n + k), n = visible days.
Outputs queue_submission_{base, sp2, sp4, sp8}.csv (all queue rows; ongoing rows are 0 - the chain swaps onset rows only)
and CONTROL lines: base onset rows vs tfb-t2v13 and vs the chain base file.
"""
import glob, json, os, time, gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
N_SEEDS = 10  # [R8]

class SeedBag:  # [R8] mean of the onset CatBoost over seeds 0..N_SEEDS-1 (production used random_seed 0 only)
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], 0)


def fit_bag(X, y, **fit_kw):  # [R8]
    ms = []
    for s in range(N_SEEDS):
        m = CatBoostClassifier(iterations=600, depth=8, learning_rate=0.08, thread_count=4, verbose=0, random_seed=s)
        m.fit(X, y, **fit_kw)
        ms.append(m)
    return SeedBag(ms)

T0 = time.time()
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
OUT = '/kaggle/working'
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']
SMOKE = False
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
KS = (2, 4, 8)
REF_V13 = 'tfb-t2v13'                       # CONTROL 1: the v13 kernel the chain's onset rows come from
REF_BASE = ('tfb-e3', 'submission_E3.csv')  # CONTROL 2: the current chain base file
panels = {}
for panel in PANELS:
    P = load_panel(panel)
    tw = build_windows(P, global_spacing=False, dense_ongoing=True)
    rng = np.random.default_rng(0)
    og = [w for w in tw if w[2] == 'queue_ongoing']; on = [w for w in tw if w[2] == 'queue_onset']
    if len(og) > MAX_OG:
        og = [og[i] for i in sorted(rng.choice(len(og), MAX_OG, replace=False))]
    P['trainw'] = on + og
    log(panel, 'train windows', pd.Series([w[2] for w in P['trainw']]).value_counts().to_dict())
    panels[panel] = P

# ---- CatBoost onset exactly as v13 (same windows, rows, subsample, seed)
Xo, yo = [], []
for panel, P in panels.items():
    P['prior_all'] = prior_table(P, np.ones(len(P['days']), bool))
    R = make_rows(P, P['trainw'], P['prior_all'])
    X, y, e, meta, cols = to_matrix(R, 'queue_onset', None, None)
    Xo.append(X[e]); yo.append(y[e])
    del R, X; gc.collect()
_Xo = np.concatenate(Xo); _yo = np.concatenate(yo).astype(int); del Xo, yo
_sub = np.random.default_rng(0).choice(len(_Xo), min(len(_Xo), 1_200_000), replace=False)
cat_on = fit_bag(_Xo[_sub], _yo[_sub]); log('catboost onset bag of', N_SEEDS, 'seeds trained on', len(_sub), 'rows (v9 features)')  # [R8]
del _Xo, _yo; gc.collect()
JP = [cols.index(c) for c in ('prior', 'prior_now', 'prior_alldays')]


def masked_cube(panel, split, li):
    """visible eligible speed of the released masked month: [D, 288, L] (NaN = not visible), sorted days."""
    files = sorted(glob.glob(f'{REL}/corridors/{panel}/{split}/mainline_states_masked/*/*.parquet'),
                   key=lambda f: os.path.basename(f)[-18:-8])           # files sit in mask_regime=R* folders: sort by date
    days = [os.path.basename(f)[-18:-8].replace('_', '-') for f in files]
    V = np.full((len(files), 288, len(li)), np.nan, np.float32)
    for d, f in enumerate(files):
        x = pd.read_parquet(f, columns=['timestamp', 'link_id', 'speed_kmh', 'is_score_eligible'])
        x = x[(x.is_score_eligible == 1) & x.speed_kmh.notna()]
        ts = pd.to_datetime(x.timestamp, utc=True)
        x = x.assign(t=(ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(), l=x.link_id.astype(str).map(li)).dropna(subset=['l'])
        g = x.groupby(['t', 'l']).speed_kmh.mean().reset_index()
        V[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.speed_kmh.to_numpy()
    return days, V


def scenario_cums(P):
    """cumulative (over days, by weekday class) queued / visible counts of the validation + private scenario, summed over slots
    +-2.  Day j of the timeline = March days then April days; cum[c, j] sums the days < j of class c (0 wkday, 1 wkend, 2 all)."""
    dv, Vv = masked_cube(P['panel'], 'validation', P['li']); dp, Vp = masked_cube(P['panel'], 'private', P['li'])
    days = dv + dp; V = np.concatenate([Vv, Vp]); del Vv, Vp
    assert days == sorted(days) and len(set(days)) == len(days), (len(days), len(set(days)))
    E = np.isfinite(V); Q = (E & (np.nan_to_num(V, nan=1e9) <= P['cut'][None, None, :])).astype(np.float32); E = E.astype(np.float32)

    def box(a):
        c = np.cumsum(np.pad(a, ((0, 0), (3, 2), (0, 0))), 1)
        return c[:, 5:] - c[:, :-5]
    Q, E = box(Q), box(E)
    wk = pd.to_datetime(pd.Series(days)).dt.dayofweek.to_numpy() >= 5
    D = len(days); cQ = np.zeros((3, D + 1) + Q.shape[1:], np.float32); cE = cQ.copy()
    for c, m in enumerate((~wk, wk, np.ones(D, bool))):
        cQ[c, 1:] = np.cumsum(Q * m[:, None, None], 0); cE[c, 1:] = np.cumsum(E * m[:, None, None], 0)
    log(P['panel'], 'scenario days', len(dv), '+', len(dp), 'visible share', round(float(np.isfinite(V).mean()), 3))
    return {dd: i for i, dd in enumerate(days)}, cQ, cE


subs = []; stats = []
for panel, P in panels.items():
    dpos, cQ, cE = scenario_cums(P)
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
        Fe, qlast = window_features(P, Vh, Fh, tod, dow, P['prior_all'], VT)
        R = dict(d=np.zeros(W, int), t=tod, cond=wi.condition.to_numpy(), qlast=qlast, W=W, L=L, rows=[])
        for h in range(1, F + 1):
            feats = dict(Fe); feats['h'] = np.full((W, L), h, np.float32)
            feats['prior'], feats['prior_now'], feats['prior_alldays'] = prior_feats(P['prior_all'], dow, tod, h)
            R['rows'].append((h, feats, None, None))
        Xc, _, _, meta, _ = to_matrix(R, 'queue_onset', cols, None)
        wv, lv = meta[:, 0], meta[:, 1]
        # same-scenario prior per onset row: days of the scenario strictly before the window's day (date < T's day)
        jd = np.array([dpos[pd.Timestamp(x).strftime('%Y-%m-%d')] for x in org.to_numpy()])
        cls = (dow >= 5).astype(int); s6 = np.minimum(tod + 6, 287)
        cr, jr, s6r, tr_ = cls[wv], jd[wv], s6[wv], tod[wv]
        trv = [P['prior_all'][cr, s6r, lv], P['prior_all'][cr, tr_, lv], P['prior_all'][2, s6r, lv]]
        spv, nv = [], []
        for cc, ss in ((cr, s6r), (cr, tr_), (np.full_like(cr, 2), s6r)):
            q = cQ[cc, jr, ss, lv]; ee = cE[cc, jr, ss, lv]
            spv.append(np.where(ee > 0, q / np.maximum(ee, 1e-9), 0.0).astype(np.float32)); nv.append(ee / 5.0)
        variants = {'base': None, **{f'sp{k}': k for k in KS}}
        preds = {}
        for name, k in variants.items():
            Xv = Xc.copy()
            if k is not None:
                for j, jp in enumerate(JP):
                    a_ = nv[j] / (nv[j] + k)
                    Xv[:, jp] = (1 - a_) * trv[j] + a_ * spv[j]
            P6 = np.zeros((W, F, L), np.float32)
            if len(Xv):
                P6[wv, meta[:, 2] - 1, lv] = cat_on.predict_proba(Xv)[:, 1]
            preds[name] = np.stack([decode(P6[i].ravel()).reshape(F, L) if R['cond'][i] == 'queue_onset' else np.zeros((F, L), bool)
                                    for i in range(W)])
        ison = R['cond'] == 'queue_onset'
        st = dict(panel=panel, split=split, onset_windows=int(ison.sum()), mean_scen_days=float(jd[ison].mean()),
                  mean_a4_prior=float((nv[0] / (nv[0] + 4)).mean()) if len(nv[0]) else 0.0)
        for name in variants:
            st[f'pos_{name}'] = int(preds[name][ison].sum())
            if name != 'base':
                a, b = preds['base'][ison], preds[name][ison]
                st[f'jac_{name}'] = float((a & b).sum() / max((a | b).sum(), 1))
        stats.append(st); log(json.dumps(st))
        tpl['w'] = tpl.window_id.map(wpos); tpl['l'] = tpl.link_id.map(P['li'])
        tts = pd.to_datetime(tpl.timestamp, utc=True)
        tpl['h'] = ((tts - tpl.window_id.map(org)).dt.total_seconds() // 300).astype(int)
        okr = tpl.w.notna() & tpl.l.notna() & tpl.h.between(1, F)
        assert okr.all(), 'unmapped template rows'
        for name, pr in preds.items():
            tpl[f'q_{name}'] = pr[tpl.w.astype(int), tpl.h.astype(int) - 1, tpl.l.astype(int)].astype(int)
        tpl['condition'] = tpl.window_id.map(wi.set_index('window_id').condition)
        subs.append(tpl[['window_id', 'timestamp', 'link_id', 'condition'] + [f'q_{n}' for n in variants]])
        del R, Xc; gc.collect()
    del cQ, cE; gc.collect()

S = pd.concat(subs, ignore_index=True)
log('queue rows', len(S), 'dupes', int(S.duplicated(['window_id', 'timestamp', 'link_id']).sum()))
print(pd.DataFrame(stats).round(4).to_string(index=False), flush=True)
for name in ('base',) + tuple(f'sp{k}' for k in KS):
    S[['window_id', 'timestamp', 'link_id', f'q_{name}']].rename(columns={f'q_{name}': 'queue_pred'}).to_csv(
        f'{OUT}/queue_submission_{name}.csv', index=False)

# ---- CONTROLS: the base variant must reproduce the onset rows in use
on = S.condition == 'queue_onset'
S['ts'] = pd.to_datetime(S.timestamp, utc=True)
try:
    v13 = pd.read_csv(glob.glob(f'/kaggle/input/**/{REF_V13}/**/queue_submission.csv', recursive=True)[0],
                      dtype={'window_id': str, 'link_id': str})
    v13['ts'] = pd.to_datetime(v13.timestamp, utc=True)
    m = S[on].merge(v13[['window_id', 'ts', 'link_id', 'queue_pred']], on=['window_id', 'ts', 'link_id'], how='left')
    a, b = m.q_base.to_numpy().astype(bool), m.queue_pred.fillna(-1).to_numpy() == 1
    log(f'CONTROL v13 onset agreement {float((a == b).mean()):.6f} jaccard {float((a & b).sum() / max((a | b).sum(), 1)):.5f} rows {len(m)}')
except Exception as ex:
    log('CONTROL v13 skipped:', type(ex).__name__, str(ex)[:200])
try:
    key = pd.read_csv(f'{REL}/submission_key.csv', dtype=str)
    kq = key[key.task == 'queue'][['submission_id', 'window_id', 'timestamp', 'link_id']].copy()
    kq['ts'] = pd.to_datetime(kq.timestamp, utc=True)
    base = pd.read_csv(glob.glob(f'/kaggle/input/**/{REF_BASE[0]}/**/{REF_BASE[1]}', recursive=True)[0],
                       usecols=['submission_id', 'task', 'queue_pred'])
    base = base[base.task == 'queue']; base['submission_id'] = base.submission_id.astype(str)
    m = S[on].merge(kq[['submission_id', 'window_id', 'ts', 'link_id']], on=['window_id', 'ts', 'link_id'], how='left')
    m = m.merge(base[['submission_id', 'queue_pred']], on='submission_id', how='left')
    for name in ('base',) + tuple(f'sp{k}' for k in KS):
        a, b = m[f'q_{name}'].to_numpy().astype(bool), m.queue_pred.fillna(-1).to_numpy() == 1
        log(f'CONTROL chain-base onset vs {name}: agreement {float((a == b).mean()):.6f} jaccard {float((a & b).sum() / max((a | b).sum(), 1)):.5f}'
            f' changed share {float((a != b).mean()):.5f} rows {len(m)}')
except Exception as ex:
    log('CONTROL base skipped:', type(ex).__name__, str(ex)[:200])
log('OSP DONE')


# ---- [R8] CONTROL: agreement of every written queue file with the original kernel tfb-t2osp
for fn in ['queue_submission_base.csv', 'queue_submission_sp4.csv']:
    try:
        a = pd.read_csv(f'{OUT}/{fn}', dtype={'window_id': str, 'link_id': str})
        b = pd.read_csv(glob.glob(f'/kaggle/input/**/tfb-t2osp/**/{fn}', recursive=True)[0], dtype={'window_id': str, 'link_id': str})
        m = a.merge(b, on=['window_id', 'timestamp', 'link_id'], how='outer', suffixes=('', '_orig'), indicator=True)
        assert (m._merge == 'both').all(), 'row sets differ'
        x, y = m.queue_pred.to_numpy().astype(bool), m.queue_pred_orig.to_numpy().astype(bool)
        log(f'R8 CONTROL {fn}: rows {len(m)} changed {int((x != y).sum())} pos {int(x.sum())} vs orig {int(y.sum())} '
            f'jaccard {(x & y).sum() / max((x | y).sum(), 1):.4f}')
    except Exception as ex:
        log('R8 CONTROL', fn, 'skipped:', type(ex).__name__, str(ex)[:200])
log('R8 BAG DONE seeds', N_SEEDS)
