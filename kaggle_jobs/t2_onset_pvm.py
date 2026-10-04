"""Task 2 onset for the PRIVATE month with same-scenario data (pv).  validation (March) and private (April) come from one scenario
seed (release config: "The scenario seed for validation and private is drawn at random"); train comes from another.
  1. CatBoost onset exactly as v13 (train split only) -> 'base' (CONTROL: must equal the onset rows in use).
  2. 'sp4': the same model with the three prior columns blended per cell with the scenario prior of the released masked-layer
     days strictly before the window's day (March windows: March days < d; April windows: all March + April days < d).
  3. 'pv': for APRIL windows only, a bag of 3 CatBoost models trained on the train rows + the onset windows of the released March
     month (observed labels, production onset rule, 72-slot spacing; sample weight 10 / 20 / 40), March rows carrying the blended
     prior of the March days before their own day; April features use the sp4 blend.  March windows keep sp4 (no March-trained
     model ever touches a March window).  All data used for an April window is dated before April 1 or before the window's day.
  Lab (pseudo-March onset windows of days >= 15, March days < 15 used as same-scenario data): production 0.707, sp4 0.812,
  pv bag 0.858 (6/6 panels).
  4. 'pvm' (public probe): MARCH windows on day d >= 1 use a CatBoost trained on the train rows + the released March onset
     windows of the days < d only (weight 20; data dated before T), with the sp4 blend; day-0 windows keep sp4; April = pv.
Outputs queue_submission_{base, sp4, pv, pvb, pvm}.csv (pvb = March base + April pv; ongoing rows 0: the chain swaps onset rows only).
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
cat_on = CatBoostClassifier(iterations=600, depth=8, learning_rate=0.08, thread_count=4, verbose=0, random_seed=0)
cat_on.fit(_Xo[_sub], _yo[_sub]); log('catboost onset trained on', len(_sub), 'rows (v9 features)')
_Xo_keep, _yo_keep = _Xo[_sub], _yo[_sub]
del _Xo, _yo; gc.collect()
JP = [cols.index(c) for c in ('prior', 'prior_now', 'prior_alldays')]
K_BLEND = 4
W_MARCH = (10, 20, 40)


def masked_cube(panel, split, li):
    """released masked month: visible eligible speed / flow [D, 288, L] (NaN = not visible), eligibility flag, sorted days."""
    files = sorted(glob.glob(f'{REL}/corridors/{panel}/{split}/mainline_states_masked/*/*.parquet'),
                   key=lambda f: os.path.basename(f)[-18:-8])           # files sit in mask_regime=R* folders: sort by date
    days = [os.path.basename(f)[-18:-8].replace('_', '-') for f in files]
    V = np.full((len(files), 288, len(li)), np.nan, np.float32); Fq = V.copy(); Em = np.zeros(V.shape, bool)
    for d, f in enumerate(files):
        x = pd.read_parquet(f, columns=['timestamp', 'link_id', 'speed_kmh', 'flow_vph', 'is_score_eligible'])
        ts = pd.to_datetime(x.timestamp, utc=True)
        x = x.assign(t=(ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(), l=x.link_id.astype(str).map(li)).dropna(subset=['l'])
        e = x[x.is_score_eligible == 1]
        Em[d, e.t.to_numpy(), e.l.astype(int).to_numpy()] = True
        x = x[(x.is_score_eligible == 1) & x.speed_kmh.notna()]
        g = x.groupby(['t', 'l'])[['speed_kmh', 'flow_vph']].mean().reset_index()
        V[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.speed_kmh.to_numpy()
        Fq[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.flow_vph.to_numpy()
    return days, V, Fq, Em


def scenario_cums(P):
    """cumulative (over days, by weekday class) queued / visible counts of the validation + private scenario, summed over slots
    +-2.  Day j of the timeline = March days then April days; cum[c, j] sums the days < j of class c (0 wkday, 1 wkend, 2 all)."""
    dv, Vv, Fv, Ev = masked_cube(P['panel'], 'validation', P['li']); dp, Vp, _, _ = masked_cube(P['panel'], 'private', P['li'])
    P['march'] = dict(days=dv, V=Vv, Fq=Fv, Em=Ev, dow=pd.to_datetime(pd.Series(dv)).dt.dayofweek.to_numpy())
    days = dv + dp; V = np.concatenate([Vv, Vp]); del Vp
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


def blend_prior(Xc, meta, P, jd, cls, tod, cQ, cE, k):
    """the three prior columns of onset rows (h = 6) blended per cell with the scenario prior of the days before the window's day."""
    wv, lv = meta[:, 0], meta[:, 1]
    s6 = np.minimum(tod + 6, 287)
    cr, jr, s6r, tr_ = cls[wv], jd[wv], s6[wv], tod[wv]
    trv = [P['prior_all'][cr, s6r, lv], P['prior_all'][cr, tr_, lv], P['prior_all'][2, s6r, lv]]
    Xk = Xc.copy(); a_mean = 0.0
    for j, (cc, ss) in enumerate(((cr, s6r), (cr, tr_), (np.full_like(cr, 2), s6r))):
        q = cQ[cc, jr, ss, lv]; ee = cE[cc, jr, ss, lv]
        sp = np.where(ee > 0, q / np.maximum(ee, 1e-9), 0.0).astype(np.float32); n = ee / 5.0; a_ = n / (n + k)
        Xk[:, JP[j]] = (1 - a_) * trv[j] + a_ * sp
        if j == 0 and len(a_):
            a_mean = float(a_.mean())
    return Xk, a_mean


def march_onset_windows(M, cut):
    """production onset rule on the visible masked March month (coverage from eligibility, windows touching a blanked slot
    dropped), picks >= 72 slots apart in one chronological scan (as build_windows for the onset condition)."""
    V, Em = M['V'], M['Em']; E = np.isfinite(V); Q = E & (np.nan_to_num(V, nan=1e9) <= cut[None, None, :])
    blank = (Em.sum(2) >= 5) & (E.sum(2) == 0)
    picks, last = [], -10 ** 9
    for d in range(V.shape[0]):
        for t in range(H, 288 - F):
            g = d * 288 + t
            if g - last < SPACING or blank[d, t - H:t + F + 1].any() or Em[d, t - H:t].mean() < 0.7:
                continue
            if established(Q[d, t - H:t]) or not Q[d, t + 1:t + 1 + F].any():
                continue
            picks.append((d, t)); last = g
    return picks, Q, E


# ---- same-scenario training rows: onset windows of the released March month (observed labels), blended prior features
scen = {}
Xm, ym, dm = [], [], []
for panel, P in panels.items():
    scen[panel] = scenario_cums(P)
    dpos, cQ, cE = scen[panel]
    M = P['march']
    picks, Qm, Emv = march_onset_windows(M, P['cut'])
    if not picks:
        continue
    d = np.array([w[0] for w in picks]); t = np.array([w[1] for w in picks])
    idx = t[:, None] + np.arange(-H, 0)[None]
    Fe, qlast = window_features(P, M['V'][d[:, None], idx], M['Fq'][d[:, None], idx], t, M['dow'][d], P['prior_all'], M['V'][d, t])
    W_, L_ = qlast.shape
    feats = dict(Fe); feats['h'] = np.full((W_, L_), 6, np.float32)
    feats['prior'], feats['prior_now'], feats['prior_alldays'] = prior_feats(P['prior_all'], M['dow'][d], t, 6)
    R = dict(d=d, t=t, cond=np.array(['queue_onset'] * W_), W=W_, L=L_, qlast=qlast,
             rows=[(6, feats, Qm[d, np.minimum(t + 6, 287)], Emv[d, np.minimum(t + 6, 287)])])
    X, y, e, meta, _ = to_matrix(R, 'queue_onset', cols, None)
    jd = np.array([dpos[M['days'][x]] for x in d]); cls = (M['dow'][d] >= 5).astype(int)
    Xk, a_mean = blend_prior(X, meta, P, jd, cls, t, cQ, cE, K_BLEND)
    Xm.append(Xk[e]); ym.append(y[e].astype(int)); dm.append(d[meta[:, 0]][e])
    log(panel, 'March onset training windows', len(picks), 'rows', int(e.sum()), 'pos', int(y[e].sum()), 'mean a', round(a_mean, 3))
    del P['march'], X, Xk, R, Fe; gc.collect()
Xm = np.concatenate(Xm); ym = np.concatenate(ym); dm = np.concatenate(dm)
log('March training rows', len(Xm), 'pos rate', round(float(ym.mean()), 4))
cat_pv = []
for wm in W_MARCH:
    m_ = CatBoostClassifier(iterations=600, depth=8, learning_rate=0.08, thread_count=4, verbose=0, random_seed=0)
    m_.fit(np.concatenate([_Xo_keep, Xm]), np.concatenate([_yo_keep, ym]),
           sample_weight=np.r_[np.ones(len(_Xo_keep)), np.full(len(Xm), float(wm))])
    cat_pv.append(m_); log(f'catboost onset + March (weight {wm}) trained')
# per-day models for the MARCH (validation) onset windows: train rows + March windows of the days < d only
vdays = set()
for panel in panels:
    wv_ = pd.read_csv(f'{REL}/task2/{panel}/validation/window_index.csv')
    wv_ = wv_[wv_.condition == 'queue_onset']
    vdays |= {pd.Timestamp(x).day - 1 for x in pd.to_datetime(wv_.forecast_origin, utc=True)}
cat_day = {}
for dd in sorted(vdays):
    msk = dm < dd
    if dd < 1 or not msk.any():
        continue
    m_ = CatBoostClassifier(iterations=600, depth=8, learning_rate=0.08, thread_count=4, verbose=0, random_seed=0)
    m_.fit(np.concatenate([_Xo_keep, Xm[msk]]), np.concatenate([_yo_keep, ym[msk]]),
           sample_weight=np.r_[np.ones(len(_Xo_keep)), np.full(int(msk.sum()), 20.0)])
    cat_day[dd] = m_
    log(f'per-day model for March day index {dd} (date 2031-03-{dd + 1:02d}): March rows {int(msk.sum())} (pos {int(ym[msk].sum())})')
log('validation onset window day indices', sorted(vdays), '| per-day models', sorted(cat_day))
del _Xo_keep, _yo_keep; gc.collect()

subs = []; stats = []
for panel, P in panels.items():
    dpos, cQ, cE = scen.pop(panel)
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
        cls = (dow >= 5).astype(int)
        Xk, a_mean = blend_prior(Xc, meta, P, jd, cls, tod, cQ, cE, K_BLEND)
        variants = {'base': 'base', 'sp4': 'sp4', 'pv': 'pv', 'pvm': 'pvm'}
        wday = np.array([pd.Timestamp(x).day - 1 for x in org.to_numpy()])
        preds = {}
        for name in variants:
            P6 = np.zeros((W, F, L), np.float32)
            if len(Xc):
                if name == 'base':
                    p_ = cat_on.predict_proba(Xc)[:, 1]
                elif name == 'pvm' and split == 'validation':        # per-day models: March windows of days < d only
                    p_ = cat_on.predict_proba(Xk)[:, 1]
                    rd = wday[wv]
                    for dd, m_ in cat_day.items():
                        mm = rd == dd
                        if mm.any():
                            p_[mm] = m_.predict_proba(Xk[mm])[:, 1]
                elif name == 'sp4' or split == 'validation':        # March windows never use the all-March models
                    p_ = cat_on.predict_proba(Xk)[:, 1]
                else:
                    p_ = np.mean([m_.predict_proba(Xk)[:, 1] for m_ in cat_pv], 0)
                P6[wv, meta[:, 2] - 1, lv] = p_
            preds[name] = np.stack([decode(P6[i].ravel()).reshape(F, L) if R['cond'][i] == 'queue_onset' else np.zeros((F, L), bool)
                                    for i in range(W)])
        ison = R['cond'] == 'queue_onset'
        st = dict(panel=panel, split=split, onset_windows=int(ison.sum()), mean_scen_days=float(jd[ison].mean()), mean_a4_prior=a_mean)
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
        tpl['condition'] = tpl.window_id.map(wi.set_index('window_id').condition); tpl['split'] = split
        subs.append(tpl[['window_id', 'timestamp', 'link_id', 'condition', 'split'] + [f'q_{n}' for n in variants]])
        del R, Xc; gc.collect()
    del cQ, cE; gc.collect()

S = pd.concat(subs, ignore_index=True)
log('queue rows', len(S), 'dupes', int(S.duplicated(['window_id', 'timestamp', 'link_id']).sum()))
print(pd.DataFrame(stats).round(4).to_string(index=False), flush=True)
for name in ('base', 'sp4', 'pv', 'pvm'):
    S[['window_id', 'timestamp', 'link_id', f'q_{name}']].rename(columns={f'q_{name}': 'queue_pred'}).to_csv(
        f'{OUT}/queue_submission_{name}.csv', index=False)
# pvb: March onset rows from the production model (base), April onset rows from the March-augmented bag
isval = (S.split == 'validation').to_numpy()
S['q_pvb'] = np.where(isval, S.q_base, S.q_pv)
S[['window_id', 'timestamp', 'link_id', 'q_pvb']].rename(columns={'q_pvb': 'queue_pred'}).to_csv(f'{OUT}/queue_submission_pvb.csv', index=False)
log('split check: validation rows', int(isval.sum()), 'private rows', int((~isval).sum()),
    '| pv == sp4 on validation rows:', bool((S.q_pv[isval] == S.q_sp4[isval]).all()))

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
    for name in ('base', 'sp4', 'pv', 'pvb', 'pvm'):
        a, b = m[f'q_{name}'].to_numpy().astype(bool), m.queue_pred.fillna(-1).to_numpy() == 1
        log(f'CONTROL chain-base onset vs {name}: agreement {float((a == b).mean()):.6f} jaccard {float((a & b).sum() / max((a | b).sum(), 1)):.5f}'
            f' changed share {float((a != b).mean()):.5f} rows {len(m)}')
except Exception as ex:
    log('CONTROL base skipped:', type(ex).__name__, str(ex)[:200])
log('PVM DONE')
