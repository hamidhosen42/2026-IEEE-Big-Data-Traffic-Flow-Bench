"""Task 1 (+ Task 3 proxy) pipeline on Kaggle.

Validation: train split, targets from the official task1 train template, 3 contiguous day blocks (frozen).
Methods scored on identical cells:
  temporal   per-station linear interpolation in time
  base       the base notebook's recipe (temporal 85/15 spatial, density roll, soft FD, conservation smoothing)
  lgbm       LightGBM on neighbour/interp/profile features, residual to temporal interpolation
Final: LightGBM refit on all train targets, applied to the validation and private templates.
"""
import glob, json, os, re, time, gc
import numpy as np
import pandas as pd
import lightgbm as lgb

T0 = time.time()
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
OUT = '/kaggle/working'
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S',
          'D12_I5_N', 'D12_I5_S', 'D12_I405_N', 'D12_I405_S']
SMOKE = True
N_TREES, MAX_ROWS = (60, 150_000) if SMOKE else (300, 600_000)
if SMOKE:
    PANELS = ['D12_I405_S', 'D7_I10_W']
FAMILY = lambda p: p.rsplit('_', 1)[0]
RATE = {'R1': .2, 'R2': .3, 'R3': .5}


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


# ------------------------------------------------------------------ loading
def read_split(panel, split, layer):
    fs = sorted(glob.glob(f'{REL}/corridors/{panel}/{split}/{layer}/*/*.parquet'))
    if SMOKE:   # same calendar days in every layer: every 9th day
        keep = sorted({os.path.basename(f)[-18:-8] for f in fs})[::9]
        fs = [f for f in fs if os.path.basename(f)[-18:-8] in keep]
    parts = []
    for f in fs:
        d = pd.read_parquet(f, columns=['date', 'timestamp', 'station_id', 'link_id', 'milepost',
                                        'speed_kmh', 'flow_vph', 'occupancy', 'is_score_eligible'])
        m = re.search(r'mask_regime=(R\d)', f)
        d['regime'] = m.group(1) if m else ''
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


class Cube:
    """(day, 288, station) arrays for one panel/split; stations in milepost order."""

    def __init__(self, masked, stations, full=None):
        self.days = sorted(masked.date.unique())
        self.reg = masked.drop_duplicates('date').set_index('date').regime.reindex(self.days).to_numpy()
        self.S = len(stations)
        sidx = pd.Index(stations)
        shape = (len(self.days), 288, self.S)

        def put(df, col, fill=np.nan, dtype=np.float32):
            a = np.full(shape, fill, dtype=dtype)
            ts = pd.to_datetime(df.timestamp, utc=True)
            di = pd.Index(self.days).get_indexer(df.date)
            ok = di >= 0
            ti = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
            si = sidx.get_indexer(df.station_id)
            ok &= si >= 0
            a[di[ok], ti[ok], si[ok]] = df[col].to_numpy()[ok]
            return a
        self.mv = put(masked, 'speed_kmh'); self.mq = put(masked, 'flow_vph'); self.mo = put(masked, 'occupancy')
        self.melig = put(masked, 'is_score_eligible', 0, np.int8).astype(bool)
        self.dow = pd.to_datetime(pd.Series(self.days)).dt.dayofweek.to_numpy()
        if full is not None:
            self.v = put(full, 'speed_kmh'); self.q = put(full, 'flow_vph')
            self.elig = put(full, 'is_score_eligible', 0, np.int8).astype(bool) & np.isfinite(self.v)
        self.put = put


def template_mask(panel, split, cube, stations):
    t = pd.read_csv(f'{REL}/task1/{panel}/{split}/sample_submission_state.csv', dtype=str)
    t['date'] = t.timestamp.str[:10]
    ts = pd.to_datetime(t.timestamp, utc=True)
    di = pd.Index(cube.days).get_indexer(t.date)
    si = pd.Index(stations).get_indexer(t.station_id)
    ti = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
    ok = (di >= 0) & (si >= 0)
    m = np.zeros((len(cube.days), 288, len(stations)), bool)
    m[di[ok], ti[ok], si[ok]] = True
    return m, t, (di, ti, si, ok)


# ------------------------------------------------------------------ features
def interp_axis(a, axis, limit=None):
    b = np.moveaxis(a, axis, -1)
    sh = b.shape
    flat = b.reshape(-1, sh[-1]).astype(np.float64)
    out = flat.copy()
    x = np.arange(sh[-1])
    for i in range(flat.shape[0]):
        y = flat[i]; ok = np.isfinite(y)
        if ok.sum() == 0 or ok.all():
            continue
        f = np.interp(x, x[ok], y[ok])
        if limit is not None:
            idx = np.where(ok)[0]; pos = np.searchsorted(idx, x)
            dl = x - idx[np.clip(pos - 1, 0, len(idx) - 1)]; dr = idx[np.clip(pos, 0, len(idx) - 1)] - x
            f = np.where(np.minimum(np.abs(dl), np.abs(dr)) <= limit, f, np.nan)
        out[i] = np.where(ok, y, f)
    return np.moveaxis(out.reshape(sh), -1, axis).astype(np.float32)


def shift(a, dt=0, ds=0):
    """value at (t+dt, s+ds); NaN outside."""
    out = np.full_like(a, np.nan)
    T, S = a.shape[1], a.shape[2]
    ts_src = slice(max(dt, 0), T + min(dt, 0)); ts_dst = slice(max(-dt, 0), T + min(-dt, 0))
    ss_src = slice(max(ds, 0), S + min(ds, 0)); ss_dst = slice(max(-ds, 0), S + min(-ds, 0))
    out[:, ts_dst, ss_dst] = a[:, ts_src, ss_src]
    return out


def gap_dist(obs):
    T = obs.shape[1]; fin = np.isfinite(obs)
    t = np.arange(T)[None, :, None]
    prev = np.maximum.accumulate(np.where(fin, t, -10 ** 6), axis=1)
    nxt = np.minimum.accumulate(np.where(fin, t, 10 ** 6)[:, ::-1], axis=1)[:, ::-1]
    return np.minimum(t - prev, 99).astype(np.float32), np.minimum(nxt - t, 99).astype(np.float32)


def base_recipe(c, st, hv, hq):
    """The base notebook's Task 1 recipe, re-implemented on the station cube (milepost order)."""
    lanes = st.lanes.to_numpy()[None, None, :]; vf = st.vf.to_numpy()[None, None, :]
    cap = st.cap.to_numpy()[None, None, :]; kj = st.k_jam.to_numpy()[None, None, :]
    t12v = interp_axis(c.mv, 1, 12); t12q = interp_axis(c.mq / lanes, 1, 12)
    s8v = interp_axis(t12v, 2, 8); s8q = interp_axis(t12q, 2, 8)
    pv = np.where(np.isfinite(t12v), 0.85 * t12v + 0.15 * np.where(np.isfinite(s8v), s8v, t12v),
                  np.where(np.isfinite(s8v), 0.9 * s8v + 0.1 * hv, hv))
    pq = np.where(np.isfinite(t12q), 0.85 * t12q + 0.15 * np.where(np.isfinite(s8q), s8q, t12q),
                  np.where(np.isfinite(s8q), 0.9 * s8q + 0.1 * hq / lanes, hq / lanes)) * lanes
    pv = np.clip(pv, 5, vf * 1.05); pq = np.maximum(pq, 50)
    k = pq / np.maximum(pv, 1)
    kp = np.pad(k, ((0, 0), (1, 1), (0, 0)), mode='edge')
    pq = np.clip(0.85 * pq + 0.15 * (kp[:, :-2] + kp[:, 1:-1] + kp[:, 2:]) / 3 * pv, 50, cap * 1.15)
    kc = cap / vf; w = cap / np.maximum(kj - kc, 1); k = pq / np.maximum(pv, 1)
    qfd = np.where(k <= kc, vf * k, w * np.maximum(kj - k, 0))
    pq = np.where((qfd > 0) & (qfd < cap * 1.3), 0.85 * pq + 0.15 * qfd, pq)
    for _ in range(5):
        for j in range(pq.shape[2] - 1):
            cc = 0.2 * (pq[:, :, j + 1] - pq[:, :, j]); pq[:, :, j] += cc; pq[:, :, j + 1] -= cc
    return np.clip(pv, 5, vf * 1.05), np.clip(pq, 50, cap * 1.15)


def lowrank(a, rank=3):
    """per-day rank-k SVD reconstruction of a (day, 288, S) matrix (NaN filled with column means)."""
    out = np.full_like(a, np.nan)
    for d in range(a.shape[0]):
        m = a[d].astype(np.float64)
        if not np.isfinite(m).any():
            continue
        cm = np.nanmean(m, 0); cm = np.where(np.isfinite(cm), cm, np.nanmean(m))
        m = np.where(np.isfinite(m), m, cm[None, :])
        mu = m.mean(0, keepdims=True); U, sv, Vt = np.linalg.svd(m - mu, full_matrices=False)
        out[d] = (U[:, :rank] * sv[:rank]) @ Vt[:rank] + mu
    return out.astype(np.float32)


def lowrank_days(a, rank=3):
    """per-station rank-k SVD across the split's days: (day x 288) matrix per station -> the month's own daily pattern."""
    out = np.full_like(a, np.nan)
    for s_ in range(a.shape[2]):
        m = a[:, :, s_].astype(np.float64)
        if not np.isfinite(m).any() or m.shape[0] < 3:
            continue
        cm = np.nanmean(m, 0); cm = np.where(np.isfinite(cm), cm, np.nanmean(m))
        m = np.where(np.isfinite(m), m, cm[None, :])
        mu = m.mean(0, keepdims=True); U, sv, Vt = np.linalg.svd(m - mu, full_matrices=False)
        out[:, :, s_] = (U[:, :rank] * sv[:rank]) @ Vt[:rank] + mu
    return out.astype(np.float32)


NEW_LR = ('lr5_v', 'lr5_q', 'lr_o', 'lrd_v', 'lrd_q')


def features(c, st, hv, hq):
    lanes = st.lanes.to_numpy()[None, None, :]; vf = st.vf.to_numpy()[None, None, :]
    shape = c.mv.shape
    tv = interp_axis(c.mv, 1); tq = interp_axis(c.mq, 1)
    tv = np.where(np.isfinite(tv), tv, hv); tq = np.where(np.isfinite(tq), tq, hq)
    sv = interp_axis(c.mv, 2); sq = interp_axis(c.mq, 2)
    sv = np.where(np.isfinite(sv), sv, tv); sq = np.where(np.isfinite(sq), sq, tq)
    gp, gn = gap_dist(c.mv)
    nv = c.mv / vf; nq = c.mq / lanes
    F = {'tv': tv / vf, 'tq': tq / lanes, 'sv': sv / vf, 'sq': sq / lanes, 'hv': hv / vf, 'hq': hq / lanes,
         'tk': tq / np.maximum(tv, 1) / lanes, 'gp': gp, 'gn': gn,
         'tod': np.broadcast_to(np.arange(288, dtype=np.float32)[None, :, None], shape),
         'dow': np.broadcast_to(c.dow.astype(np.float32)[:, None, None], shape),
         'lanes': np.broadcast_to(lanes.astype(np.float32), shape), 'vf': np.broadcast_to(vf.astype(np.float32), shape),
         'pos': np.broadcast_to(np.linspace(0, 1, shape[2], dtype=np.float32)[None, None, :], shape),
         'rate': np.broadcast_to(np.array([RATE.get(r, .3) for r in c.reg], np.float32)[:, None, None], shape)}
    for ds in (-2, -1, 1, 2):
        for dt in (-1, 0, 1):
            F[f'v{ds:+d}{dt:+d}'] = shift(nv, dt, ds); F[f'q{ds:+d}{dt:+d}'] = shift(nq, dt, ds)
    for dt in (-2, -1, 1, 2):
        F[f'v0{dt:+d}'] = shift(nv, dt, 0); F[f'q0{dt:+d}'] = shift(nq, dt, 0)
    for ds in (-1, 0, 1):
        for dt in (-1, 0, 1):
            if (ds, dt) != (0, 0):
                F[f'o{ds:+d}{dt:+d}'] = shift(c.mo, dt, ds)
    F['to'] = interp_axis(c.mo, 1); F['so'] = interp_axis(c.mo, 2)
    # neighbour-anchored ratio transfer
    for ds in (-1, 1):
        nb = shift(nq, 0, ds)
        with np.errstate(all='ignore'):
            r = interp_axis(np.log(nq) - np.log(nb), 1)
        F[f'rq{ds:+d}'] = nb * np.exp(r)
    F['lr_v'] = lowrank(F['tv']); F['lr_q'] = lowrank(F['tq'])
    F['lr5_v'] = lowrank(F['tv'], 5); F['lr5_q'] = lowrank(F['tq'], 5); F['lr_o'] = lowrank(F['to'])
    F['lrd_v'] = lowrank_days(F['tv']); F['lrd_q'] = lowrank_days(F['tq'])
    return F, tv, tq


def stack(F, mask):
    return np.column_stack([np.asarray(F[k])[mask] for k in F]).astype(np.float32), list(F)


# ------------------------------------------------------------------ scoring
def s_state(v, q, tv_, tq_, lanes, reg):
    out = {}
    for r in ('R1', 'R2', 'R3'):
        m = reg == r
        if m.sum() == 0:
            continue
        rv = np.sqrt(np.mean((v[m] - tv_[m]) ** 2)); rq = np.sqrt(np.mean(((q[m] - tq_[m]) / lanes[m]) ** 2))
        out[r] = 0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600)
    return float(np.mean(list(out.values()))), out


def lwr_proxy(c, st, tmask, pv, pq):
    L = st.length_km.to_numpy()[None, None, :]
    vv = np.where(tmask, pv, c.v); qq = np.where(tmask, pq, c.q)
    N = qq / np.clip(vv, 1, None) * L; Nt = c.q / np.clip(c.v, 1, None) * L
    ok = c.elig[:, :-1] & c.elig[:, 1:]
    dN = N[:, 1:] - N[:, :-1]; dNt = Nt[:, 1:] - Nt[:, :-1]
    out = {}
    for r in ('R1', 'R2', 'R3'):
        m = ok & (c.reg == r)[:, None, None]
        if m.sum():
            out[r] = max(0, 1 - np.nansum(np.abs(dN - dNt)[m]) / np.nansum(np.abs(dNt)[m]))
    return float(np.mean(list(out.values())))




def predict_variants(Xm, tvn, tqn, vf_, ln_, reg_, MA, MB, MC, ixA, ixB, ixC):
    """returns {variant: (speed, flow)} for rows of Xm; tvn/tqn are the normalised interpolation bases."""
    out = {}
    for name, (mv, mq), ix in (('A', MA, ixA), ('B', MB, ixB)):
        out[name] = (np.clip((tvn + mv.predict(Xm[:, ix])) * vf_, 3, 1.05 * vf_), np.maximum((tqn + mq.predict(Xm[:, ix])) * ln_, 20))
    for nm, (mv, mq) in (('D', MC), ('E', ME_G[0]), ('F', ME_G[-1])):
        out[nm] = (np.clip((tvn + mv.predict(Xm[:, ixC])) * vf_, 3, 1.05 * vf_), np.maximum((tqn + mq.predict(Xm[:, ixC])) * ln_, 20))
    return out

ME_G = [None]
# ------------------------------------------------------------------ main
GAP = 18


def gap_origins(panel, days):
    fs = glob.glob(f'/kaggle/input/**/pseudo_{panel}.npz', recursive=True)
    if not fs:
        return []
    z = np.load(fs[0], allow_pickle=True)
    all_days = sorted(os.path.basename(f)[-18:-8].replace('_', '-') for f in
                      glob.glob(f'{REL}/corridors/{panel}/train/mainline_states/*/*.parquet'))
    pos = {dd: i for i, dd in enumerate(days)}
    return [(pos[all_days[d]], int(t)) for d, t in zip(z['d'], z['t']) if all_days[d] in pos]


def darken(c, origins):
    mv, mq, mo = c.mv.copy(), c.mq.copy(), c.mo.copy(); gapm = np.zeros(mv.shape, bool)
    for d, t in origins:
        mv[d, t + 1:t + 1 + GAP] = np.nan; mq[d, t + 1:t + 1 + GAP] = np.nan; mo[d, t + 1:t + 1 + GAP] = np.nan
        gapm[d, t + 1:t + 1 + GAP] = True
    return mv, mq, mo, gapm


def split_gap_share(panel, split, stations):
    wi_p = f'{REL}/task2/{panel}/{split}/window_index.csv'
    tpl = pd.read_csv(f'{REL}/task1/{panel}/{split}/sample_submission_state.csv', usecols=['timestamp'])
    if not os.path.exists(wi_p):
        return 0.0
    wi = pd.read_csv(wi_p); ts = pd.to_datetime(tpl.timestamp, utc=True); ing = np.zeros(len(ts), bool)
    for o in pd.to_datetime(wi.forecast_origin, utc=True):
        ing |= (ts > o) & (ts <= o + pd.Timedelta(minutes=5 * GAP))
    return float(ing.mean())


results = []
subs = []
val_rows = []
params = dict(objective='regression', learning_rate=0.1, num_leaves=127, min_data_in_leaf=200, max_bin=63,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4)
for panel in PANELS:
    tp = time.time()
    fdp = pd.read_csv(f'{REL}/corridors/{panel}/network/fd_parameters.csv', dtype={'link_id': str}).set_index('link_id')
    tr_full = read_split(panel, 'train', 'mainline_states')
    tr_mask = read_split(panel, 'train', 'mainline_states_masked')
    st = tr_full.drop_duplicates('station_id').set_index('station_id')[['link_id', 'milepost']].sort_values('milepost')
    st = st.join(fdp.rename(columns={'free_speed_kmh': 'vf', 'capacity_vph': 'cap'})[['lanes', 'length_km', 'vf', 'cap', 'critical_density', 'k_jam']], on='link_id')
    stations = st.index.tolist()
    c = Cube(tr_mask, stations, full=tr_full); del tr_full, tr_mask; gc.collect()
    lanes3 = np.broadcast_to(st.lanes.to_numpy()[None, None, :], c.mv.shape)
    # profile from released (masked-layer) train observations: weekday x slot x station
    hv = np.full((7, 288, len(stations)), np.nan, np.float32); hq = hv.copy()
    for w in range(7):
        sel = c.dow == w
        with np.errstate(all='ignore'):
            hv[w] = np.nanmean(c.mv[sel], 0); hq[w] = np.nanmean(c.mq[sel], 0)
    hv = np.where(np.isfinite(hv), hv, np.nanmean(hv)); hq = np.where(np.isfinite(hq), hq, np.nanmean(hq))
    tmask, _, _ = template_mask(panel, 'train', c, stations)
    tmask &= c.elig
    assert tmask.sum() > 0, 'no Task 1 targets mapped'
    origins = gap_origins(panel, c.days)
    gshare = split_gap_share(panel, 'validation', stations)
    mv0, mq0 = c.mv, c.mq
    c.mv, c.mq, c.mo, gapm = darken(c, origins)                     # train on the darkened view
    F, tv, tq = features(c, st, hv[c.dow], hq[c.dow])
    X, cols = stack(F, tmask)
    in_gap = gapm[tmask]
    log(panel, 'gap origins', len(origins), 'targets in simulated gaps', int(in_gap.sum()),
        'validation gap share of targets', round(gshare, 4))
    vf3 = np.broadcast_to(st.vf.to_numpy()[None, None, :], c.mv.shape)
    yv = (c.v / vf3)[tmask] - F['tv'][tmask]; yq = (c.q / lanes3)[tmask] - F['tq'][tmask]
    dayid = np.broadcast_to(np.arange(len(c.days))[:, None, None], c.mv.shape)[tmask]
    block = np.minimum(dayid * 3 // len(c.days), 2)                      # 3 contiguous blocks (frozen)
    reg_t = np.broadcast_to(c.reg[:, None, None], c.mv.shape)[tmask]
    rng = np.random.default_rng(0)
    # ---------------- final fit on all train targets, predict validation + private
    idx = np.arange(len(X))
    if len(idx) > MAX_ROWS:
        idx = rng.choice(idx, MAX_ROWS, replace=False)
    NOPROF = {'hv', 'hq', 'tod', 'dow'}
    ixA = [cols.index(k) for k in cols if k not in NOPROF and not k.startswith('lr')]
    ixB = [cols.index(k) for k in cols if k not in NOPROF and k not in NEW_LR]          # = current best (T1 v5 B)
    ixD = [cols.index(k) for k in cols if k not in NOPROF]
    fit = lambda ix, rows: (lgb.train(params, lgb.Dataset(X[rows][:, ix], yv[rows]), N_TREES),
                            lgb.train(params, lgb.Dataset(X[rows][:, ix], yq[rows]), N_TREES))
    MA = fit(ixA, idx); MB = fit(ixB, idx)
    MC = fit(ixD, idx)
    del ME_G[1:]
    idx2 = np.arange(len(X))
    if len(idx2) > 2 * MAX_ROWS:
        idx2 = rng.choice(idx2, 2 * MAX_ROWS, replace=False)
    pbig = dict(params, num_leaves=255, min_data_in_leaf=100, learning_rate=0.05)
    ME = (lgb.train(pbig, lgb.Dataset(X[idx2][:, ixD], yv[idx2]), 60 if SMOKE else 900),
          lgb.train(pbig, lgb.Dataset(X[idx2][:, ixD], yq[idx2]), 60 if SMOKE else 900))
    ME_G[0] = ME
    idx4 = np.arange(len(X))
    if len(idx4) > 4 * MAX_ROWS:
        idx4 = rng.choice(idx4, 4 * MAX_ROWS, replace=False)
    pF = dict(params, num_leaves=511, min_data_in_leaf=100, learning_rate=0.04)
    ME_G.append((lgb.train(pF, lgb.Dataset(X[idx4][:, ixD], yv[idx4]), 60 if SMOKE else 1500),
                 lgb.train(pF, lgb.Dataset(X[idx4][:, ixD], yq[idx4]), 60 if SMOKE else 1500)))
    imp = pd.Series(MC[1].feature_importance('gain'), index=[cols[i] for i in ixD]).sort_values(ascending=False)
    log(panel, 'top flow features', imp.head(8).round(0).to_dict())
    del X, F, c; gc.collect()
    for split in ('validation', 'private'):
        sm = read_split(panel, split, 'mainline_states_masked')
        cs = Cube(sm, stations); del sm
        if split == 'validation':
            # internal holdout on the validation month itself: hide 15% of released eligible cells
            rng2 = np.random.default_rng(7)
            hold = cs.melig & np.isfinite(cs.mv) & np.isfinite(cs.mq) & (rng2.random(cs.mv.shape) < 0.15)
            hv_true, hq_true = cs.mv[hold].copy(), cs.mq[hold].copy()
            ch = Cube.__new__(Cube); ch.__dict__.update(cs.__dict__)
            ch.mv = np.where(hold, np.nan, cs.mv); ch.mq = np.where(hold, np.nan, cs.mq); ch.mo = np.where(hold, np.nan, cs.mo)
            Fh, tvh, tqh = features(ch, st, hv[ch.dow], hq[ch.dow])
            Xh = np.column_stack([np.asarray(Fh[k])[hold] for k in cols]).astype(np.float32)
            vfh = np.broadcast_to(st.vf.to_numpy()[None, None, :], ch.mv.shape)[hold]
            lnh = np.broadcast_to(st.lanes.to_numpy()[None, None, :], ch.mv.shape)[hold]
            regh = np.broadcast_to(ch.reg[:, None, None], ch.mv.shape)[hold]
            bvh, bqh = base_recipe(ch, st, hv[ch.dow], hq[ch.dow])
            cand = {'temporal': (tvh[hold], tqh[hold]), 'base': (bvh[hold], bqh[hold])}
            cand.update(predict_variants(Xh, Fh['tv'][hold], Fh['tq'][hold], vfh, lnh, regh, MA, MB, MC, ixA, ixB, ixD))
            vh = {k: round(s_state(a, b, hv_true, hq_true, lnh, regh)[0], 4) for k, (a, b) in cand.items()}
            row_v = dict(panel=panel, family=FAMILY(panel), **{f'val_{k}': v for k, v in vh.items()})
            val_rows.append(row_v); log(panel, 'VALIDATION-month internal holdout S_state', vh)
            del Fh, Xh, ch
        tm, tpl, (di, ti, si, ok) = template_mask(panel, split, cs, stations)
        Fs, tvs, tqs = features(cs, st, hv[cs.dow], hq[cs.dow])
        Xs = np.column_stack([np.asarray(Fs[k])[di[ok], ti[ok], si[ok]] for k in cols]).astype(np.float32)
        vfs = st.vf.to_numpy()[si[ok]]; lns = st.lanes.to_numpy()[si[ok]]
        regs = tpl.mask_regime.to_numpy()[ok]
        V3 = predict_variants(Xs, Fs['tv'][di[ok], ti[ok], si[ok]], Fs['tq'][di[ok], ti[ok], si[ok]], vfs, lns, regs, MA, MB, MC, ixA, ixB, ixD)
        out = tpl.loc[ok, ['timestamp', 'station_id', 'link_id', 'mask_regime']].copy()
        out.insert(0, 'panel', panel)
        for vname, (pvs, pqs) in V3.items():
            out[f'speed_{vname}'] = pvs; out[f'flow_{vname}'] = pqs
        out['speed_kmh'] = out.speed_B; out['flow_vph'] = out.flow_B
        miss = int((~ok).sum())
        log(panel, split, 'template rows', len(tpl), 'predicted', len(out), 'unmapped', miss,
            'speed range', np.round([out.speed_kmh.min(), out.speed_kmh.max()], 1).tolist(),
            'flow<50 share', round(float((out.flow_vph < 50).mean()), 4))
        subs.append(out)
        del Fs, Xs, cs; gc.collect()
    log(panel, f'panel time {time.time() - tp:.0f}s')

V = pd.DataFrame(val_rows); V.to_csv(f'{OUT}/t1_valholdout.csv', index=False)
print('\n===== validation-month internal holdout (family-mean S_state) =====')
print(V.groupby('family')[[c for c in V.columns if c.startswith('val_')]].mean().mean().round(4).to_string())
S = pd.concat(subs, ignore_index=True)
base_cols = ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']
for vname in ('E', 'F'):
    S[base_cols].assign(speed_kmh=S[f'speed_{vname}'], flow_vph=S[f'flow_{vname}']).to_csv(f'{OUT}/state_submission_{vname}.csv', index=False)
S[base_cols + ['speed_kmh', 'flow_vph']].to_csv(f'{OUT}/state_submission.csv', index=False)
log('state rows', len(S), 'NaN', int(S[['speed_kmh', 'flow_vph']].isna().sum().sum()))
