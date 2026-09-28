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
SMOKE = False
N_TREES, MAX_ROWS = (60, 150_000) if SMOKE else (700, 1_500_000)
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
                                        'speed_kmh', 'flow_vph', 'is_score_eligible'])
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
        self.mv = put(masked, 'speed_kmh'); self.mq = put(masked, 'flow_vph')
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
    # neighbour-anchored ratio transfer
    for ds in (-1, 1):
        nb = shift(nq, 0, ds)
        with np.errstate(all='ignore'):
            r = interp_axis(np.log(nq) - np.log(nb), 1)
        F[f'rq{ds:+d}'] = nb * np.exp(r)
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
    mv, mq = c.mv.copy(), c.mq.copy(); gapm = np.zeros(mv.shape, bool)
    for d, t in origins:
        mv[d, t + 1:t + 1 + GAP] = np.nan; mq[d, t + 1:t + 1 + GAP] = np.nan; gapm[d, t + 1:t + 1 + GAP] = True
    return mv, mq, gapm


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
params = dict(objective='regression', learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
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
    c.mv, c.mq, gapm = darken(c, origins)                     # train on the darkened view
    F, tv, tq = features(c, st, hv[c.dow], hq[c.dow])
    X, cols = stack(F, tmask)
    in_gap = gapm[tmask]
    log(panel, 'gap origins', len(origins), 'targets in simulated gaps', int(in_gap.sum()),
        'validation gap share of targets', round(gshare, 4))
    vf3 = np.broadcast_to(st.vf.to_numpy()[None, None, :], c.mv.shape)
    yv = (c.v / vf3)[tmask] - F['tv'][tmask]; yq = (c.q / lanes3)[tmask] - F['tq'][tmask]
    dayid = np.broadcast_to(np.arange(len(c.days))[:, None, None], c.mv.shape)[tmask]
    block = np.minimum(dayid * 3 // len(c.days), 2)                      # 3 contiguous blocks (frozen)
    pv_oof = np.zeros(len(X), np.float32); pq_oof = np.zeros(len(X), np.float32)
    rng = np.random.default_rng(0)
    for k in range(3):
        tr = np.where(block != k)[0]; te = np.where(block == k)[0]
        if len(tr) > MAX_ROWS:
            tr = rng.choice(tr, MAX_ROWS, replace=False)
        for y, out in ((yv, pv_oof), (yq, pq_oof)):
            m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), N_TREES)
            out[te] = m.predict(X[te])
    reg_t = np.broadcast_to(c.reg[:, None, None], c.mv.shape)[tmask]
    truth_v, truth_q, ln = c.v[tmask], c.q[tmask], lanes3[tmask]
    gv = (F['tv'][tmask] + pv_oof) * vf3[tmask]; gq = np.maximum((F['tq'][tmask] + pq_oof) * ln, 20)
    bv, bq = base_recipe(c, st, hv[c.dow], hq[c.dow])
    row = dict(panel=panel, family=FAMILY(panel), n_targets=int(tmask.sum()), stations=len(stations), gap_share=gshare)
    for name, pv_, pq_ in (('temporal', tv[tmask], tq[tmask]), ('base', bv[tmask], bq[tmask]), ('lgbm', gv, gq)):
        s, sr = s_state(pv_, pq_, truth_v, truth_q, ln, reg_t)
        PV = np.where(tmask, 0, np.nan).astype(np.float32); PQ = PV.copy()
        PV[tmask] = pv_; PQ[tmask] = pq_
        row[f'{name}_state'] = s; row[f'{name}_R3'] = sr.get('R3', np.nan)
        row[f'{name}_lwr'] = lwr_proxy(c, st, tmask, PV, PQ)
        for k in range(3):
            mk = block == k
            row[f'{name}_state_b{k}'] = s_state(pv_[mk], pq_[mk], truth_v[mk], truth_q[mk], ln[mk], reg_t[mk])[0]
        # regular vs gap cells, and the validation-like mixture (MSE weighted by the split's gap share)
        for part, mk in (('reg', ~in_gap), ('gap', in_gap)):
            if mk.sum():
                row[f'{name}_rmse_v_{part}'] = float(np.sqrt(np.mean((pv_[mk] - truth_v[mk]) ** 2)))
                row[f'{name}_rmse_q_{part}'] = float(np.sqrt(np.mean(((pq_[mk] - truth_q[mk]) / ln[mk]) ** 2)))
        if in_gap.sum():
            mse_v = (1 - gshare) * row[f'{name}_rmse_v_reg'] ** 2 + gshare * row[f'{name}_rmse_v_gap'] ** 2
            mse_q = (1 - gshare) * row[f'{name}_rmse_q_reg'] ** 2 + gshare * row[f'{name}_rmse_q_gap'] ** 2
            row[f'{name}_state_mix'] = 0.54 * max(0, 1 - np.sqrt(mse_v) / 25) + 0.46 * max(0, 1 - np.sqrt(mse_q) / 600)
        else:
            row[f'{name}_state_mix'] = row[f'{name}_state']
    results.append(row)
    log(panel, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    # ---------------- final fit on all train targets, predict validation + private
    idx = np.arange(len(X))
    if len(idx) > MAX_ROWS:
        idx = rng.choice(idx, MAX_ROWS, replace=False)
    mv_ = lgb.train(params, lgb.Dataset(X[idx], yv[idx]), N_TREES)
    mq_ = lgb.train(params, lgb.Dataset(X[idx], yq[idx]), N_TREES)
    imp = pd.Series(mq_.feature_importance('gain'), index=cols).sort_values(ascending=False)
    log(panel, 'top flow features', imp.head(8).round(0).to_dict())
    del X, F, c; gc.collect()
    for split in ('validation', 'private'):
        sm = read_split(panel, split, 'mainline_states_masked')
        cs = Cube(sm, stations); del sm
        tm, tpl, (di, ti, si, ok) = template_mask(panel, split, cs, stations)
        Fs, tvs, tqs = features(cs, st, hv[cs.dow], hq[cs.dow])
        Xs = np.column_stack([np.asarray(Fs[k])[di[ok], ti[ok], si[ok]] for k in cols]).astype(np.float32)
        vfs = st.vf.to_numpy()[si[ok]]; lns = st.lanes.to_numpy()[si[ok]]
        pvs = (Fs['tv'][di[ok], ti[ok], si[ok]] + mv_.predict(Xs)) * vfs
        pqs = np.maximum((Fs['tq'][di[ok], ti[ok], si[ok]] + mq_.predict(Xs)) * lns, 20)
        out = tpl.loc[ok, ['timestamp', 'station_id', 'link_id', 'mask_regime']].copy()
        out.insert(0, 'panel', panel)
        out['speed_kmh'] = np.clip(pvs, 3, None); out['flow_vph'] = pqs
        miss = int((~ok).sum())
        log(panel, split, 'template rows', len(tpl), 'predicted', len(out), 'unmapped', miss,
            'speed range', np.round([out.speed_kmh.min(), out.speed_kmh.max()], 1).tolist(),
            'flow<50 share', round(float((out.flow_vph < 50).mean()), 4))
        subs.append(out)
        del Fs, Xs, cs; gc.collect()
    log(panel, f'panel time {time.time() - tp:.0f}s')

R = pd.DataFrame(results)
R.to_csv(f'{OUT}/t1_results.csv', index=False)
print('\n===== Task 1 train CV (3 contiguous blocks), competition aggregation =====')
for name in ('temporal', 'base', 'lgbm'):
    fam = R.groupby('family')[f'{name}_state'].mean().mean()
    famL = R.groupby('family')[f'{name}_lwr'].mean().mean()
    blocks = [R.groupby('family')[f'{name}_state_b{k}'].mean().mean() for k in range(3)]
    famM = R.groupby('family')[f'{name}_state_mix'].mean().mean()
    print(f'{name:9s} S_state={fam:.4f}  S_state_valmix={famM:.4f}  blocks={np.round(blocks, 4).tolist()}  std={np.std(blocks):.4f}  LWR_proxy={famL:.4f}')
print(R[[c for c in R.columns if 'rmse' in c or c in ('panel', 'gap_share')]].round(3).to_string())
S = pd.concat(subs, ignore_index=True)
S.to_csv(f'{OUT}/state_submission.csv', index=False)
log('state rows', len(S), 'NaN', int(S[['speed_kmh', 'flow_vph']].isna().sum().sum()))
