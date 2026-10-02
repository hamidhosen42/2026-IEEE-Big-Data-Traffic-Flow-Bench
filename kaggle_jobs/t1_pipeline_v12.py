"""Task 1 v12 (bagging member): identical to v11 FTA except seed 1 everywhere it is free: train-row subsample rng(1),
transductive hidden-cell rounds rng(2000+...), LightGBM seed 1. Holdout (rng 7) unchanged. Averaged with v11 FTA in the swap kernel.
Only FTA is trained (FT dropped) to save time.

v11 notes: Task 1 v11: transductive rows (v9, LB +0.0017) + ASM features (v10, holdout +0.0005). Compared on the val-month holdout:
FT (transductive, no ASM; = v9 FT) vs FTA (transductive + ASM). Both fitted on train rows + the same transductive rows.

v10 notes: ASM/GASM characteristic-kernel features (research C A1): mask-normalised kernel averages of the
visible speed / per-lane flow along 4 wave speeds (+-80 km/h free-flow, +-18 km/h congested; both signs because the
travel direction vs milepost order is not assumed), +-5 slots x +-5 stations with real milepost distance.
Compared on the val-month holdout: F (v8 features) vs FA (+ ASM features). No transductive rows here (TR_ROUNDS = 0).

v9 notes: transductive rows. Task 1 is offline, so released (observed) cells of the validation and private months
are hidden in random rounds and used as extra training rows (features from the hidden view, target = released value).
The validation-month internal holdout (15%, seed 7) is hidden in every view and never used as a target.
Compared on that holdout: F (train rows only, = v8 F) vs FT (F + transductive rows).

Task 1 (+ Task 3 proxy) pipeline on Kaggle.

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
N_TREES, MAX_ROWS = (60, 150_000) if SMOKE else (300, 600_000)
if SMOKE:
    PANELS = ['D12_I405_S', 'D7_I10_W']
FAMILY = lambda p: p.rsplit('_', 1)[0]
RATE = {'R1': .2, 'R2': .3, 'R3': .5}
TR_ROUNDS, TR_HIDE = 2, 0.20
ASM_WAVES, ASM_K, ASM_SIGMA, ASM_TAU = (80.0, -80.0, 18.0, -18.0), 5, 0.6, 5.0


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
    F.update(asm_feats(nv, nq, st.milepost.to_numpy()))
    return F, tv, tq


def asm_feats(nv, nq, x):
    """ASM-style kernel smoothing along characteristics. x: milepost (km) per station, in cube station order."""
    D, T, S = nv.shape
    out = {}
    num = {(c, ch): np.zeros(nv.shape, np.float32) for c in ASM_WAVES for ch in 'vq'}
    den = {c: np.zeros(nv.shape, np.float32) for c in ASM_WAVES}
    xi = np.asarray(x, np.float64)
    for ds in range(-ASM_K, ASM_K + 1):
        dx = np.full(S, np.nan)
        if ds >= 0:
            dx[:S - ds] = xi[ds:] - xi[:S - ds]
        else:
            dx[-ds:] = xi[:S + ds] - xi[-ds:]
        for dt in range(-ASM_K, ASM_K + 1):
            if ds == 0 and dt == 0:
                continue
            sv = shift(nv, dt, ds); sq = shift(nq, dt, ds)
            ok = np.isfinite(sv) & np.isfinite(sq)
            sv = np.where(ok, sv, 0).astype(np.float32); sq = np.where(ok, sq, 0).astype(np.float32); okf = ok.astype(np.float32)
            for c in ASM_WAVES:
                w = np.exp(-np.abs(dx) / ASM_SIGMA - np.abs(dt * 5.0 - dx / c * 60.0) / ASM_TAU)
                w = np.nan_to_num(w, nan=0.0).astype(np.float32)[None, None, :]
                num[c, 'v'] += sv * w; num[c, 'q'] += sq * w; den[c] += okf * w
    for c in ASM_WAVES:
        with np.errstate(all='ignore'):
            out[f'asm_v{c:+.0f}'] = np.where(den[c] > 1e-6, num[c, 'v'] / den[c], np.nan).astype(np.float32)
            out[f'asm_q{c:+.0f}'] = np.where(den[c] > 1e-6, num[c, 'q'] / den[c], np.nan).astype(np.float32)
        out[f'asm_w{c:+.0f}'] = den[c]
    return out


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




def predict_variants(Xm, tvn, tqn, vf_, ln_, reg_, MODELS, ixC):
    """returns {variant: (speed, flow)} for rows of Xm (all cols); each model carries its own column index."""
    out = {}
    for nm, (mv, mq, ix) in MODELS.items():
        out[nm] = (np.clip((tvn + mv.predict(Xm[:, ix])) * vf_, 3, 1.05 * vf_), np.maximum((tqn + mq.predict(Xm[:, ix])) * ln_, 20))
    return out

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
    rng = np.random.default_rng(1)
    NOPROF = {'hv', 'hq', 'tod', 'dow'}
    ixD = [cols.index(k) for k in cols if k not in NOPROF]
    ixF = [i for i in ixD if not cols[i].startswith('asm')]; posF = [ixD.index(i) for i in ixF]
    idx4 = np.arange(len(X))
    if len(idx4) > 4 * MAX_ROWS:
        idx4 = rng.choice(idx4, 4 * MAX_ROWS, replace=False)
    Xtr, yvtr, yqtr = X[idx4][:, ixD], yv[idx4], yq[idx4]
    del X, F, c, yv, yq; gc.collect()
    # ---------------- target-month cubes + transductive rows
    CS, HOLD = {}, {}
    TX, TYV, TYQ = [], [], []
    for split in ('validation', 'private'):
        sm = read_split(panel, split, 'mainline_states_masked')
        cs = Cube(sm, stations); del sm
        CS[split] = cs
        if not TR_ROUNDS:
            HOLD[split] = (cs.melig & np.isfinite(cs.mv) & np.isfinite(cs.mq) & (np.random.default_rng(7).random(cs.mv.shape) < 0.15)) if split == 'validation' else np.zeros(cs.mv.shape, bool)
            continue
        obs = cs.melig & np.isfinite(cs.mv) & np.isfinite(cs.mq)
        hold = np.zeros(cs.mv.shape, bool)
        if split == 'validation':
            hold = obs & (np.random.default_rng(7).random(cs.mv.shape) < 0.15)
        HOLD[split] = hold
        vf3s = np.broadcast_to(st.vf.to_numpy()[None, None, :], cs.mv.shape)
        ln3s = np.broadcast_to(st.lanes.to_numpy()[None, None, :], cs.mv.shape)
        for r in range(TR_ROUNDS):
            hid = obs & ~hold & (np.random.default_rng(2000 + 10 * r + (split == 'private')).random(cs.mv.shape) < TR_HIDE)
            cv = Cube.__new__(Cube); cv.__dict__.update(cs.__dict__)
            gone = hid | hold
            cv.mv = np.where(gone, np.nan, cs.mv); cv.mq = np.where(gone, np.nan, cs.mq); cv.mo = np.where(gone, np.nan, cs.mo)
            Fr, _, _ = features(cv, st, hv[cv.dow], hq[cv.dow])
            Xr = np.column_stack([np.asarray(Fr[cols[i]])[hid] for i in ixD]).astype(np.float32)
            TX.append(Xr); TYV.append((cs.mv / vf3s)[hid] - Fr['tv'][hid]); TYQ.append((cs.mq / ln3s)[hid] - Fr['tq'][hid])
            del Fr, cv; gc.collect()
        log(panel, split, 'transductive rows so far', sum(len(t) for t in TX))
    pF = dict(params, num_leaves=511, min_data_in_leaf=100, learning_rate=0.04, seed=1, bagging_seed=11, feature_fraction_seed=21)
    NT = 60 if SMOKE else 1500
    TX = np.concatenate(TX); TYV = np.concatenate(TYV); TYQ = np.concatenate(TYQ)
    if len(TX) > 4 * MAX_ROWS:
        sel = rng.choice(len(TX), 4 * MAX_ROWS, replace=False); TX, TYV, TYQ = TX[sel], TYV[sel], TYQ[sel]
    XA = np.concatenate([Xtr, TX]); YV = np.concatenate([yvtr, TYV]); YQ = np.concatenate([yqtr, TYQ])
    del TX, TYV, TYQ, Xtr; gc.collect()
    MODELS = {'FTA': (lgb.train(pF, lgb.Dataset(XA, YV), NT), lgb.train(pF, lgb.Dataset(XA, YQ), NT), ixD)}
    imp = pd.Series(MODELS['FTA'][0].feature_importance('gain'), index=[cols[i] for i in ixD]).sort_values(ascending=False)
    log(panel, 'FTA speed top features', imp.head(10).round(0).to_dict())
    log(panel, 'train+transductive rows', len(XA))
    del XA, YV, YQ; gc.collect()
    for split in ('validation', 'private'):
        cs = CS[split]
        if split == 'validation':
            # internal holdout on the validation month itself: hide 15% of released eligible cells (same cells as v8)
            hold = HOLD[split]
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
            cand.update(predict_variants(Xh, Fh['tv'][hold], Fh['tq'][hold], vfh, lnh, regh, MODELS, ixD))
            vh = {k: round(s_state(a, b, hv_true, hq_true, lnh, regh)[0], 4) for k, (a, b) in cand.items()}
            row_v = dict(panel=panel, family=FAMILY(panel), **{f'val_{k}': v for k, v in vh.items()})
            val_rows.append(row_v); log(panel, 'VALIDATION-month internal holdout S_state', vh)
            del Fh, Xh, ch
        tm, tpl, (di, ti, si, ok) = template_mask(panel, split, cs, stations)
        Fs, tvs, tqs = features(cs, st, hv[cs.dow], hq[cs.dow])
        Xs = np.column_stack([np.asarray(Fs[k])[di[ok], ti[ok], si[ok]] for k in cols]).astype(np.float32)
        vfs = st.vf.to_numpy()[si[ok]]; lns = st.lanes.to_numpy()[si[ok]]
        regs = tpl.mask_regime.to_numpy()[ok]
        V3 = predict_variants(Xs, Fs['tv'][di[ok], ti[ok], si[ok]], Fs['tq'][di[ok], ti[ok], si[ok]], vfs, lns, regs, MODELS, ixD)
        out = tpl.loc[ok, ['timestamp', 'station_id', 'link_id', 'mask_regime']].copy()
        out.insert(0, 'panel', panel)
        for vname, (pvs, pqs) in V3.items():
            out[f'speed_{vname}'] = pvs; out[f'flow_{vname}'] = pqs
        out['speed_kmh'] = out.speed_FTA; out['flow_vph'] = out.flow_FTA
        miss = int((~ok).sum())
        log(panel, split, 'template rows', len(tpl), 'predicted', len(out), 'unmapped', miss,
            'speed range', np.round([out.speed_kmh.min(), out.speed_kmh.max()], 1).tolist(),
            'flow<50 share', round(float((out.flow_vph < 50).mean()), 4))
        subs.append(out)
        del Fs, Xs, cs, CS[split]; gc.collect()
    log(panel, f'panel time {time.time() - tp:.0f}s')

V = pd.DataFrame(val_rows); V.to_csv(f'{OUT}/t1_valholdout.csv', index=False)
print('\n===== validation-month internal holdout (family-mean S_state) =====')
print(V.groupby('family')[[c for c in V.columns if c.startswith('val_')]].mean().mean().round(4).to_string())
S = pd.concat(subs, ignore_index=True)
base_cols = ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']
for vname in ('FTA',):
    S[base_cols].assign(speed_kmh=S[f'speed_{vname}'], flow_vph=S[f'flow_{vname}']).to_csv(f'{OUT}/state_submission_{vname}.csv', index=False)
S[base_cols + ['speed_kmh', 'flow_vph']].to_csv(f'{OUT}/state_submission.csv', index=False)
log('state rows', len(S), 'NaN', int(S[['speed_kmh', 'flow_vph']].isna().sum().sum()))
