"""Task 1 t1bspec member = BLACKOUT SPECIALIST (t1ramp member, seed 13, with selection-faithful simulated blackouts and
30 timing columns). BLACKOUT-ONLY role (swap kernel: Task 2 blackout rows only). Same panel halves, data loading, outputs,
gates as the source kaggle_jobs/t1_ramp_member_a/b.py except
  (1) simulated blackouts in the validation / private months at SELECTION-FAITHFUL origins (bs_faithful_cands): onset =
      first crossing (no observed queued cell in T+1..T+5, one at T+6, as the official onset windows), ongoing = run-starts
      of the faithful ongoing map (time-filled persistence <= 0.9), balanced half onset (official 5 + 5), DARK_N per month;
  (2) features() = t1ramp features + 30 timing columns BCOLS (bspec_feats; NaN outside blackout rows): condition flag,
      slot offset, onset breakdown flag (k >= 6), oriented boundary values before T / after T+18 (own, +-2 stations),
      corridor queued share at both boundaries, onset-aware piecewise interpolation minus linear, ramp flow over the gap.
      Origins per view: train split = pseudo gap origins with their cond (tfb-t2lab-full pseudo_<panel>.npz); simulated
      blackout view = the simulated origins with their type; released month (template prediction) = the OFFICIAL Task 2
      origins with the condition of task2/<panel>/<split>/window_index.csv; transductive rounds = none;
  (3) feature-count gate = V12 + FD + 9 + 30.
  Local pseudo-holdout: analysis/t1lab/t1bspec_exp.py / t1bspec_report.py (results analysis/t2lab/results/t1bspec).
Previous notes:
Task 1 t1ramp member (= t1dark member, seed 13, + 9 RAMP feature columns). BLACKOUT-ONLY role: replaces the t1dark member
on the Task 2 blackout rows in the swap kernel (lab: W 0.75 >= W 1.0 > W 0.5; W 0.5 is the drop-in). Same panel halves,
data loading, outputs, gates as the source kaggle_jobs/t1_t1dark_member_a/b.py except
  (1) ramp_cube(): released on-/off-ramp flows of corridors/<panel>/<split>/ramp_states (every valued cell, eligible or
      not) summed onto the station of the attached mainline link (network/ramp_attachment_map.csv); ramps are released in
      the blackouts. Panels without a ramp map / ramp files get all-NaN ramp cubes (features NaN, LightGBM handles it).
  (2) features() = production features + 9 ramp columns appended after the FD columns (ramp_feats(): smoothed on/off flow,
      attached-ramp count, ramp-anchored ratio transfer at the own link and over +-2 stations, their ratio to the temporal
      interpolation, net ramp flow and conservation-implied flow from the upstream neighbour).
  (3) feature-count gate = V12 + FD + 9.
  Local pseudo-holdout (analysis/t1lab/t1ramp_exp.py / t1ramp_report.py, results analysis/t2lab/results/t1ramp, 4 panels x
  2 months, 2 seeds): blackout cells dark+ramp vs dark speed RMSE 9.70 -> 8.82 (-9.1 %), flow/lane 80.96 -> 78.02 (-3.6 %),
  8/8 panel-months; ordinary cells neutral (speed -0.006 [-0.033, +0.021]); deployment 0.5 bag + 0.5 member on blackout
  rows: J proxy +0.00052 [+0.00046, +0.00058] vs the t1dark member, 8/8.
  Needs kernel source hosen42/tfb-t2lab-full (pseudo_<panel>.npz, train gap origins) attached, like the t1dark member.
Previous notes:
Task 1 t1dark member (= v13 FD-feature member, seed 13) + SIMULATED-BLACKOUT transductive rows.
t1dark notes: identical to t1_pipeline_v13a/b.py (same panel halves, data loading, features, model, outputs, gates) except
  (1) seed 13 wherever it is free: train-row subsample rng(13), transductive rounds rng(14000+...), LightGBM seed 13;
  (2) in every validation / private month of a panel with Task 2 windows (task2/<panel>/<split>/window_index.csv),
      up to DARK_N queue-like origins T are drawn on the released masked month (eligibility coverage >= 0.7 in
      T-12..T-1, an observed queued link (v <= 0.6 vf) in T+1..T+6, ongoing ones with persistence <= 0.9), at least
      GAP slots away from the official blackouts and from each other; rows T+1..T+GAP are blanked at every station
      (all channels, as the release does after an official origin) and the released eligible cells inside them
      (validation 15% holdout excluded) become extra transductive rows (features from that view, label = released value).
  Local pseudo-holdout (analysis/t1lab/t1dark_exp.py): see analysis/t2lab/results/t1dark.
Previous notes (v13):
Task 1 v13 (FD-feature bagging member, 3 transductive rounds, cap 6x rows, 15% val holdout kept): seed 7.
v13 notes: identical to v12s5 (same panel split per half, data loading, output names, gates) except
  (1) seed 7 wherever it is free: train-row subsample rng(7), transductive hidden-cell rounds rng(8000+...), LightGBM seed 7
      (holdout rng(7) on the val-month cube unchanged, so the holdout cells are the same as s4/s5);
  (2) features(): + 20 FD-implied columns (congested-branch density / flow implied by 8 speed views, flow-implied speed,
      density ratio, w, k_jam per lane), exactly as scratchpad t1loc/fdfeat2.py.
  Gate vs s4/s5 of the same half on t1_valholdout.csv (val_FTA): KEEP mean delta >= +0.0010 and >= 8/10 panels better;
  DROP delta < +0.0005 or >= 3 panels worse.
v12 notes: Task 1 v12 (bagging member): identical to v11 FTA except seed 1 everywhere it is free: train-row subsample rng(1),
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
import glob, json, os, re, time, gc, resource
import numpy as np
import pandas as pd
import lightgbm as lgb

T0 = time.time()
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
OUT = '/kaggle/working'
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S',
          'D12_I5_N', 'D12_I5_S', 'D12_I405_N', 'D12_I405_S']
PANELS = PANELS[:5]
SMOKE = False
N_TREES, MAX_ROWS = (60, 150_000) if SMOKE else (300, 600_000)
if SMOKE:
    PANELS = ['D7_I10_W']
FAMILY = lambda p: p.rsplit('_', 1)[0]
RATE = {'R1': .2, 'R2': .3, 'R3': .5}
TR_ROUNDS, TR_HIDE = 3, 0.20
ASM_WAVES, ASM_K, ASM_SIGMA, ASM_TAU = (80.0, -80.0, 18.0, -18.0), 5, 0.6, 5.0


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def peak_gb():
    """peak resident set size of this process (Linux ru_maxrss is in KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


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


def features_base(c, st, hv, hq):
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
    n12 = len(F)
    assert n12 == V12_NFEAT, f'v12 feature count changed: {n12} != {V12_NFEAT}'
    F.update(fd_feats(F, st, shape))
    assert len(F) == n12 + N_FD, f'FD feature count {len(F) - n12} != {N_FD}'
    return F, tv, tq


V12_NFEAT = 78     # 15 base + 24 v/q nb + 8 v0/q0 + 8 occ + to/so + 2 rq + 7 low-rank + 12 ASM
FD_SPEEDS = ('tv', 'sv', 'v-1+0', 'v+1+0', 'v0-1', 'v0+1', 'asm_v-18', 'asm_v+18')
FD_COLS = [f'fd{kind}_{nm}' for nm in FD_SPEEDS for kind in 'kq'] + ['fdv_tq', 'fdk_ratio', 'fd_w', 'fd_kj']
N_FD = len(FD_COLS)
assert N_FD == 20


# ------------------------------------------------------------------ t1ramp: released ramp flows -> 9 feature columns
RCOLS = ['rp_on', 'rp_off', 'rp_n', 'rp_own', 'rp_nb', 'rp_rel_own', 'rp_rel_nb', 'rp_net', 'rp_cons']
N_RAMP = len(RCOLS)


def ramp_cube(panel, split, st, days):
    """released on-/off-ramp flows (vph) summed onto the station of the attached mainline link: dict(ON, OFF [D, 288, S],
    NaN = no valued ramp cell / no ramp). Every VALUED ramp cell is used (eligible or not; the ineligible-but-valued
    cells are as accurate). Ramps are never hidden in the Task 2 blackouts. Panels without ramps -> all-NaN cubes."""
    D, S = len(days), len(st)
    ON = np.full((D, 288, S), np.nan, np.float32); OFF = ON.copy()
    fmap = f'{REL}/corridors/{panel}/network/ramp_attachment_map.csv'
    if not os.path.exists(fmap):
        return dict(ON=ON, OFF=OFF)
    amap = pd.read_csv(fmap, dtype=str)
    l2s = {l: i for i, l in enumerate(st.link_id.astype(str))}
    amap = amap[amap.nearest_mainline_link_id.isin(l2s)]
    rl = dict(zip(amap.ramp_link_id, amap.nearest_mainline_link_id.map(l2s)))
    dpos = {str(dd)[:10]: i for i, dd in enumerate(days)}
    for f in sorted(glob.glob(f'{REL}/corridors/{panel}/{split}/ramp_states/**/*.parquet', recursive=True)):
        r = pd.read_parquet(f, columns=['timestamp', 'ramp_link_id', 'ramp_type', 'flow_vph'])
        r = r[r.flow_vph.notna() & (r.flow_vph >= 0) & r.ramp_link_id.isin(rl)]
        if r.empty:
            continue
        ts = pd.to_datetime(r.timestamp, utc=True)
        r = r.assign(d=ts.dt.strftime('%Y-%m-%d').map(dpos), t=(ts.dt.hour * 12 + ts.dt.minute // 5), l=r.ramp_link_id.map(rl),
                     on=r.ramp_type.str.upper().isin(['OR', 'ON'])).dropna(subset=['d'])
        for flag, A in ((True, ON), (False, OFF)):
            g = r[r.on == flag].groupby(['d', 't', 'l']).flow_vph.sum().reset_index()
            A[g.d.astype(int).to_numpy(), g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.flow_vph.to_numpy()
    return dict(ON=ON, OFF=OFF)


def ramp_direction(panel):
    """+1: traffic flows toward increasing milepost (E/N panels), -1: W/S panels (research/topology_audit.md)."""
    return 1 if panel.split('_')[-1] in ('E', 'N') else -1


def ramp_feats(mq, ramp, st, tqn):
    """9 ramp columns per (day, slot, station) (= analysis/t1lab/t1ramp_exp.py ramp_feats). Ramp flows are ~1% of the
    mainline flow but track it (per-link corr ~0.8, ratio CV ~0.27): the useful signal is the ramp-ANCHORED RATIO TRANSFER
    (like rq+-1): mainline flow / ramp flow interpolated in time from the released mainline slots, times the ramp flow."""
    ln = st.lanes.to_numpy()[None, None, :].astype(np.float32)
    nq = mq / ln
    def sm(A):                                       # centered 3-slot mean, then filled within the day (interpolation)
        with np.errstate(all='ignore'):
            a = np.nanmean(np.stack([shift(A, -1, 0), A, shift(A, 1, 0)]), 0)
        return interp_axis(a, 1)
    def transfer(A):                                 # A > 0 (vph) -> per-lane mainline flow estimate
        with np.errstate(all='ignore'):
            r = interp_axis(np.log(nq) - np.log(A), 1)
            return (A * np.exp(r)).astype(np.float32)
    son, soff = sm(ramp['ON']), sm(ramp['OFF'])
    F = {'rp_on': son / ln, 'rp_off': soff / ln,
         'rp_n': np.broadcast_to((ramp['has_on'].astype(np.float32) + ramp['has_off'])[None, None, :], son.shape)}
    with np.errstate(all='ignore'):
        F['rp_own'] = np.nanmean(np.stack([transfer(son + 5), transfer(soff + 5)]), 0)
        tot = np.nan_to_num(son) + np.nan_to_num(soff)
        G = sum(np.nan_to_num(shift(tot, 0, ds)) for ds in (-2, -1, 0, 1, 2))
        F['rp_nb'] = transfer(np.where(G > 0, G + 5, np.nan))
        F['rp_rel_own'] = F['rp_own'] / tqn - 1; F['rp_rel_nb'] = F['rp_nb'] / tqn - 1
        net = np.nan_to_num(son) - np.nan_to_num(soff)
        up = -ramp['dirn']                           # upstream neighbour index offset
        net2 = net + np.nan_to_num(shift(net, 0, up))
        F['rp_net'] = net2 / ln
        F['rp_cons'] = (shift(tqn * ln, 0, up) + net2) / ln   # conservation-implied: upstream flow + net ramp flow between
    return {k: np.asarray(v, np.float32) for k, v in F.items()}


# ================================================================== BSPEC BLOCK (copied verbatim into the kernel)
BS_OFFS = (('0', 0), ('u1', -1), ('u2', -2), ('d1', 1), ('d2', 2))
BCOLS = (['bs_cond', 'bs_k', 'bs_brk'] + [f'bs_{w}_{ch}{o}' for w in ('pre', 'post') for ch in 'vq' for o, _ in BS_OFFS]
         + ['bs_qsh_pre', 'bs_qsh_post', 'bs_pwd_v', 'bs_pwd_q', 'bs_rnet_cum', 'bs_ron_rel', 'bs_roff_rel'])
N_BS = len(BCOLS)
BS_GAP, BS_ONSET_K = 18, 6                       # blackout rows T+1..T+18; official onset queue first at T+6 (T+30 min)


def bs_edge(A, d, t0, step, lim=3):
    """first finite value of A[d, t0 + step * j, :] for j = 0..lim-1 (per station; NaN if none)."""
    out = np.full(A.shape[2], np.nan, np.float32)
    for j in range(lim):
        t = t0 + step * j
        if 0 <= t < A.shape[1]:
            a = A[d, t]; m = np.isnan(out) & np.isfinite(a); out[m] = a[m]
    return out


def bs_orient(x, off):
    """value at station index s + off (NaN outside)."""
    y = np.full_like(x, np.nan)
    if off > 0:
        y[:-off] = x[off:]
    elif off < 0:
        y[-off:] = x[:off]
    else:
        y[:] = x
    return y


def bspec_feats(mv, mq, origins, vf, ln, dirn, tv, tq, rp_on, rp_off, rp_net):
    """timing columns of the blackout rows T+1..T+BS_GAP of every origin (d, T, c) (c = 0 onset / 1 ongoing); NaN elsewhere.
    mv / mq: the view (blackout rows blank), vph / km/h; vf, ln per station; dirn = +1 if traffic flows toward increasing
    station index; tv / tq / rp_* = production feature cubes (normalised per vf / per lane)."""
    D, T, S = mv.shape
    out = {k: np.full((D, T, S), np.nan, np.float32) for k in BCOLS}
    with np.errstate(all='ignore'):
        nv = mv / vf[None, None, :]; nq = mq / ln[None, None, :]
        for d, t0, c in origins:
            d, t0, c = int(d), int(t0), int(c)
            kk = np.arange(1, BS_GAP + 1); kk = kk[t0 + kk < T]
            if len(kk) == 0:
                continue
            rows = t0 + kk; K = kk[:, None].astype(np.float32)
            pre = dict(v=bs_edge(nv, d, t0, -1), q=bs_edge(nq, d, t0, -1))
            post = dict(v=bs_edge(nv, d, t0 + BS_GAP + 1, 1), q=bs_edge(nq, d, t0 + BS_GAP + 1, 1))
            out['bs_cond'][d, rows] = c
            out['bs_k'][d, rows] = K
            out['bs_brk'][d, rows] = ((c == 0) & (K >= BS_ONSET_K)).astype(np.float32)
            for w, Dd in (('pre', pre), ('post', post)):
                for ch in 'vq':
                    for o, off in BS_OFFS:
                        out[f'bs_{w}_{ch}{o}'][d, rows] = bs_orient(Dd[ch], off * dirn)[None, :]
                fin = np.isfinite(Dd['v'])
                out[f'bs_qsh_{w}'][d, rows] = float((Dd['v'][fin] <= 0.6).sum() / max(1, fin.sum()))
            frac = np.clip((K - (BS_ONSET_K - 1)) / (BS_GAP + 1 - (BS_ONSET_K - 1)), 0, 1) if c == 0 else K / (BS_GAP + 1)
            for ch, tx in (('v', tv), ('q', tq)):
                a, b = pre[ch][None, :], post[ch][None, :]
                base = tx[d, rows]
                pw = a + (b - a) * frac
                out[f'bs_pwd_{ch}'][d, rows] = np.where(np.isfinite(pw), pw - base, 0.0)
            out['bs_rnet_cum'][d, rows] = np.cumsum(np.nan_to_num(rp_net[d, rows]), 0) * (5.0 / 60.0)
            out['bs_ron_rel'][d, rows] = (rp_on[d, rows] + 1.0) / (rp_on[d, t0][None, :] + 1.0) - 1.0
            out['bs_roff_rel'][d, rows] = (rp_off[d, rows] + 1.0) / (rp_off[d, t0][None, :] + 1.0) - 1.0
    return out


def bs_fill_time36(V, maxgap=36):
    """linear interpolation along time within each day for interior gaps <= maxgap slots (SELECTION ONLY)."""
    out = V.copy(); D, T, L = V.shape; idx = np.arange(T)
    for d in range(D):
        okd = np.isfinite(V[d])
        for l in np.where(okd.any(0) & ~okd.all(0))[0]:
            ok = okd[:, l]
            if ok.sum() < 2:
                continue
            yi = np.interp(idx, idx[ok], V[d, ok, l])
            prev = np.maximum.accumulate(np.where(ok, idx, -10 ** 6)); nxt = np.minimum.accumulate(np.where(ok, idx, 10 ** 6)[::-1])[::-1]
            m = (~ok) & (nxt - prev - 1 <= maxgap) & (prev >= 0) & (nxt < T)
            out[d, m, l] = yi[m]
    return out


def bs_faithful_cands(mv, obs, Em, rr, vf, gap=18, h=12, f=6):
    """selection-faithful blackout origins (d, T, c) on a released masked month, >= gap slots from the blank rows rr:
    c = 0 onset FIRST CROSSING (coverage >= 0.7 in T-h..T-1, no established observed queue in the history, no observed
    queued cell in T+1..T+5, an observed queued cell at T+6); c = 1 ongoing RUN-STARTS of the faithful map (coverage,
    established observed queue, a queued time-filled cell in T+1..T+f, persistence of the time-filled state <= 0.9)."""
    cut = 0.6 * vf[None, None, :]
    Qm = obs & (mv <= cut)
    V = np.where(obs, mv, np.nan).astype(np.float32)
    VF = bs_fill_time36(V); MF = np.isfinite(VF); QF = MF & (VF <= cut)
    D = mv.shape[0]
    og = np.zeros((D, 288), bool); out = []
    for d in range(D):
        for t in range(h, 288 - gap):
            if Em[d, t - h:t].mean() < 0.7:
                continue
            Qh = Qm[d, t - h:t]
            if Qh.sum() >= 2 and (Qh.sum(0) >= 2).any():
                Tf = QF[d, t + 1:t + 1 + f]; Mf = MF[d, t + 1:t + 1 + f]
                if not Tf.any():
                    continue
                S_ = np.repeat(QF[d, t][None], f, 0)
                u = ((Tf | S_) & Mf).sum(); i = (Tf & S_ & Mf).sum()
                og[d, t] = (1.0 if u == 0 else i / u) <= 0.9
            elif (not Qm[d, t + 1:t + f].any()) and Qm[d, t + f].any():
                if not rr[d, max(t + 1 - gap, 0):min(t + 2 * gap + 1, 288)].any():
                    out.append((d, t, 0))
    for d, t in zip(*np.where(og)):
        if not og[d, t - 1] and not rr[d, max(t + 1 - gap, 0):min(t + 2 * gap + 1, 288)].any():
            out.append((int(d), int(t), 1))
    return sorted(out)
# ================================================================== END BSPEC BLOCK


def features(c, st, hv, hq):
    """t1ramp: production features + the 9 ramp columns (appended after the FD columns) from c.ramp (set on every cube;
    views copy it with __dict__)."""
    F, tv, tq = features_base(c, st, hv, hq)
    F.update(ramp_feats(c.mq, c.ramp, st, F['tq']))
    F.update(bspec_feats(c.mv, c.mq, getattr(c, 'bs_origins', []), st.vf.to_numpy(), st.lanes.to_numpy(), c.ramp['dirn'],
                         F['tv'], F['tq'], F['rp_on'], F['rp_off'], F['rp_net']))   # t1bspec: timing columns (blackout rows only)
    assert len(F) == V12_NFEAT + N_FD + N_RAMP + N_BS and list(F)[-N_BS - N_RAMP:-N_BS] == RCOLS and list(F)[-N_BS:] == BCOLS
    return F, tv, tq


def fd_feats(F, st, shape):
    """FD-implied features (= scratchpad t1loc/fdfeat2.py). Per station, per lane:
    w = cap / max(k_jam - k_crit, 1e-6) (congested wave speed), kj = k_jam.
    For each speed view vv (km/h = normalised view * vf): congested-branch density w*kj/(max(vv,1)+w) and flow w*kj*vv/(max(vv,1)+w);
    fdv_tq = per-lane interp flow / max(kj - tq/w, 0.5) / vf; fdk_ratio = tk / fdk_tv; fd_w, fd_kj. NaN views stay NaN."""
    ln = st.lanes.to_numpy().astype(np.float64)
    capS = st.cap.to_numpy() / ln; kcS = st.critical_density.to_numpy() / ln; kjS = st.k_jam.to_numpy() / ln
    w_ = (capS / np.maximum(kjS - kcS, 1e-6))[None, None, :]; kj_ = kjS[None, None, :]
    vf = st.vf.to_numpy().astype(np.float64)[None, None, :]
    out = {}
    with np.errstate(all='ignore'):
        for nm in FD_SPEEDS:
            vv = np.asarray(F[nm], np.float64) * vf
            den = np.clip(vv, 1, None) + w_
            out[f'fdk_{nm}'] = (w_ * kj_ / den).astype(np.float32)
            out[f'fdq_{nm}'] = (w_ * kj_ * vv / den).astype(np.float32)
        tqn = np.asarray(F['tq'], np.float64)
        out['fdv_tq'] = (tqn / np.clip(kj_ - tqn / w_, 0.5, None) / vf).astype(np.float32)
        out['fdk_ratio'] = (np.asarray(F['tk'], np.float64) / out['fdk_tv']).astype(np.float32)
    out['fd_w'] = np.broadcast_to(w_.astype(np.float32), shape)
    out['fd_kj'] = np.broadcast_to(kj_.astype(np.float32), shape)
    assert list(out) == FD_COLS
    return out


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
    return [(pos[all_days[d]], int(t), 0 if str(cc) == 'queue_onset' else 1) for d, t, cc in zip(z['d'], z['t'], z['cond'])
            if all_days[d] in pos]                                   # t1bspec: + condition (0 onset / 1 ongoing)


def darken(c, origins):
    mv, mq, mo = c.mv.copy(), c.mq.copy(), c.mo.copy(); gapm = np.zeros(mv.shape, bool)
    for d, t, _ in origins:
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


# ------------------------------------------------------------------ t1dark: simulated blackouts in the target months
DARK_N, DARK_SEED = 48, 1300


def real_rows(panel, split, cs):
    """[day, 288] rows inside the official blackouts (T+1..T+GAP after each window origin) plus any all-blank row."""
    rr = (cs.melig.sum(2) >= 5) & (np.isfinite(cs.mv).sum(2) == 0)
    pos = {str(d): i for i, d in enumerate(cs.days)}
    for o in pd.to_datetime(pd.read_csv(f'{REL}/task2/{panel}/{split}/window_index.csv').forecast_origin, utc=True):
        d = o.strftime('%Y-%m-%d')
        if d in pos:
            t = o.hour * 12 + o.minute // 5
            rr[pos[d], t + 1:t + 1 + GAP] = True
    return rr


def official_origins(panel, split, cs):
    """t1bspec: (day, T, c) of the official Task 2 windows of this month, c = 0 queue_onset / 1 queue_ongoing (window_index)."""
    f = f'{REL}/task2/{panel}/{split}/window_index.csv'
    if not os.path.exists(f):
        return []
    wi = pd.read_csv(f); pos = {str(d): i for i, d in enumerate(cs.days)}; out = []
    for o, cond in zip(pd.to_datetime(wi.forecast_origin, utc=True), wi.condition.astype(str)):
        d = o.strftime('%Y-%m-%d')
        if d in pos:
            out.append((pos[d], o.hour * 12 + o.minute // 5, 0 if cond == 'queue_onset' else 1))
    assert set(wi.condition.astype(str)) <= {'queue_onset', 'queue_ongoing'}, set(wi.condition)
    return out


def sim_blackouts(cs, st, rr, seed):
    """t1bspec: [day, 288] rows of up to DARK_N simulated blackouts at SELECTION-FAITHFUL origins (bs_faithful_cands: onset
    first crossing at T+6, ongoing run-starts), balanced half onset, >= 2 GAP apart; also returns the (d, T, c) list."""
    vf = st.vf.to_numpy()
    obs = cs.melig & np.isfinite(cs.mv) & np.isfinite(cs.mq)
    cands = bs_faithful_cands(cs.mv, obs, cs.melig, rr, vf, GAP, 12, 6)
    rng = np.random.default_rng(seed); acc = []
    order = rng.permutation(len(cands))
    on = [i for i in order if cands[i][2] == 0]; og = [i for i in order if cands[i][2] == 1]
    for pool, cap in ((on, DARK_N // 2), (og, DARK_N), (on, DARK_N)):     # balanced like the official windows (half onset)
        for i in pool:
            if len(acc) >= cap:
                break
            d, t, c = cands[i]
            if (d, t, c) not in acc and all(dd != d or abs(tt - t) >= 2 * GAP for dd, tt, _ in acc):
                acc.append((d, t, c))
    tb = np.zeros(cs.mv.shape[:2], bool)
    for d, t, _ in acc:
        tb[d, t + 1:t + 1 + GAP] = True
    return tb, len(cands), len(acc), sorted(acc)


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
    RT = ramp_cube(panel, 'train', st, c.days)                       # t1ramp: released ramp flows (train split)
    RHAS = dict(has_on=np.isfinite(RT['ON']).any((0, 1)), has_off=np.isfinite(RT['OFF']).any((0, 1)), dirn=ramp_direction(panel))
    c.ramp = dict(RT, **RHAS); del RT
    log(panel, 't1ramp: stations with on-ramps', int(RHAS['has_on'].sum()), 'off-ramps', int(RHAS['has_off'].sum()), 'of', len(stations),
        '| valued share on %.3f off %.3f' % (np.isfinite(c.ramp['ON'][:, :, RHAS['has_on']]).mean() if RHAS['has_on'].any() else 0,
                                             np.isfinite(c.ramp['OFF'][:, :, RHAS['has_off']]).mean() if RHAS['has_off'].any() else 0))
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
    if os.path.exists(f'{REL}/task2/{panel}/train/window_index.csv'):                 # t1dark: never train without the simulated train gaps
        assert origins, f'no gap origins for {panel}: is tfb-t2lab-full (pseudo_<panel>.npz) attached?'
    gshare = split_gap_share(panel, 'validation', stations)
    mv0, mq0 = c.mv, c.mq
    c.mv, c.mq, c.mo, gapm = darken(c, origins)                     # train on the darkened view
    c.bs_origins = origins                                          # t1bspec: timing columns on the train gap rows
    F, tv, tq = features(c, st, hv[c.dow], hq[c.dow])
    X, cols = stack(F, tmask)
    # hard gates (v13): feature count = v12 + 20, no all-NaN FD column on the training rows
    assert len(cols) == V12_NFEAT + N_FD + N_RAMP + N_BS and cols[V12_NFEAT:V12_NFEAT + N_FD] == FD_COLS \
        and cols[-N_BS - N_RAMP:-N_BS] == RCOLS and cols[-N_BS:] == BCOLS, (len(cols), cols[-N_BS:])
    _gp = gapm[tmask]; _bc = X[:, cols.index('bs_cond')]
    assert np.array_equal(np.isfinite(_bc), _gp), 't1bspec: timing columns must be finite exactly on the train gap rows'
    log(panel, 't1bspec: train gap rows', int(_gp.sum()), 'onset share', round(float((_bc[_gp] == 0).mean()), 3) if _gp.any() else '-',
        'boundary finite share', round(float(np.isfinite(X[_gp, cols.index('bs_pre_v0')]).mean()), 3) if _gp.any() else '-')
    fd_fin = {k: float(np.isfinite(X[:, cols.index(k)]).mean()) for k in FD_COLS}
    assert all(v > 0 for v in fd_fin.values()), f'all-NaN FD columns: {[k for k, v in fd_fin.items() if v == 0]}'
    log(panel, 'features', len(cols), '(v12', V12_NFEAT, '+ FD', N_FD, ') FD finite share min',
        round(min(fd_fin.values()), 4), {k: round(v, 3) for k, v in fd_fin.items() if v < 0.99})
    in_gap = gapm[tmask]
    log(panel, 'gap origins', len(origins), 'targets in simulated gaps', int(in_gap.sum()),
        'validation gap share of targets', round(gshare, 4))
    vf3 = np.broadcast_to(st.vf.to_numpy()[None, None, :], c.mv.shape)
    yv = (c.v / vf3)[tmask] - F['tv'][tmask]; yq = (c.q / lanes3)[tmask] - F['tq'][tmask]
    dayid = np.broadcast_to(np.arange(len(c.days))[:, None, None], c.mv.shape)[tmask]
    block = np.minimum(dayid * 3 // len(c.days), 2)                      # 3 contiguous blocks (frozen)
    reg_t = np.broadcast_to(c.reg[:, None, None], c.mv.shape)[tmask]
    rng = np.random.default_rng(13)
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
        cs.ramp = dict(ramp_cube(panel, split, st, cs.days), **RHAS)   # t1ramp: released ramp flows of the target month
        cs.bs_origins = official_origins(panel, split, cs)               # t1bspec: official windows + window_index condition
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
            hid = obs & ~hold & (np.random.default_rng(14000 + 10 * r + (split == 'private')).random(cs.mv.shape) < TR_HIDE)
            cv = Cube.__new__(Cube); cv.__dict__.update(cs.__dict__); cv.bs_origins = []
            gone = hid | hold
            cv.mv = np.where(gone, np.nan, cs.mv); cv.mq = np.where(gone, np.nan, cs.mq); cv.mo = np.where(gone, np.nan, cs.mo)
            Fr, _, _ = features(cv, st, hv[cv.dow], hq[cv.dow])
            Xr = np.column_stack([np.asarray(Fr[cols[i]])[hid] for i in ixD]).astype(np.float32)
            TX.append(Xr); TYV.append((cs.mv / vf3s)[hid] - Fr['tv'][hid]); TYQ.append((cs.mq / ln3s)[hid] - Fr['tq'][hid])
            del Fr, cv; gc.collect()
        if DARK_N and os.path.exists(f'{REL}/task2/{panel}/{split}/window_index.csv'):     # t1dark blackout rows
            rr = real_rows(panel, split, cs)
            tb, n_c, n_s, sims = sim_blackouts(cs, st, rr, DARK_SEED + (split == 'private'))
            assert n_s > 0, f'no simulated blackouts for {panel} {split}'
            TB = np.broadcast_to(tb[:, :, None], cs.mv.shape)
            hid = obs & TB & ~hold
            cv = Cube.__new__(Cube); cv.__dict__.update(cs.__dict__); cv.bs_origins = sims   # t1bspec: simulated types
            gone = TB | hold
            cv.mv = np.where(gone, np.nan, cs.mv); cv.mq = np.where(gone, np.nan, cs.mq); cv.mo = np.where(gone, np.nan, cs.mo)
            Fr, _, _ = features(cv, st, hv[cv.dow], hq[cv.dow])
            Xr = np.column_stack([np.asarray(Fr[cols[i]])[hid] for i in ixD]).astype(np.float32)
            TX.append(Xr); TYV.append((cs.mv / vf3s)[hid] - Fr['tv'][hid]); TYQ.append((cs.mq / ln3s)[hid] - Fr['tq'][hid])
            log(panel, split, 't1bspec: official origins', len(cs.bs_origins), 'onset', sum(o[2] == 0 for o in cs.bs_origins),
                '| simulated onset', sum(s[2] == 0 for s in sims), 'ongoing', sum(s[2] == 1 for s in sims))
            log(panel, split, 't1dark: real blackout rows', int(rr.sum()), '| simulated blackouts', n_s, 'of', n_c,
                'candidates | blackout transductive rows', int(hid.sum()))
            del Fr, cv; gc.collect()
        log(panel, split, 'transductive rows so far', sum(len(t) for t in TX))
    pF = dict(params, num_leaves=511, min_data_in_leaf=100, learning_rate=0.04, seed=13, bagging_seed=23, feature_fraction_seed=33)
    NT = 100 if SMOKE else 1500
    TX = np.concatenate(TX); TYV = np.concatenate(TYV); TYQ = np.concatenate(TYQ)
    if len(TX) > 6 * MAX_ROWS:
        sel = rng.choice(len(TX), 6 * MAX_ROWS, replace=False); TX, TYV, TYQ = TX[sel], TYV[sel], TYQ[sel]
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
        _bc = Xs[:, cols.index('bs_cond')]
        log(panel, split, 't1bspec: template rows inside official blackouts', int(np.isfinite(_bc).sum()),
            'onset', int((_bc == 0).sum()), 'ongoing', int((_bc == 1).sum()))
        bad = [k for k in FD_COLS if not np.isfinite(Xs[:, cols.index(k)]).any()]
        assert not bad, f'{panel} {split}: all-NaN FD columns on template rows {bad}'
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
    log(panel, f'panel time {time.time() - tp:.0f}s', f'peak RSS {peak_gb():.2f} GB')

V = pd.DataFrame(val_rows); V.to_csv(f'{OUT}/t1_valholdout.csv', index=False)
print('\n===== validation-month internal holdout (family-mean S_state) =====')
print(V.groupby('family')[[c for c in V.columns if c.startswith('val_')]].mean().mean().round(4).to_string())
print('\n===== per-panel val-month holdout S_state (gate: compare val_FTA with s4/s5 of the same half) =====')
print(V[['panel'] + [c for c in V.columns if c.startswith('val_')]].to_string(index=False))
print('mean val_FTA', round(float(V.val_FTA.mean()), 5))
S = pd.concat(subs, ignore_index=True)
base_cols = ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']
for vname in ('FTA',):
    S[base_cols].assign(speed_kmh=S[f'speed_{vname}'], flow_vph=S[f'flow_{vname}']).to_csv(f'{OUT}/state_submission_{vname}.csv', index=False)
S[base_cols + ['speed_kmh', 'flow_vph']].to_csv(f'{OUT}/state_submission.csv', index=False)
log('state rows', len(S), 'NaN', int(S[['speed_kmh', 'flow_vph']].isna().sum().sum()))
log(f'peak RSS {peak_gb():.2f} GB')
if SMOKE:
    assert peak_gb() < 25, f'peak RSS {peak_gb():.2f} GB >= 25 GB'
    log('SMOKE gates OK: features', V12_NFEAT + N_FD, 'no all-NaN FD columns, peak RSS < 25 GB')
