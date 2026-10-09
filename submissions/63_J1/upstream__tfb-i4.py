"""FDRECON / TV swap (Task 1/3 post-processing on the exact base file; no training). Derived from the local draft
scratchpad/t1loc/swap_fdrecon_tv.py with the data loading, output naming and hard gates of swap_combo15.py.

Step R (FD-gated reconciliation, DO_RECON): at state rows with v < GATE*vf, move the per-lane density q/v onto the released congested
FD branch implied by the predicted speed, k_FD(v) = w*kj/(v+w), w = cap/(kj-kc) (all per lane). FD parameters are looked up by
(panel, link_id) by AVERAGING the station rows of fd_parameters.csv per link, then kc = cap/vf and kj = max(kj, 1.05 kc), exactly like
repo/src/task3/score_task3.py network_parameters(). r = clip(kFD/k, e^-0.5, e^0.5); v' = clip(v r^-0.75, 3, 1.05 vf); q' = max(q r^0.25, 20).
Step T (1-D TV, DO_TV): Hisernberg-style TV smoothing of per-lane density inside every maximal temporal run of target cells per station
(a run breaks where consecutive target timestamps are more than 5 min apart; no UTC-date grouping). Free cells tau 0.0075 / anchor 0.001,
gate cells tau 0.02 / anchor 0.005, each scaled by the run's mean density, 300 dual iterations; anchors = released eligible cells of the
same split at t0-5min / tend+5min. Gate cells split the density change 0.75 to speed, free cells put it all on flow.
Panel exclusion rule (fixed before any LB result): on released eligible congested cells (v < 0.6 vf) of mainline_states_masked
(validation + private pooled), rel = (q/lanes/v - kFD(v))/kFD(v); a panel is EXCLUDED from R and T when |median(rel)| > 2%, or the IQR
is not inside [-5%, +5%], or it has fewer than EXCL_MIN_N such cells.
Outputs (one run): submission_FDR.csv (R only), submission_FDRTV.csv (R then T), submission_TV.csv (T only).
Local evidence (scratchpad, D12_I405_S, real train targets, 2-member bag): R dJ +0.0013, R+T +0.0017 (S_state +0.001, proxy LWR +0.013).
Queue and Task 4 rows untouched (asserted byte-identical)."""
import glob, re, numpy as np, pandas as pd

SMOKE = False                                   # True: diagnostics run on COMBO15, outputs prefixed smoke_ (never submit them)
DO_RECON, DO_TV = True, True
BASE_KERNEL, BASE_FILE = 'tfb-i4d', 'submission_T1DARK.csv'
if SMOKE:
    BASE_KERNEL, BASE_FILE = 'tfb-combo15', 'submission_COMBO15.csv'
PREFIX = 'smoke_submission' if SMOKE else 'submission'
GATE, A = 0.6, 0.75
LOGR_MAX = 0.5
TV_FREE, TV_FREE_A, TV_GATE, TV_GATE_A, TV_ITERS = 0.0075, 0.001, 0.02, 0.005, 300
RUN_GAP_NS = 5 * 60 * 10 ** 9                   # a TV run breaks where consecutive target timestamps are > 5 min apart
W_RANGE, KJL_RANGE = (8.0, 40.0), (60.0, 200.0)  # km/h, veh/km/lane
EXCL_MED, EXCL_IQR, EXCL_MIN_N = 0.02, 0.05, 200
CHANGED_EXPECT = (0.013, 0.022)                 # expected share of state rows changed by R (warning only)
OUT_DIR = '/kaggle/working'
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
BASE_PATH = glob.glob(f'/kaggle/input/**/{BASE_KERNEL}/**/{BASE_FILE}', recursive=True)[0]
print('SMOKE', SMOKE, '| DO_RECON', DO_RECON, '| DO_TV', DO_TV, '| base', BASE_PATH, flush=True)
assert DO_RECON or DO_TV, 'nothing to do'

base = pd.read_csv(BASE_PATH)
key = pd.read_csv(f'{REL}/submission_key.csv', dtype=str)
iss = (base.task == 'state').to_numpy(); sid = base.loc[iss, 'submission_id'].to_numpy().astype('int64')
kcols = [c for c in ('submission_id', 'panel', 'timestamp', 'station_id', 'link_id', 'mask_regime', 'split') if c in key.columns]
ks = key[key.task == 'state'][kcols].copy()
ks['submission_id'] = ks.submission_id.astype('int64')
assert ks.submission_id.is_unique and len(ks) == len(sid), 'state rows of base and key differ'
ks = ks.set_index('submission_id').loc[sid].reset_index()              # row order = base state rows (positional index 0..N-1)
ks['station_id'] = ks.station_id.astype(str); ks['link_id'] = ks.link_id.astype(str)
ts = pd.to_datetime(ks.timestamp, utc=True)
tsns = ts.dt.tz_convert(None).to_numpy('datetime64[ns]').astype('int64')
mon = ts.dt.strftime('%Y-%m').to_numpy()
month_split = np.where(mon == '2031-03', 'validation', np.where(mon == '2031-04', 'private', 'train'))
SPLIT_MAP = {'validation': 'validation', 'val': 'validation', 'public': 'validation', 'private': 'private', 'test': 'private', 'train': 'train'}
if 'split' in ks.columns:
    sp = ks.split.astype(str).str.strip().str.lower().map(SPLIT_MAP)
    print('split from key column | unmapped', int(sp.isna().sum()), '| agreement with timestamp month', round(float((sp.fillna('?').to_numpy() == month_split).mean()), 5))
    ks['split'] = sp.fillna(pd.Series(month_split, index=ks.index)).to_numpy()
else:
    print('split from timestamp month (no split column in submission_key)')
    ks['split'] = month_split
if 'mask_regime' in ks.columns:
    ks['mask_regime'] = ks.mask_regime.astype(str).str.replace('mask_regime=', '', regex=False)
    print('mask_regime from key:', ks.mask_regime.value_counts().to_dict())
print('state rows per split:', ks.split.value_counts().to_dict(), flush=True)
for (panel, split) in ks[['panel', 'split']].drop_duplicates().itertuples(index=False):
    assert glob.glob(f'{REL}/corridors/{panel}/{split}/mainline_states_masked'), f'no released dir for {panel}/{split}'
PANELS = sorted(ks.panel.unique())


# ---------------- FD parameters per (panel, link_id): average the station rows per link, as the scorer does
def fd_params(panel):
    f = pd.read_csv(f'{REL}/corridors/{panel}/network/fd_parameters.csv', dtype={'link_id': str})
    have = [c for c in ('free_speed_kmh', 'capacity_vph', 'lanes', 'k_jam') if c in f.columns]
    assert {'free_speed_kmh', 'capacity_vph', 'lanes'} <= set(have), f'{panel}: fd_parameters.csv lacks columns, has {list(f.columns)}'
    m = f.groupby('link_id', as_index=False)[have].mean()
    m = m.merge(f.groupby('link_id').size().rename('n_rows').reset_index(), on='link_id')
    m['lanes'] = m.lanes.clip(lower=1.0)
    m['critical_density'] = m.capacity_vph / m.free_speed_kmh.clip(lower=1)    # recomputed from the averages, never averaged
    if 'k_jam' not in m.columns:
        m['k_jam'] = np.nan
    m['k_jam'] = m.k_jam.fillna(100.0 * m.lanes)
    m['k_jam'] = np.maximum(m.k_jam, 1.05 * m.critical_density)
    m['capl'] = m.capacity_vph / m.lanes; m['kcl'] = m.critical_density / m.lanes; m['kjl'] = m.k_jam / m.lanes
    m['w'] = m.capl / np.maximum(m.kjl - m.kcl, 1e-6)
    m['panel'] = panel
    return m


FD = pd.concat([fd_params(p) for p in PANELS], ignore_index=True)
unit_fail = []
for panel in PANELS:
    used = set(ks.loc[ks.panel == panel, 'link_id'])
    m = FD[(FD.panel == panel) & FD.link_id.isin(used)]
    print(f'{panel:12s} FD links used {len(m):3d}/{len(used):3d} (multi-station links {int((m.n_rows > 1).sum())}) | per lane: '
          f'cap [{m.capl.min():.0f},{m.capl.max():.0f}] kc [{m.kcl.min():.1f},{m.kcl.max():.1f}] kj [{m.kjl.min():.1f},{m.kjl.max():.1f}] '
          f'| w [{m.w.min():.1f},{m.w.max():.1f}] km/h | vf [{m.free_speed_kmh.min():.1f},{m.free_speed_kmh.max():.1f}]', flush=True)
    bw = m[(m.w < W_RANGE[0]) | (m.w > W_RANGE[1]) | ~np.isfinite(m.w)]
    bk = m[(m.kjl < KJL_RANGE[0]) | (m.kjl > KJL_RANGE[1]) | ~np.isfinite(m.kjl)]
    if len(bw):
        unit_fail.append(f'{panel}: w outside {W_RANGE} km/h on links {bw.link_id.tolist()[:10]} (values {bw.w.round(2).tolist()[:10]})')
    if len(bk):
        unit_fail.append(f'{panel}: kj/lane outside {KJL_RANGE} veh/km on links {bk.link_id.tolist()[:10]} (values {bk.kjl.round(1).tolist()[:10]})')
# links with implausible FD parameters (e.g. per-lane capacity ~7,700 vph -> w > 200 km/h) are EXCLUDED link-wise (left untouched);
# a panel where more than 5% of its links fail still aborts the run.
bad = ((FD.w < W_RANGE[0]) | (FD.w > W_RANGE[1]) | ~np.isfinite(FD.w) | (FD.kjl < KJL_RANGE[0]) | (FD.kjl > KJL_RANGE[1]) | ~np.isfinite(FD.kjl))
share = bad.groupby(FD.panel).mean()
print('links excluded for implausible FD parameters:', int(bad.sum()), '| per panel share:', share[share > 0].round(3).to_dict())
for line in unit_fail:
    print('  excluded ->', line)
if (share > 0.05).any():
    raise RuntimeError('FD UNIT CHECK FAILED (>5% of links in a panel) - not writing any submission: ' + str(share[share > 0.05].to_dict()))
FD = FD[~bad].copy()
print('FD unit checks passed after link-wise exclusion: w in', W_RANGE, 'km/h, kj/lane in', KJL_RANGE, 'veh/km')

p = ks[['panel', 'link_id']].merge(FD, on=['panel', 'link_id'], how='left', validate='many_to_one')
hasfd = p.lanes.notna().to_numpy()
print('state rows without FD parameters (left untouched):', int((~hasfd).sum()))
ln = p.lanes.fillna(1.0).to_numpy(float); vf = p.free_speed_kmh.fillna(1e9).to_numpy(float)
kjl = p.kjl.fillna(0.0).to_numpy(float); w = p.w.fillna(1.0).to_numpy(float)


def kfd_of(v, w_, kj_):
    return w_ * kj_ / (np.clip(v, 1, None) + w_)


# ---------------- released eligible cells per (panel, split) (diagnostic + TV anchors)
REL_CACHE = {}


def released(panel, split):
    if (panel, split) not in REL_CACHE:
        parts = []
        for f in sorted(glob.glob(f'{REL}/corridors/{panel}/{split}/mainline_states_masked/*/*.parquet')):
            d = pd.read_parquet(f, columns=['timestamp', 'station_id', 'link_id', 'speed_kmh', 'flow_vph', 'is_score_eligible'])
            d = d[(d.is_score_eligible == 1) & d.speed_kmh.notna() & d.flow_vph.notna()]
            parts.append(d[['timestamp', 'station_id', 'link_id', 'speed_kmh', 'flow_vph']])
        if parts:
            d = pd.concat(parts, ignore_index=True)
            d['station_id'] = d.station_id.astype(str); d['link_id'] = d.link_id.astype(str)
            d['ts'] = pd.to_datetime(d.timestamp, utc=True).dt.tz_convert(None).to_numpy('datetime64[ns]').astype('int64')
            d = d.drop(columns='timestamp').merge(FD.loc[FD.panel == panel, ['link_id', 'lanes', 'free_speed_kmh', 'kjl', 'w']], on='link_id', how='left')
            d = d[d.lanes.notna()].drop_duplicates(['station_id', 'ts'])
            d['k'] = d.flow_vph / d.lanes / d.speed_kmh.clip(lower=1)
        else:
            d = pd.DataFrame(columns=['station_id', 'link_id', 'speed_kmh', 'flow_vph', 'ts', 'lanes', 'free_speed_kmh', 'kjl', 'w', 'k'])
        REL_CACHE[(panel, split)] = d
    return REL_CACHE[(panel, split)]


# ---------------- diagnostic + exclusion rule (fixed before any LB result)
EXCLUDED = []
print(f'diagnostic on released eligible congested cells (v < {GATE} vf): rel = (k_obs - kFD(v_obs)) / kFD(v_obs)')
for panel in PANELS:
    pooled = []
    for split in ('validation', 'private'):
        d = released(panel, split)
        if len(d) == 0:
            print(f'  {panel:12s} {split:10s} no released cells'); continue
        g = d.speed_kmh.to_numpy() < GATE * d.free_speed_kmh.to_numpy()
        kf = kfd_of(d.speed_kmh.to_numpy(float), d.w.to_numpy(float), d.kjl.to_numpy(float))
        rel = (d.k.to_numpy(float)[g] - kf[g]) / kf[g]
        pooled.append(rel)
        if len(rel):
            print(f'  {panel:12s} {split:10s} cells {len(d):7d} gate share {g.mean():.4f} n_gate {len(rel):6d} | median {np.median(rel):+.4f} '
                  f'IQR [{np.percentile(rel, 25):+.4f}, {np.percentile(rel, 75):+.4f}]')
        else:
            print(f'  {panel:12s} {split:10s} cells {len(d):7d} gate share 0')
    rel = np.concatenate(pooled) if pooled else np.zeros(0)
    if len(rel) < EXCL_MIN_N:
        EXCLUDED.append(panel); why = f'n_gate {len(rel)} < {EXCL_MIN_N}'
    else:
        med, q25, q75 = np.median(rel), np.percentile(rel, 25), np.percentile(rel, 75)
        bad = abs(med) > EXCL_MED or q25 < -EXCL_IQR or q75 > EXCL_IQR
        why = f'pooled n {len(rel)} median {med:+.4f} IQR [{q25:+.4f}, {q75:+.4f}]'
        if bad:
            EXCLUDED.append(panel)
    print(f'  {panel:12s} -> {"EXCLUDED" if panel in EXCLUDED else "kept"} ({why})', flush=True)
print('EXCLUDED panels:', EXCLUDED)
APPLY = hasfd & ~ks.panel.isin(EXCLUDED).to_numpy()
print('state rows eligible for R/T:', int(APPLY.sum()), '/', len(APPLY))
if not APPLY.any():
    print('WARNING: every panel excluded (or no FD) - all outputs equal the base; do NOT submit them')

v0 = base.loc[iss, 'speed_kmh'].to_numpy(float); q0 = base.loc[iss, 'flow_vph'].to_numpy(float)
vf_ok = APPLY & np.isfinite(v0) & np.isfinite(q0)


# ---------------- step R: FD-gated reconciliation
def recon(v, q):
    g = vf_ok & (v < GATE * vf)
    k = q / ln / np.clip(v, 1, None)
    r = np.where(g, kfd_of(v, w, kjl) / np.maximum(k, 1e-9), 1.0)
    r = np.exp(np.clip(np.log(np.maximum(r, 1e-9)), -LOGR_MAX, LOGR_MAX))
    vn = np.where(g, np.clip(v * r ** (-A), 3, 1.05 * vf), v)
    qn = np.where(g, np.maximum(q * r ** (1 - A), 20.0), q)
    assert (vn[~g] == v[~g]).all() and (qn[~g] == q[~g]).all(), 'R changed a non-gate row'
    print(f'R: gate share {g.mean():.4f} | median |log r| in gate {np.median(np.abs(np.log(r[g]))) if g.any() else 0:.4f}', flush=True)
    return vn, qn


# ---------------- step T: TV inside temporal runs of target cells
def tv_solve(kk, aL, aR, lam, lama, iters):
    R, n = kk.shape
    hasL, hasR = np.isfinite(aL), np.isfinite(aR)
    aL0, aR0 = np.where(hasL, aL, 0.0), np.where(hasR, aR, 0.0)
    lamL, lamR = np.where(hasL, lama, 0.0), np.where(hasR, lama, 0.0)
    u = np.zeros((R, max(n - 1, 0))); uL = np.zeros(R); uR = np.zeros(R)

    def prim():
        x = kk.copy()
        if n > 1:
            x[:, :-1] += u; x[:, 1:] -= u
        x[:, 0] -= uL; x[:, -1] -= uR
        return x
    for _ in range(iters):
        x = prim()
        if n > 1:
            u = np.clip(u + 0.2 * (x[:, 1:] - x[:, :-1]), -lam[:, None], lam[:, None])
        uL = np.clip(uL + 0.2 * (x[:, 0] - aL0), -lamL, lamL); uR = np.clip(uR + 0.2 * (x[:, -1] - aR0), -lamR, lamR)
    return prim()


def tv(v, q, tag):
    vn, qn = v.copy(), q.copy()
    K = q / ln / np.clip(v, 1, None)
    G = v < GATE * vf
    for (panel, split), grp in ks[vf_ok].groupby(['panel', 'split'], sort=True):
        o = grp.assign(_t=tsns[grp.index.to_numpy()]).sort_values(['station_id', '_t'], kind='mergesort')
        rows = o.index.to_numpy(); st = o.station_id.to_numpy(); t = o._t.to_numpy()
        same = st[1:] == st[:-1]
        assert not (same & (t[1:] == t[:-1])).any(), f'{panel}/{split}: duplicate (station, timestamp) target rows'
        new = np.ones(len(rows), bool); new[1:] = ~same | (t[1:] - t[:-1] > RUN_GAP_NS)
        starts = np.flatnonzero(new); lens = np.diff(np.append(starts, len(rows)))
        rel = released(panel, split)
        kobs = pd.Series(rel.k.to_numpy(float), index=pd.MultiIndex.from_arrays([rel.station_id.to_numpy(), rel.ts.to_numpy()])) if len(rel) else None

        def anchor(stn, tt):
            if kobs is None:
                return np.full(len(stn), np.nan)
            return kobs.reindex(pd.MultiIndex.from_arrays([stn, tt])).to_numpy(float)
        aLall = anchor(st[starts], t[starts] - RUN_GAP_NS)
        aRall = anchor(st[starts + lens - 1], t[starts + lens - 1] + RUN_GAP_NS)
        nchg = 0
        for n in np.unique(lens):
            sel = lens == n
            idx = rows[starts[sel][:, None] + np.arange(n)[None, :]]
            kk = K[idx]; gg = G[idx]
            mk = np.maximum(kk.mean(1), 1e-6)
            xf = tv_solve(kk, aLall[sel], aRall[sel], TV_FREE * mk, TV_FREE_A * mk, TV_ITERS)
            xg = tv_solve(kk, aLall[sel], aRall[sel], TV_GATE * mk, TV_GATE_A * mk, TV_ITERS)
            kn = np.where(gg, xg, xf)
            rr = np.exp(np.clip(np.log(np.maximum(kn, 1e-9) / np.maximum(kk, 1e-9)), -LOGR_MAX, LOGR_MAX))
            a = np.where(gg, A, 0.0)
            chg = rr != 1.0
            vv, qq = v[idx], q[idx]
            vn[idx] = np.where(chg & gg, np.clip(vv * rr ** (-a), 3, 1.05 * vf[idx]), vv)
            qn[idx] = np.where(chg, np.maximum(qq * rr ** (1 - a), 20.0), qq)
            nchg += int(chg.sum())
        print(f'T[{tag}] {panel:12s} {split:10s} rows {len(rows):7d} runs {len(starts):6d} (max len {lens.max()}) | anchors L {np.isfinite(aLall).mean():.3f} '
              f'R {np.isfinite(aRall).mean():.3f} | rows changed {nchg} | rms dq {np.sqrt(np.mean((qn[rows] - q[rows]) ** 2)):.3f}', flush=True)
    assert (vn[~vf_ok] == v[~vf_ok]).all() and (qn[~vf_ok] == q[~vf_ok]).all(), 'T changed a row outside the applied panels'
    return vn, qn


# ---------------- write + hard gates (swap_combo15 asserts + byte-identical queue/path controls)
sample = pd.read_csv(f'{REL}/sample_submission.csv', usecols=['submission_id']).submission_id.to_numpy()
isq = (base.task == 'queue').to_numpy()
assert np.array_equal(base.queue_pred.to_numpy(), np.round(base.queue_pred.to_numpy())), 'base queue_pred not integer'
flow_low_base = float((q0 < 50).mean())


def write(vn, qn, name):
    chg = (vn != v0) | (qn != q0)
    dv, dq = np.abs(vn - v0)[chg], np.abs(qn - q0)[chg]
    share = float(chg.mean())
    print(f'{name}: state rows changed {share:.4%} ({int(chg.sum())}) | median |dv| {np.median(dv) if len(dv) else 0:.3f} km/h '
          f'| median |dq| {np.median(dq) if len(dq) else 0:.2f} vph | max |dv| {dv.max() if len(dv) else 0:.2f} | max |dq| {dq.max() if len(dq) else 0:.1f}')
    if name == 'FDR' and not (CHANGED_EXPECT[0] <= share <= CHANGED_EXPECT[1]):
        print(f'WARNING: FDR changed share {share:.4%} outside expected {CHANGED_EXPECT}')
    for pn in PANELS:
        m = (ks.panel == pn).to_numpy()
        print(f'   {pn:12s} changed {chg[m].mean():.4%}', end='')
    print()
    out = base.copy()
    out.loc[iss, 'speed_kmh'] = vn; out.loc[iss, 'flow_vph'] = qn
    assert (out.loc[isq, 'queue_pred'].to_numpy() == base.loc[isq, 'queue_pred'].to_numpy()).all()
    assert np.array_equal(out.queue_pred.to_numpy(), base.queue_pred.to_numpy()), 'queue_pred differs from base'
    assert np.array_equal(out.path_flow.to_numpy(), base.path_flow.to_numpy(), equal_nan=True), 'path_flow differs from base'
    assert np.array_equal(out.loc[~iss, ['speed_kmh', 'flow_vph']].to_numpy(), base.loc[~iss, ['speed_kmh', 'flow_vph']].to_numpy(), equal_nan=True)
    assert len(out) == len(sample) and (out.submission_id.to_numpy() == sample).all()
    assert np.isfinite(out[['speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']].to_numpy()).all()
    assert (out[['speed_kmh', 'flow_vph', 'path_flow']] >= 0).all().all()
    assert float((out.loc[iss, 'flow_vph'] < 50).mean()) <= flow_low_base + 0.001
    out['queue_pred'] = out.queue_pred.astype(int)
    path = f'{OUT_DIR}/{PREFIX}_{name}.csv'
    out.to_csv(path, index=False)
    chk = pd.read_csv(path, usecols=['submission_id', 'queue_pred', 'path_flow'])
    assert (chk.submission_id.to_numpy() == sample).all()
    assert np.array_equal(chk.queue_pred.to_numpy(), base.queue_pred.to_numpy().astype(int)), 'written queue_pred differs from base'
    assert np.array_equal(chk.path_flow.to_numpy(), base.path_flow.to_numpy(), equal_nan=True), 'written path_flow differs from base'
    print(path, 'OK', len(out), flush=True)


if DO_RECON:
    v1, q1 = recon(v0, q0)
    write(v1, q1, 'FDR')
    if DO_TV:
        v2, q2 = tv(v1, q1, 'FDRTV')
        write(v2, q2, 'FDRTV')
if DO_TV:
    v3, q3 = tv(v0, q0, 'TV')
    write(v3, q3, 'TV')
print('DONE | excluded', EXCLUDED)
