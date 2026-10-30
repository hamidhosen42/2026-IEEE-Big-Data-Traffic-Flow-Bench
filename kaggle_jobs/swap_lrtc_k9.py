"""K9 = K7 with the round-5 post-hoc Task 1 lead 'r2only + blkVQ@0.15' (EXP_T1_R5 / EXP_T1_K9_PREP, prepared 2026-10-30, NOT pushed).
Base = tfb-k5d (unchanged, the same base as tfb-k7), so the only changes against tfb-k7 are the step-M ranks and the new step B.
Lab: analysis/t1lab/t1div5_lrtc.py + t1div5_report.py posthoc=1 (arms_of: r2only = ordq, blkVQ = blkvq) on stand-in S4 (= tfb-k7);
POST-HOC (written after the round-5 gate result, NOT gate-eligible): r2only + blkVQ@0.15 dJ +0.00037 [+0.00030, +0.00045] 8/8 vs S4
(val +0.00031, pri +0.00044; ord scope +0.00018, blk scope +0.00019). Chosen over stkQ@0.35 + blkVQ@0.15 (+0.00040) before this
kernel because it is simpler (no 2 % self-validation stack). Expected LB ~ +0.0002 (lab x ~0.6; at the public noise floor).
Step M (changed): as K7 but the flow LRTC ranks are (24, 40, 10) (= lrtc2r; pilot lrtc5_pilot.json, self-val flow RMSE 29.2 vs 33.0
  vph/lane for (8, 40, 10)); ordinary rows q <- max(0.65 q + 0.35 max(q_lrtc2r, 20), 20) (the lab arm's floors); speed and blackout
  rows untouched.
Step B (new, lab arm blkVQ@0.15): blackout rows (= the tfb-k5d override rows, so B acts after the override; ordinary rows untouched):
  v <- clip(0.85 v + 0.15 v_lrtc, 3, 1.05 vf), q <- max(0.85 q + 0.15 q_lrtc, 20), with the ROUND-4 members of the same month:
  v_lrtc = speed LRTC ranks (8, 20, 20) clipped to [3, 1.05 vf], q_lrtc = flow LRTC ranks (8, 40, 10) (= K7's step-M member) floored at
  20; vf = the link's free speed of fd_parameters.csv for every link (incl. the FD-excluded links; the lab clips everywhere).
Every LRTC: per (panel, split) released cube of its own channel (every finite value of mainline_states_masked = the lab view), each
station normalised by its own mean, 60 EM iterations (functions unchanged from K7 = t1div_lrtc.py). Steps R + T unchanged (verbatim).
Outputs submission_LRTC.csv (M + B, diagnostics) and submission_FDRTV.csv (M, B, then R, then T = the candidate).
Tests: analysis/t1lab/t1k9_parity.py (kernel functions on the lab views / arrays == lab), t1lrtc_k9_fntest.py (whole kernel).

(swap_lrtc_k7.py docstring follows)
K7 = LRTC flow member + FDRECON / TV swap (EXP_T1_DIV4, prepared 2026-10-30, NOT pushed). Base = tfb-k5d (the production T1 mix
incl. the blackout override, BEFORE recon / TV; the same base as tfb-k5), so the only change against tfb-k5 is step M.

Step M (new, lab analysis/t1lab/t1div_lrtc.py + t1div_report.py, PRE-REGISTERED gate in analysis/t1lab/t1div_lib.py ROUND 4):
per (panel, split) EM low-rank tensor completion of the released month's FLOW cube (days x 288 x stations; released cells = every
finite flow_vph of mainline_states_masked, eligible or not = exactly the lab view), each station normalised by its own mean,
ranks (day, time-of-day, station) = (8, 40, 10) (pilot: self-validation on released non-evaluation cells), 60 iterations:
Z = mean over the 3 unfoldings of the rank-r truncated-SVD projection, released cells re-imposed after every iteration.
On ORDINARY state rows (not in (origin, origin + 90 min] of an official Task 2 window = not a tfb-k5d blackout row):
q <- (1 - W_LRTC) q + W_LRTC q_lrtc, W_LRTC = 0.35; speed unchanged; blackout rows untouched.
Lab (4 panels x 2 months, stand-in S3 + recon + TV): deploy dJ +0.00054 [+0.00046, +0.00064] 8/8, minus control +0.00080;
ordinary-rows-only (= this kernel) +0.00054 [+0.00045, +0.00064] 8/8. Expected LB +0.00015..+0.00025 (members transferred ~0.3-0.45x).
Then steps R and T exactly as kaggle_jobs/swap_fdrecon_k5.py (code below copied verbatim; only the outputs differ):
outputs submission_LRTC.csv (M only, diagnostics) and submission_FDRTV.csv (M, then R, then T = the candidate).

(swap_fdrecon_k5.py docstring follows)
FDRECON / TV swap (Task 1/3 post-processing on the exact base file; no training). Derived from the local draft
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
import glob, os, re, numpy as np, pandas as pd

SMOKE = False                                   # True: diagnostics run on COMBO15, outputs prefixed smoke_ (never submit them)
DO_RECON, DO_TV = True, True
BASE_KERNEL, BASE_FILE = 'tfb-k5d', 'submission_K5D.csv'   # EXP_T1_XGB chain (was tfb-k3d)
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
W_LRTC = 0.35                                   # EXP_T1_DIV4 flow-only LRTC member weight (ordinary state rows)
LRTC_RANKS, LRTC_ITERS = (24, 40, 10), 60        # step M: lrtc2r (day, time-of-day, station) ranks (K7: (8, 40, 10)); EM iterations
W_BLK = 0.15                                    # step B weight on blackout rows (EXP_T1_W5 blkVQ@0.15)
BLK_V_RANKS, BLK_Q_RANKS = (8, 20, 20), (8, 40, 10)   # step B: round-4 speed / flow LRTC ranks (lrtc_pilot.json)
LRTC_MIN_OBS = 200                              # a station needs >= this many released cells of the channel in the split
WRITE_FDR, WRITE_TV = False, False              # only submission_LRTC.csv and submission_FDRTV.csv are written
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
VF_ALL = FD[['panel', 'link_id', 'free_speed_kmh']].copy()     # step B speed clip: every link's free speed (before the exclusion)
FD = FD[~bad].copy()
print('FD unit checks passed after link-wise exclusion: w in', W_RANGE, 'km/h, kj/lane in', KJL_RANGE, 'veh/km')

p = ks[['panel', 'link_id']].merge(FD, on=['panel', 'link_id'], how='left', validate='many_to_one')
hasfd = p.lanes.notna().to_numpy()
print('state rows without FD parameters (left untouched):', int((~hasfd).sum()))
ln = p.lanes.fillna(1.0).to_numpy(float); vf = p.free_speed_kmh.fillna(1e9).to_numpy(float)
kjl = p.kjl.fillna(0.0).to_numpy(float); w = p.w.fillna(1.0).to_numpy(float)
vfc = ks[['panel', 'link_id']].merge(VF_ALL, on=['panel', 'link_id'], how='left', validate='many_to_one').free_speed_kmh.to_numpy(float)
print('step B speed clip: state rows without a free speed (no upper clip)', int((~np.isfinite(vfc)).sum()))
vfc = np.where(np.isfinite(vfc), vfc, 1e9)


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


# ---------------- steps M + B (EXP_T1_R5): low-rank tensor completion members (functions = analysis/t1lab/t1div_lrtc.py, as K7)
def lr_unfold(X, i):
    return np.moveaxis(X, i, 0).reshape(X.shape[i], -1)


def lr_refold(M_, i, shape):
    rest = [shape[j] for j in range(3) if j != i]
    return np.moveaxis(M_.reshape([shape[i]] + rest), 0, i)


def lr_proj(M_, r):
    """rank-r truncated-SVD projection via the smaller Gram matrix."""
    m, n = M_.shape
    r = min(r, m, n)
    if m <= n:
        _, U = np.linalg.eigh(M_ @ M_.T); U = U[:, -r:]
        return U @ (U.T @ M_)
    _, V = np.linalg.eigh(M_.T @ M_); V = V[:, -r:]
    return (M_ @ V) @ V.T


def lr_init(Y):
    ok = np.isfinite(Y)
    with np.errstate(invalid='ignore', divide='ignore'):
        m_ts = np.nansum(Y, 0) / ok.sum(0)
        m_s = np.nansum(Y, (0, 1)) / ok.sum((0, 1))
    m_s = np.where(np.isfinite(m_s), m_s, 1.0)
    m_ts = np.where(np.isfinite(m_ts), m_ts, m_s[None, :])
    return np.where(ok, Y, m_ts[None, :, :])


def lrtc(Y, ranks, iters):
    """Y (days, 288, stations) normalised, NaN = missing -> Z, the low-rank estimate of every cell."""
    ok = np.isfinite(Y); X = lr_init(Y).astype(np.float64); Y0 = np.where(ok, Y, 0.0)
    Z = X
    for _ in range(iters):
        Z = sum(lr_refold(lr_proj(lr_unfold(X, i), ranks[i]), i, X.shape) for i in range(3)) / 3.0
        X = np.where(ok, Y0, Z)
    return Z


def lrtc_member(C, ranks):
    """released cube C (days, 288, stations), NaN = not released -> completed cube in C's units; each station normalised by its own
    mean over the released cells, fallback the mean of the station means (= t1div_lrtc.norm_view + lrtc, K7's lrtc_month)."""
    with np.errstate(invalid='ignore'):
        mu = np.nanmean(C, (0, 1))
    mu = np.where(np.isfinite(mu), mu, np.nanmean(mu))
    with np.errstate(all='ignore'):
        return lrtc(C / mu[None, None, :], ranks, LRTC_ITERS) * mu[None, None, :]


def member_values(Zv, Zq, Zq2, ix, vf_):
    """member values at the cube cells ix = (day, slot, station) with the lab's storage clips (t1div_lrtc.py / t1div5_lrtc.py):
    speed (8, 20, 20) in [3, 1.05 vf], flows (8, 40, 10) / (24, 40, 10) >= 20."""
    return np.clip(Zv[ix], 3, 1.05 * vf_), np.maximum(Zq[ix], 20.0), np.maximum(Zq2[ix], 20.0)


def blend_m(q, qm, use):
    """step M at rows `use` (lab arm r2only = t1div5_report.arms_of ordq): q <- max((1 - W_LRTC) q + W_LRTC q_lrtc2r, 20)."""
    with np.errstate(invalid='ignore'):
        return np.where(use, np.maximum((1 - W_LRTC) * q + W_LRTC * qm, 20.0), q)


def blend_b(v, q, vb, qb, use, vf_):
    """step B at rows `use` (lab arm blkVQ@0.15 = t1div5_report.arms_of blkvq): both channels toward the round-4 members."""
    with np.errstate(invalid='ignore'):
        return (np.where(use, np.clip((1 - W_BLK) * v + W_BLK * vb, 3, 1.05 * vf_), v),
                np.where(use, np.maximum((1 - W_BLK) * q + W_BLK * qb, 20.0), q))


def lrtc_month(panel, split):
    """released speed / flow cubes of one (panel, split) -> (days, stations, Zv (8, 20, 20), Zq (8, 40, 10), Zq2 (24, 40, 10),
    released cells per station (speed, flow), released shares (speed, flow))."""
    files = sorted(glob.glob(f'{REL}/corridors/{panel}/{split}/mainline_states_masked/*/*.parquet'))
    assert files, f'{panel}/{split}: no mainline_states_masked files'
    d = pd.concat([pd.read_parquet(f, columns=['timestamp', 'station_id', 'speed_kmh', 'flow_vph']) for f in files], ignore_index=True)
    d['station_id'] = d.station_id.astype(str)
    t = pd.to_datetime(d.timestamp, utc=True)
    day = t.dt.strftime('%Y-%m-%d').to_numpy(); slot = (t.dt.hour * 12 + t.dt.minute // 5).to_numpy()
    days = np.unique(day); stations = np.unique(d.station_id.to_numpy())
    di = np.searchsorted(days, day); si = np.searchsorted(stations, d.station_id.to_numpy())
    assert not pd.DataFrame({'d': di, 't': slot, 's': si}).duplicated().any(), f'{panel}/{split}: duplicate (day, slot, station) cells'
    cube = {}
    for c in ('speed_kmh', 'flow_vph'):
        x = d[c].to_numpy(float); okx = np.isfinite(x)
        cube[c] = np.full((len(days), 288, len(stations)), np.nan)
        cube[c][di[okx], slot[okx], si[okx]] = x[okx]
    V, Q = cube['speed_kmh'], cube['flow_vph']
    Zv, Zq, Zq2 = lrtc_member(V, BLK_V_RANKS), lrtc_member(Q, BLK_Q_RANKS), lrtc_member(Q, LRTC_RANKS)
    assert np.isfinite(Zv).all() and np.isfinite(Zq).all() and np.isfinite(Zq2).all(), f'{panel}/{split}: non-finite LRTC output'
    return (days, stations, Zv, Zq, Zq2, np.isfinite(V).sum((0, 1)), np.isfinite(Q).sum((0, 1)), float(np.isfinite(V).mean()),
            float(np.isfinite(Q).mean()))


# blackout rows exactly as tfb-k5d (t1_div2_swap.py): timestamp in (origin, origin + 90 min] of an official window of the panel
inb = np.zeros(len(ks), bool)
for panel in PANELS:
    pm = (ks.panel == panel).to_numpy(); tsp = ts[pm]; hit = np.zeros(int(pm.sum()), bool)
    for split in ('validation', 'private'):
        wi_p = f'{REL}/task2/{panel}/{split}/window_index.csv'
        if os.path.exists(wi_p):
            for o in pd.to_datetime(pd.read_csv(wi_p).forecast_origin, utc=True):
                hit |= ((tsp > o) & (tsp <= o + pd.Timedelta(minutes=90))).to_numpy()
    inb[np.flatnonzero(pm)[hit]] = True
assert 0.005 < inb.mean() < 0.05, f'blackout share {inb.mean():.4f} outside the expected range'
qL = np.full(len(ks), np.nan); vB = qL.copy(); qB = qL.copy()     # step-M flow member (24, 40, 10); step-B speed / flow members
dstr = ts.dt.strftime('%Y-%m-%d').to_numpy(); slt = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(); stn = ks.station_id.to_numpy()
for (panel, split), grp in ks.groupby(['panel', 'split'], sort=True):
    days, stations, Zv, Zq, Zq2, nv, nq, shv, shq = lrtc_month(panel, split)
    r = grp.index.to_numpy()
    di = np.clip(np.searchsorted(days, dstr[r]), 0, len(days) - 1); si = np.clip(np.searchsorted(stations, stn[r]), 0, len(stations) - 1)
    ok = (days[di] == dstr[r]) & (stations[si] == stn[r])
    zv, zq, zq2 = member_values(Zv, Zq, Zq2, (di, slt[r], si), vfc[r])
    okq = ok & (nq[si] >= LRTC_MIN_OBS); okv = ok & (nv[si] >= LRTC_MIN_OBS)
    qL[r[okq]] = zq2[okq]; qB[r[okq]] = zq[okq]; vB[r[okv]] = zv[okv]
    u = okq & ~inb[r]; rb = r[okq & okv & inb[r]]
    print(f'M {panel:12s} {split:10s} days {len(days)} stations {len(stations)} released flow share {shq:.3f} | rows {len(r)} '
          f'mapped {okq.mean():.4f} ordinary used {int(u.sum())} | rms (q_lrtc - q)/lane {np.sqrt(np.mean(((qL[r[u]] - q0[r[u]]) / ln[r[u]]) ** 2)) if u.any() else 0:.2f}', flush=True)
    print(f'B {panel:12s} {split:10s} released speed share {shv:.3f} | blackout rows {int(inb[r].sum())} used {len(rb)} | rms (v_lrtc - v) '
          f'{np.sqrt(np.mean((vB[rb] - v0[rb]) ** 2)) if len(rb) else 0:.2f} km/h | rms (q_lrtc - q)/lane '
          f'{np.sqrt(np.mean(((qB[rb] - q0[rb]) / ln[rb]) ** 2)) if len(rb) else 0:.2f}', flush=True)
useM = np.isfinite(qL) & ~inb & np.isfinite(q0)
shareM = float(useM.sum() / max(int((~inb).sum()), 1))
rmsM = float(np.sqrt(np.mean(((qL - q0) / ln)[useM] ** 2))); corrM = float(np.corrcoef(qL[useM], q0[useM])[0, 1])
print(f'M: W_LRTC {W_LRTC} ranks {LRTC_RANKS} | ordinary rows blended {int(useM.sum())} share {shareM:.4f} | blackout rows untouched by M '
      f'{int(inb.sum())} | rms (q_lrtc - q)/lane {rmsM:.2f} corr {corrM:.5f}', flush=True)
assert shareM > 0.97, f'LRTC covers only {shareM:.4f} of the ordinary state rows'
assert rmsM < 150.0 and corrM > 0.97, f'LRTC member looks broken (rms/lane {rmsM:.1f}, corr {corrM:.4f})'
useB = inb & np.isfinite(vB) & np.isfinite(qB) & np.isfinite(v0) & np.isfinite(q0)
shareB = float(useB.sum() / max(int(inb.sum()), 1))
rmsBv = float(np.sqrt(np.mean((vB - v0)[useB] ** 2))); corrBv = float(np.corrcoef(vB[useB], v0[useB])[0, 1])
rmsBq = float(np.sqrt(np.mean(((qB - q0) / ln)[useB] ** 2))); corrBq = float(np.corrcoef(qB[useB], q0[useB])[0, 1])
print(f'B: W_BLK {W_BLK} ranks v {BLK_V_RANKS} q {BLK_Q_RANKS} | blackout rows blended {int(useB.sum())} share {shareB:.4f} | rms (v_lrtc - v) '
      f'{rmsBv:.2f} km/h corr {corrBv:.4f} | rms (q_lrtc - q)/lane {rmsBq:.2f} corr {corrBq:.4f}', flush=True)
assert shareB > 0.97, f'LRTC covers only {shareB:.4f} of the blackout state rows'
# lab blackout cells (16 panel-month-folds): speed rms 6-13 km/h corr 0.81-0.96, flow/lane rms 63-136 corr 0.47-0.85 -> broken only beyond
assert rmsBv < 30.0 and corrBv > 0.5 and rmsBq < 300.0 and corrBq > 0.3, f'step-B members look broken ({rmsBv:.1f} {corrBv:.3f} {rmsBq:.1f} {corrBq:.3f})'
vb0, qb0 = v0.copy(), q0.copy()                 # K5D base (before steps M and B)
q0 = blend_m(q0, qL, useM)
assert (q0[~useM] == qb0[~useM]).all(), 'M changed a row outside the ordinary blended rows'
print('M OK | rows changed', int((q0 != qb0).sum()), '| speed unchanged (step M touches flow only)', flush=True)
qm0 = q0.copy()
v0, q0 = blend_b(v0, q0, vB, qB, useB, vfc)
assert (v0[~useB] == vb0[~useB]).all() and (q0[~useB] == qm0[~useB]).all(), 'B changed a row outside the blackout rows'
print('B OK | rows changed: speed', int((v0 != vb0).sum()), 'flow', int((q0 != qm0).sum()), '| ordinary rows unchanged (step B touches blackout rows only)',
      flush=True)
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
flow_low_base = float((qb0 < 50).mean())                       # K5D base flows (before steps M and B)


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


write(v0, q0, 'LRTC')                           # steps M + B (diagnostics; 'changed' below is relative to this file)
if DO_RECON:
    v1, q1 = recon(v0, q0)
    if WRITE_FDR:
        write(v1, q1, 'FDR')
    if DO_TV:
        v2, q2 = tv(v1, q1, 'FDRTV')
        write(v2, q2, 'FDRTV')
if DO_TV and WRITE_TV:
    v3, q3 = tv(v0, q0, 'TV')
    write(v3, q3, 'TV')
print('DONE | excluded', EXCLUDED)
