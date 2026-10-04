"""v23 (track ogfeat) = v22 (F03 bag: np511 + id511 x seeds 0, 1) with EXTRA ONGOING FEATURES appended to both column sets.
New columns (all from the 60-min history, the masked slot-T value, static vectors and the ramp columns; nothing after T):
  OG_STATIC (per window x link), OG_STEP (step-dependent extrapolations), OG_ADD_PROD (production columns re-admitted to np / id).
Diff to t2_model_v22.py: (a) feature block og_* (copied from analysis/t2lab/ogfeat_lib.py), (b) make_rows / load_split append the new
columns, (c) the v20-style CUR rows are no longer built (they were only logged in v22), (d) column sets / variant names:
fs4new = mean of the 4 new-feature models (main), fs2new = np + id seed 0 (fallback if the time guard skips a model), fs_npnew.
CONTROL / footprint: fs4new vs tfb-t2v22 queue_submission_fs4.csv when that output is attached (agreement, Jaccard; not expected to be 1).
Outputs: queue_submission_<variant>.csv (onset rows 0), t2v23_probs.npz, t2v23_footprint.csv, t2v23_trainsets.csv.

v22 notes: v22 = v21 restricted to the selection-faithful F03 recipe, bagged: F03 np511 + id511 with seeds 0 and 1 (4 models).
Variants: fs2 (seed 0 np+id; must reproduce tfb-t2v21 fs2), fs4 (mean of the 4 models), fs_np. Footprints are vs fs2.
CONTROL: fs2 vs tfb-t2v21/queue_submission_fs2.csv when that output is attached (optional).

v21 notes: v21 = selection-faithful ONGOING training (Task 2). Derived from t2_model_v20.py: same data loading, features, ramp features,
keep mask, decode, template mapping. Onset rows are NOT produced by this kernel (written as 0; the swap keeps the base onset rows).

Why: the official selector evaluates the ongoing persistence cap (IoU <= 0.9) on the full true state, so official ongoing windows
are young / first-crossing queues, while v20's dense training windows use the observed state with missing = free and are dominated
by stable queues. Here the train-window SELECTION additionally uses a time-filled state (linear interpolation within the day over
interior gaps <= 36 slots; no spatial fill): a "faithful" ongoing candidate at T has history coverage >= 0.7, an established queue
in the observed history, a queued cell of the filled state in T+1..T+6 and IoU(filled state T+1..T+6, filled state at T repeated)
<= 0.9 over all filled cells. Run-start = faithful candidate whose previous slot is not a candidate.
The filled state uses train-split future values and is used ONLY to pick / weight TRAIN windows (offline window construction);
it is never a feature and is never computed for validation / private data. No validation or private labels are used anywhere.

Training sets (ongoing rows only, per-window weights via lgb.Dataset(weight=...)):
  CUR = v20 dense windows (unweighted)                     -> cur_np, cur_id (seed 0; = v20 og_big / og_id; control)
  F03 = run-starts (1.0) + other faithful candidates on the t % 3 grid (0.3, subsampled to MAX_OG per panel) -> fs_np, fs_id
  H   = CUR windows (0.3; run-starts among them 1.0) + the remaining run-starts (1.0)                        -> h_np
Outputs: queue_submission_<name>.csv for cur2 (mean cur np+id = v20 npid), fs_np, fs_id, fs2 (mean F03 np+id), h_np,
fsblend (0.5 * cur2 probs + 0.5 * fs2 probs, then decode); t2v23_probs.npz (ongoing probabilities per model);
t2v23_footprint.csv, t2v23_trainsets.csv (diagnostics).
"""
import glob, json, os, time, gc
import numpy as np
import pandas as pd
import lightgbm as lgb

T0 = time.time()
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
OUT = '/kaggle/working'
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']
SMOKE = False
if SMOKE:
    PANELS = ['D7_I10_E', 'D12_I5_S']
H, F, SPACING = 12, 6, 72
N_BIG = 40 if SMOKE else 1800               # trees of the 511-leaf ongoing models (v20: 1800)
MAX_OG = 300 if SMOKE else 3000             # dense ongoing windows per panel (CUR, and the non-run-start part of F03)
FILL_GAP, PERS_MAX = 36, 0.9                # faithful rule: time fill over interior gaps <= 36 slots, persistence cap 0.9
W_RS, W_DENSE = 1.0, 0.3                    # window weights: run-starts / everything else in F03 and H
TIME_LIMIT_S = 8.3 * 3600                   # a model is skipped if its predicted end exceeds this (Kaggle budget 9 h)
ONGOING = 'queue_ongoing'
OG_STATIC = ['c_q3', 'c_q6', 'c_q9', 'c_q12', 'c_d3', 'c_d6', 'c_d12', 'c_rel6', 'c_n0', 'c_new3', 'c_clr3', 'c_new6', 'c_clr6', 'c_first', 'c_peak', 'k_run', 'k_ntr', 'k_qf6', 'k_dvr3', 'k_dvr6', 'k_vrmax6', 'k_sdlo0', 'k_sdhi0', 'k_cs0', 'k_nq7_0', 'k_nqhi8_0', 'k_nqlo8_0', 'k_sdlo3', 'k_sdhi3', 'k_cs3', 'k_nq7_3', 'k_nqhi8_3', 'k_nqlo8_3', 'k_sdlo6', 'k_sdhi6', 'k_cs6', 'k_nq7_6', 'k_nqhi8_6', 'k_nqlo8_6', 'k_vlo3', 'k_vhi3', 'k_vlo6', 'k_vhi6', 'e_side', 'e_dist', 'e_size', 'e_rel', 'e_n3', 'e_n6', 'e_x3', 'e_x6', 'e_vr0', 'e_dvr3', 'e_dvr6', 'e_fr0', 'e_in_fr0', 'e_in_dfr3', 'e_out_fr0', 'e_out_dfr3', 'e_in_vr0', 'e_in_dvr3', 'e_out_vr0', 'e_out_dvr3', 'e_ramp_tail', 'e_ramp_head', 'e_off_head', 'e_ramp_in', 'e_surge_head']
OG_STEP = ['kx_lo3', 'kx_hi3', 'kx_lo6', 'kx_hi6', 'kx_min3']
OG_ADD_PROD = ['tod']          # production columns re-admitted to the np / id sets
CAPD = 30.0
BIG = 10 ** 6


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
    ON, OFF = ramp_cube(panel, 'train', li, days)
    return dict(panel=panel, links=links, li=li, cut=cut, vf=vf, cap=cap, lanes=lanes, length=length,
                days=days, dow=dow, V=V, Fq=Fq, E=E, Q=Q, ON=ON, OFF=OFF)


def ramp_cube(panel, split, li, days):
    """released on/off-ramp flows summed onto the attached mainline link: [D, 288, L], NaN = not observed / not attached."""
    net = f'{REL}/corridors/{panel}/network'
    D, L = len(days), len(li)
    ON = np.full((D, 288, L), np.nan, np.float32); OFF = ON.copy()
    amap = pd.read_csv(f'{net}/ramp_attachment_map.csv', dtype=str)
    amap = amap[amap.nearest_mainline_link_id.isin(li)]
    rl = dict(zip(amap.ramp_link_id, amap.nearest_mainline_link_id.map(li)))
    dpos = {dd: i for i, dd in enumerate(days)}
    for f in sorted(glob.glob(f'{REL}/corridors/{panel}/{split}/ramp_states/**/*.parquet', recursive=True)):
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
    return ON, OFF


def add_ramp(Fe, ON, OFF, d, t, cap):
    """ramp features from slots t-3..t-1 (strictly before the forecast origin T)."""
    capl = cap[None, :]
    with np.errstate(all='ignore'):
        on3 = np.nanmean(ON[d[:, None], t[:, None] + np.arange(-3, 0)[None]], 1) / capl
        off3 = np.nanmean(OFF[d[:, None], t[:, None] + np.arange(-3, 0)[None]], 1) / capl
        on12 = np.nanmean(ON[d[:, None], t[:, None] + np.arange(-H, -3)[None]], 1) / capl
    Fe['ramp_on3'] = on3; Fe['ramp_off3'] = off3; Fe['ramp_on_surge'] = on3 - on12
    for nm, a in (('on', on3), ('off', off3)):
        a0 = np.nan_to_num(a)
        Fe[f'ramp_{nm}_pm3'] = sum(shift_links(a0, k) for k in (-3, -2, -1, 1, 2, 3)).astype(np.float32)


# ============================================================================================ kernel-portable block
def og_ffill(a):
    out = a.copy()
    for i in range(1, a.shape[1]):
        m = np.isnan(out[:, i]); out[:, i][m] = out[:, i - 1][m]
    return out


def og_dist(q, direction):
    """distance in links to the nearest True at lower (-1) / higher (+1) index, the link itself included; BIG if none."""
    L = q.shape[-1]; idx = np.arange(L)
    if direction < 0:
        return idx - np.maximum.accumulate(np.where(q, idx, -BIG), axis=-1)
    return np.minimum.accumulate(np.where(q, idx, BIG)[..., ::-1], axis=-1)[..., ::-1] - idx


def og_close(q, gap=2):
    """fill runs of <= gap non-queued links that lie between two queued links."""
    return q | ((og_dist(q, -1) + og_dist(q, 1) - 1) <= gap)


def og_rsum(A, lo, hi):
    """sum and count of the finite values A[w, lo..hi] (inclusive, clipped to the corridor; empty when lo > hi)."""
    W, L = A.shape; ok = np.isfinite(A)
    cs = np.zeros((W, L + 1), np.float64); cs[:, 1:] = np.cumsum(np.where(ok, A, 0.0), 1)
    cn = np.zeros((W, L + 1), np.int64); cn[:, 1:] = np.cumsum(ok, 1)
    lo_ = np.clip(lo, 0, L); hi_ = np.maximum(np.clip(hi + 1, 0, L), lo_)
    return (np.take_along_axis(cs, hi_, 1) - np.take_along_axis(cs, lo_, 1),
            np.take_along_axis(cn, hi_, 1) - np.take_along_axis(cn, lo_, 1))


def og_rmean(A, lo, hi):
    s, n = og_rsum(A, lo, hi)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def og_edges(q):
    """closed queue state and edge distances (corridor ends count as free): qc, dql, dqh (to a queued link, BIG if none),
    dfl, dfh (to a non-queued link)."""
    L = q.shape[-1]; idx = np.arange(L)
    qc = og_close(q)
    dql, dqh = og_dist(qc, -1), og_dist(qc, 1)
    dfl = np.minimum(og_dist(~qc, -1), idx + 1); dfh = np.minimum(og_dist(~qc, 1), L - idx)
    return qc, dql, dqh, dfl, dfh


def og_extra(P, Vh, Fh, VT, Fe):
    """extra ongoing features; Vh, Fh [W, 12, L], VT [W, L] exactly as passed to window_features, Fe = its output after
    add_ramp (only ramp_on3 / ramp_off3 / ramp_on_surge are read).  Returns dict name -> [W, L] float32.
    Time index of the 13-slot array: 12 = T (masked-layer value), 12 - k = T - 5k minutes."""
    with np.errstate(all='ignore'):
        cut = P['cut']; W, _, L = Vh.shape; idx = np.arange(L)[None, :]
        vr = np.concatenate([Vh / cut[None, None, :], (VT / cut[None, :])[:, None, :]], 1)    # same dtype / ops as production vr, vT
        fr = Fh / P['cap'][None, None, :]
        vf = og_ffill(vr); vb = og_ffill(vf[:, ::-1])[:, ::-1]                 # forward fill, then the leading gap backwards
        frf = og_ffill(fr); frb = og_ffill(frf[:, ::-1])[:, ::-1]
        qraw = np.isfinite(vr) & (np.nan_to_num(vr, nan=9.0) <= 1.0)
        Q = np.nan_to_num(vb, nan=9.0) <= 1.0                                   # [W, 13, L] filled queue state
        X = {}
        bc = lambda v: np.repeat(np.asarray(v, np.float32)[:, None], L, 1)
        # ---------------- g1b corridor context
        sh = Q.mean(2)                                                          # [W, 13]
        X['c_q3'] = bc(sh[:, 9]); X['c_q6'] = bc(sh[:, 6]); X['c_q9'] = bc(sh[:, 3]); X['c_q12'] = bc(sh[:, 0])
        X['c_d3'] = bc(sh[:, 12] - sh[:, 9]); X['c_d6'] = bc(sh[:, 12] - sh[:, 6]); X['c_d12'] = bc(sh[:, 12] - sh[:, 0])
        X['c_rel6'] = bc((sh[:, 12] - sh[:, 6]) / (sh[:, 12] + sh[:, 6] + 1e-6))
        X['c_n0'] = bc(Q[:, 12].sum(1))
        X['c_new3'] = bc((Q[:, 12] & ~Q[:, 9]).sum(1)); X['c_clr3'] = bc((~Q[:, 12] & Q[:, 9]).sum(1))
        X['c_new6'] = bc((Q[:, 12] & ~Q[:, 6]).sum(1)); X['c_clr6'] = bc((~Q[:, 12] & Q[:, 6]).sum(1))
        anyq = qraw.any(2)
        X['c_first'] = bc(np.where(anyq.any(1), 12 - np.argmax(anyq, 1), -1))  # slots since the first visible queued cell
        X['c_peak'] = bc(sh[:, 12] / (sh.max(1) + 1e-6))
        # ---------------- g2 per-link kinematics
        same = (Q == Q[:, 12:13])[:, ::-1]
        X['k_run'] = np.where(same.all(1), 13, np.argmin(same, 1))              # slots the link has been in its current state
        X['k_ntr'] = (Q[:, 1:] != Q[:, :-1]).sum(1)
        X['k_qf6'] = Q[:, 7:].mean(1)
        X['k_dvr3'] = vb[:, 12] - vb[:, 9]; X['k_dvr6'] = vb[:, 12] - vb[:, 6]
        X['k_vrmax6'] = np.nanmax(np.where(np.isfinite(vb[:, 7:]), vb[:, 7:], -9.0), 1)
        X['k_vrmax6'] = np.where(X['k_vrmax6'] < -8, np.nan, X['k_vrmax6'])
        sd = {}; E0 = None
        for lag, tag in ((12, '0'), (9, '3'), (6, '6')):
            qc, dql, dqh, dfl, dfh = og_edges(Q[:, lag])
            if lag == 12:
                E0 = (qc, dql, dqh, dfl, dfh)
            sd['lo' + tag] = np.where(qc, np.minimum(dfl, CAPD), -np.minimum(dql, CAPD))   # > 0 inside: links to the tail edge
            sd['hi' + tag] = np.where(qc, np.minimum(dfh, CAPD), -np.minimum(dqh, CAPD))   # > 0 inside: links to the head edge
            X['k_sdlo' + tag] = sd['lo' + tag]; X['k_sdhi' + tag] = sd['hi' + tag]
            X['k_cs' + tag] = np.where(qc, np.minimum(dfl + dfh - 1, 3 * CAPD), 0)
            qf = Q[:, lag].astype(np.float32)
            X['k_nq7_' + tag] = og_rsum(qf, idx - 3, idx + 3)[0]
            X['k_nqhi8_' + tag] = og_rsum(qf, idx + 1, idx + 8)[0]
            X['k_nqlo8_' + tag] = og_rsum(qf, idx - 8, idx - 1)[0]
        X['k_vlo3'] = sd['lo0'] - sd['lo3']; X['k_vhi3'] = sd['hi0'] - sd['hi3']
        X['k_vlo6'] = sd['lo0'] - sd['lo6']; X['k_vhi6'] = sd['hi0'] - sd['hi6']
        # ---------------- g3 demand / supply at the ends of the reference cluster
        qc, dql, dqh, dfl, dfh = E0
        use_hi = ~qc & (dqh <= dql) & (dqh < BIG // 2)
        use_lo = ~qc & ~use_hi & (dql < BIG // 2)
        a_h = np.clip(idx + dqh, 0, L - 1); b_h = a_h + np.take_along_axis(dfh, a_h, 1) - 1
        b_l = np.clip(idx - dql, 0, L - 1); a_l = b_l - np.take_along_axis(dfl, b_l, 1) + 1
        A = np.where(qc, idx - dfl + 1, np.where(use_hi, a_h, a_l)); B = np.where(qc, idx + dfh - 1, np.where(use_hi, b_h, b_l))
        ok = qc | use_hi | use_lo
        size = (B - A + 1).astype(np.float32)
        nanify = lambda v: np.where(ok, v, np.nan)
        X['e_side'] = np.where(qc, 0, np.where(use_hi, 1, np.where(use_lo, -1, np.nan)))
        X['e_dist'] = np.where(qc, 0, np.minimum(np.minimum(dql, dqh), 2 * CAPD))
        X['e_size'] = nanify(np.minimum(size, 3 * CAPD))
        X['e_rel'] = np.where(qc, (idx - A) / np.maximum(size - 1, 1), np.nan)
        q12, q9, q6 = (Q[:, k].astype(np.float32) for k in (12, 9, 6))
        X['e_n3'] = nanify(og_rsum(q9, A, B)[0] / size); X['e_n6'] = nanify(og_rsum(q6, A, B)[0] / size)
        n_now = og_rsum(q12, A - 5, B + 5)[0]
        X['e_x3'] = nanify(og_rsum(q9, A - 5, B + 5)[0] - n_now); X['e_x6'] = nanify(og_rsum(q6, A - 5, B + 5)[0] - n_now)
        v12, v9, v6 = vb[:, 12], vb[:, 9], vb[:, 6]; f11, f8 = frb[:, 11], frb[:, 8]
        m0 = og_rmean(v12, A, B)
        X['e_vr0'] = nanify(m0); X['e_dvr3'] = nanify(m0 - og_rmean(v9, A, B)); X['e_dvr6'] = nanify(m0 - og_rmean(v6, A, B))
        X['e_fr0'] = nanify(og_rmean(f11, A, B))
        i0 = og_rmean(f11, A - 3, A - 1); o0 = og_rmean(f11, B + 1, B + 3)
        X['e_in_fr0'] = nanify(i0); X['e_in_dfr3'] = nanify(i0 - og_rmean(f8, A - 3, A - 1))
        X['e_out_fr0'] = nanify(o0); X['e_out_dfr3'] = nanify(o0 - og_rmean(f8, B + 1, B + 3))
        iv = og_rmean(v12, A - 3, A - 1); ov = og_rmean(v12, B + 1, B + 3)
        X['e_in_vr0'] = nanify(iv); X['e_in_dvr3'] = nanify(iv - og_rmean(v9, A - 3, A - 1))
        X['e_out_vr0'] = nanify(ov); X['e_out_dvr3'] = nanify(ov - og_rmean(v9, B + 1, B + 3))
        ron = np.asarray(Fe['ramp_on3'], np.float32); rof = np.asarray(Fe['ramp_off3'], np.float32)
        rsg = np.asarray(Fe['ramp_on_surge'], np.float32)
        X['e_ramp_tail'] = nanify(og_rsum(ron, A - 3, A + 1)[0]); X['e_ramp_head'] = nanify(og_rsum(ron, B - 1, B + 3)[0])
        X['e_off_head'] = nanify(og_rsum(rof, B - 1, B + 3)[0]); X['e_ramp_in'] = nanify(og_rsum(ron, A, B)[0])
        X['e_surge_head'] = nanify(og_rsum(rsg, B - 1, B + 3)[0])
    return {k: np.asarray(v, np.float32) for k, v in X.items()}


OG_STEP_ALL = ['kx_lo3', 'kx_hi3', 'kx_lo6', 'kx_hi6', 'kx_min3']


def og_step(S, h):
    """step-dependent columns from the static ones (S: name -> array) and the step h (array or scalar, 1..6)."""
    h = np.asarray(h, np.float32)
    lo3 = S['k_sdlo0'] + h / 3.0 * S['k_vlo3']; hi3 = S['k_sdhi0'] + h / 3.0 * S['k_vhi3']
    return {'kx_lo3': lo3, 'kx_hi3': hi3, 'kx_lo6': S['k_sdlo0'] + h / 6.0 * S['k_vlo6'], 'kx_hi6': S['k_sdhi0'] + h / 6.0 * S['k_vhi6'],
            'kx_min3': np.minimum(lo3, hi3)}
# ============================================================================================ end of portable block


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


# ------------------------------------------------------------------ faithful ongoing selection (TRAIN windows only; never a feature)
def fill_time(a, maxgap):
    """linear interpolation along time within each day for interior gaps <= maxgap slots ([D, 288, L])."""
    out = a.copy(); Dd, Tt, Ll = a.shape; idx = np.arange(Tt)
    for d in range(Dd):
        okd = np.isfinite(a[d])
        for l in np.where(okd.any(0) & ~okd.all(0))[0]:
            ok = okd[:, l]
            if ok.sum() < 2:
                continue
            y = a[d, :, l]
            yi = np.interp(idx, idx[ok], y[ok])
            prev = np.maximum.accumulate(np.where(ok, idx, -10 ** 6)); nxt = np.minimum.accumulate(np.where(ok, idx, 10 ** 6)[::-1])[::-1]
            m = (~ok) & (nxt - prev - 1 <= maxgap) & (prev >= 0) & (nxt < Tt)
            out[d, m, l] = yi[m]
    return out


def faithful_candidates(P):
    """(cand, pers): [D, 288] faithful ongoing candidate map and filled persistence IoU (NaN outside slots 12..281).
    Uses the train future (time-filled state): offline train-window construction only."""
    Q, E = P['Q'], P['E']; D, _, L = Q.shape
    TS = np.arange(H, 288 - F)
    VF = fill_time(P['V'], FILL_GAP); MF = np.isfinite(VF); QF = MF & (VF <= P['cut'][None, None, :])
    del VF
    cE = np.concatenate([np.zeros((D, 1), np.int64), np.cumsum(E.sum(2), 1)], 1)
    cov = (cE[:, TS] - cE[:, TS - H]) / float(H * L)
    cQ = np.concatenate([np.zeros((D, 1, L), np.int16), np.cumsum(Q, 1, dtype=np.int16)], 1)
    est = ((cQ[:, TS] - cQ[:, TS - H]) >= 2).any(2)
    aqf = np.concatenate([np.zeros((D, 1), np.int64), np.cumsum(QF.any(2), 1)], 1)
    hq = (aqf[:, TS + F + 1] - aqf[:, TS + 1]) > 0
    inter = np.zeros((D, len(TS)), np.int32); union = inter.copy()
    S = QF[:, H:288 - F]
    for h in range(1, F + 1):
        Th = QF[:, H + h:288 - F + h]; M = MF[:, H + h:288 - F + h]
        inter += (Th & S & M).sum(2); union += ((Th | S) & M).sum(2)
    pers_ts = np.where(union == 0, 1.0, inter / np.maximum(union, 1))
    cand = np.zeros((D, 288), bool); pers = np.full((D, 288), np.nan)
    cand[:, H:288 - F] = (cov >= 0.7) & est & hq & (pers_ts <= PERS_MAX)
    pers[:, H:288 - F] = pers_ts
    return cand, pers


def pq(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return 'n=0'
    q = np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9])
    return f'n={len(x)} q10/25/50/75/90 ' + '/'.join(f'{v:.3f}' for v in q) + f' share>0.9 {np.mean(x > 0.9):.3f} share>0.8 {np.mean(x > 0.8):.3f}'


def select_windows(P):
    """P['trainw'] = v20 training windows (onset + dense ongoing, identical draw); P['exw'] = run-starts + faithful t%3 rest."""
    tw = build_windows(P, global_spacing=False, dense_ongoing=True)
    rng = np.random.default_rng(0)
    og = [w for w in tw if w[2] == 'queue_ongoing']; on = [w for w in tw if w[2] == 'queue_onset']
    n_og_all = len(og)
    if len(og) > MAX_OG:
        og = [og[i] for i in sorted(rng.choice(len(og), MAX_OG, replace=False))]
    P['trainw'] = on + og
    cand, pers = faithful_candidates(P)
    prev = np.zeros_like(cand); prev[:, 1:] = cand[:, :-1]
    rs = cand & ~prev
    rest = cand & ~rs & (np.arange(288) % 3 == 0)[None, :]
    rd, rt = np.where(rs); fd, ft = np.where(rest); n_rest_all = len(fd)
    if n_rest_all > MAX_OG:
        k = np.sort(np.random.default_rng(1).choice(n_rest_all, MAX_OG, replace=False)); fd, ft = fd[k], ft[k]
    g = np.r_[rd * 288 + rt, fd * 288 + ft].astype(np.int64); isrs = np.r_[np.ones(len(rd), bool), np.zeros(len(fd), bool)]
    o = np.argsort(g, kind='stable'); g, isrs = g[o], isrs[o]
    cur_g = {w[0] * 288 + w[1] for w in og}
    P['exw'] = [(int(x // 288), int(x % 288), ONGOING) for x in g]
    P['ex_rs'] = isrs; P['ex_incur'] = np.array([int(x) in cur_g for x in g], bool)
    dc = np.array([w[0] for w in P['trainw']]); tc = np.array([w[1] for w in P['trainw']])
    isog = np.array([w[2] == ONGOING for w in P['trainw']])
    P['cur_rs'] = isog & rs[dc, tc]
    cur_faith = isog & cand[dc, tc]
    P['cand'], P['rs'], P['pers'] = cand, rs, pers
    st = dict(panel=P['panel'], days=len(P['days']), cur_onset=len(on), cur_og_all=n_og_all, cur_og=len(og),
              cur_og_faithful=int(cur_faith.sum()), cur_og_runstart=int(P['cur_rs'].sum()),
              faith_all_slots=int(cand.sum()), runstarts=len(rd), faith3_rest_all=n_rest_all, faith3_rest_used=len(fd),
              F03_windows=len(g), H_windows=len(og) + int((isrs & ~P['ex_incur']).sum()),
              runstarts_already_in_cur=int((isrs & P['ex_incur']).sum()))
    log(P['panel'], 'windows', json.dumps({k: v for k, v in st.items() if k != 'panel'}))
    log(P['panel'], 'filled persistence | CUR ongoing:', pq(pers[dc[isog], tc[isog]]))
    log(P['panel'], 'filled persistence | run-starts :', pq(pers[rd, rt]))
    log(P['panel'], 'filled persistence | F03 rest   :', pq(pers[fd, ft]))
    P['pers_sets'] = dict(cur=pers[dc[isog], tc[isog]], rs=pers[rd, rt], rest=pers[fd, ft])
    return st


def read_official(panel, P):
    wi = pd.read_csv(f'{REL}/task2/{panel}/train/window_index.csv')
    return [(P['days'].index(pd.Timestamp(r.forecast_origin).strftime('%Y-%m-%d')),
             pd.Timestamp(r.forecast_origin).hour * 12 + pd.Timestamp(r.forecast_origin).minute // 5, r.condition)
            for r in wi.itertuples() if pd.Timestamp(r.forecast_origin).strftime('%Y-%m-%d') in P['days']]


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
    add_ramp(Fe, P['ON'], P['OFF'], d, t, P['cap'])
    OGX = og_extra(P, Vh, Fh, VT, Fe); Fe.update({k: OGX[k] for k in OG_STATIC})          # v23: extra ongoing features
    W, L = qlast.shape
    rows = []
    for h in range(1, F + 1):
        slot = np.minimum(t + h, 287)
        feats = {k: v for k, v in Fe.items()}
        feats['h'] = np.full((W, L), h, np.float32)
        feats.update({k: v for k, v in og_step(OGX, h).items() if k in OG_STEP})
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
ID_DROP = {'prior', 'prior_now', 'prior_alldays', 'tod', 'dow'}


def predict_windows(R, models, cols):
    """probabilities [W, 6, L] for all windows in R."""
    P6 = np.zeros((R['W'], F, R['L']), np.float32)
    for cond, mdl in models.items():
        keep = ongoing_keep(R) if cond == 'queue_ongoing' else None
        X, _, _, meta, _ = to_matrix(R, cond, cols, keep)
        if len(X):
            P6[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = mdl.predict(X)
    return P6


# ------------------------------------------------------------------ training rows
def panel_rows(P, cols=None):
    """ongoing training rows of one panel: CUR (exactly as v20: same windows, same slot-T mask draw, same keep mask) and EX
    (run-starts + faithful rest). Returns cols and two dicts(X, y, rs[, incur])."""
    P['prior_all'] = prior_table(P, np.ones(len(P['days']), bool))
    R = make_rows(P, P['exw'], P['prior_all'])                              # v23: the v20-style CUR rows are not built
    X, y, e, meta, cols = to_matrix(R, ONGOING, cols, ongoing_keep(R))
    ex = dict(X=X[e], y=y[e], rs=P['ex_rs'][meta[:, 0]][e], incur=P['ex_incur'][meta[:, 0]][e])
    del R, X; gc.collect()
    return cols, ex


# ------------------------------------------------------------------ released windows (validation / private)
def load_split(P, panel, split):
    """features of the released windows exactly as v20: 60-min history, masked slot-T value, ramp slots < T. No labels."""
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
    wdays = sorted(wi.date.unique())
    ONs, OFFs = ramp_cube(panel, split, P['li'], wdays)
    add_ramp(Fe, ONs, OFFs, wi.date.map({dd: i for i, dd in enumerate(wdays)}).to_numpy(), tod, P['cap'])
    log(panel, split, 'ramp on3 finite share', round(float(np.isfinite(Fe['ramp_on3']).mean()), 3))
    OGX = og_extra(P, Vh, Fh, VT, Fe); Fe.update({k: OGX[k] for k in OG_STATIC})          # v23: extra ongoing features
    R = dict(d=np.zeros(W, int), t=tod, cond=wi.condition.to_numpy(), qlast=qlast, W=W, L=L, rows=[])
    for h in range(1, F + 1):
        feats = dict(Fe); feats['h'] = np.full((W, L), h, np.float32)
        feats.update({k: v for k, v in og_step(OGX, h).items() if k in OG_STEP})
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = prior_feats(P['prior_all'], dow, tod, h)
        R['rows'].append((h, feats, None, None))
    return wi, tpl, R


VARIANTS = {'fs2new': ('fs_np', 'fs_id'),                              # np + id seed 0 with the new features (fallback)
            'fs4new': ('fs_np', 'fs_id', 'fs_np1', 'fs_id1'),          # F03 bag of 4 with the new features (main)
            'fs_npnew': ('fs_np',)}


def predict_split(P, panel, split, models, foot, probs):
    """decoded masks for every available variant mapped onto the queue template; footprint rows vs cur2 appended to `foot`."""
    wi, tpl, R = load_split(P, panel, split)
    W, L = R['W'], R['L']
    is_og = (wi.condition == ONGOING).to_numpy()
    P6 = {name: predict_windows(R, {ONGOING: mdl}, cols) for name, (mdl, cols) in models.items()}
    for name, p in P6.items():
        probs[f'{panel}__{split}__{name}'] = p[is_og].astype(np.float16)
    probs[f'{panel}__{split}__window_id'] = wi.window_id.to_numpy()[is_og].astype(str)
    probs[f'{panel}__{split}__qlast'] = R['qlast'][is_og]
    preds = {}
    for v, parts in VARIANTS.items():
        if not all(m in P6 for m in parts):
            continue
        p = P6[parts[0]] if len(parts) == 1 else (1.0 / len(parts)) * sum([P6[m] for m in parts[1:]], P6[parts[0]])
        pr = np.zeros((W, F, L), bool)                                    # onset windows stay 0 (not produced by this kernel)
        for i in np.where(is_og)[0]:
            pr[i] = decode(p[i].ravel()).reshape(F, L)
        preds[v] = pr
    for v, pr in preds.items():
        for i in np.where(is_og)[0]:
            a, b = pr[i], preds['fs2new'][i]
            row = dict(panel=panel, split=split, variant=v, window_id=wi.window_id.iloc[i], changed=bool((a != b).any()),
                       jacc_cur2=float((a & b).sum() / max((a | b).sum(), 1)), size_T=int(R['qlast'][i].sum()))
            for h in range(F):
                row[f'chg_h{h + 1}'] = int((a[h] != b[h]).sum()); row[f'add_h{h + 1}'] = int((a[h] & ~b[h]).sum())
                row[f'size_h{h + 1}'] = int(a[h].sum())
            foot.append(row)
    wpos = {w: i for i, w in enumerate(wi.window_id)}
    org = pd.to_datetime(wi.set_index('window_id').forecast_origin, utc=True)
    tpl['w'] = tpl.window_id.map(wpos); tpl['l'] = tpl.link_id.map(P['li'])
    tts = pd.to_datetime(tpl.timestamp, utc=True)
    tpl['h'] = ((tts - tpl.window_id.map(org)).dt.total_seconds() // 300).astype(int)
    okr = tpl.w.notna() & tpl.l.notna() & tpl.h.between(1, F)
    for name, pr in preds.items():
        col = f'queue_pred_{name}'
        tpl[col] = 0
        tpl.loc[okr, col] = pr[tpl.w[okr].astype(int), tpl.h[okr].astype(int) - 1, tpl.l[okr].astype(int)].astype(int)
    tpl['is_og'] = tpl.window_id.map(dict(zip(wi.window_id, is_og))).fillna(False).astype(bool)
    log(panel, split, 'windows', W, 'ongoing', int(is_og.sum()), 'template rows', len(tpl), 'unmapped', int((~okr).sum()),
        'positive share (ongoing rows)', {n: round(float(tpl.loc[tpl.is_og, f'queue_pred_{n}'].mean()), 4) for n in preds})
    return tpl[['window_id', 'timestamp', 'link_id', 'is_og'] + [f'queue_pred_{n}' for n in preds]], list(preds)


def print_footprint(foot):
    """footprint of every variant against cur2 on the official ongoing windows (predictions only, no labels)."""
    fp = pd.DataFrame(foot)
    if fp.empty:
        return fp
    hs = range(1, F + 1)
    for split, g in fp.groupby('split'):
        log(f'FOOTPRINT {split}: ongoing windows {g[g.variant == "fs2new"].shape[0]} | mean queued cells at T (q_last) '
            f'{g[g.variant == "fs2new"].size_T.mean():.2f}')
        for v, x in g.groupby('variant', sort=False):
            log(f'  {v:8s} windows changed {int(x.changed.sum())}/{len(x)} | mean Jaccard vs fs2new {x.jacc_cur2.mean():.4f}'
                f' | cells changed per step h1..h6 {[int(x[f"chg_h{h}"].sum()) for h in hs]} (added {[int(x[f"add_h{h}"].sum()) for h in hs]})'
                f' | mean predicted size per step {[round(float(x[f"size_h{h}"].mean()), 2) for h in hs]}')
        pv = g.pivot_table(index='panel', columns='variant', values='changed', aggfunc='sum')
        log(f'  windows changed per panel ({split}):', pv.to_dict('index'))
    return fp


def control_check(S, names):
    """footprint of fs4new (fallback fs2new) against tfb-t2v22 queue_submission_fs4.csv on ongoing rows, if that output is attached.
    The models differ by design (new features), so this is a sanity range, not an identity: Jaccard well below 0.85 = suspicious."""
    ref = sorted(glob.glob('/kaggle/input/**/tfb-t2v22/**/queue_submission_fs4.csv', recursive=True))
    v = 'fs4new' if 'fs4new' in names else ('fs2new' if 'fs2new' in names else None)
    if not ref or v is None:
        log('CONTROL: skipped (tfb-t2v22 queue_submission_fs4.csv not attached or no variant)', ref)
        return None
    r = pd.read_csv(ref[0], dtype={'window_id': str, 'timestamp': str, 'link_id': str})
    m = S.merge(r.rename(columns={'queue_pred': 'ref'}), on=['window_id', 'timestamp', 'link_id'], how='left')
    og = m.is_og.to_numpy() & m.ref.notna().to_numpy()
    a = m[f'queue_pred_{v}'].to_numpy()[og].astype(int); b = m.ref.to_numpy()[og].astype(int)
    agree = float((a == b).mean()) if og.any() else float('nan')
    jac = float((a & b).sum() / max((a | b).sum(), 1))
    isval = m.window_id.astype(str).str.contains('val', case=False).to_numpy()[og]
    for nm, k in (('validation', isval), ('private', ~isval)):
        if k.any():
            log(f'CONTROL {v} vs v22 fs4 [{nm}]: ongoing rows {int(k.sum())} | changed {int((a[k] != b[k]).sum())} | Jaccard '
                f'{float((a[k] & b[k]).sum() / max((a[k] | b[k]).sum(), 1)):.5f} | positives new {int(a[k].sum())} ref {int(b[k].sum())}')
    log(f'CONTROL {v} vs {ref[0]}: ongoing rows {int(og.sum())} | agreement {agree:.5f} | Jaccard of queued cells {jac:.5f}'
        f' | positives {v} {int(a.sum())} ref {int(b.sum())}' + (' | OK' if jac >= 0.85 else ' | WARNING'))
    return agree


# ------------------------------------------------------------------ main
def main():
    panels = {}; stats = []
    for panel in PANELS:
        P = load_panel(panel)
        stats.append(select_windows(P))
        off = [w for w in read_official(panel, P) if w[2] == ONGOING]
        if off:
            od = np.array([w[0] for w in off]); ot = np.array([w[1] for w in off])
            log(panel, f'official train ongoing windows {len(off)}: faithful {int(P["cand"][od, ot].sum())}, run-starts {int(P["rs"][od, ot].sum())},'
                f' filled persistence {np.round(P["pers"][od, ot], 3).tolist()}')
            stats[-1].update(official_og=len(off), official_faithful=int(P['cand'][od, ot].sum()), official_runstart=int(P['rs'][od, ot].sum()))
        panels[panel] = P
    st = pd.DataFrame(stats); st.to_csv(f'{OUT}/t2v23_trainsets.csv', index=False)
    log('TRAIN SETS total:', st.drop(columns=['panel']).sum().to_dict())
    for k, nm in (('cur', 'CUR ongoing'), ('rs', 'run-starts'), ('rest', 'F03 rest')):
        log(f'filled persistence ALL panels | {nm}:', pq(np.concatenate([P['pers_sets'][k] for P in panels.values()])))

    # ---- training rows (ongoing only)
    cols = None; EX = []
    for panel, P in panels.items():
        cols, ex = panel_rows(P, cols)
        log(panel, f'rows EX {len(ex["y"])} (pos {ex["y"].mean() if len(ex["y"]) else float("nan"):.3f};'
            f' run-start rows {int(ex["rs"].sum())}, pos {ex["y"][ex["rs"]].mean() if ex["rs"].any() else float("nan"):.3f})')
        EX.append(ex)
        for k in ('V', 'Fq', 'E', 'Q', 'ON', 'OFF', 'cand', 'rs', 'pers'):       # only network vectors + prior are needed from here on
            P.pop(k, None)
        gc.collect()
    cat = lambda L_, k: np.concatenate([x[k] for x in L_])
    Xe, ye, re_ = cat(EX, 'X'), cat(EX, 'y').astype(np.float32), cat(EX, 'rs')
    del EX; gc.collect()
    cols_np = [x for x in cols if x not in PRIOR_COLS or x in OG_ADD_PROD]; ix_np = [cols.index(x) for x in cols_np]
    cols_id = [x for x in cols if x not in ID_DROP or x in OG_ADD_PROD]; ix_id = [cols.index(x) for x in cols_id]
    w_f = np.where(re_, W_RS, W_DENSE).astype(np.float32)
    new_cols = [x for x in cols_np if x in OG_STATIC or x in OG_STEP or x in OG_ADD_PROD]
    assert len(new_cols) == len(OG_STATIC) + len(OG_STEP) + len(OG_ADD_PROD), 'a new column is missing from the matrix'
    log(f'features: all {len(cols)}, np {len(cols_np)}, id {len(cols_id)} | new {len(new_cols)}: {new_cols}')
    log('new feature finite share / mean:', {c: (round(float(np.isfinite(Xe[:200000, cols.index(c)]).mean()), 3),
                                               round(float(np.nanmean(Xe[:200000, cols.index(c)])), 3)) for c in new_cols})
    log(f'POSITIVE RATES | F03 rows {len(ye)} pos {ye.mean():.4f} weighted {np.average(ye, weights=w_f) if len(ye) else float("nan"):.4f}'
        f' (run-start rows {int(re_.sum())} pos {ye[re_].mean() if re_.any() else float("nan"):.4f}; rest pos {ye[~re_].mean() if (~re_).any() else float("nan"):.4f})')
    P511 = dict(PARAMS, num_leaves=511, min_data_in_leaf=40, learning_rate=0.03)   # = v20 og_big / og_id (seed 0)
    models = {}; rate = [None]

    def fit(name, getX, y, w, cset, params=None):
        n = len(y)
        if rate[0] is not None and time.time() - T0 + rate[0] * n * 1.15 > TIME_LIMIT_S:
            log(f'SKIPPED model {name}: predicted {rate[0] * n / 3600:.2f} h would pass the time limit'); return
        t1 = time.time()
        X = getX()
        mdl = lgb.train(params or P511, lgb.Dataset(X, y, weight=w), N_BIG)
        del X; gc.collect()
        dt = time.time() - t1; rate[0] = max(rate[0] or 0.0, dt / max(n, 1))
        models[name] = (mdl, cset)
        log(f'model {name} trained: {n} rows x {len(cset)} cols, {N_BIG} trees, {dt / 60:.1f} min, weighted {w is not None}')

    S1 = dict(P511, seed=1, bagging_seed=11, feature_fraction_seed=21)
    fit('fs_np', lambda: Xe[:, ix_np], ye, w_f, cols_np)
    fit('fs_id', lambda: Xe[:, ix_id], ye, w_f, cols_id)
    fit('fs_np1', lambda: Xe[:, ix_np], ye, w_f, cols_np, S1)
    fit('fs_id1', lambda: Xe[:, ix_id], ye, w_f, cols_id, S1)
    del Xe; gc.collect()

    # ---- released windows
    subs = []; foot = []; probs = {}; names = None
    for panel, P in panels.items():
        for split in ('validation', 'private'):
            tpl, names = predict_split(P, panel, split, models, foot, probs)
            subs.append(tpl)
    S = pd.concat(subs, ignore_index=True)
    for name in names:
        S[['window_id', 'timestamp', 'link_id', f'queue_pred_{name}']].rename(columns={f'queue_pred_{name}': 'queue_pred'}).to_csv(
            f'{OUT}/queue_submission_{name}.csv', index=False)
    np.savez_compressed(f'{OUT}/t2v23_probs.npz', **probs)
    log('queue rows', len(S), 'dupes', int(S.duplicated(['window_id', 'timestamp', 'link_id']).sum()), 'ongoing rows', int(S.is_og.sum()),
        'files', [f'queue_submission_{n}.csv' for n in names], '| onset rows are 0 in every file')
    fp = print_footprint(foot)
    if len(fp):
        fp.to_csv(f'{OUT}/t2v23_footprint.csv', index=False)
    control_check(S, names)
    log('done')


main()
