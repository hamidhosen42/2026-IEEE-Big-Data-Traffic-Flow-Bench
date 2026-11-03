"""pvogf (track pvogf, E2 2026-10-09; lab analysis/t2lab/pvogf_*.py, generator analysis/t2lab/pvogf_make_kernel.py) = pvaf (tfb-t2ogpvaf,
pick A's April ongoing pvo) + the 98 + 5 OGFEAT columns of lab arm all2 (g1b corridor context, g2 per-link kinematics + kx step
extrapolations, g3 demand / supply at the queue ends, g1c corridor flow / speed level and trend, g2b per-link volatility / smoothed
state; no tod / dow) in all three members npB / idB / fullB.  The new columns use exactly the inputs pvaf feeds window_features in its
three row builders: the all-finite 60-min history [T-60, T), the all-finite (masked) slot-T value and the ramp columns of add_ramp
(slots < T).  Nothing after T.  April windows use only March + the April days before d (scenario prior), as pvaf.
Feature block copied verbatim from analysis/t2lab/ogfeat_lib.py.  The released-March rows are built by march_rows() (same code as pvaf).
Outputs: queue_submission_pvogf.csv (0.25 npB + 0.25 idB + 0.5 fullB), queue_submission_pvogf_npid.csv, queue_submission_pvogfv.csv
(growth veto: a private ongoing window keeps the pvo rows of tfb-t2ogpvaf if pvogf adds more than VETO_G cells outside the queue at T;
written only if tfb-t2ogpvaf is attached), t2pvogf_probs.npz (private member probabilities, window ids, qlast),
t2pvogf_veto_windows.csv.  Validation ongoing rows are copied from tfb-t2v22af fs4af (as pvaf); onset rows 0.  CONTROL lines vs
tfb-t2ogpvaf queue_submission_pvo.csv (rows / windows changed per split).

pvaf notes: Task 2 ONGOING for the PRIVATE month with same-scenario data AND all-finite inputs (ogpvaf) = t2_ongoing_pv.py with every finite
released speed / flow value (ineligible-but-valued cells, about 7 %, never Task-1-masked) fed into the FEATURES of the train F03
rows, the released-March training windows and the released windows (history + slot T); labels, window selection, eligibility
and the queue priors stay on eligible cells.  The same input change gave LB +0.0009 for the train-only F03 bag (tfb-t2v22af).
Validation rows are copied from tfb-t2v22af fs4af (current public base); only private rows change (pvo -> pvoaf).
"""
import glob, json, os, time, gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from tfb.runtime import ctx as TFB  # [TFB] vendored from consolidation/lineage_src/tfb-t2ogpvogf.py; patched lines end in [TFB]

T0 = time.time()
REL = TFB.rel  # [TFB]
OUT = TFB.out  # [TFB]
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']
SMOKE = TFB.smoke  # [TFB] False in every real run (True: D7_I10_E + D12_I5_S, few trees)
if SMOKE:
    PANELS = ['D7_I10_E', 'D12_I5_S']
H, F, SPACING = 12, 6, 72
N_BIG = 40 if SMOKE else 1800               # trees of the 511-leaf ongoing models (v20: 1800)
MAX_OG = 300 if SMOKE else 3000             # dense ongoing windows per panel (CUR, and the non-run-start part of F03)
FILL_GAP, PERS_MAX = 36, 0.9                # faithful rule: time fill over interior gaps <= 36 slots, persistence cap 0.9
W_RS, W_DENSE = 1.0, 0.3                    # window weights: run-starts / everything else in F03 and H
TIME_LIMIT_S = 8.3 * 3600                   # a model is skipped if its predicted end exceeds this (Kaggle budget 9 h)
ONGOING = 'queue_ongoing'
ALLFIN = True                                # features from all finite released values (labels / selection: eligible cells)
OG_STATIC = ['c_q3', 'c_q6', 'c_q9', 'c_q12', 'c_d3', 'c_d6', 'c_d12', 'c_rel6', 'c_n0', 'c_new3', 'c_clr3', 'c_new6', 'c_clr6', 'c_first', 'c_peak', 'k_run', 'k_ntr', 'k_qf6', 'k_dvr3', 'k_dvr6', 'k_vrmax6', 'k_sdlo0', 'k_sdhi0', 'k_cs0', 'k_nq7_0', 'k_nqhi8_0', 'k_nqlo8_0', 'k_sdlo3', 'k_sdhi3', 'k_cs3', 'k_nq7_3', 'k_nqhi8_3', 'k_nqlo8_3', 'k_sdlo6', 'k_sdhi6', 'k_cs6', 'k_nq7_6', 'k_nqhi8_6', 'k_nqlo8_6', 'k_vlo3', 'k_vhi3', 'k_vlo6', 'k_vhi6', 'e_side', 'e_dist', 'e_size', 'e_rel', 'e_n3', 'e_n6', 'e_x3', 'e_x6', 'e_vr0', 'e_dvr3', 'e_dvr6', 'e_fr0', 'e_in_fr0', 'e_in_dfr3', 'e_out_fr0', 'e_out_dfr3', 'e_in_vr0', 'e_in_dvr3', 'e_out_vr0', 'e_out_dvr3', 'e_ramp_tail', 'e_ramp_head', 'e_off_head', 'e_ramp_in', 'e_surge_head', 'c2_fr0', 'c2_dfr3', 'c2_dfr6', 'c2_dfr11', 'c2_vr0', 'c2_dvr3', 'c2_dvr6', 'c2_dvr12', 'c2_vq0', 'c2_dvq3', 'c2_dvq6', 'c2_frq0', 'c2_dfrq3', 'c2_slow0', 'c2_dslow3', 'k2_vstd', 'k2_vmin', 'k2_vmax', 'k2_z', 'k2_vm5', 'k2_dvm5_3', 'k2_dvm5_6', 'k2_fm5', 'k2_dfm5_3', 'k2_fr_up', 'k2_dfr_up3', 'k2_fr_dn', 'k2_dfr_dn3', 'k2_vr_up', 'k2_vr_dn']
OG_STEP = ['kx_lo3', 'kx_hi3', 'kx_lo6', 'kx_hi6', 'kx_min3']
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
    Vx = V.copy() if ALLFIN else None; Fx = V.copy() if ALLFIN else None       # ineligible cells that carry a value
    for d, f in enumerate(files):
        x = pd.read_parquet(f, columns=['timestamp', 'link_id', 'speed_kmh', 'flow_vph', 'is_score_eligible'])
        x = x[x.speed_kmh.notna()]
        ts = pd.to_datetime(x.timestamp, utc=True)
        x = x.assign(t=(ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(), l=x.link_id.astype(str).map(li)).dropna(subset=['l'])
        el = (x.is_score_eligible == 1).to_numpy()
        g = x[el].groupby(['t', 'l'])[['speed_kmh', 'flow_vph']].mean().reset_index()
        V[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.speed_kmh.to_numpy()
        Fq[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.flow_vph.to_numpy()
        if ALLFIN:
            g = x[~el].groupby(['t', 'l'])[['speed_kmh', 'flow_vph']].mean().reset_index()
            Vx[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.speed_kmh.to_numpy()
            Fx[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.flow_vph.to_numpy()
    E = np.isfinite(V); Q = E & (V <= cut[None, None, :])
    dow = pd.to_datetime(pd.Series(days)).dt.dayofweek.to_numpy()
    ON, OFF = ramp_cube(panel, 'train', li, days)
    P = dict(panel=panel, links=links, li=li, cut=cut, vf=vf, cap=cap, lanes=lanes, length=length,
             days=days, dow=dow, V=V, Fq=Fq, E=E, Q=Q, ON=ON, OFF=OFF)
    if ALLFIN:                                                             # FEATURE cubes: eligible value, else the ineligible value
        P['Vf'] = np.where(E, V, Vx); P['Ff'] = np.where(E, Fq, Fx)
        log(panel, 'all-finite inputs: eligible share', round(float(E.mean()), 4), '| + ineligible valued', round(float(np.isfinite(Vx).mean()), 4))
    return P


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


def og_extra2(P, Vh, Fh, VT, Fe):
    """round-2 features (same inputs as og_extra): g1c corridor flow / speed level and trend (c2_*), g2b per-link volatility and
    spatially smoothed speed / flow with trends (k2_*)."""
    with np.errstate(all='ignore'):
        cut = P['cut']; W, _, L = Vh.shape; idx = np.arange(L)[None, :]
        vr = np.concatenate([Vh / cut[None, None, :], (VT / cut[None, :])[:, None, :]], 1)
        fr = Fh / P['cap'][None, None, :]
        vb = og_ffill(og_ffill(vr)[:, ::-1])[:, ::-1]; frb = og_ffill(og_ffill(fr)[:, ::-1])[:, ::-1]
        Q = np.nan_to_num(vb, nan=9.0) <= 1.0
        X = {}
        bc = lambda v: np.repeat(np.asarray(v, np.float32)[:, None], L, 1)
        nm = lambda a, m=None: (np.where(np.isfinite(a) & (True if m is None else m), a, 0.0).sum(1)
                                / np.maximum((np.isfinite(a) & (True if m is None else m)).sum(1), 1))
        # ---------------- g1c corridor level / trend of flow and speed
        f0 = nm(frb[:, 11]); v0 = nm(vb[:, 12]); q12 = Q[:, 12]
        X['c2_fr0'] = bc(f0); X['c2_dfr3'] = bc(f0 - nm(frb[:, 8])); X['c2_dfr6'] = bc(f0 - nm(frb[:, 5])); X['c2_dfr11'] = bc(f0 - nm(frb[:, 0]))
        X['c2_vr0'] = bc(v0); X['c2_dvr3'] = bc(v0 - nm(vb[:, 9])); X['c2_dvr6'] = bc(v0 - nm(vb[:, 6])); X['c2_dvr12'] = bc(v0 - nm(vb[:, 0]))
        vq = nm(vb[:, 12], q12)
        X['c2_vq0'] = bc(vq); X['c2_dvq3'] = bc(vq - nm(vb[:, 9], q12)); X['c2_dvq6'] = bc(vq - nm(vb[:, 6], q12))
        fq = nm(frb[:, 11], q12)
        X['c2_frq0'] = bc(fq); X['c2_dfrq3'] = bc(fq - nm(frb[:, 8], q12))
        slow = lambda a: ((a > 1.0) & (a <= 1.3)).mean(1)
        X['c2_slow0'] = bc(slow(vb[:, 12])); X['c2_dslow3'] = bc(slow(vb[:, 12]) - slow(vb[:, 9]))
        # ---------------- g2b per-link volatility, smoothed speed / flow and trends
        okv = np.isfinite(vr); n = okv.sum(1)
        mu = np.where(okv, vr, 0.0).sum(1) / np.maximum(n, 1)
        sdv = np.sqrt(np.where(okv, (vr - mu[:, None, :]) ** 2, 0.0).sum(1) / np.maximum(n - 1, 1))
        X['k2_vstd'] = np.where(n >= 3, sdv, np.nan)
        X['k2_vmin'] = np.where(n > 0, np.where(okv, vr, 9.0).min(1), np.nan); X['k2_vmax'] = np.where(n > 0, np.where(okv, vr, -9.0).max(1), np.nan)
        X['k2_z'] = (vb[:, 12] - 1.0) / (np.where(n >= 3, sdv, 0.1) + 0.03)
        m0 = og_rmean(vb[:, 12], idx - 2, idx + 2)
        X['k2_vm5'] = m0; X['k2_dvm5_3'] = m0 - og_rmean(vb[:, 9], idx - 2, idx + 2); X['k2_dvm5_6'] = m0 - og_rmean(vb[:, 6], idx - 2, idx + 2)
        g0 = og_rmean(frb[:, 11], idx - 2, idx + 2)
        X['k2_fm5'] = g0; X['k2_dfm5_3'] = g0 - og_rmean(frb[:, 8], idx - 2, idx + 2)
        u0 = og_rmean(frb[:, 11], idx - 6, idx - 1); d0 = og_rmean(frb[:, 11], idx + 1, idx + 6)
        X['k2_fr_up'] = u0; X['k2_dfr_up3'] = u0 - og_rmean(frb[:, 8], idx - 6, idx - 1)
        X['k2_fr_dn'] = d0; X['k2_dfr_dn3'] = d0 - og_rmean(frb[:, 8], idx + 1, idx + 6)
        X['k2_vr_up'] = og_rmean(vb[:, 12], idx - 6, idx - 1); X['k2_vr_dn'] = og_rmean(vb[:, 12], idx + 1, idx + 6)
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
    Vh = P.get('Vf', P['V'])[d[:, None], idx]; Fh = P.get('Ff', P['Fq'])[d[:, None], idx]
    # slot T itself is only visible through the Task-1-masked layer (~50-66% of cells): simulate that
    rng = np.random.default_rng(len(wins))
    VT = np.where(rng.random((len(d), P['V'].shape[2])) < 0.62, P['V'][d, t], np.nan)
    if 'Vf' in P:                                                          # ineligible-valued cells are never Task-1-masked
        VT = np.where(P['E'][d, t], VT, P['Vf'][d, t])
    Fe, qlast = window_features(P, Vh, Fh, t, P['dow'][d], prior, VT)
    add_ramp(Fe, P['ON'], P['OFF'], d, t, P['cap'])
    OGX = og_extra(P, Vh, Fh, VT, Fe); OGX.update(og_extra2(P, Vh, Fh, VT, Fe))          # pvogf: ogfeat columns
    Fe.update({k: OGX[k] for k in OG_STATIC})
    W, L = qlast.shape
    rows = []
    for h in range(1, F + 1):
        slot = np.minimum(t + h, 287)
        feats = {k: v for k, v in Fe.items()}
        feats['h'] = np.full((W, L), h, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = prior_feats(prior, P['dow'][d], t, h)
        feats.update({k: v for k, v in og_step(OGX, h).items() if k in OG_STEP})          # pvogf: step-dependent columns
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
if TFB.cfg['lgb_repro'] is not None:  # [TFB] reproducibility switch (tfb/config.py LGB_REPRO): deterministic=True, force_row_wise=True;
    PARAMS.update(TFB.cfg['lgb_repro'])  # [TFB] None = the original setting (auto row/col-wise choice by timing, not repeatable)


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
    R = make_rows(P, P['trainw'], P['prior_all'])
    X, y, e, meta, cols = to_matrix(R, ONGOING, cols, ongoing_keep(R))
    cur = dict(X=X[e], y=y[e], rs=P['cur_rs'][meta[:, 0]][e])
    del R, X; gc.collect()
    R = make_rows(P, P['exw'], P['prior_all'])
    if R is None:
        ex = dict(X=np.zeros((0, len(cols)), np.float32), y=np.zeros(0, bool), rs=np.zeros(0, bool), incur=np.zeros(0, bool))
    else:
        X, y, e, meta, _ = to_matrix(R, ONGOING, cols, ongoing_keep(R))
        ex = dict(X=X[e], y=y[e], rs=P['ex_rs'][meta[:, 0]][e], incur=P['ex_incur'][meta[:, 0]][e])
        del R, X; gc.collect()
    return cols, cur, ex


# ------------------------------------------------------------------ released windows (validation / private)
def load_split(P, panel, split):
    """features of the released windows exactly as v20: 60-min history, masked slot-T value, ramp slots < T. No labels."""
    wi = pd.read_csv(f'{REL}/task2/{panel}/{split}/window_index.csv')
    hist = pd.read_parquet(f'{REL}/task2/{panel}/{split}/window_history.parquet')
    tpl = pd.read_csv(f'{REL}/task2/{panel}/{split}/sample_submission_queue.csv', dtype=str)
    W, L = len(wi), len(P['links'])
    Vh = np.full((W, H, L), np.nan, np.float32); Fh = Vh.copy()
    wpos = {w: i for i, w in enumerate(wi.window_id)}
    hist = hist[((hist.is_score_eligible == 1) | ALLFIN) & hist.speed_kmh.notna()]
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
        m = m[((m.is_score_eligible == 1) | ALLFIN) & m.speed_kmh.notna()]
        for r in grp.itertuples():
            x = m[m.timestamp == r.forecast_origin]
            x = x[x.link_id.isin(P['li'])].groupby('link_id').speed_kmh.mean()
            VT[wpos[r.window_id], x.index.map(P['li']).to_numpy().astype(np.int64)] = x.to_numpy()  # [TFB] [POSTT-R1] empty slot-T frame -> no-op (was IndexError)
    log(panel, split, 'slot-T visible share', round(float(np.isfinite(VT).mean()), 3),
        'history filled share', round(float(np.isfinite(Vh).mean()), 3))
    Fe, qlast = window_features(P, Vh, Fh, tod, dow, P['prior_all'], VT)
    wdays = sorted(wi.date.unique())
    ONs, OFFs = ramp_cube(panel, split, P['li'], wdays)
    add_ramp(Fe, ONs, OFFs, wi.date.map({dd: i for i, dd in enumerate(wdays)}).to_numpy().astype(np.int64), tod.astype(np.int64), P['cap'])  # [TFB] [POSTT-R6] split without windows (was IndexError)
    log(panel, split, 'ramp on3 finite share', round(float(np.isfinite(Fe['ramp_on3']).mean()), 3))
    OGX = og_extra(P, Vh, Fh, VT, Fe); OGX.update(og_extra2(P, Vh, Fh, VT, Fe))          # pvogf: ogfeat columns
    Fe.update({k: OGX[k] for k in OG_STATIC})
    R = dict(d=np.zeros(W, int), t=tod, cond=wi.condition.to_numpy(), qlast=qlast, W=W, L=L, rows=[])
    for h in range(1, F + 1):
        feats = dict(Fe); feats['h'] = np.full((W, L), h, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = prior_feats(P['prior_all'], dow, tod, h)
        feats.update({k: v for k, v in og_step(OGX, h).items() if k in OG_STEP})          # pvogf: step-dependent columns
        R['rows'].append((h, feats, None, None))
    return wi, tpl, R


VARIANTS = {'fs2': ('fs_np', 'fs_id'),                                 # = v21 fs2 (control)
            'fs4': ('fs_np', 'fs_id', 'fs_np1', 'fs_id1'),             # F03 bag of 4
            'fs_np': ('fs_np',)}


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
            a, b = pr[i], preds['fs2'][i]
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
        log(f'FOOTPRINT {split}: ongoing windows {g[g.variant == "fs2"].shape[0]} | mean queued cells at T (q_last) '
            f'{g[g.variant == "fs2"].size_T.mean():.2f}')
        for v, x in g.groupby('variant', sort=False):
            log(f'  {v:8s} windows changed {int(x.changed.sum())}/{len(x)} | mean Jaccard vs fs2 {x.jacc_cur2.mean():.4f}'
                f' | cells changed per step h1..h6 {[int(x[f"chg_h{h}"].sum()) for h in hs]} (added {[int(x[f"add_h{h}"].sum()) for h in hs]})'
                f' | mean predicted size per step {[round(float(x[f"size_h{h}"].mean()), 2) for h in hs]}')
        pv = g.pivot_table(index='panel', columns='variant', values='changed', aggfunc='sum')
        log(f'  windows changed per panel ({split}):', pv.to_dict('index'))
    return fp


def control_check(S, names):
    """agreement of fs2 with tfb-t2v21 queue_submission_fs2.csv on ongoing rows, if that output is attached (optional)."""
    ref = TFB.kglob('tfb-t2v21', 'queue_submission_fs2.csv')  # [TFB] optional CONTROL reference
    if not ref or 'fs2' not in names:
        log('CONTROL: skipped (tfb-t2v21 queue_submission_fs2.csv not attached or fs2 missing)', ref)
        return None
    r = pd.read_csv(ref[0], dtype={'window_id': str, 'timestamp': str, 'link_id': str})
    m = S.merge(r.rename(columns={'queue_pred': 'ref'}), on=['window_id', 'timestamp', 'link_id'], how='left')
    og = m.is_og.to_numpy() & m.ref.notna().to_numpy()
    a = m.queue_pred_fs2.to_numpy()[og].astype(int); b = m.ref.to_numpy()[og].astype(int)
    agree = float((a == b).mean()) if og.any() else float('nan')
    jac = float((a & b).sum() / max((a | b).sum(), 1))
    log(f'CONTROL fs2 vs {ref[0]}: ongoing rows {int(og.sum())} | agreement {agree:.5f} | Jaccard of queued cells {jac:.5f}'
        f' | positives fs2 {int(a.sum())} ref {int(b.sum())}' + (' | OK' if jac >= 0.98 else ' | WARNING'))
    return agree


# ------------------------------------------------------------------ main (ongoing pv: private month only)
WM = 5.0                                    # weight multiplier of the March windows (lab: 5)
K_BLEND = 4                                 # same-scenario prior shrinkage a = n / (n + K_BLEND)
P_PV = dict(PARAMS, num_leaves=511, min_data_in_leaf=40, learning_rate=0.05)
N_PV = 60 if SMOKE else 900
REF_V22 = tuple(TFB.cfg['ref_v22'])  # [TFB] validation rows are copied from here; original ('tfb-t2v22af', 'queue_submission_fs4af.csv'), entry script fs4occ (rows never used)
REF_BASE = ('tfb-f5', 'submission_F5.csv')             # current chain base (CONTROL)
REF_PVO = ('tfb-t2ogpvaf', 'queue_submission_pvo.csv')   # pvogf: pick A's current April ongoing rows (veto reference + CONTROL)
VETO_G = TFB.cfg['veto_g']  # [TFB] pvogf: growth veto threshold (cells outside the queue at T, all 6 steps); 10


def masked_month(panel, split, li):
    """released masked month: visible eligible speed / flow [D, 288, L] (NaN = not visible), eligibility flag, sorted days."""
    files = sorted(glob.glob(f'{REL}/corridors/{panel}/{split}/mainline_states_masked/*/*.parquet'),
                   key=lambda f: os.path.basename(f)[-18:-8])           # files sit in mask_regime=R* folders: sort by date
    days = [os.path.basename(f)[-18:-8].replace('_', '-') for f in files]
    V = np.full((len(files), 288, len(li)), np.nan, np.float32); Fq = V.copy(); Em = np.zeros(V.shape, bool)
    Vx = V.copy(); Fx = V.copy()                                           # ineligible cells that carry a value (never masked)
    for d, f in enumerate(files):
        x = pd.read_parquet(f, columns=['timestamp', 'link_id', 'speed_kmh', 'flow_vph', 'is_score_eligible'])
        ts = pd.to_datetime(x.timestamp, utc=True)
        x = x.assign(t=(ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(), l=x.link_id.astype(str).map(li)).dropna(subset=['l'])
        e = x[x.is_score_eligible == 1]
        Em[d, e.t.to_numpy(), e.l.astype(int).to_numpy()] = True
        g = x[(x.is_score_eligible == 1) & x.speed_kmh.notna()].groupby(['t', 'l'])[['speed_kmh', 'flow_vph']].mean().reset_index()
        V[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.speed_kmh.to_numpy()
        Fq[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.flow_vph.to_numpy()
        g = x[(x.is_score_eligible != 1) & x.speed_kmh.notna()].groupby(['t', 'l'])[['speed_kmh', 'flow_vph']].mean().reset_index()
        Vx[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.speed_kmh.to_numpy()
        Fx[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.flow_vph.to_numpy()
    if ALLFIN:
        Vx = np.where(np.isfinite(V), V, Vx); Fx = np.where(np.isfinite(V), Fq, Fx)
    else:
        Vx, Fx = V, Fq
    return days, V, Fq, Em, Vx, Fx


def scenario(P):
    """March cubes (for training windows) + cumulative same-scenario queue counts of March + April (prior), by weekday class,
    summed over slots +-2: cum[c, j] = sum over timeline days < j of class c (0 weekday, 1 weekend, 2 all)."""
    dv, Vv, Fv, Ev, Vfv, Ffv = masked_month(P['panel'], 'validation', P['li']); dp, Vp, _, _, _, _ = masked_month(P['panel'], 'private', P['li'])
    ONv, OFFv = ramp_cube(P['panel'], 'validation', P['li'], dv)
    M = dict(days=dv, V=Vv, Fq=Fv, Em=Ev, Vf=Vfv, Ff=Ffv, ON=ONv, OFF=OFFv, dow=pd.to_datetime(pd.Series(dv)).dt.dayofweek.to_numpy())
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
    log(P['panel'], 'scenario days', len(dv), '+', len(dp))
    return M, {dd: i for i, dd in enumerate(days)}, cQ, cE


def march_faithful(M, cut):
    """faithful ongoing candidates of the visible masked March month (the lab's pseudo-March selector): coverage from the
    eligibility flag, windows touching a blanked slot dropped, established queue in the visible history, a queued cell of the
    time-filled visible state in T+1..T+6 and persistence IoU <= 0.9 over the filled cells.  TRAINING windows only."""
    V, Em = M['V'], M['Em']; E = np.isfinite(V); Q = E & (np.nan_to_num(V, nan=1e9) <= cut[None, None, :])
    VF = fill_time(V, FILL_GAP); MF = np.isfinite(VF); QF = MF & (VF <= cut[None, None, :]); del VF
    blank = (Em.sum(2) >= 5) & (E.sum(2) == 0)
    D = V.shape[0]; cand = np.zeros((D, 288), bool)
    for d in range(D):
        for t in range(H, 288 - F):
            if blank[d, t - H:t + F + 1].any() or Em[d, t - H:t].mean() < 0.7 or not established(Q[d, t - H:t]):
                continue
            Tf = QF[d, t + 1:t + 1 + F]; Mf = MF[d, t + 1:t + 1 + F]
            if not Tf.any():
                continue
            S = np.repeat(QF[d, t][None], F, 0)
            u = ((Tf | S) & Mf).sum(); i = (Tf & S & Mf).sum()
            cand[d, t] = (1.0 if u == 0 else i / u) <= PERS_MAX
    return cand, Q, E


def blend_prior_og(X, meta, P, jd, cls, tod, cQ, cE, cols):
    """copy of X with prior / prior_now / prior_alldays blended per row (window day -> scenario days before it; h-dependent slot)."""
    jp = [cols.index(c) for c in ('prior', 'prior_now', 'prior_alldays')]
    wv, lv, hv = meta[:, 0], meta[:, 1], meta[:, 2]
    cr, jr, tr_ = cls[wv], jd[wv], tod[wv]; sh = np.minimum(tr_ + hv, 287)
    trv = [P['prior_all'][cr, sh, lv], P['prior_all'][cr, tr_, lv], P['prior_all'][2, sh, lv]]
    Xb = X.copy(); a0 = 0.0
    for j, (cc, ss) in enumerate(((cr, sh), (cr, tr_), (np.full_like(cr, 2), sh))):
        q = cQ[cc, jr, ss, lv]; ee = cE[cc, jr, ss, lv]
        sp = np.where(ee > 0, q / np.maximum(ee, 1e-9), 0.0).astype(np.float32); n = ee / 5.0; a_ = n / (n + K_BLEND)
        Xb[:, jp[j]] = (1 - a_) * trv[j] + a_ * sp
        if j == 0 and len(a_):
            a0 = float(a_.mean())
    return Xb, a0


def march_rows(P, M, Qm, Emv, d, t, cols):
    """pvogf: the released-March training rows of pvaf main() as a function, + the ogfeat columns.  Windows (d, t) of the masked March
    month: all-finite masked history + all-finite masked slot T (Vf / Ff), visible labels Qm / Emv, all-train prior in the prior columns."""
    idx = t[:, None] + np.arange(-H, 0)[None]
    Vh, Fh, VT = M['Vf'][d[:, None], idx], M['Ff'][d[:, None], idx], M['Vf'][d, t]
    Fe, qlast = window_features(P, Vh, Fh, t, M['dow'][d], P['prior_all'], VT)
    add_ramp(Fe, M['ON'], M['OFF'], d, t, P['cap'])
    OGX = og_extra(P, Vh, Fh, VT, Fe); OGX.update(og_extra2(P, Vh, Fh, VT, Fe))          # pvogf: ogfeat columns
    Fe.update({k: OGX[k] for k in OG_STATIC})
    W_, L_ = qlast.shape
    R = dict(d=d, t=t, cond=np.array([ONGOING] * W_), qlast=qlast, W=W_, L=L_, rows=[])
    for h in range(1, F + 1):
        feats = dict(Fe); feats['h'] = np.full((W_, L_), h, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = prior_feats(P['prior_all'], M['dow'][d], t, h)
        feats.update({k: v for k, v in og_step(OGX, h).items() if k in OG_STEP})          # pvogf: step-dependent columns
        R['rows'].append((h, feats, Qm[d, np.minimum(t + h, 287)], Emv[d, np.minimum(t + h, 287)]))
    X, y, e, meta, _ = to_matrix(R, ONGOING, cols, ongoing_keep(R))
    return R, X, y, e, meta


PVO_CACHE = {}


def pvo_rows():
    """pvogf: tfb-t2ogpvaf queue_submission_pvo.csv (pick A's current April ongoing rows), parsed once; None if not attached."""
    if 'df' not in PVO_CACHE:
        f = TFB.kglob(REF_PVO[0], REF_PVO[1])  # [TFB] optional veto reference
        PVO_CACHE['df'] = None
        if f:
            r = pd.read_csv(f[0], dtype={'window_id': str, 'timestamp': str, 'link_id': str})
            r['ts'] = pd.to_datetime(r.timestamp, utc=True)
            PVO_CACHE['df'] = r[['window_id', 'ts', 'link_id', 'queue_pred']].rename(columns={'queue_pred': 'q_ref'})
            log('REF pvo:', f[0], len(r), 'rows')
        else:
            log('REF pvo: not attached (no veto variant, no CONTROL vs pvo)')
    return PVO_CACHE['df']


def veto_vs_pvo(tpl, B, qlast, is_og, wi, g=None):
    """pvogf: per ongoing window the pvogf decision B unless it adds more than VETO_G cells outside the queue at T (qlast, all 6 steps)
    compared with pvo; then the pvo decision.  Returns (template column, per-window info frame)."""
    g = VETO_G if g is None else g
    ref = pvo_rows()
    assert ref is not None, 'pvo reference not attached'
    mm = tpl.assign(ts=pd.to_datetime(tpl.timestamp, utc=True))[['window_id', 'ts', 'link_id']].merge(
        ref, on=['window_id', 'ts', 'link_id'], how='left', validate='one_to_one')
    assert len(mm) == len(tpl) and mm.q_ref.notna().all(), 'pvo rows missing'
    W, F_, L = B.shape
    w_, h_, l_ = tpl.w.to_numpy().astype(int), tpl.h.to_numpy().astype(int) - 1, tpl.l.to_numpy().astype(int)
    A = np.zeros((W, F_, L), bool); A[w_, h_, l_] = mm.q_ref.to_numpy().astype(int) == 1
    q = np.repeat(qlast[:, None, :], F_, 1)
    add = (B & ~A & ~q).sum((1, 2))
    vet = is_og & (add > g)
    out = np.where(vet[:, None, None], A, B)
    info = pd.DataFrame(dict(window_id=wi.window_id.to_numpy(), addout=add, veto=vet, n_pvo=A.sum((1, 2)), n_pvogf=B.sum((1, 2)),
                             cells_changed=(A != B).sum((1, 2))))[is_og]
    return out[w_, h_, l_].astype(int), info


def main():
    panels = {}
    for panel in PANELS:
        P = load_panel(panel)
        select_windows(P)
        panels[panel] = P
    # ---- F03 training rows exactly as v22 (EX set: run-starts 1.0 + faithful t%3 rest 0.3), train prior in the prior columns
    cols = None; EX = []
    for panel, P in panels.items():
        P['prior_all'] = prior_table(P, np.ones(len(P['days']), bool))
        R = make_rows(P, P['exw'], P['prior_all'])
        X, y, e, meta, cols = to_matrix(R, ONGOING, cols, ongoing_keep(R))
        EX.append(dict(X=X[e], y=y[e], w=np.where(P['ex_rs'][meta[:, 0]][e], W_RS, W_DENSE).astype(np.float32)))
        del R, X
        for k in ('V', 'Fq', 'E', 'Q', 'ON', 'OFF', 'cand', 'rs', 'pers', 'trainw', 'Vf', 'Ff'):
            P.pop(k, None)
        gc.collect()
        log(panel, 'F03 rows', len(EX[-1]['y']), 'pos', round(float(EX[-1]['y'].mean()), 4))
    # ---- same-scenario training rows: faithful ongoing windows of the released March month (observed labels)
    MR = []; scen = {}
    for panel, P in panels.items():
        M, dpos, cQ, cE = scenario(P)
        scen[panel] = (dpos, cQ, cE)
        cand, Qm, Emv = march_faithful(M, P['cut'])
        prev = np.zeros_like(cand); prev[:, 1:] = cand[:, :-1]
        rs = cand & ~prev; rest = cand & ~rs & (np.arange(288) % 3 == 0)[None, :]
        wl = sorted([(int(a), int(b), W_RS) for a, b in zip(*np.where(rs))] + [(int(a), int(b), W_DENSE) for a, b in zip(*np.where(rest))])
        if not wl:
            continue
        d = np.array([w[0] for w in wl]); t = np.array([w[1] for w in wl]); ww = np.array([w[2] for w in wl], np.float32)
        R, X, y, e, meta = march_rows(P, M, Qm, Emv, d, t, cols)                       # pvogf: pvaf code + ogfeat columns
        jd = np.array([dpos[M['days'][x]] for x in d]); cls = (M['dow'][d] >= 5).astype(int)
        Xb, a0 = blend_prior_og(X, meta, P, jd, cls, t, cQ, cE, cols)
        MR.append(dict(X=X[e], Xb=Xb[e], y=y[e], w=(ww[meta[:, 0]] * WM)[e]))
        log(panel, 'March ongoing training windows', len(wl), f'(run-starts {int(rs.sum())})', 'rows', int(e.sum()),
            'pos', round(float(y[e].mean()), 4), 'mean a', round(a0, 3))
        del R, X, Xb, M; gc.collect()
    cat_ = lambda L_, k: np.concatenate([x[k] for x in L_])
    Xe, ye, we = cat_(EX, 'X'), cat_(EX, 'y').astype(np.float32), cat_(EX, 'w'); del EX
    if not MR:  # [TFB] [POSTT-R2] no March faithful window before the cut (was ValueError)
        MR = [dict(X=np.zeros((0, len(cols)), np.float32), Xb=np.zeros((0, len(cols)), np.float32), y=np.zeros(0, bool), w=np.zeros(0, np.float32))]  # [TFB] [POSTT-R2]
    Xm, Xmb, ym, wm = cat_(MR, 'X'), cat_(MR, 'Xb'), cat_(MR, 'y').astype(np.float32), cat_(MR, 'w'); del MR
    gc.collect()
    cols_np = [x for x in cols if x not in PRIOR_COLS]; ix_np = [cols.index(x) for x in cols_np]
    cols_id = [x for x in cols if x not in ID_DROP]; ix_id = [cols.index(x) for x in cols_id]
    new_cols = [x for x in cols if x in OG_STATIC or x in OG_STEP]                       # pvogf: ogfeat columns in npB / idB / fullB
    assert len(new_cols) == len(OG_STATIC) + len(OG_STEP) and all(c in cols_np and c in cols_id for c in new_cols), 'ogfeat column missing'
    assert 'tod' not in cols_np and 'dow' not in cols_id
    log(f'pvogf features: all {len(cols)}, np {len(cols_np)}, id {len(cols_id)} | new {len(new_cols)} | Xe {Xe.shape} {Xe.nbytes / 1e9:.2f} GB'
        f' | Xm {Xm.shape} | new-column finite share (first 200k F03 rows) min '
        f'{min(float(np.isfinite(Xe[:200000, cols.index(c)]).mean()) for c in new_cols):.3f}')
    log(f'rows: F03 {len(ye)} (pos {ye.mean():.4f}) | March {len(ym)} (pos {ym.mean():.4f}, weight sum {wm.sum():.0f} vs F03 {we.sum():.0f})')
    y_all = np.concatenate([ye, ym]); w_all = np.concatenate([we, wm])
    models = {}
    for name, getX in (('npB', lambda: np.concatenate([Xe[:, ix_np], Xm[:, ix_np]])),
                       ('idB', lambda: np.concatenate([Xe[:, ix_id], Xm[:, ix_id]])),
                       ('fullB', lambda: np.concatenate([Xe, Xmb]))):
        t1 = time.time(); X = getX()
        models[name] = lgb.train(P_PV, lgb.Dataset(X, y_all, weight=w_all), N_PV)
        del X; gc.collect()
        log(f'model {name} trained: {len(y_all)} rows, {N_PV} trees, {(time.time() - t1) / 60:.1f} min')
    del Xe, Xm, Xmb; gc.collect()

    # ---- released windows: private ongoing windows get the March-augmented models (full model fed the blended prior)
    subs = []; stats = []; PROBS = {}; VETO = dict(ok=pvo_rows() is not None, info=[])
    for panel, P in panels.items():
        dpos, cQ, cE = scen.pop(panel)
        for split in ('validation', 'private'):
            wi, tpl, R = load_split(P, panel, split)
            W, L = R['W'], R['L']
            is_og = (wi.condition == ONGOING).to_numpy()
            org = pd.to_datetime(wi.set_index('window_id').forecast_origin, utc=True)
            tod = (org.dt.hour * 12 + org.dt.minute // 5).to_numpy(); dow = org.dt.dayofweek.to_numpy()
            preds = {n: np.zeros((W, F, L), bool) for n in ('pvogf', 'pvogf_npid')}
            if split == 'private':
                X, _, _, meta, _ = to_matrix(R, ONGOING, cols, ongoing_keep(R))
                jd = np.array([dpos.get(pd.Timestamp(x).strftime('%Y-%m-%d'), len(dpos)) for x in org.to_numpy()], dtype=np.int64)  # [TFB] [POSTT-R3/R4] day without masked file -> all available days; empty split -> int dtype (was KeyError / IndexError)
                Xb, a0 = blend_prior_og(X, meta, P, jd, (dow >= 5).astype(int), tod, cQ, cE, cols)
                pn = models['npB'].predict(X[:, ix_np]); pi = models['idB'].predict(X[:, ix_id]); pf = models['fullB'].predict(Xb)
                for nm_, p_ in (('npB', pn), ('idB', pi), ('fullB', pf)):                      # pvogf: member probabilities
                    P6m = np.zeros((W, F, L), np.float32); P6m[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = p_
                    PROBS[f'{panel}__private__{nm_}'] = P6m[is_og]
                PROBS[f'{panel}__private__window_id'] = wi.window_id.to_numpy()[is_og].astype(str)
                PROBS[f'{panel}__private__qlast'] = R['qlast'][is_og]
                for name, p in (('pvogf', 0.25 * pn + 0.25 * pi + 0.5 * pf), ('pvogf_npid', 0.5 * (pn + pi))):
                    P6 = np.zeros((W, F, L), np.float32); P6[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = p
                    for i in np.where(is_og)[0]:
                        preds[name][i] = decode(P6[i].ravel()).reshape(F, L)
                stats.append(dict(panel=panel, ongoing_windows=int(is_og.sum()), mean_a=round(a0, 3),
                                  size_T=float(R['qlast'][is_og].sum(1).mean()),
                                  size_pvogf=float(preds['pvogf'][is_og].sum((1, 2)).mean() / F),
                                  size_npid=float(preds['pvogf_npid'][is_og].sum((1, 2)).mean() / F)))
                log(json.dumps(stats[-1]))
            wpos = {w: i for i, w in enumerate(wi.window_id)}
            tpl['w'] = tpl.window_id.map(wpos); tpl['l'] = tpl.link_id.map(P['li'])
            tts = pd.to_datetime(tpl.timestamp, utc=True)
            tpl['h'] = ((tts - tpl.window_id.map(org)).dt.total_seconds() // 300).astype(int)
            okr = tpl.w.notna() & tpl.l.notna() & tpl.h.between(1, F)
            assert okr.all(), 'unmapped template rows'
            for name, pr in preds.items():
                tpl[f'q_{name}'] = pr[tpl.w.astype(int), tpl.h.astype(int) - 1, tpl.l.astype(int)].astype(int)
            tpl['q_pvogfv'] = tpl['q_pvogf']                                       # pvogf: growth veto vs pvo (private ongoing windows)
            if split == 'private' and VETO['ok']:
                try:
                    tpl['q_pvogfv'], vinfo = veto_vs_pvo(tpl, preds['pvogf'], R['qlast'], is_og, wi)
                    VETO['info'].append(vinfo.assign(panel=panel))
                except Exception as ex:
                    VETO['ok'] = False; log('VETO vs pvo failed for', panel, type(ex).__name__, str(ex)[:200])
            tpl['is_og'] = tpl.window_id.map(dict(zip(wi.window_id, is_og))).astype(bool); tpl['split'] = split
            subs.append(tpl[['window_id', 'timestamp', 'link_id', 'is_og', 'split', 'q_pvogf', 'q_pvogf_npid', 'q_pvogfv']])
    S = pd.concat(subs, ignore_index=True)
    print(pd.DataFrame(stats).to_string(index=False), flush=True)
    # validation (public) ongoing rows: copied from the production v22 fs4 file, so the public score cannot change
    ref = pd.read_csv(TFB.kfile(REF_V22[0], REF_V22[1]),  # [TFB]
                      dtype={'window_id': str, 'link_id': str})
    S['ts'] = pd.to_datetime(S.timestamp, utc=True); ref['ts'] = pd.to_datetime(ref.timestamp, utc=True)
    S = S.merge(ref[['window_id', 'ts', 'link_id', 'queue_pred']].rename(columns={'queue_pred': 'q_fs4'}), on=['window_id', 'ts', 'link_id'],
                how='left', validate='one_to_one')
    assert S.q_fs4.notna().all()
    val = (S.split == 'validation').to_numpy()
    OUTS = ('pvogf', 'pvogf_npid') + (('pvogfv',) if VETO['ok'] else ())
    for name in OUTS:
        q = np.where(val, S.q_fs4.astype(int), S[f'q_{name}'])
        q = np.where(S.is_og.to_numpy(), q, 0)
        S[f'out_{name}'] = q.astype(int)
        S[['window_id', 'timestamp', 'link_id', f'out_{name}']].rename(columns={f'out_{name}': 'queue_pred'}).to_csv(
            f'{OUT}/queue_submission_{name}.csv', index=False)
        og = S.is_og.to_numpy()
        for sp_ in ('validation', 'private'):
            m = og & (S.split == sp_).to_numpy()
            a, b = S[f'out_{name}'].to_numpy()[m].astype(bool), S.q_fs4.to_numpy()[m].astype(bool)
            log(f'{name} vs fs4 on {sp_} ongoing rows: changed share {float((a != b).mean()):.5f} jaccard {float((a & b).sum() / max((a | b).sum(), 1)):.4f}'
                f' positives {int(a.sum())} vs {int(b.sum())}')
    log('queue rows', len(S), 'dupes', int(S.duplicated(['window_id', 'timestamp', 'link_id']).sum()))
    # pvogf CONTROL: every output vs tfb-t2ogpvaf queue_submission_pvo.csv, per split (validation rows: both are fs4af copies)
    try:
        ref = pvo_rows()
        if ref is not None:
            mm = S[['window_id', 'ts', 'link_id', 'split', 'is_og']].merge(ref, on=['window_id', 'ts', 'link_id'], how='left', validate='one_to_one')
            log(f'CONTROL pvo reference rows matched {int(mm.q_ref.notna().sum())} of {len(mm)}')
            for name in OUTS:
                for sp_ in ('validation', 'private'):
                    m = (mm.split == sp_).to_numpy() & mm.q_ref.notna().to_numpy()
                    a = S[f'out_{name}'].to_numpy()[m].astype(int); b = mm.q_ref.to_numpy()[m].astype(int)
                    wch = pd.Series(a != b).groupby(mm.window_id.to_numpy()[m]).any()
                    log(f'CONTROL {name} vs pvo | {sp_}: rows changed {int((a != b).sum())}/{len(a)} ({float((a != b).mean()):.4%}) | windows changed'
                        f' {int(wch.sum())}/{len(wch)} | positives {int(a.sum())} vs pvo {int(b.sum())}'
                        f' | Jaccard {float((a & b).sum() / max((a | b).sum(), 1)):.4f}')
        if VETO['info']:
            vi = pd.concat(VETO['info'], ignore_index=True); vi.to_csv(f'{OUT}/t2pvogf_veto_windows.csv', index=False)
            log(f'VETO (> {VETO_G} cells outside the queue at T vs pvo): private ongoing windows vetoed {int(vi.veto.sum())}/{len(vi)}'
                f' {sorted(vi.window_id[vi.veto].tolist())} | pvogf windows changed vs pvo {int((vi.cells_changed > 0).sum())}')
    except Exception as ex:
        log('CONTROL vs pvo skipped:', type(ex).__name__, str(ex)[:200])
    np.savez_compressed(f'{OUT}/t2pvogf_probs.npz', **PROBS)
    # CONTROL: the copied validation rows must equal the chain base's ongoing rows
    try:
        key = pd.read_csv(f'{REL}/submission_key.csv', dtype=str)
        kq = key[key.task == 'queue'][['submission_id', 'window_id', 'timestamp', 'link_id']].copy()
        kq['ts'] = pd.to_datetime(kq.timestamp, utc=True)
        base = pd.read_csv(TFB.kfile(REF_BASE[0], REF_BASE[1]),  # [TFB] CONTROL only (optional input)
                           usecols=['submission_id', 'task', 'queue_pred'])
        base = base[base.task == 'queue']; base['submission_id'] = base.submission_id.astype(str)
        m = S[S.is_og].merge(kq[['submission_id', 'window_id', 'ts', 'link_id']], on=['window_id', 'ts', 'link_id'], how='left')
        m = m.merge(base[['submission_id', 'queue_pred']], on='submission_id', how='left')
        for sp_ in ('validation', 'private'):
            mm = m[m.split == sp_]
            a, b = mm.out_pvogf.to_numpy().astype(bool), mm.queue_pred.fillna(-1).to_numpy() == 1
            log(f'CONTROL chain-base ongoing vs pvogf ({sp_}): agreement {float((a == b).mean()):.6f} changed share {float((a != b).mean()):.5f} rows {len(mm)}')
    except Exception as ex:
        log('CONTROL base skipped:', type(ex).__name__, str(ex)[:200])
    log('PVOGF DONE')


main()
