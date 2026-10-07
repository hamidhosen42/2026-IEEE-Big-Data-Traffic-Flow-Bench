"""v25ck = v22af (F03 bag: np511 + id511, seeds 0 and 1, ALL-FINITE inputs) + SAME-DAY CONTEXT ("episode clock") features
for the ongoing model (track ogclock).  16 columns appended to both np and id: per link ck_run / ck_first / ck_clear / ck_frac /
ck_minr / ck_nep (current queued run, first queued observation today, last queued->free transition, share of today's observed
slots queued, minimum speed ratio today, episodes today) and per corridor cc_first / cc_peak / cc_peak_sh / cc_since_peak /
cc_rel / cc_d6 / cc_d12 / cc_d24 / cc_age / cc_area (first slot with >= 2 queued links, peak queued-link count so far, time since
the peak, count at T / peak, 30 / 60 / 120-min count change, corridor episode age, queued link-slots today).
Inputs (all timestamped <= T): slots < T-12 = the same-day Task-1-MASKED layer (released windows: mainline_states_masked of the
window's day; train rows: the train cube with eligible cells thinned i.i.d. at a per-day visible share drawn from the measured
released regimes 0.507 / 0.705 / 0.803 (shares 0.297 / 0.359 / 0.344); ineligible-valued cells are never masked); slots T-12..T-1
= window_history (train: raw history, as production); slot T = the masked slot-T value (train: the production 62 % draw).
Lab (track ogclock; F03 rows, 255 leaves / lr 0.05 / 500 trees, eligible-only inputs): train OOF official-like windows (n 2,992)
np +0.0065 (CI +0.0045..+0.0085, 7/8 panels), id +0.0054 (+0.0035..+0.0074, 5/8); pseudo-March offlike_m (n 298) np +0.0030
(-0.0050..+0.0109), id -0.0017 (-0.0102..+0.0065), np+id blend +0.0012 (-0.0068..+0.0087); the new-scenario gain sits on mid-run
windows (+0.017..+0.022, CI > 0) and comes from the corridor group.  Not measured: ALL-FINITE inputs together with ck columns.
Variant fs4ck -> queue_submission_fs4ck.csv.  CONTROL: footprint vs the attached tfb-t2v22 queue_submission_fs4.csv.
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
ALLFIN = True                                # features from all finite released values (labels / selection: eligible cells)
CKF = True                                   # same-day context features (track ogclock)
CK_REGIMES, CK_REGIME_P = (0.507, 0.705, 0.803), (0.297, 0.359, 0.344)   # measured per-day visible share of the masked layer


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
    if CKF:
        P['THIN'] = ck_thin(panel, D, L)
        log(panel, 'ck: emulated masked-layer share of eligible cells', round(float(P['THIN'][E].mean()), 4))
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


# ------------------------------------------------------------------ same-day context features (track ogclock)
# ============================================================================================ kernel-portable block
def ck_ffill(a):
    """forward fill NaN along axis 1 of [W, S, L]."""
    out = a.copy()
    for i in range(1, a.shape[1]):
        m = np.isnan(out[:, i]); out[:, i][m] = out[:, i - 1][m]
    return out


def ck_bfill(a):
    out = a.copy()
    for i in range(a.shape[1] - 2, -1, -1):
        m = np.isnan(out[:, i]); out[:, i][m] = out[:, i + 1][m]
    return out


def ck_features(Rr, T):
    """Rr [W, S, L] speed ratio V / v_cut of the day so far (NaN = not visible; every slot > T must be NaN); T [W] origin slots.
    Returns dict name -> [W, L] float32 (CK_ALL order)."""
    Rr = np.asarray(Rr, np.float32); T = np.asarray(T, np.int64)
    W, S, L = Rr.shape
    Tf = T.astype(np.float32)[:, None]
    S3 = np.arange(S)[None, :, None]
    vis = np.isfinite(Rr)
    q = vis & (np.nan_to_num(Rr, nan=9.0) <= 1.0)
    st = np.where(vis, q.astype(np.float32), np.nan).astype(np.float32)
    f = ck_ffill(st); b = ck_bfill(st)
    nanc = np.full((W, 1, L), np.nan, np.float32)
    prev = np.concatenate([nanc, f[:, :-1]], 1); nxt = np.concatenate([b[:, 1:], nanc], 1)
    q2 = q | (vis & ~q & (prev == 1) & (nxt == 1))                       # isolated free observation inside a queued run: closed
    del f, b, prev, nxt
    st2 = np.where(vis, q2.astype(np.float32), np.nan).astype(np.float32)
    f2 = ck_ffill(st2)
    prev2 = np.concatenate([nanc, f2[:, :-1]], 1)
    out = {}
    nobs = vis.sum(1); nqo = q.sum(1)
    with np.errstate(all='ignore'):
        mn = np.where(vis, Rr, np.inf).min(1)
        anyq = q.any(1); fq = np.argmax(q, 1)
        lfree = np.where(vis & ~q2, S3, -1).max(1)
        fo = np.where(vis.any(1), np.argmax(vis, 1), -1)
        last = f2[:, -1]
        start = np.where(lfree >= 0, lfree, fo - 1)
        ltr = np.where(vis & ~q2 & (prev2 == 1), S3, -1).max(1)
        out['ck_run'] = np.where(last == 1, Tf - start, np.where(np.isnan(last), np.nan, 0.0))
        out['ck_first'] = np.where(anyq, Tf - fq, np.nan)
        out['ck_clear'] = np.where(ltr >= 0, Tf - ltr, np.nan)
        out['ck_frac'] = np.where(nobs > 0, nqo / np.maximum(nobs, 1), np.nan)
        out['ck_minr'] = np.where(np.isfinite(mn), mn, np.nan)
        out['ck_nep'] = (vis & q2 & (prev2 != 1)).sum(1).astype(np.float32)
        # corridor: per slot, the number of links whose latest (closed) observed state is queued
        nqs = (f2 == 1).sum(2).astype(np.float32)                         # [W, S]
        sl = np.arange(S)[None, :]
        valid = sl <= T[:, None]
        ar = np.arange(W)
        nT = nqs[ar, T]
        nqv = np.where(valid, nqs, -1.0)
        peak = nqv.max(1)
        lastpk = S - 1 - np.argmax((nqv == peak[:, None])[:, ::-1], 1)
        m2 = (nqs >= 2) & valid
        first2 = np.where(m2.any(1), np.argmax(m2, 1), -1)
        z = nqs == 0
        zp = np.concatenate([np.ones((W, 2), bool), z], 1)
        z3 = zp[:, 2:] & zp[:, 1:-1] & zp[:, :-2]
        lz = np.where(z3 & valid, sl, -1).max(1)
        m1 = (nqs >= 1) & valid
        first1 = np.where(m1.any(1), np.argmax(m1, 1), 0)
        cc = {'cc_first': np.where(first2 >= 0, T - first2, np.nan), 'cc_peak': peak, 'cc_peak_sh': peak / L,
              'cc_since_peak': (T - lastpk).astype(np.float32), 'cc_rel': nT / np.maximum(peak, 1)}
        for k in (6, 12, 24):
            cc[f'cc_d{k}'] = nT - nqs[ar, np.maximum(T - k, 0)]
        cc['cc_age'] = np.where(nT > 0, np.where(lz >= 0, T - lz, T - first1 + 1), 0.0)
        cc['cc_area'] = np.where(valid, nqs, 0).sum(1) / L
    for k, v in cc.items():
        out[k] = np.repeat(np.asarray(v, np.float32)[:, None], L, 1)
    return {k: np.asarray(out[k], np.float32) for k in CK_ALL}


def ck_compose(Vday, T, Vh, VT, cut, vis_mask=None):
    """day-so-far ratio cube for windows: Vday [W, S, L] speed of the window's day (NaN = not visible) used for slots < T-12
    (optionally thinned by vis_mask [W, S, L] bool), Vh [W, 12, L] raw history for T-12..T-1, VT [W, L] slot-T value."""
    W, S, L = Vday.shape
    s = np.arange(S)[None, :, None]; Tt = np.asarray(T)[:, None, None]
    pre = Vday if vis_mask is None else np.where(vis_mask, Vday, np.nan)
    R = np.where(s < Tt - H, pre, np.nan).astype(np.float32)
    ar = np.arange(W)
    for j in range(H):
        R[ar, np.asarray(T) - H + j] = Vh[:, j]
    R[ar, np.asarray(T)] = VT
    return R / np.asarray(cut, np.float32)[None, None, :]


CK_LINK = ['ck_run', 'ck_first', 'ck_clear', 'ck_frac', 'ck_minr', 'ck_nep']
CK_CORR = ['cc_first', 'cc_peak', 'cc_peak_sh', 'cc_since_peak', 'cc_rel', 'cc_d6', 'cc_d12', 'cc_d24', 'cc_age', 'cc_area']
CK_ALL = CK_LINK + CK_CORR


def ck_thin(panel, D, L):
    """[D, 288, L] bool: emulated masked-layer visibility of the train days (per-day regime, i.i.d. cells)."""
    pi = PANELS.index(panel) if panel in PANELS else 0
    rates = np.random.default_rng(7000 + pi).choice(np.array(CK_REGIMES), D, p=np.array(CK_REGIME_P))
    out = np.zeros((D, 288, L), bool)
    for d in range(D):
        out[d] = np.random.default_rng(10 ** 6 * (pi + 1) + d).random((288, L)) < rates[d]
    return out


def ck_train(P, d, t, Vh, VT, chunk=192):
    """ck features of train windows: eligible cells before T-12 thinned by P['THIN'], ineligible-valued cells always visible."""
    V = P.get('Vf', P['V']); L = V.shape[2]
    out = {k: np.empty((len(d), L), np.float32) for k in CK_ALL}
    for a in range(0, len(d), chunk):
        sl = slice(a, a + chunk); dd = d[sl]
        pre = np.where(P['E'][dd] & ~P['THIN'][dd], np.nan, V[dd]).astype(np.float32)
        X = ck_features(ck_compose(pre, t[sl], Vh[sl], VT[sl], P['cut']), t[sl])
        for k in CK_ALL:
            out[k][sl] = X[k]
    return out


def ck_day_cube(m, li, L):
    """[288, L] speed of one masked-layer day file (already filtered to the released cells used as features)."""
    ts = pd.to_datetime(m.timestamp, utc=True)
    x = m.assign(t=(ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(), l=m.link_id.astype(str).map(li)).dropna(subset=['l'])
    g = x.groupby(['t', 'l']).speed_kmh.mean().reset_index()
    C = np.full((288, L), np.nan, np.float32)
    C[g.t.to_numpy().astype(int), g.l.to_numpy().astype(int)] = g.speed_kmh.to_numpy()
    return C


def ck_split(P, wi, DAY, tod, Vh, VT, chunk=192):
    """ck features of the released windows: the masked layer of the window's day (slots < T-12), window_history, slot-T value."""
    W, L = VT.shape
    out = {k: np.empty((W, L), np.float32) for k in CK_ALL}
    dates = wi.date.astype(str).to_numpy(); empty = np.full((288, L), np.nan, np.float32)
    for a in range(0, W, chunk):
        sl = slice(a, a + chunk)
        pre = np.stack([DAY.get(dt_, empty) for dt_ in dates[sl]])
        X = ck_features(ck_compose(pre, tod[sl], Vh[sl], VT[sl], P['cut']), tod[sl])
        for k in CK_ALL:
            out[k][sl] = X[k]
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
    Vh = P.get('Vf', P['V'])[d[:, None], idx]; Fh = P.get('Ff', P['Fq'])[d[:, None], idx]
    # slot T itself is only visible through the Task-1-masked layer (~50-66% of cells): simulate that
    rng = np.random.default_rng(len(wins))
    VT = np.where(rng.random((len(d), P['V'].shape[2])) < 0.62, P['V'][d, t], np.nan)
    if 'Vf' in P:                                                          # ineligible-valued cells are never Task-1-masked
        VT = np.where(P['E'][d, t], VT, P['Vf'][d, t])
    Fe, qlast = window_features(P, Vh, Fh, t, P['dow'][d], prior, VT)
    add_ramp(Fe, P['ON'], P['OFF'], d, t, P['cap'])
    CK = ck_train(P, d, t, Vh, VT) if CKF else {}
    W, L = qlast.shape
    rows = []
    for h in range(1, F + 1):
        slot = np.minimum(t + h, 287)
        feats = {k: v for k, v in Fe.items()}
        feats['h'] = np.full((W, L), h, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = prior_feats(prior, P['dow'][d], t, h)
        feats.update(CK)
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
    DAY = {}
    for date, grp in wi.groupby('date'):
        fs = glob.glob(f'{REL}/corridors/{panel}/{split}/mainline_states_masked/*/*{date.replace("-", "_")}.parquet')
        if not fs:
            continue
        m = pd.read_parquet(fs[0], columns=['timestamp', 'link_id', 'speed_kmh', 'is_score_eligible'])
        m = m[((m.is_score_eligible == 1) | ALLFIN) & m.speed_kmh.notna()]
        if CKF:
            DAY[date] = ck_day_cube(m, P['li'], L)
        for r in grp.itertuples():
            x = m[m.timestamp == r.forecast_origin]
            x = x[x.link_id.isin(P['li'])].groupby('link_id').speed_kmh.mean()
            VT[wpos[r.window_id], x.index.map(P['li']).to_numpy()] = x.to_numpy()
    log(panel, split, 'slot-T visible share', round(float(np.isfinite(VT).mean()), 3),
        'history filled share', round(float(np.isfinite(Vh).mean()), 3))
    Fe, qlast = window_features(P, Vh, Fh, tod, dow, P['prior_all'], VT)
    CK = ck_split(P, wi, DAY, tod, Vh, VT) if CKF else {}
    if CKF:
        log(panel, split, 'ck: masked-layer days', len(DAY), '| ck_run > 0 share', round(float((CK['ck_run'] > 0).mean()), 3),
            '| cc_peak mean', round(float(CK['cc_peak'][:, 0].mean()), 2))
    wdays = sorted(wi.date.unique())
    ONs, OFFs = ramp_cube(panel, split, P['li'], wdays)
    add_ramp(Fe, ONs, OFFs, wi.date.map({dd: i for i, dd in enumerate(wdays)}).to_numpy(), tod, P['cap'])
    log(panel, split, 'ramp on3 finite share', round(float(np.isfinite(Fe['ramp_on3']).mean()), 3))
    R = dict(d=np.zeros(W, int), t=tod, cond=wi.condition.to_numpy(), qlast=qlast, W=W, L=L, rows=[])
    for h in range(1, F + 1):
        feats = dict(Fe); feats['h'] = np.full((W, L), h, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = prior_feats(P['prior_all'], dow, tod, h)
        feats.update(CK)
        R['rows'].append((h, feats, None, None))
    return wi, tpl, R


VARIANTS = {'fs2ck': ('fs_np', 'fs_id'),                               # seed-0 pair (all-finite inputs + ck columns)
            'fs4ck': ('fs_np', 'fs_id', 'fs_np1', 'fs_id1'),           # F03 bag of 4, all-finite inputs + ck columns (candidate)
            'fs_npck': ('fs_np',)}


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
            a, b = pr[i], preds[next(iter(preds))][i]
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
        b0 = g.variant.iloc[0]
        log(f'FOOTPRINT {split} (vs {b0}): ongoing windows {g[g.variant == b0].shape[0]} | mean queued cells at T (q_last) '
            f'{g[g.variant == b0].size_T.mean():.2f}')
        for v, x in g.groupby('variant', sort=False):
            log(f'  {v:8s} windows changed {int(x.changed.sum())}/{len(x)} | mean Jaccard vs {b0} {x.jacc_cur2.mean():.4f}'
                f' | cells changed per step h1..h6 {[int(x[f"chg_h{h}"].sum()) for h in hs]} (added {[int(x[f"add_h{h}"].sum()) for h in hs]})'
                f' | mean predicted size per step {[round(float(x[f"size_h{h}"].mean()), 2) for h in hs]}')
        pv = g.pivot_table(index='panel', columns='variant', values='changed', aggfunc='sum')
        log(f'  windows changed per panel ({split}):', pv.to_dict('index'))
    return fp


def control_check(S, names):
    """FOOTPRINT of every variant vs tfb-t2v22/queue_submission_fs4.csv (eligible-only production rows), by split."""
    ref = sorted(glob.glob('/kaggle/input/**/tfb-t2v22/**/queue_submission_fs4.csv', recursive=True))
    if not ref:
        log('FOOTPRINT: skipped (tfb-t2v22 queue_submission_fs4.csv not attached)'); return None
    r = pd.read_csv(ref[0], dtype={'window_id': str, 'timestamp': str, 'link_id': str})
    m = S.merge(r.rename(columns={'queue_pred': 'ref'}), on=['window_id', 'timestamp', 'link_id'], how='left')
    m = m[m.is_og & m.ref.notna()]
    m['split'] = np.where(m.window_id.isin(VAL_IDS), 'validation', 'private')
    for name in names:
        for split, x in m.groupby('split'):
            a = x[f'queue_pred_{name}'].to_numpy().astype(int); b = x.ref.to_numpy().astype(int)
            chg = x.assign(c=(a != b)).groupby('window_id').c.any()
            log(f'FOOTPRINT {name} vs tfb-t2v22 fs4 | {split}: windows changed {int(chg.sum())}/{len(chg)} | rows changed {int((a != b).sum())}/{len(a)}'
                f' ({(a != b).mean():.4%}) | Jaccard of queued cells {(a & b).sum() / max((a | b).sum(), 1):.4f} | positives {int(a.sum())} ref {int(b.sum())}')
    return None


VAL_IDS = set()


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
    st = pd.DataFrame(stats); st.to_csv(f'{OUT}/t2v25ck_trainsets.csv', index=False)
    log('TRAIN SETS total:', st.drop(columns=['panel']).sum().to_dict())
    for k, nm in (('cur', 'CUR ongoing'), ('rs', 'run-starts'), ('rest', 'F03 rest')):
        log(f'filled persistence ALL panels | {nm}:', pq(np.concatenate([P['pers_sets'][k] for P in panels.values()])))

    # ---- training rows (ongoing only)
    cols = None; CUR = []; EX = []
    for panel, P in panels.items():
        cols, cur, ex = panel_rows(P, cols)
        log(panel, f'rows CUR {len(cur["y"])} (pos {cur["y"].mean():.3f}) | EX {len(ex["y"])} (pos {ex["y"].mean() if len(ex["y"]) else float("nan"):.3f};'
            f' run-start rows {int(ex["rs"].sum())}, pos {ex["y"][ex["rs"]].mean() if ex["rs"].any() else float("nan"):.3f})')
        CUR.append(cur); EX.append(ex)
        for k in ('V', 'Fq', 'E', 'Q', 'ON', 'OFF', 'cand', 'rs', 'pers', 'Vf', 'Ff', 'THIN'):       # only network vectors + prior are needed from here on
            P.pop(k, None)
        gc.collect()
    cat = lambda L_, k: np.concatenate([x[k] for x in L_])
    Xc, yc, rc = cat(CUR, 'X'), cat(CUR, 'y').astype(np.float32), cat(CUR, 'rs')
    Xe, ye, re_, ie = cat(EX, 'X'), cat(EX, 'y').astype(np.float32), cat(EX, 'rs'), cat(EX, 'incur')
    del CUR, EX; gc.collect()
    cols_np = [x for x in cols if x not in PRIOR_COLS]; ix_np = [cols.index(x) for x in cols_np]
    cols_id = [x for x in cols if x not in ID_DROP]; ix_id = [cols.index(x) for x in cols_id]
    w_f = np.where(re_, W_RS, W_DENSE).astype(np.float32)
    hm = re_ & ~ie                                                         # run-start rows whose window is not already in CUR
    y_h = np.concatenate([yc, ye[hm]]); w_h = np.concatenate([np.where(rc, W_RS, W_DENSE), np.full(int(hm.sum()), W_RS)]).astype(np.float32)
    log(f'features: all {len(cols)}, np {len(cols_np)}, id {len(cols_id)} | ck columns in np: {[c for c in cols_np if c in CK_ALL]}')
    log(f'POSITIVE RATES | CUR rows {len(yc)} pos {yc.mean():.4f} | F03 rows {len(ye)} pos {ye.mean():.4f} weighted {np.average(ye, weights=w_f) if len(ye) else float("nan"):.4f}'
        f' (run-start rows {int(re_.sum())} pos {ye[re_].mean() if re_.any() else float("nan"):.4f}; rest pos {ye[~re_].mean() if (~re_).any() else float("nan"):.4f})'
        f' | H rows {len(y_h)} pos {y_h.mean():.4f} weighted {np.average(y_h, weights=w_h):.4f} (CUR run-start rows {int(rc.sum())}, added run-start rows {int(hm.sum())})')
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
    del Xc, Xe; gc.collect()

    # ---- released windows
    subs = []; foot = []; probs = {}; names = None
    for panel, P in panels.items():
        for split in ('validation', 'private'):
            tpl, names = predict_split(P, panel, split, models, foot, probs)
            if split == 'validation':
                VAL_IDS.update(tpl.window_id.unique().tolist())
            subs.append(tpl)
    S = pd.concat(subs, ignore_index=True)
    for name in names:
        S[['window_id', 'timestamp', 'link_id', f'queue_pred_{name}']].rename(columns={f'queue_pred_{name}': 'queue_pred'}).to_csv(
            f'{OUT}/queue_submission_{name}.csv', index=False)
    np.savez_compressed(f'{OUT}/t2v25ck_probs.npz', **probs)
    log('queue rows', len(S), 'dupes', int(S.duplicated(['window_id', 'timestamp', 'link_id']).sum()), 'ongoing rows', int(S.is_og.sum()),
        'files', [f'queue_submission_{n}.csv' for n in names], '| onset rows are 0 in every file')
    fp = print_footprint(foot)
    if len(fp):
        fp.to_csv(f'{OUT}/t2v25ck_footprint.csv', index=False)
    control_check(S, names)
    log('done')


main()
