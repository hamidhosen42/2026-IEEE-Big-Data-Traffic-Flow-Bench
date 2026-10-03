"""t2 onset v19 (onset rows only): CatBoost onset bag + stage-2 stacking on the window probability profile.
Push as tfb-t2onv19 with the competition data + kernel source hosen42/tfb-t2v13 (used ONLY for the CONTROL print).

Stage 1 = the v13 production onset recipe (CatBoost depth 8 / 600 it / lr 0.08 / 4 threads on T+30 rows of onset windows only),
4-fold OOF on contiguous day blocks:
  A0     = v13 view (weekday-class priors), seed 0; final fit replicates v13's slot-T mask RNG (default_rng(len(trainw)),
           onset windows first) and its default_rng(0) row permutation -> must reproduce the tfb-t2v13 onset rows (CONTROL >= 0.999).
  A1, A2 = A0 view, seeds 1 / 2.
  B      = all-days prior (T+30) and all-days prior_now (T); no weekday-class prior/prior_now, no dq_hi30 / dq_lo30; seed 0.
  C      = B + the 18 hz_ hazard/gradient features (history and slot T only); seed 0.
  No oprior feature and no ramp features in any arm (ramp features hurt onset).
Stage 2 = LightGBM (31 leaves, min_data 100, lr 0.05, 300 rounds, 3 seeds) on the 24 travel-direction profile features of the window's
stage-1 probability vector, trained on pseudo-window OOF rows only. Final p = 0.3 * stage2 + 0.7 * stage1, expected-IoU top-m at T+30 only.
Local evidence (oof9o/t2oof/t2oofH caches, sklearn HGB stage 2, nested by fold): stack w0.3 on A0 +0.0033 +- 0.0013,
mean(A0,B,C) +0.0028 +- 0.0016, mean(A0,B,C) + stack w0.3 +0.0077 +- 0.0020 (4/4 folds), w0.5 +0.0097 +- 0.0025 (4/4).
Data rule: 12-slot history + masked slot T only (<= T); priors from train only.
Outputs: queue_submission_{ABC_st03,ABC_st05,A0_st03,A0_st05,ABC,A0}.csv (all queue rows of the template; onset windows filled at T+30,
every other row 0 -> swap with COND='queue_onset'), cv_onset_v19.csv (per arm x set: per-fold IoU, paired delta vs A0 with bootstrap
SD and z, windows better/worse), cv_onset_v19_windows.csv (per-window OOF IoU), onset_probs_v19.parquet (val/private probabilities).
SMOKE = True: 2 panels, 90 days, 50 CatBoost iterations, + the output shape checks (always on).
"""
import glob, os, time, gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier

T0 = time.time()
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
OUT = '/kaggle/working'
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']
WS = {'D7_I10_W', 'D7_I210_W', 'D7_I405_S', 'D12_I5_S'}       # travel direction: index + 1 = upstream (topology audit)
SMOKE = True
if SMOKE:
    PANELS = ['D7_I10_E', 'D12_I5_S']
H, F, SPACING = 12, 6, 72
MAX_OG = 300 if SMOKE else 3000                                  # only used to replicate the v13 slot-T mask RNG
CAT = dict(iterations=50 if SMOKE else 600, depth=8, learning_rate=0.08, thread_count=4, verbose=0)
ST2 = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.9,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4)
ST2_ROUNDS = 40 if SMOKE else 300
WEIGHTS = (0.3, 0.5)
OUT_VARIANTS = ['ABC_st03', 'ABC_st05', 'A0_st03', 'A0_st05', 'ABC', 'A0']     # written as queue_submission_<v>.csv
N_BOOT = 1000


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


# ------------------------------------------------------------------ data (verbatim from t2_model_v13.py)
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


def ffill_time(a):
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
    """v13 features (verbatim) + the t2_oofH hazard block under 'hz_' names (kept out of the A column list)."""
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
    # ---- hazard / gradient block (t2_oofH.py, history and slot T only), appended under hz_ names
    with np.errstate(all='ignore'):
        Fe['hz_vr_d1'] = last - vrf[:, -2]; Fe['hz_vr_d3'] = last - vrf[:, -4]
        Fe['hz_vr_acc'] = (last - vrf[:, -2]) - (vrf[:, -2] - vrf[:, -3])
        Fe['hz_vr_std3'] = np.nanstd(vr[:, -3:], 1); Fe['hz_vr_std6'] = np.nanstd(vr[:, -6:], 1); Fe['hz_vr_std12'] = np.nanstd(vr, 1)
        Fe['hz_fr_d1'] = frf[:, -1] - frf[:, -2]; Fe['hz_fr_std3'] = np.nanstd(fr[:, -3:], 1)
        kc = (P['cap'] / np.maximum(P['vf'], 1))[None, :]
        kk = (Fh[:, -1] / np.maximum(Vh[:, -1], 1)) / kc
        kk3 = np.nanmean(Fh[:, -3:] / np.maximum(Vh[:, -3:], 1), 1) / kc
        Fe['hz_k_rel'] = kk; Fe['hz_k_rel3'] = kk3
        Fe['hz_dv_hi'] = last - shift_links(last, 1); Fe['hz_dv_lo'] = last - shift_links(last, -1)
        Fe['hz_dk_hi'] = kk3 - shift_links(kk3, 1); Fe['hz_dk_lo'] = kk3 - shift_links(kk3, -1)
        up = np.stack([shift_links(frf[:, -1], s_) for s_ in (-1, -2, -3)]); dn = np.stack([shift_links(frf[:, -1], s_) for s_ in (1, 2, 3)])
        Fe['hz_fr_lo3'] = np.nanmean(up, 0); Fe['hz_fr_hi3'] = np.nanmean(dn, 0)
        up0 = np.stack([shift_links(frf[:, -4], s_) for s_ in (-1, -2, -3)]); dn0 = np.stack([shift_links(frf[:, -4], s_) for s_ in (1, 2, 3)])
        Fe['hz_fr_lo3_slope'] = Fe['hz_fr_lo3'] - np.nanmean(up0, 0); Fe['hz_fr_hi3_slope'] = Fe['hz_fr_hi3'] - np.nanmean(dn0, 0)
    return Fe, qlast


def prior_table(P, day_mask):
    out = []
    wkend = P['dow'] >= 5
    for m in (day_mask & ~wkend, day_mask & wkend, day_mask):
        E = P['E'][m]; Q = P['Q'][m]
        with np.errstate(all='ignore'):
            out.append(Q.sum(0) / np.maximum(E.sum(0), 1))
    return np.stack(out).astype(np.float32)          # [3, 288, L]


def decode(p):
    order = np.argsort(-p); ps = p[order]; cs = np.cumsum(ps); tot = cs[-1]
    m = np.arange(1, len(p) + 1)
    j = np.argmax(cs / (m + (tot - cs)))
    out = np.zeros(len(p), bool); out[order[:j + 1]] = True
    return out


# ------------------------------------------------------------------ onset rows (T+30 only)
def onset_feats(P, Fe, dow, tod, prior):
    """add h / priors exactly as v13 (A view) plus the all-days prior_now for the B/C view."""
    W, L = Fe['vr_last'].shape
    feats = dict(Fe); cls = (np.asarray(dow) >= 5).astype(int); slot = np.minimum(np.asarray(tod) + F, 287)
    feats['h'] = np.full((W, L), F, np.float32)
    feats['prior'], feats['prior_now'], feats['prior_alldays'] = prior[cls, slot], prior[cls, np.asarray(tod)], prior[2, slot]
    feats['prior_now_ad'] = prior[2, np.asarray(tod)]
    return feats


def train_rows(P, wins, prior, n_rng=None):
    """onset rows for windows `wins` (all onset). n_rng replicates the v13 slot-T mask RNG (default_rng(len(trainw)) over
    onset + ongoing windows, onset first) so that A0 reproduces the COMBO15 onset model exactly."""
    d = np.array([w[0] for w in wins]); t = np.array([w[1] for w in wins])
    idx = t[:, None] + np.arange(-H, 0)[None]
    Vh = P['V'][d[:, None], idx]; Fh = P['Fq'][d[:, None], idx]
    n = n_rng if n_rng is not None else len(wins)
    rng = np.random.default_rng(n)
    mask = rng.random((n, P['V'].shape[2]))[:len(wins)] < 0.62
    VT = np.where(mask, P['V'][d, t], np.nan)
    Fe, _ = window_features(P, Vh, Fh, t, P['dow'][d], prior, VT)
    feats = onset_feats(P, Fe, P['dow'][d], t, prior)
    lab = P['Q'][d, np.minimum(t + F, 287)]; ev = P['E'][d, np.minimum(t + F, 287)]
    return feats, lab, ev, d, t


def mat(feats, cols):
    return np.stack([np.asarray(feats[c]) for c in cols], -1).astype(np.float32)        # [W, L, f]


def view_cols(feats):
    a = [c for c in feats if not c.startswith('hz_') and c != 'prior_now_ad']
    assert a[-4:] == ['h', 'prior', 'prior_now', 'prior_alldays'], a[-6:]
    b = [c for c in a if c not in ('prior', 'prior_now', 'prior_alldays', 'dq_hi30', 'dq_lo30')] + ['prior_alldays', 'prior_now_ad']
    c = b + [x for x in feats if x.startswith('hz_')]
    return {'A': a, 'B': b, 'C': c}


MODELS = [('A0', 'A', 0), ('A1', 'A', 1), ('A2', 'A', 2), ('B', 'B', 0), ('C', 'C', 0)]      # (arm, view, CatBoost seed)


def fit_stage1(blocks, cols, seed):
    """blocks: list of (feats, lab, ev); rows = eligible cells, window-major / link-minor, panels in PANELS order;
    shuffled with default_rng(0) exactly like v13 (_sub = choice(len, min(len, 1.2M), replace=False))."""
    X = np.concatenate([mat(f, cols).reshape(-1, len(cols))[e.ravel()] for f, l, e in blocks])
    y = np.concatenate([l.ravel()[e.ravel()] for f, l, e in blocks]).astype(int)
    sub = np.random.default_rng(0).choice(len(X), min(len(X), 1_200_000), replace=False)
    m = CatBoostClassifier(random_seed=seed, **CAT); m.fit(X[sub], y[sub])
    return m


def predict_stage1(m, feats, cols):
    M = mat(feats, cols); W, L, k = M.shape
    return m.predict_proba(M.reshape(-1, k))[:, 1].reshape(W, L).astype(np.float32)


# ------------------------------------------------------------------ stage 2: window probability profile
def shift0(a, k):
    out = np.zeros_like(a)
    if k > 0: out[:-k] = a[k:]
    elif k < 0: out[-k:] = a[:k]
    else: out[:] = a
    return out


def profile(p, ws):
    q = p[::-1] if ws else p
    L = len(q); qc = np.clip(q, 1e-5, 1 - 1e-5); lg = np.log(qc / (1 - qc))
    srt = np.sort(q)[::-1]; am = int(np.argmax(q))
    rank = np.empty(L); rank[np.argsort(-q)] = np.arange(L)
    on = q >= 0.05; run_len = np.zeros(L); run_sum = np.zeros(L); run_max = np.zeros(L); i = 0
    while i < L:
        if on[i]:
            j = i
            while j < L and on[j]:
                j += 1
            run_len[i:j] = j - i; run_sum[i:j] = q[i:j].sum(); run_max[i:j] = q[i:j].max(); i = j
        else:
            i += 1
    cols = [q, lg]
    for k in (1, 2, 3):
        cols += [shift0(q, k), shift0(q, -k)]
    cols += [np.max([shift0(q, k) for k in (-2, -1, 1, 2)], 0), np.sum([shift0(q, k) for k in (-3, -2, -1, 1, 2, 3)], 0),
             np.full(L, srt[0]), np.full(L, srt[1] if L > 1 else 0), np.full(L, q.sum()),
             np.full(L, (q > 0.5).sum()), np.full(L, (q > 0.1).sum()), np.full(L, (q > 0.02).sum()),
             rank, q / max(srt[0], 1e-6), np.arange(L) - am, np.abs(np.arange(L) - am), run_len, run_sum, run_max,
             np.full(L, float(ws))]
    M = np.stack(cols, 1).astype(np.float32)
    return M[::-1] if ws else M


def stage2_fit(P1, LAB, EV, PAN):
    X = np.concatenate([profile(p, pn in WS) for p, pn in zip(P1, PAN)])
    y = np.concatenate(LAB).astype(np.float32); e = np.concatenate(EV)
    return [lgb.train(dict(ST2, seed=s, bagging_seed=s + 10, feature_fraction_seed=s + 20), lgb.Dataset(X[e], y[e]), ST2_ROUNDS)
            for s in (0, 1, 2)]


def stage2_predict(models, p, ws):
    M = profile(p, ws)
    return np.mean([m.predict(M) for m in models], 0).astype(np.float32)




# v13 onset column order (64; generated from t2_model_v13.window_features + h/prior/prior_now/prior_alldays): hard gate for arm A
V13_COLS = ['vr_last', 'vr_raw_last', 'vr_l1', 'vr_l2', 'vr_l3', 'vr_l6', 'vr_l11', 'vr_min3', 'vr_mean12', 'vr_slope', 'fr_last',
            'fr_mean3', 'fr_slope', 'q_last', 'q_count', 'q_age', 'miss_last', 'vr_T', 'has_T', 'vr_n-3', 'q_n-3', 'fr_n-3', 'vr_n-2',
            'q_n-2', 'fr_n-2', 'vr_n-1', 'q_n-1', 'fr_n-1', 'vr_n+1', 'q_n+1', 'fr_n+1', 'vr_n+2', 'q_n+2', 'fr_n+2', 'vr_n+3', 'q_n+3',
            'fr_n+3', 'vr_min_dn1_3', 'vr_min_up1_3', 'vr_min_dn4_8', 'vr_min_up4_8', 'vr_min_dn9_16', 'vr_min_up9_16', 'dq_hi', 'dq_lo',
            'dq_hi15', 'dq_lo15', 'dq_hi30', 'dq_lo30', 'share_q', 'share_q15', 'n_q_hist_win', 'pos', 'lanes', 'cap_lane',
            'cap_ratio_hi', 'cap_ratio_lo', 'length', 'tod', 'dow', 'h', 'prior', 'prior_now', 'prior_alldays']


def check_cols(cols):
    assert cols['A'] == V13_COLS, 'arm A columns differ from v13'
    assert len(cols['B']) == len(V13_COLS) - 3 and 'prior' not in cols['B'] and 'dq_hi30' not in cols['B'] and 'dq_lo30' not in cols['B']
    assert cols['B'][-2:] == ['prior_alldays', 'prior_now_ad']
    hz = cols['C'][len(cols['B']):]
    assert cols['C'][:len(cols['B'])] == cols['B'] and len(hz) == 18 and all(c.startswith('hz_') for c in hz)
    assert not any('ramp' in c or 'oprior' in c for v in cols.values() for c in v)
    log('columns: A', len(cols['A']), '| B', len(cols['B']), '| C', len(cols['C']))


# ------------------------------------------------------------------ windows
panels = {}
for panel in PANELS:
    P = load_panel(panel)
    tw = build_windows(P, global_spacing=False, dense_ongoing=True)
    og = [w for w in tw if w[2] == 'queue_ongoing']; on = [w for w in tw if w[2] == 'queue_onset']
    P['on'] = on; P['n_rng'] = len(on) + min(len(og), MAX_OG)           # v13: trainw = on + og (og subsampled to MAX_OG)
    wi = pd.read_csv(f'{REL}/task2/{panel}/train/window_index.csv')
    P['official'] = [(P['days'].index(pd.Timestamp(r.forecast_origin).strftime('%Y-%m-%d')),
                      pd.Timestamp(r.forecast_origin).hour * 12 + pd.Timestamp(r.forecast_origin).minute // 5, r.condition)
                     for r in wi.itertuples() if pd.Timestamp(r.forecast_origin).strftime('%Y-%m-%d') in P['days']
                     and r.condition == 'queue_onset']
    D = len(P['days']); P['block'] = np.minimum(np.arange(D) * 4 // D, 3)
    log(panel, 'onset windows', len(on), 'official onset', len(P['official']), 'rng n', P['n_rng'])
    panels[panel] = P

# ------------------------------------------------------------------ 4-fold OOF stage 1 (train onset windows + official)
oof = []          # dicts: panel, set, fold, d, t, probs{arm: [L]}, lab [L], ev [L]
cols = None
for k in range(4):
    blocks = []; evalsets = []
    for panel, P in panels.items():
        prior = prior_table(P, P['block'] != k)
        trw = [w for w in P['on'] if P['block'][w[0]] != k]
        f, l, e, _, _ = train_rows(P, trw, prior); blocks.append((f, l, e))
        if cols is None:
            cols = view_cols(f); check_cols(cols)
        for tag, ws_ in (('pseudo', [w for w in P['on'] if P['block'][w[0]] == k]),
                         ('official', [w for w in P['official'] if P['block'][w[0]] == k])):
            if ws_:
                evalsets.append((panel, tag) + train_rows(P, ws_, prior))
    models = {nm: fit_stage1(blocks, cols[v], s) for nm, v, s in MODELS}
    log(f'fold {k}: stage-1 models trained on {sum(int(e.sum()) for _, _, e in blocks)} rows')
    for panel, tag, f, l, e, d, t in evalsets:
        pr = {nm: predict_stage1(models[nm], f, cols[v]) for nm, v, s in MODELS}
        for i in range(len(d)):
            oof.append(dict(panel=panel, set=tag, fold=k, d=int(d[i]), t=int(t[i]), lab=l[i], ev=e[i],
                            probs={nm: pr[nm][i] for nm in pr}))
    del blocks, models, evalsets; gc.collect()

SRC = {'A0': lambda r: r['A0'], 'A1': lambda r: r['A1'], 'A2': lambda r: r['A2'], 'B': lambda r: r['B'], 'C': lambda r: r['C'],
       'A3': lambda r: (r['A0'] + r['A1'] + r['A2']) / 3,
       'ABC': lambda r: (r['A0'] + r['B'] + r['C']) / 3,
       'A3BC': lambda r: (r['A0'] + r['A1'] + r['A2'] + r['B'] + r['C']) / 5}
ST_SRC = ('A0', 'ABC')                                          # stage 2 is fitted on these stage-1 sources

# nested stage 2 (fit on pseudo OOF rows of the other folds) for the CV table
p2_oof = {s: [None] * len(oof) for s in ST_SRC}
for k in range(4):
    tr = [r for r in oof if r['set'] == 'pseudo' and r['fold'] != k]
    for s in ST_SRC:
        ms = stage2_fit([SRC[s](r['probs']) for r in tr], [r['lab'] for r in tr], [r['ev'] for r in tr], [r['panel'] for r in tr])
        for i, r in enumerate(oof):
            if r['fold'] == k:
                p2_oof[s][i] = stage2_predict(ms, SRC[s](r['probs']), r['panel'] in WS)
log('nested stage 2 done')


def variants_of(probs, p2):
    v = {nm: f(probs) for nm, f in SRC.items()}
    for s in ST_SRC:
        for w in WEIGHTS:
            v[f'{s}_st{int(w * 10):02d}'] = w * p2[s] + (1 - w) * SRC[s](probs)
    return v


rows = []
for i, r in enumerate(oof):
    for nm, p in variants_of(r['probs'], {s: p2_oof[s][i] for s in ST_SRC}).items():
        rows.append(dict(panel=r['panel'], set=r['set'], fold=r['fold'], d=r['d'], t=r['t'], variant=nm,
                         iou=iou(r['lab'], decode(p), r['ev'])))
cv = pd.DataFrame(rows); cv.to_csv(f'{OUT}/cv_onset_v19_windows.csv', index=False)


def agg(df):
    """official aggregation: window -> panel -> family -> mean (NaN if empty)."""
    if len(df) == 0:
        return np.nan
    g = df.groupby('panel').iou.mean().reset_index(); g['fam'] = g.panel.str.rsplit('_', n=1).str[0]
    return g.groupby('fam').iou.mean().mean()


def agg_fast(iou_v, pan_ix, fam_of_pan, n_pan, n_fam):
    """vectorised agg() for the bootstrap: iou_v [B, n], pan_ix [B, n] -> [B]."""
    B = iou_v.shape[0]; out = np.zeros(B)
    s = np.zeros((B, n_pan)); c = np.zeros((B, n_pan))
    for j in range(n_pan):
        m = pan_ix == j; s[:, j] = (iou_v * m).sum(1); c[:, j] = m.sum(1)
    pm = s / np.maximum(c, 1)
    for b in range(B):
        ok = c[b] > 0
        fm = [pm[b, ok & (fam_of_pan == f)].mean() for f in range(n_fam) if (ok & (fam_of_pan == f)).any()]
        out[b] = np.mean(fm)
    return out


summ = []
base = cv[cv.variant == 'A0'].reset_index(drop=True)
for nm in cv.variant.unique():
    x = cv[cv.variant == nm].reset_index(drop=True)
    assert (x[['panel', 'set', 'fold', 'd', 't']].values == base[['panel', 'set', 'fold', 'd', 't']].values).all()
    for st in ('pseudo', 'official'):
        xs, bs = x[x.set == st].reset_index(drop=True), base[base.set == st].reset_index(drop=True)
        if len(xs) == 0:
            continue
        pans = sorted(xs.panel.unique()); fams = sorted({p.rsplit('_', 1)[0] for p in pans})
        pix = xs.panel.map({p: i for i, p in enumerate(pans)}).to_numpy()
        fop = np.array([fams.index(p.rsplit('_', 1)[0]) for p in pans])
        rng = np.random.default_rng(0)
        ix = np.stack([np.concatenate([rng.choice(np.where(pix == j)[0], (pix == j).sum()) for j in range(len(pans))]) for _ in range(N_BOOT)])
        dlt = (agg_fast(xs.iou.to_numpy()[ix], pix[ix], fop, len(pans), len(fams))
               - agg_fast(bs.iou.to_numpy()[ix], pix[ix], fop, len(pans), len(fams)))
        d_all = agg(xs) - agg(bs); sd = float(np.std(dlt))
        row = dict(arm=nm, set=st, n_windows=len(xs), iou=agg(xs), iou_A0=agg(bs), delta=d_all, boot_sd=sd,
                   z=d_all / sd if sd > 0 else np.nan,
                   better=int((xs.iou.to_numpy() > bs.iou.to_numpy() + 1e-9).sum()), worse=int((xs.iou.to_numpy() < bs.iou.to_numpy() - 1e-9).sum()))
        for f in range(4):
            row[f'iou_f{f}'] = agg(xs[xs.fold == f]); row[f'delta_f{f}'] = agg(xs[xs.fold == f]) - agg(bs[bs.fold == f])
        summ.append(row)
summ = pd.DataFrame(summ); summ.to_csv(f'{OUT}/cv_onset_v19.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
for st in ('pseudo', 'official'):
    t_ = summ[summ.set == st]
    if len(t_):
        print(f'\n===== CV onset IoU ({st}; nested stage 2; paired vs A0, bootstrap {N_BOOT} within panel) =====')
        print(t_[['arm', 'n_windows', 'iou', 'delta', 'boot_sd', 'z', 'better', 'worse'] + [f'delta_f{f}' for f in range(4)]
                 + [f'iou_f{f}' for f in range(4)]].round(4).to_string(index=False), flush=True)

# ------------------------------------------------------------------ final stage 1 on all onset windows, stage 2 on all pseudo OOF
blocks = []
for panel, P in panels.items():
    P['prior_all'] = prior_table(P, np.ones(len(P['days']), bool))
    f, l, e, _, _ = train_rows(P, P['on'], P['prior_all'], n_rng=P['n_rng']); blocks.append((f, l, e))
final = {nm: fit_stage1(blocks, cols[v], s) for nm, v, s in MODELS}
del blocks; gc.collect()
pk = [r for r in oof if r['set'] == 'pseudo']
st2 = {s: stage2_fit([SRC[s](r['probs']) for r in pk], [r['lab'] for r in pk], [r['ev'] for r in pk], [r['panel'] for r in pk]) for s in ST_SRC}
log('final stage-1 and stage-2 models trained (stage 2 on', len(pk), 'pseudo OOF windows)')

subs = []; prob_rows = []; fp = {}; n_on_total = 0
for panel, P in panels.items():
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
        tod = (org.dt.hour * 12 + org.dt.minute // 5).to_numpy(); dow = org.dt.dayofweek.to_numpy()
        VT = np.full((W, L), np.nan, np.float32)
        for date, grp in wi.groupby('date'):
            fs = glob.glob(f'{REL}/corridors/{panel}/{split}/mainline_states_masked/*/*{date.replace("-", "_")}.parquet')
            if not fs:
                continue
            m = pd.read_parquet(fs[0], columns=['timestamp', 'link_id', 'speed_kmh', 'is_score_eligible'])
            m = m[(m.is_score_eligible == 1) & m.speed_kmh.notna()]
            for r in grp.itertuples():
                x = m[m.timestamp == r.forecast_origin]                        # slot T itself: allowed (<= T)
                x = x[x.link_id.isin(P['li'])].groupby('link_id').speed_kmh.mean()
                VT[wpos[r.window_id], x.index.map(P['li']).to_numpy()] = x.to_numpy()
        log(panel, split, 'slot-T visible share', round(float(np.isfinite(VT).mean()), 3),
            'history filled share', round(float(np.isfinite(Vh).mean()), 3))
        Fe, _ = window_features(P, Vh, Fh, tod, dow, P['prior_all'], VT)
        feats = onset_feats(P, Fe, dow, tod, P['prior_all'])
        pr = {nm: predict_stage1(final[nm], feats, cols[v]) for nm, v, s in MODELS}
        is_on = (wi.condition == 'queue_onset').to_numpy(); n_on_total += int(is_on.sum())
        preds = {}
        for i in np.where(is_on)[0]:
            probs = {nm: pr[nm][i] for nm in pr}
            p2 = {s: stage2_predict(st2[s], SRC[s](probs), panel in WS) for s in ST_SRC}
            for nm, p in variants_of(probs, p2).items():
                preds.setdefault(nm, np.zeros((W, L), bool))[i] = decode(p)
            prob_rows.append(dict(panel=panel, split=split, window_id=wi.window_id.iloc[i],
                                  **{f'p_{nm}': probs[nm].tolist() for nm in probs}, **{f'p2_{s}': p2[s].tolist() for s in ST_SRC}))
        if not preds:                                                       # no onset windows in this panel/split
            preds = {nm: np.zeros((W, L), bool) for nm in ['A0'] + OUT_VARIANTS}
        # validation / private footprint vs A0 (decoded masks over all links at T+30 of onset windows)
        for nm, pm in preds.items():
            dif = (pm != preds['A0'])[is_on]
            a = fp.setdefault((split, nm), [0, 0, 0]); a[0] += int(dif.any(1).sum()); a[1] += int(dif.sum()); a[2] += int(is_on.sum())
        tpl['w'] = tpl.window_id.map(wpos); tpl['l'] = tpl.link_id.map(P['li'])
        tts = pd.to_datetime(tpl.timestamp, utc=True)
        tpl['h'] = ((tts - tpl.window_id.map(org)).dt.total_seconds() // 300).astype(int)
        okr = tpl.w.notna() & tpl.l.notna() & (tpl.h == F)                  # onset truth lives at T+30 only
        tpl['allow'] = okr & tpl.window_id.map(dict(zip(wi.window_id, is_on))).fillna(False).astype(bool)
        for nm, pm in preds.items():
            col = f'queue_pred_{nm}'; tpl[col] = 0
            tpl.loc[okr, col] = pm[tpl.w[okr].astype(int), tpl.l[okr].astype(int)].astype(int)
        log(panel, split, 'windows', W, 'onset windows', int(is_on.sum()), 'template rows', len(tpl), 'unmapped', int((~okr & (tpl.h == F)).sum()),
            {nm: int(tpl[f'queue_pred_{nm}'].sum()) for nm in OUT_VARIANTS})
        subs.append(tpl[['window_id', 'timestamp', 'link_id', 'allow'] + [f'queue_pred_{nm}' for nm in preds]])
S = pd.concat(subs, ignore_index=True)
KEYS = ['window_id', 'timestamp', 'link_id']
assert not S.duplicated(KEYS).any(), 'duplicate queue keys'
for nm in OUT_VARIANTS:
    S[KEYS + [f'queue_pred_{nm}']].rename(columns={f'queue_pred_{nm}': 'queue_pred'}).to_csv(f'{OUT}/queue_submission_{nm}.csv', index=False)
pd.DataFrame(prob_rows).to_parquet(f'{OUT}/onset_probs_v19.parquet', index=False)

print('\n===== validation footprint vs A0 (onset windows changed / cells changed at T+30, of N onset windows) =====')
for split in ('validation', 'private'):
    for nm in sorted({k[1] for k in fp}):
        a = fp[(split, nm)]
        print(f'{split:10s} {nm:9s} windows changed {a[0]:4d} / {a[2]:4d} | cells changed {a[1]:5d}', flush=True)

# ------------------------------------------------------------------ hard gates: every output has the template's shape
for nm in OUT_VARIANTS:
    o = pd.read_csv(f'{OUT}/queue_submission_{nm}.csv', dtype=str)
    assert list(o.columns) == KEYS + ['queue_pred'], (nm, list(o.columns))
    assert len(o) == len(S) and (o[KEYS].values == S[KEYS].values).all(), (nm, 'keys differ from the template')
    assert o.queue_pred.notna().all() and o.queue_pred.isin(['0', '1']).all(), (nm, 'queue_pred not binary')
    qp = o.queue_pred.astype(int).to_numpy()
    assert (qp[~S.allow.to_numpy()] == 0).all(), (nm, 'positives outside onset windows / T+30')
    npos = o.assign(q=qp)[S.allow.to_numpy()].groupby('window_id').q.sum()
    log(f'gate OK queue_submission_{nm}.csv rows {len(o)} positives {int(qp.sum())} onset windows with 0 positives {int((npos == 0).sum())}')
pq = pd.read_parquet(f'{OUT}/onset_probs_v19.parquet')
assert len(pq) == n_on_total, ('onset_probs rows', len(pq), n_on_total)
assert all(len(a) == len(panels[p]['links']) for a, p in zip(pq.p_A0, pq.panel))
cvs = pd.read_csv(f'{OUT}/cv_onset_v19.csv')
assert set(cvs.arm) == set(cv.variant.unique()) and set(OUT_VARIANTS) <= set(cvs.arm)
log('gate OK onset_probs_v19.parquet', pq.shape, '| cv_onset_v19.csv', cvs.shape, '| template rows', len(S))

# ------------------------------------------------------------------ CONTROL: A0 must reproduce the tfb-t2v13 onset rows (>= 0.999)
v13 = glob.glob('/kaggle/input/**/tfb-t2v13/**/queue_submission.csv', recursive=True)
if v13:
    q = pd.read_csv(v13[0], dtype={'window_id': str, 'link_id': str}); q['timestamp'] = pd.to_datetime(q.timestamp, utc=True)
    a = S[KEYS + ['queue_pred_A0', 'allow']].copy(); a['timestamp'] = pd.to_datetime(a.timestamp, utc=True)
    onw = set(pd.concat([pd.read_csv(f, dtype=str) for f in glob.glob(f'{REL}/task2/*/*/window_index.csv') if '/train/' not in f])
              .query('condition == "queue_onset"').window_id)
    mm = a[a.window_id.isin(onw)].merge(q, on=KEYS, how='left', validate='one_to_one')
    assert mm.queue_pred.notna().all(), 'v13 file misses onset keys'
    agree = float((mm.queue_pred_A0 == mm.queue_pred).mean())
    t30 = mm[mm.allow]; agree30 = float((t30.queue_pred_A0 == t30.queue_pred).mean())
    nwin = int(t30.assign(dd=(t30.queue_pred_A0 != t30.queue_pred)).groupby('window_id').dd.any().sum())
    log('CONTROL A0 vs tfb-t2v13 onset rows agreement', round(agree, 6), '| T+30 rows only', round(agree30, 6),
        '| windows differing', nwin, '| positives A0 / v13', int(mm.queue_pred_A0.sum()), int(mm.queue_pred.sum()))
    log('CONTROL', 'PASS' if agree >= 0.999 else 'FAIL (do not submit; A0 does not reproduce v13)', '(smoke runs are expected to FAIL)' if SMOKE else '')
else:
    log('CONTROL SKIPPED: tfb-t2v13 output not attached')
log('done')
