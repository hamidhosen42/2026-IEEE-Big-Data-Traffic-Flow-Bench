"""TRACK ogfeat: new FEATURES for the ongoing (F03) model.

Everything is computed from the SAME inputs as production window_features: the 60-min history cubes Vh, Fh [W, 12, L],
the Task-1-masked slot-T value VT [W, L], static link vectors and the production ramp columns (slots T-3..T-1).
Nothing after T is read.  og_extra() + og_step() are written to be pasted into the Kaggle kernel unchanged.

Groups
  g1a  tod, dow                      (production columns that neither np nor id contains)
  g1b  c_*   corridor context: share of the corridor queued at T, T-15, T-30, T-45, T-60, slopes, churn, age of the visible queue
  g2   k_*   per-link kinematics: run length of the current state, transitions, signed depth to the queue edge on each side
             now / 15 / 30 min ago (closed state), edge displacement, cluster size, local queued counts;
       kx_*  step-dependent: signed depth extrapolated with the edge displacement to step h
  g3   e_*   demand / supply at the ends of the reference cluster (own cluster or nearest cluster): inflow / outflow flow ratio
             and speed beyond tail and head with 15-min trends, cluster mean speed and trend, ramp inflow at tail / head
Cache: data/t2local/cache/ogfeat/fe_<PANEL>__<set>.npy  [W, L, K] float32 aligned with lab.rows(panel, set).win
"""
import gc, json, os, sys, time, types, warnings
import numpy as np
import pandas as pd
import lab
import exp_firstcross_lib as fc

H, F = lab.H, lab.F
CAPD = 30.0
BIG = 10 ** 6
ODIR = f'{lab.CACHE}/ogfeat'
RES = f'{lab.ROOT}/analysis/t2lab/results/ogfeat'
for _d in (ODIR, RES):
    os.makedirs(_d, exist_ok=True)


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
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
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
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
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


OG_STEP = ['kx_lo3', 'kx_hi3', 'kx_lo6', 'kx_hi6', 'kx_min3']


def og_step(S, h):
    """step-dependent columns from the static ones (S: name -> array) and the step h (array or scalar, 1..6)."""
    h = np.asarray(h, np.float32)
    lo3 = S['k_sdlo0'] + h / 3.0 * S['k_vlo3']; hi3 = S['k_sdhi0'] + h / 3.0 * S['k_vhi3']
    return {'kx_lo3': lo3, 'kx_hi3': hi3, 'kx_lo6': S['k_sdlo0'] + h / 6.0 * S['k_vlo6'], 'kx_hi6': S['k_sdhi0'] + h / 6.0 * S['k_vhi6'],
            'kx_min3': np.minimum(lo3, hi3)}
# ============================================================================================ end of portable block


_NAMES = None


def static_names():
    global _NAMES
    if _NAMES is None:
        f = f'{ODIR}/names.json'
        if os.path.exists(f):
            _NAMES = json.load(open(f))
        else:
            raise FileNotFoundError('run ogfeat_build.py first')
    return list(_NAMES)


def static_names2():
    f = f'{ODIR}/names2.json'
    return json.load(open(f)) if os.path.exists(f) else []


def group_cols(g):
    """extra column names of a group: (production columns to add, static extra names, step extra names)."""
    n = static_names()
    if g == 'g1a':
        return ['tod', 'dow'], [], []
    if g == 'tod':
        return ['tod'], [], []
    if g == 'g1b':
        return [], [x for x in n if x.startswith('c_')], []
    if g == 'g2':
        return [], [x for x in n if x.startswith('k_')], list(OG_STEP)
    if g == 'g2s':                                   # g2 without the step-dependent extrapolations
        return [], [x for x in n if x.startswith('k_')], []
    if g == 'g3':
        return [], [x for x in n if x.startswith('e_')], []
    if g == 'g1c':
        return [], [x for x in static_names2() if x.startswith('c2_')], []
    if g == 'g2b':
        return [], [x for x in static_names2() if x.startswith('k2_')], []
    if g.startswith('cols:'):                        # explicit list  cols:a+b+c
        want = g[5:].split('+'); st = [x for x in want if x in n or x in static_names2()]; sp = [x for x in want if x in OG_STEP]
        pr = [x for x in want if x in lab.all_cols()]
        assert len(st) + len(sp) + len(pr) == len(want), want
        return pr, st, sp
    raise KeyError(g)


def spec_cols(base, groups):
    """(production columns, static extra names, step names) for base 'np' | 'id' and a list of groups."""
    pc = lab.colset(base); st, sp = [], []
    for g in groups:
        a, b, c = group_cols(g)
        pc = pc + [x for x in a if x not in pc]; st += [x for x in b if x not in st]; sp += [x for x in c if x not in sp]
    need = {'k_sdlo0', 'k_sdhi0', 'k_vlo3', 'k_vhi3', 'k_vlo6', 'k_vhi6'}
    return pc, st, sp


# ------------------------------------------------------------------ train-split inputs (same slot-T draw as production)
def train_inputs(P, win):
    d = win.d.to_numpy(); t = win.t.to_numpy()
    idx = t[:, None] + np.arange(-H, 0)[None]
    Vh = P['V'][d[:, None], idx]; Fh = P['Fq'][d[:, None], idx]
    rng = np.random.default_rng(len(win))
    VT = np.where(rng.random((len(d), P['V'].shape[2])) < 0.62, P['V'][d, t], np.nan)
    return Vh, Fh, VT, d, t


def build_fe(panel, set_name, check=True, refresh=False):
    """[W, L, K] static extra features of the ongoing windows of a lab window set (cached)."""
    f = f'{ODIR}/fe_{panel}__{set_name}.npy'
    if os.path.exists(f) and not refresh:
        return np.load(f, mmap_mode='r')
    rc = lab.rows(panel, set_name, lab.ONGOING)
    P = lab.load_panel(panel)
    Vh, Fh, VT, d, t = train_inputs(P, rc.win)
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        Fe, qlast = lab.window_features(P, Vh, Fh, t, P['dow'][d], lab.prior_table(P, np.ones(len(P['days']), bool)), VT)
        lab.add_ramp(Fe, P['ON'], P['OFF'], d, t, P['cap'])
    if check:                                        # my inputs == the inputs behind the cached production rows
        assert np.array_equal(qlast, rc.qlast), (panel, set_name, 'qlast differs')
        m = rc.h == 1
        for c in ('vr_T', 'vr_last', 'dq_hi', 'fr_last', 'ramp_on3', 'vr_l11', 'share_q15'):
            a = np.asarray(Fe[c], np.float32)[rc.wi[m], rc.li[m]]; b = np.asarray(rc.X[:, rc.cols.index(c)])[m]
            assert np.array_equal(a, b, equal_nan=True), (panel, set_name, c)
    parts = []                                       # windows are independent: chunked to keep the float64 temporaries small
    for a in range(0, len(d), 1024):
        sl = slice(a, a + 1024)
        parts.append(og_extra(P, Vh[sl], Fh[sl], VT[sl], {k: Fe[k][sl] for k in ('ramp_on3', 'ramp_off3', 'ramp_on_surge')}))
    X = {k: np.concatenate([p_[k] for p_ in parts]) for k in parts[0]}
    names = list(X)
    if check:                                        # my filled state at T == production qlast
        assert np.array_equal(X['c_n0'][:, 0], qlast.sum(1).astype(np.float32)), (panel, set_name, 'state at T differs from qlast')
    nf = f'{ODIR}/names.json'
    if os.path.exists(nf):
        assert json.load(open(nf)) == names, 'feature list changed: delete the ogfeat cache'
    else:
        json.dump(names, open(nf, 'w'))
    FE = np.stack([X[k] for k in names], -1).astype(np.float32)
    tmp = f'{f}.{os.getpid()}.tmp.npy'
    np.save(tmp, FE); os.replace(tmp, f)                                   # atomic: concurrent readers never see a partial file
    return np.load(f, mmap_mode='r')


def build_fe2(panel, set_name, refresh=False):
    """[W, L, K2] round-2 static features of a lab window set (cached as fe2_<PANEL>__<set>.npy)."""
    f = f'{ODIR}/fe2_{panel}__{set_name}.npy'
    if os.path.exists(f) and not refresh:
        return np.load(f, mmap_mode='r')
    rc = lab.rows(panel, set_name, lab.ONGOING)
    P = lab.load_panel(panel)
    Vh, Fh, VT, d, t = train_inputs(P, rc.win)
    parts = [og_extra2(P, Vh[a:a + 1024], Fh[a:a + 1024], VT[a:a + 1024], None) for a in range(0, len(d), 1024)]
    X = {k: np.concatenate([p_[k] for p_ in parts]) for k in parts[0]}
    nf = f'{ODIR}/names2.json'
    if os.path.exists(nf):
        assert json.load(open(nf)) == list(X), 'round-2 feature list changed: delete fe2_* and names2.json'
    else:
        json.dump(list(X), open(nf, 'w'))
    tmp = f'{f}.{os.getpid()}.tmp.npy'
    np.save(tmp, np.stack([X[k] for k in X], -1).astype(np.float32)); os.replace(tmp, f)
    return np.load(f, mmap_mode='r')


class FE2:
    """pair of static caches (round 1, round 2) addressed by column name."""
    def __init__(self, fe1, fe2):
        self.fe1, self.fe2 = fe1, fe2


def extra_matrix(FE, wi, li, h, st, sp):
    if isinstance(FE, FE2):                                              # columns may come from either cache
        n1 = static_names(); s1 = [c for c in st if c in n1]; s2 = [c for c in st if c not in n1]
        A = extra_matrix1(FE.fe1, wi, li, h, s1, sp)
        if not s2:
            return A
        n2 = static_names2(); ix = np.array([n2.index(c) for c in s2])
        B = np.asarray(FE.fe2)[np.asarray(wi)[:, None], np.asarray(li)[:, None], ix[None, :]]
        out = np.empty((len(wi), len(st) + len(sp)), np.float32)
        for j, c in enumerate(st):
            out[:, j] = A[:, s1.index(c)] if c in s1 else B[:, s2.index(c)]
        out[:, len(st):] = A[:, len(s1):]
        return out
    return extra_matrix1(FE, wi, li, h, st, sp)


def extra_matrix1(FE, wi, li, h, st, sp):
    """[n, len(st) + len(sp)] extra columns for rows (wi, li, h)."""
    names = static_names()
    out = np.empty((len(wi), len(st) + len(sp)), np.float32)
    if not len(wi) or not (st or sp):
        return out
    need = list(dict.fromkeys(list(st) + (['k_sdlo0', 'k_sdhi0', 'k_vlo3', 'k_vhi3', 'k_vlo6', 'k_vhi6'] if sp else [])))
    ix = np.array([names.index(c) for c in need])
    FEa = np.asarray(FE)
    E = FEa[np.asarray(wi)[:, None], np.asarray(li)[:, None], ix[None, :]]
    for j, c in enumerate(st):
        out[:, j] = E[:, need.index(c)]
    if sp:
        S = og_step({c: E[:, need.index(c)] for c in need}, h)
        for j, c in enumerate(sp):
            out[:, len(st) + j] = S[c]
    return out


def get_fe(panel, set_name, st):
    """round-1 cache, or the pair of caches when a round-2 column is requested."""
    n1 = static_names()
    if any(c not in n1 for c in st):
        return FE2(build_fe(panel, set_name), build_fe2(panel, set_name))
    return build_fe(panel, set_name)


def full_matrix(rc, FE, mask, pc, st, sp, k):
    r = np.where(mask)[0]
    Xp = rc.matrix(mask, pc, k)
    if not (st or sp):
        return Xp
    Xe = extra_matrix(FE, rc.wi[r], rc.li[r], rc.h[r], st, sp)
    return np.concatenate([Xp, Xe], 1)


def train_matrix(k, pc, st, sp, spec=None, panels=None):
    """F03 rows (same rows / order / weights as exp_firstcross_lib.train_matrix) with the extra columns appended."""
    spec = spec or fc.ARMS['F03']
    parts = []; n = 0
    for panel in panels or lab.PANELS:
        for name, sw in spec['train']:
            rc = lab.rows(panel, name, lab.ONGOING)
            if not rc.n:
                continue
            ww = np.full(len(rc.win), float(sw), np.float32)
            if spec['wfn'] is not None:
                ww = ww * np.asarray(spec['wfn'](rc.win, name, panel), np.float32)
            m = rc.e & rc.keep5[:, k] & (rc.fold != k) & (ww[rc.wi] > 0)
            c = int(m.sum()); parts.append((panel, name, rc, m, ww, c)); n += c
    nc = len(pc) + len(st) + len(sp)
    X = np.empty((n, nc), np.float32); y = np.empty(n, np.float32); w = np.empty(n, np.float32); o = 0
    for panel, name, rc, m, ww, c in parts:
        FE = get_fe(panel, name, st) if (st or sp) else None
        X[o:o + c] = full_matrix(rc, FE, m, pc, st, sp, k); y[o:o + c] = rc.y[m]; w[o:o + c] = ww[rc.wi[m]]; o += c
    return X, y, w


def predict_set(models, panel, set_name, k, pc, st, sp, widx=None):
    """P6 [W, 6, L] for the fold-k windows of a lab evaluation set (as lab.predict_set, extra columns appended)."""
    rc = lab.rows(panel, set_name, lab.ONGOING)
    P6 = np.zeros((len(rc.win), F, rc.L), np.float32)
    if not rc.n:
        return rc, P6
    wmask = (rc.win.fold.to_numpy() == k) if widx is None else np.isin(np.arange(len(rc.win)), widx)
    m = rc.keep5[:, k] & wmask[rc.wi]
    if m.any():
        FE = get_fe(panel, set_name, st) if (st or sp) else None
        X = full_matrix(rc, FE, m, pc, st, sp, k)
        ms = models if isinstance(models, (list, tuple)) else [models]
        p = sum(mdl.predict(X, num_threads=lab.THREADS) for mdl in ms) / len(ms)
        P6[rc.wi[m], rc.h[m] - 1, rc.li[m]] = p
    return rc, P6


# ------------------------------------------------------------------ pseudo-March (masked validation month, new scenario)
PM_SETS = ('offlike_m', 'runstart_m')


def val_panel(panel):
    P = lab.load_panel(panel)
    z = np.load(P['npz'], allow_pickle=False)
    days = np.array([str(x) for x in z['validation_days']]); o = np.argsort(days)
    V = z['validation_Vm'][o].astype(np.float32); Fq = z['validation_Fm'][o].astype(np.float32); Em = z['validation_Em'][o]
    E = np.isfinite(V); Q = E & (V <= P['cut'][None, None, :])
    Pv = dict(panel=panel, links=P['links'], li=P['li'], cut=P['cut'], vf=P['vf'], cap=P['cap'], lanes=P['lanes'], length=P['length'],
              days=list(days[o]), dow=pd.to_datetime(pd.Series(days[o])).dt.dayofweek.to_numpy(), V=V, Fq=Fq, E=E, Q=Q, Em=Em,
              ON=z['validation_ON'][o].astype(np.float32), OFF=z['validation_OFF'][o].astype(np.float32))
    return P, Pv


def pm_select(Pv):
    """copy of exp_firstcross_pmarch.select: faithful ongoing candidates on the masked validation month (blank-touching dropped)."""
    from exp_firstcross_prod_snippet import fill_time36
    V, E, Q, Em = Pv['V'], Pv['E'], Pv['Q'], Pv['Em']; D, _, L = V.shape
    VF = fill_time36(V); MF = np.isfinite(VF); QF = MF & (VF <= Pv['cut'][None, None, :])
    blank = (Em.sum(2) >= 5) & (E.sum(2) == 0)
    og = np.zeros((D, 288), bool); on = og.copy(); pers = np.full((D, 288), np.nan)
    for d in range(D):
        for t in range(H, 288 - F):
            if blank[d, t - H:t + F + 1].any() or Em[d, t - H:t].mean() < 0.7:
                continue
            if lab.established(Q[d, t - H:t]):
                Tf = QF[d, t + 1:t + 1 + F]; Mf = MF[d, t + 1:t + 1 + F]
                if not Tf.any():
                    continue
                S = np.repeat(QF[d, t][None], F, 0)
                u = ((Tf | S) & Mf).sum(); i = (Tf & S & Mf).sum()
                pers[d, t] = 1.0 if u == 0 else i / u
                og[d, t] = pers[d, t] <= 0.9
            else:
                on[d, t] = Q[d, t + 1:t + 1 + F].any()
    return on, og, pers


def build_pm(panel, refresh=False):
    """pseudo-March ongoing rows of one panel for PM_SETS: production matrix (all columns, all-train prior), meta, static
    extras, truth / ev / qlast, window frame.  Cached as one npz + pkl per (panel, set)."""
    out = {}
    if all(os.path.exists(f'{ODIR}/pm_{panel}__{s}.npz') for s in PM_SETS) and not refresh:
        for s in PM_SETS:
            z = np.load(f'{ODIR}/pm_{panel}__{s}.npz'); out[s] = dict({k: z[k] for k in z.files}, win=pd.read_pickle(f'{ODIR}/pm_{panel}__{s}.pkl'))
        return out
    P, Pv = val_panel(panel)
    on, og, pers = pm_select(Pv)
    D = len(Pv['days'])
    g_on, g_og = lab._glist(on), lab._glist(og); sset = set()
    for d0 in range(D):
        _, b = lab.greedy(g_on, g_og, 'interleaved', start_day=d0); sset.update(b)
    dd, tt = np.where(og); rs = ~og[dd, tt - 1]
    sets = {'offlike_m': sorted(sset), 'runstart_m': sorted(zip(dd[rs].tolist(), tt[rs].tolist()))}
    prior = lab.prior_table(P, np.ones(len(P['days']), bool))
    vis_day = Pv['E'].sum((1, 2)) / np.maximum(Pv['Em'].sum((1, 2)), 1)
    for s, wl in sets.items():
        d = np.array([w[0] for w in wl]); t = np.array([w[1] for w in wl])
        idx = t[:, None] + np.arange(-H, 0)[None]
        Vh = Pv['V'][d[:, None], idx]; Fh = Pv['Fq'][d[:, None], idx]; VT = Pv['V'][d, t]
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            Fe, qlast = lab.window_features(P, Vh, Fh, t, Pv['dow'][d], prior, VT)
            lab.add_ramp(Fe, Pv['ON'], Pv['OFF'], d, t, P['cap'])
        W, L = qlast.shape
        R = dict(d=d, t=t, cond=np.array([lab.ONGOING] * W), qlast=qlast, W=W, L=L, rows=[])
        for h in range(1, F + 1):
            feats = dict(Fe); feats['h'] = np.full((W, L), h, np.float32)
            feats['prior'], feats['prior_now'], feats['prior_alldays'] = lab.prior_feats(prior, Pv['dow'][d], t, h)
            R['rows'].append((h, feats, Pv['Q'][d, np.minimum(t + h, 287)], Pv['E'][d, np.minimum(t + h, 287)]))
        X, y, e, meta, call = lab.to_matrix(R, lab.ONGOING, None, lab.ongoing_keep(R))
        assert list(call) == lab.all_cols()
        EX = og_extra(P, Vh, Fh, VT, Fe)
        assert list(EX) == static_names()
        FE = np.stack([EX[k] for k in EX], -1).astype(np.float32)
        win = pd.DataFrame(dict(d=d, t=t, cond=lab.ONGOING, fold=0, pers_fill=pers[d, t], runstart=og[d, t] & ~og[d, t - 1], faithful=og[d, t],
                                vis=vis_day[d], hist_vis=[float(Pv['E'][a, b - H:b].sum() / max(Pv['Em'][a, b - H:b].sum(), 1)) for a, b in zip(d, t)]))
        dat = dict(X=X, meta=meta.astype(np.int32), FE=FE, truth=np.stack([Pv['Q'][d, t + h] for h in range(1, F + 1)], 1),
                   ev=np.stack([Pv['E'][d, t + h] for h in range(1, F + 1)], 1), qlast=qlast)
        tmp = f'{ODIR}/pm_{panel}__{s}.{os.getpid()}.tmp.npz'
        win.to_pickle(f'{ODIR}/pm_{panel}__{s}.pkl'); np.savez(tmp, **dat); os.replace(tmp, f'{ODIR}/pm_{panel}__{s}.npz')
        out[s] = dict(dat, win=win)
    return out


def build_pm2(panel, refresh=False):
    """round-2 static features of the cached pseudo-March sets: {set: [W, L, K2]} (cached as pm2_<PANEL>__<set>.npy)."""
    out = {}
    if all(os.path.exists(f'{ODIR}/pm2_{panel}__{s}.npy') for s in PM_SETS) and not refresh:
        return {s: np.load(f'{ODIR}/pm2_{panel}__{s}.npy') for s in PM_SETS}
    P, Pv = val_panel(panel); pm = build_pm(panel)
    for s in PM_SETS:
        win = pm[s]['win']; d = win.d.to_numpy(); t = win.t.to_numpy()
        idx = t[:, None] + np.arange(-H, 0)[None]
        X = og_extra2(P, Pv['V'][d[:, None], idx], Pv['Fq'][d[:, None], idx], Pv['V'][d, t], None)
        assert list(X) == static_names2()
        tmp = f'{ODIR}/pm2_{panel}__{s}.{os.getpid()}.tmp.npy'
        np.save(tmp, np.stack([X[k] for k in X], -1).astype(np.float32)); os.replace(tmp, f'{ODIR}/pm2_{panel}__{s}.npy')
        out[s] = np.load(f'{ODIR}/pm2_{panel}__{s}.npy')
    return out


def pm_fe(panel, s, z, st):
    """static cache object of a pseudo-March set for the requested columns."""
    n1 = static_names()
    if any(c not in n1 for c in st):
        return FE2(z['FE'], build_pm2(panel)[s])
    return z['FE']


def pm_score(models, pc, st, sp, arm):
    """score models (list -> mean of probabilities) on the cached pseudo-March sets; returns the per-window frame."""
    res = []; allc = lab.all_cols(); ix = [allc.index(c) for c in pc]
    ms = models if isinstance(models, (list, tuple)) else [models]
    for panel in lab.PANELS:
        pm = build_pm(panel)
        for s in PM_SETS:
            z = pm[s]; meta = z['meta']
            if not len(z['win']):
                continue
            X = z['X'][:, ix]
            if st or sp:
                X = np.concatenate([X, extra_matrix(pm_fe(panel, s, z, st), meta[:, 0], meta[:, 1], meta[:, 2], st, sp)], 1)
            P6 = np.zeros((len(z['win']), F, z['qlast'].shape[1]), np.float32)
            P6[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = sum(m.predict(X, num_threads=lab.THREADS) for m in ms) / len(ms)
            rc = types.SimpleNamespace(truth=z['truth'], ev=z['ev'], qlast=z['qlast'], win=z['win'], name=s, panel=panel, cond=lab.ONGOING,
                                       L=z['qlast'].shape[1])
            df = lab.score_probs(rc, P6, arm=arm, set_name=s)
            df['vis'] = z['win'].vis.to_numpy(); df['hist_vis'] = z['win'].hist_vis.to_numpy()
            res.append(df)
    return pd.concat(res, ignore_index=True)


# ------------------------------------------------------------------ statistics
def delta_row(A, B, label, set_name):
    """paired delta B - A on one evaluation set: diff, CI, n, panels positive."""
    a, b = fc.paired(A, B)
    if not len(a):
        return dict(cmp=label, set=set_name, n=0)
    bt = lab.paired_bootstrap(a, b)
    pa = a.groupby('panel').iou.mean(); pb = b.groupby('panel').iou.mean()
    r = dict(cmp=label, set=set_name, n=bt['n_windows'], base=lab.aggregate(a), new=lab.aggregate(b), diff=bt['diff'], lo=bt['lo'], hi=bt['hi'],
             p_pos=bt['p_pos'], panels_pos=f'{int((pb - pa > 0).sum())}/{len(pa)}')
    for hh in (1, 3, 6):
        r[f'd_s{hh}'] = lab.aggregate(b, f's{hh}') - lab.aggregate(a, f's{hh}')
    for c in ('fn_new', 'fp_old', 'fp_new'):
        r['d_' + c] = lab.aggregate(b, c) - lab.aggregate(a, c)
    return r
