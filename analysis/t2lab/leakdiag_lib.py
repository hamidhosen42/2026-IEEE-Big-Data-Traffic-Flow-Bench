"""DIAGNOSTIC - NOT FOR SUBMISSION (analysis/LEAK_DIAG_2026-10-29.md).

This module deliberately reads Task 2 data timestamped AFTER the forecast origin T:
  * on-ramp flows inside the horizon T+5..T+30 (and its +-5 min neighbours), and
  * mainline cells after the 60-min buffer (T+95..T+105).
The organizer rule (forum 742068, research/organizer_verified_facts.md item 1) puts both OUT OF SCOPE for Task 2.
The only purpose is to measure how much such a leak could be worth (does it explain the 0.93 public scores?).
NEVER import this module from production, kernel or consolidation code.
"""
import numpy as np
import pandas as pd

import lab

H, F = lab.H, lab.F
PMIN = 10.0                    # a ramp cell counts only where its same-slot 'on' level is >= 10 vph (night zeros are noise)
THR = 0.3                      # ramp "switched off": flow <= 0.3 x same-slot 'on' level (public notebook: 0.2 x, 81 % exact zeros)
PROF_Q = 90                    # v3: the 'on' level is the 90th percentile over days (v2 used the median with PMIN 50: at recurrently
                               # queued slots the median is ~0 because the ramp is off on most days, so exactly the informative
                               # cells were dropped; many on-ramps also run at 10-30 vph)
VIS_PB = 0.52 / 0.764          # thin observed train cells to the val/private masked-layer visibility after the buffer (0.52)
PB0, PB1 = 19, 21              # first visible slots after the blanked horizon + buffer (T+1..T+18 blanked, measured)
ALPHA = 20.0                   # shrinkage of the ramp -> link mapping tables


def ramp_profile(ON, dow):
    """same-slot 'on' level [2 (weekday, weekend), 288, L]: PROF_Q-th percentile over the days of the class, slots t-1..t+1 pooled."""
    D, T, L = ON.shape
    P = np.full((2, T, L), np.nan, np.float32)
    wk = dow >= 5
    for c, m in enumerate((~wk, wk)):
        if not m.any():
            continue
        A = ON[m]; A3 = np.concatenate([A[:, [0]], A, A[:, [-1]]], 1)
        with np.errstate(all='ignore'):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                P[c] = np.nanpercentile(np.concatenate([A3[:, :-2], A3[:, 1:-1], A3[:, 2:]], 0), PROF_Q, 0)
    return P


def ramp_cubes(ON, dow, prof=None):
    """ratio R = ON / profile (NaN where not usable), ok (usable ramp cell), flag (ramp switched off), A (ramp-link indices)."""
    prof = ramp_profile(ON, dow) if prof is None else prof
    pr = prof[(dow >= 5).astype(int)]
    ok = np.isfinite(ON) & (pr >= PMIN)
    with np.errstate(all='ignore'):
        R = np.where(ok, ON / pr, np.nan).astype(np.float32)
    flag = ok & (R <= THR)
    A = np.where(ok.any((0, 1)))[0]
    return R, ok, flag, A, prof


def mapping(flag, ok, Q, E, day_mask, A, alpha=ALPHA):
    """ramp-link -> mainline-link tables learned on the days of day_mask (train labels; fold-safe when day_mask excludes the fold):
    Mpos[a, k] = P(Q_k | ramp a switched off, k observed)  (shrunk to the base rate of k)
    Rc[a, k]   = P(ramp a switched off | Q_k, ramp a usable) (shrunk to 0 when k is rarely queued)
    Mneg[a, k] = P(Q_k | ramp a usable and NOT switched off, k observed)."""
    L = Q.shape[2]
    Fl = flag[day_mask][:, :, A].reshape(-1, len(A)).astype(np.float64)
    Ok = ok[day_mask][:, :, A].reshape(-1, len(A)).astype(np.float64)
    Qm = Q[day_mask].reshape(-1, L).astype(np.float64); Em = E[day_mask].reshape(-1, L).astype(np.float64)
    with np.errstate(all='ignore'):     # macOS Accelerate emits spurious matmul warnings; values verified against a brute-force count
        n_fq = Fl.T @ Qm; n_fe = Fl.T @ Em; n_oq = Ok.T @ Qm; n_oe = Ok.T @ Em
    base = Qm.sum(0) / np.maximum(Em.sum(0), 1.0)
    Mpos = (n_fq + alpha * base[None]) / (n_fe + alpha)
    Rc = n_fq / (n_oq + alpha)
    Mneg = (n_oq - n_fq + alpha * base[None]) / (n_oe - n_fe + alpha)
    return Mpos.astype(np.float32), Rc.astype(np.float32), Mneg.astype(np.float32)


RAMP_FEATS = ['self_has', 'self_ok', 'self_ratio', 'self_flag', 'self_flag3', 'nb1', 'nb2', 'nb3', 'lk_pos', 'lk_pos3', 'lk_neg', 'lk_mneg']
PB_FEATS = ['pb_vis', 'pb_q', 'pb_vrmin', 'pb_nb2', 'pb_any']


def _nb_count(fl, r):
    """number of switched-off ramp links within |j - k| <= r (excluding k) along the link axis; fl [W, L] bool."""
    W, L = fl.shape
    c = np.concatenate([np.zeros((W, 1), np.int32), np.cumsum(fl, 1, dtype=np.int32)], 1)
    lo = np.clip(np.arange(L) - r, 0, L); hi = np.clip(np.arange(L) + r + 1, 0, L)
    return (c[:, hi] - c[:, lo] - fl).astype(np.float32)


def ramp_feats(d, t, flag, ok, R, A, Mpos, Rc, Mneg, L):
    """leak features [W, F, L, len(RAMP_FEATS)] from the ramp cubes at the horizon slots t+1..t+6 (POST-T DATA)."""
    W = len(d); out = np.zeros((W, F, L, len(RAMP_FEATS)), np.float32)
    has = np.zeros(L, np.float32); has[A] = 1
    T = flag.shape[1]
    for h in range(1, F + 1):
        s = np.minimum(t + h, T - 1); sm = np.minimum(t + h - 1, T - 1); sp = np.minimum(t + h + 1, T - 1)
        fl = flag[d, s]; okk = ok[d, s]; rr = R[d, s]
        fl3 = fl | flag[d, sm] | flag[d, sp]
        FA = fl[:, A]; FA3 = fl3[:, A]; NA = okk[:, A] & ~FA
        lk_pos = (FA[:, :, None] * Mpos[None]).max(1) if len(A) else np.zeros((W, L), np.float32)
        lk_pos3 = (FA3[:, :, None] * Mpos[None]).max(1) if len(A) else np.zeros((W, L), np.float32)
        lk_neg = (NA[:, :, None] * Rc[None]).max(1) if len(A) else np.zeros((W, L), np.float32)
        # strongest "free" evidence: the lowest P(Q_k | usable and not switched off) among usable ramps that see k (Rc > 0.3)
        sees = Rc[None] > 0.3
        mn = np.where(NA[:, :, None] & sees, Mneg[None], np.nan)
        with np.errstate(all='ignore'):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                lk_mneg = np.nanmin(mn, 1) if len(A) else np.full((W, L), np.nan, np.float32)
        cols = [np.broadcast_to(has, (W, L)), okk, np.nan_to_num(rr, nan=-1.0), fl, fl3, _nb_count(fl, 1), _nb_count(fl, 2), _nb_count(fl, 3),
                lk_pos, lk_pos3, lk_neg, np.nan_to_num(lk_mneg, nan=-1.0)]
        out[:, h - 1] = np.stack([np.asarray(c, np.float32) for c in cols], -1)
    return out


def pb_feats(d, t, V, vis, cut, vf, L):
    """post-buffer mainline features (POST-T DATA): visible cells of slots t+19..t+21 (thinned to the masked-layer visibility).
    The same values are used for every horizon step.  [W, F, L, len(PB_FEATS)]."""
    W = len(d); T = V.shape[1]
    out = np.zeros((W, F, L, len(PB_FEATS)), np.float32)
    s0 = t + PB0; ok_w = (t + PB1) <= T - 1
    vr = np.full((W, PB1 - PB0 + 1, L), np.nan, np.float32)
    for j, o in enumerate(range(PB0, PB1 + 1)):
        s = np.minimum(t + o, T - 1)
        v = np.where(vis[d, s], V[d, s], np.nan) / vf[None, :]
        vr[:, j] = np.where(ok_w[:, None], v, np.nan)
    vis3 = np.isfinite(vr); q3 = vis3 & (vr <= 0.6)
    pb_vis = vis3.any(1).astype(np.float32)
    pb_q = np.where(vis3[:, 0], q3[:, 0].astype(np.float32), -1.0)
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pb_vrmin = np.nan_to_num(np.nanmin(vr, 1), nan=2.0)
    qa = q3.any(1)
    c = np.concatenate([np.zeros((W, 1), np.int32), np.cumsum(qa, 1, dtype=np.int32)], 1)
    lo = np.clip(np.arange(L) - 2, 0, L); hi = np.clip(np.arange(L) + 3, 0, L)
    pb_nb2 = (c[:, hi] - c[:, lo]).astype(np.float32)
    pb_any = np.where(pb_vis > 0, qa.astype(np.float32), -1.0)
    st = np.stack([pb_vis, pb_q, pb_vrmin, pb_nb2, pb_any], -1)
    out[:] = st[:, None]
    return out


def score_frame(truth, ev, qlast, win, P6, panel, cond, set_name, arm):
    """per-window scores with the production decode (lab.score_probs) for arbitrary probability cubes."""
    import types
    rc = types.SimpleNamespace(truth=truth, ev=ev, qlast=qlast, win=win, name=set_name, panel=panel, cond=cond, L=truth.shape[2])
    return lab.score_probs(rc, P6, arm=arm, set_name=set_name)
