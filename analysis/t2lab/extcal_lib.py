"""TRACK extcal (2026-10-16): per-step EXTENT calibration of pick A's April ongoing rows (pvogfv = kernel tfb-t2ogpvogf).
Builds on the pvogf lab (pvogf_lib.py: caches data/t2local/cache/pvogf/, probabilities results/pvogf/probs/{MD_s0, MD_s1, FULL_s0}.pkl).
New files only (extcal_*.py, results/extcal/, cache/extcal/); nothing existing is edited.

PRE-REGISTERED DECISION RULE (written 2026-10-16 08:50 +06, before any extcal number was computed; not changed afterwards)
  Baseline B0 = pvogfv lab analog: veto10(pvo, pvogf), pvo = 0.25 base_np + 0.25 base_id + 0.5 base_full, pvogf = the same on the
  ogf arm (probabilities averaged over seeds 0 and 1), production decode (expected-IoU top-m over the 6 x L cells of a window).
  Every candidate changes ONLY the pvogf probabilities / decode and is then passed through the same veto10 against uncalibrated pvo:
    a1  Platt per (h, panel): logit p' = alpha * logit p + beta, weighted log-loss over the keep-region cells with a visible label;
        Newton with ridge 1.0 on (alpha - 1, beta); a group with < 200 cells or < 20 positives keeps p.
    a2  isotonic per (h, panel) (increasing, clipped to [1e-4, 1 - 1e-4]), p' = iso(p) + 1e-6 p (ranking kept); same small-group rule.
    a3  Platt per (h, panel, cell inside / outside the queue at T (qlast)); same fit and small-group rule.
    b   size model: LightGBM poisson regression (63 leaves, lr 0.05, 400 trees, min_data 50, seed 0) of the visible true queued count
        at T+h on window-level features <= T: h, panel id, tod, weekend, c_q3 c_q6 c_q9 c_q12 c_d3 c_d6 c_d12 c_rel6 c_n0 c_new3 c_clr3
        c_new6 c_clr6 c_first c_peak (window-level ogfeat columns), n keep-region cells, sum over keep cells of prior at T+h and of
        prior_now (same-scenario blended prior for March / April windows, train prior for train windows).  Rows: F03 lab windows of the
        train split (lab weights) + March faithful windows (run-starts 1.0 / rest 0.3, x WM 5) of the fitting days.  Decode: per step,
        the top round(pred_h) cells by the pvogf probability at step h.
    c1  per-step factor f_h on the decoder's per-step count m_h: m'_h = round(f_h m_h), top m'_h cells by p at step h;
        f_h in {0.60, 0.62, ..., 1.40} chosen to maximize the aggregate IoU of the fitting windows (coordinate ascent over h from f = 1,
        2 sweeps), pooled over panels.
    c2  as c1 but only outside qlast: the decoder's inside-qlast cells are kept; the outside-qlast count m_out,h is scaled by f_h and
        filled with the top outside-qlast cells by p at step h.
  Fitting data (many-day lab): pseudo-March windows of March days < 15 = faithful March windows (run-starts + every-3rd rest) U
  offlike_m(days < 15).  Probabilities are OUT-OF-SAMPLE by a 2-fold day-parity cross-fit inside days < 15 (mode CF15): ogf np / id /
  full members trained on the F03 rows + the March rows of the other fold's days (x WM 5; full member on the blended-prior rows), lab
  scale (255 leaves, lr 0.05, min_data 40, 500 trees), seeds 0 and 1 averaged, pvogf = 0.25 np + 0.25 id + 0.5 full.
  (b) needs no probabilities for fitting (rows = F03 + March faithful windows of days < 15).
  Evaluation: offlike_m windows of March days >= 15 with the MD probabilities (F03 + March days < 15), candidate vs B0,
  lab.paired_bootstrap (2000 resamples within panel), panels = sign of the per-panel mean delta.
  ADOPT a candidate only if BOTH
    (1) many-day offlike_m delta >= +0.005 IoU, 95 % CI lower bound > 0 and >= 6/8 panels positive;
    (2) FM-PM private >= +0.0005 (LB-equivalent).  Footprint = candidate vs pvogfv on the lab-emulated April decisions (FULL_s0 = F03 +
        all 31 March days, results/pvogf/private_decisions_emulated.pkl); the April calibration is re-fitted exactly as above on all
        31 March days (mode CF31: day-parity cross-fit over all March days, members trained on F03 + the other fold's March days; (b)
        refitted on F03 + all March faithful windows).  Each changed window is priced by fpmatch_lib.forecast (panel-exact, +-2 h, sign
        and size of the all-cell dn) over the candidate's seed-average many-day per-window pairs vs B0 (offlike_m + runstart_m, days
        >= 15); FM-PM = 0.15 x the mean over the 40 private ongoing windows (unchanged windows count 0).  Computed only for candidates
        that pass (1).
  If several pass: the higher FM-PM private wins.
  AMENDMENT (2026-10-16 09:17 +06, after a code smoke run with seed-0 cross-fit only, BEFORE any b2 number was computed; the rule
  above is unchanged).  The smoke run showed that (b) as written is mis-specified for the lab: about 45 % of the March label cells are
  hidden (ev False), the size target counts visible cells only, but the decode picks the top cells among ALL cells, so the visible
  extent is roughly halved (b and the per-step oracle both lose about -0.25..-0.33).  (b) stays in the table as pre-registered (FAIL).
  Added, under the SAME gates (1) and (2):
    b2  = b with the target and the decode on the same scored cells: feature n_vis_h (number of scored cells at T+h: eligible cells
          for train windows, visible label cells for March windows, all links for the April windows), decode = the top round(pred_h)
          cells by p among the scored cells (the lab's ev mask; in April every link is scored).
  The per-step oracle is reported the same way (top n_true_h among the scored cells, non-binding).
  Candidates are now 7 (Bonferroni-style CI 99.3 %; the printed 99.2 % column is kept).
  Reported, not binding: runstart_m delta, each seed alone, the per-step size oracle, calibration diagnostics (reliability per h,
  inside / outside qlast), fitted parameters, number of candidates (6: a Bonferroni-style 99.2 % CI is also printed).
"""
import json, os
import numpy as np
import pandas as pd
import lab

ROOT = lab.ROOT
H, F, ONGOING = lab.H, lab.F, lab.ONGOING
PVC = f'{lab.CACHE}/pvogf'
CACHE = f'{lab.CACHE}/extcal'
RES = f'{ROOT}/analysis/t2lab/results/extcal'
PVRES = f'{ROOT}/analysis/t2lab/results/pvogf'
for _d in (CACHE, RES, f'{RES}/probs', f'{CACHE}/models'):
    os.makedirs(_d, exist_ok=True)
CUT = 15
WM = 5.0
CANDS = ('a1', 'a2', 'a3', 'b', 'b2', 'c1', 'c2')
FGRID = np.round(np.arange(0.60, 1.4001, 0.02), 2)
SIZE_STATIC = ['c_q3', 'c_q6', 'c_q9', 'c_q12', 'c_d3', 'c_d6', 'c_d12', 'c_rel6', 'c_n0', 'c_new3', 'c_clr3', 'c_new6', 'c_clr6',
               'c_first', 'c_peak']
SIZE_COLS = ['h', 'panel_id', 'tod', 'weekend'] + SIZE_STATIC + ['n_keep', 'prior_sum_h', 'prior_now_sum']
SIZE_COLS_B2 = SIZE_COLS + ['n_vis_h']
EPS = 1e-4


# ------------------------------------------------------------------ probabilities / decode
def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def decode_one(p6):
    """production decode (= K.decode) on one window [6, L] -> bool [6, L]."""
    p = p6.ravel(); order = np.argsort(-p); cs = np.cumsum(p[order]); tot = cs[-1]
    j = np.argmax(cs / (np.arange(1, len(p) + 1) + (tot - cs)))
    out = np.zeros(len(p), bool); out[order[:j + 1]] = True
    return out.reshape(p6.shape)


def decode6(P6):
    return np.stack([decode_one(P6[i]) for i in range(len(P6))]) if len(P6) else np.zeros(P6.shape, bool)


def veto(A, B, qlast, g=10):
    q = np.repeat(qlast[:, None, :], F, 1)
    add = (B & ~A & ~q).sum((1, 2))
    return np.where((add > g)[:, None, None], A, B), add


def topm_step(p, m):
    """bool mask of the top-m entries of p (1-d); m clipped to [0, len(p)]."""
    m = int(min(max(m, 0), len(p))); out = np.zeros(len(p), bool)
    if m:
        out[np.argsort(-p, kind='stable')[:m]] = True
    return out


def rescale_steps(P6, D6, f, qlast=None):
    """c1 (qlast None): per step h, m'_h = round(f_h m_h) top cells by p.  c2 (qlast given): inside-qlast cells of D6 kept, the
    outside-qlast count scaled by f_h and filled with the top outside cells by p."""
    out = np.zeros_like(D6)
    for i in range(len(P6)):
        for h in range(F):
            p = P6[i, h]; dd = D6[i, h]
            if qlast is None:
                out[i, h] = topm_step(p, np.floor(f[h] * dd.sum() + 0.5))
            else:
                q = qlast[i]; o = ~q
                mo = np.floor(f[h] * (dd & o).sum() + 0.5)
                sel = np.zeros(len(p), bool); sel[q] = dd[q]
                po = np.where(o, p, -1.0); sel |= topm_step(po, min(mo, o.sum()))
                out[i, h] = sel
    return out


def size_decode(P6, m6, scored=None):
    """b: per step, the top round(m) cells by p (b2 / oracle: among the scored cells only)."""
    out = np.zeros(P6.shape, bool)
    for i in range(len(P6)):
        for h in range(F):
            p = P6[i, h] if scored is None else np.where(scored[i, h], P6[i, h], -1.0)
            m = np.floor(m6[i, h] + 0.5) if scored is None else min(np.floor(m6[i, h] + 0.5), scored[i, h].sum())
            out[i, h] = topm_step(p, m)
    return out


def score_dec(dec, truth, ev, qlast):
    T = truth & ev; Pp = dec & ev
    u = (T | Pp).sum((1, 2)); it = (T & Pp).sum((1, 2))
    return dict(iou=np.where(u == 0, 1.0, it / np.maximum(u, 1)), n_pred=Pp.sum((1, 2)), n_all=dec.sum((1, 2)), n_true=T.sum((1, 2)))


def agg_iou(iou, panel):
    """competition aggregation of per-window IoU (window -> panel -> family -> mean)."""
    df = pd.DataFrame(dict(iou=iou, panel=panel)); df['family'] = df.panel.str.rsplit('_', n=1).str[0]
    return lab.aggregate(df)


# ------------------------------------------------------------------ calibration fits
def platt_fit(p, y, ridge=1.0, iters=25):
    """weighted-free Platt on logit: returns (alpha, beta); Newton with ridge on (alpha - 1, beta)."""
    x = logit(p).astype(np.float64); y = y.astype(np.float64); th = np.array([1.0, 0.0])
    Xd = np.stack([x, np.ones_like(x)], 1)
    for _ in range(iters):
        z = Xd @ th; s = 1 / (1 + np.exp(-z))
        g = Xd.T @ (s - y) + ridge * (th - np.array([1.0, 0.0]))
        Hm = (Xd * (s * (1 - s))[:, None]).T @ Xd + ridge * np.eye(2)
        step = np.linalg.solve(Hm, g); th = th - step
        if np.abs(step).max() < 1e-7:
            break
    return float(th[0]), float(th[1])


def platt_apply(p, ab):
    a, b = ab
    return 1 / (1 + np.exp(-(a * logit(p) + b)))


def iso_fit(p, y):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(y_min=EPS, y_max=1 - EPS, increasing=True, out_of_bounds='clip')
    ir.fit(p.astype(np.float64), y.astype(np.float64))
    return (ir.X_thresholds_.astype(np.float64), ir.y_thresholds_.astype(np.float64))


def iso_apply(p, xy):
    xs, ys = xy
    return np.interp(p.astype(np.float64), xs, ys) + 1e-6 * p


def small(y):
    return len(y) < 200 or y.sum() < 20


def fit_cells(cells, kind):
    """cells: DataFrame(panel, h, inq, p, y) of keep-region visible cells of the fitting windows.
    kind a1 / a2 / a3 -> dict {group key: params or None}."""
    keys = ['panel', 'h'] + (['inq'] if kind == 'a3' else [])
    out = {}
    for k, g in cells.groupby(keys):
        p = g.p.to_numpy(); y = g.y.to_numpy()
        if small(y):
            out[k] = None
        elif kind == 'a2':
            out[k] = iso_fit(p, y)
        else:
            out[k] = platt_fit(p, y)
    return out


def apply_cal(P6, panel, qlast, kind, params, keep=None):
    """recalibrated copy of P6 [W, 6, L] of one panel.  Cells with p == 0 (outside the keep region) stay 0."""
    Q = P6.copy()
    nz = P6 > 0
    for h in range(F):
        for inq in ((0, 1) if kind == 'a3' else (None,)):
            k = (panel, h + 1) if inq is None else (panel, h + 1, inq)
            prm = params.get(k)
            if prm is None:
                continue
            m = nz[:, h]
            if inq is not None:
                m = m & (qlast == bool(inq))
            v = P6[:, h][m]
            Q[:, h][m] = iso_apply(v, prm) if kind == 'a2' else platt_apply(v, prm)
    return Q


def cum_tables(P6, D6, truth, ev, qlast, outside):
    """per window / step: cumulative visible hits and visible picks of the cells ranked by p (outside qlast only if `outside`),
    the decoder's count m_h of those cells, the fixed (inside-qlast) hits / visible picks of c2, the true visible total."""
    W, _, L = P6.shape
    ch = np.zeros((W, F, L + 1), np.int32); cv = np.zeros((W, F, L + 1), np.int32)
    m0 = np.zeros((W, F), np.int32); mx = np.zeros((W, F), np.int32); fh = np.zeros((W, F), np.int32); fv = np.zeros((W, F), np.int32)
    T = truth & ev
    for i in range(W):
        q = qlast[i]
        for h in range(F):
            p = P6[i, h]; dd = D6[i, h]
            if outside:
                po = np.where(~q, p, -1.0); order = np.argsort(-po, kind='stable')[:int((~q).sum())]
                m0[i, h] = (dd & ~q).sum(); fh[i, h] = (dd & q & T[i, h]).sum(); fv[i, h] = (dd & q & ev[i, h]).sum()
            else:
                order = np.argsort(-p, kind='stable'); m0[i, h] = dd.sum()
            n = len(order); mx[i, h] = n
            ch[i, h, 1:n + 1] = np.cumsum(T[i, h][order]); ch[i, h, n + 1:] = ch[i, h, n]
            cv[i, h, 1:n + 1] = np.cumsum(ev[i, h][order]); cv[i, h, n + 1:] = cv[i, h, n]
    return dict(ch=ch, cv=cv, m0=m0, mx=mx, fh=fh, fv=fv, Tt=T.sum((1, 2)))


def iou_from_tables(tb, f):
    m = np.minimum(np.floor(tb['m0'] * np.asarray(f)[None, :] + 0.5).astype(np.int64), tb['mx'])
    W = len(m); wi = np.arange(W)[:, None]; hi = np.arange(F)[None, :]
    inter = (tb['ch'][wi, hi, m] + tb['fh']).sum(1); pv = (tb['cv'][wi, hi, m] + tb['fv']).sum(1)
    u = tb['Tt'] + pv - inter
    return np.where(u == 0, 1.0, inter / np.maximum(u, 1))


def fit_factors(tables, panels):
    """c1 / c2: coordinate ascent of f_h on the aggregate IoU of the fitting windows (tables: list of cum_tables, one per panel)."""
    f = np.ones(F)
    pan = np.concatenate([np.full(len(tb['Tt']), p) for tb, p in zip(tables, panels)])
    score = lambda fv: agg_iou(np.concatenate([iou_from_tables(tb, fv) for tb in tables]), pan)
    trace = [('start', score(f))]
    for sweep in range(2):
        for h in range(F):
            best = (score(f), f[h])
            for v in FGRID:
                g = f.copy(); g[h] = v; s = score(g)
                if s > best[0] + 1e-12:
                    best = (s, v)
            f[h] = best[1]; trace.append((f'sweep{sweep} h{h + 1}', best[0]))
    return f, trace
