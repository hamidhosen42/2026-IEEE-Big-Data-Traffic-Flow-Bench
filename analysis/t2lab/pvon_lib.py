"""Track pvon: shared helpers (new file; copies of the small helpers of exp_onset_marchtrain.py / exp_onset_wrongscen.py,
which run their experiment at import time and therefore cannot be imported).

Masked-layer emulation on the TRAIN cubes (measured on the released months, see REPORT.md):
  * every day has one mask regime: visible share of the eligible cells = 0.5 / 0.7 / 0.8 (one third of the days each),
  * inside a day the mask is i.i.d. per eligible cell (lag-1 correlation of visibility among eligible cells: 0.00 in time
    and across links on days without official windows),
  * slot T of a window is read from the same masked layer (visible share = the day's regime),
  * the 12 history slots of the OFFICIAL windows are fully visible (raw), their 6 horizon slots are blank.
"""
import gc, os, time, types, warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lab
from exp_firstcross_prod_snippet import fill_time36

H, F = lab.H, lab.F
PRIOR3 = ('prior', 'prior_now', 'prior_alldays')
COLS = lab.all_cols()
JP = [COLS.index(c) for c in PRIOR3]
KEEPC = [i for i, c in enumerate(COLS) if not c.startswith('ramp_')]          # v13-like feature set (production onset has no ramp)
COLS_NR = [COLS[i] for i in KEEPC]
K_BLEND = 4
REGIMES = (0.5, 0.7, 0.8)
OUT = 'results/pvon'
os.makedirs(OUT, exist_ok=True)
SMOKE = os.environ.get('PVON_SMOKE') == '1'                    # seconds-scale plumbing test: 2 panels, 30 iterations
PANELS = ['D7_I10_W', 'D12_I5_S'] if SMOKE else list(lab.PANELS)
STAG = '_smoke' if SMOKE else ''


# ------------------------------------------------------------------ released March month (masked layer)
def val_panel(panel, split='validation'):
    P = lab.load_panel(panel)
    z = np.load(P['npz'], allow_pickle=False)
    days = np.array([str(x) for x in z[f'{split}_days']]); o = np.argsort(days)
    V = z[f'{split}_Vm'][o].astype(np.float32); Fq = z[f'{split}_Fm'][o].astype(np.float32); Em = z[f'{split}_Em'][o]
    E = np.isfinite(V); Q = E & (V <= P['cut'][None, None, :])
    Pv = dict(panel=panel, links=P['links'], li=P['li'], cut=P['cut'], vf=P['vf'], cap=P['cap'], lanes=P['lanes'], length=P['length'],
              days=list(days[o]), dow=pd.to_datetime(pd.Series(days[o])).dt.dayofweek.to_numpy(), V=V, Fq=Fq, E=E, Q=Q, Em=Em,
              ON=z[f'{split}_ON'][o].astype(np.float32), OFF=z[f'{split}_OFF'][o].astype(np.float32))
    Pv['share'] = E.sum((1, 2)) / np.maximum(Em.sum((1, 2)), 1)
    return P, Pv


def select(Pv, need_og=True):
    """onset / faithful-ongoing candidate maps on a masked month (visible data only); identical to exp_onset_marchtrain.select.
    need_og=False skips the (slow) time fill and returns og = all False (enough for spaced onset training picks)."""
    V, E, Q, Em = Pv['V'], Pv['E'], Pv['Q'], Pv['Em']; D, _, L = V.shape
    if need_og:
        VF = fill_time36(V); MF = np.isfinite(VF); QF = MF & (VF <= Pv['cut'][None, None, :])
    blank = (Em.sum(2) >= 5) & (E.sum(2) == 0)
    og = np.zeros((D, 288), bool); on = og.copy()
    # cheap vectorised pre-filters (same booleans as the loop below would compute)
    cE = np.concatenate([np.zeros((D, 1)), np.cumsum(Em.sum(2), 1)], 1)
    cB = np.concatenate([np.zeros((D, 1)), np.cumsum(blank, 1)], 1)
    for d in range(D):
        for t in range(H, 288 - F):
            if cB[d, t + F + 1] - cB[d, t - H] > 0 or (cE[d, t] - cE[d, t - H]) / (H * L) < 0.7:
                continue
            if lab.established(Q[d, t - H:t]):
                if not need_og:
                    continue
                Tf = QF[d, t + 1:t + 1 + F]; Mf = MF[d, t + 1:t + 1 + F]
                if not Tf.any():
                    continue
                S = np.repeat(QF[d, t][None], F, 0)
                u = ((Tf | S) & Mf).sum(); i = (Tf & S & Mf).sum()
                og[d, t] = (1.0 if u == 0 else i / u) <= 0.9
            else:
                on[d, t] = Q[d, t + 1:t + 1 + F].any()
    return on, og


def spaced_picks(on, d_lo=0, d_hi=10 ** 9, days=None):
    """production build_windows logic for onset: one chronological scan, picks >= 72 slots apart (among onset picks only)."""
    g = lab._glist(on); picks = []; last = -10 ** 9
    for x in g:
        dd = int(x // 288)
        if dd >= d_hi:
            break
        if x - last >= lab.SPACING:
            last = x                                           # NOTE: the scan state advances on every day (as the kernel)
            if dd >= d_lo and (days is None or dd in days):
                picks.append((dd, int(x % 288)))
    return picks


def multistart_onset(on, og, D):
    """official-like onset windows: faithful greedy (interleaved, 5 + 5 quota, 360 min) restarted at 00:00 of every day."""
    g_on, g_og = lab._glist(on), lab._glist(og); s = set()
    for d0 in range(D):
        a, _ = lab.greedy(g_on, g_og, 'interleaved', start_day=d0); s.update(a)
    return sorted(s)


def onset_rows(P, S, wins, prior, VT=None):
    """production-shaped onset rows (h = 6 only) of the windows `wins` [(d, t), ...] on the state dict S
    (V, Fq, Q, E, dow, ON, OFF: masked month, raw train panel or emulated-masked train panel).
    VT: slot-T values [W, L] (default: S['V'][d, t], i.e. the masked layer itself)."""
    d = np.array([w[0] for w in wins]); t = np.array([w[1] for w in wins])
    idx = t[:, None] + np.arange(-H, 0)[None]
    Vh = S['V'][d[:, None], idx]; Fh = S['Fq'][d[:, None], idx]
    VT = S['V'][d, t] if VT is None else VT
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        Fe, qlast = lab.window_features(P, Vh, Fh, t, S['dow'][d], prior, VT)
        lab.add_ramp(Fe, S['ON'], S['OFF'], d, t, P['cap'])
    W, L = qlast.shape
    feats = dict(Fe); feats['h'] = np.full((W, L), 6, np.float32)
    feats['prior'], feats['prior_now'], feats['prior_alldays'] = lab.prior_feats(prior, S['dow'][d], t, 6)
    s6 = np.minimum(t + 6, 287)
    return dict(d=d, t=t, cond=np.array([lab.ONSET] * W), qlast=qlast, W=W, L=L, rows=[(6, feats, S['Q'][d, s6], S['E'][d, s6])],
                hist_cov=np.isfinite(Vh).mean((1, 2)), T_cov=np.isfinite(VT).mean(1))


def matrix(R):
    X, y, e, meta, call = lab.to_matrix(R, lab.ONSET, None, None)
    assert list(call) == COLS
    return X, y.astype(bool), e.astype(bool), meta


def cum_tables(Pv):
    """cumQ, cumE [3, D+1, 288, L]: visible queued / visible counts summed over slots +-2 and over the days < d of class c."""
    Q = Pv['Q'].astype(np.float32); E = Pv['E'].astype(np.float32)

    def box(a):
        c = np.cumsum(np.pad(a, ((0, 0), (3, 2), (0, 0))), 1)
        return c[:, 5:] - c[:, :-5]
    Q, E = box(Q), box(E)
    D = Q.shape[0]; wk = Pv['dow'] >= 5
    cQ = np.zeros((3, D + 1) + Q.shape[1:], np.float32); cE = cQ.copy()
    for c, m in enumerate((~wk, wk, np.ones(D, bool))):
        cQ[c, 1:] = np.cumsum(Q * m[:, None, None], 0); cE[c, 1:] = np.cumsum(E * m[:, None, None], 0)
    return cQ, cE


def blend_rows(X, R, meta, dow, prior_tr, cQ, cE, k=K_BLEND, day=None):
    """X copy with the three prior columns blended per row with the scenario prior of the days < d (d = the window's day, or
    `day` [W] when given)."""
    wi, li = meta[:, 0], meta[:, 1]
    d, t = R['d'], R['t']; cls = (dow[d] >= 5).astype(int); s6 = np.minimum(t + 6, 287)
    dd = d if day is None else np.asarray(day)
    cr, dr, s6r, tr_ = cls[wi], dd[wi], s6[wi], t[wi]
    trv = [prior_tr[cr, s6r, li], prior_tr[cr, tr_, li], prior_tr[2, s6r, li]]
    Xk = X.copy()
    for j, (cc, ss) in enumerate(((cr, s6r), (cr, tr_), (np.full_like(cr, 2), s6r))):
        q = cQ[cc, dr, ss, li]; ee = cE[cc, dr, ss, li]
        sp = np.where(ee > 0, q / np.maximum(ee, 1e-9), 0.0); n = ee / 5.0; a_ = n / (n + k)
        Xk[:, JP[j]] = (1 - a_) * trv[j] + a_ * sp
    return Xk


# ------------------------------------------------------------------ masked-layer emulation on the train cubes
def mask_train(P, seed, regimes=REGIMES, share=None):
    """emulated masked layer of the train panel: per-day regime (visible share of the eligible cells) drawn uniformly from
    `regimes` (or the fixed array `share` [D]), cells hidden i.i.d.  Returns a state dict like val_panel's Pv
    (V, Fq, E = visible, Q = visible & queued, Em = raw eligibility, share, block) on the TRAIN days."""
    rng = np.random.default_rng(seed * 1000 + lab.PANELS.index(P['panel']))
    D = len(P['days'])
    s = rng.choice(np.asarray(regimes, np.float32), size=D) if share is None else np.asarray(share, np.float32)
    vis = P['E'] & (rng.random(P['E'].shape, dtype=np.float32) < s[:, None, None])
    V = np.where(vis, P['V'], np.nan).astype(np.float32); Fq = np.where(vis, P['Fq'], np.nan).astype(np.float32)
    return dict(panel=P['panel'], cut=P['cut'], days=P['days'], dow=P['dow'], V=V, Fq=Fq, E=vis, Q=vis & P['Q'], Em=P['E'],
                ON=P['ON'], OFF=P['OFF'], share=s, block=P['block'])


def raw_state(P):
    """the raw train panel as a state dict (Em = E)."""
    return dict(panel=P['panel'], cut=P['cut'], days=P['days'], dow=P['dow'], V=P['V'], Fq=P['Fq'], E=P['E'], Q=P['Q'], Em=P['E'],
                ON=P['ON'], OFF=P['OFF'], share=np.ones(len(P['days']), np.float32), block=P['block'])


def prod_VT(P, d, t, n_seed, p=0.62):
    """production slot-T simulation (make_rows): rng(len(wins)), 62 % of the raw observed cells visible."""
    rng = np.random.default_rng(n_seed)
    return np.where(rng.random((len(d), P['V'].shape[2])) < p, P['V'][d, t], np.nan)


# ------------------------------------------------------------------ models / scoring
def fit_cat(X, y, w=None, seed=0, it=600):
    m = CatBoostClassifier(iterations=30 if SMOKE else it, depth=8, learning_rate=0.08, thread_count=lab.THREADS, verbose=0, random_seed=seed)
    m.fit(X, y, sample_weight=w)
    return m


def namespace(panel, name, d, t, truth, ev, qlast, L, fold=0):
    return types.SimpleNamespace(truth=truth, ev=ev, qlast=qlast, win=pd.DataFrame(dict(d=d, t=t, cond=lab.ONSET, fold=fold)), name=name,
                                 panel=panel, cond=lab.ONSET, L=L)


def score(rc, meta, p, arm, W=None):
    W = len(rc.win) if W is None else W
    P6 = np.zeros((W, F, rc.L), np.float32); P6[meta[:, 0], F - 1, meta[:, 1]] = p
    return lab.score_probs(rc, P6, arm=arm, set_name=rc.name)


def compare(Rr, a, b, extra=''):
    """paired bootstrap b - a with the per-panel sign count; returns a dict."""
    A = Rr[Rr.arm == a]; B = Rr[Rr.arm == b]
    bt = lab.paired_bootstrap(A, B)
    pa = A.groupby('panel').iou.mean(); pb = B.groupby('panel').iou.mean(); dl = pb - pa
    return dict(a=a, b=b, iou_a=lab.aggregate(A), iou_b=lab.aggregate(B), diff=bt['diff'], lo=bt['lo'], hi=bt['hi'], p_pos=bt['p_pos'],
                n=bt['n_windows'], pos=int((dl > 1e-9).sum()), neg=int((dl < -1e-9).sum()), panels=len(dl), note=extra)


def fmt(c):
    return (f"{c['b']:>22s} - {c['a']:<14s} {c['iou_b']:.4f} vs {c['iou_a']:.4f}  {c['diff']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"
            f"  panels +{c['pos']}/-{c['neg']} of {c['panels']}  n={c['n']} {c['note']}")
