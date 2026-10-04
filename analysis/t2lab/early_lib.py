"""Track `early`: library for early-month same-scenario prior designs (NEW file, nothing existing is modified).

Model: the production onset CatBoost (v13-like feature set = lab columns without ramp_*), UNCHANGED. Only the three prior
columns (prior, prior_now, prior_alldays) of the evaluation rows are substituted.

Bundles (one per panel):
  march_bundle(panel)   214 pseudo-March onset windows (same selection as exp_onset_splitprior.py), masked-layer histories,
                        scenario layer = masked March month, model prior = all-train prior          -> new-scenario regime
  train_bundle(panel)   official-like onset windows of train fold 0 (raw histories), scenario layer = the same train days with
                        34% of the observed cells hidden, model prior = fold-safe prior (folds 1-3)  -> in-distribution regime
Evidence: per window a list of scenario days strictly before the window's day (optionally the same day, slots < T), box sums
over slots [s+lo, s+hi] of visible queued cells (O), visible cells (e) and the queued count EXPECTED under the class-matched
train prior (X).  Design:  new = clip(base + a * (O/e - ref)),  a = n / (n + k)
  ref = 'base' -> classic blend (1-a) base + a O/e        ref = 'box' -> residual form, ref = X/e (train prior averaged over the
  same visible cells / days / classes): unbiased in-distribution for any box width and any class mix of the evidence days.
"""
import gc, os, pickle, sys, time, types, warnings
import numpy as np
import pandas as pd
import lab
from exp_firstcross_prod_snippet import fill_time36

H, F = lab.H, lab.F
PRIOR3 = ('prior', 'prior_now', 'prior_alldays')
COLS = lab.all_cols()
KEEPI = [i for i, c in enumerate(COLS) if not c.startswith('ramp_')]          # v13-like feature set (lab NORAMP=1)
JP = [COLS.index(c) for c in PRIOR3]
OUT = 'results/early'
CACHE = f'{lab.CACHE}/early'
for _d in (OUT, CACHE):
    os.makedirs(_d, exist_ok=True)


# ------------------------------------------------------------------ copied from exp_onset_splitprior.py (that module runs at import)
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


def select(Pv):
    V, E, Q, Em = Pv['V'], Pv['E'], Pv['Q'], Pv['Em']; D, _, L = V.shape
    VF = fill_time36(V); MF = np.isfinite(VF); QF = MF & (VF <= Pv['cut'][None, None, :])
    blank = (Em.sum(2) >= 5) & (E.sum(2) == 0)
    og = np.zeros((D, 288), bool); on = og.copy()
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
                og[d, t] = (1.0 if u == 0 else i / u) <= 0.9
            else:
                on[d, t] = Q[d, t + 1:t + 1 + F].any()
    return on, og, blank


def onset_rows(P, Pv, wins, prior):
    d = np.array([w[0] for w in wins]); t = np.array([w[1] for w in wins])
    idx = t[:, None] + np.arange(-H, 0)[None]
    Vh = Pv['V'][d[:, None], idx]; Fh = Pv['Fq'][d[:, None], idx]; VT = Pv['V'][d, t]
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        Fe, qlast = lab.window_features(P, Vh, Fh, t, Pv['dow'][d], prior, VT)
        lab.add_ramp(Fe, Pv['ON'], Pv['OFF'], d, t, P['cap'])
    W, L = qlast.shape
    R = dict(d=d, t=t, cond=np.array([lab.ONSET] * W), qlast=qlast, W=W, L=L, rows=[])
    for h in range(1, F + 1):
        feats = dict(Fe); feats['h'] = np.full((W, L), h, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = lab.prior_feats(prior, Pv['dow'][d], t, h)
        R['rows'].append((h, feats, Pv['Q'][d, np.minimum(t + h, 287)], Pv['E'][d, np.minimum(t + h, 287)]))
    return R


# ------------------------------------------------------------------ bundles
def march_bundle(panel, refresh=False):
    f = f'{CACHE}/march_{panel}.pkl'
    if os.path.exists(f) and not refresh:
        return pickle.load(open(f, 'rb'))
    P, Pv = val_panel(panel)
    on, og, blank = select(Pv)
    D = len(Pv['days'])
    g_on, g_og = lab._glist(on), lab._glist(og); s = set(); picks = {}
    for d0 in range(D):
        a, bb = lab.greedy(g_on, g_og, 'interleaved', start_day=d0)
        s.update(a); picks[d0] = (a, bb)
    wins = sorted(s)
    prior_tr = lab.prior_table(P, np.ones(len(P['days']), bool))
    R = onset_rows(P, Pv, [(a, b, lab.ONSET) for a, b in wins], prior_tr)
    X, y, e, meta, call = lab.to_matrix(R, lab.ONSET, None, None)
    assert list(call) == COLS
    d, t = R['d'], R['t']; W, L = R['W'], R['L']
    assert np.array_equal(meta[:, 0], np.repeat(np.arange(W), L)) and np.array_equal(meta[:, 1], np.tile(np.arange(L), W))
    b = dict(panel=panel, family=panel.rsplit('_', 1)[0], kind='march', L=L, D=D, W=W, dow=Pv['dow'], wk=Pv['dow'] >= 5,
             Qv=Pv['Q'], Ev=Pv['E'], prior_tr=prior_tr, X=X.reshape(W, L, -1), d=d, t=t,
             truth=np.stack([Pv['Q'][d, t + h] for h in range(1, F + 1)], 1),
             ev=np.stack([Pv['E'][d, t + h] for h in range(1, F + 1)], 1), qlast=R['qlast'], on=on, og=og, blank=blank,
             picks=picks, days=Pv['days'], wdow=Pv['dow'][d])
    pickle.dump(b, open(f, 'wb'), protocol=4)
    lab.drop_panel(panel)
    return b


def train_bundle(panel, hide=0.34, seed=0, refresh=False):
    """held-out TRAIN fold 0: official-like onset windows (raw histories), fold-safe prior (prior5[:, 0]); scenario layer =
    the fold-0 train days with `hide` of the observed cells hidden i.i.d. (emulates the released masked layer)."""
    f = f'{CACHE}/train0_{panel}_h{int(hide * 100)}_s{seed}.pkl'
    if os.path.exists(f) and not refresh:
        return pickle.load(open(f, 'rb'))
    P = lab.load_panel(panel)
    rc = lab.rows(panel, 'offlike', lab.ONSET)
    widx = np.where(rc.win.fold.to_numpy() == 0)[0]
    me = np.isin(rc.wi, widx)
    X = rc.matrix(me, COLS, 0)
    W, L = len(widx), rc.L
    remap = {w: i for i, w in enumerate(widx)}
    wi = np.array([remap[w] for w in rc.wi[me]]); li = rc.li[me].astype(int)
    assert np.array_equal(wi, np.repeat(np.arange(W), L)) and np.array_equal(li, np.tile(np.arange(L), W)) and (rc.h[me] == 6).all()
    d = rc.win.d.to_numpy()[widx]; t = rc.win.t.to_numpy()[widx]
    D0 = int((P['block'] == 0).sum()); assert d.max() < D0
    rng = np.random.default_rng(seed * 1000 + lab.PANELS.index(panel))
    keep = rng.random((D0, 288, L)) >= hide
    Ev = P['E'][:D0] & keep; Qv = P['Q'][:D0] & keep
    on, og = lab.candidates(P)
    g_on, g_og = lab._glist(on), lab._glist(og); picks = {}
    for d0 in range(D0):
        a, bb = lab.greedy(g_on, g_og, lab.FAITHFUL_SCAN, start_day=d0)
        picks[d0] = (a, bb)
    key = {(int(a), int(c)): i for i, (a, c) in enumerate(zip(d, t))}
    b = dict(panel=panel, family=panel.rsplit('_', 1)[0], kind='train0', L=L, D=D0, W=W, dow=P['dow'][:D0], wk=P['dow'][:D0] >= 5,
             Qv=Qv, Ev=Ev, prior_tr=lab.prior_table(P, P['block'] != 0), X=X.reshape(W, L, -1), d=d, t=t, truth=rc.truth[widx],
             ev=rc.ev[widx], qlast=rc.qlast[widx], on=on[:D0], og=og[:D0], blank=np.zeros((D0, 288), bool), picks=picks,
             days=P['days'][:D0], wdow=P['dow'][d], key=key)
    pickle.dump(b, open(f, 'wb'), protocol=4)
    lab.drop_panel(panel)
    return b


# ------------------------------------------------------------------ evidence
class Evidence:
    """cumulative slot sums of one scenario layer: cQ / cE / cX [D, 289, L] (visible queued, visible, expected queued under the
    class-matched train prior).  blank [D, 288]: slots hidden for all links (official-window horizons in the replay)."""

    def __init__(self, b, blank=None):
        Ev = b['Ev']; Qv = b['Qv']
        if blank is not None and blank.any():
            Ev = Ev & ~blank[:, :, None]; Qv = Qv & Ev
        cls = b['wk'].astype(int)
        X0 = Ev * b['prior_tr'][cls]                                   # [D, 288, L]
        z = np.zeros((Ev.shape[0], 1, Ev.shape[2]), np.float32)
        self.cQ = np.concatenate([z, np.cumsum(Qv, 1, dtype=np.float32)], 1)
        self.cE = np.concatenate([z, np.cumsum(Ev, 1, dtype=np.float32)], 1)
        self.cX = np.concatenate([z, np.cumsum(X0, 1, dtype=np.float32)], 1)

    def box(self, days, s, lo, hi, tmax=288):
        """sums over slots [s+lo, s+hi] clipped to [0, tmax) for the given days -> O, e, X [n_days, L]"""
        a = min(max(s + lo, 0), tmax); z = max(min(s + hi + 1, tmax), a)
        return self.cQ[days, z] - self.cQ[days, a], self.cE[days, z] - self.cE[days, a], self.cX[days, z] - self.cX[days, a]


def link_box(a, sp):
    """sum over links l-sp..l+sp (edges truncated); a [..., L]"""
    if not sp:
        return a
    c = np.cumsum(np.pad(a, [(0, 0)] * (a.ndim - 1) + [(sp + 1, sp)]), -1)
    return c[..., 2 * sp + 1:] - c[..., :-(2 * sp + 1)]


def item_stats(b, items, box, sp=0, sameday=False, eps=0.02):
    """items: list of (window index, evidence days (array, all < d), Evidence).  box = (lo, hi) slot offsets.
    Returns S[slotkind][grp] = dict(O, e, X, nd, F, LL0, R) arrays [n_items, L]; slotkind 's6' (slot T+6) / 't' (slot T);
    grp 'same' (evidence days of the window's weekday class), 'other' (the other class), 'sd' (same day, slots < T; only if
    sameday).  nd = number of days with at least one visible cell in the box.  Per-day sums over those days (for the
    mixture design): F = sum of the daily queued shares f = O/e, R = sum of the daily train references r = X/e,
    LL0 = sum of f log r' + (1 - f) log(1 - r') with r' = clip(r, eps, 1 - eps) (quasi log-likelihood under the train prior)."""
    lo, hi = box; n = len(items); L = b['L']
    QS = ('O', 'e', 'X', 'nd', 'F', 'LL0', 'R')
    S = {sk: {g: {q: np.zeros((n, L), np.float32) for q in QS} for g in ('same', 'other', 'sd')} for sk in ('s6', 't')}

    def put(T, k, O, e, X):
        ok = e > 0
        f = np.where(ok, O / np.maximum(e, 1e-9), 0.0); r = np.where(ok, X / np.maximum(e, 1e-9), 0.0)
        rc = np.clip(r, eps, 1 - eps)
        T['O'][k] = O.sum(0); T['e'][k] = e.sum(0); T['X'][k] = X.sum(0); T['nd'][k] = ok.sum(0)
        T['F'][k] = f.sum(0); T['R'][k] = r.sum(0)
        T['LL0'][k] = np.where(ok, f * np.log(rc) + (1 - f) * np.log(1 - rc), 0.0).sum(0)
    for k, (i, days, evd) in enumerate(items):
        d, t = int(b['d'][i]), int(b['t'][i]); c = bool(b['wk'][d])
        days = np.asarray(days, int)
        for sk, s in (('s6', min(t + 6, 287)), ('t', t)):
            if len(days):
                O, e, X = evd.box(days, s, lo, hi)
                O, e, X = link_box(O, sp), link_box(e, sp), link_box(X, sp)
                same = b['wk'][days] == c
                for g, m in (('same', same), ('other', ~same)):
                    if m.any():
                        put(S[sk][g], k, O[m], e[m], X[m])
            if sameday:
                O, e, X = evd.box(np.array([d]), s, lo, hi, tmax=t)      # slots < T only
                put(S[sk]['sd'], k, link_box(O, sp), link_box(e, sp), link_box(X, sp))
    return S


def design_priors(base3, S, box, sp=0, ref='box', k=4.0, cw=1.0, ncount='days', direction='both', cols='all', sameday=False,
                  cw_all=1.0, kneg=None, form='blend', pi=0.5, mu=0.05, nu=1.0):
    """base3 [n_items, L, 3] current prior columns (prior, prior_now, prior_alldays).  Returns the substituted copy.
    cw: weight of other-class evidence days in the two class-specific columns (0 = separate, 1 = pooled);
    cw_all: weight of other-class days in prior_alldays (1 = all days, as production sp4).
    form 'blend':  new = clip(base + a (O/e - ref)), a = n / (n + k);  ref='base' -> classic blend, ref='box' -> residual.
        ncount: 'obs' -> n = visible observations / box length (production sp4), 'days' -> n = days with a visible cell.
        kneg: separate shrinkage k for negative deviations.
    form 'bma':  per-cell two-hypothesis mixture.  H0: the scenario follows the train prior (daily box share ~ train reference r);
        H1: new unknown share theta ~ Beta(mu nu, (1 - mu) nu).  Each evidence day is one trial with fractional outcome f.
        w1 = P(H1 | days), prior odds (1 - pi) / pi;  new = clip(base + w1 (theta_hat - mean r)).
    direction: both | pos (only raise) | neg (only lower)."""
    from scipy.special import betaln
    lo, hi = box; blen = float((hi - lo + 1) * (2 * sp + 1))
    out = base3.copy()
    use = {'all': (0, 1, 2), 'prior': (0,), 'prior_all': (0, 2), 'prior_now': (0, 1), 'now': (1,), 'alld': (2,)}[cols]
    for j, (sk, w_o) in enumerate((('s6', cw), ('t', cw), ('s6', cw_all))):
        if j not in use:
            continue
        T = S[sk]

        def tot(q):
            v = T['same'][q] + w_o * T['other'][q]
            return v + T['sd'][q] if sameday else v
        if form == 'blend':
            O, e, X, nd = tot('O'), tot('e'), tot('X'), tot('nd')
            n = e / blen if ncount == 'obs' else nd
            ok = e > 0
            est = np.where(ok, O / np.maximum(e, 1e-9), 0.0)
            rf = base3[..., j] if ref == 'base' else np.where(ok, X / np.maximum(e, 1e-9), 0.0)
            dev = np.where(ok, est - rf, 0.0)
            a = n / (n + k)
            if kneg is not None:
                a = np.where(dev < 0, n / (n + kneg), a)
            dev = a * dev
        else:
            n, Fs, LL0, R = tot('nd'), tot('F'), tot('LL0'), tot('R')
            al, be = mu * nu, (1 - mu) * nu
            ll1 = betaln(al + Fs, be + n - Fs) - betaln(al, be)
            z = np.log((1 - pi) / pi) + ll1 - LL0
            w1 = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
            th = (al + Fs) / (al + be + n)
            rbar = np.where(n > 0, R / np.maximum(n, 1e-9), 0.0)
            dev = np.where(n > 0, w1 * (th - rbar), 0.0)
        if direction == 'pos':
            dev = np.maximum(dev, 0)
        elif direction == 'neg':
            dev = np.minimum(dev, 0)
        out[..., j] = np.clip(base3[..., j] + dev, 0.0, 1.0)
    return out


# ------------------------------------------------------------------ model + scoring
def load_cat(path='results/splitprior/cat_onset_noramp.cbm'):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(); m.load_model(path)
    return m


def predict_items(cat, b, widx, prior3=None):
    """probabilities [n_items, L] for the windows widx with the prior columns replaced by prior3 [n_items, L, 3] (None = base)."""
    X = b['X'][widx].copy()
    if prior3 is not None:
        X[..., JP] = prior3
    n, L, C = X.shape
    return cat.predict_proba(X.reshape(n * L, C)[:, KEEPI], thread_count=lab.THREADS)[:, 1].reshape(n, L)


def iou_items(b, widx, p):
    """production decode (expected-IoU top-m over the 6 x L cells; only step 6 carries probability) + visible-cell IoU."""
    L = b['L']; out = np.zeros(len(widx)); npred = np.zeros(len(widx), int)
    for k, i in enumerate(widx):
        P6 = np.zeros((F, L), np.float32); P6[F - 1] = p[k]
        pred = lab.decode(P6.ravel()).reshape(F, L)
        T = b['truth'][i] & b['ev'][i]; Pp = pred & b['ev'][i]
        u = (T | Pp).sum()
        out[k] = 1.0 if u == 0 else (T & Pp).sum() / u
        npred[k] = pred.sum()
    return out, npred


def frame(b, widx, iou, tag=None, extra=None):
    df = pd.DataFrame(dict(panel=b['panel'], family=b['family'], d=b['d'][widx], t=b['t'][widx],
                           cond=lab.ONSET if tag is None else tag, iou=iou))
    for k, v in (extra or {}).items():
        df[k] = v
    return df


def boot(A, B, col='iou'):
    bt = lab.paired_bootstrap(A, B, col=col)
    key = ['panel', 'd', 't', 'cond']
    m = A[key + [col]].merge(B[key + [col]], on=key, suffixes=('_a', '_b'))
    pp = m.groupby('panel')[[f'{col}_a', f'{col}_b']].mean()
    dd = pp[f'{col}_b'] - pp[f'{col}_a']
    return dict(diff=bt['diff'], lo=bt['lo'], hi=bt['hi'], p_pos=bt['p_pos'], n=bt['n_windows'],
                pos=int((dd > 1e-9).sum()), neg=int((dd < -1e-9).sum()), npan=int(len(dd)))


def fmt(bt):
    return f"{bt['diff']:+.4f} [{bt['lo']:+.4f}, {bt['hi']:+.4f}] panels +{bt['pos']}/-{bt['neg']} of {bt['npan']} (n={bt['n']})"
