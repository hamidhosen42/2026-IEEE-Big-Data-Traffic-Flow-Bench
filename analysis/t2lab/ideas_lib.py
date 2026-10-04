"""Track `ideas`: library for the lab test of the research agent's ideas (NEW file; nothing existing is modified).

Idea 1  day-excluded (leave-one-day-out / leave-one-group-out) queue priors in the TRAINING rows.
Idea 2  time-free onset-site share + time-tolerant train priors as onset features (+ blind-window fallback).
Idea 3  prior-timing jitter (augmented copies of the train onset windows with the prior clock shifted).
Idea 5  flow relative to the site's own pre-breakdown flow.

Everything is derived from the train split (labels of the allowed train days) and, for a window, from data at or before T.
`Ctx(panel, day_mask)` holds the sums; `Ctx.tables(excl_days)` returns every train-derived table with a set of days removed:
  prior [3, 288, L]   production prior_table without the excluded days
  tol6 / tol12 [288, L]   max of the slot prior over +-6 / +-12 slots and both day classes
  site_on [L]         share of onset events (production onset training windows, queue first observed at T+30) containing the link
  site_rel, site_dist, site_rank [L]   site_on / max, distance to the nearest link with site_on >= 0.1, number of stronger links
  prior_day [L]       share of days with any queued observation on the link
  qpre [L]            median over the link's onset events of its flow in T-15..T-5 (vph); NaN with fewer than 3 events
`row_feats(ctx, group, d, t, l, h)` gathers them per row, excluding all days of the row's own group
(group = arange(D): leave-one-day-out; group = (d // 7) % 4: leave-week-group-out; group = None: nothing excluded).
"""
import gc, os, pickle, sys, time, types, warnings
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d
import lab
from exp_firstcross_prod_snippet import fill_time36

H, F = lab.H, lab.F
PRIOR3 = ('prior', 'prior_now', 'prior_alldays')
COLS = lab.all_cols()
KEEPI = [i for i, c in enumerate(COLS) if not c.startswith('ramp_')]          # production v13 onset columns (lab NORAMP=1)
JP = [COLS.index(c) for c in PRIOR3]
OUT = 'results/ideas'
CACHE = f'{lab.CACHE}/ideas'
for _d in (OUT, CACHE):
    os.makedirs(_d, exist_ok=True)
SITE = ['site_on', 'site_rel', 'site_dist', 'site_rank', 'prior_day']
TOL = ['prior_tol6', 'prior_tol12']
FLOW = ['fq_site', 'fq_site_nb', 'fq_site_slope', 'fq_site_rank']
EXTRA = SITE + TOL + FLOW
SITE_MIN = 0.1


# ------------------------------------------------------------------ train-derived tables with excluded days
def onset_events(P, panel):
    """production onset training windows (dense1500 onset windows): day, slot, queued / observed links at T+30, flow T-15..T-5."""
    win = lab.window_set(panel, 'dense1500'); win = win[win.cond == lab.ONSET]
    d = win.d.to_numpy(); t = win.t.to_numpy(); s6 = np.minimum(t + 6, 287)
    idx = t[:, None] + np.arange(-3, 0)[None]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q3 = np.nanmean(P['Fq'][d[:, None], idx], 1)                       # [n_ev, L] mean observed flow in T-15..T-5
    return dict(d=d, t=t, B=P['Q'][d, s6], O=P['E'][d, s6], q3=q3)


class Ctx:
    def __init__(self, panel, day_mask=None, P=None):
        P = P or lab.load_panel(panel)
        D = len(P['days'])
        day_mask = np.ones(D, bool) if day_mask is None else np.asarray(day_mask, bool)
        self.P, self.panel, self.mask, self.D, self.L = P, panel, day_mask, D, len(P['links'])
        self.wk = P['dow'] >= 5
        self.S, self.N = self._sums(day_mask)
        self.anyq = P['Q'].any(1)                                         # [D, L]
        self.ev = onset_events(P, panel)

    def _sums(self, m):
        Q, E, wk = self.P['Q'], self.P['E'], self.wk
        m3 = (m & ~wk, m & wk, m)
        return (np.stack([Q[x].sum(0) for x in m3]).astype(np.float64), np.stack([E[x].sum(0) for x in m3]).astype(np.float64))

    def tables(self, excl=None, flow=True):
        """excl: bool [D] days to remove on top of ~day_mask (None = nothing)."""
        m = self.mask if excl is None else (self.mask & ~excl)
        if excl is None:
            S, N = self.S, self.N
        else:
            Sx, Nx = self._sums(self.mask & excl)
            S, N = self.S - Sx, self.N - Nx
        prior = (S / np.maximum(N, 1)).astype(np.float32)                 # == production prior_table(P, m)
        pm = np.maximum(prior[0], prior[1])
        out = dict(prior=prior, tol6=maximum_filter1d(pm, 13, axis=0, mode='nearest'), tol12=maximum_filter1d(pm, 25, axis=0, mode='nearest'))
        ev = self.ev; ke = m[ev['d']]
        nb = ev['B'][ke].sum(0).astype(np.float64); no = ev['O'][ke].sum(0).astype(np.float64)
        site = (nb / np.maximum(no, 1)).astype(np.float32); L = self.L
        out['site_on'] = site
        out['site_rel'] = site / max(float(site.max()), 1e-9)
        pos = np.where(site >= SITE_MIN)[0]; idx = np.arange(L)
        out['site_dist'] = (np.abs(idx[:, None] - pos[None]).min(1) if len(pos) else np.full(L, 99)).astype(np.float32)
        out['site_rank'] = np.minimum((site[None, :] > site[:, None]).sum(1), 30).astype(np.float32)
        out['prior_day'] = (self.anyq[m].sum(0) / max(float(m.sum()), 1.0)).astype(np.float32)
        if flow:
            q = np.where(ev['B'][ke], ev['q3'][ke], np.nan)
            cnt = np.isfinite(q).sum(0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                med = np.nanmedian(q, 0) if len(q) else np.full(L, np.nan)
            out['qpre'] = np.where(cnt >= 3, med, np.nan).astype(np.float32)
        return out


def row_feats(ctx, group, d, t, l, h, flow=True, cls=None):
    """per row: prior3 [n, 3] (prior, prior_now, prior_alldays) and extra [n, len(SITE + TOL)] + qpre [n] with all days of the
    row's own group removed (group None: nothing removed).  d, t, l, h: day, origin slot, link, step per row.
    cls: day-class override per row (idea 3: the prior clock / class of an augmented copy); the excluded days still follow d."""
    n = len(d); cls = ctx.wk[d].astype(int) if cls is None else np.asarray(cls, int); sh = np.minimum(t + h, 287)
    p3 = np.zeros((n, 3), np.float32); ex = np.zeros((n, len(SITE) + len(TOL)), np.float32); qpre = np.full(n, np.nan, np.float32)
    if group is None:
        gr = np.zeros(n, int); todo = [(0, None)]
    else:
        gr = group[d]; todo = [(g, group == g) for g in np.unique(gr)]
    for g, excl in todo:
        r = np.where(gr == g)[0]
        T = ctx.tables(excl, flow)
        c = cls[r]; lr = l[r]
        p3[r, 0] = T['prior'][c, sh[r], lr]; p3[r, 1] = T['prior'][c, t[r], lr]; p3[r, 2] = T['prior'][2, sh[r], lr]
        for j, nm in enumerate(SITE):
            ex[r, j] = T[nm][lr]
        ex[r, len(SITE)] = T['tol6'][sh[r], lr]; ex[r, len(SITE) + 1] = T['tol12'][sh[r], lr]
        if flow:
            qpre[r] = T['qpre'][lr]
    return p3, ex, qpre


def flow_feats(X, qpre, W, L, cap):
    """idea 5: [n, 4] flow in T-15..T-5 relative to the link's own pre-onset flow, the mean of the same ratio over the +-3
    neighbour links, the 45-min flow change in the same unit, rank of the ratio within the window (0 = largest).
    Rows must be the COMPLETE (window, link) grid in C order.  X: full-column matrix (uses fr_mean3, fr_slope)."""
    c = COLS.index; capr = np.tile(cap, W).astype(np.float32)
    assert len(X) == W * L
    with np.errstate(all='ignore'):
        r = ((X[:, c('fr_mean3')] * capr) / qpre).astype(np.float32).reshape(W, L)       # NaN where the link has < 3 onset events
        slope = ((X[:, c('fr_slope')] * capr) / qpre).astype(np.float32).reshape(W, L)
    fin = np.isfinite(r); v = np.where(fin, r, 0.0).astype(np.float64)
    cs = np.concatenate([np.zeros((W, 1)), np.cumsum(v, 1)], 1); cn = np.concatenate([np.zeros((W, 1)), np.cumsum(fin, 1)], 1)
    lo = np.maximum(np.arange(L) - 3, 0); hi = np.minimum(np.arange(L) + 4, L)
    sm = cs[:, hi] - cs[:, lo] - v; nn = cn[:, hi] - cn[:, lo] - fin
    nb = np.where(nn > 0, sm / np.maximum(nn, 1), np.nan)
    rr = np.where(fin, r, -np.inf)
    order = np.argsort(-rr, 1, kind='stable'); ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.repeat(np.arange(L)[None], W, 0), 1)
    rk = np.where(fin, ranks, np.nan)
    return np.stack([r, nb, slope, rk], -1).reshape(W * L, 4).astype(np.float32)


# ------------------------------------------------------------------ March pseudo-onset windows (copied from exp_onset_splitprior.py)
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
    return on, og


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


def cum_tables(Pv, smooth=True):
    """cumQ, cumE [3, D+1, 288, L]: sums over the days < d of class c (0 weekday, 1 weekend, 2 all); smooth -> slots +-2."""
    Q = Pv['Q'].astype(np.float32); E = Pv['E'].astype(np.float32)
    if smooth:
        def box(a):
            c = np.cumsum(np.pad(a, ((0, 0), (3, 2), (0, 0))), 1)
            return c[:, 5:] - c[:, :-5]
        Q, E = box(Q), box(E)
    D = Q.shape[0]; wk = Pv['dow'] >= 5
    cQ = np.zeros((3, D + 1) + Q.shape[1:], np.float32); cE = cQ.copy()
    for c, m in enumerate((~wk, wk, np.ones(D, bool))):
        cQ[c, 1:] = np.cumsum(Q * m[:, None, None], 0); cE[c, 1:] = np.cumsum(E * m[:, None, None], 0)
    return cQ, cE


def march_bundle(panel, refresh=False):
    """the 214 pseudo-March onset windows of exp_onset_splitprior.py (masked-layer histories, visible labels), all-train prior
    in the prior columns, the same-scenario prior (days < d, slots +-2) for the s<k> blends, and the idea-2/5 extras."""
    f = f'{CACHE}/march_{panel}.pkl'
    if os.path.exists(f) and not refresh:
        return pickle.load(open(f, 'rb'))
    P, Pv = val_panel(panel)
    on, og = select(Pv)
    D = len(Pv['days'])
    g_on, g_og = lab._glist(on), lab._glist(og); s = set()
    for d0 in range(D):
        a, _ = lab.greedy(g_on, g_og, 'interleaved', start_day=d0); s.update(a)
    wins = sorted(s)
    ctx = Ctx(panel, P=P)
    prior_tr = lab.prior_table(P, np.ones(len(P['days']), bool))
    R = onset_rows(P, Pv, [(a, b, lab.ONSET) for a, b in wins], prior_tr)
    X, y, e, meta, call = lab.to_matrix(R, lab.ONSET, None, None)
    assert list(call) == COLS
    wi, li = meta[:, 0], meta[:, 1]
    d, t = R['d'], R['t']; cls = (Pv['dow'][d] >= 5).astype(int); s6 = np.minimum(t + 6, 287)
    c_r, d_r, s6_r, t_r = cls[wi], d[wi], s6[wi], t[wi]
    trv = np.stack([prior_tr[c_r, s6_r, li], prior_tr[c_r, t_r, li], prior_tr[2, s6_r, li]], 1)
    assert np.array_equal(trv, X[:, JP])
    cQ, cE = cum_tables(Pv, True)
    sp, nn = [], []
    for cc, ss in ((c_r, s6_r), (c_r, t_r), (np.full_like(c_r, 2), s6_r)):
        q = cQ[cc, d_r, ss, li]; ee = cE[cc, d_r, ss, li]
        sp.append(np.where(ee > 0, q / np.maximum(ee, 1e-9), 0.0)); nn.append(ee / 5.0)
    # train-derived extras for a NEW-scenario window: nothing excluded; class / slot of the March window
    T = ctx.tables(None)
    ex = np.zeros((len(X), len(SITE) + len(TOL)), np.float32)
    for j, nm in enumerate(SITE):
        ex[:, j] = T[nm][li]
    ex[:, len(SITE)] = T['tol6'][s6_r, li]; ex[:, len(SITE) + 1] = T['tol12'][s6_r, li]
    assert np.array_equal(wi, np.repeat(np.arange(R['W']), R['L'])) and np.array_equal(li, np.tile(np.arange(R['L']), R['W']))
    fl = flow_feats(X, T['qpre'][li], R['W'], R['L'], P['cap'])
    assert np.allclose(T['prior'], prior_tr, atol=1e-6)
    b = dict(panel=panel, family=panel.rsplit('_', 1)[0], L=R['L'], W=R['W'], D=D, X=X, EX=np.concatenate([ex, fl], 1), wi=wi, li=li,
             d=d, t=t, y=y, e=e, sp=np.stack(sp, 1).astype(np.float32), nn=np.stack(nn, 1).astype(np.float32),
             truth=np.stack([Pv['Q'][d, t + h] for h in range(1, F + 1)], 1), ev=np.stack([Pv['E'][d, t + h] for h in range(1, F + 1)], 1),
             qlast=R['qlast'], dow=Pv['dow'][d], maxprior=np.array([prior_tr[c, s, :].max() for c, s in zip(cls, s6)]),
             site_on=T['site_on'], days=Pv['days'])
    pickle.dump(b, open(f, 'wb'), protocol=4)
    lab.drop_panel(panel)
    return b


def march_rc(b):
    win = pd.DataFrame(dict(d=b['d'], t=b['t'], cond=lab.ONSET, fold=0))
    return types.SimpleNamespace(truth=b['truth'], ev=b['ev'], qlast=b['qlast'], win=win, name='march_on', panel=b['panel'],
                                 cond=lab.ONSET, L=b['L'])


def march_prior(b, k=None):
    """prior columns [n, 3] of the March rows: k None -> all-train prior; else the s<k> blend with the March days < d."""
    tr = b['X'][:, JP]
    if k is None:
        return tr
    a = b['nn'] / (b['nn'] + k)
    return ((1 - a) * tr + a * b['sp']).astype(np.float32)


# ------------------------------------------------------------------ train rows / fold evaluation rows
def train_rows(panel, k, groups=('L1', 'L7')):
    """dense1500 onset training rows outside fold k (k = 4: all days).  Returns (dict, ctx): X (prior columns = the self-inclusive
    prior of the allowed days, as production), y, e (label observed: production trains on e rows only), row keys, EX_S (extras
    from the self-inclusive tables) and P3_<g> / EX_<g> (day-excluded priors / extras; L1 leave-one-day-out, L7 leave-week-group-out).
    All arrays cover the complete (window, link) grid; apply `e` before fitting."""
    P = lab.load_panel(panel); D = len(P['days'])
    rc = lab.rows(panel, 'dense1500', lab.ONSET)
    widx = np.where(rc.win.fold.to_numpy() != k)[0]; W, L = len(widx), rc.L
    m = np.isin(rc.wi, widx)
    X = rc.matrix(m, COLS, k)
    wi = rc.wi[m].astype(int); l = rc.li[m].astype(int); h = rc.h[m].astype(int)
    assert np.array_equal(wi, np.repeat(widx, L)) and np.array_equal(l, np.tile(np.arange(L), W)) and (h == 6).all()
    d = rc.win.d.to_numpy()[wi]; t = rc.win.t.to_numpy()[wi]
    out = dict(X=X, y=rc.y[m].astype(int), e=rc.e[m], d=d, t=t, l=l, h=h, W=W, L=L, panel=panel, cap=P['cap'])
    mask = (P['block'] != k) if k < 4 else np.ones(D, bool)
    ctx = Ctx(panel, mask, P)
    p3, ex, qpre = row_feats(ctx, None, d, t, l, h)
    assert np.allclose(p3, X[:, JP], atol=1e-6), float(np.abs(p3 - X[:, JP]).max())
    out['EX_S'] = np.concatenate([ex, flow_feats(X, qpre, W, L, P['cap'])], 1)
    for nm, grp in (('L1', np.arange(D)), ('L7', (np.arange(D) // 7) % 4)):
        if nm in groups:
            p3, ex, qpre = row_feats(ctx, grp, d, t, l, h)
            out[f'P3_{nm}'] = p3
            out[f'EX_{nm}'] = np.concatenate([ex, flow_feats(X, qpre, W, L, P['cap'])], 1)
    return out, ctx


def eval_rows(panel, k, ctx, set_name='offlike'):
    """held-out fold-k onset windows of `set_name` with the fold-safe prior and the fold-safe extras (nothing of fold k inside)."""
    P = ctx.P
    rc = lab.rows(panel, set_name, lab.ONSET)
    widx = np.where(rc.win.fold.to_numpy() == k)[0]; W, L = len(widx), rc.L
    me = np.isin(rc.wi, widx)
    X = rc.matrix(me, COLS, k)
    wi = rc.wi[me].astype(int); l = rc.li[me].astype(int); h = rc.h[me].astype(int)
    assert np.array_equal(wi, np.repeat(widx, L)) and np.array_equal(l, np.tile(np.arange(L), W)) and (h == 6).all()
    wloc = np.repeat(np.arange(W), L)
    d = rc.win.d.to_numpy()[wi]; t = rc.win.t.to_numpy()[wi]
    p3, ex, qpre = row_feats(ctx, None, d, t, l, h)
    assert np.allclose(p3, X[:, JP], atol=1e-6), float(np.abs(p3 - X[:, JP]).max())
    EX = np.concatenate([ex, flow_feats(X, qpre, W, L, P['cap'])], 1)
    sub = types.SimpleNamespace(truth=rc.truth[widx], ev=rc.ev[widx], qlast=rc.qlast[widx], win=rc.win.iloc[widx].reset_index(drop=True),
                                name=f'{set_name}_cv', panel=panel, cond=lab.ONSET, L=rc.L)
    T = ctx.tables(None, False)
    wd = sub.win.d.to_numpy(); wt = sub.win.t.to_numpy()
    maxprior = np.array([T['prior'][int(ctx.wk[a]), min(b + 6, 287), :].max() for a, b in zip(wd, wt)])
    return dict(sub=sub, X=X, EX=EX, wloc=wloc, l=l, d=d, t=t, maxprior=maxprior, site_on=T['site_on'], panel=panel, W=W, L=L)


def score(sub, wloc, l, p, arm, set_name=None):
    P6 = np.zeros((len(sub.win), F, sub.L), np.float32); P6[wloc, F - 1, l] = p
    return lab.score_probs(sub, P6, arm=arm, set_name=set_name or sub.name)


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
