"""DOWNSIDE of the private onset bet: what if the private month were NOT the March scenario but train-like?
Emulation: target = held-out TRAIN days (fold 0, official-like onset windows 'offlike'); the 'same-scenario' data we would wrongly
use = the released March month.  Arms (CatBoost onset as production, trained on train rows of folds 1-3 with the fold-safe prior):
  A        fold-safe train prior (production behaviour on a train-like month)
  A_mp     A with the prior columns blended with the March prior (all 31 days, a = n/(n+4))      = sp4 on a train-like month
  B_mp     bag (w 10/20/40) trained on folds 1-3 rows + all March onset windows (blended prior), evaluated with the March blend
           = the private pv bag on a train-like month
Decision support for the hedge (second final pick)."""
import gc, os, sys, time, types, warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lab
from exp_firstcross_prod_snippet import fill_time36

H, F = lab.H, lab.F
K = 4
NORAMP = True
OUT = 'results/splitprior_wrong'; os.makedirs(OUT, exist_ok=True)
PRIOR3 = ('prior', 'prior_now', 'prior_alldays')
cols = lab.all_cols(); JP = [cols.index(c) for c in PRIOR3]
KEEPC = [i for i, c in enumerate(cols) if not (NORAMP and c.startswith('ramp_'))]
pd.set_option('display.width', 220)


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


def onset_cands(Pv):
    V, E, Q, Em = Pv['V'], Pv['E'], Pv['Q'], Pv['Em']; D = V.shape[0]
    blank = (Em.sum(2) >= 5) & (E.sum(2) == 0)
    on = np.zeros((D, 288), bool)
    for d in range(D):
        for t in range(H, 288 - F):
            if blank[d, t - H:t + F + 1].any() or Em[d, t - H:t].mean() < 0.7 or lab.established(Q[d, t - H:t]):
                continue
            on[d, t] = Q[d, t + 1:t + 1 + F].any()
    return on


def onset_rows(P, Pv, wins, prior):
    d = np.array([w[0] for w in wins]); t = np.array([w[1] for w in wins])
    idx = t[:, None] + np.arange(-H, 0)[None]
    Vh = Pv['V'][d[:, None], idx]; Fh = Pv['Fq'][d[:, None], idx]; VT = Pv['V'][d, t]
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        Fe, qlast = lab.window_features(P, Vh, Fh, t, Pv['dow'][d], prior, VT)
        lab.add_ramp(Fe, Pv['ON'], Pv['OFF'], d, t, P['cap'])
    W, L = qlast.shape
    feats = dict(Fe); feats['h'] = np.full((W, L), 6, np.float32)
    feats['prior'], feats['prior_now'], feats['prior_alldays'] = lab.prior_feats(prior, Pv['dow'][d], t, 6)
    return dict(d=d, t=t, cond=np.array([lab.ONSET] * W), qlast=qlast, W=W, L=L,
                rows=[(6, feats, Pv['Q'][d, np.minimum(t + 6, 287)], Pv['E'][d, np.minimum(t + 6, 287)])])


def cum_tables(Pv):
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


def blend(Xp, cls_r, s6_r, t_r, l_r, day_r, cQ, cE):
    """Xp: [n, 3] base prior values (prior, prior_now, prior_alldays); returns blended copy using March days < day_r."""
    out = Xp.copy()
    for j, (cc, ss) in enumerate(((cls_r, s6_r), (cls_r, t_r), (np.full_like(cls_r, 2), s6_r))):
        q = cQ[cc, day_r, ss, l_r]; ee = cE[cc, day_r, ss, l_r]
        sp = np.where(ee > 0, q / np.maximum(ee, 1e-9), 0.0); n = ee / 5.0; a_ = n / (n + K)
        out[:, j] = (1 - a_) * Xp[:, j] + a_ * sp
    return out


# ---- train rows (folds 1-3, fold-safe prior k=0) and fold-0 evaluation windows (offlike onset) with the March blend
Xt, yt, EV, MT = [], [], [], []
for panel in lab.PANELS:
    rc = lab.rows(panel, 'dense1500', lab.ONSET)
    m = rc.e & (rc.fold != 0)
    Xt.append(rc.matrix(m, cols, 0)); yt.append(rc.y[m].astype(int))
    P, Pv = val_panel(panel)
    cQ, cE = cum_tables(Pv); D = len(Pv['days'])
    ev = lab.rows(panel, 'offlike', lab.ONSET)
    widx = np.where(ev.win.fold.to_numpy() == 0)[0]
    me = np.isin(ev.wi, widx)
    Xe = ev.matrix(me, cols, 0)
    win = ev.win
    d_r = win.d.to_numpy()[ev.wi[me]]; t_r = win.t.to_numpy()[ev.wi[me]]; l_r = ev.li[me].astype(int)
    cls_r = (P['dow'][d_r] >= 5).astype(int); s6_r = np.minimum(t_r + 6, 287)
    Xe_mp = Xe.copy(); Xe_mp[:, JP] = blend(Xe[:, JP], cls_r, s6_r, t_r, l_r, np.full_like(d_r, D), cQ, cE)   # all 31 March days
    sub = types.SimpleNamespace(truth=ev.truth[widx], ev=ev.ev[widx], qlast=ev.qlast[widx], win=win.iloc[widx].reset_index(drop=True),
                                name='train_fold0_offlike', panel=panel, cond=lab.ONSET, L=ev.L)
    remap = {w: i for i, w in enumerate(widx)}
    EV.append((sub, Xe, Xe_mp, np.array([remap[w] for w in ev.wi[me]]), l_r))
    # all March onset training windows (spaced picks), blended prior with March days < own day
    on = onset_cands(Pv); g = lab._glist(on); picks = []; last = -10 ** 9
    for x in g:
        if x - last >= lab.SPACING:
            picks.append((int(x // 288), int(x % 288))); last = x
    prior_tr = lab.prior_table(P, P['block'] != 0)
    R = onset_rows(P, Pv, [(a, b, lab.ONSET) for a, b in picks], prior_tr)
    X, y, e, meta, call = lab.to_matrix(R, lab.ONSET, None, None); assert list(call) == cols
    dm = R['d'][meta[:, 0]]; tm = R['t'][meta[:, 0]]; lm = meta[:, 1].astype(int)
    X[:, JP] = blend(X[:, JP], (Pv['dow'][dm] >= 5).astype(int), np.minimum(tm + 6, 287), tm, lm, dm, cQ, cE)
    MT.append((X[e], y[e].astype(int)))
    lab.log(panel, 'fold-0 eval windows', len(widx), '| March training windows', len(picks))
    del cQ, cE; lab.drop_panel(panel); gc.collect()

Xt = np.concatenate(Xt); yt = np.concatenate(yt)
Xm = np.concatenate([a for a, _ in MT]); ym = np.concatenate([b for _, b in MT])
kc = KEEPC
catA = CatBoostClassifier(iterations=600, depth=8, learning_rate=0.08, thread_count=lab.THREADS, verbose=0, random_seed=0)
catA.fit(Xt[:, kc], yt); lab.log('A trained')
bag = []
for w in (10, 20, 40):
    m_ = CatBoostClassifier(iterations=600, depth=8, learning_rate=0.08, thread_count=lab.THREADS, verbose=0, random_seed=0)
    m_.fit(np.concatenate([Xt[:, kc], Xm[:, kc]]), np.concatenate([yt, ym]), sample_weight=np.r_[np.ones(len(Xt)), np.full(len(Xm), float(w))])
    bag.append(m_); lab.log(f'B w{w} trained')
res = []
for sub, Xe, Xe_mp, wloc, l_r in EV:
    W, L = len(sub.win), sub.L
    for name, p in (('A', catA.predict_proba(Xe[:, kc])[:, 1]), ('A_mp', catA.predict_proba(Xe_mp[:, kc])[:, 1]),
                    ('B_mp', np.mean([m_.predict_proba(Xe_mp[:, kc])[:, 1] for m_ in bag], 0))):
        P6 = np.zeros((W, F, L), np.float32); P6[wloc, F - 1, l_r] = p
        res.append(lab.score_probs(sub, P6, arm=name, set_name=sub.name))
Rr = pd.concat(res, ignore_index=True); Rr.to_pickle(f'{OUT}/train_fold0_wrongscen.pkl')
A0 = Rr[Rr.arm == 'A']
print('\n== held-out TRAIN days (fold 0), official-like onset windows: n =', len(A0))
for a in ('A', 'A_mp', 'B_mp'):
    B0 = Rr[Rr.arm == a]
    bt = lab.paired_bootstrap(A0, B0) if a != 'A' else dict(diff=0, lo=0, hi=0)
    pa = A0.groupby('panel').iou.mean(); pb = B0.groupby('panel').iou.mean()
    print(f'{a:6s} {lab.aggregate(B0):.4f}  vs A {bt["diff"]:+.4f} [{bt["lo"]:+.4f}, {bt["hi"]:+.4f}]  panels+ {int((pb - pa > 1e-9).sum())}/{len(pa)}')
lab.log('DONE')
