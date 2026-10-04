"""Onset for the private month: does adding SAME-SCENARIO training windows (released March, observed labels) to the train-split
onset model help beyond the same-scenario prior?  Temporal split inside March: training windows from March days < 15 only,
evaluation on the pseudo-onset windows of days >= 15 (official-like greedy, as exp_onset_splitprior.py).
Arms (CatBoost onset as production v13 params, cols = lab production columns):
  A      train split only, train prior at evaluation                 (production)
  A_s4   train split only, same-scenario blended prior (s4)          (= kernel tfb-t2osp sp4)
  B<w>   train split + March days < 15 onset windows (weight w), same-scenario blended prior both in the March training rows
         (prior of the March days before each window's day) and at evaluation
For April this corresponds to: train + all 31 March days, scenario prior from March + April days < d.
Decision support only (March labels): may steer the private file."""
import gc, json, os, sys, time, types, warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lab
from exp_firstcross_prod_snippet import fill_time36

H, F = lab.H, lab.F
OUT = 'results/splitprior_mt'; os.makedirs(OUT, exist_ok=True)
TAG = ('_'.join(sys.argv[1:]) or 'default') + ('_noramp' if os.environ.get('NORAMP') == '1' else '')
CUT_DAY = int(sys.argv[1]) if len(sys.argv) > 1 else 15
K = 4
WS = tuple(float(x) for x in sys.argv[2].split(',')) if len(sys.argv) > 2 else (1, 5, 20)
DENSE = len(sys.argv) > 3 and sys.argv[3] == 'dense'
EVAL_FROM = int(sys.argv[4]) if len(sys.argv) > 4 else 15
PRIOR3 = ('prior', 'prior_now', 'prior_alldays')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
cols = lab.all_cols(); JP = [cols.index(c) for c in PRIOR3]
NORAMP = os.environ.get('NORAMP') == '1'                     # v13-like feature set: drop the ramp columns (model input only)
KEEPC = [i for i, c in enumerate(cols) if not (NORAMP and c.startswith('ramp_'))]


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


def blend_rows(X, R, meta, Pv, prior_tr, cQ, cE, k):
    """X copy with the three prior columns blended per row (window day d -> days < d of the scenario)."""
    wi, li = meta[:, 0], meta[:, 1]
    d, t = R['d'], R['t']; cls = (Pv['dow'][d] >= 5).astype(int); s6 = np.minimum(t + 6, 287)
    cr, dr, s6r, tr_ = cls[wi], d[wi], s6[wi], t[wi]
    trv = [prior_tr[cr, s6r, li], prior_tr[cr, tr_, li], prior_tr[2, s6r, li]]
    Xk = X.copy()
    for j, (cc, ss) in enumerate(((cr, s6r), (cr, tr_), (np.full_like(cr, 2), s6r))):
        q = cQ[cc, dr, ss, li]; ee = cE[cc, dr, ss, li]
        sp = np.where(ee > 0, q / np.maximum(ee, 1e-9), 0.0); n = ee / 5.0; a_ = n / (n + k)
        Xk[:, JP[j]] = (1 - a_) * trv[j] + a_ * sp
    return Xk


# ---- March rows: training windows (days < CUT_DAY, production-style onset windows: greedy within onset, 72-slot spacing)
#      and evaluation windows (days >= CUT_DAY, official-like multi-start greedy)
tr_X, tr_y, ev = [], [], []
for panel in lab.PANELS:
    P, Pv = val_panel(panel)
    on, og = select(Pv)
    D = len(Pv['days'])
    prior_tr = lab.prior_table(P, np.ones(len(P['days']), bool))
    cQ, cE = cum_tables(Pv)
    # training windows: production build_windows logic for onset (spacing among onset picks only), restricted to days < CUT_DAY
    g = lab._glist(on); picks = []; last = -10 ** 9
    for x in g:
        if x // 288 >= CUT_DAY:
            break
        if DENSE:
            if x % 3 == 0:
                picks.append((int(x // 288), int(x % 288)))
        elif x - last >= lab.SPACING:
            picks.append((int(x // 288), int(x % 288))); last = x
    if picks:
        R = onset_rows(P, Pv, [(a, b, lab.ONSET) for a, b in picks], prior_tr)
        X, y, e, meta, call = lab.to_matrix(R, lab.ONSET, None, None); assert list(call) == cols
        Xk = blend_rows(X, R, meta, Pv, prior_tr, cQ, cE, K)
        tr_X.append(Xk[e]); tr_y.append(y[e].astype(int))
    g_on, g_og = lab._glist(on), lab._glist(og); s = set()
    for d0 in range(D):
        a, _ = lab.greedy(g_on, g_og, 'interleaved', start_day=d0); s.update(a)
    wins = sorted(w for w in s if w[0] >= max(CUT_DAY, EVAL_FROM))
    if wins:
        R = onset_rows(P, Pv, [(a, b, lab.ONSET) for a, b in wins], prior_tr)
        X, y, e, meta, call = lab.to_matrix(R, lab.ONSET, None, None)
        Xk = blend_rows(X, R, meta, Pv, prior_tr, cQ, cE, K)
        d_, t_ = R['d'], R['t']
        rc = types.SimpleNamespace(truth=np.stack([Pv['Q'][d_, t_ + h] for h in range(1, F + 1)], 1),
                                   ev=np.stack([Pv['E'][d_, t_ + h] for h in range(1, F + 1)], 1), qlast=R['qlast'],
                                   win=pd.DataFrame(dict(d=d_, t=t_, cond=lab.ONSET, fold=0)), name='march_on16', panel=panel,
                                   cond=lab.ONSET, L=R['L'])
        ev.append((rc, X, Xk, meta, R['W'], R['L']))
    lab.log(panel, 'march train windows', len(picks), 'eval windows', len(wins))
    lab.drop_panel(panel); gc.collect()

Xm = np.concatenate(tr_X); ym = np.concatenate(tr_y)
lab.log('March training rows', len(Xm), 'pos rate', round(float(ym.mean()), 4))
if NORAMP:
    catA = None
else:
    catA = CatBoostClassifier(); catA.load_model('results/splitprior/cat_onset.cbm')
Xt, yt = [], []
for p in lab.PANELS:
    rc = lab.rows(p, 'dense1500', lab.ONSET); r = np.where(rc.e)[0]
    Xt.append(np.asarray(rc.X[r])); yt.append(rc.y[r].astype(int))
Xt = np.concatenate(Xt); yt = np.concatenate(yt)
if NORAMP:
    Xt = np.ascontiguousarray(Xt[:, KEEPC]); Xm = np.ascontiguousarray(Xm[:, KEEPC])
    catA = CatBoostClassifier(iterations=600, depth=8, learning_rate=0.08, thread_count=lab.THREADS, verbose=0, random_seed=0)
    catA.fit(Xt, yt); lab.log('A (no ramp) trained')
models = {}
for w in WS:
    X = np.concatenate([Xt, Xm]); y = np.concatenate([yt, ym]); sw = np.r_[np.ones(len(Xt)), np.full(len(Xm), float(w))]
    m = CatBoostClassifier(iterations=600, depth=8, learning_rate=0.08, thread_count=lab.THREADS, verbose=0, random_seed=0)
    t0 = time.time(); m.fit(X, y, sample_weight=sw); models[f'B{w}'] = m
    lab.log(f'B{w:g} trained {time.time() - t0:.0f}s'); del X, y; gc.collect()

res = []
for rc, X, Xk, meta, W, L in ev:
    if NORAMP:
        X, Xk = X[:, KEEPC], Xk[:, KEEPC]
    bag = []
    for name, mdl, XX in [('A', catA, X), ('A_s4', catA, Xk)] + [(n, m, Xk) for n, m in models.items()]:
        P6 = np.zeros((W, F, L), np.float32); P6[meta[:, 0], F - 1, meta[:, 1]] = mdl.predict_proba(XX)[:, 1]
        res.append(lab.score_probs(rc, P6, arm=name, set_name='march_on16'))
        if name.startswith('B'):
            bag.append(P6)
    if len(bag) > 1:
        res.append(lab.score_probs(rc, np.mean(bag, 0), arm='Bbag', set_name='march_on16'))
Rr = pd.concat(res, ignore_index=True); Rr.to_pickle(f'{OUT}/march_onset_marchtrain_{TAG}.pkl')
S = pd.DataFrame([dict(arm=a, iou=lab.aggregate(Rr[Rr.arm == a]), n=int((Rr.arm == a).sum())) for a in dict.fromkeys(Rr.arm)])
print('\n== pseudo-March onset, days >= 15 =='); print(S.round(4).to_string(index=False))
A0 = Rr[Rr.arm == 'A_s4']
for a in dict.fromkeys(Rr.arm):
    if a in ('A_s4',):
        continue
    bt = lab.paired_bootstrap(A0, Rr[Rr.arm == a])
    pa = A0.groupby('panel').iou.mean(); pb = Rr[Rr.arm == a].groupby('panel').iou.mean()
    print(f'{a} - A_s4: {bt["diff"]:+.4f} [{bt["lo"]:+.4f}, {bt["hi"]:+.4f}] panels+ {int((pb - pa > 1e-9).sum())}/{int((abs(pb - pa) > 1e-9).sum())}')
lab.log('DONE')
