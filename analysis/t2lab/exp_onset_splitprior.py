"""Onset under the new scenario seed: does a SAME-SCENARIO location prior help the production onset model?

Model: CatBoost onset as production v13 (600 it, depth 8, lr 0.08), trained on the train split only (lab dense1500 onset rows,
all-train prior in the prior columns, as production).
Windows: onset pseudo-windows of the masked March month (production onset rule on visible data, faithful ongoing candidates for
the interleaved greedy, official-like greedy restarted at every day, blank-touching windows dropped; as exp_onset_insplit_signal.py).
Features: production window_features on the masked-layer history + masked slot-T value; the three prior columns are substituted:
  base   all-train prior (production)
  b<k>   per cell (1-a) train + a split, split = P(queued | link, slot, class) over the March days < d (visible cells only),
         a = n/(n+k) with n = number of visible observations of that cell in those days (0 -> train prior)
  s<k>   same with the split counts summed over slots +-2 (n counted in days: summed visible / 5)
Labels: visible masked-layer cells of T+5..T+30.  Day bins: d<5 = days 1-5 (where the official windows sit, public-like),
d>=15 = April-like (>= 15 same-scenario days of history; April itself would have all of March).
Decision support only (March labels): it may steer the private file / motivate a public probe."""
import gc, json, os, sys, time, types, warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lab
from exp_firstcross_prod_snippet import fill_time36

H, F = lab.H, lab.F
DROPC = [c for c in os.environ.get('DROP', '').split(',') if c]
MODE = sys.argv[1] if len(sys.argv) > 1 else 'lt_d'          # split-prior days: lt_d = days < d; gap10 = days < d-10; first10 = days < min(d, 10); recent10 = d-10 <= day < d; exp<tau> = all days < d, recency-weighted
OUT = 'results/splitprior' + ('' if MODE == 'lt_d' else f'_{MODE}') + ('_drop_' + '_'.join(DROPC) if DROPC else '') + ('_noramp' if os.environ.get('NORAMP') == '1' else '')
os.makedirs(OUT, exist_ok=True)
KS = (1, 2, 4, 8, 16, 32)
PRIOR3 = ('prior', 'prior_now', 'prior_alldays')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)

# ---------------------------------------------------------------- 1. production-like CatBoost onset model (train split only)
cols = lab.all_cols()
KEEPI = [i for i, c in enumerate(cols) if c not in DROPC and not (os.environ.get('NORAMP') == '1' and c.startswith('ramp_'))]
mpath = 'results/splitprior/cat_onset' + ('_drop_' + '_'.join(DROPC) if DROPC else '') + ('_noramp' if os.environ.get('NORAMP') == '1' else '') + '.cbm'
cat = CatBoostClassifier()
if os.path.exists(mpath):
    cat.load_model(mpath)
else:
    Xs, ys = [], []
    for p in lab.PANELS:
        rc = lab.rows(p, 'dense1500', lab.ONSET)
        assert list(rc.cols) == cols, p
        r = np.where(rc.e)[0]
        Xs.append(np.asarray(rc.X[r])); ys.append(rc.y[r].astype(int))
    X = np.concatenate(Xs); y = np.concatenate(ys); del Xs
    sub = np.random.default_rng(0).choice(len(X), min(len(X), 1_200_000), replace=False)
    cat = CatBoostClassifier(iterations=600, depth=8, learning_rate=0.08, thread_count=lab.THREADS, verbose=0, random_seed=0)
    t0 = time.time(); cat.fit(X[sub][:, KEEPI], y[sub])
    lab.log('catboost onset trained on', len(sub), 'rows', f'{time.time() - t0:.0f}s', 'pos rate', round(float(y.mean()), 4))
    cat.save_model(mpath); del X, y; gc.collect()
imp = pd.Series(cat.get_feature_importance(), index=[cols[i] for i in KEEPI]).sort_values(ascending=False)
print('top onset features:', imp.head(12).round(2).to_dict())
JP = [cols.index(c) for c in PRIOR3]


# ---------------------------------------------------------------- 2. March pseudo-onset windows (copied from exp_onset_insplit_signal.py)
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


def cum_tables(Pv, smooth, tau=None):
    """cumQ, cumE [3, D+1, 288, L]: sums over the days < d of class c (0 weekday, 1 weekend, 2 all); smooth -> summed over slots +-2.
    tau -> exponentially recency-weighted sums: S[d] = sum_{j<d} rho^(d-1-j) x_j, rho = exp(-1/tau)."""
    Q = Pv['Q'].astype(np.float32); E = Pv['E'].astype(np.float32)
    if smooth:
        def box(a):
            c = np.cumsum(np.pad(a, ((0, 0), (3, 2), (0, 0))), 1)
            return c[:, 5:] - c[:, :-5]
        Q, E = box(Q), box(E)
    D = Q.shape[0]; wk = Pv['dow'] >= 5
    cQ = np.zeros((3, D + 1) + Q.shape[1:], np.float32); cE = cQ.copy()
    for c, m in enumerate((~wk, wk, np.ones(D, bool))):
        if tau is None:
            cQ[c, 1:] = np.cumsum(Q * m[:, None, None], 0); cE[c, 1:] = np.cumsum(E * m[:, None, None], 0)
        else:
            rho = np.float32(np.exp(-1.0 / tau))
            for dd in range(D):
                cQ[c, dd + 1] = rho * cQ[c, dd] + Q[dd] * m[dd]; cE[c, dd + 1] = rho * cE[c, dd] + E[dd] * m[dd]
    return cQ, cE


rows_out = []; info = []
for panel in lab.PANELS:
    t1 = time.time()
    P, Pv = val_panel(panel)
    on, og = select(Pv)
    D = len(Pv['days'])
    g_on, g_og = lab._glist(on), lab._glist(og); s = set()
    for d0 in range(D):
        a, _ = lab.greedy(g_on, g_og, 'interleaved', start_day=d0); s.update(a)
    wins = sorted(s)
    if not wins:
        continue
    prior_tr = lab.prior_table(P, np.ones(len(P['days']), bool))
    R = onset_rows(P, Pv, [(a, b, lab.ONSET) for a, b in wins], prior_tr)
    X, y, e, meta, call = lab.to_matrix(R, lab.ONSET, None, None)
    assert list(call) == cols
    wi, li = meta[:, 0], meta[:, 1]
    d, t = R['d'], R['t']; cls = (Pv['dow'][d] >= 5).astype(int); s6 = np.minimum(t + 6, 287)
    TAU = float(MODE[3:]) if MODE.startswith('exp') else None
    lo, hi = {'lt_d': (0 * d, d), 'gap10': (0 * d, np.maximum(d - 10, 0)), 'first10': (0 * d, np.minimum(d, 10)),
              'recent10': (np.maximum(d - 10, 0), d)}.get(MODE, (None, d))
    c_r, d_r, s6_r, t_r = cls[wi], hi[wi], s6[wi], t[wi]
    lo_r = None if lo is None else lo[wi]
    trv = [prior_tr[c_r, s6_r, li], prior_tr[c_r, t_r, li], prior_tr[2, s6_r, li]]
    tabs = {}
    for sm in (False, True):
        cQ, cE = cum_tables(Pv, sm, TAU); div = 5.0 if sm else 1.0
        sp, n = [], []
        for cc, ss in ((c_r, s6_r), (c_r, t_r), (np.full_like(c_r, 2), s6_r)):
            q = cQ[cc, d_r, ss, li]; ee = cE[cc, d_r, ss, li]
            if lo_r is not None:
                q = q - cQ[cc, lo_r, ss, li]; ee = ee - cE[cc, lo_r, ss, li]
            sp.append(np.where(ee > 0, q / np.maximum(ee, 1e-9), 0.0)); n.append(ee / div)
        tabs[sm] = (sp, n)
        del cQ, cE
    win = pd.DataFrame(dict(d=d, t=t, cond=lab.ONSET, fold=0))
    rc = types.SimpleNamespace(truth=np.stack([Pv['Q'][d, t + h] for h in range(1, F + 1)], 1),
                               ev=np.stack([Pv['E'][d, t + h] for h in range(1, F + 1)], 1),
                               qlast=R['qlast'], win=win, name='march_on', panel=panel, cond=lab.ONSET, L=R['L'])
    variants = {'base': None}
    for k in KS:
        variants[f'b{k}'] = (False, k); variants[f's{k}'] = (True, k)
    variants['b_only'] = (False, 1e-6); variants['s_only'] = (True, 1e-6)
    for name, v in variants.items():
        Xv = X.copy()
        if v is not None:
            sm, k = v; sp, n = tabs[sm]
            for j, jp in enumerate(JP):
                a_ = n[j] / (n[j] + k)
                Xv[:, jp] = (1 - a_) * trv[j] + a_ * sp[j]
        p = cat.predict_proba(Xv[:, KEEPI])[:, 1]
        P6 = np.zeros((R['W'], F, R['L']), np.float32); P6[wi, F - 1, li] = p
        df = lab.score_probs(rc, P6, arm=name, set_name='march_on')
        rows_out.append(df)
    info.append(dict(panel=panel, windows=len(wins), d_lt5=int((d < 5).sum()), d_ge15=int((d >= 15).sum())))
    lab.log(panel, info[-1], f'{time.time() - t1:.0f}s')
    del X, R; lab.drop_panel(panel); gc.collect()

Rr = pd.concat(rows_out, ignore_index=True)
Rr.to_pickle(f'{OUT}/march_onset_variants.pkl')
print(pd.DataFrame(info).to_string(index=False))
Rr['dbin'] = pd.cut(Rr.d, [-1, 4, 14, 40], labels=['d1-5', 'd6-15', 'd16+'])
rows = []
for name in dict.fromkeys(Rr.arm):
    r = dict(arm=name)
    for b in ('all', 'd1-5', 'd6-15', 'd16+'):
        g = Rr[(Rr.arm == name) & ((Rr.dbin == b) if b != 'all' else True)]
        r[b] = lab.aggregate(g); r[f'n_{b}'] = len(g)
    rows.append(r)
S = pd.DataFrame(rows); print('\n== March pseudo-onset IoU (window -> panel -> family -> mean) =='); print(S.round(4).to_string(index=False))
S.to_csv(f'{OUT}/summary.csv', index=False)
print('\n== paired bootstrap vs base ==')
out = []
A0 = Rr[Rr.arm == 'base']
for name in dict.fromkeys(Rr.arm):
    if name == 'base':
        continue
    B0 = Rr[Rr.arm == name]
    r = dict(arm=name)
    for b in ('all', 'd1-5', 'd16+'):
        A = A0 if b == 'all' else A0[A0.dbin == b]; B = B0 if b == 'all' else B0[B0.dbin == b]
        bt = lab.paired_bootstrap(A.drop(columns='dbin'), B.drop(columns='dbin'))
        r[f'{b}_diff'] = bt['diff']; r[f'{b}_lo'] = bt['lo']; r[f'{b}_hi'] = bt['hi']
    pa = A0.groupby('panel').iou.mean(); pb = B0.groupby('panel').iou.mean(); r['panels_pos'] = f'{int((pb - pa > 1e-9).sum())}/{int((abs(pb - pa) > 1e-9).sum())}'
    out.append(r)
Dl = pd.DataFrame(out); print(Dl.round(4).to_string(index=False)); Dl.to_csv(f'{OUT}/deltas.csv', index=False)
best = Dl.sort_values('all_diff').arm.iloc[-1]
pp = pd.DataFrame({'base': A0.groupby('panel').iou.mean(), best: Rr[Rr.arm == best].groupby('panel').iou.mean(),
                   'n': A0.groupby('panel').size()})
pp['delta'] = pp[best] - pp.base
print(f'\nper panel: {best} vs base'); print(pp.round(4).to_string())
lab.log('DONE')
