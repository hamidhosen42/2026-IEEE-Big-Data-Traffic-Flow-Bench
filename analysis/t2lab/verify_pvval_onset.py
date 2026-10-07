"""verify_pvval_onset.py -- adversarial verifier of track pvval, candidate onset_april_pv_bag_as_built.

Independent re-run of the key pseudo-March onset comparison (windows of days >= CUT, March days < CUT = same-scenario data):
  sp4 (production CatBoost, prior blended with the March days < d)  vs  bag of 3 CatBoost (train rows + March rows x 10/20/40).
Independence from the pvon/pvval code:
  * rows / features / prior blend / March-window rule = the KERNEL's own functions (kaggle_jobs/t2_onset_pv.py compiled with ast,
    its main code never runs) on the lab's exported cubes -> also a parity check of the recipe as built;
  * train windows = the kernel's build_windows + og cap 3000 / rng(0) (the lab used dense1500 -> other slot-T masks);
  * evaluation windows = my own re-implementation of the official-like selector (multi-start interleaved greedy, 5 + 5 quota,
    72 slots, t36-filled persistence rule) on the masked March layer, compared with the pvon window list;
  * CatBoost seeds 0 / 1 / 2 (claim: seed 0 only).
Extra arms: bagX = weights 25/50/100 (about the kernel's 2.5x total March weight), w5 (bagL = 5/10/20 check, seed 0),
masked-train control with my own mask emulation (seed 11, one seed): A_mask / B_mask bag.
Usage: verify_pvval_onset.py [SEEDS] [CUT]      (VPV_SMOKE=1: 2 panels, 30 iterations)"""
import ast, gc, glob, json, os, re, sys, time, warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lab
from exp_firstcross_prod_snippet import fill_time36

warnings.filterwarnings('ignore')
SEEDS = [int(s) for s in sys.argv[1].split(',')] if len(sys.argv) > 1 else [0, 1, 2]
CUT = int(sys.argv[2]) if len(sys.argv) > 2 else 15
SMOKE = os.environ.get('VPV_SMOKE') == '1'
IT = 30 if SMOKE else 600
PANELS = ['D7_I10_W', 'D12_I5_S'] if SMOKE else list(lab.PANELS)
OUT = f'{lab.ROOT}/analysis/t2lab/results/verify_pvval'
os.makedirs(OUT, exist_ok=True)
TAG = f'cut{CUT}_s{"-".join(map(str, SEEDS))}' + ('_smoke' if SMOKE else '')
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


# ------------------------------------------------------------------ kernel functions (compiled from the kernel file itself)
KFILE = f'{lab.ROOT}/kaggle_jobs/t2_onset_pv.py'
SRC = open(KFILE).read()
TREE = ast.parse(SRC)
FDEFS = [n for n in TREE.body if isinstance(n, ast.FunctionDef)]
K = {'np': np, 'pd': pd, 'glob': glob, 'os': os, 'gc': gc, 'time': time, 'json': json, 'T0': T0}
_m = re.search(r'^H, F, SPACING = (\d+), (\d+), (\d+)$', SRC, re.M)
K.update(H=int(_m.group(1)), F=int(_m.group(2)), SPACING=int(_m.group(3)))
exec(compile(ast.Module(body=FDEFS, type_ignores=[]), KFILE, 'exec'), K)
H, F, SPACING = K['H'], K['F'], K['SPACING']
assert (H, F, SPACING) == (12, 6, 72)
assert 'K_BLEND = 4' in SRC and 'W_MARCH = (10, 20, 40)' in SRC and "MAX_OG = 300 if SMOKE else 3000" in SRC
log('kernel functions:', ' '.join(f.name for f in FDEFS))


# ------------------------------------------------------------------ my own selector (official-like, masked layer)
def my_candidates(V, Em, cut):
    D = V.shape[0]
    with np.errstate(invalid='ignore'):
        E = np.isfinite(V); Q = E & (V <= cut[None, None, :])
        VF = fill_time36(V); MF = np.isfinite(VF); QF = MF & (VF <= cut[None, None, :])
    blank = (Em.sum(2) >= 5) & (E.sum(2) == 0)
    on = np.zeros((D, 288), bool); og = on.copy()
    for d in range(D):
        for t in range(H, 288 - F):
            if blank[d, t - H:t + F + 1].any() or Em[d, t - H:t].mean() < 0.7:
                continue
            qh = Q[d, t - H:t]
            if (qh.sum(0) >= 2).any():                       # established: >= 2 queued obs on one link (implies >= 2 in total)
                Tf = QF[d, t + 1:t + 1 + F]; Mf = MF[d, t + 1:t + 1 + F]
                if Tf.any():
                    S = np.broadcast_to(QF[d, t], Tf.shape)
                    u = ((Tf | S) & Mf).sum(); i = (Tf & S & Mf).sum()
                    og[d, t] = (i / u if u else 1.0) <= 0.9
            else:
                on[d, t] = Q[d, t + 1:t + 1 + F].any()
    return on, og, Q, E


def my_multistart(on, og, D, quota=5, spacing=72):
    g_on = np.flatnonzero(on.ravel()); g_og = np.flatnonzero(og.ravel())
    allg = np.concatenate([g_on, g_og]); isog = np.r_[np.zeros(len(g_on), bool), np.ones(len(g_og), bool)]
    o = np.argsort(allg, kind='stable'); allg = allg[o]; isog = isog[o]
    picks = set()
    for d0 in range(D):
        n_on = n_og = 0; last = -10 ** 9
        for g, isg in zip(allg.tolist(), isog.tolist()):
            if g < d0 * 288 or g - last < spacing:
                continue
            if n_on >= quota and n_og >= quota:
                break
            if isg:
                if n_og < quota:
                    n_og += 1; last = g
            elif n_on < quota:
                n_on += 1; last = g; picks.add((g // 288, g % 288))
    return sorted(picks)


def scen_cums(V, cut, dow):
    """the kernel's scenario_cums math on one masked month (cum[c, j] = days < j of class c, slots +-2)."""
    E = np.isfinite(V); Q = (E & (np.nan_to_num(V, nan=1e9) <= cut[None, None, :])).astype(np.float32); E = E.astype(np.float32)

    def box(a):
        c = np.cumsum(np.pad(a, ((0, 0), (3, 2), (0, 0))), 1)
        return c[:, 5:] - c[:, :-5]
    Q, E = box(Q), box(E)
    wk = dow >= 5; D = V.shape[0]
    cQ = np.zeros((3, D + 1) + Q.shape[1:], np.float32); cE = cQ.copy()
    for c, msk in enumerate((~wk, wk, np.ones(D, bool))):
        cQ[c, 1:] = np.cumsum(Q * msk[:, None, None], 0); cE[c, 1:] = np.cumsum(E * msk[:, None, None], 0)
    return cQ, cE


def onset_rows_k(P, M, wins, prior, Qlab, Elab, cols):
    """exactly the kernel's March-row construction (t2_onset_pv.py lines 396-404) for windows `wins` on the cube M."""
    d = np.array([w[0] for w in wins]); t = np.array([w[1] for w in wins])
    idx = t[:, None] + np.arange(-H, 0)[None]
    with np.errstate(all='ignore'):
        Fe, qlast = K['window_features'](P, M['V'][d[:, None], idx], M['Fq'][d[:, None], idx], t, M['dow'][d], prior, M['V'][d, t])
    W_, L_ = qlast.shape
    feats = dict(Fe); feats['h'] = np.full((W_, L_), 6, np.float32)
    feats['prior'], feats['prior_now'], feats['prior_alldays'] = K['prior_feats'](prior, M['dow'][d], t, 6)
    R = dict(d=d, t=t, cond=np.array(['queue_onset'] * W_), W=W_, L=L_, qlast=qlast,
             rows=[(6, feats, Qlab[d, np.minimum(t + 6, 287)], Elab[d, np.minimum(t + 6, 287)])])
    X, y, e, meta, _ = K['to_matrix'](R, 'queue_onset', cols, None)
    return R, X, y.astype(bool), e.astype(bool), meta


# ------------------------------------------------------------------ data
pv_ref = pd.read_pickle(f'{lab.ROOT}/analysis/t2lab/results/pvon/step2_cut15_s1_S.pkl') if CUT == 15 else None
cols = None
TR_X, TR_y, MA_X, MA_y, MK_X, MK_y, EV = [], [], [], [], [], [], []
info = []
for pi, panel in enumerate(PANELS):
    t1 = time.time()
    P = lab.load_panel(panel)
    # ---- raw train rows: the kernel recipe (build_windows dense, og cap 3000 with rng(0), make_rows rng(len(wins)))
    tw = K['build_windows'](P, global_spacing=False, dense_ongoing=True)
    og_ = [w for w in tw if w[2] == 'queue_ongoing']; on_ = [w for w in tw if w[2] == 'queue_onset']
    if len(og_) > 3000:
        r0 = np.random.default_rng(0)
        og_ = [og_[i] for i in sorted(r0.choice(len(og_), 3000, replace=False))]
    prior_all = K['prior_table'](P, np.ones(len(P['days']), bool))
    P['prior_all'] = prior_all
    with np.errstate(all='ignore'):
        R = K['make_rows'](P, on_ + og_, prior_all)
    X, y, e, meta, c_ = K['to_matrix'](R, 'queue_onset', None, None)
    if cols is None:
        cols = c_; K['JP'] = [cols.index(c) for c in ('prior', 'prior_now', 'prior_alldays')]
        log('n cols', len(cols), 'ramp cols', sum(c.startswith('ramp') for c in cols))
    assert c_ == cols
    TR_X.append(X[e]); TR_y.append(y[e].astype(int)); n_raw = int(e.sum())
    del R, X; gc.collect()
    # ---- March masked month (lab export of validation_Vm / Fm / Em)
    z = np.load(P['npz'], allow_pickle=False)
    days = np.array([str(x) for x in z['validation_days']]); o = np.argsort(days)
    M = dict(days=list(days[o]), V=z['validation_Vm'][o].astype(np.float32), Fq=z['validation_Fm'][o].astype(np.float32),
             Em=z['validation_Em'][o].astype(bool))
    M['dow'] = pd.to_datetime(pd.Series(M['days'])).dt.dayofweek.to_numpy()
    D = len(M['days'])
    cQ, cE = scen_cums(M['V'], P['cut'], M['dow'])
    # ---- same-scenario training rows: the kernel's march_onset_windows, restricted to days < CUT
    picks, Qm, Emv = K['march_onset_windows'](M, P['cut'])
    ptr = [w for w in picks if w[0] < CUT]
    nm = 0
    if ptr:
        R, X, y, e, meta = onset_rows_k(P, M, ptr, prior_all, Qm, Emv, cols)
        jd = R['d']; cls = (M['dow'][R['d']] >= 5).astype(int)
        Xk, _ = K['blend_prior'](X, meta, P, jd, cls, R['t'], cQ, cE, 4)
        MA_X.append(Xk[e]); MA_y.append(y[e].astype(int)); nm = int(e.sum())
    # ---- evaluation windows (my selector), days >= CUT
    on_c, og_c, Qv, Ev_ = my_candidates(M['V'], M['Em'], P['cut'])
    wins = [w for w in my_multistart(on_c, og_c, D) if w[0] >= CUT]
    same = None
    if pv_ref is not None:
        ref = pv_ref[(pv_ref.arm == 'A_s4') & (pv_ref.panel == panel)]
        same = sorted(zip(ref.d.astype(int), ref.t.astype(int))) == sorted(wins)
    R, X, y, e, meta = onset_rows_k(P, M, wins, prior_all, Qv, Ev_, cols)
    jd = R['d']; cls = (M['dow'][R['d']] >= 5).astype(int)
    Xk, a_mean = K['blend_prior'](X, meta, P, jd, cls, R['t'], cQ, cE, 4)
    d_, t_ = R['d'], R['t']
    truth = np.stack([Qv[d_, t_ + h] for h in range(1, F + 1)], 1); ev = np.stack([Ev_[d_, t_ + h] for h in range(1, F + 1)], 1)
    EV.append(dict(panel=panel, d=d_, t=t_, W=R['W'], L=R['L'], meta=meta, X=X, Xk=Xk, truth=truth, ev=ev))
    # ---- masked-train control: my own emulation (per-day share 0.5 / 0.7 / 0.8, i.i.d. cells), windows re-selected with the
    #      kernel's onset rule on the masked train cube, masked features, visible labels
    rng = np.random.default_rng(11 * 1000 + pi)
    s = rng.choice(np.array([0.5, 0.7, 0.8], np.float32), size=len(P['days']))
    vis = P['E'] & (rng.random(P['E'].shape, dtype=np.float32) < s[:, None, None])
    Mk = dict(V=np.where(vis, P['V'], np.nan).astype(np.float32), Fq=np.where(vis, P['Fq'], np.nan).astype(np.float32), Em=P['E'],
              dow=P['dow'])
    pk, Qk, Ek = K['march_onset_windows'](Mk, P['cut'])
    R, X, y, e, meta = onset_rows_k(P, Mk, pk, prior_all, Qk, Ek, cols)
    MK_X.append(X[e]); MK_y.append(y[e].astype(int))
    info.append(dict(panel=panel, raw_on_w=len(on_), raw_rows=n_raw, march_w=len(ptr), march_rows=nm, eval_w=len(wins),
                     same_eval_as_pvon=same, mask_w=len(pk), mask_rows=int(e.sum()), a_mean=round(a_mean, 3)))
    log(info[-1], f'{time.time() - t1:.0f}s')
    del Mk, vis, R, X, z; lab.drop_panel(panel); gc.collect()

print(pd.DataFrame(info).to_string(index=False), flush=True)
Xo = np.concatenate(TR_X); yo = np.concatenate(TR_y); Xm = np.concatenate(MA_X); ym = np.concatenate(MA_y)
Xk_ = np.concatenate(MK_X); yk_ = np.concatenate(MK_y)
del TR_X, MA_X, MK_X; gc.collect()
sub = np.random.default_rng(0).choice(len(Xo), min(len(Xo), 1_200_000), replace=False)          # the kernel's subsample
Xo, yo = Xo[sub], yo[sub]
log('raw train rows', len(Xo), 'pos', round(float(yo.mean()), 4), '| March rows', len(Xm), 'pos', round(float(ym.mean()), 4),
    '| masked train rows', len(Xk_), 'pos', round(float(yk_.mean()), 4))


def fit(X, y, w, seed):
    t0 = time.time()
    m = CatBoostClassifier(iterations=IT, depth=8, learning_rate=0.08, thread_count=lab.THREADS, verbose=0, random_seed=seed)
    m.fit(X, y, sample_weight=w)
    return m, time.time() - t0


def pred(m, key):
    return [m.predict_proba(ev[key])[:, 1] for ev in EV]


PR = {}
XB = np.concatenate([Xo, Xm]); YB = np.concatenate([yo, ym])
for seed in SEEDS:
    m, dt = fit(Xo, yo, None, seed); PR[f'base_s{seed}'] = pred(m, 'X'); PR[f'sp4_s{seed}'] = pred(m, 'Xk'); log(f'base seed {seed} {dt:.0f}s')
    ws = (10, 20, 40) + ((5,) if seed == SEEDS[0] else ()) + ((25, 50, 100) if seed == SEEDS[min(1, len(SEEDS) - 1)] else ())
    for w in ws:
        m, dt = fit(XB, YB, np.r_[np.ones(len(Xo)), np.full(len(Xm), float(w))], seed)
        PR[f'w{w}_s{seed}'] = pred(m, 'Xk'); log(f'w{w} seed {seed} {dt:.0f}s')
    PR[f'bag_s{seed}'] = [np.mean(z_, 0) for z_ in zip(*[PR[f'w{w}_s{seed}'] for w in (10, 20, 40)])]
    if f'w5_s{seed}' in PR:
        PR[f'bagL_s{seed}'] = [np.mean(z_, 0) for z_ in zip(*[PR[f'w{w}_s{seed}'] for w in (5, 10, 20)])]
    if f'w25_s{seed}' in PR:
        PR[f'bagX_s{seed}'] = [np.mean(z_, 0) for z_ in zip(*[PR[f'w{w}_s{seed}'] for w in (25, 50, 100)])]
# masked-train control (one CatBoost seed = SEEDS[1] or SEEDS[0]); masked train rows carry the raw-train total weight
ms = SEEDS[min(1, len(SEEDS) - 1)]
wk = len(Xo) / len(Xk_)
m, dt = fit(Xk_, yk_, np.full(len(Xk_), wk), ms); PR[f'Amask_s{ms}'] = pred(m, 'Xk'); log(f'Amask {dt:.0f}s')
XB2 = np.concatenate([Xk_, Xm]); YB2 = np.concatenate([yk_, ym])
for w in (10, 20, 40):
    m, dt = fit(XB2, YB2, np.r_[np.full(len(Xk_), wk), np.full(len(Xm), float(w))], ms); PR[f'Bmask_w{w}_s{ms}'] = pred(m, 'Xk')
    log(f'Bmask w{w} {dt:.0f}s')
PR[f'Bmask_bag_s{ms}'] = [np.mean(z_, 0) for z_ in zip(*[PR[f'Bmask_w{w}_s{ms}'] for w in (10, 20, 40)])]

# ------------------------------------------------------------------ scoring (kernel decode / iou)
rows = []
for arm, plist in PR.items():
    for ev, p in zip(EV, plist):
        P6 = np.zeros((ev['W'], F, ev['L']), np.float32); P6[ev['meta'][:, 0], ev['meta'][:, 2] - 1, ev['meta'][:, 1]] = p
        for i in range(ev['W']):
            pr_ = K['decode'](P6[i].ravel()).reshape(F, -1)
            rows.append(dict(arm=arm, panel=ev['panel'], family=ev['panel'].rsplit('_', 1)[0], cond='queue_onset', set='march_on',
                             d=int(ev['d'][i]), t=int(ev['t'][i]), iou=float(K['iou'](ev['truth'][i], pr_, ev['ev'][i])),
                             npred=int(pr_.sum())))
S = pd.DataFrame(rows)
KEY = ['panel', 'family', 'cond', 'set', 'd', 't']


def seedmean(prefix, name):
    g = S[S.arm.isin([f'{prefix}_s{s}' for s in SEEDS])].groupby(KEY, as_index=False).iou.mean()
    g['arm'] = name
    return g


S = pd.concat([S, seedmean('base', 'base_mean'), seedmean('sp4', 'sp4_mean'), seedmean('bag', 'bag_mean')], ignore_index=True)
S.to_pickle(f'{OUT}/onset_{TAG}.pkl')
print('\n== pseudo-March onset, days >=', CUT, ', n =', int((S.arm == f'sp4_s{SEEDS[0]}').sum()), 'windows ==')
print('  '.join(f'{a} {lab.aggregate(S[S.arm == a]):.4f}' for a in dict.fromkeys(S.arm)))


def cmp(a, b, drop=None):
    A = S[S.arm == a]; B = S[S.arm == b]
    if drop:
        A = A[~A.panel.isin(drop)]; B = B[~B.panel.isin(drop)]
    if not len(A):
        return dict(B=b, A=a, n=0, drop=','.join(drop or []))
    bt = lab.paired_bootstrap(A, B)
    dl = B.groupby('panel').iou.mean() - A.groupby('panel').iou.mean()
    nwin = int((B.set_index(KEY).iou - A.set_index(KEY).iou).abs().gt(1e-12).sum())
    return dict(B=b, A=a, n=bt['n_windows'], iou_A=lab.aggregate(A), iou_B=lab.aggregate(B), diff=bt['diff'], lo=bt['lo'], hi=bt['hi'],
                pos=int((dl > 1e-9).sum()), neg=int((dl < -1e-9).sum()), win_changed=nwin, drop=','.join(drop or []),
                best_panel=dl.idxmax(), best_gain=float(dl.max()))


out = []
for s_ in SEEDS:
    out += [cmp(f'base_s{s_}', f'sp4_s{s_}'), cmp(f'sp4_s{s_}', f'bag_s{s_}')]
out += [cmp('base_mean', 'sp4_mean'), cmp('sp4_mean', 'bag_mean')]
for a in PR:
    if a.startswith('w') and a.split('_')[1] in [f's{s}' for s in SEEDS]:
        out.append(cmp(f'sp4_{a.split("_")[1]}', a))
for a in [x for x in PR if x.startswith('bagL') or x.startswith('bagX')]:
    sd = a.split('_')[1]
    out += [cmp(f'sp4_{sd}', a), cmp(f'bag_{sd}', a)]
out += [cmp(f'sp4_s{ms}', f'Amask_s{ms}'), cmp(f'Amask_s{ms}', f'Bmask_bag_s{ms}'), cmp(f'bag_s{ms}', f'Bmask_bag_s{ms}')]
# drop the best panel (by bag - sp4 gain) and the best two
c0 = cmp('sp4_mean', 'bag_mean'); b1 = c0['best_panel']
dl = S[S.arm == 'bag_mean'].groupby('panel').iou.mean() - S[S.arm == 'sp4_mean'].groupby('panel').iou.mean()
top2 = list(dl.sort_values(ascending=False).index[:2])
out += [cmp('sp4_mean', 'bag_mean', [b1]), cmp('sp4_mean', 'bag_mean', top2)]
C = pd.DataFrame(out)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(C.round(4).to_string(index=False))
C.to_csv(f'{OUT}/onset_{TAG}_cmp.csv', index=False)
print('\nper-panel bag_mean - sp4_mean:', dl.round(4).to_dict())
# per-window parity with the pvon results (seed 0 only; the lab trained on dense1500 rows -> not bit-identical)
if pv_ref is not None and f'sp4_s{SEEDS[0]}' in set(S.arm):
    for mine, theirs in ((f'sp4_s{SEEDS[0]}', 'A_s4'), (f'bag_s{SEEDS[0]}', 'B_bag'), (f'base_s{SEEDS[0]}', 'A')):
        a = S[S.arm == mine].set_index(['panel', 'd', 't']).iou; b = pv_ref[pv_ref.arm == theirs].set_index(['panel', 'd', 't']).iou
        j = a.to_frame('m').join(b.to_frame('p'), how='inner')
        log(f'parity {mine} vs pvon {theirs}: n {len(j)}, identical {(j.m - j.p).abs().lt(1e-9).mean():.3f}, mean {j.m.mean():.4f} vs {j.p.mean():.4f}')
log('DONE')
