"""Track `ideas`: onset arms (ideas 1A, 2, 3, 5).  CatBoost onset exactly as production v13 (600 it, depth 8, lr 0.08; the lab
NORAMP=1 column set), trained on the production onset training windows (dense1500 onset rows, e only).

Usage (through the semaphore):  ideas_onset.py <arm,arm,...> [seeds=0,1,2] [ks=4,0,1,2,3]
  k = 4   final model on all 273 train days -> the 214 pseudo-March onset windows (masked histories, visible labels) with the
          all-train prior (base) and the same-scenario blends s4 / s8 (March days < d)          [new-scenario regime]
          (training rows in the production order: rng(0) permutation, as exp_onset_splitprior.py)
  k = 0..3 fold model (rows of the other 3 day blocks, plain order as exp_onset_wrongscen.py) -> held-out official-like onset
          windows of fold k with the fold-safe prior                                             [in-distribution regime]
          fold 0 additionally with the WRONG-scenario prior (fold-safe prior blended with all of March, a = n/(n+4)) = `wrong`
Arms: prior scheme of the TRAINING rows (S = self-inclusive as production, L1 = leave-one-day-out, L7 = leave-week-group-out
with (d // 7) % 4) + extra feature groups + augmentation.  Evaluation rows always carry the plain prior of the allowed days.
Per (arm, k) one file results/ideas/onset/<arm>__k<k>.pkl = dict(df = per-window rows, probs = {(set, panel, seed): p})."""
import gc, os, sys, time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lab, ideas_lib as il

ARGS = dict(a.split('=', 1) for a in sys.argv[2:] if '=' in a)
SEEDS = [int(x) for x in ARGS.get('seeds', '0,1,2').split(',')]
KS = [int(x) for x in ARGS.get('ks', '4,0,1,2,3').split(',')]
DRY = os.environ.get('DRY') == '1'            # plumbing check: no model is trained, nothing meaningful is scored
ODIR = f'{il.OUT}/onset' + ('_dry' if DRY else ''); os.makedirs(ODIR, exist_ok=True)
SITE, TOL, FLOW = il.SITE, il.TOL, il.FLOW
ARMS = {
    'S': dict(prior='S'), 'L1': dict(prior='L1'), 'L7': dict(prior='L7'),
    'S_site': dict(prior='S', extra=SITE), 'S_sitetol': dict(prior='S', extra=SITE + TOL),
    'L1_site': dict(prior='L1', extra=SITE), 'L1_tol': dict(prior='L1', extra=TOL), 'L1_sitetol': dict(prior='L1', extra=SITE + TOL),
    'L7_site': dict(prior='L7', extra=SITE), 'L7_sitetol': dict(prior='L7', extra=SITE + TOL),
    'L1_site2': dict(prior='L1', extra=['site_on', 'prior_day']), 'L1_site2tol': dict(prior='L1', extra=['site_on', 'prior_day'] + TOL),
    'L1_flow': dict(prior='L1', extra=FLOW), 'L1_sitetol_flow': dict(prior='L1', extra=SITE + TOL + FLOW),
    'L7_sitetol_flow': dict(prior='L7', extra=SITE + TOL + FLOW),
    # idea 3: prior-timing jitter (one augmented copy per train window, weight w) / random prior dropout
    'L1_jit05': dict(prior='L1', jitter=0.5), 'L1_jit10': dict(prior='L1', jitter=1.0),
    'L1_sitetol_jit05': dict(prior='L1', extra=SITE + TOL, jitter=0.5), 'L1_sitetol_jit10': dict(prior='L1', extra=SITE + TOL, jitter=1.0),
    'L7_sitetol_jit05': dict(prior='L7', extra=SITE + TOL, jitter=0.5),
    'L1_sitetol_drop30': dict(prior='L1', extra=SITE + TOL, drop=0.3), 'L1_drop30': dict(prior='L1', drop=0.3),
    'S_jit05': dict(prior='S', jitter=0.5),
    # optional arm of idea 1: panel-balanced row weights (each panel the same total weight)
    'L1_pbal': dict(prior='L1', pbal=True), 'S_pbal': dict(prior='S', pbal=True),
}
arms = sys.argv[1].split(',')
for a in arms:
    assert a in ARMS, a
KP = il.KEEPI; JPK = [KP.index(j) for j in il.JP]; TODK = KP.index(il.COLS.index('tod')); DOWK = KP.index(il.COLS.index('dow'))
EXI = {n: i for i, n in enumerate(il.EXTRA)}
D_ALL = 273


def groups(D, scheme):
    return np.arange(D) if scheme == 'L1' else (np.arange(D) // 7) % 4


def design(X, P3, EX, spec):
    """[n, len(KP) + extras]: production columns (prior columns replaced by P3 when given) + the arm's extra columns."""
    M = np.ascontiguousarray(X[:, KP])
    if P3 is not None:
        M[:, JPK] = P3
    ex = spec.get('extra') or []
    if ex:
        M = np.concatenate([M, EX[:, [EXI[n] for n in ex]]], 1)
    return M


def jitter_copy(tr, ctx, spec, rng):
    """idea 3: one augmented copy of every training window: the prior clock shifted by delta in +-6..18 slots, the day class
    flipped with probability 0.25 (prior / prior_now / prior_alldays / tolerant priors looked up there, own day still excluded),
    tod += delta, dow = a day of that class.  State features, site features and labels unchanged."""
    W, L = tr['W'], tr['L']; P = ctx.P; D = len(P['days'])
    t_w = tr['t'].reshape(W, L)[:, 0]; d_w = tr['d'].reshape(W, L)[:, 0]
    delta = rng.integers(6, 19, W) * rng.choice([-1, 1], W)
    t2 = np.clip(t_w + delta, 0, 281)
    c0 = ctx.wk[d_w].astype(int); flip = rng.random(W) < 0.25; c2 = np.where(flip, 1 - c0, c0)
    dow2 = np.where(flip, np.where(c2 == 1, rng.integers(5, 7, W), rng.integers(0, 5, W)), P['dow'][d_w])
    grp = None if spec['prior'] == 'S' else groups(D, spec['prior'])
    p3, ex, _ = il.row_feats(ctx, grp, tr['d'], np.repeat(t2, L), tr['l'], tr['h'], flow=False, cls=np.repeat(c2, L))
    EX = tr[f"EX_{spec['prior']}"].copy()
    EX[:, [EXI[n] for n in TOL]] = ex[:, len(SITE):len(SITE) + len(TOL)]
    M = design(tr['X'], p3, EX, spec)
    M[:, TODK] = np.repeat(t2, L); M[:, DOWK] = np.repeat(dow2, L)
    return M


def train_design(TR, spec, k):
    Xs, ys, ws = [], [], []
    for pi, (tr, ctx) in enumerate(TR):
        e = tr['e']; sch = spec['prior']
        M = design(tr['X'], None if sch == 'S' else tr[f'P3_{sch}'], tr[f'EX_{sch}'], spec)
        y = tr['y']
        if spec.get('drop'):
            rng = np.random.default_rng(1000 + pi)
            dw = np.repeat(rng.random(tr['W']) < spec['drop'], tr['L'])
            M[np.ix_(dw, JPK)] = 0.0; M[dw, TODK] = np.nan
        w = np.ones(len(M), np.float32)
        if spec.get('pbal'):
            w[:] = 1.0 / max(int(e.sum()), 1)
        Xs.append(M[e]); ys.append(y[e]); ws.append(w[e])
        if spec.get('jitter'):
            Mj = jitter_copy(tr, ctx, spec, np.random.default_rng(2000 + pi))
            Xs.append(Mj[e]); ys.append(y[e]); ws.append(np.full(int(e.sum()), spec['jitter'], np.float32))
    X = np.concatenate(Xs); y = np.concatenate(ys); w = np.concatenate(ws)
    if spec.get('pbal'):
        w = w * len(w) / w.sum()
    if k == 4:                                   # production row order: rng(0) subsample / permutation
        sub = np.random.default_rng(0).choice(len(X), min(len(X), 1_200_000), replace=False)
        X, y, w = X[sub], y[sub], w[sub]
    return X, y, (None if np.all(w == 1) else w)


def march_wrong_blend(panel, ev):
    """fold-0 evaluation rows with the WRONG-scenario prior: fold-safe train prior blended with all 31 March days (a = n/(n+4))."""
    P, Pv = il.val_panel(panel)
    cQ, cE = il.cum_tables(Pv, True); D = len(Pv['days'])
    d, t, l = ev['d'], ev['t'], ev['l']
    cls = (P['dow'][d] >= 5).astype(int); s6 = np.minimum(t + 6, 287)
    out = ev['X'][:, il.JP].copy()
    for j, (cc, ss) in enumerate(((cls, s6), (cls, t), (np.full_like(cls, 2), s6))):
        q = cQ[cc, D, ss, l]; ee = cE[cc, D, ss, l]
        sp = np.where(ee > 0, q / np.maximum(ee, 1e-9), 0.0); n = ee / 5.0; a = n / (n + 4.0)
        out[:, j] = (1 - a) * out[:, j] + a * sp
    return out.astype(np.float32)


for k in KS:
    todo = [a for a in arms if not os.path.exists(f'{ODIR}/{a}__k{k}.pkl')]
    if not todo:
        lab.log(f'k={k}: all arms on disk'); continue
    t1 = time.time()
    TR = [il.train_rows(p, k) for p in lab.PANELS]
    if k == 4:
        EV = []
        for p in lab.PANELS:
            b = il.march_bundle(p)
            EV.append(dict(sub=il.march_rc(b), X=b['X'], EX=b['EX'], wloc=b['wi'], l=b['li'], maxprior=b['maxprior'], panel=p,
                           pri={'march_base': None, 'march_s4': il.march_prior(b, 4), 'march_s8': il.march_prior(b, 8)}))
    else:
        EV = []
        for p, (tr, ctx) in zip(lab.PANELS, TR):
            ev = il.eval_rows(p, k, ctx)
            ev['pri'] = {'cv': None}
            if k == 0:
                ev['pri']['wrong'] = march_wrong_blend(p, ev)
            EV.append(ev)
    lab.log(f'k={k}: rows built in {time.time() - t1:.0f}s; train grid rows {sum(len(tr["X"]) for tr, _ in TR)}, '
            f'e rows {sum(int(tr["e"].sum()) for tr, _ in TR)}, eval windows {sum(len(ev["sub"].win) for ev in EV)}')
    for a in todo:
        spec = ARMS[a]; t2 = time.time()
        X, y, w = train_design(TR, spec, k)
        res = []; probs = {}
        for sd in SEEDS:
            if not DRY:
                cat = CatBoostClassifier(iterations=600, depth=8, learning_rate=0.08, thread_count=lab.THREADS, verbose=0, random_seed=sd)
                cat.fit(X, y, sample_weight=w)
            for ev in EV:
                for sname, pri in ev['pri'].items():
                    M = design(ev['X'], pri, ev['EX'], spec)
                    assert M.shape[1] == X.shape[1]
                    if DRY:                     # plumbing check without any training: random scores
                        p = np.random.default_rng(sd).random(len(M)).astype(np.float32) * 0.01
                    else:
                        p = cat.predict_proba(M, thread_count=lab.THREADS)[:, 1].astype(np.float32)
                    df = il.score(ev['sub'], ev['wloc'], ev['l'], p, a, sname)
                    df['seed'] = sd; df['k'] = k; df['maxprior'] = ev['maxprior']
                    res.append(df); probs[(sname, ev['panel'], sd)] = p.astype(np.float16)
            if not DRY:
                del cat
        df = pd.concat(res, ignore_index=True)
        pd.to_pickle(dict(df=df, probs=probs, spec={kk: vv for kk, vv in spec.items()}, n_train=len(X), n_feat=X.shape[1]),
                     f'{ODIR}/{a}__k{k}.pkl')
        msg = ', '.join(f'{s} ' + '/'.join(f'{lab.aggregate(df[(df.set == s) & (df.seed == sd)]):.4f}' for sd in SEEDS)
                        for s in dict.fromkeys(df.set))
        lab.log(f'k={k} arm {a}: {len(X)} rows x {X.shape[1]} cols, pos {y.mean():.4f}, {len(SEEDS)} seed(s) in {time.time() - t2:.0f}s | {msg}')
        del X, y, w; gc.collect()
    del TR, EV; gc.collect()
lab.log('DONE')
