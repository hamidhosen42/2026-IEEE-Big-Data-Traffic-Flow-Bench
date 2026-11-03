"""phase2_onset.py -- EXP_T2_PHASE2 onset re-test (PREREG results/phase2/PREREG.txt, asserted).
Same recipe as phase_onset.py.  Modes:
  train <seed> ...   train + cache the full-train base / phase CatBoost models (cache/phase/onset/cat_{arm}_s{seed}.cbm)
  oof <k>            E2: fold-k held-out models (seeds 0, 1, 2), scored on the fold-k offlike onset windows -> results/phase2/oof_f{k}.pkl
  pm                 E1: 214 pseudo-March onset windows, seeds 0..19, all-train prior ('base') and sp4 prior -> results/phase2/pm.pkl"""
import gc, os, sys, time, types, warnings
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
import lab, ogfeat_lib as og
import phase_lib as ph

ph.check_prereg2()
H, F = lab.H, lab.F
cols = lab.all_cols()
ODIR = f'{ph.ARMD}/onset'; os.makedirs(ODIR, exist_ok=True)
SEEDS = tuple(range(20)); OOF_SEEDS = (0, 1, 2)
MODE = sys.argv[1]


def fit(X, y, seed):
    m = CatBoostClassifier(iterations=600, depth=8, learning_rate=0.08, thread_count=lab.THREADS, verbose=0, random_seed=seed,
                           train_dir=f'{ph.ARMD}/catboost_info_{MODE}_{os.getpid()}')
    m.fit(X, y); return m


def train_rows(prior_k=None):
    Xs, ys, Zs, fs = [], [], [], []
    for p in lab.PANELS:
        rc = lab.rows(p, 'dense1500', lab.ONSET); assert list(rc.cols) == cols, p
        r = np.where(rc.e)[0]
        Xs.append(np.asarray(rc.X[r]) if prior_k is None else rc.matrix(rc.e, cols, prior_k))
        ys.append(rc.y[r].astype(int)); fs.append(rc.fold[r])
        Zs.append(ph.phase_cols(rc.win.t.to_numpy()[rc.wi[r]], rc.h[r]))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(Zs), np.concatenate(fs)


def model_path(arm, s):
    return f'{ODIR}/cat_{arm}_s{s}.cbm'


if MODE == 'train':
    X, y, Z, _ = train_rows()
    sub = np.random.default_rng(0).choice(len(X), min(len(X), 1_200_000), replace=False)
    lab.log('onset train rows', len(X), 'sub', len(sub))
    for s in map(int, sys.argv[2:]):
        for arm in ('base', 'phase'):
            f = model_path(arm, s)
            if os.path.exists(f):
                lab.log(f'{arm} s{s} cached'); continue
            t0 = time.time()
            m = fit(X[sub] if arm == 'base' else np.concatenate([X[sub], Z[sub]], 1), y[sub], s)
            m.save_model(f); lab.log(f'{arm} s{s} trained {time.time() - t0:.0f}s')
    lab.log('DONE')

elif MODE == 'oof':
    k = int(sys.argv[2])
    X, y, Z, fo = train_rows(prior_k=k)
    tr = np.where(fo != k)[0]; tr = np.random.default_rng(0).choice(tr, min(len(tr), 1_200_000), replace=False)
    lab.log(f'fold {k}: train rows {len(tr)}')
    ev = []
    for panel in lab.PANELS:
        rc = lab.rows(panel, 'offlike', lab.ONSET)
        widx = np.where(rc.win.fold.to_numpy() == k)[0]
        if not len(widx) or not rc.n:
            continue
        msk = (rc.win.fold.to_numpy() == k)[rc.wi] & (rc.h == F)
        Xe = rc.matrix(msk, cols, k); r = np.where(msk)[0]; Ze = ph.phase_cols(rc.win.t.to_numpy()[rc.wi[r]], rc.h[r])
        ev.append((panel, rc, widx, r, Xe, Ze))
    out = []
    for s in OOF_SEEDS:
        for arm in ('base', 'phase'):
            t0 = time.time()
            m = fit(X[tr] if arm == 'base' else np.concatenate([X[tr], Z[tr]], 1), y[tr], s)
            for panel, rc, widx, r, Xe, Ze in ev:
                P6 = np.zeros((len(rc.win), F, rc.L), np.float32)
                P6[rc.wi[r], F - 1, rc.li[r]] = m.predict_proba(Xe if arm == 'base' else np.concatenate([Xe, Ze], 1))[:, 1]
                df = lab.score_probs(rc, P6, widx, arm=arm, set_name='offlike'); df['seed'] = s; df['a'] = arm; df['k'] = k
                out.append(df)
            lab.log(f'fold {k} {arm} s{s} {time.time() - t0:.0f}s')
    pd.concat(out, ignore_index=True).to_pickle(f'{ph.RES2}/oof_f{k}.pkl'); lab.log('DONE')

elif MODE == 'pm':
    onset_rows, cum_tables = ph.onset_rows, ph.cum_tables
    models = {}
    for s in SEEDS:
        for arm in ('base', 'phase'):
            m = CatBoostClassifier(); m.load_model(model_path(arm, s)); models[(arm, s)] = m
    JP = [cols.index(c) for c in ('prior', 'prior_now', 'prior_alldays')]
    rows_out = []
    for panel in lab.PANELS:
        P, Pv = og.val_panel(panel)
        on, ogm, _ = og.pm_select(Pv)
        D = len(Pv['days'])
        g_on, g_og = lab._glist(on), lab._glist(ogm); sset = set()
        for d0 in range(D):
            a, _ = lab.greedy(g_on, g_og, 'interleaved', start_day=d0); sset.update(a)
        wins = sorted(sset)
        if not wins:
            continue
        prior_tr = lab.prior_table(P, np.ones(len(P['days']), bool))
        R = onset_rows(P, Pv, [(a, b, lab.ONSET) for a, b in wins], prior_tr)
        Xm, _, _, meta, call = lab.to_matrix(R, lab.ONSET, None, None); assert list(call) == cols
        m6 = meta[:, 2] == F; Xm = Xm[m6]; meta = meta[m6]
        Zm = ph.phase_cols(R['t'][meta[:, 0]], meta[:, 2])
        d, t = R['d'], R['t']
        # sp4 prior (exp_onset_splitprior.py s4: smoothed +-2 slots, a = n/(n+4), March days < d), step-6 rows only
        wi, li = meta[:, 0], meta[:, 1]
        cls = (Pv['dow'][d] >= 5).astype(int); s6 = np.minimum(t + 6, 287)
        c_r, d_r, s6_r, t_r = cls[wi], d[wi], s6[wi], t[wi]
        trv = [prior_tr[c_r, s6_r, li], prior_tr[c_r, t_r, li], prior_tr[2, s6_r, li]]
        cQ, cE = cum_tables(Pv, True)
        Xs4 = Xm.copy()
        for j, (cc, ss) in enumerate(((c_r, s6_r), (c_r, t_r), (np.full_like(c_r, 2), s6_r))):
            q = cQ[cc, d_r, ss, li]; ee = cE[cc, d_r, ss, li]
            sp = np.where(ee > 0, q / np.maximum(ee, 1e-9), 0.0); n = ee / 5.0; a_ = n / (n + 4)
            Xs4[:, JP[j]] = (1 - a_) * trv[j] + a_ * sp
        del cQ, cE
        win = pd.DataFrame(dict(d=d, t=t, cond=lab.ONSET, fold=0))
        rc = types.SimpleNamespace(truth=np.stack([Pv['Q'][d, t + h] for h in range(1, F + 1)], 1),
                                   ev=np.stack([Pv['E'][d, t + h] for h in range(1, F + 1)], 1),
                                   qlast=R['qlast'], win=win, name='march_on', panel=panel, cond=lab.ONSET, L=R['L'])
        for (arm, s), m in models.items():
            for pv, Xb in (('base', Xm), ('sp4', Xs4)):
                Xa = Xb if arm == 'base' else np.concatenate([Xb, Zm], 1)
                P6 = np.zeros((R['W'], F, R['L']), np.float32); P6[wi, F - 1, li] = m.predict_proba(Xa)[:, 1]
                df = lab.score_probs(rc, P6, arm=f'{arm}_s{s}_{pv}', set_name='march_on'); df['seed'] = s; df['a'] = arm; df['pv'] = pv
                rows_out.append(df)
        lab.log(panel, 'pm onset windows', len(wins))
        lab.drop_panel(panel); gc.collect()
    pd.concat(rows_out, ignore_index=True).to_pickle(f'{ph.RES2}/pm.pkl'); lab.log('DONE')
