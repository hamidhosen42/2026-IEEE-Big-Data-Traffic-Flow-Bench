"""pvogf TRAIN-OOF sanity check (run through run_heavy; design + PRE-REGISTERED rule in pvogf_lib.py; this check is NOT binding).
pvo recipe in the TRAIN scenario, held-out fold k (default 0): members trained on the F03 lab rows of folds != k (fold-k-excluded train
prior, kernel make_rows, all-finite inputs) + the faithful windows of the masked March month (all 31 days, x WM 5, kernel March rows,
full matrix blended with the fold-k train prior), base arm = pvaf columns vs ogf arm = + the 103 ogfeat columns; evaluated on the
fold-k official-40 / offlike / runstart train windows (train-style slot-T simulation, eligible labels; the full member sees the fold-k
train prior = the train scenario's own prior).  pvo = 0.25 np + 0.25 id + 0.5 full.  255 leaves, lr 0.05, 500 trees, seed 0.
Output: results/pvogf/oof_f<k>.pkl {'df': per-window frame, 'P': probabilities}; row cache data/t2local/cache/pvogf/oof<k>/.
Usage: pvogf_oof.py [fold=0] [seed=0]"""
import gc, json, os, sys, time
import numpy as np
import pandas as pd
import lab
import pvogf_lib as G

kw = dict(a.split('=') for a in sys.argv[1:])
k = int(kw.get('fold', 0)); SEED = int(kw.get('seed', 0))
F, ONGOING = G.F, G.ONGOING
OD = f'{G.CACHE}/oof{k}'; os.makedirs(OD, exist_ok=True)
EVS = ('official', 'offlike', 'runstart')
cols = G.load_cols(); CS = G.colsets(cols)
T00 = time.time()

# ---------------------------------------------------------------- rows (cached per panel)
for panel in lab.PANELS:
    if os.path.exists(f'{OD}/done_{panel}'):
        continue
    PK = G.panel_pk(panel)
    prior = G.K.prior_table(PK, PK['block'] != k); PK['prior_all'] = prior          # fold-k train prior everywhere (no leak of fold k)
    Xs, ys, ws = [], [], []
    for name, wins, ww in G.f03_lab_windows(panel, k=k):
        X, y, e, meta, c_, R = G.train_rows(PK, wins, prior, cols); assert list(c_) == cols
        Xs.append(X[e]); ys.append(y[e].astype(np.float32)); ws.append(ww[meta[:, 0]][e].astype(np.float32)); del X, R
    np.save(f'{OD}/f03_{panel}.npy', np.concatenate(Xs)); np.savez(f'{OD}/f03_{panel}.npz', y=np.concatenate(ys), w=np.concatenate(ws)); del Xs
    M, dpos, cQ, cE = G.K2.scenario(PK)
    cand, Qm, Emv = G.K2.march_faithful(M, PK['cut'])
    prev = np.zeros_like(cand); prev[:, 1:] = cand[:, :-1]
    rs = cand & ~prev; rest = cand & ~rs & (np.arange(288) % 3 == 0)[None, :]
    wl = sorted([(int(a), int(b), G.K2.W_RS) for a, b in zip(*np.where(rs))] + [(int(a), int(b), G.K2.W_DENSE) for a, b in zip(*np.where(rest))])
    d = np.array([w[0] for w in wl]); t = np.array([w[1] for w in wl]); ww = np.array([w[2] for w in wl], np.float32)
    R, X, y, e, meta = G.masked_rows(PK, M, Qm, Emv, d, t, cols)
    jd = np.array([dpos[M['days'][x]] for x in d]); cls = (M['dow'][d] >= 5).astype(int)
    Xb, _ = G.K2.blend_prior_og(X, meta, PK, jd, cls, t, cQ, cE, cols)
    np.save(f'{OD}/march_{panel}.npy', X[e]); np.save(f'{OD}/marchb_{panel}.npy', Xb[e])
    np.savez(f'{OD}/march_{panel}.npz', y=y[e].astype(np.float32), w=(ww[meta[:, 0]] * G.WM)[e]); del R, X, Xb
    for s in EVS:
        win = lab.window_set(panel, s); win = win[(win.cond == ONGOING)].reset_index(drop=True)
        widx = np.where(win.fold.to_numpy() == k)[0]
        if not len(widx):
            continue
        wins = list(zip(win.d.to_numpy()[widx].tolist(), win.t.to_numpy()[widx].tolist(), [ONGOING] * len(widx)))
        X, y, e, meta, c_, R = G.train_rows(PK, wins, prior, cols)
        dd = R['d']; tt = R['t']
        truth = np.stack([PK['Q'][dd, np.minimum(tt + h, 287)] for h in range(1, F + 1)], 1)
        ev = np.stack([PK['E'][dd, np.minimum(tt + h, 287)] for h in range(1, F + 1)], 1)
        np.savez(f'{OD}/ev_{panel}__{s}.npz', X=X, meta=meta.astype(np.int32), truth=truth, ev=ev, qlast=R['qlast'], d=dd, t=tt)
        del X, R
    open(f'{OD}/done_{panel}', 'w').write('ok')
    lab.log(panel, 'oof rows built', f'{time.time() - T00:.0f}s')
    del PK, M, cQ, cE; lab.drop_panel(panel); gc.collect()

# ---------------------------------------------------------------- training (column subsets copied from the memmaps per member)
F03 = [dict(X=np.load(f'{OD}/f03_{p}.npy', mmap_mode='r'), **dict(np.load(f'{OD}/f03_{p}.npz'))) for p in lab.PANELS]
MAR = [dict(X=np.load(f'{OD}/march_{p}.npy', mmap_mode='r'), Xb=np.load(f'{OD}/marchb_{p}.npy', mmap_mode='r'), **dict(np.load(f'{OD}/march_{p}.npz')))
       for p in lab.PANELS]
y_all = np.concatenate([a['y'] for a in F03 + MAR]).astype(np.float32); w_all = np.concatenate([a['w'] for a in F03 + MAR]).astype(np.float32)
lab.log(f'rows F03 {sum(len(a["y"]) for a in F03)} + March {sum(len(a["y"]) for a in MAR)}')


def build_X(member):
    arm, m = member.split('_'); ix = np.array([cols.index(c) for c in CS[(arm, m)]])
    X = np.empty((len(y_all), len(ix)), np.float32); o = 0
    for a in F03:
        n = len(a['y'])
        for s in range(0, n, 500000):
            e = min(n, s + 500000); X[o + s:o + e] = a['X'][s:e][:, ix]
        o += n
    for a in MAR:
        n = len(a['y']); X[o:o + n] = np.asarray(a['Xb'] if m == 'full' else a['X'])[:, ix]; o += n
    return X


EVB = []
for p in lab.PANELS:
    for s in EVS:
        f = f'{OD}/ev_{p}__{s}.npz'
        if os.path.exists(f):
            EVB.append((s, p, dict(np.load(f))))
out = f'{G.RES}/oof_f{k}.pkl'
P = pd.read_pickle(out)['P'] if os.path.exists(out) else {}
for member in [f'{a}_{m}' for a in ('base', 'ogf') for m in ('np', 'id', 'full')]:
    if any(key[2] == member for key in P):
        continue
    arm, m = member.split('_'); cs = CS[(arm, m)]; ix = np.array([cols.index(c) for c in cs])
    t1 = time.time(); mdl = G.lgb_fit(lambda: build_X(member), y_all, w_all, cs, SEED)
    for s, p, z in EVB:
        W, L = z['qlast'].shape; P6 = np.zeros((W, F, L), np.float32)
        P6[z['meta'][:, 0], z['meta'][:, 2] - 1, z['meta'][:, 1]] = mdl.predict(np.ascontiguousarray(z['X'][:, ix]), num_threads=lab.THREADS)
        P[(s, p, member)] = P6.astype(np.float16)
    pd.to_pickle(dict(P=P), out)
    lab.log(f'{member}: {len(cs)} cols, {time.time() - t1:.0f}s'); del mdl; gc.collect()

# ---------------------------------------------------------------- scoring
rows = []
for s, p, z in EVB:
    fam = p.rsplit('_', 1)[0]
    for arm in ('base', 'ogf'):
        P6 = 0.25 * P[(s, p, f'{arm}_np')].astype(np.float32) + 0.25 * P[(s, p, f'{arm}_id')].astype(np.float32) + 0.5 * P[(s, p, f'{arm}_full')].astype(np.float32)
        dec = G.decode6(P6); sc = G.score_dec(dec, z['truth'], z['ev'], z['qlast'])
        rows.append(pd.DataFrame(dict(arm=f'pvo_{arm}', set=s, panel=p, family=fam, cond=ONGOING, d=z['d'], t=z['t'], **sc)))
D = pd.concat(rows, ignore_index=True)
pd.to_pickle(dict(P=P, df=D), out)
for s in EVS:
    A = D[(D.set == s) & (D.arm == 'pvo_base')]; B = D[(D.set == s) & (D.arm == 'pvo_ogf')]
    bt = lab.paired_bootstrap(A, B); pa = A.groupby('panel').iou.mean(); pb = B.groupby('panel').iou.mean()
    lab.log(f'TRAIN OOF fold {k} {s}: n {len(A)} | pvo {lab.aggregate(A):.4f} | pvo-ogf {lab.aggregate(B):.4f} | diff {bt["diff"]:+.4f} [{bt["lo"]:+.4f}, {bt["hi"]:+.4f}]'
            f' panels +{int((pb - pa > 1e-9).sum())}/{len(pa)}')
lab.log('PVOGF OOF DONE', f'{time.time() - T00:.0f}s')
