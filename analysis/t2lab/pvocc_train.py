"""pvocc TRAIN (run through run_heavy): lab-scale occupancy-on members on the pvogf caches + the pvocc side-cars (design + PRE-REGISTERED
rule: pvocc_lib.py).  The training rows, weights, row order, prediction blocks and fit path are those of pvogf_train.py / pvogf_oof.py;
the O arms append the 13 occupancy columns (side-car) after the cached columns.
  MODE MD   F03 rows + March rows of days < CUT (x WM): ogfO_ / baseO_ np, id, full -> pseudo-March sets offlike_m, runstart_m (days >= CUT)
  MODE FULL F03 rows + March rows of ALL days (x WM)                                 -> the 40 official private ongoing windows
  MODE OOF  pvogf_oof.py fold-0 rows (F03 folds != 0 + all March days x WM, fold-0 prior, fold-0-free calibration) -> fold-0 train sets
'full' members train on the blended-prior March matrix (marchb) and predict with Xb (= kernel fullB).  255 leaves, lr 0.05, min_data 40,
500 trees, lab.lgb_params(seed).  Members 'base_np' / 'ogf_np' (no occupancy) are the INTEGRITY (iv) path: they must reproduce the cached
pvogf model trees of the same MODE / seed.
Output: results/pvocc/probs/<MODE>_s<seed>.pkl (MD / FULL) or results/pvocc/oof_f0.pkl {(set, panel, member): P6 float16 [W, 6, L]};
models data/t2local/cache/pvocc/models/<MODE>_s<seed>_<member>.txt; integrity results/pvocc/parity_iv.json.
Usage: pvocc_train.py MODE seeds=0 [members=ogf_np,ogfO_np,...]"""
import gc, json, os, sys, time
import numpy as np
import pandas as pd
import lab
import pvogf_lib as G
import pvocc_lib as V

MODE = sys.argv[1]
kw = dict(a.split('=') for a in sys.argv[2:])
SEEDS = [int(s) for s in kw.get('seeds', '0').split(',')]
NT, LEAVES = 500, 255
F = V.F
cols = G.load_cols(); CS = V.colsets_o(cols)
ALL_MEMBERS = [f'{a}_{m}' for a in ('ogfO', 'baseO') for m in ('np', 'id', 'full')]
MEMBERS = kw['members'].split(',') if 'members' in kw else ALL_MEMBERS
assert MODE in ('MD', 'FULL', 'OOF') and all(m.split('_')[0] in ('base', 'ogf', 'baseO', 'ogfO') for m in MEMBERS)
V.check_prereg()
T00 = time.time()
lab.log('MODE', MODE, 'seeds', SEEDS, 'members', MEMBERS, 'trees', NT, 'leaves', LEAVES, 'threads', lab.THREADS)

# ---------------------------------------------------------------- training rows (memmaps + side-cars), exactly as pvogf_train / pvogf_oof
src_dir = f'{V.GC}/oof0' if MODE == 'OOF' else V.GC
tag = 'oof' if MODE == 'OOF' else 'md'
F03, MAR, SIDE = [], [], {}
for p in lab.PANELS:
    S = V.Side(p, tag); SIDE[p] = S
    z = np.load(f'{src_dir}/f03_{p}.npz')
    F03.append(dict(X=np.load(f'{src_dir}/f03_{p}.npy', mmap_mode='r'), y=z['y'], w=z['w'], O=S['f03']))
    assert len(S['f03']) == len(z['y']), p
    z = np.load(f'{src_dir}/march_{p}.npz')
    if MODE == 'MD':
        sel = np.where(z['d'] < G.CUT)[0]
    else:
        sel = np.arange(len(z['y']))
    w = z['w'][sel] * (G.WM if MODE != 'OOF' else 1.0)          # pvogf_oof stores the March weights with WM already applied
    MAR.append(dict(X=np.load(f'{src_dir}/march_{p}.npy', mmap_mode='r'), Xb=np.load(f'{src_dir}/marchb_{p}.npy', mmap_mode='r'), sel=sel,
                    y=z['y'][sel], w=w, O=S['march'], dmax=int(z['d'][sel].max()) if ('d' in z and len(sel)) else -1))
    assert len(S['march']) == len(z['y']), p
y_all = np.concatenate([a['y'] for a in F03] + [a['y'] for a in MAR]).astype(np.float32)
w_all = np.concatenate([a['w'] for a in F03] + [a['w'] for a in MAR]).astype(np.float32)
if MODE == 'MD':
    assert max(a['dmax'] for a in MAR) < G.CUT
lab.log(f'rows: F03 {sum(len(a["y"]) for a in F03)} | March {sum(len(a["y"]) for a in MAR)} (weight {sum(a["w"].sum() for a in MAR):.0f})')


def member_cols(member):
    arm, m = member.split('_')
    return arm, m, CS[(arm, m)]


def build_X(member):
    arm, m, cs = member_cols(member)
    occ = arm.endswith('O'); base = CS[(arm[:-1] if occ else arm, m)]
    ix = np.array([cols.index(c) for c in base]); nb = len(ix)
    X = np.empty((len(y_all), len(cs)), np.float32); o = 0
    for a in F03:
        n = len(a['y'])
        for s in range(0, n, 500000):
            e = min(n, s + 500000); X[o + s:o + e, :nb] = a['X'][s:e][:, ix]
            if occ:
                X[o + s:o + e, nb:] = a['O'][s:e]
        o += n
    for a in MAR:
        src = a['Xb'] if m == 'full' else a['X']
        if len(a['sel']):
            X[o:o + len(a['sel']), :nb] = np.asarray(src[a['sel']])[:, ix]
            if occ:
                X[o:o + len(a['sel']), nb:] = a['O'][a['sel']]
        o += len(a['sel'])
    assert o == len(y_all)
    return X


# ---------------------------------------------------------------- prediction blocks (X / Xb + side-car rows)
def blocks():
    out = []
    if MODE == 'MD':
        for p in lab.PANELS:
            for s in G.PM_SETS:
                z = np.load(f'{V.GC}/ev_{p}__{s}.npz')
                out.append((s, p, z['X'], z['Xb'], SIDE[p][f'ev__{s}'], z['meta'], z['qlast'].shape))
    elif MODE == 'FULL':
        for p in lab.PANELS:
            z = np.load(f'{V.GC}/priv_{p}.npz')
            out.append(('private', p, z['X'], z['Xb'], SIDE[p]['priv'], z['meta'], z['qlast'].shape))
    else:
        for p in lab.PANELS:
            for s in ('official', 'offlike', 'runstart'):
                f = f'{src_dir}/ev_{p}__{s}.npz'
                if os.path.exists(f):
                    z = np.load(f)
                    out.append((s, p, z['X'], z['X'], SIDE[p][f'ev__{s}'], z['meta'], z['qlast'].shape))
    return out


def trees(s):
    return s[s.index('Tree=0'):s.index('end of trees')]


BL = blocks()
for seed in SEEDS:
    f = f'{V.RES}/oof_f0.pkl' if MODE == 'OOF' else f'{V.RES}/probs/{MODE}_s{seed}.pkl'
    probs = pd.read_pickle(f) if os.path.exists(f) else {}
    for member in MEMBERS:
        if any(k[2] == member for k in probs):
            lab.log(f'seed {seed} {member}: found, skipped'); continue
        arm, m, cs = member_cols(member)
        occ = arm.endswith('O'); base = CS[(arm[:-1] if occ else arm, m)]; ix = np.array([cols.index(c) for c in base])
        t1 = time.time()
        mdl = V.lgb_fit_locked(lambda: build_X(member), y_all, w_all, cs, seed, NT, LEAVES)
        mf = f'{V.CACHE}/models/{MODE}_s{seed}_{member}.txt'; mdl.save_model(mf)
        t2 = time.time()
        for s, p, X, Xb, O, meta, (W, L) in BL:
            P6 = np.zeros((W, F, L), np.float32)
            if len(meta):
                src = Xb if m == 'full' else X
                Xp = np.ascontiguousarray(src[:, ix]) if not occ else np.ascontiguousarray(np.hstack([src[:, ix], O]))
                P6[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = mdl.predict(Xp, num_threads=lab.THREADS)
            probs[(s, p, member)] = P6.astype(np.float16)
        tmp = f'{f}.tmp{os.getpid()}'; pd.to_pickle(probs, tmp); os.replace(tmp, f)
        imp = pd.Series(mdl.feature_importance('gain'), index=cs)
        occ_share = float(imp[[c for c in cs if c in V.OCC_COLS]].sum() / imp.sum()) if occ else 0.0
        lab.log(f'seed {seed} {member}: {len(cs)} cols, trained {t2 - t1:.0f}s, predicted {time.time() - t2:.0f}s | gain share of occupancy cols {occ_share:.3f}'
                f' | top5 {imp.sort_values(ascending=False).index[:5].tolist()}'
                + (f' | occ top {imp[V.OCC_COLS].sort_values(ascending=False).index[:3].tolist()}' if occ else ''))
        if not occ and MODE in ('MD', 'FULL'):                         # INTEGRITY (iv): the no-occupancy path vs the cached pvogf model
            ref = f'{G.CACHE}/models/{MODE}_s{seed}_{member}.txt'
            same_trees = os.path.exists(ref) and trees(open(ref).read()) == trees(mdl.model_to_string())
            cp = pd.read_pickle(f'{G.RES}/probs/{MODE}_s{seed}.pkl')
            dmax = max(float(np.max(np.abs(probs[(s, p, member)].astype(np.float32) - cp[(s, p, member)].astype(np.float32)))) for s, p, *_ in BL)
            r = dict(mode=MODE, seed=seed, member=member, threads=lab.THREADS, trees_identical=bool(same_trees), probs_max_abs_diff=dmax)
            pf = f'{V.RES}/parity_iv.json'; old = json.load(open(pf)) if os.path.exists(pf) else []
            json.dump([o for o in old if (o['mode'], o['seed'], o['member']) != (MODE, seed, member)] + [r], open(pf, 'w'), indent=1)
            lab.log('INTEGRITY (iv)', json.dumps(r))
        del mdl; gc.collect()
lab.log('PVOCC TRAIN DONE', MODE, f'{time.time() - T00:.0f}s')
