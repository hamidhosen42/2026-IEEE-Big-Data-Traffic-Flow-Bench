"""pvogf TRAIN (run through run_heavy): lab-scale members on the pvogf_build.py caches (design + PRE-REGISTERED rule: pvogf_lib.py).
  MODE T0   train-only members (F03 rows only): base_np, base_id, ogf_np, ogf_id         -> pseudo-March eval sets + official private
  MODE MD   many-day regime: F03 rows + March rows of days < CUT (x WM): base_/ogf_ np, id, full -> pseudo-March eval sets (days >= CUT)
  MODE FULL kernel regime: F03 rows + March rows of ALL days (x WM): base_/ogf_ np, id, full     -> official private windows
'full' members are trained on the F03 rows (all-train prior) + the March rows with the blended same-scenario prior (= kernel fullB) and
predict with the blended matrices (Xb).  255 leaves, lr 0.05, min_data 40, 500 trees (lab.lgb_params, seed s).
Output: results/pvogf/probs/<MODE>_s<seed>.pkl = {(set, panel, member): P6 float16 [W, 6, L]} (set 'private' for the official windows),
models: data/t2local/cache/pvogf/models/<MODE>_s<seed>_<member>.txt
Usage: pvogf_train.py MODE seeds=0,1 [trees=500] [leaves=255] [members=base_np,...]"""
import gc, json, os, sys, time
import numpy as np
import pandas as pd
import lab
import pvogf_lib as G

MODE = sys.argv[1]
kw = dict(a.split('=') for a in sys.argv[2:])
SEEDS = [int(s) for s in kw.get('seeds', '0').split(',')]
NT = int(kw.get('trees', 500)); LEAVES = int(kw.get('leaves', 255))
C = G.CACHE; F = G.F
cols = G.load_cols(); CS = G.colsets(cols)
ALL_MEMBERS = [f'{a}_{m}' for a in ('base', 'ogf') for m in (('np', 'id') if MODE == 'T0' else ('np', 'id', 'full'))]
MEMBERS = kw['members'].split(',') if 'members' in kw else ALL_MEMBERS
assert MODE in ('T0', 'MD', 'FULL') and set(MEMBERS) <= set(ALL_MEMBERS)
T00 = time.time()
lab.log('MODE', MODE, 'seeds', SEEDS, 'members', MEMBERS, 'trees', NT, 'leaves', LEAVES, 'threads', lab.THREADS)

# ---------------------------------------------------------------- training rows (memmaps; column subsets copied per member)
F03 = []; MAR = []
for p in lab.PANELS:
    z = np.load(f'{C}/f03_{p}.npz')
    F03.append(dict(X=np.load(f'{C}/f03_{p}.npy', mmap_mode='r'), y=z['y'], w=z['w']))
    if MODE != 'T0':
        z = np.load(f'{C}/march_{p}.npz')
        sel = np.where(z['d'] < G.CUT)[0] if MODE == 'MD' else np.arange(len(z['y']))
        MAR.append(dict(X=np.load(f'{C}/march_{p}.npy', mmap_mode='r'), Xb=np.load(f'{C}/marchb_{p}.npy', mmap_mode='r'), sel=sel,
                        y=z['y'][sel], w=z['w'][sel] * G.WM, dmax=int(z['d'][sel].max()) if len(sel) else -1))
y_all = np.concatenate([a['y'] for a in F03] + [a['y'] for a in MAR]).astype(np.float32)
w_all = np.concatenate([a['w'] for a in F03] + [a['w'] for a in MAR]).astype(np.float32)
n_f03 = sum(len(a['y']) for a in F03); n_mar = sum(len(a['y']) for a in MAR)
if MODE == 'MD':
    assert max(a['dmax'] for a in MAR) < G.CUT
lab.log(f'rows: F03 {n_f03} (pos {np.concatenate([a["y"] for a in F03]).mean():.4f}, weight {sum(a["w"].sum() for a in F03):.0f})'
        f' | March {n_mar} (weight {sum(a["w"].sum() for a in MAR):.0f}' + (f', days <= {max(a["dmax"] for a in MAR)}' if MAR else '') + ')')


def build_X(member):
    arm, m = member.split('_'); ix = np.array([cols.index(c) for c in CS[(arm, m)]])
    X = np.empty((len(y_all), len(ix)), np.float32); o = 0
    for a in F03:
        n = len(a['y'])
        for s in range(0, n, 500000):
            e = min(n, s + 500000); X[o + s:o + e] = a['X'][s:e][:, ix]
        o += n
    for a in MAR:
        src = a['Xb'] if m == 'full' else a['X']
        if len(a['sel']):
            X[o:o + len(a['sel'])] = np.asarray(src[a['sel']])[:, ix]
        o += len(a['sel'])
    assert o == len(y_all)
    return X


# ---------------------------------------------------------------- prediction blocks
def blocks():
    out = []
    if MODE in ('T0', 'MD'):
        for p in lab.PANELS:
            for s in G.PM_SETS:
                z = np.load(f'{C}/ev_{p}__{s}.npz')
                out.append((s, p, z['X'], z['Xb'], z['meta'], z['qlast'].shape))
    if MODE in ('T0', 'FULL'):
        for p in lab.PANELS:
            z = np.load(f'{C}/priv_{p}.npz')
            out.append(('private', p, z['X'], z['Xb'], z['meta'], z['qlast'].shape))
    return out


BL = blocks()
for seed in SEEDS:
    f = f'{G.RES}/probs/{MODE}_s{seed}.pkl'
    probs = pd.read_pickle(f) if os.path.exists(f) else {}
    for member in MEMBERS:
        if any(k[2] == member for k in probs):
            lab.log(f'seed {seed} {member}: found, skipped'); continue
        arm, m = member.split('_'); cs = CS[(arm, m)]; ix = np.array([cols.index(c) for c in cs])
        t1 = time.time()
        mdl = G.lgb_fit(lambda: build_X(member), y_all, w_all, cs, seed, NT, LEAVES)
        mdl.save_model(f'{C}/models/{MODE}_s{seed}_{member}.txt')
        t2 = time.time()
        for s, p, X, Xb, meta, (W, L) in BL:
            P6 = np.zeros((W, F, L), np.float32)
            if len(meta):
                src = Xb if m == 'full' else X
                P6[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = mdl.predict(np.ascontiguousarray(src[:, ix]), num_threads=lab.THREADS)
            probs[(s, p, member)] = P6.astype(np.float16)
        tmp = f'{f}.tmp{os.getpid()}'; pd.to_pickle(probs, tmp); os.replace(tmp, f)
        imp = pd.Series(mdl.feature_importance('gain'), index=cs); new_share = float(imp[[c for c in cs if c in G.NEW]].sum() / imp.sum())
        lab.log(f'seed {seed} {member}: {len(cs)} cols, trained {t2 - t1:.0f}s, predicted {time.time() - t2:.0f}s | gain share of ogfeat cols {new_share:.3f}'
                f' | top5 {imp.sort_values(ascending=False).index[:5].tolist()}')
        del mdl; gc.collect()
lab.log('PVOGF TRAIN DONE', MODE, f'{time.time() - T00:.0f}s')
