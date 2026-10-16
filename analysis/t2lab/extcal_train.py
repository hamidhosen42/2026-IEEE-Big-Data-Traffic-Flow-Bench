"""extcal CROSS-FIT (run through run_heavy; design + PRE-REGISTERED rule in extcal_lib.py): out-of-sample pvogf probabilities on the
March fitting windows (extcal_build.py fit_<P>.npz).
  MODE CF15  fitting windows of March days < 15; fold f = day parity; members trained on the F03 rows + the pvogf March rows of days < 15
             with parity != f (x WM 5)  -> calibration for the many-day lab (applied to the MD probabilities of days >= 15)
  MODE CF31  fitting windows of all 31 March days; members trained on F03 + March rows of all days with parity != f
             -> calibration for the lab-emulated April decisions (applied to FULL_s0) and for the kernel constants
Members ogf_np / ogf_id / ogf_full (pvogf column sets; full on the blended-prior rows), 255 leaves, lr 0.05, min_data 40, 500 trees.
Output: results/extcal/probs/<MODE>_s<seed>.pkl = {(panel, member): P6 float16 [W_fit, 6, L]} (windows outside the mode are 0).
Usage: extcal_train.py MODE seeds=0,1 [members=ogf_np,ogf_id,ogf_full]"""
import gc, os, sys, time
import numpy as np
import pandas as pd
import lab
import pvogf_lib as G
import extcal_lib as E

MODE = sys.argv[1]
kw = dict(a.split('=') for a in sys.argv[2:])
SEEDS = [int(s) for s in kw.get('seeds', '0').split(',')]
MEMBERS = kw.get('members', 'ogf_np,ogf_id,ogf_full').split(',')
NT = int(kw.get('trees', 500))
assert MODE in ('CF15', 'CF31')
C = G.CACHE; F = G.F
cols = G.load_cols(); CS = G.colsets(cols)
T00 = time.time()
lab.log('MODE', MODE, 'seeds', SEEDS, 'members', MEMBERS, 'threads', lab.THREADS)
F03 = []
for p in lab.PANELS:
    z = np.load(f'{C}/f03_{p}.npz')
    F03.append(dict(X=np.load(f'{C}/f03_{p}.npy', mmap_mode='r'), y=z['y'], w=z['w']))
MZ = {p: dict(np.load(f'{C}/march_{p}.npz')) for p in lab.PANELS}
FIT = {p: np.load(f'{E.CACHE}/fit_{p}.npz') for p in lab.PANELS}


def in_mode(d):
    return (d < E.CUT) if MODE == 'CF15' else np.ones(len(d), bool)


def build(member, fold):
    arm, m = member.split('_'); ix = np.array([cols.index(c) for c in CS[(arm, m)]])
    MAR = []
    for p in lab.PANELS:
        d = MZ[p]['d']; sel = np.where(in_mode(d) & (d % 2 != fold))[0]
        MAR.append(dict(src=np.load(f'{C}/{"marchb" if m == "full" else "march"}_{p}.npy', mmap_mode='r'), sel=sel,
                        y=MZ[p]['y'][sel], w=MZ[p]['w'][sel] * E.WM, days=sorted(set(d[sel].tolist()))))
    y = np.concatenate([a['y'] for a in F03] + [a['y'] for a in MAR]).astype(np.float32)
    w = np.concatenate([a['w'] for a in F03] + [a['w'] for a in MAR]).astype(np.float32)

    def get_X():
        X = np.empty((len(y), len(ix)), np.float32); o = 0
        for a in F03:
            n = len(a['y'])
            for s in range(0, n, 500000):
                e = min(n, s + 500000); X[o + s:o + e] = a['X'][s:e][:, ix]
            o += n
        for a in MAR:
            if len(a['sel']):
                X[o:o + len(a['sel'])] = np.asarray(a['src'][a['sel']])[:, ix]
            o += len(a['sel'])
        assert o == len(y)
        return X
    days = sorted(set(sum([a['days'] for a in MAR], [])))
    return get_X, y, w, ix, days, sum(len(a['sel']) for a in MAR)


for seed in SEEDS:
    f = f'{E.RES}/probs/{MODE}_s{seed}' + ('' if NT == 500 else f'_t{NT}') + '.pkl'
    probs = pd.read_pickle(f) if os.path.exists(f) else {}
    for member in MEMBERS:
        if (lab.PANELS[0], member) in probs:
            lab.log(f'seed {seed} {member}: found, skipped'); continue
        arm, m = member.split('_'); cs = CS[(arm, m)]
        out = {p: np.zeros((len(FIT[p]['d']), F, FIT[p]['qlast'].shape[1]), np.float32) for p in lab.PANELS}
        for fold in (0, 1):
            t1 = time.time()
            get_X, y, w, ix, days, n_mar = build(member, fold)
            lab.log(f'seed {seed} {member} fold {fold}: rows {len(y)} (March {n_mar}, days {days[0]}..{days[-1]}, n days {len(days)})')
            mdl = G.lgb_fit(get_X, y, w, cs, seed, NT, 255)
            t2 = time.time()
            for p in lab.PANELS:
                z = FIT[p]; d = z['d']; meta = z['meta']
                wsel = in_mode(d) & (d % 2 == fold)
                r = wsel[meta[:, 0]]
                if r.any():
                    src = z['Xb'] if m == 'full' else z['X']
                    mr = meta[r]
                    out[p][mr[:, 0], mr[:, 2] - 1, mr[:, 1]] = mdl.predict(np.ascontiguousarray(src[r][:, ix]), num_threads=lab.THREADS)
            lab.log(f'seed {seed} {member} fold {fold}: trained {t2 - t1:.0f}s, predicted {time.time() - t2:.0f}s')
            del mdl; gc.collect()
        for p in lab.PANELS:
            probs[(p, member)] = out[p].astype(np.float16)
        tmp = f'{f}.tmp{os.getpid()}'; pd.to_pickle(probs, tmp); os.replace(tmp, f)
lab.log('EXTCAL CROSS-FIT DONE', MODE, f'{time.time() - T00:.0f}s')
