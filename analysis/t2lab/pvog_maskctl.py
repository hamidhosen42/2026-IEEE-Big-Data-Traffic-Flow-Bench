"""pvog EXP 2: masking-adaptation control on pseudo-March (days >= CUT).
MODE = M : the F03 training rows of the TRAIN split with histories / slot T masked like the released masked layer (i.i.d. per
           cell, day regime 0.5 / 0.7 / 0.8 visible drawn from the March shares; every observed cell stays a label)
MODE = X : raw + masked mix (a fixed random half of the F03 windows raw from the lab cache, the other half masked)
For each column set (np, id, full): a train-only model and a model with the March rows of days < CUT added (x WM, the kernel
recipe).  Member probabilities on the pseudo-March blocks are saved (results/pvog/probs/<MODE>_<CUT>_<WM>.pkl).
The real same-scenario training effect = (masked train + March) - (masked train only); masking adaptation = (masked train
only) - (raw train only, exp 1)."""
import gc, os, sys, time
import numpy as np
import pandas as pd
import lab
import exp_firstcross_lib as fc
import pvog_lib as pv

MODE = sys.argv[1]
CUT = int(sys.argv[2]) if len(sys.argv) > 2 else 15
WM = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
SMOKE = bool(int(os.environ.get('SMOKE', 0)))
NT = 20 if SMOKE else 500
PANELS = lab.PANELS[:2] if SMOKE else lab.PANELS
TAG = f'{MODE}_{CUT}_{WM:g}' + ('_smoke' if SMOKE else '')
sp = fc.ARMS['F03']
parts = [pv.masked_f03_rows(p) for p in PANELS]
lab.log('masked F03 rows', sum(len(z['y']) for z in parts), 'regime shares', np.round(pv.march_regime_share(), 3).tolist())
TR, EV = pv.march_data(CUT, WM, eval_from=0, panels=PANELS)
lab.log('March training rows', len(TR['y']), 'weight sum', round(float(TR['w'].sum()), 1))


def train_rows(cs):
    ix = pv.IX[cs]
    if MODE == 'M':
        sel = [np.ones(len(z['y']), bool) for z in parts]
    else:
        sel = [z['half'] for z in parts]
    Xm = [np.ascontiguousarray(z['X'][s][:, ix]) for z, s in zip(parts, sel)]
    ym = [z['y'][s] for z, s in zip(parts, sel)]; wm_ = [z['w'][s] for z, s in zip(parts, sel)]
    n_mask = sum(len(a) for a in ym)
    if MODE == 'X':
        Xr, yr, wr, info = fc.train_matrix(sp['train'], 4, pv.COLS[cs], pv.wfn_half(sp['wfn'], False), panels=PANELS)
        Xm.append(Xr); ym.append(yr); wm_.append(wr)
    X = np.concatenate(Xm); y = np.concatenate(ym).astype(np.float32); w = np.concatenate(wm_).astype(np.float32)
    return X, y, w, n_mask


members = {}
for cs in ('np', 'id', 'full'):
    t0 = time.time()
    X, y, w, n_mask = train_rows(cs)
    m0 = pv.fit_plus(X, y, w, None, None, None, pv.COLS[cs], NT)
    lab.log(f'{MODE} {cs}: train-only model on {len(y)} rows ({n_mask} masked), pos {y.mean():.3f}, weight sum {w.sum():.0f}, {time.time() - t0:.0f}s')
    t0 = time.time()
    Xmar = TR['Xf'] if cs == 'full' else np.ascontiguousarray(TR['X'][:, pv.IX[cs]])
    m1 = pv.fit_plus(X, y, w, Xmar, TR['y'], TR['w'], pv.COLS[cs], NT)
    lab.log(f'{MODE} {cs}: + March model on {len(y) + len(TR["y"])} rows, {time.time() - t0:.0f}s')
    if not SMOKE:
        m0.save_model(f'{pv.RES}/models/model_{TAG}_{cs}.txt'); m1.save_model(f'{pv.RES}/models/model_{TAG}_{cs}B.txt')
    del X, y, w, Xmar; gc.collect()
    if cs == 'full':
        members[f'{MODE}_full_s4'] = (m0, 'full', True); members[f'{MODE}_full_tr'] = (m0, 'full', False)
        members[f'{MODE}B_full'] = (m1, 'full', True)
    else:
        members[f'{MODE}_{cs}'] = (m0, cs, False); members[f'{MODE}B_{cs}'] = (m1, cs, False)
P6 = pv.predict_members(EV, members)
pv.save_probs(TAG, P6)
ARMS = {f'{MODE}:np+id': {f'{MODE}_np': 0.5, f'{MODE}_id': 0.5},
        f'{MODE}:npid+full_s4': {f'{MODE}_np': 0.25, f'{MODE}_id': 0.25, f'{MODE}_full_s4': 0.5},
        f'{MODE}B:np+id': {f'{MODE}B_np': 0.5, f'{MODE}B_id': 0.5},
        f'{MODE}B:npid+full': {f'{MODE}B_np': 0.25, f'{MODE}B_id': 0.25, f'{MODE}B_full': 0.5}}
Rr = pd.concat([pv.score_members(EV, P6, wts, nm) for nm, wts in ARMS.items()], ignore_index=True)
Rr.to_pickle(f'{pv.RES}/exp2_maskctl_{TAG}.pkl')
for sname in ('offlike_m', 'runstart_m'):
    T = pv.cmp_table(Rr, f'{MODE}:np+id', [sname], min_d=CUT)
    print(f'\n== {sname}, days >= {CUT} (base = {MODE}:np+id) =='); print(T.round(4).to_string(index=False))
lab.log('DONE')
