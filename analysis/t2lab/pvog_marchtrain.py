"""pvog EXP 1: reproduction of exp_ongoing_marchtrain.py "<CUT> <WM>" (default 15 5) with the shared pvog helpers.
Same data, same models (LightGBM 255 leaves / lr 0.05 / 500 trees, seed 0), same arms; in addition the three March-trained
models and all member probabilities on the pseudo-March blocks (all days) are saved for the later experiments.
Regime: train-only models = RAW-history training rows; March rows + all evaluation windows = MASKED layer, visible labels."""
import gc, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import lab
import exp_firstcross_lib as fc
import pvog_lib as pv

CUT = int(sys.argv[1]) if len(sys.argv) > 1 else 15
WM = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
TAG = f'R_{CUT}_{WM:g}'
m_np = lgb.Booster(model_file=f'{pv.MD_FINAL}/model_F03_np255_500.txt')
m_id = lgb.Booster(model_file=f'{pv.MD_FINAL}/model_F03_id255_500.txt')
m_full = lgb.Booster(model_file=f'{pv.MD_FINAL}/model_F03_full255_500.txt')

TR, EV = pv.march_data(CUT, WM, eval_from=0)
lab.log('March training rows', len(TR['y']), 'pos', round(float(TR['y'].mean()), 4), 'weight sum', round(float(TR['w'].sum()), 1))
sp = fc.ARMS['F03']
newm = {}
for nm, cs in (('np', 'np'), ('id', 'id'), ('full', 'full')):
    t0 = time.time()
    X, y, w, info = fc.train_matrix(sp['train'], 4, pv.COLS[cs], sp['wfn'])
    Xm = TR['Xf'] if cs == 'full' else np.ascontiguousarray(TR['X'][:, pv.IX[cs]])
    n_tr, ws = len(y), float(w.sum())
    newm[nm] = pv.fit_plus(X, y, w, Xm, TR['y'], TR['w'], pv.COLS[cs])
    newm[nm].save_model(f'{pv.RES}/models/model_{TAG}_{nm}B.txt')
    lab.log(f'{nm}B trained on {n_tr + len(TR["y"])} rows ({len(TR["y"])} March; weight sum train {ws:.0f} / March {TR["w"].sum():.0f}) {time.time() - t0:.0f}s')
    del X, y, w, Xm; gc.collect()

members = {'R_np': (m_np, 'np', False), 'R_id': (m_id, 'id', False), 'R_full_s4': (m_full, 'full', True), 'R_full_tr': (m_full, 'full', False),
           'RB_np': (newm['np'], 'np', False), 'RB_id': (newm['id'], 'id', False), 'RB_full': (newm['full'], 'full', True)}
P6 = pv.predict_members(EV, members)
pv.save_probs(TAG, P6)
ARMS = {'np+id': {'R_np': 0.5, 'R_id': 0.5},
        'npid+full_s4': {'R_np': 0.25, 'R_id': 0.25, 'R_full_s4': 0.5},
        'npB+idB': {'RB_np': 0.5, 'RB_id': 0.5},
        'npBidB+fullB': {'RB_np': 0.25, 'RB_id': 0.25, 'RB_full': 0.5},
        'npid+fullB': {'R_np': 0.25, 'R_id': 0.25, 'RB_full': 0.5},
        'fullB': {'RB_full': 1.0}}
Rr = pd.concat([pv.score_members(EV, P6, wts, nm) for nm, wts in ARMS.items()], ignore_index=True)
Rr.to_pickle(f'{pv.RES}/exp1_marchtrain_{CUT}_{WM:g}.pkl')
for sname in ('offlike_m', 'runstart_m'):
    T = pv.cmp_table(Rr, 'np+id', [sname], min_d=CUT)
    print(f'\n== {sname}, days >= {CUT}: n = {int(T.n.iloc[0])} =='); print(T.round(4).to_string(index=False))
lab.log('DONE')
