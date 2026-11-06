"""round8_train.py -- ROUND 8 model training (PREREG results/round8/PREREG.md asserted via round8_lib).  Recipe = EXP_T2_PHASE2.
  prep                        E1 rows (8 panels) + FEAT columns of the dense1500 / offlike onset rows (cached)
  e2 <base|feat> <k> <s...>   fold-k models (dense1500 onset rows outside fold k, fold-k prior, subsample rng(0)) -> cache/round8/models
  e1feat <s...>               full-train FEAT models (all-train prior, subsample rng(0)), base twins = cached cat_base_s*.cbm"""
import os, sys, time, warnings
import numpy as np
import lab
import round8_lib as r8

warnings.filterwarnings('ignore')
MODE = sys.argv[1]
if MODE == 'prep':
    E = r8.e1_all(); lab.log('E1 windows', {p: e['W'] for p, e in E.items()}, 'total', sum(e['W'] for e in E.values()))
    for p in lab.PANELS:
        r8.feat_offlike(p)
    X, y, fo, Z = r8.train_rows(None, True); lab.log('train rows', X.shape, Z.shape, 'pos', round(float(y.mean()), 4))
elif MODE == 'e2':
    arm, k = sys.argv[2], int(sys.argv[3])
    X, y, fo, Z = r8.train_rows(k, arm == 'feat')
    tr = np.where(fo != k)[0]; tr = np.random.default_rng(0).choice(tr, min(len(tr), 1_200_000), replace=False)
    Xa = X[tr] if arm == 'base' else np.concatenate([X[tr], Z[tr]], 1)
    for s in map(int, sys.argv[4:]):
        f = f'{r8.MD}/e2_{arm}_f{k}_s{s}.cbm'
        if os.path.exists(f):
            lab.log('cached', f); continue
        t0 = time.time(); m = r8.fit(Xa, y[tr], s); m.save_model(f); lab.log(f'e2 {arm} f{k} s{s} {time.time() - t0:.0f}s')
elif MODE == 'e1feat':
    X, y, fo, Z = r8.train_rows(None, True)
    sub = np.random.default_rng(0).choice(len(X), min(len(X), 1_200_000), replace=False)
    Xa = np.concatenate([X[sub], Z[sub]], 1)
    for s in map(int, sys.argv[2:]):
        f = f'{r8.MD}/e1_feat_s{s}.cbm'
        if os.path.exists(f):
            lab.log('cached', f); continue
        t0 = time.time(); m = r8.fit(Xa, y[sub], s); m.save_model(f); lab.log(f'e1 feat s{s} {time.time() - t0:.0f}s')
lab.log('DONE')
