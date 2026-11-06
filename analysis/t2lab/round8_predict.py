"""round8_predict.py -- ROUND 8 step-6 probabilities of every model on E1 / E2 rows (PREREG asserted via round8_lib).
  e1   base s0..19 (cached EXP_T2_PHASE / PHASE2 models) on the all-train-prior and sp4 rows; FEAT s0..19 (if trained)
  e2   per fold k: base s0..9 and FEAT s0..9 (if trained) on the fold-k offlike onset rows
Output cache/round8/pred_e1.pkl, pred_e2.pkl (dict -> arrays [n_seeds, n_rows], NaN rows for missing models)."""
import os, sys, warnings
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
import lab
import round8_lib as r8

warnings.filterwarnings('ignore')
MODE = sys.argv[1]


def load(f):
    if not os.path.exists(f):
        return None
    m = CatBoostClassifier(); m.load_model(f); return m


def proba(m, X):
    return m.predict_proba(X, thread_count=lab.THREADS)[:, 1]


if MODE == 'e1':
    E = r8.e1_all(); out = {}
    base = [load(r8.E1_MODEL.format(s=s)) for s in range(20)]; feat = [load(f'{r8.MD}/e1_feat_s{s}.cbm') for s in range(20)]
    assert all(m is not None for m in base)
    lab.log('feat models available', sum(m is not None for m in feat))
    for p, e in E.items():
        XZ = np.concatenate([e['Xm'], e['Z']], 1)
        out[p] = dict(base=np.stack([proba(m, e['Xm']) for m in base]), base_sp4=np.stack([proba(m, e['Xs4']) for m in base]),
                      feat=np.stack([proba(m, XZ) if m is not None else np.full(len(XZ), np.nan) for m in feat]))
        lab.log(p, len(XZ), 'rows')
    pd.to_pickle(out, f'{r8.CD}/pred_e1.pkl')
elif MODE == 'e2':
    out = {}
    for k in range(4):
        rows = r8.e2_rows(k)
        base = [load(f'{r8.MD}/e2_base_f{k}_s{s}.cbm') for s in range(10)]; feat = [load(f'{r8.MD}/e2_feat_f{k}_s{s}.cbm') for s in range(10)]
        lab.log(f'fold {k}: base models {sum(m is not None for m in base)}, feat models {sum(m is not None for m in feat)}')
        for p, r in rows.items():
            XZ = np.concatenate([r['X'], r['Z']], 1)
            out[(k, p)] = dict(base=np.stack([proba(m, r['X']) if m is not None else np.full(len(XZ), np.nan) for m in base]),
                               feat=np.stack([proba(m, XZ) if m is not None else np.full(len(XZ), np.nan) for m in feat]))
    pd.to_pickle(out, f'{r8.CD}/pred_e2.pkl')
lab.log('DONE')
