"""early track: fold-safe production-like onset CatBoost for the in-distribution downside test (as exp_onset_wrongscen.py arm A):
trained on the dense1500 onset rows of folds 1-3 with the fold-safe prior (prior5[:, 0]), v13-like columns (no ramp).
Saves results/early/cat_onset_noramp_f0.cbm and builds the train fold-0 bundles (official-like onset windows, raw histories)."""
import os, time
import numpy as np
from catboost import CatBoostClassifier
import lab, early_lib as el

path = f'{el.OUT}/cat_onset_noramp_f0.cbm'
if not os.path.exists(path):
    Xt, yt = [], []
    for panel in lab.PANELS:
        rc = lab.rows(panel, 'dense1500', lab.ONSET)
        m = rc.e & (rc.fold != 0)
        Xt.append(rc.matrix(m, el.COLS, 0)); yt.append(rc.y[m].astype(int))
    Xt = np.concatenate(Xt); yt = np.concatenate(yt)
    cat = CatBoostClassifier(iterations=600, depth=8, learning_rate=0.08, thread_count=lab.THREADS, verbose=0, random_seed=0)
    t0 = time.time(); cat.fit(Xt[:, el.KEEPI], yt)
    lab.log('fold-safe onset CatBoost trained on', len(Xt), 'rows, pos rate', round(float(yt.mean()), 4), f'{time.time() - t0:.0f}s')
    cat.save_model(path)
for panel in lab.PANELS:
    b = el.train_bundle(panel)
    lab.log(panel, 'train fold-0 bundle: windows', b['W'], 'L', b['L'], 'days', b['D'], 'visible share', round(float(b['Ev'].mean()), 3))
lab.log('DONE')
