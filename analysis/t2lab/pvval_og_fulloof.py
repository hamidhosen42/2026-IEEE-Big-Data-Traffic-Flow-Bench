"""pvval: IN-DISTRIBUTION check of the train-only FULL ongoing member (the 0.5 'full' half of the April blend).
OOF over the 4 train day blocks: F03 rows, column set 'full' (prior columns kept), LightGBM 255 leaves / lr 0.05 / 500 trees /
seed 0 = identical rows and parameters as the reference arms fc_F03_np255 / fc_F03_id255 (only the columns differ).
At evaluation the prior columns hold the fold-safe TRAIN prior (= the right-scenario prior in-distribution).
Then: np+id (reference blend) vs 0.25 np + 0.25 id + 0.5 full (the April recipe shape) and other weights, on offlike / official.
Fold files -> data/t2local/cache/pvval/oof/pvval_F03_full255/ (fc.arm_dir redirected; nothing in cache/oof is touched)."""
import os, sys
import numpy as np
import pandas as pd
import lab
import exp_firstcross_lib as fc
import pvval_lib as V

ARM = 'pvval_F03_full255'
ODIR = f'{V.CACHE}/oof/{ARM}'
os.makedirs(ODIR, exist_ok=True)
fc.arm_dir = lambda arm: f'{V.CACHE}/oof/{arm}'
folds = tuple(int(x) for x in sys.argv[1].split(',')) if len(sys.argv) > 1 else (0, 1, 2, 3)
if folds != (9,):
    fc.run_arm(ARM, fc.ARMS['F03'], cols='full', leaves=255, lr=0.05, n_trees=500, mdl=40, folds=folds, evals=('official', 'offlike'))


def load_full(set_name, panel):
    z = np.load(f'{ODIR}/probs__{set_name}__{panel}.npz')
    return z['P6'].astype(np.float32), z['done']


BL = {'np+id': {'np': 0.5, 'id': 0.5}, 'npid+full': {'np': 0.25, 'id': 0.25, 'full': 0.5}, 'np+id+full/3': {'np': 1 / 3, 'id': 1 / 3, 'full': 1 / 3},
      'npid75+full25': {'np': 0.375, 'id': 0.375, 'full': 0.25}, 'full': {'full': 1.0}, 'np': {'np': 1.0}, 'id': {'id': 1.0}}
res = []
for s in ('offlike', 'official'):
    for panel in lab.PANELS:
        rc = lab.rows(panel, s, lab.ONGOING)
        if not len(rc.win):
            continue
        pn, dn = lab.load_probs('fc_F03_np255', s, panel); pi, di = lab.load_probs('fc_F03_id255', s, panel)
        pf, df_ = load_full(s, panel)
        done = dn & di & df_
        widx = np.where(done)[0]
        mem = {'np': pn.astype(np.float32), 'id': pi.astype(np.float32), 'full': pf}
        for nm, wts in BL.items():
            P6 = sum(wt * mem[m] for m, wt in wts.items()).astype(np.float32)
            res.append(lab.score_probs(rc, P6, widx, arm=nm, set_name=s))
R = pd.concat(res, ignore_index=True)
R.to_pickle(f'{V.RES}/fulloof_blends.pkl')
rows = []
for s in ('offlike', 'official'):
    for b in BL:
        if b != 'np+id':
            rows.append(V.pair(R, 'np+id', b, s))
    for k in range(4):
        rows.append(dict(V.pair(R, 'np+id', 'npid+full', s, sel=lambda g, k=k: g.fold == k), set=f'{s}/f{k}'))
T = pd.DataFrame(rows); T.to_csv(f'{V.RES}/fulloof_table.csv', index=False)
for c in rows:
    print(V.fmt(c))
lab.log('DONE')
