"""Arm A = current production ongoing recipe at local scale: prior-free 'np' columns (ramp features included), LightGBM
255 leaves / lr 0.05 / min_data_in_leaf 40 / 500 trees, dense production windows capped at 1500 ongoing per panel.
OOF over the 4 contiguous day blocks; evaluated on the official 40, official-like faithful windows and current-style pseudo windows.
Usage (through run_heavy): baseline_A.py [arm_name] [n_trees] [num_leaves] [folds e.g. 0123] [learning_rate]"""
import sys, numpy as np, pandas as pd, lab
arm = sys.argv[1] if len(sys.argv) > 1 else 'A_np255'
n_trees = int(sys.argv[2]) if len(sys.argv) > 2 else 500
leaves = int(sys.argv[3]) if len(sys.argv) > 3 else 255
folds = tuple(int(c) for c in sys.argv[4]) if len(sys.argv) > 4 else (0, 1, 2, 3)
lr = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50)
df = lab.run_arm(arm, train='dense1500', evals=('official', 'offlike', 'pseudo'), cols='np',
                 params=lab.lgb_params(num_leaves=leaves, learning_rate=lr, min_data_in_leaf=40), n_trees=n_trees, folds=folds)
print('\n== arm', arm, '(ongoing, OOF) =='); print(lab.summary(df).round(4).to_string(index=False))
off = df[df.set == 'official']
print('\nofficial-40 per panel:'); print(off.groupby('panel').iou.agg(['mean', 'count']).round(4).T.to_string())
print('official-40 plain mean', round(off.iou.mean(), 4), 'aggregated', round(lab.aggregate(off), 4))
ol = df[df.set == 'offlike']
print('\nofficial-like per fold:', {int(k): round(lab.aggregate(g), 4) for k, g in ol.groupby('fold')})
print('official-like run-starts only:', round(lab.aggregate(ol[ol.runstart]), 4), 'n', int(ol.runstart.sum()))
ps = df[df.set == 'pseudo']
print('pseudo per fold:', {int(k): round(lab.aggregate(g), 4) for k, g in ps.groupby('fold')})
print('pseudo faithful subset:', round(lab.aggregate(ps[ps.faithful]), 4), 'n', int(ps.faithful.sum()), '| stable (pers_fill > 0.9):',
      round(lab.aggregate(ps[ps.pers_fill > 0.9]), 4), 'n', int((ps.pers_fill > 0.9).sum()))
# persistence reference (qlast repeated, production last-valid fill incl. simulated slot-T)
ref = []
for s in ('official', 'offlike', 'pseudo'):
    for panel in lab.PANELS:
        rc = lab.rows(panel, s)
        if len(rc.win):
            ref.append(lab.score_probs(rc, lab.persistence_probs(rc), arm='persist_qlast', set_name=s))
ref = pd.concat(ref, ignore_index=True)
print('\n== persistence reference =='); print(lab.summary(ref).round(4).to_string(index=False))
