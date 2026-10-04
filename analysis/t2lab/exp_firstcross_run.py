"""Run firstcross arms.  Usage (through run_heavy):
    exp_firstcross_run.py <ARM[,ARM...]> [cols=np] [leaves=255] [lr=0.05] [n_trees=500] [folds=0123] [seeds=0] [tag]
Arm cache name: fc_<ARM>_<cols><leaves>[_<tag>]   (A at np/255/0.05/500 is the lab's cached A_np255; fc_A_* is a re-run)."""
import sys
import pandas as pd
import lab
import exp_firstcross_lib as fc

arms = sys.argv[1].split(',')
cols = sys.argv[2] if len(sys.argv) > 2 else 'np'
leaves = int(sys.argv[3]) if len(sys.argv) > 3 else 255
lr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
n_trees = int(sys.argv[5]) if len(sys.argv) > 5 else 500
folds = tuple(int(c) for c in sys.argv[6]) if len(sys.argv) > 6 else (0, 1, 2, 3)
seeds = tuple(int(c) for c in sys.argv[7]) if len(sys.argv) > 7 else (0,)
tag = ('_' + sys.argv[8]) if len(sys.argv) > 8 else ''
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50)
for a in arms:
    name = f'fc_{a}_{cols}{leaves}{tag}'
    lab.log(f'=== {name}: cols {cols}, leaves {leaves}, lr {lr}, trees {n_trees}, folds {folds}, seeds {seeds}')
    df = fc.run_arm(name, fc.ARMS[a], cols=cols, leaves=leaves, lr=lr, n_trees=n_trees, folds=folds, seeds=seeds)
    print(f'\n== {name} (folds on disk: {sorted(df.fold.unique().tolist())}) ==')
    print(lab.summary(df).round(4).to_string(index=False), flush=True)
lab.log('DONE', arms)
