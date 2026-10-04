"""TRACK ogmodel: run one model STRUCTURE with the F03 recipe (OOF folds 0..3 + final models k = 4).
Usage (through run_heavy):
    ogmodel_run.py <STRUCT[,STRUCT...]> [cols=np] [leaves=255] [lr=0.05] [n_trees=500] [folds=01234] [seed=0] [tag]
Arm name: og_<STRUCT>_<cols><leaves>[_s<seed>][_<tag>]   (cache: data/t2local/cache/ogmodel/oof/<arm>, models/<arm>__g*.txt)
Reference = lab arm fc_F03_<cols>255 (same rows, weights, parameters; `joint` reproduces it)."""
import sys
import pandas as pd
import lab
import ogmodel_lib as og

structs = sys.argv[1].split(',')
cols = sys.argv[2] if len(sys.argv) > 2 else 'np'
leaves = int(sys.argv[3]) if len(sys.argv) > 3 else 255
lr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
n_trees = int(sys.argv[5]) if len(sys.argv) > 5 else 500
folds = tuple(int(c) for c in sys.argv[6]) if len(sys.argv) > 6 else (0, 1, 2, 3, 4)
seed = int(sys.argv[7]) if len(sys.argv) > 7 else 0
tag = ('_' + sys.argv[8]) if len(sys.argv) > 8 else ''
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50)
ref = lab.load_arm(f'fc_F03_{cols}255')
for st in structs:
    arm = f'og_{st}_{cols}{leaves}' + (f'_s{seed}' if seed else '') + tag
    lab.log(f'=== {arm}: struct {st} {og.STRUCTS[st]}, cols {cols}, leaves {leaves}, lr {lr}, trees {n_trees}, folds {folds}, seed {seed}')
    df = og.run_struct(arm, st, cols=cols, leaves=leaves, lr=lr, n_trees=n_trees, folds=folds, seed=seed)
    if not len(df):
        continue
    print(f'\n== {arm} (folds on disk: {sorted(df.fold.unique().tolist())}) ==')
    print(lab.summary(df).round(4).to_string(index=False), flush=True)
    rows = [og.delta_row(ref[ref.set == s], df[df.set == s], arm, s) for s in og.EVALS]
    print(f'paired vs fc_F03_{cols}255 (common windows):')
    print(pd.DataFrame(rows).round(4).to_string(index=False), flush=True)
lab.log('DONE', structs)
