"""TRACK ogcap: one (structure, fold, seed) job with the F03 recipe at production-like capacity.
Usage (through run_heavy):  ogcap_run.py <struct> <folds e.g. 0 | 02 | 4> [cols=np] [leaves=511] [lr=0.05] [n_trees=900] [seed=0]
k = 0..3: OOF fold (scored on official / offlike / pseudo); k = 4: final models on all train days (pseudo-March).
Arm name: cap_<struct>_<cols><leaves>_l<lr*100>_t<n_trees>[_s<seed>]   (data/t2local/cache/ogcap/oof/<arm>/fold<k>.pkl)"""
import sys
import lab
import ogcap_lib as oc

struct = sys.argv[1]
folds = [int(c) for c in sys.argv[2]]
cols = sys.argv[3] if len(sys.argv) > 3 else 'np'
leaves = int(sys.argv[4]) if len(sys.argv) > 4 else 511
lr = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
n_trees = int(sys.argv[6]) if len(sys.argv) > 6 else 900
seed = int(sys.argv[7]) if len(sys.argv) > 7 else 0
lab.log(f'ogcap_run {struct} folds {folds} cols {cols} leaves {leaves} lr {lr} trees {n_trees} seed {seed} threads {lab.THREADS}')
for k in folds:
    oc.run_fold(struct, k, cols=cols, leaves=leaves, lr=lr, n_trees=n_trees, seed=seed)
lab.log('DONE')
