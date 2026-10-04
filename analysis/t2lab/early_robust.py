"""early track (light): is the mixture gain a single-panel artefact?  Drop-one-panel deltas and bootstrap P(delta > 0)."""
import numpy as np, pandas as pd
import lab, early_lib as el
pd.set_option('display.width', 250)
R1 = pd.read_pickle(f'{el.OUT}/final_e1.pkl'); R2 = pd.read_pickle(f'{el.OUT}/final_e2h18.pkl')
sat = R2[R2.d0.isin([7, 14, 21, 28])]
sets = {'E1 n=2': R1[R1.nrec == 2], 'E1 n=3': R1[R1.nrec == 3], 'E1 n=4': R1[R1.nrec == 4], 'E2 faithful, all starts, n=3': R2[R2.n == 3],
        'E2 faithful, all starts, n=4': R2[R2.n == 4], 'E2 faithful Saturday starts': sat}
for tag, X in sets.items():
    print(f'\n== {tag}')
    for name in ('mix9', 'mix95'):
        for ref in ('base', 'sp4'):
            A = X[X.design == ref]; Bf = X[X.design == name]
            bt = lab.paired_bootstrap(A, Bf)
            pa = A.groupby('panel').iou.mean(); pb = Bf.groupby('panel').iou.mean(); dl = pb - pa
            loo = [float(dl.drop(p).mean()) for p in dl.index]
            print(f'{name:6s} vs {ref:4s}: {bt["diff"]:+.4f} [{bt["lo"]:+.4f}, {bt["hi"]:+.4f}] P(>0)={bt["p_pos"]:.3f} | per-panel deltas '
                  + ' '.join(f'{v:+.3f}' for v in dl.to_numpy()) + f' | drop-one-panel mean: min {min(loo):+.4f} max {max(loo):+.4f}')
