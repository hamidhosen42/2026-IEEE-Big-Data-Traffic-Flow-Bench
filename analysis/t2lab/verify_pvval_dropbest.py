"""verify_pvval_dropbest.py -- light: per-panel deltas and drop-best-panel / drop-best-family checks on the CLAIMED pm_scores.pkl."""
import numpy as np, pandas as pd
import lab
R = pd.read_pickle(f'{lab.ROOT}/analysis/t2lab/results/pvval/pm_scores.pkl')
R = R[R.d >= 15]
for s in ('offlike_m', 'runstart_m'):
    g = R[R.set == s]
    for a, b in (('R:np+id', 'RB:npid+full'), ('R:npid+full_s4', 'RB:npid+full'), ('R:np+id', 'RB:np+id'), ('M:npid+full_s4', 'MB11:npid+full')):
        A = g[g.arm == a]; B = g[g.arm == b]
        per = (B.groupby('panel').iou.mean() - A.groupby('panel').iou.mean()).sort_values()
        bt = lab.paired_bootstrap(A, B)
        best = per.index[-1]; fam_best = best.rsplit('_', 1)[0]
        bt2 = lab.paired_bootstrap(A[A.panel != best], B[B.panel != best])
        bt3 = lab.paired_bootstrap(A[A.family != fam_best], B[B.family != fam_best])
        n = A.groupby('panel').size()
        print(f'{s} {b} - {a}: {bt["diff"]:+.4f} [{bt["lo"]:+.4f},{bt["hi"]:+.4f}] | drop {best} {bt2["diff"]:+.4f} [{bt2["lo"]:+.4f},{bt2["hi"]:+.4f}]'
              f' | drop family {fam_best} {bt3["diff"]:+.4f} [{bt3["lo"]:+.4f},{bt3["hi"]:+.4f}]')
        print('   per panel:', ' '.join(f'{p}:{v:+.3f}(n={n[p]})' for p, v in per.items()))
