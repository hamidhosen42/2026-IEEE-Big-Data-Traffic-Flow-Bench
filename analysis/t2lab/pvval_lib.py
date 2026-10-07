"""Track pvval: shared helpers.  Reuses pvog_lib (imported, never edited) and redirects its output / cache directories to
results/pvval and data/t2local/cache/pvval, so nothing of the earlier tracks is overwritten."""
import os
import numpy as np
import pandas as pd
import lab
import pvog_lib as pv

RES = f'{lab.ROOT}/analysis/t2lab/results/pvval'
CACHE = f'{lab.CACHE}/pvval'
for _d in (RES, CACHE, f'{RES}/probs', f'{RES}/models'):
    os.makedirs(_d, exist_ok=True)
pv.RES = RES                 # save_probs / load_probs / model files
pv.CACHE = CACHE             # masked F03 row cache, regime shares


def march_cached(cut=15, wm=5.0):
    """pv.march_data(cut, wm, eval_from=0) cached on disk (TR = March training rows of days < cut, EV = pseudo-March blocks)."""
    f = f'{CACHE}/march_{cut}_{wm:g}.pkl'
    if os.path.exists(f):
        return pd.read_pickle(f)
    TR, EV = pv.march_data(cut, wm, eval_from=0, verbose=True)
    tmp = f'{f}.tmp{os.getpid()}'
    pd.to_pickle((TR, EV), tmp); os.replace(tmp, f)
    return TR, EV


def pair(R, a, b, sname, min_d=None, max_d=None, sel=None):
    """paired bootstrap B - A on one evaluation set (lab.paired_bootstrap) + per-panel sign count."""
    g = R[R.set == sname]
    if min_d is not None:
        g = g[g.d >= min_d]
    if max_d is not None:
        g = g[g.d <= max_d]
    if sel is not None:
        g = g[sel(g)]
    A = g[g.arm == a]; B = g[g.arm == b]
    bt = lab.paired_bootstrap(A, B)
    pa = A.groupby('panel').iou.mean(); pb = B.groupby('panel').iou.mean(); dl = pb - pa
    return dict(set=sname, B=b, A=a, n=bt['n_windows'], iou_A=lab.aggregate(A), iou_B=lab.aggregate(B), diff=bt['diff'], lo=bt['lo'],
                hi=bt['hi'], pos=int((dl > 1e-9).sum()), neg=int((dl < -1e-9).sum()), panels=len(dl))


def fmt(c):
    return (f"{c['set']:>10s} {c['B']:>22s} - {c['A']:<22s} n={c['n']:<5d} {c['iou_B']:.4f} vs {c['iou_A']:.4f}  "
            f"{c['diff']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]  panels +{c['pos']}/-{c['neg']} of {c['panels']}")
