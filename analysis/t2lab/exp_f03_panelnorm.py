"""F03 with per-panel weight normalisation (F03N): the metric weighs panels (within families) equally, but the pooled F03 model
sees ~3x more rows from big panels (D7_I405_N: 352 links) than from small ones (D7_I210_W: 81).  F03N rescales the window
weights of each training set so that every panel carries the same total (window weight x links).  Same folds / evals as the
firstcross arms; compared with fc_F03_np255 (paired bootstrap).  Usage: exp_f03_panelnorm.py [cols=np]"""
import sys
import numpy as np, pandas as pd
import lab
import exp_firstcross_lib as fc

cols = sys.argv[1] if len(sys.argv) > 1 else 'np'
L = {p: len(np.load(f'{lab.DATA}/t2data_{p}.npz')['links']) for p in lab.PANELS} if hasattr(lab, 'DATA') else None
if L is None:
    import glob, os
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 't2local')
    L = {p: len(np.load(f'{base}/t2data_{p}.npz')['links']) for p in lab.PANELS}
f03 = fc.wfn_F(0.3)
tot = {}                                                    # unnormalised total (sum w * L) per set and panel
for s in ('faith3', 'runstart'):
    for p in lab.PANELS:
        win = lab.window_set(p, s)
        win = win[win.cond == lab.ONGOING].reset_index(drop=True) if 'cond' in win else win
        tot[(s, p)] = float(f03(win, s, p).sum() * L[p])
target = {s: np.mean([tot[(s, p)] for p in lab.PANELS]) for s in ('faith3', 'runstart')}
print('links per panel', L)
print('per-panel scale factors:', {s: {p: round(target[s] / max(tot[(s, p)], 1e-9), 3) for p in lab.PANELS} for s in target})


def wfn(win, set_name, panel):
    w = f03(win, set_name, panel)
    return (w * (target[set_name] / max(tot[(set_name, panel)], 1e-9))).astype(np.float32)


fc.ARMS['F03N'] = dict(train=[('faith3', 1.0), ('runstart', 1.0)], wfn=wfn)
name = f'fc_F03N_{cols}255'
df = fc.run_arm(name, fc.ARMS['F03N'], cols=cols, leaves=255, lr=0.05, n_trees=500)
base = lab.load_arm(f'fc_F03_{cols}255')
pd.set_option('display.width', 220)
print(lab.summary(df).round(4).to_string(index=False))
for s in ('official', 'offlike', 'pseudo'):
    a, b = base[base.set == s], df[df.set == s]
    print(s, 'F03N - F03:', lab.paired_bootstrap(a, b))
    pp = pd.DataFrame({'F03': a.groupby('panel').iou.mean(), 'F03N': b.groupby('panel').iou.mean()}); pp['d'] = pp.F03N - pp.F03
    print(pp.round(4).to_string())
