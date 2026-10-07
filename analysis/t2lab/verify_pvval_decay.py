"""verify_pvval_decay.py -- light: does the March-row gain decay with the distance between the March training days (< CUT) and the
evaluation day?  Claimed table (results/pvval/pm_scores.pkl, CUT 15) and, if given, a verify_pvval_og scores file."""
import sys
import numpy as np, pandas as pd
import lab
def split_report(R, a, b, ranges, label):
    for s in ('offlike_m', 'runstart_m'):
        g = R[R.set == s]
        for lo, hi in ranges:
            gg = g[(g.d >= lo) & (g.d <= hi)]
            A = gg[gg.arm == a][['set', 'panel', 'family', 'd', 't', 'cond', 'iou']] if 'cond' in gg else gg[gg.arm == a].assign(cond='og')
            B = gg[gg.arm == b][['set', 'panel', 'family', 'd', 't', 'cond', 'iou']] if 'cond' in gg else gg[gg.arm == b].assign(cond='og')
            if len(A) < 10:
                continue
            bt = lab.paired_bootstrap(A, B, n=1000)
            print(f'{label} {s} days {lo}-{hi} {b} - {a}: {bt["diff"]:+.4f} [{bt["lo"]:+.4f}, {bt["hi"]:+.4f}] n={bt["n_windows"]}')
R = pd.read_pickle(f'{lab.ROOT}/analysis/t2lab/results/pvval/pm_scores.pkl')
for a, b in (('R:np+id', 'RB:npid+full'), ('R:npid+full_s4', 'RB:npid+full'), ('R:np+id', 'RB:np+id')):
    split_report(R, a, b, ((15, 22), (23, 30)), 'claimed')
for f in sys.argv[1:]:
    S = pd.read_pickle(f); cut = int(S.d.min())
    mid = (cut + 30) // 2
    for a, b in (('A', 'B'), ('A_s4', 'B'), ('A1', 'B'), ('A1', 'B_npid')):
        if a in set(S.arm) and b in set(S.arm):
            split_report(S, a, b, ((cut, mid), (mid + 1, 30)), f.rsplit('/', 1)[-1])
