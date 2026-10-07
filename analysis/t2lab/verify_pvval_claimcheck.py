"""verify_pvval_claimcheck.py -- light (no training): re-reads the pvon pickles behind the pvval onset claim.
Recomputes bag - A_s4, per-panel gains, drop-best-panel / drop-best-two, window concentration, B_bag vs B_bagL identity,
the matched-regime control, and the harm / shift rows that feed the expected private delta."""
import numpy as np
import pandas as pd
import lab

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
R1 = pd.read_pickle('results/pvon/step2_cut15_s1_S.pkl')
KEY = ['panel', 'd', 't']


def arm(a, R=R1):
    return R[R.arm == a].copy()


def pair(A, B, drop=()):
    A = A[~A.panel.isin(drop)]; B = B[~B.panel.isin(drop)]
    bt = lab.paired_bootstrap(A, B)
    dl = B.groupby('panel').iou.mean() - A.groupby('panel').iou.mean()
    return dict(diff=round(bt['diff'], 4), lo=round(bt['lo'], 4), hi=round(bt['hi'], 4), n=bt['n_windows'],
                pos=int((dl > 1e-9).sum()), neg=int((dl < -1e-9).sum()), drop=','.join(drop))


A, B = arm('A_s4'), arm('B_bag')
m = A.set_index(KEY)[['iou']].join(B.set_index(KEY)[['iou']], rsuffix='_b')
m['dd'] = m.iou_b - m.iou
print('n windows', len(m), '| windows changed', int((m.dd.abs() > 1e-9).sum()), '| better', int((m.dd > 1e-9).sum()),
      '| worse', int((m.dd < -1e-9).sum()))
dl = m.groupby(level=0).dd.mean().sort_values(ascending=False)
print('per-panel mean diff (B_bag - A_s4):', dl.round(4).to_dict())
print('per-panel n windows:', m.groupby(level=0).size().to_dict())
print('top-10 window diffs:', m.dd.sort_values(ascending=False).head(10).round(3).tolist())
print('sum of all window diffs', round(m.dd.sum(), 3), '| sum of top-5', round(m.dd.sort_values(ascending=False).head(5).sum(), 3))
print('claimed:', pair(A, B))
print('drop best panel:', pair(A, B, (dl.index[0],)))
print('drop best two  :', pair(A, B, tuple(dl.index[:2])))
# B_bag vs B_bagL: identical per window?
bl = arm('B_bagL').set_index(KEY).iou; bb = B.set_index(KEY).iou
print('B_bagL == B_bag per window:', bool((bl - bb).abs().max() < 1e-12), '| B_w5 vs B_w40 identical windows:',
      float(((arm('B_w5').set_index(KEY).iou - arm('B_w40').set_index(KEY).iou).abs() < 1e-12).mean()))
# single-weight arms vs A_s4 (variants tried on this same set)
for a in ('B_w2', 'B_w5', 'B_w10', 'B_w20', 'B_w40', 'B_bag', 'B_bagL'):
    print(f'{a:8s}', pair(A, arm(a)))
# matched regime (seed 1 only here)
print('AmS_all - A_s4      ', pair(A, arm('AmS_all')))
print('BmS_all_bag - AmS   ', pair(arm('AmS_all'), arm('BmS_all_bag')))
# early-days slice of the evaluation (days 15-20, closer in time to the March training days) vs late days (>= 21)
for lo_, hi_ in ((15, 20), (21, 31)):
    s = lambda X: X[(X.d >= lo_) & (X.d <= hi_)]
    print(f'days {lo_}-{hi_}:', pair(s(A), s(B)))
