"""pvog EXP 4a (light: prediction only, no training): the TRAIN-ONLY members (final F03 np / id / full models of
results/firstcross/final, trained on raw train rows) on every pseudo-March window (all days: no March label is used by these
models; the blended prior of a window uses only the March days before its day).  K sweep of the prior blend and day bins."""
import numpy as np
import pandas as pd
import lightgbm as lgb
import lab
import pvog_lib as pv

m_np = lgb.Booster(model_file=f'{pv.MD_FINAL}/model_F03_np255_500.txt')
m_id = lgb.Booster(model_file=f'{pv.MD_FINAL}/model_F03_id255_500.txt')
m_full = lgb.Booster(model_file=f'{pv.MD_FINAL}/model_F03_full255_500.txt')
_, EV = pv.march_data(15, 5.0, eval_from=0, verbose=False)
members = {'R_np': (m_np, 'np', False), 'R_id': (m_id, 'id', False), 'R_full_s4': (m_full, 'full', True), 'R_full_tr': (m_full, 'full', False)}
P6 = pv.predict_members(EV, members)
pv.save_probs('R0', P6)
ARMS = {'R:np+id': {'R_np': 0.5, 'R_id': 0.5}, 'R:npid+full_s4': {'R_np': 0.25, 'R_id': 0.25, 'R_full_s4': 0.5},
        'R:np+id+full_s4/3': {'R_np': 1 / 3, 'R_id': 1 / 3, 'R_full_s4': 1 / 3}, 'R:npid+full_tr': {'R_np': 0.25, 'R_id': 0.25, 'R_full_tr': 0.5},
        'R:npid75+full_s4_25': {'R_np': 0.375, 'R_id': 0.375, 'R_full_s4': 0.25}, 'R:full_s4': {'R_full_s4': 1.0}, 'R:full_tr': {'R_full_tr': 1.0}}
R = pd.concat([pv.score_members(EV, P6, w, nm) for nm, w in ARMS.items()], ignore_index=True)
R.to_pickle(f'{pv.RES}/exp4a_trainonly_pm.pkl')
rows = []
for sname in ('offlike_m', 'runstart_m'):
    for lo, hi in ((0, 30), (0, 4), (5, 14), (15, 30), (15, 22), (23, 30)):
        g = R[(R.set == sname) & (R.d >= lo) & (R.d <= hi)]
        A = g[g.arm == 'R:np+id']
        for b in ARMS:
            if b == 'R:np+id':
                continue
            B = g[g.arm == b]; bt = lab.paired_bootstrap(A, B)
            pa = A.groupby('panel').iou.mean(); pb = B.groupby('panel').iou.mean()
            rows.append(dict(set=sname, days=f'{lo + 1}-{hi + 1}', arm=b, n=bt['n_windows'], iou_base=lab.aggregate(A), iou=lab.aggregate(B), diff=bt['diff'], lo=bt['lo'], hi=bt['hi'],
                             panels_pos=f'{int((pb - pa > 1e-9).sum())}/{len(pa)}'))
T = pd.DataFrame(rows); T.to_csv(f'{pv.RES}/exp4a_trainonly_pm.csv', index=False)
print(T.round(4).to_string(index=False))
rows = []
for sname in ('offlike_m', 'runstart_m'):
    for rg in (0.5, 0.7, 0.8):
        for lo in (0, 15):
            g = R[(R.set == sname) & (R.regime == rg) & (R.d >= lo)]
            A = g[g.arm == 'R:np+id']
            for b in ('R:npid+full_s4', 'R:npid+full_tr'):
                B = g[g.arm == b]; bt = lab.paired_bootstrap(A, B)
                pa = A.groupby('panel').iou.mean(); pb = B.groupby('panel').iou.mean()
                rows.append(dict(set=sname, regime=rg, days=f'>={lo + 1}', arm=b, n=bt['n_windows'], iou_base=lab.aggregate(A), iou=lab.aggregate(B), diff=bt['diff'], lo=bt['lo'], hi=bt['hi'],
                                 panels_pos=f'{int((pb - pa > 1e-9).sum())}/{len(pa)}'))
T2 = pd.DataFrame(rows); T2.to_csv(f'{pv.RES}/exp4a_trainonly_pm_regime.csv', index=False)
print('\nby day regime of the window (visible share of eligible cells)'); print(T2.round(4).to_string(index=False))
