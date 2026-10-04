"""early track: IN-DISTRIBUTION downside of the early-month prior designs (required check), model-based.
Target = held-out TRAIN days (fold 0), official-like onset windows (522, RAW histories), fold-safe CatBoost (folds 1-3) and fold-safe
prior.  'Scenario evidence' = the previous days of the SAME train scenario with 34% of the observed cells hidden (emulates the
released masked layer).  The train prior is already the right prior here, so any change is pure noise cost.
  E1  evidence = the n most recent days (n = 1, 2, 3, 4, all earlier fold-0 days), nothing else hidden
  E2  faithful month-start replay (official greedy restarted at d0, 18 slots hidden after each pick); Saturday starts = public calendar."""
import sys, time
import numpy as np, pandas as pd
import lab, early_lib as el, early_engine as en

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 500)
cat = el.load_cat(f'{el.OUT}/cat_onset_noramp_f0.cbm')
B = [el.train_bundle(p) for p in lab.PANELS]
M = dict(box=(-6, 6), form='bma', mu=0.05, nu=1.0, cw=1.0)
designs = {'sp4': dict(box=(-2, 2), ref='base', k=4, cw=0.0, ncount='obs'), 'sp8': dict(box=(-2, 2), ref='base', k=8, cw=0.0, ncount='obs'),
           'blend_w24': dict(box=(-24, 24), ref='base', k=4, cw=1.0, ncount='obs'),
           'mix8': dict(M, pi=0.8), 'mix9': dict(M, pi=0.9), 'mix95': dict(M, pi=0.95), 'mix98': dict(M, pi=0.98),
           'mix9_pos': dict(M, pi=0.9, direction='pos'), 'mix9_neg': dict(M, pi=0.9, direction='neg')}
ds = list(designs)
allR = []
for n in (1, 2, 3, 4, 99):
    R = en.run(cat, B, designs, lambda b: en.items_recent(b, n), f'recent{n}'); R['nrec'] = n; allR.append(R)
R1 = pd.concat(allR, ignore_index=True); R1.to_pickle(f'{el.OUT}/indist_e1.pkl')
R2 = en.run(cat, B, designs, lambda b: en.items_replay(b, hide=18), 'replay18'); R2.to_pickle(f'{el.OUT}/indist_e2h18.pkl')
print('######## IN-DISTRIBUTION (train fold 0, fold-safe model + prior, RAW histories; evidence = previous train days, 34% hidden)')
for n in (1, 2, 3, 4, 99):
    print(f'\n-- E1 n = {n if n < 99 else "all earlier fold-0 days"}')
    print(en.table(R1[R1.nrec == n], ref='base', ref2='sp4', designs=['base'] + ds).round(4).to_string(index=False))
    if n == 2:
        print(en.panel_table(R1[R1.nrec == 2], ['base'] + ds).round(3).to_string())
for n in (1, 2, 3, 4):
    print(f'\n-- E2 faithful replay (all starts), n = d - d0 = {n}')
    print(en.table(R2[R2.n == n], ref='base', ref2='sp4', designs=['base'] + ds).round(4).to_string(index=False))
dow0 = {b['panel']: b['dow'] for b in B}
sat = R2[[dow0[p][d0] == 5 for p, d0 in zip(R2.panel, R2.d0)]]
print('\n######## E2 faithful replay, SATURDAY starts (public calendar) on train fold 0; items by n:', sat[sat.design == 'base'].groupby('n').size().to_dict())
print(en.table(sat, ref='base', ref2='sp4', designs=['base'] + ds).round(4).to_string(index=False))
print(en.panel_table(sat, ['base'] + ds).round(3).to_string())
w = {0: 8, 1: 8, 2: 9, 3: 8, 4: 7}
print('\n-- Saturday replay re-weighted to the official composition (8/8/9/8/7 windows at n = 0/1/2/3/>=4), plain window means')
for name in ['base'] + ds:
    y = sat[sat.design == name]; g = y.groupby(np.minimum(y.n, 4)).iou.mean()
    print(f'   {name:10s} {sum(w[k] * g[k] for k in w) / 40:.4f}')
lab.log('DONE')
