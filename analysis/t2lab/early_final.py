"""early track: FINAL tables of the candidate designs on pseudo-March (new scenario, masked histories, visible labels).
  E1   evidence = the n most recent scenario days, nothing hidden (the powered emulation asked for), n = 1, 2, 3, 4, all
  E2   faithful month-start replay: evidence = days d0..d-1, 18 slots hidden after every pick of the replay (as the released
       masked layer does around official windows); all 31 starts by n, and the Saturday starts = public calendar."""
import sys, time
import numpy as np, pandas as pd
import lab, early_lib as el, early_engine as en

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 500)
cat = el.load_cat()
B = [el.march_bundle(p) for p in lab.PANELS]
M = dict(box=(-6, 6), form='bma', mu=0.05, nu=1.0, cw=1.0)
designs = {'sp4': dict(box=(-2, 2), ref='base', k=4, cw=0.0, ncount='obs'), 'sp8': dict(box=(-2, 2), ref='base', k=8, cw=0.0, ncount='obs'),
           'blend_w24': dict(box=(-24, 24), ref='base', k=4, cw=1.0, ncount='obs'),
           'mix8': dict(M, pi=0.8), 'mix9': dict(M, pi=0.9), 'mix95': dict(M, pi=0.95), 'mix98': dict(M, pi=0.98)}
ds = list(designs)
allR = []
for n in (1, 2, 3, 4, 99):
    R = en.run(cat, B, designs, lambda b: en.items_recent(b, n), f'recent{n}'); R['nrec'] = n; allR.append(R)
R1 = pd.concat(allR, ignore_index=True); R1.to_pickle(f'{el.OUT}/final_e1.pkl')
R2 = en.run(cat, B, designs, lambda b: en.items_replay(b, hide=18), 'replay18'); R2.to_pickle(f'{el.OUT}/final_e2h18.pkl')
print('######## E1 (214 pseudo-March onset windows; evidence = the n most recent scenario days, nothing hidden)')
for n in (1, 2, 3, 4, 99):
    print(f'\n-- n = {n if n < 99 else "all days < d"}')
    print(en.table(R1[R1.nrec == n], ref='base', ref2='sp4', designs=['base'] + ds).round(4).to_string(index=False))
    if n in (2, 99):
        print(en.panel_table(R1[R1.nrec == n], ['base'] + ds).round(3).to_string())
x = R1[(R1.nrec == 99) & (R1.d >= 15)]
print('\n-- APRIL-LIKE regime: all earlier March days, windows on days >= 16 (>= 15 scenario days)')
print(en.table(x, ref='base', ref2='sp4', designs=['base'] + ds).round(4).to_string(index=False))
print(en.panel_table(x, ['base'] + ds).round(3).to_string())
print('\n######## E2 faithful replay (18 slots hidden after every pick), all 31 start days')
for n in (1, 2, 3, 4):
    print(f'\n-- n = d - d0 = {n}')
    print(en.table(R2[R2.n == n], ref='base', ref2='sp4', designs=['base'] + ds).round(4).to_string(index=False))
sat = R2[R2.d0.isin([7, 14, 21, 28])]
print('\n######## E2 faithful replay, SATURDAY starts (d0 = 7, 14, 21, 28) = public calendar; items by n:', sat[sat.design == 'base'].groupby('n').size().to_dict())
print(en.table(sat, ref='base', ref2='sp4', designs=['base'] + ds).round(4).to_string(index=False))
print(en.panel_table(sat, ['base'] + ds).round(3).to_string())
for lab_, m in (('n=0 (Saturday, no evidence)', sat.n == 0), ('n=1 (Sunday <- Saturday)', sat.n == 1), ('n=2 (Monday <- Sat+Sun)', sat.n == 2),
                ('n=3 (Tuesday)', sat.n == 3), ('n>=4 (Wednesday+)', sat.n >= 4)):
    y = sat[m]
    print(f'\n-- Saturday starts, {lab_}: items', int((y.design == 'base').sum()))
    print(en.table(y, ref='base', ref2='sp4', designs=['base'] + ds, boot=False).round(4).to_string(index=False))
# official-composition reweighting of the Saturday replay: 8 / 8 / 9 / 8 / 7 windows at n = 0 / 1 / 2 / 3 / >= 4
w = {0: 8, 1: 8, 2: 9, 3: 8, 4: 7}
print('\n-- Saturday replay re-weighted to the official composition (8/8/9/8/7 windows at n = 0/1/2/3/>=4), plain window means')
for name in ['base'] + ds:
    y = sat[sat.design == name]; g = y.groupby(np.minimum(y.n, 4)).iou.mean()
    print(f'   {name:10s} {sum(w[k] * g[k] for k in w) / 40:.4f}')
lab.log('DONE')
