"""early track, stage E: FAITHFUL month-start replay.  The released masked layer hides T+1..T+18 (90 min) around every official
window (measured on the 80 official validation windows: visible share 0.00 at offsets 1..18, 0.76 at -12..-1, 0.52 elsewhere), so
the replay hides 18 slots after every pick (onset + ongoing) of the replay.  Boxes reaching beyond the hidden block are added.
Evaluated on all starts (by n) and on the Saturday starts (public calendar)."""
import sys, time
import numpy as np, pandas as pd
import lab, early_lib as el, early_engine as en

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 500)
cat = el.load_cat()
B = [el.march_bundle(p) for p in lab.PANELS]
BOXES = {'w6': (-6, 6), 'w12': (-12, 12), 'w24': (-24, 24), 'f12': (-2, 12), 'f24': (-2, 24), 'f36': (-2, 36), 'g24': (-6, 24), 'g36': (-12, 36)}
designs = {'sp4': dict(box=(-2, 2), ref='base', k=4, cw=0.0, ncount='obs'), 'sp8': dict(box=(-2, 2), ref='base', k=8, cw=0.0, ncount='obs'),
           'sp2': dict(box=(-2, 2), ref='base', k=2, cw=0.0, ncount='obs'),
           'blend_w24': dict(box=(-24, 24), ref='base', k=4, cw=1.0, ncount='obs'),
           'blend_f24': dict(box=(-2, 24), ref='base', k=4, cw=1.0, ncount='obs'),
           'resid_w24': dict(box=(-24, 24), ref='box', k=4, cw=0.5, ncount='obs')}
for bn, box in BOXES.items():
    for pi in (0.8, 0.9, 0.95):
        designs[f'mix|{bn}|pi{pi}'] = dict(box=box, form='bma', pi=pi, mu=0.05, nu=1.0, cw=1.0)
print(len(designs), 'designs')
t0 = time.time()
R2 = en.run(cat, B, designs, lambda b: en.items_replay(b, hide=18), 'replay18'); R2.to_pickle(f'{el.OUT}/scanE_e2h18.pkl')
lab.log(f'replay (hide 18) done {time.time() - t0:.0f}s items', int((R2.design == 'base').sum()))
x = R2[R2.n.between(1, 4)]
a2 = x.groupby(['design', 'n'])[['family', 'panel', 'iou']].apply(lab.aggregate).unstack('n')
D2 = (a2 - a2.loc['base']).drop(index='base'); D2.columns = [f'n{c}' for c in D2.columns]; D2['early'] = D2.mean(1)
sat = R2[R2.d0.isin([7, 14, 21, 28])]
a3 = sat.groupby('design')[['family', 'panel', 'iou']].apply(lab.aggregate)
T = pd.concat([D2, (a3 - a3['base']).drop(index='base').rename('sat')], axis=1)
T.to_csv(f'{el.OUT}/scanE_deltas.csv')
print('base by n (all starts):', a2.loc['base'].round(4).to_dict(), '| Saturday starts base', round(float(a3['base']), 4), 'items', int((sat.design == 'base').sum()))
print(T.loc[list(designs)].round(4).to_string())
names = ['sp4', 'sp8', 'blend_w24', 'mix|w6|pi0.9', 'mix|w12|pi0.9', 'mix|w24|pi0.9', 'mix|f24|pi0.9', 'mix|g24|pi0.9', 'mix|g36|pi0.9']
print('\n######## Saturday starts, hide 18: bootstrap vs base and vs sp4')
print(en.table(sat, ref='base', ref2='sp4', designs=['base'] + names).round(4).to_string(index=False))
print(en.panel_table(sat, ['base'] + names).round(3).to_string())
for n in (1, 2, 3, 4):
    print(f'\n-- all starts, n = {n}')
    print(en.table(R2[R2.n == n], ref='base', ref2='sp4', designs=['base'] + names).round(4).to_string(index=False))
lab.log('DONE')
