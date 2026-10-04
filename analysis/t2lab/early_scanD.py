"""early track, stage D: one-at-a-time variations around the mixture centre designs (box +-6, mu 0.05, nu 1, classes pooled,
eps 0.02, all three prior columns, both directions, no spatial smoothing, no same-day evidence), pi = 0.8 and 0.9:
  eps, class weight cw, spatial smoothing +-1 link, direction (pos = site discovery only, neg = anti-prior only),
  column subsets, same-day evidence (slots < T).   E1 recent-n + E2 replay (all starts by n; Saturday starts)."""
import sys, time
import numpy as np, pandas as pd
import lab, early_lib as el, early_engine as en

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 500)
cat = el.load_cat()
B = [el.march_bundle(p) for p in lab.PANELS]
designs = {'sp4': dict(box=(-2, 2), ref='base', k=4, cw=0.0, ncount='obs')}
for pi in (0.8, 0.9):
    c0 = dict(box=(-6, 6), form='bma', pi=pi, mu=0.05, nu=1.0, cw=1.0)
    V = {'C0': {}, 'eps.005': dict(eps=0.005), 'eps.01': dict(eps=0.01), 'eps.05': dict(eps=0.05), 'cw0': dict(cw=0.0), 'cw.5': dict(cw=0.5),
         'cwall0': dict(cw_all=0.0), 'sp1': dict(sp=1), 'pos': dict(direction='pos'), 'neg': dict(direction='neg'),
         'c_prior': dict(cols='prior'), 'c_prior_all': dict(cols='prior_all'), 'c_prior_now': dict(cols='prior_now'), 'c_now': dict(cols='now'),
         'c_alld': dict(cols='alld'), 'sameday': dict(sameday=True), 'sameday_w24': dict(sameday=True, box=(-24, 24)),
         'sameday_day': dict(sameday=True, box=(-288, 288)), 'box_f6': dict(box=(-2, 6)), 'box_f12': dict(box=(-2, 12)), 'box_b6': dict(box=(-6, 2))}
    for vn, kw in V.items():
        designs[f'pi{pi}|{vn}'] = dict(c0, **kw)
print(len(designs), 'designs')
allR = []
for n in (1, 2, 3, 4, 99):
    R = en.run(cat, B, designs, lambda b: en.items_recent(b, n), f'recent{n}'); R['nrec'] = n; allR.append(R)
R1 = pd.concat(allR, ignore_index=True); R1.to_pickle(f'{el.OUT}/scanD_e1.pkl')
R2 = en.run(cat, B, designs, lambda b: en.items_replay(b), 'replay'); R2.to_pickle(f'{el.OUT}/scanD_e2.pkl')
agg = R1.groupby(['design', 'nrec'])[['family', 'panel', 'iou']].apply(lab.aggregate).unstack('nrec')
D = (agg - agg.loc['base']).drop(index='base'); D.columns = [f'e1_n{c}' for c in D.columns]
D['e1_early'] = D[['e1_n1', 'e1_n2', 'e1_n3', 'e1_n4']].mean(1)
x = R2[R2.n.between(0, 4)]
a2 = x.groupby(['design', 'n'])[['family', 'panel', 'iou']].apply(lab.aggregate).unstack('n')
D2 = (a2 - a2.loc['base']).drop(index='base'); D2.columns = [f'e2_n{c}' for c in D2.columns]
D2['e2_early'] = D2[['e2_n1', 'e2_n2', 'e2_n3', 'e2_n4']].mean(1)
sat = R2[R2.d0.isin([7, 14, 21, 28])]
a3 = sat.groupby('design')[['family', 'panel', 'iou']].apply(lab.aggregate)
D3 = (a3 - a3['base']).drop(index='base').rename('e2_sat')
T = pd.concat([D, D2, D3], axis=1)
T.to_csv(f'{el.OUT}/scanD_deltas.csv')
print(T.loc[list(designs)].round(4).to_string())
lab.log('DONE')
