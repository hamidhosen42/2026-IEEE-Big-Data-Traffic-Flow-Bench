"""early track, stage B grid on E1: the per-cell mixture design ('bma').
  H0 the scenario follows the train prior, H1 new share theta ~ Beta(mu nu, (1 - mu) nu); w1 = P(H1 | evidence days);
  new = clip(base + w1 (theta_hat - train reference of the box)).   Grid: box x pi x mu x nu x cw (class weight).
Reference designs: sp4 (production), sp8, and the stage A leaders."""
import sys, time
import numpy as np, pandas as pd
import lab, early_lib as el, early_engine as en

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 500)
cat = el.load_cat()
B = [el.march_bundle(p) for p in lab.PANELS]
BOXES = {'w2': (-2, 2), 'w6': (-6, 6), 'w12': (-12, 12), 'w24': (-24, 24)}
designs = {'sp4': dict(box=(-2, 2), ref='base', k=4, cw=0.0, ncount='obs'), 'sp8': dict(box=(-2, 2), ref='base', k=8, cw=0.0, ncount='obs'),
           'A_w24p_k4': dict(box=(-24, 24), ref='base', k=4, cw=1.0, ncount='obs'),
           'A_w12p_k2': dict(box=(-12, 12), ref='base', k=2, cw=1.0, ncount='obs'),
           'A_r24_k4': dict(box=(-24, 24), ref='box', k=4, cw=0.5, ncount='obs')}
for bn, box in BOXES.items():
    for pi in (0.5, 0.8):
        for mu in (0.05, 0.2):
            for nu in (1.0, 2.0):
                for cw in (0.0, 0.5, 1.0):
                    designs[f'bma|{bn}|pi{pi}|mu{mu}|nu{nu}|cw{cw}'] = dict(box=box, form='bma', pi=pi, mu=mu, nu=nu, cw=cw)
print(len(designs), 'designs')
allR = []
for n in (1, 2, 3, 4, 99):
    t0 = time.time()
    R = en.run(cat, B, designs, lambda b: en.items_recent(b, n), f'recent{n}')
    R['nrec'] = n; allR.append(R)
    lab.log(f'n={n} done {time.time() - t0:.0f}s')
R = pd.concat(allR, ignore_index=True)
R.to_pickle(f'{el.OUT}/scanB.pkl')
agg = R.groupby(['design', 'nrec'])[['family', 'panel', 'iou']].apply(lab.aggregate).unstack('nrec')
base = agg.loc['base']
D = (agg - base).drop(index='base')
D['early'] = D[[1, 2, 3, 4]].mean(1)
print('\n== reference designs =='); print(D[~D.index.str.startswith('bma')].round(4).to_string())
Db = D[D.index.str.startswith('bma')].copy()
parts = Db.index.to_series().str.split('|', expand=True).iloc[:, 1:]; parts.columns = ['box', 'pi', 'mu', 'nu', 'cw']
Db = pd.concat([parts, Db], axis=1)
Db.to_csv(f'{el.OUT}/scanB_deltas.csv')
print('\n== marginal means (bma) ==')
for c in ('box', 'pi', 'mu', 'nu', 'cw'):
    print(Db.groupby(c)[[1, 2, 3, 4, 'early', 99]].mean().round(4).to_string()); print()
print('== early delta: rows box|cw, columns pi|mu|nu ==')
print(Db.pivot_table(index=['box', 'cw'], columns=['pi', 'mu', 'nu'], values='early').round(4).to_string())
print('\n== all-days delta: rows box|cw, columns pi|mu|nu ==')
print(Db.pivot_table(index=['box', 'cw'], columns=['pi', 'mu', 'nu'], values=99).round(4).to_string())
print('\n== top 20 bma by early delta =='); print(Db.sort_values('early', ascending=False).head(20).round(4).to_string())
lab.log('DONE')
