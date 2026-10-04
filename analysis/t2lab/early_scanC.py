"""early track, stage C: refined mixture grid (box x pi x mu x nu, eps = 0.02, classes pooled) on
  E1  recent-n (214 windows, n = 1, 2, 3, 4, all)                         -> results/early/scanC_e1.pkl
  E2  month-start replay, every start day (1155 window x start items)    -> results/early/scanC_e2.pkl
      (Saturday starts 7 / 14 / 21 / 28 = the public calendar; evidence = days d0..d-1, official-like horizons hidden)."""
import sys, time
import numpy as np, pandas as pd
import lab, early_lib as el, early_engine as en

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 500)
cat = el.load_cat()
B = [el.march_bundle(p) for p in lab.PANELS]
BOXES = {'w4': (-4, 4), 'w6': (-6, 6), 'w12': (-12, 12)}
designs = {'sp4': dict(box=(-2, 2), ref='base', k=4, cw=0.0, ncount='obs'), 'sp8': dict(box=(-2, 2), ref='base', k=8, cw=0.0, ncount='obs'),
           'A_w24p_k4': dict(box=(-24, 24), ref='base', k=4, cw=1.0, ncount='obs'),
           'A_w12p_k2': dict(box=(-12, 12), ref='base', k=2, cw=1.0, ncount='obs')}
for bn, box in BOXES.items():
    for pi in (0.7, 0.8, 0.9, 0.95):
        for mu in (0.02, 0.05, 0.1):
            for nu in (0.5, 1.0, 2.0):
                designs[f'bma|{bn}|pi{pi}|mu{mu}|nu{nu}'] = dict(box=box, form='bma', pi=pi, mu=mu, nu=nu, cw=1.0)
print(len(designs), 'designs')
allR = []
for n in (1, 2, 3, 4, 99):
    t0 = time.time()
    R = en.run(cat, B, designs, lambda b: en.items_recent(b, n), f'recent{n}')
    R['nrec'] = n; allR.append(R)
    lab.log(f'E1 n={n} done {time.time() - t0:.0f}s')
R1 = pd.concat(allR, ignore_index=True); R1.to_pickle(f'{el.OUT}/scanC_e1.pkl')
t0 = time.time()
R2 = en.run(cat, B, designs, lambda b: en.items_replay(b), 'replay'); R2.to_pickle(f'{el.OUT}/scanC_e2.pkl')
lab.log(f'E2 replay done {time.time() - t0:.0f}s items', int((R2.design == 'base').sum()))

agg = R1.groupby(['design', 'nrec'])[['family', 'panel', 'iou']].apply(lab.aggregate).unstack('nrec')
D = (agg - agg.loc['base']).drop(index='base'); D.columns = [f'e1_n{c}' for c in D.columns]
D['e1_early'] = D[['e1_n1', 'e1_n2', 'e1_n3', 'e1_n4']].mean(1)
x = R2[R2.n.between(1, 4)]
a2 = x.groupby(['design', 'n'])[['family', 'panel', 'iou']].apply(lab.aggregate).unstack('n')
D2 = (a2 - a2.loc['base']).drop(index='base'); D2.columns = [f'e2_n{c}' for c in D2.columns]
D2['e2_early'] = D2.mean(1)
sat = R2[R2.d0.isin([7, 14, 21, 28])]
a3 = sat.groupby('design')[['family', 'panel', 'iou']].apply(lab.aggregate)
D3 = (a3 - a3['base']).drop(index='base').rename('e2_sat')
T = pd.concat([D, D2, D3], axis=1)
T.to_csv(f'{el.OUT}/scanC_deltas.csv')
print('\nbase: E1', round(float(agg.loc['base'].iloc[0]), 4), '| E2 by n', a2.loc['base'].round(4).to_dict(), '| E2 Saturday starts', round(float(a3['base']), 4), 'n items', int((sat.design == 'base').sum()))
print('\n== reference designs =='); print(T[~T.index.str.startswith('bma')].round(4).to_string())
Tb = T[T.index.str.startswith('bma')].copy()
parts = Tb.index.to_series().str.split('|', expand=True).iloc[:, 1:]; parts.columns = ['box', 'pi', 'mu', 'nu']
Tb = pd.concat([parts, Tb], axis=1)
print('\n== marginal means (bma) ==')
for c in ('box', 'pi', 'mu', 'nu'):
    print(Tb.groupby(c)[['e1_n1', 'e1_n2', 'e1_n3', 'e1_n4', 'e1_early', 'e1_n99', 'e2_n1', 'e2_n2', 'e2_n3', 'e2_n4', 'e2_early', 'e2_sat']].mean().round(4).to_string()); print()
for v in ('e1_early', 'e2_early', 'e2_sat', 'e1_n99'):
    print(f'== {v}: rows box|pi, columns mu|nu =='); print(Tb.pivot_table(index=['box', 'pi'], columns=['mu', 'nu'], values=v).round(4).to_string()); print()
print('== top 15 by e1_early =='); print(Tb.sort_values('e1_early', ascending=False).head(15).round(4).to_string())
print('\n== top 15 by e2_sat =='); print(Tb.sort_values('e2_sat', ascending=False).head(15).round(4).to_string())
print('\ncorrelation between criteria (bma designs):'); print(Tb[['e1_early', 'e2_early', 'e2_sat', 'e1_n99']].corr().round(2).to_string())
lab.log('DONE')
