"""early track, stage A grid on E1 (214 pseudo-March onset windows, evidence = the n most recent scenario days, n = 1..4, all):
  ref    base (classic blend towards the box share) | box (residual: deviation of the box share from the train prior of the box)
  box    +-2, +-6, +-12, +-24, whole day, forward-skewed -2..+6 and -2..+12 (slots around the target slot)
  cw     weight of other-weekday-class evidence days in the class-specific columns: 0 (separate) / 0.5 / 1 (pooled)
  k      0.5, 1, 2, 4, 8  (a = n / (n + k))
  ncount obs (n = visible observations / box length, production sp4) | days (n = days with a visible cell in the box)
All three prior columns substituted, no spatial smoothing, both directions.  Saves the per-window frame."""
import sys, time
import numpy as np, pandas as pd
import lab, early_lib as el, early_engine as en

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 500)
cat = el.load_cat()
B = [el.march_bundle(p) for p in lab.PANELS]
BOXES = {'w2': (-2, 2), 'w6': (-6, 6), 'w12': (-12, 12), 'w24': (-24, 24), 'day': (-288, 288), 'f6': (-2, 6), 'f12': (-2, 12)}
designs = {}
for ref in ('base', 'box'):
    for bn, box in BOXES.items():
        for cw in (0.0, 0.5, 1.0):
            for k in (0.5, 1, 2, 4, 8):
                for nc in ('obs', 'days'):
                    designs[f'{ref}|{bn}|cw{cw}|k{k}|{nc}'] = dict(box=box, ref=ref, k=k, cw=cw, ncount=nc, cw_all=1.0)
print(len(designs), 'designs')
allR = []
for n in (1, 2, 3, 4, 99):
    t0 = time.time()
    R = en.run(cat, B, designs, lambda b: en.items_recent(b, n), f'recent{n}')
    R['nrec'] = n; allR.append(R)
    lab.log(f'n={n} done {time.time() - t0:.0f}s')
R = pd.concat(allR, ignore_index=True)
R.to_pickle(f'{el.OUT}/scanA.pkl')

# aggregated IoU per design and n
agg = R.groupby(['design', 'nrec']).apply(lab.aggregate).unstack('nrec')
base = agg.loc['base']
D = (agg - base).drop(index='base')
D['early'] = D[[1, 2, 3, 4]].mean(1)
parts = D.index.to_series().str.split('|', expand=True); parts.columns = ['ref', 'box', 'cw', 'k', 'nc']
D = pd.concat([parts, D], axis=1)
D.to_csv(f'{el.OUT}/scanA_deltas.csv')
print('\nbase IoU', round(float(base[1]), 4))
print('\n== marginal means of the early delta (mean over n = 1..4 of IoU - base) ==')
for c in ('ref', 'box', 'cw', 'k', 'nc'):
    print(D.groupby(c)[[1, 2, 3, 4, 'early', 99]].mean().round(4).to_string()); print()
print('== ref x box (mean over cw, k, nc) =='); print(D.pivot_table(index='box', columns='ref', values='early').round(4).to_string())
print('\n== ref x cw =='); print(D.pivot_table(index='cw', columns='ref', values='early').round(4).to_string())
for ref in ('base', 'box'):
    for nc in ('obs', 'days'):
        x = D[(D.ref == ref) & (D.nc == nc)]
        print(f'\n== early delta, ref={ref}, ncount={nc}: rows box|cw, columns k ==')
        print(x.pivot_table(index=['box', 'cw'], columns='k', values='early').round(4).to_string())
        print(f'-- all days (n=99), ref={ref}, ncount={nc}')
        print(x.pivot_table(index=['box', 'cw'], columns='k', values=99).round(4).to_string())
print('\n== top 25 by early delta =='); print(D.sort_values('early', ascending=False).head(25).round(4).to_string())
print('\n== top 15 by all-days delta =='); print(D.sort_values(99, ascending=False).head(15).round(4).to_string())
lab.log('DONE')
