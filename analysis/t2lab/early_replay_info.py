"""early track (light): what does the month-start replay look like?  picks per start day, n distribution, weekday mix."""
import numpy as np, pandas as pd
import lab, early_lib as el, early_engine as en
pd.set_option('display.width', 250)
rows = []
for p in lab.PANELS:
    b = el.march_bundle(p)
    items, meta = en.items_replay(b)
    meta['panel'] = p; meta['d'] = [int(b['d'][i]) for i, _, _ in items]; meta['t'] = [int(b['t'][i]) for i, _, _ in items]
    meta['dow0'] = b['dow'][meta.d0.to_numpy()]; meta['dow'] = b['dow'][meta.d.to_numpy()]
    rows.append(meta)
    if p == 'D7_I10_W':
        for d0 in (0, 7, 14, 21, 28):
            on, og = b['picks'][d0]
            print(p, 'start', d0, 'onset picks', on, 'ongoing picks', og)
M = pd.concat(rows, ignore_index=True)
print('all starts: items', len(M), 'unique windows', len(M.drop_duplicates(['panel', 'd', 't'])))
print(M.groupby('n').size().to_string())
S = M[M.dow0 == 5]
print('\nSaturday starts:', sorted(S.d0.unique()), 'items', len(S))
print(pd.crosstab(S.d0, S.n).to_string())
print(pd.crosstab(S.panel, S.n).to_string())
print('\nofficial validation onset windows: n = day index: 8 x n0, 8 x n1, 9 x n2, 8 x n3, 6 x n4, 1 x n6')
