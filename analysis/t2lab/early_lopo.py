"""early track (light): leave-one-panel-out design selection on the stage C grid (108 mixture designs + 4 references).
For each held-out panel the design with the best mean early delta (E1, n = 1..4) on the other 7 panels is selected and scored
on the held-out panel -> honest estimate of the gain of 'the selected design' (selection optimism removed)."""
import numpy as np, pandas as pd
import lab, early_lib as el
pd.set_option('display.width', 250)
R1 = pd.read_pickle(f'{el.OUT}/scanC_e1.pkl'); R2 = pd.read_pickle(f'{el.OUT}/scanC_e2.pkl')
pm = R1[R1.nrec < 99].groupby(['design', 'nrec', 'panel']).iou.mean().unstack('panel')          # [design x nrec] x panel
dl = pm - pm.loc['base'].reindex(pm.index.get_level_values('nrec')).to_numpy()
early = dl.groupby(level='design').mean()                                                       # design x panel: mean over n = 1..4
early = early.drop(index='base')
bma = early[early.index.str.startswith('bma')]
print('in-sample best design (all 8 panels):', bma.mean(1).idxmax(), round(float(bma.mean(1).max()), 4))
rows = []
for p in early.columns:
    others = [c for c in early.columns if c != p]
    best = bma[others].mean(1).idxmax()
    r = dict(held_out=p, selected=best, heldout_delta_early=float(early.loc[best, p]), sp4=float(early.loc['sp4', p]))
    for n in (1, 2, 3, 4):
        r[f'n{n}'] = float(dl.loc[(best, n), p])
    rows.append(r)
L = pd.DataFrame(rows); print(L.round(4).to_string(index=False))
print('LOPO-selected mixture: early delta', round(L.heldout_delta_early.mean(), 4), '| by n', {n: round(L[f"n{n}"].mean(), 4) for n in (1, 2, 3, 4)},
      '| sp4', round(L.sp4.mean(), 4), '| panels positive', int((L.heldout_delta_early > 1e-9).sum()), 'negative', int((L.heldout_delta_early < -1e-9).sum()))
# distribution of the whole mixture grid (how special is the pick?)
print('\nmixture grid (108 designs): early delta quantiles', bma.mean(1).quantile([0, .1, .25, .5, .75, .9, 1]).round(4).to_dict())
print('share of mixture designs beating sp4 on early:', round(float((bma.mean(1) > early.loc['sp4'].mean()).mean()), 3),
      '| beating the best blend A_w24p_k4:', round(float((bma.mean(1) > early.loc['A_w24p_k4'].mean()).mean()), 3))
# same on the Saturday-start replay (criterion = Saturday aggregate), selection still by E1-early on the other panels
sat = R2[R2.d0.isin([7, 14, 21, 28])]
ps = sat.groupby(['design', 'panel']).iou.mean().unstack('panel'); ds = ps - ps.loc['base']
print('\nSaturday-start replay delta of the LOPO-selected designs (selection by E1 on the other 7 panels):',
      round(float(np.mean([ds.loc[r.selected, r.held_out] for r in L.itertuples()])), 4), '| sp4', round(float(ds.loc['sp4'].mean()), 4))
