"""early track (light): does a TIME-FILLED evidence layer help?  Each evidence day (strictly before the window's day) is filled
per link by linear interpolation of the visible speeds over gaps <= maxgap slots (fill_time36, the lab's selection fill).  This
(a) roughly doubles the usable cells of the masked layer and (b) bridges the 90-minute blocks hidden after official windows
(free before the block, queued after it -> queued in the second half of the block).  Legal: only days before the window's day."""
import sys, time
import numpy as np, pandas as pd
import lab, early_lib as el, early_engine as en
from exp_firstcross_prod_snippet import fill_time36

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
cat = el.load_cat()
B = [el.march_bundle(p) for p in lab.PANELS]
VV = {}
for b in B:
    P, Pv = el.val_panel(b['panel']); VV[b['panel']] = (Pv['V'], P['cut']); lab.drop_panel(b['panel'])


def filled(maxgap):
    def make(b, blank=None):
        V, cut = VV[b['panel']]; V2 = V.copy()
        if blank is not None and blank.any():
            V2[blank] = np.nan
        VF = fill_time36(V2, maxgap); Ev = np.isfinite(VF)
        return el.Evidence(dict(b, Ev=Ev, Qv=Ev & (VF <= cut[None, None, :])), None)
    return make


M = dict(box=(-6, 6), form='bma', mu=0.05, nu=1.0, cw=1.0)
designs = {'sp4': dict(box=(-2, 2), ref='base', k=4, cw=0.0, ncount='obs'), 'mix9': dict(M, pi=0.9), 'mix95': dict(M, pi=0.95), 'mix8': dict(M, pi=0.8),
           'mix9w4': dict(M, pi=0.9, box=(-4, 4)), 'mix9w12': dict(M, pi=0.9, box=(-12, 12))}
ds = list(designs); out = {}
for tag, mk in (('raw masked layer', None), ('filled, gaps <= 36', filled(36)), ('filled, gaps <= 18', filled(18))):
    print(f'\n################ evidence layer: {tag}')
    for n in (1, 2, 4, 99):
        evs = {b['panel']: (mk or el.Evidence)(b, None) for b in B}
        R = en.run(cat, B, designs, lambda b: en.items_recent(b, n, evs[b['panel']]), f'recent{n}')
        print(f'-- E1 n = {n if n < 99 else "all"}'); print(en.table(R, ref='base', ref2='sp4', designs=['base'] + ds).round(4).to_string(index=False))
    R2 = en.run(cat, B, designs, lambda b: en.items_replay(b, starts=[7, 14, 21, 28], hide=18, make=mk), 'sat18')
    out[tag] = R2
    print('-- E2 faithful Saturday replay'); print(en.table(R2, ref='base', ref2='sp4', designs=['base'] + ds).round(4).to_string(index=False))
    print(en.panel_table(R2, ['base'] + ds).round(3).to_string())
A = out['raw masked layer']; Bf = out['filled, gaps <= 36']
for nm in ('mix9', 'mix95'):
    print(f'\nSaturday replay, {nm}: filled(36) vs raw evidence layer:', el.fmt(el.boot(A[A.design == nm], Bf[Bf.design == nm])))
pd.to_pickle(out, f'{el.OUT}/fill_sat.pkl')
