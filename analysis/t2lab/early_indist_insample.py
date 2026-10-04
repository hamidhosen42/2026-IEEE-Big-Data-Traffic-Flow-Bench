"""early track (light, NO training): in-distribution sensitivity with the EXISTING all-train onset CatBoost.
NOT fold-safe: the model was trained on these train days (windows of fold 0 included), so the base IoU is in-sample and the
measured cost of a prior perturbation is only INDICATIVE (memorised rows may react less to the prior columns).  It complements
(a) the fold-safe model-free prior-skill cost (early_modelfree.py) and (b) the fold-safe model-based test (early_indist.py),
which needs the fold-safe CatBoost from the semaphore queue.
Windows: official-like onset windows of train fold 0 (RAW histories), all-train prior as base and as reference table;
evidence = the previous train days with 34% of the observed cells hidden."""
import pickle
import numpy as np, pandas as pd
import lab, early_lib as el, early_engine as en

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
cat = el.load_cat()                                                      # all-train model (results/splitprior/cat_onset_noramp.cbm)
B = []
for panel in lab.PANELS:
    b = dict(el.train_bundle(panel))
    P = lab.load_panel(panel)
    rc = lab.rows(panel, 'offlike', lab.ONSET)
    widx = np.where(rc.win.fold.to_numpy() == 0)[0]; me = np.isin(rc.wi, widx)
    b['X'] = rc.matrix(me, el.COLS, 4).reshape(b['W'], b['L'], -1)       # k = 4: all-train prior in the prior columns
    b['prior_tr'] = lab.prior_table(P, np.ones(len(P['days']), bool))
    B.append(b); lab.drop_panel(panel)
M = dict(box=(-6, 6), form='bma', mu=0.05, nu=1.0, cw=1.0)
designs = {'sp4': dict(box=(-2, 2), ref='base', k=4, cw=0.0, ncount='obs'), 'sp8': dict(box=(-2, 2), ref='base', k=8, cw=0.0, ncount='obs'),
           'blend_w24': dict(box=(-24, 24), ref='base', k=4, cw=1.0, ncount='obs'),
           'mix8': dict(M, pi=0.8), 'mix9': dict(M, pi=0.9), 'mix95': dict(M, pi=0.95), 'mix98': dict(M, pi=0.98)}
ds = list(designs)
print('######## IN-SAMPLE all-train model on train fold-0 official-like onset windows (RAW histories) - indicative only')
for n in (1, 2, 3, 4, 99):
    R = en.run(cat, B, designs, lambda b: en.items_recent(b, n), f'recent{n}')
    print(f'\n-- E1 n = {n if n < 99 else "all earlier fold-0 days"}')
    print(en.table(R, ref='base', ref2='sp4', designs=['base'] + ds).round(4).to_string(index=False))
R2 = en.run(cat, B, designs, lambda b: en.items_replay(b, hide=18), 'replay18')
dow0 = {b['panel']: b['dow'] for b in B}
sat = R2[[dow0[p][d0] == 5 for p, d0 in zip(R2.panel, R2.d0)]]
print('\n-- E2 faithful replay, Saturday starts (public calendar); items by n:', sat[sat.design == 'base'].groupby('n').size().to_dict())
print(en.table(sat, ref='base', ref2='sp4', designs=['base'] + ds).round(4).to_string(index=False))
