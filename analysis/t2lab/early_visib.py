"""early track (light): direction of the masked-vs-raw history confound.
Official windows have RAW histories (76% of cells), pseudo-March windows masked ones (about 50%).  We cannot un-mask March, but we
can mask MORE: the same 214 windows with a further 34% of the visible history cells hidden (about 33% visible, slot-T value too).
If the gain of a prior design grows when the history gets poorer, the gain at raw visibility is smaller than measured at 50%;
if it is flat, the confound matters less.  Evidence layer and labels are unchanged."""
import numpy as np, pandas as pd
import lab, early_lib as el, early_engine as en

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
cat = el.load_cat()
M = dict(box=(-6, 6), form='bma', mu=0.05, nu=1.0, cw=1.0)
designs = {'sp4': dict(box=(-2, 2), ref='base', k=4, cw=0.0, ncount='obs'), 'mix9': dict(M, pi=0.9), 'mix95': dict(M, pi=0.95), 'mix8': dict(M, pi=0.8)}
B50 = [el.march_bundle(p) for p in lab.PANELS]; B33 = []
for b in B50:
    P, Pv = el.val_panel(b['panel'])
    rng = np.random.default_rng(100 + lab.PANELS.index(b['panel']))
    hide = rng.random(Pv['V'].shape) < 0.34
    Pv2 = dict(Pv, V=np.where(hide, np.nan, Pv['V']), Fq=np.where(hide, np.nan, Pv['Fq']))
    R = el.onset_rows(P, Pv2, [(int(a), int(c), lab.ONSET) for a, c in zip(b['d'], b['t'])], b['prior_tr'])
    X, _, _, meta, cols = lab.to_matrix(R, lab.ONSET, None, None)
    b2 = dict(b, X=X.reshape(b['W'], b['L'], -1)); B33.append(b2)
    lab.drop_panel(b['panel'])
vis = lambda Bs: float(np.mean([np.isfinite(b['X'][..., el.COLS.index('vr_raw_last')]).mean() for b in Bs]))
print('share of links with a visible last history slot: masked layer', round(vis(B50), 3), '| extra-masked', round(vis(B33), 3))
for tag, Bs in (('history ~50% visible (pseudo-March as is)', B50), ('history ~33% visible (extra 34% hidden)', B33)):
    print(f'\n######## {tag}')
    for n in (2, 4, 99):
        R = en.run(cat, Bs, designs, lambda b: en.items_recent(b, n), f'recent{n}')
        print(f'-- E1 n = {n if n < 99 else "all"}'); print(en.table(R, ref='base', ref2='sp4', designs=['base'] + list(designs)).round(4).to_string(index=False))
    R2 = en.run(cat, Bs, designs, lambda b: en.items_replay(b, starts=[7, 14, 21, 28], hide=18), 'sat18')
    print('-- E2 faithful Saturday replay'); print(en.table(R2, ref='base', ref2='sp4', designs=['base'] + list(designs)).round(4).to_string(index=False))
