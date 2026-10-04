"""early track (light, NO model): how much do the designs move the `prior` column, and how good is the column alone at locating
the T+30 queue (size-oracle top-m IoU over the visible links, as exp_onset_insplit_signal.py)?
  march   pseudo-March onset windows (new scenario)            train0  held-out train fold 0 (fold-safe prior; previous train
  days with 34% of the cells hidden as evidence) = IN-DISTRIBUTION: the train prior is already right, any change is noise."""
import sys
import numpy as np, pandas as pd
import lab, early_lib as el, early_engine as en
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
M = dict(box=(-6, 6), form='bma', mu=0.05, nu=1.0, cw=1.0)
designs = {'sp4': dict(box=(-2, 2), ref='base', k=4, cw=0.0, ncount='obs'), 'sp8': dict(box=(-2, 2), ref='base', k=8, cw=0.0, ncount='obs'),
           'blend_w24': dict(box=(-24, 24), ref='base', k=4, cw=1.0, ncount='obs'),
           'mix8': dict(M, pi=0.8), 'mix9': dict(M, pi=0.9), 'mix95': dict(M, pi=0.95), 'mix98': dict(M, pi=0.98)}


def topm(score, y, vis):
    s = score[vis]; yy = y[vis]; m = int(yy.sum())
    top = np.argsort(-s, kind='stable')[:m]; pred = np.zeros(len(s), bool); pred[top] = True
    return float((pred & yy).sum() / max((pred | yy).sum(), 1))


for kind, maker in (('march', el.march_bundle), ('train0', el.train_bundle)):
    out = []
    for panel in lab.PANELS:
        b = maker(panel)
        for n in (1, 2, 3, 4, 99):
            items, _ = en.items_recent(b, n)
            base3 = b['X'][..., el.JP]; cache = {}
            cols = {'base': base3}
            for name, dz in designs.items():
                dz = dict(dz); box = dz.pop('box')
                if box not in cache:
                    cache[box] = el.item_stats(b, items, box)
                cols[name] = el.design_priors(base3, cache[box], box, **dz)
            for i in range(b['W']):
                y = b['truth'][i][5]; vis = b['ev'][i][5]
                if not (y & vis).any():
                    continue
                r = dict(panel=panel, family=b['family'], d=int(b['d'][i]), t=int(b['t'][i]), cond=lab.ONSET, nrec=n)
                for name, c in cols.items():
                    r[f'iou_{name}'] = topm(c[i, :, 0], y, vis)
                    r[f'mv_{name}'] = float((np.abs(c[i, :, 0] - base3[i, :, 0]) > 0.05).mean())
                out.append(r)
        if kind == 'train0':
            lab.drop_panel(panel)
    T = pd.DataFrame(out); T.to_pickle(f'{el.OUT}/modelfree_{kind}.pkl')
    print(f'\n######## {kind}: size-oracle top-m IoU of the `prior` column alone (aggregated window -> panel -> family -> mean) and share of cells moved by > 0.05')
    for n in (1, 2, 3, 4, 99):
        x = T[T.nrec == n]; A = x.rename(columns={'iou_base': 'iou'})
        print(f'-- n = {n if n < 99 else "all"} (windows {len(x)})  base {lab.aggregate(A):.4f}')
        for name in designs:
            Bf = x.rename(columns={f'iou_{name}': 'iou'})
            bt = el.boot(A[['panel', 'family', 'd', 't', 'cond', 'iou']], Bf[['panel', 'family', 'd', 't', 'cond', 'iou']])
            print(f'   {name:10s} {lab.aggregate(Bf):.4f}  {el.fmt(bt)}  | cells moved {x[f"mv_{name}"].mean():.4f}')
