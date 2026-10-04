"""TRACK ogmodel: add the pseudo-March set `midrun3_m` to the cache = faithful ongoing candidates of the masked validation month
on the t % 3 grid that are NOT run-starts (mid-run windows; the March counterpart of the F03 weight-0.3 training windows).
Same construction / caveats as ogmodel_pmarch_rows.py (masked histories, visible labels, blank-touching windows dropped)."""
import gc, time
import numpy as np
import pandas as pd
import lab
import ogmodel_lib as og

H, F = lab.H, lab.F
info = []
for panel in lab.PANELS:
    t1 = time.time()
    P, Pv = og.val_panel(panel)
    on, ogc, stable, pers, blank = og.select(Pv)
    dd, tt = np.where(ogc); k = ogc[dd, tt - 1] & (tt % 3 == 0)
    wl = sorted(zip(dd[k].tolist(), tt[k].tolist()))
    if not wl:
        continue
    vis_day = Pv['E'].sum((1, 2)) / np.maximum(Pv['Em'].sum((1, 2)), 1)
    prior = lab.prior_table(P, np.ones(len(P['days']), bool))
    R = og.val_rows(P, Pv, [(a, b, lab.ONGOING) for a, b in wl], prior, use_T=True)
    X, y, e, meta, call = lab.to_matrix(R, lab.ONGOING, None, lab.ongoing_keep(R))
    d_, t_ = R['d'], R['t']
    np.savez_compressed(og.pm_file(panel, 'midrun3_m', 'T'), X=X, y=y.astype(bool), e=e.astype(bool), meta=meta.astype(np.int32), cols=np.array(call),
                        truth=np.stack([Pv['Q'][d_, t_ + h] for h in range(1, F + 1)], 1), ev=np.stack([Pv['E'][d_, t_ + h] for h in range(1, F + 1)], 1),
                        qlast=R['qlast'], d=d_, t=t_, L=R['L'], pers_fill=pers[d_, t_], runstart=np.zeros(len(d_), bool), vis=vis_day[d_],
                        hist_vis=np.array([float(Pv['E'][a, b - H:b].sum() / max(Pv['Em'][a, b - H:b].sum(), 1)) for a, b in zip(d_, t_)]))
    info.append(dict(panel=panel, windows=len(wl), rows=len(X)))
    lab.drop_panel(panel); del X; gc.collect()
print(pd.DataFrame(info).to_string(index=False)); print('total windows', sum(i['windows'] for i in info))
