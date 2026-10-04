"""TRACK ogmodel: cache the pseudo-March ongoing rows once (window selection + features copied from exp_firstcross_pmarch.py).
Sets: offlike_m (faithful greedy multi-start on the masked validation month) and runstart_m (faithful run-starts); windows
whose [T-60, T+30] touches a blanked slot (official horizons + buffer) are dropped by `select`.
Two feature versions: 'T' (masked slot-T value as production) and 'noT' (no slot-T value: S5 contingency).
Labels = visible masked-layer cells of T+5..T+30.  Used for ANALYSIS of train-only models (public-facing / private-facing
regime with MASKED histories, visible-only labels); no model is trained on these labels here.
Usage (through run_heavy): ogmodel_pmarch_rows.py"""
import gc, json, time
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
    D = len(Pv['days'])
    g_on, g_og = lab._glist(on), lab._glist(ogc); s = set()
    for d0 in range(D):
        _, b = lab.greedy(g_on, g_og, 'interleaved', start_day=d0); s.update(b)
    dd, tt = np.where(ogc); rs = ~ogc[dd, tt - 1]
    sets = {'offlike_m': sorted(s), 'runstart_m': sorted(zip(dd[rs].tolist(), tt[rs].tolist()))}
    vis_day = Pv['E'].sum((1, 2)) / np.maximum(Pv['Em'].sum((1, 2)), 1)
    prior = lab.prior_table(P, np.ones(len(P['days']), bool))
    for sname, wl in sets.items():
        if not wl:
            continue
        for vt in ('T', 'noT'):
            R = og.val_rows(P, Pv, [(a, b, lab.ONGOING) for a, b in wl], prior, use_T=(vt == 'T'))
            X, y, e, meta, call = lab.to_matrix(R, lab.ONGOING, None, lab.ongoing_keep(R))
            d_, t_ = R['d'], R['t']
            np.savez_compressed(og.pm_file(panel, sname, vt), X=X, y=y.astype(bool), e=e.astype(bool), meta=meta.astype(np.int32), cols=np.array(call),
                                truth=np.stack([Pv['Q'][d_, t_ + h] for h in range(1, F + 1)], 1), ev=np.stack([Pv['E'][d_, t_ + h] for h in range(1, F + 1)], 1),
                                qlast=R['qlast'], d=d_, t=t_, L=R['L'], pers_fill=pers[d_, t_], runstart=ogc[d_, t_] & ~ogc[d_, t_ - 1], vis=vis_day[d_],
                                hist_vis=np.array([float(Pv['E'][a, b - H:b].sum() / max(Pv['Em'][a, b - H:b].sum(), 1)) for a, b in zip(d_, t_)]),
                                hist_cov=np.array([float(Pv['E'][a, b - H:b].mean()) for a, b in zip(d_, t_)]),
                                has_T=np.isfinite(Pv['V'][d_, t_]).mean(1))
            info.append(dict(panel=panel, set=sname, vt=vt, windows=len(wl), rows=len(X), pos=float(y[e].mean()), label_vis=float(e.mean()),
                             hist_cov=float(np.mean([Pv['E'][a, b - H:b].mean() for a, b in zip(d_, t_)])), q_T=float(R['qlast'].sum(1).mean())))
            del X; gc.collect()
    lab.log(panel, {k: len(v) for k, v in sets.items()}, f'{time.time() - t1:.0f}s')
    lab.drop_panel(panel)
I = pd.DataFrame(info); I.to_csv(f'{og.RES}/pmarch_rows_info.csv', index=False)
pd.set_option('display.width', 250)
print(I.round(4).to_string(index=False))
lab.log('DONE')
