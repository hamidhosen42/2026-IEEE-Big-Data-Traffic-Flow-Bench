"""early track (light): detail of the official validation onset windows whose step-6 prediction changes (base / sp4 / mix9)."""
import numpy as np, pandas as pd
import lab, early_lib as el, early_engine as en
F = lab.F
cat = el.load_cat()
M = dict(box=(-6, 6), form='bma', mu=0.05, nu=1.0, cw=1.0)
designs = {'sp4': dict(box=(-2, 2), ref='base', k=4, cw=0.0, ncount='obs'), 'mix9': dict(M, pi=0.9), 'mix8': dict(M, pi=0.8), 'mix95': dict(M, pi=0.95)}
tot = {k: [0, 0] for k in designs}
for panel in lab.PANELS:
    P, Pv = el.val_panel(panel)
    prior_tr = lab.prior_table(P, np.ones(len(P['days']), bool))
    R = lab.split_rows(P, 'validation', prior_tr)
    X, _, _, meta, cols = lab.to_matrix(R, lab.ONSET, None, None)
    wi = R['wi']; on = np.where(R['cond'] == lab.ONSET)[0]; L = R['L']
    dpos = {d: i for i, d in enumerate(Pv['days'])}
    d = np.array([dpos[x] for x in wi.date.to_numpy()[on]]); t = R['t'][on]
    b = dict(panel=panel, family=panel.rsplit('_', 1)[0], L=L, D=len(Pv['days']), W=len(on), wk=Pv['dow'] >= 5, Qv=Pv['Q'], Ev=Pv['E'],
             prior_tr=prior_tr, X=X.reshape(len(on), L, -1), d=d, t=t)
    evd = el.Evidence(b)
    items, _ = en.items_recent(b, 99, evd)
    widx = np.arange(len(on)); base3 = b['X'][..., el.JP]
    pri = {'base': base3}; prob = {}; pred = {}
    for name, dz in dict(base=None, **designs).items():
        if dz is not None:
            dz = dict(dz); box = dz.pop('box'); pri[name] = el.design_priors(base3, el.item_stats(b, items, box), box, **dz)
        prob[name] = el.predict_items(cat, b, widx, None if dz is None else pri[name])
        pr = np.zeros((len(on), L), bool)
        for i in widx:
            P6 = np.zeros((F, L), np.float32); P6[F - 1] = prob[name][i]
            pr[i] = lab.decode(P6.ravel()).reshape(F, L)[F - 1]
        pred[name] = pr
    for i in widx:
        if not any((pred[n][i] != pred['base'][i]).any() for n in designs):
            continue
        s6 = t[i] + 6
        print(f'\n{panel} day {d[i] + 1} (dow {Pv["dow"][d[i]]}) T={t[i] // 12:02d}:{t[i] % 12 * 5:02d}  evidence days {d[i]}')
        links = sorted(set(np.where(np.logical_or.reduce([pred[n][i] for n in pred]))[0].tolist()))
        for n in pred:
            print(f'   {n:6s} pred {np.where(pred[n][i])[0].tolist()}')
        print('   links          ', links)
        for n in pri:
            print(f'   prior {n:6s}   ', np.round(pri[n][i, links, 0], 2).tolist())
        for n in prob:
            print(f'   prob  {n:6s}   ', np.round(prob[n][i, links], 2).tolist())
        for dd in range(d[i]):
            O, e, Xx = evd.box(np.array([dd]), s6, -6, 6)
            print(f'   day {dd + 1} ({"we" if b["wk"][dd] else "wd"}) queued:visible (train-expected) in +-6 box:',
                  ' '.join(f'{int(O[0, l])}:{int(e[0, l])}({Xx[0, l]:.1f})' for l in links))
    lab.drop_panel(panel)
