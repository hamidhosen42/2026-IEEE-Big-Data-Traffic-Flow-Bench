"""early track (light): what do the designs do on the 40 OFFICIAL validation onset windows (raw histories, true calendar,
evidence = masked March days strictly before the window's day)?  Local replica of the production CatBoost (lab all-train model).
(1) how many windows / cells change vs base and vs the production sp4;
(2) PROXY check: the official horizons T+5..T+30 are hidden in the masked layer, but slots T+35.. are visible again.  Proxy
    truth for step 6 = visible queued cells at T+35 (first slot after the horizon; a queue born at T+30 is almost always still
    there, possibly one link longer).  Proxy IoU over the cells visible at T+35.  n = 40, noisy, uses March data after T:
    analysis only (motivates a public probe)."""
import warnings
import numpy as np, pandas as pd
import lab, early_lib as el, early_engine as en

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
F = lab.F
cat = el.load_cat()
M = dict(box=(-6, 6), form='bma', mu=0.05, nu=1.0, cw=1.0)
designs = {'sp4': dict(box=(-2, 2), ref='base', k=4, cw=0.0, ncount='obs'), 'sp8': dict(box=(-2, 2), ref='base', k=8, cw=0.0, ncount='obs'),
           'mix8': dict(M, pi=0.8), 'mix9': dict(M, pi=0.9), 'mix95': dict(M, pi=0.95), 'mix9w4': dict(M, pi=0.9, box=(-4, 4))}
rows = []
for panel in lab.PANELS:
    P, Pv = el.val_panel(panel)
    prior_tr = lab.prior_table(P, np.ones(len(P['days']), bool))
    R = lab.split_rows(P, 'validation', prior_tr)
    X, _, _, meta, cols = lab.to_matrix(R, lab.ONSET, None, None)
    assert list(cols) == el.COLS
    wi = R['wi']; on = np.where(R['cond'] == lab.ONSET)[0]; L = R['L']
    dpos = {d: i for i, d in enumerate(Pv['days'])}
    d = np.array([dpos[x] for x in wi.date.to_numpy()[on]]); t = R['t'][on]
    assert np.array_equal(meta[:, 0], np.repeat(on, L))
    b = dict(panel=panel, family=panel.rsplit('_', 1)[0], L=L, D=len(Pv['days']), W=len(on), wk=Pv['dow'] >= 5, Qv=Pv['Q'], Ev=Pv['E'],
             prior_tr=prior_tr, X=X.reshape(len(on), L, -1), d=d, t=t)
    items, _ = en.items_recent(b, 99)
    widx = np.arange(len(on)); base3 = b['X'][..., el.JP]
    pred = {}
    for name, dz in dict(base=None, **designs).items():
        if dz is None:
            p3 = None
        else:
            dz = dict(dz); box = dz.pop('box')
            p3 = el.design_priors(base3, el.item_stats(b, items, box), box, **dz)
        p = el.predict_items(cat, b, widx, p3)
        pr = np.zeros((len(on), L), bool)
        for i in widx:
            P6 = np.zeros((F, L), np.float32); P6[F - 1] = p[i]
            pr[i] = lab.decode(P6.ravel()).reshape(F, L)[F - 1]
        pred[name] = pr
    for i in widx:
        q7 = Pv['Q'][d[i], t[i] + 7]; v7 = Pv['E'][d[i], t[i] + 7]
        r = dict(panel=panel, family=b['family'], d=int(d[i]), t=int(t[i]), cond=lab.ONSET, dow=int(Pv['dow'][d[i]]), vis7=float(v7.mean()), q7=int(q7.sum()))
        for name, pr in pred.items():
            u = ((pr[i] | q7) & v7).sum()
            r[f'iou_{name}'] = 1.0 if u == 0 else float((pr[i] & q7 & v7).sum() / u)
            r[f'n_{name}'] = int(pr[i].sum())
            r[f'chg_{name}'] = int((pr[i] != pred['base'][i]).sum()); r[f'chg4_{name}'] = int((pr[i] != pred['sp4'][i]).sum())
        rows.append(r)
    lab.drop_panel(panel)
T = pd.DataFrame(rows); T.to_pickle(f'{el.OUT}/official_check.pkl')
names = ['base'] + list(designs)
print('official validation onset windows:', len(T), '| by day index', T.groupby('d').size().to_dict(), '| mean visible share at T+35', round(T.vis7.mean(), 3),
      '| windows with a visible queued cell at T+35:', int((T.q7 > 0).sum()))
print('\n== changes of the step-6 prediction (cells / windows) ==')
for n in names[1:]:
    print(f'{n:7s} vs base: {int(T[f"chg_{n}"].sum()):3d} cells in {int((T[f"chg_{n}"] > 0).sum()):2d} windows | vs sp4: {int(T[f"chg4_{n}"].sum()):3d} cells in {int((T[f"chg4_{n}"] > 0).sum()):2d} windows'
          f' | predicted cells {int(T[f"n_{n}"].sum())} (base {int(T.n_base.sum())})')
print('\n== PROXY IoU (visible queued cells at T+35), aggregated window -> panel -> family -> mean ==')
for n in names:
    A = T.rename(columns={'iou_base': 'iou'}); Bf = T.rename(columns={f'iou_{n}': 'iou'}) if n != 'base' else A
    s = f'{n:7s} {lab.aggregate(Bf):.4f}'
    if n != 'base':
        s += '  vs base ' + el.fmt(el.boot(A[['panel', 'family', 'd', 't', 'cond', 'iou']], Bf[['panel', 'family', 'd', 't', 'cond', 'iou']]))
        A4 = T.rename(columns={'iou_sp4': 'iou'})
        if n != 'sp4':
            s += '  | vs sp4 ' + el.fmt(el.boot(A4[['panel', 'family', 'd', 't', 'cond', 'iou']], Bf[['panel', 'family', 'd', 't', 'cond', 'iou']]))
    print(s)
print('\n== by day index (plain window means of the proxy IoU) ==')
print(T.groupby('d')[[f'iou_{n}' for n in names]].mean().round(3).assign(n=T.groupby('d').size()).to_string())
print('\n== per panel =='); print(T.groupby('panel')[[f'iou_{n}' for n in names]].mean().round(3).to_string())
ch = T[(T.chg4_mix9 > 0) | (T.chg_sp4 > 0) | (T.chg4_mix8 > 0)]
print('\n== windows where sp4 / mix9 / mix8 differ =='); print(ch[['panel', 'd', 't', 'dow', 'q7', 'vis7'] + [f'iou_{n}' for n in names] + [f'n_{n}' for n in names]].round(2).to_string(index=False))
